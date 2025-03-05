import os
import json
import logging
import shutil
import uuid
import traceback
from typing import List, Dict, Optional

import ollama  # Official Ollama library
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.llms.base import LLM

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt

import requests

# For connecting to a remote ChromaDB instance:
from chromadb.config import Settings

# ------------------------------------------------------------------------------
# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Load Environment Variables
load_dotenv()
PDF_STORAGE_DIR = os.getenv("UPLOADED_PDFS", "uploaded_pdfs")
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# Model & service configuration from environment variables:
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "mxbai-embed-large")
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = os.getenv("CHROMADB_PORT", "8000")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

# Load ChromaDB persistence directory from environment variable
CHROMADB_PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIR", "./chroma_db")
os.makedirs(CHROMADB_PERSIST_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Authentication Configuration

SECRET_KEY = "your_secret_key_here"  # Change this to a secure random key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Token expires in 60 minutes

# Simulated in-memory user database
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "hashed_password": "$2b$12$ZCMfg4khcBXTor88gYxOJeD92P55biVxFLWKsQGuh9EpNejTmIgZ.",  # Password: "test123"
    }
}

# Password hashing utility
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for login
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)



def get_user(username: str):
    """Fetch user from in-memory database."""
    return fake_users_db.get(username)


def authenticate_user(username: str, password: str):
    """Authenticate user and verify credentials."""
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta):
    """Generate a JWT token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Verify JWT token and extract user information."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in fake_users_db:
            raise HTTPException(status_code=401, detail="Invalid token")
        return fake_users_db[username]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate token")


# ------------------------------------------------------------------------------
# Ollama Wrappers using the official library

class OllamaLLM(LLM):
    model: str
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False  # Ensure that we get a single JSON response.
        }
        if stop:
            payload["stop"] = stop
        url = f"{OLLAMA_API_URL}/generate"
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            # Check for the key "response" (used in your examples)
            text = data.get("response") or data.get("output")
            if not text:
                raise ValueError("No generated text returned from Ollama.generate. Response was: " + str(data))
            return text
        except Exception as e:
            logger.error("Error calling Ollama.generate: %s", e)
            raise e

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._call(prompt, stop)

def process_embedding(embedding):
    """
    Process the raw embedding returned from the API.

    If the embedding is a list of one element that is itself a list,
    then flatten it by one level (to get a flat list of numbers).
    Then, convert all elements to floats.
    """
    # If the embedding is a list with a single element and that element is a list,
    # assume that the extra nesting should be removed.
    if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
        embedding = embedding[0]
    try:
        return [float(x) for x in embedding]
    except Exception as e:
        logger.error("Error converting embedding values to float: %s", e)
        raise e

class OllamaEmbeddingWrapper:
    def __init__(self, model: str):
        self.model = model

    def __call__(self, text: str) -> List[float]:
        """
        Call the Ollama embed endpoint and return a flat list of floats.
        """
        payload = {
            "model": self.model,
            "input": text
        }
        url = f"{OLLAMA_API_URL}/embed"
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            # The API may return the key "embedding" or "embeddings".
            raw_embedding = data.get("embedding") or data.get("embeddings")
            if raw_embedding is None:
                raise ValueError("No embedding returned from Ollama.embed. Response was: " + str(data))
            processed = process_embedding(raw_embedding)
            return processed
        except Exception as e:
            logger.error("Error calling Ollama.embed: %s", e)
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Given a list of texts, return a list where each element is a flat list of floats.
        """
        # Call the __call__ method for each text.
        return [self.__call__(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        For a query, return the same as __call__.
        """
        return self.__call__(text)

# Instantiate our Ollama wrappers:
ollama_llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=0.0)
embedding_model = OllamaEmbeddingWrapper(model=EMBEDDING_MODEL_NAME)

# ------------------------------------------------------------------------------
# Configure ChromaDB Client (running via Docker)
client_settings = Settings(
    chroma_server_host=CHROMADB_HOST,
    chroma_server_http_port=CHROMADB_PORT,
)
# Initialize Chroma vector store with a collection named "docs"
vectorstore = Chroma(
    collection_name="docs",
    embedding_function=embedding_model,  # your previously defined embedding wrapper
    client_settings=client_settings,
    persist_directory=CHROMADB_PERSIST_DIR
)

vectorstore.get()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 6})

# ------------------------------------------------------------------------------
# In-memory session memory for conversation context
session_memory: Dict[str, ConversationBufferMemory] = {}

# ------------------------------------------------------------------------------
# Pydantic Models for API

class GradeDocuments(BaseModel):
    """Model for a grader's binary score output."""
    binary_score: str = Field(
        description="Document relevance score: 'yes' or 'no'"
    )

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    resources: List[dict] = Field(
        description="List of resources (e.g., PDF filename and page number)."
    )

# ------------------------------------------------------------------------------
# Define the QA Prompt and Chain

qa_system_prompt = """You are an assistant for question-answering tasks.

{chat_history}

Using only the provided documents, answer the following question accurately and concisely. If the documents do not contain sufficient information to address the question, acknowledge that.

When responding:
1. Answer the question using specific information from the documents.
2. Cite relevant documents as references.
3. Do not add external information.

**Example:**

**Question:** Come devono essere valutate le attività e passività monetarie e non monetarie in valuta estera secondo gli ITAS?

**Provided Documents:**
1. Documento 1: "Gli elementi monetari in valuta estera devono essere convertiti nella valuta funzionale utilizzando il tasso di cambio di chiusura. Gli elementi non monetari valutati al costo storico devono essere convertiti utilizzando il tasso alla data dell'iscrizione iniziale."
2. Documento 2: "Gli elementi non monetari valutati a valori correnti devono essere convertiti utilizzando il tasso di cambio alla data in cui è stato determinato il valore corrente. La valuta funzionale per le amministrazioni è l'euro, salvo eccezioni previste dalla legge."

**Answer:** Gli elementi monetari in valuta estera devono essere convertiti nella valuta funzionale utilizzando il tasso di cambio di chiusura. Gli elementi non monetari valutati al costo storico usano il tasso alla data dell'iscrizione iniziale, mentre quelli valutati a valori correnti usano il tasso di cambio alla data della valutazione del valore corrente

Your answer should be clear, self-contained, concise and in italian. Don' return text with something like <doc> inside. 
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "Retrieved documents:\n\n<docs>{documents}</docs>\n\nUser question:\n<question>{question}</question>")
    ]
)

# Default conversation memory; per-session memory will be stored in session_memory.
default_memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

qa_chain = LLMChain(
    llm=ollama_llm,
    prompt=qa_prompt,
    memory=default_memory,
    output_parser=StrOutputParser()
)

# ------------------------------------------------------------------------------
# Helper Functions

def grade_document(question: str, document: str) -> str:
    """
    Uses the Ollama LLM to grade document relevance.
    The prompt instructs the model to output JSON with a binary score.
    """
    grading_prompt = f"""
You are a grader assessing whether a retrieved document is relevant to the user question.
If the document contains information or keywords related to the question, respond with "yes"; otherwise, "no".

Retrieved document:
{document}

User question:
{question}

Respond in JSON format as follows:
{{"binary_score": "<yes/no>"}}
"""
    try:
        response = ollama_llm(grading_prompt)
        result = json.loads(response)
        return result.get("binary_score", "no").lower()
    except Exception as e:
        logger.error("Error grading document: %s", e)
        return "no"

def format_docs(docs: List) -> str:
    """Format a list of documents for LLM input."""
    return "\n".join(
        f"<doc{i+1}>:\nTitle: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content}\n</doc{i+1}>"
        for i, doc in enumerate(docs)
    )

def get_resources(docs: List) -> List[dict]:
    """Extract resource information from documents."""
    return [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page_number": doc.metadata.get("page_number", "Unknown")
        }
        for doc in docs
    ]

def get_memory_for_session(session_id: str) -> ConversationBufferMemory:
    if session_id not in session_memory:
        session_memory[session_id] = ConversationBufferMemory(memory_key="chat_history", input_key="question")
    return session_memory[session_id]

def create_or_get_session_id(x_session_id: Optional[str] = Header(None)) -> str:
    session_id = x_session_id or str(uuid.uuid4())
    logger.info("Session ID used: %s", session_id)
    return session_id

# ------------------------------------------------------------------------------
# Initialize FastAPI and API Endpoints

app = FastAPI(title="Document QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.30.4:3000"],  # Add both local and Docker network IPs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/", summary="API Root")
def read_root():
    return {"message": "Welcome to the Document QA API. Visit /docs for API documentation."}

# ------------------------------------------------------------------------------
# Authentication Endpoints

class UserRegister(BaseModel):
    username: str
    password: str
    full_name: str

@app.get("/users", summary="Get list of users")
def list_users(current_user: dict = Depends(get_current_user)):
    """
    Return a list of registered users.
    Only non-sensitive information is returned.
    """
    users = [
        {"username": user["username"], "full_name": user["full_name"]}
        for user in fake_users_db.values()
    ]
    return {"users": users}


@app.post("/register", summary="Register a new user")
def register_user(user: UserRegister):
    """Register a new user (in-memory)."""
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = pwd_context.hash(user.password)
    fake_users_db[user.username] = {
        "username": user.username,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
    }
    return {"message": "User registered successfully"}


@app.post("/login", summary="Login to get an access token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and return JWT access token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}


from fastapi.staticfiles import StaticFiles

# Serve static files (PDFs) from 'uploaded_pdfs' directory
app.mount("/pdfs", StaticFiles(directory=PDF_STORAGE_DIR), name="pdfs")


@app.post("/add_document", summary="Upload and add a PDF document to the index")
def add_document(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    logger.info("Received file upload: %s", file.filename)

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(PDF_STORAGE_DIR, file.filename)

    try:
        # Save uploaded file with its original filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("File successfully saved at: %s", file_path)

        # Load PDF and split into pages
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for page_num, doc in enumerate(docs):
            doc.metadata['source'] = file.filename  # Use original filename as reference
            doc.metadata['page_number'] = page_num + 1

        # Split long documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=600, chunk_overlap=100
        )
        doc_splits = text_splitter.split_documents(docs)
        doc_splits = [doc for doc in doc_splits if doc.page_content.strip()]

        logger.info("Adding %d document chunks to vector store", len(doc_splits))
        vectorstore.add_documents(doc_splits)

    except Exception as e:
        logger.error("Error processing PDF: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    return {
        "message": f"Successfully added {file.filename}"
    }


@app.post("/query", response_model=QueryResponse, summary="Query the LLM model")
def query_llm(request: QueryRequest, x_session_id: str = Depends(create_or_get_session_id)):
    question = request.question
    session_id = x_session_id
    qa_chain.memory = get_memory_for_session(session_id)
    logger.info("Received query: %s", question)
    try:
        # Retrieve documents using the vector store
        retrieved_docs = retriever.get_relevant_documents(question)
        logger.info("Retrieved %d documents", len(retrieved_docs))

        # Grade documents for relevance
        docs_to_use = []
        for doc in retrieved_docs:
            grade = grade_document(question, doc.page_content)
            logger.info("Grading result: %s", grade)
            if grade == 'yes':
                docs_to_use.append(doc)

        if not docs_to_use:
            return QueryResponse(
                answer="Non sono presenti informazioni riguardo questo argomento",
                resources=[]
            )

        formatted_docs = format_docs(docs_to_use)
        # Generate an answer using the QA chain
        answer = qa_chain.run({"documents": formatted_docs, "question": question})
        resources = get_resources(docs_to_use)
        response = QueryResponse(answer=answer, resources=resources)
        return JSONResponse(content=response.dict(), headers={"x-session-id": session_id})
    except Exception as e:
        logger.error("Query failed: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/list_documents", summary="List all stored documents")
def list_documents():
    try:
        # Retrieve all stored document entries with metadata
        all_docs = vectorstore.get(include=["documents", "metadatas"])

        # Validate that expected keys exist in the response
        if not all_docs or "documents" not in all_docs or "metadatas" not in all_docs:
            raise HTTPException(status_code=404, detail="No documents found in the vector store.")

        # Extract documents and metadata
        documents = all_docs["documents"] or []
        metadatas = all_docs["metadatas"] or []

        # Ensure both lists are the same length
        if len(documents) != len(metadatas):
            raise HTTPException(status_code=500, detail="Mismatch between documents and metadata.")

        # Format the response properly
        document_list = [
            {
                "source": metadata.get("source", "Unknown"),
                "page_number": metadata.get("page_number", "Unknown"),
                "content": doc[:200]  # Preview first 200 characters
            }
            for doc, metadata in zip(documents, metadatas)
        ]

        return {"documents": document_list}

    except Exception as e:
        logger.error("Error retrieving document list: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve document list.")


@app.delete("/delete_document/{filename}", summary="Delete a specific document from the vector store and static folder")
def delete_document(filename: str, current_user: dict = Depends(get_current_user)):
    try:
        # Retrieve documents and metadata explicitly
        all_docs = vectorstore.get(include=["documents", "metadatas", "ids"])

        if not all_docs or "documents" not in all_docs or "metadatas" not in all_docs or "ids" not in all_docs:
            raise HTTPException(status_code=404, detail="No documents found in the vector store.")

        # Extract relevant data
        documents = all_docs["documents"] or []
        metadatas = all_docs["metadatas"] or []
        doc_ids = all_docs["ids"] or []

        # Ensure consistency
        if len(documents) != len(metadatas) or len(documents) != len(doc_ids):
            raise HTTPException(status_code=500, detail="Mismatch between documents, metadata, and IDs.")

        # Identify document IDs to delete based on filename match
        doc_ids_to_delete = [
            doc_id for doc_id, metadata in zip(doc_ids, metadatas)
            if metadata.get("source") == filename
        ]

        if not doc_ids_to_delete:
            raise HTTPException(status_code=404, detail=f"No documents found for filename: {filename}")

        # Remove documents from vector store
        vectorstore.delete(doc_ids_to_delete)

        # Construct file path
        file_path = os.path.join(PDF_STORAGE_DIR, filename)

        # Check if file exists and delete it
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Successfully deleted file: %s", file_path)
        else:
            logger.warning("File not found: %s", file_path)

        return {"message": f"Successfully deleted {len(doc_ids_to_delete)} documents and the file {filename}"}

    except Exception as e:
        logger.error("Error deleting document: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete document.")




# ------------------------------------------------------------------------------
# Run the application if executed as main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info")

