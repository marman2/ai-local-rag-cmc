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

def _process_embedding(embedding):
    """
    Convert the embedding to a list of floats (or list of lists of floats).
    This function assumes that embedding is a list (or nested list) of values
    that can be converted to float.
    """
    try:
        if isinstance(embedding, list):
            # Check if the first element is a list (i.e. a list of lists)
            if embedding and isinstance(embedding[0], list):
                # Convert each sublist to a list of floats
                return [[float(x) for x in sublist] for sublist in embedding]
            else:
                # Convert the single list to floats
                return [float(x) for x in embedding]
        else:
            raise ValueError("Embedding is not a list.")
    except Exception as e:
        logger.error("Error processing embedding: %s", e)
        raise e

class OllamaEmbeddingWrapper:
    def __init__(self, model: str):
        self.model = model

    def __call__(self, text: str) -> List[float]:
        payload = {
            "model": self.model,
            "input": text
        }
        url = f"{OLLAMA_API_URL}/embed"
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            # Check for both "embedding" and "embeddings" keys.
            raw_embedding = data.get("embedding") or data.get("embeddings")
            if raw_embedding is None:
                raise ValueError("No embedding returned from Ollama.embed. Response was: " + str(data))
            # Process the raw embedding into a proper list of floats (or list of lists of floats)
            processed_embedding = _process_embedding(raw_embedding)
            return processed_embedding
        except Exception as e:
            logger.error("Error calling Ollama.embed: %s", e)
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.__call__(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
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
    persist_directory="/tmp/chroma"
)
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

Your answer should be clear, self-contained, and concise.
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
    allow_origins=["*"],  # In production, restrict this to allowed origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="API Root")
def read_root():
    return {"message": "Welcome to the Document QA API. Visit /docs for API documentation."}

@app.post("/add_document", summary="Upload and add a PDF document to the index")
def add_document(file: UploadFile = File(...)):
    logger.info("Received file upload: %s", file.filename)
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(PDF_STORAGE_DIR, file.filename)
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load PDF and split into pages
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for page_num, doc in enumerate(docs):
            doc.metadata['source'] = file.filename
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

    return {"message": f"Successfully added {file.filename}"}

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

# ------------------------------------------------------------------------------
# Run the application if executed as main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info")

