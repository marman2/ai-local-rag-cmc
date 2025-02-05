# ---- Build Stage ----
FROM python:3.10-slim AS builder

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends gcc

WORKDIR /app

# Copy dependency definitions and install them in a virtual environment
COPY requirements.txt .
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# ---- Final Stage ----
FROM python:3.10-slim

# Create a non-root user for security
RUN useradd --create-home appuser

WORKDIR /app

# Copy the virtual environment and application from the builder stage
COPY --from=builder /app/venv ./venv
COPY --from=builder /app .

# Create the directory for PDFs and adjust permissions so the appuser can write there.
RUN mkdir -p uploaded_pdfs && chown -R appuser:appuser uploaded_pdfs

# Ensure the virtual environment is used
ENV PATH="/app/venv/bin:$PATH"

# Expose the port for Uvicorn
EXPOSE 8000

# Switch to the non-root user
USER appuser

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
