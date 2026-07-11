FROM python:3.9-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only (much smaller than CUDA build — ~800MB vs 2.5GB)
RUN pip install --no-cache-dir \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Pin numpy<2 (required for torch + faiss compatibility)
RUN pip install --no-cache-dir "numpy<2"

# Core ML/NLP deps
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    "sentence-transformers>=2.7.0,<3.0" \
    faiss-cpu>=1.8.0 \
    huggingface_hub>=0.23.0

# Application dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    "uvicorn[standard]>=0.29.0" \
    pydantic>=2.7.0 \
    pydantic-settings>=2.2.0 \
    python-multipart>=0.0.9 \
    pymongo>=4.6.0 \
    pandas>=2.2.0 \
    scikit-learn>=1.4.0 \
    joblib>=1.4.0 \
    python-jose[cryptography]>=3.3.0 \
    bcrypt>=4.0.0 \
    requests>=2.31.0 \
    httpx>=0.27.0

# Copy application code
COPY . .

# HuggingFace Spaces listens on port 7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
