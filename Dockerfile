FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models

# CPU-only torch (the default Linux wheel pulls GBs of CUDA libs)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Bake the embedding model into the image so startup needs no download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY app/ app/
COPY data/chunks.jsonl data/chunks.jsonl

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]