import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def get_model(name=MODEL_NAME):
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model


def load_chunks(path="data/chunks.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(chunks):
    texts = [c["text"] for c in chunks]
    emb = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(emb.shape[1])   # inner product = cosine on normalized vectors
    index.add(np.asarray(emb, dtype="float32"))
    return index


def search(query, index, chunks, k=4):
    q = get_model().encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), k)
    return [
        {**chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i != -1
    ]