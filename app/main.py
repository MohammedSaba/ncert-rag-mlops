import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import answer
from app.retriever import build_index, load_chunks

CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
INDEX_PATH = os.getenv("INDEX_PATH", "data/index.faiss")
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
REQ_LOG = os.path.join(LOG_DIR, "requests.jsonl")
FB_LOG = os.path.join(LOG_DIR, "feedback.jsonl")

state = {}


def log_jsonl(path, record):
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@asynccontextmanager
async def lifespan(app):
    chunks = load_chunks(CHUNKS_PATH)
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = build_index(chunks)   # rebuild if missing (Docker / CI)
    state["chunks"], state["index"] = chunks, index
    yield


app = FastAPI(title="NCERT Science Tutor (RAG)", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class FeedbackRequest(BaseModel):
    request_id: str
    helpful: bool
    comment: Optional[str] = Field(None, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state.get("chunks", []))}


@app.post("/ask")
def ask(req: AskRequest):
    rid = uuid.uuid4().hex[:12]
    t0 = time.time()
    try:
        result = answer(req.question, state["index"], state["chunks"])
    except Exception as e:
        log_jsonl(REQ_LOG, {"request_id": rid, "question": req.question, "error": str(e)})
        raise HTTPException(status_code=502, detail="LLM backend unavailable")

    latency_ms = round((time.time() - t0) * 1000)
    top_score = result["sources"][0]["score"] if result["sources"] else None
    log_jsonl(REQ_LOG, {
        "request_id": rid,
        "question": req.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "top_score": top_score,
        "tokens": result["tokens"],
        "latency_ms": latency_ms,
    })
    return {"request_id": rid, "latency_ms": latency_ms, **result}


@app.post("/feedback")
def feedback(fb: FeedbackRequest):
    log_jsonl(FB_LOG, fb.dict())
    return {"status": "recorded"}
