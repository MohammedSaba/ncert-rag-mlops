import os
from dotenv import load_dotenv
from openai import OpenAI
from app.retriever import search

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful tutor for NCERT Class 10 Science students. "
    "Answer ONLY using the provided context. "
    "If the answer is not in the context, say: \"I couldn't find this in the textbook.\" "
    "Keep answers clear and simple, and mention the page number(s) you used."
)


def answer(query, index, chunks, k=4):
    hits = search(query, index, chunks, k)
    context = "\n\n".join(f"[Page {h['page']}] {h['text']}" for h in hits)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [{"page": h["page"], "score": round(h["score"], 3)} for h in hits],
        "tokens": resp.usage.total_tokens if resp.usage else None,
    }