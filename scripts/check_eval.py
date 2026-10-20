import json
from app.retriever import load_chunks

chunks = load_chunks()
for line in open("data/eval.jsonl", encoding="utf-8"):
    item = json.loads(line)
    n = sum(all(e.lower() in c["text"].lower() for e in item["evidence"]) for c in chunks)
    flag = "  <-- FIX (0 matches)" if n == 0 else ("  <-- too broad" if n > 8 else "")
    print(f"{n:3d}  {item['q']}{flag}")