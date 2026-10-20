import json
import sys
from app.retriever import load_chunks, build_index, search

THRESHOLD = 0.90   # minimum hit@3

chunks = load_chunks("data/chunks.jsonl")
index = build_index(chunks)
eval_set = [json.loads(l) for l in open("data/eval.jsonl", encoding="utf-8")]

hits = 0
for item in eval_set:
    top = search(item["q"], index, chunks, k=3)
    if any(all(e.lower() in h["text"].lower() for e in item["evidence"]) for h in top):
        hits += 1

score = hits / len(eval_set)
print(f"hit@3 = {score:.2f} (threshold {THRESHOLD})")
if score < THRESHOLD:
    print("EVAL GATE FAILED")
    sys.exit(1)
print("EVAL GATE PASSED")