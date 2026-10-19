import argparse
import json
from app.chunking import load_pages, chunk_pages_sentence, chunk_pages_fixed

parser = argparse.ArgumentParser()
parser.add_argument("--pdf", default="data/science10.pdf")
parser.add_argument("--out", default="data/chunks.jsonl")
parser.add_argument("--strategy", choices=["sentence", "fixed"], default="sentence")
parser.add_argument("--chunk_size", type=int, default=800)
parser.add_argument("--overlap", type=int, default=150)        # chars, fixed only
parser.add_argument("--overlap_sents", type=int, default=1)    # sentences, sentence only
args = parser.parse_args()

pages = load_pages(args.pdf)
if args.strategy == "sentence":
    chunks = chunk_pages_sentence(pages, args.chunk_size, args.overlap_sents)
else:
    chunks = chunk_pages_fixed(pages, args.chunk_size, args.overlap)

with open(args.out, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

lens = [len(c["text"]) for c in chunks]
print(f"Strategy: {args.strategy} | Pages: {len(pages)} | Chunks: {len(chunks)}")
print(f"Chunk length min/avg/max: {min(lens)} / {sum(lens)//len(lens)} / {max(lens)}")
for c in chunks[:3]:
    print(f"\n--- chunk {c['id']} (page {c['page']}) ---\n{c['text']}")