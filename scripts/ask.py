import sys
import faiss
from app.retriever import load_chunks
from app.rag import answer

chunks = load_chunks()
index = faiss.read_index("data/index.faiss")

question = " ".join(sys.argv[1:]) or "What is a chemical equation?"
result = answer(question, index, chunks)

print("\nANSWER:\n", result["answer"])
print("\nSOURCES:", result["sources"])
print("TOKENS:", result["tokens"])