import faiss
from app.retriever import load_chunks, build_index

chunks = load_chunks()
index = build_index(chunks)
faiss.write_index(index, "data/index.faiss")
print(f"Indexed {index.ntotal} chunks -> data/index.faiss")