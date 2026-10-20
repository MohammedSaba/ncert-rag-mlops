import json
import time
import faiss
import numpy as np
import mlflow
from app.chunking import load_pages, chunk_pages_sentence, chunk_pages_fixed
from app.retriever import get_model

PDF = "data/science10.pdf"
EVAL = "data/eval.jsonl"
MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
STRATEGIES = ["fixed", "sentence"]
SIZES = [500, 800, 1200]
MAX_K = 5


def is_relevant(text, evidence):
    t = text.lower()
    return all(e.lower() in t for e in evidence)


def run(strategy, size, model_name, pages, eval_set):
    if strategy == "sentence":
        chunks = chunk_pages_sentence(pages, size, 1)
    else:
        chunks = chunk_pages_fixed(pages, size, 150)

    model = get_model(model_name)
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.asarray(emb, dtype="float32"))

    ranks, per_q, lat = [], [], []
    for item in eval_set:
        t0 = time.time()
        q = model.encode([item["q"]], normalize_embeddings=True)
        _, ids = index.search(np.asarray(q, dtype="float32"), MAX_K)
        lat.append((time.time() - t0) * 1000)
        rank = None
        for r, i in enumerate(ids[0], 1):
            if is_relevant(chunks[i]["text"], item["evidence"]):
                rank = r
                break
        ranks.append(rank)
        per_q.append({"q": item["q"], "rank": rank})

    n = len(ranks)
    metrics = {f"hit_at_{k}": sum(1 for r in ranks if r and r <= k) / n for k in (1, 3, 5)}
    metrics["mrr"] = sum(1 / r for r in ranks if r) / n
    metrics["avg_retrieval_ms"] = sum(lat) / n
    metrics["num_chunks"] = len(chunks)
    return metrics, per_q


def main():
    eval_set = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
    pages = load_pages(PDF)
    mlflow.set_experiment("ncert-retrieval")

    for model_name in MODELS:
        for strategy in STRATEGIES:
            for size in SIZES:
                short = model_name.split("/")[-1]
                with mlflow.start_run(run_name=f"{strategy}-{size}-{short}"):
                    mlflow.log_params({
                        "strategy": strategy,
                        "chunk_size": size,
                        "embedding_model": model_name,
                        "eval_questions": len(eval_set),
                    })
                    metrics, per_q = run(strategy, size, model_name, pages, eval_set)
                    mlflow.log_metrics(metrics)
                    mlflow.log_dict(per_q, "per_question.json")
                    print(f"{strategy:8s} {size:5d} {short:20s} "
                          f"hit@1={metrics['hit_at_1']:.2f} hit@3={metrics['hit_at_3']:.2f} "
                          f"hit@5={metrics['hit_at_5']:.2f} mrr={metrics['mrr']:.2f}")


if __name__ == "__main__":
    main()