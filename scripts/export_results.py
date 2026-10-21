import mlflow

exp = mlflow.get_experiment_by_name("ncert-retrieval")
df = mlflow.search_runs([exp.experiment_id], order_by=["metrics.mrr DESC"])
cols = ["tags.mlflow.runName", "params.strategy", "params.chunk_size",
        "params.embedding_model", "metrics.hit_at_1", "metrics.hit_at_3",
        "metrics.hit_at_5", "metrics.mrr", "metrics.num_chunks"]
out = df[cols].rename(columns=lambda c: c.split(".")[-1]).round(3)
out.to_csv("results/retrieval_experiments.csv", index=False)
print(out)