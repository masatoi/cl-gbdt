"""Report mean NDCG@10 over the supplied held-out queries."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / "data" / "test-meta.csv")


def ndcg(labels, scores):
    n = min(10, len(labels))
    discounts = 1 / np.log2(np.arange(n) + 2)
    actual = np.argsort(-scores, kind="stable")[:n]
    ideal = np.sort(labels)[::-1][:n]
    best = ((2.0 ** ideal - 1) * discounts).sum()
    return ((2.0 ** labels[actual] - 1) * discounts).sum() / best if best else 0.0


for backend in ("lightgbm", "xgboost"):
    scores = np.loadtxt(ROOT / "output" / f"{backend}-test.csv")
    if len(scores) != len(data) or not np.isfinite(scores).all():
        raise ValueError(f"Bad {backend} predictions")
    data["score"] = scores
    values = [ndcg(g.relevance.to_numpy(), g.score.to_numpy())
              for _, g in data.groupby("qid", sort=False)]
    print(f"{backend}: NDCG@10={np.mean(values):.4f} on {len(values)} test queries")
