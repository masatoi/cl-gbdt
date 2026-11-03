"""Report mean per-user NDCG@10 on the rated-item candidate sets."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
test = pd.read_csv(ROOT / "data" / "test-meta.csv")


def ndcg(labels, scores, k=10):
    n = min(k, len(labels))
    order = np.argsort(-scores, kind="stable")[:n]
    ideal = np.sort(labels)[::-1][:n]
    discounts = 1 / np.log2(np.arange(n) + 2)
    dcg = ((2.0 ** labels[order] - 1) * discounts).sum()
    idcg = ((2.0 ** ideal - 1) * discounts).sum()
    return dcg / idcg if idcg else 0.0


for name in ("baseline", "lightgbm", "xgboost"):
    scores = test.baseline.to_numpy() if name == "baseline" else np.loadtxt(
        ROOT / "output" / f"{name}-test.csv")
    if len(scores) != len(test) or not np.isfinite(scores).all():
        raise ValueError(f"Unexpected predictions for {name}")
    test["score"] = scores
    values = [ndcg(g.relevance.to_numpy(), g.score.to_numpy())
              for _, g in test.groupby("user", sort=False)]
    print(f"{name}: mean NDCG@10={np.mean(values):.4f} over {len(values)} users")
