"""Evaluate sampled sparse click predictions on the future test prefix."""
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
y = np.loadtxt(ROOT / "data" / "test-label.csv")
print(f"test rows={len(y)} prevalence={y.mean():.6f}")
for name in ("lightgbm", "xgboost"):
    score = np.loadtxt(ROOT / "output" / f"{name}-test.csv")
    if len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError(f"Bad {name} predictions")
    print(f"{name}: ap={average_precision_score(y, score):.5f} "
          f"auc={roc_auc_score(y, score):.5f} logloss={log_loss(y, score):.5f}")
