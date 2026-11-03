"""Choose F1 threshold on validation only; evaluate future test snapshots."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
valid_y = pd.read_csv(ROOT / "data" / "valid-meta.csv").target.to_numpy()
test_y = pd.read_csv(ROOT / "data" / "test-meta.csv").target.to_numpy()
for backend in ("lightgbm", "xgboost"):
    valid = np.loadtxt(ROOT / "output" / f"{backend}-valid.csv")
    pred = np.loadtxt(ROOT / "output" / f"{backend}-test.csv")
    if len(valid) != len(valid_y) or len(pred) != len(test_y):
        raise ValueError("Wrong number of predictions")
    precision, recall, thresholds = precision_recall_curve(valid_y, valid)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = thresholds[np.argmax(f1)]
    chosen = pred >= threshold
    print(f"{backend}: threshold={threshold:.6f} ap={average_precision_score(test_y, pred):.4f} "
          f"auc={roc_auc_score(test_y, pred):.4f} "
          f"precision={precision_score(test_y, chosen, zero_division=0):.4f} "
          f"recall={recall_score(test_y, chosen, zero_division=0):.4f}")
print(f"test prevalence={test_y.mean():.4f}")
