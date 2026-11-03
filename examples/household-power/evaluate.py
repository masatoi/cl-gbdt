"""Evaluate predictions and the current-value persistence baseline."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
test = pd.read_csv(ROOT / "data" / "test-meta.csv")
y = test.target.to_numpy()
peak = y >= 2.0
weight = np.where(peak, 3.0, 1.0)
for name in ("persistence", "lightgbm", "xgboost", "lightgbm-custom", "xgboost-custom"):
    if name.endswith("-custom") and not (ROOT / "output" / f"{name}-test.csv").exists():
        continue
    pred = test.now.to_numpy() if name == "persistence" else np.loadtxt(
        ROOT / "output" / f"{name}-test.csv")
    if len(pred) != len(y) or not np.isfinite(pred).all():
        raise ValueError(f"Bad {name} predictions")
    error = pred - y
    print(f"{name}: mae={np.mean(np.abs(error)):.5f} "
          f"rmse={np.sqrt(np.mean(error ** 2)):.5f} "
          f"weighted_mae={np.average(np.abs(error), weights=weight):.5f} "
          f"peak_mean_underprediction={np.mean(-error[peak]):.5f}")
print(f"test rows={len(y)} peaks={peak.sum()} (target >= 2kW)")
