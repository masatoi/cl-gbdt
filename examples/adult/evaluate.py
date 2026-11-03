"""Report held-out metrics and descriptive subgroup errors."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
test = pd.read_csv(ROOT / "data" / "test-meta.csv")
for backend in ("lightgbm", "xgboost"):
    score = np.loadtxt(ROOT / "output" / f"{backend}-test.csv")
    if len(score) != len(test) or not np.isfinite(score).all():
        raise ValueError(f"Bad {backend} scores")
    pred = score >= 0.5
    print(f"{backend}: accuracy={accuracy_score(test.income, pred):.4f} "
          f"roc_auc={roc_auc_score(test.income, score):.4f} "
          f"average_precision={average_precision_score(test.income, score):.4f}")
    for sex in sorted(test.sex.dropna().unique()):
        selected = test.sex.eq(sex).to_numpy()
        actual = test.income.to_numpy()[selected]
        guesses = pred[selected]
        print(f"  sex={sex}: n={selected.sum()} accuracy={accuracy_score(actual, guesses):.4f} "
              f"positive_rate={guesses.mean():.4f} "
              f"false_negative_rate={(~guesses[actual == 1]).mean():.4f}")
