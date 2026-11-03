"""Prepare the UCI Adult data for the native categorical cl-gbdt example."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
           "hours_per_week", "native_country", "income"]
CATS = ["workclass", "education", "marital_status", "occupation", "relationship", "race",
        "sex", "native_country"]
FEATURES = COLUMNS[:-1]


def load(path):
    frame = pd.read_csv(path, names=COLUMNS, skipinitialspace=True, comment="|", na_values="?")
    frame = frame.dropna(subset=["income"]).copy()
    frame["income"] = frame["income"].str.rstrip(".").map({"<=50K": 0, ">50K": 1})
    if frame["income"].isna().any():
        raise ValueError("Unexpected income label")
    return frame


def main():
    source = load(DATA / "adult.data")
    test = load(DATA / "adult.test")
    train, valid = train_test_split(source, test_size=0.2, random_state=42,
                                    stratify=source.income)
    # 0 denotes missing and 1 unseen. The dictionary is fitted on training rows only.
    maps = {col: {value: i + 2 for i, value in enumerate(sorted(train[col].dropna().unique()))}
            for col in CATS}
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        x = part[FEATURES].copy()
        for col in CATS:
            values = part[col]
            x[col] = values.map(maps[col]).fillna(1).where(values.notna(), 0)
        a = np.asarray(x, dtype="<f4", order="C")
        if not np.isfinite(a).all():
            raise ValueError(f"Non-finite numeric data in {name}")
        a.tofile(DATA / f"{name}-x.f32")
        np.asarray(part.income, dtype="<f4").tofile(DATA / f"{name}-y.f32")
        print(f"{name}: rows={len(part)}, positives={part.income.sum()}")
    test[["income", "sex", "race"]].to_csv(DATA / "test-meta.csv", index=False)
    (DATA / "categories.json").write_text(json.dumps(maps, indent=2) + "\n")
    sizes = (f"(:train {len(train)} :valid {len(valid)} :test {len(test)} "
             f":features {len(FEATURES)} :categorical "
             f"({' '.join(str(FEATURES.index(c)) for c in CATS)}))\n")
    (DATA / "sizes.lisp").write_text(sizes)


if __name__ == "__main__":
    main()
