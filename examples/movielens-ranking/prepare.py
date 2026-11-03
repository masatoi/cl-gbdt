"""Build user-disjoint ranking groups from the MovieLens 100K dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "ml-100k"
ratings = pd.read_csv(RAW / "u.data", sep="\t", names=["user", "item", "rating", "timestamp"])
users = pd.read_csv(RAW / "u.user", sep="|", names=["user", "age", "sex", "occupation", "zip"])
items = pd.read_csv(RAW / "u.item", sep="|", encoding="latin1", header=None)
items = items[[0, 2, *range(5, 24)]]
items.columns = ["item", "release", *[f"genre_{i}" for i in range(19)]]
items["release_year"] = pd.to_datetime(items.release, format="%d-%b-%Y", errors="coerce").dt.year
items["release_year"] = items.release_year.fillna(items.release_year.median())
frame = ratings.merge(users, on="user", validate="many_to_one").merge(
    items, on="item", validate="many_to_one")
frame["male"] = (frame.sex == "M").astype(int)
frame["relevance"] = frame.rating - 1  # 0..4; a rating is not an exposure or a click.
users_train, users_test = train_test_split(users.user.to_numpy(), test_size=0.2, random_state=42)
users_train, users_valid = train_test_split(users_train, test_size=0.25, random_state=42)
feature_cols = ["age", "male", "release_year", *[f"genre_{i}" for i in range(19)]]
parts = {}
for name, ids in (("train", users_train), ("valid", users_valid), ("test", users_test)):
    # Stable user groups are required by both ranking backends.
    part = frame[frame.user.isin(ids)].sort_values(["user", "timestamp", "item"]).copy()
    parts[name] = part
    np.asarray(part[feature_cols], dtype="<f4", order="C").tofile(DATA / f"{name}-x.f32")
    np.asarray(part.relevance, dtype="<f4").tofile(DATA / f"{name}-y.f32")
    sizes = part.groupby("user", sort=False).size().to_list()
    (DATA / f"{name}-groups.lisp").write_text("(" + " ".join(map(str, sizes)) + ")\n")
    print(f"{name}: users={len(sizes)} rows={len(part)}")
means = parts["train"].groupby("item").relevance.mean()
global_mean = parts["train"].relevance.mean()
test = parts["test"][["user", "item", "relevance"]].copy()
test["baseline"] = test.item.map(means).fillna(global_mean)
test.to_csv(DATA / "test-meta.csv", index=False)
(DATA / "sizes.lisp").write_text(
    f"(:train {len(parts['train'])} :valid {len(parts['valid'])} :test {len(parts['test'])} "
    f":features {len(feature_cols)})\n")
