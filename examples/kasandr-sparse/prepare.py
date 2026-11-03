"""Hash sparse KASANDR features on a reproducible sample of the public data."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "de"
FIELDS = ["userid", "offerid", "category", "merchant"]
BUCKETS = 2048  # separate hash space for each field
COLS = FIELDS + ["utcdate", "rating"]


def read(path, n, seed):
    # The source is not globally sorted. Draw without replacement from the complete file,
    # then sort by time. Chunked CSV reading keeps the 3 GB training file out of memory.
    with path.open("rb") as stream:
        total = sum(1 for _ in stream) - 1
    if total < n:
        raise ValueError(f"Expected at least {n} rows in {path}; got {total}")
    selected = np.sort(np.random.default_rng(seed).choice(total, size=n, replace=False))
    pieces = []
    offset = 0
    for chunk in pd.read_csv(path, sep="\t", usecols=COLS, chunksize=250_000,
                             dtype={c: "string" for c in FIELDS}):
        lo = np.searchsorted(selected, offset)
        hi = np.searchsorted(selected, offset + len(chunk))
        if hi > lo:
            pieces.append(chunk.iloc[selected[lo:hi] - offset].copy())
        offset += len(chunk)
    if offset != total:
        raise ValueError(f"Input row count changed during reading: {total} to {offset}")
    frame = pd.concat(pieces, ignore_index=True)
    if frame[FIELDS].isna().any().any() or not frame.rating.isin([0, 1]).all():
        raise ValueError("Unexpected missing fields or labels")
    frame["utcdate"] = pd.to_datetime(frame.utcdate)
    print(f"{path.name}: sampled {len(frame)} of {total} data rows")
    return frame.sort_values("utcdate", kind="stable")


source = read(RAW / "train_de.csv", 500_000, 42)
test = read(RAW / "test_de.csv", 100_000, 43)
cutoff = source.utcdate.iloc[400_000]
train = source[source.utcdate < cutoff]
valid = source[source.utcdate >= cutoff]
if train.empty or valid.empty or source.utcdate.max() >= test.utcdate.min():
    raise ValueError("Temporal split boundaries overlap")
parts = {"train": train, "valid": valid, "test": test}
for name, part in parts.items():
    # Four namespace-separated, deterministic pandas string hashes per row.
    indexes = np.empty((len(part), len(FIELDS)), dtype="<i4")
    for i, col in enumerate(FIELDS):
        hashes = pd.util.hash_pandas_object(part[col], index=False).to_numpy(dtype="uint64")
        indexes[:, i] = (hashes % BUCKETS + i * BUCKETS).astype("<i4")
    indexes.tofile(DATA / f"{name}-indices.i32")
    (np.arange(len(part) + 1, dtype="<i4") * len(FIELDS)).tofile(DATA / f"{name}-indptr.i32")
    np.asarray(part.rating, dtype="<f4").tofile(DATA / f"{name}-y.f32")
    print(f"{name}: rows={len(part)} positives={part.rating.sum()} "
          f"dates={part.utcdate.min()}..{part.utcdate.max()}")
np.savetxt(DATA / "test-label.csv", test.rating, fmt="%d")
(DATA / "sizes.lisp").write_text(
    f"(:train {len(train)} :valid {len(valid)} :test {len(test)} "
    f":features {BUCKETS * len(FIELDS)} :nonzeros-per-row {len(FIELDS)})\n")
print(f"training prevalence={train.rating.mean():.8f}, buckets/field={BUCKETS}")
