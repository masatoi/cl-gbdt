"""Prepare a locally supplied MSLR-WEB10K fold for the grouped ranking example."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("fold", type=Path, help="directory containing train.txt, vali.txt, test.txt")
args = parser.parse_args()
counts = {}
query_sets = {}
for split, filename in (("train", "train.txt"), ("valid", "vali.txt"), ("test", "test.txt")):
    matrix, labels, qids = load_svmlight_file(args.fold / filename, n_features=136,
                                               zero_based=False, query_id=True,
                                               dtype=np.float32)
    if (len(qids) == 0 or len(qids) != len(labels) or (qids < 0).any() or (labels < 0).any() or
            (labels > 4).any() or not np.equal(labels, np.floor(labels)).all()):
        raise ValueError(f"Invalid or empty {split} ranking data")
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"Non-finite {split} features")
    order = np.argsort(qids, kind="stable")
    qids, labels = qids[order], labels[order]
    # Dense input preserves zero semantics across backends; memory cost is rows * 136 * 4 bytes.
    dense = np.asarray(matrix[order].toarray(), dtype="<f4", order="C")
    dense.tofile(DATA / f"{split}-x.f32")
    np.asarray(labels, dtype="<f4").tofile(DATA / f"{split}-y.f32")
    unique, sizes = np.unique(qids, return_counts=True)
    query_sets[split] = set(unique.tolist())
    counts[split] = len(qids)
    (DATA / f"{split}-groups.lisp").write_text("(" + " ".join(map(str, sizes)) + ")\n")
    if split == "test":
        pd.DataFrame({"qid": qids, "relevance": labels}).to_csv(DATA / "test-meta.csv", index=False)
    print(f"{split}: rows={len(qids)} queries={len(unique)}")
    del dense, matrix
if any(query_sets[a] & query_sets[b] for a, b in (("train", "valid"), ("train", "test"),
                                                ("valid", "test"))):
    raise ValueError("Query IDs overlap between supplied splits")
(DATA / "sizes.lisp").write_text(
    f"(:train {counts['train']} :valid {counts['valid']} :test {counts['test']} :features 136)\n")
