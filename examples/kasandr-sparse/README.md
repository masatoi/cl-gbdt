# KASANDR: sparse click prediction

Train on a reproducible sample of [UCI KASANDR](https://archive.ics.uci.edu/dataset/385/kasandr),
an e-commerce advertisement dataset, using `cl-gbdt:make-csr-matrix`. The source has
millions of observations and high-cardinality user, offer, category and merchant IDs.
UCI distributes it under CC BY 4.0; credit the dataset creators when using it. No data
or generated matrices are committed here.

The run needs both shared libraries, SBCL on a little-endian host, CFFI, Python,
NumPy, pandas and scikit-learn. Reserve at least several gigabytes for the compressed
and extracted data. Preparation scans the entire raw training and test files.

## Run

From the repository root:

```sh
mkdir -p examples/kasandr-sparse/data
curl -L --fail https://archive.ics.uci.edu/static/public/385/kasandr.zip \
  -o examples/kasandr-sparse/data/source.zip
unzip -p examples/kasandr-sparse/data/source.zip de.tar.bz2 \
  > examples/kasandr-sparse/data/de.tar.bz2
tar --no-same-owner -xjf examples/kasandr-sparse/data/de.tar.bz2 -C examples/kasandr-sparse/data
python3 -m venv /tmp/cl-gbdt-kasandr-venv
/tmp/cl-gbdt-kasandr-venv/bin/pip install 'numpy==2.3.5' 'pandas==2.2.3' 'scikit-learn==1.8.0'
/tmp/cl-gbdt-kasandr-venv/bin/python examples/kasandr-sparse/prepare.py
ros run -- --non-interactive \
  --eval '(ql:quickload (quote (:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified)) :silent t)' \
  --load examples/kasandr-sparse/train.lisp
/tmp/cl-gbdt-kasandr-venv/bin/python examples/kasandr-sparse/evaluate.py
```

The downloaded outer ZIP's SHA-256 was
`1851d21603d24c4baa060d4e73ce98a2393effe3d0af68b59e63fb0bd86db509`.

## Method

The preparation program samples 500,000 training-file rows and 100,000 test-file rows
without replacement with seeds 42 and 43. The archived files examined in this run had
15,844,717 and 638,478 data rows respectively. The original row order is not globally
chronological, so each sample is sorted by timestamp. The training-file sample is split
approximately 80/20 by timestamp, preserving equal timestamps on one side of the cut;
the test sample is from the separately provided later period. The program checks that
the sampled periods do not overlap. This is a bounded sample, not full-data training.

Each of the four ID fields is mapped to its own 2,048 hash buckets. Thus each row has
four explicitly stored ones and 8,192 possible columns; the CSR representation avoids
materializing a dense matrix. Hash collisions lose information, and parameters and
hash-space size are illustrative rather than tuned. Input encoding uses pandas' string
hashing, so the package version is pinned above.

The representation's absent entries need special care. With default LightGBM settings,
an absent CSR entry is zero and is declared `:implicit-value 0d0`. XGBoost interprets
an absent entry as missing: the example first checks that it refuses a matrix declaring
zero, then intentionally trains a separate `:implicit-value :missing` representation.
**The two experiments therefore do not have the same missing-value semantics**; their
scores should not be used as a controlled backend comparison. XGBoost's sparse path also
does not support `:contrib` prediction. See the
[sparse-input guide](../../docs/user-guide/data-and-prediction.md#sparse-input-csr-matrices).

## Recorded result

| Split | Rows | Click rate |
|---|---:|---:|
| Training | 399,998 | 0.0355 |
| Validation | 100,002 | 0.0802 |
| Test | 100,000 | 0.3209 |

| Test metric | LightGBM | XGBoost |
|---|---:|---:|
| Average precision | 0.4519 | 0.4081 |
| ROC-AUC | 0.6590 | 0.5820 |
| Log loss | 1.0525 | 1.0618 |

The large difference in observed click rates is a major distribution shift in this
sample. Probabilities are not calibrated; no production click-rate or revenue claim
should be inferred. The experiments use different backend-specific parameters, and
the feature hashing obscures individual source identifiers.
