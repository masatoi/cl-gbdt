# Credit card fraud detection

Train LightGBM and XGBoost through the unified API on all 284,807 transactions in the
Credit Card Fraud Detection dataset. The Lisp code trains and predicts; Python prepares
the data, chooses a threshold from the validation set, calculates metrics and plots a
precision-recall curve. No raw dataset, binary matrices, predictions or plots are tracked.

## Requirements and data

- The cl-gbdt checkout and both shared libraries, installed as in the
  [project README](../../README.markdown#installation).
- SBCL on a little-endian machine. `train.lisp` uses SBCL array pinning, CFFI and C
  `fread` to load the prepared little-endian float32 matrices efficiently.
- Python 3.11 or newer, NumPy, pandas, scikit-learn and matplotlib. The recorded run used
  Python 3.12, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0 and matplotlib 3.10.8.

The original dataset is [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
by the ULB Machine Learning Group and Worldline. Provide its `creditcard.csv` yourself.
The recorded run used the public [STAT 432 teaching mirror](https://github.com/stat432/credit-analysis)
at commit `125090b731479d1566bdd8b7ad1cf674fdb2bb1e`. That mirror's CSV had SHA-256
`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`; byte-for-byte
identity with Kaggle's copy was not checked.

One way to retrieve that mirror, without putting the dataset in this repository:

```sh
git clone https://github.com/stat432/credit-analysis.git /tmp/cl-gbdt-credit-data
git -C /tmp/cl-gbdt-credit-data checkout 125090b731479d1566bdd8b7ad1cf674fdb2bb1e
unzip -p /tmp/cl-gbdt-credit-data/data-raw/creditcard.csv.zip creditcard.csv \
  > /tmp/cl-gbdt-creditcard.csv
sha256sum /tmp/cl-gbdt-creditcard.csv
```

## Run

From the **repository root**:

```sh
python3 -m venv /tmp/cl-gbdt-fraud-venv
/tmp/cl-gbdt-fraud-venv/bin/pip install \
  'numpy==2.3.5' 'pandas==2.2.3' 'scikit-learn==1.8.0' 'matplotlib==3.10.8'
/tmp/cl-gbdt-fraud-venv/bin/python examples/credit-card-fraud/prepare.py \
  /tmp/cl-gbdt-creditcard.csv

ros run -- --non-interactive \
  --eval '(ql:quickload (quote (:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified)) :silent t)' \
  --load examples/credit-card-fraud/train.lisp

/tmp/cl-gbdt-fraud-venv/bin/python examples/credit-card-fraud/evaluate.py
/tmp/cl-gbdt-fraud-venv/bin/python examples/credit-card-fraud/plot.py
```

The scripts write to the ignored `data/` and `output/` directories here. `evaluate.py`
writes metrics as `output/metrics.json`; `plot.py` writes `output/precision-recall.png`.
The Lisp script runs both backends when loaded. In an existing REPL, load both `/unified`
systems first. With cl-mcp, use `load-system` and then `repl-eval` to load the script.

## Evaluation and limits

Rows are sorted by transaction time, with equal timestamps kept together at split
boundaries. Training, validation and test contain 170,888, 56,958 and 56,961 rows;
the latter two contain 57 and 75 frauds. `Time` is used for splitting but excluded from
the 29 model features. Validation log loss determines the best training iteration;
the threshold maximizing validation F1 is then applied to the test set. The evaluation
reports average precision, ROC-AUC, precision, recall, confusion counts and top-100
performance. [Recorded results](../README.md#credit-card-fraud-detection) are from a
single split and should not be interpreted as a production estimate.

The original data already contains PCA-transformed features; the population used to fit
that PCA is unknown. The preparation script also reports hash-based matches of feature
vectors across time splits; these may be repeated transactions and are not necessarily
identical records. The evaluator reports test metrics both with and without rows whose
feature hashes appeared in earlier splits. Hashes can collide.

The backends use different, untuned parameter vocabularies. This example does not calibrate
probabilities or optimize fraud losses against the operational costs of false alarms.
