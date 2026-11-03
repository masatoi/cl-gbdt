# Credit card fraud detection

Analyze all 284,807 transactions in the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
created by the ULB Machine Learning Group and Worldline. Preparation, training, threshold
selection, metrics and the precision–recall visualization are implemented in Common Lisp.

After the [common setup](../README.md#setup), run:

```lisp
(cl-gbdt-examples:fetch-example-data :credit-card-fraud)
(load "examples/credit-card-fraud/prepare.lisp")
(load "examples/credit-card-fraud/train.lisp")
(load "examples/credit-card-fraud/evaluate.lisp")
```

The downloader uses the public [STAT 432 teaching mirror](https://github.com/stat432/credit-analysis)
at commit `125090b731479d1566bdd8b7ad1cf674fdb2bb1e`. The recorded CSV's SHA-256 was
`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`; identity with
Kaggle's copy was not checked. To supply your own file, put it at `data/creditcard.csv`
or set `CL_GBDT_CREDIT_CSV` to its pathname before loading `prepare.lisp`.

Evaluation writes `output/metrics.sexp` and `output/precision-recall.svg`. The latter
shows test precision–recall curves and the points selected using validation F1. Data and
generated outputs are ignored by Git. Preparation records split counts and hash-based
duplicate diagnostics in `data/metadata.sexp`.

Rows are sorted by `Time`, with equal timestamps kept together. Training, validation
and test have 170,888, 56,958 and 56,961 rows; validation and test contain 57 and 75
frauds. `Time` is excluded from the 29 model features. Validation log loss selects the
iteration; the F1-maximizing threshold is chosen on validation only. Feature-vector
matches with earlier partitions are detected using deterministic 64-bit FNV-1a hashes
of the source fields; hash collisions and different textual representations can affect
this diagnostic. Repeated vectors need not be duplicate transactions.

| Test metric | LightGBM | XGBoost |
|---|---:|---:|
| Average precision | 0.7404 | 0.7961 |
| ROC-AUC | 0.8775 | 0.9814 |
| Precision | 0.9783 | 0.9167 |
| Recall | 0.6000 | 0.7333 |
| True positives / false positives | 45 / 1 | 55 / 5 |

The LightGBM best iteration was 1, the XGBoost best iteration 159. Probabilities are not
calibrated. The source covers only about two days, and the original PCA fit population
is unknown. This single temporal split is not a production estimate or general backend
comparison.
