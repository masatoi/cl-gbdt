# Open-data experiments

These are observations from running the unified API against public data, using both the
LightGBM and XGBoost backends. They document individual runs, not a benchmark or a claim
that either backend is generally more accurate. The runnable examples are:

- [`toy-data/`](toy-data/README.md): three small classification and regression datasets.
- [`credit-card-fraud/`](credit-card-fraud/README.md): a larger, imbalanced binary
  classification dataset, with chronological splits and threshold selection.

The runs used cl-gbdt commit `9b7a6ac5d3a6f39399f466a56a7a2d8417289e9a`, SBCL 2.2.9,
ASDF 3.3.7.4, LightGBM 4.7.0 and XGBoost 3.4.1 on Linux x86-64. Data preparation and
metric calculation used Python and scikit-learn 1.8.0; training and prediction used
`cl-gbdt`. Backend parameters differed and were not tuned. No data or trained models are
committed here.

## Small datasets

The Iris, Breast Cancer Wisconsin (Diagnostic), and Diabetes datasets were loaded from
scikit-learn, split approximately 60:20:20 into training, validation and test sets with
seed 42, and stratified for classification. Validation loss drove early stopping; test
scores were computed from the best iteration. The Diabetes input used `scaled=False`.

| Dataset and test metric | Training / validation / test rows | LightGBM | XGBoost | Baseline |
|---|---:|---:|---:|---:|
| [Iris](https://archive.ics.uci.edu/dataset/53/iris), accuracy | 90 / 30 / 30 | 0.9667 | 0.9000 | 0.3333 |
| [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), accuracy | 341 / 114 / 114 | 0.9649 | 0.9737 | 0.6316 |
| [Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html), RMSE | 265 / 88 / 89 | 57.622 | 55.369 | 76.906 |

The classification baseline predicts the majority class in the training set; the
regression baseline predicts the training-set mean. For the breast-cancer example the
scikit-learn labels were reversed so that malignant is 1. Both models identified 39 of
the 42 malignant test cases. These are small, single-split results, not evidence suitable
for medical use.

## Credit card fraud detection

The [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset contains 284,807 transactions, 492 of them fraudulent. The run obtained the CSV
from the [STAT 432 teaching mirror](https://github.com/stat432/credit-analysis) at commit
`125090b731479d1566bdd8b7ad1cf674fdb2bb1e`; the CSV's SHA-256 was
`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`. Its contents were not
verified byte-for-byte against Kaggle's copy.

Rows were sorted by `Time` and split about 60:20:20, keeping identical timestamps on the
same side of each boundary. `Time` was used only for splitting; the model received `V1`
through `V28` and `Amount`. A validation-loss early stopping rule selected the iteration,
then the validation set selected the probability threshold maximizing F1. The held-out
test set had 56,961 transactions, including 75 frauds.

| Test metric | LightGBM | XGBoost |
|---|---:|---:|
| Average precision | 0.7404 | 0.7961 |
| ROC-AUC | 0.8775 | 0.9814 |
| Precision | 0.9783 | 0.9167 |
| Recall | 0.6000 | 0.7333 |
| True positives | 45 | 55 |
| False positives | 1 | 5 |
| False negatives | 30 | 20 |

LightGBM's best validation-loss iteration was 1, while XGBoost's was 159; the LightGBM
threshold selected on validation was approximately 0.99999576. Neither model's scores were
calibrated. The dataset covers only about two days, and its PCA features were already
provided; the scope of the original PCA fit is unknown. These findings should not be
interpreted as a production-ready fraud detector or a general comparison of backends.
