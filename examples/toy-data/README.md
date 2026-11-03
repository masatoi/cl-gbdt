# Small public datasets

Train and evaluate both backends on Iris (multiclass), Breast Cancer Wisconsin
(binary classification) and Diabetes (regression). The dataset files are loaded
directly; no Python runtime is required.

After the [common setup](../README.md#setup), run:

```lisp
(cl-gbdt-examples:fetch-example-data :toy-data)
(load "examples/toy-data/prepare.lisp")
(load "examples/toy-data/analysis.lisp")
```

The downloader gets the original numeric files from scikit-learn's `1.8.0` source
tree. The Diabetes feature file contains unscaled values. Preparation uses a fixed
xorshift32 shuffle, with class-stratified approximately 60:20:20 splits for
classification. Validation loss selects the iteration; the held-out test set is
scored at that iteration. The breast-cancer labels are reversed so malignant is 1.

| Test metric | Train / valid / test | LightGBM | XGBoost | Baseline |
|---|---:|---:|---:|---:|
| Iris accuracy | 90 / 30 / 30 | 0.9333 | 0.9333 | 0.3333 |
| Breast cancer accuracy | 341 / 113 / 115 | 0.9826 | 0.9913 | 0.6261 |
| Diabetes RMSE | 265 / 88 / 89 | 59.064 | 57.469 | 74.791 |

The classification baseline predicts the majority training class; the regression
baseline predicts the training mean. The models identified 42/43 and 43/43 malignant
test cases respectively. These small, single-split results are not evidence suitable
for medical use.

Permutation importance shuffles each test feature once with a fixed seed and reports
the resulting increase in log loss or MSE. It is exploratory, not causal; correlated
features and the permutation affect it. Do not use the same test importance to tune a
model and then report that test set as an unbiased final evaluation.

Iris and Breast Cancer Wisconsin (Diagnostic) originate from the UCI Machine Learning
Repository: [Iris](https://doi.org/10.24432/C56C76), [Breast Cancer Wisconsin](https://doi.org/10.24432/C5DW2B).
The [Diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
predicts disease progression after one year, not blood glucose or diabetes presence.
