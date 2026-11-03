# Small public datasets

Train and evaluate both backends on Iris (multiclass), Breast Cancer Wisconsin
(binary classification), and Diabetes (regression). Python prepares the input data;
Common Lisp trains, predicts, evaluates, and reports permutation importance.

## Requirements

- The cl-gbdt checkout, both backend shared libraries and Roswell/Quicklisp, installed
  as described in the [project README](../../README.markdown#installation).
- Python 3.11 or newer with scikit-learn. The recorded run used scikit-learn 1.8.0.

Commands below start at the **repository root**:

```sh
python3 -m venv /tmp/cl-gbdt-examples-venv
/tmp/cl-gbdt-examples-venv/bin/pip install 'scikit-learn==1.8.0'
/tmp/cl-gbdt-examples-venv/bin/python examples/toy-data/prepare_data.py

ros run -- --non-interactive \
  --eval '(ql:quickload (quote (:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified)) :silent t)' \
  --load examples/toy-data/analysis.lisp
```

`prepare_data.py` writes the split data to the ignored `examples/toy-data/data/`
directory; it does not need a separate dataset download. Loading `analysis.lisp` runs
all three datasets on both backends. For an existing Lisp image, load those same three
systems before loading `analysis.lisp`. From a REPL with cl-mcp, use `load-system` for
each system, then `repl-eval` to load the example by absolute path.

## Method

- A fixed random seed of 42 creates training, validation and test sets of approximately
  60:20:20; classification splits are stratified. Validation loss selects the best
  iteration through early stopping, and the held-out test set is scored at that iteration.
- Classification reports a confusion matrix and a majority-class baseline. In the breast
  cancer data, the original scikit-learn labels are reversed: 1 means malignant, 0 benign.
  The threshold is 0.5.
- Diabetes uses `scaled=False` to avoid scaling all rows before the split. It predicts a
  quantitative measure of disease progression after one year, not the presence of
  diabetes or a blood-glucose reading. Regression reports RMSE, MAE, R² and the
  training-mean baseline.
- Permutation importance is measured on the test set by shuffling one column once with
  a fixed seed. It reports the resulting increase in log loss or MSE. It is an
  exploratory diagnostic, not a causal effect; correlated features and the chosen
  permutation affect it. Do not use these test-set values to tune a model and then
  report the same test set as a final unbiased evaluation.

See the [experiment index](../README.md) for the observed results and limitations.
Iris and Breast Cancer Wisconsin (Diagnostic) come from the UCI Machine Learning
Repository, both under CC BY 4.0: Fisher (1936), [Iris](https://doi.org/10.24432/C56C76);
Wolberg et al. (1993), [Breast Cancer Wisconsin](https://doi.org/10.24432/C5DW2B).
Diabetes is provided by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html),
with its original reference to Efron et al., “Least Angle Regression” (2004).
