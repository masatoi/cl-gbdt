# Adult: categorical classification and model interpretation

Predict whether income exceeds $50,000/year from the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult).
This educational example uses both `cl-gbdt` backends with native categorical splits,
feature importance, contribution predictions, early stopping, and model save/load.

Requires the shared libraries and Lisp systems described in the
[project README](../../README.markdown#installation), SBCL on a little-endian machine,
CFFI, Python 3.11+, NumPy, pandas and scikit-learn. Python prepares matrices and metrics;
the Lisp program trains, predicts, and checks model round-trips. Data and outputs are ignored.

## Run

From the repository root:

```sh
mkdir -p examples/adult/data
curl -L --fail https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data \
  -o examples/adult/data/adult.data
curl -L --fail https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test \
  -o examples/adult/data/adult.test
python3 -m venv /tmp/cl-gbdt-adult-venv
/tmp/cl-gbdt-adult-venv/bin/pip install 'numpy==2.3.5' 'pandas==2.2.3' 'scikit-learn==1.8.0'
/tmp/cl-gbdt-adult-venv/bin/python examples/adult/prepare.py
ros run -- --non-interactive \
  --eval '(ql:quickload (quote (:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified)) :silent t)' \
  --load examples/adult/train.lisp
/tmp/cl-gbdt-adult-venv/bin/python examples/adult/evaluate.py
```

Alternatively, load both `/unified` systems in a REPL, then load `train.lisp`.
The preparation and Lisp output formats assume little-endian IEEE-754 single floats.

## Design and recorded run

The original `adult.data` is split 80/20, stratified by label with seed 42, into 26,048
training and 6,513 validation rows. `adult.test` supplies the separate 16,281-row test
set. Category dictionaries are fitted on training data alone: category code 0 means a
missing value and 1 an unseen value. Thus missing categories are represented explicitly,
not through a backend's native NaN handling. XGBoost uses `tree_method=hist` as required
for its categorical splits. The same 14 columns are used by both models.

Both models select the best validation-log-loss iteration and predict the test set at
that iteration. The contribution and feature-importance examples use the full model,
as does the model save/reload check; they need not match the best-iteration predictions.
The final contribution value is the bias term, and all values sum to the raw margin.
The script checks that the reloaded full model reproduces full-model test predictions.

| Test metric | LightGBM | XGBoost |
|---|---:|---:|
| Accuracy at 0.5 | 0.8737 | 0.8735 |
| ROC-AUC | 0.9269 | 0.9261 |
| Average precision | 0.8240 | 0.8228 |

The recorded input SHA-256 values were `5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d`
for `adult.data` and `a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05`
for `adult.test`. The results use SBCL 2.2.9, LightGBM 4.7.0 and XGBoost 3.4.1.
Parameters were not tuned; these values are not a benchmark of the backends.

The 1994 data include sensitive attributes such as race and sex, which are used as
features here. The evaluator also reports descriptive results by sex; these are not a
fairness certification or causal explanation. This is an educational example, not a model
for real decisions about individuals. Feature contributions describe model predictions,
not causal effects.
