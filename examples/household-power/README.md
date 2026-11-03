# Household power: minute-level forecasting

Predict the active power one hour ahead using the [UCI Individual Household Electric Power
Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
This example demonstrates a time-based holdout and a Lisp custom evaluation callback with
early stopping on both backends. The measurements were collected in 2006–2010.

Requirements match [Adult](../adult/README.md): SBCL, CFFI, both shared libraries,
Python, NumPy, pandas and scikit-learn. No raw data or generated output is committed.

## Run

From the repository root:

```sh
mkdir -p examples/household-power/data
curl -L --fail 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip' \
  -o examples/household-power/data/source.zip
unzip -q examples/household-power/data/source.zip -d examples/household-power/data
python3 -m venv /tmp/cl-gbdt-power-venv
/tmp/cl-gbdt-power-venv/bin/pip install 'numpy==2.3.5' 'pandas==2.2.3'
/tmp/cl-gbdt-power-venv/bin/python examples/household-power/prepare.py
ros run -- --non-interactive \
  --eval '(ql:quickload (quote (:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified)) :silent t)' \
  --load examples/household-power/train.lisp
# Optional: also run a custom objective on a smaller training subset.
ros run -- --non-interactive \
  --eval '(ql:quickload (quote (:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified)) :silent t)' \
  --load examples/household-power/custom-objective.lisp
/tmp/cl-gbdt-power-venv/bin/python examples/household-power/evaluate.py
```

The recorded archive SHA-256 was
`9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff`.

## Method and result

At minute `t`, the target is the observed active power at `t + 60 minutes`; features
include current power, lags of 1, 60 and 1,440 minutes, trailing 60/1,440-minute means,
hour, and weekday. Missing observations make any row requiring them ineligible; the raw
2,075,259 rows yield 1,371,409 train, 290,091 validation and 288,556 test rows. A
60-minute gap before each split ensures an earlier split's targets end before the next
split's prediction origins. Test features can use past observations from earlier periods.

Training uses squared error, but `:evaluation` computes weighted absolute error on the
validation set: true targets at or above 2 kW get weight 3, all others weight 1. The
custom metric selects the best iteration; it does not change the training objective.

| Test metric, kW | Persistence | LightGBM | XGBoost |
|---|---:|---:|---:|
| MAE | 0.5840 | 0.4892 | 0.4899 |
| RMSE | 0.9322 | 0.7125 | 0.7118 |
| Weighted MAE | 0.7451 | 0.6210 | 0.6221 |
| Mean underprediction on peak targets | 1.2744 | 1.2402 | 1.2433 |

The persistence baseline predicts the current measurement. The held-out test set contains
29,020 peak observations. These are untuned, single-household, single-split results; the
conditional peak underprediction is a signed mean, not a bound. The model assumes current
power is available at prediction time, and weather or real operational constraints are
not modeled.

The optional custom objective uses an asymmetric squared error: underpredicting a target
of at least 2 kW has three times the penalty. It explicitly supplies both gradient and
Hessian. To keep the callback example manageable it trains on the *first 100,000* training
rows rather than the full training set, and uses the same weighted-MAE validation rule.
The corresponding test MAE / peak mean underprediction were 0.5722 / 1.0311 kW for
LightGBM and 0.5902 / 1.0463 kW for XGBoost. Because the training sets differ, the
differences from the table cannot be attributed to the objective alone.
