# Household power: minute-level forecasting

Predict active power one hour ahead from the [UCI household power dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
The example uses a temporal holdout, a custom validation metric and an optional custom
objective, all with Common Lisp callbacks.

After the [common setup](../README.md#setup), run:

```lisp
(cl-gbdt-examples:fetch-example-data :household-power)
(load "examples/household-power/prepare.lisp")
(load "examples/household-power/train.lisp")
;; Optional: asymmetric custom objective on the first 100,000 training rows.
(load "examples/household-power/custom-objective.lisp")
(load "examples/household-power/evaluate.lisp")
```

The source ZIP SHA-256 in the recorded run was
`9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff`.
At minute `t`, the target is power at `t+60`; features are current power, lags of 1, 60
and 1,440 minutes, trailing 60/1,440-minute means, hour and weekday. Missing values
make dependent rows ineligible. The 2,075,259 raw minutes yield 1,371,409 training,
290,091 validation and 288,556 test rows. A 60-minute gap keeps earlier labels before
the next split's origins.

Training uses squared error; the custom validation metric weights absolute error by
3 for true targets of at least 2 kW and by 1 otherwise.

| Test metric, kW | Persistence | LightGBM | XGBoost |
|---|---:|---:|---:|
| MAE | 0.5840 | 0.4892 | 0.4899 |
| RMSE | 0.9322 | 0.7125 | 0.7118 |
| Weighted MAE | 0.7451 | 0.6210 | 0.6221 |
| Mean peak underprediction | 1.2744 | 1.2402 | 1.2433 |

The test set has 29,020 peaks. The optional custom objective penalizes underprediction
of a peak three times as heavily and supplies gradient and Hessian. On its smaller
training subset, MAE / mean peak underprediction were 0.5722 / 1.0311 kW for
LightGBM and 0.5902 / 1.0463 kW for XGBoost. Since training sets differ, this does
not isolate the objective's effect. Results are untuned and cover one household.
