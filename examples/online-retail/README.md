# Online Retail: repeat purchase prediction

Use the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online-retail)
to predict whether a customer with a purchase in the previous 90 days will make a
qualifying purchase in the next 30 days. This example demonstrates time-based snapshots,
native categorical features and per-row sample weights in `cl-gbdt`.

Requirements match [Adult](../adult/README.md), plus `openpyxl` for the input workbook.
No workbook, prepared matrices or predictions are committed.

## Run

From the repository root:

```sh
mkdir -p examples/online-retail/data
curl -L --fail 'https://archive.ics.uci.edu/static/public/352/online+retail.zip' \
  -o examples/online-retail/data/source.zip
unzip -q examples/online-retail/data/source.zip -d examples/online-retail/data
python3 -m venv /tmp/cl-gbdt-retail-venv
/tmp/cl-gbdt-retail-venv/bin/pip install \
  'numpy==2.3.5' 'pandas==2.2.3' 'scikit-learn==1.8.0' 'openpyxl==3.1.5'
/tmp/cl-gbdt-retail-venv/bin/python examples/online-retail/prepare.py
ros run -- --non-interactive \
  --eval '(ql:quickload (quote (:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified)) :silent t)' \
  --load examples/online-retail/train.lisp
/tmp/cl-gbdt-retail-venv/bin/python examples/online-retail/evaluate.py
```

The recorded archive SHA-256 is
`f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b`.

## Method and result

Remove rows without customer ID, date, country or product, cancellations, nonpositive
quantities and nonpositive prices. This leaves 397,884 positive transaction lines. For
each observation date, build customers' 90-day histories using only preceding records;
the label is whether they appear in the next 30 days. Features include recency, number
of invoices, spending, units, distinct products, mean unit price, last country and
30-day spending. Country dictionaries are fitted on training snapshots only; 0 denotes
an unseen country. Cancellations and returns are excluded, not netted against purchases.

April–August 2011 snapshots form the 9,620 training rows, September the 1,923 validation
rows and November the 2,461 test rows. The last training and validation label horizons
end before the following split's observation date. A customer can occur in multiple
snapshots and splits; the task models returning customers rather than generalization to
entirely new customers. During training, each customer's total weight across training
snapshots is equal, and weights are normalized to mean one. Validation and test metrics
are unweighted. Best iteration uses validation log loss; the F1-maximizing threshold is
chosen on validation only.

| Test metric | LightGBM | XGBoost |
|---|---:|---:|
| Average precision | 0.6687 | 0.6659 |
| ROC-AUC | 0.6989 | 0.6932 |
| Precision | 0.5347 | 0.5159 |
| Recall | 0.7292 | 0.7689 |

The test positive rate is 0.4291. The example has untuned parameters, one temporal test
window, and no probability calibration. Its transaction data cover only one retailer;
these figures are not estimates for another business.
