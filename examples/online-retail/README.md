# Online Retail: repeat purchase prediction

Predict whether a customer with a purchase in the previous 90 days will purchase in the
next 30 days using the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online-retail).
This example uses temporal snapshots, native categorical features and row weights.

After the [common setup](../README.md#setup), run:

```lisp
(cl-gbdt-examples:fetch-example-data :online-retail)
;; Downloads a pinned lisp-xl source revision if it is not already installed.
(cl-gbdt-examples:ensure-retail-reader)
(load "examples/online-retail/prepare.lisp")
(load "examples/online-retail/train.lisp")
(load "examples/online-retail/evaluate.lisp")
```

The Lisp [lisp-xl](https://github.com/defunkydrummer/lisp-xl) reader streams rows
from the original workbook. Its pinned source is installed into ignored example data;
Quicklisp supplies its dependencies. The workbook, prepared data and predictions are
not committed. The recorded source ZIP SHA-256 was
`f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b`.

Rows missing a required customer, date, invoice, country or product, cancellations,
and rows with nonpositive quantities or prices are removed, leaving 397,884 positive
transaction lines. Each observation date uses only the preceding 90 days for features
and the next 30 days for the label. Features include recency, distinct invoices and
products, spending, units, mean price, recent spending and the last country. Country
codes are fitted on training snapshots only. Cancellations and returns are excluded,
not netted against purchases.

April–August 2011 snapshots form 9,620 training rows, September 1,923 validation
rows, and November 2,461 test rows. Label horizons do not cross the next split's
observation date. A customer may occur in several snapshots and splits. Each
customer's total training weight is equal, normalized to mean one; evaluation is
unweighted. Validation log loss selects the iteration and validation F1 selects the
threshold.

| Test metric | LightGBM | XGBoost |
|---|---:|---:|
| Average precision | 0.6687 | 0.6659 |
| ROC-AUC | 0.6989 | 0.6932 |
| Precision | 0.5347 | 0.5159 |
| Recall | 0.7292 | 0.7689 |

Test prevalence is 0.4291. These untuned, uncalibrated results cover one retailer and
one temporal test window; they should not be generalized to another business.
