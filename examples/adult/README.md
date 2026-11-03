# Adult: categorical classification and model interpretation

Predict whether income exceeds $50,000/year using the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult).
The example exercises native categorical splits, feature importance, contributions,
early stopping and model save/load with both backends.

After the [common setup](../README.md#setup), run from the repository root:

```lisp
(cl-gbdt-examples:fetch-example-data :adult)
(load "examples/adult/prepare.lisp")
(load "examples/adult/train.lisp")
(load "examples/adult/evaluate.lisp")
```

Preparation splits the original `adult.data` 80/20 by label with the example's
deterministic xorshift32 generator: 26,049 training and 6,512 validation rows.
`adult.test` is the separate 16,281-row test set. Category dictionaries are fitted on
training rows only and saved in `data/categories.sexp`; 0 represents a missing value
and 1 an unseen value. The same 14 features are used by both models. XGBoost uses
`tree_method=hist` for categorical splits.

Validation log loss selects the best iteration for test predictions. Contributions,
feature importance and model round-trip checks use the full model. Contributions include
a bias term and sum to the raw margin.

| Test metric | LightGBM | XGBoost |
|---|---:|---:|
| Accuracy at 0.5 | 0.8727 | 0.8708 |
| ROC-AUC | 0.9266 | 0.9256 |
| Average precision | 0.8244 | 0.8216 |

The recorded run used SBCL 2.2.9, LightGBM 4.7.0 and XGBoost 3.4.1. Input SHA-256
values were `5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d`
for `adult.data` and `a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05`
for `adult.test`. These untuned results are from one split, not a backend benchmark.

The 1994 data include sensitive attributes, including race and sex, as features. The
evaluator reports descriptive results by sex, not a fairness certification or causal
explanation. The example is not suitable for real decisions about individuals.
