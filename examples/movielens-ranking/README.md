# MovieLens: grouped ranking

Train both ranking backends to order films a user rated by the user's rating. This is
an educational substitute for MSLR-WEB10K: Microsoft's agreement could not be retrieved
in the environment used for this run. The example exercises `make-dataset :group`,
LightGBM's `lambdarank` and XGBoost's `rank:ndcg`.

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) is distributed for
research under [GroupLens' conditions](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt):
no redistribution, and no commercial or revenue-bearing use without permission. Consult
those terms before use. Data are not committed. The recorded ZIP SHA-256 is
`50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`.

The environment requirements match [Adult](../adult/README.md): both backend shared
libraries, SBCL, CFFI, Python, NumPy, pandas, and scikit-learn.

## Run

From the repository root:

```sh
mkdir -p examples/movielens-ranking/data
curl -L --fail https://files.grouplens.org/datasets/movielens/ml-100k.zip \
  -o examples/movielens-ranking/data/ml-100k.zip
unzip -q examples/movielens-ranking/data/ml-100k.zip -d examples/movielens-ranking/data
python3 -m venv /tmp/cl-gbdt-ranking-venv
/tmp/cl-gbdt-ranking-venv/bin/pip install 'numpy==2.3.5' 'pandas==2.2.3' 'scikit-learn==1.8.0'
/tmp/cl-gbdt-ranking-venv/bin/python examples/movielens-ranking/prepare.py
ros run -- --non-interactive \
  --eval '(ql:quickload (quote (:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified)) :silent t)' \
  --load examples/movielens-ranking/train.lisp
/tmp/cl-gbdt-ranking-venv/bin/python examples/movielens-ranking/evaluate.py
```

## Evaluation and limits

Users, rather than individual ratings, are split into train/validation/test with seed 42:
565/189/189 users, with 59,811/20,496/19,693 ratings. Within each split rows are made
contiguous by user; the group-length vector is passed to each dataset. The labels are the
original 1–5 ratings shifted to 0–4. Features are age, sex, release year, and movie genres.
The test baseline orders movies by their average training-user rating, falling back to the
training global average for an unseen film. Mean per-user NDCG@10 uses gain `2^rating-1`
on the shifted labels, with zero for a group with zero ideal DCG.

| Test mean NDCG@10 | Value |
|---|---:|
| Training-set movie mean baseline | 0.6901 |
| LightGBM | 0.6721 |
| XGBoost | 0.6588 |

The untuned models did not outperform the baseline in this single split. These candidate
lists contain only films each user actually rated, so this does not measure ranking among
all available movies or establish recommendation quality. Users are disjoint but the split
is not chronological; future information leakage across calendar time is possible. Release
dates may be missing and are median-imputed. The example has no user-history features and
does not simulate exposures, negative feedback, or cold-start performance in production.
