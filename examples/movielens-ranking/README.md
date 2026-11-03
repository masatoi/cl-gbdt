# MovieLens: grouped ranking

Rank the movies each user rated in [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/).
The example exercises `make-dataset :group`, LightGBM `lambdarank`, XGBoost `rank:ndcg`
and mean per-user NDCG@10.

MovieLens is distributed under [GroupLens' terms](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt):
no redistribution, and no commercial or revenue-bearing use without permission.
Review those terms before acquiring and using it. After the [common setup](../README.md#setup):

```lisp
(cl-gbdt-examples:fetch-example-data :movielens-ranking)
(load "examples/movielens-ranking/prepare.lisp")
(load "examples/movielens-ranking/train.lisp")
(load "examples/movielens-ranking/evaluate.lisp")
```

The recorded source ZIP SHA-256 was
`50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`.
Users are assigned with the reproducible xorshift32 generator, keeping them disjoint:
565/188/190 users and 59,811/18,510/21,679 ratings in train/validation/test. Within
each split rows are contiguous by user. Labels are the original 1–5 ratings shifted
to 0–4; features are age, sex, release year and movie genres. Missing release years
are replaced by the median. The baseline ranks movies by mean training-user rating,
falling back to the global training mean. NDCG uses `2^label - 1` gain and assigns
zero to a group with zero ideal DCG.

| Test mean NDCG@10 | Value |
|---|---:|
| Training-set movie mean baseline | 0.6736 |
| LightGBM | 0.6686 |
| XGBoost | 0.6484 |

The untuned models do not beat the baseline in this split. Candidate lists include
only movies users rated, not all available movies. Users are disjoint but the split
is not chronological. The example does not simulate exposure, negative feedback or
production cold-start performance.
