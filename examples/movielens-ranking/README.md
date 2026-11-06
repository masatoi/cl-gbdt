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

Each script works on its own directory, the one it is loaded from. The trainer and
evaluator themselves live in `train-core.lisp` and `evaluate-core.lisp` and take the
directory as an argument; the [MSLR scripts](../mslr-web10k/README.md) reuse them that way
for their own directory, so both examples can run in one image without mixing outputs.

The recorded source ZIP SHA-256 was
`50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`.
Users are assigned with the reproducible xorshift32 generator, keeping them disjoint:
565/188/190 users and 59,811/18,510/21,679 ratings in train/validation/test. Within
each split rows are contiguous by user. Labels are the original 1–5 ratings shifted
to 0–4; features are age, sex, release year and movie genres. Missing release years
are replaced by the median. The baseline ranks movies by mean training-user rating,
falling back to the global training mean. NDCG uses `2^label - 1` gain and assigns
zero to a group with zero ideal DCG. Evaluation requires finite integer relevance
values from 0 through 4. Equal prediction scores keep their input order. Within each
user, preparation orders ratings by timestamp from earliest to latest, then by movie ID
when timestamps are equal, so ties in score follow that order.

Preparation invalidates previous predictions before replacing the split files and writes
completion metadata only after all splits are ready. Train again after preparation or a
failed training attempt. Training stages both backend predictions and attempts to restore
the previous files if publication fails. Evaluation checks that the predictions and data
belong to the same preparation before it prints results. Folds prepared before generation
tracking was added must be prepared again.

Both ranking examples use [`ranking-state.lisp`](../ranking-state.lisp), which keeps their
state with the other examples' [`lifecycle.lisp`](../lifecycle.lisp) under ranking file
names. They require Unix `flock` and `/dev/urandom`. Preparation, training
and evaluation take a nonblocking lock on the example directory; an overlapping operation
fails immediately. Keep `output/.ranking.lock` in place while an operation is running.
Evaluation can use an existing read-only lock file and output directory; preparation and
training require write access.
After an interrupted publication, `output/.movielens-training-*` may hold
`.previous-*-test.csv` backups; inspect them before deleting that directory.

| Test mean NDCG@10 | Value |
|---|---:|
| Training-set movie mean baseline | 0.6736 |
| LightGBM | 0.6686 |
| XGBoost | 0.6484 |

The untuned models do not beat the baseline in this split. Candidate lists include
only movies users rated, not all available movies. Users are disjoint but the split
is not chronological. The example does not simulate exposure, negative feedback or
production cold-start performance.
