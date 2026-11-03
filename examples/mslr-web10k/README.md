# MSLR-WEB10K: preparing a ranking fold

**Status: the pipeline has been smoke-tested with synthetic data only. It has not been
run on MSLR-WEB10K itself.** Microsoft's [dataset page](https://www.microsoft.com/en-us/research/project/mslr/)
requires users to read and accept an online license agreement. The agreement link was
inaccessible in the environment where this example was prepared; no MSLR data or metrics
were obtained. Review the current agreement yourself and acquire the dataset in accordance
with its terms before running this example. No dataset is included in this repository.

The script accepts one supplied fold containing `train.txt`, `vali.txt` and `test.txt`,
each in the original SVMlight-style format with `qid` and 136 features. It sorts rows by
query, checks that queries do not overlap across splits, and writes dense float32 matrices
and group lengths. The shared [MovieLens ranking trainer](../movielens-ranking/train.lisp)
then trains LightGBM `lambdarank` and XGBoost `rank:ndcg` models using those groups.

## Run with a separately acquired fold

Requires SBCL on little-endian hardware, CFFI, both shared libraries, Python 3.11+, NumPy,
pandas and scikit-learn. The matrix is dense to keep zero semantics consistent between
backends; allow at least `rows × 136 × 4` bytes for each materialized matrix and additional
space and memory for parsing and training. From the repository root:

```sh
python3 -m venv /tmp/cl-gbdt-mslr-venv
/tmp/cl-gbdt-mslr-venv/bin/pip install 'numpy==2.3.5' 'pandas==2.2.3' 'scikit-learn==1.8.0'
/tmp/cl-gbdt-mslr-venv/bin/python examples/mslr-web10k/prepare.py /path/to/MSLR-WEB10K/Fold1
CL_GBDT_RANKING_ROOT="$PWD/examples/mslr-web10k" ros run -- --non-interactive \
  --eval '(ql:quickload (quote (:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified)) :silent t)' \
  --load examples/movielens-ranking/train.lisp
/tmp/cl-gbdt-mslr-venv/bin/python examples/mslr-web10k/evaluate.py
```

The evaluator reports mean per-query NDCG@10 with gain `2^label - 1`; zero-ideal queries
score zero. The trainer uses fixed, untuned parameters and 80 rounds. Reusing the source's
separate validation and test sets does not by itself make these results a benchmark.
The examples' synthetic smoke test is not evidence of real-data accuracy or of successful
processing of the official archive. Data, generated files, and predictions are ignored.
