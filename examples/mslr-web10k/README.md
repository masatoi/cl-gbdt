# MSLR-WEB10K: preparing a ranking fold

**Status: tested with synthetic data only, not the MSLR-WEB10K dataset.**
[Microsoft's dataset page](https://www.microsoft.com/en-us/research/project/mslr/)
requires users to read and accept a license agreement. That agreement was inaccessible
in the environment where this example was developed. Acquire the dataset yourself in
accordance with its current terms; no MSLR data or real-data metrics are included.

After the [common setup](../README.md#setup), supply a fold containing `train.txt`,
`vali.txt` and `test.txt`. Set the path below to your own fold before loading:

```lisp
(require :sb-posix)
(sb-posix:setenv "CL_GBDT_MSLR_FOLD" "/path/to/MSLR-WEB10K/Fold1" 1)
(load "examples/mslr-web10k/prepare.lisp")
(load "examples/mslr-web10k/train.lisp")
(load "examples/mslr-web10k/evaluate.lisp")
```

`train.lisp` and `evaluate.lisp` here use the MovieLens example's shared trainer and
evaluator, passing this directory explicitly, so the two examples can share an image.
Both ranking examples use the shared state and locking code in
[`ranking-state.lisp`](../ranking-state.lisp), which is built on the other examples'
[`lifecycle.lisp`](../lifecycle.lisp).

The adapter parses SVMlight-style rows with query IDs and 136 features, sorts each
split by query, and rejects missing IDs, invalid labels and query IDs shared between
splits. The representation is dense to preserve zero semantics across backends; allow
at least `rows × 136 × 4` bytes per matrix plus parsing and training memory. The shared
trainer uses fixed parameters and 80 rounds. Evaluation requires finite integer relevance
values from 0 through 4 and reports mean per-query NDCG@10 using `2^label - 1` gain; a
zero-ideal query scores zero. Equal prediction scores keep their input order, which this
adapter preserves within each query from the source fold. In the MovieLens example, that
input order is rating timestamp from earliest to latest within each user, then movie ID
when timestamps are equal. Successful synthetic tests
do not establish that the official archive has been processed successfully.

Preparation removes its completion marker before replacing data and restores it only
after every split is written. If preparation fails, rerun it successfully before
training or evaluation; both refuse to use the incomplete fold. Preparation also
invalidates previous predictions without deleting them. Run training again before
evaluation after preparing a fold or after a failed training attempt.

Each prepared fold gets a new generation. Training keeps the previous prediction files
until both backends succeed; if score publication fails, it attempts to restore those
files. Once training acquires the lock, it invalidates the old completion marker; a
subsequent failure leaves evaluation disabled. If publishing the new marker fails after
both scores were replaced, the previous scores are restored as well. Evaluation checks
the generation again before
printing results. If a fold was prepared before generation tracking was added, run
`prepare.lisp` again before training it.

This adapter requires a Unix system with `flock` and `/dev/urandom`. Preparation,
training and evaluation take a nonblocking lock on this directory: an overlapping
operation fails immediately, and the lock is released when its process exits. Keep
`output/.ranking.lock` in place while any operation is running. If a process stops during
publication, `output/.mslr-training-*` can contain `.previous-*-test.csv` backups;
inspect those files before removing the temporary directory. The lock coordinates these
MSLR scripts; manual changes to generated files are outside that coordination.
Evaluation can use an existing read-only lock file and output directory; preparation and
training require write access.
