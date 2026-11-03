# MSLR-WEB10K: preparing a ranking fold

**Status: tested with synthetic data only, not the MSLR-WEB10K dataset.**
[Microsoft's dataset page](https://www.microsoft.com/en-us/research/project/mslr/)
requires users to read and accept a license agreement. That agreement was inaccessible
in the environment where this example was developed. Acquire the dataset yourself in
accordance with its current terms; no MSLR data or real-data metrics are included.

After the [common setup](../README.md#setup), supply a fold containing `train.txt`,
`vali.txt` and `test.txt`. Set the two paths below to your own paths before loading:

```lisp
(require :sb-posix)
(sb-posix:setenv "CL_GBDT_MSLR_FOLD" "/path/to/MSLR-WEB10K/Fold1" 1)
(sb-posix:setenv "CL_GBDT_RANKING_ROOT"
                 "/absolute/path/to/cl-gbdt/examples/mslr-web10k" 1)
(load "examples/mslr-web10k/prepare.lisp")
(load "examples/movielens-ranking/train.lisp")
(load "examples/movielens-ranking/evaluate.lisp")
```

The adapter parses SVMlight-style rows with query IDs and 136 features, sorts each
split by query, and rejects missing IDs, invalid labels and query IDs shared between
splits. The representation is dense to preserve zero semantics across backends; allow
at least `rows × 136 × 4` bytes per matrix plus parsing and training memory. The shared
trainer uses fixed parameters and 80 rounds. Evaluation reports mean per-query NDCG@10
using `2^label - 1` gain; a zero-ideal query scores zero. Successful synthetic tests
do not establish that the official archive has been processed successfully.
