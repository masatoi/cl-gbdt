# Open-data experiments

These examples use `cl-gbdt` with both the LightGBM and XGBoost backends. Dataset
retrieval, decompression, preprocessing, model training and evaluation are implemented
in Common Lisp. Generated data, models and predictions are not committed.

| Example | Techniques |
|---|---|
| [Small datasets](toy-data/README.md) | Multiclass and binary classification, regression, permutation importance |
| [Credit card fraud](credit-card-fraud/README.md) | Chronological split, imbalanced classification, threshold selection |
| [Adult](adult/README.md) | Categorical features, feature contributions, model persistence |
| [MovieLens](movielens-ranking/README.md) | User-disjoint learning to rank, grouped NDCG |
| [MSLR-WEB10K](mslr-web10k/README.md) | Ranking-fold adapter; synthetic data only |
| [Household power](household-power/README.md) | Time-series features, custom metric and objective |
| [Online Retail](online-retail/README.md) | Excel input, temporal snapshots, categorical features and weights |
| [KASANDR](kasandr-sparse/README.md) | Sampling large files, feature hashing and sparse CSR matrices |

## Setup

Install `cl-gbdt` and the two backend shared libraries. If the libraries are already
installed, set `CL_GBDT_LIGHTGBM_LIB` and `CL_GBDT_XGBOOST_LIB` to their paths; see
[backend discovery](../docs/user-guide/backends.md#finding-the-shared-library). The
project's optional `tools/fetch-libs.sh` uses Python, but it is not needed when the
shared libraries are installed separately. The examples use SBCL, CFFI and
little-endian IEEE-754 single floats. Dataset retrieval requires a filesystem that supports
hard links and the Common Lisp systems `dexador`, `zip`, `semz.decompress`, `archive` and
`ironclad`, available through Quicklisp. The dataset preparation, training and evaluation
examples use Unix `flock` and `/dev/urandom` for their state management, which
[`lifecycle.lisp`](lifecycle.lisp) implements for all of them. If a private staging
directory cannot be removed once a run has finished, a warning names it and the run's
outcome stands; remove the directory by hand.
MovieLens and MSLR have their own usage terms; read their respective guides first.

From a REPL whose working directory is the repository root:

```lisp
(ql:quickload '(:dexador :zip :semz.decompress :archive :ironclad
                :cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified))
(load "examples/fetch-data.lisp")

;; For example, download, preprocess, train and evaluate Adult:
(cl-gbdt-examples:fetch-example-data :adult)
(load "examples/adult/prepare.lisp")
(load "examples/adult/train.lisp")
(load "examples/adult/evaluate.lisp")
```

The downloader streams files to disk, supports `HTTPS_PROXY` through Dexador, checks
HTTP failures, and extracts only explicitly named archive members. Every download is
checked against a SHA-256 recorded in `fetch-data.lisp` before it reaches its final name,
and so is a kept file each time it is reused. A download digest mismatch signals
`digest-mismatch`: if a publisher has changed a file, the recorded results no longer
describe it. Existing extracted files are compared with the archive before reuse; a
mismatch is reported without replacing the file. Remove a corrupt or incomplete cached
file before downloading it again. MSLR must be acquired separately after accepting the publisher's agreement. The Online
Retail guide includes an additional step to load its pinned Excel reader.

The example tests run in separate Lisp processes so their temporary definitions and
environment changes do not affect the calling REPL. The child needs ASDF 3.3.7 or newer.
If a current ASDF was loaded only into the parent image, provide its source file before
running the tests:

```lisp
(load "examples/test-process.lisp")
(setf cl-gbdt-example-test-process:*asdf-bootstrap* #P"/path/to/asdf.lisp")
```

Alternatively set `CL_GBDT_ASDF_BOOTSTRAP` to that file. No extra setting is needed when
SBCL's normal startup already loads a current ASDF. If a local dependency is available
only through a custom ASDF searcher or registry function, load that system in the calling
REPL first; its registered source will then be available to the child. To run the tests:

```lisp
(load "examples/tests.lisp")
(load "examples/tests-fetch.lisp")
(load "examples/tests-fetch-safety.lisp")
(load "examples/tests-fetch-atomicity.lisp")
(load "examples/tests-ranking.lisp")
(load "examples/tests-kasandr.lisp")
(load "examples/tests-fraud.lisp")
(load "examples/tests-label-validation.lisp")
(load "examples/tests-mslr.lisp")
(load "examples/tests-toy.lisp")
(load "examples/tests-utility-safety.lisp")
(load "examples/tests-example-lifecycle.lisp")
(load "examples/tests-state.lisp")
(load "examples/tests-test-process.lisp")
(load "examples/tests-train-io.lisp")
```

CI runs every one of these through `tools/ci/run-example-tests.lisp`, which can also be
run from the repository root with
`ros run -- --non-interactive --load tools/ci/run-example-tests.lisp`.

The examples are reproducible within their stated dependencies, but are not a benchmark.
Randomized splits now use the specified xorshift32 generator in Common Lisp; they are
not the same splits previously generated by scikit-learn. Backend parameters differ and
are not tuned. See each example for its data source, method, results and limitations.
