# cl-gbdt

[![tests](https://github.com/masatoi/cl-gbdt/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/masatoi/cl-gbdt/actions/workflows/test.yml)
[![lint](https://github.com/masatoi/cl-gbdt/actions/workflows/lint.yml/badge.svg?branch=master)](https://github.com/masatoi/cl-gbdt/actions/workflows/lint.yml)

A Common Lisp library that wraps two gradient boosting decision tree
implementations -- [LightGBM](https://github.com/microsoft/LightGBM) and
[XGBoost](https://github.com/dmlc/xgboost) -- behind a single high-level API.

Full design rationale lives in
`docs/superpowers/specs/2026-08-02-cl-gbdt-design.md` (not checked in; see that
file if you have it locally).

## Status

**Functional.** Both backends implement all 13 generic functions of the unified API --
`make-dataset`, `dataset-num-rows`, `dataset-num-features`, `train`,
`update-one-iteration`, `predict`, `save-model`, `load-model`, `model-to-string`,
`feature-importance`, `evaluation`, `free-dataset` and `free-booster` -- against the real
LightGBM and XGBoost shared libraries, exercised by 946 functional assertions across 15 test
files (design doc section 12, layer 2), in addition to 649 assertions across 22 test files
that need no shared library at all (layer 1). `train` also returns a `training-report` as
its secondary value, and takes
`:early-stopping` to end a run once a watched metric stops improving -- see
[Training report](docs/user-guide/training.md#training-report). `make-dataset` and
`predict` also accept a `csr-matrix` wherever they accept a dense matrix -- see [Sparse
input](docs/user-guide/data-and-prediction.md#sparse-input-csr-matrices) -- and both also
take `:missing`, the value in the caller's data that means missing, gated on the
`:missing-value` capability that only XGBoost provides -- see [Missing
values](docs/user-guide/data-and-prediction.md#missing-values). `make-dataset` also
takes `:categorical-features`, the 0-based columns that hold categories rather than
quantities, gated on the `:categorical-features` capability that both backends provide
-- `predict` takes no such argument, the trained trees already carrying the category
sets they split on -- see [Categorical
features](docs/user-guide/data-and-prediction.md#categorical-features). `predict`
also returns the SHAPE the backend states for the result it just wrote as a second value --
a list of integers in `array-dimensions` order, or `NIL` where the backend states none --
gated on the `:prediction-shape` capability that both backends provide; XGBoost reads its
own `out_shape`/`out_dim` back from the library and LightGBM derives what it can, stating
`NIL` for `:leaf-index` -- see
[Prediction shape](docs/user-guide/data-and-prediction.md#prediction-shape). `train` also
takes `:objective`, a function that turns the current raw scores into a gradient and a
Hessian so a run boosts against the caller's own loss, gated on the `:custom-objective`
capability that both backends provide; the two libraries flatten that array in opposite
orders and the wrapper absorbs it, and on LightGBM `:objective` overrides any `objective`
in `:parameters` -- all five spellings that library honours -- forcing it to `"none"`,
since the library refuses the combination outright -- see [Custom
objective](docs/user-guide/custom-training.md#custom-objective). `train` also takes
`:evaluation`, a function called
once per dataset per iteration with that dataset's `predict :kind :normal` scores and the
dataset's index -- `0` the training set, `N+1` the Nth `:valid-sets` entry -- that returns a
metric name and a real or `NIL` value, gated on the `:custom-evaluation` capability that both
backends provide out of different lists, LightGBM probing it and XGBoost declaring it; the
values become their own report series, watchable by `:early-stopping` under the returned
name, appended after the library's own series rather than replacing them -- see [Custom
evaluation](docs/user-guide/custom-training.md#custom-evaluation). See [Usage](#usage)
below for a worked example.

Loading `cl-gbdt` itself still does not require either `liblightgbm.so` or
`libxgboost.so` to be installed -- see [Systems](#systems): a shared library is opened
only by an explicit `open-backend` call, from whichever backend system(s) you load on
top of the core. Each backend ships as **two** systems: `cl-gbdt/lightgbm` is that
backend's own API alone -- it opens and closes the library, builds datasets and boosters,
trains, predicts, saves and reloads a model, renders one as a string, reports feature
importance and evaluation metrics, and answers a dataset's own shape, and none of the
thirteen portable generic functions above are part of it -- while `cl-gbdt/lightgbm/unified`
adds their LightGBM methods, and core `cl-gbdt` with them. `cl-gbdt/xgboost` and
`cl-gbdt/xgboost/unified` divide the same way. Every example below that calls one of
those thirteen loads a `/unified` system; the few that do not are the ones
demonstrating what a Layer 1 system does and does not carry.

## Usage

A worked example first, then the details a caller moving between the two backends
needs. Every code block below was actually run to produce the output pasted beneath it
(SBCL via `ros run`, with `./tools/fetch-libs.sh`'s vendored libraries already present).

For every published symbol rather than a worked subset, see
[`docs/API-REFERENCE.md`](docs/API-REFERENCE.md) -- generated from the docstrings
`cl-gbdt`, `cl-gbdt/lightgbm` and `cl-gbdt/xgboost` export, one entry per symbol, so it
never drifts from the code the way hand-maintained prose can. This README remains what
that reference is not: the worked explanations, the differences between the two
backends, and the reasoning behind them.

For where every one of those symbols stands against the functional suite, see
[`docs/FUNCTIONAL-COVERAGE.md`](docs/FUNCTIONAL-COVERAGE.md) -- alongside
[`ffi-spec/BINDING-COVERAGE.md`](ffi-spec/BINDING-COVERAGE.md), which does the same for
the C API one layer below. A green `check-functional-coverage.lisp` means every symbol
has a recorded position -- named by the suite's own sources, or a row saying why
not -- not that its behavior has actually been exercised, and a row's honesty is
something only a reader can check, not the script.

### Quick start

Load the core system and one backend's `/unified` system -- `cl-gbdt/lightgbm` alone
carries no `cl-gbdt:train`; see [Two systems per
backend](docs/user-guide/backends.md#two-systems-per-backend) -- open the
backend, build a dataset from a
`double-float` matrix and its labels, train, predict, and free everything with
`with-dataset`/`with-booster` -- explicit resource management is this library's
documented first-class pattern; see their docstrings for why finalizers are only a
safety net, not something to rely on.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(let* ((backend (cl-gbdt:open-backend :lightgbm))
       (matrix (make-array '(8 2) :element-type 'double-float
                            :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0)
                                                 (0.0d0 2.0d0) (0.0d0 3.0d0)
                                                 (5.0d0 0.0d0) (5.0d0 1.0d0)
                                                 (5.0d0 2.0d0) (5.0d0 3.0d0))))
       (label (make-array 8 :element-type 'single-float
                             :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0))))
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset
                                   backend matrix
                                   :label label
                                   :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                  :verbose -1)))
    (format t "rows=~D features=~D~%"
            (cl-gbdt:dataset-num-rows dataset) (cl-gbdt:dataset-num-features dataset))
    (cl-gbdt:with-booster (booster (cl-gbdt:train
                                     backend dataset
                                     :num-rounds 10
                                     :parameters '(:objective "binary" :num-leaves 2
                                                    :min-data-in-leaf 1 :min-data-in-bin 1
                                                    :verbose -1)))
      (format t "predictions:~%~S~%" (cl-gbdt:predict booster matrix))))
  (cl-gbdt:close-backend backend)
  (format t "done~%"))
```

Output:

```
rows=8 features=2
predictions:
#2A((0.17926923885828328d0)
    (0.17926923885828328d0)
    (0.17926923885828328d0)
    (0.17926923885828328d0)
    (0.8207307611417167d0)
    (0.8207307611417167d0)
    (0.8207307611417167d0)
    (0.8207307611417167d0))
done
```

`with-booster` nests inside `with-dataset`, never the other way around: a booster holds
a strong reference to the dataset it was trained on, so `with-dataset`'s cleanup must run
after `with-booster`'s, and only this nesting guarantees that order. `close-backend` runs
last, outside both, since it closes the shared library both handles' pointers are backed
by. `label`, `weight` and `group` each accept any sequence, not only a `single-float`
array -- every element is coerced internally.

## Systems

| System | Purpose |
|---|---|
| `cl-gbdt` | Core: package, condition hierarchy, matrix marshalling, backend registry and `open-backend` protocol, the unified API's generic functions -- no methods, and no shared library required to load it |
| `cl-gbdt/lightgbm` | **Layer 1 for LightGBM, and nothing above it.** Library discovery and the `%`-wrappers (`src/lightgbm/native.lisp`) over the generated CFFI bindings (`src/lightgbm/c-api.lisp`), plus the backend's CLOS types and the `initialize-backend`/`shutdown-backend` pair that opens and closes the shared library (`src/lightgbm/classes.lisp`), plus the fourteen finished operations a caller invokes -- `create-dataset`, `create-booster`, `update-one-iteration`, `predict`, `free-dataset`, `free-booster`, `save-model`, `load-model`, `model-to-string`, `feature-importance`, `evaluation`, `dataset-num-rows`, `dataset-num-features`, `create-dataset-from-file` (`src/lightgbm/api.lisp`) -- published together by `src/lightgbm/all.lisp`. Carries none of the 13 unified-API methods, and does not define the `cl-gbdt` package |
| `cl-gbdt/lightgbm/unified` | That plus all 13 unified-API methods (`src/lightgbm/protocol.lisp`), aggregated by `src/lightgbm/unified.lisp`, which also depends on core `cl-gbdt` so the `cl-gbdt:` spelling exists. **This is the system a caller of `cl-gbdt:train` loads** |
| `cl-gbdt/xgboost` | Layer 1 for XGBoost, exactly as above: `src/xgboost/native.lisp` and `src/xgboost/c-api.lisp`, plus `src/xgboost/classes.lisp` and `src/xgboost/api.lisp` -- the latter holding the same fourteen operations and, additionally, `slice-model`, an XGBoost-only operation that builds a booster handle -- published by `src/xgboost/all.lisp` |
| `cl-gbdt/xgboost/unified` | That plus all 13 unified-API methods (`src/xgboost/protocol.lisp`), aggregated by `src/xgboost/unified.lisp`, core `cl-gbdt` included |
| `cl-gbdt/regen` | The binding emitter (`src/regen/`). Development-only -- never appears in `cl-gbdt`'s, `cl-gbdt/lightgbm`'s, or `cl-gbdt/xgboost`'s dependency graph, so an ordinary user never needs it or its dependencies (`alexandria`, `com.inuoe.jzon`). `cffi/c2ffi` is *not* one of them -- it is a dependency of `tools/regen.lisp`, which quickloads it directly, not of the `cl-gbdt/regen` system itself |
| `cl-gbdt/tests` | The Rove test suite |

**Which one to load.** `cl-gbdt/lightgbm` gives you the LightGBM-specific API and opens the
library; `cl-gbdt:train` and the other twelve portable generic functions are not part of it.
`cl-gbdt/lightgbm/unified` gives you that *plus* the portable API, and is what the [quick
start](#quick-start) uses. Loading both backends' `/unified` systems is how a program drives
one portable API over both libraries. Naming core `cl-gbdt` alongside a `/unified` system, as
the examples above do, is belt-and-braces: `/unified` already depends on it, and the examples
name it because they use symbols from it directly.

**Layer 1 alone trains, predicts, persists and reports.** A program that loads
`cl-gbdt/lightgbm` or `cl-gbdt/xgboost` and nothing else builds a dataset, builds a booster
over it, advances it one iteration at a time, scores with it, saves the model and reloads
it, renders it as a string, reports its feature importance and evaluation metrics, asks the
dataset its own shape, and frees both -- `create-dataset`, `create-booster`,
`update-one-iteration`, `predict`, `free-dataset`, `free-booster`, `save-model`,
`load-model`, `model-to-string`, `feature-importance`, `evaluation`, `dataset-num-rows`,
`dataset-num-features`, with a worked example of the first six in [Two systems per
backend](docs/user-guide/backends.md#two-systems-per-backend) and the rest exercised by
`tests/functional/{lightgbm,xgboost}-standalone.lisp` -- plus a fourteenth,
`create-dataset-from-file`, which builds the dataset straight from a file instead (see [File
input](docs/user-guide/file-input.md#file-input)), exercised by the same two files. What such
a program still cannot
do is `cl-gbdt:train`'s own concepts: no training report, no early stopping and no
`:objective` or `:evaluation` callback, none of which have a Layer 1 counterpart to be.

Each Layer 1 backend system (`cl-gbdt/lightgbm` depends on `cl-gbdt/src/lightgbm/all`,
`cl-gbdt/xgboost` on `cl-gbdt/src/xgboost/all`) depends on `cffi`, its own generated
`c-api.lisp`, and the individual `cl-gbdt/src/*` leaf systems it needs -- `handle`,
`backend`, `conditions`, `parameters`, `data`, `library`, `foreign`, and the `config/`
renderers -- but **not** on `cl-gbdt/src/protocol`, and not on the aggregate `cl-gbdt`
system/package that re-exports all of those under one name (see `src/all.lisp`). That
absence is what `tools/ci/check-layer-separation.lisp` asserts on every CI run: it reads each
Layer 1 file's `uiop:define-package` form, walks the whole closure itself rather than asking a
loaded image, and fails the build if `cl-gbdt/src/protocol`, the training report, any of the
three files under `src/training/`, or the bare name `cl-gbdt` turns up in it. The `/unified`
systems add exactly those
edges, by way of `src/<backend>/unified.lisp`. Each backend is loadable independently of the other, so the
library works on a machine where only one of the two shared libraries is present.

## License

MIT; see `LICENSE`.

The upstream C API headers vendored under `ffi-spec/` remain under their own terms --
LightGBM's are MIT, XGBoost's are Apache 2.0 -- and each retains its upstream copyright
notice, and both upstream licence texts are vendored verbatim under `LICENSES/` so that
recipients receive a copy rather than a link. `THIRD-PARTY-LICENSES.md` records what is
redistributed and under what. cl-gbdt does
not bundle either backend's compiled library; `tools/fetch-libs.sh` fetches those into the
git-ignored `vendor/` directory for development only.
