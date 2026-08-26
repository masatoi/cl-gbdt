# cl-gbdt

[![tests](https://github.com/masatoi/cl-gbdt/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/masatoi/cl-gbdt/actions/workflows/test.yml)
[![lint](https://github.com/masatoi/cl-gbdt/actions/workflows/lint.yml/badge.svg?branch=master)](https://github.com/masatoi/cl-gbdt/actions/workflows/lint.yml)

A Common Lisp library that wraps two gradient boosting decision tree
implementations -- [LightGBM](https://github.com/lightgbm-org/LightGBM) and
[XGBoost](https://github.com/dmlc/xgboost) -- behind a single high-level API.

The same `cl-gbdt:train` and `cl-gbdt:predict` call drives either library, so a program can
switch backends without being rewritten; where the two genuinely disagree, this library says
so rather than emulating one on the other. Each backend's own package is published as well,
for the operations only that library has.

The layering, and the reasoning behind every decision in it, are recorded in
[`docs/cl-gbdt-layered-api-implementation-policy.md`](docs/cl-gbdt-layered-api-implementation-policy.md).

## Status

**Version `0.1.0`, the first tagged release.** Install it from a git checkout (see
[Installation](#installation)); there is no Quicklisp release to depend on yet.

`0.x` means the published surface can still change in a minor bump. What is published is a
compatibility obligation from the moment it is published, per policy section 14 -- that has
been true since before there was a version number and has not changed -- but until `1.0.0` the
version number does not promise that a minor release keeps it. A breaking change will say so in
its release notes.

**Functional.** Both backends implement all 13 generic functions of the unified API --
`make-dataset`, `dataset-num-rows`, `dataset-num-features`, `train`, `update-one-iteration`,
`predict`, `save-model`, `load-model`, `model-to-string`, `feature-importance`, `evaluation`,
`free-dataset` and `free-booster` -- against the real LightGBM and XGBoost shared libraries,
exercised on every CI run by a functional suite that calls those libraries for real, on top of
a suite that needs no shared library at all.

Where each published symbol stands against that functional suite is recorded, one row per
symbol, in [`docs/FUNCTIONAL-COVERAGE.md`](docs/FUNCTIONAL-COVERAGE.md) -- and
[`ffi-spec/BINDING-COVERAGE.md`](ffi-spec/BINDING-COVERAGE.md) does the same for the C API one
layer below. Both are checked by CI. Neither states a count, and neither does this file:
coverage here is guaranteed by classification, not by a number.

Core `cl-gbdt` loads, and is tested, with neither `liblightgbm.so` nor `libxgboost.so`
present. A shared library is opened only by an explicit `open-backend` call, from whichever
backend system you loaded on top of the core -- see [Systems](#systems).

## Supported environments

Two different claims, kept in separate columns: what CI runs on every push, and what was
measured by hand once and recorded rather than re-run. The last row is neither -- it is a
requirement, and is marked as one.

| | CI-verified | Also measured |
|---|---|---|
| Common Lisp | SBCL, through Roswell | -- |
| Linux | x86_64, aarch64 | -- |
| macOS | aarch64 | -- |
| Windows | not tested | -- |
| LightGBM | 4.0.0, 4.7.0 (pinned) | 3.0.0, the inferred lower bound, has no aarch64 wheel on PyPI and stays untested there |
| XGBoost | 2.0.0, 3.4.1 (pinned) | **1.7.0 fails**: 105 of 106 assertions pass (measured 2026-08-06, when the suite had 106), the ranking round trip does not |
| ASDF | no version verified: CI runs `ros install asdf`, taking whatever the current release is that day | **Requirement, not a measurement: 3.3.7 or newer** -- see below |

The pinned versions are the ones `ffi-spec/VERSIONS` names and `./tools/fetch-libs.sh`
downloads, and are what the suite runs against on all three platforms. The lower version in
each library's row is the low end of the *verified* range `src/version.lisp` records, run by a
separate Linux x86_64 job on push and weekly rather than on every pull request.

**XGBoost 1.7.0 is a real, measured incompatibility, not a gap in coverage.** Every assertion
a header comparison could reach passes; one behavioural assertion about `rank:pairwise` does
not, which is what moved `*xgboost-version-range*`'s lower bound to 2.0.0. Which assertion,
and why header comparison cannot see it, is in [Backend
differences](docs/user-guide/backend-differences.md#where-the-two-backends-genuinely-differ).
`open-backend` warns with `untested-backend-version` when the loaded XGBoost falls outside the
recorded compatible range; LightGBM's C API has no version entry point, so nothing is checked
there and no such warning is ever signalled.

**ASDF 3.3.7 is a hard requirement, not a recommendation.** Roswell ships 3.3.1, whose
`package-inferred-system` dependency scanner does not know the `:local-nicknames` clause and
dies with `:LOCAL-NICKNAMES fell through ECASE expression` on any system reaching
`src/regen/emit.lisp`. `ros install asdf` below is the fix, and it is needed once per machine.

## Installation

Every command below assumes [Roswell](https://github.com/roswell/roswell), which is how this
project runs SBCL everywhere, CI included; `ros` is its command, and installing it is step
zero. `./tools/fetch-libs.sh` additionally needs `python3`, `pip` and `unzip` on `PATH` --
it takes the shared libraries out of the backends' PyPI wheels.

```bash
git clone https://github.com/masatoi/cl-gbdt ~/.roswell/local-projects/cl-gbdt
ros install asdf
cd ~/.roswell/local-projects/cl-gbdt
./tools/fetch-libs.sh
```

Three things happen there, and the quick start below assumes all three:

1. **Clone somewhere ASDF already looks.** `~/.roswell/local-projects/` is that place under
   Roswell; any directory on `asdf:*central-registry*` or under a configured source registry
   works as well. Nothing else makes the systems findable.
2. **`ros install asdf`** replaces Roswell's bundled 3.3.1 -- see [Supported
   environments](#supported-environments) for what breaks without it.
3. **`./tools/fetch-libs.sh`** downloads LightGBM's and XGBoost's prebuilt shared libraries
   into the git-ignored `vendor/` directory, via `python3 -m pip download` and `unzip` of the
   resulting wheels. This repository bundles neither library. If you already
   have them installed, skip this step and point `CL_GBDT_LIGHTGBM_LIB` /
   `CL_GBDT_XGBOOST_LIB`, or `open-backend`'s `:path`, at them instead -- see [Finding the
   shared library](docs/user-guide/backends.md#finding-the-shared-library) for the full search
   order.

Only the backend you actually intend to open needs its library present; loading core
`cl-gbdt` needs neither.

## Quick start

Load the core system and one backend's `/unified` system -- `cl-gbdt/lightgbm` alone carries
no `cl-gbdt:train`; see [Systems](#systems) -- open the backend, build a dataset from a
`double-float` matrix and its labels, train, predict, and free everything.

`with-dataset` and `with-booster` free their handles on the way out, and
[`with-backend`](docs/user-guide/threads.md#with-backend) does the same for the backend around
both of them: explicit resource management is this library's documented first-class pattern,
and the finalizers behind it are a safety net, not something to rely on.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(cl-gbdt:with-backend (backend (cl-gbdt:open-backend :lightgbm))
  (let ((matrix (make-array '(8 2) :element-type 'double-float
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
                                                      :min-data-in-leaf 1
                                                      :min-data-in-bin 1
                                                      :verbose -1)))
        (format t "predictions:~%~S~%" (cl-gbdt:predict booster matrix))))))
(format t "done~%")
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

Three things that block a first run if they are got wrong:

- **`with-booster` nests inside `with-dataset`, never the other way around.** A booster holds
  a strong reference to the dataset it was trained on, so `with-dataset`'s cleanup must run
  after `with-booster`'s, and only this nesting guarantees that order.
- **`with-backend` nests outside both `with-dataset` and `with-booster`, never inside them.**
  `close-backend` calls `cffi:close-foreign-library`, which both handles' pointers are backed
  by, so it must run last; `with-backend` closes its backend on every exit from its body,
  normal or not, the same way `with-dataset` and `with-booster` already guarantee for their
  own handles.
- **`:parameters` is each library's own vocabulary**, keyword-spelled: `:min-data-in-leaf`
  becomes `min_data_in_leaf`, `t`/`nil` become `"true"`/`"false"`. There is no allowlist, so
  a backend-specific key works unchanged -- see
  [Parameters](docs/user-guide/backends.md#parameters).

`label`, `weight` and `group` each accept any sequence, not only a `single-float` array --
every element is coerced internally.

## Systems

| System | Purpose |
|---|---|
| `cl-gbdt` | Core: the package, the condition hierarchy, matrix marshalling, the backend registry and `open-backend` protocol, and the unified API's generic functions -- no methods, and no shared library needed to load it |
| `cl-gbdt/lightgbm` | **Layer 1 for LightGBM alone.** Opens and closes the library and publishes LightGBM's own fourteen operations, from `create-dataset` through `create-dataset-from-file`, plus `booster-eval` and `booster-eval-names`. Carries none of the 13 unified-API methods, and does not define the `cl-gbdt` package |
| `cl-gbdt/lightgbm/unified` | That, plus LightGBM's methods on all 13 unified generics, plus core `cl-gbdt`. **This is what a caller of `cl-gbdt:train` loads** |
| `cl-gbdt/xgboost` | Layer 1 for XGBoost, dividing identically -- the same fourteen operations under the same names in a different package, plus `slice-model`, `evaluate-one-iteration` and `booster-boosted-rounds` |
| `cl-gbdt/xgboost/unified` | That, plus XGBoost's methods on all 13 unified generics, and core `cl-gbdt` with them |
| `cl-gbdt/tests` | The Rove suite that needs no shared library |
| `cl-gbdt/tests/functional` | The Rove suite that calls the real libraries; needs `./tools/fetch-libs.sh` first |
| `cl-gbdt/regen` | The binding emitter (`src/regen/`). Development-only; never in any other system's dependency graph |
| `cl-gbdt/docgen` | The API-reference emitter (`src/docgen/`). Development-only, on the same terms |

**Which one to load.** A `/unified` system for the portable API; both backends' `/unified`
systems to drive both libraries through it from one program. A bare Layer 1 system if you
want one library's own API and nothing else -- that is enough on its own to train, predict,
persist and report, but not to use `cl-gbdt:train`'s training report, early stopping or
callbacks, which have no Layer 1 counterpart. Calling a portable generic with only Layer 1
loaded signals `backend-methods-not-loaded`, naming the system to load.

What each Layer 1 system publishes, what it deliberately does not, and a worked example that
never defines the `cl-gbdt` package at all are in [Two systems per
backend](docs/user-guide/backends.md#two-systems-per-backend) and [Backend-specific
packages](docs/user-guide/backends.md#backend-specific-packages). CI enforces the split:
`tools/ci/check-layer-separation.lisp` fails the build if a Layer 1 file's dependency closure
ever reaches the unified protocol, the training files, or the bare `cl-gbdt` system.

## Features

Everything below is on the unified API and works on both backends unless the table says
otherwise. **The Capability column names the keyword `backend-supports-p` answers for that
feature, or `--` where the feature has no registered capability at all** -- naming one does
not by itself mean an operation refuses on it, only that the question can be asked and
answered about the shared library that was actually loaded (see [Asking a backend what it can
do](docs/user-guide/backends.md#asking-a-backend-what-it-can-do)). The guide column is where
each feature is explained, with the worked example and the measurement behind it.

| Feature | Capability | LightGBM | XGBoost | Guide |
|---|---|---|---|---|
| Training report, `:early-stopping` | `:evaluation-history`, `:early-stopping` | yes | yes | [Training report](docs/user-guide/training.md#training-report), [Stopping early](docs/user-guide/training.md#stopping-early-early-stopping) |
| `:num-iteration :best` | -- | `predict`, `save-model`, `model-to-string` | `predict` only | [`:num-iteration :best`](docs/user-guide/training.md#num-iteration-best) |
| Sparse input (`csr-matrix`) | `:sparse-input` | yes, all four `predict` kinds | yes, `:normal` and `:raw` only | [Sparse input](docs/user-guide/data-and-prediction.md#sparse-input-csr-matrices) |
| `:missing` sentinel | `:missing-value` | **no** -- no such key in its C API | yes | [Missing values](docs/user-guide/data-and-prediction.md#missing-values) |
| `:categorical-features` | `:categorical-features` | yes | yes, `tree_method` `hist` or `approx` | [Categorical features](docs/user-guide/data-and-prediction.md#categorical-features) |
| `predict`'s shape, second value | `:prediction-shape` | derived; `NIL` for `:leaf-index` | stated verbatim by the library | [Prediction shape](docs/user-guide/data-and-prediction.md#prediction-shape) |
| `train`'s `:objective` | `:custom-objective` | yes | yes | [Custom objective](docs/user-guide/custom-training.md#custom-objective) |
| `train`'s `:evaluation` | `:custom-evaluation` | yes | yes | [Custom evaluation](docs/user-guide/custom-training.md#custom-evaluation) |
| `create-dataset-from-file` | -- (**Layer 1 only**) | format inferred from the file | format required positionally | [File input](docs/user-guide/file-input.md) |

One thing worth knowing before you reach the guide, for each:

- **Training report** -- one `training-series` per metric per dataset, values oldest-first,
  indexed the way `evaluation` indexes them. Recording roughly doubles LightGBM's `train`
  time; `:record-history nil` turns it off.
- **Sparse input** -- an entry a `csr-matrix` does not store is `0.0` to LightGBM and
  *missing* to XGBoost, which changes the trained model silently rather than erroring.
- **`:missing`** -- LightGBM signals `capability-unavailable` for any non-`NIL` value, a
  `NaN` included; XGBoost compares the sentinel at single precision.
- **`:categorical-features`** -- `predict` takes no such argument on either backend; the
  trained trees already carry the category sets they split on.
- **Prediction shape** -- the first value is unchanged, so a caller who ignores the second
  sees exactly the behaviour that predates the feature.
- **`:objective`** -- on LightGBM it overrides any `objective` in `:parameters`, all five
  spellings, forcing it to `"none"`; the library refuses the combination outright.
- **`:evaluation`** -- its values become their own report series, watchable by
  `:early-stopping`; it requires `:record-history t` and one metric name per dataset index.
- **File input** -- no unified form and no capability to ask about: XGBoost's format
  argument, declared wrong, can end the process in a thread no Lisp handler can reach.

## Where the backends differ

The differences most likely to break a program moved between backends. The full catalogue,
every row of it, is in [Backend
differences](docs/user-guide/backend-differences.md#where-the-two-backends-genuinely-differ),
which links each row on to the guide that measures it.

| | LightGBM | XGBoost |
|---|---|---|
| `make-dataset`'s `:reference` | Aligns the new dataset's bin mapper to an existing one's, and is required for a `train` `:valid-sets` entry | `unsupported-argument` -- no bin-mapper concept |
| `make-dataset`'s `:parameters` | Configures the dataset's own binning (`max_bin` and friends) | `unsupported-argument` -- the creation config recognizes none of those keys and silently ignores what it does not know |
| `:missing`, the missing-value sentinel | `capability-unavailable` for any non-`NIL` value -- the C API has no such key | Supported; the sentinel is a key the creation and prediction configs already read |
| `predict`'s `:kind` on a `csr-matrix` | All four kinds | `:normal` and `:raw` only -- its CSR entry point is the *inplace* one, which refuses `:contrib` and `:leaf-index` |
| `save-model`'s `:num-iteration` | Limits how many trees are saved | `unsupported-argument` -- `XGBoosterSaveModel` always saves every round |
| Model slicing | No counterpart at all; `(backend-supports-p backend :model-slicing)` is `nil` | `cl-gbdt/xgboost:slice-model`, Layer 1 and XGBoost-only: a new booster over a half-open range of the parent's layers |
| `backend-version` | Always `nil` -- the C API has no version entry point | A `"MAJOR.MINOR.PATCH"` string, e.g. `"3.4.1"` |

## Known limitations

- **Windows is not tested.** CI covers Linux x86_64, Linux aarch64 and macOS aarch64. Nothing
  in the code is known to be Windows-specific; nothing has been run there either.
- **Only SBCL is tested, and parts of the code are SBCL-only.** `sb-sys:with-pinned-objects`
  pins every array handed to either library -- in `src/data.lisp`, which both backends share,
  and in each backend's own `native.lisp` -- and `sb-ext:native-namestring` composes every
  path *either* library is given: XGBoost's file-input and model-persistence paths, and
  LightGBM's `create-dataset-from-file`, `save-model` and `load-model`. Neither construct is
  portable Common Lisp, and no other implementation has been tried.
- **A file being written while it is read is accepted silently.** Both libraries take a file
  truncated mid-row as though it were complete -- no error, a dataset built from whatever
  whole rows preceded the cut -- and the window between this wrapper classifying a file and
  the library reading it cannot be closed from Lisp. `create-dataset-from-file` expects a
  finished file.
- **A FIFO or device file is not detected.** ANSI Common Lisp cannot portably ask whether a
  path names a named pipe rather than a regular file, so nothing refuses one; a FIFO with no
  writer blocks indefinitely inside the read that classifies it, with no diagnostic. Both this
  and the point above are detailed in [File input](docs/user-guide/file-input.md#file-input).
- **No Quicklisp release.** Installation is a git checkout, as above.

## Documentation

Every document this repository tracks, and the question each one answers. The licence texts
themselves -- `LICENSE` and `LICENSES/` -- are named under [License](#license) below.

**Every code block in the seven guides below, and the quick start above, was actually run to
produce the output pasted beneath it** -- SBCL via `ros run`, with `./tools/fetch-libs.sh`'s
vendored libraries present. An `Output:` block is a transcript, not an illustration.

Outside [Data and prediction](docs/user-guide/data-and-prediction.md), each of those blocks
also **stands alone**: it loads the systems it needs and defines its own fixtures, so it can
be pasted into a fresh REPL as it stands -- measured, by extracting all twenty and running
each in a fresh image. Data and prediction's blocks instead build on each other: nine of its
fourteen reuse a matrix, a label or a helper defined in an earlier block of that same guide,
so read its blocks in order, or evaluate the earlier ones first.

### Using the library

| Document | Answers |
|---|---|
| [`docs/user-guide/backends.md`](docs/user-guide/backends.md) | How each backend splits into two systems, what its own package publishes, how the shared library is found, how `:parameters` is rendered, and how to ask a backend what it supports |
| [`docs/user-guide/training.md`](docs/user-guide/training.md) | What `train` returns as its training report, how `:early-stopping` uses it, and what `:num-iteration :best` and `:record-history` do |
| [`docs/user-guide/data-and-prediction.md`](docs/user-guide/data-and-prediction.md) | Sparse input, missing values, categorical features, and the shape `predict` reports |
| [`docs/user-guide/custom-training.md`](docs/user-guide/custom-training.md) | `train`'s `:objective` and `:evaluation` callbacks -- boosting against your own loss and recording your own metrics |
| [`docs/user-guide/file-input.md`](docs/user-guide/file-input.md) | `create-dataset-from-file` on each backend, and everything measured about getting its format argument wrong |
| [`docs/user-guide/backend-differences.md`](docs/user-guide/backend-differences.md) | The full catalogue of where LightGBM and XGBoost genuinely differ, plus the version matrix behind [Supported environments](#supported-environments) |
| [`docs/user-guide/threads.md`](docs/user-guide/threads.md) | What is safe to do from more than one thread, what is not, and why cl-gbdt adds no locking of its own |
| [`docs/API-REFERENCE.md`](docs/API-REFERENCE.md) | Every symbol `cl-gbdt`, `cl-gbdt/lightgbm` and `cl-gbdt/xgboost` export, one entry each. Generated from the docstrings, so it cannot drift from the code -- never hand-edit it |

### Working on the library

| Document | Answers |
|---|---|
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to [run the tests](CONTRIBUTING.md#running-the-tests) and the [functional tests](CONTRIBUTING.md#running-the-functional-tests), what [CI](CONTRIBUTING.md#continuous-integration) checks, and how to regenerate the [bindings](CONTRIBUTING.md#regenerating-the-bindings) and the [API reference](CONTRIBUTING.md#regenerating-the-api-reference) |
| [`docs/cl-gbdt-layered-api-implementation-policy.md`](docs/cl-gbdt-layered-api-implementation-policy.md) | Why the API is layered the way it is: the capability model, resource safety, package boundaries, the compatibility obligation, and what is deliberately out of scope |
| [`docs/FUNCTIONAL-COVERAGE.md`](docs/FUNCTIONAL-COVERAGE.md) | Where every published symbol stands against the functional suite -- covered, unproven, or exempt with a stated reason. Checked by CI |
| [`ffi-spec/BINDING-COVERAGE.md`](ffi-spec/BINDING-COVERAGE.md) | The same, one layer down: every generated C binding as wrapped, planned or excluded, with the reason. Checked by CI |
| [`ffi-spec/ABI-BLACKLIST.md`](ffi-spec/ABI-BLACKLIST.md) | Which generated C bindings must never be called -- they changed meaning upstream while keeping their names, so symbol probing cannot catch them -- why, and what to call instead. Checked by CI |
| [`CLAUDE.md`](CLAUDE.md), [`prompts/`](prompts/) | How AI agents are expected to work in this repository -- the REPL-driven loop, the tool split, and the gates |
| [`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md) | What is redistributed here and under whose terms |

## License

MIT; see [`LICENSE`](LICENSE).

The upstream C API headers vendored under `ffi-spec/` remain under their own terms --
LightGBM's are MIT, XGBoost's are Apache 2.0 -- and each retains its upstream copyright
notice; both upstream licence texts are vendored verbatim under `LICENSES/` so that
recipients receive a copy rather than a link.
[`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md) records what is redistributed and under
what. cl-gbdt does not bundle either backend's compiled library; `tools/fetch-libs.sh` fetches
those into the git-ignored `vendor/` directory for development only.
