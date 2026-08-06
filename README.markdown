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

**Functional.** Both backends implement all 12 generic functions of the unified API --
`make-dataset`, `dataset-num-rows`, `dataset-num-features`, `train`,
`update-one-iteration`, `predict`, `save-model`, `load-model`, `model-to-string`,
`feature-importance`, `free-dataset` and `free-booster` -- against the real LightGBM and
XGBoost shared libraries, exercised by 103 functional assertions (design doc section 12,
layer 2), in addition to 204 assertions that need no shared library at all (layer 1).
See [Usage](#usage) below for a worked example.

Loading `cl-gbdt` itself still does not require either `liblightgbm.so` or
`libxgboost.so` to be installed -- see [Systems](#systems): a shared library is opened
only by an explicit `open-backend` call, from whichever backend system(s) you load on
top of the core.

## Usage

A worked example first, then the details a caller moving between the two backends
needs. Every code block below was actually run to produce the output pasted beneath it
(SBCL via `ros run`, with `./tools/fetch-libs.sh`'s vendored libraries already present).

### Quick start

Load the core system and one backend system, open the backend, build a dataset from a
`double-float` matrix and its labels, train, predict, and free everything with
`with-dataset`/`with-booster` -- explicit resource management is this library's
documented first-class pattern; see their docstrings for why finalizers are only a
safety net, not something to rely on.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm) :silent t)

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

### Two backends, two ASDF systems

`cl-gbdt/lightgbm` and `cl-gbdt/xgboost` are separate systems (`cl-gbdt.asd`). Loading
one does not load the other, and neither is a dependency of core `cl-gbdt`; load
whichever one (or both) matches the shared library you have.

Loading only a backend system, without the core `cl-gbdt` system, attaches its methods
to the generic functions but does not define the `cl-gbdt` package itself:

```lisp
(ql:quickload :cl-gbdt/lightgbm :silent t)
(format t "~S~%" (find-package :cl-gbdt))
```

Output:

```
NIL
```

`make-dataset` and the rest are still callable, package-qualified as
`cl-gbdt/src/protocol:make-dataset`, but the friendly `cl-gbdt:` spelling needs the core
system loaded too -- as the quick start above does, loading both at once.

### Finding the shared library

`open-backend`'s `:path`, when supplied, wins outright over everything else. Otherwise,
in order: `CL_GBDT_LIGHTGBM_LIB` / `CL_GBDT_XGBOOST_LIB`, then the vendored directory
`./tools/fetch-libs.sh` writes to (`vendor/lightgbm/lib/` / `vendor/xgboost/lib/`), then
CFFI's own system library search.

**A set-but-nonexistent environment variable is an error, not a fall-through to the next
source** -- a typo in an override that silently loads a different library would be worse
than a failure:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm) :silent t)
(handler-case (cl-gbdt:open-backend :lightgbm)
  (error (c) (format t "SIGNALED: ~A: ~A~%" (type-of c) c)))
```

Run with `CL_GBDT_LIGHTGBM_LIB=/nonexistent/lib_lightgbm.so` in the environment, this
prints:

```
SIGNALED: BACKEND-LIBRARY-NOT-FOUND: Shared library for LIGHTGBM not found. Searched:
  /nonexistent/lib_lightgbm.so
```

### Parameters

`make-dataset`'s and `train`'s `:parameters` is a plist. Each keyword is downcased with
dashes turned into underscores; each value is stringified and passed through untouched,
so a backend-specific key works with no per-key allowlist:

```lisp
(ql:quickload :cl-gbdt :silent t)
(cl-gbdt:normalize-parameters
 (list :num-leaves 31 :learning-rate 0.05d0 :verbose -1 :feature-fraction 1/3
       :use-two-round-loading t :early-stopping-round nil))
```

Output:

```
(("num_leaves" . "31") ("learning_rate" . "0.05") ("verbose" . "-1")
 ("feature_fraction" . "0.3333333333333333") ("use_two_round_loading" . "true")
 ("early_stopping_round" . "false"))
```

`t`/`nil` become LightGBM's own spelling for a boolean, `"true"`/`"false"`; a ratio
prints as a decimal, not `"1/3"`; a string value passes through unquoted -- see
`:objective "binary:logistic"` below, whose colon survives intact.

### Where the two backends genuinely differ

A caller moving code from one backend to the other needs this in one place -- checked
directly against both backends' source, not only the differences the design doc calls
out first:

| | LightGBM | XGBoost |
|---|---|---|
| `make-dataset`'s `:reference` | Aligns the new dataset's bin mapper to an existing one's (required for a `train` `:valid-sets` entry) | Signals `unsupported-argument` -- no bin-mapper concept |
| `make-dataset`'s `:group` | Attaches ranking group sizes | Signals `unsupported-argument` -- not yet wired up on this backend |
| `make-dataset`'s `:parameters` | Configures the dataset's own binning (`max_bin` and friends) | **Silently accepted and ignored.** Unlike `:reference`/`:group` above, this does not signal: XGBoost's hyperparameters are all booster-level, and an empty `:parameters` is the overwhelmingly common call, so nothing changes silently underneath a caller who never passes one |
| `update-one-iteration`'s return value | `nil` once an iteration produces no further split -- a real signal | Always `t` after a successful call; XGBoost's booster protocol has no equivalent signal |
| `save-model`'s `:num-iteration` | Limits how many trees are saved | Signals `unsupported-argument` -- `XGBoosterSaveModel` always saves every round |
| `model-to-string`'s `:num-iteration` | Limits the rounds serialized | Signals `unsupported-argument` -- no iteration-limited variant exists |
| `feature-importance`'s `:num-iteration` | Limits the importance calculation | Signals `unsupported-argument` -- no iteration-limited variant exists |
| `feature-importance`'s result shape | Always one number per feature | Signals `unsupported-argument` instead of returning a result when the model reports a multi-dimensional score shape -- a `gblinear` booster's importance on a multi-class model, whose scores are a per-class matrix with no single-value reduction this backend will invent |
| `backend-version` | Always `nil` -- LightGBM's C API has no version entry point | A `"MAJOR.MINOR.PATCH"` string, e.g. `"3.3.0"` |

Run together against both backends:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm :cl-gbdt/xgboost) :silent t)

(defparameter *matrix*
  (make-array '(4 2) :element-type 'double-float
                      :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0)
                                           (5.0d0 0.0d0) (5.0d0 1.0d0))))
(defparameter *label*
  (make-array 4 :element-type 'single-float :initial-contents '(0.0 0.0 1.0 1.0)))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (format t "LightGBM backend-version: ~S~%" (cl-gbdt:backend-version lgbm))
  (format t "XGBoost  backend-version: ~S~%" (cl-gbdt:backend-version xgb))

  (cl-gbdt:with-dataset (lgbm-ds1 (cl-gbdt:make-dataset lgbm *matrix* :label *label*
                                     :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                    :verbose -1)))
    (cl-gbdt:with-dataset (lgbm-ds2 (cl-gbdt:make-dataset lgbm *matrix* :label *label*
                                       :reference lgbm-ds1
                                       :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                      :verbose -1)))
      (format t "LightGBM :reference accepted; aligned dataset has ~D rows~%"
              (cl-gbdt:dataset-num-rows lgbm-ds2))))

  (cl-gbdt:with-dataset (xgb-ds1 (cl-gbdt:make-dataset xgb *matrix* :label *label*))
    (handler-case (cl-gbdt:make-dataset xgb *matrix* :label *label* :reference xgb-ds1)
      (error (c) (format t "XGBoost  :reference SIGNALED ~A: ~A~%" (type-of c) c)))
    (handler-case (cl-gbdt:make-dataset xgb *matrix* :label *label* :group '(2 2))
      (error (c) (format t "XGBoost  :group SIGNALED ~A: ~A~%" (type-of c) c)))
    (cl-gbdt:with-dataset (xgb-ds-ignored (cl-gbdt:make-dataset xgb *matrix* :label *label*
                                             :parameters '(:max-bin 3)))
      (format t "XGBoost  :parameters silently accepted, ~D rows~%"
              (cl-gbdt:dataset-num-rows xgb-ds-ignored)))

    (cl-gbdt:with-booster (xgb-booster (cl-gbdt:train xgb xgb-ds1 :num-rounds 1
                                          :parameters '(:objective "binary:logistic"
                                                         :max-depth 2 :verbosity 0)))
      (format t "XGBoost  update-one-iteration => ~S~%"
              (cl-gbdt:update-one-iteration xgb-booster))
      (dolist (call (list (lambda () (cl-gbdt:save-model xgb-booster "/tmp/m.json"
                                                           :num-iteration 1))
                           (lambda () (cl-gbdt:model-to-string xgb-booster :num-iteration 1))
                           (lambda () (cl-gbdt:feature-importance xgb-booster :num-iteration 1))))
        (handler-case (funcall call)
          (error (c) (format t "SIGNALED ~A: ~A~%" (type-of c) c))))))

  (cl-gbdt:with-dataset (lgbm-ds (cl-gbdt:make-dataset lgbm *matrix* :label *label*
                                    :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                   :verbose -1)))
    (cl-gbdt:with-booster (lgbm-booster (cl-gbdt:train lgbm lgbm-ds :num-rounds 1
                                           :parameters '(:objective "binary" :num-leaves 2
                                                          :min-data-in-leaf 1 :min-data-in-bin 1
                                                          :verbose -1)))
      (format t "LightGBM update-one-iteration => ~S~%"
              (cl-gbdt:update-one-iteration lgbm-booster))))

  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM backend-version: NIL
XGBoost  backend-version: "3.3.0"
LightGBM :reference accepted; aligned dataset has 4 rows
XGBoost  :reference SIGNALED UNSUPPORTED-ARGUMENT: make-dataset's :reference is not supported by XGBOOST: XGBoost has no bin-mapper alignment; :reference is a LightGBM-only concept.
XGBoost  :group SIGNALED UNSUPPORTED-ARGUMENT: make-dataset's :group is not supported by XGBOOST: ranking group sizes are not yet attached by this backend.
XGBoost  :parameters silently accepted, 4 rows
XGBoost  update-one-iteration => T
SIGNALED UNSUPPORTED-ARGUMENT: save-model's :num-iteration is not supported by XGBOOST: XGBoosterSaveModel has no iteration limit; every boosted round is saved.
SIGNALED UNSUPPORTED-ARGUMENT: model-to-string's :num-iteration is not supported by XGBOOST: XGBoosterSaveModelToBuffer has no iteration limit.
SIGNALED UNSUPPORTED-ARGUMENT: feature-importance's :num-iteration is not supported by XGBOOST: XGBoosterFeatureScore has no iteration limit.
LightGBM update-one-iteration => T
```

And, separately, the `feature-importance` shape rejection above, which needs a linear
(`gblinear`), multi-class booster to trigger -- a `multi:softprob` objective over 9 rows
in 3 classes:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/xgboost) :silent t)

(let* ((backend (cl-gbdt:open-backend :xgboost))
       (rows-per-class 3) (num-classes 3) (cols 3)
       (rows (* rows-per-class num-classes))
       (matrix (make-array (list rows cols) :element-type 'double-float))
       (label (make-array rows :element-type 'single-float)))
  (dotimes (row rows)
    (let ((class (floor row rows-per-class)) (offset (mod row rows-per-class)))
      (dotimes (col cols)
        (setf (aref matrix row col) (coerce (+ (* class 10) offset col) 'double-float)))
      (setf (aref label row) (coerce class 'single-float))))
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend matrix :label label))
    (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 5
                                     :parameters (list :booster "gblinear"
                                                        :objective "multi:softprob"
                                                        :num-class num-classes
                                                        :verbosity 0)))
      (handler-case (cl-gbdt:feature-importance booster :kind :split)
        (error (c) (format t "feature-importance SIGNALED: ~A: ~A~%" (type-of c) c)))))
  (cl-gbdt:close-backend backend))
```

Output:

```
feature-importance SIGNALED: UNSUPPORTED-ARGUMENT: feature-importance's booster is not supported by XGBOOST: XGBoosterFeatureScore reported a 2-dimensional shape (3 3) instead of one score per feature -- most likely a linear (gblinear) booster's :split importance on a multi-class model, whose scores are a per-class matrix; no single value per feature can be derived without inventing a reduction this backend does not vouch for.
```

## Systems

| System | Purpose |
|---|---|
| `cl-gbdt` | Core: package, condition hierarchy, matrix marshalling, backend registry and `open-backend` protocol, the unified API's generic functions -- no methods, and no shared library required to load it |
| `cl-gbdt/lightgbm` | The LightGBM backend: all 12 unified-API methods (`src/lightgbm/backend.lisp`), built on the generated CFFI bindings for the LightGBM C API (`src/lightgbm/c-api.lisp`) |
| `cl-gbdt/xgboost` | The XGBoost backend: all 12 unified-API methods (`src/xgboost/backend.lisp`), built on the generated CFFI bindings for the XGBoost C API (`src/xgboost/c-api.lisp`) |
| `cl-gbdt/regen` | The binding emitter (`src/regen/`). Development-only -- never appears in `cl-gbdt`'s, `cl-gbdt/lightgbm`'s, or `cl-gbdt/xgboost`'s dependency graph, so an ordinary user never needs it or its dependencies (`alexandria`, `com.inuoe.jzon`). `cffi/c2ffi` is *not* one of them -- it is a dependency of `tools/regen.lisp`, which quickloads it directly, not of the `cl-gbdt/regen` system itself |
| `cl-gbdt/tests` | The Rove test suite |

Each backend system (`cl-gbdt/lightgbm` depends on `cl-gbdt/src/lightgbm/backend`,
`cl-gbdt/xgboost` on `cl-gbdt/src/xgboost/backend`) implements all 12 methods of the
unified API for its library. It depends on `cffi`, its own generated `c-api.lisp`, and
the individual `cl-gbdt/src/*` leaf systems its methods need -- `protocol`, `handle`,
`backend`, `conditions`, `parameters`, `data`, `library`, `foreign` -- but not on the
aggregate `cl-gbdt` system/package that re-exports all of those under one name (see
`src/all.lisp`). Loading only a backend system therefore attaches its methods to the
generic functions but leaves the `cl-gbdt` package itself undefined; load core
`cl-gbdt` too for the `cl-gbdt:make-dataset` spelling rather than
`cl-gbdt/src/protocol:make-dataset` (see [Usage](#usage)). Each backend is loadable
independently of the other, so the library works on a machine where only one of the two
shared libraries is present.

## Running the tests

**First, once: ASDF 3.3.7 or newer is required.** Roswell ships 3.3.1, whose
`package-inferred-system` dependency scanner does not know the `:local-nicknames`
clause; loading any system that reaches `src/regen/emit.lisp` dies with
`:LOCAL-NICKNAMES fell through ECASE expression` before a single test runs.

```bash
ros install asdf
```

Then:

```lisp
(ql:quickload :cl-gbdt/tests)
(asdf:test-system :cl-gbdt/tests)
```

or, from the shell (this project does not put `sbcl` on `PATH`; use `ros run`):

```bash
ros run -- --non-interactive \
  --eval '(ql:quickload :cl-gbdt/tests :silent t)' \
  --eval '(asdf:test-system :cl-gbdt/tests)'
```

No shared library is required: every test in this branch runs against the
generated bindings' source text, a mock backend, or in-process data structures
(design doc section 12, layer 1). Rove prints one line per test *suite*, not per
`deftest` form -- read the printed test names rather than trusting the "N tests
completed" count.

To run a single test from the REPL:

```lisp
(rove:run-test 'cl-gbdt/tests/backend::some-test-name)
```

## Running the functional tests

`cl-gbdt/tests/functional` is a separate system that calls the real LightGBM and
XGBoost shared libraries -- design doc section 12, layer 2. It exercises the raw FFI
directly: loading each library, reading its version, and running a small train/predict
round trip against a trivially separable dataset. Each round trip asserts more than
final prediction values -- every handle it creates is non-null, every output buffer's
length matches the row count, and the boosting iteration count reads back correctly --
and, as the property that ties the FFI plumbing together, that positive-label
predictions come back higher than negative-label ones.

This system is SBCL-only: both round trips pin arrays with `sb-sys` primitives
directly, unlike `src/data.lisp`'s `#+sbcl`-guarded idiom, and have no portable
fallback.

Run `./tools/fetch-libs.sh` first to vendor the libraries into `vendor/`. Then:

```bash
ros run -- --non-interactive \
  --eval '(ql:quickload :cl-gbdt/tests/functional :silent t)' \
  --eval '(asdf:test-system :cl-gbdt/tests/functional)'
```

A backend whose library is missing skips rather than fails, naming
`./tools/fetch-libs.sh` in the skip message -- `vendor/` is git-ignored, so a fresh
clone legitimately has neither library yet. `CL_GBDT_LIGHTGBM_LIB` and
`CL_GBDT_XGBOOST_LIB` override discovery, for pointing the suite at a system-wide
install instead of the vendored copy.

## Continuous integration

Two workflows, so the badges above report independently — a GitHub Actions badge covers a
whole workflow file, not a job:

- `.github/workflows/test.yml` runs both suites on Linux x86_64, Linux aarch64 and macOS
  aarch64. The matrix is the point: the bindings are generated on one machine and
  committed, so passing on that machine proves little. macOS is also the only place the
  `.dylib` discovery path is exercised at all.
- `.github/workflows/lint.yml` runs the static checks on one target, since nothing they
  look at varies by machine.

The logic lives in scripts rather than in the YAML, so the same checks run locally:

```bash
CL_GBDT_TEST_SYSTEM=cl-gbdt/tests ros run -- --non-interactive \
  --load tools/ci/run-tests.lisp          # layer 1
CL_GBDT_TEST_SYSTEM=cl-gbdt/tests/functional ros run -- --non-interactive \
  --load tools/ci/run-tests.lisp          # layer 2, needs ./tools/fetch-libs.sh first
ros run -- --non-interactive --load tools/ci/lint.lisp
ros run -- --non-interactive --load tools/ci/check-leaf-systems.lisp
```

Two things those scripts do that the plain commands above do not, and that CI needs:

- **They exit non-zero when a test fails.** `asdf:test-system` exits 0 regardless, so a job
  invoking it directly would be permanently green. `rove:run` returns false on failure and
  the script turns that into a status.
- **They check which foreign libraries were opened.** Layer 1 must open none; layer 2 must
  open both. A functional run where every test skipped for want of `vendor/` prints the same
  summary as one where every test passed, so the difference has to be asserted rather than
  inferred.

`tools/ci/lint.lisp` runs mallet *and* a column-width check, because mallet does not check
line length — a 132-column file passes it without comment. mallet is not in the Quicklisp
dist; the workflow clones it into `~/.roswell/local-projects/`, pinned, and a current ASDF is
installed first because Roswell ships 3.3.1, which predates `:local-nicknames` in
`uiop:define-package`.

**`tools/ci/check-leaf-systems.lisp` loads every leaf system alone, each in its own fresh
`ros run` subprocess.** This guards the principal risk this library's
`:package-inferred-system` layout carries: ASDF infers a file's dependencies *only* from its
`uiop:define-package` clauses — `:use`, `:import-from` (even one naming zero symbols), and
`:local-nicknames`. A file that calls, say, `cffi:defcfun` without naming `#:cffi` in one of
those clauses gets no declared dependency on CFFI; it loads correctly whenever something else
in the same image happened to load CFFI first, and breaks the moment load order shifts. Because
that failure is order-dependent, loading every leaf into one shared image — the obvious way to
"prove" they all load — proves nothing: the first file's load satisfies the next file's
undeclared dependency. Each leaf therefore gets a subprocess with a fresh image, where nothing
but what it declares is on hand.

The check also doubles as the enforcement mechanism for this project's naming convention: **a
leaf system's name is `cl-gbdt/` followed by its path from the repository root, extension
dropped** — `src/lightgbm/c-api.lisp` names `cl-gbdt/src/lightgbm/c-api`, and its
`uiop:define-package` form must name that same symbol. The checker derives its list of systems
to check from the filesystem (every `.lisp` file under `src/` and `tests/`, generated files
included) rather than from a hardcoded list, so a new file is picked up automatically the next
time CI runs — a contributor adding one only needs to follow the path-is-the-name rule and give
the package the `defpackage` clauses its file actually needs.

**On macOS the functional tests also need `brew install libomp`.** The macOS wheels link
against `@rpath/libomp.dylib` and, unlike the manylinux ones, do not vendor an OpenMP
runtime, so `dlopen` fails without it.

## Regenerating the bindings

`src/lightgbm/c-api.lisp` and `src/xgboost/c-api.lisp` are generated, checked in,
and architecture-independent by construction (design doc section 5.1). **You do
not need to regenerate them to use, build, or test this library** -- the normal
build reads them as ordinary `cl-source-file`s and depends on nothing but `cffi`.
Regeneration is a developer-only step, for when a header is updated or the
emitter itself changes.

Regeneration needs Docker (to run c2ffi in a pinned, reproducible LLVM 18
environment) and, for the header-fetch step, network access. Run these in order
from the repository root:

```bash
tools/fetch-headers.sh
docker build -f tools/Dockerfile.c2ffi -t cl-gbdt-c2ffi:llvm-18 tools/
ros run -- --non-interactive --load tools/regen.lisp
```

1. `tools/fetch-headers.sh` downloads the headers reachable from each backend's
   `c_api.h`, at the tags pinned in `ffi-spec/VERSIONS`, into `ffi-spec/`. No
   patching or hand-editing.
2. The `docker build` compiles c2ffi, pinned to an exact commit on its LLVM-18
   branch, into the `cl-gbdt-c2ffi:llvm-18` image. `tools/c2ffi.sh` invokes that
   image; `tools/regen.lisp` invokes `tools/c2ffi.sh`.
3. `tools/regen.lisp` runs c2ffi over the vendored headers for the local
   architecture, then runs `cl-gbdt/src/regen/all:emit-bindings` over its output, and
   validates the result (minimum function count, required symbols present, no
   architecture-dependent types) before replacing the committed file. A failed
   validation leaves the previously committed file untouched.

Expected output ends with two lines like:

```
==> LIGHTGBM
    99 functions, 12 constants -> src/lightgbm/c-api.lisp
==> XGBOOST
    78 functions, 0 constants -> src/xgboost/c-api.lisp
done
```

c2ffi generates for the local architecture only; design doc section 5.2 explains
why cross-architecture generation was tried and abandoned (it fails, and fails
silently -- see section 5.3 for how the emitter defends against that). The
architecture independence of the *output* does not depend on which architecture
generated it (section 5.1), so any one machine's regeneration is sufficient for
everyone.

`ffi-spec/ABI-BLACKLIST.md` records which C functions cl-gbdt must never call, and
why, independent of whether they happen to be emitted.

## License

MIT; see `LICENSE`.

The upstream C API headers vendored under `ffi-spec/` remain under their own terms --
LightGBM's are MIT, XGBoost's are Apache 2.0 -- and each retains its upstream copyright
notice, and both upstream licence texts are vendored verbatim under `LICENSES/` so that
recipients receive a copy rather than a link. `THIRD-PARTY-LICENSES.md` records what is
redistributed and under what. cl-gbdt does
not bundle either backend's compiled library; `tools/fetch-libs.sh` fetches those into the
git-ignored `vendor/` directory for development only.
