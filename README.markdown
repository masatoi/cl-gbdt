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
LightGBM and XGBoost shared libraries, exercised by 820 functional assertions across 15 test
files (design doc section 12, layer 2), in addition to 551 assertions across 21 test files
that need no shared library at all (layer 1). `train` also returns a `training-report` as
its secondary value, and takes
`:early-stopping` to end a run once a watched metric stops improving -- see
[Training report](#training-report) below. `make-dataset` and `predict` also accept a
`csr-matrix` wherever they accept a dense matrix -- see [Sparse
input](#sparse-input-csr-matrices) below -- and both also take `:missing`, the value in
the caller's data that means missing, gated on the `:missing-value` capability that only
XGBoost provides -- see [Missing values](#missing-values) below. `make-dataset` also
takes `:categorical-features`, the 0-based columns that hold categories rather than
quantities, gated on the `:categorical-features` capability that both backends provide
-- `predict` takes no such argument, the trained trees already carrying the category
sets they split on -- see [Categorical features](#categorical-features) below. `predict`
also returns the SHAPE the backend states for the result it just wrote as a second value --
a list of integers in `array-dimensions` order, or `NIL` where the backend states none --
gated on the `:prediction-shape` capability that both backends provide; XGBoost reads its
own `out_shape`/`out_dim` back from the library and LightGBM derives what it can, stating
`NIL` for `:leaf-index` -- see [Prediction shape](#prediction-shape) below. `train` also
takes `:objective`, a function that turns the current raw scores into a gradient and a
Hessian so a run boosts against the caller's own loss, gated on the `:custom-objective`
capability that both backends provide; the two libraries flatten that array in opposite
orders and the wrapper absorbs it, and on LightGBM `:objective` overrides any `objective`
in `:parameters` -- all five spellings that library honours -- forcing it to `"none"`,
since the library refuses the combination outright -- see [Custom
objective](#custom-objective) below. `train` also takes `:evaluation`, a function called
once per dataset per iteration with that dataset's `predict :kind :normal` scores and the
dataset's index -- `0` the training set, `N+1` the Nth `:valid-sets` entry -- that returns a
metric name and a real or `NIL` value, gated on the `:custom-evaluation` capability that both
backends provide out of different lists, LightGBM probing it and XGBoost declaring it; the
values become their own report series, watchable by `:early-stopping` under the returned
name, appended after the library's own series rather than replacing them -- see [Custom
evaluation](#custom-evaluation) below. See [Usage](#usage) below for a worked example.

Loading `cl-gbdt` itself still does not require either `liblightgbm.so` or
`libxgboost.so` to be installed -- see [Systems](#systems): a shared library is opened
only by an explicit `open-backend` call, from whichever backend system(s) you load on
top of the core. Each backend ships as **two** systems: `cl-gbdt/lightgbm` is that
backend's own API alone -- it opens and closes the library, builds datasets and boosters,
trains and predicts, and none of the thirteen portable generic functions above are part of
it -- while `cl-gbdt/lightgbm/unified`
adds their LightGBM methods, and core `cl-gbdt` with them. `cl-gbdt/xgboost` and
`cl-gbdt/xgboost/unified` divide the same way. Every example below that calls one of
those thirteen loads a `/unified` system; the few that do not are the ones
demonstrating what a Layer 1 system does and does not carry.

## Usage

A worked example first, then the details a caller moving between the two backends
needs. Every code block below was actually run to produce the output pasted beneath it
(SBCL via `ros run`, with `./tools/fetch-libs.sh`'s vendored libraries already present).

### Quick start

Load the core system and one backend's `/unified` system -- `cl-gbdt/lightgbm` alone
carries no `cl-gbdt:train`; see [the section below](#two-systems-per-backend) -- open the
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

### Two systems per backend

`cl-gbdt.asd` declares **two** systems for each backend, and the difference between them is
the first thing to get right:

| System | What it carries |
|---|---|
| `cl-gbdt/lightgbm` | **Layer 1 alone.** Opens and closes the LightGBM shared library, and publishes LightGBM's own API: the six finished operations `create-dataset`, `create-booster`, `update-one-iteration`, `predict`, `free-dataset` and `free-booster` -- a whole training run and the inference after it -- plus `booster-eval`, `booster-eval-names` and the `lightgbm-backend` class, plus the shared basis a standalone caller needs: `open-backend`, `close-backend`, `backend-supports-p` and its siblings, `make-csr-matrix` and the `csr-matrix` readers, `handle-released-p`, `booster-training-set`, `booster-validation-sets`, and the whole condition hierarchy. **`cl-gbdt:train` and the other twelve portable generic functions are not part of it**, and loading it does not define the `cl-gbdt` package at all |
| `cl-gbdt/lightgbm/unified` | That, plus `src/lightgbm/protocol.lisp` -- LightGBM's methods on all thirteen portable generics -- plus core `cl-gbdt` itself, which it depends on. This is what the quick start above loads, and what every example here that calls `cl-gbdt:train` loads |

`cl-gbdt/xgboost` and `cl-gbdt/xgboost/unified` divide identically -- the same six
operations under the same names, different symbols in a different package -- and XGBoost's
Layer 1 API additionally publishes `slice-model`, `evaluate-one-iteration` and
`booster-boosted-rounds`. Loading one backend never loads the other, and no backend system
is a dependency of core `cl-gbdt`; load whichever matches the shared library you have.
**Loading both backends' `/unified` systems is how one program drives both libraries through
one portable API** -- that is what the two-backend examples further down do.

A Layer 1 system alone opens the library, answers capability questions and closes it again,
without the `cl-gbdt` package existing -- run in a fresh image, since anything that has
already loaded a `/unified` system has core `cl-gbdt` too:

```lisp
(ql:quickload :cl-gbdt/lightgbm :silent t)

(format t "cl-gbdt package: ~S~%" (find-package :cl-gbdt))

(let ((backend (cl-gbdt/lightgbm:open-backend :lightgbm)))
  (format t "name:            ~S~%" (cl-gbdt/lightgbm:backend-name backend))
  (format t "open:            ~S~%" (cl-gbdt/lightgbm:backend-open-p backend))
  (format t "sparse input:    ~S~%" (cl-gbdt/lightgbm:backend-supports-p backend :sparse-input))
  (cl-gbdt/lightgbm:close-backend backend)
  (format t "open:            ~S~%" (cl-gbdt/lightgbm:backend-open-p backend)))
```

Output:

```
cl-gbdt package: NIL
name:            :LIGHTGBM
open:            T
sparse input:    T
open:            NIL
```

**A Layer 1 system alone also trains and predicts.** Six operations per backend carry a whole
run -- `create-dataset`, `create-booster`, `update-one-iteration`, `predict`, `free-dataset`
and `free-booster` -- with no unified API in the image at all. Both blocks below run in one
fresh image that never defines the `cl-gbdt` package:

```lisp
(ql:quickload '(:cl-gbdt/lightgbm :cl-gbdt/xgboost) :silent t)

(format t "cl-gbdt package: ~S~%" (find-package :cl-gbdt))

(defparameter *matrix* (make-array '(16 2) :element-type 'double-float))
(defparameter *label* (make-array 16 :element-type 'double-float))
(dotimes (row 16)
  (setf (aref *matrix* row 0) (coerce row 'double-float)
        (aref *matrix* row 1) (coerce (mod row 4) 'double-float)
        (aref *label* row) (if (< row 8) 0d0 1d0)))

;; LightGBM: dataset, booster, twenty iterations, predictions, both frees.
(let* ((backend (cl-gbdt/lightgbm:open-backend :lightgbm))
       (parameters '(:objective "binary" :num-leaves 3 :min-data-in-leaf 1
                     :min-data-in-bin 1 :verbose -1))
       (data (cl-gbdt/lightgbm:create-dataset backend *matrix* :label *label*
                                              :parameters parameters))
       (booster (cl-gbdt/lightgbm:create-booster backend data :parameters parameters)))
  (dotimes (round 20) (cl-gbdt/lightgbm:update-one-iteration booster))
  (multiple-value-bind (result shape) (cl-gbdt/lightgbm:predict booster *matrix*)
    (format t "LightGBM  shape ~S  row 0 ~,3F  row 15 ~,3F~%"
            shape (aref result 0 0) (aref result 15 0)))
  (cl-gbdt/lightgbm:free-booster booster)
  (cl-gbdt/lightgbm:free-dataset data)
  (cl-gbdt/lightgbm:close-backend backend))

;; XGBoost: the same six operations, in its own parameter vocabulary.
(let* ((backend (cl-gbdt/xgboost:open-backend :xgboost))
       (data (cl-gbdt/xgboost:create-dataset backend *matrix* :label *label*))
       (booster (cl-gbdt/xgboost:create-booster
                 backend data
                 :parameters '(:objective "binary:logistic" :max-depth 2 :eta 0.5))))
  (dotimes (round 20) (cl-gbdt/xgboost:update-one-iteration booster))
  (multiple-value-bind (result shape) (cl-gbdt/xgboost:predict booster *matrix*)
    (format t "XGBoost   shape ~S  row 0 ~,3F  row 15 ~,3F~%"
            shape (aref result 0 0) (aref result 15 0)))
  (cl-gbdt/xgboost:free-booster booster)
  (cl-gbdt/xgboost:free-dataset data)
  (cl-gbdt/xgboost:close-backend backend))
```

Output:

```
cl-gbdt package: NIL
LightGBM  shape (16 1)  row 0 0.066  row 15 0.934
XGBoost   shape (16 1)  row 0 0.134  row 15 0.866
```

Row 0's label is 0 and row 15's is 1, so both backends learned the boundary; `predict`'s
second value is the shape it states for the array it just wrote, exactly as the unified
`predict` states one. `:parameters` is each library's **own** vocabulary in both blocks --
nothing at this layer translates a key, which is why LightGBM gets `"binary"` and XGBoost
`"binary:logistic"` -- and XGBoost's `create-dataset` takes no `:parameters` at all, its
creation config having no such concept (see [Where the two backends genuinely
differ](#where-the-two-backends-genuinely-differ)). `tests/functional/lightgbm-standalone.lisp`
and `tests/functional/xgboost-standalone.lisp` are the same run as a test, each naming its
backend's public package and no other system *of this project* -- `rove` aside, they declare
nothing -- so that the claim is enforced by the build rather than asserted here.

**What a Layer 1 caller still cannot do** is everything else the thirteen generics carry.
Two lists, and they are not the same kind of gap. Seven operations simply have no Layer 1
counterpart yet -- `save-model`, `load-model`, `model-to-string`, `feature-importance`,
`evaluation`, `dataset-num-rows` and `dataset-num-features` -- and the next increment brings
them down; the first bullet of
`docs/cl-gbdt-layered-api-implementation-policy.md`'s フォローアップ section records that.
The training report, early stopping, and `train`'s `:objective` and `:evaluation` callbacks
are the other list, and they stay where they are: they are `cl-gbdt:train`'s own concepts,
built *around* the loop above rather than being operations inside it, so there is nothing at
Layer 1 for them to be counterparts of.

Loading core `cl-gbdt` next to a Layer 1 system gives you the `cl-gbdt:` spelling but not the
methods behind it, and asking for one anyway is a named condition rather than a bare
`no-applicable-method` error -- it names the system to load:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm) :silent t)

(let ((backend (cl-gbdt:open-backend :lightgbm))
      (matrix (make-array '(2 2) :element-type 'double-float :initial-element 1.0d0)))
  (handler-case (cl-gbdt:make-dataset backend matrix)
    (error (c) (format t "SIGNALED ~A~%  ~A~%" (type-of c) c)))
  (cl-gbdt:close-backend backend))
```

Output:

```
SIGNALED BACKEND-METHODS-NOT-LOADED
  CL-GBDT/SRC/PROTOCOL:MAKE-DATASET has no method for LIGHTGBM: that backend's unified-API methods are not loaded. Load cl-gbdt/lightgbm/unified -- cl-gbdt/lightgbm is that backend's own API alone, without cl-gbdt's portable one.
```

`backend-methods-not-loaded` is a `backend-error`, so `handler-case` on the existing
condition hierarchy already catches it; `backend-methods-not-loaded-generic-function` reads
back which generic was asked for. Every one of the thirteen has such a default method, so
whichever one a program reaches first says the same thing.

### Backend-specific packages

`cl-gbdt/lightgbm` and `cl-gbdt/xgboost` name a package as well as the Layer 1 system in the
heading above. `docs/cl-gbdt-layered-api-implementation-policy.md` section 3 calls these
the public packages for backend-specific API -- distinct from each backend's internal
`cl-gbdt/src/lightgbm/all`/`cl-gbdt/src/xgboost/all` aggregation, and, always, from the raw
generated CFFI bindings (`cl-gbdt/src/lightgbm/c-api`, `cl-gbdt/src/xgboost/c-api`), which
neither package re-exports (policy sections 3 and 11):

```lisp
(ql:quickload :cl-gbdt/lightgbm :silent t)

(flet ((from-conditions-p (symbol)
         (string= "CL-GBDT/SRC/CONDITIONS" (package-name (symbol-package symbol)))))
  (let ((symbols (sort (loop :for symbol :being :the :external-symbols :of "CL-GBDT/LIGHTGBM"
                             :collect symbol)
                       #'string< :key #'symbol-name)))
    (format t "~D external symbols, ~D of them the condition hierarchy~%"
            (length symbols) (count-if #'from-conditions-p symbols))
    (format t "~S~%" (remove-if #'from-conditions-p symbols))))
```

Output:

```
81 external symbols, 49 of them the condition hierarchy
(CL-GBDT/SRC/BACKEND:*KNOWN-CAPABILITIES*
 CL-GBDT/SRC/BACKEND:BACKEND-CAPABILITIES CL-GBDT/SRC/BACKEND:BACKEND-INFO
 CL-GBDT/SRC/BACKEND:BACKEND-LIBRARY-PATH CL-GBDT/SRC/BACKEND:BACKEND-NAME
 CL-GBDT/SRC/BACKEND:BACKEND-OPEN-P CL-GBDT/SRC/BACKEND:BACKEND-SUPPORTS-P
 CL-GBDT/SRC/BACKEND:BACKEND-VERSION CL-GBDT/SRC/HANDLE:BOOSTER
 CL-GBDT/SRC/LIGHTGBM/NATIVE:BOOSTER-EVAL
 CL-GBDT/SRC/LIGHTGBM/NATIVE:BOOSTER-EVAL-NAMES
 CL-GBDT/SRC/HANDLE:BOOSTER-TRAINING-SET
 CL-GBDT/SRC/HANDLE:BOOSTER-VALIDATION-SETS CL-GBDT/SRC/BACKEND:CLOSE-BACKEND
 CL-GBDT/SRC/LIGHTGBM/API:CREATE-BOOSTER
 CL-GBDT/SRC/LIGHTGBM/API:CREATE-DATASET CL-GBDT/SRC/DATA:CSR-MATRIX
 CL-GBDT/SRC/DATA:CSR-MATRIX-INDICES CL-GBDT/SRC/DATA:CSR-MATRIX-INDPTR
 CL-GBDT/SRC/DATA:CSR-MATRIX-NUM-COLUMNS CL-GBDT/SRC/DATA:CSR-MATRIX-NUM-ROWS
 CL-GBDT/SRC/DATA:CSR-MATRIX-VALUES CL-GBDT/SRC/HANDLE:DATASET
 CL-GBDT/SRC/LIGHTGBM/API:FREE-BOOSTER CL-GBDT/SRC/LIGHTGBM/API:FREE-DATASET
 CL-GBDT/SRC/HANDLE:HANDLE-BACKEND CL-GBDT/SRC/HANDLE:HANDLE-RELEASED-P
 CL-GBDT/SRC/LIGHTGBM/CLASSES:LIGHTGBM-BACKEND CL-GBDT/SRC/DATA:MAKE-CSR-MATRIX
 CL-GBDT/SRC/BACKEND:OPEN-BACKEND CL-GBDT/SRC/LIGHTGBM/API:PREDICT
 CL-GBDT/SRC/LIGHTGBM/API:UPDATE-ONE-ITERATION)
```

Those 81 fall into three groups. **LightGBM's own API** is nine of them: the six finished
operations of the standalone example above -- `create-dataset`, `create-booster`,
`update-one-iteration`, `predict`, `free-dataset` and `free-booster`, all homed in
`cl-gbdt/src/lightgbm/api` -- plus that backend's own evaluation entry points, `booster-eval`
and `booster-eval-names`, and the `lightgbm-backend` class, useful for `typep` or for
specializing your own methods on one specific backend rather than the shared `backend`
(`open-backend` itself never needs it, since it looks classes up by the
`:lightgbm`/`:xgboost` keyword internally, not by this symbol). Four of those nine names --
`predict`, `update-one-iteration`, `free-dataset`, `free-booster` -- are *also* names
`cl-gbdt` exports, and these are **different symbols**: plain functions here, generic
functions there, so an image holding both packages has to say which it means. **The shared
basis** is the other twenty-three non-condition symbols: `open-backend`, `close-backend`,
`backend-supports-p` and the rest of the backend readers, `handle-released-p`,
`handle-backend`, `booster-training-set` and `booster-validation-sets`, the
`dataset`/`booster` handle classes, and `make-csr-matrix` with the `csr-matrix` type and its
five readers, so that the sparse half of `create-dataset` and `predict` is reachable from the
package that publishes them. These are republished here, rather than left to core `cl-gbdt`,
so that a program loading this Layer 1 system alone can open, question and close a backend
without naming an internal package -- and unlike the four doubled operation names, they are
the very symbols `cl-gbdt` exports, one symbol reached two ways with nothing to
disambiguate. And **the condition hierarchy** is the remaining 49, re-exported whole from
`cl-gbdt/src/conditions`: every type and accessor there is already reviewed public API, so a
Layer 1 caller catches `foreign-call-error` or `backend-library-not-found` by the same name
a unified-API caller does.

`cl-gbdt/xgboost` is the same shape -- 82 external symbols, the same 49 conditions and the
same 23 shared-basis symbols -- with ten of its own rather than nine: the same six
operations under its own package's symbols, plus `xgboost-backend`,
`slice-model`, `booster-boosted-rounds` (see [the capability
section](#asking-a-backend-what-it-can-do)), and an
`evaluate-one-iteration` of its own, which takes different arguments and returns something
different (see [the differences table](#where-the-two-backends-genuinely-differ)): the two
operations were deliberately given different names -- `cl-gbdt/lightgbm:booster-eval` reads
the validation data LightGBM attached at train time, addressed by index, while
`cl-gbdt/xgboost:evaluate-one-iteration` evaluates whatever DMatrices the caller passes it
and ignores `valid-sets` entirely -- rather than one shared name that a caller `:use`-ing
both packages would have to resolve a conflict over. Package-qualify them anyway, so it
stays clear at the call site which backend's entry point you mean, or use the portable
`cl-gbdt:evaluation` instead, which is one name for both.

Nothing else from either backend's `native.lisp` -- library discovery, the
raw-status-code checker, the internal `%`-functions that turn a raw C call into something
safe to call from Lisp -- is published: none of those is a reviewed, Lisp-level
backend-specific operation, only infrastructure the two backend systems already use
internally. Backend-specific safe API -- LightGBM's `rollback-one-iteration`, `refit`, and
the rest of policy section 3's Layer 1 examples -- is added here one contract at a time as
it is designed and reviewed, not by widening today's re-export to cover `native` wholesale.
The six operations above are the most recent addition made that way, and they come from a
third file, `src/<backend>/api.lisp`, rather than from `native.lisp` at all: `native.lisp`
holds the `%`-functions that take and return raw pointers, `api.lisp` the finished
operations built on top of them that take a backend or a handle and hand back a handle or a
result.

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

### Asking a backend what it can do

Not every operation exists on both backends, and not every operation exists in every
version of one backend's shared library. `backend-supports-p` is the question:

```lisp
(let ((backend (cl-gbdt:open-backend :xgboost)))
  (cl-gbdt:backend-supports-p backend :model-slicing))   ; => T
```

Four things it promises, each of which is the point of it existing at all:

- **A true answer means the foreign symbols that capability needs actually resolved in the
  shared library that was loaded** -- probed once at `open-backend` and recorded in
  `backend-capabilities` -- not that the headers cl-gbdt's bindings were generated from
  declared them. An XGBoost too old to have `XGBoosterSlice` is a working XGBoost that
  cannot slice, and it opens normally: policy section 8's *optional* symbol tier fails only
  the capability, never `open-backend`, unlike a missing *required* symbol, which signals
  `missing-foreign-symbols` immediately.
- **A false answer never means cl-gbdt will substitute something else.** There is no silent
  fallback and no emulation anywhere behind this API. The operation is simply unavailable.
- **An unregistered name signals `unknown-capability` rather than answering `nil`**, so a
  misspelling cannot be mistaken for "supported, but not here". The registered names live
  in `*known-capabilities*`; a name is registered as soon as it is a question worth asking,
  whether or not any backend answers true to it yet.
- **Asking first is never required.** The operation re-checks for itself and signals
  `capability-unavailable` -- a distinct condition from `unknown-capability`, since the
  question was well formed and the answer is simply no. `backend-supports-p` is for a
  caller who wants to branch *before* calling, not a precondition anything depends on:

```lisp
;; Signals capability-unavailable on a library without XGBoosterSlice, whether or not
;; the caller asked backend-supports-p first. It never falls back to anything else.
(cl-gbdt/xgboost:slice-model booster :begin 0 :end 5)
```

`backend-info` reports the whole probed plist, false capabilities included, so it shows
what was asked as well as what was answered. Nine of the ten registered capabilities
answer true somewhere today: `:model-slicing`, on XGBoost only -- see the model-slicing row
in the table below -- plus `:evaluation-history` and `:early-stopping` on both backends,
since `train` records a history and takes `:early-stopping` (see
[Training report](#training-report)), `:sparse-input` on both, since both libraries
export the CSR entry points it names (see [Sparse input](#sparse-input-csr-matrices)),
`:missing-value` on XGBoost only (see [Missing values](#missing-values)),
`:categorical-features` on both (see [Categorical features](#categorical-features)),
`:prediction-shape` on both, since `predict` states a shape for the result it just predicted
on both libraries (see [Prediction shape](#prediction-shape)), `:custom-objective` on
both, since `train` boosts against a caller's own gradient and Hessian on both libraries (see
[Custom objective](#custom-objective)), and `:custom-evaluation` on both, since `train`
records a caller-written metric per dataset on both libraries (see [Custom
evaluation](#custom-evaluation)). The tenth, `:multidimensional-feature-score`, is
registered and false everywhere, which says "not supported yet" rather than "never heard of
it".

`:evaluation-history` is true unconditionally rather than probed. The C functions behind it
are in each backend's `*required-symbols*`, so a library missing them never opens at all
and there is no state in which an open backend cannot record a history; each backend names
the capability in its own `*provided-capabilities*`, which `open-backend` records as true
without a symbol lookup. A probe can only answer from a symbol that might be absent, which
is the right shape for `:model-slicing` and the wrong one for a feature that is simply
always there.

### Where the two backends genuinely differ

A caller moving code from one backend to the other needs this in one place -- checked
directly against both backends' source, not only the differences the design doc calls
out first:

| | LightGBM | XGBoost |
|---|---|---|
| `make-dataset`'s `:reference` | Aligns the new dataset's bin mapper to an existing one's (required for a `train` `:valid-sets` entry) | Signals `unsupported-argument` -- no bin-mapper concept |
| `make-dataset`'s `:parameters` | Configures the dataset's own binning (`max_bin` and friends) | Signals `unsupported-argument`: the vendored header (`ffi-spec/xgboost/include/xgboost/c_api.h`) documents only `missing`/`nthread`/`data_split_mode` for `XGDMatrixCreateFromDense`'s config JSON, none of which are LightGBM's dataset-level binning keys, and confirmed empirically, the library silently ignores any other key rather than rejecting it -- forwarding `:parameters` there regardless would just move today's silent drop one layer deeper instead of fixing it |
| `update-one-iteration`'s return value | `nil` once an iteration produces no further split -- a real signal | Always `t` after a successful call; XGBoost's booster protocol has no equivalent signal |
| `predict`'s `:kind` on a `csr-matrix` | All four kinds, `LGBM_BoosterPredictForCSR` serving each of them with the same values the dense path produces | `:normal` and `:raw` only. `XGBoosterPredictFromCSR` is that library's *inplace* prediction entry point, not a CSR spelling of the dense call, and it refuses `:contrib` and `:leaf-index` -- passed through as `foreign-call-error`. See [Sparse input](#sparse-input-csr-matrices) for the measured matrix and the workaround |
| `predict`'s second value (shape) | States the result array's own `array-dimensions` for `:normal`/`:raw`, derives one for `:contrib`, and states `NIL` for `:leaf-index` -- there is no property this project can check for that kind's sub-layout | Reads `out_shape`/`out_dim` back from the library and states it verbatim, on both matrix forms and for every `:kind` that form serves (all four dense, `:normal`/`:raw` only on a `csr-matrix`, per the row above). See [Prediction shape](#prediction-shape) for both derivations and the measured shapes |
| `save-model`'s `:num-iteration` | Limits how many trees are saved | Signals `unsupported-argument` -- `XGBoosterSaveModel` always saves every round |
| `model-to-string`'s `:num-iteration` | Limits the rounds serialized | Signals `unsupported-argument` -- no iteration-limited variant exists |
| `feature-importance`'s `:num-iteration` | Limits the importance calculation | Signals `unsupported-argument` -- no iteration-limited variant exists |
| `feature-importance`'s result shape | Always one number per feature | Signals `unsupported-argument` instead of returning a result when the model reports a multi-dimensional score shape -- a `gblinear` booster's importance on a multi-class model, whose scores are a per-class matrix with no single-value reduction this backend will invent |
| What `evaluation` evaluates | The datasets `train` attached, read back by index (`LGBM_BoosterGetEval`): the library computed these metrics during training and this reads them out | The booster's own retained training set and `:valid-sets` entries, which this backend hands to `XGBoosterEvalOneIter` explicitly -- that call evaluates whatever DMatrices it is given and consults nothing the booster was built with, so passing the retained ones is what makes the index mean the same thing on both backends |
| `evaluation`'s values | `LGBM_BoosterGetEval`'s own doubles, returned unmodified -- the secondary value says `:value-source :library-doubles` | Parsed out of the single formatted line `XGBoosterEvalOneIter` produces -- `:value-source :parsed-text`, with that line itself kept verbatim under `:raw`, and a value XGBoost spelled `inf`/`nan` coming back as `nil` rather than a number. The same line is `cl-gbdt/xgboost:evaluate-one-iteration`'s own primary value at Layer 1, for a caller who wants it without going through the portable API |
| Model slicing | No counterpart at all: LightGBM's C API has nothing that extracts a range of boosting rounds into a new model, so `(backend-supports-p backend :model-slicing)` is `nil` and there is no LightGBM function to call | `cl-gbdt/xgboost:slice-model` (Layer 1, XGBoost-only), over `XGBoosterSlice`. Returns a new booster holding a half-open `[begin, end)` range of the parent's layers, independent of it -- freeing the parent leaves the slice usable. Deliberately not part of the unified API: with no LightGBM counterpart a portable version could only signal for every caller of one backend, or emulate, and emulating is what [the capability model](#asking-a-backend-what-it-can-do) exists to rule out |
| `train`'s `:objective` | **Overrides** any `objective` in `:parameters` -- all five spellings this library honours, `objective_type`, `app`, `application` and `loss` included -- forcing it to `"none"`, since `LGBM_BoosterUpdateOneIterCustom` refuses to run while the booster holds an objective function at all | Never rewrites `:parameters`; a configured objective's own prediction transform stays in effect, so a custom-objective run's `predict :kind :normal` differs from `:raw` there, while LightGBM's are identical. See [Custom objective](#custom-objective) for both |
| `backend-version` | Always `nil` -- LightGBM's C API has no version entry point | A `"MAJOR.MINOR.PATCH"` string, e.g. `"3.3.0"` |
| Untested-version warning | Never signalled -- there is no version to compare, so `open-backend` never checks one | `open-backend` signals `untested-backend-version` (a warning, not an error) when the loaded version falls outside the recorded supported range |

`src/version.lisp` records that supported range as two distinguishable claims: a narrow
*verified* one (the versions the functional suite above has actually run against)
and a wider *inferred* one (the range across which `tools/check-upstream.lisp` confirms cl-gbdt's
imported C functions' declarations are unchanged). The warning gates on the wider inferred
range -- a version different from the exact tested one is the common case for a compatible
caller, not a signal of trouble.

A version matrix (task 4) turned part of each inferred range into a measured one by actually
running the functional suite -- not just comparing headers -- against the range's endpoints.
The counts below are what that suite had when the matrix was measured; it has grown since,
and the rows are left as the measurement recorded them rather than restated against a total
that run never saw:

| library | version | result |
|---|---|---|
| LightGBM | 3.0.0 | not tested -- no aarch64 wheel exists on PyPI for this release, confirmed directly (`pip download lightgbm==3.0.0 --only-binary=:all:` finds no candidate); permanently inferred-only on this platform |
| LightGBM | 4.0.0 | ✅ all 106 assertions pass |
| LightGBM | 4.7.0 (pinned) | ✅ all 106 assertions pass |
| XGBoost | 1.7.0 | ❌ 105 of 106 pass; the ranking round trip fails (see below) |
| XGBoost | 2.0.0 | ✅ all 106 assertions pass |
| XGBoost | 3.3.0 (pinned) | ✅ all 106 assertions pass |

XGBoost 1.7.0 is a real, measured incompatibility, not a gap in coverage: every assertion
`tools/check-upstream.lisp` cannot see -- plain classification and multiclass round trips,
`feature-importance`, save/load, every close-backend guard -- passes unchanged, but
`xgboost-api-ranking-round-trip-respects-group-boundaries` (tests/functional/xgboost-api.lisp)
does not. That test trains a deliberately low-capacity `rank:pairwise` booster and asserts
predictions increase strictly within each query group; at 1.7.0 the first two rows of each
group tie instead. `XGBoosterUpdateOneIter` and `XGDMatrixSetUIntInfo` are both still present
and both still return success -- this is `rank:pairwise`'s internal behavior differing between
releases, the exact kind of break header comparison cannot see because no function's
declaration changed. XGBoost 2.0.0 was tried next and passed everything 3.3.0 does, so the
recorded range's lower bound moved to 2.0.0, not 1.7.0 -- see `*xgboost-version-range*`'s
docstring in `src/version.lisp` for the full account, including why this pulled both the
*verified* and the *inferred* bound up together rather than leaving 1.7.0 covered by a
header-only claim the functional suite had, by then, already disproven.

The matrix runs on Linux x86_64 only, not all three platforms `.github/workflows/test.yml`
already covers for the pinned versions -- see [Continuous integration](#continuous-integration)
for why that is a deliberate restriction, not a gap.

Run together against both backends:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

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
    (cl-gbdt:with-dataset (xgb-ds-grouped (cl-gbdt:make-dataset xgb *matrix* :label *label*
                                             :group '(2 2)))
      (format t "XGBoost  :group accepted; grouped dataset has ~D rows~%"
              (cl-gbdt:dataset-num-rows xgb-ds-grouped)))
    (handler-case (cl-gbdt:make-dataset xgb *matrix* :label *label* :parameters '(:max-bin 3))
      (error (c) (format t "XGBoost  :parameters SIGNALED ~A: ~A~%" (type-of c) c)))

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
XGBoost  :group accepted; grouped dataset has 4 rows
XGBoost  :parameters SIGNALED UNSUPPORTED-ARGUMENT: make-dataset's :parameters is not supported by XGBOOST: XGDMatrixCreateFromDense's config JSON only recognizes missing/nthread/data_split_mode, none of which are LightGBM's dataset-level binning parameters, and the library silently ignores any other key rather than rejecting it.
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
(ql:quickload '(:cl-gbdt :cl-gbdt/xgboost/unified) :silent t)

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

And `evaluation`, whose two rows above are about how each backend produces the numbers
rather than about one backend refusing something. The same call, the same 8 rows, the same
one `:valid-sets` entry, on both backends -- dataset 0 is the training set, dataset 1 is
that validation set, and the metric names are each library's own:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *eval-matrix*
  (make-array '(8 2) :element-type 'double-float
                      :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                           (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                           (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *eval-label*
  (make-array 8 :element-type 'single-float
                 :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defun show (name backend dataset-parameters booster-parameters reference-p)
  (cl-gbdt:with-dataset (train-set (apply #'cl-gbdt:make-dataset backend *eval-matrix*
                                          :label *eval-label* dataset-parameters))
    (cl-gbdt:with-dataset (valid-set (apply #'cl-gbdt:make-dataset backend *eval-matrix*
                                            :label *eval-label*
                                            (append (when reference-p
                                                      (list :reference train-set))
                                                    dataset-parameters)))
      (cl-gbdt:with-booster (booster (cl-gbdt:train backend train-set :num-rounds 5
                                                     :valid-sets (list valid-set)
                                                     :parameters booster-parameters))
        (multiple-value-bind (entries provenance) (cl-gbdt:evaluation booster)
          (format t "~A entries:    ~S~%~A provenance: ~S~%" name entries name provenance))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (show "LightGBM" lgbm
        '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
        '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
          :verbose -1 :metric "binary_logloss,auc")
        t)
  (show "XGBoost " xgb '()
        '(:objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0
          :eval-metric "logloss" :eval-metric "error")
        nil)
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM entries:    ((0 "binary_logloss" 0.35374722486733523d0)
                      (0 "auc" 1.0d0)
                      (1 "binary_logloss" 0.35374722486733523d0)
                      (1 "auc" 1.0d0))
LightGBM provenance: (:VALUE-SOURCE :LIBRARY-DOUBLES)
XGBoost  entries:    ((0 "logloss" 0.4740770012140274d0) (0 "error" 0.0d0)
                      (1 "logloss" 0.4740770012140274d0) (1 "error" 0.0d0))
XGBoost  provenance: (:VALUE-SOURCE :PARSED-TEXT :RAW
                      "[5]	0-logloss:0.47407700121402740	0-error:0.00000000000000000	1-logloss:0.47407700121402740	1-error:0.00000000000000000")
```

Dataset 0 and dataset 1 agree here because the validation set is built over the same rows
as the training set. Nothing names those datasets in the call above: LightGBM knows a
validation set by its index and by nothing else, so the portable API reports the position
the caller supplied it in rather than inventing `"valid_0"`-style names on its own. The
`0-`/`1-` prefixes inside XGBoost's `:raw` line are the names this backend must pass
`XGBoosterEvalOneIter` -- that call demands one per DMatrix -- and are the indices
themselves for exactly that reason.

### Training report

`train` returns two values: the booster, and a `training-report` of the run just
completed. Its `training-report-series` is a list of `training-series`, one per metric per
dataset -- the same (index, metric-name) pairs `evaluation` reports for the trained
booster, in the same order. Each series carries `training-series-index` (0 for the training
set, 1 for the first `:valid-sets` entry, 2 for the second, and so on),
`training-series-metric` (the backend's own name for the metric) and
`training-series-values` (one value per completed iteration, oldest first, `NIL` where a
value could not be read as a real).

`:valid-sets` accepts two element forms, freely mixed in one list: a bare dataset, whose
series carry no name, or a `(name . dataset)` cons, where `name` is a string that becomes
`training-series-name` for every series recorded at that dataset's index. A series always
carries an index; it carries a non-`NIL` name only when the validation set it came from was
given one. The training set is never a `:valid-sets` entry, so its own series are always
index 0 with a `NIL` name -- nothing here invents a name for it, or for a validation set
passed bare. Two validation sets may legitimately share one name; their index, not their
name, is what tells the two apart in the report.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defun show-report (name backend dataset-parameters booster-parameters reference-p)
  (cl-gbdt:with-dataset (train-set (apply #'cl-gbdt:make-dataset backend *eval-matrix*
                                          :label *eval-label* dataset-parameters))
    (cl-gbdt:with-dataset (valid-set (apply #'cl-gbdt:make-dataset backend *eval-matrix*
                                            :label *eval-label*
                                            (append (when reference-p
                                                      (list :reference train-set))
                                                    dataset-parameters)))
      ;; NOT `with-booster': it binds the primary value only, so a caller who wants the
      ;; report -- `train''s secondary value -- uses `multiple-value-bind' and frees the
      ;; booster itself instead.
      (multiple-value-bind (booster report)
          (cl-gbdt:train backend train-set :num-rounds 5
                          ;; A named :valid-sets entry: a bare dataset would leave
                          ;; TRAINING-SERIES-NAME NIL, same as the training set's own.
                          :valid-sets (list (cons "valid" valid-set))
                          :parameters booster-parameters)
        (unwind-protect
             (progn
               (dolist (series (cl-gbdt:training-report-series report))
                 (format t "~A series: index=~S name=~S metric=~S last=~S~%"
                         name (cl-gbdt:training-series-index series)
                         (cl-gbdt:training-series-name series)
                         (cl-gbdt:training-series-metric series)
                         (aref (cl-gbdt:training-series-values series) 4)))
               (format t "~A best-iteration: ~S~%" name
                       (cl-gbdt:training-report-best-iteration report)))
          (cl-gbdt:free-booster booster))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (show-report "LightGBM" lgbm
        '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
        '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
          :verbose -1 :metric "binary_logloss,auc")
        t)
  (show-report "XGBoost " xgb '()
        '(:objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0
          :eval-metric "logloss" :eval-metric "error")
        nil)
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM series: index=0 name=NIL metric="binary_logloss" last=0.35374722486733523d0
LightGBM series: index=0 name=NIL metric="auc" last=1.0d0
LightGBM series: index=1 name="valid" metric="binary_logloss" last=0.35374722486733523d0
LightGBM series: index=1 name="valid" metric="auc" last=1.0d0
LightGBM best-iteration: NIL
XGBoost  series: index=0 name=NIL metric="logloss" last=0.4740770012140274d0
XGBoost  series: index=0 name=NIL metric="error" last=0.0d0
XGBoost  series: index=1 name="valid" metric="logloss" last=0.4740770012140274d0
XGBoost  series: index=1 name="valid" metric="error" last=0.0d0
XGBoost  best-iteration: NIL
```

Dataset 1's series carry the name `"valid"` on both backends; dataset 0's -- the training
set -- stay `NIL` regardless, and so would dataset 1's if `valid-set` had been passed bare
instead of as `(cons "valid" valid-set)`. `training-report-best-iteration`,
`training-report-best-score` and `training-report-early-stopped-p` are all `NIL` above
because that call gave `train` no `:early-stopping` -- see the next section for what fills
them in, and [`:num-iteration :best`](#num-iteration-best) below for what the booster's own
best iteration is then good for.

A malformed `:valid-sets` element signals one of two conditions, kept distinct because they
are different mistakes: a `(name . dataset)` cons whose `name` is not a string signals
`unsupported-argument`, naming `:valid-sets` and the offending element; a cons whose
`dataset` half is not this backend's own kind of handle signals `wrong-backend-reference` --
the same condition a bare wrong-backend dataset already signals elsewhere in this API. Both
are checked before any foreign call. Duplicate names are not one of these mistakes: two
validation sets sharing a name train and report normally, distinguished by index.

#### Stopping early: `:early-stopping`

`train` takes `:early-stopping`, a plist that ends the run once a watched metric stops
improving. All four keys are required, with no default for any of them:

| Key | Meaning |
|---|---|
| `:metric` | A string, the metric to watch, spelled the way this backend spells it in `evaluation` -- LightGBM's `"binary_logloss"`, XGBoost's `"logloss"` |
| `:dataset` | Which dataset's copy of that metric to watch: a string naming a `:valid-sets` entry, or an integer index (0 the training set, N+1 the Nth `:valid-sets` entry). A name matching two entries signals `unsupported-argument` -- two entries may share a name, but a watcher has to watch exactly one, so pass the index instead |
| `:direction` | `:lower-is-better` or `:higher-is-better` |
| `:rounds` | A positive integer: how many consecutive non-improving iterations are tolerated before the run stops |

`:direction` is required, and is never inferred, because neither library's C API exposes
whether a metric improves upward or downward -- there is nothing to read it off, and
guessing from the metric's *name* (a lookup table mapping `"logloss"` to "lower" and
`"auc"` to "higher") would be the same guess with a table in front of it. The caller
already knows which way their own metric goes; this API does not pretend to.

`training-report-best-iteration`, `-best-score` and `-early-stopped-p`, and the returned
booster's own `booster-best-iteration`, are filled by a run given `:early-stopping` -- but
not unconditionally: `:num-rounds` zero or negative never lets the watcher see an iteration
at all, and a run every one of whose watched values came back unreadable (see `evaluation`'s
account of a value the backend reported but could not be parsed) never has a real value to
call best, so `-best-iteration` and `-best-score` stay `NIL` in both cases even though
`-early-stopped-p` can still turn `T` in the second one. `NIL` keeps meaning "not
determined" on the report, exactly as it does with no `:early-stopping` at all, never
"iteration 0".

`:early-stopping` together with `:record-history nil` signals `unsupported-argument`: early
stopping needs the very per-iteration evaluation `:record-history nil` exists to skip, and
reading it costs the same whether one series is watched or every series is recorded, so
there is no cheaper middle path to offer a caller who asks for both.

The example below builds a validation set from the training labels *inverted*, which is a
doc-only trick to force the watched metric to worsen from the very first iteration and
provoke a stop within a handful of rounds -- never something a real validation set is built
from. It also demonstrates [`:num-iteration :best`](#num-iteration-best), covered right
after, in the same run: `predict`, `save-model` and `model-to-string` accept `:best`
wherever they accept `:num-iteration`, an additional value alongside `NIL` (every round)
and an explicit integer, never a new default. `:best` resolves to the booster's own
`booster-best-iteration` before anything else runs, and a booster with no best iteration to
resolve against signals `unsupported-argument` -- see that section for the full contract.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *es-matrix*
  (make-array '(8 2) :element-type 'double-float
                      :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                           (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                           (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *es-label*
  (make-array 8 :element-type 'single-float
                 :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))
(defparameter *es-inverted-label*
  (map '(vector single-float) (lambda (x) (- 1.0 x)) *es-label*))

(defun train-early-stopped (backend dataset-parameters booster-parameters metric reference-p)
  "Return (values BOOSTER REPORT TRAIN-SET VALID-SET). The caller frees all four -- the two
datasets after BOOSTER, since BOOSTER retains both strongly for its own lifetime, exactly
the order `with-booster' nested inside `with-dataset' would enforce above."
  (let* ((train-set (apply #'cl-gbdt:make-dataset backend *es-matrix* :label *es-label*
                            dataset-parameters))
         (valid-set (apply #'cl-gbdt:make-dataset backend *es-matrix*
                            :label *es-inverted-label*
                            (append (when reference-p (list :reference train-set))
                                    dataset-parameters))))
    (multiple-value-bind (booster report)
        (cl-gbdt:train backend train-set :num-rounds 1000
                        :valid-sets (list (cons "valid" valid-set))
                        :early-stopping (list :metric metric :dataset "valid"
                                               :direction :lower-is-better :rounds 3)
                        :parameters booster-parameters)
      (values booster report train-set valid-set))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (multiple-value-bind (booster report train-set valid-set)
      (train-early-stopped lgbm '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1
                                                 :verbose -1))
                            '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1
                              :min-data-in-bin 1 :verbose -1 :metric "binary_logloss")
                            "binary_logloss" t)
    (unwind-protect
         (progn
           (format t "LightGBM num-rounds=~S early-stopped-p=~S best-iteration=~S~%"
                   (cl-gbdt:training-report-num-rounds report)
                   (cl-gbdt:training-report-early-stopped-p report)
                   (cl-gbdt:training-report-best-iteration report))
           (format t "LightGBM predict :num-iteration :best differs from every round: ~S~%"
                   (not (equalp (cl-gbdt:predict booster *es-matrix*)
                                 (cl-gbdt:predict booster *es-matrix* :num-iteration :best))))
           (cl-gbdt:save-model booster "/tmp/lgbm-best.txt" :num-iteration :best)
           (format t "LightGBM save-model :num-iteration :best wrote /tmp/lgbm-best.txt: ~S~%"
                   (and (probe-file "/tmp/lgbm-best.txt") t)))
      (cl-gbdt:free-booster booster)
      (cl-gbdt:free-dataset valid-set)
      (cl-gbdt:free-dataset train-set)))
  (cl-gbdt:close-backend lgbm))

(let ((xgb (cl-gbdt:open-backend :xgboost)))
  (multiple-value-bind (booster report train-set valid-set)
      (train-early-stopped xgb '()
                            '(:objective "binary:logistic" :max-depth 2 :eta 0.5
                              :verbosity 0 :eval-metric "logloss")
                            "logloss" nil)
    (unwind-protect
         (progn
           (format t "XGBoost  num-rounds=~S early-stopped-p=~S best-iteration=~S~%"
                   (cl-gbdt:training-report-num-rounds report)
                   (cl-gbdt:training-report-early-stopped-p report)
                   (cl-gbdt:training-report-best-iteration report))
           ;; XGBoost's save-model asymmetry, unaffected by :best: XGBoosterSaveModel has no
           ;; iteration limit at all -- it always writes every round -- so :best resolves to
           ;; an integer first and then meets the exact `unsupported-argument' check an
           ;; explicit :num-iteration already does. No special case is written around it.
           (handler-case
               (cl-gbdt:save-model booster "/tmp/xgb-best.json" :num-iteration :best)
             (error (c) (format t "XGBoost  save-model :num-iteration :best SIGNALED ~A: ~A~%"
                                 (type-of c) c)))
           ;; The escape hatch: slice to the best iteration first, then save the slice, which
           ;; `save-model' accepts with no :num-iteration at all -- a sliced booster's every
           ;; round already is the range the caller wanted.
           (let ((sliced (cl-gbdt/xgboost:slice-model
                          booster :begin 0 :end (cl-gbdt:booster-best-iteration booster))))
             (unwind-protect
                  (progn
                    (cl-gbdt:save-model sliced "/tmp/xgb-sliced.json")
                    (format t "XGBoost  slice-model to the best iteration then save-model ~
                               wrote /tmp/xgb-sliced.json: ~S~%"
                            (and (probe-file "/tmp/xgb-sliced.json") t)))
               (cl-gbdt:free-booster sliced))))
      (cl-gbdt:free-booster booster)
      (cl-gbdt:free-dataset valid-set)
      (cl-gbdt:free-dataset train-set)))
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM num-rounds=4 early-stopped-p=T best-iteration=1
LightGBM predict :num-iteration :best differs from every round: T
LightGBM save-model :num-iteration :best wrote /tmp/lgbm-best.txt: T
XGBoost  num-rounds=4 early-stopped-p=T best-iteration=1
XGBoost  save-model :num-iteration :best SIGNALED UNSUPPORTED-ARGUMENT: save-model's :num-iteration is not supported by XGBOOST: XGBoosterSaveModel has no iteration limit; every boosted round is saved.
XGBoost  slice-model to the best iteration then save-model wrote /tmp/xgb-sliced.json: T
```

The LightGBM booster kept fitting the training data for three more rounds after its watched
validation metric stopped improving at iteration 1, which is why predicting from its best
iteration alone differs from predicting over the full, unstopped run.

#### `:num-iteration :best`

`predict`, `save-model` and `model-to-string` accept `:best` wherever they accept
`:num-iteration`, demonstrated together with `:early-stopping` in the example just above --
an additional value alongside `NIL` (every round) and an explicit integer, never a new
default: `NIL` keeps meaning "every round" on every booster, including one that has a best
iteration to resolve `:best` against. `feature-importance` also accepts `:num-iteration` but
does not accept `:best`; only the three named above do.

`:best` resolves to the booster's own `booster-best-iteration` -- the same iteration
`training-report-best-iteration` named when `train` was given `:early-stopping` -- before
anything else runs: LightGBM's own `:num-iteration` resolution knows only `NIL` and an
integer, so `:best` must already be one by the time it gets there. A booster with no best
iteration to resolve against -- never trained with `:early-stopping`, or a run that hit one
of the two `NIL` cases the previous section describes -- signals `unsupported-argument`:
the question has no answer for that booster, and this API does not invent one.

**The save-model asymmetry is not smoothed over for `:best`.** LightGBM's `save-model`
honours `:num-iteration`, `:best` included, and writes a file limited to that many trees.
XGBoost's `XGBoosterSaveModel` has no iteration limit at all, so `save-model` there already
signals `unsupported-argument` for any non-`NIL` `:num-iteration`; `:best` resolves to an
integer first and then meets that exact check, with no special case written around it, as
the output above shows. The escape hatch is `cl-gbdt/xgboost:slice-model` (see [the
differences table](#where-the-two-backends-genuinely-differ) and [Backend-specific
packages](#backend-specific-packages) above), also shown above: slice to the best iteration
first, then save the slice.

#### Turning recording off: `:record-history`

Recording is not free, and it is on by default. `train` reads the whole evaluation once per
iteration, for every dataset the booster holds. Measured here over 500 rounds on 2000 rows ×
20 columns with two metrics configured, that roughly **doubled** LightGBM's wall-clock
`train` time -- with and without a validation set -- and added roughly **70-80%** to
XGBoost's with one validation set attached. XGBoost with no validation set stayed inside the
measurement noise, that backend evaluating every dataset in one call rather than one call
each. Treat these as orders of magnitude on one machine: run-to-run variance on the same
code is easily ±15%.

`train` therefore takes `:record-history`, `t` by default:

```lisp
(multiple-value-bind (booster report)
    (cl-gbdt:train backend train-set :num-rounds 500 :record-history nil)
  ;; report is a training-report with no series, over 500 rounds.
  (cl-gbdt:free-booster booster))
```

With `:record-history nil` no evaluation is read at all, and `train` costs what it cost
before it recorded anything -- measured against the commit this branch started from, the two
agree within the noise on both backends. The secondary value is still a `training-report`,
never `nil`, so a caller destructuring two values never has to handle two shapes: its
`training-report-series` is empty and its `training-report-num-rounds` is the run's length,
exactly as a run with no metric configured reports.

Recording also decides, on XGBoost, which `:valid-sets` entries `train` accepts at all. An
unlabelled DMatrix is the case this was found through: `XGBoosterUpdateOneIter` trains on it
happily, while `XGBoosterEvalOneIter` refuses it (`label and prediction size not match`). So
with recording on -- the default -- such an entry now fails `train` itself with
`foreign-call-error`, where before this branch it trained normally and failed only a later
`evaluation` call. The general rule is that any configuration whose evaluation path errors
while its update path does not now fails the whole run. `:record-history nil` never reaches
the evaluation path and restores the older behaviour. LightGBM tolerates the same input,
recording finite values, so this is XGBoost-specific in practice.

### Sparse input: CSR matrices

`make-csr-matrix` builds a `csr-matrix`, the one sparse form this API accepts. `make-dataset`
and `predict` each take one wherever they take a dense matrix -- neither generic's lambda list
changed to allow it -- and the dataset a `csr-matrix` builds is an ordinary dataset that
nothing downstream, `train` included, can distinguish from a densely-built one.

`INDPTR`, `INDICES` and `VALUES` may each be **any sequence**, and are validated and coerced
once, at construction: `INDPTR` and `INDICES` to `(simple-array (signed-byte 32) (*))`,
`VALUES` to `(simple-array double-float (*))`. A backend method therefore only has to pin
what the struct already holds. The four slots are **read-only**: `csr-matrix-indptr` and its
three siblings are readers with no `setf` expander, so that construction-time validation
cannot be undone afterwards. **`NUM-COLUMNS` is required and never inferred** from the
largest index `INDICES` happens to hold: a matrix's declared width and its largest stored
index are different facts -- the trailing columns can legitimately hold nothing at all, and
the stored indices cannot tell that apart from a matrix that simply is not that wide. Only
the caller knows the first. `NUM-ROWS` is not a slot; it is `(1- (length indptr))`, which
`csr-matrix-num-rows` returns, so there is no second copy of the row count to keep in sync.

A malformed matrix signals `dimension-mismatch`, and a value that cannot be coerced signals
`unsupported-element-type` -- both from `make-csr-matrix` itself, next to the mistake, rather
than from a foreign call several frames later:

```lisp
(ql:quickload :cl-gbdt :silent t)

;; Four rows, four columns. Row 2 stores nothing at all -- a repeated INDPTR entry is an
;; empty row, which is legal. INDPTR, INDICES and VALUES may each be any sequence.
(let ((csr (cl-gbdt:make-csr-matrix :indptr '(0 2 3 3 5)
                                    :indices '(0 3 1 0 2)
                                    :values '(1.0 2.0 3.0 4.0 5.0)
                                    :num-columns 4)))
  (format t "indptr:      ~S~%  ~S~%" (cl-gbdt:csr-matrix-indptr csr)
          (type-of (cl-gbdt:csr-matrix-indptr csr)))
  (format t "indices:     ~S~%" (cl-gbdt:csr-matrix-indices csr))
  (format t "values:      ~S~%  ~S~%" (cl-gbdt:csr-matrix-values csr)
          (type-of (cl-gbdt:csr-matrix-values csr)))
  (format t "num-columns: ~S~%" (cl-gbdt:csr-matrix-num-columns csr))
  (format t "num-rows:    ~S~%" (cl-gbdt:csr-matrix-num-rows csr)))

(dolist (bad (list (list :indptr '(0 2) :indices '(0 3) :values '(1.0) :num-columns 4)
                   (list :indptr '(0 1) :indices '(9) :values '(1.0) :num-columns 4)
                   (list :indptr '(0 1) :indices '(0) :values '("x") :num-columns 4)))
  (handler-case (apply #'cl-gbdt:make-csr-matrix bad)
    (error (c) (format t "SIGNALED ~A~%  ~A~%" (type-of c) c))))
```

Output:

```
indptr:      #(0 2 3 3 5)
  (SIMPLE-ARRAY (SIGNED-BYTE 32) (5))
indices:     #(0 3 1 0 2)
values:      #(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0)
  (SIMPLE-ARRAY DOUBLE-FLOAT (5))
num-columns: 4
num-rows:    4
SIGNALED DIMENSION-MISMATCH
  Dimension mismatch. Expected: INDICES and VALUES to have the same length, got: (2
                                                                                  1)
SIGNALED DIMENSION-MISMATCH
  Dimension mismatch. Expected: a column index in [0, 4), got: 9
SIGNALED UNSUPPORTED-ELEMENT-TYPE
  Element type (SIMPLE-ARRAY CHARACTER (1)) is not supported. Use DOUBLE-FLOAT or SINGLE-FLOAT.
```

Sparse input is a capability, `:sparse-input`, true on both vendored backends. Both
`make-dataset` and `predict` re-check it for themselves rather than trusting the caller to
have asked first, and signal `capability-unavailable` when it is false -- never a silent
conversion to a dense matrix, exactly as [the capability
model](#asking-a-backend-what-it-can-do) requires everywhere else. The dataset's feature
count is the declared `NUM-COLUMNS`; a `NUM-COLUMNS` that disagrees with the booster's own
feature count at prediction time is the library's own refusal to report, reaching the caller
as `foreign-call-error` in that library's words rather than as a check invented here.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *dense*
  (make-array '(8 2) :element-type 'double-float
                      :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0)
                                           (0.0d0 2.0d0) (0.0d0 3.0d0)
                                           (5.0d0 0.0d0) (5.0d0 1.0d0)
                                           (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *label*
  (make-array 8 :element-type 'single-float
                 :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

;; The same eight rows as CSR, every element stored explicitly -- zeros included. See
;; "An absent entry is not a zero" below for why dropping them is not the same matrix.
(defparameter *sparse*
  (cl-gbdt:make-csr-matrix
   :indptr '(0 2 4 6 8 10 12 14 16)
   :indices '(0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1)
   :values '(0 0 0 1 0 2 0 3 5 0 5 1 5 2 5 3)
   :num-columns 2))

(defun show-sparse (name backend dataset-parameters booster-parameters)
  (format t "~A backend-supports-p :sparse-input => ~S~%"
          name (cl-gbdt:backend-supports-p backend :sparse-input))
  (cl-gbdt:with-dataset (dataset (apply #'cl-gbdt:make-dataset backend *sparse*
                                        :label *label* dataset-parameters))
    (format t "~A dataset from the csr-matrix: rows=~D features=~D~%"
            name (cl-gbdt:dataset-num-rows dataset)
            (cl-gbdt:dataset-num-features dataset))
    (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 10
                                                   :parameters booster-parameters))
      (dolist (kind '(:normal :raw :contrib :leaf-index))
        ;; The dense call first, so its success is on the record before the sparse one
        ;; is attempted -- materialising the rows densely is the documented workaround.
        (let ((dense (cl-gbdt:predict booster *dense* :kind kind)))
          (handler-case
              (format t "~A ~S: dense ok; sparse equals dense => ~S~%" name kind
                      (equalp (cl-gbdt:predict booster *sparse* :kind kind) dense))
            ;; XGBoost's message carries a multi-line stack trace; line 1 is the refusal.
            (error (c) (let ((text (princ-to-string c)))
                         (format t "~A ~S: dense ok; sparse SIGNALED ~A~%  ~A~%" name kind
                                 (type-of c)
                                 (subseq text 0 (position #\Newline text)))))))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (show-sparse "LightGBM" lgbm
               '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
               '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
                 :verbose -1))
  (show-sparse "XGBoost " xgb '()
               '(:objective "binary:logistic" :max-depth 2 :verbosity 0))
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM backend-supports-p :sparse-input => T
LightGBM dataset from the csr-matrix: rows=8 features=2
LightGBM :NORMAL: dense ok; sparse equals dense => T
LightGBM :RAW: dense ok; sparse equals dense => T
LightGBM :CONTRIB: dense ok; sparse equals dense => T
LightGBM :LEAF-INDEX: dense ok; sparse equals dense => T
XGBoost  backend-supports-p :sparse-input => T
XGBoost  dataset from the csr-matrix: rows=8 features=2
XGBoost  :NORMAL: dense ok; sparse equals dense => T
XGBoost  :RAW: dense ok; sparse equals dense => T
XGBoost  :CONTRIB: dense ok; sparse SIGNALED FOREIGN-CALL-ERROR
  XGBoosterPredictFromCSR returned -1: [17:33:19] /__w/xgboost/xgboost/src/learner.cc:1264: Unsupported prediction type:2
XGBoost  :LEAF-INDEX: dense ok; sparse SIGNALED FOREIGN-CALL-ERROR
  XGBoosterPredictFromCSR returned -1: [17:33:19] /__w/xgboost/xgboost/src/learner.cc:1264: Unsupported prediction type:6
```

The bracketed time in XGBoost's two messages is XGBoost's own wall-clock stamp, so those two
lines are the only part of this output that differs from one run to the next.

#### `predict`'s KIND on a `csr-matrix`: XGBoost serves two of the four

`make-dataset` takes a `csr-matrix` identically on both backends. `predict` does not:

| `predict`'s KIND | dense, both backends | `csr-matrix`, LightGBM | `csr-matrix`, XGBoost |
|---|---|---|---|
| `:normal` | works | works | works |
| `:raw` | works | works | works |
| `:contrib` | works | works | **`foreign-call-error`** |
| `:leaf-index` | works | works | **`foreign-call-error`** |

`XGBoosterPredictFromCSR` is not the CSR spelling of `XGBoosterPredictFromDMatrix`: the
vendored header (`ffi-spec/xgboost/include/xgboost/c_api.h`) documents it as *inplace
prediction from CPU CSR matrix*, a different code path, and `learner.cc` refuses the prediction
type codes `:contrib` and `:leaf-index` map to -- `2` and `6`, the two numbers the messages
above name -- while accepting `:normal`'s `0` and `:raw`'s `1`. Those refusals reach the caller
as `foreign-call-error` naming the call that failed. Nothing here emulates around it: routing
those two KINDs through a transient DMatrix instead would mean this wrapper, not the library,
deciding which C entry point a KIND gets, and would leave the very symbol `:sparse-input`
declares for prediction unused. LightGBM's `LGBM_BoosterPredictForCSR` has no such restriction
and serves all four, with the same values the dense path produces.

**The workaround is to materialise the rows as a dense matrix** -- a 2D `double-float` or
`single-float` array, or a `foreign-matrix` -- and predict on that, which is what the block
above does for every KIND before trying the sparse call. Note what the workaround is *not*:
`predict`'s MATRIX argument accepts a 2D array, a `foreign-matrix` or a `csr-matrix`, and
**a dataset is not one of them**, so building the rows into a dataset with `make-dataset`
leads to no prediction at all. Materialising is a real cost -- avoiding it is the reason to
pass a `csr-matrix` in the first place -- and this API charges it rather than hiding it.

#### An absent entry is not a zero, and the two libraries disagree about it

An entry a `csr-matrix` does not store is *absent*, and the two libraries read absence
differently: **LightGBM reads an absent entry as `0.0`** (its own `zero_as_missing` is off by
default), while **XGBoost reads one as missing**. No config key changes either -- that is
what CSR means to each library. This matters because dropping zeros is exactly what a CSR
conversion normally does, and the disagreement changes results silently rather than erroring:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

;; *dense*, *label* and *sparse* as defined in the previous block.
;; The same eight rows with the zeros dropped -- the conversion a sparse format normally
;; performs. Row 0 is (0.0 0.0) and now stores nothing at all.
(defparameter *zeros-dropped*
  (cl-gbdt:make-csr-matrix
   :indptr '(0 0 1 2 3 4 6 8 10)
   :indices '(1 1 1 0 0 1 0 1 0 1)
   :values '(1 2 3 5 5 1 5 2 5 3)
   :num-columns 2))

(defun first-column (predictions)
  (loop :for row :below (array-dimension predictions 0)
        :collect (aref predictions row 0)))

(defun compare (name backend dataset-parameters booster-parameters)
  (flet ((train-on (matrix)
           (cl-gbdt:with-dataset (dataset (apply #'cl-gbdt:make-dataset backend matrix
                                                 :label *label* dataset-parameters))
             (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 10
                                                            :parameters booster-parameters))
               (first-column (cl-gbdt:predict booster *dense*))))))
    (let ((stored (train-on *sparse*))
          (dropped (train-on *zeros-dropped*)))
      (format t "~A every element stored: ~{~,4F~^ ~}~%" name stored)
      (format t "~A zeros dropped:        ~{~,4F~^ ~}~%" name dropped)
      (format t "~A the two agree: ~S~%" name (equal stored dropped)))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (compare "LightGBM" lgbm
           '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
           '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
             :verbose -1))
  (compare "XGBoost " xgb '()
           '(:objective "binary:logistic" :max-depth 2 :verbosity 0))
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM every element stored: 0.1793 0.1793 0.1793 0.1793 0.8207 0.8207 0.8207 0.8207
LightGBM zeros dropped:        0.1793 0.1793 0.1793 0.1793 0.8207 0.8207 0.8207 0.8207
LightGBM the two agree: T
XGBoost  every element stored: 0.4256 0.4256 0.4256 0.4256 0.5744 0.5744 0.5744 0.5744
XGBoost  zeros dropped:        0.5744 0.5744 0.5744 0.5744 0.5744 0.5744 0.5744 0.5744
XGBoost  the two agree: NIL
```

Two matrices describing the same eight rows, differing only in whether the zeros are stored.
LightGBM's two boosters agree to the last digit, which is the absent-entry-is-`0.0` reading
demonstrated rather than asserted. XGBoost's do not: dropping the zeros took feature 0 away
from all four class-0 rows, and the run that trained on what was left no longer separates the
two classes at all -- every row comes back with the positive class's value. Nothing signalled;
the numbers simply changed.

So a `csr-matrix` is not a portable compression of a dense matrix. It is portable when every
element is stored -- as `*sparse*` above does, and as the functional suite's `dense-to-csr`
helper does for exactly this reason -- and it means two different things when entries are
omitted. Omit them when *missing* is what you mean and you are on XGBoost, or when `0.0` is
what you mean and you are on LightGBM; store them when the same matrix has to mean the same
thing on both. Both halves are asserted per backend by the functional suite's
`an-omitted-entry-is-zero-to-lightgbm-and-missing-to-xgboost`, on a fixture that also sends
each library a row storing nothing at all.

#### Why CSR only, and not CSC

XGBoost has `XGDMatrixCreateFromCSC` -- it is bound in `src/xgboost/c-api.lisp` like every
other emitted function -- but there is **no `XGBoosterPredictFromCSC`** anywhere in its C API;
`XGBoosterPredictFromCSR` is the only sparse prediction entry point it offers. Supporting CSC
would therefore put a format into this API that a caller could build a dataset from and then
not predict with: `make-dataset` would take it, `predict` would refuse it, and the refusal
would be a property of one backend rather than of the format. LightGBM does have both
(`LGBM_DatasetCreateFromCSC` and `LGBM_BoosterPredictForCSC`), which would make the gap
backend-specific in a way nothing else in the unified API is. CSR is the format both libraries
can do both halves of, so CSR is the format this API takes.

### Missing values

`make-dataset` and `predict` both take `:missing`, the value in the caller's own matrix that
means *missing* -- the datum a caller wrote in place of one they do not have, such as the
`-999.0` a CSV convention often uses. `:missing` is a `real` or `NIL`; anything else signals
`unsupported-argument` naming `:missing` and the backend. `NIL`, the default on both
operations, is exactly today's behaviour: the wrapper sends `"missing":NaN` unconditionally,
the same as every call made before either operation took this argument at all. This applies
identically whether MATRIX is dense or a `csr-matrix` -- `make-dataset`'s own `:missing`
reaches the same creation-config key either way.

A non-`NIL` `:missing` needs the `:missing-value` capability, answerable through
`backend-supports-p`, and **both operations re-check it themselves** rather than trusting a
caller who asked first. XGBoost declares it -- the sentinel is a key its creation and
prediction config JSONs already read -- and LightGBM does not: its C API has no
missing-value key at all, so `:missing` there signals `capability-unavailable` for *any*
non-`NIL` value, a `NaN` included, even though a `NaN` is in fact what LightGBM's own
ingestion path already treats as missing. A capability whose meaning depended on which value
was passed could not be stated by `backend-supports-p` at all -- it answers about the
backend, and never sees the argument. The capability gate also fires *first*, by design: a
non-`real` `:missing` on LightGBM signals `capability-unavailable`, not
`unsupported-argument` -- the renderer's own type check is never reached there, since only a
backend that passed the gate has a renderer left to reach.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *mv-matrix*
  (make-array '(8 3) :element-type 'double-float
                      :initial-contents '((0.0d0 1.0d0 2.0d0) (0.0d0 2.0d0 1.0d0)
                                           (0.0d0 1.0d0 2.0d0) (0.0d0 2.0d0 1.0d0)
                                           (5.0d0 1.0d0 2.0d0) (5.0d0 2.0d0 1.0d0)
                                           (5.0d0 1.0d0 2.0d0) (5.0d0 2.0d0 1.0d0))))
(defparameter *mv-label*
  (make-array 8 :element-type 'single-float
                 :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))
;; Column 0 alone carries the class -- 0.0 for the first four rows, 5.0 for the last four --
;; while columns 1 and 2 repeat the same two values on both halves and carry nothing. Row 7 is
;; positive-class, so punching its column-0 cell takes away the only signal that row has.
(defparameter *mv-sentinel* -999.0d0)

(defun mv-holed (&optional (value *mv-sentinel*) (row 7))
  "*MV-MATRIX*, with ROW's column 0 replaced by VALUE."
  (let ((matrix (make-array '(8 3) :element-type 'double-float)))
    (dotimes (r 8) (dotimes (c 3) (setf (aref matrix r c) (aref *mv-matrix* r c))))
    (setf (aref matrix row 0) value)
    matrix))

(defun mv-quiet-nan ()
  "A quiet double-float NaN, built from its bits so no arithmetic can trap."
  (sb-kernel:make-double-float -524288 0))

(defun mv-train-predict (xgb matrix &key missing)
  "Train an XGBoost booster on MATRIX/*MV-LABEL* and return its row-7 prediction on the
clean *MV-MATRIX*."
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb matrix :label *mv-label*
                                                        :missing missing))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                      :parameters '(:objective "binary:logistic" :max-depth 2
                                                    :eta 0.5 :verbosity 0)))
      (aref (cl-gbdt:predict booster *mv-matrix*) 7 0))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (format t "LightGBM backend-supports-p :missing-value => ~S~%"
          (cl-gbdt:backend-supports-p lgbm :missing-value))
  (format t "XGBoost  backend-supports-p :missing-value => ~S~%"
          (cl-gbdt:backend-supports-p xgb :missing-value))

  ;; LightGBM signals regardless of the value -- even a NaN, which is in fact what its own
  ;; ingestion path already treats as missing -- because its C API has no missing-value key at
  ;; all: the capability gate fires before the value is even looked at.
  (dolist (value (list *mv-sentinel* (mv-quiet-nan)))
    (handler-case (cl-gbdt:make-dataset lgbm (mv-holed value) :label *mv-label* :missing value
                                         :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                        :verbose -1))
      (error (c) (format t "LightGBM make-dataset :missing ~A SIGNALED ~A: ~A~%"
                          (if (sb-ext:float-nan-p value) "<NaN>" value) (type-of c) c))))

  ;; The gate fires first even for a non-real :missing: LightGBM never reaches the renderer's
  ;; own type check at all, so this is CAPABILITY-UNAVAILABLE too, not UNSUPPORTED-ARGUMENT.
  (handler-case (cl-gbdt:make-dataset lgbm (mv-holed) :label *mv-label* :missing "-999.0"
                                       :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                      :verbose -1))
    (error (c) (format t "LightGBM make-dataset :missing \"-999.0\" SIGNALED ~A: ~A~%"
                        (type-of c) c)))

  ;; XGBoost provides the capability, so a non-real :missing reaches the renderer's own check
  ;; instead of the capability gate.
  (handler-case (cl-gbdt:make-dataset xgb (mv-holed) :label *mv-label* :missing "-999.0")
    (error (c) (format t "XGBoost  make-dataset :missing \"-999.0\" SIGNALED ~A: ~A~%"
                        (type-of c) c)))

  ;; :missing selects a sentinel VALUE, not a policy: it changes what the model learns.
  (format t "XGBoost  row 7, trained with :missing ~S: ~S~%" *mv-sentinel*
          (mv-train-predict xgb (mv-holed) :missing *mv-sentinel*))
  (format t "XGBoost  row 7, trained with that same cell read literally: ~S~%"
          (mv-train-predict xgb (mv-holed)))

  ;; The wrapper renders the sentinel itself rather than letting the Lisp printer choose: a
  ;; bare `princ' of a double would emit "1.0d-5", and XGBoost's JSON config parser rejects
  ;; that exponent marker outright. A caller may still pass a `d0'-marked double; only what
  ;; reaches the library is affected.
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb *mv-matrix* :label *mv-label*
                                                        :missing 1.0d-5))
    (format t "XGBoost  make-dataset :missing 1.0d-5 (a Lisp exponent marker) works: rows=~S~%"
            (cl-gbdt:dataset-num-rows dataset)))

  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM backend-supports-p :missing-value => NIL
XGBoost  backend-supports-p :missing-value => T
LightGBM make-dataset :missing -999.0d0 SIGNALED CAPABILITY-UNAVAILABLE: LIGHTGBM does not provide :MISSING-VALUE in the library that is loaded.
LightGBM make-dataset :missing <NaN> SIGNALED CAPABILITY-UNAVAILABLE: LIGHTGBM does not provide :MISSING-VALUE in the library that is loaded.
LightGBM make-dataset :missing "-999.0" SIGNALED CAPABILITY-UNAVAILABLE: LIGHTGBM does not provide :MISSING-VALUE in the library that is loaded.
XGBoost  make-dataset :missing "-999.0" SIGNALED UNSUPPORTED-ARGUMENT: :missing is not supported by XGBOOST: the value that means missing must be a real number, or NIL for the backend's own default.
XGBoost  row 7, trained with :missing -999.0d0: 0.622459352016449d0
XGBoost  row 7, trained with that same cell read literally: 0.5d0
XGBoost  make-dataset :missing 1.0d-5 (a Lisp exponent marker) works: rows=8
```

`:missing` selects a sentinel *value*, not a policy. It does not turn missing-value handling
on or off, and it does not make `0.0` mean missing -- LightGBM's `use_missing` and
`zero_as_missing` flags are unchanged by any of this, and stay exactly where they were,
reachable through `make-dataset`'s `:parameters`. The last line of the output above is a
separate, JSON-rendering fact worth knowing on its own: XGBoost's config parser rejects a
Lisp double's own exponent marker outright (`1.0d-5` fails with `json.cc:409: Expecting:
","`, measured against the vendored library) but accepts `1.0e-5`. The wrapper renders
`:missing` itself for exactly this reason, so a caller may still write `1.0d-5` and have it
reach the library correctly.

#### `predict`'s own `:missing`

`predict` takes the identical argument, checked against the `:missing-value` capability
**separately** from `make-dataset`'s own check -- the two operations reach two different
config sites, and a backend could in principle gate one and not the other. A dense matrix's
sentinel becomes a key in the transient DMatrix's own *creation* config, the same one
`make-dataset` fills; a `csr-matrix`'s sentinel goes into `XGBoosterPredictFromCSR`'s
*inplace predict* config instead, since that call builds no DMatrix at all -- shown below
with `:kind :normal`, one of the two kinds XGBoost's sparse `predict` serves at all (see
[Sparse input](#sparse-input-csr-matrices) above). Both config sites are demonstrated first,
on one booster trained with no `:missing` anywhere, so nothing about how the model was
trained can account for what changes; the single-precision measurement that follows trains a
second booster, for a reason its own comment explains:

```lisp
(defparameter *mv-narrowing-sentinel* 16777217.0d0
  "Not exactly representable in single-float: 16777217 is 2^24+1, and single-float spacing at
2^24 is 2, so this narrows to 16777216.0.")
(defparameter *mv-shared-float32* 16777216.0d0
  "A different double-float from *MV-NARROWING-SENTINEL*, but the same value once both are
narrowed to single-float.")
(defparameter *mv-own-float32* 16777224.0d0
  "2^24+8, a multiple of single-float's spacing there, so this is exactly representable and is
therefore its own single-float, distinct from *MV-NARROWING-SENTINEL*'s.")

(defun mv-csr (matrix)
  "MATRIX as a `cl-gbdt:csr-matrix' with every element stored explicitly."
  (let* ((rows (array-dimension matrix 0))
         (cols (array-dimension matrix 1))
         (indptr (make-array (1+ rows)))
         (indices (make-array (* rows cols)))
         (values (make-array (* rows cols)))
         (pos 0))
    (dotimes (r rows)
      (setf (aref indptr r) pos)
      (dotimes (c cols)
        (setf (aref indices pos) c)
        (setf (aref values pos) (aref matrix r c))
        (incf pos)))
    (setf (aref indptr rows) pos)
    (cl-gbdt:make-csr-matrix :indptr indptr :indices indices :values values :num-columns cols)))

(let ((xgb (cl-gbdt:open-backend :xgboost)))
  ;; predict's own :missing, re-checked separately from make-dataset's -- trained on the CLEAN
  ;; matrix, so nothing about how the model was trained can account for what follows.
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb *mv-matrix* :label *mv-label*))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                      :parameters '(:objective "binary:logistic" :max-depth 2
                                                    :eta 0.5 :verbosity 0)))
      (format t "predict row 7, :missing ~S on the holed matrix: ~S~%" *mv-sentinel*
              (aref (cl-gbdt:predict booster (mv-holed) :missing *mv-sentinel*) 7 0))
      (format t "predict row 7, that same holed matrix read literally: ~S~%"
              (aref (cl-gbdt:predict booster (mv-holed)) 7 0))
      ;; The other config site: a csr-matrix's :missing reaches XGBoosterPredictFromCSR's own
      ;; inplace predict config rather than a transient DMatrix's creation config.
      (format t "predict on a csr-matrix, :missing ~S: ~S~%" *mv-sentinel*
              (aref (cl-gbdt:predict booster (mv-csr (mv-holed)) :kind :normal
                                     :missing *mv-sentinel*) 7 0))
      (format t "predict on that csr-matrix, the same cell read literally: ~S~%"
              (aref (cl-gbdt:predict booster (mv-csr (mv-holed)) :kind :normal) 7 0))))

  ;; Single precision: XGBoost gives every split a default direction for a value it reads as
  ;; missing. A booster trained on the clean fixture sends both 2^24-sized values the same
  ;; direction it already sends a genuine missing value, so it cannot tell the two readings
  ;; apart below. Training instead with a NaN hole in a NEGATIVE-class row (row 3, unlike row
  ;; 7's positive class) teaches a default direction the two readings do separate on.
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb (mv-holed (mv-quiet-nan) 3)
                                                        :label *mv-label*))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                      :parameters '(:objective "binary:logistic" :max-depth 2
                                                    :eta 0.5 :verbosity 0)))
      (format t "predict row 7, :missing ~S vs a stored ~S (shares its float32): ~S~%"
              *mv-narrowing-sentinel* *mv-shared-float32*
              (aref (cl-gbdt:predict booster (mv-holed *mv-shared-float32*)
                                     :missing *mv-narrowing-sentinel*) 7 0))
      (format t "predict row 7, :missing ~S vs a stored ~S (its own float32): ~S~%"
              *mv-narrowing-sentinel* *mv-own-float32*
              (aref (cl-gbdt:predict booster (mv-holed *mv-own-float32*)
                                     :missing *mv-narrowing-sentinel*) 7 0))
      (format t "predict row 7, a stored NaN, for comparison: ~S~%"
              (aref (cl-gbdt:predict booster (mv-holed (mv-quiet-nan))) 7 0))))

  (cl-gbdt:close-backend xgb))
```

Output:

```
predict row 7, :missing -999.0d0 on the holed matrix: 0.622459352016449d0
predict row 7, that same holed matrix read literally: 0.3775406777858734d0
predict on a csr-matrix, :missing -999.0d0: 0.622459352016449d0
predict on that csr-matrix, the same cell read literally: 0.3775406777858734d0
predict row 7, :missing 1.6777217d7 vs a stored 1.6777216d7 (shares its float32): 0.3775406777858734d0
predict row 7, :missing 1.6777217d7 vs a stored 1.6777224d7 (its own float32): 0.622459352016449d0
predict row 7, a stored NaN, for comparison: 0.3775406777858734d0
```

The last three lines are the single-precision fact: XGBoost compares `:missing` against the
data at **single precision**, whatever the matrix's own element type. `16777217.0d0` is
2^24 + 1, one past single-float's spacing of 2 at that magnitude, so it narrows to
`16777216.0`; a stored `16777216.0d0` -- a genuinely different `double-float` -- shares that
narrowing and so reads as missing, while a stored `16777224.0d0` -- 2^24 + 8, itself exactly
representable in `single-float` -- does not. Two `double-float`s that round to the same
`single-float` therefore both count as missing against a sentinel that narrows to it.
Measured directly at the raw XGBoost level too, over the identical 24-cell fixture, before
either functional test in `tests/functional/missing-value.lisp` existed:
`XGDMatrixNumNonMissing` keeps 22 of the 24 entries when the sentinel matches the
shared-float32 datum, and all 24 when it does not -- the same distinction the predictions
above show, at the level a caller actually observes it.

#### Training and prediction sentinels are not tied together

Nothing connects the `:missing` a dataset was built with to the `:missing` a later `predict`
call names. XGBoost does not record a dataset's sentinel on the booster trained from it, and
none is written into a saved model either, so predicting with a *different* sentinel than
training used -- or with none at all -- is a call the library accepts and never reports.
Keeping the two consistent is the caller's own responsibility; nothing here detects that they
are not:

```lisp
;; Nothing ties predict's :missing to the sentinel :missing used at training time. XGBoost does
;; not record a dataset's sentinel on the booster trained from it, and none is written into a
;; saved model either -- so a caller who predicts with a DIFFERENT sentinel than training used,
;; or with none at all, gets no error: just silently different numbers.
(let ((xgb (cl-gbdt:open-backend :xgboost)))
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb (mv-holed -999.0d0) :label *mv-label*
                                                        :missing -999.0d0))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                      :parameters '(:objective "binary:logistic" :max-depth 2
                                                    :eta 0.5 :verbosity 0)))
      (format t "trained with :missing -999.0d0; predict :missing -999.0d0 (matches): ~S~%"
              (aref (cl-gbdt:predict booster (mv-holed -999.0d0) :missing -999.0d0) 7 0))
      (format t "same booster; predict :missing -1.0d0 (a DIFFERENT sentinel; no error): ~S~%"
              (aref (cl-gbdt:predict booster (mv-holed -999.0d0) :missing -1.0d0) 7 0))
      (format t "same booster; predict with no :missing at all (no error either): ~S~%"
              (aref (cl-gbdt:predict booster (mv-holed -999.0d0)) 7 0))))
  (cl-gbdt:close-backend xgb))
```

Output:

```
trained with :missing -999.0d0; predict :missing -999.0d0 (matches): 0.622459352016449d0
same booster; predict :missing -1.0d0 (a DIFFERENT sentinel; no error): 0.3775406777858734d0
same booster; predict with no :missing at all (no error either): 0.3775406777858734d0
```

The booster above was trained with `:missing -999.0d0`. Asking `predict` for the matching
sentinel reads row 7 as missing, exactly as training did. Asking for `-1.0d0` instead -- a
sentinel training never used -- signals nothing at all; row 7 is simply read literally, the
same result omitting `:missing` from `predict` entirely already gives. A caller who trains
with one sentinel and predicts with another, or forgets `:missing` on one side, gets a
booster and a prediction that both ran without complaint, and numbers that silently do not
mean what the caller intended.

### Categorical features

`make-dataset` takes `:categorical-features`, a list of 0-based column indices naming which
columns of the caller's own matrix hold *categories* rather than *quantities* -- so a split on
one of them partitions the category set instead of thresholding an ordinal that has no order.
`NIL`, the default, is exactly today's behaviour: every column is read as a quantity, the same
as every call made before the argument existed.

Each backend renders the list its own way. XGBoost attaches it to the finished DMatrix as the
`"feature_type"` field -- `"c"` for each named column, `"q"` for every other -- through the
same `XGDMatrixSetStrFeatureInfo` call `:feature-names` already uses, under a different key.
LightGBM instead composes a `categorical_feature` entry into the parameter string that builds
the dataset -- see [LightGBM: `categorical_feature` and its four
aliases](#lightgbm-categorical_feature-and-its-four-aliases) below for what that means when a
caller also writes the key by hand.

`predict` takes no such argument at all, on either backend. A booster trained from a dataset
built with categorical columns predicts correctly from a plain matrix regardless -- measured
below -- because the trained trees already carry the category sets they split on. XGBoost in
particular records nothing about which columns were categorical on the booster itself: a model
it saves carries an empty `"feature_types":[]`, the same field a model trained with no
categorical column at all would save.

`:categorical-features` needs the `:categorical-features` capability, answerable through
`backend-supports-p` and true on both vendored backends, and **`make-dataset` re-checks it
itself** rather than trusting a caller who asked first, exactly as [the capability
model](#asking-a-backend-what-it-can-do) requires everywhere else. It applies identically
whether the caller's matrix is dense or a `csr-matrix` -- the column count checked against is
the matrix's own, and for a `csr-matrix` that is its **declared** `NUM-COLUMNS`, not the
largest index actually stored. An index that is not an integer, is negative, is at or beyond
the matrix's last column, or was named more than once, signals `unsupported-argument` naming
`:categorical-features` and the backend, from `make-dataset` itself rather than from either
library:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

;; Six categories in column 0, alternating class by category -- 0, 2 and 4 positive, 1, 3 and 5
;; negative -- four rows apiece, 24 rows in all. No threshold on the ordinal 0..5 separates an
;; alternating pattern; a categorical split choosing the subset {0, 2, 4} does. Column 1 is
;; noise: it alternates independently of the class and carries no signal.
;; Taken from tests/functional/categorical-features.lisp's own fixture and parameters.
(defparameter *num-categories* 6)
(defparameter *rows-per-category* 4)

(defun category-matrix ()
  (let* ((rows (* *num-categories* *rows-per-category*))
         (matrix (make-array (list rows 2) :element-type 'double-float)))
    (dotimes (row rows)
      (setf (aref matrix row 0) (coerce (floor row *rows-per-category*) 'double-float))
      (setf (aref matrix row 1) (coerce (mod row 2) 'double-float)))
    matrix))

(defun category-labels ()
  (let* ((rows (* *num-categories* *rows-per-category*))
         (label-vector (make-array rows :element-type 'single-float)))
    (dotimes (row rows)
      (setf (aref label-vector row)
            (if (evenp (floor row *rows-per-category*)) 1.0 0.0)))
    label-vector))

(defun category-means (predictions)
  (loop :for category :below *num-categories*
        :collect (/ (loop :for row :from (* category *rows-per-category*)
                            :below (* (1+ category) *rows-per-category*)
                          :sum (row-major-aref predictions row))
                    *rows-per-category*)))

(defun demo (name backend dataset-parameters booster-parameters)
  (format t "~A backend-supports-p :categorical-features => ~S~%"
          name (cl-gbdt:backend-supports-p backend :categorical-features))
  (flet ((run (categorical-features)
           (cl-gbdt:with-dataset
               (dataset (apply #'cl-gbdt:make-dataset backend (category-matrix)
                               :label (category-labels)
                               (append (when dataset-parameters
                                         (list :parameters dataset-parameters))
                                       (when categorical-features
                                         (list :categorical-features categorical-features)))))
             (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 20
                                                           :parameters booster-parameters))
               (category-means (cl-gbdt:predict booster (category-matrix)))))))
    (format t "~A category means, :categorical-features '(0): ~S~%" name (run '(0)))
    (format t "~A category means, the same matrix read as quantities: ~S~%" name (run nil))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (demo "XGBoost " xgb nil
        '(:objective "binary:logistic" :max-depth 1 :verbosity 0 :min-child-weight 0
          :tree-method "hist"))
  (demo "LightGBM" lgbm '(:min-data-in-leaf 1 :min-data-in-bin 1 :min-data-per-group 1
                           :cat-smooth 0 :cat-l2 0 :verbose -1)
        '(:objective "binary" :max-depth 1 :verbose -1 :min-data-in-leaf 1 :min-data-per-group 1
          :cat-smooth 0 :cat-l2 0))

  ;; predict never takes :categorical-features -- every predict call above already omits it,
  ;; and still routes each row down the trained categorical splits correctly. XGBoost's own
  ;; saved model confirms it records nothing about which columns were categorical:
  (cl-gbdt:with-dataset
      (dataset (cl-gbdt:make-dataset xgb (category-matrix) :label (category-labels)
                                      :categorical-features '(0)))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                      :parameters '(:objective "binary:logistic" :max-depth 1
                                                    :verbosity 0 :min-child-weight 0
                                                    :tree-method "hist")))
      (let* ((json (cl-gbdt:model-to-string booster))
             (pos (search "\"feature_types\"" json)))
        (format t "XGBoost  saved model's own feature_types field: ~A~%"
                (subseq json pos (+ pos 25))))))

  ;; The same comparison over a csr-matrix, the other form make-dataset accepts -- every element
  ;; stored explicitly, so nothing here is about an absent CSR entry (see Sparse input above).
  (flet ((as-csr (matrix)
           (let* ((rows (array-dimension matrix 0)) (columns (array-dimension matrix 1))
                  (indptr (make-array (1+ rows))) (indices (make-array (* rows columns)))
                  (values (make-array (* rows columns))) (position 0))
             (dotimes (row rows)
               (setf (aref indptr row) position)
               (dotimes (column columns)
                 (setf (aref indices position) column)
                 (setf (aref values position) (aref matrix row column))
                 (incf position)))
             (setf (aref indptr rows) position)
             (cl-gbdt:make-csr-matrix :indptr indptr :indices indices :values values
                                      :num-columns columns))))
    (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb (as-csr (category-matrix))
                                                          :label (category-labels)
                                                          :categorical-features '(0)))
      (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 20
                                        :parameters '(:objective "binary:logistic" :max-depth 1
                                                      :verbosity 0 :min-child-weight 0
                                                      :tree-method "hist")))
        (format t "XGBoost  category means from a csr-matrix dataset, ~
                   :categorical-features '(0): ~S~%"
                (category-means (cl-gbdt:predict booster (category-matrix)))))))

  ;; A bad index signals unsupported-argument naming :categorical-features and the backend,
  ;; from make-dataset itself, before either backend's own creation call is reached.
  (dolist (indices (list '("0") '(-1) '(2) '(0 0)))
    (handler-case
        (cl-gbdt:free-dataset
         (cl-gbdt:make-dataset xgb (category-matrix) :label (category-labels)
                                :categorical-features indices))
      (error (c) (format t "XGBoost  :categorical-features ~S SIGNALED ~A: ~A~%"
                          indices (type-of c) c))))

  ;; A :valid-sets entry built WITHOUT :categorical-features, alongside a training set built
  ;; WITH it, provokes nothing: training succeeds and both entries evaluate the same way.
  (cl-gbdt:with-dataset
      (xgb-train (cl-gbdt:make-dataset xgb (category-matrix) :label (category-labels)
                                        :categorical-features '(0)))
    (cl-gbdt:with-dataset (xgb-valid (cl-gbdt:make-dataset xgb (category-matrix)
                                                            :label (category-labels)))
      (cl-gbdt:with-booster
          (booster (cl-gbdt:train xgb xgb-train :num-rounds 5 :valid-sets (list xgb-valid)
                                   :parameters '(:objective "binary:logistic" :max-depth 1
                                                 :verbosity 0 :min-child-weight 0
                                                 :tree-method "hist" :eval-metric "logloss")))
        (format t "XGBoost  evaluation, train :categorical-features '(0), valid without it: ~S~%"
                (cl-gbdt:evaluation booster)))))
  (cl-gbdt:with-dataset
      (lgbm-train (cl-gbdt:make-dataset lgbm (category-matrix) :label (category-labels)
                                         :categorical-features '(0)
                                         :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                       :min-data-per-group 1 :cat-smooth 0
                                                       :cat-l2 0 :verbose -1)))
    (cl-gbdt:with-dataset (lgbm-valid (cl-gbdt:make-dataset lgbm (category-matrix)
                                                             :label (category-labels)
                                                             :reference lgbm-train
                                                             :parameters '(:min-data-in-leaf 1
                                                                           :min-data-in-bin 1
                                                                           :min-data-per-group 1
                                                                           :cat-smooth 0 :cat-l2 0
                                                                           :verbose -1)))
      (cl-gbdt:with-booster
          (booster (cl-gbdt:train lgbm lgbm-train :num-rounds 5 :valid-sets (list lgbm-valid)
                                   :parameters '(:objective "binary" :max-depth 1 :verbose -1
                                                 :min-data-in-leaf 1 :min-data-per-group 1
                                                 :cat-smooth 0 :cat-l2 0
                                                 :metric "binary_logloss")))
        (format t "LightGBM evaluation, train :categorical-features '(0), valid without it: ~S~%"
                (cl-gbdt:evaluation booster)))))

  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
XGBoost  backend-supports-p :categorical-features => T
XGBoost  category means, :categorical-features '(0): (0.9739266037940979d0
                                                      0.026073377579450607d0
                                                      0.9739266037940979d0
                                                      0.026073377579450607d0
                                                      0.9739266037940979d0
                                                      0.026073377579450607d0)
XGBoost  category means, the same matrix read as quantities: (0.8159463405609131d0
                                                              0.40605124831199646d0
                                                              0.5471441149711609d0
                                                              0.46124157309532166d0
                                                              0.5562390089035034d0
                                                              0.19264821708202362d0)
LightGBM backend-supports-p :categorical-features => T
LightGBM category means, :categorical-features '(0): (0.9344864001786668d0
                                                      0.0655135998213332d0
                                                      0.9344864001786668d0
                                                      0.0655135998213332d0
                                                      0.9344864001786668d0
                                                      0.0655135998213332d0)
LightGBM category means, the same matrix read as quantities: (0.8154497898954673d0
                                                              0.4733477173729823d0
                                                              0.5002265133575413d0
                                                              0.5002265133575413d0
                                                              0.5265399565045455d0
                                                              0.184374537772783d0)
XGBoost  saved model's own feature_types field: "feature_types":[],"gradi
XGBoost  category means from a csr-matrix dataset, :categorical-features '(0): (0.9739266037940979d0
                                                                                0.026073377579450607d0
                                                                                0.9739266037940979d0
                                                                                0.026073377579450607d0
                                                                                0.9739266037940979d0
                                                                                0.026073377579450607d0)
XGBoost  :categorical-features ("0") SIGNALED UNSUPPORTED-ARGUMENT: :categorical-features is not supported by XGBOOST: each categorical column must be a non-negative integer index.
XGBoost  :categorical-features (-1) SIGNALED UNSUPPORTED-ARGUMENT: :categorical-features is not supported by XGBOOST: each categorical column must be a non-negative integer index.
XGBoost  :categorical-features (2) SIGNALED UNSUPPORTED-ARGUMENT: :categorical-features is not supported by XGBOOST: categorical column index 2 is beyond the matrix's 2 columns.
XGBoost  :categorical-features (0 0) SIGNALED UNSUPPORTED-ARGUMENT: :categorical-features is not supported by XGBOOST: the same categorical column was named more than once.
XGBoost  evaluation, train :categorical-features '(0), valid without it: ((0
                                                                           "logloss"
                                                                           0.17660880088806152d0)
                                                                          (1
                                                                           "logloss"
                                                                           0.17660880088806152d0))
LightGBM evaluation, train :categorical-features '(0), valid without it: ((0
                                                                           "binary_logloss"
                                                                           0.35374722486733495d0)
                                                                          (1
                                                                           "binary_logloss"
                                                                           0.353747224867335d0))
```

Six categories in one column, alternating class by category, is a fixture where no
threshold on the ordinal separates the two classes but a categorical split choosing the subset
`{0, 2, 4}` does -- taken directly from `tests/functional/categorical-features.lisp`, whose own
comments measure why the small-fixture parameters above are needed, one setting at a time, on
rows this few: `max_depth` `1` on both backends, or spare tree capacity rebuilds the
alternating pattern from the plain ordinal and hides what this fixture measures; XGBoost's
`min_child_weight` `0`, since its default's per-leaf hessian check over four rows rejects the
split; and LightGBM's `min_data_in_leaf`, `min_data_in_bin`, `min_data_per_group`, `cat_smooth`
and `cat_l2`, each of which blocks or weakens a categorical split at its own default on a
fixture this small. On both backends the categorical arm answers one score per class -- `0.974`
positive / `0.026` negative on XGBoost, `0.934`/`0.066` on LightGBM -- while the plain arm, the
identical matrix read as quantities, cannot express that split: six different scores that
barely separate the classes on XGBoost, and on LightGBM two categories (2 and 3) that tie
exactly at `0.500`.

`predict` above is called with no argument naming the categorical column at all -- there is no
such argument to give it -- and every prediction still comes out right: on the dense matrix,
and identically on a `csr-matrix` built from the same rows (`0.9739266037940979d0` and its five
siblings again, digit for digit). XGBoost's own saved model confirms the mechanism: an empty
`"feature_types":[]`, the same field a model trained with no categorical column at all would
save -- the category sets a split needs live in the trees themselves, not on the booster.

The four `unsupported-argument` signals above are the renderer's own rejections
(`cl-gbdt/src/config/categorical-features`), shared by both backends, so the identical four
checks refuse a bad index on LightGBM as well -- naming `LIGHTGBM` in place of `XGBOOST` and
nothing else.

The last two lines are the answer to a question the [missing values](#missing-values) section
above invites: a `:valid-sets` entry built *without* `:categorical-features`, alongside a
training set built *with* it, provokes nothing at all. Training succeeds, and the two entries
evaluate the same way on both backends -- exactly, on XGBoost, and on LightGBM to within
floating-point noise on the order of `1d-17`. That noise is not from this feature: repeating an
otherwise identical comparison with no categorical column anywhere shows the same run-to-run
jitter in LightGBM's `evaluation`, from one run of this section to the next.

#### XGBoost: `tree_method` must be `hist` or `approx`

Measured directly: a dataset built with `:categorical-features` trains successfully with
`tree_method` `hist` and with `approx`, but `exact` refuses it -- and only once `train` reaches
`XGBoosterUpdateOneIter`, not at `make-dataset`. The wrapper does not pre-validate
`tree_method`; it is ordinary `:parameters` business, set on the booster rather than the
dataset, and nothing here stops a caller from setting it wrong:

```lisp
(let ((xgb (cl-gbdt:open-backend :xgboost)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (dataset (cl-gbdt:make-dataset xgb (category-matrix) :label (category-labels)
                                          :categorical-features '(0)))
        (format t "make-dataset with :categorical-features '(0) succeeded: rows=~D~%"
                (cl-gbdt:dataset-num-rows dataset))
        (handler-case
            (cl-gbdt:with-booster
                (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                        :parameters '(:objective "binary:logistic" :max-depth 1
                                                      :verbosity 0 :min-child-weight 0
                                                      :tree-method "exact")))
              (declare (ignore booster))
              (format t "train with tree_method exact succeeded (unexpected)~%"))
          (error (c)
            (let ((text (princ-to-string c)))
              (format t "train with tree_method exact SIGNALED ~A:~%  ~A~%" (type-of c)
                      (subseq text 0 (position #\Newline text))))))
        (cl-gbdt:with-booster
            (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                    :parameters '(:objective "binary:logistic" :max-depth 1
                                                  :verbosity 0 :min-child-weight 0
                                                  :tree-method "hist")))
          (declare (ignore booster))
          (format t "train with tree_method hist succeeded~%"))
        (cl-gbdt:with-booster
            (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                    :parameters '(:objective "binary:logistic" :max-depth 1
                                                  :verbosity 0 :min-child-weight 0
                                                  :tree-method "approx")))
          (declare (ignore booster))
          (format t "train with tree_method approx succeeded~%")))
    (cl-gbdt:close-backend xgb)))
```

Output:

```
make-dataset with :categorical-features '(0) succeeded: rows=24
train with tree_method exact SIGNALED FOREIGN-CALL-ERROR:
  XGBoosterUpdateOneIter returned -1: [13:50:17] /__w/xgboost/xgboost/src/tree/updater_colmaker.cc:107: Updater `grow_colmaker` or `exact` tree method doesn't support categorical features.
train with tree_method hist succeeded
train with tree_method approx succeeded
```

The bracketed time in XGBoost's message is its own wall-clock stamp, the only part of this
output that changes between runs -- the same caveat [Sparse input](#sparse-input-csr-matrices)
makes about a different message above. The dataset is built and the feature types attached
without complaint whatever `tree_method` will later be; it is `train`, not `make-dataset`, that
finds out `exact` cannot use them.

#### LightGBM: `categorical_feature` and its four aliases

LightGBM's own name for the categorical column list is a parameter-string key,
`categorical_feature`, and the library also honours four synonyms for it -- measured against
the vendored 4.7.0: `cat_feature`, `categorical_column`, `cat_column` and `categorical_features`.
Supplying `:categorical-features` and any of those five spellings in `:parameters` together
signals `unsupported-argument` naming `make-dataset`'s own `:parameters` argument -- not
because the wrapper owns the key outright, but because LightGBM keeps the *first* occurrence of
a duplicated key while `make-dataset` appends its own entry *last*, so the argument the caller
explicitly named would be the one silently discarded. `:parameters` **on its own is
unaffected**: a caller who names no categorical column and writes `categorical_feature` there
by hand -- policy section 6's escape hatch for a backend's own vocabulary -- keeps working
exactly as it did before this argument existed. A near-miss such as `cat_features`, the plural
of the honoured `cat_feature`, is not itself an alias -- LightGBM does not honour it -- and is
never refused, alongside `:categorical-features` or without it: two ways of saying the same
thing remain reachable on this backend, by design.

```lisp
(defparameter *dataset-parameters*
  '(:min-data-in-leaf 1 :min-data-in-bin 1 :min-data-per-group 1 :cat-smooth 0 :cat-l2 0
    :verbose -1))

(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (progn
        ;; Both :categorical-features and one of the five spellings LightGBM honours for the
        ;; same key in :parameters: refused, naming make-dataset's own :parameters argument.
        (dolist (key '(:categorical-feature :cat-feature :categorical-column :cat-column
                       :categorical-features))
          (handler-case
              (cl-gbdt:free-dataset
               (cl-gbdt:make-dataset lgbm (category-matrix) :label (category-labels)
                                      :categorical-features '(0)
                                      :parameters (append (list key "0") *dataset-parameters*)))
            (error (c) (format t ":categorical-features '(0) alongside :parameters ~S SIGNALED ~A~%"
                                key (type-of c)))))
        ;; :parameters alone, naming no :categorical-features, is untouched -- the escape hatch.
        (cl-gbdt:with-dataset
            (dataset (cl-gbdt:make-dataset lgbm (category-matrix) :label (category-labels)
                                            :parameters (append '(:categorical-feature "0")
                                                                *dataset-parameters*)))
          (format t ":parameters :categorical-feature alone (no :categorical-features) built ~
                     a dataset: rows=~D~%"
                  (cl-gbdt:dataset-num-rows dataset)))
        ;; cat_features, the plural of the honoured cat_feature, is not itself an alias and is
        ;; never refused -- it reaches the library untouched even alongside :categorical-features.
        (cl-gbdt:with-dataset
            (dataset (cl-gbdt:make-dataset lgbm (category-matrix) :label (category-labels)
                                            :categorical-features '(0)
                                            :parameters (append '(:cat-features "0")
                                                                *dataset-parameters*)))
          (format t ":categorical-features '(0) alongside :parameters :cat-features (not an ~
                     alias) built a dataset: rows=~D~%"
                  (cl-gbdt:dataset-num-rows dataset))))
    (cl-gbdt:close-backend lgbm)))
```

Output:

```
:categorical-features '(0) alongside :parameters :CATEGORICAL-FEATURE SIGNALED UNSUPPORTED-ARGUMENT
:categorical-features '(0) alongside :parameters :CAT-FEATURE SIGNALED UNSUPPORTED-ARGUMENT
:categorical-features '(0) alongside :parameters :CATEGORICAL-COLUMN SIGNALED UNSUPPORTED-ARGUMENT
:categorical-features '(0) alongside :parameters :CAT-COLUMN SIGNALED UNSUPPORTED-ARGUMENT
:categorical-features '(0) alongside :parameters :CATEGORICAL-FEATURES SIGNALED UNSUPPORTED-ARGUMENT
:parameters :categorical-feature alone (no :categorical-features) built a dataset: rows=24
:categorical-features '(0) alongside :parameters :cat-features (not an alias) built a dataset: rows=24
```

#### The values inside a categorical column are passed through unvalidated

`:categorical-features` validates which *columns* are categorical -- the four checks above --
never what *values* sit inside them. Measured on LightGBM: a fractional category id truncates
to an integer one silently, and a negative one is converted to missing, with a warning on the
library's own stderr and nothing that reaches Lisp as a condition either way.

```lisp
(defun quiet-nan ()
  "A quiet double-float NaN, built from its bits so no arithmetic can trap."
  (sb-kernel:make-double-float -524288 0))

(defun category-matrix-with-cell (row value)
  "CATEGORY-MATRIX, with ROW's column 0 replaced by VALUE."
  (let ((matrix (category-matrix)))
    (setf (aref matrix row 0) value)
    matrix))

(defun train-predict (backend matrix)
  (cl-gbdt:with-dataset
      (dataset (cl-gbdt:make-dataset backend matrix :label (category-labels)
                                      :categorical-features '(0)
                                      :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                    :min-data-per-group 1 :cat-smooth 0
                                                    :cat-l2 0 :verbose 1)))
    (cl-gbdt:with-booster
        (booster (cl-gbdt:train backend dataset :num-rounds 20
                                :parameters '(:objective "binary" :max-depth 1 :verbose -1
                                              :min-data-in-leaf 1 :min-data-per-group 1
                                              :cat-smooth 0 :cat-l2 0)))
      (category-means (cl-gbdt:predict booster (category-matrix))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (progn
        ;; A fractional category id truncates to an integer one: every row's column 0 gets
        ;; +0.7, and the categories it truncates to (0..5) are exactly the same as before.
        (format t "integer category ids:                ~S~%"
                (train-predict lgbm (category-matrix)))
        (let ((offset (category-matrix)))
          (dotimes (row (array-dimension offset 0)) (incf (aref offset row 0) 0.7d0))
          (format t "the same ids, each +0.7 (fractional): ~S~%" (train-predict lgbm offset)))

        ;; A negative category value prints a warning on LightGBM's own stderr and is converted
        ;; to missing -- verified against an explicit NaN in the identical cell, which reaches
        ;; the same prediction and prints no warning at all: the two are the same event.
        (format t "row 1 (category 0) set to -1.0d0 (negative):~%  ~S~%"
                (train-predict lgbm (category-matrix-with-cell 1 -1.0d0)))
        (format t "row 1 (category 0) set to an explicit NaN instead:~%  ~S~%"
                (train-predict lgbm (category-matrix-with-cell 1 (quiet-nan)))))
    (cl-gbdt:close-backend lgbm)))
```

Output:

```
integer category ids:                (0.9344864001786668d0 0.0655135998213332d0
                                      0.9344864001786668d0 0.0655135998213332d0
                                      0.9344864001786668d0 0.0655135998213332d0)
the same ids, each +0.7 (fractional): (0.9344864001786668d0
                                       0.0655135998213332d0
                                       0.9344864001786668d0
                                       0.0655135998213332d0
                                       0.9344864001786668d0
                                       0.0655135998213332d0)
[LightGBM] [Warning] Met negative value in categorical features, will convert it to NaN
row 1 (category 0) set to -1.0d0 (negative):
  (0.9344864001786668d0 0.06551359982133317d0 0.9344864001786668d0
   0.06551359982133317d0 0.9344864001786668d0 0.06551359982133317d0)
row 1 (category 0) set to an explicit NaN instead:
  (0.9344864001786668d0 0.06551359982133317d0 0.9344864001786668d0
   0.06551359982133317d0 0.9344864001786668d0 0.06551359982133317d0)
```

Every fractional id in the second run truncates to the same integer category the first run
used, so the two predictions match digit for digit. The negative id in the third run is the
only line that prints a warning -- and its own predictions match the fourth run's, where the
same cell holds an explicit NaN instead of `-1.0d0`, digit for digit as well: LightGBM's
"convert it to NaN" is not a figure of speech, and the fourth run reaches the identical code
path silently, an actual NaN never having been negative to begin with. Both of those runs
differ from the first two only in the three negative categories' shared score
(`0.0655135998213332d0` becomes `0.06551359982133317d0`; category 0's own score, positive and
untouched by the corrupted row, stays bit-identical) -- a real difference, if a small one on
this fixture, from the model having one fewer valid example of column 0 to learn category 0
from. The wrapper validates the *indices* `:categorical-features` names; it never validates
the *values* sitting in the columns those indices point at.

### Prediction shape

`predict` returns two values. The FIRST is exactly what it has always been -- the same
`(simple-array double-float (* *))`, same dimensions, same elements, for every `KIND`, dense
or sparse -- untouched by anything below. The SECOND is new: the SHAPE the backend states for
the result it just wrote, as a list of integers in `array-dimensions` order, or `NIL` where
the backend states none. A caller who ignores it sees behaviour identical to before this
feature existed.

`:prediction-shape` is the capability, answerable through `backend-supports-p` and true on
both vendored backends -- but **no operation refuses on it**. There is no argument asking for
a shape, so a false answer would mean only that the second value is always `NIL`; `predict`
would keep predicting exactly as it does today, on every `KIND`, dense or sparse. That is not
the general rule for a capability in this API -- six of the ten registered capabilities
are re-checked by the operation they gate and signal `capability-unavailable` when they read
false: `:sparse-input`, `:missing-value`, `:categorical-features`, `:custom-objective` and
`:custom-evaluation` (each documented above, on the operation that checks it -- the last two
at [Custom objective](#custom-objective) and [Custom evaluation](#custom-evaluation)) and
`:model-slicing` (see [Asking a backend what it
can do](#asking-a-backend-what-it-can-do)). `:prediction-shape` is simply not one of those
six, because it gates nothing a caller asks for.

The two backends fill the second value from opposite directions. XGBoost's prediction entry
points write an `out_shape`/`out_dim` pair, and `predict` reads that pair straight back and
states exactly what the library said. LightGBM's do not -- `LGBM_BoosterCalcNumPredict`
returns an element count and nothing else -- so LightGBM's second value is DERIVED, and only
as far as the derivation can go: `:normal` and `:raw` state the result array's own
`array-dimensions` (there is nothing to add to what the array already says), `:contrib` is
derived from that element count, the row count, and a further library call this derivation
makes, `LGBM_BoosterGetNumFeature`, and `:leaf-index` states `NIL` -- see below for why.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

;; Eighteen rows, four columns, three classes, six rows per class -- large enough that
;; :leaf-index and :contrib's extra axes (rounds, output groups, features+1) are all
;; different numbers, so a reader cannot mistake one axis for another by coincidence.
(defparameter *shape-matrix*
  (let* ((rows-per-class 6) (num-classes 3) (cols 4) (rows (* rows-per-class num-classes))
         (matrix (make-array (list rows cols) :element-type 'double-float)))
    (dotimes (row rows)
      (let ((class (floor row rows-per-class)) (offset (mod row rows-per-class)))
        (dotimes (col cols)
          (setf (aref matrix row col) (coerce (+ (* class 10) offset col) 'double-float)))))
    matrix))
(defparameter *shape-label*
  (let* ((rows-per-class 6) (rows (* rows-per-class 3))
         (label (make-array rows :element-type 'single-float)))
    (dotimes (row rows) (setf (aref label row) (coerce (floor row rows-per-class) 'single-float)))
    label))

(defun show-shapes (name backend dataset-parameters booster-parameters)
  (format t "~A backend-supports-p :prediction-shape => ~S~%"
          name (cl-gbdt:backend-supports-p backend :prediction-shape))
  (cl-gbdt:with-dataset (dataset (apply #'cl-gbdt:make-dataset backend *shape-matrix*
                                        :label *shape-label* dataset-parameters))
    (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 4
                                                   :parameters booster-parameters))
      (dolist (kind '(:normal :raw :leaf-index :contrib))
        (multiple-value-bind (result shape) (cl-gbdt:predict booster *shape-matrix* :kind kind)
          (format t "~A ~S: array-dimensions ~S, shape ~S~%"
                  name kind (array-dimensions result) shape))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (show-shapes "LightGBM" lgbm
               '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
               '(:objective "multiclass" :num-class 3 :num-leaves 2 :min-data-in-leaf 1
                 :min-data-in-bin 1 :verbose -1))
  (show-shapes "XGBoost " xgb '()
               '(:objective "multi:softprob" :num-class 3 :max-depth 3 :eta 0.5 :verbosity 0))
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM backend-supports-p :prediction-shape => T
LightGBM :NORMAL: array-dimensions (18 3), shape (18 3)
LightGBM :RAW: array-dimensions (18 3), shape (18 3)
LightGBM :LEAF-INDEX: array-dimensions (18 12), shape NIL
LightGBM :CONTRIB: array-dimensions (18 15), shape (18 3 5)
XGBoost  backend-supports-p :prediction-shape => T
XGBoost  :NORMAL: array-dimensions (18 3), shape (18 3)
XGBoost  :RAW: array-dimensions (18 3), shape (18 3)
XGBoost  :LEAF-INDEX: array-dimensions (18 12), shape (18 4 3 1)
XGBoost  :CONTRIB: array-dimensions (18 15), shape (18 3 5)
```

`:normal` and `:raw` state `(18 3)` on both backends -- the array's own `array-dimensions`,
so there is nothing here beyond what the first value already said. `:leaf-index` and
`:contrib` are where the two backends diverge. XGBoost states four and three axes
respectively, RICHER than the `18x12` and `18x15` arrays `predict`'s first value returns for
them: before this branch, `predict` folded those same axes into the array's own two,
discarding the structure the library had already reported. LightGBM's `:contrib` derives the
identical three axes arithmetically from a count and two further numbers; its `:leaf-index`
states `NIL` -- LightGBM's `predict` still returns the `18x12` array for it, exactly as
before, since no operation refuses on this capability and a `NIL` second value changes
nothing about the first.

#### Binary models are multidimensional too

The case a reader guesses wrong: `:leaf-index` and `:contrib`'s extra axes look like a
multiclass artifact in the block above, where every shape happens to mention 3. They are not.

```lisp
;; A trivially separable eight-row three-column fixture, in the same spirit as
;; tests/functional/support.lisp's make-separable-dataset -- one output group,
;; unlike *SHAPE-MATRIX*'s three.
(defparameter *shape-bin-matrix*
  (let ((rows 8) (cols 3))
    (let ((m (make-array (list rows cols) :element-type 'double-float)))
      (dotimes (i rows)
        (dotimes (j cols) (setf (aref m i j) (coerce (/ (+ i j) 10) 'double-float))))
      m)))
(defparameter *shape-bin-label*
  (let ((rows 8))
    (let ((l (make-array rows :element-type 'single-float)))
      (dotimes (i rows)
        (setf (aref l i) (if (> (aref *shape-bin-matrix* i 0) 0.35d0) 1.0 0.0)))
      l)))

(let ((xgb (cl-gbdt:open-backend :xgboost)))
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb *shape-bin-matrix*
                                                        :label *shape-bin-label*))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 4
                                     :parameters '(:objective "binary:logistic" :max-depth 2
                                                   :eta 0.5 :verbosity 0)))
      (dolist (kind '(:leaf-index :contrib))
        (multiple-value-bind (result shape)
            (cl-gbdt:predict booster *shape-bin-matrix* :kind kind)
          (format t "XGBoost binary model ~S: array-dimensions ~S, shape ~S~%"
                  kind (array-dimensions result) shape)))))
  (cl-gbdt:close-backend xgb))
```

Output:

```
XGBoost binary model :LEAF-INDEX: array-dimensions (8 4), shape (8 4 1 1)
XGBoost binary model :CONTRIB: array-dimensions (8 4), shape (8 1 4)
```

One output group, and both shapes are still multidimensional -- `:leaf-index` four axes,
`:contrib` three. This fixture is also where the first value alone stops being enough: at
four rounds over three columns, `:leaf-index`'s folded width (4 rounds x 1 class) and
`:contrib`'s (1 class x (3 features + 1)) are both 4, so the array `predict` returns is
`8x4` for either `KIND` -- the same shape, from two calls that mean completely different
things. The second value is what tells them apart: `(8 4 1 1)` against `(8 1 4)`, not the same
list even though both multiply out to 4 x 8 elements.

#### What backs LightGBM's derived ordering, and the view built from it

`:contrib`'s three axes are `(rows classes features+1)`, CLASS-MAJOR: every output group's
own `features+1` contributions sit together, one group after another. The arithmetic in
`contrib-shape` divides the element count into three numbers exactly as well with the last
two axes swapped -- `(rows features+1 classes)`, FEATURE-MAJOR -- so the ordering is a claim
the division alone cannot support. What supports it is a property of what SHAP contributions
mean: the contributions for one output group sum to that group's own `:raw` score. Grouped
the way the shape claims, they should; grouped the other way, they should not.

```lisp
;; *SHAPE-MATRIX* and *SHAPE-LABEL* as defined above.
(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset lgbm *shape-matrix* :label *shape-label*
                                   :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                 :verbose -1)))
    (cl-gbdt:with-booster (booster (cl-gbdt:train lgbm dataset :num-rounds 4
                                     :parameters '(:objective "multiclass" :num-class 3
                                                   :num-leaves 2 :min-data-in-leaf 1
                                                   :min-data-in-bin 1 :verbose -1)))
      (let ((raw (cl-gbdt:predict booster *shape-matrix* :kind :raw)))
        (multiple-value-bind (contrib shape) (cl-gbdt:predict booster *shape-matrix* :kind :contrib)
          (format t "contrib array-dimensions ~S, derived shape ~S~%"
                  (array-dimensions contrib) shape)
          ;; The one-form N-dimensional view: SHAPE describes CONTRIB's own buffer, so the
          ;; displaced array below reads it three-dimensionally without copying anything.
          (let ((view (make-array shape :element-type 'double-float :displaced-to contrib))
                (worst 0.0d0))
            (destructuring-bind (rows classes width) shape
              (dotimes (row rows)
                (dotimes (class classes)
                  (let ((summed (loop :for feature :below width
                                       :sum (aref view row class feature))))
                    (setf worst (max worst (abs (- summed (aref raw row class)))))))))
            (format t "class-major sums vs :raw, worst absolute difference: ~,3E~%" worst))
          ;; The control: the same buffer, read with the last two axes swapped -- what
          ;; :contrib's shape would be if the derivation had guessed the wrong order.
          (let* ((bad-shape (list (first shape) (third shape) (second shape)))
                 (view (make-array bad-shape :element-type 'double-float :displaced-to contrib))
                 (worst 0.0d0))
            (destructuring-bind (rows width classes) bad-shape
              (dotimes (row rows)
                (dotimes (class classes)
                  (let ((summed (loop :for feature :below width
                                       :sum (aref view row feature class))))
                    (setf worst (max worst (abs (- summed (aref raw row class)))))))))
            (format t "feature-major control vs :raw, worst absolute difference: ~,3E~%" worst))))))
  (cl-gbdt:close-backend lgbm))
```

Output:

```
contrib array-dimensions (18 15), derived shape (18 3 5)
class-major sums vs :raw, worst absolute difference: 2.220d-16
feature-major control vs :raw, worst absolute difference: 7.783d-1
```

`(make-array shape :displaced-to contrib)` is the one-form N-dimensional view: `SHAPE`
already describes `CONTRIB`'s own storage, so the displaced array reads that same buffer
three-dimensionally, `(aref view row class feature)` in place of hand-rolled row/column
arithmetic on the flat array -- the point of returning a shape at all, not a footnote to it.
Read that way, class-major reproduces the raw scores to `2.220d-16`, floating-point roundoff
and nothing more; read the other way, the feature-major control misses every one of the 54
`(row, class)` sums by up to `7.783d-1`, some fifteen orders of magnitude larger. That gap is
what turns the ordering from an assumption into a measurement, held by
`lightgbm-s-derived-contrib-shape-is-the-one-the-numbers-support` in
`tests/functional/prediction-shape.lisp`.

`:leaf-index` gets no such derivation. Its element count divides by iterations and output
groups exactly as `:contrib`'s divides by output groups and width, but a leaf index is an
opaque identifier -- it sums to nothing and agrees with nothing, so there is no SHAP-sum-style
property here to check a guessed ordering against. Asserting an ordering with nothing to check
it against would be exactly the mistake the measurement above exists to avoid: a shape stated
on arithmetic alone, with no SHAP-sum-style test and no feature-major control to catch it if
the axes were transposed. `NIL` is what `predict`'s second value means everywhere a backend
states none, and LightGBM's `:leaf-index` result -- the `18x12` array -- is entirely
unaffected by stating it.

#### On a `csr-matrix`, XGBoost states a shape for two kinds out of four

[Sparse input](#sparse-input-csr-matrices) above measures that XGBoost's sparse entry point,
`XGBoosterPredictFromCSR`, serves only `:normal` and `:raw`, refusing `:contrib` and
`:leaf-index` with `foreign-call-error` before either produces a result. A shape is read only
from a call that returned one, so the same split holds here: a `csr-matrix` reaches a shape
for the two `KIND`s that succeed, and never reaches one for the two that do not.

```lisp
;; *SHAPE-MATRIX* and *SHAPE-LABEL* as defined above.
(defun dense-to-csr (matrix)
  "MATRIX as a `csr-matrix' with every element stored explicitly -- see 'An absent entry
is not a zero' above for why dropping the zeros would describe a different matrix to XGBoost."
  (let* ((rows (array-dimension matrix 0)) (cols (array-dimension matrix 1))
         (indptr (make-array (1+ rows))) (indices (make-array (* rows cols)))
         (values (make-array (* rows cols))) (pos 0))
    (dotimes (r rows)
      (setf (aref indptr r) pos)
      (dotimes (c cols)
        (setf (aref indices pos) c) (setf (aref values pos) (aref matrix r c)) (incf pos)))
    (setf (aref indptr rows) pos)
    (cl-gbdt:make-csr-matrix :indptr indptr :indices indices :values values :num-columns cols)))

(let ((xgb (cl-gbdt:open-backend :xgboost))
      (csr (dense-to-csr *shape-matrix*)))
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb *shape-matrix* :label *shape-label*))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 4
                                     :parameters '(:objective "multi:softprob" :num-class 3
                                                   :max-depth 3 :eta 0.5 :verbosity 0)))
      (dolist (kind '(:normal :raw :leaf-index :contrib))
        (handler-case
            (multiple-value-bind (result shape) (cl-gbdt:predict booster csr :kind kind)
              (format t "XGBoost csr-matrix ~S: array-dimensions ~S, shape ~S~%"
                      kind (array-dimensions result) shape))
          ;; XGBoost's message carries a multi-line stack trace; line 1 is the refusal.
          (error (c) (let ((text (princ-to-string c)))
                       (format t "XGBoost csr-matrix ~S: SIGNALED ~A~%  ~A~%" kind (type-of c)
                               (subseq text 0 (position #\Newline text)))))))))
  (cl-gbdt:close-backend xgb))
```

Output:

```
XGBoost csr-matrix :NORMAL: array-dimensions (18 3), shape (18 3)
XGBoost csr-matrix :RAW: array-dimensions (18 3), shape (18 3)
XGBoost csr-matrix :LEAF-INDEX: SIGNALED FOREIGN-CALL-ERROR
  XGBoosterPredictFromCSR returned -1: [08:23:10] /__w/xgboost/xgboost/src/learner.cc:1264: Unsupported prediction type:6
XGBoost csr-matrix :CONTRIB: SIGNALED FOREIGN-CALL-ERROR
  XGBoosterPredictFromCSR returned -1: [08:23:10] /__w/xgboost/xgboost/src/learner.cc:1264: Unsupported prediction type:2
```

As in Sparse input above, the bracketed time in XGBoost's two messages is XGBoost's own
wall-clock stamp, the only part of this output that differs run to run. `:normal` and `:raw`
state `(18 3)`, identical to the dense call earlier in this section; `:leaf-index` and
`:contrib` never reach a shape, or a result, at all. LightGBM has no such split -- its CSR
entry point serves all four `KIND`s, and states, or declines to state, a shape for each of
them exactly as it does on a dense matrix.

### Custom objective

`train` also takes `:objective`, a function that turns the current raw scores into a gradient
and a Hessian, so a run boosts against the caller's own loss instead of one built into the
library. It needs the `:custom-objective` capability, answerable through `backend-supports-p`
and true on both vendored backends; `train` re-checks it itself and signals
`capability-unavailable` for a non-`NIL` `:objective` when it reads false, before any foreign
call.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(defparameter *co-matrix*
  (make-array '(8 1) :element-type 'double-float
              :initial-contents '((0.0d0) (1.0d0) (2.0d0) (3.0d0)
                                   (4.0d0) (5.0d0) (6.0d0) (7.0d0))))
(defparameter *co-label*
  (make-array 8 :element-type 'double-float
              :initial-contents '(0.0d0 1.0d0 4.0d0 9.0d0 16.0d0 25.0d0 36.0d0 49.0d0)))

(defun squared-error (scores)
  "GRAD = prediction - label, HESS = 1 -- squared error's own derivatives."
  (let* ((rows (array-dimension scores 0))
         (grad (make-array (list rows 1) :element-type 'double-float))
         (hess (make-array (list rows 1) :element-type 'double-float :initial-element 1.0d0)))
    (dotimes (row rows (values grad hess))
      (setf (aref grad row 0) (- (aref scores row 0) (aref *co-label* row))))))

(let ((backend (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (dataset (cl-gbdt:make-dataset backend *co-matrix* :label *co-label*
                                          :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                        :verbose -1)))
        (cl-gbdt:with-booster
            (built-in (cl-gbdt:train backend dataset :num-rounds 5
                                      :parameters '(:objective "regression" :num-leaves 4
                                                    :min-data-in-leaf 1 :min-data-in-bin 1
                                                    :verbose -1 :boost-from-average nil)))
          (cl-gbdt:with-booster
              (custom (cl-gbdt:train backend dataset :num-rounds 5
                                      :parameters '(:num-leaves 4 :min-data-in-leaf 1
                                                    :min-data-in-bin 1 :verbose -1
                                                    :boost-from-average nil)
                                      :objective #'squared-error))
            (format t "built-in :raw:~%~S~%" (cl-gbdt:predict built-in *co-matrix* :kind :raw))
            (format t "custom  :raw:~%~S~%" (cl-gbdt:predict custom *co-matrix* :kind :raw)))))
    (cl-gbdt:close-backend backend)))
```

Output:

```
built-in :raw:
#2A((0.9006424501538277d0)
    (0.9006424501538277d0)
    (0.9006424501538277d0)
    (3.7816945374011977d0)
    (5.989342808723447d0)
    (11.236049175262446d0)
    (13.940538883209221d0)
    (19.681847476959206d0))
custom  :raw:
#2A((0.9006424501538277d0)
    (0.9006424501538277d0)
    (0.9006424501538277d0)
    (3.7816945374011977d0)
    (5.989342808723447d0)
    (11.236049175262446d0)
    (13.940538883209221d0)
    (19.681847476959206d0))
```

`SQUARED-ERROR` above is squared error's own gradient and Hessian, `grad = prediction - label`
and `hess = 1` -- the same derivatives LightGBM's built-in `"regression"` objective uses -- and
the two runs land on the identical model, digit for digit: a custom objective is not an
approximation of the library's own, it drives the same trees when it computes the same thing.

`:objective` is called once per iteration, before that iteration's update, with **one
argument**: the booster's current raw scores for its training set, as a `(ROWS GROUPS)`
`double-float` array -- the margin, before any sigmoid or softmax transform, and the same shape
and element type `predict` returns. `GROUPS` is 1 for regression and binary classification and
`num_class` for multiclass. It must return **two values**, the gradient and the Hessian, each a
`(ROWS GROUPS)` array. The **shape** is what is checked -- the wrong rank, the wrong
dimensions, or one value instead of two signals `dimension-mismatch` before any foreign call,
so a wrongly shaped array is never read as though it had the right shape:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(defparameter *co-matrix*
  (make-array '(8 1) :element-type 'double-float
              :initial-contents '((0.0d0) (1.0d0) (2.0d0) (3.0d0)
                                   (4.0d0) (5.0d0) (6.0d0) (7.0d0))))
(defparameter *co-label*
  (make-array 8 :element-type 'double-float
              :initial-contents '(0.0d0 1.0d0 4.0d0 9.0d0 16.0d0 25.0d0 36.0d0 49.0d0)))

(let ((backend (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (dataset (cl-gbdt:make-dataset backend *co-matrix* :label *co-label*
                                          :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                        :verbose -1)))
        ;; A flat (ROWS) vector instead of the required (ROWS GROUPS) array -- the shape a
        ;; caller who thinks in one dimension returns.
        (handler-case
            (cl-gbdt:train backend dataset :num-rounds 1
                            :objective (lambda (scores)
                                         (declare (ignore scores))
                                         (values (make-array 8 :element-type 'double-float
                                                                :initial-element 0.0d0)
                                                 (make-array '(8 1) :element-type 'double-float
                                                                    :initial-element 1.0d0))))
          (error (c) (format t "SIGNALED ~A~%  ~A~%" (type-of c) c))))
    (cl-gbdt:close-backend backend)))
```

Output:

```
SIGNALED DIMENSION-MISMATCH
  Dimension mismatch. Expected: (8 1), got: (GRADIENT (8) HESSIAN (8 1))
```

The **element type is not** part of that check, and deliberately. `double-float`,
`single-float` and a general array whose elements are reals -- what `(make-array (list rows 1))`
with no `:element-type` gives, the most natural thing to write -- are all accepted, and all
three train the identical model on both backends, because each element is coerced to the
`single-float` the C signature's `const float*` takes as the buffer is written. An element that
is *not* a real -- a string, `NIL`, a complex -- signals `unsupported-element-type` naming that
element's own type, at the write and before the library has been called: the same condition,
with the same value in `unsupported-element-type-given`, that a `csr-matrix` holding a non-real
value already signals from `make-csr-matrix`. Nothing scans either array a second time to say
so; the check rides along with the coercion that was happening anyway.

The two libraries want that `(ROWS GROUPS)` array flattened into their C buffers in opposite
orders -- LightGBM **group-major** (row I of group K at `(+ (* K ROWS) I)`), XGBoost
**row-major** (row I of group K at `(+ (* I GROUPS) K)`, what an `__array_interface__` of shape
`[ROWS, GROUPS]` means) -- and each backend's own code absorbs that difference. **The
flattening is the wrapper's job, not the caller's**: `:objective` is handed, and returns, one
`(ROWS GROUPS)` array on both backends, whichever order the library underneath actually wants
it in. Both orderings are measured rather than assumed -- a gradient confined to one output
group moves only that group's raw score under the correct layout and smears across every group
under the other -- held by
`a-gradient-in-one-output-group-moves-only-that-group` in
`tests/functional/custom-objective.lisp`, which runs the same fixture on both backends and
would fail if either flattening were transposed.

`:objective` is the only place inside `train`'s loop where code cl-gbdt did not write runs, and
that code can reach the handles the loop is holding: `free-dataset` on the training set, on a
`:valid-sets` entry, or `close-backend` on the backend itself. All three are **caught**, not
crashed on. `train` re-runs its own dataset and backend checks the moment the objective
returns, before the iteration makes another foreign call, and reads fresh pointers from them --
so freeing the training set from inside an objective signals `released-handle-error` naming
that dataset, exactly as freeing it anywhere else in this library does. Without that re-check
the loop hands a pointer into freed memory straight to C: measured, LightGBM died with
`Memory fault at 0x543447170e8a6` and XGBoost with `Signal 7 received`, killing the process
rather than signalling anything a caller could handle.

#### LightGBM forces `objective` to `"none"`

`LGBM_BoosterUpdateOneIterCustom` refuses to run at all while the booster holds an objective
function -- `Check failed: objective_function_ == nullptr`, measured against the vendored
library to return non-zero and train nothing -- so a non-`NIL` `:objective` on LightGBM
**overrides** any `objective` entry in `:parameters`, forcing it to `"none"` before
`LGBM_BoosterCreate` ever sees the string. This is not a convenience the caller can opt out of:
the combination it replaces has no working form to preserve. Every other parameter passes
through untouched and in its original order, `num_class` included, which is still what tells
LightGBM how many output groups a multiclass custom objective has.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(defparameter *co-matrix*
  (make-array '(8 1) :element-type 'double-float
              :initial-contents '((0.0d0) (1.0d0) (2.0d0) (3.0d0)
                                   (4.0d0) (5.0d0) (6.0d0) (7.0d0))))
(defparameter *co-label*
  (make-array 8 :element-type 'double-float
              :initial-contents '(0.0d0 1.0d0 4.0d0 9.0d0 16.0d0 25.0d0 36.0d0 49.0d0)))

(defun squared-error (scores)
  "GRAD = prediction - label, HESS = 1 -- squared error's own derivatives."
  (let* ((rows (array-dimension scores 0))
         (grad (make-array (list rows 1) :element-type 'double-float))
         (hess (make-array (list rows 1) :element-type 'double-float :initial-element 1.0d0)))
    (dotimes (row rows (values grad hess))
      (setf (aref grad row 0) (- (aref scores row 0) (aref *co-label* row))))))

;; A caller who explicitly names LightGBM's own "regression" objective in :parameters
;; alongside :objective still gets the identical model a run naming no objective there gets --
;; proof the override happened, not merely documented.
(let ((backend (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (dataset (cl-gbdt:make-dataset backend *co-matrix* :label *co-label*
                                          :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                        :verbose -1)))
        (cl-gbdt:with-booster
            (silent (cl-gbdt:train backend dataset :num-rounds 5
                                    :parameters '(:num-leaves 4 :min-data-in-leaf 1
                                                  :min-data-in-bin 1 :verbose -1
                                                  :boost-from-average nil)
                                    :objective #'squared-error))
          (cl-gbdt:with-booster
              (overridden (cl-gbdt:train backend dataset :num-rounds 5
                                          :parameters '(:objective "regression" :num-leaves 4
                                                        :min-data-in-leaf 1 :min-data-in-bin 1
                                                        :verbose -1 :boost-from-average nil)
                                          :objective #'squared-error))
            (format t "no objective named in :parameters, :raw:~%~S~%"
                    (cl-gbdt:predict silent *co-matrix* :kind :raw))
            (format t "\"regression\" named in :parameters too, :raw:~%~S~%"
                    (cl-gbdt:predict overridden *co-matrix* :kind :raw)))))
    (cl-gbdt:close-backend backend)))
```

Output:

```
no objective named in :parameters, :raw:
#2A((0.9006424501538277d0)
    (0.9006424501538277d0)
    (0.9006424501538277d0)
    (3.7816945374011977d0)
    (5.989342808723447d0)
    (11.236049175262446d0)
    (13.940538883209221d0)
    (19.681847476959206d0))
"regression" named in :parameters too, :raw:
#2A((0.9006424501538277d0)
    (0.9006424501538277d0)
    (0.9006424501538277d0)
    (3.7816945374011977d0)
    (5.989342808723447d0)
    (11.236049175262446d0)
    (13.940538883209221d0)
    (19.681847476959206d0))
```

Naming `"regression"` explicitly changes nothing: `train` drops every `objective` entry
`:parameters` holds and appends its own `:objective "none"` last, so the two runs above are the
identical booster. **XGBoost's `:parameters` are never rewritten** --
`XGBoosterTrainOneIter` has no such restriction, measured to accept a custom update with any
objective set -- so there is nothing on that backend to override.

**"Every `objective` entry" means all five spellings LightGBM honours**, not the literal one
alone. That library reads `objective_type`, `app`, `application` and `loss` as aliases for
`objective` -- its own `LGBM_DumpParamAliases` returns
`"objective": ["app", "loss", "application", "objective_type"]`, and each of the four is live
in the vendored 4.7.0, `:app "binary"` training the identical model `:objective "binary"`
trains. All five are dropped, so `:app "regression"` alongside `:objective #'squared-error`
behaves exactly like the `:objective "regression"` run above. The list can only be enumerated,
never prefix-matched: `apps` is *not* an alias, and neither is `objective_seed`, which is a
real LightGBM parameter in its own right -- dropping either would silently delete a caller's
configuration. Only keys that render to one of the five are touched; everything else passes
through in its original order.

Finally, a non-`NIL` `:objective` must be a `function`. A number, a string, or a *symbol*
naming a function signals `unsupported-argument` naming `train's :objective`, before any
foreign call and so before a booster exists -- on both backends. The symbol case is refused
deliberately: `funcall` would have accepted it and resolved it afresh at each iteration
against whatever global definition happened to be in force, rather than against what the
caller passed.

#### The remaining divergence: what `:normal` means under a custom objective

Because LightGBM's objective is forced to `"none"` while XGBoost's stays whatever the caller
configured, the two backends disagree about what `predict`'s `:kind :normal` means under a
custom objective. LightGBM applies no transform at all, so `:normal` equals `:raw` there. A
configured XGBoost objective's own prediction transform stays in effect regardless of who
supplied the gradient, so with `binary:logistic` still set, `:normal` returns probabilities of
a margin the caller's own loss produced, while `:raw` returns that margin untouched:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *co-matrix*
  (make-array '(8 1) :element-type 'double-float
              :initial-contents '((0.0d0) (1.0d0) (2.0d0) (3.0d0)
                                   (4.0d0) (5.0d0) (6.0d0) (7.0d0))))
(defparameter *co-binary-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defun logistic-objective (scores)
  "GRAD/HESS for logistic loss over *CO-BINARY-LABEL*, from the raw margin SCORES."
  (let* ((rows (array-dimension scores 0))
         (grad (make-array (list rows 1) :element-type 'double-float))
         (hess (make-array (list rows 1) :element-type 'double-float)))
    (dotimes (row rows (values grad hess))
      (let ((p (/ 1.0d0 (+ 1.0d0 (exp (- (aref scores row 0)))))))
        (setf (aref grad row 0) (- p (aref *co-binary-label* row)))
        (setf (aref hess row 0) (max 1d-6 (* p (- 1.0d0 p))))))))

;; LightGBM's :normal equals its :raw under a custom objective, since :objective forces
;; "objective":"none". XGBoost's configured objective keeps transforming: :normal differs
;; from :raw there.
(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (unwind-protect
      (progn
        (cl-gbdt:with-dataset
            (dataset (cl-gbdt:make-dataset lgbm *co-matrix* :label *co-binary-label*
                                            :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                          :verbose -1)))
          (cl-gbdt:with-booster
              (booster (cl-gbdt:train lgbm dataset :num-rounds 5
                                       :parameters '(:num-leaves 4 :min-data-in-leaf 1
                                                     :min-data-in-bin 1 :verbose -1)
                                       :objective #'logistic-objective))
            (format t "LightGBM :normal:~%~S~%" (cl-gbdt:predict booster *co-matrix* :kind :normal))
            (format t "LightGBM :raw:~%~S~%" (cl-gbdt:predict booster *co-matrix* :kind :raw))))
        (cl-gbdt:with-dataset
            (dataset (cl-gbdt:make-dataset xgb *co-matrix* :label *co-binary-label*))
          (cl-gbdt:with-booster
              (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                       :parameters '(:objective "binary:logistic" :max-depth 2
                                                     :eta 0.5d0 :verbosity 0)
                                       :objective #'logistic-objective))
            (format t "XGBoost  :normal:~%~S~%" (cl-gbdt:predict booster *co-matrix* :kind :normal))
            (format t "XGBoost  :raw:~%~S~%" (cl-gbdt:predict booster *co-matrix* :kind :raw)))))
    (cl-gbdt:close-backend lgbm)
    (cl-gbdt:close-backend xgb)))
```

Output:

```
LightGBM :normal:
#2A((-0.857090443203849d0)
    (-0.857090443203849d0)
    (-0.857090443203849d0)
    (-0.857090443203849d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0))
LightGBM :raw:
#2A((-0.857090443203849d0)
    (-0.857090443203849d0)
    (-0.857090443203849d0)
    (-0.857090443203849d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0))
XGBoost  :normal:
#2A((0.3775406777858734d0)
    (0.3775406777858734d0)
    (0.3775406777858734d0)
    (0.3775406777858734d0)
    (0.622459352016449d0)
    (0.622459352016449d0)
    (0.622459352016449d0)
    (0.622459352016449d0))
XGBoost  :raw:
#2A((-0.5d0) (-0.5d0) (-0.5d0) (-0.5d0) (0.5d0) (0.5d0) (0.5d0) (0.5d0))
```

LightGBM's `:normal` and `:raw` match to the last digit; XGBoost's do not, and its `:normal`
values sit in `(0, 1)`, a sigmoid of its own `:raw` margin. One custom-objective run, two
meanings for `:normal` -- a caller moving the same `:objective` function between backends gets
a probability from one and a margin from the other unless they account for it.

#### What `:objective` sees on XGBoost's training set, and DART

XGBoost has no counterpart to LightGBM's `LGBM_BoosterGetPredict`, which simply hands back
scores the booster already holds; each iteration's scores for `:objective` are instead a fresh
margin prediction over the training `DMatrix`, sent with `"training":true` in that call's
config JSON. The vendored header
(`ffi-spec/xgboost/include/xgboost/c_api.h:1180-1191`) documents that key as distinguishing two
prediction scenarios, obtaining `y_pred` versus "obtain[ing] the prediction for computing
gradients", and says the second "applies when you are defining a custom objective function".
On the default `gbtree` booster the two values were measured to train identical models; the
same header names DART's training-time dropout as the case where they differ, since dropped
trees make "the prediction result... different from the one obtained by normal inference step".
**That DART difference is the header's own statement, not a measurement taken in this
repository** -- no test here exercises `:booster "dart"` together with `:objective`, though
`:booster` reaches XGBoost's parameters untouched, so the combination is reachable and simply
unmeasured.

Two further things this argument does not change. A library metric configured through
`:parameters` relates to the library's own objective, not to the caller's, so what
`:record-history` records and what `:early-stopping` watches -- see [Training
report](#training-report) -- may be meaningless under a custom objective; nothing here signals
about that, the caller decides. And `:objective` is `funcall`ed inside `train`'s own
floating-point-trap mask, so the caller's Lisp arithmetic runs under the same masked convention
the two C libraries are written against: `(/ 1.0d0 0.0d0)` yields infinity rather than
signalling, on x86-64 as well as on aarch64.

### Custom evaluation

`train` also takes `:evaluation`, a function called once per dataset per iteration, after
that iteration's update, so a run records the caller's own measure of fit beside the
library's own metrics. It needs the `:custom-evaluation` capability, answerable through
`backend-supports-p` and true on both vendored backends; `train` re-checks it itself and
signals `capability-unavailable` for a non-`NIL` `:evaluation` when it reads false, before
any foreign call.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *ce-matrix*
  (make-array '(8 2) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                   (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                   (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *ce-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defun my-logloss (scores index)
  "A caller-written binary log loss over SCORES -- predict :kind :normal's probabilities, not
the margin :objective would see. INDEX is ignored here since both datasets below share
*CE-LABEL*."
  (declare (ignore index))
  (let ((rows (array-dimension scores 0)) (sum 0d0))
    (dotimes (row rows (values "my_logloss" (/ sum rows)))
      (let ((p (min (max (aref scores row 0) 1d-15) (- 1d0 1d-15)))
            (y (coerce (aref *ce-label* row) 'double-float)))
        (incf sum (- (+ (* y (log p)) (* (- 1d0 y) (log (- 1d0 p))))))))))

(defun show-custom-evaluation (name backend dataset-parameters booster-parameters reference-p)
  (cl-gbdt:with-dataset (train-set (apply #'cl-gbdt:make-dataset backend *ce-matrix*
                                          :label *ce-label* dataset-parameters))
    (cl-gbdt:with-dataset (valid-set (apply #'cl-gbdt:make-dataset backend *ce-matrix*
                                            :label *ce-label*
                                            (append (when reference-p
                                                      (list :reference train-set))
                                                    dataset-parameters)))
      (multiple-value-bind (booster report)
          (cl-gbdt:train backend train-set :num-rounds 5
                          :valid-sets (list (cons "valid" valid-set))
                          :parameters booster-parameters
                          :evaluation #'my-logloss)
        (unwind-protect
             (dolist (series (cl-gbdt:training-report-series report))
               (format t "~A series: index=~S metric=~S last=~S~%"
                       name (cl-gbdt:training-series-index series)
                       (cl-gbdt:training-series-metric series)
                       (aref (cl-gbdt:training-series-values series) 4)))
          (cl-gbdt:free-booster booster))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (show-custom-evaluation "LightGBM" lgbm
        '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
        '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
          :verbose -1 :metric "binary_logloss")
        t)
  (show-custom-evaluation "XGBoost " xgb '()
        '(:objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0
          :eval-metric "logloss")
        nil)
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM series: index=0 metric="binary_logloss" last=0.35374722486733523d0
LightGBM series: index=1 metric="binary_logloss" last=0.35374722486733523d0
LightGBM series: index=0 metric="my_logloss" last=0.35374722486733523d0
LightGBM series: index=1 metric="my_logloss" last=0.35374722486733523d0
XGBoost  series: index=0 metric="logloss" last=0.4740770012140274d0
XGBoost  series: index=1 metric="logloss" last=0.4740770012140274d0
XGBoost  series: index=0 metric="my_logloss" last=0.47407697467999527d0
XGBoost  series: index=1 metric="my_logloss" last=0.47407697467999527d0
```

`MY-LOGLOSS` above reimplements the same binary log loss both libraries already compute, and
its series lands on each backend's own to the last few digits -- not because the two are
forced to agree, but because a caller-written metric over the same probabilities the library
scored really does compute the same number. It also shows the ordering `training-report-series`
holds to: the two `training-series` for `"binary_logloss"`/`"logloss"` -- one per dataset,
library metrics first -- come before either of `"my_logloss"`'s, on both backends. `train`'s
generic docstring states this as a guarantee rather than an accident of this example: the
library's own series are exactly what `evaluation` already reports, in the same order, and
`:evaluation`'s own entries are APPENDED after every one of them, so they form a PREFIX of
`training-report-series` (`src/protocol.lisp`). The append itself happens once per
backend, in `%custom-evaluation-entries`
(`src/lightgbm/protocol.lisp`/`src/xgboost/protocol.lisp`), and
`training-report-from-history` preserves first-seen order rather than sorting anything
(`src/training/history.lisp`), which is what turns "appended last" into "prefix" once
the whole run's history is folded. `evaluation` itself never reports a custom metric -- it
asks the library what the library computed, and the library never computed this one
(`src/protocol.lisp`).

`:evaluation` is called with **two arguments**: SCORES, that dataset's current predictions as
a `(ROWS GROUPS)` `double-float` array, and the dataset's **INDEX** -- `0` for the training
set, `N+1` for the Nth `:valid-sets` entry, the same numbering `:early-stopping`'s `:dataset`
key and `evaluation`'s own `DATASET-INDEX` already use
(`src/protocol.lisp`). It must return **two values**, a metric NAME (a string) and a
VALUE (a real or `NIL`); a NAME that is not a string, or a VALUE that is neither, signals
`unsupported-argument` (`custom-metric-entry` in `src/training/custom-metric.lisp`).
A real VALUE is **recorded as a `double-float`**, coerced where the entry is built rather than
stored as returned: `training-series-values` documents every element of every series as a
`double-float` or `NIL`, and both libraries' own values already are doubles, so a caller
returning `1/3` reads `0.3333333333333333d0` back out of its own series rather than a `ratio`
landing in a slot every other consumer was promised held doubles (same file). A real too large
for a `double-float` to hold records the signed infinity, identically on every platform: the
coercion is wrapped the same way `%rational-json` (`src/config/missing-value.lisp`) wraps its
own, because whether `coerce` *signals* on such a value is a property of the platform's
floating-point traps rather than of the value -- see that function's docstring, which records
the split.
`NIL` means "not computable this iteration" -- a fold whose
denominator was zero, a metric undefined before some minimum number of rows -- and is
recorded in its place in the series rather than dropped, counting as no improvement to an
`:early-stopping` watcher exactly as a value the backend itself could not report does
(`src/protocol.lisp`).

SCORES is what `predict :kind :normal` returns for that dataset, and **not** the margin
`:objective` is handed -- with a classification objective configured, these are the
transformed probabilities. INDEX is not decorative: `predict :kind :normal` on the trained
booster, and the array `:evaluation` was handed for that same dataset during the run, are the
identical array, checked below for both the training set (index 0) and the one `:valid-sets`
entry (index 1):

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(defparameter *ce-matrix*
  (make-array '(8 2) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                   (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                   (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *ce-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (train-set (cl-gbdt:make-dataset lgbm *ce-matrix* :label *ce-label*
                       :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)))
        (cl-gbdt:with-dataset
            (valid-set (cl-gbdt:make-dataset lgbm *ce-matrix* :label *ce-label*
                         :reference train-set
                         :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)))
          (let ((last-scores (make-array 2 :initial-element nil)))
            (cl-gbdt:with-booster
                (booster (cl-gbdt:train lgbm train-set :num-rounds 5
                           :valid-sets (list valid-set)
                           :parameters '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1
                                         :min-data-in-bin 1 :verbose -1)
                           ;; INDEX is 0 for the training set, 1 for the first (and only)
                           ;; :valid-sets entry -- the same numbering :early-stopping's
                           ;; :dataset key already uses.
                           :evaluation (lambda (scores index)
                                         (setf (aref last-scores index) scores)
                                         (values "captured" 0.0d0))))
              (format t "index 0's SCORES is predict :kind :normal's: ~S~%"
                      (equalp (aref last-scores 0)
                              (cl-gbdt:predict booster *ce-matrix* :kind :normal)))
              (format t "index 1's SCORES is predict :kind :normal's: ~S~%"
                      (equalp (aref last-scores 1)
                              (cl-gbdt:predict booster *ce-matrix* :kind :normal)))
              (format t "index 0's SCORES is NOT predict :kind :raw's: ~S~%"
                      (not (equalp (aref last-scores 0)
                                   (cl-gbdt:predict booster *ce-matrix* :kind :raw))))))))
    (cl-gbdt:close-backend lgbm)))
```

Output:

```
index 0's SCORES is predict :kind :normal's: T
index 1's SCORES is predict :kind :normal's: T
index 0's SCORES is NOT predict :kind :raw's: T
```

This is measured differently on each backend, and the two measurements are not the same
kind of fact: on LightGBM, `%booster-predictions` reads `LGBM_BoosterGetPredict`, a value the
library already holds rather than a fresh prediction, so agreeing with `predict` says two
different C functions agree -- measured on both a 40-row training set and a 17-row
validation set to `0.0`, and `0.706` away from `:raw` under `objective=binary`
(`src/lightgbm/native.lisp`). On XGBoost, `%booster-predictions` runs a fresh
`XGBoosterPredictFromDMatrix` prediction pass over that dataset's own `DMatrix`, the same
call `predict` itself makes, so agreeing says that `DMatrix` and a fresh one built from the
same rows answer alike -- also measured to `0.0` on both datasets, and `0.756` away from
`:raw` under `binary:logistic` after five iterations
(`src/xgboost/native.lisp`). Neither figure stands in for the other's,
and the two are not compared -- policy section 13; what they share is only that both
backends' SCORES equal `predict :kind :normal`'s. Under a custom `:objective` the two then
part company exactly as [Custom objective](#custom-objective) already describes: LightGBM
forces `objective=none`, so `:normal` and `:raw` coincide there and SCORES equals what
`:objective` was handed, while XGBoost rewrites nothing, so a configured `binary:logistic`
keeps transforming and `:evaluation` reads probabilities while `:objective` reads the margin
behind them, in the same run.

A custom metric's values become **series of their own** in the report, one per (INDEX, NAME)
pair, indistinguishable in shape from the library's own -- so `:early-stopping` can watch one
with nothing extra arranged for it, by giving `:metric` the name the function returns and
`:dataset` the index it was returned for:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(defparameter *ce-matrix*
  (make-array '(8 2) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                   (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                   (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *ce-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defun stalling-metric ()
  "3.0, 2.0, 1.0, 1.0, 1.0, ... -- ignores SCORES entirely. What is under test is that a
caller's own metric name reaches :early-stopping at all, not that a real model produced it."
  (let ((calls 0))
    (lambda (scores index)
      (declare (ignore scores index))
      (incf calls)
      (values "stalls" (coerce (max 1 (- 4 calls)) 'double-float)))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (train-set (cl-gbdt:make-dataset lgbm *ce-matrix* :label *ce-label*
                       :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)))
        (multiple-value-bind (booster report)
            (cl-gbdt:train lgbm train-set :num-rounds 10
                            ;; The library's own metric turned off, so the only series in the
                            ;; report -- and the only thing :early-stopping could be watching --
                            ;; is the caller's own.
                            :parameters '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1
                                          :min-data-in-bin 1 :verbose -1 :metric "none")
                            :evaluation (stalling-metric)
                            :early-stopping (list :metric "stalls" :dataset 0
                                                   :direction :lower-is-better :rounds 2))
          (cl-gbdt:free-booster booster)
          (format t "ran ~S of 10 rounds~%" (cl-gbdt:training-report-num-rounds report))
          (format t "early-stopped-p: ~S~%" (cl-gbdt:training-report-early-stopped-p report))
          (format t "best-iteration: ~S~%" (cl-gbdt:training-report-best-iteration report))
          (format t "best-score: ~S~%" (cl-gbdt:training-report-best-score report))))
    (cl-gbdt:close-backend lgbm)))
```

Output:

```
ran 5 of 10 rounds
early-stopped-p: T
best-iteration: 3
best-score: 1.0d0
```

The value improves at iterations 1, 2 and 3 and then holds at `1.0`; improvement is strict, so
a plateau does not count, and two consecutive non-improving iterations (`:rounds 2`) stop the
run at iteration 5 with iteration 3 recorded best -- driven entirely by a metric that never
reads its SCORES argument, which is the point: what reaches the watcher is the (INDEX, NAME)
pair `:evaluation` returned, the same mechanism a library metric reaches it through, not
anything specific to `:evaluation`.

`train` refuses a non-`NIL` `:evaluation` in two more shapes, both checked before any foreign
call. `:record-history nil` together with `:evaluation` signals `unsupported-argument`: a
custom metric's whole result is the per-iteration series `:record-history nil` exists not to
build, so the values would be computed at full cost and then dropped -- the same
contradiction `:early-stopping` and `:record-history nil` already make. And `:evaluation`
must be a `function`; a number, a string, or a **symbol** naming a real function of the right
arity all signal `unsupported-argument` naming `:evaluation` -- the symbol deliberately,
since `funcall` would have accepted it happily and resolved it afresh each iteration against
whatever global definition happened to be in force, rather than against what the caller
passed:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *ce-matrix*
  (make-array '(8 2) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                   (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                   (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *ce-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defparameter *ce-booster-parameters*
  '((:lightgbm :objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
     :verbose -1)
    (:xgboost :objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0)))
(defparameter *ce-dataset-parameters*
  '((:lightgbm :parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
    (:xgboost)))

(defun a-constant-metric (scores index)
  "A SYMBOL naming a real function of the right arity, so `funcall' would have accepted it
happily -- which is exactly why :evaluation refuses it explicitly rather than leaving the
mistake to surface some other way."
  (declare (ignore scores index))
  (values "constant" 0.5d0))

(dolist (name '(:lightgbm :xgboost))
  (let ((backend (cl-gbdt:open-backend name)))
    (unwind-protect
        (cl-gbdt:with-dataset
            (dataset (apply #'cl-gbdt:make-dataset backend *ce-matrix* :label *ce-label*
                             (cdr (assoc name *ce-dataset-parameters*))))
          (format t "~A :evaluation with :record-history nil:~%" name)
          (handler-case
              (cl-gbdt:free-booster
               (cl-gbdt:train backend dataset :num-rounds 3 :record-history nil
                               :parameters (cdr (assoc name *ce-booster-parameters*))
                               :evaluation (lambda (scores index)
                                             (declare (ignore scores index))
                                             (values "x" 0.0d0))))
            (error (c) (format t "  SIGNALED ~A: ~A~%" (type-of c) c)))
          (dolist (value (list 42 'a-constant-metric))
            (format t "~A :evaluation ~S:~%" name value)
            (handler-case
                (cl-gbdt:free-booster
                 (cl-gbdt:train backend dataset :num-rounds 3
                                 :parameters (cdr (assoc name *ce-booster-parameters*))
                                 :evaluation value))
              (error (c) (format t "  SIGNALED ~A: ~A~%" (type-of c) c)))))
      (cl-gbdt:close-backend backend))))
```

Output:

```
LIGHTGBM :evaluation with :record-history nil:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by LIGHTGBM: a custom metric is recorded per iteration, which :record-history NIL skips; pass :record-history T, or drop :evaluation.
LIGHTGBM :evaluation 42:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by LIGHTGBM: the custom metric must be a function of two arguments, or NIL for the library's own metrics only -- got 42.
LIGHTGBM :evaluation A-CONSTANT-METRIC:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by LIGHTGBM: the custom metric must be a function of two arguments, or NIL for the library's own metrics only -- got A-CONSTANT-METRIC.
XGBOOST :evaluation with :record-history nil:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by XGBOOST: a custom metric is recorded per iteration, which :record-history NIL skips; pass :record-history T, or drop :evaluation.
XGBOOST :evaluation 42:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by XGBOOST: the custom metric must be a function of two arguments, or NIL for the library's own metrics only -- got 42.
XGBOOST :evaluation A-CONSTANT-METRIC:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by XGBOOST: the custom metric must be a function of two arguments, or NIL for the library's own metrics only -- got A-CONSTANT-METRIC.
```

Both checks live in each backend's own `%check-custom-evaluation`
(`src/lightgbm/protocol.lisp`/`src/xgboost/protocol.lisp`), ahead of the capability
check that runs first; identical wording on both backends because both call the same
backend-neutral checks underneath.

A NAME colliding with one the library itself reports for the **same** dataset index signals
`unsupported-argument` too -- checked at the end of the first iteration, the first moment
there is a real evaluation to compare against. The pair (INDEX, NAME) is what a series is
keyed by, so two different quantities under one pair would corrupt the series rather than
produce two; what is compared is what this booster **actually reported**, not a list of
well-known metric names, which is why the identical name is accepted the moment the library
reports no metric at all:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *ce-matrix*
  (make-array '(8 2) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                   (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                   (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *ce-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defparameter *ce-library-metric-names*
  '((:lightgbm . "binary_logloss") (:xgboost . "logloss")))

;; One booster parameter plist per backend WITH the library's metric on, and one WITHOUT --
;; LightGBM's own "metric none", XGBoost's own "disable_default_eval_metric 1".
(defparameter *ce-with-metric*
  '((:lightgbm :objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
     :verbose -1 :metric "binary_logloss")
    (:xgboost :objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0
     :eval-metric "logloss")))
(defparameter *ce-without-metric*
  '((:lightgbm :objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
     :verbose -1 :metric "none")
    (:xgboost :objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0
     :disable-default-eval-metric 1)))
(defparameter *ce-dataset-parameters*
  '((:lightgbm :parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
    (:xgboost)))

(dolist (name '(:lightgbm :xgboost))
  (let ((backend (cl-gbdt:open-backend name))
        (library-name (cdr (assoc name *ce-library-metric-names*))))
    (unwind-protect
        (cl-gbdt:with-dataset
            (dataset (apply #'cl-gbdt:make-dataset backend *ce-matrix* :label *ce-label*
                             (cdr (assoc name *ce-dataset-parameters*))))
          (flet ((train-named (parameters)
                   (cl-gbdt:train backend dataset :num-rounds 3 :parameters parameters
                                   :evaluation (lambda (scores index)
                                                 (declare (ignore scores index))
                                                 (values library-name 0.5d0)))))
            (format t "~A :evaluation returns ~S, the library's own name, while it is ~
                       configured:~%" name library-name)
            (handler-case
                (cl-gbdt:free-booster
                 (train-named (cdr (assoc name *ce-with-metric*))))
              (error (c) (format t "  SIGNALED ~A: ~A~%" (type-of c) c)))
            (format t "~A :evaluation returns ~S while the library reports no metric at ~
                       all:~%" name library-name)
            (multiple-value-bind (booster report)
                (train-named (cdr (assoc name *ce-without-metric*)))
              (cl-gbdt:free-booster booster)
              (format t "  accepted; series pairs: ~S~%"
                      (mapcar (lambda (series)
                                (cons (cl-gbdt:training-series-index series)
                                      (cl-gbdt:training-series-metric series)))
                              (cl-gbdt:training-report-series report))))))
      (cl-gbdt:close-backend backend))))
```

Output:

```
LIGHTGBM :evaluation returns "binary_logloss", the library's own name, while it is configured:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by LIGHTGBM: "binary_logloss" already names a metric the library reports for dataset index 0.
LIGHTGBM :evaluation returns "binary_logloss" while the library reports no metric at all:
  accepted; series pairs: ((0 . "binary_logloss"))
XGBOOST :evaluation returns "logloss", the library's own name, while it is configured:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by XGBOOST: "logloss" already names a metric the library reports for dataset index 0.
XGBOOST :evaluation returns "logloss" while the library reports no metric at all:
  accepted; series pairs: ((0 . "logloss"))
```

The check itself is backend-neutral, `check-metric-name-collision`
(`src/training/custom-metric.lisp`), given each iteration's own library entries to
compare against rather than a static list. The same name at a **different** index does not
collide -- not constructible against either vendored library, since both report the same
metric list for every dataset they retain, so this project's own assertion of that half of
the keying is at layer 1, over a written entry list rather than a measured one
(`check-metric-name-collision-allows-a-name-the-library-uses-elsewhere` in
`tests/custom-metric.lisp`).

`:evaluation` must return **the same name for a given index on every iteration** of the run.
Two indices may return two different names; what is refused is one index's name changing
between iterations. The first iteration's name is remembered per index, and a later iteration
returning a different one for that index signals `unsupported-argument` naming `:evaluation`,
mid-run and at the very call that changed it:

```
train's :evaluation is not supported by XGBOOST: the custom metric returned name
"another_metric" for dataset index 0 after returning "my_logloss" for it; one name per dataset
index is required for the whole run.
```

This is a requirement rather than a nicety, and it is what keeps `train`'s promise that
**every series is exactly `training-report-num-rounds` long** true of a caller's own series as
well as the library's. A series holds one value per completed iteration, keyed by the (INDEX,
NAME) pair, so a name that varies asks for something no series can be:

- Varying **without ever colliding** gives one series per name it took, each pushed only on
  the iterations that name appeared in -- several series, all shorter than the run, each
  misaligned with the iterations its values were measured at.
- Varying **into the library's own name** for that index is the case the collision check
  above cannot reach, since that check runs on the first iteration only: a caller returning a
  safe name then and a colliding one afterwards walks straight past it. From the iteration the
  two names meet, one key collects two values per iteration and its series comes out
  `1 + 2(N-1)` values long over N rounds -- *longer* than `training-report-num-rounds` says the
  run was, and silently. It would also break the "at most one entry matches a given (index,
  metric) pair" invariant `:early-stopping` reads under: `find-if` returns the first of the
  two, so a watcher would read one value per iteration and never learn the other existed
  (`%find-watched-entry` in `src/training/early-stopping.lisp`).

Pinning the name closes both, and closes the second without a second collision check: once
each index's name is fixed at the first iteration, the only name that can ever reach the
library's is the one the collision check already compared. Like the collision check it is
backend-neutral -- `make-metric-name-pin` and `pin-metric-name`
(`src/training/custom-metric.lisp`), one pin per `train` call.

Returning **one string object and rewriting it in place** is refused on exactly the same
terms, and it is not something the pin could have caught by itself. A string is mutable, so a
caller keeping one name buffer and refilling it each iteration would have handed the pin an
object that compares `string=` with itself however its characters changed -- and every
recorded entry would have held that same object, to be read once, at the end of the run, under
whatever the name said by then. Measured before this was closed, four rounds on LightGBM with
`metric "binary_logloss"` and a 14-character name rewritten to `"binary_logloss"` from the
second iteration: `train` returned normally, nothing signalled, and the report held a single
**eight-value series for a four-round run**. What closes it is that `custom-metric-entry`
`copy-seq`s the name into the entry it builds, and both `train` methods take the name back
**out of that entry** for the collision check and the pin -- so the history, the pin and the
collision check all hold one snapshot and the caller's own object reaches none of them
(`src/training/custom-metric.lisp`,
`%custom-evaluation-entries` in `src/lightgbm/protocol.lisp`/`src/xgboost/protocol.lisp`).

A NAME that is not a string at all, or a VALUE that is neither a real nor `NIL`, is refused
the same way and at the same point in the run -- `custom-metric-entry`, in that same file --
so `(values :my-metric 0.25d0)` and `(values "my_logloss" "0.25")` each signal
`unsupported-argument` naming `:evaluation` rather than reaching the report.

`:evaluation` runs inside `train`'s own floating-point-trap mask, on the same terms
`:objective` does (see [Custom objective](#custom-objective)): the caller's own arithmetic
does not trap, so `(/ 1.0d0 0.0d0)` yields infinity rather than signalling
`division-by-zero`, on x86-64 as well as on aarch64. A handle it frees, or a backend it
closes, is caught the moment it returns and before the next dataset's predictions are read --
`%custom-evaluation-entries` re-checks between two consecutive datasets rather than once per
iteration, which is what makes freeing a `:valid-sets` entry from inside the FIRST dataset's
call signal `released-handle-error` naming that dataset rather than faulting the process on
the second dataset's read.

#### The measured cost

A custom metric adds a `predict :kind :normal`-shaped array read plus a Lisp call, per
dataset per iteration, on top of what `:record-history` already reads. Measured the same way
as [that section](#turning-recording-off-record-history) -- 2000 rows x 20 columns, one
validation set, `:record-history t` in **both** arms so the only difference is whether
`:evaluation` is supplied:

```lisp
;;;; Same shape of fixture as the :record-history measurement above: 2000 rows x 20 columns,
;;;; 1500 rounds, one validation set. RECORD-HISTORY is T in both arms, so the only difference
;;;; between the two is whether :evaluation is supplied.

(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *rows* 2000)
(defparameter *valid-rows* 500)
(defparameter *columns* 20)
(defparameter *rounds* 1500)

(defun make-fixture-matrix (rows columns offset)
  (let ((matrix (make-array (list rows columns) :element-type 'double-float)))
    (dotimes (row rows matrix)
      (dotimes (col columns)
        (setf (aref matrix row col)
              (coerce (mod (+ (* 7 (+ row offset)) (* 13 col)) 97) 'double-float))))))

(defun make-fixture-label (rows offset)
  (let ((label (make-array rows :element-type 'single-float)))
    (dotimes (row rows label)
      (setf (aref label row) (if (evenp (+ row offset)) 1.0 0.0)))))

(defparameter *train-matrix* (make-fixture-matrix *rows* *columns* 0))
(defparameter *train-label* (make-fixture-label *rows* 0))
(defparameter *valid-matrix* (make-fixture-matrix *valid-rows* *columns* 5))
(defparameter *valid-label* (make-fixture-label *valid-rows* 5))

(defun my-metric (scores index)
  "A representative caller-written metric: mean log loss over SCORES against this run's own
label vector for dataset INDEX -- real arithmetic over every row, not a constant, so the
measurement includes a Lisp-side cost proportional to row count and not just the array read."
  (let* ((labels* (if (zerop index) *train-label* *valid-label*))
         (rows (array-dimension scores 0))
         (sum 0d0))
    (dotimes (row rows)
      (let ((p (min (max (aref scores row 0) 1d-15) (- 1d0 1d-15)))
            (y (coerce (aref labels* row) 'double-float)))
        (incf sum (- (+ (* y (log p)) (* (- 1d0 y) (log (- 1d0 p))))))))
    (values "my_logloss" (/ sum rows))))

(defun run-once (backend-name make-dataset-parameters booster-parameters reference-p
                  evaluation-p)
  "Train once and return the wall-clock seconds, :record-history T throughout. EVALUATION-P T
supplies :EVALUATION #'MY-METRIC; NIL supplies none."
  (let ((backend (cl-gbdt:open-backend backend-name)))
    (unwind-protect
         (cl-gbdt:with-dataset
             (train-set (apply #'cl-gbdt:make-dataset backend *train-matrix*
                                :label *train-label* make-dataset-parameters))
           (cl-gbdt:with-dataset
               (valid-set (apply #'cl-gbdt:make-dataset backend *valid-matrix*
                                  :label *valid-label*
                                  (append (when reference-p (list :reference train-set))
                                          make-dataset-parameters)))
             (let ((start (get-internal-real-time)))
               (multiple-value-bind (booster report)
                   (apply #'cl-gbdt:train backend train-set
                          :valid-sets (list valid-set) :num-rounds *rounds*
                          :record-history t :parameters booster-parameters
                          (when evaluation-p (list :evaluation #'my-metric)))
                 (declare (ignore report))
                 (cl-gbdt:free-booster booster))
               (/ (float (- (get-internal-real-time) start) 1d0)
                  internal-time-units-per-second))))
      (cl-gbdt:close-backend backend))))

(defun report-timing (backend-name make-dataset-parameters booster-parameters reference-p)
  ;; One untimed warm-up run per arm first, then 5 timed runs each, interleaved
  ;; WITHOUT/WITH/WITHOUT/... so neither arm is systematically first or last.
  (run-once backend-name make-dataset-parameters booster-parameters reference-p nil)
  (run-once backend-name make-dataset-parameters booster-parameters reference-p t)
  (let ((without '()) (with '()))
    (dotimes (i 5)
      (push (run-once backend-name make-dataset-parameters booster-parameters reference-p nil)
            without)
      (push (run-once backend-name make-dataset-parameters booster-parameters reference-p t)
            with))
    (setf without (nreverse without) with (nreverse with))
    (format t "~A without :evaluation: ~{~,3F~^ ~} seconds (mean ~,3F)~%"
            backend-name without (/ (reduce #'+ without) (length without)))
    (format t "~A with    :evaluation: ~{~,3F~^ ~} seconds (mean ~,3F)~%"
            backend-name with (/ (reduce #'+ with) (length with)))
    (format t "~A ratio (with / without): ~,3F~%"
            backend-name (/ (/ (reduce #'+ with) (length with))
                             (/ (reduce #'+ without) (length without))))))

(report-timing :lightgbm
               '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
               '(:objective "binary" :num-leaves 31 :verbose -1 :metric "binary_logloss,auc"
                 :min-data-in-leaf 1 :min-data-in-bin 1)
               t)

(report-timing :xgboost
               '()
               '(:objective "binary:logistic" :max-depth 6 :eta 0.3d0 :verbosity 0
                 :eval-metric "logloss" :eval-metric "error")
               nil)
```

Output, two independent runs:

```
LIGHTGBM without :evaluation: 2.614 2.654 2.783 2.467 2.668 seconds (mean 2.637)
LIGHTGBM with    :evaluation: 2.946 3.053 2.790 2.946 3.025 seconds (mean 2.952)
LIGHTGBM ratio (with / without): 1.119
XGBOOST without :evaluation: 0.353 0.643 0.445 0.555 0.536 seconds (mean 0.506)
XGBOOST with    :evaluation: 0.703 0.697 0.700 0.818 0.830 seconds (mean 0.750)
XGBOOST ratio (with / without): 1.480
```

```
LIGHTGBM without :evaluation: 2.458 2.542 2.466 2.646 2.552 seconds (mean 2.533)
LIGHTGBM with    :evaluation: 3.062 2.899 2.811 2.983 2.831 seconds (mean 2.917)
LIGHTGBM ratio (with / without): 1.152
XGBOOST without :evaluation: 0.535 0.551 0.492 0.442 0.609 seconds (mean 0.526)
XGBOOST with    :evaluation: 0.720 0.764 0.859 0.875 0.760 seconds (mean 0.796)
XGBOOST ratio (with / without): 1.513
```

`:evaluation` added roughly **12-15%** to LightGBM's wall-clock `train` time here, and
roughly **48-51%** to XGBoost's. The two are each this backend's own ratio and are not
compared with one another -- policy section 13 -- and the gap between them is explained by
the same mechanism the SCORES paragraph above already measured: LightGBM's per-dataset read
is a cached value the booster already holds, so the added cost is close to the Lisp call and
the array copy alone, while XGBoost's is a whole extra `XGBoosterPredictFromDMatrix` pass per
dataset per iteration -- a real prediction, not a cached read, which is the more expensive of
the two operations on either backend. Treat these as orders of magnitude on one machine, not
precise figures -- run-to-run variance on the same code was as wide as 1.12-1.15 for LightGBM and
1.48-1.51 for XGBoost across the two runs above, and an earlier pair of runs at 500 rounds
(a fifth of the round count, and so closer to the noise floor of process startup and dataset
construction) ranged 1.08-1.15 for LightGBM and 1.20-2.27 for XGBoost.

`:custom-evaluation` is answerable through `backend-supports-p` on both backends, but the two
true answers come out of different lists for a reason that is a fact about the two
*libraries* rather than a difference a caller of `:evaluation` can see: LightGBM's
per-dataset read needs three C functions, none of them in that backend's required set, so it
is PROBED like `:custom-objective`; XGBoost's needs one, which IS required there, so it is
DECLARED (`src/backend.lisp`). Every check `:evaluation` is put through, and
every error it can signal, is identical prose on both backends, as the refusal output above
shows -- so this is not a row in [the differences table](#where-the-two-backends-genuinely-differ):
there is nothing here a caller's code, as opposed to a reader of `backend-info`'s probed
plist, can tell apart.

## Systems

| System | Purpose |
|---|---|
| `cl-gbdt` | Core: package, condition hierarchy, matrix marshalling, backend registry and `open-backend` protocol, the unified API's generic functions -- no methods, and no shared library required to load it |
| `cl-gbdt/lightgbm` | **Layer 1 for LightGBM, and nothing above it.** Library discovery and the `%`-wrappers (`src/lightgbm/native.lisp`) over the generated CFFI bindings (`src/lightgbm/c-api.lisp`), plus the backend's CLOS types and the `initialize-backend`/`shutdown-backend` pair that opens and closes the shared library (`src/lightgbm/classes.lisp`), plus the six finished operations a caller invokes -- `create-dataset`, `create-booster`, `update-one-iteration`, `predict`, `free-dataset`, `free-booster` (`src/lightgbm/api.lisp`) -- published together by `src/lightgbm/all.lisp`. Carries none of the 13 unified-API methods, and does not define the `cl-gbdt` package |
| `cl-gbdt/lightgbm/unified` | That plus all 13 unified-API methods (`src/lightgbm/protocol.lisp`), aggregated by `src/lightgbm/unified.lisp`, which also depends on core `cl-gbdt` so the `cl-gbdt:` spelling exists. **This is the system a caller of `cl-gbdt:train` loads** |
| `cl-gbdt/xgboost` | Layer 1 for XGBoost, exactly as above: `src/xgboost/native.lisp` and `src/xgboost/c-api.lisp`, plus `src/xgboost/classes.lisp` and `src/xgboost/api.lisp` -- the latter holding the same six operations and, additionally, `slice-model`, an XGBoost-only operation that builds a booster handle -- published by `src/xgboost/all.lisp` |
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

**Layer 1 alone trains and predicts.** A program that loads `cl-gbdt/lightgbm` or
`cl-gbdt/xgboost` and nothing else builds a dataset, builds a booster over it, advances it
one iteration at a time, scores with it and frees both -- `create-dataset`,
`create-booster`, `update-one-iteration`, `predict`, `free-dataset`, `free-booster`, with a
worked example under [Two systems per backend](#two-systems-per-backend) above. What such a
program still cannot do is `save-model`, `load-model`, `model-to-string`,
`feature-importance`, `evaluation`, `dataset-num-rows` or `dataset-num-features`, and it has
no training report, no early stopping and no `:objective` or `:evaluation` callback -- the
last three being `cl-gbdt:train`'s own concepts rather than operations with a Layer 1
counterpart. Bringing the first list down is the remaining follow-up work, tracked as the
first bullet of `docs/cl-gbdt-layered-api-implementation-policy.md`'s フォローアップ section.

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
- The same workflow's `version-matrix` job (task 4) reruns layer 2 -- only layer 2, since
  layer 1 needs no library and gains nothing from repetition -- against the endpoints of the
  recorded compatible-version range: LightGBM 4.0.0, XGBoost 2.0.0, and XGBoost 1.7.0. The
  1.7.0 leg is `continue-on-error: true` and expected to stay red -- it is the version-matrix
  table's failing row, kept running rather than deleted once it stopped supporting the wider
  claim (see [Usage](#usage)'s table above): a red job that keeps confirming a measured
  incompatibility is worth more than a green matrix with the contradicting case quietly
  removed, and `continue-on-error` keeps it from blocking merges while it does that. The
  pinned versions (4.7.0/3.3.0) are already covered by the job above, so this job does
  not repeat them. **One platform only, Linux x86_64** -- the three-platform matrix above
  exists to catch platform-specific bugs (byte order, calling convention, `.dylib` vs `.so`
  discovery) in bindings generated once and committed; a library *version* difference is a
  property of the upstream C source, identical across every platform cl-gbdt runs on, so
  crossing this axis with all three platforms would have tripled the job's cost for no new
  information. Crossing the two axes fully was also rejected for the same reason -- a
  LightGBM version and an XGBoost version load two independent shared libraries with no
  interaction between them, so each axis's own endpoints are varied one at a time against the
  other's pinned version, not against each other's endpoints too.
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
ros run -- --non-interactive --load tools/ci/check-layer-separation.lisp
ros run -- --non-interactive --load tools/ci/check-float-traps.lisp
ros run -- --non-interactive --load tools/ci/check-abi-blacklist.lisp
```

The last three are source scans, not loads: `check-layer-separation.lisp` proves no Layer 1
system reaches the unified API (see [Systems](#systems)), `check-float-traps.lisp` proves
every backend `defmethod` and every publicly exported backend `defun` wraps its body in
`with-foreign-float-traps-masked`, and `check-abi-blacklist.lisp` proves no backend imports a
C entry point `ffi-spec/ABI-BLACKLIST.md` rules out, that every import a backend does make is
declared in its `*required-symbols*` or `*optional-symbols*`, and that every capability
either list declares is registered in `*known-capabilities*`.

Two things the test scripts do that the plain commands above do not, and that CI needs:

- **They exit non-zero when a test fails.** `asdf:test-system` exits 0 regardless, so a job
  invoking it directly would be permanently green. `rove:run` returns false on failure and
  the script turns that into a status.
- **They check which foreign libraries were opened.** Layer 1 must open none; layer 2 must
  open both. rove counts a skip as pending rather than as a failure, so a functional run in
  which every library-dependent test skipped for want of `vendor/` still exits green and still
  reports each test file as completed — the summary reads the same as one where both libraries
  really were called. The difference has to be asserted rather than inferred.

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
