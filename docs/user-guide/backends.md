# Backends

How each backend splits into two systems, what its own package publishes, how its shared library is found, how `:parameters` is rendered, and how to ask it what it supports.

## Two systems per backend

`cl-gbdt.asd` declares **two** systems for each backend, and the difference between them is
the first thing to get right:

| System | What it carries |
|---|---|
| `cl-gbdt/lightgbm` | **Layer 1 alone.** Opens and closes the LightGBM shared library, and publishes LightGBM's own API: the fourteen finished operations `create-dataset`, `create-booster`, `update-one-iteration`, `predict`, `free-dataset`, `free-booster`, `save-model`, `load-model`, `model-to-string`, `feature-importance`, `evaluation`, `dataset-num-rows`, `dataset-num-features` and `create-dataset-from-file` -- a whole training run, the inference after it, persisting and reloading the model, reporting on it, and reading a dataset straight from a file -- plus `booster-eval`, `booster-eval-names` and the `lightgbm-backend` class, plus the shared basis a standalone caller needs: `open-backend`, `close-backend`, `backend-supports-p` and its siblings, `make-csr-matrix` and the `csr-matrix` readers, `handle-released-p`, `booster-training-set`, `booster-validation-sets`, and the whole condition hierarchy. **`cl-gbdt:train` and the other twelve portable generic functions are not part of it**, and loading it does not define the `cl-gbdt` package at all |
| `cl-gbdt/lightgbm/unified` | That, plus `src/lightgbm/protocol.lisp` -- LightGBM's methods on all thirteen portable generics -- plus core `cl-gbdt` itself, which it depends on. This is what the [quick start](../../README.markdown#quick-start) loads, and what every example in these guides that calls `cl-gbdt:train` loads |

`cl-gbdt/xgboost` and `cl-gbdt/xgboost/unified` divide identically -- the same fourteen
operations under the same names, different symbols in a different package -- and XGBoost's
Layer 1 API additionally publishes `slice-model`, `evaluate-one-iteration` and
`booster-boosted-rounds`. Loading one backend never loads the other, and no backend system
is a dependency of core `cl-gbdt`; load whichever matches the shared library you have.
**Loading both backends' `/unified` systems is how one program drives both libraries through
one portable API** -- that is what the two-backend examples in [Backend
differences](backend-differences.md) and [Training](training.md) do.

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

Opening and closing a backend from more than one thread has its own rules, distinct from
everything else on this page -- see [Thread safety](threads.md) for what is and is not safe to
do with a `backend` once more than one thread can reach it.

**A Layer 1 system alone trains, predicts, persists and reports.** Fourteen operations are
common to both backends -- XGBoost adds `slice-model`, which LightGBM has no counterpart for
-- and they are reachable with no unified API in the image at all: six of them
carry a whole run -- `create-dataset`, `create-booster`, `update-one-iteration`, `predict`,
`free-dataset` and `free-booster`, the ones exercised by both blocks below -- seven more
persist the trained model and answer questions about it -- `save-model`, `load-model`,
`model-to-string`, `feature-importance`, `evaluation`, `dataset-num-rows` and
`dataset-num-features`, exercised by `tests/functional/{lightgbm,xgboost}-standalone.lisp`
rather than by a worked example here -- and one more, `create-dataset-from-file`, builds a
dataset straight from a file instead of the matrix the six above use (see [File
input](file-input.md#file-input)). Both blocks below run in one fresh image that never
defines the `cl-gbdt` package:

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
differ](backend-differences.md#where-the-two-backends-genuinely-differ)).
`tests/functional/lightgbm-standalone.lisp`
and `tests/functional/xgboost-standalone.lisp` are the same run as a test, each naming its
backend's public package and no other system *of this project* -- `rove` aside, they declare
nothing -- so that the claim is enforced by the build rather than asserted here.

**What a Layer 1 caller still cannot do** is `cl-gbdt:train`'s own concepts: the training
report, early stopping, and `train`'s `:objective` and `:evaluation` callbacks. They stay
where they are, built *around* the loop above rather than being operations inside it, so
there is nothing at Layer 1 for them to be counterparts of. Everything else the thirteen
generics carry now has a Layer 1 counterpart, but the two backends' lambda lists are not
identical for the seven newest ones:

- LightGBM's `save-model`, `model-to-string` and `feature-importance` take `:num-iteration`;
  XGBoost's take none, its C API having no iteration limit on any of the three. The
  `unsupported-argument` refusal a unified-API caller sees for that keyword on XGBoost is
  signalled by the Layer 2 method, and exists only because LightGBM honours the argument.
- `:num-iteration :best` is a Layer 2 concept -- `booster-best-iteration` is written by
  `train` and nothing else -- and is refused at Layer 1 wherever the keyword exists at all.

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

## Backend-specific packages

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
94 external symbols, 53 of them the condition hierarchy
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
 CL-GBDT/SRC/LIGHTGBM/API:CREATE-DATASET
 CL-GBDT/SRC/LIGHTGBM/API:CREATE-DATASET-FROM-FILE CL-GBDT/SRC/DATA:CSR-MATRIX
 CL-GBDT/SRC/DATA:CSR-MATRIX-IMPLICIT-VALUE CL-GBDT/SRC/DATA:CSR-MATRIX-INDICES
 CL-GBDT/SRC/DATA:CSR-MATRIX-INDPTR CL-GBDT/SRC/DATA:CSR-MATRIX-NUM-COLUMNS
 CL-GBDT/SRC/DATA:CSR-MATRIX-NUM-ROWS CL-GBDT/SRC/DATA:CSR-MATRIX-VALUES
 CL-GBDT/SRC/HANDLE:DATASET CL-GBDT/SRC/LIGHTGBM/API:DATASET-NUM-FEATURES
 CL-GBDT/SRC/LIGHTGBM/API:DATASET-NUM-ROWS CL-GBDT/SRC/LIGHTGBM/API:EVALUATION
 CL-GBDT/SRC/LIGHTGBM/API:FEATURE-IMPORTANCE
 CL-GBDT/SRC/LIGHTGBM/API:FREE-BOOSTER CL-GBDT/SRC/LIGHTGBM/API:FREE-DATASET
 CL-GBDT/SRC/HANDLE:HANDLE-BACKEND CL-GBDT/SRC/HANDLE:HANDLE-RELEASED-P
 CL-GBDT/SRC/LIGHTGBM/CLASSES:LIGHTGBM-BACKEND
 CL-GBDT/SRC/LIGHTGBM/API:LOAD-MODEL CL-GBDT/SRC/DATA:MAKE-CSR-MATRIX
 CL-GBDT/SRC/LIGHTGBM/API:MODEL-TO-STRING CL-GBDT/SRC/BACKEND:OPEN-BACKEND
 CL-GBDT/SRC/LIGHTGBM/API:PREDICT CL-GBDT/SRC/LIGHTGBM/API:SAVE-MODEL
 CL-GBDT/SRC/LIGHTGBM/API:UPDATE-ONE-ITERATION)
```

Those 94 fall into three groups. **LightGBM's own API** is seventeen of them: the fourteen
finished operations enumerated above -- `create-dataset`, `create-booster`,
`update-one-iteration`, `predict`, `free-dataset`, `free-booster`, `save-model`,
`load-model`, `model-to-string`, `feature-importance`, `evaluation`, `dataset-num-rows`,
`dataset-num-features` and `create-dataset-from-file`, all homed in
`cl-gbdt/src/lightgbm/api` -- plus that backend's own
evaluation entry points, `booster-eval` and `booster-eval-names`, and the `lightgbm-backend`
class, useful for `typep` or for specializing your own methods on one specific backend
rather than the shared `backend` (`open-backend` itself never needs it, since it looks
classes up by the `:lightgbm`/`:xgboost` keyword internally, not by this symbol). Eleven of
those seventeen names -- `predict`, `update-one-iteration`, `free-dataset`, `free-booster`,
`save-model`, `load-model`, `model-to-string`, `feature-importance`, `evaluation`,
`dataset-num-rows` and `dataset-num-features` -- are *also* names `cl-gbdt` exports, and
these are **different symbols**: plain functions here, generic functions there, so an image
holding both packages has to say which it means. **The shared basis** is the other
twenty-four non-condition symbols: `open-backend`, `close-backend`,
`backend-supports-p` and the rest of the backend readers, `handle-released-p`,
`handle-backend`, `booster-training-set` and `booster-validation-sets`, the
`dataset`/`booster` handle classes, and `make-csr-matrix` with the `csr-matrix` type and its
six readers, so that the sparse half of `create-dataset` and `predict` is reachable from the
package that publishes them. These are republished here, rather than left to core `cl-gbdt`,
so that a program loading this Layer 1 system alone can open, question and close a backend
without naming an internal package -- and unlike the eleven doubled operation names, they are
the very symbols `cl-gbdt` exports, one symbol reached two ways with nothing to
disambiguate. And **the condition hierarchy** is the remaining 53, re-exported whole from
`cl-gbdt/src/conditions`: every type and accessor there is already reviewed public API, so a
Layer 1 caller catches `foreign-call-error` or `backend-library-not-found` by the same name
a unified-API caller does.

`cl-gbdt/xgboost` is the same shape -- 95 external symbols, the same 53 conditions and the
same 24 shared-basis symbols -- with eighteen of its own rather than seventeen: the same
fourteen operations under its own package's symbols, plus `xgboost-backend`,
`slice-model`, `booster-boosted-rounds` (see [the capability
section](#asking-a-backend-what-it-can-do)), and an
`evaluate-one-iteration` of its own, which takes different arguments and returns something
different (see
[the differences table](backend-differences.md#where-the-two-backends-genuinely-differ)): the two
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
The fourteen operations above are what that incremental process has produced so far, and they
come from a third file, `src/<backend>/api.lisp`, rather than from `native.lisp` at all:
`native.lisp` holds the `%`-functions that take and return raw pointers, `api.lisp` the
finished operations built on top of them that take a backend or a handle and hand back a
handle or a result.

## Finding the shared library

Every `open-backend` needs a matching `close-backend`.
[`with-backend`](threads.md#with-backend) pairs them for you the way `with-dataset` and
`with-booster` pair theirs, and goes outside both, since `close-backend` unloads the library
their pointers are backed by. All three are **unified-API macros**, published by `cl-gbdt` and
by neither backend package -- the symbol listing above is the evidence -- so a program on a
bare Layer 1 system closes and frees by hand because that is the only thing available to it,
as the [Two systems per backend](#two-systems-per-backend) examples do.

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

## Parameters

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

## Asking a backend what it can do

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
what was asked as well as what was answered. Nine of the ten registered capabilities answer
true somewhere today: `:model-slicing`, on XGBoost only -- see the model-slicing row in
[the differences table](backend-differences.md#where-the-two-backends-genuinely-differ) --
plus `:evaluation-history` and `:early-stopping` on both backends, since `train` records
a history and takes `:early-stopping` (see [Training report](training.md#training-report)),
`:sparse-input` on both, since both libraries export the CSR entry points it names (see [Sparse
input](data-and-prediction.md#sparse-input-csr-matrices)), `:missing-value` on XGBoost only
(see [Missing values](data-and-prediction.md#missing-values)), `:categorical-features`
on both (see [Categorical features](data-and-prediction.md#categorical-features)),
`:prediction-shape` on both, since `predict` states a shape for the result it just predicted
on both libraries (see [Prediction shape](data-and-prediction.md#prediction-shape)),
`:custom-objective` on both, since `train` boosts against a caller's own gradient and
Hessian on both libraries (see [Custom objective](custom-training.md#custom-objective)), and
`:custom-evaluation` on both, since `train` records a caller-written metric per dataset on
both libraries (see [Custom evaluation](custom-training.md#custom-evaluation)). The tenth,
`:multidimensional-feature-score`, is registered and false everywhere, which says "not
supported yet" rather than "never heard of it".

`:evaluation-history` is true unconditionally rather than probed. The C functions behind it
are in each backend's `*required-symbols*`, so a library missing them never opens at all
and there is no state in which an open backend cannot record a history; each backend names
the capability in its own `*provided-capabilities*`, which `open-backend` records as true
without a symbol lookup. A probe can only answer from a symbol that might be absent, which
is the right shape for `:model-slicing` and the wrong one for a feature that is simply
always there.
