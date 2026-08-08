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
LightGBM and XGBoost shared libraries, exercised by 304 functional assertions (design doc
section 12, layer 2), in addition to 335 assertions that need no shared library at all
(layer 1). `train` also returns a `training-report` as its secondary value, and takes
`:early-stopping` to end a run once a watched metric stops improving -- see
[Training report](#training-report) below. See [Usage](#usage) below for a worked example.

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

### Backend-specific packages

`cl-gbdt/lightgbm` and `cl-gbdt/xgboost` name a package now, not only the system in the
heading above. `docs/cl-gbdt-layered-api-implementation-policy.md` section 3 calls these
the public packages for backend-specific API -- distinct from each backend's internal
`cl-gbdt/src/lightgbm/all`/`cl-gbdt/src/xgboost/all` aggregation, and, always, from the raw
generated CFFI bindings (`cl-gbdt/src/lightgbm/c-api`, `cl-gbdt/src/xgboost/c-api`), which
neither package re-exports (policy sections 3 and 11):

```lisp
(ql:quickload :cl-gbdt/lightgbm :silent t)
(format t "~S~%"
        (sort (loop :for symbol :being :the :external-symbols :of "CL-GBDT/LIGHTGBM"
                    :collect symbol)
              #'string< :key #'symbol-name))
```

Output:

```
(CL-GBDT/SRC/LIGHTGBM/NATIVE:BOOSTER-EVAL
 CL-GBDT/SRC/LIGHTGBM/NATIVE:BOOSTER-EVAL-NAMES
 CL-GBDT/SRC/LIGHTGBM/PROTOCOL:LIGHTGBM-BACKEND)
```

Today that is `cl-gbdt/lightgbm`'s entire published surface: the backend's own CLOS class
-- useful for `typep` or for specializing your own methods on one specific backend rather
than the shared `backend`; `open-backend` itself never needs it, since it looks classes up
by the `:lightgbm`/`:xgboost` keyword internally, not by this symbol -- plus that backend's
own evaluation entry points. `cl-gbdt/xgboost` publishes `xgboost-backend`, `slice-model`
and `booster-boosted-rounds` (see [the capability
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
XGBoost's `slice-model` is the most recent one added that way.

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
what was asked as well as what was answered. Two capabilities answer true today:
`:model-slicing`, on XGBoost only -- see the model-slicing row in the table below -- and
`:evaluation-history`, on both backends, since `train` records one (see
[Training report](#training-report)).

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
| `save-model`'s `:num-iteration` | Limits how many trees are saved | Signals `unsupported-argument` -- `XGBoosterSaveModel` always saves every round |
| `model-to-string`'s `:num-iteration` | Limits the rounds serialized | Signals `unsupported-argument` -- no iteration-limited variant exists |
| `feature-importance`'s `:num-iteration` | Limits the importance calculation | Signals `unsupported-argument` -- no iteration-limited variant exists |
| `feature-importance`'s result shape | Always one number per feature | Signals `unsupported-argument` instead of returning a result when the model reports a multi-dimensional score shape -- a `gblinear` booster's importance on a multi-class model, whose scores are a per-class matrix with no single-value reduction this backend will invent |
| What `evaluation` evaluates | The datasets `train` attached, read back by index (`LGBM_BoosterGetEval`): the library computed these metrics during training and this reads them out | The booster's own retained training set and `:valid-sets` entries, which this backend hands to `XGBoosterEvalOneIter` explicitly -- that call evaluates whatever DMatrices it is given and consults nothing the booster was built with, so passing the retained ones is what makes the index mean the same thing on both backends |
| `evaluation`'s values | `LGBM_BoosterGetEval`'s own doubles, returned unmodified -- the secondary value says `:value-source :library-doubles` | Parsed out of the single formatted line `XGBoosterEvalOneIter` produces -- `:value-source :parsed-text`, with that line itself kept verbatim under `:raw`, and a value XGBoost spelled `inf`/`nan` coming back as `nil` rather than a number. The same line is `cl-gbdt/xgboost:evaluate-one-iteration`'s own primary value at Layer 1, for a caller who wants it without going through the portable API |
| Model slicing | No counterpart at all: LightGBM's C API has nothing that extracts a range of boosting rounds into a new model, so `(backend-supports-p backend :model-slicing)` is `nil` and there is no LightGBM function to call | `cl-gbdt/xgboost:slice-model` (Layer 1, XGBoost-only), over `XGBoosterSlice`. Returns a new booster holding a half-open `[begin, end)` range of the parent's layers, independent of it -- freeing the parent leaves the slice usable. Deliberately not part of the unified API: with no LightGBM counterpart a portable version could only signal for every caller of one backend, or emulate, and emulating is what [the capability model](#asking-a-backend-what-it-can-do) exists to rule out |
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

And `evaluation`, whose two rows above are about how each backend produces the numbers
rather than about one backend refusing something. The same call, the same 8 rows, the same
one `:valid-sets` entry, on both backends -- dataset 0 is the training set, dataset 1 is
that validation set, and the metric names are each library's own:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm :cl-gbdt/xgboost) :silent t)

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
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm :cl-gbdt/xgboost) :silent t)

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
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm :cl-gbdt/xgboost) :silent t)

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

## Systems

| System | Purpose |
|---|---|
| `cl-gbdt` | Core: package, condition hierarchy, matrix marshalling, backend registry and `open-backend` protocol, the unified API's generic functions -- no methods, and no shared library required to load it |
| `cl-gbdt/lightgbm` | The LightGBM backend: all 13 unified-API methods (`src/lightgbm/protocol.lisp`), built on the Layer 1 wrappers in `src/lightgbm/native.lisp` over the generated CFFI bindings for the LightGBM C API (`src/lightgbm/c-api.lisp`), and published together by `src/lightgbm/all.lisp` |
| `cl-gbdt/xgboost` | The XGBoost backend: all 13 unified-API methods (`src/xgboost/protocol.lisp`), built on the Layer 1 wrappers in `src/xgboost/native.lisp` over the generated CFFI bindings for the XGBoost C API (`src/xgboost/c-api.lisp`), and published together by `src/xgboost/all.lisp` |
| `cl-gbdt/regen` | The binding emitter (`src/regen/`). Development-only -- never appears in `cl-gbdt`'s, `cl-gbdt/lightgbm`'s, or `cl-gbdt/xgboost`'s dependency graph, so an ordinary user never needs it or its dependencies (`alexandria`, `com.inuoe.jzon`). `cffi/c2ffi` is *not* one of them -- it is a dependency of `tools/regen.lisp`, which quickloads it directly, not of the `cl-gbdt/regen` system itself |
| `cl-gbdt/tests` | The Rove test suite |

Each backend system (`cl-gbdt/lightgbm` depends on `cl-gbdt/src/lightgbm/all`,
`cl-gbdt/xgboost` on `cl-gbdt/src/xgboost/all`) implements all 13 methods of the
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
