;;;; native.lisp --- XGBoost backend, Layer 1: library discovery, the error wrapper, and
;;;; every internal function that turns a raw call into libxgboost.so into something
;;;; safe to call from Lisp -- out parameters into return values, error codes into
;;;; conditions, raw pointers accepted only where a caller already validated them.
;;;;
;;;; Nothing here is a CLOS protocol method and nothing here depends on
;;;; `cl-gbdt/src/xgboost/protocol' -- see that file for the thirteen unified-API
;;;; `defmethod's this module exists to support, and `cl-gbdt/src/xgboost/classes' for the
;;;; two more, `initialize-backend' and `shutdown-backend', that this file's library-discovery
;;;; parameters serve. `cl-gbdt/src/xgboost/protocol''s own docstrings, not this
;;;; file's, are the place each function's role in the unified API is explained; the
;;;; docstrings below describe only what each function does to the C API.

(uiop:define-package #:cl-gbdt/src/xgboost/native
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/xgboost/c-api
                #:xgb-get-last-error
                #:xg-boost-version
                #:xgd-matrix-create-from-dense
                #:xgd-matrix-create-from-csr
                #:xgd-matrix-create-from-uri
                #:xgd-matrix-set-info-from-interface
                #:xgd-matrix-set-u-int-info
                #:xgd-matrix-set-str-feature-info
                #:xgd-matrix-free
                #:xgd-matrix-num-row
                #:xgd-matrix-num-col
                #:xg-booster-create
                #:xg-booster-free
                #:xg-booster-slice
                #:xg-booster-set-param
                #:xg-booster-get-num-feature
                #:xg-booster-boosted-rounds
                #:xg-booster-update-one-iter
                #:xg-booster-train-one-iter
                #:xg-booster-eval-one-iter
                #:xg-booster-predict-from-d-matrix
                #:xg-booster-predict-from-csr
                #:xg-booster-save-model
                #:xg-booster-load-model
                #:xg-booster-save-model-to-buffer
                #:xg-booster-feature-score)
  (:import-from #:cl-gbdt/src/xgboost/array-interface
                #:array-interface-json)
  (:import-from #:cl-gbdt/src/backend
                #:backend-name
                #:backend-open-p)
  (:import-from #:cl-gbdt/src/handle
                #:%check-handle-kind
                #:handle-live-pointer
                #:handle-released-p
                #:dataset
                #:booster
                #:booster-training-set
                #:booster-validation-sets)
  (:import-from #:cl-gbdt/src/conditions
                #:backend-not-open
                #:foreign-call-error
                #:released-handle-error
                #:wrong-backend-reference
                #:unsupported-argument
                #:dimension-mismatch)
  (:import-from #:cl-gbdt/src/parameters
                #:normalize-parameters)
  (:import-from #:cl-gbdt/src/config/feature-names
                #:check-feature-names)
  (:import-from #:cl-gbdt/src/config/missing-value
                #:missing-value-json)
  (:import-from #:cl-gbdt/src/config/objective
                #:objective-single-float)
  (:import-from #:cl-gbdt/src/data
                #:with-foreign-matrix
                #:write-foreign-sequence)
  (:import-from #:cl-gbdt/src/foreign
                #:check-foreign-call
                #:with-foreign-float-traps-masked)
  (:export #:check-xgb
           #:*library-env-var*
           #:*vendor-library-directory*
           #:*vendor-library-pattern*
           #:*default-library-name*
           #:*required-symbols*
           #:*optional-symbols*
           #:*provided-capabilities*
           #:%check-backend-open
           #:%check-xgboost-dataset
           #:%check-unsupported
           #:%read-version
           #:%create-dmatrix
           #:%create-dmatrix-from-uri
           #:%create-dmatrix-from-csr
           #:%set-info-field
           #:%set-group-field
           #:%set-feature-names
           #:%set-feature-types
           #:%free-dmatrix
           #:%free-dmatrix-unchecked
           #:%dataset-num-rows
           #:%dataset-num-features
           #:%create-booster
           #:%set-parameters
           #:%boosted-rounds
           #:%update-one-iteration
           #:%check-booster-datasets-live
           #:%free-booster
           #:%free-booster-unchecked
           #:%predict-type
           #:%resolve-num-iteration
           #:%predict-config-json
           #:%total-element-count
           #:%reported-shape
           #:%predict-ncol
           #:%predict-from-dmatrix
           #:%predict-from-csr
           #:%booster-predictions
           #:%train-one-iteration-custom
           #:%save-model
           #:%load-model
           #:%save-model-to-buffer
           #:%feature-importance-type
           #:%booster-num-features
           #:%feature-score-index
           #:%check-feature-score-dim
           #:%feature-score
           #:%split-eval-label
           #:%check-xgboost-booster
           #:evaluate-one-iteration
           #:%read-evaluation
           #:booster-boosted-rounds
           #:%slice))

(in-package #:cl-gbdt/src/xgboost/native)

;;; ---------------------------------------------------------------------------
;;; Floating-point trap safety
;;;
;;; Almost nothing here wraps itself in `with-foreign-float-traps-masked' -- almost every
;;; function in this file is only ever called from inside an already-masked body: a
;;; `cl-gbdt/src/xgboost/protocol' `defmethod' for a caller that went through the unified API,
;;; or a `cl-gbdt/src/xgboost/api' `defun' for a caller that reached this backend's Layer 1
;;; directly -- either establishes the mask, at operation-body granularity, before making any
;;; of the calls below. See that file's identical commentary, and
;;; `with-foreign-float-traps-masked''s own docstring in `cl-gbdt/src/foreign', for why:
;;; SBCL enables floating-point traps by default on x86-64 and not on aarch64, and
;;; XGBoost's own numeric code -- confirmed for the softmax normalization behind a
;;; `multi:softprob' prediction -- was written and tested against the C convention of
;;; those traps staying masked.
;;;
;;; The two public entry points near the end of this file are the exception:
;;; `evaluate-one-iteration' in the Evaluation section and `booster-boosted-rounds' in the
;;; Model slicing section after it. Both are exported directly from `cl-gbdt/xgboost' (policy
;;; section 3's Layer 1) -- see `src/xgboost/all.lisp' -- so neither is ever reached through a
;;; `cl-gbdt/src/xgboost/protocol' `defmethod'. There is no outer call site left to establish
;;; the mask for them, so each wraps its own whole body instead, the same method-body
;;; granularity `protocol.lisp' uses, just with the wrapping function living here rather than
;;; there -- mirrors `cl-gbdt/src/lightgbm/native''s identical
;;; `booster-eval'/`booster-eval-names' exception.
;;; `tools/ci/check-float-traps.lisp' checks this directly: it reads the sibling `all.lisp''s
;;; public `:export' clause and requires every `defun' named there to open with this macro,
;;; the same rule it already applied to every `defmethod' in `protocol.lisp'. The other public
;;; entry points this backend has -- `slice-model' and the thirteen finished operations
;;; `create-dataset', `create-booster', `update-one-iteration', `predict', `free-dataset',
;;; `free-booster', `save-model', `load-model', `model-to-string', `feature-importance',
;;; `evaluation', `dataset-num-rows' and `dataset-num-features' -- are covered by the identical
;;; rule but live in `api.lisp' rather than here, each wrapping its own whole body there; see
;;; the Model slicing section below for why `slice-model' in particular could never live in
;;; this file. Thirteen plus `slice-model' is fourteen, which is what `check-float-traps'
;;; prints for that file.

;;; ---------------------------------------------------------------------------
;;; Error checking

(defun %last-error-message ()
  "Return XGBoost's last error message as a Lisp string, or NIL when
`XGBGetLastError' returns a null pointer or an empty string.

An empty string is not a null pointer, but it means the same thing here: no message
is available, most often because nothing in this process has failed yet, or because the
call that just failed -- `XGBoosterSlice''s out-of-bounds path is the known case, see
`%slice' -- never wrote one. Treating it as NIL lets `foreign-call-error''s own `(or
message \"(no message)\")' fallback fire instead of printing a bare colon. This does NOT
by itself fix a *stale*, non-empty message left over from an earlier, unrelated failure
in the same process -- `%slice' guards against that separately, by never routing the
out-of-bounds code through this function at all."
  (let ((pointer (xgb-get-last-error)))
    (unless (cffi:null-pointer-p pointer)
      (let ((message (cffi:foreign-string-to-lisp pointer)))
        (unless (zerop (length message))
          message)))))

(defun check-xgb (code function-name)
  "Signal `foreign-call-error' when CODE reports failure, otherwise return CODE.

XGBoost returns 0 on success and a nonzero status -- documented as -1 -- on failure,
with the detail available from `XGBGetLastError'. FUNCTION-NAME identifies which C
function reported CODE, for the condition's report.

Thin wrapper around `check-foreign-call' supplying `%last-error-message' as XGBoost's
last-error thunk, matching `cl-gbdt/src/lightgbm/native''s `check-lgbm' -- see that
function's docstring for why it stays a named wrapper instead of calling
`check-foreign-call' directly at each call site."
  (check-foreign-call code function-name #'%last-error-message))

(defun %check-backend-open (backend)
  "Signal `backend-not-open' when BACKEND is not open.

`make-dataset', `train' and `load-model' each create a brand-new handle directly from
BACKEND -- there is no existing handle for the check to route through the way
`handle-live-pointer' does for every other operation in this file, since none exists yet.
Each of them calls this first, before touching any foreign function, so a backend a
caller has closed (or never opened) is never reached by `XGDMatrixCreateFromDense',
`XGBoosterCreate' or `XGBoosterLoadModel' with a library that may no longer be mapped.

`cl-gbdt/src/xgboost/api''s `create-dataset' and `create-booster' call it as well, for the
same reason and ahead of the first two of those three C functions: they are Layer 1 entry
points a caller reaches without going through `make-dataset' or `train' at all, so neither
may inherit the check its Layer 2 counterpart makes."
  (unless (backend-open-p backend)
    (error 'backend-not-open :backend (backend-name backend))))

(defun %check-xgboost-dataset (backend dataset argument-description dataset-class)
  "Return DATASET's live foreign pointer, after confirming DATASET is of type
DATASET-CLASS -- `cl-gbdt/src/xgboost/classes''s `xgboost-dataset'. ARGUMENT-DESCRIPTION
names which caller-supplied argument DATASET came from -- e.g. \"train's dataset
argument\" -- for `wrong-backend-reference''s report.

DATASET-CLASS is a parameter, not a symbol named directly in this file, because this
file cannot depend on `cl-gbdt/src/xgboost/classes' -- that package reads this one's
`*required-symbols*' and library-discovery parameters, so the edge already runs the other
way and naming it here would close a cycle. The two callers,
`cl-gbdt/src/xgboost/protocol''s `train' and `cl-gbdt/src/xgboost/api''s `create-booster',
pass `'xgboost-dataset' at each of their call sites instead.

Every caller-supplied dataset argument -- `train''s DATASET and each entry of its
:VALID-SETS, and the same two arguments of `create-booster' -- must pass through here
before reaching a foreign call that expects a `DMatrixHandle'. `handle-live-pointer' alone
is not enough: it only guards against a released handle or a closed backend, and happily
returns *any* handle's pointer regardless of kind, including a booster's. Nothing in
`cl-gbdt/src/xgboost/api' dispatches on a handle any more -- `dataset-num-rows' and
`free-dataset' were `defmethod's until the Layer 1 split and are plain `defun's now -- so
every operation there makes its kind check explicitly, this function or
`%check-object-class' depending on whether it needs the handle to be live. A booster's own
pointer reaching `XGBoosterCreate''s DMatrix array is exactly the corruption this check
exists to prevent -- the identical hazard killed the process across several threads on the
LightGBM branch.

Signals `wrong-backend-reference' when DATASET is not of type DATASET-CLASS -- built by a
different backend, or not a dataset at all -- and whatever `handle-live-pointer' signals
otherwise: `released-handle-error' for an already-freed DATASET, `backend-not-open' when
DATASET's own backend has since been closed.

This does not additionally check that DATASET was built by BACKEND specifically, only
that it is of type DATASET-CLASS -- see `cl-gbdt/src/lightgbm/native''s
`%check-lightgbm-dataset', which this mirrors, for why: two backend instances over the
same shared library are a legitimate way for a caller to hold datasets from."
  (unless (typep dataset dataset-class)
    (error 'wrong-backend-reference
           :backend (backend-name backend)
           :given (class-name (class-of dataset))
           :argument argument-description))
  (handle-live-pointer dataset))

(defun %check-unsupported (backend argument value reason)
  "Signal `unsupported-argument' when VALUE is non-nil, naming ARGUMENT and REASON.

`make-dataset' calls this for keywords the protocol declares but this backend has no way
to honor. Left silently ignored, a caller moving a working call from LightGBM to XGBoost
would get a dataset that looks fine but was not built the way the caller asked -- the same
failure mode `wrong-backend-reference' exists to prevent for handle arguments, but here
for an argument that is not a handle at all."
  (when value
    (error 'unsupported-argument
           :backend (backend-name backend)
           :argument argument
           :reason reason)))

;;; ---------------------------------------------------------------------------
;;; Library discovery

(defparameter *library-env-var* "CL_GBDT_XGBOOST_LIB"
  "Environment variable overriding XGBoost's shared-library discovery.")

(defparameter *vendor-library-directory* "vendor/xgboost/lib/"
  "Repository-relative directory `tools/fetch-libs.sh' writes XGBoost's shared library
to. Not `vendor/xgboost.libs/' -- that directory holds XGBoost's own bundled dependencies,
and `libxgboost.so''s RPATH points at it; flattening the two together has broken this
repository once already.")

(defparameter *vendor-library-pattern* "libxgboost.*"
  "Basename pattern for XGBoost's shared library within *vendor-library-directory*. The
extension stays wild because `tools/fetch-libs.sh' preserves whatever the platform's wheel
ships -- `.so' on Linux, `.dylib' on macOS -- and discovery must not assume either.")

(defparameter *default-library-name* "libxgboost"
  "Name passed to CFFI's own system library search when no other candidate is found.

CFFI's `:default' designator only appends the platform's shared-library suffix (`.so' on
Linux, per `cffi::default-library-suffix') to this string -- it never adds a `lib' prefix,
confirmed against CFFI's own `load-foreign-library-helper' and empirically on this host:
with the vendored directory pushed onto `cffi:*foreign-library-directories*',
`(:default \"xgboost\")' fails to resolve `libxgboost.so' while `(:default \"libxgboost\")'
succeeds. So this constant has to carry the library's full on-disk basename, `lib'
included -- unlike LightGBM's, whose compiled basename genuinely omits it; see
`cl-gbdt/src/lightgbm/native''s identical constant.")

(defparameter *required-symbols*
  '("XGBoostVersion"
    "XGBGetLastError"
    "XGDMatrixCreateFromDense"
    "XGDMatrixCreateFromURI"
    "XGDMatrixSetInfoFromInterface"
    "XGDMatrixSetUIntInfo"
    "XGDMatrixSetStrFeatureInfo"
    "XGDMatrixFree"
    "XGDMatrixNumRow"
    "XGDMatrixNumCol"
    "XGBoosterCreate"
    "XGBoosterFree"
    "XGBoosterSetParam"
    "XGBoosterGetNumFeature"
    "XGBoosterBoostedRounds"
    "XGBoosterUpdateOneIter"
    "XGBoosterEvalOneIter"
    "XGBoosterPredictFromDMatrix"
    "XGBoosterSaveModel"
    "XGBoosterLoadModel"
    "XGBoosterSaveModelToBuffer"
    "XGBoosterFeatureScore")
  "C function names this backend calls, checked with `probe-foreign-symbols' right after
the library loads.")

(defparameter *optional-symbols*
  '((:model-slicing "XGBoosterSlice")
    (:sparse-input "XGDMatrixCreateFromCSR" "XGBoosterPredictFromCSR")
    (:custom-objective "XGBoosterTrainOneIter"))
  "Capability name to the C function names that capability needs.

Unlike `*required-symbols*', whose absence makes `open-backend' signal
`missing-foreign-symbols', a name missing from here disables one capability and nothing else
-- policy section 8. An XGBoost too old to have `XGBoosterSlice' is a working XGBoost that
cannot slice, not a broken installation.

`XGBoosterSlice' is bound in c-api.lisp and called only from `slice-model', which checks the
capability before reaching it.

`:sparse-input' names both the ingestion entry point and the prediction one, even though
`make-dataset' reaches only the first. The capability is one answer about whether this
backend can take a `csr-matrix' at all, and a caller told \"yes\" who could build a dataset
but not predict from one would have been told a half-truth. Listing both from the start means
the answer never changes meaning as the sparse path grows.

`:custom-objective' names ONE function where LightGBM's entry of the same name needs four:
`XGBoosterTrainOneIter' takes the caller's gradient and Hessian, and everything else one
iteration of `train''s custom loop calls is already required here. The scores the caller's
function is handed come from `XGBoosterPredictFromDMatrix' -- `%booster-predictions' below is
a `:raw' prediction over the training DMatrix, not a separate score-reading entry point the
way LightGBM's `LGBM_BoosterGetPredict' is -- the output-group count is not read from the library
at all but divided out of that prediction's own element count by `%predict-ncol', and the round
number the update is given comes from `XGBoosterBoostedRounds', through `%boosted-rounds', the
same way the built-in `%update-one-iteration' gets its own. Both of those C names are in
`*required-symbols*' above, so neither adds a name here.

It belongs HERE and not in `*provided-capabilities*' below: `XGBoosterTrainOneIter' is NOT in
`*required-symbols*' above, so an XGBoost too old to export it opens perfectly well and simply
cannot boost against a caller's own gradient -- which is exactly the state a probe exists to
detect. `probe-capabilities' records PROVIDED entries ahead of probed ones, so naming a
capability in both lists would make the probe's answer unreachable; its own docstring calls
that combination a contradiction in the backend's declarations, and
`tools/ci/check-abi-blacklist.lisp''s own header names the same arrangement among the things
it cannot catch -- a capability declared provided is recorded true on a library that lacks the
symbol, and nothing there would notice. This is the same shape as `:sparse-input' and
`:model-slicing' above, and the opposite of `:missing-value' and `:categorical-features'
below, whose entry points are required already. `cl-gbdt/src/lightgbm/native''s entry of this
same name is probed for the same reason.

`XGBoosterBoostOneIter', the other entry point that takes a gradient and a Hessian, is bound
in c-api.lisp and named nowhere here, because nothing calls it. The vendored header
(`ffi-spec/xgboost/include/xgboost/c_api.h') marks it `@deprecated since 2.1.0' and
`XGBoosterTrainOneIter' `@since 2.0.0'; it also takes bare `float*' pointers and a flat
length, where `XGBoosterTrainOneIter' takes two `__array_interface__' descriptors that can
state a (ROWS GROUPS) shape.")

(defparameter *provided-capabilities*
  '(:evaluation-history :early-stopping :missing-value :categorical-features
    :prediction-shape :custom-evaluation)
  "Capabilities this backend provides unconditionally, recorded true at `open-backend'
without being probed -- `probe-capabilities''s PROVIDED, which says why a probe cannot
express this.

`:evaluation-history' is here rather than in `*optional-symbols*' because the C function
`train' records a history with -- `XGBoosterEvalOneIter', reached through `%read-evaluation'
-- is in `*required-symbols*' above. A library missing it never opens at all, so there is no
state in which this backend is open and cannot record a history, and nothing for a probe to
answer differently from one open to the next.
`:early-stopping' is here for a different reason: it needs no C function at all. Both `train'
loops run in Lisp, so the stop decision is `cl-gbdt/src/training/early-stopping''s pure code,
which every open backend has by construction. A probe has nothing to look for, and the
capability cannot vary between one open and the next.
`:missing-value' is here for the first reason again, and more plainly than
`:evaluation-history': the sentinel is a KEY IN A CONFIG JSON -- `\"missing\"', read by
`XGDMatrixCreateFromDense', `XGDMatrixCreateFromCSR' and `XGBoosterPredictFromCSR' -- not
a C function of its own. Every entry point that reads it is already in `*required-symbols*'
or `*optional-symbols*' for another reason, so there is no symbol whose presence or absence
could make this capability true or false, and a probe has nothing left to decide. A library
that opens at all can be told which value means missing.
`:categorical-features' is here for that same first reason once more: the C function that
carries it, `XGDMatrixSetStrFeatureInfo', is already in `*required-symbols*' above -- it is
how FEATURE-NAMES has always been attached -- and marking a column categorical is the same
call under a different field name. A library missing it never opens at all, so a probe in
`*optional-symbols*' would have nothing left to decide.

`:prediction-shape' is here for that first reason a fourth time, and it needs no C function of
its own at all: the shape is the `out_shape'/`out_dim' pair the prediction entry points already
write on every call, read back by `%reported-shape' instead of only multiplied out by
`%total-element-count'. `XGBoosterPredictFromDMatrix' is in `*required-symbols*' above, so a
library that opens at all reports a shape for the dense path; `XGBoosterPredictFromCSR' is
OPTIONAL, under `:sparse-input', and a library lacking it still predicts densely -- which is
exactly why this capability can be declared unconditionally rather than probed, since no
library this backend will open can fail to report a shape. There is no symbol whose presence or
absence could make it true or false.

No operation re-checks it: nothing takes an argument asking for a shape, so a backend answering
false returns NIL as `predict''s second value rather than signalling. That is not a break with
a uniform rule -- of the five names beside it in this list, `:missing-value',
`:categorical-features' and `:custom-evaluation' ARE re-checked, by
`cl-gbdt/src/xgboost/protocol''s `%check-missing-value', `%check-categorical-features' and
`%check-custom-evaluation', while `:evaluation-history' and `:early-stopping' are re-checked
nowhere either. See `cl-gbdt/src/backend''s `*known-capabilities*', where the whole split is
stated.

`:custom-evaluation' is here for that first reason a fifth time: the C function `train' hands
a caller's own `:evaluation' metric one dataset's predictions with --
`XGBoosterPredictFromDMatrix', reached through `%booster-predictions' -- is in
`*required-symbols*' above, and `%predict-config-json' and `%predict-ncol' beside it are this
file's own pure code. A library missing that entry point never opens at all, so there is no
state in which this backend is open and cannot hand a caller's metric one dataset's
predictions, and nothing for a probe to answer differently from one open to the next.

It is the FIRST capability this list holds that the sibling backend PROBES, so the asymmetry
is worth stating rather than leaving to be read as an inconsistency:
`cl-gbdt/src/lightgbm/native' names `:custom-evaluation' in its `*optional-symbols*' because
its own read needs `LGBM_BoosterGetPredict', `LGBM_BoosterGetNumPredict' and
`LGBM_BoosterGetNumClasses', not one of which is in ITS `*required-symbols*' -- so a LightGBM
missing any of the three opens perfectly well and cannot serve a custom metric, which is
exactly the state a probe exists to detect. No XGBoost this backend will open is in the
corresponding state. The two backends answer the same capability from opposite lists because
their libraries put the same entry point on opposite sides of required, not because they
disagree about what the capability means.

Naming it in BOTH lists on ONE backend is the thing that would be wrong, and is why this entry
is not hedged by also listing `XGBoosterPredictFromDMatrix' under `*optional-symbols*' above:
`probe-capabilities' records PROVIDED entries ahead of probed ones, so the probe's answer would
be unreachable, and its own docstring calls that combination a contradiction in the backend's
declarations. `tools/ci/check-abi-blacklist.lisp''s own header names the reverse risk this
choice accepts -- a capability declared provided is recorded true on a library that lacks the
symbol, and nothing there would notice -- which is precisely why the C function it rests on has
to be a REQUIRED one, as it is.

Every name here must be registered in `cl-gbdt/src/backend''s `*known-capabilities*', or
`backend-supports-p' would signal `unknown-capability' for a capability the plist claims;
`tools/ci/check-abi-blacklist.lisp''s CHECK C is what enforces that, for this list and
`*optional-symbols*' alike.")

(defun %read-version ()
  "Return XGBoost's version as a \"MAJOR.MINOR.PATCH\" string, read via `XGBoostVersion'.

Unlike every other foreign call in this file, `XGBoostVersion' returns `:void' and has no
status to check with `check-xgb' -- it cannot report failure. Unlike LightGBM, whose C API
has no runtime version query at all (`lightgbm-backend' leaves `backend-version' NIL for
exactly that reason), XGBoost exposes this, so this backend populates it. The asymmetry is
between the libraries, not an inconsistency between the two Lisp backends."
  (cffi:with-foreign-objects ((major :int) (minor :int) (patch :int))
    (xg-boost-version major minor patch)
    (format nil "~D.~D.~D"
            (cffi:mem-ref major :int) (cffi:mem-ref minor :int) (cffi:mem-ref patch :int))))

;;; ---------------------------------------------------------------------------
;;; Datasets

(defun %dense-matrix-config-json (missing)
  "Return the config JSON `XGDMatrixCreateFromDense' expects, with MISSING as its sentinel
for a missing value.

MISSING is rendered by `missing-value-json', which signals `unsupported-argument' against
`:xgboost' for anything that is neither a `real' nor NIL, and renders NIL as `NaN' -- the
value this wrapper fixed the sentinel at unconditionally before `make-dataset' took a
:MISSING argument, and therefore what a caller who passes none keeps getting. NaN is also
the only sentinel `with-foreign-matrix' can produce for a `double-float' or `single-float'
array element without the caller choosing one, which is why it was the fixed choice.

`nthread' and `data_split_mode' -- the other two keys the vendored header documents for this
call -- stay unexposed, as they were when this was a constant string: nothing in the unified
API has a use for either yet. Only the sentinel became a caller's decision, because only the
sentinel is a property of the caller's own DATA rather than of how XGBoost should process
it."
  (format nil "{\"missing\":~A}" (missing-value-json missing :xgboost)))

(defun %array-interface-typestr (element-type)
  "Map ELEMENT-TYPE, as `with-foreign-matrix' reports it, to the NumPy array-interface
typestr XGBoost's `array-interface-json' expects.

`ecase', not `case': `with-foreign-matrix' promises only `double-float' or `single-float'
-- see `cl-gbdt/src/data''s `foreign-element-type' -- so any other value reaching here is
a bug in this file, not a value XGBoost should silently be told the shape of."
  (ecase element-type
    (double-float "<f8")
    (single-float "<f4")))

(defun %create-dmatrix (matrix missing)
  "Build a DMatrix from MATRIX via `XGDMatrixCreateFromDense', returning its raw pointer.
MISSING is the value in MATRIX that means *missing*, or NIL for this backend's own default;
it reaches the library through `%dense-matrix-config-json'.

MATRIX's foreign buffer is described to XGBoost with `array-interface-json' rather than
handed over as a pointer and dimensions -- `XGDMatrixCreateFromDense' takes the array
interface, unlike LightGBM's `LGBM_DatasetCreateFromMat'. The buffer only needs to stay
pinned for the duration of this call, since XGBoost copies it into its own representation
before returning.

The config JSON is built BEFORE the matrix is pinned, so a MISSING that
`%dense-matrix-config-json' refuses signals with nothing yet held: rejecting the argument
never has to unwind out of a pin or a foreign allocation. That is a property of this
function's own body alone: `predict' in `cl-gbdt/src/xgboost/api' -- which is where that
procedure lives now, the protocol method of the same name having kept only the portable
contract -- calls this function once, pins nothing of MATRIX itself around that call, and
reads the DMatrix's row count back afterward via `%dataset-num-rows' rather than a pin of
its own -- so the property holds at that call site with nothing further required of it. The
other caller, `create-dataset' in the same file, reaches this through `%dataset-pointer' and
likewise pins nothing of MATRIX around it."
  (let ((config-json (%dense-matrix-config-json missing)))
    (with-foreign-matrix (data-pointer nrow ncol element-type) matrix
      (let ((typestr (%array-interface-typestr element-type)))
        (cffi:with-foreign-string (data (array-interface-json data-pointer typestr nrow ncol))
          (cffi:with-foreign-string (config config-json)
            (cffi:with-foreign-object (out :pointer)
              (check-xgb (xgd-matrix-create-from-dense data config out)
                         "XGDMatrixCreateFromDense")
              (cffi:mem-ref out :pointer))))))))

(defun %json-string (string)
  "Return STRING as a double-quoted JSON string literal, escaping every character JSON
requires escaped -- `\"', `\\', and the C0 control characters (U+0000-U+001F), the latter
written `\\uXXXX' except for the four JSON gives their own short escape (`\\b', `\\t',
`\\n', `\\f', `\\r'). Every other character, ASCII or not, is copied through unescaped --
valid per RFC 8259, which requires escaping only the characters above and leaves the rest to
the string's own encoding.

`%uri-config-json' is the one caller. URI, unlike every other config-json builder's argument
in this file, is caller-controlled path data rather than a number, so the bare `~A'
interpolation `%dense-matrix-config-json', `%csr-matrix-config-json' and
`%predict-config-json' use for their own arguments is not safe here -- a path holding a
literal `\"' or `\\', neither of which `cl-gbdt/src/xgboost/file-input:file-uri''s own guard
rejects, would otherwise corrupt the JSON this function returns."
  (with-output-to-string (out)
    (write-char #\" out)
    (loop :for char :across string
          :do (case char
                (#\" (write-string "\\\"" out))
                (#\\ (write-string "\\\\" out))
                (#\Backspace (write-string "\\b" out))
                (#\Tab (write-string "\\t" out))
                (#\Newline (write-string "\\n" out))
                (#\Page (write-string "\\f" out))
                (#\Return (write-string "\\r" out))
                (t (if (< (char-code char) #x20)
                       (format out "\\u~4,'0X" (char-code char))
                       (write-char char out)))))
    (write-char #\" out)))

(defun %uri-config-json (uri)
  "Return the config JSON `XGDMatrixCreateFromURI' expects for URI,
`cl-gbdt/src/xgboost/file-input:file-uri''s composed string.

Two keys, both measured (record section 2): `\"uri\"', holding URI itself, and `\"silent\"',
fixed at 0. Every measured call used exactly `{\"uri\": \"<path>\", \"silent\": 0}' together
-- whether `\"silent\"' can be dropped and only `\"uri\"' sent was never isolated, so this
always sends both rather than guess at an untested combination. `data_split_mode', the
vendored header's third documented key for this call, was never exercised either and stays
unexposed, mirroring `%dense-matrix-config-json' leaving `nthread'/`data_split_mode' alone
for the identical reason.

URI's own value is rendered with `%json-string' rather than the `~A' interpolation this
file's other config-json builders use, because URI -- unlike MISSING or the predict config's
numeric keys -- is caller-controlled path data that can hold a character JSON itself
requires escaped."
  (format nil "{\"uri\":~A,\"silent\":0}" (%json-string uri)))

(defun %create-dmatrix-from-uri (uri)
  "Build a DMatrix from URI via `XGDMatrixCreateFromURI', returning its raw pointer.

URI is `cl-gbdt/src/xgboost/file-input:file-uri''s composed string -- PATH plus a
`?format=...' query segment, or no query at all for `:binary' -- and the format/contents
agreement `cl-gbdt/src/xgboost/api''s `create-dataset-from-file' exists to guarantee has
already been checked, by that function, before this one is ever reached: a mismatch that
disagrees the wrong way SIGSEGVs inside a thread dmlc creates for the parse, and there is
nothing this function, or any Lisp code running after the call, could do about that. This
function makes the foreign call and nothing else, exactly as `%create-dmatrix' does for the
dense path."
  (let ((config-json (%uri-config-json uri)))
    (cffi:with-foreign-string (config config-json)
      (cffi:with-foreign-object (out :pointer)
        (check-xgb (xgd-matrix-create-from-uri config out) "XGDMatrixCreateFromURI")
        (cffi:mem-ref out :pointer)))))

#+sbcl
(defun %call-with-pinned-csr (indptr indices values function)
  "Pin INDPTR, INDICES and VALUES and call FUNCTION with a foreign pointer to each, in that
order.

The three vectors come straight out of a `cl-gbdt/src/data' `csr-matrix', which already
stores them as the specialized `(simple-array (signed-byte 32) (*))' and `(simple-array
double-float (*))' the C API wants -- so there is nothing to convert here and nothing to
check, only memory to hold still. Each is a rank-one simple-array, whose object and whose
storage are one and the same, unlike the 2D case `cl-gbdt/src/data''s
`%call-with-pinned-matrix' has to pin a separately-allocated `array-storage-vector' for.

The pointers are valid only for the duration of FUNCTION, which is all the sparse ingestion
path needs: `XGDMatrixCreateFromCSR' copies the rows into XGBoost's own representation before
it returns, exactly the lifetime `%create-dmatrix' relies on for the dense path.

Duplicated verbatim from `cl-gbdt/src/lightgbm/native', which needs the identical helper for
`LGBM_DatasetCreateFromCSR'. The one file both could share it from -- `cl-gbdt/src/data' --
is re-exported wholesale by `cl-gbdt', so putting it there would publish a raw pinning
primitive as part of the unified API's surface. This backend pair is where the duplication
belongs instead, alongside `%set-feature-names' and `%free-*-unchecked', which mirror each
other across the two backends for the same reason."
  (sb-sys:with-pinned-objects (indptr indices values)
    (flet ((sap-pointer (vector)
             (cffi:make-pointer (sb-sys:sap-int (sb-sys:vector-sap vector)))))
      (funcall function (sap-pointer indptr) (sap-pointer indices) (sap-pointer values)))))

#-sbcl
(defun %call-with-pinned-csr (indptr indices values function)
  "Copy INDPTR, INDICES and VALUES into foreign buffers and call FUNCTION with a pointer to
each, in that order.

The fallback for an implementation with no way to pin a Lisp array, mirroring
`cl-gbdt/src/data''s `%call-with-copied-matrix' -- same contract as the SBCL version above,
paid for with a copy. `#'identity' as every coercer: a `csr-matrix' has already coerced all
three vectors to exactly these element types, so there is nothing left for
`write-foreign-sequence' to convert."
  (cffi:with-foreign-objects ((indptr-buffer :int32 (length indptr))
                              (indices-buffer :int32 (length indices))
                              (values-buffer :double (length values)))
    (write-foreign-sequence indptr-buffer :int32 indptr #'identity)
    (write-foreign-sequence indices-buffer :int32 indices #'identity)
    (write-foreign-sequence values-buffer :double values #'identity)
    (funcall function indptr-buffer indices-buffer values-buffer)))

(defun %csr-matrix-config-json (missing)
  "Return the config JSON `XGDMatrixCreateFromCSR' expects, with MISSING as its sentinel for
a missing value. The same sentinel `%dense-matrix-config-json' renders for the dense path,
and for the same reason -- see that function's docstring, whose reasoning about MISSING's
rendering and about XGBoost's other config keys applies here unchanged.

Kept as its own function rather than shared with the dense one, because the two are
arguments to two different C entry points -- not because the key sets differ today. They do
not: the vendored header (`ffi-spec/xgboost/include/xgboost/c_api.h') documents
`XGDMatrixCreateFromCSR''s CONFIG by cross-reference, \"See @ref XGDMatrixCreateFromDense
for details\", so the recognized keys are `missing' plus the optional `nthread' and
`data_split_mode' for both. What nothing binds is that they stay identical: they are
separate parameters of separate functions, and one builder per call means a future change to
what this wrapper sends to one of them is a decision about that call rather than a silent
change to the other's.

An entry a `csr-matrix' does not store is absent, not equal to the sentinel, and XGBoost
treats an absent CSR entry as missing regardless of what this says -- that is what CSR means
to the library and no config key changes it. This sentinel decides only what a *stored* value
means, which is exactly what it decides on the dense path."
  (format nil "{\"missing\":~A}" (missing-value-json missing :xgboost)))

(defun %create-dmatrix-from-csr (indptr indices values num-columns missing)
  "Build a DMatrix from a `csr-matrix''s INDPTR, INDICES and VALUES via
`XGDMatrixCreateFromCSR', returning its raw pointer. NUM-COLUMNS is the matrix's declared
width, passed as `XGDMatrixCreateFromCSR''s own NCOL rather than left to the library to infer
from the largest index present -- the two are different facts (see `make-csr-matrix''s
docstring), and only the caller knows the first. MISSING is the value among VALUES that means
*missing*, or NIL for this backend's own default; it reaches the library through
`%csr-matrix-config-json'.

Each of the three vectors is described to XGBoost with its own `array-interface-json', the
same way `%create-dmatrix' describes a dense buffer: this entry point takes three separate
array-interface descriptors where LightGBM's takes three raw pointers and a pair of dtype
tags. The typestrs are fixed rather than derived -- `csr-matrix' stores INDPTR and INDICES as
`(signed-byte 32)' and VALUES as `double-float' and nothing else, so there is no per-call
element type to map the way `%array-interface-typestr' maps one for the dense path.

The buffers only need to stay pinned for the duration of this call, since XGBoost copies
them into its own representation before returning. The config JSON is built before any of
them is pinned, for the reason `%create-dmatrix' gives."
  (let ((config-json (%csr-matrix-config-json missing)))
    (%call-with-pinned-csr
     indptr indices values
     (lambda (indptr-pointer indices-pointer values-pointer)
       (cffi:with-foreign-string
           (indptr-json (array-interface-json indptr-pointer "<i4" (length indptr)))
         (cffi:with-foreign-string
             (indices-json (array-interface-json indices-pointer "<i4" (length indices)))
           (cffi:with-foreign-string
               (values-json (array-interface-json values-pointer "<f8" (length values)))
             (cffi:with-foreign-string (config config-json)
               (cffi:with-foreign-object (out :pointer)
                 (check-xgb (xgd-matrix-create-from-csr indptr-json indices-json values-json
                                                        num-columns config out)
                            "XGDMatrixCreateFromCSR")
                 (cffi:mem-ref out :pointer))))))))))

(defun %set-info-field (dataset-pointer field-name values)
  "Attach the sequence VALUES to DATASET-POINTER's FIELD-NAME via
`XGDMatrixSetInfoFromInterface'. Used for `label' and `weight', both of which XGBoost
stores as single-precision floats regardless of the training matrix's own element type --
the same convention `cl-gbdt/src/lightgbm/native' follows for the same two fields.

VALUES is copied into a freshly allocated foreign buffer -- via `write-foreign-sequence',
coercing each element to `single-float' -- rather than pinning a caller-supplied Lisp
array directly, so this accepts any sequence, not only an SBCL simple-array."
  (let ((count (length values)))
    (cffi:with-foreign-object (buffer :float count)
      (write-foreign-sequence buffer :float values (lambda (value) (coerce value 'single-float)))
      (cffi:with-foreign-string (field field-name)
        (cffi:with-foreign-string (descriptor (array-interface-json buffer "<f4" count))
          (check-xgb (xgd-matrix-set-info-from-interface dataset-pointer field descriptor)
                     "XGDMatrixSetInfoFromInterface"))))))

(defun %set-group-field (dataset-pointer group)
  "Attach the sequence GROUP -- ranking query-group sizes -- to DATASET-POINTER via
`XGDMatrixSetUIntInfo' under XGBoost's \"group\" field.

XGBoost's group array is `unsigned' (`:uint' in CFFI terms), unlike LightGBM's `int32' --
see `cl-gbdt/src/lightgbm/native''s `%set-group-field', which writes the same query-group
sizes through `LGBM_DatasetSetField' as `int32' instead. Both hold the same nonnegative
row counts; only the C prototypes differ. `XGDMatrixSetUIntInfo' is deprecated upstream
in favor of `XGDMatrixSetInfoFromInterface', but is still the entry point this backend
uses for GROUP -- it takes a plain array rather than requiring a JSON array-interface
descriptor, and remains part of the vendored library's ABI.

GROUP is copied into a freshly allocated foreign buffer -- via `write-foreign-sequence',
rounding each element to an integer -- rather than pinning a caller-supplied Lisp array
directly, matching `%set-info-field''s reasoning for LABEL and WEIGHT above."
  (let ((count (length group)))
    (cffi:with-foreign-object (buffer :uint count)
      (write-foreign-sequence buffer :uint group #'round)
      (cffi:with-foreign-string (field "group")
        (check-xgb (xgd-matrix-set-u-int-info dataset-pointer field buffer count)
                   "XGDMatrixSetUIntInfo")))))

(defun %set-feature-names (dataset-pointer feature-names)
  "Attach FEATURE-NAMES, a list of strings, to DATASET-POINTER via
`XGDMatrixSetStrFeatureInfo' under XGBoost's \"feature_name\" field.

Signals `unsupported-argument' against `:xgboost' when FEATURE-NAMES is not a proper
list, via `check-feature-names' -- checked before COUNT is computed, since `length' on a
dotted list is exactly the raw `type-error' that check exists to head off. Mirrors
`cl-gbdt/src/lightgbm/native''s identical guard in its own `%set-feature-names'.

Every string successfully allocated is freed on any exit, including one signaled partway
through the allocation loop itself -- ALLOCATED tracks exactly how many of the COUNT slots
hold a real `foreign-string-alloc' result, matching
`cl-gbdt/src/lightgbm/native''s `%set-feature-names', which this mirrors call-for-call
apart from the field name and the C function it calls."
  (check-feature-names feature-names :xgboost)
  (let ((count (length feature-names))
        (allocated 0))
    (cffi:with-foreign-object (names :pointer count)
      (unwind-protect
           (progn
             (loop :for name :in feature-names
                   :for index :from 0
                   :do (setf (cffi:mem-aref names :pointer index)
                             (cffi:foreign-string-alloc name))
                       (setf allocated (1+ index)))
             (cffi:with-foreign-string (field "feature_name")
               (check-xgb (xgd-matrix-set-str-feature-info dataset-pointer field names count)
                          "XGDMatrixSetStrFeatureInfo")))
        (dotimes (index allocated)
          (cffi:foreign-string-free (cffi:mem-aref names :pointer index)))))))

(defun %set-feature-types (dataset-pointer feature-types)
  "Attach FEATURE-TYPES, a list of strings, to DATASET-POINTER via
`XGDMatrixSetStrFeatureInfo' under XGBoost's \"feature_type\" field.

One string per column, as `cl-gbdt/src/config/categorical-features''s
`categorical-feature-types' renders them. Mirrors `%set-feature-names' above call for call,
apart from the field name -- including how it frees on a signal partway through the
allocation loop, whose reasoning that function's docstring carries."
  (let ((count (length feature-types))
        (allocated 0))
    (cffi:with-foreign-object (types :pointer count)
      (unwind-protect
           (progn
             (loop :for type :in feature-types
                   :for index :from 0
                   :do (setf (cffi:mem-aref types :pointer index)
                             (cffi:foreign-string-alloc type))
                       (setf allocated (1+ index)))
             (cffi:with-foreign-string (field "feature_type")
               (check-xgb (xgd-matrix-set-str-feature-info dataset-pointer field types count)
                          "XGDMatrixSetStrFeatureInfo")))
        (dotimes (index allocated)
          (cffi:foreign-string-free (cffi:mem-aref types :pointer index)))))))

(defun %free-dmatrix (pointer)
  "Free the DMatrix at POINTER via `XGDMatrixFree', signalling `foreign-call-error' when
the library reports failure.

Extracted so `cl-gbdt/src/xgboost/api''s `free-dataset' and `predict' each delegate
the call itself, rather than naming `xgd-matrix-free' directly, to this file. Those were
`cl-gbdt/src/xgboost/protocol''s methods of the same names when the extraction was made --
policy section 3's Layer 2 delegating to Layer 1 -- and both methods now delegate in turn
to the Layer 1 functions, so both ends of this call are Layer 1. `free-dataset' passes this
to `release-handle' as its release function; `predict' calls it directly on its transient
DMatrix, inside a `handler-case' that turns a failure into a `warn' instead of letting it
propagate over whatever that function is already returning."
  (check-xgb (xgd-matrix-free pointer) "XGDMatrixFree"))

(defun %free-dmatrix-unchecked (pointer)
  "Free the DMatrix at POINTER via `XGDMatrixFree' without checking its returned status.

`cl-gbdt/src/xgboost/api''s `create-dataset' calls this from its cleanup path when
ownership of a partially built dataset never transferred to a handle -- a signal already
unwinding the stack there must not be replaced by a status-check failure from this
best-effort free, so unlike `%free-dmatrix' this does not consult `check-xgb' at all. It is
the only caller: `make-dataset' held that cleanup path until `create-dataset' took the whole
procedure over, and now reaches this function through it rather than naming it at all."
  (xgd-matrix-free pointer))

(defun %dataset-num-rows (pointer)
  "Return the row count of the DMatrix at POINTER, read via `XGDMatrixNumRow'."
  (cffi:with-foreign-object (out :uint64)
    (check-xgb (xgd-matrix-num-row pointer out) "XGDMatrixNumRow")
    (cffi:mem-ref out :uint64)))

(defun %dataset-num-features (pointer)
  "Return the feature (column) count of the DMatrix at POINTER, read via
`XGDMatrixNumCol'."
  (cffi:with-foreign-object (out :uint64)
    (check-xgb (xgd-matrix-num-col pointer out) "XGDMatrixNumCol")
    (cffi:mem-ref out :uint64)))

;;; ---------------------------------------------------------------------------
;;; Training

(defun %create-booster (dmatrix-pointers)
  "Create a booster over DMATRIX-POINTERS via `XGBoosterCreate', returning its pointer.
DMATRIX-POINTERS is a list of raw DMatrix pointers -- the training set's first, then each
validation set's -- or NIL for a booster with no data at all, which is how `load-model'
builds one before `XGBoosterLoadModel' populates it.

Unlike LightGBM's `LGBM_BoosterCreate', which takes a single training-set handle and
gains validation sets afterward through `LGBM_BoosterAddValidData', XGBoost's
`XGBoosterCreate' takes the whole array of DMatrix handles a booster will ever reference
up front -- there is no separate \"add valid data\" entry point. An empty
DMATRIX-POINTERS is passed to `XGBoosterCreate' as a null pointer with length 0, rather
than a zero-length foreign array, to avoid depending on whether a zero-count
`cffi:with-foreign-object' allocation is meaningful.

Signals `foreign-call-error' when creation reports success but writes a null handle --
the same guard `cl-gbdt/src/xgboost/api''s `create-dataset' applies to
`XGDMatrixCreateFromDense', for the same reason: every later call through this handle would
otherwise dereference it blindly."
  (flet ((create-with (dmats count)
           (cffi:with-foreign-object (out :pointer)
             (check-xgb (xg-booster-create dmats count out) "XGBoosterCreate")
             (let ((booster-pointer (cffi:mem-ref out :pointer)))
               (when (cffi:null-pointer-p booster-pointer)
                 (error 'foreign-call-error
                        :function-name "XGBoosterCreate"
                        :code 0
                        :message "reported success but returned a null booster handle"))
               booster-pointer))))
    (let ((count (length dmatrix-pointers)))
      (if (zerop count)
          (create-with (cffi:null-pointer) 0)
          (cffi:with-foreign-object (dmats :pointer count)
            (loop :for pointer :in dmatrix-pointers
                  :for index :from 0
                  :do (setf (cffi:mem-aref dmats :pointer index) pointer))
            (create-with dmats count))))))

(defun %set-parameters (booster-pointer parameters)
  "Apply PARAMETERS, a plist, to BOOSTER-POINTER via `XGBoosterSetParam', one call per
entry of `normalize-parameters'.

Unlike LightGBM, which takes a single space-separated \"key=value\" string at
`LGBM_BoosterCreate' time, XGBoost has no bulk-parameter entry point -- each (NAME
. VALUE) pair becomes its own foreign call. Every one of them is checked with
`check-xgb', not just the boosting calls: the functional tests on the LightGBM branch
found five update calls all returning 0 while the model did not train, so a status code
alone is necessary but not sufficient, and there is no reason to trust this call more
than that one."
  (dolist (pair (normalize-parameters parameters))
    (cffi:with-foreign-string (name (car pair))
      (cffi:with-foreign-string (value (cdr pair))
        (check-xgb (xg-booster-set-param booster-pointer name value) "XGBoosterSetParam")))))

(defun %boosted-rounds (booster-pointer)
  "Return BOOSTER-POINTER's boosted-round count, read via `XGBoosterBoostedRounds'."
  (cffi:with-foreign-object (out :int)
    (check-xgb (xg-booster-boosted-rounds booster-pointer out) "XGBoosterBoostedRounds")
    (cffi:mem-ref out :int)))

(defun %update-one-iteration (booster-pointer train-data-pointer)
  "Advance BOOSTER-POINTER by one boosting iteration on TRAIN-DATA-POINTER via
`XGBoosterUpdateOneIter'.

Unlike LightGBM's `LGBM_BoosterUpdateOneIter', which tracks its own iteration count
internally, `XGBoosterUpdateOneIter' takes the round index as an explicit `iter'
argument. Rather than this file maintaining a separate counter that could drift from
XGBoost's own, `iter' is read back fresh via `%boosted-rounds' immediately before each
call -- the read-back is load-bearing, not just a diagnostic: a booster whose round
count did not start at 0 (in principle, a future entry point resuming training) would
have a locally-tracked counter silently repeating rounds already boosted."
  (check-xgb (xg-booster-update-one-iter
              booster-pointer (%boosted-rounds booster-pointer) train-data-pointer)
             "XGBoosterUpdateOneIter"))

(defun %check-booster-datasets-live (booster)
  "Signal `released-handle-error' when any dataset BOOSTER depends on -- its training set, or
any validation set attached at creation by `train' or `create-booster' -- has already been
freed.

The two kinds are NOT equally dangerous, and saying so together would overstate one and
understate the other. Both were measured against the vendored library by freeing the dataset
and then calling `%update-one-iteration' directly -- past this check, and past
`handle-live-pointer', on the raw pointers:

  TRAINING SET  `XGBoosterUpdateOneIter' takes the DMatrix handle explicitly and
                `update-one-iteration' reads it back out of `booster-training-set', so a
                freed one goes straight into C as a pointer to storage `XGDMatrixFree'
                deleted. Measured: SB-SYS:MEMORY-FAULT-ERROR at #x0 in 3/3 runs, from inside
                the library. SBCL turning that fault into a condition on this platform is a
                signal handler catching a segfault, not a contract to rely on.
  VALIDATION SET  Returns normally, and again on the next call. `XGBoosterUpdateOneIter'
                never consults the array `XGBoosterCreate' was given -- it takes its one
                DMatrix explicitly -- and XGBoost keeps its own reference to each DMatrix in
                that array, so the object outlives the handle naming it. There is no fault to
                head off here; what this check buys is that a caller who frees a validation
                set and keeps boosting is told so, instead of the run continuing against data
                it has already released. Those retained handles are not decorative -- this
                backend's `evaluation' hands exactly them to `XGBoosterEvalOneIter', reading
                each through `handle-live-pointer' first for this same reason.

So the training-set arm is a memory-safety guard and the validation-set arm is a contract
guard, and both are checked here, before any foreign call, because a caller cannot tell from
the outside which one it is about to trip. `cl-gbdt/src/lightgbm/native''s function of the
same name draws no such distinction, and its docstring says why it should not: on that
library the booster keeps its validation sets' own stored pointers and works against them
every iteration, so both kinds are the training-set case there.

`booster-training-set' is NIL for a `load-model' booster, which has no training set and needs
no check -- `cl-gbdt/src/xgboost/api''s `update-one-iteration' signals `missing-training-set'
for that case rather than this function; `booster-validation-sets' is NIL when no validation
set was attached."
  (let ((training-set (booster-training-set booster)))
    (when (and training-set (handle-released-p training-set))
      (error 'released-handle-error :object training-set)))
  (dolist (validation-set (booster-validation-sets booster))
    (when (handle-released-p validation-set)
      (error 'released-handle-error :object validation-set))))

(defun %free-booster (pointer)
  "Free the booster at POINTER via `XGBoosterFree', signalling `foreign-call-error' when
the library reports failure.

Extracted the same way `%free-dmatrix' is -- see that function's docstring, including the
relocation both have since made -- so `cl-gbdt/src/xgboost/api''s `free-booster' delegates
the call itself to this file instead of naming `xg-booster-free' directly."
  (check-xgb (xg-booster-free pointer) "XGBoosterFree"))

(defun %free-booster-unchecked (pointer)
  "Free the booster at POINTER via `XGBoosterFree' without checking its returned status.

`cl-gbdt/src/xgboost/api''s `create-booster', `load-model' and `slice-model' each call this
from their cleanup path when ownership of a partially or newly built booster never
transferred to a handle -- see `%free-dmatrix-unchecked''s docstring for why a signal
already unwinding the stack there must not be replaced by a status-check failure from this
best-effort free.
`cl-gbdt/src/xgboost/protocol''s `train' no longer calls this: it now holds a handle from
`create-booster' rather than a raw pointer, so its own cleanup calls `free-booster' instead,
wrapped the same way for the same reason -- see the comment at its own cleanup."
  (xg-booster-free pointer))

;;; ---------------------------------------------------------------------------
;;; Inference

(defun %predict-type (kind)
  "Map the protocol's KIND keyword onto XGBoost's predict-config `\"type\"' value.

`ecase', not `case': an unrecognized KIND must error rather than silently predicting
something else."
  (ecase kind
    (:normal 0)
    (:raw 1)
    (:contrib 2)
    (:leaf-index 6)))

(defun %resolve-num-iteration (num-iteration)
  "Return NUM-ITERATION as XGBoost spells it on the wire: 0 means all iterations --
`\"iteration_end\":0' becomes \"the size of tree model\", per
`XGBoosterPredictFromDMatrix''s own documentation -- which is what NIL means in the
protocol."
  (or num-iteration 0))

(defun %predict-config-json (predict-type iteration-end
                             &key (training nil) (missing nil missing-supplied-p))
  "Return the JSON config `XGBoosterPredictFromDMatrix' expects for PREDICT-TYPE (already
mapped by `%predict-type') and ITERATION-END (already resolved by
`%resolve-num-iteration').

TRAINING renders the `\"training\"' key, which is always present: false by default, true when
this argument is. The key is required -- omitting it entirely returns -1, `Argument
`training` is required for `XGBoosterPredictFromDMatrix`', confirmed against the vendored
library -- so what this argument decides is the VALUE, never whether the key appears, which is
the opposite of MISSING below.

What the value means is documented in the vendored header
(`ffi-spec/xgboost/include/xgboost/c_api.h:1180-1191'), against this very config: prediction
runs in one of two scenarios, obtaining `y_pred' from the model, or \"obtain[ing] the
prediction for computing gradients\", and \"the second scenario applies when you are defining a
custom objective function\". So false is right for `predict', whose caller asked for a
prediction, and true is right for the read `train''s OBJECTIVE branch makes through
`%booster-predictions', whose result exists only to be handed to a caller's objective function
-- that ONE call site is the only caller passing true. `%booster-predictions' takes TRAINING as
its own argument rather than fixing it, because `train''s EVALUATION branch reads through the
same function and passes false: what that branch hands the caller is an ordinary prediction,
scenario one, not a gradient's input -- see that function.

The two are NOT interchangeable in general, whatever a `gbtree' measurement suggests: the same
header names DART's training-time dropout as the case where they differ, \"the prediction
result will be different from the one obtained by normal inference step due to dropped trees\".
`:booster \"dart\"' reaches `%set-parameters' untouched, so that configuration is reachable
from the unified API.

`\"strict_shape\":true' always: without it, XGBoost's non-strict mode squeezes away a
single-class model's trailing dimension inconsistently with a multi-class model's --
exactly the assumption `predict' exists to avoid making. With it, `out_shape' and
`out_dim' report a shape this file can trust uniformly across every KIND and class
count.

MISSING, when SUPPLIED AT ALL, adds the `\"missing\"' key and is the value that means
*missing*, rendered by `missing-value-json' the way `%dense-matrix-config-json' and
`%csr-matrix-config-json' render the same argument for the two ingestion entry points -- NIL
included, which renders as `NaN', the sentinel this wrapper sent unconditionally before
`predict' took a :MISSING argument and therefore what a caller who names none keeps getting.
What decides whether the key appears is the argument being SUPPLIED, not its value: NIL is a
sentinel here rather than an absence, so it cannot also stand for one.

Left out entirely by the dense path, because `XGBoosterPredictFromDMatrix' has no use for the
key -- a DMatrix settled its own missing sentinel when it was built, out of the config
`%create-dmatrix' builds -- and supplied by `XGBoosterPredictFromCSR', which is INPLACE
prediction with no DMatrix behind it and therefore nothing to have settled one: confirmed
against the vendored library, that call refuses outright without the key (\"Argument
`missing` is required\"), rather than assuming a default."
  (format nil "{\"type\":~D,\"training\":~A,\"iteration_begin\":0,~
\"iteration_end\":~D,\"strict_shape\":true~A}"
          predict-type (if training "true" "false") iteration-end
          (if missing-supplied-p
              (format nil ",\"missing\":~A" (missing-value-json missing :xgboost))
              "")))

(defun %total-element-count (shape-pointer dim)
  "Return the product of the DIM `:uint64' entries at SHAPE-POINTER -- the total element
count `XGBoosterPredictFromDMatrix' reports through its `out_shape'/`out_dim' pair,
however many dimensions the library used. `predict' divides this by the row count to get
the result's column width, rather than assuming a class count -- see that function's
docstring."
  (let ((total 1))
    (dotimes (index dim total)
      (setf total (* total (cffi:mem-aref shape-pointer :uint64 index))))))

(defun %reported-shape (shape-pointer dim)
  "Return the DIM entries at SHAPE-POINTER as a fresh list of integers.

`XGBoosterPredictFromDMatrix' writes the result's shape here -- as does `XGBoosterPredictFromCSR'
on the sparse path, which this reads the same way -- and `%total-element-count' beside
this reads only its product. Both are wanted: the product sizes the buffer, the shape is what
`predict' hands back as its second value. Measured, the shape is richer than the buffer's own
dimensions for two kinds -- a four-round three-class model over four features reports
(rows 4 3 1) for `:leaf-index' and (rows 3 5) for `:contrib', and even a four-round BINARY
model over three features reports (rows 4 1 1) and (rows 1 4) -- so folding it to
[rows, total/rows], which is all this backend did before, threw away structure the library had
already stated."
  (loop :for index :below dim
        :collect (cffi:mem-aref shape-pointer :uint64 index)))

(defun %predict-ncol (element-count nrow)
  "Return ELEMENT-COUNT's per-row width for a matrix of NROW rows.

Mirrors `cl-gbdt/src/lightgbm/native''s function of the same name and purpose. NROW = 0
is handled directly -- there is no row to give a width to -- and every other case is
asserted to divide evenly: `%total-element-count' reporting a total that is not an exact
multiple of NROW would mean either this file's shape arithmetic or XGBoost's own report
is wrong, which is worth surfacing loudly rather than truncating silently."
  (if (zerop nrow)
      0
      (multiple-value-bind (quotient remainder) (truncate element-count nrow)
        (assert (zerop remainder) ()
                "XGBoosterPredictFromDMatrix reported ~D elements for ~D rows, not an ~
                 exact multiple of the row count" element-count nrow)
        quotient)))

(defun %predict-from-dmatrix
    (booster-pointer dmatrix-pointer config out-shape out-dim out-result)
  "Run `XGBoosterPredictFromDMatrix' over BOOSTER-POINTER and DMATRIX-POINTER with CONFIG,
writing OUT-SHAPE, OUT-DIM and OUT-RESULT, and signal `foreign-call-error' when the
library reports failure.

Extracted so `cl-gbdt/src/xgboost/api''s `predict' delegates the call itself to this
file instead of naming `xg-booster-predict-from-d-matrix' directly. It was
`cl-gbdt/src/xgboost/protocol''s `predict' -- policy section 3's Layer 2 delegating to
Layer 1 -- when the extraction was made; that method now holds the portable contract only
and delegates the procedure to the Layer 1 function of the same name, so both ends of this
call are Layer 1. `predict' still owns interpreting OUT-SHAPE, OUT-DIM and OUT-RESULT
afterward, via `%total-element-count' and `%predict-ncol', and copying OUT-RESULT's
contents out before it returns."
  (check-xgb (xg-booster-predict-from-d-matrix
              booster-pointer dmatrix-pointer config out-shape out-dim out-result)
             "XGBoosterPredictFromDMatrix"))

(defun %predict-from-csr (booster-pointer indptr indices values num-columns predict-type
                           iteration-end missing out-shape out-dim out-result)
  "Run `XGBoosterPredictFromCSR' over the booster at BOOSTER-POINTER against the matrix a
`csr-matrix''s INDPTR, INDICES and VALUES describes, NUM-COLUMNS wide, writing OUT-SHAPE,
OUT-DIM and OUT-RESULT, and signal `foreign-call-error' when the library reports failure.
MISSING is the value among VALUES that means *missing*, or NIL for this backend's own
default.

Unlike `%predict-from-dmatrix' above, which takes the config string its caller built, this
builds its own -- `(%predict-config-json ... :missing missing)' -- and its own three
`array-interface-json' descriptors, the same way `%create-dmatrix-from-csr' does for
ingestion where `%create-dmatrix' takes one. The three vectors are pinned for the duration
of the call and no longer, which is all this needs: `XGBoosterPredictFromCSR' reads them and
writes OUT-RESULT before it returns, and the config JSON is built before any of them is
pinned, for the reason `%create-dmatrix' gives. The typestrs are fixed rather than derived,
because `csr-matrix' fixes INDPTR/INDICES at `(signed-byte 32)' and VALUES at `double-float'
and there is no per-call element type to map through `%array-interface-typestr'.

`XGBoosterPredictFromCSR' is XGBoost's INPLACE prediction, not the CSR spelling of
`XGBoosterPredictFromDMatrix' -- the vendored header
(`ffi-spec/xgboost/include/xgboost/c_api.h') documents it as such -- and that has two
consequences a caller sees, both measured against the vendored library:

  - It covers only PREDICT-TYPE 0 (`:normal') and 1 (`:raw'). 2 (`:contrib') and 6
    (`:leaf-index') come back as a clean nonzero return, \"Unsupported prediction type:N\",
    which `check-xgb' reports as `foreign-call-error'. This file does not pre-empt that
    refusal with a check of its own, and does not route those two KINDs through a transient
    DMatrix to work around it: either would be this wrapper inventing a policy over the
    library's own, and the capability `:sparse-input' declares this very entry point for.
  - `\"missing\"' is required in the config, which is why MISSING is always passed -- and,
    because it is required, this is also where a caller's own sentinel reaches XGBoost for a
    `csr-matrix', where the dense path instead settles one on the transient DMatrix it
    builds. See `%predict-config-json'.

The `m' parameter, an optional proxy DMatrix carrying meta info, is a null pointer: nothing
in this backend's prediction path has meta info to attach to a matrix it is only reading.

`cl-gbdt/src/xgboost/api''s `predict' still owns interpreting OUT-SHAPE, OUT-DIM and
OUT-RESULT afterward, via `%total-element-count' and `%predict-ncol', and copying
OUT-RESULT's contents out before it returns -- none of which differs between the two entry
points."
  (let ((config-json (%predict-config-json predict-type iteration-end :missing missing)))
    (%call-with-pinned-csr
     indptr indices values
     (lambda (indptr-pointer indices-pointer values-pointer)
       (cffi:with-foreign-string
           (indptr-json (array-interface-json indptr-pointer "<i4" (length indptr)))
         (cffi:with-foreign-string
             (indices-json (array-interface-json indices-pointer "<i4" (length indices)))
           (cffi:with-foreign-string
               (values-json (array-interface-json values-pointer "<f8" (length values)))
             (cffi:with-foreign-string (config config-json)
               (check-xgb (xg-booster-predict-from-csr
                           booster-pointer indptr-json indices-json values-json num-columns
                           config (cffi:null-pointer) out-shape out-dim out-result)
                          "XGBoosterPredictFromCSR")))))))))

;;; ---------------------------------------------------------------------------
;;; Custom objective and custom evaluation
;;;
;;; Below the Inference section rather than inside Training above, because the first of these
;;; two functions IS a prediction: `%booster-predictions' reaches `%predict-config-json',
;;; `%predict-type', `%predict-from-dmatrix', `%total-element-count' and `%predict-ncol', every
;;; one of them defined above. Keeping the pair here keeps this file's definition order and its
;;; call order the same. `cl-gbdt/src/lightgbm/native' keeps its own pair of the same names in
;;; its Training section instead, which is not an inconsistency between the two files: that
;;; library reads a booster's current scores with `LGBM_BoosterGetPredict', which is not a
;;; prediction entry point and depends on nothing in that file's Inference section.
;;;
;;; The first of the two serves BOTH caller-supplied functions `train' takes -- :OBJECTIVE's
;;; per-iteration score read and :EVALUATION's per-dataset one -- which is why it is named for
;;; what it returns rather than for either caller, and why the section it sits in names both.
;;; The second serves :OBJECTIVE alone; :EVALUATION reaches no update entry point at all.
;;;
;;; Neither function wraps itself in `with-foreign-float-traps-masked', and both reach the
;;; shared library. That is this file's ordinary rule, not an omission: both are reached only
;;; from `cl-gbdt/src/xgboost/protocol''s `train', whose whole body is already wrapped. See the
;;; Floating-point trap safety section at the top of this file -- what makes
;;; `evaluate-one-iteration' and `booster-boosted-rounds' exceptions to it is that
;;; `src/xgboost/all.lisp' publishes them from `cl-gbdt/xgboost', leaving no `defmethod' above
;;; them to establish the mask. Neither of these two is published from anywhere, and
;;; `tools/ci/check-float-traps.lisp' reads that same `:export' clause, so it has nothing to
;;; say about either -- a clean run of it is not evidence about these two functions.

(defun %booster-predictions (booster-pointer dmatrix-pointer rows kind
                             &key (training nil))
  "Return BOOSTER-POINTER's current predictions of KIND for DMATRIX-POINTER, as a
(ROWS GROUPS) `double-float' array.

ROWS is THAT DMatrix's own row count and never another's: on the fixture
tests/functional/custom-evaluation.lisp trains, the training set has 40 rows and the
validation set 17, and a caller handing one where the other belongs would mis-shape the array
rather than fail. Nothing here can catch that -- `%predict-ncol' below divides the library's
reported element count by whatever ROWS says -- so `train' reads each dataset's own count with
`%dataset-num-rows' and passes it beside that dataset's own pointer.

An `XGBoosterPredictFromDMatrix' call: `%predict-config-json' with KIND's predict type, as
`%predict-type' maps it, and ITERATION-END 0, every iteration the booster has so far.
`predict' builds its config through the same function, differing in that it resolves
ITERATION-END through `%resolve-num-iteration' from a caller's :NUM-ITERATION where this
passes the literal 0, and in TRAINING below.

Named after `cl-gbdt/src/lightgbm/native''s function of the same name because it answers the
same question -- what does this booster currently predict for that dataset -- and is asked it
from the same two places: `train''s OBJECTIVE branch, once per iteration for the training set,
and `train''s EVALUATION branch, once per dataset per iteration. One reader per backend rather
than one per caller, which is what the sibling has whether it wants it or not
(`LGBM_BoosterGetPredict' offers no kind to choose). What differs is the cost and what is
read: this one PREDICTS AFRESH on every call, a full prediction pass per dataset per
iteration, where LightGBM's hands back values that library already holds. This library has no
counterpart to `LGBM_BoosterGetPredict' at all, which is why the pass is paid rather than
avoided.

KIND is a parameter, not a constant, because the two call sites genuinely want different
numbers:

  - `train''s OBJECTIVE branch passes `:raw'. A gradient is computed from the MARGIN, and this
    backend does not rewrite PARAMETERS, so a configured objective's prediction transform is
    still in effect and `:normal' would hand that caller probabilities. See
    `cl-gbdt/src/xgboost/protocol''s `train'.
  - `train''s EVALUATION branch passes `:normal'. What a caller's metric is handed is that
    dataset's PREDICTION -- the very number `predict' returns for the same rows -- which is
    what `train''s generic-function docstring promises and what makes a caller-written metric
    comparable with the library's own. Measured on the vendored library over one
    `objective=binary:logistic' booster and the fixture above: this array agrees with
    `predict :kind :normal' over that dataset's own matrix to 0.0 for BOTH the 40-row training
    set and the 17-row validation set, under a library objective and again under a caller's
    own :OBJECTIVE.

  Under a caller's own :OBJECTIVE the two kinds are still NOT the same numbers here, unlike on
  LightGBM where `train' forces `objective=none' and the transform disappears with it: this
  backend rewrites nothing, so `binary:logistic''s transform stays in effect and the same
  booster's `:normal' read sits 0.756 from its `:raw' one -- measured on the fixture above,
  largest absolute difference over the training set after five iterations, and the SAME 0.756
  whether those five iterations ran the library's objective or a caller's. That equality is
  the point: the divergence is a property of the configured transform, which this backend
  leaves alone, not of the custom objective. It grows with the run -- 0.564 after one
  iteration, 0.677 after three, 0.906 after ten, on that same fixture -- so it is a direction,
  not a constant. It is the divergence `cl-gbdt/src/xgboost/protocol''s `predict' and `train'
  already document as the price of not rewriting PARAMETERS, inherited here rather than
  introduced: an :EVALUATION and an :OBJECTIVE in the same run are handed different numbers
  because they asked for different ones.

TRAINING renders `%predict-config-json''s `\"training\"' key, which is required either way --
omitting it returns -1, `Argument `training` is required for `XGBoosterPredictFromDMatrix`'.
TRUE is the value the vendored header asks for on the OBJECTIVE path: it names two prediction
scenarios (`ffi-spec/xgboost/include/xgboost/c_api.h:1180-1191'), obtaining `y_pred' and
\"obtain[ing] the prediction for computing gradients\", and says \"the second scenario applies
when you are defining a custom objective function\". The EVALUATION path is scenario ONE and
passes the default false, exactly as `predict' does -- which is also what keeps the agreement
measured above from resting on the two scenarios happening to coincide.

Nothing existing moves when that flag changes value, and that is measured rather than
assumed: on the default `gbtree' booster, `false' and `true' train identical models here.
Where they are NOT identical is the case the same header names -- DART, whose training-time
dropout means \"the prediction result will be different from the one obtained by normal
inference step due to dropped trees\". That difference is the header's statement, NOT a
measurement taken here; what a `:booster \"dart\"' run does under this flag has not been
measured. It is reachable, though: PARAMETERS reach `%set-parameters' untouched, so a DART
run's :OBJECTIVE sees the dropped ensemble while its :EVALUATION sees the full one.

The buffer is read ROW-MAJOR -- row I of output group K at `(+ (* I GROUPS) K)' -- which is
how `cl-gbdt/src/xgboost/protocol''s `predict' already reads the identical buffer from the
identical call, and which the multiclass layout test in
tests/functional/custom-objective.lisp holds directly: it asserts on the SCORES its
objective was handed at the second of two iterations, which is the only place a wrong
reading of this buffer shows up (at iteration 1 every score is 0).

GROUPS comes from the element count the library reports, the same `%total-element-count'
and `%predict-ncol' pair `predict' uses, rather than from a `num_class' parameter this file
would have to parse. `\"strict_shape\":true' is what makes that division trustworthy for a
single-group model as well as a multiclass one; see `%predict-config-json'.

OUT-RESULT is XGBoost's own memory, valid only until the next call into this booster, so
every element is copied out and coerced to `double-float' before this returns -- the same
lifetime `predict' works to, and the reason the copy loop below runs before anything else
here touches the library."
  (let ((config (%predict-config-json (%predict-type kind) 0 :training training)))
    (cffi:with-foreign-objects ((out-shape :pointer) (out-dim :uint64) (out-result :pointer))
      (cffi:with-foreign-string (config-string config)
        (check-xgb (%predict-from-dmatrix booster-pointer dmatrix-pointer config-string
                                          out-shape out-dim out-result)
                   "XGBoosterPredictFromDMatrix"))
      (let* ((element-count (%total-element-count (cffi:mem-ref out-shape :pointer)
                                                  (cffi:mem-ref out-dim :uint64)))
             (groups (%predict-ncol element-count rows))
             (buffer (cffi:mem-ref out-result :pointer))
             (predictions (make-array (list rows groups) :element-type 'double-float)))
        (dotimes (row rows predictions)
          (dotimes (group groups)
            (setf (aref predictions row group)
                  (coerce (cffi:mem-aref buffer :float (+ (* row groups) group))
                          'double-float))))))))

(defun %train-one-iteration-custom (booster-pointer dmatrix-pointer iteration grad hess)
  "Advance BOOSTER-POINTER by one iteration on the caller's GRAD and HESS, via
`XGBoosterTrainOneIter'. Both are (ROWS GROUPS) arrays.

They are flattened **row-major** -- row I of output group K at `(+ (* I GROUPS) K)' -- which
is what an `__array_interface__' of shape `[ROWS, GROUPS]' means, and converted to
`single-float'. Measured against the vendored library: a gradient written for output group 0
alone moves only that group's raw score under this layout and smears across all three under
the group-major one `cl-gbdt/src/lightgbm/native''s `%update-one-iteration-custom' needs.

The descriptor says `\"<f4\"', and **XGBoost validates that field against nothing**:
measured, `\"<f4\"' over a `float64' buffer produces garbage (1.2e27) and `\"<i4\"' is
accepted in silence and trains a different model (4.13 away from the built-in run). Nothing
downstream will report a wrong typestr, so this call site is the only thing keeping the
buffer from being misread.

The conversion goes through `objective-single-float', so an element that is not a real signals
`unsupported-element-type' naming its type rather than a `type-error' from inside `coerce'.
GRAD and HESS may be `double-float', `single-float' or general arrays --
`check-objective-result' checks their shape and leaves their elements to that function, one
element at a time as they are written, which is why nothing scans either array twice.

ITERATION is the round number the C signature asks for, and `train' reads it back from
`%boosted-rounds' at each call, the same way `%update-one-iteration' does. The REASON that
function gives does not carry over unchanged: it reads the count back so a booster whose
rounds did not start at 0 cannot have a local counter repeat rounds already boosted, whereas
the vendored header says of THIS entry point's `iter' that \"when training continuation is
used, the count should restart\" (`ffi-spec/xgboost/include/xgboost/c_api.h:1099-1100'), which
is a restarting local counter. The two conventions agree here only because `train' always
builds a fresh booster, so `%boosted-rounds' runs 0, 1, 2, ... -- a future entry point that
continued training an existing booster would have to choose between them. Measured, the value
does not affect the result at all: 0-based, always-zero and 1-based all reproduce the built-in
run exactly. It is sent honestly rather than relied on.

Both buffers only need to stay alive for the duration of this call: `XGBoosterTrainOneIter'
reads them and returns, exactly the lifetime `%create-dmatrix' relies on for the array
interface it hands `XGDMatrixCreateFromDense'."
  (let* ((rows (array-dimension grad 0))
         (groups (array-dimension grad 1))
         (count (* rows groups)))
    (cffi:with-foreign-objects ((grad-buffer :float count) (hess-buffer :float count))
      (dotimes (row rows)
        (dotimes (group groups)
          (let ((index (+ (* row groups) group)))
            (setf (cffi:mem-aref grad-buffer :float index)
                  (objective-single-float (aref grad row group)))
            (setf (cffi:mem-aref hess-buffer :float index)
                  (objective-single-float (aref hess row group))))))
      (cffi:with-foreign-string
          (grad-json (array-interface-json grad-buffer "<f4" rows groups))
        (cffi:with-foreign-string
            (hess-json (array-interface-json hess-buffer "<f4" rows groups))
          (check-xgb (xg-booster-train-one-iter booster-pointer dmatrix-pointer iteration
                                                grad-json hess-json)
                     "XGBoosterTrainOneIter"))))))

;;; ---------------------------------------------------------------------------
;;; Persistence

(defun %save-model (pointer filename)
  "Save the booster at POINTER to FILENAME via `XGBoosterSaveModel', signalling
`foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/xgboost/api''s `save-model' delegates the call itself to
this file instead of naming `xg-booster-save-model' directly."
  (check-xgb (xg-booster-save-model pointer filename) "XGBoosterSaveModel"))

(defun %load-model (booster-pointer filename)
  "Load a model from FILENAME into the booster at BOOSTER-POINTER via
`XGBoosterLoadModel', signalling `foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/xgboost/api''s `load-model' delegates the call itself to
this file instead of naming `xg-booster-load-model' directly."
  (check-xgb (xg-booster-load-model booster-pointer filename) "XGBoosterLoadModel"))

(defun %save-model-to-buffer (pointer config out-len out-dptr)
  "Serialize the booster at POINTER into an XGBoost-owned buffer via
`XGBoosterSaveModelToBuffer', writing OUT-LEN and OUT-DPTR, and signal
`foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/xgboost/api''s `model-to-string' delegates the call itself
to this file instead of naming `xg-booster-save-model-to-buffer' directly. `model-to-string'
still owns copying OUT-DPTR's contents out via OUT-LEN afterward."
  (check-xgb (xg-booster-save-model-to-buffer pointer config out-len out-dptr)
             "XGBoosterSaveModelToBuffer"))

;;; ---------------------------------------------------------------------------
;;; Feature importance

(defun %feature-importance-type (kind)
  "Map the protocol's KIND keyword onto XGBoost's `\"importance_type\"' config string.

`:gain' maps to XGBoost's `\"total_gain\"', not its `\"gain\"' -- the vendored header
(`ffi-spec/xgboost/include/xgboost/c_api.h') documents `\"gain\"' as the *average* gain
across the splits a feature is used in and `\"total_gain\"' as the sum, while LightGBM's
`C_API_FEATURE_IMPORTANCE_GAIN' -- what `cl-gbdt/src/lightgbm/native''s
`%feature-importance-type' maps `:gain' onto
-- and `feature-importance''s own generic-function docstring both mean the total. Mapping
to `\"gain\"' here would have this backend silently return a different quantity than
LightGBM under the identical keyword, the exact failure mode this project treats as most
serious: a working call moved between backends and returning different numbers without
either raising an error or the caller noticing.

`ecase', not `case': an unrecognized KIND must error rather than silently returning a
different importance measure."
  (ecase kind
    (:split "weight")
    (:gain "total_gain")))

(defun %booster-num-features (booster-pointer)
  "Return BOOSTER-POINTER's feature count, read via `XGBoosterGetNumFeature'.

`feature-importance' uses this to size its result -- see that method's docstring for
why `XGBoosterFeatureScore''s own `out_n_features' cannot be used for that instead."
  (cffi:with-foreign-object (out :uint64)
    (check-xgb (xg-booster-get-num-feature booster-pointer out) "XGBoosterGetNumFeature")
    (cffi:mem-ref out :uint64)))

(defun %feature-score-index (name)
  "Return the 0-based column index that `XGBoosterFeatureScore''s feature name NAME
identifies -- XGBoost's own default naming convention, the letter `\"f\"' followed by
the index, e.g. `\"f0\"', `\"f1\"'.

`XGBoosterFeatureScore' reports each score against a feature *name*, not a column
index, and `feature-importance' needs the index to scatter that score into a dense,
per-column result. This backend never attaches feature names to the booster itself --
confirmed empirically against the vendored library: a DMatrix built with
`make-dataset''s FEATURE-NAMES still reports `\"f1\"', not the caller's own name, from
this call -- so NAME is always XGBoost's default form, never a caller-supplied one.

Signals `foreign-call-error' when NAME does not match that form: a value reaching here
in any other shape would mean XGBoost's naming convention no longer matches what this
function assumes, and trusting an unparsed guess at the index would risk scattering a
score into the wrong column silently."
  (if (and (> (length name) 1)
           (char= (char name 0) #\f)
           (every #'digit-char-p (subseq name 1)))
      (parse-integer name :start 1)
      (error 'foreign-call-error
             :function-name "XGBoosterFeatureScore"
             :code 0
             :message (format nil "unexpected feature name ~S in out_features; expected ~
                                    XGBoost's default \"f<index>\" form"
                               name))))

(defun %check-feature-score-dim (backend out-dim out-shape)
  "Signal `unsupported-argument' when `XGBoosterFeatureScore' just wrote more than one
score per feature into OUT-DIM/OUT-SHAPE.

The vendored header (`ffi-spec/xgboost/include/xgboost/c_api.h') documents exactly one
combination producing that: a linear (`gblinear') booster's `weight' importance type on
a multi-class model, whose scores come back as a row-major [n_features, n_classes]
matrix rather than one number per feature. \"For tree model, out_n_feature is always
equal to out_n_scores\" per that same header -- confirmed empirically here too, across
every tree-booster case measured, multi-class included, which all report OUT-DIM 1.

`feature-importance' promises one entry per feature, matching
`cl-gbdt/src/lightgbm/api''s own `feature-importance': `LGBM_BoosterFeatureImportance'
has no shape output at all, and, confirmed empirically, already sums a multi-class
model's per-class contributions into one number per feature inside the library before
this ever sees it. XGBoost's per-class matrix has no such library-computed summary for
this backend to read back instead, and inventing a reduction here would risk being
actively misleading: `weight' on a linear booster is a signed coefficient, so summing
across classes could cancel a feature that matters a great deal for telling two classes
apart down to a value near zero. Rather than silently return a slice of the matrix --
the bug this guards against, `%feature-score-index' scattering only the first
`n_features' raw scores, which are really one feature's whole row -- or a reduction this
backend cannot vouch for, this refuses the combination outright, before any of
OUT-SCORES is read."
  (let ((dim (cffi:mem-ref out-dim :uint64)))
    (when (> dim 1)
      (let* ((shape-pointer (cffi:mem-ref out-shape :pointer))
             (shape (loop :for index :below dim
                          :collect (cffi:mem-aref shape-pointer :uint64 index))))
        (error 'unsupported-argument
               :backend (backend-name backend)
               :argument "feature-importance's booster"
               :reason (format nil "XGBoosterFeatureScore reported a ~D-dimensional shape ~
                                     ~A instead of one score per feature -- most likely a ~
                                     linear (gblinear) booster's :split importance on a ~
                                     multi-class model, whose scores are a per-class ~
                                     matrix; no single value per feature can be derived ~
                                     without inventing a reduction this backend does not ~
                                     vouch for"
                               dim shape))))))

(defun %feature-score
    (pointer config out-n-features out-features out-dim out-shape out-scores)
  "Run `XGBoosterFeatureScore' over the booster at POINTER with CONFIG, writing
OUT-N-FEATURES, OUT-FEATURES, OUT-DIM, OUT-SHAPE and OUT-SCORES, and signal
`foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/xgboost/api''s `feature-importance' delegates the call
itself to this file instead of naming `xg-booster-feature-score' directly.
`feature-importance' still owns interpreting the five out parameters afterward, via
`%check-feature-score-dim', `%booster-num-features' and `%feature-score-index'."
  (check-xgb (xg-booster-feature-score
              pointer config out-n-features out-features out-dim out-shape out-scores)
             "XGBoosterFeatureScore"))

;;; ---------------------------------------------------------------------------
;;; Evaluation
;;;
;;; `evaluate-one-iteration' below is this file's first export that is not a `%'-prefixed internal
;;; helper: Task 3 (policy section 3's Layer 1) exports it directly from `cl-gbdt/xgboost'
;;; -- see `src/xgboost/all.lisp' -- rather than through a `cl-gbdt/src/xgboost/protocol'
;;; `defmethod' the way every other public operation on a booster is reached. See this
;;; file's own "Floating-point trap safety" header comment for what that changes about the
;;; mask, and `cl-gbdt/src/lightgbm/native''s identical "Evaluation" section for LightGBM's
;;; own version of this same Layer 1 addition -- the two differ here exactly as much as
;;; `LGBM_BoosterGetEval' and `XGBoosterEvalOneIter' themselves do; see
;;; docs/superpowers/specs/2026-08-06-evaluation-api-design.md section 2 for the asymmetry
;;; both were built against.

(defun %check-xgboost-booster (booster argument-description)
  "Return BOOSTER's live foreign pointer, after confirming BOOSTER is a booster built by
the `:xgboost' backend.

Thin wrapper over `cl-gbdt/src/handle''s `%check-handle-kind', which carries the contract,
the conditions and the reason the check cannot be spelled as `(typep booster
'xgboost-booster)' here. ARGUMENT-DESCRIPTION names which caller-supplied argument BOOSTER
came from, for `wrong-backend-reference''s report. Mirrors
`cl-gbdt/src/lightgbm/native''s `%check-lightgbm-booster'."
  (%check-handle-kind booster 'booster :xgboost argument-description))

(defun %check-xgboost-eval-dataset (dataset argument-description)
  "Return DATASET's live foreign pointer, after confirming DATASET is a dataset built by
the `:xgboost' backend.

`evaluate-one-iteration''s DATASETS argument is a list of caller-supplied handles this file
must check before handing their pointers to `XGBoosterEvalOneIter'. The existing
`%check-xgboost-dataset' cannot be reused: both of its callers --
`cl-gbdt/src/xgboost/protocol''s `train' and `cl-gbdt/src/xgboost/api''s `create-booster'
-- already hold the concrete `xgboost-dataset' class symbol to pass it, and
`evaluate-one-iteration' has no such caller to get one from.

Thin wrapper over `cl-gbdt/src/handle''s `%check-handle-kind', which carries the contract
and the conditions. ARGUMENT-DESCRIPTION names which caller-supplied argument DATASET came
from, for `wrong-backend-reference''s report."
  (%check-handle-kind dataset 'dataset :xgboost argument-description))

(defun %eval-one-iter (booster-pointer iteration dmatrix-pointers names)
  "Run `XGBoosterEvalOneIter' over the booster at BOOSTER-POINTER, at ITERATION, against
DMATRIX-POINTERS, each labeled by the corresponding entry of NAMES, and return the exact
string it wrote to `out_result' -- e.g. \"[5]\\ttrain-logloss:0.1\\tvalid-logloss:0.2\".

DMATRIX-POINTERS and NAMES must be the same length -- `XGBoosterEvalOneIter''s own
`dmats' and `evnames' arrays, one caller-chosen name per DMatrix to evaluate. Checked
here, before either foreign array is built: NAMES shorter than DMATRIX-POINTERS would
otherwise leave trailing slots of the `evnames' array pointing at whatever garbage
`cffi:with-foreign-object' happened to allocate, which `XGBoosterEvalOneIter' would then
dereference as a C string -- a segfault or worse, not a catchable condition, exactly the
hazard this file's every other caller-supplied-array construction (`%set-feature-names',
`%create-booster') already checks its own inputs against before allocating. Signals
`dimension-mismatch' instead.

An empty DMATRIX-POINTERS -- `evaluate-one-iteration' called with no datasets to evaluate -- passes
a null pointer and length 0 to `XGBoosterEvalOneIter' rather than a zero-length foreign
array, mirroring `%create-booster''s identical choice for the same reason: to avoid
depending on whether a zero-count `cffi:with-foreign-object' allocation is meaningful.

Every string successfully allocated for NAMES is freed on any exit, including one
signaled partway through the allocation loop itself, matching `%set-feature-names''s
identical cleanup discipline.

`out_result' points into memory XGBoost itself owns, valid only until the next call into
this booster; this copies its contents into a fresh Lisp string before returning, so the
caller never holds a pointer into XGBoost's own buffer."
  (let ((count (length dmatrix-pointers)))
    (unless (= count (length names))
      (error 'dimension-mismatch
             :expected (format nil "~D name~:P, one per dataset" count)
             :given (format nil "~D name~:P" (length names))))
    (flet ((call (dmats evnames)
             (cffi:with-foreign-object (out-result :pointer)
               (check-xgb (xg-booster-eval-one-iter
                           booster-pointer iteration dmats evnames count out-result)
                          "XGBoosterEvalOneIter")
               (let ((result-pointer (cffi:mem-ref out-result :pointer)))
                 (if (cffi:null-pointer-p result-pointer)
                     ""
                     (cffi:foreign-string-to-lisp result-pointer))))))
      (if (zerop count)
          (call (cffi:null-pointer) (cffi:null-pointer))
          (cffi:with-foreign-object (dmats :pointer count)
            (loop :for pointer :in dmatrix-pointers
                  :for index :from 0
                  :do (setf (cffi:mem-aref dmats :pointer index) pointer))
            (let ((allocated 0))
              (cffi:with-foreign-object (evnames :pointer count)
                (unwind-protect
                     (progn
                       (loop :for name :in names
                             :for index :from 0
                             :do (setf (cffi:mem-aref evnames :pointer index)
                                       (cffi:foreign-string-alloc name))
                                 (setf allocated (1+ index)))
                       (call dmats evnames))
                  (dotimes (index allocated)
                    (cffi:foreign-string-free (cffi:mem-aref evnames :pointer index)))))))))))

(defparameter *%eval-value-reader-package*
  (or (find-package '#:cl-gbdt/src/xgboost/native/eval-value-scratch)
      (make-package '#:cl-gbdt/src/xgboost/native/eval-value-scratch :use nil))
  "Throwaway package used only as `*package*' while `%read-eval-value' reads a numeric
token out of `XGBoosterEvalOneIter''s raw result string.

A token that is not valid `double-float' syntax -- XGBoost's own `\"inf\"', `\"-inf\"',
`\"nan\"' or `\"-nan\"' spellings for a non-finite value included -- still reads as a
plain symbol rather than signalling on the read itself, and that symbol has to be
interned *somewhere*. Left unbound, `*package*' would be whatever was current when
`evaluate-one-iteration' was called -- the caller's own package -- so `read-from-string' would
silently leave `INF' or `NAN' newly accessible there, a runtime `intern' as a side effect
of parsing a result string, which this project's own style guide forbids. Binding it to
this throwaway package instead confines every such symbol here, never touching the
caller's. Mirrors `tools/ci/check-float-traps.lisp''s identical scratch-package idiom,
for the identical reason.")

(defun %read-eval-value (text)
  "Return TEXT -- one `%parse-eval-result' field's value substring -- read as a
`double-float', or NIL when TEXT is not valid `double-float' syntax.

XGBoost formats a metric value through `std::ostream', which spells a non-finite double
as `\"inf\"', `\"-inf\"', `\"nan\"' or `\"-nan\"' -- confirmed reachable from real
objectives, e.g. `mape' on a zero label or `rmsle' on a negative prediction. None of those
four are `double-float' syntax: each reads as a plain symbol instead of signalling on the
read itself, so `realp' is what actually rejects them here, after a successful read, not
the `handler-case' below. TEXT = \"\" -- an empty field, e.g. a trailing tab with nothing
after it -- does signal instead, `end-of-file' from `read-from-string' with nothing left
to read; `handler-case' below catches `error' generally, not only that one specific
condition, matching this project's own established idiom for a best-effort operation
whose exact failure mode is not the point -- see e.g. `cl-gbdt/src/xgboost/protocol''s
`make-dataset' and `free-dataset', which catch `error' the same way around
`%free-dmatrix-unchecked'/`%free-dmatrix'. TEXT is arbitrary text from a raw result
string this function does not otherwise control the shape of, so a reader condition this
docstring has not enumerated is exactly the case this stays broad for. Either way this
returns NIL rather than signalling out of `%parse-eval-result' and, through it,
`evaluate-one-iteration' -- see that function's own docstring for why losing RAW over a value this
function cannot make sense of is not acceptable.

`*read-eval*' is bound to NIL so a stray `#.' in an unexpected metric string cannot run
code, and `*package*' to *%eval-value-reader-package*, never the caller's own -- see that
parameter's docstring for why. A `let', not a `let*': none of these three bindings'
init-forms refers to another of them -- their dependency is dynamic, not lexical, which is
exactly what `let' already provides here: every one of the three is in effect for the
whole BODY below, including the nested `let' that reads TEXT, which is all this needs."
  (let ((*read-eval* nil)
        (*read-default-float-format* 'double-float)
        (*package* *%eval-value-reader-package*))
    (let ((value (handler-case (read-from-string text)
                   (error () nil))))
      (when (realp value)
        (coerce value 'double-float)))))

(defun %parse-eval-result (raw)
  "Parse RAW -- `XGBoosterEvalOneIter''s own result string, as `%eval-one-iter' returns
it unmodified -- into a fresh list of (LABEL . VALUE) conses, `evaluate-one-iteration''s own
derived interpretation of an undocumented text format, never a substitute for RAW itself.

Confirmed directly against the vendored library: RAW looks like
\"[5]\\ttrain-logloss:0.1\\tvalid-logloss:0.2\", a leading \"[iteration]\" marker, then
one tab-separated field per (dataset, metric) pair, each field a LABEL and a VALUE joined
by a colon. This splits on tab, drops the leading marker, and for each remaining field
splits on that field's *last* colon -- VALUE is always numeric text with no colon of its
own, so the last colon in the field is always the LABEL/VALUE boundary regardless of what
LABEL itself contains.

LABEL is returned exactly as XGBoost wrote it, e.g. \"train-logloss\" -- the
caller-supplied name (`evaluate-one-iteration''s own NAMES argument) and XGBoost's own metric name,
joined with a hyphen. This does not attempt to split LABEL further into the two: a hyphen
is also legal inside a caller-supplied name, so nothing in RAW alone can tell the two
apart -- only `evaluate-one-iteration''s own caller, which already knows every entry of NAMES
verbatim, can. VALUE is `%read-eval-value''s result: a `double-float' for ordinary numeric
text, or NIL when that text is not valid `double-float' syntax -- XGBoost's own
`\"inf\"'/`\"-inf\"'/`\"nan\"'/`\"-nan\"' spellings for a non-finite metric value are the
known case, confirmed reachable from real objectives (see `%read-eval-value'). A NIL VALUE
still keeps its LABEL and its place in the result list -- this never drops a field outright
just because its own value did not parse; RAW is what to consult for its literal text.

Returns NIL when RAW has no fields after the marker -- an empty DMATRIX-POINTERS reaching
`%eval-one-iter', so XGBoost had nothing to evaluate -- or when RAW itself is empty."
  (let ((fields (uiop:split-string raw :separator '(#\Tab))))
    (loop :for field :in (rest fields)
          :for colon := (position #\: field :from-end t)
          :when colon
            :collect (cons (subseq field 0 colon)
                            (%read-eval-value (subseq field (1+ colon)))))))

(defun %split-eval-label (label names)
  "Return (VALUES INDEX METRIC-NAME) for LABEL, one `%parse-eval-result' entry's label,
given NAMES -- the exact list of dataset names that was passed to `evaluate-one-iteration' for the
call LABEL came out of. INDEX is LABEL's dataset's position in NAMES; METRIC-NAME is
what is left of LABEL after that name and the hyphen joining it to XGBoost's own metric
name.

`%parse-eval-result' deliberately does not do this itself -- see its docstring: a hyphen
is legal inside a caller-supplied name too, so nothing in the raw string alone can tell
the two halves apart, and only a caller that knows NAMES verbatim can. This is that
caller's half of the split, kept here beside the parser rather than in
`cl-gbdt/src/xgboost/protocol' so that all of this backend's knowledge of
`XGBoosterEvalOneIter''s undocumented text format lives in one file.

The match is on the whole `\"NAME-\"' prefix, not on NAME alone, and no two distinct
entries of NAMES can both match one LABEL that way: if `\"A-\"' and `\"B-\"' both
prefixed it, one of A and B would have to be the other followed by a hyphen, which
`evaluation''s own decimal-index names never are.

Signals `foreign-call-error' when no entry of NAMES prefixes LABEL that way, rather than
guessing an index or dropping the field: a label that cannot be attributed to a dataset
would mean `XGBoosterEvalOneIter' no longer builds its labels the way this file assumes,
and reporting a metric against the wrong dataset silently is exactly the failure this
project treats as most serious. Mirrors `%feature-score-index''s identical refusal to
guess when `XGBoosterFeatureScore' reports a feature name outside the form it documents."
  (loop :for name :in names
        :for index :from 0
        :for prefix := (concatenate 'string name "-")
        :when (and (> (length label) (length prefix))
                   (string= prefix label :end2 (length prefix)))
          :do (return (values index (subseq label (length prefix))))
        :finally (error 'foreign-call-error
                        :function-name "XGBoosterEvalOneIter"
                        :code 0
                        :message (format nil "result label ~S matches none of the dataset ~
                                              names ~S this call supplied" label names))))

(defun evaluate-one-iteration (booster datasets names)
  "Evaluate BOOSTER against DATASETS via `XGBoosterEvalOneIter', labeling each entry of
DATASETS with the corresponding entry of NAMES, at BOOSTER's own current boosted-round
count (read fresh via `XGBoosterBoostedRounds', the same source `update-one-iteration'
reads its own `iter' argument from).

Unlike LightGBM's `booster-eval', which reads the validation sets `train' already
attached to BOOSTER by index, XGBoost's `XGBoosterEvalOneIter' evaluates whatever
DMatrices the caller passes here and does not consult BOOSTER's own `:valid-sets' at all
-- this is the asymmetry that
docs/superpowers/specs/2026-08-06-evaluation-api-design.md section 2 documents between
the two libraries' C APIs; a portable Layer 2 method built on top of this still has to
decide what to pass as DATASETS, which is exactly what that design leaves for the layer
above this one to settle, not this function.

Returns two values:

  RAW    The exact string `XGBoosterEvalOneIter' wrote to `out_result', e.g.
         \"[5]\\ttrain-logloss:0.1\\tvalid-logloss:0.2\". This is the only value this
         function can promise is faithful to what XGBoost actually produced -- its
         format is not documented as stable. RAW is authoritative; PARSED below is
         derived from it and must never be substituted for it.

  PARSED A fresh list of (LABEL . VALUE) conses, this function's own interpretation of
         RAW -- see `%parse-eval-result' for exactly how, and why it does not attempt to
         split LABEL back into a dataset name and a metric name. VALUE is a `double-float'
         for ordinary numeric text, or NIL when XGBoost wrote something `%read-eval-value'
         cannot read as one -- its own `\"inf\"'/`\"nan\"' spellings for a non-finite
         metric are the known case; the LABEL stays regardless, only VALUE goes missing.
         PARSED is empty when DATASETS is empty. A field PARSED cannot make sense of never
         costs RAW: this always returns RAW once the foreign call itself has succeeded, no
         matter what PARSED does with it afterward. If RAW and PARSED ever disagree, RAW is
         what XGBoost said and PARSED is a bug in this function, not the other way around.

DATASETS and NAMES must be the same length, checked in `%eval-one-iter' before either
foreign array is built; see that function's docstring for the hazard an unchecked
mismatch would reach. Every entry of DATASETS, and BOOSTER itself, is read through
`handle-live-pointer' and confirmed built by the `:xgboost' backend before any foreign
call -- passing a released or wrong-backend handle straight to `XGBoosterEvalOneIter'
would be a segfault, not a catchable condition, exactly as for every other
caller-supplied handle in this backend. Signals `wrong-backend-reference' when BOOSTER or
any DATASETS entry was not built by `:xgboost', `released-handle-error' for an
already-freed one, `backend-not-open' when its own backend has since been closed, and
`dimension-mismatch' when DATASETS and NAMES differ in length."
  (with-foreign-float-traps-masked
    (let* ((booster-pointer
             (%check-xgboost-booster booster "evaluate-one-iteration's booster argument"))
           (dataset-pointers
             (loop :for dataset :in datasets
                   :for index :from 0
                   :collect (%check-xgboost-eval-dataset
                             dataset
                             (format nil "evaluate-one-iteration's datasets entry ~D" index))))
           (raw (%eval-one-iter booster-pointer (%boosted-rounds booster-pointer)
                                 dataset-pointers names)))
      (values raw (%parse-eval-result raw)))))

(defun %read-evaluation (booster-pointer dataset-pointers)
  "Return the booster at BOOSTER-POINTER's evaluation entries against DATASET-POINTERS, as a
fresh list of (INDEX METRIC-NAME VALUE) lists, via `%eval-one-iter', `%parse-eval-result' and
`%split-eval-label' -- INDEX is a DATASET-POINTERS entry's position, METRIC-NAME and VALUE
are `%parse-eval-result''s own reading of `XGBoosterEvalOneIter''s formatted output for it.
Names each entry of DATASET-POINTERS to `%eval-one-iter' by its own decimal index -- \"0\",
\"1\", ... -- built from `(length dataset-pointers)' alone, exactly as
`cl-gbdt/src/xgboost/api''s `evaluation' describes doing; `%split-eval-label' takes each
result label back apart against those same names.

Also returns RAW, the exact string `%eval-one-iter' wrote, as a second value: `evaluation''s
own `:value-source :parsed-text :raw' provenance needs it verbatim, and once this function
returns, this is the only place that string still exists.

The caller owns every guard this needs before calling: BOOSTER-POINTER and every entry of
DATASET-POINTERS must already be live handles' pointers built by the `:xgboost' backend, and
the whole call must already be inside `with-foreign-float-traps-masked''s dynamic extent --
like every other `%'-function in this file, this does not establish any of those itself.
`cl-gbdt/src/xgboost/api''s `evaluation' -- what `cl-gbdt/src/xgboost/protocol''s method of
the same name now delegates to wholly -- and `train''s per-iteration recording loop both call
this same function, on the pointers each already has in hand, rather than each computing
entries its own way -- that is what keeps the numbers `evaluation' reports after training and
the numbers recorded during training from ever being able to disagree."
  (let* (;; `~D', not `princ-to-string': `~D' binds `*print-base*' to 10 itself, so a
         ;; caller who has bound it to something else gets the decimal names this
         ;; function's docstring promises rather than that base's digits.
         (names (loop :for index :below (length dataset-pointers)
                       :collect (format nil "~D" index)))
         (raw (%eval-one-iter booster-pointer (%boosted-rounds booster-pointer)
                               dataset-pointers names)))
    (values (loop :for (label . value) :in (%parse-eval-result raw)
                  :collect (multiple-value-bind (index metric-name)
                               (%split-eval-label label names)
                             (list index metric-name value)))
            raw)))

;;; ---------------------------------------------------------------------------
;;; Model slicing
;;;
;;; `XGBoosterSlice' is this project's first OPTIONAL foreign symbol: policy section 8's
;;; tier whose absence disables one capability rather than failing `open-backend'. It is
;;; declared in `*optional-symbols*' above, probed into the backend's `backend-capabilities'
;;; at `open-backend', and answered by `backend-supports-p' as `:model-slicing'. LightGBM has
;;; no counterpart, which is exactly why section 4's criterion keeps this out of the unified
;;; API: there is no operation both backends can mean the same thing by, and emulating one on
;;; LightGBM is the silent fallback section 7 forbids.
;;;
;;; The public operation itself, `slice-model', is NOT here. `XGBoosterSlice' produces a new
;;; `BoosterHandle', and wrapping a fresh foreign pointer in a handle means naming the
;;; concrete class `xgboost-booster', which is defined in `cl-gbdt/src/xgboost/classes' --
;;; a file this one cannot depend on, since that one already depends on this one for
;;; `*required-symbols*' and the library-discovery parameters, so the edge would close a
;;; cycle. Every `make-handle' call in this project is therefore in a file that loads after
;;; `native.lisp': `slice-model' sits in `api.lisp' -- Layer 1 still, since it is XGBoost's
;;; own operation and not part of the unified API -- and calls `%slice' below for the foreign
;;; half, exactly as `load-model' in `protocol.lisp' calls `%load-model'. It lived in
;;; `classes.lisp' until `api.lisp' existed, that having been the only Layer 1 file after this
;;; one; it moved because it is an operation over the booster class rather than part of the
;;; shared library's lifetime, and the constraint above admits either file equally.
;;; `booster-boosted-rounds' has no such constraint -- it
;;; returns an integer -- so it stays here beside the `%boosted-rounds' it wraps.

(defun booster-boosted-rounds (booster)
  "Return the number of boosting rounds BOOSTER holds, via `XGBoosterBoostedRounds'.

This is the count `slice-model''s BEGIN and END are indices into, and the only way for a
caller to learn the valid range before asking for a slice: `XGBoosterSlice' signals
`foreign-call-error' rather than clamping when END exceeds this number. It is also the
round index `update-one-iteration' boosts at, read from the same C function, so a booster
`train' ran for N rounds reports N here.

Takes a booster handle, not a pointer, and reads it through `%check-xgboost-booster' --
policy section 10: a Layer 1 entry point a caller reaches directly must validate the handle
it is given, since no `defmethod' specializer has ruled out a foreign one first.

Signals `wrong-backend-reference' when BOOSTER was not built by the XGBoost backend,
`released-handle-error' when it has been freed, `backend-not-open' when its backend has
been closed, and `foreign-call-error' when the call itself fails."
  (with-foreign-float-traps-masked
    (%boosted-rounds
     (%check-xgboost-booster booster "booster-boosted-rounds's booster argument"))))

(defun %slice (booster-pointer begin end step)
  "Return a fresh `BoosterHandle' holding BOOSTER-POINTER's layers from BEGIN to END taken
STEP at a time, via `XGBoosterSlice'.

BEGIN, END and STEP are passed to C unchanged; this function translates nothing. END is
`XGBoosterSlice''s own `end_layer', where 0 means \"through the last layer\" -- see
`slice-model', which owns turning a Lisp caller's NIL into that 0 and refusing to turn an
explicit 0 into it.

Returns a raw pointer the caller must wrap in a handle or free; `slice-model' is that
caller. Shaped like `%create-booster' rather than `%load-model' for the same reason: the
foreign object it returns does not exist until it succeeds, so there is nothing to clean up
on the failure path.

Signals `foreign-call-error' when `XGBoosterSlice' fails -- which is every out-of-range
request, confirmed against the vendored library: END past the last layer, BEGIN below zero,
STEP below one, an empty interval (\"Empty slice is not allowed\"), and a STEP that does not
divide the interval evenly are all clean nonzero returns, never a crash -- or when it
reports success while leaving a null handle behind, the same refusal to trust a
success-with-null that `%create-booster' makes for `XGBoosterCreate'.

The vendored header (`ffi-spec/xgboost/include/xgboost/c_api.h') documents
`XGBoosterSlice''s return value as \"0 when success, -1 when failure happens, -2 when
index is out of bound\", and -2 gets special handling here: confirmed against the
vendored library, that path is taken before upstream's usual exception-to-`XGBGetLastError'
plumbing runs, so the last-error buffer is left holding whatever an earlier, unrelated
failure wrote in this process, or empty when nothing has failed yet -- either way not a
description of BEGIN/END/STEP actually being out of bounds. Every other nonzero code (-1)
does go through that plumbing and gets `check-xgb''s ordinary handling, since upstream's
own message is accurate there. Code -2 instead signals `foreign-call-error' with a message
this function builds itself, naming BEGIN, END and STEP -- the most likely mistake a
`slice-model' caller makes, per that function's own docstring, and, before this fix, the
one this backend reported most misleadingly.

That guard is a little stronger here than `%create-booster''s: the out slot is set to a
null pointer before the call, so it also catches a `XGBoosterSlice' that returned success
without writing the slot at all, which a bare `cffi:with-foreign-object' would leave holding
whatever was on the stack and read back as a plausible-looking handle. Not retrofitted to
`%create-booster' as part of this change -- that would be an unrelated edit to a function
this work does not otherwise touch."
  (cffi:with-foreign-object (out :pointer)
    (setf (cffi:mem-ref out :pointer) (cffi:null-pointer))
    (let ((code (xg-booster-slice booster-pointer begin end step out)))
      (if (= code -2)
          (error 'foreign-call-error
                 :function-name "XGBoosterSlice"
                 :code code
                 :message (format nil "begin ~D, end ~D, step ~D names a layer range out ~
                                        of bounds for this booster -- XGBoosterSlice ~
                                        reports this case (code -2) without setting ~
                                        XGBGetLastError; see booster-boosted-rounds for ~
                                        the valid range"
                                   begin end step))
          (check-xgb code "XGBoosterSlice")))
    (let ((slice-pointer (cffi:mem-ref out :pointer)))
      (when (cffi:null-pointer-p slice-pointer)
        (error 'foreign-call-error
               :function-name "XGBoosterSlice"
               :code 0
               :message "reported success but returned a null booster handle"))
      slice-pointer)))
