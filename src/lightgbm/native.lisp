;;;; native.lisp --- LightGBM backend, Layer 1: library discovery, the error wrapper, and
;;;; every internal function that turns a raw call into lib_lightgbm.so into something
;;;; safe to call from Lisp -- out parameters into return values, error codes into
;;;; conditions, raw pointers accepted only where a caller already validated them.
;;;;
;;;; Nothing here is a CLOS protocol method and nothing here depends on
;;;; `cl-gbdt/src/lightgbm/protocol' -- see that file for the fifteen `defmethod's this
;;;; module exists to support. `cl-gbdt/src/lightgbm/protocol''s own docstrings, not this
;;;; file's, are the place each function's role in the unified API is explained; the
;;;; docstrings below describe only what each function does to the C API.

(uiop:define-package #:cl-gbdt/src/lightgbm/native
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/lightgbm/c-api
                #:lgbm-get-last-error
                #:lgbm-dataset-create-from-mat
                #:lgbm-dataset-create-from-csr
                #:lgbm-dataset-set-field
                #:lgbm-dataset-set-feature-names
                #:lgbm-dataset-free
                #:lgbm-dataset-get-num-data
                #:lgbm-dataset-get-num-feature
                #:lgbm-booster-create
                #:lgbm-booster-add-valid-data
                #:lgbm-booster-update-one-iter
                #:lgbm-booster-update-one-iter-custom
                #:lgbm-booster-get-num-classes
                #:lgbm-booster-get-num-predict
                #:lgbm-booster-get-predict
                #:lgbm-booster-free
                #:lgbm-booster-calc-num-predict
                #:lgbm-booster-predict-for-mat
                #:lgbm-booster-predict-for-csr
                #:lgbm-booster-save-model
                #:lgbm-booster-save-model-to-string
                #:lgbm-booster-create-from-modelfile
                #:lgbm-booster-feature-importance
                #:lgbm-booster-get-num-feature
                #:lgbm-booster-get-eval-counts
                #:lgbm-booster-get-eval-names
                #:lgbm-booster-get-eval
                #:+c-api-dtype-float32+
                #:+c-api-dtype-float64+
                #:+c-api-dtype-int32+
                #:+c-api-predict-normal+
                #:+c-api-predict-raw-score+
                #:+c-api-predict-leaf-index+
                #:+c-api-predict-contrib+
                #:+c-api-feature-importance-split+
                #:+c-api-feature-importance-gain+)
  (:import-from #:cl-gbdt/src/backend
                #:backend-name
                #:backend-open-p)
  (:import-from #:cl-gbdt/src/handle
                #:%check-handle-kind
                #:handle-live-pointer
                #:handle-released-p
                #:booster
                #:booster-training-set
                #:booster-validation-sets)
  (:import-from #:cl-gbdt/src/conditions
                #:backend-not-open
                #:foreign-call-error
                #:released-handle-error
                #:wrong-backend-reference)
  (:import-from #:cl-gbdt/src/parameters
                #:normalize-parameters)
  (:import-from #:cl-gbdt/src/config/feature-names
                #:check-feature-names)
  (:import-from #:cl-gbdt/src/config/objective
                #:objective-single-float)
  (:import-from #:cl-gbdt/src/data
                #:with-foreign-matrix
                #:write-foreign-sequence)
  (:import-from #:cl-gbdt/src/foreign
                #:check-foreign-call
                #:with-foreign-float-traps-masked)
  (:export #:check-lgbm
           #:*library-env-var*
           #:*vendor-library-directory*
           #:*vendor-library-pattern*
           #:*default-library-name*
           #:*required-symbols*
           #:*optional-symbols*
           #:*provided-capabilities*
           #:%check-backend-open
           #:%check-lightgbm-dataset
           #:%reference-pointer
           #:%parameter-string
           #:%data-type
           #:%create-dataset
           #:%create-dataset-from-csr
           #:%set-info-field
           #:%set-group-field
           #:%set-feature-names
           #:%free-dataset-unchecked
           #:%dataset-num-rows
           #:%dataset-num-features
           #:%free-dataset
           #:%create-booster
           #:%add-valid-data
           #:%update-one-iteration
           #:%booster-num-classes
           #:%booster-predictions
           #:%update-one-iteration-custom
           #:%check-booster-datasets-live
           #:%free-booster-unchecked
           #:%free-booster
           #:%predict-type
           #:%resolve-num-iteration
           #:%calc-num-predict
           #:%predict-ncol
           #:%predict-for-mat
           #:%predict-for-csr
           #:%save-model
           #:%create-booster-from-modelfile
           #:%save-model-to-string
           #:%feature-importance-type
           #:%booster-num-features
           #:%feature-importance
           #:%read-evaluation
           #:booster-eval-names
           #:booster-eval))

(in-package #:cl-gbdt/src/lightgbm/native)

;;; ---------------------------------------------------------------------------
;;; Floating-point trap safety
;;;
;;; Almost nothing here wraps itself in `with-foreign-float-traps-masked' -- almost every
;;; function in this file is only ever called from inside a `cl-gbdt/src/lightgbm/protocol'
;;; `defmethod' body that already established the mask, at method-body granularity, before
;;; making any of the calls below. See that file's identical commentary, and
;;; `with-foreign-float-traps-masked''s own docstring in `cl-gbdt/src/foreign', for why:
;;; SBCL enables floating-point traps by default on x86-64 and not on aarch64, and
;;; `cl-gbdt/src/xgboost/native''s identical commentary describes the concrete case
;;; (XGBoost's `multi:softprob' softmax) that surfaced this. LightGBM has not tripped it
;;; yet, but its C API is exactly as unprotected against SBCL's x86-64 trap defaults.
;;;
;;; `booster-eval-names' and `booster-eval', in the Evaluation section near the end of this
;;; file, are the exception: Phase 2 (policy section 3's Layer 1) exports both directly from
;;; `cl-gbdt/lightgbm' -- see `src/lightgbm/all.lisp' -- so neither is ever reached through a
;;; `cl-gbdt/src/lightgbm/protocol' `defmethod'. There is no outer call site left to establish
;;; the mask for them, so each wraps its own whole body instead, the same method-body
;;; granularity `protocol.lisp' uses, just with the wrapping function living here rather than
;;; there. `tools/ci/check-float-traps.lisp' does not scan this file for `defmethod' forms
;;; the way it does `protocol.lisp' -- see that script's own header -- so this pair is not
;;; machine-checked; verified by hand instead, alongside every other new export this file
;;; ever adds directly to a public package.

;;; ---------------------------------------------------------------------------
;;; Error checking

(defun %last-error-message ()
  "Return LightGBM's last error message as a Lisp string, or NIL when
`LGBM_GetLastError' returns a null pointer."
  (let ((pointer (lgbm-get-last-error)))
    (unless (cffi:null-pointer-p pointer)
      (cffi:foreign-string-to-lisp pointer))))

(defun check-lgbm (code function-name)
  "Signal `foreign-call-error' when CODE reports failure, otherwise return CODE.

LightGBM returns 0 on success and a nonzero status -- documented as -1 -- on
failure, with the detail available from `LGBM_GetLastError'. FUNCTION-NAME
identifies which C function reported CODE, for the condition's report. Every
foreign call this backend makes goes through this: the functional tests found
five `LGBM_BoosterUpdateOneIter' calls all returning 0 while the model did not
train, so a status code alone is necessary but not sufficient -- this is the
part that is sufficient to check here.

Thin wrapper around `check-foreign-call' supplying `%last-error-message' as
LightGBM's last-error thunk -- see that function's docstring for the shared
0-on-success idiom both backends follow. Kept as a named wrapper, rather than
calling `check-foreign-call' directly at each of this file's call sites, so
none of them needed editing when the check itself moved to
`cl-gbdt/src/foreign'."
  (check-foreign-call code function-name #'%last-error-message))

(defun %check-backend-open (backend)
  "Signal `backend-not-open' when BACKEND is not open.

`make-dataset', `train' and `load-model' each create a brand-new handle
directly from BACKEND -- there is no existing handle for the check to route
through the way `handle-live-pointer' does for every other operation in this
file, since none exists yet. Each of them calls this first, before touching
any foreign function, so a backend a caller has closed (or never opened) is
never reached by `LGBM_DatasetCreateFromMat', `LGBM_BoosterCreate' or
`LGBM_BoosterCreateFromModelfile' with a library that may no longer be mapped."
  (unless (backend-open-p backend)
    (error 'backend-not-open :backend (backend-name backend))))

(defun %check-lightgbm-dataset (backend dataset argument-description dataset-class)
  "Return DATASET's live foreign pointer, after confirming DATASET is of type
DATASET-CLASS -- `cl-gbdt/src/lightgbm/protocol''s `lightgbm-dataset'.
ARGUMENT-DESCRIPTION names which caller-supplied argument DATASET came from --
e.g. \"train's dataset argument\" -- for `wrong-backend-reference''s report.

DATASET-CLASS is a parameter, not a symbol named directly in this file, because
this file must not depend on `cl-gbdt/src/lightgbm/protocol' -- see policy
section 11 and this file's own header -- and `lightgbm-dataset' is defined
there. Each caller -- `%reference-pointer', and `cl-gbdt/src/lightgbm/protocol''s
`train' at both of its own call sites -- passes `'lightgbm-dataset' explicitly
instead. Mirrors `cl-gbdt/src/xgboost/native''s `%check-xgboost-dataset', which
hit the identical constraint during that backend's own Phase 1 split.

Every caller-supplied dataset argument in this file -- `make-dataset''s
:REFERENCE, `train''s DATASET, and each entry of `train''s :VALID-SETS -- must
pass through here before reaching a foreign call that expects a
`DatasetHandle'. `handle-live-pointer' alone is not enough: it only guards
against a released handle or a closed backend, and happily returns *any*
handle's pointer regardless of kind, including a booster's -- `make-dataset'
and `train' both dispatch on the backend, not on the handle, so unlike
`dataset-num-rows' or `free-dataset' there is no CLOS specializer already
ruling out the wrong kind of handle. A booster's own pointer reaching
`LGBM_BoosterCreate' as its training-set argument is exactly the corruption
this check exists to prevent.

Signals `wrong-backend-reference' when DATASET is not of type DATASET-CLASS --
built by a different backend, or not a dataset at all -- and whatever
`handle-live-pointer' signals otherwise: `released-handle-error' for an
already-freed DATASET, `backend-not-open' when DATASET's own backend has since
been closed.

This does not additionally check that DATASET was built by BACKEND
specifically, only that it is of type DATASET-CLASS. Two `lightgbm-backend'
instances over the same shared library are a legitimate way for a caller to
hold datasets from -- both read through the same C API, so rejecting a
dataset built by one while training on the other would break working code for
no safety gain. The dangerous case, a handle whose own backend has been
closed, is already caught above, by `handle-live-pointer''s `backend-open-p'
check on DATASET's own backend, not BACKEND."
  (unless (typep dataset dataset-class)
    (error 'wrong-backend-reference
           :backend (backend-name backend)
           :given (class-name (class-of dataset))
           :argument argument-description))
  (handle-live-pointer dataset))

;;; ---------------------------------------------------------------------------
;;; Library discovery

(defparameter *library-env-var* "CL_GBDT_LIGHTGBM_LIB"
  "Environment variable overriding LightGBM's shared-library discovery.")

(defparameter *vendor-library-directory* "vendor/lightgbm/lib/"
  "Repository-relative directory `tools/fetch-libs.sh' writes LightGBM's shared
library to.")

(defparameter *vendor-library-pattern* "lib_lightgbm.*"
  "Basename pattern for LightGBM's shared library within *vendor-library-directory*.
The extension stays wild because `tools/fetch-libs.sh' preserves whatever the
platform's wheel ships -- `.so' on Linux, `.dylib' on macOS -- and discovery must
not assume either.")

(defparameter *default-library-name* "lib_lightgbm"
  "Name passed to CFFI's own system library search when no other candidate is found.

CFFI's `:default' designator only appends the platform's shared-library suffix (`.so' on
Linux) to this string -- it never adds a `lib' prefix, the same as
`cl-gbdt/src/xgboost/native''s identical constant, whose docstring carries the full
measurement. LightGBM's compiled basename is `_lightgbm', but the file on disk is
`lib_lightgbm.so', so this constant has to spell out `lib_lightgbm' in full or CFFI's
system search never finds it.")

(defparameter *required-symbols*
  '("LGBM_GetLastError"
    "LGBM_DatasetCreateFromMat"
    "LGBM_DatasetSetField"
    "LGBM_DatasetSetFeatureNames"
    "LGBM_DatasetFree"
    "LGBM_DatasetGetNumData"
    "LGBM_DatasetGetNumFeature"
    "LGBM_BoosterCreate"
    "LGBM_BoosterAddValidData"
    "LGBM_BoosterUpdateOneIter"
    "LGBM_BoosterFree"
    "LGBM_BoosterCalcNumPredict"
    "LGBM_BoosterPredictForMat"
    "LGBM_BoosterSaveModel"
    "LGBM_BoosterSaveModelToString"
    "LGBM_BoosterCreateFromModelfile"
    "LGBM_BoosterFeatureImportance"
    "LGBM_BoosterGetNumFeature"
    "LGBM_BoosterGetEvalCounts"
    "LGBM_BoosterGetEvalNames"
    "LGBM_BoosterGetEval")
  "C function names this backend calls, checked with `probe-foreign-symbols' right
after the library loads.")

(defparameter *optional-symbols*
  '((:sparse-input "LGBM_DatasetCreateFromCSR" "LGBM_BoosterPredictForCSR")
    (:custom-objective "LGBM_BoosterUpdateOneIterCustom" "LGBM_BoosterGetPredict"
     "LGBM_BoosterGetNumPredict" "LGBM_BoosterGetNumClasses")
    (:custom-evaluation "LGBM_BoosterGetPredict" "LGBM_BoosterGetNumPredict"
     "LGBM_BoosterGetNumClasses"))
  "Capability name to the C function names that capability needs.

Unlike `*required-symbols*', whose absence makes `open-backend' signal
`missing-foreign-symbols', a name missing from here disables one capability and nothing else
-- policy section 8. LightGBM having no counterpart to `XGBoosterSlice' is why
`(backend-supports-p backend :model-slicing)' is false here -- which is the case the whole
capability model is tested against; the entry above is what makes `:sparse-input' true.

`:sparse-input' names both the ingestion entry point and the prediction one, even though
`make-dataset' reaches only the first. The capability is one answer about whether this
backend can take a `csr-matrix' at all, and a caller told \"yes\" who could build a dataset
but not predict from one would have been told a half-truth. Listing both from the start means
the answer never changes meaning as the sparse path grows.

`:custom-objective' names all four functions one iteration of `train''s custom loop makes,
for the same reason: `LGBM_BoosterUpdateOneIterCustom' performs the update,
`LGBM_BoosterGetPredict' reads the raw scores the caller's function is handed,
`LGBM_BoosterGetNumPredict' says how long that read's buffer is, and
`LGBM_BoosterGetNumClasses' says how many output groups those scores have. A library
providing the update but not the read could not run one iteration, so a capability answering
true off the update alone would be answering about less than the caller asked.
`LGBM_BoosterGetNumPredict' is here because the score read goes through
`%booster-predictions', which sizes its buffer with `%num-predict' -- so the objective branch
of `train' calls it once per iteration, exactly as the evaluation branch does. It was missing
from this entry while that was already true, which would have let a LightGBM exporting the
other three open, answer `:custom-objective' TRUE, and then die at the first iteration with
SBCL's undefined-alien-function error rather than with the `capability-unavailable' this
probe exists to produce.

`:custom-evaluation' names the three functions `%booster-predictions' makes for ONE dataset:
`LGBM_BoosterGetNumPredict' for that dataset's own buffer length, `LGBM_BoosterGetPredict'
for the predictions the caller's `:evaluation' function is handed, and
`LGBM_BoosterGetNumClasses' for how many output groups they have. All three are also named by
`:custom-objective' above -- both branches read scores through the same `%booster-predictions'
-- which is expected rather than a duplication to collapse: `probe-capabilities' probes each
ENTRY independently, and `tools/ci/check-abi-blacklist.lisp' maps an imported C name back to
the LIST of capabilities naming it, so one name serving two capabilities is a list of two
rather than a conflict. Collapsing the two entries into one would make a library missing
`LGBM_BoosterUpdateOneIterCustom' -- which the evaluation path never calls -- answer false for
a capability it can perfectly well provide.

Both custom entries belong HERE and not in `*provided-capabilities*' below, and the
distinction is not cosmetic: not one of the four C names between them is in
`*required-symbols*' above, so a LightGBM missing any of them opens perfectly well and simply
cannot boost against a caller's own gradient, or hand a caller's own metric one dataset's
predictions -- which is exactly the state a probe exists to detect. `probe-capabilities'
records PROVIDED entries ahead of probed ones, so naming a capability in both lists would
make the probe's answer unreachable; its own docstring calls that combination a contradiction
in the backend's declarations. This is the same shape as `:sparse-input' above, and the
opposite of `:categorical-features' below, whose column list is a parameter-string key with
no symbol to look for at all.")

(defparameter *provided-capabilities*
  '(:evaluation-history :early-stopping :categorical-features :prediction-shape)
  "Capabilities this backend provides unconditionally, recorded true at `open-backend'
without being probed -- `probe-capabilities''s PROVIDED, which says why a probe cannot
express this.

`:evaluation-history' is here rather than in `*optional-symbols*' because the C functions
`train' records a history with -- `LGBM_BoosterGetEvalCounts', `LGBM_BoosterGetEvalNames'
and `LGBM_BoosterGetEval', all reached through `%read-evaluation' -- are in
`*required-symbols*' above. A library missing any of them never opens at all, so there is no
state in which this backend is open and cannot record a history, and nothing for a probe to
answer differently from one open to the next.
`:early-stopping' is here for a different reason: it needs no C function at all. Both `train'
loops run in Lisp, so the stop decision is `cl-gbdt/src/training/early-stopping''s pure code,
which every open backend has by construction. A probe has nothing to look for, and the
capability cannot vary between one open and the next.
`:categorical-features' is here for a third reason, and the plainest of the three: on this
backend the column list is a KEY IN THE PARAMETER STRING -- `categorical_feature', written
by `make-dataset' through `%parameter-string' -- so no C symbol carries it at all, and
there is nothing a probe could look for. That is where this differs from XGBoost's entry of
the same name, which is provided because the call that attaches the list,
`XGDMatrixSetStrFeatureInfo', is already required there. The two entry points that read
this backend's parameter string, `LGBM_DatasetCreateFromMat' and
`LGBM_DatasetCreateFromCSR', are already accounted for -- the first in `*required-symbols*'
above, the second in `*optional-symbols*' under `:sparse-input' -- so a library that opens
at all can be told which of its columns hold categories.

`:prediction-shape' is here for a fourth reason, and one that sets it apart from XGBoost's
entry of the same name: this library states no shape whatsoever.
`LGBM_BoosterCalcNumPredict' -- in `*required-symbols*' above already, `predict' having always
sized its output buffer from it -- returns an ELEMENT COUNT and nothing more, where XGBoost's
prediction entry points write an `out_shape'/`out_dim' pair its wrapper reads straight back. So
`predict''s second value is DERIVED here, and what the capability says on this backend is that
the mechanism is present, NOT that the library stated anything. Measured, and asserted kind by
kind in tests/functional/prediction-shape.lisp: `:normal' and `:raw' come back as the result
array's own dimensions, `:contrib' as the three axes
`cl-gbdt/src/config/prediction-shape''s `contrib-shape' divides that element count into, and
`:leaf-index' as NIL -- its sub-layout has no property this project can check, and a shape
asserted without one would be a guess in a measurement's clothes. All of that is this wrapper's
own Lisp, present in every image that has it, so a probe would have nothing to look for even
were there a symbol to look for it in.

No operation re-checks it. Nothing takes an argument asking for a shape, so a backend answering
false returns NIL as `predict''s second value and predicts exactly as it otherwise would,
rather than signalling. Six of the ten names in `cl-gbdt/src/backend''s
`*known-capabilities*' ARE re-checked by the operation they gate -- `:sparse-input',
`:missing-value', `:categorical-features', `:custom-objective' and `:custom-evaluation', by
`%check-sparse-input', `%check-missing-value', `%check-categorical-features',
`%check-custom-objective' and `%check-custom-evaluation' in
`cl-gbdt/src/lightgbm/protocol', and `:model-slicing', by XGBoost's `slice-model' -- and the
other four, this one among them, are re-checked nowhere. See `*known-capabilities*' itself,
where that split is stated in full.

Every name here must be registered in `cl-gbdt/src/backend''s `*known-capabilities*', or
`backend-supports-p' would signal `unknown-capability' for a capability the plist claims;
`tools/ci/check-abi-blacklist.lisp''s CHECK C is what enforces that, for this list and
`*optional-symbols*' alike.")

;;; ---------------------------------------------------------------------------
;;; Datasets

(defun %parameter-string (parameters)
  "Return PARAMETERS, a plist, as the space-separated \"key=value\" string
LightGBM's C API expects. NIL yields the empty string."
  (format nil "~{~A~^ ~}"
          (mapcar (lambda (pair) (format nil "~A=~A" (car pair) (cdr pair)))
                   (normalize-parameters parameters))))

(defun %data-type (element-type)
  "Map ELEMENT-TYPE, as `with-foreign-matrix' reports it, to LightGBM's
`C_API_DTYPE_*' constant.

`ecase', not `case': `with-foreign-matrix' promises only `double-float' or
`single-float' -- see `cl-gbdt/src/data''s `foreign-element-type' -- so any other
value reaching here is a bug in this file, not a value LightGBM should be told
the shape of. Mirrors `cl-gbdt/src/xgboost/native''s `%array-interface-typestr',
the same per-element-type mapping for XGBoost's array-interface JSON instead of
a raw dtype constant.

Extracted out of `make-dataset' and `predict''s method bodies during this
backend's own Phase 1 split, so `cl-gbdt/src/lightgbm/protocol' never needs to
name a `C_API_DTYPE_*' constant itself -- see policy section 11's rule against a
backend-specific public package re-exporting the raw C API, which a bare
`ecase' left in `protocol.lisp' referencing these constants directly would have
violated even without exporting them further."
  (ecase element-type
    (double-float +c-api-dtype-float64+)
    (single-float +c-api-dtype-float32+)))

(defun %create-dataset (matrix parameter-string reference-pointer)
  "Build a LightGBM dataset from MATRIX via `LGBM_DatasetCreateFromMat', returning
its raw pointer. PARAMETER-STRING is `%parameter-string''s space-separated
\"key=value\" form; REFERENCE-POINTER is `%reference-pointer''s result -- a
`DatasetHandle' to align MATRIX's bin mapper to, or a null pointer for none.

Extracted so `cl-gbdt/src/lightgbm/protocol''s `make-dataset' delegates the call
itself to this file instead of naming `lgbm-dataset-create-from-mat' directly --
see policy section 3's Layer 2 delegating to Layer 1. `make-dataset' still owns
checking the returned pointer for null afterward. Mirrors
`cl-gbdt/src/xgboost/native''s `%create-dmatrix': MATRIX's element type decides
LightGBM's `C_API_DTYPE_*' constant via `%data-type', the way `%create-dmatrix'
decides XGBoost's array-interface typestr."
  (with-foreign-matrix (data-pointer nrow ncol element-type) matrix
    (let ((data-type (%data-type element-type)))
      (cffi:with-foreign-string (parameter-cstring parameter-string)
        (cffi:with-foreign-object (out :pointer)
          (check-lgbm (lgbm-dataset-create-from-mat
                       data-pointer data-type nrow ncol 1
                       parameter-cstring reference-pointer out)
                      "LGBM_DatasetCreateFromMat")
          (cffi:mem-ref out :pointer))))))

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
path needs: `LGBM_DatasetCreateFromCSR' copies the rows into LightGBM's own representation
before it returns, exactly the lifetime `with-foreign-matrix' gives the dense path.

Duplicated verbatim in `cl-gbdt/src/xgboost/native', which needs the identical helper for
`XGDMatrixCreateFromCSR'. The one file both could share it from -- `cl-gbdt/src/data' -- is
re-exported wholesale by `cl-gbdt', so putting it there would publish a raw pinning primitive
as part of the unified API's surface. This backend pair is where the duplication belongs
instead, alongside `%set-feature-names' and `%free-*-unchecked', which mirror each other
across the two backends for the same reason."
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

(defun %create-dataset-from-csr (indptr indices values num-columns parameter-string
                                  reference-pointer)
  "Build a LightGBM dataset from a `csr-matrix''s INDPTR, INDICES and VALUES via
`LGBM_DatasetCreateFromCSR', returning its raw pointer. NUM-COLUMNS is the matrix's declared
width; PARAMETER-STRING and REFERENCE-POINTER mean exactly what they do for `%create-dataset'
above, and reach the identical C parameters.

The three vectors are passed to C as raw pointers -- `%call-with-pinned-csr' above -- rather
than described by anything resembling XGBoost's array-interface JSON, matching
`LGBM_DatasetCreateFromMat' taking a bare pointer and dimensions. Their element types are
therefore declared to the library instead, as the `C_API_DTYPE_*' tags `INDPTR-TYPE' and
`DATA-TYPE': `csr-matrix' fixes them at construction, so unlike `%create-dataset' there is
nothing to map through `%data-type' here. `LGBM_DatasetCreateFromCSR''s INDICES parameter has
no such tag at all -- it is a plain `const int32_t*' -- which is why only two are passed.

NINDPTR is INDPTR's own length, one more than the row count, and NELEM the number of stored
elements, which `csr-matrix' guarantees equals both VALUES' length and INDPTR's last entry.
`make-dataset' still owns checking the returned pointer for null afterward, as it does for
the dense path."
  (%call-with-pinned-csr
   indptr indices values
   (lambda (indptr-pointer indices-pointer values-pointer)
     (cffi:with-foreign-string (parameter-cstring parameter-string)
       (cffi:with-foreign-object (out :pointer)
         (check-lgbm (lgbm-dataset-create-from-csr
                      indptr-pointer +c-api-dtype-int32+ indices-pointer
                      values-pointer +c-api-dtype-float64+
                      (length indptr) (length values) num-columns
                      parameter-cstring reference-pointer out)
                     "LGBM_DatasetCreateFromCSR")
         (cffi:mem-ref out :pointer))))))

(defun %set-dataset-field (dataset-pointer field-name values cffi-type dtype coercer)
  "Attach the sequence VALUES to DATASET-POINTER's FIELD-NAME via
`LGBM_DatasetSetField'. Each element is coerced through COERCER and stored as
CFFI-TYPE; DTYPE is the matching C_API_DTYPE constant.

Not exported: `%set-info-field' and `%set-group-field' below are the only
callers, and each hardcodes its own CFFI-TYPE/DTYPE/COERCER so that
`cl-gbdt/src/lightgbm/protocol' -- where FIELD-NAME originates -- never has to
name a `C_API_DTYPE_*' constant itself."
  (let ((count (length values)))
    (cffi:with-foreign-object (buffer cffi-type count)
      (write-foreign-sequence buffer cffi-type values coercer)
      (cffi:with-foreign-string (name field-name)
        (check-lgbm (lgbm-dataset-set-field dataset-pointer name buffer count dtype)
                    "LGBM_DatasetSetField")))))

(defun %set-info-field (dataset-pointer field-name values)
  "Attach VALUES to DATASET-POINTER's FIELD-NAME -- \"label\" or \"weight\" -- via
`%set-dataset-field', coerced to `single-float' and stored as LightGBM's
`C_API_DTYPE_FLOAT32'.

A thin wrapper hardcoding `%set-dataset-field''s CFFI-TYPE/DTYPE/COERCER
arguments, added during this backend's own Phase 1 split for the reason
`%set-dataset-field''s own docstring gives. Matches
`cl-gbdt/src/xgboost/native''s `%set-info-field', which hardcodes the identical
dtype for XGBoost's own `label'/`weight'."
  (%set-dataset-field dataset-pointer field-name values :float
                       +c-api-dtype-float32+
                       (lambda (value) (coerce value 'single-float))))

(defun %set-group-field (dataset-pointer group)
  "Attach the sequence GROUP -- ranking query-group sizes -- to DATASET-POINTER via
`%set-dataset-field' under LightGBM's \"group\" field, as `int32'
(`C_API_DTYPE_INT32'). See `cl-gbdt/src/xgboost/native''s `%set-group-field' for
the same field via XGBoost's differently-typed `unsigned' equivalent."
  (%set-dataset-field dataset-pointer "group" group :int32
                       +c-api-dtype-int32+ #'round))

(defun %set-feature-names (dataset-pointer feature-names)
  "Attach FEATURE-NAMES, a list of strings, to DATASET-POINTER via
`LGBM_DatasetSetFeatureNames'.

Signals `unsupported-argument' against `:lightgbm' when FEATURE-NAMES is not a proper
list, via `check-feature-names' -- checked before COUNT is computed, since `length' on a
dotted list is exactly the raw `type-error' that check exists to head off.

Every string successfully allocated is freed on any exit, including one signaled
partway through the allocation loop itself -- ALLOCATED tracks exactly how many
of the COUNT slots hold a real `foreign-string-alloc' result, so cleanup never
calls `foreign-string-free' on an uninitialized foreign-object slot."
  (check-feature-names feature-names :lightgbm)
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
             (check-lgbm (lgbm-dataset-set-feature-names dataset-pointer names count)
                         "LGBM_DatasetSetFeatureNames"))
        (dotimes (index allocated)
          (cffi:foreign-string-free (cffi:mem-aref names :pointer index)))))))

(defun %reference-pointer (backend reference dataset-class)
  "Return the foreign pointer `make-dataset' should pass as
`LGBM_DatasetCreateFromMat''s `reference' argument for REFERENCE, or a null
pointer when REFERENCE is NIL.

DATASET-CLASS is `cl-gbdt/src/lightgbm/protocol''s `lightgbm-dataset', passed in
by the caller for the same reason `%check-lightgbm-dataset' takes it as a
parameter instead of naming the symbol directly -- see that function's
docstring.

Delegates everything else to `%check-lightgbm-dataset': signals
`wrong-backend-reference' when REFERENCE is not of type DATASET-CLASS, or
whatever `handle-live-pointer' signals when it is one but already freed or its
own backend has since closed."
  (if (null reference)
      (cffi:null-pointer)
      (%check-lightgbm-dataset backend reference "make-dataset's :reference" dataset-class)))

(defun %free-dataset-unchecked (pointer)
  "Free the dataset at POINTER via `LGBM_DatasetFree' without checking its
returned status.

`make-dataset' calls this from its cleanup path when ownership of a partially
built dataset never transferred to a handle -- a signal already unwinding the
stack there must not be replaced by a status-check failure from this
best-effort free. Mirrors `cl-gbdt/src/xgboost/native''s
`%free-dmatrix-unchecked'."
  (lgbm-dataset-free pointer))

(defun %dataset-num-rows (pointer)
  "Return the row count of the dataset at POINTER, read via `LGBM_DatasetGetNumData'."
  (cffi:with-foreign-object (out :int32)
    (check-lgbm (lgbm-dataset-get-num-data pointer out) "LGBM_DatasetGetNumData")
    (cffi:mem-ref out :int32)))

(defun %dataset-num-features (pointer)
  "Return the feature (column) count of the dataset at POINTER, read via
`LGBM_DatasetGetNumFeature'."
  (cffi:with-foreign-object (out :int32)
    (check-lgbm (lgbm-dataset-get-num-feature pointer out) "LGBM_DatasetGetNumFeature")
    (cffi:mem-ref out :int32)))

(defun %free-dataset (pointer)
  "Free the dataset at POINTER via `LGBM_DatasetFree', signalling
`foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/lightgbm/protocol''s `free-dataset' delegates the call
itself to this file instead of naming `lgbm-dataset-free' directly."
  (check-lgbm (lgbm-dataset-free pointer) "LGBM_DatasetFree"))

;;; ---------------------------------------------------------------------------
;;; Training

(defun %create-booster (train-data-pointer parameter-string)
  "Create a booster on TRAIN-DATA-POINTER via `LGBM_BoosterCreate', returning its
pointer.

Signals `foreign-call-error' when creation reports success but writes a null
handle -- the same guard `make-dataset' applies to `LGBM_DatasetCreateFromMat',
for the same reason: every later call through this handle would otherwise
dereference it blindly."
  (let ((booster-pointer
          (cffi:with-foreign-string (parameter-cstring parameter-string)
            (cffi:with-foreign-object (out :pointer)
              (check-lgbm (lgbm-booster-create train-data-pointer parameter-cstring out)
                          "LGBM_BoosterCreate")
              (cffi:mem-ref out :pointer)))))
    (when (cffi:null-pointer-p booster-pointer)
      (error 'foreign-call-error
             :function-name "LGBM_BoosterCreate"
             :code 0
             :message "reported success but returned a null booster handle"))
    booster-pointer))

(defun %add-valid-data (booster-pointer valid-set-pointers)
  "Attach each pointer in VALID-SET-POINTERS to BOOSTER-POINTER via
`LGBM_BoosterAddValidData'.

`train' has already run every entry of its VALID-SETS through
`%check-lightgbm-dataset' -- type-checked and read to a live pointer -- before
this is ever called, so this function makes no check of its own: it exists
only to keep the foreign-call loop separate from that validation.
`LGBM_BoosterAddValidData' dereferences each pointer directly, so a stale or
wrong-kind one reaching this point is a segfault, not a catchable condition,
which is exactly what the caller's validation pass exists to rule out first."
  (dolist (pointer valid-set-pointers)
    (check-lgbm (lgbm-booster-add-valid-data booster-pointer pointer)
                "LGBM_BoosterAddValidData")))

(defun %update-one-iteration (booster-pointer)
  "Advance BOOSTER-POINTER by one boosting iteration via
`LGBM_BoosterUpdateOneIter'. Returns the raw `produced_empty_tree' out
parameter: nonzero when this iteration produced no split."
  (cffi:with-foreign-object (finished :int)
    (check-lgbm (lgbm-booster-update-one-iter booster-pointer finished)
                "LGBM_BoosterUpdateOneIter")
    (cffi:mem-ref finished :int)))

(defun %booster-num-classes (booster-pointer)
  "Return how many output groups BOOSTER-POINTER's model has, via
`LGBM_BoosterGetNumClasses'.

One for regression and binary classification, `num_class' for multiclass -- and `num_class'
is still what supplies it under `objective=none', which is why forcing the objective does not
also have to supply the group count."
  (cffi:with-foreign-object (classes :int)
    (check-lgbm (lgbm-booster-get-num-classes booster-pointer classes)
                "LGBM_BoosterGetNumClasses")
    (cffi:mem-ref classes :int)))

(defun %num-predict (booster-pointer data-index)
  "Return how many `double's `LGBM_BoosterGetPredict' will write for the dataset at
DATA-INDEX on BOOSTER-POINTER, via `LGBM_BoosterGetNumPredict'.

The length is PER DATASET and cannot be derived from the training set's row count:
measured against the vendored library, this call answers 17 for a 17-row validation set
attached to a 40-row training set. That is the whole reason it is called at all rather
than the buffer being sized from a row count the caller already has."
  (cffi:with-foreign-object (out-len :int64)
    (check-lgbm (lgbm-booster-get-num-predict booster-pointer data-index out-len)
                "LGBM_BoosterGetNumPredict")
    (cffi:mem-ref out-len :int64)))

(defun %booster-predictions (booster-pointer data-index rows)
  "Return the predictions BOOSTER-POINTER currently holds for the dataset at DATA-INDEX, as
a (ROWS GROUPS) `double-float' array.

DATA-INDEX is LightGBM's own `data_idx' numbering, passed straight through: 0 is the
training set the booster was built from, 1 the first dataset attached with
`LGBM_BoosterAddValidData', 2 the second. ROWS is THAT dataset's own row count and never
another dataset's -- the length check below is what holds a caller to it.

`LGBM_BoosterGetPredict' reads the values LightGBM already holds for that dataset rather
than predicting afresh, which is both cheaper and, for the training set, the number the next
gradient has to be computed from. Its buffer is flat and **group-major**: row I of output
group K sits at `(+ (* K ROWS) I)'. Measured against `predict :kind :raw' on the same booster
under `objective=none', where the two agree exactly; read row-major they differ.

The values are `predict :kind :normal''s. Measured against the vendored library on one
`objective=binary' booster over a 40-row training set and a 17-row validation set: for BOTH
datasets this buffer agrees with `predict :kind :normal' over that dataset's own matrix to
0.0 and differs from `predict :kind :raw' by 0.706, so what is held here is the transformed
prediction, not the margin. Under `objective=none' -- which `train' forces for a
caller-supplied :OBJECTIVE, and only then -- there is no transform to apply, so `:normal'
and `:raw' are the same numbers and this buffer agrees with both to 0.0. Also measured, on
the same two datasets.

The buffer's length comes from `%num-predict' for this DATA-INDEX rather than from
`ROWS x GROUPS', for the reason that function's docstring records. GROUPS comes from
`LGBM_BoosterGetNumClasses', which is a property of the model rather than of any one
dataset, and the two readings are ASSERTED to agree given ROWS: a mismatch means either the
wrong dataset's row count reached this function -- the exact bug per-dataset lengths exist
to rule out -- or the library disagreeing with itself, and truncating silently would hide
both."
  (let ((groups (%booster-num-classes booster-pointer))
        (count (%num-predict booster-pointer data-index)))
    (assert (= count (* rows groups)) ()
            "LGBM_BoosterGetNumPredict reports ~D elements for data_idx ~D, expected ~D ~
             for ~D rows x ~D groups"
            count data-index (* rows groups) rows groups)
    (cffi:with-foreign-objects ((out-len :int64) (buffer :double count))
      (check-lgbm (lgbm-booster-get-predict booster-pointer data-index out-len buffer)
                  "LGBM_BoosterGetPredict")
      (assert (= count (cffi:mem-ref out-len :int64)) ()
              "LGBM_BoosterGetPredict wrote ~D elements for data_idx ~D, expected ~D"
              (cffi:mem-ref out-len :int64) data-index count)
      (let ((predictions (make-array (list rows groups) :element-type 'double-float)))
        (dotimes (row rows predictions)
          (dotimes (group groups)
            (setf (aref predictions row group)
                  (cffi:mem-aref buffer :double (+ (* group rows) row)))))))))

(defun %update-one-iteration-custom (booster-pointer grad hess)
  "Advance BOOSTER-POINTER by one iteration on the caller's GRAD and HESS, via
`LGBM_BoosterUpdateOneIterCustom'. Both are (ROWS GROUPS) arrays.

They are flattened **group-major** -- row I of output group K at `(+ (* K ROWS) I)' -- and
converted to `single-float', which is the only element type the C signature's `const float*'
admits. Measured: a gradient of alternating -1/+1 for group 0 and 0 elsewhere moves only group
0's raw score under this layout and smears across all three under the row-major one.

The conversion goes through `objective-single-float', so an element that is not a real signals
`unsupported-element-type' naming its type rather than a `type-error' from inside `coerce'.
GRAD and HESS may be `double-float', `single-float' or general arrays -- `check-objective-result'
checks their shape and leaves their elements to that function, one element at a time as they
are written, which is why nothing scans either array twice.

The `produced_empty_tree' out parameter is read because the C function requires the pointer,
and then discarded. `train''s ordinary loop already discards `%update-one-iteration''s
equivalent -- the public `update-one-iteration' returns it, `train' does not look at it -- and
a custom loop that ended early on the flag would make the two loops behave differently for the
same underlying condition."
  (let* ((rows (array-dimension grad 0))
         (groups (array-dimension grad 1))
         (count (* rows groups)))
    (cffi:with-foreign-objects ((grad-buffer :float count)
                                (hess-buffer :float count)
                                (produced :int))
      (dotimes (row rows)
        (dotimes (group groups)
          (let ((index (+ (* group rows) row)))
            (setf (cffi:mem-aref grad-buffer :float index)
                  (objective-single-float (aref grad row group)))
            (setf (cffi:mem-aref hess-buffer :float index)
                  (objective-single-float (aref hess row group))))))
      (check-lgbm (lgbm-booster-update-one-iter-custom booster-pointer grad-buffer hess-buffer
                                                       produced)
                  "LGBM_BoosterUpdateOneIterCustom"))))

(defun %check-booster-datasets-live (booster)
  "Signal `released-handle-error' when any dataset BOOSTER depends on -- its training
set, or any validation set attached via `train''s VALID-SETS -- has already been
freed.

`LGBM_BoosterUpdateOneIter' dereferences the booster's internal `train_data_'
pointer, and evaluates metrics against each attached validation set's stored
pointer too -- none of which `LGBM_DatasetFree' owns or clears when the
corresponding dataset is freed. Calling it after any of those datasets has been
freed out from under the booster is a segfault, not a catchable Lisp condition, so
every one of them has to be checked here, before any foreign call.
`booster-training-set' is NIL for a `load-model' booster, which has no training set
and needs no check; `booster-validation-sets' is NIL when `train' was called with
no VALID-SETS."
  (let ((training-set (booster-training-set booster)))
    (when (and training-set (handle-released-p training-set))
      (error 'released-handle-error :object training-set)))
  (dolist (validation-set (booster-validation-sets booster))
    (when (handle-released-p validation-set)
      (error 'released-handle-error :object validation-set))))

(defun %free-booster-unchecked (pointer)
  "Free the booster at POINTER via `LGBM_BoosterFree' without checking its
returned status.

`train' and `load-model' each call this from their cleanup path when ownership
of a partially built booster never transferred to a handle -- see
`%free-dataset-unchecked''s docstring for why a signal already unwinding the
stack there must not be replaced by a status-check failure from this
best-effort free."
  (lgbm-booster-free pointer))

(defun %free-booster (pointer)
  "Free the booster at POINTER via `LGBM_BoosterFree', signalling
`foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/lightgbm/protocol''s `free-booster' delegates the call
itself to this file instead of naming `lgbm-booster-free' directly."
  (check-lgbm (lgbm-booster-free pointer) "LGBM_BoosterFree"))

;;; ---------------------------------------------------------------------------
;;; Inference

(defun %predict-type (kind)
  "Map the protocol's KIND keyword onto LightGBM's `C_API_PREDICT_*' constant.

`ecase', not `case': an unrecognized KIND must error rather than silently
predicting something else."
  (ecase kind
    (:normal +c-api-predict-normal+)
    (:raw +c-api-predict-raw-score+)
    (:leaf-index +c-api-predict-leaf-index+)
    (:contrib +c-api-predict-contrib+)))

(defun %resolve-num-iteration (num-iteration)
  "Return NUM-ITERATION as LightGBM spells it on the wire: 0 means all
iterations, which is what NIL means in the protocol."
  (or num-iteration 0))

(defun %calc-num-predict (booster-pointer nrow predict-type start-iteration num-iteration)
  "Return the true element count `LGBM_BoosterPredictForMat' will write, via
`LGBM_BoosterCalcNumPredict'.

The row count alone is only the right buffer size for a single-class
objective -- this is how the true count, including any additional classes, is
obtained instead of assumed."
  (cffi:with-foreign-object (out :int64)
    (check-lgbm (lgbm-booster-calc-num-predict
                 booster-pointer nrow predict-type start-iteration num-iteration out)
                "LGBM_BoosterCalcNumPredict")
    (cffi:mem-ref out :int64)))

(defun %predict-ncol (element-count nrow)
  "Return ELEMENT-COUNT's per-row width for a matrix of NROW rows.

`(/ element-count nrow)' alone signals `division-by-zero' when the matrix has no
rows, and silently yields a ratio, not an integer, if `LGBM_BoosterCalcNumPredict'
ever reports a count that is not an exact multiple of NROW. NROW = 0 has an
obvious answer -- there is no row to give a width to -- so it is handled directly
rather than routed through the assertion below."
  (if (zerop nrow)
      0
      (multiple-value-bind (quotient remainder) (truncate element-count nrow)
        (assert (zerop remainder) ()
                "LGBM_BoosterCalcNumPredict reported ~D elements for ~D rows, not an ~
                 exact multiple of the row count" element-count nrow)
        quotient)))

(defun %predict-for-mat (pointer data-pointer data-type nrow ncol predict-type
                          iteration-count parameter-cstring out-len buffer)
  "Run `LGBM_BoosterPredictForMat' over the booster at POINTER against the matrix
described by DATA-POINTER/DATA-TYPE/NROW/NCOL, writing OUT-LEN and BUFFER, and
signal `foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/lightgbm/protocol''s `predict' delegates the call
itself to this file instead of naming `lgbm-booster-predict-for-mat' directly --
see policy section 3's Layer 2 delegating to Layer 1. `predict' still owns
sizing BUFFER from `%calc-num-predict', deriving DATA-TYPE via `%data-type', and
copying BUFFER's contents out afterward."
  (check-lgbm (lgbm-booster-predict-for-mat
               pointer data-pointer data-type nrow ncol 1 predict-type 0
               iteration-count parameter-cstring out-len buffer)
              "LGBM_BoosterPredictForMat"))

(defun %predict-for-csr (pointer indptr indices values num-col predict-type iteration-count
                          parameter-cstring out-len buffer)
  "Run `LGBM_BoosterPredictForCSR' over the booster at POINTER against the matrix a
`csr-matrix''s INDPTR, INDICES and VALUES describes, NUM-COL wide, writing OUT-LEN and
BUFFER, and signal `foreign-call-error' when the library reports failure.

The CSR counterpart of `%predict-for-mat' above, and the second of this file's two callers
of `%call-with-pinned-csr' -- the three vectors are pinned for the duration of the call and
no longer, which is all this needs: `LGBM_BoosterPredictForCSR' reads them and fills BUFFER
before it returns, exactly the lifetime `%create-dataset-from-csr' relies on for ingestion.

Their element types are declared to the library as the `C_API_DTYPE_*' tags INDPTR-TYPE and
DATA-TYPE, fixed here rather than mapped through `%data-type': `csr-matrix' fixes both at
construction, so unlike `%predict-for-mat' there is no per-call element type to map.
`LGBM_BoosterPredictForCSR''s INDICES parameter carries no such tag at all -- it is a plain
`const int32_t*' -- which is why only two are passed, the same asymmetry
`%create-dataset-from-csr' documents for the ingestion entry point.

NINDPTR is INDPTR's own length, one more than the row count, and NELEM the number of stored
elements. START-ITERATION is 0, exactly as `%predict-for-mat' passes it: the protocol
exposes no start-iteration override on either path. `predict' still owns sizing BUFFER from
`%calc-num-predict', checking OUT-LEN against that size afterward, and copying BUFFER's
contents out -- none of which differs between the two entry points."
  (%call-with-pinned-csr
   indptr indices values
   (lambda (indptr-pointer indices-pointer values-pointer)
     (check-lgbm (lgbm-booster-predict-for-csr
                  pointer indptr-pointer +c-api-dtype-int32+ indices-pointer
                  values-pointer +c-api-dtype-float64+
                  (length indptr) (length values) num-col
                  predict-type 0 iteration-count parameter-cstring out-len buffer)
                 "LGBM_BoosterPredictForCSR"))))

;;; ---------------------------------------------------------------------------
;;; Persistence

(defun %save-model (pointer num-iteration filename)
  "Save the booster at POINTER to FILENAME via `LGBM_BoosterSaveModel', limited to
NUM-ITERATION trees (0 for all of them -- `%resolve-num-iteration''s wire form),
and signal `foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/lightgbm/protocol''s `save-model' delegates the call
itself to this file instead of naming `lgbm-booster-save-model' directly."
  (check-lgbm (lgbm-booster-save-model
               pointer 0 num-iteration +c-api-feature-importance-split+ filename)
              "LGBM_BoosterSaveModel"))

(defun %create-booster-from-modelfile (filename out-num-iterations out)
  "Create a booster from FILENAME via `LGBM_BoosterCreateFromModelfile', writing
OUT-NUM-ITERATIONS and OUT, and signal `foreign-call-error' when the library
reports failure.

Extracted so `cl-gbdt/src/lightgbm/protocol''s `load-model' delegates the call
itself to this file instead of naming `lgbm-booster-create-from-modelfile'
directly. `load-model' still owns checking OUT for a null pointer afterward."
  (check-lgbm (lgbm-booster-create-from-modelfile filename out-num-iterations out)
              "LGBM_BoosterCreateFromModelfile"))

(defun %save-model-to-string (booster-pointer num-iteration)
  "Return BOOSTER-POINTER's model as a Lisp string via the two-call
`LGBM_BoosterSaveModelToString' idiom: a first call with a modest buffer
learns the required length from its OUT-LEN parameter, and -- only when that
length exceeds what was already tried -- a second call with a buffer of
exactly that length fetches the whole string. The model's serialized length is
not knowable ahead of time, so one fixed-size buffer cannot be assumed to fit."
  (let ((probe-length 1024))
    (cffi:with-foreign-object (out-len :int64)
      (cffi:with-foreign-object (probe :char probe-length)
        (check-lgbm (lgbm-booster-save-model-to-string
                     booster-pointer 0 num-iteration +c-api-feature-importance-split+
                     probe-length out-len probe)
                    "LGBM_BoosterSaveModelToString")
        (let ((length (cffi:mem-ref out-len :int64)))
          (if (<= length probe-length)
              (cffi:foreign-string-to-lisp probe)
              (cffi:with-foreign-object (buffer :char length)
                (check-lgbm (lgbm-booster-save-model-to-string
                             booster-pointer 0 num-iteration +c-api-feature-importance-split+
                             length out-len buffer)
                            "LGBM_BoosterSaveModelToString")
                (cffi:foreign-string-to-lisp buffer))))))))

;;; ---------------------------------------------------------------------------
;;; Feature importance

(defun %feature-importance-type (kind)
  "Map the protocol's KIND keyword onto LightGBM's
`C_API_FEATURE_IMPORTANCE_*' constant.

`ecase', not `case': an unrecognized KIND must error rather than silently
returning a different importance measure."
  (ecase kind
    (:split +c-api-feature-importance-split+)
    (:gain +c-api-feature-importance-gain+)))

(defun %booster-num-features (pointer)
  "Return the booster at POINTER's feature count, read via
`LGBM_BoosterGetNumFeature'.

`feature-importance' uses this to size its result -- works whether the booster
came from `train' or `load-model', unlike a booster's training set, which
`load-model' leaves unbound."
  (cffi:with-foreign-object (out :int)
    (check-lgbm (lgbm-booster-get-num-feature pointer out) "LGBM_BoosterGetNumFeature")
    (cffi:mem-ref out :int)))

(defun %feature-importance (pointer num-iteration importance-type buffer)
  "Run `LGBM_BoosterFeatureImportance' over the booster at POINTER with
NUM-ITERATION and IMPORTANCE-TYPE, writing BUFFER, and signal
`foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/lightgbm/protocol''s `feature-importance' delegates
the call itself to this file instead of naming `lgbm-booster-feature-importance'
directly. `feature-importance' still owns sizing BUFFER from
`%booster-num-features' and copying its contents out afterward."
  (check-lgbm (lgbm-booster-feature-importance pointer num-iteration importance-type buffer)
              "LGBM_BoosterFeatureImportance"))

;;; ---------------------------------------------------------------------------
;;; Evaluation
;;;
;;; `booster-eval-names' and `booster-eval' below are this file's first two exports that
;;; are not `%'-prefixed internal helpers: Phase 2 (policy section 3's Layer 1) exports
;;; both directly from `cl-gbdt/lightgbm' -- see `src/lightgbm/all.lisp' -- rather than
;;; through a `cl-gbdt/src/lightgbm/protocol' `defmethod' the way every other public
;;; operation on a booster is reached. See this file's own "Floating-point trap safety"
;;; header comment for what that changes about the mask.

(defun %check-lightgbm-booster (booster argument-description)
  "Return BOOSTER's live foreign pointer, after confirming BOOSTER is a booster built by
the `:lightgbm' backend.

Thin wrapper over `cl-gbdt/src/handle''s `%check-handle-kind', which carries the contract,
the conditions and the reason the check cannot be spelled as `(typep booster
'lightgbm-booster)' here. ARGUMENT-DESCRIPTION names which caller-supplied argument BOOSTER
came from, for `wrong-backend-reference''s report.

Kept as a named wrapper, rather than calling `%check-handle-kind' directly at each call
site, so `booster-eval' and `booster-eval-names' name the backend once between them instead
of once each."
  (%check-handle-kind booster 'booster :lightgbm argument-description))

(defun %booster-eval-count (pointer)
  "Return the number of evaluation metrics configured on the booster at POINTER, read via
`LGBM_BoosterGetEvalCounts'.

`booster-eval-names' and `booster-eval' both call this to size their buffers -- 0 for a
booster trained with `metric=none' or returned by `load-model', neither of which ever had
metrics attached."
  (cffi:with-foreign-object (out :int)
    (check-lgbm (lgbm-booster-get-eval-counts pointer out) "LGBM_BoosterGetEvalCounts")
    (cffi:mem-ref out :int)))

(defun %booster-eval-names (pointer count)
  "Return the COUNT evaluation metric names configured on the booster at POINTER, as a
list of strings, via the two-call `LGBM_BoosterGetEvalNames' idiom: a first call with a
modest per-name buffer learns the longest name's required length from its OUT-BUFFER-LEN
parameter, and -- only when that length exceeds what was already tried -- a second call
with every buffer sized to exactly that length fetches them all. No single metric name's
length is knowable ahead of time, so one fixed-size buffer cannot be assumed to fit every
one of them. Mirrors `%save-model-to-string''s two-call idiom, extended from one buffer to
COUNT of them, since `LGBM_BoosterGetEvalNames' fills COUNT separate `char*' buffers
rather than one string.

Returns NIL when COUNT is 0 without making any foreign call: there is no data_idx or other
per-call validation `LGBM_BoosterGetEvalNames' performs -- it reports the same COUNT names
regardless of which dataset they will later be read against -- so skipping the call when
there is nothing to fetch loses no safety, unlike `%booster-eval' below."
  (if (zerop count)
      nil
      (flet ((fetch (buffer-length)
               (cffi:with-foreign-objects ((out-len :int) (out-buffer-len :size)
                                            (out-strs :pointer count))
                 (dotimes (index count)
                   (setf (cffi:mem-aref out-strs :pointer index)
                         (cffi:foreign-alloc :char :count buffer-length)))
                 (unwind-protect
                      (progn
                        (check-lgbm (lgbm-booster-get-eval-names
                                     pointer count out-len buffer-length
                                     out-buffer-len out-strs)
                                    "LGBM_BoosterGetEvalNames")
                        (values (cffi:mem-ref out-buffer-len :size)
                                (loop :for index :below count
                                      :collect (cffi:foreign-string-to-lisp
                                                (cffi:mem-aref out-strs :pointer index)))))
                   (dotimes (index count)
                     (cffi:foreign-free (cffi:mem-aref out-strs :pointer index)))))))
        (let ((probe-length 64))
          (multiple-value-bind (required names) (fetch probe-length)
            (if (<= required probe-length)
                names
                (nth-value 1 (fetch required))))))))

(defun %booster-eval (pointer data-index count)
  "Return the COUNT evaluation metric values for the dataset at DATA-INDEX on the booster
at POINTER, via `LGBM_BoosterGetEval', as a fresh `(simple-array double-float (count))' in
the same order `%booster-eval-names' reports the metrics' names.

Always calls `LGBM_BoosterGetEval', even when COUNT is 0: unlike `%booster-eval-names',
this call also validates DATA-INDEX against the datasets BOOSTER actually has attached,
and skipping it would skip that check too, silently returning an empty array for an
out-of-range DATA-INDEX instead of signalling -- confirmed directly against the vendored
library, which rejects an out-of-range DATA-INDEX the same way whether COUNT is 0 or not.
The output buffer is sized `(max count 1)' doubles so a zero-metric booster never asks
CFFI for a zero-length foreign array.

`LGBM_BoosterGetEval' writes its own element count back through OUT-LEN; this is asserted
equal to COUNT rather than trusted silently, since the buffer was sized from
`%booster-eval-count' and a mismatch would mean either an under-filled result or a write
past the allocated buffer going unnoticed -- the same check `predict' makes against
`LGBM_BoosterCalcNumPredict'."
  (cffi:with-foreign-objects ((out-len :int) (buffer :double (max count 1)))
    (check-lgbm (lgbm-booster-get-eval pointer data-index out-len buffer) "LGBM_BoosterGetEval")
    (assert (= count (cffi:mem-ref out-len :int)) ()
            "LGBM_BoosterGetEval wrote ~D elements, expected ~D from LGBM_BoosterGetEvalCounts"
            (cffi:mem-ref out-len :int) count)
    (let ((result (make-array count :element-type 'double-float)))
      (dotimes (index count result)
        (setf (aref result index) (cffi:mem-aref buffer :double index))))))

(defun %read-evaluation (booster-pointer dataset-count)
  "Return the booster at BOOSTER-POINTER's evaluation entries for datasets 0 through
DATASET-COUNT - 1, as a fresh list of (INDEX METRIC-NAME VALUE) lists, via
`%booster-eval-count', `%booster-eval-names' and `%booster-eval' -- one entry per (dataset,
metric) pair, dataset-major, in the order `evaluation''s own generic function contract
promises, pairing entry N of the single metric-name list `%booster-eval-names' reports for
the whole booster with entry N of each dataset's own `%booster-eval' result.

The caller owns every guard this needs before calling: BOOSTER-POINTER must already be a
live handle's pointer, `%check-booster-datasets-live' must already have run for the booster
these datasets belong to, and the whole call must already be inside
`with-foreign-float-traps-masked''s dynamic extent -- like every other `%'-function in this
file, this does not establish any of those itself. `cl-gbdt/src/lightgbm/protocol''s
`evaluation' method and `train''s per-iteration recording loop both call this same function,
on the pointer and dataset count each already has in hand, rather than each computing
entries its own way -- that is what keeps the numbers `evaluation' reports after training and
the numbers recorded during training from ever being able to disagree."
  (let* ((count (%booster-eval-count booster-pointer))
         (names (%booster-eval-names booster-pointer count)))
    (loop :for index :below dataset-count
          :append (loop :for name :in names
                        :for value :across (%booster-eval booster-pointer index count)
                        :collect (list index name value)))))

(defun booster-eval-names (booster)
  "Return the names of BOOSTER's configured evaluation metrics, as a fresh list of
strings, via `LGBM_BoosterGetEvalCounts' and `LGBM_BoosterGetEvalNames'.

The returned list belongs to the caller outright; cl-gbdt keeps no reference to it. Entry
N here names entry N of the value list `booster-eval' returns for any DATA-INDEX -- metric
names are configured once for the whole booster, not per dataset, which is why this takes
no DATA-INDEX argument of its own, unlike `booster-eval'. NIL when BOOSTER has no metrics
configured -- trained with `metric=none', or returned by `load-model', which never had
metrics attached in the first place.

Signals `wrong-backend-reference' when BOOSTER was not built by the LightGBM backend,
`released-handle-error' when it has already been freed, and `backend-not-open' when its
own backend has since been closed."
  (with-foreign-float-traps-masked
    (let* ((pointer (%check-lightgbm-booster booster "booster-eval-names's booster argument"))
           (count (%booster-eval-count pointer)))
      (%booster-eval-names pointer count))))

(defun booster-eval (booster data-index)
  "Return BOOSTER's evaluation metric values for the dataset at DATA-INDEX, as a fresh
`(simple-array double-float (*))', via `LGBM_BoosterGetEvalCounts' and
`LGBM_BoosterGetEval'. Entry N corresponds to entry N of `booster-eval-names' -- call that
separately for the metric names, since LightGBM reports them independently of DATA-INDEX.

DATA-INDEX is LightGBM's own numbering, passed straight through to `LGBM_BoosterGetEval'
unmodified: 0 means the training set BOOSTER was built from; 1 means the first dataset in
`train''s :VALID-SETS, 2 the second, and so on. Nothing here assigns a name to any of
them -- see design policy section 4's rule against inventing dataset names for LightGBM.

Signals `foreign-call-error' when DATA-INDEX is negative or exceeds the number of datasets
BOOSTER actually has attached -- confirmed directly against the vendored library, which
rejects both with a descriptive `LGBM_GetLastError' message rather than reading out of
bounds. Also signals `wrong-backend-reference' when BOOSTER was not built by the LightGBM
backend, `released-handle-error' when BOOSTER itself or any dataset it retains has already
been freed, and `backend-not-open' when its own backend has since been closed.

`%check-booster-datasets-live' runs before any foreign call here, and covers every dataset
BOOSTER retains rather than only the one DATA-INDEX addresses: `LGBM_BoosterGetEval'
evaluates through metric objects built over each attached dataset's own label and weight
arrays, none of which `LGBM_DatasetFree' clears from the booster. Evaluating after any of
them was freed reads memory that is no longer ours -- it does not reliably crash, which is
what makes it worth a guard rather than a warning. This is the same check
`update-one-iteration' has always made, and the one the `evaluation' method makes at
Layer 2; it is repeated here because this function is public in `cl-gbdt/lightgbm' and so
is reachable without going through that method at all.

It runs after `%check-lightgbm-booster' rather than before it, unlike in the `evaluation'
method, because that method reaches its body only for an argument CLOS already dispatched
as a `lightgbm-booster' while this one accepts whatever the caller passes: reading
`booster-training-set' out of a dataset would fail as a slot type error instead of the
`wrong-backend-reference' this promises. Neither check makes a foreign call, so both still
precede every one of them."
  (with-foreign-float-traps-masked
    (let ((pointer (%check-lightgbm-booster booster "booster-eval's booster argument")))
      (%check-booster-datasets-live booster)
      (%booster-eval pointer data-index (%booster-eval-count pointer)))))
