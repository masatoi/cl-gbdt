;;;; native.lisp --- XGBoost backend, Layer 1: library discovery, the error wrapper, and
;;;; every internal function that turns a raw call into libxgboost.so into something
;;;; safe to call from Lisp -- out parameters into return values, error codes into
;;;; conditions, raw pointers accepted only where a caller already validated them.
;;;;
;;;; Nothing here is a CLOS protocol method and nothing here depends on
;;;; `cl-gbdt/src/xgboost/protocol' -- see that file for the fifteen `defmethod's this
;;;; module exists to support. `cl-gbdt/src/xgboost/protocol''s own docstrings, not this
;;;; file's, are the place each function's role in the unified API is explained; the
;;;; docstrings below describe only what each function does to the C API.

(uiop:define-package #:cl-gbdt/src/xgboost/native
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/xgboost/c-api
                #:xgb-get-last-error
                #:xg-boost-version
                #:xgd-matrix-create-from-dense
                #:xgd-matrix-set-info-from-interface
                #:xgd-matrix-set-u-int-info
                #:xgd-matrix-set-str-feature-info
                #:xgd-matrix-free
                #:xgd-matrix-num-row
                #:xgd-matrix-num-col
                #:xg-booster-create
                #:xg-booster-free
                #:xg-booster-set-param
                #:xg-booster-get-num-feature
                #:xg-booster-boosted-rounds
                #:xg-booster-update-one-iter
                #:xg-booster-eval-one-iter
                #:xg-booster-predict-from-d-matrix
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
           #:%check-backend-open
           #:%check-xgboost-dataset
           #:%check-unsupported
           #:%read-version
           #:%create-dmatrix
           #:%set-info-field
           #:%set-group-field
           #:%set-feature-names
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
           #:%predict-ncol
           #:%predict-from-dmatrix
           #:%save-model
           #:%load-model
           #:%save-model-to-buffer
           #:%feature-importance-type
           #:%booster-num-features
           #:%feature-score-index
           #:%check-feature-score-dim
           #:%feature-score
           #:%split-eval-label
           #:evaluate-one-iteration))

(in-package #:cl-gbdt/src/xgboost/native)

;;; ---------------------------------------------------------------------------
;;; Floating-point trap safety
;;;
;;; Almost nothing here wraps itself in `with-foreign-float-traps-masked' -- almost every
;;; function in this file is only ever called from inside a `cl-gbdt/src/xgboost/protocol'
;;; `defmethod' body that already established the mask, at method-body granularity, before
;;; making any of the calls below. See that file's identical commentary, and
;;; `with-foreign-float-traps-masked''s own docstring in `cl-gbdt/src/foreign', for why:
;;; SBCL enables floating-point traps by default on x86-64 and not on aarch64, and
;;; XGBoost's own numeric code -- confirmed for the softmax normalization behind a
;;; `multi:softprob' prediction -- was written and tested against the C convention of
;;; those traps staying masked.
;;;
;;; `evaluate-one-iteration', in the Evaluation section near the end of this file, is the exception:
;;; Task 3 (policy section 3's Layer 1) exports it directly from `cl-gbdt/xgboost' -- see
;;; `src/xgboost/all.lisp' -- so it is never reached through a `cl-gbdt/src/xgboost/protocol'
;;; `defmethod'. There is no outer call site left to establish the mask for it, so it wraps
;;; its own whole body instead, the same method-body granularity `protocol.lisp' uses, just
;;; with the wrapping function living here rather than there -- mirrors
;;; `cl-gbdt/src/lightgbm/native''s identical `booster-eval'/`booster-eval-names' exception.
;;; `tools/ci/check-float-traps.lisp' checks this directly: it reads the sibling `all.lisp''s
;;; public `:export' clause and requires every `defun' named there to open with this macro,
;;; the same rule it already applied to every `defmethod' in `protocol.lisp'.

;;; ---------------------------------------------------------------------------
;;; Error checking

(defun %last-error-message ()
  "Return XGBoost's last error message as a Lisp string, or NIL when
`XGBGetLastError' returns a null pointer."
  (let ((pointer (xgb-get-last-error)))
    (unless (cffi:null-pointer-p pointer)
      (cffi:foreign-string-to-lisp pointer))))

(defun check-xgb (code function-name)
  "Signal `foreign-call-error' when CODE reports failure, otherwise return CODE.

XGBoost returns 0 on success and a nonzero status -- documented as -1 -- on failure,
with the detail available from `XGBGetLastError'. FUNCTION-NAME identifies which C
function reported CODE, for the condition's report.

Thin wrapper around `check-foreign-call' supplying `%last-error-message' as XGBoost's
last-error thunk, matching `cl-gbdt/src/lightgbm/backend''s `check-lgbm' -- see that
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
`XGBoosterCreate' or `XGBoosterLoadModel' with a library that may no longer be mapped."
  (unless (backend-open-p backend)
    (error 'backend-not-open :backend (backend-name backend))))

(defun %check-xgboost-dataset (backend dataset argument-description dataset-class)
  "Return DATASET's live foreign pointer, after confirming DATASET is of type
DATASET-CLASS -- `cl-gbdt/src/xgboost/protocol''s `xgboost-dataset'. ARGUMENT-DESCRIPTION
names which caller-supplied argument DATASET came from -- e.g. \"train's dataset
argument\" -- for `wrong-backend-reference''s report.

DATASET-CLASS is a parameter, not a symbol named directly in this file, because this
file must not depend on `cl-gbdt/src/xgboost/protocol' -- see policy section 11 and
this file's own header -- and `xgboost-dataset' is defined there. The caller, `train',
passes `'xgboost-dataset' at each call site instead.

Every caller-supplied dataset argument -- `train''s DATASET and each entry of `train''s
:VALID-SETS -- must pass through here before reaching a foreign call that expects a
`DMatrixHandle'. `handle-live-pointer' alone is not enough: it only guards against a
released handle or a closed backend, and happily returns *any* handle's pointer
regardless of kind, including a booster's -- `train' dispatches on the backend, not on
the handle, so unlike `dataset-num-rows' or `free-dataset' there is no CLOS specializer
already ruling out the wrong kind of handle. A booster's own pointer reaching
`XGBoosterCreate''s DMatrix array is exactly the corruption this check exists to prevent
-- the identical hazard killed the process across several threads on the LightGBM branch.

Signals `wrong-backend-reference' when DATASET is not of type DATASET-CLASS -- built by a
different backend, or not a dataset at all -- and whatever `handle-live-pointer' signals
otherwise: `released-handle-error' for an already-freed DATASET, `backend-not-open' when
DATASET's own backend has since been closed.

This does not additionally check that DATASET was built by BACKEND specifically, only
that it is of type DATASET-CLASS -- see `cl-gbdt/src/lightgbm/backend''s
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
`cl-gbdt/src/lightgbm/backend''s identical constant.")

(defparameter *required-symbols*
  '("XGBoostVersion"
    "XGBGetLastError"
    "XGDMatrixCreateFromDense"
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
  '((:model-slicing "XGBoosterSlice"))
  "Capability name to the C function names that capability needs.

Unlike `*required-symbols*', whose absence makes `open-backend' signal
`missing-foreign-symbols', a name missing from here disables one capability and nothing else
-- policy section 8. An XGBoost too old to have `XGBoosterSlice' is a working XGBoost that
cannot slice, not a broken installation.

`XGBoosterSlice' is bound in c-api.lisp and called only from `slice-model', which checks the
capability before reaching it.")

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

(defparameter *dense-matrix-config-json* "{\"missing\":NaN}"
  "Config JSON passed to `XGDMatrixCreateFromDense'. Fixes the sentinel for a missing
value at IEEE NaN -- the value `with-foreign-matrix' can actually produce for a
`double-float' or `single-float' array element -- rather than exposing XGBoost's other
config keys (`nthread', `data_split_mode') through the protocol; nothing in the unified
API has a use for them yet.")

(defun %array-interface-typestr (element-type)
  "Map ELEMENT-TYPE, as `with-foreign-matrix' reports it, to the NumPy array-interface
typestr XGBoost's `array-interface-json' expects.

`ecase', not `case': `with-foreign-matrix' promises only `double-float' or `single-float'
-- see `cl-gbdt/src/data''s `foreign-element-type' -- so any other value reaching here is
a bug in this file, not a value XGBoost should silently be told the shape of."
  (ecase element-type
    (double-float "<f8")
    (single-float "<f4")))

(defun %create-dmatrix (matrix)
  "Build a DMatrix from MATRIX via `XGDMatrixCreateFromDense', returning its raw pointer.

MATRIX's foreign buffer is described to XGBoost with `array-interface-json' rather than
handed over as a pointer and dimensions -- `XGDMatrixCreateFromDense' takes the array
interface, unlike LightGBM's `LGBM_DatasetCreateFromMat'. The buffer only needs to stay
pinned for the duration of this call, since XGBoost copies it into its own representation
before returning."
  (with-foreign-matrix (data-pointer nrow ncol element-type) matrix
    (let ((typestr (%array-interface-typestr element-type)))
      (cffi:with-foreign-string (data (array-interface-json data-pointer typestr nrow ncol))
        (cffi:with-foreign-string (config *dense-matrix-config-json*)
          (cffi:with-foreign-object (out :pointer)
            (check-xgb (xgd-matrix-create-from-dense data config out)
                       "XGDMatrixCreateFromDense")
            (cffi:mem-ref out :pointer)))))))

(defun %set-info-field (dataset-pointer field-name values)
  "Attach the sequence VALUES to DATASET-POINTER's FIELD-NAME via
`XGDMatrixSetInfoFromInterface'. Used for `label' and `weight', both of which XGBoost
stores as single-precision floats regardless of the training matrix's own element type --
the same convention `cl-gbdt/src/lightgbm/backend' follows for the same two fields.

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
see `cl-gbdt/src/lightgbm/backend''s `make-dataset', which writes the same query-group
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

Every string successfully allocated is freed on any exit, including one signaled partway
through the allocation loop itself -- ALLOCATED tracks exactly how many of the COUNT slots
hold a real `foreign-string-alloc' result, matching
`cl-gbdt/src/lightgbm/backend''s `%set-feature-names', which this mirrors call-for-call
apart from the field name and the C function it calls."
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

(defun %free-dmatrix (pointer)
  "Free the DMatrix at POINTER via `XGDMatrixFree', signalling `foreign-call-error' when
the library reports failure.

Extracted so `cl-gbdt/src/xgboost/protocol''s `free-dataset' and `predict' each delegate
the call itself, rather than naming `xgd-matrix-free' directly, to this file -- see
policy section 3's Layer 2 delegating to Layer 1. `free-dataset' passes this to
`release-handle' as its release function; `predict' calls it directly on its transient
DMatrix, inside a `handler-case' that turns a failure into a `warn' instead of letting it
propagate over whatever this method is already returning."
  (check-xgb (xgd-matrix-free pointer) "XGDMatrixFree"))

(defun %free-dmatrix-unchecked (pointer)
  "Free the DMatrix at POINTER via `XGDMatrixFree' without checking its returned status.

`make-dataset' calls this from its cleanup path when ownership of a partially built
dataset never transferred to a handle -- a signal already unwinding the stack there must
not be replaced by a status-check failure from this best-effort free, so unlike
`%free-dmatrix' this does not consult `check-xgb' at all."
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
the same guard `make-dataset' applies to `XGDMatrixCreateFromDense', for the same reason:
every later call through this handle would otherwise dereference it blindly."
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
  "Signal `released-handle-error' when any dataset BOOSTER depends on -- its training set,
or any validation set attached via `train''s VALID-SETS -- has already been freed.

`XGBoosterUpdateOneIter' dereferences the DMatrix pointer passed to it directly, and
XGBoost's internal caches keep every DMatrix handle given to `XGBoosterCreate' alive by
pointer, not by anything `XGDMatrixFree' clears when the corresponding dataset is freed.
Calling it after any of those datasets has been freed out from under the booster is a
segfault, not a catchable Lisp condition, so every one of them has to be checked here,
before any foreign call. `booster-training-set' is NIL for a `load-model' booster, which
has no training set and needs no check; `booster-validation-sets' is NIL when `train' was
called with no VALID-SETS."
  (let ((training-set (booster-training-set booster)))
    (when (and training-set (handle-released-p training-set))
      (error 'released-handle-error :object training-set)))
  (dolist (validation-set (booster-validation-sets booster))
    (when (handle-released-p validation-set)
      (error 'released-handle-error :object validation-set))))

(defun %free-booster (pointer)
  "Free the booster at POINTER via `XGBoosterFree', signalling `foreign-call-error' when
the library reports failure.

Extracted the same way `%free-dmatrix' is -- see that function's docstring --
so `cl-gbdt/src/xgboost/protocol''s `free-booster' delegates the call itself to this
file instead of naming `xg-booster-free' directly."
  (check-xgb (xg-booster-free pointer) "XGBoosterFree"))

(defun %free-booster-unchecked (pointer)
  "Free the booster at POINTER via `XGBoosterFree' without checking its returned status.

`train' and `load-model' each call this from their cleanup path when ownership of a
partially built booster never transferred to a handle -- see `%free-dmatrix-unchecked''s
docstring for why a signal already unwinding the stack there must not be replaced by a
status-check failure from this best-effort free."
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

(defun %predict-config-json (predict-type iteration-end)
  "Return the JSON config `XGBoosterPredictFromDMatrix' expects for PREDICT-TYPE (already
mapped by `%predict-type') and ITERATION-END (already resolved by
`%resolve-num-iteration').

`\"strict_shape\":true' always: without it, XGBoost's non-strict mode squeezes away a
single-class model's trailing dimension inconsistently with a multi-class model's --
exactly the assumption `predict' exists to avoid making. With it, `out_shape' and
`out_dim' report a shape this file can trust uniformly across every KIND and class
count."
  (format nil "{\"type\":~D,\"training\":false,\"iteration_begin\":0,~
\"iteration_end\":~D,\"strict_shape\":true}"
          predict-type iteration-end))

(defun %total-element-count (shape-pointer dim)
  "Return the product of the DIM `:uint64' entries at SHAPE-POINTER -- the total element
count `XGBoosterPredictFromDMatrix' reports through its `out_shape'/`out_dim' pair,
however many dimensions the library used. `predict' divides this by the row count to get
the result's column width, rather than assuming a class count -- see that function's
docstring."
  (let ((total 1))
    (dotimes (index dim total)
      (setf total (* total (cffi:mem-aref shape-pointer :uint64 index))))))

(defun %predict-ncol (element-count nrow)
  "Return ELEMENT-COUNT's per-row width for a matrix of NROW rows.

Mirrors `cl-gbdt/src/lightgbm/backend''s function of the same name and purpose. NROW = 0
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

Extracted so `cl-gbdt/src/xgboost/protocol''s `predict' delegates the call itself to this
file instead of naming `xg-booster-predict-from-d-matrix' directly -- see policy section
3's Layer 2 delegating to Layer 1. `predict' still owns interpreting OUT-SHAPE, OUT-DIM
and OUT-RESULT afterward, via `%total-element-count' and `%predict-ncol', and copying
OUT-RESULT's contents out before it returns."
  (check-xgb (xg-booster-predict-from-d-matrix
              booster-pointer dmatrix-pointer config out-shape out-dim out-result)
             "XGBoosterPredictFromDMatrix"))

;;; ---------------------------------------------------------------------------
;;; Persistence

(defun %save-model (pointer filename)
  "Save the booster at POINTER to FILENAME via `XGBoosterSaveModel', signalling
`foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/xgboost/protocol''s `save-model' delegates the call itself to
this file instead of naming `xg-booster-save-model' directly."
  (check-xgb (xg-booster-save-model pointer filename) "XGBoosterSaveModel"))

(defun %load-model (booster-pointer filename)
  "Load a model from FILENAME into the booster at BOOSTER-POINTER via
`XGBoosterLoadModel', signalling `foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/xgboost/protocol''s `load-model' delegates the call itself to
this file instead of naming `xg-booster-load-model' directly."
  (check-xgb (xg-booster-load-model booster-pointer filename) "XGBoosterLoadModel"))

(defun %save-model-to-buffer (pointer config out-len out-dptr)
  "Serialize the booster at POINTER into an XGBoost-owned buffer via
`XGBoosterSaveModelToBuffer', writing OUT-LEN and OUT-DPTR, and signal
`foreign-call-error' when the library reports failure.

Extracted so `cl-gbdt/src/xgboost/protocol''s `model-to-string' delegates the call itself
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
`C_API_FEATURE_IMPORTANCE_GAIN' -- what `cl-gbdt/src/lightgbm/backend' maps `:gain' onto
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
`cl-gbdt/src/lightgbm/backend''s own `feature-importance': `LGBM_BoosterFeatureImportance'
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

Extracted so `cl-gbdt/src/xgboost/protocol''s `feature-importance' delegates the call
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
`%check-xgboost-dataset' cannot be reused: its only caller, `cl-gbdt/src/xgboost/protocol''s
`train', already holds the concrete `xgboost-dataset' class symbol to pass it, and
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
