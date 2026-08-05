;;;; backend.lisp --- XGBoost backend: library discovery, and all 12 methods of the
;;;; unified API's protocol.

(uiop:define-package #:cl-gbdt/src/xgboost/backend
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/xgboost/c-api
                #:xgb-get-last-error
                #:xg-boost-version
                #:xgd-matrix-create-from-dense
                #:xgd-matrix-set-info-from-interface
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
                #:xg-booster-predict-from-d-matrix
                #:xg-booster-save-model
                #:xg-booster-load-model
                #:xg-booster-save-model-to-buffer
                #:xg-booster-feature-score)
  (:import-from #:cl-gbdt/src/xgboost/array-interface
                #:array-interface-json)
  (:import-from #:cl-gbdt/src/backend
                #:backend
                #:backend-name
                #:backend-library-path
                #:backend-version
                #:backend-open-p
                #:probe-foreign-symbols
                #:register-backend
                #:initialize-backend
                #:shutdown-backend)
  (:import-from #:cl-gbdt/src/protocol
                #:make-dataset
                #:dataset-num-rows
                #:dataset-num-features
                #:free-dataset
                #:train
                #:update-one-iteration
                #:predict
                #:save-model
                #:load-model
                #:model-to-string
                #:feature-importance
                #:free-booster)
  (:import-from #:cl-gbdt/src/handle
                #:dataset
                #:booster
                #:make-handle
                #:release-handle
                #:handle-live-pointer
                #:handle-released-p
                #:handle-backend
                #:booster-training-set
                #:booster-validation-sets)
  (:import-from #:cl-gbdt/src/conditions
                #:missing-foreign-symbols
                #:backend-not-open
                #:foreign-call-error
                #:released-handle-error
                #:wrong-backend-reference
                #:unsupported-argument
                #:missing-training-set)
  (:import-from #:cl-gbdt/src/parameters
                #:normalize-parameters)
  (:import-from #:cl-gbdt/src/data
                #:with-foreign-matrix
                #:write-foreign-sequence)
  (:import-from #:cl-gbdt/src/library
                #:resolve-and-load-library)
  (:import-from #:cl-gbdt/src/foreign
                #:check-foreign-call
                #:with-foreign-float-traps-masked)
  (:export #:xgboost-backend))

(in-package #:cl-gbdt/src/xgboost/backend)

;;; ---------------------------------------------------------------------------
;;; Floating-point trap safety
;;;
;;; Every method below that reaches into libxgboost.so -- all twelve protocol methods
;;; plus `initialize-backend' (`XGBoostVersion') and `shutdown-backend' (closing the
;;; library can run its own static finalizers) -- wraps its entire body in
;;; `with-foreign-float-traps-masked'. See that macro's docstring in
;;; `cl-gbdt/src/foreign' for why: SBCL enables floating-point traps by default on
;;; x86-64 and not on aarch64, and XGBoost's own numeric code -- confirmed for the
;;; softmax normalization behind a `multi:softprob' prediction -- was written and
;;; tested against the C convention of those traps staying masked. Method-body
;;; granularity, not per-call, so a call added later inside an already-wrapped method
;;; cannot reopen this gap by omission.

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

(defun %check-xgboost-dataset (backend dataset argument-description)
  "Return DATASET's live foreign pointer, after confirming DATASET is an
`xgboost-dataset'. ARGUMENT-DESCRIPTION names which caller-supplied argument DATASET came
from -- e.g. \"train's dataset argument\" -- for `wrong-backend-reference''s report.

Every caller-supplied dataset argument in this file -- `train''s DATASET and each entry
of `train''s :VALID-SETS -- must pass through here before reaching a foreign call that
expects a `DMatrixHandle'. `handle-live-pointer' alone is not enough: it only guards
against a released handle or a closed backend, and happily returns *any* handle's
pointer regardless of kind, including a booster's -- `train' dispatches on the backend,
not on the handle, so unlike `dataset-num-rows' or `free-dataset' there is no CLOS
specializer already ruling out the wrong kind of handle. A booster's own pointer
reaching `XGBoosterCreate''s DMatrix array is exactly the corruption this check exists
to prevent -- the identical hazard killed the process across several threads on the
LightGBM branch.

Signals `wrong-backend-reference' when DATASET is not an `xgboost-dataset' -- built by a
different backend, or not a dataset at all -- and whatever `handle-live-pointer' signals
otherwise: `released-handle-error' for an already-freed DATASET, `backend-not-open' when
DATASET's own backend has since been closed.

This does not additionally check that DATASET was built by BACKEND specifically, only
that it is *an* `xgboost-dataset' -- see `cl-gbdt/src/lightgbm/backend''s
`%check-lightgbm-dataset', which this mirrors, for why: two backend instances over the
same shared library are a legitimate way for a caller to hold datasets from."
  (unless (typep dataset 'xgboost-dataset)
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
    "XGDMatrixSetStrFeatureInfo"
    "XGDMatrixFree"
    "XGDMatrixNumRow"
    "XGDMatrixNumCol"
    "XGBoosterCreate"
    "XGBoosterFree"
    "XGBoosterSetParam"
    "XGBoosterBoostedRounds"
    "XGBoosterUpdateOneIter"
    "XGBoosterPredictFromDMatrix"
    "XGBoosterSaveModel"
    "XGBoosterLoadModel"
    "XGBoosterSaveModelToBuffer"
    "XGBoosterFeatureScore")
  "C function names this backend calls, checked with `probe-foreign-symbols' right after
the library loads.")

;;; ---------------------------------------------------------------------------
;;; The backend class

(defclass xgboost-backend (backend)
  ((foreign-library :initform nil
                     :accessor %xgboost-foreign-library
                     :documentation "The `cffi:foreign-library' `initialize-backend'
loaded, kept so `shutdown-backend' can close exactly this one."))
  (:documentation "A connection to the XGBoost shared library, implementing cl-gbdt's
unified backend protocol."))

(register-backend :xgboost 'xgboost-backend)

;;; Handles must be subclassed per backend, not shared -- see
;;; `cl-gbdt/src/lightgbm/backend''s identical commentary above `lightgbm-dataset'.
;;; `dataset-num-rows', `predict', `free-dataset' and `free-booster' dispatch on the
;;; HANDLE, so a method on the core `dataset' or `booster' class would be replaced, not
;;; specialized, the moment the other backend defined its own -- an XGBoost dataset would
;;; silently answer through LightGBM's C API, or vice versa.

(defclass xgboost-dataset (dataset) ()
  (:documentation "A dataset (a DMatrix) held by the XGBoost library."))

(defclass xgboost-booster (booster) ()
  (:documentation "A booster held by the XGBoost library."))

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

(defmethod initialize-backend ((backend xgboost-backend) &key path)
  "Load XGBoost's shared library and record its capabilities on BACKEND.

Discovery order: PATH, then *library-env-var*, then the vendored directory under
*vendor-library-directory*, then CFFI's system library search for
*default-library-name* -- see `resolve-and-load-library' for the exact rules and the
conditions each failure mode signals.

Once a library is loaded, every name in *required-symbols* must resolve via
`probe-foreign-symbols', passed the `cffi:foreign-library' just loaded as :LIBRARY -- see
that function's docstring for the SBCL caveat: it validates the library argument but,
on this platform, cannot actually scope the symbol search to it -- or this signals
`missing-foreign-symbols'.

`open-backend' only marks a backend open -- and so only calls `close-backend' on it --
once this method returns normally. So if the symbol probe (or anything else after the
library loads) signals, the library is closed right here before the condition propagates;
otherwise it would stay mapped into the process with BACKEND dropped and nothing left able
to close it."
  (with-foreign-float-traps-masked
    (multiple-value-bind (library library-path)
        (resolve-and-load-library backend :path path
                                           :env-var *library-env-var*
                                           :directory *vendor-library-directory*
                                           :pattern *vendor-library-pattern*
                                           :default-name *default-library-name*)
      (let ((succeeded nil))
        (unwind-protect
             (progn
               (setf (%xgboost-foreign-library backend) library)
               (setf (backend-library-path backend) library-path)
               (let ((missing (probe-foreign-symbols *required-symbols* :library library)))
                 (when missing
                   (error 'missing-foreign-symbols
                          :backend (backend-name backend) :names missing)))
               (setf (backend-version backend) (%read-version))
               (setf succeeded t))
          (unless succeeded
            (handler-case (cffi:close-foreign-library library)
              (error () nil))
            (setf (%xgboost-foreign-library backend) nil)))
        backend))))

(defmethod shutdown-backend ((backend xgboost-backend))
  "Close XGBoost's shared library.

`cffi:close-foreign-library' drops cl-gbdt's own reference and, on platforms where the C
loader honors `dlclose' reference counting, may unmap the library; POSIX does not
guarantee an actual unload, so this cannot promise the library's code and data are gone
from the process afterward -- only that cl-gbdt no longer holds it open."
  (with-foreign-float-traps-masked
    (let ((library (%xgboost-foreign-library backend)))
      (when library
        (cffi:close-foreign-library library)
        (setf (%xgboost-foreign-library backend) nil)))
    backend))

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

(defmethod make-dataset ((backend xgboost-backend) matrix
                          &key label weight group feature-names parameters reference)
  "Build an XGBoost dataset (a DMatrix) from MATRIX via `XGDMatrixCreateFromDense',
attaching LABEL and WEIGHT with `XGDMatrixSetInfoFromInterface' and FEATURE-NAMES with
`XGDMatrixSetStrFeatureInfo' when supplied. See the `make-dataset' generic function's
docstring for what each argument means.

PARAMETERS is accepted, for lambda-list compatibility with the protocol's other backend,
but currently unused: LightGBM's dataset-level PARAMETERS configures its binning (`max_bin'
and friends), which has no XGBoost analogue -- XGBoost's own hyperparameters are booster-
level, set one at a time with `XGBoosterSetParam' once training exists. Unlike REFERENCE
and GROUP below, an empty PARAMETERS is also the overwhelmingly common case, so this stays
silent rather than signalling on every ordinary call; nothing changes silently underneath a
caller who does not pass any.

REFERENCE and GROUP both signal `unsupported-argument' rather than being silently dropped:
REFERENCE is a LightGBM-only concept -- aligning a new dataset's bin mapper to an existing
one's, which XGBoost has nothing resembling -- and GROUP (ranking group sizes) is simply not
yet wired up on this backend. Either one accepted and discarded here would let a caller move
a working `make-dataset' call from LightGBM to XGBoost and get a dataset that looks fine but
was not built the way the caller asked, which is exactly the failure mode this project keeps
finding.

Signals `foreign-call-error' when dataset creation reports success but writes a null
handle -- a library-contract violation, but one every later call through this handle would
otherwise dereference blindly.

The raw DMatrix handle exists in C from the moment `XGDMatrixCreateFromDense' returns, but
`make-handle' does not take ownership of it until the very end -- attaching LABEL, WEIGHT
or FEATURE-NAMES can each signal first (a wrong-length `:label' is the commonest way).
OWNED tracks whether `make-handle' ran; when it did not, the raw DMatrix is freed here
instead of orphaned.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'."
  (declare (ignore parameters))
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (%check-unsupported
     backend "make-dataset's :reference" reference
     "XGBoost has no bin-mapper alignment; :reference is a LightGBM-only concept")
    (%check-unsupported backend "make-dataset's :group" group
                         "ranking group sizes are not yet attached by this backend")
    (let ((dataset-pointer (%create-dmatrix matrix)))
      (when (cffi:null-pointer-p dataset-pointer)
        (error 'foreign-call-error
               :function-name "XGDMatrixCreateFromDense"
               :code 0
               :message "reported success but returned a null dataset handle"))
      (let ((owned nil))
        (unwind-protect
             (progn
               (when label
                 (%set-info-field dataset-pointer "label" label))
               (when weight
                 (%set-info-field dataset-pointer "weight" weight))
               (when feature-names
                 (%set-feature-names dataset-pointer feature-names))
               (prog1
                   (make-handle 'xgboost-dataset dataset-pointer backend :dataset)
                 (setf owned t)))
          (unless owned
            (handler-case (xgd-matrix-free dataset-pointer)
              (error () nil))))))))

(defmethod dataset-num-rows ((dataset xgboost-dataset))
  "Return DATASET's row count, read via `XGDMatrixNumRow'."
  (with-foreign-float-traps-masked
    (let ((pointer (handle-live-pointer dataset)))
      (cffi:with-foreign-object (out :uint64)
        (check-xgb (xgd-matrix-num-row pointer out) "XGDMatrixNumRow")
        (cffi:mem-ref out :uint64)))))

(defmethod dataset-num-features ((dataset xgboost-dataset))
  "Return DATASET's feature count, read via `XGDMatrixNumCol'."
  (with-foreign-float-traps-masked
    (let ((pointer (handle-live-pointer dataset)))
      (cffi:with-foreign-object (out :uint64)
        (check-xgb (xgd-matrix-num-col pointer out) "XGDMatrixNumCol")
        (cffi:mem-ref out :uint64)))))

(defmethod free-dataset ((dataset xgboost-dataset))
  "Free DATASET via `XGDMatrixFree'. Does nothing if it was already freed.

Unlike every other operation in this file, this does not go through `handle-live-pointer'
and so does not signal `backend-not-open' when DATASET's backend has already been closed
-- see `cl-gbdt/src/lightgbm/backend''s `free-dataset' for why: this runs from
`with-dataset''s `unwind-protect' cleanup form, and signalling there would replace whatever
condition is already unwinding the stack instead of letting it propagate. So when the
backend is closed, the handle is instead marked released without calling `XGDMatrixFree' --
the shared library may no longer be mapped into the process, so that call cannot be
trusted not to crash -- and a `warn' reports the foreign memory as leaked, since it is
genuinely unreclaimable at that point."
  (with-foreign-float-traps-masked
    (if (backend-open-p (handle-backend dataset))
        (release-handle
         dataset
         (lambda (pointer) (check-xgb (xgd-matrix-free pointer) "XGDMatrixFree")))
        (let ((already-released (handle-released-p dataset)))
          (release-handle dataset (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing an XGBoost dataset after its backend was closed: the foreign ~
                   dataset was not freed and its memory is leaked."))))))

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

(defmethod train ((backend xgboost-backend) dataset
                   &key valid-sets (num-rounds 100) parameters)
  "Train an XGBoost booster on DATASET for NUM-ROUNDS boosting iterations.

Builds the booster with `XGBoosterCreate' over DATASET and every VALID-SETS entry's
DMatrix handle together -- see `%create-booster' for why XGBoost takes the whole set up
front rather than adding validation data afterward. Applies PARAMETERS one at a time via
`XGBoosterSetParam', then drives `XGBoosterUpdateOneIter' NUM-ROUNDS times. See the
`train' generic function's docstring for what each argument means; NUM-ROUNDS defaults
to 100 when not supplied.

DATASET and every entry of VALID-SETS are each run through `%check-xgboost-dataset'
before any foreign call. `train' dispatches on BACKEND, not on DATASET, so unlike
`dataset-num-rows' or `free-dataset' there is no CLOS specializer here to rule out the
wrong kind of handle first -- without this, `handle-live-pointer' would happily hand
`XGBoosterCreate' a booster's own pointer to use as one of its DMatrix handles. Signals
`wrong-backend-reference' when DATASET or a VALID-SETS entry is not an `xgboost-dataset',
and `released-handle-error' or `backend-not-open' when one is but has already been freed
or had its own backend closed.

The returned booster retains DATASET as its training set and a fresh copy of VALID-SETS
as its validation sets, keeping all of them alive for the booster's lifetime and letting
`update-one-iteration' notice if any is freed out from under it -- see
`%check-booster-datasets-live'. The copy matters: VALID-SETS is the caller's own list,
and `make-handle' would otherwise store that exact list object rather than a snapshot of
it. A caller who destructively removes an entry from VALID-SETS after `train' returns --
`delete', `(setf (cdr ...))', reusing the list elsewhere with `nconc' -- would silently
remove it from the booster's view too, since both would be the same cons cells; the
DMatrix `XGBoosterCreate' already attached would then go unchecked by
`%check-booster-datasets-live' even though XGBoost still holds its pointer -- the same
hazard `cl-gbdt/src/lightgbm/backend''s `train' guards against, for the identical reason.
Free the result with `free-booster' or wrap it in `with-booster'.

The raw booster handle exists in C from the moment `XGBoosterCreate' returns, but
`make-handle' does not take ownership of it until the very end -- a rejected parameter or
a mid-loop failure can each signal first. OWNED tracks whether `make-handle' ran; when it
did not, the raw booster is freed here instead of orphaned.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (let* ((valid-sets (copy-list valid-sets))
           (train-data-pointer
             (%check-xgboost-dataset backend dataset "train's dataset argument"))
           (valid-set-pointers
             (mapcar (lambda (valid-set)
                       (%check-xgboost-dataset backend valid-set "a train :valid-sets entry"))
                     valid-sets)))
      (let ((booster-pointer (%create-booster (cons train-data-pointer valid-set-pointers))))
        (let ((owned nil))
          (unwind-protect
               (progn
                 (%set-parameters booster-pointer parameters)
                 (dotimes (round num-rounds)
                   (declare (ignorable round))
                   (%update-one-iteration booster-pointer train-data-pointer))
                 (prog1
                     (make-handle 'xgboost-booster booster-pointer backend :booster
                                  :training-set dataset :validation-sets valid-sets)
                   (setf owned t)))
            (unless owned
              (handler-case (xg-booster-free booster-pointer)
                (error () nil)))))))))

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

(defmethod update-one-iteration ((booster xgboost-booster))
  "Advance BOOSTER by one boosting iteration via `XGBoosterUpdateOneIter'.

Unlike LightGBM's `LGBM_BoosterUpdateOneIter', which reads the booster's internal
training-set pointer implicitly, XGBoost's version takes the DMatrix handle explicitly,
so this reads it back from `booster-training-set' rather than being able to omit it. A
`load-model' booster's training set is NIL by design -- see the `booster' class'
documentation -- and handing `XGBoosterUpdateOneIter' a null DMatrixHandle would not
come back as a status code the way a bad parameter does: it is a null-pointer dereference
inside XGBoost's own implementation. That case is rejected here, before the foreign call,
for the same reason `%check-booster-datasets-live' exists for the pointers it does check.

XGBoost also reports no `produced_empty_tree'-style signal from this call, unlike
LightGBM -- there is nothing for this backend to report a false return for, so unlike
`cl-gbdt/src/lightgbm/backend''s method of the same name, this always returns true after
a successful call; the generic function's \"returns false when no further split was
possible\" applies only insofar as a backend can report it, which this one cannot.

Signals `released-handle-error' when BOOSTER's training set, or any of its validation
sets, has already been freed -- see `%check-booster-datasets-live'. Signals
`missing-training-set' when BOOSTER has no training set at all -- a `load-model'
booster, which never went through `train' -- since handing `XGBoosterUpdateOneIter' a
null DMatrixHandle in that case is a null-pointer dereference, not something it can
reject with a status code."
  (with-foreign-float-traps-masked
    (%check-booster-datasets-live booster)
    (let ((training-set (booster-training-set booster)))
      (unless training-set
        (error 'missing-training-set :booster booster))
      (%update-one-iteration (handle-live-pointer booster) (handle-live-pointer training-set)))
    t))

(defmethod free-booster ((booster xgboost-booster))
  "Free BOOSTER via `XGBoosterFree'. Does nothing if it was already freed.

See `free-dataset''s docstring for why this does not signal `backend-not-open' when
BOOSTER's backend has already been closed -- the same `with-booster' cleanup-form
reasoning applies here."
  (with-foreign-float-traps-masked
    (if (backend-open-p (handle-backend booster))
        (release-handle
         booster
         (lambda (pointer) (check-xgb (xg-booster-free pointer) "XGBoosterFree")))
        (let ((already-released (handle-released-p booster)))
          (release-handle booster (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing an XGBoost booster after its backend was closed: the foreign ~
                   booster was not freed and its memory is leaked."))))))

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

(defmethod predict ((booster xgboost-booster) matrix &key (kind :normal) num-iteration)
  "Predict on MATRIX with BOOSTER via `XGBoosterPredictFromDMatrix'.

KIND and NUM-ITERATION are as the `predict' generic function documents. Predictions
start from iteration 0 -- the protocol exposes no start-iteration override.

MATRIX is built into a transient DMatrix via `%create-dmatrix' first --
`XGBoosterPredictFromDMatrix' takes a DMatrix handle, unlike LightGBM's
`LGBM_BoosterPredictForMat', which predicts straight off a raw pointer and row/column
counts. The transient DMatrix is freed before this returns, on every exit path, since
nothing else retains it. Its free is checked with `check-xgb', not discarded outright:
a failure there is reported with `warn' rather than an error, matching `free-dataset''s
own reasoning for warning instead of signalling, since raising an error from cleanup
would replace whatever condition is already propagating on an unwinding exit -- but on
the ordinary success path, a failed free still leaks foreign memory and is worth
reporting rather than passing over in silence.

The output buffer's total element count comes from `XGBoosterPredictFromDMatrix''s own
`out_shape'/`out_dim' report, not from the row count alone -- the row count is only
correct for a single-class objective. The second array dimension is that total divided
by the row count, guarded by `%predict-ncol' -- the same derivation
`cl-gbdt/src/lightgbm/backend' uses for its own row-count-alone pitfall, and the one that
tells a three-class `multi:softprob' model's predictions apart from a binary model's.

`out_result' is XGBoost's own memory, valid only until the next call into this booster,
so every element is copied out, coerced from `single-float' to `double-float', before
this returns.

Deliberately does not scan the result for NaN or infinity. `with-foreign-float-traps-masked'
around this method's body stops SBCL from turning an intermediate invalid operation inside
XGBoost's own computation -- e.g. `multi:softprob''s softmax normalization -- into a signal;
it does not, and cannot, stop XGBoost from legitimately returning a non-finite value as a
final result (`:raw' scores in particular are not bounded the way a transformed prediction
is). Rejecting or flagging one here would be a policy this wrapper does not otherwise
impose on any other method's output, invented for this fix rather than driven by a
reported failure -- a caller that cannot tolerate a non-finite prediction should check for
one itself."
  (with-foreign-float-traps-masked
    (let ((booster-pointer (handle-live-pointer booster))
          (predict-type (%predict-type kind))
          (iteration-end (%resolve-num-iteration num-iteration)))
      (with-foreign-matrix (data-pointer nrow ncol element-type) matrix
        (let ((dmatrix-pointer (%create-dmatrix matrix)))
          (when (cffi:null-pointer-p dmatrix-pointer)
            (error 'foreign-call-error
                   :function-name "XGDMatrixCreateFromDense"
                   :code 0
                   :message "reported success but returned a null dataset handle"))
          (unwind-protect
               (cffi:with-foreign-string
                   (config (%predict-config-json predict-type iteration-end))
                 (cffi:with-foreign-objects ((out-shape :pointer) (out-dim :uint64)
                                              (out-result :pointer))
                   (check-xgb (xg-booster-predict-from-d-matrix
                               booster-pointer dmatrix-pointer config
                               out-shape out-dim out-result)
                              "XGBoosterPredictFromDMatrix")
                   (let* ((dim (cffi:mem-ref out-dim :uint64))
                          (shape-pointer (cffi:mem-ref out-shape :pointer))
                          (element-count (%total-element-count shape-pointer dim))
                          (ncol-result (%predict-ncol element-count nrow))
                          (result-buffer (cffi:mem-ref out-result :pointer))
                          (result (make-array (list nrow ncol-result)
                                               :element-type 'double-float)))
                     (dotimes (row nrow)
                       (dotimes (col ncol-result)
                         (setf (aref result row col)
                               (coerce (cffi:mem-aref result-buffer :float
                                                       (+ (* row ncol-result) col))
                                       'double-float))))
                     result)))
            (handler-case (check-xgb (xgd-matrix-free dmatrix-pointer) "XGDMatrixFree")
              (error (condition)
                (warn "Freeing predict's temporary XGBoost dataset failed: the foreign ~
                       dataset was not freed and its memory is leaked. ~A" condition)))))))))

;;; ---------------------------------------------------------------------------
;;; Persistence

(defmethod save-model ((booster xgboost-booster) path &key num-iteration)
  "Save BOOSTER's model to PATH via `XGBoosterSaveModel'.

Signals `unsupported-argument' when NUM-ITERATION is supplied: unlike LightGBM's
`LGBM_BoosterSaveModel', `XGBoosterSaveModel' takes no iteration limit -- it always
saves every boosted round -- and silently ignoring the argument would be exactly the
failure mode `unsupported-argument' exists to prevent, per `%check-unsupported'.

Returns PATH."
  (with-foreign-float-traps-masked
    (%check-unsupported
     (handle-backend booster) "save-model's :num-iteration" num-iteration
     "XGBoosterSaveModel has no iteration limit; every boosted round is saved")
    (let ((pointer (handle-live-pointer booster)))
      (cffi:with-foreign-string (filename (namestring path))
        (check-xgb (xg-booster-save-model pointer filename) "XGBoosterSaveModel")))
    path))

(defmethod load-model ((backend xgboost-backend) path)
  "Load an XGBoost model from PATH and return a new booster.

Unlike LightGBM's `LGBM_BoosterCreateFromModelfile', which allocates the booster and
loads the model in a single call, XGBoost splits the two: `XGBoosterCreate' first builds
a booster with no DMatrix handles at all -- see `%create-booster' -- and only then does
`XGBoosterLoadModel' populate it from PATH.

The returned booster has no training set -- see the `booster' class' documentation --
since PATH names a model, not a dataset.

The raw booster handle exists in C from the moment `XGBoosterCreate' returns, but
`make-handle' does not take ownership of it until `XGBoosterLoadModel' has also
succeeded. OWNED tracks whether `make-handle' ran; when it did not, the raw booster is
freed here instead of orphaned.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (let ((booster-pointer (%create-booster nil)))
      (let ((owned nil))
        (unwind-protect
             (progn
               (cffi:with-foreign-string (filename (namestring path))
                 (check-xgb (xg-booster-load-model booster-pointer filename)
                            "XGBoosterLoadModel"))
               (prog1
                   (make-handle 'xgboost-booster booster-pointer backend :booster)
                 (setf owned t)))
          (unless owned
            (handler-case (xg-booster-free booster-pointer)
              (error () nil))))))))

(defmethod model-to-string ((booster xgboost-booster) &key num-iteration)
  "Return BOOSTER's model as a JSON string via `XGBoosterSaveModelToBuffer'.

Signals `unsupported-argument' when NUM-ITERATION is supplied: `XGBoosterSaveModelToBuffer''s
config JSON has no iteration-limiting key, only `\"format\"' -- see `save-model' for the
same guard on the sibling entry point, and for why silently ignoring it is not an option.

`out_dptr' is XGBoost's own memory, copied out via `foreign-string-to-lisp' with an
explicit `:count' from `out_len' rather than trusted to be null-terminated at the right
place."
  (with-foreign-float-traps-masked
    (%check-unsupported (handle-backend booster) "model-to-string's :num-iteration"
                         num-iteration "XGBoosterSaveModelToBuffer has no iteration limit")
    (let ((pointer (handle-live-pointer booster)))
      (cffi:with-foreign-string (config "{\"format\":\"json\"}")
        (cffi:with-foreign-objects ((out-len :uint64) (out-dptr :pointer))
          (check-xgb (xg-booster-save-model-to-buffer pointer config out-len out-dptr)
                     "XGBoosterSaveModelToBuffer")
          (cffi:foreign-string-to-lisp (cffi:mem-ref out-dptr :pointer)
                                        :count (cffi:mem-ref out-len :uint64)))))))

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

(defmethod feature-importance ((booster xgboost-booster) &key (kind :split) num-iteration)
  "Return BOOSTER's per-feature importances via `XGBoosterFeatureScore'.

Signals `unsupported-argument' when NUM-ITERATION is supplied: `XGBoosterFeatureScore''s
config JSON has no iteration-limiting key, only `importance_type', `feature_map' and
`feature_names' -- honoring it would require slicing the booster first, which this
backend does not do, so this refuses rather than silently scoring every round instead of
the requested subset.

The result has one entry per feature, indexed by column, matching
`cl-gbdt/src/lightgbm/backend''s `feature-importance' -- zero for a feature never used
in a split. `XGBoosterFeatureScore' itself reports the opposite: `out_n_features' and
`out_scores' cover only features that appear in at least one split, so a feature never
split on is absent from its report, not present with a zero -- confirmed directly
against the vendored library and documented upstream. Left as `XGBoosterFeatureScore'
returns it, the result's length would be the number of *used* features, not the
dataset's column count, and its indices would not correspond to column positions --
sparse where LightGBM's equivalent is always dense. This builds a dense vector of
`%booster-num-features' entries instead, initialized to zero, and scatters each
reported score into the column `%feature-score-index' recovers from its feature name."
  (with-foreign-float-traps-masked
    (%check-unsupported (handle-backend booster) "feature-importance's :num-iteration"
                         num-iteration "XGBoosterFeatureScore has no iteration limit")
    (let ((pointer (handle-live-pointer booster))
          (importance-type (%feature-importance-type kind)))
      (cffi:with-foreign-string
          (config (format nil "{\"importance_type\":\"~A\"}" importance-type))
        (cffi:with-foreign-objects ((out-n-features :uint64) (out-features :pointer)
                                     (out-dim :uint64) (out-shape :pointer)
                                     (out-scores :pointer))
          (check-xgb (xg-booster-feature-score
                      pointer config out-n-features out-features out-dim out-shape out-scores)
                     "XGBoosterFeatureScore")
          (let ((used-count (cffi:mem-ref out-n-features :uint64))
                (features-pointer (cffi:mem-ref out-features :pointer))
                (scores-pointer (cffi:mem-ref out-scores :pointer))
                (result (make-array (%booster-num-features pointer)
                                     :element-type 'double-float :initial-element 0.0d0)))
            (dotimes (used-index used-count result)
              (let* ((name (cffi:foreign-string-to-lisp
                            (cffi:mem-aref features-pointer :pointer used-index)))
                     (column (%feature-score-index name)))
                (setf (aref result column)
                      (coerce (cffi:mem-aref scores-pointer :float used-index)
                              'double-float))))))))))
