;;;; backend.lisp --- XGBoost backend: library discovery and the dataset half of the
;;;; unified API's protocol. Training, inference and persistence follow in a later task.

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
                #:xgd-matrix-num-col)
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
                #:free-dataset)
  (:import-from #:cl-gbdt/src/handle
                #:dataset
                #:booster
                #:make-handle
                #:release-handle
                #:handle-live-pointer
                #:handle-released-p
                #:handle-backend)
  (:import-from #:cl-gbdt/src/conditions
                #:missing-foreign-symbols
                #:backend-not-open
                #:foreign-call-error
                #:unsupported-argument)
  (:import-from #:cl-gbdt/src/data
                #:with-foreign-matrix
                #:write-foreign-sequence)
  (:import-from #:cl-gbdt/src/library
                #:resolve-and-load-library)
  (:import-from #:cl-gbdt/src/foreign
                #:check-foreign-call)
  (:export #:xgboost-backend))

(in-package #:cl-gbdt/src/xgboost/backend)

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

`make-dataset' creates a brand-new handle directly from BACKEND -- there is no existing
handle for the check to route through the way `handle-live-pointer' does for every other
operation in this file, since none exists yet. It calls this first, before touching any
foreign function, so a backend a caller has closed (or never opened) is never reached by
`XGDMatrixCreateFromDense' with a library that may no longer be mapped."
  (unless (backend-open-p backend)
    (error 'backend-not-open :backend (backend-name backend))))

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

(defparameter *default-library-name* "xgboost"
  "Name passed to CFFI's own system library search when no other candidate is found.
Unlike LightGBM's compiled basename, XGBoost's matches its public name directly -- the
file is `libxgboost.so', and the leading `lib' is the platform prefix `:default' adds, not
part of the name given to it.")

(defparameter *required-symbols*
  '("XGBoostVersion"
    "XGBGetLastError"
    "XGDMatrixCreateFromDense"
    "XGDMatrixSetInfoFromInterface"
    "XGDMatrixSetStrFeatureInfo"
    "XGDMatrixFree"
    "XGDMatrixNumRow"
    "XGDMatrixNumCol")
  "C function names this backend calls, checked with `probe-foreign-symbols' right after
the library loads. Training, inference and persistence add more here in a later task.")

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
                 (error 'missing-foreign-symbols :backend (backend-name backend) :names missing)))
             (setf (backend-version backend) (%read-version))
             (setf succeeded t))
        (unless succeeded
          (handler-case (cffi:close-foreign-library library)
            (error () nil))
          (setf (%xgboost-foreign-library backend) nil)))
      backend)))

(defmethod shutdown-backend ((backend xgboost-backend))
  "Close XGBoost's shared library.

`cffi:close-foreign-library' drops cl-gbdt's own reference and, on platforms where the C
loader honors `dlclose' reference counting, may unmap the library; POSIX does not
guarantee an actual unload, so this cannot promise the library's code and data are gone
from the process afterward -- only that cl-gbdt no longer holds it open."
  (let ((library (%xgboost-foreign-library backend)))
    (when library
      (cffi:close-foreign-library library)
      (setf (%xgboost-foreign-library backend) nil)))
  backend)

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
  (%check-backend-open backend)
  (%check-unsupported backend "make-dataset's :reference" reference
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
            (error () nil)))))))

(defmethod dataset-num-rows ((dataset xgboost-dataset))
  "Return DATASET's row count, read via `XGDMatrixNumRow'."
  (let ((pointer (handle-live-pointer dataset)))
    (cffi:with-foreign-object (out :uint64)
      (check-xgb (xgd-matrix-num-row pointer out) "XGDMatrixNumRow")
      (cffi:mem-ref out :uint64))))

(defmethod dataset-num-features ((dataset xgboost-dataset))
  "Return DATASET's feature count, read via `XGDMatrixNumCol'."
  (let ((pointer (handle-live-pointer dataset)))
    (cffi:with-foreign-object (out :uint64)
      (check-xgb (xgd-matrix-num-col pointer out) "XGDMatrixNumCol")
      (cffi:mem-ref out :uint64))))

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
  (if (backend-open-p (handle-backend dataset))
      (release-handle
       dataset
       (lambda (pointer) (check-xgb (xgd-matrix-free pointer) "XGDMatrixFree")))
      (let ((already-released (handle-released-p dataset)))
        (release-handle dataset (lambda (pointer) (declare (ignore pointer))))
        (unless already-released
          (warn "Freeing an XGBoost dataset after its backend was closed: the foreign ~
                 dataset was not freed and its memory is leaked.")))))
