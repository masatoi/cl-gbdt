;;;; backend.lisp --- LightGBM backend: library discovery and the dataset half of
;;;; the unified API's protocol.
;;;;
;;;; Training, inference and persistence are Tasks 3 and 4; this file only brings
;;;; a dataset into being and reads it back.

(uiop:define-package #:cl-gbdt/src/lightgbm/backend
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/lightgbm/c-api
                #:lgbm-get-last-error
                #:lgbm-dataset-create-from-mat
                #:lgbm-dataset-set-field
                #:lgbm-dataset-set-feature-names
                #:lgbm-dataset-free
                #:lgbm-dataset-get-num-data
                #:lgbm-dataset-get-num-feature
                #:+c-api-dtype-float32+
                #:+c-api-dtype-float64+
                #:+c-api-dtype-int32+)
  (:import-from #:cl-gbdt/src/backend
                #:backend
                #:backend-name
                #:backend-library-path
                #:backend-version
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
                #:make-handle
                #:release-handle
                #:handle-live-pointer)
  (:import-from #:cl-gbdt/src/conditions
                #:backend-library-not-found
                #:backend-library-load-failed
                #:missing-foreign-symbols
                #:foreign-call-error)
  (:import-from #:cl-gbdt/src/parameters
                #:normalize-parameters)
  (:import-from #:cl-gbdt/src/data
                #:with-foreign-matrix)
  (:export #:lightgbm-backend))

(in-package #:cl-gbdt/src/lightgbm/backend)

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
part that is sufficient to check here."
  (if (zerop code)
      code
      (error 'foreign-call-error
             :function-name function-name
             :code code
             :message (%last-error-message))))

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

(defparameter *required-symbols*
  '("LGBM_GetLastError"
    "LGBM_DatasetCreateFromMat"
    "LGBM_DatasetSetField"
    "LGBM_DatasetSetFeatureNames"
    "LGBM_DatasetFree"
    "LGBM_DatasetGetNumData"
    "LGBM_DatasetGetNumFeature")
  "C function names this backend calls, checked with `probe-foreign-symbols' right
after the library loads. Grows in Tasks 3 and 4 as more of the API is wired up.")

(defun %env-library-path ()
  "Return the value of *library-env-var*, or NIL when it is unset or empty."
  (let ((value (uiop:getenv *library-env-var*)))
    (when (and value (plusp (length value)))
      value)))

(defun %vendor-search-path ()
  "Return the merged pathname `directory' is searched with for the vendored
LightGBM library."
  (merge-pathnames *vendor-library-pattern*
                    (asdf:system-relative-pathname "cl-gbdt" *vendor-library-directory*)))

(defun %vendor-library-path ()
  "Return the vendored LightGBM library's path, or NIL when none is present.

Searches *vendor-library-directory* for *vendor-library-pattern*, exactly as
`tests/functional/support.lisp' does for the test suite. `tools/fetch-libs.sh'
vendors at most one file, so the first match is returned without further checking."
  (first (directory (%vendor-search-path))))

(defun %resolve-library-path (backend path)
  "Return the library path `initialize-backend' should load, honoring PATH, then
*library-env-var*, then the vendored directory, in that order. Returns NIL when
none of the three names a candidate; the caller then falls back to CFFI's own
system search.

Signals `backend-library-not-found' when *library-env-var* is set but names a
path that does not exist -- the same strict rule `tests/functional/support.lisp'
follows: a bad override is an error, not a silent fall-through to the next
source."
  (let ((override (%env-library-path)))
    (cond
      (path path)
      (override
       (if (probe-file override)
           override
           (error 'backend-library-not-found
                  :backend (backend-name backend) :searched (list override))))
      (t (%vendor-library-path)))))

(defun %load-candidate (backend path)
  "Load PATH as BACKEND's shared library. Returns the `cffi:foreign-library' CFFI
reports having loaded.

Signals `backend-library-load-failed' when `cffi:load-foreign-library' rejects
PATH, wrapping whatever condition it signaled as :cause."
  (handler-case (cffi:load-foreign-library path)
    (error (condition)
      (error 'backend-library-load-failed
             :backend (backend-name backend) :path path :cause condition))))

(defun %load-system-search (backend)
  "Let CFFI search the system library paths for LightGBM's shared library, whose
compiled basename is `_lightgbm' -- the file is `lib_lightgbm.so', and the
leading `lib' is the platform prefix `:default' adds, not part of the name given
to it. Returns the `cffi:foreign-library' CFFI reports having loaded.

Signals `backend-library-not-found' when nothing turns up anywhere -- PATH,
*library-env-var* and the vendored directory were all already ruled out by the
time this runs."
  (handler-case (cffi:load-foreign-library '(:default "_lightgbm"))
    (error ()
      (error 'backend-library-not-found
             :backend (backend-name backend)
             :searched (list (namestring (%vendor-search-path))
                              "system library search path")))))

;;; ---------------------------------------------------------------------------
;;; The backend class

(defclass lightgbm-backend (backend)
  ((foreign-library :initform nil
                     :accessor %lightgbm-foreign-library
                     :documentation "The `cffi:foreign-library' `initialize-backend'
loaded, kept so `shutdown-backend' can close exactly this one."))
  (:documentation "A connection to the LightGBM shared library, implementing
cl-gbdt's unified backend protocol."))

(register-backend :lightgbm 'lightgbm-backend)

(defmethod initialize-backend ((backend lightgbm-backend) &key path)
  "Load LightGBM's shared library and record its capabilities on BACKEND.

Discovery order: PATH, then *library-env-var*, then the vendored directory under
*vendor-library-directory*, then CFFI's system library search -- see
`%resolve-library-path' and `%load-system-search' for the exact rules and the
conditions each failure mode signals.

Once a library is loaded, every name in *required-symbols* must resolve via
`probe-foreign-symbols' or this signals `missing-foreign-symbols' -- the
version-mismatch check that function exists for. LightGBM's C API has no
runtime version query, so `backend-version' is left NIL rather than guessed."
  (let* ((candidate (%resolve-library-path backend path))
         (library (if candidate
                       (%load-candidate backend candidate)
                       (%load-system-search backend))))
    (setf (%lightgbm-foreign-library backend) library)
    (setf (backend-library-path backend) (namestring (cffi:foreign-library-pathname library))))
  (let ((missing (probe-foreign-symbols *required-symbols*)))
    (when missing
      (error 'missing-foreign-symbols :backend (backend-name backend) :names missing)))
  (setf (backend-version backend) nil)
  backend)

(defmethod shutdown-backend ((backend lightgbm-backend))
  "Close LightGBM's shared library.

`cffi:close-foreign-library' drops cl-gbdt's own reference and, on platforms
where the C loader honors `dlclose' reference counting, may unmap the library;
POSIX does not guarantee an actual unload, so this cannot promise the library's
code and data are gone from the process afterward -- only that cl-gbdt no
longer holds it open."
  (let ((library (%lightgbm-foreign-library backend)))
    (when library
      (cffi:close-foreign-library library)
      (setf (%lightgbm-foreign-library backend) nil)))
  backend)

;;; ---------------------------------------------------------------------------
;;; Datasets

(defun %parameter-string (parameters)
  "Return PARAMETERS, a plist, as the space-separated \"key=value\" string
LightGBM's C API expects. NIL yields the empty string."
  (format nil "~{~A~^ ~}"
          (mapcar (lambda (pair) (format nil "~A=~A" (car pair) (cdr pair)))
                   (normalize-parameters parameters))))

(defun %write-foreign-sequence (pointer cffi-type sequence coercer)
  "Copy SEQUENCE into the foreign array at POINTER, each element passed through
COERCER before being stored as CFFI-TYPE."
  (let ((vector (coerce sequence 'vector)))
    (dotimes (index (length vector))
      (setf (cffi:mem-aref pointer cffi-type index) (funcall coercer (aref vector index))))))

(defun %set-dataset-field (dataset-pointer field-name values cffi-type dtype coercer)
  "Attach the sequence VALUES to DATASET-POINTER's FIELD-NAME via
`LGBM_DatasetSetField'. Each element is coerced through COERCER and stored as
CFFI-TYPE; DTYPE is the matching C_API_DTYPE constant."
  (let ((count (length values)))
    (cffi:with-foreign-object (buffer cffi-type count)
      (%write-foreign-sequence buffer cffi-type values coercer)
      (cffi:with-foreign-string (name field-name)
        (check-lgbm (lgbm-dataset-set-field dataset-pointer name buffer count dtype)
                    "LGBM_DatasetSetField")))))

(defun %set-feature-names (dataset-pointer feature-names)
  "Attach FEATURE-NAMES, a list of strings, to DATASET-POINTER via
`LGBM_DatasetSetFeatureNames'."
  (let ((count (length feature-names)))
    (cffi:with-foreign-object (names :pointer count)
      (loop :for name :in feature-names
            :for index :from 0
            :do (setf (cffi:mem-aref names :pointer index) (cffi:foreign-string-alloc name)))
      (unwind-protect
           (check-lgbm (lgbm-dataset-set-feature-names dataset-pointer names count)
                       "LGBM_DatasetSetFeatureNames")
        (dotimes (index count)
          (cffi:foreign-string-free (cffi:mem-aref names :pointer index)))))))

(defmethod make-dataset ((backend lightgbm-backend) matrix
                          &key label weight group feature-names parameters)
  "Build a LightGBM dataset from MATRIX via `LGBM_DatasetCreateFromMat', attaching
LABEL, WEIGHT and GROUP with `LGBM_DatasetSetField' and FEATURE-NAMES with
`LGBM_DatasetSetFeatureNames' when supplied. See the `make-dataset' generic
function's docstring for what each argument means.

Signals `foreign-call-error' when dataset creation reports success but writes a
null handle -- a library-contract violation, but one every later call through
this handle would otherwise dereference blindly."
  (let* ((parameter-string (%parameter-string parameters))
         (dataset-pointer
           (with-foreign-matrix (data-pointer nrow ncol element-type) matrix
             (let ((data-type (ecase element-type
                                 (double-float +c-api-dtype-float64+)
                                 (single-float +c-api-dtype-float32+))))
               (cffi:with-foreign-string (parameter-cstring parameter-string)
                 (cffi:with-foreign-object (out :pointer)
                   (check-lgbm (lgbm-dataset-create-from-mat
                                data-pointer data-type nrow ncol 1
                                parameter-cstring (cffi:null-pointer) out)
                               "LGBM_DatasetCreateFromMat")
                   (cffi:mem-ref out :pointer)))))))
    (when (cffi:null-pointer-p dataset-pointer)
      (error 'foreign-call-error
             :function-name "LGBM_DatasetCreateFromMat"
             :code 0
             :message "reported success but returned a null dataset handle"))
    (when label
      (%set-dataset-field dataset-pointer "label" label :float +c-api-dtype-float32+
                           (lambda (value) (coerce value 'single-float))))
    (when weight
      (%set-dataset-field dataset-pointer "weight" weight :float +c-api-dtype-float32+
                           (lambda (value) (coerce value 'single-float))))
    (when group
      (%set-dataset-field dataset-pointer "group" group :int32 +c-api-dtype-int32+ #'round))
    (when feature-names
      (%set-feature-names dataset-pointer feature-names))
    (make-handle 'dataset dataset-pointer (backend-name backend) :dataset)))

;;; `dataset' is the single handle class shared by every backend (see
;;; `cl-gbdt/src/handle'); this plan is LightGBM-only, so specializing directly on
;;; it is unambiguous today. A second backend implementing these same three
;;; methods on the same class would need its own dataset subclass to disambiguate
;;; -- deliberately left for whichever plan adds XGBoost.

(defmethod dataset-num-rows ((dataset dataset))
  "Return DATASET's row count, read via `LGBM_DatasetGetNumData'."
  (let ((pointer (handle-live-pointer dataset)))
    (cffi:with-foreign-object (out :int32)
      (check-lgbm (lgbm-dataset-get-num-data pointer out) "LGBM_DatasetGetNumData")
      (cffi:mem-ref out :int32))))

(defmethod dataset-num-features ((dataset dataset))
  "Return DATASET's feature count, read via `LGBM_DatasetGetNumFeature'."
  (let ((pointer (handle-live-pointer dataset)))
    (cffi:with-foreign-object (out :int32)
      (check-lgbm (lgbm-dataset-get-num-feature pointer out) "LGBM_DatasetGetNumFeature")
      (cffi:mem-ref out :int32))))

(defmethod free-dataset ((dataset dataset))
  "Free DATASET via `LGBM_DatasetFree'. Does nothing if it was already freed."
  (release-handle
   dataset
   (lambda (pointer) (check-lgbm (lgbm-dataset-free pointer) "LGBM_DatasetFree"))))
