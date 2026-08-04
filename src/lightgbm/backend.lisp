;;;; backend.lisp --- LightGBM backend: library discovery and all 12 methods of
;;;; the unified API's protocol.

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
                #:lgbm-booster-create
                #:lgbm-booster-add-valid-data
                #:lgbm-booster-update-one-iter
                #:lgbm-booster-free
                #:lgbm-booster-calc-num-predict
                #:lgbm-booster-predict-for-mat
                #:lgbm-booster-save-model
                #:lgbm-booster-save-model-to-string
                #:lgbm-booster-create-from-modelfile
                #:lgbm-booster-feature-importance
                #:lgbm-booster-get-num-feature
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
                #:backend-library-not-found
                #:backend-library-load-failed
                #:missing-foreign-symbols
                #:backend-not-open
                #:foreign-call-error
                #:released-handle-error
                #:wrong-backend-reference)
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
    "LGBM_BoosterGetNumFeature")
  "C function names this backend calls, checked with `probe-foreign-symbols' right
after the library loads.")

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

;;; Handles must be subclassed per backend, not shared. `make-dataset' and `train'
;;; dispatch on the backend, so they would be unambiguous either way -- but
;;; `dataset-num-rows', `predict', `free-dataset' and `free-booster' dispatch on the
;;; HANDLE. A method on the core `dataset' class would be replaced, not specialized,
;;; the moment the XGBoost backend defined its own, and the failure would be a
;;; LightGBM dataset silently answering through XGBoost's C API.

(defclass lightgbm-dataset (dataset) ()
  (:documentation "A dataset held by the LightGBM library."))

(defclass lightgbm-booster (booster) ()
  (:documentation "A booster held by the LightGBM library."))

(defmethod initialize-backend ((backend lightgbm-backend) &key path)
  "Load LightGBM's shared library and record its capabilities on BACKEND.

Discovery order: PATH, then *library-env-var*, then the vendored directory under
*vendor-library-directory*, then CFFI's system library search -- see
`%resolve-library-path' and `%load-system-search' for the exact rules and the
conditions each failure mode signals.

Once a library is loaded, every name in *required-symbols* must resolve via
`probe-foreign-symbols', passed the `cffi:foreign-library' just loaded as
:LIBRARY -- see that function's docstring for the SBCL caveat: it validates
the library argument but, on this platform, cannot actually scope the symbol
search to it -- or this signals `missing-foreign-symbols' -- the
version-mismatch check that function exists for. LightGBM's C API has no
runtime version query, so `backend-version' is left NIL rather than guessed.

`open-backend' only marks a backend open -- and so only calls `close-backend' on
it -- once this method returns normally. So if the symbol probe (or anything
else after the library loads) signals, the library is closed right here before
the condition propagates; otherwise it would stay mapped into the process with
BACKEND dropped and nothing left able to close it."
  (let* ((candidate (%resolve-library-path backend path))
         (library (if candidate
                       (%load-candidate backend candidate)
                       (%load-system-search backend)))
         (succeeded nil))
    (unwind-protect
         (progn
           (setf (%lightgbm-foreign-library backend) library)
           (setf (backend-library-path backend)
                 (namestring (cffi:foreign-library-pathname library)))
           (let ((missing (probe-foreign-symbols *required-symbols* :library library)))
             (when missing
               (error 'missing-foreign-symbols :backend (backend-name backend) :names missing)))
           (setf (backend-version backend) nil)
           (setf succeeded t))
      (unless succeeded
        (handler-case (cffi:close-foreign-library library)
          (error () nil))
        (setf (%lightgbm-foreign-library backend) nil)))
    backend))

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
`LGBM_DatasetSetFeatureNames'.

Every string successfully allocated is freed on any exit, including one signaled
partway through the allocation loop itself -- ALLOCATED tracks exactly how many
of the COUNT slots hold a real `foreign-string-alloc' result, so cleanup never
calls `foreign-string-free' on an uninitialized foreign-object slot."
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

(defun %reference-pointer (backend reference)
  "Return the foreign pointer `make-dataset' should pass as
`LGBM_DatasetCreateFromMat''s `reference' argument for REFERENCE, or a null pointer
when REFERENCE is NIL.

Signals `wrong-backend-reference' when REFERENCE is not a `lightgbm-dataset' --
a dataset built by another backend is just an opaque pointer as far as this C API
is concerned, and handing it to `LGBM_DatasetCreateFromMat' would be undefined
behaviour, not something the library can reject on its own. Otherwise reads
REFERENCE's pointer through `handle-live-pointer', the same guard every other
handle read in this file goes through: a freed or foreign reference handed to a C
function that dereferences it directly is a segfault, not a catchable condition."
  (if (null reference)
      (cffi:null-pointer)
      (progn
        (unless (typep reference 'lightgbm-dataset)
          (error 'wrong-backend-reference
                 :backend (backend-name backend)
                 :given (class-name (class-of reference))))
        (handle-live-pointer reference))))

(defmethod make-dataset ((backend lightgbm-backend) matrix
                          &key label weight group feature-names parameters reference)
  "Build a LightGBM dataset from MATRIX via `LGBM_DatasetCreateFromMat', attaching
LABEL, WEIGHT and GROUP with `LGBM_DatasetSetField' and FEATURE-NAMES with
`LGBM_DatasetSetFeatureNames' when supplied. See the `make-dataset' generic
function's docstring for what each argument means, including REFERENCE.

Signals `foreign-call-error' when dataset creation reports success but writes a
null handle -- a library-contract violation, but one every later call through
this handle would otherwise dereference blindly. Signals `wrong-backend-reference'
when REFERENCE is supplied but is not a `lightgbm-dataset', `released-handle-error'
when it has already been freed, and `backend-not-open' when its backend has since
been closed -- see `%reference-pointer'.

The raw dataset handle exists in C from the moment `LGBM_DatasetCreateFromMat'
returns, but `make-handle' does not take ownership of it until the very end --
attaching LABEL, WEIGHT, GROUP or FEATURE-NAMES can each signal first (a
wrong-length `:label' is the commonest way). OWNED tracks whether `make-handle'
ran; when it did not, the raw dataset is freed here instead of orphaned.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'."
  (%check-backend-open backend)
  (let* ((reference-pointer (%reference-pointer backend reference))
         (parameter-string (%parameter-string parameters))
         (dataset-pointer
           (with-foreign-matrix (data-pointer nrow ncol element-type) matrix
             (let ((data-type (ecase element-type
                                 (double-float +c-api-dtype-float64+)
                                 (single-float +c-api-dtype-float32+))))
               (cffi:with-foreign-string (parameter-cstring parameter-string)
                 (cffi:with-foreign-object (out :pointer)
                   (check-lgbm (lgbm-dataset-create-from-mat
                                data-pointer data-type nrow ncol 1
                                parameter-cstring reference-pointer out)
                               "LGBM_DatasetCreateFromMat")
                   (cffi:mem-ref out :pointer)))))))
    (when (cffi:null-pointer-p dataset-pointer)
      (error 'foreign-call-error
             :function-name "LGBM_DatasetCreateFromMat"
             :code 0
             :message "reported success but returned a null dataset handle"))
    (let ((owned nil))
      (unwind-protect
           (progn
             (when label
               (%set-dataset-field dataset-pointer "label" label :float +c-api-dtype-float32+
                                    (lambda (value) (coerce value 'single-float))))
             (when weight
               (%set-dataset-field dataset-pointer "weight" weight :float +c-api-dtype-float32+
                                    (lambda (value) (coerce value 'single-float))))
             (when group
               (%set-dataset-field dataset-pointer "group" group :int32 +c-api-dtype-int32+
                                    #'round))
             (when feature-names
               (%set-feature-names dataset-pointer feature-names))
             (prog1
                 (make-handle 'lightgbm-dataset dataset-pointer backend :dataset)
               (setf owned t)))
        (unless owned
          (handler-case (lgbm-dataset-free dataset-pointer)
            (error () nil)))))))

(defmethod dataset-num-rows ((dataset lightgbm-dataset))
  "Return DATASET's row count, read via `LGBM_DatasetGetNumData'."
  (let ((pointer (handle-live-pointer dataset)))
    (cffi:with-foreign-object (out :int32)
      (check-lgbm (lgbm-dataset-get-num-data pointer out) "LGBM_DatasetGetNumData")
      (cffi:mem-ref out :int32))))

(defmethod dataset-num-features ((dataset lightgbm-dataset))
  "Return DATASET's feature count, read via `LGBM_DatasetGetNumFeature'."
  (let ((pointer (handle-live-pointer dataset)))
    (cffi:with-foreign-object (out :int32)
      (check-lgbm (lgbm-dataset-get-num-feature pointer out) "LGBM_DatasetGetNumFeature")
      (cffi:mem-ref out :int32))))

(defmethod free-dataset ((dataset lightgbm-dataset))
  "Free DATASET via `LGBM_DatasetFree'. Does nothing if it was already freed.

Unlike every other operation in this file, this does not go through
`handle-live-pointer' and so does not signal `backend-not-open' when DATASET's
backend has already been closed. `free-dataset' runs from `with-dataset''s
`unwind-protect' cleanup form, and a non-local exit is exactly when that cleanup
runs; signalling there would replace whatever condition is already unwinding the
stack instead of letting it propagate. So when the backend is closed, the handle is
instead marked released without calling `LGBM_DatasetFree' -- the shared library may
no longer be mapped into the process, so that call cannot be trusted not to crash --
and a `warn' reports the foreign memory as leaked, since it is genuinely
unreclaimable at that point."
  (if (backend-open-p (handle-backend dataset))
      (release-handle
       dataset
       (lambda (pointer) (check-lgbm (lgbm-dataset-free pointer) "LGBM_DatasetFree")))
      (let ((already-released (handle-released-p dataset)))
        (release-handle dataset (lambda (pointer) (declare (ignore pointer))))
        (unless already-released
          (warn "Freeing a LightGBM dataset after its backend was closed: the foreign ~
                 dataset was not freed and its memory is leaked.")))))

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

(defun %add-valid-data (booster-pointer valid-sets)
  "Attach each dataset in VALID-SETS to BOOSTER-POINTER via
`LGBM_BoosterAddValidData'.

Each VALID-SET's liveness is checked, via `handle-live-pointer', before it is
passed to C -- the same guard `update-one-iteration' applies to a booster's
training set: `LGBM_BoosterAddValidData' dereferences the raw pointer directly,
and one already freed by `free-dataset' is a segfault, not a catchable
condition."
  (dolist (valid-set valid-sets)
    (let ((pointer (handle-live-pointer valid-set)))
      (check-lgbm (lgbm-booster-add-valid-data booster-pointer pointer)
                  "LGBM_BoosterAddValidData"))))

(defun %update-one-iteration (booster-pointer)
  "Advance BOOSTER-POINTER by one boosting iteration via
`LGBM_BoosterUpdateOneIter'. Returns the raw `produced_empty_tree' out
parameter: nonzero when this iteration produced no split."
  (cffi:with-foreign-object (finished :int)
    (check-lgbm (lgbm-booster-update-one-iter booster-pointer finished)
                "LGBM_BoosterUpdateOneIter")
    (cffi:mem-ref finished :int)))

(defmethod train ((backend lightgbm-backend) dataset
                   &key valid-sets (num-rounds 100) parameters)
  "Train a LightGBM booster on DATASET for NUM-ROUNDS boosting iterations.

Builds the booster with `LGBM_BoosterCreate' from PARAMETERS, attaches each of
VALID-SETS with `LGBM_BoosterAddValidData', then drives
`LGBM_BoosterUpdateOneIter' NUM-ROUNDS times. See the `train' generic
function's docstring for what each argument means; NUM-ROUNDS defaults to 100
when not supplied.

The returned booster retains DATASET as its training set and a fresh copy of
VALID-SETS as its validation sets, keeping all of them alive for the booster's
lifetime and letting `update-one-iteration' notice if any is freed out from
under it -- see `%check-booster-datasets-live'. The copy matters: VALID-SETS is
the caller's own list, and `make-handle' would otherwise store that exact list
object rather than a snapshot of it. A caller who destructively removes an
entry from VALID-SETS after `train' returns -- `delete', `(setf (cdr ...))',
reusing the list elsewhere with `nconc' -- would silently remove it from the
booster's view too, since both would be the same cons cells; the dataset
`LGBM_BoosterAddValidData' already attached would then go unchecked by
`%check-booster-datasets-live' even though LightGBM still holds its pointer.
Free the result with `free-booster' or wrap it in `with-booster'.

The raw booster handle exists in C from the moment `LGBM_BoosterCreate' returns,
but `make-handle' does not take ownership of it until the very end -- a stale
VALID-SETS entry or a mid-loop failure can each signal first. OWNED tracks
whether `make-handle' ran; when it did not, the raw booster is freed here
instead of orphaned.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'."
  (%check-backend-open backend)
  (let ((valid-sets (copy-list valid-sets)))
    (let ((booster-pointer (%create-booster (handle-live-pointer dataset)
                                             (%parameter-string parameters))))
      (let ((owned nil))
        (unwind-protect
             (progn
               (%add-valid-data booster-pointer valid-sets)
               (dotimes (round num-rounds)
                 (declare (ignorable round))
                 (%update-one-iteration booster-pointer))
               (prog1
                   (make-handle 'lightgbm-booster booster-pointer backend :booster
                                :training-set dataset :validation-sets valid-sets)
                 (setf owned t)))
          (unless owned
            (handler-case (lgbm-booster-free booster-pointer)
              (error () nil))))))))

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

(defmethod update-one-iteration ((booster lightgbm-booster))
  "Advance BOOSTER by one boosting iteration via `LGBM_BoosterUpdateOneIter'.

Returns false once an iteration produces no further split, per the generic
function's contract. Signals `released-handle-error' when BOOSTER's training set,
or any of its validation sets, has already been freed -- see
`%check-booster-datasets-live'."
  (%check-booster-datasets-live booster)
  (zerop (%update-one-iteration (handle-live-pointer booster))))

(defmethod free-booster ((booster lightgbm-booster))
  "Free BOOSTER via `LGBM_BoosterFree'. Does nothing if it was already freed.

See `free-dataset''s docstring for why this does not signal `backend-not-open' when
BOOSTER's backend has already been closed -- the same `with-booster' cleanup-form
reasoning applies here."
  (if (backend-open-p (handle-backend booster))
      (release-handle
       booster
       (lambda (pointer) (check-lgbm (lgbm-booster-free pointer) "LGBM_BoosterFree")))
      (let ((already-released (handle-released-p booster)))
        (release-handle booster (lambda (pointer) (declare (ignore pointer))))
        (unless already-released
          (warn "Freeing a LightGBM booster after its backend was closed: the foreign ~
                 booster was not freed and its memory is leaked.")))))

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

(defmethod predict ((booster lightgbm-booster) matrix &key (kind :normal) num-iteration)
  "Predict on MATRIX with BOOSTER via `LGBM_BoosterPredictForMat'.

KIND and NUM-ITERATION are as the `predict' generic function documents.
Predictions start from iteration 0 -- the protocol exposes no start-iteration
override.

The output buffer's element count comes from `LGBM_BoosterCalcNumPredict', not
from the row count alone: the row count is only correct for a single-class
objective. The second array dimension is that count divided by the row count,
guarded by `%predict-ncol'. `LGBM_BoosterPredictForMat' also writes its own
element count back through OUT-LEN; this is asserted equal to
`LGBM_BoosterCalcNumPredict''s count rather than trusted silently, since the
buffer was sized from the latter and a mismatch would mean either an
under-filled result or a write past the allocated buffer going unnoticed."
  (let ((pointer (handle-live-pointer booster))
        (predict-type (%predict-type kind))
        (iteration-count (%resolve-num-iteration num-iteration)))
    (with-foreign-matrix (data-pointer nrow ncol element-type) matrix
      (let ((data-type (ecase element-type
                          (double-float +c-api-dtype-float64+)
                          (single-float +c-api-dtype-float32+)))
            (element-count (%calc-num-predict pointer nrow predict-type 0 iteration-count)))
        (let* ((ncol-result (%predict-ncol element-count nrow))
               (result (make-array (list nrow ncol-result) :element-type 'double-float)))
          (cffi:with-foreign-string (parameter-cstring "")
            (cffi:with-foreign-objects ((out-len :int64) (buffer :double element-count))
              (check-lgbm (lgbm-booster-predict-for-mat
                           pointer data-pointer data-type nrow ncol 1 predict-type 0
                           iteration-count parameter-cstring out-len buffer)
                          "LGBM_BoosterPredictForMat")
              (assert (= element-count (cffi:mem-ref out-len :int64)) ()
                      "LGBM_BoosterPredictForMat wrote ~D elements, expected ~D from ~
                       LGBM_BoosterCalcNumPredict"
                      (cffi:mem-ref out-len :int64) element-count)
              (dotimes (row nrow)
                (dotimes (col ncol-result)
                  (setf (aref result row col)
                        (cffi:mem-aref buffer :double (+ (* row ncol-result) col)))))))
          result)))))

;;; ---------------------------------------------------------------------------
;;; Persistence

(defmethod save-model ((booster lightgbm-booster) path &key num-iteration)
  "Save BOOSTER's model to PATH via `LGBM_BoosterSaveModel'.

NUM-ITERATION limits how many trees are saved; nil saves all of them, which
LightGBM spells as 0. Returns PATH."
  (let ((pointer (handle-live-pointer booster)))
    (cffi:with-foreign-string (filename (namestring path))
      (check-lgbm (lgbm-booster-save-model
                   pointer 0 (%resolve-num-iteration num-iteration)
                   +c-api-feature-importance-split+ filename)
                  "LGBM_BoosterSaveModel")))
  path)

(defmethod load-model ((backend lightgbm-backend) path)
  "Load a LightGBM model from PATH via `LGBM_BoosterCreateFromModelfile' and
return a new booster.

The returned booster has no training set -- see the `booster' class'
documentation -- since PATH names a model, not a dataset.

Signals `backend-not-open' before the foreign call when BACKEND is not open --
see `%check-backend-open'."
  (%check-backend-open backend)
  (let ((booster-pointer
          (cffi:with-foreign-string (filename (namestring path))
            (cffi:with-foreign-objects ((out-num-iterations :int) (out :pointer))
              (check-lgbm (lgbm-booster-create-from-modelfile filename out-num-iterations out)
                          "LGBM_BoosterCreateFromModelfile")
              (cffi:mem-ref out :pointer)))))
    (when (cffi:null-pointer-p booster-pointer)
      (error 'foreign-call-error
             :function-name "LGBM_BoosterCreateFromModelfile"
             :code 0
             :message "reported success but returned a null booster handle"))
    (make-handle 'lightgbm-booster booster-pointer backend :booster)))

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

(defmethod model-to-string ((booster lightgbm-booster) &key num-iteration)
  "Return BOOSTER's model as a string via `LGBM_BoosterSaveModelToString'."
  (%save-model-to-string (handle-live-pointer booster) (%resolve-num-iteration num-iteration)))

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

(defmethod feature-importance ((booster lightgbm-booster) &key (kind :split) num-iteration)
  "Return BOOSTER's per-feature importances via `LGBM_BoosterFeatureImportance'.

The result has one entry per feature. The width comes from
`LGBM_BoosterGetNumFeature', which works whether BOOSTER came from `train' or
`load-model' -- unlike a booster's training set, which `load-model' leaves
unbound."
  (let* ((pointer (handle-live-pointer booster))
         (importance-type (%feature-importance-type kind))
         (count (cffi:with-foreign-object (out :int)
                  (check-lgbm (lgbm-booster-get-num-feature pointer out)
                              "LGBM_BoosterGetNumFeature")
                  (cffi:mem-ref out :int)))
         (result (make-array count :element-type 'double-float)))
    (cffi:with-foreign-object (buffer :double count)
      (check-lgbm (lgbm-booster-feature-importance
                   pointer (%resolve-num-iteration num-iteration) importance-type buffer)
                  "LGBM_BoosterFeatureImportance")
      (dotimes (index count)
        (setf (aref result index) (cffi:mem-aref buffer :double index))))
    result))
