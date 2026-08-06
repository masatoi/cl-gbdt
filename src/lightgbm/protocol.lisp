;;;; protocol.lisp --- LightGBM backend, Layer 2: the classes and all fourteen methods of
;;;; the unified API's protocol, each delegating its C calls to
;;;; `cl-gbdt/src/lightgbm/native'.

(uiop:define-package #:cl-gbdt/src/lightgbm/protocol
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/lightgbm/native
                #:%check-backend-open
                #:%check-lightgbm-dataset
                #:%reference-pointer
                #:%parameter-string
                #:%data-type
                #:%create-dataset
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
                #:%check-booster-datasets-live
                #:%free-booster-unchecked
                #:%free-booster
                #:%predict-type
                #:%resolve-num-iteration
                #:%calc-num-predict
                #:%predict-ncol
                #:%predict-for-mat
                #:%save-model
                #:%create-booster-from-modelfile
                #:%save-model-to-string
                #:%feature-importance-type
                #:%booster-num-features
                #:%feature-importance
                #:*library-env-var*
                #:*vendor-library-directory*
                #:*vendor-library-pattern*
                #:*default-library-name*
                #:*required-symbols*)
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
                #:handle-backend)
  (:import-from #:cl-gbdt/src/conditions
                #:missing-foreign-symbols
                #:foreign-call-error)
  (:import-from #:cl-gbdt/src/data
                #:with-foreign-matrix)
  (:import-from #:cl-gbdt/src/library
                #:resolve-and-load-library)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:export #:lightgbm-backend))

(in-package #:cl-gbdt/src/lightgbm/protocol)

;;; ---------------------------------------------------------------------------
;;; Floating-point trap safety
;;;
;;; Every method below that reaches into lib_lightgbm.so -- all twelve protocol
;;; methods plus `initialize-backend' and `shutdown-backend', which load and unload
;;; the library itself -- wraps its entire body in `with-foreign-float-traps-masked'.
;;; See that macro's docstring in `cl-gbdt/src/foreign' for why, and
;;; `cl-gbdt/src/xgboost/protocol''s identical commentary for the concrete case
;;; (XGBoost's `multi:softprob' softmax) that surfaced this: LightGBM has not tripped
;;; it yet, but its C API is exactly as unprotected against SBCL's x86-64 trap
;;; defaults, so it gets the same treatment rather than waiting for its own CI
;;; failure. Method-body granularity, not per-call, so a call added later inside an
;;; already-wrapped method cannot reopen this gap by omission. Every actual C call a
;;; method below makes goes through `cl-gbdt/src/lightgbm/native', but the mask is
;;; established here, around the whole method body, not inside that file -- see its
;;; own header.

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
*vendor-library-directory*, then CFFI's system library search for
*default-library-name* -- see `resolve-and-load-library' for the exact rules and
the conditions each failure mode signals.

Once a library is loaded, every name in *required-symbols* must resolve via
`probe-foreign-symbols', passed the `cffi:foreign-library' just loaded as
:LIBRARY -- see that function's docstring for the SBCL caveat: it validates
the library argument but, on this platform, cannot actually scope the symbol
search to it -- or this signals `missing-foreign-symbols' -- the
version-mismatch check that function exists for. LightGBM's C API has no
runtime version query, so `backend-version' is left NIL rather than guessed --
and, unlike `cl-gbdt/src/xgboost/protocol''s `initialize-backend', this never
calls `cl-gbdt/src/version''s `check-backend-version': with nothing to read, a
call here could never confirm compatibility, only ever warn on every single
open, which is not a check worth leaving in. See `*lightgbm-version-range*''s
docstring for the fuller explanation of this asymmetry between the backends.

`open-backend' only marks a backend open -- and so only calls `close-backend' on
it -- once this method returns normally. So if the symbol probe (or anything
else after the library loads) signals, the library is closed right here before
the condition propagates; otherwise it would stay mapped into the process with
BACKEND dropped and nothing left able to close it."
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
               (setf (%lightgbm-foreign-library backend) library)
               (setf (backend-library-path backend) library-path)
               (let ((missing (probe-foreign-symbols *required-symbols* :library library)))
                 (when missing
                   (error 'missing-foreign-symbols
                          :backend (backend-name backend) :names missing)))
               (setf (backend-version backend) nil)
               (setf succeeded t))
          (unless succeeded
            (handler-case (cffi:close-foreign-library library)
              (error () nil))
            (setf (%lightgbm-foreign-library backend) nil)))
        backend))))

(defmethod shutdown-backend ((backend lightgbm-backend))
  "Close LightGBM's shared library.

`cffi:close-foreign-library' drops cl-gbdt's own reference and, on platforms
where the C loader honors `dlclose' reference counting, may unmap the library;
POSIX does not guarantee an actual unload, so this cannot promise the library's
code and data are gone from the process afterward -- only that cl-gbdt no
longer holds it open."
  (with-foreign-float-traps-masked
    (let ((library (%lightgbm-foreign-library backend)))
      (when library
        (cffi:close-foreign-library library)
        (setf (%lightgbm-foreign-library backend) nil)))
    backend))

;;; ---------------------------------------------------------------------------
;;; Datasets

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
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (let* ((reference-pointer (%reference-pointer backend reference 'lightgbm-dataset))
           (parameter-string (%parameter-string parameters))
           (dataset-pointer (%create-dataset matrix parameter-string reference-pointer)))
      (when (cffi:null-pointer-p dataset-pointer)
        (error 'foreign-call-error
               :function-name "LGBM_DatasetCreateFromMat"
               :code 0
               :message "reported success but returned a null dataset handle"))
      (let ((owned nil))
        (unwind-protect
             (progn
               (when label
                 (%set-info-field dataset-pointer "label" label))
               (when weight
                 (%set-info-field dataset-pointer "weight" weight))
               (when group
                 (%set-group-field dataset-pointer group))
               (when feature-names
                 (%set-feature-names dataset-pointer feature-names))
               (prog1
                   (make-handle 'lightgbm-dataset dataset-pointer backend :dataset)
                 (setf owned t)))
          (unless owned
            (handler-case (%free-dataset-unchecked dataset-pointer)
              (error () nil))))))))

(defmethod dataset-num-rows ((dataset lightgbm-dataset))
  "Return DATASET's row count, read via `LGBM_DatasetGetNumData'."
  (with-foreign-float-traps-masked
    (%dataset-num-rows (handle-live-pointer dataset))))

(defmethod dataset-num-features ((dataset lightgbm-dataset))
  "Return DATASET's feature count, read via `LGBM_DatasetGetNumFeature'."
  (with-foreign-float-traps-masked
    (%dataset-num-features (handle-live-pointer dataset))))

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
  (with-foreign-float-traps-masked
    (if (backend-open-p (handle-backend dataset))
        (release-handle dataset (lambda (pointer) (%free-dataset pointer)))
        (let ((already-released (handle-released-p dataset)))
          (release-handle dataset (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing a LightGBM dataset after its backend was closed: the foreign ~
                   dataset was not freed and its memory is leaked."))))))

;;; ---------------------------------------------------------------------------
;;; Training

(defmethod train ((backend lightgbm-backend) dataset
                   &key valid-sets (num-rounds 100) parameters)
  "Train a LightGBM booster on DATASET for NUM-ROUNDS boosting iterations.

Builds the booster with `LGBM_BoosterCreate' from PARAMETERS, attaches each of
VALID-SETS with `LGBM_BoosterAddValidData', then drives
`LGBM_BoosterUpdateOneIter' NUM-ROUNDS times. See the `train' generic
function's docstring for what each argument means; NUM-ROUNDS defaults to 100
when not supplied.

DATASET and every entry of VALID-SETS are each run through
`%check-lightgbm-dataset' before any foreign call. `train' dispatches on
BACKEND, not on DATASET, so unlike `dataset-num-rows' or `free-dataset' there
is no CLOS specializer here to rule out the wrong kind of handle first --
without this, `handle-live-pointer' would happily hand `LGBM_BoosterCreate' a
booster's own pointer to use as its training-set `DatasetHandle'. Signals
`wrong-backend-reference' when DATASET or a VALID-SETS entry is not a
`lightgbm-dataset', and `released-handle-error' or `backend-not-open' when one
is but has already been freed or had its own backend closed.

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
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (let* ((valid-sets (copy-list valid-sets))
           (train-data-pointer
             (%check-lightgbm-dataset backend dataset "train's dataset argument"
                                       'lightgbm-dataset))
           (valid-set-pointers
             (mapcar (lambda (valid-set)
                       (%check-lightgbm-dataset
                        backend valid-set "a train :valid-sets entry" 'lightgbm-dataset))
                     valid-sets)))
      (let ((booster-pointer
              (%create-booster train-data-pointer (%parameter-string parameters))))
        (let ((owned nil))
          (unwind-protect
               (progn
                 (%add-valid-data booster-pointer valid-set-pointers)
                 (dotimes (round num-rounds)
                   (declare (ignorable round))
                   (%update-one-iteration booster-pointer))
                 (prog1
                     (make-handle 'lightgbm-booster booster-pointer backend :booster
                                  :training-set dataset :validation-sets valid-sets)
                   (setf owned t)))
            (unless owned
              (handler-case (%free-booster-unchecked booster-pointer)
                (error () nil)))))))))

(defmethod update-one-iteration ((booster lightgbm-booster))
  "Advance BOOSTER by one boosting iteration via `LGBM_BoosterUpdateOneIter'.

Returns false once an iteration produces no further split, per the generic
function's contract. Signals `released-handle-error' when BOOSTER's training set,
or any of its validation sets, has already been freed -- see
`%check-booster-datasets-live'."
  (with-foreign-float-traps-masked
    (%check-booster-datasets-live booster)
    (zerop (%update-one-iteration (handle-live-pointer booster)))))

(defmethod free-booster ((booster lightgbm-booster))
  "Free BOOSTER via `LGBM_BoosterFree'. Does nothing if it was already freed.

See `free-dataset''s docstring for why this does not signal `backend-not-open' when
BOOSTER's backend has already been closed -- the same `with-booster' cleanup-form
reasoning applies here."
  (with-foreign-float-traps-masked
    (if (backend-open-p (handle-backend booster))
        (release-handle booster (lambda (pointer) (%free-booster pointer)))
        (let ((already-released (handle-released-p booster)))
          (release-handle booster (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing a LightGBM booster after its backend was closed: the foreign ~
                   booster was not freed and its memory is leaked."))))))

;;; ---------------------------------------------------------------------------
;;; Inference

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
under-filled result or a write past the allocated buffer going unnoticed.

Deliberately does not scan the result for NaN or infinity -- see
`cl-gbdt/src/xgboost/protocol''s `predict' for the identical reasoning, which
applies here unchanged: `with-foreign-float-traps-masked' restores the C
calling convention around this call, it does not and should not decide what
counts as a valid model output."
  (with-foreign-float-traps-masked
    (let ((pointer (handle-live-pointer booster))
          (predict-type (%predict-type kind))
          (iteration-count (%resolve-num-iteration num-iteration)))
      (with-foreign-matrix (data-pointer nrow ncol element-type) matrix
        (let ((data-type (%data-type element-type))
              (element-count (%calc-num-predict pointer nrow predict-type 0 iteration-count)))
          (let* ((ncol-result (%predict-ncol element-count nrow))
                 (result (make-array (list nrow ncol-result) :element-type 'double-float)))
            (cffi:with-foreign-string (parameter-cstring "")
              (cffi:with-foreign-objects ((out-len :int64) (buffer :double element-count))
                (%predict-for-mat pointer data-pointer data-type nrow ncol predict-type
                                   iteration-count parameter-cstring out-len buffer)
                (assert (= element-count (cffi:mem-ref out-len :int64)) ()
                        "LGBM_BoosterPredictForMat wrote ~D elements, expected ~D from ~
                         LGBM_BoosterCalcNumPredict"
                        (cffi:mem-ref out-len :int64) element-count)
                (dotimes (row nrow)
                  (dotimes (col ncol-result)
                    (setf (aref result row col)
                          (cffi:mem-aref buffer :double (+ (* row ncol-result) col)))))))
            result))))))

;;; ---------------------------------------------------------------------------
;;; Persistence

(defmethod save-model ((booster lightgbm-booster) path &key num-iteration)
  "Save BOOSTER's model to PATH via `LGBM_BoosterSaveModel'.

NUM-ITERATION limits how many trees are saved; nil saves all of them, which
LightGBM spells as 0. Returns PATH."
  (with-foreign-float-traps-masked
    (let ((pointer (handle-live-pointer booster)))
      (cffi:with-foreign-string (filename (namestring path))
        (%save-model pointer (%resolve-num-iteration num-iteration) filename)))
    path))

(defmethod load-model ((backend lightgbm-backend) path)
  "Load a LightGBM model from PATH via `LGBM_BoosterCreateFromModelfile' and
return a new booster.

The returned booster has no training set -- see the `booster' class'
documentation -- since PATH names a model, not a dataset.

The raw booster handle exists in C from the moment `LGBM_BoosterCreateFromModelfile'
returns, but `make-handle' does not take ownership of it until it also succeeds --
mirroring `cl-gbdt/src/xgboost/protocol''s `load-model', which has the identical
OWNED/`unwind-protect' pattern for the same reason: nothing here guarantees
`make-handle' cannot signal, and a raw handle it never took ownership of would
otherwise be orphaned rather than freed.

Signals `backend-not-open' before the foreign call when BACKEND is not open --
see `%check-backend-open'."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (let ((booster-pointer
            (cffi:with-foreign-string (filename (namestring path))
              (cffi:with-foreign-objects ((out-num-iterations :int) (out :pointer))
                (%create-booster-from-modelfile filename out-num-iterations out)
                (cffi:mem-ref out :pointer)))))
      (when (cffi:null-pointer-p booster-pointer)
        (error 'foreign-call-error
               :function-name "LGBM_BoosterCreateFromModelfile"
               :code 0
               :message "reported success but returned a null booster handle"))
      (let ((owned nil))
        (unwind-protect
             (prog1
                 (make-handle 'lightgbm-booster booster-pointer backend :booster)
               (setf owned t))
          (unless owned
            (handler-case (%free-booster-unchecked booster-pointer)
              (error () nil))))))))

(defmethod model-to-string ((booster lightgbm-booster) &key num-iteration)
  "Return BOOSTER's model as a string via `LGBM_BoosterSaveModelToString'."
  (with-foreign-float-traps-masked
    (%save-model-to-string (handle-live-pointer booster)
                            (%resolve-num-iteration num-iteration))))

;;; ---------------------------------------------------------------------------
;;; Feature importance

(defmethod feature-importance ((booster lightgbm-booster) &key (kind :split) num-iteration)
  "Return BOOSTER's per-feature importances via `LGBM_BoosterFeatureImportance'.

The result has one entry per feature. The width comes from
`LGBM_BoosterGetNumFeature', which works whether BOOSTER came from `train' or
`load-model' -- unlike a booster's training set, which `load-model' leaves
unbound."
  (with-foreign-float-traps-masked
    (let* ((pointer (handle-live-pointer booster))
           (importance-type (%feature-importance-type kind))
           (count (%booster-num-features pointer))
           (result (make-array count :element-type 'double-float)))
      (cffi:with-foreign-object (buffer :double count)
        (%feature-importance pointer (%resolve-num-iteration num-iteration) importance-type
                              buffer)
        (dotimes (index count)
          (setf (aref result index) (cffi:mem-aref buffer :double index))))
      result)))
