;;;; protocol.lisp --- XGBoost backend, Layer 2: the classes and all fifteen methods of
;;;; the unified API's protocol, each delegating its C calls to
;;;; `cl-gbdt/src/xgboost/native'.
;;;;
;;;; With one Layer 1 exception, `slice-model' at the end of this file: it builds a new
;;;; booster handle, which needs the concrete `xgboost-booster' class defined here, and
;;;; `native.lisp' may not depend on this file to name it. See that section's own comment.

(uiop:define-package #:cl-gbdt/src/xgboost/protocol
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/xgboost/native
                #:%check-backend-open
                #:%check-xgboost-dataset
                #:%check-unsupported
                #:%read-version
                #:*library-env-var*
                #:*vendor-library-directory*
                #:*vendor-library-pattern*
                #:*default-library-name*
                #:*required-symbols*
                #:*optional-symbols*
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
                #:%check-xgboost-booster
                #:evaluate-one-iteration
                #:%slice)
  (:import-from #:cl-gbdt/src/backend
                #:backend
                #:backend-name
                #:backend-library-path
                #:backend-version
                #:backend-capabilities
                #:backend-supports-p
                #:backend-open-p
                #:probe-foreign-symbols
                #:probe-capabilities
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
                #:evaluation
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
                #:foreign-call-error
                #:missing-training-set
                #:capability-unavailable)
  (:import-from #:cl-gbdt/src/data
                #:with-foreign-matrix)
  (:import-from #:cl-gbdt/src/library
                #:resolve-and-load-library)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:import-from #:cl-gbdt/src/version
                #:check-backend-version
                #:*xgboost-version-range*)
  (:export #:xgboost-backend
           #:slice-model))

(in-package #:cl-gbdt/src/xgboost/protocol)

;;; ---------------------------------------------------------------------------
;;; Floating-point trap safety
;;;
;;; Every method below that reaches into libxgboost.so -- all thirteen protocol methods
;;; plus `initialize-backend' (`XGBoostVersion') and `shutdown-backend' (closing the
;;; library can run its own static finalizers) -- wraps its entire body in
;;; `with-foreign-float-traps-masked'. See that macro's docstring in
;;; `cl-gbdt/src/foreign' for why: SBCL enables floating-point traps by default on
;;; x86-64 and not on aarch64, and XGBoost's own numeric code -- confirmed for the
;;; softmax normalization behind a `multi:softprob' prediction -- was written and
;;; tested against the C convention of those traps staying masked. Method-body
;;; granularity, not per-call, so a call added later inside an already-wrapped method
;;; cannot reopen this gap by omission. Every actual C call a method below makes goes
;;; through `cl-gbdt/src/xgboost/native', but the mask is established here, around the
;;; whole method body, not inside that file -- see its own header. `slice-model' near
;;; the end of this file is this file's first non-`defmethod' entry point bound by the
;;; identical rule -- it wraps its own whole body the same way; see its own section below.

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

Only once that required check has passed does this probe *optional-symbols* via
`probe-capabilities' and record the result on `backend-capabilities' -- unlike a missing
required symbol, a missing optional one never signals; it only makes `backend-supports-p'
answer NIL for the capability that symbol backs.

Once `backend-version' is read, `check-backend-version' compares it against
`*xgboost-version-range*' and signals `untested-backend-version' -- a warning, not an
error -- when it falls outside that range's recorded bounds. This is the only backend
this runs for: LightGBM's C API has no version entry point, so `backend-version' stays
NIL there and there is nothing to compare -- see
`cl-gbdt/src/lightgbm/backend''s `initialize-backend'.

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
               (setf (backend-capabilities backend)
                     (probe-capabilities *optional-symbols* :library library))
               (setf (backend-version backend) (%read-version))
               (check-backend-version :xgboost (backend-version backend)
                                       *xgboost-version-range*)
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

(defmethod make-dataset ((backend xgboost-backend) matrix
                          &key label weight group feature-names parameters reference)
  "Build an XGBoost dataset (a DMatrix) from MATRIX via `XGDMatrixCreateFromDense',
attaching LABEL and WEIGHT with `XGDMatrixSetInfoFromInterface', GROUP with
`XGDMatrixSetUIntInfo', and FEATURE-NAMES with `XGDMatrixSetStrFeatureInfo' when
supplied. See the `make-dataset' generic function's docstring for what each argument
means.

REFERENCE and PARAMETERS both signal `unsupported-argument' rather than being silently
dropped: REFERENCE is a LightGBM-only concept -- aligning a new dataset's bin mapper to an
existing one's, which XGBoost has nothing resembling. PARAMETERS is more subtle: the
vendored header (`ffi-spec/xgboost/include/xgboost/c_api.h') documents exactly three keys
for `XGDMatrixCreateFromDense''s config JSON -- `\"missing\"' (fixed by this backend at
*dense-matrix-config-json*, not caller-configurable), `\"nthread\"' and
`\"data_split_mode\"' -- none of which correspond to what a caller moving a working call
from LightGBM actually means by dataset-level PARAMETERS there: binning knobs such as
`max_bin' and `min_data_in_bin'. Forwarding `normalize-parameters''s output into that
config JSON regardless would not raise anything either: confirmed empirically against the
vendored library, `XGDMatrixCreateFromDense' returns success and silently ignores an
unrecognized config key rather than rejecting it, which would just move today's silent
drop one layer deeper, into C, instead of fixing it. Either PARAMETERS or REFERENCE
accepted and discarded here would let a caller move a working `make-dataset' call from
LightGBM to XGBoost and get a dataset that looks fine but was not built the way the caller
asked, which is exactly the failure mode this project keeps finding.

Signals `foreign-call-error' when dataset creation reports success but writes a null
handle -- a library-contract violation, but one every later call through this handle would
otherwise dereference blindly.

The raw DMatrix handle exists in C from the moment `XGDMatrixCreateFromDense' returns, but
`make-handle' does not take ownership of it until the very end -- attaching LABEL, WEIGHT,
GROUP or FEATURE-NAMES can each signal first (a wrong-length `:label' is the commonest
way). OWNED tracks whether `make-handle' ran; when it did not, the raw DMatrix is freed
here instead of orphaned.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (%check-unsupported
     backend "make-dataset's :reference" reference
     "XGBoost has no bin-mapper alignment; :reference is a LightGBM-only concept")
    (%check-unsupported
     backend "make-dataset's :parameters" parameters
     (format nil "XGDMatrixCreateFromDense's config JSON only recognizes missing/nthread/~
                   data_split_mode, none of which are LightGBM's dataset-level binning ~
                   parameters, and the library silently ignores any other key rather than ~
                   rejecting it"))
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
               (when group
                 (%set-group-field dataset-pointer group))
               (when feature-names
                 (%set-feature-names dataset-pointer feature-names))
               (prog1
                   (make-handle 'xgboost-dataset dataset-pointer backend :dataset)
                 (setf owned t)))
          (unless owned
            (handler-case (%free-dmatrix-unchecked dataset-pointer)
              (error () nil))))))))

(defmethod dataset-num-rows ((dataset xgboost-dataset))
  "Return DATASET's row count, read via `XGDMatrixNumRow'."
  (with-foreign-float-traps-masked
    (%dataset-num-rows (handle-live-pointer dataset))))

(defmethod dataset-num-features ((dataset xgboost-dataset))
  "Return DATASET's feature count, read via `XGDMatrixNumCol'."
  (with-foreign-float-traps-masked
    (%dataset-num-features (handle-live-pointer dataset))))

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
        (release-handle dataset (lambda (pointer) (%free-dmatrix pointer)))
        (let ((already-released (handle-released-p dataset)))
          (release-handle dataset (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing an XGBoost dataset after its backend was closed: the foreign ~
                   dataset was not freed and its memory is leaked."))))))

;;; ---------------------------------------------------------------------------
;;; Training

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
             (%check-xgboost-dataset backend dataset "train's dataset argument"
                                      'xgboost-dataset))
           (valid-set-pointers
             (mapcar (lambda (valid-set)
                       (%check-xgboost-dataset backend valid-set "a train :valid-sets entry"
                                                'xgboost-dataset))
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
              (handler-case (%free-booster-unchecked booster-pointer)
                (error () nil)))))))))

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
        (release-handle booster (lambda (pointer) (%free-booster pointer)))
        (let ((already-released (handle-released-p booster)))
          (release-handle booster (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing an XGBoost booster after its backend was closed: the foreign ~
                   booster was not freed and its memory is leaked."))))))

;;; ---------------------------------------------------------------------------
;;; Inference

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
                   (%predict-from-dmatrix
                    booster-pointer dmatrix-pointer config out-shape out-dim out-result)
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
            (handler-case (%free-dmatrix dmatrix-pointer)
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
        (%save-model pointer filename)))
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
                 (%load-model booster-pointer filename))
               (prog1
                   (make-handle 'xgboost-booster booster-pointer backend :booster)
                 (setf owned t)))
          (unless owned
            (handler-case (%free-booster-unchecked booster-pointer)
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
          (%save-model-to-buffer pointer config out-len out-dptr)
          (cffi:foreign-string-to-lisp (cffi:mem-ref out-dptr :pointer)
                                        :count (cffi:mem-ref out-len :uint64)))))))

;;; ---------------------------------------------------------------------------
;;; Feature importance

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
reported score into the column `%feature-score-index' recovers from its feature name.

Signals `unsupported-argument' instead of returning a result at all when
`XGBoosterFeatureScore' reports more than one score per feature -- see
`%check-feature-score-dim'. In practice this is a linear (`gblinear') booster's `:split'
importance on a multi-class model: its scores are a per-class matrix, not one number per
feature, and there is no single value this backend can derive from that matrix without
inventing a reduction XGBoost itself does not define."
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
          (%feature-score
           pointer config out-n-features out-features out-dim out-shape out-scores)
          (%check-feature-score-dim (handle-backend booster) out-dim out-shape)
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

;;; ---------------------------------------------------------------------------
;;; Evaluation

(defmethod evaluation ((booster xgboost-booster))
  "Return BOOSTER's evaluation metrics via `evaluate-one-iteration', this backend's own Layer 1
evaluation function -- see the `evaluation' generic function's docstring for the portable
contract this satisfies.

`XGBoosterEvalOneIter' evaluates whatever DMatrices it is handed and consults nothing the
booster was built with, so the datasets this evaluates are BOOSTER's own retained
handles: its training set first, then each `train' :VALID-SETS entry in the order the
caller supplied them. That is what makes DATASET-INDEX mean the same thing here as it
does on LightGBM, which can only evaluate what training attached -- measured before this
method was written: for one booster, one set of handles and one iteration,
`XGBoosterEvalOneIter' called directly and this path through `evaluate-one-iteration' produce
byte-identical result strings, and both agree with the logloss and error rate computed
independently from `predict' on the same data. A `load-model' booster retains no dataset
at all, which is the case an empty result comes from.

Each dataset is named to `XGBoosterEvalOneIter' by its own decimal index -- \"0\" for the
training set, \"1\" for the first validation set -- because the call requires one name per
DMatrix and builds each result label by joining that name to the metric's name with a
hyphen. `%split-eval-label' takes the label back apart against those same names, which is
the only way to recover the metric name: nothing in the result string alone marks where
one half ends and the other begins. Those names are an argument to a C call, never a
dataset name this API reports -- the caller sees the index, exactly as on LightGBM.

The values are `%parse-eval-result''s reading of `XGBoosterEvalOneIter''s formatted
output, which is what the secondary value's `:value-source :parsed-text' says, and its
`:raw' carries that output unmodified so nothing the library actually wrote is lost to
the parse. A field whose value the parser could not read as a `double-float' -- XGBoost
spells a non-finite one \"inf\" or \"nan\" -- keeps its entry with VALUE NIL rather than
disappearing from the result.

`evaluate-one-iteration' reads BOOSTER and every dataset handed to it through `handle-live-pointer'
before any foreign call, so a freed booster or a freed retained dataset signals
`released-handle-error' from there; unlike `cl-gbdt/src/lightgbm/protocol''s method, this
one needs no separate `%check-booster-datasets-live', since every dataset it evaluates is
one it passes to Layer 1 explicitly and is checked there by name."
  (with-foreign-float-traps-masked
    (let* ((training-set (booster-training-set booster))
           (datasets (if training-set
                         (cons training-set (booster-validation-sets booster))
                         '()))
           ;; `~D', not `princ-to-string': `~D' binds `*print-base*' to 10 itself, so a
           ;; caller who has bound it to something else gets the decimal names this
           ;; method's docstring promises rather than that base's digits.
           (names (loop :for index :below (length datasets) :collect (format nil "~D" index))))
      (multiple-value-bind (raw parsed) (evaluate-one-iteration booster datasets names)
        (values (loop :for (label . value) :in parsed
                      :collect (multiple-value-bind (index metric-name)
                                   (%split-eval-label label names)
                                 (list index metric-name value)))
                (list :value-source :parsed-text :raw raw))))))

;;; ---------------------------------------------------------------------------
;;; Model slicing
;;;
;;; `slice-model' is the one function in this file that is not a protocol method, and the
;;; only Layer 1 entry point that does not live in `cl-gbdt/src/xgboost/native' beside its
;;; siblings `evaluate-one-iteration' and `booster-boosted-rounds'. It is here because it
;;; returns a NEW booster: `make-handle' needs the concrete class `xgboost-booster', which is
;;; defined in this file, and `native.lisp' must not depend on this one (policy section 11).
;;; Every other `make-handle' call in this project is in a `protocol.lisp' for exactly that
;;; reason -- `load-model' above is the closest sibling, and this follows its shape: the
;;; guards and the handle construction here, the foreign call delegated to a `%'-function in
;;; `native.lisp' (`%slice'). See that file's own Model slicing section for the other half.
;;;
;;; Deliberately NOT a generic function in `cl-gbdt/src/protocol'. Section 4's criterion for
;;; the unified API is that both backends can mean the same thing by an operation; LightGBM
;;; has no counterpart to `XGBoosterSlice', so a portable `slice-model' would either signal
;;; on one backend for every caller or emulate -- and emulation is the silent fallback
;;; section 7 forbids. It is published from `cl-gbdt/xgboost' instead, which is what section
;;; 3's Layer 1 and section 11 are for.

(defun slice-model (booster &key (begin 0) end (step 1))
  "Return a new booster holding BOOSTER's layers from BEGIN to END, taken STEP at a time.

The interval is HALF-OPEN, `[BEGIN, END)': END names the first layer left out, so slicing a
ten-round booster with `:begin 0 :end 5' gives five rounds, not six. Measured against the
vendored libxgboost, whose header documents no interval semantics at all; XGBoost's own
rejection of `:begin 5 :end 5' as \"Empty slice is not allowed\" is the same reading from
the other side.

END defaults to NIL, meaning through the last layer, and is passed to `XGBoosterSlice' as
its own 0. NIL rather than 0 in Lisp because a caller writing `:END 0' means \"nothing\",
and silently reading that as \"everything\" is the kind of translation policy section 5
exists to prevent -- so an explicit `:END 0' signals `unsupported-argument' rather than
being forwarded to a C 0 that would mean the opposite. Every other out-of-range request is
XGBoost's own to refuse, and it does, with `foreign-call-error': END past the last layer,
BEGIN below zero, STEP below one, and a STEP that does not divide the interval evenly.

The returned booster belongs to the caller, who frees it with `free-booster'. It is
INDEPENDENT of BOOSTER: `XGBoosterSlice' copies the layers it selects, so freeing BOOSTER
first is legitimate, and the slice keeps predicting the same values afterward -- verified
against the vendored library with BOOSTER and the DMatrix it was trained on both freed. It
therefore retains no parent, exactly as a `load-model' booster retains no training set;
retaining one anyway would make freeing BOOSTER signal `released-handle-error' on correct
code. For the same reason the slice has no training set of its own, so `evaluation' and
`update-one-iteration' on it behave as they do for a `load-model' booster.

Signals `wrong-backend-reference' when BOOSTER was not built by the XGBoost backend,
`released-handle-error' when it has been freed, `backend-not-open' when its backend has
been closed, `capability-unavailable' when the loaded library has no `XGBoosterSlice',
`unsupported-argument' for an explicit `:END 0', and `foreign-call-error' when the slice
itself fails.

The capability is re-checked here rather than assumed: policy section 7 requires the
operation to signal for itself, so a caller who never asked `backend-supports-p' gets a
typed condition instead of a missing-symbol crash. The handle check runs first, before the
capability check, so handing this a LightGBM booster reports the wrong handle rather than
the true-but-irrelevant news that the backend it came from cannot slice."
  (with-foreign-float-traps-masked
    (let ((pointer (%check-xgboost-booster booster "slice-model's booster argument"))
          (backend (handle-backend booster)))
      (unless (backend-supports-p backend :model-slicing)
        (error 'capability-unavailable
               :backend (backend-name backend) :capability :model-slicing))
      (%check-unsupported
       backend "slice-model's :END of 0" (eql end 0)
       (format nil "0 would be an empty slice, which XGBoost rejects outright, but ~
                    XGBoosterSlice's own end_layer 0 means the last layer -- pass NIL for ~
                    that rather than letting the two readings collide"))
      ;; Unlike `load-model' and `train' above, no further foreign call runs between the
      ;; handle appearing in C and `make-handle' taking ownership of it -- `%slice' returns a
      ;; booster that is already complete, and `make-handle' is the very next thing that runs.
      ;; The `owned' unwind-protect dance is still needed, though: `make-handle' itself --
      ;; `make-instance' or finalizer attachment -- can signal, e.g. on `storage-condition',
      ;; and a signal there must not orphan the foreign booster `%slice' already returned.
      (let ((slice-pointer (%slice pointer begin (or end 0) step))
            (owned nil))
        (unwind-protect
             (prog1
                 (make-handle 'xgboost-booster slice-pointer backend :booster)
               (setf owned t))
          (unless owned
            (handler-case (%free-booster-unchecked slice-pointer)
              (error () nil))))))))
