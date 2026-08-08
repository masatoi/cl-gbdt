;;;; protocol.lisp --- LightGBM backend, Layer 2: the classes and all fifteen methods of
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
                #:*library-env-var*
                #:*vendor-library-directory*
                #:*vendor-library-pattern*
                #:*default-library-name*
                #:*required-symbols*
                #:*optional-symbols*
                #:*provided-capabilities*)
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
                #:booster-validation-sets
                #:%resolve-best-num-iteration
                #:%reject-best-num-iteration)
  (:import-from #:cl-gbdt/src/conditions
                #:missing-foreign-symbols
                #:foreign-call-error
                #:unsupported-argument
                #:capability-unavailable)
  (:import-from #:cl-gbdt/src/data
                #:with-foreign-matrix
                #:csr-matrix
                #:csr-matrix-indptr
                #:csr-matrix-indices
                #:csr-matrix-values
                #:csr-matrix-num-columns
                #:csr-matrix-num-rows)
  (:import-from #:cl-gbdt/src/training/history
                #:training-report-from-history)
  (:import-from #:cl-gbdt/src/training/early-stopping
                #:train-early-stopping-watcher
                #:observe-iteration
                #:watcher-best-iteration
                #:watcher-best-score
                #:watcher-stopped-p)
  (:import-from #:cl-gbdt/src/library
                #:resolve-and-load-library)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:export #:lightgbm-backend))

(in-package #:cl-gbdt/src/lightgbm/protocol)

;;; ---------------------------------------------------------------------------
;;; Floating-point trap safety
;;;
;;; Every method below that reaches into lib_lightgbm.so -- all thirteen protocol
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
version-mismatch check that function exists for. Only once that required check
has passed does this probe *optional-symbols* via `probe-capabilities' and
record the result on `backend-capabilities' -- unlike a missing required
symbol, a missing optional one never signals; it only makes
`backend-supports-p' answer NIL for the capability that symbol backs.
*provided-capabilities* goes to the same call as :PROVIDED, recording the
capabilities this backend provides unconditionally -- nothing is probed for
them, because the C functions they need are in *required-symbols* and the probe
above has already passed.

LightGBM's C API has no runtime version query, so `backend-version' is left
NIL rather than guessed --
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
               (setf (backend-capabilities backend)
                     (probe-capabilities *optional-symbols*
                                         :provided *provided-capabilities*
                                         :library library))
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
;;; The `:sparse-input' gate

(defun %check-sparse-input (backend)
  "Signal `capability-unavailable' when BACKEND's `:sparse-input' capability reads false.

Policy section 7 requires the operation itself to re-check a capability rather than trusting
the caller to have asked `backend-supports-p' first, so a caller who never asked gets a typed
condition instead of a missing-symbol crash. Both operations this backend gates on
`:sparse-input' call this -- `%dataset-pointer' below, on `make-dataset''s behalf, and
`predict' -- so the two cannot come to disagree about which capability they name or which
backend they blame.

Only a `csr-matrix' argument ever reaches this. A dense matrix needs neither
`LGBM_DatasetCreateFromCSR' nor `LGBM_BoosterPredictForCSR' to exist, and must keep working
on a library that has neither."
  (unless (backend-supports-p backend :sparse-input)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :sparse-input)))

;;; ---------------------------------------------------------------------------
;;; The `:missing-value' gate

(defun %check-missing-value (backend)
  "Signal `capability-unavailable' when BACKEND's `:missing-value' capability reads false.

Which, on this backend, is always: `LightGBM/include/LightGBM/c_api.h' contains no `missing'
anywhere, so there is no argument, key or field through which a caller could name the value
that means missing, and nothing for this backend to declare the capability from. LightGBM's
`use_missing' and `zero_as_missing' are not that: they are policy flags in the parameter
string -- whether missing values are handled at all, and whether a zero counts as one -- and
they are already reachable through `make-dataset''s :PARAMETERS, which is policy section 6's
documented escape hatch for exactly a backend's own vocabulary.

So this signals regardless of the VALUE, a NaN included -- even though a NaN in the matrix is
in fact what LightGBM's own ingestion path already treats as missing, so `:missing' naming
one would be a request the library happens to honour. A capability whose answer depended on
which value was passed could not be stated by `backend-supports-p' at all: it answers about
the backend, not about an argument it has not seen. Accepting the one value that would work
and refusing the rest would make `:missing' mean something different here than on a backend
that takes any of them.

Policy section 7 requires the operation itself to re-check the capability rather than trusting
the caller to have asked `backend-supports-p' first -- the same rule `%check-sparse-input'
above follows for `:sparse-input'. Mirrors `cl-gbdt/src/xgboost/protocol''s function of the
same name, which is the same shape against an answer that is true.

Only a non-NIL :MISSING ever reaches this. NIL means the backend's own default -- what every
caller has always got -- so a caller who passes nothing needs no capability at all, and every
existing call keeps working unchanged."
  (unless (backend-supports-p backend :missing-value)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :missing-value)))

;;; ---------------------------------------------------------------------------
;;; Datasets

(defun %dataset-pointer (backend matrix parameter-string reference-pointer)
  "Return two values: the raw LightGBM dataset pointer built from MATRIX, and the name
of the C function that produced it, for the null-handle check `make-dataset' makes
afterward.

MATRIX is either a `csr-matrix' -- `LGBM_DatasetCreateFromCSR', through
`%create-dataset-from-csr' -- or anything `with-foreign-matrix' accepts --
`LGBM_DatasetCreateFromMat', through `%create-dataset'. PARAMETER-STRING and
REFERENCE-POINTER reach both entry points identically: neither argument means anything
different for a sparse matrix than for a dense one, which is exactly what `make-dataset''s
own contract promises about them.

The `:sparse-input' capability is re-checked on the sparse branch rather than assumed --
`%check-sparse-input' above, which carries the reasoning.

A `defun', not a second `make-dataset' method specialized on `csr-matrix': `make-dataset'
dispatches on BACKEND and MATRIX is its second required argument, so a method pair would
split every one of the shared steps below -- the ownership dance, LABEL, WEIGHT, GROUP,
FEATURE-NAMES -- across two bodies that must not drift, to vary one call. This keeps that
whole method single and varies only the call that actually differs."
  (if (typep matrix 'csr-matrix)
      (progn
        (%check-sparse-input backend)
        (values (%create-dataset-from-csr (csr-matrix-indptr matrix)
                                          (csr-matrix-indices matrix)
                                          (csr-matrix-values matrix)
                                          (csr-matrix-num-columns matrix)
                                          parameter-string reference-pointer)
                "LGBM_DatasetCreateFromCSR"))
      (values (%create-dataset matrix parameter-string reference-pointer)
              "LGBM_DatasetCreateFromMat")))

(defmethod make-dataset ((backend lightgbm-backend) matrix
                          &key label weight group feature-names parameters reference missing)
  "Build a LightGBM dataset from MATRIX -- a dense matrix via `LGBM_DatasetCreateFromMat',
a `csr-matrix' via `LGBM_DatasetCreateFromCSR' -- attaching LABEL, WEIGHT and GROUP with
`LGBM_DatasetSetField' and FEATURE-NAMES with `LGBM_DatasetSetFeatureNames' when supplied.
See the `make-dataset' generic function's docstring for what each argument means,
including REFERENCE, and for what a `csr-matrix' changes about none of them.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `%dataset-pointer', which checks it. Every
other argument behaves identically either way: PARAMETERS and REFERENCE reach the sparse
entry point as the same two C parameters they reach the dense one as, and LABEL, WEIGHT,
GROUP and FEATURE-NAMES are attached to the finished dataset by the calls below, which
never see which entry point built it.

Signals `capability-unavailable' naming `:missing-value' for a non-NIL MISSING, whatever the
value is and whatever form MATRIX takes -- this backend has no C-API route for a missing-value
sentinel at all. See `%check-missing-value' above, which carries the reasoning, including why
a NaN LightGBM would in fact honour is refused with the rest. MISSING NIL, the default, is
this backend's own default and reaches no check: every call that does not name a sentinel
behaves exactly as it did before the argument existed.

Signals `foreign-call-error' when dataset creation reports success but writes a
null handle -- a library-contract violation, but one every later call through
this handle would otherwise dereference blindly. Signals `wrong-backend-reference'
when REFERENCE is supplied but is not a `lightgbm-dataset', `released-handle-error'
when it has already been freed, and `backend-not-open' when its backend has since
been closed -- see `%reference-pointer'.

The raw dataset handle exists in C from the moment the creation call returns, but
`make-handle' does not take ownership of it until the very end -- attaching LABEL,
WEIGHT, GROUP or FEATURE-NAMES can each signal first (a wrong-length `:label' is the
commonest way). OWNED tracks whether `make-handle' ran; when it did not, the raw dataset
is freed here instead of orphaned.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (when missing
      (%check-missing-value backend))
    (let ((reference-pointer (%reference-pointer backend reference 'lightgbm-dataset))
          (parameter-string (%parameter-string parameters)))
      (multiple-value-bind (dataset-pointer function-name)
          (%dataset-pointer backend matrix parameter-string reference-pointer)
        (when (cffi:null-pointer-p dataset-pointer)
          (error 'foreign-call-error
                 :function-name function-name
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
                (error () nil)))))))))

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

(defun %valid-set-name (backend entry)
  "Return the name half of ENTRY, one element of `train''s :VALID-SETS: NIL when ENTRY is
a bare dataset, or ENTRY's car when ENTRY is a (NAME . DATASET) cons and NAME is a string.

Signals `unsupported-argument' naming :VALID-SETS and ENTRY itself when ENTRY is a cons
whose car is not a string -- before any foreign call, and before the dataset half of
ENTRY is checked against `lightgbm-dataset' by `%check-lightgbm-dataset', which runs
afterward on `%valid-set-dataset''s result. That later check is what turns a cons whose
cdr is not this backend's own kind of dataset into `wrong-backend-reference' instead --
a different mistake from this one, and so a different condition."
  (if (consp entry)
      (let ((name (car entry)))
        (unless (stringp name)
          (error 'unsupported-argument
                 :backend (backend-name backend)
                 :argument "train's :valid-sets"
                 :reason (format nil "each element must be a dataset or a ~
                                      (string . dataset) cons; ~S's car is not a string"
                                 entry)))
        name)
      nil))

(defun %valid-set-dataset (entry)
  "Return the dataset half of ENTRY, one element of `train''s :VALID-SETS: ENTRY itself
when it is a bare dataset, or its cdr when it is a (NAME . DATASET) cons.

Does not check that the result is a `lightgbm-dataset' -- `%check-lightgbm-dataset' does
that afterward, on every element `%valid-set-name' has already let through."
  (if (consp entry) (cdr entry) entry))

(defmethod train ((backend lightgbm-backend) dataset
                   &key valid-sets (num-rounds 100) parameters (record-history t)
                        early-stopping)
  "Train a LightGBM booster on DATASET for up to NUM-ROUNDS boosting iterations, and
return it and a `training-report' of the run.

Builds the booster with `LGBM_BoosterCreate' from PARAMETERS, attaches each of
VALID-SETS with `LGBM_BoosterAddValidData', then drives
`LGBM_BoosterUpdateOneIter' NUM-ROUNDS times -- or fewer, when EARLY-STOPPING
ends the run first. See the `train' generic function's docstring for what each
argument means, and for what the secondary value holds; NUM-ROUNDS defaults to
100 when not supplied.

Each VALID-SETS element is either a dataset, whose series carry no name, or a
(NAME . DATASET) cons, where NAME is a string that reaches `training-series-name' for
every series recorded at that dataset's index -- see `%valid-set-name' and
`%valid-set-dataset', which split VALID-SETS into two parallel lists, of datasets and of
names, once at the top of this method; everything below reads the datasets list under
the name VALID-SETS, exactly as before this method accepted names at all. Two entries
may legitimately share one NAME: their index, not their name, is what a caller uses to
tell them apart in the report, so this is accepted rather than rejected as a duplicate.
The training set is never a VALID-SETS entry and is always index 0 with a NIL name.

When RECORD-HISTORY is true -- the default -- this reads the whole evaluation after each
iteration through `%read-evaluation': the same function the `evaluation' method calls, on
the same booster pointer and the same dataset count, which is what keeps the history and
what `evaluation' answers afterward from being able to disagree. `training-report-from-history'
folds the run's worth of them into the report once the loop is done; that fold is backend-
neutral and shared with `cl-gbdt/src/xgboost/protocol''s `train', so what the two backends
record cannot drift apart either. It orders series by the (DATASET-INDEX, METRIC-NAME)
pair's first appearance, which for this backend is `%read-evaluation''s own dataset-major
order, so the report's series arrive in exactly the order `evaluation' reports its entries
in without anything being sorted.

RECORD-HISTORY NIL skips that read entirely -- one `LGBM_BoosterGetEval' call per dataset
per iteration, which is what makes recording cost real wall-clock time (see the `train'
generic's docstring for the measured figures). The loop is then exactly the
`LGBM_BoosterUpdateOneIter' loop this method ran before it recorded anything, and the
report it still returns as its secondary value has an empty series list over the same
NUM-ROUNDS -- `training-report-from-history' over an empty history, the same shape a run
with `metric=none' produces.

A read that fails propagates, freeing the booster through the OWNED dance below rather
than returning a report whose series are shorter than the run: a short series is
indistinguishable from one a buggy loop recorded, and \"one value per iteration\" is the
invariant a caller reading the report relies on.

EARLY-STOPPING watches one of those recorded series and ends the loop once it has stopped
improving -- see the `train' generic function's docstring for the spec's four required
keys, and `train-early-stopping-watcher' for why it cannot be combined with
RECORD-HISTORY NIL.
The watcher sees each iteration's entries exactly as the history records them, off the one
`%read-evaluation' call this loop already makes, so what stopped the run and what the
report shows can never be two different readings. `training-report-num-rounds' needs
nothing extra to report the shortened run: it has counted actual iterations since Phase 3a.

DATASET and every VALID-SETS entry's dataset half are each run through
`%check-lightgbm-dataset' before any foreign call. `train' dispatches on
BACKEND, not on DATASET, so unlike `dataset-num-rows' or `free-dataset' there
is no CLOS specializer here to rule out the wrong kind of handle first --
without this, `handle-live-pointer' would happily hand `LGBM_BoosterCreate' a
booster's own pointer to use as its training-set `DatasetHandle'. Signals
`wrong-backend-reference' when DATASET or a VALID-SETS entry's dataset half is
not a `lightgbm-dataset', and `released-handle-error' or `backend-not-open'
when one is but has already been freed or had its own backend closed. A
VALID-SETS entry that is a cons with a non-string car never reaches this check
at all: `%valid-set-name' signals `unsupported-argument' for it first, which is
the different mistake a malformed name is, kept distinct from a wrong dataset
handle.

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
    (let* ((valid-set-entries (copy-list valid-sets))
           (train-data-pointer
             (%check-lightgbm-dataset backend dataset "train's dataset argument"
                                       'lightgbm-dataset))
           (valid-set-names
             (mapcar (lambda (entry) (%valid-set-name backend entry)) valid-set-entries))
           (valid-sets (mapcar #'%valid-set-dataset valid-set-entries))
           (valid-set-pointers
             (mapcar (lambda (valid-set)
                       (%check-lightgbm-dataset
                        backend valid-set "a train :valid-sets entry" 'lightgbm-dataset))
                     valid-sets))
           (dataset-count (1+ (length valid-set-pointers)))
           (dataset-names (cons nil valid-set-names))
           ;; Built before `LGBM_BoosterCreate', so a malformed spec -- or one asking for
           ;; early stopping with RECORD-HISTORY NIL -- signals with no raw booster handle
           ;; in existence yet to unwind. NIL when EARLY-STOPPING is NIL, which is what the
           ;; loop below tests to decide whether it can end early at all.
           (watcher (train-early-stopping-watcher (backend-name backend) early-stopping
                                                   record-history dataset-names))
           (history '())
           ;; Counted rather than taken from NUM-ROUNDS: the loop below runs zero iterations
           ;; for a negative count, so a caller passing :NUM-ROUNDS -1 gets an untrained
           ;; booster -- as it did before this branch -- and the report must say 0 ran, not
           ;; -1. `training-report-num-rounds' promises how many iterations actually ran, and
           ;; it is also what makes an early-stopped run report its true, shortened length
           ;; with nothing further to do here.
           (completed-rounds 0))
      (let ((booster-pointer
              (%create-booster train-data-pointer (%parameter-string parameters))))
        (let ((owned nil))
          (unwind-protect
               (progn
                 (%add-valid-data booster-pointer valid-set-pointers)
                 ;; ROUND is 1-based, which is the numbering `observe-iteration' answers
                 ;; `watcher-best-iteration' in and the report publishes.
                 (loop :for round :from 1 :to num-rounds
                       :do (%update-one-iteration booster-pointer)
                           (incf completed-rounds)
                           (let ((entries (when record-history
                                            (%read-evaluation booster-pointer dataset-count))))
                             (when record-history
                               (push entries history))
                             (when (and watcher (observe-iteration watcher entries round))
                               (return))))
                 (let* ((best-iteration (and watcher (watcher-best-iteration watcher)))
                        (report (training-report-from-history
                                 (reverse history) completed-rounds dataset-names
                                 :best-iteration best-iteration
                                 :best-score (and watcher (watcher-best-score watcher))
                                 :early-stopped-p (and watcher (watcher-stopped-p watcher)
                                                   (< completed-rounds num-rounds)))))
                   (multiple-value-prog1
                       (values (make-handle 'lightgbm-booster booster-pointer backend :booster
                                            :training-set dataset
                                            :validation-sets valid-sets
                                            :best-iteration best-iteration)
                               report)
                     (setf owned t))))
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
  "Predict on MATRIX with BOOSTER -- a dense matrix via `LGBM_BoosterPredictForMat', a
`csr-matrix' via `LGBM_BoosterPredictForCSR'.

KIND and NUM-ITERATION are as the `predict' generic function documents, NUM-ITERATION's
:BEST resolved by `%resolve-best-num-iteration' before `%resolve-num-iteration' ever
sees it. Predictions start from iteration 0 -- the protocol exposes no start-iteration
override.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `%check-sparse-input', which checks it before
any foreign call. Everything else means exactly what it means for a dense matrix: both
entry points take the same PREDICT-TYPE, the same START-ITERATION/NUM-ITERATION pair and
the same parameter string, and both fill the same buffer in the same row-major order, so
KIND and NUM-ITERATION are honoured identically on either path -- all four KINDs included,
unlike `cl-gbdt/src/xgboost/protocol''s `predict', whose sparse entry point is XGBoost's
inplace prediction and covers only two of them. A `csr-matrix' whose NUM-COLUMNS is not
BOOSTER's own feature count is LightGBM's own mistake to catch, and it does, with a clean
nonzero return this reports as `foreign-call-error' (\"The number of features in data (N) is
not the same as it was in training data (M).\"); nothing here pre-empts that check.

The output buffer's element count comes from `LGBM_BoosterCalcNumPredict', not
from the row count alone: the row count is only correct for a single-class
objective. That count is read the same way for either matrix kind -- it depends on
BOOSTER, the row count, KIND and NUM-ITERATION, and on nothing about how the rows are
laid out. The second array dimension is that count divided by the row count,
guarded by `%predict-ncol'. Whichever entry point ran also writes its own
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
          (iteration-count
            (%resolve-num-iteration
             (%resolve-best-num-iteration booster num-iteration "predict's :num-iteration"))))
      ;; The buffer sizing, the OUT-LEN check and the copy-out are identical for both entry
      ;; points and live here once; CALL is the only thing that differs between them, which
      ;; is exactly how much of this method a `csr-matrix' changes.
      (flet ((predict-into (nrow function-name call)
               (let* ((element-count
                        (%calc-num-predict pointer nrow predict-type 0 iteration-count))
                      (ncol-result (%predict-ncol element-count nrow))
                      (result (make-array (list nrow ncol-result)
                                          :element-type 'double-float)))
                 (cffi:with-foreign-string (parameter-cstring "")
                   (cffi:with-foreign-objects ((out-len :int64)
                                               (buffer :double element-count))
                     (funcall call parameter-cstring out-len buffer)
                     (assert (= element-count (cffi:mem-ref out-len :int64)) ()
                             "~A wrote ~D elements, expected ~D from ~
                              LGBM_BoosterCalcNumPredict"
                             function-name (cffi:mem-ref out-len :int64) element-count)
                     (dotimes (row nrow)
                       (dotimes (col ncol-result)
                         (setf (aref result row col)
                               (cffi:mem-aref buffer :double
                                              (+ (* row ncol-result) col)))))))
                 result)))
        (if (typep matrix 'csr-matrix)
            (progn
              (%check-sparse-input (handle-backend booster))
              (predict-into (csr-matrix-num-rows matrix) "LGBM_BoosterPredictForCSR"
                            (lambda (parameter-cstring out-len buffer)
                              (%predict-for-csr pointer
                                                (csr-matrix-indptr matrix)
                                                (csr-matrix-indices matrix)
                                                (csr-matrix-values matrix)
                                                (csr-matrix-num-columns matrix)
                                                predict-type iteration-count
                                                parameter-cstring out-len buffer))))
            (with-foreign-matrix (data-pointer nrow ncol element-type) matrix
              (predict-into nrow "LGBM_BoosterPredictForMat"
                            (lambda (parameter-cstring out-len buffer)
                              (%predict-for-mat pointer data-pointer
                                                (%data-type element-type) nrow ncol
                                                predict-type iteration-count
                                                parameter-cstring out-len buffer)))))))))

;;; ---------------------------------------------------------------------------
;;; Persistence

(defmethod save-model ((booster lightgbm-booster) path &key num-iteration)
  "Save BOOSTER's model to PATH via `LGBM_BoosterSaveModel'.

NUM-ITERATION limits how many trees are saved, :BEST resolved by
`%resolve-best-num-iteration' first; nil saves all of them, which LightGBM spells as 0.
Returns PATH."
  (with-foreign-float-traps-masked
    (let ((pointer (handle-live-pointer booster))
          (resolved (%resolve-best-num-iteration booster num-iteration
                                                  "save-model's :num-iteration")))
      (cffi:with-foreign-string (filename (namestring path))
        (%save-model pointer (%resolve-num-iteration resolved) filename)))
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
  "Return BOOSTER's model as a string via `LGBM_BoosterSaveModelToString'.

NUM-ITERATION's :BEST is resolved by `%resolve-best-num-iteration' before
`%resolve-num-iteration' ever sees it, exactly as `predict' and `save-model' resolve it."
  (with-foreign-float-traps-masked
    (%save-model-to-string
     (handle-live-pointer booster)
     (%resolve-num-iteration
      (%resolve-best-num-iteration booster num-iteration "model-to-string's :num-iteration")))))

;;; ---------------------------------------------------------------------------
;;; Feature importance

(defmethod feature-importance ((booster lightgbm-booster) &key (kind :split) num-iteration)
  "Return BOOSTER's per-feature importances via `LGBM_BoosterFeatureImportance'.

The result has one entry per feature. The width comes from
`LGBM_BoosterGetNumFeature', which works whether BOOSTER came from `train' or
`load-model' -- unlike a booster's training set, which `load-model' leaves
unbound.

NUM-ITERATION does not accept :BEST, unlike `predict', `save-model' and
`model-to-string' -- `%reject-best-num-iteration' signals `unsupported-argument' for it
rather than letting it reach `LGBM_BoosterFeatureImportance' as raw, uninterpreted data."
  (with-foreign-float-traps-masked
    (let* ((pointer (handle-live-pointer booster))
           (importance-type (%feature-importance-type kind))
           (count (%booster-num-features pointer))
           (result (make-array count :element-type 'double-float))
           (resolved (%reject-best-num-iteration booster num-iteration
                                                  "feature-importance's :num-iteration")))
      (cffi:with-foreign-object (buffer :double count)
        (%feature-importance pointer (%resolve-num-iteration resolved) importance-type
                              buffer)
        (dotimes (index count)
          (setf (aref result index) (cffi:mem-aref buffer :double index))))
      result)))

;;; ---------------------------------------------------------------------------
;;; Evaluation

(defmethod evaluation ((booster lightgbm-booster))
  "Return BOOSTER's evaluation metrics via `%read-evaluation', the pointer-level reader this
backend shares between this method and `train''s per-iteration recording loop, so the two
can never disagree -- see the `evaluation' generic function's docstring for the portable
contract this satisfies.

Reads one `LGBM_BoosterGetEval' result per dataset BOOSTER retains, in the order
`train' attached them, and pairs entry N of each with entry N of the single metric-name
list `LGBM_BoosterGetEvalNames' reports for the whole booster -- LightGBM configures
metrics once per booster, not per dataset, so the names are read once and reused for
every dataset rather than re-read per index.

DATASET-INDEX is passed straight through to `LGBM_BoosterGetEval' as its `data_idx':
this backend's own numbering already is the portable contract's numbering, 0 for the
training set and 1 upward for each `:VALID-SETS' entry in order, so nothing here
renumbers anything. The datasets are counted from BOOSTER's own retained handles rather
than asked of the library, which has no entry point reporting how many are attached; a
`load-model' booster retains none, and is the case that count is 0 for. LightGBM does
answer `data_idx' 0 for such a booster -- with an empty result, confirmed against the
vendored library, since a model file carries no metrics -- but there is no dataset behind
that index for the portable contract to name, so it is not evaluated at all rather than
reported as index 0.

The values are `LGBM_BoosterGetEval''s own doubles, returned unmodified, which is what
the secondary value's `:value-source :library-doubles' says; unlike XGBoost's, nothing
here parses text, so there is no :RAW to keep and no VALUE is ever NIL.

`%check-booster-datasets-live' runs before any foreign call this method makes, including
inside `%read-evaluation': `LGBM_BoosterGetEval' evaluates each attached validation set
through the metric objects built over that dataset's own label and weight arrays, none of
which `LGBM_DatasetFree' clears from the booster, so evaluating after one of them was
freed is a use-after-free rather than a catchable condition -- the identical hazard
`update-one-iteration' guards against with the same call."
  (with-foreign-float-traps-masked
    (%check-booster-datasets-live booster)
    (let ((pointer (handle-live-pointer booster))
          (dataset-count (if (booster-training-set booster)
                             (1+ (length (booster-validation-sets booster)))
                             0)))
      (values (%read-evaluation pointer dataset-count)
              (list :value-source :library-doubles)))))
