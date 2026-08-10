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
                #:*provided-capabilities*
                #:%create-dmatrix
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
                #:%training-scores
                #:%train-one-iteration-custom
                #:%save-model
                #:%load-model
                #:%save-model-to-buffer
                #:%feature-importance-type
                #:%booster-num-features
                #:%feature-score-index
                #:%check-feature-score-dim
                #:%feature-score
                #:%check-xgboost-booster
                #:%read-evaluation
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
                #:booster-validation-sets
                #:%resolve-best-num-iteration
                #:%reject-best-num-iteration)
  (:import-from #:cl-gbdt/src/conditions
                #:missing-foreign-symbols
                #:foreign-call-error
                #:missing-training-set
                #:unsupported-argument
                #:capability-unavailable)
  (:import-from #:cl-gbdt/src/data
                #:csr-matrix
                #:csr-matrix-indptr
                #:csr-matrix-indices
                #:csr-matrix-values
                #:csr-matrix-num-columns
                #:csr-matrix-num-rows)
  (:import-from #:cl-gbdt/src/config/categorical-features
                #:categorical-feature-types)
  ;; `check-objective-result' only. `cl-gbdt/src/config/objective' also exports
  ;; `objective-parameters', which rewrites a parameter plist's `objective' entry to "none" --
  ;; that is LightGBM's alone, forced there because `LGBM_BoosterUpdateOneIterCustom' refuses
  ;; to run while the booster holds an objective function. `XGBoosterTrainOneIter' accepts a
  ;; custom update with any objective set, so this backend rewrites nothing and importing the
  ;; second symbol would leave an unused name suggesting otherwise.
  (:import-from #:cl-gbdt/src/config/objective
                #:check-objective-result)
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
answer NIL for the capability that symbol backs. *provided-capabilities* goes to the same
call as :PROVIDED, recording the capabilities this backend provides unconditionally --
nothing is probed for them, because the C functions they need are in *required-symbols* and
the probe above has already passed.

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
                     (probe-capabilities *optional-symbols*
                                         :provided *provided-capabilities*
                                         :library library))
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
;;; The `:sparse-input' gate

(defun %check-sparse-input (backend)
  "Signal `capability-unavailable' when BACKEND's `:sparse-input' capability reads false.

Policy section 7 requires the operation itself to re-check a capability rather than trusting
the caller to have asked `backend-supports-p' first, so a caller who never asked gets a typed
condition instead of a missing-symbol crash -- the same rule `slice-model' at the end of this
file follows for `:model-slicing'. Both operations this backend gates on `:sparse-input' call
this -- `%dataset-pointer' below, on `make-dataset''s behalf, and `predict' -- so the two
cannot come to disagree about which capability they name or which backend they blame.
Mirrors `cl-gbdt/src/lightgbm/protocol''s function of the same name.

Only a `csr-matrix' argument ever reaches this. A dense matrix needs neither
`XGDMatrixCreateFromCSR' nor `XGBoosterPredictFromCSR' to exist, and must keep working on a
library that has neither."
  (unless (backend-supports-p backend :sparse-input)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :sparse-input)))

;;; ---------------------------------------------------------------------------
;;; The `:missing-value' gate

(defun %check-missing-value (backend)
  "Signal `capability-unavailable' when BACKEND's `:missing-value' capability reads false.

Policy section 7 requires the operation itself to re-check a capability rather than trusting
the caller to have asked `backend-supports-p' first -- the same rule `%check-sparse-input'
above follows for `:sparse-input'. This backend answers true unconditionally, which does not
make the check redundant: it is what keeps the two backends' code saying the same thing, so
`make-dataset' here and `make-dataset' in `cl-gbdt/src/lightgbm/protocol' gate the argument
identically and neither has to be read to know what the other does. Mirrors that file's
function of the same name.

Only a non-NIL :MISSING ever reaches this. NIL means the backend's own default sentinel --
what every caller has always got -- so a caller who passes nothing needs no capability at
all."
  (unless (backend-supports-p backend :missing-value)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :missing-value)))

;;; ---------------------------------------------------------------------------
;;; The `:categorical-features' gate

(defun %check-categorical-features (backend)
  "Signal `capability-unavailable' when BACKEND's `:categorical-features' capability reads
false.

Policy section 7 requires the operation itself to re-check a capability rather than trusting
the caller to have asked `backend-supports-p' first -- the same rule `%check-sparse-input' and
`%check-missing-value' above follow for their own. This backend answers true unconditionally,
which does not make the check redundant: it is what keeps the two backends' code saying the
same thing, so `make-dataset' here and `make-dataset' in `cl-gbdt/src/lightgbm/protocol' gate
the argument identically and neither has to be read to know what the other does.

Only a non-NIL :CATEGORICAL-FEATURES ever reaches this. NIL means what every caller has always
got -- no feature-type vector attached at all, every column read as a quantity -- so a caller
who passes nothing needs no capability."
  (unless (backend-supports-p backend :categorical-features)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :categorical-features)))

;;; ---------------------------------------------------------------------------
;;; The `:custom-objective' gate

(defun %check-custom-objective (backend objective)
  "Signal `capability-unavailable' when OBJECTIVE is non-NIL and BACKEND does not provide
`:custom-objective', and `unsupported-argument' when OBJECTIVE is non-NIL and is not a
function.

Only a non-NIL OBJECTIVE ever reaches either error: NIL means what every caller has always
got, the library computing its own gradient, so a caller who passes nothing needs no
capability and cannot fail the type check either.
Policy section 7 requires the operation itself to re-check rather than trusting the caller to
have asked `backend-supports-p' first -- the same rule `%check-sparse-input',
`%check-missing-value' and `%check-categorical-features' above follow for their own. Mirrors
`cl-gbdt/src/lightgbm/protocol''s function of the same name.

Like LightGBM's, this backend's answer is PROBED rather than declared: `XGBoosterTrainOneIter'
-- the entry point `train''s custom loop makes its update through -- is named in
`*optional-symbols*' rather than `*required-symbols*', so an XGBoost too old to export it
opens normally and reads false here. See that variable's own docstring for why the entry
belongs there and not in `*provided-capabilities*', and for why one C name covers this where
LightGBM's entry needs three.

Refusing rather than falling back is what keeps a caller from silently getting a run boosted
against `reg:squarederror' when they asked for their own loss, which is the silent fallback
policy section 7 forbids. That matters more here than on LightGBM: this backend does not
rewrite PARAMETERS, so an ignored OBJECTIVE would leave a perfectly ordinary configured
objective training a perfectly ordinary model, with nothing about the result to show the
caller's function was never called.

The type check is here, beside the capability check, rather than left to the `funcall' in
`train''s loop. By then the booster handle exists and one iteration's scores have already
been read out of the library -- at the cost of a full prediction pass, on this backend -- so
`:objective 42' would surface as SBCL's own untyped `type-error' from mid-loop, naming
neither the argument nor the backend, where every other malformed argument here signals
`unsupported-argument' before any foreign call. `functionp' rather than a `function' type
declaration: a symbol naming a function is NOT accepted, since `funcall' would resolve it
against whatever global definition happened to be in force at each iteration rather than
against what the caller passed. Mirrors `cl-gbdt/src/lightgbm/protocol''s guard exactly, down
to the ARGUMENT string, so the two backends refuse the same value with the same report."
  (when objective
    (unless (backend-supports-p backend :custom-objective)
      (error 'capability-unavailable
             :backend (backend-name backend) :capability :custom-objective))
    (unless (functionp objective)
      (error 'unsupported-argument
             :backend (backend-name backend)
             :argument "train's :objective"
             :reason (format nil "the custom objective must be a function of one argument, ~
                                  or NIL for the library's own gradient -- got ~S"
                             objective)))))

;;; ---------------------------------------------------------------------------
;;; The `:custom-evaluation' gate

(defun %check-custom-evaluation (backend evaluation record-history)
  "Signal `capability-unavailable' when EVALUATION is non-NIL and BACKEND does not provide
`:custom-evaluation', and `unsupported-argument' when EVALUATION is non-NIL and either
RECORD-HISTORY is NIL or EVALUATION is not a function.

Only a non-NIL EVALUATION ever reaches any of the three: NIL means what every caller has
always got, the library's own metrics and nothing else, so a caller who passes nothing needs
no capability and cannot fail either check. Policy section 7 requires the operation itself to
re-check rather than trusting the caller to have asked `backend-supports-p' first -- the same
rule `%check-sparse-input', `%check-missing-value', `%check-categorical-features' and
`%check-custom-objective' above follow for their own. Mirrors
`cl-gbdt/src/lightgbm/protocol''s function of the same name, down to the ARGUMENT string, so
the two backends refuse the same value with the same report.

TODAY ONLY THE FIRST OF THE THREE IS REACHABLE HERE, and that is the honest state rather than
a stub: this backend names `:custom-evaluation' in NEITHER `*provided-capabilities*' NOR
`*optional-symbols*', so `backend-supports-p' reads a capability missing from the plist as
unavailable and every non-NIL EVALUATION is refused before the RECORD-HISTORY and `functionp'
checks can run. That is the same shape LightGBM's `:missing-value' answer has -- the ABSENCE
of a declaration rather than a declaration of absence. The two later checks are written now,
in the order and wording LightGBM's are, so that the commit which declares the capability
makes them live rather than having to invent them.

The argument is accepted by `train''s lambda list rather than being absent from it: `train'
is one generic function, so a method that did not take the keyword at all would answer a
caller who named it with SBCL's `unknown-keyword-argument' rather than with the typed
`capability-unavailable' every other unavailable capability on this backend answers with."
  (when evaluation
    (unless (backend-supports-p backend :custom-evaluation)
      (error 'capability-unavailable
             :backend (backend-name backend) :capability :custom-evaluation))
    (unless record-history
      (error 'unsupported-argument
             :backend (backend-name backend)
             :argument "train's :evaluation"
             :reason (format nil "a custom metric is recorded per iteration, which ~
                                  :record-history NIL skips; pass :record-history T, or ~
                                  drop :evaluation")))
    (unless (functionp evaluation)
      (error 'unsupported-argument
             :backend (backend-name backend)
             :argument "train's :evaluation"
             :reason (format nil "the custom metric must be a function of two arguments, ~
                                  or NIL for the library's own metrics only -- got ~S"
                             evaluation)))))

;;; ---------------------------------------------------------------------------
;;; Datasets

(defun %creation-function-name (matrix)
  "Return the name of the C entry point a DMatrix would be built from MATRIX with:
`XGDMatrixCreateFromCSR' for a `csr-matrix', `XGDMatrixCreateFromDense' for anything
`with-foreign-matrix' accepts.

Separate from `%dataset-pointer', which returns the same string alongside the pointer it
built, because `make-dataset' refuses :PARAMETERS before any pointer exists and its refusal
has to name the call the caller's own arguments would have reached. Telling a caller who
passed a `csr-matrix' about `XGDMatrixCreateFromDense''s config JSON names a function that
call was never going to make."
  (if (typep matrix 'csr-matrix) "XGDMatrixCreateFromCSR" "XGDMatrixCreateFromDense"))

(defun %dataset-pointer (backend matrix missing)
  "Return two values: the raw DMatrix pointer built from MATRIX, and the name of the C
function that produced it, for the null-handle check `make-dataset' makes afterward.

MATRIX is either a `csr-matrix' -- `XGDMatrixCreateFromCSR', through
`%create-dmatrix-from-csr' -- or anything `with-foreign-matrix' accepts --
`XGDMatrixCreateFromDense', through `%create-dmatrix'. MISSING, the value that means
*missing*, or NIL for this backend's own default, reaches both entry points identically:
it is a key in the creation config JSON either way, and means nothing different for a
sparse matrix than for a dense one.

The `:sparse-input' capability is re-checked on the sparse branch rather than assumed --
`%check-sparse-input' above, which carries the reasoning. `:missing-value' is checked by
`make-dataset' itself rather than here, since MISSING reaches both branches alike and there
is no branch for its check to belong to.

A `defun', not a second `make-dataset' method specialized on `csr-matrix' -- see
`cl-gbdt/src/lightgbm/protocol''s function of the same name and purpose, which this
mirrors, for why."
  (let ((function-name (%creation-function-name matrix)))
    (if (typep matrix 'csr-matrix)
        (progn
          (%check-sparse-input backend)
          (values (%create-dmatrix-from-csr (csr-matrix-indptr matrix)
                                            (csr-matrix-indices matrix)
                                            (csr-matrix-values matrix)
                                            (csr-matrix-num-columns matrix)
                                            missing)
                  function-name))
        (values (%create-dmatrix matrix missing) function-name))))

(defmethod make-dataset ((backend xgboost-backend) matrix
                          &key label weight group feature-names parameters reference missing
                            categorical-features)
  "Build an XGBoost dataset (a DMatrix) from MATRIX -- a dense matrix via
`XGDMatrixCreateFromDense', a `csr-matrix' via `XGDMatrixCreateFromCSR' -- attaching LABEL
and WEIGHT with `XGDMatrixSetInfoFromInterface', GROUP with `XGDMatrixSetUIntInfo', and
FEATURE-NAMES with `XGDMatrixSetStrFeatureInfo' when supplied. See the `make-dataset'
generic function's docstring for what each argument means.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `%dataset-pointer', which checks it. LABEL,
WEIGHT, GROUP and FEATURE-NAMES behave identically either way: they are attached to the
finished DMatrix by the calls below, which never see which entry point built it. REFERENCE
and PARAMETERS are refused for a `csr-matrix' exactly as they are for a dense matrix, and
for the same reasons, spelled out below.

MISSING, the value that means *missing*, becomes the `\"missing\"' key of whichever creation
config JSON MATRIX's form reaches. It needs this backend's `:missing-value' capability, which
`%check-missing-value' re-checks below rather than trusting the caller to have asked, and it
signals `unsupported-argument' for anything that is neither a `real' nor NIL -- see
`missing-value-json', which renders it. NIL, the default, sends the IEEE NaN this backend
sent unconditionally before the argument existed, so a caller who passes nothing gets exactly
what they got before. The comparison the library then makes is at SINGLE precision, whatever
MATRIX's own element type: two `double-float's that share a `single-float' both count as
missing against a sentinel that narrows to it.

CATEGORICAL-FEATURES, a list of 0-based column indices, is attached with the same
`XGDMatrixSetStrFeatureInfo' FEATURE-NAMES uses, under the `\"feature_type\"' field instead of
`\"feature_name\"' -- one string per column, `\"c\"' for a named column and `\"q\"' for every
other, as `categorical-feature-types' renders them. It needs this backend's
`:categorical-features' capability, which `%check-categorical-features' re-checks below rather
than trusting the caller to have asked, and it signals `unsupported-argument' naming
`:categorical-features' for an index that is not an integer, is negative, is beyond MATRIX's
last column, or was named twice. NIL, the default, attaches no `\"feature_type\"' at all --
exactly what every call sent before the argument existed, not a vector of `\"q\"'.

The list is rendered from the CALLER's MATRIX, before `%dataset-pointer' builds anything, so a
bad index signals with no DMatrix yet allocated and the range check is made against the same
count `cl-gbdt/src/lightgbm/protocol''s `make-dataset' checks against. The attachment then has
to wait until after creation, `XGDMatrixSetStrFeatureInfo' needing a handle -- which is also
why a `csr-matrix' needs nothing of its own here: the two creation branches have converged by
the time it runs, and the renderer reads a `csr-matrix''s declared column count where it reads
a dense matrix's second dimension.

Measured, and the reason a dataset that builds here can still fail later: `tree_method exact'
refuses categorical features at `train', not at `make-dataset'. The DMatrix is built and the
types attached without complaint, and `XGBoosterUpdateOneIter' then returns -1 with
`Updater `grow_colmaker` or `exact` tree method doesn't support categorical features'. That is
:PARAMETERS' business, not this method's -- `hist' and `approx' both work -- and nothing here
pre-validates an updater it is not given.

REFERENCE and PARAMETERS both signal `unsupported-argument' rather than being silently
dropped: REFERENCE is a LightGBM-only concept -- aligning a new dataset's bin mapper to an
existing one's, which XGBoost has nothing resembling. PARAMETERS is more subtle: the
vendored header (`ffi-spec/xgboost/include/xgboost/c_api.h') documents exactly three keys
for `XGDMatrixCreateFromDense''s config JSON -- `\"missing\"', which now has its own
:MISSING argument above and so is not what a caller reaches for PARAMETERS to set,
`\"nthread\"' and `\"data_split_mode\"' -- none of which correspond to what a caller moving
a working call from LightGBM actually means by dataset-level PARAMETERS there: binning knobs
such as
`max_bin' and `min_data_in_bin'. Forwarding `normalize-parameters''s output into that
config JSON regardless would not raise anything either: confirmed empirically against the
vendored library, `XGDMatrixCreateFromDense' returns success and silently ignores an
unrecognized config key rather than rejecting it, which would just move today's silent
drop one layer deeper, into C, instead of fixing it. The same holds for a `csr-matrix': that
header documents `XGDMatrixCreateFromCSR''s config by cross-reference to
`XGDMatrixCreateFromDense', so it is the same three keys either way, and the refusal below
names whichever of the two the caller's own MATRIX would have reached -- see
`%creation-function-name'. Either PARAMETERS or REFERENCE accepted and discarded here would
let a caller move a working `make-dataset' call from LightGBM to XGBoost and get a dataset
that looks fine but was not built the way the caller asked, which is exactly the failure
mode this project keeps finding.

Signals `foreign-call-error' when dataset creation reports success but writes a null
handle -- a library-contract violation, but one every later call through this handle would
otherwise dereference blindly.

The raw DMatrix handle exists in C from the moment the creation call returns, but
`make-handle' does not take ownership of it until the very end -- attaching LABEL, WEIGHT,
GROUP, FEATURE-NAMES or the rendered feature types can each signal first (a wrong-length
`:label' is the commonest way). OWNED tracks whether `make-handle' ran; when it did not, the
raw DMatrix is freed here instead of orphaned.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (when missing
      (%check-missing-value backend))
    (when categorical-features
      (%check-categorical-features backend))
    (%check-unsupported
     backend "make-dataset's :reference" reference
     "XGBoost has no bin-mapper alignment; :reference is a LightGBM-only concept")
    (%check-unsupported
     backend "make-dataset's :parameters" parameters
     (format nil "~A's config JSON only recognizes missing/nthread/data_split_mode, none ~
                   of which are LightGBM's dataset-level binning parameters, and the ~
                   library silently ignores any other key rather than rejecting it"
             (%creation-function-name matrix)))
    ;; Rendered before creation, attached after: the renderer takes the caller's own MATRIX,
    ;; so a bad index signals here with nothing yet allocated, while the attachment needs a
    ;; DMatrix handle to attach to. See this method's docstring.
    (let ((feature-types (categorical-feature-types categorical-features matrix
                                                    (backend-name backend))))
      (multiple-value-bind (dataset-pointer function-name)
          (%dataset-pointer backend matrix missing)
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
                 (when feature-types
                   (%set-feature-types dataset-pointer feature-types))
                 (prog1
                     (make-handle 'xgboost-dataset dataset-pointer backend :dataset)
                   (setf owned t)))
            (unless owned
              (handler-case (%free-dmatrix-unchecked dataset-pointer)
                (error () nil)))))))))

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

(defun %valid-set-name (backend entry)
  "Return the name half of ENTRY, one element of `train''s :VALID-SETS: NIL when ENTRY is
a bare dataset, or ENTRY's car when ENTRY is a (NAME . DATASET) cons and NAME is a string.

Signals `unsupported-argument' naming :VALID-SETS and ENTRY itself, via `%check-unsupported',
when ENTRY is a cons whose car is not a string -- before any foreign call, and before the
dataset half of ENTRY is checked against `xgboost-dataset' by `%check-xgboost-dataset',
which runs afterward on `%valid-set-dataset''s result. That later check is what turns a
cons whose cdr is not this backend's own kind of dataset into `wrong-backend-reference'
instead -- a different mistake from this one, and so a different condition."
  (if (consp entry)
      (let ((name (car entry)))
        (%check-unsupported
         backend "train's :valid-sets" (not (stringp name))
         (format nil "each element must be a dataset or a (string . dataset) cons; ~S's ~
                      car is not a string" entry))
        name)
      nil))

(defun %valid-set-dataset (entry)
  "Return the dataset half of ENTRY, one element of `train''s :VALID-SETS: ENTRY itself
when it is a bare dataset, or its cdr when it is a (NAME . DATASET) cons.

Does not check that the result is an `xgboost-dataset' -- `%check-xgboost-dataset' does
that afterward, on every element `%valid-set-name' has already let through."
  (if (consp entry) (cdr entry) entry))

(defun %recheck-train-datasets (backend dataset valid-sets)
  "Re-run `train''s own opening checks over BACKEND, DATASET and VALID-SETS, and return two
values: DATASET's freshly read live pointer, and a fresh list of the VALID-SETS entries'.

`train''s loop calls this after every `funcall' of a caller-supplied OBJECTIVE, which is the
only point in the loop where code this library did not write runs. That code may free the
training set -- `free-dataset' from inside the objective is the case this was found through --
and the pointer `train' read once before the loop is then a pointer into freed memory that
`XGBoosterTrainOneIter' takes as its DMatrix argument. Measured before this function existed:
`Signal 7 received', a bus error killing the process rather than signalling. The checks are
exactly the ones `train' already ran, so the caller gets `released-handle-error',
`backend-not-open' or `wrong-backend-reference' -- the typed conditions every other freed
handle in this library produces -- and never a fault. The RETURN VALUES are the point of the
exercise: re-checking and then going on to use the pointers read before the loop would fix
nothing, so `train' assigns both to the variables it reads from, and rebuilds its
DATASET-POINTERS list out of them.

The VALID-SETS entries are re-checked and returned because the same iteration hands their
pointers to `XGBoosterEvalOneIter' through `%read-evaluation' whenever RECORD-HISTORY is true
-- a freed DMatrix there is the identical use-after-free the training set is, which is why
`%check-booster-datasets-live' guards both on the public `update-one-iteration' path.

BACKEND itself is re-checked with `%check-backend-open' because `close-backend' unmaps the
shared library and the objective can call it. `handle-live-pointer' already refuses a handle
whose OWN backend has been closed, which covers the ordinary case where DATASET was built by
BACKEND; the check here is what covers the case `%check-xgboost-dataset' documents as
legitimate and therefore does not catch -- a dataset built by a second `xgboost-backend'
instance over the same library, whose own backend is still open while BACKEND is not. It
costs one slot read per iteration.

Mirrors `cl-gbdt/src/lightgbm/protocol''s function of the same name, which returns the
training pointer alone: that library's own custom update takes no DMatrix argument and its
`%read-evaluation' takes a dataset COUNT rather than pointers, so there is nothing there for
the validation half to be returned for."
  (%check-backend-open backend)
  (let ((train-data-pointer
          (%check-xgboost-dataset backend dataset "train's dataset argument"
                                   'xgboost-dataset)))
    (values train-data-pointer
            (mapcar (lambda (valid-set)
                      (%check-xgboost-dataset backend valid-set "a train :valid-sets entry"
                                               'xgboost-dataset))
                    valid-sets))))

(defmethod train ((backend xgboost-backend) dataset
                   &key valid-sets (num-rounds 100) parameters (record-history t)
                        early-stopping objective evaluation)
  "Train an XGBoost booster on DATASET for up to NUM-ROUNDS boosting iterations, and
return it and a `training-report' of the run.

Builds the booster with `XGBoosterCreate' over DATASET and every VALID-SETS entry's
DMatrix handle together -- see `%create-booster' for why XGBoost takes the whole set up
front rather than adding validation data afterward. Applies PARAMETERS one at a time via
`XGBoosterSetParam', then drives `XGBoosterUpdateOneIter' NUM-ROUNDS times -- or fewer,
when EARLY-STOPPING ends the run first. See the `train' generic function's docstring for
what each argument means, and for what the secondary value holds; NUM-ROUNDS defaults to
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
iteration through `%read-evaluation': the same function the `evaluation' method calls, over
the same DMatrix pointers in the same order, which is what keeps the history and what
`evaluation' answers afterward from being able to disagree.
`training-report-from-history' folds the run's worth of them into the report once the loop
is done; that fold is backend-neutral and shared with `cl-gbdt/src/lightgbm/protocol''s
`train', so what the two backends record cannot drift apart either. It orders series by the
(DATASET-INDEX, METRIC-NAME) pair's first appearance, which for this backend is the order
`XGBoosterEvalOneIter' formatted its own result in, so the report's series arrive in exactly
the order `evaluation' reports its entries in without anything being sorted. A field the
parse could not read as a `double-float' is recorded as NIL, keeping its place in the series
rather than shortening it -- see `training-series-values'.

RECORD-HISTORY NIL skips that read entirely -- one `XGBoosterEvalOneIter' call per
iteration, plus the parse of the line it formats, which is what makes recording cost real
wall-clock time (see the `train' generic's docstring for the measured figures). The loop is
then exactly the `XGBoosterUpdateOneIter' loop this method ran before it recorded anything,
and the report it still returns as its secondary value has an empty series list over the
same NUM-ROUNDS -- `training-report-from-history' over an empty history, the same shape a
run with `disable_default_eval_metric=1' produces.

Skipping the read also widens what this method accepts, which matters here more than it
does on LightGBM: `XGBoosterEvalOneIter' evaluates every DMatrix it is handed, and refuses
one it cannot evaluate -- an unlabelled DMatrix passed in VALID-SETS is the case this was
found through, which `XGBoosterUpdateOneIter' trains on without complaint while the
evaluation call signals `foreign-call-error' (\"label and prediction size not match\"). With
RECORD-HISTORY true that failure now propagates out of `train' itself, through the OWNED
dance below, where before this backend recorded anything it surfaced only at a later
`evaluation' call. RECORD-HISTORY NIL never reaches the evaluation path and so trains such a
configuration exactly as before.

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

OBJECTIVE replaces `XGBoosterUpdateOneIter' with `XGBoosterTrainOneIter' for every iteration
of the loop, driven by the gradient and Hessian the caller's own function returns -- see the
`train' generic function's docstring for what that function is called with and what it must
return. Signals `capability-unavailable' naming `:custom-objective' for a non-NIL OBJECTIVE
when the capability reads false, before any foreign call: see `%check-custom-objective'
above, which reads the capability rather than this backend's name, and `*optional-symbols*'
for why the answer here is probed rather than declared. OBJECTIVE NIL, the default, reaches
no check and runs exactly the `XGBoosterUpdateOneIter' loop this method has always run.

PARAMETERS is passed through untouched, unlike `cl-gbdt/src/lightgbm/protocol''s `train',
which forces `objective' to \"none\" because `LGBM_BoosterUpdateOneIterCustom' refuses to run
while the booster holds an objective function. `XGBoosterTrainOneIter' has no such
restriction -- measured, a custom update is accepted with any objective set -- so there is
nothing here to override, and this method calls `objective-parameters' nowhere. What that
costs the caller is that the configured objective's PREDICTION TRANSFORM stays in effect:
with `binary:logistic' still set, `predict :kind :normal' on the resulting booster returns
probabilities of a margin the caller's own loss produced, while `:raw' returns that margin.
The generic function's docstring states this as the caller's decision; nothing here signals
or warns about it. `num_class' is likewise just another parameter, and 3 of it alone gives
three output groups -- no `multi:*' objective is needed for a multiclass custom-objective run.

Each iteration reads the booster's current raw scores with `%training-scores' -- an
`XGBoosterPredictFromDMatrix' margin prediction over the training DMatrix, this library
having no counterpart to LightGBM's `LGBM_BoosterGetPredict' that hands back scores it
already holds. It costs a prediction pass per iteration, which that backend's loop does not
pay. The result reaches OBJECTIVE as a (ROWS GROUPS) `double-float' array, where ROWS comes
from `%dataset-num-rows' on the training set's own pointer and GROUPS is divided out of the
prediction's reported element count by `%predict-ncol' rather than read from a parameter.
What comes back is checked by `check-objective-result' -- `cl-gbdt/src/config/objective''s,
the same backend-neutral pure code LightGBM's `train' calls, so both backends refuse the same
shapes with the same `dimension-mismatch' -- and only then flattened into the C buffers,
ROW-MAJOR on this backend (row I of group K at `(+ (* I GROUPS) K)', which is what an
`__array_interface__' of shape `[ROWS, GROUPS]' means) and converted to `single-float'. Both
the flattening and the score layout are measured; see `%train-one-iteration-custom' and
`%training-scores'. The flattening is this method's business and not the caller's: OBJECTIVE
is handed, and returns, a (ROWS GROUPS) array whichever order the library underneath wants it
in -- LightGBM wants the other one.

OBJECTIVE is funcalled inside this method's own `with-foreign-float-traps-masked' body wrap,
so the caller's Lisp arithmetic runs under the masked convention on x86-64 as well as on
aarch64 -- `(/ 1.0d0 0.0d0)' yields infinity there rather than signalling
`division-by-zero'. Nothing about that is specific to a custom objective; it is simply where
in `train' the caller's code now runs. A condition the caller's function does signal
propagates out of `train' through the OWNED dance below, freeing the raw booster handle
rather than orphaning it, exactly as a mid-loop foreign failure does.

An objective that frees a handle this loop depends on, or closes BACKEND, is caught rather
than crashed on: `%recheck-train-datasets' re-runs this method's own opening checks the
moment the `funcall' returns, and TRAIN-DATA-POINTER, VALID-SET-POINTERS and
DATASET-POINTERS are all reassigned from what it returns, so nothing after the caller's code
uses a pointer read before it. See that function for what each of the three re-checks is
for. This is the only place the loop needs it -- the OBJECTIVE NIL branch beside it runs no
caller code at all.

Neither RECORD-HISTORY nor EARLY-STOPPING is disabled by OBJECTIVE, and neither is made
meaningful by it: a metric configured through PARAMETERS relates to the library's own
objective, not to the caller's, and this method neither signals nor warns about that -- see
the `train' generic function's docstring, which states it as the caller's decision.

The argument is accepted by this lambda list rather than being absent from it: `train' is one
generic function, so a method that did not take the keyword at all would answer a caller who
named it with SBCL's `unknown-keyword-argument' rather than with the typed condition every
other unavailable capability on this backend answers with.

EVALUATION is REFUSED here, and that is all this method does with it today: this backend
declares `:custom-evaluation' in neither of its two capability lists, so
`%check-custom-evaluation' above signals `capability-unavailable' for every non-NIL value
before any foreign call, and no per-dataset metric is computed or recorded. The argument is
in this lambda list for the same reason OBJECTIVE's paragraph just above gives for its own --
`train' is one generic function and every method must accept every key -- and refusing it is
what policy section 7 requires of a capability this backend does not have, rather than
accepting it and silently recording only the library's own metrics. See the `train' generic
function's docstring for the contract a backend that DOES provide the capability implements.

DATASET and every VALID-SETS entry's dataset half are each run through
`%check-xgboost-dataset' before any foreign call. `train' dispatches on BACKEND, not on
DATASET, so unlike `dataset-num-rows' or `free-dataset' there is no CLOS specializer here
to rule out the wrong kind of handle first -- without this, `handle-live-pointer' would
happily hand `XGBoosterCreate' a booster's own pointer to use as one of its DMatrix
handles. Signals `wrong-backend-reference' when DATASET or a VALID-SETS entry's dataset
half is not an `xgboost-dataset', and `released-handle-error' or `backend-not-open' when
one is but has already been freed or had its own backend closed. A VALID-SETS entry that
is a cons with a non-string car never reaches this check at all: `%valid-set-name' signals
`unsupported-argument' for it first, which is the different mistake a malformed name is,
kept distinct from a wrong dataset handle.

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
    (%check-custom-objective backend objective)
    (%check-custom-evaluation backend evaluation record-history)
    (let* ((valid-set-entries (copy-list valid-sets))
           (train-data-pointer
             (%check-xgboost-dataset backend dataset "train's dataset argument"
                                      'xgboost-dataset))
           (valid-set-names
             (mapcar (lambda (entry) (%valid-set-name backend entry)) valid-set-entries))
           (valid-sets (mapcar #'%valid-set-dataset valid-set-entries))
           (valid-set-pointers
             (mapcar (lambda (valid-set)
                       (%check-xgboost-dataset backend valid-set "a train :valid-sets entry"
                                                'xgboost-dataset))
                     valid-sets))
           (dataset-pointers (cons train-data-pointer valid-set-pointers))
           (dataset-names (cons nil valid-set-names))
           ;; Built before `XGBoosterCreate', so a malformed spec -- or one asking for early
           ;; stopping with RECORD-HISTORY NIL -- signals with no raw booster handle in
           ;; existence yet to unwind. NIL when EARLY-STOPPING is NIL, which is what the
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
      (let ((booster-pointer (%create-booster dataset-pointers)))
        (let ((owned nil))
          (unwind-protect
               (progn
                 (%set-parameters booster-pointer parameters)
                 ;; ROUND is 1-based, which is the numbering `observe-iteration' answers
                 ;; `watcher-best-iteration' in and the report publishes.
                 (loop :for round :from 1 :to num-rounds
                       :do (if objective
                               ;; `%boosted-rounds', not ROUND: this is XGBoost's own 0-based
                               ;; `iter' argument, and reading it back from the booster is
                               ;; exactly what `%update-one-iteration' does for the built-in
                               ;; branch beside this one -- see that function for why the
                               ;; count is not tracked locally. ROUND is 1-based and belongs
                               ;; to the report and the early-stopping watcher, not to C.
                               (let ((scores (%training-scores
                                              booster-pointer train-data-pointer
                                              (%dataset-num-rows train-data-pointer))))
                                 (multiple-value-bind (grad hess) (funcall objective scores)
                                   ;; Before anything else this iteration does, and before the
                                   ;; next one reads TRAIN-DATA-POINTER again: the caller's
                                   ;; own code has just run and may have freed a handle this
                                   ;; loop holds a raw pointer to. DATASET-POINTERS is rebuilt
                                   ;; rather than left alone -- `%read-evaluation' below reads
                                   ;; it, and it would otherwise still hold the stale ones.
                                   (multiple-value-setq (train-data-pointer valid-set-pointers)
                                     (%recheck-train-datasets backend dataset valid-sets))
                                   (setf dataset-pointers
                                         (cons train-data-pointer valid-set-pointers))
                                   (check-objective-result grad hess
                                                           (array-dimension scores 0)
                                                           (array-dimension scores 1))
                                   (%train-one-iteration-custom
                                    booster-pointer train-data-pointer
                                    (%boosted-rounds booster-pointer) grad hess)))
                               (%update-one-iteration booster-pointer train-data-pointer))
                           (incf completed-rounds)
                           ;; Primary value only: `%read-evaluation''s RAW is `evaluation''s
                           ;; provenance, and a report carries no per-iteration raw text.
                           (let ((entries (when record-history
                                            (%read-evaluation booster-pointer
                                                              dataset-pointers))))
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
                       (values (make-handle 'xgboost-booster booster-pointer backend :booster
                                            :training-set dataset
                                            :validation-sets valid-sets
                                            :best-iteration best-iteration)
                               report)
                     (setf owned t))))
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

(defmethod predict ((booster xgboost-booster) matrix
                    &key (kind :normal) num-iteration missing)
  "Predict on MATRIX with BOOSTER -- a dense matrix via `XGBoosterPredictFromDMatrix', a
`csr-matrix' via `XGBoosterPredictFromCSR'.

KIND and NUM-ITERATION are as the `predict' generic function documents, NUM-ITERATION's
:BEST resolved by `%resolve-best-num-iteration' before `%resolve-num-iteration' ever sees
it. Predictions start from iteration 0 -- the protocol exposes no start-iteration override.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `%check-sparse-input', which checks it before
any foreign call.

MISSING, the value in MATRIX that means *missing*, reaches the library through a DIFFERENT
config for each of MATRIX's two forms, and neither is the one `make-dataset' fills. A dense
MATRIX becomes a transient DMatrix, so its sentinel is a key in THAT DMatrix's creation
config -- `%create-dmatrix' below, exactly as for a dataset that outlives the call. A
`csr-matrix' builds no DMatrix at all, so its sentinel is a key in the INPLACE PREDICT config
instead -- `%predict-from-csr', which needs the key anyway. Same argument, same meaning, two
config strings built by two functions: see `%predict-config-json', whose own docstring says
why the dense path leaves the key out of the predict config rather than sending the sentinel
twice.

MISSING needs this backend's `:missing-value' capability, which `%check-missing-value'
re-checks below before any foreign call rather than inheriting the check `make-dataset' made
on the dataset BOOSTER was trained from -- policy section 7 asks each operation to check for
itself. It signals `unsupported-argument' for anything that is neither a `real' nor NIL, see
`missing-value-json', and NIL -- the default -- sends the IEEE NaN this backend sent
unconditionally before the argument existed, so a caller who passes nothing predicts exactly
what they predicted before. The comparison the library then makes is at SINGLE precision,
whatever MATRIX's own element type, as it is on the ingestion path.

Nothing here relates MISSING to the sentinel BOOSTER's training dataset was built with:
XGBoost does not record a DMatrix's sentinel on the booster, so the two are independent and
their disagreement is undetectable. See the `predict' generic function's docstring, where
that is stated as the caller's responsibility.

A dense MATRIX is built into a transient DMatrix via `%create-dmatrix' first --
`XGBoosterPredictFromDMatrix' takes a DMatrix handle, unlike LightGBM's
`LGBM_BoosterPredictForMat', which predicts straight off a raw pointer and row/column
counts. It is built first, before anything else here, so a MISSING that
`%dense-matrix-config-json' refuses signals with nothing pinned and no foreign allocation
held -- the property `%create-dmatrix''s own docstring claims. The transient DMatrix is
freed before this returns, on every exit path, since nothing else retains it. Its free is
checked with `check-xgb', not discarded outright: a failure there is reported with `warn'
rather than an error, matching
`free-dataset''s own reasoning for warning instead of signalling, since raising an error
from cleanup would replace whatever condition is already propagating on an unwinding exit
-- but on the ordinary success path, a failed free still leaks foreign memory and is worth
reporting rather than passing over in silence.

A `csr-matrix' builds no DMatrix at all: `XGBoosterPredictFromCSR' is XGBoost's INPLACE
prediction and reads the three vectors where they lie, so there is nothing transient to
free and no `unwind-protect' around it. That saves a copy, and it is the entry point the
`:sparse-input' capability declares -- but it is a different code path from
`XGBoosterPredictFromDMatrix', not a CSR spelling of it, and **it covers only `:normal' and
`:raw'**. `:contrib' and `:leaf-index' on a `csr-matrix' signal `foreign-call-error'
(\"Unsupported prediction type:2\" and \":6\" respectively), measured against the vendored
library. Both work on a dense matrix, and materialising the rows as one -- a 2D
`double-float' or `single-float' array, or a `foreign-matrix' -- is the only way to reach
either KIND for rows a caller holds sparsely: those are the only other forms
`call-with-foreign-matrix' has a method for, and a dataset is not among them, so routing the
rows through `make-dataset' leads nowhere `predict' can be called on.
That refusal is the library's own and is left to it, exactly as a `csr-matrix' whose
NUM-COLUMNS is not BOOSTER's feature count is (\"Number of columns in data must equal to the
trained model\"). NUM-ITERATION is honoured identically on both paths -- the same
`iteration_begin'/`iteration_end' pair reaches the same config JSON, which additionally
carries the `\"missing\"' key inplace prediction requires; see `%predict-config-json'.

The output buffer's total element count comes from the C call's own `out_shape'/`out_dim'
report, not from the row count alone -- the row count is only
correct for a single-class objective. The second array dimension is that total divided
by the row count, guarded by `%predict-ncol' -- the same derivation
`cl-gbdt/src/lightgbm/backend' uses for its own row-count-alone pitfall, and the one that
tells a three-class `multi:softprob' model's predictions apart from a binary model's. Both
entry points report it the same way, `\"strict_shape\":true' being set for both.

That same report is also RETURNED, as this method's second value: `%reported-shape' reads
`out_shape' back as a list of integers instead of only multiplying it out, and neither entry
point interprets or reshapes it. It is never NIL here: `out_dim' was measured 2 for `:normal'
and `:raw', 3 for `:contrib' and 4 for `:leaf-index' on both entry points, so
`%reported-shape''s empty-loop case -- a zero DIM -- does not arise. This backend declares
`:prediction-shape' in `*provided-capabilities*' to say so, and nothing re-checks that
declaration: there is no argument to refuse, and what the declaration says is that the
mechanism is present, not that the shape is non-NIL. Measured against the vendored library, the
shape is RICHER than the first value's own dimensions for two kinds: a four-round three-class
model over four features reports (rows 4 3 1) for `:leaf-index' where the array is rows x 12, and
(rows 3 5) for `:contrib' where the array is rows x 15. A four-round BINARY model over three
features reports (rows 4 1 1) and (rows 1 4) -- multidimensional there too, its one output
group notwithstanding -- so this is not a multiclass-only difference. The first value is
untouched by any of it.

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
          (iteration-end
            (%resolve-num-iteration
             (%resolve-best-num-iteration booster num-iteration "predict's :num-iteration"))))
      (when missing
        (%check-missing-value (handle-backend booster)))
      ;; Reading `out_shape'/`out_dim'/`out_result' back into the result array, and back out
      ;; as this method's two return values, is identical for both entry points and lives
      ;; here once; CALL is the only thing that differs between them, which is exactly how
      ;; much of this method a `csr-matrix' changes. Both entry points report the shape the
      ;; same way -- `"strict_shape":true' is set for both -- so neither branch special-cases
      ;; the KIND, and the two KINDs the sparse one refuses never get here at all.
      (flet ((predict-into (nrow call)
               (cffi:with-foreign-objects ((out-shape :pointer) (out-dim :uint64)
                                           (out-result :pointer))
                 (funcall call out-shape out-dim out-result)
                 (let* ((dim (cffi:mem-ref out-dim :uint64))
                        (shape-pointer (cffi:mem-ref out-shape :pointer))
                        (element-count (%total-element-count shape-pointer dim))
                        ;; Read off the same pointer as the count above and before anything
                        ;; else touches this booster: `out_shape' is XGBoost's own memory,
                        ;; valid only until the next call into it, as `out_result' is.
                        (shape (%reported-shape shape-pointer dim))
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
                   (values result shape)))))
        (if (typep matrix 'csr-matrix)
            (progn
              (%check-sparse-input (handle-backend booster))
              (predict-into (csr-matrix-num-rows matrix)
                            (lambda (out-shape out-dim out-result)
                              (%predict-from-csr booster-pointer
                                                 (csr-matrix-indptr matrix)
                                                 (csr-matrix-indices matrix)
                                                 (csr-matrix-values matrix)
                                                 (csr-matrix-num-columns matrix)
                                                 predict-type iteration-end missing
                                                 out-shape out-dim out-result))))
            ;; The transient DMatrix is built first, so a MISSING that
            ;; `%dense-matrix-config-json' refuses signals with nothing pinned and no foreign
            ;; allocation held -- the property `%create-dmatrix''s own docstring claims. NROW
            ;; comes from the DMatrix `%create-dmatrix' already built, via `%dataset-num-rows',
            ;; rather than a second pin of MATRIX just to read its row count back.
            (let ((dmatrix-pointer (%create-dmatrix matrix missing)))
              (when (cffi:null-pointer-p dmatrix-pointer)
                (error 'foreign-call-error
                       :function-name "XGDMatrixCreateFromDense"
                       :code 0
                       :message "reported success but returned a null dataset handle"))
              (unwind-protect
                   (let ((nrow (%dataset-num-rows dmatrix-pointer)))
                     (cffi:with-foreign-string
                         (config (%predict-config-json predict-type iteration-end))
                       (predict-into nrow
                                     (lambda (out-shape out-dim out-result)
                                       (%predict-from-dmatrix booster-pointer dmatrix-pointer
                                                              config out-shape out-dim
                                                              out-result)))))
                (handler-case (%free-dmatrix dmatrix-pointer)
                  (error (condition)
                    (warn "Freeing predict's temporary XGBoost dataset failed: the ~
                           foreign dataset was not freed and its memory is leaked. ~A"
                          condition))))))))))

;;; ---------------------------------------------------------------------------
;;; Persistence

(defmethod save-model ((booster xgboost-booster) path &key num-iteration)
  "Save BOOSTER's model to PATH via `XGBoosterSaveModel'.

Signals `unsupported-argument' when NUM-ITERATION is supplied: unlike LightGBM's
`LGBM_BoosterSaveModel', `XGBoosterSaveModel' takes no iteration limit -- it always
saves every boosted round -- and silently ignoring the argument would be exactly the
failure mode `unsupported-argument' exists to prevent, per `%check-unsupported'. :BEST is
resolved by `%resolve-best-num-iteration' first, into an integer, which then meets this
same check exactly as an explicit integer would -- not special-cased around it. A caller
who wants a file that stops at the best iteration slices to it first with
`cl-gbdt/xgboost:slice-model' and saves the slice instead.

Returns PATH."
  (with-foreign-float-traps-masked
    (let ((resolved (%resolve-best-num-iteration booster num-iteration
                                                  "save-model's :num-iteration")))
      (%check-unsupported
       (handle-backend booster) "save-model's :num-iteration" resolved
       "XGBoosterSaveModel has no iteration limit; every boosted round is saved"))
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
:BEST is resolved by `%resolve-best-num-iteration' first, into an integer, which then
meets this same check exactly as an explicit integer would.

`out_dptr' is XGBoost's own memory, copied out via `foreign-string-to-lisp' with an
explicit `:count' from `out_len' rather than trusted to be null-terminated at the right
place."
  (with-foreign-float-traps-masked
    (let ((resolved (%resolve-best-num-iteration booster num-iteration
                                                  "model-to-string's :num-iteration")))
      (%check-unsupported (handle-backend booster) "model-to-string's :num-iteration"
                           resolved "XGBoosterSaveModelToBuffer has no iteration limit"))
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
inventing a reduction XGBoost itself does not define.

NUM-ITERATION does not accept :BEST, unlike `predict', `save-model' and
`model-to-string' -- `%reject-best-num-iteration' signals `unsupported-argument' for it
explicitly, ahead of the blanket rejection just below that would otherwise catch it only
incidentally, as any other non-NIL value."
  (with-foreign-float-traps-masked
    (%reject-best-num-iteration booster num-iteration "feature-importance's :num-iteration")
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
  "Return BOOSTER's evaluation metrics via `%read-evaluation', the pointer-level reader this
backend shares between this method and `train''s per-iteration recording loop, so the two
can never disagree -- see the `evaluation' generic function's docstring for the portable
contract this satisfies.

`XGBoosterEvalOneIter' evaluates whatever DMatrices it is handed and consults nothing the
booster was built with, so the datasets this evaluates are BOOSTER's own retained
handles: its training set first, then each `train' :VALID-SETS entry in the order the
caller supplied them. That is what makes DATASET-INDEX mean the same thing here as it
does on LightGBM, which can only evaluate what training attached -- measured before this
method was written: for one booster, one set of handles and one iteration,
`XGBoosterEvalOneIter' called directly and this path through `%read-evaluation' produce
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

This method reads BOOSTER and every dataset it evaluates through `handle-live-pointer'
itself, before calling `%read-evaluation', so a freed booster or a freed retained dataset
signals `released-handle-error' right here; unlike `cl-gbdt/src/lightgbm/protocol''s
method, this one needs no separate `%check-booster-datasets-live', since every dataset it
evaluates is one it resolves and checks explicitly, by its own handle, before any foreign
call."
  (with-foreign-float-traps-masked
    (let* ((booster-pointer (handle-live-pointer booster))
           (training-set (booster-training-set booster))
           (datasets (if training-set
                         (cons training-set (booster-validation-sets booster))
                         '()))
           (dataset-pointers (mapcar #'handle-live-pointer datasets)))
      (multiple-value-bind (entries raw) (%read-evaluation booster-pointer dataset-pointers)
        (values entries (list :value-source :parsed-text :raw raw))))))

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
