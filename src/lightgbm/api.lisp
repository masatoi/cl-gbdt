;;;; api.lisp --- LightGBM's Layer 1 operations, over the handle classes.
;;;;
;;;; `native.lisp' holds the `%'-functions: each takes and returns raw pointers, and none of them
;;;; is a finished operation. `classes.lisp' holds the CLOS types and the shared library's
;;;; lifetime. This file is what a caller of `cl-gbdt/lightgbm' actually invokes -- operations
;;;; that take a backend or a handle, do the whole job, and hand back a handle or a result.
;;;;
;;;; Every OPERATION here reaches the shared library, so every one wraps its whole body in
;;;; `with-foreign-float-traps-masked' -- see `protocol.lisp''s header for why, and
;;;; `tools/ci/check-float-traps.lisp', which scans this file and holds every name the sibling
;;;; `all.lisp' exports to that rule. The `%'-prefixed helpers are not separately wrapped and
;;;; are not policed by that check: each runs inside the body wrap of the one operation that
;;;; calls it -- `%dataset-pointer' inside `create-dataset', `%prediction-shape' inside
;;;; `predict' -- and each says so where it is defined, since nothing in CI would notice a
;;;; future caller reaching it from outside an already-masked extent.
;;;;
;;;; Nothing here may depend on `cl-gbdt/src/protocol' or the training files: this file is Layer 1,
;;;; and `tools/ci/check-layer-separation.lisp' fails the build if it ever does. That is not a
;;;; stylistic preference. A caller who loaded `cl-gbdt/lightgbm' alone has no unified API in the
;;;; image, and these functions are the whole of what they can call.
;;;;
;;;; Loads after `classes.lisp' and cannot precede it: every operation below takes or returns a
;;;; `lightgbm-dataset' or a `lightgbm-booster', and `%reference-pointer' and
;;;; `%check-lightgbm-dataset' are each handed a class name as a symbol.

(uiop:define-package #:cl-gbdt/src/lightgbm/api
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/lightgbm/classes
                #:lightgbm-booster
                #:lightgbm-dataset)
  (:import-from #:cl-gbdt/src/lightgbm/native
                #:%add-valid-data
                #:%booster-num-features
                #:%calc-num-predict
                #:%check-backend-open
                #:%check-booster-datasets-live
                #:%check-lightgbm-dataset
                #:%create-booster
                #:%create-dataset
                #:%create-dataset-from-csr
                #:%data-type
                #:%free-booster
                #:%free-booster-unchecked
                #:%free-dataset
                #:%free-dataset-unchecked
                #:%parameter-string
                #:%predict-for-csr
                #:%predict-for-mat
                #:%predict-ncol
                #:%predict-type
                #:%reference-pointer
                #:%resolve-num-iteration
                #:%set-feature-names
                #:%set-group-field
                #:%set-info-field
                #:%update-one-iteration)
  (:import-from #:cl-gbdt/src/backend
                #:backend-name
                #:backend-open-p
                #:backend-supports-p)
  (:import-from #:cl-gbdt/src/conditions
                #:capability-unavailable
                #:foreign-call-error)
  (:import-from #:cl-gbdt/src/config/prediction-shape
                #:contrib-shape)
  (:import-from #:cl-gbdt/src/data
                #:csr-matrix
                #:csr-matrix-indices
                #:csr-matrix-indptr
                #:csr-matrix-num-columns
                #:csr-matrix-num-rows
                #:csr-matrix-values
                #:with-foreign-matrix)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:import-from #:cl-gbdt/src/handle
                #:%reject-best-num-iteration
                #:handle-backend
                #:handle-live-pointer
                #:handle-released-p
                #:release-handle
                #:with-pointer-ownership)
  (:export #:create-booster
           #:create-dataset
           #:free-booster
           #:free-dataset
           #:predict
           #:update-one-iteration))

(in-package #:cl-gbdt/src/lightgbm/api)

;;; ---------------------------------------------------------------------------
;;; The `:sparse-input' gate

(defun %check-sparse-input (backend)
  "Signal `capability-unavailable' when BACKEND's `:sparse-input' capability reads false.

Policy section 7 requires the operation itself to re-check a capability rather than trusting
the caller to have asked `backend-supports-p' first, so a caller who never asked gets a typed
condition instead of a missing-symbol crash. Both operations this backend gates on
`:sparse-input' call this -- `%dataset-pointer' below, on `create-dataset''s behalf, and
`predict' at the end of this file -- so the two cannot come to disagree about which capability
they name or which backend they blame.

Not exported, unlike every other name this file defines that a caller reaches. It was, while
`cl-gbdt/src/lightgbm/protocol''s `predict' held the prediction procedure and imported this
gate from here; that procedure is now `predict' below, both call sites are in this file, and
an export nothing outside it needs is one more claim to keep true.

Only a `csr-matrix' argument ever reaches this. A dense matrix needs neither
`LGBM_DatasetCreateFromCSR' nor `LGBM_BoosterPredictForCSR' to exist, and must keep working
on a library that has neither."
  (unless (backend-supports-p backend :sparse-input)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :sparse-input)))

;;; ---------------------------------------------------------------------------
;;; Datasets

(defun %dataset-pointer (backend matrix parameter-string reference-pointer)
  "Return two values: the raw LightGBM dataset pointer built from MATRIX, and the name
of the C function that produced it, for the null-handle check `create-dataset' makes
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
whole procedure single and varies only the call that actually differs."
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

(defun create-dataset (backend matrix &key label weight group feature-names parameters
                                        reference)
  "Build and return a `lightgbm-dataset' from MATRIX on BACKEND.

MATRIX is a dense matrix -- built via `LGBM_DatasetCreateFromMat' -- or a `csr-matrix', built
via `LGBM_DatasetCreateFromCSR'. Nothing else about this function varies with which of the two
it is: see `%dataset-pointer' above, the only form here that branches on it.

LABEL and WEIGHT are attached to the finished dataset with `LGBM_DatasetSetField' under those
LightGBM field names, GROUP with the same call under `group', and FEATURE-NAMES with
`LGBM_DatasetSetFeatureNames'. Each is attached only when supplied; a NIL one is not written
as an empty field. REFERENCE is another `lightgbm-dataset' whose bin mapper this one should
align to -- what `LGBM_DatasetCreateFromMat' calls its `reference' argument, and what a
validation set needs to be binned the same way as the training set it will be scored against
-- or NIL for none.

PARAMETERS is a plist in LightGBM'S OWN vocabulary, rendered by `%parameter-string' and handed
to the creation call verbatim; nothing here translates a key or a value, and no key is added.
`:categorical-feature' is one such key among the rest -- this function does not know it as a
concept, and a caller who wants columns read as categories writes that key here, already
rendered as the string LightGBM expects. `cl-gbdt/src/lightgbm/protocol''s `make-dataset' is
what turns the portable :CATEGORICAL-FEATURES argument into exactly that entry before calling
this.

Signals `backend-not-open' before any foreign call when BACKEND is not open -- see
`%check-backend-open'. Signals `capability-unavailable' naming `:sparse-input' when MATRIX is
a `csr-matrix' and that capability reads false, `wrong-backend-reference' when REFERENCE is
supplied and is not a `lightgbm-dataset', `released-handle-error' when it has already been
freed, and `backend-not-open' when its own backend has since been closed -- see
`%reference-pointer'. Signals `foreign-call-error' when the creation call reports success but
writes a null handle: a library-contract violation, but one every later call through this
handle would otherwise dereference blindly.

The raw dataset handle exists in C from the moment the creation call returns, but
`make-handle' does not take ownership of it until the very end -- attaching LABEL, WEIGHT,
GROUP or FEATURE-NAMES can each signal first (a wrong-length LABEL is the commonest way).
`with-pointer-ownership' spans exactly that gap: the pointer is owned by nobody inside its
body, and any exit that has not called TAKE-OWNERSHIP frees the raw dataset here instead of
orphaning it."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (let ((reference-pointer (%reference-pointer backend reference 'lightgbm-dataset))
          (parameter-string (%parameter-string parameters)))
      (multiple-value-bind (dataset-pointer function-name)
          (%dataset-pointer backend matrix parameter-string reference-pointer)
        (when (cffi:null-pointer-p dataset-pointer)
          (error 'foreign-call-error
                 :function-name function-name
                 :code 0
                 :message "reported success but returned a null dataset handle"))
        (with-pointer-ownership (dataset-pointer #'%free-dataset-unchecked take-ownership)
          (when label
            (%set-info-field dataset-pointer "label" label))
          (when weight
            (%set-info-field dataset-pointer "weight" weight))
          (when group
            (%set-group-field dataset-pointer group))
          (when feature-names
            (%set-feature-names dataset-pointer feature-names))
          (take-ownership 'lightgbm-dataset backend :dataset))))))

(defun free-dataset (dataset)
  "Free DATASET via `LGBM_DatasetFree'. Does nothing if it was already freed, and returns no
useful value.

Unlike every other operation that reads an existing handle -- `free-booster' below excepted,
which takes this same path for this same reason -- this does not go through
`handle-live-pointer' and so does not signal `backend-not-open' when DATASET's backend has
already been closed. A free runs from a cleanup form -- the `unwind-protect' a Layer 1 caller
writes for itself, as tests/functional/lightgbm-standalone.lisp does, or the one inside
`cl-gbdt''s `with-dataset', which reaches this function through the method that delegates to
it and which a caller of `cl-gbdt/lightgbm' alone does not have -- and a non-local exit is
exactly when that cleanup runs; signalling there would replace whatever condition is already
unwinding the stack instead of letting it propagate. So when
the backend is closed, the handle is instead marked released without calling
`LGBM_DatasetFree' -- the shared library may no longer be mapped into the process, so that
call cannot be trusted not to crash -- and a `warn' reports the foreign memory as leaked,
since it is genuinely unreclaimable at that point."
  (with-foreign-float-traps-masked
    (if (backend-open-p (handle-backend dataset))
        (release-handle dataset (lambda (pointer) (%free-dataset pointer)))
        (let ((already-released (handle-released-p dataset)))
          (release-handle dataset (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing a LightGBM dataset after its backend was closed: the foreign ~
                   dataset was not freed and its memory is leaked."))))))

;;; ---------------------------------------------------------------------------
;;; Boosters
;;;
;;; `cl-gbdt/src/lightgbm/protocol''s `train' does NOT call `create-booster', and a reader
;;; who assumes every training run exercises it would be wrong. `train' must hand
;;; `make-handle' a `:best-iteration' its own loop computes, and `booster-best-iteration' is
;;; a `:reader'-only slot set at construction, so `train' has to own the pointer across its
;;; whole loop and build the handle at the end; `create-booster' builds it at the start, by
;;; the same argument every other Layer 1 operation follows. See `train''s own call site,
;;; which carries that reasoning where an editor tempted to merge the two will meet it. The
;;; two small functions below ARE what their methods call, wholesale.

(defun create-booster (backend dataset &key parameters valid-sets)
  "Create a booster over DATASET on BACKEND via `LGBM_BoosterCreate', returning a
`lightgbm-booster'.

The result is UNTRAINED: `LGBM_BoosterCreate' allocates the model and fixes its parameters,
and every boosting iteration comes from a later `update-one-iteration'. Free it with
`free-booster'.

PARAMETERS is a plist in LightGBM'S OWN vocabulary, rendered by `%parameter-string' and
handed to the creation call verbatim; nothing here translates a key or a value, and no key
is added. VALID-SETS is a list of `lightgbm-dataset's -- bare datasets, not the
(NAME . DATASET) entries `cl-gbdt''s `train' accepts, a name being a training-report concept
with no meaning at this layer -- attached afterward with `LGBM_BoosterAddValidData' in the
order given. The booster retains DATASET and a COPY of VALID-SETS, which keeps them alive
for its lifetime and lets `update-one-iteration' notice a dataset freed out from under it.
The copy is what makes that promise hold: were the caller's own list object stored, a later
`delete' or `(setf (cdr ...))' on it would remove an entry from the booster's view while
LightGBM still held that dataset's pointer.

Signals `backend-not-open' before any foreign call when BACKEND is not open -- see
`%check-backend-open' -- `wrong-backend-reference' when DATASET or a VALID-SETS entry is not
a `lightgbm-dataset', and `released-handle-error' or `backend-not-open' when one is but has
already been freed or had its own backend closed; see `%check-lightgbm-dataset', which is
what rules out `LGBM_BoosterCreate' being handed a booster's own pointer as its training
set. This function dispatches on nothing, so that check is the only thing standing between a
wrong-kind handle and a segfault. Signals `foreign-call-error' when creation reports success
but writes a null handle -- that check lives in `%create-booster', beside the call it guards,
and is not repeated here.

Every check runs before the creation call, so a rejected VALID-SETS entry leaves no booster
in existence at all. The raw handle then exists in C from the moment that call returns and
nothing in Lisp references it until `make-handle' runs -- `with-pointer-ownership' spans
exactly that gap, so a validation set that fails to attach frees the booster rather than
orphaning it."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    ;; `let', not `let*': no binding here reads another, so the sequential scope `let*' adds
    ;; would claim a dependency that is not there. Ordering is NOT what separates the two --
    ;; both evaluate their init forms left to right -- and the ordering this function depends
    ;; on is not among these bindings at all: the checks precede the creation call because
    ;; that call sits in the nested form below, in this `let''s body. `%create-booster' is
    ;; deliberately NOT among these bindings, for exactly that reason -- it belongs inside its
    ;; own form, where the raw handle's lifetime begins.
    (let ((train-data-pointer
            (%check-lightgbm-dataset backend dataset "create-booster's dataset argument"
                                      'lightgbm-dataset))
          (valid-set-pointers
            (mapcar (lambda (valid-set)
                      (%check-lightgbm-dataset backend valid-set
                                                "a create-booster :valid-sets entry"
                                                'lightgbm-dataset))
                    valid-sets))
          (validation-sets (copy-list valid-sets)))
      (let ((booster-pointer (%create-booster train-data-pointer
                                              (%parameter-string parameters))))
        (with-pointer-ownership (booster-pointer #'%free-booster-unchecked take-ownership)
          (%add-valid-data booster-pointer valid-set-pointers)
          (take-ownership 'lightgbm-booster backend :booster
                          :training-set dataset
                          :validation-sets validation-sets))))))

(defun update-one-iteration (booster)
  "Advance BOOSTER by one boosting iteration via `LGBM_BoosterUpdateOneIter'.

Returns false when THIS iteration produced no further split -- LightGBM's own
`produced_empty_tree' out parameter, read by `%update-one-iteration' and inverted here, so
that a true return means the iteration did produce one. Per call, and about that call only:
the C function recomputes the flag every time and promises nothing about the next one, so a
false return is not a latch and a loop that stops on it is making its own decision, not
reading one the library has made. `cl-gbdt/src/protocol''s generic says the same in the
portable words \"returns false when no further split was possible\".

Signals `released-handle-error' when BOOSTER's training set, or any of its validation sets,
has already been freed -- see `%check-booster-datasets-live', which runs before any foreign
call because `LGBM_BoosterUpdateOneIter' dereferences those datasets' pointers itself and a
freed one is a segfault rather than a catchable condition. Signals `released-handle-error'
for a freed BOOSTER, and `backend-not-open' when its backend has since been closed -- see
`handle-live-pointer'."
  (with-foreign-float-traps-masked
    (%check-booster-datasets-live booster)
    (zerop (%update-one-iteration (handle-live-pointer booster)))))

(defun free-booster (booster)
  "Free BOOSTER via `LGBM_BoosterFree'. Does nothing if it was already freed, and returns no
useful value.

See `free-dataset' above for why this does not signal `backend-not-open' when BOOSTER's
backend has already been closed, but marks the handle released and `warn's the foreign
memory leaked instead -- the same cleanup-form reasoning applies here, whether the
`unwind-protect' is the caller's own or the one inside `cl-gbdt''s `with-booster'."
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

(defun %prediction-shape (kind result element-count nrow booster-pointer)
  "Return the shape `predict' states for its KIND result -- a list of integers in
`array-dimensions' order -- or NIL where this backend states none.

RESULT is the array `predict' is about to return, ELEMENT-COUNT the count
`LGBM_BoosterCalcNumPredict' gave for it, NROW the row count whichever entry point just ran was
handed, and BOOSTER-POINTER the booster it ran against.

No SHAPE is read back from the library here, because there is no shape to read: this backend's
prediction entry points report an element count and no axes at all, where XGBoost's write an
`out_shape'/`out_dim' pair `cl-gbdt/src/xgboost/protocol''s `predict' hands back verbatim. Every
value below is DERIVED, and each KIND gets only what can be derived from what is known.

That is not to say this function makes no foreign call. The `:contrib' arm calls
`%booster-num-features' -- `LGBM_BoosterGetNumFeature' -- for the feature count it derives a
width from. That is the one library call this function makes, and it runs inside `predict''s
`with-foreign-float-traps-masked' body wrap, which is the whole of this function's trap
protection: `%prediction-shape' is `%'-prefixed and named by no `:export' clause, so
`tools/ci/check-float-traps.lisp' does not police it -- that check holds `defun's the backend's
PUBLIC package exports, read from the second `define-package' in `src/lightgbm/all.lisp'. So the
constraint any future change has to preserve is stated here and nowhere else: a second library
call added below, or a caller reaching this function from outside an already-masked dynamic
extent, must establish the mask itself, and nothing in CI will notice if it does not.

What each KIND gets:

  `:normal', `:raw'  RESULT's own `array-dimensions'. One column per output group is the whole
                     of what these hold and the array already says so, so there is nothing
                     further to state.
  `:contrib'         `contrib-shape''s (NROW classes width), width being one contribution per
                     feature plus the bias and classes what is left of ELEMENT-COUNT once the
                     other two divide out. That function answers NIL, never signalling and never
                     guessing, for any of the FOUR cases its own docstring enumerates: NROW zero
                     or negative, a negative feature count, a division leaving a remainder, and
                     an exact division whose quotient is zero. The last two are its way of
                     saying the layout is not the one this arithmetic describes; the first two,
                     that the inputs never described a layout at all. Read that docstring rather
                     than this summary -- the contract is the four cases, not just the division.
                     The CLASS-MAJOR ordering the shape implies does not follow from
                     the count, which divides identically with the last two axes swapped: it is
                     a measured claim, held by
                     `lightgbm-s-derived-contrib-shape-is-the-one-the-numbers-support' in
                     tests/functional/prediction-shape.lisp, whose feature-major arm is the
                     control that makes it a measurement rather than a restatement.
  `:leaf-index'      NIL. ELEMENT-COUNT divides by iterations and output groups much as
                     `:contrib''s divides by output groups and width, but this project has no
                     property that tells the resulting orderings apart -- a leaf index is an
                     opaque identifier, summing to nothing and agreeing with nothing -- and an
                     ordering asserted without one is a guess. NIL means what it means
                     everywhere `predict''s second value appears: no shape is stated. It is not
                     an error and nothing signals.

NROW rather than RESULT's first dimension, though the two are equal, because it is the count
the entry point that just ran was handed and the one ELEMENT-COUNT was computed against --
which for a `csr-matrix' is `csr-matrix-num-rows' and not any dimension of MATRIX."
  (ecase kind
    ((:normal :raw) (array-dimensions result))
    (:contrib (contrib-shape element-count nrow (%booster-num-features booster-pointer)))
    (:leaf-index nil)))

(defun predict (booster matrix &key (kind :normal) num-iteration)
  "Predict on MATRIX with BOOSTER, returning two values: the result array and the SHAPE this
backend states for it.

MATRIX is a dense matrix -- predicted through `LGBM_BoosterPredictForMat' -- or a `csr-matrix',
through `LGBM_BoosterPredictForCSR'. KIND is `:normal', `:raw', `:leaf-index' or `:contrib',
mapped onto LightGBM's own `C_API_PREDICT_*' constant by `%predict-type', which signals for
anything else. Predictions start from iteration 0; nothing here exposes a start-iteration
override.

NUM-ITERATION is a positive integer, or NIL for every iteration -- which LightGBM spells as 0,
and `%resolve-num-iteration' is what writes it that way. :BEST is REFUSED, with
`unsupported-argument' naming this backend and \"predict's :num-iteration\": only `train' writes
a booster's `best-iteration', and a booster built by `create-booster' has none, so at this layer
the keyword would name an empty slot. `%reject-best-num-iteration' is what refuses it, and its
own docstring measures why the refusal has to be explicit -- `%resolve-num-iteration' is
`(or num-iteration 0)', so :BEST would otherwise reach a foreign call expecting an integer as
uninterpreted data and come back a raw CFFI `type-error' rather than a `cl-gbdt' condition. The
refusal is invisible to Layer 2: `cl-gbdt/src/lightgbm/protocol''s `predict' method resolves
:BEST first -- by `%resolve-best-num-iteration', which signals `unsupported-argument' when the
booster has no best iteration to resolve it against -- and calls this with the integer that
resolution produced, so the keyword itself never arrives from there.

Signals `capability-unavailable' naming `:sparse-input' when MATRIX is a `csr-matrix' and that
capability reads false -- see `%check-sparse-input' above, which checks it before any foreign
call. Everything else means exactly what it means for a dense matrix: both entry points take
the same PREDICT-TYPE, the same START-ITERATION/NUM-ITERATION pair and the same parameter
string, and both fill the same buffer in the same row-major order, so KIND and NUM-ITERATION
are honoured identically on either path -- all four KINDs included, unlike
`cl-gbdt/src/xgboost/protocol''s `predict', whose sparse entry point is XGBoost's inplace
prediction and covers only two of them. A `csr-matrix' whose NUM-COLUMNS is not BOOSTER's own
feature count is LightGBM's own mistake to catch, and it does, with a clean nonzero return this
reports as `foreign-call-error' (\"The number of features in data (N) is not the same as it was
in training data (M).\"); nothing here pre-empts that check.

Signals `released-handle-error' for a freed BOOSTER, and `backend-not-open' when its backend
has since been closed -- see `handle-live-pointer', which is read before anything is allocated,
and before NUM-ITERATION is examined, so a freed booster handed :BEST is reported as freed
rather than as having no best iteration.

The output buffer's element count comes from `LGBM_BoosterCalcNumPredict', not from the row
count alone: the row count is only correct for a single-class objective. That count is read the
same way for either matrix kind -- it depends on BOOSTER, the row count, KIND and
NUM-ITERATION, and on nothing about how the rows are laid out. The second array dimension is
that count divided by the row count, guarded by `%predict-ncol'. Whichever entry point ran also
writes its own element count back through OUT-LEN; this is asserted equal to
`LGBM_BoosterCalcNumPredict''s count rather than trusted silently, since the buffer was sized
from the latter and a mismatch would mean either an under-filled result or a write past the
allocated buffer going unnoticed.

No prediction call here ever states the result's SHAPE, and that count is the whole of what one
reports bearing on it: `LGBM_BoosterCalcNumPredict' returns a number and nothing else, and
neither entry point reports axes the way XGBoost's `out_shape'/`out_dim' pair does -- so the
SECOND value is DERIVED rather than reported. `%prediction-shape' above is where that happens,
from the element count, the row count and BOOSTER's own feature count -- the last read by a
further library call, `LGBM_BoosterGetNumFeature', which runs inside this function's own
`with-foreign-float-traps-masked' body wrap like every other call it makes. It states a shape
only for the KINDs those three determine one for: `:normal' and `:raw' get the result array's
own dimensions, `:contrib' the three axes `contrib-shape' divides the count into (NIL for any
of the four cases that function's own docstring enumerates), and `:leaf-index' NIL. This
backend declares `:prediction-shape' in `*provided-capabilities*' to say the mechanism is here;
nothing re-checks that declaration, there being no argument to refuse. The first value is
untouched by all of it -- same dimensions, same elements, every KIND, either entry point.

Deliberately does not scan the result for NaN or infinity -- see
`cl-gbdt/src/xgboost/protocol''s `predict' for the identical reasoning, which applies here
unchanged: `with-foreign-float-traps-masked' restores the C calling convention around this
call, it does not and should not decide what counts as a valid model output."
  (with-foreign-float-traps-masked
    ;; POINTER is bound first, and a `let' evaluates its init forms left to right, so a freed
    ;; BOOSTER is refused by `handle-live-pointer' before `%reject-best-num-iteration' can
    ;; report a :BEST it would also have refused -- the same precedence the `predict' method
    ;; above this layer keeps between a freed handle and its own argument checks.
    (let ((pointer (handle-live-pointer booster))
          (predict-type (%predict-type kind))
          (iteration-count (%resolve-num-iteration
                            (%reject-best-num-iteration booster num-iteration
                                                        "predict's :num-iteration"))))
      ;; The buffer sizing, the OUT-LEN check and the copy-out are identical for both entry
      ;; points and live here once; CALL is the only thing that differs between them, which
      ;; is exactly how much of this function a `csr-matrix' changes.
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
                 ;; The second value. Derived from ELEMENT-COUNT and NROW, both of which this
                 ;; function already held for its buffer sizing, plus BOOSTER's own feature
                 ;; count -- nothing about how MATRIX is laid out enters into it, which is why
                 ;; one call here serves both entry points. See `%prediction-shape' above.
                 (values result (%prediction-shape kind result element-count nrow pointer)))))
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
