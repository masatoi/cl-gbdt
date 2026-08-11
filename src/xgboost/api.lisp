;;;; api.lisp --- XGBoost's Layer 1 operations, over the handle classes.
;;;;
;;;; `native.lisp' holds the `%'-functions: each takes and returns raw pointers, and none of
;;;; them is a finished operation. `classes.lisp' holds the CLOS types and the shared library's
;;;; lifetime. This file is what a caller of `cl-gbdt/xgboost' actually invokes -- operations
;;;; that take a backend or a handle, do the whole job, and hand back a handle or a result.
;;;;
;;;; `slice-model' is here for a reason of its own, and moved out of `classes.lisp' when this
;;;; file appeared. It was Layer 1 already -- a public XGBoost-specific entry point, never a
;;;; protocol method -- and it is an operation over the booster class rather than any part of
;;;; the shared library's lifetime, which is the whole of what `classes.lisp' holds besides the
;;;; types themselves. It could never live in `native.lisp' beside its siblings
;;;; `evaluate-one-iteration' and `booster-boosted-rounds': it returns a NEW booster, so
;;;; `make-handle' must name the concrete `xgboost-booster' class, and `native.lisp' must not
;;;; depend on `classes.lisp' (policy section 11). That constraint is what pinned it to
;;;; `classes.lisp'; this file names `classes.lisp' too, so the constraint no longer picks that
;;;; file out. See its own Model slicing section below.
;;;;
;;;; Every OPERATION here reaches the shared library, so every one wraps its whole body in
;;;; `with-foreign-float-traps-masked' -- see `protocol.lisp''s header for why, and
;;;; `tools/ci/check-float-traps.lisp', which scans this file and holds every name the sibling
;;;; `all.lisp' exports to that rule. The `%'-prefixed helpers are not separately wrapped and
;;;; are not policed by that check. `%dataset-pointer' is the only one of the three that reaches
;;;; the library at all, and it runs inside the body wrap of its one caller, `create-dataset';
;;;; it says so where it is defined, since nothing in CI would notice a future caller reaching
;;;; it from outside an already-masked extent. `%check-sparse-input' and
;;;; `%creation-function-name' make no foreign call -- the first reads a capability plist, the
;;;; second returns a constant string -- which is what makes them safe for
;;;; `cl-gbdt/src/xgboost/protocol' to call, as it does both.
;;;;
;;;; Nothing here may depend on `cl-gbdt/src/protocol' or the training files: this file is Layer
;;;; 1, and `tools/ci/check-layer-separation.lisp' fails the build if it ever does. That is not
;;;; a stylistic preference. A caller who loaded `cl-gbdt/xgboost' alone has no unified API in
;;;; the image, and these functions are the whole of what they can call.
;;;;
;;;; Loads after `classes.lisp' and cannot precede it: every operation below takes or returns an
;;;; `xgboost-dataset' or an `xgboost-booster', and `make-handle' is handed a class name as a
;;;; symbol.

(uiop:define-package #:cl-gbdt/src/xgboost/api
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/xgboost/classes
                #:xgboost-booster
                #:xgboost-dataset)
  (:import-from #:cl-gbdt/src/xgboost/native
                #:%check-backend-open
                #:%check-booster-datasets-live
                #:%check-unsupported
                #:%check-xgboost-booster
                #:%check-xgboost-dataset
                #:%create-booster
                #:%create-dmatrix
                #:%create-dmatrix-from-csr
                #:%free-booster
                #:%free-booster-unchecked
                #:%free-dmatrix
                #:%free-dmatrix-unchecked
                #:%set-feature-names
                #:%set-feature-types
                #:%set-group-field
                #:%set-info-field
                #:%set-parameters
                #:%slice
                #:%update-one-iteration)
  (:import-from #:cl-gbdt/src/backend
                #:backend-name
                #:backend-open-p
                #:backend-supports-p)
  (:import-from #:cl-gbdt/src/conditions
                #:capability-unavailable
                #:foreign-call-error
                #:missing-training-set)
  (:import-from #:cl-gbdt/src/data
                #:csr-matrix
                #:csr-matrix-indices
                #:csr-matrix-indptr
                #:csr-matrix-num-columns
                #:csr-matrix-values)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:import-from #:cl-gbdt/src/handle
                #:booster-training-set
                #:handle-backend
                #:handle-live-pointer
                #:handle-released-p
                #:release-handle
                #:with-pointer-ownership)
  (:export #:%check-sparse-input
           #:%creation-function-name
           #:create-booster
           #:create-dataset
           #:free-booster
           #:free-dataset
           #:slice-model
           #:update-one-iteration))

(in-package #:cl-gbdt/src/xgboost/api)

;;; ---------------------------------------------------------------------------
;;; The `:sparse-input' gate

(defun %check-sparse-input (backend)
  "Signal `capability-unavailable' when BACKEND's `:sparse-input' capability reads false.

Policy section 7 requires the operation itself to re-check a capability rather than trusting
the caller to have asked `backend-supports-p' first, so a caller who never asked gets a typed
condition instead of a missing-symbol crash -- the same rule `slice-model' at the end of this
file follows for `:model-slicing'. Both operations this backend gates on `:sparse-input' call
this -- `%dataset-pointer' below, on `create-dataset''s behalf, and
`cl-gbdt/src/xgboost/protocol''s `predict' -- so the two cannot come to disagree about which
capability they name or which backend they blame.
Mirrors `cl-gbdt/src/lightgbm/api''s function of the same name.

Exported from this package, unlike `create-dataset' and its siblings, but NOT from
`cl-gbdt/xgboost': `predict' still holds the prediction procedure in `protocol.lisp' and calls
this, so the symbol has to cross a package boundary, while an internal capability gate on the
backend's public surface would be one more claim to keep true. `all.lisp' imports this
package's public names one at a time rather than re-exporting it whole for exactly that
reason.

Only a `csr-matrix' argument ever reaches this. A dense matrix needs neither
`XGDMatrixCreateFromCSR' nor `XGBoosterPredictFromCSR' to exist, and must keep working on a
library that has neither."
  (unless (backend-supports-p backend :sparse-input)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :sparse-input)))

;;; ---------------------------------------------------------------------------
;;; Datasets

(defun %creation-function-name (matrix)
  "Return the name of the C entry point a DMatrix would be built from MATRIX with:
`XGDMatrixCreateFromCSR' for a `csr-matrix', `XGDMatrixCreateFromDense' for anything
`with-foreign-matrix' accepts.

Separate from `%dataset-pointer', which returns the same string alongside the pointer it
built, because `cl-gbdt/src/xgboost/protocol''s `make-dataset' refuses :PARAMETERS before any
pointer exists and its refusal has to name the call the caller's own arguments would have
reached. Telling a caller who passed a `csr-matrix' about `XGDMatrixCreateFromDense''s config
JSON names a function that call was never going to make. That method is why this is exported
from this package at all: the refusal it words is the portable contract and stays at Layer 2,
while the name it words it with is a fact about this library's own entry points and belongs
here beside the calls that make them.

Makes no foreign call -- it reads MATRIX's own type and returns a constant string -- so it
needs no float-trap mask of its own and none of its callers has to establish one for it."
  (if (typep matrix 'csr-matrix) "XGDMatrixCreateFromCSR" "XGDMatrixCreateFromDense"))

(defun %dataset-pointer (backend matrix missing)
  "Return two values: the raw DMatrix pointer built from MATRIX, and the name of the C
function that produced it, for the null-handle check `create-dataset' makes afterward.

MATRIX is either a `csr-matrix' -- `XGDMatrixCreateFromCSR', through
`%create-dmatrix-from-csr' -- or anything `with-foreign-matrix' accepts --
`XGDMatrixCreateFromDense', through `%create-dmatrix'. MISSING, the value that means
*missing*, or NIL for this backend's own default, reaches both entry points identically:
it is a key in the creation config JSON either way, and means nothing different for a
sparse matrix than for a dense one.

The `:sparse-input' capability is re-checked on the sparse branch rather than assumed --
`%check-sparse-input' above, which carries the reasoning. The `:missing-value' capability is
NOT checked here or anywhere else in this file: it gates the portable :MISSING argument of
`cl-gbdt/src/xgboost/protocol''s `make-dataset', which checks it there, and MISSING reaches
both branches alike so there would be no branch for such a check to belong to.

A `defun', not a second `create-dataset' method specialized on `csr-matrix' -- see
`cl-gbdt/src/lightgbm/api''s function of the same name and purpose, which this mirrors, for
why.

Reaches the shared library, and is the one function in this file that does so without a
`with-foreign-float-traps-masked' of its own: it runs inside `create-dataset''s body wrap,
which is its whole trap protection. `%'-prefixed and named by no `:export' clause of the
sibling `all.lisp', so `tools/ci/check-float-traps.lisp' does not police it -- a caller
reaching this from outside an already-masked dynamic extent must establish the mask itself,
and nothing in CI will notice if it does not."
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

(defun create-dataset (backend matrix &key label weight group feature-names missing
                                        feature-types)
  "Build and return an `xgboost-dataset' -- a DMatrix -- from MATRIX on BACKEND.

MATRIX is a dense matrix -- built via `XGDMatrixCreateFromDense' -- or a `csr-matrix', built
via `XGDMatrixCreateFromCSR'. Nothing else about this function varies with which of the two it
is: see `%dataset-pointer' above, the only form here that branches on it.

LABEL and WEIGHT are attached to the finished DMatrix with `XGDMatrixSetInfoFromInterface'
under those XGBoost field names, GROUP with `XGDMatrixSetUIntInfo', and FEATURE-NAMES and
FEATURE-TYPES with `XGDMatrixSetStrFeatureInfo' under its `\"feature_name\"' and
`\"feature_type\"' fields. Each is attached only when supplied; a NIL one is not written as an
empty field, so a caller who passes nothing gets a DMatrix with no such field at all rather
than a vector of defaults.

MISSING is the value in MATRIX that means *missing* -- a real, or NIL for this library's own
default, the IEEE NaN. It becomes the `\"missing\"' key of whichever creation config JSON
MATRIX's own form reaches, so it is honoured identically on both paths. The comparison the
library then makes is at SINGLE precision, whatever MATRIX's element type: two `double-float's
that share a `single-float' both count as missing against a sentinel that narrows to it.

FEATURE-TYPES is a list of strings, one per column, in XGBoost'S OWN vocabulary: `\"c\"' for a
categorical column and `\"q\"' for a quantitative one. Nothing here renders it, range-checks it
or knows what a category is -- `cl-gbdt/src/xgboost/protocol''s `make-dataset' is what turns
the portable :CATEGORICAL-FEATURES column list into exactly this, by
`categorical-feature-types', before calling this. A list of the wrong length is XGBoost's own
to refuse.

There is no :PARAMETERS and no :REFERENCE, and their absence is not a refusal: this library's
creation config JSON documents three keys -- `\"missing\"', which has its own argument above,
`\"nthread\"' and `\"data_split_mode\"' -- and has no concept resembling LightGBM's bin-mapper
alignment at all, so there is nothing at this layer for either argument to name.
`cl-gbdt/src/xgboost/protocol''s `make-dataset' does signal `unsupported-argument' for both,
because the unified generic promises the arguments and a caller moving a working call across
backends has to be told rather than silently ignored; see that method's docstring for the
measurement behind it.

Signals `backend-not-open' before any foreign call when BACKEND is not open -- see
`%check-backend-open'. Signals `capability-unavailable' naming `:sparse-input' when MATRIX is a
`csr-matrix' and that capability reads false -- see `%check-sparse-input'. Signals
`foreign-call-error' when the creation call reports success but writes a null handle: a
library-contract violation, but one every later call through this handle would otherwise
dereference blindly.

The raw DMatrix handle exists in C from the moment the creation call returns, but `make-handle'
does not take ownership of it until the very end -- attaching LABEL, WEIGHT, GROUP,
FEATURE-NAMES or FEATURE-TYPES can each signal first (a wrong-length LABEL is the commonest
way). `with-pointer-ownership' spans exactly that gap: the pointer is owned by nobody inside
its body, and any exit that has not called TAKE-OWNERSHIP frees the raw DMatrix here instead of
orphaning it."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (multiple-value-bind (dataset-pointer function-name)
        (%dataset-pointer backend matrix missing)
      (when (cffi:null-pointer-p dataset-pointer)
        (error 'foreign-call-error
               :function-name function-name
               :code 0
               :message "reported success but returned a null dataset handle"))
      (with-pointer-ownership (dataset-pointer #'%free-dmatrix-unchecked take-ownership)
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
        (take-ownership 'xgboost-dataset backend :dataset)))))

(defun free-dataset (dataset)
  "Free DATASET via `XGDMatrixFree'. Does nothing if it was already freed, and returns no
useful value.

Unlike every other operation that reads an existing handle, this does not go through
`handle-live-pointer' and so does not signal `backend-not-open' when DATASET's backend has
already been closed. `free-dataset' runs from `with-dataset''s `unwind-protect' cleanup form,
and a non-local exit is exactly when that cleanup runs; signalling there would replace whatever
condition is already unwinding the stack instead of letting it propagate. So when the backend
is closed, the handle is instead marked released without calling `XGDMatrixFree' -- the shared
library may no longer be mapped into the process, so that call cannot be trusted not to crash
-- and a `warn' reports the foreign memory as leaked, since it is genuinely unreclaimable at
that point."
  (with-foreign-float-traps-masked
    (if (backend-open-p (handle-backend dataset))
        (release-handle dataset (lambda (pointer) (%free-dmatrix pointer)))
        (let ((already-released (handle-released-p dataset)))
          (release-handle dataset (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing an XGBoost dataset after its backend was closed: the foreign ~
                   dataset was not freed and its memory is leaked."))))))

;;; ---------------------------------------------------------------------------
;;; Boosters
;;;
;;; `cl-gbdt/src/xgboost/protocol''s `train' does NOT call `create-booster', and a reader who
;;; assumes every training run exercises it would be wrong. `train' must hand `make-handle' a
;;; `:best-iteration' its own loop computes, and `booster-best-iteration' is a `:reader'-only
;;; slot set at construction, so `train' has to own the pointer across its whole loop and build
;;; the handle at the end; `create-booster' builds it at the start, by the same argument every
;;; other Layer 1 operation follows. See `train''s own call site, which carries that reasoning
;;; where an editor tempted to merge the two will meet it -- and note that the same is true of
;;; `cl-gbdt/src/lightgbm/api''s `create-booster', for the same two reasons: this is a property
;;; of the shared `handle' class and of what a training loop owns, not of either library. The
;;; two small functions below ARE what their methods call, wholesale.

(defun create-booster (backend dataset &key parameters valid-sets)
  "Create a booster over DATASET on BACKEND via `XGBoosterCreate', returning an
`xgboost-booster'.

The result is UNTRAINED: `XGBoosterCreate' allocates the model and every boosting iteration
comes from a later `update-one-iteration'. Free it with `free-booster'.

VALID-SETS is a list of `xgboost-dataset's -- bare datasets, not the (NAME . DATASET) entries
`cl-gbdt''s `train' accepts, a name being a training-report concept with no meaning at this
layer. Unlike LightGBM's `LGBM_BoosterCreate', which takes the training set alone and gains
validation sets afterward through `LGBM_BoosterAddValidData', `XGBoosterCreate' takes the
whole array of DMatrix handles a booster will ever reference AT ONCE -- the training set
first, then each validation set in the order given -- and this library has no \"add valid
data\" entry point at all. Nothing is attached after creation here because there is nothing to
attach it with: a validation set left out of the creation call could never be added later.

PARAMETERS reaches the booster the other way round for the same reason of asymmetry: a plist
in XGBOOST'S OWN vocabulary, applied AFTER creation by `%set-parameters', one
`XGBoosterSetParam' call per pair, this library having no bulk-parameter argument the way
`LGBM_BoosterCreate''s parameter string is. Nothing here translates a key or a value, and no
key is added. A parameter XGBoost refuses signals `foreign-call-error' from inside the
ownership form below, which frees the raw booster rather than orphaning it -- so a rejected
parameter, unlike a rejected VALID-SETS entry, does leave a booster briefly in existence.

The booster retains DATASET and a COPY of VALID-SETS, which keeps them alive for its lifetime
and lets `update-one-iteration' notice a dataset freed out from under it. The copy is what
makes that promise hold: were the caller's own list object stored, a later `delete' or
`(setf (cdr ...))' on it would remove an entry from the booster's view while XGBoost still
held that dataset's pointer.

Signals `backend-not-open' before any foreign call when BACKEND is not open -- see
`%check-backend-open' -- `wrong-backend-reference' when DATASET or a VALID-SETS entry is not
an `xgboost-dataset', and `released-handle-error' or `backend-not-open' when one is but has
already been freed or had its own backend closed; see `%check-xgboost-dataset'. That check
runs on EVERY element of the array this backend hands `XGBoosterCreate', not on DATASET alone,
and it is what rules out a booster's own pointer arriving there as a DMatrix handle. This
function dispatches on nothing, so it is the only thing standing between a wrong-kind handle
and a segfault. Signals `foreign-call-error' when creation reports success but writes a null
handle -- that check lives in `%create-booster', beside the call it guards, and is not
repeated here.

Every check runs before the creation call, so a rejected VALID-SETS entry leaves no booster in
existence at all. The raw handle then exists in C from the moment that call returns and
nothing in Lisp references it until `make-handle' runs -- `with-pointer-ownership' spans
exactly that gap, so a parameter that fails to apply frees the booster rather than orphaning
it."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    ;; `let', not `let*': no binding here reads another, and the checks still run before the
    ;; creation call below because a `let' evaluates its init forms left to right, which is
    ;; the whole ordering this function needs. `%create-booster' is deliberately NOT among
    ;; them -- it belongs inside its own form, where the raw handle's lifetime begins.
    (let ((train-data-pointer
            (%check-xgboost-dataset backend dataset "create-booster's dataset argument"
                                     'xgboost-dataset))
          (valid-set-pointers
            (mapcar (lambda (valid-set)
                      (%check-xgboost-dataset backend valid-set
                                               "a create-booster :valid-sets entry"
                                               'xgboost-dataset))
                    valid-sets))
          (validation-sets (copy-list valid-sets)))
      ;; The training set first, then each validation set: `XGBoosterCreate' takes the array
      ;; positionally and `cl-gbdt/src/xgboost/protocol''s `train' builds the identical list
      ;; the identical way, which is what makes a booster built here and one built there hold
      ;; the same DMatrix handles in the same order.
      (let ((booster-pointer (%create-booster (cons train-data-pointer valid-set-pointers))))
        (with-pointer-ownership (booster-pointer #'%free-booster-unchecked take-ownership)
          (%set-parameters booster-pointer parameters)
          (take-ownership 'xgboost-booster backend :booster
                          :training-set dataset
                          :validation-sets validation-sets))))))

(defun update-one-iteration (booster)
  "Advance BOOSTER by one boosting iteration via `XGBoosterUpdateOneIter'. Always returns
true after a successful call.

Unlike LightGBM's `LGBM_BoosterUpdateOneIter', which reads the booster's internal
training-set pointer implicitly, XGBoost's version takes the DMatrix handle explicitly, so
this reads it back from `booster-training-set' rather than being able to omit it. A
`load-model' booster's training set is NIL by design -- see the `booster' class'
documentation -- and handing `XGBoosterUpdateOneIter' a null DMatrixHandle would not come back
as a status code the way a bad parameter does: it is a null-pointer dereference inside
XGBoost's own implementation. That case is rejected here, before the foreign call, for the
same reason `%check-booster-datasets-live' exists for the pointers it does check.

XGBoost also reports no `produced_empty_tree'-style signal from this call, unlike LightGBM, so
there is nothing here to return false for -- where `cl-gbdt/src/lightgbm/api''s function of
the same name returns false once an iteration produced no further split, this one cannot tell,
and says so by always returning true.

Signals `released-handle-error' when BOOSTER's training set, or any of its validation sets,
has already been freed -- see `%check-booster-datasets-live', which runs before any foreign
call because `XGBoosterUpdateOneIter' dereferences those datasets' pointers itself and a freed
one is a segfault rather than a catchable condition. Signals `missing-training-set' when
BOOSTER has no training set at all. Signals `released-handle-error' for a freed BOOSTER, and
`backend-not-open' when its backend has since been closed -- see `handle-live-pointer'."
  (with-foreign-float-traps-masked
    (%check-booster-datasets-live booster)
    (let ((training-set (booster-training-set booster)))
      (unless training-set
        (error 'missing-training-set :booster booster))
      (%update-one-iteration (handle-live-pointer booster) (handle-live-pointer training-set)))
    t))

(defun free-booster (booster)
  "Free BOOSTER via `XGBoosterFree'. Does nothing if it was already freed, and returns no
useful value.

See `free-dataset' above for why this does not signal `backend-not-open' when BOOSTER's
backend has already been closed, but marks the handle released and `warn's the foreign memory
leaked instead -- the same cleanup-form reasoning applies here, `with-booster' being the macro
whose `unwind-protect' calls this one."
  (with-foreign-float-traps-masked
    (if (backend-open-p (handle-backend booster))
        (release-handle booster (lambda (pointer) (%free-booster pointer)))
        (let ((already-released (handle-released-p booster)))
          (release-handle booster (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing an XGBoost booster after its backend was closed: the foreign ~
                   booster was not freed and its memory is leaked."))))))

;;; ---------------------------------------------------------------------------
;;; Model slicing
;;;
;;; `slice-model' returns a NEW booster: `make-handle' needs the concrete class
;;; `xgboost-booster', which is defined in `classes.lisp', and `native.lisp' must not depend on
;;; that file (policy section 11). That is why it sat in `classes.lisp' until this file existed
;;; -- see this file's header -- and it follows `cl-gbdt/src/xgboost/protocol''s `load-model'
;;; unchanged in shape: the guards and the handle construction here, the foreign call delegated
;;; to a `%'-function in `native.lisp' (`%slice'). See that file's own Model slicing section for
;;; the other half.
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
      ;; Unlike `cl-gbdt/src/xgboost/protocol''s `load-model' and `train', no further foreign
      ;; call runs between the handle appearing in C and `make-handle' taking ownership of it
      ;; -- `%slice' returns a booster that is already complete, and `make-handle' is the very
      ;; next thing that runs.
      ;; `with-pointer-ownership' is still needed, though: `make-handle' itself --
      ;; `make-instance' or finalizer attachment -- can signal, e.g. on `storage-condition',
      ;; and a signal there must not orphan the foreign booster `%slice' already returned.
      (let ((slice-pointer (%slice pointer begin (or end 0) step)))
        (with-pointer-ownership (slice-pointer #'%free-booster-unchecked take-ownership)
          (take-ownership 'xgboost-booster backend :booster))))))
