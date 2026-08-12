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
;;;; second returns a constant string. That is what makes the second safe for
;;;; `cl-gbdt/src/xgboost/protocol''s `make-dataset' to call, as it does; the first is called
;;;; only from inside this file now that `predict' lives here.
;;;;
;;;; Nothing here may depend on `cl-gbdt/src/protocol' or the training files: this file is Layer
;;;; 1, and `tools/ci/check-layer-separation.lisp' fails the build if it ever does. That is not
;;;; a stylistic preference. A caller who loaded `cl-gbdt/xgboost' alone has no unified API in
;;;; the image, and these functions are the whole of what they can call.
;;;;
;;;; Loads after `classes.lisp' and cannot precede it: every operation below takes or returns an
;;;; `xgboost-dataset' or an `xgboost-booster', and `make-handle' and `%check-object-class' are
;;;; each handed a class name as a symbol.
;;;;
;;;; Every operation below that takes a caller-supplied HANDLE checks its kind before letting
;;;; its pointer reach C, and none of them may skip that: each was a `defmethod' specialized on
;;;; `xgboost-dataset' or `xgboost-booster' before the Layer 1 split, and that specializer WAS
;;;; the check. A plain `defun' takes whatever it is given. `%check-xgboost-dataset' and
;;;; `%check-xgboost-booster' are what the operations that require a LIVE handle use;
;;;; `%check-object-class' below is what `free-dataset' and `free-booster' use instead, they
;;;; being the two that must keep working on a handle that is neither, and what
;;;; `dataset-num-rows' and `dataset-num-features' pair with `handle-live-pointer' for, neither
;;;; having a BACKEND argument of its own to hand `%check-xgboost-dataset' for its report.
;;;;
;;;; The three operations that take a caller-supplied BACKEND rather than a handle --
;;;; `create-dataset', `create-booster' and `load-model' -- are under the identical rule for
;;;; the identical reason, and `%check-object-class' is their check too, handed
;;;; `xgboost-backend' where the frees hand it a handle class. Their `defmethod' ancestors
;;;; specialized on `xgboost-backend', so the backend argument was type-checked by the same
;;;; mechanism the handle arguments were, and `%check-backend-open' does not replace it: that
;;;; function asks whether the object is OPEN, not whose it is.

(uiop:define-package #:cl-gbdt/src/xgboost/api
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/xgboost/classes
                #:xgboost-backend
                #:xgboost-booster
                #:xgboost-dataset)
  (:import-from #:cl-gbdt/src/xgboost/native
                #:%booster-num-features
                #:%check-backend-open
                #:%check-booster-datasets-live
                #:%check-feature-score-dim
                #:%check-unsupported
                #:%check-xgboost-booster
                #:%check-xgboost-dataset
                #:%create-booster
                #:%create-dmatrix
                #:%create-dmatrix-from-csr
                #:%dataset-num-features
                #:%dataset-num-rows
                #:%feature-importance-type
                #:%feature-score
                #:%feature-score-index
                #:%free-booster
                #:%free-booster-unchecked
                #:%free-dmatrix
                #:%free-dmatrix-unchecked
                #:%load-model
                #:%predict-config-json
                #:%predict-from-csr
                #:%predict-from-dmatrix
                #:%predict-ncol
                #:%predict-type
                #:%read-evaluation
                #:%reported-shape
                #:%resolve-num-iteration
                #:%save-model
                #:%save-model-to-buffer
                #:%set-feature-names
                #:%set-feature-types
                #:%set-group-field
                #:%set-info-field
                #:%set-parameters
                #:%slice
                #:%total-element-count
                #:%update-one-iteration)
  (:import-from #:cl-gbdt/src/backend
                #:backend-name
                #:backend-open-p
                #:backend-supports-p)
  (:import-from #:cl-gbdt/src/conditions
                #:capability-unavailable
                #:foreign-call-error
                #:missing-training-set
                #:wrong-backend-reference)
  (:import-from #:cl-gbdt/src/data
                #:csr-matrix
                #:csr-matrix-indices
                #:csr-matrix-indptr
                #:csr-matrix-num-columns
                #:csr-matrix-num-rows
                #:csr-matrix-values)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:import-from #:cl-gbdt/src/handle
                #:%reject-best-num-iteration
                #:booster-training-set
                #:booster-validation-sets
                #:handle-backend
                #:handle-live-pointer
                #:handle-released-p
                #:release-handle
                #:with-pointer-ownership)
  (:export #:%creation-function-name
           #:create-booster
           #:create-dataset
           #:dataset-num-features
           #:dataset-num-rows
           #:evaluation
           #:feature-importance
           #:free-booster
           #:free-dataset
           #:load-model
           #:model-to-string
           #:predict
           #:save-model
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
this -- `%dataset-pointer' below, on `create-dataset''s behalf, and `predict' further down
this file -- so the two cannot come to disagree about which capability they name or which
backend they blame.
Mirrors `cl-gbdt/src/lightgbm/api''s function of the same name.

Not exported, unlike every other name this file defines that a caller reaches. It was, while
`cl-gbdt/src/xgboost/protocol''s `predict' held the prediction procedure and imported this
gate from here; that procedure is now `predict' below, both call sites are in this file, and
an export nothing outside it needs is one more claim to keep true. `all.lisp' imports this
package's public names one at a time rather than re-exporting it whole, so the symbol was
never on `cl-gbdt/xgboost''s surface either way.

Only a `csr-matrix' argument ever reaches this. A dense matrix needs neither
`XGDMatrixCreateFromCSR' nor `XGBoosterPredictFromCSR' to exist, and must keep working on a
library that has neither."
  (unless (backend-supports-p backend :sparse-input)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :sparse-input)))

;;; ---------------------------------------------------------------------------
;;; The class gate, for the frees and the creators

(defun %check-object-class (object class noun argument-description)
  "Signal `wrong-backend-reference' unless OBJECT is of type CLASS, reporting NOUN as the kind
of thing that was wanted and ARGUMENT-DESCRIPTION as the argument OBJECT came from. Returns
no useful value, and reads nothing else about OBJECT at all. Seven callers, in three groups:
`free-dataset' and `free-booster', where this is what stands between a wrong-kind pointer and
`XGDMatrixFree' or `XGBoosterFree' dereferencing it; `create-dataset', `create-booster' and
`load-model', where OBJECT is the BACKEND -- which is why this is not named for handles -- and
where each calls this FIRST, ahead of `%check-backend-open', that check asking only whether
the object is OPEN; and `dataset-num-rows' and `dataset-num-features', which need a LIVE
handle, unlike the frees, but have no BACKEND argument of their own to hand
`%check-xgboost-dataset' for its report -- neither signature takes one, both being readers of
an existing DATASET rather than builders -- so each pairs this check with `handle-live-pointer'
instead, calling it only once this one has already confirmed DATASET's kind:
`wrong-backend-reference' for the wrong kind, and whatever `handle-live-pointer' signals
otherwise, `released-handle-error' for an already-freed DATASET or `backend-not-open' for one
whose backend has since closed. All seven were `defmethod's specialized on the concrete class
before the Layer 1 split, and that specializer was this check.

The mirror of `cl-gbdt/src/lightgbm/api''s function of the same name, WHICH CARRIES THE WHOLE
ARGUMENT: why the two frees cannot use `%check-xgboost-dataset' or `%check-xgboost-booster'
the way every operation in this file that requires a live handle does, why the creators'
openness check does not subsume this one and why their call must precede it, why a `typep'
against the CONCRETE class subsumes kind and backend where `cl-gbdt/src/handle''s
`%check-handle-kind' needs a pair, why NOUN is passed rather than derived from CLASS, and what
the two libraries measurably did without the check at either group of call sites. None of it is
repeated here: the reasoning is identical for both backends -- it is about the shape of a
Layer 1 `defun' and of a package-inferred dependency edge, not about either C API -- so a
second copy would be a second thing to keep true. Read it there. Measured on this side,
`create-dataset' handed a LIGHTGBM backend likewise ACCEPTED it and returned an
`xgboost-dataset' whose `handle-backend' names `:lightgbm'.

The two definitions are structurally forced even though the reasoning is not: each names its
own backend's concrete classes, and `src/lightgbm/api.lisp' and this file depend on neither
each other nor any shared file that could hold one copy."
  (unless (typep object class)
    (error 'wrong-backend-reference
           :backend :xgboost
           :given (class-name (class-of object))
           :argument argument-description
           :expected noun)))

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

Signals `wrong-backend-reference' when BACKEND is not an `xgboost-backend' -- the other
backend's object, or not a backend at all -- before anything else is read from it and ahead of
the openness check below, which for another backend's object would answer about the wrong
shared library; see `%check-object-class'. Signals `backend-not-open' before any foreign
call when BACKEND is not open -- see `%check-backend-open'. Signals `capability-unavailable'
naming `:sparse-input' when MATRIX is a `csr-matrix' and that capability reads false -- see
`%check-sparse-input'. Signals `foreign-call-error' when the creation call reports success but
writes a null handle: a library-contract violation, but one every later call through this
handle would otherwise dereference blindly.

The raw DMatrix handle exists in C from the moment the creation call returns, but `make-handle'
does not take ownership of it until the very end -- attaching LABEL, WEIGHT, GROUP,
FEATURE-NAMES or FEATURE-TYPES can each signal first (a wrong-length LABEL is the commonest
way). `with-pointer-ownership' spans exactly that gap: the pointer is owned by nobody inside
its body, and any exit that has not called TAKE-OWNERSHIP frees the raw DMatrix here instead of
orphaning it."
  (with-foreign-float-traps-masked
    (%check-object-class backend 'xgboost-backend "backend"
                         "create-dataset's backend argument")
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

Signals `wrong-backend-reference' when DATASET is not an `xgboost-dataset' -- a booster, a
dataset built by another backend, or not a handle at all -- before anything is read from it
and before any foreign call. This function dispatches on nothing, so `%check-object-class' is
the only thing between a wrong-kind pointer and `XGDMatrixFree' dereferencing it; see that
function for why the frees cannot use the same check the rest of this file uses.

Unlike every other operation that reads an existing handle -- `free-booster' below excepted,
which takes this same path for this same reason -- this does not go through
`handle-live-pointer' and so does not signal `backend-not-open' when DATASET's backend has
already been closed. A free runs from a cleanup form -- the `unwind-protect' a Layer 1 caller
writes for itself, as tests/functional/xgboost-standalone.lisp does, or the one inside
`cl-gbdt''s `with-dataset', which reaches this function through the method that delegates to it
and which a caller of `cl-gbdt/xgboost' alone does not have -- and a non-local exit is exactly
when that cleanup runs; signalling there would replace whatever
condition is already unwinding the stack instead of letting it propagate. So when the backend
is closed, the handle is instead marked released without calling `XGDMatrixFree' -- the shared
library may no longer be mapped into the process, so that call cannot be trusted not to crash
-- and a `warn' reports the foreign memory as leaked, since it is genuinely unreclaimable at
that point."
  (with-foreign-float-traps-masked
    (%check-object-class dataset 'xgboost-dataset "dataset"
                         "free-dataset's dataset argument")
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
key is added. Nor does anything REJECT one: measured against the vendored library,
`XGBoosterSetParam' validates nothing at all -- an unknown key, a non-numeric `eta', an
objective that does not exist and a bogus `tree_method' were each accepted here without a
status code, and the last three surfaced later, as a `foreign-call-error' out of the first
call that makes XGBoost configure its learner, which is the `XGBoosterBoostedRounds' inside
`update-one-iteration'. The unknown key never surfaced at all. So this function returns a
booster for parameters it cannot honour, and the error, when there is one, names a C function
the caller never wrote.

The booster retains DATASET and a COPY of VALID-SETS, which keeps them alive for its lifetime
and lets `update-one-iteration' notice a dataset freed out from under it. The copy is what
makes that promise hold: were the caller's own list object stored, a later `delete' or
`(setf (cdr ...))' on it would remove an entry from the booster's view while XGBoost still
held that dataset's pointer.

Signals `wrong-backend-reference' when BACKEND is not an `xgboost-backend' -- the other
backend's object, or not a backend at all -- before anything else is read from it and ahead of
the openness check, which for another backend's object would answer about the wrong shared
library; see `%check-object-class'. Signals `backend-not-open' before any foreign call when
BACKEND is not open -- see `%check-backend-open' -- `wrong-backend-reference' when DATASET or
a VALID-SETS entry is not an `xgboost-dataset', and `released-handle-error' or
`backend-not-open' when one is but has already been freed or had its own backend closed; see
`%check-xgboost-dataset'. That check runs on EVERY element of the array this backend hands
`XGBoosterCreate', not on DATASET alone, and it is what rules out a booster's own pointer
arriving there as a DMatrix handle. This function dispatches on nothing, so those two checks
are the only thing standing between a wrong-kind handle -- or another backend's object where
this one belongs -- and a segfault. Signals `foreign-call-error' when creation reports success
but writes a null handle -- that check lives in `%create-booster', beside the call it guards,
and is not repeated here.

Every check runs before the creation call, so a rejected VALID-SETS entry leaves no booster in
existence at all. The raw handle then exists in C from the moment that call returns and
nothing in Lisp references it until `make-handle' runs -- `with-pointer-ownership' spans
exactly that gap, so anything signalling in between frees the booster rather than orphaning
it. `%set-parameters' is the whole of what runs in that gap, and it does signal, just not for
any of the parameters the measurement above covers: it renders PARAMETERS through
`normalize-parameters', which signals `data-error' for an ODD-LENGTH plist rather than let a
final key go silently missing. Measured -- `:parameters '(:eta)' signals `data-error' from
inside this form and `%free-booster-unchecked' runs exactly once. That is this form's live
failure mode, not a prospective one, and it is what a value XGBoost itself would have
tolerated cannot produce.

`cl-gbdt/src/lightgbm/api''s `create-booster' looks symmetrical here and is not. The same odd
plist signals the same `data-error' there, but with NO booster freed -- measured, zero calls
to that backend's `%free-booster-unchecked' -- because `%parameter-string' is an ARGUMENT to
`LGBM_BoosterCreate' and so runs before any handle exists. What its own ownership form catches
instead is `%add-valid-data' refusing a mismatched bin mapper. The asymmetry is the one this
docstring opens with, reaching all the way down: parameters go in at creation there and after
creation here."
  (with-foreign-float-traps-masked
    (%check-object-class backend 'xgboost-backend "backend"
                         "create-booster's backend argument")
    (%check-backend-open backend)
    ;; `let', not `let*': no binding here reads another, so the sequential scope `let*' adds
    ;; would claim a dependency that is not there. Ordering is NOT what separates the two --
    ;; both evaluate their init forms left to right -- and the ordering this function depends
    ;; on is not among these bindings at all: the checks precede the creation call because
    ;; that call sits in the nested form below, in this `let''s body. `%create-booster' is
    ;; deliberately NOT among these bindings, for exactly that reason -- it belongs inside its
    ;; own form, where the raw handle's lifetime begins.
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
call and whose docstring measures why the two kinds are not the same hazard: a freed TRAINING
set faults inside the library, since its handle is what this call passes to C, while a freed
VALIDATION set is refused on contract rather than for safety, that array not being consulted
here at all. Signals `missing-training-set' when
BOOSTER has no training set at all.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, a LightGBM booster, or not a handle at all -- and this function dispatches on
nothing, so `%check-xgboost-booster' is the only thing between such a handle's pointer and
`XGBoosterUpdateOneIter'. Signals `released-handle-error' for a freed BOOSTER, and
`backend-not-open' when its backend has since been closed -- both from the
`handle-live-pointer' inside `%check-xgboost-booster', which is why the kind check is first: a
handle this backend never built is the wrong handle whatever its state, and both
`%check-booster-datasets-live' and the `booster-training-set' read below would otherwise take
slots off it before anything questioned what it was.

PRECEDENCE, when more than one of those is true at once. BOOSTER's own state is examined
before its datasets' and before the question of whether it has one, so a fault in the booster
or its backend WINS over both. Measured against the vendored library, before and after the
kind check moved ahead of the other two:

  booster freed + training set freed   was `released-handle-error' naming the DATASET,
                                       is `released-handle-error' naming the BOOSTER
  training set freed + backend closed  was `released-handle-error' naming the dataset,
                                       is `backend-not-open'
  freed booster with NO training set   was `missing-training-set',
                                       is `released-handle-error'
  training set freed alone             `released-handle-error' naming the dataset, unchanged

All three changes are deliberate and none is a widening: a released handle, or a shared
library `close-backend' has unmapped, is a more fundamental fault than the dataset a
still-usable booster points at, and the third row was reading a RELEASED booster's slots in
order to discover that it had no training set -- answering a question about a handle that
should not have been read at all. The third row has no LightGBM counterpart, that backend
having no `missing-training-set' guard; `cl-gbdt/src/lightgbm/api''s `update-one-iteration'
records the other two."
  (with-foreign-float-traps-masked
    (let ((booster-pointer
            (%check-xgboost-booster booster "update-one-iteration's booster argument")))
      (%check-booster-datasets-live booster)
      (let ((training-set (booster-training-set booster)))
        (unless training-set
          (error 'missing-training-set :booster booster))
        (%update-one-iteration booster-pointer (handle-live-pointer training-set))))
    t))

(defun free-booster (booster)
  "Free BOOSTER via `XGBoosterFree'. Does nothing if it was already freed, and returns no
useful value.

Signals `wrong-backend-reference' when BOOSTER is not an `xgboost-booster' -- a dataset, a
booster built by another backend, or not a handle at all -- before anything is read from it
and before any foreign call, for the reason `free-dataset' above states and by the same
`%check-object-class'.

See `free-dataset' above for why this does not signal `backend-not-open' when BOOSTER's
backend has already been closed, but marks the handle released and `warn's the foreign memory
leaked instead -- the same cleanup-form reasoning applies here, whether the `unwind-protect'
is the caller's own or the one inside `cl-gbdt''s `with-booster'."
  (with-foreign-float-traps-masked
    (%check-object-class booster 'xgboost-booster "booster"
                         "free-booster's booster argument")
    (if (backend-open-p (handle-backend booster))
        (release-handle booster (lambda (pointer) (%free-booster pointer)))
        (let ((already-released (handle-released-p booster)))
          (release-handle booster (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing an XGBoost booster after its backend was closed: the foreign ~
                   booster was not freed and its memory is leaked."))))))

;;; ---------------------------------------------------------------------------
;;; Inference

(defun predict (booster matrix &key (kind :normal) num-iteration missing)
  "Predict on MATRIX with BOOSTER, returning two values: the result array and the SHAPE this
backend states for it.

MATRIX is a dense matrix -- predicted through `XGBoosterPredictFromDMatrix' -- or a
`csr-matrix', through `XGBoosterPredictFromCSR'. KIND is `:normal', `:raw', `:leaf-index' or
`:contrib', mapped onto XGBoost's own prediction-type number by `%predict-type', which signals
for anything else. Predictions start from iteration 0; nothing here exposes a start-iteration
override.

NUM-ITERATION is a positive integer, or NIL for every iteration -- which XGBoost spells as an
`\"iteration_end\"' of 0, and `%resolve-num-iteration' is what writes it that way. :BEST is
REFUSED, with `unsupported-argument' naming this backend and \"predict's :num-iteration\":
only `train' writes a booster's `best-iteration', and a booster built by `create-booster' has
none, so at this layer the keyword would name an empty slot. `%reject-best-num-iteration' is
what refuses it, and its own docstring says why the refusal has to be explicit --
`%resolve-num-iteration' is `(or num-iteration 0)', so the keyword passes straight through it
as uninterpreted data. MEASURED at this layer with the refusal removed, on both entry points:
:BEST reaches `%predict-config-json''s `~D' directive and renders into the config JSON as the
bare token `BEST' -- `{...,\"iteration_end\":BEST,\"strict_shape\":true}' -- so the call comes
back as a `foreign-call-error' quoting XGBoost's own JSON parser (\"Unknown construct, around
character position: 63\"). That is a `cl-gbdt' condition, unlike the raw CFFI `type-error'
`%reject-best-num-iteration''s docstring records for LightGBM's integer-typed call, but it
names a parse failure at a character offset rather than the argument the caller got wrong, and
its wording is XGBoost's to change. The refusal is invisible to Layer 2:
`cl-gbdt/src/xgboost/protocol''s `predict' method resolves :BEST first -- by
`%resolve-best-num-iteration', which signals `unsupported-argument' when the booster has no
best iteration to resolve it against -- and calls this with the integer that resolution
produced, so the keyword itself never arrives from there.

MISSING is the value in MATRIX that means *missing* -- a real, or NIL for this library's own
default, the IEEE NaN -- and it reaches the library through a DIFFERENT config for each of
MATRIX's two forms, neither of them the one `create-dataset' fills. A dense MATRIX becomes a
transient DMatrix, so its sentinel is a key in THAT DMatrix's creation config --
`%create-dmatrix', exactly as for a dataset that outlives the call. A `csr-matrix' builds no
DMatrix at all, so its sentinel is a key in the INPLACE PREDICT config instead --
`%predict-from-csr', which needs the key anyway. Same argument, same meaning, two config
strings built by two functions: see `%predict-config-json', whose own docstring says why the
dense path leaves the key out of the predict config rather than sending the sentinel twice.
`missing-value-json' signals `unsupported-argument' for a value that is neither a real nor
NIL, and the comparison the library then makes is at SINGLE precision, whatever MATRIX's own
element type, as it is on the ingestion path. The `:missing-value' CAPABILITY is not checked
here: it gates the portable :MISSING argument of `cl-gbdt/src/xgboost/protocol''s `predict',
which checks it there, and MISSING reaches both branches alike so there would be no branch for
such a check to belong to -- the same division `%dataset-pointer' above records for the
ingestion path.

Signals `capability-unavailable' naming `:sparse-input' when MATRIX is a `csr-matrix' and that
capability reads false -- see `%check-sparse-input' above, which checks it before any foreign
call.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, a LightGBM booster, or not a handle at all. This function dispatches on nothing, so
`%check-xgboost-booster' is the only thing between such a handle's pointer and
`XGBoosterPredictFromDMatrix' or `XGBoosterPredictFromCSR'.

Signals `released-handle-error' for a freed BOOSTER, and `backend-not-open' when its backend
has since been closed -- both from the `handle-live-pointer' inside `%check-xgboost-booster',
which is read before anything is allocated, and before NUM-ITERATION is examined, so a freed
booster handed :BEST is reported as freed rather than as having passed a keyword this layer
refuses.

A dense MATRIX is built into a transient DMatrix via `%create-dmatrix' first --
`XGBoosterPredictFromDMatrix' takes a DMatrix handle, unlike LightGBM's
`LGBM_BoosterPredictForMat', which predicts straight off a raw pointer and row/column
counts. It is built first, before anything else here, so a MISSING that
`%dense-matrix-config-json' refuses signals with nothing pinned and no foreign allocation
held -- the property `%create-dmatrix''s own docstring claims. The transient DMatrix is
freed before this returns, on every exit path, since nothing else retains it. Its free is
checked with `check-xgb', not discarded outright: a failure there is reported with `warn'
rather than an error, matching `free-dataset''s own reasoning for warning instead of
signalling, since raising an error from cleanup would replace whatever condition is already
propagating on an unwinding exit -- but on the ordinary success path, a failed free still
leaks foreign memory and is worth reporting rather than passing over in silence.

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
rows through `create-dataset' leads nowhere `predict' can be called on.
That refusal is the library's own and is left to it, exactly as a `csr-matrix' whose
NUM-COLUMNS is not BOOSTER's feature count is (\"Number of columns in data must equal to the
trained model\"). NUM-ITERATION is honoured identically on both paths -- the same
`iteration_begin'/`iteration_end' pair reaches the same config JSON, which additionally
carries the `\"missing\"' key inplace prediction requires; see `%predict-config-json'.

The output buffer's total element count comes from the C call's own `out_shape'/`out_dim'
report, not from the row count alone -- the row count is only
correct for a single-class objective. The second array dimension is that total divided
by the row count, guarded by `%predict-ncol' -- the same derivation
`cl-gbdt/src/lightgbm/native''s `%predict-ncol' makes for its own row-count-alone pitfall,
and the one that
tells a three-class `multi:softprob' model's predictions apart from a binary model's. Both
entry points report it the same way, `\"strict_shape\":true' being set for both.

That same report is also RETURNED, as this function's second value: `%reported-shape' reads
`out_shape' back as a list of integers instead of only multiplying it out, and neither entry
point interprets or reshapes it. Reading the shape BACK is what parts this backend from
`cl-gbdt/src/lightgbm/api''s `%prediction-shape', which has no such call to read and derives
what it can from an element count instead, stating NIL for `:leaf-index'. It is never NIL
here: `out_dim' was measured 2 for `:normal'
and `:raw', 3 for `:contrib' and 4 for `:leaf-index' on both entry points, so
`%reported-shape''s empty-loop case -- a zero DIM -- does not arise. This backend declares
`:prediction-shape' in `*provided-capabilities*' to say so, and nothing re-checks that
declaration: there is no argument to refuse, and what the declaration says is that the
mechanism is present, not that the shape is non-NIL. Measured against the vendored library,
the shape is RICHER than the first value's own dimensions for two kinds: a four-round
three-class model over four features reports (rows 4 3 1) for `:leaf-index' where the array is
rows x 12, and
(rows 3 5) for `:contrib' where the array is rows x 15. A four-round BINARY model over three
features reports (rows 4 1 1) and (rows 1 4) -- multidimensional there too, its one output
group notwithstanding -- so this is not a multiclass-only difference. The first value is
untouched by any of it.

`out_result' is XGBoost's own memory, valid only until the next call into this booster,
so every element is copied out, coerced from `single-float' to `double-float', before
this returns.

Deliberately does not scan the result for NaN or infinity. `with-foreign-float-traps-masked'
around this body stops SBCL from turning an intermediate invalid operation inside
XGBoost's own computation -- e.g. `multi:softprob''s softmax normalization -- into a signal;
it does not, and cannot, stop XGBoost from legitimately returning a non-finite value as a
final result (`:raw' scores in particular are not bounded the way a transformed prediction
is). Rejecting or flagging one here would be a policy this wrapper does not otherwise
impose on any other operation's output, invented for this fix rather than driven by a
reported failure -- a caller that cannot tolerate a non-finite prediction should check for
one itself."
  (with-foreign-float-traps-masked
    ;; BOOSTER-POINTER is bound first, and a `let' evaluates its init forms left to right, so
    ;; a wrong-kind or freed BOOSTER is refused by `%check-xgboost-booster' before
    ;; `%reject-best-num-iteration' can report a :BEST it would also have refused -- the same
    ;; precedence the `predict' method above this layer keeps between a bad handle and its own
    ;; argument checks.
    (let ((booster-pointer (%check-xgboost-booster booster "predict's booster argument"))
          (predict-type (%predict-type kind))
          (iteration-end
            (%resolve-num-iteration
             (%reject-best-num-iteration booster num-iteration "predict's :num-iteration"))))
      ;; Reading `out_shape'/`out_dim'/`out_result' back into the result array, and back out
      ;; as this function's two return values, is identical for both entry points and lives
      ;; here once; CALL is the only thing that differs between them, which is exactly how
      ;; much of this function a `csr-matrix' changes. Both entry points report the shape the
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
;;;
;;; `save-model' and `model-to-string' below repeat the same shape: a class guard on BOOSTER
;;; as the very first thing the body evaluates, then the foreign call -- no :BEST refusal
;;; here, since neither takes a NUM-ITERATION at all, unlike LightGBM's identical two, which
;;; do and carry that middle step. The repetition is load-bearing all the same:
;;; `tools/ci/check-layer-1-guards.lisp' requires that first form to be a `%CHECK-'-prefixed
;;; call on BOOSTER itself, and neither a shared macro nor a body-taking function wrapping it
;;; is a shape its walk recognises, so folding the duplication away would report both entry
;;; points unguarded rather than remove it.

(defun save-model (booster path)
  "Save BOOSTER's model to PATH via `XGBoosterSaveModel', and return PATH.

Takes no iteration limit, unlike `cl-gbdt/src/lightgbm/api''s `save-model':
`XGBoosterSaveModel' has no such parameter and always writes every boosted round. The
argument is therefore ABSENT here rather than refused -- a Layer 1 caller who names it gets
Common Lisp's own unknown-keyword error, which is the right report for a keyword that does
not exist. `cl-gbdt/src/xgboost/protocol''s method is where `unsupported-argument' is
signalled, because that refusal exists only because the unified API promised a portable
:NUM-ITERATION that LightGBM honours.

XGBoost selects its serialization format from PATH's extension -- `.json' and `.ubj' are the
current ones -- and reports an unrecognized extension itself, as `foreign-call-error'.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, a LightGBM booster, or not a handle at all. This function dispatches on nothing, so
`%check-xgboost-booster' is the only thing between such a handle's pointer and
`XGBoosterSaveModel'; see `%check-object-class' above on what wrong-kind pointers did when
measured against the vendored library. `released-handle-error' and `backend-not-open' come
from the `handle-live-pointer' inside that same check."
  (with-foreign-float-traps-masked
    (let ((pointer (%check-xgboost-booster booster "save-model's booster argument")))
      (cffi:with-foreign-string (filename (namestring path))
        (%save-model pointer filename)))
    path))

(defun load-model (backend path)
  "Load an XGBoost model from PATH and return a new booster built against BACKEND.

Unlike LightGBM's `LGBM_BoosterCreateFromModelfile', which allocates the booster and loads the
model in one call, XGBoost splits the two: `XGBoosterCreate' first builds a booster with no
DMatrix handles at all, and only then does `XGBoosterLoadModel' populate it from PATH.

The returned booster has no training set -- see the `booster' class' documentation -- since
PATH names a model, not a dataset. `evaluation' on it therefore reports nothing, and
`update-one-iteration' signals `missing-training-set'.

`with-pointer-ownership' spans exactly the window in which the raw booster is owned by
nobody: any exit that has not called TAKE-OWNERSHIP -- a failing `XGBoosterLoadModel' the
likeliest -- frees it here instead of orphaning it.

Signals `wrong-backend-reference' when BACKEND is not an `xgboost-backend', checked FIRST,
ahead of `%check-backend-open', which asks only whether the object is open and answers that
truthfully for the wrong library. Signals `backend-not-open' when BACKEND is closed."
  (with-foreign-float-traps-masked
    (%check-object-class backend 'xgboost-backend "backend" "load-model's backend argument")
    (%check-backend-open backend)
    (let ((booster-pointer (%create-booster nil)))
      (with-pointer-ownership (booster-pointer #'%free-booster-unchecked take-ownership)
        (cffi:with-foreign-string (filename (namestring path))
          (%load-model booster-pointer filename))
        (take-ownership 'xgboost-booster backend :booster)))))

(defun model-to-string (booster)
  "Return BOOSTER's model as a JSON string via `XGBoosterSaveModelToBuffer'.

Takes no iteration limit, for the reason `save-model' above states: that entry point's config
JSON has only a `\"format\"' key. The text this returns is a complete model document and can
be written to a `.json' file and handed back to `load-model'.

`out_dptr' is XGBoost's own memory, copied out via `foreign-string-to-lisp' with an explicit
`:count' from `out_len' rather than trusted to be null-terminated at the right place.

Signals `wrong-backend-reference', `released-handle-error' and `backend-not-open' exactly as
`save-model' does, and for the same reason: this function dispatches on nothing."
  (with-foreign-float-traps-masked
    (let ((pointer (%check-xgboost-booster booster "model-to-string's booster argument")))
      (cffi:with-foreign-string (config "{\"format\":\"json\"}")
        (cffi:with-foreign-objects ((out-len :uint64) (out-dptr :pointer))
          (%save-model-to-buffer pointer config out-len out-dptr)
          (cffi:foreign-string-to-lisp (cffi:mem-ref out-dptr :pointer)
                                        :count (cffi:mem-ref out-len :uint64)))))))

;;; ---------------------------------------------------------------------------
;;; Feature importance

(defun feature-importance (booster &key (kind :split))
  "Return BOOSTER's per-feature importances via `XGBoosterFeatureScore', as a fresh
`(simple-array double-float (*))' with one entry per feature, indexed by column -- zero for a
feature never used in a split.

`XGBoosterFeatureScore' itself reports the opposite: `out_n_features' and `out_scores' cover
only features that appear in at least one split, so a feature never split on is absent from
its report rather than present with a zero. Left as it comes back, the result's length would
be the number of USED features and its indices would not correspond to columns -- sparse where
LightGBM's equivalent is always dense. This builds a dense vector of `%booster-num-features'
entries instead and scatters each reported score into the column `%feature-score-index'
recovers from its feature name.

KIND is `:split' or `:gain', mapped by `%feature-importance-type' onto `\"weight\"' and
`\"total_gain\"' -- the latter deliberately, XGBoost's own `\"gain\"' being an average where
LightGBM's `:gain' and this project's contract mean the total.

Takes no :NUM-ITERATION, unlike `cl-gbdt/src/lightgbm/api''s `feature-importance':
`XGBoosterFeatureScore''s config JSON has no iteration-limiting key. The argument is absent
here rather than refused; `cl-gbdt/src/xgboost/protocol''s method is where
`unsupported-argument' is signalled, that refusal existing only because LightGBM honours the
argument.

Signals `unsupported-argument' instead of returning a result at all when
`XGBoosterFeatureScore' reports more than one score per feature -- see
`%check-feature-score-dim'. In practice that is a `gblinear' booster's `:split' importance on
a multi-class model, whose scores are a per-class matrix with no single value to derive from
it.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend, and
`released-handle-error' or `backend-not-open' from the `handle-live-pointer' inside that
check, which runs before anything else is read."
  (with-foreign-float-traps-masked
    (let ((pointer (%check-xgboost-booster booster "feature-importance's booster argument"))
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

(defun evaluation (booster)
  "Return BOOSTER's evaluation metrics as two values: a list of (DATASET-INDEX METRIC-NAME
VALUE) entries, and a plist stating where the values came from.

`XGBoosterEvalOneIter' evaluates whatever DMatrices it is handed and consults nothing the
booster was built with, so what this evaluates is BOOSTER's own retained handles: its training
set first, then each validation set in the order it was given. That is what makes
DATASET-INDEX mean here what it means on LightGBM, which can only evaluate what training
attached. A booster from `load-model' retains none, and is the case an empty result comes
from.

Each dataset is named to the C call by its own decimal index -- \"0\", \"1\" -- because the
call requires one name per DMatrix and builds each result label by joining that name to the
metric's with a hyphen; `%split-eval-label' takes the label back apart against those same
names, which is the only way to recover the metric name. Those names are an argument to a C
call, never a dataset name this API reports.

The second value is `(:value-source :parsed-text :raw TEXT)': VALUE is `%parse-eval-result''s
reading of formatted output, and :RAW carries that output unmodified so nothing the library
wrote is lost to the parse. A field the parser could not read as a `double-float' -- XGBoost
spells a non-finite one \"inf\" or \"nan\" -- keeps its entry with VALUE NIL rather than
disappearing.

The kind check runs first and every retained dataset is then read through
`handle-live-pointer', so a freed booster or a freed retained dataset signals
`released-handle-error' here; unlike `cl-gbdt/src/lightgbm/api''s `evaluation', this needs no
separate `%check-booster-datasets-live', every dataset it evaluates being one it resolves and
checks explicitly before any foreign call."
  (with-foreign-float-traps-masked
    (let* ((booster-pointer (%check-xgboost-booster booster "evaluation's booster argument"))
           (training-set (booster-training-set booster))
           (datasets (if training-set
                         (cons training-set (booster-validation-sets booster))
                         '()))
           (dataset-pointers (mapcar #'handle-live-pointer datasets)))
      (multiple-value-bind (entries raw) (%read-evaluation booster-pointer dataset-pointers)
        (values entries (list :value-source :parsed-text :raw raw))))))

;;; ---------------------------------------------------------------------------
;;; Dataset metadata

(defun dataset-num-rows (dataset)
  "Return DATASET's row count, read via `XGDMatrixNumRow'.

Signals `wrong-backend-reference' when DATASET is not an `xgboost-dataset' -- a booster, the
other backend's dataset, or not a handle at all. This function dispatches on nothing, so
`%check-object-class' is the only thing between a wrong-kind pointer and the foreign call.
Unlike the frees, which use that same check and then tolerate a released handle, this requires
a LIVE one and reads it through `handle-live-pointer' immediately after."
  (with-foreign-float-traps-masked
    (%check-object-class dataset 'xgboost-dataset "dataset"
                         "dataset-num-rows's dataset argument")
    (%dataset-num-rows (handle-live-pointer dataset))))

(defun dataset-num-features (dataset)
  "Return DATASET's feature count, read via `XGDMatrixNumCol'.

Signals what `dataset-num-rows' above signals, on the same terms and through the same two
calls."
  (with-foreign-float-traps-masked
    (%check-object-class dataset 'xgboost-dataset "dataset"
                         "dataset-num-features's dataset argument")
    (%dataset-num-features (handle-live-pointer dataset))))

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
