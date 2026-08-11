;;;; api.lisp --- LightGBM's Layer 1 operations, over the handle classes.
;;;;
;;;; `native.lisp' holds the `%'-functions: each takes and returns raw pointers, and none of them
;;;; is a finished operation. `classes.lisp' holds the CLOS types and the shared library's
;;;; lifetime. This file is what a caller of `cl-gbdt/lightgbm' actually invokes -- operations
;;;; that take a backend or a handle, do the whole job, and hand back a handle or a result.
;;;;
;;;; Every function here reaches the shared library, so every one wraps its whole body in
;;;; `with-foreign-float-traps-masked' -- see `protocol.lisp''s header for why, and
;;;; `tools/ci/check-float-traps.lisp', which scans this file and holds every name the sibling
;;;; `all.lisp' exports to that rule.
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
                #:%check-backend-open
                #:%check-booster-datasets-live
                #:%check-lightgbm-dataset
                #:%create-booster
                #:%create-dataset
                #:%create-dataset-from-csr
                #:%free-booster
                #:%free-booster-unchecked
                #:%free-dataset
                #:%free-dataset-unchecked
                #:%parameter-string
                #:%reference-pointer
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
  (:import-from #:cl-gbdt/src/data
                #:csr-matrix
                #:csr-matrix-indices
                #:csr-matrix-indptr
                #:csr-matrix-num-columns
                #:csr-matrix-values)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:import-from #:cl-gbdt/src/handle
                #:handle-backend
                #:handle-live-pointer
                #:handle-released-p
                #:release-handle
                #:with-pointer-ownership)
  (:export #:%check-sparse-input
           #:create-booster
           #:create-dataset
           #:free-booster
           #:free-dataset
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
`cl-gbdt/src/lightgbm/protocol''s `predict', which imports it from here and is the reason this
otherwise-internal name is exported at all -- so the two cannot come to disagree about which
capability they name or which backend they blame.

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

Unlike every other operation that reads an existing handle, this does not go through
`handle-live-pointer' and so does not signal `backend-not-open' when DATASET's backend has
already been closed. `free-dataset' runs from `with-dataset''s `unwind-protect' cleanup form,
and a non-local exit is exactly when that cleanup runs; signalling there would replace
whatever condition is already unwinding the stack instead of letting it propagate. So when
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
    ;; `let', not `let*': no binding here reads another, and the checks still run before the
    ;; creation call below because a `let' evaluates its init forms left to right, which is
    ;; the whole ordering this function needs. `%create-booster' is deliberately NOT among
    ;; them -- it belongs inside its own form, where the raw handle's lifetime begins.
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

Returns false once an iteration produces no further split -- LightGBM's own
`produced_empty_tree' out parameter, read by `%update-one-iteration' and inverted here, so
that a true return means the iteration did produce one.

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
memory leaked instead -- the same cleanup-form reasoning applies here, `with-booster' being
the macro whose `unwind-protect' calls this one."
  (with-foreign-float-traps-masked
    (if (backend-open-p (handle-backend booster))
        (release-handle booster (lambda (pointer) (%free-booster pointer)))
        (let ((already-released (handle-released-p booster)))
          (release-handle booster (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing a LightGBM booster after its backend was closed: the foreign ~
                   booster was not freed and its memory is leaked."))))))
