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
;;;; `lightgbm-dataset', and `%reference-pointer' is handed that class name as a symbol.

(uiop:define-package #:cl-gbdt/src/lightgbm/api
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/lightgbm/classes
                #:lightgbm-dataset)
  (:import-from #:cl-gbdt/src/lightgbm/native
                #:%check-backend-open
                #:%create-dataset
                #:%create-dataset-from-csr
                #:%free-dataset
                #:%free-dataset-unchecked
                #:%parameter-string
                #:%reference-pointer
                #:%set-feature-names
                #:%set-group-field
                #:%set-info-field)
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
                #:handle-released-p
                #:release-handle
                #:with-pointer-ownership)
  (:export #:%check-sparse-input
           #:create-dataset
           #:free-dataset))

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
