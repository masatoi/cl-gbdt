;;;; handle.lisp --- CLOS wrappers around foreign dataset and booster pointers.
;;;;
;;;; The protocol dispatches on datasets and boosters -- `dataset-num-rows', `predict',
;;;; `free-dataset' -- so they must be objects, not raw pointers. Free-once,
;;;; use-after-free detection and the leak report are identical for both backends and
;;;; both object kinds, so they live here rather than being written twice.

(uiop:define-package #:cl-gbdt/src/handle
  (:use #:cl)
  (:import-from #:trivial-garbage #:finalize)
  (:import-from #:cl-gbdt/src/backend
                #:backend-open-p
                #:backend-name)
  (:import-from #:cl-gbdt/src/conditions
                #:released-handle-error
                #:backend-not-open
                #:unfreed-handle-warning
                #:wrong-backend-reference)
  (:export #:handle #:dataset #:booster
           #:handle-pointer #:handle-backend #:booster-training-set
           #:booster-validation-sets #:booster-best-iteration
           #:handle-released-p #:handle-live-pointer
           #:make-handle #:release-handle))

(in-package #:cl-gbdt/src/handle)

(defclass handle ()
  ((pointer :initarg :pointer
            :reader handle-pointer
            :documentation "CFFI pointer to the underlying foreign object.")
   (released :initarg :released
             :reader %handle-released-cell
             :documentation "A one-element list shared with this handle's finalizer
closure. Its CAR is true once the handle has been released.

This cannot be a plain slot: a finalizer that closed over the handle itself, to read
a slot through it, would keep the handle reachable forever and therefore never run.
The cons cell is reachable from both the handle and the closure without either one
keeping the other alive.")
   (backend :initarg :backend
            :reader handle-backend
            :documentation "The `backend' instance that owns this handle's foreign
pointer. Read through this, never assumed still open, so `handle-live-pointer' can
tell whether that backend has since been closed -- a pointer into a backend whose
shared library `close-backend' has unmapped is exactly as dangerous as one already
freed."))
  (:documentation "A CLOS wrapper around a foreign dataset or booster pointer.

Build instances with `make-handle', never directly."))

(defclass dataset (handle) ()
  (:documentation "A backend's training or validation dataset."))

(defclass booster (handle)
  ((training-set :initarg :training-set
                 :initform nil
                 :reader booster-training-set
                 :documentation "The dataset this booster was trained on, or NIL for a
`load-model' booster, which has none.")
   (validation-sets :initarg :validation-sets
                     :initform nil
                     :reader booster-validation-sets
                     :documentation "The validation datasets passed to `train' via
`:valid-sets', or NIL when none were given. `LGBM_BoosterAddValidData' stores each
one's pointer inside the booster exactly as `LGBM_BoosterCreate' does for the
training set, so these are retained strongly for the same liveness-checking reason
as TRAINING-SET: freeing one out from under a live booster is the same hazard.")
   (best-iteration :initarg :best-iteration
                    :initform nil
                    :reader booster-best-iteration
                    :documentation "The iteration an early-stopped `train' run judged
best, or NIL.

Set only when `train' was given `:early-stopping', and only when its watcher actually
determined a best iteration -- NIL for a `load-model' booster, for one trained without
`:early-stopping', and for one trained with it whose watcher never got the chance to see
one (see `train''s docstring for the two ways that happens even with `:early-stopping'
supplied). `predict', `save-model' and `model-to-string''s `:num-iteration :best' resolve
against this slot, signalling `unsupported-argument' rather than assuming a default when
it is NIL."))
  (:documentation "A trained, or in-progress, backend model.

Retains its training set strongly, which is what makes `with-booster''s docstring --
that nesting it inside `with-dataset' cannot invert the release order -- true."))

(defun %make-finalizer (released kind)
  "Return a closure that warns `unfreed-handle-warning' (naming KIND) unless RELEASED's
CAR is true.

`make-handle' attaches the returned closure to a handle via `finalize'. The closure is
built here, separately from that call, so a test can invoke it directly with a RELEASED
cell it controls instead of depending on garbage collection timing to provoke it."
  (lambda ()
    (unless (car released)
      (warn 'unfreed-handle-warning :kind kind))))

(defun make-handle (class-name pointer backend kind
                     &key training-set validation-sets best-iteration)
  "Create an instance of CLASS-NAME wrapping POINTER for BACKEND, with a finalizer
attached that warns `unfreed-handle-warning' (naming KIND) if the instance is
garbage-collected before `release-handle' runs on it.

TRAINING-SET, VALIDATION-SETS and BEST-ITERATION, when supplied, become the new
instance's `training-set', `validation-sets' and `best-iteration' initargs; only
`booster' has slots for them. Free the result with `release-handle'."
  (let* ((released (list nil))
         (handle (apply #'make-instance class-name
                         :pointer pointer :released released :backend backend
                         (append (when training-set (list :training-set training-set))
                                 (when validation-sets
                                   (list :validation-sets validation-sets))
                                 (when best-iteration
                                   (list :best-iteration best-iteration))))))
    (finalize handle (%make-finalizer released kind))
    handle))

(defun handle-released-p (handle)
  "Return true when HANDLE has already been released."
  (and (car (%handle-released-cell handle)) t))

(defun handle-live-pointer (handle)
  "Return HANDLE's foreign pointer, or signal `released-handle-error' when HANDLE has
already been released, or `backend-not-open' when HANDLE's backend has since been
closed.

Every operation that reaches into a backend's C API must read the pointer through this
function, never `handle-pointer' directly: passing an already-freed pointer back to C
is a segfault, which is not a catchable Lisp condition -- it kills the process
outright, skipping `unwind-protect' and leaking everything else. A pointer into a
backend whose shared library `close-backend' has since closed is the same hazard --
nothing clears it -- so that is checked here too, turning it into a catchable
condition instead of a possible crash."
  (cond
    ((handle-released-p handle) (error 'released-handle-error :object handle))
    ((not (backend-open-p (handle-backend handle)))
     (error 'backend-not-open :backend (backend-name (handle-backend handle))))
    (t (handle-pointer handle))))

(defun release-handle (handle free-function)
  "Call FREE-FUNCTION on HANDLE's pointer exactly once, then mark HANDLE released.

FREE-FUNCTION is a function of one argument, HANDLE's pointer. A second call on an
already-released HANDLE does nothing -- this is what lets `free-dataset' and
`free-booster' promise that freeing an already-freed handle is a no-op. Also cancels
HANDLE's finalizer, since that finalizer exists only to report a free that never
happened, and this one just did."
  (let ((released (%handle-released-cell handle)))
    (unless (car released)
      (funcall free-function (handle-pointer handle))
      (setf (car released) t)
      (trivial-garbage:cancel-finalization handle)))
  (values))

(defun %check-handle-kind (object kind backend-keyword argument-description)
  "Return OBJECT's live foreign pointer, after confirming OBJECT is a KIND -- the symbol
`dataset' or `booster' -- built by the backend named BACKEND-KEYWORD.
ARGUMENT-DESCRIPTION names which caller-supplied argument OBJECT came from, for
`wrong-backend-reference''s report.

This is the check a backend needs where CLOS cannot make it: a `defmethod' specialized on
`lightgbm-dataset' has already ruled out every other handle before its body runs, but a
function taking a caller-supplied handle as a plain argument -- `cl-gbdt/lightgbm''s
`booster-eval', `cl-gbdt/xgboost''s `evaluate-one-iteration' -- has not.
`handle-live-pointer' alone is not enough there: it returns any handle's pointer regardless
of which library allocated it, so a LightGBM `DatasetHandle' would reach
`XGBoosterEvalOneIter' as a `DMatrixHandle' and be dereferenced as one. Both are
`cffi:foreign-pointer' at the call, so nothing downstream can tell them apart, and the
result is a memory fault rather than a condition -- reproduced deliberately, by removing
this check, in `cl-gbdt/tests/functional/evaluation''s
`evaluation-layer-1-entry-points-reject-the-other-backends-handles'.

KIND is the backend-agnostic base class from this file, not either backend's concrete
subclass, because neither backend's `native.lisp' may depend on its own `protocol.lisp'
where those subclasses live -- see policy section 11. Together with BACKEND-KEYWORD it
identifies exactly the same set of objects the concrete class would: only a
`lightgbm-backend''s operations build a handle whose `handle-backend' has `backend-name'
`:lightgbm', and that name is set once, at `open-backend'.

Deliberately compares the backend's NAME rather than its identity, so two backend
instances over the same shared library interoperate -- both read through the same C API,
and the dangerous case, a handle whose own backend has been closed, is caught by
`handle-live-pointer' below rather than here.

The report's noun comes from KIND itself rather than a separate argument, so the two
cannot disagree: a caller asking for a `booster' and getting the report \"must be a
dataset\" would be the defect this exists to describe.

Signals `wrong-backend-reference' when OBJECT is not a KIND at all -- the other kind of
handle, or any non-handle value -- or is one built by a different backend, and whatever
`handle-live-pointer' signals otherwise: `released-handle-error' for an already-freed
OBJECT, `backend-not-open' when its own backend has since been closed."
  (unless (and (typep object kind)
               (eq (backend-name (handle-backend object)) backend-keyword))
    (error 'wrong-backend-reference
           :backend backend-keyword
           :given (class-name (class-of object))
           :argument argument-description
           :expected (string-downcase (symbol-name kind))))
  (handle-live-pointer object))
