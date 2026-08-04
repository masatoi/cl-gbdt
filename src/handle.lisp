;;;; handle.lisp --- CLOS wrappers around foreign dataset and booster pointers.
;;;;
;;;; The protocol dispatches on datasets and boosters -- `dataset-num-rows', `predict',
;;;; `free-dataset' -- so they must be objects, not raw pointers. Free-once,
;;;; use-after-free detection and the leak report are identical for both backends and
;;;; both object kinds, so they live here rather than being written twice.

(uiop:define-package #:cl-gbdt/src/handle
  (:use #:cl)
  (:import-from #:trivial-garbage #:finalize)
  (:import-from #:cl-gbdt/src/conditions
                #:released-handle-error
                #:unfreed-handle-warning)
  (:export #:handle #:dataset #:booster
           #:handle-pointer #:handle-backend #:booster-training-set
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
            :documentation "Backend name, either `:lightgbm' or `:xgboost'."))
  (:documentation "A CLOS wrapper around a foreign dataset or booster pointer.

Build instances with `make-handle', never directly."))

(defclass dataset (handle) ()
  (:documentation "A backend's training or validation dataset."))

(defclass booster (handle)
  ((training-set :initarg :training-set
                 :initform nil
                 :reader booster-training-set
                 :documentation "The dataset this booster was trained on, or NIL for a
`load-model' booster, which has none."))
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

(defun make-handle (class-name pointer backend kind &key training-set)
  "Create an instance of CLASS-NAME wrapping POINTER for BACKEND, with a finalizer
attached that warns `unfreed-handle-warning' (naming KIND) if the instance is
garbage-collected before `release-handle' runs on it.

TRAINING-SET, when supplied, becomes the new instance's `training-set' initarg; only
`booster' has a slot for it. Free the result with `release-handle'."
  (let* ((released (list nil))
         (handle (apply #'make-instance class-name
                         :pointer pointer :released released :backend backend
                         (when training-set (list :training-set training-set)))))
    (finalize handle (%make-finalizer released kind))
    handle))

(defun handle-released-p (handle)
  "Return true when HANDLE has already been released."
  (and (car (%handle-released-cell handle)) t))

(defun handle-live-pointer (handle)
  "Return HANDLE's foreign pointer, or signal `released-handle-error' when HANDLE has
already been released.

Every operation that reaches into a backend's C API must read the pointer through this
function, never `handle-pointer' directly: passing an already-freed pointer back to C
is a segfault, which is not a catchable Lisp condition -- it kills the process
outright, skipping `unwind-protect' and leaking everything else."
  (if (handle-released-p handle)
      (error 'released-handle-error :object handle)
      (handle-pointer handle)))

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
