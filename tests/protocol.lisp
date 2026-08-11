;;;; protocol.lisp --- Tests for the unified API's own behaviour, independent of any backend.
;;;;
;;;; `cl-gbdt:backend' is instantiable and has no unified-API methods of its own, which is
;;;; exactly the state a program reaches by loading cl-gbdt and cl-gbdt/lightgbm without
;;;; cl-gbdt/lightgbm/unified. So these need no shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/protocol
  (:use #:cl #:rove)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/protocol)

(defun bare-backend ()
  "Return a backend whose class has no unified-API methods.

Named `:lightgbm' so the report's advice can be checked verbatim: the condition builds the
system name out of the backend's own name."
  (make-instance 'cl-gbdt:backend :name :lightgbm))

(defun drop (handle)
  "Release HANDLE with a free function that does nothing, cancelling its finalizer."
  (cl-gbdt:release-handle handle (lambda (pointer) (declare (ignore pointer)) nil)))

(deftest backend-methods-not-loaded-is-a-backend-error
  (testing "it sits under backend-error, and so under gbdt-error"
    (ok (subtypep 'cl-gbdt:backend-methods-not-loaded 'cl-gbdt:backend-error)
        "a caller handling backend-error catches it")
    (ok (subtypep 'cl-gbdt:backend-methods-not-loaded 'cl-gbdt:gbdt-error)
        "and so does one handling gbdt-error")))

(deftest a-backend-dispatching-generic-says-which-system-to-load
  (testing "train names the backend, the generic function, and the system"
    (let ((caught (handler-case (progn (cl-gbdt:train (bare-backend) nil) nil)
                    (cl-gbdt:backend-methods-not-loaded (condition) condition))))
      (ok caught "the condition was signalled")
      (ok (eq :lightgbm (cl-gbdt:backend-error-backend caught)) "it names the backend")
      (ok (eq 'cl-gbdt:train
              (cl-gbdt:backend-methods-not-loaded-generic-function caught))
          "it names the generic function that was called")
      (ok (search "cl-gbdt/lightgbm/unified" (princ-to-string caught))
          "its report names the system to load"))))

(deftest make-dataset-and-load-model-signal-it-too
  (testing "every generic that dispatches on the backend is covered"
    (ok (handler-case (progn (cl-gbdt:make-dataset (bare-backend) #2A((1d0)) ) nil)
          (cl-gbdt:backend-methods-not-loaded () t))
        "make-dataset signals it")
    (ok (handler-case (progn (cl-gbdt:load-model (bare-backend) #p"/nonexistent") nil)
          (cl-gbdt:backend-methods-not-loaded () t))
        "load-model signals it")))

(deftest a-handle-dispatching-generic-names-the-handles-backend
  (testing "dataset-num-rows reaches the backend through the handle"
    (let* ((handle (cl-gbdt:make-handle 'cl-gbdt:dataset (cffi:make-pointer 1)
                                        (bare-backend) :dataset))
           (caught (handler-case (progn (cl-gbdt:dataset-num-rows handle) nil)
                     (cl-gbdt:backend-methods-not-loaded (condition) condition))))
      (ok caught "the condition was signalled")
      (ok (eq :lightgbm (cl-gbdt:backend-error-backend caught))
          "it names the backend the handle belongs to")
      (drop handle))))
