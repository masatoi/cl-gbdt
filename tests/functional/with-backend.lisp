;;;; with-backend.lisp --- `with-backend' closes its backend, on every exit path.
;;;;
;;;; A functional test rather than a layer 1 one, deliberately. `with-dataset' and
;;;; `with-booster' are testable without a shared library because `free-dataset' and
;;;; `free-booster' are generic functions, so `tests/backend.lisp' specialises them on a
;;;; `mock-handle'. `close-backend' is a `defun', not a generic, so the same trick would
;;;; mean reaching into `cl-gbdt/src/backend' with `::' to force the openness slot -- and
;;;; no test in this repository uses `::'. A real backend makes the observation directly:
;;;; `backend-open-p' answers before and after.

(uiop:define-package #:cl-gbdt/tests/functional/with-backend
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt)
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library))

(in-package #:cl-gbdt/tests/functional/with-backend)

(deftest with-backend-closes-on-normal-exit
  (with-backend-library (:xgboost)
    (testing "the backend is open inside the body and closed after it"
      (let ((captured nil))
        (cl-gbdt:with-backend (backend (cl-gbdt:open-backend :xgboost))
          (setf captured backend)
          (ok (cl-gbdt:backend-open-p backend) "open inside the body"))
        (ok (not (cl-gbdt:backend-open-p captured)) "closed after the body")))))

(deftest with-backend-closes-on-nonlocal-exit
  (with-backend-library (:xgboost)
    (testing "the backend is closed even when the body signals"
      (let ((captured nil))
        (ignore-errors
         (cl-gbdt:with-backend (backend (cl-gbdt:open-backend :xgboost))
           (setf captured backend)
           (error "boom")))
        (ok (not (cl-gbdt:backend-open-p captured)) "closed after a non-local exit")))))
