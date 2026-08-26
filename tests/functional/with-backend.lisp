;;;; with-backend.lisp --- `with-backend' closes its backend, on every exit path.
;;;;
;;;; A functional test rather than a layer 1 one, deliberately. `with-dataset' and
;;;; `with-booster' are testable without a shared library because `free-dataset' and
;;;; `free-booster' are generic functions, so `tests/backend.lisp' specialises them on a
;;;; `mock-handle'. `close-backend' is a `defun', not a generic, so the same trick would
;;;; mean reaching into `cl-gbdt/src/backend' with `::' to force the openness slot. A real
;;;; backend makes the observation directly:
;;;; `backend-open-p' answers before and after.

(uiop:define-package #:cl-gbdt/tests/functional/with-backend
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt)
  ;; Zero symbols: this file's two tests both open :xgboost -- the LightGBM one moved to
  ;; tests/backend.lisp already -- through `cl-gbdt:open-backend' and `cl-gbdt:with-backend',
  ;; the portable API, not either backend's own package. This clause is what runs
  ;; `register-backend' at load time to make :xgboost known to `open-backend' -- see
  ;; `register-backend' in xgboost/classes.lisp. Without it, package-inferred-system has no
  ;; edge to that file at all and `(cl-gbdt:open-backend :xgboost)' below would signal
  ;; `unknown-backend'. `unified' rather than `all' since the Layer 1 split: `all' is Layer 1
  ;; alone now, and this file exercises the portable unified surface rather than either
  ;; backend's own Layer 1 package, matching the identical clause in evaluation.lisp. No
  ;; LightGBM clause: this file no longer has a LightGBM test to need one for.
  (:import-from #:cl-gbdt/src/xgboost/unified)
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
