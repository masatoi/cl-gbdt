(in-package #:cl-user)

(load (merge-pathnames "test-process.lisp" *load-truename*))

(let ((body (merge-pathnames "state-test-body.lisp" *load-truename*)))
  (if cl-gbdt-example-test-process:*child*
      (load body)
      (cl-gbdt-example-test-process:run-test-in-child
       body :systems '(:cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified))))
