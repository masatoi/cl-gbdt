;;;; package.lisp --- Package for the functional tests.
;;;;
;;;; These tests call the real LightGBM and XGBoost shared libraries. They live in
;;;; their own system so that cl-gbdt/tests stays green with nothing installed.

(defpackage #:cl-gbdt/functional-tests
  (:use #:cl #:rove)
  ;; The generated FFI packages deliberately export nothing -- their own docstrings say
  ;; nothing outside the backend systems should call them directly. These tests are the
  ;; exception: their whole purpose is to exercise that raw layer. They therefore reach in
  ;; with a double colon, which keeps the trespass visible at every call site rather than
  ;; hiding it behind an export list the design does not want.
  (:local-nicknames (#:lgbm #:cl-gbdt.lightgbm.ffi)
                    (#:xgb #:cl-gbdt.xgboost.ffi))
  (:export #:backend-library-path
           #:ensure-backend-library
           #:with-backend-library
           #:make-separable-dataset
           #:predictions-separate-p
           #:array-interface-json))
