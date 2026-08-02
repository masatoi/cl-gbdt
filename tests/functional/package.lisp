;;;; package.lisp --- Package for the functional tests.
;;;;
;;;; These tests call the real LightGBM and XGBoost shared libraries. They live in
;;;; their own system so that cl-gbdt/tests stays green with nothing installed.

(defpackage #:cl-gbdt/functional-tests
  (:use #:cl #:rove)
  (:local-nicknames (#:lgbm #:cl-gbdt.lightgbm.ffi)
                    (#:xgb #:cl-gbdt.xgboost.ffi))
  (:export #:backend-library-path
           #:ensure-backend-library
           #:with-backend-library
           #:make-separable-dataset
           #:predictions-separate-p))
