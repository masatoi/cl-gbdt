(defsystem "cl-gbdt"
  :version "0.0.1"
  :author "Satoshi Imai <satoshi.imai@gmail.com>"
  :license "MIT"
  :description "Common Lisp wrapper exposing LightGBM and XGBoost through one API"
  :class :package-inferred-system
  :depends-on ("cl-gbdt/src/all")
  :in-order-to ((test-op (test-op "cl-gbdt/tests"))))

(defsystem "cl-gbdt/regen"
  :description "Binding emitter. Development only; never part of a build."
  :license "MIT"
  :class :package-inferred-system
  :depends-on ("cl-gbdt/src/regen/all"))

(defsystem "cl-gbdt/lightgbm"
  :description "LightGBM backend for cl-gbdt"
  :license "MIT"
  :class :package-inferred-system
  :depends-on ("cl-gbdt/src/lightgbm/c-api"))

(defsystem "cl-gbdt/xgboost"
  :description "XGBoost backend for cl-gbdt"
  :license "MIT"
  :class :package-inferred-system
  :depends-on ("cl-gbdt/src/xgboost/c-api"))

(defsystem "cl-gbdt/tests"
  :author "Satoshi Imai <satoshi.imai@gmail.com>"
  :license "MIT"
  :description "Test system for cl-gbdt"
  :class :package-inferred-system
  :depends-on ("cl-gbdt/tests/conditions"
               "cl-gbdt/tests/data"
               "cl-gbdt/tests/regen"
               "cl-gbdt/tests/bindings"
               "cl-gbdt/tests/backend")
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-gbdt/functional-tests"
  :description "Tests that call the real LightGBM and XGBoost shared libraries. SBCL
only: the round trips pin arrays with sb-sys directly, unlike src/data.lisp's #+sbcl
guard, and have no portable fallback."
  :license "MIT"
  :depends-on ("cl-gbdt" "cl-gbdt/lightgbm" "cl-gbdt/xgboost" "rove")
  :components ((:module "tests/functional"
                :serial t
                :components ((:file "support")
                             (:file "lightgbm")
                             (:file "xgboost"))))
  :perform (test-op (op c) (symbol-call :rove :run c)))
