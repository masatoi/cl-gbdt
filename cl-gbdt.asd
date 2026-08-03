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
  :depends-on ("cl-gbdt" "cffi")
  :serial t
  :components ((:module "src/lightgbm"
                :components ((:file "package")
                             (:file "c-api")))))

(defsystem "cl-gbdt/xgboost"
  :description "XGBoost backend for cl-gbdt"
  :license "MIT"
  :depends-on ("cl-gbdt" "cffi")
  :serial t
  :components ((:module "src/xgboost"
                :components ((:file "package")
                             (:file "c-api")))))

(defsystem "cl-gbdt/tests"
  :author "Satoshi Imai <satoshi.imai@gmail.com>"
  :license "MIT"
  :depends-on ("cl-gbdt"
               "cl-gbdt/regen"
               "cl-gbdt/lightgbm"
               "cl-gbdt/xgboost"
               "rove")
  :components ((:module "tests"
                :serial t
                :components
                ((:file "package")
                 (:file "conditions")
                 (:file "data")
                 (:file "regen")
                 (:file "bindings")
                 (:file "backend"))))
  :description "Test system for cl-gbdt"
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-gbdt/functional-tests"
  :description "Tests that call the real LightGBM and XGBoost shared libraries. SBCL
only: the round trips pin arrays with sb-sys directly, unlike src/data.lisp's #+sbcl
guard, and have no portable fallback."
  :license "MIT"
  :depends-on ("cl-gbdt" "cl-gbdt/lightgbm" "cl-gbdt/xgboost" "rove")
  :components ((:module "tests/functional"
                :serial t
                :components ((:file "package")
                             (:file "support")
                             (:file "lightgbm")
                             (:file "xgboost"))))
  :perform (test-op (op c) (symbol-call :rove :run c)))
