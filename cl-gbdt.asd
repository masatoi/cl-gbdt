(defsystem "cl-gbdt"
  :version "0.0.1"
  :author ""
  :license ""
  :description "Common Lisp wrapper exposing LightGBM and XGBoost through one API"
  :depends-on ("cffi" "alexandria" "trivial-garbage")
  :serial t
  :components ((:module "src"
                :components ((:file "package")
                             (:file "conditions")
                             (:file "data")
                             (:file "backend")
                             (:file "protocol"))))
  :in-order-to ((test-op (test-op "cl-gbdt/tests"))))

(defsystem "cl-gbdt/regen"
  :description "Binding emitter. Development only; never part of a build."
  :depends-on ("cffi/c2ffi" "com.inuoe.jzon" "alexandria")
  :serial t
  :components ((:module "src/regen"
                :components ((:file "package")
                             (:file "types")
                             (:file "emit")))))

(defsystem "cl-gbdt/lightgbm"
  :description "LightGBM backend for cl-gbdt"
  :depends-on ("cl-gbdt" "cffi")
  :serial t
  :components ((:module "src/lightgbm"
                :components ((:file "package")
                             (:file "c-api")))))

(defsystem "cl-gbdt/xgboost"
  :description "XGBoost backend for cl-gbdt"
  :depends-on ("cl-gbdt" "cffi")
  :serial t
  :components ((:module "src/xgboost"
                :components ((:file "package")
                             (:file "c-api")))))

(defsystem "cl-gbdt/tests"
  :author ""
  :license ""
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
