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
  :depends-on ("cl-gbdt/src/lightgbm/backend"))

(defsystem "cl-gbdt/xgboost"
  :description "XGBoost backend for cl-gbdt"
  :license "MIT"
  :class :package-inferred-system
  :depends-on ("cl-gbdt/src/xgboost/backend"))

(defsystem "cl-gbdt/tests"
  :author "Satoshi Imai <satoshi.imai@gmail.com>"
  :license "MIT"
  :description "Test system for cl-gbdt"
  :class :package-inferred-system
  ;; Hand-maintained: package-inferred-system does not scan tests/ on its own, so a
  ;; new tests/*.lisp file needs a matching entry added here by hand. Forget one and
  ;; rove happily runs one suite fewer, all green -- a green run of a suite that
  ;; simply never existed, not a red one. tests/bindings.lisp's
  ;; test-suite-depends-on-lists-every-test-file asserts this list stays complete.
  :depends-on ("cl-gbdt/tests/conditions"
               "cl-gbdt/tests/data"
               "cl-gbdt/tests/regen"
               "cl-gbdt/tests/bindings"
               "cl-gbdt/tests/backend"
               "cl-gbdt/tests/handle"
               "cl-gbdt/tests/parameters"
               "cl-gbdt/tests/library"
               "cl-gbdt/tests/foreign")
  :perform (test-op (op c) (symbol-call :rove :run c)))

;;; Named for its path, like every other system here. That is not only for consistency:
;;; rove discovers a package-inferred-system's tests by walking the dependencies whose
;;; names have the system's own name as a literal string prefix
;;; (`rove/core/suite/file.lisp', `system-component-p'). The former name,
;;; `cl-gbdt/functional-tests', is not a prefix of `cl-gbdt/tests/functional/lightgbm',
;;; so rove found nothing and reported "0 tests completed" -- a green run of an empty
;;; suite.
(defsystem "cl-gbdt/tests/functional"
  :description "Tests that call the real LightGBM and XGBoost shared libraries. SBCL
only: the round trips pin arrays with sb-sys directly, unlike src/data.lisp's #+sbcl
guard, and have no portable fallback."
  :license "MIT"
  :class :package-inferred-system
  :depends-on ("cl-gbdt/tests/functional/lightgbm"
               "cl-gbdt/tests/functional/lightgbm-api"
               "cl-gbdt/tests/functional/xgboost"
               "cl-gbdt/tests/functional/xgboost-api")
  :perform (test-op (op c) (symbol-call :rove :run c)))
