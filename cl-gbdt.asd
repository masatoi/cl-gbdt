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
  :depends-on ("cl-gbdt/src/lightgbm/all"))

(defsystem "cl-gbdt/xgboost"
  :description "XGBoost backend for cl-gbdt"
  :license "MIT"
  :class :package-inferred-system
  :depends-on ("cl-gbdt/src/xgboost/all"))

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
               "cl-gbdt/tests/foreign"
               "cl-gbdt/tests/version"
               "cl-gbdt/tests/training-report"
               "cl-gbdt/tests/training-history"
               "cl-gbdt/tests/training-early-stopping"
               "cl-gbdt/tests/missing-value"
               "cl-gbdt/tests/categorical-features"
               "cl-gbdt/tests/feature-names"
               "cl-gbdt/tests/prediction-shape"
               "cl-gbdt/tests/objective"
               "cl-gbdt/tests/custom-metric")
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
               "cl-gbdt/tests/functional/xgboost-api"
               ;; Backend-neutral: the same assertions against both backends, per policy
               ;; section 13's portable-contract split. See its own header for why it is
               ;; not part of either backend's file.
               "cl-gbdt/tests/functional/evaluation"
               ;; Backend-neutral too, and built on the file above: it imports that file's
               ;; *FIXTURES* rather than restating them. `train''s secondary value, the
               ;; training report -- policy section 9, Phase 3a.
               "cl-gbdt/tests/functional/training-report"
               ;; The same *FIXTURES* again, one phase further on: `train''s
               ;; :EARLY-STOPPING, which shortens the run itself rather than only
               ;; describing it -- policy section 9, Phase 3b.
               "cl-gbdt/tests/functional/early-stopping"
               ;; The same *FIXTURES* once more: `make-dataset' ingesting a
               ;; `cl-gbdt:csr-matrix' on both backends, and the `:sparse-input'
               ;; capability that gates it.
               "cl-gbdt/tests/functional/sparse-input"
               ;; And once more again: `make-dataset''s :MISSING, the value that means
               ;; missing, and the `:missing-value' capability that gates it -- true on
               ;; XGBoost, false on LightGBM, which signals instead.
               "cl-gbdt/tests/functional/missing-value"
               ;; The same *FIXTURES* once more: `make-dataset''s
               ;; :CATEGORICAL-FEATURES, which columns hold categories rather than
               ;; quantities, and the `:categorical-features' capability that gates it.
               "cl-gbdt/tests/functional/categorical-features"
               ;; The same *FIXTURES* one last time, and the only entry here about
               ;; `predict''s SECOND return value: the shape the backend reports for the
               ;; result it just wrote, and the `:prediction-shape' capability, which no
               ;; operation refuses on -- a false answer means a NIL second value rather
               ;; than a signal.
               "cl-gbdt/tests/functional/prediction-shape"
               ;; The one entry here that does NOT build on those *FIXTURES*: a custom
               ;; objective's claim is that it reproduces a built-in one exactly, so it
               ;; states a deterministic fixture of its own. `train''s :OBJECTIVE, and the
               ;; `:custom-objective' capability that gates it.
               "cl-gbdt/tests/functional/custom-objective")
  :perform (test-op (op c) (symbol-call :rove :run c)))
