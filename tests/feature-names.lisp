;;;; feature-names.lisp --- Tests for validating a feature-name list.
;;;;
;;;; Backend-independent and pure, so these need no shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/feature-names
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/config/feature-names
                #:check-feature-names)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/feature-names)

(defun %check (names)
  (check-feature-names names :xgboost))

(deftest check-feature-names-passes-a-proper-list-through
  ;; The checker returns its argument so a caller can wrap the call site without
  ;; restructuring it, and NIL -- no names -- is as valid as a list of them.
  (testing "a list of strings comes back unchanged"
    (ok (equal '("a" "b") (%check '("a" "b"))) "what a proper list returns"))
  (testing "NIL comes back unchanged"
    (ok (null (%check nil)) "what NIL returns")))

(deftest check-feature-names-rejects-an-improper-list
  ;; `listp' is true for a dotted list, which is why the obvious guard is not enough:
  ;; the traversal below it then fails with a raw type-error instead of the typed
  ;; condition make-dataset's docstring promises.
  (testing "a dotted pair signals unsupported-argument"
    (ok (handler-case (progn (%check '("a" . "b")) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether a dotted pair was rejected"))
  (testing "a list that ends dotted signals unsupported-argument"
    (ok (handler-case (progn (%check '("a" "b" . "c")) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether a list ending dotted was rejected"))
  (testing "a non-list signals unsupported-argument"
    (ok (handler-case (progn (%check "a") nil)
          (cl-gbdt:unsupported-argument () t))
        "whether a bare string was rejected")))

(deftest check-feature-names-rejects-a-circular-list-without-hanging
  ;; A circular list makes an unguarded `loop :for ... :in' run forever. The timeout is
  ;; what turns a regression into a failing test rather than a CI job that hangs until it
  ;; is killed -- so a future edit that drops the check fails here in seconds.
  (testing "a circular list signals rather than looping"
    (ok (handler-case
            (sb-ext:with-timeout 5
              (let ((circular (list "a")))
                (setf (cdr circular) circular)
                (%check circular)
                nil))
          (cl-gbdt:unsupported-argument () t)
          (sb-ext:timeout () nil))
        "whether a circular list was rejected within five seconds")))

(deftest check-feature-names-names-the-backend-it-was-given
  (testing "the condition carries the backend name"
    (ok (eq :lightgbm
            (handler-case (progn (check-feature-names '("a" . "b") :lightgbm) nil)
              (cl-gbdt:unsupported-argument (c)
                (cl-gbdt:unsupported-argument-backend c))))
        "the backend named by the condition")))
