;;;; docgen.lisp --- Tests for the API-reference emitter.
;;;;
;;;; Layer 1: no shared library, and no backend system either. Every fixture here is defined in
;;;; this file, so the emitter is exercised on input this file controls rather than on the
;;;; library's own 174 published symbols, which tools/ci/check-api-reference.lisp covers whole.

(uiop:define-package #:cl-gbdt/tests/docgen
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/docgen/all))

(in-package #:cl-gbdt/tests/docgen)

(defpackage #:cl-gbdt/tests/docgen/alpha
  (:use #:cl)
  (:export #:shared #:only-alpha))

(defpackage #:cl-gbdt/tests/docgen/beta
  (:use #:cl)
  (:import-from #:cl-gbdt/tests/docgen/alpha #:shared)
  (:export #:shared #:only-beta))

(deftest published-symbols-qualifies-by-priority-order
  (testing "a symbol exported from two packages is qualified by the earlier one"
    (let* ((names '("cl-gbdt/tests/docgen/alpha" "cl-gbdt/tests/docgen/beta"))
           (found (cl-gbdt/src/docgen/all:published-symbols names))
           (shared (find "SHARED" found :key (lambda (p)
                                               (symbol-name
                                                (cl-gbdt/src/docgen/all:published-symbol p)))
                                        :test #'string=)))
      (ok (string= "cl-gbdt/tests/docgen/alpha"
                   (cl-gbdt/src/docgen/all:published-qualifier shared)))
      (ok (equal '("cl-gbdt/tests/docgen/alpha" "cl-gbdt/tests/docgen/beta")
                 (cl-gbdt/src/docgen/all:published-exported-from shared))))))

(deftest published-symbols-sorts-by-name-then-qualifier
  (testing "the order is total and does not depend on do-external-symbols"
    (let* ((names '("cl-gbdt/tests/docgen/alpha" "cl-gbdt/tests/docgen/beta"))
           (found (cl-gbdt/src/docgen/all:published-symbols names)))
      (ok (equal '("ONLY-ALPHA" "ONLY-BETA" "SHARED")
                 (mapcar (lambda (p)
                           (symbol-name (cl-gbdt/src/docgen/all:published-symbol p)))
                         found))))))

(deftest symbol-kind-names-every-kind-the-surface-holds
  (testing "each defining form maps to its own keyword"
    (ok (eq :function (cl-gbdt/src/docgen/all:symbol-kind 'fixture-function)))
    (ok (eq :generic-function (cl-gbdt/src/docgen/all:symbol-kind 'fixture-generic)))
    (ok (eq :macro (cl-gbdt/src/docgen/all:symbol-kind 'fixture-macro)))
    (ok (eq :class (cl-gbdt/src/docgen/all:symbol-kind 'fixture-class)))
    (ok (eq :condition (cl-gbdt/src/docgen/all:symbol-kind 'fixture-condition)))
    (ok (eq :structure (cl-gbdt/src/docgen/all:symbol-kind 'fixture-struct)))
    (ok (eq :variable (cl-gbdt/src/docgen/all:symbol-kind '*fixture-variable*)))))

(deftest symbol-kind-refuses-a-symbol-it-cannot-classify
  (testing "an unclassifiable published symbol is an error, not a silent blank entry"
    (ok (handler-case (progn (cl-gbdt/src/docgen/all:symbol-kind 'no-such-fixture-at-all) nil)
          (error () t)))))

(deftest check-introspection-primitives-names-what-is-missing
  (testing "a missing primitive is reported by name rather than failing later"
    (ok (cl-gbdt/src/docgen/all:check-introspection-primitives))
    (ok (handler-case
            (progn (cl-gbdt/src/docgen/all:check-introspection-primitives
                    '(("SB-PCL" "NO-SUCH-INTERNAL-ACCESSOR")))
                   nil)
          (error (e) (search "NO-SUCH-INTERNAL-ACCESSOR" (princ-to-string e)))))))

(defun fixture-function (a b) "Add A and B." (+ a b))

(defgeneric fixture-generic (object) (:documentation "Return something about OBJECT."))

(defmacro fixture-macro ((var form) &body body) "Bind VAR to FORM." `(let ((,var ,form)) ,@body))

(defclass fixture-class () () (:documentation "A class fixture."))

(define-condition fixture-condition (error) () (:documentation "A condition fixture."))

(defstruct fixture-struct "A structure fixture." (field nil))

(defparameter *fixture-variable* nil "A variable fixture.")
