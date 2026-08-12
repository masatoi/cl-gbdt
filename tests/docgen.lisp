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

(define-condition fixture-documented-condition (error)
  ((code :initarg :code
         :reader fixture-documented-condition-code
         :documentation "The status the foreign call returned."))
  (:documentation "A condition whose slot carries its own text."))

(defclass fixture-documented-class ()
  ((tag :initarg :tag
        :reader fixture-documented-class-tag
        :documentation "What this object is called."))
  (:documentation "A class whose slot carries its own text."))

(deftest slot-documentation-reaches-condition-slots
  (testing "documentation cannot read a condition slot's text in SBCL; this can"
    (let ((slots (cl-gbdt/src/docgen/all:type-slots 'fixture-documented-condition)))
      (ok (= 1 (length slots)))
      (ok (string= "The status the foreign call returned."
                   (cl-gbdt/src/docgen/all:slot-info-documentation (first slots))))
      (ok (equal '(fixture-documented-condition-code)
                 (cl-gbdt/src/docgen/all:slot-info-readers (first slots)))))))

(deftest slot-documentation-reaches-standard-class-slots
  (testing "the same accessor serves defclass slots, so there is one mechanism not two"
    (let ((slots (cl-gbdt/src/docgen/all:type-slots 'fixture-documented-class)))
      (ok (string= "What this object is called."
                   (cl-gbdt/src/docgen/all:slot-info-documentation (first slots)))))))

(deftest symbol-documentation-reads-one-doc-type-per-kind
  (testing "a type answers under both 'type and 'structure; only one is read"
    (ok (string= "A structure fixture."
                 (cl-gbdt/src/docgen/all:symbol-documentation 'fixture-struct :structure)))
    (ok (string= "Add A and B."
                 (cl-gbdt/src/docgen/all:symbol-documentation 'fixture-function :function)))
    (ok (string= "A variable fixture."
                 (cl-gbdt/src/docgen/all:symbol-documentation '*fixture-variable* :variable)))))

(deftest reader-index-maps-structure-accessors-and-constructors
  (testing "a defstruct's accessors and constructor are found structurally, not by name"
    (let* ((published (cl-gbdt/src/docgen/all:published-symbols
                       '("cl-gbdt/tests/docgen/gamma")))
           (index (cl-gbdt/src/docgen/all:reader-index published)))
      ;; FIXTURE-INDEXED-STRUCT and its accessor/constructor are exported by GAMMA, but a bare
      ;; slot name like FIELD is not published by anything -- it is data hung off a published
      ;; type, not itself part of a package's public surface -- so it stays unexported there and
      ;; is reached here with `::' rather than `:'. Unqualified symbols in this LET* would read
      ;; into CL-GBDT/TESTS/DOCGEN, a different symbol from GAMMA's own, so GETHASH would look up
      ;; the wrong key entirely.
      (ok (equal '(:slot cl-gbdt/tests/docgen/gamma:fixture-indexed-struct
                   cl-gbdt/tests/docgen/gamma::field)
                 (gethash 'cl-gbdt/tests/docgen/gamma:fixture-indexed-struct-field index)))
      (ok (equal '(:constructor cl-gbdt/tests/docgen/gamma:fixture-indexed-struct)
                 (gethash 'cl-gbdt/tests/docgen/gamma:make-fixture-indexed-struct index))))))

(deftest reader-index-maps-condition-readers-to-their-slot
  (testing "a published reader points at the type and slot it reads"
    (let* ((published (cl-gbdt/src/docgen/all:published-symbols
                       '("cl-gbdt/tests/docgen/gamma")))
           (index (cl-gbdt/src/docgen/all:reader-index published)))
      ;; See the previous test for why GAMMA's symbols need explicit qualification here.
      (ok (equal '(:slot cl-gbdt/tests/docgen/gamma:fixture-indexed-condition
                   cl-gbdt/tests/docgen/gamma::code)
                 (gethash 'cl-gbdt/tests/docgen/gamma:fixture-indexed-condition-code
                          index))))))

(defpackage #:cl-gbdt/tests/docgen/gamma
  (:use #:cl)
  (:export #:fixture-indexed-struct
           #:fixture-indexed-struct-field
           #:make-fixture-indexed-struct
           #:fixture-indexed-condition
           #:fixture-indexed-condition-code))

(in-package #:cl-gbdt/tests/docgen/gamma)

(defstruct fixture-indexed-struct "An indexed structure fixture." (field nil))

(define-condition fixture-indexed-condition (error)
  ((code :initarg :code :reader fixture-indexed-condition-code
         :documentation "A code."))
  (:documentation "An indexed condition fixture."))

(in-package #:cl-gbdt/tests/docgen)
