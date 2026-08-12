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

(deftest render-lambda-list-prints-names-not-packages
  (testing "internal symbols print as bare lower-case names"
    (ok (string= "(booster matrix &key kind)"
                 (cl-gbdt/src/docgen/all:render-lambda-list
                  (list 'cl-gbdt/tests/docgen::booster 'cl-gbdt/tests/docgen::matrix
                        '&key 'cl-gbdt/tests/docgen::kind))))))

(deftest render-lambda-list-keeps-keywords-colons
  (testing "a keyword default keeps its colon; downcasing symbol-name alone would lose it"
    (ok (string= "(&key (kind :normal) (step 1))"
                 (cl-gbdt/src/docgen/all:render-lambda-list
                  '(&key (kind :normal) (step 1)))))))

(deftest render-lambda-list-renders-nested-and-uninterned
  (testing "macro destructuring, and a defstruct constructor's ((:key #:name) default)"
    (ok (string= "((pointer nrow ncol) &body body)"
                 (cl-gbdt/src/docgen/all:render-lambda-list
                  '((pointer nrow ncol) &body body))))
    (ok (string= "(&key ((:verified-low verified-low) nil))"
                 (cl-gbdt/src/docgen/all:render-lambda-list
                  (list '&key (list (list :verified-low (make-symbol "VERIFIED-LOW")) nil)))))))

(deftest render-lambda-list-drops-aux
  (testing "&aux is an implementation detail of the body, not part of the contract"
    (ok (string= "(a b)"
                 (cl-gbdt/src/docgen/all:render-lambda-list '(a b &aux (c 1)))))))

(defgeneric fixture-dispatch (object other)
  (:documentation "Two-argument fixture generic."))

(defmethod fixture-dispatch ((object fixture-class) other)
  "Method on the class fixture."
  (list object other))

(deftest render-method-signature-pairs-parameters-with-specializers
  (testing "an unspecialized parameter shows T, which is what the method dispatches on"
    (let ((method (first (sb-mop:generic-function-methods #'fixture-dispatch))))
      (ok (string= "(fixture-dispatch (object fixture-class) (other t))"
                   (cl-gbdt/src/docgen/all:render-method-signature 'fixture-dispatch method))))))

(defgeneric fixture-key-method (x &key mode other)
  (:documentation "Fixture generic exercising a &key tail."))

(defmethod fixture-key-method ((x integer) &key (mode :normal) other)
  "Method exercising a &key tail."
  (list x mode other))

(deftest render-method-signature-renders-key-tail
  (testing "a &key tail renders with each parameter, defaults kept as pairs"
    (let ((method (first (sb-mop:generic-function-methods #'fixture-key-method))))
      (ok (string= "(fixture-key-method (x integer) &key (mode :normal) other)"
                   (cl-gbdt/src/docgen/all:render-method-signature 'fixture-key-method
                                                                    method))))))

(defgeneric fixture-zero-req-method (&key mode)
  (:documentation "Fixture generic with zero required parameters."))

(defmethod fixture-zero-req-method (&key (mode :normal))
  "Method with zero required parameters."
  mode)

(deftest render-method-signature-renders-zero-required-parameters
  (testing "an empty required list contributes nothing; no double space before &key"
    (let ((method (first (sb-mop:generic-function-methods #'fixture-zero-req-method))))
      (ok (string= "(fixture-zero-req-method &key (mode :normal))"
                   (cl-gbdt/src/docgen/all:render-method-signature 'fixture-zero-req-method
                                                                    method))))))

(defgeneric fixture-aux-method (x)
  (:documentation "Fixture generic whose method tail is only &aux."))

(defmethod fixture-aux-method ((x integer) &aux (y 1))
  "Method whose tail is only &aux, dropped entirely rather than left as a trailing space."
  (list x y))

(deftest render-method-signature-drops-aux-only-tail
  (testing "an &aux-only tail leaves no trailing space"
    (let ((method (first (sb-mop:generic-function-methods #'fixture-aux-method))))
      (ok (string= "(fixture-aux-method (x integer))"
                   (cl-gbdt/src/docgen/all:render-method-signature 'fixture-aux-method
                                                                    method))))))

(deftest render-lambda-list-refuses-dotted-lambda-lists
  (testing "a dotted lambda list is refused rather than silently flattened"
    (ok (handler-case (progn (cl-gbdt/src/docgen/all:render-lambda-list '(a b . c)) nil)
          (error (e) (search "dotted" (princ-to-string e)))))
    (ok (handler-case (progn (cl-gbdt/src/docgen/all:render-lambda-list '((a b . c) other))
                              nil)
          (error (e) (search "dotted" (princ-to-string e)))))))

(deftest render-documentation-fences-verbatim
  (testing "the docstring is the file's own bytes, so a diff is a docstring diff"
    (ok (string= (format nil "```text~%Two lines,~%hand wrapped.~%```~%")
                 (cl-gbdt/src/docgen/all:render-documentation
                  (format nil "Two lines,~%hand wrapped."))))))

(deftest render-documentation-refuses-a-docstring-that-breaks-its-fence
  (testing "a triple-backquote line is an error, not broken Markdown"
    (ok (handler-case
            (progn (cl-gbdt/src/docgen/all:render-documentation
                    (format nil "Before~%```~%After"))
                   nil)
          (error () t)))))

(deftest entry-anchor-is-stable-and-explicit
  (testing "the generator writes its own anchors rather than guessing GitHub's slugger"
    (ok (string= "cl-gbdt-lightgbm-predict"
                 (cl-gbdt/src/docgen/all:entry-anchor "cl-gbdt/lightgbm" 'predict)))
    (ok (string= "cl-gbdt-known-capabilities"
                 (cl-gbdt/src/docgen/all:entry-anchor "cl-gbdt" '*known-capabilities*)))))

(deftest render-entry-writes-a-function-entry
  (testing "heading, kind, signature, exports and the fenced docstring, in that order"
    (let ((text (with-output-to-string (stream)
                  (cl-gbdt/src/docgen/all:render-entry
                   (cl-gbdt/src/docgen/all:make-entry
                    :symbol 'fixture-function :qualifier "cl-gbdt/tests/docgen"
                    :exported-from '("cl-gbdt/tests/docgen") :kind :function
                    :documentation "Add A and B." :lambda-list '(a b))
                   stream))))
      (ok (search "## `cl-gbdt/tests/docgen:fixture-function`" text))
      (ok (search "- **Kind** function" text))
      (ok (search "- **Signature** `(fixture-function a b)`" text))
      (ok (search "```text" text))
      (ok (search "Add A and B." text)))))

(deftest render-entry-points-a-reader-at-its-type
  (testing "the slot's text lives on the type; the reader entry is one line"
    (let ((text (with-output-to-string (stream)
                  (cl-gbdt/src/docgen/all:render-entry
                   (cl-gbdt/src/docgen/all:make-entry
                    :symbol 'fixture-documented-condition-code
                    :qualifier "cl-gbdt/tests/docgen"
                    :exported-from '("cl-gbdt/tests/docgen") :kind :generic-function
                    :points-at '(:slot fixture-documented-condition code)
                    :lambda-list '(condition))
                   stream))))
      (ok (search "Reader of `cl-gbdt/tests/docgen:fixture-documented-condition`'s `code` slot."
                  text))
      (ok (not (search "```text" text))))))

(deftest render-entry-signature-line-for-a-zero-argument-function
  (testing "a NIL lambda list still gets a Signature line, as (name) with no stray space"
    (let ((text (with-output-to-string (stream)
                  (cl-gbdt/src/docgen/all:render-entry
                   (cl-gbdt/src/docgen/all:make-entry
                    :symbol 'fixture-function :qualifier "cl-gbdt/tests/docgen"
                    :exported-from '("cl-gbdt/tests/docgen") :kind :function
                    :documentation "Add A and B." :lambda-list nil)
                   stream))))
      ;; Ruling 2 decides the line by KIND, not by whether LAMBDA-LIST is non-NIL, so a
      ;; zero-argument function still gets one; the brief's own subseq-based body, once its
      ;; guard is loosened to match, would print "(fixture-function )" with a stray space
      ;; before the close paren -- the exact defect class ruling 1 exists to avoid.
      (ok (search "- **Signature** `(fixture-function)`" text)))))

(deftest render-entry-omits-signature-line-for-non-callable-kinds
  (testing "a variable entry never gets a Signature line, whatever LAMBDA-LIST holds"
    (let ((text (with-output-to-string (stream)
                  (cl-gbdt/src/docgen/all:render-entry
                   (cl-gbdt/src/docgen/all:make-entry
                    :symbol '*fixture-variable* :qualifier "cl-gbdt/tests/docgen"
                    :exported-from '("cl-gbdt/tests/docgen") :kind :variable
                    :documentation "A variable fixture." :lambda-list '(should-not-appear))
                   stream))))
      (ok (not (search "**Signature**" text))))))

(deftest render-entry-points-a-constructor-at-its-type
  (testing "a structure constructor also renders as one line pointing at its type"
    (let ((text (with-output-to-string (stream)
                  (cl-gbdt/src/docgen/all:render-entry
                   (cl-gbdt/src/docgen/all:make-entry
                    :symbol 'cl-gbdt/tests/docgen/gamma:make-fixture-indexed-struct
                    :qualifier "cl-gbdt/tests/docgen/gamma"
                    :exported-from '("cl-gbdt/tests/docgen/gamma") :kind :function
                    :points-at
                    '(:constructor cl-gbdt/tests/docgen/gamma:fixture-indexed-struct)
                    :lambda-list '(&key field))
                   stream))))
      (ok (search "Constructor of the `cl-gbdt/tests/docgen/gamma:fixture-indexed-struct`"
                  text))
      (ok (not (search "```text" text))))))
