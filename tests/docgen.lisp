;;;; docgen.lisp --- Tests for the API-reference emitter.
;;;;
;;;; Layer 1: no shared library, and no backend system either. Every fixture here is defined in
;;;; this file, so the emitter is exercised on input this file controls rather than on the
;;;; library's own 174 published symbols, which tools/ci/check-api-reference.lisp covers whole.

(uiop:define-package #:cl-gbdt/tests/docgen
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/docgen/all)
  ;; PUBLISHED-SYMBOLS is called below with "cl-gbdt" as one of its package names, so the
  ;; symbol CL-GBDT names a package must actually exist in this system's own dependency
  ;; closure -- not merely by accident, because a sibling test file in cl-gbdt/tests happens
  ;; to load core first. This is the documented zero-symbol form: every call below is
  ;; package-qualified, so nothing is imported from it.
  (:import-from #:cl-gbdt/src/all))

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

(deftest render-entry-suppresses-methods-on-a-pointer-entry
  (testing "a reader entry with methods attached renders the pointer line and no Methods section"
    (let* ((method-signature
             "(fixture-documented-condition-code (condition fixture-documented-condition))")
           (leak-marker "SBCL-ATTACHED-SLOT-TEXT-MUST-NOT-APPEAR-HERE")
           (text (with-output-to-string (stream)
                   (cl-gbdt/src/docgen/all:render-entry
                    (cl-gbdt/src/docgen/all:make-entry
                     :symbol 'fixture-documented-condition-code
                     :qualifier "cl-gbdt/tests/docgen"
                     :exported-from '("cl-gbdt/tests/docgen") :kind :generic-function
                     :points-at '(:slot fixture-documented-condition code)
                     :lambda-list '(condition)
                     ;; A non-empty METHODS list, as COLLECT-ENTRIES would actually attach for a
                     ;; DEFCLASS slot's reader, whose accessor method carries the slot's own
                     ;; :DOCUMENTATION under SBCL. If RENDER-ENTRY's pointer-entry guard were
                     ;; ever removed, this is what would leak into a Methods section.
                     :methods (list (cons method-signature leak-marker)))
                    stream))))
      ;; The pointer line still renders: a non-NIL METHODS must not suppress the whole entry,
      ;; only the Methods section.
      (ok (search "Reader of `cl-gbdt/tests/docgen:fixture-documented-condition`'s `code` slot."
                  text))
      ;; Hard pins, not a substring search over a larger blob that could pass by accident: the
      ;; heading must be entirely absent, and so must the attached method's own marked text --
      ;; proving the content is suppressed, not merely that the heading is missing while the
      ;; text leaks some other way.
      (ok (not (search "### Methods" text)))
      (ok (not (search leak-marker text)))
      (ok (not (search (format nil "#### `~A`" method-signature) text))))))

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

(deftest render-entry-writes-a-type-entrys-superclasses-line
  (testing "a type entry renders its Superclasses line and never a Signature line"
    (let ((text (with-output-to-string (stream)
                  (cl-gbdt/src/docgen/all:render-entry
                   (cl-gbdt/src/docgen/all:make-entry
                    :symbol 'fixture-documented-class :qualifier "cl-gbdt/tests/docgen"
                    :exported-from '("cl-gbdt/tests/docgen") :kind :class
                    :documentation "A class whose slot carries its own text."
                    :superclasses '(standard-object))
                   stream))))
      (ok (search "- **Superclasses** `standard-object`" text))
      (ok (not (search "- **Signature**" text))))))

(deftest render-entry-writes-a-slots-section
  (testing "each slot in the Slots section shows its name, its readers, and its fenced text"
    (let* ((slots (cl-gbdt/src/docgen/all:type-slots 'fixture-documented-class))
           (text (with-output-to-string (stream)
                   (cl-gbdt/src/docgen/all:render-entry
                    (cl-gbdt/src/docgen/all:make-entry
                     :symbol 'fixture-documented-class :qualifier "cl-gbdt/tests/docgen"
                     :exported-from '("cl-gbdt/tests/docgen") :kind :class
                     ;; ENTRY-DOCUMENTATION is left NIL here on purpose, as
                     ;; RENDER-ENTRY-SLOTS-SECTION-OMITS-FENCE-FOR-AN-UNDOCUMENTED-SLOT
                     ;; already does below: with it set, the entry's own fence would satisfy
                     ;; the "```text" and slot-text assertions below regardless of whether
                     ;; the SLOT's own text is fenced, so this test would pin nothing about
                     ;; the Slots section at all.
                     :slots slots)
                    stream))))
      (ok (search "### Slots" text))
      (ok (search "#### `tag`" text))
      (ok (search "- **Readers** `fixture-documented-class-tag`" text))
      (ok (search "```text" text))
      (ok (search "What this object is called." text)))))

(deftest render-entry-writes-a-methods-section
  (testing "each method in the Methods section shows its signature and its fenced docstring"
    (let* ((method (first (sb-mop:generic-function-methods #'fixture-dispatch)))
           (signature (cl-gbdt/src/docgen/all:render-method-signature 'fixture-dispatch method))
           (text (with-output-to-string (stream)
                   (cl-gbdt/src/docgen/all:render-entry
                    (cl-gbdt/src/docgen/all:make-entry
                     :symbol 'fixture-dispatch :qualifier "cl-gbdt/tests/docgen"
                     :exported-from '("cl-gbdt/tests/docgen") :kind :generic-function
                     ;; ENTRY-DOCUMENTATION is left NIL here on purpose -- see the matching
                     ;; comment in RENDER-ENTRY-WRITES-A-SLOTS-SECTION above. With it set, the
                     ;; entry's own fence would satisfy the "```text" assertion below whether
                     ;; or not the METHOD's own docstring is fenced.
                     :lambda-list '(object other)
                     :methods (list (cons signature "Method on the class fixture.")))
                    stream))))
      (ok (search "### Methods" text))
      (ok (search (format nil "#### `~A`" signature) text))
      (ok (search "```text" text))
      (ok (search "Method on the class fixture." text)))))

(deftest render-entry-slots-section-omits-fence-for-an-undocumented-slot
  (testing "a slot with no documentation contributes no fence at all, not an empty one"
    (let* ((slots (cl-gbdt/src/docgen/all:type-slots 'fixture-struct))
           (text (with-output-to-string (stream)
                   (cl-gbdt/src/docgen/all:render-entry
                    (cl-gbdt/src/docgen/all:make-entry
                     :symbol 'fixture-struct :qualifier "cl-gbdt/tests/docgen"
                     :exported-from '("cl-gbdt/tests/docgen") :kind :structure
                     :slots slots)
                    stream))))
      ;; ENTRY-DOCUMENTATION is left NIL here on purpose, so the only way "```text" could
      ;; appear in TEXT at all is through the Slots section -- isolating what this test pins.
      (ok (search "#### `field`" text))
      (ok (not (search "```text" text))))))

(defpackage #:cl-gbdt/tests/docgen/delta
  (:use #:cl)
  (:export #:fixture-published-generic))

(in-package #:cl-gbdt/tests/docgen/delta)

(defgeneric fixture-published-generic (object)
  (:documentation "A published generic fixture."))

(defmethod fixture-published-generic ((object integer))
  "The integer method."
  object)

(in-package #:cl-gbdt/tests/docgen)

(deftest emit-api-reference-is-deterministic
  (testing "two runs over one image give identical bytes; nothing about the machine leaks in"
    (let ((first (with-output-to-string (s)
                   (cl-gbdt/src/docgen/all:emit-api-reference
                    '("cl-gbdt/tests/docgen/gamma") s)))
          (second (with-output-to-string (s)
                    (cl-gbdt/src/docgen/all:emit-api-reference
                     '("cl-gbdt/tests/docgen/gamma") s))))
      (ok (string= first second)))))

(deftest emit-api-reference-indexes-every-published-symbol
  (testing "the index links to an anchor the body actually defines"
    (let ((text (with-output-to-string (s)
                  (cl-gbdt/src/docgen/all:emit-api-reference
                   '("cl-gbdt/tests/docgen/gamma") s))))
      (dolist (name '("fixture-indexed-struct" "fixture-indexed-struct-field"
                      "make-fixture-indexed-struct" "fixture-indexed-condition"
                      "fixture-indexed-condition-code"))
        (ok (search (format nil "](#cl-gbdt-tests-docgen-gamma-~A)" name) text))
        (ok (search (format nil "<a id=\"cl-gbdt-tests-docgen-gamma-~A\"></a>" name) text))))))

(deftest emit-api-reference-refuses-duplicate-anchors
  (testing "two entries with one anchor would make the index ambiguous"
    (ok (handler-case
            (progn (cl-gbdt/src/docgen/all:check-anchors-unique
                    (list (cons "same" 'alpha) (cons "same" 'beta)))
                   nil)
          (error () t)))))

(deftest collect-entries-attaches-methods-to-their-generic
  (testing "a generic's entry carries every method's signature and docstring"
    (let* ((entries (cl-gbdt/src/docgen/all:collect-entries '("cl-gbdt/tests/docgen/delta")))
           (entry (find 'cl-gbdt/tests/docgen/delta:fixture-published-generic entries
                        :key #'cl-gbdt/src/docgen/all:entry-symbol)))
      (ok (= 1 (length (cl-gbdt/src/docgen/all:entry-methods entry))))
      (ok (search "fixture-published-generic"
                  (car (first (cl-gbdt/src/docgen/all:entry-methods entry))))))))

(defpackage #:cl-gbdt/tests/docgen/epsilon
  (:use #:cl)
  (:export #:fixture-mismatched-condition))

(in-package #:cl-gbdt/tests/docgen/epsilon)

(define-condition fixture-mismatched-condition (error)
  ((code :initarg :code :reader fixture-mismatched-condition-code
         :documentation "A code."))
  (:documentation "A condition fixture whose reader is republished under a different package."))

(in-package #:cl-gbdt/tests/docgen)

(defpackage #:cl-gbdt/tests/docgen/zeta
  (:use #:cl)
  (:import-from #:cl-gbdt/tests/docgen/epsilon #:fixture-mismatched-condition-code)
  (:export #:fixture-mismatched-condition-code))

(deftest collect-entries-signals-when-a-readers-type-has-a-different-qualifier
  (testing "ruling 1: render-entry names the type using the reader's own qualifier"
    (ok (handler-case
            (progn (cl-gbdt/src/docgen/all:collect-entries
                    '("cl-gbdt/tests/docgen/zeta" "cl-gbdt/tests/docgen/epsilon"))
                   nil)
          (error (e)
            (let ((message (princ-to-string e)))
              (and (search "FIXTURE-MISMATCHED-CONDITION-CODE" message)
                   (search "FIXTURE-MISMATCHED-CONDITION" message)
                   (search "cl-gbdt/tests/docgen/zeta" message)
                   (search "cl-gbdt/tests/docgen/epsilon" message))))))))

(defpackage #:cl-gbdt/tests/docgen/eta-a
  (:use #:cl)
  (:export #:widget))

(in-package #:cl-gbdt/tests/docgen/eta-a)

(defclass widget () ())

(in-package #:cl-gbdt/tests/docgen)

(defpackage #:cl-gbdt/tests/docgen/eta-b
  (:use #:cl)
  (:export #:widget))

(in-package #:cl-gbdt/tests/docgen/eta-b)

(defclass widget () ())

(in-package #:cl-gbdt/tests/docgen)

(defpackage #:cl-gbdt/tests/docgen/theta
  (:use #:cl)
  (:export #:fixture-tie-generic))

(in-package #:cl-gbdt/tests/docgen/theta)

(defgeneric fixture-tie-generic (object)
  (:documentation "Fixture generic whose two methods render identical signature text."))

(defmethod fixture-tie-generic ((object cl-gbdt/tests/docgen/eta-a:widget))
  "Method on eta-a's widget."
  object)

(defmethod fixture-tie-generic ((object cl-gbdt/tests/docgen/eta-b:widget))
  "Method on eta-b's widget."
  object)

(in-package #:cl-gbdt/tests/docgen)

(deftest collect-entries-orders-tied-methods-by-specializer-package
  (testing "two methods whose rendered signature text ties still sort deterministically"
    (let* ((entries (cl-gbdt/src/docgen/all:collect-entries '("cl-gbdt/tests/docgen/theta")))
           (entry (find 'cl-gbdt/tests/docgen/theta:fixture-tie-generic entries
                        :key #'cl-gbdt/src/docgen/all:entry-symbol))
           (methods (cl-gbdt/src/docgen/all:entry-methods entry)))
      (ok (= 2 (length methods)))
      ;; Both specializers print as "widget" once RENDER-SPECIALIZER drops their package --
      ;; confirming the fixture really does tie under the old (rendered-text) sort key.
      (ok (string= (car (first methods)) (car (second methods))))
      ;; The docstrings differ, so which one landed first still pins the actual order: eta-a
      ;; sorts before eta-b by package name, independent of SB-MOP's own method-list order.
      (ok (string= "Method on eta-a's widget." (cdr (first methods))))
      (ok (string= "Method on eta-b's widget." (cdr (second methods)))))))

(defpackage #:cl-gbdt/tests/docgen/iota
  (:use #:cl)
  (:export #:fixture-self-documented-condition #:fixture-self-documented-condition-code))

(in-package #:cl-gbdt/tests/docgen/iota)

(define-condition fixture-self-documented-condition (error)
  ((code :initarg :code :reader fixture-self-documented-condition-code
         :documentation "The slot's own text, which the reader's entry must not show."))
  (:documentation "A condition fixture whose reader carries its own separate documentation."))

(setf (documentation 'fixture-self-documented-condition-code 'function)
      "The reader's own text, which must win over a pointer to its slot.")

(in-package #:cl-gbdt/tests/docgen)

(deftest collect-entries-does-not-point-a-self-documented-reader-at-its-type
  (testing "a reader with its own docstring keeps it; no pointer line, no ruling-1 signal"
    (let* ((entries (cl-gbdt/src/docgen/all:collect-entries '("cl-gbdt/tests/docgen/iota")))
           (entry (find 'cl-gbdt/tests/docgen/iota:fixture-self-documented-condition-code
                        entries :key #'cl-gbdt/src/docgen/all:entry-symbol)))
      (ok (null (cl-gbdt/src/docgen/all:entry-points-at entry)))
      (ok (string= "The reader's own text, which must win over a pointer to its slot."
                   (cl-gbdt/src/docgen/all:entry-documentation entry))))))

(deftest every-published-reader-slot-is-documented
  (testing "the floor tools/ci/check-api-reference.lisp enforces, at layer 1 speed"
    (let ((offenders '()))
      (dolist (package '("cl-gbdt"))
        (let ((found (cl-gbdt/src/docgen/all:published-symbols (list package))))
          (dolist (item found)
            (let ((symbol (cl-gbdt/src/docgen/all:published-symbol item)))
              (when (and (find-class symbol nil)
                         (not (typep (find-class symbol) 'structure-class)))
                (dolist (slot (cl-gbdt/src/docgen/all:type-slots symbol))
                  (when (and (cl-gbdt/src/docgen/all:slot-info-readers slot)
                             (null (cl-gbdt/src/docgen/all:slot-info-documentation slot)))
                    (push (cl-gbdt/src/docgen/all:slot-info-name slot) offenders))))))))
      (ok (null offenders) (format nil "undocumented published slots: ~S" offenders)))))

(defpackage #:cl-gbdt/tests/docgen/mu
  (:use #:cl)
  (:export #:fixture-partial-class #:fixture-partial-class-published))

(in-package #:cl-gbdt/tests/docgen/mu)

(defclass fixture-partial-class ()
  ((published :initarg :published
              :reader fixture-partial-class-published
              :documentation "A slot whose reader is published.")
   ;; Not in this package's :EXPORT clause, on purpose -- the same shape as `backend''s
   ;; internal `backend-openp' accessor one hyphen from the exported `backend-open-p'
   ;; wrapping it (finding 1's own example).
   (unpublished :initarg :unpublished
                :reader fixture-partial-class-unpublished
                :documentation "A slot whose reader is not published."))
  (:documentation "A class fixture with one published and one unpublished reader."))

(in-package #:cl-gbdt/tests/docgen)

(deftest collect-entries-filters-an-unpublished-reader-out-of-its-slot
  (testing "finding 1: an internal :READER stays off the reference though TYPE-SLOTS sees it"
    (let* ((entries (cl-gbdt/src/docgen/all:collect-entries '("cl-gbdt/tests/docgen/mu")))
           (entry (find 'cl-gbdt/tests/docgen/mu:fixture-partial-class entries
                        :key #'cl-gbdt/src/docgen/all:entry-symbol))
           (slots (cl-gbdt/src/docgen/all:entry-slots entry))
           ;; Neither slot name is exported by MU -- see GAMMA's own tests above for why a
           ;; bare slot name is reached with `::' rather than `:' here.
           (published-slot (find 'cl-gbdt/tests/docgen/mu::published slots
                                  :key #'cl-gbdt/src/docgen/all:slot-info-name))
           (unpublished-slot (find 'cl-gbdt/tests/docgen/mu::unpublished slots
                                    :key #'cl-gbdt/src/docgen/all:slot-info-name)))
      (ok (equal '(cl-gbdt/tests/docgen/mu:fixture-partial-class-published)
                 (cl-gbdt/src/docgen/all:slot-info-readers published-slot)))
      (ok (null (cl-gbdt/src/docgen/all:slot-info-readers unpublished-slot))))))

(deftest render-entry-omits-the-readers-line-for-a-slot-with-no-published-reader
  (testing "finding 1: an empty READERS list renders no Readers line at all, not an empty one"
    (let* ((entries (cl-gbdt/src/docgen/all:collect-entries '("cl-gbdt/tests/docgen/mu")))
           (entry (find 'cl-gbdt/tests/docgen/mu:fixture-partial-class entries
                        :key #'cl-gbdt/src/docgen/all:entry-symbol))
           (text (with-output-to-string (stream)
                   (cl-gbdt/src/docgen/all:render-entry entry stream)))
           (unpublished-heading (search "#### `unpublished`" text)))
      (ok (search "- **Readers** `fixture-partial-class-published`" text))
      (ok unpublished-heading)
      ;; The unpublished slot's own section -- from its heading to the next "####" or the end
      ;; of TEXT -- carries no Readers line, not an empty one: FILTER-SLOT-READERS already
      ;; dropped the unpublished reader inside COLLECT-ENTRIES, and RENDER-ENTRY only ever
      ;; omits the whole line, never prints one with nothing after it.
      (let* ((section-end (or (search "####" text :start2 (+ unpublished-heading 4))
                               (length text)))
             (section (subseq text unpublished-heading section-end)))
        (ok (not (search "**Readers**" section)))))))

(defpackage #:cl-gbdt/tests/docgen/kappa-p
  (:use #:cl)
  (:export #:q-r))

(in-package #:cl-gbdt/tests/docgen/kappa-p)

(defun q-r () "Fixture function for an anchor collision." nil)

(in-package #:cl-gbdt/tests/docgen)

(defpackage #:cl-gbdt/tests/docgen/kappa-p-q
  (:use #:cl)
  (:export #:r))

(in-package #:cl-gbdt/tests/docgen/kappa-p-q)

(defun r () "Fixture function for an anchor collision." nil)

(in-package #:cl-gbdt/tests/docgen)

(deftest emit-api-reference-is-wired-to-check-anchors-unique
  (testing "finding 6: EMIT-API-REFERENCE itself signals, not just a direct call to the guard"
    ;; cl-gbdt/tests/docgen/kappa-p exports Q-R and cl-gbdt/tests/docgen/kappa-p-q exports R;
    ;; ENTRY-ANCHOR joins qualifier and symbol name with a hyphen, so both collapse to the same
    ;; anchor ("...kappa-p-q-r") -- exactly the collision ENTRY-ANCHOR's own docstring warns it
    ;; cannot detect on its own. Running EMIT-API-REFERENCE end to end here, rather than calling
    ;; CHECK-ANCHORS-UNIQUE directly the way EMIT-API-REFERENCE-REFUSES-DUPLICATE-ANCHORS
    ;; already does above, is what proves the guard is still wired into the real emission path
    ;; and not merely correct in isolation -- an edit dropping the call from
    ;; EMIT-API-REFERENCE would pass every other test in this file and still pass this one,
    ;; unless this one exists.
    (ok (handler-case
            (progn (with-output-to-string (stream)
                     (cl-gbdt/src/docgen/all:emit-api-reference
                      '("cl-gbdt/tests/docgen/kappa-p" "cl-gbdt/tests/docgen/kappa-p-q")
                      stream))
                   nil)
          (error () t)))))
