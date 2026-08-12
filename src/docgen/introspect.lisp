;;;; introspect.lisp --- Read the published surface and its documentation out of a loaded image.
;;;;
;;;; Development only; never part of a build. The public surface of this library is what its
;;;; packages export, and that is a fact only a loaded image holds: `cl-gbdt' re-exports ten
;;;; packages, each with its own :export clause. Deriving it from source would give that fact a
;;;; second home to go stale in -- the same argument that made `wrapped' derived rather than
;;;; written down in ffi-spec/BINDING-COVERAGE.md.
;;;;
;;;; This file gathers; src/docgen/render.lisp renders. Nothing here writes Markdown.

(uiop:define-package #:cl-gbdt/src/docgen/introspect
  (:use #:cl)
  (:export #:+required-primitives+
           #:check-introspection-primitives
           #:published
           #:published-symbol
           #:published-qualifier
           #:published-exported-from
           #:published-symbols
           #:symbol-kind))

(in-package #:cl-gbdt/src/docgen/introspect)

(eval-when (:load-toplevel :execute)
  (require :sb-introspect))

(defparameter +required-primitives+
  '(("SB-PCL" "%SLOT-DEFINITION-DOCUMENTATION")
    ("SB-KERNEL" "DD-CONSTRUCTORS")
    ("SB-INTROSPECT" "FUNCTION-LAMBDA-LIST"))
  "The non-ANSI symbols this emitter calls, as (package-name symbol-name) string pairs.

`documentation' cannot reach a `define-condition' slot's :documentation in SBCL, and a
`defstruct''s constructors are not reachable through the MOP at all; these three fill both
gaps. They are checked by name before anything else runs so that their disappearance in a
future SBCL is a named failure rather than a reference that silently loses 62 bodies.")

(defun check-introspection-primitives (&optional (primitives +required-primitives+))
  "Return T when every entry of PRIMITIVES exists, or signal naming those that do not."
  (let ((missing (remove-if (lambda (pair)
                              (destructuring-bind (package name) pair
                                (let ((found (and (find-package package)
                                                  (find-symbol name package))))
                                  (and found (fboundp found)))))
                            primitives)))
    (when missing
      (error "This SBCL is missing introspection primitives the API-reference emitter needs: ~
~{~{~A::~A~}~^, ~}. See src/docgen/introspect.lisp's +required-primitives+."
             missing))
    t))

(defun primitive (package name)
  "Return the function named by PACKAGE and NAME, checking that it is there."
  (check-introspection-primitives (list (list package name)))
  (fdefinition (find-symbol name package)))

(defstruct (published (:constructor %make-published))
  "One published symbol: what it is called, and by which packages.

QUALIFIER is the package the reference names it by -- the first package in the caller's own
priority order that exports it -- so `cl-gbdt:predict' and `cl-gbdt/lightgbm:predict', which
are different symbols, get different headings. EXPORTED-FROM lists every package that exports
it, in that same order."
  (symbol nil :read-only t)
  (qualifier nil :read-only t)
  (exported-from nil :read-only t))

(defun published-symbols (package-names)
  "Return the union of PACKAGE-NAMES' external symbols as `published' structs.

PACKAGE-NAMES is a list of package-name strings in priority order: the earlier a package, the
more it is preferred as a symbol's QUALIFIER. The result is sorted by symbol name and then by
qualifier, a total order that does not depend on `do-external-symbols'."
  (let ((table (make-hash-table)))
    (dolist (name package-names)
      ;; NAME is a string in the caller's own case (this codebase's convention is lower
      ;; case, matching its ASDF system names); the reader upcases every package name it
      ;; creates from a #:token, so the lookup itself must upcase too. QUALIFIER and
      ;; EXPORTED-FROM below still store NAME verbatim, in the caller's own case.
      (let ((package (or (find-package (string-upcase name))
                         (error "No package named ~A; load its system first." name))))
        (do-external-symbols (symbol package)
          (pushnew name (gethash symbol table) :test #'string=))))
    (let ((result '()))
      (maphash (lambda (symbol names)
                 (let ((ordered (remove-if-not (lambda (n) (member n names :test #'string=))
                                               package-names)))
                   (push (%make-published :symbol symbol
                                          :qualifier (first ordered)
                                          :exported-from ordered)
                         result)))
               table)
      (sort result (lambda (a b)
                     (let ((na (symbol-name (published-symbol a)))
                           (nb (symbol-name (published-symbol b))))
                       (if (string= na nb)
                           (string< (published-qualifier a) (published-qualifier b))
                           (string< na nb))))))))

(defun symbol-kind (symbol)
  "Return the keyword naming what SYMBOL is, or signal when it is nothing this emitter renders.

No published symbol is both a type and `fboundp', so the answer is single-valued and needs no
precedence rule. An unclassifiable symbol is an error rather than a blank entry: a published
symbol the reference cannot describe is a defect in the surface, not in the reference."
  (let ((class (find-class symbol nil)))
    (cond (class (cond ((subtypep symbol 'condition) :condition)
                       ((typep class 'structure-class) :structure)
                       (t :class)))
          ((and (fboundp symbol) (macro-function symbol)) :macro)
          ((and (fboundp symbol) (typep (fdefinition symbol) 'generic-function))
           :generic-function)
          ((fboundp symbol) :function)
          ((boundp symbol) :variable)
          (t (error "Cannot classify the published symbol ~S: it names no function, macro, ~
generic function, class, condition, structure or bound variable."
                    symbol)))))
