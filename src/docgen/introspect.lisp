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
           #:symbol-kind
           #:symbol-documentation
           #:slot-documentation
           #:type-slots
           #:filter-slot-readers
           #:slot-info
           #:slot-info-name
           #:slot-info-readers
           #:slot-info-documentation
           #:reader-index))

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
precedence rule. That is a measured premise, not a guarantee, and the class branch below is
where it would first stop holding -- CLASS wins unconditionally there, so a future symbol
that is both a type and `fboundp' would have its function or macro signature and docstring
silently dropped from the reference with no failure anywhere, the exact defect class this
docstring's second sentence already rules against for an unclassifiable symbol. Checked here
rather than left implicit, the same way the final clause below already refuses a symbol this
function cannot classify at all."
  (let ((class (find-class symbol nil)))
    (cond (class
           (when (fboundp symbol)
             (error "The published symbol ~S is both a type (SYMBOL-KIND's class branch) ~
and FBOUNDP, so its class-wins precedence rule no longer holds: this call would classify ~S ~
as a class, condition or structure and silently drop its function or macro signature and ~
docstring from the reference. Give SYMBOL-KIND an explicit rule for this case instead of ~
letting one branch win by accident."
                    symbol symbol))
           (cond ((subtypep symbol 'condition) :condition)
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

(defun symbol-documentation (symbol kind)
  "Return SYMBOL's docstring for KIND, reading exactly one doc-type.

SBCL answers a type's docstring under both the `type' and the `structure' doc-type, so a sweep
of all four doc-types double-counts every type. One doc-type per kind is what the reference
emits and what its floor counts."
  (ecase kind
    ((:function :generic-function :macro) (documentation symbol 'function))
    ((:class :condition :structure) (documentation symbol 'type))
    (:variable (documentation symbol 'variable))))

(defun slot-documentation (slot)
  "Return SLOT's :documentation, for a condition slot as well as a standard-class slot.

`(documentation slot t)' warns and returns NIL for a condition slot in SBCL
(SB-PCL::CONDITION-DIRECT-SLOT-DEFINITION is not a doc-type T object); this internal reader
answers for both, so there is one mechanism here rather than two."
  (funcall (primitive "SB-PCL" "%SLOT-DEFINITION-DOCUMENTATION") slot))

(defstruct (slot-info (:constructor %make-slot-info))
  "One direct slot of a published type: its name, its readers, and its own text."
  (name nil :read-only t)
  (readers nil :read-only t)
  (documentation nil :read-only t))

(defun type-slots (type-symbol)
  "Return TYPE-SYMBOL's slots as `slot-info' structs, in the order they were written.

The two branches answer a different question, because their underlying primitives do. The CLOS
branch calls `class-direct-slots', so it answers only TYPE-SYMBOL's own direct slots -- an
inherited slot is documented once, under the type that declares it. The structure branch has no
such filter: `sb-kernel:dd-slots' returns every slot TYPE-SYMBOL's `defstruct' description
holds, including any inherited through `:include'."
  (let ((class (find-class type-symbol)))
    (if (typep class 'structure-class)
        ;; No published structure uses :include today, so DD-SLOTS never actually returns an
        ;; inherited slot here. If one ever does, this branch would need to filter DD-SLOTS down
        ;; to the slots TYPE-SYMBOL's own DEFSTRUCT form added -- e.g. by comparing against the
        ;; :include'd description's own DD-SLOTS -- the way the CLOS branch already gets that
        ;; filtering for free from CLASS-DIRECT-SLOTS.
        (let ((description (sb-kernel:find-defstruct-description type-symbol)))
          (mapcar (lambda (dsd)
                    (%make-slot-info :name (sb-kernel:dsd-name dsd)
                                     :readers (list (sb-kernel:dsd-accessor-name dsd))
                                     :documentation nil))
                  (sb-kernel:dd-slots description)))
        (progn
          (sb-mop:finalize-inheritance class)
          (mapcar (lambda (slot)
                    (%make-slot-info :name (sb-mop:slot-definition-name slot)
                                     :readers (sb-mop:slot-definition-readers slot)
                                     :documentation (slot-documentation slot)))
                  (sb-mop:class-direct-slots class))))))

(defun filter-slot-readers (slot published-p)
  "Return a copy of SLOT whose READERS are only those satisfying PUBLISHED-P.

TYPE-SLOTS above answers a narrower question than the reference needs: what SBCL's MOP or
`defstruct' description says reads the slot, full stop, with no notion of which package
exports what. `backend-openp' reads `backend''s `openp' slot exactly as much as the exported
`backend-open-p' wrapping it does, and TYPE-SLOTS cannot tell them apart -- doing so needs the
published-symbol set, which is `src/docgen/emit.lisp''s to know, not this file's: this file
gathers raw introspection facts, and `collect-entries' already builds that set as
`qualifier-index' for an unrelated purpose. So the caller passes it in as PUBLISHED-P, and
filtering happens where the answer to \"is this published\" already lives, rather than
duplicating that set here or leaving TYPE-SLOTS to guess at it."
  (%make-slot-info :name (slot-info-name slot)
                    :readers (remove-if-not published-p (slot-info-readers slot))
                    :documentation (slot-info-documentation slot)))

(defun reader-index (published)
  "Map each published reader, accessor and structure constructor to what it reads.

A reader's entry in the reference is a pointer at its type, so the slot's text appears exactly
once, on the type. A structure's accessors and constructor are read out of its defstruct
description rather than inferred from its name: `make-csr-matrix' is a plain `defun' while
`make-version-range' is a constructor, and no naming rule separates those two."
  (let ((index (make-hash-table))
        (symbols (mapcar #'published-symbol published)))
    (dolist (symbol symbols index)
      (when (find-class symbol nil)
        (let ((class (find-class symbol)))
          (if (typep class 'structure-class)
              (let ((description (sb-kernel:find-defstruct-description symbol)))
                (dolist (dsd (sb-kernel:dd-slots description))
                  (setf (gethash (sb-kernel:dsd-accessor-name dsd) index)
                        (list :slot symbol (sb-kernel:dsd-name dsd))))
                (dolist (constructor (funcall (primitive "SB-KERNEL" "DD-CONSTRUCTORS")
                                              description))
                  (setf (gethash (car constructor) index) (list :constructor symbol))))
              (dolist (slot (type-slots symbol))
                (dolist (reader (slot-info-readers slot))
                  (setf (gethash reader index)
                        (list :slot symbol (slot-info-name slot)))))))))))
