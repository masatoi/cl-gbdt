;;;; emit.lisp --- Assemble the published surface into the whole reference document.
;;;;
;;;; Development only. This file knows nothing about SBCL internals, LightGBM or XGBoost; it
;;;; consumes only what src/docgen/introspect.lisp gathered and what src/docgen/render.lisp
;;;; knows how to render, which is what lets tests/docgen.lisp exercise the whole assembly on
;;;; synthetic packages of its own rather than on the library's own 174 published symbols.

(uiop:define-package #:cl-gbdt/src/docgen/emit
  (:use #:cl)
  ;; Every call below is package-qualified (CL-GBDT/SRC/DOCGEN/INTROSPECT:... and
  ;; CL-GBDT/SRC/DOCGEN/RENDER:...), so both clauses name zero symbols -- the project's
  ;; documented way to declare the dependency without importing anything.
  (:import-from #:cl-gbdt/src/docgen/introspect)
  (:import-from #:cl-gbdt/src/docgen/render)
  (:export #:+public-packages+
           #:check-anchors-unique
           #:collect-entries
           #:emit-api-reference
           #:write-api-reference))

(in-package #:cl-gbdt/src/docgen/emit)

(defparameter +public-packages+ '("cl-gbdt" "cl-gbdt/lightgbm" "cl-gbdt/xgboost")
  "The public packages, in the priority order that decides an entry's heading.

The order is the whole rule: a symbol is named by the first package here that exports it, so
`cl-gbdt:predict' and `cl-gbdt/lightgbm:predict' -- different symbols, one name -- get
different headings, each one a name that works at a REPL.")

(defun entry-methods-of (symbol kind)
  "Return (SIGNATURE . DOCSTRING) for each method of SYMBOL, sorted by signature."
  (when (eq kind :generic-function)
    (sort (mapcar (lambda (method)
                    (cons (cl-gbdt/src/docgen/render:render-method-signature symbol method)
                          (documentation method t)))
                  (sb-mop:generic-function-methods (fdefinition symbol)))
          #'string< :key #'car)))

(defun entry-lambda-list-of (symbol kind)
  "Return SYMBOL's lambda list, or NIL for a kind that has none."
  (case kind
    (:generic-function (sb-mop:generic-function-lambda-list (fdefinition symbol)))
    ((:function :macro)
     (funcall (find-symbol "FUNCTION-LAMBDA-LIST" "SB-INTROSPECT") symbol))
    (t nil)))

(defun qualifier-index (published)
  "Return a hash table mapping each PUBLISHED item's symbol to its own qualifier.

`check-reader-type-qualifier' looks up a points-at target's TYPE here, by the same
priority-order rule that decided the reader's own heading, rather than re-deriving it a
second way."
  (let ((table (make-hash-table)))
    (dolist (item published table)
      (setf (gethash (cl-gbdt/src/docgen/introspect:published-symbol item) table)
            (cl-gbdt/src/docgen/introspect:published-qualifier item)))))

(defun check-reader-type-qualifier (reader-symbol reader-qualifier target qualifiers)
  "Signal unless TARGET's type is published under READER-QUALIFIER, same as READER-SYMBOL.

`render-entry''s pointer line (\"Reader of `X:foo`'s `bar` slot\") names the pointed-at type
using the READER's own qualifier, not the type's own -- see that function's comment. A reader
and its type are two different symbols, each independently subject to the priority-order rule,
so nothing else rules out them landing under different packages; measured true for every
published reader today, but only this check enforces it."
  (when target
    (let* ((type-symbol (second target))
           (type-qualifier (gethash type-symbol qualifiers)))
      (when (and type-qualifier (not (string= reader-qualifier type-qualifier)))
        (error "The type ~S is published as ~A, but its reader ~S is published as ~A -- ~
render-entry's pointer line would name the wrong package for the type it points at."
               type-symbol type-qualifier reader-symbol reader-qualifier)))))

(defun collect-entries (package-names)
  "Return one `entry' per published symbol of PACKAGE-NAMES, in the reference's own order."
  (let* ((published (cl-gbdt/src/docgen/introspect:published-symbols package-names))
         (index (cl-gbdt/src/docgen/introspect:reader-index published))
         (qualifiers (qualifier-index published)))
    (mapcar
     (lambda (item)
       (let* ((symbol (cl-gbdt/src/docgen/introspect:published-symbol item))
              (qualifier (cl-gbdt/src/docgen/introspect:published-qualifier item))
              (kind (cl-gbdt/src/docgen/introspect:symbol-kind symbol))
              (own-documentation
                (cl-gbdt/src/docgen/introspect:symbol-documentation symbol kind))
              (points-at (unless own-documentation (gethash symbol index))))
         (when points-at
           (check-reader-type-qualifier symbol qualifier points-at qualifiers))
         (cl-gbdt/src/docgen/render:make-entry
          :symbol symbol
          :qualifier qualifier
          :exported-from (cl-gbdt/src/docgen/introspect:published-exported-from item)
          :kind kind
          :documentation own-documentation
          :lambda-list (entry-lambda-list-of symbol kind)
          :superclasses (when (member kind '(:class :condition))
                          (mapcar #'class-name
                                  (sb-mop:class-direct-superclasses (find-class symbol))))
          :slots (when (member kind '(:class :condition :structure))
                   (cl-gbdt/src/docgen/introspect:type-slots symbol))
          :methods (entry-methods-of symbol kind)
          :points-at points-at)))
     published)))

(defun check-anchors-unique (pairs)
  "Signal when two entries would share one anchor; PAIRS is a list of (ANCHOR . SYMBOL)."
  (let ((seen (make-hash-table :test #'equal)))
    (dolist (pair pairs t)
      (let ((previous (gethash (car pair) seen)))
        (when previous
          (error "Two published symbols share the anchor ~S: ~S and ~S."
                 (car pair) previous (cdr pair)))
        (setf (gethash (car pair) seen) (cdr pair))))))

(defun emit-header (stream)
  "Write the file's own explanation of what it is."
  (format stream "# API Reference~%~%~
This file is GENERATED from the docstrings of every symbol `cl-gbdt`, `cl-gbdt/lightgbm` and~%~
`cl-gbdt/xgboost` export. **Never edit it by hand** -- the same rule `src/*/c-api.lisp`~%~
carries. Regenerate it with:~%~%~
```~%ros run -- --non-interactive --load tools/gen-api-reference.lisp~%```~%~%~
`tools/ci/check-api-reference.lisp` regenerates it into a temporary file and fails the build~%~
when the committed copy differs, so a docstring edited without regenerating is caught rather~%~
than silently shipped.~%~%~
Docstrings appear verbatim inside `text` fences: they are hand-wrapped for the REPL, and they~%~
hold Lisp conventions Markdown would render as something else.~%~%"))

(defun emit-index (package-names entries stream)
  "Write the per-package index of names, each linking to its entry.

Takes PACKAGE-NAMES rather than reading `+public-packages+': the caller decides which packages
the document covers, and tests/docgen.lisp exercises this on synthetic packages of its own."
  (format stream "## Packages~%~%")
  (dolist (package package-names)
    (let ((members (remove-if-not
                    (lambda (entry)
                      (member package (cl-gbdt/src/docgen/render:entry-exported-from entry)
                              :test #'string=))
                    entries)))
      (format stream "### `~A` -- ~D symbols~%~%" package (length members))
      (dolist (entry members)
        (format stream "- [`~(~A~)`](#~A)~%"
                (symbol-name (cl-gbdt/src/docgen/render:entry-symbol entry))
                (cl-gbdt/src/docgen/render:entry-anchor
                 (cl-gbdt/src/docgen/render:entry-qualifier entry)
                 (cl-gbdt/src/docgen/render:entry-symbol entry))))
      (terpri stream))))

(defun emit-api-reference (package-names stream)
  "Write the whole reference for PACKAGE-NAMES to STREAM."
  (cl-gbdt/src/docgen/introspect:check-introspection-primitives)
  (let ((entries (collect-entries package-names)))
    (check-anchors-unique
     (mapcar (lambda (entry)
               (cons (cl-gbdt/src/docgen/render:entry-anchor
                      (cl-gbdt/src/docgen/render:entry-qualifier entry)
                      (cl-gbdt/src/docgen/render:entry-symbol entry))
                     (cl-gbdt/src/docgen/render:entry-symbol entry)))
             entries))
    (emit-header stream)
    (emit-index package-names entries stream)
    (format stream "## Symbols~%~%")
    (dolist (entry entries)
      (cl-gbdt/src/docgen/render:render-entry entry stream))))

(defun write-api-reference (package-names pathname)
  "Write the reference for PACKAGE-NAMES to PATHNAME, and return PATHNAME."
  (with-open-file (stream pathname :direction :output :if-exists :supersede
                                   :external-format :utf-8)
    (emit-api-reference package-names stream))
  pathname)
