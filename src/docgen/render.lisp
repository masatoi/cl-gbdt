;;;; render.lisp --- Turn what src/docgen/introspect.lisp gathered into Markdown text.
;;;;
;;;; Development only. This file knows nothing about how its input was obtained, which is what
;;;; lets tests/docgen.lisp exercise it on synthetic input rather than on the library's own
;;;; published surface.

(uiop:define-package #:cl-gbdt/src/docgen/render
  (:use #:cl)
  ;; Every SLOT-INFO-* reader below is called package-qualified
  ;; (CL-GBDT/SRC/DOCGEN/INTROSPECT:SLOT-INFO-...), so this names zero symbols -- the project's
  ;; documented way to declare the dependency without importing anything.
  (:import-from #:cl-gbdt/src/docgen/introspect)
  (:export #:render-lambda-list
           #:render-method-signature
           #:entry
           #:make-entry
           #:entry-symbol
           #:entry-qualifier
           #:entry-exported-from
           #:entry-kind
           #:entry-documentation
           #:entry-lambda-list
           #:entry-superclasses
           #:entry-slots
           #:entry-methods
           #:entry-points-at
           #:entry-anchor
           #:render-documentation
           #:render-entry))

(in-package #:cl-gbdt/src/docgen/render)

(defun render-atom (object)
  "Return OBJECT as reference text: a name for a symbol, a printed form for anything else.

A keyword keeps its colon -- downcasing `symbol-name' alone would turn (kind :normal) into
(kind normal), which is a different default. An uninterned symbol, which is what a defstruct
constructor's lambda list holds, prints as its name like any other."
  (cond ((keywordp object) (format nil ":~(~A~)" (symbol-name object)))
        ((symbolp object) (string-downcase (symbol-name object)))
        (t (let ((*print-case* :downcase)
                 (*print-readably* nil))
             (prin1-to-string object)))))

(defun require-proper-list (list)
  "Signal an error naming LIST if LIST is not a proper list.

This renderer refuses to render a dotted lambda list -- whether the top-level lambda list
itself or a nested destructuring sub-pattern within it -- rather than flatten the dot's
rest-cons into what would print as an ordinary positional parameter, silently discarding the
distinction the dot exists to express."
  (unless (listp (cdr (last list)))
    (error "render-lambda-list does not render dotted lambda lists: ~S" list)))

(defun render-lambda-list-element (element)
  "Return ELEMENT as reference text, descending into destructuring and default pairs.

Signals through `require-proper-list' if ELEMENT is a dotted sub-pattern, rather than flatten
it -- see that function."
  (if (consp element)
      (progn
        (require-proper-list element)
        (format nil "(~{~A~^ ~})" (mapcar #'render-lambda-list-element element)))
      (render-atom element)))

(defun render-lambda-list-elements (lambda-list)
  "Return LAMBDA-LIST's visible elements as a list of reference-text strings, not yet joined.

Shared by `render-lambda-list', which wraps the result in parens, and `render-method-signature',
which splices it into a larger list of parts instead of formatting a parenthesized string and
then stripping the parens back off -- string surgery that produced malformed output when the
result was empty. Drops everything from `&aux' onward, since it binds locals in the body and is
no part of what a caller passes; signals through `require-proper-list' if LAMBDA-LIST is
dotted."
  (require-proper-list lambda-list)
  (let ((visible (loop for element in lambda-list
                       until (eq element '&aux)
                       collect element)))
    (mapcar #'render-lambda-list-element visible)))

(defun render-lambda-list (lambda-list)
  "Return LAMBDA-LIST as reference text: names only, lower case, defaults kept as pairs.

`sb-introspect:function-lambda-list' answers with the defining package's own internal symbols,
which are unreadable printed whole. Everything after `&aux' is dropped: it binds locals in the
body and is no part of what a caller passes. Signals an error, naming LAMBDA-LIST or whichever
nested sub-pattern is at fault, if either is a dotted list -- see `require-proper-list'."
  (format nil "(~{~A~^ ~})" (render-lambda-list-elements lambda-list)))

(defun render-specializer (specializer)
  "Return SPECIALIZER as reference text: a class name, or (eql OBJECT)."
  (if (typep specializer 'sb-mop:eql-specializer)
      (format nil "(eql ~A)" (render-atom (sb-mop:eql-specializer-object specializer)))
      (render-atom (class-name specializer))))

(defun render-method-signature (name method)
  "Return METHOD's signature as reference text, each required parameter with its specializer.

An unspecialized required parameter shows T rather than nothing: T is what it dispatches on,
and hiding it would make two methods that differ only there print identically. The name, each
required parameter, and the rendered `&optional'/`&rest'/`&key' tail are collected as one flat
list of strings and joined exactly once -- an empty required list or an empty tail then
contributes nothing by construction, rather than needing the two-symptom string surgery this
replaced (a trailing space when the tail was `&aux'-only, a doubled space with no required
parameters) to strip a stray space back out after the fact."
  (let* ((lambda-list (sb-mop:method-lambda-list method))
         (specializers (sb-mop:method-specializers method))
         (required (loop for parameter in lambda-list
                         for specializer in specializers
                         collect (format nil "(~A ~A)"
                                         (render-atom parameter)
                                         (render-specializer specializer))))
         (tail (render-lambda-list-elements (nthcdr (length specializers) lambda-list))))
    (format nil "(~{~A~^ ~})" (list* (render-atom name) (append required tail)))))

(defstruct (entry (:constructor make-entry))
  "One published symbol as the reference describes it.

POINTS-AT is non-NIL exactly for a reader, accessor or structure constructor: (:slot TYPE
SLOT-NAME) or (:constructor TYPE). Such an entry renders as one line pointing at its type,
because the slot's own text belongs on the type and appears there once."
  (symbol nil) (qualifier nil) (exported-from nil) (kind nil) (documentation nil)
  (lambda-list nil) (superclasses nil) (slots nil) (methods nil) (points-at nil))

(defun entry-anchor (qualifier symbol)
  "Return the explicit HTML anchor id for SYMBOL published as QUALIFIER.

Explicit rather than inferred: GitHub's heading slugger is not a contract, and the index has to
link to every entry. Every character outside a-z and 0-9 becomes a hyphen, runs collapse, and
leading and trailing hyphens go."
  ;; This mapping is lossy by construction and can collide: (entry-anchor "cl-gbdt" 'foo-bar)
  ;; and (entry-anchor "cl-gbdt-foo" 'bar) both yield "cl-gbdt-foo-bar", because the hyphen
  ;; joining QUALIFIER to SYMBOL-NAME reads back identically to a hyphen standing in for some
  ;; other character. This function does not detect that -- Task 5's check-anchors-unique is
  ;; the guard, and it signals rather than silently emitting an index with an ambiguous link.
  (let* ((raw (format nil "~A-~A" qualifier (symbol-name symbol)))
         (mapped (map 'string
                      (lambda (character)
                        (if (or (alphanumericp character)) (char-downcase character) #\-))
                      raw))
         (collapsed (with-output-to-string (out)
                      (loop with previous = #\-
                            for character across mapped
                            unless (and (char= character #\-) (char= previous #\-))
                              do (write-char character out)
                            do (setf previous character)))))
    (string-trim "-" collapsed)))

(defun render-documentation (documentation)
  "Return DOCUMENTATION inside a `text' fence, verbatim, or the empty string for NIL.

Verbatim, not reflowed as Markdown prose: these docstrings are hand-wrapped inside 100 columns
for `(documentation ...)' at the REPL, and they hold the Lisp `foo' convention, *earmuffed*
names, ~A directives and lines beginning with a hyphen -- all of which Markdown would render as
something else. It also keeps the byte-for-byte check honest: the file holds the docstring
itself, so a diff is a docstring diff and never a rendering-rule diff."
  (if (null documentation)
      ""
      (let ((lines (uiop:split-string documentation :separator '(#\Newline))))
        (dolist (line lines)
          (when (and (>= (length line) 3) (string= "```" (subseq line 0 3)))
            (error "A docstring contains a line starting with a code fence, which would break ~
out of the reference's own fence: ~S. Rewrite the docstring." line)))
        (format nil "```text~%~A~%```~%" documentation))))

(defun render-entry (entry stream)
  "Write ENTRY to STREAM as one Markdown section."
  (let ((name (string-downcase (symbol-name (entry-symbol entry)))))
    (format stream "<a id=\"~A\"></a>~%~%## `~A:~A`~%~%"
            (entry-anchor (entry-qualifier entry) (entry-symbol entry))
            (entry-qualifier entry) name)
    (format stream "- **Kind** ~(~A~)~%" (substitute #\Space #\- (string (entry-kind entry))))
    ;; A Signature line is decided by KIND, not by whether LAMBDA-LIST happens to be non-NIL --
    ;; a zero-argument function has a NIL lambda list too, and still needs `(name)' here rather
    ;; than no line at all.
    (when (member (entry-kind entry) '(:function :generic-function :macro))
      (format stream "- **Signature** `~A`~%"
              (render-lambda-list (cons (entry-symbol entry) (entry-lambda-list entry)))))
    (when (entry-superclasses entry)
      (format stream "- **Superclasses** ~{`~(~A~)`~^, ~}~%"
              (mapcar #'symbol-name (entry-superclasses entry))))
    (format stream "- **Exported from** ~{`~A`~^, ~}~%~%" (entry-exported-from entry))
    (let ((target (entry-points-at entry)))
      ;; Both branches below name the pointed-at TYPE using ENTRY's own QUALIFIER, i.e. they
      ;; assume the reader and its type are published under the same package. Measured true for
      ;; all 174 symbols this reference publishes today; if Task 5 ever assembles a reader whose
      ;; type is qualified differently, this line would print the wrong package and needs its
      ;; own signal rather than a silently wrong link.
      (cond ((and target (eq (first target) :slot))
             (format stream "Reader of `~A:~(~A~)`'s `~(~A~)` slot. See `~A:~(~A~)`.~%~%"
                     (entry-qualifier entry) (symbol-name (second target))
                     (symbol-name (third target))
                     (entry-qualifier entry) (symbol-name (second target))))
            ((and target (eq (first target) :constructor))
             (format stream "Constructor of the `~A:~(~A~)` structure. See `~A:~(~A~)`.~%~%"
                     (entry-qualifier entry) (symbol-name (second target))
                     (entry-qualifier entry) (symbol-name (second target))))
            (t (write-string (render-documentation (entry-documentation entry)) stream)
               (terpri stream))))
    (when (entry-slots entry)
      (format stream "### Slots~%~%")
      (dolist (slot (entry-slots entry))
        (format stream "#### `~(~A~)`~%~%"
                (symbol-name (cl-gbdt/src/docgen/introspect:slot-info-name slot)))
        (let ((readers (cl-gbdt/src/docgen/introspect:slot-info-readers slot)))
          (when readers
            (format stream "- **Readers** ~{`~(~A~)`~^, ~}~%~%" (mapcar #'symbol-name readers))))
        (write-string (render-documentation
                       (cl-gbdt/src/docgen/introspect:slot-info-documentation slot))
                      stream)
        (terpri stream)))
    (when (entry-methods entry)
      (format stream "### Methods~%~%")
      (dolist (method (entry-methods entry))
        (format stream "#### `~A`~%~%" (car method))
        (write-string (render-documentation (cdr method)) stream)
        (terpri stream)))))
