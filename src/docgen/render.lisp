;;;; render.lisp --- Turn what src/docgen/introspect.lisp gathered into Markdown text.
;;;;
;;;; Development only. This file knows nothing about how its input was obtained, which is what
;;;; lets tests/docgen.lisp exercise it on synthetic input rather than on the library's own
;;;; published surface.

(uiop:define-package #:cl-gbdt/src/docgen/render
  (:use #:cl)
  (:export #:render-lambda-list
           #:render-method-signature))

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
