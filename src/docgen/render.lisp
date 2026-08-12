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

(defun render-lambda-list-element (element)
  "Return ELEMENT as reference text, descending into destructuring and default pairs."
  (if (consp element)
      (format nil "(~{~A~^ ~})"
              (mapcar #'render-lambda-list-element
                      (if (listp (cdr (last element)))
                          element
                          (append (butlast element) (list (car (last element))
                                                          (cdr (last element)))))))
      (render-atom element)))

(defun render-lambda-list (lambda-list)
  "Return LAMBDA-LIST as reference text: names only, lower case, defaults kept as pairs.

`sb-introspect:function-lambda-list' answers with the defining package's own internal symbols,
which are unreadable printed whole. Everything after `&aux' is dropped: it binds locals in the
body and is no part of what a caller passes."
  (let ((visible (loop for element in lambda-list
                       until (eq element '&aux)
                       collect element)))
    (format nil "(~{~A~^ ~})" (mapcar #'render-lambda-list-element visible))))

(defun render-specializer (specializer)
  "Return SPECIALIZER as reference text: a class name, or (eql OBJECT)."
  (if (typep specializer 'sb-mop:eql-specializer)
      (format nil "(eql ~A)" (render-atom (sb-mop:eql-specializer-object specializer)))
      (render-atom (class-name specializer))))

(defun render-method-signature (name method)
  "Return METHOD's signature as reference text, each required parameter with its specializer.

An unspecialized required parameter shows T rather than nothing: T is what it dispatches on,
and hiding it would make two methods that differ only there print identically."
  (let* ((lambda-list (sb-mop:method-lambda-list method))
         (specializers (sb-mop:method-specializers method))
         (required (loop for parameter in lambda-list
                         for specializer in specializers
                         collect (format nil "(~A ~A)"
                                         (render-atom parameter)
                                         (render-specializer specializer))))
         (rest (nthcdr (length specializers) lambda-list)))
    (format nil "(~A ~{~A~^ ~}~@[ ~A~])"
            (render-atom name)
            required
            (when rest
              (let ((text (render-lambda-list rest)))
                (subseq text 1 (1- (length text))))))))
