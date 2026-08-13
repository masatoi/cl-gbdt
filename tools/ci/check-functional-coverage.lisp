;;;; check-functional-coverage.lisp --- Every published symbol has a position against the
;;;; functional suite.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-functional-coverage.lisp
;;;;
;;;; S4-2 of the Layer 1 standalone-library programme. The programme's fifth binding decision
;;;; requires a mechanically-checked functional test for every published symbol; this is the
;;;; mechanism. A symbol is COVERED when the functional suite's own sources name it, which this
;;;; script derives -- never written down, the same rule that keeps `wrapped' out of
;;;; ffi-spec/BINDING-COVERAGE.md. What cannot be derived, namely WHY a symbol has no functional
;;;; test, is written down in docs/FUNCTIONAL-COVERAGE.md, and this script fails the build on a
;;;; published symbol that is neither covered nor classified.
;;;;
;;;; WHAT THIS DOES NOT GUARANTEE. That the suite's source NAMES a symbol is not that the
;;;; symbol's contract is exercised: a condition named in a `handler-case' clause counts, and so
;;;; does a function called once in a fixture nothing asserts on. This check gives every symbol a
;;;; recorded position. It does not measure how well anything is tested, and a reader who takes a
;;;; green run for the latter has been misled -- which is why docs/FUNCTIONAL-COVERAGE.md says so
;;;; in its own header too.

(require :asdf)

(asdf:load-system "cl-gbdt/lightgbm/unified")
(asdf:load-system "cl-gbdt/xgboost/unified")
(asdf:load-system "cl-gbdt/docgen")
;;; Loading the functional test system is what makes its packages -- and their :local-nicknames,
;;; which four of its files use -- exist, so the standard reader can resolve `xgb::foo' and
;;; `support:make-separable-dataset' below. Re-deriving those nicknames from the :local-nicknames
;;; clauses would give the package system's own fact a second home to go stale in. It needs
;;; neither shared library: nothing here calls `open-backend', and tools/ci/check-leaf-systems.lisp
;;; already loads every one of these files in lint.yml, which never fetches them.
(asdf:load-system "cl-gbdt/tests/functional")

(defun die (format-control &rest arguments)
  "Print a failure to `*error-output*' and exit non-zero."
  (format *error-output* "~&check-functional-coverage: ~?~%" format-control arguments)
  (uiop:quit 1))

(defun published-list ()
  "Return the published surface, as S4-1's emitter derives it.

The surface has one derivation in this repository -- `cl-gbdt/docgen''s -- and both this check
and docs/API-REFERENCE.md read it from there. A second computation of `what is published' is
exactly the second home this programme keeps refusing to create."
  (cl-gbdt/src/docgen/introspect:published-symbols
   cl-gbdt/src/docgen/emit:+public-packages+))

(defun functional-test-files ()
  "Return the functional suite's source files, sorted, so reporting is deterministic."
  (sort (directory (merge-pathnames "tests/functional/*.lisp" (uiop:getcwd)))
        #'string< :key #'namestring))

(defun package-definition-form-p (form)
  "True when FORM is a `defpackage' or `uiop:define-package'."
  (and (consp form)
       (symbolp (first form))
       (member (symbol-name (first form)) '("DEFPACKAGE" "DEFINE-PACKAGE") :test #'string=)))

(defun functional-references ()
  "Return a hash table whose keys are the symbols the functional suite's sources name.

Reading with the standard reader, rather than searching text, is what makes a name inside a
comment not count. Excluding each file's own package form is what makes a name that appears only
in an `:import-from' clause not count: importing a symbol is not testing it. The whole file
otherwise counts, not only its `deftest' forms -- these files define fixtures and helpers as
top-level forms, and `make-csr-matrix' is reached only that way."
  (let ((referenced (make-hash-table)))
    (labels ((walk (form)
               (cond ((symbolp form) (setf (gethash form referenced) t))
                     ((consp form) (walk (car form)) (walk (cdr form))))))
      (dolist (path (functional-test-files) referenced)
        (with-open-file (in path)
          (let ((*package* (find-package "CL-USER"))
                (*read-eval* nil))
            (handler-case
                (loop for form = (read in nil :eof)
                      until (eq form :eof)
                      do (when (and (consp form) (eq (first form) 'cl:in-package))
                           (let ((package (find-package (string (second form)))))
                             (unless package
                               (die "~A names package ~A, which does not exist. Is ~
cl-gbdt/tests/functional loaded?" path (second form)))
                             (setf *package* package)))
                         (unless (package-definition-form-p form)
                           (walk form)))
              (error (condition)
                (die "cannot read ~A: ~A" path condition)))))))))
