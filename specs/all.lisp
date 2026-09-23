;;;; all.lisp --- The executable specification bundle, and the names in it.
;;;;
;;;; Loading this registers every definition under specs/ in `cl-spec:*registry*'. It opens no
;;;; shared library, instruments nothing and generates nothing: cl-spec core is enough to
;;;; register, and `cl-gbdt/specs/check-it' adds the backend that running needs.
;;;;
;;;; The definitions stay top-level forms in their own files. cl-spec's own bundle wraps its
;;;; definitions in a `register-specifications' `defun' instead; that hides every
;;;; `defspec-function' inside a function body, where structural editors cannot address it by
;;;; name, and keeping them top-level is what lets `lisp-edit-form' edit a contract with
;;;; form_type `defspec-function' and form_name the target, as CLAUDE.md asks.
;;;; `register-specifications' here re-`load's those files for a registry that lost them.
;;;; Reloading the system is not a substitute: whether `asdf:load-system' re-evaluates a
;;;; dependency depends on the ASDF plan, and where it does it also recompiles cl-spec and
;;;; cl-gbdt themselves. See docs/cl-spec-dogfooding.md, G7.
;;;;
;;;; `contract-names' and `property-names' are what tests/specs/checks.lisp runs. That file
;;;; also asserts the two lists match what the registry holds, so a definition added under
;;;; specs/ and left off them fails there rather than going unchecked.

(uiop:define-package #:cl-gbdt/specs/all
  (:use #:cl)
  ;; Bare: `register-specifications' names ASDF's functions package-qualified.
  (:import-from #:asdf)
  (:import-from #:cl-gbdt/specs/prediction-shape)
  (:import-from #:cl-gbdt/specs/parameters)
  (:import-from #:cl-gbdt/specs/objective)
  (:import-from #:cl-gbdt/specs/training-report)
  (:import-from #:cl-gbdt/specs/history)
  (:export #:contract-names
           #:property-names
           #:register-specifications))

(in-package #:cl-gbdt/specs/all)

(defun contract-names ()
  "Return the target of every Function Spec this bundle registers."
  (list 'cl-gbdt/src/config/prediction-shape:contrib-shape
        'cl-gbdt/src/parameters:normalize-parameters
        'cl-gbdt/src/config/objective:objective-single-float
        'cl-gbdt/src/training-report:make-training-series
        'cl-gbdt/src/training-report:make-training-report
        'cl-gbdt/src/training/history:training-report-from-history))

(defun property-names ()
  "Return the name of every Property this bundle registers."
  (list 'cl-gbdt/specs/parameters:normalize-parameters-keeps-order-and-renames-keys
        'cl-gbdt/specs/parameters:normalize-parameters-values-denote-themselves
        'cl-gbdt/specs/parameters:normalize-parameters-ignores-the-caller-s-printer
        'cl-gbdt/specs/objective:objective-parameters-ends-with-the-one-canonical-objective
        'cl-gbdt/specs/objective:objective-parameters-keeps-every-other-entry-in-order
        'cl-gbdt/specs/objective:objective-parameters-is-idempotent
        'cl-gbdt/specs/history:history-yields-one-series-per-pair-in-first-appearance-order
        'cl-gbdt/specs/history:history-series-values-are-the-pair-s-values-in-order
        'cl-gbdt/specs/history:history-series-name-is-the-dataset-s-name))

(defparameter *specification-systems*
  '("cl-gbdt/specs/values" "cl-gbdt/specs/prediction-shape" "cl-gbdt/specs/parameters"
    "cl-gbdt/specs/objective" "cl-gbdt/specs/training-report" "cl-gbdt/specs/history")
  "Every specs/ file that registers a definition, as its inferred system name, in dependency
order: `values' first, since the others name its specs.")

(defun register-specifications ()
  "Register every definition in this bundle again, in `cl-spec:*registry*' as currently bound.

For a registry that lost them -- after `cl-spec:clear-registry', or a freshly bound
`cl-spec:*registry*'. Each file in `*specification-systems*' is `load'ed from its source, so
exactly the specification forms are re-evaluated: not cl-spec, not cl-gbdt, and not by way of
an ASDF plan whose reach depends on what that image last compiled. Opens no shared library
and generates nothing, like loading the bundle. Returns `contract-names' and
`property-names' as two values."
  (dolist (name *specification-systems*)
    ;; An inferred system's own pathname is the project root; its one child is the file.
    (dolist (file (asdf:component-children (asdf:find-system name)))
      (load (asdf:component-pathname file))))
  (values (contract-names) (property-names)))
