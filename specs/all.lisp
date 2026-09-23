;;;; all.lisp --- The executable specification bundle, and the names in it.
;;;;
;;;; Loading this registers every definition under specs/ in `cl-spec:*registry*'. It opens no
;;;; shared library, instruments nothing and generates nothing: cl-spec core is enough to
;;;; register, and `cl-gbdt/specs/check-it' adds the backend that running needs.
;;;;
;;;; Re-registration is by reloading -- `(asdf:load-system "cl-gbdt/specs" :force t)' -- not by
;;;; a `register-specifications' function. cl-spec's own bundle wraps its definitions in one;
;;;; that hides every `defspec-function' inside a `defun', where structural editors cannot
;;;; address it by name. Keeping them top-level is what lets `lisp-edit-form' edit a contract
;;;; with form_type `defspec-function' and form_name the target, as CLAUDE.md asks. See
;;;; docs/cl-spec-dogfooding.md for the trade-off.
;;;;
;;;; `contract-names' and `property-names' are what tests/specs/checks.lisp runs. That file
;;;; also asserts the two lists match what the registry holds, so a definition added under
;;;; specs/ and left off them fails there rather than going unchecked.

(uiop:define-package #:cl-gbdt/specs/all
  (:use #:cl)
  (:import-from #:cl-gbdt/specs/prediction-shape)
  (:export #:contract-names
           #:property-names))

(in-package #:cl-gbdt/specs/all)

(defun contract-names ()
  "Return the target of every Function Spec this bundle registers."
  (list 'cl-gbdt/src/config/prediction-shape:contrib-shape))

(defun property-names ()
  "Return the name of every Property this bundle registers."
  (list))
