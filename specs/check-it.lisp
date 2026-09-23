;;;; check-it.lisp --- The specification bundle with a generator backend installed.
;;;;
;;;; `cl-gbdt/specs' registers definitions with cl-spec core alone. Running them --
;;;; `check-function', `run-property', `spec-check' -- also needs a generator backend, and this
;;;; file's only job is to declare the dependency on cl-spec's check-it backend, so that
;;;; loading `cl-gbdt/specs/check-it' is the one step that prepares a session to check.

(uiop:define-package #:cl-gbdt/specs/check-it
  (:use #:cl)
  (:import-from #:cl-gbdt/specs/all)
  (:import-from #:cl-spec/src/backends/check-it))

(in-package #:cl-gbdt/specs/check-it)
