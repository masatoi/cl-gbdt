;;;; all.lisp --- The public face of cl-gbdt.
;;;;
;;;; Under package-inferred-system each source file owns its own package. This file
;;;; reassembles them into CL-GBDT, which is what callers use and what the library
;;;; has always exported. Adding a symbol to any of the four packages below exports
;;;; it from CL-GBDT automatically; nothing is listed twice.
;;;;
;;;; The list of four packages below is hand-maintained: package-inferred-system
;;;; does not enumerate src/ on its own. Add a src/foo.lisp and forget to list its
;;;; package here, and nothing ever refers to cl-gbdt/src/foo, so ASDF has no edge
;;;; to discover it by -- (ql:quickload :cl-gbdt) never even compiles the file, let
;;;; alone exports its symbols. There is no separate failing test for this one: the
;;;; silent failure is the file never being loaded at all, not an assertion to trip.

(uiop:define-package #:cl-gbdt/src/all
  (:use-reexport #:cl-gbdt/src/conditions
                 #:cl-gbdt/src/data
                 #:cl-gbdt/src/backend
                 #:cl-gbdt/src/protocol))

;;; package-inferred-system infers a file's dependencies from the file's *first*
;;; defpackage form only. This second form's :use-reexport clause is therefore
;;; inert for dependency purposes -- a new dependency belongs in the
;;; #:cl-gbdt/src/all form above, not here.
(uiop:define-package #:cl-gbdt
  (:use-reexport #:cl-gbdt/src/all))
