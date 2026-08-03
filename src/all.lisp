;;;; all.lisp --- The public face of cl-gbdt.
;;;;
;;;; Under package-inferred-system each source file owns its own package. This file
;;;; reassembles them into CL-GBDT, which is what callers use and what the library
;;;; has always exported. Adding a symbol to any of the four packages below exports
;;;; it from CL-GBDT automatically; nothing is listed twice.

(uiop:define-package #:cl-gbdt/src/all
  (:use-reexport #:cl-gbdt/src/conditions
                 #:cl-gbdt/src/data
                 #:cl-gbdt/src/backend
                 #:cl-gbdt/src/protocol))

(uiop:define-package #:cl-gbdt
  (:use-reexport #:cl-gbdt/src/all))
