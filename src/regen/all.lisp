;;;; all.lisp --- The emitter's public face.
;;;;
;;;; This system is loaded only when a developer regenerates the bindings. It never
;;;; appears in the normal build's dependency graph.

(uiop:define-package #:cl-gbdt/src/regen/all
  (:use-reexport #:cl-gbdt/src/regen/types
                 #:cl-gbdt/src/regen/emit))
