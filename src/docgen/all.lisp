;;;; all.lisp --- The docgen emitter's public face.
;;;;
;;;; Mirrors src/regen/all.lisp: one package reassembling the files that make up the emitter,
;;;; which the `cl-gbdt/docgen' system in cl-gbdt.asd depends on. Development only.

(uiop:define-package #:cl-gbdt/src/docgen/all
  (:use-reexport #:cl-gbdt/src/docgen/introspect
                 #:cl-gbdt/src/docgen/render))
