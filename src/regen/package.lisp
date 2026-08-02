;;;; package.lisp --- Package definition for the binding emitter.
;;;;
;;;; This package is loaded only when a developer regenerates the bindings. It
;;;; never appears in the normal build's dependency graph.

(defpackage #:cl-gbdt.regen
  (:use #:cl)
  (:local-nicknames (#:jzon #:com.inuoe.jzon))
  (:export #:+typedef-map+
           #:+builtin-map+
           #:unmapped-type
           #:unmapped-type-tag
           #:unmapped-type-context
           #:cffi-type
           #:lisp-name
           #:emit-bindings))
