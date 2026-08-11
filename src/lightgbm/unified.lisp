;;;; unified.lisp --- LightGBM plus the unified API's methods.
;;;;
;;;; `cl-gbdt/lightgbm' is Layer 1 alone: it opens the library, and it publishes this backend's
;;;; own operations. This system is that plus `protocol.lisp', the file that implements
;;;; `cl-gbdt''s thirteen portable generic functions for LightGBM. A caller who wants
;;;; `cl-gbdt:train' loads this; a caller who wants LightGBM loads the other one.
;;;;
;;;; The zero-symbol `cl-gbdt/src/all' clause is a dependency declaration, the idiom CLAUDE.md
;;;; describes. It is what makes the `cl-gbdt' package exist for a caller who loaded only this
;;;; system -- before the split, `(ql:quickload :cl-gbdt/lightgbm)' left that package undefined
;;;; and a caller had to name an internal package to catch a condition. Nothing in this file
;;;; refers to it by name, so the clause names no symbols; `cl-gbdt/src/lightgbm/protocol' does
;;;; not carry the edge on its own, importing `cl-gbdt/src/protocol' and the other internal
;;;; packages it specializes on rather than the `cl-gbdt' aggregation those are published
;;;; through.
;;;;
;;;; `cl-gbdt/src/lightgbm/all' and `cl-gbdt/src/lightgbm/protocol' both export
;;;; `lightgbm-backend' -- the one symbol `cl-gbdt/src/lightgbm/classes' defines and both
;;;; re-export -- so `:use-reexport'ing the pair names it twice with no conflict: it is the
;;;; same symbol reached two ways, not two symbols of one name.
;;;;
;;;; Deliberately does not reexport `cl-gbdt/src/lightgbm/c-api', the raw CFFI bindings, for
;;;; the reason `src/lightgbm/all.lisp''s header gives.

(uiop:define-package #:cl-gbdt/src/lightgbm/unified
  (:use-reexport #:cl-gbdt/src/lightgbm/all
                 #:cl-gbdt/src/lightgbm/protocol)
  (:import-from #:cl-gbdt/src/all))
