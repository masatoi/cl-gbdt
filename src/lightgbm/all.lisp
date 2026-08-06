;;;; all.lisp --- The LightGBM backend's public face.
;;;;
;;;; Reassembles `cl-gbdt/src/lightgbm/native' (Layer 1: library discovery, the error
;;;; wrapper, every %-function) and `cl-gbdt/src/lightgbm/protocol' (Layer 2: the classes
;;;; and the fourteen protocol methods) into one package. This is what `cl-gbdt/lightgbm'
;;;; (see cl-gbdt.asd) depends on, following the same shape `src/xgboost/all.lisp' uses
;;;; for the same reason.
;;;;
;;;; Deliberately does not reexport `cl-gbdt/src/lightgbm/c-api', the raw CFFI bindings --
;;;; see policy sections 3 and 11. That package's own docstring already says nothing
;;;; outside the backend system should call it directly; reexporting it here would make
;;;; every one of those raw C symbols part of this system's public surface instead.

(uiop:define-package #:cl-gbdt/src/lightgbm/all
  (:use-reexport #:cl-gbdt/src/lightgbm/native
                 #:cl-gbdt/src/lightgbm/protocol))

;;; ---------------------------------------------------------------------------
;;; The public package
;;;
;;; Policy section 3 names `cl-gbdt/lightgbm' as the public package for LightGBM-specific
;;; API, and section 11 requires backend-specific public symbols come only from it. Until
;;; this form, it existed as an ASDF system name only -- `cl-gbdt/src/lightgbm/all' above
;;; is an internal aggregation, not what section 11 means by "public". This is the thing
;;; that name now resolves to, following `src/all.lisp''s shape: two `uiop:define-package'
;;; forms in one file, the second re-exporting a curated subset of the first, rather than a
;;; separate file -- no reason to diverge from that precedent appeared.
;;;
;;; Deliberately NOT `:use-reexport #:cl-gbdt/src/lightgbm/all' wholesale. That package
;;; re-exports every symbol `cl-gbdt/src/lightgbm/native' exports -- every internal
;;; `%'-function (never designed as a contract) plus `check-lgbm' and the library-discovery
;;; parameters (`*library-env-var*', `*vendor-library-directory*', `*vendor-library-pattern*',
;;; `*default-library-name*', `*required-symbols*'), which are implementation details of
;;; `resolve-and-load-library' and `initialize-backend', not operations a caller invokes
;;; directly. Re-exporting `all' wholesale would put every one of those on this system's
;;; public surface by accident -- exactly what this task's brief warns an accidental export
;;; becomes: a compatibility obligation.
;;;
;;; So this re-exports `cl-gbdt/src/lightgbm/protocol' only, which today exports exactly one
;;; symbol, `lightgbm-backend' -- the CLOS class a caller can specialize methods on or check
;;; with `typep'. Nothing from `native' is published here: none of its current exports is a
;;; reviewed, Lisp-level LightGBM-specific operation -- that is Phase 2's job
;;; (docs/superpowers/specs/2026-08-06-evaluation-api-design.md), which adds functions here
;;; built and documented as public contracts, named explicitly when they are, never picked up
;;; by widening this clause.
;;;
;;; And, as ever: never `#:cl-gbdt/src/lightgbm/c-api', the raw CFFI bindings -- see the
;;; comment above `cl-gbdt/src/lightgbm/all'.
(uiop:define-package #:cl-gbdt/lightgbm
  (:use-reexport #:cl-gbdt/src/lightgbm/protocol))

;;; This second form's dependency on `protocol' is already covered by the first form's own
;;; `:use-reexport' above -- package-inferred-system infers a file's dependencies from the
;;; file's *first* defpackage form only (see `src/all.lisp''s identical comment), so a
;;; brand-new dependency this form alone needs, one the first form does not already list,
;;; belongs there, not here.
