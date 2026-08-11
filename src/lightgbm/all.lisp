;;;; all.lisp --- The LightGBM backend's public face.
;;;;
;;;; Reassembles `cl-gbdt/src/lightgbm/native' (Layer 1: library discovery, the error
;;;; wrapper, every %-function) and `cl-gbdt/src/lightgbm/classes' (Layer 1: the backend's
;;;; CLOS types and the `initialize-backend'/`shutdown-backend' pair that opens and closes
;;;; the shared library) into one package. This is what `cl-gbdt/lightgbm' (see cl-gbdt.asd)
;;;; depends on, following the same shape `src/xgboost/all.lisp' uses for the same reason.
;;;;
;;;; Layer 1 and nothing else. `cl-gbdt/src/lightgbm/protocol' (Layer 2: the thirteen
;;;; protocol methods) was named here until the split and is not any more, which is what
;;;; makes `cl-gbdt/lightgbm' a system a caller can load without pulling in the unified API
;;;; at all -- `(ql:quickload :cl-gbdt/lightgbm)' now leaves the `cl-gbdt' package undefined.
;;;; `src/lightgbm/unified.lisp' is the aggregation that names both, and
;;;; `cl-gbdt/lightgbm/unified' the system a caller who wants `cl-gbdt:train' loads.
;;;;
;;;; Deliberately does not reexport `cl-gbdt/src/lightgbm/c-api', the raw CFFI bindings --
;;;; see policy sections 3 and 11. That package's own docstring already says nothing
;;;; outside the backend system should call it directly; reexporting it here would make
;;;; every one of those raw C symbols part of this system's public surface instead.

(uiop:define-package #:cl-gbdt/src/lightgbm/all
  (:use-reexport #:cl-gbdt/src/lightgbm/native
                 #:cl-gbdt/src/lightgbm/classes))

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
;;; Three symbols, each pulled explicitly by name. `lightgbm-backend' from `classes' is the
;;; CLOS class a caller can specialize methods on or check with `typep'; `booster-eval-names'
;;; and `booster-eval' from `native' are Phase 2's first LightGBM-specific safe API
;;; (docs/superpowers/specs/2026-08-06-evaluation-api-design.md, policy section 3's Layer 1).
;;; This published `lightgbm-backend' by way of `:use-reexport #:cl-gbdt/src/lightgbm/protocol'
;;; until the split, that package having re-exported the one symbol and nothing else. `classes'
;;; is where the class now lives, but it is imported from by name rather than `:use-reexport'ed
;;; in protocol's place: it also exports `lightgbm-dataset', `lightgbm-booster' and
;;; `%lightgbm-foreign-library', none of them reviewed public API, and re-exporting it whole
;;; would put all three on this surface by accident -- the same hazard the
;;; `cl-gbdt/src/lightgbm/all' paragraph above describes for `native', and the shape
;;; `cl-gbdt/xgboost' already used for `slice-model'. Nothing else from `native' is published
;;; here either: none of its remaining exports is a reviewed, Lisp-level LightGBM-specific
;;; operation. A future Phase 2 addition follows the same shape -- named explicitly in both an
;;; `:import-from' and the `:export' clause below, never picked up by widening either into a
;;; blanket re-export of a whole package; see this comment block's own earlier paragraph for
;;; why that stays forbidden.
;;;
;;; The `:export' clause names all three because `:import-from' imports without exporting, so
;;; it is now the only thing publishing any of them. It carries a second job as well --
;;; `tools/ci/check-float-traps.lisp' reads exactly this `:export' to decide which `defun's are
;;; entry points reached without a `defmethod' to inherit a float-trap mask from, so a public
;;; `defun' that appeared here only by way of `:use-reexport' would slip that check silently.
;;;
;;; And, as ever: never `#:cl-gbdt/src/lightgbm/c-api', the raw CFFI bindings -- see the
;;; comment above `cl-gbdt/src/lightgbm/all'.
(uiop:define-package #:cl-gbdt/lightgbm
  (:import-from #:cl-gbdt/src/lightgbm/classes
                #:lightgbm-backend)
  (:import-from #:cl-gbdt/src/lightgbm/native
                #:booster-eval-names
                #:booster-eval)
  (:export #:lightgbm-backend
           #:booster-eval-names
           #:booster-eval))

;;; This second form's dependencies on `classes' and `native' are already covered by the first
;;; form's own `:use-reexport' above -- package-inferred-system infers a file's dependencies
;;; from the file's *first* defpackage form only (see `src/all.lisp''s identical comment), so a
;;; brand-new dependency this form alone needs, one the first form does not already list,
;;; belongs there, not here. That now rules `protocol' out for this form as well: the first
;;; form no longer names it, and naming it here alone would give this file a Layer 2 edge
;;; again, undoing the split.
