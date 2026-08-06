;;;; all.lisp --- The XGBoost backend's public face.
;;;;
;;;; Reassembles `cl-gbdt/src/xgboost/native' (Layer 1: library discovery, the error
;;;; wrapper, every %-function) and `cl-gbdt/src/xgboost/protocol' (Layer 2: the classes
;;;; and the fifteen protocol methods) into one package. This is what `cl-gbdt/xgboost'
;;;; (see cl-gbdt.asd) depends on, following the same shape `src/all.lisp' and
;;;; `src/regen/all.lisp' use for the same reason.
;;;;
;;;; Deliberately does not reexport `cl-gbdt/src/xgboost/c-api', the raw CFFI bindings --
;;;; see policy sections 3 and 11. That package's own docstring already says nothing
;;;; outside the backend system should call it directly; reexporting it here would make
;;;; every one of those raw C symbols part of this system's public surface instead.

(uiop:define-package #:cl-gbdt/src/xgboost/all
  (:use-reexport #:cl-gbdt/src/xgboost/native
                 #:cl-gbdt/src/xgboost/protocol))

;;; ---------------------------------------------------------------------------
;;; The public package
;;;
;;; Policy section 3 names `cl-gbdt/xgboost' as the public package for XGBoost-specific
;;; API, and section 11 requires backend-specific public symbols come only from it. Until
;;; this form, it existed as an ASDF system name only -- `cl-gbdt/src/xgboost/all' above is
;;; an internal aggregation, not what section 11 means by "public". This is the thing that
;;; name now resolves to, following `src/all.lisp''s shape: two `uiop:define-package' forms
;;; in one file, the second re-exporting a curated subset of the first, rather than a
;;; separate file -- no reason to diverge from that precedent appeared, matching
;;; `cl-gbdt/src/lightgbm/all''s identical choice for the same reason.
;;;
;;; Deliberately NOT `:use-reexport #:cl-gbdt/src/xgboost/all' wholesale. That package
;;; re-exports every symbol `cl-gbdt/src/xgboost/native' exports -- every internal
;;; `%'-function (never designed as a contract) plus `check-xgb' and the library-discovery
;;; parameters (`*library-env-var*', `*vendor-library-directory*', `*vendor-library-pattern*',
;;; `*default-library-name*', `*required-symbols*'), which are implementation details of
;;; `resolve-and-load-library' and `initialize-backend', not operations a caller invokes
;;; directly. Re-exporting `all' wholesale would put every one of those on this system's
;;; public surface by accident -- exactly what this task's brief warns an accidental export
;;; becomes: a compatibility obligation.
;;;
;;; This re-exports `cl-gbdt/src/xgboost/protocol' in full -- today just `xgboost-backend',
;;; the CLOS class a caller can specialize methods on or check with `typep' -- plus one named
;;; symbol pulled explicitly from `native': `booster-eval', Task 3's XGBoost-specific Layer 1
;;; addition (docs/superpowers/specs/2026-08-06-evaluation-api-design.md, policy section 3's
;;; Layer 1), mirroring `cl-gbdt/lightgbm''s identical `booster-eval-names'/`booster-eval'
;;; pair from Task 2. Nothing else from `native' is published here: none of its remaining
;;; exports is a reviewed, Lisp-level XGBoost-specific operation. A future Phase 2 addition
;;; follows the same shape -- named explicitly in both the `:import-from' and the `:export'
;;; clause below, never picked up by widening either into a blanket re-export of `native' as
;;; a whole; see this comment block's own earlier paragraph for why that stays forbidden.
;;;
;;; And, as ever: never `#:cl-gbdt/src/xgboost/c-api', the raw CFFI bindings -- see the
;;; comment above `cl-gbdt/src/xgboost/all'.
(uiop:define-package #:cl-gbdt/xgboost
  (:use-reexport #:cl-gbdt/src/xgboost/protocol)
  (:import-from #:cl-gbdt/src/xgboost/native
                #:booster-eval)
  (:export #:booster-eval))

;;; This second form's dependency on `protocol' is already covered by the first form's own
;;; `:use-reexport' above -- package-inferred-system infers a file's dependencies from the
;;; file's *first* defpackage form only (see `src/all.lisp''s identical comment), so a
;;; brand-new dependency this form alone needs, one the first form does not already list,
;;; belongs there, not here.
