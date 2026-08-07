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
;;; This re-exports `cl-gbdt/src/xgboost/protocol' in full -- `xgboost-backend', the CLOS class
;;; a caller can specialize methods on or check with `typep', and `slice-model', the capability
;;; work's Layer 1 addition, which lives in `protocol' rather than `native' only because it
;;; builds a booster handle and so must name the concrete `xgboost-booster' class (see that
;;; file's Model slicing section) -- plus two named symbols pulled explicitly from `native':
;;; `evaluate-one-iteration' (docs/superpowers/specs/2026-08-06-evaluation-api-design.md) and
;;; `booster-boosted-rounds', the round count `slice-model''s interval is expressed against.
;;; All three are policy section 3's Layer 1, mirroring `cl-gbdt/lightgbm''s
;;; `booster-eval-names'/`booster-eval' pair. Nothing else from `native' is published here:
;;; none of its remaining exports is a reviewed, Lisp-level XGBoost-specific operation. A
;;; future Phase 2 addition follows the same shape -- named explicitly in both the
;;; `:import-from' and the `:export' clause below, never picked up by widening either into a
;;; blanket re-export of `native' as a whole; see this comment block's own earlier paragraph
;;; for why that stays forbidden.
;;;
;;; `slice-model' is named in `:export' below even though the `:use-reexport' above already
;;; publishes it, and that is deliberate rather than redundant. This clause is the reviewed
;;; public surface: `tools/ci/check-float-traps.lisp' reads exactly this `:export' to decide
;;; which `defun's are entry points reached without a `defmethod' to inherit a float-trap mask
;;; from, so a public `defun' that appeared here only by way of `:use-reexport' would slip that
;;; check silently. A future Layer 1 addition in `protocol' is listed here for the same reason;
;;; a CLOS class such as `xgboost-backend' needs no entry, having no body to mask.
;;;
;;; And, as ever: never `#:cl-gbdt/src/xgboost/c-api', the raw CFFI bindings -- see the
;;; comment above `cl-gbdt/src/xgboost/all'.
(uiop:define-package #:cl-gbdt/xgboost
  (:use-reexport #:cl-gbdt/src/xgboost/protocol)
  (:import-from #:cl-gbdt/src/xgboost/native
                #:evaluate-one-iteration
                #:booster-boosted-rounds)
  (:export #:evaluate-one-iteration
           #:booster-boosted-rounds
           #:slice-model))

;;; This second form's dependency on `protocol' is already covered by the first form's own
;;; `:use-reexport' above -- package-inferred-system infers a file's dependencies from the
;;; file's *first* defpackage form only (see `src/all.lisp''s identical comment), so a
;;; brand-new dependency this form alone needs, one the first form does not already list,
;;; belongs there, not here.
