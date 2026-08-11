;;;; all.lisp --- The XGBoost backend's public face.
;;;;
;;;; Reassembles `cl-gbdt/src/xgboost/native' (Layer 1: library discovery, the error
;;;; wrapper, every %-function) and `cl-gbdt/src/xgboost/classes' (Layer 1: the backend's CLOS
;;;; types, the `initialize-backend'/`shutdown-backend' pair that opens and closes the shared
;;;; library, and `slice-model') into one package. This is what `cl-gbdt/xgboost' (see
;;;; cl-gbdt.asd) depends on, following the same shape `src/all.lisp' and
;;;; `src/regen/all.lisp' use for the same reason.
;;;;
;;;; Layer 1 and nothing else. `cl-gbdt/src/xgboost/protocol' (Layer 2: the thirteen protocol
;;;; methods) was named here until the split and is not any more, which is what makes
;;;; `cl-gbdt/xgboost' a system a caller can load without pulling in the unified API at all --
;;;; `(ql:quickload :cl-gbdt/xgboost)' now leaves the `cl-gbdt' package undefined.
;;;; `src/xgboost/unified.lisp' is the aggregation that names both, and
;;;; `cl-gbdt/xgboost/unified' the system a caller who wants `cl-gbdt:train' loads.
;;;;
;;;; Deliberately does not reexport `cl-gbdt/src/xgboost/c-api', the raw CFFI bindings --
;;;; see policy sections 3 and 11. That package's own docstring already says nothing
;;;; outside the backend system should call it directly; reexporting it here would make
;;;; every one of those raw C symbols part of this system's public surface instead.

(uiop:define-package #:cl-gbdt/src/xgboost/all
  (:use-reexport #:cl-gbdt/src/xgboost/native
                 #:cl-gbdt/src/xgboost/classes))

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
;;; Four symbols, each pulled explicitly by name. `xgboost-backend' is the CLOS class a caller
;;; can specialize methods on or check with `typep', and `slice-model' the capability work's
;;; Layer 1 addition, which lives in `classes' rather than in `native' only because it builds a
;;; booster handle and so must name the concrete `xgboost-booster' class (see that file's Model
;;; slicing section); `evaluate-one-iteration'
;;; (docs/superpowers/specs/2026-08-06-evaluation-api-design.md) and `booster-boosted-rounds',
;;; the round count `slice-model''s interval is expressed against, come from `native'. All four
;;; are policy section 3's Layer 1, mirroring `cl-gbdt/lightgbm''s identical shape. Nothing
;;; else from `native' is published here: none of its remaining exports is a reviewed,
;;; Lisp-level XGBoost-specific operation. A future Phase 2 addition follows the same shape --
;;; named explicitly in both an `:import-from' and the `:export' clause below, never picked up
;;; by widening either into a blanket re-export of a whole package; see this comment block's
;;; own earlier paragraph for why that stays forbidden.
;;;
;;; `xgboost-backend' reached this surface by way of `:use-reexport #:cl-gbdt/src/xgboost/protocol'
;;; until the split, that package having re-exported the one symbol and nothing else. `classes'
;;; is where the class now lives, but it is imported from by name rather than `:use-reexport'ed
;;; in protocol's place, for the reason that clause was already written this way for
;;; `slice-model': it also exports `xgboost-dataset', `xgboost-booster' and
;;; `%xgboost-foreign-library', none of them reviewed public API, and re-exporting it whole
;;; would put all three on this surface by accident -- the same hazard the
;;; `cl-gbdt/src/xgboost/all' paragraph above describes for `native'. The first form's
;;; `:use-reexport' of `classes' is what covers this form's dependency on it, per the closing
;;; comment below.
;;;
;;; Every one of the four is named in `:export' below because that is now the only thing
;;; publishing them: they arrive through `:import-from', which imports without exporting. The
;;; clause carries a second job as well -- it is the reviewed public surface, and
;;; `tools/ci/check-float-traps.lisp' reads exactly this `:export' to decide which `defun's
;;; are entry points reached without a `defmethod' to inherit a float-trap mask from, so a
;;; public `defun' that appeared here only by way of `:use-reexport' would slip that check
;;; silently. That scan no longer reaches `slice-model' itself, though: its
;;; +BACKEND-FILE-PATTERNS+ globs `src/*/native.lisp' and `src/*/protocol.lisp' only, so a
;;; public `defun' in `classes.lisp' is named here but matched nowhere until that list grows
;;; a `src/*/classes.lisp' entry. A future Layer 1 addition is listed here for the same
;;; reasons; a CLOS class such as `xgboost-backend' needs no entry, having no body to mask.
;;;
;;; And, as ever: never `#:cl-gbdt/src/xgboost/c-api', the raw CFFI bindings -- see the
;;; comment above `cl-gbdt/src/xgboost/all'.
(uiop:define-package #:cl-gbdt/xgboost
  (:import-from #:cl-gbdt/src/xgboost/classes
                #:xgboost-backend
                #:slice-model)
  (:import-from #:cl-gbdt/src/xgboost/native
                #:evaluate-one-iteration
                #:booster-boosted-rounds)
  (:export #:xgboost-backend
           #:evaluate-one-iteration
           #:booster-boosted-rounds
           #:slice-model))

;;; This second form's dependencies on `classes' and `native' are already covered by the first
;;; form's own `:use-reexport' above -- package-inferred-system infers a file's dependencies
;;; from the file's *first* defpackage form only (see `src/all.lisp''s identical comment), so a
;;; brand-new dependency this form alone needs, one the first form does not already list,
;;; belongs there, not here. That now rules `protocol' out for this form as well: the first
;;; form no longer names it, and naming it here alone would give this file a Layer 2 edge
;;; again, undoing the split.
