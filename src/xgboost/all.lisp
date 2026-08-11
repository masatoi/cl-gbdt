;;;; all.lisp --- The XGBoost backend's public face.
;;;;
;;;; Reassembles `cl-gbdt/src/xgboost/native' (Layer 1: library discovery, the error
;;;; wrapper, every %-function), `cl-gbdt/src/xgboost/classes' (Layer 1: the backend's CLOS
;;;; types and the `initialize-backend'/`shutdown-backend' pair that opens and closes the
;;;; shared library) and `cl-gbdt/src/xgboost/api' (Layer 1: the finished operations built on
;;;; top of those two, `slice-model' among them) into one package. This is what
;;;; `cl-gbdt/xgboost' (see cl-gbdt.asd) depends on, following the same shape `src/all.lisp'
;;;; and `src/regen/all.lisp' use for the same reason.
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
                 #:cl-gbdt/src/xgboost/classes
                 #:cl-gbdt/src/xgboost/api))

;;; ---------------------------------------------------------------------------
;;; The public package
;;;
;;; Policy section 3 names `cl-gbdt/xgboost' as the public package for XGBoost-specific
;;; API, and section 11 requires backend-specific public symbols come only from it. Until
;;; this form, it existed as an ASDF system name only -- `cl-gbdt/src/xgboost/all' above is
;;; an internal aggregation, not what section 11 means by "public". This is the thing that
;;; name now resolves to, following `src/all.lisp''s shape: two `uiop:define-package' forms
;;; in one file rather than a separate file -- no reason to diverge from that precedent
;;; appeared, matching `cl-gbdt/src/lightgbm/all''s identical choice for the same reason.
;;; The second form is no longer a subset of the first, though: it began as a curated
;;; selection from it, and Task 7 added the shared-basis and condition symbols described
;;; below, which come from `cl-gbdt/src/backend', `cl-gbdt/src/handle' and
;;; `cl-gbdt/src/conditions' -- packages the first form does not name at all.
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
;;; Six symbols so far, each pulled explicitly by name. `xgboost-backend' from `classes' is the
;;; CLOS class a caller can specialize methods on or check with `typep'; `evaluate-one-iteration'
;;; (docs/superpowers/specs/2026-08-06-evaluation-api-design.md) and `booster-boosted-rounds',
;;; the round count `slice-model''s interval is expressed against, come from `native'.
;;; `slice-model' itself, the capability work's Layer 1 addition, and `create-dataset' and
;;; `free-dataset', the first finished operations to reach this surface, come from `api' -- the
;;; latter two being the procedure that used to sit inside `cl-gbdt/src/xgboost/protocol''s
;;; `make-dataset' and `free-dataset' methods, which now check their portable arguments and call
;;; these. `free-dataset' here is NOT `cl-gbdt''s generic of that name: it is a plain function
;;; and a different symbol, so a caller who has both packages in an image must name which one
;;; they mean, exactly as they already must for anything else two packages export under one
;;; name. `slice-model' was imported from `classes' until `api' existed; it moved because it is
;;; an operation over the booster class rather than part of the library's lifetime, and the
;;; symbol a caller reaches is unchanged by that move -- this clause is the only thing that had
;;; to notice.
;;;
;;; All six are policy section 3's Layer 1, mirroring `cl-gbdt/lightgbm''s identical shape.
;;; Nothing else from `native' is published here: none of its remaining exports is a reviewed,
;;; Lisp-level XGBoost-specific operation. `api' is `:import-from'ed rather than
;;; `:use-reexport'ed for the reason `classes' is: it also exports `%check-sparse-input' and
;;; `%creation-function-name', internal helpers that exist at that layer only because
;;; `protocol.lisp''s `make-dataset' and `predict' call them too, and re-exporting the package
;;; whole would put both on this public surface by accident. A future Phase 2 addition follows
;;; the same shape -- named explicitly in both an `:import-from' and the `:export' clause below,
;;; never picked up by widening either into a blanket re-export of a whole package; see this
;;; comment block's own earlier paragraph for why that stays forbidden.
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
;;; Task 7 adds the shared-basis symbols a caller who loads `cl-gbdt/xgboost' ALONE -- never
;;; `cl-gbdt' itself -- needs to open, question and close a backend, and to catch what it
;;; signals by name instead of reaching for `cl-gbdt/src/conditions:foreign-call-error', an
;;; internal package, directly. `:use-reexport #:cl-gbdt/src/conditions' brings in that whole
;;; package: every condition type and accessor it exports is already reviewed public API (see
;;; that file's own `:export' clause), so re-exporting it whole carries none of the
;;; "every %-function comes along for the ride" hazard the paragraphs above describe for
;;; `native'. `:import-from #:cl-gbdt/src/backend' and `:import-from #:cl-gbdt/src/handle' stay
;;; narrow, by name, matching `cl-gbdt/lightgbm''s identical choice for the identical reason:
;;; both packages also export symbols meant for someone IMPLEMENTING a backend, not someone
;;; using one -- `register-backend', `find-backend-class', `initialize-backend',
;;; `shutdown-backend', `probe-foreign-symbols' and `probe-capabilities' from `backend';
;;; `make-handle', `release-handle', `handle-pointer' and `handle-live-pointer' from `handle'
;;; -- and re-exporting either package whole would put every one of those on this surface by
;;; accident, the identical hazard already described above. `csr-matrix' and the rest of
;;; `cl-gbdt/src/data' still stay out, though the reason has changed: dataset construction is
;;; now at this layer, and `create-dataset' below does take a `csr-matrix', so a Layer 1 caller
;;; who wants the sparse path has to reach `cl-gbdt:make-csr-matrix' -- which means loading the
;;; unified core -- or name `cl-gbdt/src/data' directly. `cl-gbdt/lightgbm' does NOT leave it
;;; there: it publishes `make-csr-matrix', the `csr-matrix' type and its five readers (see its
;;; own comment at the same place), and it did so when its `predict' reached Layer 1 and half of
;;; two published contracts would otherwise have been unreachable from the package publishing
;;; them. This backend has only the one such contract so far, `create-dataset'; its `predict' is
;;; still at Layer 2. So the sparse surface is deferred to the task that moves that method down,
;;; which is where the sibling took the decision and where the same argument will apply here --
;;; not left open indefinitely, and not a side effect of moving `make-dataset''s procedure down.
;;; The dense path, which is every ordinary array, needs nothing from `cl-gbdt/src/data' at all.
;;;
;;; Every one of them is named in `:export' below because that is now the only thing
;;; publishing them: they arrive through `:import-from', which imports without exporting. The
;;; clause carries a second job as well -- it is the reviewed public surface, and
;;; `tools/ci/check-float-traps.lisp' reads exactly this `:export' to decide which `defun's
;;; are entry points reached without a `defmethod' to inherit a float-trap mask from, so a
;;; public `defun' that appeared here only by way of `:use-reexport' would slip that check
;;; silently. That scan does reach all three `api' functions: its +BACKEND-FILE-PATTERNS+ globs
;;; `src/*/api.lisp' and `src/*/classes.lisp' alongside `src/*/native.lisp' and
;;; `src/*/protocol.lisp', so a public `defun' in either is both named here and matched there --
;;; the scan reports `api.lisp' as "0 defmethods, 3 public defuns, 0 unmasked" and `classes.lisp'
;;; as "2 defmethods, 0 public defuns, 0 unmasked" now that `slice-model' has moved. A future
;;; Layer 1 addition is listed here for the same reasons; a CLOS class such as `xgboost-backend'
;;; needs no entry, having no body to mask.
;;; None of Task 7's additions are `defun's in THIS file either -- `open-backend' and
;;; `close-backend' are `defun's in `src/backend.lisp', outside the check's own file-pattern
;;; glob, and the condition accessors `:use-reexport'ed from `cl-gbdt/src/conditions' are
;;; reader functions `define-condition' generates, not `defun's the check would count.
;;;
;;; And, as ever: never `#:cl-gbdt/src/xgboost/c-api', the raw CFFI bindings -- see the
;;; comment above `cl-gbdt/src/xgboost/all'.
(uiop:define-package #:cl-gbdt/xgboost
  (:use-reexport #:cl-gbdt/src/conditions)
  (:import-from #:cl-gbdt/src/backend
                #:*known-capabilities*
                #:backend-capabilities
                #:backend-info
                #:backend-library-path
                #:backend-name
                #:backend-open-p
                #:backend-supports-p
                #:backend-version
                #:close-backend
                #:open-backend)
  (:import-from #:cl-gbdt/src/handle
                #:booster
                #:dataset
                #:handle-backend
                #:handle-released-p)
  (:import-from #:cl-gbdt/src/xgboost/classes
                #:xgboost-backend)
  (:import-from #:cl-gbdt/src/xgboost/native
                #:evaluate-one-iteration
                #:booster-boosted-rounds)
  (:import-from #:cl-gbdt/src/xgboost/api
                #:create-dataset
                #:free-dataset
                #:slice-model)
  (:export #:xgboost-backend
           #:evaluate-one-iteration
           #:booster-boosted-rounds
           #:slice-model
           #:create-dataset
           #:free-dataset
           #:*known-capabilities*
           #:backend-capabilities
           #:backend-info
           #:backend-library-path
           #:backend-name
           #:backend-open-p
           #:backend-supports-p
           #:backend-version
           #:close-backend
           #:open-backend
           #:booster
           #:dataset
           #:handle-backend
           #:handle-released-p))

;;; This second form's dependencies on `classes', `native' and `api' are already covered by the
;;; first form's own `:use-reexport' above -- package-inferred-system infers a file's dependencies
;;; from the file's *first* defpackage form only (see `src/all.lisp''s identical comment), so a
;;; brand-new dependency this form alone needs, one the first form does not already list,
;;; belongs there, not here. That now rules `protocol' out for this form as well: the first
;;; form no longer names it, and naming it here alone would give this file a Layer 2 edge
;;; again, undoing the split. Task 7's three new clauses need no such addition either: `classes'
;;; itself already `:import-from's `cl-gbdt/src/backend', `cl-gbdt/src/handle' and
;;; `cl-gbdt/src/conditions' (see its own package form), so loading `cl-gbdt/src/xgboost/all'
;;; -- which the first form's `:use-reexport' of `classes' already requires -- transitively
;;; loads all three ASDF systems before this second form is ever read.
