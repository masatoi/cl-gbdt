;;;; all.lisp --- The LightGBM backend's public face.
;;;;
;;;; Reassembles `cl-gbdt/src/lightgbm/native' (Layer 1: library discovery, the error
;;;; wrapper, every %-function), `cl-gbdt/src/lightgbm/classes' (Layer 1: the backend's
;;;; CLOS types and the `initialize-backend'/`shutdown-backend' pair that opens and closes
;;;; the shared library) and `cl-gbdt/src/lightgbm/api' (Layer 1: the finished operations
;;;; built on top of those two) into one package. This is what `cl-gbdt/lightgbm' (see
;;;; cl-gbdt.asd) depends on, following the same shape `src/xgboost/all.lisp' uses for the
;;;; same reason.
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
                 #:cl-gbdt/src/lightgbm/classes
                 #:cl-gbdt/src/lightgbm/api))

;;; ---------------------------------------------------------------------------
;;; The public package
;;;
;;; Policy section 3 names `cl-gbdt/lightgbm' as the public package for LightGBM-specific
;;; API, and section 11 requires backend-specific public symbols come only from it. Until
;;; this form, it existed as an ASDF system name only -- `cl-gbdt/src/lightgbm/all' above
;;; is an internal aggregation, not what section 11 means by "public". This is the thing
;;; that name now resolves to, following `src/all.lisp''s shape: two `uiop:define-package'
;;; forms in one file rather than a separate file -- no reason to diverge from that
;;; precedent appeared. The second form is no longer a subset of the first, though: it began
;;; as a curated selection from it, and Task 7 added the shared-basis and condition symbols
;;; described below, which come from `cl-gbdt/src/backend', `cl-gbdt/src/handle' and
;;; `cl-gbdt/src/conditions' -- packages the first form does not name at all.
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
;;; Eight symbols so far, each pulled explicitly by name. `lightgbm-backend' from `classes' is
;;; the CLOS class a caller can specialize methods on or check with `typep'; `booster-eval-names'
;;; and `booster-eval' from `native' are Phase 2's first LightGBM-specific safe API
;;; (docs/superpowers/specs/2026-08-06-evaluation-api-design.md, policy section 3's Layer 1);
;;; `create-dataset', `free-dataset', `create-booster', `update-one-iteration' and
;;; `free-booster' from `api' are the finished operations -- the procedure that used to sit
;;; inside `cl-gbdt/src/lightgbm/protocol''s methods of those names, which now check their
;;; portable arguments and call these. Together they are a whole training run at this layer:
;;; build a dataset, build a booster on it, advance it, free both. `create-booster' is the one
;;; with no caller inside this library -- `train' builds its own booster, for the reason its
;;; creation call records -- so it is published on the strength of its own contract rather than
;;; of a method that exercises it. `free-dataset', `free-booster' and `update-one-iteration'
;;; here are NOT `cl-gbdt''s generics of those names: they are plain functions and different
;;; symbols, so a caller who has both packages in an image must name which one they mean,
;;; exactly as they already must for anything else two packages export under one name.
;;;
;;; `api' is `:import-from'ed rather than `:use-reexport'ed for the reason `classes' is: it
;;; also exports `%check-sparse-input', an internal capability gate that exists at that layer
;;; only because `protocol.lisp''s `predict' calls it too, and re-exporting the package whole
;;; would put it on this public surface by accident.
;;;
;;; This published `lightgbm-backend' by way of `:use-reexport #:cl-gbdt/src/lightgbm/protocol'
;;; until the split, that package having re-exported the one symbol and nothing else. `classes'
;;; is where the class now lives, but it is imported from by name rather than `:use-reexport'ed
;;; in protocol's place: it also exports `lightgbm-dataset', `lightgbm-booster' and
;;; `%lightgbm-foreign-library', none of them reviewed public API, and re-exporting it whole
;;; would put all three on this surface by accident -- the same hazard the
;;; `cl-gbdt/src/lightgbm/all' paragraph above describes for `native'. Nothing else from
;;; `native' is published here either: none of its remaining exports is a reviewed, Lisp-level
;;; LightGBM-specific operation. A future Phase 2 addition follows the same shape -- named
;;; explicitly in both an `:import-from' and the `:export' clause below, never picked up by
;;; widening either into a blanket re-export of a whole package; see this comment block's own
;;; earlier paragraph for why that stays forbidden.
;;;
;;; Task 7 adds the shared-basis symbols a caller who loads `cl-gbdt/lightgbm' ALONE -- never
;;; `cl-gbdt' itself -- needs to open, question and close a backend, and to catch what it
;;; signals by name instead of reaching for `cl-gbdt/src/conditions:foreign-call-error', an
;;; internal package, directly. `:use-reexport #:cl-gbdt/src/conditions' brings in that whole
;;; package: every condition type and accessor it exports is already reviewed public API (see
;;; that file's own `:export' clause), so re-exporting it whole carries none of the
;;; "every %-function comes along for the ride" hazard the paragraphs above describe for
;;; `native'. `:import-from #:cl-gbdt/src/backend' and `:import-from #:cl-gbdt/src/handle' stay
;;; narrow, by name, for the reason `classes' and `native' do above: both packages also export
;;; symbols meant for someone IMPLEMENTING a backend, not someone using one --
;;; `register-backend', `find-backend-class', `initialize-backend', `shutdown-backend',
;;; `probe-foreign-symbols' and `probe-capabilities' from `backend'; `make-handle',
;;; `release-handle', `handle-pointer' and `handle-live-pointer' from `handle' -- and
;;; re-exporting either package whole would put every one of those on this surface by accident,
;;; the identical hazard already described above. `csr-matrix' and the rest of
;;; `cl-gbdt/src/data' still stay out, though the reason has changed: dataset construction is
;;; now at this layer, and `create-dataset' below does take a `csr-matrix', so a Layer 1
;;; caller who wants the sparse path has to reach `cl-gbdt:make-csr-matrix' -- which means
;;; loading the unified core -- or name `cl-gbdt/src/data' directly. Publishing that
;;; constructor here is a decision about the SPARSE surface, not a side effect of moving
;;; `make-dataset''s procedure down, so it is left to be made deliberately. The dense path,
;;; which is every ordinary array, needs nothing from `cl-gbdt/src/data' at all.
;;;
;;; The `:export' clause names all of them because `:import-from' imports without exporting, so
;;; it is now the only thing publishing any of them. It carries a second job as well --
;;; `tools/ci/check-float-traps.lisp' reads exactly this `:export' to decide which `defun's are
;;; entry points reached without a `defmethod' to inherit a float-trap mask from, so a public
;;; `defun' that appeared here only by way of `:use-reexport' would slip that check silently.
;;; None of Task 7's additions are `defun's in THIS file, though -- `open-backend' and
;;; `close-backend' are `defun's in `src/backend.lisp', outside the check's own file-pattern
;;; glob, and the condition accessors `:use-reexport'ed from `cl-gbdt/src/conditions' are
;;; reader functions the `define-condition' macro generates, not `defun's the check would ever
;;; count as unmasked entry points.
;;;
;;; And, as ever: never `#:cl-gbdt/src/lightgbm/c-api', the raw CFFI bindings -- see the
;;; comment above `cl-gbdt/src/lightgbm/all'.
(uiop:define-package #:cl-gbdt/lightgbm
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
  (:import-from #:cl-gbdt/src/lightgbm/classes
                #:lightgbm-backend)
  (:import-from #:cl-gbdt/src/lightgbm/native
                #:booster-eval-names
                #:booster-eval)
  (:import-from #:cl-gbdt/src/lightgbm/api
                #:create-booster
                #:create-dataset
                #:free-booster
                #:free-dataset
                #:update-one-iteration)
  (:export #:lightgbm-backend
           #:booster-eval-names
           #:booster-eval
           #:create-booster
           #:create-dataset
           #:free-booster
           #:free-dataset
           #:update-one-iteration
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
;;; again, undoing the split. Task 7's three new clauses need no such addition: `classes'
;;; itself already `:import-from's `cl-gbdt/src/backend', `cl-gbdt/src/handle' and
;;; `cl-gbdt/src/conditions' (see its own package form), so loading `cl-gbdt/src/lightgbm/all'
;;; -- which the first form's `:use-reexport' of `classes' already requires -- transitively
;;; loads all three ASDF systems before this second form is ever read.
