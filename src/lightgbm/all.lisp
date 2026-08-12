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
;;; Nine symbols from those three packages, each pulled explicitly by name. `lightgbm-backend'
;;; from `classes' is the CLOS class a caller can specialize methods on or check with `typep';
;;; `booster-eval-names' and `booster-eval' from `native' are Phase 2's first
;;; LightGBM-specific safe API
;;; (docs/superpowers/specs/2026-08-06-evaluation-api-design.md, policy section 3's Layer 1);
;;; `create-dataset', `free-dataset', `create-booster', `update-one-iteration', `predict' and
;;; `free-booster' from `api' are the finished operations. Five of the six are procedure lifted
;;; out of `cl-gbdt/src/lightgbm/protocol' -- `free-dataset', `update-one-iteration', `predict'
;;; and `free-booster' out of the methods of those very names, and `create-dataset' out of
;;; `make-dataset', whose portable name it does not share -- each of those methods now checking
;;; its portable arguments and calling the function here. `create-booster' is the sixth and is
;;; not lifted from anything: no protocol method ever built a booster OVER A DATASET on its
;;; own, `train' having always built one inline as part of a run. (`load-model' builds one too,
;;; but from a file and with no dataset in sight, so it is not this function under another
;;; name.) Together they are a whole training run at this layer, and now the inference that
;;; follows it: build a dataset, build a booster on it, advance it, score with it, free both.
;;; `create-booster' is the one with no caller inside this library -- `train' builds its own
;;; booster, for the reason its creation call records -- so it is published on the strength of
;;; its own contract rather than of a method that exercises it.
;;; `free-dataset', `free-booster', `update-one-iteration' and `predict' here are NOT
;;; `cl-gbdt''s generics of those names: they are plain functions and different symbols, so a
;;; caller who has both packages in an image must name which one they mean, exactly as they
;;; already must for anything else two packages export under one name.
;;;
;;; `api' is `:import-from'ed rather than `:use-reexport'ed for the reason `classes' is, and
;;; the reason survives that package's `:export' clause holding nothing but those six
;;; operations today: `:use-reexport' would publish whatever that clause grows next
;;; automatically, and the `:export' clause below is what `tools/ci/check-float-traps.lisp'
;;; reads to decide which `defun's are entry points -- see the paragraph on that check below,
;;; which is why a name arriving here without being written here is a hazard rather than a
;;; convenience. `%check-sparse-input' is the name that used to make this concrete: `api'
;;; exported it while `protocol.lisp''s `predict' called it from the other file, and
;;; `:use-reexport' would have put an internal capability gate on this public surface. That
;;; procedure is now `api''s own `predict', both of the gate's call sites are in that file, and
;;; the name is no longer exported at all.
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
;;; the identical hazard already described above.
;;;
;;; The sparse surface is that deliberate decision, taken now rather than left open: both
;;; `create-dataset' and `predict' accept a `csr-matrix' wherever they accept a dense matrix,
;;; and until this form named it, a Layer 1 caller could reach the sparse half of either only
;;; by loading the unified core for `cl-gbdt:make-csr-matrix' or by naming the internal
;;; `cl-gbdt/src/data' directly -- so half of two published contracts was unreachable from the
;;; package that publishes them. `make-csr-matrix', the `csr-matrix' structure type and its
;;; five readers are named; `foreign-matrix', `with-foreign-matrix' and the rest of that package
;;; stay out, being the plumbing a dense matrix is handed to C through and not something a
;;; caller builds. The dense path, which is every ordinary array, still needs none of it.
;;; `booster-training-set' and `booster-validation-sets' from `handle' are here for the same
;;; reason: `create-booster' retains both, and without these a caller who built a booster at
;;; this layer could not read back what it attached -- `handle-released-p', already published,
;;; is what makes the answer worth having, since a retained dataset can have been freed.
;;;
;;; None of those nine is a symbol of this package's own: they are the very symbols
;;; `cl-gbdt/src/data' and `cl-gbdt/src/handle' define, so unlike `predict' and the three
;;; other doubled operation names above, a caller holding both `cl-gbdt' and `cl-gbdt/lightgbm'
;;; sees one symbol reached two ways and has nothing to disambiguate.
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
  (:import-from #:cl-gbdt/src/data
                #:csr-matrix
                #:csr-matrix-indices
                #:csr-matrix-indptr
                #:csr-matrix-num-columns
                #:csr-matrix-num-rows
                #:csr-matrix-values
                #:make-csr-matrix)
  (:import-from #:cl-gbdt/src/handle
                #:booster
                #:booster-training-set
                #:booster-validation-sets
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
                #:dataset-num-features
                #:dataset-num-rows
                #:evaluation
                #:feature-importance
                #:free-booster
                #:free-dataset
                #:load-model
                #:model-to-string
                #:predict
                #:save-model
                #:update-one-iteration)
  (:export #:lightgbm-backend
           #:booster-eval-names
           #:booster-eval
           #:create-booster
           #:create-dataset
           #:dataset-num-features
           #:dataset-num-rows
           #:evaluation
           #:feature-importance
           #:free-booster
           #:free-dataset
           #:load-model
           #:model-to-string
           #:predict
           #:save-model
           #:update-one-iteration
           #:csr-matrix
           #:csr-matrix-indices
           #:csr-matrix-indptr
           #:csr-matrix-num-columns
           #:csr-matrix-num-rows
           #:csr-matrix-values
           #:make-csr-matrix
           #:booster-training-set
           #:booster-validation-sets
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
;;; loads all three ASDF systems before this second form is ever read. The
;;; `cl-gbdt/src/data' clause added for the sparse surface is the same case one package
;;; further out: `api', which the first form does `:use-reexport', `:import-from's that package
;;; itself -- it has to, `create-dataset' and `predict' both branching on `csr-matrix' -- so
;;; the system is loaded by the time this form is read, and the first form needs no edit.
