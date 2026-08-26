;;;; all.lisp --- The public face of cl-gbdt.
;;;;
;;;; Under package-inferred-system each source file owns its own package. This file
;;;; reassembles them into CL-GBDT, which is what callers use and what the library
;;;; has always exported. Adding a symbol to any of the ten packages below exports
;;;; it from CL-GBDT automatically; nothing is listed twice.
;;;;
;;;; The list of ten packages below is hand-maintained: package-inferred-system
;;;; does not enumerate src/ on its own. Add a src/foo.lisp and forget to list its
;;;; package here, and nothing ever refers to cl-gbdt/src/foo, so ASDF has no edge
;;;; to discover it by -- (ql:quickload :cl-gbdt) never even compiles the file, let
;;;; alone exports its symbols. tools/ci/check-source-reachability.lisp is now the
;;;; failing test for a file no system reaches at all, walking every defsystem in
;;;; cl-gbdt.asd to find any .lisp under src/ or tests/ that no closure reaches --
;;;; but it does not catch a src/ file reached only through a test's own dependency:
;;;; that one still loads under cl-gbdt/tests while cl-gbdt itself never compiles
;;;; it. So this list still has to be maintained by hand and by eye; the check
;;;; narrows the silent-failure window rather than closing it.

(uiop:define-package #:cl-gbdt/src/all
  (:use-reexport #:cl-gbdt/src/conditions
                 #:cl-gbdt/src/data
                 #:cl-gbdt/src/backend
                 #:cl-gbdt/src/protocol
                 #:cl-gbdt/src/handle
                 #:cl-gbdt/src/parameters
                 #:cl-gbdt/src/library
                 #:cl-gbdt/src/foreign
                 #:cl-gbdt/src/version
                 #:cl-gbdt/src/training-report))

;;; Backend-specific files do not belong in the list above. `cl-gbdt' is the core system
;;; and loads without either shared library; `cl-gbdt/lightgbm' and `cl-gbdt/xgboost' are
;;; separate systems that do not depend on it at all, and `cl-gbdt/lightgbm/unified' and
;;; `cl-gbdt/xgboost/unified' the ones layered on top of it.
;;; Listing `cl-gbdt/src/xgboost/array-interface' here made
;;; the core depend on an XGBoost file and re-exported an XGBoost implementation detail from
;;; `CL-GBDT'. Every consumer imports it directly from its own package instead, which is why
;;; nothing broke when it was removed. `src/lightgbm/native'/`classes'/`protocol'/`all'/
;;; `unified' (once a single `src/lightgbm/backend.lisp', split into native/protocol/all by
;;; the Phase 1 layering and split again when Layer 1 became a system of its own) were never
;;; listed, for the same reason.

;;; package-inferred-system infers a file's dependencies from the file's *first*
;;; defpackage form only. This second form's :use-reexport clause is therefore
;;; inert for dependency purposes -- a new dependency belongs in the
;;; #:cl-gbdt/src/all form above, not here.
(uiop:define-package #:cl-gbdt
  (:use-reexport #:cl-gbdt/src/all))
