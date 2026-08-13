;;;; gen-api-reference.lisp --- Regenerate docs/API-REFERENCE.md from the docstrings.
;;;;
;;;; Developer-only. Run from the repository root:
;;;;   ros run -- --non-interactive --load tools/gen-api-reference.lisp
;;;;
;;;; Loads both backends' unified systems because the reference covers all three public
;;;; packages, and a package that is not loaded exports nothing. Neither shared library is
;;;; opened: that happens only on an explicit `open-backend' call.

(require :asdf)

(asdf:load-system "cl-gbdt/lightgbm/unified")
(asdf:load-system "cl-gbdt/xgboost/unified")
(asdf:load-system "cl-gbdt/docgen")

(let ((output (merge-pathnames "docs/API-REFERENCE.md" (uiop:getcwd))))
  (cl-gbdt/src/docgen/emit:write-api-reference cl-gbdt/src/docgen/emit:+public-packages+ output)
  (format t "~&wrote ~A~%" output))
