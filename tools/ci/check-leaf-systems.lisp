;;;; check-leaf-systems.lisp --- Load every leaf system alone, in a fresh image.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-leaf-systems.lisp
;;;;
;;;; package-inferred-system infers a file's dependencies *only* from its defpackage
;;;; clauses -- :use, :import-from (even one naming zero symbols), :local-nicknames. A
;;;; file that calls, say, cffi:defcfun without naming #:cffi in one of those clauses
;;;; gets no declared dependency on CFFI. It loads successfully whenever something else
;;;; happens to have loaded CFFI first, and breaks the moment load order shifts -- a bug
;;;; that hides behind a green build, because the obvious way to prove a set of systems
;;;; loads -- quickload them all into one image -- is exactly what hides it: the first
;;;; file's load satisfies the next file's undeclared dependency.
;;;;
;;;; So every leaf here is loaded in its own subprocess, with a fresh Lisp image, so
;;;; nothing but its own declared dependencies is on hand when it loads.

(require :asdf)

(defparameter +leaf-roots+ '("src" "tests")
  "Directories searched, recursively, for `.lisp' files naming a leaf system.")

(defparameter +system-prefix+ "cl-gbdt/"
  "Every leaf system's name is this prefix followed by its path.")

(defun leaf-system-name (path root)
  "Return the leaf system name that PATH, relative to ROOT, names.

package-inferred-system names a file's system after its path from the primary
system's root, extension dropped -- `src/lightgbm/c-api.lisp' names
`cl-gbdt/src/lightgbm/c-api'. Deriving names this way, instead of listing them by
hand, means a file added later is checked automatically instead of silently
skipped."
  (format nil "~A~A" +system-prefix+
          (uiop:unix-namestring
           (uiop:enough-pathname (make-pathname :type nil :defaults path) root))))

(defun leaf-systems ()
  "Return the sorted list of every leaf system name under +leaf-roots+.

Includes the generated `src/lightgbm/c-api.lisp' and `src/xgboost/c-api.lisp'.
`tools/ci/lint.lisp' excludes those two for style reasons, but machine output is
exactly the kind of file that can reference a package it never declares, so
excluding it here would blind the one check built to catch that."
  (let ((root (uiop:getcwd)))
    (sort (mapcan (lambda (subdir)
                     (mapcar (lambda (path) (leaf-system-name path root))
                             (directory (merge-pathnames (format nil "~A/**/*.lisp" subdir)
                                                          root))))
                   +leaf-roots+)
          #'string<)))

(defun load-in-fresh-image (system)
  "Load SYSTEM alone in a subprocess with a fresh Lisp image.

Returns (values SUCCESS-P COMBINED-OUTPUT). A fresh `ros run' subprocess, rather
than an in-image `asdf:load-system' or a shared worker, is the entire point: it
starts with nothing loaded but Quicklisp's client, so SYSTEM's own
`uiop:define-package' clauses are the only thing that can pull in a dependency."
  (multiple-value-bind (output error-output status)
      (uiop:run-program (list "ros" "run" "--" "--non-interactive"
                               "--eval" (format nil "(ql:quickload ~S :silent t)" system))
                         :output :string
                         :error-output :string
                         :ignore-error-status t)
    (values (zerop status) (concatenate 'string output error-output))))

(let ((systems (leaf-systems))
      (failures '()))
  (format t "~&checking ~D leaf system~:P, each alone in a fresh image~%" (length systems))
  (dolist (system systems)
    (multiple-value-bind (ok output) (load-in-fresh-image system)
      (cond (ok
             (format t "~&PASS ~A~%" system))
            (t
             (push system failures)
             (format t "~&FAIL ~A~%" system)
             (format *error-output* "~&----- ~A -----~%~A~&----- end ~A -----~%"
                     system output system)))))
  (format t "~&~D/~D leaf systems load alone~%"
          (- (length systems) (length failures)) (length systems))
  (when failures
    (format *error-output* "~&failing leaf systems (undeclared dependency, most likely):~%")
    (dolist (system (reverse failures))
      (format *error-output* "~&  ~A~%" system)))
  (uiop:quit (if failures 1 0)))
