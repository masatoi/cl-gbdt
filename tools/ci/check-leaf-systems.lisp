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
  "Directories searched, recursively, for `.lisp' files naming a leaf system.

Hardcoded to these two. `leaf-systems' below is automatic only in the sense that
a new file *under one of these roots* needs no update here -- a file added under
some other root would not be picked up at all.")

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

(defun undefined-reference-p (output)
  "True when OUTPUT reports a name the compiler could not resolve.

An undeclared dependency surfaces in two different ways, and a non-zero exit status
only reveals one of them:

  cffi:defcfun with no #:cffi clause     READ error -- the load fails outright
  parse-body with no #:alexandria clause undefined function -- a STYLE-WARNING,
                                         and the load SUCCEEDS

Only the first is visible in the exit status, so a checker reading that alone passes
a file whose imports are missing. Promoting warnings to errors does not work either:
SBCL defers undefined-name warnings to the end of the compilation unit, where
`compile-file' has already returned its WARNINGS-P, and doing so would also fail on
unrelated pre-existing style warnings such as the ignored-variable ones in
tests/functional/lightgbm.lisp.

These two are not exhaustive, either. A dropped import of a *condition or class*
name -- `(error 'dimension-mismatch)' with no clause naming the package that
exports `dimension-mismatch' -- interns a fresh, unrelated symbol in the local
package at read time. That symbol is never used as a function or variable, so it
triggers neither a READ error nor an undefined-function warning; the file loads
fine and silently signals the wrong condition (or none, if nothing ever handles
the impostor symbol).

The trade-off: a dependency library emitting such a warning during its own first
compile would fail this check too. That is the right direction to err -- a loud,
diagnosable failure rather than a silent pass -- and this project's dependencies
(cffi, alexandria, rove, jzon) compile clean."
  (or (search "undefined function" output)
      (search "undefined variable" output)))

(defun load-in-fresh-image (system)
  "Load SYSTEM alone in a subprocess with a fresh Lisp image.

Returns (values SUCCESS-P COMBINED-OUTPUT). A fresh `ros run' subprocess, rather
than an in-image `asdf:load-system' or a shared worker, is the entire point: it
starts with nothing loaded but Quicklisp's client, so SYSTEM's own
`uiop:define-package' clauses are the only thing that can pull in a dependency.

The exit status alone catches only half the failure mode, which is why the output
is scanned too. See `undefined-reference-p'.

After loading, the subprocess also asserts that a package named after SYSTEM
actually exists. ASDF derives a package-inferred-system's name from the file's
*path*, never from the name inside its `uiop:define-package' form -- a file at
`src/foo.lisp' declaring `#:cl-gbdt/src/bar' loads cleanly and, without this
check, would report PASS. This is the enforcement for the path-is-the-name
convention the rest of this project relies on."
  (multiple-value-bind (output error-output status)
      (uiop:run-program (list "ros" "run" "--" "--non-interactive"
                               ;; quickload proves the leaf resolves with only its declared
                               ;; dependencies present, but it swallows compiler output --
                               ;; verified, with or without :silent. So the leaf is then
                               ;; recompiled with `asdf:load-system :force t', which does
                               ;; print, giving `undefined-reference-p' something to read.
                               ;; :force t forces this system only, not its dependencies.
                               "--eval" (format nil
                                                "(progn (ql:quickload ~S :silent t)~
                                                 (asdf:load-system ~S :force t)~
                                                 (unless (find-package (string-upcase ~S))~
                                                   (format t \"no package named ~~A after ~
                                                              loading system ~~A~~%\" ~S ~S)~
                                                   (uiop:quit 3)))"
                                                system system system system system))
                         :output :string
                         :error-output :string
                         :ignore-error-status t)
    (let ((combined (concatenate 'string output error-output)))
      (values (and (zerop status) (not (undefined-reference-p combined)))
              combined))))

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
