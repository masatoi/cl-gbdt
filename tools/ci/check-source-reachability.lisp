;;;; check-source-reachability.lisp --- every source file is reachable from a declared system.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-source-reachability.lisp
;;;;
;;;; Run it from the repository root; see THE FLOORS below for what happens if you do not.
;;;;
;;;; WHAT THIS GUARDS. `src/all.lisp' hand-maintains the list of packages that make up
;;;; `cl-gbdt''s public surface, and its own header says what happens when that list falls
;;;; behind: "Add a src/foo.lisp and forget to list its package here, and nothing ever refers
;;;; to cl-gbdt/src/foo, so ASDF has no edge to discover it by -- (ql:quickload :cl-gbdt)
;;;; never even compiles the file, let alone exports its symbols. There is no separate failing
;;;; test for this one: the silent failure is the file never being loaded at all, not an
;;;; assertion to trip." This file is that separate failing test.
;;;;
;;;; WHAT IT DOES NOT ASSERT. Not that every file appears in `src/all.lisp'. That would be
;;;; wrong: the backend files are deliberately absent from it, `cl-gbdt/lightgbm' and
;;;; `cl-gbdt/xgboost' being separate systems that do not depend on core `cl-gbdt' at all, and
;;;; `src/regen/' and `src/docgen/' belong to development-only systems. The invariant is that
;;;; every file is in SOME declared system's dependency closure, which is what makes the
;;;; difference between a file that is loaded and one that is dead. Which list a file belongs
;;;; in is a design decision this check deliberately knows nothing about.
;;;;
;;;; `tests/' IS IN SCOPE, and that is the half worth explaining.
;;;; `tools/ci/check-leaf-systems.lisp' already loads every leaf system alone in a fresh
;;;; subprocess, and it would load an orphaned test file happily -- while that file still
;;;; never ran in the suite. Loading and running are different properties and only the first
;;;; was guarded. `cl-gbdt.asd' records an instance in its own comments: a test system once
;;;; named so that rove's prefix match missed it reported "0 tests completed", a green run of
;;;; an empty suite.
;;;;
;;;; `tools/' IS NOT. `tools/ci/*.lisp' are standalone `--load' scripts rather than ASDF
;;;; components, so including them would make every one of them a permanent false positive.
;;;;
;;;; THE SYSTEM LIST IS DERIVED, NOT WRITTEN DOWN. It comes from reading `cl-gbdt.asd' and
;;;; collecting each `defsystem' form's name. A checker that exists BECAUSE a hand-maintained
;;;; list went stale must not introduce a second one -- and this one would fall behind in the
;;;; direction of silence, since a system missing from the checker's own list makes its files
;;;; look unreachable rather than making the checker notice it is blind.
;;;;
;;;; Reading the `.asd' as source rather than asking a loaded image follows
;;;; `tools/ci/check-support-matrix.lisp', which reads `.github/workflows/test.yml' without a
;;;; YAML parser for the same reason: what the file says is the thing under test. `*read-eval*'
;;;; is bound to NIL while reading, as `tools/ci/check-layer-separation.lisp' does, so a stray
;;;; `#.' cannot execute code during a build check.
;;;;
;;;; NOTHING IS COMPILED OR LOADED. `asdf:required-components' builds a plan statically;
;;;; measured in a fresh image with nothing preloaded, it gives the same answer as one where
;;;; every system was already loaded.
;;;;
;;;; THE FLOORS. If a glob matched nothing -- run from the wrong directory, or a pattern that
;;;; stopped recursing -- the unreached set would be empty and this check would pass while
;;;; having examined nothing. `tools/ci/check-binding-coverage.lisp' carries an empty-parse
;;;; guard for exactly that failure, and so does this.
;;;;
;;;; The floors are SANITY GUARDS, NOT RATCHETS. They exist to catch a broken glob, they are
;;;; set well below the real counts, and adding a source file must never require editing them.
;;;; That distinction matters here: this project carries a floor that SHOULD rise with the
;;;; surface (`+minimum-published-symbols+' in check-api-reference.lisp) and a ratchet that
;;;; must NOT (`+maximum-unproven+' in check-functional-coverage.lisp), and a third kind that
;;;; should simply be left alone needs to say so rather than invite a well-meaning bump.
;;;;
;;;; One floor per tree, not a shared one: `src/' and `tests/' are globbed separately and a
;;;; pattern can break for one without breaking the other, so a single combined number would
;;;; let a broken `tests/' glob hide behind a healthy `src/' count.

(require :asdf)

(defparameter +system-definition+ "cl-gbdt.asd"
  "The file whose `defsystem' forms name every system this check walks.")

(defparameter +trees+
  '(("src" . 30) ("tests" . 30))
  "Each tree this check scans, paired with the floor on how many `.lisp' files it must find
there. Measured 2026-08-26: `src/' holds 42 and `tests/' 41. The floors sit well below both on
purpose -- they catch a glob that matched nothing, not a tree that grew, and adding a file must
never require touching them. See THE FLOORS in this file's header.

Hardcoded to these two, the same as `tools/ci/check-leaf-systems.lisp''s `+leaf-roots+'. A
tree added elsewhere -- `examples/', `contrib/' -- with `.lisp' sources wired to no system
leaves both that check and this one green, and nothing here would say why.")

(defparameter +minimum-systems+ 5
  "Floor on how many `defsystem' forms reading `cl-gbdt.asd' must yield. Nine are declared
today. Zero would mean the read found nothing -- a moved file, a changed name -- and a check
that walked no systems would report every source file as unreachable, which is loud, or, if the
trees were empty too, nothing at all, which is not.")

(defun die (format-control &rest arguments)
  "Print FORMAT-CONTROL/ARGUMENTS to *ERROR-OUTPUT* as a FAIL line and exit with status 1.

Shaped like the `die' in tools/ci/check-api-reference.lisp: every caller has already printed
its own offenders, so this line is a summary of what came before it, never the only line a
failure prints."
  (format *error-output* "~&FAIL ~?~%" format-control arguments)
  (finish-output *error-output*)
  (uiop:quit 1))

(defun declared-system-names (path)
  "Return the name of every `defsystem' form in PATH, in file order.

Reads rather than loads, in a throwaway package so that the symbols an `.asd' happens to
mention are never interned anywhere that matters, and with `*read-eval*' bound to NIL so a
stray `#.' cannot run code. A form counts when its head prints as DEFSYSTEM and its second
element is a string -- both `asdf:defsystem' and a bare `defsystem' therefore match, since only
the symbol NAME is compared."
  (let ((*read-eval* nil)
        (scratch (or (find-package '#:cl-gbdt-reachability-scratch)
                     (make-package '#:cl-gbdt-reachability-scratch :use nil)))
        (names '()))
    (with-open-file (stream path :if-does-not-exist nil)
      (unless stream
        (die "~A not found -- run this from the repository root." path))
      (let ((*package* scratch))
        (loop :for form := (read stream nil :eof)
              :until (eq form :eof)
              :do (when (and (consp form)
                             (symbolp (car form))
                             (string= (symbol-name (car form)) "DEFSYSTEM")
                             (stringp (second form)))
                    (push (second form) names)))))
    (nreverse names)))

(defun reached-pathnames (system-names)
  "Return a hash table whose keys are the truename namestrings of every source file the systems
named by SYSTEM-NAMES transitively reach.

`:other-systems t' is what makes this transitive across system boundaries -- without it a file
pulled in only through `cl-gbdt/lightgbm/unified' would look unreachable from `cl-gbdt'. A
component whose pathname does not exist is skipped rather than signalling: `probe-file' first,
because a plan may name a file a `.asd' declares and the tree does not carry, and this check's
job is to report unreachable files, not to be the first thing that notices a missing one."
  (let ((reached (make-hash-table :test #'equal)))
    (dolist (name system-names reached)
      (let ((system (asdf:find-system name nil)))
        (unless system
          (die "system ~A is not on ASDF's search path -- likely cause: this repository ~
                is not registered with ASDF (e.g. missing the ~~/.roswell/local-projects ~
                symlink CI creates); run this from the repository root with the project ~
                registered."
               name))
        (dolist (component (asdf:required-components system
                                                     :other-systems t
                                                     :keep-component 'asdf:cl-source-file))
          (let ((path (asdf:component-pathname component)))
            (when (and path (probe-file path))
              (setf (gethash (namestring (truename path)) reached) t))))))))

(defun tree-files (root tree)
  "Return every `.lisp' file under TREE, a directory name relative to ROOT.

Globs the filesystem directly rather than asking `git ls-files' the way
`tools/ci/check-doc-links.lisp' does -- deliberately: an untracked REPL scratch file such
as `src/try.lisp' reddens this check locally the moment it exists, before it is ever
`git add'-ed, while CI, which only ever has tracked files checked out, stays green."
  (directory (merge-pathnames (format nil "~A/**/*.lisp" tree) root)))

(let ((root (truename (uiop:getcwd)))
      (system-names (declared-system-names +system-definition+)))
  (when (< (length system-names) +minimum-systems+)
    (die "read only ~D `defsystem' form~:P from ~A, fewer than the floor of ~D. ~
          Either this is not the repository root or that file's shape has changed."
         (length system-names) +system-definition+ +minimum-systems+))
  (format t "~&~D systems declared in ~A~%" (length system-names) +system-definition+)
  (let ((reached (reached-pathnames system-names))
        (unreachable '()))
    (dolist (entry +trees+)
      (destructuring-bind (tree . floor) entry
        (let ((files (tree-files root tree)))
          (when (< (length files) floor)
            (die "found only ~D `.lisp' file~:P under ~A/, fewer than the floor of ~D. ~
                  A glob that matches nothing would let this check pass having examined ~
                  nothing; see THE FLOORS in this file's header."
                 (length files) tree floor))
          (let ((missed (remove-if (lambda (file)
                                     (gethash (namestring (truename file)) reached))
                                   files)))
            (format t "~&~A/: ~D file~:P, ~D unreachable~%" tree (length files)
                    (length missed))
            (dolist (file missed)
              (format *error-output* "~&FAIL ~A is in no declared system's dependency closure ~
                                      -- nothing compiles or loads it.~%"
                      (enough-namestring file root))
              (push file unreachable))))))
    (if unreachable
        (die "~D source file~:P unreachable -- add each to the system that should carry it, ~
              or delete it."
             (length unreachable))
        (progn
          (format t "~&PASS: every source file is reachable from a declared system~%")
          (finish-output)
          (uiop:quit 0)))))
