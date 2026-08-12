;;;; check-layer-separation.lisp --- Layer 1 must not depend on Layer 2.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-layer-separation.lisp
;;;;
;;;; Run it from the repository root; see THE FLOOR below for what happens if you do not.
;;;;
;;;; `cl-gbdt/lightgbm' and `cl-gbdt/xgboost' are each meant to be usable on their own,
;;;; without the unified API. A caller who wants only LightGBM's own surface -- its backend
;;;; class, the library's lifetime, `booster-eval' -- should not have to load the thirteen
;;;; portable generics, the training report, the history, the early-stopping watcher and the
;;;; custom-metric entry, all of which exist to serve `train''s contract and none of which
;;;; that caller ever reaches. Splitting the files apart once was easy; the split survives
;;;; only if a later `:import-from #:cl-gbdt/src/protocol' added to `native.lisp' or
;;;; `classes.lisp' fails the build. That is this file.
;;;;
;;;; Nothing else would notice. `cl-gbdt/lightgbm' would keep loading -- it would simply pull
;;;; Layer 2 in with it -- every test would stay green, and the systems would still be named
;;;; as though the layers were separate. The failure this guards against is not a broken
;;;; build; it is a claim in `cl-gbdt.asd' and in both `all.lisp' headers quietly becoming
;;;; false.
;;;;
;;;; HOW IT DECIDES
;;;;
;;;; It reads the first DEFPACKAGE form of each file -- the same form, found the same way,
;;;; that ASDF's package-inferred-system infers dependencies from -- and walks the graph
;;;; itself rather than asking ASDF, so it reports what the source says, not what a
;;;; previously-loaded image happens to contain. Nothing is loaded, compiled or evaluated:
;;;; `read', with `*read-eval*' bound to NIL so a stray `#.' cannot run code either, is all
;;;; that touches the source.
;;;;
;;;; It reads `cl-gbdt.asd' the same way first, because the walk walks a PACKAGE and the
;;;; claim being checked is about a SYSTEM. `cl-gbdt/lightgbm' is Layer 1 only for as long as
;;;; its `:depends-on' names `cl-gbdt/src/lightgbm/all' and nothing else; repoint it at
;;;; `cl-gbdt/src/lightgbm/unified' and `(ql:quickload :cl-gbdt/lightgbm)' loads the entire
;;;; unified API -- the exact property this check exists to protect -- while every closure
;;;; below stays clean, because the package it walks did not move. That was not hypothetical
;;;; either: with the pairing asserted only in a docstring, the reviewer's one-line
;;;; `:depends-on' edit printed PASS and exited 0. `check-system-dependency' is what reads the
;;;; `.asd' and refuses to accept the premise on the strength of a comment.
;;;;
;;;; The first DEFPACKAGE form, not the first form: `package-form' skips whatever precedes
;;;; it, exactly as ASDF's `stream-defpackage-form' does. Reading the first form
;;;; unconditionally instead is fail-open, and was: one `(defparameter cl-user::*harmless* 1)'
;;;; prepended to `src/lightgbm/native.lisp' took that closure from 13 packages to 8 and still
;;;; printed PASS -- and went on printing PASS with a real `:import-from
;;;; #:cl-gbdt/src/protocol' sitting in the package form it had stopped reading.
;;;;
;;;; The FIRST such form specifically, because each backend's `all.lisp' holds two: the
;;;; internal aggregation ASDF names the system after, and the reviewed public package
;;;; (`cl-gbdt/lightgbm', `cl-gbdt/xgboost') carrying the export list. ASDF reads only the
;;;; first, so this reads only the first. The public package's name has no file of its own,
;;;; which is why `package-file' returning NIL is a normal answer for a non-root name rather
;;;; than an error: a package with no file declares no dependencies to walk.
;;;;
;;;; THE BARE NAME `cl-gbdt'
;;;;
;;;; `cl-gbdt' is a dependency like any other, and the most damaging one on this list: ASDF's
;;;; `package-name-system' falls back to the same-named system, so `(:import-from #:cl-gbdt
;;;; #:train)' in a Layer 1 file pulls in `cl-gbdt.asd''s primary system -- which is
;;;; `src/all.lisp', which is the entire unified core. It is deliberately matched separately
;;;; from the `cl-gbdt/' prefix, because it has no slash and a prefix test written with one
;;;; discards it. That is not hypothetical: this check shipped with exactly that bug, and
;;;; reported 13 packages and PASS on a tree where `(ql:quickload "cl-gbdt/lightgbm")'
;;;; demonstrably defined `CL-GBDT/SRC/PROTOCOL' and `CL-GBDT/SRC/TRAINING-REPORT'.
;;;;
;;;; THE FLOOR
;;;;
;;;; A check that resolves paths against the current directory reports a clean PASS when it
;;;; finds no files at all -- every closure is the root alone, no closure contains a Layer 2
;;;; package, exit 0. Printing the closure sizes makes that visible to a human reading the
;;;; output, but exit 0 is what CI consults, and a human is not who this runs for. So the
;;;; sizes are a floor, not a note: a root whose file cannot be found, or whose closure is one
;;;; package, fails. That covers being run from the wrong directory, a typo or rename in
;;;; +LAYER-1-SYSTEMS+, and an `all.lisp' that moved -- each of which otherwise passes
;;;; silently, and each of which turns the two failures above into loud ones too. A missing
;;;; `cl-gbdt.asd', or one defining no system by the name +LAYER-1-SYSTEMS+ gives, fails for
;;;; the same reason. The sibling `tools/ci/check-abi-blacklist.lisp' carries an empty-parse
;;;; guard for the same reason.
;;;;
;;;; WHAT THIS CANNOT CATCH
;;;;
;;;;   - A reference to a Layer 2 symbol that no package clause declares. ASDF would not
;;;;     record the dependency either, so Layer 1 still loads without Layer 2 present and
;;;;     breaks at the call site instead. `tools/ci/check-leaf-systems.lisp' is the check for
;;;;     that failure mode, loading each leaf alone in a fresh image; this one reads
;;;;     declarations only.
;;;;   - A Layer 2 package added later under a name +LAYER-2-PACKAGES+ does not list. That
;;;;     list is hand-maintained, the same caveat `check-leaf-systems.lisp''s +LEAF-ROOTS+ and
;;;;     `check-float-traps.lisp''s +BACKEND-FILE-PATTERNS+ carry for themselves: a new
;;;;     `src/training/*.lisp' needs a line added here, and nothing but review notices if it
;;;;     does not get one.
;;;;   - A third backend, or any Layer 1 system not named in +LAYER-1-SYSTEMS+. Only the two
;;;;     pairs listed there are checked and walked -- though a root that stops resolving, and
;;;;     a system name `cl-gbdt.asd' no longer defines, now fail rather than passing
;;;;     vacuously, per THE FLOOR.
;;;;   - A Layer 2 system that starts depending on something it should not. Layer 2 is
;;;;     expected to reach Layer 1 and the unified core both; only the two Layer 1 systems'
;;;;     `:depends-on' entries are read.
;;;;   - Anything about the other direction, or about Layer 2's internal shape. Layer 2 is
;;;;     expected to depend on Layer 1; that is not a finding.
;;;;   - A dependency introduced by a clause under a reader conditional this implementation
;;;;     excludes. `read' honours `#+'/`#-' with this host's own features, which is the right
;;;;     answer for the host ASDF will infer on, and the wrong one for every other host.

(require :asdf)

(define-condition malformed-source (error)
  ((path :initarg :path :reader malformed-source-path)
   (detail :initarg :detail :reader malformed-source-detail))
  (:report (lambda (condition stream)
             (format stream "~A: ~A" (malformed-source-path condition)
                     (malformed-source-detail condition))))
  (:documentation "Signalled when a file cannot be read the way package-inferred-system
would read it -- no defpackage form at all, or a clause whose dependency contribution this
script cannot classify. Both are reported as failures rather than skipped, because either
one silently subtracts from what the walk can see."))

(defparameter +layer-1-systems+
  '(("cl-gbdt/lightgbm" . "cl-gbdt/src/lightgbm/all")
    ("cl-gbdt/xgboost" . "cl-gbdt/src/xgboost/all"))
  "Each backend's Layer 1 ASDF system, paired with the package this check walks for it.

The pairing is the premise every closure below rests on: walking the package is walking the
system only while `cl-gbdt.asd' names that package as the system's sole dependency.
`check-system-dependency' reads the `.asd' and verifies exactly that, because a premise
asserted here and nowhere else is a premise that goes on reading true after it stops being
true -- see this file's HOW IT DECIDES for the measured PASS that produced.

Each package must also resolve to a real file -- see `check-root' -- so a rename here fails
loudly instead of checking nothing.")

(defparameter +system-definition-file+ "cl-gbdt.asd"
  "The system definition read by `check-system-dependency', relative to the current
directory. Resolved the same way every source file here is, so being run from the wrong
directory fails rather than checking nothing.")

(defparameter +layer-2-packages+
  '("cl-gbdt"
    "cl-gbdt/src/protocol"
    "cl-gbdt/src/training-report"
    "cl-gbdt/src/training/history"
    "cl-gbdt/src/training/early-stopping"
    "cl-gbdt/src/training/custom-metric")
  "Packages that belong to the unified API. None may appear in a Layer 1 closure.

`src/protocol.lisp' holds the thirteen portable generics; the four training files are the
report, the history, the early-stopping watcher and the custom-metric entry, all of which
exist to serve `train''s contract and none of which a backend-specific caller needs.

`cl-gbdt' -- the bare name, no slash -- is the primary system in `cl-gbdt.asd', whose sole
dependency is `cl-gbdt/src/all'. Depending on it pulls in the whole unified core in one
clause, which makes it the worst entry on this list rather than an edge case; see this
file's header for why it needs matching separately from the `cl-gbdt/' prefix.

Aggregates such as `cl-gbdt/src/all' are deliberately absent: they reach these packages
themselves, so the walk finds the violation through them without their being listed.

`cl-gbdt/src/config/categorical-features' and `cl-gbdt/src/config/prediction-shape' are
absent deliberately too, and for a different reason: `src/config/' is a mixed directory --
`feature-names', `objective' and `missing-value' from the same directory are already in both
Layer 1 closures -- and those two files hold argument validation, not unified-API code. They
were Layer 2 by usage when this paragraph was written, and half of that is no longer true:
`prediction-shape' is now reached by `src/lightgbm/api.lisp', which derives `predict''s
second value with its `contrib-shape', and by nothing in `protocol.lisp' at all. That is the
case this paragraph anticipated in the abstract, arriving for real; `categorical-features' is
still reached only by the two `protocol.lisp' files. Neither belongs on this list, because
this list is about content and neither file's content is unified-API code.")

(defparameter +dependency-clauses+
  '(:use :mix :reexport :use-reexport :mix-reexport)
  "Clauses whose every argument names a package depended upon.")

(defparameter +dependency-first-argument-clauses+
  '(:import-from :shadowing-import-from)
  "Clauses whose FIRST argument names a package depended upon; the rest name symbols.")

(defparameter +dependency-free-clauses+
  '(:nicknames :documentation :shadow :export :intern :unintern :recycle :size :lock)
  "Clauses that introduce no dependency, enumerated rather than assumed.

Listing these explicitly is what lets an UNRECOGNISED clause be a failure instead of a
silent no-op. A clause nobody classified is precisely the one a future violation gets
written with -- that is how `:local-nicknames' was missed on this script's first pass --
so it fails and asks to be classified, the same posture ASDF takes when its own
`package-dependencies' meets an option it does not know (a continuable error, which is
fatal in a non-interactive build).

`:lock' and `:size' are SBCL's and CL's respectively; ASDF classifies `:lock' the same
way and does not know `:size' at all.")

(defun package-relative-path (package-name)
  "Return the repository-relative path PACKAGE-NAME's file would have, or NIL for a name
that owns no file.

A package named `cl-gbdt/src/lightgbm/all' lives in `src/lightgbm/all.lisp'. Two kinds of
name own no file and yield NIL: the bare `cl-gbdt', which `cl-gbdt.asd' defines directly,
and `cl-gbdt/lightgbm', the public package defined as a second form inside `all.lisp'."
  (let ((prefix "cl-gbdt/"))
    (when (and (> (length package-name) (length prefix))
               (string= prefix package-name :end2 (length prefix)))
      (format nil "~A.lisp" (subseq package-name (length prefix))))))

(defun package-file (package-name)
  "Return the existing pathname of PACKAGE-NAME's file, or NIL when there is none.

NIL means either that PACKAGE-NAME owns no file (see `package-relative-path') or that its
file is not where the current directory says it should be. Those are the same answer here
because a package with no readable file declares no dependencies this script can walk; for
a name in +LAYER-1-ROOTS+ the difference matters and `check-root' refuses to treat NIL as
an answer at all."
  (let ((path (package-relative-path package-name)))
    (when path
      (probe-file (merge-pathnames path (uiop:getcwd))))))

(defun defpackage-form-p (form)
  "True when FORM is a `defpackage' or `uiop:define-package' form.

Matched by symbol NAME rather than identity, so a form read into this script's reader
package still matches -- the same convention `tools/ci/check-float-traps.lisp' uses. ASDF
compares against `*defpackage-forms*', which holds exactly these two symbols."
  (and (consp form)
       (symbolp (car form))
       (member (symbol-name (car form)) '("DEFPACKAGE" "DEFINE-PACKAGE") :test #'string=)))

(defun package-form (file)
  "Return FILE's first `defpackage'/`uiop:define-package' form, read as data.

Skips whatever precedes it, exactly as ASDF's `stream-defpackage-form' does: a comment, a
`declaim', an `in-package', anything. Taking the first form unconditionally instead is
fail-open -- see this file's header for the measurement.

`*read-eval*' is NIL so a `#.' in the source cannot run code, and `*package*' is CL-USER
because every package designator in these forms is written `#:like-this' -- uninterned, and
so harmless to read anywhere. `uiop:define-package' itself is the one qualified symbol, and
`(require :asdf)' above has already made UIOP exist to intern it against.

Signals `malformed-source' when FILE contains no such form: package-inferred-system could
not name a system after that file either, so silently returning no dependencies would hide
a broken file behind a passing check."
  (with-open-file (stream file)
    (let ((*package* (find-package :cl-user))
          (*read-eval* nil))
      (or (loop :for form := (read stream nil nil)
                :while form
                :when (defpackage-form-p form)
                  :return form)
          (error 'malformed-source
                 :path (enough-namestring file (uiop:getcwd))
                 :detail "contains no defpackage form")))))

(defun clause-packages (clause file)
  "Return the package designators CLAUSE names, as lower-case strings.

FILE is used only for reporting. Signals `malformed-source' for a clause head this script
cannot classify -- see +DEPENDENCY-FREE-CLAUSES+ for why an unknown clause fails rather
than contributing nothing."
  (flet ((name (designator) (string-downcase (string designator))))
    (let ((head (first clause)))
      (cond
        ((member head +dependency-clauses+) (mapcar #'name (rest clause)))
        ((member head +dependency-first-argument-clauses+) (list (name (second clause))))
        ;; `(:local-nicknames (#:nick #:actual) ...)': the nickname is local to the file, the
        ;; second element of each pair is the real package and the real dependency.
        ((eq head :local-nicknames)
         (loop :for pair :in (rest clause)
               :when (consp pair)
                 :collect (name (second pair))))
        ;; SBCL's `:implement' depends on every package it names -- implementing a package
        ;; that does not exist yet is not meaningful -- and ASDF infers it the same way.
        ((eq head :implement) (mapcar #'name (rest clause)))
        ((member head +dependency-free-clauses+) nil)
        (t (error 'malformed-source
                  :path file
                  :detail (format nil "unclassified package clause ~S; add it to ~
                                       +DEPENDENCY-CLAUSES+ or +DEPENDENCY-FREE-CLAUSES+"
                                  head)))))))

(defun cl-gbdt-package-p (name)
  "True when NAME is this project's own package: the bare `cl-gbdt' or anything under it.

The bare name is tested separately because it carries no slash, and a `cl-gbdt/' prefix
test -- which is what this script shipped with -- silently discards it. See this file's
header."
  (or (string= name "cl-gbdt")
      (uiop:string-prefix-p "cl-gbdt/" name)))

(defun direct-dependencies (package-name)
  "Return the cl-gbdt packages PACKAGE-NAME's own file declares."
  (let ((file (package-file package-name)))
    (when file
      (let ((relative (enough-namestring file (uiop:getcwd))))
        (remove-if-not #'cl-gbdt-package-p
                       (loop :for clause :in (cddr (package-form file))
                             :when (consp clause)
                               :append (clause-packages clause relative)))))))

(defun closure (root)
  "Return every cl-gbdt package reachable from ROOT, ROOT included, sorted."
  (let ((seen (make-hash-table :test #'equal)))
    (labels ((walk (name)
               (unless (gethash name seen)
                 (setf (gethash name seen) t)
                 (mapc #'walk (direct-dependencies name)))))
      (walk root))
    (sort (loop :for name :being :the :hash-keys :of seen :collect name) #'string<)))

(defun defsystem-form-p (form)
  "True when FORM is a `defsystem' form.

Matched by symbol NAME rather than identity, exactly as `defpackage-form-p' matches its
own, so `defsystem' and `asdf:defsystem' both count and neither has to be interned as
itself in this script's reader package."
  (and (consp form)
       (symbolp (car form))
       (string= (symbol-name (car form)) "DEFSYSTEM")))

(defun system-definition-forms ()
  "Return every `defsystem' form in +SYSTEM-DEFINITION-FILE+, read as data.

Read exactly as `package-form' reads a source file, and for the same reasons: `*read-eval*'
NIL so a `#.' cannot run code, `*package*' CL-USER so every symbol interns harmlessly as
data. Nothing here is evaluated, so a `.asd' whose `:perform' clause calls something is
still only a list.

Signals `malformed-source' when the file is not where the current directory says it should
be, and when it holds no `defsystem' form at all. Both are the empty-parse hazard THE FLOOR
describes: returning no systems would make every name below simply absent, and absence
must not read as agreement."
  (let ((file (probe-file (merge-pathnames +system-definition-file+ (uiop:getcwd)))))
    (unless file
      (error 'malformed-source
             :path +system-definition-file+
             :detail (format nil "no such file under ~A -- run this check from the ~
                                  repository root"
                             (uiop:getcwd))))
    (with-open-file (stream file)
      (let ((*package* (find-package :cl-user))
            (*read-eval* nil))
        (or (loop :for form := (read stream nil nil)
                  :while form
                  :when (defsystem-form-p form)
                    :collect form)
            (error 'malformed-source
                   :path (enough-namestring file (uiop:getcwd))
                   :detail "contains no defsystem form"))))))

(defun system-form (name forms)
  "Return the form in FORMS defining the system called NAME, or NIL when there is none."
  (find-if (lambda (form)
             (and (cdr form)
                  (string= (string-downcase (string (second form))) name)))
           forms))

(defun system-dependencies (form name)
  "Return the systems FORM's `:depends-on' clause names, as lower-case strings.

NAME is used only for reporting. Signals `malformed-source' when there is no `:depends-on'
clause at all, and for an entry that is not a plain designator: ASDF also accepts
`(:version ...)' and `(:feature ...)' there, and quietly reducing one of those to nothing
would let a system that depends on more than its Layer 1 root look as though it depends on
exactly it -- the same fail-closed posture `clause-packages' takes for an unknown package
clause."
  (let ((declared (getf (cddr form) :depends-on :absent)))
    (when (eq declared :absent)
      (error 'malformed-source
             :path +system-definition-file+
             :detail (format nil "system ~A has no :depends-on clause" name)))
    (loop :for entry :in declared
          :unless (or (stringp entry) (symbolp entry))
            :do (error 'malformed-source
                       :path +system-definition-file+
                       :detail (format nil "system ~A has a :depends-on entry this script ~
                                            cannot classify: ~S"
                                       name entry))
          :collect (string-downcase (string entry)))))

(defun check-system-dependency (name root forms)
  "Report on the ASDF system NAME. True when `cl-gbdt.asd' declares ROOT as its sole
dependency.

This is the premise +LAYER-1-SYSTEMS+ states and the walk cannot see: `check-root' walks a
package, and that walk says something about a system only while the system names that
package's file and nothing besides."
  (let ((form (system-form name forms)))
    (cond
      ((null form)
       (format t "~&~A: FAIL: ~A defines no system by this name~%" name +system-definition-file+)
       (format t "  Nothing below walks the system itself, so this name has to resolve here.~%")
       (format t "  Either the system was renamed, or +LAYER-1-SYSTEMS+ is stale.~%")
       nil)
      (t
       (let ((declared (system-dependencies form name)))
         (cond
           ((equal declared (list root))
            (format t "~&~A: depends on ~A alone~%" name root)
            t)
           (t
            (format t "~&~A: FAIL: ~A declares :depends-on ~S~%"
                    name +system-definition-file+ declared)
            (format t "  but this check walks ~A, so the closure below says nothing about~%"
                    root)
            (format t "  what loading ~A actually pulls in.~%" name)
            (format t "  Point the system back at its Layer 1 root, or update ~
                       +LAYER-1-SYSTEMS+ to~%")
            (format t "  name the package that is now this system's sole dependency.~%")
            nil)))))))

(defparameter +minimum-closure-size+ 1
  "A closure this size or smaller is treated as nothing having been walked.

One means the root reached nothing at all, which no `all.lisp' in this repository can
honestly produce -- each re-exports at least two packages. The floor is deliberately not
set higher: a tighter bound would encode a guess about future file layout and fail on a
legitimate reorganisation, while this one only ever fires on a walk that did not happen.")

(defun check-root (root)
  "Report on ROOT. Returns true when ROOT resolves to a file, reaches a plausible number of
packages, and reaches no Layer 2 package.

The first two conditions are the floor described in this file's header: without them, a run
that finds no files prints the same PASS a clean run prints, and exit 0 is what CI reads."
  (let ((file (package-file root)))
    (cond
      ((null file)
       (format t "~&~A: FAIL: no file for this package~@[ (expected ~A)~]~%"
               root (package-relative-path root))
       (format t "  Packages resolve against the current directory, which is ~A.~%"
               (uiop:getcwd))
       (format t "  Run this check from the repository root, or fix +LAYER-1-SYSTEMS+ if ~
                  the file moved.~%")
       nil)
      (t
       (let* ((reached (closure root))
              (violations (remove-if-not (lambda (name)
                                           (member name +layer-2-packages+ :test #'string=))
                                         reached)))
         (format t "~&~A: ~D packages in its closure~%" root (length reached))
         (cond
           ((<= (length reached) +minimum-closure-size+)
            (format t "  FAIL: nothing was walked -- a closure of ~D cannot be right for an ~
                       all.lisp.~%" (length reached))
            nil)
           (violations
            (dolist (violation violations)
              (format t "  FAIL: reaches ~A, which is Layer 2~%" violation))
            nil)
           (t t)))))))

(handler-case
    (let ((ok t)
          (systems (system-definition-forms)))
      (loop :for (name . root) :in +layer-1-systems+
            :do (unless (check-system-dependency name root systems) (setf ok nil))
                (unless (check-root root) (setf ok nil)))
      ;; The failure line is deliberately vaguer than the passing one. Not every failure
      ;; here is a layering violation -- an unresolvable root reaches nothing at all, and a
      ;; system pointed somewhere else was never walked -- and a summary claiming Layer 2 was
      ;; reached would be a false report of the finding above it.
      (if ok
          (format t "~&PASS: no Layer 1 system reaches the unified API~%")
          (format t "~&FAIL: the Layer 1 / Layer 2 separation is not verified -- see above~%"))
      (finish-output)
      (uiop:quit (if ok 0 1)))
  (malformed-source (condition)
    (format t "~&FAIL: ~A~%" condition)
    (finish-output)
    (uiop:quit 1)))
