;;;; check-layer-separation.lisp --- Layer 1 must not depend on Layer 2.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-layer-separation.lisp
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
;;;; It reads the FIRST `uiop:define-package' form of each file -- the same form ASDF's
;;;; package-inferred-system infers dependencies from -- and walks the graph itself rather
;;;; than asking ASDF, so it reports what the source says, not what a previously-loaded image
;;;; happens to contain. Nothing is loaded, compiled or evaluated: `read', with `*read-eval*'
;;;; bound to NIL so a stray `#.' cannot run code either, is all that touches the source.
;;;;
;;;; The FIRST form specifically, because each backend's `all.lisp' holds two: the internal
;;;; aggregation ASDF names the system after, and the reviewed public package
;;;; (`cl-gbdt/lightgbm', `cl-gbdt/xgboost') carrying the export list. ASDF reads only the
;;;; first, so this reads only the first. The public package's name has no file of its own,
;;;; which is why `package-file' returning NIL is a normal answer here rather than an error:
;;;; a package with no file declares no dependencies to walk.
;;;;
;;;; `clause-packages' recognises the same clauses ASDF's own `package-dependencies' does --
;;;; `:use', `:mix', `:reexport', `:use-reexport', `:mix-reexport', `:import-from',
;;;; `:shadowing-import-from' and `:local-nicknames'. Recognising fewer would be the one bug
;;;; that makes this check quietly vacuous: the missed clause is exactly the one a future
;;;; violation would be written with, and a check that cannot see it still prints PASS.
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
;;;;   - A third backend, or any Layer 1 root not named in +LAYER-1-ROOTS+. Only the two roots
;;;;     listed there are walked.
;;;;   - Anything about the other direction, or about Layer 2's internal shape. Layer 2 is
;;;;     expected to depend on Layer 1; that is not a finding.

(require :asdf)

(defparameter +layer-1-roots+
  '("cl-gbdt/src/lightgbm/all" "cl-gbdt/src/xgboost/all")
  "The package each backend's Layer 1 system depends on.

`cl-gbdt.asd' names exactly these two as `cl-gbdt/lightgbm''s and `cl-gbdt/xgboost''s sole
dependency, so walking them is walking those systems.")

(defparameter +layer-2-packages+
  '("cl-gbdt/src/protocol"
    "cl-gbdt/src/training-report"
    "cl-gbdt/src/training/history"
    "cl-gbdt/src/training/early-stopping"
    "cl-gbdt/src/training/custom-metric")
  "Packages that belong to the unified API. None may appear in a Layer 1 closure.

`src/protocol.lisp' holds the thirteen portable generics; the four training files are the
report, the history, the early-stopping watcher and the custom-metric entry, all of which
exist to serve `train''s contract and none of which a backend-specific caller needs.

Aggregates such as `cl-gbdt/src/all' are deliberately absent: they reach these packages
themselves, so the walk finds the violation through them without their being listed.")

(defun package-file (package-name)
  "Return the pathname PACKAGE-NAME's file would have, or NIL when there is none.

A package named `cl-gbdt/src/lightgbm/all' lives in `src/lightgbm/all.lisp'. Names with no
file -- `cl-gbdt/lightgbm', the public package defined as a second form in `all.lisp' -- have
no dependencies of their own to walk."
  (let ((prefix "cl-gbdt/"))
    (when (and (> (length package-name) (length prefix))
               (string= prefix package-name :end2 (length prefix)))
      (let ((path (format nil "~A.lisp" (subseq package-name (length prefix)))))
        (probe-file (merge-pathnames path (uiop:getcwd)))))))

(defun clause-packages (clause)
  "Return the package designators CLAUSE names, as lower-case strings.

The clause set is ASDF's, not a subset of it: `package-dependencies' in
`asdf/package-inferred-system' derives a file's dependencies from exactly these, and a
clause this function does not recognise is a dependency this check cannot see."
  (flet ((name (designator) (string-downcase (string designator))))
    (case (first clause)
      ((:use :mix :reexport :use-reexport :mix-reexport) (mapcar #'name (rest clause)))
      ((:import-from :shadowing-import-from) (list (name (second clause))))
      ;; `(:local-nicknames (#:nick #:actual) ...)': the nickname is local to the file, the
      ;; second element of each pair is the real package and the real dependency.
      ((:local-nicknames) (loop :for pair :in (rest clause)
                                :when (consp pair)
                                  :collect (name (second pair))))
      (t nil))))

(defun package-form (file)
  "Return FILE's first top-level form, read as data.

`*read-eval*' is NIL so a `#.' in the source cannot run code, and `*package*' is CL-USER
because every package designator in these forms is written `#:like-this' -- uninterned, and
so harmless to read anywhere. `uiop:define-package' itself is the one qualified symbol, and
`(require :asdf)' above has already made UIOP exist to intern it against."
  (with-open-file (stream file)
    (let ((*package* (find-package :cl-user))
          (*read-eval* nil))
      (read stream nil nil))))

(defun direct-dependencies (package-name)
  "Return the cl-gbdt packages PACKAGE-NAME's own file declares."
  (let ((file (package-file package-name)))
    (when file
      (let ((form (package-form file)))
        (remove-if-not
         (lambda (name) (uiop:string-prefix-p "cl-gbdt/" name))
         (loop :for clause :in (cddr form)
               :when (consp clause)
                 :append (clause-packages clause)))))))

(defun closure (root)
  "Return every cl-gbdt package reachable from ROOT, ROOT included, sorted."
  (let ((seen (make-hash-table :test #'equal)))
    (labels ((walk (name)
               (unless (gethash name seen)
                 (setf (gethash name seen) t)
                 (mapc #'walk (direct-dependencies name)))))
      (walk root))
    (sort (loop :for name :being :the :hash-keys :of seen :collect name) #'string<)))

(defun check-root (root)
  "Report on ROOT. Returns true when its closure contains no Layer 2 package.

The closure's size is printed on every run, passing or failing. A walker that stopped at the
root would report 1 or 2 packages and then find no violation, printing the same PASS a real
walk prints -- the count is what tells a reader which of the two just happened."
  (let* ((reached (closure root))
         (violations (remove-if-not (lambda (name) (member name +layer-2-packages+
                                                           :test #'string=))
                                    reached)))
    (format t "~&~A: ~D packages in its closure~%" root (length reached))
    (dolist (violation violations)
      (format t "  FAIL: reaches ~A, which is Layer 2~%" violation))
    (null violations)))

(let ((ok t))
  (dolist (root +layer-1-roots+)
    (unless (check-root root) (setf ok nil)))
  (if ok
      (format t "~&PASS: no Layer 1 system reaches the unified API~%")
      (format t "~&FAIL: a Layer 1 system reaches the unified API~%"))
  (finish-output)
  (uiop:quit (if ok 0 1)))
