;;;; check-functional-coverage.lisp --- Every published symbol has a position against the
;;;; functional suite.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-functional-coverage.lisp
;;;;
;;;; S4-2 of the Layer 1 standalone-library programme. The programme's fifth binding decision
;;;; requires a mechanically-checked functional test for every published symbol; this is the
;;;; mechanism. A symbol is COVERED when the functional suite's own sources name it, which this
;;;; script derives -- never written down, the same rule that keeps `wrapped' out of
;;;; ffi-spec/BINDING-COVERAGE.md. What cannot be derived, namely WHY a symbol has no functional
;;;; test, is written down in docs/FUNCTIONAL-COVERAGE.md, and this script fails the build on a
;;;; published symbol that is neither covered nor classified.
;;;;
;;;; WHAT THIS DOES NOT GUARANTEE. That the suite's source NAMES a symbol is not that the
;;;; symbol's contract is exercised: a condition named in a `handler-case' clause counts, and so
;;;; does a function called once in a fixture nothing asserts on. This check gives every symbol a
;;;; recorded position. It does not measure how well anything is tested, and a reader who takes a
;;;; green run for the latter has been misled -- which is why docs/FUNCTIONAL-COVERAGE.md says so
;;;; in its own header too.

(require :asdf)

(asdf:load-system "cl-gbdt/lightgbm/unified")
(asdf:load-system "cl-gbdt/xgboost/unified")
(asdf:load-system "cl-gbdt/docgen")
;;; Loading the functional test system is what makes its packages -- and their :local-nicknames,
;;; which four of its files use -- exist, so the standard reader can resolve `xgb::foo' and
;;; `support:make-separable-dataset' below. Re-deriving those nicknames from the :local-nicknames
;;; clauses would give the package system's own fact a second home to go stale in. It needs
;;; neither shared library: nothing here calls `open-backend', and tools/ci/check-leaf-systems.lisp
;;; already loads every one of these files in lint.yml, which never fetches them.
(asdf:load-system "cl-gbdt/tests/functional")

(defun die (format-control &rest arguments)
  "Print a failure to `*error-output*' and exit non-zero."
  (format *error-output* "~&check-functional-coverage: ~?~%" format-control arguments)
  (uiop:quit 1))

(defun published-list ()
  "Return the published surface, as S4-1's emitter derives it.

The surface has one derivation in this repository -- `cl-gbdt/docgen''s -- and both this check
and docs/API-REFERENCE.md read it from there. A second computation of `what is published' is
exactly the second home this programme keeps refusing to create."
  (cl-gbdt/src/docgen/introspect:published-symbols
   cl-gbdt/src/docgen/emit:+public-packages+))

(defun functional-test-files ()
  "Return the functional suite's source files, sorted, so reporting is deterministic."
  (sort (directory (merge-pathnames "tests/functional/*.lisp" (uiop:getcwd)))
        #'string< :key #'namestring))

(defun package-definition-form-p (form)
  "True when FORM is a `defpackage' or `uiop:define-package'."
  (and (consp form)
       (symbolp (first form))
       (member (symbol-name (first form)) '("DEFPACKAGE" "DEFINE-PACKAGE") :test #'string=)))

(defun functional-references ()
  "Return a hash table whose keys are the symbols the functional suite's sources name.

Reading with the standard reader, rather than searching text, is what makes a name inside a
comment not count. Excluding each file's own package form is what makes a name that appears only
in an `:import-from' clause not count: importing a symbol is not testing it. The whole file
otherwise counts, not only its `deftest' forms -- these files define fixtures and helpers as
top-level forms, and `make-csr-matrix' is reached only that way."
  (let ((referenced (make-hash-table)))
    (labels ((walk (form)
               (cond ((symbolp form) (setf (gethash form referenced) t))
                     ((consp form) (walk (car form)) (walk (cdr form)))
                     ;; ARRAYP rather than VECTORP so a rank-2+ literal (#2A(...), which a
                     ;; fixture could conceivably read as data) is descended too, and NOT
                     ;; STRINGP so a string is not walked one character at a time -- a
                     ;; character is never a symbol, so that walk was harmless, just wasted
                     ;; work on every string literal in every file.
                     ((and (arrayp form) (not (stringp form)))
                      (dotimes (i (array-total-size form))
                        (walk (row-major-aref form i)))))))
      (dolist (path (functional-test-files) referenced)
        (with-open-file (in path)
          (let ((*package* (find-package "CL-USER"))
                (*read-eval* nil))
            (handler-case
                (loop for form = (read in nil :eof)
                      until (eq form :eof)
                      do (when (and (consp form) (eq (first form) 'cl:in-package))
                           (let ((package (find-package (string (second form)))))
                             (unless package
                               (die "~A names package ~A, which does not exist. Is ~
cl-gbdt/tests/functional loaded?" path (second form)))
                             (setf *package* package)))
                         (unless (package-definition-form-p form)
                           (walk form)))
              (error (condition)
                (die "cannot read ~A: ~A" path condition)))))))))

(defparameter +coverage-file+ "docs/FUNCTIONAL-COVERAGE.md"
  "The classification this check reads, relative to the repository root.")

(defparameter +unproven-heading+ "## Unproven"
  "The one literal heading whose rows are gaps rather than decisions.")

(defparameter +exempt-heading-prefix+ "## Exempt"
  "Every heading whose rows are decisions begins with this.

A prefix rather than a list, so a section can carry its own argument in its heading -- which is
where an exemption's reason belongs, per this file's own convention.")

(defstruct (row (:constructor %make-row))
  "One row of the classification: which section it sits in, what it names, and where."
  (section nil :read-only t)
  (kind nil :read-only t)
  (name nil :read-only t)
  (line nil :read-only t))

(defun table-row-name (line line-number)
  "Return the backquoted symbol name in LINE's first cell, or NIL when LINE is not a data row.

A data row looks like `| `cl-gbdt:predict` | note |'. The header row's first cell is the
literal `Symbol', and the separator row's first cell holds only `-' and `:' characters (the
markdown alignment row, `|---|---|' or `|:---|---:|'); both return NIL, which is how they are
skipped without a special case. Any other `|'-leading line is a data row whose first cell must
be backquoted -- a first cell that lost its closing backtick would otherwise look exactly like
one of those two skip cases and vanish silently, rather than failing the stale-row check (if
the suite now covers its symbol) or the unclassified check (if nothing does). LINE-NUMBER is
used only to name the row in that error."
  (let ((trimmed (string-trim " " line)))
    (when (and (plusp (length trimmed)) (char= (char trimmed 0) #\|))
      (let* ((cell (string-trim " " (subseq trimmed 1 (position #\| trimmed :start 1))))
             (length (length cell)))
        (cond ((and (>= length 3) (char= (char cell 0) #\`) (char= (char cell (1- length)) #\`))
               (subseq cell 1 (1- length)))
              ((string= cell "Symbol") nil)
              ((and (plusp (length cell))
                    (every (lambda (c) (member c '(#\- #\:))) cell))
               nil)
              (t (die "~A:~D: a table row's first cell, ~S, is not backquoted like `` `foo` ``."
                       +coverage-file+ line-number cell)))))))

(defun parse-coverage-file ()
  "Return the classification's rows, in file order.

A row before the file's first `## ' heading, or under a heading this checker does not recognise,
is an error rather than a silently ignored line: a renamed heading is exactly how a row goes
quiet, which is the failure ffi-spec/BINDING-COVERAGE.md's own history warns about.

Only a `## ' (exactly two hashes) line is read as a heading; a `### ' sub-heading is
deliberately left unenforced rather than rejected. The file has none today -- a fresh check
before this docstring was written found no line starting `### ' -- and a stray one changes
nothing this checker guarantees: its rows still resolve to the nearest enclosing `## ' section,
still get classified, and still block an unclassified symbol exactly as they would without it.
What a `### ' would misrepresent is which EXEMPT REASON a row falls under, a prose-accuracy
question for a human reviewer, not the one thing this build step checks -- so there is nothing
here for the script to enforce until a real `### ' heading exists to get it wrong."
  (let ((rows '())
        (section nil)
        (kind nil)
        (line-number 0))
    (with-open-file (in (merge-pathnames +coverage-file+ (uiop:getcwd))
                        :if-does-not-exist nil)
      (unless in
        (die "~A does not exist." +coverage-file+))
      (loop for line = (read-line in nil nil)
            while line
            do (incf line-number)
               (cond ((and (>= (length line) 3) (string= "## " (subseq line 0 3)))
                      (setf section (string-right-trim '(#\Space #\Tab) line))
                      (setf kind (cond ((string= section +unproven-heading+) :unproven)
                                       ((and (>= (length section) (length +exempt-heading-prefix+))
                                             (string= +exempt-heading-prefix+
                                                      (subseq section 0
                                                              (length +exempt-heading-prefix+))))
                                        :exempt)
                                       (t nil))))
                     (t (let ((name (table-row-name line line-number)))
                          (when name
                            (unless section
                              (die "~A:~D: a row before the file's first heading: ~A"
                                   +coverage-file+ line-number name))
                            (unless kind
                              (die "~A:~D: the row ~A sits under ~S, which is neither ~S nor a ~
heading beginning ~S."
                                   +coverage-file+ line-number name section
                                   +unproven-heading+ +exempt-heading-prefix+))
                            (push (%make-row :section section :kind kind :name name
                                             :line line-number)
                                  rows)))))))
    (nreverse rows)))

(defun resolve-row-symbol (name line qualifiers)
  "Return the published symbol NAME denotes, or die naming the row that is wrong.

NAME is `package:symbol'. The package must be one of the three public ones and must export the
name -- a row that names an unpublished symbol is a typo, or an export that went away and took
its row's meaning with it. QUALIFIERS is a hash table from symbol to that symbol's own
canonical qualifier (`cl-gbdt/src/docgen/introspect:published-qualifier', the same derivation
docs/API-REFERENCE.md's own heading uses): NAME's package must match it exactly. Seventy-two of
the published symbols export from more than one public package, so without this check a row
could name a symbol by ANY package that happens to export it and still resolve -- passing every
other check here while drifting from the one heading docs/API-REFERENCE.md actually gives that
entry."
  (let ((colon (position #\: name)))
    (unless colon
      (die "~A:~D: ~A is not a qualified name; rows read `package:symbol'."
           +coverage-file+ line name))
    (let* ((package-name (subseq name 0 colon))
           (symbol-name (subseq name (1+ colon)))
           (package (find-package (string-upcase package-name))))
      (when (and (plusp (length symbol-name)) (char= (char symbol-name 0) #\:))
        (die "~A:~D: ~A uses a double colon; rows name published symbols only."
             +coverage-file+ line name))
      (unless (member package-name cl-gbdt/src/docgen/emit:+public-packages+ :test #'string-equal)
        (die "~A:~D: ~A names ~A, which is not one of the public packages."
             +coverage-file+ line name package-name))
      (multiple-value-bind (symbol status) (find-symbol (string-upcase symbol-name) package)
        (unless (eq status :external)
          (die "~A:~D: ~A is not exported from ~A." +coverage-file+ line name package-name))
        (let ((canonical (gethash symbol qualifiers)))
          (unless (string-equal package-name canonical)
            (die "~A:~D: ~A names its symbol by a package that exports it, but not its ~
canonical one; docs/API-REFERENCE.md heads this entry `~A:~(~A~)`, so this row should too."
                 +coverage-file+ line name canonical (symbol-name symbol))))
        symbol))))

(defun qualifier-table (published)
  "Return a hash table mapping each PUBLISHED item's symbol to its own canonical qualifier.

The same derivation `cl-gbdt/src/docgen/emit''s own `qualifier-index' makes for the API
reference, rebuilt here rather than shared: that function is internal to `emit.lisp', and this
script's only use for it is the one lookup `resolve-row-symbol' needs."
  (let ((table (make-hash-table)))
    (dolist (item published table)
      (setf (gethash (cl-gbdt/src/docgen/introspect:published-symbol item) table)
            (cl-gbdt/src/docgen/introspect:published-qualifier item)))))

(defun check-classification (published referenced rows)
  "Run the five structural checks over the classification, dying on the first that fails."
  (let ((classified (make-hash-table))
        (seen (make-hash-table))
        (qualifiers (qualifier-table published)))
    ;; 4. a duplicate row, and 3. a row naming an unpublished symbol or the wrong package for
    ;; it (resolve-row-symbol dies on either).
    (dolist (row rows)
      (let ((symbol (resolve-row-symbol (row-name row) (row-line row) qualifiers)))
        (let ((previous (gethash symbol seen)))
          (when previous
            (die "~A:~D: ~A is already classified at line ~D."
                 +coverage-file+ (row-line row) (row-name row) previous)))
        (setf (gethash symbol seen) (row-line row))
        (setf (gethash symbol classified) row)))
    ;; 2. a stale row: the suite now names this symbol, so the row outlived its own reason.
    (let ((stale (remove-if-not (lambda (symbol) (gethash symbol referenced))
                                (loop for symbol being the hash-keys of classified
                                      collect symbol))))
      (when stale
        ;; The second row~P below must be a fresh ~P consuming its own argument, not ~:P backing
        ;; up: after ~:[~;s~] above consumes the boolean, a colon-P here would back up onto THAT
        ;; boolean rather than the count, printing "rows" even when exactly one row is stale.
        (die "~D row~:P name~:[~;s~] a symbol the functional suite now references; delete the ~
row~P: ~{~%  ~A:~D ~A~}"
             (length stale) (= 1 (length stale)) (length stale)
             (loop for symbol in (sort stale #'string< :key #'symbol-name)
                   for row = (gethash symbol classified)
                   append (list +coverage-file+ (row-line row) (row-name row))))))
    ;; 1. every published symbol is covered or classified.
    (let ((unclassified (remove-if (lambda (entry)
                                     (let ((symbol (cl-gbdt/src/docgen/introspect:published-symbol
                                                    entry)))
                                       (or (gethash symbol referenced)
                                           (gethash symbol classified))))
                                   published)))
      (when unclassified
        (die "~D published symbol~:P ~:[are~;is~] neither referenced by the functional suite nor ~
classified in ~A: ~{~%  ~A~}"
             (length unclassified) (= 1 (length unclassified)) +coverage-file+
             (mapcar (lambda (entry)
                       (format nil "~A:~(~A~)"
                               (cl-gbdt/src/docgen/introspect:published-qualifier entry)
                               (symbol-name
                                (cl-gbdt/src/docgen/introspect:published-symbol entry))))
                     unclassified))))
    classified))

(defparameter +minimum-covered+ 87
  "The fewest published symbols the functional suite may reference.

A floor because the two together are invisible to every other check here: delete an export AND its
row, or delete a test AND add a row, and the file stays self-consistent while the surface's proof
shrank. Raise it when the covered count rises; lowering it is a deliberate act that needs an
argument in the pull request. Re-measure rather than trusting this comment.")

(defparameter +maximum-unproven+ 33
  "The most rows `## Unproven' may hold.

This is the ratchet. Publishing a symbol with no functional test forces a new `## Unproven' row,
which forces raising this constant -- a visible, reviewable act rather than a quiet one. The list
may shrink freely, and lowering this constant as it shrinks is what keeps the ratchet tight.
Set from the count Task 7 measured, not from this comment.

Raised from 29 to 33 by file-input-layer-1's Task 2, which publishes `file-format-mismatch' and
its three readers with no functional test yet -- Task 3/4's `create-dataset-from-file', the
operation that signals it, does not exist until then.")

;;; check-api-reference.lisp and check-binding-coverage.lisp both end in an anonymous top-level
;;; form. This driver is named instead: Task 8 has to come back and add two floor checks to it,
;;; and a named `defun' is a form `lisp-edit-form' can target structurally, where an anonymous
;;; top-level form is not -- S4-2's own Task 1 hit exactly that limit trying to remove its own
;;; scratch reporting form and had to fall back to `rm'/`fs-write-file' on an untracked file, a
;;; workaround that would not extend to editing an already-committed one.

(defun main ()
  "Run every check and report the result; exit non-zero on the first failure."
  (let* ((published (published-list))
         (referenced (functional-references))
         (rows (parse-coverage-file))
         (classified (check-classification published referenced rows))
         (covered (- (length published) (hash-table-count classified)))
         (unproven (count :unproven rows :key #'row-kind))
         (exempt (count :exempt rows :key #'row-kind)))
    (when (< covered +minimum-covered+)
      (die "the functional suite references ~D published symbols, below the floor of ~D. If an ~
export or a test was removed on purpose, lower +minimum-covered+ in this file and say why in the ~
pull request." covered +minimum-covered+))
    (when (> unproven +maximum-unproven+)
      (die "~A holds ~D unproven rows, above the ceiling of ~D. A new published symbol without ~
a functional test has to raise +maximum-unproven+ in this file, which is the point: the work ~
list does not grow quietly." +coverage-file+ unproven +maximum-unproven+))
    (format t "~&published symbols: ~D~%covered by the functional suite: ~D~%classified: ~D ~
(~D unproven, ~D exempt)~%"
            (length published) covered (hash-table-count classified) unproven exempt)
    (format t "check-functional-coverage: every published symbol has a position~%")))

(main)
