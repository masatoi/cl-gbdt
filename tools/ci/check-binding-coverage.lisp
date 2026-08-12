;;;; check-binding-coverage.lisp --- Every generated C binding is wrapped, planned, or
;;;; excluded; an unclassified one fails the build.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-binding-coverage.lisp
;;;;
;;;; ffi-spec/BINDING-COVERAGE.md is this project's record that every C function emitted
;;;; into src/*/c-api.lisp -- 177 across both backends today -- has a recorded position:
;;;; `wrapped' (a backend's native.lisp imports it), `planned' (a row under that file's
;;;; single `## Planned' heading), or `excluded' (a row under one of its `## Excluded —
;;;; <reason>' headings, the heading itself carrying the argument). That file's own header
;;;; says coverage there is guaranteed by classification, not by a percentage; this script
;;;; is the enforcement of that promise -- an unclassified `defcfun' is a build failure.
;;;;
;;;; PORTED, VERBATIM, FROM tools/ci/check-abi-blacklist.lisp
;;;;
;;;; `read-top-level-forms', `symbol-name-string=', `table-row-first-cell',
;;;; `backticked-name', `defcfun-form-p', `defcfun-c-and-lisp-names', `c-name-map',
;;;; `define-package-form', `import-from-names' and `backend-imports' are that file's own
;;;; functions, unchanged -- see its header for what each does and why it is subtle. This
;;;; project's established shape is to port shared C-facing parsing into each standalone
;;;; checker rather than factor it into a library both load (tools/check-upstream.lisp's
;;;; own header states the same choice explicitly, about these same functions); the
;;;; duplication here is deliberate, not an oversight.
;;;;
;;;; WHAT IS NOT PORTED, AND WHY
;;;;
;;;; `blacklisted-names' is ported too, for CHECK D below, but is the one function on the
;;;; list above that is NOT what does this file's own classification parse. Its own
;;;; docstring says its parse is "independent ... from which of the two tables a row is
;;;; in" -- it returns a flat list and deliberately discards the section a row came from.
;;;; This checker's own parse cannot discard that: the section is the only thing that
;;;; tells a `planned' row from an `excluded' one, and an excluded row's reason lives in
;;;; its heading rather than in the row itself. `classified-entries' below is the new
;;;; traversal that keeps it: same row parser (`table-row-first-cell' + `backticked-name'),
;;;; new section tracking.
;;;;
;;;; WHAT THIS CHECKS
;;;;
;;;; 1. Two floors, checked first and using a directory glob rather than a fixed path list,
;;;;    so that running this script from the wrong directory reports a floor violation
;;;;    instead of an unhandled file-system error: fewer than two `c-api.lisp' files found,
;;;;    or fewer than 150 `defcfun' forms in total across them. 150 is below today's 177
;;;;    and far above zero -- it catches a glob that matched one file instead of two,
;;;;    without failing the day someone regenerates against a smaller upstream header.
;;;; 2. Empty-parse guard: `classified-entries' finding zero rows in
;;;;    ffi-spec/BINDING-COVERAGE.md fails before any of the checks below run, worded like
;;;;    check-abi-blacklist.lisp's own -- a check that silently matches nothing is worse
;;;;    than no check. A missing file reaches this guard too, rather than signalling a raw
;;;;    file-system error: `classified-entries' opens with `:if-does-not-exist nil'.
;;;; 3. Section-vocabulary check: every section `classified-entries' recorded must be
;;;;    literally `Planned' or begin `Excluded'; anything else fails, naming the offending
;;;;    heading. This guards `planned-section-p''s own literal comparison -- without it, a
;;;;    row filed under a heading this checker does not recognise (`## Planned for 4.0',
;;;;    say, or one row that precedes the file's first `## ' heading at all, leaving it with
;;;;    no section) would fall through to CHECK D's `excluded-names' bucket by default,
;;;;    silently reclassifying rather than failing anything.
;;;; 4. CHECK A -- every `defcfun' C name across both backends is wrapped, planned, or
;;;;    excluded. This is the point of the whole file; an unclassified name fails, named
;;;;    individually.
;;;; 5. CHECK B -- every classified name (planned or excluded) is a real `defcfun' on some
;;;;    backend. One that is not fails: a typo in the row, or a function upstream removed
;;;;    while the row stayed behind.
;;;; 6. CHECK C -- no classified name is also wrapped. One that is fails: someone wrapped
;;;;    it and left the row behind -- the failure ffi-spec/BINDING-COVERAGE.md's own header
;;;;    names as the one this file is most likely to develop.
;;;; 7. CHECK D -- every function ffi-spec/ABI-BLACKLIST.md's "still present in the
;;;;    generated bindings" table names, intersected with the `defcfun' names actually
;;;;    emitted, is classified `excluded' here. That intersection needs no awareness of
;;;;    which of ABI-BLACKLIST.md's two tables a name is in: its two tables are "still
;;;;    present" and "moot -- removed upstream, never emitted", so intersecting the full
;;;;    name list (`blacklisted-names', reused as-is) against the emitted `defcfun' set
;;;;    already is the still-present set -- a function moving between those two tables
;;;;    needs no edit here. `blacklisted-names' itself gets an empty-parse guard at its own
;;;;    call site, on the same reasoning as item 2 above, since CHECK D depends on it being
;;;;    non-empty.
;;;;
;;;; Like check-abi-blacklist.lisp, every file here is read as data via `read', never
;;;; loaded or evaluated: nothing in src/*/c-api.lisp or src/*/native.lisp runs, and no
;;;; foreign library opens.
;;;;
;;;; WHAT THIS CANNOT CATCH
;;;;
;;;;   - Whether a `planned' row's Note or an `excluded' section's heading argument is
;;;;     actually a good one -- this checks presence and non-overlap, not the prose.
;;;;   - A call made without going through an `:import-from' clause -- see
;;;;     tools/ci/check-abi-blacklist.lisp's own "WHAT THIS CANNOT CATCH" for the same
;;;;     caveat; this script computes the wrapped set the identical way.
;;;;   - A name removed from ABI-BLACKLIST.md entirely rather than moved between its two
;;;;     tables -- CHECK D only ever reads names still present in that file's tables; one
;;;;     deleted from both stops being an obligation this check enforces, silently.
;;;;   - Whether the *right* heading covers a given excluded row, only that it is
;;;;     *recognisable* as one -- `classified-entries' records whichever `## ' heading
;;;;     precedes a row most recently, however the surrounding prose grouped it, and the
;;;;     section-vocabulary check (item 3 above) only requires that heading begin
;;;;     `Excluded'. A row filed under the wrong *particular* exclusion heading -- CUDA
;;;;     reasoning attached to an Arrow-only function, say -- passes just as cleanly as the
;;;;     right one.

(require :asdf)

;;; `read' below never evaluates anything, but it still needs the CFFI package to *exist*
;;; to intern the `cffi:...' symbols c-api.lisp and native.lisp reference as data -- see
;;; tools/ci/check-abi-blacklist.lisp's header for the identical note.
(ql:quickload "cffi" :silent t)

(defparameter +coverage-path+ "ffi-spec/BINDING-COVERAGE.md"
  "Path, relative to the repository root, of the classification record this check
enforces.")

(defparameter +blacklist-path+ "ffi-spec/ABI-BLACKLIST.md"
  "Path, relative to the repository root, of the record CHECK D cross-checks against.")

(defparameter +c-api-glob+ "src/*/c-api.lisp"
  "Glob, relative to the repository root, for every backend's generated bindings file.
Used for the floor checks below instead of a fixed per-backend list (contrast
+BACKENDS+ further down, and tools/ci/check-abi-blacklist.lisp's own): `directory' on a
glob that matches nothing returns NIL rather than signalling, so running this script from
the wrong directory -- ffi-spec/, say -- trips the file-count floor with a clear message
instead of an unhandled file-system error the moment some fixed path is opened.")

(defparameter +minimum-c-api-files+ 2
  "Floor on how many `c-api.lisp' files +C-API-GLOB+ must find. Below today's two backends
only if a backend's generated file has gone missing, or this script was run from the
wrong directory.")

(defparameter +minimum-defcfun-count+ 150
  "Floor on the total `cffi:defcfun' forms found across every `c-api.lisp' file. Below
today's 177 and far above zero: catches a glob that matched one file instead of two,
without failing the day someone regenerates against a smaller upstream header.")

(defparameter +backends+
  '((:lightgbm "src/lightgbm/c-api.lisp" "CL-GBDT/SRC/LIGHTGBM/C-API"
               "src/lightgbm/native.lisp")
    (:xgboost "src/xgboost/c-api.lisp" "CL-GBDT/SRC/XGBOOST/C-API"
              "src/xgboost/native.lisp"))
  "(backend-id c-api-path c-api-package-name backend-path) for each backend, all paths
relative to the repository root -- the identical shape and identical paths
tools/ci/check-abi-blacklist.lisp's own +BACKENDS+ declares, for the same reason: see
that parameter's docstring for why both fourth elements name a `native.lisp'.")

;;; ---- Reading Lisp source as data, never as code (ported verbatim; see header) ----

(defun read-top-level-forms (path)
  "Read PATH's top-level forms as data and return them as a list."
  (let ((*package* (or (find-package '#:cl-gbdt/tools/ci/binding-coverage-scratch)
                        (make-package '#:cl-gbdt/tools/ci/binding-coverage-scratch
                                       :use '(#:cl))))
        (*read-eval* nil))
    (with-open-file (in path)
      (loop :for form := (read in nil :eof)
            :until (eq form :eof)
            :collect form))))

(defun symbol-name-string= (symbol name)
  "True when SYMBOL's name is the string NAME, regardless of SYMBOL's package."
  (and (symbolp symbol) (string= (symbol-name symbol) name)))

(defun table-row-first-cell (line)
  "Return the trimmed first cell of LINE if it looks like a markdown table row -- starts
with `|' once leading whitespace is trimmed -- or NIL otherwise.

This is what makes the parse independent of the prose around the tables: a heading, a
paragraph, or a blank line never starts with `|' after trimming, so only genuine table
rows (including the header and separator rows, filtered out below by having no
backticks) reach `backticked-name'."
  (let ((trimmed (string-trim '(#\Space #\Tab) line)))
    (when (and (plusp (length trimmed)) (char= (char trimmed 0) #\|))
      (let* ((rest (subseq trimmed 1))
             (bar (position #\| rest)))
        (string-trim '(#\Space #\Tab) (if bar (subseq rest 0 bar) rest))))))

(defun backticked-name (cell)
  "Return the text between the first pair of backticks in CELL, or NIL if CELL has fewer
than two backticks. Applied only to a table row's *first* cell (see
`table-row-first-cell'), so this never picks up a backticked replacement function named
in some other column."
  (let ((start (position #\` cell)))
    (when start
      (let ((end (position #\` cell :start (1+ start))))
        (when end
          (subseq cell (1+ start) end))))))

(defun defcfun-form-p (form)
  "True when FORM is a top-level `(cffi:defcfun ...)' form."
  (and (consp form) (symbol-name-string= (car form) "DEFCFUN")))

(defun defcfun-c-and-lisp-names (form)
  "Return (VALUES C-NAME LISP-NAME) for a `cffi:defcfun' FORM naming both the C symbol
and the Lisp function, i.e. `(cffi:defcfun (\"C_Name\" lisp-name) ...)' -- the only form
every function in src/*/c-api.lisp uses (verified: every `cffi:defcfun' in both files has
a two-element name spec, never the single-string form that derives the Lisp name from the
C name automatically). Returns NIL for any form that is not shaped this way, so a future
regeneration that starts using the single-string form fails this check's mapping instead
of silently mismatching."
  (let ((spec (second form)))
    (when (and (consp spec) (stringp (first spec)) (symbolp (second spec)))
      (values (first spec) (symbol-name (second spec))))))

(defun c-name-map (path)
  "Return a hash table mapping every Lisp function name (string, upper-case, as `read'
produces) defined by a `cffi:defcfun' in PATH to its C name."
  (let ((table (make-hash-table :test #'equal)))
    (dolist (form (read-top-level-forms path) table)
      (when (defcfun-form-p form)
        (multiple-value-bind (c-name lisp-name) (defcfun-c-and-lisp-names form)
          (when lisp-name
            (setf (gethash lisp-name table) c-name)))))))

(defun define-package-form (forms)
  "Return the `(uiop:define-package ...)' form in FORMS, or NIL if none is present."
  (find-if (lambda (form) (and (consp form) (symbol-name-string= (car form) "DEFINE-PACKAGE")))
           forms))

(defun import-from-names (define-package-form target-package-name)
  "Return the list of imported symbol-name strings (upper-case) from DEFINE-PACKAGE-FORM's
`(:import-from #:TARGET-PACKAGE-NAME ...)' clause, or NIL if no clause names that package."
  (let ((clause (find-if (lambda (c)
                            (and (consp c)
                                 (symbol-name-string= (car c) "IMPORT-FROM")
                                 (consp (cdr c))
                                 (symbolp (second c))
                                 (string= (symbol-name (second c)) target-package-name)))
                          (cddr define-package-form))))
    (mapcar #'symbol-name (cddr clause))))

(defun backend-imports (backend-path c-api-package-name)
  "Return the list of Lisp names BACKEND-PATH imports from C-API-PACKAGE-NAME, via its
`uiop:define-package' form's matching `:import-from' clause."
  (let* ((forms (read-top-level-forms backend-path))
         (form (define-package-form forms)))
    (unless form
      (error "no uiop:define-package form found in ~A" backend-path))
    (import-from-names form c-api-package-name)))

(defun blacklisted-names (path)
  "Return the list of backtick-quoted C function names in every markdown table row's
first column across all of PATH, in file order. See this file's header for the parse's
independence from surrounding prose and from which of the two tables a row is in.

Ported verbatim from tools/ci/check-abi-blacklist.lisp; used here only for CHECK D, which
needs the union of ABI-BLACKLIST.md's two tables and nothing about which row is in
which -- see this file's header."
  (with-open-file (in path)
    (loop :for line := (read-line in nil)
          :while line
          :for cell := (table-row-first-cell line)
          :for name := (and cell (backticked-name cell))
          :when name :collect name)))

;;; ---- New: the classification parse, keeping the section `blacklisted-names' discards ----

(defun classified-entries (path)
  "Return a list of (NAME . SECTION) for every backtick-quoted C function name in a markdown
table row's first column across all of PATH, in file order.

Where `blacklisted-names' above deliberately discards which section a row came from --
that check treats both of its tables alike -- this one cannot: the section heading is the
whole of what distinguishes a planned entry from an excluded one, and an excluded entry's
heading is also where its reason is written. So the row parse is `blacklisted-names''s
own, and the traversal around it is new.

Opens with `:if-does-not-exist nil', unlike `blacklisted-names': a missing PATH then
returns NIL rather than signalling, so the empty-parse guard at this file's call site can
honestly say a missing file is one of the reasons it found nothing, instead of that case
being unreachable behind a raw file-system error."
  (with-open-file (in path :if-does-not-exist nil)
    (when in
      (loop :with section := nil
            :for line := (read-line in nil)
            :while line
            :for trimmed := (string-trim '(#\Space #\Tab) line)
            :when (and (> (length trimmed) 3) (string= "## " (subseq trimmed 0 3)))
              :do (setf section (string-trim '(#\Space #\Tab) (subseq trimmed 3)))
            :when (let ((cell (table-row-first-cell line)))
                    (and cell (backticked-name cell)))
              :collect (cons (backticked-name (table-row-first-cell line)) section)))))

(defun planned-section-p (section)
  "True when SECTION is the planned heading rather than an exclusion heading.

Compared against the literal heading rather than by prefix, so a future `## Planned for 4.0'
is a section this check does not silently treat as planned."
  (and section (string= section "Planned")))

(defun excluded-section-p (section)
  "True when SECTION begins with the literal prefix `Excluded' -- the family of headings
`## Excluded — <reason>' uses, whatever reason follows. NIL (a row with no `## ' heading
above it at all) is not: `(and section ...)' short-circuits on that case before `string='
would need a start/end past SECTION's own length."
  (and section
       (<= (length "Excluded") (length section))
       (string= "Excluded" section :end2 (length "Excluded"))))

(defun known-section-p (section)
  "True when SECTION is a section this checker's vocabulary recognises: literally
`Planned' (per `planned-section-p') or beginning `Excluded' (per `excluded-section-p').
Anything else -- including NIL -- is not."
  (or (planned-section-p section) (excluded-section-p section)))

(defun check-section-vocabulary (entries)
  "Fail on every section among ENTRIES (a `classified-entries' result) that
`known-section-p' does not recognise, naming each offending heading once. Returns the
offending sections, deduplicated.

Exists because `planned-section-p''s comparison is deliberately literal, not a prefix
match (see its own docstring) -- but that literalness only does its job if something
fails loudly when a row's actual heading does not match it. Without this check, a
renamed or new heading `known-section-p' has never seen -- `## Planned for 4.0', `##
Deferred', or a row that precedes the file's first `## ' heading and so carries no
section at all -- does not fail CHECK D's own comparison against `excluded-names' either:
`excluded-names' is built as \"every classified name whose section is not the planned
one\", so any unrecognised section falls into it by default. A still-emitted blacklisted
function filed under such a heading would then read as `excluded' to CHECK D and the
build would stay green, which is exactly the silent reclassification this check exists to
turn into a failure instead."
  (let ((bad (remove-duplicates
              (loop :for (nil . section) :in entries
                    :unless (known-section-p section)
                      :collect section)
              :test #'equal)))
    (dolist (section bad)
      (format *error-output*
              "~&FAIL ~A is not a section this checker recognises -- must be literally ~
               \"Planned\" or begin \"Excluded\". Rename the heading, or move the row.~%"
              (if section (format nil "~S" section)
                  "a row with no `## ' heading above it")))
    bad))

;;; ---- The wrapped set, and the four checks ----

(defun wrapped-c-names (name-map c-api-package-name backend-path)
  "Return the list of C function names BACKEND-PATH's `native.lisp' actually wraps --
every name it imports from C-API-PACKAGE-NAME that resolves, through NAME-MAP (a
`c-name-map' result, the caller's to build and reuse rather than this function's to
re-derive), to a C name. An import that does not resolve is silently skipped: it names a
constant, not a function, and distinguishing that from a genuine typo is
tools/ci/check-abi-blacklist.lisp's job, already run as its own gate; this checker only
needs the C names that DO resolve."
  (let ((imports (backend-imports backend-path c-api-package-name)))
    (loop :for lisp-name :in imports
          :for c-name := (gethash lisp-name name-map)
          :when c-name :collect c-name)))

(defun check-a-unclassified (backend-id backend-c-names wrapped classified)
  "CHECK A: report every C name in BACKEND-C-NAMES that is neither in WRAPPED nor in
CLASSIFIED (the flat list of every planned-or-excluded name from `classified-entries').
This is the point of the whole file: an unclassified `defcfun' fails the build. Returns
the list of unclassified names."
  (let ((unclassified (remove-if (lambda (name)
                                    (or (member name wrapped :test #'string=)
                                        (member name classified :test #'string=)))
                                  backend-c-names)))
    (dolist (name unclassified)
      (format *error-output*
              "~&FAIL ~(~A~): ~A is a defcfun with no classification -- not wrapped, not ~
               planned, not excluded. Add a row to ~A.~%"
              backend-id name +coverage-path+))
    unclassified))

(defun check-b-unresolved (entries all-c-names)
  "CHECK B: report every classified name in ENTRIES (from `classified-entries') that is
not a real `defcfun' C name in ALL-C-NAMES, the union across both backends. Returns the
offending entries."
  (let ((bad (remove-if (lambda (entry) (member (car entry) all-c-names :test #'string=))
                         entries)))
    (dolist (entry bad)
      (format *error-output*
              "~&FAIL ~A (section ~S in ~A) names no real cffi:defcfun on either backend -- ~
               a typo in the row, or a function upstream has removed while the row stayed.~%"
              (car entry) (cdr entry) +coverage-path+))
    bad))

(defun check-c-wrapped-and-classified (entries all-wrapped)
  "CHECK C: report every classified name in ENTRIES that is also in ALL-WRAPPED, the union
of both backends' wrapped C names -- someone wrapped it and left the row behind. Returns
the offending entries."
  (let ((bad (remove-if-not (lambda (entry) (member (car entry) all-wrapped :test #'string=))
                             entries)))
    (dolist (entry bad)
      (format *error-output*
              "~&FAIL ~A (section ~S in ~A) is classified but is also wrapped -- remove the ~
               row, its function is already covered.~%"
              (car entry) (cdr entry) +coverage-path+))
    bad))

(defun check-d-blacklist-excluded (blacklist all-c-names excluded-names)
  "CHECK D: report every name in BLACKLIST (an ABI-BLACKLIST.md `blacklisted-names' result)
that is still emitted -- a member of ALL-C-NAMES -- but is not classified `excluded' --
absent from EXCLUDED-NAMES, the classified names whose section is not the planned one. See
this file's header for why intersecting BLACKLIST with ALL-C-NAMES already is the 'still
present' half of ABI-BLACKLIST.md, with no need to know which of its two tables a name
came from. Returns the offending names."
  (let* ((still-present (remove-if-not (lambda (name) (member name all-c-names :test #'string=))
                                        blacklist))
         (bad (remove-if (lambda (name) (member name excluded-names :test #'string=))
                          still-present)))
    (dolist (name bad)
      (format *error-output*
              "~&FAIL ~A is on the ABI blacklist (~A) and still emitted, but is not ~
               classified `excluded' in ~A.~%"
              name +blacklist-path+ +coverage-path+))
    bad))

;;; ---- Main check ----

(let* ((root (uiop:getcwd))
       (c-api-files (directory (merge-pathnames +c-api-glob+ root))))
  (format t "~&found ~D c-api.lisp file~:P matching ~A~%" (length c-api-files) +c-api-glob+)
  (when (< (length c-api-files) +minimum-c-api-files+)
    (format *error-output*
            "~&FAIL: found only ~D c-api.lisp file~:P matching ~A, fewer than the floor of ~
             ~D. This check must be run from the repository root; a backend's generated ~
             bindings file may also be missing.~%"
            (length c-api-files) +c-api-glob+ +minimum-c-api-files+)
    (uiop:quit 1))
  (let ((defcfun-total (reduce #'+ c-api-files
                                :key (lambda (path) (hash-table-count (c-name-map path))))))
    (format t "~&~D defcfun form~:P total across those file~:P~%" defcfun-total)
    (when (< defcfun-total +minimum-defcfun-count+)
      (format *error-output*
              "~&FAIL: found only ~D defcfun form~:P across ~D file~:P, fewer than the ~
               floor of ~D.~%"
              defcfun-total (length c-api-files) +minimum-defcfun-count+)
      (uiop:quit 1)))
  (let* ((coverage-file (merge-pathnames +coverage-path+ root))
         (entries (classified-entries coverage-file)))
    (format t "~&parsed ~D classified entr~:@P from ~A~%" (length entries) +coverage-path+)
    ;; Empty-parse guard: fails before CHECK A-D run, worded like
    ;; check-abi-blacklist.lisp's own -- a check that silently matches nothing is worse
    ;; than no check.
    (when (null entries)
      (format *error-output*
              "~&FAIL: parsed zero classified entries from ~A.~@
               A binding-coverage check that matches nothing is worse than no check at ~
               all -- this is the same failure shape as mallet silently not checking ~
               line length. Either the file is missing, empty, or its table format no ~
               longer matches what `classified-entries' parses (a backtick-quoted name ~
               in a markdown table row's first column).~%"
              +coverage-path+)
      (uiop:quit 1))
    ;; Section-vocabulary check: fails before CHECK A-D run, same reasoning as the
    ;; empty-parse guard just above -- see this file's header, item 3.
    (let ((bad-sections (check-section-vocabulary entries)))
      (when bad-sections
        (uiop:quit 1)))
    (let* ((classified-names (mapcar #'car entries))
           (excluded-names (mapcar #'car (remove-if (lambda (e) (planned-section-p (cdr e)))
                                                      entries)))
           (blacklist-file (merge-pathnames +blacklist-path+ root))
           ;; A missing ABI-BLACKLIST.md would otherwise crash inside `blacklisted-names'
           ;; itself (it is ported verbatim from check-abi-blacklist.lisp and does not
           ;; open with `:if-does-not-exist nil'); catching that here lets the guard just
           ;; below treat "missing" and "present but empty" alike, honestly.
           (blacklist (handler-case (blacklisted-names blacklist-file)
                        (file-error () nil)))
           (all-c-names '())
           (all-wrapped '())
           (unclassified-total 0))
      (format t "~&parsed ~D blacklisted function name~:P from ~A~%"
              (length blacklist) +blacklist-path+)
      ;; CHECK D depends on BLACKLIST being non-empty; an unguarded empty parse would
      ;; report "check D: 0 ..." having examined nothing, the same failure shape the
      ;; empty-parse guard above exists to prevent -- see this file's header, item 7.
      (when (null blacklist)
        (format *error-output*
                "~&FAIL: parsed zero blacklisted function names from ~A. CHECK D depends ~
                 on this list being non-empty, the same way check-abi-blacklist.lisp's ~
                 own empty-parse guard protects its identical parse. Either the file is ~
                 missing, empty, or its table format no longer matches what ~
                 `blacklisted-names' parses (a backtick-quoted name in a markdown table ~
                 row's first column).~%"
                +blacklist-path+)
        (uiop:quit 1))
      (dolist (spec +backends+)
        (destructuring-bind (id c-api-path c-api-package-name backend-path) spec
          (let* ((full-c-api-path (merge-pathnames c-api-path root))
                 (full-backend-path (merge-pathnames backend-path root))
                 (name-map (c-name-map full-c-api-path))
                 (backend-c-names (loop :for c-name :being :the :hash-values :of name-map
                                         :collect c-name))
                 (wrapped (wrapped-c-names name-map c-api-package-name full-backend-path))
                 (planned (remove-if-not
                           (lambda (e) (and (planned-section-p (cdr e))
                                             (member (car e) backend-c-names :test #'string=)))
                           entries))
                 (excluded (remove-if-not
                            (lambda (e) (and (not (planned-section-p (cdr e)))
                                              (member (car e) backend-c-names :test #'string=)))
                            entries))
                 (unclassified (check-a-unclassified id backend-c-names wrapped
                                                      classified-names)))
            (format t "~&~(~A~): ~D defcfun, ~D wrapped, ~D planned, ~D excluded, ~D ~
                       unclassified~%"
                    id (length backend-c-names) (length wrapped) (length planned)
                    (length excluded) (length unclassified))
            (incf unclassified-total (length unclassified))
            (setf all-c-names (append all-c-names backend-c-names))
            (setf all-wrapped (append all-wrapped wrapped)))))
      (let ((unresolved (check-b-unresolved entries all-c-names))
            (wrapped-and-classified (check-c-wrapped-and-classified entries all-wrapped))
            (blacklist-violations (check-d-blacklist-excluded blacklist all-c-names
                                                                excluded-names)))
        (format t "~&check B: ~D classified name~:P naming no real defcfun~%"
                (length unresolved))
        (format t "~&check C: ~D classified name~:P also wrapped~%"
                (length wrapped-and-classified))
        (format t "~&check D: ~D still-emitted blacklisted name~:P not classified excluded~%"
                (length blacklist-violations))
        (format t "~&~D binding~:P classified / ~D unclassified across ~D backend~:P~%"
                (- (length all-c-names) unclassified-total) unclassified-total
                (length +backends+))
        (uiop:quit (if (or (plusp unclassified-total) unresolved wrapped-and-classified
                            blacklist-violations)
                       1
                       0))))))
