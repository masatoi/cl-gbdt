;;;; check-abi-blacklist.lisp --- No backend imports a blacklisted C function, and every
;;;; backend import is declared in that backend's *required-symbols*.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-abi-blacklist.lisp
;;;;
;;;; Two independent checks live here, CHECK A and CHECK B below, because both start from
;;;; the identical piece of work: for each backend, reading `src/*/c-api.lisp' for every
;;;; `cffi:defcfun' form to build a Lisp-name -> C-name map, then reading the backend's
;;;; `backend.lisp' for the `:import-from' clause that names its c-api package and mapping
;;;; every imported symbol back to a C name through that table (see `check-backend').
;;;; Splitting them into two scripts would mean deriving that map twice from the same
;;;; source, in two places that could drift out of sync with each other -- precisely the
;;;; duplication this project's design doc (section 2) lists as a hazard: two
;;;; implementations of the same C-facing logic whose fixes need to land in both.
;;;;
;;;; CHECK A: no backend imports a blacklisted function
;;;;
;;;; ffi-spec/ABI-BLACKLIST.md records upstream C functions that changed meaning between
;;;; the reference implementations' vendored headers and the versions cl-gbdt targets,
;;;; while keeping the same symbol name and, sometimes, the same argument count. A call
;;;; through the generated CFFI bindings to one of these succeeds -- no undefined-symbol
;;;; error, no link error, no wrong-arity error -- and does the wrong thing. That file's
;;;; own opening line is "The only defence is to never call them." This is that defence
;;;; made mechanical: it fails the build if either backend ever imports one.
;;;;
;;;; Measurement recorded in docs/superpowers/specs/2026-08-06-abi-tracking-design.md
;;;; found that of the functions that changed or vanished across many upstream releases,
;;;; almost none touched the 38 functions the two backends actually import -- and the
;;;; near-exact exception set is this blacklist. The blacklist is not a list of hazards
;;;; to note for later; it is the thing that keeps the imported set stable, and this
;;;; check is what keeps that true instead of relying on human memory.
;;;;
;;;; CHECK B: no backend import escapes *required-symbols*
;;;;
;;;; Each backend declares a `*required-symbols*' list of C function names, checked with
;;;; `probe-foreign-symbols' immediately after the shared library loads (see
;;;; `initialize-backend' in each `backend.lisp'). A function the backend actually calls
;;;; but omits from that list is not probed: a library missing it, or exposing it with an
;;;; incompatible signature, opens successfully and only fails later, at the call site,
;;;; instead of loudly at `open-backend' via `missing-foreign-symbols'. Design doc section
;;;; 8 recorded exactly this gap for XGBoost -- `XGDMatrixSetUIntInfo' and
;;;; `XGBoosterGetNumFeature' were both called but neither was declared -- found once, by
;;;; inspection, which is the argument for checking it mechanically instead.
;;;;
;;;; This assumes every imported function is required -- true of both backends today, so
;;;; there is nothing to exempt. Design doc section 8 also describes an *optional* symbol
;;;; tier: a function whose absence should degrade one capability rather than fail
;;;; `open-backend' outright. Neither backend declares one, and this check does not invent
;;;; that tier -- doing so with no real optional symbol to test it against would mean
;;;; shipping the distinction unverified, the same failure shape as a check that has never
;;;; been seen to fail (see the empty-parse guard below). Section 7's capability work is
;;;; where that tier belongs; until then, "imported but not required" is unconditionally a
;;;; failure here.
;;;;
;;;; WHAT THIS CHECKS
;;;;
;;;; 1. Parses every markdown table row's first column in ABI-BLACKLIST.md for a
;;;;    backtick-quoted C function name (see `blacklisted-names'). Both tables in the
;;;;    file are ordinary rows to this parser -- it does not care which section a row is
;;;;    under, only that the line looks like `| \`Name\` | ... | ... |'.
;;;; 2. Fails loudly, before checking anything else, if that parse finds zero names (see
;;;;    the empty-parse guard below). A check that silently matches nothing is worse than
;;;;    no check: this project shipped that exact shape once, when a whole branch treated
;;;;    a clean `mallet' run as covering the 100-column rule, which `mallet' has never
;;;;    checked (see tools/ci/lint.lisp's own header).
;;;; 3. For each backend, builds the Lisp-name -> C-name map and the imported-name list
;;;;    described above (`check-backend'). CHECK A reports any import whose C name is on
;;;;    the blacklist. CHECK B separately reads the backend's `*required-symbols*'
;;;;    `defparameter' (`required-c-names') and reports any import whose C name is absent
;;;;    from it. The two reports are independent: an import can fail either, both, or
;;;;    neither.
;;;;
;;;; Like tools/ci/check-float-traps.lisp, every file here is read as data via `read',
;;;; never loaded or evaluated -- nothing in src/*/c-api.lisp or src/*/backend.lisp runs,
;;;; and no foreign library opens. This keeps the check usable from `tools/ci/lint.lisp'
;;;; and keeps layer 1's `foreign libraries open: NIL' invariant untouched by it.
;;;;
;;;; WHAT THIS CANNOT CATCH
;;;;
;;;;   - A blacklisted or undeclared call made without going through an `:import-from'
;;;;     clause -- e.g. a fully package-qualified call
;;;;     `cl-gbdt/src/lightgbm/c-api:lgbm-dataset-create-from-mats' written directly in a
;;;;     backend file. Every call in both backends today goes through `:import-from', by
;;;;     convention, but this script trusts that convention rather than scanning call
;;;;     sites for qualified symbols.
;;;;   - A backend file that does not follow the `src/<backend>/native.lisp' (both
;;;;     backends, since this branch's Task 2 and Task 3 splits) and `src/<backend>/c-api.lisp'
;;;;     layout named in +BACKENDS+ below -- a future backend added under a different
;;;;     convention needs that list extended, the same caveat +BACKEND-FILE-PATTERN+
;;;;     carries in check-float-traps.lisp and +LEAF-ROOTS+ carries in
;;;;     check-leaf-systems.lisp.
;;;;   - A new hazard upstream introduces that nobody has added to ABI-BLACKLIST.md yet
;;;;     (CHECK A). This script enforces the list; it does not maintain it. Design doc
;;;;     section 11 and this branch's Task 2 describe the drift-detection tool that
;;;;     maintenance needs.
;;;;   - Whether a name declared in `*required-symbols*' but no longer imported anywhere
;;;;     (CHECK B checks only the import -> required direction, not the reverse) is stale.
;;;;     A stale entry is over-cautious, not unsafe, so this does not flag it.
;;;;   - Whether the *right* symbols are optional versus required at all -- CHECK B only
;;;;     enforces the required-only assumption stated above; it cannot tell a genuinely
;;;;     optional capability from one nobody has split out yet.

(require :asdf)

;;; Every scanned c-api.lisp and backend.lisp file references `cffi:...' symbols. `read'
;;; below never evaluates anything, but it still needs the CFFI package to *exist* to
;;; intern those qualified symbols as data -- an unqualified bare symbol would intern
;;; fine into any package, but `cffi:defcfun' would signal "package CFFI does not exist"
;;; without this. ASDF's own load above already makes the UIOP package (for
;;; `uiop:define-package') available.
(ql:quickload "cffi" :silent t)

(defparameter +blacklist-path+ "ffi-spec/ABI-BLACKLIST.md"
  "Path, relative to the repository root, of the first-class blacklist record.")

(defparameter +backends+
  '((:lightgbm "src/lightgbm/c-api.lisp" "CL-GBDT/SRC/LIGHTGBM/C-API"
               "src/lightgbm/native.lisp")
    (:xgboost "src/xgboost/c-api.lisp" "CL-GBDT/SRC/XGBOOST/C-API"
              "src/xgboost/native.lisp"))
  "(backend-id c-api-path c-api-package-name backend-path) for each backend, all paths
relative to the repository root. See WHAT THIS CANNOT CATCH above for what adding a
backend that does not follow this layout would require.

Both fourth elements point at a `native.lisp', not a `backend.lisp' -- the Phase 1 split
(XGBoost in this branch's Task 2, LightGBM in Task 3) moved both `*required-symbols*'
and the `:import-from' clause naming the c-api package there, out of the single file
this list originally named for each backend in turn. `protocol.lisp', the other half of
each split, imports the c-api package from nowhere at all -- it calls no C function
directly, only `native.lisp''s wrappers -- so it has nothing for this check to read.")

;;; ---- Reading Lisp source as data, never as code ----
;;;
;;; Identical technique to tools/ci/check-float-traps.lisp's `read-top-level-forms': a
;;; throwaway package that uses only CL, so every bare symbol in these files interns
;;; harmlessly regardless of whether its own package is defined yet, and `*read-eval*'
;;; bound to NIL so a stray `#.' cannot run code either.

(defun read-top-level-forms (path)
  "Read PATH's top-level forms as data and return them as a list."
  (let ((*package* (or (find-package '#:cl-gbdt/tools/ci/abi-blacklist-scratch)
                        (make-package '#:cl-gbdt/tools/ci/abi-blacklist-scratch
                                       :use '(#:cl))))
        (*read-eval* nil))
    (with-open-file (in path)
      (loop :for form := (read in nil :eof)
            :until (eq form :eof)
            :collect form))))

(defun symbol-name-string= (symbol name)
  "True when SYMBOL's name is the string NAME, regardless of SYMBOL's package."
  (and (symbolp symbol) (string= (symbol-name symbol) name)))

;;; ---- Step 1: parsing ABI-BLACKLIST.md's tables ----

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

(defun blacklisted-names (path)
  "Return the list of backtick-quoted C function names in every markdown table row's
first column across all of PATH, in file order. See this file's header for the parse's
independence from surrounding prose and from which of the two tables a row is in."
  (with-open-file (in path)
    (loop :for line := (read-line in nil)
          :while line
          :for cell := (table-row-first-cell line)
          :for name := (and cell (backticked-name cell))
          :when name :collect name)))

;;; ---- Step 2: C-name -> Lisp-name map from a c-api.lisp file ----

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

(defun constant-names (path)
  "Return the names (string, upper-case) of every `defconstant' in PATH.

A backend imports constants such as `+c-api-dtype-float32+' from the same c-api package
it imports functions from, and those have no `cffi:defcfun' behind them. Without this
they would look exactly like an unresolved function name to `check-backend'."
  (let ((names '()))
    (dolist (form (read-top-level-forms path) names)
      (when (and (consp form)
                 (member (first form) '(defconstant cl:defconstant))
                 (symbolp (second form)))
        (push (symbol-name (second form)) names)))))

;;; ---- Step 3: the backend's :import-from clause ----

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

;;; ---- Step 4: the backend's *required-symbols* declaration (CHECK B) ----

(defun defparameter-form-p (form)
  "True when FORM is a top-level `(defparameter ...)' form."
  (and (consp form) (symbol-name-string= (car form) "DEFPARAMETER")))

(defun required-symbols-form (forms)
  "Return the `(defparameter *required-symbols* ...)' form in FORMS, or NIL if none is
present."
  (find-if (lambda (form)
             (and (defparameter-form-p form)
                  (symbol-name-string= (second form) "*REQUIRED-SYMBOLS*")))
           forms))

(defun required-c-names (backend-path)
  "Return the list of C function name strings BACKEND-PATH's `*required-symbols*'
declares -- what `probe-foreign-symbols' checks at `open-backend', per that parameter's
own docstring in each backend file.

Signals an error, rather than silently treating a name as uncovered, when the form is
missing or is not a quoted literal list of strings -- the same reasoning
`backend-imports' applies to a missing `uiop:define-package' form: a checker that cannot
find what it is supposed to check against must say so loudly, not report a coverage
result that happens to look like a real one (here, every import would look like a gap
against a required-set this function silently treated as empty)."
  (let* ((forms (read-top-level-forms backend-path))
         (form (required-symbols-form forms)))
    (unless form
      (error "no *required-symbols* defparameter found in ~A" backend-path))
    (let ((value-form (third form)))
      (unless (and (consp value-form) (symbol-name-string= (car value-form) "QUOTE"))
        (error "*required-symbols* in ~A is not a quoted literal list" backend-path))
      (mapcar (lambda (name)
                (unless (stringp name)
                  (error "non-string entry ~S in *required-symbols* in ~A" name backend-path))
                name)
              (second value-form)))))

;;; ---- Main check ----

(defun check-backend (id c-api-path c-api-package-name backend-path blacklist)
  "Return (VALUES BLACKLIST-VIOLATIONS COVERAGE-VIOLATIONS) for one backend spec from
+BACKENDS+, printing progress as it goes. Each is a list of (ID LISP-NAME C-NAME).

BLACKLIST-VIOLATIONS holds every import whose C name is on BLACKLIST -- CHECK A in this
file's header. COVERAGE-VIOLATIONS holds every import whose C name is absent from
BACKEND-PATH's `*required-symbols*' -- CHECK B. Both read from the same NAME-MAP,
CONSTANTS and IMPORTS computed once below, per this file's header note on why the two
checks share one parser rather than each deriving its own. They are otherwise
independent: a single import can fail either, both, or neither."
  (let ((name-map (c-name-map c-api-path))
        (constants (constant-names c-api-path))
        (imports (backend-imports backend-path c-api-package-name))
        (required (required-c-names backend-path))
        (blacklist-violations '())
        (coverage-violations '()))
    (format t "~&~A: ~D imported name~:P from ~A, ~D required symbol~:P declared~%"
            (enough-namestring backend-path (uiop:getcwd)) (length imports)
            (enough-namestring c-api-path (uiop:getcwd)) (length required))
    (dolist (lisp-name imports)
      (let ((c-name (gethash lisp-name name-map)))
        (cond
          ;; An imported name with no `defcfun' behind it is reported, not skipped. Nothing
          ;; else notices one: `:import-from' on a symbol the source package does not export
          ;; interns it silently rather than erroring, so the build stays green -- verified.
          ;; Skipping it would also blind CHECK A to the blacklist's "removed upstream, never
          ;; emitted" table, whose entries have no `defcfun' by definition and would therefore
          ;; never map. Treating "I could not resolve this" as "this is fine" is the same
          ;; failure this file's empty-parse guard exists to prevent. Not run through CHECK B
          ;; either -- there is no resolved C name to look up in *required-symbols*.
          ((member lisp-name constants :test #'string=))   ; a constant, not a function
          ((null c-name)
           (push (list id lisp-name "<unresolved>") blacklist-violations)
           (format *error-output*
                   "~&FAIL ~A imports ~A, which has no `cffi:defcfun' in ~A. Either it is ~
                    misspelled, or it names something removed upstream -- see the second ~
                    table in ~A.~%"
                   (enough-namestring backend-path (uiop:getcwd)) lisp-name
                   (enough-namestring c-api-path (uiop:getcwd)) +blacklist-path+))
          (t
           (when (member c-name blacklist :test #'string=)
             (push (list id lisp-name c-name) blacklist-violations)
             (format *error-output*
                     "~&FAIL ~A imports ~A, which is ~A, a blacklisted C function ~
                      (see ~A)~%"
                     (enough-namestring backend-path (uiop:getcwd)) lisp-name c-name
                     +blacklist-path+))
           (unless (member c-name required :test #'string=)
             (push (list id lisp-name c-name) coverage-violations)
             (format *error-output*
                     "~&FAIL ~A imports ~A, which is ~A, but its *required-symbols* does ~
                      not declare that name. probe-foreign-symbols would not check it at ~
                      open-backend, so an incompatible library could open successfully and ~
                      fail only later, at this call.~%"
                     (enough-namestring backend-path (uiop:getcwd)) lisp-name c-name))))))
    (values blacklist-violations coverage-violations)))

(let* ((root (uiop:getcwd))
       (blacklist-file (merge-pathnames +blacklist-path+ root))
       (blacklist (blacklisted-names blacklist-file)))
  (format t "~&parsed ~D blacklisted function name~:P from ~A~%"
          (length blacklist) +blacklist-path+)
  ;; Step 3 of the task: fail loudly, before checking anything else, if the parse found
  ;; nothing. 38 functions are imported today and none are blacklisted, so an unguarded
  ;; empty parse would report the same "0 violations" a real, working check would -- the
  ;; exact silent-pass shape this file's header describes.
  (when (zerop (length blacklist))
    (format *error-output*
            "~&FAIL: parsed zero blacklisted function names from ~A.~@
             A blacklist check that matches nothing is worse than no check at all -- ~
             this is the same failure shape as mallet silently not checking line ~
             length. Either the file is missing, empty, or its table format no longer ~
             matches what `blacklisted-names' parses (a backtick-quoted name in a ~
             markdown table row's first column).~%"
            +blacklist-path+)
    (uiop:quit 1))
  (let ((blacklist-violations '())
        (coverage-violations '()))
    (dolist (spec +backends+)
      (destructuring-bind (id c-api-path c-api-package-name backend-path) spec
        (multiple-value-bind (backend-blacklist-violations backend-coverage-violations)
            (check-backend id
                           (merge-pathnames c-api-path root)
                           c-api-package-name
                           (merge-pathnames backend-path root)
                           blacklist)
          (setf blacklist-violations (append blacklist-violations backend-blacklist-violations))
          (setf coverage-violations
                (append coverage-violations backend-coverage-violations)))))
    (format t "~&~D blacklisted import~:P found across ~D backend~:P~%"
            (length blacklist-violations) (length +backends+))
    (format t "~&~D required-symbols coverage gap~:P found across ~D backend~:P~%"
            (length coverage-violations) (length +backends+))
    (uiop:quit (if (or blacklist-violations coverage-violations) 1 0))))
