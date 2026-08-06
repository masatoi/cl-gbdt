;;;; check-abi-blacklist.lisp --- No backend imports a blacklisted C function.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-abi-blacklist.lisp
;;;;
;;;; ffi-spec/ABI-BLACKLIST.md records upstream C functions that changed meaning between
;;;; the reference implementations' vendored headers and the versions cl-gbdt targets,
;;;; while keeping the same symbol name and, sometimes, the same argument count. A call
;;;; through the generated CFFI bindings to one of these succeeds -- no undefined-symbol
;;;; error, no link error, no wrong-arity error -- and does the wrong thing. That file's
;;;; own opening line is "The only defence is to never call them." This script is that
;;;; defence made mechanical: it fails the build if either backend ever imports one.
;;;;
;;;; Measurement recorded in docs/superpowers/specs/2026-08-06-abi-tracking-design.md
;;;; found that of the functions that changed or vanished across many upstream releases,
;;;; almost none touched the 38 functions the two backends actually import -- and the
;;;; near-exact exception set is this blacklist. The blacklist is not a list of hazards
;;;; to note for later; it is the thing that keeps the imported set stable, and this
;;;; check is what keeps that true instead of relying on human memory.
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
;;;; 3. For each backend, reads its `src/*/c-api.lisp' for every `cffi:defcfun' form and
;;;;    builds a Lisp-name -> C-name map, then reads the backend's `backend.lisp' for the
;;;;    `:import-from' clause that names its c-api package and maps every imported symbol
;;;;    back to a C name through that table. Any import whose C name is on the blacklist
;;;;    is reported as a violation.
;;;;
;;;; Like tools/ci/check-float-traps.lisp, every file here is read as data via `read',
;;;; never loaded or evaluated -- nothing in src/*/c-api.lisp or src/*/backend.lisp runs,
;;;; and no foreign library opens. This keeps the check usable from `tools/ci/lint.lisp'
;;;; and keeps layer 1's `foreign libraries open: NIL' invariant untouched by it.
;;;;
;;;; WHAT THIS CANNOT CATCH
;;;;
;;;;   - A blacklisted call made without going through an `:import-from' clause -- e.g. a
;;;;     fully package-qualified call `cl-gbdt/src/lightgbm/c-api:lgbm-dataset-create-from-mats'
;;;;     written directly in a backend file. Every call in both backends today goes
;;;;     through `:import-from', by convention, but this script trusts that convention
;;;;     rather than scanning call sites for qualified symbols.
;;;;   - A backend file that does not follow the `src/<backend>/backend.lisp' and
;;;;     `src/<backend>/c-api.lisp' layout named in +BACKENDS+ below -- a future backend
;;;;     added under a different convention needs that list extended, the same caveat
;;;;     +BACKEND-FILE-PATTERN+ carries in check-float-traps.lisp and +LEAF-ROOTS+ carries
;;;;     in check-leaf-systems.lisp.
;;;;   - A new hazard upstream introduces that nobody has added to ABI-BLACKLIST.md yet.
;;;;     This script enforces the list; it does not maintain it. Design doc section 11 and
;;;;     this branch's Task 2 describe the drift-detection tool that maintenance needs.

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
               "src/lightgbm/backend.lisp")
    (:xgboost "src/xgboost/c-api.lisp" "CL-GBDT/SRC/XGBOOST/C-API"
              "src/xgboost/backend.lisp"))
  "(backend-id c-api-path c-api-package-name backend-path) for each backend, all paths
relative to the repository root. See WHAT THIS CANNOT CATCH above for what adding a
backend that does not follow this layout would require.")

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

;;; ---- Main check ----

(defun check-backend (id c-api-path c-api-package-name backend-path blacklist)
  "Return the list of (ID LISP-NAME C-NAME) violations for one backend spec from
+BACKENDS+, printing progress as it goes."
  (let ((name-map (c-name-map c-api-path))
        (constants (constant-names c-api-path))
        (imports (backend-imports backend-path c-api-package-name))
        (violations '()))
    (format t "~&~A: ~D imported name~:P from ~A~%"
            (enough-namestring backend-path (uiop:getcwd)) (length imports)
            (enough-namestring c-api-path (uiop:getcwd)))
    (dolist (lisp-name imports violations)
      (let ((c-name (gethash lisp-name name-map)))
        (cond
          ;; An imported name with no `defcfun' behind it is reported, not skipped. Nothing
          ;; else notices one: `:import-from' on a symbol the source package does not export
          ;; interns it silently rather than erroring, so the build stays green -- verified.
          ;; Skipping it would also blind this check to the blacklist's "removed upstream,
          ;; never emitted" table, whose entries have no `defcfun' by definition and would
          ;; therefore never map. Treating "I could not resolve this" as "this is fine" is
          ;; the same failure this file's empty-parse guard exists to prevent.
          ((member lisp-name constants :test #'string=))   ; a constant, not a function
          ((null c-name)
           (push (list id lisp-name "<unresolved>") violations)
           (format *error-output*
                   "~&FAIL ~A imports ~A, which has no `cffi:defcfun' in ~A. Either it is ~
                    misspelled, or it names something removed upstream -- see the second ~
                    table in ~A.~%"
                   (enough-namestring backend-path (uiop:getcwd)) lisp-name
                   (enough-namestring c-api-path (uiop:getcwd)) +blacklist-path+))
          ((member c-name blacklist :test #'string=)
           (push (list id lisp-name c-name) violations)
           (format *error-output*
                   "~&FAIL ~A imports ~A, which is ~A, a blacklisted C function ~
                    (see ~A)~%"
                   (enough-namestring backend-path (uiop:getcwd)) lisp-name c-name
                   +blacklist-path+)))))))

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
  (let ((violations
          (mapcan (lambda (spec)
                    (destructuring-bind (id c-api-path c-api-package-name backend-path) spec
                      (check-backend id
                                     (merge-pathnames c-api-path root)
                                     c-api-package-name
                                     (merge-pathnames backend-path root)
                                     blacklist)))
                  +backends+)))
    (format t "~&~D blacklisted import~:P found across ~D backend~:P~%"
            (length violations) (length +backends+))
    (uiop:quit (if violations 1 0))))
