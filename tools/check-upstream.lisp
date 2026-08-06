;;;; check-upstream.lisp --- Detect upstream ABI drift in cl-gbdt's imported C functions.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/check-upstream.lisp
;;;;
;;;; Checks both backends' imported functions against the tags ffi-spec/VERSIONS pins --
;;;; the tags the vendored headers and generated bindings were produced from. Override
;;;; either tag to check a hypothetical or newer release instead:
;;;;
;;;;   CHECK_UPSTREAM_LIGHTGBM_TAG=v5.0.0 ros run -- --non-interactive \
;;;;     --load tools/check-upstream.lisp
;;;;   CHECK_UPSTREAM_XGBOOST_TAG=v1.5.0 ros run -- --non-interactive \
;;;;     --load tools/check-upstream.lisp
;;;;
;;;; ffi-spec/ABI-BLACKLIST.md names this tool as its own maintenance path: a function
;;;; this reports ABSENT should move from that file's "still present" table to "moot",
;;;; and one reported CHANGED should be added to "still present".
;;;;
;;;; WHY THE COMPARISON IS RESTRICTED TO IMPORTED FUNCTIONS
;;;;
;;;; docs/superpowers/specs/2026-08-06-abi-tracking-design.md section 1 measured both
;;;; whole-header comparison (3 of 11 LightGBM transitions broke something, 4 of 7
;;;; XGBoost transitions did) and used-function comparison (18 LightGBM functions,
;;;; v3.0.0-v4.7.0, zero breaking changes; 20 XGBoost functions, v1.7.0-v3.3.0, zero
;;;; breaking changes). Whole-header comparison makes both projects look unstable by
;;;; counting churn in functions cl-gbdt never calls; restricting to the imported set is
;;;; what shows the wide compatible range, and is the only comparison this tool makes.
;;;;
;;;; This is a port of the prototype that produced those tables
;;;; (.superpowers/sdd/2026-08-06-abi-tracking/analyze.py), restricted per function to
;;;; the imported set the same way check-abi-blacklist.lisp already computes it, with two
;;;; bugs the prototype had found and fixed here -- see NORMALIZATION below.
;;;;
;;;; WHAT THIS CHECKS
;;;;
;;;; 1. For each backend, the imported C function names: every `:import-from' name in
;;;;    the backend's c-api package that resolves to a `cffi:defcfun', exactly as
;;;;    tools/ci/check-abi-blacklist.lisp computes its own import list (constants
;;;;    excluded -- they have no C declaration to compare).
;;;; 2. The vendored header's declaration of each imported name, normalized.
;;;; 3. The same declarations fetched fresh from the tag under check (pinned by
;;;;    ffi-spec/VERSIONS, or overridden per backend by the environment variables
;;;;    above), also normalized.
;;;; 4. Reports, per imported function: ABSENT (gone from the checked tag), CHANGED
;;;;    (present but normalized differently), or silently fine (matches).
;;;;
;;;; A fetch failure -- offline, DNS, a tag that does not exist upstream -- is reported
;;;; as FAIL, never folded into "0 problems found". A drift checker that cannot look and
;;;; says nothing changed is the same defect class as an empty-parse guard that never
;;;; fires: see tools/ci/check-abi-blacklist.lisp's own header for the incident that
;;;; established that principle on this branch.
;;;;
;;;; NORMALIZATION
;;;;
;;;; `int* out' and `int *out' are the same ABI; so are a parameter renamed and a
;;;; parameter with `const' added or dropped. Ported from analyze.py's `norm_param':
;;;; split each parameter on whitespace and `*', drop the trailing token as a parameter
;;;; name unless it is a recognised base-type keyword (so a lone `int' stays a type, but
;;;; `DatasetHandle handle' loses `handle'), drop `const', sort what remains, and
;;;; reattach `*' once per pointer level. Applied to both the return type and every
;;;; parameter.
;;;;
;;;; Porting surfaced two bugs beyond the known one (analyze.py's regex missed
;;;; void-returning declarations):
;;;;
;;;;   - The prototype's per-declaration regex is anchored only on the macro name
;;;;     (`XGB_DLL', `LIGHTGBM_C_EXPORT'), which also appears, unanchored, inside that
;;;;     macro's own `#define'. In xgboost/c_api.h that reads:
;;;;       #define XGB_DLL XGB_EXTERN_C __declspec(dllexport)
;;;;       ...
;;;;       XGB_DLL void XGBoostVersion(int *major, int *minor, int *patch);
;;;;     A DOTALL regex starting the match at the `#define' line's `XGB_DLL' scans
;;;;     forward for the first `);' it can find -- which is `XGBoostVersion's own
;;;;     closing paren -- and consumes it as bogus "parameters" of a bogus declaration
;;;;     named `__declspec'. `XGBoostVersion' never gets its own match: this is the exact
;;;;     mechanism behind "regex misses void-returning declarations", not something
;;;;     specific to `void' -- any declaration would be swallowed if it happened to sit
;;;;     first after the macro's `#define' block, which XGBoostVersion does. Verified by
;;;;     isolating the offending match: its captured name was `__declspec', not
;;;;     `XGBoostVersion'. Fixed here two ways: preprocessor directive lines (`#...') are
;;;;     stripped before scanning, and every declaration match is anchored at the start
;;;;     of a line, so a macro name appearing mid-line inside surviving text cannot seed
;;;;     a false match either.
;;;;   - A function declared `foo()' and the same function declared `foo(void)' are the
;;;;     same ABI -- zero arguments either way -- but splitting an empty parameter string
;;;;     on cl-ppcre's `split' returns NIL, not a single empty token the way Python's
;;;;     `re.split' does, so porting `norm_param' as-is normalized `()' to no parameters
;;;;     and `(void)' to `("void")': two different signatures for the same function.
;;;;     Verified against XGBGetLastError, declared `(void)' in XGBoost v1.5.0 and `()'
;;;;     in v3.3.0 -- an empty parameter string is special-cased to `("void")' before the
;;;;     split, below, closing that false positive.
;;;;
;;;; Verified against every cached LightGBM tag v3.0.0-v4.7.0 and XGBoost tag v1.5.0-
;;;; v3.3.0, restricted to the imported sets: LightGBM reports zero changes across all
;;;; eleven transitions; XGBoost reports exactly one, `XGBoosterSaveModelToBuffer' first
;;;; appearing between v1.5.0 and v1.6.0 -- the known gap this tool's own proof (see
;;;; .superpowers/sdd/2026-08-06-abi-tracking/task-2-report.md) points it at directly.
;;;;
;;;; WHAT THIS CANNOT CATCH
;;;;
;;;;   - A change to what a typedef used in a signature actually resolves to (e.g.
;;;;     `bst_ulong' changing width) -- this is a textual comparison of the declaration,
;;;;     never a compile, so a typedef's own definition changing without its use site's
;;;;     text changing is invisible here.
;;;;   - A behavioral change that keeps the same declared signature -- this tool only
;;;;     ever looks at declarations, the same limitation `cffi:foreign-symbol-pointer'
;;;;     probing has, which is why ABI-BLACKLIST.md exists at all for the cases this
;;;;     project already knows about.
;;;;   - A call made without going through an `:import-from' clause -- see
;;;;     tools/ci/check-abi-blacklist.lisp's own "WHAT THIS CANNOT CATCH" for the same
;;;;     caveat; this tool computes its imported set the identical way.

(require :asdf)

;;; `cffi' so `read' below can intern the `cffi:...' symbols c-api.lisp and backend.lisp
;;; reference as data (see tools/ci/check-abi-blacklist.lisp's header for why `read'
;;; needs the package to exist even though nothing here is evaluated). `cl-ppcre' for the
;;; declaration scan and normalization.
(ql:quickload '("cffi" "cl-ppcre") :silent t)

(defparameter +backends+
  '((:id "LightGBM"
     :versions-key "lightgbm"
     :macro "LIGHTGBM_C_EXPORT"
     :c-api-path "src/lightgbm/c-api.lisp"
     :c-api-package "CL-GBDT/SRC/LIGHTGBM/C-API"
     :backend-path "src/lightgbm/native.lisp"
     :vendored-header "ffi-spec/lightgbm/include/LightGBM/c_api.h"
     :url "https://raw.githubusercontent.com/microsoft/LightGBM/~A/include/LightGBM/c_api.h"
     :tag-env-var "CHECK_UPSTREAM_LIGHTGBM_TAG")
    (:id "XGBoost"
     :versions-key "xgboost"
     :macro "XGB_DLL"
     :c-api-path "src/xgboost/c-api.lisp"
     :c-api-package "CL-GBDT/SRC/XGBOOST/C-API"
     :backend-path "src/xgboost/native.lisp"
     :vendored-header "ffi-spec/xgboost/include/xgboost/c_api.h"
     :url "https://raw.githubusercontent.com/dmlc/xgboost/~A/include/xgboost/c_api.h"
     :tag-env-var "CHECK_UPSTREAM_XGBOOST_TAG"))
  "One plist per backend. Paths are relative to the repository root. `:url' has one `~A'
for the tag being checked. Only `tools/fetch-headers.sh' vendors headers
permanently into ffi-spec/ -- this reads the same upstream files by the same URL shape,
into memory, for a single ephemeral comparison, and never writes them to disk.")

;;; ---- Reading Lisp source as data, never as code ----
;;;
;;; Identical technique to tools/ci/check-abi-blacklist.lisp and
;;; tools/ci/check-float-traps.lisp: a throwaway package using only CL, so every bare
;;; symbol in these files interns harmlessly, and `*read-eval*' bound to NIL so a stray
;;; `#.' cannot run code.

(defun read-top-level-forms (path)
  "Read PATH's top-level forms as data and return them as a list."
  (let ((*package* (or (find-package '#:cl-gbdt/tools/check-upstream-scratch)
                        (make-package '#:cl-gbdt/tools/check-upstream-scratch
                                       :use '(#:cl))))
        (*read-eval* nil))
    (with-open-file (in path)
      (loop :for form := (read in nil :eof)
            :until (eq form :eof)
            :collect form))))

(defun symbol-name-string= (symbol name)
  "True when SYMBOL's name is the string NAME, regardless of SYMBOL's package."
  (and (symbolp symbol) (string= (symbol-name symbol) name)))

;;; ---- The imported-function set, computed the same way check-abi-blacklist.lisp does ----

(defun defcfun-form-p (form)
  "True when FORM is a top-level `(cffi:defcfun ...)' form."
  (and (consp form) (symbol-name-string= (car form) "DEFCFUN")))

(defun defcfun-c-and-lisp-names (form)
  "Return (VALUES C-NAME LISP-NAME) for a `cffi:defcfun' FORM naming both the C symbol
and the Lisp function, i.e. `(cffi:defcfun (\"C_Name\" lisp-name) ...)'. Returns NIL for
any other shape."
  (let ((spec (second form)))
    (when (and (consp spec) (stringp (first spec)) (symbolp (second spec)))
      (values (first spec) (symbol-name (second spec))))))

(defun c-name-map (path)
  "Return a hash table mapping every Lisp function name (string, upper-case) defined by
a `cffi:defcfun' in PATH to its C name."
  (let ((table (make-hash-table :test #'equal)))
    (dolist (form (read-top-level-forms path) table)
      (when (defcfun-form-p form)
        (multiple-value-bind (c-name lisp-name) (defcfun-c-and-lisp-names form)
          (when lisp-name
            (setf (gethash lisp-name table) c-name)))))))

(defun constant-names (path)
  "Return the names (string, upper-case) of every `defconstant' in PATH -- imported
constants have no `cffi:defcfun' behind them and must not be looked up as functions."
  (let ((names '()))
    (dolist (form (read-top-level-forms path) names)
      (when (and (consp form)
                 (member (first form) '(defconstant cl:defconstant))
                 (symbolp (second form)))
        (push (symbol-name (second form)) names)))))

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
  "Return the list of Lisp names BACKEND-PATH imports from C-API-PACKAGE-NAME."
  (let* ((forms (read-top-level-forms backend-path))
         (form (define-package-form forms)))
    (unless form
      (error "no uiop:define-package form found in ~A" backend-path))
    (import-from-names form c-api-package-name)))

(defun imported-c-function-names (c-api-path c-api-package-name backend-path)
  "Return the sorted list of upstream C function names BACKEND-PATH actually calls --
every `:import-from' name from C-API-PACKAGE-NAME that resolves to a `cffi:defcfun',
constants excluded. This is the used-function set section 1 of the design doc turns on:
see this file's header for why the comparison stops here instead of covering every
declaration in the header."
  (let ((name-map (c-name-map c-api-path))
        (constants (constant-names c-api-path))
        (imports (backend-imports backend-path c-api-package-name)))
    (sort (loop :for lisp-name :in imports
                :unless (member lisp-name constants :test #'string=)
                  :collect (or (gethash lisp-name name-map)
                                (error "~A imports ~A, which has no cffi:defcfun in ~A. ~
                                        tools/ci/check-abi-blacklist.lisp diagnoses this ~
                                        case specifically; run that first."
                                       backend-path lisp-name c-api-path)))
          #'string<)))

;;; ---- Declaration extraction and normalization, ported from analyze.py ----
;;;
;;; See this file's header, section NORMALIZATION, for what changed in the port.

(defparameter +type-keywords+
  '("const" "unsigned" "signed" "int" "char" "float" "double" "void" "long" "short"
    "size_t" "bst_ulong" "bst_float" "uint32_t" "uint64_t" "int32_t" "int64_t" "uint8_t")
  "Base C type keywords recognised by `normalize-parameter'. Any other trailing token in
a parameter is assumed to be the parameter's name, not part of its type -- this is what
lets `DatasetHandle handle' and `DatasetHandle data' normalize identically.")

(defun normalize-parameter (raw)
  "Normalize one C parameter or return-type string RAW to a whitespace- and
naming-independent form: `int* out', `int *out', and `int* data' all normalize to
\"int*\"; `const char* name' normalizes to \"char*\". See NORMALIZATION above."
  (let ((cleaned (ppcre:regex-replace-all "\\[\\]" (string-trim '(#\Space #\Tab #\Newline
                                                                   #\Return)
                                                                 raw)
                                           "*")))
    (if (or (string= cleaned "void") (string= cleaned ""))
        "void"
        (let ((stars (count #\* cleaned))
              (tokens (remove-if (lambda (s) (zerop (length s)))
                                  (ppcre:split "[\\s\\*]+" cleaned))))
          (when (and (> (length tokens) 1)
                     (not (member (car (last tokens)) +type-keywords+ :test #'string=)))
            (setf tokens (butlast tokens)))          ; drop the parameter name
          (setf tokens (remove "const" tokens :test #'string=))
          (format nil "~{~A~^ ~}~A"
                  (sort tokens #'string<)
                  (make-string stars :initial-element #\*))))))

(defun split-parameters (params)
  "Split a C parameter list PARAMS on top-level commas (never inside nested parens, e.g.
a function-pointer parameter). An all-whitespace PARAMS -- `()' with nothing between the
parens -- is treated as the single parameter \"\", which `normalize-parameter' turns
into \"void\", the same as an explicit `(void)'. See NORMALIZATION above for why this
cannot be left to fall out of the split itself."
  (if (zerop (length (string-trim '(#\Space #\Tab #\Newline #\Return) params)))
      (list "")
      (ppcre:split ",(?![^()]*\\))" params)))

(defun strip-comments (text)
  "Remove C block comments (`/* ... */') and line comments (`// ...') from TEXT."
  (ppcre:regex-replace-all "//[^\\n]*"
                            (ppcre:regex-replace-all "(?s)/\\*.*?\\*/" text "")
                            ""))

(defun strip-preprocessor-lines (text)
  "Remove every line of TEXT whose first non-whitespace character is `#'.

Without this, a macro's own `#define' -- which textually contains the macro's name,
e.g. `#define XGB_DLL XGB_EXTERN_C __declspec(dllexport)' -- can seed a false
declaration match that swallows the next real declaration whole. See NORMALIZATION
above; this is the fix for the bug that made `XGBoostVersion' disappear."
  (ppcre:regex-replace-all "(?m)^[ \\t]*#.*$" text ""))

(defun function-declarations (text macro)
  "Return a hash table mapping every function name declared with MACRO in TEXT to its
normalized (RETURN-TYPE . PARAMETER-LIST) signature. Comments and preprocessor
directives are stripped first, and a declaration must start at the beginning of a line
-- both defend against MACRO's name appearing outside an actual declaration; see
NORMALIZATION above."
  (let ((clean (strip-preprocessor-lines (strip-comments text)))
        (table (make-hash-table :test #'equal)))
    (ppcre:do-register-groups (return-type name params)
        ((format nil "(?ms)^[ \\t]*~A\\s+([\\w\\s\\*]+?)\\b(\\w+)\\s*\\((.*?)\\)\\s*;" macro)
         clean)
      (setf (gethash name table)
            (cons (normalize-parameter (concatenate 'string return-type " r"))
                  (mapcar #'normalize-parameter (split-parameters params)))))
    table))

(defun read-file-text (path)
  "Return the full contents of PATH as a string."
  (with-open-file (in path :external-format :utf-8)
    (let* ((text (make-string (file-length in)))
           (read (read-sequence text in)))
      (subseq text 0 read))))

;;; ---- Fetching upstream, and failing clearly when that is not possible ----

(defun fetch-url (url)
  "Fetch URL and return (VALUES SUCCESS-P BODY-OR-DIAGNOSTIC).

A non-zero curl exit status -- no network, DNS failure, or a tag curl's `--fail' turns a
404 into (exit 22) -- returns NIL and a diagnostic string, never NIL and an empty body
silently treated as \"nothing to report\". See this file's header: a fetch failure is a
FAIL, not a clean run."
  (handler-case
      (multiple-value-bind (output error-output status)
          (uiop:run-program (list "curl" "-sSL" "--fail" url)
                             :output :string :error-output :string
                             :ignore-error-status t)
        (if (zerop status)
            (values t output)
            (values nil (format nil "curl exited ~D fetching ~A~@[ -- ~A~]" status url
                                 (let ((trimmed (string-trim '(#\Newline #\Space)
                                                              error-output)))
                                   (and (plusp (length trimmed)) trimmed))))))
    (error (condition)
      (values nil (format nil "could not run curl for ~A: ~A" url condition)))))

(defun pinned-tag (project root)
  "Return the tag ffi-spec/VERSIONS pins for PROJECT (\"lightgbm\" or \"xgboost\"),
relative to ROOT -- the default every backend is checked against unless its
environment-variable override (see +BACKENDS+) is set."
  (with-open-file (in (merge-pathnames "ffi-spec/VERSIONS" root))
    (loop :for line := (read-line in nil)
          :while line
          :for space := (position #\Space line)
          :when (and space (string= project (subseq line 0 space)))
            :return (string-trim '(#\Space #\Return) (subseq line (1+ space)))
          :finally (error "no pinned tag for ~A in ffi-spec/VERSIONS" project))))

;;; ---- Main check ----

(defun check-backend (spec root)
  "Check one backend SPEC (a plist from +BACKENDS+) against its tag, relative to ROOT.
Prints progress and per-function findings. Returns one of :FETCH-FAILED (network or
upstream problem -- see WHAT THIS CHECKS), :INCONSISTENT (this tool's own parser found
no vendored declaration for something the backend imports, a bug in this tool or a stale
vendored header, not upstream drift), :CHANGED (at least one imported function is
ABSENT or its signature changed), or :CLEAN."
  (let* ((id (getf spec :id))
         (tag (or (uiop:getenv (getf spec :tag-env-var))
                  (pinned-tag (getf spec :versions-key) root)))
         (imports (imported-c-function-names (merge-pathnames (getf spec :c-api-path) root)
                                              (getf spec :c-api-package)
                                              (merge-pathnames (getf spec :backend-path) root)))
         (baseline (function-declarations
                    (read-file-text (merge-pathnames (getf spec :vendored-header) root))
                    (getf spec :macro)))
         (url (format nil (getf spec :url) tag)))
    (format t "~&~%=== ~A ~A: ~D imported function~:P ===~%" id tag (length imports))
    (format t "fetching ~A~%" url)
    (multiple-value-bind (ok body) (fetch-url url)
      (unless ok
        (format *error-output* "~&FAIL ~A: ~A~%" id body)
        (return-from check-backend :fetch-failed))
      (let ((candidate (function-declarations body (getf spec :macro)))
            (inconsistent 0)
            (problems 0))
        (dolist (name imports)
          (multiple-value-bind (base base-p) (gethash name baseline)
            (multiple-value-bind (candidate-decl candidate-p) (gethash name candidate)
              (cond
                ((not base-p)
                 (incf inconsistent)
                 (format *error-output*
                         "~&FAIL ~A: ~A is imported but this tool's own parser found no ~
                          declaration for it in the vendored header -- the parser or the ~
                          vendored spec is out of sync, not upstream drift.~%"
                         id name))
                ((not candidate-p)
                 (incf problems)
                 (format t "  ABSENT   ~A -- present in the vendored header, gone from ~A~%"
                         name tag))
                ((not (equal base candidate-decl))
                 (incf problems)
                 (format t "  CHANGED  ~A~%    vendored: ~A~%    ~A:  ~A~%"
                         name base tag candidate-decl))))))
        (format t "~D/~D imported function~:P unchanged~%"
                (- (length imports) problems inconsistent) (length imports))
        (cond ((plusp inconsistent) :inconsistent)
              ((plusp problems) :changed)
              (t :clean))))))

(let* ((root (uiop:getcwd))
       (results (mapcar (lambda (spec) (check-backend spec root)) +backends+)))
  (format t "~&~%")
  (cond
    ((member :fetch-failed results)
     (format t "FAIL: could not fetch at least one upstream header -- see FAIL lines ~
                above. This is reported as a failure, never as \"no drift found\".~%")
     (uiop:quit 1))
    ((member :inconsistent results)
     (format t "FAIL: this tool's own parser disagrees with a vendored header -- see ~
                FAIL lines above.~%")
     (uiop:quit 1))
    ((member :changed results)
     (format t "FAIL: upstream drift detected -- see ABSENT/CHANGED lines above. Update ~
                ffi-spec/ABI-BLACKLIST.md's tables to match.~%")
     (uiop:quit 1))
    (t
     (format t "clean: every imported function's declaration matches at the checked tag.~%")
     (uiop:quit 0))))
