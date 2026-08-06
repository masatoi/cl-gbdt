;;;; check-float-traps.lisp --- Every backend defmethod, and every backend-file defun
;;;; a public package exports directly, masks SBCL's float traps.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-float-traps.lisp
;;;;
;;;; SBCL enables the :invalid, :divide-by-zero and :overflow floating-point traps by
;;;; default on x86-64 and not on aarch64. LightGBM and XGBoost are C code written and
;;;; tested against the C convention, where those traps stay masked -- an intermediate
;;;; NaN or infinity partway through their own numeric code (XGBoost's `multi:softprob'
;;;; softmax is the case that surfaced this) is data to keep working with, not a signal.
;;;; `cl-gbdt/src/foreign''s `with-foreign-float-traps-masked' restores that convention;
;;;; every `defmethod' in a backend file -- all twelve protocol methods plus
;;;; `initialize-backend' and `shutdown-backend' -- wraps its entire body in it. See that
;;;; macro's docstring for the full reasoning and
;;;; `.superpowers/sdd/2026-08-04-xgboost-backend/float-traps-report.md' for the
;;;; incident that added it. A `defmethod' added later that forgets the wrap behaves
;;;; identically to a wrapped one on this aarch64 development host -- it fails only on
;;;; an x86-64 runner, and only if some test happens to exercise the exact path that
;;;; produces a trapping intermediate value. That is exactly how the bug reached CI
;;;; last time: nothing checked for the wrap's absence.
;;;;
;;;; SECOND GAP THIS CLOSES: the evaluation-api branch's Task 2 (LightGBM's
;;;; `booster-eval'/`booster-eval-names') added the first `native.lisp' functions ever
;;;; reached directly by end users -- exported straight from the backend's public
;;;; package (`cl-gbdt/lightgbm', `cl-gbdt/xgboost') via the second `define-package' form
;;;; in each backend's `all.lisp', per policy section 3's Layer 1 and that file's own
;;;; header comment, rather than through a `cl-gbdt/src/*/protocol' `defmethod'. Until
;;;; then, every call into a backend's shared library was provably reached from a
;;;; `defmethod''s already-masked dynamic extent, which is what let this script trust
;;;; every `%'-prefixed helper in `native.lisp' without checking it directly -- see WHAT
;;;; THIS CANNOT CATCH below. A plain `defun' promoted straight to the public package has
;;;; no such `defmethod' to inherit the mask from, so it must establish its own, the same
;;;; method-body granularity `protocol.lisp' uses -- and until this update, nothing
;;;; checked that it did. `src/lightgbm/all.lisp' and `src/xgboost/all.lisp' commit every
;;;; future Phase 2 addition to the identical shape (see their own header comments), so
;;;; this is not a one-off: Tasks 3 and 4 add more such entry points, and this is the
;;;; guard that has to keep seeing them.
;;;;
;;;; WHAT THIS CHECKS
;;;;
;;;; 1. Every `defmethod' form in each file matched by +BACKEND-FILE-PATTERNS+ (never
;;;;    loaded or evaluated -- only read as data) must have `with-foreign-float-traps-masked'
;;;;    as the first form of its body, after any leading docstring and `declare's. That is
;;;;    the exact invariant the backend files document for themselves: method-body
;;;;    granularity, not per-call, so a call added later inside an already-wrapped method
;;;;    cannot reopen the gap by omission -- only a whole new (or newly unwrapped)
;;;;    `defmethod' can.
;;;; 2. Every `defun' form in the same files whose name is EXPORTed by the backend's public
;;;;    package -- the *second* `define-package' form in the sibling `all.lisp' (see
;;;;    `public-export-names' and `sibling-all-lisp') -- is held to the identical rule: its
;;;;    first body form, after any docstring and `declare's, must be
;;;;    `with-foreign-float-traps-masked'. This is deliberately narrower than "every exported
;;;;    symbol in `native.lisp'": `native.lisp''s own `:export' clause also lists
;;;;    `check-lgbm'/`check-xgb' and the library-discovery internals, none of which the
;;;;    public package re-exports and none of which is ever called from outside an
;;;;    already-masked dynamic extent -- exactly the case this check still trusts, per (1)
;;;;    below. Only a name the *public* package promotes is a library-reaching entry point
;;;;    with no `defmethod' left to inherit a mask from.
;;;;
;;;; WHAT THIS CANNOT CATCH
;;;;
;;;; This is a source-level scan, not a dynamic one -- a scan of the text can never
;;;; prove dynamic coverage, only textual presence. Concretely:
;;;;
;;;;   - A private (`%'-prefixed, or exported only from `native.lisp' and never from the
;;;;     public package) helper function that calls a C entry point directly, if it is ever
;;;;     invoked from somewhere other than an already-wrapped `defmethod''s or already-wrapped
;;;;     public `defun''s dynamic extent. Every current helper (`%update-one-iteration',
;;;;     `%read-version', `%booster-eval-count', ...) is only ever called from one of those
;;;;     two, and now that (2) above exists, that claim covers the public `defun's too, not
;;;;     only the `defmethod's -- but this script still does not verify the call graph either
;;;;     way, it trusts it, exactly as the existing manual audit in float-traps-report.md did.
;;;;   - A form placed as a SIBLING of `with-foreign-float-traps-masked' inside a `defmethod'
;;;;     or public `defun' body, after it, rather than nested inside its `&body' -- such a
;;;;     form would not be masked even though the first body form is. Every current method or
;;;;     public `defun''s body either is the macro call itself or starts with it wrapping
;;;;     everything else, so this has not happened, but this check only inspects the first
;;;;     body form, not every one.
;;;;   - A backend file that does not match any of +BACKEND-FILE-PATTERNS+, or a public
;;;;     package whose export list this script cannot find because its `all.lisp' does not
;;;;     follow the documented two-`define-package'-forms shape -- `public-export-names'
;;;;     returns NIL rather than erroring when that happens, which silently checks zero public
;;;;     `defun's for that file rather than failing loudly. A future backend added under a
;;;;     different path or package convention would need both +BACKEND-FILE-PATTERNS+ and
;;;;     this assumption extended, the same limitation `tools/ci/check-leaf-systems.lisp''s
;;;;     +LEAF-ROOTS+ documents for itself.
;;;;   - Whether masking those three specific traps is the right set for whatever a newly
;;;;     added call does -- this only checks that the macro is present as the wrapping form,
;;;;     not that its argument list still matches what the call needs.
;;;;
;;;; Most importantly: this cannot be verified against the actual failure it guards,
;;;; because this development host cannot produce it. aarch64 SBCL enables none of
;;;; these traps by default, and the underlying hardware does not implement trapped
;;;; floating-point exceptions at all -- `feenableexcept' returns -1, confirmed
;;;; empirically; it is an optional part of the ARMv8 FP architecture this CPU lacks.
;;;; QEMU's x86-64 TCG emulation does not deliver the trap as SIGFPE either, also
;;;; confirmed empirically (see float-traps-report.md for both measurements). So there
;;;; is no way, on this host, to write a test that fails without the mask and passes
;;;; with it. A source scan is what is available; it is not a substitute for the
;;;; dynamic coverage it cannot get, and this file does not claim otherwise -- the same
;;;; honesty `tools/ci/lint.lisp' insists on for mallet not checking line length.

(require :asdf)

;;; Every scanned file references `cffi:...' symbols, and every `all.lisp' references
;;; `uiop:...' ones. `read' below never evaluates anything, but it still needs the CFFI and
;;; UIOP packages to *exist* to intern those qualified symbols as data -- an unqualified bare
;;; symbol would intern fine into any package, but `cffi:defcfun' or `uiop:define-package'
;;; would signal "package ... does not exist" without this. `(require :asdf)' above already
;;; makes UIOP exist (ASDF 3 vendors it), so only CFFI needs an explicit load.
(ql:quickload "cffi" :silent t)

(defparameter +backend-file-patterns+
  '("src/*/backend.lisp" "src/*/native.lisp" "src/*/protocol.lisp")
  "Globs, relative to the repository root, for files this check scans.

Both backends used to keep every protocol method in one `backend.lisp'. XGBoost's Task 2
and LightGBM's Task 3 each split that file into `native.lisp' (Layer 1: the %-functions,
the error wrapper, and -- since the evaluation-api branch's Task 2 -- any public
entry-point `defun's too) and `protocol.lisp' (Layer 2: the classes and all fourteen
methods) -- so most `defmethod' forms live in `protocol.lisp', not `native.lisp', but a
public `defun' can now legitimately live in either. `backend.lisp' is still listed even
though neither backend has one anymore: a future backend added under that older
convention should still be caught by this scan rather than silently exempted from it. A
backend split under some other naming convention entirely would still silently not be
scanned -- extend this list rather than assume it stays accurate on its own, the same
caveat `tools/ci/check-leaf-systems.lisp''s +LEAF-ROOTS+ carries for the same reason.")

(defun backend-files ()
  "Return the sorted list of pathnames matching any of +BACKEND-FILE-PATTERNS+."
  (sort (remove-duplicates
         (mapcan (lambda (pattern) (directory (merge-pathnames pattern (uiop:getcwd))))
                 +backend-file-patterns+)
         :test #'equal)
        #'string< :key #'namestring))

(defun read-top-level-forms (path)
  "Read PATH's top-level forms as data and return them as a list.

`read', never `load' or `compile-file': nothing in PATH runs, and no side effect of
loading it (registering a CLOS class, defining a backend) happens here. Forms are
read with `*package*' bound to a throwaway package that uses only CL, so a bare
symbol -- everything in these files except its `cffi:...'/`uiop:...' references --
interns harmlessly regardless of whether PATH's own package is defined yet.
`*read-eval*' is bound to NIL so a stray `#.' in the source cannot run code either."
  (let ((*package* (or (find-package '#:cl-gbdt/tools/ci/float-traps-scratch)
                        (make-package '#:cl-gbdt/tools/ci/float-traps-scratch
                                       :use '(#:cl))))
        (*read-eval* nil))
    (with-open-file (in path)
      (loop :for form := (read in nil :eof)
            :until (eq form :eof)
            :collect form))))

(defun symbol-name-string= (symbol name)
  "True when SYMBOL's name is the string NAME, regardless of SYMBOL's package."
  (and (symbolp symbol) (string= (symbol-name symbol) name)))

(defun defmethod-form-p (form)
  "True when FORM is a top-level `(defmethod ...)' form."
  (and (consp form) (symbol-name-string= (car form) "DEFMETHOD")))

(defun defun-form-p (form)
  "True when FORM is a top-level `(defun ...)' form."
  (and (consp form) (symbol-name-string= (car form) "DEFUN")))

(defun define-package-form-p (form)
  "True when FORM is a top-level `(defpackage ...)' or `(uiop:define-package ...)' form."
  (and (consp form)
       (or (symbol-name-string= (car form) "DEFPACKAGE")
           (symbol-name-string= (car form) "DEFINE-PACKAGE"))))

(defun split-defmethod (form)
  "Return (VALUES QUALIFIERS LAMBDA-LIST BODY) for a `defmethod' FORM.

Qualifiers (e.g. `:before') are whatever non-list forms appear between the method
name and its lambda list. DEFMETHOD's grammar guarantees the lambda list is the
first LIST in that position -- a qualifier is never itself a list -- so the first
list encountered after the name is taken as the lambda list unconditionally."
  (let ((tail (cddr form))
        (qualifiers '()))
    (loop :while (and tail (not (listp (car tail))))
          :do (push (pop tail) qualifiers))
    (values (nreverse qualifiers) (car tail) (cdr tail))))

(defun skip-docstring-and-declares (body)
  "Return BODY with a leading docstring and any leading `(declare ...)' forms removed.

A leading string only counts as a docstring when more forms follow it -- a
single-form body that is just a string is that string's return value, not
documentation, though no method or public defun in these files takes that form
today."
  (when (and (stringp (car body)) (cdr body))
    (setf body (cdr body)))
  (loop :while (and body (consp (car body)) (symbol-name-string= (caar body) "DECLARE"))
        :do (setf body (cdr body)))
  body)

(defun method-signature (form)
  "Return a human-readable \"NAME QUALIFIERS LAMBDA-LIST\" string for a `defmethod' FORM,
for reporting.

`*package*' is bound to the same scratch package `read-top-level-forms' read FORM's
symbols into, so they print bare -- e.g. `DATASET' -- instead of package-qualified
with this script's own throwaway reader package, which would otherwise leak into
every report line as noise unrelated to the actual finding."
  (let ((*package* (find-package '#:cl-gbdt/tools/ci/float-traps-scratch)))
    (multiple-value-bind (qualifiers lambda-list body) (split-defmethod form)
      (declare (ignore body))
      (format nil "~A~{ ~S~} ~S" (second form) qualifiers lambda-list))))

(defun trap-masked-p (form)
  "True when `defmethod' FORM's first body form, after any docstring and `declare's, is
a call to `with-foreign-float-traps-masked'. See this file's header for exactly what
that does and does not prove."
  (multiple-value-bind (qualifiers lambda-list raw-body) (split-defmethod form)
    (declare (ignore qualifiers lambda-list))
    (let ((body (skip-docstring-and-declares raw-body)))
      (and body
           (consp (first body))
           (symbol-name-string= (car (first body)) "WITH-FOREIGN-FLOAT-TRAPS-MASKED")))))

(defun defun-trap-masked-p (form)
  "True when `defun' FORM's first body form, after any docstring and `declare's, is a call
to `with-foreign-float-traps-masked'. Same rule as `trap-masked-p', adapted for `defun''s
simpler grammar -- `(defun name lambda-list . body)' has no method qualifiers or
combination to split off first."
  (let ((body (skip-docstring-and-declares (cdddr form))))
    (and body
         (consp (first body))
         (symbol-name-string= (car (first body)) "WITH-FOREIGN-FLOAT-TRAPS-MASKED"))))

(defun export-clause-names (package-form)
  "Return the list of symbol-name strings named in PACKAGE-FORM's `(:export ...)'
clause(s) -- PACKAGE-FORM is a `defpackage' or `uiop:define-package' form. Multiple
`:export' clauses are all collected, though every package form in this repository
uses exactly one."
  (loop :for clause :in (cddr package-form)
        :when (and (consp clause) (symbol-name-string= (car clause) "EXPORT"))
          :append (mapcar #'symbol-name (remove-if-not #'symbolp (cdr clause)))))

(defun public-export-names (all-lisp-path)
  "Return the symbol-name strings EXPORTed by the *last* `define-package'/`defpackage'
form in ALL-LISP-PATH -- the backend's public package, e.g. `cl-gbdt/lightgbm' or
`cl-gbdt/xgboost'. Both backends' `all.lisp' documents this exact shape in its own header:
two `define-package' forms, the first an internal aggregation, the second (this one) the
reviewed public surface a Phase 2 addition names explicitly. Returns NIL, rather than
erroring, when ALL-LISP-PATH does not exist or contains no package form -- see this
script's own \"WHAT THIS CANNOT CATCH\" for what that silently gives up checking."
  (when (probe-file all-lisp-path)
    (let ((package-forms (remove-if-not #'define-package-form-p
                                         (read-top-level-forms all-lisp-path))))
      (when package-forms
        (export-clause-names (car (last package-forms)))))))

(defun sibling-all-lisp (path)
  "Return PATH's sibling `all.lisp' -- the file in the same directory that assembles the
backend's public package, e.g. `src/lightgbm/native.lisp' -> `src/lightgbm/all.lisp'."
  (make-pathname :name "all" :defaults path))

(defun public-entry-point-defun-p (form public-names)
  "True when FORM is a top-level `(defun ...)' whose name PUBLIC-NAMES contains -- the
backend's public package exports it directly, so end users reach it without ever going
through a `defmethod'. See this file's header, point 2 of WHAT THIS CHECKS, for why that
makes it a library-reaching entry point this script must not silently trust the way it
trusts every other helper in `native.lisp'."
  (and (defun-form-p form)
       (symbolp (second form))
       (member (symbol-name (second form)) public-names :test #'string=)))

(defun check-file (path public-names)
  "Return (VALUES METHOD-COUNT PUBLIC-DEFUN-COUNT VIOLATIONS) for PATH. VIOLATIONS is a
list of readable signature strings for every unmasked `defmethod' (see `trap-masked-p')
or unmasked public entry-point `defun' named in PUBLIC-NAMES (see
`public-entry-point-defun-p' and `defun-trap-masked-p')."
  (let* ((forms (read-top-level-forms path))
         (methods (remove-if-not #'defmethod-form-p forms))
         (public-defuns (remove-if-not (lambda (form)
                                          (public-entry-point-defun-p form public-names))
                                        forms))
         (method-violations (mapcar #'method-signature (remove-if #'trap-masked-p methods)))
         (defun-violations (mapcar (lambda (form) (format nil "~S" (second form)))
                                    (remove-if #'defun-trap-masked-p public-defuns))))
    (values (length methods) (length public-defuns) (append method-violations
                                                              defun-violations))))

(let ((files (backend-files))
      (total-methods 0)
      (total-public-defuns 0)
      (failures '()))
  (format t "~&checking ~D backend file~:P for float-trap-masked defmethods and public ~
             entry-point defuns~%"
          (length files))
  (dolist (path files)
    (let ((relative (enough-namestring path (uiop:getcwd)))
          (public-names (public-export-names (sibling-all-lisp path))))
      (multiple-value-bind (method-count public-defun-count violations)
          (check-file path public-names)
        (incf total-methods method-count)
        (incf total-public-defuns public-defun-count)
        (format t "~&~A: ~D defmethod~:P, ~D public defun~:P, ~D unmasked~%"
                relative method-count public-defun-count (length violations))
        (dolist (signature violations)
          (push (format nil "~A: ~A" relative signature) failures)
          (format *error-output*
                  "~&FAIL ~A: ~A is not wrapped in WITH-FOREIGN-FLOAT-TRAPS-MASKED~%"
                  relative signature)))))
  (format t "~&~D defmethod~:P + ~D public defun~:P = ~D form~:P checked across ~D file~:P, ~
             ~D unmasked~%"
          total-methods total-public-defuns (+ total-methods total-public-defuns)
          (length files) (length failures))
  (when failures
    (format *error-output* "~&unmasked forms (source scan only -- see this file's ~
                             header for what that does and does not prove):~%")
    (dolist (failure (reverse failures))
      (format *error-output* "~&  ~A~%" failure)))
  (uiop:quit (if failures 1 0)))
