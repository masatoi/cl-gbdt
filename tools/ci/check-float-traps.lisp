;;;; check-float-traps.lisp --- Every backend defmethod masks SBCL's float traps.
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
;;;; WHAT THIS CHECKS
;;;;
;;;; Every `defmethod' form in each file matched by +BACKEND-FILE-PATTERN+ (never
;;;; loaded or evaluated -- only read as data) must have `with-foreign-float-traps-masked'
;;;; as the first form of its body, after any leading docstring and `declare's. That is
;;;; the exact invariant the backend files document for themselves: method-body
;;;; granularity, not per-call, so a call added later inside an already-wrapped method
;;;; cannot reopen the gap by omission -- only a whole new (or newly unwrapped)
;;;; `defmethod' can.
;;;;
;;;; WHAT THIS CANNOT CATCH
;;;;
;;;; This is a source-level scan, not a dynamic one -- a scan of the text can never
;;;; prove dynamic coverage, only textual presence. Concretely:
;;;;
;;;;   - A private helper function (this project's convention prefixes them `%') that
;;;;     calls a C entry point directly, if it is ever invoked from somewhere other than
;;;;     an already-wrapped `defmethod''s dynamic extent. Every current helper
;;;;     (`%update-one-iteration', `%read-version', ...) is only ever called from a
;;;;     wrapped method's body; this script does not verify that call graph, it trusts
;;;;     it, exactly as the existing manual audit in float-traps-report.md did.
;;;;   - A form placed as a SIBLING of `with-foreign-float-traps-masked' inside a
;;;;     `defmethod' body, after it, rather than nested inside its `&body' -- such a
;;;;     form would not be masked even though the method's *first* body form is. Every
;;;;     current method's body is the single macro call itself, so this has not
;;;;     happened, but this check only inspects the first body form, not every one.
;;;;   - A backend file that does not match any of +BACKEND-FILE-PATTERNS+ -- a future
;;;;     backend added under a different path convention would need that list extended,
;;;;     the same limitation `tools/ci/check-leaf-systems.lisp''s +LEAF-ROOTS+ documents
;;;;     for itself.
;;;;   - Whether masking those three specific traps is the right set for whatever a
;;;;     newly added call does -- this only checks that the macro is present as the
;;;;     wrapping form, not that its argument list still matches what the call needs.
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

;;; Every scanned file references `cffi:...' symbols. `read' below never evaluates
;;; anything, but it still needs the CFFI package to *exist* to intern those qualified
;;; symbols as data -- an unqualified bare symbol would intern fine into any package,
;;; but `cffi:defcfun' would signal "package CFFI does not exist" without this.
(ql:quickload "cffi" :silent t)

(defparameter +backend-file-patterns+
  '("src/*/backend.lisp" "src/*/native.lisp" "src/*/protocol.lisp")
  "Globs, relative to the repository root, for files this check scans.

LightGBM still keeps every protocol method in one `backend.lisp'. XGBoost's Task 2 split
that file into `native.lisp' (Layer 1: no `defmethod' at all, only the %-functions and
the error wrapper) and `protocol.lisp' (Layer 2: the classes and all fourteen methods) --
so the file that actually holds XGBoost's `defmethod' forms is `protocol.lisp', not
`native.lisp'. Both are still listed: `native.lisp' currently contributes zero
`defmethod' forms to scan, but a future edit that moved one there by mistake -- or a
LightGBM split that names its own halves differently -- should still be caught by this
scan rather than silently exempted from it, the same reasoning
`tools/ci/check-abi-blacklist.lisp' gives for reading `native.lisp' specifically for the
c-api `:import-from' clause. A backend split under some other naming convention entirely
would still silently not be scanned -- extend this list rather than assume it stays
accurate on its own, the same caveat `tools/ci/check-leaf-systems.lisp''s +LEAF-ROOTS+
carries for the same reason.")

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
symbol -- everything in these files except its `cffi:...' references -- interns
harmlessly regardless of whether PATH's own package is defined yet. `*read-eval*' is
bound to NIL so a stray `#.' in the source cannot run code either."
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
documentation, though no method in these files takes that form today."
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

(defun check-file (path)
  "Return (VALUES METHOD-COUNT VIOLATIONS) for PATH. VIOLATIONS is a list of readable
method-signature strings for every `defmethod' not masked -- see `trap-masked-p'."
  (let* ((forms (read-top-level-forms path))
         (methods (remove-if-not #'defmethod-form-p forms))
         (violations (mapcar #'method-signature (remove-if #'trap-masked-p methods))))
    (values (length methods) violations)))

(let ((files (backend-files))
      (total-methods 0)
      (failures '()))
  (format t "~&checking ~D backend file~:P for float-trap-masked defmethods~%" (length files))
  (dolist (path files)
    (let ((relative (enough-namestring path (uiop:getcwd))))
      (multiple-value-bind (method-count violations) (check-file path)
        (incf total-methods method-count)
        (format t "~&~A: ~D defmethod~:P, ~D unmasked~%"
                relative method-count (length violations))
        (dolist (signature violations)
          (push (format nil "~A: ~A" relative signature) failures)
          (format *error-output*
                  "~&FAIL ~A: ~A is not wrapped in WITH-FOREIGN-FLOAT-TRAPS-MASKED~%"
                  relative signature)))))
  (format t "~&~D defmethod~:P checked across ~D file~:P, ~D unmasked~%"
          total-methods (length files) (length failures))
  (when failures
    (format *error-output* "~&unmasked defmethods (source scan only -- see this file's ~
                             header for what that does and does not prove):~%")
    (dolist (failure (reverse failures))
      (format *error-output* "~&  ~A~%" failure)))
  (uiop:quit (if failures 1 0)))
