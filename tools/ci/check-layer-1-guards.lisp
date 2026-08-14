;;;; check-layer-1-guards.lisp --- Every public Layer 1 entry point checks the class of the
;;;; object it was handed, before that object's pointer reaches C.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-layer-1-guards.lisp
;;;;
;;;; Run it from the repository root; see THE FLOOR below for what happens if you do not.
;;;;
;;;; WHY THE HAZARD EXISTS: THE SPECIALIZER WAS THE CHECK
;;;;
;;;; Every operation this scans reached its present shape the same way. It was a `defmethod'
;;;; specialized on a concrete class -- `lightgbm-dataset', `xgboost-booster',
;;;; `lightgbm-backend' -- and the Layer 1 split moved it into `src/<backend>/api.lisp' as a
;;;; plain `defun', so that a caller who loaded only `cl-gbdt/lightgbm' or `cl-gbdt/xgboost'
;;;; could invoke it without the unified API in the image. That specializer WAS a type check,
;;;; the only one those bodies ever had. A `defun' takes whatever it is given, so converting
;;;; one drops the check while leaving nothing to see: the body is identical, every test keeps
;;;; passing, and the argument's class is simply never asked about again. Two consequences
;;;; were found by review rather than by anything in this repository, and each was measured
;;;; against the vendored libraries:
;;;;
;;;;   - `cl-gbdt/lightgbm:free-dataset' handed an `xgboost-dataset' passed that handle's
;;;;     pointer to `LGBM_DatasetFree'. One measurement was a `SB-SYS:MEMORY-FAULT-ERROR' at
;;;;     #xFFFFFFFFFFFFFFF8; another, on a different machine, was a SILENT RETURN -- which is
;;;;     the worse outcome, the process carrying on over freed or never-allocated memory.
;;;;   - `cl-gbdt/lightgbm:create-dataset' handed the XGBOOST backend ACCEPTED it and built a
;;;;     `lightgbm-dataset' -- `LGBM_DatasetCreateFromMat' never sees a backend object at all
;;;;     -- recording the XGBoost backend on the result. Closing the LightGBM backend that had
;;;;     really built it then left that handle reading live, and `free-dataset' on it called
;;;;     into a library `close-backend' had already unloaded.
;;;;
;;;; Both are fixed: every public Layer 1 `defun' now opens with a class check
;;;; (`%check-object-class', `%check-lightgbm-booster', `%check-xgboost-booster', ...). Until
;;;; this file, nothing verified that. The hazard is not a one-off either -- the next
;;;; increment converts further operations per backend by the identical move, and this is the
;;;; guard that has to keep seeing them.
;;;;
;;;; WHAT THIS CHECKS
;;;;
;;;; A `defun' in a backend's `api.lisp' or `native.lisp' that THAT BACKEND'S PUBLIC PACKAGE
;;;; EXPORTS, and whose FIRST REQUIRED ARGUMENT is named `backend', `dataset' or `booster',
;;;; must call a class check on that argument as the first thing its body does.
;;;;
;;;; Public means the `:export' clause of the LAST `uiop:define-package' form in the sibling
;;;; `all.lisp' -- which is the SECOND of the two both files hold: an internal aggregation
;;;; first, then `cl-gbdt/lightgbm' or `cl-gbdt/xgboost', the reviewed public surface both
;;;; files document in their own headers. Read as "the last" so that it stays the public one
;;;; if a third form ever appears, and `public-export-names' says the same. Nothing else
;;;; counts: `native.lisp''s own `:export'
;;;; clause publishes every `%'-function and the library-discovery internals, none of which a
;;;; caller reaches and every one of which runs after a public guard has already passed. That
;;;; is the same test `tools/ci/check-float-traps.lisp' applies to decide which `defun's are
;;;; entry points, deliberately duplicated here rather than shared -- see WHY THIS IS NOT PART
;;;; OF check-float-traps.lisp below.
;;;;
;;;; The first-argument name filter is what separates an operation over a caller-supplied
;;;; foreign object from one over plain data. It is also, today, no filter at all: all 31
;;;; public `defun's in these four files take exactly such an argument. It exists so that a
;;;; future public helper over numbers or strings is not reported as unguarded.
;;;;
;;;; HOW A GUARD IS RECOGNISED
;;;;
;;;; Structurally, never by a fixed list of names -- a fixed list would go stale the moment
;;;; someone adds a checker, which is precisely the failure mode this check exists to prevent,
;;;; arriving by a different door. Three conditions, all structural:
;;;;
;;;;   1. the call's head is a symbol whose name begins with `%CHECK-';
;;;;   2. its FIRST ARGUMENT IS THE PARAMETER ITSELF -- identically, not merely mentioned
;;;;      somewhere among the arguments;
;;;;   3. the function it names can REFUSE AN OBJECT FOR ITS CLASS, which is read off that
;;;;      function's own `defun': its body NAMES `wrong-backend-reference', directly or
;;;;      through another `%CHECK-' function it calls. `%check-object-class' signals it
;;;;      itself; `%check-lightgbm-booster' and `%check-xgboost-booster' reach it through
;;;;      `src/handle.lisp''s `%check-handle-kind', which is why that file is read too.
;;;;
;;;; Conditions 2 and 3 are each the difference between this check working and this check
;;;; certifying the very bugs it was written for. Both were measured, by deleting a real guard
;;;; and running:
;;;;
;;;;   - With 2 relaxed to "mentions the parameter anywhere", `free-dataset' rewritten to
;;;;     `(%check-object-class (handle-backend dataset) 'lightgbm-backend ...)' reports 6
;;;;     guarded, 0 unguarded, PASS, exit 0 -- while DATASET's own class is never asked about
;;;;     and the first bug listed above is live again. A guard on something DERIVED FROM the
;;;;     parameter is not a guard on the parameter.
;;;;   - With 3 absent, deleting `create-dataset''s class gate outright PASSES: the form left
;;;;     behind is `(%check-backend-open backend)', whose name carries the prefix and whose
;;;;     first argument is the parameter. That is the exact shape of the second bug above.
;;;;
;;;; So the recognition asks what the named function can actually do, and asks the SOURCE
;;;; rather than a list: adding a class checker under any new name needs no edit here, and a
;;;; `%check-'-named function that checks something else can never be mistaken for one.
;;;;
;;;; Condition 3 tests that the definition NAMES the condition, not that it signals it, and
;;;; the weaker test is the right one to want. A `defun' that merely mentions
;;;; `wrong-backend-reference' in a `handler-case' would qualify -- but only if EVERY
;;;; definition of that name did, which is what `class-check-p''s `every' over same-named
;;;; definitions buys: `%check-object-class' is defined once per backend, so mutating one
;;;; file's copy does not vouch for the other's. Testing for `(error 'wrong-backend-reference
;;;; ...)' as a shape instead would be tighter here and would break the moment a checker
;;;; signals through a helper, which is exactly how `%check-handle-kind' already works one
;;;; level down.
;;;;
;;;; "The first thing its body does" is the first form EVALUATED, not the first form written,
;;;; and the two differ in seventeen of the thirty-one -- counted fresh against today's set,
;;;; entry point by entry point, not carried over from when this comment was written against
;;;; seventeen entry points total. Every body opens with
;;;; `with-foreign-float-traps-masked' (see `tools/ci/check-float-traps.lisp', which is what
;;;; holds them to that), so the guard is at best the first form INSIDE it; and most of these
;;;; operations need the checked pointer, so the guard is the initialization form of the first
;;;; `let'/`let*' binding rather than a statement of its own. `booster-boosted-rounds' nests
;;;; it one further, as the argument of the `%'-function that consumes it. So this walks
;;;; inward from the body to the leftmost form actually evaluated first, through exactly these
;;;; shapes:
;;;;
;;;;   `with-foreign-float-traps-masked', `progn', `locally'   first body form
;;;;   `let', `let*'                                            first binding that has an
;;;;                                                            initialization form, else the
;;;;                                                            first body form
;;;;   `multiple-value-bind'                                    the values form
;;;;   a call to a `%'-prefixed function                        its first argument
;;;;   a call to a `%CHECK-' function                           nothing: the walk stops there
;;;;
;;;; The `%'-function case is safe only because every `%'-prefixed name in this project is a
;;;; `defun', so its arguments really are evaluated before it, left to right. The `%CHECK-'
;;;; case is about the REPORT and nothing else: by the time the walk would descend into one,
;;;; it has already been asked whether that call is a guard and been told no, and what the
;;;; reader needs to see is that call rather than its first argument. Any OTHER shape
;;;; ends the walk, and the form the walk ended on is then tested as a guard and reported
;;;; verbatim when it is not one -- fail-closed, the same posture
;;;; `check-layer-separation.lisp' takes for a package clause it cannot classify. A body whose
;;;; guard is real but reached through a shape not listed above fails this check and should
;;;; either be rewritten to lead with the guard or have its shape added here.
;;;;
;;;; THE FLOOR
;;;;
;;;; A scan that resolves paths against the current directory prints a clean PASS when it
;;;; finds nothing at all: no files, no entry points, no violations, exit 0.
;;;; `check-layer-separation.lisp' shipped with three such holes and they were found in
;;;; review, not by CI. So every state in which this check would report success while having
;;;; examined nothing is a failure instead:
;;;;
;;;;   - a file pattern matching no file (being run from the wrong directory, or a rename);
;;;;   - a file that matches a pattern and cannot be read -- see `read-top-level-forms', which
;;;;     reports it as one line rather than letting it reach the top level as a backtrace;
;;;;   - fewer files than +MINIMUM-ENTRY-POINT-FILES+;
;;;;   - a sibling `all.lisp' that is missing, holds no package form, or whose public package
;;;;     exports nothing -- `check-float-traps.lisp' returns NIL there and silently checks
;;;;     zero `defun's for that file, which is the one part of its technique NOT copied;
;;;;   - fewer entry points in total than +MINIMUM-ENTRY-POINTS+.
;;;;
;;;; The per-file and total counts are printed on a PASSING run too, so a human reading the
;;;; output can see that it looked at something rather than inferring it from exit 0.
;;;;
;;;; WHY THIS IS NOT PART OF check-float-traps.lisp
;;;;
;;;; That check already finds a backend's public `defun's the same way, and the two could
;;;; share the scan. They deliberately do not. They fail for different reasons -- one says a
;;;; foreign call can trap on x86-64, the other says a wrong-class pointer can reach C -- and
;;;; a build that goes red should name what it defends in the step that failed. The duplicated
;;;; part is small (a reader, an export-list lookup, a `defun' predicate) and each copy is free
;;;; to diverge where its own rule needs it to, as the empty-export-list floor above already
;;;; does.
;;;;
;;;; WHAT THIS CANNOT CATCH
;;;;
;;;; A source-level scan, not a dynamic one: what the source says is all it establishes, and
;;;; it neither loads nor evaluates anything -- not even the guards whose definitions it reads
;;;; for condition 3.
;;;;
;;;;   - Whether the check names the RIGHT CLASS. Condition 3 above establishes that the
;;;;     guard can refuse an object for its class, not that it refuses the right ones:
;;;;     `%check-object-class' with `xgboost-backend' written into LightGBM's `create-dataset'
;;;;     passes here. The functional suite is what holds these to their behaviour --
;;;;     tests/functional/lightgbm-standalone.lisp and its XGBoost counterpart pass each
;;;;     operation the other backend's object and require `wrong-backend-reference'.
;;;;   - An argument OTHER than the first. `create-booster' takes a backend and a dataset and
;;;;     checks both; only the first is required to be checked here, because only the first is
;;;;     what a reader can identify from the lambda list alone without knowing the operation.
;;;;   - A public entry point in a file matched by no pattern in +ENTRY-POINT-FILE-PATTERNS+.
;;;;     `slice-model' makes that concrete rather than hypothetical: it lived in
;;;;     `src/xgboost/classes.lisp' until `api.lisp' existed. A move back, or a new Layer 1
;;;;     file under some other name, needs this list extended -- the same hand-maintained-list
;;;;     caveat `check-float-traps.lisp''s +BACKEND-FILE-PATTERNS+ and
;;;;     `check-leaf-systems.lisp''s +LEAF-ROOTS+ each carry for themselves.
;;;;   - A first argument that takes a foreign object under some name other than the three in
;;;;     +HANDLE-PARAMETER-NAMES+.
;;;;   - A guard DELETED together with the entry point that held it. The floor below notices a
;;;;     count that falls under +MINIMUM-ENTRY-POINTS+, which is what catches a whole file
;;;;     going unread; it cannot notice one operation of thirty-one being removed on purpose.
;;;;     Raise that constant when the set grows, so the same protection applies at the new
;;;;     size.

(require :asdf)

;;; Every scanned file references `cffi:...' symbols, and every `all.lisp' references
;;; `uiop:...' ones. `read' below never evaluates anything, but it still needs those packages
;;; to EXIST to intern the qualified symbols as data. `(require :asdf)' above makes UIOP exist
;;; (ASDF 3 vendors it), so only CFFI needs an explicit load.
(ql:quickload "cffi" :silent t)

(define-condition scan-error (error)
  ((path :initarg :path :reader scan-error-path)
   (detail :initarg :detail :reader scan-error-detail))
  (:report (lambda (condition stream)
             (format stream "~A: ~A" (scan-error-path condition)
                     (scan-error-detail condition))))
  (:documentation "Signalled for any state in which this check would examine nothing and
still exit 0 -- a pattern matching no file, an `all.lisp' with no public export list, too few
files or too few entry points. See THE FLOOR in this file's header."))

(defparameter +entry-point-file-patterns+
  '("src/*/api.lisp" "src/*/native.lisp")
  "Globs, relative to the repository root, for the files this check scans.

`api.lisp' holds a backend's finished Layer 1 operations -- the ones that take a backend or a
handle, do the whole job, and hand back a handle or a result -- lifted out of the
`protocol.lisp' method bodies that used to hold them. `native.lisp' holds the `%'-functions,
and since the evaluation API also the handful of public entry points that need no CLOS class
of their own (`booster-eval', `booster-eval-names', `evaluate-one-iteration',
`booster-boosted-rounds').

`classes.lisp' is deliberately absent: it holds each backend's CLOS types and the shared
library's lifetime, and its public surface is the backend CLASS rather than any `defun'.
That is a fact about today's tree and not a rule -- `slice-model' lived there until `api.lisp'
existed -- so see WHAT THIS CANNOT CATCH, and extend this list rather than assume it stays
accurate on its own.")

(defparameter +handle-parameter-names+
  '("BACKEND" "DATASET" "BOOSTER")
  "First-required-argument names that mark a `defun' as taking a caller-supplied foreign
object, and so as needing a class guard.

These are the three the converted `defmethod's specialized on: the backend the two creators
build against, and the two handle kinds every other operation reads. Compared by NAME, since
every form here is read into a throwaway package, and upper case because that is what `read'
produces from this project's lower-case source.")

(defparameter +guard-name-prefix+ "%CHECK-"
  "The prefix a guard call's head must carry -- the first of the three conditions in HOW A
GUARD IS RECOGNISED.

Recognising a guard by SHAPE rather than by a list of known checker names is deliberate: a
list would have to be updated whenever someone adds a checker, and an out-of-date list would
report a guarded function as unguarded -- or, far worse, let a renamed checker go
unrecognised while the reader assumed the list was current.")

(defparameter +class-refusal-condition+ "WRONG-BACKEND-REFERENCE"
  "The condition a `%CHECK-' function must be able to signal to count as a CLASS check -- the
third condition in HOW A GUARD IS RECOGNISED.

This is what tells `%check-object-class' and `%check-handle-kind' apart from
`%check-backend-open', `%check-sparse-input' and every other `%check-'-named function that
asks about something the caller's object is not. Read out of the named function's own source
rather than from a list of names, so that a class checker added under a new name is recognised
without an edit here.")

(defparameter +guard-definition-file-patterns+
  '("src/*/api.lisp" "src/*/native.lisp" "src/handle.lisp")
  "Globs, relative to the repository root, for the files searched for a guard's own `defun'.

The scanned files themselves, plus `src/handle.lisp': `%check-lightgbm-booster' and
`%check-xgboost-booster' are thin wrappers over that file's `%check-handle-kind', and without
it neither would be recognised as reaching `wrong-backend-reference'.

A guard whose definition is not found under these patterns is NOT taken on trust -- it fails
as though it were no class check at all, which is loud and fail-closed. The floor this
parameter needs is therefore automatic: a list that matched nothing would fail every one of
the thirty-three entry points at once rather than passing them.")

(defparameter +minimum-entry-point-files+ 4
  "Fewer scanned files than this means nothing was walked, and fails.

Four is what the two backends' `api.lisp' and `native.lisp' come to. A third backend raises
the real number and still clears the floor; a wrong working directory or a rename drops it to
zero, which is the case this catches. See THE FLOOR.")

(defparameter +minimum-entry-points+ 33
  "Fewer public entry points found in total than this means the scan lost sight of something,
and fails.

Thirty-three is today's set: `create-dataset', `create-booster', `update-one-iteration',
`predict', `free-dataset', `free-booster', `save-model', `load-model', `model-to-string',
`feature-importance', `evaluation', `dataset-num-rows' and `dataset-num-features' in each
backend's `api.lisp', plus XGBoost's `slice-model' there and both backends'
`create-dataset-from-file'; plus `booster-eval' and `booster-eval-names' in LightGBM's
`native.lisp' and `evaluate-one-iteration' and `booster-boosted-rounds' in XGBoost's. Raise
this when the set grows, so that a later disappearance is caught at the new size rather than
absorbed by the old slack.")

(defun %restart-associated-with-p (restart condition)
  "True when RESTART is associated with CONDITION specifically, rather than merely visible
from here.

`find-restart' with a CONDITION argument does not answer this on its own: it returns the
restarts associated with CONDITION *plus* every restart associated with no condition at all,
so an unrelated outer `continue' -- `load''s, say -- matches a name lookup exactly as the
reader's own does. Invoking that one would abort the top-level form driving this scan and exit
0 with most of the repository unscanned.

The test is that an associated restart is invisible to any OTHER condition, while an
unassociated one is visible to all of them."
  (and (member restart (compute-restarts condition))
       (not (member restart (compute-restarts (make-condition 'simple-error))))))

(defun %unknown-package-reader-error-p (condition)
  "True when CONDITION is a `package-error' naming a package that does not exist -- the one
case `read-top-level-forms' resolves rather than reports.

The name is a STRING for exactly that case: the reader has a package name off the source token
and nothing to look it up in. Every other `package-error' carries a package OBJECT, which is
what keeps this NARROW -- most importantly SBCL's `package-locked-error', a `package-error'
subtype whose `continue' restart proceeds with the locked operation. A handler that invoked
that would let `(cl:totally-bogus-name)' in a scanned file intern into `COMMON-LISP' and the
scan exit 0, turning a loud failure into a silent one. That is not a hypothetical narrowing:
`tools/ci/check-float-traps.lisp' shipped the naive version first and review found exactly
this hole, which is why the version copied here is the narrowed one."
  (let ((name (package-error-package condition)))
    (and (stringp name) (not (find-package name)))))

(defun %intern-unresolvable-qualified-symbol (condition)
  "Resolve a `package-error' from `read' that names a package this image does not have, by
invoking the `continue' restart the reader established for it -- which reads the offending
token as a symbol of the same name in `*package*' instead.

Declines -- returns normally, leaving CONDITION to whatever handler is next, which in a
non-interactive run means the scan fails -- for every other `package-error', and for one whose
`continue' restart is not the reader's own. See `%unknown-package-reader-error-p' and
`%restart-associated-with-p' for what each of those excludes and why."
  (when (%unknown-package-reader-error-p condition)
    (let ((restart (find-restart 'continue condition)))
      (when (and restart (%restart-associated-with-p restart condition))
        (invoke-restart restart)))))

(defun read-top-level-forms (path)
  "Read PATH's top-level forms as data and return them as a list.

`read', never `load' or `compile-file': nothing in PATH runs. Forms are read with `*package*'
bound to a throwaway package that uses only CL, so a bare symbol interns harmlessly whether or
not PATH's own package is defined yet, and `*read-eval*' is NIL so a stray `#.' cannot run
code either.

The handler is for this project's own qualified references. `src/lightgbm/protocol.lisp' names
`cl-gbdt/src/lightgbm/api:free-dataset' in full because it also imports a GENERIC FUNCTION of
that name, and the scanned files carry similar tokens; reading one needs the named package to
EXIST, and nothing here loads project code to make it. The token is read as a bare symbol of
the same name instead, which loses nothing this script uses -- every form below is inspected by
NAME and never by symbol identity.

A file that matches a pattern but cannot be READ -- unreadable to this process, or truncated
mid-form -- is reported as `scan-error', one line like every other floor state, rather than
reaching the top level as a backtrace. It fails either way; the difference is whether the
reader of a red build is told which file and why in the first line they see. Those two cases
only: a package-lock violation in a scanned file, which
`%intern-unresolvable-qualified-symbol' declines and which is no `file-error',
`stream-error' or `reader-error', still escapes as an unhandled condition. That is
deliberate and is not widened here -- failing loudly on a locked package is the outcome the
narrowing above exists for."
  (let ((*package* (or (find-package '#:cl-gbdt/tools/ci/layer-1-guard-scratch)
                       (make-package '#:cl-gbdt/tools/ci/layer-1-guard-scratch
                                     :use '(#:cl))))
        (*read-eval* nil))
    (handler-case
        (handler-bind ((package-error #'%intern-unresolvable-qualified-symbol))
          (with-open-file (in path)
            (loop :for form := (read in nil :eof)
                  :until (eq form :eof)
                  :collect form)))
      ((or file-error stream-error reader-error) (condition)
        ;; Newlines squeezed out: several of these conditions report over two or three lines,
        ;; and the promise above is one line.
        (error 'scan-error
               :path (enough-namestring path (uiop:getcwd))
               :detail (format nil "cannot be read as data: ~A"
                               (substitute #\Space #\Newline (princ-to-string condition))))))))

(defun symbol-name-string= (symbol name)
  "True when SYMBOL's name is the string NAME, regardless of SYMBOL's package."
  (and symbol (symbolp symbol) (string= (symbol-name symbol) name)))

(defun defun-form-p (form)
  "True when FORM is a top-level `(defun ...)' form."
  (and (consp form) (symbol-name-string= (car form) "DEFUN")))

(defun define-package-form-p (form)
  "True when FORM is a top-level `(defpackage ...)' or `(uiop:define-package ...)' form."
  (and (consp form)
       (or (symbol-name-string= (car form) "DEFPACKAGE")
           (symbol-name-string= (car form) "DEFINE-PACKAGE"))))

(defun files-matching (patterns)
  "Return the sorted pathnames matching PATTERNS, relative to the current directory.

Signals `scan-error' for a pattern that matches nothing at all: resolved against the current
directory, every pattern matches nothing when this is run from anywhere but the repository
root, and a scan of no files is not a passing scan. See THE FLOOR."
  (let ((files '()))
    (dolist (pattern patterns)
      (let ((matched (directory (merge-pathnames pattern (uiop:getcwd)))))
        (unless matched
          (error 'scan-error
                 :path pattern
                 :detail (format nil "matched no file under ~A -- either this check is not ~
                                      being run from the repository root, or what the pattern ~
                                      names has moved and the list naming it is stale"
                                 (uiop:getcwd))))
        (setf files (append files matched))))
    (sort (remove-duplicates files :test #'equal) #'string< :key #'namestring)))

(defun export-clause-names (package-form)
  "Return the symbol-name strings named in PACKAGE-FORM's `(:export ...)' clause(s).

Multiple `:export' clauses are all collected, though every package form in this repository
uses exactly one."
  (loop :for clause :in (cddr package-form)
        :when (and (consp clause) (symbol-name-string= (car clause) "EXPORT"))
          :append (mapcar #'symbol-name (remove-if-not #'symbolp (cdr clause)))))

(defun sibling-all-lisp (path)
  "Return PATH's sibling `all.lisp' -- the file in the same directory that assembles the
backend's public package, e.g. `src/lightgbm/native.lisp' -> `src/lightgbm/all.lisp'."
  (make-pathname :name "all" :defaults path))

(defun public-export-names (all-lisp-path)
  "Return the symbol-name strings EXPORTed by the LAST `define-package'/`defpackage' form in
ALL-LISP-PATH -- the backend's public package, `cl-gbdt/lightgbm' or `cl-gbdt/xgboost'. Both
backends' `all.lisp' documents that exact shape in its own header: two `define-package' forms,
the first an internal aggregation, the second the reviewed public surface.

Signals `scan-error' when the file does not exist, holds no package form, or holds one whose
`:export' clause is empty. `tools/ci/check-float-traps.lisp' returns NIL for all three and
goes on to check zero `defun's for that file; that is the one part of its technique this
script does not copy, because an empty export list here is indistinguishable in the output
from a file with nothing to find."
  (let ((file (probe-file all-lisp-path)))
    (unless file
      (error 'scan-error
             :path (enough-namestring all-lisp-path (uiop:getcwd))
             :detail (format nil "no such file -- every scanned file must have a sibling ~
                                  all.lisp naming its backend's public package")))
    (let ((package-forms (remove-if-not #'define-package-form-p (read-top-level-forms file))))
      (unless package-forms
        (error 'scan-error
               :path (enough-namestring file (uiop:getcwd))
               :detail "contains no defpackage form"))
      (let ((names (export-clause-names (car (last package-forms)))))
        (unless names
          (error 'scan-error
                 :path (enough-namestring file (uiop:getcwd))
                 :detail (format nil "its last defpackage form exports nothing, so no defun ~
                                      in this directory can be recognised as a public entry ~
                                      point")))
        names))))

(defun lambda-list-keyword-p (symbol)
  "True when SYMBOL is a lambda-list keyword -- a symbol whose name starts with `&'.

Tested by name rather than by membership in `lambda-list-keywords', for the same reason
everything else here is: these symbols were read into a throwaway package and are not `eq' to
CL's."
  (let ((name (symbol-name symbol)))
    (and (plusp (length name)) (char= (char name 0) #\&))))

(defun first-required-parameter (lambda-list)
  "Return LAMBDA-LIST's first required parameter, or NIL when it has none.

NIL for an empty lambda list, for one that opens with `&optional', `&key' or `&rest', and for
a destructuring pattern -- none of which a caller passes a bare handle through."
  (let ((first (car lambda-list)))
    (when (and first (symbolp first) (not (lambda-list-keyword-p first)))
      first)))

(defun entry-point-p (form public-names)
  "True when FORM is a `defun' that PUBLIC-NAMES exports and whose first required parameter is
one of +HANDLE-PARAMETER-NAMES+ -- an operation a caller reaches directly, over an object the
caller supplied, with no `defmethod' specializer left to have type-checked it."
  (and (defun-form-p form)
       (symbolp (second form))
       (member (symbol-name (second form)) public-names :test #'string=)
       (let ((parameter (first-required-parameter (third form))))
         (and parameter
              (member (symbol-name parameter) +handle-parameter-names+ :test #'string=)))))

(defun skip-docstring-and-declares (body)
  "Return BODY with a leading docstring and any leading `(declare ...)' forms removed.

A leading string only counts as a docstring when more forms follow it -- a single-form body
that is just a string is that string's return value, not documentation."
  (when (and (stringp (car body)) (cdr body))
    (setf body (cdr body)))
  (loop :while (and body (consp (car body)) (symbol-name-string= (caar body) "DECLARE"))
        :do (setf body (cdr body)))
  body)

(defun mentions-symbol-p (tree name)
  "True when a symbol named NAME appears anywhere in TREE."
  (cond ((and tree (symbolp tree)) (string= (symbol-name tree) name))
        ((consp tree) (or (mentions-symbol-p (car tree) name)
                          (mentions-symbol-p (cdr tree) name)))
        (t nil)))

(defun guard-names-in (tree)
  "Return the names of every +GUARD-NAME-PREFIX+ symbol appearing anywhere in TREE."
  (let ((names '()))
    (labels ((walk (node)
               (cond ((and node (symbolp node))
                      (let ((name (symbol-name node)))
                        (when (uiop:string-prefix-p +guard-name-prefix+ name)
                          (pushnew name names :test #'string=))))
                     ((consp node) (walk (car node)) (walk (cdr node))))))
      (walk tree))
    names))

(defun guard-definitions ()
  "Return a hash table from each +GUARD-NAME-PREFIX+ function's name to the list of bodies
defining it, read from +GUARD-DEFINITION-FILE-PATTERNS+.

A LIST of bodies, because `%check-object-class' is defined once per backend. `class-check-p'
requires every definition of a name to qualify, so two functions sharing a name cannot have
one of them vouch for the other."
  (let ((table (make-hash-table :test #'equal)))
    (dolist (path (files-matching +guard-definition-file-patterns+) table)
      (dolist (form (read-top-level-forms path))
        (when (and (defun-form-p form)
                   (symbolp (second form))
                   (second form)
                   (uiop:string-prefix-p +guard-name-prefix+ (symbol-name (second form))))
          (push (cdddr form) (gethash (symbol-name (second form)) table)))))))

(defun class-check-p (name definitions)
  "True when the function called NAME can refuse an object for its CLASS -- the third
condition in HOW A GUARD IS RECOGNISED.

Its body names +CLASS-REFUSAL-CONDITION+, or it calls another +GUARD-NAME-PREFIX+ function
that does. False for a name DEFINITIONS does not hold at all: a guard whose definition this
scan cannot read is not taken on trust.

The condition name is matched as a SYMBOL, so a docstring that merely discusses
`wrong-backend-reference' -- every one of these functions does -- does not qualify its
function; only a form that names the symbol counts. SEEN stops the walk revisiting a name,
which also makes a cycle among checkers terminate."
  (labels ((refuses-class-p (body seen)
             (or (mentions-symbol-p body +class-refusal-condition+)
                 (some (lambda (called) (qualifies-p called seen)) (guard-names-in body))))
           (qualifies-p (name seen)
             (and (not (member name seen :test #'string=))
                  (let ((bodies (gethash name definitions)))
                    (and bodies
                         (every (lambda (body) (refuses-class-p body (cons name seen)))
                                bodies))))))
    (qualifies-p name '())))

(defun guard-call-p (form parameter definitions)
  "True when FORM is a call to a +GUARD-NAME-PREFIX+ function whose FIRST ARGUMENT IS
PARAMETER ITSELF and which can refuse an object for its class -- the three conditions in HOW A
GUARD IS RECOGNISED.

PARAMETER identically, in first argument position, and not merely mentioned somewhere in the
argument tree. A guard on something DERIVED FROM the parameter is not a guard on the
parameter, and the difference is the whole defect: `(%check-object-class (handle-backend
dataset) 'lightgbm-backend ...)' mentions DATASET, names a real class checker, and leaves
DATASET's own class never asked about -- measured, it reported `free-dataset' as guarded and
exited 0 while the first bug in this file's header was live again. Every one of the
thirty-three guards passes its parameter bare and first, so this costs the passing set
nothing.

The other two conditions are no less load-bearing. The prefix alone would accept a check of
some unrelated object; and prefix plus a first argument accept `(%check-backend-open
backend)', which is exactly what deleting `create-dataset''s class gate leaves behind --
measured, and the reason `class-check-p' exists."
  (and (consp form)
       (symbolp (car form))
       (car form)
       (uiop:string-prefix-p +guard-name-prefix+ (symbol-name (car form)))
       (symbol-name-string= (second form) (symbol-name parameter))
       (class-check-p (symbol-name (car form)) definitions)))

(defun binding-initialization-form (bindings body)
  "Return the form a `let'/`let*' with BINDINGS and BODY evaluates first.

The first binding that HAS an initialization form, since `(x)' and a bare `x' evaluate
nothing; the first body form when no binding has one."
  (loop :for binding :in bindings
        :when (and (consp binding) (cdr binding))
          :return (second binding)
        :finally (return (first body))))

(defun first-evaluated-subform (form)
  "Return the subform of FORM this script believes is evaluated before any other, or NIL when
FORM's shape is not one it walks into.

The four shapes are listed in this file's header, with the reason the `%'-function case is
sound. Anything else -- including a macro whose first subform is not evaluated first -- stops
the walk, and `leading-guard' then reports the form the walk stopped on. NIL is returned both
for a shape not recognised and for a recognised shape whose first evaluated subform is
literally NIL; the two need no distinguishing, since neither is a guard call.

A +GUARD-NAME-PREFIX+ call stops the walk even though it is a `%'-function, and this is only
about the report. `leading-guard' has already asked whether it is a guard and been told no, so
what the reader needs to see is THAT CALL -- `(%check-backend-open backend)' -- and not its
first argument, which is where descending would leave the walk."
  (let ((head (car form)))
    (when (and head (symbolp head))
      (let ((name (symbol-name head)))
        (cond
          ((member name '("LET" "LET*") :test #'string=)
           (binding-initialization-form (second form) (cddr form)))
          ((member name '("WITH-FOREIGN-FLOAT-TRAPS-MASKED" "PROGN" "LOCALLY") :test #'string=)
           (second form))
          ((string= name "MULTIPLE-VALUE-BIND") (third form))
          ((uiop:string-prefix-p +guard-name-prefix+ name) nil)
          ((and (plusp (length name)) (char= (char name 0) #\%) (cdr form)) (second form))
          (t nil))))))

(defun leading-guard (form parameter definitions)
  "Return (VALUES GUARD LAST-FORM) for `defun' FORM's guard on PARAMETER.

GUARD is the guard call FORM's body evaluates first, and LAST-FORM is NIL. When the body
evaluates something else first, GUARD is NIL and LAST-FORM is the form the walk stopped on,
for the failure report. The walk terminates because every step moves strictly inward."
  (let ((current (first (skip-docstring-and-declares (cdddr form)))))
    (loop
      (cond ((not (consp current)) (return (values nil current)))
            ((guard-call-p current parameter definitions) (return (values current nil)))
            (t (let ((next (first-evaluated-subform current)))
                 (if next
                     (setf current next)
                     (return (values nil current)))))))))

(defun leading-form-description (form)
  "Return a short printed representation of FORM, for a failure report.

`*package*' is the scratch package `read-top-level-forms' read FORM's symbols into, so they
print bare instead of qualified with this script's throwaway reader package. The printer is
bounded, and pretty-printing disabled, so that a whole function body cannot arrive in the
report and one finding stays one line -- a violation is one line of CI output to scroll back
to, not a listing."
  (let ((*package* (find-package '#:cl-gbdt/tools/ci/layer-1-guard-scratch))
        (*print-pretty* nil)
        (*print-level* 2)
        (*print-length* 3))
    (if form (format nil "~S" form) "an empty body")))

(defun violation-description (form parameter last-form definitions)
  "Return the sentence reported for an unguarded entry point FORM.

Three sentences, because a `%CHECK-' call that is not a guard is a NEAR MISS and the reader
is owed which of the two conditions it missed on: it cannot refuse anything for its class, or
it can but was pointed at something other than PARAMETER. Anything else gets the plain
sentence naming the form that ran instead."
  (let ((head (and (consp last-form) (symbolp (car last-form)) (car last-form))))
    (cond
      ((not (and head (uiop:string-prefix-p +guard-name-prefix+ (symbol-name head))))
       (format nil "~(~A~) does not check ~(~A~)'s class first -- it evaluates ~A"
               (second form) parameter (leading-form-description last-form)))
      ((not (class-check-p (symbol-name head) definitions))
       (format nil "~(~A~) does not check ~(~A~)'s class first -- it reaches ~(~A~), which ~
                    cannot signal ~(~A~) and so establishes no object's class"
               (second form) parameter head +class-refusal-condition+))
      (t
       (format nil "~(~A~) does not check ~(~A~)'s class first -- it reaches ~(~A~) on ~A, ~
                    not on ~(~A~) itself"
               (second form) parameter head (leading-form-description (second last-form))
               parameter)))))

(defun check-file (path public-names definitions)
  "Return (VALUES ENTRY-POINT-COUNT VIOLATIONS) for PATH. VIOLATIONS is a list of readable
strings, one per public entry point (see `entry-point-p') whose body does not begin by
checking the class of its first required argument (see `leading-guard')."
  (let ((entry-points (remove-if-not (lambda (form) (entry-point-p form public-names))
                                     (read-top-level-forms path)))
        (violations '()))
    (dolist (form entry-points)
      (let ((parameter (first-required-parameter (third form))))
        (multiple-value-bind (guard last-form) (leading-guard form parameter definitions)
          (unless guard
            (push (violation-description form parameter last-form definitions) violations)))))
    (values (length entry-points) (nreverse violations))))

(defun report-file (path definitions)
  "Scan PATH and print its one-line result. Returns (VALUES ENTRY-POINT-COUNT VIOLATIONS),
having already written every violation to `*error-output*'."
  (let ((relative (enough-namestring path (uiop:getcwd))))
    (multiple-value-bind (count violations)
        (check-file path (public-export-names (sibling-all-lisp path)) definitions)
      (format t "~&~A: ~D public defun~:P taking a handle or backend, ~D guarded, ~
                 ~D unguarded~%"
              relative count (- count (length violations)) (length violations))
      (dolist (violation violations)
        (format *error-output* "~&FAIL ~A: ~A~%" relative violation))
      (values count (mapcar (lambda (violation) (format nil "~A: ~A" relative violation))
                            violations)))))

(defun check-floors (file-count entry-point-count)
  "Signal `scan-error' when FILE-COUNT or ENTRY-POINT-COUNT is under its floor -- the states
in which this check would report success while having examined nothing. See THE FLOOR."
  (when (< file-count +minimum-entry-point-files+)
    (error 'scan-error
           :path "+ENTRY-POINT-FILE-PATTERNS+"
           :detail (format nil "matched ~D file~:P, fewer than the ~D this repository has; ~
                                nothing can have been walked"
                           file-count +minimum-entry-point-files+)))
  (when (< entry-point-count +minimum-entry-points+)
    (error 'scan-error
           :path "+MINIMUM-ENTRY-POINTS+"
           :detail (format nil "found ~D public entry point~:P, fewer than the ~D this ~
                                repository has; an export list, a file or a whole backend ~
                                went unread"
                           entry-point-count +minimum-entry-points+))))

(handler-case
    (let ((files (files-matching +entry-point-file-patterns+))
          (definitions (guard-definitions))
          (total 0)
          (failures '()))
      (format t "~&checking ~D backend file~:P for Layer 1 entry-point class guards~%"
              (length files))
      (dolist (path files)
        (multiple-value-bind (count violations) (report-file path definitions)
          (incf total count)
          (setf failures (append failures violations))))
      (format t "~&~D public entry point~:P checked across ~D file~:P, ~D unguarded~%"
              total (length files) (length failures))
      (check-floors (length files) total)
      (when failures
        (format *error-output*
                "~&unguarded entry points -- each was a defmethod whose specializer was the ~
                 class check~%")
        (dolist (failure failures)
          (format *error-output* "~&  ~A~%" failure)))
      (if failures
          (format t "~&FAIL: a Layer 1 entry point can be handed another backend's object~%")
          (format t "~&PASS: every Layer 1 entry point checks the class of the object it was ~
                     handed~%"))
      (finish-output)
      (uiop:quit (if failures 1 0)))
  (scan-error (condition)
    (format t "~&FAIL: ~A~%" condition)
    (finish-output)
    (uiop:quit 1)))
