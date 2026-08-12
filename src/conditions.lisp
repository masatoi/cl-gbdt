;;;; conditions.lisp --- Condition hierarchy for cl-gbdt.
;;;;
;;;; The C APIs report failure through int return codes, so every foreign call is
;;;; wrapped and converted into a condition immediately. Nothing propagates silently.

(uiop:define-package #:cl-gbdt/src/conditions
  (:use #:cl)
  (:export #:gbdt-error
           #:backend-error
           #:backend-error-backend
           #:backend-library-not-found
           #:backend-library-not-found-searched
           #:backend-library-load-failed
           #:backend-library-load-failed-path
           #:backend-library-load-failed-cause
           #:missing-foreign-symbols
           #:missing-foreign-symbols-names
           #:backend-not-open
           #:unknown-backend
           #:unknown-backend-registered
           #:foreign-call-error
           #:foreign-call-error-code
           #:foreign-call-error-message
           #:foreign-call-error-function-name
           #:released-handle-error
           #:released-handle-error-object
           #:wrong-backend-reference
           #:wrong-backend-reference-backend
           #:wrong-backend-reference-given
           #:wrong-backend-reference-argument
           #:wrong-backend-reference-expected
           #:unsupported-argument
           #:unsupported-argument-backend
           #:unsupported-argument-argument
           #:unsupported-argument-reason
           #:missing-training-set
           #:missing-training-set-booster
           #:unfreed-handle-warning
           #:unfreed-handle-warning-kind
           #:data-error
           #:dimension-mismatch
           #:dimension-mismatch-expected
           #:dimension-mismatch-given
           #:unsupported-element-type
           #:unsupported-element-type-given
           #:untested-backend-version
           #:untested-backend-version-backend
           #:untested-backend-version-version
           #:untested-backend-version-tested
           #:unknown-capability
           #:unknown-capability-capability
           #:unknown-capability-known
           #:capability-unavailable
           #:capability-unavailable-capability
           #:backend-methods-not-loaded
           #:backend-methods-not-loaded-generic-function))

(in-package #:cl-gbdt/src/conditions)

(define-condition gbdt-error (error)
  ()
  (:documentation "Base type for every error cl-gbdt signals."))

(define-condition backend-error (gbdt-error)
  ((backend :initarg :backend
            :initform nil
            :reader backend-error-backend
            :documentation "Backend name, either `:lightgbm' or `:xgboost'."))
  (:documentation "Base type for errors during backend initialization or connection."))

(define-condition backend-library-not-found (backend-error)
  ((searched :initarg :searched
             :initform nil
             :reader backend-library-not-found-searched
             :documentation "List of paths that were searched."))
  (:report
   (lambda (condition stream)
     (format stream "Shared library for ~A not found. Searched:~{~%  ~A~}"
             (backend-error-backend condition)
             (backend-library-not-found-searched condition))))
  (:documentation "The backend's shared library could not be located."))

(define-condition backend-library-load-failed (backend-error)
  ((path :initarg :path :initform nil :reader backend-library-load-failed-path)
   (cause :initarg :cause :initform nil :reader backend-library-load-failed-cause))
  (:report
   (lambda (condition stream)
     (format stream "Failed to load the shared library ~A for ~A: ~A"
             (backend-library-load-failed-path condition)
             (backend-error-backend condition)
             (backend-library-load-failed-cause condition))))
  (:documentation "The shared library was found but could not be loaded."))

(define-condition missing-foreign-symbols (backend-error)
  ((names :initarg :names
          :initform nil
          :reader missing-foreign-symbols-names
          :documentation "List of C function names that were not found."))
  (:report
   (lambda (condition stream)
     (format stream
             "The ~A library is missing ~D function(s) cl-gbdt requires.~@
              The version is probably incompatible. Missing:~{~%  ~A~}"
             (backend-error-backend condition)
             (length (missing-foreign-symbols-names condition))
             (missing-foreign-symbols-names condition))))
  (:documentation "The loaded library lacks functions cl-gbdt needs.

Detected by symbol probing. This is the most reliable signal of a version mismatch."))

(define-condition backend-not-open (backend-error)
  ()
  (:report
   (lambda (condition stream)
     (format stream "Backend ~A is not open. Call OPEN-BACKEND first."
             (backend-error-backend condition))))
  (:documentation "An operation was attempted on a backend that is not open."))

(define-condition unknown-backend (backend-error)
  ((registered :initarg :registered
               :initform nil
               :reader unknown-backend-registered
               :documentation "List of currently registered backend names."))
  (:report
   (lambda (condition stream)
     (format stream "~A is not a registered backend. Registered:~{~%  ~A~}"
             (backend-error-backend condition)
             (unknown-backend-registered condition))))
  (:documentation "OPEN-BACKEND was called with a name no backend system has registered.

Distinct from `backend-not-open', which is about an operation attempted on a
backend instance that exists but has not been opened yet; this is about a name
that `find-backend-class' does not know at all."))

(define-condition foreign-call-error (gbdt-error)
  ((function-name :initarg :function-name
                  :initform nil
                  :reader foreign-call-error-function-name)
   (code :initarg :code :initform nil :reader foreign-call-error-code)
   (message :initarg :message :initform nil :reader foreign-call-error-message))
  (:report
   (lambda (condition stream)
     (format stream "~A returned ~A: ~A"
             (foreign-call-error-function-name condition)
             (foreign-call-error-code condition)
             (or (foreign-call-error-message condition) "(no message)"))))
  (:documentation "A C API call returned a non-zero status.

MESSAGE holds whatever LGBM_GetLastError or XGBGetLastError reported."))

(define-condition released-handle-error (gbdt-error)
  ((object :initarg :object :initform nil :reader released-handle-error-object))
  (:report
   (lambda (condition stream)
     (format stream "Attempted to use the already-released handle ~A."
             (released-handle-error-object condition))))
  (:documentation "A handle was used after it had been freed."))

(define-condition unfreed-handle-warning (warning)
  ((kind :initarg :kind :initform nil :reader unfreed-handle-warning-kind))
  (:report
   (lambda (condition stream)
     (format stream "A ~(~A~) handle was garbage-collected without being freed. ~
                     Wrap it in `with-dataset' or `with-booster', or call the matching ~
                     `free-' function."
             (or (unfreed-handle-warning-kind condition) "gbdt"))))
  (:documentation "A handle was collected while still holding a live foreign pointer.

Signalled from a finalizer, which reports and does **not** free: running the C free from
whatever thread the GC chose would give no ordering guarantee between a booster and the
dataset it holds, and `with-booster' nested inside `with-dataset' exists precisely to
guarantee that order."))

(define-condition data-error (gbdt-error)
  ()
  (:documentation "Base type for errors in the supplied data."))

(define-condition dimension-mismatch (data-error)
  ((expected :initarg :expected :initform nil :reader dimension-mismatch-expected)
   (given :initarg :given :initform nil :reader dimension-mismatch-given))
  (:report
   (lambda (condition stream)
     (format stream "Dimension mismatch. Expected: ~A, got: ~A"
             (dimension-mismatch-expected condition)
             (dimension-mismatch-given condition))))
  (:documentation "An array's rank or size differs from what was expected."))

(define-condition unsupported-element-type (data-error)
  ((given :initarg :given
          :initform nil
          :reader unsupported-element-type-given
          :documentation "Element type of the array that was supplied."))
  (:report
   (lambda (condition stream)
     (format stream
             "Element type ~A is not supported. Use DOUBLE-FLOAT or SINGLE-FLOAT."
             (unsupported-element-type-given condition))))
  (:documentation "An array with an unsupported element type was supplied."))

(define-condition wrong-backend-reference (data-error)
  ((backend :initarg :backend
            :initform nil
            :reader wrong-backend-reference-backend
            :documentation "Name of the backend the caller was operating against.")
   (given :initarg :given
          :initform nil
          :reader wrong-backend-reference-given
          :documentation "The class of the object actually passed where a handle of kind
EXPECTED, built by BACKEND, was expected.")
   (argument :initarg :argument
             :initform nil
             :reader wrong-backend-reference-argument
             :documentation "Description of which caller-supplied argument received the wrong
value, e.g. \"make-dataset's :reference\" or \"train's dataset argument\", for the report.")
   (expected :initarg :expected
             :initform "dataset"
             :reader wrong-backend-reference-expected
             :documentation "What kind of thing the argument should have been, as a noun
for the report -- \"dataset\" (the default, and every caller before `booster-eval' and
`booster-eval-names'), \"booster\", or \"backend\", the noun the two backends'
`%check-lightgbm-backend' and `%check-xgboost-backend' pass for a Layer 1 `create-dataset'
or `create-booster' handed the OTHER backend's object. Defaulting to \"dataset\" means the
two oldest callers, `%check-lightgbm-dataset' and `%check-xgboost-dataset', need no change
of their own to keep reporting exactly what they always have."))
  (:report
   (lambda (condition stream)
     (format stream "~A must be a ~A built by ~A itself, not ~A."
             (or (wrong-backend-reference-argument condition) "The argument")
             (wrong-backend-reference-expected condition)
             (wrong-backend-reference-backend condition)
             (wrong-backend-reference-given condition))))
  (:documentation "A caller-supplied dataset or booster argument did not belong to the same
backend as the operation it was passed to, or was the wrong kind of handle outright -- or,
for a Layer 1 operation that takes a BACKEND rather than a handle, that argument was another
backend's object.

Several entry points funnel through this same condition: `make-dataset''s :REFERENCE,
`train''s DATASET argument, and each entry of `train''s :VALID-SETS all expect a dataset,
EXPECTED's default; `cl-gbdt/lightgbm''s `booster-eval' and `booster-eval-names' expect a
booster instead, the first callers to pass `:expected \"booster\"' -- without that slot,
reusing this condition unmodified for them produced a self-contradictory report, e.g.
\"must be a dataset built by LIGHTGBM itself, not LIGHTGBM-DATASET\" when a dataset was
rejected because a *booster* was required. Every one of these entry points hands a
caller-supplied handle straight to a foreign function that expects a specific handle kind,
without a CLOS specializer to rule out the wrong one first -- `make-dataset' and `train'
both dispatch on the backend, not on the handle, and `booster-eval'/`booster-eval-names'
are plain functions with no CLOS dispatch at all, so a handle built by a different
backend, of the wrong kind, or not a handle at all, would otherwise reach the foreign call
unexamined. `cl-gbdt/src/lightgbm/api''s and `cl-gbdt/src/xgboost/api''s `create-dataset' and
`create-booster' are here for the same reason one step earlier: each was a `defmethod'
specialized on its own backend class, and as a plain `defun' takes whatever backend object it
is given -- which decides which library's entry point runs and which backend the resulting
handle records its liveness against.

A handle is an opaque pointer as far as the underlying C API is concerned: handing it the
wrong one is undefined behaviour once it crosses the FFI boundary, not something the C
library can reject on its own. This is checked here, before any foreign call, so the
failure is a condition instead of a crash or silent corruption."))

(define-condition unsupported-argument (data-error)
  ((backend :initarg :backend
            :initform nil
            :reader unsupported-argument-backend
            :documentation "Name of the backend the argument was passed to.")
   (argument :initarg :argument
             :initform nil
             :reader unsupported-argument-argument
             :documentation "Description of which caller-supplied argument is unsupported,
e.g. \"make-dataset's :reference\", for the report.")
   (reason :initarg :reason
           :initform nil
           :reader unsupported-argument-reason
           :documentation "Why BACKEND cannot honor the argument."))
  (:report
   (lambda (condition stream)
     (format stream "~A is not supported by ~A~@[: ~A~]."
             (or (unsupported-argument-argument condition) "The argument")
             (unsupported-argument-backend condition)
             (unsupported-argument-reason condition))))
  (:documentation "A caller supplied an argument that BACKEND has no way to honor.

Unlike `wrong-backend-reference', which is about a caller-supplied *handle* built by the
wrong backend, this is about an argument naming a concept the backend does not have at all
-- `make-dataset''s :REFERENCE on XGBoost, for example, which has no bin-mapper to align to.
Signalling here, instead of silently discarding the argument, is deliberate: this project has
repeatedly found a silently-dropped argument to be the failure mode where a caller moves
working code from one backend to the other and gets different, wrong behaviour with no
indication anything changed."))

(define-condition missing-training-set (data-error)
  ((booster :initarg :booster
            :initform nil
            :reader missing-training-set-booster
            :documentation "The booster UPDATE-ONE-ITERATION was called on."))
  (:report
   (lambda (condition stream)
     (format stream "~A has no training set to advance with UPDATE-ONE-ITERATION -- ~
                     was it returned by LOAD-MODEL?"
             (missing-training-set-booster condition))))
  (:documentation "UPDATE-ONE-ITERATION was called on a booster with no training set.

A `load-model' booster has no training set -- see the `booster' class' documentation --
so there is no dataset to advance it against. XGBoost's `XGBoosterUpdateOneIter' takes
the training DMatrix as an explicit argument, unlike LightGBM's, which reads its own
internal training-set pointer implicitly; a null DMatrixHandle would not come back as a
status code the way a bad parameter does, but as a null-pointer dereference inside
XGBoost's own implementation. This is signalled before that foreign call runs, for the
same reason `%check-booster-datasets-live' checks the pointers it does."))

(define-condition untested-backend-version (warning)
  ((backend :initarg :backend :initform nil :reader untested-backend-version-backend)
   (version :initarg :version :initform nil :reader untested-backend-version-version)
   (tested :initarg :tested :initform nil :reader untested-backend-version-tested))
  (:report
   (lambda (condition stream)
     (format stream
             "~A version ~A has not been tested with cl-gbdt. Tested: ~{~A~^, ~}~@
              If something misbehaves, switch to a tested version."
             (untested-backend-version-backend condition)
             (untested-backend-version-version condition)
             (untested-backend-version-tested condition))))
  (:documentation "A library version outside the tested list was loaded.

This is a warning rather than an error so that a new upstream release does not
immediately render cl-gbdt unusable."))

(define-condition unknown-capability (gbdt-error)
  ((capability :initarg :capability
               :reader unknown-capability-capability
               :documentation "The keyword the caller asked about.")
   (known :initarg :known
          :initform nil
          :reader unknown-capability-known
          :documentation "Every capability name this build knows."))
  (:report (lambda (condition stream)
             (format stream "~S is not a known capability. Known capabilities: ~{~S~^, ~}."
                     (unknown-capability-capability condition)
                     (unknown-capability-known condition))))
  (:documentation "Signalled when a capability keyword is not in `*known-capabilities*'.

A programming error, not a backend limitation: returning NIL for a misspelled capability
would be indistinguishable from \"this backend does not support it\", and a caller that
believes the second silently skips the feature. Policy section 7 forbids exactly that
silent path."))

(define-condition capability-unavailable (backend-error)
  ((capability :initarg :capability
               :reader capability-unavailable-capability
               :documentation "The capability the operation needed."))
  (:report (lambda (condition stream)
             (format stream "~A does not provide ~S in the library that is loaded."
                     (backend-error-backend condition)
                     (capability-unavailable-capability condition))))
  (:documentation "Signalled when an operation needs a capability this backend does not have.

Distinct from `unknown-capability': the question was well formed and the answer is no.
Policy section 7 requires the operation itself to signal this rather than relying on the
caller having asked `backend-supports-p' first, and forbids falling back to some other
behaviour instead."))

(define-condition backend-methods-not-loaded (backend-error)
  ((generic-function :initarg :generic-function
                     :reader backend-methods-not-loaded-generic-function
                     :documentation "The unified-API generic function that was called."))
  (:report (lambda (condition stream)
             (let ((backend (backend-error-backend condition)))
               (format stream
                       "~S has no method for ~A: that backend's unified-API methods are ~
                        not loaded. Load cl-gbdt/~(~A~)/unified -- cl-gbdt/~(~A~) is ~
                        that backend's own API alone, without cl-gbdt's portable one."
                       (backend-methods-not-loaded-generic-function condition)
                       backend backend backend))))
  (:documentation "Signalled when a unified-API generic function is called on a backend
whose methods for it were never loaded.

`cl-gbdt/lightgbm' registers the backend class and opens the library; `cl-gbdt/lightgbm/unified'
is what adds the methods implementing `cl-gbdt''s portable generic functions. A program that
loaded the first and not the second can open a backend and then find no method to call, and
this condition is that state said plainly. Without it the caller sees the implementation's own
no-applicable-method error, which names neither the backend nor the system to load.

Never signalled while both systems are loaded: each fallback method is specialized on a base
class, so a backend's own method is always more specific."))
