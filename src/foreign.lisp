;;;; foreign.lisp --- Shared foreign-call status checking, used by every backend.
;;;;
;;;; LightGBM and XGBoost both report failure the same way: a C function returns 0 on
;;;; success and a nonzero status on failure, with the detail available from a
;;;; `*GetLastError' entry point returning `char *'. c2ffi discards `const', so both
;;;; generated bindings render that return type as `:pointer' regardless of which
;;;; library declares it -- decoding the pointer into a Lisp string, and guarding
;;;; against it being null, is each backend's own job, since the entry point's name
;;;; is the only part that differs between them.

(uiop:define-package #:cl-gbdt/src/foreign
  (:use #:cl)
  (:import-from #:cl-gbdt/src/conditions
                #:foreign-call-error)
  (:export #:check-foreign-call
           #:with-foreign-float-traps-masked))

(in-package #:cl-gbdt/src/foreign)

(defmacro with-foreign-float-traps-masked (&body body)
  "Evaluate BODY with the :INVALID, :DIVIDE-BY-ZERO and :OVERFLOW floating-point traps
masked, restoring whatever trap set was in effect once BODY returns or unwinds.

Both wrapped libraries are C code, written and tested against the ordinary C
floating-point environment, where these three IEEE-754 exceptions are masked by
default: an intermediate NaN or infinity produced partway through a computation is
data to keep working with, not a signal. SBCL does not honor that convention
uniformly -- confirmed empirically: x86-64 SBCL enables all three traps by default,
aarch64 SBCL enables none of them -- so a call into LightGBM's or XGBoost's own
numeric code (the softmax normalization behind an XGBoost `multi:softprob' prediction
is the case that surfaced this) can take an intermediate value the library was written
to tolerate and, only on x86-64, turn it into an uncaught SBCL
FLOATING-POINT-INVALID-OPERATION in the middle of a foreign call.

Masking here restores the environment every call into these libraries was always
supposed to run in; it does not suppress a real error. SBCL enabling these traps is
the anomaly this macro corrects for, not the library's own arithmetic. It is a
correction, not a general license to ignore floating-point exceptions: callers that
mask a *caller-supplied* computation, not just a foreign call, would be hiding their
own bugs instead of restoring someone else's calling convention.

Every entry point in `cl-gbdt/src/xgboost/backend' and `cl-gbdt/src/lightgbm/backend'
wraps its entire body in this macro, not just the specific call this was first found
through -- see those files' commentary for the enumeration.

Expands to a plain PROGN under #-SBCL: this project's foreign-call boundary is
SBCL-specific throughout already (see `cl-gbdt/src/data''s `%call-with-pinned-matrix'
for the same split), and no other implementation this project targets is known to
enable these traps by default."
  #+sbcl
  `(sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow) ,@body)
  #-sbcl
  `(progn ,@body))

(defun check-foreign-call (code function-name last-error)
  "Signal `foreign-call-error' when CODE reports failure, otherwise return CODE.

Both wrapped libraries use the same idiom: 0 on success, nonzero on failure, with the
detail behind a `*GetLastError' entry point returning a `char *'. LAST-ERROR is a
function of no arguments returning that message as a string, or NIL -- it is the only
part that differs between backends, so it is the only part passed in. FUNCTION-NAME
identifies which C function reported CODE, for the condition's report."
  (if (zerop code)
      code
      (error 'foreign-call-error
             :function-name function-name
             :code code
             :message (funcall last-error))))
