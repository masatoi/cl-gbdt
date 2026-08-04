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
  (:export #:check-foreign-call))

(in-package #:cl-gbdt/src/foreign)

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
