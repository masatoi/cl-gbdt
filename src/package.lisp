;;;; package.lisp --- Package definition for the cl-gbdt core.

(defpackage #:cl-gbdt
  (:use #:cl)
  (:export
   ;; Conditions
   #:gbdt-error
   #:backend-error
   #:backend-error-backend
   #:backend-library-not-found
   #:backend-library-not-found-searched
   #:backend-library-load-failed
   #:missing-foreign-symbols
   #:missing-foreign-symbols-names
   #:backend-not-open
   #:foreign-call-error
   #:foreign-call-error-code
   #:foreign-call-error-message
   #:foreign-call-error-function-name
   #:released-handle-error
   #:data-error
   #:dimension-mismatch
   #:unsupported-element-type
   #:unsupported-element-type-given
   #:untested-backend-version))
