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
           #:data-error
           #:dimension-mismatch
           #:dimension-mismatch-expected
           #:dimension-mismatch-given
           #:unsupported-element-type
           #:unsupported-element-type-given
           #:untested-backend-version
           #:untested-backend-version-backend
           #:untested-backend-version-version
           #:untested-backend-version-tested))

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
