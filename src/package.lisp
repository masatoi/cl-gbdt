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
   #:untested-backend-version
   ;; Data handoff
   #:foreign-matrix
   #:foreign-matrix-pointer
   #:foreign-matrix-rows
   #:foreign-matrix-cols
   #:foreign-matrix-element-type
   #:call-with-foreign-matrix
   #:with-foreign-matrix
   #:foreign-element-type
   ;; Backends
   #:backend
   #:backend-name
   #:backend-library-path
   #:backend-capabilities
   #:backend-version
   #:backend-open-p
   #:backend-info
   #:register-backend
   #:find-backend-class
   #:open-backend
   #:close-backend
   #:initialize-backend
   #:shutdown-backend
   #:probe-foreign-symbols
   ;; Unified API
   #:make-dataset
   #:dataset-num-rows
   #:dataset-num-features
   #:train
   #:update-one-iteration
   #:predict
   #:save-model
   #:load-model
   #:model-to-string
   #:feature-importance
   #:free-dataset
   #:free-booster
   #:with-dataset
   #:with-booster))
