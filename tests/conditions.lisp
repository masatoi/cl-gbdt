;;;; conditions.lisp --- Tests for the condition hierarchy.

(in-package #:cl-gbdt/tests)

(deftest condition-hierarchy
  (testing "every condition inherits from gbdt-error"
    (dolist (type '(cl-gbdt:backend-error
                    cl-gbdt:backend-library-not-found
                    cl-gbdt:backend-library-load-failed
                    cl-gbdt:missing-foreign-symbols
                    cl-gbdt:backend-not-open
                    cl-gbdt:foreign-call-error
                    cl-gbdt:released-handle-error
                    cl-gbdt:data-error
                    cl-gbdt:dimension-mismatch
                    cl-gbdt:unsupported-element-type))
      (ok (subtypep type 'cl-gbdt:gbdt-error)
          (format nil "~A is a subtype of gbdt-error" type))))
  (testing "untested-backend-version is a warning, not an error"
    (ok (subtypep 'cl-gbdt:untested-backend-version 'warning))
    (ng (subtypep 'cl-gbdt:untested-backend-version 'error))))

(deftest missing-foreign-symbols-reports-names
  (testing "the missing function names appear in the report"
    (let ((text (handler-case
                    (error 'cl-gbdt:missing-foreign-symbols
                           :backend :lightgbm
                           :names '("LGBM_GetMaxThreads" "LGBM_SetMaxThreads"))
                  (cl-gbdt:missing-foreign-symbols (c)
                    (princ-to-string c)))))
      (ok (search "LGBM_GetMaxThreads" text))
      (ok (search "LGBM_SetMaxThreads" text))
      (ok (search "2" text) "the count is included"))))

(deftest foreign-call-error-carries-backend-message
  (testing "the C-side error message is retained and shown in the report"
    (let ((condition (make-condition 'cl-gbdt:foreign-call-error
                                     :function-name "LGBM_BoosterCreate"
                                     :code -1
                                     :message "Cannot create Booster")))
      (ok (equal "LGBM_BoosterCreate"
                 (cl-gbdt:foreign-call-error-function-name condition)))
      (ok (eql -1 (cl-gbdt:foreign-call-error-code condition)))
      (ok (search "Cannot create Booster" (princ-to-string condition))))))

(deftest backend-library-not-found-lists-searched-paths
  (testing "the searched paths appear in the report"
    (let ((text (princ-to-string
                 (make-condition 'cl-gbdt:backend-library-not-found
                                 :backend :xgboost
                                 :searched '("/usr/local/lib/libxgboost.so"
                                             "/opt/lib/libxgboost.so")))))
      (ok (search "/usr/local/lib/libxgboost.so" text))
      (ok (search "/opt/lib/libxgboost.so" text)))))

(deftest unsupported-element-type-names-the-type
  (testing "the offending element type appears in the report"
    (let ((text (princ-to-string
                 (make-condition 'cl-gbdt:unsupported-element-type
                                 :given '(unsigned-byte 8)))))
      (ok (search "UNSIGNED-BYTE" text))
      (ok (search "DOUBLE-FLOAT" text) "the supported type is suggested"))))
