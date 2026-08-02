;;;; xgboost.lisp --- Round-trip test against the real XGBoost library.

(in-package #:cl-gbdt/functional-tests)

(defun xgb-last-error ()
  "Return XGBoost's last error message as a Lisp string."
  (cffi:foreign-string-to-lisp (xgb::xgb-get-last-error)))

(defmacro xgb-check (form)
  "Evaluate FORM, an XGBoost call, and assert it returned the success status."
  `(let ((status ,form))
     (ok (zerop status)
         (format nil "~A returned ~D~@[: ~A~]"
                 ',(first form) status (unless (zerop status) (xgb-last-error))))))

(defun set-booster-parameters (booster parameters)
  "Apply PARAMETERS, an alist of name and value strings, to BOOSTER."
  (loop :for (name . value) :in parameters
        :do (cffi:with-foreign-string (foreign-name name)
              (cffi:with-foreign-string (foreign-value value)
                (xgb::xg-booster-set-param booster foreign-name foreign-value)))))

(deftest xgboost-library-loads
  (with-backend-library (:xgboost)
    (testing "the shared library loads standalone from its vendored layout"
      (ok (find "libxgboost.so" (cffi:list-foreign-libraries)
                :key (lambda (library)
                       (file-namestring (cffi:foreign-library-pathname library)))
                :test #'string=)
          "libxgboost.so is among the loaded foreign libraries"))
    (testing "the version reads back through three out-parameters"
      (cffi:with-foreign-objects ((major :int) (minor :int) (patch :int))
        (xgb::xg-boost-version major minor patch)
        (let ((version (list (cffi:mem-ref major :int)
                             (cffi:mem-ref minor :int)
                             (cffi:mem-ref patch :int))))
          (ok (every #'integerp version) (format nil "version is ~S" version))
          (ok (plusp (first version)) "the major version is positive"))))))

(deftest xgboost-trains-and-predicts
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix labels) (make-separable-dataset)
      (let ((rows (array-dimension matrix 0))
            (dmatrix nil)
            (booster nil))
        (unwind-protect
             (progn
               (testing "a DMatrix is built through the array interface"
                 (cffi:with-foreign-object (out :pointer)
                   (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
                     (declare (ignore element-type))
                     (cffi:with-foreign-string
                         (data (array-interface-json pointer "<f8" nrow ncol))
                       (cffi:with-foreign-string (config "{\"missing\":NaN}")
                         (xgb-check (xgb::xgd-matrix-create-from-dense data config out)))))
                   (setf dmatrix (cffi:mem-ref out :pointer))
                   (ok (not (cffi:null-pointer-p dmatrix)) "the DMatrix handle is non-null")))

               (testing "labels attach through the array interface"
                 (sb-sys:with-pinned-objects (labels)
                   (let ((pointer (cffi:make-pointer
                                   (sb-sys:sap-int (sb-sys:vector-sap labels)))))
                     (cffi:with-foreign-string (field "label")
                       (cffi:with-foreign-string
                           (descriptor (array-interface-json pointer "<f4" rows))
                         (xgb-check (xgb::xgd-matrix-set-info-from-interface
                                     dmatrix field descriptor)))))))

               (testing "five boosting rounds run"
                 (cffi:with-foreign-objects ((out :pointer) (matrices :pointer 1))
                   (setf (cffi:mem-aref matrices :pointer 0) dmatrix)
                   (xgb-check (xgb::xg-booster-create matrices 1 out))
                   (setf booster (cffi:mem-ref out :pointer)))
                 (set-booster-parameters booster
                                         '(("objective" . "binary:logistic")
                                           ("max_depth" . "2")
                                           ("eta" . "0.5")
                                           ("verbosity" . "0")))
                 (dotimes (iteration 5)
                   (xgb-check (xgb::xg-booster-update-one-iter booster iteration dmatrix))))

               (testing "predictions come back with the right shape and ordering"
                 (cffi:with-foreign-objects ((length :uint64) (out :pointer))
                   (xgb-check (xgb::xg-booster-predict booster dmatrix 0 0 0 length out))
                   (ok (= rows (cffi:mem-ref length :uint64))
                       (format nil "prediction count is ~D, expected ~D"
                               (cffi:mem-ref length :uint64) rows))
                   (let ((buffer (cffi:mem-ref out :pointer))
                         (predictions (make-array rows :element-type 'single-float)))
                     (dotimes (index rows)
                       (setf (aref predictions index) (cffi:mem-aref buffer :float index)))
                     (ok (predictions-separate-p predictions labels)
                         (format nil "predictions separate by label: ~S" predictions))))))
          (when booster (xgb::xg-booster-free booster))
          (when dmatrix (xgb::xgd-matrix-free dmatrix)))))))
