;;;; lightgbm.lisp --- Round-trip test against the real LightGBM library.

(in-package #:cl-gbdt/functional-tests)

(defun lgbm-last-error ()
  "Return LightGBM's last error message as a Lisp string."
  (cffi:foreign-string-to-lisp (lgbm::lgbm-get-last-error)))

(defmacro lgbm-check (form)
  "Evaluate FORM, a LightGBM call, and assert it returned the success status.

On failure the report carries the library's own message, which is far more useful than
the bare status code."
  `(let ((status ,form))
     (ok (zerop status)
         (format nil "~A returned ~D~@[: ~A~]"
                 ',(first form) status (unless (zerop status) (lgbm-last-error))))))

(deftest lightgbm-library-loads
  (with-backend-library (:lightgbm)
    (testing "the shared library loads standalone from its vendored layout"
      (ok (find "lib_lightgbm.so" (cffi:list-foreign-libraries)
                :key (lambda (library)
                       (file-namestring (cffi:foreign-library-pathname library)))
                :test #'string=)
          "lib_lightgbm.so is among the loaded foreign libraries"))
    (testing "a string return value crosses the boundary intact"
      (let ((message (lgbm-last-error)))
        (ok (stringp message))
        (ok (plusp (length message)) (format nil "message was ~S" message))))))

(deftest lightgbm-trains-and-predicts
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix labels) (make-separable-dataset)
      (let ((rows (array-dimension matrix 0))
            (dataset nil)
            (booster nil))
        (unwind-protect
             (progn
               (testing "a dataset is built from a 2D double-float array"
                 (cffi:with-foreign-object (out :pointer)
                   (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
                     (declare (ignore element-type))
                     (cffi:with-foreign-string (parameters
                                                "min_data_in_leaf=1 min_data_in_bin=1 verbose=-1")
                       (lgbm-check (lgbm::lgbm-dataset-create-from-mat
                                    pointer 1 nrow ncol 1 parameters
                                    (cffi:null-pointer) out))))
                   (setf dataset (cffi:mem-ref out :pointer))
                   (ok (not (cffi:null-pointer-p dataset)) "the dataset handle is non-null")))

               (testing "labels attach to the dataset"
                 (sb-sys:with-pinned-objects (labels)
                   (let ((pointer (cffi:make-pointer
                                   (sb-sys:sap-int (sb-sys:vector-sap labels)))))
                     (cffi:with-foreign-string (field "label")
                       (lgbm-check (lgbm::lgbm-dataset-set-field
                                    dataset field pointer rows 0))))))

               (testing "five boosting rounds run and the iteration count reads back"
                 (cffi:with-foreign-object (out :pointer)
                   (cffi:with-foreign-string (parameters
                                              "objective=binary num_leaves=2 min_data_in_leaf=1 min_data_in_bin=1 verbose=-1")
                     (lgbm-check (lgbm::lgbm-booster-create dataset parameters out)))
                   (setf booster (cffi:mem-ref out :pointer)))
                 (cffi:with-foreign-object (finished :int)
                   (dotimes (iteration 5)
                     (lgbm-check (lgbm::lgbm-booster-update-one-iter booster finished))))
                 (cffi:with-foreign-object (iterations :int)
                   (lgbm-check (lgbm::lgbm-booster-get-current-iteration booster iterations))
                   (ok (= 5 (cffi:mem-ref iterations :int))
                       (format nil "iteration count is ~D" (cffi:mem-ref iterations :int)))))

               (testing "predictions come back with the right shape and ordering"
                 (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
                   (declare (ignore element-type))
                   (cffi:with-foreign-objects ((length :int64) (out :double rows))
                     (cffi:with-foreign-string (parameters "")
                       (lgbm-check (lgbm::lgbm-booster-predict-for-mat
                                    booster pointer 1 nrow ncol 1 0 0 -1 parameters
                                    length out)))
                     (ok (= rows (cffi:mem-ref length :int64))
                         (format nil "prediction count is ~D, expected ~D"
                                 (cffi:mem-ref length :int64) rows))
                     (let ((predictions (make-array rows :element-type 'double-float)))
                       (dotimes (index rows)
                         (setf (aref predictions index) (cffi:mem-aref out :double index)))
                       (ok (predictions-separate-p predictions labels)
                           (format nil "predictions separate by label: ~S" predictions)))))))
          (when booster (lgbm::lgbm-booster-free booster))
          (when dataset (lgbm::lgbm-dataset-free dataset)))))))
