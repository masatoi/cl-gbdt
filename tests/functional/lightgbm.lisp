;;;; lightgbm.lisp --- Round-trip test against the real LightGBM library.

(uiop:define-package #:cl-gbdt/tests/functional/lightgbm
  (:use #:cl #:rove)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt)
  ;; The generated FFI packages deliberately export nothing -- their own docstrings say
  ;; nothing outside the backend systems should call them directly. These tests are the
  ;; exception: their whole purpose is to exercise that raw layer. They therefore reach in
  ;; with a double colon, which keeps the trespass visible at every call site rather than
  ;; hiding it behind an export list the design does not want.
  (:local-nicknames (#:lgbm #:cl-gbdt/src/lightgbm/c-api)
                     ;; F1 needs *default-library-name*, an internal (unexported) special
                     ;; -- reached the same double-colon way as lgbm:: above. It lives in
                     ;; native.lisp (Layer 1), not protocol.lisp, since this branch's
                     ;; Task 3 split the old backend.lisp in two.
                     (#:lightgbm-native #:cl-gbdt/src/lightgbm/native))
  (:import-from #:cl-gbdt/tests/functional/support
                #:backend-library-path
                #:with-backend-library
                #:resolve-via-cffi-default
                #:make-separable-dataset
                #:predictions-separate-p))

(in-package #:cl-gbdt/tests/functional/lightgbm)

(defun lgbm-last-error ()
  "Return LightGBM's last error message as a Lisp string."
  (cffi:foreign-string-to-lisp (lgbm::lgbm-get-last-error)))

(defmacro lgbm-check (form)
  "Evaluate FORM, a LightGBM call, and assert it returned the success status.

On failure the report carries the library's own message, which is far more useful than
the bare status code.
Returns what `ok' returns, so a caller can gate the rest of a sequence on it."
  `(let ((status ,form))
     (ok (zerop status)
         (format nil "~A returned ~D~@[: ~A~]"
                 ',(first form) status (unless (zerop status) (lgbm-last-error))))))

(deftest lightgbm-library-loads
  (with-backend-library (:lightgbm)
    (testing "the shared library loads standalone from its vendored layout"
      (let ((basename (file-namestring (backend-library-path :lightgbm))))
        (ok (find basename (cffi:list-foreign-libraries)
                  :key (lambda (library)
                         (let ((pathname (cffi:foreign-library-pathname library)))
                           (and pathname (file-namestring pathname))))
                  :test #'equal)
            (format nil "~A is among the loaded foreign libraries" basename))))
    (testing "a string return value crosses the boundary intact"
      (let ((message (lgbm-last-error)))
        (ok (stringp message))
        (ok (plusp (length message)) (format nil "message was ~S" message))))))

;;; F1: *default-library-name* used to be "_lightgbm". Same measurement as
;;; tests/functional/xgboost.lisp's identical test: CFFI's `:default' designator only
;;; appends the platform's shared-library suffix -- it never adds a `lib' prefix -- so the
;;; compiled basename alone ("_lightgbm") never resolves and the full on-disk basename
;;; ("lib_lightgbm") must.

(deftest lightgbm-default-library-name-resolves-through-cffi-default
  (with-backend-library (:lightgbm)
    (testing "the :default designator resolves *default-library-name* to the vendored file"
      (let ((resolved (resolve-via-cffi-default
                        :lightgbm lightgbm-native::*default-library-name*)))
        (ok resolved
            (format nil "(:default ~S) did not resolve to any file"
                    lightgbm-native::*default-library-name*))
        (when resolved
          (ok (equal (file-namestring resolved)
                      (file-namestring (backend-library-path :lightgbm)))
              (format nil "resolved to ~A, expected the vendored ~A"
                      resolved (backend-library-path :lightgbm))))))))

(defparameter *dataset-parameters*
  "min_data_in_leaf=1 min_data_in_bin=1 verbose=-1"
  "LightGBM dataset parameters for the eight-row fixture.

The two minima are required: the defaults refuse to bin or split so few rows, which
would leave every prediction identical and fail the separation assertion for a reason
that has nothing to do with the FFI. `verbose=-1' keeps the library off standard
output during the suite.")

(defparameter *booster-parameters*
  "objective=binary num_leaves=2 min_data_in_leaf=1 min_data_in_bin=1 verbose=-1"
  "LightGBM booster parameters for the eight-row fixture. See *dataset-parameters*.")

(deftest lightgbm-trains-and-predicts
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((rows (array-dimension matrix 0))
            (dataset nil)
            (booster nil))
        (unwind-protect
             ;; Each handle-creating step gates the rest. rove's `ok' records a failure
             ;; and returns rather than unwinding, so without these guards a failed
             ;; creation would carry a null or garbage handle into the next C call --
             ;; and a segfault there would take the whole process down, skipping this
             ;; `unwind-protect' and leaking whatever had been allocated.
             (block round-trip
               (testing "a dataset is built from a 2D double-float array"
                 (cffi:with-foreign-object (out :pointer)
                   (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
                     (declare (ignore element-type))
                     (cffi:with-foreign-string (parameters *dataset-parameters*)
                       (unless (lgbm-check (lgbm::lgbm-dataset-create-from-mat
                                            pointer 1 nrow ncol 1 parameters
                                            (cffi:null-pointer) out))
                         (return-from round-trip))))
                   (setf dataset (cffi:mem-ref out :pointer))
                   (unless (ok (not (cffi:null-pointer-p dataset))
                               "the dataset handle is non-null")
                     (return-from round-trip))))

               (testing "labels attach to the dataset"
                 (sb-sys:with-pinned-objects (label-vector)
                   (let ((pointer (cffi:make-pointer
                                   (sb-sys:sap-int (sb-sys:vector-sap label-vector)))))
                     (cffi:with-foreign-string (field "label")
                       (unless (lgbm-check (lgbm::lgbm-dataset-set-field
                                            dataset field pointer rows 0))
                         (return-from round-trip))))))

               (testing "five boosting rounds run and the iteration count reads back"
                 (cffi:with-foreign-object (out :pointer)
                   (cffi:with-foreign-string (parameters *booster-parameters*)
                     (unless (lgbm-check (lgbm::lgbm-booster-create dataset parameters out))
                       (return-from round-trip)))
                   (setf booster (cffi:mem-ref out :pointer))
                   (unless (ok (not (cffi:null-pointer-p booster))
                               "the booster handle is non-null")
                     (return-from round-trip)))
                 (cffi:with-foreign-object (finished :int)
                   (dotimes (iteration 5)
                     ;; `ignorable', not `ignore': `dotimes' itself reads and sets ITERATION
                     ;; to drive the loop even though this body never touches it, so
                     ;; `ignore' conflicts with the macroexpansion's own use and produces
                     ;; three STYLE-WARNINGs, not the "this really is unused" declaration it
                     ;; looks like -- matching `cl-gbdt/src/lightgbm/protocol''s `train' and
                     ;; `cl-gbdt/src/xgboost/protocol''s `train', which both hit the same
                     ;; thing and both already use `ignorable' for it.
                     (declare (ignorable iteration))
                     (lgbm-check (lgbm::lgbm-booster-update-one-iter booster finished))))
                 (cffi:with-foreign-object (iterations :int)
                   (lgbm-check (lgbm::lgbm-booster-get-current-iteration booster iterations))
                   (ok (= 5 (cffi:mem-ref iterations :int))
                       (format nil "iteration count is ~D" (cffi:mem-ref iterations :int)))))

               (testing "predictions come back with the right shape and ordering"
                 (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
                   (declare (ignore element-type))
                   ;; `rows' is the correct output-buffer length only because this
                   ;; fixture's objective is "binary", where LightGBM's num_class is 1
                   ;; and the required length is num_class * num_data (c_api.h:1307-1311).
                   ;; Production code must not assume num_class = 1 and instead size the
                   ;; buffer from `LGBM_BoosterCalcNumPredict', bound as
                   ;; `lgbm-booster-calc-num-predict' at src/lightgbm/c-api.lisp:423.
                   (cffi:with-foreign-objects ((prediction-count :int64) (out :double rows))
                     (cffi:with-foreign-string (parameters "")
                       (unless (lgbm-check (lgbm::lgbm-booster-predict-for-mat
                                            booster pointer 1 nrow ncol 1 0 0 -1
                                            parameters prediction-count out))
                         (return-from round-trip)))
                     ;; `out' is our own with-foreign-objects allocation of ROWS doubles;
                     ;; a short count leaves its remainder uninitialised, so the copy
                     ;; below must not run when the count assertion fails.
                     (unless (ok (= rows (cffi:mem-ref prediction-count :int64))
                                 (format nil "prediction count is ~D, expected ~D"
                                         (cffi:mem-ref prediction-count :int64) rows))
                       (return-from round-trip))
                     (let ((predictions (make-array rows :element-type 'double-float)))
                       (dotimes (index rows)
                         (setf (aref predictions index) (cffi:mem-aref out :double index)))
                       (ok (predictions-separate-p predictions label-vector)
                           (format nil "predictions separate by label: ~S" predictions)))))))
          (when booster (lgbm::lgbm-booster-free booster))
          (when dataset (lgbm::lgbm-dataset-free dataset)))))))
