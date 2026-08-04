;;;; xgboost.lisp --- Round-trip test against the real XGBoost library.

(uiop:define-package #:cl-gbdt/tests/functional/xgboost
  (:use #:cl #:rove)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt)
  ;; The generated FFI packages deliberately export nothing -- their own docstrings say
  ;; nothing outside the backend systems should call them directly. These tests are the
  ;; exception: their whole purpose is to exercise that raw layer. They therefore reach in
  ;; with a double colon, which keeps the trespass visible at every call site rather than
  ;; hiding it behind an export list the design does not want.
  (:local-nicknames (#:xgb #:cl-gbdt/src/xgboost/c-api))
  (:import-from #:cl-gbdt/tests/functional/support
                #:backend-library-path
                #:with-backend-library
                #:make-separable-dataset
                #:predictions-separate-p
                #:array-interface-json))

(in-package #:cl-gbdt/tests/functional/xgboost)

(defun xgb-last-error ()
  "Return XGBoost's last error message as a Lisp string."
  (cffi:foreign-string-to-lisp (xgb::xgb-get-last-error)))

(defmacro xgb-check (form)
  "Evaluate FORM, an XGBoost call, and assert it returned the success status.

Returns what `ok' returns, so a caller can gate the rest of a sequence on it."
  `(let ((status ,form))
     (ok (zerop status)
         (format nil "~A returned ~D~@[: ~A~]"
                 ',(first form) status (unless (zerop status) (xgb-last-error))))))

(defun set-booster-parameters (booster parameters)
  "Apply PARAMETERS, an alist of name and value strings, to BOOSTER.

Asserts every `XGBoosterSetParam' call rather than discarding its status: a silently
rejected parameter would leave the booster training under different settings than
the test wired up, while every later call kept returning success regardless."
  (loop :for (name . value) :in parameters
        :do (cffi:with-foreign-string (foreign-name name)
              (cffi:with-foreign-string (foreign-value value)
                (xgb-check (xgb::xg-booster-set-param booster foreign-name foreign-value))))))

(deftest xgboost-library-loads
  (with-backend-library (:xgboost)
    (testing "the shared library loads standalone from its vendored layout"
      (let ((basename (file-namestring (backend-library-path :xgboost))))
        (ok (find basename (cffi:list-foreign-libraries)
                  :key (lambda (library)
                         (let ((pathname (cffi:foreign-library-pathname library)))
                           (and pathname (file-namestring pathname))))
                  :test #'equal)
            (format nil "~A is among the loaded foreign libraries" basename))))
    (testing "the version reads back through three out-parameters"
      (cffi:with-foreign-objects ((major :int) (minor :int) (patch :int))
        (xgb::xg-boost-version major minor patch)
        (let ((version (list (cffi:mem-ref major :int)
                             (cffi:mem-ref minor :int)
                             (cffi:mem-ref patch :int))))
          (ok (and (every (lambda (component) (>= component 0)) version)
                   (>= (first version) 1))
              (format nil "version is a plausible triple: ~S" version))
          (ok (plusp (first version)) "the major version is positive"))))))

(deftest xgboost-trains-and-predicts
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((rows (array-dimension matrix 0))
            (dmatrix nil)
            (booster nil))
        (unwind-protect
             ;; Gated the same way as the LightGBM round trip: rove's `ok' records a
             ;; failure and returns rather than unwinding, so without these an invalid
             ;; handle would reach the next C call, and a segfault there would kill the
             ;; process, skip this `unwind-protect' and leak.
             (block round-trip
               (testing "a DMatrix is built through the array interface"
                 (cffi:with-foreign-object (out :pointer)
                   (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
                     (declare (ignore element-type))
                     (cffi:with-foreign-string
                         (data (array-interface-json pointer "<f8" nrow ncol))
                       (cffi:with-foreign-string (config "{\"missing\":NaN}")
                         (unless (xgb-check (xgb::xgd-matrix-create-from-dense
                                             data config out))
                           (return-from round-trip)))))
                   (setf dmatrix (cffi:mem-ref out :pointer))
                   (unless (ok (not (cffi:null-pointer-p dmatrix))
                               "the DMatrix handle is non-null")
                     (return-from round-trip))))

               (testing "labels attach through the array interface"
                 (sb-sys:with-pinned-objects (label-vector)
                   (let ((pointer (cffi:make-pointer
                                   (sb-sys:sap-int (sb-sys:vector-sap label-vector)))))
                     (cffi:with-foreign-string (field "label")
                       (cffi:with-foreign-string
                           (descriptor (array-interface-json pointer "<f4" rows))
                         (unless (xgb-check (xgb::xgd-matrix-set-info-from-interface
                                             dmatrix field descriptor))
                           (return-from round-trip)))))))

               (testing "five boosting rounds run and the iteration count reads back"
                 (cffi:with-foreign-objects ((out :pointer) (matrices :pointer 1))
                   (setf (cffi:mem-aref matrices :pointer 0) dmatrix)
                   (unless (xgb-check (xgb::xg-booster-create matrices 1 out))
                     (return-from round-trip))
                   (setf booster (cffi:mem-ref out :pointer))
                   (unless (ok (not (cffi:null-pointer-p booster))
                               "the booster handle is non-null")
                     (return-from round-trip)))
                 (set-booster-parameters booster
                                         '(("objective" . "binary:logistic")
                                           ("max_depth" . "2")
                                           ("eta" . "0.5")
                                           ("verbosity" . "0")))
                 (dotimes (iteration 5)
                   (xgb-check (xgb::xg-booster-update-one-iter booster iteration dmatrix)))
                 (cffi:with-foreign-object (rounds :int)
                   (unless (xgb-check (xgb::xg-booster-boosted-rounds booster rounds))
                     (return-from round-trip))
                   (ok (= 5 (cffi:mem-ref rounds :int))
                       (format nil "boosted-rounds count is ~D" (cffi:mem-ref rounds :int)))))

               (testing "predictions come back with the right shape and ordering"
                 (cffi:with-foreign-objects ((prediction-count :uint64) (out :pointer))
                   (unless (xgb-check (xgb::xg-booster-predict booster dmatrix 0 0 0
                                                               prediction-count out))
                     (return-from round-trip))
                   ;; `buffer' below is XGBoost's own memory, sized to what it actually
                   ;; produced; a short count means reading ROWS floats out of it would
                   ;; be an out-of-bounds foreign read, so the copy must not proceed.
                   (unless (ok (= rows (cffi:mem-ref prediction-count :uint64))
                               (format nil "prediction count is ~D, expected ~D"
                                       (cffi:mem-ref prediction-count :uint64) rows))
                     (return-from round-trip))
                   ;; The buffer belongs to XGBoost and stays valid only until the next
                   ;; prediction on this booster, so copy the values out now.
                   (let ((buffer (cffi:mem-ref out :pointer))
                         (predictions (make-array rows :element-type 'single-float)))
                     (dotimes (index rows)
                       (setf (aref predictions index) (cffi:mem-aref buffer :float index)))
                     (ok (predictions-separate-p predictions label-vector)
                         (format nil "predictions separate by label: ~S" predictions))))))
          (when booster (xgb::xg-booster-free booster))
          (when dmatrix (xgb::xgd-matrix-free dmatrix)))))))
