;;;; cl-gbdt hands-on: classification and regression on open datasets.
;;;; Load cl-gbdt and both /unified systems before loading this file.
(defpackage #:cl-gbdt-open-data (:use #:cl))
(in-package #:cl-gbdt-open-data)

(defparameter *root* (uiop:pathname-directory-pathname *load-truename*))

(defun read-splits (name)
  ;; Input is generated locally by prepare_data.py; no reader evaluation is permitted.
  (with-open-file (in (merge-pathnames (format nil "data/~A.sexp" name) *root*))
    (let ((*read-eval* nil)) (read in))))

(defun rows->arrays (rows)
  (let* ((n (length rows)) (p (1- (length (first rows))))
         (x (make-array (list n p) :element-type 'double-float))
         (y (make-array n :element-type 'single-float)))
    (loop for row in rows for i from 0 do
      (unless (= (length row) (1+ p)) (error "Inconsistent row width"))
      (setf (aref y i) (coerce (first row) 'single-float))
      (loop for value in (rest row) for j from 0 do
        (setf (aref x i j) (coerce value 'double-float))))
    (values x y)))

(defun params (dataset backend)
  (ecase backend
    (:lightgbm
     (append '(:num-threads 1 :verbosity -1 :learning-rate 0.05 :num-leaves 15
               :min-data-in-leaf 5 :feature-fraction 1.0 :seed 42)
             (ecase dataset
               (:iris '(:objective "multiclass" :num-class 3 :metric "multi_logloss"))
               (:breast-cancer '(:objective "binary" :metric "binary_logloss"))
               (:diabetes '(:objective "regression" :metric "l2")))))
    (:xgboost
     (append '(:nthread 1 :verbosity 0 :eta 0.05 :max-depth 3 :seed 42)
             (ecase dataset
               (:iris '(:objective "multi:softprob" :num-class 3 :eval-metric "mlogloss"))
               (:breast-cancer '(:objective "binary:logistic" :eval-metric "logloss"))
               (:diabetes '(:objective "reg:squarederror" :eval-metric "rmse")))))))

(defun metric-name (dataset backend)
  (ecase dataset
    (:iris (ecase backend (:lightgbm "multi_logloss") (:xgboost "mlogloss")))
    (:breast-cancer (ecase backend (:lightgbm "binary_logloss") (:xgboost "logloss")))
    (:diabetes (ecase backend (:lightgbm "l2") (:xgboost "rmse")))))

(defun class-prediction (prediction row classes)
  (if (= classes 2)
      (if (>= (aref prediction row 0) 0.5d0) 1 0)
      (loop with best = 0 for j from 1 below classes
            when (> (aref prediction row j) (aref prediction row best)) do (setf best j)
            finally (return best))))

(defun evaluate-classification (train-y y prediction classes)
  (let ((train-counts (make-array classes :initial-element 0)))
    (loop for value across train-y do (incf (aref train-counts (round value))))
    (let ((majority (loop with best = 0 for j from 1 below classes
                          when (> (aref train-counts j) (aref train-counts best)) do (setf best j)
                          finally (return best))))
      (format t "  majority baseline accuracy=~,4F (train majority class=~D)~%"
              (/ (count majority y :key #'round) (length y) 1d0) majority)))
  (let ((confusion (make-array (list classes classes) :initial-element 0))
        (correct 0))
    (loop for i below (length y) for actual = (round (aref y i))
          for predicted = (class-prediction prediction i classes) do
            (incf (aref confusion actual predicted))
            (when (= actual predicted) (incf correct)))
    (format t "  accuracy=~,4F (~D/~D)~%" (/ correct (length y) 1.0d0) correct (length y))
    (format t "  confusion matrix (rows=actual, columns=predicted): ~S~%" confusion)
    (when (= classes 2)
      (let ((tp (aref confusion 1 1)) (fn (aref confusion 1 0)) (fp (aref confusion 0 1)))
        (format t "  malignant recall=~,4F precision=~,4F (positive=malignant)~%"
                (if (zerop (+ tp fn)) 0d0 (/ tp (+ tp fn) 1d0))
                (if (zerop (+ tp fp)) 0d0 (/ tp (+ tp fp) 1d0)))))))

(defun mean-of (v) (/ (reduce #'+ v) (length v) 1.0d0))

(defun evaluate-regression (train-y test-y prediction)
  (let* ((n (length test-y)) (avg (mean-of test-y)) (train-avg (mean-of train-y))
         (se 0d0) (ae 0d0) (total 0d0) (base-se 0d0))
    (loop for i below n for actual = (aref test-y i)
          for predicted = (aref prediction i 0) do
            (incf se (expt (- predicted actual) 2))
            (incf ae (abs (- predicted actual)))
            (incf total (expt (- actual avg) 2))
            (incf base-se (expt (- actual train-avg) 2)))
    (format t "  RMSE=~,3F MAE=~,3F R2=~,4F~%" (sqrt (/ se n)) (/ ae n) (- 1 (/ se total)))
    (format t "  baseline RMSE=~,3F (predict training-set mean=~,3F)~%"
            (sqrt (/ base-se n)) train-avg)))

(defun loss (dataset labels prediction)
  (let ((sum 0d0) (n (length labels)))
    (loop for i below n for actual = (aref labels i) do
      (incf sum
            (ecase dataset
              (:diabetes (expt (- (aref prediction i 0) actual) 2))
              (:iris (- (log (max 1d-15 (min 1d0 (aref prediction i (round actual)))))))
              (:breast-cancer
               (let* ((p (max 1d-15 (min (- 1d0 1d-15) (aref prediction i 0)))))
                 (- (+ (* actual (log p)) (* (- 1 actual) (log (- 1 p))))))))))
    (/ sum n)))

(defun permutation-indices (n)
  ;; Fisher-Yates using a fixed local LCG, so this diagnostic is repeatable.
  (let ((indices (make-array n)) (state 12345))
    (dotimes (i n) (setf (aref indices i) i))
    (loop for i downfrom (1- n) above 0 do
      (setf state (mod (+ (* 1664525 state) 1013904223) (expt 2 32)))
      (rotatef (aref indices i) (aref indices (mod state (1+ i)))))
    indices))

(defun show-permutation-importance (dataset booster x y names prediction)
  ;; An exploratory test-set diagnostic, never used to choose model parameters.
  (let* ((n (array-dimension x 0)) (p (array-dimension x 1))
         (permutation (permutation-indices n)) (baseline (loss dataset y prediction))
         (scores nil))
    (dotimes (j p)
      (let ((permuted (make-array (array-dimensions x) :element-type 'double-float)))
        (dotimes (k (array-total-size x))
          (setf (row-major-aref permuted k) (row-major-aref x k)))
        (dotimes (i n) (setf (aref permuted i j) (aref x (aref permutation i) j)))
        (let ((changed (cl-gbdt:predict booster permuted :num-iteration :best)))
          (push (cons (nth j names) (- (loss dataset y changed) baseline)) scores))))
    (setf scores (sort scores #'> :key #'cdr))
    (format t "  permutation importance: increase in ~A, one fixed shuffle (top 5):~%"
            (if (eq dataset :diabetes) "MSE" "log loss"))
    (loop for (name . score) in scores for i below 5 do
      (format t "    ~A: ~,5F~%" name score))))

(defun run-example (dataset backend splits)
  (multiple-value-bind (train-x train-y) (rows->arrays (getf splits :train))
    (multiple-value-bind (valid-x valid-y) (rows->arrays (getf splits :valid))
      (multiple-value-bind (test-x test-y) (rows->arrays (getf splits :test))
        (format t "~%== ~A / ~A ==~%" dataset backend)
        (format t "  rows train/valid/test = ~D/~D/~D, features=~D~%"
                (length train-y) (length valid-y) (length test-y) (array-dimension train-x 1))
        (cl-gbdt:with-backend (be (cl-gbdt:open-backend backend))
          (cl-gbdt:with-dataset (train-set (cl-gbdt:make-dataset be train-x :label train-y))
            ;; LightGBM requires validation datasets to share the training bin mapper.
            (cl-gbdt:with-dataset (valid-set
                                    (if (eq backend :lightgbm)
                                        (cl-gbdt:make-dataset be valid-x :label valid-y
                                                              :reference train-set)
                                        (cl-gbdt:make-dataset be valid-x :label valid-y)))
              (multiple-value-bind (booster report)
                  (cl-gbdt:train be train-set :num-rounds 250
                    :valid-sets (list (cons "validation" valid-set))
                    :early-stopping (list :metric (metric-name dataset backend)
                                          :dataset "validation"
                                          :direction :lower-is-better :rounds 20)
                    :parameters (params dataset backend))
                (unwind-protect
                     (let ((predictions (cl-gbdt:predict booster test-x :num-iteration :best)))
                       (format t "  iterations=~D best-iteration=~D stopped-early=~S~%"
                               (cl-gbdt:training-report-num-rounds report)
                               (cl-gbdt:training-report-best-iteration report)
                               (cl-gbdt:training-report-early-stopped-p report))
                       (ecase dataset
                         (:iris (evaluate-classification train-y test-y predictions 3))
                         (:breast-cancer (evaluate-classification train-y test-y predictions 2))
                         (:diabetes (evaluate-regression train-y test-y predictions)))
                       (show-permutation-importance dataset booster test-x test-y
                                                    (getf splits :feature-names) predictions))
                  (cl-gbdt:free-booster booster))))))))))

(dolist (entry '((:iris . "iris") (:breast-cancer . "breast_cancer") (:diabetes . "diabetes")))
  (let ((splits (read-splits (cdr entry))))
    (dolist (backend '(:lightgbm :xgboost))
      (run-example (car entry) backend splits))))
