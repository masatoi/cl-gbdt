(in-package #:cl-user)

(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(load (merge-pathnames "../train-io.lisp" *load-truename*))

(defpackage #:adult-example (:use #:cl))
(in-package #:adult-example)

;; Dense little-endian float32 matrices written by prepare.lisp (common.lisp's WRITE-F32)
;; and read by ../train-io.lisp; SBCL and CFFI required.

(defvar *root* (uiop:pathname-directory-pathname *load-truename*))

(defun output-file (name)
  (merge-pathnames name (or cl-gbdt-example-lifecycle:*training-output-directory*
                            (merge-pathnames "output/" *root*))))

(defun load-inputs (sizes)
  "Read every input matrix once; both backends train on the same arrays."
  (flet ((f32 (name &rest dimensions)
           (cl-gbdt-examples:load-f32 (merge-pathnames name (merge-pathnames "data/" *root*))
                                      dimensions)))
    (let ((p (getf sizes :features)))
      (list :train-x (f32 "train-x.f32" (getf sizes :train) p)
            :train-y (f32 "train-y.f32" (getf sizes :train))
            :valid-x (f32 "valid-x.f32" (getf sizes :valid) p)
            :valid-y (f32 "valid-y.f32" (getf sizes :valid))
            :test-x (f32 "test-x.f32" (getf sizes :test) p)))))

(defun report-contributions (booster matrix)
  ;; SHAP values sum to the untransformed margin, including the bias term.
  (let ((one (make-array (list 1 (array-dimension matrix 1)) :element-type 'single-float)))
    (loop for j below (array-dimension matrix 1) do (setf (aref one 0 j) (aref matrix 0 j)))
    (multiple-value-bind (values shape) (cl-gbdt:predict booster one :kind :contrib)
      (let ((sum (loop for j below (array-total-size values) sum (row-major-aref values j)))
            (raw (aref (cl-gbdt:predict booster one :kind :raw) 0 0)))
        (format t "contrib shape=~S sum=~,6F raw=~,6F~%" shape sum raw)
        (loop for label in '("age" "workclass" "fnlwgt" "education" "education_num"
                             "marital_status" "occupation" "relationship" "race" "sex"
                             "capital_gain" "capital_loss" "hours_per_week" "native_country" "bias")
              for j from 0
              do (format t "  ~A: ~,6F~%" label (row-major-aref values j)))
        (unless (< (abs (- sum raw)) 1d-4)
          (error "Contributions do not sum to raw score."))))))

(defun run (name sizes inputs)
  (let ((train-x (getf inputs :train-x))
         (train-y (getf inputs :train-y))
         (valid-x (getf inputs :valid-x))
         (valid-y (getf inputs :valid-y))
         (test-x (getf inputs :test-x))
         (cats (getf sizes :categorical))
         (model (output-file (if (eq name :lightgbm) "model.txt" "model.json"))))
    (format t "~&== ~A ==~%" name)
    (cl-gbdt:with-backend (be (cl-gbdt:open-backend name))
      (cl-gbdt:with-dataset (tr (cl-gbdt:make-dataset be train-x :label train-y
                                                       :categorical-features cats))
        (cl-gbdt:with-dataset (va (if (eq name :lightgbm)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y
                                                            :categorical-features cats
                                                            :reference tr)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y
                                                            :categorical-features cats)))
          (multiple-value-bind (booster report)
              (cl-gbdt:train be tr :num-rounds 150 :valid-sets (list (cons "valid" va))
                             :early-stopping
                             (list :metric (if (eq name :lightgbm) "binary_logloss" "logloss")
                                   :dataset "valid" :direction :lower-is-better :rounds 15)
                             :parameters
                             (if (eq name :lightgbm)
                                 '(:objective "binary" :metric "binary_logloss" :num-threads 4
                                   :num-leaves 31 :learning-rate 0.05 :verbosity -1 :seed 42)
                                 '(:objective "binary:logistic" :eval-metric "logloss"
                                   :tree-method "hist" :max-depth 6 :eta 0.05 :nthread 4
                                   :verbosity 0 :seed 42)))
            (unwind-protect
                 (progn
                   (format t "rounds=~D best=~D~%" (cl-gbdt:training-report-num-rounds report)
                           (cl-gbdt:training-report-best-iteration report))
                   (cl-gbdt-examples:write-scores
                    (output-file (format nil "~(~A~)-test.csv" name))
                    (cl-gbdt:predict booster test-x :num-iteration :best))
                   (format t "feature-importance=~S~%"
                           (coerce (cl-gbdt:feature-importance booster) 'list))
                   (report-contributions booster test-x)
                   ;; Save every trained round, then compare the same full-model prediction.
                   (cl-gbdt:save-model booster model)
                   (let ((loaded (cl-gbdt:load-model be model)))
                     (unwind-protect
                          (let* ((a (cl-gbdt:predict booster test-x))
                                 (b (cl-gbdt:predict loaded test-x))
                                 (diff (loop for i below (array-total-size a)
                                             maximize (abs (- (row-major-aref a i)
                                                              (row-major-aref b i))))))
                            (format t "saved-model max-prediction-difference=~,9F~%" diff)
                            (unless (< diff 1d-6) (error "Model round trip changed predictions.")))
                       (cl-gbdt:free-booster loaded))))
              (cl-gbdt:free-booster booster))))))))

(let ((*root* (uiop:pathname-directory-pathname *load-truename*)))
  (cl-gbdt-example-lifecycle:with-training (*root* :standard
                                             '("lightgbm-test.csv" "xgboost-test.csv"
                                               "model.txt" "model.json"))
    (let* ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
                    (let ((*read-eval* nil)) (read in))))
           (inputs (load-inputs sizes)))
      (dolist (backend '(:lightgbm :xgboost)) (run backend sizes inputs)))))
