(in-package #:cl-user)

(defpackage #:adult-example (:use #:cl))
(in-package #:adult-example)

;; Dense little-endian float32 matrices prepared by prepare.py; SBCL and CFFI required.
(unless (member :little-endian *features*)
  (error "This example needs little-endian float32 data."))
(defparameter *root* (uiop:pathname-directory-pathname *load-truename*))
(cffi:defcfun ("fopen" c-fopen) :pointer (name :string) (mode :string))
(cffi:defcfun ("fread" c-fread) :size
  (buffer :pointer) (size :size) (count :size) (stream :pointer))
(cffi:defcfun ("fclose" c-fclose) :int (stream :pointer))

(defun load-f32 (filename dimensions)
  (let* ((path (merge-pathnames (format nil "data/~A" filename) *root*))
         (array (make-array dimensions :element-type 'single-float))
         (storage (sb-ext:array-storage-vector array))
         (file (c-fopen (namestring path) "rb")))
    (when (cffi:null-pointer-p file) (error "Cannot open ~A" path))
    (unwind-protect
         (sb-sys:with-pinned-objects (array storage)
           (let ((count (c-fread (cffi:make-pointer (sb-sys:sap-int (sb-sys:vector-sap storage)))
                                 4 (array-total-size array) file)))
             (unless (= count (array-total-size array))
               (error "Short read from ~A: ~D floats" path count))))
      (c-fclose file))
    array))

(defun write-scores (name predictions)
  (with-open-file (out (merge-pathnames (format nil "output/~A-test.csv" name) *root*)
                       :direction :output :if-exists :supersede)
    (loop for i below (array-dimension predictions 0)
          do (format out "~,15F~%" (aref predictions i 0)))))

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

(defun run (name sizes)
  (let* ((p (getf sizes :features))
         (train-x (load-f32 "train-x.f32" (list (getf sizes :train) p)))
         (train-y (load-f32 "train-y.f32" (list (getf sizes :train))))
         (valid-x (load-f32 "valid-x.f32" (list (getf sizes :valid) p)))
         (valid-y (load-f32 "valid-y.f32" (list (getf sizes :valid))))
         (test-x (load-f32 "test-x.f32" (list (getf sizes :test) p)))
         (cats (getf sizes :categorical))
         (model (merge-pathnames (if (eq name :lightgbm)
                                     "output/model.txt" "output/model.json")
                                 *root*)))
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
                   (write-scores (string-downcase (symbol-name name))
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

(ensure-directories-exist (merge-pathnames "output/" *root*))
(let ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
               (let ((*read-eval* nil)) (read in)))))
  (dolist (backend '(:lightgbm :xgboost)) (run backend sizes)))
