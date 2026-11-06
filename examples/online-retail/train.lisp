(in-package #:cl-user)

(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(load (merge-pathnames "../train-io.lisp" *load-truename*))

(defpackage #:online-retail-example (:use #:cl))
(in-package #:online-retail-example)

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
            :train-w (f32 "train-w.f32" (getf sizes :train))
            :valid-x (f32 "valid-x.f32" (getf sizes :valid) p)
            :valid-y (f32 "valid-y.f32" (getf sizes :valid))
            :test-x (f32 "test-x.f32" (getf sizes :test) p)))))

(defun run (name sizes inputs)
  (let ((train-x (getf inputs :train-x))
         (train-y (getf inputs :train-y))
         (train-w (getf inputs :train-w))
         (valid-x (getf inputs :valid-x))
         (valid-y (getf inputs :valid-y))
         (test-x (getf inputs :test-x))
         (cats (getf sizes :categorical))
         (backend-name (string-downcase (symbol-name name))))
    (format t "~&== ~A ==~%" name)
    (cl-gbdt:with-backend (be (cl-gbdt:open-backend name))
      (cl-gbdt:with-dataset (tr (cl-gbdt:make-dataset be train-x :label train-y :weight train-w
                                                       :categorical-features cats))
        (cl-gbdt:with-dataset (va (if (eq name :lightgbm)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y
                                                            :categorical-features cats
                                                            :reference tr)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y
                                                            :categorical-features cats)))
          (multiple-value-bind (booster report)
              (cl-gbdt:train be tr :num-rounds 120 :valid-sets (list (cons "valid" va))
                             :early-stopping
                             (list :metric (if (eq name :lightgbm) "binary_logloss" "logloss")
                                   :dataset "valid" :direction :lower-is-better :rounds 12)
                             :parameters
                             (if (eq name :lightgbm)
                                 '(:objective "binary" :metric "binary_logloss" :num-threads 4
                                   :num-leaves 15 :learning-rate 0.05 :verbosity -1 :seed 42)
                                 '(:objective "binary:logistic" :eval-metric "logloss"
                                   :tree-method "hist" :max-depth 4 :eta 0.05 :nthread 4
                                   :verbosity 0 :seed 42)))
            (unwind-protect
                 (progn
                   (format t "rounds=~D best=~D~%" (cl-gbdt:training-report-num-rounds report)
                           (cl-gbdt:training-report-best-iteration report))
                   (cl-gbdt-examples:write-scores
                    (output-file (format nil "~A-valid.csv" backend-name))
                    (cl-gbdt:predict booster valid-x :num-iteration :best))
                   (cl-gbdt-examples:write-scores
                    (output-file (format nil "~A-test.csv" backend-name))
                    (cl-gbdt:predict booster test-x :num-iteration :best)))
              (cl-gbdt:free-booster booster))))))))

(let ((*root* (uiop:pathname-directory-pathname *load-truename*)))
  (cl-gbdt-example-lifecycle:with-training (*root* :standard
                                             '("lightgbm-valid.csv" "lightgbm-test.csv"
                                             "xgboost-valid.csv" "xgboost-test.csv"))
    (let* ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
                    (let ((*read-eval* nil)) (read in))))
           (inputs (load-inputs sizes)))
      (dolist (backend '(:lightgbm :xgboost)) (run backend sizes inputs)))))
