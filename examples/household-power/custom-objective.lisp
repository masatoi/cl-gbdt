(in-package #:cl-user)

(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(load (merge-pathnames "../train-io.lisp" *load-truename*))

(defpackage #:power-custom-objective (:use #:cl))
(in-package #:power-custom-objective)

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
      (list :train-x (f32 "mini-x.f32" (getf sizes :mini) p)
            :train-y (f32 "mini-y.f32" (getf sizes :mini))
            :valid-x (f32 "valid-x.f32" (getf sizes :valid) p)
            :valid-y (f32 "valid-y.f32" (getf sizes :valid))
            :test-x (f32 "test-x.f32" (getf sizes :test) p)))))

(defun asymmetric-squared (scores labels)
  ;; 1/2 * weight * (prediction - target)^2; peak underestimates cost three times as much.
  (let* ((n (length labels))
         (gradient (make-array (list n 1) :element-type 'double-float))
         (hessian (make-array (list n 1) :element-type 'double-float)))
    (dotimes (i n)
      (let* ((target (aref labels i))
             (error (- (aref scores i 0) target))
             (weight (if (and (>= target 2.0) (< error 0)) 3d0 1d0)))
        (setf (aref gradient i 0) (* weight error)
              (aref hessian i 0) weight)))
    (values gradient hessian)))

(defun weighted-mae (scores labels)
  (let ((total 0d0) (weights 0d0))
    (loop for i below (length labels)
          for y = (aref labels i)
          for weight = (if (>= y 2.0) 3d0 1d0)
          do (incf total (* weight (abs (- (aref scores i 0) y))))
             (incf weights weight))
    (/ total weights)))

(defun run (name inputs)
  (let ((train-x (getf inputs :train-x))
         (train-y (getf inputs :train-y))
         (valid-x (getf inputs :valid-x))
         (valid-y (getf inputs :valid-y))
         (test-x (getf inputs :test-x)))
    (format t "~&== ~A custom objective ==~%" name)
    (cl-gbdt:with-backend (be (cl-gbdt:open-backend name))
      (cl-gbdt:with-dataset (tr (cl-gbdt:make-dataset be train-x :label train-y))
        (cl-gbdt:with-dataset (va (if (eq name :lightgbm)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y :reference tr)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y)))
          (multiple-value-bind (booster report)
              (cl-gbdt:train be tr :num-rounds 60 :valid-sets (list (cons "valid" va))
                             :objective (lambda (raw) (asymmetric-squared raw train-y))
                             :evaluation (lambda (scores index)
                                           (values "peak_weighted_mae"
                                                   (and (= index 1)
                                                        (weighted-mae scores valid-y))))
                             :early-stopping (list :metric "peak_weighted_mae" :dataset "valid"
                                                   :direction :lower-is-better :rounds 8)
                             :parameters
                             (if (eq name :lightgbm)
                                 '(:num-threads 4 :num-leaves 31 :learning-rate 0.08
                                   :verbosity -1 :seed 42)
                                 '(:objective "reg:squarederror" :tree-method "hist" :max-depth 6
                                   :eta 0.08 :nthread 4 :verbosity 0 :seed 42)))
            (unwind-protect
                 (progn
                   (format t "rounds=~D best=~D~%" (cl-gbdt:training-report-num-rounds report)
                           (cl-gbdt:training-report-best-iteration report))
                   (cl-gbdt-examples:write-scores
                    (output-file (format nil "~(~A~)-custom-test.csv" name))
                    (cl-gbdt:predict booster test-x :num-iteration :best)))
              (cl-gbdt:free-booster booster))))))))

(let ((*root* (uiop:pathname-directory-pathname *load-truename*)))
  (cl-gbdt-example-lifecycle:with-training (*root* :custom
                                             '("lightgbm-custom-test.csv"
                                               "xgboost-custom-test.csv"))
    (let* ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
                    (let ((*read-eval* nil)) (read in))))
           (inputs (load-inputs sizes)))
      (dolist (backend '(:lightgbm :xgboost)) (run backend inputs)))))
