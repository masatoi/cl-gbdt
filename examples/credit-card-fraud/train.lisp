(in-package #:cl-user)

(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(load (merge-pathnames "../train-io.lisp" *load-truename*))

;;;; SBCL: ../train-io.lisp reads prepare.lisp's little-endian float32 files (written by
;;;; common.lisp's WRITE-F32) directly into simple arrays.
;;;; Load CFFI, cl-gbdt and both unified backends before loading this file.
(defpackage #:fraud-analysis (:use #:cl))
(in-package #:fraud-analysis)

(defvar *root* (uiop:pathname-directory-pathname *load-truename*))

(defun seconds () (/ (get-internal-real-time) internal-time-units-per-second 1d0))

(defun write-best-scores (booster x name)
  (cl-gbdt-examples:write-scores
   (merge-pathnames (format nil "~A.csv" name)
                    (or cl-gbdt-example-lifecycle:*training-output-directory*
                        (merge-pathnames "output/" *root*)))
   (cl-gbdt:predict booster x :num-iteration :best)
   :rows (array-dimension x 0)))

(defun load-inputs (sizes)
  "Read every input matrix once; both backends train on the same arrays."
  (flet ((f32 (name &rest dimensions)
           (cl-gbdt-examples:load-f32 (merge-pathnames name (merge-pathnames "data/" *root*))
                                      dimensions)))
    (let ((cols (getf sizes :features)))
      (list :train-x (f32 "train-x.f32" (getf sizes :train) cols)
            :train-y (f32 "train-y.f32" (getf sizes :train))
            :valid-x (f32 "valid-x.f32" (getf sizes :valid) cols)
            :valid-y (f32 "valid-y.f32" (getf sizes :valid))
            :test-x (f32 "test-x.f32" (getf sizes :test) cols)))))

(defun run (backend inputs)
  (let ((train-x (getf inputs :train-x))
         (train-y (getf inputs :train-y))
         (valid-x (getf inputs :valid-x))
         (valid-y (getf inputs :valid-y))
         (test-x (getf inputs :test-x))
         (start (seconds)))
    (format t "~&== ~A ==~%" backend)
    (cl-gbdt:with-backend (be (cl-gbdt:open-backend backend))
      (cl-gbdt:with-dataset (tr (cl-gbdt:make-dataset be train-x :label train-y))
        (cl-gbdt:with-dataset (va (if (eq backend :lightgbm)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y :reference tr)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y)))
          (multiple-value-bind (booster report)
              (cl-gbdt:train be tr :num-rounds 250 :valid-sets (list (cons "valid" va))
                             :early-stopping (list :metric (if (eq backend :lightgbm)
                                                               "binary_logloss"
                                                               "logloss")
                                                   :dataset "valid"
                                                   :direction :lower-is-better
                                                   :rounds 20)
                             :parameters (if (eq backend :lightgbm)
                                             '(:objective "binary" :metric "binary_logloss"
                                               :num-threads 4 :verbosity -1 :num-leaves 31
                                               :learning-rate 0.05 :seed 42)
                                             '(:objective "binary:logistic" :eval-metric "logloss"
                                               :tree-method "hist" :nthread 4 :max-depth 6
                                               :eta 0.05 :verbosity 0 :seed 42)))
            (unwind-protect
                 (let ((training-end (seconds)))
                   (format t "rounds=~D best=~D early-stopped=~S setup-and-train-seconds=~,3F~%"
                           (cl-gbdt:training-report-num-rounds report)
                           (cl-gbdt:training-report-best-iteration report)
                           (cl-gbdt:training-report-early-stopped-p report)
                           (- training-end start))
                   (write-best-scores booster valid-x (format nil "~(~A~)-valid" backend))
                   (write-best-scores booster test-x (format nil "~(~A~)-test" backend))
                   (format t "total-seconds=~,3F~%" (- (seconds) start)))
              (cl-gbdt:free-booster booster))))))))

(let ((*root* (uiop:pathname-directory-pathname *load-truename*)))
  (cl-gbdt-example-lifecycle:with-training (*root* :standard
                                             '("lightgbm-valid.csv" "lightgbm-test.csv"
                                             "xgboost-valid.csv" "xgboost-test.csv"))
    (let* ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
                    (let ((*read-eval* nil)) (read in))))
           (inputs (load-inputs sizes)))
      (dolist (backend '(:lightgbm :xgboost)) (run backend inputs)))))
