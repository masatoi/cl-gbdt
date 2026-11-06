(in-package #:cl-user)

;;;; Shared MovieLens ranking trainer; callers provide input and output directories.

(load (merge-pathnames "../train-io.lisp" *load-truename*))

(defpackage #:movielens-ranking-example (:use #:cl))
(in-package #:movielens-ranking-example)

;; Dense little-endian float32 matrices written by prepare.lisp (common.lisp's WRITE-F32)
;; and read by ../train-io.lisp; SBCL and CFFI required.
(unless (member :little-endian *features*)
  (error "This example needs little-endian float32 data."))

(defvar *root* nil
  "Dynamically bound to the example directory for one invocation.")

(defvar *output-directory* nil
  "Dynamically bound output directory for the current trainer invocation.")

(defun load-f32 (filename dimensions)
  "Read FILENAME from the current example's data directory with the shared loader."
  (cl-gbdt-examples:load-f32 (merge-pathnames (format nil "data/~A" filename) *root*)
                             dimensions))

(defun write-scores (name predictions)
  "Write PREDICTIONS as NAME-test.csv in the current output directory."
  (cl-gbdt-examples:write-scores
   (merge-pathnames (format nil "~A-test.csv" name) *output-directory*) predictions))

(defun read-groups (name)
  (with-open-file (in (merge-pathnames (format nil "data/~A-groups.lisp" name) *root*))
    (let ((*read-eval* nil)) (read in))))

(defun run (name sizes)
  (let* ((p (getf sizes :features))
         (train-x (load-f32 "train-x.f32" (list (getf sizes :train) p)))
         (train-y (load-f32 "train-y.f32" (list (getf sizes :train))))
         (valid-x (load-f32 "valid-x.f32" (list (getf sizes :valid) p)))
         (valid-y (load-f32 "valid-y.f32" (list (getf sizes :valid))))
         (test-x (load-f32 "test-x.f32" (list (getf sizes :test) p)))
         (train-groups (read-groups "train"))
         (valid-groups (read-groups "valid")))
    (assert (= (reduce #'+ train-groups) (getf sizes :train)))
    (assert (= (reduce #'+ valid-groups) (getf sizes :valid)))
    (format t "~&== ~A ==~%" name)
    (cl-gbdt:with-backend (be (cl-gbdt:open-backend name))
      (cl-gbdt:with-dataset (tr (cl-gbdt:make-dataset be train-x :label train-y
                                                       :group train-groups))
        (cl-gbdt:with-dataset (va (if (eq name :lightgbm)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y
                                                            :group valid-groups :reference tr)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y
                                                            :group valid-groups)))
          (multiple-value-bind (booster report)
              (cl-gbdt:train be tr :num-rounds 80 :valid-sets (list (cons "valid" va))
                             :parameters
                             (if (eq name :lightgbm)
                                 '(:objective "lambdarank" :metric "ndcg" :eval-at "10"
                                   :num-threads 4 :num-leaves 15 :learning-rate 0.05
                                   :verbosity -1 :seed 42)
                                 '(:objective "rank:ndcg" :eval-metric "ndcg@10"
                                   :tree-method "hist" :max-depth 4 :eta 0.05 :nthread 4
                                   :verbosity 0 :seed 42)))
            (unwind-protect
                 (progn
                   (format t "trained ~D ranking rounds~%"
                           (cl-gbdt:training-report-num-rounds report))
                   (write-scores (string-downcase (symbol-name name))
                                 (cl-gbdt:predict booster test-x)))
              (cl-gbdt:free-booster booster))))))))


(defun train-ranking (root output-directory)
  (let ((*root* root) (*output-directory* output-directory))
    (ensure-directories-exist *output-directory*)
    (let ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
                   (let ((*read-eval* nil)) (read in)))))
      (dolist (backend '(:lightgbm :xgboost)) (run backend sizes)))))
