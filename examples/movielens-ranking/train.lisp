(in-package #:cl-user)

(defpackage #:movielens-ranking-example (:use #:cl))
(in-package #:movielens-ranking-example)

;; Dense little-endian float32 matrices prepared by prepare.py; SBCL and CFFI required.
(unless (member :little-endian *features*)
  (error "This example needs little-endian float32 data."))

(defparameter *root*
  (let ((override (uiop:getenv "CL_GBDT_RANKING_ROOT")))
    (if override
        (uiop:ensure-directory-pathname override)
        (uiop:pathname-directory-pathname *load-truename*))))

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

(ensure-directories-exist (merge-pathnames "output/" *root*))
(let ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
               (let ((*read-eval* nil)) (read in)))))
  (dolist (backend '(:lightgbm :xgboost)) (run backend sizes)))
