(in-package #:cl-user)

(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(load (merge-pathnames "../train-io.lisp" *load-truename*))

(defpackage #:kasandr-sparse-example (:use #:cl))
(in-package #:kasandr-sparse-example)

;; prepare.lisp writes little-endian float32 labels (common.lisp's WRITE-F32) and int32 CSR
;; offsets and indices (WRITE-I32); ../train-io.lisp reads them. SBCL and CFFI required.

(defvar *root* (uiop:pathname-directory-pathname *load-truename*))

(defun output-file (name)
  (merge-pathnames name (or cl-gbdt-example-lifecycle:*training-output-directory*
                            (merge-pathnames "output/" *root*))))

(defun load-csr-parts (split rows width)
  "Read SPLIT's CSR row offsets and column indices, WIDTH stored entries in each row."
  (flet ((i32 (name length)
           (cl-gbdt-examples:load-i32
            (merge-pathnames (format nil "data/~A-~A.i32" split name) *root*) length)))
    (list (i32 "indptr" (1+ rows)) (i32 "indices" (* rows width)))))

(defun make-csr (parts columns implicit-value)
  "Build a CSR matrix from LOAD-CSR-PARTS' arrays; every stored value is 1."
  (destructuring-bind (offsets indices) parts
    (cl-gbdt:make-csr-matrix :indptr offsets :indices indices
                             :values (make-array (length indices) :element-type 'double-float
                                                                  :initial-element 1d0)
                             :num-columns columns :implicit-value implicit-value)))

(defun load-inputs (sizes)
  "Read every input file once; each backend builds its CSR matrices from the same arrays."
  (let ((width (getf sizes :nonzeros-per-row)))
    (flet ((f32 (name length)
             (cl-gbdt-examples:load-f32 (merge-pathnames name (merge-pathnames "data/" *root*))
                                        (list length))))
      (list :train-x (load-csr-parts "train" (getf sizes :train) width)
            :train-y (f32 "train-y.f32" (getf sizes :train))
            :valid-x (load-csr-parts "valid" (getf sizes :valid) width)
            :valid-y (f32 "valid-y.f32" (getf sizes :valid))
            :test-x (load-csr-parts "test" (getf sizes :test) width)))))

(defun run (name sizes inputs)
  (let* ((p (getf sizes :features))
         ;; These declarations intentionally expose the different backend meanings.
         (implicit (if (eq name :lightgbm) 0d0 :missing))
         (train-x (make-csr (getf inputs :train-x) p implicit))
         (train-y (getf inputs :train-y))
         (valid-x (make-csr (getf inputs :valid-x) p implicit))
         (valid-y (getf inputs :valid-y))
         (test-x (make-csr (getf inputs :test-x) p implicit)))
    (format t "~&== ~A CSR absent=~S ==~%" name implicit)
    (cl-gbdt:with-backend (be (cl-gbdt:open-backend name))
      (when (eq name :xgboost)
        ;; A matrix whose absent entries mean zero is refused by this backend.
        (let ((zero-csr (cl-gbdt:make-csr-matrix :indptr #(0 1) :indices #(0)
                                                 :values #(1d0) :num-columns 2
                                                 :implicit-value 0d0)))
          (handler-case
              (let ((unexpected (cl-gbdt:make-dataset be zero-csr :label #(0))))
                (cl-gbdt:free-dataset unexpected)
                (error "XGBoost unexpectedly accepted zero-as-absent."))
            (cl-gbdt:unsupported-argument ()
              (format t "verified: XGBoost refuses :implicit-value 0~%")))))
      (cl-gbdt:with-dataset (tr (cl-gbdt:make-dataset be train-x :label train-y))
        (cl-gbdt:with-dataset (va (if (eq name :lightgbm)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y :reference tr)
                                      (cl-gbdt:make-dataset be valid-x :label valid-y)))
          (multiple-value-bind (booster report)
              (cl-gbdt:train be tr :num-rounds 60 :valid-sets (list (cons "valid" va))
                             :early-stopping
                             (list :metric (if (eq name :lightgbm) "binary_logloss" "logloss")
                                   :dataset "valid" :direction :lower-is-better :rounds 8)
                             :parameters
                             (if (eq name :lightgbm)
                                 '(:objective "binary" :metric "binary_logloss" :num-threads 4
                                   :num-leaves 15 :learning-rate 0.1 :verbosity -1 :seed 42)
                                 '(:objective "binary:logistic" :eval-metric "logloss"
                                   :tree-method "hist" :max-depth 4 :eta 0.1 :nthread 4
                                   :verbosity 0 :seed 42)))
            (unwind-protect
                 (progn
                   (format t "rounds=~D best=~D~%" (cl-gbdt:training-report-num-rounds report)
                           (cl-gbdt:training-report-best-iteration report))
                   (cl-gbdt-examples:write-scores
                    (output-file (format nil "~(~A~)-test.csv" name))
                    (cl-gbdt:predict booster test-x :num-iteration :best)))
              (cl-gbdt:free-booster booster))))))))

(let ((*root* (uiop:pathname-directory-pathname *load-truename*)))
  (cl-gbdt-example-lifecycle:with-training (*root* :standard
                                             '("lightgbm-test.csv" "xgboost-test.csv"))
    (let* ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
                    (let ((*read-eval* nil)) (read in))))
           (inputs (load-inputs sizes)))
      (dolist (backend '(:lightgbm :xgboost)) (run backend sizes inputs)))))
