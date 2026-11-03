(in-package #:cl-user)

(defpackage #:kasandr-sparse-example (:use #:cl))
(in-package #:kasandr-sparse-example)

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


(defun load-i32 (filename length)
  (let* ((path (merge-pathnames (format nil "data/~A" filename) *root*))
         (array (make-array length :element-type '(signed-byte 32)))
         (file (c-fopen (namestring path) "rb")))
    (when (cffi:null-pointer-p file) (error "Cannot open ~A" path))
    (unwind-protect
         (sb-sys:with-pinned-objects (array)
           (let ((count (c-fread (cffi:make-pointer (sb-sys:sap-int (sb-sys:vector-sap array)))
                                 4 length file)))
             (unless (= count length) (error "Short read from ~A: ~D integers" path count))))
      (c-fclose file))
    array))

(defun read-csr (split rows columns width implicit-value)
  (let* ((nnz (* rows width))
         (offsets (load-i32 (format nil "~A-indptr.i32" split) (1+ rows)))
         (indices (load-i32 (format nil "~A-indices.i32" split) nnz))
         (values (make-array nnz :element-type 'double-float :initial-element 1d0)))
    (cl-gbdt:make-csr-matrix :indptr offsets :indices indices :values values
                             :num-columns columns :implicit-value implicit-value)))

(defun run (name sizes)
  (let* ((p (getf sizes :features)) (width (getf sizes :nonzeros-per-row))
         ;; These declarations intentionally expose the different backend meanings.
         (implicit (if (eq name :lightgbm) 0d0 :missing))
         (train-x (read-csr "train" (getf sizes :train) p width implicit))
         (train-y (load-f32 "train-y.f32" (list (getf sizes :train))))
         (valid-x (read-csr "valid" (getf sizes :valid) p width implicit))
         (valid-y (load-f32 "valid-y.f32" (list (getf sizes :valid))))
         (test-x (read-csr "test" (getf sizes :test) p width implicit)))
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
                   (write-scores (string-downcase (symbol-name name))
                                 (cl-gbdt:predict booster test-x :num-iteration :best)))
              (cl-gbdt:free-booster booster))))))))

(ensure-directories-exist (merge-pathnames "output/" *root*))
(let ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
               (let ((*read-eval* nil)) (read in)))))
  (dolist (backend '(:lightgbm :xgboost)) (run backend sizes)))
