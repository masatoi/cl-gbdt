(in-package #:cl-user)

(defpackage #:online-retail-example (:use #:cl))
(in-package #:online-retail-example)

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


(defun write-scores (name split predictions)
  (with-open-file (out (merge-pathnames (format nil "output/~A-~A.csv" name split) *root*)
                       :direction :output :if-exists :supersede)
    (loop for i below (array-dimension predictions 0)
          do (format out "~,15F~%" (aref predictions i 0)))))

(defun run (name sizes)
  (let* ((p (getf sizes :features))
         (train-x (load-f32 "train-x.f32" (list (getf sizes :train) p)))
         (train-y (load-f32 "train-y.f32" (list (getf sizes :train))))
         (train-w (load-f32 "train-w.f32" (list (getf sizes :train))))
         (valid-x (load-f32 "valid-x.f32" (list (getf sizes :valid) p)))
         (valid-y (load-f32 "valid-y.f32" (list (getf sizes :valid))))
         (test-x (load-f32 "test-x.f32" (list (getf sizes :test) p)))
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
                   (write-scores backend-name "valid"
                                 (cl-gbdt:predict booster valid-x :num-iteration :best))
                   (write-scores backend-name "test"
                                 (cl-gbdt:predict booster test-x :num-iteration :best)))
              (cl-gbdt:free-booster booster))))))))

(ensure-directories-exist (merge-pathnames "output/" *root*))
(let ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
               (let ((*read-eval* nil)) (read in)))))
  (dolist (backend '(:lightgbm :xgboost)) (run backend sizes)))
