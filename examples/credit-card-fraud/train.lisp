;;;; SBCL x86-64: read NumPy's little-endian float32 buffers directly into simple arrays.
;;;; Load CFFI, cl-gbdt and both unified backends before loading this file.
(defpackage #:fraud-analysis (:use #:cl))
(in-package #:fraud-analysis)
(unless (member :little-endian *features*)
  (error "This example expects little-endian float32 binary data."))
(defparameter *root* (uiop:pathname-directory-pathname *load-truename*))
(cffi:defcfun ("fopen" c-fopen) :pointer (name :string) (mode :string))
(cffi:defcfun ("fread" c-fread) :size
  (buffer :pointer) (size :size) (nmemb :size) (stream :pointer))
(cffi:defcfun ("fclose" c-fclose) :int (stream :pointer))

(defun load-f32 (filename dimensions)
  (let* ((pathname (merge-pathnames (format nil "data/~A" filename) *root*))
         (array (make-array dimensions :element-type 'single-float))
         (storage (sb-ext:array-storage-vector array))
         (file (c-fopen (namestring pathname) "rb")))
    (when (cffi:null-pointer-p file) (error "Cannot open ~A" pathname))
    (unwind-protect
         (sb-sys:with-pinned-objects (array storage)
           (let ((count (c-fread (cffi:make-pointer (sb-sys:sap-int (sb-sys:vector-sap storage)))
                                 4 (array-total-size array) file)))
             (unless (= count (array-total-size array))
               (error "Short read from ~A: ~D/~D floats" pathname count (array-total-size array)))))
      (c-fclose file))
    array))

(defun seconds () (/ (get-internal-real-time) internal-time-units-per-second 1d0))
(defun write-scores (booster x name)
  (let ((p (cl-gbdt:predict booster x :num-iteration :best)))
    (unless (equal (array-dimensions p) (list (array-dimension x 0) 1))
      (error "Unexpected prediction dimensions ~S" (array-dimensions p)))
    (with-open-file (out (merge-pathnames (format nil "output/~A.csv" name) *root*)
                         :direction :output :if-exists :supersede)
      (loop for i below (array-dimension x 0) do (format out "~,15F~%" (aref p i 0))))))

(defun run (backend sizes)
  (let* ((n (getf sizes :train)) (v (getf sizes :valid)) (nt (getf sizes :test))
         (cols (getf sizes :features))
         (train-x (load-f32 "train-x.f32" (list n cols)))
         (train-y (load-f32 "train-y.f32" (list n)))
         (valid-x (load-f32 "valid-x.f32" (list v cols)))
         (valid-y (load-f32 "valid-y.f32" (list v)))
         (test-x (load-f32 "test-x.f32" (list nt cols)))
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
                   (write-scores booster valid-x (format nil "~(~A~)-valid" backend))
                   (write-scores booster test-x (format nil "~(~A~)-test" backend))
                   (format t "total-seconds=~,3F~%" (- (seconds) start)))
              (cl-gbdt:free-booster booster))))))))

(ensure-directories-exist (merge-pathnames "output/" *root*))
(let ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
               (let ((*read-eval* nil)) (read in)))))
  (dolist (backend '(:lightgbm :xgboost)) (run backend sizes)))
