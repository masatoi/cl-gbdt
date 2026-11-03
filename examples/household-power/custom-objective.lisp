(in-package #:cl-user)

(defpackage #:power-custom-objective (:use #:cl))
(in-package #:power-custom-objective)

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

(defun run (name sizes)
  (let* ((p (getf sizes :features))
         (train-x (load-f32 "mini-x.f32" (list (getf sizes :mini) p)))
         (train-y (load-f32 "mini-y.f32" (list (getf sizes :mini))))
         (valid-x (load-f32 "valid-x.f32" (list (getf sizes :valid) p)))
         (valid-y (load-f32 "valid-y.f32" (list (getf sizes :valid))))
         (test-x (load-f32 "test-x.f32" (list (getf sizes :test) p))))
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
                   (write-scores (format nil "~(~A~)-custom" name)
                                 (cl-gbdt:predict booster test-x :num-iteration :best)))
              (cl-gbdt:free-booster booster))))))))

(ensure-directories-exist (merge-pathnames "output/" *root*))
(let ((sizes (with-open-file (in (merge-pathnames "data/sizes.lisp" *root*))
               (let ((*read-eval* nil)) (read in)))))
  (dolist (backend '(:lightgbm :xgboost)) (run backend sizes)))
