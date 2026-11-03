(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

(defparameter *toy-root* (uiop:pathname-directory-pathname *load-truename*))
(defparameter *iris-names*
  '("sepal length (cm)" "sepal width (cm)" "petal length (cm)" "petal width (cm)"))
(defparameter *cancer-names*
  (loop for prefix in '("mean" "error" "worst")
        append (loop for name in '("radius" "texture" "perimeter" "area" "smoothness"
                                   "compactness" "concavity" "concave points" "symmetry"
                                   "fractal dimension")
                     collect (format nil "~A ~A" prefix name))))
(defparameter *diabetes-names* '("age" "sex" "bmi" "bp" "s1" "s2" "s3" "s4" "s5" "s6"))

(defun read-toy-classification (path nfeatures reverse-labels)
  (with-open-file (in path)
    (let* ((header (read-csv-row in))
           (expected (parse-integer (first header)))
           (rows (make-array 0 :adjustable t :fill-pointer 0)))
      (unless (= (parse-integer (second header)) nfeatures) (error "Unexpected feature count"))
      (loop for fields = (read-csv-row in) while fields
            do (unless (= (length fields) (1+ nfeatures)) (error "Unexpected row width"))
               (let* ((label (parse-integer (car (last fields))))
                      (features (mapcar (lambda (s) (coerce (parse-number s) 'double-float))
                                        (butlast fields))))
                 (vector-push-extend
                  (cons (if reverse-labels (- 1 label) label) features) rows)))
      (unless (= (length rows) expected) (error "Unexpected dataset length"))
      rows)))

(defun read-toy-diabetes (data)
  (with-open-file (xs (merge-pathnames "diabetes_data_raw.csv" data))
    (with-open-file (ys (merge-pathnames "diabetes_target.csv" data))
      (let ((rows (make-array 0 :adjustable t :fill-pointer 0)))
        (loop for x = (read-line xs nil nil)
              for y = (read-line ys nil nil)
              while (or x y)
              do (unless (and x y) (error "Diabetes feature/label length mismatch"))
                 (let ((fields (remove "" (split-fields x #\Space) :test #'string=)))
                   (unless (= (length fields) 10) (error "Diabetes feature count mismatch"))
                   (vector-push-extend
                    (cons (parse-number y)
                          (mapcar (lambda (s) (coerce (parse-number s) 'double-float)) fields))
                    rows)))
        rows))))

(defun toy-partitions (rows classification)
  (if (not classification)
      (multiple-value-bind (tr va te) (split-indexes (length rows) 42)
        (mapcar (lambda (indices) (map 'vector (lambda (i) (aref rows i)) indices))
                (list tr va te)))
      (let ((parts (list nil nil nil))
            (classes (sort (remove-duplicates (map 'list #'first rows)) #'<)))
        (dolist (class classes)
          (let* ((indices (coerce (loop for row across rows for i from 0
                                        when (= (first row) class) collect i)
                                  'vector))
                 (order (shuffle-vector indices (+ 42 class)))
                 (train (floor (* 0.6d0 (length order))))
                 (valid (floor (* 0.2d0 (length order)))))
            (loop for index across order for i from 0
                  for split = (cond ((< i train) 0) ((< i (+ train valid)) 1) (t 2))
                  do (push (aref rows index) (nth split parts)))))
        (loop for part in parts for seed from 100
              collect (shuffle-vector (coerce part 'vector) seed)))))

(defun write-toy (name rows names classification)
  (let* ((data (merge-pathnames "data/" *toy-root*))
         (parts (toy-partitions rows classification))
         (path (merge-pathnames (format nil "~A.sexp" name) data)))
    (with-open-file (out path :direction :output :if-exists :supersede)
      (let ((*print-readably* t) (*print-case* :downcase))
        (write-char #\( out)
        (loop for key in '(:train :valid :test) for part in parts
              do (format out " ~S (~%" key)
                 (loop for row across part do (write row :stream out) (terpri out))
                 (write-char #\) out))
        (format out " :feature-names ")
        (write names :stream out)
        (format out ")~%")))
    (format t "~A: train=~D valid=~D test=~D features=~D~%"
            name (length (first parts)) (length (second parts)) (length (third parts))
            (length names))))

(defun prepare-toy ()
  (ensure-example-directories *toy-root*)
  (let ((data (merge-pathnames "data/" *toy-root*)))
    (write-toy "iris" (read-toy-classification (merge-pathnames "iris.csv" data) 4 nil)
               *iris-names* t)
    (write-toy "breast_cancer"
               (read-toy-classification (merge-pathnames "breast_cancer.csv" data) 30 t)
               *cancer-names* t)
    (write-toy "diabetes" (read-toy-diabetes data) *diabetes-names* nil)))

(prepare-toy)
