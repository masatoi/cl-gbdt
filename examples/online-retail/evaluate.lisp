(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))

(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(in-package #:cl-gbdt-examples)

(defun retail-labels (path)
  (map 'vector
       (lambda (value)
         (unless (and (finite-real-p value) (or (= value 0) (= value 1)))
           (error "Invalid retail label ~S in ~A" value path))
         (round value))
       (read-f32 path)))

(defun retail-validation-threshold (labels scores)
  (unless (= (length labels) (length scores))
    (error "Retail validation labels and scores have different lengths"))
  (loop for score across scores
        unless (and (finite-real-p score) (<= 0 score 1))
          do (error "Invalid validation probability ~S" score))
  (best-threshold labels scores))

(defun evaluate-retail (root)
  (cl-gbdt-example-lifecycle:with-evaluation (root)
    (let ((valid (retail-labels (merge-pathnames "data/valid-y.f32" root)))
          (test (retail-labels (merge-pathnames "data/test-y.f32" root))))
      (dolist (backend '("lightgbm" "xgboost"))
        (let* ((pv (read-scores (merge-pathnames (format nil "output/~A-valid.csv" backend) root)))
               (pt (read-scores (merge-pathnames (format nil "output/~A-test.csv" backend) root)))
               (threshold (retail-validation-threshold valid pv))
               (m (binary-metrics test pt threshold)))
          (format t "~A: threshold=~,6F ap=~,4F auc=~,4F precision=~,4F recall=~,4F~%"
                  backend threshold (getf m :average-precision) (getf m :roc-auc)
                  (getf m :precision) (getf m :recall))))
      (format t "test prevalence=~,4F~%" (/ (count 1 test) (length test) 1d0)))))

(evaluate-retail (uiop:pathname-directory-pathname *load-truename*))
