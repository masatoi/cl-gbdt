(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

(defun evaluate-retail (root)
  (let ((valid (map 'vector #'round (read-f32 (merge-pathnames "data/valid-y.f32" root))))
        (test (map 'vector #'round (read-f32 (merge-pathnames "data/test-y.f32" root)))))
    (dolist (backend '("lightgbm" "xgboost"))
      (let* ((pv (read-scores (merge-pathnames (format nil "output/~A-valid.csv" backend) root)))
             (pt (read-scores (merge-pathnames (format nil "output/~A-test.csv" backend) root)))
             (threshold (best-threshold valid pv))
             (m (binary-metrics test pt threshold)))
        (format t "~A: threshold=~,6F ap=~,4F auc=~,4F precision=~,4F recall=~,4F~%"
                backend threshold (getf m :average-precision) (getf m :roc-auc)
                (getf m :precision) (getf m :recall))))
    (format t "test prevalence=~,4F~%" (/ (count 1 test) (length test) 1d0))))

(evaluate-retail (uiop:pathname-directory-pathname *load-truename*))
