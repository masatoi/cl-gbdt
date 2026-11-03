(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

(defun evaluate-clicks (root)
  (let* ((labels (map 'vector #'round (read-f32 (merge-pathnames "data/test-y.f32" root))))
         (positive (count 1 labels)))
    (format t "test rows=~D prevalence=~,6F~%" (length labels) (/ positive (length labels) 1d0))
    (dolist (backend '("lightgbm" "xgboost"))
      (let* ((scores (read-scores (merge-pathnames (format nil "output/~A-test.csv" backend)
                                                   root)))
             (m (binary-metrics labels scores)))
        (format t "~A: ap=~,5F auc=~,5F logloss=~,5F~%"
                backend (getf m :average-precision) (getf m :roc-auc) (getf m :log-loss))))))

(evaluate-clicks (uiop:pathname-directory-pathname *load-truename*))
