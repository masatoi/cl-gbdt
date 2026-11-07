(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))

(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(in-package #:cl-gbdt-examples)

(defun evaluate-clicks (root)
  (cl-gbdt-example-lifecycle:with-evaluation (root)
    (let* ((labels (map 'vector (lambda (value)
                                  (unless (and (finite-real-p value) (or (= value 0) (= value 1)))
                                    (error "Invalid KASANDR label ~S" value))
                                  (round value))
                        (read-f32 (merge-pathnames "data/test-y.f32" root))))
           (positive (count 1 labels)))
      (format t "test rows=~D prevalence=~,6F~%" (length labels) (/ positive (length labels) 1d0))
      (dolist (backend '("lightgbm" "xgboost"))
        (let* ((scores (read-scores (merge-pathnames (format nil "output/~A-test.csv" backend)
                                                     root)))
               (m (binary-metrics labels scores 0.5d0 t)))
          (flet ((metric-text (value)
                   (if value (format nil "~,5F" value) "undefined")))
            (format t "~A: ap=~A auc=~A logloss=~,5F~%"
                    backend (metric-text (getf m :average-precision))
                    (metric-text (getf m :roc-auc)) (getf m :log-loss))))))))

(evaluate-clicks (uiop:pathname-directory-pathname *load-truename*))
