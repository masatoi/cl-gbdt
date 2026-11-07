(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))

(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(in-package #:cl-gbdt-examples)

(defun evaluate-adult (root)
  (cl-gbdt-example-lifecycle:with-evaluation (root)
    (let ((labels (make-array 0 :adjustable t :fill-pointer 0))
          (sexes (make-array 0 :adjustable t :fill-pointer 0)))
      (with-open-file (in (merge-pathnames "data/test-meta.csv" root))
        (unless (equal (read-csv-row in) '("income" "sex" "race"))
          (error "Unexpected Adult metadata columns"))
        (loop for row = (read-csv-row in) while row
              do (vector-push-extend (parse-number (first row)) labels)
                 (vector-push-extend (second row) sexes)))
      (dolist (backend '("lightgbm" "xgboost"))
        (let* ((scores (read-scores (merge-pathnames (format nil "output/~A-test.csv" backend)
                                                     root)))
               (m (binary-metrics labels scores))
               (accuracy (/ (+ (getf m :tp) (getf m :tn)) (length labels) 1d0)))
          (format t "~A: accuracy=~,4F roc_auc=~,4F average_precision=~,4F~%"
                  backend accuracy (getf m :roc-auc) (getf m :average-precision))
          (dolist (sex (sort (remove-duplicates (coerce sexes 'list) :test #'string=) #'string<))
            (let ((n 0) (correct 0) (predicted 0) (positive 0) (false-negative 0))
              (dotimes (i (length labels))
                (when (string= (aref sexes i) sex)
                  (incf n)
                  (let ((yes (>= (aref scores i) 0.5d0))
                        (actual (= (aref labels i) 1)))
                    (when (eql yes actual) (incf correct))
                    (when yes (incf predicted))
                    (when actual (incf positive))
                    (when (and actual (not yes)) (incf false-negative)))))
              (format t "  sex=~A: n=~D accuracy=~,4F positive_rate=~,4F false_negative_rate=~A~%"
                      sex n (/ correct n 1d0) (/ predicted n 1d0)
                      (if (plusp positive) (format nil "~,4F" (/ false-negative positive 1d0))
                          "undefined")))))))))

(evaluate-adult (uiop:pathname-directory-pathname *load-truename*))
