(in-package #:cl-gbdt-examples)

;;;; Shared ranking evaluator; callers provide the example directory.

(defun evaluate-ranking (root)
  "Print the baseline's and both backends' mean NDCG@10 on ROOT's test queries."
  (let* ((data (merge-pathnames "data/" root))
         (groups (with-open-file (in (merge-pathnames "test-groups.lisp" data))
                   (let ((*read-eval* nil)) (read in))))
         (relevance (read-f32 (merge-pathnames "test-y.f32" data)))
         (meta (merge-pathnames "test-meta.csv" data)))
    (unless (every (lambda (label)
                     (and (finite-real-p label) (<= 0 label 4) (= label (truncate label))))
                   relevance)
      (error "Ranking relevance must be an integer from 0 to 4."))
    (when (probe-file meta)
      (with-open-file (in meta)
        (let* ((header (read-csv-row in))
               (index (position "baseline" header :test #'string=)))
          (when index
            (let ((baseline (make-array 0 :adjustable t :fill-pointer 0)))
              (loop for row = (read-csv-row in) while row
                    do (vector-push-extend (parse-number (nth index row)) baseline))
              (format t "baseline: NDCG@10=~,4F over ~D groups~%"
                      (ranking-ndcg relevance baseline groups) (length groups)))))))
    (dolist (backend '("lightgbm" "xgboost"))
      (let ((scores (read-scores (merge-pathnames (format nil "output/~A-test.csv" backend)
                                                  root))))
        (format t "~A: NDCG@10=~,4F over ~D groups~%"
                backend (ranking-ndcg relevance scores groups) (length groups))))))
