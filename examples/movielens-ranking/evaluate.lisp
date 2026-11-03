(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

(defun evaluate-ranking (root)
  (let* ((data (merge-pathnames "data/" root))
         (groups (with-open-file (in (merge-pathnames "test-groups.lisp" data))
                   (let ((*read-eval* nil)) (read in))))
         (labels (read-f32 (merge-pathnames "test-y.f32" data)))
         (meta (merge-pathnames "test-meta.csv" data)))
    (when (probe-file meta)
      (with-open-file (in meta)
        (let* ((header (read-csv-row in))
               (index (position "baseline" header :test #'string=)))
          (when index
            (let ((baseline (make-array 0 :adjustable t :fill-pointer 0)))
              (loop for row = (read-csv-row in) while row
                    do (vector-push-extend (parse-number (nth index row)) baseline))
              (format t "baseline: NDCG@10=~,4F over ~D groups~%"
                      (ranking-ndcg labels baseline groups) (length groups)))))))
    (dolist (backend '("lightgbm" "xgboost"))
      (let ((scores (read-scores (merge-pathnames (format nil "output/~A-test.csv" backend)
                                                  root))))
        (format t "~A: NDCG@10=~,4F over ~D groups~%"
                backend (ranking-ndcg labels scores groups) (length groups))))))

(let ((override (uiop:getenv "CL_GBDT_RANKING_ROOT")))
  (evaluate-ranking (if override (uiop:ensure-directory-pathname override)
                        (uiop:pathname-directory-pathname *load-truename*))))
