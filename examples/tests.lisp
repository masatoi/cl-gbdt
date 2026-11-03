(in-package #:cl-user)

(load (merge-pathnames "common.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

(defun run-example-tests ()
  (let* ((labels #(0 1 0 1))
         (scores #(0.5d0 0.5d0 0.25d0 0.75d0))
         (metrics (binary-metrics labels scores 0.5d0)))
    (assert (< (abs (- (getf metrics :roc-auc) 0.875d0)) 1d-12))
    (assert (< (abs (- (getf metrics :average-precision) (/ 5d0 6d0))) 1d-12))
    (assert (= (getf metrics :tp) 2))
    (assert (= (getf metrics :fp) 1))
    (assert (= (best-threshold labels scores) 0.5d0))
    (assert (= (ranking-ndcg #(3 2 0) #(3d0 2d0 0d0) '(3)) 1d0))
    (assert (= (ranking-ndcg #(0 0) #(1d0 2d0) '(2)) 0d0)))
  (multiple-value-bind (train valid test) (split-indexes 100 42)
    (assert (= (length train) 60))
    (assert (= (length valid) 20))
    (assert (= (length test) 20))
    (assert (= (length (remove-duplicates
                        (concatenate 'list train valid test))) 100))
    (multiple-value-bind (again) (split-indexes 100 42)
      (assert (equalp train again))))
  (with-input-from-string (stream (format nil "a,~Cb,~%c~C,~Cd~C~Ce~C~%1,2,3~%"
                                          #\" #\" #\" #\" #\" #\"))
    (assert (equal (read-csv-row stream)
                   (list "a" (format nil "b,~%c") "d\"e")))
    (assert (equal (read-csv-row stream) '("1" "2" "3")))
    (assert (null (read-csv-row stream))))
  (assert (handler-case (progn (parse-number "#.(error \"unsafe\")") nil)
            (error () t)))
  (let ((path (merge-pathnames "toy-data/data/test-f32.bin" *load-truename*))
        (wanted #(1.5f0 -2.25f0 0f0)))
    (ensure-directories-exist path)
    (unwind-protect
         (progn
           (with-open-file (out path :direction :output :if-exists :supersede
                                     :element-type '(unsigned-byte 8))
             (loop for value across wanted do (write-f32 out value)))
           (assert (equalp wanted (read-f32 path)))
           (with-open-file (out path :direction :output :if-exists :supersede
                                     :element-type '(unsigned-byte 8))
             (write-byte 0 out))
           (assert (handler-case (progn (read-f32 path) nil) (error () t))))
      (when (probe-file path) (delete-file path))))
  (format t "Example utility tests passed.~%"))

(run-example-tests)
