(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))

(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(in-package #:cl-gbdt-examples)

(defun fraud-labels (path)
  (map 'vector (lambda (x)
                 (unless (or (= x 0) (= x 1)) (error "Invalid fraud label ~A" x))
                 (round x))
       (read-f32 path)))

(defun read-bytes-vector (path)
  (with-open-file (in path :element-type '(unsigned-byte 8))
    (let ((values (make-array (file-length in) :element-type '(unsigned-byte 8))))
      (unless (= (read-sequence values in) (length values)) (error "Short read: ~A" path))
      values)))

(defun fraud-top-k (labels scores k)
  "Count frauds among the K highest SCORES, independently of row order.
Rows tied with the score at the cut share its remaining places: they contribute the
expected number of frauds under a uniformly random tie break, so :FRAUDS-FOUND,
:PRECISION and :RECALL can be fractional. :TIED-AT-CUTOFF is the size of a tied
group straddling the cut, or 0. :RECALL is NIL when there is no fraud."
  (unless (and (integerp k) (plusp k)) (error "Top-k cutoff must be a positive integer"))
  (unless (= (length labels) (length scores)) (error "Mismatched label and score counts"))
  (let* ((n (min k (length labels)))
         (positives (count 1 labels))
         (cut (when (plusp n)
                (aref (sort (copy-seq scores) #'>) (1- n))))
         (above 0) (above-hits 0) (tied 0) (tied-hits 0))
    (dotimes (i (length labels))
      (let ((score (aref scores i)) (label (aref labels i)))
        (cond ((null cut))
              ((> score cut) (incf above) (incf above-hits label))
              ((= score cut) (incf tied) (incf tied-hits label)))))
    (let ((found (if (plusp tied)
                     (+ above-hits (* (- n above) (/ tied-hits tied)))
                     above-hits)))
      (list :k n :frauds-found (coerce found 'double-float)
            :precision (when (plusp n) (/ found n 1d0))
            :recall (when (plusp positives) (/ found positives 1d0))
            :tied-at-cutoff (if (> (+ above tied) n) tied 0)))))

(defun fraud-unseen (labels scores seen)
  (unless (= (length labels) (length scores) (length seen)) (error "Unseen mask length mismatch"))
  (let ((ys (make-array 0 :adjustable t :fill-pointer 0))
        (ps (make-array 0 :adjustable t :fill-pointer 0)))
    (dotimes (i (length labels))
      (unless (member (aref seen i) '(0 1)) (error "Invalid seen flag"))
      (when (zerop (aref seen i))
        (vector-push-extend (aref labels i) ys)
        (vector-push-extend (aref scores i) ps)))
    (values ys ps)))

(defun write-pr-path (out labels scores color)
  (let ((order (stable-sort (coerce (loop for i below (length labels) collect i) 'vector)
                             #'> :key (lambda (i) (aref scores i))))
         (positives (count 1 labels)) (tp 0) (seen 0) (previous-x 80d0))
    (format out "<polyline fill='none' stroke='~A' stroke-width='2' points='80,30 " color)
    (loop for start = 0 then end
          for score = (aref scores (aref order start))
          for end = (or (position-if (lambda (i) (/= (aref scores i) score))
                                     order :start start) (length order))
          do (loop for j from start below end
                   do (incf seen) (incf tp (aref labels (aref order j))))
             (let ((x (+ 80 (* 720 (/ tp positives 1d0))))
                   (y (- 500 (* 470 (/ tp seen 1d0)))))
               (format out "~,2F,~,2F ~,2F,~,2F " previous-x y x y)
               (setf previous-x x))
          when (= end (length order)) do (return))
    (format out "'/>~%")))

(defun write-fraud-plot (path labels scores-list metrics)
  (with-open-file (out path :direction :output :if-exists :supersede)
    (format out "~A~A~%" "<svg xmlns='http://www.w3.org/2000/svg' width='840' "
            "height='560' viewBox='0 0 840 560'>")
    (format out "~A~A~%" "<rect width='840' height='560' fill='white'/>"
            "<path d='M80 30V500H800' fill='none' stroke='black'/>")
    (format out "<text x='280' y='20'>Credit card fraud: test precision-recall</text>~%")
    (format out "<text x='400' y='545'>Recall</text><text x='8' y='25'>Precision</text>~%")
    (let ((prevalence (/ (count 1 labels) (length labels) 1d0)))
      (format out "<path d='M80 ~,2F H800' stroke='#6b7280' stroke-dasharray='5 5'/>~%"
              (- 500 (* 470 prevalence)))
      (loop for scores in scores-list for color in '("#ed8b36" "#2265ae") for m in metrics
            do (write-pr-path out labels scores color)
               (format out "<circle cx='~,2F' cy='~,2F' r='5' fill='~A'/>~%"
                       (+ 80 (* 720 (getf m :recall)))
                       (- 500 (* 470 (getf m :precision))) color))
      (format out "~A~A~%" "<rect x='96' y='380' width='215' height='78' fill='white' "
              "stroke='#e5e7eb'/>")
      (loop for color in '("#ed8b36" "#2265ae") for model in '("LightGBM" "XGBoost")
            for m in metrics for y from 400 by 20
            do (format out "<text x='108' y='~D' fill='~A'>~A AP=~,3F</text>~%"
                       y color model (getf m :average-precision)))
      (format out "<text x='108' y='440' fill='#6b7280'>Random baseline: ~,4F</text>~%"
              prevalence))
    (format out "<text x='70' y='520'>0</text><text x='790' y='520'>1</text>")
    (format out "<text x='55' y='505'>0</text><text x='55' y='35'>1</text></svg>~%")))

(defun evaluate-fraud (root)
  (cl-gbdt-example-lifecycle:with-evaluation (root)
    (cl-gbdt-example-lifecycle:with-report-outputs
        (root '("metrics.sexp" "precision-recall.svg"))
      (let* ((data (merge-pathnames "data/" root)) (out (merge-pathnames "output/" root))
             (yv (fraud-labels (merge-pathnames "valid-y.f32" data)))
             (yt (fraud-labels (merge-pathnames "test-y.f32" data)))
             (seen (read-bytes-vector (merge-pathnames "test-seen.u8" data)))
             (results nil) (scores-list nil) (test-metrics nil))
        (dolist (model '("lightgbm" "xgboost"))
          (let* ((pv (read-scores (merge-pathnames (format nil "~A-valid.csv" model) out)))
                 (pt (read-scores (merge-pathnames (format nil "~A-test.csv" model) out)))
                 (threshold (best-threshold yv pv))
                 (validation (binary-metrics yv pv threshold))
                 (test (binary-metrics yt pt threshold))
                 (default (binary-metrics yt pt 0.5d0)))
            (multiple-value-bind (uy up) (fraud-unseen yt pt seen)
              ;; The unseen-feature subset may lack fraud, or rows, altogether; its
              ;; undefined metrics are reported as NIL rather than ending the run.
              (let ((unseen (when (plusp (length uy)) (binary-metrics uy up threshold t)))
                    (top (fraud-top-k yt pt 100)))
                (push (list model :validation validation :test-tuned test :test-default default
                            :test-top100 top :test-unseen-features unseen)
                      results)
                (format t "~%== ~A == threshold from validation: ~,8F~%" model threshold)
                (flet ((text (value digits)
                         (if value (format nil "~,vF" digits value) "undefined")))
                  (dolist (pair (list (cons "validation" validation) (cons "test_tuned" test)
                                      (cons "test_default" default) (cons "test_unseen" unseen)))
                    (let ((m (cdr pair)))
                      (if m
                          (format t (concatenate 'string "~A: n=~D fraud=~D AP=~A AUC=~A "
                                                 "P=~A R=~A F1=~A TP=~D FP=~D FN=~D TN=~D~%")
                                  (car pair) (getf m :rows) (getf m :positives)
                                  (text (getf m :average-precision) 5)
                                  (text (getf m :roc-auc) 5) (text (getf m :precision) 4)
                                  (text (getf m :recall) 4) (text (getf m :f1) 4)
                                  (getf m :tp) (getf m :fp) (getf m :fn) (getf m :tn))
                          (format t "~A: no rows~%" (car pair)))))
                  (format t "test_top100: k=~D frauds=~,2F P=~A R=~A tied_at_cutoff=~D~%"
                          (getf top :k) (getf top :frauds-found) (text (getf top :precision) 4)
                          (text (getf top :recall) 4) (getf top :tied-at-cutoff)))))
            (push pt scores-list)
            (push test test-metrics)))
        (with-open-file (stream (merge-pathnames "metrics.sexp"
                                         cl-gbdt-example-lifecycle:*report-output-directory*)
                                :direction :output :if-exists :supersede)
          (let ((*print-readably* t) (*print-case* :downcase))
            (write (nreverse results) :stream stream) (terpri stream)))
        (write-fraud-plot (merge-pathnames "precision-recall.svg"
                                            cl-gbdt-example-lifecycle:*report-output-directory*) yt
                          (nreverse scores-list) (nreverse test-metrics))))))

(evaluate-fraud (uiop:pathname-directory-pathname *load-truename*))
