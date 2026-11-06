(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests-fraud.lisp to execute this test in an isolated process."))

(require :sb-posix)

;;;; Exercise the real credit card fraud evaluator on small, isolated fixtures.

(load (merge-pathnames "common.lisp" *load-truename*))

(load (merge-pathnames "lifecycle.lisp" *load-truename*))

(in-package #:cl-gbdt-examples)

(defparameter *fraud-test-examples-root* (uiop:pathname-directory-pathname *load-truename*))

;; Defined by the copied evaluate.lisp, which these tests load before calling it.
(declaim (ftype function fraud-top-k))

(defun write-fraud-test-fixture (root test-labels test-scores seen)
  "Write validation and test labels, both backends' scores and the unseen-feature mask."
  (flet ((labels-file (name labels)
           (let ((path (merge-pathnames (format nil "data/~A-y.f32" name) root)))
             (ensure-directories-exist path)
             (with-open-file (out path :direction :output :if-exists :supersede
                                       :element-type '(unsigned-byte 8))
               (dolist (label labels) (write-f32 out label)))))
         (scores-file (name scores)
           (dolist (backend '("lightgbm" "xgboost"))
             (let ((path (merge-pathnames (format nil "output/~A-~A.csv" backend name) root)))
               (ensure-directories-exist path)
               (with-open-file (out path :direction :output :if-exists :supersede)
                 (dolist (score scores) (format out "~,15F~%" score)))))))
    (labels-file "valid" '(0f0 1f0))
    (scores-file "valid" '(0.1d0 0.9d0))
    (labels-file "test" test-labels)
    (scores-file "test" test-scores)
    (with-open-file (out (merge-pathnames "data/test-seen.u8" root)
                         :direction :output :if-exists :supersede
                         :element-type '(unsigned-byte 8))
      (dolist (flag seen) (write-byte flag out)))))

(defun run-fraud-test-evaluator (root test-labels test-scores seen)
  "Run the copied evaluator and return its printed output and its metrics.sexp."
  (write-fraud-test-fixture root test-labels test-scores seen)
  (let ((output (let ((*error-output* (make-string-output-stream)))
                  (with-output-to-string (*standard-output*)
                    (load (merge-pathnames "evaluate.lisp" root))))))
    (values output
            (with-open-file (in (merge-pathnames "output/metrics.sexp" root))
              (let ((*read-eval* nil)) (read in))))))

(defun close-to-p (expected actual)
  (and (realp actual) (< (abs (- expected actual)) 1d-12)))

(defun test-fraud-top-k ()
  ;; One fraud among three rows tied at the cut for two places: the result is the
  ;; expectation over tie breaks, whichever row holds the fraud.
  (let ((first (fraud-top-k #(1 0 0 0) #(0.5d0 0.5d0 0.5d0 0.1d0) 2))
        (last (fraud-top-k #(0 0 1 0) #(0.5d0 0.5d0 0.5d0 0.1d0) 2)))
    (assert (equal first last))
    (assert (= (getf first :k) 2))
    (assert (close-to-p (/ 2d0 3d0) (getf first :frauds-found)))
    (assert (close-to-p (/ 1d0 3d0) (getf first :precision)))
    (assert (close-to-p (/ 2d0 3d0) (getf first :recall)))
    (assert (= (getf first :tied-at-cutoff) 3)))
  ;; Rows strictly above the cut count fully; a tie entirely inside it is exact.
  (let ((result (fraud-top-k #(1 0 1 0) #(0.9d0 0.9d0 0.5d0 0.5d0) 3)))
    (assert (close-to-p 1.5d0 (getf result :frauds-found)))
    (assert (= (getf result :tied-at-cutoff) 2)))
  (let ((result (fraud-top-k #(1 0 1 0) #(0.9d0 0.8d0 0.7d0 0.6d0) 2)))
    (assert (close-to-p 1d0 (getf result :frauds-found)))
    (assert (close-to-p 0.5d0 (getf result :recall)))
    (assert (= (getf result :tied-at-cutoff) 0)))
  ;; K beyond the row count covers every row.
  (let ((result (fraud-top-k #(1 0) #(0.2d0 0.1d0) 100)))
    (assert (= (getf result :k) 2))
    (assert (close-to-p 1d0 (getf result :recall))))
  ;; Recall is undefined, not a division by zero, when there is no fraud.
  (let ((result (fraud-top-k #(0 0) #(0.2d0 0.1d0) 1)))
    (assert (close-to-p 0d0 (getf result :frauds-found)))
    (assert (close-to-p 0d0 (getf result :precision)))
    (assert (null (getf result :recall))))
  (dolist (k '(0 -1 1.5))
    (assert (handler-case (progn (fraud-top-k #(1 0) #(0.2d0 0.1d0) k) nil)
              (error () t)))))

(defun run-fraud-tests ()
  (let* ((tree (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-fraud-test-XXXXXX"
                                              (uiop:temporary-directory))))))
         (root (merge-pathnames "credit-card-fraud/" tree)))
    (unwind-protect
         (progn
           (dolist (file '("common.lisp" "lifecycle.lisp" "credit-card-fraud/evaluate.lisp"))
             (let ((path (merge-pathnames file tree)))
               (ensure-directories-exist path)
               (uiop:copy-file (merge-pathnames file *fraud-test-examples-root*) path)))
           (cl-gbdt-example-lifecycle:with-preparation (root) nil)
           (cl-gbdt-example-lifecycle:with-training (root) nil)
           ;; Unseen-feature rows without fraud are reported, not an aborted evaluation.
           (multiple-value-bind (output metrics)
               (run-fraud-test-evaluator root '(0f0 1f0 0f0) '(0.1d0 0.9d0 0.2d0) '(0 1 0))
             (dolist (entry metrics)
               (let ((unseen (getf (rest entry) :test-unseen-features)))
                 (assert (= (getf unseen :rows) 2))
                 (assert (= (getf unseen :positives) 0))
                 (assert (null (getf unseen :average-precision)))
                 (assert (null (getf unseen :roc-auc)))
                 (assert (null (getf unseen :recall)))))
             (assert (search "test_unseen: n=2 fraud=0 AP=undefined AUC=undefined" output)))
           ;; With no unseen-feature rows at all, that subset is reported as absent.
           (multiple-value-bind (output metrics)
               (run-fraud-test-evaluator root '(0f0 1f0) '(0.1d0 0.9d0) '(1 1))
             (dolist (entry metrics)
               (assert (member :test-unseen-features (rest entry)))
               (assert (null (getf (rest entry) :test-unseen-features))))
             (assert (search "test_unseen: no rows" output)))
           ;; The ordinary report is unchanged in shape.
           (multiple-value-bind (output metrics)
               (run-fraud-test-evaluator root '(0f0 1f0 0f0) '(0.1d0 0.9d0 0.2d0) '(0 0 0))
             (assert (equal (mapcar #'first metrics) '("lightgbm" "xgboost")))
             (dolist (entry metrics)
               (let ((top (getf (rest entry) :test-top100))
                     (unseen (getf (rest entry) :test-unseen-features)))
                 (assert (= (getf top :k) 3))
                 (assert (close-to-p 1d0 (getf top :recall)))
                 (assert (close-to-p 1d0 (getf unseen :roc-auc)))))
             (assert (search "test_unseen: n=3 fraud=1 AP=1.00000 AUC=1.00000" output)))
           (test-fraud-top-k))
      (when (probe-file tree) (uiop:delete-directory-tree tree :validate t))))
  (format t "Credit card fraud evaluation tests passed.~%"))

(run-fraud-tests)
