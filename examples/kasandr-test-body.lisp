(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests-kasandr.lisp to execute this test in an isolated process."))

(require :sb-posix)

;;;; Exercise the real KASANDR evaluator on small, isolated prediction fixtures.

(load (merge-pathnames "common.lisp" *load-truename*))

(load (merge-pathnames "lifecycle.lisp" *load-truename*))

(in-package #:cl-gbdt-examples)

(defparameter *kasandr-test-examples-root* (uiop:pathname-directory-pathname *load-truename*))

(defun write-kasandr-test-fixture (root labels scores)
  (let ((label-path (merge-pathnames "data/test-y.f32" root)))
    (ensure-directories-exist label-path)
    (with-open-file (out label-path :direction :output :if-exists :supersede
                                    :element-type '(unsigned-byte 8))
      (dolist (label labels) (write-f32 out label))))
  (dolist (backend '("lightgbm" "xgboost"))
    (let ((path (merge-pathnames (format nil "output/~A-test.csv" backend) root)))
      (ensure-directories-exist path)
      (with-open-file (out path :direction :output :if-exists :supersede)
        (dolist (score scores) (format out "~,15F~%" score))))))

(defun run-kasandr-test-evaluator (root labels scores)
  (write-kasandr-test-fixture root labels scores)
  (let ((*error-output* (make-string-output-stream)))
    (with-output-to-string (*standard-output*)
      (load (merge-pathnames "evaluate.lisp" root)))))

(defun run-kasandr-tests ()
  (let* ((tree (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-kasandr-test-XXXXXX"
                                              (uiop:temporary-directory))))))
         (root (merge-pathnames "kasandr-sparse/" tree)))
    (unwind-protect
         (progn
           (dolist (file '("common.lisp" "lifecycle.lisp" "kasandr-sparse/evaluate.lisp"))
             (let ((path (merge-pathnames file tree)))
               (ensure-directories-exist path)
               (uiop:copy-file (merge-pathnames file *kasandr-test-examples-root*) path)))
           (cl-gbdt-example-lifecycle:with-preparation (root) nil)
           (cl-gbdt-example-lifecycle:with-training (root) nil)
           (let ((negative (run-kasandr-test-evaluator root '(0f0) '(0.2d0)))
                 (positive (run-kasandr-test-evaluator root '(1f0) '(0.8d0)))
                 (both (run-kasandr-test-evaluator root '(0f0 1f0) '(0.1d0 0.9d0))))
             (dolist (backend '("lightgbm" "xgboost"))
               (assert (search (format nil "~A: ap=undefined auc=undefined logloss=0.22314"
                                       backend)
                               negative))
               (assert (search (format nil "~A: ap=1.00000 auc=undefined logloss=0.22314"
                                       backend)
                               positive))
               (assert (search (format nil "~A: ap=1.00000 auc=1.00000 logloss=0.10536"
                                       backend)
                               both))))
           (let ((negative (binary-metrics #(0) #(0.2d0) 0.5d0 t))
                 (false-positive (binary-metrics #(0) #(0.8d0) 0.5d0 t)))
             (assert (null (getf negative :recall)))
             (assert (null (getf negative :f1)))
             (assert (null (getf false-positive :recall)))
             (assert (= 0d0 (getf false-positive :f1))))
           (assert (handler-case (progn (binary-metrics #(0) #(0.2d0)) nil)
                     (error () t)))
           (dolist (invalid-label '(0.4f0 1.4f0))
             (assert (handler-case
                         (progn
                           (run-kasandr-test-evaluator root (list invalid-label) '(0.2d0))
                           nil)
                       (error () t))))
           (assert (handler-case
                       (progn (run-kasandr-test-evaluator root '(0f0) '(1.5d0)) nil)
                     (error () t)))
           (assert (handler-case
                       (progn (run-kasandr-test-evaluator root '(0f0 1f0) '(0.2d0)) nil)
                     (error () t))))
      (when (probe-file tree) (uiop:delete-directory-tree tree :validate t))))
  (format t "KASANDR evaluation tests passed.~%"))

(run-kasandr-tests)
