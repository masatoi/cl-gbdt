(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests-toy.lisp to execute this test in an isolated process."))

(require :sb-posix)
(load (merge-pathnames "common.lisp" *load-truename*))
(load (merge-pathnames "lifecycle.lisp" *load-truename*))

(defparameter *toy-test-examples-root* (uiop:pathname-directory-pathname *load-truename*))

(defun toy-test-fixture (tree name count)
  (let* ((examples (merge-pathnames (format nil "~A/examples/" name) tree))
         (root (merge-pathnames "toy-data/" examples)))
    (dolist (file '("common.lisp" "lifecycle.lisp" "toy-data/prepare.lisp"
                    "toy-data/analysis.lisp" "toy-data/state.lisp"))
      (let ((source (merge-pathnames file *toy-test-examples-root*))
            (target (merge-pathnames file examples)))
        (when (uiop:file-exists-p source)
          (ensure-directories-exist target)
          (uiop:copy-file source target))))
    (flet ((classification (file classes features)
             (let ((path (merge-pathnames (format nil "data/~A.csv" file) root)))
               (ensure-directories-exist path)
               (with-open-file (out path :direction :output :if-exists :error)
                 (format out "~D,~D~%" (* count classes) features)
                 (dotimes (class classes)
                   (dotimes (i count)
                     (dotimes (j features)
                       (format out "~,4F," (+ (* class 2d0) (/ (+ i j) 100d0))))
                     (format out "~D~%" class)))))))
      (classification "iris" 3 4)
      (classification "breast_cancer" 2 30))
    (with-open-file (xs (merge-pathnames "data/diabetes_data_raw.csv" root)
                        :direction :output :if-exists :error)
      (with-open-file (ys (merge-pathnames "data/diabetes_target.csv" root)
                          :direction :output :if-exists :error)
        (dotimes (i (* count 3))
          (dotimes (j 10)
            (format xs "~,4F " (+ (/ i 100d0) (/ j 1000d0))))
          (terpri xs)
          (format ys "~D~%" (+ 10 (mod (* i i) 70))))))
    root))

(defun toy-test-with-roots (function)
  (let ((tree (uiop:ensure-directory-pathname
               (sb-posix:mkdtemp
                (namestring (merge-pathnames "cl-gbdt-toy-test-XXXXXX"
                                            (uiop:temporary-directory)))))))
    (unwind-protect
         (funcall function (toy-test-fixture tree "a" 15)
                           (toy-test-fixture tree "b" 20))
      (uiop:delete-directory-tree tree :validate t))))

(defun toy-test-load (root name)
  (with-output-to-string (*standard-output*)
    (load (merge-pathnames name root))))

(defun toy-test-preparation-reentry ()
  (toy-test-with-roots
   (lambda (a b)
     (toy-test-load a "prepare.lisp")
     (let ((nested nil) (sentinel "SECOND CHECKOUT MUST STAY UNCHANGED"))
       (handler-bind
           ((sb-kernel:redefinition-warning
              (lambda (condition)
                (when (and (not nested)
                           (search "READ-TOY-CLASSIFICATION" (princ-to-string condition)))
                  (setf nested t)
                  (toy-test-load b "prepare.lisp")
                  (with-open-file (out (merge-pathnames "data/iris.sexp" b)
                                       :direction :output :if-exists :supersede)
                    (write-string sentinel out)))
                (muffle-warning condition))))
         (toy-test-load a "prepare.lisp"))
       (assert nested)
       (assert (string= sentinel (uiop:read-file-string (merge-pathnames "data/iris.sexp" b))))))))

(defun toy-test-analysis-reentry ()
  (toy-test-with-roots
   (lambda (a b)
     (toy-test-load a "prepare.lisp")
     (toy-test-load b "prepare.lisp")
     (toy-test-load a "analysis.lisp")
     (let* ((nested nil)
            (output
             (handler-bind
                 ((sb-kernel:redefinition-warning
                    (lambda (condition)
                      (when (and (not nested)
                                 (search "READ-SPLITS" (princ-to-string condition)))
                        (setf nested t)
                        (toy-test-load b "analysis.lisp"))
                      (muffle-warning condition))))
               (toy-test-load a "analysis.lisp"))))
       (assert nested)
       (assert (search "rows train/valid/test = 27/9/9" output))
       (assert (not (search "rows train/valid/test = 36/12/12" output)))))))

(defun toy-test-error-containing-p (function text)
  (handler-case (progn (funcall function) nil)
    (error (condition) (search text (princ-to-string condition) :test #'char-equal))))

(defun toy-test-incomplete-preparation ()
  (toy-test-with-roots
   (lambda (a b)
     (declare (ignore b))
     (toy-test-load a "prepare.lisp")
     (let* ((root-symbol (find-symbol "*TOY-ROOT*" :cl-gbdt-examples))
            (write-symbol (find-symbol "WRITE-TOY" :cl-gbdt-examples))
            (prepare-symbol (find-symbol "PREPARE-TOY" :cl-gbdt-examples))
            (original (symbol-function write-symbol)))
       (unwind-protect
            (progn
              (setf (symbol-function write-symbol)
                    (lambda (name &rest args)
                      (if (string= name "breast_cancer")
                          (error "Injected preparation failure")
                          (apply original name args))))
              (progv (list root-symbol) (list a)
                (assert (toy-test-error-containing-p
                         (lambda () (funcall prepare-symbol)) "Injected preparation failure"))))
         (setf (symbol-function write-symbol) original))
       (assert (toy-test-error-containing-p
                (lambda () (toy-test-load a "analysis.lisp")) "prepared"))))))

(defun toy-test-preserve-prior-splits ()
  (toy-test-with-roots
   (lambda (a b)
     (declare (ignore b))
     (toy-test-load a "prepare.lisp")
     (let* ((path (merge-pathnames "data/iris.sexp" a))
            (root-symbol (find-symbol "*TOY-ROOT*" :cl-gbdt-examples))
            (write-symbol (find-symbol "WRITE-TOY" :cl-gbdt-examples))
            (prepare-symbol (find-symbol "PREPARE-TOY" :cl-gbdt-examples))
            (original (symbol-function write-symbol)))
       (with-open-file (out path :direction :output :if-exists :append)
         (write-string ";; Previous data must survive a failed run." out))
       (let ((previous (uiop:read-file-string path)))
         (unwind-protect
              (progn
                (setf (symbol-function write-symbol)
                      (lambda (name &rest args)
                        (if (string= name "breast_cancer")
                            (error "Injected preparation failure")
                            (apply original name args))))
                (progv (list root-symbol) (list a)
                  (assert (toy-test-error-containing-p
                           (lambda () (funcall prepare-symbol)) "Injected preparation failure"))))
           (setf (symbol-function write-symbol) original))
         (assert (string= previous (uiop:read-file-string path))))))))

(defun toy-test-same-root-lock ()
  (toy-test-with-roots
   (lambda (a b)
     (declare (ignore b))
     (toy-test-load a "prepare.lisp")
     (cl-gbdt-example-lifecycle::call-with-lock
      a (lambda ()
          (assert (toy-test-error-containing-p
                   (lambda () (toy-test-load a "prepare.lisp")) "busy"))
          (assert (toy-test-error-containing-p
                   (lambda () (toy-test-load a "analysis.lisp")) "busy")))))))

(defun toy-test-active-preparation-reentry ()
  (toy-test-with-roots
   (lambda (a b)
     (toy-test-load a "prepare.lisp")
     (let* ((write-symbol (find-symbol "WRITE-TOY" :cl-gbdt-examples))
            (original (symbol-function write-symbol))
            (nested nil)
            (sentinel "Nested preparation must retain its files"))
       (unwind-protect
            (progn
              (setf (symbol-function write-symbol)
                    (lambda (&rest args)
                      (unless nested
                        (setf nested t)
                        (toy-test-load b "prepare.lisp")
                        (with-open-file (out (merge-pathnames "data/iris.sexp" b)
                                             :direction :output :if-exists :supersede)
                          (write-string sentinel out)))
                      (apply original args)))
              (progv (list (find-symbol "*TOY-ROOT*" :cl-gbdt-examples)) (list a)
                (with-output-to-string (*standard-output*)
                  (funcall (find-symbol "PREPARE-TOY" :cl-gbdt-examples)))))
         (setf (symbol-function write-symbol) original))
       (assert nested)
       (assert (string= sentinel (uiop:read-file-string (merge-pathnames "data/iris.sexp" b))))
       (assert (search "train=27" (toy-test-load a "prepare.lisp")))))))

(defun toy-test-aborted-publication ()
  (dolist (phase '(:rename :marker))
    (toy-test-with-roots
     (lambda (a b)
       (declare (ignore b))
       (toy-test-load a "prepare.lisp")
       (let* ((paths (mapcar (lambda (name) (merge-pathnames name a))
                             '("data/iris.sexp" "data/breast_cancer.sexp" "data/diabetes.sexp")))
              (root-symbol (find-symbol "*TOY-ROOT*" :cl-gbdt-examples))
              (prepare-symbol (find-symbol "PREPARE-TOY" :cl-gbdt-examples))
              (intercept (if (eq phase :rename)
                             'cl-gbdt-example-lifecycle::rename-output
                             'cl-gbdt-example-lifecycle::publish-generation))
              (original (symbol-function intercept))
              (count 0)
              (previous nil))
         (dolist (path paths)
           (with-open-file (out path :direction :output :if-exists :append)
             (format out "~%;; Previous ~A must survive.~%" (pathname-name path))))
         (setf previous (mapcar #'uiop:read-file-string paths))
         (unwind-protect
              (progn
                (setf (symbol-function intercept)
                      (lambda (&rest args)
                        (prog1 (apply original args)
                          (incf count)
                          ;; After three backups, the fourth rename installs a new split.
                          (when (or (eq phase :marker) (= count 4))
                            (throw 'toy-injected :injected)))))
                (assert (eq :injected
                            (catch 'toy-injected
                              (progv (list root-symbol) (list a)
                                (funcall prepare-symbol))
                              :fell-through))))
           (setf (symbol-function intercept) original))
         (assert (equal previous (mapcar #'uiop:read-file-string paths)))
         (assert (toy-test-error-containing-p
                  (lambda () (toy-test-load a "analysis.lisp")) "prepared")))))))

(let ((failures nil))
  (dolist (test '(toy-test-preparation-reentry toy-test-analysis-reentry
                  toy-test-incomplete-preparation toy-test-preserve-prior-splits
                  toy-test-same-root-lock toy-test-active-preparation-reentry
                  toy-test-aborted-publication))
    (handler-case
        (progn (funcall test) (format t "PASS ~A~%" test))
      (error (condition)
        (push test failures)
        (format t "FAIL ~A: ~A~%" test condition))))
  (when failures (error "Toy tests failed: ~S" (nreverse failures))))
