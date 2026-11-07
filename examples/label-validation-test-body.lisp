(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests-label-validation.lisp to execute this test in an isolated process."))

(require :sb-posix)
(load (merge-pathnames "common.lisp" *load-truename*))

(load (merge-pathnames "lifecycle.lisp" *load-truename*))

(defparameter *label-test-examples-root* (uiop:pathname-directory-pathname *load-truename*))

(defun label-test-error-p (thunk message)
  (handler-case (progn (funcall thunk) nil)
    (error (condition) (search message (princ-to-string condition) :test #'char-equal))))

(defun write-retail-label-test-fixture (root valid test valid-scores test-scores)
  (dolist (entry (list (cons "valid" valid) (cons "test" test)))
    (let ((path (merge-pathnames (format nil "data/~A-y.f32" (car entry)) root)))
      (ensure-directories-exist path)
      (with-open-file (out path :direction :output :if-exists :supersede
                                :element-type '(unsigned-byte 8))
        (dolist (value (cdr entry)) (cl-gbdt-examples:write-f32 out value)))))
  (dolist (backend '("lightgbm" "xgboost"))
    (dolist (entry (list (cons "valid" valid-scores) (cons "test" test-scores)))
      (let ((path (merge-pathnames (format nil "output/~A-~A.csv" backend (car entry)) root)))
        (ensure-directories-exist path)
        (with-open-file (out path :direction :output :if-exists :supersede)
          (dolist (score (cdr entry)) (format out "~A~%" score)))))))

(defun load-retail-label-test-evaluator (root)
  (let ((*error-output* (make-string-output-stream)))
    (with-output-to-string (*standard-output*)
      (load (merge-pathnames "evaluate.lisp" root)))))

(defun run-retail-label-validation-tests ()
  (let* ((tree (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-label-retail-XXXXXX"
                                             (uiop:temporary-directory))))))
         (root (merge-pathnames "online-retail/" tree)))
    (unwind-protect
         (progn
           (dolist (file '("common.lisp" "lifecycle.lisp" "online-retail/evaluate.lisp"))
             (let ((target (merge-pathnames file tree)))
               (ensure-directories-exist target)
               (uiop:copy-file (merge-pathnames file *label-test-examples-root*) target)))
           (cl-gbdt-example-lifecycle:with-preparation (root) nil)
           (cl-gbdt-example-lifecycle:with-training (root) nil)
           (flet ((fixture (valid test scores)
                    (write-retail-label-test-fixture root valid test scores '("0.1" "0.9"))))
             (fixture '(0f0 1f0) '(0f0 1f0) '("0.1" "0.9"))
             (let ((output (load-retail-label-test-evaluator root)))
               (assert (search "threshold=0.900000 ap=1.0000 auc=1.0000" output)))
             (dolist (bad '(0.4f0 1.4f0))
               (fixture (list bad 1f0) '(0f0 1f0) '("0.1" "0.9"))
               (assert (label-test-error-p
                        (lambda () (load-retail-label-test-evaluator root))
                        "Invalid retail label"))
               (fixture '(0f0 1f0) (list 0f0 bad) '("0.1" "0.9"))
               (assert (label-test-error-p
                        (lambda () (load-retail-label-test-evaluator root))
                        "Invalid retail label")))
             (dolist (bits '(#x7fc00000 #x7f800000))
               ;; Write the raw IEEE bits: WRITE-F32 correctly refuses non-finite input.
               (fixture '(0f0 1f0) '(0f0 1f0) '("0.1" "0.9"))
               (with-open-file (out (merge-pathnames "data/valid-y.f32" root)
                                    :direction :output :if-exists :supersede
                                    :element-type '(unsigned-byte 8))
                 (cl-gbdt-examples:write-i32 out bits)
                 (cl-gbdt-examples:write-i32 out #x3f800000))
               (assert (label-test-error-p
                        (lambda () (load-retail-label-test-evaluator root))
                        "Invalid retail label"))
               (let ((invalid (sb-kernel:make-single-float bits))
                     (threshold (symbol-function
                                 (find-symbol "RETAIL-VALIDATION-THRESHOLD"
                                              :cl-gbdt-examples))))
                 (assert (label-test-error-p
                          (lambda () (funcall threshold #(0 1) (vector 0.1d0 invalid)))
                          "Invalid validation probability"))))
             (dolist (scores '(("-0.1" "0.9") ("0.1" "5.0")))
               (fixture '(0f0 1f0) '(0f0 1f0) scores)
               (assert (label-test-error-p
                        (lambda () (load-retail-label-test-evaluator root))
                        "Invalid validation probability")))
             (fixture '(0f0 1f0) '(0f0 1f0) '("0.9"))
             (assert (label-test-error-p
                      (lambda () (load-retail-label-test-evaluator root))
                      "validation"))))
      (uiop:delete-directory-tree tree :validate t)))
  (format t "Retail evaluation input tests passed.~%"))

(defun load-toy-label-test-definitions ()
  "Run the real toy example on isolated valid data before testing its functions."
  (let* ((tree (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-label-toy-XXXXXX"
                                             (uiop:temporary-directory))))))
         (example (merge-pathnames "toy-data/" tree)))
    (unwind-protect
         (progn
           (dolist (file '("common.lisp" "lifecycle.lisp" "toy-data/state.lisp"
                           "toy-data/analysis.lisp"))
             (let ((target (merge-pathnames file tree)))
               (ensure-directories-exist target)
               (uiop:copy-file (merge-pathnames file *label-test-examples-root*) target)))
           (flet ((rows (count kind)
                    (loop for i below count
                          for y = (ecase kind
                                    (:iris (mod i 3))
                                    (:breast-cancer (mod i 2))
                                    (:diabetes (+ 10 (* 2 (mod i 7)))))
                          collect (list y (+ (* 1d0 y) (/ (mod i 7) 100d0))))))
             (dolist (spec '(("iris" :iris) ("breast_cancer" :breast-cancer)
                             ("diabetes" :diabetes)))
               (let* ((name (first spec))
                      (kind (second spec))
                      (path (merge-pathnames (format nil "data/~A.sexp" name) example)))
                 (ensure-directories-exist path)
                 (with-open-file (out path :direction :output :if-exists :error)
                   (write (list :train (rows 90 kind) :valid (rows 30 kind)
                                :test (rows 30 kind) :feature-names '("value"))
                          :stream out)))))
           (cl-gbdt-example-lifecycle:with-preparation (example) nil)
           (unless (find-package :cl-gbdt-open-data)
             (make-package :cl-gbdt-open-data :use '(:cl)))
           ;; Do not leave the copied example's temporary *ROOT* in the caller's image.
           (let* ((root-symbol (intern "*ROOT*" :cl-gbdt-open-data))
                  (output
                    (progv (list root-symbol) (list nil)
                      (with-output-to-string (*standard-output*)
                        (load (merge-pathnames "analysis.lisp" example))))))
             (dolist (heading '("IRIS / LIGHTGBM" "IRIS / XGBOOST"
                                "BREAST-CANCER / LIGHTGBM" "BREAST-CANCER / XGBOOST"
                                "DIABETES / LIGHTGBM" "DIABETES / XGBOOST"))
               (assert (search heading output)))))
      (uiop:delete-directory-tree tree :validate t))))

(defun make-toy-label-test-splits (classes invalid-split value)
  (flet ((rows ()
           (loop for label below classes collect (list (float label 1f0) (float label 1d0)))))
    (let ((splits (list :train (rows) :valid (rows) :test (rows) :feature-names '("value"))))
      (when invalid-split (setf (first (first (getf splits invalid-split))) value))
      splits)))

(defun run-toy-label-validation-tests ()
  (load-toy-label-test-definitions)
  (let* ((pkg :cl-gbdt-open-data)
         (run (symbol-function (find-symbol "RUN-EXAMPLE" pkg)))
         (loss (symbol-function (find-symbol "LOSS" pkg)))
         (evaluate (symbol-function (find-symbol "EVALUATE-CLASSIFICATION" pkg)))
         (open (symbol-function 'cl-gbdt:open-backend)))
    (unwind-protect
         (progn
           ;; No invalid split may reach either training backend.
           (setf (symbol-function 'cl-gbdt:open-backend)
                 (lambda (&rest ignored)
                   (declare (ignore ignored))
                   (error "Backend was reached before label validation")))
           (dolist (case '((:breast-cancer 2 0.4f0) (:breast-cancer 2 2f0)
                           (:breast-cancer 2 1.00000000001d0)
                           (:iris 3 1.4f0) (:iris 3 3f0)))
             (destructuring-bind (dataset classes invalid) case
               (dolist (split '(:train :valid :test))
                 (let ((splits (make-toy-label-test-splits classes split invalid)))
                   (assert (label-test-error-p
                            (lambda () (funcall run dataset :lightgbm splits))
                            "Invalid classification label"))))))
           ;; Non-finite values must fail before conversion, evaluation or loss.
           (dolist (bits '(#x7fc00000 #x7f800000))
             (let ((invalid (sb-kernel:make-single-float bits)))
               (dolist (split '(:train :valid :test))
                 (let ((splits (make-toy-label-test-splits 2 split invalid)))
                   (assert (label-test-error-p
                            (lambda () (funcall run :breast-cancer :lightgbm splits))
                            "Invalid classification label"))))
               (assert (label-test-error-p
                        (lambda ()
                          (funcall loss :breast-cancer
                                   (vector invalid 1f0) #2A((0.1d0) (0.9d0))))
                        "Invalid classification label"))))
           ;; Both standalone accuracy and loss must use the same label rules.
           (let ((pred #2A((0.1d0) (0.9d0))))
             (assert (label-test-error-p
                      (lambda () (funcall evaluate #(0f0 1f0) #(0.4f0 1.4f0) pred 2))
                      "Invalid classification label"))
             (assert (label-test-error-p
                      (lambda () (funcall evaluate #(0f0 1.4f0) #(0f0 1f0) pred 2))
                      "Invalid classification label"))
             (assert (label-test-error-p
                      (lambda () (funcall loss :breast-cancer #(0.4f0 1.4f0) pred))
                      "Invalid classification label"))
             (assert (plusp (funcall loss :breast-cancer #(0f0 1f0) pred)))
             (let ((output (with-output-to-string (*standard-output*)
                             (funcall evaluate #(0f0 1f0) #(0f0 1f0) pred 2))))
               (assert (search "accuracy=1.0000" output))))
           (assert (label-test-error-p
                    (lambda ()
                      (funcall loss :iris #(0f0 3f0)
                               #2A((0.8d0 0.1d0 0.1d0) (0.1d0 0.8d0 0.1d0))))
                    "Invalid classification label")))
      (setf (symbol-function 'cl-gbdt:open-backend) open)))
  (format t "Toy classification input tests passed.~%"))

(run-retail-label-validation-tests)
(run-toy-label-validation-tests)
