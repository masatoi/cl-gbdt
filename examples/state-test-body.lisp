(in-package #:cl-user)

(unless (let* ((package (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS"))
               (symbol (and package (find-symbol "*CHILD*" package))))
          (and symbol (boundp symbol) (symbol-value symbol)))
  (error "Load examples/tests-state.lisp to run these tests in a child process."))

;;;; Checks the shared example state code in lifecycle.lisp and ranking-state.lisp against
;;;; roots it must handle: relative roots, roots whose names contain pathname wildcard
;;;; characters, and staging directories that cannot be removed. Every tree is a scratch
;;;; directory, so no prepared data of the user's is touched.

(require :sb-posix)
(load (merge-pathnames "common.lisp" *load-truename*))
(load (merge-pathnames "lifecycle.lisp" *load-truename*))
(load (merge-pathnames "ranking-state.lisp" *load-truename*))

(defpackage #:cl-gbdt-state-test
  (:use #:cl))

(in-package #:cl-gbdt-state-test)

(defparameter *sources* (uiop:pathname-directory-pathname *load-truename*)
  "The examples directory the files under test are copied from.")

(defparameter *ranking-generation* "00000000000000000000000000000001"
  "The prepared generation every ranking fixture records.")

(defparameter *score-names* '("lightgbm-test.csv" "xgboost-test.csv"))

(defun make-scratch (name)
  "Make a fresh directory whose native name starts with NAME, which may hold wildcards."
  (sb-ext:parse-native-namestring
   (sb-posix:mkdtemp (concatenate 'string (sb-ext:native-namestring (uiop:temporary-directory))
                                  name "XXXXXX"))
   nil *default-pathname-defaults* :as-directory t))

(defun write-text (path text)
  (ensure-directories-exist path)
  (with-open-file (out path :direction :output :if-exists :supersede)
    (write-string text out)))

(defun write-scores (directory text)
  (dolist (name *score-names*)
    (write-text (merge-pathnames name directory) text)))

(defun write-ranking-fixture (base)
  (cl-gbdt-examples:write-sizes (ensure-directories-exist (merge-pathnames "data/sizes.lisp" base))
                                :test 3 :generation *ranking-generation*))

(defun exercise-lifecycle-cycle (root base)
  "Prepare, train, evaluate and report through ROOT, whose absolute form is BASE."
  (cl-gbdt-example-lifecycle:with-preparation (root)
    (write-text (merge-pathnames "data/prepared.txt" base) "prepared"))
  (cl-gbdt-example-lifecycle:with-training (root :standard *score-names*)
    (write-scores cl-gbdt-example-lifecycle:*training-output-directory* "lifecycle scores"))
  (let ((printed
          (with-output-to-string (*standard-output*)
            (cl-gbdt-example-lifecycle:with-evaluation (root)
              (cl-gbdt-example-lifecycle:with-report-outputs (root '("report.txt"))
                (write-text (merge-pathnames "report.txt"
                                             cl-gbdt-example-lifecycle:*report-output-directory*)
                            "report"))
              (write-string (uiop:read-file-string
                             (merge-pathnames "output/lightgbm-test.csv" base)))))))
    (assert (string= "lifecycle scores" printed)))
  (assert (probe-file (merge-pathnames "output/.example.lock" base)))
  (assert (probe-file (merge-pathnames "output/.trained-generation" base)))
  (assert (string= "report" (uiop:read-file-string (merge-pathnames "output/report.txt" base))))
  (assert (null (uiop:subdirectories (merge-pathnames "output/" base)))))

(defun exercise-ranking-cycle (root base)
  "Train and evaluate a ranking directory through ROOT, whose absolute form is BASE."
  (write-ranking-fixture base)
  (cl-gbdt-examples::call-with-ranking-training
   root (lambda (stage) (write-scores stage "ranking scores")))
  (let ((printed
          (with-output-to-string (*standard-output*)
            (cl-gbdt-examples::call-with-ranking-evaluation
             root (lambda ()
                    (write-string (uiop:read-file-string
                                   (merge-pathnames "output/xgboost-test.csv" base))))))))
    (assert (string= "ranking scores" printed)))
  (assert (probe-file (merge-pathnames "output/.ranking.lock" base)))
  (assert (string= (format nil "~S~%" *ranking-generation*)
                   (uiop:read-file-string (merge-pathnames "output/predictions.ready" base))))
  (assert (null (uiop:subdirectories (merge-pathnames "output/" base)))))

(defun test-relative-roots ()
  ;; Lisp file functions resolve a relative root against *DEFAULT-PATHNAME-DEFAULTS*,
  ;; system calls against the process directory; both must reach the same tree.
  (let* ((scratch (make-scratch "cl-gbdt-state-relative-"))
         (prefix (car (last (pathname-directory scratch))))
         (lifecycle (format nil "~A-lifecycle/" prefix))
         (ranking (format nil "~A-ranking/" prefix))
         (cwd (uiop:getcwd)))
    (unwind-protect
         (let ((*default-pathname-defaults* scratch))
           (assert (not (equal (truename cwd) (truename scratch))))
           (exercise-lifecycle-cycle (pathname lifecycle) (merge-pathnames lifecycle scratch))
           (exercise-ranking-cycle ranking (merge-pathnames ranking scratch))
           (dolist (name (list lifecycle ranking))
             (assert (not (probe-file (merge-pathnames name cwd))))))
      (uiop:delete-directory-tree scratch :validate t))))

(defun copy-example-sources (examples files)
  (dolist (file files)
    (let ((target (merge-pathnames file examples)))
      (ensure-directories-exist target)
      (uiop:copy-file (merge-pathnames file *sources*) target))))

(defun exercise-movielens-evaluation-script (examples)
  "Load the real MovieLens evaluator from EXAMPLES, which derives its root from its truename."
  (copy-example-sources examples '("common.lisp" "lifecycle.lisp" "ranking-state.lisp"
                                   "movielens-ranking/evaluate.lisp"
                                   "movielens-ranking/evaluate-core.lisp"))
  (let ((root (merge-pathnames "movielens-ranking/" examples)))
    (write-ranking-fixture root)
    (let ((labels (merge-pathnames "data/test-y.f32" root)))
      (with-open-file (out labels :direction :output :element-type '(unsigned-byte 8))
        (dolist (label '(2f0 1f0 0f0)) (cl-gbdt-examples:write-f32 out label))))
    (cl-gbdt-examples:write-groups (merge-pathnames "data/test-groups.lisp" root) '(3))
    (write-text (merge-pathnames "output/predictions.ready" root)
                (format nil "~S~%" *ranking-generation*))
    (write-scores (merge-pathnames "output/" root) (format nil "3.0~%2.0~%1.0~%"))
    (let ((printed (with-output-to-string (*standard-output*)
                     (load (merge-pathnames "evaluate.lisp" root)))))
      (assert (search "lightgbm: NDCG@10=1.0000" printed)))
    (assert (probe-file (merge-pathnames "output/.ranking.lock" root)))))

(defun test-wildcard-characters-in-roots ()
  ;; SBCL's NAMESTRING escapes these characters; a system call given that string
  ;; names a different file.
  (let ((scratch (make-scratch "cl-gbdt-state[1]*-")))
    (unwind-protect
         (progn
           (assert (not (wild-pathname-p scratch)))
           (let ((lifecycle (merge-pathnames "lifecycle/" scratch))
                 (ranking (merge-pathnames "ranking/" scratch)))
             (exercise-lifecycle-cycle lifecycle lifecycle)
             (exercise-ranking-cycle ranking ranking))
           (exercise-movielens-evaluation-script (merge-pathnames "examples/" scratch)))
      (uiop:delete-directory-tree scratch :validate t))))

(defun make-stage-undeletable (stage)
  "Leave a file in a read-only subdirectory of STAGE, so that removing STAGE fails."
  (let ((stuck (merge-pathnames "stuck/" stage)))
    (write-text (merge-pathnames "kept" stuck) "kept")
    (sb-posix:chmod (sb-ext:native-namestring stuck) #o500)))

(defun release-undeletable-stages (output)
  (dolist (stage (uiop:subdirectories output))
    (let ((stuck (merge-pathnames "stuck/" stage)))
      (when (probe-file stuck)
        (sb-posix:chmod (sb-ext:native-namestring stuck) #o700)))
    (uiop:delete-directory-tree stage :validate t)))

(defun call-capturing-warnings (thunk)
  "Call THUNK. Return the error it signalled, or NIL, and the warnings it issued."
  (let ((warnings nil)
        (failure nil))
    (handler-bind ((warning (lambda (condition)
                              (push condition warnings)
                              (muffle-warning condition))))
      (handler-case (funcall thunk)
        (error (condition) (setf failure condition))))
    (values failure (nreverse warnings))))

(defun check-stage-cleanup-failure (root marker train)
  "TRAIN is called with ROOT and a function of the staging directory."
  (let ((output (merge-pathnames "output/" root)))
    ;; A run whose outputs and marker were published stays committed.
    (multiple-value-bind (failure warnings)
        (call-capturing-warnings
         (lambda ()
           (funcall train root (lambda (stage)
                                 (write-scores stage "new scores")
                                 (make-stage-undeletable stage)))))
      (release-undeletable-stages output)
      (assert (null failure) () "Stage cleanup failure reported as ~A" failure)
      (assert warnings () "Stage cleanup failure was not reported")
      (assert (probe-file (merge-pathnames marker output)))
      (assert (string= "new scores"
                       (uiop:read-file-string (merge-pathnames "lightgbm-test.csv" output)))))
    ;; A failed run reports its own error rather than the cleanup error.
    (multiple-value-bind (failure warnings)
        (call-capturing-warnings
         (lambda ()
           (funcall train root (lambda (stage)
                                 (make-stage-undeletable stage)
                                 (error "Primary training failure")))))
      (release-undeletable-stages output)
      (assert (and failure (search "Primary training failure" (princ-to-string failure)))
              () "Primary failure replaced by ~A" failure)
      (assert warnings () "Stage cleanup failure was not reported")
      (assert (not (probe-file (merge-pathnames marker output))))
      (assert (string= "new scores"
                       (uiop:read-file-string (merge-pathnames "lightgbm-test.csv" output)))))))

(defun test-stage-cleanup-failure ()
  (if (zerop (sb-posix:geteuid))
      (format t "Stage cleanup check skipped for root.~%")
      (let ((scratch (make-scratch "cl-gbdt-state-cleanup-")))
        (unwind-protect
             (let ((lifecycle (merge-pathnames "lifecycle/" scratch))
                   (ranking (merge-pathnames "ranking/" scratch)))
               (cl-gbdt-example-lifecycle:with-preparation (lifecycle) nil)
               (check-stage-cleanup-failure
                lifecycle ".trained-generation"
                (lambda (root function)
                  (cl-gbdt-example-lifecycle:with-training (root :standard *score-names*)
                    (funcall function cl-gbdt-example-lifecycle:*training-output-directory*))))
               (write-ranking-fixture ranking)
               (check-stage-cleanup-failure
                ranking "predictions.ready"
                (lambda (root function)
                  (cl-gbdt-examples::call-with-ranking-training root function))))
          (release-undeletable-stages (merge-pathnames "lifecycle/output/" scratch))
          (release-undeletable-stages (merge-pathnames "ranking/output/" scratch))
          (uiop:delete-directory-tree scratch :validate t)))))

(let ((failures nil))
  (dolist (test '(test-stage-cleanup-failure test-relative-roots
                  test-wildcard-characters-in-roots))
    (handler-case
        (progn (funcall test) (format t "PASS ~A~%" test))
      (error (condition)
        (push test failures)
        (format t "FAIL ~A: ~A~%" test condition))))
  (when failures (error "Example state tests failed: ~S" (nreverse failures)))
  (format t "Example state tests passed.~%"))
