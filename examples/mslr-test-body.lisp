(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests-mslr.lisp to execute this test in an isolated process."))

;;;; Exercise failed MSLR preparation against isolated, previously prepared data.

(require :sb-posix)
(load (merge-pathnames "common.lisp" *load-truename*))

;; The fault injections below name the shared publication code's rename function.
(load (merge-pathnames "lifecycle.lisp" *load-truename*))

(in-package #:cl-gbdt-examples)

(declaim (special *mslr-root*))

(declaim (ftype function read-mslr mslr-row-qid mslr-row-label prepare-mslr))

(defparameter *mslr-test-examples-root* (uiop:pathname-directory-pathname *load-truename*))

(defun write-mslr-test-fold (root &optional invalid-file alternate-labels)
  (dolist (spec '(("train.txt" 1 40) ("vali.txt" 101 20) ("test.txt" 201 20)))
    (destructuring-bind (name first-qid count) spec
      (let ((path (merge-pathnames name root)))
        (ensure-directories-exist path)
        (with-open-file (out path :direction :output :if-exists :supersede)
          (dotimes (i count)
            (format out "~D qid:~D 1:~D~%"
                    (if (and invalid-file (string= name invalid-file) (zerop i))
                        5
                        (cond ((and invalid-file (string= name "train.txt")) (1+ (mod i 4)))
                              (alternate-labels (mod (1+ i) 3))
                              (t (mod i 3))))
                    (+ first-qid (floor i 10)) i)))))))

(defun load-mslr-test-file (path)
  (let ((*standard-output* (make-string-output-stream))
        (*error-output* (make-string-output-stream)))
    (load path)))

(defun mslr-test-load-fails-p (path)
  (handler-case (progn (load-mslr-test-file path) nil)
    (error () t)))

(defun read-mslr-test-sexp (path)
  (with-open-file (in path)
    (let ((*read-eval* nil)) (read in))))

(defun test-mslr-query-order-is-stable (root)
  (let ((path (merge-pathnames "data/query-order-fixture.txt" root)))
    (with-open-file (out path :direction :output :if-exists :supersede)
      (format out "2 qid:12 1:1~%1 qid:13 1:1~%0 qid:12 1:1~%"))
    (let ((rows (read-mslr path :test (make-hash-table))))
      (assert (equal '(12 12 13) (loop for row across rows collect (mslr-row-qid row))))
      (assert (equal '(2 0 1) (loop for row across rows collect (mslr-row-label row)))))))

(defun test-mslr-evaluation-generation-change (root)
  ;; The FIFO stops the real evaluator while another thread changes its metadata.
  (let* ((sizes (merge-pathnames "data/sizes.lisp" root))
         (sizes-text (uiop:read-file-string sizes))
         (score (merge-pathnames "output/xgboost-test.csv" root))
         (scores-text (uiop:read-file-string score))
         (backup (merge-pathnames "output/xgboost-test.csv.backup" root))
         (captured (make-string-output-stream))
         (worker nil)
         (worker-error nil)
         (failed nil))
    (rename-file score backup)
    (unwind-protect
         (progn
           (sb-posix:mkfifo (namestring score) #o600)
           (setf worker
                 (sb-thread:make-thread
                  (lambda ()
                    (handler-case
                        (with-open-file (out score :direction :output :if-exists :overwrite)
                          (let ((metadata (read-mslr-test-sexp sizes)))
                            (setf (getf metadata :generation) "changed-during-evaluation")
                            (apply #'write-sizes sizes metadata))
                          (write-string scores-text out))
                      (error (condition) (setf worker-error condition))))))
           (handler-case
               (let ((*standard-output* captured)
                     (*error-output* (make-string-output-stream)))
                 (load (merge-pathnames "evaluate.lisp" root)))
             (error () (setf failed t)))
           (sb-thread:join-thread worker :timeout 10)
           (when worker-error (error worker-error))
           (assert failed)
           (assert (not (search "NDCG@10" (get-output-stream-string captured)))))
      (when (and worker (sb-thread:thread-alive-p worker))
        (sb-thread:terminate-thread worker))
      (when (probe-file score) (delete-file score))
      (rename-file backup score)
      (with-open-file (out sizes :direction :output :if-exists :supersede)
        (write-string sizes-text out)))))

(defun test-mslr-training-generation-change (root)
  ;; Even an external metadata replacement during training must not publish scores.
  (let* ((sizes (merge-pathnames "data/sizes.lisp" root))
         (sizes-text (uiop:read-file-string sizes))
         (light (merge-pathnames "output/lightgbm-test.csv" root))
         (xgb (merge-pathnames "output/xgboost-test.csv" root))
         (old-light (uiop:read-file-string light))
         (old-xgb (uiop:read-file-string xgb))
         (original (symbol-function 'cl-gbdt:predict))
         (calls 0))
    (unwind-protect
         (progn
           (setf (symbol-function 'cl-gbdt:predict)
                 (lambda (&rest args)
                   (incf calls)
                   (when (= calls 2)
                     (let* ((metadata (read-mslr-test-sexp sizes))
                            (generation (copy-seq (getf metadata :generation))))
                       (setf (char generation 0) (if (char= (char generation 0) #\A) #\B #\A)
                             (getf metadata :generation) generation)
                       (apply #'write-sizes sizes metadata)))
                   (apply original args)))
           (assert (mslr-test-load-fails-p (merge-pathnames "train.lisp" root)))
           (assert (= calls 2))
           (assert (string= old-light (uiop:read-file-string light)))
           (assert (string= old-xgb (uiop:read-file-string xgb)))
           (assert (mslr-test-load-fails-p (merge-pathnames "evaluate.lisp" root))))
      (setf (symbol-function 'cl-gbdt:predict) original)
      (with-open-file (out sizes :direction :output :if-exists :supersede)
        (write-string sizes-text out)))))

(defun call-with-rename-hook (hook function)
  "Call FUNCTION; from the second backend prediction on, renames go through HOOK.
HOOK receives the original rename function and both pathnames."
  (let ((original-predict (symbol-function 'cl-gbdt:predict))
        (original-rename (symbol-function 'cl-gbdt-example-lifecycle::rename-output))
        (predictions 0))
    (unwind-protect
         (progn
           (setf (symbol-function 'cl-gbdt:predict)
                 (lambda (&rest args)
                   (multiple-value-prog1 (apply original-predict args)
                     (when (= (incf predictions) 2)
                       (setf (symbol-function 'cl-gbdt-example-lifecycle::rename-output)
                             (lambda (from to) (funcall hook original-rename from to)))))))
           (funcall function))
      (setf (symbol-function 'cl-gbdt:predict) original-predict
            (symbol-function 'cl-gbdt-example-lifecycle::rename-output) original-rename))))

(defun test-mslr-ready-publication-failure (root)
  ;; Refuse a failed ready-file rename even after both new scores were published.
  (let ((renames 0)
        (injected nil))
    (call-with-rename-hook
     (lambda (rename from to)
       ;; Count publication renames only; recovery renames follow the failure.
       (unless injected
         (when (= (incf renames) 5)
           (setf injected t)
           (error "Simulated ready publication failure")))
       (funcall rename from to))
     (lambda ()
       (assert (mslr-test-load-fails-p (merge-pathnames "train.lisp" root)))))
    (assert (= renames 5))
    (assert (not (probe-file (merge-pathnames "output/predictions.ready" root))))
    (assert (null (directory (merge-pathnames "output/.mslr-generation-*" root))))
    (assert (mslr-test-load-fails-p (merge-pathnames "evaluate.lisp" root)))))

(defun test-mslr-publication-nonlocal-exit (root stop-after)
  (let* ((light (merge-pathnames "output/lightgbm-test.csv" root))
         (xgb (merge-pathnames "output/xgboost-test.csv" root))
         (old-light (uiop:read-file-string light))
         (old-xgb (uiop:read-file-string xgb))
         (renames 0)
         (injected nil))
    (call-with-rename-hook
     (lambda (rename from to)
       (funcall rename from to)
       (unless injected
         (when (= (incf renames) stop-after)
           (setf injected t)
           (throw 'test-interrupt :interrupted))))
     (lambda ()
       (assert (eq :interrupted
                   (catch 'test-interrupt
                     (load-mslr-test-file (merge-pathnames "train.lisp" root)))))))
    (assert (= renames stop-after))
    (assert (string= old-light (uiop:read-file-string light)))
    (assert (string= old-xgb (uiop:read-file-string xgb)))
    (assert (mslr-test-load-fails-p (merge-pathnames "evaluate.lisp" root)))
    (assert (null (uiop:subdirectories (merge-pathnames "output/" root))))))

(defun test-mslr-nonlocal-recovery-failure (root)
  ;; Block restoration after an interrupted publication.  The backup must survive.
  (let* ((xgb (merge-pathnames "output/xgboost-test.csv" root))
         (old-xgb (uiop:read-file-string xgb))
         (renames 0)
         (injected nil)
         (failure nil))
    (unwind-protect
         (progn
           (call-with-rename-hook
            (lambda (rename from to)
              (funcall rename from to)
              (unless injected
                (when (= (incf renames) 4)
                  (setf injected t)
                  ;; Put a nonempty directory where the backup must be restored.
                  (delete-file to)
                  (ensure-directories-exist
                   (merge-pathnames ".keep" (uiop:ensure-directory-pathname to)))
                  (throw 'test-interrupt :interrupted))))
            (lambda ()
              (handler-case
                  (catch 'test-interrupt
                    (load-mslr-test-file (merge-pathnames "train.lisp" root)))
                (error (condition) (setf failure condition)))))
           (assert failure)
           (assert (= renames 4))
           (let* ((stages (uiop:subdirectories (merge-pathnames "output/" root)))
                  (stage (find-if (lambda (path) (search ".mslr-training-" (namestring path)))
                                  stages))
                  (backup (and stage (merge-pathnames ".previous-xgboost-test.csv" stage))))
             (assert stage)
             (assert (string= old-xgb (uiop:read-file-string backup))))
           (assert (not (probe-file (merge-pathnames "output/predictions.ready" root)))))
      (let ((blocker (uiop:ensure-directory-pathname xgb)))
        (when (probe-file blocker) (uiop:delete-directory-tree blocker :validate t)))
      (dolist (stage (uiop:subdirectories (merge-pathnames "output/" root)))
        (when (search ".mslr-training-" (namestring stage))
          (uiop:delete-directory-tree stage :validate t))))))

(defun test-mslr-publication-rollback (root)
  ;; Fail after both backups and the first new prediction have been renamed.
  (let ((light (merge-pathnames "output/lightgbm-test.csv" root))
         (xgb (merge-pathnames "output/xgboost-test.csv" root))
         (renames 0)
         (injected nil))
    (dolist (entry (list (cons light 17) (cons xgb 23)))
      (with-open-file (out (car entry) :direction :output :if-exists :supersede)
        (dotimes (i 20) (format out "~D.0~%" (+ (cdr entry) i)))))
    (let ((old-light (uiop:read-file-string light))
          (old-xgb (uiop:read-file-string xgb)))
      (call-with-rename-hook
       (lambda (rename from to)
         ;; Count publication renames only; recovery renames follow the failure.
         (unless injected
           (when (= (incf renames) 4)
             (setf injected t)
             (error "Simulated publication failure")))
         (funcall rename from to))
       (lambda ()
         (assert (mslr-test-load-fails-p (merge-pathnames "train.lisp" root)))))
      (assert (= renames 4))
      (assert (string= old-light (uiop:read-file-string light)))
      (assert (string= old-xgb (uiop:read-file-string xgb)))
      (assert (mslr-test-load-fails-p (merge-pathnames "evaluate.lisp" root))))))

(defun run-mslr-preparation-tests ()
  (let* ((tree (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-mslr-test-XXXXXX"
                                             (uiop:temporary-directory))))))
         (examples (merge-pathnames "examples/" tree))
         (root (merge-pathnames "mslr-web10k/" examples))
         (fold (merge-pathnames "fold/" tree))
         (sizes (merge-pathnames "data/sizes.lisp" root))
         (old-fold (sb-ext:posix-getenv "CL_GBDT_MSLR_FOLD")))
    (unwind-protect
         (progn
           (dolist (file '("common.lisp" "lifecycle.lisp" "ranking-state.lisp" "train-io.lisp"
                           "mslr-web10k/state.lisp" "mslr-web10k/prepare.lisp"
                           "mslr-web10k/train.lisp" "mslr-web10k/evaluate.lisp"
                           "movielens-ranking/train.lisp" "movielens-ranking/train-core.lisp"
                           "movielens-ranking/evaluate.lisp"
                           "movielens-ranking/evaluate-core.lisp"))
             (let ((path (merge-pathnames file examples))
                   (source (merge-pathnames file *mslr-test-examples-root*)))
               ;; train-io.lisp is copied once it exists for the trainer to load.
               (when (probe-file source)
                 (ensure-directories-exist path)
                 (uiop:copy-file source path))))
           (sb-posix:setenv "CL_GBDT_MSLR_FOLD" (namestring fold) 1)
           (write-mslr-test-fold fold)
           (let ((existing (merge-pathnames "data/sizes.lisp.part" root)))
             (ensure-directories-exist existing)
             (with-open-file (out existing :direction :output :if-exists :supersede)
               (write-string "pre-existing file" out))
             (load-mslr-test-file (merge-pathnames "prepare.lisp" root))
             (assert (string= "pre-existing file" (uiop:read-file-string existing)))
             (delete-file existing))
           (assert (probe-file sizes))
           (test-mslr-query-order-is-stable root)
           ;; Existing predictions make a stale evaluation otherwise appear successful.
           (load-mslr-test-file (merge-pathnames "train.lisp" root))
           (load-mslr-test-file (merge-pathnames "evaluate.lisp" root))
           (test-mslr-evaluation-generation-change root)
           ;; A ready file from an earlier fold cannot validate the next fold.
           (let* ((ready (merge-pathnames "output/predictions.ready" root))
                  (old-ready (uiop:read-file-string ready))
                  (generation (getf (read-mslr-test-sexp sizes) :generation)))
             (assert (stringp generation))
             (write-mslr-test-fold fold nil t)
             (load-mslr-test-file (merge-pathnames "prepare.lisp" root))
             (let ((next (getf (read-mslr-test-sexp sizes) :generation)))
               (assert (and (stringp next) (not (string= generation next))))
               (load-mslr-test-file (merge-pathnames "prepare.lisp" root))
               (let ((after (getf (read-mslr-test-sexp sizes) :generation)))
                 (assert (and (stringp after) (not (string= generation after))
                              (not (string= next after))))))
             (with-open-file (out ready :direction :output :if-exists :supersede
                                        :if-does-not-exist :create)
               (write-string old-ready out))
             (assert (mslr-test-load-fails-p (merge-pathnames "evaluate.lisp" root)))
             (load-mslr-test-file (merge-pathnames "train.lisp" root))
             (load-mslr-test-file (merge-pathnames "evaluate.lisp" root)))
           ;; Interrupt the old fold's train between backends with a new preparation.
           ;; A completed train must never make this mixed pair evaluable.
           (let ((original (symbol-function 'cl-gbdt:predict))
                 (calls 0)
                 (attempted nil)
                 (prepared nil))
             (write-mslr-test-fold fold nil t)
             (unwind-protect
                  (progn
                    (setf (symbol-function 'cl-gbdt:predict)
                          (lambda (&rest args)
                            (incf calls)
                            (when (= calls 2)
                              (setf attempted t
                                    prepared
                                    (not (mslr-test-load-fails-p
                                          (merge-pathnames "prepare.lisp" root)))))
                            (apply original args)))
                    (let ((trained
                            (not (mslr-test-load-fails-p (merge-pathnames "train.lisp" root)))))
                      (assert attempted)
                      (if prepared
                          (assert (or (not trained)
                                      (mslr-test-load-fails-p
                                       (merge-pathnames "evaluate.lisp" root))))
                          (progn
                            (assert trained)
                            (assert (not (mslr-test-load-fails-p
                                          (merge-pathnames "evaluate.lisp" root))))))))
               (setf (symbol-function 'cl-gbdt:predict) original)))
           ;; A failed retraining must not leave earlier predictions marked current.
           (let ((labels (merge-pathnames "data/train-y.f32" root))
                 (backup (merge-pathnames "data/train-y.f32.backup" root)))
             (rename-file labels backup)
             (unwind-protect
                  (progn
                    (assert (mslr-test-load-fails-p (merge-pathnames "train.lisp" root)))
                    (assert (mslr-test-load-fails-p (merge-pathnames "evaluate.lisp" root))))
               (rename-file backup labels)))
           (load-mslr-test-file (merge-pathnames "train.lisp" root))
           (load-mslr-test-file (merge-pathnames "evaluate.lisp" root))
           ;; If publication of the second backend fails, preserve the first
           ;; backend's existing CSV and invalidate the ready marker.
           (let ((light (merge-pathnames "output/lightgbm-test.csv" root))
                 (score (merge-pathnames "output/xgboost-test.csv" root))
                 (backup (merge-pathnames "output/xgboost-test.csv.backup" root))
                 (blocker (merge-pathnames "output/xgboost-test.csv/" root)))
             (with-open-file (out light :direction :output :if-exists :supersede)
               (dotimes (i 20) (format out "~D.0~%" i)))
             (let ((old-light (uiop:read-file-string light)))
               (rename-file score backup)
               (ensure-directories-exist (merge-pathnames ".keep" blocker))
               (unwind-protect
                    (progn
                      (assert (mslr-test-load-fails-p (merge-pathnames "train.lisp" root)))
                      (assert (string= old-light (uiop:read-file-string light)))
                      (assert (mslr-test-load-fails-p (merge-pathnames "evaluate.lisp" root))))
                 (uiop:delete-directory-tree blocker :validate t)
                 (rename-file backup score))))
           (load-mslr-test-file (merge-pathnames "train.lisp" root))
           (load-mslr-test-file (merge-pathnames "evaluate.lisp" root))
           (test-mslr-training-generation-change root)
           (load-mslr-test-file (merge-pathnames "train.lisp" root))
           (load-mslr-test-file (merge-pathnames "evaluate.lisp" root))
           (test-mslr-publication-rollback root)
           (dotimes (index 4)
             (load-mslr-test-file (merge-pathnames "train.lisp" root))
             (load-mslr-test-file (merge-pathnames "evaluate.lisp" root))
             (test-mslr-publication-nonlocal-exit root (1+ index)))
           (assert (null (uiop:subdirectories (merge-pathnames "output/" root))))
           (load-mslr-test-file (merge-pathnames "train.lisp" root))
           (test-mslr-nonlocal-recovery-failure root)
           (load-mslr-test-file (merge-pathnames "train.lisp" root))
           (load-mslr-test-file (merge-pathnames "evaluate.lisp" root))
           (test-mslr-ready-publication-failure root)
           (load-mslr-test-file (merge-pathnames "train.lisp" root))
           (load-mslr-test-file (merge-pathnames "evaluate.lisp" root))
           ;; Repreparing a different fold must not make old predictions look current.
           (let* ((score-path (merge-pathnames "output/lightgbm-test.csv" root))
                  (old-scores (uiop:read-file-string score-path)))
             (write-mslr-test-fold fold nil t)
             (load-mslr-test-file (merge-pathnames "prepare.lisp" root))
             (assert (string= old-scores (uiop:read-file-string score-path)))
             (assert (mslr-test-load-fails-p (merge-pathnames "evaluate.lisp" root)))
             (load-mslr-test-file (merge-pathnames "train.lisp" root))
             (load-mslr-test-file (merge-pathnames "evaluate.lisp" root)))
           (dolist (invalid '("train.txt" "vali.txt" "test.txt"))
             (write-mslr-test-fold fold invalid)
             (assert (mslr-test-load-fails-p (merge-pathnames "prepare.lisp" root)))
             (assert (not (probe-file sizes)))
             (assert (mslr-test-load-fails-p (merge-pathnames "train.lisp" root)))
             (assert (mslr-test-load-fails-p (merge-pathnames "evaluate.lisp" root)))
             (write-mslr-test-fold fold)
             (load-mslr-test-file (merge-pathnames "prepare.lisp" root))
             (assert (probe-file sizes))
             (load-mslr-test-file (merge-pathnames "train.lisp" root))
             (load-mslr-test-file (merge-pathnames "evaluate.lisp" root)))
           ;; Simulate a failed metadata write after all split files have been written.
           (let ((original (symbol-function 'write-sizes)))
             (unwind-protect
                  (progn
                    (setf (symbol-function 'write-sizes)
                          (lambda (path &rest values)
                            (declare (ignore values))
                            (with-open-file (out path :direction :output :if-exists :supersede)
                              (write-string "(:train" out))
                            (error "Simulated metadata write failure")))
                    (assert (let ((*standard-output* (make-string-output-stream))
                                  (*mslr-root* root))
                              (handler-case (progn (prepare-mslr) nil) (error () t))))
                    (assert (not (probe-file sizes)))
                    (assert (null (directory (merge-pathnames "data/.mslr-sizes-*" root))))
                    (assert (mslr-test-load-fails-p (merge-pathnames "evaluate.lisp" root))))
               (setf (symbol-function 'write-sizes) original)))
           (load-mslr-test-file (merge-pathnames "prepare.lisp" root))
           (assert (probe-file sizes)))
      (if old-fold (sb-posix:setenv "CL_GBDT_MSLR_FOLD" old-fold 1)
          (sb-posix:unsetenv "CL_GBDT_MSLR_FOLD"))
      (when (probe-file tree) (uiop:delete-directory-tree tree :validate t))))
  (format t "MSLR state and preparation tests passed.~%"))

(run-mslr-preparation-tests)
