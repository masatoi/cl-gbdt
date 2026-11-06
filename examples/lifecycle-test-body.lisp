(in-package #:cl-user)

(unless (let* ((package (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS"))
               (symbol (and package (find-symbol "*CHILD*" package))))
          (and symbol (boundp symbol) (symbol-value symbol)))
  (error "Load examples/tests-example-lifecycle.lisp to run these tests in a child process."))

(require :sb-posix)
(load (merge-pathnames "common.lisp" *load-truename*))

(load (merge-pathnames "lifecycle.lisp" *load-truename*))

(defparameter *lifecycle-test-sources* (uiop:pathname-directory-pathname *load-truename*))

(defun lifecycle-test-write-labels (path values)
  (ensure-directories-exist path)
  (with-open-file (out path :direction :output :if-exists :supersede
                            :element-type '(unsigned-byte 8))
    (dolist (value values) (cl-gbdt-examples:write-f32 out value))))

(defun lifecycle-test-write-text (path text)
  (ensure-directories-exist path)
  (with-open-file (out path :direction :output :if-exists :supersede)
    (write-string text out)))

(defun lifecycle-test-make-tree ()
  (let ((tree (uiop:ensure-directory-pathname
               (sb-posix:mkdtemp
                (namestring (merge-pathnames "cl-gbdt-example-lifecycle-XXXXXX"
                                             (uiop:temporary-directory)))))))
    (dolist (file (cons "common.lisp"
                        ;; The trainers load the shared reader once it exists.
                        (remove-if-not (lambda (file)
                                         (probe-file (merge-pathnames file
                                                                      *lifecycle-test-sources*)))
                                       '("lifecycle.lisp" "train-io.lisp"))))
      (uiop:copy-file (merge-pathnames file *lifecycle-test-sources*)
                      (merge-pathnames file tree)))
    (dolist (name '("adult" "credit-card-fraud" "online-retail"
                    "kasandr-sparse" "household-power"))
      (let ((root (merge-pathnames (format nil "~A/" name) tree))
             (source (merge-pathnames (format nil "~A/evaluate.lisp" name)
                                      *lifecycle-test-sources*)))
        (ensure-directories-exist (merge-pathnames "evaluate.lisp" root))
        (uiop:copy-file source (merge-pathnames "evaluate.lisp" root))
        (dolist (file '("prepare.lisp" "train.lisp"))
          (uiop:copy-file (merge-pathnames (format nil "~A/~A" name file)
                                           *lifecycle-test-sources*)
                          (merge-pathnames file root)))
        (when (equal name "household-power")
          (uiop:copy-file (merge-pathnames "household-power/custom-objective.lisp"
                                           *lifecycle-test-sources*)
                          (merge-pathnames "custom-objective.lisp" root)))
        (dolist (split '("valid" "test"))
          (lifecycle-test-write-labels
           (merge-pathnames (format nil "data/~A-y.f32" split) root) '(0f0 1f0))
          (dolist (backend '("lightgbm" "xgboost"))
            (lifecycle-test-write-text
             (merge-pathnames (format nil "output/~A-~A.csv" backend split) root)
             (format nil "0.1~%0.9~%"))))
        (when (equal name "adult")
          (lifecycle-test-write-text (merge-pathnames "data/test-meta.csv" root)
                                     (format nil "income,sex,race~%0,A,X~%1,B,Y~%")))
        (when (equal name "credit-card-fraud")
          (with-open-file (out (merge-pathnames "data/test-seen.u8" root)
                               :direction :output :if-exists :supersede
                               :element-type '(unsigned-byte 8))
            (write-byte 0 out) (write-byte 0 out)))
        (when (equal name "household-power")
          (lifecycle-test-write-text (merge-pathnames "data/test-meta.csv" root)
                                     (format nil "now,target~%1,2~%2,3~%")))))
    tree))

(defun lifecycle-test-load-evaluator (root)
  (let ((*error-output* (make-string-output-stream)))
    (with-output-to-string (*standard-output*)
      (load (merge-pathnames "evaluate.lisp" root)))))

(defun lifecycle-test-rejects-stale ()
  (let ((tree (lifecycle-test-make-tree)))
    (unwind-protect
         (dolist (name '("adult" "credit-card-fraud" "online-retail"
                         "kasandr-sparse" "household-power"))
           (let* ((root (merge-pathnames (format nil "~A/" name) tree))
                  (rejected (handler-case
                                (progn (lifecycle-test-load-evaluator root) nil)
                              (error (condition)
                                (search "state is not current" (princ-to-string condition))))))
             (assert rejected () "~A evaluated stale unmarked predictions" name)))
      (uiop:delete-directory-tree tree :validate t))))

(defun lifecycle-test-mark-current (root)
  ;; The fixture supplies synthetic files instead of running the data publishers.
  (cl-gbdt-example-lifecycle:with-preparation (root) nil)
  (cl-gbdt-example-lifecycle:with-training (root) nil))

(defun lifecycle-test-fails-p (thunk)
  (let ((*error-output* (make-string-output-stream)))
    (handler-case (progn (funcall thunk) nil) (error () t))))

(defun lifecycle-test-phases ()
  (let ((tree (lifecycle-test-make-tree))
         (other (lifecycle-test-make-tree))
         (old-source (sb-ext:posix-getenv "CL_GBDT_CREDIT_CSV"))
         (old-xl (find-package "LISP-XL"))
         (extra-exports nil)
         (extra-functions nil))
    (unwind-protect
         (progn
           ;; The workbook is intentionally absent; defining its API symbols
           ;; allows the real prepare file to be read in a fresh test image.
           (unless (find-package "LISP-XL")
             (make-package "LISP-XL" :use '("CL")))
           (dolist (name '("READ-SHEET" "PROCESS-SHEET" "CLOSE-SHEET"))
             (multiple-value-bind (symbol status) (find-symbol name "LISP-XL")
               (let ((symbol (or symbol (intern name "LISP-XL"))))
                 (unless (eq status :external)
                   (push (cons symbol status) extra-exports)
                   (export symbol "LISP-XL"))
                 (unless (fboundp symbol)
                   (push symbol extra-functions)
                   (setf (symbol-function symbol)
                         (lambda (&rest ignored)
                           (declare (ignore ignored))
                           (error "Workbook fixture is intentionally absent")))))))
           (sb-posix:setenv "CL_GBDT_CREDIT_CSV"
                            (namestring (merge-pathnames "absent.csv" tree)) 1)
           (dolist (entry '(("adult" "*ADULT-ROOT*" "ADULT-EXAMPLE")
                            ("credit-card-fraud" "*FRAUD-ROOT*" "FRAUD-ANALYSIS")
                            ("online-retail" "*RETAIL-ROOT*" "ONLINE-RETAIL-EXAMPLE")
                            ("kasandr-sparse" "*KASANDR-ROOT*" "KASANDR-SPARSE-EXAMPLE")
                            ("household-power" "*POWER-ROOT*" "HOUSEHOLD-POWER-EXAMPLE")))
             (let* ((name (first entry))
                    (root (merge-pathnames (format nil "~A/" name) tree))
                    (other-root (merge-pathnames (format nil "~A/" name) other))
                    (score (merge-pathnames "output/lightgbm-test.csv" root))
                    (old-score (uiop:read-file-string score)))
               (lifecycle-test-mark-current root)
               (assert (search "lightgbm" (lifecycle-test-load-evaluator root)))
               ;; The real trainer fails due to missing input. Its old score
               ;; file remains, but the old successful training is invalidated.
               (assert (lifecycle-test-fails-p
                        (lambda () (load (merge-pathnames "train.lisp" root)))))
               (assert (string= old-score (uiop:read-file-string score)))
               (assert (lifecycle-test-fails-p
                        (lambda () (lifecycle-test-load-evaluator root))))
               ;; A failed real preparation invalidates both old markers too.
               (lifecycle-test-mark-current root)
               (assert (lifecycle-test-fails-p
                        (lambda () (load (merge-pathnames "prepare.lisp" root)))))
               (assert (string= old-score (uiop:read-file-string score)))
               (assert (lifecycle-test-fails-p
                        (lambda () (lifecycle-test-load-evaluator root))))
               ;; Loading the same script in another checkout while this
               ;; checkout's special variable is bound must affect the other.
               (lifecycle-test-mark-current root)
               (lifecycle-test-mark-current other-root)
               (let ((symbol (intern (second entry) "CL-GBDT-EXAMPLES"))
                      (before (uiop:read-file-string
                               (merge-pathnames "data/.prepared-generation" root))))
                 (progv (list symbol) (list root)
                   (assert (lifecycle-test-fails-p
                            (lambda ()
                              (load (merge-pathnames "prepare.lisp" other-root))))))
                 (assert (string= before
                                  (uiop:read-file-string
                                   (merge-pathnames "data/.prepared-generation" root))))
                 (assert (search "lightgbm" (lifecycle-test-load-evaluator root)))
                 (assert (lifecycle-test-fails-p
                          (lambda () (lifecycle-test-load-evaluator other-root)))))
               ;; The trainer's own package also uses dynamic roots.
               (lifecycle-test-mark-current other-root)
               (let ((symbol (intern "*ROOT*" (third entry)))
                      (before (uiop:read-file-string
                               (merge-pathnames "output/.trained-generation" root))))
                 (progv (list symbol) (list root)
                   (assert (lifecycle-test-fails-p
                            (lambda () (load (merge-pathnames "train.lisp" other-root))))))
                 (assert (string= before
                                  (uiop:read-file-string
                                   (merge-pathnames "output/.trained-generation" root))))
                 (assert (search "lightgbm" (lifecycle-test-load-evaluator root)))
                 (assert (lifecycle-test-fails-p
                          (lambda () (lifecycle-test-load-evaluator other-root)))))))
           ;; Optional custom predictions must never be associated with a
           ;; later fold merely because their old CSVs still exist.
           (let ((root (merge-pathnames "household-power/" tree)))
             (dolist (backend '("lightgbm" "xgboost"))
               (lifecycle-test-write-text
                (merge-pathnames (format nil "output/~A-custom-test.csv" backend) root)
                (format nil "0.1~%0.9~%")))
             (lifecycle-test-mark-current root)
             (assert (not (search "-custom:" (lifecycle-test-load-evaluator root))))
             (cl-gbdt-example-lifecycle:with-training (root :custom) nil)
             (assert (search "lightgbm-custom:" (lifecycle-test-load-evaluator root)))
             (cl-gbdt-example-lifecycle:with-preparation (root) nil)
             (cl-gbdt-example-lifecycle:with-training (root) nil)
             (assert (not (search "-custom:" (lifecycle-test-load-evaluator root)))))
           ;; Nonlocal exits invalidate the matching stage; a lock for the
           ;; same root fails immediately while a different root is usable.
           (let ((a (merge-pathnames "adult/" tree))
                 (b (merge-pathnames "adult/" other)))
             (lifecycle-test-mark-current a)
             (catch 'cancel
               (cl-gbdt-example-lifecycle:with-training (a) (throw 'cancel nil)))
             (assert (lifecycle-test-fails-p (lambda () (lifecycle-test-load-evaluator a))))
             (lifecycle-test-mark-current a)
             (catch 'cancel
               (cl-gbdt-example-lifecycle:with-preparation (a) (throw 'cancel nil)))
             (assert (lifecycle-test-fails-p (lambda () (lifecycle-test-load-evaluator a))))
             (cl-gbdt-example-lifecycle::call-with-lock
              a (lambda ()
                  (assert (lifecycle-test-fails-p
                           (lambda () (cl-gbdt-example-lifecycle::call-with-lock a (lambda () t)))))
                  (cl-gbdt-example-lifecycle::call-with-lock b (lambda () t))))))
      (if old-source (sb-posix:setenv "CL_GBDT_CREDIT_CSV" old-source 1)
          (sb-posix:unsetenv "CL_GBDT_CREDIT_CSV"))
      (dolist (symbol extra-functions) (fmakunbound symbol))
      (if old-xl
          (dolist (entry extra-exports)
            (unexport (car entry) old-xl)
            (unless (cdr entry) (unintern (car entry) old-xl)))
          (when (find-package "LISP-XL") (delete-package "LISP-XL")))
      (uiop:delete-directory-tree tree :validate t)
      (uiop:delete-directory-tree other :validate t))))

(defun lifecycle-test-write-dense (root split rows columns regression-p)
  (let ((x (merge-pathnames (format nil "data/~A-x.f32" split) root))
        (y (merge-pathnames (format nil "data/~A-y.f32" split) root)))
    (with-open-file (xs x :direction :output :if-exists :supersede
                          :element-type '(unsigned-byte 8))
      (with-open-file (ys y :direction :output :if-exists :supersede
                            :element-type '(unsigned-byte 8))
        (dotimes (i rows)
          (let ((label (if regression-p (+ 1f0 (* 2f0 (mod i 2))) (float (mod i 2) 1f0))))
            (cl-gbdt-examples:write-f32 ys label)
            (dotimes (j columns)
              (cl-gbdt-examples:write-f32 xs
                (if (= j (1- columns)) (float (mod i 2) 1f0)
                    (+ (float (mod i 2) 1f0) (/ (mod i 7) 10f0)))))))))))

(defun lifecycle-test-write-csr (root split rows)
  (flet ((path (suffix) (merge-pathnames (format nil "data/~A-~A.i32" split suffix) root)))
    (with-open-file (indices (path "indices") :direction :output :if-exists :supersede
                                                   :element-type '(unsigned-byte 8))
      (with-open-file (indptr (path "indptr") :direction :output :if-exists :supersede
                                                 :element-type '(unsigned-byte 8))
        (cl-gbdt-examples:write-i32 indptr 0)
        (dotimes (i rows)
          (dolist (column (if (evenp i) '(0 1 2 4) '(0 1 3 5)))
            (cl-gbdt-examples:write-i32 indices column))
          (cl-gbdt-examples:write-i32 indptr (* (1+ i) 4)))))))

(defun lifecycle-test-write-training-fixture (root name)
  (let* ((power (equal name "household-power"))
         (csr (equal name "kasandr-sparse"))
         (columns (cond ((equal name "adult") 14) ((equal name "credit-card-fraud") 29)
                        ((equal name "online-retail") 8) (csr 8192) (t 3))))
    (cl-gbdt-example-lifecycle:with-preparation (root)
      (dolist (split '(("train" 60) ("valid" 20) ("test" 20)))
        (if csr
            (progn
              (lifecycle-test-write-csr root (first split) (second split))
              (lifecycle-test-write-labels
               (merge-pathnames (format nil "data/~A-y.f32" (first split)) root)
               (loop for i below (second split) collect (float (mod i 2) 1f0))))
            (lifecycle-test-write-dense root (first split) (second split) columns power)))
      (when power (lifecycle-test-write-dense root "mini" 60 columns t))
      (when (equal name "online-retail")
        (lifecycle-test-write-labels (merge-pathnames "data/train-w.f32" root)
                                     (loop repeat 60 collect 1f0)))
      (when (equal name "credit-card-fraud")
        (with-open-file (out (merge-pathnames "data/test-seen.u8" root)
                             :direction :output :if-exists :supersede
                             :element-type '(unsigned-byte 8))
          (dotimes (i 20) (write-byte 0 out))))
      (when (equal name "adult")
        (lifecycle-test-write-text
         (merge-pathnames "data/test-meta.csv" root)
         (with-output-to-string (out)
           (write-line "income,sex,race" out)
           (dotimes (i 20) (format out "~D,~A,X~%" (mod i 2) (if (evenp i) "A" "B"))))))
      (when power
        (lifecycle-test-write-text
         (merge-pathnames "data/test-meta.csv" root)
         (with-output-to-string (out)
           (write-line "now,target" out)
           (dotimes (i 20) (format out "~D,~D~%" (1+ (mod i 2)) (+ 1 (* 2 (mod i 2))))))))
      (with-open-file (out (merge-pathnames "data/sizes.lisp" root)
                           :direction :output :if-exists :supersede)
        (write (list :train 60 :valid 20 :test 20 :mini 60 :features columns
                     :categorical nil :nonzeros-per-row 4)
               :stream out)))))

(defun lifecycle-test-real-backends ()
  (let ((tree (lifecycle-test-make-tree)))
    (unwind-protect
         (dolist (name '("adult" "credit-card-fraud" "online-retail"
                         "kasandr-sparse" "household-power"))
           (let ((root (merge-pathnames (format nil "~A/" name) tree)))
             (lifecycle-test-write-training-fixture root name)
             (handler-case
                 (let ((*standard-output* (make-string-output-stream)))
                   (load (merge-pathnames "train.lisp" root)))
               (error (condition) (error "~A real backend training failed: ~A" name condition)))
             (let ((output (lifecycle-test-load-evaluator root)))
               (assert (and (search "lightgbm" output) (search "xgboost" output))))
             ;; If the second backend cannot open, preserve the entire old
             ;; first score file even though the first backend completed.
             (let ((score (merge-pathnames "output/lightgbm-test.csv" root))
                    (previous (with-output-to-string (out)
                                (dotimes (i 20) (write-line "0.12345" out))))
                    (open-backend (symbol-function 'cl-gbdt:open-backend)))
               (lifecycle-test-write-text score previous)
               (unwind-protect
                    (progn
                      (setf (symbol-function 'cl-gbdt:open-backend)
                            (lambda (backend &rest arguments)
                              (if (eq backend :xgboost)
                                  (error "Simulated second-backend failure")
                                  (apply open-backend backend arguments))))
                      (assert (lifecycle-test-fails-p
                               (lambda ()
                                 (let ((*standard-output* (make-string-output-stream)))
                                   (load (merge-pathnames "train.lisp" root)))))))
                 (setf (symbol-function 'cl-gbdt:open-backend) open-backend))
               (assert (string= previous (uiop:read-file-string score)) ()
                       "~A replaced an old score after a failed training" name)
               (assert (lifecycle-test-fails-p
                        (lambda () (lifecycle-test-load-evaluator root))))
               (let ((*standard-output* (make-string-output-stream)))
                 (load (merge-pathnames "train.lisp" root))))
             (when (equal name "household-power")
               (handler-case
                   (let ((*standard-output* (make-string-output-stream)))
                     (load (merge-pathnames "custom-objective.lisp" root)))
                 (error (condition) (error "Custom backend training failed: ~A" condition)))
               (let ((output (lifecycle-test-load-evaluator root)))
                 (assert (and (search "lightgbm-custom:" output)
                              (search "xgboost-custom:" output)))))))
      (uiop:delete-directory-tree tree :validate t))))

(defun lifecycle-test-publication-rollback ()
  (let* ((tree (lifecycle-test-make-tree))
         (root (merge-pathnames "kasandr-sparse/" tree))
         (names '("lightgbm-test.csv" "xgboost-test.csv"))
         (first (merge-pathnames "output/lightgbm-test.csv" root))
         (second (merge-pathnames "output/xgboost-test.csv" root))
         (old-first (uiop:read-file-string first))
         (old-second (uiop:read-file-string second)))
    (unwind-protect
         (labels ((write-staged ()
                    (dolist (name names)
                      (lifecycle-test-write-text
                       (merge-pathnames name cl-gbdt-example-lifecycle:*training-output-directory*)
                       "new"))))
           ;; Abort immediately after installing the first staged file.
           (lifecycle-test-mark-current root)
           (let ((rename (symbol-function 'cl-gbdt-example-lifecycle::rename-output))
                 (calls 0))
             (unwind-protect
                  (progn
                    (setf (symbol-function 'cl-gbdt-example-lifecycle::rename-output)
                          (lambda (from to)
                            (funcall rename from to)
                            (when (= (incf calls) 3) (throw 'cancel :interrupted))))
                    (assert (eq :interrupted
                                (catch 'cancel
                                  (cl-gbdt-example-lifecycle:with-training
                                      (root :standard names)
                                    (write-staged))
                                  :unexpected))))
               (setf (symbol-function 'cl-gbdt-example-lifecycle::rename-output) rename)))
           (assert (string= old-first (uiop:read-file-string first)))
           (assert (string= old-second (uiop:read-file-string second)))
           (assert (lifecycle-test-fails-p (lambda () (lifecycle-test-load-evaluator root))))
           ;; A nonlocal exit immediately after publishing the marker must
           ;; remove it and restore both previous score files as well.
           (lifecycle-test-mark-current root)
           (let ((publish (symbol-function 'cl-gbdt-example-lifecycle::publish-generation)))
             (unwind-protect
                  (progn
                    (setf (symbol-function 'cl-gbdt-example-lifecycle::publish-generation)
                          (lambda (directory kind value)
                            (funcall publish directory kind value)
                            (when (eq kind :standard) (throw 'cancel :marker-interrupted))))
                    (assert (eq :marker-interrupted
                                (catch 'cancel
                                  (cl-gbdt-example-lifecycle:with-training
                                      (root :standard names)
                                    (write-staged))
                                  :unexpected))))
               (setf (symbol-function 'cl-gbdt-example-lifecycle::publish-generation) publish)))
           (assert (string= old-first (uiop:read-file-string first)))
           (assert (string= old-second (uiop:read-file-string second)))
           (assert (lifecycle-test-fails-p (lambda () (lifecycle-test-load-evaluator root))))
           ;; If rollback itself fails, leave the old files in the private
           ;; directory and report where they can be recovered.
           (lifecycle-test-mark-current root)
           (let ((rename (symbol-function 'cl-gbdt-example-lifecycle::rename-output))
                 (calls 0)
                 (message nil))
             (unwind-protect
                  (progn
                    (setf (symbol-function 'cl-gbdt-example-lifecycle::rename-output)
                          (lambda (from to)
                            (incf calls)
                            (when (> calls 3) (error "Simulated restore failure"))
                            (funcall rename from to)
                            (when (= calls 3) (error "Simulated publication failure"))))
                    (let ((*error-output* (make-string-output-stream)))
                      (setf message
                            (handler-case
                                (progn (cl-gbdt-example-lifecycle:with-training
                                           (root :standard names)
                                         (write-staged)) nil)
                              (error (condition) (princ-to-string condition))))))
               (setf (symbol-function 'cl-gbdt-example-lifecycle::rename-output) rename))
             (assert (and message (search "recovery failed" message)))
             (let* ((directories (uiop:subdirectories (merge-pathnames "output/" root)))
                    (stage (find-if (lambda (path)
                                      (search ".example-training-" (namestring path)))
                                    directories)))
               (assert (and stage (search (namestring stage) message)))
               (assert (string= old-first
                                (uiop:read-file-string
                                 (merge-pathnames ".previous-lightgbm-test.csv" stage))))
               (assert (string= old-second
                                (uiop:read-file-string
                                 (merge-pathnames ".previous-xgboost-test.csv" stage)))))
             (assert (lifecycle-test-fails-p (lambda () (lifecycle-test-load-evaluator root))))))
      (uiop:delete-directory-tree tree :validate t))))

(defun lifecycle-test-interrupted-rollback ()
  (let* ((tree (lifecycle-test-make-tree))
         (root (merge-pathnames "kasandr-sparse/" tree))
         (names '("lightgbm-test.csv" "xgboost-test.csv"))
         (previous (uiop:read-file-string (merge-pathnames "output/lightgbm-test.csv" root)))
         (rename (symbol-function 'cl-gbdt-example-lifecycle::rename-output))
         (calls 0)
         (stderr (make-string-output-stream)))
    (unwind-protect
         (progn
           (lifecycle-test-mark-current root)
           (unwind-protect
                (progn
                  (setf (symbol-function 'cl-gbdt-example-lifecycle::rename-output)
                        (lambda (from to)
                          (incf calls)
                          (when (= calls 4) (throw 'cancel :recovery-interrupted))
                          (funcall rename from to)
                          (when (= calls 3) (error "Simulated publication failure"))))
                  (let ((*error-output* stderr))
                    (assert
                     (eq :recovery-interrupted
                         (catch 'cancel
                           (cl-gbdt-example-lifecycle:with-training (root :standard names)
                             (dolist (name names)
                               (lifecycle-test-write-text
                                (merge-pathnames
                                 name cl-gbdt-example-lifecycle:*training-output-directory*)
                                "new")))
                           :unexpected)))))
             (setf (symbol-function 'cl-gbdt-example-lifecycle::rename-output) rename))
           (let ((stage (find-if (lambda (path)
                                    (search ".example-training-" (namestring path)))
                                  (uiop:subdirectories (merge-pathnames "output/" root))))
                  (message (get-output-stream-string stderr)))
             (assert (and stage (search (namestring stage) message)))
             (assert (string= previous
                              (uiop:read-file-string
                               (merge-pathnames ".previous-lightgbm-test.csv" stage)))))
           (assert (lifecycle-test-fails-p (lambda () (lifecycle-test-load-evaluator root)))))
      (uiop:delete-directory-tree tree :validate t))))

(defun lifecycle-test-independent-roots ()
  (let* ((tree (lifecycle-test-make-tree))
         (other (lifecycle-test-make-tree))
         (a (merge-pathnames "online-retail/" tree))
         (b (merge-pathnames "online-retail/" other))
         (ready (sb-thread:make-semaphore))
         (release (sb-thread:make-semaphore))
         (worker nil))
    (unwind-protect
         (progn
           (lifecycle-test-mark-current a)
           (lifecycle-test-mark-current b)
           (let ((old-a (uiop:read-file-string (merge-pathnames "data/.prepared-generation" a))))
             (setf worker
                   (sb-thread:make-thread
                    (lambda ()
                      (cl-gbdt-example-lifecycle::call-with-lock
                       a (lambda ()
                           (sb-thread:signal-semaphore ready)
                           (unless (sb-thread:wait-on-semaphore release :timeout 10)
                             (error "Timed out waiting to release test lock")))))))
             (assert (sb-thread:wait-on-semaphore ready :timeout 5))
             (assert (lifecycle-test-fails-p (lambda () (lifecycle-test-load-evaluator a))))
             ;; While A is held by another thread, B can prepare, train and
             ;; evaluate normally, without changing A's generation.
             (lifecycle-test-mark-current b)
             (assert (search "lightgbm" (lifecycle-test-load-evaluator b)))
             (assert (string= old-a
                              (uiop:read-file-string
                               (merge-pathnames "data/.prepared-generation" a))))
             (sb-thread:signal-semaphore release)
             (sb-thread:join-thread worker :timeout 5)
             (setf worker nil)
             (assert (search "lightgbm" (lifecycle-test-load-evaluator a)))))
      (when worker
        (sb-thread:signal-semaphore release)
        (sb-thread:join-thread worker :timeout 5 :default nil))
      (uiop:delete-directory-tree tree :validate t)
      (uiop:delete-directory-tree other :validate t))))

(defun lifecycle-test-ready-invalidation-failure ()
  (let* ((tree (lifecycle-test-make-tree))
         (root (merge-pathnames "kasandr-sparse/" tree))
         (names '("lightgbm-test.csv" "xgboost-test.csv"))
         (remove (symbol-function 'cl-gbdt-example-lifecycle::remove-marker))
         (publish (symbol-function 'cl-gbdt-example-lifecycle::publish-generation))
         (published nil))
    (unwind-protect
         (progn
           (lifecycle-test-mark-current root)
           (unwind-protect
                (progn
                  (setf (symbol-function 'cl-gbdt-example-lifecycle::remove-marker)
                        (lambda (directory kind)
                          (when (and published (eq kind :standard))
                            (error "Simulated ready marker deletion failure"))
                          (funcall remove directory kind))
                        (symbol-function 'cl-gbdt-example-lifecycle::publish-generation)
                        (lambda (directory kind value)
                          (funcall publish directory kind value)
                          (when (eq kind :standard)
                            (setf published t)
                            (throw 'cancel :interrupted))))
                  (let ((*error-output* (make-string-output-stream)))
                    (handler-case
                        (catch 'cancel
                          (cl-gbdt-example-lifecycle:with-training (root :standard names)
                            (dolist (name names)
                              (lifecycle-test-write-text
                               (merge-pathnames
                                name cl-gbdt-example-lifecycle:*training-output-directory*)
                               "new"))))
                      (error () nil))))
             (setf (symbol-function 'cl-gbdt-example-lifecycle::remove-marker) remove
                   (symbol-function 'cl-gbdt-example-lifecycle::publish-generation) publish))
           (assert published)
           (assert (probe-file (merge-pathnames "output/.trained-generation" root)))
           (assert (not (probe-file (merge-pathnames "data/.prepared-generation" root))))
           (assert (lifecycle-test-fails-p (lambda () (lifecycle-test-load-evaluator root)))))
      (uiop:delete-directory-tree tree :validate t))))

(defun lifecycle-test-read-only-evaluation ()
  (let* ((tree (lifecycle-test-make-tree))
         (root (merge-pathnames "online-retail/" tree))
         (lock (merge-pathnames "output/.example.lock" root))
         (data (merge-pathnames "data/" root))
         (output (merge-pathnames "output/" root)))
    (unwind-protect
         (progn
           (lifecycle-test-mark-current root)
           (sb-posix:chmod (namestring lock) #o444)
           (sb-posix:chmod (namestring data) #o555)
           (sb-posix:chmod (namestring output) #o555)
           (assert (search "lightgbm" (lifecycle-test-load-evaluator root)))
           ;; Another evaluation holding the read-only lock still excludes this one.
           (cl-gbdt-example-lifecycle:call-with-lock
            root (lambda ()
                   (assert (lifecycle-test-fails-p
                            (lambda () (lifecycle-test-load-evaluator root)))))
            :read-only-ok t))
      (when (probe-file data) (sb-posix:chmod (namestring data) #o700))
      (when (probe-file output) (sb-posix:chmod (namestring output) #o700))
      (when (probe-file lock) (sb-posix:chmod (namestring lock) #o600))
      (uiop:delete-directory-tree tree :validate t))))

(defun lifecycle-test-both-marker-invalidations-fail ()
  (let* ((tree (lifecycle-test-make-tree))
         (root (merge-pathnames "kasandr-sparse/" tree))
         (names '("lightgbm-test.csv" "xgboost-test.csv"))
         (remove (symbol-function 'cl-gbdt-example-lifecycle::remove-marker))
         (publish (symbol-function 'cl-gbdt-example-lifecycle::publish-generation))
         (old (uiop:read-file-string (merge-pathnames "output/lightgbm-test.csv" root)))
         (new (format nil "0.9~%0.1~%"))
         (published nil)
         (message nil))
    (unwind-protect
         (progn
           (lifecycle-test-mark-current root)
           (unwind-protect
                (progn
                  (setf (symbol-function 'cl-gbdt-example-lifecycle::remove-marker)
                        (lambda (directory kind)
                          (when (and published (member kind '(:standard :prepared)))
                            (error "Simulated state deletion failure for ~A" kind))
                          (funcall remove directory kind))
                        (symbol-function 'cl-gbdt-example-lifecycle::publish-generation)
                        (lambda (directory kind value)
                          (funcall publish directory kind value)
                          (when (eq kind :standard)
                            (setf published t)
                            (throw 'cancel :interrupted))))
                  (let ((*error-output* (make-string-output-stream)))
                    (setf message
                          (handler-case
                              (catch 'cancel
                                (cl-gbdt-example-lifecycle:with-training (root :standard names)
                                  (dolist (name names)
                                    (lifecycle-test-write-text
                                     (merge-pathnames
                                      name cl-gbdt-example-lifecycle:*training-output-directory*)
                                     new))))
                            (error (condition) (princ-to-string condition))))))
             (setf (symbol-function 'cl-gbdt-example-lifecycle::remove-marker) remove
                   (symbol-function 'cl-gbdt-example-lifecycle::publish-generation) publish))
           (assert (and published (stringp message) (search "recovery failed" message)))
           ;; The current marker cannot be removed, so retain the matching new
           ;; pair and keep the old pair in the reported recovery directory.
           (dolist (name names)
             (assert (string= new (uiop:read-file-string
                                   (merge-pathnames (format nil "output/~A" name) root)))))
           (let ((stage (find-if (lambda (path) (search ".example-training-" (namestring path)))
                                 (uiop:subdirectories (merge-pathnames "output/" root)))))
             (assert (and stage (search (namestring stage) message)))
             (assert (string= old (uiop:read-file-string
                                   (merge-pathnames ".previous-lightgbm-test.csv" stage)))))
           (assert (search "auc=0.00000" (lifecycle-test-load-evaluator root))))
      (uiop:delete-directory-tree tree :validate t))))

(defun lifecycle-test-fraud-report-publication ()
  (let* ((tree (lifecycle-test-make-tree))
         (root (merge-pathnames "credit-card-fraud/" tree))
         (metrics (merge-pathnames "output/metrics.sexp" root))
         (svg (merge-pathnames "output/precision-recall.svg" root))
         (old-metrics "prior metrics")
         (old-svg "prior svg"))
    (unwind-protect
         (progn
           (lifecycle-test-mark-current root)
           ;; Load and invoke the real evaluator once before injecting faults.
           (lifecycle-test-load-evaluator root)
           (lifecycle-test-write-text metrics old-metrics)
           (lifecycle-test-write-text svg old-svg)
           (let ((plot (symbol-function 'cl-gbdt-examples::write-fraud-plot))
                 (output (make-string-output-stream)))
             (unwind-protect
                  (progn
                    (setf (symbol-function 'cl-gbdt-examples::write-fraud-plot)
                          (lambda (&rest ignored)
                            (declare (ignore ignored))
                            (error "Simulated SVG generation failure")))
                    (let ((*standard-output* output))
                      (assert (lifecycle-test-fails-p
                               (lambda ()
                                 (funcall (symbol-function 'cl-gbdt-examples::evaluate-fraud)
                                          root))))))
               (setf (symbol-function 'cl-gbdt-examples::write-fraud-plot) plot))
             (assert (zerop (length (get-output-stream-string output)))))
           (assert (string= old-metrics (uiop:read-file-string metrics)))
           (assert (string= old-svg (uiop:read-file-string svg)))
           ;; An abort during publication must restore both report files too.
           (let ((rename (symbol-function 'cl-gbdt-example-lifecycle::rename-output))
                 (calls 0))
             (unwind-protect
                  (progn
                    (setf (symbol-function 'cl-gbdt-example-lifecycle::rename-output)
                          (lambda (from to)
                            (funcall rename from to)
                            (when (= (incf calls) 3) (throw 'cancel :interrupted))))
                    (let ((*standard-output* (make-string-output-stream)))
                      (assert (eq :interrupted
                                  (catch 'cancel
                                    (funcall (symbol-function 'cl-gbdt-examples::evaluate-fraud)
                                             root)
                                    :unexpected)))))
               (setf (symbol-function 'cl-gbdt-example-lifecycle::rename-output) rename)))
           (assert (string= old-metrics (uiop:read-file-string metrics)))
           (assert (string= old-svg (uiop:read-file-string svg)))
           (assert (search "lightgbm" (lifecycle-test-load-evaluator root))))
      (uiop:delete-directory-tree tree :validate t))))

(defun run-example-lifecycle-tests ()
  (let ((created-packages nil)
        (roots nil))
    (unwind-protect
         (progn
           (dolist (package '("ADULT-EXAMPLE" "FRAUD-ANALYSIS" "ONLINE-RETAIL-EXAMPLE"
                              "KASANDR-SPARSE-EXAMPLE" "HOUSEHOLD-POWER-EXAMPLE"
                              "POWER-CUSTOM-OBJECTIVE"))
             (unless (find-package package)
               (push (make-package package :use '("CL")) created-packages))
             (push (intern "*ROOT*" package) roots))
           (dolist (name '("*ADULT-ROOT*" "*FRAUD-ROOT*" "*RETAIL-ROOT*"
                           "*KASANDR-ROOT*" "*POWER-ROOT*"))
             (push (intern name "CL-GBDT-EXAMPLES") roots))
           ;; DEFVARs in copied examples must not leave an old scratch path
           ;; in a live image, whether the symbol was bound or unbound.
           (progv roots (mapcar (lambda (symbol)
                                  (and (boundp symbol) (symbol-value symbol)))
                                roots)
             (lifecycle-test-rejects-stale)
             (lifecycle-test-phases)
             (lifecycle-test-publication-rollback)
             (lifecycle-test-fraud-report-publication)
             (lifecycle-test-interrupted-rollback)
             (lifecycle-test-ready-invalidation-failure)
             (lifecycle-test-both-marker-invalidations-fail)
             (lifecycle-test-independent-roots)
             (lifecycle-test-read-only-evaluation)
             (lifecycle-test-real-backends)))
      (dolist (package created-packages) (delete-package package)))))

(run-example-lifecycle-tests)

(format t "Example lifecycle tests passed.~%")
