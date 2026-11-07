(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests-ranking.lisp to execute this test in an isolated process."))

(require :sb-posix)

;;;; Checks that the MovieLens and MSLR ranking examples keep to their own directories when
;;;; both run in one image. Everything runs on a scratch copy of the example files, so the
;;;; fixtures here cannot replace data a user has prepared. Needs the backends from the
;;;; common setup in examples/README.md, because the trainer has to run for real.

(load (merge-pathnames "common.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

(load (merge-pathnames "ranking-state.lisp" *load-truename*))

(declaim (special *movie-root*))

(declaim (ftype function prepare-movie))

(defparameter *ranking-test-examples-root* (uiop:pathname-directory-pathname *load-truename*))

(defparameter *ranking-test-sources*
  '("common.lisp" "lifecycle.lisp" "ranking-state.lisp" "train-io.lisp"
    "movielens-ranking/prepare.lisp"
    "movielens-ranking/train.lisp" "movielens-ranking/train-core.lisp"
    "movielens-ranking/evaluate.lisp" "movielens-ranking/evaluate-core.lisp"
    "mslr-web10k/state.lisp" "mslr-web10k/train.lisp" "mslr-web10k/evaluate.lisp")
  "The example files copied into each scratch tree, relative to the examples directory.
A file that does not exist yet, such as a shared reader the trainer will load, is skipped.")

(defun make-ranking-test-tree ()
  "Return a fresh scratch directory holding a copy of *RANKING-TEST-SOURCES*."
  (let ((tree (uiop:ensure-directory-pathname
               (sb-posix:mkdtemp
                (namestring (merge-pathnames "cl-gbdt-ranking-test-XXXXXX"
                                            (uiop:temporary-directory)))))))
    (dolist (source *ranking-test-sources*)
      (let ((target (merge-pathnames source tree))
            (path (merge-pathnames source *ranking-test-examples-root*)))
        (when (probe-file path)
          (ensure-directories-exist target)
          (uiop:copy-file path target))))
    tree))

(defun write-f32-file (path values)
  (ensure-directories-exist path)
  (with-open-file (out path :direction :output :if-exists :supersede
                            :element-type '(unsigned-byte 8))
    (map nil (lambda (value) (write-f32 out value)) values)))

(defun write-evaluate-fixture (root labels scores)
  "Give ROOT one test query with LABELS, and SCORES as both backends' predictions."
  (write-f32-file (merge-pathnames "data/test-y.f32" root) labels)
  (write-groups (merge-pathnames "data/test-groups.lisp" root) (list (length labels)))
  (write-sizes (merge-pathnames "data/sizes.lisp" root)
               :test (length labels) :generation "00000000000000000000000000000001")
  (let ((ready (merge-pathnames "output/predictions.ready" root)))
    (ensure-directories-exist ready)
    (with-open-file (out ready :direction :output :if-exists :supersede)
      (write-line "\"00000000000000000000000000000001\"" out)))
  (dolist (backend '("lightgbm" "xgboost"))
    (let ((path (merge-pathnames (format nil "output/~A-test.csv" backend) root)))
      (ensure-directories-exist path)
      (with-open-file (out path :direction :output :if-exists :supersede)
        (format out "~{~,4F~%~}" scores)))))

(defun load-quietly (path)
  "Load PATH and return everything it printed."
  (with-output-to-string (*standard-output*)
    (load path)))

(defun ranking-test-load-fails-p (path)
  (let ((*error-output* (make-string-output-stream)))
    (handler-case (progn (load-quietly path) nil)
      (error () t))))

(defun ranking-test-fails-p (thunk)
  (let ((*error-output* (make-string-output-stream)))
    (handler-case (progn (funcall thunk) nil)
      (error () t))))

(defun write-movie-raw-fixture (root &optional alternate)
  (let ((raw (merge-pathnames "data/ml-100k/" root)))
    (ensure-directories-exist (merge-pathnames "u.user" raw))
    (with-open-file (out (merge-pathnames "u.user" raw)
                         :direction :output :if-exists :supersede)
      (dotimes (user 20) (format out "~D|30|M|work|00000~%" (1+ user))))
    (with-open-file (out (merge-pathnames "u.item" raw)
                         :direction :output :if-exists :supersede)
      (dotimes (item 8)
        (format out "~D|Movie|01-Jan-1999||url~{~A~}~%"
                (1+ item) (loop repeat 19 collect "|0"))))
    (with-open-file (out (merge-pathnames "u.data" raw)
                         :direction :output :if-exists :supersede)
      (dotimes (user 20)
        ;; Reverse the input; adjacent item IDs share a timestamp.
        (dotimes (offset 8)
          (let ((item (- 7 offset)))
            (format out "~D~C~D~C~D~C~D~%" (1+ user) #\Tab (1+ item) #\Tab
                    (1+ (mod (+ user item (if alternate 1 0)) 5))
                    #\Tab (floor item 2))))))))

(defun test-ranking-relevance-validation ()
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (dolist (example '("movielens-ranking/" "mslr-web10k/"))
           (let ((root (merge-pathnames example tree)))
             ;; Binary float bits for 5, 2.5, -1, positive infinity, and NaN.
             (dolist (bits '(#x40a00000 #x40200000 #xbf800000 #x7f800000 #x7fc00000))
               (write-evaluate-fixture root '(0f0 0f0 2f0) '(3d0 2d0 1d0))
               (with-open-file (out (merge-pathnames "data/test-y.f32" root)
                                    :direction :io :if-exists :overwrite
                                    :element-type '(unsigned-byte 8))
                 (file-position out 4)
                 (dotimes (byte 4) (write-byte (ldb (byte 8 (* byte 8)) bits) out)))
               (let ((captured (make-string-output-stream))
                     (failed nil))
                 (handler-case
                     (let ((*standard-output* captured)
                           (*error-output* (make-string-output-stream)))
                       (load (merge-pathnames "evaluate.lisp" root)))
                   (error () (setf failed t)))
                 (assert failed)
                 (assert (not (search "NDCG@10" (get-output-stream-string captured))))))))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-ranking-evaluate-read-only-output ()
  (if (zerop (sb-posix:geteuid))
      (format t "Read-only ranking check skipped for root.~%")
      (let ((tree (make-ranking-test-tree)))
        (unwind-protect
             (dolist (example '("movielens-ranking/" "mslr-web10k/"))
               (let* ((root (merge-pathnames example tree))
                      (output (merge-pathnames "output/" root))
                      (lock (merge-pathnames ".ranking.lock" output))
                      (evaluate (merge-pathnames "evaluate.lisp" root)))
                 (write-evaluate-fixture root '(2f0 1f0 0f0) '(3d0 2d0 1d0))
                 ;; First evaluation creates the persistent lock file.
                 (load-quietly evaluate)
                 (let ((output-mode (logand #o777 (sb-posix:stat-mode (sb-posix:stat output))))
                       (lock-mode (logand #o777 (sb-posix:stat-mode (sb-posix:stat lock)))))
                   (unwind-protect
                        (progn
                          (sb-posix:chmod lock #o400)
                          (sb-posix:chmod output #o500)
                          (assert (search "lightgbm: NDCG@10=1.0000"
                                          (load-quietly evaluate)))
                          (let ((blocked nil))
                            (call-with-ranking-lock
                             root (lambda ()
                                    (setf blocked (ranking-test-load-fails-p evaluate)))
                             :read-only-ok t)
                            (assert blocked))
                          ;; Training still needs a writable lock descriptor.
                          (let ((condition nil)
                                (*error-output* (make-string-output-stream)))
                            (handler-case (load-quietly (merge-pathnames "train.lisp" root))
                              (error (caught) (setf condition caught)))
                            (assert (typep condition 'sb-posix:syscall-error))
                            (assert (= (sb-posix:syscall-errno condition) sb-posix:eacces))))
                     (sb-posix:chmod output output-mode)
                     (sb-posix:chmod lock lock-mode)))))
          (uiop:delete-directory-tree tree :validate t)))))

(defun test-ranking-missing-lock-read-only-directory ()
  (unless (zerop (sb-posix:geteuid))
    (let ((tree (make-ranking-test-tree)))
      (unwind-protect
           (dolist (example '("movielens-ranking/" "mslr-web10k/"))
             (let* ((root (merge-pathnames example tree))
                    (output (merge-pathnames "output/" root))
                    (mode nil)
                    (condition nil))
               (write-evaluate-fixture root '(2f0 1f0 0f0) '(3d0 2d0 1d0))
               (setf mode (logand #o777 (sb-posix:stat-mode (sb-posix:stat output))))
               (unwind-protect
                    (progn
                      (sb-posix:chmod output #o500)
                      (let ((*error-output* (make-string-output-stream)))
                        (handler-case (load-quietly (merge-pathnames "evaluate.lisp" root))
                          (error (caught) (setf condition caught))))
                      (assert (typep condition 'sb-posix:syscall-error))
                      ;; Creating the missing lock was denied; ENOENT from a read-only
                      ;; fallback must not replace that useful error.
                      (assert (= (sb-posix:syscall-errno condition) sb-posix:eacces)))
                 (sb-posix:chmod output mode))))
        (uiop:delete-directory-tree tree :validate t)))))

(defun test-ranking-invalid-lock-is-reported ()
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (dolist (example '("movielens-ranking/" "mslr-web10k/"))
           (let* ((root (merge-pathnames example tree))
                  (lock-directory (merge-pathnames "output/.ranking.lock/" root))
                  (condition nil))
             (write-evaluate-fixture root '(2f0 1f0 0f0) '(3d0 2d0 1d0))
             (ensure-directories-exist (merge-pathnames ".keep" lock-directory))
             (let ((*error-output* (make-string-output-stream)))
               (handler-case (load-quietly (merge-pathnames "evaluate.lisp" root))
                 (error (caught) (setf condition caught))))
             (assert (typep condition 'sb-posix:syscall-error))
             (assert (= (sb-posix:syscall-errno condition) sb-posix:eisdir))))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-ranking-ties-follow-input-order ()
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (dolist (example '("movielens-ranking/" "mslr-web10k/"))
           (let ((root (merge-pathnames example tree)))
             (write-evaluate-fixture root '(2f0 0f0) '(1d0 1d0))
             (let ((result (load-quietly (merge-pathnames "evaluate.lisp" root))))
               (assert (search "lightgbm: NDCG@10=1.0000" result))
               (assert (search "xgboost: NDCG@10=1.0000" result)))
             (write-evaluate-fixture root '(0f0 2f0) '(1d0 1d0))
             (let ((result (load-quietly (merge-pathnames "evaluate.lisp" root))))
               (assert (search "lightgbm: NDCG@10=0.6309" result))
               (assert (search "xgboost: NDCG@10=0.6309" result)))))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-evaluate-reads-its-own-example ()
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let ((mslr (merge-pathnames "mslr-web10k/" tree))
               (movie (merge-pathnames "movielens-ranking/" tree)))
           ;; Scores in label order give NDCG 1. Reversed, DCG is 1/log2(3) + 3/log2(4)
           ;; = 2.1309 against an ideal 3 + 1/log2(3) = 3.6309, which is 0.5869.
           (write-evaluate-fixture mslr '(2f0 1f0 0f0) '(3d0 2d0 1d0))
           (write-evaluate-fixture movie '(2f0 1f0 0f0) '(1d0 2d0 3d0))
           (assert (search "lightgbm: NDCG@10=1.0000"
                           (load-quietly (merge-pathnames "evaluate.lisp" mslr))))
           ;; This order is the regression: MSLR first, then MovieLens in the same image.
           (assert (search "lightgbm: NDCG@10=0.5869"
                           (load-quietly (merge-pathnames "evaluate.lisp" movie))))
           (assert (search "lightgbm: NDCG@10=1.0000"
                           (load-quietly (merge-pathnames "evaluate.lisp" mslr)))))
      (uiop:delete-directory-tree tree :validate t))))

(defun write-train-fixture (root test-rows)
  "Give ROOT a small three-split ranking problem whose test split has TEST-ROWS rows."
  (let ((sizes (list :train 40 :valid 12 :test test-rows)))
    (loop for (name rows group-size) in `(("train" 40 8) ("valid" 12 6)
                                          ("test" ,test-rows ,test-rows))
          for labels = (loop for i below rows collect (float (mod i 3) 1f0))
          for features = (loop for i below rows
                               append (list (float (mod i 3) 1f0) (float i 1f0)))
          do (write-f32-file (merge-pathnames (format nil "data/~A-x.f32" name) root) features)
             (write-f32-file (merge-pathnames (format nil "data/~A-y.f32" name) root) labels)
             (write-groups (merge-pathnames (format nil "data/~A-groups.lisp" name) root)
                           (loop repeat (/ rows group-size) collect group-size)))
    (apply #'write-sizes (merge-pathnames "data/sizes.lisp" root)
           (append sizes (list :features 2 :generation "00000000000000000000000000000001")))))

(defun count-file-lines (path)
  (with-open-file (in path)
    (loop while (read-line in nil nil) count t)))

(defun test-evaluate-consumes-root ()
  ;; Reenter MovieLens from the actual MSLR evaluator's first ranking callback, on
  ;; the evaluator's own thread. The evaluator reloads RANKING-NDCG, so the hook is
  ;; an encapsulation, which survives redefinition. A FIFO cannot pause the evaluator
  ;; portably instead: SBCL on Darwin never sees end of file on one.
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let ((mslr (merge-pathnames "mslr-web10k/" tree))
               (movie (merge-pathnames "movielens-ranking/" tree))
               (nested nil))
           (write-evaluate-fixture mslr '(2f0 1f0 0f0) '(3d0 2d0 1d0))
           (write-evaluate-fixture movie '(2f0 1f0 0f0) '(1d0 2d0 3d0))
           (sb-int:encapsulate
            'ranking-ndcg 'consumes-root
            (lambda (ndcg &rest args)
              (unless nested
                (setf nested :running)
                (setf nested (load-quietly (merge-pathnames "evaluate.lisp" movie))))
              (apply ndcg args)))
           (unwind-protect
                (let ((outer (load-quietly (merge-pathnames "evaluate.lisp" mslr))))
                  (assert (search "lightgbm: NDCG@10=1.0000" outer))
                  (assert (and (stringp nested)
                               (search "lightgbm: NDCG@10=0.5869" nested))))
             (sb-int:unencapsulate 'ranking-ndcg 'consumes-root)))
      (uiop:delete-directory-tree tree :validate t))))

(defun exercise-ranking-warning-reentry (operation movie-locked)
  ;; A real SBCL redefinition warning interrupts LOAD before its entry form.
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let* ((mslr (merge-pathnames "mslr-web10k/" tree))
                (movie (merge-pathnames "movielens-ranking/" tree))
                (script (ecase operation (:evaluate "evaluate.lisp") (:train "train.lisp")))
                (movie-script (merge-pathnames script movie))
                (mslr-script (merge-pathnames script mslr))
                (parse-warnings 0)
                (triggered nil)
                (nested-result nil)
                (nested-error nil))
           (ecase operation
             (:evaluate
              (write-evaluate-fixture mslr '(2f0 1f0 0f0) '(3d0 2d0 1d0))
              (write-evaluate-fixture movie '(2f0 1f0 0f0) '(1d0 2d0 3d0)))
             (:train
              (write-train-fixture mslr 6)
              (write-train-fixture movie 4)))
           ;; Prime the shared definitions so reloading them signals a warning.
           (load-quietly movie-script)
           (when (eq operation :train)
             (dolist (name '("lightgbm" "xgboost"))
               (delete-file (merge-pathnames (format nil "output/~A-test.csv" name) movie))))
           (flet ((run-outer ()
                    (handler-bind
                        ((sb-kernel:redefinition-warning
                           (lambda (condition)
                             (let ((message (string-upcase (princ-to-string condition))))
                               (when (search "PARSE-NUMBER" message) (incf parse-warnings))
                               (when (and (not triggered)
                                          (or (and (search "PARSE-NUMBER" message)
                                                   (= parse-warnings 2))
                                              (search (ecase operation
                                                        (:evaluate "EVALUATE-RANKING")
                                                        (:train "LOAD-F32"))
                                                      message)))
                                 (setf triggered t)
                                 (let ((*error-output* (make-string-output-stream)))
                                   (handler-case (setf nested-result (load-quietly movie-script))
                                     (error (caught) (setf nested-error caught))))))
                             (muffle-warning condition))))
                      (load-quietly mslr-script))))
             (let ((outer-result (if movie-locked
                                     (call-with-ranking-lock movie #'run-outer)
                                     (run-outer))))
               (assert triggered)
               (if movie-locked
                   (progn
                     (assert nested-error)
                     (assert (search "busy" (princ-to-string nested-error)
                                     :test #'char-equal)))
                   (progn
                     (assert (not nested-error))
                     (assert nested-result)))
               (ecase operation
                 (:evaluate
                  (assert (search "lightgbm: NDCG@10=1.0000" outer-result))
                  (unless movie-locked
                    (assert (search "lightgbm: NDCG@10=0.5869" nested-result))))
                 (:train
                  (dolist (name '("lightgbm" "xgboost"))
                    (let ((mslr-score (merge-pathnames (format nil "output/~A-test.csv" name) mslr))
                          (movie-score
                            (merge-pathnames (format nil "output/~A-test.csv" name) movie)))
                      (assert (= 6 (count-file-lines mslr-score)))
                      (if movie-locked
                          (assert (not (probe-file movie-score)))
                          (assert (= 4 (count-file-lines movie-score))))))))
               ;; The outer example also must take its own lock.
               (assert (call-with-ranking-lock
                        mslr (lambda () (ranking-test-load-fails-p mslr-script)))))))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-ranking-warning-reentry ()
  (dolist (operation '(:evaluate :train))
    (exercise-ranking-warning-reentry operation nil)
    (exercise-ranking-warning-reentry operation t)))

(defun test-ranking-scripts-before-example-package ()
  ;; Each child starts with the backend systems but without the examples package.
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let* ((movie (merge-pathnames "movielens-ranking/" tree))
                (body (merge-pathnames "fresh-ranking-check.lisp" tree))
                (evaluate (merge-pathnames "evaluate.lisp" movie))
                (train (merge-pathnames "train.lisp" movie)))
           (write-evaluate-fixture movie '(2f0 1f0 0f0) '(1d0 2d0 3d0))
           (with-open-file (out body :direction :output :if-exists :supersede)
             (format out "(assert (not (find-package ~S)))~%" "CL-GBDT-EXAMPLES")
             (format out "(assert (search ~S (with-output-to-string (*standard-output*) ~
                          (load #P~S))))~%"
                     "lightgbm: NDCG@10=0.5869" (namestring evaluate)))
           (cl-gbdt-example-test-process:run-test-in-child
            body :systems '(:cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified)
            :output nil :error-output nil)
           (write-train-fixture movie 4)
           (with-open-file (out body :direction :output :if-exists :supersede)
             (format out "(assert (not (find-package ~S)))~%" "CL-GBDT-EXAMPLES")
             (format out "(load #P~S)~%" (namestring train)))
           (cl-gbdt-example-test-process:run-test-in-child
            body :systems '(:cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified)
            :output nil :error-output nil)
           (dolist (backend '("lightgbm" "xgboost"))
             (assert (= 4 (count-file-lines
                           (merge-pathnames (format nil "output/~A-test.csv" backend) movie))))))
      (uiop:delete-directory-tree tree :validate t))))

(defun ranking-test-directory-designator (directory kind trailing-slash)
  (let* ((with-slash (namestring (uiop:ensure-directory-pathname directory)))
         (name (if trailing-slash with-slash (subseq with-slash 0 (1- (length with-slash))))))
    (ecase kind (:string name) (:pathname (pathname name)))))

(defun exercise-ranking-evaluation-directory-designator (kind trailing-slash)
  ;; The shared entry points accept any designator of the example directory.
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let* ((target (merge-pathnames "mslr-web10k/" tree))
                (root (ranking-test-directory-designator target kind trailing-slash)))
           (write-evaluate-fixture target '(2f0 1f0 0f0) '(3d0 2d0 1d0))
           (load (merge-pathnames "movielens-ranking/evaluate-core.lisp" tree))
           (assert (search "lightgbm: NDCG@10=1.0000"
                           (with-output-to-string (*standard-output*)
                             (call-with-ranking-evaluation
                              root (lambda () (evaluate-ranking target))))))
           (assert (probe-file (merge-pathnames "output/.ranking.lock" target)))
           (assert (not (probe-file (merge-pathnames "output/" tree))))
           (assert (call-with-ranking-lock
                    target (lambda ()
                             (ranking-test-fails-p
                              (lambda () (call-with-ranking-evaluation root (constantly nil))))))))
      (uiop:delete-directory-tree tree :validate t))))

(defun exercise-ranking-training-directory-designator (kind trailing-slash)
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let* ((target (merge-pathnames "mslr-web10k/" tree))
                (root (ranking-test-directory-designator target kind trailing-slash))
                (train (lambda (stage)
                         (dolist (backend '("lightgbm" "xgboost"))
                           (with-open-file (out (merge-pathnames
                                                 (format nil "~A-test.csv" backend) stage)
                                                :direction :output)
                             (format out "1.0~%2.0~%"))))))
           (write-train-fixture target 6)
           (call-with-ranking-training root train)
           (dolist (backend '("lightgbm" "xgboost"))
             (assert (= 2 (count-file-lines
                           (merge-pathnames (format nil "output/~A-test.csv" backend) target)))))
           (assert (probe-file (merge-pathnames "output/predictions.ready" target)))
           (assert (not (probe-file (merge-pathnames "output/" tree))))
           (let* ((ready (merge-pathnames "output/predictions.ready" target))
                  (before (uiop:read-file-string ready)))
             (assert (call-with-ranking-lock
                      target (lambda ()
                               (ranking-test-fails-p
                                (lambda () (call-with-ranking-training root train))))))
             (assert (string= before (uiop:read-file-string ready)))))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-ranking-directory-designators ()
  (let ((failures nil))
    (dolist (kind '(:string :pathname))
      (dolist (trailing-slash '(nil t))
        (dolist (operation '(:evaluate :train))
          (handler-case
              (ecase operation
                (:evaluate (exercise-ranking-evaluation-directory-designator kind trailing-slash))
                (:train (exercise-ranking-training-directory-designator kind trailing-slash)))
            (error (condition)
              (push (list operation kind trailing-slash (princ-to-string condition)) failures))))))
    (when failures (error "Directory designators failed: ~S" (nreverse failures)))))

(defun test-train-writes-to-its-own-example ()
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let* ((mslr (merge-pathnames "mslr-web10k/" tree))
                (movie (merge-pathnames "movielens-ranking/" tree))
                (mslr-scores (merge-pathnames "output/lightgbm-test.csv" mslr))
                (movie-scores (merge-pathnames "output/lightgbm-test.csv" movie)))
           (write-train-fixture mslr 6)
           (write-train-fixture movie 4)
           ;; Start MovieLens while MSLR is predicting with its first backend.
           ;; Once it returns, the suspended MSLR run must keep its original root.
           (let ((original (symbol-function 'cl-gbdt:predict))
                 (nested nil))
             (unwind-protect
                  (progn
                    (setf (symbol-function 'cl-gbdt:predict)
                          (lambda (&rest args)
                            (unless nested
                              (setf nested t)
                              (load-quietly (merge-pathnames "train.lisp" movie)))
                            (apply original args)))
                    (load-quietly (merge-pathnames "train.lisp" mslr))
                    (assert nested))
               (setf (symbol-function 'cl-gbdt:predict) original)))
           (assert (= (count-file-lines movie-scores) 4))
           (assert (= (count-file-lines mslr-scores) 6))
           (assert (= (count-file-lines (merge-pathnames "output/xgboost-test.csv" movie)) 4))
           (assert (= (count-file-lines (merge-pathnames "output/xgboost-test.csv" mslr)) 6)))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-movie-preparation-orders-by-time ()
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let ((root (merge-pathnames "movielens-ranking/" tree)))
           (write-movie-raw-fixture root)
           (load-quietly (merge-pathnames "prepare.lisp" root))
           (with-open-file (in (merge-pathnames "data/test-meta.csv" root))
             (read-csv-row in)
             (let ((previous-user nil) (previous-time nil) (previous-item nil) (rows 0))
               (loop for row = (read-csv-row in) while row
                     for user = (parse-integer (first row))
                     for item = (parse-integer (second row))
                     ;; The fixture assigns each pair of adjacent items the same time.
                     for time = (floor (1- item) 2)
                     do (when (eql user previous-user)
                          (assert (<= previous-time time))
                          (when (= previous-time time) (assert (< previous-item item))))
                        (setf previous-user user previous-time time previous-item item)
                        (incf rows))
               (assert (plusp rows)))))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-movie-interrupted-preparation ()
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let* ((root (merge-pathnames "movielens-ranking/" tree))
                (sizes (merge-pathnames "data/sizes.lisp" root))
                (light (merge-pathnames "output/lightgbm-test.csv" root))
                (xgb (merge-pathnames "output/xgboost-test.csv" root)))
           (write-movie-raw-fixture root)
           (load-quietly (merge-pathnames "prepare.lisp" root))
           (load-quietly (merge-pathnames "train.lisp" root))
           (let ((old-light (uiop:read-file-string light))
                 (old-xgb (uiop:read-file-string xgb))
                 (original (symbol-function 'write-movie-partition))
                 (calls 0)
                 (failed nil))
             (write-movie-raw-fixture root t)
             (unwind-protect
                  (progn
                    (setf (symbol-function 'write-movie-partition)
                          (lambda (&rest args)
                            (incf calls)
                            (when (= calls 2) (error "Simulated preparation failure"))
                            (apply original args)))
                    (handler-case
                        (let ((*movie-root* root)
                              (*standard-output* (make-string-output-stream)))
                          (prepare-movie))
                      (error () (setf failed t)))
                    (assert failed)
                    (assert (= calls 2))
                    (assert (not (probe-file sizes)))
                    (assert (string= old-light (uiop:read-file-string light)))
                    (assert (string= old-xgb (uiop:read-file-string xgb)))
                    (assert (ranking-test-load-fails-p (merge-pathnames "evaluate.lisp" root))))
               (setf (symbol-function 'write-movie-partition) original))))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-movie-metadata-failure ()
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let* ((root (merge-pathnames "movielens-ranking/" tree))
                (sizes (merge-pathnames "data/sizes.lisp" root)))
           (write-movie-raw-fixture root)
           (load-quietly (merge-pathnames "prepare.lisp" root))
           (load-quietly (merge-pathnames "train.lisp" root))
           (let ((original (symbol-function 'write-sizes))
                 (failed nil))
             (unwind-protect
                  (progn
                    (setf (symbol-function 'write-sizes)
                          (lambda (path &rest values)
                            (declare (ignore values))
                            (with-open-file (out path :direction :output :if-exists :supersede)
                              (write-string "(:train" out))
                            (error "Simulated metadata write failure")))
                    (handler-case
                        (let ((*movie-root* root)
                              (*standard-output* (make-string-output-stream)))
                          (prepare-movie))
                      (error () (setf failed t)))
                    (assert failed)
                    (assert (not (probe-file sizes)))
                    (assert (null (directory (merge-pathnames "data/.movielens-sizes-*" root))))
                    (assert (ranking-test-load-fails-p (merge-pathnames "evaluate.lisp" root))))
               (setf (symbol-function 'write-sizes) original))))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-movie-training-failure ()
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let* ((root (merge-pathnames "movielens-ranking/" tree))
                (light (merge-pathnames "output/lightgbm-test.csv" root))
                (xgb (merge-pathnames "output/xgboost-test.csv" root)))
           (write-movie-raw-fixture root)
           (load-quietly (merge-pathnames "prepare.lisp" root))
           (load-quietly (merge-pathnames "train.lisp" root))
           (with-open-file (out light :direction :output :if-exists :supersede)
             (dotimes (i 32) (format out "~D.0~%" i)))
           (let ((old-light (uiop:read-file-string light))
                 (old-xgb (uiop:read-file-string xgb))
                 (original (symbol-function 'cl-gbdt:predict))
                 (calls 0))
             (unwind-protect
                  (progn
                    (setf (symbol-function 'cl-gbdt:predict)
                          (lambda (&rest args)
                            (incf calls)
                            (when (= calls 2) (error "Simulated backend failure"))
                            (apply original args)))
                    (assert (ranking-test-load-fails-p (merge-pathnames "train.lisp" root)))
                    (assert (= calls 2))
                    (assert (string= old-light (uiop:read-file-string light)))
                    (assert (string= old-xgb (uiop:read-file-string xgb)))
                    (assert (ranking-test-load-fails-p (merge-pathnames "evaluate.lisp" root))))
               (setf (symbol-function 'cl-gbdt:predict) original))))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-movie-prepare-during-training ()
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let ((root (merge-pathnames "movielens-ranking/" tree))
               (calls 0)
               (prepared nil)
               (overlap-train-failed nil)
               (overlap-eval-failed nil)
               (original (symbol-function 'cl-gbdt:predict)))
           (write-movie-raw-fixture root)
           (load-quietly (merge-pathnames "prepare.lisp" root))
           (load-quietly (merge-pathnames "train.lisp" root))
           (write-movie-raw-fixture root t)
           (unwind-protect
                (progn
                  (setf (symbol-function 'cl-gbdt:predict)
                        (lambda (&rest args)
                          (incf calls)
                          (when (= calls 2)
                            (setf prepared
                                  (not (ranking-test-load-fails-p
                                        (merge-pathnames "prepare.lisp" root)))
                                  overlap-train-failed
                                  (ranking-test-load-fails-p (merge-pathnames "train.lisp" root))
                                  overlap-eval-failed
                                  (ranking-test-load-fails-p
                                   (merge-pathnames "evaluate.lisp" root))))
                          (apply original args)))
                  (let ((trained (not (ranking-test-load-fails-p
                                       (merge-pathnames "train.lisp" root)))))
                    (assert (= calls 2))
                    (assert overlap-train-failed)
                    (assert overlap-eval-failed)
                    (if prepared
                        (assert (or (not trained)
                                    (ranking-test-load-fails-p
                                     (merge-pathnames "evaluate.lisp" root))))
                        (assert trained))))
             (setf (symbol-function 'cl-gbdt:predict) original)))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-movie-evaluation-generation-change ()
  ;; Change the metadata as the real evaluator opens its second score file. See
  ;; TEST-EVALUATE-CONSUMES-ROOT for why this is an encapsulation and not a FIFO.
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let* ((root (merge-pathnames "movielens-ranking/" tree))
                (sizes (merge-pathnames "data/sizes.lisp" root))
                (captured (make-string-output-stream))
                (changed nil)
                (failed nil))
           (write-evaluate-fixture root '(2f0 1f0 0f0) '(3d0 2d0 1d0))
           (let ((sizes-text (uiop:read-file-string sizes)))
             (sb-int:encapsulate
              'read-scores 'movie-generation-change
              (lambda (read path)
                (when (and (not changed) (string= "xgboost-test" (pathname-name path)))
                  (setf changed t)
                  (let ((metadata (with-open-file (in sizes)
                                    (let ((*read-eval* nil)) (read in)))))
                    (setf (getf metadata :generation) "00000000000000000000000000000002")
                    (apply #'write-sizes sizes metadata)))
                (funcall read path)))
             (unwind-protect
                  (progn
                    (handler-case
                        (let ((*standard-output* captured)
                              (*error-output* (make-string-output-stream)))
                          (load (merge-pathnames "evaluate.lisp" root)))
                      (error () (setf failed t)))
                    (assert changed)
                    (assert failed)
                    (assert (not (search "NDCG@10" (get-output-stream-string captured)))))
               (sb-int:unencapsulate 'read-scores 'movie-generation-change)
               (with-open-file (out sizes :direction :output :if-exists :supersede)
                 (write-string sizes-text out)))))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-movie-reprepare-invalidates ()
  (let ((tree (make-ranking-test-tree)))
    (unwind-protect
         (let* ((root (merge-pathnames "movielens-ranking/" tree))
                (light (merge-pathnames "output/lightgbm-test.csv" root))
                (xgb (merge-pathnames "output/xgboost-test.csv" root))
                (ready (merge-pathnames "output/predictions.ready" root)))
           (write-movie-raw-fixture root)
           (load-quietly (merge-pathnames "prepare.lisp" root))
           (load-quietly (merge-pathnames "train.lisp" root))
           (load-quietly (merge-pathnames "evaluate.lisp" root))
           (let ((old-light (uiop:read-file-string light))
                 (old-xgb (uiop:read-file-string xgb))
                 (old-ready (and (probe-file ready) (uiop:read-file-string ready))))
             (write-movie-raw-fixture root t)
             (load-quietly (merge-pathnames "prepare.lisp" root))
             (assert (string= old-light (uiop:read-file-string light)))
             (assert (string= old-xgb (uiop:read-file-string xgb)))
             (assert (ranking-test-load-fails-p (merge-pathnames "evaluate.lisp" root)))
             (assert old-ready)
             (with-open-file (out ready :direction :output :if-exists :supersede
                                        :if-does-not-exist :create)
               (write-string old-ready out))
             (assert (ranking-test-load-fails-p (merge-pathnames "evaluate.lisp" root)))
             (load-quietly (merge-pathnames "train.lisp" root))
             (load-quietly (merge-pathnames "evaluate.lisp" root))))
      (uiop:delete-directory-tree tree :validate t))))

(defun test-movie-preparation-reentry ()
  (let ((tree-a (make-ranking-test-tree))
        (tree-b (make-ranking-test-tree)))
    (unwind-protect
         (let ((a (merge-pathnames "movielens-ranking/" tree-a))
               (b (merge-pathnames "movielens-ranking/" tree-b)))
           (write-movie-raw-fixture a)
           (write-movie-raw-fixture b)
           (load-quietly (merge-pathnames "prepare.lisp" a))
           (load-quietly (merge-pathnames "prepare.lisp" b))
           (let ((old-a (read-f32 (merge-pathnames "data/train-y.f32" a)))
                 (old-b (read-f32 (merge-pathnames "data/train-y.f32" b)))
                 (original (symbol-function 'read-movie-users))
                 (nested nil))
             (write-movie-raw-fixture a t)
             (unwind-protect
                  (progn
                    (setf (symbol-function 'read-movie-users)
                          (lambda (path)
                            (unless nested
                              (setf nested t)
                              (load-quietly (merge-pathnames "prepare.lisp" b)))
                            (funcall original path)))
                    (let ((*movie-root* a)
                          (*standard-output* (make-string-output-stream)))
                      (prepare-movie))
                    (assert nested)
                    (assert (not (equalp old-a (read-f32 (merge-pathnames "data/train-y.f32" a)))))
                    (assert (equalp old-b (read-f32 (merge-pathnames "data/train-y.f32" b)))))
               (setf (symbol-function 'read-movie-users) original))))
      (uiop:delete-directory-tree tree-a :validate t)
      (uiop:delete-directory-tree tree-b :validate t))))

(defun run-ranking-root-tests ()
  (test-ranking-directory-designators)
  (test-evaluate-reads-its-own-example)
  (test-ranking-evaluate-read-only-output)
  (test-ranking-missing-lock-read-only-directory)
  (test-ranking-invalid-lock-is-reported)
  (test-ranking-ties-follow-input-order)
  (test-evaluate-consumes-root)
  (test-ranking-warning-reentry)
  (test-ranking-scripts-before-example-package)
  (test-train-writes-to-its-own-example)
  (test-ranking-relevance-validation)
  (test-movie-preparation-orders-by-time)
  (test-movie-reprepare-invalidates)
  (test-movie-evaluation-generation-change)
  (test-movie-interrupted-preparation)
  (test-movie-metadata-failure)
  (test-movie-training-failure)
  (test-movie-prepare-during-training)
  (test-movie-preparation-reentry)
  (format t "Ranking example directory tests passed.~%"))

(run-ranking-root-tests)
