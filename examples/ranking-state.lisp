(in-package #:cl-user)

;;;; Shared state for one prepared ranking directory.

(require :sb-posix)

(load (merge-pathnames "lifecycle.lisp" *load-truename*))

(in-package #:cl-gbdt-examples)

;; Both ranking examples keep their state with the shared code in lifecycle.lisp, under
;; their own file names: output/.ranking.lock, data/sizes.lisp's :GENERATION entry and
;; output/predictions.ready.

(defparameter *ranking-score-names* '("lightgbm-test.csv" "xgboost-test.csv")
  "The prediction files one ranking training publishes together.")

(defun ranking-layout (name prefix)
  "Return the state layout of the ranking example NAME, whose private files use PREFIX.
The lock is output/.ranking.lock, the prepared generation is the :GENERATION entry of
data/sizes.lisp and the completion marker is output/predictions.ready."
  (cl-gbdt-example-lifecycle:make-layout
   :name name :prefix prefix :lock "output/.ranking.lock"
   :markers '((:prepared "data/sizes.lisp" :generation)
              (:standard "output/predictions.ready"))))

(defun call-with-ranking-lock (root function &key (name "MSLR") (prefix "mslr") read-only-ok)
  "Call FUNCTION while holding the lock of the ranking directory ROOT.
READ-ONLY-OK accepts an existing lock file that cannot be opened for writing."
  (cl-gbdt-example-lifecycle:call-with-lock
   root function :layout (ranking-layout name prefix) :read-only-ok read-only-ok))

(defun call-with-ranking-preparation (root function &key (name "MSLR") (prefix "mslr"))
  "Prepare the ranking directory ROOT under its lock, and return FUNCTION's value.
FUNCTION writes the split files and returns the property list for data/sizes.lisp,
which is published with a new :GENERATION entry only after FUNCTION returns. Earlier
predictions are kept but invalidated first."
  (let ((root (cl-gbdt-example-lifecycle:example-root root)))
    (cl-gbdt-example-lifecycle:call-with-preparation
     root
     (lambda ()
       (let ((sizes (funcall function))
             (generation cl-gbdt-example-lifecycle:*prepared-generation*))
         (cl-gbdt-example-lifecycle:publish-file
          (merge-pathnames "data/sizes.lisp" root) (format nil ".~A-sizes-XXXXXX" prefix)
          (lambda (path)
            (apply #'write-sizes path (append sizes (list :generation generation)))))
         sizes))
     :layout (ranking-layout name prefix))))

(defun call-with-ranking-training (root function &key (name "MSLR") (prefix "mslr"))
  "Train the ranking directory ROOT against its current preparation, under its lock.
FUNCTION is called with a private directory for both backends' score files. They are
published together with output/predictions.ready only if FUNCTION returns, and the
previous files are restored if that publication fails."
  (cl-gbdt-example-lifecycle:call-with-training
   root :standard *ranking-score-names*
   (lambda () (funcall function cl-gbdt-example-lifecycle:*training-output-directory*))
   :layout (ranking-layout name prefix)))

(defun call-with-ranking-evaluation (root function &key (name "MSLR") (prefix "mslr"))
  "Call FUNCTION to evaluate the ranking directory ROOT under its lock.
Its output is printed only if the data and predictions are still current when it returns."
  (cl-gbdt-example-lifecycle:call-with-evaluation
   root function :layout (ranking-layout name prefix)))

;; The historical mslr-named helpers below are shared by both ranking examples.
