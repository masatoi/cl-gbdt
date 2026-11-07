(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests-utility-safety.lisp to execute this test in an isolated process."))

(require :sb-posix)

;;;; The utility tests must not replace a user's existing generated-data file.

(require :asdf)
(defparameter *utility-safety-examples-root* (uiop:pathname-directory-pathname *load-truename*))

(defun test-utility-file-safety ()
  (let* ((tree (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-utility-safety-XXXXXX"
                                              (uiop:temporary-directory))))))
         (existing (merge-pathnames "toy-data/data/test-f32.bin" tree))
         (original #(42 99 7)))
    (unwind-protect
         (progn
           (dolist (file '("common.lisp" "tests.lisp" "test-process.lisp"
                           "utility-test-body.lisp"))
             (let ((target (merge-pathnames file tree)))
               (ensure-directories-exist target)
               (uiop:copy-file (merge-pathnames file *utility-safety-examples-root*) target)))
           (ensure-directories-exist existing)
           (with-open-file (out existing :direction :output :if-exists :supersede
                                         :element-type '(unsigned-byte 8))
             (write-sequence original out))
           (let ((*standard-output* (make-string-output-stream)))
             (load (merge-pathnames "tests.lisp" tree)))
           (assert (probe-file existing))
           (with-open-file (in existing :element-type '(unsigned-byte 8))
             (let ((bytes (make-array (file-length in) :element-type '(unsigned-byte 8))))
               (assert (= (read-sequence bytes in) (length bytes)))
               (assert (equalp bytes original)))))
      (when (probe-file tree) (uiop:delete-directory-tree tree :validate t))))
  (format t "Utility file safety test passed.~%"))

(test-utility-file-safety)
