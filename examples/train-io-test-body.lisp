(in-package #:cl-user)

(unless (let* ((package (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS"))
               (symbol (and package (find-symbol "*CHILD*" package))))
          (and symbol (boundp symbol) (symbol-value symbol)))
  (error "Run examples/tests-train-io.lisp to execute this test in an isolated process."))

(require :sb-posix)

;;;; The trainers' shared binary reader and score writer.

(load (merge-pathnames "train-io.lisp" *load-truename*))

(defun train-io-test-error-p (thunk message)
  "True when THUNK signals an error whose report contains MESSAGE."
  (handler-case (progn (funcall thunk) nil)
    (error (condition) (search message (princ-to-string condition)))))

(defun train-io-test-write (path writer values)
  (with-open-file (out path :direction :output :if-exists :supersede
                            :element-type '(unsigned-byte 8))
    (dolist (value values) (funcall writer out value))))

(defun test-train-io ()
  (let* ((base (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (sb-ext:native-namestring
                  (merge-pathnames "cl-gbdt-train-io-XXXXXX" (uiop:temporary-directory))))))
         ;; A checkout under such a directory is legal; Lisp namestrings escape these.
         (tricky (sb-ext:parse-native-namestring
                  (concatenate 'string (sb-ext:native-namestring base) "repo[1]*\\x/")))
         (floats (merge-pathnames "data/m.f32" tricky))
         (integers (merge-pathnames "data/i.i32" tricky))
         (scores (merge-pathnames "output/s.csv" tricky))
         (values '(1.5 -2.25 3.0 0.125 4.0 -0.5)))
    (unwind-protect
         (progn
           (ensure-directories-exist floats)
           (ensure-directories-exist scores)
           (train-io-test-write floats #'cl-gbdt-examples:write-f32 values)
           (train-io-test-write integers #'cl-gbdt-examples:write-i32 '(0 -1 2147483647))
           ;; Values come back in row-major order, as single floats.
           (let ((matrix (cl-gbdt-examples:load-f32 floats '(2 3))))
             (assert (equal (array-dimensions matrix) '(2 3)))
             (assert (eq (array-element-type matrix) 'single-float))
             (assert (equal (loop for i below 6 collect (row-major-aref matrix i))
                            (mapcar (lambda (v) (coerce v 'single-float)) values))))
           (let ((vector (cl-gbdt-examples:load-f32 floats 6)))
             (assert (= (aref vector 5) -0.5)))
           (assert (equalp (cl-gbdt-examples:load-i32 integers 3) #(0 -1 2147483647)))
           ;; A file whose size does not match the requested shape is refused either way.
           (assert (train-io-test-error-p
                    (lambda () (cl-gbdt-examples:load-f32 floats '(2 4)))
                    "Short read"))
           (assert (train-io-test-error-p
                    (lambda () (cl-gbdt-examples:load-f32 floats '(5)))
                    "holds more than 5 floats"))
           (assert (train-io-test-error-p
                    (lambda () (cl-gbdt-examples:load-i32 integers 2))
                    "holds more than 2 integers"))
           (assert (train-io-test-error-p
                    (lambda () (cl-gbdt-examples:load-f32 (merge-pathnames "absent.f32" tricky)
                                                          '(1)))
                    "Cannot open"))
           ;; Scores are written one per line, and only one-column matrices are accepted.
           (let ((predictions (make-array '(2 1) :element-type 'double-float
                                                 :initial-contents '((0.25d0) (0.75d0)))))
             (cl-gbdt-examples:write-scores scores predictions :rows 2)
             (assert (string= (uiop:read-file-string scores)
                              (format nil "~,15F~%~,15F~%" 0.25d0 0.75d0)))
             (assert (train-io-test-error-p
                      (lambda () (cl-gbdt-examples:write-scores scores predictions :rows 3))
                      "Unexpected prediction dimensions")))
           (assert (train-io-test-error-p
                    (lambda ()
                      (cl-gbdt-examples:write-scores
                       scores (make-array '(2 2) :element-type 'double-float
                                                 :initial-element 0d0)))
                    "Unexpected prediction dimensions"))
           (assert (string= (uiop:read-file-string scores)
                            (format nil "~,15F~%~,15F~%" 0.25d0 0.75d0))))
      (uiop:delete-directory-tree base :validate t)))
  (format t "Train I/O test passed.~%"))

(test-train-io)
