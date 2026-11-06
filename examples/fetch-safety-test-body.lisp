(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests-fetch-safety.lisp to execute this test in an isolated process."))

(require :sb-posix)

;; Reproduce a collision with the old predictable test directory without touching
;; the checkout's data directory or any directory not created by this test.
(let* ((root (uiop:ensure-directory-pathname
              (sb-posix:mkdtemp
               (namestring (merge-pathnames "cl-gbdt-fetch-safety-XXXXXX"
                                            (uiop:temporary-directory))))))
       (state (make-random-state t))
       (expected (random most-positive-fixnum (make-random-state state)))
       (sentinel (merge-pathnames
                  (format nil "toy-data/data/fetch-tests/~36R/hello.bz2" expected)
                  root))
       (*random-state* state))
  (unwind-protect
       (progn
         (dolist (file '("common.lisp" "fetch-data.lisp" "tests-fetch.lisp"
                         "test-process.lisp" "fetch-test-body.lisp"))
           (let ((destination (merge-pathnames file root)))
             (uiop:copy-file (merge-pathnames file *load-truename*) destination)))
         (ensure-directories-exist sentinel)
         (with-open-file (out sentinel :direction :output :if-exists :error
                                       :element-type '(unsigned-byte 8))
           (write-sequence #(42 99 7) out))
         ;; Loading the copied fetch-data.lisp defines *EXAMPLES-ROOT*. Bind it
         ;; dynamically so the caller's checkout path survives this scratch run.
         (unless (find-package "CL-GBDT-EXAMPLES")
           (load (merge-pathnames "common.lisp" *load-truename*)))
         (let* ((symbol (intern "*EXAMPLES-ROOT*" "CL-GBDT-EXAMPLES"))
                (was-bound (boundp symbol))
                (previous (and was-bound (symbol-value symbol))))
           (progv (list symbol) (list previous)
             (load (merge-pathnames "tests-fetch.lisp" root)))
           (assert (eql was-bound (boundp symbol)))
           (when was-bound (assert (equal previous (symbol-value symbol)))))
         (assert (probe-file sentinel))
         (with-open-file (in sentinel :element-type '(unsigned-byte 8))
           (let ((bytes (make-array 3 :element-type '(unsigned-byte 8))))
             (assert (= (read-sequence bytes in) 3))
             (assert (equalp bytes #(42 99 7))))))
    (uiop:delete-directory-tree root :validate t)))

(format t "Fetch test data safety test passed.~%")
