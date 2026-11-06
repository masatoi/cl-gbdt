(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests-test-process.lisp to execute this test in an isolated process."))

(require :sb-posix)
(load (merge-pathnames "fetch-data.lisp" *load-truename*))

(defun test-example-process-bootstrap-environment ()
  "An explicit bootstrap environment variable reaches nested test processes."
  (let ((bootstrap cl-gbdt-example-test-process:*asdf-bootstrap*)
        (previous (sb-posix:getenv "CL_GBDT_ASDF_BOOTSTRAP"))
        (cl-gbdt-example-test-process:*asdf-bootstrap* nil)
        (body (merge-pathnames "utility-test-body.lisp" *load-truename*)))
    (when bootstrap
      (unwind-protect
           (progn
             (sb-posix:setenv "CL_GBDT_ASDF_BOOTSTRAP" (namestring (pathname bootstrap)) 1)
             (multiple-value-bind (out err status)
                 (cl-gbdt-example-test-process:run-test-in-child
                  body :systems '(:cffi) :output nil :error-output nil)
               (declare (ignore err))
               (assert (zerop status))
               (assert (search "Example utility tests passed." out))))
        (if previous
            (sb-posix:setenv "CL_GBDT_ASDF_BOOTSTRAP" previous 1)
            (sb-posix:unsetenv "CL_GBDT_ASDF_BOOTSTRAP"))))))

(test-example-process-bootstrap-environment)

(defun test-example-process-without-user-init ()
  "A child runs when no user init file loads Quicklisp for it, as on a fresh CI runner.
Roswell's Quicklisp has a local-init file that needs Roswell's own package, so a child of a
Roswell image must load Roswell's init.lisp before Quicklisp."
  (let* ((home (sb-posix:getenv "HOME"))
         (empty (uiop:ensure-directory-pathname
                 (format nil "~Acl-gbdt-empty-home-~36R"
                         (uiop:native-namestring (uiop:temporary-directory))
                         (random most-positive-fixnum (make-random-state t)))))
         (body (merge-pathnames "body.lisp" empty)))
    (ensure-directories-exist body)
    (with-open-file (out body :direction :output)
      (write-line "(format t \"child without user init ran~%\")" out))
    (unwind-protect
         (progn
           ;; SBCL finds the user init file through HOME; an empty HOME has none.
           (sb-posix:setenv "HOME" (uiop:native-namestring empty) 1)
           (multiple-value-bind (out err status)
               (cl-gbdt-example-test-process:run-test-in-child
                body :output nil :error-output nil :ignore-error-status t)
             (unless (and (zerop status) (search "child without user init ran" out))
               (error "A child without a user init file failed (status ~D):~%~A" status err))))
      (if home (sb-posix:setenv "HOME" home 1) (sb-posix:unsetenv "HOME"))
      (uiop:delete-directory-tree empty :validate t :if-does-not-exist :ignore))))

(test-example-process-without-user-init)

(defun test-example-process-local-system ()
  "A parent-registered local dependency remains discoverable in the child."
  (let* ((root (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-local-system-XXXXXX"
                                              (uiop:temporary-directory))))))
         (name "cl-gbdt-isolated-local-fixture")
         (asd (merge-pathnames "cl-gbdt-isolated-local-fixture.asd" root))
         (body (merge-pathnames "utility-test-body.lisp" *load-truename*)))
    (unwind-protect
         (progn
           (with-open-file (out asd :direction :output :if-exists :error)
             (write-line "(asdf:defsystem \"cl-gbdt-isolated-local-fixture\" :components ())"
                         out))
           (asdf:load-asd asd)
           (asdf:load-system name)
           (multiple-value-bind (out err status)
               (cl-gbdt-example-test-process:run-test-in-child
                body :systems '(:cffi :cl-gbdt-isolated-local-fixture)
                :output nil :error-output nil :ignore-error-status t)
             (assert (zerop status) () "Local system was unavailable in child: ~A" err)
             (assert (search "Example utility tests passed." out))))
      (asdf:clear-system name)
      (uiop:delete-directory-tree root :validate t))))

(test-example-process-local-system)

(defun test-example-process-central-registry ()
  "Direct, relative and standard expression registry entries reach the child."
  (let* ((root (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-central-system-XXXXXX"
                                              (uiop:temporary-directory))))))
         (local (merge-pathnames "local/" root))
         (name "cl-gbdt-central-local-fixture")
         (asd (merge-pathnames "cl-gbdt-central-local-fixture.asd" local))
         (body (merge-pathnames "utility-test-body.lisp" *load-truename*))
         (*default-pathname-defaults* root))
    (unwind-protect
         (progn
           (ensure-directories-exist asd)
           (with-open-file (out asd :direction :output :if-exists :error)
             (write-line "(asdf:defsystem \"cl-gbdt-central-local-fixture\" :components ())"
                         out))
           (dolist (entry (list local #P"local/" `(merge-pathnames "local/" ,root)))
             (let ((asdf:*central-registry* (cons entry asdf:*central-registry*)))
               (assert (not (asdf:registered-system name)))
               (multiple-value-bind (out err status)
                   (cl-gbdt-example-test-process:run-test-in-child
                    body :systems '(:cffi :cl-gbdt-central-local-fixture)
                    :output nil :error-output nil :ignore-error-status t)
                 (assert (zerop status) () "Central system was unavailable in child: ~A" err)
                 (assert (search "Example utility tests passed." out)))
               (assert (not (asdf:registered-system name))))))
      (uiop:delete-directory-tree root :validate t))))

(test-example-process-central-registry)

(defun test-example-process-source-registry ()
  "A local dependency in the parent's source registry is visible to the child."
  (let* ((root (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-source-system-XXXXXX"
                                              (uiop:temporary-directory))))))
         (name "cl-gbdt-source-local-fixture")
         (asd (merge-pathnames "cl-gbdt-source-local-fixture.asd" root))
         (body (merge-pathnames "utility-test-body.lisp" *load-truename*))
         (asdf/source-registry:*source-registry* nil)
         (asdf/source-registry:*source-registry-parameter* nil))
    (unwind-protect
         (progn
           (with-open-file (out asd :direction :output :if-exists :error)
             (write-line "(asdf:defsystem \"cl-gbdt-source-local-fixture\" :components ())"
                         out))
           (asdf:initialize-source-registry
            `(:source-registry (:directory ,(namestring root)) :ignore-inherited-configuration))
           (assert (not (asdf:registered-system name)))
           (multiple-value-bind (out err status)
               (cl-gbdt-example-test-process:run-test-in-child
                body :systems '(:cffi :cl-gbdt-source-local-fixture)
                :output nil :error-output nil :ignore-error-status t)
             (assert (zerop status) () "Source system was unavailable in child: ~A" err)
             (assert (search "Example utility tests passed." out)))
           (assert (not (asdf:registered-system name))))
      (uiop:delete-directory-tree root :validate t))))

(test-example-process-source-registry)

(defun test-example-process-registry-precedence ()
  "An unregistered central-registry system wins over a source-registry duplicate."
  (let* ((root (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-registry-order-XXXXXX"
                                              (uiop:temporary-directory))))))
         (central (merge-pathnames "central/" root))
         (source (merge-pathnames "source/" root))
         (name "cl-gbdt-registry-order-fixture")
         (body (merge-pathnames "check.lisp" root))
         (asdf/source-registry:*source-registry* nil)
         (asdf/source-registry:*source-registry-parameter* nil))
    (unwind-protect
         (progn
           (loop for (directory package) in
                 (list (list central "CENTRAL-FIXTURE-SELECTED")
                       (list source "SOURCE-FIXTURE-SELECTED"))
                 for asd = (merge-pathnames "cl-gbdt-registry-order-fixture.asd" directory)
                 for file = (merge-pathnames "marker.lisp" directory)
                 do (ensure-directories-exist asd)
                    (with-open-file (out asd :direction :output :if-exists :error)
                      (format out "(asdf:defsystem ~S :components ((:file ~S)))~%"
                              name "marker"))
                    (with-open-file (out file :direction :output :if-exists :error)
                      (format out "(defpackage ~S (:use))~%" package)))
           (with-open-file (out body :direction :output :if-exists :error)
             (format out "(unless (find-package ~S) (error ~S))~%"
                     "CENTRAL-FIXTURE-SELECTED" "wrong source"))
           (asdf:initialize-source-registry
            `(:source-registry (:directory ,(namestring source)) :ignore-inherited-configuration))
           (dolist (entry (list central `(if t ,central nil)))
             (let ((asdf:*central-registry* (list entry)))
               (assert (not (asdf:registered-system name)))
               (multiple-value-bind (found system path) (asdf:locate-system name)
                 (declare (ignore system))
                 (assert found)
                 (assert (equal (probe-file path)
                                (probe-file (merge-pathnames
                                             "cl-gbdt-registry-order-fixture.asd" central)))))
               (multiple-value-bind (out err status)
                   (cl-gbdt-example-test-process:run-test-in-child
                    body :systems '(:cl-gbdt-registry-order-fixture)
                    :output nil :error-output nil :ignore-error-status t)
                 (declare (ignore out))
                 (assert (zerop status) () "Child chose wrong source: ~A" err))
               (assert (not (asdf:registered-system name))))))
      (uiop:delete-directory-tree root :validate t))))

(test-example-process-registry-precedence)

(defvar *test-registry-side-effect* nil)

(defun test-example-process-registry-side-effects ()
  "A deferred registry expression must run only in the child."
  (let* ((root (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-registry-effect-XXXXXX"
                                              (uiop:temporary-directory))))))
         (asd (merge-pathnames "cl-gbdt-registry-effect-fixture.asd" root))
         (body (merge-pathnames "utility-test-body.lisp" *load-truename*))
         (*test-registry-side-effect* nil)
         (asdf:*central-registry*
           (list `(progn (setf *test-registry-side-effect* t) ,root))))
    (unwind-protect
         (progn
           (with-open-file (out asd :direction :output :if-exists :error)
             (write-line "(asdf:defsystem \"cl-gbdt-registry-effect-fixture\" :components ())"
                         out))
           (multiple-value-bind (out err status)
               (cl-gbdt-example-test-process:run-test-in-child
                body :systems '(:cffi :cl-gbdt-registry-effect-fixture)
                :output nil :error-output nil :ignore-error-status t)
             (assert (zerop status) () "Deferred registry entry failed: ~A" err)
             (assert (search "Example utility tests passed." out)))
           (assert (null *test-registry-side-effect*)))
      (uiop:delete-directory-tree root :validate t))))

(test-example-process-registry-side-effects)

(defun test-example-process-isolation ()
  "Concurrent child tests must not replace this image's functions, registry or environment."
  (let* ((directory (uiop:pathname-directory-pathname *load-truename*))
         (check (symbol-function 'cl-gbdt-examples::check-sha256))
         (predict (symbol-function 'cl-gbdt:predict))
         (system (asdf:registered-system :lisp-xl))
         (fold (sb-posix:getenv "CL_GBDT_MSLR_FOLD"))
         (fetch-body (merge-pathnames "fetch-test-body.lisp" directory))
         (ranking-body (merge-pathnames "mslr-test-body.lisp" directory))
         (fetch-systems '(:dexador :zip :semz.decompress :archive :ironclad))
         (ranking-systems '(:cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified))
         (threads
           (mapcar (lambda (spec)
                     (sb-thread:make-thread
                      (lambda ()
                        (handler-case
                            (progn
                              (cl-gbdt-example-test-process:run-test-in-child
                               (car spec) :systems (cdr spec) :output nil :error-output nil)
                              :passed)
                          (error (condition) condition)))))
                   (list (cons fetch-body fetch-systems) (cons ranking-body ranking-systems)))))
    (unwind-protect
         (progn
           (sb-ext:with-timeout 180
             (loop while (some #'sb-thread:thread-alive-p threads)
                   do (assert (eq check (symbol-function 'cl-gbdt-examples::check-sha256)))
                      (assert (eq predict (symbol-function 'cl-gbdt:predict)))
                      (assert (eq system (asdf:registered-system :lisp-xl)))
                      (assert (equal fold (sb-posix:getenv "CL_GBDT_MSLR_FOLD")))
                      (sleep 0.01)))
           (dolist (thread threads)
             (let ((result (sb-thread:join-thread thread :timeout 180 :default :timeout)))
               (assert (eq result :passed) () "Child test failed: ~A" result)))
           (assert (eq check (symbol-function 'cl-gbdt-examples::check-sha256)))
           (assert (eq predict (symbol-function 'cl-gbdt:predict)))
           (assert (eq system (asdf:registered-system :lisp-xl)))
           (assert (equal fold (sb-posix:getenv "CL_GBDT_MSLR_FOLD"))))
      (dolist (thread threads)
        (when (sb-thread:thread-alive-p thread)
          (sb-thread:join-thread thread :timeout 180 :default nil))))))

(test-example-process-isolation)
(format t "Example test process isolation passed.~%")
