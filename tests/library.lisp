;;;; library.lisp --- Tests for shared shared-library discovery.
;;;;
;;;; Every case here is a discovery failure, deliberately: the point is to prove the
;;;; PATH -> ENV-VAR -> DIRECTORY -> system-search order and the strict-ENV-VAR rule
;;;; without ever successfully loading a real shared library into the image (layer 1).
;;;; A successful load is exercised for real by the LightGBM functional suite instead.

(uiop:define-package #:cl-gbdt/tests/library
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/library)

(defun %test-backend ()
  "Return a fresh `cl-gbdt:backend' instance for `resolve-and-load-library' to name in
its condition reports. No `initialize-backend' method is needed: this file never lets
resolution reach a real load."
  (make-instance 'cl-gbdt:backend :name :library-test))

(deftest resolve-and-load-library-honors-explicit-path
  (testing "an explicit :path is tried before :env-var or :directory, and reported as such"
    ;; Neither :env-var nor :directory names anything real, so if :path were not tried
    ;; first this would instead signal backend-library-not-found from a later stage --
    ;; distinguishing the two proves :path won.
    (let ((condition (handler-case
                          (progn (cl-gbdt:resolve-and-load-library
                                  (%test-backend)
                                  :path "/nonexistent/cl-gbdt-test-path.so"
                                  :env-var "CL_GBDT_TEST_LIBRARY_UNSET_VAR"
                                  :directory "tests/fixtures-that-do-not-exist/"
                                  :pattern "*.nonexistent"
                                  :default-name "cl_gbdt_test_definitely_not_a_real_library")
                                 nil)
                        (cl-gbdt:backend-library-load-failed (c) c))))
      (ok condition "an explicit :path that fails to load signals backend-library-load-failed")
      (ok (equal "/nonexistent/cl-gbdt-test-path.so"
                  (cl-gbdt:backend-library-load-failed-path condition))))))

(deftest resolve-and-load-library-env-var-is-strict
  (testing (format nil "an :env-var set to a nonexistent path signals ~
                         backend-library-not-found, not a fall-through to :directory or ~
                         the system search")
    (let ((variable "CL_GBDT_TEST_LIBRARY_ENV_VAR_NONEXISTENT"))
      (unwind-protect
           (progn
             (setf (uiop:getenv variable) "/nonexistent/cl-gbdt-test-library.so")
             (let ((condition (handler-case
                                   (progn (cl-gbdt:resolve-and-load-library
                                           (%test-backend)
                                           :env-var variable
                                           :directory "tests/fixtures-that-do-not-exist/"
                                           :pattern "*.nonexistent"
                                           :default-name
                                           "cl_gbdt_test_definitely_not_a_real_library")
                                          nil)
                                 (cl-gbdt:backend-library-not-found (c) c))))
               (ok condition)
               ;; :searched names only the override -- not the directory search or the
               ;; system search string -- which is what proves this did not fall through.
               (ok (equal (list "/nonexistent/cl-gbdt-test-library.so")
                          (cl-gbdt:backend-library-not-found-searched condition)))))
        (setf (uiop:getenv variable) nil)))))

(deftest resolve-and-load-library-falls-through-an-absent-directory
  (testing (format nil "no :path, an unset :env-var, and a :directory with no match all ~
                         fall through to CFFI's own system search")
    (let ((condition (handler-case
                          (progn (cl-gbdt:resolve-and-load-library
                                  (%test-backend)
                                  :env-var "CL_GBDT_TEST_LIBRARY_ANOTHER_UNSET_VAR"
                                  :directory "tests/fixtures-that-do-not-exist/"
                                  :pattern "*.nonexistent"
                                  :default-name "cl_gbdt_test_definitely_not_a_real_library")
                                 nil)
                        (cl-gbdt:backend-library-not-found (c) c))))
      (ok condition
          "reaching the system search and failing there still signals backend-library-not-found")
      (testing "the report names both the directory search and the system search as tried"
        (let ((searched (cl-gbdt:backend-library-not-found-searched condition)))
          (ok (= 2 (length searched)))
          (ok (search "tests/fixtures-that-do-not-exist" (first searched)))
          (ok (equal "system library search path" (second searched))))))))
