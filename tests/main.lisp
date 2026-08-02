(defpackage cl-gbdt/tests/main
  (:use :cl
        :cl-gbdt
        :rove))
(in-package :cl-gbdt/tests/main)

;; NOTE: To run this test file, execute `(asdf:test-system :cl-gbdt)' in your Lisp.

(deftest test-target-1
  (testing "should (= 1 1) to be true"
    (ok (= 1 1))))
