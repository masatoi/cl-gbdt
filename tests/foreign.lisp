;;;; foreign.lisp --- Tests for shared foreign-call status checking.
;;;;
;;;; LAST-ERROR is a plain Lisp function stand-in here, not a call into LightGBM or
;;;; XGBoost, so these need no shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/foreign
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/foreign)

(deftest check-foreign-call-returns-the-code-on-success
  (testing "a zero code returns the code, and never calls the last-error thunk"
    (let ((called nil))
      (ok (zerop (cl-gbdt:check-foreign-call 0 "SomeFunction" (lambda () (setf called t)))))
      (ng called "the last-error thunk is not called when CODE is zero"))))

(deftest check-foreign-call-signals-on-failure
  (testing (format nil "a nonzero code signals foreign-call-error carrying the function ~
                         name, code, and whatever the thunk returned")
    (let ((condition (handler-case
                          (progn (cl-gbdt:check-foreign-call -1 "SomeFunction"
                                                              (lambda () "boom"))
                                 nil)
                        (cl-gbdt:foreign-call-error (c) c))))
      (ok condition)
      (ok (equal "SomeFunction" (cl-gbdt:foreign-call-error-function-name condition)))
      (ok (eql -1 (cl-gbdt:foreign-call-error-code condition)))
      (ok (equal "boom" (cl-gbdt:foreign-call-error-message condition))))))

(deftest check-foreign-call-tolerates-a-nil-message
  (testing "a last-error thunk returning nil does not break the report"
    (let ((condition (handler-case
                          (progn (cl-gbdt:check-foreign-call -1 "SomeFunction" (lambda () nil))
                                 nil)
                        (cl-gbdt:foreign-call-error (c) c))))
      (ok condition)
      (ok (null (cl-gbdt:foreign-call-error-message condition)))
      (ok (search "(no message)" (princ-to-string condition))))))
