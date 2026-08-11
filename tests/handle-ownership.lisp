;;;; handle-ownership.lisp --- Tests for `with-pointer-ownership'.
;;;;
;;;; A CFFI pointer stand-in and counting closures play the backend's "free" function, so
;;;; these need no shared library (layer 1). The pointer is never dereferenced.

(uiop:define-package #:cl-gbdt/tests/handle-ownership
  (:use #:cl #:rove)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/handle
                #:with-pointer-ownership)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/handle-ownership)

(define-condition body-failure (error) ()
  (:documentation "Signalled by a test body to unwind out of `with-pointer-ownership'. A
condition class of its own, rather than `error', so an assertion that it is what escaped
cannot be satisfied by an error raised somewhere else -- the cleanup's own, in particular."))

(defparameter *pointer* (cffi:make-pointer 1)
  "A pointer stand-in, never dereferenced.")

(defun test-backend ()
  "Return a backend instance for `make-handle' to record on a handle. Never opened: nothing
here reaches a shared library."
  (make-instance 'cl-gbdt:backend :name :ownership-test))

(defun counting-free ()
  "Return (values FREE-FUNCTION COUNT-CELL). FREE-FUNCTION ignores its argument and
increments the car of COUNT-CELL."
  (let ((count (list 0)))
    (values (lambda (pointer) (declare (ignore pointer)) (incf (car count)))
            count)))

(defun drop (handle)
  "Release HANDLE with a free function that does nothing, so its finalizer is cancelled and
no `unfreed-handle-warning' reaches a later garbage collection."
  (cl-gbdt:release-handle handle (lambda (pointer) (declare (ignore pointer)) nil)))

(deftest taking-ownership-suppresses-the-free
  (testing "a body that takes ownership leaves the pointer alone"
    (multiple-value-bind (free count) (counting-free)
      (let ((handle (with-pointer-ownership (*pointer* free take-ownership)
                      (take-ownership 'cl-gbdt:dataset (test-backend) :dataset))))
        (ok (typep handle 'cl-gbdt:dataset) "the operator's handle is the form's value")
        (ok (= 0 (car count)) "the free function was not called")
        (drop handle)))))

(deftest a-body-that-signals-frees-the-pointer
  (testing "the free function runs once and the body's own condition escapes"
    (multiple-value-bind (free count) (counting-free)
      (let ((caught (handler-case
                        (with-pointer-ownership (*pointer* free take-ownership)
                          (error 'body-failure))
                      (body-failure (condition) condition))))
        (ok (typep caught 'body-failure) "the body's condition reached the caller")
        (ok (= 1 (car count)) "the free function ran exactly once")))))

(deftest an-error-from-the-free-function-does-not-replace-the-original
  (testing "a failing cleanup cannot mask the condition that caused the unwind"
    (let ((caught (handler-case
                      (with-pointer-ownership
                          (*pointer*
                           (lambda (pointer)
                             (declare (ignore pointer))
                             (error "cleanup failed"))
                           take-ownership)
                        (error 'body-failure))
                    (error (condition) condition))))
      (ok (typep caught 'body-failure)
          "the body's condition escaped, not the cleanup's"))))

(deftest the-bodys-values-are-the-forms-values
  (testing "a body returning two values returns both"
    (multiple-value-bind (free count) (counting-free)
      (declare (ignore count))
      (multiple-value-bind (handle extra)
          (with-pointer-ownership (*pointer* free take-ownership)
            (values (take-ownership 'cl-gbdt:dataset (test-backend) :dataset) :report))
        (ok (typep handle 'cl-gbdt:dataset) "the first value is the handle")
        (ok (eq :report extra) "the second value survived")
        (drop handle)))))

(deftest a-signal-after-ownership-moved-does-not-free
  (testing "once the handle exists, unwinding is the handle's problem, not this macro's"
    (multiple-value-bind (free count) (counting-free)
      (let ((handle nil))
        (handler-case
            (with-pointer-ownership (*pointer* free take-ownership)
              (setf handle (take-ownership 'cl-gbdt:dataset (test-backend) :dataset))
              (error 'body-failure))
          (body-failure () nil))
        (ok (= 0 (car count)) "the free function did not run")
        (ok (typep handle 'cl-gbdt:dataset) "the handle was made before the signal")
        (drop handle)))))

(deftest the-pointer-form-is-evaluated-once
  (testing "a side-effecting pointer form runs exactly once"
    (multiple-value-bind (free count) (counting-free)
      (declare (ignore count))
      (let ((evaluations 0)
            (handle nil))
        (setf handle
              (with-pointer-ownership ((progn (incf evaluations) *pointer*) free take-ownership)
                (take-ownership 'cl-gbdt:dataset (test-backend) :dataset)))
        (ok (= 1 evaluations) "the pointer form was evaluated exactly once")
        (drop handle)))))

(deftest a-body-that-never-takes-ownership-frees-on-a-normal-return
  (testing "a body that simply returns without taking ownership still frees"
    (multiple-value-bind (free count) (counting-free)
      (let ((result (with-pointer-ownership (*pointer* free take-ownership)
                      :nothing-taken)))
        (ok (eq :nothing-taken result) "the body's value is still the form's value")
        (ok (= 1 (car count)) "the free function ran")))))
