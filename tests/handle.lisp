;;;; handle.lisp --- Tests for the handle layer.
;;;;
;;;; A CFFI pointer stand-in and a counting closure play the backend's "free"
;;;; function, so these need no shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/handle
  (:use #:cl #:rove)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/handle #:%make-finalizer)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/handle)

(defun counting-free ()
  "Return (values FREE-FUNCTION COUNT-CELL). FREE-FUNCTION ignores its argument and
increments the car of COUNT-CELL each time it is called."
  (let ((count (list 0)))
    (values (lambda (pointer) (declare (ignore pointer)) (incf (car count)))
            count)))

(deftest release-handle-frees-exactly-once
  (testing "the free function is called exactly once"
    (multiple-value-bind (free count) (counting-free)
      (let ((handle (cl-gbdt:make-handle 'cl-gbdt:dataset (cffi:make-pointer 1)
                                          :mock :dataset)))
        (cl-gbdt:release-handle handle free)
        (ok (= 1 (car count)))))))

(deftest release-handle-second-call-is-a-no-op
  (testing "a second release leaves the counter at 1 and signals nothing"
    (multiple-value-bind (free count) (counting-free)
      (let ((handle (cl-gbdt:make-handle 'cl-gbdt:dataset (cffi:make-pointer 1)
                                          :mock :dataset)))
        (cl-gbdt:release-handle handle free)
        (cl-gbdt:release-handle handle free)
        (ok (= 1 (car count)))))))

(deftest live-pointer-on-released-handle-signals
  (testing "handle-live-pointer signals released-handle-error once released"
    (multiple-value-bind (free count) (counting-free)
      (declare (ignore count))
      (let ((handle (cl-gbdt:make-handle 'cl-gbdt:dataset (cffi:make-pointer 1)
                                          :mock :dataset)))
        (cl-gbdt:release-handle handle free)
        ;; handler-case, not rove's `signals' -- `signals' does not reliably catch
        ;; conditions raised inside `restart-case', a documented pitfall in this
        ;; repo's own prompts/repl-driven-development.md.
        (ok (handler-case (progn (cl-gbdt:handle-live-pointer handle) nil)
              (cl-gbdt:released-handle-error () t)))))))

(deftest live-pointer-on-live-handle-returns-the-pointer
  (testing "handle-live-pointer returns the pointer the handle was made with"
    (let* ((pointer (cffi:make-pointer 1))
           (handle (cl-gbdt:make-handle 'cl-gbdt:dataset pointer :mock :dataset)))
      (ok (cffi:pointer-eq pointer (cl-gbdt:handle-live-pointer handle))))))

(deftest finalizer-closure-reports-only-when-unreleased
  ;; Invoking the closure directly, not provoking a collection: depending on GC
  ;; timing would make this flaky. `%make-finalizer' is the helper `make-handle'
  ;; itself uses, so this exercises the exact closure a real handle carries.
  (testing "with the flag unset, the closure warns unfreed-handle-warning"
    (let ((cell (list nil)))
      (ok (handler-case (progn (funcall (%make-finalizer cell :dataset)) nil)
            (cl-gbdt:unfreed-handle-warning () t)))))
  (testing "with the flag set, the closure signals nothing"
    (let ((cell (list t)))
      (ok (handler-case (progn (funcall (%make-finalizer cell :dataset)) t)
            (cl-gbdt:unfreed-handle-warning () nil))))))

(deftest booster-retains-training-set
  (testing "a booster's training-set reader returns what it was made with"
    (let* ((dataset (cl-gbdt:make-handle 'cl-gbdt:dataset (cffi:make-pointer 1)
                                          :mock :dataset))
           (booster (cl-gbdt:make-handle 'cl-gbdt:booster (cffi:make-pointer 2)
                                          :mock :booster :training-set dataset)))
      (ok (eq dataset (cl-gbdt:booster-training-set booster))))))
