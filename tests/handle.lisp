;;;; handle.lisp --- Tests for the handle layer.
;;;;
;;;; A CFFI pointer stand-in and a counting closure play the backend's "free"
;;;; function, so these need no shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/handle
  (:use #:cl #:rove)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/handle #:%make-finalizer #:%check-handle-kind)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/handle)

(defclass %mock-backend (cl-gbdt:backend) ()
  (:documentation "A backend stand-in for handle tests: opens and closes without any
real foreign library, so `handle-live-pointer''s backend-open check has something to
test against without needing LightGBM or XGBoost installed. Mirrors tests/backend.lisp's
`mock-backend', kept separate rather than shared to avoid a cross-test-file
dependency between two otherwise-independent leaf systems."))

(defmethod cl-gbdt:initialize-backend ((backend %mock-backend) &key path)
  (declare (ignore path))
  backend)

(defmethod cl-gbdt:shutdown-backend ((backend %mock-backend))
  backend)

(cl-gbdt:register-backend :handle-test '%mock-backend)

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
    (let* ((backend (cl-gbdt:open-backend :handle-test))
           (pointer (cffi:make-pointer 1))
           (handle (cl-gbdt:make-handle 'cl-gbdt:dataset pointer backend :dataset)))
      (unwind-protect
           (ok (cffi:pointer-eq pointer (cl-gbdt:handle-live-pointer handle)))
        (cl-gbdt:close-backend backend)))))

(deftest live-pointer-on-handle-whose-backend-is-closed-signals-backend-not-open
  (testing "handle-live-pointer signals backend-not-open once the backend is closed"
    (let* ((backend (cl-gbdt:open-backend :handle-test))
           (handle (cl-gbdt:make-handle 'cl-gbdt:dataset (cffi:make-pointer 1)
                                         backend :dataset)))
      (cl-gbdt:close-backend backend)
      ;; handler-case, not rove's `signals' -- see live-pointer-on-released-handle-signals
      ;; above for why.
      (ok (handler-case (progn (cl-gbdt:handle-live-pointer handle) nil)
            (cl-gbdt:backend-not-open () t))))))

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

;;; `%check-handle-kind' is the guard the two backends' Layer 1 entry points use where CLOS
;;; cannot make the check for them, and getting it wrong means a handle allocated by one
;;; shared library being dereferenced by the other -- a memory fault, not a condition. Until
;;; this file, its wrong-BACKEND branch could only be reached with both real libraries
;;; loaded; tests/functional/evaluation.lisp does that end to end. These cover the same
;;; branch here, at layer 1, where two mock backends cost nothing and no library is needed.

(defclass %other-mock-backend (cl-gbdt:backend) ()
  (:documentation "A second backend stand-in, registered under a different name than
`%mock-backend', so a handle built by one can be offered to a check expecting the other --
the wrong-backend case, which needs two distinct `backend-name's and nothing else."))

(defmethod cl-gbdt:initialize-backend ((backend %other-mock-backend) &key path)
  (declare (ignore path))
  backend)

(defmethod cl-gbdt:shutdown-backend ((backend %other-mock-backend))
  backend)

(cl-gbdt:register-backend :other-handle-test '%other-mock-backend)

(defun %handle-on (backend-name class)
  "Return a live CLASS handle whose backend is a freshly opened BACKEND-NAME."
  (cl-gbdt:make-handle class (cffi:make-pointer 1)
                       (cl-gbdt:open-backend backend-name) :dataset))

(deftest check-handle-kind-accepts-the-right-kind-from-the-right-backend
  (testing "the live pointer comes back unchanged"
    (let ((handle (%handle-on :handle-test 'cl-gbdt:dataset)))
      (ok (cffi:pointer-eq (cffi:make-pointer 1)
                           (%check-handle-kind
                            handle 'cl-gbdt:dataset :handle-test "a dataset argument"))
          "%check-handle-kind did not return the handle's pointer"))))

(deftest check-handle-kind-rejects-the-right-kind-from-another-backend
  ;; The branch with no coverage before this test: right class, wrong library. Weakening it
  ;; is what reproduces the memory fault in tests/functional/evaluation.lisp.
  (testing "a dataset from another backend signals wrong-backend-reference"
    (let ((handle (%handle-on :other-handle-test 'cl-gbdt:dataset)))
      (ok (handler-case
              (progn (%check-handle-kind
                      handle 'cl-gbdt:dataset :handle-test "a dataset argument")
                     nil)
            (cl-gbdt:wrong-backend-reference () t))
          "%check-handle-kind accepted a dataset from another backend"))))

(deftest check-handle-kind-rejects-the-wrong-kind-from-the-right-backend
  (testing "a booster where a dataset belongs signals wrong-backend-reference"
    (let ((handle (%handle-on :handle-test 'cl-gbdt:booster)))
      (ok (handler-case
              (progn (%check-handle-kind
                      handle 'cl-gbdt:dataset :handle-test "a dataset argument")
                     nil)
            (cl-gbdt:wrong-backend-reference () t))
          "%check-handle-kind accepted a booster as a dataset"))))

(deftest check-handle-kind-reports-the-kind-it-expected
  ;; The noun in the report is derived from KIND rather than passed separately, so that a
  ;; caller asking for a booster can never be told it "must be a dataset".
  (testing "the report names the kind the caller asked for"
    (let ((handle (%handle-on :handle-test 'cl-gbdt:dataset)))
      (ok (search "must be a booster"
                  (handler-case
                      (progn (%check-handle-kind
                              handle 'cl-gbdt:booster :handle-test "a booster argument")
                             "")
                    (cl-gbdt:wrong-backend-reference (c) (princ-to-string c))))
          "the report did not name booster as the expected kind"))))
