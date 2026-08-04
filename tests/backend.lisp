;;;; backend.lisp --- Tests for the backend registry and protocol.
;;;;
;;;; A mock backend is used, so no shared library is required (layer 1).

(uiop:define-package #:cl-gbdt/tests/backend
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/backend)

(defclass mock-backend (cl-gbdt:backend)
  ((initialize-count :initform 0 :accessor mock-backend-initialize-count)
   (shutdown-count :initform 0 :accessor mock-backend-shutdown-count)
   (fail-with :initarg :fail-with :initform nil :accessor mock-backend-fail-with))
  (:documentation "Test backend that never opens a shared library."))

(defmethod cl-gbdt:initialize-backend ((backend mock-backend) &key path)
  (declare (ignore path))
  (when (mock-backend-fail-with backend)
    (error 'cl-gbdt:backend-library-not-found
           :backend (cl-gbdt:backend-name backend)
           :searched '("/nowhere/libmock.so")))
  (incf (mock-backend-initialize-count backend))
  (setf (cl-gbdt:backend-capabilities backend) '(:mock t)
        (cl-gbdt:backend-version backend) "0.0.0-mock")
  backend)

(defmethod cl-gbdt:shutdown-backend ((backend mock-backend))
  (incf (mock-backend-shutdown-count backend))
  backend)

(cl-gbdt:register-backend :mock 'mock-backend)

(defclass failing-backend (mock-backend)
  ((fail-with :initform t))
  (:documentation "Test backend whose initialization always fails."))

(cl-gbdt:register-backend :failing 'failing-backend)

(deftest registry-resolves-registered-backend
  (testing "a registered backend resolves to its class name"
    (ok (eq 'mock-backend (cl-gbdt:find-backend-class :mock))))
  (testing "an unregistered name resolves to nil"
    (ok (null (cl-gbdt:find-backend-class :no-such-backend)))))

(deftest open-backend-initializes-and-marks-open
  (testing "open-backend runs initialization and marks the backend open"
    (let ((backend (cl-gbdt:open-backend :mock)))
      (unwind-protect
           (progn
             (ok (typep backend 'mock-backend))
             (ok (cl-gbdt:backend-open-p backend))
             (ok (= 1 (mock-backend-initialize-count backend)))
             (ok (eq :mock (cl-gbdt:backend-name backend))))
        (cl-gbdt:close-backend backend)))))

(deftest close-backend-is-idempotent
  (testing "calling close-backend twice shuts down only once"
    (let ((backend (cl-gbdt:open-backend :mock)))
      (cl-gbdt:close-backend backend)
      (cl-gbdt:close-backend backend)
      (ok (= 1 (mock-backend-shutdown-count backend)))
      (ng (cl-gbdt:backend-open-p backend)))))

(deftest open-backend-rejects-unknown-name
  (testing "an unregistered backend name signals unknown-backend, not backend-not-open"
    (ok (handler-case (progn (cl-gbdt:open-backend :no-such-backend) nil)
          (cl-gbdt:unknown-backend () t))))
  (testing "the report names the requested backend and lists the registered ones"
    (let ((text (handler-case (progn (cl-gbdt:open-backend :no-such-backend) nil)
                  (cl-gbdt:unknown-backend (c) (princ-to-string c)))))
      (ok (search "NO-SUCH-BACKEND" text))
      (ok (search "MOCK" text)))))

(deftest initialization-failure-propagates
  (testing "open-backend propagates an initialization failure"
    (ok (handler-case (progn (cl-gbdt:open-backend :failing) nil)
          (cl-gbdt:backend-library-not-found () t))))
  (testing "an instance whose initialization failed is not marked open"
    (let ((backend (make-instance 'failing-backend :name :failing)))
      (ok (handler-case
              (progn (cl-gbdt:initialize-backend backend) nil)
            (cl-gbdt:backend-library-not-found () t)))
      (ng (cl-gbdt:backend-open-p backend)))))

(deftest backend-info-reports-capabilities-and-version
  (testing "backend-info reports capabilities and version"
    (let ((backend (cl-gbdt:open-backend :mock)))
      (unwind-protect
           (let ((info (cl-gbdt:backend-info backend)))
             (ok (eq :mock (getf info :name)))
             (ok (equal "0.0.0-mock" (getf info :version)))
             (ok (equal '(:mock t) (getf info :capabilities)))
             (ok (eq t (getf info :open))))
        (cl-gbdt:close-backend backend)))))

(deftest probe-foreign-symbols-detects-missing
  (testing "nonexistent C function names are reported"
    (let ((missing (cl-gbdt:probe-foreign-symbols
                    '("cl_gbdt_definitely_not_a_real_symbol_1"
                      "cl_gbdt_definitely_not_a_real_symbol_2"))))
      (ok (= 2 (length missing)))
      (ok (member "cl_gbdt_definitely_not_a_real_symbol_1" missing :test #'string=))))
  (testing "a libc function is found"
    (ok (null (cl-gbdt:probe-foreign-symbols '("strlen"))))))

;;; F3: `probe-foreign-symbols' used to call `cffi:foreign-symbol-pointer' with no
;;; `:library', which searches every foreign library the image currently has
;;; loaded. It now accepts a LIBRARY keyword and passes it straight through. This
;;; layer stays free of any real foreign library (see this file's header
;;; comment), so it cannot load a second, distinguishable LightGBM here. What it
;;; can prove without loading anything: the default keeps today's behavior
;;; working, and LIBRARY reaches `cffi:foreign-symbol-pointer' rather than being
;;; silently dropped -- a designator CFFI has never loaded signals instead of
;;; falling back to a global search. It does NOT prove LIBRARY actually scopes
;;; the search on this project's platform: verified by hand against the vendored
;;; LightGBM and XGBoost libraries (see `probe-foreign-symbols''s docstring),
;;; SBCL's own `cffi-sbcl.lisp' validates LIBRARY and then ignores it, always
;;; resolving through SBCL's global linkage table. That caveat belongs in the
;;; docstring, which carries it; this test only covers what it can actually
;;; observe without a real library.

(deftest probe-foreign-symbols-honors-library-keyword
  (testing "an explicit :default keyword behaves exactly like omitting it"
    (ok (null (cl-gbdt:probe-foreign-symbols '("strlen") :library :default))))
  (testing "LIBRARY reaches cffi:foreign-symbol-pointer instead of being ignored"
    ;; A designator CFFI has never loaded signals rather than silently searching
    ;; globally -- this is only observable at all if LIBRARY is actually threaded
    ;; through, not dropped on the floor.
    (ok (handler-case
            (progn (cl-gbdt:probe-foreign-symbols
                    '("strlen") :library :cl-gbdt-tests-unregistered-library)
                   nil)
          (error () t))
        "probing an unregistered LIBRARY did not signal")))

(deftest protocol-generic-functions-exist
  (testing "every unified API generic function is defined"
    (dolist (name '(cl-gbdt:make-dataset
                    cl-gbdt:dataset-num-rows
                    cl-gbdt:dataset-num-features
                    cl-gbdt:train
                    cl-gbdt:update-one-iteration
                    cl-gbdt:predict
                    cl-gbdt:save-model
                    cl-gbdt:load-model
                    cl-gbdt:model-to-string
                    cl-gbdt:feature-importance
                    cl-gbdt:free-dataset
                    cl-gbdt:free-booster))
      (ok (typep (fdefinition name) 'generic-function)
          (format nil "~A is a generic function" name)))))

(defvar *freed* nil
  "Records which handles were freed, for the with-dataset / with-booster tests.")

(defclass mock-handle () ())

(defmethod cl-gbdt:free-dataset ((handle mock-handle))
  (push :dataset *freed*))

(defmethod cl-gbdt:free-booster ((handle mock-handle))
  (push :booster *freed*))

(deftest with-dataset-frees-on-normal-exit
  (testing "the dataset is freed on normal exit"
    (let ((*freed* nil))
      (cl-gbdt:with-dataset (dataset (make-instance 'mock-handle))
        (ok (typep dataset 'mock-handle)))
      (ok (equal '(:dataset) *freed*)))))

(deftest with-dataset-frees-on-nonlocal-exit
  (testing "the dataset is freed even when the body signals"
    (let ((*freed* nil))
      (ignore-errors
       (cl-gbdt:with-dataset (dataset (make-instance 'mock-handle))
         (declare (ignore dataset))
         (error "boom")))
      (ok (equal '(:dataset) *freed*)))))

(deftest with-booster-frees-on-nonlocal-exit
  (testing "the booster is likewise always freed"
    (let ((*freed* nil))
      (ignore-errors
       (cl-gbdt:with-booster (booster (make-instance 'mock-handle))
         (declare (ignore booster))
         (error "boom")))
      (ok (equal '(:booster) *freed*)))))
