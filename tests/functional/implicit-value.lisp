;;;; implicit-value.lisp --- The CSR absence declaration, refused per backend.
;;;;
;;;; `make-csr-matrix''s :IMPLICIT-VALUE says what an entry the matrix does not store means in
;;;; the caller's own data. LightGBM reads an absent entry as 0.0 and XGBoost reads one as
;;;; missing, so exactly one of the two value declarations is true on each. :NONE -- nothing is
;;;; absent -- is true on both, and `make-csr-matrix' has already verified it structurally.
;;;;
;;;; What a green run here does NOT mean: nothing in this file establishes what either library
;;;; does with an absent entry. That is
;;;; `an-omitted-entry-is-zero-to-lightgbm-and-missing-to-xgboost' in sparse-input.lisp, which
;;;; measures it. This file asserts only that the wrapper refuses what disagrees with it.

(uiop:define-package #:cl-gbdt/tests/functional/implicit-value
  ;; Zero symbols: every reference below is package-qualified.
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt)
  ;; Zero symbols, both: they run at load time to register the backends and define their
  ;; methods on `cl-gbdt''s generics. `unified' rather than `all' since the Layer 1 split.
  (:import-from #:cl-gbdt/src/lightgbm/unified)
  (:import-from #:cl-gbdt/src/xgboost/unified)
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library))

(in-package #:cl-gbdt/tests/functional/implicit-value)

(defparameter *labels* '(0 0 1 1)
  "Four rows, two per class.

Four rather than the two a refusal needs, because two of the tests below must actually TRAIN in
order to have a booster to call `predict' on, and LightGBM bins its data before it splits --
a two-row fixture is thin enough that a binning warning or a degenerate model would be noise in
a file that is not about training at all. Feature 0 is 1, 3, 5, 7 down the rows, so a single
threshold separates the classes.")

(defun declared-csr (implicit-value)
  "Return a four-row, two-column `csr-matrix' declaring IMPLICIT-VALUE.

Row 3 stores only column 0, so the matrix genuinely has an absent entry -- a declaration about
absence on a matrix that has none would be vacuous, and :NONE would not even build."
  (cl-gbdt:make-csr-matrix :indptr '(0 2 4 6 7) :indices '(0 1 0 1 0 1 0)
                           :values '(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0 6.0d0 7.0d0)
                           :num-columns 2
                           :implicit-value implicit-value))

(defun complete-csr (implicit-value)
  "Return a four-row, two-column `csr-matrix' storing every element, declaring IMPLICIT-VALUE.

The training fixture, and the only one `:NONE' can be declared on."
  (cl-gbdt:make-csr-matrix :indptr '(0 2 4 6 8) :indices '(0 1 0 1 0 1 0 1)
                           :values '(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0 6.0d0 7.0d0 8.0d0)
                           :num-columns 2
                           :implicit-value implicit-value))

(defun refusal-reason (thunk)
  "Run THUNK and return the `unsupported-argument' it signals, printed. NIL if it does not."
  (handler-case (progn (funcall thunk) nil)
    (cl-gbdt:unsupported-argument (condition) (princ-to-string condition))))

(defmacro with-open-backend ((variable backend) &body body)
  "Open BACKEND, bind it to VARIABLE for BODY, and close it however BODY leaves."
  `(with-backend-library (,backend)
     (let ((,variable (cl-gbdt:open-backend ,backend)))
       (unwind-protect (progn ,@body)
         (cl-gbdt:close-backend ,variable)))))

(deftest lightgbm-refuses-a-missing-declaration
  (with-open-backend (backend :lightgbm)
    (testing "make-dataset refuses :IMPLICIT-VALUE :MISSING on LightGBM"
      (let ((reason (refusal-reason
                     (lambda ()
                       (cl-gbdt:make-dataset backend (declared-csr :missing)
                                             :label *labels*)))))
        (ok reason "whether it signalled unsupported-argument at all")
        (ok (and reason (search "zero_as_missing" reason))
            (format nil "whether the reason names the flag cl-gbdt does not read: ~A" reason))))
    (testing "predict refuses it too -- a separate code path, separately gated"
      (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend (complete-csr nil)
                                                           :label *labels*))
        (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 1
                                                      :parameters '(:objective "binary"
                                                                    :num-leaves 2
                                                                    :min-data-in-leaf 1
                                                                    :min-data-in-bin 1
                                                                    :verbose -1)))
          (ok (refusal-reason
               (lambda () (cl-gbdt:predict booster (declared-csr :missing))))
              "whether predict refused the same declaration"))))))

(deftest lightgbm-accepts-a-zero-and-none-declaration
  (with-open-backend (backend :lightgbm)
    (testing "a zero declaration matches how LightGBM reads absence"
      (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend (declared-csr 0.0d0)
                                                           :label *labels*))
        (ok (= 4 (cl-gbdt:dataset-num-rows dataset)) "the dataset was built")))
    (testing ":NONE is accepted -- it is true on either backend"
      (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend (complete-csr :none)
                                                           :label *labels*))
        (ok (= 4 (cl-gbdt:dataset-num-rows dataset)) "the dataset was built")))))

(deftest xgboost-refuses-a-zero-declaration
  (with-open-backend (backend :xgboost)
    (testing "make-dataset refuses a zero :IMPLICIT-VALUE on XGBoost"
      (let ((reason (refusal-reason
                     (lambda ()
                       (cl-gbdt:make-dataset backend (declared-csr 0.0d0)
                                             :label *labels*)))))
        (ok reason "whether it signalled unsupported-argument at all")
        (ok (and reason (search "missing" reason))
            (format nil "whether the reason says absence is missing here: ~A" reason))))
    (testing "predict refuses it too -- a separate code path, separately gated"
      (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend (complete-csr nil)
                                                           :label *labels*))
        (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 1
                                                      :parameters '(:objective "binary:logistic"
                                                                    :max-depth 2
                                                                    :verbosity 0)))
          (ok (refusal-reason
               (lambda () (cl-gbdt:predict booster (declared-csr 0.0d0))))
              "whether predict refused the same declaration"))))))

(deftest xgboost-accepts-a-missing-and-none-declaration
  (with-open-backend (backend :xgboost)
    (testing ":MISSING matches how XGBoost reads absence"
      (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend (declared-csr :missing)
                                                           :label *labels*))
        (ok (= 4 (cl-gbdt:dataset-num-rows dataset)) "the dataset was built")))
    (testing ":NONE is accepted -- it is true on either backend"
      (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend (complete-csr :none)
                                                           :label *labels*))
        (ok (= 4 (cl-gbdt:dataset-num-rows dataset)) "the dataset was built")))))

(deftest an-undeclared-matrix-is-checked-by-neither-backend
  ;; The compatibility guarantee: NIL means today's behaviour, on both backends, unchanged.
  (dolist (backend-name '(:lightgbm :xgboost))
    (with-open-backend (backend backend-name)
      (testing (format nil "~A: an undeclared matrix builds a dataset as it always did"
                       backend-name)
        (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend (declared-csr nil)
                                                             :label *labels*))
          (ok (= 4 (cl-gbdt:dataset-num-rows dataset)) "the dataset was built"))))))

(deftest csr-matrix-implicit-value-reads-back-what-was-declared
  ;; Needs no backend: the refusals above are only trustworthy proof of anything if
  ;; make-dataset and predict are reading the same value make-csr-matrix stored, and this is
  ;; the reader they read it through.
  (testing "the four legal declarations round-trip through csr-matrix-implicit-value"
    (ok (null (cl-gbdt:csr-matrix-implicit-value (declared-csr nil)))
        "NIL declares nothing")
    (ok (eql :missing (cl-gbdt:csr-matrix-implicit-value (declared-csr :missing)))
        ":MISSING round-trips")
    (ok (eql 0.0d0 (cl-gbdt:csr-matrix-implicit-value (declared-csr 0.0d0)))
        "a zero round-trips, canonicalized to 0.0d0")
    (ok (eql :none (cl-gbdt:csr-matrix-implicit-value (complete-csr :none)))
        ":NONE round-trips")))
