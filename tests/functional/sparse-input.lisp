;;;; sparse-input.lisp --- Portable contract tests for `make-dataset' on a `csr-matrix'.
;;;;
;;;; `cl-gbdt:make-dataset' now accepts a `cl-gbdt:csr-matrix' wherever it accepted a dense
;;;; matrix, on both backends, gated by the `:sparse-input' capability. Like
;;;; tests/functional/evaluation.lisp and tests/functional/training-report.lisp beside it,
;;;; every test below runs the identical assertions over that first file's *FIXTURES*, once
;;;; per backend, so the two backends cannot drift apart in shape or meaning without one of
;;;; them failing here. The one exception is the last test, which needs a backend that aligns
;;;; bin mappers and so runs on LightGBM alone -- see its own comment.
;;;;
;;;; Numbers are never compared BETWEEN backends: policy section 13 asks for shape, order and
;;;; meaning, not numeric agreement, and the two libraries train different models from the
;;;; same rows by design. `sparse-and-dense-training-agree' does compare numbers, but only a
;;;; backend's own sparse result against its own dense one -- the comparison is within one
;;;; backend, never across the two.
;;;;
;;;; Every prediction below is made on a DENSE matrix, including the ones read out of a
;;;; sparsely-trained booster. `predict' does not take a `csr-matrix' yet; that is the next
;;;; task. Predicting densely here keeps these tests about ingestion, which is what this task
;;;; changed.

(uiop:define-package #:cl-gbdt/tests/functional/sparse-input
  ;; Zero symbols: every reference below is package-qualified. Declared so this file's
  ;; dependency on the unified API is explicit rather than inherited, matching the identical
  ;; clause in evaluation.lisp and training-report.lisp.
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt)
  ;; Zero symbols, both of them: their only job is to run at load time and register
  ;; :lightgbm and :xgboost with `open-backend'. Without these clauses,
  ;; package-inferred-system has no edge to those files and `(cl-gbdt:open-backend
  ;; :lightgbm)' below would signal `unknown-backend'.
  (:import-from #:cl-gbdt/src/lightgbm/all)
  (:import-from #:cl-gbdt/src/xgboost/all)
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library
                #:make-separable-dataset
                #:predictions-separate-p)
  ;; The fixture table and its dataset builder come from evaluation.lisp rather than being
  ;; restated here, for the reason that file's own export comment gives: a second table
  ;; saying the same thing in its own words is how two files that must agree stop agreeing.
  ;; `make-fixture-dataset' passes MATRIX straight through to `cl-gbdt:make-dataset', so it
  ;; builds a dataset from a `csr-matrix' exactly as it does from a dense one -- which is
  ;; the property this whole file is about.
  (:import-from #:cl-gbdt/tests/functional/evaluation
                #:*fixtures*
                #:make-fixture-dataset))

(in-package #:cl-gbdt/tests/functional/sparse-input)

(defparameter *prediction-tolerance* 1d-9
  "How far two `cl-gbdt:predict' results may differ, element for element, and still count as
the same numbers in `sparse-and-dense-training-agree'.

Exact equality is what is actually expected: the dense and CSR ingestion paths hand the
library the same values, and both libraries train deterministically from a fixed dataset, so
the two boosters should be identical trees. The tolerance is here because that expectation
rests on the two C entry points accumulating in the same order, which neither library
documents, and a last-bit difference would be a true negative dressed as a failure. It is
small enough that nothing this test exists to catch survives it: a transposed index, a
dropped row or a matrix read as zeros moves a probability by far more than 1d-9.")

(defun dense-to-csr (matrix &key (num-columns (array-dimension matrix 1)))
  "Return MATRIX, a 2D `double-float' array, as a `cl-gbdt:csr-matrix' NUM-COLUMNS wide.

Every element of MATRIX is stored explicitly, zeros included, rather than only the non-zero
ones a CSR conversion usually keeps. The two libraries do not agree on what an ABSENT entry
means -- LightGBM reads one as 0.0 while its own `zero_as_missing' is off, XGBoost reads one
as missing -- so a conversion that dropped zeros would be describing different data to the
two backends, and `sparse-and-dense-training-agree' below would be asserting something
different on each. Storing every element leaves the `csr-matrix' and MATRIX describing the
same numbers on both, which is the property that test is about.

NUM-COLUMNS defaults to MATRIX's own width. A larger value declares trailing columns that
hold nothing at all -- no entry of INDICES names them -- which is what
`sparse-dataset-reports-the-declared-width' needs and where the absent-entry case is
covered instead.

This is the only place a `csr-matrix' is built from the suite's dense fixture, so the two
forms of the same data cannot drift apart."
  (let* ((rows (array-dimension matrix 0))
         (cols (array-dimension matrix 1))
         (stored (* rows cols))
         (indptr (make-array (1+ rows)))
         (indices (make-array stored))
         (values (make-array stored))
         (position 0))
    (dotimes (row rows)
      (setf (aref indptr row) position)
      (dotimes (col cols)
        (setf (aref indices position) col)
        (setf (aref values position) (aref matrix row col))
        (incf position)))
    (setf (aref indptr rows) position)
    (cl-gbdt:make-csr-matrix :indptr indptr :indices indices :values values
                             :num-columns num-columns)))

(defun single-column-matrix (values)
  "Return VALUES, a list of reals, as a `(simple-array double-float (N 1))'.

The `:reference' test below needs two datasets one feature wide whose values bin
differently, which is easier to state as two lists than as two 2D array literals."
  (let ((matrix (make-array (list (length values) 1) :element-type 'double-float)))
    (loop :for value :in values
          :for row :from 0
          :do (setf (aref matrix row 0) (coerce value 'double-float)))
    matrix))

(defun prediction-column (predictions column)
  "Return COLUMN of PREDICTIONS -- a `cl-gbdt:predict' result -- as a fresh `(simple-array
double-float (*))'.

`predictions-separate-p' takes a 1D sequence while `predict' returns a 2D array, one row per
input row and one column per class. Every fixture's objective is binary, so COLUMN is always
0, but the shape still has to be unpacked by hand."
  (let* ((rows (array-dimension predictions 0))
         (result (make-array rows :element-type 'double-float)))
    (dotimes (row rows result)
      (setf (aref result row) (aref predictions row column)))))

(defun predictions-agree-p (left right)
  "True when LEFT and RIGHT, two `cl-gbdt:predict' results, have the same shape and no pair
of corresponding elements differs by more than *PREDICTION-TOLERANCE*."
  (and (equal (array-dimensions left) (array-dimensions right))
       (loop :for index :below (array-total-size left)
             :always (<= (abs (- (row-major-aref left index) (row-major-aref right index)))
                         *prediction-tolerance*))))

(defun fixture-for (backend-name)
  "Return the *FIXTURES* entry for BACKEND-NAME."
  (find backend-name *fixtures* :key (lambda (fixture) (getf fixture :backend))))

;;; ---------------------------------------------------------------------------
;;; The capability this task ships
;;;
;;; Policy section 7 registers `:sparse-input' as a question `backend-supports-p' answers,
;;; and both backends now ingest a `csr-matrix'. A NIL answer would say, in that function's
;;; own words, that the feature is unavailable here and the operation signals
;;; `capability-unavailable' -- neither of which is true once this task lands.
;;;
;;; The capability assertion and a real sparse training run are deliberately in ONE test
;;; rather than two. This project has twice shipped a feature whose capability keyword stayed
;;; false, because the assertion that the feature works and the assertion that the backend
;;; admits to it lived in different tests and only the first was written. Tying them together
;;; means the capability cannot be forgotten without the working-feature assertion going with
;;; it.

(deftest sparse-input-capability-is-true-where-it-is-demonstrated
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend (dense-to-csr matrix)
                                                    label-vector))
                 (cl-gbdt:with-booster
                     (booster (cl-gbdt:train backend train-set :num-rounds 5
                                             :parameters (getf fixture :booster-parameters)))
                   (testing (format nil "~A: backend-supports-p reports :sparse-input"
                                    (getf fixture :backend))
                     (ok (eq t (cl-gbdt:backend-supports-p backend :sparse-input))
                         (format nil "the probed capabilities were ~S"
                                 (cl-gbdt:backend-capabilities backend))))
                   (testing (format nil "~A: a booster trained on the csr-matrix separates ~
                                         the two classes" (getf fixture :backend))
                     (let ((predictions (cl-gbdt:predict booster matrix)))
                       (ok (predictions-separate-p (prediction-column predictions 0)
                                                   label-vector)
                           (format nil "the predictions were ~S" predictions))))))
            (cl-gbdt:close-backend backend)))))))

;;; The assertion that proves the sparse path carries the data rather than merely accepting
;;; it. `dense-to-csr' describes exactly the same numbers as the dense fixture, so a model
;;; trained on either must be the same model. An implementation that transposed INDICES
;;; against INDPTR, dropped the last row by passing the wrong NINDPTR, or handed the library
;;; a buffer of zeros would satisfy every shape assertion in this file and fail here.
;;;
;;; Within one backend only, never across the two -- LightGBM and XGBoost train different
;;; models from the same rows by design, and policy section 13 forbids comparing their
;;; numbers to each other.

(deftest sparse-and-dense-training-agree
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (parameters (getf fixture :booster-parameters)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (dense-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (sparse-set (make-fixture-dataset fixture backend (dense-to-csr matrix)
                                                       label-vector))
                   (cl-gbdt:with-booster
                       (dense-booster (cl-gbdt:train backend dense-set :num-rounds 5
                                                     :parameters parameters))
                     (cl-gbdt:with-booster
                         (sparse-booster (cl-gbdt:train backend sparse-set :num-rounds 5
                                                        :parameters parameters))
                       (let ((dense (cl-gbdt:predict dense-booster matrix))
                             (sparse (cl-gbdt:predict sparse-booster matrix)))
                         (testing (format nil "~A: the sparsely-trained model predicts what ~
                                               the densely-trained one does"
                                          (getf fixture :backend))
                           (ok (predictions-agree-p dense sparse)
                               (format nil "dense ~S, sparse ~S" dense sparse)))
                         ;; Without this, two models that both predict a constant would
                         ;; agree perfectly and say nothing. The fixture is separable, so a
                         ;; model that learned anything at all orders the classes.
                         (testing (format nil "~A: both models learned something to agree ~
                                               about" (getf fixture :backend))
                           (ok (and (predictions-separate-p (prediction-column dense 0)
                                                            label-vector)
                                    (predictions-separate-p (prediction-column sparse 0)
                                                            label-vector))
                               (format nil "dense ~S, sparse ~S" dense sparse))))))))
            (cl-gbdt:close-backend backend)))))))

;;; NUM-COLUMNS is the matrix's DECLARED width, not the largest index it happens to store --
;;; see `cl-gbdt:make-csr-matrix''s own docstring for why the two are different facts. The
;;; fixture here is four columns wide with nothing stored in the last one, so a backend that
;;; inferred the width from INDICES would report 3 and fail. The row count comes from INDPTR
;;; the same way and is asserted alongside it.

(deftest sparse-dataset-reports-the-declared-width
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (rows (array-dimension matrix 0)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (dataset (make-fixture-dataset
                             fixture backend (dense-to-csr matrix :num-columns 4)
                             label-vector))
                 (testing (format nil "~A: dataset-num-features is the declared NUM-COLUMNS, ~
                                       not the largest stored index"
                                  (getf fixture :backend))
                   (ok (= 4 (cl-gbdt:dataset-num-features dataset))
                       (format nil "dataset-num-features was ~S"
                               (cl-gbdt:dataset-num-features dataset))))
                 (testing (format nil "~A: dataset-num-rows is INDPTR's row count"
                                  (getf fixture :backend))
                   (ok (= rows (cl-gbdt:dataset-num-rows dataset))
                       (format nil "dataset-num-rows was ~S"
                               (cl-gbdt:dataset-num-rows dataset)))))
            (cl-gbdt:close-backend backend)))))))

;;; Policy section 7's central rule: the operation re-checks the capability itself rather
;;; than trusting the caller to have asked `backend-supports-p' first, and signals rather
;;; than falling back to a dense conversion or anything else. Both vendored libraries do
;;; provide the CSR entry point -- the first test in this file asserts exactly that -- so the
;;; only way to reach this branch is to overwrite the probed plist, which is what a library
;;; too old to have the symbol would have produced at `open-backend'. This is the same way
;;; `xgboost-api-slice-model-signals-when-the-capability-is-absent' reaches its own gate, and
;;; the same way tests/backend.lisp drives the mock backend's capabilities. Without it the
;;; whole gate is a branch nobody has seen taken.
;;;
;;; `handler-case', not rove's `signals', which does not reliably catch a condition raised
;;; inside `restart-case'; the condition TYPE is asserted, not merely that something
;;; signalled.

(deftest sparse-input-without-the-capability-signals
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (declare (ignore label-vector))
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (progn
                 (setf (cl-gbdt:backend-capabilities backend) '(:sparse-input nil))
                 (testing (format nil "~A: make-dataset signals capability-unavailable, ~
                                       naming the capability and the backend"
                                  (getf fixture :backend))
                   (let ((condition (handler-case
                                        (progn (cl-gbdt:make-dataset backend
                                                                     (dense-to-csr matrix))
                                               nil)
                                      (cl-gbdt:capability-unavailable (c) c))))
                     (ok condition
                         "make-dataset signalled instead of building a dataset")
                     (ok (and condition
                              (eq :sparse-input
                                  (cl-gbdt:capability-unavailable-capability condition)))
                         (format nil "the condition named capability ~S"
                                 (and condition
                                      (cl-gbdt:capability-unavailable-capability
                                       condition))))
                     (ok (and condition
                              (eq (getf fixture :backend)
                                  (cl-gbdt:backend-error-backend condition)))
                         (format nil "the condition named backend ~S"
                                 (and condition
                                      (cl-gbdt:backend-error-backend condition)))))))
            (cl-gbdt:close-backend backend)))))))

;;; :REFERENCE and :PARAMETERS behave for a `csr-matrix' exactly as they do for a dense one,
;;; and nothing else in this file would catch a sparse path that accepted either and quietly
;;; dropped it on the floor.
;;;
;;; LightGBM only, and deliberately not weakened into something both backends can assert.
;;; XGBoost's fixture sets :ALIGNS-BIN-MAPPERS false because that backend signals
;;; `unsupported-argument' for both keywords rather than accepting them -- there is no
;;; bin-mapper alignment in XGBoost to align to, and `XGDMatrixCreateFromCSR''s config JSON
;;; recognizes none of LightGBM's dataset-level binning parameters. So there is nothing on
;;; that backend for this test to assert about honouring them, and asserting the weaker
;;; "both backends do whatever they do" on both would test nothing on either.
;;;
;;; The evidence for :REFERENCE is acceptance: `LGBM_BoosterAddValidData' refuses a
;;; validation set whose bin mapper differs from the training set's, and the two value lists
;;; below share no value, so they bin independently -- confirmed against the vendored library
;;; by tests/functional/lightgbm-api.lisp's own dense pair of the same two assertions, whose
;;; distributions these are. The refusal half is asserted here too rather than assumed, since
;;; without it the acceptance half could pass on a sparse path that ignored :REFERENCE
;;; entirely and simply happened to bin the two sets alike.
;;;
;;; The evidence for :PARAMETERS is a value the library itself rejects: LightGBM's config
;;; parser checks `max_bin > 1' while reading `LGBM_DatasetCreateFromCSR''s parameter string,
;;; so `:max-bin 1' fails dataset construction outright. A sparse path that dropped
;;; :PARAMETERS would build that dataset happily.

(defparameter *reference-training-values* '(0.0 5.0 0.0 5.0 0.0 5.0 0.0 5.0)
  "Two distinct feature values for the `:reference' test's training set, sharing none of
*REFERENCE-VALIDATION-VALUES*. Taken from tests/functional/lightgbm-api.lisp's dense
:reference tests, which established against the vendored library that these two
distributions bin independently.")

(defparameter *reference-training-labels*
  (make-array 8 :element-type 'single-float
                :initial-contents '(0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0))
  "Binary labels matching *REFERENCE-TRAINING-VALUES* row for row -- the fixture's \"binary\"
objective needs a label on the training set regardless of what this test is checking.")

(defparameter *reference-validation-values* '(0.3 1.1 1.7 2.3 2.9 3.5 4.1 4.9)
  "Feature values spread across the training set's range but sharing none of
*REFERENCE-TRAINING-VALUES*, so a dataset built from these bins independently unless
`:reference' aligns it with the training set.")

(deftest sparse-dataset-honours-reference-and-parameters
  (with-backend-library (:lightgbm)
    (let* ((fixture (fixture-for :lightgbm))
           (parameters (getf fixture :dataset-parameters))
           (booster-parameters (getf fixture :booster-parameters))
           (training-csr (dense-to-csr (single-column-matrix *reference-training-values*)))
           (validation-csr (dense-to-csr (single-column-matrix *reference-validation-values*)))
           (backend (cl-gbdt:open-backend :lightgbm)))
      (unwind-protect
           (cl-gbdt:with-dataset
               (train-set (cl-gbdt:make-dataset backend training-csr
                                                :label *reference-training-labels*
                                                :parameters parameters))
             (cl-gbdt:with-dataset
                 (aligned (cl-gbdt:make-dataset backend validation-csr
                                                :reference train-set
                                                :parameters parameters))
               (testing "train attaches a sparse valid set built with :reference"
                 (ok (handler-case
                         (cl-gbdt:with-booster
                             (booster (cl-gbdt:train backend train-set :num-rounds 1
                                                     :valid-sets (list aligned)
                                                     :parameters booster-parameters))
                           (and booster t))
                       (cl-gbdt:foreign-call-error () nil))
                     "train accepted the :reference-built sparse valid set")))
             (cl-gbdt:with-dataset
                 (independent (cl-gbdt:make-dataset backend validation-csr
                                                    :parameters parameters))
               ;; The control for the assertion above: these two sets really do bin
               ;; differently, so acceptance there was :REFERENCE doing its job.
               (testing "train refuses a sparse valid set with its own bin mapper"
                 (ok (handler-case
                         (progn (cl-gbdt:train backend train-set :num-rounds 1
                                               :valid-sets (list independent)
                                               :parameters booster-parameters)
                                nil)
                       (cl-gbdt:foreign-call-error () t))
                     "train refused the independently-binned sparse valid set")))
             (testing ":parameters reaches the library, which rejects max_bin=1"
               (ok (handler-case
                       (progn (cl-gbdt:free-dataset
                               (cl-gbdt:make-dataset backend training-csr
                                                     :label *reference-training-labels*
                                                     :parameters '(:max-bin 1 :verbose -1)))
                              nil)
                     (cl-gbdt:foreign-call-error () t))
                   "make-dataset signalled foreign-call-error for max_bin=1")))
        (cl-gbdt:close-backend backend)))))
