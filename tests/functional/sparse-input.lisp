;;;; sparse-input.lisp --- Portable contract tests for `make-dataset' and `predict' on a
;;;; `csr-matrix'.
;;;;
;;;; `cl-gbdt:make-dataset' and `cl-gbdt:predict' both now accept a `cl-gbdt:csr-matrix'
;;;; wherever they accepted a dense matrix, on both backends, gated by the `:sparse-input'
;;;; capability. Like tests/functional/evaluation.lisp and
;;;; tests/functional/training-report.lisp beside it, every test below runs the identical
;;;; assertions over that first file's *FIXTURES*, once per backend, so the two backends
;;;; cannot drift apart in shape or meaning without one of them failing here. There are three
;;;; exceptions, each with its own comment: `sparse-dataset-honours-reference-and-parameters'
;;;; needs a backend that aligns bin mappers and so runs on LightGBM alone,
;;;; `sparse-prediction-kind-support-is-what-each-library-offers' asserts a DIFFERENT measured
;;;; fact per backend because the two libraries' sparse prediction entry points genuinely
;;;; cover different KINDs, and `an-omitted-entry-is-zero-to-lightgbm-and-missing-to-xgboost'
;;;; does the same for the one thing the two libraries read differently about a `csr-matrix'
;;;; itself -- an entry it does not store.
;;;;
;;;; Numbers are never compared BETWEEN backends: policy section 13 asks for shape, order and
;;;; meaning, not numeric agreement, and the two libraries train different models from the
;;;; same rows by design. `sparse-and-dense-training-agree' and
;;;; `sparse-and-dense-prediction-agree' do compare numbers, but only a backend's own sparse
;;;; result against its own dense one -- the comparison is within one backend, never across
;;;; the two.
;;;;
;;;; The ingestion tests in the first half of this file predict on a DENSE matrix even when
;;;; the booster was trained sparsely. That is deliberate and stays that way: it keeps each of
;;;; those tests about the one path it is named for, so a prediction-side regression cannot
;;;; make an ingestion test fail (or, worse, cancel out and let it pass).

(uiop:define-package #:cl-gbdt/tests/functional/sparse-input
  ;; Zero symbols: every reference below is package-qualified. Declared so this file's
  ;; dependency on the unified API is explicit rather than inherited, matching the identical
  ;; clause in evaluation.lisp and training-report.lisp.
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt)
  ;; Zero symbols, both of them: they run at load time to register :lightgbm and :xgboost
  ;; with `open-backend' and to define each backend's methods on `cl-gbdt''s generics.
  ;; Without these clauses, package-inferred-system has no edge to those files and
  ;; `(cl-gbdt:open-backend :lightgbm)' below would signal `unknown-backend'. `unified'
  ;; rather than `all' since the Layer 1 split: `all' is Layer 1 alone now, carrying the
  ;; `register-backend' call but not the protocol methods, so `train' below would find no
  ;; applicable method.
  (:import-from #:cl-gbdt/src/lightgbm/unified)
  (:import-from #:cl-gbdt/src/xgboost/unified)
  ;; `dense-to-csr' stores every element of the matrix it converts, zeros included. What this
  ;; file needs that for: `sparse-and-dense-training-agree' below compares a sparsely-trained
  ;; model against a densely-trained one, so a conversion that dropped zeros would be
  ;; describing different data to the two backends -- 0.0 to LightGBM, missing to XGBoost --
  ;; and that test would be asserting something different on each.
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library
                #:make-separable-dataset
                #:predictions-separate-p
                #:*prediction-tolerance*
                #:predictions-agree-p
                #:dense-to-csr)
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

(defun fixture-for (backend-name)
  "Return the *FIXTURES* entry for BACKEND-NAME."
  (find backend-name *fixtures* :key (lambda (fixture) (getf fixture :backend))))

;;; ---------------------------------------------------------------------------
;;; The shared helper every comparison below rests on
;;;
;;; `predictions-agree-p' SIGNALS on a shape mismatch rather than answering NIL -- see its own
;;; docstring in tests/functional/support.lisp for why. Four assertions in this file read
;;; `(not (predictions-agree-p ...))', and a NIL answer satisfies every one of them without
;;; comparing a single number. Nothing pinned that branch, so deleting it -- a plausible
;;; "simplification" back to `(and (equal dims) ...)' -- would restore that false pass. This is
;;; the pin: measured with the `unless' deleted from `predictions-agree-p', the two mismatch
;;; assertions below go red, the control below them stays green, and no other file in layer 2
;;; changes.
;;;
;;; It lives here rather than in support.lisp because that file defines no test of its own,
;;; and adding one there would move rove's per-file test count. This file was chosen among the
;;; three that call the helper because its own comparisons are the shape-varied ones: :KIND
;;; :RAW against :NORMAL, a limited :NUM-ITERATION against an unlimited one, a dense matrix
;;; against two different CSR forms of it.
;;;
;;; No `with-backend-library', and no booster: the helper is pure Lisp over two arrays, so two
;;; `double-float' arrays of different shapes are the whole input and no shared library is
;;; reached. `handler-case', not rove's `signals', which does not reliably catch a condition
;;; raised inside `restart-case'; the condition TYPE is asserted, not merely that something
;;; signalled. That type is `simple-error', which is what the helper's bare `(error "...")'
;;; signals -- measured.
;;;
;;; The argument ORDER in both mismatch calls is load-bearing: the SMALLER array is LEFT. The
;;; loop under the guard runs to `(array-total-size left)' and indexes RIGHT with that same
;;; index, so with the guard deleted a smaller LEFT stays in bounds on RIGHT, the loop finds
;;; every pair equal, and the function returns T -- a clean FAILED assertion. Reversed, it
;;; would index past RIGHT's end and raise `sb-int:invalid-array-index-error', which
;;; `(simple-error () t)' does not catch -- measured: that condition is a subtype of
;;; `type-error' and not of `simple-error' -- so the test would ERROR instead of failing, and
;;; pin the guard far less legibly.

(deftest predictions-agree-p-signals-on-a-shape-mismatch
  (let ((two-by-one (make-array '(2 1) :element-type 'double-float :initial-element 0d0))
        (two-by-two (make-array '(2 2) :element-type 'double-float :initial-element 0d0))
        (three-by-one (make-array '(3 1) :element-type 'double-float :initial-element 0d0)))
    (testing "a differing column count signals rather than answering NIL"
      (ok (handler-case (progn (predictions-agree-p two-by-one two-by-two) nil)
            (simple-error () t))
          "whether comparing a (2 1) result against a (2 2) one signalled"))
    (testing "a differing row count signals too"
      (ok (handler-case (progn (predictions-agree-p two-by-one three-by-one) nil)
            (simple-error () t))
          "whether comparing a (2 1) result against a (3 1) one signalled"))
    ;; The control: the guard rejects mismatched shapes only, not every call. Without it, a
    ;; helper that signalled unconditionally would pass both assertions above.
    (testing "two results of the same shape still compare normally"
      (ok (predictions-agree-p two-by-one
                               (make-array '(2 1) :element-type 'double-float
                                                  :initial-element 0d0))
          "whether two identical (2 1) results still agree"))))

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
;;; That rests on `dense-to-csr' storing EVERY element, zeros included, and this is the test
;;; that needs it: the fixture's single zero is element [0][0], and an entry a `csr-matrix'
;;; does not store is 0.0 to LightGBM but MISSING to XGBoost. A conversion that dropped it
;;; would leave the sparse arm trained on different data from the dense arm -- and on
;;; DIFFERENTLY different data per backend, so this test would be asserting something other
;;; than what it says, and something other than that on each.
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
                   ;; `free-dataset' on the success branch, the same way the :parameters
                   ;; assertion in `sparse-dataset-honours-reference-and-parameters' frees
                   ;; the dataset it does not expect to get: this branch is only reached if
                   ;; the gate has regressed, and a leaked handle is a poor second failure
                   ;; to hand whoever is already reading the first one.
                   (let ((condition (handler-case
                                        (progn (cl-gbdt:free-dataset
                                                (cl-gbdt:make-dataset backend
                                                                      (dense-to-csr matrix)))
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
                 ;; `with-booster' on the success branch, as the assertion above already
                 ;; uses inside its own `handler-case': this branch is only reached if the
                 ;; refusal has regressed, and the booster `train' would then have returned
                 ;; is the one handle in this file with nothing else to free it.
                 (ok (handler-case
                         (cl-gbdt:with-booster
                             (booster (cl-gbdt:train backend train-set :num-rounds 1
                                                     :valid-sets (list independent)
                                                     :parameters booster-parameters))
                           (declare (ignorable booster))
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

;;; ---------------------------------------------------------------------------
;;; Prediction from a `csr-matrix'
;;;
;;; Everything below trains DENSELY and predicts both ways, which is the opposite of the
;;; ingestion tests above and is the point: a model built through a path this task did not
;;; touch is the fixed reference the two prediction paths are compared against, so what these
;;; tests can fail on is `predict' alone. `sparse-and-dense-training-agree' above already
;;; covers the other direction.

(defparameter *training-rounds* 5
  "How many boosting rounds `train-dense-booster' runs.

Named rather than written into that function, because
`sparse-prediction-honours-num-iteration' needs a round count STRICTLY BELOW it to ask for,
and a test asking for 2 rounds out of a booster that turned out to hold only 2 would be
asserting an equality that holds for the wrong reason.")

(defun train-dense-booster (fixture backend matrix label-vector)
  "Train a booster on BACKEND from the dense MATRIX and LABEL-VECTOR for *TRAINING-ROUNDS*
rounds, using FIXTURE's own parameters, and return it. The caller frees it --
`cl-gbdt:with-booster' is the usual way.

The dataset is freed before this returns, since none of the prediction tests below needs it
and every one of them would otherwise have to nest a `cl-gbdt:with-dataset' it never
mentions again. A booster outlives the dataset it was trained on for prediction purposes on
both backends -- `predict' consults neither backend's retained training-set handle."
  (cl-gbdt:with-dataset (train-set (make-fixture-dataset fixture backend matrix label-vector))
    (cl-gbdt:train backend train-set :num-rounds *training-rounds*
                   :parameters (getf fixture :booster-parameters))))

;;; The assertion that proves the sparse prediction path carries the data rather than merely
;;; accepting it. `dense-to-csr' describes exactly the same numbers as the dense fixture, and
;;; the booster is one and the same object, so the two calls must produce one and the same
;;; result. An implementation that transposed INDICES against INDPTR, passed the wrong NINDPTR
;;; and dropped the last row, or handed the library a buffer of zeros would satisfy every
;;; shape assertion in this file and fail here.
;;;
;;; Within one backend only, never across the two -- see this file's header.

(deftest sparse-and-dense-prediction-agree
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-booster
                   (booster (train-dense-booster fixture backend matrix label-vector))
                 (let ((dense (cl-gbdt:predict booster matrix))
                       (sparse (cl-gbdt:predict booster (dense-to-csr matrix))))
                   (testing (format nil "~A: predict on the csr-matrix answers what predict ~
                                         on the equivalent dense matrix does"
                                    (getf fixture :backend))
                     (ok (predictions-agree-p dense sparse)
                         (format nil "dense ~S, sparse ~S" dense sparse)))
                   ;; Without this, a `predict' that returned a constant either way would
                   ;; agree perfectly and say nothing. The fixture is separable, so a model
                   ;; that learned anything at all orders the two classes.
                   (testing (format nil "~A: the sparse result orders the two classes, so ~
                                         the agreement is about something"
                                    (getf fixture :backend))
                     (ok (predictions-separate-p (prediction-column sparse 0) label-vector)
                         (format nil "the sparse predictions were ~S" sparse)))))
            (cl-gbdt:close-backend backend)))))))

;;; KIND reaches the library on the sparse path exactly as it does on the dense one, rather
;;; than the sparse branch being special-cased to `predict''s default. `:raw' is the KIND both
;;; backends' sparse entry points support (see
;;; `sparse-prediction-kind-support-is-what-each-library-offers' below for the two they do not
;;; agree on), and it is also the one whose result is unmistakably different from `:normal''s:
;;; both fixtures' objectives are logistic, so a raw score is the log-odds behind the
;;; probability `:normal' returns. The second assertion is the control -- without it, a sparse
;;; path that ignored KIND and always predicted `:normal' would satisfy the first one, since
;;; the dense side would then be compared against a matching mistake only if the dense path
;;; broke too.

(deftest sparse-prediction-honours-kind
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-booster
                   (booster (train-dense-booster fixture backend matrix label-vector))
                 (let* ((csr (dense-to-csr matrix))
                        (dense-raw (cl-gbdt:predict booster matrix :kind :raw))
                        (sparse-raw (cl-gbdt:predict booster csr :kind :raw))
                        (sparse-normal (cl-gbdt:predict booster csr :kind :normal)))
                   (testing (format nil "~A: :kind :raw on the csr-matrix answers what ~
                                         :kind :raw on the dense matrix does"
                                    (getf fixture :backend))
                     (ok (predictions-agree-p dense-raw sparse-raw)
                         (format nil "dense ~S, sparse ~S" dense-raw sparse-raw)))
                   (testing (format nil "~A: :raw and :normal on the same csr-matrix are ~
                                         different numbers, so KIND reached the library"
                                    (getf fixture :backend))
                     (ok (not (predictions-agree-p sparse-raw sparse-normal))
                         (format nil ":raw ~S, :normal ~S" sparse-raw sparse-normal)))))
            (cl-gbdt:close-backend backend)))))))

;;; NUM-ITERATION reaches the library on the sparse path, and reaches it as the ROUND LIMIT
;;; rather than as anything else.
;;;
;;; Nothing else in this file would catch that. Every other test here predicts with
;;; NUM-ITERATION defaulted, which both backends resolve to the wire value 0 -- and 0 is also
;;; what each passes as the start of the range. LightGBM's `%predict-for-csr' spells that pair
;;; out positionally, as `predict_type, start_iteration, num_iteration', a hand-written
;;; argument list unique to that function; XGBoost's reaches `"iteration_begin"' and
;;; `"iteration_end"' in a config string. With both halves 0, transposing them is invisible.
;;;
;;; Measured against both vendored libraries before this test was written, so the equality
;;; below is not vacuous and the transposition really is caught. On the eight-row fixture at
;;; five rounds, row 0's prediction is:
;;;
;;;   LightGBM  0.40567521221114555 for (start 0, num 2), 0.38338210973810 for (start 2, num 0)
;;;   XGBoost   0.29127201437950134 for (begin 0, end 2), 0.30579683 for (begin 2, end 0)
;;;
;;; Both gaps are seven orders of magnitude above *PREDICTION-TOLERANCE*. Neither fixture has
;;; converged by round 2 either -- both backends' predictions still move at every round from 1
;;; to *TRAINING-ROUNDS* -- which is what the second assertion pins, so a `predict' that
;;; ignored NUM-ITERATION entirely and always used every round could not pass the first one by
;;; the two answers happening to coincide.

(deftest sparse-prediction-honours-num-iteration
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (rounds 2))
          (unwind-protect
               (cl-gbdt:with-booster
                   (booster (train-dense-booster fixture backend matrix label-vector))
                 (let* ((csr (dense-to-csr matrix))
                        (dense-limited (cl-gbdt:predict booster matrix
                                                        :num-iteration rounds))
                        (sparse-limited (cl-gbdt:predict booster csr :num-iteration rounds))
                        (sparse-all (cl-gbdt:predict booster csr)))
                   (testing (format nil "~A: :num-iteration ~D on the csr-matrix answers ~
                                         what it answers on the dense matrix"
                                    (getf fixture :backend) rounds)
                     (ok (predictions-agree-p dense-limited sparse-limited)
                         (format nil "dense ~S, sparse ~S" dense-limited sparse-limited)))
                   ;; The control. Without it the assertion above would still pass on a
                   ;; sparse path that dropped NUM-ITERATION on the floor, since the dense
                   ;; side would then be the only one honouring it -- and it would pass
                   ;; vacuously on a fixture whose model stopped changing before round 2.
                   (testing (format nil "~A: ~D rounds is a real limit -- it differs from ~
                                         all ~D" (getf fixture :backend) rounds
                                    *training-rounds*)
                     (ok (not (predictions-agree-p sparse-limited sparse-all))
                         (format nil "~D rounds ~S, all rounds ~S"
                                 rounds sparse-limited sparse-all)))))
            (cl-gbdt:close-backend backend)))))))

;;; Policy section 7's central rule, applied to `predict' this time: the operation re-checks
;;; the capability itself rather than trusting the caller to have asked `backend-supports-p'
;;; first, and signals rather than falling back to a dense conversion or anything else. The
;;; false answer is produced the same way `sparse-input-without-the-capability-signals' above
;;; produces it -- by overwriting the probed plist, which is what a library too old to have
;;; the symbol would have produced at `open-backend' -- for the same reason: both vendored
;;; libraries do provide the CSR entry points, so this branch is otherwise one nobody has seen
;;; taken.
;;;
;;; The plist is overwritten AFTER the booster is trained, so what this exercises is
;;; `predict''s own gate and not `make-dataset''s: the training set was built from a dense
;;; matrix and never needed the capability at all.
;;;
;;; `handler-case', not rove's `signals', which does not reliably catch a condition raised
;;; inside `restart-case'; the condition TYPE is asserted, not merely that something signalled.

(deftest sparse-prediction-without-the-capability-signals
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-booster
                   (booster (train-dense-booster fixture backend matrix label-vector))
                 (setf (cl-gbdt:backend-capabilities backend) '(:sparse-input nil))
                 (testing (format nil "~A: predict signals capability-unavailable, naming ~
                                       the capability and the backend"
                                  (getf fixture :backend))
                   (let ((condition (handler-case
                                        (progn (cl-gbdt:predict booster (dense-to-csr matrix))
                                               nil)
                                      (cl-gbdt:capability-unavailable (c) c))))
                     (ok condition
                         "predict signalled instead of returning predictions")
                     (ok (and condition
                              (eq :sparse-input
                                  (cl-gbdt:capability-unavailable-capability condition)))
                         (format nil "the condition named capability ~S"
                                 (and condition
                                      (cl-gbdt:capability-unavailable-capability condition))))
                     (ok (and condition
                              (eq (getf fixture :backend)
                                  (cl-gbdt:backend-error-backend condition)))
                         (format nil "the condition named backend ~S"
                                 (and condition
                                      (cl-gbdt:backend-error-backend condition)))))))
            (cl-gbdt:close-backend backend)))))))

;;; A `csr-matrix' whose NUM-COLUMNS is not the model's own feature count is the library's
;;; mistake to catch, not this wrapper's: nothing here pre-empts a consistency check the
;;; library already makes, so the assertion is that the caller gets a typed `cl-gbdt'
;;; condition rather than a crash or a silently wrong answer.
;;;
;;; Measured against both vendored libraries before this test was written, in both directions
;;; and on both backends -- all four refuse with a clean nonzero return, which `check-lgbm' /
;;; `check-xgb' turn into `foreign-call-error':
;;;
;;;   LightGBM  "The number of features in data (N) is not the same as it was in training
;;;             data (3)."
;;;   XGBoost   "Check failed: n_features_data == n_features_model (N vs. 3) : Number of
;;;             columns in data must equal to the trained model."
;;;
;;; Both directions are asserted, not just one, because each library reports EQUALITY rather
;;; than a bound: a path that only checked "at least as many features as the model" would pass
;;; the too-wide half and fail the too-narrow one. The too-wide matrix declares four columns
;;; over the three-column fixture with nothing stored in the last, which is the same shape
;;; `sparse-dataset-reports-the-declared-width' builds -- so this also pins that a width both
;;; backends happily INGEST is still refused at prediction time against a model of a different
;;; width, rather than the two facts being confused for each other.

(deftest sparse-prediction-with-the-wrong-width-is-refused-by-the-library
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        ;; NARROW is built BEFORE the backend is opened -- `let' evaluates its init forms
        ;; left to right, so the order below is the guarantee. Bound after `open-backend' it
        ;; would still be outside the `unwind-protect', so a `dense-to-csr' that signalled --
        ;; which is what a regression in `make-csr-matrix''s validation would do -- would
        ;; leak an open backend.
        (let ((narrow (dense-to-csr (make-separable-dataset :cols 2)))
              (backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-booster
                   (booster (train-dense-booster fixture backend matrix label-vector))
                 (testing (format nil "~A: a csr-matrix wider than the model is refused"
                                  (getf fixture :backend))
                   (ok (handler-case
                           (progn (cl-gbdt:predict booster (dense-to-csr matrix
                                                                         :num-columns 4))
                                  nil)
                         (cl-gbdt:foreign-call-error () t))
                       "predict signalled foreign-call-error for a four-column matrix"))
                 (testing (format nil "~A: a csr-matrix narrower than the model is refused"
                                  (getf fixture :backend))
                   (ok (handler-case
                           (progn (cl-gbdt:predict booster narrow) nil)
                         (cl-gbdt:foreign-call-error () t))
                       "predict signalled foreign-call-error for a two-column matrix")))
            (cl-gbdt:close-backend backend)))))))

;;; The one measured asymmetry this task introduces, asserted rather than papered over.
;;;
;;; `LGBM_BoosterPredictForCSR' is LightGBM's ordinary prediction entry point in its CSR
;;; spelling and honours all four of the protocol's KINDs, exactly as
;;; `LGBM_BoosterPredictForMat' does. `XGBoosterPredictFromCSR' is not the CSR spelling of
;;; `XGBoosterPredictFromDMatrix': it is XGBoost's INPLACE prediction, a different code path
;;; whose header documents it as such, and it supports only `:normal' and `:raw' -- measured
;;; against the vendored library, which refuses the other two with "Unsupported prediction
;;; type:2" (`:contrib') and "Unsupported prediction type:6" (`:leaf-index'), a clean nonzero
;;; return that reaches the caller as `foreign-call-error'.
;;;
;;; This is asserted per backend rather than weakened into something both can satisfy, for the
;;; same reason `sparse-dataset-honours-reference-and-parameters' above runs on one backend:
;;; "each backend does whatever it does" would test nothing on either. It is also not a number
;;; compared between backends -- each half asserts only what its own library did.
;;;
;;; The XGBoost half is a real contract, not a lament: what the caller must be able to rely on
;;; is that an unsupported KIND on the sparse path is REFUSED, with a typed condition naming
;;; the failed call, rather than silently answering with some other KIND's numbers. That is
;;; what is asserted. If a future XGBoost lifts the restriction, this test failing is the
;;; correct way to find out.

(deftest sparse-prediction-kind-support-is-what-each-library-offers
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((fixture (fixture-for :lightgbm))
            (backend (cl-gbdt:open-backend :lightgbm)))
        (unwind-protect
             (cl-gbdt:with-booster
                 (booster (train-dense-booster fixture backend matrix label-vector))
               (let ((csr (dense-to-csr matrix)))
                 (dolist (kind '(:contrib :leaf-index))
                   (testing (format nil "lightgbm: :kind ~S on a csr-matrix answers what it ~
                                         answers on the dense matrix" kind)
                     (let ((dense (cl-gbdt:predict booster matrix :kind kind))
                           (sparse (cl-gbdt:predict booster csr :kind kind)))
                       (ok (predictions-agree-p dense sparse)
                           (format nil "dense ~S, sparse ~S" dense sparse)))))))
          (cl-gbdt:close-backend backend)))))
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((fixture (fixture-for :xgboost))
            (backend (cl-gbdt:open-backend :xgboost)))
        (unwind-protect
             (cl-gbdt:with-booster
                 (booster (train-dense-booster fixture backend matrix label-vector))
               (let ((csr (dense-to-csr matrix)))
                 (dolist (kind '(:contrib :leaf-index))
                   (testing (format nil "xgboost: :kind ~S on a csr-matrix is refused by ~
                                         inplace prediction, not silently answered" kind)
                     (ok (handler-case
                             (progn (cl-gbdt:predict booster csr :kind kind) nil)
                           (cl-gbdt:foreign-call-error () t))
                         (format nil "predict signalled foreign-call-error for :kind ~S"
                                 kind))))
                 ;; The control: the same two KINDs do work on the DENSE path, so what the
                 ;; assertions above pin is the entry point's own coverage and not a KIND
                 ;; this backend cannot serve at all.
                 (dolist (kind '(:contrib :leaf-index))
                   (testing (format nil "xgboost: :kind ~S still works on the dense path"
                                    kind)
                     (ok (arrayp (cl-gbdt:predict booster matrix :kind kind))
                         (format nil "dense :kind ~S returned an array" kind))))))
          (cl-gbdt:close-backend backend))))))

;;; ---------------------------------------------------------------------------
;;; A genuinely sparse matrix: entries omitted, and one row storing nothing at all
;;;
;;; Everything above this line converts through `dense-to-csr', which stores every element.
;;; That is right for what those tests assert -- a matrix meaning the same thing on both
;;; backends -- but it makes every fixture above STRUCTURALLY DENSE: INDPTR is an arithmetic
;;; sequence and INDICES the cycle 0,1,...,NCOL-1 repeated once per row. Nothing above would
;;; fail against an implementation that ignored both and read VALUES row-major, and nothing
;;; above sends either library the one thing a sparse format exists for: an entry that is
;;; not there.
;;;
;;; The fixture below does. It is the same shape of demonstration
;;; docs/user-guide/data-and-prediction.md's "An absent entry is not a zero" section carries,
;;; run as assertions.

(defparameter *omitted-entry-rows*
  '((0 0 0) (0 1 0) (0 0 2) (0 3 3)
    (5 0 0) (5 1 0) (5 0 2) (5 3 3))
  "Eight rows of three integer columns, coerced to `double-float' where they are used.

Purpose-built, and deliberately not `make-separable-dataset''s fixture, whose only zero is a
single element. Here ROW 0 IS ENTIRELY ZERO, so a CSR that omits zeros stores nothing at all
for it -- the empty row, a repeated INDPTR entry, that no other test in this suite sends to
either library -- and every remaining row omits at least one entry, so INDPTR is not an
arithmetic sequence and INDICES is not a repeating cycle.

Column 0 is the one carrying the class: 0.0 for the four rows labelled 0 and 5.0 for the
four labelled 1. Omitting zeros therefore takes column 0 away from exactly the negative-class
rows, which is what makes the two libraries' readings of an absent entry produce visibly
different numbers rather than the same ones by luck. Columns 1 and 2 carry no class
information whatever, by construction: the two halves hold identical values there.")

(defparameter *omitted-entry-labels*
  (make-array 8 :element-type 'single-float
                :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0))
  "Labels for *OMITTED-ENTRY-ROWS*, row for row: the first four rows are the negative class
and the last four the positive one, which is exactly what column 0 encodes.")

(defun omitted-entry-dense ()
  "Return *OMITTED-ENTRY-ROWS* as a `(simple-array double-float (8 3))'.

The fixed reference both CSR forms below are compared against. It stores every element, so
it means the same thing to both libraries and nothing about it depends on how either reads
an entry that is absent."
  (let* ((rows (length *omitted-entry-rows*))
         (columns (length (first *omitted-entry-rows*)))
         (matrix (make-array (list rows columns) :element-type 'double-float)))
    (loop :for row :in *omitted-entry-rows*
          :for i :from 0
          :do (loop :for value :in row
                    :for j :from 0
                    :do (setf (aref matrix i j) (coerce value 'double-float))))
    matrix))

(defun omitted-entry-csr (&key (omit-zeros t))
  "Return *OMITTED-ENTRY-ROWS* as a `cl-gbdt:csr-matrix'.

With OMIT-ZEROS true, the default, only the non-zero elements are stored -- the conversion a
sparse format normally performs, and the one that leaves row 0 storing nothing. With it
false the same eight rows are stored element for element, zeros included, which is what
`dense-to-csr' does for the rest of this file.

The pair is the point: the two forms describe the same eight rows to a reader that takes an
absent entry for 0.0, and two different matrices to one that takes it for missing."
  (let ((indptr (list 0))
        (indices '())
        (values '())
        (stored 0))
    (dolist (row *omitted-entry-rows*)
      (loop :for value :in row
            :for column :from 0
            :do (unless (and omit-zeros (zerop value))
                  (push column indices)
                  (push (coerce value 'double-float) values)
                  (incf stored)))
      (push stored indptr))
    (cl-gbdt:make-csr-matrix :indptr (nreverse indptr) :indices (nreverse indices)
                             :values (nreverse values)
                             :num-columns (length (first *omitted-entry-rows*)))))

(defun predictions-after-training-on (fixture backend matrix)
  "Train a booster on BACKEND from MATRIX -- one of the three forms of the omitted-entry
fixture -- for *TRAINING-ROUNDS* rounds with FIXTURE's own parameters, and return its
predictions on the DENSE form of those same eight rows.

Predicting on the dense form whatever MATRIX was is what makes two such results comparable:
what then differs between them is the data each model was TRAINED on, not the data each was
asked about."
  (cl-gbdt:with-booster
      (booster (cl-gbdt:with-dataset
                   (dataset (make-fixture-dataset fixture backend matrix
                                                  *omitted-entry-labels*))
                 (cl-gbdt:train backend dataset :num-rounds *training-rounds*
                                :parameters (getf fixture :booster-parameters))))
    (cl-gbdt:predict booster (omitted-entry-dense))))

(defun every-prediction-equals-p (predictions value)
  "True when every element of PREDICTIONS is within *PREDICTION-TOLERANCE* of VALUE."
  (loop :for index :below (array-total-size predictions)
        :always (<= (abs (- (row-major-aref predictions index) value))
                    *prediction-tolerance*)))

;;; The shape half, and the only thing about an omitted entry both backends agree on: the row
;;; count comes from INDPTR and the width from NUM-COLUMNS, neither from what is stored. Row 0
;;; stores nothing and is still a row; column 0 is stored for only four of the eight rows and
;;; the matrix is still three wide. An implementation that read the twelve stored VALUES
;;; row-major would report four rows here and pass every other test in this file.

(deftest omitted-entries-and-an-empty-row-keep-the-declared-shape
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      ;; The csr-matrix is built before the backend is opened -- `let' evaluates its init
      ;; forms left to right -- so a `make-csr-matrix' that signalled could not leak an open
      ;; backend past the `unwind-protect' below.
      (let ((sparse (omitted-entry-csr))
            (rows (length *omitted-entry-rows*))
            (columns (length (first *omitted-entry-rows*)))
            (backend (cl-gbdt:open-backend (getf fixture :backend))))
        (unwind-protect
             (progn
               (cl-gbdt:with-dataset
                   (dataset (make-fixture-dataset fixture backend sparse
                                                  *omitted-entry-labels*))
                 (testing (format nil "~A: make-dataset reports INDPTR's row count even ~
                                       though one row stores nothing"
                                  (getf fixture :backend))
                   (ok (= rows (cl-gbdt:dataset-num-rows dataset))
                       (format nil "dataset-num-rows was ~S"
                               (cl-gbdt:dataset-num-rows dataset))))
                 (testing (format nil "~A: make-dataset reports the declared width even ~
                                       though column 0 is stored for half the rows"
                                  (getf fixture :backend))
                   (ok (= columns (cl-gbdt:dataset-num-features dataset))
                       (format nil "dataset-num-features was ~S"
                               (cl-gbdt:dataset-num-features dataset)))))
               (cl-gbdt:with-booster
                   (booster (train-dense-booster fixture backend (omitted-entry-dense)
                                                 *omitted-entry-labels*))
                 (testing (format nil "~A: predict answers one row per INDPTR row, the ~
                                       empty one included" (getf fixture :backend))
                   (let ((predictions (cl-gbdt:predict booster sparse)))
                     (ok (= rows (array-dimension predictions 0))
                         (format nil "the predictions were ~S" predictions))))))
          (cl-gbdt:close-backend backend))))))

;;; What an omitted entry MEANS, asserted per backend because the two libraries genuinely
;;; disagree -- the same reason `sparse-prediction-kind-support-is-what-each-library-offers'
;;; above is written this way. Weakening it into something both could satisfy would assert
;;; nothing on either, and this is the one difference between the backends that changes
;;; numbers without signalling.
;;;
;;; Measured against both vendored libraries before either half was written, one booster
;;; trained on the dense fixture and asked about all three forms of the same eight rows
;;; (column 0 of each result, five rounds):
;;;
;;;   LightGBM  dense           0.297948 x4, 0.702052 x4
;;;             every element   0.297948 x4, 0.702052 x4   (identical to dense)
;;;             zeros omitted   0.297948 x4, 0.702052 x4   (identical to dense)
;;;   XGBoost   dense           0.153286 x4, 0.846714 x4
;;;             every element   0.153286 x4, 0.846714 x4   (identical to dense)
;;;             zeros omitted   0.846714 x8                (all eight rows, not four)
;;;
;;; LightGBM reads an absent entry as 0.0 -- its own `zero_as_missing' is off by default --
;;; so omitting the zeros described the same matrix and the numbers did not move at all.
;;; XGBoost reads one as missing, so the four negative-class rows lost column 0 entirely and
;;; followed the model's default direction, landing on the value a row whose column 0 is
;;; PRESENT and positive gets. That last equality is within one booster, so it is exact
;;; rather than approximate, and it is the positive statement of "missing" that a bare "the
;;; two differ" would not make. No number here is compared with a number from the other
;;; backend.

(deftest an-omitted-entry-is-zero-to-lightgbm-and-missing-to-xgboost
  (with-backend-library (:lightgbm)
    ;; The three matrices are built before the backend is opened, for the reason
    ;; `omitted-entries-and-an-empty-row-keep-the-declared-shape' above gives.
    (let ((complete (omitted-entry-csr :omit-zeros nil))
          (sparse (omitted-entry-csr))
          (dense (omitted-entry-dense))
          (fixture (fixture-for :lightgbm))
          (backend (cl-gbdt:open-backend :lightgbm)))
      (unwind-protect
           (progn
             (cl-gbdt:with-booster
                 (booster (train-dense-booster fixture backend dense
                                               *omitted-entry-labels*))
               (let ((from-dense (cl-gbdt:predict booster dense))
                     (from-complete (cl-gbdt:predict booster complete))
                     (from-sparse (cl-gbdt:predict booster sparse)))
                 (testing (format nil "lightgbm: predicting on the matrix with the zeros ~
                                       omitted answers what the dense matrix does -- an ~
                                       absent entry is 0.0")
                   (ok (predictions-agree-p from-dense from-sparse)
                       (format nil "dense ~S, omitted ~S" from-dense from-sparse)))
                 ;; The control: the CSR path itself is faithful, so the agreement above is
                 ;; about how absence is read and not about CSR having been read correctly.
                 (testing "lightgbm: so does the matrix storing every element"
                   (ok (predictions-agree-p from-dense from-complete)
                       (format nil "dense ~S, complete ~S" from-dense from-complete)))))
             (let ((from-complete (predictions-after-training-on fixture backend complete))
                   (from-sparse (predictions-after-training-on fixture backend sparse)))
               (testing (format nil "lightgbm: a booster TRAINED on the omitted-entry ~
                                     matrix predicts what one trained on the complete ~
                                     matrix does")
                 (ok (predictions-agree-p from-complete from-sparse)
                     (format nil "complete ~S, omitted ~S" from-complete from-sparse)))
               ;; Without this the agreement above could be two models that both learned
               ;; nothing and both answer a constant.
               (testing "lightgbm: both of those models learned to order the two classes"
                 (ok (and (predictions-separate-p (prediction-column from-complete 0)
                                                  *omitted-entry-labels*)
                          (predictions-separate-p (prediction-column from-sparse 0)
                                                  *omitted-entry-labels*))
                     (format nil "complete ~S, omitted ~S" from-complete from-sparse)))))
        (cl-gbdt:close-backend backend))))
  (with-backend-library (:xgboost)
    (let ((complete (omitted-entry-csr :omit-zeros nil))
          (sparse (omitted-entry-csr))
          (dense (omitted-entry-dense))
          (fixture (fixture-for :xgboost))
          (backend (cl-gbdt:open-backend :xgboost)))
      (unwind-protect
           (progn
             (cl-gbdt:with-booster
                 (booster (train-dense-booster fixture backend dense
                                               *omitted-entry-labels*))
               (let ((from-dense (cl-gbdt:predict booster dense))
                     (from-complete (cl-gbdt:predict booster complete))
                     (from-sparse (cl-gbdt:predict booster sparse)))
                 (testing (format nil "xgboost: predicting on the matrix with the zeros ~
                                       omitted answers something else than the dense ~
                                       matrix -- an absent entry is not 0.0")
                   (ok (not (predictions-agree-p from-dense from-sparse))
                       (format nil "dense ~S, omitted ~S" from-dense from-sparse)))
                 ;; The control: the CSR path itself is faithful, so the difference above is
                 ;; the omission and not CSR having been read wrong.
                 (testing (format nil "xgboost: the matrix storing every element does ~
                                       answer what the dense one does")
                   (ok (predictions-agree-p from-dense from-complete)
                       (format nil "dense ~S, complete ~S" from-dense from-complete)))
                 ;; The positive half: absent is MISSING, so every row -- the four that lost
                 ;; column 0 included -- follows the model's default direction, which lands
                 ;; where a present, positive column 0 lands. Same booster on both sides.
                 (testing (format nil "xgboost: with column 0 absent every row takes the ~
                                       value a row whose column 0 is present and positive ~
                                       gets")
                   (ok (every-prediction-equals-p
                        from-sparse
                        (aref from-dense (1- (length *omitted-entry-rows*)) 0))
                       (format nil "omitted ~S, dense ~S" from-sparse from-dense)))))
             (let ((from-complete (predictions-after-training-on fixture backend complete))
                   (from-sparse (predictions-after-training-on fixture backend sparse)))
               (testing (format nil "xgboost: a booster TRAINED on the omitted-entry ~
                                     matrix predicts something else than one trained on ~
                                     the complete matrix")
                 (ok (not (predictions-agree-p from-complete from-sparse))
                     (format nil "complete ~S, omitted ~S" from-complete from-sparse)))
               ;; What the difference IS, rather than merely that there is one: the model
               ;; trained on the complete matrix orders the two classes on the dense rows and
               ;; the one trained without those zeros does not, because the rows it was asked
               ;; to learn the negative class from had no column 0 at all.
               (testing (format nil "xgboost: the complete matrix's model orders the two ~
                                     classes and the omitted-entry matrix's does not")
                 (ok (and (predictions-separate-p (prediction-column from-complete 0)
                                                  *omitted-entry-labels*)
                          (not (predictions-separate-p (prediction-column from-sparse 0)
                                                       *omitted-entry-labels*)))
                     (format nil "complete ~S, omitted ~S" from-complete from-sparse)))))
        (cl-gbdt:close-backend backend)))))
