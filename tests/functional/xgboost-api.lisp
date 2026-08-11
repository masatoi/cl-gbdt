;;;; xgboost-api.lisp --- Round trip through cl-gbdt's unified API, XGBoost backend.
;;;;
;;;; tests/functional/xgboost.lisp calls the raw FFI bindings directly, proving the
;;;; generated bindings match the XGBoost ABI. This file never touches
;;;; cl-gbdt/src/xgboost/c-api or the FFI package's `xgb::' symbols; it goes through
;;;; `open-backend', `with-dataset' + `make-dataset', `with-booster' + `train',
;;;; `predict' and `close-backend' exactly as any caller of cl-gbdt would, and stays
;;;; green only because the twelve XGBoost protocol methods actually do what the
;;;; unified API promises.

(uiop:define-package #:cl-gbdt/tests/functional/xgboost-api
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt)
  ;; Zero symbols: nothing below refers to this package by name. Its job is to run at load
  ;; time, registering :xgboost with `open-backend' -- see `register-backend' near the top
  ;; of src/xgboost/classes.lisp -- and defining XGBoost's methods on `cl-gbdt''s generics.
  ;; Without this clause, package-inferred-system has no edge to those files at all, and
  ;; `(cl-gbdt:open-backend :xgboost)' below would signal `unknown-backend'. Depends on
  ;; `unified', the aggregation the leaf `cl-gbdt/xgboost/unified' itself depends on, rather
  ;; than `protocol' directly, so this exercises the same load path a real caller would. Not
  ;; `all': since the Layer 1 split that is Layer 1 alone, and `train' below would find no
  ;; applicable method.
  (:import-from #:cl-gbdt/src/xgboost/unified)
  ;; Zero symbols, same load-time reason as the clause above, for the other backend:
  ;; `xgboost-api-slice-model-rejects-a-lightgbm-booster' needs a real LightGBM booster to
  ;; hand `slice-model', and `(cl-gbdt:open-backend :lightgbm)' signals `unknown-backend'
  ;; unless something has loaded that backend's `register-backend' call. This one test is
  ;; the file's only LightGBM reference and it is still an XGBoost test -- it asserts that
  ;; *XGBoost's* entry point rejects a foreign handle, which is why it belongs here rather
  ;; than in the backend-neutral `cl-gbdt/tests/functional/evaluation', whose own package
  ;; form carries both clauses for the different reason that it runs the same assertions
  ;; against both backends. `unified' for the same reason as the clause above: that test
  ;; trains its LightGBM booster through `cl-gbdt:train'.
  (:import-from #:cl-gbdt/src/lightgbm/unified)
  ;; Zero symbols, same reasoning as the `#:cl-gbdt' clause above: every reference below is
  ;; package-qualified, `cl-gbdt/xgboost:evaluate-one-iteration'. Declared anyway so this file's
  ;; dependency on Task 3's new public function is explicit rather than riding along on
  ;; `#:cl-gbdt/src/xgboost/unified' above, which exists here only for its load-time side
  ;; effects, not for anything it exports. Matches
  ;; `cl-gbdt/tests/functional/lightgbm-api''s identical clause for Task 2's
  ;; `cl-gbdt/lightgbm'.
  (:import-from #:cl-gbdt/xgboost)
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library
                #:make-separable-dataset
                #:predictions-separate-p
                #:make-multiclass-dataset
                #:predictions-match-labels-p
                #:within-group-strictly-increasing-p))

(in-package #:cl-gbdt/tests/functional/xgboost-api)

(defparameter *booster-parameters*
  '(:objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0)
  "XGBoost booster parameters for the guard tests below, which only need a booster to
exist, not to model anything well. See tests/functional/xgboost.lisp's
set-booster-parameters for the same values against the raw FFI. Unlike LightGBM,
XGBoost's `make-dataset' takes no dataset-level parameter that affects training --
see that method's docstring -- so there is no XGBoost analogue of LightGBM's own
*dataset-parameters*.")

(defparameter *round-trip-matrix*
  (make-array '(8 3) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0 0.0d0)
                                   (0.0d0 0.0d0 1.0d0)
                                   (0.0d0 1.0d0 0.0d0)
                                   (0.0d0 1.0d0 1.0d0)
                                   (1.0d0 0.0d0 0.0d0)
                                   (1.0d0 0.0d0 1.0d0)
                                   (1.0d0 1.0d0 0.0d0)
                                   (1.0d0 1.0d0 1.0d0)))
  "Feature matrix for `xgboost-api-round-trip': a three-bit multiplexer, not
`make-separable-dataset''s columns. Those three columns are a constant shift of each
other -- element [i][j] is (i + j)/10 -- so whichever one a tree splits on first
perfectly separates the labels alone and no boosting round ever needs another. That
is fine for LightGBM's `LGBM_BoosterFeatureImportance', which always reports one
score per feature in the dataset regardless of use, but not for XGBoost's
`XGBoosterFeatureScore', which reports scores only for features that appear in at
least one split -- confirmed directly against the vendored library, and documented
upstream: a feature never split on is absent from the result, not reported with a
zero. Column 0 here instead selects which of columns 1 and 2 is the label -- see
*ROUND-TRIP-LABELS* -- so no model that gets every row right can do it by splitting
on one column alone.")

(defparameter *round-trip-labels*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 1.0 1.0 0.0 1.0 0.0 1.0))
  "Labels matching *ROUND-TRIP-MATRIX*, row for row: column 1's value when column 0
is 0.0, column 2's value when column 0 is 1.0.")

(defparameter *round-trip-booster-parameters*
  '(:objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0 :min-child-weight 0)
  "XGBoost booster parameters for `xgboost-api-round-trip'. :MIN-CHILD-WEIGHT 0 is needed
for the model to actually recover the multiplexer in *ROUND-TRIP-MATRIX* -- confirmed
empirically: this file's default *BOOSTER-PARAMETERS* above, applied to the same data,
converge to a single split on one column, which cannot separate every row -- see
*ROUND-TRIP-MATRIX*'s docstring for why the fixture rules that out.

A reduced :LAMBDA was also part of this plist before `feature-importance' was fixed to
return a dense, per-column vector: with the old sparse result, the \"one entry per
feature\" assertion below only passed because these exact parameters happened to drive
every one of the three columns into a split. Now that the result's length no longer
depends on which columns were actually split on, that reduced :LAMBDA is not needed for
either assertion to hold and is dropped -- otherwise identical to *BOOSTER-PARAMETERS*
above but for :MIN-CHILD-WEIGHT 0, confirmed empirically to still separate.")

(defparameter *multiclass-booster-parameters*
  '(:objective "multi:softprob" :num-class 3 :max-depth 3 :eta 0.5 :verbosity 0)
  "XGBoost booster parameters for the nine-row, three-class fixture. `:num-class' must
equal `make-multiclass-dataset''s NUM-CLASSES for `predict' to size its second
dimension correctly -- see `xgboost-api-multiclass-round-trip'.")

(defun %column (matrix column)
  "Return COLUMN of the 2D array MATRIX as a fresh `(simple-array double-float (*))'.

`predictions-separate-p' takes a 1D sequence; `predict' returns a 2D array, one row
per input row and one column per class. `xgboost-api-round-trip''s objective is
binary, so COLUMN is always 0, but the shape still has to be unpacked by hand."
  (let* ((rows (array-dimension matrix 0))
         (result (make-array rows :element-type 'double-float)))
    (dotimes (row rows result)
      (setf (aref result row) (aref matrix row column)))))

(deftest xgboost-api-round-trip
  (with-backend-library (:xgboost)
    (let ((matrix *round-trip-matrix*)
          (label-vector *round-trip-labels*)
          (backend (cl-gbdt:open-backend :xgboost)))
      (let ((rows (array-dimension matrix 0))
            (cols (array-dimension matrix 1)))
        (testing "open-backend marks the backend open"
          (ok (cl-gbdt:backend-open-p backend)))
        (unwind-protect
             (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset
                                              backend matrix :label label-vector))
               (testing "dataset-num-rows and dataset-num-features match the fixture"
                 (ok (= rows (cl-gbdt:dataset-num-rows dataset))
                     (format nil "dataset-num-rows is ~D, expected ~D"
                             (cl-gbdt:dataset-num-rows dataset) rows))
                 (ok (= cols (cl-gbdt:dataset-num-features dataset))
                     (format nil "dataset-num-features is ~D, expected ~D"
                             (cl-gbdt:dataset-num-features dataset) cols)))

               (cl-gbdt:with-booster (booster (cl-gbdt:train
                                                backend dataset :num-rounds 20
                                                :parameters *round-trip-booster-parameters*))
                 (let ((predictions (cl-gbdt:predict booster matrix)))
                   (testing "predict returns a 2D double-float array with the fixture's rows"
                     (ok (typep predictions '(simple-array double-float (* *)))
                         (format nil "predict's element type is ~A"
                                 (array-element-type predictions)))
                     (ok (= rows (array-dimension predictions 0))
                         (format nil "predict's row count is ~D, expected ~D"
                                 (array-dimension predictions 0) rows)))

                   (testing "predictions separate along the label boundary"
                     (ok (predictions-separate-p (%column predictions 0) label-vector)
                         (format nil "predictions: ~S" predictions)))

                   (testing "feature-importance has one entry per feature"
                     (let ((importance (cl-gbdt:feature-importance booster)))
                       (ok (= cols (length importance))
                           (format nil "feature-importance length is ~D, expected ~D"
                                   (length importance) cols))))

                   (testing "save-model then load-model reproduces the original predictions"
                     (uiop:with-temporary-file (:pathname path :type "json")
                       (cl-gbdt:save-model booster path)
                       (cl-gbdt:with-booster (loaded (cl-gbdt:load-model backend path))
                         (ok (equalp predictions (cl-gbdt:predict loaded matrix))
                             "reloaded predictions equal the original's, elementwise")))))))
          (cl-gbdt:close-backend backend))
        (testing "close-backend marks the backend closed"
          (ng (cl-gbdt:backend-open-p backend)))))))

;;; Task 3 wired `check-backend-version' into this backend's `initialize-backend' --
;;; see `cl-gbdt/src/xgboost/protocol''s `initialize-backend' and
;;; `cl-gbdt/src/version''s `*xgboost-version-range*'. The vendored library here is
;;; 3.3.0, XGBoost's own recorded VERIFIED point and *XGBOOST-VERSION-RANGE*'s
;;; INFERRED-HIGH -- inside the range by construction. A warning here would mean the
;;; range itself is recorded wrong, not that the vendored library is somehow suspect.

(deftest xgboost-api-open-backend-against-vendored-library-warns-nothing
  (with-backend-library (:xgboost)
    (testing "opening XGBoost against the vendored library signals no version warning"
      (let ((warnings nil))
        ;; handler-bind, not handler-case: a WARNING does not unwind the stack on its
        ;; own, so handler-case around open-backend would establish a non-local exit
        ;; and change control flow instead of merely observing -- the identical
        ;; pitfall this repo's own prompts/repl-driven-development.md documents for
        ;; rove's `signals'.
        (handler-bind ((cl-gbdt:untested-backend-version
                          (lambda (c) (push c warnings))))
          (let ((backend (cl-gbdt:open-backend :xgboost)))
            (cl-gbdt:close-backend backend)))
        (ok (null warnings)
            (format nil "unexpected untested-backend-version warning(s): ~S" warnings))))))

;;; The LightGBM branch that first built its own version of this fixture documented
;;; that its single objective, "binary", cannot catch a `predict' buffer sized as the
;;; row count alone -- a single-class objective's row count and true element count
;;; are the same number. XGBoost's `predict' derives its column count from
;;; `XGBoosterPredictFromDMatrix''s own `out_shape'/`out_dim' report divided by the
;;; row count, in `%predict-ncol', rather than assuming a class count -- this is the
;;; same proof for this backend: a three-class `multi:softprob' booster's shape is
;;; (9 3), not (9 1).

(deftest xgboost-api-multiclass-round-trip
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-multiclass-dataset)
      (let ((rows (array-dimension matrix 0))
            (backend (cl-gbdt:open-backend :xgboost)))
        (unwind-protect
             (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset
                                              backend matrix :label label-vector))
               (cl-gbdt:with-booster (booster (cl-gbdt:train
                                                backend dataset :num-rounds 10
                                                :parameters *multiclass-booster-parameters*))
                 (let ((predictions (cl-gbdt:predict booster matrix)))
                   (testing "predict's second dimension equals the class count"
                     (ok (= rows (array-dimension predictions 0))
                         (format nil "predict's row count is ~D, expected ~D"
                                 (array-dimension predictions 0) rows))
                     (ok (= 3 (array-dimension predictions 1))
                         (format nil "predict's column count is ~D, expected 3 classes"
                                 (array-dimension predictions 1))))
                   (testing "predictions pick out the right class per row"
                     (ok (predictions-match-labels-p predictions label-vector)
                         (format nil "predictions: ~S" predictions))))))
          (cl-gbdt:close-backend backend))))))

(defparameter *ranking-matrix*
  (make-array '(8 2) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0) (0.0d0 3.0d0)
                                   (100.0d0 0.0d0) (100.0d0 1.0d0) (100.0d0 2.0d0)
                                   (100.0d0 3.0d0)))
  "Two four-row query groups. Column 1 carries the real within-group rank signal, 0..3 in
both groups; column 0 is a constant per-group \"regime\" indicator, 0 in the first group
and 100 in the second, deliberately identical in shape but not value between the two --
see `xgboost-api-ranking-round-trip-respects-group-boundaries' for why that makes GROUP
load-bearing rather than decorative.")

(defparameter *ranking-labels*
  (make-array 8 :element-type 'single-float :initial-contents '(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0))
  "Relevance grades matching *RANKING-MATRIX*, increasing with column 1 within each of its
two groups.")

(defparameter *ranking-group* '(4 4)
  "Two query groups of four rows each, matching *RANKING-MATRIX* and *RANKING-LABELS*.")

;;; Task 2 wired GROUP up on this backend via `XGDMatrixSetUIntInfo' -- see
;;; `cl-gbdt/src/xgboost/native''s `%set-group-field'. Setting the field is not evidence
;;; ranking actually works, so this trains a small `rank:pairwise' model and checks the
;;; property that objective exists for: predictions increase in step with true relevance
;;; *within* each query group, not raw score values -- `within-group-strictly-increasing-p'
;;; is this file's ranking analogue of `predictions-separate-p' and
;;; `predictions-match-labels-p' above.
;;;
;;; *RANKING-MATRIX*'s column 0 is what makes the assertion below load bearing rather than
;;; decorative: once GROUP is dropped and the whole matrix is treated as a single query,
;;; column 0 alone -- constant per group, but different between the two -- trivially
;;; resolves nearly every pair the objective compares (any row of the first group against
;;; any row of the second), so a deliberately low-capacity model (NUM-ROUNDS 3, :MAX-DEPTH
;;; 1: one split per round) spends its whole budget on that shortcut and never resolves
;;; column 1's real within-group order. Confirmed directly against the vendored library,
;;; with GROUP dropped from this exact call and everything else unchanged: predictions come
;;; back tied within each group -- (-0.76d0 -0.35d0 0.19d0 0.19d0 -0.36d0 0.05d0 0.59d0
;;; 0.59d0) -- instead of the four strictly increasing values this test requires, so this
;;; assertion goes red rather than passing vacuously. With GROUP respected, column 0 is
;;; constant within every pair `rank:pairwise' ever compares and carries no gradient there,
;;; forcing the model onto column 1 instead -- confirmed deterministic across repeated runs
;;; at this NUM-ROUNDS/:MAX-DEPTH.

(deftest xgboost-api-ranking-round-trip-respects-group-boundaries
  (with-backend-library (:xgboost)
    (let ((backend (cl-gbdt:open-backend :xgboost)))
      (unwind-protect
           (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset
                                            backend *ranking-matrix* :label *ranking-labels*
                                            :group *ranking-group*))
             (cl-gbdt:with-booster (booster (cl-gbdt:train
                                              backend dataset :num-rounds 3
                                              :parameters '(:objective "rank:pairwise"
                                                            :max-depth 1 :eta 0.5
                                                            :verbosity 0 :min-child-weight 0)))
               (let ((predictions (cl-gbdt:predict booster *ranking-matrix*)))
                 (testing "predictions increase strictly within each query group"
                   (ok (within-group-strictly-increasing-p
                        (loop :for row :below 8 :collect (aref predictions row 0))
                        *ranking-group*)
                       (format nil "predictions: ~S" predictions))))))
        (cl-gbdt:close-backend backend)))))

;;; Mutation testing on the LightGBM branch's analogous round trip found that
;;; `free-dataset' could be replaced with a no-op and every assertion stayed green --
;;; see tests/functional/lightgbm-api.lisp's commentary above its own version of this
;;; test. XGBoost's `free-dataset' has the identical shape: `release-handle' guarded
;;; by `backend-open-p', warn-and-leak when the backend is already closed. This
;;; proves the same guard here: a freed dataset used afterward signals instead of
;;; reaching `XGDMatrixNumRow' with a stale pointer.

(deftest xgboost-api-free-dataset-releases-the-handle
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost)))
        (unwind-protect
             (let ((dataset (cl-gbdt:make-dataset backend matrix :label label-vector)))
               (cl-gbdt:free-dataset dataset)
               (testing "reading a freed dataset signals released-handle-error"
                 (ok (handler-case (progn (cl-gbdt:dataset-num-rows dataset) nil)
                       (cl-gbdt:released-handle-error () t))
                     "dataset-num-rows on a freed dataset did not signal"))
               (testing "freeing a second time is a no-op"
                 (ok (handler-case (progn (cl-gbdt:free-dataset dataset) t)
                       (error () nil))
                     "a second free-dataset signaled")))
          (cl-gbdt:close-backend backend))))))

;;; `XGDMatrixCreateFromDense' produces the raw DMatrix before `make-dataset' ever
;;; attaches label/weight/feature-names, and `make-handle' does not take ownership of
;;; that raw pointer until every attachment step has succeeded -- see `make-dataset''s
;;; docstring. A wrong-length :LABEL is the commonest way `XGDMatrixSetInfoFromInterface'
;;; fails; without the `unwind-protect' freeing the raw pointer when OWNED stays nil,
;;; this would leave that DMatrix permanently orphaned: `make-dataset' would signal
;;; without ever returning a handle the caller could free. Ported from
;;; tests/functional/lightgbm-api.lisp's identical test, which this backend had no
;;; equivalent of -- this leak-prevention path had never executed, nor had
;;; `foreign-call-error', anywhere in this file before this test.

(deftest xgboost-api-make-dataset-wrong-length-label-signals-without-leaking
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (declare (ignore label-vector))
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (wrong-length-label (make-array 1 :element-type 'single-float
                                               :initial-element 0.0)))
        (unwind-protect
             (testing "a wrong-length label signals foreign-call-error"
               ;; handler-case, not rove's `signals' -- see this file's other guard
               ;; tests for why.
               (ok (handler-case
                       (progn (cl-gbdt:make-dataset backend matrix :label wrong-length-label)
                              nil)
                     (cl-gbdt:foreign-call-error () t))
                   "make-dataset with a wrong-length label did not signal foreign-call-error"))
          (cl-gbdt:close-backend backend))))))

;;; The same wiring test tests/functional/lightgbm-api.lisp carries against its own backend,
;;; and for the same reason: `check-feature-names' has one call site in each backend's
;;; `%set-feature-names', and tests/feature-names.lisp pins the guard itself rather than either
;;; call. One test per backend, because one deleted line is one backend's regression and a test
;;; on the other backend stays green through it.
;;;
;;; Measured with this test in place and the `(check-feature-names feature-names :xgboost)'
;;; line deleted from src/xgboost/native.lisp: layer 2 failed exactly one file -- this one --
;;; with the same `TYPE-ERROR' from `(LENGTH (a . b))', raised this time inside
;;; `CL-GBDT/SRC/XGBOOST/NATIVE:%SET-FEATURE-NAMES'.
;;;
;;; No :PARAMETERS here, unlike the LightGBM twin: this backend refuses that argument outright
;;; (see `xgboost-api-make-dataset-with-parameters-signals-unsupported-argument' below), and
;;; refuses it BEFORE the DMatrix is built, so passing one would signal the same condition
;;; type for an entirely different reason.

(deftest xgboost-api-make-dataset-improper-feature-names-signals
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (declare (ignore label-vector))
      (let ((backend (cl-gbdt:open-backend :xgboost)))
        (unwind-protect
             (let ((condition
                     (handler-case
                         (progn (cl-gbdt:make-dataset backend matrix
                                                      :feature-names '("a" . "b"))
                                nil)
                       (cl-gbdt:unsupported-argument (c) c))))
               (testing "a dotted :feature-names signals unsupported-argument"
                 (ok condition "whether make-dataset signalled unsupported-argument"))
               (testing "and the condition names the argument the caller passed"
                 (ok (and condition
                          (equal ":feature-names"
                                 (cl-gbdt:unsupported-argument-argument condition)))
                     (format nil "the argument named by the condition: ~S"
                             (and condition
                                  (cl-gbdt:unsupported-argument-argument condition))))))
          (cl-gbdt:close-backend backend))))))

;;; `XGBoosterUpdateOneIter' dereferences the DMatrix pointer handed to it directly --
;;; unlike LightGBM, which reads its training set through an internal pointer,
;;; XGBoost's version takes it as an explicit argument, read back from
;;; `booster-training-set' -- so a freed training set reaching it is the same
;;; segfault hazard `%check-booster-datasets-live' exists to intercept on the
;;; LightGBM backend. This proves it does here too.

(deftest xgboost-api-update-one-iteration-after-training-set-freed-signals
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (dataset nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend dataset :num-rounds 1
                                             :parameters *booster-parameters*))
               (cl-gbdt:free-dataset dataset)
               (testing "update-one-iteration on a booster whose training set was freed signals"
                 ;; handler-case, not rove's `signals' -- `signals' does not reliably
                 ;; catch conditions raised inside `restart-case', a documented
                 ;; pitfall in this repo's own prompts/repl-driven-development.md.
                 (ok (handler-case (progn (cl-gbdt:update-one-iteration booster) nil)
                       (cl-gbdt:released-handle-error () t))
                     "update-one-iteration did not signal released-handle-error")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (cl-gbdt:close-backend backend)))))))

;;; `train' attaches every :valid-sets entry's DMatrix to `XGBoosterCreate' up front --
;;; see `%create-booster' -- so a validation set freed afterward is the same hazard as
;;; the training set above. This proves `%check-booster-datasets-live' covers it too.

(deftest xgboost-api-update-one-iteration-after-validation-set-freed-signals
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (dataset nil)
            (valid-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf valid-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend dataset :num-rounds 1
                                             :valid-sets (list valid-set)
                                             :parameters *booster-parameters*))
               (cl-gbdt:free-dataset valid-set)
               (testing "update-one-iteration on a booster whose validation set was freed signals"
                 ;; handler-case, not rove's `signals' -- see this file's other guard
                 ;; tests for why.
                 (ok (handler-case (progn (cl-gbdt:update-one-iteration booster) nil)
                       (cl-gbdt:released-handle-error () t))
                     "update-one-iteration did not signal released-handle-error")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when dataset (cl-gbdt:free-dataset dataset))
            (cl-gbdt:close-backend backend)))))))

;;; `train' copies VALID-SETS -- see `%create-booster' -- rather than storing the
;;; caller's own list, for the identical reason `cl-gbdt/src/lightgbm/backend''s
;;; `train' does: a caller who destructively truncates their own list after `train'
;;; returns must not silently remove a dataset from the booster's own view of what it
;;; depends on. This proves the guard still fires once the caller's own list has been
;;; mutated that way.

(deftest xgboost-api-update-one-iteration-survives-caller-mutating-valid-sets-list
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (dataset nil)
            (valid-set-1 nil)
            (valid-set-2 nil)
            (callers-valid-sets nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf valid-set-1 (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf valid-set-2 (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf callers-valid-sets (list valid-set-1 valid-set-2))
               (setf booster (cl-gbdt:train backend dataset :num-rounds 1
                                             :valid-sets callers-valid-sets
                                             :parameters *booster-parameters*))
               ;; Destructively truncate the caller's own list after train returns --
               ;; the same cons cell `train' saw, if it kept that list rather than a
               ;; copy of it.
               (setf (cdr callers-valid-sets) nil)
               (cl-gbdt:free-dataset valid-set-2)
               (testing "update-one-iteration still notices the freed validation set"
                 ;; handler-case, not rove's `signals' -- see this file's other guard
                 ;; tests for why.
                 (ok (handler-case (progn (cl-gbdt:update-one-iteration booster) nil)
                       (cl-gbdt:released-handle-error () t))
                     (format nil "update-one-iteration did not signal ~
                                   released-handle-error after the caller truncated its ~
                                   own valid-sets list"))))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when dataset (cl-gbdt:free-dataset dataset))
            (when valid-set-1 (cl-gbdt:free-dataset valid-set-1))
            (cl-gbdt:close-backend backend)))))))

;;; Task 3 added a guard here: `update-one-iteration' needs the DMatrix XGBoost
;;; trained on, read back from `booster-training-set', and a `load-model' booster has
;;; none -- without the guard a null pointer would reach `XGBoosterUpdateOneIter', the
;;; same shape of hazard as the freed-training-set case above but for a booster that
;;; never had a training set to begin with. The guard originally signalled a bare
;;; `simple-error'; it now signals `missing-training-set', a typed condition matching
;;; every other guard in this file. This proves it does, and that it is catchable by
;;; type rather than by parsing a message string.

(deftest xgboost-api-update-one-iteration-on-load-model-booster-signals-missing-training-set
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (dataset nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend dataset :num-rounds 1
                                             :parameters *booster-parameters*))
               (uiop:with-temporary-file (:pathname path :type "json")
                 (cl-gbdt:save-model booster path)
                 (cl-gbdt:with-booster (loaded (cl-gbdt:load-model backend path))
                   (testing (format nil "update-one-iteration on a load-model booster ~
                                          signals missing-training-set")
                     (ok (handler-case (progn (cl-gbdt:update-one-iteration loaded) nil)
                           (cl-gbdt:missing-training-set () t))
                         "update-one-iteration did not signal missing-training-set")))))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when dataset (cl-gbdt:free-dataset dataset))
            (cl-gbdt:close-backend backend)))))))

;;; `close-backend' calls `cffi:close-foreign-library', which may unmap the shared
;;; library from the process; POSIX does not guarantee it, but does not forbid it
;;; either -- see `cl-gbdt/src/lightgbm/backend''s identical commentary above its own
;;; version of this test. `handle-live-pointer' checks `backend-open-p' before
;;; returning a pointer, turning what would otherwise be a call into a library that
;;; might no longer be mapped into `backend-not-open'. This proves it does, using
;;; `dataset-num-rows' as a representative operation.

(deftest xgboost-api-operation-after-close-backend-signals-backend-not-open
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let* ((backend (cl-gbdt:open-backend :xgboost))
             (dataset (cl-gbdt:make-dataset backend matrix :label label-vector)))
        (cl-gbdt:close-backend backend)
        (unwind-protect
             (testing "dataset-num-rows after close-backend signals backend-not-open"
               (ok (handler-case (progn (cl-gbdt:dataset-num-rows dataset) nil)
                     (cl-gbdt:backend-not-open () t))
                   "dataset-num-rows did not signal backend-not-open"))
          (cl-gbdt:free-dataset dataset))))))

;;; `free-dataset' is deliberately the exception to the guard above -- see
;;; `cl-gbdt/src/xgboost/protocol''s `free-dataset' docstring for why it must not
;;; signal `backend-not-open' from `with-dataset''s cleanup form. This proves freeing
;;; after `close-backend' neither signals nor brings the process down.

(deftest xgboost-api-free-dataset-after-close-backend-does-not-signal
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let* ((backend (cl-gbdt:open-backend :xgboost))
             (dataset (cl-gbdt:make-dataset backend matrix :label label-vector)))
        (cl-gbdt:close-backend backend)
        (testing "free-dataset after close-backend does not signal"
          (ok (handler-case (progn (cl-gbdt:free-dataset dataset) t)
                (error () nil))
              "free-dataset signaled after its backend was closed"))))))

;;; `make-dataset', `train' and `load-model' each create a brand-new handle directly
;;; from a backend, unlike every other operation in this file, which reads an
;;; existing handle and so is already covered by `handle-live-pointer''s
;;; `backend-open-p' check -- see `cl-gbdt/src/lightgbm/backend''s identical F2
;;; commentary. This proves each of the three signals `backend-not-open' instead of
;;; reaching its first foreign call against a library that might no longer be
;;; mapped, one entry point per test.

(deftest xgboost-api-make-dataset-after-close-backend-signals-backend-not-open
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost)))
        (cl-gbdt:close-backend backend)
        (testing "make-dataset after close-backend signals backend-not-open"
          ;; handler-case, not rove's `signals' -- see this file's other guard tests
          ;; for why.
          (ok (handler-case
                  (progn (cl-gbdt:make-dataset backend matrix :label label-vector) nil)
                (cl-gbdt:backend-not-open () t))
              "make-dataset did not signal backend-not-open"))))))

(deftest xgboost-api-train-after-close-backend-signals-backend-not-open
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      ;; Two separate :xgboost backend instances, not one -- see
      ;; `cl-gbdt/tests/functional/lightgbm-api''s identical test for why a second,
      ;; still-open instance is needed to isolate `train''s own check on its BACKEND
      ;; argument from `handle-live-pointer''s check on DATASET's own backend.
      (let ((closed-backend (cl-gbdt:open-backend :xgboost))
            (open-backend (cl-gbdt:open-backend :xgboost))
            (dataset nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset open-backend matrix :label label-vector))
               (cl-gbdt:close-backend closed-backend)
               (testing "train after close-backend signals backend-not-open"
                 (ok (handler-case
                         (progn (cl-gbdt:train closed-backend dataset :num-rounds 1
                                                :parameters *booster-parameters*)
                                nil)
                       (cl-gbdt:backend-not-open () t))
                     "train did not signal backend-not-open")))
          (progn
            (when dataset (cl-gbdt:free-dataset dataset))
            (cl-gbdt:close-backend open-backend)))))))

(deftest xgboost-api-load-model-after-close-backend-signals-backend-not-open
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (dataset nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend dataset :num-rounds 1
                                             :parameters *booster-parameters*))
               (uiop:with-temporary-file (:pathname path :type "json")
                 (cl-gbdt:save-model booster path)
                 (cl-gbdt:close-backend backend)
                 (testing "load-model after close-backend signals backend-not-open"
                   (ok (handler-case (progn (cl-gbdt:load-model backend path) nil)
                         (cl-gbdt:backend-not-open () t))
                       "load-model did not signal backend-not-open"))))
          ;; backend is already closed by the time this runs: both frees take the
          ;; closed-backend path and only warn, never signal.
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when dataset (cl-gbdt:free-dataset dataset))))))))

;;; `train' dispatches on BACKEND, not on DATASET, so there is no CLOS specializer
;;; ruling out a booster passed where a dataset belongs -- `%check-xgboost-dataset'
;;; is what rejects it instead, before any foreign call. This proves it does.

(deftest xgboost-api-train-with-a-booster-as-dataset-signals
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost)))
        (unwind-protect
             (cl-gbdt:with-dataset (training-set (cl-gbdt:make-dataset
                                                   backend matrix :label label-vector))
               (cl-gbdt:with-booster (booster (cl-gbdt:train
                                                backend training-set :num-rounds 1
                                                :parameters *booster-parameters*))
                 (testing "train with a booster as its dataset argument signals"
                   (ok (handler-case
                           (progn (cl-gbdt:train backend booster :num-rounds 1
                                                  :parameters *booster-parameters*)
                                  nil)
                         (cl-gbdt:wrong-backend-reference () t))
                       "train accepted a booster as its dataset argument"))))
          (cl-gbdt:close-backend backend))))))

;;; Same hazard, reached through :valid-sets instead of DATASET itself -- see
;;; `%check-xgboost-dataset''s docstring. This proves a booster inside :valid-sets is
;;; rejected the same way.

(deftest xgboost-api-train-with-a-booster-in-valid-sets-signals
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost)))
        (unwind-protect
             (cl-gbdt:with-dataset (training-set (cl-gbdt:make-dataset
                                                   backend matrix :label label-vector))
               (cl-gbdt:with-booster (other-booster (cl-gbdt:train
                                                      backend training-set :num-rounds 1
                                                      :parameters *booster-parameters*))
                 (testing "train with a booster inside :valid-sets signals"
                   (ok (handler-case
                           (progn (cl-gbdt:train backend training-set :num-rounds 1
                                                  :valid-sets (list other-booster)
                                                  :parameters *booster-parameters*)
                                  nil)
                         (cl-gbdt:wrong-backend-reference () t))
                       "train accepted a booster inside :valid-sets"))))
          (cl-gbdt:close-backend backend))))))

;;; `make-dataset' signals `unsupported-argument' for :REFERENCE and :PARAMETERS rather
;;; than silently ignoring them -- see `%check-unsupported''s docstring and, for
;;; PARAMETERS, `make-dataset''s own docstring on this backend for why the vendored header
;;; rules out forwarding it into `XGDMatrixCreateFromDense''s config JSON instead. :GROUP
;;; used to be a third signalling argument here -- Task 2 wired it up via
;;; `XGDMatrixSetUIntInfo' instead of refusing it, so it now has its own round trip above,
;;; `xgboost-api-ranking-round-trip-respects-group-boundaries', rather than a refusal test.
;;; Neither :REFERENCE nor :PARAMETERS had a test at all before this: LightGBM's own
;;; :REFERENCE is a real feature, exercised by four of its own tests in
;;; tests/functional/lightgbm-api.lisp, and PARAMETERS is a real feature on both backends
;;; -- see *dataset-parameters* there -- so this backend's refusal of each is a deliberate
;;; substitute for that behaviour, not dead code with nothing depending on it. This proves
;;; both signal.

(deftest xgboost-api-make-dataset-with-reference-signals-unsupported-argument
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost)))
        (unwind-protect
             (cl-gbdt:with-dataset (reference (cl-gbdt:make-dataset
                                                backend matrix :label label-vector))
               (testing "make-dataset with :reference signals unsupported-argument"
                 ;; handler-case, not rove's `signals' -- see this file's other guard
                 ;; tests for why.
                 (ok (handler-case
                         (progn (cl-gbdt:make-dataset backend matrix :label label-vector
                                                       :reference reference)
                                nil)
                       (cl-gbdt:unsupported-argument () t))
                     "make-dataset with :reference did not signal unsupported-argument")))
          (cl-gbdt:close-backend backend))))))

(deftest xgboost-api-make-dataset-with-parameters-signals-unsupported-argument
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost)))
        (unwind-protect
             (testing "make-dataset with :parameters signals unsupported-argument"
               ;; handler-case, not rove's `signals' -- see this file's other guard
               ;; tests for why.
               (ok (handler-case
                       (progn (cl-gbdt:make-dataset backend matrix :label label-vector
                                                     :parameters '(:max-bin 3))
                              nil)
                     (cl-gbdt:unsupported-argument () t))
                   "make-dataset with :parameters did not signal unsupported-argument"))
          (cl-gbdt:close-backend backend))))))

;;; `save-model' signals `unsupported-argument' when NUM-ITERATION is supplied --
;;; `XGBoosterSaveModel' has no iteration limit, unlike LightGBM's `LGBM_BoosterSaveModel'
;;; -- see that method's docstring. This proves it does, one of five `unsupported-argument'
;;; call sites in this file with no test at all before this.

(deftest xgboost-api-save-model-with-num-iteration-signals-unsupported-argument
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (dataset nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend dataset :num-rounds 1
                                             :parameters *booster-parameters*))
               (uiop:with-temporary-file (:pathname path :type "json")
                 (testing "save-model with :num-iteration signals unsupported-argument"
                   ;; handler-case, not rove's `signals' -- see this file's other guard
                   ;; tests for why.
                   (ok (handler-case
                           (progn (cl-gbdt:save-model booster path :num-iteration 1) nil)
                         (cl-gbdt:unsupported-argument () t))
                       "save-model with :num-iteration did not signal unsupported-argument"))))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when dataset (cl-gbdt:free-dataset dataset))
            (cl-gbdt:close-backend backend)))))))

;;; F2: `XGBoosterFeatureScore' reports a per-class matrix, not one score per feature, for
;;; a linear (`gblinear') booster's `:split' importance on a multi-class model -- see
;;; `cl-gbdt/src/xgboost/native''s `%check-feature-score-dim' for the full measurement
;;; against the vendored library. Before the fix, `feature-importance' read only the
;;; first `n-features' raw scores out of that matrix -- exactly one feature's whole
;;; per-class row -- and scattered them across every column as if each belonged to a
;;; different feature: no error, a plausible-looking three-number result that was
;;; actually feature 0's three class scores, with feature 1's own score never read at
;;; all. Confirmed red against the code this fixes: the call returned
;;; `(-0.077..d0 0.0038..d0 0.0304..d0)' with no error, three numbers that were really
;;; feature 0's row of a 3x3 matrix -- verified directly against
;;; `XGBoosterFeatureScore''s own `out_dim'/`out_shape', which report `2' and `(3 3)' for
;;; this exact fixture. This proves the fix rejects the combination instead of returning
;;; it.

(defparameter *gblinear-multiclass-booster-parameters*
  '(:booster "gblinear" :objective "multi:softprob" :num-class 3 :verbosity 0)
  "XGBoost booster parameters producing the shape `feature-importance' cannot reduce to
one number per feature: a linear booster's `weight' importance on a multi-class model is
a row-major [n_features, n_classes] matrix, confirmed directly against the vendored
library -- see `cl-gbdt/src/xgboost/native''s `%check-feature-score-dim'.")

(deftest xgboost-api-feature-importance-on-gblinear-multiclass-signals-unsupported-argument
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-multiclass-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (dataset nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend dataset :num-rounds 5
                                             :parameters
                                             *gblinear-multiclass-booster-parameters*))
               (testing "feature-importance on a gblinear multi-class booster signals ~
                         unsupported-argument instead of returning a wrong result"
                 ;; handler-case, not rove's `signals' -- see this file's other guard
                 ;; tests for why.
                 (ok (handler-case (progn (cl-gbdt:feature-importance booster) nil)
                       (cl-gbdt:unsupported-argument () t))
                     "feature-importance did not signal unsupported-argument")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when dataset (cl-gbdt:free-dataset dataset))
            (cl-gbdt:close-backend backend)))))))

;;; ---------------------------------------------------------------------------
;;; Evaluation
;;;
;;; Task 3: `cl-gbdt/xgboost:evaluate-one-iteration', the XGBoost backend's first Layer 1
;;; (backend-specific safe API) function -- see
;;; docs/cl-gbdt-layered-api-implementation-policy.md section 3 and
;;; docs/superpowers/specs/2026-08-06-evaluation-api-design.md. It is a plain function
;;; exported directly from `cl-gbdt/xgboost', not a `defmethod' reached through `cl-gbdt',
;;; so it is exercised here, alongside this file's other round trips through the public
;;; API, rather than in tests/functional/xgboost.lisp, which never wraps a raw pointer in
;;; a handle object at all. Mirrors `cl-gbdt/tests/functional/lightgbm-api''s own Task 2
;;; evaluation section, but this section's shape differs from LightGBM's own exactly as
;;; much as `LGBM_BoosterGetEval' and `XGBoosterEvalOneIter' do -- see
;;; docs/superpowers/specs/2026-08-06-evaluation-api-design.md section 2: one function
;;; here, not two, and DATASETS/NAMES are caller-supplied arguments to `evaluate-one-iteration'
;;; itself rather than read back from `train''s `:valid-sets'.

(defparameter *eval-booster-parameters*
  '(:objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0 :min-child-weight 0
    :eval-metric "logloss" :eval-metric "error")
  "XGBoost booster parameters for the evaluation tests below. `:min-child-weight 0' is
needed for `xgboost-api-evaluate-one-iteration-tracks-actual-model-fit' below to show any change at
all: confirmed empirically against the vendored library, the default `:min-child-weight 1'
lets the very first boosting round already saturate every leaf's Hessian sum below that
threshold on this file's eight-row fixture, refusing every split after it and leaving
every later round's tree empty -- `predict' returns bit-identical output at 1 round and at
30 without this. Two metrics, not one, so a test that only checked a single label/value
pair could not pass by accident if `evaluate-one-iteration' silently dropped every field after the
first -- the same reasoning `cl-gbdt/tests/functional/lightgbm-api''s own
*eval-booster-parameters* gives for its two LightGBM metrics.")

(defparameter *%raw-field-value-reader-package*
  (or (find-package '#:cl-gbdt/tests/functional/xgboost-api/raw-field-scratch)
      (make-package '#:cl-gbdt/tests/functional/xgboost-api/raw-field-scratch :use nil))
  "Throwaway package used only as `*package*' while `%raw-field-value' reads a numeric
token, for the identical reason `cl-gbdt/src/xgboost/native''s own
`*%eval-value-reader-package*' exists -- a token that is not valid `double-float' syntax
(XGBoost's `\"inf\"'/`\"nan\"' spellings included) still reads as a plain symbol, which
would otherwise be interned into this test file's own `*package*' as a side effect of the
read. Deliberately a *different* package object from `native''s own scratch package:
`%raw-field-value' exists specifically to be an independent re-derivation of a (label,
value) pair from RAW, not a second call path that happens to share the parser under
test's own plumbing -- see that function's docstring.")

(defun %raw-field-value (raw label)
  "Return the double-float value following \"LABEL:\" in RAW, up to the next tab or the
end of RAW; NIL when LABEL does not appear in RAW at all, or when the text after its
colon is not valid `double-float' syntax -- XGBoost's own `\"inf\"'/`\"-inf\"'/`\"nan\"'/
`\"-nan\"' spellings for a non-finite value included.

Independent of `cl-gbdt/src/xgboost/native''s own `%parse-eval-result': this is the
functional suite's own re-derivation of a (label, value) pair straight out of the raw
string, used only to check that `evaluate-one-iteration''s PARSED return value agrees with its RAW
one -- reusing the function under test to check itself would prove nothing. `*package*'
is bound to *%raw-field-value-reader-package* for the read, never this file's own -- see
that parameter's docstring. `handler-case' catches `error' broadly rather than one named
condition, matching `cl-gbdt/src/xgboost/native''s own `%read-eval-value', which this
function deliberately does not call -- see that function's docstring for why a broad
catch is the right scope here, not a narrower one."
  (let ((start (search (concatenate 'string label ":") raw)))
    (when start
      (let* ((value-start (+ start (length label) 1))
             (tab (position #\Tab raw :start value-start))
             (value-end (or tab (length raw)))
             (*read-eval* nil)
             (*read-default-float-format* 'double-float)
             (*package* *%raw-field-value-reader-package*)
             (value (handler-case (read-from-string (subseq raw value-start value-end))
                      (error () nil))))
        (when (realp value)
          (coerce value 'double-float))))))

;;; Task 3 review, Important 1: XGBoost formats a metric value through `std::ostream',
;;; which spells a non-finite double as "inf", "-inf", "nan" or "-nan" -- reachable from
;;; real objectives, e.g. `mape' on a zero label or `rmsle' on a negative prediction, or
;;; from an empty field (a trailing tab with nothing after it). `%parse-eval-result' used
;;; to call a bare `read-from-string'/`coerce' on that text with no guard, so any of those
;;; shapes signalled TYPE-ERROR (a symbol read from "inf"/"nan" is not REAL) or
;;; END-OF-FILE (nothing to read from "") and aborted `evaluate-one-iteration' entirely --
;;; destroying RAW along with it, exactly what policy section 5 forbids and that function's own
;;; docstring calls "the only value this function can promise is faithful". This proves the
;;; parser survives every shape the review probed, calling it directly -- `::' because
;;; `%parse-eval-result' is a deliberately unexported internal of
;;; `cl-gbdt/src/xgboost/native', reached here to test its defensiveness in isolation from
;;; any foreign call, which no shared library is needed for; this test does not wrap itself
;;; in `with-backend-library' for exactly that reason -- it exercises no FFI call at all,
;;; and skipping it on a machine with no vendored library would be wrong.

(deftest xgboost-api-parse-eval-result-survives-non-finite-and-malformed-fields
  (testing "a plain finite value still parses to a double-float"
    (ok (equal '(("train-logloss" . 0.1d0))
               (cl-gbdt/src/xgboost/native::%parse-eval-result
                (format nil "[5]~Ctrain-logloss:0.1" #\Tab)))
        (format nil "got ~S"
                (cl-gbdt/src/xgboost/native::%parse-eval-result
                 (format nil "[5]~Ctrain-logloss:0.1" #\Tab)))))
  (testing "\"inf\" does not signal; the field survives, keyed by LABEL, with a NIL value"
    (ok (equal '(("train-mape" . nil))
               (cl-gbdt/src/xgboost/native::%parse-eval-result
                (format nil "[5]~Ctrain-mape:inf" #\Tab)))
        (format nil "got ~S"
                (cl-gbdt/src/xgboost/native::%parse-eval-result
                 (format nil "[5]~Ctrain-mape:inf" #\Tab)))))
  (testing "\"-inf\" does not signal either"
    (ok (equal '(("train-mape" . nil))
               (cl-gbdt/src/xgboost/native::%parse-eval-result
                (format nil "[5]~Ctrain-mape:-inf" #\Tab)))
        (format nil "got ~S"
                (cl-gbdt/src/xgboost/native::%parse-eval-result
                 (format nil "[5]~Ctrain-mape:-inf" #\Tab)))))
  (testing "\"nan\" does not signal; the field survives with a NIL value"
    (ok (equal '(("train-auc" . nil))
               (cl-gbdt/src/xgboost/native::%parse-eval-result
                (format nil "[5]~Ctrain-auc:nan" #\Tab)))
        (format nil "got ~S"
                (cl-gbdt/src/xgboost/native::%parse-eval-result
                 (format nil "[5]~Ctrain-auc:nan" #\Tab)))))
  (testing "\"-nan\" does not signal either"
    (ok (equal '(("train-auc" . nil))
               (cl-gbdt/src/xgboost/native::%parse-eval-result
                (format nil "[5]~Ctrain-auc:-nan" #\Tab)))
        (format nil "got ~S"
                (cl-gbdt/src/xgboost/native::%parse-eval-result
                 (format nil "[5]~Ctrain-auc:-nan" #\Tab)))))
  (testing "an empty value after the colon does not signal end-of-file"
    (ok (equal '(("train-x" . nil))
               (cl-gbdt/src/xgboost/native::%parse-eval-result
                (format nil "[5]~Ctrain-x:" #\Tab)))
        (format nil "got ~S"
                (cl-gbdt/src/xgboost/native::%parse-eval-result
                 (format nil "[5]~Ctrain-x:" #\Tab)))))
  (testing "a mix of finite and non-finite fields keeps every label, only the ~
            non-finite ones lose their value"
    (let ((raw (format nil "[5]~Ctrain-logloss:0.1~Ctrain-mape:inf" #\Tab #\Tab)))
      (ok (equal '(("train-logloss" . 0.1d0) ("train-mape" . nil))
                 (cl-gbdt/src/xgboost/native::%parse-eval-result raw))
          (format nil "got ~S" (cl-gbdt/src/xgboost/native::%parse-eval-result raw))))))

;;; Task 3 review, Minor 2, same finding: reading "inf"/"nan" interns a real symbol
;;; somewhere, and without `%read-eval-value''s throwaway `*package*' binding it would
;;; land in whatever package was current when `evaluate-one-iteration' was called -- a runtime
;;; `intern' as a side effect of parsing a result string. This proves reading a
;;; non-finite field does not touch this test's own `*package*'.

(deftest xgboost-api-parse-eval-result-does-not-intern-into-the-callers-package
  (let ((accessible-before (nth-value 1 (find-symbol "INF" *package*))))
    (cl-gbdt/src/xgboost/native::%parse-eval-result (format nil "[5]~Ctrain-mape:inf" #\Tab))
    (testing "INF was not accessible in this file's own package before the call"
      (ok (null accessible-before)
          (format nil "INF was already accessible in ~A: ~A" *package* accessible-before)))
    (testing "INF is still not accessible in this file's own package after the call"
      (ok (null (nth-value 1 (find-symbol "INF" *package*)))
          (format nil "INF became accessible in ~A" *package*)))))

;;; Step 3 of this task's brief: the raw string is returned intact, and the parse agrees
;;; with it. Two datasets and two metrics, so this also proves `evaluate-one-iteration' does not
;;; silently drop or transpose any of the four (dataset, metric) fields XGBoost reports.

(deftest xgboost-api-evaluate-one-iteration-raw-string-and-parse-agree
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (valid-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf valid-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 5
                                             :parameters *eval-booster-parameters*))
               (multiple-value-bind (raw parsed)
                   (cl-gbdt/xgboost:evaluate-one-iteration
                    booster (list train-set valid-set) '("train" "valid"))
                 (testing "raw starts with XGBoosterEvalOneIter's own \"[iteration]\" marker"
                   (ok (and (plusp (length raw)) (char= (char raw 0) #\[))
                       (format nil "raw was ~S" raw)))
                 (testing "parsed has one entry per (dataset, metric) pair -- two datasets, ~
                           two metrics"
                   (ok (= 4 (length parsed)) (format nil "parsed was ~S" parsed)))
                 (dolist (label '("train-logloss" "train-error" "valid-logloss" "valid-error"))
                   (let ((parsed-value (cdr (assoc label parsed :test #'string=)))
                         (raw-value (%raw-field-value raw label)))
                     (testing (format nil "~A is present in both raw and parsed" label)
                       (ok (and parsed-value raw-value)
                           (format nil "parsed: ~S, raw: ~S" parsed raw)))
                     (testing (format nil "~A's parsed value agrees with raw's" label)
                       (ok (and parsed-value raw-value (= parsed-value raw-value))
                           (format nil "~A: parsed ~S, raw ~S" label parsed-value
                                   raw-value)))))))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when valid-set (cl-gbdt:free-dataset valid-set))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; Design doc section 7's explicit ask: a test that would fail if evaluation were faked.
;;; See `cl-gbdt/tests/functional/lightgbm-api''s identical commentary above its own
;;; version of this test for why shape/range assertions alone cannot catch a stub.

(deftest xgboost-api-evaluate-one-iteration-tracks-actual-model-fit
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (undertrained nil)
            (overtrained nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf undertrained (cl-gbdt:train backend train-set :num-rounds 1
                                                  :parameters *eval-booster-parameters*))
               (setf overtrained (cl-gbdt:train backend train-set :num-rounds 30
                                                 :parameters *eval-booster-parameters*))
               (testing "more boosting rounds lowers logloss on the training set"
                 (let ((loss-1 (%raw-field-value
                                (cl-gbdt/xgboost:evaluate-one-iteration
                                 undertrained (list train-set) '("train"))
                                "train-logloss"))
                       (loss-30 (%raw-field-value
                                 (cl-gbdt/xgboost:evaluate-one-iteration
                                  overtrained (list train-set) '("train"))
                                 "train-logloss")))
                   (ok (< loss-30 loss-1)
                       (format nil "1 round: ~S, 30 rounds: ~S" loss-1 loss-30)))))
          (progn
            (when undertrained (cl-gbdt:free-booster undertrained))
            (when overtrained (cl-gbdt:free-booster overtrained))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; An empty DATASETS -- nothing to evaluate -- is a real call, not an error: confirmed
;;; directly against the vendored library, `XGBoosterEvalOneIter' still succeeds with a
;;; null `dmats'/`evnames' pair and length 0, returning just the bare "[iteration]" marker.
;;; This proves `evaluate-one-iteration' returns that marker as RAW and an empty PARSED, not a
;;; signal.

(deftest xgboost-api-evaluate-one-iteration-with-no-datasets-returns-empty-parse
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 1
                                             :parameters *eval-booster-parameters*))
               (multiple-value-bind (raw parsed)
                   (cl-gbdt/xgboost:evaluate-one-iteration booster nil nil)
                 (testing "raw is still XGBoosterEvalOneIter's own non-empty marker string"
                   (ok (plusp (length raw)) (format nil "raw was ~S" raw)))
                 (testing "parsed is empty"
                   (ok (null parsed) (format nil "parsed was ~S" parsed)))))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; `%eval-one-iter' checks DATASETS and NAMES are the same length before either foreign
;;; array is built -- a shorter NAMES would otherwise leave `evnames' entries pointing at
;;; whatever garbage memory `cffi:with-foreign-object' happened to allocate, which
;;; `XGBoosterEvalOneIter' would dereference as a C string: a segfault, not a catchable
;;; condition. This proves the check runs instead, before that allocation.

(deftest xgboost-api-evaluate-one-iteration-with-mismatched-names-signals-dimension-mismatch
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 1
                                             :parameters *eval-booster-parameters*))
               (testing "evaluate-one-iteration with more names than datasets signals ~
                         dimension-mismatch"
                 ;; handler-case, not rove's `signals' -- see this file's other guard
                 ;; tests for why.
                 (ok (handler-case
                         (progn (cl-gbdt/xgboost:evaluate-one-iteration
                                 booster (list train-set) '("train" "extra"))
                                nil)
                       (cl-gbdt:dimension-mismatch () t))
                     "evaluate-one-iteration did not signal dimension-mismatch")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; `evaluate-one-iteration' is a plain function, not a `defmethod' -- unlike every other operation
;;; this file guards against a mismatched handle, CLOS dispatch does not rule out the wrong
;;; kind on its own here. `%check-xgboost-booster' is the explicit check that stands in for
;;; it; this proves it runs before any foreign call, using a dataset handle as the wrong
;;; kind, mirroring `cl-gbdt/tests/functional/lightgbm-api''s identical proof for
;;; `booster-eval'/`booster-eval-names'.

(deftest xgboost-api-evaluate-one-iteration-with-a-dataset-as-booster-argument-signals
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (testing "evaluate-one-iteration with a dataset as its booster argument signals, ~
                         naming a booster as what was expected"
                 (let ((message (handler-case
                                     (progn (cl-gbdt/xgboost:evaluate-one-iteration
                                             train-set (list train-set) '("train"))
                                            nil)
                                   (cl-gbdt:wrong-backend-reference (c) (princ-to-string c)))))
                   (ok message "evaluate-one-iteration accepted a dataset as its booster argument")
                   (ok (and message (search "must be a booster built by" message))
                       (format nil "report was ~S" message)))))
          (progn
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; Same hazard, reached through DATASETS instead of BOOSTER: `%check-xgboost-eval-dataset'
;;; is what rejects a booster passed where a dataset belongs, before
;;; `XGBoosterEvalOneIter' ever runs. This also stands in for "a DMatrix from the other
;;; backend": an `xgboost-booster' is the wrong kind for the identical reason a LightGBM
;;; dataset would be -- neither is an XGBoost `dataset', which is all this check actually
;;; tests, mirroring `cl-gbdt/tests/functional/lightgbm-api''s own
;;; `lightgbm-api-make-dataset-with-a-non-dataset-reference-signals' commentary on the same
;;; point.

(deftest xgboost-api-evaluate-one-iteration-with-a-booster-in-datasets-signals
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 1
                                             :parameters *eval-booster-parameters*))
               (testing "evaluate-one-iteration with a booster inside datasets signals, naming a ~
                         dataset as what was expected"
                 (let ((message (handler-case
                                     (progn (cl-gbdt/xgboost:evaluate-one-iteration
                                             booster (list booster) '("train"))
                                            nil)
                                   (cl-gbdt:wrong-backend-reference (c) (princ-to-string c)))))
                   (ok message "evaluate-one-iteration accepted a booster inside datasets")
                   (ok (and message (search "must be a dataset built by" message))
                       (format nil "report was ~S" message)))))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; `evaluate-one-iteration' reads BOOSTER's pointer through `handle-live-pointer', the same guard
;;; every other operation in this file goes through -- this proves it, mirroring
;;; `xgboost-api-free-dataset-releases-the-handle''s proof for datasets.

(deftest xgboost-api-evaluate-one-iteration-after-free-booster-signals
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 1
                                             :parameters *eval-booster-parameters*))
               (cl-gbdt:free-booster booster)
               (testing "evaluate-one-iteration on a freed booster signals released-handle-error"
                 (ok (handler-case
                         (progn (cl-gbdt/xgboost:evaluate-one-iteration
                                 booster (list train-set) '("train"))
                                nil)
                       (cl-gbdt:released-handle-error () t))
                     "evaluate-one-iteration did not signal released-handle-error")))
          (progn
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; Same guard, for a DATASETS entry instead of BOOSTER itself: `%check-xgboost-eval-dataset'
;;; reads it through `handle-live-pointer' too. This proves a freed dataset inside DATASETS
;;; is rejected before `XGBoosterEvalOneIter' ever dereferences its stale pointer.

(deftest xgboost-api-evaluate-one-iteration-with-a-freed-dataset-signals
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (other-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf other-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 1
                                             :parameters *eval-booster-parameters*))
               (cl-gbdt:free-dataset other-set)
               (testing "evaluate-one-iteration with a freed dataset in datasets signals ~
                         released-handle-error"
                 (ok (handler-case
                         (progn (cl-gbdt/xgboost:evaluate-one-iteration
                                 booster (list other-set) '("other"))
                                nil)
                       (cl-gbdt:released-handle-error () t))
                     "evaluate-one-iteration did not signal released-handle-error")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; Same guard as `xgboost-api-operation-after-close-backend-signals-backend-not-open'
;;; above, for `evaluate-one-iteration' instead of `dataset-num-rows'.

(deftest xgboost-api-evaluate-one-iteration-after-close-backend-signals-backend-not-open
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let* ((backend (cl-gbdt:open-backend :xgboost))
             (train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
             (booster (cl-gbdt:train backend train-set :num-rounds 1
                                      :parameters *eval-booster-parameters*)))
        (cl-gbdt:close-backend backend)
        (unwind-protect
             (testing "evaluate-one-iteration after close-backend signals backend-not-open"
               (ok (handler-case
                       (progn (cl-gbdt/xgboost:evaluate-one-iteration
                               booster (list train-set) '("train"))
                              nil)
                     (cl-gbdt:backend-not-open () t))
                   "evaluate-one-iteration did not signal backend-not-open"))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))))))))

;;; ---------------------------------------------------------------------------
;;; Model slicing
;;;
;;; `cl-gbdt/xgboost:slice-model' is the operation policy section 7's capability model
;;; gates, and `cl-gbdt/xgboost:booster-boosted-rounds' is the query its interval is
;;; expressed against. Both are Layer 1 and XGBoost-only: LightGBM's C API has no
;;; counterpart to `XGBoosterSlice', so there is no portable generic function for these to
;;; be methods of -- see policy section 4's criterion for the unified API, and
;;; `cl-gbdt/tests/functional/evaluation''s `capability-model-discriminates-between-the-backends'
;;; for the pair of assertions that pins that asymmetry itself.
;;;
;;; `XGBoosterSlice''s three undocumented behaviours -- its interval semantics, whether it
;;; really reduces the model, and whether the result depends on its parent's memory -- were
;;; measured against the vendored library before these tests were written, not assumed;
;;; `+parent-rounds+'/`+sliced-rounds+' below and
;;; `xgboost-api-slice-model-outlives-its-parent' are the three answers written down as
;;; assertions.

(defconstant +parent-rounds+ 10
  "Boosting rounds the slicing tests train their parent booster for. Ten rather than the
one or two the guard tests above use, so that a half-open [0,5) slice is unambiguously
smaller than its parent and `+sliced-rounds+' can distinguish half-open from closed.")

(defconstant +sliced-rounds+ 5
  "Rounds a `:begin 0 :end 5' slice of a `+parent-rounds+'-round booster has.

Measured, not derived: `XGBoosterSlice''s header documents no interval semantics at all.
Against the vendored libxgboost 3.3.0 the answer is 5, i.e. the interval is half-open
[BEGIN, END) -- `:begin 3 :end 7' likewise gives 4, and XGBoost rejects `:begin 5 :end 5'
outright with \"Empty slice is not allowed\", which only a half-open reading explains. A
closed [BEGIN, END] reading would make this 6.")

(deftest xgboost-api-slice-model-returns-a-smaller-model
  ;; The assertion that a slice-model returning its argument unchanged would fail. Every other
  ;; assertion about slicing in this file would pass against that implementation, which is why
  ;; this one is here and why it compares predictions rather than only counting rounds.
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (booster nil)
            (sliced nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend train-set :num-rounds +parent-rounds+
                                             :parameters *eval-booster-parameters*))
               (setf sliced (cl-gbdt/xgboost:slice-model booster :begin 0 :end 5))
               (testing "the parent kept every round it was trained for"
                 (ok (= +parent-rounds+ (cl-gbdt/xgboost:booster-boosted-rounds booster))
                     "booster-boosted-rounds disagreed with train's :num-rounds"))
               (testing "the slice keeps the layers the interval names"
                 (ok (= +sliced-rounds+ (cl-gbdt/xgboost:booster-boosted-rounds sliced))
                     "slice-model did not produce the round count its interval promises"))
               (testing "the slice predicts differently from its parent"
                 (ok (not (equalp (cl-gbdt:predict sliced matrix)
                                  (cl-gbdt:predict booster matrix)))
                     "slice-model returned a model that predicts identically to its parent")))
          (progn
            (when sliced (cl-gbdt:free-booster sliced))
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; The lifetime answer, measured before `slice-model' was written and the reason it builds an
;;; ordinary booster with no retained parent, exactly as `load-model' does. `XGBoosterSlice'
;;; copies the layers it selects into a fresh booster: after the parent and the DMatrix it was
;;; trained on are both freed, the slice still predicts, and predicts the same values it did
;;; while its parent was alive. Had this faulted or drifted instead, `slice-model' would have
;;; had to retain the parent and liveness-check it the way `train' does its datasets -- and
;;; retaining it anyway, "just in case", is not the safe default it looks like: it would make
;;; the `free-booster' below signal `released-handle-error' on correct code.

(deftest xgboost-api-slice-model-outlives-its-parent
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (booster nil)
            (sliced nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend train-set :num-rounds +parent-rounds+
                                             :parameters *eval-booster-parameters*))
               (setf sliced (cl-gbdt/xgboost:slice-model booster :begin 0 :end 5))
               (let ((before (cl-gbdt:predict sliced matrix)))
                 (cl-gbdt:free-booster booster)
                 (cl-gbdt:free-dataset train-set)
                 (testing "the slice still predicts once its parent and its data are freed"
                   (ok (equalp before (cl-gbdt:predict sliced matrix))
                       "slice-model's result did not survive its parent being freed"))
                 (testing "the slice still reports its own round count"
                   (ok (= +sliced-rounds+ (cl-gbdt/xgboost:booster-boosted-rounds sliced))
                       "slice-model's result lost its layers when its parent was freed"))))
          (progn
            (when sliced (cl-gbdt:free-booster sliced))
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

(deftest xgboost-api-slice-model-rejects-a-lightgbm-booster
  ;; Check order: the handle-kind guard runs before the capability check, so a LightGBM
  ;; booster reports the wrong handle rather than "LightGBM cannot slice" -- true, but the
  ;; wrong mistake. This test is what pins that order.
  ;;
  ;; It lives in this file, not evaluation.lisp: it is about one backend's entry point
  ;; rejecting a foreign handle, not a portable contract asserted against both.
  ;; Only the LightGBM backend is opened. `slice-model' reaches `%check-handle-kind' before
  ;; any XGBoost call, so it signals without the XGBoost library being open at all -- which
  ;; is itself part of what this asserts: the guard runs before the library does.
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((lightgbm (cl-gbdt:open-backend :lightgbm))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset
                                lightgbm matrix :label label-vector
                                :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                              :verbose -1)))
               (setf booster (cl-gbdt:train lightgbm train-set :num-rounds 1
                                             :parameters '(:objective "binary"
                                                           :num-leaves 2
                                                           :min-data-in-leaf 1
                                                           :min-data-in-bin 1
                                                           :verbose -1)))
               (testing "slice-model rejects a LightGBM booster as a wrong handle"
                 (ok (handler-case
                         (progn (cl-gbdt/xgboost:slice-model booster :end 1) nil)
                       (cl-gbdt:wrong-backend-reference () t))
                     "slice-model accepted a LightGBM booster")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend lightgbm)))))))

(deftest xgboost-api-slice-model-on-a-freed-booster-signals
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 2
                                             :parameters *eval-booster-parameters*))
               (cl-gbdt:free-booster booster)
               (testing "slice-model on a freed booster signals released-handle-error"
                 (ok (handler-case
                         (progn (cl-gbdt/xgboost:slice-model booster :end 1) nil)
                       (cl-gbdt:released-handle-error () t))
                     "slice-model did not signal released-handle-error")))
          (progn
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; The two things `slice-model' promises about a bad interval, one of which is its own
;;; refusal and one of which is XGBoost's, passed through. `:END 0' is the refusal: NIL and 0
;;; would otherwise both reach `XGBoosterSlice' as its own `end_layer' 0, which means "through
;;; the last layer" -- so a caller writing `:END 0' for "no layers" would silently get the
;;; whole model, the translation policy section 5 exists to prevent. An END past the last
;;; layer is XGBoost's own refusal, asserted here because the docstring promises it comes back
;;; as a condition rather than a crash or a clamped result.

(deftest xgboost-api-slice-model-rejects-an-interval-it-cannot-honour
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend train-set :num-rounds +parent-rounds+
                                             :parameters *eval-booster-parameters*))
               (testing "an explicit :END of 0 signals rather than meaning the whole model"
                 (ok (handler-case
                         (progn (cl-gbdt/xgboost:slice-model booster :begin 0 :end 0) nil)
                       (cl-gbdt:unsupported-argument () t))
                     "slice-model accepted :END 0, which XGBoost would read as \"everything\""))
               (testing "an END past the last layer is XGBoost's error, not a clamp"
                 ;; `XGBoosterSlice' returns -2, not -1, for this path, and takes it before
                 ;; upstream's usual exception-to-XGBGetLastError plumbing runs -- confirmed
                 ;; against the vendored library, `XGBGetLastError' comes back empty in a fresh
                 ;; process for this call, or holding an earlier, unrelated failure's message
                 ;; verbatim in a process where one had already happened. Asserting only the
                 ;; condition type, as this test used to, would stay green for either of those
                 ;; broken messages; this checks the message itself instead.
                 (let* ((out-of-range-end (1+ +parent-rounds+))
                        (condition
                          (handler-case
                              (progn (cl-gbdt/xgboost:slice-model
                                      booster :begin 0 :end out-of-range-end)
                                     nil)
                            (cl-gbdt:foreign-call-error (c) c))))
                   (ok condition
                       "slice-model did not surface XGBoost's refusal of an out-of-range END")
                   (ok (and condition
                            (plusp (length (or (cl-gbdt:foreign-call-error-message condition)
                                                ""))))
                       "slice-model's foreign-call-error carried no message -- either a bare ~
                        colon (an empty XGBGetLastError) or the (no message) fallback")
                   (ok (and condition
                            (search (format nil "end ~D" out-of-range-end)
                                    (or (cl-gbdt:foreign-call-error-message condition) "")))
                       "slice-model's foreign-call-error message did not name the offending ~
                        END, so it cannot be this call's own diagnostic"))))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; Policy section 7's central rule, and the only assertion in the suite that exercises it:
;;; the operation re-checks the capability itself rather than trusting the caller to have
;;; asked `backend-supports-p' first, and signals rather than falling back to anything else.
;;; The vendored library does provide `XGBoosterSlice' -- `capability-model-discriminates-
;;; between-the-backends' asserts exactly that -- so the only way to reach this branch is to
;;; overwrite the probed plist, which is what an XGBoost too old to have the symbol would
;;; have produced at `open-backend'. Without this test the whole capability gate is a branch
;;; nobody has seen taken.

(deftest xgboost-api-slice-model-signals-when-the-capability-is-absent
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :xgboost))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix :label label-vector))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 2
                                             :parameters *eval-booster-parameters*))
               (setf (cl-gbdt:backend-capabilities backend) '(:model-slicing nil))
               (testing "slice-model signals capability-unavailable, naming the capability"
                 (let ((condition (handler-case
                                      (progn (cl-gbdt/xgboost:slice-model booster :end 1) nil)
                                    (cl-gbdt:capability-unavailable (c) c))))
                   (ok condition "slice-model sliced with the capability reported absent")
                   (ok (and condition
                            (eq :model-slicing
                                (cl-gbdt:capability-unavailable-capability condition)))
                       "capability-unavailable did not name :model-slicing")
                   (ok (and condition (eq :xgboost (cl-gbdt:backend-error-backend condition)))
                       "capability-unavailable did not name the backend"))))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; Task 7 (docs/superpowers/sdd/2026-08-11-layer1-separation): `cl-gbdt/xgboost' now
;;; re-exports the shared-basis symbols -- `open-backend', `backend-open-p', `backend-name',
;;; `backend-supports-p', `close-backend' and the whole `cl-gbdt/src/conditions' hierarchy --
;;; that a caller who loads this backend ALONE, never `cl-gbdt' itself, needs to work with it
;;; and catch what it signals without naming an internal package. Every other test in this
;;; file goes through `cl-gbdt:', proving the unified API; this one goes through
;;; `cl-gbdt/xgboost:' throughout, proving the standalone surface added by this task instead.
;;; No `:path' is passed to `open-backend' here, matching every other test above: this test's
;;; own `with-backend-library' has already loaded the vendored library via
;;; `cffi:load-foreign-library', so CFFI's own default resolution finds it without one.

(deftest the-backend-package-publishes-what-a-standalone-caller-needs
  (testing "opening, asking and closing are all reachable without naming an internal package"
    (with-backend-library (:xgboost)
      (let ((backend (cl-gbdt/xgboost:open-backend :xgboost)))
        (unwind-protect
             (progn
               (ok (cl-gbdt/xgboost:backend-open-p backend) "the backend reports itself open")
               (ok (eq :xgboost (cl-gbdt/xgboost:backend-name backend)) "and names itself")
               (ok (typep (cl-gbdt/xgboost:backend-supports-p backend :sparse-input) 'boolean)
                   "a capability question answers")
               (ok (subtypep 'cl-gbdt/xgboost:capability-unavailable
                             'cl-gbdt/xgboost:gbdt-error)
                   "the condition hierarchy is reachable from this package too"))
          (cl-gbdt/xgboost:close-backend backend))))))
