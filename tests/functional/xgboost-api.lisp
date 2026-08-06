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
  ;; Zero symbols: nothing below refers to this package by name. Its only job is to
  ;; run at load time and register :xgboost with `open-backend' -- see
  ;; `register-backend' near the top of src/xgboost/protocol.lisp. Without this
  ;; clause, package-inferred-system has no edge to that file at all, and
  ;; `(cl-gbdt:open-backend :xgboost)' below would signal `unknown-backend'. Depends on
  ;; `all', the leaf `cl-gbdt/xgboost' itself depends on, rather than `protocol'
  ;; directly, so this exercises the same load path a real caller would.
  (:import-from #:cl-gbdt/src/xgboost/all)
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
