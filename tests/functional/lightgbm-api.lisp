;;;; lightgbm-api.lisp --- Round trip through cl-gbdt's unified API, LightGBM backend.
;;;;
;;;; tests/functional/lightgbm.lisp calls the raw FFI bindings directly, proving the
;;;; generated bindings match the LightGBM ABI. This file never touches
;;;; cl-gbdt/src/lightgbm/c-api or the FFI package's `lgbm::' symbols; it goes through
;;;; `open-backend', `with-dataset' + `make-dataset', `with-booster' + `train',
;;;; `predict' and `close-backend' exactly as any caller of cl-gbdt would, and stays
;;;; green only because the twelve LightGBM protocol methods actually do what the
;;;; unified API promises.

(uiop:define-package #:cl-gbdt/tests/functional/lightgbm-api
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt)
  ;; Zero symbols: nothing below refers to this package by name. Its job is to run at load
  ;; time, registering :lightgbm with `open-backend' -- see `register-backend' near the top
  ;; of src/lightgbm/classes.lisp -- and defining LightGBM's methods on `cl-gbdt''s
  ;; generics. Without this clause, package-inferred-system has no edge to those files at
  ;; all, and `(cl-gbdt:open-backend :lightgbm)' below would signal `unknown-backend'.
  ;; Depends on `unified', the aggregation the leaf `cl-gbdt/lightgbm/unified' itself
  ;; depends on, rather than `protocol' directly, so this exercises the same load path a
  ;; real caller would. Not `all': since the Layer 1 split that is Layer 1 alone, and
  ;; `train' below would find no applicable method.
  (:import-from #:cl-gbdt/src/lightgbm/unified)
  ;; Zero symbols, same reasoning as the `#:cl-gbdt' clause above: every reference below is
  ;; package-qualified, `cl-gbdt/lightgbm:booster-eval-names' and `cl-gbdt/lightgbm:booster-eval'.
  ;; Declared anyway so this file's dependency on Task 2's new public functions is explicit
  ;; rather than riding along on `#:cl-gbdt/src/lightgbm/unified' above, which exists here
  ;; only for its load-time side effects, not for anything it exports.
  (:import-from #:cl-gbdt/lightgbm)
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library
                #:make-separable-dataset
                #:predictions-separate-p
                #:make-multiclass-dataset
                #:predictions-match-labels-p
                #:within-group-strictly-increasing-p))

(in-package #:cl-gbdt/tests/functional/lightgbm-api)

(defparameter *dataset-parameters*
  '(:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)
  "LightGBM dataset parameters for the eight-row fixture, as the plist `make-dataset'
accepts. The two minima are required: the defaults refuse to bin or split so few
rows, which would leave every prediction identical and fail the separation
assertion for a reason that has nothing to do with the unified API. `verbose -1'
keeps the library off standard output during the suite. See
tests/functional/lightgbm.lisp's *dataset-parameters* for the same rule stated
against the raw FFI.")

(defparameter *booster-parameters*
  '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)
  "LightGBM booster parameters for the eight-row fixture. See *dataset-parameters*.")

(defparameter *multiclass-dataset-parameters*
  '(:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)
  "LightGBM dataset parameters for the nine-row, three-class fixture. Same values as
*dataset-parameters*, for the same reason: the defaults refuse to bin or split so few
rows.")

(defparameter *multiclass-booster-parameters*
  '(:objective "multiclass" :num-class 3 :num-leaves 4 :min-data-in-leaf 1
    :min-data-in-bin 1 :verbose -1)
  "LightGBM booster parameters for the nine-row, three-class fixture. `:num-class'
must equal `make-multiclass-dataset''s NUM-CLASSES for `predict' to size its second
dimension correctly -- see `lightgbm-api-multiclass-round-trip'.")

(defun %single-column-matrix (values)
  "Return VALUES, a list of reals, as a `(simple-array double-float (N 1))' matrix with
one row per element and a single feature column."
  (let* ((rows (length values))
         (matrix (make-array (list rows 1) :element-type 'double-float)))
    (loop :for value :in values
          :for row :from 0
          :do (setf (aref matrix row 0) (coerce value 'double-float)))
    matrix))

(defparameter *reference-training-values* '(0.0 5.0 0.0 5.0 0.0 5.0 0.0 5.0)
  "Two distinct feature values for the training set in the :reference tests below.
LightGBM bins a dataset built from just these two values very differently from one
built from *reference-validation-values*, which shares none of them -- exactly the
mismatch `LGBM_BoosterAddValidData' rejects unless the validation set was built with
`:reference' pointing at the training set.")

(defparameter *reference-training-labels*
  (make-array 8 :element-type 'single-float
                :initial-contents '(0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0))
  "Binary labels matching *reference-training-values*, row for row -- `train''s
\"binary\" objective needs a label on the training set regardless of what the
:reference tests below are checking.")

(defparameter *reference-validation-values* '(0.3 1.1 1.7 2.3 2.9 3.5 4.1 4.9)
  "Feature values spread across the training set's range but sharing none of
*reference-training-values*, so a dataset built from these bins independently unless
:reference aligns it with the training set.")

(defun %column (matrix column)
  "Return COLUMN of the 2D array MATRIX as a fresh `(simple-array double-float (*))'.

`predictions-separate-p' takes a 1D sequence; `predict' returns a 2D array, one row
per input row and one column per class. The fixture's objective is binary, so
COLUMN is always 0, but the shape still has to be unpacked by hand."
  (let* ((rows (array-dimension matrix 0))
         (result (make-array rows :element-type 'double-float)))
    (dotimes (row rows result)
      (setf (aref result row) (aref matrix row column)))))

(deftest lightgbm-api-round-trip
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((rows (array-dimension matrix 0))
            (cols (array-dimension matrix 1))
            (backend (cl-gbdt:open-backend :lightgbm)))
        (testing "open-backend marks the backend open"
          (ok (cl-gbdt:backend-open-p backend)))
        (unwind-protect
             (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset
                                              backend matrix :label label-vector
                                              :parameters *dataset-parameters*))
               (testing "dataset-num-rows and dataset-num-features match the fixture"
                 (ok (= rows (cl-gbdt:dataset-num-rows dataset))
                     (format nil "dataset-num-rows is ~D, expected ~D"
                             (cl-gbdt:dataset-num-rows dataset) rows))
                 (ok (= cols (cl-gbdt:dataset-num-features dataset))
                     (format nil "dataset-num-features is ~D, expected ~D"
                             (cl-gbdt:dataset-num-features dataset) cols)))

               (cl-gbdt:with-booster (booster (cl-gbdt:train
                                                backend dataset :num-rounds 5
                                                :parameters *booster-parameters*))
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
                     (uiop:with-temporary-file (:pathname path :type "txt")
                       (cl-gbdt:save-model booster path)
                       (cl-gbdt:with-booster (loaded (cl-gbdt:load-model backend path))
                         (ok (equalp predictions (cl-gbdt:predict loaded matrix))
                             "reloaded predictions equal the original's, elementwise")))))))
          (cl-gbdt:close-backend backend))
        (testing "close-backend marks the backend closed"
          (ng (cl-gbdt:backend-open-p backend)))))))

;;; The LightGBM branch that first built this fixture recorded that its single
;;; objective, "binary", is structurally unable to catch a `predict' buffer sized as
;;; the row count alone: a single-class objective's row count and true element count
;;; are the same number, so a bug that dropped `LGBM_BoosterCalcNumPredict' entirely
;;; would leave every assertion in `lightgbm-api-round-trip' green. A three-class
;;; `multiclass' booster does not have that coincidence -- `predict''s result has
;;; three columns, not one -- so this closes the gap that branch could only document.

(deftest lightgbm-api-multiclass-round-trip
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-multiclass-dataset)
      (let ((rows (array-dimension matrix 0))
            (backend (cl-gbdt:open-backend :lightgbm)))
        (unwind-protect
             (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset
                                              backend matrix :label label-vector
                                              :parameters *multiclass-dataset-parameters*))
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
and 100 in the second. Shared with tests/functional/xgboost-api.lisp's identical fixture of
the same name, though this file's own mutation-testing result is different -- see the
commentary above `lightgbm-api-ranking-round-trip-respects-group-boundaries'.")

(defparameter *ranking-labels*
  (make-array 8 :element-type 'single-float :initial-contents '(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0))
  "Relevance grades matching *RANKING-MATRIX*, increasing with column 1 within each of its
two groups.")

(defparameter *ranking-group* '(4 4)
  "Two query groups of four rows each, matching *RANKING-MATRIX* and *RANKING-LABELS*.")

;;; `make-dataset''s GROUP has had a round trip on this backend since before Task 2 --
;;; `%set-dataset-field' attaches it via `LGBM_DatasetSetField' -- but Task 2 found it had
;;; never actually been exercised end to end: setting the field is not evidence ranking
;;; works. This trains a small `lambdarank' model and checks the property that objective
;;; exists for: predictions increase in step with true relevance *within* each query group,
;;; not raw score values -- `within-group-strictly-increasing-p' is this file's ranking
;;; analogue of `predictions-match-labels-p' above.
;;;
;;; Unlike this same fixture's mutation-testing result on the XGBoost branch -- where
;;; dropping GROUP leaves `rank:pairwise' free to substitute *RANKING-MATRIX*'s column 0 for
;;; the real per-group signal in column 1, producing a plausible-looking but wrong ordering
;;; -- LightGBM's `lambdarank' objective refuses to train at all without one: confirmed
;;; directly against the vendored library, dropping :GROUP from this exact call (everything
;;; else unchanged) signals `foreign-call-error', "LGBM_BoosterCreate returned -1: Ranking
;;; tasks require query information", rather than training a bad model silently. Both are
;;; the real signal the brief for this task asks for -- a wrong answer on one backend, an
;;; outright refusal on the other -- so this test only needs to prove the GROUP-respecting
;;; path produces correct order; GROUP's necessity here is already proven by the library's
;;; own refusal to omit it, not by a second, redundant assertion in this file.

(deftest lightgbm-api-ranking-round-trip-respects-group-boundaries
  (with-backend-library (:lightgbm)
    (let ((backend (cl-gbdt:open-backend :lightgbm)))
      (unwind-protect
           (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset
                                            backend *ranking-matrix* :label *ranking-labels*
                                            :group *ranking-group*
                                            :parameters '(:min-data-in-leaf 1
                                                          :min-data-in-bin 1 :verbose -1)))
             (cl-gbdt:with-booster (booster (cl-gbdt:train
                                              backend dataset :num-rounds 20
                                              :parameters '(:objective "lambdarank"
                                                            :num-leaves 4 :min-data-in-leaf 1
                                                            :min-data-in-bin 1 :verbose -1
                                                            :label-gain "0,1,2,3,4,5,6,7,8")))
               (let ((predictions (cl-gbdt:predict booster *ranking-matrix*)))
                 (testing "predictions increase strictly within each query group"
                   (ok (within-group-strictly-increasing-p
                        (loop :for row :below 8 :collect (aref predictions row 0))
                        *ranking-group*)
                       (format nil "predictions: ~S" predictions))))))
        (cl-gbdt:close-backend backend)))))

;;; Mutation testing on the round trip above found that `free-dataset' could be replaced
;;; with a no-op and every assertion stayed green: nothing there frees explicitly, nothing
;;; double-frees, and the leak finalizer only reports on collection -- which a short-lived
;;; test process never provokes. This closes that gap. A `free-dataset' that does not
;;; release leaves the handle usable, and `released-handle-error' is what notices.

(deftest lightgbm-api-free-dataset-releases-the-handle
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm)))
        (unwind-protect
             (let ((dataset (cl-gbdt:make-dataset backend matrix
                                                   :label label-vector
                                                   :parameters *dataset-parameters*)))
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

;;; `LGBM_BoosterUpdateOneIter' dereferences the booster's internal `train_data_'
;;; pointer, which `LGBM_DatasetFree' does not know about and does not clear. Freeing
;;; the training dataset out from under a live booster and then calling
;;; `update-one-iteration' is a segfault -- confirmed directly against the vendored
;;; library -- not a catchable Lisp condition, so the guard in
;;; `%check-training-set-live' has to run before any foreign call. This proves it does.

(deftest lightgbm-api-update-one-iteration-after-training-set-freed-signals
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (dataset nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset backend matrix
                                                    :label label-vector
                                                    :parameters *dataset-parameters*))
               (setf booster (cl-gbdt:train backend dataset :num-rounds 1
                                             :parameters *booster-parameters*))
               (cl-gbdt:free-dataset dataset)
               (testing "update-one-iteration on a booster whose training set was freed signals"
                 ;; handler-case, not rove's `signals' -- `signals' does not reliably
                 ;; catch conditions raised inside `restart-case', a documented pitfall
                 ;; in this repo's own prompts/repl-driven-development.md.
                 (ok (handler-case (progn (cl-gbdt:update-one-iteration booster) nil)
                       (cl-gbdt:released-handle-error () t))
                     "update-one-iteration did not signal released-handle-error")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (cl-gbdt:close-backend backend)))))))

;;; `LGBM_BoosterAddValidData' stores a validation dataset's pointer inside the booster
;;; exactly as `LGBM_BoosterCreate' does for the training set, but `train' used to pass
;;; only the training set to `make-handle' -- `%check-booster-datasets-live' (formerly
;;; `%check-training-set-live') had no way to know a validation set even existed.
;;; Freeing one and continuing to train was the same segfault as the training-set case
;;; above, confirmed directly against the vendored library. This proves the guard now
;;; covers validation sets too.

(deftest lightgbm-api-update-one-iteration-after-validation-set-freed-signals
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (dataset nil)
            (valid-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset backend matrix
                                                    :label label-vector
                                                    :parameters *dataset-parameters*))
               (setf valid-set (cl-gbdt:make-dataset backend matrix
                                                      :label label-vector
                                                      :parameters *dataset-parameters*))
               (setf booster (cl-gbdt:train backend dataset :num-rounds 1
                                             :valid-sets (list valid-set)
                                             :parameters *booster-parameters*))
               (cl-gbdt:free-dataset valid-set)
               (testing "update-one-iteration on a booster whose validation set was freed signals"
                 ;; handler-case, not rove's `signals' -- `signals' does not reliably
                 ;; catch conditions raised inside `restart-case', a documented pitfall
                 ;; in this repo's own prompts/repl-driven-development.md.
                 (ok (handler-case (progn (cl-gbdt:update-one-iteration booster) nil)
                       (cl-gbdt:released-handle-error () t))
                     "update-one-iteration did not signal released-handle-error")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when dataset (cl-gbdt:free-dataset dataset))
            (cl-gbdt:close-backend backend)))))))

;;; F1: `train' used to pass VALID-SETS -- the caller's own list -- straight to
;;; `make-handle', so the booster's `validation-sets' slot held that exact list
;;; object, not a snapshot of it. A caller who destructively edits VALID-SETS after
;;; `train' returns -- here, truncating it with `(setf (cdr ...))' the way `delete'
;;; would for a real removal -- edited the booster's view of its own dependencies
;;; too, since both were the same cons cells. Freeing the dataset that fell off the
;;; end and continuing to train would then reach `LGBM_BoosterUpdateOneIter'
;;; unchecked. This proves the guard still fires once the caller's own list has
;;; been mutated that way -- it only holds with the fix in place, since a snapshot
;;; taken at `train' time cannot be reached through the caller's list at all.

(deftest lightgbm-api-update-one-iteration-survives-caller-mutating-valid-sets-list
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (dataset nil)
            (valid-set-1 nil)
            (valid-set-2 nil)
            (callers-valid-sets nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset backend matrix
                                                    :label label-vector
                                                    :parameters *dataset-parameters*))
               (setf valid-set-1 (cl-gbdt:make-dataset backend matrix
                                                        :label label-vector
                                                        :parameters *dataset-parameters*))
               (setf valid-set-2 (cl-gbdt:make-dataset backend matrix
                                                        :label label-vector
                                                        :parameters *dataset-parameters*))
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

;;; I2: `LGBM_DatasetCreateFromMat' produces a fully-binned raw dataset before
;;; `make-dataset' ever attaches label/weight/group/feature-names, and `make-handle'
;;; does not take ownership of that raw pointer until every attachment step has
;;; succeeded. A wrong-length `:label' -- the commonest way `LGBM_DatasetSetField'
;;; fails -- used to leave that raw dataset permanently orphaned: `make-dataset'
;;; signaled without ever returning a handle the caller could free. This proves the
;;; signal still happens; see this file's final report for how the leak fix itself
;;; (freeing the raw pointer in `make-dataset''s `unwind-protect' before propagating)
;;; was checked, since a leaked *raw* C pointer here has no Lisp object or finalizer
;;; to observe going missing.

(deftest lightgbm-api-make-dataset-wrong-length-label-signals-without-leaking
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (declare (ignore label-vector))
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (wrong-length-label (make-array 1 :element-type 'single-float
                                               :initial-element 0.0)))
        (unwind-protect
             (testing "a wrong-length label signals foreign-call-error"
               (ok (handler-case
                       (progn (cl-gbdt:make-dataset backend matrix :label wrong-length-label
                                                     :parameters *dataset-parameters*)
                              nil)
                     (cl-gbdt:foreign-call-error () t))
                   "make-dataset with a wrong-length label did not signal foreign-call-error"))
          (cl-gbdt:close-backend backend))))))

;;; `make-dataset' reaches `cl-gbdt/src/config/feature-names''s `check-feature-names' only
;;; through `%set-feature-names', and only through the `(when feature-names ...)' branch that
;;; calls it. tests/feature-names.lisp pins the guard by calling it directly; nothing pinned
;;; the WIRING. Measured with this test in place and the
;;; `(check-feature-names feature-names :lightgbm)' line deleted from src/lightgbm/native.lisp:
;;; layer 1 stayed green at 411 assertions over 16 files, and layer 2 failed exactly one file
;;; -- this one -- with rove reporting "Raise an error while testing" and a `TYPE-ERROR' raised
;;; by `(LENGTH (a . b))' inside the `%SET-FEATURE-NAMES' that `make-dataset' had just called.
;;;
;;; That raw `type-error' is the milder of the two things the guard heads off: `(length
;;; circular)' does not return at all, measured on SBCL 2.6.7. `%set-feature-names' computes
;;; that length one line under the call.
;;;
;;; `handler-case', not rove's `signals' -- see this file's other guard tests for why. The
;;; condition TYPE is asserted and so is the argument it names, since several of
;;; `make-dataset''s arguments signal this same type.

(deftest lightgbm-api-make-dataset-improper-feature-names-signals
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (declare (ignore label-vector))
      (let ((backend (cl-gbdt:open-backend :lightgbm)))
        (unwind-protect
             (let ((condition
                     (handler-case
                         (progn (cl-gbdt:make-dataset backend matrix
                                                      :feature-names '("a" . "b")
                                                      :parameters *dataset-parameters*)
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

;;; `close-backend' calls `cffi:close-foreign-library', which may unmap the shared
;;; library from the process; POSIX does not guarantee it, but does not forbid it
;;; either. Before this fix, a handle stored only its backend's keyword name, so
;;; nothing connected a live dataset back to the backend that owned it, and any
;;; operation after `close-backend' ran foreign code that might no longer be mapped --
;;; a segfault, not a catchable condition. `handle-live-pointer' now checks
;;; `backend-open-p' before returning a pointer, turning that into `backend-not-open'.
;;; This proves it does, using `dataset-num-rows' as a representative operation.

(deftest lightgbm-api-operation-after-close-backend-signals-backend-not-open
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let* ((backend (cl-gbdt:open-backend :lightgbm))
             (dataset (cl-gbdt:make-dataset backend matrix
                                             :label label-vector
                                             :parameters *dataset-parameters*)))
        (cl-gbdt:close-backend backend)
        (unwind-protect
             (testing "dataset-num-rows after close-backend signals backend-not-open"
               ;; handler-case, not rove's `signals' -- see this file's other guard
               ;; tests for why.
               (ok (handler-case (progn (cl-gbdt:dataset-num-rows dataset) nil)
                     (cl-gbdt:backend-not-open () t))
                   "dataset-num-rows did not signal backend-not-open"))
          (cl-gbdt:free-dataset dataset))))))

;;; `free-dataset' is deliberately the exception to the guard above: it runs from
;;; `with-dataset''s `unwind-protect' cleanup, and signalling `backend-not-open' there
;;; during a non-local exit would replace the condition already unwinding the stack
;;; instead of letting it propagate. It must also not crash -- the exact hazard this
;;; whole finding is about -- so it does not call `LGBM_DatasetFree' once the backend
;;; is closed; the handle is just marked released and a warning notes the leak. This
;;; proves freeing after `close-backend' neither signals nor brings the process down.

(deftest lightgbm-api-free-dataset-after-close-backend-does-not-signal
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let* ((backend (cl-gbdt:open-backend :lightgbm))
             (dataset (cl-gbdt:make-dataset backend matrix
                                             :label label-vector
                                             :parameters *dataset-parameters*)))
        (cl-gbdt:close-backend backend)
        (testing "free-dataset after close-backend does not signal"
          (ok (handler-case (progn (cl-gbdt:free-dataset dataset) t)
                (error () nil))
              "free-dataset signaled after its backend was closed"))))))

;;; F2: `make-dataset', `train' and `load-model' each create a brand-new handle
;;; directly from a backend, unlike every other operation in this file, which reads
;;; an existing handle and so is already covered by `handle-live-pointer''s
;;; `backend-open-p' check. Before this fix, none of the three checked
;;; `backend-open-p' itself: closing a backend and then calling one of them reached
;;; its first foreign call -- `LGBM_DatasetCreateFromMat', `LGBM_BoosterCreate' or
;;; `LGBM_BoosterCreateFromModelfile' -- against a library that might no longer be
;;; mapped. This proves each of the three now signals `backend-not-open' instead,
;;; one entry point per test.

(deftest lightgbm-api-make-dataset-after-close-backend-signals-backend-not-open
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm)))
        (cl-gbdt:close-backend backend)
        (testing "make-dataset after close-backend signals backend-not-open"
          ;; handler-case, not rove's `signals' -- see this file's other guard tests
          ;; for why.
          (ok (handler-case
                  (progn (cl-gbdt:make-dataset backend matrix :label label-vector
                                                :parameters *dataset-parameters*)
                         nil)
                (cl-gbdt:backend-not-open () t))
              "make-dataset did not signal backend-not-open"))))))

(deftest lightgbm-api-train-after-close-backend-signals-backend-not-open
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      ;; Two separate :lightgbm backend instances, not one: DATASET's own backend
      ;; (OPEN-BACKEND) stays open for the whole test, so handle-live-pointer's own
      ;; backend-open-p check -- already guarding every read of an existing handle
      ;; since the previous round -- cannot be what makes this test pass. Only
      ;; train's own check on its BACKEND argument, CLOSED-BACKEND, can. One
      ;; instance closed while the other's shared library is still loaded is safe:
      ;; both wrap the same underlying dlopen'd library, and close-backend's effect
      ;; being tested here is the Lisp-level open flag, not the mapping itself.
      (let ((closed-backend (cl-gbdt:open-backend :lightgbm))
            (open-backend (cl-gbdt:open-backend :lightgbm))
            (dataset nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset open-backend matrix
                                                    :label label-vector
                                                    :parameters *dataset-parameters*))
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

(deftest lightgbm-api-load-model-after-close-backend-signals-backend-not-open
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (dataset nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf dataset (cl-gbdt:make-dataset backend matrix
                                                    :label label-vector
                                                    :parameters *dataset-parameters*))
               (setf booster (cl-gbdt:train backend dataset :num-rounds 1
                                             :parameters *booster-parameters*))
               (uiop:with-temporary-file (:pathname path :type "txt")
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

;;; P1: `make-dataset' used to always pass a null `reference' to
;;; `LGBM_DatasetCreateFromMat', so a validation dataset always computed its own,
;;; independent bin mapper -- regardless of what training set it was meant to validate
;;; against. `LGBM_BoosterAddValidData' refuses to attach a validation dataset whose bin
;;; mapper differs from the training set's; confirmed directly against the vendored
;;; library using *reference-training-values* and *reference-validation-values* above,
;;; which share no values and so bin independently. `train''s :valid-sets was therefore
;;; unusable for real held-out data -- it only appeared to work in
;;; `lightgbm-api-round-trip' because that test reuses the same matrix for both training
;;; and validation, which happens to produce identical bins. This proves the fix: a
;;; validation dataset built with `:reference' set to the training dataset attaches
;;; successfully.

(deftest lightgbm-api-make-dataset-with-reference-attaches-as-valid-set
  (with-backend-library (:lightgbm)
    (let ((backend (cl-gbdt:open-backend :lightgbm))
          (training-set nil)
          (validation-set nil)
          (booster nil))
      (unwind-protect
           (progn
             (setf training-set
                   (cl-gbdt:make-dataset
                    backend (%single-column-matrix *reference-training-values*)
                    :label *reference-training-labels*
                    :parameters *dataset-parameters*))
             (setf validation-set
                   (cl-gbdt:make-dataset
                    backend (%single-column-matrix *reference-validation-values*)
                    :reference training-set
                    :parameters *dataset-parameters*))
             (testing "train attaches a validation set built with :reference"
               (ok (handler-case
                       (progn (setf booster
                                    (cl-gbdt:train backend training-set :num-rounds 1
                                                   :valid-sets (list validation-set)
                                                   :parameters *booster-parameters*))
                              t)
                     (cl-gbdt:foreign-call-error () nil))
                   "train signaled foreign-call-error for a :reference-built valid set")))
        (progn
          (when booster (cl-gbdt:free-booster booster))
          (when validation-set (cl-gbdt:free-dataset validation-set))
          (when training-set (cl-gbdt:free-dataset training-set))
          (cl-gbdt:close-backend backend))))))

;;; Same two distributions, but the validation dataset is built without :reference this
;;; time -- documenting why the option above exists, and, since this is the same
;;; assertion as before the fix, would notice a future change that silently stopped
;;; threading REFERENCE through to `LGBM_DatasetCreateFromMat'.

(deftest lightgbm-api-make-dataset-without-reference-valid-set-is-refused
  (with-backend-library (:lightgbm)
    (let ((backend (cl-gbdt:open-backend :lightgbm))
          (training-set nil)
          (validation-set nil))
      (unwind-protect
           (progn
             (setf training-set
                   (cl-gbdt:make-dataset
                    backend (%single-column-matrix *reference-training-values*)
                    :label *reference-training-labels*
                    :parameters *dataset-parameters*))
             (setf validation-set
                   (cl-gbdt:make-dataset
                    backend (%single-column-matrix *reference-validation-values*)
                    :parameters *dataset-parameters*))
             (testing "train refuses a valid set with its own, independently-binned mapper"
               (ok (handler-case
                       (progn (cl-gbdt:train backend training-set :num-rounds 1
                                              :valid-sets (list validation-set)
                                              :parameters *booster-parameters*)
                              nil)
                     (cl-gbdt:foreign-call-error () t))
                   "train did not signal foreign-call-error for a mismatched bin mapper")))
        (progn
          (when validation-set (cl-gbdt:free-dataset validation-set))
          (when training-set (cl-gbdt:free-dataset training-set))
          (cl-gbdt:close-backend backend))))))

;;; `%reference-pointer' reads :reference through `handle-live-pointer' before handing
;;; its pointer to `LGBM_DatasetCreateFromMat', the same guard every other handle read in
;;; this file goes through: a freed reference passed straight to C would be a segfault,
;;; not a catchable condition. This proves the guard runs.

(deftest lightgbm-api-make-dataset-with-freed-reference-signals
  (with-backend-library (:lightgbm)
    (let ((backend (cl-gbdt:open-backend :lightgbm))
          (training-set nil))
      (unwind-protect
           (progn
             (setf training-set
                   (cl-gbdt:make-dataset
                    backend (%single-column-matrix *reference-training-values*)
                    :label *reference-training-labels*
                    :parameters *dataset-parameters*))
             (cl-gbdt:free-dataset training-set)
             (testing "make-dataset with a freed :reference signals released-handle-error"
               (ok (handler-case
                       (progn (cl-gbdt:make-dataset
                               backend (%single-column-matrix *reference-validation-values*)
                               :reference training-set
                               :parameters *dataset-parameters*)
                              nil)
                     (cl-gbdt:released-handle-error () t))
                   "make-dataset did not signal released-handle-error for a freed :reference")))
        (cl-gbdt:close-backend backend)))))

;;; The cross-backend guard is reachable today, without an XGBoost `make-dataset' to
;;; produce a foreign dataset: any handle that is not a `lightgbm-dataset' trips it, and a
;;; booster is one. Worth asserting now rather than when XGBoost lands, because what this
;;; guards against -- handing `LGBM_DatasetCreateFromMat' a pointer to something that is
;;; not a LightGBM dataset -- is undefined behaviour the library cannot reject for us.

(deftest lightgbm-api-make-dataset-with-a-non-dataset-reference-signals
  (with-backend-library (:lightgbm)
    (let ((backend (cl-gbdt:open-backend :lightgbm))
          (matrix (%single-column-matrix *reference-training-values*)))
      (unwind-protect
           (cl-gbdt:with-dataset (training-set (cl-gbdt:make-dataset
                                                 backend matrix
                                                 :label *reference-training-labels*
                                                 :parameters *dataset-parameters*))
             (cl-gbdt:with-booster (booster (cl-gbdt:train
                                              backend training-set :num-rounds 1
                                              :parameters *booster-parameters*))
               (testing "make-dataset with a booster as :reference signals"
                 (ok (handler-case
                         (progn (cl-gbdt:make-dataset
                                 backend (%single-column-matrix *reference-validation-values*)
                                 :reference booster
                                 :parameters *dataset-parameters*)
                                nil)
                       (cl-gbdt:wrong-backend-reference () t))
                     "make-dataset accepted a booster as :reference"))))
        (cl-gbdt:close-backend backend)))))

;;; Round 4: `train' dispatches on BACKEND, not on DATASET, so unlike
;;; `dataset-num-rows' or `free-dataset' there was no CLOS specializer ruling out a
;;; booster passed where a dataset belongs. `handle-live-pointer' does not check
;;; kind, only liveness and the owning backend's open state, so a booster's own
;;; pointer used to reach `LGBM_BoosterCreate' as its training-set argument --
;;; confirmed directly against the vendored library to corrupt SBCL's heap and kill
;;; the process outright, not raise a catchable condition. This proves `train' now
;;; rejects it with `wrong-backend-reference' before any foreign call runs.

(deftest lightgbm-api-train-with-a-booster-as-dataset-signals
  (with-backend-library (:lightgbm)
    (let ((backend (cl-gbdt:open-backend :lightgbm))
          (matrix (%single-column-matrix *reference-training-values*)))
      (unwind-protect
           (cl-gbdt:with-dataset (training-set (cl-gbdt:make-dataset
                                                 backend matrix
                                                 :label *reference-training-labels*
                                                 :parameters *dataset-parameters*))
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
        (cl-gbdt:close-backend backend)))))

;;; Same hazard, reached through :valid-sets instead of DATASET itself:
;;; `LGBM_BoosterAddValidData' dereferences each entry's pointer directly, and
;;; `%add-valid-data' used to check only liveness, never kind. This proves a
;;; booster inside :valid-sets is rejected the same way, before
;;; `LGBM_BoosterAddValidData' -- or even `LGBM_BoosterCreate' -- ever runs.

(deftest lightgbm-api-train-with-a-booster-in-valid-sets-signals
  (with-backend-library (:lightgbm)
    (let ((backend (cl-gbdt:open-backend :lightgbm))
          (matrix (%single-column-matrix *reference-training-values*)))
      (unwind-protect
           (cl-gbdt:with-dataset (training-set (cl-gbdt:make-dataset
                                                 backend matrix
                                                 :label *reference-training-labels*
                                                 :parameters *dataset-parameters*))
             (cl-gbdt:with-booster (other-booster (cl-gbdt:train
                                                    backend training-set :num-rounds 1
                                                    :parameters *booster-parameters*))
               (testing "train with a booster inside :valid-sets signals, naming a ~
                         dataset as what was expected"
                 ;; Regression: `wrong-backend-reference' gained an EXPECTED slot so
                 ;; `booster-eval'/`booster-eval-names' could report "booster" instead of
                 ;; "dataset" -- this asserts the pre-existing :valid-sets path still
                 ;; reports "dataset", its own default, unaffected by that addition.
                 (let ((message (handler-case
                                     (progn (cl-gbdt:train backend training-set :num-rounds 1
                                                            :valid-sets (list other-booster)
                                                            :parameters *booster-parameters*)
                                            nil)
                                   (cl-gbdt:wrong-backend-reference (c) (princ-to-string c)))))
                   (ok message "train accepted a booster inside :valid-sets")
                   (ok (and message (search "must be a dataset built by" message))
                       (format nil "report was ~S" message))))))
        (cl-gbdt:close-backend backend)))))

;;; Task 2: `cl-gbdt/lightgbm:booster-eval-names' and `cl-gbdt/lightgbm:booster-eval', the
;;; LightGBM backend's first Layer 1 (backend-specific safe API) functions -- see
;;; docs/cl-gbdt-layered-api-implementation-policy.md section 3 and
;;; docs/superpowers/specs/2026-08-06-evaluation-api-design.md. Both are plain functions
;;; exported directly from `cl-gbdt/lightgbm', not `defmethod's reached through `cl-gbdt',
;;; so they are exercised here, alongside this file's other round trips through the public
;;; API, rather than in tests/functional/lightgbm.lisp, which never wraps a raw pointer in a
;;; handle object at all.

(defparameter *eval-booster-parameters*
  '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1
    :metric "binary_logloss,auc")
  "LightGBM booster parameters for the evaluation tests below. Two metrics, not one, so a
test that only checked a single name/value pair could not pass by accident if
`booster-eval' or `booster-eval-names' silently dropped every entry after the first.")

;;; Step 3 of this task's brief: metric names come from `LGBM_BoosterGetEvalNames' and their
;;; count matches `LGBM_BoosterGetEvalCounts'. *EVAL-BOOSTER-PARAMETERS* configures exactly
;;; two metrics, "binary_logloss" and "auc", in that order -- LightGBM reports metric names
;;; in configuration order, confirmed directly against the vendored library -- so this checks
;;; order as well as count, which a test using a single default metric could not distinguish
;;; from a name/value pair silently transposed or dropped.

(deftest lightgbm-api-booster-eval-names-and-values-match-configured-metrics
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (train-set nil)
            (valid-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix
                                                       :label label-vector
                                                       :parameters *dataset-parameters*))
               (setf valid-set (cl-gbdt:make-dataset backend matrix
                                                      :label label-vector
                                                      :parameters *dataset-parameters*))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 5
                                             :valid-sets (list valid-set)
                                             :parameters *eval-booster-parameters*))
               (testing "booster-eval-names reports both configured metrics, in order"
                 (ok (equal '("binary_logloss" "auc")
                            (cl-gbdt/lightgbm:booster-eval-names booster))
                     (format nil "booster-eval-names returned ~S"
                             (cl-gbdt/lightgbm:booster-eval-names booster))))
               (testing "booster-eval on the training set (index 0) has one finite, ~
                         metric-appropriate value per metric"
                 (let ((values (cl-gbdt/lightgbm:booster-eval booster 0)))
                   (ok (= 2 (length values)) (format nil "length was ~D" (length values)))
                   (ok (plusp (aref values 0))
                       (format nil "binary_logloss was ~S" (aref values 0)))
                   (ok (<= 0.0d0 (aref values 1) 1.0d0)
                       (format nil "auc was ~S" (aref values 1)))))
               (testing "booster-eval on the validation set (index 1) has one value per metric"
                 (let ((values (cl-gbdt/lightgbm:booster-eval booster 1)))
                   (ok (= (length (cl-gbdt/lightgbm:booster-eval-names booster)) (length values))
                       (format nil "length was ~D" (length values))))))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when valid-set (cl-gbdt:free-dataset valid-set))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; Design doc section 7's explicit ask: a test that would fail if evaluation were faked.
;;; Every other assertion in this file about `booster-eval' checks shape and range, both of
;;; which a function returning plausible constants could pass by accident. Training the same
;;; separable fixture for one round versus thirty, then reading `binary_logloss' back off the
;;; same data each was trained on, cannot: confirmed directly against the vendored library,
;;; one round's `binary_logloss' is ~0.598 and thirty rounds' is ~0.024 on this fixture -- the
;;; metric has to fall as the model actually fits the data better, which a stub has no way to
;;; reproduce without actually training the booster.

(deftest lightgbm-api-booster-eval-tracks-actual-model-fit
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (train-set nil)
            (undertrained nil)
            (overtrained nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix
                                                       :label label-vector
                                                       :parameters *dataset-parameters*))
               (setf undertrained (cl-gbdt:train backend train-set :num-rounds 1
                                                  :parameters *eval-booster-parameters*))
               (setf overtrained (cl-gbdt:train backend train-set :num-rounds 30
                                                 :parameters *eval-booster-parameters*))
               (testing "more boosting rounds lowers binary_logloss on the training set"
                 (let ((loss-1 (aref (cl-gbdt/lightgbm:booster-eval undertrained 0) 0))
                       (loss-30 (aref (cl-gbdt/lightgbm:booster-eval overtrained 0) 0)))
                   (ok (< loss-30 loss-1)
                       (format nil "1 round: ~S, 30 rounds: ~S" loss-1 loss-30)))))
          (progn
            (when undertrained (cl-gbdt:free-booster undertrained))
            (when overtrained (cl-gbdt:free-booster overtrained))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; `metric=none' is LightGBM's own way to configure zero evaluation metrics -- confirmed
;;; directly against the vendored library, `LGBM_BoosterGetEvalCounts' reports 0 and
;;; `LGBM_BoosterGetEval' reports an empty result rather than erroring. `%booster-eval-names'
;;; skips its own foreign call in that case, while `%booster-eval' still makes its, with a
;;; one-double buffer it never reads from -- see both functions' docstrings in native.lisp.
;;; This proves the public functions return the expected empty values, not a signal or a
;;; garbage-filled array.

(deftest lightgbm-api-booster-eval-with-no-metrics-configured-returns-empty
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix
                                                       :label label-vector
                                                       :parameters *dataset-parameters*))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 1
                                             :parameters '(:objective "binary" :num-leaves 2
                                                           :min-data-in-leaf 1
                                                           :min-data-in-bin 1 :verbose -1
                                                           :metric "None")))
               (testing "booster-eval-names is empty when no metric is configured"
                 (ok (null (cl-gbdt/lightgbm:booster-eval-names booster))
                     (format nil "got ~S" (cl-gbdt/lightgbm:booster-eval-names booster))))
               (testing "booster-eval on the training set is a zero-length array"
                 (let ((values (cl-gbdt/lightgbm:booster-eval booster 0)))
                   (ok (typep values '(simple-array double-float (0)))
                       (format nil "got ~S" values)))))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; DATA-INDEX 0 always means the training set; with no :valid-sets attached, index 1 is out
;;; of range -- confirmed directly against the vendored library, `LGBM_BoosterGetEval' rejects
;;; it with a "Check failed: data_idx >= 0 && data_idx <= ..." message rather than reading
;;; past the end of an internal array. This proves `booster-eval' turns that into
;;; `foreign-call-error' rather than crashing or returning a bogus result.

(deftest lightgbm-api-booster-eval-out-of-range-data-index-signals
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix
                                                       :label label-vector
                                                       :parameters *dataset-parameters*))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 1
                                             :parameters *eval-booster-parameters*))
               (testing "booster-eval with no attached validation set at index 1 signals"
                 ;; handler-case, not rove's `signals' -- see this file's other guard
                 ;; tests for why.
                 (ok (handler-case (progn (cl-gbdt/lightgbm:booster-eval booster 1) nil)
                       (cl-gbdt:foreign-call-error () t))
                     "booster-eval did not signal foreign-call-error for an out-of-range index")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; `booster-eval' and `booster-eval-names' are plain functions, not `defmethod's -- unlike
;;; every other operation this file guards against a mismatched handle, CLOS dispatch does not
;;; rule out the wrong kind on its own here. `%check-lightgbm-booster' is the explicit check
;;; that stands in for it; this proves it runs before either foreign call, using a dataset
;;; handle -- the same one `train''s own DATASET argument accepts -- as the wrong kind.

(deftest lightgbm-api-booster-eval-with-a-dataset-as-booster-argument-signals
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (train-set nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix
                                                       :label label-vector
                                                       :parameters *dataset-parameters*))
               (testing "booster-eval-names with a dataset as its argument signals, ~
                         naming a booster as what was expected"
                 ;; Regression: the old report format hardcoded "dataset" even when
                 ;; rejecting an actual dataset because a booster was required, e.g.
                 ;; "must be a dataset built by LIGHTGBM itself, not LIGHTGBM-DATASET" --
                 ;; self-contradictory. `%check-lightgbm-booster' now passes `:expected
                 ;; "booster"'; this asserts the report says so.
                 (let ((message (handler-case
                                     (progn (cl-gbdt/lightgbm:booster-eval-names train-set)
                                            nil)
                                   (cl-gbdt:wrong-backend-reference (c) (princ-to-string c)))))
                   (ok message "booster-eval-names accepted a dataset as its booster argument")
                   (ok (and message (search "must be a booster built by" message))
                       (format nil "report was ~S" message))))
               (testing "booster-eval with a dataset as its argument signals, naming a ~
                         booster as what was expected"
                 (let ((message (handler-case
                                     (progn (cl-gbdt/lightgbm:booster-eval train-set 0) nil)
                                   (cl-gbdt:wrong-backend-reference (c) (princ-to-string c)))))
                   (ok message "booster-eval accepted a dataset as its booster argument")
                   (ok (and message (search "must be a booster built by" message))
                       (format nil "report was ~S" message)))))
          (progn
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; `booster-eval' and `booster-eval-names' read BOOSTER's pointer through
;;; `handle-live-pointer', the same guard every other operation in this file goes through --
;;; this proves it, mirroring `lightgbm-api-free-dataset-releases-the-handle''s proof for
;;; datasets.

(deftest lightgbm-api-booster-eval-after-free-booster-signals
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix
                                                       :label label-vector
                                                       :parameters *dataset-parameters*))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 1
                                             :parameters *eval-booster-parameters*))
               (cl-gbdt:free-booster booster)
               (testing "booster-eval-names on a freed booster signals released-handle-error"
                 (ok (handler-case (progn (cl-gbdt/lightgbm:booster-eval-names booster) nil)
                       (cl-gbdt:released-handle-error () t))
                     "booster-eval-names did not signal released-handle-error"))
               (testing "booster-eval on a freed booster signals released-handle-error"
                 (ok (handler-case (progn (cl-gbdt/lightgbm:booster-eval booster 0) nil)
                       (cl-gbdt:released-handle-error () t))
                     "booster-eval did not signal released-handle-error")))
          (progn
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; Same guard as the free-booster proof above, for a closed backend instead -- mirroring
;;; `lightgbm-api-operation-after-close-backend-signals-backend-not-open''s proof for
;;; `dataset-num-rows'.

(deftest lightgbm-api-booster-eval-after-close-backend-signals
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let* ((backend (cl-gbdt:open-backend :lightgbm))
             (train-set (cl-gbdt:make-dataset backend matrix
                                               :label label-vector
                                               :parameters *dataset-parameters*))
             (booster (cl-gbdt:train backend train-set :num-rounds 1
                                      :parameters *eval-booster-parameters*)))
        (cl-gbdt:close-backend backend)
        (unwind-protect
             (progn
               (testing "booster-eval-names after close-backend signals backend-not-open"
                 (ok (handler-case (progn (cl-gbdt/lightgbm:booster-eval-names booster) nil)
                       (cl-gbdt:backend-not-open () t))
                     "booster-eval-names did not signal backend-not-open"))
               (testing "booster-eval after close-backend signals backend-not-open"
                 (ok (handler-case (progn (cl-gbdt/lightgbm:booster-eval booster 0) nil)
                       (cl-gbdt:backend-not-open () t))
                     "booster-eval did not signal backend-not-open")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))))))))

;;; Codex flagged this on PR #11: `booster-eval' is a public entry point, so a caller can
;;; reach `LGBM_BoosterGetEval' without going through the `evaluation' method that guards
;;; the retained datasets. `LGBM_DatasetFree' clears none of the dataset pointers the
;;; booster stores, so evaluating after one of them was freed is a segfault, not a
;;; catchable condition -- the identical hazard `update-one-iteration' has always guarded
;;; against, and the one the `evaluation' method guards at Layer 2. These two tests hold
;;; the Layer 1 entry point to the same standard for both the training set and a
;;; validation set.

(deftest lightgbm-api-booster-eval-after-training-set-freed-signals
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (train-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix
                                                      :label label-vector
                                                      :parameters *dataset-parameters*))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 1
                                             :parameters *eval-booster-parameters*))
               (cl-gbdt:free-dataset train-set)
               (testing "booster-eval on a booster whose training set was freed signals"
                 ;; handler-case, not rove's `signals' -- `signals' does not reliably
                 ;; catch conditions raised inside `restart-case', a documented pitfall
                 ;; in this repo's own prompts/repl-driven-development.md.
                 (ok (handler-case (progn (cl-gbdt/lightgbm:booster-eval booster 0) nil)
                       (cl-gbdt:released-handle-error () t))
                     "booster-eval did not signal released-handle-error")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (cl-gbdt:close-backend backend)))))))

(deftest lightgbm-api-booster-eval-after-validation-set-freed-signals
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((backend (cl-gbdt:open-backend :lightgbm))
            (train-set nil)
            (valid-set nil)
            (booster nil))
        (unwind-protect
             (progn
               (setf train-set (cl-gbdt:make-dataset backend matrix
                                                      :label label-vector
                                                      :parameters *dataset-parameters*))
               (setf valid-set (cl-gbdt:make-dataset backend matrix
                                                      :label label-vector
                                                      :parameters *dataset-parameters*))
               (setf booster (cl-gbdt:train backend train-set :num-rounds 1
                                             :valid-sets (list valid-set)
                                             :parameters *eval-booster-parameters*))
               (cl-gbdt:free-dataset valid-set)
               (testing "booster-eval at the training index still notices the freed ~
                         validation set"
                 ;; Index 0 is the training set, which is still live -- the guard has to
                 ;; cover every dataset the booster retains, not just the one addressed,
                 ;; because LightGBM builds its metric objects over all of them.
                 (ok (handler-case (progn (cl-gbdt/lightgbm:booster-eval booster 0) nil)
                       (cl-gbdt:released-handle-error () t))
                     "booster-eval did not signal released-handle-error"))
               (testing "booster-eval at the freed validation set's own index signals"
                 (ok (handler-case (progn (cl-gbdt/lightgbm:booster-eval booster 1) nil)
                       (cl-gbdt:released-handle-error () t))
                     "booster-eval did not signal released-handle-error")))
          (progn
            (when booster (cl-gbdt:free-booster booster))
            (when train-set (cl-gbdt:free-dataset train-set))
            (cl-gbdt:close-backend backend)))))))

;;; Task 7 (docs/superpowers/sdd/2026-08-11-layer1-separation): `cl-gbdt/lightgbm' now
;;; re-exports the shared-basis symbols -- `open-backend', `backend-open-p', `backend-name',
;;; `backend-supports-p', `close-backend' and the whole `cl-gbdt/src/conditions' hierarchy --
;;; that a caller who loads this backend ALONE, never `cl-gbdt' itself, needs to work with it
;;; and catch what it signals without naming an internal package. Every other test in this
;;; file goes through `cl-gbdt:', proving the unified API; this one goes through
;;; `cl-gbdt/lightgbm:' throughout, proving the standalone surface added by this task instead.
;;; No `:path' is passed to `open-backend' here, matching every other test above: this test's
;;; own `with-backend-library' has already loaded the vendored library via
;;; `cffi:load-foreign-library', so CFFI's own default resolution finds it without one.

(deftest the-backend-package-publishes-what-a-standalone-caller-needs
  (testing "opening, asking and closing are all reachable without naming an internal package"
    (with-backend-library (:lightgbm)
      (let ((backend (cl-gbdt/lightgbm:open-backend :lightgbm)))
        (unwind-protect
             (progn
               (ok (cl-gbdt/lightgbm:backend-open-p backend) "the backend reports itself open")
               (ok (eq :lightgbm (cl-gbdt/lightgbm:backend-name backend)) "and names itself")
               (ok (typep (cl-gbdt/lightgbm:backend-supports-p backend :sparse-input) 'boolean)
                   "a capability question answers")
               (ok (subtypep 'cl-gbdt/lightgbm:capability-unavailable
                             'cl-gbdt/lightgbm:gbdt-error)
                   "the condition hierarchy is reachable from this package too"))
          (cl-gbdt/lightgbm:close-backend backend))))))
