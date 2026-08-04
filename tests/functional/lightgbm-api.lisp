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
  ;; Zero symbols: nothing below refers to this package by name. Its only job is to
  ;; run at load time and register :lightgbm with `open-backend' -- see
  ;; `register-backend' at the bottom of src/lightgbm/backend.lisp. Without this
  ;; clause, package-inferred-system has no edge to that file at all, and
  ;; `(cl-gbdt:open-backend :lightgbm)' below would signal `unknown-backend'.
  (:import-from #:cl-gbdt/src/lightgbm/backend)
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library
                #:make-separable-dataset
                #:predictions-separate-p))

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
                     "update-one-iteration did not signal released-handle-error after ~
                      the caller truncated its own valid-sets list")))
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
               (testing "train with a booster inside :valid-sets signals"
                 (ok (handler-case
                         (progn (cl-gbdt:train backend training-set :num-rounds 1
                                                :valid-sets (list other-booster)
                                                :parameters *booster-parameters*)
                                nil)
                       (cl-gbdt:wrong-backend-reference () t))
                     "train accepted a booster inside :valid-sets"))))
        (cl-gbdt:close-backend backend)))))
