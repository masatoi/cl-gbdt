;;;; evaluation.lisp --- Portable contract tests for `cl-gbdt:evaluation'.
;;;;
;;;; Policy section 13's first test category -- "同一のテストを両backendに適用し" -- and
;;;; design section 7's "the same assertions against both backends". Every test below runs
;;;; the identical assertions over *FIXTURES*, once per backend, so the two backends cannot
;;;; drift apart in shape, order or meaning without one of them failing here.
;;;;
;;;; This is the functional suite's first backend-neutral file, and it is deliberately not
;;;; part of either backend's own. tests/functional/lightgbm.lisp and xgboost.lisp call the
;;;; raw FFI bindings; lightgbm-api.lisp and xgboost-api.lisp go through the public API but
;;;; each is about ONE backend -- their own headers say so, and Tasks 2 and 3 put each
;;;; backend's Layer 1 evaluation-primitive tests (`booster-eval' for LightGBM,
;;;; `evaluate-one-iteration' for XGBoost) in the matching one. A portable contract test
;;;; belongs in neither: asserting XGBoost's behavior inside a file named for LightGBM would
;;;; miscategorise it, and writing the assertions twice, once per backend file, is exactly
;;;; the drift "the same assertions against both backends" exists to rule out. Backend-
;;;; specific evaluation behavior -- LightGBM's metric names against
;;;; `LGBM_BoosterGetEvalCounts', XGBoost's raw string against its own parse -- stays in
;;;; those two files, where it already is.
;;;;
;;;; Numbers are never compared BETWEEN backends here: policy section 13 asks for shape,
;;;; order and meaning, not numeric agreement, and the two libraries train different models
;;;; from the same rows by design.

(uiop:define-package #:cl-gbdt/tests/functional/evaluation
  (:use #:cl #:rove)
  ;; Zero symbols: every reference below is package-qualified. Declared so this file's
  ;; dependency on the unified API is explicit rather than inherited from the backend
  ;; systems below, matching the identical clause in lightgbm-api.lisp.
  (:import-from #:cl-gbdt)
  ;; Zero symbols, both of them: their only job is to run at load time and register
  ;; :lightgbm and :xgboost with `open-backend' -- see `register-backend' in each backend's
  ;; protocol.lisp. Without these clauses, package-inferred-system has no edge to those
  ;; files at all and `(cl-gbdt:open-backend :lightgbm)' below would signal
  ;; `unknown-backend'. This is the one file in the suite that needs both, being the one
  ;; that runs the same assertions against both.
  (:import-from #:cl-gbdt/src/lightgbm/all)
  (:import-from #:cl-gbdt/src/xgboost/all)
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library
                #:make-separable-dataset))

(in-package #:cl-gbdt/tests/functional/evaluation)

(defparameter *fixtures*
  (list
   (list :backend :lightgbm
         :dataset-parameters '(:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)
         :booster-parameters '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1
                               :min-data-in-bin 1 :verbose -1 :metric "binary_logloss,auc")
         :loss-metric "binary_logloss"
         :aligns-bin-mappers t
         :model-file-type "txt"
         :value-source :library-doubles)
   (list :backend :xgboost
         :dataset-parameters nil
         :booster-parameters '(:objective "binary:logistic" :max-depth 2 :eta 0.5
                               :verbosity 0 :min-child-weight 0
                               :eval-metric "logloss" :eval-metric "error")
         :loss-metric "logloss"
         :aligns-bin-mappers nil
         :model-file-type "json"
         :value-source :parsed-text))
  "One entry per backend: what each needs to train a two-metric booster on this suite's
eight-row separable fixture, and the name each spells its own log loss with.

The two parameter plists are not translations of each other and are not meant to be: each
is the parameter vocabulary its own library documents, which is the point of `train''s
:PARAMETERS being an escape hatch (policy section 6). What the tests below assert is that
the RESULT of `cl-gbdt:evaluation' has the same shape either way, not that the inputs do.

:LOSS-METRIC is that backend's own name for log loss -- LightGBM's \"binary_logloss\" and
XGBoost's \"logloss\" -- used by the fit-ordering test, which needs a metric that falls
as a model fits better, and by the entry-per-dataset test's index-discrimination check,
which needs a metric whose value differs between the training set and an
`INVERT-LABELS'-built validation set. Two metrics are configured, not one, so a portable
assertion about one metric per dataset cannot pass by accident against an implementation
that dropped every entry after the first.

:DATASET-PARAMETERS is NIL for XGBoost because that backend signals `unsupported-argument'
for `make-dataset''s :PARAMETERS rather than accepting and ignoring it; :ALIGNS-BIN-MAPPERS
is false there for the same reason, applied to :REFERENCE. LightGBM needs both: its
defaults refuse to bin or split eight rows, and a validation set it will accept has to
share the training set's bin mapper. `make-fixture-dataset' passes each keyword only to
the backend that has one.

:MODEL-FILE-TYPE is the extension each library's `save-model' writes -- XGBoost decides
its serialization format from it and rejects an unknown one -- matching the two backend
files' own save/load round trips.

:VALUE-SOURCE is what this backend's own `evaluation' method sets `cl-gbdt:evaluation''s
second return value's `:VALUE-SOURCE' to -- LightGBM's own doubles versus XGBoost's
parsed text, see `cl-gbdt:evaluation''s docstring. Pinning it here means a backend that
started reporting the other backend's provenance would fail the fixture-neutral test
below rather than pass by construction.")

(defun make-fixture-dataset (fixture backend matrix label &key reference)
  "Build a dataset on BACKEND from MATRIX and LABEL, passing only the `make-dataset'
keywords FIXTURE's backend accepts -- see *FIXTURES* for which those are and why the other
backend refuses them. REFERENCE is honoured only by a backend that aligns bin mappers, and
ignored by one that does not, so one call site works for both."
  (apply #'cl-gbdt:make-dataset backend matrix :label label
         (append (let ((parameters (getf fixture :dataset-parameters)))
                   (when parameters (list :parameters parameters)))
                 (when (and reference (getf fixture :aligns-bin-mappers))
                   (list :reference reference)))))

(defun metric-names (entries index)
  "Return the metric names ENTRIES -- a `cl-gbdt:evaluation' result -- carries for the
dataset at INDEX, in the order they appear."
  (loop :for entry :in entries
        :when (eql (first entry) index)
          :collect (second entry)))

(defun entry-value (entries index metric-name)
  "Return the value ENTRIES carries for METRIC-NAME on the dataset at INDEX, or NIL when
there is no such entry."
  (third (find-if (lambda (entry)
                    (and (eql (first entry) index)
                         (string= (second entry) metric-name)))
                  entries)))

(defun finite-double-p (value)
  "True when VALUE is a `double-float' that is neither an infinity nor a NaN.

`sb-ext:float-nan-p' rather than `(= value value)': SBCL enables the :INVALID
floating-point trap by default on x86-64, where comparing a NaN signals instead of
answering false. That is the same trap convention `with-foreign-float-traps-masked'
suspends around a foreign call, and this check deliberately runs outside one -- the
values it inspects have already been copied into Lisp."
  (and (typep value 'double-float)
       (not (sb-ext:float-nan-p value))
       (not (sb-ext:float-infinity-p value))))

(defun invert-labels (label-vector)
  "Return a fresh `(simple-array single-float (*))' with each of LABEL-VECTOR's 0.0/1.0
entries flipped to the other class.

Used to build a validation set whose labels disagree with the training set's, over the
same feature matrix, so a booster fit to LABEL-VECTOR scores clearly worse against the
result than against its own training labels -- dataset 0 and dataset 1 are then not
numerically identical by construction, which they would be if the validation set were
built from LABEL-VECTOR unchanged (both backends' bin-mapper alignment only cares about
the feature matrix, never labels, so reusing it while inverting labels is safe)."
  (map '(simple-array single-float (*)) (lambda (label) (if (zerop label) 1.0 0.0))
       label-vector))

;;; The shape, order and meaning assertion (design section 7): one entry per metric per
;;; dataset, names non-empty, values finite, and the secondary value saying which of the
;;; two ways the numbers were produced. Two datasets and two metrics on each backend, so
;;; an implementation that dropped, duplicated or transposed any of the four (dataset,
;;; metric) pairs fails here rather than looking plausible.

(deftest evaluation-reports-one-entry-per-metric-per-dataset
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (valid-label-vector (invert-labels label-vector)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix valid-label-vector
                                                      :reference train-set))
                   (cl-gbdt:with-booster
                       (booster (cl-gbdt:train backend train-set :num-rounds 5
                                               :valid-sets (list valid-set)
                                               :parameters (getf fixture :booster-parameters)))
                     (multiple-value-bind (entries provenance) (cl-gbdt:evaluation booster)
                       (let ((names (metric-names entries 0))
                             (backend-name (getf fixture :backend))
                             (metric (getf fixture :loss-metric)))
                         (testing (format nil "~A: every entry is a (DATASET-INDEX ~
                                               METRIC-NAME VALUE) list" backend-name)
                           (ok (and entries
                                    (every (lambda (entry)
                                             (and (= 3 (length entry))
                                                  (integerp (first entry))
                                                  (stringp (second entry))
                                                  (plusp (length (second entry)))))
                                           entries))
                               (format nil "entries were ~S" entries)))
                         (testing (format nil "~A: the training set is dataset 0 and the ~
                                               one :valid-sets entry is dataset 1, each ~
                                               metric once, in that order" backend-name)
                           (ok (equal (append (make-list (length names) :initial-element 0)
                                              (make-list (length names) :initial-element 1))
                                      (mapcar #'first entries))
                               (format nil "indices were ~S" (mapcar #'first entries))))
                         (testing (format nil "~A: the same metrics, in the same order, ~
                                               for both datasets" backend-name)
                           (ok (and names (equal names (metric-names entries 1)))
                               (format nil "dataset 0 had ~S, dataset 1 had ~S"
                                       names (metric-names entries 1))))
                         (let ((training-loss (entry-value entries 0 metric))
                               (validation-loss (entry-value entries 1 metric)))
                           (testing (format nil "~A: dataset 1's ~A differs from dataset ~
                                                 0's -- the inverted-label validation set ~
                                                 built by INVERT-LABELS proves index 0 and ~
                                                 index 1 are not the same evaluation ~
                                                 reported twice" backend-name metric)
                             (ok (and training-loss validation-loss
                                      (/= training-loss validation-loss))
                                 (format nil "dataset 0 ~A was ~S, dataset 1 ~A was ~S"
                                         metric training-loss metric validation-loss)))
                           (when (eq (getf fixture :backend) :lightgbm)
                             (testing (format nil "~A: the inverted-label validation set ~
                                                   at index 1 fits worse than the ~
                                                   training set at index 0" backend-name)
                               (ok (and training-loss validation-loss
                                        (> validation-loss training-loss))
                                   (format nil "dataset 0 ~A was ~S, dataset 1 ~A was ~S"
                                           metric training-loss metric validation-loss)))))
                         (testing (format nil "~A: every value is a finite double-float"
                                          backend-name)
                           (ok (every (lambda (entry) (finite-double-p (third entry)))
                                      entries)
                               (format nil "values were ~S" (mapcar #'third entries))))
                         (testing (format nil "~A: the secondary value says where the ~
                                               numbers came from, and matches this ~
                                               backend's own way of producing them"
                                          backend-name)
                           (ok (eq (getf provenance :value-source)
                                   (getf fixture :value-source))
                               (format nil "expected ~S, provenance was ~S"
                                       (getf fixture :value-source) provenance)))
                         (testing (format nil "~A: a parsed backend keeps the raw text it ~
                                               parsed, a library-doubles one has none"
                                          backend-name)
                           (if (eq (getf provenance :value-source) :parsed-text)
                               (ok (let ((raw (getf provenance :raw)))
                                     (and (stringp raw)
                                          (plusp (length raw))
                                          (every (lambda (name) (search name raw)) names)))
                                   (format nil "raw was ~S, metrics ~S"
                                           (getf provenance :raw) names))
                               (ok (null (getf provenance :raw))
                                   (format nil "provenance was ~S" provenance)))))))))
            (cl-gbdt:close-backend backend)))))))

;;; A booster trained with no :valid-sets at all still has its training set to evaluate --
;;; dataset 0, and nothing else. Both libraries configure a metric from the objective even
;;; when the caller names none, so this is a real result, not an empty one.

(deftest evaluation-without-validation-sets-reports-the-training-set-only
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-booster
                     (booster (cl-gbdt:train backend train-set :num-rounds 3
                                             :parameters (getf fixture :booster-parameters)))
                   (let ((entries (cl-gbdt:evaluation booster)))
                     (testing (format nil "~A: every entry is dataset 0, and there is at ~
                                           least one" (getf fixture :backend))
                       (ok (and entries
                                (every (lambda (entry) (eql 0 (first entry))) entries))
                           (format nil "entries were ~S" entries))))))
            (cl-gbdt:close-backend backend)))))))

;;; A `load-model' booster retains no dataset -- no training set, no validation sets -- so
;;; there is nothing for the portable contract to index. Both backends return an empty
;;; result rather than inventing a dataset 0 for a model file that never had one:
;;; confirmed directly against both vendored libraries, LightGBM answers `data_idx' 0 for
;;; such a booster with an empty value list and no configured metric names, and XGBoost's
;;; `XGBoosterEvalOneIter' with no DMatrices returns only its bare iteration marker.

(deftest evaluation-of-a-load-model-booster-is-empty
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (uiop:with-temporary-file (:pathname path
                                            :type (getf fixture :model-file-type))
                   (cl-gbdt:with-booster
                       (booster (cl-gbdt:train backend train-set :num-rounds 3
                                               :parameters (getf fixture :booster-parameters)))
                     (cl-gbdt:save-model booster path))
                   (cl-gbdt:with-booster (loaded (cl-gbdt:load-model backend path))
                     (testing (format nil "~A: a load-model booster evaluates to an empty ~
                                           result" (getf fixture :backend))
                       (ok (null (cl-gbdt:evaluation loaded))
                           (format nil "entries were ~S" (cl-gbdt:evaluation loaded)))))))
            (cl-gbdt:close-backend backend)))))))

;;; Design section 7's explicit ask, and the one assertion in this file a stub could not
;;; pass: train the same data for one round and for thirty, then read the log loss back
;;; off the same validation set. A `cl-gbdt:evaluation' that returned plausible constants
;;; -- right shape, right names, right index order, finite values -- satisfies every other
;;; test here and fails this one, which was confirmed by actually stubbing it (see this
;;; task's report). The validation set is built over the same rows as the training set on
;;; purpose: this asserts the metric tracks fit, so the fit has to improve on the very
;;; data being measured, which overfitting on held-out rows would not guarantee. Dataset 1,
;;; not 0, so the number read back comes from the :valid-sets entry rather than the
;;; training set -- on LightGBM that is a different `data_idx', on XGBoost a different
;;; DMatrix, and a portable API that quietly evaluated only the training set would pass
;;; the same assertion made at index 0.

(deftest evaluation-orders-two-models-by-their-fit
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (metric (getf fixture :loss-metric)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix label-vector
                                                      :reference train-set))
                   (cl-gbdt:with-booster
                       (undertrained
                        (cl-gbdt:train backend train-set :num-rounds 1
                                       :valid-sets (list valid-set)
                                       :parameters (getf fixture :booster-parameters)))
                     (cl-gbdt:with-booster
                         (overtrained
                          (cl-gbdt:train backend train-set :num-rounds 30
                                         :valid-sets (list valid-set)
                                         :parameters (getf fixture :booster-parameters)))
                       (let ((loss-1 (entry-value (cl-gbdt:evaluation undertrained) 1 metric))
                             (loss-30 (entry-value (cl-gbdt:evaluation overtrained) 1 metric)))
                         (testing (format nil "~A: thirty rounds beat one on ~A, read off ~
                                               the validation set"
                                          (getf fixture :backend) metric)
                           (ok (and loss-1 loss-30 (< loss-30 loss-1))
                               (format nil "1 round: ~S, 30 rounds: ~S" loss-1 loss-30))))))))
            (cl-gbdt:close-backend backend)))))))

;;; Policy section 13's "released handle" line, for the two handles evaluation reads. A
;;; freed validation set is the dangerous one: both libraries evaluate it through memory
;;; the dataset owns and neither clears the booster's reference to it when it is freed, so
;;; an unguarded call here would be a use-after-free rather than a condition. `handler-case',
;;; not rove's `signals', which does not reliably catch a condition raised inside
;;; `restart-case'.

(deftest evaluation-on-a-freed-handle-signals-released-handle-error
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (train-set nil)
              (valid-set nil)
              (booster nil))
          (unwind-protect
               (progn
                 (setf train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (setf valid-set (make-fixture-dataset fixture backend matrix label-vector
                                                       :reference train-set))
                 (setf booster (cl-gbdt:train backend train-set :num-rounds 3
                                              :valid-sets (list valid-set)
                                              :parameters (getf fixture :booster-parameters)))
                 (cl-gbdt:free-dataset valid-set)
                 (testing (format nil "~A: evaluation after its validation set was freed ~
                                       signals" (getf fixture :backend))
                   (ok (handler-case (progn (cl-gbdt:evaluation booster) nil)
                         (cl-gbdt:released-handle-error () t))
                       "evaluation did not signal for a freed validation set"))
                 (cl-gbdt:free-booster booster)
                 (testing (format nil "~A: evaluation on a freed booster signals"
                                  (getf fixture :backend))
                   (ok (handler-case (progn (cl-gbdt:evaluation booster) nil)
                         (cl-gbdt:released-handle-error () t))
                       "evaluation did not signal released-handle-error for a freed booster")))
            (progn
              (when booster (cl-gbdt:free-booster booster))
              (when valid-set (cl-gbdt:free-dataset valid-set))
              (when train-set (cl-gbdt:free-dataset train-set))
              (cl-gbdt:close-backend backend))))))))

;;; Policy section 13's "closed backend" line: the same guard every other operation on a
;;; handle goes through, since the shared library the pointers are backed by may no longer
;;; be mapped once `close-backend' has run.

(deftest evaluation-after-close-backend-signals-backend-not-open
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let* ((backend (cl-gbdt:open-backend (getf fixture :backend)))
               (train-set (make-fixture-dataset fixture backend matrix label-vector))
               (booster (cl-gbdt:train backend train-set :num-rounds 3
                                       :parameters (getf fixture :booster-parameters))))
          (cl-gbdt:close-backend backend)
          (unwind-protect
               (testing (format nil "~A: evaluation after close-backend signals"
                                (getf fixture :backend))
                 (ok (handler-case (progn (cl-gbdt:evaluation booster) nil)
                       (cl-gbdt:backend-not-open () t))
                     "evaluation did not signal backend-not-open"))
            ;; Both frees warn that the foreign memory is leaked -- it genuinely is, the
            ;; library having been closed out from under them -- and mark the handles
            ;; released so their finalizers do not warn a second time at GC. Same shape as
            ;; the two backend files' own close-backend guard tests.
            (progn
              (cl-gbdt:free-booster booster)
              (cl-gbdt:free-dataset train-set))))))))
