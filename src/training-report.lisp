;;;; training-report.lisp --- What a training run recorded.
;;;;
;;;; Core and backend-neutral: a report is data, holds no foreign pointer, and loads without
;;;; either shared library. `train' fills one in and returns it as its secondary value.

(uiop:define-package #:cl-gbdt/src/training-report
  (:use #:cl)
  (:export #:training-series
           #:training-series-index
           #:training-series-name
           #:training-series-metric
           #:training-series-values
           #:make-training-series
           #:training-report
           #:training-report-series
           #:training-report-num-rounds
           #:training-report-best-iteration
           #:training-report-best-score
           #:training-report-early-stopped-p
           #:make-training-report))

(in-package #:cl-gbdt/src/training-report)

(defclass training-series ()
  ((index :initarg :index
          :reader training-series-index
          :documentation "The dataset's position among the datasets the booster retains: 0 is
the training set, 1 the first `train' :VALID-SETS entry, and so on. This is each backend's own
identifier -- LightGBM knows a validation set by index and by nothing else -- so it is always
present, whether or not the caller supplied a name.")
   (name :initarg :name
         :initform nil
         :reader training-series-name
         :documentation "The name the caller gave this dataset in `train''s :VALID-SETS, or NIL.

NIL for the training set, which is never a :VALID-SETS entry and so has no name a caller could
supply, and NIL for any validation set passed bare rather than as a (NAME . DATASET) cons.
Nothing here invents one.")
   (metric :initarg :metric
           :reader training-series-metric
           :documentation "The metric's name, as the backend spells it -- LightGBM's
\"binary_logloss\" and XGBoost's \"logloss\" are different strings for the same idea, and
neither is translated.")
   (values :initarg :values
           :reader training-series-values
           :documentation "One element per completed iteration, in order: a `double-float', or
NIL where the backend reported a value that could not be read as a real.

A `simple-vector' rather than a `(vector double-float)' precisely so NIL can appear. XGBoost
formats metric values through `std::ostream', which writes `nan' and `inf' for non-finite
doubles; policy section 5 says such a field is reported as unreadable rather than dropped or
replaced by an invented number, and dropping it would also slide every later value one
iteration earlier.

`double-float' holds for a `train' :EVALUATION's own values too, and holds by COERCION rather
than by luck: a caller's function may return any real, and `custom-metric-entry'
(`cl-gbdt/src/training/custom-metric') coerces what it returns before the entry is built, so
a 1/3 is recorded as 0.3333333333333333d0 rather than as a `ratio'. That is what keeps this
slot's promise the same one for every series in a report, whoever produced it."))
  (:documentation "One metric's values over one dataset, across a training run."))

(defclass training-report ()
  ((series :initarg :series
           :reader training-report-series
           :documentation "A list of `training-series', one per metric per dataset. Empty when
the booster has no metrics configured -- LightGBM's `metric=none'.")
   (num-rounds :initarg :num-rounds
               :reader training-report-num-rounds
               :documentation "How many boosting iterations actually ran. Equal to `train''s
:NUM-ROUNDS in this phase; Phase 3b's early stopping is what can make it smaller, which is why
it is recorded rather than left for the caller to assume.")
   (best-iteration :initarg :best-iteration
                   :initform nil
                   :reader training-report-best-iteration
                   :documentation "Which iteration produced `best-score', or NIL.

Filled only when `train' was given :EARLY-STOPPING; every other run -- including one that
completes normally with :EARLY-STOPPING never given at all -- leaves this NIL. Finding the
best iteration means knowing whether a metric improves upward or downward, and policy
section 9 forbids inferring that from the metric's name; :EARLY-STOPPING is what supplies
that direction explicitly, which is what makes filling this slot possible at all. NIL means
\"not determined\", never \"iteration 0\" -- a run's best iteration can genuinely be iteration
0, and NIL must stay distinguishable from that answer.")
   (best-score :initarg :best-score
               :initform nil
               :reader training-report-best-score
               :documentation "The best value `best-iteration' achieved, or NIL.

Filled under exactly the same condition as `training-report-best-iteration' and for the same
reason: only when `train' was given :EARLY-STOPPING. NIL here means \"not determined\", the
same as for `best-iteration' -- not \"no improvement was ever seen\", which would be a real,
reportable outcome once :EARLY-STOPPING is in use.")
   (early-stopped-p :initarg :early-stopped-p
                    :initform nil
                    :reader training-report-early-stopped-p
                    :documentation "True when the run stopped before `num-rounds' iterations
because :EARLY-STOPPING's patience was exhausted.

NIL in two cases this slot does not distinguish: :EARLY-STOPPING was given but the run
completed on its own before triggering it, and :EARLY-STOPPING was not given at all. Neither
case stopped early for a reason this slot needs to report, which is why both read the same
as \"not determined\" reads for `best-iteration' and `best-score'."))
  (:documentation "What a training run recorded, returned as `train''s secondary value.

Policy section 9 asks a training report to express the dataset, the metric, the per-iteration
values, the best iteration, the best score, and whether early stopping happened. All six are
here; the last three read NIL unless `train' was given :EARLY-STOPPING -- see each slot's own
documentation for why NIL is \"not determined\" rather than an invented default. A class rather
than a plist so a caller can rely on each field saying what it means regardless of whether
this run used early stopping."))

(defun make-training-series (&key index name metric values)
  "Return a `training-series' for METRIC on the dataset at INDEX, optionally called NAME.

VALUES is one element per completed iteration; see the slot's own documentation for why a NIL
element is legal."
  (make-instance 'training-series :index index :name name :metric metric :values values))

(defun make-training-report (&key series num-rounds best-iteration best-score
                              early-stopped-p)
  "Return a `training-report' over SERIES, recorded across NUM-ROUNDS iterations.

BEST-ITERATION, BEST-SCORE and EARLY-STOPPED-P default to NIL, which is the honest value for
a run that was not given :EARLY-STOPPING; a caller that was given it passes what its watcher
found. See `training-report-best-iteration', `-best-score' and `-early-stopped-p' for why NIL
means \"not determined\" rather than an invented default."
  (make-instance 'training-report :series series :num-rounds num-rounds
                 :best-iteration best-iteration :best-score best-score
                 :early-stopped-p early-stopped-p))

(defmethod print-object ((series training-series) stream)
  (print-unreadable-object (series stream :type t)
    (format stream "index ~D~@[ name ~S~] metric ~S, ~D value~:P"
            (training-series-index series)
            (training-series-name series)
            (training-series-metric series)
            (length (training-series-values series)))))

(defmethod print-object ((report training-report) stream)
  (print-unreadable-object (report stream :type t)
    (format stream "~D series, ~D round~:P"
            (length (training-report-series report))
            (training-report-num-rounds report))))
