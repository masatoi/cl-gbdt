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
iteration earlier."))
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
   (best-iteration :initform nil
                   :reader training-report-best-iteration
                   :documentation "NIL in this phase. Filled by Phase 3b.

Finding the best iteration means knowing whether a metric improves upward or downward, and
policy section 9 forbids inferring that from the metric's name. NIL says \"not determined\",
which is not the same as iteration 0.")
   (best-score :initform nil
               :reader training-report-best-score
               :documentation "NIL in this phase. Filled by Phase 3b, alongside
`training-report-best-iteration' and for the same reason.")
   (early-stopped-p :initform nil
                    :reader training-report-early-stopped-p
                    :documentation "NIL in this phase: no training run can stop early yet.
Filled by Phase 3b."))
  (:documentation "What a training run recorded, returned as `train''s secondary value.

Policy section 9 asks a training report to express the dataset, the metric, the per-iteration
values, the best iteration, the best score, and whether early stopping happened. The first
three are here now; the last three are Phase 3b's and read NIL until then. A class rather than
a plist so that Phase 3b filling them adds slots without breaking a caller, and so each field
can say what it means."))

(defun make-training-series (&key index name metric values)
  "Return a `training-series' for METRIC on the dataset at INDEX, optionally called NAME.

VALUES is one element per completed iteration; see the slot's own documentation for why a NIL
element is legal."
  (make-instance 'training-series :index index :name name :metric metric :values values))

(defun make-training-report (&key series num-rounds)
  "Return a `training-report' over SERIES, recorded across NUM-ROUNDS iterations.

`training-report-best-iteration', `-best-score' and `-early-stopped-p' are NIL: Phase 3b fills
them, and there is no honest value for them before it does."
  (make-instance 'training-report :series series :num-rounds num-rounds))

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
