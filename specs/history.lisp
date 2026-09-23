;;;; history.lisp --- Executable contract and Properties for `training-report-from-history'.
;;;;
;;;; The function folds a run's per-iteration evaluations into series, and its docstring makes
;;;; four separate promises. Each is its own Property, so a failure names the one broken:
;;;; one series per distinct (index, metric) pair in first-appearance order; each series'
;;;; values are its pair's values in iteration order with NIL keeping its slot; each series
;;;; carries its dataset's name; and NUM-ROUNDS and the early-stopping fields are recorded as
;;;; given (the Function Spec).
;;;;
;;;; HISTORY's lists and tuples are generated from built-in specs, not a whole-call generator,
;;;; so counterexamples shrink: a NIL-dropping implementation's failure shrank to one
;;;; iteration holding one entry. Only the metric name has a custom generator (see
;;;; `metric-name' for why), and it is drawn from three names so that pairs repeat.
;;;; Generated histories need not look like a real backend's -- a pair may be missing from one
;;;; iteration or repeat within one -- and the Properties are stated for that wider domain,
;;;; which is what the implementation's hash-table fold actually guarantees.

(uiop:define-package #:cl-gbdt/specs/history
  (:use #:cl)
  (:import-from #:cl-spec/main
                #:defgenerator
                #:defproperty
                #:defspec
                #:defspec-function)
  (:import-from #:cl-gbdt/src/training-report
                #:training-report-series
                #:training-report-num-rounds
                #:training-report-best-iteration
                #:training-report-best-score
                #:training-report-early-stopped-p
                #:training-series-index
                #:training-series-metric
                #:training-series-name
                #:training-series-values)
  (:import-from #:cl-gbdt/src/training/history
                #:training-report-from-history)
  (:import-from #:cl-gbdt/specs/training-report
                #:training-report-object)
  (:import-from #:cl-gbdt/specs/values
                #:finite-double)
  (:export #:history-yields-one-series-per-pair-in-first-appearance-order
           #:history-series-values-are-the-pair-s-values-in-order
           #:history-series-name-is-the-dataset-s-name))

(in-package #:cl-gbdt/specs/history)

(defgenerator metric-name-generator ()
  ;; Three names, so that generated histories repeat (index, metric) pairs; each draw is a
  ;; fresh string, as a backend's would be.
  (copy-seq (nth (random 3) '("l2" "auc" "binary_logloss"))))

;;; Any string validates: `(member "l2" ...)' would compare with EQL and so admit only the
;;; string objects written here, refusing every metric name a backend actually returns.
(defspec metric-name (type string)
  (:generator metric-name-generator))

;;; `(and (type list) ...)' because a `tuple' alone admits a vector, and the function under
;;; test destructures each entry as a list.
(defspec history-entry
  (and (type list)
       (tuple (range integer 0 3) metric-name (nullable finite-double))))

(defspec evaluation-history
  (list-of (list-of history-entry :max-length 6) :max-length 6))

(defspec history-dataset-names
  (list-of (nullable string) :min-length 4 :max-length 4))

(defun history-pairs (history)
  "Every (INDEX . METRIC) pair in HISTORY, in iteration order and then entry order."
  (loop :for entries :in history
        :append (loop :for (index metric) :in entries :collect (cons index metric))))

(defun series-pair (series)
  "SERIES's (INDEX . METRIC) pair."
  (cons (training-series-index series) (training-series-metric series)))

(defun pair-values (history index metric)
  "Every value HISTORY records for (INDEX, METRIC), in order, as a simple-vector."
  (coerce (loop :for entries :in history
                :append (loop :for (entry-index entry-metric value) :in entries
                              :when (and (= entry-index index) (string= entry-metric metric))
                                :collect value))
          'simple-vector))

(defspec-function training-report-from-history
  "A well-formed report whose rounds and early-stopping fields are exactly what was passed."
  (:args (history evaluation-history)
         (num-rounds (range integer 0 50))
         (dataset-names history-dataset-names)
         &key ((:best-iteration best-iteration) (nullable (range integer 0 50)))
              ((:best-score best-score) (nullable finite-double))
              ((:early-stopped-p early-stopped-p) boolean))
  (:returns training-report-object)
  (:post (and (= (training-report-num-rounds result) num-rounds)
              (eql (training-report-best-iteration result) best-iteration)
              (eql (training-report-best-score result) best-score)
              (eq (training-report-early-stopped-p result) early-stopped-p))))

(defproperty history-yields-one-series-per-pair-in-first-appearance-order
    ((history evaluation-history) (dataset-names history-dataset-names))
  "Series are HISTORY's distinct (index, metric) pairs, in the order each was first seen."
  (:about training-report-from-history)
  (:kind :invariant)
  (:trials (:smoke 20 :normal 200))
  (equal (mapcar #'series-pair
                 (training-report-series
                  (training-report-from-history history 0 dataset-names)))
         (remove-duplicates (history-pairs history) :test #'equal :from-end t)))

(defproperty history-series-values-are-the-pair-s-values-in-order
    ((history evaluation-history) (dataset-names history-dataset-names))
  "A series holds every value its pair was recorded with, in iteration order, NIL included."
  (:about training-report-from-history)
  (:kind :invariant)
  (:trials (:smoke 20 :normal 200))
  (every (lambda (series)
           (equalp (training-series-values series)
                   (pair-values history (training-series-index series)
                                (training-series-metric series))))
         (training-report-series (training-report-from-history history 0 dataset-names))))

(defproperty history-series-name-is-the-dataset-s-name
    ((history evaluation-history) (dataset-names history-dataset-names))
  "Every series at dataset index N carries DATASET-NAMES's element N."
  (:about training-report-from-history)
  (:kind :invariant)
  (:trials (:smoke 20 :normal 200))
  (every (lambda (series)
           (equal (training-series-name series)
                  (elt dataset-names (training-series-index series))))
         (training-report-series (training-report-from-history history 0 dataset-names))))
