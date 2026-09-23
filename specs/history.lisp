;;;; history.lisp --- Executable contract and Properties for `training-report-from-history'.
;;;;
;;;; The function folds a run's per-iteration evaluations into series, and its docstring makes
;;;; four separate promises. Each is its own Property, so a failure names the one broken:
;;;; one series per distinct (index, metric) pair in first-appearance order; each series'
;;;; values are its pair's values in iteration order with NIL keeping its slot; each series
;;;; carries its dataset's name; and NUM-ROUNDS and the early-stopping fields are recorded as
;;;; given (the Function Spec).
;;;;
;;;; Two domains are kept apart on purpose. The Function Spec declares what the function
;;;; ACCEPTS -- any history of (INDEX METRIC-NAME VALUE) lists, any round count, names covering
;;;; the indices -- and samples it with a whole-call generator. The Properties declare what they
;;;; SAMPLE, the `sampled-*' specs, from built-in specs so that counterexamples shrink: a
;;;; NIL-dropping implementation's failure shrank to one iteration holding one entry. Sampled
;;;; histories need not look like a real backend's -- a pair may be missing from one iteration
;;;; or repeat within one -- and the Properties are stated for that wider shape, which is what
;;;; the implementation's hash-table fold actually guarantees.

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
                #:draw-metric
                #:draw-name
                #:training-report-object)
  (:import-from #:cl-gbdt/specs/values
                #:draw-finite-double
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

;;; What the contract admits. The documented shape, unbounded: an entry is an (INDEX
;;; METRIC-NAME VALUE) list -- `(and (type list) ...)' because a `tuple' alone admits a vector,
;;; and the function destructures a list -- with VALUE a double or NIL, the two things a
;;; series' values may hold. DATASET-NAMES must cover every index used, which is the `:pre'.
(defspec history-entry
  (and (type list)
       (tuple (range integer 0 *) string (or null (type double-float)))))

(defspec evaluation-history
  (list-of (list-of history-entry)))

(defspec history-dataset-names
  (list-of (nullable string)))

;;; What the Properties sample, and so the domain their claims are checked over: small
;;; histories, indices 0..3 against four names, three metric names so that pairs repeat.
;;; Built-in generation, so a counterexample shrinks. Not what the function accepts -- that is
;;; the three specs above.
(defspec sampled-history-entry
  (and (type list)
       (tuple (range integer 0 3) metric-name (nullable finite-double))))

(defspec sampled-history
  (list-of (list-of sampled-history-entry :max-length 6) :max-length 6))

(defspec sampled-dataset-names
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

(defun names-cover-indices-p (history dataset-names)
  "True when DATASET-NAMES has an element at every dataset index HISTORY uses -- the only
positions the function reads, with `elt'."
  (loop :for entries :in history
        :always (loop :for (index) :in entries :always (< index (length dataset-names)))))

(defgenerator training-report-from-history-arguments ()
  ;; The contract's own sampling: up to six iterations of up to six entries over up to eight
  ;; datasets, and a round count and early-stopping keys independent of HISTORY's length.
  ;; A whole-call generator because DATASET-NAMES must cover HISTORY's indices.
  (let* ((datasets (1+ (random 8)))
         (history (loop :repeat (random 7)
                        :collect (loop :repeat (random 7)
                                       :collect (list (random datasets) (draw-metric)
                                                      (if (zerop (random 3))
                                                          nil
                                                          (draw-finite-double))))))
         (rounds (if (zerop (random 4)) (random 100000) (random 51))))
    (append (list history rounds (loop :repeat datasets :collect (draw-name)))
            (when (zerop (random 2))
              (list :best-iteration (if (zerop (random 3)) nil (random (1+ rounds)))
                    :best-score (if (zerop (random 3)) nil (draw-finite-double))))
            (when (zerop (random 2))
              (list :early-stopped-p (zerop (random 2)))))))

(defspec-function training-report-from-history
  "A well-formed report whose rounds and early-stopping fields are exactly what was passed."
  (:args (history evaluation-history)
         (num-rounds (range integer 0 *))
         (dataset-names history-dataset-names)
         &key ((:best-iteration best-iteration) (nullable (range integer 0 *)))
              ((:best-score best-score) (nullable real))
              ((:early-stopped-p early-stopped-p) boolean))
  (:args-generator training-report-from-history-arguments)
  (:pre (names-cover-indices-p history dataset-names))
  (:returns training-report-object)
  (:post (and (= (training-report-num-rounds result) num-rounds)
              (eql (training-report-best-iteration result) best-iteration)
              (eql (training-report-best-score result) best-score)
              (eq (training-report-early-stopped-p result) early-stopped-p))))

(defproperty history-yields-one-series-per-pair-in-first-appearance-order
    ((history sampled-history) (dataset-names sampled-dataset-names))
  "Series are HISTORY's distinct (index, metric) pairs, in the order each was first seen."
  (:about training-report-from-history)
  (:kind :invariant)
  (:trials (:smoke 20 :normal 200))
  (equal (mapcar #'series-pair
                 (training-report-series
                  (training-report-from-history history 0 dataset-names)))
         (remove-duplicates (history-pairs history) :test #'equal :from-end t)))

(defproperty history-series-values-are-the-pair-s-values-in-order
    ((history sampled-history) (dataset-names sampled-dataset-names))
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
    ((history sampled-history) (dataset-names sampled-dataset-names))
  "Every series at dataset index N carries DATASET-NAMES's element N."
  (:about training-report-from-history)
  (:kind :invariant)
  (:trials (:smoke 20 :normal 200))
  (every (lambda (series)
           (equal (training-series-name series)
                  (elt dataset-names (training-series-index series))))
         (training-report-series (training-report-from-history history 0 dataset-names))))
