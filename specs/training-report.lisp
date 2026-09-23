;;;; training-report.lisp --- `object-of' specs for the training report, and its constructors.
;;;;
;;;; A report and its series are value objects observed through their readers, which is what
;;;; cl-spec's `object-of' describes: it validates an instance by reading it, without the MOP
;;;; and without constructing one. That is enough for a RETURN contract, and the two
;;;; constructors below use it as one. It is not enough for an INPUT: `object-of' generates
;;;; nothing by itself, so `training-series-object' names a generator that builds a series
;;;; through `make-training-series'.
;;;;
;;;; `series-values' is the slot's documented type -- a `simple-vector' of double-floats and
;;;; NILs -- and has no generator (an `and' of a type and a `satisfies' has no strategy).
;;;; `series-values-input' is the generable spelling of the same domain; `vector-of' happens to
;;;; produce simple-vectors. Two specs for one concept is recorded in
;;;; docs/cl-spec-dogfooding.md.
;;;;
;;;; The constructors take only keyword arguments, and a generated keyword call may omit any of
;;;; them; with `:pre' demanding the ones that matter, 175 of 200 generated calls were refused.
;;;; Their whole-call generators always supply every key instead.

(uiop:define-package #:cl-gbdt/specs/training-report
  (:use #:cl)
  (:import-from #:cl-spec/main
                #:defgenerator
                #:defspec
                #:defspec-function)
  (:import-from #:cl-gbdt/src/training-report
                #:make-training-report
                #:make-training-series
                #:training-report
                #:training-report-best-iteration
                #:training-report-best-score
                #:training-report-early-stopped-p
                #:training-report-num-rounds
                #:training-report-series
                #:training-series
                #:training-series-index
                #:training-series-metric
                #:training-series-name
                #:training-series-values)
  (:import-from #:cl-gbdt/specs/values
                #:finite-double)
  (:export #:series-values
           #:series-values-input
           #:training-report-object
           #:training-series-object))

(in-package #:cl-gbdt/specs/training-report)

(defparameter *metrics* '("l2" "auc" "binary_logloss")
  "Metric names the generators draw from.")

(defun double-or-nil-p (object)
  "True when OBJECT is a `double-float' or NIL -- one element of a series' values."
  (or (null object) (typep object 'double-float)))

(defun draw-values ()
  "Return a fresh simple-vector of zero to five doubles and NILs."
  (coerce (loop :repeat (random 6)
                :collect (if (zerop (random 3))
                             nil
                             (* (float (- (random 2001) 1000) 1d0) (expt 10d0 (- (random 7) 3)))))
          'simple-vector))

(defun draw-name ()
  "Return NIL or a short fresh string."
  (if (zerop (random 2)) nil (format nil "valid-~D" (random 10))))

(defspec series-values
  (and (type simple-vector) (vector-of (satisfies double-or-nil-p))))

(defspec series-values-input
  (vector-of (nullable finite-double) :max-length 6))

(defgenerator training-series-generator ()
  (make-training-series :index (random 4) :name (draw-name)
                        :metric (nth (random (length *metrics*)) *metrics*)
                        :values (draw-values)))

(defspec training-series-object
  (object-of training-series
    (:required (training-series-index (range integer 0 *))
               (training-series-name (nullable string))
               (training-series-metric string)
               (training-series-values series-values)))
  (:generator training-series-generator))

(defspec training-report-object
  (object-of training-report
    (:required (training-report-series (list-of training-series-object))
               (training-report-num-rounds (range integer 0 *))
               (training-report-best-iteration (nullable (range integer 0 *)))
               (training-report-best-score (nullable real))
               (training-report-early-stopped-p boolean))))

(defgenerator make-training-series-arguments ()
  (list :index (random 21) :name (draw-name)
        :metric (nth (random (length *metrics*)) *metrics*) :values (draw-values)))

(defspec-function make-training-series
  "The series reports back, through its readers, exactly what it was built from."
  (:args &key ((:index index) (range integer 0 20))
              ((:name name) (nullable string))
              ((:metric metric) string)
              ((:values values) series-values-input))
  (:args-generator make-training-series-arguments)
  (:returns training-series-object)
  (:post (and (eql (training-series-index result) index)
              (eq (training-series-name result) name)
              (eq (training-series-metric result) metric)
              (eq (training-series-values result) values))))

(defgenerator make-training-report-arguments ()
  (list :series (loop :repeat (random 4)
                      :collect (make-training-series
                                :index (random 4) :name (draw-name)
                                :metric (nth (random (length *metrics*)) *metrics*)
                                :values (draw-values)))
        :num-rounds (random 51)
        :best-iteration (if (zerop (random 2)) nil (random 51))
        :best-score (if (zerop (random 2)) nil (float (random 1000) 1d0))
        :early-stopped-p (zerop (random 2))))

(defspec-function make-training-report
  "The report reports back, through its readers, exactly what it was built from."
  (:args &key ((:series series) (list-of training-series-object))
              ((:num-rounds num-rounds) (range integer 0 50))
              ((:best-iteration best-iteration) (nullable (range integer 0 50)))
              ((:best-score best-score) (nullable finite-double))
              ((:early-stopped-p early-stopped-p) boolean))
  (:args-generator make-training-report-arguments)
  (:returns training-report-object)
  (:post (and (eq (training-report-series result) series)
              (eql (training-report-num-rounds result) num-rounds)
              (eql (training-report-best-iteration result) best-iteration)
              (eql (training-report-best-score result) best-score)
              (eq (training-report-early-stopped-p result) early-stopped-p))))
