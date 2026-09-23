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
;;;; NILs -- and is both the return contract's and the constructor's argument domain. It has no
;;;; generator (an `and' of a type and a `satisfies' has no strategy), which is no loss: every
;;;; contract here draws its calls from a whole-call generator, and those draw values with
;;;; `draw-values'. An earlier version kept a second, generable spec for the argument, which
;;;; admitted an adjustable vector the constructor then stored as given (G8 in
;;;; docs/cl-spec-dogfooding.md). Declared domains here are the documented ones, unbounded;
;;;; the generators alone keep draws small.
;;;;
;;;; The constructors take only keyword arguments, and a keyword the caller omits is NIL. So
;;;; each contract names, with a supplied-p variable and `:pre', the keys whose NIL no slot
;;;; admits; without that it would promise a well-formed result for `(make-training-series)'.
;;;; A generated keyword call may omit any key, and under that `:pre' 175 of 200 generated calls
;;;; to `make-training-series' and 150 of 200 to `make-training-report' were refused at seed 42
;;;; (docs/cl-spec-dogfooding.md, G5), so whole-call generators supply those keys -- which
;;;; changes what is sampled, not what the contract admits.

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
                #:draw-finite-double)
  (:export #:draw-metric
           #:draw-name
           #:draw-values
           #:series-values
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
                :collect (if (zerop (random 3)) nil (draw-finite-double)))
          'simple-vector))

(defun draw-name ()
  "Return NIL or a short fresh string."
  (if (zerop (random 2)) nil (format nil "valid-~D" (random 10))))

(defspec series-values
  (and (type simple-vector) (vector-of (satisfies double-or-nil-p))))

(defun draw-metric ()
  "Return a fresh copy of one of `*metrics*', as a backend's own string would be."
  (copy-seq (nth (random (length *metrics*)) *metrics*)))

(defgenerator training-series-generator ()
  (make-training-series :index (random 4) :name (draw-name) :metric (draw-metric)
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
  ;; Every required key, always; NAME only sometimes, so its NIL default is exercised too.
  (append (list :index (if (zerop (random 4)) (random 100000) (random 21))
                :metric (draw-metric) :values (draw-values))
          (when (zerop (random 2)) (list :name (draw-name)))))

(defspec-function make-training-series
  "Given its index, metric and values, the series reports back through its readers the same
contents it was built from."
  ;; The slots' own documented domains, unbounded; the generator keeps draws small.
  (:args &key ((:index index) (range integer 0 *) index-p)
              ((:name name) (nullable string))
              ((:metric metric) string metric-p)
              ((:values values) series-values values-p))
  (:args-generator make-training-series-arguments)
  ;; NAME may be omitted: NIL is the documented unnamed series. The other three default to
  ;; NIL too, which no series slot admits, so a call without them is outside the contract.
  (:pre index-p metric-p values-p)
  (:returns training-series-object)
  (:post (and (eql (training-series-index result) index)
              (equal (training-series-name result) name)
              (equal (training-series-metric result) metric)
              (equalp (training-series-values result) values))))

(defgenerator make-training-report-arguments ()
  ;; NUM-ROUNDS always; each other key only sometimes, so their NIL defaults are exercised.
  (let ((rounds (if (zerop (random 4)) (random 100000) (random 51))))
    (append (list :num-rounds rounds)
            (when (zerop (random 2))
              (list :series (loop :repeat (random 4)
                                  :collect (make-training-series
                                            :index (random 4) :name (draw-name)
                                            :metric (draw-metric) :values (draw-values)))))
            (when (zerop (random 2))
              (list :best-iteration (if (zerop (random 3)) nil (random (1+ rounds)))
                    :best-score (if (zerop (random 3)) nil (draw-finite-double))))
            (when (zerop (random 2))
              (list :early-stopped-p (zerop (random 2)))))))

(defspec-function make-training-report
  "Given its round count, the report reports back through its readers the same contents it
was built from."
  ;; The slots' own documented domains, unbounded -- a best score may be an infinity, which a
  ;; custom evaluation records for a value too large for a double; the generator keeps draws
  ;; small and finite.
  (:args &key ((:series series) (list-of training-series-object))
              ((:num-rounds num-rounds) (range integer 0 *) num-rounds-p)
              ((:best-iteration best-iteration) (nullable (range integer 0 *)))
              ((:best-score best-score) (nullable real))
              ((:early-stopped-p early-stopped-p) boolean))
  (:args-generator make-training-report-arguments)
  ;; Every other key may be omitted: NIL is an empty series list and "not determined" for the
  ;; three early-stopping fields. A report with no round count is outside the contract.
  (:pre num-rounds-p)
  (:returns training-report-object)
  (:post (and (equal (training-report-series result) series)
              (eql (training-report-num-rounds result) num-rounds)
              (eql (training-report-best-iteration result) best-iteration)
              (eql (training-report-best-score result) best-score)
              (eq (training-report-early-stopped-p result) early-stopped-p))))
