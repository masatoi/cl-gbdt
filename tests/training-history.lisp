;;;; training-history.lisp --- Tests for the per-iteration-history fold.
;;;;
;;;; `training-report-from-history' is pure Lisp over the normalized (DATASET-INDEX
;;;; METRIC-NAME VALUE) entry shape, so these need no shared library (layer 1). Both
;;;; backends' `train' methods reach the same function with the same shape, and
;;;; tests/functional/training-report.lisp asserts the end-to-end result against the real
;;;; libraries; what is asserted here is the fold's own contract -- the ordering and
;;;; alignment properties that would otherwise be observable only through a library, and
;;;; only for the entry shapes those two libraries happen to produce.

(uiop:define-package #:cl-gbdt/tests/training-history
  (:use #:cl #:rove)
  ;; Zero symbols: `cl-gbdt' is where the report accessors under test come from, and every
  ;; reference below is package-qualified.
  (:import-from #:cl-gbdt)
  (:import-from #:cl-gbdt/src/training/history
                #:training-report-from-history))

(in-package #:cl-gbdt/tests/training-history)

(defun series-of (report index metric-name)
  "Return REPORT's series for METRIC-NAME on the dataset at INDEX, or NIL when absent."
  (find-if (lambda (series)
             (and (eql index (cl-gbdt:training-series-index series))
                  (string= metric-name (cl-gbdt:training-series-metric series))))
           (cl-gbdt:training-report-series report)))

(defun keys-of (report)
  "Return REPORT's series as a list of (INDEX METRIC-NAME) lists, in series order."
  (mapcar (lambda (series)
            (list (cl-gbdt:training-series-index series)
                  (cl-gbdt:training-series-metric series)))
          (cl-gbdt:training-report-series report)))

(deftest history-yields-one-series-per-dataset-and-metric-pair
  ;; Two datasets x two metrics, recorded for three iterations: four series, not one per
  ;; dataset, not one per metric, and not twelve.
  (testing "each (index, metric) pair becomes exactly one series"
    (let ((report (training-report-from-history
                   (loop :repeat 3
                         :collect (list (list 0 "logloss" 0.5d0) (list 0 "auc" 0.6d0)
                                        (list 1 "logloss" 0.7d0) (list 1 "auc" 0.8d0)))
                   3 '(nil nil))))
      (ok (= 4 (length (cl-gbdt:training-report-series report)))
          (format nil "series were ~S" (cl-gbdt:training-report-series report)))
      (ok (equal '((0 "logloss") (0 "auc") (1 "logloss") (1 "auc")) (keys-of report))
          (format nil "keys were ~S" (keys-of report))))))

(deftest history-orders-series-by-first-appearance-not-by-sorting
  ;; The property the fold exists to preserve: `training-report-series' must list the same
  ;; pairs in the same order `evaluation' reports its entries in, so a caller who can read
  ;; one can read the other. This input is chosen so that sorting on ANY of the obvious
  ;; keys -- index, metric name, or the pair -- gives a different answer than first-seen
  ;; order does, which is what makes it discriminating rather than merely descriptive.
  (testing "series follow the entry order of the first iteration"
    (let ((report (training-report-from-history
                   (loop :repeat 2
                         :collect (list (list 1 "zeta" 0.1d0) (list 0 "alpha" 0.2d0)
                                        (list 1 "alpha" 0.3d0) (list 0 "zeta" 0.4d0)))
                   2 '(nil nil))))
      (ok (equal '((1 "zeta") (0 "alpha") (1 "alpha") (0 "zeta")) (keys-of report))
          (format nil "keys were ~S" (keys-of report))))))

(deftest history-records-one-value-per-iteration-in-order
  (testing "a series holds each iteration's value, in iteration order"
    (let* ((report (training-report-from-history
                    (list (list (list 0 "logloss" 0.9d0))
                          (list (list 0 "logloss" 0.5d0))
                          (list (list 0 "logloss" 0.2d0)))
                    3 '(nil)))
           (series (series-of report 0 "logloss")))
      (ok series "no series was built for (0, logloss)")
      (ok (equalp #(0.9d0 0.5d0 0.2d0) (cl-gbdt:training-series-values series))
          (format nil "values were ~S"
                  (and series (cl-gbdt:training-series-values series)))))))

(deftest history-keeps-a-nil-value-in-its-own-slot
  ;; Policy section 5: a field the backend could not read as a real is reported as
  ;; unreadable rather than dropped or replaced. Dropping it would also slide every later
  ;; value one iteration earlier, which is the failure this asserts against -- the series
  ;; must still be three long and 0.2d0 must still be iteration 3's value. This is the
  ;; reason `training-series-values' is a `simple-vector'.
  (testing "a NIL value neither shortens the series nor shifts its neighbours"
    (let* ((report (training-report-from-history
                    (list (list (list 0 "logloss" 0.9d0))
                          (list (list 0 "logloss" nil))
                          (list (list 0 "logloss" 0.2d0)))
                    3 '(nil)))
           (series (series-of report 0 "logloss")))
      (ok (= 3 (length (cl-gbdt:training-series-values series)))
          (format nil "values were ~S" (cl-gbdt:training-series-values series)))
      (ok (null (aref (cl-gbdt:training-series-values series) 1)) "the NIL slot was lost")
      (ok (= 0.2d0 (aref (cl-gbdt:training-series-values series) 2))
          "iteration 3's value moved"))))

(deftest history-of-a-run-with-no-metrics-is-empty-but-still-a-run
  ;; A booster with no metric configured -- LightGBM's `metric=none', XGBoost's
  ;; `disable_default_eval_metric=1' -- records an empty evaluation every iteration. That
  ;; is an empty series list, not a report that forgot how many rounds ran, which is why
  ;; NUM-ROUNDS is taken as given rather than derived from what was recorded.
  (testing "no entries recorded, but the round count survives"
    (let ((report (training-report-from-history (list '() '() '() '()) 4 '())))
      (ok (null (cl-gbdt:training-report-series report))
          (format nil "series were ~S" (cl-gbdt:training-report-series report)))
      (ok (= 4 (cl-gbdt:training-report-num-rounds report))
          (format nil "num-rounds was ~S" (cl-gbdt:training-report-num-rounds report))))))

(deftest history-of-a-zero-round-run-is-empty
  ;; `train' with :num-rounds 0 never enters its loop, so the history is empty. Nothing to
  ;; report, and no error either.
  (testing "an empty history yields an empty report"
    (let ((report (training-report-from-history '() 0 '())))
      (ok (null (cl-gbdt:training-report-series report)) "series were not empty")
      (ok (= 0 (cl-gbdt:training-report-num-rounds report)) "num-rounds was not 0"))))
