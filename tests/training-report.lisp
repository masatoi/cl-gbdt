;;;; training-report.lisp --- Tests for the training report classes.
;;;;
;;;; A report is data: these need no shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/training-report
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/training-report)

(defun %series (&key (index 0) name (metric "logloss") (values #(0.5d0 0.4d0)))
  (cl-gbdt:make-training-series :index index :name name :metric metric :values values))

(deftest training-series-reports-what-it-was-built-from
  (testing "every accessor returns its initarg unchanged"
    (let ((series (%series :index 1 :name "valid" :metric "auc" :values #(0.6d0))))
      (ok (= 1 (cl-gbdt:training-series-index series)) "index")
      (ok (equal "valid" (cl-gbdt:training-series-name series)) "name")
      (ok (equal "auc" (cl-gbdt:training-series-metric series)) "metric")
      (ok (equalp #(0.6d0) (cl-gbdt:training-series-values series)) "values"))))

(deftest training-series-name-is-nil-when-unnamed
  ;; The training set is never a :VALID-SETS entry, so no caller can name it. NIL is what
  ;; that looks like -- not an invented name, and not an absent slot.
  (testing "an unnamed series carries NIL, and still carries its index"
    (let ((series (%series :index 0)))
      (ok (null (cl-gbdt:training-series-name series)) "name was not NIL")
      (ok (= 0 (cl-gbdt:training-series-index series)) "index was lost"))))

(deftest training-series-values-may-hold-nil
  ;; Policy section 5: a value the backend could not parse is reported as unreadable rather
  ;; than dropped or replaced, and the series stays aligned with the iteration count. This is
  ;; why VALUES is a simple-vector rather than a (vector double-float).
  (testing "a NIL element is stored and does not shorten the series"
    (let ((series (%series :values (vector 0.5d0 nil 0.3d0))))
      (ok (= 3 (length (cl-gbdt:training-series-values series))) "length changed")
      (ok (null (aref (cl-gbdt:training-series-values series) 1)) "NIL element was lost"))))

(deftest training-report-reports-what-it-was-built-from
  (testing "series and num-rounds read back"
    (let* ((series (list (%series)))
           (report (cl-gbdt:make-training-report :series series :num-rounds 2)))
      (ok (eq series (cl-gbdt:training-report-series report)) "series")
      (ok (= 2 (cl-gbdt:training-report-num-rounds report)) "num-rounds"))))

(deftest training-report-phase-3b-slots-are-nil
  ;; Best iteration cannot be computed without knowing whether a metric improves upward or
  ;; downward, and policy section 9 forbids inferring that from the metric's name. Phase 3b
  ;; settles it; until then these are NIL, which says "not determined", not "iteration 0".
  (testing "best-iteration, best-score and early-stopped-p are NIL"
    (let ((report (cl-gbdt:make-training-report :series nil :num-rounds 0)))
      (ok (null (cl-gbdt:training-report-best-iteration report)) "best-iteration")
      (ok (null (cl-gbdt:training-report-best-score report)) "best-score")
      (ok (null (cl-gbdt:training-report-early-stopped-p report)) "early-stopped-p"))))

(deftest training-report-prints-readably-enough-to-debug
  (testing "print-object names the series count and the rounds"
    (let ((text (princ-to-string
                 (cl-gbdt:make-training-report :series (list (%series)) :num-rounds 7))))
      (ok (search "1" text) "the series count is not in the printed form")
      (ok (search "7" text) "the round count is not in the printed form"))))
