;;;; custom-metric.lisp --- Layer 1 tests for the custom metric's pure helpers.
;;;;
;;;; Both functions here are pure: no handle, no pointer, no shared library. They are the
;;;; parts of the custom-evaluation path that can be tested without either library present,
;;;; which is what keeps `foreign libraries open: NIL' true for this suite.

(uiop:define-package #:cl-gbdt/tests/custom-metric
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/training/custom-metric
                #:custom-metric-entry
                #:check-metric-name-collision)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-argument))

(in-package #:cl-gbdt/tests/custom-metric)

(deftest custom-metric-entry-builds-the-shape-the-recorder-reads
  ;; The (DATASET-INDEX METRIC-NAME VALUE) triple `%read-evaluation' returns and both
  ;; `training-report-from-history' and `observe-iteration' read.
  (ok (equal (custom-metric-entry "my_metric" 0.25d0 0) '(0 "my_metric" 0.25d0)))
  (ok (equal (custom-metric-entry "my_metric" 0.25d0 2) '(2 "my_metric" 0.25d0)))
  ;; NIL is how both backends already record a field they could not read as a real, and
  ;; `observe-iteration' already counts it as no improvement. It is a value, not an error.
  (ok (equal (custom-metric-entry "my_metric" nil 1) '(1 "my_metric" nil))))

(deftest custom-metric-entry-refuses-a-name-that-is-not-a-string
  ;; The fold keys series by (DATASET-INDEX, METRIC-NAME) with `equal', and `evaluation'
  ;; reports names as strings, so a keyword or a symbol would key a series nothing else
  ;; could name.
  (ok (handler-case (progn (custom-metric-entry :my-metric 0.25d0 0) nil)
        (unsupported-argument () t)))
  (ok (handler-case (progn (custom-metric-entry nil 0.25d0 0) nil)
        (unsupported-argument () t))))

(deftest custom-metric-entry-refuses-a-value-that-is-neither-real-nor-nil
  (ok (handler-case (progn (custom-metric-entry "my_metric" "0.25" 0) nil)
        (unsupported-argument () t)))
  (ok (handler-case (progn (custom-metric-entry "my_metric" #C(1 2) 0) nil)
        (unsupported-argument () t))))

(deftest check-metric-name-collision-refuses-a-name-the-library-already-uses-here
  (let ((entries '((0 "binary_logloss" 0.5d0) (1 "binary_logloss" 0.6d0))))
    (ok (handler-case (progn (check-metric-name-collision "binary_logloss" 0 entries) nil)
          (unsupported-argument () t)))))

(deftest check-metric-name-collision-allows-a-name-the-library-uses-elsewhere
  ;; The fold keys by the PAIR, so the same name at a different dataset index is a different
  ;; series and collides with nothing.
  (let ((entries '((0 "binary_logloss" 0.5d0))))
    (ok (null (multiple-value-list
               (check-metric-name-collision "binary_logloss" 1 entries))))))

(deftest check-metric-name-collision-allows-a-different-name-at-the-same-index
  (let ((entries '((0 "binary_logloss" 0.5d0))))
    (ok (null (multiple-value-list (check-metric-name-collision "my_metric" 0 entries))))))

(deftest check-metric-name-collision-allows-anything-when-no-metric-is-configured
  ;; Measured: with no metric configured `evaluation' answers NIL on both backends, so the
  ;; library-entry list this is handed is empty and nothing can collide.
  (ok (null (multiple-value-list (check-metric-name-collision "my_metric" 0 '())))))
