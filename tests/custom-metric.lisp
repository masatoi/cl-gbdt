;;;; custom-metric.lisp --- Layer 1 tests for the custom metric's pure helpers.
;;;;
;;;; Every function here is pure: no handle, no pointer, no shared library. They are the
;;;; parts of the custom-evaluation path that can be tested without either library present,
;;;; which is what keeps `foreign libraries open: NIL' true for this suite.

(uiop:define-package #:cl-gbdt/tests/custom-metric
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/training/custom-metric
                #:custom-metric-entry
                #:check-metric-name-collision
                #:make-metric-name-pin
                #:pin-metric-name)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-argument
                #:unsupported-argument-argument
                #:unsupported-argument-backend))

(in-package #:cl-gbdt/tests/custom-metric)

(deftest custom-metric-entry-builds-the-shape-the-recorder-reads
  ;; The (DATASET-INDEX METRIC-NAME VALUE) triple `%read-evaluation' returns and both
  ;; `training-report-from-history' and `observe-iteration' read.
  (ok (equal (custom-metric-entry :test-backend "my_metric" 0.25d0 0) '(0 "my_metric" 0.25d0)))
  (ok (equal (custom-metric-entry :test-backend "my_metric" 0.25d0 2) '(2 "my_metric" 0.25d0)))
  ;; NIL is how both backends already record a field they could not read as a real, and
  ;; `observe-iteration' already counts it as no improvement. It is a value, not an error.
  (ok (equal (custom-metric-entry :test-backend "my_metric" nil 1) '(1 "my_metric" nil))))

(deftest custom-metric-entry-refuses-a-name-that-is-not-a-string
  ;; The fold keys series by (DATASET-INDEX, METRIC-NAME) with `equal', and `evaluation'
  ;; reports names as strings, so a keyword or a symbol would key a series nothing else
  ;; could name.
  (ok (handler-case (progn (custom-metric-entry :test-backend :my-metric 0.25d0 0) nil)
        (unsupported-argument () t)))
  (ok (handler-case (progn (custom-metric-entry :test-backend nil 0.25d0 0) nil)
        (unsupported-argument () t))))

(deftest custom-metric-entry-refuses-a-value-that-is-neither-real-nor-nil
  (ok (handler-case (progn (custom-metric-entry :test-backend "my_metric" "0.25" 0) nil)
        (unsupported-argument () t)))
  (ok (handler-case (progn (custom-metric-entry :test-backend "my_metric" #C(1 2) 0) nil)
        (unsupported-argument () t))))

(deftest check-metric-name-collision-refuses-a-name-the-library-already-uses-here
  (let ((entries '((0 "binary_logloss" 0.5d0) (1 "binary_logloss" 0.6d0))))
    (ok (handler-case
            (progn (check-metric-name-collision :test-backend "binary_logloss" 0 entries) nil)
          (unsupported-argument () t)))))

(deftest check-metric-name-collision-allows-a-name-the-library-uses-elsewhere
  ;; The fold keys by the PAIR, so the same name at a different dataset index is a different
  ;; series and collides with nothing.
  (let ((entries '((0 "binary_logloss" 0.5d0))))
    (ok (null (multiple-value-list
               (check-metric-name-collision :test-backend "binary_logloss" 1 entries))))))

(deftest check-metric-name-collision-allows-a-different-name-at-the-same-index
  (let ((entries '((0 "binary_logloss" 0.5d0))))
    (ok (null (multiple-value-list
               (check-metric-name-collision :test-backend "my_metric" 0 entries))))))

(deftest check-metric-name-collision-allows-anything-when-no-metric-is-configured
  ;; Measured: with no metric configured `evaluation' answers NIL on both backends, so the
  ;; library-entry list this is handed is empty and nothing can collide.
  (ok (null (multiple-value-list (check-metric-name-collision :test-backend "my_metric" 0 '())))))

;;; Caught in review: a test that only checks the condition's TYPE would still pass with
;;; BACKEND-NAME dropped on the floor entirely, which is what the first version of this file
;;; did before catching up with the sibling `early-stopping.lisp' convention. These check the
;;; SLOT instead, the same way `tests/training-early-stopping.lisp''s "the condition names the
;;; backend it was actually called with, not a placeholder" test does for
;;; `train-early-stopping-watcher'.

(deftest custom-metric-entry-names-the-backend-it-was-called-on
  (let ((condition (handler-case (progn (custom-metric-entry :test-backend :my-metric 0.25d0 0)
                                         nil)
                     (unsupported-argument (c) c))))
    (ok (eq :test-backend (unsupported-argument-backend condition))
        "custom-metric-entry did not report the backend it was called with")))

(deftest check-metric-name-collision-names-the-backend-it-was-called-on
  (let* ((entries '((0 "binary_logloss" 0.5d0)))
         (condition (handler-case
                        (progn (check-metric-name-collision :test-backend "binary_logloss" 0
                                                              entries)
                               nil)
                      (unsupported-argument (c) c))))
    (ok (eq :test-backend (unsupported-argument-backend condition))
        "check-metric-name-collision did not report the backend it was called with")))

;;; ---------------------------------------------------------------------------
;;; The name pin

;;; `check-metric-name-collision' above runs on the FIRST iteration only, which is all the
;;; library's own names need -- they do not change after that. A caller's do: the same
;;; :EVALUATION may return one name on iteration 1 and another on iteration 2, and neither
;;; name is wrong on the iteration it appears in, so no per-iteration check can catch it. The
;;; pin is what makes the run remember, and the tests below are written as SEQUENCES of calls
;;; rather than single ones because that is the only shape in which the bug exists.

(deftest the-pin-accepts-one-name-per-index-for-as-long-as-it-is-repeated
  (let ((pin (make-metric-name-pin)))
    (ok (null (multiple-value-list (pin-metric-name :test-backend pin "my_metric" 0))))
    (ok (null (multiple-value-list (pin-metric-name :test-backend pin "my_metric" 0))))
    (ok (null (multiple-value-list (pin-metric-name :test-backend pin "my_metric" 0))))))

(deftest the-pin-refuses-a-name-that-changes-at-the-same-index
  ;; The ragged half: "safe" then "other" gives two series, each shorter than the run and
  ;; each misaligned with the iterations its values came from.
  (let ((pin (make-metric-name-pin)))
    (pin-metric-name :test-backend pin "safe" 0)
    (ok (handler-case (progn (pin-metric-name :test-backend pin "other" 0) nil)
          (unsupported-argument () t)))))

(deftest the-pin-refuses-a-name-that-changes-into-a-colliding-one
  ;; The long half, and the one `check-metric-name-collision' cannot reach: it runs on the
  ;; first iteration only, so a caller returning "safe" then the library's own
  ;; "binary_logloss" passes it and puts two entries under one key from iteration 2 on. The
  ;; pin refuses the change itself, which is why no second collision check is needed.
  (let ((pin (make-metric-name-pin)))
    (check-metric-name-collision :test-backend "safe" 0 '((0 "binary_logloss" 0.5d0)))
    (pin-metric-name :test-backend pin "safe" 0)
    (ok (handler-case (progn (pin-metric-name :test-backend pin "binary_logloss" 0) nil)
          (unsupported-argument () t)))))

(deftest the-pin-is-per-index-and-not-per-run
  ;; Two datasets may legitimately carry two different metric names -- the pair (INDEX, NAME)
  ;; is what a series is keyed by, so index 1's name has nothing to do with index 0's. A pin
  ;; that remembered one name for the whole run would refuse this.
  (let ((pin (make-metric-name-pin)))
    (pin-metric-name :test-backend pin "train_metric" 0)
    (ok (null (multiple-value-list (pin-metric-name :test-backend pin "valid_metric" 1))))
    ;; And each index still holds to its own afterwards, in both directions.
    (ok (handler-case (progn (pin-metric-name :test-backend pin "valid_metric" 0) nil)
          (unsupported-argument () t)))
    (ok (handler-case (progn (pin-metric-name :test-backend pin "train_metric" 1) nil)
          (unsupported-argument () t)))))

(deftest a-fresh-pin-remembers-nothing-from-another-one
  ;; `train' makes one per run. Two runs of the same :EVALUATION under two different names
  ;; must both be accepted, or a caller could not reuse a closure across runs.
  (let ((first-pin (make-metric-name-pin))
        (second-pin (make-metric-name-pin)))
    (pin-metric-name :test-backend first-pin "my_metric" 0)
    (ok (null (multiple-value-list (pin-metric-name :test-backend second-pin "other" 0))))))

(deftest the-pin-names-the-backend-it-was-called-on
  (let* ((pin (make-metric-name-pin))
         (condition (handler-case
                        (progn (pin-metric-name :test-backend pin "safe" 0)
                               (pin-metric-name :test-backend pin "other" 0)
                               nil)
                      (unsupported-argument (c) c))))
    (ok (eq :test-backend (unsupported-argument-backend condition))
        "pin-metric-name did not report the backend it was called with")
    (ok (equal "train's :evaluation" (unsupported-argument-argument condition))
        "pin-metric-name did not name train's :evaluation")))
