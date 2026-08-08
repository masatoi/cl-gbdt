;;;; training-early-stopping.lisp --- Tests for the early-stopping watcher.
;;;;
;;;; The stop decision is pure: it sees one iteration's entries at a time and answers whether
;;;; to stop. No shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/training-early-stopping
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/training/early-stopping
                #:make-early-stopping-watcher
                #:observe-iteration
                #:watcher-best-iteration
                #:watcher-best-score
                #:watcher-stopped-p)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/training-early-stopping)

(defparameter *names* '(nil "valid" "test")
  "The DATASET-NAMES shape `train' builds: position 0 is the training set, which has no name.")

(defun %watcher (&key (spec '(:metric "logloss" :dataset "valid"
                              :direction :lower-is-better :rounds 2))
                      (names *names*))
  (make-early-stopping-watcher spec names))

(defun %entries (value &key (index 1) (metric "logloss"))
  "One iteration's entries: the watched series plus a decoy the watcher must not read."
  (list (list 0 metric 99.0d0)
        (list index metric value)
        (list index "auc" 0.5d0)))

(defun %signals-unsupported-p (thunk)
  ;; handler-case, not rove's `signals' -- see prompts/repl-driven-development.md.
  (handler-case (progn (funcall thunk) nil)
    (cl-gbdt:unsupported-argument () t)))

(deftest watcher-requires-every-key
  (testing "each of the four keys is required"
    (dolist (missing '(:metric :dataset :direction :rounds))
      (let ((spec (loop :for (key value) :on '(:metric "logloss" :dataset "valid"
                                               :direction :lower-is-better :rounds 2)
                          :by #'cddr
                        :unless (eq key missing) :append (list key value))))
        (ok (%signals-unsupported-p (lambda () (%watcher :spec spec)))
            (format nil "omitting ~S did not signal" missing))))))

(deftest watcher-rejects-a-bad-direction-and-a-bad-round-count
  (testing "an unknown direction signals"
    (ok (%signals-unsupported-p
         (lambda () (%watcher :spec '(:metric "logloss" :dataset "valid"
                                      :direction :whichever :rounds 2))))
        "an unknown :direction was accepted"))
  (testing "a non-positive :rounds signals"
    (ok (%signals-unsupported-p
         (lambda () (%watcher :spec '(:metric "logloss" :dataset "valid"
                                      :direction :lower-is-better :rounds 0))))
        ":rounds 0 was accepted")))

(deftest watcher-resolves-a-dataset-by-name-and-by-index
  (testing "a name selects the series with that index"
    (let ((watcher (%watcher)))
      (observe-iteration watcher (%entries 0.5d0 :index 1) 1)
      (ok (eql 0.5d0 (watcher-best-score watcher)) "the named series was not the one read")))
  (testing "an integer selects directly, and 0 is the training set"
    (let ((watcher (%watcher :spec '(:metric "logloss" :dataset 0
                                     :direction :lower-is-better :rounds 2))))
      (observe-iteration watcher (%entries 0.5d0) 1)
      (ok (eql 99.0d0 (watcher-best-score watcher)) "index 0 did not read the training set"))))

(deftest watcher-rejects-an-unresolvable-or-ambiguous-dataset
  (testing "a name matching nothing signals"
    (ok (%signals-unsupported-p
         (lambda () (%watcher :spec '(:metric "logloss" :dataset "nope"
                                      :direction :lower-is-better :rounds 2))))
        "an unmatched :dataset name was accepted"))
  (testing "an index out of range signals"
    (ok (%signals-unsupported-p
         (lambda () (%watcher :spec '(:metric "logloss" :dataset 7
                                      :direction :lower-is-better :rounds 2))))
        "an out-of-range :dataset index was accepted"))
  (testing "a name matching two validation sets signals"
    ;; Phase 3a permits duplicate names because the index tells them apart in the report.
    ;; Here the name must select exactly one series, and taking the first would make which
    ;; one silent.
    (ok (%signals-unsupported-p
         (lambda () (%watcher :names '(nil "valid" "valid"))))
        "an ambiguous :dataset name was accepted")))

(deftest watcher-stops-after-the-configured-run-of-no-improvement
  (testing "two non-improving iterations after the best one stop the run"
    (let ((watcher (%watcher)))
      (ok (null (observe-iteration watcher (%entries 0.5d0) 1)) "stopped at the first value")
      (ok (null (observe-iteration watcher (%entries 0.4d0) 2)) "stopped on an improvement")
      (ok (null (observe-iteration watcher (%entries 0.6d0) 3)) "stopped after one bad round")
      (ok (observe-iteration watcher (%entries 0.7d0) 4) "did not stop after two bad rounds")
      (ok (watcher-stopped-p watcher) "stopped-p is false after stopping")
      (ok (eql 2 (watcher-best-iteration watcher)) "the best iteration is wrong")
      (ok (eql 0.4d0 (watcher-best-score watcher)) "the best score is wrong"))))

(deftest watcher-honours-the-direction-it-was-given
  ;; The same values, read the other way, select a different best iteration. This is what
  ;; proves the direction is used rather than accepted and ignored.
  (testing ":higher-is-better picks the largest value"
    (let ((watcher (%watcher :spec '(:metric "logloss" :dataset "valid"
                                     :direction :higher-is-better :rounds 2))))
      (observe-iteration watcher (%entries 0.5d0) 1)
      (observe-iteration watcher (%entries 0.4d0) 2)
      (observe-iteration watcher (%entries 0.6d0) 3)
      (ok (eql 3 (watcher-best-iteration watcher)) "the best iteration is wrong")
      (ok (eql 0.6d0 (watcher-best-score watcher)) "the best score is wrong"))))

(deftest watcher-treats-a-plateau-as-no-improvement
  (testing "an equal value does not reset the counter"
    (let ((watcher (%watcher)))
      (observe-iteration watcher (%entries 0.5d0) 1)
      (observe-iteration watcher (%entries 0.5d0) 2)
      (ok (observe-iteration watcher (%entries 0.5d0) 3)
          "a run that never improves did not stop")
      (ok (eql 1 (watcher-best-iteration watcher))
          "a tie moved the best iteration; improvement is strict"))))

(deftest watcher-treats-a-nil-value-as-no-improvement
  ;; Phase 3a records an unreadable field as NIL rather than dropping it. NIL cannot be
  ;; compared, and calling it an improvement would extend a run on a value nobody can read.
  (testing "a NIL value neither improves nor resets"
    (let ((watcher (%watcher)))
      (observe-iteration watcher (%entries 0.5d0) 1)
      (observe-iteration watcher (%entries nil) 2)
      (ok (observe-iteration watcher (%entries nil) 3) "NIL values did not count as no-improvement")
      (ok (eql 1 (watcher-best-iteration watcher)) "a NIL value became the best score"))))

(deftest watcher-signals-when-the-metric-is-never-reported
  ;; Raised on the first iteration, when the names are known -- not at parse time, which would
  ;; mean predicting what the library will report.
  (testing "a metric the booster does not compute signals"
    (let ((watcher (%watcher :spec '(:metric "nope" :dataset "valid"
                                     :direction :lower-is-better :rounds 2))))
      (ok (%signals-unsupported-p (lambda () (observe-iteration watcher (%entries 0.5d0) 1)))
          "an absent metric was accepted"))))
