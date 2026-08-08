;;;; training-early-stopping.lisp --- Tests for the early-stopping watcher.
;;;;
;;;; The stop decision is pure: it sees one iteration's entries at a time and answers whether
;;;; to stop. No shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/training-early-stopping
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/training/early-stopping
                #:make-early-stopping-watcher
                #:train-early-stopping-watcher
                #:observe-iteration
                #:watcher-best-iteration
                #:watcher-best-score
                #:watcher-stopped-p)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/training-early-stopping)

(defparameter *names* '(nil "valid" "test")
  "The DATASET-NAMES shape `train' builds: position 0 is the training set, which has no name.")

(defun %watcher (&key (backend :test-backend)
                      (spec '(:metric "logloss" :dataset "valid"
                              :direction :lower-is-better :rounds 2))
                      (names *names*))
  (make-early-stopping-watcher backend spec names))

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

;;; `train-early-stopping-watcher' is the entry point both backends' `train' calls, and the
;;; one rule it holds that `make-early-stopping-watcher' does not -- the contradiction with
;;; :RECORD-HISTORY NIL -- belongs to `train''s argument list rather than to a spec on its
;;; own. It was duplicated in both backend files before it lived here, where the identical
;;; error message was two files' problem to keep in sync; these assertions are what a layer-1
;;; suite can say about it that the functional suite's condition-TYPE assertions cannot.

(deftest train-watcher-passes-nil-through
  ;; The overwhelmingly common case: no :EARLY-STOPPING, so no watcher and no error, whatever
  ;; :RECORD-HISTORY says. A guard that checked RECORD-HISTORY first would reject the second
  ;; of these, which is every pre-existing `(train … :record-history nil)' caller.
  (testing "no :early-stopping means no watcher, either way round"
    (ok (null (train-early-stopping-watcher :test-backend nil t *names*))
        "a NIL spec produced a watcher with recording on")
    (ok (null (train-early-stopping-watcher :test-backend nil nil *names*))
        "a NIL spec produced a watcher with recording off")))

(deftest train-watcher-rejects-early-stopping-without-recording
  (testing ":early-stopping with :record-history NIL signals"
    (ok (%signals-unsupported-p
         (lambda () (train-early-stopping-watcher
                     :test-backend
                     '(:metric "logloss" :dataset "valid"
                       :direction :lower-is-better :rounds 2)
                     nil *names*)))
        "the contradictory combination was accepted"))
  ;; The condition's :BACKEND slot names the backend BACKEND-NAME actually was -- the
  ;; caller-supplied keyword, threaded through unexamined -- for this check exactly as it
  ;; does for the four spec validators, never a placeholder like :EARLY-STOPPING that names
  ;; no backend `open-backend' ever returns.
  (testing "the condition names the backend it was actually called with, not a placeholder"
    (flet ((backend-of (thunk)
             (handler-case (progn (funcall thunk) nil)
               (cl-gbdt:unsupported-argument (condition)
                 (cl-gbdt:unsupported-argument-backend condition)))))
      (ok (eq :test-backend
              (backend-of (lambda () (train-early-stopping-watcher
                                      :test-backend
                                      '(:metric "logloss" :dataset "valid"
                                        :direction :lower-is-better :rounds 2)
                                      nil *names*))))
          "train-early-stopping-watcher did not report the backend it was called with")
      (ok (eq :test-backend
              (backend-of (lambda () (%watcher :backend :test-backend
                                               :spec '(:metric "logloss" :dataset "valid"
                                                       :direction :lower-is-better)))))
          "a spec validator did not report the backend it was called with"))))

(deftest train-watcher-builds-a-watcher-from-a-valid-spec
  (testing "a valid spec with recording on yields a usable watcher"
    (let ((watcher (train-early-stopping-watcher
                     :test-backend
                     '(:metric "logloss" :dataset "valid"
                       :direction :lower-is-better :rounds 2)
                     t *names*)))
      (ok (and watcher (null (watcher-stopped-p watcher)))
          "no watcher, or one that had already stopped before seeing an iteration")
      (observe-iteration watcher (%entries 0.5d0) 1)
      (ok (eql 1 (watcher-best-iteration watcher))
          "the watcher did not advance on its first iteration")))
  ;; A malformed spec still reaches `make-early-stopping-watcher''s own validation rather
  ;; than being waved through by the wrapper.
  (testing "a malformed spec still signals through the wrapper"
    (ok (%signals-unsupported-p
         (lambda () (train-early-stopping-watcher
                     :test-backend '(:metric "logloss" :dataset "valid") t *names*)))
        "a spec missing :direction and :rounds was accepted")))

;;; Reported by Codex on PR #16. LightGBM returns a NaN as an ordinary `double-float', which
;;; `realp' answers T for, so a non-NIL guard admits it. Every comparison against a NaN is
;;; false in both directions, so once one becomes the best score no later value can beat it:
;;; the run stops on patience carrying a NaN best-score and a wrong best-iteration. XGBoost's
;;; text parse already yields NIL for its own `nan', so admitting LightGBM's would make one
;;; model state behave differently on the two backends.

(defparameter *nan* (sb-kernel:make-double-float -524288 0)
  "A quiet NaN, built without reading one out of a library. `(= *nan* *nan*)' is false and
does not trap, which is what `%comparable-p' relies on.")

(deftest watcher-treats-a-nan-as-no-improvement
  (testing "a NaN never becomes the best score, even as the first value seen"
    (let ((watcher (%watcher)))
      (observe-iteration watcher (%entries *nan*) 1)
      (ok (null (cl-gbdt/src/training/early-stopping::watcher-best-score watcher))
          "the best score after one NaN")
      (ok (null (watcher-best-iteration watcher))
          "the best iteration after one NaN")))
  (testing "a finite value after a NaN still becomes the best"
    (let ((watcher (%watcher)))
      (observe-iteration watcher (%entries *nan*) 1)
      (observe-iteration watcher (%entries 0.5d0) 2)
      (ok (eql 2 (watcher-best-iteration watcher))
          "the iteration a finite value claimed after a NaN")))
  (testing "a run of NaNs stops on patience, as a run of unreadable values does"
    (let ((watcher (%watcher)))
      (observe-iteration watcher (%entries 0.5d0) 1)
      (observe-iteration watcher (%entries *nan*) 2)
      (ok (observe-iteration watcher (%entries *nan*) 3)
          "whether two NaNs exhaust a :rounds 2 patience"))))

(deftest watcher-metric-error-names-train-and-the-real-backend
  ;; The sixth of the six error sites. It reports at run time rather than at construction
  ;; because :metric is the one required key that cannot be validated before the booster has
  ;; reported its metric names -- which makes it the site most likely to fire in real use.
  (testing "the condition names train's argument and the backend train was called on"
    (let* ((watcher (%watcher :backend :test-backend))
           (text (handler-case
                     (progn (observe-iteration
                             watcher (list (list 1 "other-metric" 0.5d0)) 1)
                            "")
                   (cl-gbdt:unsupported-argument (c) (princ-to-string c)))))
      (ok (search "train's :early-stopping :metric" text)
          "the argument the condition names")
      (ok (search "TEST-BACKEND" text)
          "the backend the condition names")
      (ok (not (search "make-early-stopping-watcher" text))
          "whether the internal constructor's name is absent"))))
