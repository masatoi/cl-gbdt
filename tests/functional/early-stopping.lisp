;;;; early-stopping.lisp --- Portable contract tests for `train''s :EARLY-STOPPING.
;;;;
;;;; Policy section 9, Phase 3b: `train' takes an :EARLY-STOPPING spec, watches one
;;;; (dataset, metric) series as the run proceeds, and stops the loop once that series has
;;;; failed to improve for the requested number of consecutive iterations. Like
;;;; tests/functional/training-report.lisp beside it, every test below runs the identical
;;;; assertions over tests/functional/evaluation.lisp's *FIXTURES*, once per backend, so the
;;;; two backends cannot drift apart in shape, order or meaning without one of them failing
;;;; here. Numbers are never compared BETWEEN backends, for the reason that file's header
;;;; gives: the two libraries train different models from the same rows by design.
;;;;
;;;; The watched series is always the validation set at index 1, carrying FIXTURE's own name
;;;; for log loss. Most tests below build that validation set with `INVERT-LABELS', so its
;;;; labels disagree with the training set's over the same feature matrix: a booster fitting
;;;; the training labels then scores steadily WORSE against it, which is a series that never
;;;; improves after its first iteration and so provokes a stop within a handful of rounds
;;;; however large :NUM-ROUNDS is. `EARLY-STOPPING-THAT-NEVER-TRIGGERS-REPORTS-THE-FULL-RUN'
;;;; is the one that does the opposite, on a validation set built from the training labels
;;;; unchanged.
;;;;
;;;; `cl-gbdt:with-booster' appears nowhere below, for the reason training-report.lisp's
;;;; header gives: it binds ONE value and every assertion here is about the second one. Each
;;;; test uses `multiple-value-bind' and frees the booster in its own `unwind-protect'.

(uiop:define-package #:cl-gbdt/tests/functional/early-stopping
  (:use #:cl #:rove)
  ;; Zero symbols: every reference below is package-qualified. Declared so this file's
  ;; dependency on the unified API is explicit rather than inherited, matching the identical
  ;; clause in evaluation.lisp and training-report.lisp.
  (:import-from #:cl-gbdt)
  ;; Zero symbols, both of them: their only job is to run at load time and register
  ;; :lightgbm and :xgboost with `open-backend'. Without these clauses, package-inferred-
  ;; system has no edge to those files and `(cl-gbdt:open-backend :lightgbm)' below would
  ;; signal `unknown-backend'. Declared here rather than leaned on through the
  ;; evaluation.lisp dependency below, which happens to pull both in today.
  (:import-from #:cl-gbdt/src/lightgbm/all)
  (:import-from #:cl-gbdt/src/xgboost/all)
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library
                #:make-separable-dataset)
  ;; The fixture table and its two data builders come from evaluation.lisp rather than being
  ;; restated here, exactly as training-report.lisp imports them: these tests need the same
  ;; two-metric booster over the same eight rows, and a third table saying the same thing in
  ;; its own words is how files that must agree stop agreeing.
  (:import-from #:cl-gbdt/tests/functional/evaluation
                #:*fixtures*
                #:make-fixture-dataset
                #:invert-labels))

(in-package #:cl-gbdt/tests/functional/early-stopping)

(defun watch-spec (fixture &key (dataset "valid") (direction :lower-is-better) (rounds 3))
  "Return an :EARLY-STOPPING spec watching FIXTURE's own log-loss metric.

All four keys `make-early-stopping-watcher' requires are supplied -- it has no defaults for
any of them, deliberately -- and the three a test varies to distinguish itself from its
neighbours are this function's own keywords. The metric is always FIXTURE's :LOSS-METRIC,
the one name each backend spells its own log loss with, since watching a metric the booster
never reports at all is a different test living at layer 1."
  (list :metric (getf fixture :loss-metric) :dataset dataset
        :direction direction :rounds rounds))

(defun watched-values (report fixture)
  "Return REPORT's recorded values for the series every test here watches -- FIXTURE's
log-loss metric on the validation set at index 1 -- or NIL when REPORT has no such series."
  (let ((series (find-if (lambda (series)
                           (and (eql 1 (cl-gbdt:training-series-index series))
                                (string= (getf fixture :loss-metric)
                                         (cl-gbdt:training-series-metric series))))
                         (cl-gbdt:training-report-series report))))
    (and series (cl-gbdt:training-series-values series))))

;;; The assertion an implementation that records a best iteration but never actually leaves
;;; the loop fails, and the only one here that it fails: every other assertion below would
;;; pass against it. :NUM-ROUNDS is 1000 and the watched series worsens from the second
;;; iteration onward, so a run that honours :ROUNDS 3 ends within single digits. The series'
;;; own length is asserted against the reported count as well -- a run that stopped the loop
;;; but reported 1000, or reported the shortened count while recording 1000 values, breaks
;;; the "one value per completed iteration" invariant the report rests on.

(deftest early-stopping-actually-stops-the-run
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (valid-label-vector (invert-labels label-vector)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix valid-label-vector
                                                      :reference train-set))
                   (multiple-value-bind (booster report)
                       (cl-gbdt:train backend train-set :num-rounds 1000
                                      :valid-sets (list (cons "valid" valid-set))
                                      :early-stopping (watch-spec fixture)
                                      :parameters (getf fixture :booster-parameters))
                     (unwind-protect
                          (let ((ran (cl-gbdt:training-report-num-rounds report))
                                (all-series (cl-gbdt:training-report-series report)))
                            (testing (format nil "~A: the run ended well short of its ~
                                                  1000-round limit" (getf fixture :backend))
                              (ok (and (integerp ran) (< 0 ran 100))
                                  (format nil "the run reported ~S of 1000 rounds" ran)))
                            (testing (format nil "~A: the report says the run was stopped ~
                                                  early" (getf fixture :backend))
                              (ok (cl-gbdt:training-report-early-stopped-p report)
                                  (format nil "early-stopped-p was ~S after a ~S-round run"
                                          (cl-gbdt:training-report-early-stopped-p report)
                                          ran)))
                            (testing (format nil "~A: every series is as long as the ~
                                                  shortened run" (getf fixture :backend))
                              (ok (and all-series
                                       (every (lambda (series)
                                                (= ran (length
                                                        (cl-gbdt:training-series-values
                                                         series))))
                                              all-series))
                                  (format nil "num-rounds was ~S, series were ~S"
                                          ran all-series))))
                       (cl-gbdt:free-booster booster)))))
            (cl-gbdt:close-backend backend)))))))

;;; The two slots are tied to the recorded history rather than to each other: BEST-ITERATION
;;; names an iteration the run actually reached, and BEST-SCORE is what the watched series
;;; recorded AT that iteration. An implementation that returned the watcher's own running
;;; state out of step with what it pushed onto the history -- off by one in either direction
;;; is the obvious way -- fails the second assertion. The booster's own BEST-ITERATION is
;;; asserted here too: it is what Task 4's `:num-iteration :best' resolves against, and it
;;; must be the same iteration the report names.

(deftest early-stopping-reports-a-best-iteration-within-the-run
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (valid-label-vector (invert-labels label-vector)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix valid-label-vector
                                                      :reference train-set))
                   (multiple-value-bind (booster report)
                       (cl-gbdt:train backend train-set :num-rounds 1000
                                      :valid-sets (list (cons "valid" valid-set))
                                      :early-stopping (watch-spec fixture)
                                      :parameters (getf fixture :booster-parameters))
                     (unwind-protect
                          (let ((ran (cl-gbdt:training-report-num-rounds report))
                                (best (cl-gbdt:training-report-best-iteration report))
                                (score (cl-gbdt:training-report-best-score report))
                                (recorded (watched-values report fixture)))
                            (testing (format nil "~A: best-iteration is an iteration the ~
                                                  run reached" (getf fixture :backend))
                              (ok (and (integerp best) (<= 1 best ran))
                                  (format nil "best-iteration was ~S over a ~S-round run"
                                          best ran)))
                            (testing (format nil "~A: best-score is the watched series' ~
                                                  value at that iteration"
                                             (getf fixture :backend))
                              (ok (and (integerp best) recorded
                                       (<= 1 best (length recorded))
                                       (eql score (aref recorded (1- best))))
                                  (format nil "best-iteration ~S, best-score ~S, series ~S"
                                          best score recorded)))
                            (testing (format nil "~A: the booster carries the same best ~
                                                  iteration" (getf fixture :backend))
                              (ok (eql best (cl-gbdt:booster-best-iteration booster))
                                  (format nil "the report said ~S, the booster said ~S"
                                          best (cl-gbdt:booster-best-iteration booster)))))
                       (cl-gbdt:free-booster booster)))))
            (cl-gbdt:close-backend backend)))))))

;;; :DIRECTION is used, not accepted and ignored. Both runs here see the identical series --
;;; the same data, the same metric, the same five iterations, with :ROUNDS 10 so neither can
;;; stop inside a five-round run -- and differ only in which end of it counts as best. An
;;; implementation that hardcoded one direction reports the same BEST-ITERATION twice.

(deftest early-stopping-direction-changes-which-iteration-is-best
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (valid-label-vector (invert-labels label-vector))
              (lower nil)
              (higher nil)
              (rounds-run '()))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix valid-label-vector
                                                      :reference train-set))
                   (dolist (direction '(:lower-is-better :higher-is-better))
                     (multiple-value-bind (booster report)
                         (cl-gbdt:train backend train-set :num-rounds 5
                                        :valid-sets (list (cons "valid" valid-set))
                                        :early-stopping (watch-spec fixture
                                                                    :direction direction
                                                                    :rounds 10)
                                        :parameters (getf fixture :booster-parameters))
                       (unwind-protect
                            (progn
                              (push (list direction
                                          (cl-gbdt:training-report-num-rounds report)
                                          (cl-gbdt:training-report-early-stopped-p report))
                                    rounds-run)
                              (if (eq direction :lower-is-better)
                                  (setf lower (cl-gbdt:training-report-best-iteration report))
                                  (setf higher
                                        (cl-gbdt:training-report-best-iteration report))))
                         (cl-gbdt:free-booster booster))))
                   (testing (format nil "~A: :rounds 10 never fires inside a 5-round run"
                                    (getf fixture :backend))
                     (ok (every (lambda (run) (and (eql 5 (second run)) (null (third run))))
                                rounds-run)
                         (format nil "(direction num-rounds early-stopped-p) were ~S"
                                 (reverse rounds-run))))
                   (testing (format nil "~A: the two directions pick different iterations"
                                    (getf fixture :backend))
                     (ok (and (integerp lower) (integerp higher) (/= lower higher))
                         (format nil ":lower-is-better picked ~S, :higher-is-better ~S"
                                 lower higher)))))
            (cl-gbdt:close-backend backend)))))))

;;; The three report slots and the booster's own are filled ONLY when :EARLY-STOPPING was
;;; supplied. Without it they stay NIL, which is what every caller written before this phase
;;; still sees -- and the history is recorded in full regardless, so leaving them NIL is not
;;; a matter of the run having been unobserved.

(deftest training-without-early-stopping-leaves-the-three-slots-nil
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (valid-label-vector (invert-labels label-vector)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix valid-label-vector
                                                      :reference train-set))
                   (multiple-value-bind (booster report)
                       (cl-gbdt:train backend train-set :num-rounds 5
                                      :valid-sets (list (cons "valid" valid-set))
                                      :parameters (getf fixture :booster-parameters))
                     (unwind-protect
                          (let ((all-series (cl-gbdt:training-report-series report)))
                            (testing (format nil "~A: the report's three early-stopping ~
                                                  slots are NIL" (getf fixture :backend))
                              (ok (and (null (cl-gbdt:training-report-best-iteration report))
                                       (null (cl-gbdt:training-report-best-score report))
                                       (null (cl-gbdt:training-report-early-stopped-p
                                              report)))
                                  (format nil "best-iteration ~S, best-score ~S, ~
                                               early-stopped-p ~S"
                                          (cl-gbdt:training-report-best-iteration report)
                                          (cl-gbdt:training-report-best-score report)
                                          (cl-gbdt:training-report-early-stopped-p report))))
                            (testing (format nil "~A: the booster's best iteration is NIL too"
                                             (getf fixture :backend))
                              (ok (null (cl-gbdt:booster-best-iteration booster))
                                  (format nil "the booster said ~S"
                                          (cl-gbdt:booster-best-iteration booster))))
                            (testing (format nil "~A: the history is still recorded in full"
                                             (getf fixture :backend))
                              (ok (and all-series
                                       (eql 5 (cl-gbdt:training-report-num-rounds report))
                                       (every (lambda (series)
                                                (= 5 (length
                                                      (cl-gbdt:training-series-values
                                                       series))))
                                              all-series))
                                  (format nil "num-rounds was ~S, series were ~S"
                                          (cl-gbdt:training-report-num-rounds report)
                                          all-series))))
                       (cl-gbdt:free-booster booster)))))
            (cl-gbdt:close-backend backend)))))))

;;; "Asked for and not triggered" is a different state from "not asked for", and the test
;;; above cannot tell them apart on its own: both leave EARLY-STOPPED-P NIL. Here the
;;; validation set is built from the training labels unchanged, so the watched series keeps
;;; improving, and :ROUNDS 10 could not fire inside a five-round run even if it did not --
;;; the run reaches its full :NUM-ROUNDS. BEST-ITERATION is filled all the same, which is
;;; what distinguishes this from a run that never asked.

(deftest early-stopping-that-never-triggers-reports-the-full-run
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 ;; The training labels, not `INVERT-LABELS': this is the case where the
                 ;; watched metric keeps getting better rather than worse.
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix label-vector
                                                      :reference train-set))
                   (multiple-value-bind (booster report)
                       (cl-gbdt:train backend train-set :num-rounds 5
                                      :valid-sets (list (cons "valid" valid-set))
                                      :early-stopping (watch-spec fixture :rounds 10)
                                      :parameters (getf fixture :booster-parameters))
                     (unwind-protect
                          (let ((best (cl-gbdt:training-report-best-iteration report)))
                            (testing (format nil "~A: nothing stopped the run"
                                             (getf fixture :backend))
                              (ok (null (cl-gbdt:training-report-early-stopped-p report))
                                  (format nil "early-stopped-p was ~S"
                                          (cl-gbdt:training-report-early-stopped-p report))))
                            (testing (format nil "~A: the run reached its full 5 rounds"
                                             (getf fixture :backend))
                              (ok (eql 5 (cl-gbdt:training-report-num-rounds report))
                                  (format nil "num-rounds was ~S"
                                          (cl-gbdt:training-report-num-rounds report))))
                            (testing (format nil "~A: a best iteration was determined anyway"
                                             (getf fixture :backend))
                              (ok (and (integerp best) (<= 1 best 5))
                                  (format nil "best-iteration was ~S" best))))
                       (cl-gbdt:free-booster booster)))))
            (cl-gbdt:close-backend backend)))))))

;;; The contradiction: early stopping needs the watched series, and reading the evaluation
;;; costs the same whether one series is watched or all are recorded, so there is no cheaper
;;; middle path to offer a caller who asked for both. `handler-case', not rove's `signals',
;;; which does not reliably catch a condition raised inside `restart-case'; the condition
;;; TYPE is asserted, not merely that something signalled.

(deftest early-stopping-with-record-history-nil-signals
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (valid-label-vector (invert-labels label-vector)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix valid-label-vector
                                                      :reference train-set))
                   (testing (format nil "~A: :early-stopping with :record-history NIL ~
                                         signals unsupported-argument"
                                    (getf fixture :backend))
                     (ok (handler-case
                             (multiple-value-bind (booster report)
                                 (cl-gbdt:train backend train-set :num-rounds 5
                                                :valid-sets (list (cons "valid" valid-set))
                                                :record-history nil
                                                :early-stopping (watch-spec fixture)
                                                :parameters (getf fixture
                                                                  :booster-parameters))
                               (declare (ignore report))
                               (cl-gbdt:free-booster booster)
                               nil)
                           (cl-gbdt:unsupported-argument () t))
                         (format nil "train accepted :early-stopping together with ~
                                      :record-history NIL")))))
            (cl-gbdt:close-backend backend)))))))

;;; Phase 3a deliberately lets two :VALID-SETS entries share one name -- their index tells
;;; them apart in the report -- but a watcher has to pick exactly one series, and silently
;;; taking the first match would make the choice invisible. So the ambiguous name is
;;; rejected, and the index is the way to say which one was meant. Both halves are asserted:
;;; a rejection nobody can work around would be a worse answer than accepting the first.

(deftest early-stopping-with-an-ambiguous-dataset-name-signals
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (valid-label-vector (invert-labels label-vector)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-1 (make-fixture-dataset fixture backend matrix valid-label-vector
                                                    :reference train-set))
                   (cl-gbdt:with-dataset
                       (valid-2 (make-fixture-dataset fixture backend matrix label-vector
                                                      :reference train-set))
                     (let ((valid-sets (list (cons "valid" valid-1) (cons "valid" valid-2))))
                       (testing (format nil "~A: a name matching two validation sets signals ~
                                             unsupported-argument" (getf fixture :backend))
                         (ok (handler-case
                                 (multiple-value-bind (booster report)
                                     (cl-gbdt:train backend train-set :num-rounds 3
                                                    :valid-sets valid-sets
                                                    :early-stopping (watch-spec fixture)
                                                    :parameters (getf fixture
                                                                      :booster-parameters))
                                   (declare (ignore report))
                                   (cl-gbdt:free-booster booster)
                                   nil)
                               (cl-gbdt:unsupported-argument () t))
                             (format nil "train accepted an :early-stopping :dataset name ~
                                          matching two validation sets")))
                       (multiple-value-bind (booster report)
                           (cl-gbdt:train backend train-set :num-rounds 3
                                          :valid-sets valid-sets
                                          :early-stopping (watch-spec fixture :dataset 1
                                                                              :rounds 10)
                                          :parameters (getf fixture :booster-parameters))
                         (unwind-protect
                              (testing (format nil "~A: the index picks one of them and the ~
                                                    run succeeds" (getf fixture :backend))
                                (ok (and (eql 3 (cl-gbdt:training-report-num-rounds report))
                                         (integerp (cl-gbdt:training-report-best-iteration
                                                    report)))
                                    (format nil "num-rounds ~S, best-iteration ~S"
                                            (cl-gbdt:training-report-num-rounds report)
                                            (cl-gbdt:training-report-best-iteration
                                             report))))
                           (cl-gbdt:free-booster booster)))))))
            (cl-gbdt:close-backend backend)))))))

;;; ---------------------------------------------------------------------------
;;; `:num-iteration :best'
;;;
;;; Task 4: the four tests below exercise the booster's own `booster-best-iteration' as
;;; `predict', `save-model' and `model-to-string''s `:num-iteration :best' resolves it --
;;; see `src/protocol.lisp''s `predict' docstring for the contract each asserts against.
;;; `feature-importance' also accepts `:num-iteration' but is out of this phase's scope --
;;; the brief names only these three -- so it is left untested at this, functional, layer.
;;; `%reject-best-num-iteration', the helper both backends' `feature-importance' call to
;;; refuse `:num-iteration :best', does have layer-1 coverage -- see
;;; `cl-gbdt/tests/handle''s `reject-best-num-iteration-signals-for-best' and
;;; `reject-best-num-iteration-passes-everything-else-through'.

;;; The watched series under `INVERT-LABELS' worsens from its second iteration on (see this
;;; file's header), so `watch-spec''s default -- `:rounds 3' -- reliably picks a
;;; BEST-ITERATION well short of the run's own length, which `EARLY-STOPPING-ACTUALLY-STOPS-
;;; THE-RUN' above already confirms lands under 100 of 1000. Predicting with that many fewer
;;; trees than the run actually grew must produce different numbers -- not a mechanical
;;; certainty for every possible model, which is why the "best-iteration is short of the run"
;;; assertion runs first and is itself checked, but true for this fixture, confirmed by
;;; actually running it rather than assumed. An implementation that resolved :BEST to the
;;; same answer as NIL, or that never resolved it before `%resolve-num-iteration' at all,
;;; passes every earlier test in this file and fails only here.

(deftest best-num-iteration-differs-from-every-round
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (valid-label-vector (invert-labels label-vector)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix valid-label-vector
                                                      :reference train-set))
                   (multiple-value-bind (booster report)
                       (cl-gbdt:train backend train-set :num-rounds 1000
                                      :valid-sets (list (cons "valid" valid-set))
                                      :early-stopping (watch-spec fixture)
                                      :parameters (getf fixture :booster-parameters))
                     (unwind-protect
                          (let ((ran (cl-gbdt:training-report-num-rounds report))
                                (best (cl-gbdt:training-report-best-iteration report)))
                            (testing (format nil "~A: the best iteration is short of the run"
                                             (getf fixture :backend))
                              (ok (and (integerp best) (< best ran))
                                  (format nil "best-iteration ~S, num-rounds ~S" best ran)))
                            (testing (format nil "~A: predict :num-iteration :best differs ~
                                                  from predict over every round"
                                             (getf fixture :backend))
                              (ok (not (equalp (cl-gbdt:predict booster matrix)
                                                (cl-gbdt:predict booster matrix
                                                                 :num-iteration :best)))
                                  (format nil "predict :num-iteration :best differs from ~
                                               predict over every round"))))
                       (cl-gbdt:free-booster booster)))))
            (cl-gbdt:close-backend backend)))))))

;;; A booster nobody asked to early-stop has no best iteration for `:best' to resolve
;;; against at all -- `booster-best-iteration' is NIL, and NIL means "no answer for this
;;; booster", not "use every round", which is what plain NIL already means and stays meaning.

(deftest best-num-iteration-without-early-stopping-signals
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (multiple-value-bind (booster report)
                     (cl-gbdt:train backend train-set :num-rounds 5
                                    :parameters (getf fixture :booster-parameters))
                   (declare (ignore report))
                   (unwind-protect
                        (progn
                          (testing (format nil "~A: the booster has no best iteration"
                                           (getf fixture :backend))
                            (ok (null (cl-gbdt:booster-best-iteration booster))
                                (format nil "booster-best-iteration was ~S"
                                        (cl-gbdt:booster-best-iteration booster))))
                          (testing (format nil "~A: predict :num-iteration :best signals ~
                                                unsupported-argument" (getf fixture :backend))
                            (ok (handler-case
                                    (progn (cl-gbdt:predict booster matrix :num-iteration :best)
                                           nil)
                                  (cl-gbdt:unsupported-argument () t))
                                (format nil "predict :num-iteration :best signals ~
                                             unsupported-argument on a booster with no ~
                                             best iteration"))))
                     (cl-gbdt:free-booster booster))))
            (cl-gbdt:close-backend backend)))))))

;;; The other way `booster-best-iteration' can be NIL, and the one the test above cannot
;;; tell apart from this one: a booster whose `train' call DID pass :early-stopping, but
;;; whose watcher never got the chance to see an iteration at all. :num-rounds 0 is the
;;; cheap, deterministic way to provoke it -- the loop `train' drives never runs, so
;;; `observe-iteration' is never called -- documented in `src/protocol.lisp''s `train'
;;; docstring and README.markdown's :num-iteration :best section as one of the two cases
;;; where supplying :early-stopping does not by itself guarantee a best iteration. No
;;; :valid-sets entry is needed: :dataset 0 -- the training set -- is always a valid
;;; watcher target, and with no iteration ever observed, no entry is ever read from it.

(deftest best-num-iteration-after-zero-rounds-with-early-stopping-signals
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (multiple-value-bind (booster report)
                     (cl-gbdt:train backend train-set :num-rounds 0
                                    :early-stopping (watch-spec fixture :dataset 0)
                                    :parameters (getf fixture :booster-parameters))
                   (unwind-protect
                        (progn
                          (testing (format nil "~A: a zero-round run given ~
                                                :early-stopping still has no best ~
                                                iteration" (getf fixture :backend))
                            (ok (and (eql 0 (cl-gbdt:training-report-num-rounds report))
                                     (null (cl-gbdt:training-report-best-iteration report))
                                     (null (cl-gbdt:booster-best-iteration booster)))
                                (format nil "num-rounds ~S, report best-iteration ~S, ~
                                             booster best-iteration ~S"
                                        (cl-gbdt:training-report-num-rounds report)
                                        (cl-gbdt:training-report-best-iteration report)
                                        (cl-gbdt:booster-best-iteration booster))))
                          (testing (format nil "~A: predict :num-iteration :best signals ~
                                                unsupported-argument after a zero-round ~
                                                early-stopped run"
                                           (getf fixture :backend))
                            (ok (handler-case
                                    (progn (cl-gbdt:predict booster matrix
                                                             :num-iteration :best)
                                           nil)
                                  (cl-gbdt:unsupported-argument () t))
                                (format nil "predict :num-iteration :best signals ~
                                             unsupported-argument on a zero-round ~
                                             early-stopped booster"))))
                     (cl-gbdt:free-booster booster))))
            (cl-gbdt:close-backend backend)))))))

;;; The asymmetry itself, asserted rather than smoothed over. LightGBM honours
;;; `:num-iteration' -- `:best' included -- so `save-model' writes a file that stops at the
;;; best iteration. XGBoost's `XGBoosterSaveModel' has no iteration limit at all, so `:best'
;;; resolves to an integer and then meets the very `unsupported-argument' check an explicit
;;; `:num-iteration 50' already does -- there is no special case for `:best' in that check,
;;; and none is added here. One test, branching on the fixture's own backend: the branch is
;;; the point, not an accident of how the two halves happen to be organised.

(deftest save-model-with-best-works-on-lightgbm-and-signals-on-xgboost
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (valid-label-vector (invert-labels label-vector)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix valid-label-vector
                                                      :reference train-set))
                   (uiop:with-temporary-file (:pathname path
                                              :type (getf fixture :model-file-type))
                     (multiple-value-bind (booster report)
                         (cl-gbdt:train backend train-set :num-rounds 1000
                                        :valid-sets (list (cons "valid" valid-set))
                                        :early-stopping (watch-spec fixture)
                                        :parameters (getf fixture :booster-parameters))
                       (declare (ignore report))
                       (unwind-protect
                            (ecase (getf fixture :backend)
                              (:lightgbm
                               (testing (format nil "LightGBM: save-model :num-iteration ~
                                                     :best writes a file")
                                 (ok (progn
                                       (cl-gbdt:save-model booster path :num-iteration :best)
                                       (probe-file path))
                                     (format nil "save-model :num-iteration :best writes ~
                                                  a file"))))
                              (:xgboost
                               (testing (format nil "XGBoost: save-model :num-iteration ~
                                                     :best signals unsupported-argument, ~
                                                     the same asymmetry an explicit integer ~
                                                     already does")
                                 (ok (handler-case
                                         (progn (cl-gbdt:save-model booster path
                                                                    :num-iteration :best)
                                                nil)
                                       (cl-gbdt:unsupported-argument () t))
                                     (format nil "save-model :num-iteration :best signals ~
                                                  unsupported-argument on XGBoost")))))
                         (cl-gbdt:free-booster booster))))))
            (cl-gbdt:close-backend backend)))))))

;;; `model-to-string' behaves like `save-model' for `:num-iteration': LightGBM honours it,
;;; XGBoost's `XGBoosterSaveModelToBuffer' has no iteration-limited variant and already
;;; signals `unsupported-argument' for any non-NIL value -- that half of the asymmetry is
;;; SAVE-MODEL-WITH-BEST-WORKS-ON-LIGHTGBM-AND-SIGNALS-ON-XGBOOST above, so this test
;;; exercises only the backend where `:best' actually changes what comes back, rather than
;;; repeating the same signal assertion under a different method name.

(deftest model-to-string-with-best
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (if (not (eq (getf fixture :backend) :lightgbm))
          (skip (format nil "~A: model-to-string does not accept :num-iteration at all -- ~
                             see save-model-with-best-works-on-lightgbm-and-signals-on-xgboost"
                        (getf fixture :backend)))
          (multiple-value-bind (matrix label-vector) (make-separable-dataset)
            (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
                  (valid-label-vector (invert-labels label-vector)))
              (unwind-protect
                   (cl-gbdt:with-dataset
                       (train-set (make-fixture-dataset fixture backend matrix label-vector))
                     (cl-gbdt:with-dataset
                         (valid-set (make-fixture-dataset fixture backend matrix
                                                          valid-label-vector
                                                          :reference train-set))
                       (multiple-value-bind (booster report)
                           (cl-gbdt:train backend train-set :num-rounds 1000
                                          :valid-sets (list (cons "valid" valid-set))
                                          :early-stopping (watch-spec fixture)
                                          :parameters (getf fixture :booster-parameters))
                         (declare (ignore report))
                         (unwind-protect
                              (testing (format nil "LightGBM: model-to-string ~
                                                    :num-iteration :best differs from ~
                                                    every round")
                                (ok (not (string= (cl-gbdt:model-to-string booster)
                                                   (cl-gbdt:model-to-string
                                                    booster :num-iteration :best)))
                                    (format nil "model-to-string :num-iteration :best ~
                                                 differs from model-to-string over every ~
                                                 round")))
                           (cl-gbdt:free-booster booster)))))
                (cl-gbdt:close-backend backend))))))))
