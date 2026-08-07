;;;; training-report.lisp --- Portable contract tests for `train''s training report.
;;;;
;;;; Policy section 9, Phase 3a: `train' records what every configured metric did on every
;;;; dataset at every iteration and hands it back as its SECONDARY value. Like
;;;; tests/functional/evaluation.lisp beside it, every test below runs the identical
;;;; assertions over that file's *FIXTURES*, once per backend, so the two backends cannot
;;;; drift apart in shape, order or meaning without one of them failing here. Numbers are
;;;; never compared BETWEEN backends, for the reason that file's header gives: the two
;;;; libraries train different models from the same rows by design.
;;;;
;;;; `cl-gbdt:with-booster' appears nowhere below, and its absence is the point. It binds
;;;; ONE value, so a caller who wraps `train' in it gets the booster and never sees the
;;;; report -- which is exactly the backward compatibility this phase promises, and why
;;;; every pre-existing test in this suite still passes untouched. The tests here want the
;;;; second value, so each uses `multiple-value-bind' and frees the booster in its own
;;;; `unwind-protect' instead.
;;;;
;;;; Every series' name is NIL throughout: :VALID-SETS holds bare datasets in this phase,
;;;; and nothing invents a name for one. Naming arrives in the next task.

(uiop:define-package #:cl-gbdt/tests/functional/training-report
  (:use #:cl #:rove)
  ;; Zero symbols: every reference below is package-qualified. Declared so this file's
  ;; dependency on the unified API is explicit rather than inherited, matching the identical
  ;; clause in evaluation.lisp.
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
  ;; The fixture table and its two data builders come from evaluation.lisp rather than
  ;; being restated here: these tests need the same two-metric booster over the same eight
  ;; rows that file trains, and a second table saying the same thing in its own words is
  ;; how two files that must agree stop agreeing.
  (:import-from #:cl-gbdt/tests/functional/evaluation
                #:*fixtures*
                #:make-fixture-dataset
                #:invert-labels))

(in-package #:cl-gbdt/tests/functional/training-report)

(defun find-series (report index metric-name)
  "Return REPORT's series for METRIC-NAME on the dataset at INDEX, or NIL when there is
none."
  (find-if (lambda (series)
             (and (eql index (cl-gbdt:training-series-index series))
                  (string= metric-name (cl-gbdt:training-series-metric series))))
           (cl-gbdt:training-report-series report)))

(defun series-keys (report)
  "Return REPORT's series as a fresh list of (INDEX METRIC-NAME) lists, in series order."
  (mapcar (lambda (series)
            (list (cl-gbdt:training-series-index series)
                  (cl-gbdt:training-series-metric series)))
          (cl-gbdt:training-report-series report)))

(defun entry-keys (entries)
  "Return ENTRIES -- a `cl-gbdt:evaluation' result -- as a fresh list of (INDEX
METRIC-NAME) lists, in entry order, for comparison with `series-keys'."
  (mapcar (lambda (entry) (list (first entry) (second entry))) entries))

(defun last-value (series)
  "Return SERIES' final recorded value, or :NO-VALUES when it recorded none.

:NO-VALUES rather than NIL, which is a legal recorded value in its own right -- see
`cl-gbdt:training-series-values' -- and would make an empty series indistinguishable from
one whose last field the backend could not read."
  (let ((recorded (cl-gbdt:training-series-values series)))
    (if (plusp (length recorded))
        (aref recorded (1- (length recorded)))
        :no-values)))

;;; Two metrics are configured and one validation set is passed, so: 2 datasets x 2
;;; metrics. A recorder that collapsed the datasets into one, or dropped every metric after
;;; the first, fails here.

(deftest training-report-has-one-series-per-metric-per-dataset
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
                   ;; NOT `with-booster': it binds one value and the report is the second.
                   (multiple-value-bind (booster report)
                       (cl-gbdt:train backend train-set :num-rounds 5
                                      :valid-sets (list valid-set)
                                      :parameters (getf fixture :booster-parameters))
                     (unwind-protect
                          (testing (format nil "~A: one series per metric per dataset"
                                           (getf fixture :backend))
                            (ok (= 4 (length (cl-gbdt:training-report-series report)))
                                (format nil "series were ~S"
                                        (cl-gbdt:training-report-series report))))
                       (cl-gbdt:free-booster booster)))))
            (cl-gbdt:close-backend backend)))))))

;;; One value per completed iteration, per series -- the invariant every other assertion
;;; here rests on. A recorder that skipped an iteration, or that recorded per iteration
;;; rather than per series and so produced one long vector, fails here.

(deftest training-report-series-are-as-long-as-the-run
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
                                      :valid-sets (list valid-set)
                                      :parameters (getf fixture :booster-parameters))
                     (unwind-protect
                          (let ((all-series (cl-gbdt:training-report-series report)))
                            (testing (format nil "~A: the report says how many iterations ran"
                                             (getf fixture :backend))
                              (ok (= 5 (cl-gbdt:training-report-num-rounds report))
                                  (format nil "num-rounds was ~S"
                                          (cl-gbdt:training-report-num-rounds report))))
                            (testing (format nil "~A: every series has one value per iteration"
                                             (getf fixture :backend))
                              (ok (and all-series
                                       (every (lambda (series)
                                                (= 5 (length
                                                      (cl-gbdt:training-series-values series))))
                                              all-series))
                                  (format nil "series were ~S" all-series))))
                       (cl-gbdt:free-booster booster)))))
            (cl-gbdt:close-backend backend)))))))

;;; The assertion that ties the recorded history to something independently verifiable:
;;; `cl-gbdt:evaluation' reads the trained booster directly, after training, while the
;;; history was recorded during the loop. A recorder returning plausible constants, or
;;; reading the wrong dataset, fails here. The key order is asserted too -- the series list
;;; must come back in the same order `evaluation' reports its entries in, which is what a
;;; recorder keyed on first-seen order gives and a recorder that sorted or hashed its way
;;; to an arbitrary order would not.

(deftest training-report-last-iteration-agrees-with-evaluation
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
                                      :valid-sets (list valid-set)
                                      :parameters (getf fixture :booster-parameters))
                     (unwind-protect
                          (let ((entries (cl-gbdt:evaluation booster)))
                            (testing (format nil "~A: the series are in evaluation's own order"
                                             (getf fixture :backend))
                              (ok (and entries
                                       (equal (entry-keys entries) (series-keys report)))
                                  (format nil "evaluation keys ~S, series keys ~S"
                                          (entry-keys entries) (series-keys report))))
                            (testing (format nil "~A: each series' last value is what ~
                                                  evaluation reports now"
                                             (getf fixture :backend))
                              (ok (and entries
                                       (every (lambda (entry)
                                                (let ((series (find-series report (first entry)
                                                                           (second entry))))
                                                  (and series
                                                       (eql (third entry)
                                                            (last-value series)))))
                                              entries))
                                  (format nil "evaluation was ~S, series were ~S"
                                          entries (cl-gbdt:training-report-series report)))))
                       (cl-gbdt:free-booster booster)))))
            (cl-gbdt:close-backend backend)))))))

;;; A recorder that captured one iteration and repeated it passes the agreement test above
;;; whenever the model has stopped improving; this is what catches it. The training set's
;;; own log loss, which falls as the model fits, so the first and last values of a
;;; five-round run cannot legitimately be equal.

(deftest training-report-values-change-across-iterations
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (valid-label-vector (invert-labels label-vector))
              (metric (getf fixture :loss-metric)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix valid-label-vector
                                                      :reference train-set))
                   (multiple-value-bind (booster report)
                       (cl-gbdt:train backend train-set :num-rounds 5
                                      :valid-sets (list valid-set)
                                      :parameters (getf fixture :booster-parameters))
                     (unwind-protect
                          (let* ((series (find-series report 0 metric))
                                 (recorded (and series
                                                (cl-gbdt:training-series-values series))))
                            (testing (format nil "~A: ~A on the training set moves between ~
                                                  the first iteration and the last"
                                             (getf fixture :backend) metric)
                              (ok (and recorded
                                       (= 5 (length recorded))
                                       (numberp (aref recorded 0))
                                       (numberp (aref recorded 4))
                                       (/= (aref recorded 0) (aref recorded 4)))
                                  (format nil "values were ~S" recorded))))
                       (cl-gbdt:free-booster booster)))))
            (cl-gbdt:close-backend backend)))))))

;;; Policy section 9's naming rule for this phase: a :VALID-SETS entry passed bare has no
;;; name a caller could have supplied, and the training set is never a :VALID-SETS entry at
;;; all, so every series' name is NIL. Nothing invents one -- not the dataset's index
;;; spelled as a string, not "train"/"valid", not the name each backend happens to hand its
;;; own C evaluation call.

(deftest training-report-names-are-nil-without-naming
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
                                      :valid-sets (list valid-set)
                                      :parameters (getf fixture :booster-parameters))
                     (unwind-protect
                          (let ((all-series (cl-gbdt:training-report-series report)))
                            (testing (format nil "~A: every series' name is NIL"
                                             (getf fixture :backend))
                              (ok (and all-series
                                       (every (lambda (series)
                                                (null (cl-gbdt:training-series-name series)))
                                              all-series))
                                  (format nil "names were ~S"
                                          (mapcar #'cl-gbdt:training-series-name
                                                  all-series)))))
                       (cl-gbdt:free-booster booster)))))
            (cl-gbdt:close-backend backend)))))))

;;; A booster with no metric at all still trained, so the report still says how many
;;; iterations ran -- it just has nothing to say about them. Both backends can be told to
;;; report nothing, each in its own words (see *FIXTURES*' :NO-METRIC-PARAMETERS), so this
;;; runs on both rather than on LightGBM alone. An implementation that invented a series
;;; from the objective, or that reported NUM-ROUNDS as 0 because no series was recorded,
;;; fails here.

(deftest training-report-is-empty-without-metrics
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (multiple-value-bind (booster report)
                     (cl-gbdt:train backend train-set :num-rounds 4
                                    :parameters (getf fixture :no-metric-parameters))
                   (unwind-protect
                        (progn
                          (testing (format nil "~A: no metric configured, no series"
                                           (getf fixture :backend))
                            (ok (null (cl-gbdt:training-report-series report))
                                (format nil "series were ~S"
                                        (cl-gbdt:training-report-series report))))
                          (testing (format nil "~A: the run is still reported"
                                           (getf fixture :backend))
                            (ok (= 4 (cl-gbdt:training-report-num-rounds report))
                                (format nil "num-rounds was ~S"
                                        (cl-gbdt:training-report-num-rounds report)))))
                     (cl-gbdt:free-booster booster))))
            (cl-gbdt:close-backend backend)))))))
