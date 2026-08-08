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
;;;; This file's first block of tests predates naming: :VALID-SETS holds only bare
;;;; datasets there, so every series' name is NIL, and nothing invents one. The block below
;;;; it -- naming -- is this task's own: `train''s :VALID-SETS now also accepts a
;;;; (NAME . DATASET) cons per element, mixed freely with bare datasets, and NAME reaches
;;;; `training-series-name' for every series at that dataset's index.

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

;;; ---------------------------------------------------------------------------
;;; Naming
;;;
;;; `train''s :VALID-SETS now accepts a (NAME . DATASET) cons per element, alongside the
;;; bare dataset the tests above already cover, mixed freely in one list. NAME reaches
;;; `training-series-name' for every series recorded at that dataset's index, and nowhere
;;; else -- the training set is never a :VALID-SETS entry, so its series stay NIL
;;; regardless of what any validation set is named.

(defun fixture-for (backend-name)
  "Return the *FIXTURES* entry for BACKEND-NAME."
  (find backend-name *fixtures* :key (lambda (fixture) (getf fixture :backend))))

;;; A single named validation set: its series carry index 1 and the name given, and the
;;; training set's series at index 0 stay unnamed regardless.

(deftest training-report-named-valid-set-carries-its-name-and-index
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
                       (cl-gbdt:train backend train-set :num-rounds 3
                                      :valid-sets (list (cons "valid" valid-set))
                                      :parameters (getf fixture :booster-parameters))
                     (unwind-protect
                          (let ((all-series (cl-gbdt:training-report-series report)))
                            (testing (format nil "~A: the training set's series (index 0) ~
                                                  stay unnamed" (getf fixture :backend))
                              (ok (every (lambda (series)
                                           (or (/= 0 (cl-gbdt:training-series-index series))
                                               (null (cl-gbdt:training-series-name series))))
                                         all-series)
                                  (format nil "series were ~S" all-series)))
                            (testing (format nil "~A: the named validation set's series ~
                                                  (index 1) carry name \"valid\""
                                             (getf fixture :backend))
                              (ok (and (some (lambda (series)
                                               (= 1 (cl-gbdt:training-series-index series)))
                                             all-series)
                                       (every (lambda (series)
                                                (or (/= 1 (cl-gbdt:training-series-index
                                                           series))
                                                    (string= "valid"
                                                             (cl-gbdt:training-series-name
                                                              series))))
                                              all-series))
                                  (format nil "series were ~S" all-series))))
                       (cl-gbdt:free-booster booster)))))
            (cl-gbdt:close-backend backend)))))))

;;; A mixed list -- one bare entry, one named entry -- produces both a NIL and a non-NIL
;;; name in the same report, at the two different indices the two entries occupy.

(deftest training-report-mixed-valid-sets-carry-nil-and-string-names
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
              (bare-label-vector (invert-labels label-vector)))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (bare-valid (make-fixture-dataset fixture backend matrix bare-label-vector
                                                       :reference train-set))
                   (cl-gbdt:with-dataset
                       (named-valid (make-fixture-dataset fixture backend matrix label-vector
                                                          :reference train-set))
                     (multiple-value-bind (booster report)
                         (cl-gbdt:train backend train-set :num-rounds 3
                                        :valid-sets (list bare-valid (cons "named" named-valid))
                                        :parameters (getf fixture :booster-parameters))
                       (unwind-protect
                            (let ((all-series (cl-gbdt:training-report-series report)))
                              (testing (format nil "~A: the bare entry's series (index 1) ~
                                                    are unnamed" (getf fixture :backend))
                                (ok (some (lambda (series)
                                            (and (= 1 (cl-gbdt:training-series-index series))
                                                 (null (cl-gbdt:training-series-name series))))
                                          all-series)
                                    (format nil "series were ~S" all-series)))
                              (testing (format nil "~A: the named entry's series (index 2) ~
                                                    carry name \"named\""
                                               (getf fixture :backend))
                                (ok (some (lambda (series)
                                            (and (= 2 (cl-gbdt:training-series-index series))
                                                 (string= "named"
                                                          (cl-gbdt:training-series-name
                                                           series))))
                                          all-series)
                                    (format nil "series were ~S" all-series))))
                         (cl-gbdt:free-booster booster))))))
            (cl-gbdt:close-backend backend)))))))

;;; Two validation sets sharing one name is accepted, not an error: the two libraries have
;;; no uniqueness rule for a validation set's own name, and inventing one here would be a
;;; rule neither backend has. The index -- 1 and 2 -- is what tells the two series apart.

(deftest training-report-duplicate-valid-set-names-are-accepted
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
                     (multiple-value-bind (booster report)
                         (cl-gbdt:train backend train-set :num-rounds 3
                                        :valid-sets (list (cons "valid" valid-1)
                                                          (cons "valid" valid-2))
                                        :parameters (getf fixture :booster-parameters))
                       (unwind-protect
                            (let ((all-series (cl-gbdt:training-report-series report)))
                              (testing (format nil "~A: two validation sets sharing one ~
                                                    name are accepted, distinguished by ~
                                                    index 1 and index 2"
                                               (getf fixture :backend))
                                (ok (and (some (lambda (series)
                                                 (and (= 1 (cl-gbdt:training-series-index
                                                            series))
                                                      (string= "valid"
                                                               (cl-gbdt:training-series-name
                                                                series))))
                                               all-series)
                                         (some (lambda (series)
                                                 (and (= 2 (cl-gbdt:training-series-index
                                                            series))
                                                      (string= "valid"
                                                               (cl-gbdt:training-series-name
                                                                series))))
                                               all-series))
                                    (format nil "series were ~S" all-series))))
                         (cl-gbdt:free-booster booster))))))
            (cl-gbdt:close-backend backend)))))))

;;; The two ways one :VALID-SETS element can be malformed are two different mistakes and
;;; must report as two different conditions: a cons whose car is not a string is an
;;; argument-shape mistake, `unsupported-argument'; a cons whose cdr is not this backend's
;;; own kind of dataset is a wrong handle, `wrong-backend-reference' -- the same condition
;;; a bare wrong-backend dataset already signals elsewhere in this suite. `handler-case',
;;; not rove's `signals', which does not reliably catch a condition raised inside
;;; `restart-case'; the condition TYPE is asserted, not merely that something signalled.

(deftest train-valid-sets-cons-with-non-string-car-signals-unsupported-argument
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               (cl-gbdt:with-dataset
                   (train-set (make-fixture-dataset fixture backend matrix label-vector))
                 (cl-gbdt:with-dataset
                     (valid-set (make-fixture-dataset fixture backend matrix label-vector
                                                      :reference train-set))
                   (testing (format nil "~A: a :valid-sets cons with a non-string car ~
                                        signals unsupported-argument"
                                    (getf fixture :backend))
                     (ok (handler-case
                             (progn (cl-gbdt:train backend train-set :num-rounds 1
                                                   :valid-sets (list (cons :valid valid-set))
                                                   :parameters (getf fixture
                                                                     :booster-parameters))
                                    nil)
                           (cl-gbdt:unsupported-argument () t))
                         "train did not signal unsupported-argument for a non-string name"))))
            (cl-gbdt:close-backend backend)))))))

;;; The other mistake needs a dataset from the OTHER backend as the cons' cdr, so this
;;; opens both libraries at once, mirroring evaluation.lisp's own cross-backend guard
;;; tests -- the only other place in this suite that needs both.

(deftest train-valid-sets-cons-with-wrong-backend-dataset-signals-wrong-backend-reference
  (with-backend-library (:lightgbm)
    (with-backend-library (:xgboost)
      (multiple-value-bind (matrix label-vector) (make-separable-dataset)
        (let ((lightgbm-fixture (fixture-for :lightgbm))
              (xgboost-fixture (fixture-for :xgboost))
              (lightgbm (cl-gbdt:open-backend :lightgbm))
              (xgboost (cl-gbdt:open-backend :xgboost))
              (lightgbm-train nil)
              (xgboost-train nil))
          (unwind-protect
               (progn
                 (setf lightgbm-train
                       (make-fixture-dataset lightgbm-fixture lightgbm matrix label-vector))
                 (setf xgboost-train
                       (make-fixture-dataset xgboost-fixture xgboost matrix label-vector))
                 (testing "LightGBM's train rejects a named XGBoost dataset in :valid-sets"
                   (ok (handler-case
                           (progn (cl-gbdt:train lightgbm lightgbm-train :num-rounds 1
                                                 :valid-sets (list (cons "valid" xgboost-train))
                                                 :parameters (getf lightgbm-fixture
                                                                   :booster-parameters))
                                  nil)
                         (cl-gbdt:wrong-backend-reference () t))
                       "LightGBM's train did not signal wrong-backend-reference for a ~
                        named XGBoost dataset"))
                 (testing "XGBoost's train rejects a named LightGBM dataset in :valid-sets"
                   (ok (handler-case
                           (progn (cl-gbdt:train xgboost xgboost-train :num-rounds 1
                                                 :valid-sets (list (cons "valid"
                                                                        lightgbm-train))
                                                 :parameters (getf xgboost-fixture
                                                                   :booster-parameters))
                                  nil)
                         (cl-gbdt:wrong-backend-reference () t))
                       "XGBoost's train did not signal wrong-backend-reference for a ~
                        named LightGBM dataset")))
            (progn
              (when lightgbm-train (cl-gbdt:free-dataset lightgbm-train))
              (when xgboost-train (cl-gbdt:free-dataset xgboost-train))
              (cl-gbdt:close-backend lightgbm)
              (cl-gbdt:close-backend xgboost))))))))
