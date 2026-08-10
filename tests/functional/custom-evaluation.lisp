;;;; custom-evaluation.lisp --- Portable contract tests for `train''s :EVALUATION argument.
;;;;
;;;; `cl-gbdt:train' now accepts an :EVALUATION function: called once per dataset per
;;;; iteration with that dataset's current predictions and its index, returning a metric name
;;;; and a value. The run then records the caller's own measure of fit beside the library's,
;;;; in the same report, under the same (DATASET-INDEX, METRIC-NAME) keying -- which is what
;;;; lets `:early-stopping' watch one with nothing extra arranged for it.
;;;;
;;;; The capability is `:custom-evaluation', and `train' re-checks it -- a backend that
;;;; answers false signals `cl-gbdt:capability-unavailable' for a non-NIL :EVALUATION rather
;;;; than quietly recording only the library's own metrics, which is the silent fallback
;;;; policy section 7 forbids. That is what makes the loop in the first test below an
;;;; asked/demonstrated pair rather than a single-branch test: every backend is asked, one
;;;; branch records and the other catches the refusal, and the count after the loop is what
;;;; keeps a suite in which NO backend provides the capability from passing having asserted
;;;; nothing.
;;;;
;;;; THIS FILE IS ONE-SIDED TODAY, BY DESIGN. LightGBM answers the capability true; XGBoost
;;;; declares it in neither of its capability lists and so answers false, its
;;;; `%check-custom-evaluation' refusing every non-NIL :EVALUATION. So on XGBoost every test
;;;; below except the first takes its `(when (cl-gbdt:backend-supports-p ...))' false branch
;;;; and asserts nothing, and the first one asserts on that backend through its REFUSAL
;;;; branch. That is deliberate and is why the guards are written this way: the commit that
;;;; declares the capability on XGBoost makes every test here cover it WITH NO EDIT TO THIS
;;;; FILE, exactly as tests/functional/custom-objective.lisp's guards did for :OBJECTIVE on
;;;; the day that backend stopped refusing it.
;;;;
;;;; Numbers are never compared BETWEEN backends -- policy section 13. Every comparison below
;;;; is one backend's caller-written metric against that SAME backend's own library metric,
;;;; computed on the same booster in the same run, which is the only comparison that means
;;;; anything: the two libraries build different trees from the same rows by design, while
;;;; log loss is defined identically by both, so "did the caller's function see the numbers
;;;; the library saw" is answerable within one backend and nowhere else.
;;;;
;;;; The fixture is deterministic rather than random, and stated here rather than taken from
;;;; tests/functional/evaluation.lisp's *FIXTURES*: the central test's whole claim is that a
;;;; caller-written log loss REPRODUCES the library's own to within a tolerance, and a random
;;;; fixture makes a near-miss indistinguishable from noise. It also needs a validation block
;;;; with a DIFFERENT row count from the training block, which that shared fixture has no
;;;; reason to carry.

(uiop:define-package #:cl-gbdt/tests/functional/custom-evaluation
  ;; Zero symbols: every reference below is package-qualified. Declared so this file's
  ;; dependency on the unified API is explicit rather than inherited, matching the identical
  ;; clause in evaluation.lisp, sparse-input.lisp, missing-value.lisp,
  ;; categorical-features.lisp, prediction-shape.lisp and custom-objective.lisp.
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt)
  ;; Zero symbols, both of them: their only job is to run at load time and register
  ;; :lightgbm and :xgboost with `open-backend'. Without these clauses,
  ;; package-inferred-system has no edge to those files and `(cl-gbdt:open-backend
  ;; :lightgbm)' below would signal `unknown-backend'.
  (:import-from #:cl-gbdt/src/lightgbm/all)
  (:import-from #:cl-gbdt/src/xgboost/all)
  ;; A nickname rather than an `:import-from' naming the symbols, matching
  ;; custom-objective.lisp: `with-backend-library' is the skip-or-run wrapper every test
  ;; below opens with, and the prefix keeps "this is the suite's shared library discovery"
  ;; visible at each of those call sites.
  (:local-nicknames (#:support #:cl-gbdt/tests/functional/support)))

(in-package #:cl-gbdt/tests/functional/custom-evaluation)

;;; ---------------------------------------------------------------------------
;;; The fixture

(defparameter *rows* 40
  "Rows in the training block below.")

(defparameter *valid-rows* 17
  "Rows in the validation block below -- deliberately NOT *ROWS*.

A row count is what says which dataset a read addressed. `train' hands the caller's
:EVALUATION function one array per dataset per iteration, and every one of them would have
the same shape if both blocks were the same height, so a backend that handed the training
set's predictions to every index would pass the shape test below without moving. 17 against
40 is also what the per-dataset length in
`cl-gbdt/src/lightgbm/native''s `%num-predict' was measured on: `LGBM_BoosterGetNumPredict'
answers 17 for this validation set attached to this training set.")

(defparameter *columns* 3)

(defun binary-fixture (rows offset)
  "Return two values: a (ROWS *COLUMNS*) `double-float' matrix and a `double-float' label
vector of 0.0 and 1.0, both a deterministic function of the row index and OFFSET.

OFFSET shifts the pattern so the validation block is not a prefix of the training block --
otherwise the two datasets' metrics would coincide and the index-discrimination this file
rests on would prove nothing. Deterministic rather than random, for the reason this file's
header gives."
  (let ((matrix (make-array (list rows *columns*) :element-type 'double-float))
        (labels* (make-array rows :element-type 'double-float)))
    (dotimes (row rows)
      (dotimes (column *columns*)
        (setf (aref matrix row column)
              (coerce (mod (+ (* 3 (+ row offset)) (* 7 column)) 11) 'double-float)))
      (setf (aref labels* row) (if (evenp (+ row offset)) 1d0 0d0)))
    (values matrix labels*)))

(defparameter *dataset-parameters*
  '((:lightgbm :min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)
    (:xgboost))
  "What each backend's `make-dataset' needs for the fixture above, and load-bearing rather
than defensive on LightGBM: with its defaults that library bins these columns into one bin
apiece and warns \"There are no meaningful features which satisfy the provided
configuration\", after which every tree is a root-only leaf and every prediction is one
constant. The agreement test below would still pass on a constant -- both sides would read
the same constant -- but nothing else here would be measuring anything.

XGBoost's is empty because that backend signals `unsupported-argument' for a non-NIL
:PARAMETERS on `make-dataset' at all -- the same split
`cl-gbdt/tests/functional/evaluation''s *FIXTURES* records for the same reason. NIL is
accepted there and means the same nothing it means here.")

(defparameter *metric-parameters*
  '((:lightgbm :objective "binary" :metric "binary_logloss" :num-leaves 7
     :min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)
    (:xgboost :objective "binary:logistic" :eval-metric "logloss" :max-depth 3
     :eta 0.3d0 :verbosity 0 :min-child-weight 0))
  "Each backend's own spelling of \"binary classification, reporting log loss\".

The objective matters as much as the metric: the caller-written metric below reads
`predict :kind :normal''s PROBABILITIES, and it is the classification objective that makes
the numbers `train' hands it probabilities rather than raw margins. Measured on LightGBM,
one `objective=binary' booster over this fixture: the buffer `train' hands the caller agrees
with `cl-gbdt:predict :kind :normal' over the same matrix to 0.0 for the 40-row training set
AND for the 17-row validation set, while differing from `:raw' by 0.706.")

(defparameter *no-metric-parameters*
  '((:lightgbm :objective "binary" :num-leaves 7 :min-data-in-leaf 1 :min-data-in-bin 1
     :verbose -1 :metric "none")
    (:xgboost :objective "binary:logistic" :max-depth 3 :eta 0.3d0 :verbosity 0
     :min-child-weight 0 :disable-default-eval-metric 1))
  "The same boosters with every LIBRARY metric turned off, which each library spells its own
way -- LightGBM's `metric=none' and XGBoost's `disable_default_eval_metric=1', the latter
needed because XGBoost configures a metric from the objective when the caller names none.
Both were confirmed against the vendored libraries to make `cl-gbdt:evaluation' answer NIL,
which is `cl-gbdt/tests/functional/evaluation''s *FIXTURES* :NO-METRIC-PARAMETERS finding
restated on this file's own fixture.

Two tests below need it, for two different reasons: the early-stopping test, so the ONLY
series in the report is the caller's own and the watcher cannot be watching a library metric
by accident; and the collision test, so the same name that collides under
*METRIC-PARAMETERS* is accepted here -- what the check compares against is what this booster
actually reported, not a list of well-known metric names.")

(defparameter *library-metric-names*
  '((:lightgbm . "binary_logloss") (:xgboost . "logloss"))
  "Each backend's own name for log loss, exactly as `cl-gbdt:evaluation' spells it -- the
same split *FIXTURES* records as its :LOSS-METRIC, and no more translated into one another
here than it is there. The agreement test finds the library's series by this name and the
collision test hands this name back as the caller's own.")

(defun library-metric (name)
  "Return backend NAME's own spelling of log loss -- see *LIBRARY-METRIC-NAMES*."
  (cdr (assoc name *library-metric-names*)))

(defun make-labelled-dataset (backend name matrix labels* &key reference)
  "Build a dataset on BACKEND, named NAME, from MATRIX and LABELS*, passing only the
`make-dataset' :PARAMETERS that backend accepts -- see *DATASET-PARAMETERS*. One call site
works for both backends, which is what lets every test below be written once.

REFERENCE, when supplied, is the dataset this one's bin mappers must align with -- what
`LGBM_BoosterAddValidData' requires of a validation set, and NOT optional here. Measured on
this fixture: a LightGBM validation set built WITHOUT it is accepted and trains without
complaint, but is binned independently of the training set, and the predictions the booster
then holds for it -- the very array :EVALUATION is handed -- diverge from `cl-gbdt:predict'
over the same matrix by 0.13631237652787437, against exactly 0.0 with it. That is what the
scores-against-`predict' assertion in the agreement test below would fail on, for a reason
that has nothing to do with :EVALUATION.

It reaches `make-dataset' on LightGBM only: XGBoost has no bin-mapper
alignment and signals `unsupported-argument' for a non-NIL :REFERENCE, so passing it there
would fail the call rather than be ignored. The same split, for the same reason, is what
`cl-gbdt/tests/functional/evaluation''s `make-fixture-dataset' spells as its fixtures'
:ALIGNS-BIN-MAPPERS key."
  (apply #'cl-gbdt:make-dataset backend matrix :label labels*
         :parameters (cdr (assoc name *dataset-parameters*))
         (when (and reference (eq name :lightgbm)) (list :reference reference))))

;;; ---------------------------------------------------------------------------
;;; The caller's metrics, and how their series are read back

(defparameter *custom-metric-name* "my_logloss"
  "The name the caller-written log loss returns. Deliberately NOT either library's own
spelling of log loss: the collision test below is the one place a caller hands back a name
the library already uses, and it must be the only place.")

(defparameter *metric-tolerance* 1d-9
  "How far the caller's own log loss may sit from the library's own and still count as the
same number.

The two sums are of the same per-row terms over the same rows, from the identical
`double-float' probabilities -- `train' hands the caller the very buffer the library
evaluated -- but they are computed by different code in different languages, and neither
accumulates in an order the other can be held to. So they agree to the LAST BITS rather than
exactly: measured on LightGBM over five iterations, the largest difference across the whole
series was 6.66d-16 at dataset index 0 and 1.11d-16 at index 1. `=' would be a true negative
dressed as a failure.

Six orders of magnitude of headroom is what the number is chosen for, in both directions:
nothing this file exists to catch survives 1d-9 either. A metric that ignored SCORES
entirely -- the control below, which returns a flat 0.5 -- sits about 0.19 away from a
library series measured between 0.682 and 0.698, and a validation set whose bin mappers were
never aligned moves the predictions themselves by 0.136. It is
`cl-gbdt/tests/functional/support''s *PREDICTION-TOLERANCE* restated rather than a new idea,
and is not imported from there only because what it bounds is a metric rather than a
prediction.")

(defun logloss-evaluation (label-vectors)
  "Return an :EVALUATION function computing binary log loss over the probabilities it is
handed, for whichever dataset index it is called with.

LABEL-VECTORS is a vector of label vectors indexed by DATASET INDEX -- element 0 the
training set's, element 1 the first :VALID-SETS entry's -- which is the whole reason the
index is one of the two arguments the function is called with: a metric that could not tell
the datasets apart could not label them apart either.

Both libraries define binary log loss as the unweighted mean over rows of
`-(y ln p + (1-y) ln (1-p))', so a run driven by this must reproduce the same library's own
log-loss series. That is what makes it a test of whether the caller's function actually saw
the numbers the library saw, rather than of whether it was called at all.

P is clamped away from 0 and 1 before `log'. Measured on this fixture the clamp is never
reached -- five iterations leave every probability between 0.431844 and 0.568156 -- so it
cannot be what makes the two series agree; it is there so a later fixture or a longer run
cannot turn this function into one that returns an infinity, which would compare equal to
nothing and unequal to nothing."
  (lambda (scores index)
    (let ((labels* (aref label-vectors index))
          (rows (array-dimension scores 0))
          (sum 0d0))
      (dotimes (row rows)
        (let ((p (min (max (aref scores row 0) 1d-15) (- 1d0 1d-15)))
              (y (aref labels* row)))
          (incf sum (- (+ (* y (log p)) (* (- 1d0 y) (log (- 1d0 p))))))))
      (values *custom-metric-name* (/ sum rows)))))

(defun series-values (report index metric)
  "Return the values of REPORT's series for dataset INDEX and metric name METRIC, or NIL
when the report carries no such series.

`cl-gbdt:training-series-index' and `cl-gbdt:training-series-metric' are the two accessors
that name the pair a series is keyed by; `cl-gbdt:training-series-name' is the :VALID-SETS
NAME and is not one of them -- every dataset here is passed bare, so every series' name is
NIL and it discriminates nothing."
  (let ((series (find-if (lambda (series)
                           (and (eql index (cl-gbdt:training-series-index series))
                                (equal metric (cl-gbdt:training-series-metric series))))
                         (cl-gbdt:training-report-series report))))
    (and series (cl-gbdt:training-series-values series))))

(defun series-agree-p (left right)
  "True when LEFT and RIGHT, two series' value vectors, are the same length and differ
nowhere by more than *METRIC-TOLERANCE*."
  (and (= (length left) (length right))
       (every (lambda (a b) (and (realp a) (realp b) (<= (abs (- a b)) *metric-tolerance*)))
              left right)))

(defun report-pairs (report)
  "Return REPORT's series as a list of (DATASET-INDEX . METRIC-NAME) conses, in the order
the report holds them."
  (mapcar (lambda (series)
            (cons (cl-gbdt:training-series-index series)
                  (cl-gbdt:training-series-metric series)))
          (cl-gbdt:training-report-series report)))

(defun evaluation-pairs (booster)
  "Return `cl-gbdt:evaluation''s entries for BOOSTER as (DATASET-INDEX . METRIC-NAME)
conses, in the order that function reports them."
  (mapcar (lambda (entry) (cons (first entry) (second entry)))
          (cl-gbdt:evaluation booster)))

;;; ---------------------------------------------------------------------------
;;; The capability

(deftest custom-evaluation-capability-is-true-where-it-is-demonstrated
  ;; The capability is asserted in the same test that records a custom metric, and the count
  ;; after the loop is what keeps a backend that quietly stopped providing it from leaving
  ;; this suite green having asserted nothing. Two capabilities have shipped false on this
  ;; project.
  (let ((demonstrated 0))
    (dolist (name '(:lightgbm :xgboost))
      (support:with-backend-library (name)
        (let ((backend (cl-gbdt:open-backend name)))
          (unwind-protect
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                        matrix labels*))
                   (if (cl-gbdt:backend-supports-p backend :custom-evaluation)
                       (progn
                         (incf demonstrated)
                         (multiple-value-bind (booster report)
                             (cl-gbdt:train
                              backend dataset :num-rounds 3
                              :parameters (cdr (assoc name *metric-parameters*))
                              :evaluation (logloss-evaluation (vector labels*)))
                           (cl-gbdt:free-booster booster)
                           (let ((values (series-values report 0 *custom-metric-name*)))
                             (ok values "the report carries the caller's own series")
                             (ok (and values (= 3 (length values)))
                                 "one value per completed iteration"))))
                       ;; All three assertions, not just "something signalled" -- a refusal
                       ;; naming the wrong capability or the wrong backend is a refusal a
                       ;; caller cannot act on. This is the branch XGBoost takes today, and
                       ;; the one a LightGBM missing one of its three C symbols would take.
                       (let ((condition
                               (handler-case
                                   (progn (cl-gbdt:free-booster
                                           (cl-gbdt:train
                                            backend dataset :num-rounds 1
                                            :parameters (cdr (assoc name *metric-parameters*))
                                            :evaluation (logloss-evaluation
                                                         (vector labels*))))
                                          nil)
                                 (cl-gbdt:capability-unavailable (c) c))))
                         (ok condition "train signalled instead of recording")
                         (ok (and condition
                                  (eq :custom-evaluation
                                      (cl-gbdt:capability-unavailable-capability condition)))
                             (format nil "the condition named capability ~S"
                                     (and condition
                                          (cl-gbdt:capability-unavailable-capability
                                           condition))))
                         (ok (and condition
                                  (eq name (cl-gbdt:backend-error-backend condition)))
                             (format nil "the condition named backend ~S"
                                     (and condition
                                          (cl-gbdt:backend-error-backend condition))))))))
            (cl-gbdt:close-backend backend)))))
    (ok (plusp demonstrated))))

;;; Policy section 7's central rule, watched rather than assumed: `train' re-checks the
;;; capability itself instead of trusting the caller to have asked `backend-supports-p'
;;; first, and signals rather than quietly recording only the library's metrics. LightGBM
;;; provides the capability -- the test above asserts exactly that -- so the only way to
;;; reach the gate there is to overwrite the probed plist, which is what a LightGBM missing
;;; one of `LGBM_BoosterGetPredict', `LGBM_BoosterGetNumPredict' or
;;; `LGBM_BoosterGetNumClasses' would have produced at `open-backend'. This is the same way
;;; `cl-gbdt/tests/functional/sparse-input''s `sparse-input-without-the-capability-signals',
;;; `cl-gbdt/tests/functional/missing-value''s
;;; `missing-value-without-the-capability-signals' and
;;; `cl-gbdt/tests/functional/custom-objective''s
;;; `custom-objective-without-the-capability-signals' reach their own gates. The
;;; custom-objective branch shipped its gate untested at first: both `%check-custom-objective'
;;; calls could have been deleted with the whole suite green.
;;;
;;; `handler-case', not rove's `signals', which does not reliably catch a condition raised
;;; inside `restart-case'; the condition TYPE is asserted, not merely that something
;;; signalled, and so are the capability and the backend it names.

(deftest withdrawing-the-capability-makes-train-refuse
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
               (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                      matrix labels*))
                 ;; Overwritten after the dataset is built, since building it needs the
                 ;; backend's other capabilities and this replaces the whole plist. A backend
                 ;; that already answered false would be left exactly as it opened.
                 (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
                   (setf (cl-gbdt:backend-capabilities backend) '(:custom-evaluation nil)))
                 (testing (format nil "~A: train signals capability-unavailable for a ~
                                       non-NIL :evaluation, naming the capability and the ~
                                       backend" name)
                   ;; `free-booster' on the success branch, the way
                   ;; `sparse-input-without-the-capability-signals' frees the dataset it does
                   ;; not expect to get: that branch is only reached if the gate has
                   ;; regressed, and a leaked handle is a poor second failure to hand whoever
                   ;; is already reading the first one.
                   (let ((condition
                           (handler-case
                               (progn (cl-gbdt:free-booster
                                       (cl-gbdt:train
                                        backend dataset :num-rounds 1
                                        :parameters (cdr (assoc name *metric-parameters*))
                                        :evaluation (logloss-evaluation (vector labels*))))
                                      nil)
                             (cl-gbdt:capability-unavailable (c) c))))
                     (ok condition "train signalled instead of recording")
                     (ok (and condition
                              (eq :custom-evaluation
                                  (cl-gbdt:capability-unavailable-capability condition)))
                         (format nil "the condition named capability ~S"
                                 (and condition
                                      (cl-gbdt:capability-unavailable-capability condition))))
                     (ok (and condition
                              (eq name (cl-gbdt:backend-error-backend condition)))
                         (format nil "the condition named backend ~S"
                                 (and condition
                                      (cl-gbdt:backend-error-backend condition))))))
                 ;; EVALUATION NIL reaches no check at all, so the same backend still trains
                 ;; and records normally with the capability withdrawn. Without this the gate
                 ;; could be an unconditional refusal and nothing here would notice.
                 (multiple-value-bind (booster report)
                     (cl-gbdt:train backend dataset :num-rounds 2
                                    :parameters (cdr (assoc name *metric-parameters*)))
                   (cl-gbdt:free-booster booster)
                   (ok (series-values report 0 (library-metric name))
                       "the library's own series is recorded as it always was"))))
          (cl-gbdt:close-backend backend))))))

;;; ---------------------------------------------------------------------------
;;; What the function is handed, and what it produces

(deftest a-custom-metric-agrees-with-the-library-metric-it-reimplements
  ;; The central test. A caller-written log loss over `predict :kind :normal''s
  ;; probabilities, against the SAME backend's own log-loss series for the SAME dataset in
  ;; the SAME run -- policy section 13 forbids comparing numbers across backends, and there
  ;; would be nothing to learn from it anyway.
  ;;
  ;; Both datasets, not just the training set: the caller's function is handed one array per
  ;; dataset, and a backend that handed the training set's predictions to every index would
  ;; agree at index 0 and disagree at index 1. The two library series really are different
  ;; numbers here -- measured on LightGBM, 0.682 at index 0 against 0.698 at index 1 after
  ;; five iterations -- so index 1's agreement is not index 0's agreement in disguise.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (multiple-value-bind (valid-matrix valid-labels)
                     (binary-fixture *valid-rows* 5)
                   (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                         matrix labels*))
                     (cl-gbdt:with-dataset
                         (valid-set (make-labelled-dataset backend name valid-matrix
                                                           valid-labels :reference dataset))
                       (let ((last-scores (make-array 2 :initial-element nil))
                             (logloss (logloss-evaluation (vector labels* valid-labels))))
                         (multiple-value-bind (booster report)
                             (cl-gbdt:train
                              backend dataset :valid-sets (list valid-set) :num-rounds 5
                              :parameters (cdr (assoc name *metric-parameters*))
                              :evaluation (lambda (scores index)
                                            (setf (aref last-scores index) scores)
                                            (funcall logloss scores index)))
                           (dolist (index '(0 1))
                             (let ((mine (series-values report index *custom-metric-name*))
                                   (theirs (series-values report index
                                                          (library-metric name))))
                               (ok mine (format nil "index ~D has the caller's series" index))
                               (ok theirs (format nil "index ~D has the library's series"
                                                  index))
                               (ok (and mine theirs (series-agree-p mine theirs))
                                   (format nil "index ~D: ~S against ~S" index mine theirs))))
                           ;; And the array itself is `predict :kind :normal''s for THAT
                           ;; dataset, which is what `train''s generic docstring and
                           ;; `%booster-predictions' both state and what nothing else in
                           ;; this suite holds. The last call of the run is compared, since
                           ;; :EVALUATION runs after that iteration's update and so sees the
                           ;; same model `predict' sees once `train' has returned -- measured
                           ;; 0.0 on both datasets. This is also the assertion the
                           ;; :REFERENCE-less validation set of `make-labelled-dataset''s
                           ;; docstring fails, by 0.136, while everything above it passes.
                           (ok (support:predictions-agree-p
                                (aref last-scores 0)
                                (cl-gbdt:predict booster matrix :kind :normal))
                               "index 0's array is predict :kind :normal's")
                           (ok (support:predictions-agree-p
                                (aref last-scores 1)
                                (cl-gbdt:predict booster valid-matrix :kind :normal))
                               "index 1's array is predict :kind :normal's")
                           (cl-gbdt:free-booster booster)))
                       ;; THE CONTROL. A metric that never reads SCORES at all must NOT
                       ;; reproduce the library's series; without it, a `train' that called
                       ;; the caller's function with the wrong array -- or with a stale one,
                       ;; or never -- could still pass the assertions above if the numbers
                       ;; happened to land close. It returns a plausible-looking log loss
                       ;; rather than something absurd, so what it pins is that the SCORES
                       ;; argument is load-bearing.
                       (multiple-value-bind (booster report)
                           (cl-gbdt:train
                            backend dataset :valid-sets (list valid-set) :num-rounds 5
                            :parameters (cdr (assoc name *metric-parameters*))
                            :evaluation (lambda (scores index)
                                          (declare (ignore scores index))
                                          (values *custom-metric-name* 0.5d0)))
                         (cl-gbdt:free-booster booster)
                         (dolist (index '(0 1))
                           (let ((mine (series-values report index *custom-metric-name*))
                                 (theirs (series-values report index (library-metric name))))
                             (ok (and mine theirs (not (series-agree-p mine theirs)))
                                 (format nil "index ~D: the control must not agree"
                                         index))))))))))
          (cl-gbdt:close-backend backend))))))

(deftest the-metric-is-called-once-per-dataset-per-iteration
  ;; What the call SEQUENCE is, and what each call is handed. The recorded index sequence is
  ;; compared against the full cross product of the two dataset indices with the run's
  ;; iterations, laid out dataset-major -- which pins the count (one call per dataset per
  ;; iteration, no more and no fewer), the coverage (every dataset every iteration) and the
  ;; order (index 0 before index 1 within an iteration) in one assertion, without any of them
  ;; being assumed to derive the others.
  ;;
  ;; The shape assertion is where the 17-row validation block pays off: `(40 1)' and `(17 1)'
  ;; are different arrays, so a backend that read every index's predictions out of the
  ;; training set's buffer fails here rather than passing on a coincidence.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (multiple-value-bind (valid-matrix valid-labels)
                     (binary-fixture *valid-rows* 5)
                   (let ((calls '())
                         (rounds 3))
                     (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                           matrix labels*))
                       (cl-gbdt:with-dataset
                           (valid-set (make-labelled-dataset backend name valid-matrix
                                                             valid-labels
                                                             :reference dataset))
                         (multiple-value-bind (booster report)
                             (cl-gbdt:train
                              backend dataset :valid-sets (list valid-set)
                              :num-rounds rounds
                              :parameters (cdr (assoc name *metric-parameters*))
                              :evaluation (lambda (scores index)
                                            (push (cons index (array-dimensions scores))
                                                  calls)
                                            (values *custom-metric-name* 0.5d0)))
                           (cl-gbdt:free-booster booster)
                           (setf calls (nreverse calls))
                           (ok (= rounds (cl-gbdt:training-report-num-rounds report)))
                           (ok (equal (mapcar #'car calls)
                                      (loop :repeat rounds :append '(0 1)))
                               (format nil "the index sequence was ~S" (mapcar #'car calls)))
                           ;; Each index always got its OWN dataset's shape, on every one of
                           ;; the iterations, not merely on the first.
                           (ok (every (lambda (call)
                                        (equal (cdr call)
                                               (if (zerop (car call))
                                                   (list *rows* 1)
                                                   (list *valid-rows* 1))))
                                      calls)
                               (format nil "the shapes were ~S"
                                       (remove-duplicates (mapcar #'cdr calls)
                                                          :test #'equal))))))))))
          (cl-gbdt:close-backend backend))))))

;;; ---------------------------------------------------------------------------
;;; What the recorded values are good for

(defun stalling-evaluation (name)
  "Return an :EVALUATION function whose value improves for three calls and then stalls:
3.0, 2.0, 1.0, 1.0, 1.0, ... under NAME, ignoring the predictions entirely.

Ignoring them is the point. What is under test is that a caller's metric reaches
`:early-stopping' at all, and a value derived from a real model would make WHEN the run
stopped a fact about the model rather than about the plumbing. Counting calls is counting
ITERATIONS here only because the test that uses this passes no :VALID-SETS, so there is
exactly one dataset and therefore one call per iteration."
  (let ((calls 0))
    (lambda (scores index)
      (declare (ignore scores index))
      (incf calls)
      (values name (coerce (max 1 (- 4 calls)) 'double-float)))))

(deftest early-stopping-can-watch-a-custom-metric
  ;; A custom metric is recorded into the SAME per-iteration entry list the watcher reads,
  ;; before the watcher runs -- which is the whole reason `:early-stopping' needs nothing
  ;; added to it to watch one. The booster is trained with every library metric turned off,
  ;; so the series the watcher finds cannot be a library series the run happened to also
  ;; report.
  ;;
  ;; :ROUNDS 2 against a value that improves at iterations 1, 2 and 3 and then holds:
  ;; iteration 4 is the first non-improvement (improvement is strict, so a plateau does not
  ;; count), iteration 5 the second, and two consecutive is what :ROUNDS 2 tolerates. So the
  ;; run stops after 5 of the 10 asked for, with iteration 3 the best.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   (multiple-value-bind (booster report)
                       (cl-gbdt:train
                        backend dataset :num-rounds 10
                        :parameters (cdr (assoc name *no-metric-parameters*))
                        :evaluation (stalling-evaluation *custom-metric-name*)
                        :early-stopping (list :metric *custom-metric-name* :dataset 0
                                              :direction :lower-is-better :rounds 2))
                     (cl-gbdt:free-booster booster)
                     (ok (= 5 (cl-gbdt:training-report-num-rounds report))
                         (format nil "the run ran ~D iterations"
                                 (cl-gbdt:training-report-num-rounds report)))
                     (ok (cl-gbdt:training-report-early-stopped-p report))
                     (ok (= 3 (cl-gbdt:training-report-best-iteration report))
                         (format nil "best iteration was ~S"
                                 (cl-gbdt:training-report-best-iteration report)))
                     (ok (= 1d0 (cl-gbdt:training-report-best-score report))
                         (format nil "best score was ~S"
                                 (cl-gbdt:training-report-best-score report)))
                     ;; The series is five long, not ten: an early-stopped run's series are
                     ;; as long as the run, and the caller's series is no exception.
                     (ok (= 5 (length (series-values report 0 *custom-metric-name*))))))))
          (cl-gbdt:close-backend backend))))))

(deftest the-library-series-are-still-evaluation-s-entries-in-order
  ;; `train''s generic docstring promises the library's own series are a PREFIX of the
  ;; report's, in `cl-gbdt:evaluation''s own order, with the caller's appended after them.
  ;; That is what lets a caller who could find a library series before a custom metric
  ;; existed still find it the same way. It is asserted as a prefix rather than as set
  ;; membership because the ORDER is the half that a naive `append' in the other direction
  ;; would break while leaving every series present.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (multiple-value-bind (valid-matrix valid-labels)
                     (binary-fixture *valid-rows* 5)
                   (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                         matrix labels*))
                     (cl-gbdt:with-dataset
                         (valid-set (make-labelled-dataset backend name valid-matrix
                                                           valid-labels
                                                           :reference dataset))
                       (multiple-value-bind (booster report)
                           (cl-gbdt:train
                            backend dataset :valid-sets (list valid-set) :num-rounds 3
                            :parameters (cdr (assoc name *metric-parameters*))
                            :evaluation (logloss-evaluation (vector labels* valid-labels)))
                         (let ((library-pairs (evaluation-pairs booster))
                               (report-pairs (report-pairs report)))
                           (cl-gbdt:free-booster booster)
                           ;; Non-empty on both sides, or the prefix assertion below would
                           ;; be satisfied by two empty lists.
                           (ok (plusp (length library-pairs)))
                           (ok (> (length report-pairs) (length library-pairs))
                               "the report carries strictly more series than the library")
                           (ok (equal library-pairs
                                      (subseq report-pairs 0 (length library-pairs)))
                               (format nil "~S against ~S" library-pairs report-pairs))
                           ;; And what follows the prefix is exactly the caller's own pairs,
                           ;; one per dataset -- so "appended after" is not satisfied by a
                           ;; report that dropped them.
                           (ok (equal (subseq report-pairs (length library-pairs))
                                      (list (cons 0 *custom-metric-name*)
                                            (cons 1 *custom-metric-name*)))
                               (format nil "the suffix was ~S"
                                       (subseq report-pairs
                                               (length library-pairs)))))))))))
          (cl-gbdt:close-backend backend))))))

;;; ---------------------------------------------------------------------------
;;; The two ways a caller's metric can be refused mid-run

(deftest a-colliding-metric-name-signals
  ;; The pair (DATASET-INDEX, METRIC-NAME) is what a series is keyed by, so a caller's name
  ;; that matches one the library reports for the same dataset would put two different
  ;; quantities under one key -- see `check-metric-name-collision', which names what each
  ;; reader does with that. Checked at the end of the FIRST iteration, the first moment
  ;; there is a real evaluation to compare against.
  ;;
  ;; The second arm is the control, and it is what says the check is made against WHAT THIS
  ;; BOOSTER ACTUALLY REPORTED rather than against a list of well-known metric names: the
  ;; identical name, at the identical index, is accepted on a booster with the library's
  ;; metrics turned off, and its series is recorded under it.
  ;;
  ;; The other half of the keying -- the same name at a DIFFERENT index not colliding -- is
  ;; not constructible against either library. Both report the same metric list for EVERY
  ;; attached dataset (LightGBM's `%read-evaluation' pairs one name list with each dataset's
  ;; values; XGBoost's `XGBoosterEvalOneIter' formats one line covering all of them), so
  ;; there is no run in which the library reports a name at index 0 and not at index 1.
  ;; `check-metric-name-collision-allows-a-name-the-library-uses-elsewhere' in
  ;; tests/custom-metric.lisp asserts it directly instead, at layer 1, where the entry list
  ;; is written rather than measured.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   (flet ((train-named (parameters)
                            ;; Hand back the library's OWN name for log loss, at index 0.
                            (cl-gbdt:train backend dataset :num-rounds 3
                                           :parameters parameters
                                           :evaluation
                                           (lambda (scores index)
                                             (declare (ignore scores index))
                                             (values (library-metric name) 0.5d0)))))
                     (testing (format nil "~A: the library's own metric name at the same ~
                                           index signals" name)
                       (let ((condition
                               (handler-case
                                   (progn (cl-gbdt:free-booster
                                           (train-named (cdr (assoc name
                                                                    *metric-parameters*))))
                                          nil)
                                 (cl-gbdt:unsupported-argument (c) c))))
                         (ok condition "train signalled instead of recording")
                         (ok (and condition
                                  (equal "train's :evaluation"
                                         (cl-gbdt:unsupported-argument-argument condition)))
                             (format nil "the condition named argument ~S"
                                     (and condition
                                          (cl-gbdt:unsupported-argument-argument condition))))
                         (ok (and condition
                                  (eq name (cl-gbdt:unsupported-argument-backend condition)))
                             (format nil "the condition named backend ~S"
                                     (and condition
                                          (cl-gbdt:unsupported-argument-backend
                                           condition))))))
                     (testing (format nil "~A: the same name is accepted when the library ~
                                           reports no metric at all" name)
                       (multiple-value-bind (booster report)
                           (train-named (cdr (assoc name *no-metric-parameters*)))
                         (cl-gbdt:free-booster booster)
                         (let ((values (series-values report 0 (library-metric name))))
                           (ok values "the caller's series was recorded under that name")
                           (ok (and values (= 3 (length values)))))))))))
          (cl-gbdt:close-backend backend))))))

;;; ---------------------------------------------------------------------------
;;; Caller code that misbehaves

(deftest an-error-inside-the-metric-frees-the-booster
  ;; `train' builds a raw booster handle before the loop and only takes ownership of it at
  ;; the end. A condition raised from the caller's own metric must unwind through that same
  ;; path, or the run leaks a handle the caller has no way to free. This is the same
  ;; property `an-error-inside-the-objective-propagates-and-frees-the-booster' asserts for
  ;; :OBJECTIVE, at the other point in the loop where the caller's code runs.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   (ok (handler-case
                           (progn (cl-gbdt:train
                                   backend dataset :num-rounds 5
                                   :parameters (cdr (assoc name *metric-parameters*))
                                   :evaluation (lambda (scores index)
                                                 (declare (ignore scores index))
                                                 (error "metric gave up")))
                                  nil)
                         (simple-error () t)))
                   ;; The backend is still usable afterwards, which is the observable half of
                   ;; "the booster was freed rather than orphaned".
                   (multiple-value-bind (booster report)
                       (cl-gbdt:train backend dataset :num-rounds 2
                                      :parameters (cdr (assoc name *metric-parameters*)))
                     (cl-gbdt:free-booster booster)
                     (ok (series-values report 0 (library-metric name)))))))
          (cl-gbdt:close-backend backend))))))

;;; Policy section 13's "released handle" line, for the handles `train''s own loop holds raw
;;; pointers to across a call into the caller's code. The :EVALUATION calls are the second
;;; such point in that loop, and a metric that frees a dataset between two of them leaves the
;;; NEXT dataset's prediction read pointed at freed memory -- which is why
;;; `%custom-evaluation-entries' re-checks between consecutive datasets rather than once per
;;; iteration. Measured on the :OBJECTIVE path before `%recheck-train-datasets' existed, in
;;; isolated subprocesses: LightGBM died with `Memory fault at 0x543447170e8a6' and XGBoost
;;; with `Signal 7 received' -- the process, not the test.
;;;
;;; `handler-case', not rove's `signals'; the condition TYPE is asserted, and so is the
;;; handle it names, since a run that signalled about the wrong handle would be a run that
;;; noticed by accident.

(deftest freeing-a-dataset-inside-the-metric-signals-rather-than-faulting
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (multiple-value-bind (valid-matrix valid-labels)
                     (binary-fixture *valid-rows* 5)
                   (flet ((train-freeing (victim dataset valid-sets)
                            ;; Free VICTIM from inside the metric, then return a perfectly
                            ;; ordinary name and value: what is under test is the freed
                            ;; handle alone, and a malformed return would be caught by
                            ;; `custom-metric-entry' instead and prove nothing.
                            (handler-case
                                (progn (cl-gbdt:free-booster
                                        (cl-gbdt:train
                                         backend dataset :num-rounds 5 :valid-sets valid-sets
                                         :parameters (cdr (assoc name *metric-parameters*))
                                         :evaluation
                                         (lambda (scores index)
                                           (declare (ignore scores index))
                                           (cl-gbdt:free-dataset victim)
                                           (values *custom-metric-name* 0.5d0))))
                                       nil)
                              (cl-gbdt:released-handle-error (c) c))))
                     (testing (format nil "~A: freeing the TRAINING set inside the metric"
                                      name)
                       (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                             matrix labels*))
                         (let ((condition (train-freeing dataset dataset '())))
                           (ok condition "train signalled instead of faulting")
                           (ok (and condition
                                    (eq dataset
                                        (cl-gbdt:released-handle-error-object condition)))
                               "the condition named the training set"))))
                     ;; The validation set is re-checked too, and this is the case the
                     ;; per-dataset re-check exists for: the metric runs at index 0, frees
                     ;; the validation set, and the very next thing the loop would do is read
                     ;; index 1's predictions through it.
                     (testing (format nil "~A: freeing a VALIDATION set inside the metric"
                                      name)
                       (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                             matrix labels*))
                         (cl-gbdt:with-dataset
                             (valid-set (make-labelled-dataset backend name valid-matrix
                                                               valid-labels
                                                               :reference dataset))
                           (let ((condition
                                   (train-freeing valid-set dataset (list valid-set))))
                             (ok condition "train signalled instead of faulting")
                             (ok (and condition
                                      (eq valid-set
                                          (cl-gbdt:released-handle-error-object condition)))
                                 "the condition named the validation set")))))))))
          (cl-gbdt:close-backend backend))))))
