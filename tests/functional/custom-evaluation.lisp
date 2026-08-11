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
;;;; BOTH BACKENDS ANSWER THE CAPABILITY TRUE, out of different lists: LightGBM PROBES it, in
;;;; `cl-gbdt/src/lightgbm/native''s `*optional-symbols*', since the three C functions its read
;;;; needs are optional there; XGBoost DECLARES it, in its own `*provided-capabilities*', since
;;;; the one C function its read needs is required there. Every
;;;; `(when (cl-gbdt:backend-supports-p ...))' guard below therefore runs on both, and the
;;;; guards stay rather than being deleted for the reason they were written: a backend that
;;;; stopped providing the capability must SKIP these tests rather than fail them, and the
;;;; first test's `demonstrated' count is what keeps such a skip from leaving this suite green
;;;; having asserted nothing.
;;;;
;;;; When this file first shipped it covered LightGBM alone, XGBoost declaring the capability
;;;; nowhere and refusing every non-NIL :EVALUATION; the commit that declared it made every
;;;; test here cover that backend with no edit to any test form, exactly as
;;;; tests/functional/custom-objective.lisp's guards did for :OBJECTIVE on the day that backend
;;;; stopped refusing it. What DID have to be edited is *METRIC-TOLERANCES* below, which had
;;;; been a single constant measured on LightGBM only -- see its docstring.
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
  ;; Zero symbols, both of them: they run at load time to register :lightgbm and :xgboost
  ;; with `open-backend' and to define each backend's methods on `cl-gbdt''s generics.
  ;; Without these clauses, package-inferred-system has no edge to those files and
  ;; `(cl-gbdt:open-backend :lightgbm)' below would signal `unknown-backend'. `unified'
  ;; rather than `all' since the Layer 1 split: `all' is Layer 1 alone now, carrying the
  ;; `register-backend' call but not the protocol methods, so `train' below would find no
  ;; applicable method.
  (:import-from #:cl-gbdt/src/lightgbm/unified)
  (:import-from #:cl-gbdt/src/xgboost/unified)
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
AND for the 17-row validation set, while differing from `:raw' by 0.706. Measured again on
XGBoost, one `objective=binary:logistic' booster over the same fixture: 0.0 for both datasets
and 0.756 from `:raw'. The two 0.0s are each within one backend; the 0.706 and the 0.756 are
not compared with one another and neither is a claim about the other library -- policy
section 13.")

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

(defparameter *metric-tolerances*
  '((:lightgbm . 1d-9) (:xgboost . 1d-7))
  "How far each backend's caller-written log loss may sit from that SAME backend's own and
still count as the same number.

Per backend, like *DATASET-PARAMETERS*, *METRIC-PARAMETERS* and *LIBRARY-METRIC-NAMES* above,
and for a reason of the same kind rather than as a convenience: THE TWO LIBRARIES' OWN LOG
LOSS REACHES DIFFERENT FLOATING-POINT PRECISIONS, as the two measurements below show, so no
one number bounds both. This was a single 1d-9 constant while only LightGBM provided the
capability, and the commit that declared it on XGBoost is what found that 1d-9 had been a
LightGBM measurement all along.

The two sums are of the same per-row terms over the same rows, from the identical
`double-float' probabilities -- `train' hands the caller the very buffer the library
evaluated, which the `predict :kind :normal' assertions below hold to exactly 0.0 on both
backends -- but they are computed by different code in different languages, and neither
accumulates in an order the other can be held to. So they agree to the last bits their
arithmetic HAS rather than exactly, and how many bits that is differs:

  - LightGBM's own `binary_logloss' lands within the last bits of DOUBLE of a caller
    reimplementing it in Lisp double arithmetic. Measured over five iterations, the largest
    difference across the whole series was 6.66d-16 at dataset index 0 and 1.11d-16 at index 1
    -- a handful of double ulps at 0.69, which is as near as two independently-ordered double
    summations of the same terms ever get.
  - XGBoost's own `logloss' lands only within SINGLE precision of one. Measured over five
    iterations on this fixture, the largest difference was 1.35d-8, across both datasets.

Each bound is one backend's own two readings of one run, and neither is a claim about the
other backend -- policy section 13. They are not subtracted from, divided by, or ranked against
each other anywhere here; what the two bullets say is which FLOATING-POINT TYPE each library's
own arithmetic reaches, which is a fact about that library and not a comparison of two results.

WHY 1d-7 IS A FLOOR AND NOT A TOLERANCE WIDENED UNTIL THE TEST PASSED. Two things, both
checkable from this repository:

  - Arithmetic. One ulp of `single-float' at 0.69 is 2^-24, about 6d-8. A per-row term of that
    magnitude computed in single precision therefore carries a few times 1d-8 of error, and a
    mean of forty of them cannot be pinned nearer than about 1d-8 unless the roundings happen
    to cancel. 1.35d-8 is exactly that order; 1d-9 is below the arithmetic's own floor and
    could never have been met. 1d-7 is one per-term ulp -- the ceiling of what single precision
    can promise for one of these terms -- rather than a number fitted just above 1.35d-8.
  - Reproduction, which is what converts \"the two numbers disagree\" into \"they cannot agree
    more closely\". A Lisp reimplementation that evaluates each row in `single-float' and
    accumulates in `double' reproduces XGBoost's own published value EXACTLY -- difference
    0.0d0 at every one of five rounds -- off the very array `logloss-evaluation' is handed
    here. So the array is right and the arithmetic is the whole of the gap. `logloss-evaluation'
    below is left computing in double, as any natural Lisp caller would: what this file tests
    is a caller's own ordinary metric, not a bit-for-bit reimplementation of one library's
    arithmetic.

The single-precision ATTRIBUTION -- XGBoost evaluating each row's term as `bst_float' in its
own `EvalRowLogLoss' -- is upstream source that this repository does not vendor and that no
reader here can open, so it is offered as the explanation the two checkable facts above are
consistent with, not as a citation. Nothing below rests on it: the ulp arithmetic and the
exact reproduction stand on their own, and would still justify 1d-7 if the mechanism inside
XGBoost turned out to be some other route to the same single-precision floor.

Headroom in the other direction is what both numbers are chosen for, and 1d-7 still keeps six
orders of magnitude of it: nothing this file exists to catch survives it. A metric that ignored
SCORES entirely -- the control below, which returns a flat 0.5 -- sits about 0.19 away from
each backend's own library series, measured between 0.682 and 0.698 on LightGBM and between
0.681 and 0.696 on XGBoost, each stated for its own backend and neither compared with the
other. A validation set whose bin mappers were never aligned moves the predictions themselves
by 0.136. It is `cl-gbdt/tests/functional/support''s *PREDICTION-TOLERANCE* restated rather
than a new idea, and is not imported from there only because what it bounds is a metric rather
than a prediction.

Nothing here loosens what pins the ARRAY the caller is handed. That is held by the
`predict :kind :normal' assertions in the agreement test below, which are bounded by
`cl-gbdt/tests/functional/support''s own *PREDICTION-TOLERANCE* -- a different constant, in a
different file, that this table does not touch and both backends meet at exactly 0.0.")

(defun metric-tolerance (name)
  "Return backend NAME's own agreement tolerance -- see *METRIC-TOLERANCES*."
  (cdr (assoc name *metric-tolerances*)))

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
reached on EITHER backend -- five iterations over both datasets leave every probability
between 0.431844 and 0.568156 on LightGBM, and between 0.415616 and 0.585161 on XGBoost, each
range its own backend's and neither compared with the other -- so the clamp cannot be what
makes either backend's two series agree; it is there so a later fixture or a longer run cannot
turn this function into one that returns an infinity, which would compare equal to nothing and
unequal to nothing."
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

(defun series-agree-p (left right tolerance)
  "True when LEFT and RIGHT, two series' value vectors, are the same length and differ
nowhere by more than TOLERANCE -- the calling backend's own, from `metric-tolerance'."
  (and (= (length left) (length right))
       (every (lambda (a b) (and (realp a) (realp b) (<= (abs (- a b)) tolerance)))
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
;;; The other two things `%check-custom-evaluation' refuses before any foreign call

;;; The capability is only the first of three checks that function makes, and the other two
;;; are reachable on any backend that HAS the capability -- so they are the two a caller is
;;; far likelier to hit than a missing C symbol. Both were unasserted when this file first
;;; shipped: deleting either branch left the whole suite green, and each failure it prevents
;;; is one the caller would otherwise have had to diagnose from a symptom rather than from a
;;; condition. :RECORD-HISTORY NIL would have recorded nothing at all while still paying for
;;; every metric call, and a non-function would have surfaced as SBCL's own untyped
;;; `type-error' from mid-loop, after the booster existed and one dataset's predictions had
;;; already been read, naming neither the argument nor the backend.
;;;
;;; Both are modelled on the sibling suites that already assert the identical shapes for the
;;; identical reasons -- `early-stopping-with-record-history-nil-signals' in
;;; tests/functional/early-stopping.lisp and
;;; `a-non-function-objective-is-refused-before-any-foreign-call' in
;;; tests/functional/custom-objective.lisp -- and both are guarded on the capability like
;;; everything else here, so Task 3's flip covers XGBoost with no edit.

(deftest evaluation-with-record-history-nil-signals
  ;; The same contradiction :EARLY-STOPPING and :RECORD-HISTORY NIL make: a custom metric's
  ;; whole result is the per-iteration series :RECORD-HISTORY NIL exists not to build, and
  ;; the values would be computed at full cost and then dropped. Accepting the pair and
  ;; quietly recording after all, or accepting it and recording nothing, would each be a
  ;; different answer than the one asked for.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   (testing (format nil "~A: :evaluation with :record-history NIL signals ~
                                         unsupported-argument, naming the argument and the ~
                                         backend" name)
                     (let ((condition
                             (handler-case
                                 (progn (cl-gbdt:free-booster
                                         (cl-gbdt:train
                                          backend dataset :num-rounds 3 :record-history nil
                                          :parameters (cdr (assoc name *metric-parameters*))
                                          :evaluation (logloss-evaluation (vector labels*))))
                                        nil)
                               (cl-gbdt:unsupported-argument (c) c))))
                       (ok condition "train accepted :evaluation with :record-history NIL")
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
                                        (cl-gbdt:unsupported-argument-backend condition)))))))))
          (cl-gbdt:close-backend backend))))))

(defun constant-metric (scores index)
  "An :EVALUATION function of the right arity returning a constant, defined only so the test
below has a SYMBOL naming a real one to hand `train'.

That case is the interesting one of the three: `funcall' accepts a symbol happily, so nothing
downstream would fail, and the run would compute its metric from whatever global definition
that name happened to have at each iteration rather than from what the caller passed. A
number or a string fails somewhere no matter what; a symbol quietly works, wrongly."
  (declare (ignore scores index))
  (values *custom-metric-name* 0.5d0))

(deftest a-non-function-evaluation-is-refused-before-any-foreign-call
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   (dolist (value (list 42 "my_logloss" 'constant-metric))
                     (testing (format nil "~A: train signals unsupported-argument for ~
                                           :evaluation ~S, naming the argument and the ~
                                           backend" name value)
                       (let ((condition
                               (handler-case
                                   (progn (cl-gbdt:free-booster
                                           (cl-gbdt:train
                                            backend dataset :num-rounds 3
                                            :parameters (cdr (assoc name
                                                                    *metric-parameters*))
                                            :evaluation value))
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
                                           condition))))))))))
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
  ;; numbers on each backend -- measured on LightGBM, 0.682 at index 0 against 0.698 at index 1
  ;; after five iterations, and on XGBoost 0.681 against 0.696 after its own five. Each pair is
  ;; that backend's own and the two pairs are not compared; what each says is that index 1's
  ;; agreement is not index 0's agreement in disguise, on the backend it was measured on.
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
                               (ok (and mine theirs
                                        (series-agree-p mine theirs
                                                        (metric-tolerance name)))
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
                             (ok (and mine theirs
                                      (not (series-agree-p mine theirs
                                                           (metric-tolerance name))))
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

;;; What a caller's VALUE becomes once it is in the report, which is a different question
;;; from whether the run finished -- and the run finished before this was fixed too.
;;; `cl-gbdt:training-series-values' documents every element of a series as a `double-float'
;;; or NIL, and both libraries' own values already are doubles; a caller's is the one that
;;; need not be. Measured on a four-round run before `custom-metric-entry' coerced: a metric
;;; returning 1/4 trained to :TRAINED and left element types (RATIO RATIO RATIO RATIO), 0.25
;;; left SINGLE-FLOATs and 3 left INTEGERs -- so a consumer holding to the documented
;;; invariant broke for custom metrics alone. tests/custom-metric.lisp pins the coercion in
;;; `custom-metric-entry' directly; what this asserts is the ELEMENT TYPE of what comes back
;;; out of `train', which is the promise a caller actually reads.

(deftest a-non-double-metric-value-is-recorded-as-a-double-float
  ;; The three numbers below are the CALLER's own constants round-tripped through its own
  ;; report, not either library's, so nothing here compares a number between backends --
  ;; policy section 13. Each backend is asked the same question about its own report and
  ;; answers it out of its own run.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   ;; A RATIO, then a `single-float', then an INTEGER: three types a caller
                   ;; reaches for without thinking about what slot the value lands in. One
                   ;; per iteration, so all three are exercised in one run and the series
                   ;; that comes back is a mixture if any of them survives uncoerced.
                   (let ((returned (list 1/4 0.25 3))
                         (calls 0))
                     (multiple-value-bind (booster report)
                         (cl-gbdt:train backend dataset :num-rounds 3
                                        :parameters (cdr (assoc name *metric-parameters*))
                                        :evaluation
                                        (lambda (scores index)
                                          (declare (ignore scores index))
                                          (let ((value (nth calls returned)))
                                            (incf calls)
                                            (values *custom-metric-name* value))))
                       (cl-gbdt:free-booster booster)
                       (let ((values (series-values report 0 *custom-metric-name*)))
                         (ok (and values (= 3 (length values)))
                             (format nil "~A: the caller's series holds ~S" name values))
                         (ok (and values (every (lambda (value) (typep value 'double-float))
                                                values))
                             (format nil "~A: the series' element types were ~S" name
                                     (and values (map 'list #'type-of values))))
                         ;; And the coercion preserved the numbers, so a caller reads back
                         ;; what it recorded rather than what rounding made of it.
                         (ok (and values (= 3 (length values))
                                  (every #'eql #(0.25d0 0.25d0 3.0d0) values))
                             (format nil "~A: the series read back ~S" name values))))))))
          (cl-gbdt:close-backend backend))))))

;;; ---------------------------------------------------------------------------
;;; :OBJECTIVE and :EVALUATION in the same run

;;; Three docstrings state measured facts about this one configuration, and until this test
;;; existed nothing anywhere ran it -- `src/protocol.lisp' ("On LightGBM `objective=none'
;;; leaves no transform to apply, so :NORMAL and :RAW become the same numbers and SCORES
;;; coincides with what OBJECTIVE was handed"), `cl-gbdt/src/lightgbm/native''s
;;; `%booster-predictions' ("Under `objective=none' ... this buffer agrees with both to 0.0")
;;; and `cl-gbdt/src/xgboost/native''s ("an :EVALUATION and an :OBJECTIVE in the same run are
;;; handed different numbers because they asked for different ones"). Each is a claim about
;;; what the two caller-supplied functions SEE, and each backend's answer is the opposite of
;;; the other's, which is why the expectation is a per-backend table rather than one assertion.
;;;
;;; WHAT IS COMPARED, and why it is comparable at all. Within one iteration the two functions
;;; are handed two different model states: :OBJECTIVE reads the scores BEFORE that iteration's
;;; update, :EVALUATION after it. So the array to compare :EVALUATION's iteration R against is
;;; :OBJECTIVE's iteration R+1 -- the same booster, after the same R updates, read once
;;; through each path. Comparing them within an iteration would fail on both backends for a
;;; reason that has nothing to do with either claim.
;;;
;;; Both comparisons are one backend's two readings of one run and neither is a claim about
;;; the other backend -- policy section 13. Nothing here subtracts a LightGBM number from an
;;; XGBoost one; what the table records is which of two behaviours each backend has.

(defparameter *scores-under-a-custom-objective*
  '((:lightgbm . t) (:xgboost . nil))
  "Whether :EVALUATION's SCORES and :OBJECTIVE's are the SAME numbers on each backend, for one
booster after the same number of updates.

TRUE on LightGBM because `train' forces `objective=none' there -- `LGBM_BoosterUpdateOneIterCustom'
refuses to run while the booster holds an objective function at all -- and with no objective
there is no prediction transform, so `predict :kind :normal' and `:kind :raw' become the same
numbers and the one buffer `LGBM_BoosterGetPredict' holds serves both readers.

FALSE on XGBoost because that backend rewrites no parameter: the `binary:logistic' configured
in *METRIC-PARAMETERS* goes on transforming, so :EVALUATION's `:normal' read is a probability
and :OBJECTIVE's `:raw' read is the margin behind it, in the same run and off the same trees.
That is the divergence `cl-gbdt/src/xgboost/native''s `%booster-predictions' and the README's
Custom objective section both already describe as the price of not rewriting PARAMETERS; this
table is where it is finally executed rather than only stated.

The FALSE arm is the weaker assertion of the two and is deliberately not tightened into a
figure: what it says is that the two arrays are not the same numbers, which is exactly the
claim. A specific gap would be a fact about how far one fixture's probabilities sit from their
own logits after five iterations, and would have to be re-measured for any change to the
fixture without saying anything more about :EVALUATION.")

(defun logistic-objective (labels*)
  "Return an :OBJECTIVE computing binary logistic gradient and Hessian from the MARGINS it is
handed: `grad = sigmoid(s) - y', `hess = p(1-p)'.

A real objective rather than a constant, so the model actually moves over the run and the two
score readings this test compares are five different pairs of arrays rather than five copies
of an untrained one. Reads its own shape off SCORES, so it returns what
`check-objective-result' demands on either backend."
  (lambda (scores)
    (let* ((rows (array-dimension scores 0))
           (grad (make-array (list rows 1) :element-type 'double-float))
           (hess (make-array (list rows 1) :element-type 'double-float)))
      (dotimes (row rows (values grad hess))
        (let ((p (/ 1d0 (+ 1d0 (exp (- (aref scores row 0)))))))
          (setf (aref grad row 0) (- p (aref labels* row))
                (aref hess row 0) (max (* p (- 1d0 p)) 1d-16)))))))

(deftest an-objective-and-a-metric-in-one-run-see-what-each-was-promised
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (and (cl-gbdt:backend-supports-p backend :custom-evaluation)
                        (cl-gbdt:backend-supports-p backend :custom-objective))
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   (let ((objective-scores '())
                         (evaluation-scores '())
                         (rounds 5))
                     (multiple-value-bind (booster report)
                         (cl-gbdt:train
                          backend dataset :num-rounds rounds
                          :parameters (cdr (assoc name *metric-parameters*))
                          :objective (let ((inner (logistic-objective labels*)))
                                       (lambda (scores)
                                         (push scores objective-scores)
                                         (funcall inner scores)))
                          :evaluation (lambda (scores index)
                                        (when (zerop index)
                                          (push scores evaluation-scores))
                                        (values *custom-metric-name* 0.5d0)))
                       (cl-gbdt:free-booster booster)
                       (setf objective-scores (nreverse objective-scores)
                             evaluation-scores (nreverse evaluation-scores))
                       ;; Both arguments were honoured in the one call, which is the whole of
                       ;; what was never executed before: the objective ran every iteration
                       ;; and the metric recorded a full series beside it.
                       (ok (= rounds (length objective-scores))
                           (format nil "the objective was called ~D times"
                                   (length objective-scores)))
                       (let ((values (series-values report 0 *custom-metric-name*)))
                         (ok (and values (= rounds (length values)))
                             "the caller's series is as long as the run"))
                       ;; :EVALUATION's iteration R against :OBJECTIVE's iteration R+1 --
                       ;; the same booster after the same R updates, read once through each
                       ;; path. Four such pairs over five rounds.
                       (let ((coincide (cdr (assoc name *scores-under-a-custom-objective*)))
                             (agreements 0)
                             (pairs 0))
                         (loop :for after-update :in evaluation-scores
                               :for before-next-update :in (rest objective-scores)
                               :do (incf pairs)
                                   (when (support:predictions-agree-p after-update
                                                                      before-next-update)
                                     (incf agreements)))
                         (ok (= 4 pairs) (format nil "~D comparable pairs" pairs))
                         (ok (= (if coincide pairs 0) agreements)
                             (format nil
                                     "~A: ~D of ~D pairs agreed, expected ~D -- ~
                                      :evaluation's SCORES and :objective's are ~:[not ~;~]~
                                      the same numbers on this backend"
                                     name agreements pairs (if coincide pairs 0)
                                     coincide))))))))
          (cl-gbdt:close-backend backend))))))

;;; ---------------------------------------------------------------------------
;;; What the recorded values are good for

(defun stalling-evaluation (name)
  "Return an :EVALUATION function whose value improves for three calls, is NIL on the fourth
and then stalls: 3.0, 2.0, 1.0, NIL, 1.0, 1.0, ... under NAME, ignoring the predictions
entirely.

Ignoring them is the point. What is under test is that a caller's metric reaches
`:early-stopping' at all, and a value derived from a real model would make WHEN the run
stopped a fact about the model rather than about the plumbing. Counting calls is counting
ITERATIONS here only because the test that uses this passes no :VALID-SETS, so there is
exactly one dataset and therefore one call per iteration.

THE NIL AT CALL 4 is the one place in this suite a caller's own \"not computable this
iteration\" travels the whole way -- through `custom-metric-entry', into the history, into a
series slot, and past the watcher -- which `train''s generic docstring promises twice, once
for the series and once for the watcher, and which tests/custom-metric.lisp could only pin at
the entry it builds. It is placed at 4 rather than anywhere else so the run's shape does not
change: NIL counts as NO IMPROVEMENT, exactly as the 1.0 at call 4 used to, so the test below
still stops at iteration 5 with iteration 3 best and 1.0 the best score. What it adds is that
the fourth slot of the series is NIL and the run still ended for the right reason -- if a NIL
were dropped instead of recorded, the series would be one short; if it were admitted as a
best score, the run would not have stopped where it does."
  (let ((calls 0))
    (lambda (scores index)
      (declare (ignore scores index))
      (incf calls)
      (values name (unless (= calls 4)
                     (coerce (max 1 (- 4 calls)) 'double-float))))))

(deftest early-stopping-can-watch-a-custom-metric
  ;; A custom metric is recorded into the SAME per-iteration entry list the watcher reads,
  ;; before the watcher runs -- which is the whole reason `:early-stopping' needs nothing
  ;; added to it to watch one. The booster is trained with every library metric turned off,
  ;; so the series the watcher finds cannot be a library series the run happened to also
  ;; report.
  ;;
  ;; :ROUNDS 2 against a value that improves at iterations 1, 2 and 3, is NIL at 4 and holds
  ;; at 5: iteration 4 is the first non-improvement -- a NIL cannot be compared against the
  ;; best score and so counts as none -- iteration 5 the second, since improvement is strict
  ;; and a plateau does not count either, and two consecutive is what :ROUNDS 2 tolerates. So
  ;; the run stops after 5 of the 10 asked for, with iteration 3 the best.
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
                     ;; as long as the run, and the caller's series is no exception. And the
                     ;; NIL the metric returned at iteration 4 is IN ITS OWN SLOT rather than
                     ;; dropped -- a drop would leave four values and slide the 1.0 measured
                     ;; at iteration 5 back onto iteration 4's position.
                     (let ((values (series-values report 0 *custom-metric-name*)))
                       (ok (= 5 (length values)))
                       (ok (and (= 5 (length values)) (null (aref values 3)))
                           (format nil "the recorded series was ~S" values))
                       (ok (and (= 5 (length values))
                                (every #'realp (list (aref values 0) (aref values 1)
                                                     (aref values 2) (aref values 4))))
                           "every value but the NIL one came back a real"))))))
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
;;; The ways a caller's metric can be refused mid-run

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

;;; The malformed RETURN, which is the refusal a caller is likeliest of all to meet and the
;;; only one this file had left unasserted. `train''s generic docstring and the README both
;;; promise that a NAME that is not a string, or a VALUE that is neither a real nor NIL,
;;; signals `unsupported-argument' naming :EVALUATION -- and until this test existed, nothing
;;; anywhere returned a bad one THROUGH `train'. tests/custom-metric.lisp pins
;;; `custom-metric-entry' in isolation; what was missing was any evidence that `train' calls
;;; it. Replacing either backend's `(custom-metric-entry (backend-name backend) name value
;;; index)' with a bare `(list index name value)' left the whole suite green, taking the
;;; validation AND the `backend-name' threading with it -- the same shape as the
;;; agreement-test gap this branch already caught.
;;;
;;; It is also the only refusal here that fires MID-RUN with a booster handle already in
;;; existence, so it is what exercises `train''s OWNED unwind with a TYPED condition rather
;;; than with the `simple-error' `an-error-inside-the-metric-frees-the-booster' below raises.
;;; Both malformed shapes are covered rather than one: the NAME check and the VALUE check are
;;; two separate branches of `custom-metric-entry', and a bad NAME short-circuits before the
;;; value is ever looked at, so one case cannot stand in for the other.

(deftest a-malformed-metric-return-is-refused-mid-run
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   ;; A keyword NAME, then a string VALUE: the two things `custom-metric-entry'
                   ;; refuses, each in the shape a caller actually writes them.
                   (dolist (returned (list (cons :my-metric 0.25d0)
                                           (cons *custom-metric-name* "0.25")))
                     (testing (format nil "~A: :evaluation returning ~S and ~S signals ~
                                           unsupported-argument, naming the argument and the ~
                                           backend" name (car returned) (cdr returned))
                       (let ((condition
                               (handler-case
                                   (progn (cl-gbdt:free-booster
                                           (cl-gbdt:train
                                            backend dataset :num-rounds 3
                                            :parameters (cdr (assoc name *metric-parameters*))
                                            :evaluation
                                            (lambda (scores index)
                                              (declare (ignore scores index))
                                              (values (car returned) (cdr returned)))))
                                          nil)
                                 (cl-gbdt:unsupported-argument (c) c))))
                         (ok condition
                             "train recorded the malformed return instead of signalling")
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
                                           condition)))))))
                   ;; The backend is still usable afterwards, which is the observable half of
                   ;; "the raw booster was freed by the OWNED unwind rather than orphaned"
                   ;; for a typed condition raised from inside this library's own code.
                   (multiple-value-bind (booster report)
                       (cl-gbdt:train backend dataset :num-rounds 2
                                      :parameters (cdr (assoc name *metric-parameters*)))
                     (cl-gbdt:free-booster booster)
                     (ok (series-values report 0 (library-metric name))
                         "the backend still trains after the mid-run refusal")))))
          (cl-gbdt:close-backend backend))))))

;;; The NAME that CHANGES, which is the one no single iteration can see anything wrong with:
;;; "safe" on iteration 1 and something else on iteration 2 are each a perfectly good name on
;;; the iteration they appear in. `check-metric-name-collision' runs on the first iteration
;;; only -- the first moment there is a real evaluation to compare against -- so a name that
;;; changes INTO the library's own walks straight past it and puts two entries under one
;;; (INDEX, NAME) key on every later iteration, giving a series of `1 + 2(N-1)' values over N
;;; rounds: longer than `training-report-num-rounds', misaligned with the iterations, and
;;; silent. A name that changes without ever colliding is the ragged case instead, several
;;; series each shorter than the run.
;;;
;;; `pin-metric-name' (`cl-gbdt/src/training/custom-metric') is what refuses both, by holding
;;; each dataset index to the name its FIRST call returned. tests/custom-metric.lisp pins that
;;; function directly, over written call sequences; this is what says `train' actually calls
;;; it, and it is asserted on both halves for the reason the malformed-return test above gives
;;; for asserting both of its own.

(deftest a-metric-name-that-changes-mid-run-is-refused
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   ;; The second name each way round: one that collides with the library's
                   ;; own -- the case the first-iteration collision check cannot reach -- and
                   ;; one that collides with nothing at all.
                   (dolist (second-name (list (library-metric name) "another_metric"))
                     (testing (format nil "~A: a metric renamed to ~S at iteration 2 signals"
                                      name second-name)
                       (let* ((calls 0)
                              (condition
                                (handler-case
                                    (progn (cl-gbdt:free-booster
                                            (cl-gbdt:train
                                             backend dataset :num-rounds 3
                                             :parameters (cdr (assoc name
                                                                     *metric-parameters*))
                                             :evaluation
                                             (lambda (scores index)
                                               (declare (ignore scores index))
                                               (incf calls)
                                               (values (if (= calls 1)
                                                           *custom-metric-name*
                                                           second-name)
                                                       0.5d0))))
                                           nil)
                                  (cl-gbdt:unsupported-argument (c) c))))
                         (ok condition
                             "train recorded the renamed metric instead of signalling")
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
                                          (cl-gbdt:unsupported-argument-backend condition))))
                         ;; Refused where the differing name was returned -- the SECOND
                         ;; iteration -- and not at the end of the run. There is one dataset
                         ;; here, so a call is an iteration.
                         (ok (= 2 calls)
                             (format nil "the metric was called ~D times before the refusal"
                                     calls)))))
                   ;; And a name that never changes still runs to completion, so the pin is
                   ;; not an unconditional refusal of the second iteration.
                   (multiple-value-bind (booster report)
                       (cl-gbdt:train backend dataset :num-rounds 3
                                      :parameters (cdr (assoc name *metric-parameters*))
                                      :evaluation (lambda (scores index)
                                                    (declare (ignore scores index))
                                                    (values *custom-metric-name* 0.5d0)))
                     (cl-gbdt:free-booster booster)
                     (let ((values (series-values report 0 *custom-metric-name*)))
                       (ok (and values (= 3 (length values)))
                           "an unchanging name still records one value per iteration"))))))
          (cl-gbdt:close-backend backend))))))

;;; The same NAME OBJECT, rewritten in place, which is the way past the pin the test above
;;; cannot reach. A caller that returns a fresh string each iteration is what that one
;;; models; a caller that keeps one buffer and refills it is just as ordinary, and before
;;; `custom-metric-entry' copied the name it defeated everything at once: `pin-metric-name'
;;; held the caller's own object, so `string=' compared it with itself and saw no change,
;;; and EVERY history entry held that same object, so `training-report-from-history' -- which
;;; runs once, after the loop -- folded all of them under whatever the name said by then.
;;;
;;; Measured before the copy, four rounds on LightGBM with `metric "binary_logloss"' and a
;;; 14-character name rewritten to "binary_logloss" from the second iteration: `train'
;;; returned :TRAINED, nothing signalled, and the report held ONE EIGHT-VALUE SERIES for a
;;; four-round run -- the library's four values and the caller's four braided under one key,
;;; misaligned with the iterations. That is exactly the corruption the pin was added to
;;; prevent, reached around it.
;;;
;;; tests/custom-metric.lisp asserts the copy and the pin over a written call sequence; this
;;; is what says `train' end to end is no longer fooled, which is the whole point -- it was
;;; `train' that was fooled.

(defun rewrite-name (name string)
  "Rewrite NAME IN PLACE to hold STRING, and return NAME -- the same object, different
contents. This is the whole of what a caller reusing one name buffer does between two
iterations, and the whole of what the test below does to it."
  (setf (fill-pointer name) 0)
  (loop :for character :across string :do (vector-push-extend character name))
  name)

(defun rewritable-name (initial)
  "Return a fresh adjustable string with a fill pointer, holding INITIAL -- one string object
`rewrite-name' above can go on rewriting.

Adjustable with a fill pointer rather than a plain `(simple-array character (N))' so a
rewrite can change the name's LENGTH as well as its characters: the two backends' own metric
names are 14 and 7 characters long, and a same-length `replace' would need a different
starting name per backend for a reason this test does not care about. It is an ordinary
string as far as everything under test is concerned -- `stringp' is true of it, `string='
reads it, and `copy-seq' copies exactly its active characters."
  (rewrite-name (make-array 16 :element-type 'character :adjustable t :fill-pointer 0)
                initial))

(deftest a-metric-name-object-rewritten-in-place-is-refused
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-evaluation)
               (multiple-value-bind (matrix labels*) (binary-fixture *rows* 0)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   (testing (format nil "~A: one name object rewritten into ~S at iteration ~
                                         2 signals" name (library-metric name))
                     (let* ((metric-name (rewritable-name *custom-metric-name*))
                            (calls 0)
                            (condition
                              (handler-case
                                  (progn (cl-gbdt:free-booster
                                          (cl-gbdt:train
                                           backend dataset :num-rounds 4
                                           :parameters (cdr (assoc name *metric-parameters*))
                                           :evaluation
                                           (lambda (scores index)
                                             (declare (ignore scores index))
                                             (incf calls)
                                             ;; Rewritten at the START of the second call, so
                                             ;; the FIRST returns a safe name and passes the
                                             ;; first-iteration collision check -- the whole
                                             ;; reason this is not that check's business.
                                             ;; The object returned never changes.
                                             (when (= calls 2)
                                               (rewrite-name metric-name
                                                             (library-metric name)))
                                             (values metric-name 0.5d0))))
                                         nil)
                                (cl-gbdt:unsupported-argument (c) c))))
                       ;; The rewrite really happened, so a green assertion below cannot be
                       ;; one about a name that never changed.
                       (ok (string= (library-metric name) metric-name)
                           (format nil "the name object reads ~S" (copy-seq metric-name)))
                       (ok condition
                           "train recorded the rewritten name instead of signalling")
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
                                        (cl-gbdt:unsupported-argument-backend condition))))
                       ;; Refused at the second iteration, where the rewrite happened, and
                       ;; not at the end of the run. One dataset here, so a call is an
                       ;; iteration.
                       (ok (= 2 calls)
                           (format nil "the metric was called ~D times before the refusal"
                                   calls))))
                   ;; The control: one object reused and rewritten to the SAME contents every
                   ;; iteration is accepted and records one value per iteration, so what is
                   ;; refused above is the CHANGE and not the reuse.
                   (testing (format nil "~A: one name object rewritten to the same contents ~
                                         is accepted" name)
                     (let ((metric-name (rewritable-name *custom-metric-name*)))
                       (multiple-value-bind (booster report)
                           (cl-gbdt:train backend dataset :num-rounds 4
                                          :parameters (cdr (assoc name *metric-parameters*))
                                          :evaluation
                                          (lambda (scores index)
                                            (declare (ignore scores index))
                                            (rewrite-name metric-name *custom-metric-name*)
                                            (values metric-name 0.5d0)))
                         (cl-gbdt:free-booster booster)
                         (let ((values (series-values report 0 *custom-metric-name*)))
                           (ok (and values (= 4 (length values)))
                               (format nil "the caller's series holds ~S" values))
                           ;; And exactly the shape the measurement above found broken: four
                           ;; rounds, and no series longer than the run.
                           (ok (= 4 (cl-gbdt:training-report-num-rounds report)))
                           (ok (every (lambda (series)
                                        (= 4 (length (cl-gbdt:training-series-values
                                                      series))))
                                      (cl-gbdt:training-report-series report))
                               (format nil "series lengths were ~S"
                                       (mapcar (lambda (series)
                                                 (length (cl-gbdt:training-series-values
                                                          series)))
                                               (cl-gbdt:training-report-series
                                                report)))))))))))
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
