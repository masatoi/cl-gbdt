;;;; custom-objective.lisp --- Portable contract tests for `train''s :OBJECTIVE argument.
;;;;
;;;; `cl-gbdt:train' now accepts an :OBJECTIVE function: called once per iteration with the
;;;; booster's current raw scores, returning the gradient and the Hessian that iteration is
;;;; to be built from. The run then boosts against the caller's own loss rather than one
;;;; built into the library.
;;;;
;;;; The capability is `:custom-objective', and `train' re-checks it -- a backend that
;;;; answers false signals `cl-gbdt:capability-unavailable' for a non-NIL :OBJECTIVE rather
;;;; than quietly training against the library's own objective, which is the silent fallback
;;;; policy section 7 forbids. That is what makes the loop in the first test below an
;;;; asked/demonstrated pair rather than a single-branch test: every backend is asked, one
;;;; branch trains and the other catches the refusal, and the count after the loop is what
;;;; keeps a suite in which NO backend provides the capability from passing having asserted
;;;; nothing.
;;;;
;;;; Both vendored libraries answer that capability TRUE, so that pair's refusal branch is
;;;; unreachable here and the gate would be a branch nobody had ever seen taken. Which is why
;;;; `custom-objective-without-the-capability-signals' below withdraws the capability from an
;;;; open backend and watches `train' refuse -- the same way
;;;; tests/functional/sparse-input.lisp and tests/functional/missing-value.lisp reach their
;;;; own gates. Deleting either backend's `%check-custom-objective' call left the whole suite
;;;; green before that test existed.
;;;;
;;;; Every other test here is guarded on the same capability and simply does nothing on a
;;;; backend that answers false. That is deliberate and is why the guards are written this
;;;; way: a backend gaining the capability starts asserting through these same tests with no
;;;; edit to this file.
;;;;
;;;; Numbers are never compared BETWEEN backends -- policy section 13. Every comparison below
;;;; is one backend's custom-objective run against that SAME backend's built-in run, on the
;;;; same fixture, which is the only comparison that means anything: the two libraries build
;;;; different trees from the same rows by design, and squared error is defined identically
;;;; by both, so "did the caller's numbers actually reach the trees" is answerable within one
;;;; backend and nowhere else.
;;;;
;;;; The fixture is deterministic rather than random, and stated here rather than taken from
;;;; tests/functional/evaluation.lisp's *FIXTURES*: a custom objective's whole claim is that
;;;; it REPRODUCES a built-in one to within `predictions-agree-p''s tolerance, and a random
;;;; fixture makes a near-miss indistinguishable from noise.

(uiop:define-package #:cl-gbdt/tests/functional/custom-objective
  ;; Zero symbols: every reference below is package-qualified. Declared so this file's
  ;; dependency on the unified API is explicit rather than inherited, matching the identical
  ;; clause in evaluation.lisp, sparse-input.lisp, missing-value.lisp,
  ;; categorical-features.lisp and prediction-shape.lisp.
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
  ;; A nickname rather than an `:import-from' naming the two symbols, unlike the neighbouring
  ;; files: `predictions-agree-p' is the assertion every comparison below is made through, and
  ;; the prefix is what keeps "this is the suite's shared tolerance, not this file's own idea
  ;; of agreement" visible at each of those call sites.
  (:local-nicknames (#:support #:cl-gbdt/tests/functional/support)))

(in-package #:cl-gbdt/tests/functional/custom-objective)

(defparameter *rows* 40
  "Rows in the single-output-group fixture below.")

(defparameter *columns* 3)

(defun regression-fixture ()
  "A deterministic regression fixture: a (ROWS COLUMNS) matrix and a double-float label
vector. Deterministic because a custom objective's whole claim is that it reproduces a
built-in one, and a random fixture makes a near-miss indistinguishable from noise."
  (let ((matrix (make-array (list *rows* *columns*) :element-type 'double-float))
        (labels* (make-array *rows* :element-type 'double-float)))
    (dotimes (row *rows*)
      (dotimes (column *columns*)
        (setf (aref matrix row column)
              (coerce (mod (+ (* 3 row) (* 7 column)) 11) 'double-float)))
      (setf (aref labels* row)
            (+ (* 0.5d0 (aref matrix row 0)) (- (aref matrix row 1)) (mod row 3))))
    (values matrix labels*)))

(defun squared-error-objective (labels*)
  "Return an objective function computing squared error's gradient and Hessian.

Both libraries define L2 as `grad = p - y', `hess = 1', so a run driven by this must
reproduce the same library's built-in regression objective. That is what makes it a test of
whether the caller's numbers actually reached the trees, rather than of whether the call
returned zero -- five update calls returning 0 while the model did not train is a failure
this project has already seen."
  (lambda (scores)
    (let ((rows (array-dimension scores 0)))
      (let ((grad (make-array (list rows 1) :element-type 'double-float))
            (hess (make-array (list rows 1) :element-type 'double-float
                                            :initial-element 1d0)))
        (dotimes (row rows (values grad hess))
          (setf (aref grad row 0) (- (aref scores row 0) (aref labels* row))))))))

(defparameter *multiclass-rows* 30
  "Rows in the three-output-group fixture below.")

(defparameter *multiclass-columns* 4)

(defparameter *num-classes* 3
  "How many output groups the multiclass fixture's model has.

Must equal the `:num-class' in both entries of *MULTICLASS-PARAMETERS*, which spell it as a
literal in each library's own vocabulary; nothing ties the three together but this sentence,
exactly as in `cl-gbdt/tests/functional/prediction-shape''s own multiclass fixture.

Three rather than two because the layout test below has to tell group-major from row-major,
and at TWO groups the two orderings still coincide often enough to be worth avoiding: three
is the smallest count at which a gradient confined to group 0 lands somewhere unmistakable
under each.")

(defun multiclass-fixture ()
  "A deterministic multiclass fixture: a (MULTICLASS-ROWS MULTICLASS-COLUMNS) matrix and a
`double-float' label vector of class ordinals 0, 1, 2.

The labels are never read by the objective the layout test drives -- that objective is a
function of the row index alone -- but LightGBM refuses to build a dataset with no label at
all, and a label of the wrong range would be the kind of thing to trip a later change. So
they are real class ordinals rather than filler."
  (let ((matrix (make-array (list *multiclass-rows* *multiclass-columns*)
                            :element-type 'double-float))
        (labels* (make-array *multiclass-rows* :element-type 'double-float)))
    (dotimes (row *multiclass-rows*)
      (dotimes (column *multiclass-columns*)
        (setf (aref matrix row column)
              (coerce (mod (+ (* 5 row) (* 3 column)) 13) 'double-float)))
      (setf (aref labels* row) (coerce (mod row *num-classes*) 'double-float)))
    (values matrix labels*)))

(defun first-group-only-objective ()
  "Return an objective whose gradient is non-zero in output group 0 and exactly zero in every
other group, with a Hessian of 1 everywhere.

The gradient ALTERNATES by row -- -1 on even rows, +1 on odd -- rather than holding a
constant, and that is the whole of what makes the layout test below able to fail. A constant
gradient gives every candidate split zero gain, LightGBM then produces an empty tree, and
group 0 does not move either: the correct layout would look exactly like the broken one. The
planning probe for this feature made that mistake first and had to be rewritten.

Reads GROUPS off SCORES rather than off *NUM-CLASSES*, so the array it returns is the shape
`check-objective-result' demands whatever the booster turns out to have -- and so a backend
that handed the objective the wrong group count would fail on the shape assertion below
rather than here."
  (lambda (scores)
    (let* ((rows (array-dimension scores 0))
           (groups (array-dimension scores 1))
           (grad (make-array (list rows groups) :element-type 'double-float
                                                :initial-element 0d0))
           (hess (make-array (list rows groups) :element-type 'double-float
                                                :initial-element 1d0)))
      (dotimes (row rows (values grad hess))
        (setf (aref grad row 0) (if (evenp row) -1d0 1d0))))))

(defun column-spread (result column)
  "Return how far COLUMN of RESULT, a `cl-gbdt:predict' result, varies across its rows --
its largest element minus its smallest.

A within-backend measure by construction, which is what policy section 13 requires of it: a
group's absolute score level is not comparable across the two libraries, carrying
`base_score' on XGBoost against 0 on LightGBM, but whether that group's scores VARY from row
to row is the same question on both. Zero means every row of that output group got the same
score, which for a one-iteration run means the group did not move."
  (loop :for row :below (array-dimension result 0)
        :maximize (aref result row column) :into high
        :minimize (aref result row column) :into low
        :finally (return (- high low))))

(defparameter *dataset-parameters*
  '((:lightgbm :min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)
    (:xgboost))
  "What each backend's `make-dataset' needs for the two fixtures above, and load-bearing
rather than defensive on LightGBM: with its defaults that library bins every column into one
bin apiece and warns \"There are no meaningful features which satisfy the provided
configuration\", after which every tree is a root-only leaf. Both regression runs then produce
a CONSTANT, and the built-in one's constant is not the custom one's -- measured, before these
parameters were passed: -1.5875 against 0.0 on every row. So the reproduction test below would
have failed for a reason that has nothing to do with custom objectives, and the layout test
would have had no movement in any group to tell one ordering from the other by.

XGBoost's is empty because that backend signals `unsupported-argument' for a non-NIL
:PARAMETERS on `make-dataset' at all -- the same split
`cl-gbdt/tests/functional/evaluation''s *FIXTURES* records for the same reason. NIL is
accepted there and means the same nothing it means here.")

(defun make-labelled-dataset (backend name matrix labels* &key reference)
  "Build a dataset on BACKEND, named NAME, from MATRIX and LABELS*, passing only the
`make-dataset' :PARAMETERS that backend accepts -- see *DATASET-PARAMETERS*. One call site
works for both backends and for both fixtures, which is what lets every test below be
written once.

REFERENCE, when supplied, is the dataset this one's bin mappers must align with -- what
`LGBM_BoosterAddValidData' requires of a validation set. It reaches `make-dataset' on
LightGBM only: XGBoost has no bin-mapper alignment and signals `unsupported-argument' for a
non-NIL :REFERENCE, so passing it there would fail the call rather than be ignored. The same
split, for the same reason, is what `cl-gbdt/tests/functional/evaluation''s
`make-fixture-dataset' spells as its fixtures' :ALIGNS-BIN-MAPPERS key."
  (apply #'cl-gbdt:make-dataset backend matrix :label labels*
         :parameters (cdr (assoc name *dataset-parameters*))
         (when (and reference (eq name :lightgbm)) (list :reference reference))))

(defparameter *built-in-regression-parameters*
  '((:lightgbm :objective "regression" :num-leaves 7 :min-data-in-leaf 1
     :min-data-in-bin 1 :verbose -1 :boost-from-average nil)
    (:xgboost :objective "reg:squarederror" :max-depth 3 :eta 0.3d0 :verbosity 0
     :base-score 0.0d0))
  "Each backend's own spelling of \"plain squared error, nothing clever\". `boost-from-average'
is off on LightGBM and `base_score' is 0 on XGBoost so that both runs start from the same
place their custom counterparts do, which is what lets the two be compared at all.")

(defparameter *custom-regression-parameters*
  '((:lightgbm :num-leaves 7 :min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1
     :boost-from-average nil)
    (:xgboost :max-depth 3 :eta 0.3d0 :verbosity 0 :base-score 0.0d0))
  "The same, minus the objective. LightGBM's is forced to \"none\" by `train' itself; leaving
it out here is what lets the LightGBM override test below put one back and watch it lose.")

(defparameter *multiclass-parameters*
  '((:lightgbm :num-class 3 :num-leaves 7 :learning-rate 1.0d0 :min-data-in-leaf 1
     :min-data-in-bin 1 :verbose -1 :boost-from-average nil)
    (:xgboost :objective "multi:softprob" :num-class 3 :max-depth 3 :eta 1.0d0
     :verbosity 0 :base-score 0.0d0))
  "Each backend's own way of asking for THREE output groups. The literal 3 in each must equal
*NUM-CLASSES*; see that variable, which says so and is the only thing that says so.

`num_class' is what supplies the group count on LightGBM even under the `objective=none' that
`train' forces, which is why the LightGBM row names no objective at all -- exactly as
*CUSTOM-REGRESSION-PARAMETERS* does, and for the same reason.

XGBoost's row DOES name one, and NOT because the group count needs it: `num_class 3' alone
gives that backend three output groups under a custom objective, which is what
`cl-gbdt/src/xgboost/protocol''s `train' records and what was re-measured before this
sentence was written -- dropping `multi:softprob' from this row leaves the scores shape, the
`:raw' shape and all three groups' spreads bit-identical, and `multi:softmax' in its place
does the same. It is named to keep an objective FUNCTION configured on the booster while the
custom update runs, which is the difference between the two libraries this file would
otherwise not exercise anywhere: `LGBM_BoosterUpdateOneIterCustom' refuses to run at all in
that state -- `Check failed: objective_function_ == nullptr' -- while
`XGBoosterTrainOneIter' accepts it, which is why `train' rewrites the parameter on one
backend and passes it through on the other.

`learning_rate'/`eta' is 1.0 rather than either library's default so that one iteration moves
the scores by the leaf value itself. The layout test below runs a single iteration and asks
which groups moved; at the default 0.1 the answer is the same but ten times smaller, which
buys nothing and makes the measured figures harder to recognise.")

(defparameter *quiet-group-tolerance* 1d-9
  "How far an output group's scores may vary across rows and still count as not having moved.

The layout test below asserts that the two groups whose gradient was exactly zero did not
move, and this is what \"did not move\" means numerically. It is not a fudge factor: measured,
those groups' spread is exactly 0.0 under the correct layout and 0.25 under the broken one --
see that test's own comment for both figures -- so any threshold between the two would do and
this one is the suite's own `*prediction-tolerance*' restated rather than a new idea.")

(deftest custom-objective-capability-is-true-where-it-is-demonstrated
  ;; The capability is asserted in the same test that trains with a custom objective, and the
  ;; count after the loop is what keeps a backend that quietly stopped providing it from
  ;; leaving this suite green having asserted nothing. Two capabilities have shipped false on
  ;; this project.
  (let ((demonstrated 0))
    (dolist (name '(:lightgbm :xgboost))
      (support:with-backend-library (name)
        (let ((backend (cl-gbdt:open-backend name)))
          (unwind-protect
               (multiple-value-bind (matrix labels*) (regression-fixture)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                        matrix labels*))
                   (if (cl-gbdt:backend-supports-p backend :custom-objective)
                       (progn
                         (incf demonstrated)
                         (cl-gbdt:with-booster
                             (booster (cl-gbdt:train
                                       backend dataset :num-rounds 4
                                       :parameters (cdr (assoc name
                                                               *custom-regression-parameters*))
                                       :objective (squared-error-objective labels*)))
                           (ok (arrayp (cl-gbdt:predict booster matrix :kind :raw)))))
                       ;; All three assertions, not just "something signalled" -- the same
                       ;; three `custom-objective-without-the-capability-signals' below makes,
                       ;; since a refusal naming the wrong capability or the wrong backend is
                       ;; a refusal a caller cannot act on. Both vendored libraries answer the
                       ;; capability true, so nothing reaches this branch today; it is what a
                       ;; library missing one of the C symbols would take, and the test below
                       ;; is what actually watches the gate fire.
                       (let ((condition
                               (handler-case
                                   (progn (cl-gbdt:train
                                           backend dataset :num-rounds 1
                                           :objective (squared-error-objective labels*))
                                          nil)
                                 (cl-gbdt:capability-unavailable (c) c))))
                         (ok condition "train signalled instead of training")
                         (ok (and condition
                                  (eq :custom-objective
                                      (cl-gbdt:capability-unavailable-capability condition))))
                         (ok (and condition
                                  (eq name (cl-gbdt:backend-error-backend condition))))))))
            (cl-gbdt:close-backend backend)))))
    (ok (plusp demonstrated))))

;;; Policy section 7's central rule, watched rather than assumed: `train' re-checks the
;;; capability itself instead of trusting the caller to have asked `backend-supports-p'
;;; first, and signals rather than quietly boosting against the library's own objective.
;;; Both vendored libraries provide the capability -- the test above asserts exactly that --
;;; so the only way to reach the gate is to overwrite the probed plist, which is what a
;;; LightGBM missing one of its four C symbols, or an XGBoost with no
;;; `XGBoosterTrainOneIter', would have produced at `open-backend'. This is the same way
;;; `cl-gbdt/tests/functional/sparse-input''s
;;; `sparse-input-without-the-capability-signals' and
;;; `cl-gbdt/tests/functional/missing-value''s `missing-value-without-the-capability-signals'
;;; reach their own gates. Without it the whole gate is a branch nobody has seen taken:
;;; deleting either backend's `%check-custom-objective' call left the entire suite green
;;; before this test existed, and fails it now.
;;;
;;; `handler-case', not rove's `signals', which does not reliably catch a condition raised
;;; inside `restart-case'; the condition TYPE is asserted, not merely that something
;;; signalled, and so are the capability and the backend it names.

(deftest custom-objective-without-the-capability-signals
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (multiple-value-bind (matrix labels*) (regression-fixture)
               (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                     matrix labels*))
                 ;; Overwritten after the dataset is built, since building it needs the
                 ;; backend's other capabilities and this replaces the whole plist. A backend
                 ;; that already answered false would be left exactly as it opened.
                 (when (cl-gbdt:backend-supports-p backend :custom-objective)
                   (setf (cl-gbdt:backend-capabilities backend) '(:custom-objective nil)))
                 (testing (format nil "~A: train signals capability-unavailable for a ~
                                       non-NIL :objective, naming the capability and the ~
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
                                        :parameters (cdr (assoc name
                                                                *custom-regression-parameters*))
                                        :objective (squared-error-objective labels*)))
                                      nil)
                             (cl-gbdt:capability-unavailable (c) c))))
                     (ok condition "train signalled instead of training")
                     (ok (and condition
                              (eq :custom-objective
                                  (cl-gbdt:capability-unavailable-capability condition)))
                         (format nil "the condition named capability ~S"
                                 (and condition
                                      (cl-gbdt:capability-unavailable-capability condition))))
                     (ok (and condition
                              (eq name (cl-gbdt:backend-error-backend condition)))
                         (format nil "the condition named backend ~S"
                                 (and condition
                                      (cl-gbdt:backend-error-backend condition))))))
                 ;; OBJECTIVE NIL reaches no check at all, so the same backend still trains
                 ;; normally with the capability withdrawn. Without this the gate could be an
                 ;; unconditional refusal and nothing here would notice.
                 (cl-gbdt:with-booster
                     (booster (cl-gbdt:train
                               backend dataset :num-rounds 2
                               :parameters (cdr (assoc name
                                                       *built-in-regression-parameters*))))
                   (ok (arrayp (cl-gbdt:predict booster matrix :kind :raw))))))
          (cl-gbdt:close-backend backend))))))

;;; The other way :OBJECTIVE can be wrong, and the one a caller is far likelier to reach: a
;;; value that is not a function at all. It is refused beside the capability, before any
;;; foreign call -- see either backend's `%check-custom-objective'. Left to the loop's own
;;; `funcall' it would surface as SBCL's untyped `type-error' after the booster existed and a
;;; score read had already happened, naming neither the argument nor the backend.

(deftest a-non-function-objective-is-refused-before-any-foreign-call
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-objective)
               (multiple-value-bind (matrix labels*) (regression-fixture)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   ;; A number, a string, and a SYMBOL naming a real function of one
                   ;; argument. The symbol is the interesting one: `funcall' would accept it,
                   ;; so nothing downstream would fail, and the run would boost against
                   ;; whatever global definition that name happened to have at each
                   ;; iteration rather than against what the caller passed.
                   (dolist (value (list 42 "squared-error" 'squared-error-objective))
                     (testing (format nil "~A: train signals unsupported-argument for ~
                                           :objective ~S, naming the argument and the backend"
                                      name value)
                       (let ((condition
                               (handler-case
                                   (progn (cl-gbdt:free-booster
                                           (cl-gbdt:train
                                            backend dataset :num-rounds 1
                                            :parameters
                                            (cdr (assoc name *custom-regression-parameters*))
                                            :objective value))
                                          nil)
                                 (cl-gbdt:unsupported-argument (c) c))))
                         (ok condition "train signalled instead of training")
                         (ok (and condition
                                  (equal "train's :objective"
                                         (cl-gbdt:unsupported-argument-argument condition)))
                             (format nil "the condition named argument ~S"
                                     (and condition
                                          (cl-gbdt:unsupported-argument-argument condition))))
                         (ok (and condition
                                  (eq name
                                      (cl-gbdt:unsupported-argument-backend condition)))
                             (format nil "the condition named backend ~S"
                                     (and condition
                                          (cl-gbdt:unsupported-argument-backend
                                           condition))))))))))
          (cl-gbdt:close-backend backend))))))

(deftest a-custom-squared-error-reproduces-the-built-in-one
  ;; Only on backends that provide the capability. The comparison is always one backend's
  ;; custom run against that same backend's built-in run -- policy section 13 forbids
  ;; comparing numbers across backends, and there would be nothing to learn from it anyway.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-objective)
               (multiple-value-bind (matrix labels*) (regression-fixture)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                        matrix labels*))
                   (cl-gbdt:with-booster
                       (built-in (cl-gbdt:train
                                  backend dataset :num-rounds 6
                                  :parameters (cdr (assoc name
                                                          *built-in-regression-parameters*))))
                     (cl-gbdt:with-booster
                         (custom (cl-gbdt:train
                                  backend dataset :num-rounds 6
                                  :parameters (cdr (assoc name
                                                          *custom-regression-parameters*))
                                  :objective (squared-error-objective labels*)))
                       (ok (support:predictions-agree-p
                            (cl-gbdt:predict built-in matrix :kind :raw)
                            (cl-gbdt:predict custom matrix :kind :raw)))
                       ;; The control. A gradient of zero must NOT reproduce the built-in run;
                       ;; without this, a `train' that ignored :OBJECTIVE entirely and simply
                       ;; ran the library's own objective would pass the assertion above.
                       (cl-gbdt:with-booster
                           (flat (cl-gbdt:train
                                  backend dataset :num-rounds 6
                                  :parameters (cdr (assoc name
                                                          *custom-regression-parameters*))
                                  :objective (lambda (scores)
                                               (let ((rows (array-dimension scores 0)))
                                                 (values
                                                  (make-array (list rows 1)
                                                              :element-type 'double-float
                                                              :initial-element 0d0)
                                                  (make-array (list rows 1)
                                                              :element-type 'double-float
                                                              :initial-element 1d0))))))
                         (ok (not (support:predictions-agree-p
                                   (cl-gbdt:predict built-in matrix :kind :raw)
                                   (cl-gbdt:predict flat matrix :kind :raw))))))))))
          (cl-gbdt:close-backend backend))))))

(deftest the-objective-is-called-once-per-iteration-with-the-scores-shape
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-objective)
               (multiple-value-bind (matrix labels*) (regression-fixture)
                 (let ((calls 0)
                       (shapes '()))
                   (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                          matrix labels*))
                     (multiple-value-bind (booster report)
                         (cl-gbdt:train
                          backend dataset :num-rounds 5
                          :parameters (cdr (assoc name *custom-regression-parameters*))
                          :objective (lambda (scores)
                                       (incf calls)
                                       (pushnew (array-dimensions scores) shapes :test #'equal)
                                       (funcall (squared-error-objective labels*) scores)))
                       (cl-gbdt:free-booster booster)
                       (ok (= 5 calls))
                       (ok (= 5 (cl-gbdt:training-report-num-rounds report)))
                       (ok (equal shapes (list (list *rows* 1)))))))))
          (cl-gbdt:close-backend backend))))))

(deftest a-gradient-in-one-output-group-moves-only-that-group
  ;; THE FLATTENING TEST, and the only test in this file with more than one output group. Every
  ;; other test here runs at GROUPS = 1, where `(+ (* group rows) row)' and
  ;; `(+ (* row groups) group)' are THE SAME INDEX -- so a backend that flattened the caller's
  ;; arrays in the wrong order would pass all of them green. SIX docstrings state a layout as
  ;; measured fact -- group-major in `%booster-predictions' and `%update-one-iteration-custom' in
  ;; src/lightgbm/native.lisp and `train' in src/lightgbm/protocol.lisp, row-major in
  ;; `%booster-predictions' and `%train-one-iteration-custom' in src/xgboost/native.lisp and
  ;; `train' in src/xgboost/protocol.lisp; this is what holds all six. The two score readers
  ;; share one NAME across the two files and disagree about the layout, which is not a
  ;; contradiction: each states its own library's buffer. The two backends want OPPOSITE
  ;; orders and the caller writes neither: OBJECTIVE is handed, and returns, a (ROWS GROUPS)
  ;; array on both.
  ;;
  ;; The measurement that chose this gradient, taken during planning at ONE iteration on this
  ;; fixture ON LIGHTGBM, `num_leaves' 7 and `learning_rate' 1.0 -- the maximum a group's raw
  ;; score moved. Figures from one backend, never compared against the other's; XGBoost's own
  ;; are its own, and the ZEROS are all this test reads either way:
  ;;
  ;;   group-major (correct there)  0.2  / 0    / 0     -- only group 0 moved
  ;;   row-major   (broken there)   0.25 / 0.25 / 0.25  -- smeared across all three
  ;;
  ;; The test runs two iterations rather than that one, for the reason below; the figures grow
  ;; but the zeros stay zero, which is the whole of what is asserted.
  ;;
  ;; TWO iterations, and both halves of the flattening are pinned separately. The run's OUTPUT
  ;; pins the update function -- which groups the caller's gradient actually reached -- and the
  ;; SECOND call's own SCORES argument pins the score READER -- each backend's own
  ;; `%booster-predictions' -- which is the reverse direction
  ;; and would otherwise have nothing behind it: at iteration 1 every score is 0, so no
  ;; single-iteration run can tell the two readings of that buffer apart.
  ;;
  ;; Verified to discriminate rather than assumed to, ON EACH BACKEND SEPARATELY. Each of that
  ;; backend's two functions had its index temporarily flipped to the other order in turn and
  ;; the whole layer-2 suite re-run; this test failed all four times and NOTHING ELSE IN THE
  ;; SUITE DID, which is the finding that put it here. The two backends produced the identical
  ;; pair of signatures, one flip for one:
  ;;
  ;;   the UPDATE function flipped   both quiet-group assertions fail -- the wrong groups were
  ;;   (`%update-one-iteration-       written, so both the scores read back and the final
  ;;   custom' / `%train-one-         predictions smear
  ;;   iteration-custom')
  ;;   the score READER flipped      only the SCORES quiet-group assertion fails. The
  ;;   (either backend's             predictions are untouched, because this objective derives
  ;;   `%booster-predictions')       its gradient from the ROW INDEX and never reads SCORES --
  ;;                                 which is exactly why the scores assertion is not redundant
  ;;                                 with the prediction one and has to be made separately
  ;;
  ;; Backend-neutral and guarded like every other test here, which is what let it cover
  ;; XGBoost's ROW-major absorption with no edit to this form at all on the day that backend
  ;; stopped refusing the argument. Nothing is compared between backends: `column-spread' asks
  ;; how one group varies across its own rows, within one booster, which is a question neither
  ;; library's `base_score' convention enters into.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-objective)
               (multiple-value-bind (matrix labels*) (multiclass-fixture)
                 (let ((shapes '())
                       (scores-seen '()))
                   (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                         matrix labels*))
                     (cl-gbdt:with-booster
                         (booster (cl-gbdt:train
                                   backend dataset :num-rounds 2
                                   :parameters (cdr (assoc name *multiclass-parameters*))
                                   :objective
                                   (lambda (scores)
                                     (pushnew (array-dimensions scores) shapes :test #'equal)
                                     ;; No copy: each call gets a freshly allocated array.
                                     (push scores scores-seen)
                                     (funcall (first-group-only-objective) scores))))
                       ;; The objective was handed all three groups, not one -- without this a
                       ;; backend that silently passed GROUPS 1 would make the spread
                       ;; assertions below vacuous rather than failing them.
                       (ok (equal shapes
                                  (list (list *multiclass-rows* *num-classes*))))
                       (ok (= 2 (length scores-seen)))
                       ;; What the score reader READ back after iteration 1. Group 0 had
                       ;; moved by then and the other two had not, so the same shape of
                       ;; assertion applies to the scores as to the predictions -- and it is a
                       ;; row-major READ, not a row-major write, that this pair catches.
                       (let ((second-call (first scores-seen)))
                         (ok (> (column-spread second-call 0) *quiet-group-tolerance*))
                         (ok (loop :for group :from 1 :below *num-classes*
                                   :always (<= (column-spread second-call group)
                                               *quiet-group-tolerance*))))
                       (let ((raw (cl-gbdt:predict booster matrix :kind :raw)))
                         (ok (equal (array-dimensions raw)
                                    (list *multiclass-rows* *num-classes*)))
                         ;; Group 0 got the gradient, so it must have moved.
                         (ok (> (column-spread raw 0) *quiet-group-tolerance*))
                         ;; Groups 1 and 2 got exactly zero, so they must not have. This pair
                         ;; is what a row-major flattening fails.
                         (ok (loop :for group :from 1 :below *num-classes*
                                   :always (<= (column-spread raw group)
                                               *quiet-group-tolerance*)))))))))
          (cl-gbdt:close-backend backend))))))

(defun exact-gradient-objective (element-type)
  "Return an objective whose gradient and Hessian are (ROWS 1) arrays of ELEMENT-TYPE.

The gradient is a function of the ROW INDEX -- -1.5, -0.5, 0.5, 1.5 by `(mod row 4)' -- and
the Hessian is 1 everywhere. Every one of those five numbers is EXACTLY representable in
`single-float' and in `double-float' alike, which is the whole point: both backends convert
what comes back to `single-float' before writing it into a C buffer, so the two element
types produce byte-identical buffers here and therefore the identical model. A gradient read
off SCORES could not promise that -- `(coerce (- a b) 'single-float)' and
`(- (coerce a 'single-float) (coerce b 'single-float))' may differ in the last bit, which is
a true difference of about 1e-7 against a *PREDICTION-TOLERANCE* of 1e-9.

Four distinct levels rather than the alternating pair `first-group-only-objective' uses:
this objective drives a ONE-group run, where the only thing to assert is that the model moved
at all, and four levels leave more room for a split to find gain in.

ELEMENT-TYPE `t' is the third case the contract admits and is not merely a third float type:
`(coerce x t)' is the identity, so the array comes back holding the RATIONALS -3/2, -1/2,
1/2, 3/2 and the integer 1 rather than floats of any width. That is what makes it worth
running -- \"a general array whose elements are reals\" has to mean reals of any kind, and
`objective-single-float' is what turns each of them into the `single-float' the C buffer
takes."
  (lambda (scores)
    (let* ((rows (array-dimension scores 0))
           (grad (make-array (list rows 1) :element-type element-type))
           (hess (make-array (list rows 1) :element-type element-type
                                           :initial-element (coerce 1 element-type))))
      (dotimes (row rows (values grad hess))
        (setf (aref grad row 0) (coerce (- (mod row 4) 3/2) element-type))))))

(deftest every-element-type-the-contract-admits-trains-the-same-model
  ;; `train''s generic docstring says the objective's two arrays may be `double-float',
  ;; `single-float', or a general array whose elements are reals, and layer 1 pins only that
  ;; `check-objective-result' ACCEPTS all three shapes of thing. This is the half that runs
  ;; them through a real library: the same five exactly representable numbers returned three
  ;; ways, trained, and compared -- within one backend, never across the two.
  ;;
  ;; The `t' case is the one that made the docstring wrong before this: the generic promised
  ;; `double-float' or `single-float' and named `dimension-mismatch' for "anything else",
  ;; while the code checked shape alone and trained a bare `(make-array (list rows 1))'
  ;; identically. Measured on both backends, all three runs below land on the same model, so
  ;; enforcing the documented pair would have rejected working code rather than caught a bug.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-objective)
               (multiple-value-bind (matrix labels*) (regression-fixture)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   (flet ((raw-for (element-type)
                            (cl-gbdt:with-booster
                                (booster (cl-gbdt:train
                                          backend dataset :num-rounds 4
                                          :parameters
                                          (cdr (assoc name *custom-regression-parameters*))
                                          :objective
                                          (exact-gradient-objective element-type)))
                              (cl-gbdt:predict booster matrix :kind :raw))))
                     (let ((double (raw-for 'double-float))
                           (single (raw-for 'single-float))
                           (general (raw-for t)))
                       ;; The model MOVED. Without this the equality below would be satisfied
                       ;; by two identical constants -- which is what a run whose gradient
                       ;; never reached the trees produces, and this file has seen that
                       ;; failure before.
                       (ok (> (column-spread single 0) *quiet-group-tolerance*))
                       (ok (support:predictions-agree-p double single))
                       (ok (support:predictions-agree-p double general)))))))
          (cl-gbdt:close-backend backend))))))

(deftest a-gradient-element-that-is-not-a-real-signals-before-the-library-sees-it
  ;; The other half of "a general array whose elements are reals": a general array can hold
  ;; ANYTHING, so the element check has to exist somewhere. It lives at the buffer write, one
  ;; element at a time, rather than in a validation scan over both arrays -- see
  ;; `objective-single-float'. Before it, this run reached `coerce' and produced a bare
  ;; `TYPE-ERROR' ("The value \"nope\" is not of type REAL") from inside the wrapper, which
  ;; names neither the argument nor the library and is not one of this project's conditions.
  ;;
  ;; The buffers exist when this signals -- `cffi:with-foreign-objects' has allocated them --
  ;; but no library call has been made, and that allocation unwinds with everything else.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-objective)
               (multiple-value-bind (matrix labels*) (regression-fixture)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                       matrix labels*))
                   (let ((condition
                           (handler-case
                               (progn (cl-gbdt:free-booster
                                       (cl-gbdt:train
                                        backend dataset :num-rounds 3
                                        :parameters (cdr (assoc name
                                                                *custom-regression-parameters*))
                                        :objective
                                        (lambda (scores)
                                          (let ((rows (array-dimension scores 0)))
                                            (values (make-array (list rows 1)
                                                                :initial-element "nope")
                                                    (make-array (list rows 1)
                                                                :initial-element 1))))))
                                      nil)
                             (cl-gbdt:unsupported-element-type (c) c))))
                     (ok condition "train signalled instead of training")
                     ;; The condition names what was FOUND, not merely that something was
                     ;; wrong: `(type-of "nope")' is what `%require-real-values' puts in the
                     ;; same slot for a `csr-matrix' value that is not a real.
                     (ok (and condition
                              (equal (type-of "nope")
                                     (cl-gbdt:unsupported-element-type-given condition)))
                         (format nil "the condition named ~S"
                                 (and condition
                                      (cl-gbdt:unsupported-element-type-given condition)))))
                   ;; Still usable afterwards: the failed run freed its own booster and left
                   ;; nothing of the caller's behind.
                   (cl-gbdt:with-booster
                       (booster (cl-gbdt:train
                                 backend dataset :num-rounds 2
                                 :parameters (cdr (assoc name
                                                         *built-in-regression-parameters*))))
                     (ok (arrayp (cl-gbdt:predict booster matrix :kind :raw)))))))
          (cl-gbdt:close-backend backend))))))

(deftest an-error-inside-the-objective-propagates-and-frees-the-booster
  ;; `train' holds a full booster handle from `create-booster' before the loop starts, and
  ;; frees it through an `unwind-protect' if the run does not complete. A condition raised
  ;; from the caller's own function must unwind through that same path, or the run leaks a
  ;; handle the caller has no way to free.
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-objective)
               (multiple-value-bind (matrix labels*) (regression-fixture)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                        matrix labels*))
                   (ok (handler-case
                           (progn (cl-gbdt:train
                                   backend dataset :num-rounds 5
                                   :parameters (cdr (assoc name
                                                           *custom-regression-parameters*))
                                   :objective (lambda (scores)
                                                (declare (ignore scores))
                                                (error "objective gave up")))
                                  nil)
                         (simple-error () t)))
                   ;; The backend is still usable afterwards, which is the observable half of
                   ;; "the booster was freed rather than orphaned".
                   (cl-gbdt:with-booster
                       (booster (cl-gbdt:train
                                 backend dataset :num-rounds 2
                                 :parameters (cdr (assoc name
                                                         *built-in-regression-parameters*))))
                     (ok (arrayp (cl-gbdt:predict booster matrix :kind :raw)))))))
          (cl-gbdt:close-backend backend))))))

;;; Policy section 13's "released handle" line, for the handles `train''s own loop holds raw
;;; pointers to across a call into the caller's code. `train' reads each dataset's pointer
;;; ONCE, before the loop, and the objective is the only place in that loop where code this
;;; library did not write runs -- so an objective that frees a dataset leaves the loop holding
;;; a pointer into freed memory and hands it straight to C. Measured before
;;; `%recheck-train-datasets' existed, in isolated subprocesses, on both backends: LightGBM
;;; died with `Memory fault at 0x543447170e8a6' and XGBoost with `Signal 7 received' -- the
;;; process, not the test. Everywhere else in this library a freed handle produces a typed
;;; condition, and this is what makes the custom-objective loop no exception.
;;;
;;; `handler-case', not rove's `signals', which does not reliably catch a condition raised
;;; inside `restart-case'; the condition TYPE is asserted, and so is the handle it names,
;;; since a run that signalled about the wrong handle would be a run that noticed by accident.

(deftest freeing-a-dataset-inside-the-objective-signals-rather-than-faulting
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-objective)
               (multiple-value-bind (matrix labels*) (regression-fixture)
                 (flet ((train-freeing (victim dataset valid-sets)
                          ;; Free VICTIM from inside the objective, then return a gradient
                          ;; and Hessian that are in every way correct: what is under test is
                          ;; the freed handle alone, and a wrong shape would be caught by
                          ;; `check-objective-result' instead and prove nothing.
                          (handler-case
                              (progn (cl-gbdt:free-booster
                                      (cl-gbdt:train
                                       backend dataset :num-rounds 5 :valid-sets valid-sets
                                       :parameters
                                       (cdr (assoc name *custom-regression-parameters*))
                                       :objective
                                       (let ((gradient (squared-error-objective labels*)))
                                         (lambda (scores)
                                           (cl-gbdt:free-dataset victim)
                                           (funcall gradient scores)))))
                                     nil)
                            (cl-gbdt:released-handle-error (c) c))))
                   (testing (format nil "~A: freeing the TRAINING set inside the objective"
                                    name)
                     (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                           matrix labels*))
                       (let ((condition (train-freeing dataset dataset '())))
                         (ok condition "train signalled instead of faulting")
                         (ok (and condition
                                  (eq dataset
                                      (cl-gbdt:released-handle-error-object condition)))
                             "the condition named the training set"))))
                   ;; The validation sets are re-checked too, and for the same reason: the
                   ;; same iteration evaluates them through `%read-evaluation' once the
                   ;; update returns -- LightGBM through pointers the booster stores,
                   ;; XGBoost through the list `train' built before the loop.
                   (testing (format nil "~A: freeing a VALIDATION set inside the objective"
                                    name)
                     (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                           matrix labels*))
                       (cl-gbdt:with-dataset (valid-set (make-labelled-dataset
                                                         backend name matrix labels*
                                                         :reference dataset))
                         (let ((condition
                                 (train-freeing valid-set dataset (list valid-set))))
                           (ok condition "train signalled instead of faulting")
                           (ok (and condition
                                    (eq valid-set
                                        (cl-gbdt:released-handle-error-object condition)))
                               "the condition named the validation set")))))
                   ;; The backend is still usable afterwards, which is the observable half of
                   ;; "the booster was freed rather than orphaned" on this path too.
                   (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                         matrix labels*))
                     (cl-gbdt:with-booster
                         (booster (cl-gbdt:train
                                   backend dataset :num-rounds 2
                                   :parameters (cdr (assoc name
                                                           *built-in-regression-parameters*))))
                       (ok (arrayp (cl-gbdt:predict booster matrix :kind :raw))))))))
          (cl-gbdt:close-backend backend))))))

(deftest a-wrong-shaped-gradient-signals-before-the-library-sees-it
  (dolist (name '(:lightgbm :xgboost))
    (support:with-backend-library (name)
      (let ((backend (cl-gbdt:open-backend name)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :custom-objective)
               (multiple-value-bind (matrix labels*) (regression-fixture)
                 (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend name
                                                                        matrix labels*))
                   (ok (handler-case
                           (progn (cl-gbdt:train
                                   backend dataset :num-rounds 3
                                   :parameters (cdr (assoc name
                                                           *custom-regression-parameters*))
                                   :objective
                                   (lambda (scores)
                                     (declare (ignore scores))
                                     ;; A flat vector of the right total size: the shape a
                                     ;; caller who thinks in one dimension returns.
                                     (values (make-array *rows* :element-type 'double-float
                                                                :initial-element 0d0)
                                             (make-array *rows* :element-type 'double-float
                                                                :initial-element 1d0))))
                                  nil)
                         (cl-gbdt:dimension-mismatch () t)))
                   ;; Returning one value where two are required is the same failure and must
                   ;; be the same condition, not a `type-error' from inside the buffer write.
                   (ok (handler-case
                           (progn (cl-gbdt:train
                                   backend dataset :num-rounds 3
                                   :parameters (cdr (assoc name
                                                           *custom-regression-parameters*))
                                   :objective (lambda (scores)
                                                (make-array (list (array-dimension scores 0) 1)
                                                            :element-type 'double-float
                                                            :initial-element 0d0)))
                                  nil)
                         (cl-gbdt:dimension-mismatch () t))))))
          (cl-gbdt:close-backend backend))))))

(defparameter *lightgbm-objective-keys*
  '(:objective :objective-type :app :application :loss)
  "Every key LightGBM reads as `objective', restated here rather than imported from
`cl-gbdt/src/config/objective''s `*objective-parameter-names*' -- the same way
`cl-gbdt/tests/functional/categorical-features''s `*refused-parameter-keys*' restates
`*categorical-feature-parameter-names*'. A test that imported the source's own list would
agree with it by construction and could never catch the list drifting away from the library.

The four aliases come from the vendored LightGBM 4.7.0 itself: `LGBM_DumpParamAliases'
returns that library's parameter-to-alias map, and its `objective' entry is exactly
`[\"app\", \"loss\", \"application\", \"objective_type\"]'. Measured against the same
library's behaviour before this list was written: each of the four naming \"binary\" trains
the model `objective=\"binary\"' trains, element for element, while `obj', `apps',
`applications', `losses', `loss_function', `objective_function' and `app_type' all leave the
trained numbers identical to a run naming no objective at all.

What this test can and cannot show, stated plainly because the two are easy to confuse.
LightGBM resolves the canonical `objective=none' AHEAD of an alias in the same parameter
string: measured, `app=binary objective=none' trains the custom model, and even
`app=no-such-objective objective=none' builds a booster without complaint, where
`app=no-such-objective' alone fails `LGBM_BoosterCreate' with `Unknown objective type name'.
So a surviving alias is INVISIBLE from outside the library, and no run this file could write
would tell the dropped list from the literal one. That is exactly why the aliases had to be
dropped rather than left to win a race on an undocumented precedence rule, and it is why the
DROP itself is pinned in layer 1, by `cl-gbdt/tests/objective''s
`objective-parameters-drops-every-spelling-lightgbm-honours'. What the loop below pins is the
BEHAVIOUR a caller sees -- a run naming any of the five trains the custom model -- which
before the drop was inherited from that precedence rule and is now asserted.")

(deftest lightgbm-forces-its-objective-parameter-to-none
  ;; The override, observed rather than asserted from the source: a run that names
  ;; `objective=regression' in :PARAMETERS while passing an :OBJECTIVE function must train,
  ;; and must train the same model as one that named no objective at all. Without the
  ;; override LightGBM refuses the update outright -- `Check failed: objective_function_ ==
  ;; nullptr' -- and nothing trains.
  ;;
  ;; Run for the canonical spelling AND for each of LightGBM's four aliases for it, since
  ;; each is a live key in the vendored library -- see *LIGHTGBM-OBJECTIVE-KEYS*.
  (support:with-backend-library (:lightgbm)
    (let ((backend (cl-gbdt:open-backend :lightgbm)))
      (unwind-protect
           (multiple-value-bind (matrix labels*) (regression-fixture)
             (cl-gbdt:with-dataset (dataset (make-labelled-dataset backend :lightgbm
                                                                    matrix labels*))
               (cl-gbdt:with-booster
                   (silent (cl-gbdt:train
                            backend dataset :num-rounds 4
                            :parameters (cdr (assoc :lightgbm *custom-regression-parameters*))
                            :objective (squared-error-objective labels*)))
                 (let ((expected (cl-gbdt:predict silent matrix :kind :raw)))
                   (dolist (key *lightgbm-objective-keys*)
                     (testing (format nil "~S in :parameters loses to the custom objective"
                                      key)
                       (cl-gbdt:with-booster
                           (overridden
                            (cl-gbdt:train
                             backend dataset :num-rounds 4
                             :parameters (list* key "regression"
                                                (cdr (assoc :lightgbm
                                                            *custom-regression-parameters*)))
                             :objective (squared-error-objective labels*)))
                         (ok (support:predictions-agree-p
                              expected
                              (cl-gbdt:predict overridden matrix :kind :raw))))))))))
        (cl-gbdt:close-backend backend)))))
