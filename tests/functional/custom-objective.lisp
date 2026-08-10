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
  ;; Zero symbols, both of them: their only job is to run at load time and register
  ;; :lightgbm and :xgboost with `open-backend'. Without these clauses,
  ;; package-inferred-system has no edge to those files and `(cl-gbdt:open-backend
  ;; :lightgbm)' below would signal `unknown-backend'.
  (:import-from #:cl-gbdt/src/lightgbm/all)
  (:import-from #:cl-gbdt/src/xgboost/all)
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

(defun make-labelled-dataset (backend name matrix labels*)
  "Build a dataset on BACKEND, named NAME, from MATRIX and LABELS*, passing only the
`make-dataset' :PARAMETERS that backend accepts -- see *DATASET-PARAMETERS*. One call site
works for both backends and for both fixtures, which is what lets every test below be
written once."
  (cl-gbdt:make-dataset backend matrix :label labels*
                                       :parameters (cdr (assoc name *dataset-parameters*))))

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
*CUSTOM-REGRESSION-PARAMETERS* does, and for the same reason. XGBoost's row DOES name one:
that backend's parameters are never rewritten by `train', so `multi:softprob' is how its
booster comes to have three output groups to hand the objective in the first place.

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
                       (ok (handler-case
                               (progn (cl-gbdt:train
                                       backend dataset :num-rounds 1
                                       :objective (squared-error-objective labels*))
                                      nil)
                             (cl-gbdt:capability-unavailable () t))))))
            (cl-gbdt:close-backend backend)))))
    (ok (plusp demonstrated))))

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
  ;; arrays in the wrong order would pass all of them green. Three docstrings state the
  ;; group-major layout as measured fact (`%training-scores' and `%update-one-iteration-custom'
  ;; in src/lightgbm/native.lisp, and `train' in src/lightgbm/protocol.lisp); this is what
  ;; holds them.
  ;;
  ;; The measurement that chose this gradient, taken during planning at ONE iteration on this
  ;; fixture, `num_leaves' 7 and `learning_rate' 1.0 -- the maximum a group's raw score moved:
  ;;
  ;;   group-major (correct)  0.2  / 0    / 0     -- only group 0 moved
  ;;   row-major   (broken)   0.25 / 0.25 / 0.25  -- smeared across all three
  ;;
  ;; The test runs two iterations rather than that one, for the reason below; the figures grow
  ;; but the zeros stay zero, which is the whole of what is asserted.
  ;;
  ;; TWO iterations, and both halves of the flattening are pinned separately. The run's OUTPUT
  ;; pins `%update-one-iteration-custom' -- which groups the caller's gradient actually reached
  ;; -- and the SECOND call's own SCORES argument pins `%training-scores', which is the reverse
  ;; direction and would otherwise have nothing behind it: at iteration 1 every score is 0, so
  ;; no single-iteration run can tell the two readings of that buffer apart.
  ;;
  ;; Verified to discriminate rather than assumed to. Each function's index was temporarily
  ;; flipped to row-major in turn and the whole layer-2 suite re-run; this test failed both
  ;; times and NOTHING ELSE IN THE SUITE DID, which is the finding that put it here.
  ;;
  ;;   `%update-one-iteration-custom' flipped  both quiet-group assertions fail -- the wrong
  ;;                                          groups were written, so both the scores read
  ;;                                          back and the final predictions smear
  ;;   `%training-scores' flipped              only the SCORES quiet-group assertion fails.
  ;;                                          The predictions are untouched, because this
  ;;                                          objective derives its gradient from the ROW
  ;;                                          INDEX and never reads SCORES -- which is exactly
  ;;                                          why the scores assertion is not redundant with
  ;;                                          the prediction one and has to be made separately
  ;;
  ;; Backend-neutral and guarded like every other test here, so it costs XGBoost nothing while
  ;; that backend refuses the argument and covers its ROW-major absorption with no edit once it
  ;; does not. Nothing is compared between backends: `column-spread' asks how one group varies
  ;; across its own rows, within one booster, which is a question neither library's
  ;; `base_score' convention enters into.
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
                       ;; What `%training-scores' READ back after iteration 1. Group 0 had
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

(deftest an-error-inside-the-objective-propagates-and-frees-the-booster
  ;; `train' builds a raw booster handle before the loop and only takes ownership of it at the
  ;; end. A condition raised from the caller's own function must unwind through that same
  ;; path, or the run leaks a handle the caller has no way to free.
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

(deftest lightgbm-forces-its-objective-parameter-to-none
  ;; The override, observed rather than asserted from the source: a run that names
  ;; `objective=regression' in :PARAMETERS while passing an :OBJECTIVE function must train,
  ;; and must train the same model as one that named no objective at all. Without the
  ;; override LightGBM refuses the update outright -- `Check failed: objective_function_ ==
  ;; nullptr' -- and nothing trains.
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
                 (cl-gbdt:with-booster
                     (overridden
                      (cl-gbdt:train
                       backend dataset :num-rounds 4
                       :parameters (list* :objective "regression"
                                          (cdr (assoc :lightgbm
                                                      *custom-regression-parameters*)))
                       :objective (squared-error-objective labels*)))
                   (ok (support:predictions-agree-p
                        (cl-gbdt:predict silent matrix :kind :raw)
                        (cl-gbdt:predict overridden matrix :kind :raw)))))))
        (cl-gbdt:close-backend backend)))))
