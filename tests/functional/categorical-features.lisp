;;;; categorical-features.lisp --- Portable contract tests for `make-dataset''s
;;;; :CATEGORICAL-FEATURES.
;;;;
;;;; `cl-gbdt:make-dataset' now takes a :CATEGORICAL-FEATURES keyword naming which columns of
;;;; the caller's own matrix hold CATEGORIES rather than quantities, gated by the
;;;; `:categorical-features' capability. Like tests/functional/evaluation.lisp,
;;;; tests/functional/sparse-input.lisp and tests/functional/missing-value.lisp beside it,
;;;; every test below runs over that first file's *FIXTURES*, once per backend, so the two
;;;; backends cannot drift apart in shape or meaning without one of them failing here.
;;;;
;;;; Which backend takes which branch is read from `cl-gbdt:backend-supports-p', never
;;;; hardcoded. Both answer true today, by two different mechanisms -- a field attached to a
;;;; finished DMatrix on XGBoost, a key in the parameter string that builds the dataset on
;;;; LightGBM -- and the tests below are written so a backend that later LOSES the capability,
;;;; or a third that gains it, changes which branch it takes rather than which tests exist.
;;;;
;;;; `cl-gbdt:predict' takes no :CATEGORICAL-FEATURES of its own, and there is nothing here
;;;; about one: the trained trees carry the category sets they split on, so the transient
;;;; DMatrix `predict' builds needs no feature-type vector to route a row down them.
;;;; Measured, and the reason every prediction below is a bare `cl-gbdt:predict' call on the
;;;; same matrix in both arms.
;;;;
;;;; Numbers are never compared BETWEEN backends: policy section 13 asks for shape, order and
;;;; meaning, not numeric agreement, and the two libraries train different models from the
;;;; same rows by design. Every comparison below is categorical-versus-not, WITHIN one
;;;; backend.

(uiop:define-package #:cl-gbdt/tests/functional/categorical-features
  ;; Zero symbols: every reference below is package-qualified. Declared so this file's
  ;; dependency on the unified API is explicit rather than inherited, matching the identical
  ;; clause in evaluation.lisp, sparse-input.lisp and missing-value.lisp.
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt)
  ;; Zero symbols, both of them: their only job is to run at load time and register
  ;; :lightgbm and :xgboost with `open-backend'. Without these clauses,
  ;; package-inferred-system has no edge to those files and `(cl-gbdt:open-backend
  ;; :lightgbm)' below would signal `unknown-backend'.
  (:import-from #:cl-gbdt/src/lightgbm/all)
  (:import-from #:cl-gbdt/src/xgboost/all)
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library)
  ;; The fixture table comes from evaluation.lisp rather than being restated here, for the
  ;; reason that file's own export comment gives: a second table saying the same thing in its
  ;; own words is how two files that must agree stop agreeing. Only *FIXTURES* --
  ;; `make-fixture-dataset' builds no dataset with :CATEGORICAL-FEATURES, this file's whole
  ;; subject, so `make-categorical-dataset' below is its own builder over the same table.
  (:import-from #:cl-gbdt/tests/functional/evaluation
                #:*fixtures*))

(in-package #:cl-gbdt/tests/functional/categorical-features)

(defparameter *num-categories* 6
  "How many distinct categories column 0 of `category-matrix' holds.

Six, not two: with two the categorical and the quantitative reading of the column agree
exactly -- one threshold between them separates the classes either way -- and nothing below
could tell them apart. Six with the classes ALTERNATING is the smallest fixture where no
ordering of the column separates them, so a model that can only threshold has to approximate
what a model that can group categories gets exactly.")

(defparameter *rows-per-category* 4
  "How many rows each category of `category-matrix' holds, making it 24 rows in all.

Measured, and chosen from the two sizes tried rather than assumed -- on XGBoost, the only
backend that answered `:categorical-features' true when the size was chosen. Four is the
smaller and faster of them, it still gives every category more than one row, and it is where
the two readings of column 0 are LEAST alike: the arm trained without :CATEGORICAL-FEATURES
separates its own classes by 0.086 here against 0.106 at eight rows per category, while the
categorical arm reaches 0.948 here and 0.969 there. Both sizes discriminate, and the
assertions below hold by a wide margin on either; see
`marking-a-column-categorical-changes-what-was-learned' for all twelve numbers, and for why
the size was measured at all rather than picked.

LightGBM was measured at this size only, and separates the two readings further still: 0.869
against 0.000. Four rows per category also happens to be well under LightGBM's own
`min_data_per_group' default, which is why *BOOSTER-PARAMETERS* has to say otherwise.")

(defparameter *categorical-columns* '(0)
  "The :CATEGORICAL-FEATURES argument every honouring test below builds its dataset with:
column 0 of `category-matrix', the one holding category ordinals, and not column 1.")

(defparameter *training-rounds* 20
  "How many boosting rounds every booster below is trained for.

Twenty rather than the five tests/functional/missing-value.lisp uses: the depth-1 stumps this
fixture trains move slowly, and the point of the comparison is what the two readings CONVERGE
to, not how fast. Fewer rounds shrinks both arms' separation together and blunts the
measurement rather than changing what it says.")

(defparameter *prediction-tolerance* 1d-9
  "How far two `cl-gbdt:predict' results may differ, element for element, and still count as
the same numbers below.

The same value, for the same reason, as tests/functional/missing-value.lisp's parameter of
this name: both backends train and predict deterministically from fixed data, so two runs
that built the same dataset should come off identical trees down identical paths. It is small
enough that nothing this file exists to catch survives it: the two arms of every inequality
below differ by 0.158 on XGBoost, and by 0.119 on LightGBM, at the category where they are
closest -- and the inequality needs only ONE element over the tolerance, so what it actually
rests on is the 0.435 they differ by where they are furthest apart, which is the same number
on both. All of them are eight orders of magnitude above this.")

(defparameter *separating-margin* 0.5d0
  "How far the worst-scored positive category must sit above the best-scored negative one for
`marking-a-column-categorical-changes-what-was-learned' to call the alternating classes
separated.

Measured before it was asserted, 24 rows, twenty rounds, on each backend separately. XGBoost:
the categorical arm's margin is 0.948 and the quantitative arm's is 0.086. LightGBM: 0.869 and
0.000 -- its quantitative arm does not separate the classes AT ALL, two categories of opposite
class landing on the same score. So on either backend this sits far above what a model that
cannot group categories reaches and comfortably below what one that can does. It is a
threshold on ONE arm's own numbers, never a comparison between backends.")

(defparameter *booster-parameters*
  '((:lightgbm :objective "binary" :max-depth 1 :verbose -1 :min-data-in-leaf 1
               :min-data-per-group 1 :cat-smooth 0 :cat-l2 0)
    (:xgboost :objective "binary:logistic" :max-depth 1 :verbosity 0 :min-child-weight 0
              :tree-method "hist"))
  "Per backend, keyed by backend name, the `cl-gbdt:train' parameters this file's fixture
needs. Not *FIXTURES*' own :BOOSTER-PARAMETERS, which trains a depth-2 booster on a different
eight-row problem; this fixture needs settings neither of those entries has.

Both entries cap the tree at ONE split, and everything else in each is that backend's own way
of stopping its splitting rules from rejecting a split over four rows. The two lists are not
translations of each other and are never compared: each was measured on its own library.

XGBoost:

`max_depth' 1 makes every tree a single split, so what each tree can say about column 0 is
exactly one partition of it -- a threshold under the quantitative reading, an arbitrary
category subset under the categorical one. That is the whole difference this file measures,
and a deeper tree lets the quantitative reading rebuild an alternating pattern within one tree
and hide it.

`min_child_weight' 0 is required, measured: at the default of 1 the categorical arm's own
margin collapses to 0.085 -- 0.674 / 0.457 / 0.761 / 0.241 / 0.542 / 0.325 by category --
because the per-leaf hessian sum over four rows does not clear it and splits get rejected. The
categorical arm then no longer separates the classes either, and a comparison between two arms
that both fail says nothing about the one that should have succeeded.

`tree_method' `hist' is named rather than left to the library's default, measured both ways:
the default resolves to an updater that supports categorical features and gives numbers
identical to `hist' here, but `exact' refuses them outright -- and refuses at `train', not at
`make-dataset' (`updater_colmaker.cc:107: Updater `grow_colmaker` or `exact` tree method
doesn't support categorical features'), which would surface as a `foreign-call-error' from a
call this file is not about. Naming the updater keeps that out of reach of a default change.

LightGBM, measured against the vendored 4.7.0 by dropping each setting from the list and
re-running both arms:

`max_depth' 1 for the same reason as XGBoost's, and here the measurement is decisive rather
than cautionary: WITHOUT the cap the two arms answer identically -- 0.934 / 0.066 / 0.934 /
0.066 / 0.934 / 0.066 both -- and `marking-a-column-categorical-changes-what-was-learned''s
first assertion fails outright. Twenty unconstrained trees rebuild the alternating pattern
from the ordinal alone, which is exactly what a depth-1 stump cannot do. `num_leaves' is left
at the library's default: measured, setting it to 2 alongside `max_depth' 1 changes no number.

`min_data_in_leaf' 1 is required, measured: at the default of 20 neither arm splits at all on
24 rows and both answer 0.5 for every category.

`min_data_per_group' 1 is required, measured: it is the minimum row count LightGBM demands of
each side of a CATEGORICAL split, its default is 100, and this fixture gives each category
four rows -- so at the default the categorical arm makes no split and answers 0.5 everywhere,
while the quantitative arm is unaffected. It is this backend's counterpart of XGBoost's
`min_child_weight' 0 above, and fails in exactly the same way when left alone.

`cat_smooth' 0 is required, measured: at the default of 10 the categorical arm again answers
0.5 for every category. The smoothing is applied to a per-category statistic computed over
four rows, so a default sized for hundreds erases the categories' differences entirely.

`cat_l2' 0 is required for the assertion rather than for the split, measured: at the default
of 10 the categorical arm does split, but its margin is 0.361, below *SEPARATING-MARGIN*.

`verbose' -1 is XGBoost's `verbosity' 0 above: nothing this file measures depends on it -- the
vendored library prints nothing either way here -- and it is named so a build noisier than
that one cannot put per-iteration lines into the suite's output.")

(defun category-positive-p (category)
  "True when CATEGORY's rows carry the positive label.

The one definition of which categories are the positive class: `category-labels' builds the
label vector from it and `separation-margin' splits the per-category scores by it, so the two
cannot come to disagree. Alternating -- categories 0, 2 and 4 positive, 1, 3 and 5 negative --
which is the point of the fixture: no threshold on the category ordinal separates them, and no
reordering of the categories would help, since the classes alternate under every ordering a
threshold could exploit."
  (evenp category))

(defun category-matrix ()
  "Return this file's fixture as a `(simple-array double-float (24 2))'.

Column 0 holds the category ordinal, `*ROWS-PER-CATEGORY*' consecutive rows per category,
0 through `*NUM-CATEGORIES*' - 1. Read as a QUANTITY it is an ordinal a split can only
threshold; read as a CATEGORY it is a set a split can partition freely. Nothing about the
column itself says which -- that is exactly what :CATEGORICAL-FEATURES says, and why the same
matrix is handed to both arms of every comparison below.

Column 1 is noise: it alternates 0.0 and 1.0 within every category alike, so it carries no
class information at all and no split on it can help. It is here so the type vector
:CATEGORICAL-FEATURES renders is a MIXED one -- `(\"c\" \"q\")', not a vector of one element
-- since a single-column matrix could not tell a wrapper that marked column 0 from one that
marked every column."
  (let* ((rows (* *num-categories* *rows-per-category*))
         (matrix (make-array (list rows 2) :element-type 'double-float)))
    (dotimes (row rows)
      (setf (aref matrix row 0) (coerce (floor row *rows-per-category*) 'double-float))
      (setf (aref matrix row 1) (coerce (mod row 2) 'double-float)))
    matrix))

(defun category-labels ()
  "Return the labels for `category-matrix', row for row, as a `(simple-array single-float
(24))': 1.0 for a row of a `category-positive-p' category and 0.0 otherwise."
  (let* ((rows (* *num-categories* *rows-per-category*))
         (label-vector (make-array rows :element-type 'single-float)))
    (dotimes (row rows)
      (setf (aref label-vector row)
            (if (category-positive-p (floor row *rows-per-category*)) 1.0 0.0)))
    label-vector))

(defun category-csr (matrix)
  "Return MATRIX, a `category-matrix' result, as a `cl-gbdt:csr-matrix' of the same width.

Every element is stored explicitly, zeros included, rather than only the non-zero ones a CSR
conversion usually keeps -- the same reason tests/functional/missing-value.lisp's
`fixture-csr' gives: an entry a `csr-matrix' does not store is MISSING to XGBoost whatever any
config says, and category 0's rows would otherwise arrive with their category absent rather
than present and equal to zero."
  (let* ((rows (array-dimension matrix 0))
         (columns (array-dimension matrix 1))
         (indptr (make-array (1+ rows)))
         (indices (make-array (* rows columns)))
         (values (make-array (* rows columns)))
         (position 0))
    (dotimes (row rows)
      (setf (aref indptr row) position)
      (dotimes (column columns)
        (setf (aref indices position) column)
        (setf (aref values position) (aref matrix row column))
        (incf position)))
    (setf (aref indptr rows) position)
    (cl-gbdt:make-csr-matrix :indptr indptr :indices indices :values values
                             :num-columns columns)))

(defun booster-parameters (backend-name)
  "Return BACKEND-NAME's entry in *BOOSTER-PARAMETERS*.

Signals rather than returning NIL for a backend with no entry: NIL is a perfectly valid
:PARAMETERS argument meaning \"the library's own defaults\", and those are not the settings
*BOOSTER-PARAMETERS* records the measurements for -- one of them, `min_child_weight', is
measured there to break the comparison at its default. A backend that gained the capability
without gaining an entry here would silently run a different experiment and report its numbers
as this one's."
  (let ((entry (assoc backend-name *booster-parameters*)))
    (unless entry
      (error "No entry for ~S in *BOOSTER-PARAMETERS*: a backend that answers ~
              :categorical-features true needs one before it can be trained here."
             backend-name))
    (rest entry)))

(defun make-categorical-dataset (fixture backend matrix &key categorical-features)
  "Build a dataset on BACKEND from MATRIX and `category-labels', passing CATEGORICAL-FEATURES
through to `cl-gbdt:make-dataset' when it is non-NIL and FIXTURE's own :DATASET-PARAMETERS
when it has any.

CATEGORICAL-FEATURES NIL is passed as nothing at all rather than as `:categorical-features
nil', which is what makes the two arms of every comparison below differ in exactly the way a
caller's two calls would: one names the argument, the other has never heard of it.

:DATASET-PARAMETERS is filtered the way `cl-gbdt/tests/functional/evaluation''s
`make-fixture-dataset' filters it and for the same reason -- see *FIXTURES*, where XGBoost's
is NIL because that backend refuses :PARAMETERS outright."
  (apply #'cl-gbdt:make-dataset backend matrix :label (category-labels)
         (append (let ((parameters (getf fixture :dataset-parameters)))
                   (when parameters (list :parameters parameters)))
                 (when categorical-features
                   (list :categorical-features categorical-features)))))

(defun train-on (fixture backend matrix &key categorical-features)
  "Train a booster on BACKEND from MATRIX -- dense or a `cl-gbdt:csr-matrix' -- for
*TRAINING-ROUNDS* rounds, and return its predictions on the dense `category-matrix'.

Predicting on the dense matrix whatever MATRIX was is what makes two such results comparable:
what then differs between them is how the DATASET was built, not what the model was asked
about. `cl-gbdt:predict' is handed no categorical argument because there is none to hand it --
see this file's header."
  (cl-gbdt:with-dataset
      (dataset (make-categorical-dataset fixture backend matrix
                                          :categorical-features categorical-features))
    (cl-gbdt:with-booster
        (booster (cl-gbdt:train backend dataset :num-rounds *training-rounds*
                                :parameters (booster-parameters (getf fixture :backend))))
      (cl-gbdt:predict booster (category-matrix)))))

(defun predictions-agree-p (left right)
  "True when LEFT and RIGHT, two `cl-gbdt:predict' results, have the same shape and no pair of
corresponding elements differs by more than *PREDICTION-TOLERANCE*."
  (and (equal (array-dimensions left) (array-dimensions right))
       (loop :for index :below (array-total-size left)
             :always (<= (abs (- (row-major-aref left index) (row-major-aref right index)))
                         *prediction-tolerance*))))

(defun category-means (predictions)
  "Return one mean score per category, in category order, from PREDICTIONS -- a
`cl-gbdt:predict' result over `category-matrix', whose rows are grouped by category.

`row-major-aref' indexes rows directly, which is what a one-score-per-row result makes it:
measured, both backends return a binary booster's predictions as an array of `(24 1)', and a
plain vector of 24 would index the same way.

The mean rather than any one row's score: measured, every row of a category does answer
identically in both arms, column 1 being noise neither model splits on -- but that is
something the libraries chose, not something this file asks for, and a mean says the same
thing without resting on it."
  (loop :for category :below *num-categories*
        :collect (/ (loop :for row :from (* category *rows-per-category*)
                            :below (* (1+ category) *rows-per-category*)
                          :sum (row-major-aref predictions row))
                    *rows-per-category*)))

(defun separation-margin (means)
  "Return how far the worst-scored positive category in MEANS -- a `category-means' result --
sits above the best-scored negative one. Negative when the two classes overlap at all.

One number per model, computed only from that model's own scores, so a threshold on it is
never a comparison between backends."
  (let ((positives (loop :for mean :in means
                         :for category :from 0
                         :when (category-positive-p category) :collect mean))
        (negatives (loop :for mean :in means
                         :for category :from 0
                         :unless (category-positive-p category) :collect mean)))
    (- (reduce #'min positives) (reduce #'max negatives))))

;;; ---------------------------------------------------------------------------
;;; The capability this task ships
;;;
;;; Policy section 7 registers `:categorical-features' as a question
;;; `cl-gbdt:backend-supports-p' answers. A NIL answer would say, in that function's own
;;; words, that the feature is unavailable here and the operation signals
;;; `capability-unavailable'.
;;;
;;; The capability assertion and a real categorical run are deliberately in ONE test rather
;;; than two. This project has twice shipped a feature whose capability keyword stayed false
;;; -- `:evaluation-history' and then `:early-stopping' -- because the assertion that the
;;; feature works and the assertion that the backend admits to it lived in different tests and
;;; only the first was written. Tying them together means the capability cannot be forgotten
;;; without the working-feature assertion going with it.
;;;
;;; Tying them together is not by itself enough, and the DEMONSTRATED assertion after the loop
;;; is what closes the remaining hole. Every assertion in this file that says the feature WORKS
;;; is reached only on a backend that answers `:categorical-features' true. Drop the capability
;;; from every backend that has it and all of them simply stop asserting, while
;;; `categorical-features-without-the-capability-signals' starts passing natively on both
;;; backends: green tests and not one assertion about the feature, which is the same
;;; shipped-a-false-capability failure in another disguise. `backend-supports-p' returns
;;; exactly T or NIL (`cl-gbdt/src/backend', `(and (getf ...) t)'), so the per-backend
;;; capability `ok' below is reached only down a path that already proved its own subject and
;;; cannot fail; DEMONSTRATED can, and is the assertion that actually holds the file up.

(defun predictions-with-column-0-categorical (fixture backend)
  "Train on `category-matrix' with :CATEGORICAL-FEATURES *CATEGORICAL-COLUMNS* and return the
predictions, or NIL when BACKEND refused the argument with `capability-unavailable'.

The refusal is turned into NIL rather than a failure because a backend that does not provide
`:categorical-features' is a correct backend --
`categorical-features-without-the-capability-signals' is where that refusal is asserted. A
`capability-unavailable' naming any OTHER capability is re-signalled instead: swallowing it
would let this file report a demonstration as skipped when something unrelated had broken."
  (handler-case (train-on fixture backend (category-matrix)
                          :categorical-features *categorical-columns*)
    (cl-gbdt:capability-unavailable (condition)
      (unless (eq :categorical-features
                  (cl-gbdt:capability-unavailable-capability condition))
        (error condition))
      nil)))

(deftest categorical-features-capability-is-true-where-it-is-demonstrated
  ;; ASKED is every backend whose shared library was actually present, DEMONSTRATED every one
  ;; of those that honoured a categorical column. The two are collected across the loop so the
  ;; assertion after it can be about the SET rather than about any one backend.
  (let ((asked '())
        (demonstrated '()))
    (dolist (fixture *fixtures*)
      (with-backend-library ((getf fixture :backend))
        (push (getf fixture :backend) asked)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               ;; The demonstration runs FIRST and the capability assertion hangs off its
               ;; result, rather than `backend-supports-p' gating the demonstration.
               (let ((categorical (predictions-with-column-0-categorical fixture backend)))
                 (when categorical
                   (push (getf fixture :backend) demonstrated)
                   (let ((plain (train-on fixture backend (category-matrix))))
                     (testing (format nil "~A: a dataset built with :categorical-features ~S ~
                                           trains a different model than one built without it"
                                      (getf fixture :backend) *categorical-columns*)
                       (ok (not (predictions-agree-p categorical plain))
                           (format nil "categorical ~S, plain ~S"
                                   (category-means categorical) (category-means plain)))))
                   (testing (format nil "~A: and backend-supports-p admits to ~
                                         :categorical-features"
                                    (getf fixture :backend))
                     (ok (eq t (cl-gbdt:backend-supports-p backend :categorical-features))
                         (format nil "the capabilities were ~S"
                                 (cl-gbdt:backend-capabilities backend))))))
            (cl-gbdt:close-backend backend)))))
    ;; The one assertion in this file that no capability answer can route around, and the
    ;; reason the two lists above are collected at all -- see this section's header. It says
    ;; the feature is demonstrated SOMEWHERE, which is the claim every capability-gated test
    ;; silently stops making the moment no backend answers true.
    ;;
    ;; Guarded on ASKED, not on any capability: a fresh clone with no vendored library skips
    ;; every backend through `with-backend-library' -- `rove:skip' records a pending assertion
    ;; and returns rather than unwinding -- and demanding a demonstration from a suite that
    ;; ran nothing would turn that documented skip into a failure. Library absence is the only
    ;; thing this excuses.
    (when asked
      (testing "at least one backend with a library present demonstrates :categorical-features"
        (ok demonstrated
            (format nil "asked ~S, demonstrated ~S" (reverse asked) (reverse demonstrated)))))))

;;; The load-bearing test: the column list is HONOURED, not merely accepted.
;;;
;;; Two models on the same 24 rows and the same labels, differing only in whether
;;; :CATEGORICAL-FEATURES named column 0. Two assertions, and each on its own catches an
;;; implementation that takes the argument and drops it:
;;;
;;;   (a) the two arms' predictions DIFFER. Dropped, they would be the same model twice.
;;;   (b) the categorical arm SEPARATES the alternating classes, by *SEPARATING-MARGIN* or
;;;       more. Dropped, its margin would be the quantitative arm's, which is not close.
;;;
;;; (b) is the half the plan behind this feature asked to be measured before being asserted,
;;; because a naive "positive above 0.6, negative below 0.4" test can pass on a big enough
;;; fixture with no categorical column at all: an additive ensemble of depth-1 stumps at
;;; several thresholds approximates an alternating pattern given enough rounds. So the
;;; threshold here is on the MARGIN, and it was measured on both fixture sizes tried before it
;;; was written -- XGBoost, twenty rounds, mean score per category, categories 0 through 5:
;;;
;;;   4 rows per category, plain        0.816 0.406 0.547 0.461 0.556 0.193   margin 0.086
;;;   4 rows per category, categorical  0.974 0.026 0.974 0.026 0.974 0.026   margin 0.948
;;;   8 rows per category, plain        0.825 0.378 0.549 0.443 0.623 0.179   margin 0.106
;;;   8 rows per category, categorical  0.985 0.015 0.985 0.015 0.985 0.015   margin 0.969
;;;
;;; LightGBM, measured at *ROWS-PER-CATEGORY* only, with its own *BOOSTER-PARAMETERS* entry:
;;;
;;;   4 rows per category, plain        0.815 0.473 0.500 0.500 0.527 0.184   margin 0.000
;;;   4 rows per category, categorical  0.934 0.066 0.934 0.066 0.934 0.066   margin 0.869
;;;
;;; The categorical arm answers ONE number per class on both: the split it found is exactly
;;; the set {0, 2, 4}, which is the labelling. The plain arm cannot express that set at all and
;;; lands six different scores whose classes barely separate -- on LightGBM they do not
;;; separate at all, categories 2 and 3 tying at 0.500 for a margin of exactly zero.
;;; *ROWS-PER-CATEGORY* is 4, the size at which XGBoost's plain arm does worst;
;;; *SEPARATING-MARGIN* is 0.5, between the two arms by a wide margin on either size and on
;;; either backend.
;;;
;;; The plain arm's own margin is deliberately NOT asserted to be small. That it stays small
;;; is a fact about how well each library approximates an alternating pattern, not a promise
;;; this wrapper makes, and an assertion on it would fail the day one of them got better at
;;; something. What this file needs from it is that it is not what the categorical arm
;;; produces, and (a) says exactly that.
;;;
;;; No number here is compared with a number from the other backend.

(deftest marking-a-column-categorical-changes-what-was-learned
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :categorical-features)
               (let* ((categorical (train-on fixture backend (category-matrix)
                                             :categorical-features *categorical-columns*))
                      (plain (train-on fixture backend (category-matrix)))
                      (means (category-means categorical)))
                 (testing (format nil "~A: :categorical-features ~S answers something else ~
                                       than the same matrix read as quantities"
                                  (getf fixture :backend) *categorical-columns*)
                   (ok (not (predictions-agree-p categorical plain))
                       (format nil "categorical ~S, plain ~S"
                               means (category-means plain))))
                 (testing (format nil "~A: and separates the alternating classes by at least ~
                                       ~A, which the quantitative reading does not"
                                  (getf fixture :backend) *separating-margin*)
                   (ok (>= (separation-margin means) *separating-margin*)
                       (format nil "categorical margin ~S from ~S, plain margin ~S"
                               (separation-margin means) means
                               (separation-margin (category-means plain)))))))
          (cl-gbdt:close-backend backend))))))

;;; The same comparison over a `cl-gbdt:csr-matrix', which is the other form `make-dataset'
;;; accepts.
;;;
;;; :CATEGORICAL-FEATURES is rendered from the CALLER's matrix on both backends, at a point
;;; where the dense and sparse branches have already converged -- attached to the finished
;;; DMatrix on XGBoost, written into the one parameter string both creation entry points read
;;; on LightGBM. Neither is a second code site, unlike :MISSING, which is a key in two
;;; different creation configs and so needed its sparse case tested separately. What is
;;; genuinely unpinned without this test is not the attachment but the two things around it:
;;; that the renderer reads a `csr-matrix''s DECLARED column count where it reads a dense
;;; matrix's second dimension, and that each library honours a categorical column on a dataset
;;; built from CSR at all.
;;;
;;; Measured the same way as the dense test, and -- `category-csr' storing every element
;;; explicitly -- the same numbers on each backend as that test's, digit for digit:
;;; 0.816 / 0.406 / 0.547 / 0.461 / 0.556 / 0.193 plain and 0.974 / 0.026 / 0.974 / 0.026 /
;;; 0.974 / 0.026 categorical on XGBoost, 0.815 / 0.473 / 0.500 / 0.500 / 0.527 / 0.184 plain
;;; and 0.934 / 0.066 / 0.934 / 0.066 / 0.934 / 0.066 categorical on LightGBM. Gated on
;;; `:sparse-input' as well as `:categorical-features', since a backend without the first has
;;; no `csr-matrix' to build a dataset from at all.

(deftest marking-a-column-categorical-works-on-a-csr-matrix
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
        (unwind-protect
             (when (and (cl-gbdt:backend-supports-p backend :categorical-features)
                        (cl-gbdt:backend-supports-p backend :sparse-input))
               (let* ((csr (category-csr (category-matrix)))
                      (categorical (train-on fixture backend csr
                                             :categorical-features *categorical-columns*))
                      (plain (train-on fixture backend csr))
                      (means (category-means categorical)))
                 (testing (format nil "~A: a csr-matrix dataset built with ~
                                       :categorical-features ~S answers something else than ~
                                       the same csr-matrix read as quantities"
                                  (getf fixture :backend) *categorical-columns*)
                   (ok (not (predictions-agree-p categorical plain))
                       (format nil "categorical ~S, plain ~S"
                               means (category-means plain))))
                 (testing (format nil "~A: and separates the alternating classes by at least ~
                                       ~A" (getf fixture :backend) *separating-margin*)
                   (ok (>= (separation-margin means) *separating-margin*)
                       (format nil "categorical margin ~S from ~S"
                               (separation-margin means) means)))))
          (cl-gbdt:close-backend backend))))))

;;; Policy section 7's central rule: the operation re-checks the capability itself rather than
;;; trusting the caller to have asked `backend-supports-p' first, and signals rather than
;;; quietly ignoring the argument.
;;;
;;; Both backends reach the branch from the same `backend-supports-p' answer rather than from a
;;; hardcoded backend name: one that already answers false is left exactly as it opened, and
;;; one that answers true has its capability plist overwritten the way
;;; tests/functional/missing-value.lisp and tests/functional/sparse-input.lisp overwrite one
;;; for the same purpose -- which is what a library that could not provide the capability would
;;; have produced at `open-backend'.
;;;
;;; `handler-case', not rove's `signals', which does not reliably catch a condition raised
;;; inside `restart-case'; the condition TYPE is asserted, not merely that something signalled.

(deftest categorical-features-without-the-capability-signals
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
            (matrix (category-matrix)))
        (unwind-protect
             (progn
               (when (cl-gbdt:backend-supports-p backend :categorical-features)
                 (setf (cl-gbdt:backend-capabilities backend) '(:categorical-features nil)))
               (testing (format nil "~A: make-dataset signals capability-unavailable for ~
                                     :categorical-features ~S, naming the capability and the ~
                                     backend"
                                (getf fixture :backend) *categorical-columns*)
                 ;; `free-dataset' on the success branch, the way
                 ;; `missing-value-without-the-capability-signals' frees the dataset it does
                 ;; not expect to get: this branch is only reached if the gate has regressed,
                 ;; and a leaked handle is a poor second failure to hand whoever is already
                 ;; reading the first one.
                 (let ((condition
                         (handler-case
                             (progn (cl-gbdt:free-dataset
                                     (cl-gbdt:make-dataset
                                      backend matrix
                                      :categorical-features *categorical-columns*))
                                    nil)
                           (cl-gbdt:capability-unavailable (c) c))))
                   (ok condition "make-dataset signalled instead of building a dataset")
                   (ok (and condition
                            (eq :categorical-features
                                (cl-gbdt:capability-unavailable-capability condition)))
                       (format nil "the condition named capability ~S"
                               (and condition
                                    (cl-gbdt:capability-unavailable-capability condition))))
                   (ok (and condition
                            (eq (getf fixture :backend)
                                (cl-gbdt:backend-error-backend condition)))
                       (format nil "the condition named backend ~S"
                               (and condition
                                    (cl-gbdt:backend-error-backend condition)))))))
          (cl-gbdt:close-backend backend))))))

;;; The renderer's own rejection -- `cl-gbdt/src/config/categorical-features''s
;;; `%check-indices', whose direct call is covered by tests/categorical-features.lisp --
;;; reached instead through the public API, which is a different path: it pins that the list is
;;; judged at all rather than being rendered into whatever the library would make of it, that
;;; it is judged against the CALLER's matrix rather than against some count obtained later, and
;;; that the condition blames the backend the caller was talking to.
;;;
;;; Three ways of being wrong, one per rejection the renderer makes for a reason a caller can
;;; hit: a column named as a string rather than an integer, a negative index, and an index one
;;; past the matrix's last column -- the boundary, since an off-by-one range check would accept
;;; exactly that one and no other.
;;;
;;; Only on a backend whose capability is true. On one whose capability is false the gate fires
;;; first, by design -- `categorical-features-without-the-capability-signals' above asserts
;;; exactly that -- so there would be nothing here for the renderer to reject.

(deftest a-bad-categorical-index-signals
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let* ((backend (cl-gbdt:open-backend (getf fixture :backend)))
             (matrix (category-matrix))
             (past-the-end (array-dimension matrix 1)))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :categorical-features)
               (dolist (indices (list '("0") '(-1) (list past-the-end)))
                 (testing (format nil "~A: make-dataset signals unsupported-argument for ~
                                       :categorical-features ~S, naming the argument and the ~
                                       backend"
                                  (getf fixture :backend) indices)
                   (let ((condition
                           (handler-case
                               (progn (cl-gbdt:free-dataset
                                       (cl-gbdt:make-dataset backend matrix
                                                             :categorical-features indices))
                                      nil)
                             (cl-gbdt:unsupported-argument (c) c))))
                     (ok condition "make-dataset signalled instead of building a dataset")
                     (ok (and condition
                              (equal ":categorical-features"
                                     (cl-gbdt:unsupported-argument-argument condition)))
                         (format nil "the condition named argument ~S"
                                 (and condition
                                      (cl-gbdt:unsupported-argument-argument condition))))
                     (ok (and condition
                              (eq (getf fixture :backend)
                                  (cl-gbdt:unsupported-argument-backend condition)))
                         (format nil "the condition named backend ~S"
                                 (and condition
                                      (cl-gbdt:unsupported-argument-backend condition))))))))
          (cl-gbdt:close-backend backend))))))

;;; ---------------------------------------------------------------------------
;;; The key this feature shares with :PARAMETERS, refused when BOTH are given
;;;
;;; LightGBM's own name for the column list is a key in the parameter string rather than a C
;;; argument, so :CATEGORICAL-FEATURES and :PARAMETERS reach the library through the SAME
;;; channel there -- unlike XGBoost, where the list is a field attached to a finished DMatrix
;;; and :PARAMETERS is refused outright. A caller who supplies BOTH is stating the same thing
;;; twice in one string, and which copy the library reads is measured: LightGBM takes the
;;; FIRST occurrence of a duplicated key -- `categorical_feature=0 categorical_feature=1'
;;; trains exactly what `categorical_feature=0' alone trains -- while the wrapper appends its
;;; own entry LAST. So it is the wrapper's copy, rendered from the :CATEGORICAL-FEATURES the
;;; caller explicitly named, that would silently lose. Refused, not merged and not
;;; overwritten.
;;;
;;; :PARAMETERS ALONE is untouched, and the last assertion below is what pins that. Writing
;;; `categorical_feature' there by hand, and naming no categorical column, is policy section
;;; 6's escape hatch for a backend's own vocabulary being used exactly as it always was --
;;; there is no second entry for it to collide with, so there is nothing to refuse. Every such
;;; call built a dataset before :CATEGORICAL-FEATURES existed and still does.
;;;
;;; Which spellings must be refused is a MEASUREMENT, not a guess: LightGBM's aliases appear
;;; in no vendored header. Measured against the vendored 4.7.0 on this file's own fixture and
;;; *BOOSTER-PARAMETERS*, twenty rounds, mean score per category -- an honoured spelling
;;; answers 0.934 / 0.066 / 0.934 / 0.066 / 0.934 / 0.066, margin 0.869, and an inert one
;;; reproduces the no-key row (0.815 / 0.473 / 0.500 / 0.500 / 0.527 / 0.184, margin 0.000)
;;; digit for digit:
;;;
;;;   categorical_feature    honoured     kategorical_feature   inert (control)
;;;   cat_feature            honoured     categorical_feat      inert (control)
;;;   categorical_column     honoured     cat_features          inert (control)
;;;   cat_column             honoured
;;;   categorical_features   honoured
;;;
;;; `cat_features' is why the last assertion below exists. It is the PLURAL of `cat_feature',
;;; which is honoured, and the library still ignores it -- so a refusal written as a prefix
;;; test, or as any fuzzy match, would reject a key that must keep reaching the library as the
;;; ordinary backend parameter it is. The list is exact.

(defparameter *refused-parameter-keys*
  '(:categorical-feature :cat-feature :categorical-column :cat-column :categorical-features
    "categorical_feature")
  "The :PARAMETERS keys `make-dataset' must refuse on a backend that spells its categorical
column list as a parameter, one per honoured alias plus one written the other way a caller
can write it.

Restated here rather than read from the implementation's own list, deliberately: a test that
asked the implementation which keys it refuses would pass for any list at all, including an
empty one. This list is the measurement above, written down twice on purpose.

The last entry is the string `\"categorical_feature\"' rather than a sixth alias -- the same
key a caller can equally write as the keyword `:categorical-feature', since
`cl-gbdt:normalize-parameters' downcases a key and turns its dashes into underscores. It
pins that the refusal compares against that normalized form rather than against the keyword
the caller happened to type.")

(defparameter *accepted-parameter-key* :cat-features
  "A :PARAMETERS key that must keep reaching the library: the plural of `cat_feature', which
LightGBM does NOT honour as an alias -- measured, see above. It is ordinary
backend-parameter territory, and a refusal that caught it would break a caller passing an
unrelated key.")

(defparameter *categorical-parameter-value* "0"
  "The value every :PARAMETERS key above is given: column 0, the categorical column of
`category-matrix', spelled the way LightGBM's parameter string spells a column list.

The value is never what the refusal is about -- the key alone decides -- so one value serves
the refused keys and the accepted one alike, and using the same one for both keeps the two
halves of `the-parameters-key-is-refused' differing only in the key.")

(deftest the-parameters-key-is-refused
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
            (matrix (category-matrix)))
        (unwind-protect
             ;; Asked only of a backend this file builds datasets with :PARAMETERS on, read
             ;; from *FIXTURES* rather than from a backend name. XGBoost's entry there is NIL
             ;; because that backend refuses :PARAMETERS outright, which would make every
             ;; assertion below -- the accepted ones included -- signal for a reason that has
             ;; nothing to do with categorical columns.
             (let ((base (getf fixture :dataset-parameters)))
               (when base
                 (flet ((refusal-for (key &key categorical-features)
                          "Return the `unsupported-argument' `make-dataset' signals for KEY
added to BASE -- with CATEGORICAL-FEATURES named as well when it is non-NIL -- or NIL when it
built a dataset instead, which is then freed rather than leaked.

CATEGORICAL-FEATURES NIL is passed as nothing at all rather than as `:categorical-features
nil', for the reason `make-categorical-dataset' gives: it is what makes the two halves of this
test differ in exactly the way a caller's two calls would."
                          (handler-case
                              (progn (cl-gbdt:free-dataset
                                      (apply #'cl-gbdt:make-dataset
                                             backend matrix
                                             :parameters
                                             (append base
                                                     (list key
                                                           *categorical-parameter-value*))
                                             (when categorical-features
                                               (list :categorical-features
                                                     categorical-features))))
                                     nil)
                            (cl-gbdt:unsupported-argument (c) c))))
                   (dolist (key *refused-parameter-keys*)
                     (testing (format nil "~A: make-dataset signals unsupported-argument for ~
                                           :parameters ~S alongside :categorical-features ~S, ~
                                           naming the argument and the backend"
                                      (getf fixture :backend) key *categorical-columns*)
                       (let ((condition (refusal-for
                                         key :categorical-features *categorical-columns*)))
                         (ok condition "make-dataset signalled instead of building a dataset")
                         (ok (and condition
                                  (equal "make-dataset's :parameters"
                                         (cl-gbdt:unsupported-argument-argument condition)))
                             (format nil "the condition named argument ~S"
                                     (and condition
                                          (cl-gbdt:unsupported-argument-argument condition))))
                         (ok (and condition
                                  (eq (getf fixture :backend)
                                      (cl-gbdt:unsupported-argument-backend condition)))
                             (format nil "the condition named backend ~S"
                                     (and condition
                                          (cl-gbdt:unsupported-argument-backend condition)))))))
                   (testing (format nil "~A: and does not refuse :parameters ~S alongside ~
                                         :categorical-features ~S, the library not honouring ~
                                         it as an alias"
                                    (getf fixture :backend) *accepted-parameter-key*
                                    *categorical-columns*)
                     (let ((condition (refusal-for *accepted-parameter-key*
                                                   :categorical-features
                                                   *categorical-columns*)))
                       (ok (null condition)
                           (if condition
                               (format nil "make-dataset signalled about ~S instead of ~
                                            building a dataset"
                                       (cl-gbdt:unsupported-argument-argument condition))
                               "make-dataset built a dataset"))))
                   ;; The other half of the rule, and the one an over-eager refusal breaks: a
                   ;; caller who names NO categorical column is on the :PARAMETERS escape hatch
                   ;; and must keep working. Every refused spelling is asked, not just the
                   ;; canonical one, since a gate written per-key rather than once around them
                   ;; all would pass for whichever one happened to be tested.
                   (dolist (key *refused-parameter-keys*)
                     (testing (format nil "~A: and does not refuse :parameters ~S when no ~
                                           :categorical-features is named at all"
                                      (getf fixture :backend) key)
                       (let ((condition (refusal-for key)))
                         (ok (null condition)
                             (if condition
                                 (format nil "make-dataset signalled about ~S instead of ~
                                              building a dataset"
                                         (cl-gbdt:unsupported-argument-argument condition))
                                 "make-dataset built a dataset"))))))))
          (cl-gbdt:close-backend backend))))))
