;;;; prediction-shape.lisp --- Portable contract tests for `predict''s SECOND return value.
;;;;
;;;; `cl-gbdt:predict' now returns two values: the result array it has always returned, and
;;;; the SHAPE the backend library reports for that result -- a list of integers in
;;;; `array-dimensions' order, or NIL where the backend states none. XGBoost writes an
;;;; `out_shape'/`out_dim' pair on every prediction, and this branch reads it back; before it,
;;;; the wrapper read only their product and folded every result into [rows, total/rows],
;;;; throwing away structure the library had already stated.
;;;;
;;;; The capability is `:prediction-shape', and unlike every other capability this suite
;;;; exercises, NO OPERATION REFUSES ON IT. There is no argument asking for a shape, so a
;;;; backend that answers false returns NIL as its second value rather than signalling --
;;;; which is why there is no `...-without-the-capability-signals' test here to match the ones
;;;; in tests/functional/missing-value.lisp and tests/functional/sparse-input.lisp. A check
;;;; added for symmetry with those would break `cl-gbdt:predict' outright on a backend that
;;;; simply has less to say.
;;;;
;;;; Like tests/functional/evaluation.lisp, sparse-input.lisp, missing-value.lisp and
;;;; categorical-features.lisp beside it, the two backend-neutral tests below run over that
;;;; first file's *FIXTURES*, once per backend, so the two backends cannot drift apart in
;;;; shape or meaning without one of them failing here. The two remaining tests name XGBoost
;;;; outright, the way `sparse-prediction-kind-support-is-what-each-library-offers' names both
;;;; backends in sparse-input.lisp: a shape read from a library is what THAT library reports,
;;;; and weakening the assertion into something every backend can satisfy would test nothing
;;;; on any of them.
;;;;
;;;; Numbers are never compared BETWEEN backends: policy section 13 asks for shape, order and
;;;; meaning, not numeric agreement, and the two libraries train different models from the
;;;; same rows by design. Nothing below compares a prediction on one backend with a prediction
;;;; on the other. `folded-width' is stated once for both backends, but what it states is a
;;;; count of columns rather than a prediction -- see its own docstring.

(uiop:define-package #:cl-gbdt/tests/functional/prediction-shape
  ;; Zero symbols: every reference below is package-qualified. Declared so this file's
  ;; dependency on the unified API is explicit rather than inherited, matching the identical
  ;; clause in evaluation.lisp, sparse-input.lisp, missing-value.lisp and
  ;; categorical-features.lisp.
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt)
  ;; Zero symbols, both of them: their only job is to run at load time and register
  ;; :lightgbm and :xgboost with `open-backend'. Without these clauses,
  ;; package-inferred-system has no edge to those files and `(cl-gbdt:open-backend
  ;; :lightgbm)' below would signal `unknown-backend'.
  (:import-from #:cl-gbdt/src/lightgbm/all)
  (:import-from #:cl-gbdt/src/xgboost/all)
  ;; `dense-to-csr' stores every element of the matrix it converts, zeros included. What this
  ;; file needs that for: `the-first-value-is-unchanged' asserts the SAME dimensions on the
  ;; sparse path as on the dense one, and an entry a `csr-matrix' does not store is MISSING to
  ;; XGBoost whatever any config says -- so a conversion that dropped zeros would be
  ;; describing different rows to that backend on the two paths.
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library
                #:make-multiclass-dataset
                #:make-separable-dataset
                #:predictions-match-labels-p
                #:dense-to-csr)
  ;; The fixture table and its dataset builder come from evaluation.lisp rather than being
  ;; restated here, for the reason that file's own export comment gives: a second table saying
  ;; the same thing in its own words is how two files that must agree stop agreeing.
  ;; `make-fixture-dataset' is what passes LightGBM the dataset parameters it needs to bin and
  ;; split this suite's small fixtures at all, and passes XGBoost none, which is exactly what
  ;; both models below need. Only the BOOSTER parameters are this file's own, in
  ;; *MULTICLASS-BOOSTER-PARAMETERS* below -- *FIXTURES* carries no multiclass entry.
  (:import-from #:cl-gbdt/tests/functional/evaluation
                #:*fixtures*
                #:make-fixture-dataset))

(in-package #:cl-gbdt/tests/functional/prediction-shape)

(defparameter *kinds* '(:normal :raw :leaf-index :contrib)
  "Every KIND `cl-gbdt:predict' accepts, in the order its generic function's docstring lists
them. Every test below that walks a set of kinds walks this one, so a fifth kind added to the
protocol without a shape of its own starts failing here rather than going unasserted.")

(defparameter *num-classes* 3
  "How many classes the multiclass fixture below holds, and how many output groups the model
trained on it therefore has.

Must equal the `:num-class' in both entries of *MULTICLASS-BOOSTER-PARAMETERS*, which spell it
as a literal in each library's own vocabulary; nothing ties the three together but this
sentence, exactly as in `cl-gbdt/tests/functional/xgboost-api''s own multiclass fixture.")

(defparameter *num-columns* 4
  "How many columns the multiclass fixture below has -- `make-multiclass-dataset''s COLS, four
rather than its default three.

Load-bearing, and the reason this file does not simply take the defaults. The pre-branch
`cl-gbdt:predict' folded `:leaf-index' into a width of rounds x classes and `:contrib' into
one of classes x (features + 1), so the two are THE SAME NUMBER whenever rounds = features +
1. Measured at the default three columns and this file's four rounds: both are 12, and a test
that confused one shape for the other would still pass. At four columns they are 12 and 15 --
measured too, and on both backends. See *TRAINING-ROUNDS*, the other half of the same choice.")

(defparameter *training-rounds* 4
  "How many boosting rounds every booster below is trained for.

Four, and chosen against *NUM-COLUMNS* rather than copied from a neighbouring file: the width
collision that parameter describes is at rounds = features + 1, so five rounds -- what
tests/functional/missing-value.lisp uses -- would put this file's four-column fixture right
back into it at 15 and 15. Four rounds against four columns gives 12 and 15.

It is also enough for both backends to recover every label of the multiclass fixture, measured
before `the-first-value-is-unchanged' was given its `predictions-match-labels-p' assertion.")

(defparameter *multiclass-booster-parameters*
  '((:lightgbm :objective "multiclass" :num-class 3 :num-leaves 2 :min-data-in-leaf 1
     :min-data-in-bin 1 :verbose -1)
    (:xgboost :objective "multi:softprob" :num-class 3 :max-depth 3 :eta 0.5 :verbosity 0))
  "Backend name to the booster parameters that train a *NUM-CLASSES*-class model on it.

This file's own table rather than an import, because *FIXTURES* carries only a BINARY booster
for each backend and a binary model is the one case that would let this feature look
multiclass-only -- see `a-binary-model-also-reports-more-than-two-dimensions'. Everything else
each backend needs, the dataset parameters included, still comes from *FIXTURES* through
`make-fixture-dataset'.

As with that table, the two plists are not translations of each other: each is the parameter
vocabulary its own library documents. `:num-class' must equal *NUM-CLASSES* in both.")

(defun multiclass-parameters (backend-name)
  "Return BACKEND-NAME's entry in *MULTICLASS-BOOSTER-PARAMETERS*.

Signals an error for a backend the table does not name, rather than returning NIL and leaving
`cl-gbdt:train' to whatever objective that library defaults to -- every shape assertion below
would then be about a model nobody chose, and *NUM-CLASSES* would describe none of it."
  (let ((entry (assoc backend-name *multiclass-booster-parameters*)))
    (unless entry
      (error "No multiclass booster parameters for backend ~S." backend-name))
    (rest entry)))

(defun folded-width (kind num-classes num-features)
  "Return the second dimension the PRE-BRANCH `cl-gbdt:predict' gave KIND's result: the
reported shape's total element count divided by the row count, which is all either backend did
with that shape before this branch.

Measured at cefa72c, the commit this task's work starts from, over the nine-row four-column
three-class fixture below at *TRAINING-ROUNDS* rounds -- `:normal' 3, `:raw' 3, `:leaf-index'
12, `:contrib' 15 -- and the same four numbers on BOTH backends, dense and, for the kinds each
library's sparse entry point serves, over a `dense-to-csr' `csr-matrix' too. That agreement is
why one function serves both backends here.

It is not a number compared between backends in policy section 13's sense: what agrees is a
count of columns, not a prediction. No assertion in this file weighs one backend's predictions
against the other's.

NUM-CLASSES is the model's output-group count -- *NUM-CLASSES* for the multiclass fixture, 1
for a binary one -- and NUM-FEATURES the matrix's own column count."
  (ecase kind
    ((:normal :raw) num-classes)
    (:leaf-index (* *training-rounds* num-classes))
    (:contrib (* num-classes (1+ num-features)))))

(defun reported-shape (kind rows num-classes num-features)
  "Return the shape XGBoost reports for KIND, over ROWS rows of NUM-FEATURES columns from a
model with NUM-CLASSES output groups, as a list of integers in `array-dimensions' order.

Read from `out_shape'/`out_dim' directly, at *TRAINING-ROUNDS* rounds against the vendored
library, before any assertion in this file was written:

  kind          multiclass (9 rows, 4 columns, 3 classes)   binary (8 rows, 3 columns)
  :normal       (9 3)                                       (8 1)
  :raw          (9 3)                                       (8 1)
  :leaf-index   (9 4 3 1)                                   (8 4 1 1)
  :contrib      (9 3 5)                                     (8 1 4)

`:leaf-index''s axes are rows, iterations, output groups and parallel trees -- the last is 1
because XGBoost's `num_parallel_tree' defaults to 1 and nothing here changes it. `:contrib''s
are rows, output groups, and one column per feature plus a bias term.

A binary model has ONE output group, and its two right-hand shapes are four- and
three-dimensional all the same: this is the measurement that stops a reader taking the feature
for a multiclass-only one. It is also why one function covers both columns of the table --
NUM-CLASSES 1 reproduces the binary column exactly."
  (ecase kind
    ((:normal :raw) (list rows num-classes))
    (:leaf-index (list rows *training-rounds* num-classes 1))
    (:contrib (list rows num-classes (1+ num-features)))))

(defun multiclass-fixture ()
  "Return two values: this file's multiclass feature matrix and its labels."
  (make-multiclass-dataset :num-classes *num-classes* :cols *num-columns*))

(defun train-multiclass (fixture backend matrix labels)
  "Train a *NUM-CLASSES*-class booster on BACKEND from MATRIX and LABELS, and return it.

The dataset comes from FIXTURE's own `make-fixture-dataset', so each backend gets exactly the
`make-dataset' keywords it accepts; the booster parameters come from
*MULTICLASS-BOOSTER-PARAMETERS*. The caller owns the booster and wraps this in
`cl-gbdt:with-booster'; the dataset does not outlive this call, `cl-gbdt:predict' taking a
matrix and never a dataset."
  (cl-gbdt:with-dataset (dataset (make-fixture-dataset fixture backend matrix labels))
    (cl-gbdt:train backend dataset :num-rounds *training-rounds*
                   :parameters (multiclass-parameters (getf fixture :backend)))))

(defun shape-describes-p (shape result)
  "True when SHAPE -- `cl-gbdt:predict''s second value -- is a list of positive integers whose
product is RESULT's total size and whose first entry is RESULT's row count.

The weakest thing worth asserting about a reported shape without naming a backend: it has to
describe the very result it came back with, in `array-dimensions' order, or it is not a shape
of that result at all. What each library reports beyond this is that library's own, and is
asserted by name in `xgboost-reports-the-library-s-own-shape' below."
  (and (listp shape)
       shape
       (every (lambda (axis) (and (integerp axis) (plusp axis))) shape)
       (= (reduce #'* shape) (array-total-size result))
       (= (first shape) (array-dimension result 0))))

;;; ---------------------------------------------------------------------------
;;; The capability this task ships
;;;
;;; Policy section 7 registers `:prediction-shape' as a question `cl-gbdt:backend-supports-p'
;;; answers: whether this backend states the shape of a prediction it just made. It is the one
;;; capability in this suite that NO operation re-checks -- see this file's header for why a
;;; check added for symmetry would be a regression rather than a tightening.
;;;
;;; The capability assertion and a real non-NIL shape are deliberately in ONE test rather than
;;; two. This project has twice shipped a feature whose capability keyword stayed false --
;;; `:evaluation-history' and then `:early-stopping' -- because the assertion that the feature
;;; works and the assertion that the backend admits to it lived in different tests and only the
;;; first was written.
;;;
;;; The DEMONSTRATED assertion after the loop closes the remaining hole, the one the
;;; missing-value review found: every per-backend assertion here is reached only on a backend
;;; that reported a shape, so dropping the capability from every backend that has it would
;;; leave this test green with nothing asserted. DEMONSTRATED is the assertion no capability
;;; answer can route around. It is guarded on ASKED, not on any capability: a fresh clone with
;;; no vendored library skips every backend through `with-backend-library' -- `rove:skip'
;;; records a pending assertion and returns rather than unwinding -- and demanding a
;;; demonstration from a suite that ran nothing would turn that documented skip into a failure.

(deftest prediction-shape-capability-is-true-where-it-is-demonstrated
  ;; ASKED is every backend whose shared library was actually present, DEMONSTRATED every one
  ;; of those that reported a shape for at least one kind. The two are collected across the
  ;; loop so the assertion after it can be about the SET rather than about any one backend.
  (let ((asked '())
        (demonstrated '()))
    (dolist (fixture *fixtures*)
      (with-backend-library ((getf fixture :backend))
        (push (getf fixture :backend) asked)
        (multiple-value-bind (matrix labels) (multiclass-fixture)
          (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
            (unwind-protect
                 (cl-gbdt:with-booster (booster (train-multiclass fixture backend matrix labels))
                   ;; The demonstration runs FIRST and the capability assertion hangs off its
                   ;; result, rather than `backend-supports-p' gating the demonstration.
                   (let ((reported
                           (loop :for kind :in *kinds*
                                 :for (result shape)
                                   := (multiple-value-list
                                       (cl-gbdt:predict booster matrix :kind kind))
                                 :when shape :collect (list kind result shape))))
                     (when reported
                       (push (getf fixture :backend) demonstrated)
                       (testing (format nil "~A: every shape predict reported describes the ~
                                             result it came back with"
                                        (getf fixture :backend))
                         (ok (every (lambda (entry)
                                      (destructuring-bind (kind result shape) entry
                                        (declare (ignore kind))
                                        (shape-describes-p shape result)))
                                    reported)
                             (format nil "kind, dimensions and reported shape: ~S"
                                     (mapcar (lambda (entry)
                                               (destructuring-bind (kind result shape) entry
                                                 (list kind (array-dimensions result) shape)))
                                             reported))))
                       (testing (format nil "~A: and backend-supports-p admits to ~
                                             :prediction-shape"
                                        (getf fixture :backend))
                         (ok (eq t (cl-gbdt:backend-supports-p backend :prediction-shape))
                             (format nil "the capabilities were ~S"
                                     (cl-gbdt:backend-capabilities backend)))))))
              (cl-gbdt:close-backend backend))))))
    (when asked
      (testing "at least one backend with a library present demonstrates :prediction-shape"
        (ok demonstrated
            (format nil "asked ~S, demonstrated ~S" (reverse asked) (reverse demonstrated)))))))

;;; ---------------------------------------------------------------------------
;;; The load-bearing test: the FIRST value did not move
;;;
;;; Policy section 14 is this branch's hardest constraint -- `cl-gbdt:predict''s first return
;;; value is what it always was, same dimensions and same elements, for every kind, dense and
;;; sparse. A second return value is worth nothing if adding it moved the first.
;;;
;;; The dimensions are asserted against `folded-width' -- numbers derived from the fixture's
;;; own dimensions and the kind, measured at cefa72c, the commit this task started from -- and
;;; NOT by calling `cl-gbdt:predict' twice and comparing it with itself, which would compare
;;; this task's code against this task's code and pass however wrong both were.
;;;
;;; Measured at cefa72c, over the nine-row four-column three-class fixture at four rounds, on
;;; both backends, dense; and the same numbers over its `dense-to-csr' form for the kinds each
;;; library's sparse entry point serves:
;;;
;;;   :normal (9 3)   :raw (9 3)   :leaf-index (9 12)   :contrib (9 15)
;;;
;;; The elements are covered by `predictions-match-labels-p' on the `:normal' result: it pins
;;; that the numbers still line up with their rows and columns, which a shape read back with
;;; the wrong stride would break. It does not pin exact values: every element of every result
;;; here was diffed against a snapshot taken at cefa72c when the change was made -- identical,
;;; both backends, all four kinds, dense and sparse -- and exact-value assertions are what this
;;; suite avoids everywhere else, since they break on any upstream version bump without telling
;;; anyone anything new.
;;;
;;; The sparse arm treats a `foreign-call-error' as "this library's sparse entry point does not
;;; serve this kind" rather than as a failure: measured, XGBoost's inplace prediction refuses
;;; `:leaf-index' and `:contrib' outright and LightGBM's CSR entry point serves all four. That
;;; refusal is already asserted, as a refusal, by
;;; `sparse-prediction-kind-support-is-what-each-library-offers' in
;;; tests/functional/sparse-input.lisp; there is nothing for this file to add to it, and the
;;; SERVED assertion after the inner loop is what stops the arm quietly shrinking to nothing.

(deftest the-first-value-is-unchanged
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (multiple-value-bind (matrix labels) (multiclass-fixture)
        (let ((name (getf fixture :backend))
              (backend (cl-gbdt:open-backend (getf fixture :backend)))
              (rows (array-dimension matrix 0))
              (features (array-dimension matrix 1)))
          (unwind-protect
               (cl-gbdt:with-booster (booster (train-multiclass fixture backend matrix labels))
                 (dolist (kind *kinds*)
                   (let ((result (cl-gbdt:predict booster matrix :kind kind))
                         (expected (list rows (folded-width kind *num-classes* features))))
                     (testing (format nil "~A: :kind ~S still returns an array of ~S"
                                      name kind expected)
                       (ok (equal expected (array-dimensions result))
                           (format nil "dimensions were ~S" (array-dimensions result))))))
                 (testing (format nil "~A: and :kind :normal's elements still pick out the ~
                                       right class per row" name)
                   (let ((result (cl-gbdt:predict booster matrix :kind :normal)))
                     (ok (predictions-match-labels-p result labels)
                         (format nil "predictions: ~S" result))))
                 (when (cl-gbdt:backend-supports-p backend :sparse-input)
                   (let ((csr (dense-to-csr matrix))
                         (served '()))
                     (dolist (kind *kinds*)
                       (let ((result (handler-case (cl-gbdt:predict booster csr :kind kind)
                                       (cl-gbdt:foreign-call-error () nil))))
                         (when result
                           (push kind served)
                           (let ((expected
                                   (list rows (folded-width kind *num-classes* features))))
                             (testing (format nil "~A: :kind ~S on a csr-matrix still returns ~
                                                   an array of ~S" name kind expected)
                               (ok (equal expected (array-dimensions result))
                                   (format nil "dimensions were ~S"
                                           (array-dimensions result))))))))
                     (testing (format nil "~A: its sparse entry point served at least one kind"
                                      name)
                       (ok served (format nil "kinds served: ~S" (reverse served)))))))
            (cl-gbdt:close-backend backend)))))))

;;; ---------------------------------------------------------------------------
;;; What XGBoost itself reports
;;;
;;; Named for that backend, and not weakened into something every backend can satisfy, for the
;;; reason this file's header gives. The shapes are `reported-shape''s own -- read from
;;; `out_shape'/`out_dim' before these assertions existed -- and the point of asserting them
;;; rather than only `shape-describes-p' is that two of the four are RICHER than the result
;;; array's own dimensions: `:leaf-index' comes back (9 4 3 1) where the array is 9 x 12, and
;;; `:contrib' (9 3 5) where the array is 9 x 15. That extra structure is the whole feature.
;;;
;;; The `csr-matrix' half covers the other of the two branches `predict' has to read a shape
;;; from -- `XGBoosterPredictFromCSR', a different C entry point from
;;; `XGBoosterPredictFromDMatrix' -- so a shape read on one path and not the other fails here.
;;; Only `:normal' and `:raw' arise there, that entry point refusing the other two kinds; this
;;; file takes the same measured list sparse-input.lisp asserts and does not re-derive it.

(deftest xgboost-reports-the-library-s-own-shape
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix labels) (multiclass-fixture)
      (let ((fixture (find :xgboost *fixtures* :key (lambda (entry) (getf entry :backend))))
            (backend (cl-gbdt:open-backend :xgboost))
            (rows (array-dimension matrix 0))
            (features (array-dimension matrix 1)))
        (unwind-protect
             (cl-gbdt:with-booster (booster (train-multiclass fixture backend matrix labels))
               (dolist (kind *kinds*)
                 (let ((expected (reported-shape kind rows *num-classes* features))
                       (shape (nth-value 1 (cl-gbdt:predict booster matrix :kind kind))))
                   (testing (format nil "xgboost: :kind ~S reports ~S" kind expected)
                     (ok (equal expected shape) (format nil "reported ~S" shape)))))
               (let ((csr (dense-to-csr matrix)))
                 (dolist (kind '(:normal :raw))
                   (let ((expected (reported-shape kind rows *num-classes* features))
                         (shape (nth-value 1 (cl-gbdt:predict booster csr :kind kind))))
                     (testing (format nil "xgboost: :kind ~S on a csr-matrix reports ~S"
                                      kind expected)
                       (ok (equal expected shape) (format nil "reported ~S" shape)))))))
          (cl-gbdt:close-backend backend))))))

;;; ---------------------------------------------------------------------------
;;; And the same on a binary model
;;;
;;; The case a reader guesses wrong. `:leaf-index' and `:contrib' are MULTIDIMENSIONAL on a
;;; binary model too -- (8 4 1 1) and (8 1 4) over `make-separable-dataset''s eight rows of
;;; three columns at four rounds -- so the extra structure this feature returns is not
;;; something only a multiclass objective produces.
;;;
;;; This fixture is also where the two FOLDED widths collide: three features at four rounds
;;; gives `:leaf-index' 4 x 1 = 4 and `:contrib' 1 x (3 + 1) = 4, the same number, and the
;;; pre-branch `cl-gbdt:predict' returned an 8 x 4 array for both. That collision is the
;;; opposite of *NUM-COLUMNS*' problem and is welcome here: it is precisely a pair of results
;;; the first return value cannot tell apart and the second can.

(deftest a-binary-model-also-reports-more-than-two-dimensions
  (with-backend-library (:xgboost)
    (multiple-value-bind (matrix labels) (make-separable-dataset)
      (let ((fixture (find :xgboost *fixtures* :key (lambda (entry) (getf entry :backend))))
            (backend (cl-gbdt:open-backend :xgboost))
            (rows (array-dimension matrix 0))
            (features (array-dimension matrix 1)))
        (unwind-protect
             (cl-gbdt:with-dataset (dataset (make-fixture-dataset fixture backend matrix labels))
               (cl-gbdt:with-booster
                   (booster (cl-gbdt:train backend dataset :num-rounds *training-rounds*
                                           :parameters (getf fixture :booster-parameters)))
                 (dolist (kind '(:leaf-index :contrib))
                   (multiple-value-bind (result shape)
                       (cl-gbdt:predict booster matrix :kind kind)
                     ;; One output group, not *NUM-CLASSES*: this is a binary objective.
                     (let ((expected (reported-shape kind rows 1 features)))
                       (testing (format nil "xgboost: a binary model's :kind ~S reports ~S, ~
                                             where the array is ~S" kind expected
                                             (array-dimensions result))
                         (ok (equal expected shape) (format nil "reported ~S" shape))
                         (ok (> (length shape) 2)
                             (format nil "~D dimensions reported for an array of ~D"
                                     (length shape)
                                     (length (array-dimensions result))))))))))
          (cl-gbdt:close-backend backend))))))
