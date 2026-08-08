;;;; missing-value.lisp --- Portable contract tests for `make-dataset''s and `predict''s
;;;; :MISSING.
;;;;
;;;; `cl-gbdt:make-dataset' and `cl-gbdt:predict' both now take a :MISSING keyword naming which
;;;; datum in the caller's own data means *missing*, gated by the `:missing-value' capability.
;;;; It is a VALUE, not a policy: it chooses which number means missing and does not turn
;;;; missing handling on or off. Like tests/functional/evaluation.lisp and
;;;; tests/functional/sparse-input.lisp beside
;;;; it, every test below runs over that first file's *FIXTURES*, once per backend, so the two
;;;; backends cannot drift apart in shape or meaning without one of them failing here.
;;;;
;;;; The two arguments are checked and rendered separately -- `make-dataset' puts the sentinel
;;;; in a DATASET's creation config, `predict' in a transient DMatrix's creation config for a
;;;; dense matrix and in the inplace predict config for a `csr-matrix' -- so each has its own
;;;; capability test below rather than one standing in for the other.
;;;;
;;;; Which backend takes which branch is read from `cl-gbdt:backend-supports-p', never
;;;; hardcoded. XGBoost answers true today and LightGBM false -- LightGBM's C API has no
;;;; `missing' key at all, and its `use_missing'/`zero_as_missing' are parameter-string flags
;;;; reachable through :PARAMETERS, not a sentinel -- but a backend that later gains the
;;;; capability starts being asked the honouring tests rather than silently staying in the
;;;; signalling set.
;;;;
;;;; Numbers are never compared BETWEEN backends: policy section 13 asks for shape, order and
;;;; meaning, not numeric agreement, and the two libraries train different models from the same
;;;; rows by design. Every comparison below is sentinel-versus-no-sentinel, or
;;;; sentinel-versus-NaN, WITHIN one backend.

(uiop:define-package #:cl-gbdt/tests/functional/missing-value
  ;; Zero symbols: every reference below is package-qualified. Declared so this file's
  ;; dependency on the unified API is explicit rather than inherited, matching the identical
  ;; clause in evaluation.lisp and sparse-input.lisp.
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
  ;; The fixture table and its dataset builder come from evaluation.lisp rather than being
  ;; restated here, for the reason that file's own export comment gives: a second table saying
  ;; the same thing in its own words is how two files that must agree stop agreeing.
  ;; `make-fixture-dataset' passes MATRIX and the backend's own parameters straight through to
  ;; `cl-gbdt:make-dataset'; :MISSING is supplied on top of it by `booster-trained-on' below,
  ;; since it is this file's subject and not part of any fixture.
  (:import-from #:cl-gbdt/tests/functional/evaluation
                #:*fixtures*
                #:make-fixture-dataset))

(in-package #:cl-gbdt/tests/functional/missing-value)

(defparameter *prediction-tolerance* 1d-9
  "How far two `cl-gbdt:predict' results may differ, element for element, and still count as
the same numbers below.

Exact equality is what is actually expected of every equality this file asserts: a sentinel
honoured and a stored NaN describe the same missing cell to the library, and both backends
train and predict deterministically from fixed data, so the two answers should come off
identical trees down identical paths. The tolerance is here because that expectation rests on
two config strings reaching the same code path inside the library, which nothing documents.
It is small enough that nothing this file exists to catch survives it: the measured gaps the
inequality assertions rest on are 0.026 on XGBoost's ingestion path and 0.693 on its
prediction path, seven and eight orders of magnitude above this.")

(defparameter *training-rounds* 5
  "How many boosting rounds every booster below is trained for. Five, matching
tests/functional/sparse-input.lisp's own default, which is enough for the two libraries'
predictions to have moved off their initial constant.")

(defparameter *sentinel* -999.0d0
  "The value that means *missing* in *HOLED-ROWS* -- an out-of-range number a caller would
plausibly have written into a CSV for a hole, which is exactly the case :MISSING exists for.

Exactly representable in `single-float', which matters: XGBoost compares the sentinel against
the datum at SINGLE precision, so a value that changed under the narrowing would make every
assertion below depend on rounding rather than on the sentinel.")

(defparameter *fixture-rows*
  '((0.0 1.0 2.0) (0.0 2.0 1.0) (0.0 1.0 2.0) (0.0 2.0 1.0)
    (5.0 1.0 2.0) (5.0 2.0 1.0) (5.0 1.0 2.0) (5.0 2.0 1.0))
  "Eight rows of three columns, coerced to `double-float' where they are used.

Purpose-built, and deliberately not `make-separable-dataset''s fixture, for the reason that
fixture's own construction gives: its element [i][j] is (i+j)/10, so ALL THREE of its columns
order the two classes identically, and a hole punched in one of them is simply routed around
by the other two -- the model learns the same thing and every assertion below would be
vacuous. Here COLUMN 0 ALONE carries the class -- 0.0 for the four rows labelled 0 and 5.0 for
the four labelled 1 -- while columns 1 and 2 hold the same two values in both halves and so
carry none. Punching the hole in column 0 therefore takes away the only information there is
about that row. Same shape of purpose-built fixture, and same reason, as
tests/functional/sparse-input.lisp's *OMITTED-ENTRY-ROWS*.")

(defparameter *fixture-labels*
  (make-array 8 :element-type 'single-float
                :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0))
  "Labels for *FIXTURE-ROWS*, row for row: the first four rows are the negative class and the
last four the positive one, which is exactly what column 0 encodes.")

(defparameter *hole-row* 7
  "Which row of *FIXTURE-ROWS* holds the missing cell.

A row of the POSITIVE class, not the negative one, and measured rather than assumed. With the
hole in a negative-class row the sentinel sits on the same side of every split the model would
draw as that row's real value 0.0 does, so reading it as missing changes nothing and the
inequality this file rests on would not hold. Row 7's real value is 5.0, on the far side of
every such split, so the two readings genuinely disagree -- see
`ingestion-sentinel-changes-what-was-learned' for the numbers.")

(defparameter *hole-column* 0
  "Which column of the holed row holds the missing cell: the only one carrying any class
information, per *FIXTURE-ROWS*.")

(defparameter *default-direction-row* 3
  "Which row `prediction-sentinel-resolution-is-single-precision' punches its TRAINING hole
in -- a row of the NEGATIVE class, unlike *HOLE-ROW*.

Measured, and the reason that one test trains a booster of its own rather than sharing the
clean-fixture booster the two prediction tests before it use. XGBoost gives every split a
DEFAULT DIRECTION for a value it finds missing, and a booster trained on the clean fixture
sends a missing value RIGHT -- the same way it sends any value above the split threshold, and
*DATUM-SHARING-ITS-FLOAT32* is far above it. Both readings of that datum would then land on
the same leaf and the test would assert nothing: measured 0.8467140793800354 for the datum
read literally and 0.8467140793800354 for a NaN in its place. Training with the hole in a
negative-class row instead teaches the default direction LEFT, and the two readings separate
-- measured 0.8467140793800354 read literally against 0.153285950422287 read as missing.

*SENTINEL* needs none of this, being far BELOW the split threshold and so on the far side of
it from the default direction already, which is why the tests that use it train on the clean
fixture like everything else in this file.")

(defparameter *narrowing-sentinel* 16777217.0d0
  "A sentinel that is not exactly representable in `single-float': 16777217 is 2^24 + 1, and
`single-float' spacing at 2^24 is 2, so it narrows to 16777216.0.

XGBoost compares the sentinel against the datum at SINGLE precision, whatever the matrix's own
element type, which `prediction-sentinel-resolution-is-single-precision' is the measurement
of. Probed directly with `XGDMatrixNumNonMissing' over 24 entries before that test was
written: this sentinel against *DATUM-SHARING-ITS-FLOAT32* keeps 22 of them, so it matched,
and against *DATUM-WITH-ITS-OWN-FLOAT32* keeps all 24, so it did not.")

(defparameter *datum-sharing-its-float32* 16777216.0d0
  "A datum *NARROWING-SENTINEL* matches: a DIFFERENT `double-float' from that sentinel, so a
comparison made at double precision would not match it, and the same `single-float' once
narrowed, so the comparison XGBoost actually makes does.")

(defparameter *datum-with-its-own-float32* 16777224.0d0
  "A datum *NARROWING-SENTINEL* does not match: exactly representable in `single-float' --
16777224 is 2^24 + 8, a multiple of the spacing there -- and therefore its own float32,
distinct from the sentinel's. The control half of
`prediction-sentinel-resolution-is-single-precision', without which the matching half would
be a coincidence rather than a measurement.")

(defun quiet-nan ()
  "Return a quiet NaN `double-float', built from its bits rather than by arithmetic.

`(/ 0d0 0d0)' would signal `floating-point-invalid-operation' wherever SBCL leaves the
`:invalid' trap enabled, which is its default on x86-64. Constructing the bit pattern reaches
the same value with no arithmetic at all and so needs no trap mask."
  (sb-kernel:make-double-float -524288 0))

(defun fixture-matrix (&key hole-value (hole-row *hole-row*))
  "Return *FIXTURE-ROWS* as a `(simple-array double-float (8 3))'.

HOLE-VALUE, when supplied, replaces the cell at (HOLE-ROW, *HOLE-COLUMN*); with it omitted
the matrix is whole, which is the clean matrix every prediction below is made on. HOLE-VALUE
is stored, never coerced or computed with, so a NaN passed here reaches the array without any
arithmetic that could trap.

HOLE-ROW defaults to *HOLE-ROW*, the positive-class row every test here punches. The one
call that overrides it builds a TRAINING matrix holed at *DEFAULT-DIRECTION-ROW* instead --
see that parameter for the measurement that makes it necessary."
  (let ((matrix (make-array (list (length *fixture-rows*)
                                   (length (first *fixture-rows*)))
                             :element-type 'double-float)))
    (loop :for row :in *fixture-rows*
          :for i :from 0
          :do (loop :for value :in row
                    :for j :from 0
                    :do (setf (aref matrix i j) (coerce value 'double-float))))
    (when hole-value
      (setf (aref matrix hole-row *hole-column*) hole-value))
    matrix))

(defun booster-trained-on (fixture backend matrix &key missing)
  "Train a booster on BACKEND from MATRIX and *FIXTURE-LABELS* for *TRAINING-ROUNDS* rounds
with FIXTURE's own parameters, and return it.

MISSING reaches `cl-gbdt:make-dataset' through `make-fixture-dataset', which passes it only
when it is non-NIL -- and NIL is exactly what omitting :MISSING already means, the backend's
own default, so nothing is lost by the two being the same call here. Every PREDICTION test
below leaves it NIL and names its sentinel to `cl-gbdt:predict' instead, which is what keeps
those tests about the prediction path alone.

The caller owns the booster: `train-on' below and every prediction test wrap this in
`cl-gbdt:with-booster'. The dataset does not outlive this call, exactly as it did not when
this was `train-on''s own body -- `cl-gbdt:predict' takes a matrix and never a dataset, so
nothing below needs it to."
  (cl-gbdt:with-dataset
      (dataset (make-fixture-dataset fixture backend matrix *fixture-labels*
                                     :missing missing))
    (cl-gbdt:train backend dataset :num-rounds *training-rounds*
                   :parameters (getf fixture :booster-parameters))))

(defun train-on (fixture backend matrix &key missing)
  "Train a booster on BACKEND from MATRIX with `booster-trained-on', and return its
predictions on the WHOLE fixture matrix.

Predicting on the whole matrix whatever MATRIX was is what makes two such results comparable:
what then differs between them is the data each model was TRAINED on, not the data each was
asked about. It also keeps every test that uses this about the INGESTION path alone: the
`cl-gbdt:predict' call below names no sentinel of its own, so nothing it returns depends on
`predict''s own :MISSING."
  (cl-gbdt:with-booster (booster (booster-trained-on fixture backend matrix :missing missing))
    (cl-gbdt:predict booster (fixture-matrix))))

(defun fixture-csr (matrix)
  "Return MATRIX, a `fixture-matrix' result, as a `cl-gbdt:csr-matrix' of the same width.

Every element is stored explicitly, zeros included, rather than only the non-zero ones a CSR
conversion usually keeps. An entry a `csr-matrix' does not store is MISSING to XGBoost
whatever any config says -- see that struct's own docstring, where the divergence is stated
-- so a conversion that dropped zeros would hand the library a second, unnamed missing cell
and `prediction-sentinel-works-on-a-csr-matrix' would no longer be about its sentinel alone.

tests/functional/sparse-input.lisp converts its own dense fixture the same way, but its
`dense-to-csr' is that file's internal helper over that file's fixture; the reason for
storing every element is a different one there, and stating this one here keeps it next to
the test that depends on it."
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

(defun predictions-agree-p (left right)
  "True when LEFT and RIGHT, two `cl-gbdt:predict' results, have the same shape and no pair of
corresponding elements differs by more than *PREDICTION-TOLERANCE*."
  (and (equal (array-dimensions left) (array-dimensions right))
       (loop :for index :below (array-total-size left)
             :always (<= (abs (- (row-major-aref left index) (row-major-aref right index)))
                         *prediction-tolerance*))))

;;; ---------------------------------------------------------------------------
;;; The capability this task ships
;;;
;;; Policy section 7 registers `:missing-value' as a question `cl-gbdt:backend-supports-p'
;;; answers. A NIL answer would say, in that function's own words, that the feature is
;;; unavailable here and the operation signals `capability-unavailable' -- which is exactly
;;; true of LightGBM and exactly false of XGBoost.
;;;
;;; The capability assertion and a real sentinel-honouring run are deliberately in ONE test
;;; rather than two. This project has twice shipped a feature whose capability keyword stayed
;;; false -- `:evaluation-history' and then `:early-stopping' -- because the assertion that the
;;; feature works and the assertion that the backend admits to it lived in different tests and
;;; only the first was written. Tying them together means the capability cannot be forgotten
;;; without the working-feature assertion going with it.
;;;
;;; Tying them together is not by itself enough, and the DEMONSTRATED assertion after the loop
;;; is what closes the remaining hole. Every per-backend assertion in this file -- here, in
;;; `ingestion-sentinel-changes-what-was-learned', `a-sentinel-that-is-not-a-real-signals' and
;;; `an-exponent-form-sentinel-reaches-the-library' -- is reached only on a backend that
;;; answers `:missing-value' true. Drop the capability from every backend that has it and all
;;; four of those simply stop asserting, while
;;; `missing-value-without-the-capability-signals' starts passing natively on both backends:
;;; five green tests and not one assertion about the feature, which is the same
;;; shipped-a-false-capability failure in its third disguise. `backend-supports-p' returns
;;; exactly T or NIL (`cl-gbdt/src/backend', `(and (getf ...) t)'), so the per-backend
;;; capability `ok' below is reached only down a path that already proved its own subject and
;;; cannot fail; DEMONSTRATED can, and is the assertion that actually holds the file up.

(defun predictions-with-the-sentinel-honoured (fixture backend)
  "Train on the holed fixture with :MISSING *SENTINEL* and return the predictions, or NIL when
BACKEND refused the argument with `capability-unavailable'.

The refusal is turned into NIL rather than a failure because a backend that does not provide
`:missing-value' is a correct backend -- `missing-value-without-the-capability-signals' is
where that refusal is asserted. A `capability-unavailable' naming any OTHER capability is
re-signalled instead: swallowing it would let this file report a demonstration as skipped when
something unrelated had broken."
  (handler-case (train-on fixture backend (fixture-matrix :hole-value *sentinel*)
                          :missing *sentinel*)
    (cl-gbdt:capability-unavailable (condition)
      (unless (eq :missing-value (cl-gbdt:capability-unavailable-capability condition))
        (error condition))
      nil)))

(deftest missing-value-capability-is-true-where-it-is-demonstrated
  ;; ASKED is every backend whose shared library was actually present, DEMONSTRATED every one
  ;; of those that honoured a sentinel. The two are collected across the loop so the assertion
  ;; after it can be about the SET rather than about any one backend.
  (let ((asked '())
        (demonstrated '()))
    (dolist (fixture *fixtures*)
      (with-backend-library ((getf fixture :backend))
        (push (getf fixture :backend) asked)
        (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
          (unwind-protect
               ;; The demonstration runs FIRST and the capability assertion hangs off its
               ;; result, rather than `backend-supports-p' gating the demonstration.
               (let ((honoured (predictions-with-the-sentinel-honoured fixture backend)))
                 (when honoured
                   (push (getf fixture :backend) demonstrated)
                   (let ((literal (train-on fixture backend
                                            (fixture-matrix :hole-value *sentinel*))))
                     (testing (format nil "~A: a dataset built with :missing ~A trains a ~
                                           different model than one built without it"
                                      (getf fixture :backend) *sentinel*)
                       (ok (not (predictions-agree-p honoured literal))
                           (format nil "with :missing ~S, without ~S" honoured literal))))
                   (testing (format nil "~A: and backend-supports-p admits to :missing-value"
                                    (getf fixture :backend))
                     (ok (eq t (cl-gbdt:backend-supports-p backend :missing-value))
                         (format nil "the capabilities were ~S"
                                 (cl-gbdt:backend-capabilities backend))))))
            (cl-gbdt:close-backend backend)))))
    ;; The one assertion in this file that no capability answer can route around, and the
    ;; reason the two lists above are collected at all -- see this section's header. It says
    ;; the feature is demonstrated SOMEWHERE, which is the claim the four capability-gated
    ;; tests silently stop making the moment no backend answers true.
    ;;
    ;; Guarded on ASKED, not on any capability: a fresh clone with no vendored library skips
    ;; every backend through `with-backend-library' -- `rove:skip' records a pending assertion
    ;; and returns rather than unwinding -- and demanding a demonstration from a suite that
    ;; ran nothing would turn that documented skip into a failure. Library absence is the only
    ;; thing this excuses.
    (when asked
      (testing "at least one backend with a library present demonstrates :missing-value"
        (ok demonstrated
            (format nil "asked ~S, demonstrated ~S" (reverse asked) (reverse demonstrated)))))))

;;; The load-bearing test: the sentinel is HONOURED, not merely accepted.
;;;
;;; Three models on the same eight rows, differing only in what the cell at (*HOLE-ROW*,
;;; *HOLE-COLUMN*) holds and whether :MISSING names it:
;;;
;;;   (a) the cell holds *SENTINEL*, and :MISSING *SENTINEL* says so
;;;   (b) the cell holds *SENTINEL*, and nothing says so -- it is an ordinary number
;;;   (c) the cell holds a NaN, and no :MISSING is needed -- NaN is the wrapper's own default
;;;
;;; (a) EQUALS (c): the sentinel was honoured, since the two datasets differ only in how the
;;; same missing cell was spelled. (a) DIFFERS FROM (b): the sentinel was not dropped. The
;;; inequality is the half that matters -- an implementation that accepted :MISSING and threw
;;; it away would pass every shape assertion ever written, and would pass the equality here
;;; too if the equality stood alone.
;;;
;;; Measured against the vendored XGBoost before either assertion was written, five rounds on
;;; the eight rows, predicting on the whole matrix -- the four negative rows all answer
;;; 0.153285950422287 in every case, and the four positive rows are what move:
;;;
;;;   (a) / (c)  0.8467140793800354
;;;   (b)        0.8203328251838684
;;;
;;; A gap of 0.0264, seven orders of magnitude above *PREDICTION-TOLERANCE*. The same
;;; measurement on LightGBM, taken through the NaN spelling since that backend refuses
;;; :MISSING, moves 0.2979475936814847 to 0.377711109588947 -- so this fixture is not one
;;; XGBoost happens to be sensitive to and LightGBM is not; the hole changes what is learned on
;;; both, and only the route to naming it differs.
;;;
;;; No number here is compared with a number from the other backend.

(deftest ingestion-sentinel-changes-what-was-learned
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :missing-value)
               (let ((honoured (train-on fixture backend
                                         (fixture-matrix :hole-value *sentinel*)
                                         :missing *sentinel*))
                     (literal (train-on fixture backend
                                        (fixture-matrix :hole-value *sentinel*)))
                     (nan (train-on fixture backend
                                    (fixture-matrix :hole-value (quiet-nan)))))
                 (testing (format nil "~A: :missing ~A answers what a stored NaN answers -- ~
                                       the sentinel was honoured"
                                  (getf fixture :backend) *sentinel*)
                   (ok (predictions-agree-p honoured nan)
                       (format nil "with :missing ~S, with a NaN ~S" honoured nan)))
                 (testing (format nil "~A: and answers something else than the same matrix ~
                                       read literally -- the sentinel was not dropped"
                                  (getf fixture :backend))
                   (ok (not (predictions-agree-p honoured literal))
                       (format nil "with :missing ~S, read literally ~S" honoured literal)))))
          (cl-gbdt:close-backend backend))))))

;;; Policy section 7's central rule: the operation re-checks the capability itself rather than
;;; trusting the caller to have asked `backend-supports-p' first, and signals rather than
;;; quietly ignoring the argument.
;;;
;;; LightGBM reaches this branch with no simulation at all -- its C API has no `missing' key,
;;; so the capability is false there and always will be. XGBoost's is true, so its plist is
;;; overwritten the way tests/functional/sparse-input.lisp overwrites one for the same purpose,
;;; which is what a library that could not provide the capability would have produced at
;;; `open-backend'. Both are driven from the same `backend-supports-p' answer rather than a
;;; hardcoded backend name, so this asserts the gate on EVERY backend, however each one came to
;;; have it closed.
;;;
;;; Two sentinel values, not one, because the rule is that :MISSING signals REGARDLESS OF THE
;;; VALUE. A NaN is the second: LightGBM would in fact honour a NaN, that being what its own
;;; ingestion path already treats as missing, and it still signals -- a capability whose answer
;;; depended on which value was passed could not be stated by `backend-supports-p' at all.
;;;
;;; `handler-case', not rove's `signals', which does not reliably catch a condition raised
;;; inside `restart-case'; the condition TYPE is asserted, not merely that something signalled.

(deftest missing-value-without-the-capability-signals
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend)))
            (matrix (fixture-matrix :hole-value *sentinel*)))
        (unwind-protect
             (progn
               ;; A backend that already answers false is left exactly as it opened; one that
               ;; answers true has its plist overwritten, which is the only way to reach the
               ;; gate there.
               (when (cl-gbdt:backend-supports-p backend :missing-value)
                 (setf (cl-gbdt:backend-capabilities backend) '(:missing-value nil)))
               (dolist (value (list *sentinel* (quiet-nan)))
                 (testing (format nil "~A: make-dataset signals capability-unavailable for ~
                                       :missing ~A, naming the capability and the backend"
                                  (getf fixture :backend)
                                  (if (sb-ext:float-nan-p value) "a NaN" value))
                   ;; `free-dataset' on the success branch, the way
                   ;; `sparse-input-without-the-capability-signals' frees the dataset it does
                   ;; not expect to get: this branch is only reached if the gate has regressed,
                   ;; and a leaked handle is a poor second failure to hand whoever is already
                   ;; reading the first one.
                   (let ((condition (handler-case
                                        (progn (cl-gbdt:free-dataset
                                                (cl-gbdt:make-dataset backend matrix
                                                                      :missing value))
                                               nil)
                                      (cl-gbdt:capability-unavailable (c) c))))
                     (ok condition "make-dataset signalled instead of building a dataset")
                     (ok (and condition
                              (eq :missing-value
                                  (cl-gbdt:capability-unavailable-capability condition)))
                         (format nil "the condition named capability ~S"
                                 (and condition
                                      (cl-gbdt:capability-unavailable-capability condition))))
                     (ok (and condition
                              (eq (getf fixture :backend)
                                  (cl-gbdt:backend-error-backend condition)))
                         (format nil "the condition named backend ~S"
                                 (and condition
                                      (cl-gbdt:backend-error-backend condition))))))))
          (cl-gbdt:close-backend backend))))))

;;; :MISSING takes a `real' or NIL and nothing else. This is the renderer's own rejection --
;;; `cl-gbdt/src/config/missing-value''s `missing-value-json', whose direct call is covered by
;;; tests/missing-value.lisp -- reached instead through the public API, which is a different
;;; path: it also pins that the value is judged at all rather than being formatted into the
;;; config JSON as whatever `princ' makes of it, and that the condition blames the backend the
;;; caller was talking to.
;;;
;;; Only on a backend whose capability is true. On one whose capability is false the gate fires
;;; first, by design -- the test above asserts exactly that, for this value's near neighbour --
;;; so there would be nothing here for the renderer to reject.

(deftest a-sentinel-that-is-not-a-real-signals
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :missing-value)
               (testing (format nil "~A: make-dataset signals unsupported-argument for a ~
                                     string sentinel, naming :missing and the backend"
                                (getf fixture :backend))
                 (let ((condition (handler-case
                                      (progn (cl-gbdt:free-dataset
                                              (cl-gbdt:make-dataset backend (fixture-matrix)
                                                                    :missing "-999.0"))
                                             nil)
                                    (cl-gbdt:unsupported-argument (c) c))))
                   (ok condition "make-dataset signalled instead of building a dataset")
                   (ok (and condition
                            (equal ":missing"
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
          (cl-gbdt:close-backend backend))))))

;;; The end-to-end check that what reaches the library is the RENDERER's output and not Lisp's.
;;;
;;; `1.0d-5' princs as "1.0d-5", and XGBoost's config-JSON parser rejects that outright --
;;; `json.cc:409: Expecting: ","', measured against the vendored library -- while accepting
;;; "1.0e-5", which is what `missing-value-json' emits. So a dataset that builds here is the
;;; whole statement: nothing between `make-dataset' and the C entry point reintroduced the Lisp
;;; exponent marker. Nothing shorter of an actual library call can say that, which is why this
;;; test exists in the functional suite rather than beside the renderer's own unit tests.
;;;
;;; The dataset's row count is asserted rather than merely its existence, so a `make-dataset'
;;; that returned some other backend's idea of success would still fail here.

(deftest an-exponent-form-sentinel-reaches-the-library
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :missing-value)
               (cl-gbdt:with-dataset
                   (dataset (make-fixture-dataset fixture backend (fixture-matrix)
                                                  *fixture-labels* :missing 1.0d-5))
                 (testing (format nil "~A: make-dataset accepts :missing 1.0d-5, which ~
                                       reaches the library as 1.0e-5"
                                  (getf fixture :backend))
                   (ok (= (length *fixture-rows*) (cl-gbdt:dataset-num-rows dataset))
                       (format nil "dataset-num-rows was ~S"
                               (cl-gbdt:dataset-num-rows dataset))))))
          (cl-gbdt:close-backend backend))))))

;;; ---------------------------------------------------------------------------
;;; `predict''s own :MISSING
;;;
;;; The same discrimination `ingestion-sentinel-changes-what-was-learned' makes, moved to the
;;; PREDICTION path and made against ONE fixed model. The booster below is trained on the
;;; clean fixture with no :MISSING anywhere, then asked three times, differing only in what
;;; the cell at (*HOLE-ROW*, *HOLE-COLUMN*) of the matrix it is asked about holds and whether
;;; :MISSING names it:
;;;
;;;   (a) the cell holds *SENTINEL*, and :MISSING *SENTINEL* says so
;;;   (b) the cell holds *SENTINEL*, and nothing says so -- it is an ordinary number
;;;   (c) the cell holds a NaN, and no :MISSING is needed -- NaN is the wrapper's own default
;;;
;;; (a) EQUALS (c): the sentinel was honoured, the two matrices differing only in how the same
;;; missing cell was spelled. (a) DIFFERS FROM (b): the sentinel was not dropped -- an
;;; implementation that accepted :MISSING and threw it away would pass the equality alone.
;;; Because the model is fixed, the three results differ only in the data `predict' was ASKED
;;; about, so nothing about how the model was trained can account for the gap.
;;;
;;; Measured against the vendored XGBoost before either assertion was written, five rounds on
;;; the clean eight rows -- the seven rows without a hole answer 0.153285950422287 or
;;; 0.8467140793800354 identically in all three cases, and row *HOLE-ROW* is what moves:
;;;
;;;   (a) / (c)  0.8467140793800354
;;;   (b)        0.153285950422287
;;;
;;; A gap of 0.693, more than eight orders of magnitude above *PREDICTION-TOLERANCE*.
;;; LightGBM refuses :MISSING and so is never asked; no number here is compared with one of
;;; its numbers.

(deftest prediction-sentinel-changes-what-is-predicted
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :missing-value)
               (cl-gbdt:with-booster
                   (booster (booster-trained-on fixture backend (fixture-matrix)))
                 (let* ((holed (fixture-matrix :hole-value *sentinel*))
                        (honoured (cl-gbdt:predict booster holed :missing *sentinel*))
                        (literal (cl-gbdt:predict booster holed))
                        (nan (cl-gbdt:predict booster
                                              (fixture-matrix :hole-value (quiet-nan)))))
                   (testing (format nil "~A: predict with :missing ~A answers what a stored ~
                                         NaN answers -- the sentinel was honoured"
                                    (getf fixture :backend) *sentinel*)
                     (ok (predictions-agree-p honoured nan)
                         (format nil "with :missing ~S, with a NaN ~S" honoured nan)))
                   (testing (format nil "~A: and answers something else than the same matrix ~
                                         read literally -- the sentinel was not dropped"
                                    (getf fixture :backend))
                     (ok (not (predictions-agree-p honoured literal))
                         (format nil "with :missing ~S, read literally ~S"
                                 honoured literal))))))
          (cl-gbdt:close-backend backend))))))

;;; The same three-way comparison over a `cl-gbdt:csr-matrix', which is the OTHER config site
;;; and not covered by the test above.
;;;
;;; A dense matrix reaches XGBoost through a transient DMatrix, so its sentinel is a key in
;;; that DMatrix's CREATION config -- the same config `make-dataset' fills. A `csr-matrix'
;;; builds no DMatrix at all: `XGBoosterPredictFromCSR' is inplace prediction, and the
;;; sentinel is a key in the PREDICT config instead. Two different strings built by two
;;; different functions, so a fix to one leaves the other exactly as broken as it was.
;;;
;;; :KIND :NORMAL throughout, stated rather than left to the default: XGBoost's sparse
;;; prediction entry point serves `:normal' and `:raw' only, measured during the sparse-input
;;; feature and recorded in `cl-gbdt:predict''s own docstring, so this test has two KINDs to
;;; choose between rather than four and picks the one every other prediction in this file
;;; uses.
;;;
;;; Measured the same way as the test above, and the same three numbers: (a) and (c) answer
;;; 0.8467140793800354 at row *HOLE-ROW* and (b) answers 0.153285950422287. Gated on
;;; `:sparse-input' as well as `:missing-value', since a backend without the first has no
;;; `csr-matrix' path to test the second on.

(deftest prediction-sentinel-works-on-a-csr-matrix
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
        (unwind-protect
             (when (and (cl-gbdt:backend-supports-p backend :missing-value)
                        (cl-gbdt:backend-supports-p backend :sparse-input))
               (cl-gbdt:with-booster
                   (booster (booster-trained-on fixture backend (fixture-matrix)))
                 (let* ((holed (fixture-csr (fixture-matrix :hole-value *sentinel*)))
                        (honoured (cl-gbdt:predict booster holed
                                                   :kind :normal :missing *sentinel*))
                        (literal (cl-gbdt:predict booster holed :kind :normal))
                        (nan (cl-gbdt:predict
                              booster
                              (fixture-csr (fixture-matrix :hole-value (quiet-nan)))
                              :kind :normal)))
                   (testing (format nil "~A: predict on a csr-matrix with :missing ~A answers ~
                                         what a stored NaN answers"
                                    (getf fixture :backend) *sentinel*)
                     (ok (predictions-agree-p honoured nan)
                         (format nil "with :missing ~S, with a NaN ~S" honoured nan)))
                   (testing (format nil "~A: and answers something else than the same ~
                                         csr-matrix read literally"
                                    (getf fixture :backend))
                     (ok (not (predictions-agree-p honoured literal))
                         (format nil "with :missing ~S, read literally ~S"
                                 honoured literal))))))
          (cl-gbdt:close-backend backend))))))

;;; What resolution the sentinel is compared at, which is a property a caller has to know to
;;; choose one: XGBoost narrows both sides to `single-float' first, so two `double-float's
;;; that round to the same float32 are the same value to it.
;;;
;;; *NARROWING-SENTINEL* against *DATUM-SHARING-ITS-FLOAT32* -- a different double, the same
;;; float32 -- predicts what a NaN in that cell predicts, so it matched. Against
;;; *DATUM-WITH-ITS-OWN-FLOAT32* it does not, so it did not. The second half is the control
;;; that makes the first a measurement rather than a coincidence: without it, an
;;; implementation that treated EVERY value in the holed cell as missing would pass.
;;;
;;; This test trains its own booster, holed at *DEFAULT-DIRECTION-ROW*, and that parameter
;;; carries the measurement forcing it: against a booster trained on the clean fixture, both
;;; readings of a 2^24-sized datum answer 0.8467140793800354 and neither assertion would say
;;; anything. Against this one they answer 0.153285950422287 read as missing and
;;; 0.8467140793800354 read literally.

(deftest prediction-sentinel-resolution-is-single-precision
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
        (unwind-protect
             (when (cl-gbdt:backend-supports-p backend :missing-value)
               (cl-gbdt:with-booster
                   (booster (booster-trained-on
                             fixture backend
                             (fixture-matrix :hole-value (quiet-nan)
                                             :hole-row *default-direction-row*)))
                 (let ((nan (cl-gbdt:predict booster
                                             (fixture-matrix :hole-value (quiet-nan))))
                       (matched (cl-gbdt:predict
                                 booster
                                 (fixture-matrix :hole-value *datum-sharing-its-float32*)
                                 :missing *narrowing-sentinel*))
                       (unmatched (cl-gbdt:predict
                                   booster
                                   (fixture-matrix :hole-value *datum-with-its-own-float32*)
                                   :missing *narrowing-sentinel*)))
                   (testing (format nil "~A: :missing ~A reads ~A as missing -- a different ~
                                         double sharing its single-float"
                                    (getf fixture :backend) *narrowing-sentinel*
                                    *datum-sharing-its-float32*)
                     (ok (predictions-agree-p matched nan)
                         (format nil "with the sentinel ~S, with a NaN ~S" matched nan)))
                   (testing (format nil "~A: and does not read ~A as missing -- its own ~
                                         single-float"
                                    (getf fixture :backend) *datum-with-its-own-float32*)
                     (ok (not (predictions-agree-p unmatched nan))
                         (format nil "with the sentinel ~S, with a NaN ~S"
                                 unmatched nan))))))
          (cl-gbdt:close-backend backend))))))

;;; Policy section 7's central rule again, on `predict' this time: the operation re-checks the
;;; capability itself rather than trusting the caller, and each operation checks it
;;; SEPARATELY. `missing-value-without-the-capability-signals' above asserts the same rule for
;;; `make-dataset'; neither test stands in for the other, since the two reach two different
;;; per-backend calls and a backend could gate one and not the other.
;;;
;;; One sentinel value here, not the two the ingestion test uses: the rule that :MISSING
;;; signals REGARDLESS OF THE VALUE is a property of each backend's own `%check-missing-value',
;;; which the ingestion test already pins for both values, and `predict' reaches that same
;;; function rather than a second copy of it. What is unpinned until here is that `predict'
;;; reaches it AT ALL.
;;;
;;; The booster is trained before the capability plist is overwritten. `train' reads
;;; capabilities of its own, and a backend stripped down to `(:missing-value nil)' could not
;;; be asked for a booster to predict with in the first place.

(deftest prediction-without-the-capability-signals
  (dolist (fixture *fixtures*)
    (with-backend-library ((getf fixture :backend))
      (let ((backend (cl-gbdt:open-backend (getf fixture :backend))))
        (unwind-protect
             (cl-gbdt:with-booster
                 (booster (booster-trained-on fixture backend (fixture-matrix)))
               ;; A backend that already answers false is left exactly as it opened; one that
               ;; answers true has its plist overwritten, which is the only way to reach the
               ;; gate there.
               (when (cl-gbdt:backend-supports-p backend :missing-value)
                 (setf (cl-gbdt:backend-capabilities backend) '(:missing-value nil)))
               (testing (format nil "~A: predict signals capability-unavailable for :missing ~
                                     ~A, naming the capability and the backend"
                                (getf fixture :backend) *sentinel*)
                 (let ((condition
                         (handler-case
                             (progn (cl-gbdt:predict booster
                                                     (fixture-matrix :hole-value *sentinel*)
                                                     :missing *sentinel*)
                                    nil)
                           (cl-gbdt:capability-unavailable (c) c))))
                   (ok condition "predict signalled instead of returning predictions")
                   (ok (and condition
                            (eq :missing-value
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
