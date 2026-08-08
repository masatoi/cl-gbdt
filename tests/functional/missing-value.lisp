;;;; missing-value.lisp --- Portable contract tests for `make-dataset''s :MISSING.
;;;;
;;;; `cl-gbdt:make-dataset' now takes a :MISSING keyword naming which datum in the caller's
;;;; own data means *missing*, gated by the `:missing-value' capability. It is a VALUE, not a
;;;; policy: it chooses which number means missing and does not turn missing handling on or
;;;; off. Like tests/functional/evaluation.lisp and tests/functional/sparse-input.lisp beside
;;;; it, every test below runs over that first file's *FIXTURES*, once per backend, so the two
;;;; backends cannot drift apart in shape or meaning without one of them failing here.
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
  ;; `cl-gbdt:make-dataset'; :MISSING is supplied on top of it by `train-on' below, since it is
  ;; this file's subject and not part of any fixture.
  (:import-from #:cl-gbdt/tests/functional/evaluation
                #:*fixtures*
                #:make-fixture-dataset))

(in-package #:cl-gbdt/tests/functional/missing-value)

(defparameter *prediction-tolerance* 1d-9
  "How far two `cl-gbdt:predict' results may differ, element for element, and still count as
the same numbers below.

Exact equality is what is actually expected of the one equality this file asserts: a sentinel
honoured and a stored NaN describe the same missing cell to the library, and both backends
train deterministically from a fixed dataset, so the two boosters should be identical trees.
The tolerance is here because that expectation rests on the two config strings reaching the
same code path inside the library, which nothing documents. It is small enough that nothing
this file exists to catch survives it: the measured gap the inequality assertion rests on is
0.026 on XGBoost, seven orders of magnitude above this.")

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
  "Which column of *HOLE-ROW* holds the missing cell: the only one carrying any class
information, per *FIXTURE-ROWS*.")

(defun quiet-nan ()
  "Return a quiet NaN `double-float', built from its bits rather than by arithmetic.

`(/ 0d0 0d0)' would signal `floating-point-invalid-operation' wherever SBCL leaves the
`:invalid' trap enabled, which is its default on x86-64. Constructing the bit pattern reaches
the same value with no arithmetic at all and so needs no trap mask."
  (sb-kernel:make-double-float -524288 0))

(defun fixture-matrix (&key hole-value)
  "Return *FIXTURE-ROWS* as a `(simple-array double-float (8 3))'.

HOLE-VALUE, when supplied, replaces the cell at (*HOLE-ROW*, *HOLE-COLUMN*); with it omitted
the matrix is whole, which is the clean matrix every prediction below is made on. HOLE-VALUE
is stored, never coerced or computed with, so a NaN passed here reaches the array without any
arithmetic that could trap."
  (let ((matrix (make-array (list (length *fixture-rows*)
                                   (length (first *fixture-rows*)))
                             :element-type 'double-float)))
    (loop :for row :in *fixture-rows*
          :for i :from 0
          :do (loop :for value :in row
                    :for j :from 0
                    :do (setf (aref matrix i j) (coerce value 'double-float))))
    (when hole-value
      (setf (aref matrix *hole-row* *hole-column*) hole-value))
    matrix))

(defun train-on (fixture backend matrix &key missing)
  "Train a booster on BACKEND from MATRIX and *FIXTURE-LABELS* for *TRAINING-ROUNDS* rounds
with FIXTURE's own parameters, and return its predictions on the WHOLE fixture matrix.

MISSING reaches `cl-gbdt:make-dataset' through `make-fixture-dataset', which passes it only
when it is non-NIL -- and NIL is exactly what omitting :MISSING already means, the backend's
own default, so nothing is lost by the two being the same call here.

Predicting on the whole matrix whatever MATRIX was is what makes two such results comparable:
what then differs between them is the data each model was TRAINED on, not the data each was
asked about. It also keeps every test in this file about the INGESTION path alone -- `predict'
gains its own :MISSING in the next task, and nothing here depends on it."
  (cl-gbdt:with-booster
      (booster (cl-gbdt:with-dataset
                   (dataset (make-fixture-dataset fixture backend matrix *fixture-labels*
                                                  :missing missing))
                 (cl-gbdt:train backend dataset :num-rounds *training-rounds*
                                :parameters (getf fixture :booster-parameters))))
    (cl-gbdt:predict booster (fixture-matrix))))

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
