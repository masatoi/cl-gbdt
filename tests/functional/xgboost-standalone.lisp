;;;; xgboost-standalone.lisp --- Layer 1 trains and predicts with no unified API in the image.
;;;;
;;;; The XGBoost half of tests/functional/lightgbm-standalone.lisp, and deliberately its
;;;; mirror. This file names `cl-gbdt/xgboost' -- the PUBLIC package, not the internal
;;;; `cl-gbdt/src/xgboost/all' aggregation -- and no other system of this project. No
;;;; `cl-gbdt', no `.../unified', and not tests/functional/support.lisp either, which reaches
;;;; the unified API itself (`cl-gbdt:make-csr-matrix' in its `dense-to-csr') and would drag
;;;; Layer 2 in through the back door. `tools/ci/check-leaf-systems.lisp' loads every leaf
;;;; system alone in a fresh `ros run' subprocess, so the claim in this file's name is
;;;; enforced by the build rather than asserted here in prose: were any of those edges
;;;; declared, quickloading this system alone would define the `CL-GBDT' package, which is
;;;; exactly what the branch's isolation probe checks for.
;;;;
;;;; Every reference below is therefore package-qualified. That is a consequence of the
;;;; zero-symbol import, and also the point: a reader can see for each name which package a
;;;; standalone caller would have had to load to get it.
;;;;
;;;; It states its own fixture and its own library-discovery skip rather than importing the
;;;; shared ones, for the same reason -- and needs as little to do so as its sibling: this
;;;; backend's `open-backend' performs its own discovery too (env var, then the vendored
;;;; directory, then CFFI's system search -- `cl-gbdt/src/library''s `resolve-and-load-library'),
;;;; so like that file and unlike every other file in tests/functional/ this one loads no
;;;; shared library by hand at all.
;;;;
;;;; Three things below differ from the sibling, and all three are XGBoost's own asymmetries
;;;; rather than choices made here:
;;;;
;;;;   - `create-dataset' takes no :PARAMETERS and no :REFERENCE. This library's creation
;;;;     config JSON documents `"missing"', `"nthread"' and `"data_split_mode"' and has no
;;;;     concept resembling LightGBM's bin-mapper alignment, so *PARAMETERS* below reaches
;;;;     `create-booster' alone, and a validation set needs no reference dataset to bin
;;;;     against however far its own column values sit from the training set's.
;;;;
;;;;   - `XGBoosterCreate' takes the whole array of DMatrix handles AT ONCE -- the training
;;;;     set first, then each validation set -- and this library has no "add valid data" entry
;;;;     point at all, so a validation set left out of the creation call could never be added
;;;;     afterwards. That is what makes the creation array worth a test of its own.
;;;;
;;;;   - the read-back that proves a validation set reached that array is `num_feature', not
;;;;     an evaluation index. `evaluate-one-iteration' -- `XGBoosterEvalOneIter' -- takes its
;;;;     DMatrices as ARGUMENTS and never consults the booster's own creation array, so it
;;;;     answers identically for a booster built with validation sets and one built without,
;;;;     and the sibling's index-2 assertion has no counterpart here. `num_feature' does
;;;;     discriminate: XGBoost sets it to the MAXIMUM column count over the whole creation
;;;;     array, so a five-column validation set beside a two-column training set makes it 5
;;;;     where the same booster built without that entry has 2. Measured against the vendored
;;;;     library, from both sides, and read off the same pair of validation sets attached in
;;;;     both orders so that the claim is about EVERY entry rather than about one -- see
;;;;     `layer-1-alone-puts-every-validation-set-in-the-creation-array'.
;;;;
;;;; What this file deliberately does NOT assert: that Layer 1's `create-booster' plus a loop
;;;; of `update-one-iteration' produces the same model as `cl-gbdt:train'. That comparison
;;;; needs both APIs in one image, which is the one thing this file may not have, so it lives
;;;; in tests/functional/xgboost-api.lisp as `xgboost-api-create-booster-and-train-agree' --
;;;; see the commentary there, which carries the other half of this note. The split is not
;;;; cosmetic even now that `train' calls `create-booster' for its whole construction
;;;; (.superpowers/sdd/2026-08-12-train-create-booster-merge): this file's only dependency is
;;;; `cl-gbdt/xgboost', Layer 1 alone, so `train' is not a name it can even read here, let
;;;; alone call to check against.

(uiop:define-package #:cl-gbdt/tests/functional/xgboost-standalone
  (:use #:cl #:rove)
  ;; The one and only `cl-gbdt' edge, and it names zero symbols: every call below is
  ;; package-qualified, so nothing needs importing, and the clause is here to declare the
  ;; dependency itself. ASDF resolves the name to the system of the same name in
  ;; `cl-gbdt.asd', whose closure is Layer 1 alone -- that closure is what this file exists
  ;; to test, so widening this clause to `cl-gbdt/src/xgboost/all' (every `%'-function
  ;; besides) or to anything Layer 2 would silently destroy the test rather than fail it.
  (:import-from #:cl-gbdt/xgboost))

(in-package #:cl-gbdt/tests/functional/xgboost-standalone)

(defparameter *parameters*
  '(:objective "binary:logistic" :tree-method "hist" :max-depth 3)
  "XGBoost's own vocabulary, as `create-booster' takes it. Unlike the sibling's plist this one
is handed to a single call: XGBoost's `create-dataset' has no :PARAMETERS argument to pass it
to as well, this library having no dataset-level parameter that affects training.

`normalize-parameters' renders the three keys as `objective', `tree_method' and `max_depth',
which are XGBoost's documented spellings; none of them had to be corrected against the
library. That the library ACCEPTS a key is not by itself evidence the key is spelled right --
measured, `XGBoosterSetParam' takes an unknown name such as `treemethod' without complaint --
so the spellings were checked against XGBoost's parameter documentation rather than against
the absence of an error.

None of the three is REQUIRED here, which is the other half of the difference from the
sibling, where `min_data_in_leaf 1' was load-bearing: measured against the vendored library,
this fixture separates with an empty plist too, and rather more sharply (0.001 against 0.999,
XGBoost's default `reg:squarederror' fitting the 0/1 labels directly). They are stated so that
the objective, the tree method and the depth of the run asserted below are the caller's own
rather than whatever this version of the library currently defaults to. There is no `verbose'
key either, and none is needed: measured, XGBoost prints nothing at its default verbosity for
any run in this file.")

(defun fixture ()
  "Return (values MATRIX LABEL-VECTOR): sixteen rows whose label is a step function of
column 0, which any working booster separates.

Column 0 counts 0..15 and carries the whole signal; column 1 cycles 0..3 and carries none,
so a booster that ignored the matrix's column order -- reading it transposed, say -- could
not produce the ordering asserted below. Both arrays are `double-float': the label field is
XGBoost's `float32' and `create-dataset' coerces on the way in, so a standalone caller need
not know that."
  (let ((matrix (make-array '(16 2) :element-type 'double-float))
        (label-vector (make-array 16 :element-type 'double-float)))
    (dotimes (row 16)
      (setf (aref matrix row 0) (coerce row 'double-float)
            (aref matrix row 1) (coerce (mod row 4) 'double-float)
            (aref label-vector row) (if (< row 8) 0d0 1d0)))
    (values matrix label-vector)))

(defun validation-fixture ()
  "Return (values MATRIX LABEL-VECTOR): four held-out rows of the same width as `fixture''s,
three of which its step function explains and one of which it contradicts.

Column 0 is 1.5, 3.5, 5.5 and 9.5 -- values that appear nowhere in `fixture''s matrix, which
needs no :REFERENCE here the way the sibling's does: XGBoost bins nothing at dataset creation
and `create-dataset' has no such argument to give. Every label is 0, which makes the last row
a deliberate error, so this is a genuinely held-out set rather than a copy of the training
data. Nothing below reads a metric off it -- `evaluate-one-iteration' would answer the same
for either, taking its DMatrices explicitly -- so what the labels buy is only that a reader of
`layer-1-alone-attaches-a-validation-set' is looking at a real validation set."
  (let ((matrix (make-array '(4 2) :element-type 'double-float))
        (label-vector (make-array 4 :element-type 'double-float)))
    (dotimes (row 4)
      (setf (aref matrix row 0) (if (= row 3) 9.5d0 (+ 1.5d0 (* 2 row)))
            (aref matrix row 1) (coerce row 'double-float)
            (aref label-vector row) 0d0))
    (values matrix label-vector)))

(defun wide-validation-fixture ()
  "Return (values MATRIX LABEL-VECTOR): four rows of FIVE columns, where `fixture' has two.

The width is the whole point and the only thing this fixture is for. XGBoost's `num_feature'
is the maximum column count over the array of DMatrix handles `XGBoosterCreate' was given, so
a booster that put this dataset in that array reports 5 and one that dropped it reports 2 --
the one property of a validation set that is observable from outside this library at all. See
`layer-1-alone-puts-every-validation-set-in-the-creation-array', which reads that number back
two different ways off this dataset attached both before and after `validation-fixture''s, and
note that a booster built over this pair CANNOT BE TRAINED: XGBoost requires `num_feature' to
EQUAL the training matrix's own column count for an update, which is why this fixture cannot
simply be folded into `validation-fixture' above."
  (let ((matrix (make-array '(4 5) :element-type 'double-float))
        (label-vector (make-array 4 :element-type 'double-float)))
    (dotimes (row 4)
      (dotimes (column 5)
        (setf (aref matrix row column) (coerce (+ row column) 'double-float)))
      (setf (aref label-vector row) 0d0))
    (values matrix label-vector)))

(defmacro with-open-backend ((backend) &body body)
  "Evaluate BODY with BACKEND bound to an open XGBoost backend, closing it afterwards, or
skip when the shared library is absent.

A missing library is a skip rather than a failure, because `vendor/' is git-ignored and a
fresh clone legitimately has none -- the same judgement tests/functional/support.lisp's
`with-backend-library' makes, reached here without that file. The condition is
`open-backend''s own `backend-library-not-found', caught by a name `cl-gbdt/xgboost'
publishes, so this skip needs no discovery logic of its own and no internal package.

`rove:skip' records a pending assertion and RETURNS rather than unwinding, so BODY is
guarded by an explicit test on BACKEND rather than left to run on with none to use."
  (let ((condition (gensym "CONDITION")))
    `(let ((,backend (handler-case (cl-gbdt/xgboost:open-backend :xgboost)
                       (cl-gbdt/xgboost:backend-library-not-found (,condition)
                         (skip (princ-to-string ,condition))
                         nil))))
       (when ,backend
         (unwind-protect (progn ,@body)
           (cl-gbdt/xgboost:close-backend ,backend))))))

(defun separates-p (result label-vector)
  "True when every prediction in RESULT -- `predict''s (ROWS 1) array -- whose label is 1
exceeds every prediction whose label is 0.

The ordering property, not the values: exact probabilities would break on any upstream
version bump without saying anything new. Row 0 against row 15 alone would pass for a model
that had learned the boundary in the wrong place, so both extremes of both classes are
compared."
  (let ((negatives '())
        (positives '()))
    (dotimes (row (array-dimension result 0))
      (if (zerop (aref label-vector row))
          (push (aref result row 0) negatives)
          (push (aref result row 0) positives)))
    (< (reduce #'max negatives) (reduce #'min positives))))

(deftest layer-1-alone-trains-and-predicts
  (testing "a caller with only cl-gbdt/xgboost loaded can go from a matrix to predictions"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector)))
          (unwind-protect
               (let ((booster (cl-gbdt/xgboost:create-booster backend data
                                                              :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 20)
                          (cl-gbdt/xgboost:update-one-iteration booster))
                        (multiple-value-bind (result shape)
                            (cl-gbdt/xgboost:predict booster matrix)
                          ;; One prediction per row and a single output group: a booster
                          ;; built with the wrong `num_class', or a buffer sized from the row
                          ;; count alone, would not have this shape.
                          (ok (equal '(16 1) (array-dimensions result))
                              (format nil "predict's result is ~S"
                                      (array-dimensions result)))
                          ;; `predict''s SECOND value, and the one assertion in this file
                          ;; whose expected value could not be copied from the sibling: this
                          ;; backend does not derive a shape, it reads XGBoost's own
                          ;; `out_shape' back and states it verbatim, so what belongs here is
                          ;; whatever the library says rather than whatever LightGBM says.
                          ;; Measured, it is `(16 1)' -- the same numbers the assertion above
                          ;; already examined, `out_dim' being 2 for `:normal'. The two are
                          ;; not redundant even so: this one is what a `predict' that stated
                          ;; NO shape, leaving a standalone caller to infer one, would fail.
                          (ok (equal '(16 1) shape)
                              (format nil "predict states the shape ~S" shape))
                          ;; The one assertion here that needs the booster to have actually
                          ;; TRAINED, and so the one a no-op `update-one-iteration' would
                          ;; fail: measured against the vendored library, a booster straight
                          ;; out of `create-booster' predicts exactly 0.622459352016449 for
                          ;; all sixteen rows -- XGBoost's base score put through
                          ;; `binary:logistic''s transform, with the shape the two assertions
                          ;; above check, which is why neither of them can stand in for this.
                          (ok (separates-p result label-vector)
                              (format nil "predictions: ~S" result))))
                   (cl-gbdt/xgboost:free-booster booster)))
            (cl-gbdt/xgboost:free-dataset data)))))))

;;; `create-booster''s :VALID-SETS half is invisible to everything above, and to
;;; `xgboost-api-create-booster-and-train-agree' in tests/functional/xgboost-api.lisp as well:
;;; a validation set does not change the model XGBoost trains, so comparing PREDICTIONS --
;;; either against a label boundary here or against `cl-gbdt:train''s own run there -- is
;;; blind to the validation entries of the handle array `XGBoosterCreate' is given, to the
;;; `copy-list' snapshot of the caller's list, and to the per-entry `%check-xgboost-dataset'.
;;; `train' has its own copy of that same procedure and other tests DO exercise it with
;;; validation sets, so without the three tests below `create-booster''s copy could drift
;;; alone and every suite stay green.
;;;
;;; The assertions are therefore chosen to be ones a prediction cannot make: what the booster
;;; retained, what reached the library, and what it refuses before any foreign call. None of
;;; it needs the unified API, which is why it belongs in this file rather than beside the
;;; agreement test.

(deftest layer-1-alone-attaches-a-validation-set
  (testing "create-booster's :valid-sets are retained, and the booster keeps its own view"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (multiple-value-bind (valid-matrix valid-label-vector) (validation-fixture)
          (let* ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector))
                 (valid-1 (cl-gbdt/xgboost:create-dataset backend valid-matrix
                                                          :label valid-label-vector))
                 (valid-2 (cl-gbdt/xgboost:create-dataset backend valid-matrix
                                                          :label valid-label-vector))
                 ;; The caller's own list object, kept so it can be mutated below.
                 (callers-valid-sets (list valid-1 valid-2))
                 (booster (cl-gbdt/xgboost:create-booster
                           backend data :valid-sets callers-valid-sets
                           :parameters *parameters*)))
            (unwind-protect
                 (progn
                   (dotimes (round 20)
                     (cl-gbdt/xgboost:update-one-iteration booster))
                   ;; Retention, both halves, and the two halves are not worth the same on
                   ;; this backend. A `create-booster' that handed the VALIDATION sets to
                   ;; `XGBoosterCreate' and kept no copy of them would train and predict
                   ;; identically -- and leave a standalone caller no way to ask what its
                   ;; booster depends on, and `%check-booster-datasets-live' nothing to check;
                   ;; that half is invisible to everything else in this file, which is what
                   ;; the assertion buys. The TRAINING set is different here from the sibling:
                   ;; `update-one-iteration' READS it back, `XGBoosterUpdateOneIter' taking
                   ;; that DMatrix as an explicit argument where LightGBM's reads its own, so
                   ;; a booster retaining none would have signalled `missing-training-set' in
                   ;; the loop above rather than reaching this line. What is asserted here is
                   ;; the narrower and still unchecked claim that it retained THIS dataset --
                   ;; the one the caller named -- rather than merely some dataset. Both
                   ;; readers reach a test of either backend's Layer 1 for the first time
                   ;; here; `cl-gbdt/xgboost' did not publish them until this task.
                   (ok (eq data (cl-gbdt/xgboost:booster-training-set booster))
                       "the booster reports the dataset it was built over")
                   (ok (equal (list valid-1 valid-2)
                              (cl-gbdt/xgboost:booster-validation-sets booster))
                       (format nil "booster-validation-sets is ~S"
                               (cl-gbdt/xgboost:booster-validation-sets booster)))
                   ;; The `copy-list'. Truncating the caller's own list is what a `delete' of
                   ;; the second entry does; were the booster holding that list object rather
                   ;; than a snapshot, this would silently remove VALID-2 from its view --
                   ;; and the freed-set assertion at the end of this test would then not
                   ;; signal at all.
                   (setf (cdr callers-valid-sets) nil)
                   (ok (equal (list valid-1 valid-2)
                              (cl-gbdt/xgboost:booster-validation-sets booster))
                       (format nil "after the caller truncated its own list: ~S"
                               (cl-gbdt/xgboost:booster-validation-sets booster)))
                   ;; Last, because it poisons the booster: a validation set freed out from
                   ;; under it has to be noticed BEFORE the next update, and nothing but a
                   ;; Lisp-side check is in a position to notice. `XGBoosterUpdateOneIter' is
                   ;; handed the TRAINING DMatrix alone -- see `update-one-iteration' in
                   ;; src/xgboost/api.lisp -- so the foreign call has no occasion to look at a
                   ;; validation set at all, and its status code cannot report on one.
                   ;; `%check-booster-datasets-live' is what does the noticing, and what it is
                   ;; worth differs between the two kinds of dataset it walks. For the TRAINING
                   ;; set the check averts undefined behaviour, that pointer being handed
                   ;; straight to C. For a VALIDATION set it does not: measured by mutation,
                   ;; and deterministically rather than flakily, dropping that function's
                   ;; `booster-validation-sets' loop reddens this assertion and nothing else in
                   ;; this file, and the update then returns NORMALLY -- XGBoost holds its own
                   ;; reference to each DMatrix it was given at creation, so freeing this one
                   ;; here did not leave the library holding a dangling pointer. What the check
                   ;; buys for a validation set is a uniform promise instead, and the one
                   ;; `train''s own boosters already keep: a booster refuses to advance once
                   ;; ANYTHING it was built over has been freed, rather than silently carrying
                   ;; on with a dependency the caller has destroyed.
                   (cl-gbdt/xgboost:free-dataset valid-2)
                   ;; `handler-case', not rove's `signals', which does not reliably catch a
                   ;; condition raised inside `restart-case'.
                   (ok (handler-case
                           (progn (cl-gbdt/xgboost:update-one-iteration booster) nil)
                         (cl-gbdt/xgboost:released-handle-error () t))
                       "update-one-iteration did not signal released-handle-error"))
              (progn
                (cl-gbdt/xgboost:free-booster booster)
                (cl-gbdt/xgboost:free-dataset valid-1)
                (cl-gbdt/xgboost:free-dataset data)))))))))

(deftest layer-1-alone-puts-every-validation-set-in-the-creation-array
  (testing "every :valid-sets entry reaches XGBoosterCreate's handle array, wherever it sits"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (multiple-value-bind (narrow-matrix narrow-label-vector) (validation-fixture)
          (multiple-value-bind (wide-matrix wide-label-vector) (wide-validation-fixture)
            (let* ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector))
                   (narrow-valid (cl-gbdt/xgboost:create-dataset backend narrow-matrix
                                                                 :label narrow-label-vector))
                   (wide-valid (cl-gbdt/xgboost:create-dataset backend wide-matrix
                                                               :label wide-label-vector))
                   ;; Three boosters over one training set, differing in nothing but their
                   ;; :VALID-SETS. The first two hold the SAME PAIR in the two possible
                   ;; ORDERS, which is what makes this test about the whole array rather than
                   ;; about one entry: WIDE-VALID is the number-carrying one, so an attach
                   ;; loop that kept only the head would lose it from WIDE-LAST and one that
                   ;; kept only the tail would lose it from WIDE-FIRST. A single-entry test
                   ;; catches neither -- measured, and the reason this test has this shape:
                   ;; `(cons train-data-pointer (last valid-set-pointers))' in
                   ;; `create-booster', a version that silently discards every validation set
                   ;; but the last, left every assertion in both this file and
                   ;; tests/functional/xgboost-api.lisp green while it attached one entry
                   ;; here. `create-booster''s docstring promises "the training set first,
                   ;; then each validation set in the order given"; this is the assertion
                   ;; that holds it to the word EACH, as the sibling's index-2 `booster-eval'
                   ;; read does for LightGBM.
                   ;;
                   ;; None of the three is trained: `num_feature' is settled by
                   ;; `XGBoosterCreate' from the array it was given, and the assertions below
                   ;; read it off boosters of zero rounds precisely so that nothing about the
                   ;; MODEL can be confused for the property under test.
                   (wide-first (cl-gbdt/xgboost:create-booster
                                backend data :valid-sets (list wide-valid narrow-valid)
                                :parameters *parameters*))
                   (wide-last (cl-gbdt/xgboost:create-booster
                               backend data :valid-sets (list narrow-valid wide-valid)
                               :parameters *parameters*))
                   (unattached (cl-gbdt/xgboost:create-booster backend data
                                                               :parameters *parameters*)))
              (unwind-protect
                   (progn
                     ;; `num_feature' read from the inference side, where XGBoost requires it
                     ;; to be at least the matrix's own column count. Both boosters accept
                     ;; five columns, so both have a `num_feature' of 5 -- a number that
                     ;; exists neither in their training set nor in NARROW-VALID, and so one
                     ;; that could only have come from WIDE-VALID reaching the creation array
                     ;; from the position it was given in.
                     (ok (equal '(4 1)
                                (array-dimensions
                                 (cl-gbdt/xgboost:predict wide-first wide-matrix)))
                         "the booster given the wide set FIRST predicts five columns")
                     (ok (equal '(4 1)
                                (array-dimensions
                                 (cl-gbdt/xgboost:predict wide-last wide-matrix)))
                         "the booster given the wide set LAST predicts five columns")
                     ;; The control, and the half that makes the two assertions above
                     ;; discriminating rather than merely true: the same call on the booster
                     ;; built with no validation set at all is refused by the library,
                     ;; `num_feature' there being the training set's own 2. Measured by
                     ;; mutation: a `create-booster' that hands `%create-booster' the training
                     ;; pointer alone makes all three boosters this one, reddening the two
                     ;; assertions above and the one below while leaving every other assertion
                     ;; in this file -- the whole retention test included, which is what shows
                     ;; the two tests are not doing one job twice -- green.
                     (ok (handler-case
                             (progn (cl-gbdt/xgboost:predict unattached wide-matrix) nil)
                           (cl-gbdt/xgboost:foreign-call-error () t))
                         "the unattached booster refused a five-column matrix")
                     ;; The same number read from the training side, and the reason this test
                     ;; needs its own boosters at all: an update requires `num_feature' to
                     ;; EQUAL the training matrix's column count, so neither booster above can
                     ;; be trained. XGBoost says so in as many words -- measured, the message
                     ;; is "Check failed: learner_model_param_.num_feature ==
                     ;; p_fmat->Info().num_col_ (5 vs. 2)". A caller who attaches a validation
                     ;; set of a different width therefore gets a booster that predicts but
                     ;; never trains; that is XGBoost's rule, not this wrapper's, and it is
                     ;; asserted here so that it is recorded rather than rediscovered.
                     (ok (handler-case
                             (progn (cl-gbdt/xgboost:update-one-iteration wide-first) nil)
                           (cl-gbdt/xgboost:foreign-call-error () t))
                         "a five-column validation set left the booster trainable"))
                (progn
                  (cl-gbdt/xgboost:free-booster wide-first)
                  (cl-gbdt/xgboost:free-booster wide-last)
                  (cl-gbdt/xgboost:free-booster unattached)
                  (cl-gbdt/xgboost:free-dataset wide-valid)
                  (cl-gbdt/xgboost:free-dataset narrow-valid)
                  (cl-gbdt/xgboost:free-dataset data))))))))))

(deftest layer-1-alone-refuses-a-bad-validation-set-entry
  (testing "create-booster checks each :valid-sets entry before any foreign call"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let* ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector))
               (freed (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector))
               (other-booster (cl-gbdt/xgboost:create-booster backend data
                                                              :parameters *parameters*)))
          (cl-gbdt/xgboost:free-dataset freed)
          (unwind-protect
               (progn
                 ;; Both entries below are pointers `XGBoosterCreate' would take into its
                 ;; handle array as `DMatrixHandle's -- freed memory in the first case, a
                 ;; booster's own handle in the second. Neither is something the library can
                 ;; reject for us, so the checks have to run in Lisp and before creation;
                 ;; `create-booster' dispatches on nothing, so nothing else stands in the way.
                 (ok (handler-case
                         (progn (cl-gbdt/xgboost:create-booster
                                 backend data :valid-sets (list freed)
                                 :parameters *parameters*)
                                nil)
                       (cl-gbdt/xgboost:released-handle-error () t))
                     "create-booster accepted a freed dataset in :valid-sets")
                 (ok (handler-case
                         (progn (cl-gbdt/xgboost:create-booster
                                 backend data :valid-sets (list other-booster)
                                 :parameters *parameters*)
                                nil)
                       (cl-gbdt/xgboost:wrong-backend-reference () t))
                     "create-booster accepted a booster in :valid-sets"))
            (progn
              (cl-gbdt/xgboost:free-booster other-booster)
              (cl-gbdt/xgboost:free-dataset data))))))))

;;; The mirror of tests/functional/lightgbm-standalone.lisp's
;;; `layer-1-alone-refuses-a-wrong-kind-handle', for the same reason and against the same four
;;; operations: `free-dataset', `free-booster', `predict' and `update-one-iteration' were
;;; `defmethod's specialized on `xgboost-dataset' or `xgboost-booster' until Task 4 made them
;;; the plain `defun's a standalone caller reaches here, and that specializer WAS the type
;;; check. A `defun' takes whatever it is given, and every one of these C entry points
;;; dereferences the pointer it is handed as a handle of the kind it expected. Measured with
;;; the checks removed, all four assertions below redden and three of them do so through an
;;; SBCL CORRUPTION WARNING -- memory faults at the DMatrix's own address, at #x10 and at NIL
;;; -- rather than through anything either library reported. The image survived that run and
;;; the sibling's did not, which is the spread `%check-object-class' in src/xgboost/api.lisp
;;; points at: what a wrong handle does in C is not a property a caller can be told to handle.
;;;
;;; The wrong-KIND half needs one backend and so belongs in this file, which has exactly one;
;;; the wrong-BACKEND half needs both libraries in one image, which this file may not have, and
;;; lives in tests/functional/xgboost-api.lisp as
;;; `xgboost-api-layer-1-refuses-the-other-backends-handles'.

(deftest layer-1-alone-refuses-a-wrong-kind-handle
  (testing "every handle-taking operation checks the kind before any foreign call"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let* ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector))
               (booster (cl-gbdt/xgboost:create-booster backend data
                                                        :parameters *parameters*)))
          (unwind-protect
               (progn
                 ;; `handler-case', not rove's `signals', which does not reliably catch a
                 ;; condition raised inside `restart-case'. On the condition TYPE throughout:
                 ;; the report's wording is not what a caller dispatches on.
                 (ok (handler-case (progn (cl-gbdt/xgboost:free-dataset booster) nil)
                       (cl-gbdt/xgboost:wrong-backend-reference () t))
                     "free-dataset accepted a booster")
                 ;; And it refused before doing anything at all. `release-handle' marks a
                 ;; handle released whichever way the free itself goes, so a kind check placed
                 ;; after it would leave this booster unusable AND unfreeable while still
                 ;; signalling -- an assertion on the condition alone cannot tell the two
                 ;; orders apart. The cleanup form below is what then frees it for real.
                 (ok (not (cl-gbdt/xgboost:handle-released-p booster))
                     "free-dataset released the booster before refusing it")
                 (ok (handler-case (progn (cl-gbdt/xgboost:free-booster data) nil)
                       (cl-gbdt/xgboost:wrong-backend-reference () t))
                     "free-booster accepted a dataset")
                 ;; The same order, pinned for the other free too. Nothing below covers it:
                 ;; `predict' and `update-one-iteration' check the kind BEFORE they check
                 ;; liveness, so a DATA wrongly marked released here would still answer
                 ;; `wrong-backend-reference' and leave both of them green.
                 (ok (not (cl-gbdt/xgboost:handle-released-p data))
                     "free-booster released the dataset before refusing it")
                 (ok (handler-case (progn (cl-gbdt/xgboost:predict data matrix) nil)
                       (cl-gbdt/xgboost:wrong-backend-reference () t))
                     "predict accepted a dataset as its booster")
                 ;; `update-one-iteration' is the one of the four whose check had to move
                 ;; rather than merely appear: `%check-booster-datasets-live' and the
                 ;; `booster-training-set' read both used to run first, and both take slots
                 ;; off whatever they are handed, so a DATASET reached them and failed with a
                 ;; bare CLOS no-applicable-method error instead of this typed condition. This
                 ;; assertion is what pins the order.
                 (ok (handler-case (progn (cl-gbdt/xgboost:update-one-iteration data) nil)
                       (cl-gbdt/xgboost:wrong-backend-reference () t))
                     "update-one-iteration accepted a dataset as its booster"))
            (progn
              (cl-gbdt/xgboost:free-booster booster)
              (cl-gbdt/xgboost:free-dataset data))))))))

(defun model-path (name)
  "Return a pathname for NAME in the system's temporary directory.

`uiop' rather than a new dependency clause: this file's own package form is
`uiop:define-package', so UIOP is already named here, and ASDF vendors it -- unlike anything
under `cl-gbdt/src/', naming it does not widen the closure `tools/ci/check-leaf-systems.lisp'
loads."
  (merge-pathnames name (uiop:temporary-directory)))

(deftest layer-1-alone-saves-loads-and-renders-a-model
  (testing "a caller with only cl-gbdt/xgboost loaded can persist a model and read it back"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector))
              ;; `.json', not an arbitrary name: XGBoost picks its serialization format from
              ;; the extension, and an unrecognized one is its own error rather than this
              ;; test's subject.
              (path (model-path "cl-gbdt-xgboost-standalone.json"))
              (echoed (model-path "cl-gbdt-xgboost-standalone-echo.json")))
          (unwind-protect
               (let ((booster (cl-gbdt/xgboost:create-booster backend data
                                                               :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 20)
                          (cl-gbdt/xgboost:update-one-iteration booster))
                        (ok (equal path (cl-gbdt/xgboost:save-model booster path))
                            "save-model returns the path it was given")
                        (ok (probe-file path) "save-model wrote the file")
                        (let ((reloaded (cl-gbdt/xgboost:load-model backend path)))
                          (unwind-protect
                               (progn
                                 (ok (null (cl-gbdt/xgboost:booster-training-set reloaded))
                                     "a loaded booster has no training set")
                                 (ok (equalp (cl-gbdt/xgboost:predict booster matrix)
                                             (cl-gbdt/xgboost:predict reloaded matrix))
                                     "the reloaded model predicts what the original did"))
                            (cl-gbdt/xgboost:free-booster reloaded)))
                        ;; Asserted by round trip rather than against any substring of
                        ;; XGBoost's JSON: write what it returned, load THAT, require the same
                        ;; predictions.
                        (let ((text (cl-gbdt/xgboost:model-to-string booster)))
                          (ok (and (stringp text) (plusp (length text)))
                              "model-to-string returns a non-empty string")
                          (with-open-file (stream echoed :direction :output
                                                          :if-exists :supersede)
                            (write-string text stream))
                          (let ((from-string (cl-gbdt/xgboost:load-model backend echoed)))
                            (unwind-protect
                                 (ok (equalp (cl-gbdt/xgboost:predict booster matrix)
                                             (cl-gbdt/xgboost:predict from-string matrix))
                                     "model-to-string's text is itself a loadable model")
                              (cl-gbdt/xgboost:free-booster from-string))))
                        ;; The specializer each of these lost. `handler-case', not rove's
                        ;; `signals', which does not reliably catch a condition raised inside
                        ;; `restart-case'.
                        (ok (handler-case (progn (cl-gbdt/xgboost:save-model data path) nil)
                              (cl-gbdt/xgboost:wrong-backend-reference () t))
                            "save-model accepted a dataset as its booster")
                        (ok (handler-case (progn (cl-gbdt/xgboost:model-to-string data) nil)
                              (cl-gbdt/xgboost:wrong-backend-reference () t))
                            "model-to-string accepted a dataset as its booster")
                        (ok (handler-case (progn (cl-gbdt/xgboost:load-model data path) nil)
                              (cl-gbdt/xgboost:wrong-backend-reference () t))
                            "load-model accepted a dataset as its backend")
                        (ok (handler-case (progn (cl-gbdt/xgboost:load-model nil path) nil)
                              (cl-gbdt/xgboost:wrong-backend-reference () t))
                            "load-model accepted NIL as its backend"))
                   (cl-gbdt/xgboost:free-booster booster)))
            (progn
              (cl-gbdt/xgboost:free-dataset data)
              (when (probe-file path) (delete-file path))
              (when (probe-file echoed) (delete-file echoed)))))))))

(deftest layer-1-alone-reports-importance-evaluation-and-shape
  (testing "a caller with only cl-gbdt/xgboost loaded can ask what it just trained"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector)))
          (unwind-protect
               (let ((booster (cl-gbdt/xgboost:create-booster backend data
                                                               :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 20)
                          (cl-gbdt/xgboost:update-one-iteration booster))
                        (ok (= 16 (cl-gbdt/xgboost:dataset-num-rows data))
                            "dataset-num-rows reports the fixture's row count")
                        (ok (= 2 (cl-gbdt/xgboost:dataset-num-features data))
                            "dataset-num-features reports its column count")
                        ;; One entry per FEATURE, which XGBoost's own report is NOT:
                        ;; `XGBoosterFeatureScore' covers only features that appear in a
                        ;; split, so this length is the scatter working rather than the
                        ;; library's own answer passed through.
                        (let ((split (cl-gbdt/xgboost:feature-importance booster))
                              (gain (cl-gbdt/xgboost:feature-importance booster :kind :gain)))
                          (ok (= 2 (length split))
                              (format nil ":split importance has ~D entries" (length split)))
                          (ok (= 2 (length gain))
                              (format nil ":gain importance has ~D entries" (length gain)))
                          (ok (every (lambda (value) (typep value 'double-float)) split)
                              "every :split entry is a double-float")
                          (ok (>= (aref split 0) (aref split 1))
                              (format nil ":split importance ~S ranks column 0 first" split))
                          (ok (>= (aref gain 0) (aref gain 1))
                              (format nil ":gain importance ~S ranks column 0 first" gain))
                          (ok (plusp (aref gain 0))
                              "the signal-carrying column has non-zero gain, so the model split"))
                        (multiple-value-bind (entries provenance)
                            (cl-gbdt/xgboost:evaluation booster)
                          (ok (consp entries) "evaluation reports the training set's metrics")
                          (ok (every (lambda (entry)
                                       (and (= 3 (length entry))
                                            (integerp (first entry))
                                            (stringp (second entry))
                                            (or (null (third entry))
                                                (typep (third entry) 'double-float))))
                                     entries)
                              "each entry is (dataset-index metric-name value-or-nil)")
                          (ok (every (lambda (entry) (zerop (first entry))) entries)
                              "with no validation set every entry is at index 0")
                          (ok (eq :parsed-text (getf provenance :value-source))
                              "evaluation states that it parsed the library's text")
                          (ok (stringp (getf provenance :raw))
                              "and keeps that text unmodified"))
                        ;; The specializer each of these lost.
                        (ok (handler-case
                                (progn (cl-gbdt/xgboost:feature-importance data) nil)
                              (cl-gbdt/xgboost:wrong-backend-reference () t))
                            "feature-importance accepted a dataset as its booster")
                        (ok (handler-case (progn (cl-gbdt/xgboost:evaluation data) nil)
                              (cl-gbdt/xgboost:wrong-backend-reference () t))
                            "evaluation accepted a dataset as its booster")
                        (ok (handler-case
                                (progn (cl-gbdt/xgboost:dataset-num-rows booster) nil)
                              (cl-gbdt/xgboost:wrong-backend-reference () t))
                            "dataset-num-rows accepted a booster as its dataset")
                        (ok (handler-case
                                (progn (cl-gbdt/xgboost:dataset-num-features booster) nil)
                              (cl-gbdt/xgboost:wrong-backend-reference () t))
                            "dataset-num-features accepted a booster as its dataset"))
                   (cl-gbdt/xgboost:free-booster booster)))
            (cl-gbdt/xgboost:free-dataset data)))))))

;;; The `load-model' case, separated for the reason its LightGBM twin is separated: it needs a
;;; second booster and a file, and an empty result is the case a reader is most likely to
;;; mistake for an error.

(deftest layer-1-alone-evaluates-a-loaded-model-as-empty
  (testing "a booster with no retained dataset has nothing to evaluate"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector))
              (path (model-path "cl-gbdt-xgboost-standalone-eval.json")))
          (unwind-protect
               (let ((booster (cl-gbdt/xgboost:create-booster backend data
                                                               :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 5)
                          (cl-gbdt/xgboost:update-one-iteration booster))
                        (cl-gbdt/xgboost:save-model booster path)
                        (let ((reloaded (cl-gbdt/xgboost:load-model backend path)))
                          (unwind-protect
                               (progn
                                 (ok (null (cl-gbdt/xgboost:evaluation reloaded))
                                     "a loaded booster evaluates to no entries")
                                 (ok (= 2 (length (cl-gbdt/xgboost:feature-importance
                                                   reloaded)))
                                     "but still reports one importance per feature"))
                            (cl-gbdt/xgboost:free-booster reloaded))))
                   (cl-gbdt/xgboost:free-booster booster)))
            (progn
              (cl-gbdt/xgboost:free-dataset data)
              (when (probe-file path) (delete-file path)))))))))

;;; `create-dataset-from-file' and the mismatch gate it exists for. Zero-based libsvm column
;;; indices throughout, matching `cl-gbdt/lightgbm:create-dataset-from-file''s own fixtures
;;; (`tests/functional/lightgbm-standalone.lisp'): both libraries read libsvm indices as
;;; zero-based, so a one-based fixture would leave column 0 invented and unused ahead of the
;;; three real columns -- measured directly against the vendored library before either
;;; sibling's fixtures were written this way.

(defmacro with-libsvm-fixture ((path) &body body)
  "Write a four-row three-feature libsvm fixture to a fresh file, bind PATH to it, and
delete it afterwards. Identical content to
`cl-gbdt/tests/functional/lightgbm-standalone::with-libsvm-fixture' -- not shared with it,
per this file's own header, but there is no reason for the two libraries' fixture to differ
when both read libsvm indices zero-based the same way."
  `(let ((,path (merge-pathnames (format nil "cl-gbdt-xgb-fixture-~D.libsvm" (random 1000000))
                                 (uiop:temporary-directory))))
     (unwind-protect
          (progn
            (with-open-file (stream ,path :direction :output :if-exists :supersede)
              (write-string "1 0:1.0 1:2.0 2:3.0
0 0:4.0 1:5.0 2:6.0
1 0:7.0 1:8.0 2:9.0
0 0:10.0 1:11.0 2:12.0
" stream))
            ,@body)
       (ignore-errors (delete-file ,path)))))

(defmacro with-different-libsvm-fixture ((path) &body body)
  "Write a four-row three-feature libsvm fixture whose labels and values both differ from
`with-libsvm-fixture''s, bind PATH to it, and delete it afterwards. Identical content to
`cl-gbdt/tests/functional/lightgbm-standalone::with-different-libsvm-fixture', whose
docstring records why its labels are three 1s and one 0 (mean 0.75) rather than a mere
reordering of `with-libsvm-fixture''s two 1s and two 0s (mean 0.5, which trained an
identical model on that backend's default parameters).

An earlier version of this docstring claimed this backend's `*parameters*' needed no such
care, as measured fact -- that claim was false and review round 1 (Finding 2) caught it:
measured, plain `*parameters*' collapses BOTH this fixture and `with-libsvm-fixture' to a
single-leaf tree on four rows, XGBoost's default `min_child_weight' (1) making
`binary:logistic''s total Hessian exactly 1.0 there too -- the same shape of weakness the
text above correctly denies only for LightGBM's DIFFERENT default,
`min_data_in_leaf'. `create-dataset-from-file-does-not-match-a-different-file' still passed
under that collapse, but only because differing label MEANS (0.75 against 0.5) move
`base_score' even when neither tree splits -- it was never evidence that this fixture's
differing feature VALUES were being read at all, and
`create-dataset-from-file-trains-the-same-model-as-the-matrix' had no such fallback and was
genuinely blind. `model-string-after-one-iteration''s `*file-input-parameters*' now adds
`:min-child-weight 0' for both fixtures, which is what makes this pair -- and the equality
test -- sensitive to feature values rather than only to labels. The content is kept
identical to the sibling's fixture regardless, since there was no reason for the two
libraries' fixture text to differ."
  `(let ((,path (merge-pathnames (format nil "cl-gbdt-xgb-other-fixture-~D.libsvm"
                                         (random 1000000))
                                 (uiop:temporary-directory))))
     (unwind-protect
          (progn
            (with-open-file (stream ,path :direction :output :if-exists :supersede)
              (write-string "1 0:100.0 1:200.0 2:300.0
1 0:50.0 1:60.0 2:70.0
0 0:25.0 1:15.0 2:5.0
1 0:1.0 1:1.0 2:1.0
" stream))
            ,@body)
       (ignore-errors (delete-file ,path)))))

(defmacro with-csv-fixture ((path) &body body)
  "Write a four-row three-feature CSV fixture -- real CSV, comma-delimited, no colons -- to
a fresh file, bind PATH to it, and delete it afterwards. The fixture
`create-dataset-from-file-refuses-a-format-mismatch-before-calling' below declares `:libsvm'
on: `detect-file-format' classifies it `:csv' (a comma on its first line decides that before
any libsvm token shape is even considered), so declaring it `:libsvm' is a real mismatch and
provokes the gate rather than the crash the gate exists to prevent -- see that test's own
comment for why this is the direction that is safe to provoke here."
  `(let ((,path (merge-pathnames (format nil "cl-gbdt-xgb-csv-fixture-~D.csv" (random 1000000))
                                 (uiop:temporary-directory))))
     (unwind-protect
          (progn
            (with-open-file (stream ,path :direction :output :if-exists :supersede)
              (write-string "1,1.0,2.0,3.0
0,4.0,5.0,6.0
1,7.0,8.0,9.0
0,10.0,11.0,12.0
" stream))
            ,@body)
       (ignore-errors (delete-file ,path)))))

(defmacro with-qid-libsvm-fixture ((path) &body body)
  "Write a four-row three-feature libsvm RANKING fixture -- each row carrying a `qid:'
group tag between the label and its feature pairs -- to a fresh file, bind PATH to it,
and delete it afterwards. Two groups of two rows each (`qid:1', `qid:1', `qid:2',
`qid:2'), the same feature values as `with-libsvm-fixture' so a dataset built from this
fixture has the identical row/feature shape and only the group-tag presence differs.

PR #36 review, finding P2: measured directly against the vendored library (scratchpad
qid-measurement.lisp, run in an isolated subprocess since a format/contents mismatch on
this branch is SIGSEGV-reachable) that `XGDMatrixCreateFromURI' declared `:libsvm' reads
this shape cleanly and recovers the group boundaries correctly -- `XGDMatrixGetUIntInfo'
under `\"group_ptr\"' read back `(0 2 4)' for this exact fixture, the correct cumulative
offsets for two rows per group. `detect-file-format' misclassified a `qid'-bearing line as
:CSV before this fix (`%libsvm-token-p' rejected the `qid:1' token outright, no digits
before its colon), which made `create-dataset-from-file' refuse this fixture with
`file-format-mismatch' even though XGBoost itself reads it correctly -- the false-positive
direction the gate must not take, distinct from the SIGSEGV-preventing refusal it exists
for."
  `(let ((,path (merge-pathnames (format nil "cl-gbdt-xgb-qid-fixture-~D.libsvm"
                                         (random 1000000))
                                 (uiop:temporary-directory))))
     (unwind-protect
          (progn
            (with-open-file (stream ,path :direction :output :if-exists :supersede)
              (write-string "1 qid:1 0:1.0 1:2.0 2:3.0
0 qid:1 0:4.0 1:5.0 2:6.0
1 qid:2 0:7.0 1:8.0 2:9.0
0 qid:2 0:10.0 1:11.0 2:12.0
" stream))
            ,@body)
       (ignore-errors (delete-file ,path)))))

(defparameter *file-input-parameters*
  (list* :min-child-weight 0 *parameters*)
  "*PARAMETERS* with `:min-child-weight 0' added on top, for `model-string-after-one-
iteration' alone -- not the file-wide *PARAMETERS* the rest of this file depends on.

Measured (review round 1, Finding 2): on `with-libsvm-fixture''s four rows, under plain
*PARAMETERS*, XGBoost's default `min_child_weight' (1) makes `binary:logistic''s total
Hessian exactly 1.0, so no split can leave both children at >= 1 and the tree collapses to
a single leaf -- `\"num_nodes\":\"1\"' in `model-to-string', identical whatever the feature
values are, so long as the labels are. That left
`create-dataset-from-file-trains-the-same-model-as-the-matrix' blind to whether the file's
feature values were read correctly at all -- the same shape of weakness
`cl-gbdt/lightgbm''s own `min_data_in_leaf' default has, which that backend's
`model-string-after-one-iteration' avoids by passing `*parameters*' to `create-dataset'
too; XGBoost's `create-dataset'/`create-dataset-from-file' has no dataset-level parameters
argument for that fix to reach, so this is a booster-level one instead. `:min-child-weight
0' measured to give `\"num_nodes\":\"7\"' -- the tree actually splits -- restoring the
equality test's sensitivity to feature values without changing what the equality itself
asserts.")

(defun model-string-after-one-iteration (backend dataset)
  "Train one iteration on DATASET and return the resulting model as a string, via
`create-booster' with `*file-input-parameters*' -- `*parameters*' plus `:min-child-weight
0', not the plain `*parameters*' every other booster in this file uses; see
`*file-input-parameters*''s own docstring for why the addition is load-bearing here.
Unlike the sibling's function of this name, this one takes no `:parameters' keyword of its
own to thread through `create-dataset'/`create-dataset-from-file': neither has such an
argument on this backend (see this file's header), so `*file-input-parameters*' reaches
only this one call, and there is nothing else here for a caller to keep in sync."
  (let ((booster (cl-gbdt/xgboost:create-booster backend dataset
                                                  :parameters *file-input-parameters*)))
    (unwind-protect
         (progn
           (cl-gbdt/xgboost:update-one-iteration booster)
           (cl-gbdt/xgboost:model-to-string booster))
      (cl-gbdt/xgboost:free-booster booster))))

(deftest create-dataset-from-file-reads-a-libsvm-file
  (testing "create-dataset-from-file builds a dataset XGBoost reports the right shape for"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        (let ((dataset (cl-gbdt/xgboost:create-dataset-from-file backend path :libsvm)))
          (unwind-protect
               (progn
                 (ok (= 4 (cl-gbdt/xgboost:dataset-num-rows dataset))
                     (format nil "dataset-num-rows is ~D"
                             (cl-gbdt/xgboost:dataset-num-rows dataset)))
                 (ok (= 3 (cl-gbdt/xgboost:dataset-num-features dataset))
                     (format nil "dataset-num-features is ~D"
                             (cl-gbdt/xgboost:dataset-num-features dataset))))
            (cl-gbdt/xgboost:free-dataset dataset)))))))

(deftest create-dataset-from-file-reads-a-qid-ranking-file
  (testing "a valid libsvm ranking file (qid tags) is accepted, not refused as a mismatch"
    (with-open-backend (backend)
      (with-qid-libsvm-fixture (path)
        (let ((dataset (cl-gbdt/xgboost:create-dataset-from-file backend path :libsvm)))
          (unwind-protect
               (progn
                 (ok (= 4 (cl-gbdt/xgboost:dataset-num-rows dataset))
                     (format nil "dataset-num-rows is ~D, not the fixture's 4 -- a qid \
ranking file was not read the same as the identical file without qid tags"
                             (cl-gbdt/xgboost:dataset-num-rows dataset)))
                 (ok (= 3 (cl-gbdt/xgboost:dataset-num-features dataset))
                     (format nil "dataset-num-features is ~D, not the fixture's 3 -- qid \
was read as though it were itself a feature pair"
                             (cl-gbdt/xgboost:dataset-num-features dataset))))
            (cl-gbdt/xgboost:free-dataset dataset)))))))

(deftest create-dataset-from-file-trains-the-same-model-as-the-matrix
  (testing "a dataset read from a file trains identically to the same data given as a matrix"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        (let ((from-file (cl-gbdt/xgboost:create-dataset-from-file backend path :libsvm))
              (from-matrix (cl-gbdt/xgboost:create-dataset
                            backend
                            (make-array '(4 3) :element-type 'double-float
                                               :initial-contents
                                               '((1d0 2d0 3d0) (4d0 5d0 6d0)
                                                 (7d0 8d0 9d0) (10d0 11d0 12d0)))
                            :label '(1d0 0d0 1d0 0d0))))
          (unwind-protect
               (ok (string= (model-string-after-one-iteration backend from-file)
                            (model-string-after-one-iteration backend from-matrix))
                   "a file-built dataset and a matrix-built dataset trained different models")
            (progn
              (cl-gbdt/xgboost:free-dataset from-file)
              (cl-gbdt/xgboost:free-dataset from-matrix))))))))

(deftest create-dataset-from-file-does-not-match-a-different-file
  (testing "a genuinely different file does not train the same model"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        (with-different-libsvm-fixture (other)
          (let ((a (cl-gbdt/xgboost:create-dataset-from-file backend path :libsvm))
                (b (cl-gbdt/xgboost:create-dataset-from-file backend other :libsvm)))
            (unwind-protect
                 (ng (string= (model-string-after-one-iteration backend a)
                              (model-string-after-one-iteration backend b))
                     "two different files trained the same model")
              (progn
                (cl-gbdt/xgboost:free-dataset a)
                (cl-gbdt/xgboost:free-dataset b)))))))))

;;; The test this whole branch exists for. XGBoost does not check a declared FORMAT against a
;;; file's real contents, and measured against the vendored 3.3.0
;;; (docs/superpowers/specs/2026-08-13-file-input-measurements.md section 4), the direction
;;; this test provokes -- text declared a format its first line does not match -- is not
;;; always safe: `train.csv?format=libsvm' SIGSEGVs inside a thread dmlc creates for the
;;; parse, where SBCL reports `Can't handle sig11 in non-lisp thread' and no `handler-case'
;;; anywhere can see it, the process simply being gone. IF THIS TEST EVER FAILS BY CRASHING
;;; THE IMAGE RATHER THAN BY RETURNING NIL, THAT IS THE SEGFAULT, AND THE GATE IS NOT
;;; HOLDING -- a future reader deleting this test because it looks redundant with the layer-1
;;; `file-format-mismatch' plumbing needs to know that a green run of this file is the one
;;; thing standing between a caller's mismatched FORMAT and a dead process. The CSV fixture
;;; declared `:libsvm' is deliberately the SAFE direction to provoke in a live test process:
;;; measured (record section 4), `train.libsvm?format=csv' -- the reverse mismatch -- returns
;;; code 0 and a silently wrong DMatrix rather than crashing, so it could have been used here
;;; too, but the CSV-declared-`:libsvm' direction is the one the gate's own docstring calls
;;; out as fatal, and so the one whose refusal is worth demonstrating.
(deftest create-dataset-from-file-refuses-a-format-mismatch-before-calling
  (testing "a declared format that disagrees with the file's contents is refused, not sent to \
XGDMatrixCreateFromURI"
    (with-open-backend (backend)
      (with-csv-fixture (path)
        (let ((condition (handler-case
                              (progn (cl-gbdt/xgboost:create-dataset-from-file
                                      backend path :libsvm)
                                     nil)
                            (cl-gbdt/xgboost:file-format-mismatch (c) c))))
          (ok condition "create-dataset-from-file did not signal file-format-mismatch")
          (when condition
            ;; All three slots, not just the condition's type: a caller who catches this has
            ;; to be able to read back which of PATH's declared format or its real contents
            ;; to change, which is the whole point of the condition carrying them at all --
            ;; and per Task 2/6, these are the assertions that let the three readers'
            ;; `## Unproven' rows in docs/FUNCTIONAL-COVERAGE.md be deleted rather than left
            ;; permanently loosening that ratchet.
            ;;
            ;; Review round 1, Finding 5: `(equal path (file-format-mismatch-path
            ;; condition))' alone would prove only that the slot echoes back the very Lisp
            ;; object this test already holds, not that it independently names the fixture
            ;; -- an implementation that stored the wrong pathname under some other bug would
            ;; still pass if that pathname happened to be `equal' to PATH by construction.
            ;; Reading the file the slot names back and checking it against the fixture's own
            ;; known first line is independent of that object: it fails if PATH is `equal'
            ;; but wrong in a way `equal' cannot see, and it fails if PATH does not even name
            ;; a readable file, neither of which the identity check alone would catch.
            (ok (equal path (cl-gbdt/xgboost:file-format-mismatch-path condition))
                (format nil "file-format-mismatch-path is ~S, not ~S"
                        (cl-gbdt/xgboost:file-format-mismatch-path condition) path))
            (ok (with-open-file (stream (cl-gbdt/xgboost:file-format-mismatch-path condition))
                  (string= "1,1.0,2.0,3.0" (read-line stream)))
                (format nil "file-format-mismatch-path ~S does not read back as the CSV \
fixture's own first line"
                        (cl-gbdt/xgboost:file-format-mismatch-path condition)))
            (ok (eq :libsvm (cl-gbdt/xgboost:file-format-mismatch-declared condition))
                (format nil "file-format-mismatch-declared is ~S"
                        (cl-gbdt/xgboost:file-format-mismatch-declared condition)))
            (ok (eq :csv (cl-gbdt/xgboost:file-format-mismatch-detected condition))
                (format nil "file-format-mismatch-detected is ~S"
                        (cl-gbdt/xgboost:file-format-mismatch-detected condition)))))))))

(deftest create-dataset-from-file-refuses-bad-arguments
  (testing "create-dataset-from-file checks its backend's class and FORMAT before any foreign \
call, and refuses a missing file itself rather than handing it to XGBoost"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        ;; `handler-case', not rove's `signals' -- see this file's other guard tests for why.
        (ok (handler-case
                (progn (cl-gbdt/xgboost:create-dataset-from-file nil path :libsvm) nil)
              (cl-gbdt/xgboost:wrong-backend-reference () t))
            "create-dataset-from-file accepted NIL as its backend")
        ;; NIL alone would be caught by almost any accidental check; a wrong-CLASS object is
        ;; what the measurements behind `%check-object-class' actually were.
        (let ((wrong-class (cl-gbdt/xgboost:create-dataset-from-file backend path :libsvm)))
          (unwind-protect
               (ok (handler-case
                       (progn (cl-gbdt/xgboost:create-dataset-from-file
                               wrong-class path :libsvm)
                              nil)
                     (cl-gbdt/xgboost:wrong-backend-reference () t))
                   "create-dataset-from-file accepted a dataset as its backend")
            (cl-gbdt/xgboost:free-dataset wrong-class)))
        ;; A FORMAT outside the accepted set -- refused before any foreign call, since
        ;; `detect-file-format' only ever classifies into :LIBSVM, :CSV, :BINARY or :UNKNOWN.
        (ok (handler-case
                (progn (cl-gbdt/xgboost:create-dataset-from-file backend path :tsv) nil)
              (cl-gbdt/xgboost:unsupported-argument () t))
            "create-dataset-from-file accepted :tsv as its format")
        ;; A smuggled `format' key among :uri-parameters -- refused by file-uri's own gate,
        ;; reached before detect-file-format or the foreign call.
        (ok (handler-case
                (progn (cl-gbdt/xgboost:create-dataset-from-file
                        backend path :libsvm :uri-parameters '(:format "csv"))
                       nil)
              (cl-gbdt/xgboost:unsupported-argument () t))
            "create-dataset-from-file accepted a format key among :uri-parameters")
        ;; Review round 2: :UNREADABLE is no longer a pass-through, so a missing file is now
        ;; refused by this wrapper's own gate rather than reaching XGDMatrixCreateFromURI at
        ;; all -- BEHAVIOUR CHANGE from this test's earlier form, which asserted
        ;; `foreign-call-error' here. `detect-file-format' still answers :UNREADABLE for a
        ;; missing file (no probe-file pre-check is added; `open''s own `file-error' inside
        ;; `detect-file-format' is what produces it), but every non-matching verdict is now a
        ;; `file-format-mismatch' from `create-dataset-from-file' itself, DETECTED :unreadable
        ;; included.
        (let ((missing (merge-pathnames "cl-gbdt-xgb-no-such-file.libsvm"
                                        (uiop:temporary-directory))))
          (let ((condition (handler-case
                                (progn (cl-gbdt/xgboost:create-dataset-from-file
                                        backend missing :libsvm)
                                       nil)
                              (cl-gbdt/xgboost:file-format-mismatch (c) c))))
            (ok condition
                "create-dataset-from-file did not signal file-format-mismatch for a missing \
file")
            (when condition
              (ok (eq :unreadable (cl-gbdt/xgboost:file-format-mismatch-detected condition))
                  (format nil "file-format-mismatch-detected is ~S, not :unreadable"
                          (cl-gbdt/xgboost:file-format-mismatch-detected condition))))))))))

(deftest create-dataset-from-file-signals-backend-not-open-after-close
  (testing "create-dataset-from-file signals backend-not-open for a closed backend"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        (cl-gbdt/xgboost:close-backend backend)
        (ok (handler-case
                (progn (cl-gbdt/xgboost:create-dataset-from-file backend path :libsvm) nil)
              (cl-gbdt/xgboost:backend-not-open () t))
            "create-dataset-from-file did not signal backend-not-open")))))

;;; PR #36's second re-review, Critical: a PATH built with sb-ext:parse-native-namestring
;;; carrying a literal '*' or '[' is not a CL wildcard (wild-pathname-p NIL), so the
;;; existing wild-pathname guard never runs, and file-uri would have written that
;;; character unescaped into the URI via native-namestring -- which dmlc's URI layer is
;;; documented to glob-expand. Measured afterward, in an isolated subprocess (a real
;;; glob-expansion reaching a mismatched file is exactly the SIGSEGV shape this whole
;;; branch exists to prevent): the hazard did not reproduce on the vendored XGBoost
;;; 3.3.0 -- see docs/superpowers/specs/2026-08-13-file-input-measurements.md section 13
;;; for the full record, including why this refusal is precautionary rather than a fix
;;; for a demonstrated crash. This is the mandatory test the coordinator named: two real
;;; files that both match the literal name as a glob pattern, one libsvm and one CSV, so
;;; an implementation that failed to refuse would have a real cross-contamination shape
;;; to reach, not a merely theoretical one.
(deftest create-dataset-from-file-refuses-a-literal-glob-metacharacter-with-two-matching-files
  (testing "a literal * that wild-pathname-p does not see, with a second file the glob \
pattern would match, is refused before any foreign call"
    (with-open-backend (backend)
      (let* ((dir (merge-pathnames (format nil "cl-gbdt-xgb-glob-~D/" (random 1000000))
                                   (uiop:temporary-directory)))
             (named-path (progn
                           (ensure-directories-exist dir)
                           (merge-pathnames
                            (sb-ext:parse-native-namestring "star*file.libsvm") dir)))
             (other-path (merge-pathnames
                          (sb-ext:parse-native-namestring "star-csv-file.libsvm") dir)))
        (ok (null (wild-pathname-p named-path))
            "test setup: named-path must not be a CL wildcard, or the existing guard \
would catch it instead of the one under test")
        (unwind-protect
             (progn
               ;; The named file: real libsvm.
               (with-open-file (stream named-path :direction :output :if-exists :supersede)
                 (write-string "1 0:1.0 1:2.0 2:3.0
0 0:4.0 1:5.0 2:6.0
" stream))
               ;; A second file the literal name also matches as a glob pattern
               ;; ("star*file.libsvm" -> "star" + any chars + "file.libsvm") -- real CSV,
               ;; the format/contents mismatch that SIGSEGVs XGBoost if a glob ever
               ;; reaches it declared :libsvm.
               (with-open-file (stream other-path :direction :output :if-exists :supersede)
                 (write-string "1,1.0,2.0,3.0
0,4.0,5.0,6.0
0,7.0,8.0,9.0
" stream))
               (ok (handler-case
                       (progn (cl-gbdt/xgboost:create-dataset-from-file
                               backend named-path :libsvm)
                              nil)
                     (cl-gbdt/xgboost:unsupported-argument () t))
                   "create-dataset-from-file accepted a literal * with a second file the \
glob pattern would match"))
          (progn
            (handler-case (delete-file named-path) (file-error () nil))
            (handler-case (delete-file other-path) (file-error () nil))
            (handler-case (uiop:delete-directory-tree dir :validate t)
              (file-error () nil))))))))

(deftest save-model-resolves-a-relative-path-against-default-pathname-defaults
  (testing "a relative PATH is written the way `open' would resolve it, not native-namestring'd \
bare -- proven by a same-named decoy file sitting in the process's own working directory, \
which the unresolved form would have overwritten instead"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let* ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector))
               (relative-name (format nil "cl-gbdt-p6-save-~D.json" (random 1000000)))
               (real-dir (ensure-directories-exist
                          (merge-pathnames (format nil "cl-gbdt-p6-save-real-~D/"
                                                    (random 1000000))
                                           (uiop:temporary-directory))))
               (real-path (merge-pathnames relative-name real-dir))
               (decoy-path (merge-pathnames relative-name (uiop:getcwd)))
               (decoy-content "not an xgboost model, just a decoy sentinel"))
          (unwind-protect
               (let ((booster (cl-gbdt/xgboost:create-booster backend data
                                                               :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 5) (cl-gbdt/xgboost:update-one-iteration booster))
                        ;; Plant the decoy BEFORE saving, so an unresolved bare `namestring'
                        ;; would have overwritten it -- the fix must leave it alone.
                        (with-open-file (stream decoy-path :direction :output
                                                            :if-exists :supersede)
                          (write-string decoy-content stream))
                        (let ((*default-pathname-defaults* real-dir))
                          (cl-gbdt/xgboost:save-model booster relative-name))
                        (ok (probe-file real-path)
                            "save-model did not write to *default-pathname-defaults*'s \
directory")
                        (ok (string= decoy-content
                                     (with-open-file (stream decoy-path)
                                       (let ((buffer (make-string (length decoy-content))))
                                         (read-sequence buffer stream)
                                         buffer)))
                            "save-model overwrote the decoy in the process's own working \
directory instead of writing to *default-pathname-defaults*'s")
                        (let ((reloaded (cl-gbdt/xgboost:load-model backend real-path)))
                          (unwind-protect
                               (ok (equalp (cl-gbdt/xgboost:predict booster matrix)
                                           (cl-gbdt/xgboost:predict reloaded matrix))
                                   "the model written to *default-pathname-defaults*'s \
directory does not round-trip")
                            (cl-gbdt/xgboost:free-booster reloaded))))
                   (cl-gbdt/xgboost:free-booster booster)))
            (progn
              (cl-gbdt/xgboost:free-dataset data)
              (handler-case (delete-file real-path) (file-error () nil))
              (handler-case (uiop:delete-directory-tree real-dir :validate t)
                (file-error () nil))
              (handler-case (delete-file decoy-path) (file-error () nil)))))))))

(deftest save-model-refuses-a-wild-path
  (testing "save-model refuses a wild pathname before any foreign call"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector)))
          (unwind-protect
               (let ((booster (cl-gbdt/xgboost:create-booster backend data
                                                               :parameters *parameters*)))
                 (unwind-protect
                      (let ((path (merge-pathnames (pathname "cl-gbdt-wild-save*.json")
                                                    (uiop:temporary-directory))))
                        (ok (handler-case
                                (progn (cl-gbdt/xgboost:save-model booster path) nil)
                              (cl-gbdt/xgboost:unsupported-argument () t))
                            "save-model accepted a wild pathname"))
                   (cl-gbdt/xgboost:free-booster booster)))
            (cl-gbdt/xgboost:free-dataset data)))))))

(deftest save-model-writes-a-file-with-a-literal-asterisk
  (testing "a filename whose namestring backslash-escapes a literal asterisk still saves"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector))
              (path (merge-pathnames
                     (sb-ext:parse-native-namestring
                      (format nil "cl-gbdt-star*save-~D.json" (random 1000000)))
                     (uiop:temporary-directory))))
          (unwind-protect
               (let ((booster (cl-gbdt/xgboost:create-booster backend data
                                                               :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 5) (cl-gbdt/xgboost:update-one-iteration booster))
                        (ok (equal path (cl-gbdt/xgboost:save-model booster path))
                            "save-model did not return the literal-asterisk path it was \
given")
                        (ok (probe-file path)
                            "save-model did not write the literal-asterisk path it was \
given -- a bare namestring would have escaped it to a different, nonexistent name"))
                   (cl-gbdt/xgboost:free-booster booster)))
            (progn
              (cl-gbdt/xgboost:free-dataset data)
              (handler-case (delete-file path) (file-error () nil)))))))))

(deftest load-model-resolves-a-relative-path-against-default-pathname-defaults
  (testing "a relative PATH is read the way `open' would resolve it, not native-namestring'd \
bare -- proven by a same-named decoy file sitting in the process's own working directory, \
which the unresolved form would have loaded instead"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let* ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector))
               (relative-name (format nil "cl-gbdt-p6-load-~D.json" (random 1000000)))
               (real-dir (ensure-directories-exist
                          (merge-pathnames (format nil "cl-gbdt-p6-load-real-~D/"
                                                    (random 1000000))
                                           (uiop:temporary-directory))))
               (real-path (merge-pathnames relative-name real-dir))
               (decoy-path (merge-pathnames relative-name (uiop:getcwd))))
          (unwind-protect
               (let ((booster (cl-gbdt/xgboost:create-booster backend data
                                                               :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 5) (cl-gbdt/xgboost:update-one-iteration booster))
                        ;; The REAL model, written with an already-absolute path, so this
                        ;; fixture does not itself depend on `save-model''s own fix.
                        (cl-gbdt/xgboost:save-model booster real-path)
                        ;; The DECOY: not a valid model at all, so an unresolved bare
                        ;; `namestring' loading it instead of the real file fails this test
                        ;; either by signalling or by a wrong prediction -- both a clear
                        ;; failure of the assertion below.
                        (with-open-file (stream decoy-path :direction :output
                                                            :if-exists :supersede)
                          (write-string "not an xgboost model" stream))
                        ;; A single-binding `let' nested inside another, not `let*': see
                        ;; the sibling's `create-dataset-from-file-resolves-a-relative-path-
                        ;; against-default-pathname-defaults' for why -- `reloaded''s
                        ;; init-form must run AFTER *default-pathname-defaults* is rebound, a
                        ;; dynamic dependency mallet's own let*-vs-let linter cannot see
                        ;; across a special-variable rebinding, since RELOADED's init-form
                        ;; never mentions that symbol by name.
                        (let ((*default-pathname-defaults* real-dir))
                          (let ((reloaded (cl-gbdt/xgboost:load-model
                                           backend relative-name)))
                            (unwind-protect
                                 (ok (equalp (cl-gbdt/xgboost:predict booster matrix)
                                             (cl-gbdt/xgboost:predict reloaded matrix))
                                     "load-model read the decoy in the process's own \
working directory instead of *default-pathname-defaults*'s")
                              (cl-gbdt/xgboost:free-booster reloaded)))))
                   (cl-gbdt/xgboost:free-booster booster)))
            (progn
              (cl-gbdt/xgboost:free-dataset data)
              (handler-case (delete-file real-path) (file-error () nil))
              (handler-case (uiop:delete-directory-tree real-dir :validate t)
                (file-error () nil))
              (handler-case (delete-file decoy-path) (file-error () nil)))))))))

(deftest load-model-refuses-a-wild-path
  (testing "load-model refuses a wild pathname before any foreign call"
    (with-open-backend (backend)
      (let ((path (merge-pathnames (pathname "cl-gbdt-wild-load*.json")
                                    (uiop:temporary-directory))))
        (ok (handler-case (progn (cl-gbdt/xgboost:load-model backend path) nil)
              (cl-gbdt/xgboost:unsupported-argument () t))
            "load-model accepted a wild pathname")))))

(deftest load-model-reads-a-file-with-a-literal-asterisk
  (testing "a filename whose namestring backslash-escapes a literal asterisk still loads"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let ((data (cl-gbdt/xgboost:create-dataset backend matrix :label label-vector))
              (path (merge-pathnames
                     (sb-ext:parse-native-namestring
                      (format nil "cl-gbdt-star*load-~D.json" (random 1000000)))
                     (uiop:temporary-directory))))
          (unwind-protect
               (let ((booster (cl-gbdt/xgboost:create-booster backend data
                                                               :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 5) (cl-gbdt/xgboost:update-one-iteration booster))
                        ;; Written directly via `model-to-string', bypassing `save-model'
                        ;; entirely, so this test isolates `load-model''s own path handling
                        ;; from `save-model''s.
                        (with-open-file (stream path :direction :output
                                                      :if-exists :supersede)
                          (write-string (cl-gbdt/xgboost:model-to-string booster) stream))
                        (let ((reloaded (cl-gbdt/xgboost:load-model backend path)))
                          (unwind-protect
                               (ok (equalp (cl-gbdt/xgboost:predict booster matrix)
                                           (cl-gbdt/xgboost:predict reloaded matrix))
                                   "load-model did not read the literal-asterisk path it \
was given -- a bare namestring would have escaped it to a different, nonexistent name")
                            (cl-gbdt/xgboost:free-booster reloaded))))
                   (cl-gbdt/xgboost:free-booster booster)))
            (progn
              (cl-gbdt/xgboost:free-dataset data)
              (handler-case (delete-file path) (file-error () nil)))))))))
