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
;;;; see the commentary there. The split is not cosmetic: `train' does not call
;;;; `create-booster' (see the comment at its own `%create-booster' call in
;;;; src/xgboost/protocol.lisp for the two measured reasons), so the two are separate copies
;;;; of one procedure and something has to hold them together.

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
