;;;; lightgbm-standalone.lisp --- Layer 1 trains and predicts with no unified API in the image.
;;;;
;;;; This file names `cl-gbdt/lightgbm' -- the PUBLIC package, not the internal
;;;; `cl-gbdt/src/lightgbm/all' aggregation -- and no other system of this project. No
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
;;;; shared ones, for the same reason -- and needs strikingly little to do so: this backend's
;;;; `open-backend' performs its own discovery (env var, then the vendored directory, then
;;;; CFFI's system search -- `cl-gbdt/src/library''s `resolve-and-load-library'), so unlike
;;;; every other file in tests/functional/ this one loads no shared library by hand at all.
;;;;
;;;; What this file deliberately does NOT assert: that Layer 1's `create-booster' plus a loop
;;;; of `update-one-iteration' produces the same model as `cl-gbdt:train'. That comparison
;;;; needs both APIs in one image, which is the one thing this file may not have, so it lives
;;;; in tests/functional/lightgbm-api.lisp as
;;;; `lightgbm-api-create-booster-and-train-agree' -- see the commentary there, which carries
;;;; the other half of this note. The split is not cosmetic even now that `train' calls
;;;; `create-booster' for its whole construction
;;;; (.superpowers/sdd/2026-08-12-train-create-booster-merge): this file's only dependency is
;;;; `cl-gbdt/lightgbm', Layer 1 alone, so `train' is not a name it can even read here, let
;;;; alone call to check against.

(uiop:define-package #:cl-gbdt/tests/functional/lightgbm-standalone
  (:use #:cl #:rove)
  ;; The one and only `cl-gbdt' edge, and it names zero symbols: every call below is
  ;; package-qualified, so nothing needs importing, and the clause is here to declare the
  ;; dependency itself. ASDF resolves the name to the system of the same name in
  ;; `cl-gbdt.asd', whose closure is Layer 1 alone -- that closure is what this file exists
  ;; to test, so widening this clause to `cl-gbdt/src/lightgbm/all' (every `%'-function
  ;; besides) or to anything Layer 2 would silently destroy the test rather than fail it.
  (:import-from #:cl-gbdt/lightgbm))

(in-package #:cl-gbdt/tests/functional/lightgbm-standalone)

(defparameter *parameters*
  '(:objective "binary" :num-leaves 3 :min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)
  "LightGBM's own vocabulary, as `create-dataset' and `create-booster' take it -- one plist
handed to both, since nothing at this layer translates or filters a key and LightGBM ignores
the ones that do not apply to the call it is given.

`min_data_in_leaf 1' is required, and measured so against the vendored library: the default
is 20, more than this fixture's sixteen rows, so no split ever passes and all sixteen
predictions come back as exactly 0.5 -- the separation assertion below fails for a reason
that has nothing to do with Layer 1. `min_data_in_bin 1' is NOT required -- with the default
of 3 the same run still separates, 0.380 against 0.816 -- and is here to make the margin
crisp (0.066 against 0.934) instead of leaving it to wherever the default binning's
boundaries happen to fall relative to the label boundary at row 8. `verbose -1' keeps
LightGBM's own `[LightGBM] [Info]' and `[Warning]' lines off the suite's output; measured,
those lines go to standard output rather than to standard error, so nothing else here would
have kept them out of a test run's report.")

(defun fixture ()
  "Return (values MATRIX LABEL-VECTOR): sixteen rows whose label is a step function of
column 0, which any working booster separates.

Column 0 counts 0..15 and carries the whole signal; column 1 cycles 0..3 and carries none,
so a booster that ignored the matrix's column order -- reading it transposed, say -- could
not produce the ordering asserted below. Both arrays are `double-float': the label field is
LightGBM's `float32' and `create-dataset' coerces on the way in, so a standalone caller need
not know that."
  (let ((matrix (make-array '(16 2) :element-type 'double-float))
        (label-vector (make-array 16 :element-type 'double-float)))
    (dotimes (row 16)
      (setf (aref matrix row 0) (coerce row 'double-float)
            (aref matrix row 1) (coerce (mod row 4) 'double-float)
            (aref label-vector row) (if (< row 8) 0d0 1d0)))
    (values matrix label-vector)))

(defun validation-fixture ()
  "Return (values MATRIX LABEL-VECTOR): four held-out rows, three of which `fixture''s step
function explains and one of which it contradicts.

Column 0 is 1.5, 3.5, 5.5 and 9.5 -- values that appear nowhere in `fixture''s matrix, so a
dataset built from these bins independently unless `create-dataset' is given `:reference'.
Every label is 0, which makes the last row a deliberate error: a booster trained on
`fixture' puts column-0 9.5 firmly in the positive half. That is what gives this dataset a
`binary_logloss' an order of magnitude worse than the training set's -- measured, 0.73
against 0.068 -- and so what lets `layer-1-alone-attaches-a-validation-set' tell a real
validation set at index 1 from the training set attached twice."
  (let ((matrix (make-array '(4 2) :element-type 'double-float))
        (label-vector (make-array 4 :element-type 'double-float)))
    (dotimes (row 4)
      (setf (aref matrix row 0) (if (= row 3) 9.5d0 (+ 1.5d0 (* 2 row)))
            (aref matrix row 1) (coerce row 'double-float)
            (aref label-vector row) 0d0))
    (values matrix label-vector)))

(defmacro with-open-backend ((backend) &body body)
  "Evaluate BODY with BACKEND bound to an open LightGBM backend, closing it afterwards, or
skip when the shared library is absent.

A missing library is a skip rather than a failure, because `vendor/' is git-ignored and a
fresh clone legitimately has none -- the same judgement tests/functional/support.lisp's
`with-backend-library' makes, reached here without that file. The condition is
`open-backend''s own `backend-library-not-found', caught by a name `cl-gbdt/lightgbm'
publishes, so this skip needs no discovery logic of its own and no internal package.

`rove:skip' records a pending assertion and RETURNS rather than unwinding, so BODY is
guarded by an explicit test on BACKEND rather than left to run on with none to use."
  (let ((condition (gensym "CONDITION")))
    `(let ((,backend (handler-case (cl-gbdt/lightgbm:open-backend :lightgbm)
                       (cl-gbdt/lightgbm:backend-library-not-found (,condition)
                         (skip (princ-to-string ,condition))
                         nil))))
       (when ,backend
         (unwind-protect (progn ,@body)
           (cl-gbdt/lightgbm:close-backend ,backend))))))

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

(defun model-path (name)
  "Return a pathname for NAME in the system's temporary directory.

`uiop' rather than a new dependency clause: this file's own package form is
`uiop:define-package', so UIOP is already named here, and ASDF vendors it -- unlike anything
under `cl-gbdt/src/', naming it does not widen the closure `tools/ci/check-leaf-systems.lisp'
loads. Writing into the current directory instead would leave a model file in whatever
directory the suite happened to run from."
  (merge-pathnames name (uiop:temporary-directory)))

(defmacro with-libsvm-fixture ((path) &body body)
  "Write a four-row three-feature libsvm fixture to a fresh file, bind PATH to it, and
delete it afterwards.

Column indices are zero-based (`0:', `1:', `2:') -- LightGBM's own convention for the format,
its \"Unknown format of training data\" error naming it \"LibSVM (zero-based)\" verbatim.
Measured directly against the vendored library (`repl-eval', 2026-08-13): the same four rows
written with one-based indices (`1:'/`2:'/`3:', the spelling
`docs/superpowers/specs/2026-08-13-file-input-measurements.md''s own `train.libsvm' fixture
uses) read back as FOUR features, not three -- index N names feature N of a zero-based space,
so a one-based file leaves feature 0 unused and invented ahead of the three real ones.
Zero-based indices read as exactly three, which is what lines this fixture up column-for-column
against `create-dataset-from-file-trains-the-same-model-as-the-matrix''s dense matrix below.

No leading blank line: `docs/superpowers/specs/2026-08-13-file-input-measurements.md' section
8 measured that two or more leading blank lines make LightGBM invent an extra row, so a
fixture that opened with one would fail the row-count assertion for a reason that has nothing
to do with this branch."
  `(let ((,path (merge-pathnames (format nil "cl-gbdt-fixture-~D.libsvm" (random 1000000))
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
`with-libsvm-fixture''s, bind PATH to it, and delete it afterwards.

Three of its four labels are 1, where `with-libsvm-fixture' splits its four evenly -- the
label MEANS differ (0.75 against 0.5), not merely their order. `model-string-after-one-
iteration' trains with no `:parameters' at all, so `min_data_in_leaf''s default of 20
exceeds either fixture's four rows and neither booster ever splits: each model is a single
leaf holding its training set's mean label. A fixture that only reordered the same two 1s
and two 0s would leave that mean at 0.5 in both cases and
`create-dataset-from-file-does-not-match-a-different-file' would then compare two identical
strings by accident -- measured against the vendored library before this fixture's labels
were changed to fix exactly that."
  `(let ((,path (merge-pathnames (format nil "cl-gbdt-other-fixture-~D.libsvm" (random 1000000))
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

(defun model-string-after-one-iteration (backend dataset &key parameters)
  "Train one iteration on DATASET and return the resulting model as a string.

PARAMETERS is passed to `create-booster' unchanged. Measured (review of this file's first
version, 2026-08-13): with no `:parameters' at all, LightGBM's default `min_data_in_leaf'
(20) exceeds every file-input fixture's four rows, so the booster never splits and the whole
model is a single leaf holding the training set's mean label -- `feature_infos=none none
none'. A comparison built on that alone cannot tell whether the feature VALUES were even
read, only the labels, which is why both `create-dataset-from-file-trains-the-same-model-as-
the-matrix' and its control below pass `*parameters*' -- the same plist already used for
every other booster in this file -- so the trees actually split and the comparison binds the
feature ranges and thresholds, not merely the label mean."
  (let ((booster (cl-gbdt/lightgbm:create-booster backend dataset :parameters parameters)))
    (unwind-protect
         (progn
           (cl-gbdt/lightgbm:update-one-iteration booster)
           (cl-gbdt/lightgbm:model-to-string booster))
      (cl-gbdt/lightgbm:free-booster booster))))

(deftest layer-1-alone-trains-and-predicts
  (testing "a caller with only cl-gbdt/lightgbm loaded can go from a matrix to predictions"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let ((data (cl-gbdt/lightgbm:create-dataset backend matrix :label label-vector
                                                     :parameters *parameters*)))
          (unwind-protect
               (let ((booster (cl-gbdt/lightgbm:create-booster backend data
                                                               :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 20)
                          (cl-gbdt/lightgbm:update-one-iteration booster))
                        (multiple-value-bind (result shape)
                            (cl-gbdt/lightgbm:predict booster matrix)
                          ;; One prediction per row and a single output group: a booster
                          ;; built with the wrong `num_class', or a buffer sized from the row
                          ;; count alone, would not have this shape.
                          (ok (equal '(16 1) (array-dimensions result))
                              (format nil "predict's result is ~S"
                                      (array-dimensions result)))
                          ;; `predict''s SECOND value. For `:normal', `%prediction-shape'
                          ;; returns the result array's own dimensions verbatim, so the
                          ;; numbers here are the ones the assertion above already examined
                          ;; and this adds nothing about them. What it adds is that Layer 1
                          ;; states a shape AT ALL: a `predict' that returned the array and
                          ;; nothing else would leave a standalone caller to infer one, and
                          ;; only this assertion notices.
                          (ok (equal '(16 1) shape)
                              (format nil "predict states the shape ~S" shape))
                          ;; The one assertion here that needs the booster to have actually
                          ;; TRAINED, and so the one a no-op `update-one-iteration' would
                          ;; fail: measured against the vendored library, a booster straight
                          ;; out of `create-booster' predicts exactly 0.5 for all sixteen
                          ;; rows -- with the shape the two assertions above check, which is
                          ;; why neither of them can stand in for this one.
                          (ok (separates-p result label-vector)
                              (format nil "predictions: ~S" result))))
                   (cl-gbdt/lightgbm:free-booster booster)))
            (cl-gbdt/lightgbm:free-dataset data)))))))

;;; `create-booster''s :VALID-SETS half is invisible to everything above, and to
;;; `lightgbm-api-create-booster-and-train-agree' in tests/functional/lightgbm-api.lisp as
;;; well: attaching a validation set does not change the model LightGBM trains, so comparing
;;; PREDICTIONS -- either against a label boundary here or against `cl-gbdt:train''s own run
;;; there -- is blind to `%add-valid-data', to the `copy-list' snapshot of the caller's list,
;;; and to the per-entry `%check-lightgbm-dataset'. `train' has its own copy of that same
;;; procedure and other tests DO exercise it with validation sets, so without the two tests
;;; below `create-booster''s copy could drift alone and every suite stay green.
;;;
;;; The assertions are therefore chosen to be ones a prediction cannot make: what the booster
;;; retained, what the library will evaluate, and what it refuses before any foreign call.
;;; None of it needs the unified API, which is why it belongs in this file rather than beside
;;; the agreement test.

(deftest layer-1-alone-attaches-a-validation-set
  (testing "create-booster's :valid-sets reaches the library and the booster keeps its own view"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (multiple-value-bind (valid-matrix valid-label-vector) (validation-fixture)
          (let* ((data (cl-gbdt/lightgbm:create-dataset backend matrix :label label-vector
                                                        :parameters *parameters*))
                 ;; `:reference data', because `validation-fixture' shares no column-0 value
                 ;; with `fixture': without it each of these bins independently and
                 ;; `LGBM_BoosterAddValidData' refuses the mismatched bin mapper outright.
                 (valid-1 (cl-gbdt/lightgbm:create-dataset
                           backend valid-matrix :label valid-label-vector
                           :reference data :parameters *parameters*))
                 (valid-2 (cl-gbdt/lightgbm:create-dataset
                           backend valid-matrix :label valid-label-vector
                           :reference data :parameters *parameters*))
                 ;; The caller's own list object, kept so it can be mutated below.
                 (callers-valid-sets (list valid-1 valid-2))
                 (booster (cl-gbdt/lightgbm:create-booster
                           backend data :valid-sets callers-valid-sets
                           :parameters *parameters*)))
            (unwind-protect
                 (progn
                   (dotimes (round 20)
                     (cl-gbdt/lightgbm:update-one-iteration booster))
                   ;; Retention, both halves. A `create-booster' that attached the datasets
                   ;; to LightGBM but kept neither would train and predict identically --
                   ;; and leave a standalone caller no way to ask what its booster depends
                   ;; on, and `%check-booster-datasets-live' nothing to check.
                   (ok (eq data (cl-gbdt/lightgbm:booster-training-set booster))
                       "the booster reports the dataset it was built over")
                   (ok (equal (list valid-1 valid-2)
                              (cl-gbdt/lightgbm:booster-validation-sets booster))
                       (format nil "booster-validation-sets is ~S"
                               (cl-gbdt/lightgbm:booster-validation-sets booster)))
                   ;; The `copy-list'. Truncating the caller's own list is what a `delete' of
                   ;; the second entry does; were the booster holding that list object rather
                   ;; than a snapshot, this would silently remove VALID-2 from its view while
                   ;; LightGBM still held that dataset's pointer -- and the freed-set
                   ;; assertion at the end of this test would then not signal at all.
                   (setf (cdr callers-valid-sets) nil)
                   (ok (equal (list valid-1 valid-2)
                              (cl-gbdt/lightgbm:booster-validation-sets booster))
                       (format nil "after the caller truncated its own list: ~S"
                               (cl-gbdt/lightgbm:booster-validation-sets booster)))
                   ;; Attachment itself, read back through the library rather than through
                   ;; Lisp-side bookkeeping: `LGBM_BoosterGetEval' rejects an index past the
                   ;; last attached dataset, so index 2 answering at all is what proves the
                   ;; SECOND entry reached `LGBM_BoosterAddValidData' -- an attach loop that
                   ;; stopped after the head would signal `foreign-call-error' here.
                   (ok (= (length (cl-gbdt/lightgbm:booster-eval-names booster))
                          (length (cl-gbdt/lightgbm:booster-eval booster 2)))
                       (format nil "index 2 reports ~S for metrics ~S"
                               (cl-gbdt/lightgbm:booster-eval booster 2)
                               (cl-gbdt/lightgbm:booster-eval-names booster)))
                   ;; And that the two indices address two different datasets, rather than
                   ;; one dataset answering for both: `validation-fixture' contains a row the
                   ;; training data contradicts, so its binary_logloss is an order of
                   ;; magnitude worse -- measured, 0.73 against 0.068. That gap is what makes
                   ;; the indices distinguishable at all. Measured by mutation: a
                   ;; `%booster-eval' that passes 0 to `LGBM_BoosterGetEval' whatever
                   ;; DATA-INDEX it was given leaves every other assertion in this file green
                   ;; -- including the index-2 one above, 0 being in range always -- and
                   ;; reddens this one alone.
                   (ok (> (aref (cl-gbdt/lightgbm:booster-eval booster 1) 0)
                          (aref (cl-gbdt/lightgbm:booster-eval booster 0) 0))
                       (format nil "training set ~S, validation set ~S"
                               (cl-gbdt/lightgbm:booster-eval booster 0)
                               (cl-gbdt/lightgbm:booster-eval booster 1)))
                   ;; Last, because it poisons the booster: freeing a validation set out from
                   ;; under it has to be noticed BEFORE the next foreign call, since
                   ;; `LGBM_DatasetFree' does not clear the pointer LightGBM kept and
                   ;; `LGBM_BoosterUpdateOneIter' would dereference it -- a segfault, not a
                   ;; catchable condition. This holds `create-booster''s retained snapshot to
                   ;; the standard `train''s already meets.
                   (cl-gbdt/lightgbm:free-dataset valid-2)
                   ;; `handler-case', not rove's `signals', which does not reliably catch a
                   ;; condition raised inside `restart-case'.
                   (ok (handler-case
                           (progn (cl-gbdt/lightgbm:update-one-iteration booster) nil)
                         (cl-gbdt/lightgbm:released-handle-error () t))
                       "update-one-iteration did not signal released-handle-error"))
              (progn
                (cl-gbdt/lightgbm:free-booster booster)
                (cl-gbdt/lightgbm:free-dataset valid-1)
                (cl-gbdt/lightgbm:free-dataset data)))))))))

(deftest layer-1-alone-refuses-a-bad-validation-set-entry
  (testing "create-booster checks each :valid-sets entry before any foreign call"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let* ((data (cl-gbdt/lightgbm:create-dataset backend matrix :label label-vector
                                                      :parameters *parameters*))
               (freed (cl-gbdt/lightgbm:create-dataset backend matrix :label label-vector
                                                       :parameters *parameters*))
               (other-booster (cl-gbdt/lightgbm:create-booster backend data
                                                               :parameters *parameters*)))
          (cl-gbdt/lightgbm:free-dataset freed)
          (unwind-protect
               (progn
                 ;; Both entries below are pointers `LGBM_BoosterAddValidData' would
                 ;; dereference as `DatasetHandle's -- freed memory in the first case, a
                 ;; booster's own handle in the second. Neither is something the library can
                 ;; reject for us, so the checks have to run in Lisp and before creation;
                 ;; `create-booster' dispatches on nothing, so nothing else stands in the way.
                 (ok (handler-case
                         (progn (cl-gbdt/lightgbm:create-booster
                                 backend data :valid-sets (list freed)
                                 :parameters *parameters*)
                                nil)
                       (cl-gbdt/lightgbm:released-handle-error () t))
                     "create-booster accepted a freed dataset in :valid-sets")
                 (ok (handler-case
                         (progn (cl-gbdt/lightgbm:create-booster
                                 backend data :valid-sets (list other-booster)
                                 :parameters *parameters*)
                                nil)
                       (cl-gbdt/lightgbm:wrong-backend-reference () t))
                     "create-booster accepted a booster in :valid-sets"))
            (progn
              (cl-gbdt/lightgbm:free-booster other-booster)
              (cl-gbdt/lightgbm:free-dataset data))))))))

;;; Four of the operations this file calls -- `free-dataset', `free-booster', `predict' and
;;; `update-one-iteration' -- were `defmethod's specialized on `lightgbm-dataset' or
;;; `lightgbm-booster' until Task 4 made them the plain `defun's a standalone caller reaches
;;; here. That specializer WAS the type check, and a `defun' takes whatever it is given: for a
;;; while after the split each of the four handed a caller's pointer to a C entry point that
;;; dereferences it as a handle of the kind it was expecting. Measured with the checks removed,
;;; this test does not merely redden: the first assertion below faults at #xFFFFFFFFFFFFFFF8
;;; inside `LGBM_DatasetFree', and the image then aborts -- glibc's "free(): invalid pointer",
;;; SIGABRT -- when the cleanup form goes on to free that same booster properly, so the other
;;; three assertions never get to run. `%check-object-class' in src/lightgbm/api.lisp carries
;;; the rest of that measurement.
;;;
;;; This test covers the wrong-KIND half -- a booster where a dataset was wanted and the other
;;; way round -- which needs one backend, and so belongs in this file, which has exactly one.
;;; The wrong-BACKEND half, an XGBoost handle reaching these same four, needs both libraries in
;;; one image; that is the one thing this file may not have, so it lives in
;;; tests/functional/lightgbm-api.lisp as
;;; `lightgbm-api-layer-1-refuses-the-other-backends-handles'. The two halves are one defect and
;;; the check that stops them is one `typep' either way; only the fixture differs.

(deftest layer-1-alone-refuses-a-wrong-kind-handle
  (testing "every handle-taking operation checks the kind before any foreign call"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let* ((data (cl-gbdt/lightgbm:create-dataset backend matrix :label label-vector
                                                      :parameters *parameters*))
               (booster (cl-gbdt/lightgbm:create-booster backend data
                                                         :parameters *parameters*)))
          (unwind-protect
               (progn
                 ;; `handler-case', not rove's `signals', which does not reliably catch a
                 ;; condition raised inside `restart-case'. On the condition TYPE throughout:
                 ;; the report's wording is not what a caller dispatches on.
                 (ok (handler-case (progn (cl-gbdt/lightgbm:free-dataset booster) nil)
                       (cl-gbdt/lightgbm:wrong-backend-reference () t))
                     "free-dataset accepted a booster")
                 ;; And it refused before doing anything at all. `release-handle' marks a
                 ;; handle released whichever way the free itself goes, so a kind check placed
                 ;; after it would leave this booster unusable AND unfreeable while still
                 ;; signalling -- an assertion on the condition alone cannot tell the two
                 ;; orders apart. The cleanup form below is what then frees it for real.
                 (ok (not (cl-gbdt/lightgbm:handle-released-p booster))
                     "free-dataset released the booster before refusing it")
                 (ok (handler-case (progn (cl-gbdt/lightgbm:free-booster data) nil)
                       (cl-gbdt/lightgbm:wrong-backend-reference () t))
                     "free-booster accepted a dataset")
                 ;; The same order, pinned for the other free too. Nothing below covers it:
                 ;; `predict' and `update-one-iteration' check the kind BEFORE they check
                 ;; liveness, so a DATA wrongly marked released here would still answer
                 ;; `wrong-backend-reference' and leave both of them green.
                 (ok (not (cl-gbdt/lightgbm:handle-released-p data))
                     "free-booster released the dataset before refusing it")
                 (ok (handler-case (progn (cl-gbdt/lightgbm:predict data matrix) nil)
                       (cl-gbdt/lightgbm:wrong-backend-reference () t))
                     "predict accepted a dataset as its booster")
                 ;; `update-one-iteration' is the one of the four whose check had to move
                 ;; rather than merely appear: `%check-booster-datasets-live' used to run
                 ;; first, and it reads `booster-training-set' off whatever it is handed, so a
                 ;; DATASET reached it and failed with a bare CLOS no-applicable-method error
                 ;; instead of this typed condition. This assertion is what pins the order.
                 (ok (handler-case (progn (cl-gbdt/lightgbm:update-one-iteration data) nil)
                       (cl-gbdt/lightgbm:wrong-backend-reference () t))
                     "update-one-iteration accepted a dataset as its booster"))
            (progn
              (cl-gbdt/lightgbm:free-booster booster)
              (cl-gbdt/lightgbm:free-dataset data))))))))

(deftest layer-1-alone-saves-loads-and-renders-a-model
  (testing "a caller with only cl-gbdt/lightgbm loaded can persist a model and read it back"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let ((data (cl-gbdt/lightgbm:create-dataset backend matrix :label label-vector
                                                      :parameters *parameters*))
              (path (model-path "cl-gbdt-lightgbm-standalone.txt"))
              (echoed (model-path "cl-gbdt-lightgbm-standalone-echo.txt")))
          (unwind-protect
               (let ((booster (cl-gbdt/lightgbm:create-booster backend data
                                                                :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 20)
                          (cl-gbdt/lightgbm:update-one-iteration booster))
                        (ok (equal path (cl-gbdt/lightgbm:save-model booster path))
                            "save-model returns the path it was given")
                        (ok (probe-file path) "save-model wrote the file")
                        ;; The round trip, and the only assertion here that needs the file to
                        ;; hold a real model rather than merely to exist: a reloaded booster
                        ;; predicts what the original predicted, to the bit.
                        (let ((reloaded (cl-gbdt/lightgbm:load-model backend path)))
                          (unwind-protect
                               (progn
                                 (ok (null (cl-gbdt/lightgbm:booster-training-set reloaded))
                                     "a loaded booster has no training set")
                                 (ok (equalp (cl-gbdt/lightgbm:predict booster matrix)
                                             (cl-gbdt/lightgbm:predict reloaded matrix))
                                     "the reloaded model predicts what the original did"))
                            (cl-gbdt/lightgbm:free-booster reloaded)))
                        ;; `model-to-string' is asserted the same way rather than against any
                        ;; substring of LightGBM's own model format: write what it returned to
                        ;; a file, load THAT, and require the same predictions. Nothing here
                        ;; then depends on how upstream words its header.
                        (let ((text (cl-gbdt/lightgbm:model-to-string booster)))
                          (ok (and (stringp text) (plusp (length text)))
                              "model-to-string returns a non-empty string")
                          (with-open-file (stream echoed :direction :output
                                                          :if-exists :supersede)
                            (write-string text stream))
                          (let ((from-string (cl-gbdt/lightgbm:load-model backend echoed)))
                            (unwind-protect
                                 (ok (equalp (cl-gbdt/lightgbm:predict booster matrix)
                                             (cl-gbdt/lightgbm:predict from-string matrix))
                                     "model-to-string's text is itself a loadable model")
                              (cl-gbdt/lightgbm:free-booster from-string))))
                        ;; :NUM-ITERATION reaches the library rather than being ignored. A
                        ;; one-tree save cannot predict what a twenty-round booster does, so
                        ;; this fails if the argument is dropped on the way down.
                        (cl-gbdt/lightgbm:save-model booster path :num-iteration 1)
                        (let ((truncated (cl-gbdt/lightgbm:load-model backend path)))
                          (unwind-protect
                               (ok (not (equalp (cl-gbdt/lightgbm:predict booster matrix)
                                                (cl-gbdt/lightgbm:predict truncated matrix)))
                                   "save-model's :num-iteration limits what is written")
                            (cl-gbdt/lightgbm:free-booster truncated)))
                        ;; `model-to-string' gets the identical proof, closed the same way:
                        ;; render at one round, write that text to a file, load it back, and
                        ;; require the predictions to differ from the full booster's. Without
                        ;; this, a :NUM-ITERATION silently dropped between this function and
                        ;; the library would pass -- neither the non-empty-string check above
                        ;; nor the :best refusal below would catch a truncated render standing
                        ;; in for a full one.
                        (let ((rendered (cl-gbdt/lightgbm:model-to-string
                                         booster :num-iteration 1)))
                          (with-open-file (stream echoed :direction :output
                                                          :if-exists :supersede)
                            (write-string rendered stream))
                          (let ((truncated (cl-gbdt/lightgbm:load-model backend echoed)))
                            (unwind-protect
                                 (ok (not (equalp (cl-gbdt/lightgbm:predict booster matrix)
                                                  (cl-gbdt/lightgbm:predict truncated matrix)))
                                     "model-to-string's :num-iteration limits what is written")
                              (cl-gbdt/lightgbm:free-booster truncated))))
                        ;; `handler-case', not rove's `signals', which does not reliably catch a
                        ;; condition raised inside `restart-case'. On the condition TYPE: the
                        ;; report's wording is not what a caller dispatches on.
                        (ok (handler-case
                                (progn (cl-gbdt/lightgbm:save-model booster path
                                                                     :num-iteration :best)
                                       nil)
                              (cl-gbdt/lightgbm:unsupported-argument () t))
                            "save-model refuses :best, which only train can resolve")
                        (ok (handler-case
                                (progn (cl-gbdt/lightgbm:model-to-string booster
                                                                          :num-iteration :best)
                                       nil)
                              (cl-gbdt/lightgbm:unsupported-argument () t))
                            "model-to-string refuses :best for the same reason")
                        ;; The specializer each of these lost.
                        (ok (handler-case
                                (progn (cl-gbdt/lightgbm:save-model data path) nil)
                              (cl-gbdt/lightgbm:wrong-backend-reference () t))
                            "save-model accepted a dataset as its booster")
                        (ok (handler-case
                                (progn (cl-gbdt/lightgbm:model-to-string data) nil)
                              (cl-gbdt/lightgbm:wrong-backend-reference () t))
                            "model-to-string accepted a dataset as its booster")
                        (ok (handler-case
                                (progn (cl-gbdt/lightgbm:load-model data path) nil)
                              (cl-gbdt/lightgbm:wrong-backend-reference () t))
                            "load-model accepted a dataset as its backend")
                        (ok (handler-case
                                (progn (cl-gbdt/lightgbm:load-model nil path) nil)
                              (cl-gbdt/lightgbm:wrong-backend-reference () t))
                            "load-model accepted NIL as its backend"))
                   (cl-gbdt/lightgbm:free-booster booster)))
            (progn
              (cl-gbdt/lightgbm:free-dataset data)
              (when (probe-file path) (delete-file path))
              (when (probe-file echoed) (delete-file echoed)))))))))

(deftest layer-1-alone-reports-importance-evaluation-and-shape
  (testing "a caller with only cl-gbdt/lightgbm loaded can ask what it just trained"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let ((data (cl-gbdt/lightgbm:create-dataset backend matrix :label label-vector
                                                      :parameters *parameters*)))
          (unwind-protect
               (let ((booster (cl-gbdt/lightgbm:create-booster backend data
                                                                :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 20)
                          (cl-gbdt/lightgbm:update-one-iteration booster))
                        ;; The dataset's own shape, read from the library rather than from the
                        ;; array it was built from.
                        (ok (= 16 (cl-gbdt/lightgbm:dataset-num-rows data))
                            "dataset-num-rows reports the fixture's row count")
                        (ok (= 2 (cl-gbdt/lightgbm:dataset-num-features data))
                            "dataset-num-features reports its column count")
                        ;; One entry per FEATURE, not per feature that happened to be split
                        ;; on. That is the property a dense result has and a sparse one does
                        ;; not, and it is what makes the two backends' results comparable.
                        (let ((split (cl-gbdt/lightgbm:feature-importance booster))
                              (gain (cl-gbdt/lightgbm:feature-importance booster :kind :gain)))
                          (ok (= 2 (length split))
                              (format nil ":split importance has ~D entries" (length split)))
                          (ok (= 2 (length gain))
                              (format nil ":gain importance has ~D entries" (length gain)))
                          (ok (every (lambda (value) (typep value 'double-float)) split)
                              "every :split entry is a double-float")
                          ;; Column 0 carries the whole signal and column 1 none, so the
                          ;; ordering holds however the library breaks ties. Equality is not
                          ;; asserted: a tie is a legitimate outcome for a boosted model that
                          ;; ran out of useful splits.
                          (ok (>= (aref split 0) (aref split 1))
                              (format nil ":split importance ~S ranks column 0 first" split))
                          (ok (>= (aref gain 0) (aref gain 1))
                              (format nil ":gain importance ~S ranks column 0 first" gain))
                          (ok (plusp (aref gain 0))
                              "the signal-carrying column has non-zero gain, so the model split"))
                        ;; The metric the objective configured, at the index the portable
                        ;; contract numbers the training set with.
                        (multiple-value-bind (entries provenance)
                            (cl-gbdt/lightgbm:evaluation booster)
                          (ok (consp entries) "evaluation reports the training set's metrics")
                          (ok (every (lambda (entry)
                                       (and (= 3 (length entry))
                                            (integerp (first entry))
                                            (stringp (second entry))
                                            (typep (third entry) 'double-float)))
                                     entries)
                              "each entry is (dataset-index metric-name value)")
                          (ok (every (lambda (entry) (zerop (first entry))) entries)
                              "with no validation set every entry is at index 0")
                          (ok (eq :library-doubles (getf provenance :value-source))
                              "and evaluation states where its values came from"))
                        ;; The specializer each of these lost.
                        (ok (handler-case
                                (progn (cl-gbdt/lightgbm:feature-importance data) nil)
                              (cl-gbdt/lightgbm:wrong-backend-reference () t))
                            "feature-importance accepted a dataset as its booster")
                        (ok (handler-case (progn (cl-gbdt/lightgbm:evaluation data) nil)
                              (cl-gbdt/lightgbm:wrong-backend-reference () t))
                            "evaluation accepted a dataset as its booster")
                        (ok (handler-case
                                (progn (cl-gbdt/lightgbm:dataset-num-rows booster) nil)
                              (cl-gbdt/lightgbm:wrong-backend-reference () t))
                            "dataset-num-rows accepted a booster as its dataset")
                        (ok (handler-case
                                (progn (cl-gbdt/lightgbm:dataset-num-features booster) nil)
                              (cl-gbdt/lightgbm:wrong-backend-reference () t))
                            "dataset-num-features accepted a booster as its dataset")
                        (ok (handler-case
                                (progn (cl-gbdt/lightgbm:feature-importance
                                        booster :num-iteration :best)
                                       nil)
                              (cl-gbdt/lightgbm:unsupported-argument () t))
                            "feature-importance refuses :best, which only train can resolve"))
                   (cl-gbdt/lightgbm:free-booster booster)))
            (cl-gbdt/lightgbm:free-dataset data)))))))

;;; A booster from `load-model' retains no dataset at all, which is the one case `evaluation'
;;; returns nothing for. It is asserted separately rather than folded into the test above
;;; because it needs a second booster and a file, and because it is the case a reader is most
;;; likely to believe is an error rather than a result.

(deftest layer-1-alone-evaluates-a-loaded-model-as-empty
  (testing "a booster with no retained dataset has nothing to evaluate"
    (with-open-backend (backend)
      (multiple-value-bind (matrix label-vector) (fixture)
        (let ((data (cl-gbdt/lightgbm:create-dataset backend matrix :label label-vector
                                                      :parameters *parameters*))
              (path (model-path "cl-gbdt-lightgbm-standalone-eval.txt")))
          (unwind-protect
               (let ((booster (cl-gbdt/lightgbm:create-booster backend data
                                                                :parameters *parameters*)))
                 (unwind-protect
                      (progn
                        (dotimes (round 5)
                          (cl-gbdt/lightgbm:update-one-iteration booster))
                        (cl-gbdt/lightgbm:save-model booster path)
                        (let ((reloaded (cl-gbdt/lightgbm:load-model backend path)))
                          (unwind-protect
                               (progn
                                 (ok (null (cl-gbdt/lightgbm:evaluation reloaded))
                                     "a loaded booster evaluates to no entries")
                                 (ok (= 2 (length (cl-gbdt/lightgbm:feature-importance
                                                   reloaded)))
                                     "but still reports one importance per feature"))
                            (cl-gbdt/lightgbm:free-booster reloaded))))
                   (cl-gbdt/lightgbm:free-booster booster)))
            (progn
              (cl-gbdt/lightgbm:free-dataset data)
              (when (probe-file path) (delete-file path)))))))))

(deftest create-dataset-from-file-reads-a-libsvm-file
  (testing "create-dataset-from-file builds a dataset LightGBM reports the right shape for"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        (let ((dataset (cl-gbdt/lightgbm:create-dataset-from-file backend path)))
          (unwind-protect
               (progn
                 (ok (= 4 (cl-gbdt/lightgbm:dataset-num-rows dataset))
                     (format nil "dataset-num-rows is ~D"
                             (cl-gbdt/lightgbm:dataset-num-rows dataset)))
                 (ok (= 3 (cl-gbdt/lightgbm:dataset-num-features dataset))
                     (format nil "dataset-num-features is ~D"
                             (cl-gbdt/lightgbm:dataset-num-features dataset))))
            (cl-gbdt/lightgbm:free-dataset dataset)))))))

(deftest create-dataset-from-file-trains-the-same-model-as-the-matrix
  (testing "a dataset read from a file trains identically to the same data given as a matrix"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        (let ((from-file (cl-gbdt/lightgbm:create-dataset-from-file
                          backend path :parameters *parameters*))
              (from-matrix (cl-gbdt/lightgbm:create-dataset
                            backend
                            (make-array '(4 3) :element-type 'double-float
                                               :initial-contents
                                               '((1d0 2d0 3d0) (4d0 5d0 6d0)
                                                 (7d0 8d0 9d0) (10d0 11d0 12d0)))
                            :label '(1d0 0d0 1d0 0d0)
                            :parameters *parameters*)))
          (unwind-protect
               ;; `*parameters*' on every call above, not just this comparison: with none,
               ;; LightGBM's default `min_data_in_leaf' (20) exceeds these four rows and
               ;; neither booster ever splits, so the strings compared below would encode
               ;; only the label mean and nothing about which feature values were actually
               ;; read -- see `model-string-after-one-iteration''s docstring, and this
               ;; project's review of this file's first version, which measured that an
               ;; implementation reading wholly different feature values still passed this
               ;; assertion under the weaker (no-`:parameters') form.
               (ok (string= (model-string-after-one-iteration backend from-file
                                                               :parameters *parameters*)
                            (model-string-after-one-iteration backend from-matrix
                                                               :parameters *parameters*))
                   "a file-built dataset and a matrix-built dataset trained different models")
            (progn
              (cl-gbdt/lightgbm:free-dataset from-file)
              (cl-gbdt/lightgbm:free-dataset from-matrix))))))))

;;; The control. Without this, a `path' that silently produced an empty or degenerate dataset
;;; would match nothing and pass the comparison above for a reason that has nothing to do with
;;; whether `create-dataset-from-file' actually read the file's rows.

(deftest create-dataset-from-file-does-not-match-a-different-file
  (testing "a genuinely different file does not train the same model"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        (with-different-libsvm-fixture (other)
          (let ((a (cl-gbdt/lightgbm:create-dataset-from-file
                    backend path :parameters *parameters*))
                (b (cl-gbdt/lightgbm:create-dataset-from-file
                    backend other :parameters *parameters*)))
            (unwind-protect
                 (ng (string= (model-string-after-one-iteration backend a
                                                                 :parameters *parameters*)
                              (model-string-after-one-iteration backend b
                                                                 :parameters *parameters*))
                     "two different files trained the same model")
              (progn
                (cl-gbdt/lightgbm:free-dataset a)
                (cl-gbdt/lightgbm:free-dataset b)))))))))

(deftest create-dataset-from-file-refuses-bad-arguments
  (testing "create-dataset-from-file checks its backend's class first, and reports a missing \
file as LightGBM's own"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        ;; `handler-case', not rove's `signals' -- see this file's other guard tests for why.
        (ok (handler-case (progn (cl-gbdt/lightgbm:create-dataset-from-file nil path) nil)
              (cl-gbdt/lightgbm:wrong-backend-reference () t))
            "create-dataset-from-file accepted NIL as its backend")
        ;; NIL alone would be caught by almost any accidental check. This file's own
        ;; convention for the identical guard (`layer-1-alone-saves-loads-and-renders-a-model')
        ;; also covers a wrong-CLASS object, which is what the two measurements behind
        ;; `%check-object-class' actually were.
        (let ((wrong-class (cl-gbdt/lightgbm:create-dataset-from-file backend path)))
          (unwind-protect
               (ok (handler-case
                       (progn (cl-gbdt/lightgbm:create-dataset-from-file wrong-class path) nil)
                     (cl-gbdt/lightgbm:wrong-backend-reference () t))
                   "create-dataset-from-file accepted a dataset as its backend")
            (cl-gbdt/lightgbm:free-dataset wrong-class)))
        ;; Not a `probe-file' pre-check -- a missing file is LightGBM's own to report, through
        ;; `check-lgbm' like any other failed foreign call.
        (ok (handler-case
                (progn (cl-gbdt/lightgbm:create-dataset-from-file
                        backend (merge-pathnames "cl-gbdt-no-such-file.libsvm"
                                                  (uiop:temporary-directory)))
                       nil)
              (cl-gbdt/lightgbm:foreign-call-error () t))
            "create-dataset-from-file did not signal foreign-call-error for a missing file")))))

(deftest create-dataset-from-file-signals-backend-not-open-after-close
  (testing "create-dataset-from-file signals backend-not-open for a closed backend"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        (cl-gbdt/lightgbm:close-backend backend)
        (ok (handler-case (progn (cl-gbdt/lightgbm:create-dataset-from-file backend path) nil)
              (cl-gbdt/lightgbm:backend-not-open () t))
            "create-dataset-from-file did not signal backend-not-open")))))

;;; The five tests above never supply `:parameters' or `:reference' -- an implementation that
;;; dropped `parameter-string' or `reference-pointer' at the call site, passing `""' and a
;;; null pointer unconditionally, would pass every one of them. This project already pins
;;; exactly that class of thing elsewhere in this file: `save-model''s `:num-iteration' test
;;; exists, in its own words, because a value silently dropped between a wrapper and the
;;; library would otherwise pass.
;;;
;;; `create-dataset-from-file-honours-parameters' closes the gap for `:parameters' genuinely:
;;; `:header "true"' on `with-libsvm-fixture''s file reads 3 rows rather than 4 -- measured
;;; against the vendored library (`repl-eval', 2026-08-13) -- which a dropped parameter
;;; string could not produce.
;;;
;;; `create-dataset-from-file-honours-reference' needed a second attempt to do the same for
;;; `:reference'. Its first version built the reference from `with-libsvm-fixture''s own
;;; three-feature shape and asserted 4 rows x 3 features -- exactly what
;;; `create-dataset-from-file' already returns with NO reference at all (measured), so a
;;; null reference pointer passed that assertion too, and the comment then in this spot
;;; claimed coverage the test did not have. The fix is a reference built from a WIDENED,
;;; five-column matrix: LightGBM's bin mapper, not the file, decides the resulting feature
;;; count, so a dataset built against it reads back as 4 rows x 5 features (measured) -- a
;;; shape a dropped reference pointer cannot produce, since without one the file's own three
;;; columns decide it instead.

(deftest create-dataset-from-file-honours-parameters
  (testing "create-dataset-from-file's :parameters reaches LGBM_DatasetCreateFromFile"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        ;; `header=true' consumes the fixture's first data row as a header instead -- see
        ;; `docs/superpowers/specs/2026-08-13-file-input-measurements.md' section 5 -- so an
        ;; implementation that rendered PARAMETERS into `""' regardless of what it was given
        ;; would still read 4 rows here and this assertion would catch it.
        (let ((dataset (cl-gbdt/lightgbm:create-dataset-from-file
                        backend path :parameters '(:header "true"))))
          (unwind-protect
               (ok (= 3 (cl-gbdt/lightgbm:dataset-num-rows dataset))
                   (format nil ":parameters '(:header \"true\") left dataset-num-rows at ~D, \
not 3" (cl-gbdt/lightgbm:dataset-num-rows dataset)))
            (cl-gbdt/lightgbm:free-dataset dataset)))))))

(deftest create-dataset-from-file-honours-reference
  (testing "create-dataset-from-file's :reference reaches LGBM_DatasetCreateFromFile, and is \
checked before any foreign call"
    (with-open-backend (backend)
      (with-libsvm-fixture (path)
        ;; A reference built from the FILE's own three-feature shape would not discriminate:
        ;; `create-dataset-from-file' with no reference at all already returns 4 rows x 3
        ;; features (measured), so asserting that shape passes whether or not
        ;; REFERENCE-POINTER ever reaches `LGBM_DatasetCreateFromFile'. A five-column
        ;; reference does discriminate -- LightGBM's bin mapper, not the file, then decides
        ;; the feature count -- which is why this reference is deliberately wider than PATH.
        (let ((wide-reference (cl-gbdt/lightgbm:create-dataset
                               backend
                               (make-array '(4 5) :element-type 'double-float
                                                  :initial-contents
                                                  '((1d0 2d0 3d0 4d0 5d0)
                                                    (6d0 7d0 8d0 9d0 10d0)
                                                    (11d0 12d0 13d0 14d0 15d0)
                                                    (16d0 17d0 18d0 19d0 20d0))))))
          (unwind-protect
               (let ((second (cl-gbdt/lightgbm:create-dataset-from-file
                              backend path :reference wide-reference)))
                 (unwind-protect
                      (progn
                        (ok (= 4 (cl-gbdt/lightgbm:dataset-num-rows second))
                            "a dataset built with :reference has the wrong row count")
                        (ok (= 5 (cl-gbdt/lightgbm:dataset-num-features second))
                            (format nil ":reference did not reach the foreign call -- \
dataset-num-features is ~D, not the reference's own 5"
                                    (cl-gbdt/lightgbm:dataset-num-features second))))
                   (cl-gbdt/lightgbm:free-dataset second)))
            (cl-gbdt/lightgbm:free-dataset wide-reference))
          ;; A wrong-class :reference is refused before any foreign call -- `%reference-
          ;; pointer' delegates to `%check-lightgbm-dataset', which `create-dataset' already
          ;; relies on for the identical check.
          (ok (handler-case
                  (progn (cl-gbdt/lightgbm:create-dataset-from-file backend path :reference 42)
                         nil)
                (cl-gbdt/lightgbm:wrong-backend-reference () t))
              "create-dataset-from-file accepted a non-dataset :reference")
          ;; And a freed one -- the case the docstring promises `released-handle-error' for.
          (let ((freed (cl-gbdt/lightgbm:create-dataset-from-file backend path)))
            (cl-gbdt/lightgbm:free-dataset freed)
            (ok (handler-case
                    (progn (cl-gbdt/lightgbm:create-dataset-from-file
                            backend path :reference freed)
                           nil)
                  (cl-gbdt/lightgbm:released-handle-error () t))
                "create-dataset-from-file accepted a freed :reference")))))))

;;; Review round 4's Finding N9 moved XGBoost's `file-uri' from `namestring' to
;;; `sb-ext:native-namestring' because `namestring' backslash-escapes a literal asterisk in a
;;; real filename; the whole-branch review found `create-dataset-from-file' still on
;;; `namestring', two rounds later and never revisited. The two tests below pin the fix on
;;; both of its edges: a literal asterisk in a filename now opens, and a genuinely wild
;;; pathname -- which `native-namestring' itself would refuse with an untyped error -- is
;;; refused first, with `unsupported-argument', by `%check-file-path'.

(deftest create-dataset-from-file-reads-a-file-with-a-literal-asterisk
  (testing "a filename whose namestring backslash-escapes a literal asterisk still opens"
    (with-open-backend (backend)
      ;; `sb-ext:parse-native-namestring', not the ordinary pathname reader: the asterisk
      ;; this builds is a literal character of the filename, not a CL wildcard marker --
      ;; `wild-pathname-p' is NIL for the result -- and `namestring' backslash-escapes it
      ;; regardless (measured: `.../cl-gbdt-star\*....libsvm'), which is exactly what made
      ;; LightGBM report "Cannot open data file" for a file that exists before this fix.
      (let ((path (merge-pathnames
                   (sb-ext:parse-native-namestring
                    (format nil "cl-gbdt-star*fixture-~D.libsvm" (random 1000000)))
                   (uiop:temporary-directory))))
        (with-open-file (stream path :direction :output :if-exists :supersede)
          (write-string "1 0:1.0 1:2.0 2:3.0
0 0:4.0 1:5.0 2:6.0
1 0:7.0 1:8.0 2:9.0
0 0:10.0 1:11.0 2:12.0
" stream))
        (unwind-protect
             (let ((dataset (cl-gbdt/lightgbm:create-dataset-from-file backend path)))
               (unwind-protect
                    (progn
                      (ok (= 4 (cl-gbdt/lightgbm:dataset-num-rows dataset))
                          "a literal-asterisk filename read the wrong row count")
                      (ok (= 3 (cl-gbdt/lightgbm:dataset-num-features dataset))
                          "a literal-asterisk filename read the wrong feature count"))
                 (cl-gbdt/lightgbm:free-dataset dataset)))
          (handler-case (delete-file path) (file-error () nil)))))))

(deftest create-dataset-from-file-refuses-a-wild-path
  (testing "create-dataset-from-file refuses a wild pathname before any foreign call"
    (with-open-backend (backend)
      ;; The ordinary pathname reader, unlike `sb-ext:parse-native-namestring' above: this
      ;; asterisk IS a CL wildcard marker, `wild-pathname-p' is true of the result, and
      ;; `sb-ext:native-namestring' would signal its own untyped
      ;; `sb-kernel:no-native-namestring-error' for it -- `%check-file-path' exists to
      ;; refuse this case first, with a typed condition this project's own callers can catch.
      (let ((path (merge-pathnames (pathname "cl-gbdt-wild*.libsvm")
                                    (uiop:temporary-directory))))
        (ok (handler-case (progn (cl-gbdt/lightgbm:create-dataset-from-file backend path) nil)
              (cl-gbdt/lightgbm:unsupported-argument () t))
            "create-dataset-from-file accepted a wild pathname")))))

(deftest create-dataset-from-file-resolves-a-relative-path-against-default-pathname-defaults
  (testing "a relative PATH is resolved the way `open' resolves one, not native-namestring'd \
bare -- proven by a same-named decoy file sitting in the process's own working directory, \
which the unresolved form would have opened instead"
    (with-open-backend (backend)
      (let* ((relative-name (format nil "cl-gbdt-p1-~D.libsvm" (random 1000000)))
             (real-dir (ensure-directories-exist
                        (merge-pathnames (format nil "cl-gbdt-p1-real-~D/" (random 1000000))
                                         (uiop:temporary-directory))))
             (real-path (merge-pathnames relative-name real-dir))
             (decoy-path (merge-pathnames relative-name (uiop:getcwd))))
        (unwind-protect
             (progn
               ;; The REAL file: four rows, three features, this file's usual shape.
               (with-open-file (stream real-path :direction :output :if-exists :supersede)
                 (write-string "1 0:1.0 1:2.0 2:3.0
0 0:4.0 1:5.0 2:6.0
1 0:7.0 1:8.0 2:9.0
0 0:10.0 1:11.0 2:12.0
" stream))
               ;; The DECOY, same relative name, sitting in the process's actual working
               ;; directory -- deliberately a different ROW count (two, not four) but the
               ;; SAME three-feature shape as the real file. PR #36 review, Minor M2: a
               ;; 2x1 decoy (an earlier version of this fixture) is not valid libsvm to
               ;; LightGBM at all -- "Check failed: (max_col_idx) > (0)" -- so under the
               ;; pre-fix behaviour this test errored out rather than failing the row/
               ;; feature-count assertions its own `ok' messages describe; still a valid
               ;; sentinel (it did fail before the fix and pass after) but not the failure
               ;; mode claimed. A 2x3 decoy is valid libsvm LightGBM reads cleanly, so the
               ;; pre-fix run reads it successfully and fails literally on
               ;; dataset-num-rows (2, not 4), the assertion actually written below.
               (with-open-file (stream decoy-path :direction :output :if-exists :supersede)
                 (write-string "1 0:99.0 1:98.0 2:97.0
0 0:96.0 1:95.0 2:94.0
" stream))
               ;; A single-binding `let' nested inside another, not `let*': the dataset's
               ;; own init-form must run AFTER *default-pathname-defaults* is rebound, a
               ;; dynamic dependency an ordinary `let*' linter check cannot see across a
               ;; special-variable rebinding, and an ordinary `let' would evaluate BOTH
               ;; init-forms before establishing either binding, running
               ;; create-dataset-from-file against the OLD *default-pathname-defaults*
               ;; and defeating the whole point of this test.
               (let ((*default-pathname-defaults* real-dir))
                 (let ((dataset (cl-gbdt/lightgbm:create-dataset-from-file
                                 backend relative-name)))
                   (unwind-protect
                        (progn
                          (ok (= 4 (cl-gbdt/lightgbm:dataset-num-rows dataset))
                              (format nil "dataset-num-rows is ~D, not the real file's 4 \
-- read the decoy in the process's own working directory instead"
                                      (cl-gbdt/lightgbm:dataset-num-rows dataset)))
                          (ok (= 3 (cl-gbdt/lightgbm:dataset-num-features dataset))
                              (format nil "dataset-num-features is ~D, not the real \
file's 3 -- read the decoy in the process's own working directory instead"
                                      (cl-gbdt/lightgbm:dataset-num-features dataset))))
                     (cl-gbdt/lightgbm:free-dataset dataset)))))
          (handler-case (delete-file real-path) (file-error () nil))
          (handler-case (uiop:delete-directory-tree real-dir :validate t)
            (file-error () nil))
          (handler-case (delete-file decoy-path) (file-error () nil)))))))

(deftest create-dataset-from-file-still-reports-a-missing-relative-file-as-foreign-call-error
  (testing "a missing relative PATH still reaches LGBM_DatasetCreateFromFile and comes back \
as foreign-call-error -- the P1 fix resolves PATH, it does not add a probe-file pre-check"
    (with-open-backend (backend)
      (let ((relative-name (format nil "cl-gbdt-p1-missing-~D.libsvm" (random 1000000)))
            (some-dir (ensure-directories-exist
                       (merge-pathnames
                        (format nil "cl-gbdt-p1-missing-dir-~D/" (random 1000000))
                        (uiop:temporary-directory)))))
        (unwind-protect
             (let ((*default-pathname-defaults* some-dir))
               (ok (handler-case
                       (progn (cl-gbdt/lightgbm:create-dataset-from-file
                               backend relative-name)
                              nil)
                     (cl-gbdt/lightgbm:foreign-call-error () t))
                   "create-dataset-from-file did not signal foreign-call-error for a \
missing relative file"))
          (handler-case (uiop:delete-directory-tree some-dir :validate t)
            (file-error () nil)))))))
