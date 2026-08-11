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
;;;; `lightgbm-api-create-booster-and-train-agree' -- see the commentary there. The split is
;;;; not cosmetic: `train' does not call `create-booster' (see the comment at its creation
;;;; call in src/lightgbm/protocol.lisp for the measured reason), so the two are separate
;;;; copies of one procedure and something has to hold them together.

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
                          ;; `predict''s second value, which this backend DERIVES rather than
                          ;; reads back -- see `%prediction-shape'. Layer 1 states it too, so
                          ;; a standalone caller is not left to infer it.
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
