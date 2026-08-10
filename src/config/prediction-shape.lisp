;;;; prediction-shape.lisp --- Deriving a `:contrib' prediction result's shape from counts.
;;;;
;;;; Backend-neutral and pure: no handle, no pointer, no shared library. Only LightGBM calls
;;;; this -- XGBoost reads its own shape back from the library rather than deriving one.
;;;;
;;;; Under src/config/ rather than directly under src/, and so deliberately absent from
;;;; src/all.lisp's `use-reexport' list, for the same reason `cl-gbdt/src/config/missing-value'
;;;; is absent from it -- see that file's own header. Task 3 (LightGBM's `predict') is this
;;;; file's only intended caller; publishing this from `CL-GBDT' would commit to a shape before
;;;; there is a second caller to test it against.
;;;;
;;;; Consumers: `cl-gbdt/src/lightgbm/protocol' (Task 3).

(uiop:define-package #:cl-gbdt/src/config/prediction-shape
  (:use #:cl)
  (:export #:contrib-shape))

(in-package #:cl-gbdt/src/config/prediction-shape)

(defun contrib-shape (element-count num-rows num-features)
  "Return the shape of a `:contrib' prediction result as a list of integers, or NIL when it
cannot be derived from ELEMENT-COUNT, NUM-ROWS and NUM-FEATURES.

The shape is (NUM-ROWS classes width), where WIDTH is NUM-FEATURES + 1 -- one contribution per
feature plus the bias -- and CLASSES is what is left of ELEMENT-COUNT after dividing by the
other two. Measured against both vendored libraries: a 3-class model over 4 features returns
15 values per row, and a BINARY model over 3 features returns 4, reported by XGBoost as
(rows 1 4) rather than as two dimensions.

Only LightGBM calls this. XGBoost reads its own `out_shape' back from
`XGBoosterPredictFromDMatrix' and states what the library said rather than deriving anything.
The arithmetic here is derivation; the CLASS-MAJOR ORDERING it implies is a measured claim
about LightGBM's buffer, held by the SHAP-sum test in
tests/functional/prediction-shape.lisp -- contributions for one class sum to that class's raw
score, and the same sum taken feature-major does not.

NIL rather than a signal or a guess when the numbers do not divide: NIL is what the second
return value of `predict' means everywhere it appears -- this backend states no shape here --
and a derived shape that does not account for every element would be a claim about a layout
this function has just been shown not to understand."
  (unless (or (<= num-rows 0) (minusp num-features))
    (let ((width (1+ num-features)))
      (multiple-value-bind (classes remainder)
          (truncate element-count (* num-rows width))
        (when (and (plusp classes) (zerop remainder))
          (list num-rows classes width))))))
