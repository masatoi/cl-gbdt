;;;; protocol.lisp --- Generic functions of the unified API.
;;;;
;;;; Declarations only. Methods are added by the individual backend systems.
;;;;
;;;; Both backends take string key/value parameters, so a plist is accepted and
;;;; stringified. Backend-specific parameters pass through untouched; differences
;;;; between the two training algorithms are not forced into one abstraction.

(uiop:define-package #:cl-gbdt/src/protocol
  (:use #:cl)
  (:import-from #:alexandria #:parse-body)
  (:import-from #:cl-gbdt/src/backend #:backend)
  (:export #:make-dataset
           #:dataset-num-rows
           #:dataset-num-features
           #:train
           #:update-one-iteration
           #:predict
           #:save-model
           #:load-model
           #:model-to-string
           #:feature-importance
           #:free-dataset
           #:free-booster
           #:with-dataset
           #:with-booster))

(in-package #:cl-gbdt/src/protocol)

(defgeneric make-dataset (backend matrix &key label weight group feature-names parameters)
  (:documentation "Build a training dataset for BACKEND from MATRIX.

MATRIX is anything `with-foreign-matrix' accepts. LABEL is the target vector, WEIGHT
the per-sample weights, GROUP the group sizes for ranking, and FEATURE-NAMES a list
of feature name strings. PARAMETERS is a plist passed through to the backend.

Free the result with `free-dataset' or wrap it in `with-dataset'."))

(defgeneric dataset-num-rows (dataset)
  (:documentation "Return the number of rows in DATASET."))

(defgeneric dataset-num-features (dataset)
  (:documentation "Return the number of features in DATASET."))

(defgeneric train (backend dataset &key valid-sets num-rounds parameters)
  (:documentation "Train a BACKEND model on DATASET and return a booster.

VALID-SETS is a list of validation datasets, NUM-ROUNDS the number of boosting
iterations, and PARAMETERS a plist passed through to the backend.

Free the result with `free-booster' or wrap it in `with-booster'."))

(defgeneric update-one-iteration (booster)
  (:documentation "Advance BOOSTER by one boosting iteration.

Use this to drive the training loop yourself. Returns false when no further split
was possible."))

(defgeneric predict (booster matrix &key kind num-iteration)
  (:documentation "Predict on MATRIX using BOOSTER.

KIND is `:normal' (default, transformed predictions), `:raw' (raw scores),
`:leaf-index' (leaf indices) or `:contrib' (feature contributions). NUM-ITERATION
limits how many trees are used; nil uses all of them.

Returns a `(simple-array double-float (* *))'."))

(defgeneric save-model (booster path &key num-iteration)
  (:documentation "Save BOOSTER's model to PATH."))

(defgeneric load-model (backend path)
  (:documentation "Load a model from PATH and return a BACKEND booster."))

(defgeneric model-to-string (booster &key num-iteration)
  (:documentation "Return BOOSTER's model as a string."))

(defgeneric feature-importance (booster &key kind num-iteration)
  (:documentation "Return BOOSTER's feature importances as `(simple-array double-float (*))'.

KIND is `:split' (how often a feature was used to split) or `:gain' (total gain)."))

(defgeneric free-dataset (dataset)
  (:documentation "Free DATASET. Does nothing if it was already freed."))

(defgeneric free-booster (booster)
  (:documentation "Free BOOSTER. Does nothing if it was already freed."))

(defmacro with-dataset ((var form) &body body)
  "Bind VAR to the dataset FORM returns, evaluate BODY, and always free it.

Explicit resource management is the first-class pattern; finalizers are only a
safety net.

Declarations at the head of BODY are moved onto a fresh binding of VAR that
shadows the one FORM's value is stored in, scoped to BODY alone. Splicing them
onto the outer binding instead -- the one `unwind-protect''s cleanup clause also
reads to call `free-dataset' -- would put an `(ignore VAR)' declaration from
BODY in the same scope as that read, which SBCL flags as \"reading an ignored
variable\" (verified empirically; do not simplify this back to `progn' or a
single binding)."
  (multiple-value-bind (forms declarations) (alexandria:parse-body body)
    `(let ((,var ,form))
       (unwind-protect
            (let ((,var ,var))
              ,@declarations
              ,@forms)
         (free-dataset ,var)))))

(defmacro with-booster ((var form) &body body)
  "Bind VAR to the booster FORM returns, evaluate BODY, and always free it.

A booster holds a strong reference to the dataset it was trained on, so nesting this
inside `with-dataset' cannot invert the release order. Declarations at the head of
BODY are shadow-bound as in `with-dataset', for the same reason."
  (multiple-value-bind (forms declarations) (alexandria:parse-body body)
    `(let ((,var ,form))
       (unwind-protect
            (let ((,var ,var))
              ,@declarations
              ,@forms)
         (free-booster ,var)))))
