;;;; categorical-features.lisp --- Rendering a categorical-column list for both backends.
;;;;
;;;; Backend-neutral and pure: no handle, no pointer, no shared library, so every path here
;;;; is layer-1 testable. This is where the feature's validation lives, and where a
;;;; caller's `*print-base*' could otherwise turn column 255 into the valid-looking `"FF"'
;;;; -- see `categorical-feature-string''s docstring for that.
;;;;
;;;; Under src/config/ rather than directly under src/, and so deliberately absent from
;;;; src/all.lisp's `use-reexport' list, for the same reason `src/config/missing-value.lisp'
;;;; is absent from it -- see that file's own header. Tasks 2 and 3 (each backend's own
;;;; `make-dataset') are this file's only intended callers; publishing this from `CL-GBDT'
;;;; would commit to a shape before there is a second caller to test it against.
;;;;
;;;; Consumers: `cl-gbdt/src/lightgbm/protocol' and `cl-gbdt/src/xgboost/protocol' (Tasks 2, 3).

(uiop:define-package #:cl-gbdt/src/config/categorical-features
  (:use #:cl)
  (:import-from #:cl-gbdt/src/conditions
                #:dimension-mismatch
                #:unsupported-argument)
  (:import-from #:cl-gbdt/src/data
                #:csr-matrix
                #:csr-matrix-num-columns
                #:foreign-matrix
                #:foreign-matrix-cols)
  (:export #:categorical-feature-types
           #:categorical-feature-string))

(in-package #:cl-gbdt/src/config/categorical-features)

(defun %matrix-num-features (matrix)
  "Return MATRIX's column count, for every form `make-dataset' accepts.

There are three of them, all published by `cl-gbdt/src/data': a `csr-matrix' declares its
count, a `foreign-matrix' carries it in the COLS slot it was built with, and a Lisp array
holds it as its second dimension. `with-foreign-matrix' accepts the last two and
`make-dataset' takes the first beside them, so handling only two here is not a narrower
contract but a `type-error': `(array-dimension <foreign-matrix> 1)' signals one, measured,
for a matrix form this wrapper documents as accepted.

Both renderers below take the matrix and call this rather than being handed a count, because
the two backends obtain a count at different moments -- LightGBM builds its parameter string
before any dataset exists, XGBoost attaches to a finished DMatrix -- and a range check made
against two differently-obtained counts is how the same call comes to be refused on one
backend and accepted on the other.

An array whose rank is not 2 signals `dimension-mismatch' with the same EXPECTED text
`cl-gbdt/src/data''s own `call-with-foreign-matrix' signals for it, rather than the untyped
`\"Vector axis is not zero: 1\"' `array-dimension' answered a rank-1 array with before this
check existed. A MATRIX that is no array at all still reaches a raw `type-error' here, now
from `array-rank'; that is reachable only by a caller who passes both a malformed matrix and
:CATEGORICAL-FEATURES, since nothing calls this otherwise."
  (typecase matrix
    (csr-matrix (csr-matrix-num-columns matrix))
    (foreign-matrix (foreign-matrix-cols matrix))
    (t (unless (= 2 (array-rank matrix))
         (error 'dimension-mismatch :expected "a 2D array" :given (array-dimensions matrix)))
       (array-dimension matrix 1))))

(defun %check-indices (indices num-features backend-name)
  "Signal `unsupported-argument' against BACKEND-NAME unless INDICES is a list of distinct
non-negative integers, each below NUM-FEATURES.

Both renderers call this with the same count, from `%matrix-num-features', so neither can
accept a list the other refuses.

A duplicate is rejected rather than collapsed: LightGBM would be handed \"1,1\" and XGBoost a
vector that cannot record it was said twice, so silently accepting one would make the same
call mean different things on the two backends."
  (unless (listp indices)
    (error 'unsupported-argument
           :backend backend-name
           :argument ":categorical-features"
           :reason (format nil "the categorical column list must be a list of ~
                                column indices")))
  (dolist (index indices)
    (unless (and (integerp index) (not (minusp index)))
      (error 'unsupported-argument
             :backend backend-name
             :argument ":categorical-features"
             :reason (format nil "each categorical column must be a non-negative ~
                                  integer index")))
    (when (>= index num-features)
      (error 'unsupported-argument
             :backend backend-name
             :argument ":categorical-features"
             :reason (format nil "categorical column index ~D is beyond the matrix's ~
                                  ~D column~:P"
                             index num-features))))
  (when (/= (length indices) (length (remove-duplicates indices)))
    (error 'unsupported-argument
           :backend backend-name
           :argument ":categorical-features"
           :reason (format nil "the same categorical column was named more than once")))
  indices)

(defun categorical-feature-types (indices matrix backend-name)
  "Return the per-column feature-type strings XGBoost's `\"feature_type\"' field takes for
INDICES over MATRIX, or NIL when INDICES is NIL.

Each listed column renders as `\"c\"' and every other as `\"q\"' -- the encoding the vendored
header states directly (`ffi-spec/xgboost/include/xgboost/c_api.h': `i' for integer, `q' for
quantitive, `c' for categorical). The `\"i\"', `\"int\"' and `\"float\"' types it also accepts
are deliberately not exposed; only the categorical-versus-quantitative distinction is.

NIL returns NIL rather than a vector of `\"q\"', because those are not the same request: a
full type vector where the wrapper previously set none is a change to what every existing
caller sends, and `make-dataset' attaches nothing at all when this returns NIL.

Signals `unsupported-argument' against BACKEND-NAME for anything `%check-indices' rejects."
  (when indices
    (let ((num-features (%matrix-num-features matrix)))
      (%check-indices indices num-features backend-name)
      (loop :for column :below num-features
            :collect (if (member column indices) "c" "q")))))

(defun categorical-feature-string (indices matrix backend-name)
  "Return the comma-joined decimal column list LightGBM's `categorical_feature' parameter
takes for INDICES, or NIL when INDICES is NIL.

The caller's order is preserved: LightGBM reads the value as a set, so reordering it would be
a difference the caller cannot observe and this function has no reason to introduce.

A caller's own printer bindings cannot reach the result, and the `~D' directive is what keeps
them out: `~D' binds `*print-base*' to 10 and `*print-radix*' to false itself (CLHS 22.3.2.2,
Tilde D: Decimal). Measured under `*print-base*' 16: `\"~{~D~^,~}\"' renders column 255 as
`\"255\"', where the same list through `~A' -- or through `princ-to-string' -- renders
`\"FF\"', a valid-looking token naming a different column. That is the risk
`cl-gbdt/src/config/missing-value''s `missing-value-json' docstring gives at length; its own
`let' of the two specials is load-bearing because it prints through `princ-to-string', and one
here would not be, so there is none.
`categorical-feature-string-ignores-the-caller-s-print-base' in tests/categorical-features.lisp
pins the directive in its place.

Signals `unsupported-argument' against BACKEND-NAME for anything `%check-indices' rejects --
the range check included, even though this renderer does not need the count to do its own
job, so that a list refused for XGBoost is refused for LightGBM too."
  (when indices
    (%check-indices indices (%matrix-num-features matrix) backend-name)
    (format nil "~{~D~^,~}" indices)))
