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
                #:unsupported-argument)
  (:import-from #:cl-gbdt/src/data
                #:csr-matrix
                #:csr-matrix-num-columns)
  (:export #:categorical-feature-types
           #:categorical-feature-string))

(in-package #:cl-gbdt/src/config/categorical-features)

(defun %matrix-num-features (matrix)
  "Return MATRIX's column count, for either form `make-dataset' accepts.

A `csr-matrix' declares it; anything else is what `with-foreign-matrix' accepts, whose column
count is its second array dimension. Both renderers below take the matrix and call this rather
than being handed a count, because the two backends obtain a count at different moments --
LightGBM builds its parameter string before any dataset exists, XGBoost attaches to a finished
DMatrix -- and a range check made against two differently-obtained counts is how the same call
comes to be refused on one backend and accepted on the other.

A matrix of rank below 2 reaches `array-dimension''s own error here, slightly before
`with-foreign-matrix' would have signalled its typed one. That is reachable only by a caller
who passes both a malformed matrix and :CATEGORICAL-FEATURES, since nothing calls this
otherwise."
  (if (typep matrix 'csr-matrix)
      (csr-matrix-num-columns matrix)
      (array-dimension matrix 1)))

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

`*print-base*' is bound to 10 and `*print-radix*' to NIL for the reason
`cl-gbdt/src/config/missing-value''s `missing-value-json' docstring gives at length: a caller
whose own `*print-base*' is 16 would otherwise see column 255 rendered `\"FF\"', a
valid-looking token naming a different column.

Signals `unsupported-argument' against BACKEND-NAME for anything `%check-indices' rejects --
the range check included, even though this renderer does not need the count to do its own
job, so that a list refused for XGBoost is refused for LightGBM too."
  (when indices
    (%check-indices indices (%matrix-num-features matrix) backend-name)
    (let ((*print-base* 10)
          (*print-radix* nil))
      (format nil "~{~D~^,~}" indices))))
