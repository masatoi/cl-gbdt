;;;; missing-value.lisp --- Rendering a missing-value sentinel as a JSON number token.
;;;;
;;;; Backend-neutral and pure: no handle, no pointer, no shared library. A sentinel that
;;;; reaches XGBoost as "1.0d-5" is a hard JSON parse error, and one that reaches it as
;;;; "FF" under a caller's *print-base* is a silently wrong number -- see
;;;; `missing-value-json''s docstring for both.
;;;;
;;;; Under src/config/ rather than directly under src/, and so deliberately absent from
;;;; src/all.lisp's `use-reexport' list, for the same reason `src/training/early-stopping.lisp'
;;;; is absent from it -- see that file's own header. Tasks 2 and 3 (each backend's own
;;;; `train'/`make-dataset'/`predict') are this file's only intended callers; publishing this
;;;; from `CL-GBDT' would commit to a shape before there is a second caller to test it against.
;;;;
;;;; Consumers: `cl-gbdt/src/lightgbm/protocol' and `cl-gbdt/src/xgboost/protocol' (Tasks 2, 3).

(uiop:define-package #:cl-gbdt/src/config/missing-value
  (:use #:cl)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-argument)
  (:export #:missing-value-json))

(in-package #:cl-gbdt/src/config/missing-value)

(defun missing-value-json (value backend-name)
  "Return VALUE as the JSON number token a config JSON's missing-value sentinel takes,
signalling `unsupported-argument' against BACKEND-NAME when VALUE is neither a `real'
nor NIL.

The result is a JSON *number*, never quoted. NIL renders as `\"NaN\"' because that is the
sentinel this wrapper has always sent -- an existing caller who never passes :MISSING must
keep getting the numbers they get today (policy section 14).

Printer specials are bound the way `cl-gbdt/src/parameters''s `parameter-value' binds them
and for the same reasons, which that docstring gives: `*print-base*' 10 so a caller's own
binding of 16 cannot turn 255 into the valid-looking `\"FF\"', and
`*read-default-float-format*' set to the value's own type so no float carries a `\"d0\"' or
`\"f0\"' marker. That last one is not cosmetic here: XGBoost's config parser rejects
`\"1.0d-5\"' outright (`json.cc:409: Expecting: \",\"'), measured against the vendored
library.

NaN and the infinities are tested for before the general float branch, since `floatp'
answers T for all three, and each has a bare JSON token XGBoost accepts. They are tested
with `sb-ext:float-nan-p' and `sb-ext:float-infinity-p' rather than by printing, because
SBCL princs them as `\"#<DOUBLE-FLOAT quiet NaN>\"' and
`\"#.SB-EXT:DOUBLE-FLOAT-POSITIVE-INFINITY\"'."
  (let ((*print-base* 10)
        (*print-radix* nil))
    (typecase value
      (null "NaN")
      (float
       (cond ((sb-ext:float-nan-p value) "NaN")
             ((sb-ext:float-infinity-p value)
              (if (plusp value) "Infinity" "-Infinity"))
             (t (let ((*read-default-float-format* (type-of value)))
                  (princ-to-string value)))))
      (integer (princ-to-string value))
      (rational (let ((*read-default-float-format* 'double-float))
                  (princ-to-string (coerce value 'double-float))))
      (t (error 'unsupported-argument
                :backend backend-name
                :argument ":missing"
                :reason "the value that means missing must be a real number, or NIL for
the backend's own default")))))
