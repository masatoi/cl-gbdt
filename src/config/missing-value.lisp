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

(defun %float-json (value)
  "Render the FLOAT value as a JSON number token, NaN and the infinities included.

NaN and the infinities are tested for before the general case, since `floatp' answers
T for all three and each has a bare JSON token XGBoost accepts. They are tested with
`sb-ext:float-nan-p' and `sb-ext:float-infinity-p' rather than by printing, because
SBCL princs them as `\"#<DOUBLE-FLOAT quiet NaN>\"' and
`\"#.SB-EXT:DOUBLE-FLOAT-POSITIVE-INFINITY\"'.

`(plusp value)' on an infinity is an ordinary comparison against zero, well-defined
and signalling nothing; only a NaN comparison would, and that branch is taken first."
  (cond ((sb-ext:float-nan-p value) "NaN")
        ((sb-ext:float-infinity-p value)
         (if (plusp value) "Infinity" "-Infinity"))
        (t (let ((*read-default-float-format* (type-of value)))
             (princ-to-string value)))))

(defun %rational-json (value)
  "Render the ratio VALUE as a JSON number token, coercing it to `double-float' first.

`missing-value-json''s `integer' clause takes integers ahead of this one, so VALUE is
always a non-integer rational, which `princ-to-string' would print as `\"1/3\"' -- not a
JSON number. The coerced double goes through `%float-json', so a ratio far outside the
double-float range renders as an infinity for the same reason an overflowing float does.

The coercion is wrapped in a `handler-case' because whether it *signals* is a property
of the platform rather than of VALUE: SBCL enables the `:overflow' trap by default on
x86-64 and on macOS aarch64, and none of the traps on Linux aarch64 -- see CLAUDE.md's
floating-point trap masking section for that split. The same ratio therefore raises
`floating-point-overflow' on the first two and quietly overflows to an infinity on the
third. Substituting the infinity the untrapped platforms produce makes the rendered
token identical on all of them, whatever traps the caller has in force. `(plusp value)'
is an exact comparison on the original rational, so choosing between the two infinities
cannot itself trap."
  (%float-json (handler-case (coerce value 'double-float)
                 (floating-point-overflow ()
                   (if (plusp value)
                       sb-ext:double-float-positive-infinity
                       sb-ext:double-float-negative-infinity)))))

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

NaN and the infinities are tested for before the general float case, in `%float-json',
since `floatp' answers T for all three and each has a bare JSON token XGBoost accepts;
see that function's docstring for why they are tested rather than printed. A rational
that overflows on coercion to `double-float' -- an exact ratio far outside the
double-float range -- renders as an infinity for the same reason an overflowing float
does: `%rational-json' passes it through `%float-json' too, after the coercion.

Which token VALUE renders as never depends on the floating-point trap state in force at
the call. That is worth guaranteeing because the state is not the same everywhere: SBCL
enables the `:overflow' trap by default on x86-64 and on macOS aarch64 and none of the
traps on Linux aarch64 (CLAUDE.md's floating-point trap masking section), so an
overflowing coercion signals on two of this project's three CI platforms and returns an
infinity on the third. Every caller under src/ reaches here from an XGBoost protocol
method whose whole body is masked, which hides the difference in production -- but not
from a caller outside such a body, which `cl-gbdt/tests/missing-value' is, and not from
the next caller to be written. `%rational-json' is where the difference is absorbed."
  (let ((*print-base* 10)
        (*print-radix* nil))
    (typecase value
      (null "NaN")
      (float (%float-json value))
      (integer (princ-to-string value))
      (rational (%rational-json value))
      (t (error 'unsupported-argument
                :backend backend-name
                :argument ":missing"
                :reason (format nil "the value that means missing must be a real ~
                                     number, or NIL for the backend's own default"))))))
