;;;; objective.lisp --- Pure helpers for `train''s custom objective.
;;;;
;;;; Neither function here touches a handle, a pointer or a shared library: one checks the
;;;; shape of what the caller's objective function returned, the other rewrites a parameter
;;;; plist. That is the same seam `missing-value.lisp', `categorical-features.lisp',
;;;; `feature-names.lisp' and `prediction-shape.lisp' sit on -- every boundary case is
;;;; reachable from layer 1, with no library present.
;;;;
;;;; This file is one level below `src/', so it is deliberately absent from `src/all.lisp''s
;;;; :use-reexport: nothing here is public API, and re-exporting it from `CL-GBDT' would
;;;; publish two internal helpers.

(uiop:define-package #:cl-gbdt/src/config/objective
  (:use #:cl)
  (:import-from #:cl-gbdt/src/conditions
                #:dimension-mismatch
                #:unsupported-element-type)
  ;; Imported, not reached through an export: `src/parameters' is one of the packages
  ;; `src/all.lisp' re-exports into `CL-GBDT', so exporting `parameter-name' there would
  ;; publish it as public API. Importing it here names the one file that needs it instead.
  (:import-from #:cl-gbdt/src/parameters
                #:parameter-name)
  (:export #:check-objective-result
           #:objective-single-float
           #:objective-parameters))

(in-package #:cl-gbdt/src/config/objective)

(defun %array-shape (array)
  "Return ARRAY's dimensions, or ARRAY itself when it is not an array.

An objective function that returned one value leaves the Hessian NIL, and one that returned
something else entirely leaves it that. Reporting the object rather than calling
`array-dimensions' on it is what keeps the failure a `dimension-mismatch' instead of a
`type-error' from inside the reporter."
  (if (arrayp array) (array-dimensions array) array))

(defun check-objective-result (grad hess rows groups)
  "Signal `dimension-mismatch' unless GRAD and HESS are both (ROWS GROUPS) arrays.

`train' calls this on whatever the caller's objective function returned, before anything is
written into a foreign buffer. The buffer write indexes by row and group, so an array of the
right total size but the wrong rank would be read as though it had the right one: a rank-1
vector of ROWS x GROUPS elements would produce a model, silently, from numbers in the wrong
places. This is the check that stops that, and it runs before the foreign call rather than
after, because by then the bytes are already in the library's hands.

The condition's GIVEN names both arrays -- `(:gradient DIMS :hessian DIMS)' -- rather than
just the offending one. With a single unlabelled shape a caller reading \"Expected: (40 3),
got: NIL\" cannot tell which of the two values they got wrong, and returning one value
instead of two is the most likely way to arrive here.

Element type is not checked here, and deliberately: `double-float', `single-float' and a
general array whose elements are reals are all accepted -- `(make-array (list rows 1))' with
no :ELEMENT-TYPE is what a caller most naturally writes, and it trains the same model the
specialized pair does. What each element must be is decided one element at a time by
`objective-single-float', where the buffer is written, so no separate validation scan is paid
for. A separate scan here would double a per-iteration cost purely for diagnostics."
  (unless (and (arrayp grad) (equal (array-dimensions grad) (list rows groups))
               (arrayp hess) (equal (array-dimensions hess) (list rows groups)))
    (error 'dimension-mismatch
           :expected (list rows groups)
           :given (list :gradient (%array-shape grad) :hessian (%array-shape hess))))
  (values))

(defun objective-single-float (value)
  "Return VALUE as a `single-float', signalling `unsupported-element-type' unless it is a
real.

Both backends write the caller's gradient and Hessian into a `const float*' buffer one
element at a time, coercing as they go -- see `cl-gbdt/src/lightgbm/native''s
`%update-one-iteration-custom' and `cl-gbdt/src/xgboost/native''s
`%train-one-iteration-custom'. This is that coercion, with the element's own check in front
of it, which is why the check costs the loop a `realp' test per element rather than the
extra pass a separate validation scan over both arrays would cost every iteration.

`unsupported-element-type', naming the offending element's TYPE, rather than the
`type-error' `coerce' raises for a string: it is the condition
`cl-gbdt/src/data''s `%require-real-values' already signals for a `csr-matrix' value that is
not a real, with the same `(type-of value)' in GIVEN, so the two places a caller's own
numbers reach a foreign buffer refuse a non-number the same way. It is signalled while the
foreign buffers exist -- `cffi:with-foreign-objects' has allocated them -- but before the
library has been called at all, and that allocation is unwound with everything else."
  (unless (realp value)
    (error 'unsupported-element-type :given (type-of value)))
  (coerce value 'single-float))

(defparameter *objective-parameter-names*
  '("objective" "objective_type" "app" "application" "loss")
  "Every spelling LightGBM honours for the `objective' parameter, as `parameter-name' renders
a key: lower case, underscores for dashes.

Read off the VENDORED LightGBM 4.7.0 itself rather than off a header or a web page. That
library exports `LGBM_DumpParamAliases', which returns its own parameter-to-alias map as JSON,
and its `objective' entry is exactly `[\"app\", \"loss\", \"application\", \"objective_type\"]'
-- four aliases, no more. None of the four is a parameter in its own right elsewhere in that
map, and none is an alias of any other parameter, so dropping them here can take nothing else
away from the caller.

Re-measured through the library's BEHAVIOUR as well, the way
`cl-gbdt/src/lightgbm/protocol''s `*categorical-feature-parameter-names*' was: each of the
four naming \"binary\" trains the model `objective=\"binary\"' trains, element for element,
while `obj', `objective_function', `loss_function', `app_type' and the plurals `objectives',
`apps', `applications' and `losses' all leave the trained numbers identical to a run that
named no objective at all. So this list can only be ENUMERATED, never prefix-matched or
fuzzily matched: `app' is honoured and `apps' is not.

A caller who names one of these while passing an objective function is in fact already
trained correctly by LightGBM -- it resolves the canonical `objective=none' this function
appends ahead of an alias, measured the same way. That is a precedence rule neither library
documents and nothing here can hold to, which is why the aliases are dropped rather than left
to lose a race.")

(defun objective-parameters (parameters)
  "Return PARAMETERS with every `objective' entry replaced by one naming \"none\".

LightGBM's `LGBM_BoosterUpdateOneIterCustom' refuses to run while the booster holds an
objective function -- `Check failed: objective_function_ == nullptr' -- so a custom-objective
run there has exactly one workable value for that parameter, and this is what supplies it.
Measured against the vendored library: with any other objective the call returns non-zero and
nothing trains.

Existing entries are dropped rather than left in place with a second one appended, because a
parameter string holding two `objective=' entries leaves which one wins to LightGBM. They are
matched through `parameter-name', the same function that decides what the key would have been
called in that string, so `:objective' and the string \"objective\" are one key here exactly
as they are one key to the library.

Dropped by that rendering against `*objective-parameter-names*', which is the FIVE spellings
the library honours and not the literal one alone: `objective_type', `app', `application' and
`loss' all set the same parameter, so a plist naming any of them alongside the entry appended
here would be the two-`objective=' string this paragraph exists to avoid. See that variable
for where the five come from.

Every other parameter passes through in its original order, `num_class' included: it is still
what tells LightGBM how many output groups a multiclass custom objective has, and forcing the
objective to \"none\" does not supply it."
  (append (loop :for (key value) :on parameters :by #'cddr
                :unless (member (parameter-name key) *objective-parameter-names*
                                :test #'string=)
                  :append (list key value))
          (list :objective "none")))
