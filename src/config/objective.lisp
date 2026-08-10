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
                #:dimension-mismatch)
  (:import-from #:cl-gbdt/src/parameters
                #:parameter-name)
  (:export #:check-objective-result
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

Element type is not checked: `double-float' and `single-float' are both accepted, matching
what `make-dataset' accepts for a dense matrix, and the conversion to `single-float' happens
where the buffer is written."
  (unless (and (arrayp grad) (equal (array-dimensions grad) (list rows groups))
               (arrayp hess) (equal (array-dimensions hess) (list rows groups)))
    (error 'dimension-mismatch
           :expected (list rows groups)
           :given (list :gradient (%array-shape grad) :hessian (%array-shape hess))))
  (values))

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
called in that string, so `:objective' and the string \"objective\" are both caught -- a key
that renders as `objective' is an objective however the caller spelled it.

Every other parameter passes through in its original order, `num_class' included: it is still
what tells LightGBM how many output groups a multiclass custom objective has, and forcing the
objective to \"none\" does not supply it."
  (append (loop :for (key value) :on parameters :by #'cddr
                :unless (string-equal "objective" (parameter-name key))
                  :append (list key value))
          (list :objective "none")))
