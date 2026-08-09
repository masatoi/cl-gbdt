;;;; feature-names.lisp --- Validating a caller's feature-name list.
;;;;
;;;; Backend-neutral and pure: no handle, no pointer, no shared library, so every path here
;;;; is layer-1 testable. What it catches is a list `listp' is happy with and `length' is
;;;; not: on SBCL 2.6.7, `(length '("a" . "b"))' signals a raw `type-error' and `(length
;;;; circular)' does not return -- both measured. See `check-feature-names''s docstring for
;;;; why the test is `proper-list-p' and not `listp'.
;;;;
;;;; Under src/config/ rather than directly under src/, and so deliberately absent from
;;;; src/all.lisp's `use-reexport' list, for the same reason `src/config/missing-value.lisp'
;;;; is absent from it -- see that file's own header. Both backends' `%set-feature-names'
;;;; are this file's only callers outside tests/; publishing this from `CL-GBDT' would put an
;;;; internal shape check on the public surface, where `make-dataset''s own docstring already
;;;; states the contract it enforces.
;;;;
;;;; Consumers: `cl-gbdt/src/lightgbm/native' and `cl-gbdt/src/xgboost/native', each from
;;;; its own `%set-feature-names'.

(uiop:define-package #:cl-gbdt/src/config/feature-names
  (:use #:cl)
  (:import-from #:alexandria
                #:proper-list-p)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-argument)
  (:export #:check-feature-names))

(in-package #:cl-gbdt/src/config/feature-names)

(defun check-feature-names (feature-names backend-name)
  "Return FEATURE-NAMES, after confirming it is a proper list, signalling
`unsupported-argument' against BACKEND-NAME when it is not.

Returns its argument so a call site can wrap an existing expression without restructuring it.
NIL is a proper list and passes: it means no names, which is what every caller who omits
`:feature-names' passes.

`proper-list-p' rather than `listp', for the reason
`cl-gbdt/src/config/categorical-features''s `%check-indices' gives at length: `listp' is true
of a DOTTED list too, so the guard would pass and the traversal below it fail with a raw
`type-error' instead of the typed condition `make-dataset''s docstring promises -- and a
CIRCULAR list would make that traversal run forever. `proper-list-p' answers both, and
answers the circular case without following the cycle.

The element type is deliberately not checked here. A non-string among the names is a
`foreign-string-alloc' argument error naming the value, which is already a clear report; what
this function exists for is the shape of the list, where the failure is either untyped or
absent."
  (unless (proper-list-p feature-names)
    (error 'unsupported-argument
           :backend backend-name
           :argument ":feature-names"
           :reason (format nil "the feature-name list must be a proper list of ~
                                name strings")))
  feature-names)
