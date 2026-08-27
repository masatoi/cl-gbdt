;;;; implicit-value.lisp --- Refusing a CSR absence declaration the backend does not match.
;;;;
;;;; Backend-neutral and pure: no handle, no pointer, no shared library. A `csr-matrix' may
;;;; declare what an entry it does not store means -- see `make-csr-matrix''s :IMPLICIT-VALUE
;;;; -- and the two libraries disagree about it: LightGBM reads an absent entry as 0.0,
;;;; XGBoost reads one as missing. This file holds the one comparison, so the two backends
;;;; cannot come to disagree about which declaration they refuse or what they say about it.
;;;;
;;;; REFUSAL IS THE WHOLE MECHANISM. Nothing here simulates: no parameter is set on the
;;;; caller's behalf, no matrix is densified, no INDPTR is rewritten. Policy section 15
;;;; forbids reproducing by simulation, on one backend, a concept only the other has.
;;;;
;;;; Under src/config/ rather than directly under src/, and so deliberately absent from
;;;; src/all.lisp's `use-reexport' list, for the same reason `src/config/missing-value.lisp'
;;;; is -- see that file's own header.
;;;;
;;;; Consumers: `cl-gbdt/src/lightgbm/api' and `cl-gbdt/src/xgboost/api'.

(uiop:define-package #:cl-gbdt/src/config/implicit-value
  (:use #:cl)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-argument)
  (:import-from #:cl-gbdt/src/data
                #:csr-matrix-implicit-value)
  (:export #:check-implicit-value))

(in-package #:cl-gbdt/src/config/implicit-value)

(defparameter +zero-declared-reason+
  "an absent entry is missing to this backend, not 0.0. Store the zeros explicitly and declare
:NONE if 0.0 is what you mean."
  "Why a zero declaration is refused: the backend reads absence as missing.

Says what to do instead rather than asserting that no configuration could change it. This
project measures what it claims, and no run here has established that for XGBoost.")

(defparameter +missing-declared-reason+
  "an absent entry is 0.0 to this backend, not missing. LightGBM's own `zero_as_missing' can
make one missing, but only when it is set on the booster as well as the dataset, and cl-gbdt
reads neither -- so it cannot verify this declaration. Leave :IMPLICIT-VALUE undeclared if you
have set that flag."
  "Why a `:MISSING' declaration is refused: the backend reads absence as 0.0.

Measured 2026-08-26: with `zero_as_missing' set on both the dataset and the booster, LightGBM
trained BYTE-IDENTICAL models from a matrix storing every element and from the same matrix with
its zeros dropped -- absence and a stored zero become the same thing. So `:MISSING' IS a true
claim under that configuration and this refusal refuses a true one. It says the wrapper does
not read the flag, NOT that LightGBM cannot do it: the second is false, and a false statement
does not belong on the public surface. `make-dataset' never sees the booster's parameters and
`predict' sees neither, so a check that read the dataset's would answer confidently and
wrongly.")

(defun check-implicit-value (backend-name matrix absent-means)
  "Signal `unsupported-argument' when MATRIX declares an absence meaning BACKEND-NAME does not
match. Return NIL otherwise.

ABSENT-MEANS is what an absent entry means to the calling backend: `0.0d0' for LightGBM,
`:MISSING' for XGBoost. Two of the four declarations always pass -- NIL declares nothing, and
`:NONE' declares that nothing is absent, which `make-csr-matrix' has already verified
structurally and which is therefore true on either backend. The remaining two pass only on the
backend that reads absence their way.

ABSENT-MEANS is what its backend reads **in that backend's default configuration**, so passing
is as unverified as refusing. LightGBM's `zero_as_missing' is the case that makes this concrete
-- under it a zero declaration is false and this accepts it, while `:MISSING' is true and this
refuses it -- and neither call site can see the flag, since it takes effect only when the
booster carries it too. This function therefore states a comparison against a documented
default, not against a configuration it has inspected; a caller who has changed that default
should declare nothing at all. See each backend's `%check-implicit-value' for the same point in
that backend's own terms.

`eql' rather than `equal' or a numeric `=': `make-csr-matrix' canonicalizes every zero real to
`0.0d0' precisely so this comparison can be identity-shaped, and `=' would additionally have to
guard against a NaN declaration that `%require-legal-implicit-value' has already refused."
  (let ((declared (csr-matrix-implicit-value matrix)))
    (when (and declared
               (not (eq declared :none))
               (not (eql declared absent-means)))
      (error 'unsupported-argument
             :backend backend-name
             :argument (format nil "the matrix's declared :IMPLICIT-VALUE ~S" declared)
             :reason (if (eq declared :missing)
                         +missing-declared-reason+
                         +zero-declared-reason+)))))
