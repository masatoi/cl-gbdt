;;;; objective.lisp --- Layer 1 tests for the custom objective's pure helpers.
;;;;
;;;; Both functions here are pure: no handle, no pointer, no shared library. They are the
;;;; parts of the custom-objective path that can be tested without either library present,
;;;; which is what keeps `foreign libraries open: NIL' true for this suite.

(uiop:define-package #:cl-gbdt/tests/objective
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/config/objective
                #:check-objective-result
                #:objective-single-float
                #:objective-parameters)
  (:import-from #:cl-gbdt/src/conditions
                #:dimension-mismatch
                #:unsupported-element-type
                #:unsupported-element-type-given))

(in-package #:cl-gbdt/tests/objective)

(defun grid (rows groups)
  (make-array (list rows groups) :element-type 'double-float :initial-element 0d0))

(deftest check-objective-result-accepts-the-shape-it-was-given
  (ok (null (multiple-value-list (check-objective-result (grid 4 3) (grid 4 3) 4 3))))
  ;; A single output group is still two dimensions, not one: the contract says (ROWS GROUPS)
  ;; for every model, so a regression caller writes (4 1) rather than a bare vector.
  (ok (null (multiple-value-list (check-objective-result (grid 4 1) (grid 4 1) 4 1))))
  ;; single-float is the other element type `make-dataset' already accepts for a matrix.
  (ok (null (multiple-value-list
             (check-objective-result
              (make-array '(4 3) :element-type 'single-float :initial-element 0.0)
              (make-array '(4 3) :element-type 'single-float :initial-element 0.0)
              4 3))))
  ;; And a general array, which is what `(make-array (list rows groups))' with no
  ;; :ELEMENT-TYPE gives -- the most natural thing a caller writes. The shape is the whole of
  ;; what this function judges; its elements are judged one at a time by
  ;; `objective-single-float' as the buffer is written, so an integer 1 here is as acceptable
  ;; as 1.0d0 and this must not reject either.
  (ok (null (multiple-value-list
             (check-objective-result (make-array '(4 3) :initial-element 0)
                                     (make-array '(4 3) :initial-element 1)
                                     4 3)))))

(deftest objective-single-float-coerces-every-real-it-is-given
  ;; The three element types the contract admits, plus the two non-float reals a general
  ;; array can hold: each comes back as the `single-float' the C signature's `const float*'
  ;; wants, and every value below is exactly representable, so `=' is the right assertion.
  (ok (eql 1.5f0 (objective-single-float 1.5d0)))
  (ok (eql 1.5f0 (objective-single-float 1.5f0)))
  (ok (eql 1.5f0 (objective-single-float 3/2)))
  (ok (eql 2.0f0 (objective-single-float 2)))
  (ok (typep (objective-single-float 0) 'single-float)))

(deftest objective-single-float-refuses-an-element-that-is-not-a-real
  ;; A general array can hold anything, so this is the failure a caller reaches by returning
  ;; one holding a string or NIL. It must be this library's own condition naming what was
  ;; found, not the raw `type-error' `coerce' would have signalled from inside the buffer
  ;; write -- the same condition, with the same `(type-of value)' in GIVEN, that
  ;; `cl-gbdt/src/data''s `%require-real-values' signals for a `csr-matrix' value that is not
  ;; a real.
  (dolist (value (list "nope" nil #\a '(1 2) #c(1.0d0 2.0d0)))
    (let ((condition (handler-case (progn (objective-single-float value) nil)
                       (unsupported-element-type (c) c))))
      (ok condition (format nil "~S was refused" value))
      (ok (and condition (equal (type-of value) (unsupported-element-type-given condition)))
          (format nil "the condition named ~S" (type-of value))))))

(deftest check-objective-result-refuses-a-wrong-gradient-shape
  (ok (handler-case (progn (check-objective-result (grid 3 3) (grid 4 3) 4 3) nil)
        (dimension-mismatch () t)))
  ;; A flat vector of the right total size is the mistake a caller who thinks in one
  ;; dimension makes, and it must not reach the library: the buffer write indexes by row and
  ;; group, so a rank-1 array would be read as though it were rank 2.
  (ok (handler-case
          (progn (check-objective-result
                  (make-array 12 :element-type 'double-float :initial-element 0d0)
                  (grid 4 3) 4 3)
                 nil)
        (dimension-mismatch () t))))

(deftest check-objective-result-refuses-a-wrong-hessian-shape
  (ok (handler-case (progn (check-objective-result (grid 4 3) (grid 4 2) 4 3) nil)
        (dimension-mismatch () t)))
  ;; An objective function that returns only one value leaves HESS as NIL. That is the same
  ;; mistake, and it must be a typed condition rather than a type-error from `aref'.
  (ok (handler-case (progn (check-objective-result (grid 4 3) nil 4 3) nil)
        (dimension-mismatch () t))))

(deftest check-objective-result-names-which-array-was-wrong
  ;; The condition carries one `expected' and one `given'; without naming both arrays in
  ;; `given', a caller reading "Expected: (4 3), got: NIL" cannot tell which of the two they
  ;; got wrong.
  (let ((condition (handler-case (progn (check-objective-result (grid 4 3) nil 4 3) nil)
                     (dimension-mismatch (c) c))))
    (ok (search "GRADIENT" (format nil "~A" condition)))
    (ok (search "HESSIAN" (format nil "~A" condition)))))

(deftest objective-parameters-forces-none-over-whatever-was-asked-for
  (ok (equal (objective-parameters '(:objective "multiclass" :num-class 3))
             '(:num-class 3 :objective "none")))
  ;; Every other parameter survives, in its original order. num_class especially: LightGBM
  ;; still needs it to know how many output groups a multiclass custom objective has.
  (ok (equal (objective-parameters '(:num-leaves 7 :objective "regression" :verbose -1))
             '(:num-leaves 7 :verbose -1 :objective "none"))))

(deftest objective-parameters-adds-none-when-nothing-asked-for-an-objective
  (ok (equal (objective-parameters '()) '(:objective "none")))
  (ok (equal (objective-parameters '(:num-leaves 7)) '(:num-leaves 7 :objective "none"))))

(deftest objective-parameters-matches-the-spelling-the-backend-would-have-used
  ;; `parameter-name' is what turns a key into the string LightGBM reads, so a key that
  ;; renders as "objective" must be dropped however it was spelled -- otherwise the plist
  ;; carries two objective= entries and which one wins is LightGBM's business, not ours.
  (ok (equal (objective-parameters '("objective" "regression" :num-leaves 7))
             '(:num-leaves 7 :objective "none"))))

(defparameter *objective-aliases* '(:objective-type :app :application :loss)
  "LightGBM's own aliases for `objective', restated here rather than imported from
`cl-gbdt/src/config/objective''s `*objective-parameter-names*' -- the same way
tests/functional/categorical-features.lisp restates
`*categorical-feature-parameter-names*' rather than reading it. A test that imported the
list would agree with the source by construction and could never disagree with the library.

The four come from the vendored LightGBM 4.7.0 itself, via `LGBM_DumpParamAliases':
`\"objective\": [\"app\", \"loss\", \"application\", \"objective_type\"]'.
tests/functional/custom-objective.lisp's `lightgbm-forces-its-objective-parameter-to-none'
re-measures them against the real library; this test only pins that each is dropped.")

(defparameter *objective-near-misses*
  '(:obj :objectives :apps :applications :losses :loss-function :objective-function
    :objective-seed)
  "Keys that LOOK like they name the objective and do not, every one of which must survive
`objective-parameters' untouched.

The first seven are unknown to the vendored LightGBM entirely -- measured, each leaves the
trained numbers identical to a run that named no objective at all -- and they are here
because they are so nearly the real thing. `apps' and `applications' are the ones that
matter: they are the plurals of `app' and `application', both of which ARE honoured, so a
prefix match or a fuzzy match would eat an ordinary caller parameter along with the alias.

`objective_seed' is the eighth and the sharpest, because it is not a near-miss at all but a
REAL LightGBM parameter: it appears in `LGBM_DumpParamAliases''s map in its own right, with
no aliases of its own, and it is the only parameter in that map besides `objective' itself
whose name starts with any of the five spellings. Dropping it would delete a caller's
configuration outright.")

(deftest objective-parameters-drops-every-spelling-lightgbm-honours
  ;; The four aliases are live keys in LightGBM 4.7.0, not historical spellings: a plist
  ;; naming one of them alongside the `:objective "none"' this function appends is exactly
  ;; the two-`objective=' parameter string the docstring above says must never be built.
  ;; Before this, only the literal spelling was dropped and the override survived on
  ;; LightGBM's undocumented rule that the canonical key beats an alias.
  (dolist (alias *objective-aliases*)
    (ok (equal (objective-parameters (list alias "regression" :num-leaves 7))
               '(:num-leaves 7 :objective "none"))
        (format nil "~S was dropped" alias))
    ;; Spelled as the string the library actually reads, too -- `parameter-name' is what
    ;; makes the keyword and the string one key here, as they are one key to LightGBM.
    (ok (equal (objective-parameters
                (list (substitute #\_ #\- (string-downcase (string alias))) "regression"))
               '(:objective "none"))
        (format nil "the string spelling of ~S was dropped" alias))))

(deftest objective-parameters-keeps-a-key-that-merely-looks-like-the-objective
  ;; The other half, and the half a prefix match would fail: an ordinary parameter whose name
  ;; resembles one of the five must reach the library untouched, or `objective-parameters'
  ;; silently deletes a caller's configuration.
  (dolist (key *objective-near-misses*)
    (ok (equal (objective-parameters (list key "regression"))
               (list key "regression" :objective "none"))
        (format nil "~S survived" key))))
