;;;; objective.lisp --- Layer 1 tests for the custom objective's pure helpers.
;;;;
;;;; Both functions here are pure: no handle, no pointer, no shared library. They are the
;;;; parts of the custom-objective path that can be tested without either library present,
;;;; which is what keeps `foreign libraries open: NIL' true for this suite.

(uiop:define-package #:cl-gbdt/tests/objective
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/config/objective
                #:check-objective-result
                #:objective-parameters)
  (:import-from #:cl-gbdt/src/conditions
                #:dimension-mismatch))

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
              4 3)))))

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
