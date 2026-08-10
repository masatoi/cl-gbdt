;;;; prediction-shape.lisp --- Tests for deriving a contribution result's shape.
;;;;
;;;; Backend-independent and pure, so these need no shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/prediction-shape
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/config/prediction-shape
                #:contrib-shape))

(in-package #:cl-gbdt/tests/prediction-shape)

(deftest contrib-shape-splits-a-multiclass-total
  ;; Measured against both vendored libraries: a 3-class model over 4 features returns
  ;; 3 x (4 + 1) = 15 values per row, laid out class-major.
  (testing "18 rows, 4 features, 270 values"
    (ok (equal '(18 3 5) (contrib-shape 270 18 4)) "what 270/18/4 derives")))

(deftest contrib-shape-handles-a-single-class-model
  ;; Measured: XGBoost reports (8 1 4) for a BINARY model over 3 features -- one class,
  ;; not zero, and still three dimensions. The helper must agree rather than collapsing.
  (testing "8 rows, 3 features, 32 values"
    (ok (equal '(8 1 4) (contrib-shape 32 8 3)) "what 32/8/3 derives")))

(deftest contrib-shape-answers-nil-when-the-division-is-not-exact
  ;; A total that does not divide by rows x (features + 1) means the caller's counts and
  ;; the library's buffer disagree. NIL says "no shape to state", which is what the second
  ;; return value means everywhere else; inventing one here would be worse than silence.
  (testing "a total that is not a multiple"
    (ok (null (contrib-shape 271 18 4)) "what a non-multiple derives"))
  (testing "a total smaller than one row"
    (ok (null (contrib-shape 3 18 4)) "what an undersized total derives")))

(deftest contrib-shape-answers-nil-for-an-exact-division-with-zero-classes
  ;; 0 divides evenly by 1 x (4 + 1) -- quotient 0, remainder 0 -- but no real model
  ;; reports zero classes. This is a genuinely different branch than the two above: the
  ;; (plusp classes) guard rejects it, not the (zerop remainder) one, since the division
  ;; here is exact.
  (testing "an exact division whose quotient is zero"
    (ok (null (contrib-shape 0 1 4)) "what an exact zero-class division derives")))

(deftest contrib-shape-answers-nil-for-degenerate-counts
  (testing "zero rows"
    (ok (null (contrib-shape 0 0 4)) "what zero rows derives"))
  (testing "a negative feature count"
    (ok (null (contrib-shape 270 18 -1)) "what a negative feature count derives")))
