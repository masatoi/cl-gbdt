;;;; parameters.lisp --- Tests for parameter normalization.
;;;;
;;;; Backend-independent, so these need no shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/parameters
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/parameters)

(deftest normalize-parameters-returns-name-value-pairs
  (testing "keywords become snake_case names, values become strings"
    (ok (equal '(("objective" . "binary") ("num_leaves" . "31"))
               (cl-gbdt:normalize-parameters '(:objective "binary" :num-leaves 31))))))

(deftest normalize-parameters-rejects-odd-length-plist
  (testing "an odd-length plist signals data-error"
    (ok (handler-case (progn (cl-gbdt:normalize-parameters '(:objective)) nil)
          (cl-gbdt:data-error () t)))))

;;; I3: `parameter-value' used bare `princ-to-string', whose output depends on the
;;; caller's own printer bindings and on the value's exact Lisp type -- none of which
;;; a backend's numeric parser understands. Each of the following pins down one row
;;; of the review's confirmed-rejection table.

(deftest normalize-parameters-prints-double-float-without-a-type-marker
  (testing "0.05d0 prints as 0.05, not 0.05d0"
    (ok (equal '(("learning_rate" . "0.05"))
               (cl-gbdt:normalize-parameters '(:learning-rate 0.05d0))))))

(deftest normalize-parameters-prints-small-double-float-without-a-type-marker
  (testing "1.0d-7 prints with an e exponent marker, not a d one"
    (ok (equal '(("min_gain_to_split" . "1.0e-7"))
               (cl-gbdt:normalize-parameters '(:min-gain-to-split 1.0d-7))))))

(deftest normalize-parameters-prints-booleans-as-true-and-false
  (testing "T and NIL print as LightGBM's own boolean spelling"
    (ok (equal '(("is_unbalance" . "true"))
               (cl-gbdt:normalize-parameters '(:is-unbalance t))))
    (ok (equal '(("is_unbalance" . "false"))
               (cl-gbdt:normalize-parameters '(:is-unbalance nil))))))

(deftest normalize-parameters-prints-a-ratio-as-a-decimal
  (testing "1/3 prints as a decimal, not a fraction"
    (ok (equal '(("bagging_fraction" . "0.3333333333333333"))
               (cl-gbdt:normalize-parameters '(:bagging-fraction 1/3))))))

(deftest normalize-parameters-output-is-independent-of-the-caller-s-print-base
  (testing "an integer prints in decimal even when the caller has bound *print-base* to 16"
    (let ((*print-base* 16))
      (ok (equal '(("num_leaves" . "31"))
                 (cl-gbdt:normalize-parameters '(:num-leaves 31)))))))
