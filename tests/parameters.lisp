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
