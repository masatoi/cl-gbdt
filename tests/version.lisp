;;;; version.lisp --- Tests for the recorded version ranges and their comparison.
;;;;
;;;; Every case here is a unit test of pure functions -- no shared library, no
;;;; `open-backend' call anywhere in this file (layer 1). Layer 2's
;;;; `tests/functional/xgboost-api.lisp' proves the real, wired-up behavior: opening
;;;; the vendored XGBoost library signals no warning.

(uiop:define-package #:cl-gbdt/tests/version
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/version)

;;; ---------------------------------------------------------------------------
;;; version-compare

(deftest version-compare-orders-by-component
  (testing "equal versions compare :equal"
    (ok (eq :equal (cl-gbdt:version-compare "1.2.3" "1.2.3"))))
  (testing "a smaller patch compares :less"
    (ok (eq :less (cl-gbdt:version-compare "1.2.3" "1.2.4"))))
  (testing "a larger patch compares :greater"
    (ok (eq :greater (cl-gbdt:version-compare "1.2.4" "1.2.3"))))
  (testing "an earlier, more significant component decides the order"
    (ok (eq :greater (cl-gbdt:version-compare "2.0.0" "1.9.9")))
    (ok (eq :less (cl-gbdt:version-compare "1.9.9" "2.0.0")))))

(deftest version-compare-rejects-unparseable-input
  (testing "an unparseable first argument returns NIL, not a comparison"
    (ok (null (cl-gbdt:version-compare "not-a-version" "1.2.3"))))
  (testing "an unparseable second argument returns NIL, not a comparison"
    (ok (null (cl-gbdt:version-compare "1.2.3" "not-a-version"))))
  (testing "the wrong number of components is unparseable"
    (ok (null (cl-gbdt:version-compare "1.2" "1.2.3")))
    (ok (null (cl-gbdt:version-compare "1.2.3.4" "1.2.3"))))
  (testing "NIL as either argument returns NIL, not sorting as the smallest version"
    (ok (null (cl-gbdt:version-compare nil "1.2.3")))
    (ok (null (cl-gbdt:version-compare "1.2.3" nil)))
    (ok (null (cl-gbdt:version-compare nil nil)))))

;;; ---------------------------------------------------------------------------
;;; version-in-range-p

(deftest version-in-range-p-covers-inside-and-both-boundaries
  (testing "a version strictly between the bounds is in range"
    (ok (cl-gbdt:version-in-range-p "2.0.0" "1.0.0" "3.0.0")))
  (testing "the low bound itself is in range (inclusive)"
    (ok (cl-gbdt:version-in-range-p "1.0.0" "1.0.0" "3.0.0")))
  (testing "the high bound itself is in range (inclusive)"
    (ok (cl-gbdt:version-in-range-p "3.0.0" "1.0.0" "3.0.0")))
  (testing "a single-version range (low = high) accepts exactly that version"
    (ok (cl-gbdt:version-in-range-p "3.3.0" "3.3.0" "3.3.0"))))

(deftest version-in-range-p-covers-below-and-above
  (testing "a version below the low bound is out of range"
    (ng (cl-gbdt:version-in-range-p "0.9.9" "1.0.0" "3.0.0")))
  (testing "a version above the high bound is out of range"
    (ng (cl-gbdt:version-in-range-p "3.0.1" "1.0.0" "3.0.0"))))

(deftest version-in-range-p-cannot-confirm-unparseable-or-nil
  (testing "an unparseable version cannot be confirmed in range"
    (ng (cl-gbdt:version-in-range-p "not-a-version" "1.0.0" "3.0.0")))
  (testing "a NIL version cannot be confirmed in range"
    (ng (cl-gbdt:version-in-range-p nil "1.0.0" "3.0.0")))
  (testing "unparseable bounds likewise cannot confirm anything in range"
    (ng (cl-gbdt:version-in-range-p "2.0.0" "not-a-version" "3.0.0"))
    (ng (cl-gbdt:version-in-range-p "2.0.0" "1.0.0" "not-a-version"))))

;;; ---------------------------------------------------------------------------
;;; check-backend-version -- the warn/no-warn wiring, still without any library

(defun %fixture-range ()
  "Return a `version-range' with distinct VERIFIED and INFERRED bounds, so a test can
tell which one a version was actually checked against."
  (cl-gbdt:make-version-range
   :verified-low "3.3.0" :verified-high "3.3.0"
   :verified-evidence "105 functional assertions pass against it"
   :inferred-low "1.7.0" :inferred-high "3.3.0"
   :inferred-evidence "the imported functions' declarations are unchanged"))

(defmacro %collecting-untested-backend-version-warnings (&body body)
  "Evaluate BODY, returning the list of `untested-backend-version' conditions it
signalled, most recent first.

`handler-bind', not `handler-case': a WARNING does not unwind the stack on its own --
`handler-case' would establish a non-local exit and so change control flow instead of
merely observing, a documented pitfall in this repo's own
prompts/repl-driven-development.md. `handler-bind' runs the collector and then lets
`warn' continue normally, exactly as an uninstrumented caller would see it."
  (let ((warnings (gensym "WARNINGS")))
    `(let ((,warnings nil))
       (handler-bind ((cl-gbdt:untested-backend-version
                         (lambda (c) (push c ,warnings))))
         ,@body)
       ,warnings)))

(deftest check-backend-version-does-not-warn-inside-the-inferred-range
  (testing "a version within the inferred range signals nothing"
    (ok (null (%collecting-untested-backend-version-warnings
                (cl-gbdt:check-backend-version :xgboost "2.0.0" (%fixture-range))))))
  (testing "the inferred low bound itself signals nothing"
    (ok (null (%collecting-untested-backend-version-warnings
                (cl-gbdt:check-backend-version :xgboost "1.7.0" (%fixture-range))))))
  (testing "the inferred high bound itself signals nothing"
    (ok (null (%collecting-untested-backend-version-warnings
                (cl-gbdt:check-backend-version :xgboost "3.3.0" (%fixture-range))))))
  (testing "a version outside VERIFIED but inside INFERRED does not warn"
    ;; 2.0.0 above differs from the fixture's VERIFIED-LOW/HIGH ("3.3.0") but is still
    ;; within INFERRED-LOW/HIGH ("1.7.0" to "3.3.0"); asserted again here, named for
    ;; what it proves -- gating on the wider range -- rather than folded silently into
    ;; the first case above.
    (ok (null (%collecting-untested-backend-version-warnings
                (cl-gbdt:check-backend-version :xgboost "2.0.0" (%fixture-range)))))))

(deftest check-backend-version-warns-below-the-inferred-range
  (testing "a version below the inferred low bound signals untested-backend-version"
    (let ((warnings (%collecting-untested-backend-version-warnings
                       (cl-gbdt:check-backend-version :xgboost "1.6.0" (%fixture-range)))))
      (ok (= 1 (length warnings)))
      (ok (eq :xgboost (cl-gbdt:untested-backend-version-backend (first warnings))))
      (ok (equal "1.6.0" (cl-gbdt:untested-backend-version-version (first warnings))))
      (ok (equal (cl-gbdt:version-range-tested-description (%fixture-range))
                 (cl-gbdt:untested-backend-version-tested (first warnings)))))))

(deftest check-backend-version-warns-above-the-inferred-range
  (testing "a version above the inferred high bound signals untested-backend-version"
    (let ((warnings (%collecting-untested-backend-version-warnings
                       (cl-gbdt:check-backend-version :xgboost "3.4.0" (%fixture-range)))))
      (ok (= 1 (length warnings))))))

(deftest check-backend-version-warns-on-unparseable-version
  (testing "a version that does not parse as MAJOR.MINOR.PATCH still warns"
    (let ((warnings (%collecting-untested-backend-version-warnings
                       (cl-gbdt:check-backend-version :xgboost "not-a-version"
                                                       (%fixture-range)))))
      (ok (= 1 (length warnings))))))

(deftest check-backend-version-warns-on-nil-version
  (testing "a NIL version -- e.g. a backend with no version entry point -- still warns"
    (let ((warnings (%collecting-untested-backend-version-warnings
                       (cl-gbdt:check-backend-version :xgboost nil (%fixture-range)))))
      (ok (= 1 (length warnings))))))

;;; ---------------------------------------------------------------------------
;;; version-range-tested-description

(deftest version-range-tested-description-distinguishes-verified-from-inferred
  (testing "a single-point verified range formats without a dash"
    (let ((description (cl-gbdt:version-range-tested-description (%fixture-range))))
      (ok (= 2 (length description)))
      (ok (search "3.3.0 verified" (first description)))
      (ng (search "3.3.0-3.3.0" (first description)))
      (ok (search "1.7.0-3.3.0 inferred" (second description)))))
  (testing "a verified range spanning two different versions formats with a dash"
    (let* ((range (cl-gbdt:make-version-range
                    :verified-low "4.0.0" :verified-high "4.7.0"
                    :verified-evidence "a CI version matrix ran the suite against both"
                    :inferred-low "3.0.0" :inferred-high "4.7.0"
                    :inferred-evidence "declarations unchanged"))
           (description (cl-gbdt:version-range-tested-description range)))
      (ok (search "4.0.0-4.7.0 verified" (first description))))))
