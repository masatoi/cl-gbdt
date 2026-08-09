;;;; missing-value.lisp --- Tests for rendering a missing-value sentinel as JSON.
;;;;
;;;; Backend-independent and pure, so these need no shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/missing-value
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/config/missing-value
                #:missing-value-json)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/missing-value)

(defun %json (value)
  (missing-value-json value :xgboost))

(deftest missing-value-json-renders-nil-as-the-sentinel-sent-today
  ;; The wrapper has always sent {"missing":NaN}. NIL must keep meaning exactly that, or
  ;; every existing caller's numbers change without anyone asking (policy section 14).
  (testing "NIL renders as NaN"
    (ok (equal "NaN" (%json nil)) "what NIL renders as")))

(deftest missing-value-json-renders-a-double-without-an-exponent-marker
  ;; Measured: XGBoost's config parser rejects "1.0d-5" outright --
  ;; json.cc:409 Expecting: "," -- so a Lisp marker reaching it is a hard failure.
  (testing "a double renders bare"
    (ok (equal "-999.0" (%json -999.0d0)) "what -999.0d0 renders as"))
  (testing "a small double renders in exponent form the parser accepts"
    (ok (equal "1.0e-5" (%json 1.0d-5)) "what 1.0d-5 renders as")))

(deftest missing-value-json-renders-a-single-float-without-a-marker
  (testing "a single-float renders bare too"
    (ok (equal "-999.0" (%json -999.0)) "what the single-float -999.0 renders as")))

(deftest missing-value-json-renders-a-ratio-as-a-decimal
  ;; princ-to-string on a ratio prints "1/3", which is not a JSON number.
  (testing "a ratio becomes a decimal"
    (ok (equal "0.3333333333333333" (%json 1/3)) "what 1/3 renders as")))

(deftest missing-value-json-renders-an-integer-in-decimal
  (testing "an integer renders in base ten"
    (ok (equal "255" (%json 255)) "what 255 renders as")))

(deftest missing-value-json-ignores-the-caller-s-print-base
  ;; The rendered text must not depend on bindings already in force in the caller:
  ;; under *print-base* 16 a naive princ-to-string of 255 is "FF", a valid-looking
  ;; token that means a different number.
  (testing "*print-base* 16 does not turn 255 into FF"
    (ok (equal "255" (let ((*print-base* 16)) (%json 255)))
        "what 255 renders as under *print-base* 16")))

(deftest missing-value-json-renders-nan-and-the-infinities
  ;; Measured: SBCL princs a NaN as "#<DOUBLE-FLOAT quiet NaN>" and an infinity as
  ;; "#.SB-EXT:DOUBLE-FLOAT-POSITIVE-INFINITY". XGBoost's parser accepts the bare
  ;; tokens NaN, Infinity and -Infinity.
  (testing "an explicit NaN renders as NaN"
    (ok (equal "NaN" (%json (sb-kernel:make-double-float -524288 0)))
        "what an explicit NaN renders as"))
  (testing "positive infinity renders as Infinity"
    (ok (equal "Infinity" (%json sb-ext:double-float-positive-infinity))
        "what +infinity renders as"))
  (testing "negative infinity renders as -Infinity"
    (ok (equal "-Infinity" (%json sb-ext:double-float-negative-infinity))
        "what -infinity renders as")))

(deftest missing-value-json-renders-an-overflowing-rational-as-an-infinity
  ;; Measured: (/ (expt 10 401) 3) is a ratio, so it takes the `rational' branch, not
  ;; the `float' one. Coercing it to double-float overflows, and before the fix the
  ;; rational branch princ'd that overflow raw: "#.DOUBLE-FLOAT-POSITIVE-INFINITY" (and
  ;; the negative form for the negation), neither a valid JSON number token. Reproduced
  ;; both with and without float traps masked -- the coercion signals nothing, it just
  ;; overflows -- so this is not specific to reaching XGBoost through a foreign call.
  (testing "a huge positive ratio overflows to Infinity"
    (ok (equal "Infinity" (%json (/ (expt 10 401) 3)))
        "what (/ (expt 10 401) 3) renders as"))
  (testing "a huge negative ratio overflows to -Infinity"
    (ok (equal "-Infinity" (%json (- (/ (expt 10 401) 3))))
        "what (- (/ (expt 10 401) 3)) renders as")))

(deftest missing-value-json-rejects-a-value-that-is-not-a-real
  (testing "a string signals unsupported-argument"
    (ok (handler-case (progn (%json "-999.0") nil)
          (cl-gbdt:unsupported-argument () t))
        "whether a string sentinel was rejected"))
  (testing "a complex number signals unsupported-argument"
    (ok (handler-case (progn (%json #C(1 2)) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether a complex sentinel was rejected")))

(deftest missing-value-json-names-the-backend-it-was-given
  ;; The condition has to say which backend refused, the way every other
  ;; unsupported-argument in this project does.
  (testing "the condition carries the backend name"
    (ok (eq :lightgbm
            (handler-case (progn (missing-value-json "x" :lightgbm) nil)
              (cl-gbdt:unsupported-argument (c)
                (cl-gbdt:unsupported-argument-backend c))))
        "the backend named by the condition")))
