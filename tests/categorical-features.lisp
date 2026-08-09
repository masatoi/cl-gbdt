;;;; categorical-features.lisp --- Tests for rendering a categorical-column list.
;;;;
;;;; Backend-independent and pure, so these need no shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/categorical-features
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/config/categorical-features
                #:categorical-feature-types
                #:categorical-feature-string)
  ;; Zero symbols: `cffi:null-pointer' below is the only reference, package-qualified. It
  ;; allocates nothing and opens nothing, so this file stays layer 1.
  (:import-from #:cffi)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/categorical-features)

(defun %matrix (num-features)
  "A dense matrix of one row and NUM-FEATURES columns. Only its shape is ever read."
  (make-array (list 1 num-features) :element-type 'double-float))

(defun %csr (num-features)
  "A `csr-matrix' of one empty row and NUM-FEATURES columns, for the same purpose."
  (cl-gbdt:make-csr-matrix :indptr '(0 0) :indices '() :values '()
                           :num-columns num-features))

(defun %foreign (num-features)
  "A `foreign-matrix' of one row and NUM-FEATURES columns, for the same purpose.

Its pointer is the null pointer: the renderers read the COLS slot and never the pointer, and
building one costs no foreign memory, so this stays a layer-1 fixture like the two above."
  (make-instance 'cl-gbdt:foreign-matrix :pointer (cffi:null-pointer)
                                         :rows 1 :cols num-features
                                         :element-type 'double-float))

(defun %types (indices num-features)
  (categorical-feature-types indices (%matrix num-features) :xgboost))

(defun %string (indices num-features)
  (categorical-feature-string indices (%matrix num-features) :xgboost))

(deftest categorical-feature-types-renders-nil-as-nothing-to-attach
  ;; NIL must mean "attach no feature_type at all", not "every column is quantitative".
  ;; Those differ: the second writes a full type vector where the wrapper wrote none before,
  ;; which is a change to what every existing caller sends (policy section 14).
  (testing "NIL renders as NIL, not a vector of q"
    (ok (null (%types nil 4)) "what NIL renders as")))

(deftest categorical-feature-types-marks-only-the-listed-columns
  (testing "one index"
    (ok (equal '("c" "q" "q") (%types '(0) 3)) "what (0) renders as over 3 columns"))
  (testing "several indices, given out of order"
    (ok (equal '("q" "c" "q" "c") (%types '(3 1) 4))
        "what (3 1) renders as over 4 columns"))
  (testing "every column"
    (ok (equal '("c" "c") (%types '(0 1) 2)) "what (0 1) renders as over 2 columns")))

(deftest categorical-feature-string-joins-with-commas
  (testing "NIL renders as NIL"
    (ok (null (%string nil 4)) "what NIL renders as"))
  (testing "one index"
    (ok (equal "0" (%string '(0) 3)) "what (0) renders as"))
  (testing "several indices keep the caller's order"
    (ok (equal "3,1" (%string '(3 1) 4)) "what (3 1) renders as")))

(deftest categorical-feature-string-ignores-the-caller-s-print-base
  ;; The rendered text must not depend on bindings already in force in the caller:
  ;; under *print-base* 16 a naive princ-to-string of 255 is "FF", a valid-looking
  ;; token that names a different column. src/config/missing-value.lisp records the
  ;; same reasoning for the same reason.
  ;;
  ;; What holds this up is the renderer's `~D' directive, which binds *print-base* and
  ;; *print-radix* for its own argument, so this assertion is what pins that directive:
  ;; measured, rewriting the body to `~A' or to princ-to-string answers "FF" here and
  ;; fails it. It does not pin any `let' -- there is none, for that same reason.
  (testing "*print-base* 16 does not turn column 255 into FF"
    (ok (equal "255" (let ((*print-base* 16)) (%string '(255) 256)))
        "what (255) renders as under *print-base* 16")))

(deftest categorical-features-reads-a-csr-matrix-s-column-count
  ;; Both backends hand the renderer whatever the caller passed to make-dataset, so a
  ;; csr-matrix has to answer the same question a dense matrix does -- otherwise the
  ;; range check would silently not happen on the sparse path.
  (testing "a csr-matrix gives the same type vector as a dense matrix of the same width"
    (ok (equal '("q" "c" "q")
               (categorical-feature-types '(1) (%csr 3) :xgboost))
        "what (1) renders as over a 3-column csr-matrix"))
  (testing "and the same range check"
    (ok (handler-case (progn (categorical-feature-types '(3) (%csr 3) :xgboost) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether index 3 over a 3-column csr-matrix was rejected")))

(deftest categorical-features-reads-a-foreign-matrix-s-column-count
  ;; The third form make-dataset accepts, beside a dense array and a csr-matrix. It reaches
  ;; the renderer exactly as the other two do, and reading its column count with
  ;; array-dimension signalled a raw type-error instead of range-checking anything --
  ;; measured through make-dataset on both backends before this test existed.
  (testing "a foreign-matrix gives the same type vector as a dense matrix of the same width"
    (ok (equal (%types '(1) 3) (categorical-feature-types '(1) (%foreign 3) :xgboost))
        "what (1) renders as over a 3-column foreign-matrix"))
  (testing "and the same range check"
    (ok (handler-case (progn (categorical-feature-types '(3) (%foreign 3) :xgboost) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether index 3 over a 3-column foreign-matrix was rejected")))

(deftest categorical-features-rejects-an-array-that-is-not-2d
  ;; A rank-1 array is refused with this project's own dimension-mismatch -- the same
  ;; condition `cl-gbdt/src/data' signals for the same array -- rather than with
  ;; array-dimension's untyped "Vector axis is not zero: 1", which is what it answered before.
  (let ((vector (make-array 3 :element-type 'double-float)))
    (testing "a rank-1 array signals dimension-mismatch"
      (ok (handler-case (progn (categorical-feature-types '(0) vector :xgboost) nil)
            (cl-gbdt:dimension-mismatch () t))
          "whether a rank-1 array was rejected with dimension-mismatch"))
    (testing "the string renderer rejects it too"
      (ok (handler-case (progn (categorical-feature-string '(0) vector :lightgbm) nil)
            (cl-gbdt:dimension-mismatch () t))
          "whether the string renderer rejected a rank-1 array with dimension-mismatch"))))

(deftest categorical-features-rejects-a-non-integer
  (testing "a string index signals unsupported-argument"
    (ok (handler-case (progn (%types '("0") 3) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether a string index was rejected"))
  (testing "a float index signals unsupported-argument"
    (ok (handler-case (progn (%string '(1.0) 3) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether a float index was rejected")))

(deftest categorical-features-rejects-an-out-of-range-index
  ;; Both renderers must reject it, not just the one that needs the count to do its job:
  ;; LightGBM's parameter string would otherwise carry a column that does not exist,
  ;; and the same call would be refused on one backend and accepted on the other.
  (testing "an index equal to the column count is out of range"
    (ok (handler-case (progn (%types '(3) 3) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether index 3 over 3 columns was rejected"))
  (testing "the string renderer rejects it too"
    (ok (handler-case (progn (%string '(3) 3) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether index 3 over 3 columns was rejected by the string renderer"))
  (testing "a negative index is rejected"
    (ok (handler-case (progn (%types '(-1) 3) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether index -1 was rejected")))

(deftest categorical-features-rejects-a-duplicate
  ;; A duplicate is a caller mistake, not a harmless one: LightGBM would be handed
  ;; "1,1" and XGBoost a vector in which nothing records that it was said twice, so the
  ;; two backends would silently disagree about whether the call was well formed.
  (testing "a repeated index signals unsupported-argument"
    (ok (handler-case (progn (%types '(1 1) 3) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether (1 1) was rejected")))

(deftest categorical-features-names-the-backend-it-was-given
  ;; The condition has to say which backend refused, the way every other
  ;; unsupported-argument in this project does.
  (testing "the condition carries the backend name"
    (ok (eq :lightgbm
            (handler-case
                (progn (categorical-feature-string '(-1) (%matrix 3) :lightgbm) nil)
              (cl-gbdt:unsupported-argument (c)
                (cl-gbdt:unsupported-argument-backend c))))
        "the backend named by the condition")))

(deftest categorical-features-rejects-a-non-list
  ;; A bare atom is not a list at all -- listp already refused it before this fix, and
  ;; proper-list-p must keep refusing it, so the guard's other half stays pinned too.
  (testing "a bare integer signals unsupported-argument"
    (ok (handler-case (progn (%types 0 3) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether a non-list INDICES was rejected")))

(deftest categorical-features-rejects-a-dotted-list
  ;; listp is true of a dotted list too, so it let '(0 . 1) reach dolist/length and fail
  ;; with a raw type-error instead of unsupported-argument -- measured before this fix.
  ;; proper-list-p answers both correctly.
  (testing "a minimally dotted list signals unsupported-argument"
    (ok (handler-case (progn (%types '(0 . 1) 3) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether '(0 . 1) was rejected"))
  (testing "the string renderer rejects it too"
    (ok (handler-case (progn (%string '(0 . 1) 3) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether '(0 . 1) was rejected by the string renderer"))
  (testing "a list that starts proper and ends dotted signals unsupported-argument"
    ;; '(0 1 . 2) fails at a different point in the traversal than '(0 . 1): dolist gets
    ;; further before the tail's non-list cdr trips it up.
    (ok (handler-case (progn (%types '(0 1 . 2) 3) nil)
          (cl-gbdt:unsupported-argument () t))
        "whether '(0 1 . 2) was rejected")))

(deftest categorical-features-rejects-a-circular-list
  ;; listp is true of a circular list too, and dolist over one never terminates. A
  ;; regression back to listp must fail this test in seconds, not hang the CI job that
  ;; runs it until something kills it -- so the call itself is guarded by a timeout, and
  ;; a timeout is explicitly reported as a failed assertion rather than left to make `ok'
  ;; pass on a merely non-NIL sentinel, which a keyword like :DID-NOT-TERMINATE would.
  (let ((circular (list 0)))
    (setf (cdr circular) circular)
    (testing "a circular list signals unsupported-argument rather than looping"
      (handler-case
          (sb-ext:with-timeout 5
            (ok (handler-case (progn (%types circular 3) nil)
                  (cl-gbdt:unsupported-argument () t))
                "whether the circular list was rejected instead of hanging"))
        (sb-ext:timeout ()
          (ok nil "the circular list did not terminate within the 5-second timeout"))))))
