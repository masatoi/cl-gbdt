;;;; data.lisp --- Tests for foreign matrix handoff.
;;;;
;;;; These tests require no shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/data
  (:use #:cl #:rove)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/data)

(defun read-back-doubles (pointer count)
  "Read COUNT doubles from POINTER and return them as a list."
  (loop :for i :below count :collect (cffi:mem-aref pointer :double i)))

(defun read-back-floats (pointer count)
  "Read COUNT floats from POINTER and return them as a list."
  (loop :for i :below count :collect (cffi:mem-aref pointer :float i)))

(deftest double-matrix-is-passed-row-major
  (testing "a 2D double-float array is handed over in row-major order"
    (let ((matrix (make-array '(2 3) :element-type 'double-float
                              :initial-contents '((1d0 2d0 3d0) (4d0 5d0 6d0)))))
      (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
        (ok (= 2 nrow))
        (ok (= 3 ncol))
        (ok (eq 'double-float element-type))
        (ok (equal '(1d0 2d0 3d0 4d0 5d0 6d0) (read-back-doubles pointer 6)))))))

(deftest single-matrix-is-passed-row-major
  (testing "a 2D single-float array is handed over in row-major order"
    (let ((matrix (make-array '(2 2) :element-type 'single-float
                              :initial-contents '((1.5 2.5) (3.5 4.5)))))
      (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
        (ok (= 2 nrow))
        (ok (= 2 ncol))
        (ok (eq 'single-float element-type))
        (ok (equal '(1.5 2.5 3.5 4.5) (read-back-floats pointer 4)))))))

(deftest large-matrix-round-trips
  (testing "every element survives for a larger matrix"
    (let* ((rows 64) (cols 17)
           (matrix (make-array (list rows cols) :element-type 'double-float)))
      (dotimes (i rows)
        (dotimes (j cols)
          (setf (aref matrix i j) (coerce (+ (* i cols) j) 'double-float))))
      (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
        (declare (ignore element-type))
        (ok (= rows nrow))
        (ok (= cols ncol))
        (ok (loop :for k :below (* rows cols)
                  :always (= (coerce k 'double-float)
                             (cffi:mem-aref pointer :double k))))))))

(deftest non-simple-array-is-copied
  (testing "an adjustable array is handed over correctly via the copy path"
    (let ((matrix (make-array '(2 2) :element-type 'double-float
                              :adjustable t
                              :initial-contents '((7d0 8d0) (9d0 10d0)))))
      (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
        (declare (ignore element-type))
        (ok (= 2 nrow))
        (ok (= 2 ncol))
        (ok (equal '(7d0 8d0 9d0 10d0) (read-back-doubles pointer 4)))))))

(deftest unsupported-element-type-is-rejected
  (testing "an integer array is rejected with unsupported-element-type"
    (let ((matrix (make-array '(2 2) :element-type '(unsigned-byte 8)
                              :initial-contents '((1 2) (3 4)))))
      (ok (handler-case
              (progn (cl-gbdt:with-foreign-matrix (p r c e) matrix
                       (declare (ignore p r c e))
                       nil)
                     nil)
            (cl-gbdt:unsupported-element-type () t))))))

(deftest wrong-rank-is-rejected
  (testing "a 1D array is rejected with dimension-mismatch"
    (let ((vector (make-array 4 :element-type 'double-float
                              :initial-contents '(1d0 2d0 3d0 4d0))))
      (ok (handler-case
              (progn (cl-gbdt:with-foreign-matrix (p r c e) vector
                       (declare (ignore p r c e))
                       nil)
                     nil)
            (cl-gbdt:dimension-mismatch () t))))))

(deftest foreign-matrix-is-passed-through
  (testing "a matrix already in foreign memory is passed through unchanged"
    (cffi:with-foreign-object (buffer :double 4)
      (loop :for i :below 4
            :do (setf (cffi:mem-aref buffer :double i) (coerce (1+ i) 'double-float)))
      (let ((matrix (make-instance 'cl-gbdt:foreign-matrix
                                   :pointer buffer :rows 2 :cols 2
                                   :element-type 'double-float)))
        (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
          (ok (cffi:pointer-eq buffer pointer) "passed through without copying")
          (ok (= 2 nrow))
          (ok (= 2 ncol))
          (ok (eq 'double-float element-type)))))))

(deftest pinned-matrix-survives-garbage-collection
  (testing "the pointer stays valid across a collection during the call"
    (let ((matrix (make-array '(8 8) :element-type 'double-float)))
      (dotimes (i 8)
        (dotimes (j 8)
          (setf (aref matrix i j) (coerce (+ (* i 8) j) 'double-float))))
      (cl-gbdt:with-foreign-matrix (pointer nrow ncol element-type) matrix
        (declare (ignore nrow ncol element-type))
        ;; Allocate enough to provoke a collection while C would be holding POINTER.
        (dotimes (i 200)
          (make-array 4096 :element-type 'double-float))
        #+sbcl (sb-ext:gc :full t)
        (ok (loop :for k :below 64
                  :always (= (coerce k 'double-float)
                             (cffi:mem-aref pointer :double k)))
            "every element still reads back correctly after a full GC")))))

(deftest foreign-element-type-maps-to-cffi-keywords
  (testing "Lisp element types map to CFFI type keywords"
    (ok (eq :double (cl-gbdt:foreign-element-type 'double-float)))
    (ok (eq :float (cl-gbdt:foreign-element-type 'single-float)))))

(defun %csr (&key (indptr '(0 2 3)) (indices '(0 2 1))
                  (values '(1.0d0 2.0d0 3.0d0)) (num-columns 3) implicit-value)
  (cl-gbdt:make-csr-matrix :indptr indptr :indices indices :values values
                           :num-columns num-columns :implicit-value implicit-value))

(defun %signals-dimension-mismatch-p (thunk)
  ;; handler-case, not rove's `signals' -- see prompts/repl-driven-development.md.
  (handler-case (progn (funcall thunk) nil)
    (cl-gbdt:dimension-mismatch () t)))

(defun %signals-unsupported-argument-p (thunk)
  ;; handler-case, not rove's `signals' -- see prompts/repl-driven-development.md.
  (handler-case (progn (funcall thunk) nil)
    (cl-gbdt:unsupported-argument () t)))

(deftest csr-matrix-accepts-every-legal-implicit-value
  (testing "NIL, a zero, :MISSING and :NONE are each accepted and read back"
    (ok (null (cl-gbdt:csr-matrix-implicit-value (%csr)))
        "an undeclared matrix reads back NIL")
    (ok (eq :missing (cl-gbdt:csr-matrix-implicit-value (%csr :implicit-value :missing)))
        ":MISSING reads back")
    (ok (eql 0.0d0 (cl-gbdt:csr-matrix-implicit-value (%csr :implicit-value 0)))
        "the integer 0 is canonicalized to 0.0d0")
    (ok (eql 0.0d0 (cl-gbdt:csr-matrix-implicit-value (%csr :implicit-value -0.0)))
        "negative zero is canonicalized to the same 0.0d0, so a backend may compare with EQL")))

(deftest csr-matrix-rejects-an-illegal-implicit-value
  ;; A non-zero real is refused rather than stored: no backend implies a non-zero value for
  ;; absence, so it is a claim nothing could honour.
  (testing "anything outside the legal set signals"
    (dolist (bad '(1.0d0 -3 "missing" :zero t))
      (ok (%signals-unsupported-argument-p (lambda () (%csr :implicit-value bad)))
          (format nil "whether ~S was rejected" bad)))))

(deftest csr-matrix-none-accepts-a-matrix-storing-every-element
  (testing ":NONE is accepted when each row stores all NUM-COLUMNS columns exactly once"
    (ok (cl-gbdt:csr-matrix-implicit-value
         (%csr :indptr '(0 3) :indices '(0 1 2) :values '(1.0d0 2.0d0 3.0d0)
               :num-columns 3 :implicit-value :none))
        "a fully-stored single row")))

(deftest csr-matrix-none-rejects-a-short-row
  (testing ":NONE is refused when a row stores fewer entries than the declared width"
    (ok (%signals-dimension-mismatch-p (lambda () (%csr :implicit-value :none)))
        "%csr's default matrix stores 2 then 1 of 3 columns")))

(deftest csr-matrix-none-rejects-a-row-storing-one-column-twice
  ;; The case a count alone cannot catch, and the reason `%require-every-element-stored'
  ;; stamps columns rather than comparing lengths: this row stores three entries for three
  ;; columns, so the count is right, while column 0 is stored twice and column 2 not at all.
  ;; `make-csr-matrix' does not reject duplicate indices in general -- they are legal CSR --
  ;; so nothing else in this file would notice.
  (testing ":NONE is refused when a row repeats a column and omits another"
    (ok (%signals-dimension-mismatch-p
         (lambda () (%csr :indptr '(0 3) :indices '(0 0 1) :values '(1.0d0 2.0d0 3.0d0)
                          :num-columns 3 :implicit-value :none)))
        "three entries, three columns, column 2 never stored")))

(deftest csr-matrix-reports-its-shape
  (testing "the readers return what was built, already coerced"
    (let ((m (%csr)))
      (ok (= 3 (cl-gbdt:csr-matrix-num-columns m)) "the column count")
      (ok (= 2 (cl-gbdt:csr-matrix-num-rows m)) "the row count, from INDPTR's length")
      (ok (equalp #(1.0d0 2.0d0 3.0d0) (cl-gbdt:csr-matrix-values m)) "the values")
      (ok (typep (cl-gbdt:csr-matrix-values m) '(simple-array double-float (*)))
          "whether VALUES was coerced to a specialized vector")
      (ok (typep (cl-gbdt:csr-matrix-indptr m) '(simple-array (signed-byte 32) (*)))
          "whether INDPTR was coerced to a specialized vector"))))

(deftest csr-matrix-slots-are-read-only
  ;; `make-csr-matrix' validates everything a backend later pins and hands to C, so a
  ;; writable slot would make that validation defeatable after the fact: replacing INDPTR
  ;; with a list, for instance, reaches the C API as a raw TYPE-ERROR out of the pinning
  ;; code -- an implementation-defined condition -- rather than as one of this library's
  ;; own. Every slot is `:read-only t', so there is no writer to reach.
  ;;
  ;; Asserted as the absence of the `(setf ACCESSOR)' function name rather than by running
  ;; a `setf' form and catching something: with `:read-only t' that form no longer compiles
  ;; at all, so there is nothing left to run. Each reader's own `fboundp' is asserted
  ;; alongside it, so a NIL below cannot be a misspelled symbol rather than a missing
  ;; writer.
  (testing "none of the five accessors has a writer"
    (dolist (reader '(cl-gbdt:csr-matrix-indptr cl-gbdt:csr-matrix-indices
                      cl-gbdt:csr-matrix-values cl-gbdt:csr-matrix-num-columns
                      cl-gbdt:csr-matrix-implicit-value))
      (ok (fboundp reader)
          (format nil "whether ~S names a reader at all" reader))
      (ok (not (fboundp (list 'setf reader)))
          (format nil "whether (setf ~S) is undefined, leaving the slot unwritable"
                  reader)))))

(deftest csr-matrix-accepts-any-sequence
  ;; The same convention :label, :weight and :group already follow.
  (testing "a vector argument is accepted as readily as a list"
    (let ((m (%csr :indptr #(0 2 3) :indices #(0 2 1) :values #(1.0d0 2.0d0 3.0d0))))
      (ok (= 2 (cl-gbdt:csr-matrix-num-rows m)) "the row count from a vector INDPTR"))))

(deftest csr-matrix-rejects-a-decreasing-indptr
  (testing "an INDPTR that goes backwards signals"
    (ok (%signals-dimension-mismatch-p (lambda () (%csr :indptr '(0 3 2))))
        "whether a decreasing INDPTR was rejected")))

(deftest csr-matrix-rejects-an-indptr-not-starting-at-zero
  (testing "an INDPTR whose first element is not 0 signals"
    (ok (%signals-dimension-mismatch-p (lambda () (%csr :indptr '(1 2 3))))
        "whether an INDPTR not starting at 0 was rejected")))

(deftest csr-matrix-rejects-an-indptr-disagreeing-with-the-element-count
  ;; INDPTR's last element is the number of stored elements; if it disagrees with the two
  ;; arrays' lengths, the matrix describes a different number of entries than it carries.
  (testing "an INDPTR whose last element is not the element count signals"
    (ok (%signals-dimension-mismatch-p (lambda () (%csr :indptr '(0 2 5))))
        "whether a wrong final INDPTR element was rejected")))

(deftest csr-matrix-rejects-mismatched-indices-and-values
  (testing "INDICES and VALUES of different lengths signal"
    (ok (%signals-dimension-mismatch-p
         (lambda () (%csr :indices '(0 2) :values '(1.0d0 2.0d0 3.0d0))))
        "whether mismatched INDICES and VALUES were rejected")))

(deftest csr-matrix-rejects-a-column-index-outside-the-declared-width
  ;; NUM-COLUMNS is the declared width, not a hint. An index at or above it addresses a
  ;; column the matrix says it does not have.
  (testing "an index equal to NUM-COLUMNS signals"
    (ok (%signals-dimension-mismatch-p (lambda () (%csr :indices '(0 3 1))))
        "whether an out-of-range column index was rejected"))
  (testing "a negative index signals"
    (ok (%signals-dimension-mismatch-p (lambda () (%csr :indices '(0 -1 1))))
        "whether a negative column index was rejected")))

(deftest csr-matrix-rejects-a-non-positive-num-columns
  (testing "NUM-COLUMNS of 0 signals"
    (ok (%signals-dimension-mismatch-p (lambda () (%csr :num-columns 0)))
        "whether a zero NUM-COLUMNS was rejected")))

(deftest csr-matrix-rejects-a-value-it-cannot-coerce
  ;; Coercion happens here, so the failure is reported next to the mistake rather than from
  ;; a `make-dataset' call somewhere else.
  (testing "a non-real value signals unsupported-element-type"
    (ok (handler-case (progn (%csr :values '(1.0d0 "two" 3.0d0)) nil)
          (cl-gbdt:unsupported-element-type () t))
        "whether a non-real VALUES element was rejected")))

(deftest csr-matrix-allows-an-all-zero-trailing-column
  ;; The case NUM-COLUMNS exists for: a matrix three columns wide whose third column is
  ;; entirely empty is legal, and its width is 3 rather than 2.
  (testing "a declared width wider than any index present is accepted"
    (let ((m (%csr :indices '(0 1 1) :num-columns 3)))
      (ok (= 3 (cl-gbdt:csr-matrix-num-columns m))
          "whether the declared width survived having no index reach it"))))

(deftest csr-matrix-allows-an-empty-row
  ;; A repeated INDPTR entry is a row with no stored elements -- legal, and the reason the
  ;; check is non-decreasing rather than strictly increasing.
  (testing "a repeated INDPTR entry is accepted"
    (let ((m (%csr :indptr '(0 2 2 3) :indices '(0 2 1) :values '(1.0d0 2.0d0 3.0d0))))
      (ok (= 3 (cl-gbdt:csr-matrix-num-rows m)) "the row count with one empty row"))))

(deftest csr-matrix-rejects-an-indptr-element-it-cannot-coerce
  ;; A float slips past the numeric comparisons INDPTR's shape checks use (zerop, <, =)
  ;; but cannot become a (signed-byte 32) array element -- caught here, not as a raw
  ;; TYPE-ERROR out of `make-array'.
  (testing "a non-integer INDPTR element signals unsupported-element-type"
    (ok (handler-case (progn (%csr :indptr (list 0 2.5d0 3)) nil)
          (cl-gbdt:unsupported-element-type () t))
        "whether a non-integer INDPTR element was rejected")))

(deftest csr-matrix-rejects-an-indices-element-it-cannot-coerce
  ;; NUM-COLUMNS has no upper bound of its own, so a column index within a very wide
  ;; declared width can still overflow (signed-byte 32), the type INDICES is stored as.
  (testing "an INDICES element outside (signed-byte 32) signals unsupported-element-type"
    (ok (handler-case
            (progn (%csr :indices (list 0 5000000000 1) :num-columns 5000000001) nil)
          (cl-gbdt:unsupported-element-type () t))
        "whether an out-of-range INDICES element was rejected")))
