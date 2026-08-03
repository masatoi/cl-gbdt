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
