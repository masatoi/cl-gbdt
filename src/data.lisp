;;;; data.lisp --- Hand matrices to the backends as foreign memory.
;;;;
;;;; On SBCL a 2D simple-array is stored contiguously in row-major order, so it can
;;;; be pinned and passed straight to C. Other implementations fall back to copying.
;;;;
;;;; Both LightGBM and XGBoost ingest the data into their own representation when a
;;;; Dataset or DMatrix is constructed, so pinning is only needed for the duration of
;;;; that construction call.

(uiop:define-package #:cl-gbdt/src/data
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/conditions
                #:dimension-mismatch
                #:unsupported-element-type)
  (:export #:foreign-matrix
           #:foreign-matrix-pointer
           #:foreign-matrix-rows
           #:foreign-matrix-cols
           #:foreign-matrix-element-type
           #:call-with-foreign-matrix
           #:with-foreign-matrix
           #:foreign-element-type
           #:write-foreign-sequence
           #:csr-matrix
           #:make-csr-matrix
           #:csr-matrix-indptr
           #:csr-matrix-indices
           #:csr-matrix-values
           #:csr-matrix-num-columns
           #:csr-matrix-num-rows))

(in-package #:cl-gbdt/src/data)

(defclass foreign-matrix ()
  ((pointer :initarg :pointer
            :reader foreign-matrix-pointer
            :documentation "CFFI pointer to the first element.")
   (rows :initarg :rows
         :reader foreign-matrix-rows
         :documentation "Number of rows.")
   (cols :initarg :cols
         :reader foreign-matrix-cols
         :documentation "Number of columns.")
   (element-type :initarg :element-type
                 :reader foreign-matrix-element-type
                 :documentation "Element type, `double-float' or `single-float'."))
  (:documentation "A matrix already laid out row-major in foreign memory.

Use this to pass data without going through a Lisp array. Ownership stays with the
caller; cl-gbdt never frees it."))

(defun foreign-element-type (element-type)
  "Return the CFFI type keyword corresponding to ELEMENT-TYPE.

Signals `unsupported-element-type' for anything else."
  (cond ((eq element-type 'double-float) :double)
        ((eq element-type 'single-float) :float)
        (t (error 'unsupported-element-type :given element-type))))

(defun write-foreign-sequence (pointer cffi-type sequence coercer)
  "Copy SEQUENCE into the foreign array at POINTER, each element passed through
COERCER before being stored as CFFI-TYPE."
  (let ((vector (coerce sequence 'vector)))
    (dotimes (index (length vector))
      (setf (cffi:mem-aref pointer cffi-type index) (funcall coercer (aref vector index))))))

(defun %normalized-element-type (array)
  "Return ARRAY's element type normalized to `double-float' or `single-float'.

Implementations may upgrade element types differently, so the check uses `subtypep'
in both directions rather than comparing symbols."
  (let ((element-type (array-element-type array)))
    (cond ((and (subtypep element-type 'double-float)
                (subtypep 'double-float element-type))
           'double-float)
          ((and (subtypep element-type 'single-float)
                (subtypep 'single-float element-type))
           'single-float)
          (t (error 'unsupported-element-type :given element-type)))))

(defgeneric call-with-foreign-matrix (matrix function)
  (:documentation "Make MATRIX available as foreign memory and call FUNCTION.

FUNCTION is called with four arguments: (POINTER NROW NCOL ELEMENT-TYPE). POINTER
addresses the elements laid out row-major. ELEMENT-TYPE is the symbol `double-float'
or `single-float'.

POINTER is valid only for the duration of FUNCTION and must not escape it. Add a
method here to support a new input format."))

(defmethod call-with-foreign-matrix ((matrix foreign-matrix) function)
  (funcall function
           (foreign-matrix-pointer matrix)
           (foreign-matrix-rows matrix)
           (foreign-matrix-cols matrix)
           (foreign-matrix-element-type matrix)))

(defun %call-with-copied-matrix (matrix element-type function)
  "Copy MATRIX into a foreign buffer and call FUNCTION.

Fallback path used when pinning is unavailable."
  (let ((cffi-type (foreign-element-type element-type))
        (size (array-total-size matrix)))
    (cffi:with-foreign-object (buffer cffi-type size)
      (dotimes (index size)
        (setf (cffi:mem-aref buffer cffi-type index)
              (row-major-aref matrix index)))
      (funcall function buffer (array-dimension matrix 0) (array-dimension matrix 1)
               element-type))))

#+sbcl
(defun %call-with-pinned-matrix (matrix element-type function)
  "Pin MATRIX's storage and call FUNCTION with a pointer to it.

On SBCL the storage of a 2D simple-array is a contiguous row-major vector, so it can
be handed to C without copying.

STORAGE must be pinned, not just MATRIX. An array of rank other than one is a header
object holding a reference to a separately allocated data vector, so MATRIX and
STORAGE are two distinct objects at two distinct addresses.
`sb-sys:with-pinned-objects' pins exactly the objects it is given and does not follow
references, so pinning only MATRIX would leave the garbage collector free to relocate
STORAGE while C holds a pointer into it -- a use-after-move that surfaces only under
collection pressure. MATRIX is pinned as well because it costs nothing and keeps the
pairing obvious."
  (let ((storage (sb-ext:array-storage-vector matrix)))
    (sb-sys:with-pinned-objects (matrix storage)
      (funcall function
               (cffi:make-pointer (sb-sys:sap-int (sb-sys:vector-sap storage)))
               (array-dimension matrix 0)
               (array-dimension matrix 1)
               element-type))))

(defmethod call-with-foreign-matrix ((matrix array) function)
  (unless (= 2 (array-rank matrix))
    (error 'dimension-mismatch :expected "a 2D array" :given (array-dimensions matrix)))
  (let ((element-type (%normalized-element-type matrix)))
    #+sbcl
    (if (typep matrix 'simple-array)
        (%call-with-pinned-matrix matrix element-type function)
        (%call-with-copied-matrix matrix element-type function))
    #-sbcl
    (%call-with-copied-matrix matrix element-type function)))

(defmacro with-foreign-matrix ((pointer nrow ncol element-type) matrix &body body)
  "Make MATRIX available as foreign memory and evaluate BODY.

POINTER, NROW, NCOL and ELEMENT-TYPE are bound within BODY. POINTER becomes invalid
once BODY returns."
  `(call-with-foreign-matrix
    ,matrix
    (lambda (,pointer ,nrow ,ncol ,element-type)
      (declare (ignorable ,pointer ,nrow ,ncol ,element-type))
      ,@body)))

(defstruct (csr-matrix (:constructor %make-csr-matrix))
  "A sparse matrix in compressed sparse row (CSR) format: for row ROW, the pairs
INDICES[k] and VALUES[k], for INDPTR[ROW] <= k < INDPTR[ROW+1], give that row's non-zero
columns and their values.

NUM-COLUMNS is a required argument to `make-csr-matrix' rather than inferred from the
largest index INDICES actually holds, because a matrix's declared width and its largest
stored index are different facts: the last columns can legitimately hold nothing, and the
stored indices alone cannot distinguish that from a matrix that simply is not that wide.

NUM-ROWS is derived from INDPTR's length (`(1- (length indptr))'), not stored as its own
slot -- INDPTR already fixes the row count, so a separate slot would only be a second copy
of the same fact for validation to keep in sync.

INDPTR and INDICES are `(simple-array (signed-byte 32) (*))'; VALUES is a `(simple-array
double-float (*))'. All three are already coerced to what a backend hands to the C API, so
a backend method only needs to pin them -- see `with-foreign-matrix' in this same file for
the dense equivalent.

Every slot is `:read-only t', so there is no `setf' expander for any of the four
accessors. `make-csr-matrix' is the only way to build one and it validates everything a
backend later relies on; a writable slot would make that validation defeatable after the
fact, and a matrix whose INDPTR had been replaced by a list would reach the C API as a raw
`type-error' from inside the pinning code rather than as one of this file's own
conditions. `foreign-matrix' above is reader-only for the same reason: a `csr-matrix' that
exists is one both backends can be handed.

An entry a row does not store is *absent*, not zero, and the two libraries read absence
differently: LightGBM reads an absent entry as `0.0' (its own `zero_as_missing' is off by
default) while XGBoost reads one as missing, and no config key on either changes that --
it is what CSR means to each library. So a `csr-matrix' that omits entries describes
different data to the two backends and changes trained numbers silently rather than
signalling; store every element, zeros included, when the same matrix has to mean the same
thing on both. See README.markdown's \"An absent entry is not a zero, and the two libraries
disagree about it\" for the measured runs on each."
  (indptr nil :read-only t)
  (indices nil :read-only t)
  ;; Named VALUES for the same reason `training-series' is: it is the word for what the
  ;; slot holds. Shadows `cl:values' inside a `with-slots' over this struct.
  (values nil :read-only t)
  (num-columns nil :read-only t))

(defun csr-matrix-num-rows (matrix)
  "Return MATRIX's row count, one less than the length of its INDPTR array.

See the struct's own docstring for why NUM-ROWS is derived rather than stored."
  (1- (length (csr-matrix-indptr matrix))))

(defun %require-positive-num-columns (num-columns)
  "Signal `dimension-mismatch' unless NUM-COLUMNS is a positive integer."
  (unless (and (integerp num-columns) (plusp num-columns))
    (error 'dimension-mismatch :expected "NUM-COLUMNS to be a positive integer"
           :given num-columns)))

(defun %require-matching-lengths (indices values)
  "Signal `dimension-mismatch' unless INDICES and VALUES have the same length -- one
column index per stored value."
  (unless (= (length indices) (length values))
    (error 'dimension-mismatch :expected "INDICES and VALUES to have the same length"
           :given (list (length indices) (length values)))))

(defun %require-non-decreasing-indptr-from-zero (indptr element-count)
  "Signal `dimension-mismatch' unless INDPTR is a legal CSR row-pointer vector: starts at
0, never decreases, and ends at ELEMENT-COUNT -- the number of (INDEX . VALUE) pairs
INDICES and VALUES actually store.

A repeated INDPTR entry -- an empty row -- is legal, which is why this checks
non-decreasing rather than strictly increasing."
  (when (zerop (length indptr))
    (error 'dimension-mismatch :expected "INDPTR to have at least one element" :given 0))
  (unless (zerop (aref indptr 0))
    (error 'dimension-mismatch :expected "INDPTR to start at 0" :given (aref indptr 0)))
  (loop :for i :from 1 :below (length indptr)
        :do (when (< (aref indptr i) (aref indptr (1- i)))
              (error 'dimension-mismatch
                     :expected "INDPTR to be non-decreasing"
                     :given (list (aref indptr (1- i)) (aref indptr i)))))
  (let ((last (aref indptr (1- (length indptr)))))
    (unless (= last element-count)
      (error 'dimension-mismatch
             :expected (format nil "INDPTR to end at ~D, the number of stored elements"
                                element-count)
             :given last))))

(defun %require-indices-in-range (indices num-columns)
  "Signal `dimension-mismatch' unless every element of INDICES is a column index within
[0, NUM-COLUMNS)."
  (loop :for index :across indices
        :do (unless (and (integerp index) (<= 0 index) (< index num-columns))
              (error 'dimension-mismatch
                     :expected (format nil "a column index in [0, ~D)" num-columns)
                     :given index))))

(defun %require-real-values (values)
  "Signal `unsupported-element-type' unless every element of VALUES is a real number --
what a `double-float' can represent.

Checked here, ahead of coercion, so the failure is reported next to the mistake rather
than as an opaque `coerce' error."
  (loop :for value :across values
        :do (unless (realp value)
              (error 'unsupported-element-type :given (type-of value)))))

(defun %require-int32-elements (vector)
  "Signal `unsupported-element-type' unless every element of VECTOR is representable as
`(signed-byte 32)' -- the integer type `%coerce-index-vector' stores INDPTR and INDICES
elements as.

Rejects a non-integer (e.g. a float that happens to satisfy every numeric comparison
INDPTR's own shape checks use) and an integer outside that 32-bit range alike -- either
would otherwise reach `make-array''s `:element-type' check inside `%coerce-index-vector'
as a raw, undocumented `type-error', not one of this file's own conditions."
  (loop :for value :across vector
        :do (unless (typep value '(signed-byte 32))
              (error 'unsupported-element-type :given (type-of value)))))

(defun %coerce-index-vector (vector)
  "Return VECTOR as a `(simple-array (signed-byte 32) (*))'."
  (make-array (length vector) :element-type '(signed-byte 32) :initial-contents vector))

(defun %coerce-value-vector (vector)
  "Return VECTOR as a `(simple-array double-float (*))', each element passed through
`coerce'."
  (map '(simple-array double-float (*)) (lambda (value) (coerce value 'double-float)) vector))

(defun make-csr-matrix (&key indptr indices values num-columns)
  "Return a `csr-matrix' holding INDPTR, INDICES and VALUES in standard CSR layout, one
NUM-COLUMNS wide.

INDPTR, INDICES and VALUES may each be any sequence -- list or vector -- and come back
already coerced to the specialized arrays `csr-matrix-indptr', `csr-matrix-indices' and
`csr-matrix-values' store; see the struct's own docstring for exactly which types, and
for why NUM-COLUMNS is required rather than inferred.

Signals `dimension-mismatch' when NUM-COLUMNS is not a positive integer; when INDICES
and VALUES have different lengths; when INDPTR does not start at 0, decreases anywhere,
or disagrees with INDICES/VALUES' shared length; or when an element of INDICES falls
outside [0, NUM-COLUMNS). Signals `unsupported-element-type' when an element of VALUES
is not a real number, or when an element of INDPTR or INDICES is not representable as
`(signed-byte 32)' -- the type both are stored as, so neither a non-integer nor an
integer too large for it reaches `%coerce-index-vector' as a raw `type-error'. Every
check runs against INDPTR, INDICES and VALUES before any of the three is coerced, so a
malformed matrix is rejected without paying for the copy a valid one needs."
  (%require-positive-num-columns num-columns)
  (let ((indptr-vector (coerce indptr 'vector))
        (indices-vector (coerce indices 'vector))
        (values-vector (coerce values 'vector)))
    (%require-matching-lengths indices-vector values-vector)
    (%require-int32-elements indptr-vector)
    (%require-non-decreasing-indptr-from-zero indptr-vector (length values-vector))
    (%require-int32-elements indices-vector)
    (%require-indices-in-range indices-vector num-columns)
    (%require-real-values values-vector)
    (%make-csr-matrix :indptr (%coerce-index-vector indptr-vector)
                      :indices (%coerce-index-vector indices-vector)
                      :values (%coerce-value-vector values-vector)
                      :num-columns num-columns)))
