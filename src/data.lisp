;;;; data.lisp --- Hand matrices to the backends as foreign memory.
;;;;
;;;; On SBCL a 2D simple-array is stored contiguously in row-major order, so it can
;;;; be pinned and passed straight to C. Other implementations fall back to copying.
;;;;
;;;; Both LightGBM and XGBoost ingest the data into their own representation when a
;;;; Dataset or DMatrix is constructed, so pinning is only needed for the duration of
;;;; that construction call.

(in-package #:cl-gbdt)

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
  (let* ((cffi-type (foreign-element-type element-type))
         (size (array-total-size matrix)))
    (cffi:with-foreign-object (buffer cffi-type size)
      (dotimes (index size)
        (setf (cffi:mem-aref buffer cffi-type index)
              (row-major-aref matrix index)))
      (funcall function buffer (array-dimension matrix 0) (array-dimension matrix 1)
               element-type))))

#+sbcl
(defun %call-with-pinned-matrix (matrix element-type function)
  "Pin MATRIX and call FUNCTION with a pointer to its storage.

On SBCL the storage of a 2D simple-array is a contiguous row-major vector, so it can
be handed to C without copying."
  (let ((storage (sb-ext:array-storage-vector matrix)))
    (sb-sys:with-pinned-objects (matrix)
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
