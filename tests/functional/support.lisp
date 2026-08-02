;;;; support.lisp --- Library discovery and shared fixtures for the functional tests.

(in-package #:cl-gbdt/functional-tests)

(defparameter *backend-libraries*
  '((:lightgbm "CL_GBDT_LIGHTGBM_LIB" "vendor/lightgbm/lib/lib_lightgbm.so")
    (:xgboost "CL_GBDT_XGBOOST_LIB" "vendor/xgboost/lib/libxgboost.so"))
  "For each backend: the environment variable that overrides discovery, and the
repository-relative path ./tools/fetch-libs.sh writes to.")

(defvar *loaded-libraries* (make-hash-table :test #'eq)
  "Backends whose shared library this image has already loaded.")

(defun backend-library-path (backend)
  "Return the path to BACKEND's shared library, or NIL when it is not present.

The environment variable wins over the vendored copy, so a developer can point the
suite at a system-wide install without moving files around. It is honoured strictly:
a variable that names a path which does not exist is an error, not a fall-through.
Falling through would run the tests against a different library than the developer
asked for, and then report -- wrongly -- that the variable was unset."
  (destructuring-bind (variable relative) (cdr (assoc backend *backend-libraries*))
    (let ((override (uiop:getenv variable)))
      (if (and override (plusp (length override)))
          (or (probe-file override)
              (error "~A is set to ~S, which does not exist." variable override))
          (probe-file (asdf:system-relative-pathname "cl-gbdt" relative))))))

(defun ensure-backend-library (backend)
  "Load BACKEND's shared library if it has not been loaded yet.

Returns the path that was loaded, or NIL when the library is unavailable. Loading is
memoised because `cffi:load-foreign-library' is not free and the tests call this once
per test."
  (or (gethash backend *loaded-libraries*)
      (let ((path (backend-library-path backend)))
        (when path
          (cffi:load-foreign-library path)
          (setf (gethash backend *loaded-libraries*) path)))))

(defun missing-library-message (backend)
  "Return the skip message naming what is missing and how to obtain it."
  (destructuring-bind (variable relative) (cdr (assoc backend *backend-libraries*))
    (format nil "~A not found under ~A and ~A is unset. Run ./tools/fetch-libs.sh first."
            (file-namestring relative) (directory-namestring relative) variable)))

(defmacro with-backend-library ((backend) &body body)
  "Evaluate BODY with BACKEND's shared library loaded, or skip when it is absent.

A missing library is a skip rather than a failure, because `vendor/' is git-ignored and
a fresh clone legitimately has none. The skip message names the script that fixes it,
so the result is never silent.

Note for callers: `rove:skip' records a pending assertion and returns normally rather
than unwinding, so wrap the whole test body in this macro. Anything placed after the
macro in the same `deftest' would still run with no library loaded."
  (let ((path (gensym "PATH")))
    `(let ((,path (ensure-backend-library ,backend)))
       (if (null ,path)
           (skip (missing-library-message ,backend))
           (progn ,@body)))))

(defun make-separable-dataset (&key (rows 8) (cols 3))
  "Return two values: a feature matrix and its labels, for a trivially separable problem.

The matrix is a `(simple-array double-float (ROWS COLS))' whose element [i][j] is
(i + j)/10. The labels are a `(simple-array single-float (ROWS))', 1.0 where column 0
exceeds 0.35 and 0.0 otherwise, which splits the default eight rows into four and four.

The problem is deliberately trivial: these tests check that data crosses the FFI
boundary intact, not that the libraries learn anything hard."
  (let ((matrix (make-array (list rows cols) :element-type 'double-float))
        (label-vector (make-array rows :element-type 'single-float)))
    (dotimes (i rows)
      (dotimes (j cols)
        (setf (aref matrix i j) (coerce (/ (+ i j) 10) 'double-float)))
      (setf (aref label-vector i) (if (> (aref matrix i 0) 0.35d0) 1.0 0.0)))
    (when (or (every #'zerop label-vector) (notany #'zerop label-vector))
      (error "ROWS = ~D yields only one label class; the fixture needs both." rows))
    (values matrix label-vector)))

(defun predictions-separate-p (predictions label-vector)
  "True when every positive-label prediction exceeds every negative-label one.

This is the ordering property the round trips assert instead of exact values. Exact
values would break on any upstream version bump without telling us anything new,
whereas separation can only hold if the matrix arrived row-major with the right
dimensions, the labels bound to the right rows, and the predictions came back in the
right order.

Signals an error when either class is empty, guarding against a caller passing a
single-class dataset -- for example a fixture whose ROWS is too small for
`make-separable-dataset' to split into both labels. `reduce' signals on an empty
list, so without this check that degenerate input would raise a bare CL error deep
inside `reduce' rather than one naming the actual problem; a clear error here beats
letting an empty selection produce a confident but meaningless answer."
  (let ((negatives (loop :for prediction :across predictions
                         :for label :across label-vector
                         :when (zerop label) :collect prediction))
        (positives (loop :for prediction :across predictions
                         :for label :across label-vector
                         :when (plusp label) :collect prediction)))
    (when (or (null negatives) (null positives))
      (error "Cannot judge separation: ~D negative and ~D positive labels; both classes ~
              must be present."
             (length negatives) (length positives)))
    (< (reduce #'max negatives) (reduce #'min positives))))

(defun array-interface-json (pointer typestr &rest shape)
  "Return a NumPy array-interface descriptor for POINTER as a JSON string.

XGBoost's current entry points take the buffer this way rather than as a bare pointer.
TYPESTR is a NumPy type code -- \"<f8\" for double-float, \"<f4\" for single-float. The
descriptor's shape is fixed, so no JSON library is needed to emit it.

The buffer must stay pinned and alive for as long as XGBoost reads it, which for
`XGDMatrixCreateFromDense' is the duration of that call."
  (format nil "{\"data\":[~D,false],\"typestr\":\"~A\",\"shape\":[~{~D~^,~}],\"version\":3}"
          (cffi:pointer-address pointer) typestr shape))
