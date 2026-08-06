;;;; support.lisp --- Library discovery and shared fixtures for the functional tests.

(uiop:define-package #:cl-gbdt/tests/functional/support
  (:use #:cl #:rove)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/xgboost/array-interface
                #:array-interface-json)
  (:export #:backend-library-path
           #:ensure-backend-library
           #:with-backend-library
           #:resolve-via-cffi-default
           #:make-separable-dataset
           #:predictions-separate-p
           #:within-group-strictly-increasing-p
           #:make-multiclass-dataset
           #:predictions-match-labels-p
           #:array-interface-json))

(in-package #:cl-gbdt/tests/functional/support)

(defparameter *backend-libraries*
  '((:lightgbm "CL_GBDT_LIGHTGBM_LIB" "vendor/lightgbm/lib/" "lib_lightgbm.*")
    (:xgboost "CL_GBDT_XGBOOST_LIB" "vendor/xgboost/lib/" "libxgboost.*"))
  "For each backend: the environment variable that overrides discovery, the
repository-relative directory ./tools/fetch-libs.sh writes to, and a basename pattern to
find its library within that directory. The pattern's extension is wild because
./tools/fetch-libs.sh preserves whatever the platform's wheel ships -- `.so' on Linux,
`.dylib' on macOS -- and discovery must not assume either.")

(defvar *loaded-libraries* (make-hash-table :test #'eq)
  "Backends whose shared library this image has already loaded.")

(defun backend-library-path (backend)
  "Return the path to BACKEND's shared library, or NIL when it is not present.

The environment variable wins over the vendored directory, so a developer can point the
suite at a system-wide install without moving files around. It is honoured strictly: a
variable that names a path which does not exist is an error, not a fall-through. Falling
through would run the tests against a different library than the developer asked for, and
then report -- wrongly -- that the variable was unset.

Absent an override, the vendored directory is searched for the basename pattern with
`directory'. More than one match is an error rather than an arbitrary pick, since a
silent choice between candidates could run the tests against the wrong file."
  (destructuring-bind (variable directory pattern) (cdr (assoc backend *backend-libraries*))
    (let ((override (uiop:getenv variable)))
      (if (and override (plusp (length override)))
          (or (probe-file override)
              (error "~A is set to ~S, which does not exist." variable override))
          (let ((matches (directory (merge-pathnames
                                      pattern
                                      (asdf:system-relative-pathname "cl-gbdt" directory)))))
            (case (length matches)
              (0 nil)
              (1 (first matches))
              (t (error "~D files match ~A under ~A, expected at most one: ~{~A~^, ~}."
                        (length matches) pattern directory matches))))))))

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
  (destructuring-bind (variable directory pattern) (cdr (assoc backend *backend-libraries*))
    (format nil "No file matching ~A found under ~A and ~A is unset. ~
                 Run ./tools/fetch-libs.sh first."
            pattern directory variable)))

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

(defun resolve-via-cffi-default (backend default-name)
  "Return the pathname CFFI's `:default' designator resolves DEFAULT-NAME to, with
BACKEND's own vendored directory pushed onto `cffi:*foreign-library-directories*', or
NIL when it fails to resolve there. Callers must already know BACKEND's vendored
library exists -- e.g. by running inside `with-backend-library' -- this does not check.

This exercises exactly the CFFI call `resolve-and-load-library' falls back to when no
explicit :path, env-var override, or vendored-directory match is found -- a branch
`with-backend-library''s own round trip never reaches, since the vendored directory is
always found first there. That fallback is a system-search, not a directory lookup, so
BACKEND's vendored directory has to be pushed onto CFFI's own
`*foreign-library-directories*' for `:default' to have anything to find.

The library CFFI opens here is deliberately never closed: this project's functional
suite already leaves every library it loads open for the rest of the process --
`ensure-backend-library' above never closes either -- rather than risk a premature
`cffi:close-foreign-library' unmapping symbols something else in the same run still
depends on."
  (let* ((directory (uiop:pathname-directory-pathname (backend-library-path backend)))
         (cffi:*foreign-library-directories*
           (cons directory cffi:*foreign-library-directories*)))
    (handler-case
        (cffi:foreign-library-pathname (cffi:load-foreign-library (list :default default-name)))
      (error () nil))))

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

(defun within-group-strictly-increasing-p (predictions group-sizes)
  "True when PREDICTIONS -- a list of per-row scores, in row order -- increases strictly
within each consecutive run of GROUP-SIZES, and false otherwise. GROUP-SIZES must sum to
PREDICTIONS' length; a caller error there signals from `subseq'/`nthcdr' rather than
returning a silently wrong answer.

This is the ordering property a ranking round trip can check without depending on raw
score values, which would break on any upstream version bump without telling us anything
new -- the same reasoning `predictions-separate-p' above and `predictions-match-labels-p'
below apply to a classification round trip's label boundary and multiclass argmax,
respectively."
  (loop :with remaining := predictions
        :for size :in group-sizes
        :always (prog1 (apply #'< (subseq remaining 0 size))
                  (setf remaining (nthcdr size remaining)))))

(defun make-multiclass-dataset (&key (rows-per-class 3) (num-classes 3) (cols 3))
  "Return two values: a feature matrix and its labels, for a trivially separable
NUM-CLASSES-way classification problem.

The matrix is a `(simple-array double-float (ROWS COLS))', ROWS = ROWS-PER-CLASS *
NUM-CLASSES, grouped by class: row I belongs to class `(floor I ROWS-PER-CLASS)', and
every element of that row is `(class * 10) + offset + column', where OFFSET is the
row's position within its class -- large enough a step between classes that no bin a
backend chooses can straddle two of them, while rows within a class still differ so
they are not literally identical inputs. The labels are a `(simple-array single-float
(ROWS))', 0.0 through `(NUM-CLASSES - 1).0', grouped the same way.

The defaults produce a nine-row, three-class fixture -- verified against XGBoost's
`multi:softprob' objective to make `predict' return shape (9 3).

The problem is deliberately trivial, in the same spirit as `make-separable-dataset':
these tests check that data and per-class predictions cross the FFI boundary intact,
not that the libraries learn anything hard."
  (let* ((rows (* rows-per-class num-classes))
         (matrix (make-array (list rows cols) :element-type 'double-float))
         (label-vector (make-array rows :element-type 'single-float)))
    (dotimes (row rows)
      (let ((class (floor row rows-per-class))
            (offset (mod row rows-per-class)))
        (dotimes (col cols)
          (setf (aref matrix row col) (coerce (+ (* class 10) offset col) 'double-float)))
        (setf (aref label-vector row) (coerce class 'single-float))))
    (values matrix label-vector)))

(defun predictions-match-labels-p (predictions label-vector)
  "True when every row of PREDICTIONS, a 2D array of per-class scores as `predict'
returns for a multiclass booster, has its largest value in the column LABEL-VECTOR
names for that row.

This is `make-multiclass-dataset''s analogue of `predictions-separate-p': the ordering
property a multiclass round trip can check without depending on exact probabilities,
which would break on any upstream version bump without telling us anything new."
  (let ((rows (array-dimension predictions 0))
        (cols (array-dimension predictions 1)))
    (loop :for row :below rows
          :always (let ((label (round (aref label-vector row))))
                    (loop :with best-column := 0
                          :with best-value := (aref predictions row 0)
                          :for column :from 1 :below cols
                          :when (> (aref predictions row column) best-value)
                            :do (setf best-value (aref predictions row column)
                                      best-column column)
                          :finally (return (= best-column label)))))))
