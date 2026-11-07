(in-package #:cl-user)

;;;; Binary input and score output shared by the example trainers.
;;;; Each example's prepare.lisp writes its matrices as raw little-endian float32 (and,
;;;; for CSR indices, int32) with common.lisp's WRITE-F32 and WRITE-I32. The trainers
;;;; read them straight into Lisp arrays through C stdio. SBCL and CFFI are required.

(load (merge-pathnames "common.lisp" *load-truename*))

(unless (find-package "CFFI")
  (error "Load CFFI before examples/train-io.lisp."))

(in-package #:cl-gbdt-examples)

(export '(load-f32 load-i32 write-scores) '#:cl-gbdt-examples)

(unless (member :little-endian *features*)
  (error "The example trainers read little-endian binary data into native arrays."))

(cffi:defcfun ("fopen" %fopen) :pointer (name :string) (mode :string))

(cffi:defcfun ("fread" %fread) :size
  (buffer :pointer) (size :size) (count :size) (stream :pointer))

(cffi:defcfun ("fgetc" %fgetc) :int (stream :pointer))

(cffi:defcfun ("fclose" %fclose) :int (stream :pointer))

(defun read-binary-array (path dimensions element-type noun)
  "Return a fresh array of DIMENSIONS and ELEMENT-TYPE filled from PATH's 4-byte elements.
PATH must hold exactly as many elements as the array; NOUN names them in error messages."
  (let* ((array (make-array dimensions :element-type element-type))
         (storage (sb-ext:array-storage-vector array))
         (expected (array-total-size array))
         ;; fopen needs the OS path: NAMESTRING escapes *, ?, [ and backslash.
         (file (%fopen (sb-ext:native-namestring (merge-pathnames path) :as-file t) "rb")))
    (when (cffi:null-pointer-p file) (error "Cannot open ~A" path))
    (unwind-protect
         (sb-sys:with-pinned-objects (array storage)
           (let ((count (%fread (sb-sys:vector-sap storage) 4 expected file)))
             (unless (= count expected)
               (error "Short read from ~A: ~D/~D ~A" path count expected noun))
             (unless (= (%fgetc file) -1)
               (error "~A holds more than ~D ~A" path expected noun))))
      (%fclose file))
    array))

(defun load-f32 (path dimensions)
  "Return a fresh single-float array of DIMENSIONS read from the float32 file PATH.
PATH holds the array's elements in row-major order as little-endian IEEE-754 single
floats, as common.lisp's WRITE-F32 writes them, and nothing else. DIMENSIONS is a list
of integers, or an integer for a vector. Signals an error when PATH cannot be opened or
holds fewer or more than (reduce #'* DIMENSIONS) floats."
  (read-binary-array path dimensions 'single-float "floats"))

(defun load-i32 (path length)
  "Return a fresh (signed-byte 32) vector of LENGTH read from the int32 file PATH.
PATH holds exactly LENGTH little-endian two's-complement integers, as common.lisp's
WRITE-I32 writes them. Signals an error when PATH cannot be opened or holds fewer or
more than LENGTH integers."
  (read-binary-array path length '(signed-byte 32) "integers"))

(defun write-scores (path predictions &key rows)
  "Write the one-column prediction matrix PREDICTIONS to PATH and return PATH.
Each row's score goes on its own line in ~,15F format; an existing file is superseded.
Signals an error, before opening PATH, unless PREDICTIONS has rank 2 and one column
and, when ROWS is given, exactly ROWS rows."
  (unless (and (= (array-rank predictions) 2)
               (= (array-dimension predictions 1) 1)
               (or (null rows) (= rows (array-dimension predictions 0))))
    (error "Unexpected prediction dimensions ~S" (array-dimensions predictions)))
  (with-open-file (out path :direction :output :if-exists :supersede)
    (dotimes (i (array-dimension predictions 0))
      (format out "~,15F~%" (aref predictions i 0))))
  path)
