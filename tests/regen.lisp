;;;; regen.lisp --- Tests for the binding emitter's pure parts.

(uiop:define-package #:cl-gbdt/tests/regen
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/regen/all))

(in-package #:cl-gbdt/tests/regen)

(defun typedef-table (&rest pairs)
  "Build a c2ffi-shaped typedef table from NAME/TYPE-ALIST PAIRS."
  (let ((table (make-hash-table :test #'equal)))
    (loop :for (name type) :on pairs :by #'cddr
          :do (setf (gethash name table) type))
    table))

(defun c-type (tag &optional inner)
  "Build a c2ffi-shaped type object."
  (let ((table (make-hash-table :test #'equal)))
    (setf (gethash "tag" table) tag)
    (when inner (setf (gethash "type" table) inner))
    table))

(deftest lisp-name-produces-idiomatic-symbols
  (testing "underscores and camelCase both become single dashes"
    (ok (string= "LGBM-BOOSTER-CREATE"
                 (symbol-name (cl-gbdt/src/regen/all:lisp-name "LGBM_BoosterCreate"))))
    (ok (string= "LGBM-GET-LAST-ERROR"
                 (symbol-name (cl-gbdt/src/regen/all:lisp-name "LGBM_GetLastError"))))
    (ok (string= "XGD-MATRIX-CREATE-FROM-DENSE"
                 (symbol-name (cl-gbdt/src/regen/all:lisp-name "XGDMatrixCreateFromDense")))))
  (testing "no doubled dashes survive"
    (dolist (name '("LGBM_GetLastError" "LGBM_DatasetCreateFromMat" "XGBGetLastError"))
      (ng (search "--" (symbol-name (cl-gbdt/src/regen/all:lisp-name name)))
          (format nil "~A has no doubled dash" name)))))

(deftest fixed-width-typedefs-map-to-portable-cffi-types
  (testing "the integer typedefs never become :long"
    (let ((typedefs (typedef-table)))
      (ok (eq :int32 (cl-gbdt/src/regen/all:cffi-type (c-type "int32_t") typedefs "t")))
      (ok (eq :int64 (cl-gbdt/src/regen/all:cffi-type (c-type "int64_t") typedefs "t")))
      (ok (eq :uint64 (cl-gbdt/src/regen/all:cffi-type (c-type "uint64_t") typedefs "t")))
      (ok (eq :size (cl-gbdt/src/regen/all:cffi-type (c-type "size_t") typedefs "t"))))))

(deftest pointer-types-map-correctly
  (let ((typedefs (typedef-table)))
    (testing "char* becomes :pointer, not :string"
      ;; c2ffi discards const qualifiers, so a `const char *' input and a
      ;; caller-allocated `char *' output buffer are indistinguishable in the spec
      ;; (spec 5.1). CFFI's :string is an input translation that copies a Lisp
      ;; string into a fresh allocation sized to that string; applied to an output
      ;; buffer it would let C write past the end of that allocation with no error.
      (ok (eq :pointer (cl-gbdt/src/regen/all:cffi-type
                        (c-type ":pointer" (c-type ":char")) typedefs "t"))))
    (testing "every other pointer becomes :pointer"
      (ok (eq :pointer (cl-gbdt/src/regen/all:cffi-type
                        (c-type ":pointer" (c-type ":void")) typedefs "t")))
      (ok (eq :pointer (cl-gbdt/src/regen/all:cffi-type
                        (c-type ":pointer" (c-type ":double")) typedefs "t"))))))

(deftest project-typedefs-resolve-through-the-table
  (testing "an opaque handle resolves to :pointer"
    (let ((typedefs (typedef-table "DatasetHandle" (c-type ":pointer" (c-type ":void")))))
      (ok (eq :pointer (cl-gbdt/src/regen/all:cffi-type (c-type "DatasetHandle") typedefs "t")))))
  (testing "a typedef chain resolves transitively"
    (let ((typedefs (typedef-table "bst_ulong" (c-type "uint64_t"))))
      (ok (eq :uint64 (cl-gbdt/src/regen/all:cffi-type (c-type "bst_ulong") typedefs "t"))))))

(deftest unmapped-types-are-an-error-not-a-silent-omission
  (testing "an unknown type tag signals unmapped-type naming the tag and the function"
    (let ((typedefs (typedef-table)))
      (ok (handler-case
              (progn (cl-gbdt/src/regen/all:cffi-type (c-type ":long-double") typedefs
                                              "SomeFunction")
                     nil)
            (cl-gbdt/src/regen/all:unmapped-type (c)
              (and (string= ":long-double" (cl-gbdt/src/regen/all:unmapped-type-tag c))
                   (string= "SomeFunction" (cl-gbdt/src/regen/all:unmapped-type-context c)))))))))

(deftest non-integer-constants-are-reported-not-dropped
  (testing "emit-constant refuses a non-integer macro and says so"
    (let ((entry (make-hash-table :test #'equal)))
      (setf (gethash "tag" entry) "const"
            (gethash "name" entry) "SOME_STRING_MACRO"
            (gethash "value" entry) "not an integer")
      (with-output-to-string (out)
        (ng (cl-gbdt/src/regen/emit::emit-constant entry out)
            "a non-integer macro emits nothing and returns false"))))
  (testing "an integer macro is emitted"
    (let ((entry (make-hash-table :test #'equal)))
      (setf (gethash "tag" entry) "const"
            (gethash "name" entry) "C_API_DTYPE_FLOAT64"
            (gethash "value" entry) 1)
      (let ((text (with-output-to-string (out)
                    (ok (cl-gbdt/src/regen/emit::emit-constant entry out)))))
        (ok (search "+c-api-dtype-float64+" text))
        (ok (search " 1)" text))))))

(deftest type-map-never-emits-architecture-dependent-types
  (testing "no entry in either map is :long, :unsigned-long, :llong or :ullong"
    (dolist (entry (append cl-gbdt/src/regen/all:+typedef-map+ cl-gbdt/src/regen/all:+builtin-map+))
      (ng (member (cdr entry) '(:long :unsigned-long :llong :ullong :long-long
                                :unsigned-long-long))
          (format nil "~A maps to a portable type" (car entry))))))
