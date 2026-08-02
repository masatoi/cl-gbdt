;;;; types.lisp --- Mapping C types onto portable CFFI types.
;;;;
;;;; Every C type the headers use is written to a CFFI built-in here. Nothing is
;;;; emitted whose width depends on the machine that ran the generator, so the
;;;; generated bindings are architecture-independent by construction rather than
;;;; by comparison against other architectures.

(in-package #:cl-gbdt.regen)

(defparameter +typedef-map+
  '(("int8_t" . :int8) ("uint8_t" . :uint8)
    ("int16_t" . :int16) ("uint16_t" . :uint16)
    ("int32_t" . :int32) ("uint32_t" . :uint32)
    ("int64_t" . :int64) ("uint64_t" . :uint64)
    ("size_t" . :size) ("ssize_t" . :ssize)
    ("ptrdiff_t" . :ptrdiff) ("intptr_t" . :intptr) ("uintptr_t" . :uintptr))
  "Fixed-width C typedefs and their portable CFFI equivalents.

Mapping these directly is what lets the generator ignore /usr/include entirely.
Emitting the C typedef instead would produce `:long' for `int64_t' on LP64, making
the output architecture-dependent.")

(defparameter +builtin-map+
  '((":void" . :void) (":char" . :char) (":signed-char" . :char)
    (":unsigned-char" . :unsigned-char)
    (":short" . :short) (":unsigned-short" . :unsigned-short)
    (":int" . :int) (":unsigned-int" . :unsigned-int)
    (":float" . :float) (":double" . :double) (":_Bool" . :bool)
    (":function-pointer" . :pointer) (":struct" . :pointer) (":union" . :pointer))
  "C basic types and their CFFI equivalents.

Structs and unions map to `:pointer' because this API only ever passes them by
address, and every function that does is on the ABI blacklist.")

(define-condition unmapped-type (error)
  ((tag :initarg :tag :reader unmapped-type-tag)
   (context :initarg :context :reader unmapped-type-context))
  (:report (lambda (condition stream)
             (format stream "No CFFI mapping for the C type ~S, used by ~A."
                     (unmapped-type-tag condition)
                     (unmapped-type-context condition))))
  (:documentation "A C type appeared that the maps do not cover.

Signalled rather than skipped: a silently omitted function is how a binding set
ends up quietly incomplete."))

(defun cffi-type (type typedefs context)
  "Return the CFFI type for the c2ffi TYPE object.

TYPEDEFS maps typedef names to their underlying c2ffi type objects. CONTEXT names
whatever is being emitted, for the error message. Signals `unmapped-type' when no
mapping applies."
  (let ((tag (gethash "tag" type)))
    (cond
      ;; Every pointer maps to :pointer, char* included. See spec 5.1: c2ffi
      ;; discards const qualifiers, so a `const char *' input and a caller-allocated
      ;; `char *' output buffer are indistinguishable in the spec, and CFFI's
      ;; :string translation applied to an output buffer would let C write past the
      ;; end of an allocation sized to copy a Lisp string in, not to receive one.
      ((string= tag ":pointer") :pointer)
      ((cdr (assoc tag +typedef-map+ :test #'string=)))
      ((cdr (assoc tag +builtin-map+ :test #'string=)))
      ((gethash tag typedefs)
       (cffi-type (gethash tag typedefs) typedefs context))
      (t (error 'unmapped-type :tag tag :context context)))))
