;;;; package.lisp --- Package holding the generated LightGBM bindings.

(defpackage #:cl-gbdt.lightgbm.ffi
  (:use #:cl)
  (:documentation "Raw CFFI bindings for the LightGBM C API.

Every symbol here is generated from the vendored header by tools/regen.lisp.
Nothing outside cl-gbdt/lightgbm should call these directly."))
