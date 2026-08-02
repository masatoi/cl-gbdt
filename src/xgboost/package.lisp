;;;; package.lisp --- Package holding the generated XGBoost bindings.

(defpackage #:cl-gbdt.xgboost.ffi
  (:use #:cl)
  (:documentation "Raw CFFI bindings for the XGBoost C API.

Every symbol here is generated from the vendored header by tools/regen.lisp.
Nothing outside cl-gbdt/xgboost should call these directly."))
