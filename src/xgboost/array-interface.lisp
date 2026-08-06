;;;; array-interface.lisp --- NumPy array-interface JSON descriptors for XGBoost.
;;;;
;;;; XGBoost's current data-ingestion entry points (`XGDMatrixCreateFromDense',
;;;; `XGDMatrixSetInfoFromInterface') take a buffer wrapped in this descriptor rather than a
;;;; bare pointer and dimensions the way LightGBM's C API does.

(uiop:define-package #:cl-gbdt/src/xgboost/array-interface
  (:use #:cl)
  (:import-from #:cffi)
  (:export #:array-interface-json))

(in-package #:cl-gbdt/src/xgboost/array-interface)

(defun array-interface-json (pointer typestr &rest shape)
  "Return a NumPy array-interface descriptor for POINTER as a JSON string.

XGBoost's current entry points take the buffer this way rather than as a bare pointer.
TYPESTR is a NumPy type code -- \"<f8\" for double-float, \"<f4\" for single-float. The
descriptor's shape is fixed, so no JSON library is needed to emit it.

The buffer must stay pinned and alive for as long as XGBoost reads it, which for
`XGDMatrixCreateFromDense' is the duration of that call."
  (format nil "{\"data\":[~D,false],\"typestr\":\"~A\",\"shape\":[~{~D~^,~}],\"version\":3}"
          (cffi:pointer-address pointer) typestr shape))
