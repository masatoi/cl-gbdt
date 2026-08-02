;;;; bindings.lisp --- Tests over the generated CFFI bindings.
;;;;
;;;; These need no shared library: they check the generated source, and that the
;;;; definitions load without opening anything (layer 1).

(in-package #:cl-gbdt/tests)

(defparameter +generated-bindings+
  '(("src/lightgbm/c-api.lisp" :cl-gbdt.lightgbm.ffi 90
     ("LGBM_BoosterCreate" "LGBM_DatasetCreateFromMat" "LGBM_GetLastError"))
    ("src/xgboost/c-api.lisp" :cl-gbdt.xgboost.ffi 70
     ("XGBoosterCreate" "XGDMatrixCreateFromDense" "XGBGetLastError")))
  "(path package minimum-functions required-c-names) for each generated file.")

(defun binding-source (relative)
  (uiop:read-file-string (asdf:system-relative-pathname "cl-gbdt" relative)))

(deftest generated-bindings-are-architecture-independent
  (testing "no aggregate types, whose layout would depend on the target"
    (loop :for (path) :in +generated-bindings+
          :for text := (binding-source path)
          :do (dolist (forbidden '("defcstruct" "defcunion" "defcenum"))
                (ng (search forbidden text)
                    (format nil "~A contains no ~A" path forbidden)))))
  (testing "no types whose width depends on the data model"
    (loop :for (path) :in +generated-bindings+
          :for text := (binding-source path)
          :do (dolist (forbidden '(":long)" ":unsigned-long)" ":llong)" ":ullong)"))
                (ng (search forbidden text)
                    (format nil "~A contains no ~A" path forbidden))))))

(deftest generated-bindings-are-complete
  (loop :for (path package minimum required) :in +generated-bindings+
        :for text := (binding-source path)
        :do (testing (format nil "~A has at least ~D functions" path minimum)
              (let ((count (with-input-from-string (in text)
                             (loop :for line := (read-line in nil)
                                   :while line
                                   :count (search "cffi:defcfun" line)))))
                (ok (>= count minimum)
                    (format nil "~A defines ~D functions" path count))))
            (testing (format nil "~A defines the functions the wrapper depends on" path)
              (dolist (name required)
                (ok (search (format nil "\"~A\"" name) text)
                    (format nil "~A is present" name))))))

(deftest generated-bindings-define-callable-symbols
  (testing "the emitted symbols exist and are fbound after loading"
    (dolist (spec '((:cl-gbdt.lightgbm.ffi "LGBM-DATASET-CREATE-FROM-MAT")
                    (:cl-gbdt.lightgbm.ffi "LGBM-BOOSTER-CREATE")
                    (:cl-gbdt.xgboost.ffi  "XGD-MATRIX-CREATE-FROM-DENSE")
                    (:cl-gbdt.xgboost.ffi  "XGB-GET-LAST-ERROR")))
      (destructuring-bind (package name) spec
        (let ((symbol (find-symbol name (find-package package))))
          (ok symbol (format nil "~A::~A exists" package name))
          (when symbol
            (ok (fboundp symbol) (format nil "~A::~A is fbound" package name))))))))

(deftest lightgbm-constants-are-generated
  (testing "the C_API_ constants used by the wrapper are present"
    (dolist (name '("+C-API-DTYPE-FLOAT64+" "+C-API-PREDICT-NORMAL+"))
      (ok (find-symbol name (find-package :cl-gbdt.lightgbm.ffi))
          (format nil "~A is defined" name)))))
