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

(defparameter +regen-targets+
  '(("ffi-spec/lightgbm/include/" "LightGBM/c_api.h" "cl-gbdt.lightgbm.ffi"
     ("LGBM_" "C_API_") "src/lightgbm/c-api.lisp")
    ("ffi-spec/xgboost/include/" "xgboost/c_api.h" "cl-gbdt.xgboost.ffi"
     ("XGB" "XGD") "src/xgboost/c-api.lisp"))
  "(spec-root header-name package prefixes committed-output) per backend, mirroring
tools/regen.lisp's own settings. Used to check the committed bindings against the
committed spec without running c2ffi.")

(defun find-committed-spec (spec-root)
  "Return the single c2ffi spec file under SPEC-ROOT, or nil when none is committed.

Globs ffi-spec/<backend>/include/**/c_api.*.spec instead of hardcoding an
architecture, since the spec's filename embeds whatever architecture last
regenerated it (spec 5.2: local architecture only)."
  (first (directory (merge-pathnames
                     (make-pathname :directory '(:relative :wild-inferiors)
                                    :name :wild :type "spec")
                     spec-root))))

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
                    (format nil "~A contains no ~A" path forbidden)))))
  (testing "no :string translation on any pointer"
    ;; c2ffi discards const qualifiers, so a `const char *' input and a
    ;; caller-allocated `char *' output buffer are indistinguishable in the spec
    ;; (design doc 5.1). CFFI's :string translation copies a Lisp string into a
    ;; fresh foreign allocation sized to that string; applied to an output buffer,
    ;; the C call would write past the end of that allocation with no error. If
    ;; this test starts failing, something reintroduced the special case in
    ;; `cffi-type' rather than mapping every pointer to :pointer uniformly.
    (loop :for (path) :in +generated-bindings+
          :for text := (binding-source path)
          :do (ng (search ":string" text)
                  (format nil "~A contains no :string" path)))))

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

(deftest committed-bindings-match-their-committed-spec
  (testing "re-emitting from the committed c2ffi spec reproduces the committed file byte-for-byte"
    ;; Catches a hand-edited c-api.lisp, or a bumped ffi-spec/VERSIONS with a
    ;; forgotten regeneration. Needs no Docker and no network: this re-runs the
    ;; pure emitter over the spec that is already checked in.
    (dolist (target +regen-targets+)
      (destructuring-bind (spec-root header-name package prefixes committed) target
        (let ((spec (find-committed-spec (asdf:system-relative-pathname "cl-gbdt" spec-root))))
          (if (null spec)
              (skip (format nil "no c2ffi spec committed under ~A; nothing to check ~A against"
                            spec-root committed))
              (uiop:with-temporary-file (:pathname temp :type "lisp")
                (cl-gbdt/src/regen/all:emit-bindings spec header-name package prefixes temp)
                (ok (string= (binding-source committed) (uiop:read-file-string temp))
                    (format nil "~A matches its committed spec" committed)))))))))
