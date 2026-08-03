;;;; regen.lisp --- Regenerate the CFFI bindings from the vendored headers.
;;;;
;;;; Developer-only. Run from the repository root:
;;;;   ros run -- --non-interactive --load tools/regen.lisp
;;;;
;;;; c2ffi exits 0 even when it truncates its output, so this script validates
;;;; what it produced instead of trusting the exit status.

(require :asdf)

(asdf:load-system "cl-gbdt/regen")

;;; cffi/c2ffi is used only here, not by src/regen/ itself, so it is not a
;;; dependency of the cl-gbdt/regen defsystem; this script quickloads it directly.
(ql:quickload "cffi/c2ffi" :silent t)

(setf cffi/c2ffi::*c2ffi-executable*
      (namestring (merge-pathnames "tools/c2ffi.sh" (uiop:getcwd))))

(defparameter *backends*
  '((:lightgbm
     :header "ffi-spec/lightgbm/include/LightGBM/c_api.h"
     :header-name "LightGBM/c_api.h"
     :sys-includes ("ffi-spec/lightgbm/include/")
     :package "cl-gbdt.lightgbm.ffi"
     :prefixes ("LGBM_" "C_API_")
     :output "src/lightgbm/c-api.lisp"
     :minimum-functions 90
     :required ("LGBM_BoosterCreate" "LGBM_DatasetCreateFromMat" "LGBM_GetLastError"))
    (:xgboost
     :header "ffi-spec/xgboost/include/xgboost/c_api.h"
     :header-name "xgboost/c_api.h"
     :sys-includes ()
     :package "cl-gbdt.xgboost.ffi"
     :prefixes ("XGB" "XGD")
     :output "src/xgboost/c-api.lisp"
     :minimum-functions 70
     :required ("XGBoosterCreate" "XGDMatrixCreateFromDense" "XGBGetLastError")))
  "Per-backend generation settings.")

(defun absolute (relative)
  (merge-pathnames relative (uiop:getcwd)))

(defun spec-path (header arch)
  (make-pathname :name (format nil "~A.~A" (pathname-name header) arch)
                 :type "spec" :defaults header))

(defun die (format-control &rest arguments)
  (format *error-output* "~&error: ~?~%" format-control arguments)
  (uiop:quit 1))

(defun validate (output required minimum-functions emitted)
  "Check the emitted file against the truncation guards. Dies on failure."
  (let ((text (uiop:read-file-string output)))
    (when (< emitted minimum-functions)
      (die "~A has only ~D functions, expected at least ~D.~@
            c2ffi exits 0 even when it truncates; check the include paths and the~@
            Docker image."
           output emitted minimum-functions))
    (dolist (name required)
      (unless (search (format nil "\"~A\"" name) text)
        (die "~A is missing the required function ~A." output name)))
    (dolist (forbidden '("defcstruct" "defcunion" "defcenum"))
      (when (search forbidden text)
        (die "~A contains ~A; the bindings must stay architecture-independent."
             output forbidden)))))

(let ((arch (cffi/c2ffi::local-arch)))
  (format t "~&Generating for ~A~%" arch)
  (dolist (backend *backends*)
    (destructuring-bind (name &key header header-name sys-includes package prefixes
                                output minimum-functions required)
        backend
      (format t "~&==> ~A~%" name)
      (let* ((header-path (absolute header))
             (spec (spec-path header-path arch))
             (output-path (absolute output)))
        (cffi/c2ffi::generate-spec-using-c2ffi
         header-path spec
         :arch arch
         :sys-include-paths (mapcar (lambda (d) (namestring (absolute d))) sys-includes))
        (unless (probe-file spec)
          (die "c2ffi produced no spec at ~A." spec))
        (ensure-directories-exist output-path)
        ;; Emit to a temporary path first. A failed VALIDATE calls DIE, which exits
        ;; through this UNWIND-PROTECT, so a validation failure never leaves a
        ;; deficient file at OUTPUT-PATH for the next quickload to load unnoticed.
        (let ((temp-path (make-pathname :name (format nil "~A-tmp" (pathname-name output-path))
                                         :defaults output-path)))
          (unwind-protect
               (multiple-value-bind (functions constants skipped)
                   (cl-gbdt/src/regen/all:emit-bindings spec header-name package prefixes temp-path)
                 (validate temp-path required minimum-functions functions)
                 (uiop:rename-file-overwriting-target temp-path output-path)
                 (format t "    ~D functions, ~D constants -> ~A~%"
                         functions constants output)
                 (when skipped
                   (format t "    ~D macro(s) were not integer-valued and became no constant:~
                              ~{~%      ~A~}~%"
                           (length skipped) skipped)))
            (when (probe-file temp-path)
              (delete-file temp-path))))))))

(format t "~&done~%")
