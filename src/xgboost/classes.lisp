;;;; classes.lisp --- XGBoost's CLOS types and the shared library's lifetime.
;;;;
;;;; Layer 1, not Layer 2. `initialize-backend' and `shutdown-backend' are generic functions
;;;; declared in `src/backend.lisp' -- the shared basis every layer is written against -- not
;;;; in `src/protocol.lisp', so implementing them is backend work that owes nothing to the
;;;; unified API. Keeping them here is what lets `cl-gbdt/xgboost' open and close the library
;;;; without loading Layer 2 at all.
;;;;
;;;; The types and that lifetime are now the whole of this file. `slice-model' lived here too
;;;; -- it builds a booster handle, so it must name the concrete `xgboost-booster' class
;;;; defined below, and `native.lisp' must not depend on this file -- until
;;;; `cl-gbdt/src/xgboost/api' appeared, which names this file and so satisfies that constraint
;;;; equally. It is an operation over the booster class rather than any part of the library's
;;;; lifetime, so that is where it belongs; see that file's own Model slicing section.
;;;;
;;;; Loads after `native.lisp' and cannot precede it: `initialize-backend' reads that file's
;;;; `*required-symbols*', `*optional-symbols*', `*provided-capabilities*', `*library-env-var*',
;;;; `*vendor-library-directory*', `*vendor-library-pattern*' and `*default-library-name*' and
;;;; calls its `%read-version'.
;;;;
;;;; Every form here that reaches the shared library wraps its whole body in
;;;; `with-foreign-float-traps-masked', for the reason `cl-gbdt/src/xgboost/protocol''s
;;;; "Floating-point trap safety" comment gives. Moving a form between files never removes
;;;; that wrap -- but it can move a form out from under the check that verifies it, and it did
;;;; once, for `slice-model': `tools/ci/check-float-traps.lisp''s +BACKEND-FILE-PATTERNS+ named
;;;; `src/*/native.lisp' and `src/*/protocol.lisp' only, so that function's wrap went unchecked
;;;; for a while after it moved here from `protocol.lisp'. That glob now carries
;;;; `src/*/classes.lisp' and `src/*/api.lisp' as well, so the same move again cost nothing:
;;;; the scan reports this file as "2 defmethods, 0 public defuns, 0 unmasked" and follows
;;;; `slice-model' to its new home. A wrap dropped from anything here fails the build rather
;;;; than waiting for a reviewer.

(uiop:define-package #:cl-gbdt/src/xgboost/classes
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/xgboost/native
                #:*library-env-var*
                #:*vendor-library-directory*
                #:*vendor-library-pattern*
                #:*default-library-name*
                #:*required-symbols*
                #:*optional-symbols*
                #:*provided-capabilities*
                #:%read-version)
  (:import-from #:cl-gbdt/src/backend
                #:backend
                #:backend-name
                #:backend-library-path
                #:backend-version
                #:backend-capabilities
                #:probe-foreign-symbols
                #:probe-capabilities
                #:register-backend
                #:initialize-backend
                #:shutdown-backend)
  (:import-from #:cl-gbdt/src/handle
                #:dataset
                #:booster)
  (:import-from #:cl-gbdt/src/conditions
                #:missing-foreign-symbols)
  (:import-from #:cl-gbdt/src/library
                #:resolve-and-load-library)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:import-from #:cl-gbdt/src/version
                #:check-backend-version
                #:*xgboost-version-range*)
  (:export #:xgboost-backend
           #:xgboost-dataset
           #:xgboost-booster
           #:%xgboost-foreign-library))

(in-package #:cl-gbdt/src/xgboost/classes)

;;; ---------------------------------------------------------------------------
;;; The backend class

(defclass xgboost-backend (backend)
  ((foreign-library :initform nil
                     :accessor %xgboost-foreign-library
                     :documentation "The `cffi:foreign-library' `initialize-backend'
loaded, kept so `shutdown-backend' can close exactly this one."))
  (:documentation "A connection to the XGBoost shared library, implementing cl-gbdt's
unified backend protocol."))

(register-backend :xgboost 'xgboost-backend)

;;; Handles must be subclassed per backend, not shared -- see
;;; `cl-gbdt/src/lightgbm/classes''s identical commentary above `lightgbm-dataset'.
;;; `dataset-num-rows', `predict', `free-dataset' and `free-booster' dispatch on the
;;; HANDLE, so a method on the core `dataset' or `booster' class would be replaced, not
;;; specialized, the moment the other backend defined its own -- an XGBoost dataset would
;;; silently answer through LightGBM's C API, or vice versa.

(defclass xgboost-dataset (dataset) ()
  (:documentation "A dataset (a DMatrix) held by the XGBoost library."))

(defclass xgboost-booster (booster) ()
  (:documentation "A booster held by the XGBoost library."))

(defmethod initialize-backend ((backend xgboost-backend) &key path)
  "Load XGBoost's shared library and record its capabilities on BACKEND.

Discovery order: PATH, then *library-env-var*, then the vendored directory under
*vendor-library-directory*, then CFFI's system library search for
*default-library-name* -- see `resolve-and-load-library' for the exact rules and the
conditions each failure mode signals.

Once a library is loaded, every name in *required-symbols* must resolve via
`probe-foreign-symbols', passed the `cffi:foreign-library' just loaded as :LIBRARY -- see
that function's docstring for the SBCL caveat: it validates the library argument but,
on this platform, cannot actually scope the symbol search to it -- or this signals
`missing-foreign-symbols'.

Only once that required check has passed does this probe *optional-symbols* via
`probe-capabilities' and record the result on `backend-capabilities' -- unlike a missing
required symbol, a missing optional one never signals; it only makes `backend-supports-p'
answer NIL for the capability that symbol backs. *provided-capabilities* goes to the same
call as :PROVIDED, recording the capabilities this backend provides unconditionally --
nothing is probed for them, because the C functions they need are in *required-symbols* and
the probe above has already passed.

Once `backend-version' is read, `check-backend-version' compares it against
`*xgboost-version-range*' and signals `untested-backend-version' -- a warning, not an
error -- when it falls outside that range's recorded bounds. This is the only backend
this runs for: LightGBM's C API has no version entry point, so `backend-version' stays
NIL there and there is nothing to compare -- see
`cl-gbdt/src/lightgbm/classes''s `initialize-backend'.

`open-backend' only marks a backend open -- and so only calls `close-backend' on it --
once this method returns normally. So if the symbol probe (or anything else after the
library loads) signals, the library is closed right here before the condition propagates;
otherwise it would stay mapped into the process with BACKEND dropped and nothing left able
to close it."
  (with-foreign-float-traps-masked
    (multiple-value-bind (library library-path)
        (resolve-and-load-library backend :path path
                                           :env-var *library-env-var*
                                           :directory *vendor-library-directory*
                                           :pattern *vendor-library-pattern*
                                           :default-name *default-library-name*)
      (let ((succeeded nil))
        (unwind-protect
             (progn
               (setf (%xgboost-foreign-library backend) library)
               (setf (backend-library-path backend) library-path)
               (let ((missing (probe-foreign-symbols *required-symbols* :library library)))
                 (when missing
                   (error 'missing-foreign-symbols
                          :backend (backend-name backend) :names missing)))
               (setf (backend-capabilities backend)
                     (probe-capabilities *optional-symbols*
                                         :provided *provided-capabilities*
                                         :library library))
               (setf (backend-version backend) (%read-version))
               (check-backend-version :xgboost (backend-version backend)
                                       *xgboost-version-range*)
               (setf succeeded t))
          (unless succeeded
            (handler-case (cffi:close-foreign-library library)
              (error () nil))
            (setf (%xgboost-foreign-library backend) nil)))
        backend))))

(defmethod shutdown-backend ((backend xgboost-backend))
  "Close XGBoost's shared library.

`cffi:close-foreign-library' drops cl-gbdt's own reference and, on platforms where the C
loader honors `dlclose' reference counting, may unmap the library; POSIX does not
guarantee an actual unload, so this cannot promise the library's code and data are gone
from the process afterward -- only that cl-gbdt no longer holds it open."
  (with-foreign-float-traps-masked
    (let ((library (%xgboost-foreign-library backend)))
      (when library
        (cffi:close-foreign-library library)
        (setf (%xgboost-foreign-library backend) nil)))
    backend))
