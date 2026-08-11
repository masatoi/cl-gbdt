;;;; classes.lisp --- LightGBM's CLOS types and the shared library's lifetime.
;;;;
;;;; Layer 1, not Layer 2. `initialize-backend' and `shutdown-backend' are generic functions
;;;; declared in `src/backend.lisp' -- the shared basis every layer is written against -- not in
;;;; `src/protocol.lisp', so implementing them is backend work that owes nothing to the unified
;;;; API. Keeping them here is what lets `cl-gbdt/lightgbm' open and close the library without
;;;; loading Layer 2 at all.
;;;;
;;;; Loads after `native.lisp' and cannot precede it: `initialize-backend' reads that file's
;;;; `*required-symbols*', `*optional-symbols*', `*provided-capabilities*', `*library-env-var*',
;;;; `*vendor-library-directory*', `*vendor-library-pattern*' and `*default-library-name*'.
;;;;
;;;; Every form here that reaches the shared library wraps its whole body in
;;;; `with-foreign-float-traps-masked', for the reason `cl-gbdt/src/lightgbm/protocol''s
;;;; "Floating-point trap safety" comment gives. Moving a form between files never removes that
;;;; wrap.

(uiop:define-package #:cl-gbdt/src/lightgbm/classes
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/lightgbm/native
                #:*library-env-var*
                #:*vendor-library-directory*
                #:*vendor-library-pattern*
                #:*default-library-name*
                #:*required-symbols*
                #:*optional-symbols*
                #:*provided-capabilities*)
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
  (:export #:lightgbm-backend
           #:lightgbm-dataset
           #:lightgbm-booster
           #:%lightgbm-foreign-library))

(in-package #:cl-gbdt/src/lightgbm/classes)

;;; ---------------------------------------------------------------------------
;;; The backend class

(defclass lightgbm-backend (backend)
  ((foreign-library :initform nil
                     :accessor %lightgbm-foreign-library
                     :documentation "The `cffi:foreign-library' `initialize-backend'
loaded, kept so `shutdown-backend' can close exactly this one."))
  (:documentation "A connection to the LightGBM shared library, implementing
cl-gbdt's unified backend protocol."))

(register-backend :lightgbm 'lightgbm-backend)

;;; Handles must be subclassed per backend, not shared. `make-dataset' and `train'
;;; dispatch on the backend, so they would be unambiguous either way -- but
;;; `dataset-num-rows', `predict', `free-dataset' and `free-booster' dispatch on the
;;; HANDLE. A method on the core `dataset' class would be replaced, not specialized,
;;; the moment the XGBoost backend defined its own, and the failure would be a
;;; LightGBM dataset silently answering through XGBoost's C API.

(defclass lightgbm-dataset (dataset) ()
  (:documentation "A dataset held by the LightGBM library."))

(defclass lightgbm-booster (booster) ()
  (:documentation "A booster held by the LightGBM library."))

(defmethod initialize-backend ((backend lightgbm-backend) &key path)
  "Load LightGBM's shared library and record its capabilities on BACKEND.

Discovery order: PATH, then *library-env-var*, then the vendored directory under
*vendor-library-directory*, then CFFI's system library search for
*default-library-name* -- see `resolve-and-load-library' for the exact rules and
the conditions each failure mode signals.

Once a library is loaded, every name in *required-symbols* must resolve via
`probe-foreign-symbols', passed the `cffi:foreign-library' just loaded as
:LIBRARY -- see that function's docstring for the SBCL caveat: it validates
the library argument but, on this platform, cannot actually scope the symbol
search to it -- or this signals `missing-foreign-symbols' -- the
version-mismatch check that function exists for. Only once that required check
has passed does this probe *optional-symbols* via `probe-capabilities' and
record the result on `backend-capabilities' -- unlike a missing required
symbol, a missing optional one never signals; it only makes
`backend-supports-p' answer NIL for the capability that symbol backs.
*provided-capabilities* goes to the same call as :PROVIDED, recording the
capabilities this backend provides unconditionally -- nothing is probed for
them, because the C functions they need are in *required-symbols* and the probe
above has already passed.

LightGBM's C API has no runtime version query, so `backend-version' is left
NIL rather than guessed --
and, unlike `cl-gbdt/src/xgboost/protocol''s `initialize-backend', this never
calls `cl-gbdt/src/version''s `check-backend-version': with nothing to read, a
call here could never confirm compatibility, only ever warn on every single
open, which is not a check worth leaving in. See `*lightgbm-version-range*''s
docstring for the fuller explanation of this asymmetry between the backends.

`open-backend' only marks a backend open -- and so only calls `close-backend' on
it -- once this method returns normally. So if the symbol probe (or anything
else after the library loads) signals, the library is closed right here before
the condition propagates; otherwise it would stay mapped into the process with
BACKEND dropped and nothing left able to close it."
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
               (setf (%lightgbm-foreign-library backend) library)
               (setf (backend-library-path backend) library-path)
               (let ((missing (probe-foreign-symbols *required-symbols* :library library)))
                 (when missing
                   (error 'missing-foreign-symbols
                          :backend (backend-name backend) :names missing)))
               (setf (backend-capabilities backend)
                     (probe-capabilities *optional-symbols*
                                         :provided *provided-capabilities*
                                         :library library))
               (setf (backend-version backend) nil)
               (setf succeeded t))
          (unless succeeded
            (handler-case (cffi:close-foreign-library library)
              (error () nil))
            (setf (%lightgbm-foreign-library backend) nil)))
        backend))))

(defmethod shutdown-backend ((backend lightgbm-backend))
  "Close LightGBM's shared library.

`cffi:close-foreign-library' drops cl-gbdt's own reference and, on platforms
where the C loader honors `dlclose' reference counting, may unmap the library;
POSIX does not guarantee an actual unload, so this cannot promise the library's
code and data are gone from the process afterward -- only that cl-gbdt no
longer holds it open."
  (with-foreign-float-traps-masked
    (let ((library (%lightgbm-foreign-library backend)))
      (when library
        (cffi:close-foreign-library library)
        (setf (%lightgbm-foreign-library backend) nil)))
    backend))
