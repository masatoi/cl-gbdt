;;;; backend.lisp --- Backend registration and startup.
;;;;
;;;; Shared libraries are opened by an explicit OPEN-BACKEND call, not when the
;;;; system is loaded, so cl-gbdt works on machines where only one of the two
;;;; libraries is installed.

(uiop:define-package #:cl-gbdt/src/backend
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/conditions
                #:unknown-backend)
  (:export #:backend
           #:backend-name
           #:backend-library-path
           #:backend-capabilities
           #:backend-version
           #:backend-open-p
           #:backend-info
           #:register-backend
           #:find-backend-class
           #:open-backend
           #:close-backend
           #:initialize-backend
           #:shutdown-backend
           #:probe-foreign-symbols))

(in-package #:cl-gbdt/src/backend)

(defvar *backend-classes* (make-hash-table :test #'eq)
  "Maps a backend name (a keyword) to its class name.")

(defclass backend ()
  ((name :initarg :name
         :reader backend-name
         :documentation "Backend name, either `:lightgbm' or `:xgboost'.")
   (library-path :initarg :library-path
                 :initform nil
                 :accessor backend-library-path
                 :documentation "Path of the shared library actually loaded.")
   (capabilities :initform nil
                 :accessor backend-capabilities
                 :documentation "Plist of backend-specific features. Reserved for future
symbol-probing-based detection; no backend populates this yet, so `backend-info' always
reports it as NIL.")
   (version :initform nil
            :accessor backend-version
            :documentation "Library version; an inferred value or nil when unavailable.")
   (openp :initform nil
          :accessor backend-openp
          :documentation "Whether the shared library is currently open."))
  (:documentation "A connection to a gradient boosting implementation.

Each backend specializes this class and implements `initialize-backend',
`shutdown-backend', and the unified API methods."))

(defun backend-open-p (backend)
  "Return true when BACKEND's shared library is open."
  (backend-openp backend))

(defun register-backend (name class-name)
  "Associate the backend name NAME with the class CLASS-NAME.

Each backend system calls this when it loads."
  (setf (gethash name *backend-classes*) class-name))

(defun find-backend-class (name)
  "Return the class name registered for the backend NAME, or nil."
  (gethash name *backend-classes*))

(defgeneric initialize-backend (backend &key path)
  (:documentation "Locate and load BACKEND's shared library.

When PATH is supplied it takes precedence over the search. On failure this signals
`backend-library-not-found', `backend-library-load-failed', or
`missing-foreign-symbols'. Does not populate `backend-capabilities' -- no backend
probes capabilities yet, so it stays NIL. Implemented by each backend."))

(defgeneric shutdown-backend (backend)
  (:documentation "Close BACKEND's shared library and release its resources.

Implemented by each backend."))

(defun open-backend (name &key path)
  "Open the backend NAME and return a `backend' instance.

PATH, when supplied, takes precedence over the shared library search. Signals a
condition when NAME is unregistered or initialization fails. Close a successful
instance with `close-backend'."
  (let ((class-name (find-backend-class name)))
    (unless class-name
      (error 'unknown-backend
             :backend name
             :registered (loop :for registered :being :the :hash-keys :of *backend-classes*
                                :collect registered)))
    (let ((backend (make-instance class-name :name name)))
      (initialize-backend backend :path path)
      (setf (backend-openp backend) t)
      backend)))

(defun close-backend (backend)
  "Close BACKEND. Does nothing if it is already closed."
  (when (backend-openp backend)
    (shutdown-backend backend)
    (setf (backend-openp backend) nil))
  backend)

(defun backend-info (backend)
  "Return BACKEND's state as a plist.

Keys are `:name', `:version', `:capabilities', `:library-path' and `:open'."
  (list :name (backend-name backend)
        :version (backend-version backend)
        :capabilities (backend-capabilities backend)
        :library-path (backend-library-path backend)
        :open (backend-open-p backend)))

(defun probe-foreign-symbols (names &key (library :default))
  "Return the C function names in NAMES that are absent from LIBRARY.

Returns nil when all are present. This is how version mismatches are detected. It
cannot catch a function whose name stayed the same while its signature changed;
those are avoided by design instead (see ffi-spec/ABI-BLACKLIST.md).

LIBRARY is passed straight through to `cffi:foreign-symbol-pointer' and defaults
to its own default, `:default', which searches every foreign library the image
currently has loaded. Pass the `cffi:foreign-library' object
`cffi:load-foreign-library' returns for the library just opened to scope the
probe to it -- on CFFI backends that honor the argument. SBCL, the only backend
this project runs on, is not one of them: `cffi-sbcl.lisp''s
`%foreign-symbol-pointer' takes LIBRARY only to validate it against
`cffi::get-foreign-library' (an unregistered designator still signals, which is
real and worth having) and then ignores the handle, resolving through
`sb-sys:find-foreign-symbol-address' -- SBCL's global linkage table -- exactly
as `:default' would. Verified directly: with the vendored LightGBM and XGBoost
libraries both loaded, probing \"XGBoosterCreate\" with LIBRARY bound to
LightGBM's own `foreign-library' object still reports it found. So on this
platform LIBRARY cannot by itself prove a probe came from a specific library --
the caller must still trust that PATH (or the search order above it) named the
right file; only a LightGBM already loaded by something else, providing every
name in *required-symbols* under the same names, defeats that, and no argument
to this function can detect it here."
  (remove-if (lambda (name) (cffi:foreign-symbol-pointer name :library library)) names))
