;;;; backend.lisp --- Backend registration and startup.
;;;;
;;;; Shared libraries are opened by an explicit OPEN-BACKEND call, not when the
;;;; system is loaded, so cl-gbdt works on machines where only one of the two
;;;; libraries is installed.

(uiop:define-package #:cl-gbdt/src/backend
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/conditions
                #:unknown-backend
                #:unknown-capability)
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
           #:probe-foreign-symbols
           #:*known-capabilities*
           #:backend-supports-p
           #:probe-capabilities))

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
                 :documentation "Plist of capability keyword to T/NIL, as returned by
`probe-capabilities' at `open-backend' time. Read through `backend-supports-p' rather
than directly, so an unregistered keyword signals `unknown-capability' instead of
silently reading as NIL.")
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
`missing-foreign-symbols'. Expected to populate `backend-capabilities' with the plist
`probe-capabilities' returns, so `backend-supports-p' has something to read. Implemented
by each backend."))

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

(defparameter *known-capabilities*
  '(:sparse-input
    :evaluation-history
    :early-stopping
    :model-slicing
    :multidimensional-feature-score)
  "Every capability name `backend-supports-p' will answer for.

Policy section 7's five names. A name is registered here as soon as it is a question worth
asking, whether or not any backend answers true to it yet -- `:early-stopping' is registered
and false everywhere, which says \"not supported yet\" rather than \"never heard of it\".
Registering the name is what makes a misspelling distinguishable from a real answer.")

(defun backend-supports-p (backend capability)
  "Return true when BACKEND provides CAPABILITY, NIL when it does not.

CAPABILITY must be one of `*known-capabilities*'; anything else signals
`unknown-capability' rather than answering NIL, so a typo cannot be mistaken for a
supported-but-absent feature.

A true answer means the shared library actually loaded resolved every foreign symbol the
capability needs, as probed at `open-backend' -- not that the headers cl-gbdt was built
against declared them. A false answer means the feature is unavailable here, and is never a
licence to fall back to something else silently: the operation itself signals
`capability-unavailable' (policy section 7)."
  (unless (member capability *known-capabilities*)
    (error 'unknown-capability :capability capability :known *known-capabilities*))
  (and (getf (backend-capabilities backend) capability) t))

(defun probe-capabilities (optional-symbols &key (library :default))
  "Return the capability plist for OPTIONAL-SYMBOLS, probed against LIBRARY.

OPTIONAL-SYMBOLS is an alist of a capability keyword and the C function names that capability
needs: ((:model-slicing \"XGBoosterSlice\") ...). A capability is true when every one of its
names resolves.

Every declared capability appears in the result, true and false alike, rather than only the
true ones -- `backend-info' should be able to report what was asked as well as what was
answered.

**This never signals for a missing symbol**, which is the whole difference between an optional
symbol and a required one: policy section 8 says an optional symbol's absence disables that one
capability rather than preventing the backend from opening. Callers wanting the required
behaviour use `probe-foreign-symbols' directly and signal `missing-foreign-symbols'
themselves.

LIBRARY is passed through to `probe-foreign-symbols'; see its docstring for the SBCL caveat
about scoping a probe to one library."
  (loop :for (capability . names) :in optional-symbols
        :append (list capability
                      (null (probe-foreign-symbols names :library library)))))
