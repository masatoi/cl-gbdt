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
    :multidimensional-feature-score
    :missing-value
    :categorical-features)
  "Every capability name `backend-supports-p' will answer for.

Policy section 7 named five as a MINIMUM -- it asks that at least those be answerable, not
that they be the whole list -- so this grows as new questions become worth asking.
`:missing-value' is the first name added past that floor: whether the backend lets a caller
say which value in the data means *missing*, which XGBoost does through its creation and
prediction config JSONs and LightGBM has no C-API key for at all. A name is registered here
as soon as it is a question worth asking, whether or not any backend answers true to it yet --
`:multidimensional-feature-score' is registered and false everywhere, which says \"not
supported yet\" rather than \"never heard of it\". Registering the name is what makes a
misspelling distinguishable from a real answer.

`:categorical-features' is the second name past that floor: whether the backend lets a caller
name which COLUMNS of the data hold categories rather than quantities, so a split on one of
them partitions the category set instead of thresholding an ordinal that has no order. XGBoost
answers it through the `\"feature_type\"' field its `XGDMatrixSetStrFeatureInfo' attaches to a
finished DMatrix, and LightGBM through a `categorical_feature' key in the parameter string that
builds the dataset -- a key rather than a C symbol, so nothing a symbol probe can decide. It is
a question about `make-dataset' alone: `predict' takes no such argument, the trained trees
already carrying the category sets they split on.

`:evaluation-history' is true on both backends: `train' records one, and each backend names
the capability in its own `*provided-capabilities*' rather than in `*optional-symbols*',
because the C functions it needs are already in that backend's `*required-symbols*' and a
probe therefore has nothing left to decide. See `probe-capabilities''s PROVIDED.")

(defun backend-supports-p (backend capability)
  "Return true when BACKEND provides CAPABILITY, NIL when it does not.

CAPABILITY must be one of `*known-capabilities*'; anything else signals
`unknown-capability' rather than answering NIL, so a typo cannot be mistaken for a
supported-but-absent feature.

A true answer means the shared library actually loaded resolved every foreign symbol the
capability needs, as probed at `open-backend' -- not that the headers cl-gbdt was built
against declared them -- or that the backend declared the capability unconditionally,
having nothing left to probe because everything it needs is in `*required-symbols*' and the
backend opened at all (see `probe-capabilities''s PROVIDED). A false answer means the
feature is unavailable here, and is never a licence to fall back to something else silently:
the operation itself signals `capability-unavailable' (policy section 7)."
  (unless (member capability *known-capabilities*)
    (error 'unknown-capability :capability capability :known *known-capabilities*))
  (and (getf (backend-capabilities backend) capability) t))

(defun probe-capabilities (optional-symbols &key provided (library :default))
  "Return the capability plist for OPTIONAL-SYMBOLS, probed against LIBRARY, with every
capability in PROVIDED recorded true ahead of them.

OPTIONAL-SYMBOLS is an alist of a capability keyword and the C function names that capability
needs: ((:model-slicing \"XGBoosterSlice\") ...). A capability is true when every one of its
names resolves.

PROVIDED is a list of capability keywords the backend provides unconditionally, recorded as
true without being probed. A probe cannot express \"always true\": it derives every answer
from a symbol lookup, and a capability whose C functions are all in `*required-symbols*' has
nothing left to look up -- `open-backend' has already refused to open a library missing any
of them. `:evaluation-history' is the case this exists for; leaving it out of the plist
entirely would make `backend-supports-p' answer NIL, which that function documents as the
feature being unavailable and the operation signalling `capability-unavailable', and neither
is true of a capability both backends ship.

PROVIDED's entries come first, so a capability named in both PROVIDED and OPTIONAL-SYMBOLS
reads true through `getf' whatever the probe found. That combination is a contradiction in
the backend's own declarations rather than a case with a useful meaning: a capability is
either unconditional or probed, and `tools/ci/check-abi-blacklist.lisp' checks both lists
against `*known-capabilities*' but cannot tell which list a name belongs in.

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
  (append (loop :for capability :in provided
                :append (list capability t))
          (loop :for (capability . names) :in optional-symbols
                :append (list capability
                              (null (probe-foreign-symbols names :library library))))))
