;;;; library.lisp --- Shared-library discovery, used by every backend.
;;;;
;;;; LightGBM and XGBoost each ship as a single shared library, loaded explicitly by
;;;; `open-backend' rather than at system-load time (see src/backend.lisp). Both
;;;; follow the same discovery order and the same override rules -- an explicit PATH,
;;;; then an environment variable, then a vendored directory, then CFFI's own system
;;;; search -- so that whole sequence lives here once, instead of once per backend.
;;;; Only three strings and a default search name differ between them; each backend's
;;;; own `initialize-backend' method supplies those and keeps everything after the
;;;; library is loaded -- symbol probing, version detection, cleanup on failure -- to
;;;; itself.

(uiop:define-package #:cl-gbdt/src/library
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/backend
                #:backend-name)
  (:import-from #:cl-gbdt/src/conditions
                #:backend-library-not-found
                #:backend-library-load-failed)
  (:export #:resolve-and-load-library))

(in-package #:cl-gbdt/src/library)

(defun %env-library-path (env-var)
  "Return the value of the environment variable named ENV-VAR, or NIL when it is unset
or empty."
  (let ((value (uiop:getenv env-var)))
    (when (and value (plusp (length value)))
      value)))

(defun %vendor-search-path (directory pattern)
  "Return the merged pathname DIRECTORY is searched with for PATTERN."
  (merge-pathnames pattern (asdf:system-relative-pathname "cl-gbdt" directory)))

(defun %vendor-library-path (directory pattern)
  "Return the vendored library's path within DIRECTORY, or NIL when none is present.

Searches DIRECTORY for PATTERN, exactly as `tests/functional/support.lisp' does for
the test suite. `tools/fetch-libs.sh' vendors at most one file per backend, so the
first match is returned without further checking."
  (first (directory (%vendor-search-path directory pattern))))

(defun %resolve-library-path (backend path env-var directory pattern)
  "Return the library path `resolve-and-load-library' should load, honoring PATH, then
ENV-VAR, then DIRECTORY searched for PATTERN, in that order. Returns NIL when none of
the three names a candidate; the caller then falls back to CFFI's own system search.

Signals `backend-library-not-found' when ENV-VAR is set but names a path that does not
exist -- a bad override is an error, not a silent fall-through to the next source."
  (let ((override (%env-library-path env-var)))
    (cond
      (path path)
      (override
       (if (probe-file override)
           override
           (error 'backend-library-not-found
                  :backend (backend-name backend) :searched (list override))))
      (t (%vendor-library-path directory pattern)))))

(defun %load-candidate (backend path)
  "Load PATH as BACKEND's shared library. Returns the `cffi:foreign-library' CFFI
reports having loaded.

Signals `backend-library-load-failed' when `cffi:load-foreign-library' rejects
PATH, wrapping whatever condition it signaled as :cause."
  (handler-case (cffi:load-foreign-library path)
    (error (condition)
      (error 'backend-library-load-failed
             :backend (backend-name backend) :path path :cause condition))))

(defun %load-system-search (backend directory pattern default-name)
  "Let CFFI search the system library paths for DEFAULT-NAME. Returns the
`cffi:foreign-library' CFFI reports having loaded.

Signals `backend-library-not-found' when nothing turns up anywhere -- PATH, ENV-VAR
and DIRECTORY searched for PATTERN were all already ruled out by the time this runs."
  (handler-case (cffi:load-foreign-library (list :default default-name))
    (error ()
      (error 'backend-library-not-found
             :backend (backend-name backend)
             :searched (list (namestring (%vendor-search-path directory pattern))
                              "system library search path")))))

(defun resolve-and-load-library (backend &key path env-var directory pattern default-name)
  "Locate and load BACKEND's shared library, returning the `cffi:foreign-library' loaded
and the path it came from as a second value.

Search order: PATH, then ENV-VAR, then DIRECTORY searched for PATTERN, then CFFI's own
system search for DEFAULT-NAME. ENV-VAR is honored strictly -- when it is set but names
a path that does not exist this signals `backend-library-not-found' rather than falling
through, because a typo in an override that silently loads a different library is worse
than a failure.

Signals `backend-library-not-found' (with `:searched') when no candidate is found
anywhere -- PATH, ENV-VAR and DIRECTORY all failing to name one, and CFFI's own system
search for DEFAULT-NAME failing too -- and `backend-library-load-failed' (with `:path'
and `:cause') when a candidate was found but `cffi:load-foreign-library' rejects it."
  (let* ((candidate (%resolve-library-path backend path env-var directory pattern))
         (library (if candidate
                       (%load-candidate backend candidate)
                       (%load-system-search backend directory pattern default-name))))
    (values library (namestring (cffi:foreign-library-pathname library)))))
