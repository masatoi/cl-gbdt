;;;; run-tests.lisp --- Run one test system and exit with a meaningful status.
;;;;
;;;; Usage:
;;;;   CL_GBDT_TEST_SYSTEM=cl-gbdt/tests ros run -- --non-interactive \
;;;;     --load tools/ci/run-tests.lisp
;;;;
;;;; `asdf:test-system' exits 0 even when tests fail -- verified -- so a CI job
;;;; written the obvious way would be permanently green. `rove:run' returns false on
;;;; failure, and this script turns that into an exit status.
;;;;
;;;; It also enforces the foreign-library invariant each layer claims, because the
;;;; rove summary reads the same whether the functional tests ran or all skipped:
;;;;
;;;;   cl-gbdt/tests             must open NO foreign library
;;;;   cl-gbdt/tests/functional  must open BOTH backend libraries
;;;;
;;;; Without that second check, a runner with no vendor/ directory would skip every
;;;; test and report success -- the exact outcome the functional suite exists to
;;;; prevent.

(require :asdf)

;;; Loaded first so the reader can resolve these packages in the forms below.
(ql:quickload '("cffi" "rove") :silent t)

(defparameter *system* (uiop:getenv "CL_GBDT_TEST_SYSTEM")
  "The ASDF test system to run, named by the environment.")

(defun loaded-library-names ()
  "Return the base names of the foreign libraries CFFI currently has open."
  (remove nil
          (mapcar (lambda (library)
                    (let ((path (cffi:foreign-library-pathname library)))
                      (when path (string-downcase (file-namestring path)))))
                  (cffi:list-foreign-libraries))))

(defun library-loaded-p (substring names)
  "True when some name in NAMES contains SUBSTRING."
  (some (lambda (name) (search substring name)) names))

(defun check-foreign-libraries (system names)
  "Verify SYSTEM opened the foreign libraries it is supposed to. Returns a boolean."
  (cond
    ((string= system "cl-gbdt/tests")
     (or (null names)
         (progn
           (format *error-output*
                   "~&FAIL: layer 1 must open no foreign library, but opened ~S.~@
                    Something has made the library-free layer depend on a backend.~%"
                   names)
           nil)))
    ((string= system "cl-gbdt/tests/functional")
     (or (and (library-loaded-p "lightgbm" names)
              (library-loaded-p "xgboost" names))
         (progn
           (format *error-output*
                   "~&FAIL: the functional suite opened ~S.~@
                    Both backends must actually load. A run where every test skipped~@
                    prints the same summary as one where every test passed, so this~@
                    has to be checked rather than inferred from the exit status.~@
                    Did ./tools/fetch-libs.sh run?~%"
                   names)
           nil)))
    ;; Not a pass. Both branches above match on a literal system name, so renaming a test
    ;; system silently disables the very check that tells a real run from four skipped
    ;; tests. That nearly happened when `cl-gbdt/functional-tests' became
    ;; `cl-gbdt/tests/functional'. An unrecognised name is now a failure, so the next
    ;; rename cannot pass quietly.
    (t (format *error-output*
               "~&FAIL: no foreign-library expectation is defined for ~S.~@
                Add one to check-foreign-libraries -- a test system with no expectation~@
                cannot be distinguished from one whose tests all skipped.~%"
               system)
       nil)))

(unless (and *system* (plusp (length *system*)))
  (format *error-output* "~&FAIL: set CL_GBDT_TEST_SYSTEM to the system to run.~%")
  (uiop:quit 2))

(format t "~&==> ~A~%" *system*)
(ql:quickload *system* :silent t)

(let* ((passed (rove:run *system*))
       (names (loaded-library-names))
       (libraries-ok (check-foreign-libraries *system* names)))
  (format t "~&foreign libraries open: ~S~%" names)
  (uiop:quit (if (and passed libraries-ok) 0 1)))
