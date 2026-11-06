;;;; run-example-tests.lisp --- run every open-data example suite; fail on any failure.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/run-example-tests.lisp
;;;;
;;;; The examples under examples/ are plain files that users LOAD, not systems, so neither
;;;; the rove suites nor check-source-reachability ever reach them. Without this script a
;;;; change to the library API, to examples/common.lisp or to examples/lifecycle.lisp could
;;;; break every example, and its rollback and locking tests, with every gate still green.
;;;;
;;;; WHAT RUNS. Every examples/tests*.lisp entry file, in name order. Each one runs its test
;;;; body in a child SBCL through examples/test-process.lisp and signals if the child fails;
;;;; this script records the failure and goes on, so one run reports every broken suite.
;;;;
;;;; THE CHILD'S ASDF. The children need ASDF 3.3.7 or newer, and SBCL's own (which is what
;;;; a bare `sbcl' child gets) is older. A child therefore loads a current asdf.lisp first:
;;;; CL_GBDT_ASDF_BOOTSTRAP when it is set, otherwise the newest asdf.lisp of at least 3.3.7
;;;; that `ros install asdf' left under ~/.roswell/lisp/asdf/<version>/. With neither, the
;;;; children run with whatever ASDF they have and fail with a message that says so.
;;;;
;;;; PREREQUISITES. Several suites train with the real libraries, so run this after
;;;; ./tools/fetch-libs.sh. The children quickload dexador, zip, semz.decompress, archive
;;;; and ironclad as their suites need them, so Quicklisp must be able to reach them.

(require :asdf)

(defparameter +minimum-entry-files+ 10
  "Fewest examples/tests*.lisp files this script accepts finding. It is a floor that
catches a glob matching nothing, which would otherwise pass as a green run of no suites;
adding a suite must never require raising it.")

(defparameter +asdf-header+ "This is ASDF "
  "The text that precedes the version number in the header of a built asdf.lisp.")

(defun asdf-source-version (path)
  "Return the version string PATH's `This is ASDF x.y.z' header names, or NIL."
  (with-open-file (in path)
    (loop repeat 20
          for line = (read-line in nil nil)
          while line
          for start = (search +asdf-header+ line)
          when start
            return (let* ((from (+ start (length +asdf-header+)))
                          (end (position-if-not (lambda (char)
                                                  (or (digit-char-p char) (char= char #\.)))
                                                line :start from)))
                     (subseq line from end)))))

(defun asdf-bootstrap ()
  "Return the asdf.lisp the test children should load first, or NIL when none is found."
  (let ((override (uiop:getenv "CL_GBDT_ASDF_BOOTSTRAP")))
    (if (and override (plusp (length override)))
        (uiop:parse-native-namestring override)
        (let ((current
                (loop for path in (directory (merge-pathnames ".roswell/lisp/asdf/*/asdf.lisp"
                                                              (user-homedir-pathname)))
                      for version = (asdf-source-version path)
                      when (and version (uiop:version<= "3.3.7" version))
                        collect (cons version path))))
          (cdr (first (sort current (lambda (a b) (uiop:version< b a)) :key #'car)))))))

(defun run-entry (entry)
  "Load the suite entry file ENTRY; return NIL on success, or the condition it signalled."
  (format t "~&=== ~A~%" (file-namestring entry))
  (finish-output)
  (handler-case (progn (load entry) nil)
    (error (condition) condition)))

(let* ((root (uiop:getcwd))
       (entries (sort (directory (merge-pathnames "examples/tests*.lisp" root))
                      #'string< :key #'file-namestring))
       (bootstrap (asdf-bootstrap))
       (failures '()))
  (when (< (length entries) +minimum-entry-files+)
    (format *error-output* "~&FAIL: found ~D examples/tests*.lisp files, expected at least ~D~%"
            (length entries) +minimum-entry-files+)
    (uiop:quit 1))
  (load (merge-pathnames "examples/test-process.lisp" root))
  (setf (symbol-value (find-symbol "*ASDF-BOOTSTRAP*" "CL-GBDT-EXAMPLE-TEST-PROCESS"))
        bootstrap)
  (format t "~&ASDF for the test children: ~A~%"
          (if bootstrap (uiop:native-namestring bootstrap) "their own (none found)"))
  (dolist (entry entries)
    (let ((condition (run-entry entry)))
      (if condition
          (progn
            (push (file-namestring entry) failures)
            (format t "~&FAIL ~A~%~A~%" (file-namestring entry) condition))
          (format t "~&PASS ~A~%" (file-namestring entry)))))
  (format t "~&~D/~D example suites passed~%"
          (- (length entries) (length failures)) (length entries))
  (when failures
    (format t "failing suites: ~{~A~^ ~}~%" (reverse failures)))
  (uiop:quit (if failures 1 0)))
