(in-package #:cl-user)

;;;; Run example regression tests in a fresh Lisp image.

(defpackage #:cl-gbdt-example-test-process
  (:use #:cl)
  (:export #:run-test-in-child #:*asdf-bootstrap* #:*child*
           #:*project-asd* #:*working-directory*))

(in-package #:cl-gbdt-example-test-process)

(defvar *asdf-bootstrap* nil
  "Optional pathname to a current ASDF source file for an isolated child.")

(defvar *child* nil
  "True only in a test child started by RUN-TEST-IN-CHILD.")

(defvar *project-asd* nil
  "Project ASD propagated to nested test processes.")

(defvar *working-directory* nil
  "Default working directory propagated to nested test processes.")

(defun registered-asd-sources (project-asd)
  "Return source files for parent-registered primary systems without loading them."
  (let ((seen (make-hash-table :test #'equal))
        (result (list (cons "cl-gbdt" (namestring project-asd)))))
    (setf (gethash "cl-gbdt" seen) t)
    (dolist (name (asdf:registered-systems))
      (let* ((primary (asdf:primary-system-name name))
             (system (asdf:registered-system primary))
             (source (and system (asdf:system-source-file system))))
        (when (and source (uiop:file-exists-p source) (not (gethash primary seen)))
          (setf (gethash primary seen) t)
          (push (cons primary (namestring source)) result))))
    result))

(defun source-registry-asd-sources ()
  "Snapshot initialized source-registry files without discovering new systems."
  (let ((result nil))
    (when (hash-table-p asdf/source-registry:*source-registry*)
      (maphash (lambda (name source)
                 (when (and (stringp name) (typep source '(or pathname string))
                            (string= name (asdf:primary-system-name name))
                            (uiop:file-exists-p source))
                   (push (cons name (namestring (pathname source))) result)))
               asdf/source-registry:*source-registry*))
    result))

(defun safe-registry-value (form)
  "Resolve simple pathname expressions without executing arbitrary registry code."
  (handler-case
      (typecase form
        ((or pathname string) (values form t))
        (symbol (if (boundp form) (values (symbol-value form) t) (values nil nil)))
        (cons
         (let ((operator (first form)))
           (cond
             ((eq operator 'quote)
              (if (and (consp (rest form)) (null (cddr form)))
                  (values (second form) t)
                  (values nil nil)))
             ((member operator '(merge-pathnames pathname namestring make-pathname
                                 uiop:ensure-directory-pathname uiop:pathname-directory-pathname
                                 uiop:subpathname))
              (let ((arguments
                      (loop for argument in (rest form)
                            collect (multiple-value-bind (value safe)
                                        (safe-registry-value argument)
                                      (unless safe (return-from safe-registry-value
                                                     (values nil nil)))
                                      value))))
                (values (apply (symbol-function operator) arguments) t)))
             (t (values nil nil)))))
        (t (values nil nil)))
    (error () (values nil nil))))

(defun central-registry-entries ()
  "Preserve central registry order without evaluating arbitrary code in the parent."
  (loop for entry in asdf:*central-registry*
        append
        (multiple-value-bind (value safe) (safe-registry-value entry)
          (let ((path (and safe (typep value '(or pathname string))
                           (merge-pathnames (pathname value) *default-pathname-defaults*))))
            (cond
              ((and safe (null value)) nil)
              ((and path (uiop:absolute-pathname-p path) (uiop:directory-pathname-p path))
               (list path))
              (t
               ;; ASDF evaluates such an entry itself, and only if its search reaches
               ;; it. Hand the form over as data so the child's ASDF does exactly that;
               ;; the parent never evaluates it. A form that cannot be printed readably
               ;; is replaced by one that reports the problem when it is reached.
               (let ((printable
                       (handler-case
                           (let ((*print-readably* t) (*print-circle* t)
                                 (*print-length* nil) (*print-level* nil))
                             (prin1-to-string entry)
                             t)
                         (error () nil))))
                 (list
                  (if printable
                      entry
                      '(error "Cannot transfer registry entry; load the system first."))))))))))

(defun run-test-in-child (body &key systems
                                  (asdf-bootstrap (or *asdf-bootstrap*
                                                      (uiop:getenv "CL_GBDT_ASDF_BOOTSTRAP")))
                                  (project-asd *project-asd*) (directory *working-directory*)
                                  (output *standard-output*) (error-output *error-output*)
                                  ignore-error-status)
  "Load BODY in an isolated SBCL process, loading SYSTEMS with a current ASDF.
ASDF-BOOTSTRAP or CL_GBDT_ASDF_BOOTSTRAP can name a current ASDF source file.
DIRECTORY selects the child working directory. OUTPUT and ERROR-OUTPUT are streams
or NIL. Return standard output, error output, and exit status; signal on failure unless
IGNORE-ERROR-STATUS."
  (when (and asdf-bootstrap (not (uiop:file-exists-p asdf-bootstrap)))
    (error "ASDF bootstrap file does not exist: ~A" asdf-bootstrap))
  (let* ((body (truename body))
         (body-directory (uiop:pathname-directory-pathname body))
         (asd (truename (or project-asd (merge-pathnames "../cl-gbdt.asd" body-directory))))
         (directory (truename (or directory (uiop:pathname-directory-pathname asd))))
         (helper (truename (merge-pathnames "examples/test-process.lisp"
                                            (uiop:pathname-directory-pathname asd))))
         (bootstrap (and asdf-bootstrap (truename asdf-bootstrap)))
         (ql-package (find-package "QL"))
         (ql-home-symbol (and ql-package (find-symbol "*QUICKLISP-HOME*" ql-package)))
         (ql-home (and ql-home-symbol (boundp ql-home-symbol) (symbol-value ql-home-symbol)))
         (ql-setup (and ql-home (probe-file (merge-pathnames "setup.lisp" ql-home))))
         (program (list (namestring sb-ext:*runtime-pathname*) "--noinform" "--non-interactive"))
         (sources (registered-asd-sources asd))
         (source-registry (source-registry-asd-sources))
         (registry-entries (central-registry-entries)))
    ;; A private file avoids command-line length limits for large registries.
    ;; Its data are quoted; the child loads an ASD only when it needs that system.
    (uiop:with-temporary-file (:pathname registry-file :stream registry-out
                               :prefix "cl-gbdt-test-registry-" :type "lisp")
      (let ((*print-length* nil) (*print-level* nil) (*print-readably* t))
        (format registry-out
                "(let ((sources (make-hash-table :test 'equal)) ~
                       (fallback (make-hash-table :test 'equal)))~%~
                 (dolist (entry '~S) ~
                   (setf (gethash (car entry) sources) (pathname (cdr entry))))~%~
                 (dolist (entry '~S) ~
                   (setf (gethash (car entry) fallback) (pathname (cdr entry))))~%~
                 (let ((source-searcher ~
                         (lambda (name) ~
                           (let ((primary (asdf:primary-system-name name))) ~
                             (when (string= (asdf:coerce-name name) primary) ~
                               (gethash primary fallback))))) ~
                       (inserted nil)) ~
                   (setf asdf:*system-definition-search-functions* ~
                         (loop for searcher in asdf:*system-definition-search-functions* ~
                               append (if (or (eq searcher ~
                                                   'asdf/system-registry:~
                                                    sysdef-central-registry-search) ~
                                              (eq searcher ~
                                                  #'asdf/system-registry:~
                                                    sysdef-central-registry-search)) ~
                                          (progn (setf inserted t) ~
                                                 (list searcher source-searcher)) ~
                                          (list searcher)))) ~
                   (unless inserted ~
                     (setf asdf:*system-definition-search-functions* ~
                           (append asdf:*system-definition-search-functions* ~
                                   (list source-searcher))))) ~
                 (push (lambda (name) ~
                         (let ((primary (asdf:primary-system-name name))) ~
                           (when (string= (asdf:coerce-name name) primary) ~
                             (gethash primary sources)))) ~
                       asdf:*system-definition-search-functions*))~%"
                sources source-registry)
        (format registry-out
                "(setf asdf:*central-registry* ~
                       (append '~S asdf:*central-registry*))~%"
                registry-entries))
      :close-stream
      (let ((arguments
              (append
               program
               (when bootstrap
                 (list "--eval"
                       (format nil "(let ((*error-output* (make-broadcast-stream))) (load #P~S))"
                               (namestring bootstrap))))
               (when ql-setup
                 (list "--eval"
                       (format nil
                               "(let* ((pkg (find-package \"QL\")) ~
                                (quickload (and pkg (find-symbol \"QUICKLOAD\" pkg)))) ~
                                (unless (and quickload (fboundp quickload)) (load #P~S)))"
                               (namestring ql-setup))))
               (list "--eval" "(require :asdf)"
                     "--eval"
                     (concatenate 'string
                      "(unless (uiop:version<= \"3.3.7\" (asdf:asdf-version)) "
                      "(error \"Isolated example tests require ASDF 3.3.7 or newer; "
                      "supply *asdf-bootstrap* if needed.\"))")
                     "--load" (namestring registry-file)
                     "--eval" (format nil "(asdf:load-asd #P~S)" (namestring asd)))
               (when systems
                 (list "--eval"
                       (if ql-setup
                           (format nil "(ql:quickload '~S :silent t)" systems)
                           (format nil "(dolist (system '~S) (asdf:load-system system))" systems))))
               (list "--load" (namestring helper)
                     "--eval"
                     (format nil
                             "(setf cl-gbdt-example-test-process:*child* t ~
                              cl-gbdt-example-test-process:*asdf-bootstrap* ~S ~
                              cl-gbdt-example-test-process:*project-asd* ~S ~
                              cl-gbdt-example-test-process:*working-directory* ~S)"
                             (and bootstrap (namestring bootstrap))
                             (namestring asd) (namestring directory))
                     "--eval"
                     (format nil
                             "(handler-bind ((sb-kernel:redefinition-warning #'muffle-warning)) ~
                              (load #P~S))"
                             (namestring body))
                     "--eval" "(uiop:quit 0)"))))
        (multiple-value-bind (stdout stderr status)
            (uiop:run-program arguments :directory directory :output :string :error-output :string
                              :ignore-error-status t)
          (unless (or (zerop status) ignore-error-status)
            (error "Isolated example test ~A exited with status ~D.~%~A~%~A"
                   body status stdout stderr))
          (when output (write-string stdout output))
          (when error-output (write-string stderr error-output))
          (values stdout stderr status))))))
