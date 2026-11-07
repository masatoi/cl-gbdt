(in-package #:cl-user)

;;;; Lifecycle for examples that write prepared data and backend predictions.

(require :sb-posix)

(defpackage #:cl-gbdt-example-lifecycle
  (:use #:cl)
  (:export #:with-preparation #:with-training #:with-evaluation #:with-report-outputs
           #:call-with-preparation #:call-with-training #:call-with-evaluation
           #:call-with-report-outputs #:call-with-lock #:custom-current-p
           #:make-layout #:example-root #:publish-file
           #:*training-output-directory* #:*report-output-directory*
           #:*preparation-output-directory* #:*prepared-generation*))

(in-package #:cl-gbdt-example-lifecycle)

(defvar *training-output-directory* nil
  "Private output directory used while both backend predictions are produced.")

(defvar *report-output-directory* nil
  "Private output directory used while a report pair is produced.")

(defvar *evaluation-generation* nil
  "Prepared generation held under the current evaluation lock.")

(defvar *preparation-output-directory* nil
  "Private directory used while a staged preparation produces its files.")

(defvar *prepared-generation* nil
  "Generation being prepared, bound while the body of a preparation runs.")

(defstruct (layout (:constructor make-layout
                       (&key (name "Example") (prefix "example")
                             (lock "output/.example.lock")
                             (markers '((:prepared "data/.prepared-generation")
                                        (:standard "output/.trained-generation")
                                        (:custom "output/.custom-generation"))))))
  "Where one kind of example directory keeps its lock, state markers and private files.
NAME starts error messages. PREFIX names private files and directories, such as
output/.PREFIX-training-XXXXXX. LOCK is the lock file, and MARKERS maps each state kind
to (PATH) or (PATH KEY), all relative to the example root. A (PATH) marker holds only a
generation; a (PATH KEY) marker is a property list whose KEY entry is the generation,
written by the preparation itself."
  (name "Example" :read-only t)
  (prefix "example" :read-only t)
  (lock "output/.example.lock" :read-only t)
  (markers nil :read-only t))

(setf (documentation 'make-layout 'function)
      "Return a layout; every keyword defaults to the layout of the shared examples.
See the LAYOUT structure for NAME, PREFIX, LOCK and MARKERS.")

(defvar *default-layout* (make-layout)
  "The layout of the classification and regression examples.")

(defstruct (site (:constructor %make-site (root layout)))
  "One example directory: an absolute directory pathname and its layout."
  (root nil :read-only t)
  (layout nil :read-only t))

(defun example-root (root)
  "Return the directory designator ROOT as an absolute directory pathname.
A relative ROOT is resolved against *DEFAULT-PATHNAME-DEFAULTS* and, if that is relative
too, against the process working directory, so that Lisp file functions and system calls
given the result name the same files."
  (let ((path (merge-pathnames (uiop:ensure-directory-pathname root))))
    (if (uiop:absolute-pathname-p path)
        path
        (merge-pathnames path (uiop:getcwd)))))

(defun ensure-site (root &optional layout)
  "Return ROOT as a site; a directory designator gets LAYOUT, by default the shared one."
  (if (site-p root)
      root
      (%make-site (example-root root) (or layout *default-layout*))))

(defun native (path)
  "Return the operating-system name of PATH, as a system call needs it."
  ;; NAMESTRING escapes *, ?, [ and backslash, which would name another file.
  (sb-ext:native-namestring path))

(defun native-directory (name)
  "Parse the operating-system directory NAME, such as one MKDTEMP returned."
  (sb-ext:parse-native-namestring name nil *default-pathname-defaults* :as-directory t))

(defun open-lock (path read-only-ok)
  "Open the lock file PATH, creating it; with READ-ONLY-OK, accept an existing read-only one."
  (handler-case
      (sb-posix:open (native path) (logior sb-posix:o-rdwr sb-posix:o-creat) #o600)
    (sb-posix:syscall-error (original)
      (unless (and read-only-ok
                   (member (sb-posix:syscall-errno original)
                           (list sb-posix:eacces sb-posix:eperm sb-posix:erofs)))
        (error original))
      ;; Exclusive flock works on a read-only descriptor too.
      (handler-case (sb-posix:open (native path) sb-posix:o-rdonly)
        (sb-posix:syscall-error (fallback)
          ;; If the file is absent, keep the reason its creation failed.
          (error (if (= (sb-posix:syscall-errno fallback) sb-posix:enoent)
                     original
                     fallback)))))))

(defun call-with-lock (root function &key layout read-only-ok)
  "Call FUNCTION while holding the nonblocking lock of the example directory ROOT.
ROOT is a directory designator or a site, and LAYOUT names the lock file. Another holder
makes this fail at once; the lock is released when the process exits. READ-ONLY-OK
accepts an existing lock file that cannot be opened for writing, which is enough to
evaluate a read-only result."
  (let* ((site (ensure-site root layout))
         (layout (site-layout site))
         (path (merge-pathnames (layout-lock layout) (site-root site))))
    (ensure-directories-exist path)
    (ensure-directories-exist (merge-pathnames "data/" (site-root site)))
    (let ((fd (open-lock path read-only-ok)))
      (unwind-protect
           (progn
             ;; LOCK_EX | LOCK_NB. Never unlink this file while it can be in use:
             ;; that could give a second process a different lock inode.
             (unless (zerop (cffi:foreign-funcall "flock" :int fd :int 6 :int))
               (error "~A directory is busy or cannot be locked: ~A"
                      (layout-name layout) (native (site-root site))))
             (funcall function))
        (sb-posix:close fd)))))

(defun marker-spec (site kind)
  (or (assoc kind (layout-markers (site-layout site)))
      (error "~A directories record no ~A state." (layout-name (site-layout site)) kind)))

(defun marker-path (site kind)
  (merge-pathnames (second (marker-spec site kind)) (site-root site)))

(defun remove-marker (site kind)
  ;; UNLINK does not follow a symbolic link and also sees dangling links.
  (handler-case (sb-posix:unlink (native (marker-path site kind)))
    (sb-posix:syscall-error (condition)
      (unless (= (sb-posix:syscall-errno condition) sb-posix:enoent)
        (error condition)))))

(defun new-generation ()
  "Return 128 random bits from the operating system as 32 hexadecimal digits."
  (with-open-file (in "/dev/urandom" :element-type '(unsigned-byte 8))
    (format nil "~{~2,'0X~}" (loop repeat 16 collect (read-byte in)))))

(defun valid-generation-p (value)
  (and (stringp value) (= (length value) 32)
       (every (lambda (char) (digit-char-p char 16)) value)))

(defun read-generation (site kind)
  (let ((spec (marker-spec site kind))
         (path (marker-path site kind))
         (name (layout-name (site-layout site))))
    (unless (uiop:file-exists-p path)
      (error "~A ~A state is not current in ~A; rerun prepare and training."
             name kind (native (site-root site))))
    (with-open-file (in path)
      (let* ((*read-eval* nil)
             (value (read in nil :eof))
             (generation (if (third spec)
                             (and (listp value)
                                   ;; A malformed plist is invalid state, not a crash.
                                   (handler-case (getf value (third spec))
                                     (error () nil)))
                             value)))
        (unless (and (valid-generation-p generation) (eq (read in nil :eof) :eof))
          (error "Invalid ~A ~A state in ~A" name kind (native path)))
        generation))))

(defun path-present-p (path)
  ;; LSTAT includes dangling symlinks, unlike PROBE-FILE.
  (handler-case (progn (sb-posix:lstat (native path)) t)
    (sb-posix:syscall-error (condition)
      (if (= (sb-posix:syscall-errno condition) sb-posix:enoent)
          nil
          (error condition)))))

(defun rename-output (from to)
  (sb-posix:rename (native from) (native to)))

(defun publish-file (path temporary-name writer)
  "Replace PATH atomically with a file that WRITER writes.
WRITER is called with the pathname of a new private file in PATH's directory, made from
the mkstemp template TEMPORARY-NAME. That file is renamed to PATH once WRITER returns,
and removed if anything fails first."
  (multiple-value-bind (fd name)
      (sb-posix:mkstemp (native (merge-pathnames temporary-name
                                                 (uiop:pathname-directory-pathname path))))
    (let ((published nil))
      (unwind-protect
           (progn
             (sb-posix:close fd)
             (setf fd nil)
             (let ((temporary (sb-ext:parse-native-namestring name)))
               (funcall writer temporary)
               (rename-output temporary path))
             (setf published t))
        (when fd
          (handler-case (sb-posix:close fd)
            (sb-posix:syscall-error () nil)))
        (unless published
          (handler-case (sb-posix:unlink name)
            (sb-posix:syscall-error () nil)))))))

(defun publish-generation (site kind generation)
  "Publish GENERATION as SITE's KIND marker through a private file in its directory."
  (when (third (marker-spec site kind))
    (error "The ~A marker is written by its preparation." kind))
  (publish-file (marker-path site kind)
                (format nil ".~A-generation-XXXXXX" (layout-prefix (site-layout site)))
                (lambda (temporary)
                  (with-open-file (out temporary :direction :output :if-exists :supersede
                                                 :external-format :utf-8)
                    (let ((*print-readably* t)) (write generation :stream out))
                    (terpri out)))))

(defun remove-stage (stage)
  "Remove STAGE. A failure is only a warning, so that it cannot replace the outcome."
  (handler-case (uiop:delete-directory-tree stage :validate t)
    (error (condition)
      (warn "Could not remove the private directory ~A: ~A" (native stage) condition))))

(defun call-with-staging (site template function)
  "Call FUNCTION with a new private directory and a function of one boolean argument.
The directory is made from the mkdtemp TEMPLATE, relative to SITE's root. FUNCTION
calls the second argument with true while the directory must be kept and with false
once it need not be; afterwards the directory is removed unless it must be kept."
  (let ((stage (native-directory
                (sb-posix:mkdtemp (native (merge-pathnames template (site-root site))))))
        (keep nil))
    (unwind-protect
         (funcall function stage (lambda (value) (setf keep value)))
      (unless keep (remove-stage stage)))))

(defun restore-outputs (stage backups installing invalidate keep-stage)
  "Undo an unfinished publication, reporting STAGE if its backups are still needed."
  (let ((failures nil)
        (restored nil))
    (unwind-protect
         (progn
           (let ((invalidated
                   (handler-case
                       (progn
                         ;; FINISH may have published the marker before an abort.
                         (when invalidate (funcall invalidate))
                         t)
                     (error (condition)
                       (push condition failures)
                       nil))))
             ;; If no marker can be invalidated, do not replace a newly published
             ;; set of files with an older set under it.
             (when invalidated
               (flet ((try (function)
                        (handler-case (funcall function)
                          (error (condition) (push condition failures)))))
                 (dolist (entry installing)
                   (try (lambda ()
                          ;; A missing staged source means its rename succeeded.
                          (when (and (not (path-present-p (car entry)))
                                     (path-present-p (cdr entry)))
                            (sb-posix:unlink (native (cdr entry)))))))
                 (dolist (entry backups)
                   (try (lambda ()
                          (when (path-present-p (cdr entry))
                            (rename-output (cdr entry) (car entry)))))))))
           (when failures
             (error "Output publication failed; recovery failed (~{~A~^; ~}). Backups: ~A"
                    failures (native stage)))
           (setf restored t)
           (when keep-stage (funcall keep-stage nil)))
      (unless restored
        (format *error-output* "Output recovery incomplete. Backups: ~A~%" (native stage))))))

(defun publish-outputs (site stage names &key (subdirectory "output/") finish invalidate
                                             keep-stage)
  "Move the files NAMES from STAGE into SUBDIRECTORY of SITE's root as one publication.
FINISH, if given, is called once they are all in place, typically to publish a marker.
A failure or nonlocal exit before FINISH returns calls INVALIDATE, if given, and then
restores the previous files. Their backups are kept in STAGE: KEEP-STAGE, if given, is
called with true while they are needed and with false once they are not."
  (let ((backups nil)
        (installing nil)
        (complete nil))
    (flet ((target (name)
             (merge-pathnames (format nil "~A~A" subdirectory name) (site-root site))))
      (unwind-protect
           (progn
             ;; Keep backups until both publication and its recovery have finished.
             (when keep-stage (funcall keep-stage t))
             (dolist (name names)
               (let ((source (merge-pathnames name stage)))
                 (unless (uiop:file-exists-p source)
                   (error "Nothing to publish as ~A: ~A was not produced"
                          (native (target name)) (native source)))
                 (when (uiop:directory-exists-p (uiop:ensure-directory-pathname (target name)))
                   (error "Publication destination is a directory: ~A" (native (target name))))))
             (dolist (name names)
               (let ((backup (merge-pathnames (format nil ".previous-~A" name) stage)))
                 (when (path-present-p (target name))
                   ;; Record intent before renaming, so THROW after the rename
                   ;; cannot lose the only reference to an older file.
                   (push (cons (target name) backup) backups)
                   (rename-output (target name) backup))))
             (dolist (name names)
               (let ((source (merge-pathnames name stage)))
                 (push (cons source (target name)) installing)
                 (rename-output source (target name))))
             (when finish (funcall finish))
             (setf complete t)
             (when keep-stage (funcall keep-stage nil)))
        (unless complete
          (restore-outputs stage backups installing invalidate keep-stage))))))

(defun invalidate-state (site kind)
  "Remove SITE's KIND marker or, failing that, its prepared marker."
  (handler-case (remove-marker site kind)
    (error (condition)
      ;; If a just-written marker cannot be removed, invalidate the preparation.
      (handler-case (remove-marker site :prepared)
        (error (fallback)
          (error "Cannot invalidate either state (~A; ~A)" condition fallback))))))

(defun check-prepared (site generation activity)
  (unless (string= generation (read-generation site :prepared))
    (error "~A data changed during ~A in ~A"
           (layout-name (site-layout site)) activity (native (site-root site)))))

(defun record-preparation (site generation)
  "Publish SITE's prepared marker, or check that the preparation itself recorded it."
  (if (third (marker-spec site :prepared))
      (unless (equal generation (handler-case (read-generation site :prepared)
                                  (error () nil)))
        (error "~A preparation did not record its generation in ~A"
               (layout-name (site-layout site)) (native (marker-path site :prepared))))
      (publish-generation site :prepared generation)))

(defun call-with-preparation (root function &key layout names)
  "Call FUNCTION to prepare the example directory ROOT under its lock.
Every state marker is removed first, so a failure or nonlocal exit leaves evaluation
disabled; a new prepared generation, bound to *PREPARED-GENERATION* meanwhile, is
recorded only after FUNCTION returns. With NAMES, FUNCTION writes those files to
*PREPARATION-OUTPUT-DIRECTORY*, a private directory, and they are moved into data/
together with the marker. When LAYOUT gives the prepared marker a key, FUNCTION writes
that marker itself, recording *PREPARED-GENERATION* under the key."
  (let ((site (ensure-site root layout)))
    (call-with-lock
     site
     (lambda ()
       (dolist (marker (layout-markers (site-layout site)))
         (remove-marker site (first marker)))
       (let* ((generation (new-generation))
              (*prepared-generation* generation))
         (if (null names)
             (multiple-value-prog1 (funcall function)
               (record-preparation site generation))
             (call-with-staging
              site (format nil "data/.~A-preparation-XXXXXX" (layout-prefix (site-layout site)))
              (lambda (stage keep-stage)
                (let ((results (let ((*preparation-output-directory* stage))
                                 (multiple-value-list (funcall function)))))
                  (publish-outputs site stage names
                                   :subdirectory "data/"
                                   :finish (lambda () (record-preparation site generation))
                                   :invalidate (lambda () (remove-marker site :prepared))
                                   :keep-stage keep-stage)
                  (values-list results))))))))))

(defun call-with-training (root kind names function &key layout)
  "Call FUNCTION to train against the current preparation of ROOT under its lock.
The KIND marker is removed first and published again, with the prepared generation, only
after FUNCTION returns with that preparation unchanged. With NAMES, FUNCTION writes those
files to *TRAINING-OUTPUT-DIRECTORY*, a private directory, and they are moved into
output/ together with the marker; if that fails, the previous files are restored.
Without NAMES, FUNCTION writes nothing that needs publication."
  (let ((site (ensure-site root layout)))
    (call-with-lock
     site
     (lambda ()
       (remove-marker site kind)
       (let ((generation (read-generation site :prepared)))
         (if (null names)
             ;; Useful for isolated fixtures which already supply synthetic scores.
             (let ((*training-output-directory* nil))
               (multiple-value-prog1 (funcall function)
                 (check-prepared site generation "training")
                 (publish-generation site kind generation)))
             (call-with-staging
              site (format nil "output/.~A-training-XXXXXX" (layout-prefix (site-layout site)))
              (lambda (stage keep-stage)
                (let ((results (let ((*training-output-directory* stage))
                                 (multiple-value-list (funcall function)))))
                  (check-prepared site generation "training")
                  (publish-outputs site stage names
                                   :finish (lambda ()
                                             (check-prepared site generation "publication")
                                             (publish-generation site kind generation))
                                   :invalidate (lambda () (invalidate-state site kind))
                                   :keep-stage keep-stage)
                  (values-list results))))))))))

(defun custom-current-p (root &key layout)
  "Whether both optional custom predictions belong to ROOT's current preparation."
  (handler-case
      (let ((site (ensure-site root layout)))
        (string= (read-generation site :prepared) (read-generation site :custom)))
    (error () nil)))

(defun current-generation (site trained)
  "Return SITE's prepared generation, which must also be trained when TRAINED is true."
  (let ((generation (read-generation site :prepared)))
    (when (and trained (not (string= generation (read-generation site :standard))))
      (error "~A predictions are not current in ~A; rerun training."
             (layout-name (site-layout site)) (native (site-root site))))
    generation))

(defun call-with-report-outputs (root names function &key layout)
  "Call FUNCTION to write the report files NAMES to *REPORT-OUTPUT-DIRECTORY*.
That private directory's files are moved into ROOT's output/ together once FUNCTION
returns, and the previous files are restored if that fails. Valid only while an
evaluation of ROOT holds its lock."
  (unless *evaluation-generation*
    (error "Report publication requires the evaluation lock"))
  (let ((site (ensure-site root layout)))
    (call-with-staging
     site (format nil "output/.~A-report-XXXXXX" (layout-prefix (site-layout site)))
     (lambda (stage keep-stage)
       (let ((results (let ((*report-output-directory* stage))
                        (multiple-value-list (funcall function)))))
         (publish-outputs site stage names
                          :finish (lambda ()
                                    (unless (string= *evaluation-generation*
                                                     (current-generation site t))
                                      (error "~A data changed during report publication in ~A"
                                             (layout-name (site-layout site))
                                             (native (site-root site)))))
                          :keep-stage keep-stage)
         (values-list results))))))

(defun call-with-evaluation (root function &key layout (trained t))
  "Call FUNCTION to evaluate ROOT's current results under its lock.
Its output is printed only if the preparation, and when TRAINED the predictions, are
still current once it returns. An existing read-only lock file and directory suffice."
  (let ((site (ensure-site root layout)))
    (call-with-lock
     site
     (lambda ()
       (let ((generation (current-generation site trained))
             ;; Buffer the report until a second generation check has succeeded.
             (output (make-string-output-stream))
             (values nil))
         (let ((*standard-output* output)
               (*evaluation-generation* generation))
           (setf values (multiple-value-list (funcall function))))
         (unless (string= generation (current-generation site trained))
           (error "~A data changed during evaluation in ~A"
                  (layout-name (site-layout site)) (native (site-root site))))
         (write-string (get-output-stream-string output))
         (values-list values)))
     :read-only-ok t)))

(defmacro with-preparation ((root) &body body)
  "Run BODY as the preparation of the example directory ROOT under its lock.
Old state markers are removed first and a new prepared generation is published only if
BODY returns; see CALL-WITH-PREPARATION."
  `(call-with-preparation ,root (lambda () ,@body)))

(defmacro with-training ((root &optional (kind :standard) names) &body body)
  "Run BODY as training of KIND against ROOT's current preparation, under its lock.
BODY writes the files NAMES to *TRAINING-OUTPUT-DIRECTORY*; they and the KIND marker
are published only if BODY returns. See CALL-WITH-TRAINING."
  `(call-with-training ,root ,kind ,names (lambda () ,@body)))

(defmacro with-evaluation ((root) &body body)
  "Run BODY as an evaluation of ROOT's current predictions, under its lock.
What BODY prints is shown only if the data and predictions are still current when it
returns; see CALL-WITH-EVALUATION."
  `(call-with-evaluation ,root (lambda () ,@body)))

(defmacro with-report-outputs ((root names) &body body)
  "Run BODY, within WITH-EVALUATION of ROOT, to write the report files NAMES.
BODY writes them to *REPORT-OUTPUT-DIRECTORY*; they replace ROOT's previous report
files together only if BODY returns. See CALL-WITH-REPORT-OUTPUTS."
  `(call-with-report-outputs ,root ,names (lambda () ,@body)))
