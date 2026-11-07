(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests-fetch.lisp to execute this test in an isolated process."))

(require :sb-posix)

(load (merge-pathnames "fetch-data.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

(defun test-file-bytes (path)
  (with-open-file (in path :element-type '(unsigned-byte 8))
    (let ((result (make-array (file-length in) :element-type '(unsigned-byte 8))))
      (assert (= (read-sequence result in) (length result)))
      result)))

(defun write-test-bytes (path bytes)
  (ensure-directories-exist path)
  (with-open-file (out path :direction :output :if-exists :supersede
                            :element-type '(unsigned-byte 8))
    (write-sequence bytes out)))

(defparameter *sha256-of-1-2-3*
  "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
  "SHA-256 of the octets 1, 2, 3, taken from sha256sum rather than from the code under test.")

(defparameter *sha256-of-1*
  "4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a"
  "SHA-256 of the single octet 1, taken from sha256sum rather than from the code under test.")

(defun signals-digest-mismatch-p (thunk)
  "True when THUNK signals DIGEST-MISMATCH; any other error propagates and fails the run."
  (handler-case (progn (funcall thunk) nil)
    (digest-mismatch () t)))

(defun test-fetch-file-digests (root)
  (let ((corrupted (merge-pathnames "corrupted.bin" root))
        (corrupted-part (merge-pathnames "corrupted.bin.part" root))
        (cached (merge-pathnames "cached.bin" root))
        (calls 0))
    (unwind-protect
         (flet ((fetcher (url target)
                  (declare (ignore url))
                  (incf calls)
                  (write-test-bytes target #(1 2 3))))
           ;; A response that is not the recorded file must not reach its final name.
           (assert (signals-digest-mismatch-p
                    (lambda () (fetch-file "fixture" corrupted *sha256-of-1* #'fetcher))))
           (assert (= calls 1))
           (assert (not (probe-file corrupted)))
           (assert (not (probe-file corrupted-part)))
           ;; A bad file kept from an earlier run is reported, not used, and not re-fetched
           ;; over: it is the user's to look at and remove.
           (write-test-bytes cached #(1))
           (assert (signals-digest-mismatch-p
                    (lambda () (fetch-file "fixture" cached *sha256-of-1-2-3* #'fetcher))))
           (assert (= calls 1))
           (assert (equalp (test-file-bytes cached) #(1)))
           ;; A kept file that matches is used as it is.
           (assert (equal (fetch-file "fixture" cached *sha256-of-1* #'fetcher) cached))
           (assert (= calls 1)))
      (dolist (path (list corrupted corrupted-part cached))
        (when (probe-file path) (delete-file path))))))

(defun write-marker-asd (asd marker)
  "Write a lisp-xl .asd to ASD that records in MARKER each time ASDF loads it."
  (ensure-directories-exist asd)
  (with-open-file (out asd :direction :output :if-exists :supersede)
    (format out "(progn (with-open-file (out ~S :direction :output ~
                 :if-exists :supersede) (write-line \"loaded\" out)) ~
                 (asdf:defsystem \"lisp-xl\" :components ()))~%"
            (namestring marker))))

(defun call-with-every-digest-accepted (function)
  "Call FUNCTION while CHECK-SHA256 accepts any file, so fixtures stand in for pinned ones."
  (let ((original (symbol-function 'check-sha256)))
    (unwind-protect
         (progn
           (setf (symbol-function 'check-sha256)
                 (lambda (path digest) (declare (ignore path digest)) t))
           (funcall function))
      (setf (symbol-function 'check-sha256) original))))

(defun write-retail-reader-fixture (cache marker)
  "Fill CACHE with every pinned file name, the .asd recording its loads in MARKER."
  (dolist (file '("lisp-xl.asd" "README.markdown" "src/package.lisp" "src/util.lisp"
                  "src/metadata.lisp" "src/col-process.lisp" "src/main.lisp"
                  "src/csv-export.lisp"))
    (let ((path (merge-pathnames file cache)))
      (if (equal file "lisp-xl.asd")
          (write-marker-asd path marker)
          (progn
            (ensure-directories-exist path)
            (with-open-file (out path :direction :output :if-exists :supersede)
              (write-line "fixture" out)))))))

(defun test-retail-reader-cache-integrity (root)
  "This checkout's managed cache is verified before ASDF can load any of it."
  (let* ((examples (merge-pathnames "retail-fixture/" root))
         (cache (merge-pathnames "online-retail/data/lisp-xl/" examples))
         (asd (merge-pathnames "lisp-xl.asd" cache))
         (marker (merge-pathnames "asd-loaded.txt" examples))
         (*examples-root* examples)
         (asdf:*central-registry* (cons cache asdf:*central-registry*))
         (old-system (asdf:registered-system :lisp-xl)))
    (unwind-protect
         (progn
           ;; This real ASDF fixture is intentionally not the pinned source. Its marker
           ;; makes it observable if ASDF reads it before integrity verification.
           (write-marker-asd asd marker)
           (asdf:clear-system :lisp-xl)
           (asdf:load-asd asd)
           (when (probe-file marker) (delete-file marker))
           ;; A previously registered copy must not make the next check skip the cache.
           (assert (signals-digest-mismatch-p #'ensure-retail-reader))
           (assert (not (probe-file marker)))
           ;; Let the fixture .asd stand in for a verified one, then corrupt a source.
           ;; Every kept file is checked before any missing one would be downloaded.
           (let ((source (merge-pathnames "src/main.lisp" cache))
                 (original (symbol-function 'check-sha256)))
             (ensure-directories-exist source)
             (with-open-file (out source :direction :output :if-exists :supersede)
               (write-line "bad source" out))
             (unwind-protect
                  (progn
                    (setf (symbol-function 'check-sha256)
                          (lambda (path digest)
                            (unless (equal path asd) (funcall original path digest))))
                    (assert (handler-case (progn (ensure-retail-reader) nil)
                              (digest-mismatch (condition)
                                (equal (digest-mismatch-path condition) source))))
                    (assert (not (probe-file marker))))
               (setf (symbol-function 'check-sha256) original)
               (delete-file source)))
           ;; A custom ASDF searcher may load an .asd before returning a system.
           ;; It must not get an opportunity to load an unverified managed reader.
           (asdf:clear-system :lisp-xl)
           (let ((busy nil))
             (let ((asdf:*system-definition-search-functions*
                     (cons (lambda (name)
                             (when (and (string-equal (string name) "lisp-xl") (not busy))
                               (setf busy t)
                               (unwind-protect
                                    (progn (asdf:load-asd asd)
                                           (asdf:registered-system :lisp-xl))
                                 (setf busy nil))))
                           asdf:*system-definition-search-functions*)))
               (assert (signals-digest-mismatch-p #'ensure-retail-reader))
               (assert (not (probe-file marker)))))
           ;; In a fresh image ASDF must not evaluate a corrupt cached .asd to discover it.
           (asdf:clear-system :lisp-xl)
           (assert (signals-digest-mismatch-p #'ensure-retail-reader))
           (assert (not (probe-file marker)))
           ;; A registered external system does not let the managed cache go unchecked.
           (asdf:clear-system :lisp-xl)
           (let ((external (merge-pathnames "external/lisp-xl.asd" examples)))
             (ensure-directories-exist external)
             (with-open-file (out external :direction :output :if-exists :supersede)
               (write-line "(asdf:defsystem \"lisp-xl\" :components ())" out))
             (asdf:load-asd external)
             (assert (signals-digest-mismatch-p #'ensure-retail-reader))
             (assert (not (probe-file marker)))
             ;; Without a managed cache, the registered external installation is used.
             (uiop:delete-directory-tree cache :validate t)
             (ensure-retail-reader)
             (assert (equal (truename (uiop:pathname-directory-pathname external))
                            (truename (asdf:component-pathname
                                       (asdf:registered-system :lisp-xl)))))
             ;; It also remains usable when ASDF must discover it in a fresh image.
             (asdf:clear-system :lisp-xl)
             (let ((asdf:*central-registry*
                     (cons (uiop:pathname-directory-pathname external)
                           asdf:*central-registry*)))
               (ensure-retail-reader)
               (assert (equal (truename (uiop:pathname-directory-pathname external))
                              (truename (asdf:component-pathname
                                         (asdf:registered-system :lisp-xl))))))
             (assert (not (probe-file marker)))
             (assert (not (probe-file cache)))))
      (asdf:clear-system :lisp-xl)
      (when old-system (asdf/system-registry:register-system old-system))
      (when (probe-file examples) (uiop:delete-directory-tree examples :validate t)))))

(defun test-retail-reader-symlink-cache (root)
  "A managed cache that is a symbolic link to another directory is still verified."
  (let* ((shared (merge-pathnames "shared-reader/" root))
         (checkout (merge-pathnames "checkout-a/" root))
         (alias (merge-pathnames "online-retail/data/lisp-xl/" checkout))
         (marker (merge-pathnames "symlink-asd-loaded.txt" root))
         (*examples-root* checkout)
         (asdf:*central-registry* (cons alias asdf:*central-registry*))
         (old-system (asdf:registered-system :lisp-xl)))
    (unwind-protect
         (progn
           (write-marker-asd (merge-pathnames "lisp-xl.asd" shared) marker)
           (ensure-directories-exist (merge-pathnames "online-retail/data/placeholder" checkout))
           (sb-posix:symlink (sb-ext:native-namestring shared)
                             (string-right-trim "/" (sb-ext:native-namestring alias)))
           (asdf:clear-system :lisp-xl)
           (assert (signals-digest-mismatch-p #'ensure-retail-reader) ()
                   "A symbolic-link managed cache was not checked")
           (assert (not (probe-file marker)))
           ;; Once its files pass verification, the cache is loaded through the link.
           (write-retail-reader-fixture shared marker)
           (call-with-every-digest-accepted #'ensure-retail-reader)
           (assert (probe-file marker))
           (assert (equal (truename shared)
                          (truename (asdf:component-pathname
                                     (asdf:registered-system :lisp-xl))))))
      (asdf:clear-system :lisp-xl)
      (when old-system (asdf/system-registry:register-system old-system)))))

(defun test-retail-reader-external-precedence (root)
  "Without a managed cache, a registered external system wins over any other copy."
  (let* ((cache (merge-pathnames "precedence/online-retail/data/lisp-xl/" root))
         (external (merge-pathnames "precedence/external/lisp-xl.asd" root))
         (marker (merge-pathnames "precedence/cache-loaded.txt" root))
         (*examples-root* (merge-pathnames "precedence/another-checkout/" root))
         (asdf:*central-registry* (cons cache asdf:*central-registry*))
         (old-system (asdf:registered-system :lisp-xl)))
    (unwind-protect
         (progn
           (write-retail-reader-fixture cache marker)
           (ensure-directories-exist external)
           (with-open-file (out external :direction :output :if-exists :error)
             (write-line "(asdf:defsystem \"lisp-xl\" :components ())" out))
           (asdf:clear-system :lisp-xl)
           (asdf:load-asd external)
           ;; The fake cache stands in for a valid pinned checkout so this tests
           ;; source selection, rather than digest rejection.
           (call-with-every-digest-accepted #'ensure-retail-reader)
           (assert (equal (truename (uiop:pathname-directory-pathname external))
                          (truename (asdf:component-pathname
                                     (asdf:registered-system :lisp-xl)))))
           (assert (not (probe-file marker)))
           ;; This checkout's own verified cache is preferred to the external system.
           (let ((*examples-root* (merge-pathnames "precedence/" root)))
             (call-with-every-digest-accepted #'ensure-retail-reader))
           (assert (probe-file marker))
           (assert (equal (truename cache)
                          (truename (asdf:component-pathname
                                     (asdf:registered-system :lisp-xl))))))
      (asdf:clear-system :lisp-xl)
      (when old-system (asdf/system-registry:register-system old-system)))))

(defun signals-unsafe-retail-cache-p (thunk)
  (handler-case (progn (funcall thunk) nil)
    (error (condition)
      (search "outside the managed reader cache" (princ-to-string condition)))))

(defun test-retail-reader-asd-symlink (root)
  "An .asd symlink inside the managed cache cannot bypass verification."
  (let* ((examples (merge-pathnames "file-link/" root))
         (cache (merge-pathnames "online-retail/data/lisp-xl/" examples))
         (linked (merge-pathnames "lisp-xl.asd" cache))
         (shared (merge-pathnames "shared/lisp-xl.asd" examples))
         (marker (merge-pathnames "asd-loaded.txt" examples))
         (*examples-root* examples)
         (asdf:*central-registry* (list cache))
         (old-system (asdf:registered-system :lisp-xl)))
    (unwind-protect
         (progn
           (write-retail-reader-fixture cache marker)
           (delete-file linked)
           (write-marker-asd shared marker)
           (sb-posix:symlink (sb-ext:native-namestring shared)
                             (sb-ext:native-namestring linked))
           (asdf:clear-system :lisp-xl)
           (assert (signals-unsafe-retail-cache-p #'ensure-retail-reader) ()
                   "Symlinked ASD was not rejected")
           (assert (not (probe-file marker)))
           ;; ASDF would read components beside the link's target, which were never
           ;; checked, so the link is refused even when its own content would pass.
           (assert (signals-unsafe-retail-cache-p
                    (lambda () (call-with-every-digest-accepted #'ensure-retail-reader)))
                   () "Symlinked ASD with accepted content was not rejected")
           (assert (not (probe-file marker))))
      (asdf:clear-system :lisp-xl)
      (when old-system (asdf/system-registry:register-system old-system)))))

(defun run-fetch-tests ()
  (let* ((root (uiop:ensure-directory-pathname
                (sb-posix:mkdtemp
                 (namestring (merge-pathnames "cl-gbdt-fetch-tests-XXXXXX"
                                              (uiop:temporary-directory))))))
         (download (merge-pathnames "download.bin" root))
         (download-part (merge-pathnames "download.bin.part" root))
         (bad-download (merge-pathnames "failed.bin" root))
         (bad-part (merge-pathnames "failed.bin.part" root))
         (bz2 (merge-pathnames "hello.bz2" root))
         (plain (merge-pathnames "hello.txt" root))
         (bad-bz2 (merge-pathnames "bad.bz2" root))
         (bad-plain (merge-pathnames "bad.txt" root))
         (bad-plain-part (merge-pathnames "bad.txt.part" root))
         (zip (merge-pathnames "fixture.zip" root))
         (extracted (merge-pathnames "result.txt" root))
         (calls 0))
    (unwind-protect
         (progn
           (fetch-file "fixture" download *sha256-of-1-2-3*
                       (lambda (url target)
                         (declare (ignore url))
                         (incf calls)
                         (write-test-bytes target #(1 2 3))))
           (assert (equalp (test-file-bytes download) #(1 2 3)))
           (assert (not (probe-file download-part)))
           (fetch-file "fixture" download *sha256-of-1-2-3*
                       (lambda (url target)
                         (declare (ignore url target))
                         (incf calls)))
           (assert (= calls 1))
           (assert (handler-case
                       (progn
                         (fetch-file "failing fixture" bad-download *sha256-of-1*
                                     (lambda (url target)
                                       (declare (ignore url))
                                       (write-test-bytes target #(1))
                                       (error "simulated failure")))
                         nil)
                     (error () t)))
           (assert (not (probe-file bad-download)))
           (assert (not (probe-file bad-part)))
           (test-fetch-file-digests root)
           (test-retail-reader-cache-integrity root)
           (test-retail-reader-symlink-cache root)
           (test-retail-reader-asd-symlink root)
           (test-retail-reader-external-precedence root)
           (write-test-bytes bz2
                             #(66 90 104 57 49 65 89 38 83 89 193 192 128 226 0 0
                               1 65 0 0 16 2 68 160 0 48 205 0 195 70 41 151
                               23 114 69 56 80 144 193 192 128 226))
           (decompress-file bz2 plain :bzip2)
           (assert (equalp (test-file-bytes plain) #(104 101 108 108 111 10)))
           (write-test-bytes bad-bz2 #(66 90 104))
           (assert (handler-case (progn (decompress-file bad-bz2 bad-plain :bzip2) nil)
                     (error () t)))
           (assert (not (probe-file bad-plain)))
           (assert (not (probe-file bad-plain-part)))
           (zip:with-output-to-zipfile (writer zip :if-exists :supersede)
             (zip:write-zipentry writer "a.txt" plain))
           (extract-zip-members zip root '(("a.txt" . "result.txt")))
           (assert (equalp (test-file-bytes extracted) #(104 101 108 108 111 10))))
      (uiop:delete-directory-tree root :validate t)))
  (format t "Download and archive utility tests passed.~%"))

(run-fetch-tests)
