(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests-fetch-atomicity.lisp to execute this test in an isolated process."))

(require :sb-posix)
(load (merge-pathnames "fetch-data.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

;; These tests exercise publication failures and interleavings in a private tree.
(defparameter *atomicity-sha256*
  "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81")
(defparameter *atomicity-sentinel* #(42 99 7))
(defparameter *atomicity-bzip2*
  #(66 90 104 57 49 65 89 38 83 89 193 192 128 226 0 0
    1 65 0 0 16 2 68 160 0 48 205 0 195 70 41 151
    23 114 69 56 80 144 193 192 128 226))
(defparameter *atomicity-hello* #(104 101 108 108 111 10))

;; A bzip2-compressed TAR containing the two required paths, each with "hello\n".
(defparameter *atomicity-bzip2-tar*
  #(66 90 104 57 49 65 89 38 83 89 221 123 214 105 0 0 152 251 128 193 16 0 64 64
    1 255 128 48 0 238 101 159 0 8 136 32 0 146 5 84 41 234 122 64 208 104 30 160
    104 205 64 170 65 38 154 25 26 26 0 208 31 190 46 38 178 64 149 98 33 10 107
    205 51 6 44 84 104 66 18 102 156 118 48 85 111 2 101 137 12 178 162 40 73 34
    238 41 178 85 69 33 107 98 136 201 18 102 182 10 81 114 231 4 238 38 199 244
    156 164 236 170 213 248 98 165 152 213 147 40 180 52 206 204 214 134 107 36
    193 8 31 197 220 145 78 20 36 55 94 245 154 64))

(defun atomicity-write (path bytes)
  (ensure-directories-exist path)
  (with-open-file (out path :direction :output :if-exists :supersede
                            :element-type '(unsigned-byte 8))
    (write-sequence bytes out)))

(defun atomicity-bytes (path)
  (when (probe-file path)
    (with-open-file (in path :element-type '(unsigned-byte 8))
      (let ((bytes (make-array (file-length in) :element-type '(unsigned-byte 8))))
        (assert (= (read-sequence bytes in) (length bytes)))
        bytes))))

(defun atomicity-part (path)
  (pathname (concatenate 'string (namestring path) ".part")))

(defun atomicity-make-zip (root)
  (let ((source (merge-pathnames "input.txt" root))
        (zip (merge-pathnames "fixture.zip" root)))
    (atomicity-write source *atomicity-hello*)
    (zip:with-output-to-zipfile (writer zip :if-exists :error)
      (zip:write-zipentry writer "input.txt" source))
    zip))

(defun atomicity-make-tar (root)
  (let ((source-root (merge-pathnames "tar-input/" root))
        (tar (merge-pathnames "fixture.tar" root))
        (names '("de/train_de.csv" "de/test_de.csv")))
    (dolist (name names)
      (atomicity-write (merge-pathnames name source-root) *atomicity-hello*))
    (let ((*default-pathname-defaults* source-root))
      (archive:with-open-archive (out tar :direction :output :if-exists :error)
        (dolist (name names)
          (let ((entry (archive:create-entry-from-pathname out (pathname name))))
            (with-open-file (in (merge-pathnames name source-root)
                                :element-type '(unsigned-byte 8))
              (archive:write-entry-to-archive out entry :stream in))))
        (archive:finalize-archive out)))
    tar))

(defun atomicity-fetch-keeps-part (root)
  (let* ((destination (merge-pathnames "fetch/file.bin" root))
         (part (atomicity-part destination)))
    (atomicity-write part *atomicity-sentinel*)
    (fetch-file "fixture" destination *atomicity-sha256*
                (lambda (url target)
                  (declare (ignore url))
                  (atomicity-write target #(1 2 3))))
    (assert (equalp (atomicity-bytes destination) #(1 2 3)))
    (assert (equalp (atomicity-bytes part) *atomicity-sentinel*))))

(defun atomicity-fetch-failure-keeps-part (root)
  (let* ((destination (merge-pathnames "fetch-failure/file.bin" root))
         (part (atomicity-part destination)))
    (atomicity-write part *atomicity-sentinel*)
    (assert (handler-case
                (progn
                  (fetch-file "fixture" destination *atomicity-sha256*
                              (lambda (url target)
                                (declare (ignore url))
                                (atomicity-write target #(1))
                                (error "interrupted download")))
                  nil)
              (error () t)))
    (assert (not (probe-file destination)))
    (assert (equalp (atomicity-bytes part) *atomicity-sentinel*))))

(defun atomicity-decompress-keeps-part (root)
  (let* ((source (merge-pathnames "decompress/source.bz2" root))
         (destination (merge-pathnames "decompress/output.txt" root))
         (part (atomicity-part destination)))
    (atomicity-write source *atomicity-bzip2*)
    (atomicity-write part *atomicity-sentinel*)
    (decompress-file source destination :bzip2)
    (assert (equalp (atomicity-bytes destination) *atomicity-hello*))
    (assert (equalp (atomicity-bytes part) *atomicity-sentinel*))))

(defun atomicity-decompress-failure-keeps-part (root)
  (let* ((source (merge-pathnames "decompress-failure/source.bz2" root))
         (destination (merge-pathnames "decompress-failure/output.txt" root))
         (part (atomicity-part destination)))
    (atomicity-write source #(66 90 104))
    (atomicity-write part *atomicity-sentinel*)
    (assert (handler-case (progn (decompress-file source destination :bzip2) nil)
              (error () t)))
    (assert (not (probe-file destination)))
    (assert (equalp (atomicity-bytes part) *atomicity-sentinel*))))

(defun atomicity-zip-keeps-part (root)
  (let* ((zip (atomicity-make-zip (merge-pathnames "zip-source/" root)))
         (destination (merge-pathnames "zip-output/result.txt" root))
         (part (atomicity-part destination)))
    (atomicity-write part *atomicity-sentinel*)
    (extract-zip-members zip root '(("input.txt" . "zip-output/result.txt")))
    (assert (equalp (atomicity-bytes destination) *atomicity-hello*))
    (assert (equalp (atomicity-bytes part) *atomicity-sentinel*))))

(defun atomicity-tar-keeps-part (root)
  (let* ((tar (atomicity-make-tar (merge-pathnames "tar-source/" root)))
         (destination (merge-pathnames "tar-output/de/train_de.csv" root))
         (part (atomicity-part destination)))
    (atomicity-write part *atomicity-sentinel*)
    (extract-click-tar tar (merge-pathnames "tar-output/" root))
    (assert (equalp (atomicity-bytes destination) *atomicity-hello*))
    (assert (equalp (atomicity-bytes part) *atomicity-sentinel*))))

(defun atomicity-fetch-destination-collision (root)
  ;; The destination can appear after the initial existence check.
  (let* ((destination (merge-pathnames "fetch-collision/file.bin" root))
         (called nil)
         (result
           (handler-case
               (progn
                 (fetch-file "fixture" destination *atomicity-sha256*
                             (lambda (url target)
                               (declare (ignore url))
                               (setf called t)
                               (atomicity-write target #(1 2 3))
                               (atomicity-write destination *atomicity-sentinel*)))
                 :success)
             (digest-mismatch () :digest-mismatch)
             (error () :unexpected-error))))
    (assert called)
    (assert (eq result :digest-mismatch))
    (assert (equalp (atomicity-bytes destination) *atomicity-sentinel*))))

(defun atomicity-collision-during (function destination operation
                                    &optional (other-bytes *atomicity-sentinel*)
                                              (should-succeed nil))
  ;; Create another writer's destination after OPERATION has begun but before
  ;; its output is published. Restore the function even when the operation fails.
  (let ((original (symbol-function function))
        (called nil))
    (unwind-protect
         (progn
           (setf (symbol-function function)
                 (lambda (&rest arguments)
                   (unless called
                     (setf called t)
                     (atomicity-write destination other-bytes))
                   (apply original arguments)))
           (let ((succeeded (handler-case (progn (funcall operation) t) (error () nil))))
             (assert called)
             (assert (equal succeeded should-succeed))
             (assert (equalp (atomicity-bytes destination) other-bytes))))
      (setf (symbol-function function) original))))

(defun atomicity-decompress-destination-collision (root)
  (let ((source (merge-pathnames "decompress-race/source.bz2" root))
        (destination (merge-pathnames "decompress-race/output.txt" root)))
    (atomicity-write source *atomicity-bzip2*)
    (atomicity-collision-during
     'semz.decompress:make-full-decompression-stream destination
     (lambda () (decompress-file source destination :bzip2)))))

(defun atomicity-zip-destination-collision (root)
  (let ((zip (atomicity-make-zip (merge-pathnames "zip-race-source/" root)))
        (destination (merge-pathnames "zip-race-output/result.txt" root)))
    (atomicity-collision-during
     'zip:zipfile-entry-contents destination
     (lambda () (extract-zip-members zip root '(("input.txt" . "zip-race-output/result.txt")))))))

(defun atomicity-tar-destination-collision (root)
  (let ((tar (atomicity-make-tar (merge-pathnames "tar-race-source/" root)))
        (destination (merge-pathnames "tar-race-output/de/train_de.csv" root)))
    (atomicity-collision-during
     'copy-octets destination
     (lambda () (extract-click-tar tar (merge-pathnames "tar-race-output/" root))))))

(defun atomicity-matching-destination-collisions (root)
  "A competing writer with identical content should be accepted."
  (let ((destination (merge-pathnames "fetch-identical/file.bin" root)))
    (fetch-file "fixture" destination *atomicity-sha256*
                (lambda (url target)
                  (declare (ignore url))
                  (atomicity-write target #(1 2 3))
                  (atomicity-write destination #(1 2 3))))
    (assert (equalp (atomicity-bytes destination) #(1 2 3))))
  (let ((source (merge-pathnames "decompress-identical/source.bz2" root))
        (destination (merge-pathnames "decompress-identical/output.txt" root)))
    (atomicity-write source *atomicity-bzip2*)
    (atomicity-collision-during
     'semz.decompress:make-full-decompression-stream destination
     (lambda () (decompress-file source destination :bzip2)) *atomicity-hello* t))
  (let ((zip (atomicity-make-zip (merge-pathnames "zip-identical-source/" root)))
        (destination (merge-pathnames "zip-identical-output/result.txt" root)))
    (atomicity-collision-during
     'zip:zipfile-entry-contents destination
     (lambda ()
       (extract-zip-members zip root '(("input.txt" . "zip-identical-output/result.txt"))))
     *atomicity-hello* t))
  (let ((tar (atomicity-make-tar (merge-pathnames "tar-identical-source/" root)))
        (destination (merge-pathnames "tar-identical-output/de/train_de.csv" root)))
    (atomicity-collision-during
     'copy-octets destination
     (lambda () (extract-click-tar tar (merge-pathnames "tar-identical-output/" root)))
     *atomicity-hello* t)))

(defun atomicity-kasandr-collision (root)
  "A failed conversion must not delete a concurrently created TAR file."
  (let* ((*examples-root* (merge-pathnames "kasandr-root/" root))
         (tar (merge-pathnames "kasandr-sparse/data/de.tar" *examples-root*))
         (fetch (symbol-function 'fetch-file))
         (unzip (symbol-function 'extract-zip-members))
         (decompress (symbol-function 'semz.decompress:make-full-decompression-stream))
         (called nil))
    (unwind-protect
         (progn
           ;; Replace only the remote retrieval with a tiny known compressed fixture.
           (setf (symbol-function 'fetch-file)
                 (lambda (url destination digest &optional fetcher)
                   (declare (ignore url digest fetcher))
                   destination)
                 (symbol-function 'extract-zip-members)
                 (lambda (archive-path directory pairs)
                   (declare (ignore archive-path pairs))
                   (atomicity-write (merge-pathnames "de.tar.bz2" directory) *atomicity-bzip2*)
                   directory)
                 (symbol-function 'semz.decompress:make-full-decompression-stream)
                 (lambda (&rest arguments)
                   (setf called t)
                   (atomicity-write tar *atomicity-sentinel*)
                   (apply decompress arguments)))
           (assert (handler-case (progn (fetch-example-data :kasandr-sparse) nil)
                     (error () t)))
           (assert called)
           (assert (equalp (atomicity-bytes tar) *atomicity-sentinel*)))
      (setf (symbol-function 'fetch-file) fetch
            (symbol-function 'extract-zip-members) unzip
            (symbol-function 'semz.decompress:make-full-decompression-stream) decompress))))

(defun atomicity-kasandr-cleanup-ownership (root)
  "Remove a TAR created here, but keep one already present."
  (let ((fetch (symbol-function 'fetch-file))
        (unzip (symbol-function 'extract-zip-members))
        (extract (symbol-function 'extract-click-tar)))
    (unwind-protect
         (progn
           (setf (symbol-function 'fetch-file)
                 (lambda (url destination digest &optional fetcher)
                   (declare (ignore url digest fetcher))
                   destination)
                 (symbol-function 'extract-zip-members)
                 (lambda (archive-path directory pairs)
                   (declare (ignore archive-path pairs))
                   (atomicity-write (merge-pathnames "de.tar.bz2" directory) *atomicity-bzip2*)
                   directory)
                 (symbol-function 'extract-click-tar)
                 (lambda (tar directory)
                   (declare (ignore directory))
                   (assert (streamp tar))
                   (let ((bytes (make-array (length *atomicity-hello*)
                                            :element-type '(unsigned-byte 8))))
                     (assert (= (read-sequence bytes tar) (length bytes)))
                     (assert (equalp bytes *atomicity-hello*)))))
           (dolist (already-exists '(nil t))
             (let* ((*examples-root* (merge-pathnames
                                      (if already-exists
                                          "kasandr-existing/" "kasandr-created/")
                                      root))
                    (tar (merge-pathnames "kasandr-sparse/data/de.tar" *examples-root*)))
               (when already-exists (atomicity-write tar *atomicity-hello*))
               (fetch-example-data :kasandr-sparse)
               (if already-exists
                   (assert (equalp (atomicity-bytes tar) *atomicity-hello*))
                   (assert (not (probe-file tar)))))))
      (setf (symbol-function 'fetch-file) fetch
            (symbol-function 'extract-zip-members) unzip
            (symbol-function 'extract-click-tar) extract))))

(defun atomicity-kasandr-replaced-tar (root)
  "A replacement at the shared TAR pathname must not be deleted by cleanup."
  (let* ((*examples-root* (merge-pathnames "kasandr-replaced/" root))
         (tar (merge-pathnames "kasandr-sparse/data/de.tar" *examples-root*))
         (backup (merge-pathnames "kasandr-sparse/data/previous.tar" *examples-root*))
         (fetch (symbol-function 'fetch-file))
         (unzip (symbol-function 'extract-zip-members))
         (extract (symbol-function 'extract-click-tar)))
    (unwind-protect
         (progn
           (setf (symbol-function 'fetch-file)
                 (lambda (url destination digest &optional fetcher)
                   (declare (ignore url digest fetcher)) destination)
                 (symbol-function 'extract-zip-members)
                 (lambda (archive directory pairs)
                   (declare (ignore archive pairs))
                   (atomicity-write (merge-pathnames "de.tar.bz2" directory) *atomicity-bzip2*)
                   directory)
                 (symbol-function 'extract-click-tar)
                 (lambda (path directory)
                   (declare (ignore directory))
                   (when (and (not (streamp path))
                              (equal (probe-file path) (probe-file tar)))
                     (rename-file tar backup))
                   (atomicity-write tar *atomicity-sentinel*)))
           (fetch-example-data :kasandr-sparse)
           (assert (equalp (atomicity-bytes tar) *atomicity-sentinel*)))
      (setf (symbol-function 'fetch-file) fetch
            (symbol-function 'extract-zip-members) unzip
            (symbol-function 'extract-click-tar) extract))))

(defun atomicity-existing-extractions-are-verified (root)
  "An already present output must match the archive and remain intact on error."
  (let* ((base (merge-pathnames "existing-extraction/" root))
         (compressed (merge-pathnames "input.bz2" base))
         (plain (merge-pathnames "output.txt" base))
         (zip (atomicity-make-zip (merge-pathnames "zip/" base)))
         (zip-output (merge-pathnames "zip-output.txt" base))
         (tar (atomicity-make-tar (merge-pathnames "tar/" base)))
         (tar-root (merge-pathnames "tar-output/" base))
         (tar-output (merge-pathnames "de/train_de.csv" tar-root)))
    (atomicity-write compressed *atomicity-bzip2*)
    (dolist (path (list plain zip-output tar-output))
      (atomicity-write path *atomicity-sentinel*))
    (assert (handler-case (progn (decompress-file compressed plain :bzip2) nil)
              (error () t)))
    (assert (handler-case
                (progn (extract-zip-members zip base '(("input.txt" . "zip-output.txt"))) nil)
              (error () t)))
    (assert (handler-case (progn (extract-click-tar tar tar-root) nil) (error () t)))
    (dolist (path (list plain zip-output tar-output))
      (assert (equalp (atomicity-bytes path) *atomicity-sentinel*)))))

(defun atomicity-kasandr-existing-inputs-are-verified (root)
  "Existing click TSVs do not skip verification against the downloaded archive."
  (let* ((*examples-root* (merge-pathnames "kasandr-existing-inputs/" root))
         (data (merge-pathnames "kasandr-sparse/data/" *examples-root*))
         (fetch (symbol-function 'fetch-file))
         (called nil))
    (atomicity-write (merge-pathnames "de/train_de.csv" data) *atomicity-sentinel*)
    (atomicity-write (merge-pathnames "de/test_de.csv" data) *atomicity-sentinel*)
    (unwind-protect
         (progn
           (setf (symbol-function 'fetch-file)
                 (lambda (&rest args)
                   (declare (ignore args))
                   (setf called t)
                   (error "Simulated missing archive")))
           (assert (handler-case (progn (fetch-example-data :kasandr-sparse) nil)
                     (error () t)))
           (assert called)
           (assert (equalp (atomicity-bytes (merge-pathnames "de/train_de.csv" data))
                           *atomicity-sentinel*))
           (assert (equalp (atomicity-bytes (merge-pathnames "de/test_de.csv" data))
                           *atomicity-sentinel*)))
      (setf (symbol-function 'fetch-file) fetch))))

(defun atomicity-existing-extractions-read-only (root)
  "Matching outputs are reusable without writing in their directories."
  (let* ((base (merge-pathnames "read-only/" root))
         (compressed (merge-pathnames "input.bz2" base))
         (plain (merge-pathnames "output.txt" base))
         (zip (atomicity-make-zip (merge-pathnames "zip/" base)))
         (zip-output (merge-pathnames "zip-output.txt" base))
         (tar (atomicity-make-tar (merge-pathnames "tar/" base)))
         (tar-root (merge-pathnames "tar-output/" base))
         (tar-dir (merge-pathnames "de/" tar-root)))
    (atomicity-write compressed *atomicity-bzip2*)
    (dolist (path (list plain zip-output
                        (merge-pathnames "de/train_de.csv" tar-root)
                        (merge-pathnames "de/test_de.csv" tar-root)))
      (atomicity-write path *atomicity-hello*))
    (unwind-protect
         (progn
           (sb-posix:chmod (namestring base) #o500)
           (sb-posix:chmod (namestring tar-root) #o500)
           (sb-posix:chmod (namestring tar-dir) #o500)
           (decompress-file compressed plain :bzip2)
           (extract-zip-members zip base '(("input.txt" . "zip-output.txt")))
           (extract-click-tar tar tar-root))
      (sb-posix:chmod (namestring base) #o700)
      (sb-posix:chmod (namestring tar-root) #o700)
      (sb-posix:chmod (namestring tar-dir) #o700))))

(defun atomicity-kasandr-stream-read-only (root)
  "Verify existing click TSVs through a real bzip2/TAR stream without writing."
  (let* ((*examples-root* (merge-pathnames "kasandr-stream/" root))
         (data (merge-pathnames "kasandr-sparse/data/" *examples-root*))
         (de (merge-pathnames "de/" data))
         (train (merge-pathnames "train_de.csv" de))
         (test (merge-pathnames "test_de.csv" de))
         (compressed (merge-pathnames "de.tar.bz2" data))
         (fetch (symbol-function 'fetch-file))
         (unzip (symbol-function 'extract-zip-members)))
    (atomicity-write compressed *atomicity-bzip2-tar*)
    (atomicity-write train *atomicity-hello*)
    (atomicity-write test *atomicity-hello*)
    (unwind-protect
         (progn
           (setf (symbol-function 'fetch-file)
                 (lambda (url destination digest &optional fetcher)
                   (declare (ignore url digest fetcher)) destination)
                 (symbol-function 'extract-zip-members)
                 (lambda (archive directory pairs)
                   (declare (ignore archive pairs)) directory))
           (sb-posix:chmod (namestring data) #o500)
           (sb-posix:chmod (namestring de) #o500)
           (fetch-example-data :kasandr-sparse)
           (assert (not (probe-file (merge-pathnames "de.tar" data))))
           (sb-posix:chmod (namestring data) #o700)
           (sb-posix:chmod (namestring de) #o700)
           (atomicity-write train *atomicity-sentinel*)
           (assert (handler-case (progn (fetch-example-data :kasandr-sparse) nil)
                     (error () t)))
           (assert (equalp (atomicity-bytes train) *atomicity-sentinel*)))
      (sb-posix:chmod (namestring data) #o700)
      (sb-posix:chmod (namestring de) #o700)
      (setf (symbol-function 'fetch-file) fetch
            (symbol-function 'extract-zip-members) unzip))))

(defun atomicity-stream-comparison-edges (root)
  "Check byte and sequence writes across buffer boundaries and different lengths."
  (let ((destination (merge-pathnames "stream-comparison.bin" root))
        (bytes (make-array 65540 :element-type '(unsigned-byte 8)
                                 :initial-contents
                                 (loop for i below 65540 collect (mod (* i 13) 256)))))
    (atomicity-write destination bytes)
    (flet ((writer (output)
             (write-byte (aref bytes 0) output)
             (write-sequence bytes output :start 1 :end 65537)
             (write-sequence bytes output :start 65537)))
      (verify-extracted-file destination #'writer)
      ;; Sequences other than octet vectors compare the same way.
      (verify-extracted-file destination
                             (lambda (out)
                               (write-sequence (coerce bytes 'list) out :end 3)
                               (write-sequence (coerce bytes 'simple-vector) out :start 3)))
      (setf (aref bytes 65536) (logxor 1 (aref bytes 65536)))
      (assert (handler-case (progn (verify-extracted-file destination #'writer) nil)
                (error () t)))
      (setf (aref bytes 65536) (logxor 1 (aref bytes 65536)))
      (assert (handler-case
                  (progn (verify-extracted-file
                          destination (lambda (out) (write-sequence bytes out :end 65539)))
                         nil)
                (error () t)))
      (assert (handler-case
                  (progn (verify-extracted-file
                          destination (lambda (out) (write-sequence bytes out)
                                                    (write-byte 0 out)))
                         nil)
                (error () t))))))

(defun atomicity-fetch-interleaving (root)
  ;; A validates its complete file. B starts an incomplete download before A
  ;; publishes. A's success must still mean that its destination has that digest.
  (let ((destination (merge-pathnames "fetch-race/file.bin" root))
        (original (symbol-function 'check-sha256))
        (checked (sb-thread:make-semaphore))
        (continue-a (sb-thread:make-semaphore))
        (started-b (sb-thread:make-semaphore))
        (continue-b (sb-thread:make-semaphore))
        (a nil)
        (b nil))
    (unwind-protect
         (progn
           (setf (symbol-function 'check-sha256)
                 (lambda (path digest)
                   (funcall original path digest)
                   (when (string= (sb-thread:thread-name sb-thread:*current-thread*)
                                  "atomicity-fetch-a")
                     (sb-thread:signal-semaphore checked)
                     (sb-thread:wait-on-semaphore continue-a))))
           (setf a
                 (sb-thread:make-thread
                  (lambda ()
                    (handler-case
                        (progn
                          (fetch-file "fixture-a" destination *atomicity-sha256*
                                      (lambda (url target)
                                        (declare (ignore url))
                                        (atomicity-write target #(1 2 3))))
                          :success)
                      (error () :error)))
                  :name "atomicity-fetch-a"))
           (assert (sb-thread:wait-on-semaphore checked :timeout 5))
           (setf b
                 (sb-thread:make-thread
                  (lambda ()
                    (handler-case
                        (fetch-file "fixture-b" destination *atomicity-sha256*
                                    (lambda (url target)
                                      (declare (ignore url))
                                      (with-open-file (out target :direction :output
                                                                 :if-exists :supersede
                                                                 :element-type '(unsigned-byte 8))
                                        (write-byte 1 out)
                                        (finish-output out)
                                        (sb-thread:signal-semaphore started-b)
                                        (sb-thread:wait-on-semaphore continue-b)
                                        (error "interrupted download"))))
                      (error () nil)))
                  :name "atomicity-fetch-b"))
           ;; B might be serialized by a valid implementation; don't require it
           ;; to enter its fetcher before A is allowed to finish.
           (let ((b-started (sb-thread:wait-on-semaphore started-b :timeout 1)))
             (sb-thread:signal-semaphore continue-a)
             (assert (eq (sb-thread:join-thread a :timeout 5 :default :timeout) :success))
             (assert (equalp (atomicity-bytes destination) #(1 2 3)))
             (when b-started (assert (equalp (atomicity-bytes destination) #(1 2 3)))))
           (sb-thread:signal-semaphore continue-b)
           (sb-thread:join-thread b :timeout 5 :default :timeout)
           (assert (equalp (atomicity-bytes destination) #(1 2 3))))
      (sb-thread:signal-semaphore continue-a)
      (sb-thread:signal-semaphore continue-b)
      (when (and a (sb-thread:thread-alive-p a)) (sb-thread:join-thread a))
      (when (and b (sb-thread:thread-alive-p b)) (sb-thread:join-thread b))
      (setf (symbol-function 'check-sha256) original))))

(defun atomicity-wildcard-directory-names (root)
  "Names holding characters SBCL escapes in a namestring reach the filesystem as written."
  (let* ((odd (merge-pathnames (make-pathname :directory '(:relative "repo[1]*?dir")) root))
         (download (merge-pathnames "fetch.bin" odd))
         (compressed (merge-pathnames "hello.bz2" odd))
         (plain (merge-pathnames "hello.txt" odd))
         (zip (atomicity-make-zip (merge-pathnames "wildcard-zip-source/" root)))
         (extracted (merge-pathnames "zip/result.txt" odd)))
    (ensure-directories-exist download)
    (fetch-file "fixture" download *atomicity-sha256*
                (lambda (url target)
                  (declare (ignore url))
                  (atomicity-write target #(1 2 3))))
    (assert (equalp (atomicity-bytes download) #(1 2 3)))
    ;; A kept file is verified and reused through the same names.
    (fetch-file "fixture" download *atomicity-sha256*
                (lambda (url target)
                  (declare (ignore url target))
                  (error "A verified file was downloaded again")))
    (atomicity-write compressed *atomicity-bzip2*)
    (decompress-file compressed plain :bzip2)
    (assert (equalp (atomicity-bytes plain) *atomicity-hello*))
    (extract-zip-members zip odd '(("input.txt" . "zip/result.txt")))
    (assert (equalp (atomicity-bytes extracted) *atomicity-hello*))
    ;; Nothing was created under the escaped spelling of the directory instead.
    (assert (not (probe-file (sb-ext:parse-native-namestring
                              (concatenate 'string (sb-ext:native-namestring root)
                                           "repo\\[1]\\*\\?dir/")
                              nil *default-pathname-defaults* :as-directory t))))))

(defun run-fetch-atomicity-tests ()
  (let ((root (uiop:ensure-directory-pathname
               (sb-posix:mkdtemp
                (namestring (merge-pathnames "cl-gbdt-fetch-atomicity-XXXXXX"
                                             (uiop:temporary-directory))))))
        (failures nil))
    (unwind-protect
         (dolist (test '(atomicity-fetch-keeps-part
                         atomicity-fetch-failure-keeps-part
                         atomicity-decompress-keeps-part
                         atomicity-decompress-failure-keeps-part
                         atomicity-zip-keeps-part
                         atomicity-tar-keeps-part
                         atomicity-fetch-destination-collision
                         atomicity-decompress-destination-collision
                         atomicity-zip-destination-collision
                         atomicity-tar-destination-collision
                         atomicity-matching-destination-collisions
                         atomicity-kasandr-collision
                         atomicity-kasandr-cleanup-ownership
                         atomicity-kasandr-replaced-tar
                         atomicity-existing-extractions-are-verified
                         atomicity-existing-extractions-read-only
                         atomicity-kasandr-existing-inputs-are-verified
                         atomicity-kasandr-stream-read-only
                         atomicity-stream-comparison-edges
                         atomicity-wildcard-directory-names
                         atomicity-fetch-interleaving))
           (handler-case
               (progn (funcall test root) (format t "PASS ~A~%" test))
             (error (condition)
               (push test failures)
               (format t "FAIL ~A: ~A~%" test condition))))
      (uiop:delete-directory-tree root :validate t))
    (when failures (error "Fetch atomicity tests failed: ~S" (nreverse failures)))
    (format t "Fetch atomicity tests passed.~%")))

(run-fetch-atomicity-tests)
