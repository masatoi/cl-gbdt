(in-package #:cl-user)

(require :sb-posix)

(load (merge-pathnames "common.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

(defparameter *examples-root* (uiop:pathname-directory-pathname *load-truename*))

(define-condition digest-mismatch (error)
  ((path :initarg :path :reader digest-mismatch-path)
   (expected :initarg :expected :reader digest-mismatch-expected)
   (actual :initarg :actual :reader digest-mismatch-actual))
  (:report (lambda (condition stream)
             (format stream "~A has SHA-256 ~A, but ~A was recorded for it. Either this is not ~
                             the recorded file or its publisher has changed it. A file kept ~
                             from an earlier run must be removed before it is downloaded again."
                     (digest-mismatch-path condition)
                     (digest-mismatch-actual condition)
                     (digest-mismatch-expected condition))))
  (:documentation "A source file's SHA-256 is not the one recorded for it."))

(defun check-sha256 (path expected)
  "Signal DIGEST-MISMATCH unless the file at PATH has the lower-case hex SHA-256 EXPECTED."
  (let ((actual (ironclad:byte-array-to-hex-string (ironclad:digest-file :sha256 path))))
    (unless (string= actual expected)
      (error 'digest-mismatch :path path :expected expected :actual actual))))

(defun call-with-staged-file (destination function)
  "Call FUNCTION with an exclusively created temporary file beside DESTINATION."
  (ensure-directories-exist destination)
  (uiop:call-with-temporary-file function
                                 :want-stream-p nil
                                 :directory (uiop:pathname-directory-pathname destination)
                                 :prefix ".cl-gbdt-" :suffix ".part" :type :unspecific
                                 :element-type '(unsigned-byte 8)))

(defun publish-staged-file (temporary destination)
  "Publish TEMPORARY atomically, returning NIL if DESTINATION already exists."
  (handler-case
      (progn
        ;; The stage lives beside the destination, so the link is on one filesystem.
        ;; Unlike RENAME-FILE, LINK never replaces a file that appeared meanwhile.
        ;; NAMESTRING escapes characters such as [ and *, so pass native names.
        (sb-posix:link (sb-ext:native-namestring temporary)
                       (sb-ext:native-namestring destination))
        t)
    (sb-posix:syscall-error (condition)
      (if (= (sb-posix:syscall-errno condition) sb-posix:eexist)
          nil
          (error condition)))))

(defun same-file-bytes-p (left right)
  "True if LEFT and RIGHT both exist and have exactly the same octets."
  (with-open-file (a left :element-type '(unsigned-byte 8) :if-does-not-exist nil)
    (with-open-file (b right :element-type '(unsigned-byte 8) :if-does-not-exist nil)
      (when (and a b)
        (let ((a-buffer (make-array 65536 :element-type '(unsigned-byte 8)))
              (b-buffer (make-array 65536 :element-type '(unsigned-byte 8))))
          (loop for a-count = (read-sequence a-buffer a)
                for b-count = (read-sequence b-buffer b)
                do (unless (and (= a-count b-count)
                                (loop for index below a-count
                                      always (= (aref a-buffer index) (aref b-buffer index))))
                     (return nil))
                when (zerop a-count) do (return t)))))))

(defclass comparing-octet-stream (sb-gray:fundamental-binary-output-stream)
  ((input :initarg :input :reader comparison-input)
   (buffer :initform (make-array 65536 :element-type '(unsigned-byte 8))
           :reader comparison-buffer)
   (matches :initform t :accessor comparison-matches-p))
  (:documentation "An output stream that compares the octets written to it with INPUT."))

(defmethod stream-element-type ((stream comparing-octet-stream))
  (declare (ignore stream))
  '(unsigned-byte 8))

(defmethod sb-gray:stream-write-byte ((stream comparing-octet-stream) byte)
  (unless (eql byte (read-byte (comparison-input stream) nil :eof))
    (setf (comparison-matches-p stream) nil))
  byte)

(defmethod sb-gray:stream-write-sequence ((stream comparing-octet-stream) sequence
                                          &optional (start 0) end)
  (let ((buffer (comparison-buffer stream))
         (input (comparison-input stream))
         (start (or start 0))
         (end (or end (length sequence))))
    (declare (type (simple-array (unsigned-byte 8) (*)) buffer))
    (loop for offset from start below end by (length buffer)
          for count = (min (length buffer) (- end offset))
          for read = (read-sequence buffer input :end count)
          do (unless (and (= read count)
                          (if (typep sequence '(simple-array (unsigned-byte 8) (*)))
                              (loop for i of-type fixnum below read
                                    always (= (aref buffer i) (aref sequence (+ offset i))))
                              (loop for i below read
                                    always (eql (aref buffer i) (elt sequence (+ offset i))))))
               (setf (comparison-matches-p stream) nil))))
  sequence)

(defun verify-extracted-file (destination writer)
  "Compare WRITER's octets with DESTINATION without creating another file."
  (with-open-file (input destination :element-type '(unsigned-byte 8)
                                     :if-does-not-exist nil)
    (unless input (error "The destination ~A disappeared before verification." destination))
    (let ((comparison (make-instance 'comparing-octet-stream :input input)))
      (funcall writer comparison)
      (unless (and (comparison-matches-p comparison) (eq (read-byte input nil :eof) :eof))
        (error "The destination ~A already exists and differs; it was retained." destination))))
  nil)

(defun publish-extracted-file (temporary destination)
  "Publish an extraction; return true only when this call created DESTINATION.
Retain and check any existing destination."
  (if (publish-staged-file temporary destination)
      t
      (progn
        (unless (same-file-bytes-p temporary destination)
          (error "The destination ~A already exists and differs; it was retained."
                 destination))
        nil)))

(defun fetch-file (url destination sha256
                   &optional (fetcher (lambda (source target)
                                        (uiop:symbol-call :dexador :fetch source target
                                                          :if-exists :supersede))))
  "Download URL to DESTINATION unless it exists, checking the result against SHA256.
A mismatched download is never published. A mismatched existing or concurrently
created destination is retained and signals DIGEST-MISMATCH."
  (if (probe-file destination)
      (check-sha256 destination sha256)
      (call-with-staged-file
       destination
       (lambda (temporary)
         (format t "Downloading ~A~%" url)
         (funcall fetcher url temporary)
         (check-sha256 temporary sha256)
         (unless (publish-staged-file temporary destination)
           (check-sha256 destination sha256)))))
  destination)

(defun extract-zip-members (archive-path root pairs)
  "Extract named ZIP members, checking any existing destination before reuse."
  (zip:with-zipfile (archive archive-path)
    (dolist (pair pairs)
      (let ((entry (zip:get-zipfile-entry (car pair) archive))
            (destination (merge-pathnames (cdr pair) root)))
        (unless entry (error "ZIP member ~A not found" (car pair)))
        (if (probe-file destination)
            (verify-extracted-file
             destination (lambda (output) (zip:zipfile-entry-contents entry output)))
            (call-with-staged-file
             destination
             (lambda (temporary)
               (zip:zipfile-entry-contents entry temporary)
               (publish-extracted-file temporary destination)))))))
  root)

(defun write-decompressed-file (source destination format)
  "Write SOURCE decoded as FORMAT to a private DESTINATION file."
  (with-open-file (in source :element-type '(unsigned-byte 8))
    (with-open-file (out destination :direction :output :if-exists :supersede
                                     :if-does-not-exist :create
                                     :element-type '(unsigned-byte 8))
      (let ((decoded (semz.decompress:make-full-decompression-stream format in))
            (buffer (make-array 65536 :element-type '(unsigned-byte 8))))
        (loop for count = (read-sequence buffer decoded)
              while (plusp count) do (write-sequence buffer out :end count))))))

(defun copy-octets (in out)
  (let ((buffer (make-array 65536 :element-type '(unsigned-byte 8))))
    (loop for count = (read-sequence buffer in)
          while (plusp count) do (write-sequence buffer out :end count))))

(defun decompress-file (source destination format)
  "Decompress SOURCE, verifying any existing DESTINATION before reusing it.
Return DESTINATION and whether this call created it."
  (values destination
          (if (probe-file destination)
              (with-open-file (input source :element-type '(unsigned-byte 8))
                (let ((decoded (semz.decompress:make-full-decompression-stream format input)))
                  (verify-extracted-file destination
                                         (lambda (output) (copy-octets decoded output)))))
              (call-with-staged-file
               destination
               (lambda (temporary)
                 (write-decompressed-file source temporary format)
                 (publish-extracted-file temporary destination))))))

(defun extract-click-tar (path root)
  "Extract and verify the two German TSV files under predetermined paths."
  (let ((wanted '(("de/train_de.csv" . "de/train_de.csv")
                  ("de/test_de.csv" . "de/test_de.csv")))
        (found (make-hash-table :test #'equal)))
    (archive:with-open-archive (tar path :direction :input)
      (archive:do-archive-entries (entry tar)
        (let* ((name (archive:name entry))
               (normalized (if (and (>= (length name) 2)
                                    (string= name "./" :end1 2))
                               (subseq name 2) name))
               (match (assoc normalized wanted :test #'string=)))
          (when match
            (unless (archive:entry-regular-file-p entry)
              (error "Not a regular archive member: ~A" name))
            (setf (gethash (car match) found) t)
            (let ((destination (merge-pathnames (cdr match) root)))
              (if (probe-file destination)
                  (verify-extracted-file
                   destination (lambda (output) (copy-octets (archive:entry-stream entry) output)))
                  (call-with-staged-file
                   destination
                   (lambda (temporary)
                     (with-open-file (out temporary :direction :output :if-exists :supersede
                                                    :if-does-not-exist :create
                                                    :element-type '(unsigned-byte 8))
                       (copy-octets (archive:entry-stream entry) out))
                     (publish-extracted-file temporary destination)))))))))
    (dolist (pair wanted)
      (unless (gethash (car pair) found) (error "TAR member missing: ~A" (car pair))))))

(defun fetch-example-data (example)
  "Download and unpack the public source for EXAMPLE with Common Lisp libraries.
EXAMPLE is one of :TOY-DATA, :ADULT, :CREDIT-CARD-FRAUD, :HOUSEHOLD-POWER,
:MOVIELENS-RANKING, :ONLINE-RETAIL, or :KASANDR-SPARSE.  MSLR is supplied
separately after the user accepts the publisher's terms."
  (let* ((name (string-downcase (string example)))
         (root (merge-pathnames (format nil "~A/data/" name) *examples-root*)))
    (ensure-directories-exist (merge-pathnames ".keep" root))
    (case example
      (:toy-data
       (let ((prefix (concatenate 'string "https://raw.githubusercontent.com/"
                                  "scikit-learn/scikit-learn/1.8.0/sklearn/datasets/data/")))
         (loop for (name sha256)
                 in '(("iris.csv"
                       "f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449")
                      ("breast_cancer.csv"
                       "fed3eb72d0575ef6192293f5093c6e801b1476b577d0386bf4455504522172ed"))
               do (fetch-file (concatenate 'string prefix name) (merge-pathnames name root)
                              sha256))
         (loop for (name sha256)
                 in '(("diabetes_data_raw.csv"
                       "7fc0ded571454b1982210d3bb43f0aca44eae01a0b8654a3b24022bdb6b38009")
                      ("diabetes_target.csv"
                       "8e53f65eb811df43c206f3534bb3af0e5fed213bc37ed6ba36310157d6023803"))
               do (let ((compressed (merge-pathnames (concatenate 'string name ".gz") root))
                        (plain (merge-pathnames name root)))
                    (fetch-file (concatenate 'string prefix name ".gz") compressed sha256)
                    (decompress-file compressed plain :gzip)))))
      (:adult
       (loop for (name sha256)
               in '(("adult.data"
                     "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d")
                    ("adult.test"
                     "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"))
             do (fetch-file (concatenate 'string
                                         "https://archive.ics.uci.edu/ml/"
                                         "machine-learning-databases/adult/"
                                         name)
                            (merge-pathnames name root)
                            sha256)))
      (:credit-card-fraud
       (let ((zip (merge-pathnames "source.zip" root)))
         (fetch-file (concatenate 'string
                      "https://raw.githubusercontent.com/stat432/credit-analysis/"
                      "125090b731479d1566bdd8b7ad1cf674fdb2bb1e/data-raw/creditcard.csv.zip")
                     zip
                     "d9ade8e5a6c7b39cf3afef2de0fc10ecf9337c2cebdc6cc59a54b1d9dce007cc")
         (extract-zip-members zip root '(("creditcard.csv" . "creditcard.csv")))))
      (:household-power
       (let ((zip (merge-pathnames "source.zip" root)))
         (fetch-file (concatenate 'string "https://archive.ics.uci.edu/static/public/235/"
                                  "individual+household+electric+power+consumption.zip")
                     zip
                     "9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff")
         (extract-zip-members zip root
                              '(("household_power_consumption.txt" .
                                 "household_power_consumption.txt")))))
      (:movielens-ranking
       (let ((zip (merge-pathnames "source.zip" root)))
         (fetch-file "https://files.grouplens.org/datasets/movielens/ml-100k.zip" zip
                     "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229")
         (extract-zip-members zip root '(("ml-100k/u.data" . "ml-100k/u.data")
                                        ("ml-100k/u.item" . "ml-100k/u.item")
                                        ("ml-100k/u.user" . "ml-100k/u.user")))))
      (:online-retail
       (let ((zip (merge-pathnames "source.zip" root)))
         (fetch-file "https://archive.ics.uci.edu/static/public/352/online+retail.zip" zip
                     "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b")
         (extract-zip-members zip root '(("Online Retail.xlsx" . "Online Retail.xlsx")))))
      (:kasandr-sparse
       (let ((zip (merge-pathnames "source.zip" root))
             (compressed (merge-pathnames "de.tar.bz2" root)))
         (fetch-file "https://archive.ics.uci.edu/static/public/385/kasandr.zip" zip
                     "1851d21603d24c4baa060d4e73ce98a2393effe3d0af68b59e63fb0bd86db509")
         (extract-zip-members zip root '(("de.tar.bz2" . "de.tar.bz2")))
         ;; Stream each TAR independently without creating a shared TAR file.
         (with-open-file (input compressed :element-type '(unsigned-byte 8))
           (let ((decoded (semz.decompress:make-full-decompression-stream :bzip2 input)))
             (extract-click-tar decoded root)))))
      (otherwise (error "Unsupported example: ~S" example)))
    (format t "Source files ready in ~A~%" root)
    root))

(defparameter *retail-reader-source*
  (concatenate 'string "https://raw.githubusercontent.com/defunkydrummer/"
               "lisp-xl/d02681363c2380cb653411a63924a6b74951595c/")
  "Base URL of the pinned lisp-xl revision.")

(defparameter *retail-reader-files*
  '(("lisp-xl.asd"
     "cecc57d3058de311a8eb43233c42d70328e2718fd40c5c8d155dd4846acf2b8c")
    ("README.markdown"
     "973e3f954130120535662e0c54aab4d27640ce45d13e25a91136052ca8507a23")
    ("src/package.lisp"
     "357757ea2c58c4de04403d667f8f56258e34f44523a5e45084a202270d8b1a71")
    ("src/util.lisp"
     "dd01eac15cef3b76cc2d8a982ce192b4fc2f4e1b2d2ad3e68344882e08c2c367")
    ("src/metadata.lisp"
     "96455b32076c49d3f6207bb2e96385e09f48d6c8db0c5e5f7e95c3804a3b1ab9")
    ("src/col-process.lisp"
     "63b2c71657653119b04bdcc7d6086908d4a50b48a6e0bdb19a7211b346a5819c")
    ("src/main.lisp"
     "f44a73a94dd04c08b73c81c49ee5bf8194c98a4de98ded9589fc9f629a86099d")
    ("src/csv-export.lisp"
     "595d9a6672eb8af81a2620ffd5739472a22f0ce40e8904b213457b51b97e93a5"))
  "Each file of the pinned lisp-xl revision with its recorded SHA-256.")

(defun verify-retail-reader-cache (cache)
  "Verify the managed lisp-xl CACHE directory and return the pathname of its .asd.
Every kept file is checked against its recorded SHA-256 before any missing one is
downloaded and checked. A .asd that resolves outside CACHE is refused, because ASDF
would read the system's sources beside its target, which were never checked."
  (let ((asd (merge-pathnames "lisp-xl.asd" cache)))
    (flet ((check-location ()
             (let ((resolved (probe-file asd)))
               (when (and resolved
                          (not (equal (probe-file cache)
                                      (probe-file
                                       (uiop:pathname-directory-pathname resolved)))))
                 (error "ASD ~A resolves outside the managed reader cache ~A." asd cache)))))
      (check-location)
      (loop for (file sha256) in *retail-reader-files*
            for path = (merge-pathnames file cache)
            when (probe-file path) do (check-sha256 path sha256))
      (loop for (file sha256) in *retail-reader-files*
            do (fetch-file (concatenate 'string *retail-reader-source* file)
                           (merge-pathnames file cache) sha256))
      (check-location)
      asd)))

(defun quickload-retail-reader (system)
  "Quickload lisp-xl as the registered SYSTEM, whatever other copy ASDF could find.
Other systems, including lisp-xl's dependencies, are found as usual."
  (let ((asdf:*system-definition-search-functions*
          (cons (lambda (name)
                  (when (string-equal (asdf:coerce-name name) "lisp-xl") system))
                asdf:*system-definition-search-functions*)))
    (uiop:symbol-call :ql :quickload :lisp-xl :silent t)))

(defun ensure-retail-reader ()
  "Load lisp-xl, the Excel reader of the Online Retail example.
When this checkout's managed cache, online-retail/data/lisp-xl/, exists, each pinned
file in it is checked against its recorded SHA-256, missing ones are downloaded and
checked, and only then is its .asd loaded; ASDF can rediscover that directory, so
nothing in it is loaded unverified. Otherwise a lisp-xl that ASDF has registered or
can find is used as supplied. Failing both, the pinned revision is downloaded into
the managed cache and loaded as above."
  (let ((cache (merge-pathnames "online-retail/data/lisp-xl/" *examples-root*)))
    (flet ((load-managed ()
             (let ((asd (verify-retail-reader-cache cache)))
               ;; LOAD-ASD re-reads an already registered system's own .asd, which
               ;; may be another copy, instead of the file it is given.
               (asdf:clear-system :lisp-xl)
               (asdf:load-asd asd))
             (quickload-retail-reader (asdf:registered-system :lisp-xl))))
      ;; Decide on the cache before asking ASDF, whose search can load a .asd.
      (if (probe-file cache)
          (load-managed)
          (let ((external (or (asdf:registered-system :lisp-xl)
                              (asdf:find-system :lisp-xl nil))))
            (if external
                (quickload-retail-reader external)
                (load-managed)))))))

(export 'ensure-retail-reader)

(export 'fetch-example-data)
