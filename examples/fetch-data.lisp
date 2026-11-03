(in-package #:cl-user)

(load (merge-pathnames "common.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

(defparameter *examples-root* (uiop:pathname-directory-pathname *load-truename*))

(defun fetch-file (url destination
                   &optional (fetcher (lambda (source target)
                                        (uiop:symbol-call :dexador :fetch source target
                                                          :if-exists :supersede))))
  "Download URL to DESTINATION, retaining existing files and deleting partial downloads."
  (unless (probe-file destination)
    (ensure-directories-exist destination)
    (let ((temporary (pathname (concatenate 'string (namestring destination) ".part")))
          (complete nil))
      (unwind-protect
           (progn
             (format t "Downloading ~A~%" url)
             (funcall fetcher url temporary)
             (rename-file temporary destination)
             (setf complete t))
        (unless complete
          (when (probe-file temporary) (delete-file temporary))))))
  destination)

(defun extract-zip-members (archive-path root pairs)
  "Extract only explicitly named ZIP members to predetermined destination paths."
  (zip:with-zipfile (archive archive-path)
    (dolist (pair pairs)
      (let* ((entry (zip:get-zipfile-entry (car pair) archive))
             (destination (merge-pathnames (cdr pair) root))
             (temporary (pathname (concatenate 'string (namestring destination) ".part"))))
        (unless entry (error "ZIP member ~A not found" (car pair)))
        (unless (probe-file destination)
          (ensure-directories-exist destination)
          (let ((complete nil))
            (unwind-protect
                 (progn
                   (zip:zipfile-entry-contents entry temporary)
                   (rename-file temporary destination)
                   (setf complete t))
              (unless complete (when (probe-file temporary) (delete-file temporary)))))))))
  root)

(defun decompress-file (source destination format)
  (unless (probe-file destination)
    (let ((complete nil)
          (temporary (pathname (concatenate 'string (namestring destination) ".part"))))
      (unwind-protect
           (progn
             (ensure-directories-exist destination)
             (with-open-file (in source :element-type '(unsigned-byte 8))
               (with-open-file (out temporary :direction :output :if-exists :supersede
                                              :if-does-not-exist :create
                                              :element-type '(unsigned-byte 8))
                 (let ((decoded (semz.decompress:make-full-decompression-stream format in))
                       (buffer (make-array 65536 :element-type '(unsigned-byte 8))))
                   (loop for count = (read-sequence buffer decoded)
                         while (plusp count) do (write-sequence buffer out :end count)))))
             (rename-file temporary destination)
             (setf complete t))
        (unless complete (when (probe-file temporary) (delete-file temporary))))))
  destination)

(defun copy-octets (in out)
  (let ((buffer (make-array 65536 :element-type '(unsigned-byte 8))))
    (loop for count = (read-sequence buffer in)
          while (plusp count) do (write-sequence buffer out :end count))))

(defun extract-click-tar (path root)
  "Extract only the two German TSV files; archive names cannot determine output paths."
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
            (let* ((destination (merge-pathnames (cdr match) root))
                   (temporary (pathname (concatenate 'string (namestring destination) ".part"))))
              (unless (probe-file destination)
                (ensure-directories-exist destination)
                (let ((complete nil))
                  (unwind-protect
                       (progn
                         (with-open-file (out temporary :direction :output :if-exists :supersede
                                                        :if-does-not-exist :create
                                                        :element-type '(unsigned-byte 8))
                           (copy-octets (archive:entry-stream entry) out))
                         (rename-file temporary destination)
                         (setf complete t))
                    (unless complete
                      (when (probe-file temporary) (delete-file temporary)))))))))))
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
         (dolist (name '("iris.csv" "breast_cancer.csv"))
           (fetch-file (concatenate 'string prefix name) (merge-pathnames name root)))
         (dolist (name '("diabetes_data_raw.csv" "diabetes_target.csv"))
           (let ((compressed (merge-pathnames (concatenate 'string name ".gz") root))
                 (plain (merge-pathnames name root)))
             (fetch-file (concatenate 'string prefix name ".gz") compressed)
             (decompress-file compressed plain :gzip)))))
      (:adult
       (dolist (name '("adult.data" "adult.test"))
         (fetch-file (concatenate 'string
                                  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
                                  name)
                     (merge-pathnames name root))))
      (:credit-card-fraud
       (let ((zip (merge-pathnames "source.zip" root)))
         (fetch-file (concatenate 'string
                      "https://raw.githubusercontent.com/stat432/credit-analysis/"
                      "125090b731479d1566bdd8b7ad1cf674fdb2bb1e/data-raw/creditcard.csv.zip")
                     zip)
         (extract-zip-members zip root '(("creditcard.csv" . "creditcard.csv")))))
      (:household-power
       (let ((zip (merge-pathnames "source.zip" root)))
         (fetch-file (concatenate 'string "https://archive.ics.uci.edu/static/public/235/"
                                  "individual+household+electric+power+consumption.zip") zip)
         (extract-zip-members zip root
                              '(("household_power_consumption.txt" .
                                 "household_power_consumption.txt")))))
      (:movielens-ranking
       (let ((zip (merge-pathnames "source.zip" root)))
         (fetch-file "https://files.grouplens.org/datasets/movielens/ml-100k.zip" zip)
         (extract-zip-members zip root '(("ml-100k/u.data" . "ml-100k/u.data")
                                        ("ml-100k/u.item" . "ml-100k/u.item")
                                        ("ml-100k/u.user" . "ml-100k/u.user")))))
      (:online-retail
       (let ((zip (merge-pathnames "source.zip" root)))
         (fetch-file "https://archive.ics.uci.edu/static/public/352/online+retail.zip" zip)
         (extract-zip-members zip root '(("Online Retail.xlsx" . "Online Retail.xlsx")))))
      (:kasandr-sparse
       (unless (and (probe-file (merge-pathnames "de/train_de.csv" root))
                    (probe-file (merge-pathnames "de/test_de.csv" root)))
         (let* ((zip (merge-pathnames "source.zip" root))
                (compressed (merge-pathnames "de.tar.bz2" root))
                (tar (merge-pathnames "de.tar" root))
                (created-tar (not (probe-file tar))))
           (fetch-file "https://archive.ics.uci.edu/static/public/385/kasandr.zip" zip)
           (extract-zip-members zip root '(("de.tar.bz2" . "de.tar.bz2")))
           (unwind-protect
                (progn (decompress-file compressed tar :bzip2)
                       (extract-click-tar tar root))
             (when (and created-tar (probe-file tar)) (delete-file tar))))))
      (otherwise (error "Unsupported example: ~S" example)))
    (format t "Source files ready in ~A~%" root)
    root))

(defun ensure-retail-reader ()
  "Load lisp-xl, installing its pinned source into the ignored example data directory."
  (unless (asdf:find-system :lisp-xl nil)
    (let* ((revision "d02681363c2380cb653411a63924a6b74951595c")
           (root (merge-pathnames "online-retail/data/" *examples-root*))
           (zip (merge-pathnames "lisp-xl-source.zip" root))
           (prefix (format nil "lisp-xl-~A/" revision))
           (files '("lisp-xl.asd" "README.markdown" "src/package.lisp" "src/util.lisp"
                    "src/metadata.lisp" "src/col-process.lisp" "src/main.lisp"
                    "src/csv-export.lisp")))
      (fetch-file (format nil "https://github.com/defunkydrummer/lisp-xl/archive/~A.zip"
                          revision)
                  zip)
      (extract-zip-members zip (merge-pathnames "lisp-xl/" root)
                           (mapcar (lambda (file) (cons (concatenate 'string prefix file) file))
                                   files))
      (pushnew (merge-pathnames "lisp-xl/" root) asdf:*central-registry* :test #'equal)
      (asdf:load-asd (merge-pathnames "lisp-xl/lisp-xl.asd" root))))
  (uiop:symbol-call :ql :quickload :lisp-xl :silent t))

(export 'ensure-retail-reader)

(export 'fetch-example-data)
