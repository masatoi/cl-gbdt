(in-package #:cl-user)

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

(defun run-fetch-tests ()
  (let* ((root (merge-pathnames (format nil "toy-data/data/fetch-tests/~36R/"
                                      (random most-positive-fixnum))
                                 *examples-root*))
         (download (merge-pathnames "download.bin" root))
         (download-part (merge-pathnames "download.bin.part" root))
         (bad-download (merge-pathnames "failed.bin" root))
         (bad-part (merge-pathnames "failed.bin.part" root))
         (bz2 (merge-pathnames "hello.bz2" root))
         (plain (merge-pathnames "hello.txt" root))
         (plain-part (merge-pathnames "hello.txt.part" root))
         (bad-bz2 (merge-pathnames "bad.bz2" root))
         (bad-plain (merge-pathnames "bad.txt" root))
         (bad-plain-part (merge-pathnames "bad.txt.part" root))
         (zip (merge-pathnames "fixture.zip" root))
         (extracted (merge-pathnames "result.txt" root))
         (extracted-part (merge-pathnames "result.txt.part" root))
         (files (list download download-part bad-download bad-part bz2 plain plain-part
                      bad-bz2 bad-plain bad-plain-part zip extracted extracted-part))
         (calls 0))
    (unwind-protect
         (progn
           (fetch-file "fixture" download
                       (lambda (url target)
                         (declare (ignore url))
                         (incf calls)
                         (write-test-bytes target #(1 2 3))))
           (assert (equalp (test-file-bytes download) #(1 2 3)))
           (fetch-file "fixture" download
                       (lambda (url target)
                         (declare (ignore url target))
                         (incf calls)))
           (assert (= calls 1))
           (assert (handler-case
                       (progn
                         (fetch-file "failing fixture" bad-download
                                     (lambda (url target)
                                       (declare (ignore url))
                                       (write-test-bytes target #(1))
                                       (error "simulated failure")))
                         nil)
                     (error () t)))
           (assert (not (probe-file bad-download)))
           (assert (not (probe-file bad-part)))
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
      (dolist (path files)
        (when (probe-file path) (delete-file path)))))
  (format t "Download and archive utility tests passed.~%"))

(run-fetch-tests)
