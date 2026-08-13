;;;; xgboost-file-input.lisp --- Tests for the pure layer behind XGBoost file input.
;;;;
;;;; detect-file-format and file-uri call no foreign function and touch no handle, so
;;;; every branch here is reachable without either shared library (layer 1).

(uiop:define-package #:cl-gbdt/tests/xgboost-file-input
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt/src/xgboost/file-input)
  (:import-from #:cl-gbdt/src/conditions))

(in-package #:cl-gbdt/tests/xgboost-file-input)

(defmacro with-temporary-file ((path content) &body body)
  "Write CONTENT, a string, to a fresh path under `uiop:temporary-directory'; bind PATH
to that pathname for the extent of BODY; delete the file afterward whether BODY exits
normally or not."
  (let ((path-var (gensym "PATH"))
        (content-var (gensym "CONTENT")))
    `(let* ((,content-var ,content)
            (,path-var (merge-pathnames
                        (format nil "cl-gbdt-file-input-test-~A.tmp" (gensym "F"))
                        (uiop:temporary-directory))))
       (unwind-protect
            (progn
              (with-open-file (stream ,path-var :direction :output
                                                 :if-exists :supersede
                                                 :if-does-not-exist :create)
                (write-string ,content-var stream))
              (let ((,path ,path-var))
                ,@body))
         (ignore-errors (delete-file ,path-var))))))

(defmacro with-temporary-byte-file ((path content) &body body)
  "The byte-vector sibling of `with-temporary-file': writes CONTENT, a vector of
\(unsigned-byte 8), to a fresh path under `uiop:temporary-directory' opened with
:element-type '(unsigned-byte 8); binds PATH to it for the extent of BODY; deletes the
file afterward whether BODY exits normally or not."
  (let ((path-var (gensym "PATH"))
        (content-var (gensym "CONTENT")))
    `(let* ((,content-var ,content)
            (,path-var (merge-pathnames
                        (format nil "cl-gbdt-file-input-test-~A.bin" (gensym "F"))
                        (uiop:temporary-directory))))
       (unwind-protect
            (progn
              (with-open-file (stream ,path-var :direction :output
                                                 :element-type '(unsigned-byte 8)
                                                 :if-exists :supersede
                                                 :if-does-not-exist :create)
                (write-sequence ,content-var stream))
              (let ((,path ,path-var))
                ,@body))
         (ignore-errors (delete-file ,path-var))))))

(deftest detect-file-format-classifies-libsvm
  (with-temporary-file (path "1 1:1.0 2:2.0 3:3.0
0 1:4.0 2:5.0 3:6.0
")
    (ok (eq :libsvm (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

(deftest detect-file-format-classifies-csv
  (with-temporary-file (path "1,1.0,2.0,3.0
0,4.0,5.0,6.0
")
    (ok (eq :csv (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

(deftest detect-file-format-skips-leading-blank-lines
  (with-temporary-file (path "

1 1:1.0 2:2.0 3:3.0
")
    (ok (eq :libsvm (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

(deftest detect-file-format-reports-unknown-for-an-empty-file
  (with-temporary-file (path "")
    (ok (eq :unknown (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

(deftest detect-file-format-reports-unknown-for-only-blank-lines
  (with-temporary-file (path "

")
    (ok (eq :unknown (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

(deftest detect-file-format-reports-unreadable-for-a-missing-file
  (ok (eq :unreadable
          (cl-gbdt/src/xgboost/file-input::detect-file-format
           #p"/nonexistent/cl-gbdt/no-such-file.libsvm"))))

;; Measured: without the comma guard running first, "12:00:00,1.0" satisfies
;; <digits>:<rest> and this line reads as libsvm -- the direction that SIGSEGVs XGBoost.
(deftest detect-file-format-classifies-a-timestamp-csv-as-csv
  (with-temporary-file (path "2024-01-01 12:00:00,1.0,2.0
")
    (ok (eq :csv (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

(deftest detect-file-format-classifies-an-xgboost-binary-dmatrix
  (with-temporary-byte-file (path #(#x01 #xAB #xFF #xFF 0 0 0 0))
    (ok (eq :binary (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

(deftest detect-file-format-classifies-a-lightgbm-binary-dataset
  (with-temporary-file (path "______LightGBM_Binary_File_Token______and then some")
    (ok (eq :binary (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

;; Measured (record section 1 blind spot 3, section 4): declared :libsvm this line
;; SIGSEGVs XGBoost. Only %libsvm-token-p rejecting "1.0" keeps it out of :libsvm --
;; deleting that conjunct from detect-file-format would turn this green suite silent
;; about the exact input that kills the process.
(deftest detect-file-format-classifies-space-delimited-numbers-as-csv
  (with-temporary-file (path "1 1.0 2.0 3.0
")
    (ok (eq :csv (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

;; Measured (record section 1 blind spot 1): a labels-only libsvm file has one token
;; per line, so deleting the ">= 2 tokens" guard from detect-file-format would also
;; turn this green suite silent -- the other half of step 4's fallthrough.
(deftest detect-file-format-classifies-a-labels-only-file-as-csv
  (with-temporary-file (path "1
0
")
    (ok (eq :csv (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

;; C1: a latin-1 CSV (an accented name column, e.g. "cafe" with an e-acute) is not valid
;; UTF-8 -- byte 233 (0xE9) alone, followed by a comma rather than a continuation byte,
;; fails UTF-8 decoding outright. Written byte-wise so it does not depend on the source
;; file's own encoding. Classifying on octets rather than decoded characters is what
;; makes this :CSV rather than :UNREADABLE or :UNKNOWN -- both of which would be wrong
;; here: :UNKNOWN is a mismatch in Task 4's gate, and :UNREADABLE is a pass-through, so
;; mapping a decoding failure to either would either wrongly refuse an ordinary CSV or,
;; worse, let it reach the foreign call with no format check at all.
(deftest detect-file-format-classifies-a-latin-1-csv-as-csv
  (with-temporary-byte-file (path #(99 97 102 233 44 49 46 48 44 50 46 48 10))
    (ok (eq :csv (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

(deftest file-uri-appends-the-format-as-a-query-parameter
  (ok (string= "/data/train.libsvm?format=libsvm"
               (cl-gbdt/src/xgboost/file-input::file-uri
                #p"/data/train.libsvm" :libsvm nil))))

(deftest file-uri-appends-extra-parameters-after-the-format
  (ok (string= "/data/train.csv?format=csv&label_column=0"
               (cl-gbdt/src/xgboost/file-input::file-uri
                #p"/data/train.csv" :csv '(:label_column 0)))))

;; Measured: ?format=binary is rejected ("Unknown data type binary"); a binary DMatrix
;; loads only when the URI carries no format at all.
(deftest file-uri-emits-no-format-key-for-binary
  (ok (string= "/data/train.bin"
               (cl-gbdt/src/xgboost/file-input::file-uri #p"/data/train.bin" :binary nil))))

;; Measured: dmlc accepts a raw space; %20 fails on both backends.
(deftest file-uri-leaves-a-space-in-the-path-unencoded
  (ok (string= "/data/train set.libsvm?format=libsvm"
               (cl-gbdt/src/xgboost/file-input::file-uri
                #p"/data/train set.libsvm" :libsvm nil))))

(deftest file-uri-refuses-a-path-holding-a-query-separator
  (ok (handler-case
          (progn (cl-gbdt/src/xgboost/file-input::file-uri
                  #p"/data/train.csv?format=libsvm" :csv nil)
                 nil)
        (cl-gbdt/src/conditions:unsupported-argument () t))))

(deftest file-uri-refuses-a-path-holding-a-fragment-separator
  (ok (handler-case
          (progn (cl-gbdt/src/xgboost/file-input::file-uri
                  #p"/data/train#1.csv" :csv nil)
                 nil)
        (cl-gbdt/src/conditions:unsupported-argument () t))))

(deftest file-uri-refuses-a-format-key-among-the-extra-parameters
  (ok (handler-case
          (progn (cl-gbdt/src/xgboost/file-input::file-uri
                  #p"/data/train.csv" :csv '(:format "libsvm"))
                 nil)
        (cl-gbdt/src/conditions:unsupported-argument () t))))

;; Task 4's carried-forward finding: file-uri did not guard its own FORMAT argument, only
;; PAIRS -- a FORMAT keyword rendering to "csv&format=libsvm" would compose a second
;; `format' key the same way a bad :uri-parameters entry would. create-dataset-from-file's
;; own check order makes this unreachable in practice (FORMAT is restricted to :libsvm,
;; :csv or :binary before file-uri is ever called), but file-uri is the one function in
;; this branch whose whole job is preventing injection, so it should not depend on one
;; caller getting the order right.
(deftest file-uri-refuses-a-format-holding-a-reserved-character
  (ok (handler-case
          (progn (cl-gbdt/src/xgboost/file-input::file-uri
                  #p"/data/train.csv" :|csv&format=libsvm| nil)
                 nil)
        (cl-gbdt/src/conditions:unsupported-argument () t))))
