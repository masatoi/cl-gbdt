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

;; PR #36 review, finding P2: measured against the vendored library (scratchpad
;; qid-measurement.lisp, an isolated subprocess) that XGDMatrixCreateFromURI declared
;; :libsvm reads a genuine ranking row -- "1 qid:1 1:0.5 2:0.3" -- cleanly, reporting the
;; same 4-row-by-3-feature shape as the identical fixture with "qid:1"/"qid:2" removed,
;; and correctly recovering the group boundaries (group_ptr read back as (0 2 4) for two
;; rows per group). Before this fix, %libsvm-token-p rejected "qid:1" (no digits before
;; the colon) and the whole line fell through to :CSV, so a valid ranking file declared
;; :libsvm was refused by create-dataset-from-file's own gate with file-format-mismatch --
;; a false positive the gate exists to avoid, not the SIGSEGV-preventing refusal it exists
;; to provide.
(deftest detect-file-format-classifies-a-qid-ranking-file-as-libsvm
  (with-temporary-file (path "1 qid:1 1:0.5 2:0.3
0 qid:1 1:0.1 2:0.9
1 qid:2 1:0.6 2:0.2
0 qid:2 1:0.2 2:0.1
")
    (ok (eq :libsvm (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

;; The strict reading the coordinator asked for: qid is recognized ONLY immediately after
;; the label, the one position libsvm's own grammar puts it in -- not scanned for anywhere
;; on the line. A qid-shaped token appearing later is a malformed row this function has no
;; obligation to rescue; loosening the rule to find "qid:" wherever it appears is the
;; direction that can wrongly ACCEPT a shape libsvm's grammar does not produce, which this
;; whole branch has held to be the fatal mistake to risk. %libsvm-token-p rejects
;; "qid:1" (no digits before the colon) like any other malformed token, so the line falls
;; through to :CSV -- the safe refusal, not a SIGSEGV risk, since XGBoost's own CSV reader
;; would then reject or misparse it rather than crash.
(deftest detect-file-format-refuses-a-qid-token-outside-the-second-position
  (with-temporary-file (path "1 1:0.5 qid:1 2:0.3
")
    (ok (eq :csv (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

;; qid with no feature pairs left after it -- "1 qid:1" -- falls through to :CSV exactly
;; as the plain labels-only case above does: %classify-line requires at least one
;; feature-pair token to remain after setting qid aside, and none does here.
(deftest detect-file-format-classifies-a-qid-only-line-as-csv
  (with-temporary-file (path "1 qid:1
0 qid:1
")
    (ok (eq :csv (cl-gbdt/src/xgboost/file-input::detect-file-format path)))))

;; C1: a latin-1 CSV (an accented name column, e.g. "cafe" with an e-acute) is not valid
;; UTF-8 -- byte 233 (0xE9) alone, followed by a comma rather than a continuation byte,
;; fails UTF-8 decoding outright. Written byte-wise so it does not depend on the source
;; file's own encoding. Classifying on octets rather than decoded characters is what
;; makes this :CSV rather than :UNREADABLE or :UNKNOWN -- both of which would be wrong
;; here: :UNKNOWN and :UNREADABLE are each their own mismatch in Task 4's gate (neither is
;; a pass-through -- create-dataset-from-file refuses every detect-file-format verdict but
;; an exact match with the caller's declared format), so mapping a decoding failure to
;; either would wrongly refuse an ordinary CSV for no reason detect-file-format can see in
;; its own bytes.
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

;; Review round 1, Finding 1 (Critical): a wild pathname bypasses the gate entirely.
;; detect-file-format's open fails with a file-error subtype (a wild namestring cannot be
;; opened as a single file), which detect-file-format reports as :UNREADABLE -- and
;; create-dataset-from-file's gate refuses :UNREADABLE, exactly as it refuses any other
;; verdict that is not an exact match with the caller's declared format. But dmlc does not
;; open a wild namestring as a single file at all: it glob-expands the pattern and can
;; reach a real file detect-file-format never classified, which is exactly the fatal
;; direction this branch exists to refuse. This is the third instance of the same
;; structural hole review found (a `?'/`#' in the path, a `&' inside a parameter value):
;; file-uri's own guard list catches this one before detect-file-format is even reached,
;; giving a caller a more specific reason than the generic mismatch that catching it only
;; downstream would report.
(deftest file-uri-refuses-a-wild-pathname
  (ok (handler-case
          (progn (cl-gbdt/src/xgboost/file-input::file-uri #p"/data/a*.csv" :libsvm nil)
                 nil)
        (cl-gbdt/src/conditions:unsupported-argument () t))
      "file-uri accepted a * wildcard in the path"))

(deftest file-uri-refuses-a-bracket-wildcard-pathname
  (ok (handler-case
          (progn (cl-gbdt/src/xgboost/file-input::file-uri #p"/data/[a].csv" :libsvm nil)
                 nil)
        (cl-gbdt/src/conditions:unsupported-argument () t))
      "file-uri accepted a [] wildcard in the path"))

;; PR #36 second re-review, Critical: a LITERAL '*' -- built via
;; sb-ext:parse-native-namestring, not the ordinary pathname reader above -- names a real
;; file (star*file.libsvm) rather than a CL wildcard pattern, so wild-pathname-p is NIL and
;; the guard above never runs; file-uri would have written the literal asterisk into the
;; URI unescaped via native-namestring, which dmlc's own glob layer is documented to
;; expand. Measured afterward (docs/superpowers/specs, section 13) that this specific
;; hazard did not reproduce on the vendored library -- this refusal is precautionary, not
;; a fix for a demonstrated crash; see %check-file-uri-arguments's own docstring.
(deftest file-uri-refuses-a-literal-asterisk
  (let ((path (sb-ext:parse-native-namestring "star*file.libsvm")))
    (ok (null (wild-pathname-p path))
        "test setup: this path must not be a CL wildcard, or it would be caught by the \
existing guard instead of the one under test")
    (ok (handler-case (progn (cl-gbdt/src/xgboost/file-input::file-uri path :libsvm nil)
                              nil)
          (cl-gbdt/src/conditions:unsupported-argument () t))
        "file-uri accepted a literal * that wild-pathname-p does not see as wild")))

;; The same shape for '[', which globs as a bracket-alternation, not scanned for by
;; wild-pathname-p either once the pathname is built through parse-native-namestring.
(deftest file-uri-refuses-a-literal-bracket
  (let ((path (sb-ext:parse-native-namestring "brack[ab]file.libsvm")))
    (ok (null (wild-pathname-p path))
        "test setup: this path must not be a CL wildcard")
    (ok (handler-case (progn (cl-gbdt/src/xgboost/file-input::file-uri path :libsvm nil)
                              nil)
          (cl-gbdt/src/conditions:unsupported-argument () t))
        "file-uri accepted a literal [ that wild-pathname-p does not see as wild")))

;; The control: an ordinary path, a genuinely missing plain file, and a path containing a
;; space (record section 9's case, already proven not to be wild) must all keep composing.
(deftest file-uri-still-composes-ordinary-paths-after-the-wild-pathname-guard
  (ok (string= "/data/train.csv?format=csv"
               (cl-gbdt/src/xgboost/file-input::file-uri #p"/data/train.csv" :csv nil))
      "file-uri refused an ordinary path")
  (ok (string= "/data/no-such-file.csv?format=csv"
               (cl-gbdt/src/xgboost/file-input::file-uri
                #p"/data/no-such-file.csv" :csv nil))
      "file-uri refused a merely-missing path")
  (ok (string= "/data/train set.libsvm?format=libsvm"
               (cl-gbdt/src/xgboost/file-input::file-uri
                #p"/data/train set.libsvm" :libsvm nil))
      "file-uri refused a path containing a space"))

;; Review round 2, Finding N1 (Critical): a DIRECTORY as PATH is not caught by
;; wild-pathname-p (it holds no wildcard character at all), so under the round-1 design it
;; reached create-dataset-from-file's :UNREADABLE pass-through and dmlc listed the
;; directory, parsing every file inside as though each had been declared the caller's
;; format -- SIGSEGV. detect-file-format now checks %directory-p first, deliberately,
;; before opening anything, so this is decided without ever touching the filesystem's read
;; path at all.
(deftest detect-file-format-reports-unreadable-for-a-directory
  (let ((dir (merge-pathnames "cl-gbdt-file-input-test-dir/" (uiop:temporary-directory))))
    (ensure-directories-exist dir)
    (unwind-protect
         (ok (eq :unreadable (cl-gbdt/src/xgboost/file-input::detect-file-format dir))
             "detect-file-format did not report :unreadable for a directory")
      (uiop:delete-directory-tree dir :validate t))))

;; The shape review round 2 specifically probed: PATH given WITHOUT a trailing slash, so
;; Lisp's own pathname parser treats it as a file-shaped pathname (a NAME component, no
;; DIRECTORY-list entry for it) even though the thing it names on disk is a directory.
;; %directory-p resolves through `truename', which reflects what is actually on disk
;; regardless of how PATH itself was spelled, so this must refuse identically to the
;; trailing-slash case above.
(deftest detect-file-format-reports-unreadable-for-a-directory-without-a-trailing-slash
  (let ((with-slash (merge-pathnames "cl-gbdt-file-input-test-dir2/"
                                     (uiop:temporary-directory)))
        (without-slash (merge-pathnames "cl-gbdt-file-input-test-dir2"
                                        (uiop:temporary-directory))))
    (ensure-directories-exist with-slash)
    (unwind-protect
         (ok (eq :unreadable (cl-gbdt/src/xgboost/file-input::detect-file-format
                              without-slash))
             "detect-file-format did not report :unreadable for a slash-less directory path")
      (uiop:delete-directory-tree with-slash :validate t))))

;; Review round 2, Finding N2 (Critical): dmlc splits a URI on ';' into a list of several
;; paths and reads each independently, so "a.libsvm;b.csv" declared :libsvm could match on
;; its first segment while the second reached the library entirely unclassified -- SIGSEGV
;; the moment that second file's real contents disagreed. Unlike the wild-pathname case,
;; this shape is not itself a wildcard and would have composed a perfectly normal-looking
;; URI, so it needed its own guard rather than reusing wild-pathname-p.
(deftest file-uri-refuses-a-path-holding-a-multi-path-separator
  (ok (handler-case
          (progn (cl-gbdt/src/xgboost/file-input::file-uri
                  #p"/data/a.libsvm;b.csv" :libsvm nil)
                 nil)
        (cl-gbdt/src/conditions:unsupported-argument () t))
      "file-uri accepted a ';'-separated multi-path"))

;; Review round 3, Finding N4 (Critical): detect-file-format and file-uri used to be
;; called on the caller's own PATH independently, and each resolved it differently --
;; detect-file-format through `open', honouring Lisp's *default-pathname-defaults* and
;; expanding a leading '~'; file-uri through a bare `namestring', printing PATH verbatim
;; with neither. %resolve-file-path now resolves PATH to a truename exactly once, and
;; create-dataset-from-file hands that SAME resolved pathname to both. The cheapest test
;; of that property: a relative pathname and its already-resolved truename must compose
;; the identical URI once each has gone through %resolve-file-path, since both routes
;; must land on the one truename that decides what dmlc actually opens.
(deftest resolve-file-path-and-file-uri-compose-the-same-uri-for-a-relative-path
  (with-temporary-file (path "1,1.0,2.0,3.0
")
    (let* ((dir (make-pathname :directory (pathname-directory path)))
           (relative (make-pathname :name (pathname-name path) :type (pathname-type path)))
           (*default-pathname-defaults* dir))
      (let ((resolved-from-relative
              (cl-gbdt/src/xgboost/file-input::%resolve-file-path relative))
            (resolved-from-absolute
              (cl-gbdt/src/xgboost/file-input::%resolve-file-path path)))
        (ok (and resolved-from-relative resolved-from-absolute)
            "%resolve-file-path failed to resolve either the relative or the absolute form")
        (ok (string= (cl-gbdt/src/xgboost/file-input::file-uri resolved-from-relative
                                                                :csv nil)
                     (cl-gbdt/src/xgboost/file-input::file-uri resolved-from-absolute
                                                                :csv nil))
            "a relative path and its already-absolute form composed different URIs once \
resolved")))))

;; Review round 3, Finding N5 (Important), the read-cap half: %read-byte-line used to
;; read one line without bound, so a file with no LF anywhere -- /dev/zero being the
;; extreme case that exhausts the heap, verified separately in a subprocess since it
;; would hang or exhaust this test process's own heap -- made it read forever. A plain,
;; finite REGULAR file longer than +max-first-line-bytes+ with no LF anywhere exercises
;; the cap itself, quickly and safely: detect-file-format must report :UNKNOWN rather
;; than either hanging or guessing at a classification from a line cut off mid-token.
(deftest detect-file-format-reports-unknown-for-a-line-longer-than-the-read-cap
  (let ((path (merge-pathnames "cl-gbdt-file-input-cap-test.csv"
                               (uiop:temporary-directory)))
        (cap cl-gbdt/src/xgboost/file-input::+max-first-line-bytes+))
    (with-open-file (stream path :direction :output :if-exists :supersede
                                  :element-type '(unsigned-byte 8))
      ;; ASCII '1' repeated past the cap, no LF at all -- would classify :LIBSVM-shaped
      ;; garbage or hang were it not capped; contains no comma either, so a bug that
      ;; classified the truncated prefix would misreport this as :CSV or :LIBSVM rather
      ;; than :UNKNOWN, which is exactly the failure this test exists to catch.
      (dotimes (i (+ cap 10))
        (write-byte 49 stream)))
    (unwind-protect
         (ok (eq :unknown (cl-gbdt/src/xgboost/file-input::detect-file-format path))
             "detect-file-format did not report :unknown for a line past the read cap")
      (handler-case (delete-file path) (file-error () nil)))))
