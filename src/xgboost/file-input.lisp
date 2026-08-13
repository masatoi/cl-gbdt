;;;; file-input.lisp --- The pure layer behind XGBoost's file-reading entry point.
;;;;
;;;; `detect-file-format' and `file-uri' make no foreign call and touch no handle: both
;;;; are read-only work over Lisp strings and a file's own bytes, so every branch here is
;;;; layer-1 testable without either shared library. Tasks 3 and 4 build the operation
;;;; that calls them -- `create-dataset-from-file' -- and the format-mismatch gate that
;;;; makes `detect-file-format''s answer load-bearing rather than merely informational.
;;;;
;;;; Every rule below is transcribed from
;;;; docs/superpowers/specs/2026-08-13-file-input-measurements.md, sections 1, 2, 8 and 9
;;;; -- measured against the vendored XGBoost 3.3.0 and LightGBM 4.7.0, not derived from
;;;; documentation. Where a docstring below cites "the record" or "measured", that file
;;;; is what it means.

(uiop:define-package #:cl-gbdt/src/xgboost/file-input
  (:use #:cl)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-argument)
  (:export #:detect-file-format
           #:file-uri))

(in-package #:cl-gbdt/src/xgboost/file-input)

;;; ---------------------------------------------------------------------------
;;; detect-file-format

(defparameter +xgboost-binary-magic+ #(#x01 #xAB #xFF #xFF)
  "First four bytes of an XGBoost binary DMatrix file, written by `XGDMatrixSaveBinary'.
Measured (record section 2): followed by the ASCII text \"version:\", but only these
four bytes are checked here.")

(defparameter +lightgbm-binary-magic+
  (map '(vector (unsigned-byte 8)) #'char-code "______LightGBM_Binary_File_Token______")
  "First 38 bytes of a LightGBM binary dataset file, written by `LGBM_DatasetSaveBinary',
as octets. LightGBM auto-detects this on its own load path with no parameter at all;
`detect-file-format' checks it too so a LightGBM binary file reports :BINARY here rather
than falling through to the text rule and being misread as garbled CSV.")

(defun %starts-with-bytes-p (path expected)
  "True when the file at PATH begins with the octets in EXPECTED, a vector of
\(unsigned-byte 8). False, not an error, when PATH is shorter than EXPECTED -- a short
file simply does not start with the magic. Signals `file-error' when PATH cannot be
opened at all; `detect-file-format' is what catches that."
  (with-open-file (stream path :direction :input :element-type '(unsigned-byte 8))
    (let* ((length (length expected))
           (buffer (make-array length :element-type '(unsigned-byte 8)))
           (read (read-sequence buffer stream)))
      (and (= read length) (every #'= buffer expected)))))

(defun %blank-line-p (line)
  "True when LINE, a vector of (unsigned-byte 8), contains nothing but the octets for
space (32), tab (9), or carriage return (13) -- `detect-file-format''s definition of a
line to skip while looking for the first line with content, including a
CRLF-terminated blank line whose trailing CR `%read-byte-line' leaves in place."
  (every (lambda (byte) (member byte '(32 9 13))) line))

(defun %read-byte-line (stream)
  "The octet analogue of `(read-line stream nil nil)': read bytes from STREAM up to and
excluding the next #x0A (LF), or end of file, and return them as a fresh
\(unsigned-byte 8) vector -- or NIL when STREAM is already at end of file with nothing
left to return, the one case that also ends the loop in `%first-non-blank-line'.

Classifying on octets rather than decoded characters is the point of this function: the
rule `detect-file-format' applies only ever inspects ASCII code points (comma, space,
tab, colon, digits), so nothing about it needs a decoded string, and reading bytes here
means a file whose contents are not valid text -- a latin-1 CSV with an accented column,
for one -- is classified correctly instead of failing to decode at all. See
`detect-file-format''s own docstring for why that matters: the alternative, mapping a
decoding failure to :UNREADABLE, is unsound, because :UNREADABLE is the one verdict the
gate built on top of this function passes straight through to the foreign call."
  (let ((first (read-byte stream nil nil)))
    (cond ((null first) nil)
          ((= first 10) (make-array 0 :element-type '(unsigned-byte 8)))
          (t (let ((bytes (list first)))
               (loop for next = (read-byte stream nil nil)
                     until (or (null next) (= next 10))
                     do (push next bytes))
               (coerce (nreverse bytes) '(vector (unsigned-byte 8))))))))

(defun %first-non-blank-line (path)
  "Return the first line of the file at PATH, as a vector of (unsigned-byte 8), for
which `%blank-line-p' is false, or NIL when every line is blank or PATH has no lines at
all -- a zero-byte file included, where the very first `%read-byte-line' is already
end-of-file. Opens PATH as octets, never as text, so a file that is not valid text under
any decoding is read exactly like any other file rather than signalling."
  (with-open-file (stream path :direction :input :element-type '(unsigned-byte 8))
    (loop for line = (%read-byte-line stream)
          while line
          unless (%blank-line-p line)
            return line)))

(defun %split-on-whitespace-runs (line)
  "Split LINE, a vector of (unsigned-byte 8), into tokens -- also (unsigned-byte 8)
vectors -- on runs of one or more space (32) or tab (9) octets, dropping any empty
token a leading, trailing, or repeated separator would otherwise produce."
  (let ((tokens '())
        (start nil))
    (loop for index from 0 below (length line)
          for byte = (aref line index)
          do (if (or (= byte 32) (= byte 9))
                 (when start
                   (push (subseq line start index) tokens)
                   (setf start nil))
                 (unless start
                   (setf start index))))
    (when start
      (push (subseq line start (length line)) tokens))
    (nreverse tokens)))

(defun %libsvm-token-p (token)
  "True when TOKEN, a vector of (unsigned-byte 8), matches libsvm's <digits>:<rest>
feature-pair shape: at least one ASCII digit octet (48-57) before the first colon (58),
at least one octet after it, and no comma (44) anywhere in TOKEN. Part of step 3 of
`detect-file-format''s rule."
  (let ((colon (position 58 token)))
    (and colon
         (plusp colon)
         (every (lambda (byte) (<= 48 byte 57)) (subseq token 0 colon))
         (< (1+ colon) (length token))
         (not (find 44 token)))))

(defun %classify-line (line)
  "Classify LINE, a vector of (unsigned-byte 8) already known not to be blank, as :CSV
or :LIBSVM.

A comma (44) anywhere on LINE decides :CSV outright, before LINE is even split into
tokens -- this order is load-bearing; see `detect-file-format''s docstring. Otherwise
LINE is split on runs of space and tab octets, and it reads as :LIBSVM only when there
are at least two tokens and every token after the first is a `%libsvm-token-p' feature
pair."
  (if (find 44 line)
      :csv
      (let ((tokens (%split-on-whitespace-runs line)))
        (if (and (>= (length tokens) 2)
                 (every #'%libsvm-token-p (rest tokens)))
            :libsvm
            :csv))))

(defun detect-file-format (path)
  "Classify the file at PATH as one of :LIBSVM, :CSV, :BINARY, :UNKNOWN, or :UNREADABLE.

Checked in this order:

0. Magic bytes, read before any line of text is read. The first four bytes
   #x01 #xAB #xFF #xFF mark an XGBoost binary DMatrix (`XGDMatrixSaveBinary'); the first
   38 bytes \"______LightGBM_Binary_File_Token______\" mark a LightGBM binary dataset
   (`LGBM_DatasetSaveBinary'). Either case reports :BINARY, without ever reaching the
   text rule below.
1. Otherwise, the first line of PATH that is not blank -- blank meaning every character
   is space, tab, or carriage return. No such line at all, whether PATH is a zero-byte
   file or a file of only blank lines, reports :UNKNOWN. This matters beyond the empty
   case: XGBoost accepts a blank-only file as a silent 0x0 DMatrix (record section 8), so
   :UNKNOWN turning into a mismatch is what stops that case too, not only a missing file.
2. A COMMA anywhere on that line reports :CSV. Stop.
3. Otherwise split the line into tokens on runs of SPACE and TAB. At least 2 tokens, and
   every token after the first matching <digits>:<rest> -- at least one digit before the
   colon, at least one character after it, no comma in the token -- reports :LIBSVM.
4. Otherwise, :CSV.

Step 2 must run before step 3 and must never be reordered: measured (record section 1),
without the comma guard the ordinary CSV line \"2024-01-01 12:00:00,1.0,2.0\" classifies
as LIBSVM, because \"12:00:00,1.0\" satisfies <digits>:<rest> -- and libsvm-declared-on-CSV
is the direction that SIGSEGVs XGBoost inside a non-Lisp thread no `handler-case' can
catch (record section 4). Only the comma-guarded form of this rule survives that line.

:UNREADABLE comes from `open' or a later read signalling `file-error' or `stream-error' --
a missing file (`file-error'), or a directory given as a path (SBCL opens a directory
successfully and fails only once `read-byte' or `read-sequence' touches it, as
`stream-error'). It is NOT what a file whose bytes are not valid text produces: every
read in this function, magic checks and the line-classification rule alike, is octets
throughout -- `%first-non-blank-line' never decodes -- so a latin-1 CSV or any other file
whose bytes are not valid UTF-8 is classified by its ASCII byte values exactly like any
other file, typically :CSV or :LIBSVM, rather than being refused. That is deliberate: a
gate downstream of this function (see Task 4's brief) treats :UNREADABLE as a
pass-through rather than a refusal, on the reasoning that an unopenable file is XGBoost's
own to report cleanly -- so mapping a decoding failure to :UNREADABLE would have let a
perfectly XGBoost-readable file reach the foreign call with no format check at all,
undoing this function's whole purpose. Nothing else returns :UNREADABLE: a file too
short for either magic check simply fails both and falls through to the text rule,
unreadability included.

Three blind spots, measured and left as they are rather than smoothed over (record
section 1):
- A libsvm file whose rows carry a label and no features (each line just \"1\") reports
  :CSV: it has one token, and step 3 requires at least two. Not fatal -- sent as CSV it
  reads as a one-column CSV.
- Malformed libsvm -- a non-numeric index, a trailing bare token, a comma inside a
  token -- reports :CSV for the same reason. None of the three is fatal sent as CSV.
- Space-delimited numbers (\"1 1.0 2.0 3.0\") report :CSV, which is the right answer for
  the gate this function exists to support: declared :LIBSVM the file SIGSEGVs XGBoost
  and the gate stops it. But XGBoost's own CSV reader is comma-only, so the file then
  silently yields the wrong shape -- 4 rows by 1 column, no label. This function has no
  third verdict to give that case; it can only say which of :LIBSVM or :CSV the first
  line resembles."
  (handler-case
      (if (or (%starts-with-bytes-p path +xgboost-binary-magic+)
              (%starts-with-bytes-p path +lightgbm-binary-magic+))
          :binary
          (let ((line (%first-non-blank-line path)))
            (if line (%classify-line line) :unknown)))
    ((or file-error stream-error) () :unreadable)))

;;; ---------------------------------------------------------------------------
;;; file-uri

(defun %uri-parameter-pairs (uri-parameters)
  "Return URI-PARAMETERS, a plist, as a list of (KEY . VALUE) conses in the same order."
  (loop for (key value) on uri-parameters by #'cddr
        collect (cons key value)))

(defun %format-key-p (pair)
  "True when PAIR's key, from `%uri-parameter-pairs', spells `format' under any case --
the one query key `file-uri' reserves for its own FORMAT argument."
  (string-equal (string (car pair)) "format"))

(defun %uri-reserved-char-p (char)
  "True when CHAR is one of dmlc's own URI separators -- '?', '#', or '&' -- any of
which, appearing inside a rendered query key or value, could open a second query segment
that `file-uri' did not intend, including a second `format' key."
  (member char '(#\? #\# #\&)))

(defun %pair-unsafe-p (pair)
  "True when PAIR's key or value, once rendered the way `file-uri' renders it, contains
a reserved character per `%uri-reserved-char-p'."
  (or (some #'%uri-reserved-char-p (string (car pair)))
      (some #'%uri-reserved-char-p (princ-to-string (cdr pair)))))

(defun %check-file-uri-arguments (namestring pairs)
  "Signal `unsupported-argument' when NAMESTRING or PAIRS would let a caller smuggle a
second `format' key past the one `file-uri' composes -- see `file-uri''s docstring for
why both characters, the key name, and a reserved character inside a rendered key or
value are all refused."
  (when (or (find #\? namestring) (find #\# namestring))
    (error 'unsupported-argument
           :backend :xgboost
           :argument "file-uri's path"
           :reason (format nil "~S contains a '?' or a '#', dmlc's own query and ~
                                fragment separators; a path holding either could smuggle ~
                                in a second 'format' key" namestring)))
  (when (some #'%format-key-p pairs)
    (error 'unsupported-argument
           :backend :xgboost
           :argument "file-uri's uri-parameters"
           :reason (format nil "a 'format' key is not allowed among the extra ~
                                parameters -- pass it as file-uri's own FORMAT ~
                                argument instead")))
  (let ((unsafe (find-if #'%pair-unsafe-p pairs)))
    (when unsafe
      (error 'unsupported-argument
             :backend :xgboost
             :argument "file-uri's uri-parameters"
             :reason (format nil "~S's key or value contains a '?', '#', or '&' once ~
                                  rendered, which could open a second query segment -- ~
                                  including a second 'format' key -- past this ~
                                  function's own gate" unsafe)))))

(defun file-uri (path format uri-parameters)
  "Compose the dmlc URI `XGDMatrixCreateFromURI' expects from PATH, FORMAT, and
URI-PARAMETERS: PATH's namestring, followed by a query string built from the other two.

FORMAT is a keyword, `:csv' or `:libsvm', lower-cased and appended as the query
\"format=<format>\". FORMAT `:binary' appends no format key at all -- measured (record
section 2): \"?format=binary\" is rejected outright (\"Unknown data type binary\"), and a
binary DMatrix loads only from a URI carrying no `?format=' at all. With URI-PARAMETERS
also empty, `:binary' therefore produces the bare namestring with no `?' in it.

URI-PARAMETERS is a plist of further query keys, e.g. `(:label_column 0)', each appended
as \"&key=value\" after the format key -- or, when FORMAT is `:binary', as the URI's only
query parameters, introduced with `?' instead of `&'.

Signals `unsupported-argument' before composing anything when PATH's namestring contains
a `?' or a `#' -- dmlc's own query and fragment separators, so a path holding either could
append a second `format' key of its own and defeat the format-mismatch gate built on top
of this function -- or when URI-PARAMETERS holds a `format' key under any case, the same
smuggling risk from the other side -- or when any URI-PARAMETERS key or value, once
rendered into the query string, itself contains a `?', `#', or `&': any of the three could
open a second query segment there too, the same smuggling risk one argument over.

Does not percent-encode PATH. Measured (record section 9): dmlc accepts an unencoded
space in the path and REJECTS the identical path percent-encoded as %20, with \"Cannot
find any files that matches the URI pattern\". Encoding here would break a path that
works unencoded."
  (let ((namestring (namestring path))
        (pairs (%uri-parameter-pairs uri-parameters)))
    (%check-file-uri-arguments namestring pairs)
    (with-output-to-string (out)
      (write-string namestring out)
      (let ((separator #\?))
        (unless (eq format :binary)
          (write-char separator out)
          (write-string "format=" out)
          (write-string (string-downcase (string format)) out)
          (setf separator #\&))
        (dolist (pair pairs)
          (write-char separator out)
          (write-string (string-downcase (string (car pair))) out)
          (write-char #\= out)
          (princ (cdr pair) out)
          (setf separator #\&))))))
