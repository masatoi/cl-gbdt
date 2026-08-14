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
           #:file-uri
           #:%resolve-file-path))

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

(defun %directory-p (path)
  "True when PATH resolves, via `truename', to a directory. Portable ANSI Common Lisp has
no way to ask more than this -- checked by resolving PATH and testing whether the
resulting pathname's NAME component is NIL, which is how every directory's own truename
comes back regardless of whether PATH itself carried a trailing separator. NIL when PATH
does not exist at all, or resolves to anything else `truename' can name -- an ordinary
regular file included, whose NAME component is never NIL.

`detect-file-format' calls this first, before either magic-byte check has opened
anything, so a directory is refused as a deliberate decision: `open' succeeds against one
and only a later read fails, and dmlc itself never errors on a directory PATH at all -- it
lists it and parses every file inside as though each had been declared the caller's
FORMAT, the same SIGSEGV-reachable mismatch a single wrong file is.

What this does NOT detect, because ANSI Common Lisp has no portable way to ask: whether
PATH names a FIFO, or a character or block device, rather than an ordinary regular file.
An earlier version of this function used `sb-posix:stat' to ask that too; it was removed
because the project owner declined to add `sb-posix' as a further SBCL-specific
dependency for this one check, not because removing it made this file, or this backend,
any more portable -- `file-uri' below already calls `sb-ext:native-namestring', which
does not exist outside SBCL and was added before this removal, and
`src/xgboost/native.lisp' has pinned arrays with `sb-sys:with-pinned-objects' since
before this branch began. Both keep this backend SBCL-only regardless of what
`%directory-p' does; the trade this removal made bought back no portability at all, only
the loss of a check that would have caught a FIFO or a device file. `%read-byte-line''s
own cap still bounds an unbounded device such as `/dev/zero'; a FIFO with no writer is
left as a documented limitation -- see `create-dataset-from-file''s docstring."
  (let ((truename (handler-case (truename path) (file-error () nil))))
    (and truename (null (pathname-name truename)))))

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

(defparameter +max-first-line-bytes+ (* 1024 1024)
  "The most `%read-byte-line' will ever read for one line before giving up and reporting
it truncated, rather than continuing to look for a terminating LF that may never come.

`/dev/zero' -- an infinite stream of zero bytes, no LF ever -- made the unbounded version
of `%read-byte-line' exhaust the heap. `%directory-p' cannot refuse a device file, ANSI
Common Lisp having no portable way to ask what kind of special file PATH names, so this
cap is the ONE defense `/dev/zero' has: nothing upstream of `%read-byte-line' stops it from
being opened and read. It also closes a Minor Task 2's own review found on an ordinary
REGULAR file with a pathologically long first line and no newline, a case `%directory-p'
was never going to catch either way (nothing about `truename' says how long one line
inside a file is): measured, the unbounded version conses one cons cell per byte, 142.7 MB
for an 8 MiB single-line file. One mebibyte is generous for any first line a real LIBSVM or
CSV file would have -- record section 1's fixtures are under 40 bytes -- while still
bounding the pathological case to a small, fixed multiple of itself rather than to the
size of the file.")

(defun %blank-line-p (line)
  "True when LINE, a vector of (unsigned-byte 8), contains nothing but the octets for
space (32), tab (9), or carriage return (13) -- `detect-file-format''s definition of a
line to skip while looking for the first line with content, including a
CRLF-terminated blank line whose trailing CR `%read-byte-line' leaves in place."
  (every (lambda (byte) (member byte '(32 9 13))) line))

(defun %read-byte-line (stream)
  "The octet analogue of `(read-line stream nil nil)': read bytes from STREAM up to and
excluding the next #x0A (LF), end of file, or `+max-first-line-bytes+' bytes, whichever
comes first, and return them as a fresh (unsigned-byte 8) vector -- or NIL when STREAM is
already at end of file with nothing left to return, the one case that also ends the loop
in `%first-non-blank-line'. Second value TRUNCATED: true once `+max-first-line-bytes+'
bytes have been read without an LF having already appeared among them, in which case LINE
holds exactly that many bytes, not the whole line; NIL otherwise, including at end of file.

TRUNCATED does not look ahead at the byte the cap stopped short of. Review round 4,
Finding N10: a COMPLETE, LF-terminated line of EXACTLY `+max-first-line-bytes+' content
bytes also reports TRUNCATED, because the cap is checked, and reached, before the byte
immediately after it -- which would have been that very LF -- is ever read. Measured: a
line one byte shorter classifies normally; a line of exactly the cap, or any longer,
reports TRUNCATED regardless of what its next byte actually is. This is the fail-safe
direction rather than a bug worth routing around: TRUNCATED only ever makes
`detect-file-format' answer :UNKNOWN, which only ever REFUSES a declared FORMAT
downstream, so the boundary case is refused rather than silently misjudged either way.

Classifying on octets rather than decoded characters is the point of this function: the
rule `detect-file-format' applies only ever inspects ASCII code points (comma, space,
tab, colon, digits), so nothing about it needs a decoded string, and reading bytes here
means a file whose contents are not valid text -- a latin-1 CSV with an accented column,
for one -- is classified correctly instead of failing to decode at all. See
`detect-file-format''s own docstring for why that matters: the alternative, mapping a
decoding failure to :UNREADABLE, is unsound, because a caller-readable file being refused
for no reason visible in its own bytes would be the wrong failure mode entirely.

The cap: see `+max-first-line-bytes+''s own docstring for why it exists at all.
`detect-file-format' treats a TRUNCATED line as :UNKNOWN rather than classifying the
partial bytes read -- a token cut off exactly at the cap could otherwise misclassify a
genuine, if unusually long-lined, LIBSVM file as :CSV, and this function has no way to
tell a genuine cut token from a malformed one."
  (let ((first (read-byte stream nil nil)))
    (cond ((null first) (values nil nil))
          ((= first 10) (values (make-array 0 :element-type '(unsigned-byte 8)) nil))
          (t (let ((bytes (list first))
                   (count 1)
                   (truncated nil))
               (loop
                 (when (>= count +max-first-line-bytes+)
                   (setf truncated t)
                   (return))
                 (let ((next (read-byte stream nil nil)))
                   (cond ((null next) (return))
                         ((= next 10) (return))
                         (t (push next bytes) (incf count)))))
               (values (coerce (nreverse bytes) '(vector (unsigned-byte 8))) truncated))))))

(defun %first-non-blank-line (path)
  "Return (VALUES LINE TRUNCATED): the first line of the file at PATH, as a vector of
(unsigned-byte 8), for which `%blank-line-p' is false, or NIL when every line is blank or
PATH has no lines at all -- a zero-byte file included, where the very first
`%read-byte-line' is already end-of-file. TRUNCATED is true when that first non-blank
line hit `+max-first-line-bytes+' -- `%read-byte-line''s own second value, passed
straight through the moment it is seen, since a line this function would otherwise have
picked as \"the one to classify\" hit the cap instead; see that function's own docstring
for the exact boundary, which is one byte earlier than \"before a terminating LF\" alone
would suggest. Opens PATH as
octets, never as text, so a file that is not valid text under any decoding is read
exactly like any other file rather than signalling."
  (with-open-file (stream path :direction :input :element-type '(unsigned-byte 8))
    (loop
      (multiple-value-bind (line truncated) (%read-byte-line stream)
        (cond ((null line) (return (values nil nil)))
              (truncated (return (values line t)))
              ((not (%blank-line-p line)) (return (values line nil))))))))

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

(defun %qid-token-p (token)
  "True when TOKEN, a vector of (unsigned-byte 8), is libsvm's ranking `qid:<group>' token
-- the literal ASCII bytes `q' `i' `d' `:' (113 105 100 58) followed by one or more ASCII
digit octets (48-57) and nothing else.

Measured against the vendored XGBoost 3.3.0 (docs/superpowers/specs, PR review of this
branch's merged form): a libsvm ranking file's row is `<label> qid:<group> <index>:<value>
...' -- `qid:1 1:0.5 2:0.3' -- and `XGDMatrixCreateFromURI' declared `:libsvm' reads such a
file cleanly, reporting the same row and column shape as the identical row with `qid:1'
removed and correctly recovering the group boundaries `qid' encodes (`XGDMatrixGetUIntInfo'
under `\"group_ptr\"' read back `(0 2 4)' for a two-row-per-group, two-group fixture).
LightGBM's `LGBM_DatasetCreateFromFile', separately measured against the same file, refuses
it outright with its own `\"Input format error when parsing as LibSVM\"' -- a real limitation
of that library's own parser, not of this function or of anything `create-dataset-from-file'
gates, and irrelevant to `%classify-line' below, which exists only to keep a real libsvm
ranking file from being misclassified as :CSV and refused by XGBOOST's format-mismatch gate
for a file XGBoost itself reads correctly.

`%classify-line' checks this ONLY against the token immediately after the label -- the
position the format itself puts `qid' in -- not against every token on the line: a `qid'-
shaped token anywhere else is a malformed row this function has no obligation to rescue,
and `%libsvm-token-p' failing on it (no digits before `qid''s own letters) is what still
sends such a line to :CSV, the safe refusal, rather than a loosened rule accepting a shape
libsvm's own grammar does not put there."
  (and (>= (length token) 5)
       (= (aref token 0) 113)   ; q
       (= (aref token 1) 105)   ; i
       (= (aref token 2) 100)   ; d
       (= (aref token 3) 58)    ; :
       (every (lambda (byte) (<= 48 byte 57)) (subseq token 4))))

(defun %classify-line (line)
  "Classify LINE, a vector of (unsigned-byte 8) already known not to be blank, as :CSV
or :LIBSVM.

A comma (44) anywhere on LINE decides :CSV outright, before LINE is even split into
tokens -- this order is load-bearing; see `detect-file-format''s docstring. Otherwise LINE
is split on runs of space and tab octets. When there are at least two tokens and the
SECOND -- the position libsvm's own ranking format puts it in, immediately after the
label -- is a `%qid-token-p' `qid:<group>' tag, it is set aside rather than checked as a
candidate feature pair; every token after it (rather than every token after the label) is
then required to be a `%libsvm-token-p' feature pair, and at least one such token must
remain. Without a `qid' token in that position, the rule is what it always was: at least
two tokens, and every token after the first a `%libsvm-token-p' feature pair.

The strict reading, not a loosened one: `qid' is recognized ONLY immediately after the
label, never scanned for elsewhere on the line. A `qid'-shaped token anywhere else --
`1 1:0.5 qid:1 2:0.3', malformed -- is checked by `%libsvm-token-p' like any other token,
fails it (no digits before `qid''s own letters), and the whole line falls through to :CSV,
the safe refusal, rather than the rule being loosened to find `qid' wherever it appears.
See `%qid-token-p' for the measurement establishing that XGBoost accepts a genuine `qid'
row and LightGBM does not, and why this rule exists only for XGBoost's own gate."
  (if (find 44 line)
      :csv
      (let ((tokens (%split-on-whitespace-runs line)))
        (if (< (length tokens) 2)
            :csv
            (let ((feature-tokens (if (%qid-token-p (second tokens))
                                       (cddr tokens)
                                       (rest tokens))))
              (if (and feature-tokens (every #'%libsvm-token-p feature-tokens))
                  :libsvm
                  :csv))))))

(defun detect-file-format (path)
  "Classify the file at PATH as one of :LIBSVM, :CSV, :BINARY, :UNKNOWN, or :UNREADABLE.

Checked in this order:

-1. `%directory-p', before PATH is opened at all. A directory reports :UNREADABLE
   immediately. See that function's own docstring for why this is a deliberate check
   rather than left to however `open' and a subsequent read happen to fail on one, and
   for what it cannot detect portably (a FIFO, or a character or block device).
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
   A first non-blank line of `+max-first-line-bytes+' bytes or more before an LF is
   found also reports :UNKNOWN, rather than classifying the truncated prefix -- including,
   by one byte, a COMPLETE line of EXACTLY that many content bytes, whose own terminating
   LF is never looked at (review round 4, Finding N10; fail-safe, not a bug: the boundary
   case refuses rather than guesses either way) -- see `%read-byte-line''s own docstring
   for the exact boundary and for why guessing from a cut-off token is worse than refusing.
2. A COMMA anywhere on that line reports :CSV. Stop.
3. Otherwise split the line into tokens on runs of SPACE and TAB. At least 2 tokens
   required. When the SECOND token -- immediately after the label, the position libsvm's
   ranking format puts it in -- matches `qid:<digits>' exactly, it is set aside rather
   than checked as a feature pair; every token after it (the label's token after that, if
   `qid' was not present) matching <digits>:<rest> -- at least one digit before the colon,
   at least one character after it, no comma in the token -- and at least one such token
   remaining, reports :LIBSVM. `qid' is recognized only in that one position, never
   scanned for elsewhere on the line -- see `%qid-token-p' and `%classify-line' for the
   measurement behind this (XGBoost reads a genuine `qid' row correctly; LightGBM refuses
   it, a limitation of that library's own parser this rule has no bearing on).
4. Otherwise, :CSV.

Steps 1 through 4 run once, against ONE line, and this function never reads a second one:
every verdict but :UNKNOWN, :UNREADABLE and :BINARY comes from that first non-blank line
alone. The implication, not just the mechanism: a file whose first line is genuinely
libsvm-shaped but whose later rows are not is classified :LIBSVM from that first line,
with nothing here saying anything about the rest of the file at all -- classifying it is
this function's whole job, and the rest of the file is not read to do it. PR #36's
re-review raised exactly this as a Critical finding; `create-dataset-from-file''s own
docstring records what was measured afterward (ten runs, no crash reproduced in any) and
what was only inferred, not confirmed -- see that docstring and
`docs/superpowers/specs/2026-08-13-file-input-measurements.md' section 12 for the record.

Step 2 must run before step 3 and must never be reordered: measured (record section 1),
without the comma guard the ordinary CSV line \"2024-01-01 12:00:00,1.0,2.0\" classifies
as LIBSVM, because \"12:00:00,1.0\" satisfies <digits>:<rest> -- and libsvm-declared-on-CSV
is the direction that SIGSEGVs XGBoost inside a non-Lisp thread no `handler-case' can
catch (record section 4). Only the comma-guarded form of this rule survives that line.

:UNREADABLE comes from step -1's deliberate check (a directory), or from `open' or a
later read signalling `file-error' or `stream-error' -- most concretely a missing file
(`file-error'), and also a FIFO with no writer or certain devices, which step -1 cannot
name portably and so leaves to whatever `open'/`read' does with them; see `%directory-p'
and `create-dataset-from-file' for that limitation stated plainly. It is NOT what a file
whose bytes are not valid text produces: every read in this function, magic checks and
the line-classification rule alike, is octets throughout -- `%first-non-blank-line' never
decodes -- so a latin-1 CSV or any other file whose bytes are not valid UTF-8 is
classified by its ASCII byte values exactly like any other file, typically :CSV or
:LIBSVM, rather than being refused. That is deliberate: mapping a decoding failure to
:UNREADABLE would have let a perfectly XGBoost-readable file be refused for no reason
this function can see in its own bytes. Nothing else returns :UNREADABLE: a file too
short for either magic check simply fails both and falls through to the text rule,
unreadability included.

Unlike an earlier version of this function, :UNREADABLE is no longer a signal that the
gate built on top of this function (`cl-gbdt/xgboost:create-dataset-from-file') passes
through to the foreign call. Review round 2 found that XGBoost does not treat every
:UNREADABLE case as an error either: a directory is not an error to dmlc, it is a file
list, and it parses every entry -- so \"an unopenable file is XGBoost's own to report
cleanly\", the premise the old pass-through rested on, held for a missing plain file and
did not hold for a directory or a multi-path string dmlc's own URI syntax accepts. PATH
being :UNREADABLE to this function now means `create-dataset-from-file' refuses it with
`file-format-mismatch' -- naming :UNREADABLE as DETECTED -- exactly as it refuses any
other verdict that is not an exact match with the caller's declared format; nothing this
function has not itself opened and classified ever reaches dmlc.

Three blind spots, measured and left as they are rather than smoothed over (record
section 1):
- A libsvm file whose rows carry a label and no features (each line just \"1\") reports
  :CSV. PR #36 review, Minor M1: this is refused by step 3's \"at least one feature-pair
  token must remain\" clause, not by a bare token-count check -- \"1 qid:1\" is a real
  libsvm file XGBoost reads (4 rows, 0 features, identical to this case) with TWO tokens,
  not one, and is refused the same way once `qid' is set aside and nothing is left to
  check. Not fatal either way -- sent as CSV it reads as a one- or two-column CSV.
- Malformed libsvm -- a non-numeric index, a trailing bare token, a comma inside a
  token -- reports :CSV for the same reason. None of the three is fatal sent as CSV.
- Space-delimited numbers (\"1 1.0 2.0 3.0\") report :CSV, which is the right answer for
  the gate this function exists to support: declared :LIBSVM the file SIGSEGVs XGBoost
  and the gate stops it. But XGBoost's own CSV reader is comma-only, so the file then
  silently yields the wrong shape -- 4 rows by 1 column, no label. This function has no
  third verdict to give that case; it can only say which of :LIBSVM or :CSV the first
  line resembles."
  (handler-case
      (if (%directory-p path)
          :unreadable
          (if (or (%starts-with-bytes-p path +xgboost-binary-magic+)
                  (%starts-with-bytes-p path +lightgbm-binary-magic+))
              :binary
              (multiple-value-bind (line truncated) (%first-non-blank-line path)
                (cond (truncated :unknown)
                      (line (%classify-line line))
                      (t :unknown)))))
    ((or file-error stream-error) () :unreadable)))

;;; ---------------------------------------------------------------------------
;;; %resolve-file-path -- the single resolution `create-dataset-from-file' feeds to
;;; both `detect-file-format' and `file-uri'

(defun %resolve-file-path (path)
  "Resolve PATH, a pathname designator, to the one `truename' `create-dataset-from-file'
then hands to BOTH `detect-file-format' and `file-uri', or NIL when PATH cannot be
resolved to a single existing file at all -- missing, wild (`truename' itself signals
`file-error' for a wild pathname, per the standard), or a symlink whose target does not
exist.

Review round 3, Finding N4 (Critical): before this function existed,
`create-dataset-from-file' called `detect-file-format' and `file-uri' on the SAME PATH
argument, but each resolved it independently and differently -- `detect-file-format'
through `open', which merges a relative PATH against Lisp's `*default-pathname-defaults*'
and expands a leading `~' to the caller's home directory; `file-uri' through a bare
`namestring', which prints PATH's own components verbatim, relative or `~'-prefixed or
however else it was spelled, with no such resolution at all. Two reproduced SIGSEGVs
followed directly: a relative PATH classified against one directory (Lisp's
`*default-pathname-defaults*') while dmlc, reading `file-uri''s literal relative string,
opened whatever the OS process's own working directory happened to be -- a different
directory whenever the two disagree, which nothing stops them from doing; and `~/x.libsvm'
similarly classified via `truename''s `~' expansion while dmlc received the unexpanded
string verbatim. Two resolutions of one designator make the file classified and the file
named in the URI the same only by coincidence.

The fix is this function: resolve PATH to a `truename' exactly ONCE, and pass that SAME
resolved pathname to both consumers, so there is exactly one place a path designator
becomes a concrete file and both readers agree on it by construction -- an IDENTITY the
two consumers now share, rather than two independent guesses this project would otherwise
have to keep proving stay equal for every pathname shape dmlc's own URI syntax might
one day turn out to care about. That is a deliberate narrowing of what this whole
mechanism promises: earlier design language said nothing not opened and classified by
this wrapper would reach dmlc \"including shapes nobody has thought of yet\", which
describes what must NOT get in and commits to out-guessing dmlc's syntax forever. The
contract this function actually enforces is checkable by reading its own two call sites
in `create-dataset-from-file' rather than by adversarial probing: the file dmlc opens is
the file this function resolved, because there is one resolution and both use it.

Not a guarantee that PATH still names the same file bytes when the foreign call actually
runs: nothing between this resolution and `XGDMatrixCreateFromURI' holds the file open or
otherwise prevents it being replaced on disk in between -- a TOCTOU window this wrapper
cannot close from Lisp. What this function removes is this wrapper disagreeing with
itself about which file that is; a second process racing to replace it is a different,
unclosable problem."
  (handler-case (truename path)
    (file-error () nil)))

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
  "True when CHAR is one of dmlc's own URI separators -- '?', '#', '&', or ';' -- any of
which, appearing inside a rendered query key or value, could open a second query segment
that `file-uri' did not intend, including a second `format' key, or -- ';' specifically,
review round 2's Finding N2 -- turn one path into dmlc's own multi-path list."
  (member char '(#\? #\# #\& #\;)))

(defun %pair-unsafe-p (pair)
  "True when PAIR's key or value, once rendered the way `file-uri' renders it, contains
a reserved character per `%uri-reserved-char-p'."
  (or (some #'%uri-reserved-char-p (string (car pair)))
      (some #'%uri-reserved-char-p (princ-to-string (cdr pair)))))

(defun %check-file-uri-arguments (path namestring format pairs)
  "Signal `unsupported-argument' when PATH, NAMESTRING, FORMAT, or PAIRS would let a
caller smuggle a second `format' key past the one `file-uri' composes, or let dmlc read
something other than the single file PATH names -- see `file-uri''s docstring for why
both characters, the key name, a reserved character inside a rendered key or value, a
reserved character inside FORMAT itself, and PATH being a wild pathname are all refused.

The FORMAT check closes a gap found by review rather than by measurement: `file-uri'
already refused a smuggled `format' key arriving through PAIRS, but nothing stopped FORMAT
itself from being a keyword such as `:|csv&format=libsvm|', which would render past the
gate the same way a bad PAIRS entry would. `create-dataset-from-file' in
`cl-gbdt/src/xgboost/api' already restricts FORMAT to `:libsvm', `:csv' or `:binary' before
`file-uri' is ever reached, so this is the second half of a belt-and-braces check -- an
invariant that depended on one caller getting the check order right was judged too weak for
the one function in this branch whose whole job is preventing injection.

The wild-pathname check closed a third instance of the same structural hole review found
twice already (a `?'/`#' in PATH, a `&' inside a PAIRS value): `file-uri''s guard list and
`detect-file-format''s classification were, at the time, two halves of one contract, and
anything the guard did not catch, a permissive `:unreadable' handling once handed to dmlc
unexamined. A PATH SBCL parses as a wild pathname (`a*.csv', `[a].csv') fails `open' with
a `file-error' subtype -- dmlc does not open a wild namestring as a single filename, it
glob-expands it, so a declared `:libsvm' could reach a real CSV file `detect-file-format'
never classified at all. `wild-pathname-p' is `NIL' for every ordinary path, a genuinely
missing plain file, and a path containing a space (record section 9's case), so this
refuses nothing that currently works.

The `;' check closes a fourth instance, review round 2's Finding N2: dmlc splits a URI on
`;' into a list of several paths and reads each, so `a.libsvm;b.csv' declared `:libsvm'
could match on the first segment and reach a second file `detect-file-format' never saw
at all -- unlike the wild-pathname case, this one did not even need `:unreadable' to be
permissive, since `a.libsvm;b.csv' is not itself a wild pathname and would have composed
cleanly. Review round 2 also removed `detect-file-format''s `:unreadable' pass-through
entirely (see that function's own docstring) rather than adding a fifth entry to this
list the next time dmlc's URI syntax turns out richer than whatever this file currently
enumerates; PATH must now resolve to a single existing, non-directory file (`%directory-p'
refuses a directory; a FIFO or device is not portably detectable and is left as a
documented limitation -- see `create-dataset-from-file'), and every `detect-file-format'
verdict but an exact match is a refusal in `create-dataset-from-file'. This function's
checks stay as a first line of defense regardless -- refusing a wild or `;'-holding PATH
here, before `detect-file-format' even runs, gives a caller a more specific reason than
the generic mismatch that catching it downstream would report.

**A fifth check, and the only one of the five that is precautionary rather than a fix for
a demonstrated hole: a literal `*' or `[' anywhere in PATH's namestring is refused.**
`wild-pathname-p' is `NIL' for a PATH built through `sb-ext:parse-native-namestring'
carrying one of those characters LITERALLY -- `star*file.libsvm', a real filename, not a
CL wildcard pattern -- so the wild-pathname check above does not catch it, and `file-uri'
writes that character verbatim into the URI via `native-namestring', with no CL escaping
at all. dmlc's URI layer is documented to glob-expand a path; this function therefore does
not hand it a string containing either glob metacharacter, whether or not doing so would
actually expand on any particular build.

PR #36's second re-review raised exactly this case, and it was measured, not assumed,
before this check was added -- ten runs against the vendored XGBoost 3.3.0
(`docs/superpowers/specs/2026-08-13-file-input-measurements.md' section 13). **The hazard
did NOT reproduce.** A literal filename that exists on disk is opened directly, not
glob-matched; `namestring''s backslash-escaping and `native-namestring''s bare form gave
IDENTICAL results everywhere both were tried; and a genuine pattern with no literally-
matching file reported zero matches rather than expanding to the files a shell's own glob
would find. This check is therefore precautionary, not a fix for a crash this branch
reproduced: the measurement covers one build, one platform, and local paths only, and says
nothing about a `file://' URI, another platform's dmlc, or the next XGBoost version. The
asymmetry that justifies refusing anyway, absent a reproduction: the cost of being wrong
about dmlc's glob behaviour on some OTHER build is a dead process, and the cost of this
guard is a filename built through `parse-native-namestring' that carries a literal `*' or
`[' -- exotic, and this function is not the only way to reach `XGDMatrixCreateFromURI' for
one.

This is also what makes review round 4's Finding N9 -- `file-uri' composing from
`native-namestring' rather than `namestring' -- safe regardless of whether dmlc escapes,
globs, or does neither on whatever version is actually loaded: N9 removed an unstated
premise (that dmlc's own glob parser happens to treat a backslash the way SBCL's pathname
printer does), and this check is what keeps that removal from depending on a new one."
  (when (wild-pathname-p path)
    (error 'unsupported-argument
           :backend :xgboost
           :argument "file-uri's path"
           :reason (format nil "~S is a wild pathname; dmlc glob-expands a wildcard ~
                                pattern rather than opening it as the single file it ~
                                names, which could reach a file detect-file-format ~
                                never classified" path)))
  (when (or (find #\? namestring) (find #\# namestring) (find #\; namestring))
    (error 'unsupported-argument
           :backend :xgboost
           :argument "file-uri's path"
           :reason (format nil "~S contains a '?', a '#', or a ';', dmlc's own query, ~
                                fragment, and multi-path separators; a path holding any ~
                                of them could smuggle in a second 'format' key or turn ~
                                one path into dmlc's own ';'-separated list of several"
                           namestring)))
  (when (or (find #\* namestring) (find #\[ namestring))
    (error 'unsupported-argument
           :backend :xgboost
           :argument "file-uri's path"
           :reason (format nil "~S contains a literal '*' or '[' -- see this function's ~
                                own docstring for why a literal glob metacharacter is ~
                                refused as a precaution regardless of what wild-pathname-p ~
                                reports for it" namestring)))
  (when (and (not (eq format :binary)) (some #'%uri-reserved-char-p (string format)))
    (error 'unsupported-argument
           :backend :xgboost
           :argument "file-uri's format"
           :reason (format nil "~S contains a '?', '#', or '&' once rendered, which ~
                                could open a second query segment -- including a second ~
                                'format' key -- past this function's own gate" format)))
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

Signals `unsupported-argument' before composing anything when PATH is a wild pathname
(`wild-pathname-p' true, e.g. `a*.csv' or `[a].csv') -- dmlc glob-expands such a namestring
rather than opening it as the single file it names, which can reach a file
`detect-file-format' never classified at all, defeating the format-mismatch gate the same
way a smuggled `format' key would -- or when PATH's namestring contains a `?', a `#', or a
`;' -- dmlc's own query, fragment, and multi-path separators, so a path holding any of
them could append a second `format' key of its own or turn one path into dmlc's own
`;'-separated list of several, each read and matched against the single declared FORMAT
independently -- or when FORMAT itself, once rendered, contains a `?', `#', `&', or `;' --
the identical smuggling risk one argument over, closed even though every caller in this
codebase already restricts FORMAT to `:libsvm', `:csv' or `:binary' before reaching here --
or when URI-PARAMETERS holds a `format' key under any case, the same smuggling risk from a
third side -- or when any URI-PARAMETERS key or value, once rendered into the query
string, itself contains a `?', `#', `&', or `;': any of the four could open a second query
segment there too, or reintroduce the multi-path case from inside a parameter value -- or
when PATH's namestring contains a literal `*' or `[', a precaution against dmlc's
documented glob-expansion rather than a fix for a reproduced crash; see
`%check-file-uri-arguments' for the measurement this last one rests on.

Does not percent-encode PATH. Measured (record section 9): dmlc accepts an unencoded
space in the path and REJECTS the identical path percent-encoded as %20, with \"Cannot
find any files that matches the URI pattern\". Encoding here would break a path that
works unencoded.

Writes PATH's `sb-ext:native-namestring' into the URI, not `namestring' -- review round
4, Finding N9: `namestring' backslash-escapes a character that is a CL pathname wildcard
marker but happens to be literal in an actual filename (an asterisk, concretely), so a
real file named `star*file.libsvm' namestrings as `star\\*file.libsvm'. The identity this
whole mechanism rests on -- the file dmlc opens is the file `detect-file-format'
classified -- would then depend on dmlc's own glob parser ALSO treating a backslash as an
escape for the following character, which measured to be true, but was never something
this function's own contract established; two independent programs' escaping
conventions agreeing is a coincidence to route around, not to depend on silently.
`native-namestring' is the OS's own bytes for PATH with no CL-specific escaping added at
all, identical to `namestring' for every path this project's own fixtures have ever
exercised (ordinary names, one with a space) and different only for a name a wildcard
character is literally part of. Checked for the reserved characters via the ORDINARY
`namestring' first, in `%check-file-uri-arguments' below, since escaping only ever adds a
backslash and never introduces a `?', `#', `&', or `;' that was not already there, and
since that check is also what refuses PATH as wild before `native-namestring' -- which
signals its own error for a wild pathname -- is ever reached."
  (let ((namestring (namestring path))
        (pairs (%uri-parameter-pairs uri-parameters)))
    (%check-file-uri-arguments path namestring format pairs)
    (with-output-to-string (out)
      (write-string (sb-ext:native-namestring path) out)
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
