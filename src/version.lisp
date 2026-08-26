;;;; version.lisp --- Recorded LightGBM/XGBoost version ranges, and the pure
;;;; comparison used to warn when a loaded library falls outside them.
;;;;
;;;; Two different claims are recorded here, deliberately kept distinguishable rather
;;;; than blended into one:
;;;;
;;;;   verified   the exact version(s) the functional suite (layer 2, 105 assertions)
;;;;              has actually trained and predicted against
;;;;   inferred   the wider span `tools/check-upstream.lisp' can only argue about from
;;;;              comparing C header declarations: the 38 functions cl-gbdt imports (18
;;;;              from LightGBM, 20 from XGBoost) are textually unchanged across it, so
;;;;              a library in this span is presumed ABI-compatible even though nothing
;;;;              has actually run a model against it
;;;;
;;;; The runtime check below (`check-backend-version') gates on the wider INFERRED
;;;; bounds, not the narrower VERIFIED ones -- seeing a version different from the one
;;;; the functional suite happens to run against is the common case for a compatible
;;;; caller, not a signal of trouble. Warning on that difference would fire on nearly
;;;; every correctly configured user, and a warning that fires on correct
;;;; configurations trains people to ignore it. See `check-backend-version''s
;;;; docstring.
;;;;
;;;; A CI version matrix (tracked separately) is expected to run the functional suite
;;;; against more than one released version and move part of each INFERRED span into
;;;; the VERIFIED one -- `version-range' keeps both as independent (LOW . HIGH) pairs
;;;; for exactly that: updating VERIFIED-LOW in place, without touching INFERRED-LOW or
;;;; restructuring anything else here.

(uiop:define-package #:cl-gbdt/src/version
  (:use #:cl)
  (:import-from #:cl-gbdt/src/conditions
                #:untested-backend-version)
  (:export #:version-compare
           #:version-in-range-p
           #:version-range
           #:make-version-range
           #:version-range-verified-low
           #:version-range-verified-high
           #:version-range-verified-evidence
           #:version-range-inferred-low
           #:version-range-inferred-high
           #:version-range-inferred-evidence
           #:version-range-tested-description
           #:check-backend-version
           #:*lightgbm-version-range*
           #:*xgboost-version-range*))

(in-package #:cl-gbdt/src/version)

;;; ---------------------------------------------------------------------------
;;; Pure version comparison

(defun %parse-version (version)
  "Parse VERSION into a list of three non-negative integers (MAJOR MINOR PATCH), or
NIL when VERSION is not a string of exactly that shape.

NIL itself is rejected immediately by `stringp'. So is anything with other than three
dot-separated components, or a component that is not entirely digits -- an empty
component (\"1..3\"), a sign (\"1.-2.3\"), or leading/trailing whitespace all fail the
`every #'digit-char-p' check below rather than being tolerated the way
`parse-integer' alone would be."
  (when (stringp version)
    (let ((components (uiop:split-string version :separator ".")))
      (when (and (= 3 (length components))
                 (every (lambda (component)
                          (and (plusp (length component))
                               (every #'digit-char-p component)))
                        components))
        (mapcar #'parse-integer components)))))

(defun version-compare (a b)
  "Compare two \"MAJOR.MINOR.PATCH\" version strings A and B component by component.

Returns `:less', `:equal' or `:greater' when both parse. Returns NIL when either does
not -- including either being NIL itself. NIL deliberately does not sort as the
smallest possible version: something this project cannot even parse is not
comparable at all, not merely low."
  (let ((pa (%parse-version a))
        (pb (%parse-version b)))
    (when (and pa pb)
      (loop :for x :in pa
            :for y :in pb
            :when (/= x y) :return (if (< x y) :less :greater)
            :finally (return :equal)))))

(defun version-in-range-p (version low high)
  "Return true when VERSION falls within [LOW, HIGH], inclusive, all three
\"MAJOR.MINOR.PATCH\" strings compared with `version-compare'.

Returns NIL -- \"cannot confirm this is in range\", not \"confirmed out of range\" --
when VERSION, LOW or HIGH fails to parse. VERSION = NIL needs no special case:
`version-compare' already returns NIL for it, and `member' on a NIL first argument
returns NIL in turn."
  (and (member (version-compare version low) '(:equal :greater))
       (member (version-compare version high) '(:equal :less))
       t))

;;; ---------------------------------------------------------------------------
;;; Recorded ranges

(defstruct version-range
  "What is known about a backend's compatible library versions, split into the
VERIFIED and INFERRED claims this file's header comment distinguishes.

VERIFIED-LOW and VERIFIED-HIGH bound the versions the functional suite has actually
trained and predicted against; today a single point (VERIFIED-LOW = VERIFIED-HIGH)
for both backends. VERIFIED-EVIDENCE names what backs it, for the warning's report.

INFERRED-LOW and INFERRED-HIGH bound the wider span `tools/check-upstream.lisp'
argues is ABI-compatible from C header comparison alone. INFERRED-EVIDENCE names
that argument. VERIFIED-LOW/HIGH always fall within [INFERRED-LOW, INFERRED-HIGH]."
  verified-low
  verified-high
  verified-evidence
  inferred-low
  inferred-high
  inferred-evidence)

(defparameter *lightgbm-version-range*
  (make-version-range
   :verified-low "4.0.0" :verified-high "4.7.0"
   :verified-evidence (concatenate
                        'string
                        "the functional suite passed against both endpoints, measured 2026-08 "
                        "when it had 106 assertions, "
                        "task 4's local version matrix")
   :inferred-low "3.0.0" :inferred-high "4.7.0"
   :inferred-evidence (concatenate
                        'string
                        "the 54 imported functions' declarations (28 of them "
                        "LightGBM's) are unchanged across this range, per "
                        "tools/check-upstream.lisp"))
  "LightGBM's recorded compatible-version range.

Never compared against a loaded version at runtime, unlike *XGBOOST-VERSION-RANGE* --
LightGBM's C API has no version entry point at all (`grep -c Version
src/lightgbm/c-api.lisp' reports 0), so `backend-version' is always NIL on this
backend and there is nothing to compare it against. See
`cl-gbdt/src/lightgbm/classes''s `initialize-backend' and `check-backend-version'
below. Recorded here anyway, for the same documentation purpose
`docs/user-guide/backend-differences.md''s table serves.

VERIFIED-LOW moved from \"4.7.0\" to \"4.0.0\" once task 4 actually ran the functional
suite against it -- confirmed clean, same 106 assertions as the pinned \"4.7.0\".

INFERRED-LOW is pinned at \"3.0.0\" and can never move to VERIFIED: `lightgbm==3.0.0'
predates aarch64 wheels on PyPI -- confirmed directly by task 4 (`pip download
lightgbm==3.0.0 --only-binary=:all:' finds no candidate at all on this platform) -- so
the CI version matrix this range's header comment describes cannot install it to
actually test against. This lower bound stays inferred permanently, not just until the
next matrix run.")

(defparameter *xgboost-version-range*
  (make-version-range
   :verified-low "2.0.0" :verified-high "3.4.1"
   :verified-evidence (concatenate
                        'string
                        "the functional suite passed against both endpoints, measured 2026-08 "
                        "when it had 106 assertions, "
                        "task 4's local version matrix")
   :inferred-low "2.0.0" :inferred-high "3.4.1"
   :inferred-evidence (concatenate
                        'string
                        "the 54 imported functions' declarations (26 of them "
                        "XGBoost's) are unchanged all the way back to 1.7.0, per "
                        "tools/check-upstream.lisp -- but task 4 measured 1.7.0 itself "
                        "and it failed (see below), so the claimed range does not "
                        "reach that low despite the header comparison alone allowing it"))
  "XGBoost's recorded compatible-version range -- see *LIGHTGBM-VERSION-RANGE*'s
docstring for what VERIFIED and INFERRED each mean and this file's header comment for
why the runtime check gates on INFERRED, not VERIFIED. Unlike LightGBM, this range is
actually compared against a loaded version: XGBoost's C API does expose one, read by
`cl-gbdt/src/xgboost/native''s `%read-version'.

Both bounds moved up from task 3's \"1.7.0\": task 4 ran the functional suite against
`xgboost==1.7.0' and its ranking round trip failed --
`xgboost-api-ranking-round-trip-respects-group-boundaries' in
tests/functional/xgboost-api.lisp requires predictions to increase strictly within each
query group, and 1.7.0 instead produced a tie between the first two rows of each group
where 3.3.0 keeps all four strictly increasing. Every other assertion (105 of 106,
including the plain classification and multiclass round trips, feature-importance,
save/load, and every close-backend guard) passed unchanged at 1.7.0 -- this is a real
function returning different numbers, not a symptom of a missing symbol or a crash, so
`probe-foreign-symbols' and `tools/check-upstream.lisp''s header comparison could never
have caught it. `xgboost==2.0.0' was tried next and passed everything the then-pinned
\"3.3.0\" does, so INFERRED-LOW moved up to meet VERIFIED-LOW at \"2.0.0\" rather than
leave the disproven \"1.7.0\" claim in place under a wider, ABI-only label. Nothing
between 1.7.0 and 2.0.0 was tested, so this range makes no claim about it either.

Both bounds moved again, from \"3.3.0\" to \"3.4.1\": 3.4.1 removes
`XGDMatrixCreateFromFile' from XGBoost's C API (see `ffi-spec/ABI-BLACKLIST.md''s moot
table), but that function was already blacklisted and never imported by this project,
and every declaration cl-gbdt does import is unchanged -- confirmed by
`tools/check-upstream.lisp' before the pin moved. The whole functional suite passed
against 3.4.1 on this machine, linux-aarch64 -- the same kind of evidence, a single
local run, that backed the \"3.3.0\" point it replaces, whose own VERIFIED-EVIDENCE
also cites task 4's local version matrix, so this move does not weaken the standard.
Going forward, CI's `test' job (`test.yml') is configured to run that same suite
against 3.4.1 on all three of its platforms -- linux-x86_64, linux-aarch64, and
macos-aarch64 -- on pushes to master and on every pull request, extending this
evidence beyond the one platform measured here. 3.4.0 itself was not tested, the
same way the gap between 1.7.0 and
2.0.0 above was not: this range's endpoints are what was actually measured, not a
claim about every version between them.")

(defun version-range-tested-description (range)
  "Return RANGE's evidence as a list of two strings -- the verified point/range, then
the wider inferred one -- for `untested-backend-version''s :TESTED initarg.

Kept as two separate entries rather than one blended string, so the condition's
report -- \"Tested: ~{~A~^, ~}\" -- states each claim's own strength instead of
implying both are the same kind of evidence."
  (list (format nil "~A~@[-~A~] verified (~A)"
                (version-range-verified-low range)
                (let ((high (version-range-verified-high range)))
                  (unless (string= high (version-range-verified-low range)) high))
                (version-range-verified-evidence range))
        (format nil "~A-~A inferred (~A)"
                (version-range-inferred-low range) (version-range-inferred-high range)
                (version-range-inferred-evidence range))))

;;; ---------------------------------------------------------------------------
;;; The warning

(defun check-backend-version (backend-name version range)
  "Signal `untested-backend-version' -- a WARNING, not an ERROR; see its own docstring
for why -- when VERSION, BACKEND-NAME's loaded library version, falls outside RANGE's
*inferred* bounds, or does not parse as a \"MAJOR.MINOR.PATCH\" string at all, VERSION
= NIL included. Does nothing when VERSION is confirmed within range.

Gates on RANGE's inferred bounds rather than its narrower verified ones -- see this
file's header comment: warning on every difference from the exact version the
functional suite happens to run against would fire on nearly every compatible caller.

Only `cl-gbdt/src/xgboost/classes''s `initialize-backend' calls this. It is never
called for LightGBM: with `backend-version' always NIL there, this would signal on
every single open, a check that can never actually confirm compatibility -- see
*LIGHTGBM-VERSION-RANGE*'s docstring for the fuller explanation of that asymmetry."
  (unless (version-in-range-p version
                               (version-range-inferred-low range)
                               (version-range-inferred-high range))
    (warn 'untested-backend-version
          :backend backend-name
          :version version
          :tested (version-range-tested-description range))))
