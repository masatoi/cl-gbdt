;;;; check-api-reference.lisp --- docs/API-REFERENCE.md matches the docstrings it was
;;;; generated from; a docstring edited without regenerating fails the build.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-api-reference.lisp
;;;;
;;;; docs/API-REFERENCE.md is GENERATED, by src/docgen/ (development only, never part of a
;;;; build) through tools/gen-api-reference.lisp, from the docstrings `cl-gbdt',
;;;; `cl-gbdt/lightgbm' and `cl-gbdt/xgboost' export -- see that file's own header. Nothing
;;;; stops a docstring from being edited without the regeneration step that follows it; this
;;;; script is what turns that omission into a build failure instead of a reference that
;;;; quietly lies about what a symbol does.
;;;;
;;;; Unlike tools/ci/check-binding-coverage.lisp, which this script is modelled on for its
;;;; reporting style and its floor constants, nothing here is read as inert data: the whole
;;;; point is to run the real emitter over a real loaded image and compare what it produces
;;;; against what is committed, so both backends' unified systems are loaded for real, the
;;;; same three `asdf:load-system' calls tools/gen-api-reference.lisp itself makes. This is
;;;; also why the check cannot be a rove test: `cl-gbdt/tests' deliberately loads neither
;;;; backend, so a suite run through it never has both `cl-gbdt/lightgbm' and `cl-gbdt/xgboost'
;;;; loaded in the same image at once, which is exactly what COLLECT-ENTRIES needs to see the
;;;; whole published surface.
;;;;
;;;; WHAT THIS CHECKS, in order, each stage dying immediately rather than let a later stage
;;;; run against a state the earlier one already showed is broken:
;;;;
;;;; 1. PRIMITIVES -- CHECK-INTROSPECTION-PRIMITIVES, called with no arguments against its own
;;;;    default +REQUIRED-PRIMITIVES+ (src/docgen/introspect.lisp): three non-ANSI SBCL
;;;;    internals `documentation' and the MOP cannot reach on their own. Every later stage
;;;;    reaches all three, deep inside WRITE-API-REFERENCE or COLLECT-ENTRIES; checking them
;;;;    here by name first is what turns a future SBCL dropping one of them into a message
;;;;    naming the missing primitive, rather than an undefined-function error somewhere inside
;;;;    SLOT-DOCUMENTATION or TYPE-SLOTS that names neither the primitive nor why the emitter
;;;;    needed it -- see +REQUIRED-PRIMITIVES+'s own docstring for the 62 slot bodies at stake.
;;;; 2. BYTE-FOR-BYTE -- WRITE-API-REFERENCE regenerates the reference into a fresh temporary
;;;;    file (UIOP:WITH-TEMPORARY-FILE; never over the committed file itself, so a failing run
;;;;    leaves the working tree exactly as `git status' found it), and FIRST-DIFFERENCE walks
;;;;    both strings once via UIOP:READ-FILE-STRING. A pass/fail verdict alone is not enough
;;;;    for a 307 KB, 6358-line file: the failure message below names the first differing line
;;;;    and byte offset, and the regeneration command, so a reader can jump straight to the
;;;;    disagreement instead of reading a diff of the whole document.
;;;; 3. DOCUMENTATION FLOOR -- over COLLECT-ENTRIES: every entry has its own docstring or
;;;;    points at a documented type (a reader, accessor or structure constructor legitimately
;;;;    has neither of its own; RENDER-ENTRY prints a one-line pointer at its type instead),
;;;;    and every slot a published reader exposes on a :CLASS or :CONDITION type carries its
;;;;    own :DOCUMENTATION. :STRUCTURE slots are exempt from the second half: SBCL's
;;;;    `defstruct' has no per-slot :documentation at all, and src/docgen/introspect.lisp's
;;;;    TYPE-SLOTS hardcodes NIL for every structure slot for exactly that reason -- a floor
;;;;    this check cannot lower without lying about what SBCL can express. Task 6 filled the
;;;;    twelve condition-slot docstrings the class/condition half of this floor is named
;;;;    after; this stage exists to keep them filled.
;;;; 4. SURFACE FLOOR -- per-package published-symbol counts against
;;;;    +MINIMUM-PUBLISHED-SYMBOLS+, re-measured against today's COLLECT-ENTRIES rather than
;;;;    copied from any plan or brief. This stage exists for exactly what stage 2 cannot catch:
;;;;    drop an export and regenerate honestly, and the committed and generated files agree
;;;;    with each other perfectly -- they just both agree on a smaller surface. Byte-for-byte
;;;;    equality says nothing about that; only counting the surface itself does.
;;;;
;;;; WHAT THIS CANNOT CATCH
;;;;
;;;;   - Whether a docstring's *content* is accurate or well-written -- stage 3 checks
;;;;     presence, not prose, the same limit check-binding-coverage.lisp's own header states
;;;;     for BINDING-COVERAGE.md's Notes and exclusion reasons.
;;;;   - A symbol removed from a package's :export clause and never referenced again by
;;;;     anything that still passes stage 4's *current* floor -- the floor only fails when the
;;;;     count drops BELOW it; two exports could vanish and one arrive elsewhere in the same
;;;;     regeneration without moving any package under its own floor.
;;;;   - Whether docs/API-REFERENCE.md was regenerated on the same SBCL version, or the same
;;;;     platform, that produced the committed copy -- stage 2 only compares text.

(require :asdf)

(asdf:load-system "cl-gbdt/lightgbm/unified")
(asdf:load-system "cl-gbdt/xgboost/unified")
(asdf:load-system "cl-gbdt/docgen")

(defparameter +reference-path+ "docs/API-REFERENCE.md"
  "Path, relative to the repository root, of the generated reference this check enforces.")

(defparameter +regenerate-command+
  "ros run -- --non-interactive --load tools/gen-api-reference.lisp"
  "The command a stage 2 failure message tells the reader to run.")

(defparameter +minimum-published-symbols+
  '(("cl-gbdt" . 146) ("cl-gbdt/lightgbm" . 93) ("cl-gbdt/xgboost" . 94))
  "Per-package floor on published symbol count, re-measured against a live COLLECT-ENTRIES
run on 2026-08-13 to match docs/API-REFERENCE.md's own per-package index counts as committed
that day -- not copied from any earlier plan or brief. See this file's header, item 4, for why
stage 2's byte-for-byte check cannot catch a floor violation on its own: a dropped export,
regenerated honestly, leaves the committed and generated files in perfect agreement about a
smaller surface.

Raised from 141/88/89 by file-input-layer-1's Task 6: `create-dataset-from-file' joins both
backend packages, and `file-format-mismatch' and its three readers join all three public
packages through `cl-gbdt/src/conditions'.

`cl-gbdt' raised from 145 to 146 by thread-safety-contract's Task 1: `with-backend' joins
`cl-gbdt' alone, through `src/all.lisp''s re-export of `cl-gbdt/src/protocol' -- neither
backend package publishes it, so `cl-gbdt/lightgbm' and `cl-gbdt/xgboost' stay at 93/94.")

(defun die (format-control &rest arguments)
  "Print FORMAT-CONTROL/ARGUMENTS to *ERROR-OUTPUT* as a FAIL line and exit with status 1.

Every stage below calls this exactly once, after it has already printed every individual
offender of its own via a separate FORMAT to *ERROR-OUTPUT* -- so this call's own line is
always a summary of what came before it, never the only line a failure prints. Named after,
and shaped like, tools/regen.lisp's own `die'."
  (format *error-output* "~&FAIL ~?~%" format-control arguments)
  (uiop:quit 1))

(defun first-difference (expected actual)
  "Return (LINE COLUMN OFFSET) of the first byte where ACTUAL leaves EXPECTED, or NIL."
  (loop with line = 1 with column = 1
        for offset from 0
        while (and (< offset (length expected)) (< offset (length actual)))
        for a = (char expected offset)
        for b = (char actual offset)
        do (unless (char= a b) (return (list line column offset)))
           (if (char= a #\Newline)
               (setf line (1+ line) column 1)
               (incf column))
        finally (return (unless (= (length expected) (length actual))
                          (list line column offset)))))

;;; ---- Stage 1: the introspection primitives exist ----

(defun run-stage-1-primitives ()
  "Stage 1: every introspection primitive the emitter needs is present in this SBCL.

Dies immediately on failure, naming the missing primitive or primitives -- see this file's
header, item 1, for why that message has to come from here rather than from wherever inside
WRITE-API-REFERENCE or COLLECT-ENTRIES the primitive would otherwise be missed."
  (handler-case
      (cl-gbdt/src/docgen/introspect:check-introspection-primitives)
    (error (condition)
      (die "stage 1 (primitives): ~A" condition)))
  (format t "~&stage 1 (primitives): ~D required primitive~:P present~%"
          (length cl-gbdt/src/docgen/introspect:+required-primitives+)))

;;; ---- Stage 2: byte-for-byte ----

(defun run-stage-2-byte-for-byte (root)
  "Stage 2: docs/API-REFERENCE.md is byte-for-byte what WRITE-API-REFERENCE emits today.

Regenerates into a fresh UIOP:WITH-TEMPORARY-FILE, never over the committed file itself, and
compares with UIOP:READ-FILE-STRING. On a mismatch, dies naming the first differing line and
byte offset (via FIRST-DIFFERENCE) and the command to regenerate -- see this file's header,
item 2."
  (let* ((committed-path (merge-pathnames +reference-path+ root))
         (committed (uiop:read-file-string committed-path)))
    (uiop:with-temporary-file (:pathname temp-path :type "md")
      (cl-gbdt/src/docgen/emit:write-api-reference
       cl-gbdt/src/docgen/emit:+public-packages+ temp-path)
      (let* ((generated (uiop:read-file-string temp-path))
             (difference (first-difference committed generated)))
        (when difference
          (destructuring-bind (line column offset) difference
            (die "stage 2 (byte-for-byte): ~A differs from a fresh regeneration, first at ~
                  line ~D, column ~D (byte offset ~D). Regenerate with `~A' and commit the ~
                  result."
                 +reference-path+ line column offset +regenerate-command+)))
        (format t "~&stage 2 (byte-for-byte): ~A matches a fresh regeneration (~D bytes)~%"
                +reference-path+ (length committed))))))

;;; ---- Stage 3: the documentation floor ----

(defun documented-slot-p (slot)
  "True when SLOT (a `cl-gbdt/src/docgen/introspect:slot-info') carries its own
:documentation."
  (and (cl-gbdt/src/docgen/introspect:slot-info-documentation slot) t))

(defun run-stage-3-documentation-floor (entries)
  "Stage 3: every published entry documents itself or points at a documented type, and every
slot a published reader exposes on a :CLASS or :CONDITION type carries its own
:documentation. ENTRIES is a COLLECT-ENTRIES result. Lists every offender, then dies naming
the count -- see this file's header, item 3, including why :STRUCTURE slots are exempt.

The slot half is keyed on the TYPE entries' own ENTRY-SLOTS, not on reader entries whose
ENTRY-POINTS-AT happens to be set: COLLECT-ENTRIES only sets POINTS-AT when a reader has no
docstring of its own (see COLLECT-ENTRIES's OWN-DOCUMENTATION binding), so a reader entry
that acquires its own docstring would silently walk its slot out of this floor entirely if
the floor were keyed on POINTS-AT instead -- the type's own Slots section would go on
rendering that slot with no fence, and nothing here would notice. Walking every :CLASS and
:CONDITION type's own ENTRY-SLOTS instead holds the floor for the property the spec actually
states, independent of whether any given reader happens to carry its own text. ENTRY-SLOTS
already carries only published readers -- COLLECT-ENTRIES filters them via its own internal
ENTRY-SLOTS-OF -- so a slot with no published reader at all contributes nothing here either
way."
  (let ((undocumented '())
        (undocumented-slots '()))
    (dolist (entry entries)
      (unless (or (cl-gbdt/src/docgen/render:entry-documentation entry)
                  (cl-gbdt/src/docgen/render:entry-points-at entry))
        (push entry undocumented)))
    (dolist (entry entries)
      (when (member (cl-gbdt/src/docgen/render:entry-kind entry) '(:class :condition))
        (dolist (slot (cl-gbdt/src/docgen/render:entry-slots entry))
          (when (and (cl-gbdt/src/docgen/introspect:slot-info-readers slot)
                     (not (documented-slot-p slot)))
            (push (cons entry slot) undocumented-slots)))))
    (setf undocumented (nreverse undocumented))
    (setf undocumented-slots (nreverse undocumented-slots))
    (dolist (entry undocumented)
      (format *error-output*
              "~&FAIL stage 3 (documentation floor): ~A:~(~A~) has no docstring of its own ~
               and points at no type.~%"
              (cl-gbdt/src/docgen/render:entry-qualifier entry)
              (symbol-name (cl-gbdt/src/docgen/render:entry-symbol entry))))
    (dolist (item undocumented-slots)
      (destructuring-bind (entry . slot) item
        (format *error-output*
                "~&FAIL stage 3 (documentation floor): ~A:~(~A~)'s `~(~A~)' slot has a ~
                 published reader but no :documentation.~%"
                (cl-gbdt/src/docgen/render:entry-qualifier entry)
                (symbol-name (cl-gbdt/src/docgen/render:entry-symbol entry))
                (symbol-name (cl-gbdt/src/docgen/introspect:slot-info-name slot)))))
    (let ((total (+ (length undocumented) (length undocumented-slots))))
      (when (plusp total)
        (die "stage 3 (documentation floor): ~D offender~:P -- see FAIL lines above." total))
      (format t "~&stage 3 (documentation floor): ~D entr~:@P, all documented or pointing at ~
                 a documented type~%"
              (length entries)))))

;;; ---- Stage 4: the surface floor ----

(defun package-published-count (package-name entries)
  "Count of ENTRIES (a COLLECT-ENTRIES result) exported from PACKAGE-NAME -- the same count
EMIT-INDEX prints as that package's `### `PACKAGE-NAME` -- N symbols' heading."
  (count-if (lambda (entry)
              (member package-name (cl-gbdt/src/docgen/render:entry-exported-from entry)
                      :test #'string=))
            entries))

(defun run-stage-4-surface-floor (entries)
  "Stage 4: every public package still publishes at least its +MINIMUM-PUBLISHED-SYMBOLS+
floor's worth of symbols. ENTRIES is a COLLECT-ENTRIES result. Prints every package's live
count regardless of outcome, then dies naming the packages under their floor -- see this
file's header, item 4, for why stage 2's byte-for-byte check cannot catch this on its own."
  (let ((offenders '()))
    (dolist (spec +minimum-published-symbols+)
      (destructuring-bind (package-name . floor) spec
        (let ((count (package-published-count package-name entries)))
          (format t "~&stage 4 (surface floor): ~A publishes ~D symbol~:P (floor ~D)~%"
                  package-name count floor)
          (when (< count floor)
            (push (list package-name count floor) offenders)))))
    (setf offenders (nreverse offenders))
    (dolist (offender offenders)
      (destructuring-bind (package-name count floor) offender
        (format *error-output*
                "~&FAIL stage 4 (surface floor): ~A publishes ~D symbols, below the floor of ~
                 ~D in +MINIMUM-PUBLISHED-SYMBOLS+.~%"
                package-name count floor)))
    (when offenders
      (die "stage 4 (surface floor): ~D package~:P below its floor -- an export may have ~
            been dropped and docs/API-REFERENCE.md regenerated honestly, which stage 2 ~
            cannot see (the committed and generated files would still agree, just on a ~
            smaller surface). Restore the export, or lower +MINIMUM-PUBLISHED-SYMBOLS+ ~
            deliberately."
           (length offenders)))))

;;; ---- Main check ----

(let ((root (uiop:getcwd)))
  (run-stage-1-primitives)
  (run-stage-2-byte-for-byte root)
  (let ((entries (cl-gbdt/src/docgen/emit:collect-entries
                  cl-gbdt/src/docgen/emit:+public-packages+)))
    (run-stage-3-documentation-floor entries)
    (run-stage-4-surface-floor entries)))

(format t "~&check-api-reference: all four stages passed~%")
(uiop:quit 0)
