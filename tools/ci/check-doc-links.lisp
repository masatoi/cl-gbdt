;;;; check-doc-links.lisp --- every relative Markdown link in the tracked docs resolves.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-doc-links.lisp
;;;;
;;;; README.markdown drifted to 239 KB carrying stale counts and a link into a file the
;;;; repository does not track, because nothing in CI ever read it. The split that followed
;;;; multiplied the links that can rot -- a handful became dozens once the README's own
;;;; cross-references and six new `docs/user-guide/*.md' guides started linking to each
;;;; other. This is the first check either kind of drift gets.
;;;;
;;;; WHAT IS SCANNED. The set the split's own throwaway resolvers (Tasks 2 and 3 of the
;;;; readme-split programme) already proved clean, not just the three files a narrower reading
;;;; of the brief would suggest: `README.markdown', `CONTRIBUTING.md', `CLAUDE.md',
;;;; `THIRD-PARTY-LICENSES.md', both `ffi-spec/*.md' files, the policy document, and all six
;;;; `docs/user-guide/*.md' guides -- see +SCANNED-FILES+.
;;;;
;;;; WHAT COUNTS AS A LINK. Markdown inline links, `[text](target)', matched across the whole
;;;; file rather than line by line -- a line-based first attempt earlier in this same
;;;; programme (Task 2) missed 23 of 61 links because this project's prose hard-wraps at
;;;; roughly 90 columns, routinely splitting `[text]' from `(target)' or a target itself
;;;; across a newline. The regex below (+LINK-SCANNER+) uses negated character classes for
;;;; both halves, which match a literal newline in cl-ppcre by default, so no DOTALL flag or
;;;; explicit line-joining step is needed. Fenced code blocks (``` `...` ``` ```) are blanked
;;;; out first (BLANK-FENCED-CODE-BLOCKS), line count preserved, so a pasted `#' reader macro
;;;; or an example `[...]/(...)' inside one is never mistaken for a heading or a link.
;;;;
;;;; EVERY OCCURRENCE IS ITS OWN CHECK, deliberately not deduplicated by target or by (text .
;;;; target) pair, even within one file. The same link written twice in different prose is
;;;; two independent claims that it resolves, and collapsing them would silently stop checking
;;;; one of the two -- which is exactly the defect class Task 2's line-based bug already put
;;;; this project through once. Concretely: `docs/user-guide/data-and-prediction.md' says "the
;;;; capability model" twice, hard-wrapped identically both times (once for `:missing', once
;;;; for `:categorical-features'), pointing at the same anchor; deduplicating would leave one
;;;; of those two prose sites unchecked for no reason a reader could see.
;;;;
;;;; RESOLUTION CONVENTION: a relative target's file part is resolved against the LINKING
;;;; FILE'S OWN DIRECTORY, not the repository root. This is the only convention under which
;;;; the guides' own sibling links -- `docs/user-guide/backends.md' writes plain
;;;; `backend-differences.md', not `docs/user-guide/backend-differences.md' -- resolve at all;
;;;; root-relative resolution would send every one of those into a file that does not exist at
;;;; the repository root. `README.markdown' and `CONTRIBUTING.md' sit at the repository root,
;;;; so their own links are root-relative and file-relative at once, and cannot distinguish
;;;; the two conventions -- the guides are what settle it. See RESOLVE-TARGET-FILE.
;;;;
;;;; ANCHOR DERIVATION -- GitHub's rule, VERIFIED AGAINST A REAL RENDERED PAGE rather than
;;;; trusted as written. As commonly stated (and as this task's own brief states it): lower
;;;; case the heading, drop every character that is not alphanumeric, space or hyphen, turn
;;;; each space into a hyphen. That statement is WRONG on one point: GitHub also KEEPS
;;;; underscores. Proof, not assertion: `docs/user-guide/data-and-prediction.md' carries the
;;;; heading "#### LightGBM: `categorical_feature` and its four aliases", word-for-word
;;;; identical to a heading already live in `README.markdown' on `master' (before this split),
;;;; and GitHub's own rendering metadata for that exact page states its computed anchor as
;;;; `lightgbm-categorical_feature-and-its-four-aliases' -- underscore intact. Dropping the
;;;; underscore, as the rule-as-commonly-stated would, derives
;;;; `lightgbm-categoricalfeature-and-its-four-aliases' instead and breaks a real, currently
;;;; working link. SLUGIFY therefore keeps `#\_' alongside alphanumerics, space and hyphen.
;;;; The colon-and-backtick case the brief specifically asks to check (a heading such as
;;;; "Stopping early: `:early-stopping`") WAS verified the same way against the same rendered
;;;; page and needs no correction: GitHub's own anchor for it is
;;;; `stopping-early-early-stopping', exactly what the stated rule (colon and backtick both
;;;; simply dropped, not replaced) already predicts.
;;;;
;;;; Duplicate slugs within one file are suffixed `-1', `-2', ... in order of appearance,
;;;; matching GitHub's own de-duplication (FILE-HEADING-SLUGS) -- untested by anything in the
;;;; current tree, since no two headings collide today, but a cheap correctness property to
;;;; hold anyway rather than silently mis-check the day one does.
;;;;
;;;; EXCLUDED: `docs/API-REFERENCE.md'. It is GENERATED (see
;;;; tools/ci/check-api-reference.lisp), and its 332 internal anchors are explicit `<a
;;;; id="...">' tags emitted by src/docgen/, not headings run through GitHub's slug rule at
;;;; all -- `## \`cl-gbdt:backend-supports-p\`' slugs to `cl-gbdtbackend-supports-p' under the
;;;; real rule (the colon is simply dropped, no hyphen takes its place, since there is no
;;;; space on either side of it), while the file's own index links say
;;;; `#cl-gbdt-backend-supports-p'. Checking those against SLUGIFY would report 332 failures
;;;; against a file no human hand-edits and the generator does not intend to satisfy this
;;;; rule. This checker still confirms the file itself EXISTS when something links to it
;;;; (`docs/user-guide/file-input.md' does, with no fragment); it only skips the fragment
;;;; check when the resolved target is this file. Do not remove this exclusion without fixing
;;;; the emitter to produce slug-following anchors instead -- see +EXCLUDED-FRAGMENT-TARGETS+.
;;;;
;;;; A NOTE ON COUNTS, for whoever reads this next: THIS COMMENT NAMES NONE, deliberately.
;;;;
;;;; An earlier, uncommitted, throwaway resolver run by hand during this same split (Tasks 2
;;;; and 3 of the readme-split programme) reported a materially LOWER "links checked" figure
;;;; than this script does on the same tree. That gap was never a disagreement about what
;;;; resolves -- both reported zero failures -- but the line-based bug this header describes
;;;; above: that script matched `[text](target)' within a single line, so every link this
;;;; project's ~90-column hard wrap split across a newline went uncounted. This script's
;;;; figure is the correct one; the throwaway script's was an undercount.
;;;;
;;;; Two successive drafts of this comment stated a count and were wrong: the first claimed a
;;;; pre-fix broken total that disagreed with what the script printed, explaining the gap by a
;;;; duplicate already inside it; the second fixed that by measurement and then named a
;;;; "links checked" total that the very commit correcting it changed. Both were prose
;;;; outrunning code, in the header of the check written to stop prose outrunning code -- and
;;;; the second shows why: this number moves whenever any scanned document gains or loses a
;;;; link, which is most commits that touch documentation, so a count written here is stale
;;;; almost immediately. THE SCRIPT PRINTS ITS OWN COUNTS ON EVERY RUN. Read them there, and
;;;; do not copy them back into this comment.
;;;;
;;;; WHAT THIS CANNOT CATCH
;;;;
;;;;   - A link that is syntactically correct and resolves, but points at the wrong place --
;;;;     this only checks that a target exists and, if fragmented, that some heading derives
;;;;     to the fragment, never that the destination is the one the prose actually means.
;;;;   - Reference-style links (`[text][ref]' plus a `[ref]: target' definition elsewhere) --
;;;;     none exist anywhere in +SCANNED-FILES+ today (checked by hand before writing this),
;;;;     so support was not built; a reference-style link added later passes through
;;;;     +LINK-SCANNER+ unmatched rather than being checked or flagged.
;;;;   - A target inside a fenced code block that was meant to be a real, checked link --
;;;;     BLANK-FENCED-CODE-BLOCKS treats the whole block as inert by design.

(require :asdf)

(ql:quickload '("cl-ppcre") :silent t)

(defparameter +scanned-files+
  '("README.markdown" "CONTRIBUTING.md" "CLAUDE.md" "THIRD-PARTY-LICENSES.md"
    "ffi-spec/ABI-BLACKLIST.md" "ffi-spec/BINDING-COVERAGE.md"
    "docs/cl-gbdt-layered-api-implementation-policy.md"
    "docs/user-guide/backend-differences.md" "docs/user-guide/backends.md"
    "docs/user-guide/custom-training.md" "docs/user-guide/data-and-prediction.md"
    "docs/user-guide/file-input.md" "docs/user-guide/training.md")
  "Repository-root-relative paths this check scans FOR links. Every one of these is a source
of links, not just a possible target -- a target that is not in this list (docs/FUNCTIONAL-
COVERAGE.md, LICENSE, docs/API-REFERENCE.md, ...) is still resolved and, unless excluded
below, anchor-checked; it is simply never scanned for outgoing links of its own. See this
file's header for why this set is broader than a first reading of the brief suggests.")

(defparameter +excluded-fragment-targets+
  '("docs/API-REFERENCE.md")
  "Resolved targets (repository-root-relative, as RESOLVE-TARGET-FILE produces) whose
fragment is never checked against SLUGIFY, even though the file's own existence still is.
See this file's header for why docs/API-REFERENCE.md is the one entry: its anchors are
generated <a id=...> tags, not slugged headings.")

(defparameter +external-prefixes+
  '("http:" "https:" "mailto:")
  "Target prefixes (case-insensitive) this check treats as leaving the repository and does
not resolve or count -- neither as checked nor as failed.")

(defparameter +link-scanner+
  (cl-ppcre:create-scanner "\\[([^\\[\\]]*)\\]\\(([^()]*)\\)")
  "Matches one Markdown inline link, `[text](target)'. Both groups use a negated character
class rather than `.', which is what lets a match span a hard-wrapped newline with no extra
flag -- see this file's header.")

(defparameter +heading-scanner+
  (cl-ppcre:create-scanner "^(#{1,6})[ \\t]+(.*?)[ \\t]*#*[ \\t]*$")
  "Matches one ATX heading line (`#' through `######'), capturing the hashes and the text
with trailing whitespace and closing hashes stripped.")

(defparameter +fence-marker-scanner+
  (cl-ppcre:create-scanner "^ {0,3}```")
  "Matches a fenced-code-block delimiter line, up to three leading spaces allowed per
CommonMark. Toggled on each match by BLANK-FENCED-CODE-BLOCKS; the specific fence length
(three or more backticks) is not distinguished, since nothing in +SCANNED-FILES+ nests one
fence inside another of a different length.")

(defun die (format-control &rest arguments)
  "Print FORMAT-CONTROL/ARGUMENTS to *ERROR-OUTPUT* as a FAIL line and exit with status 1.

Every offending link is already printed as its own FAIL line by the caller before this runs,
so this call's own line is always a summary, never the only line a failure prints."
  (format *error-output* "~&FAIL ~?~%" format-control arguments)
  (uiop:quit 1))

(defun blank-fenced-code-blocks (content)
  "Return CONTENT with every fenced code block's interior, and its opening and closing fence
line, replaced by an empty line -- line count preserved, so line numbers computed against the
result stay accurate. A `(defun ...)' pasted inside a fence, or a stray `#' at a line's
start, would otherwise be misread as a link or a heading."
  (let ((in-fence nil))
    (format nil "~{~A~^~%~}"
            (mapcar (lambda (line)
                      (cond ((cl-ppcre:scan +fence-marker-scanner+ line)
                             (setf in-fence (not in-fence))
                             "")
                            (in-fence "")
                            (t line)))
                    (uiop:split-string content :separator '(#\Newline))))))

(defun heading-text (line)
  "Return LINE's heading text (hashes, surrounding whitespace and trailing hashes stripped)
if LINE is an ATX heading, or NIL if it is not one."
  (cl-ppcre:register-groups-bind (hashes text) (+heading-scanner+ line)
    (declare (ignore hashes))
    text))

(defun slugify (heading-text)
  "Return HEADING-TEXT's GitHub anchor slug: lower-case, every character that is not
alphanumeric, space, hyphen or underscore removed, then each remaining space turned into a
hyphen. See this file's header for why underscore is kept despite the commonly-stated rule
omitting it -- verified against a real GitHub-rendered anchor, not merely written down."
  (substitute #\- #\Space
              (remove-if-not (lambda (ch)
                                (or (alphanumericp ch) (char= ch #\Space)
                                    (char= ch #\-) (char= ch #\_)))
                              (string-downcase heading-text))))

(defun file-heading-slugs (path)
  "Return the list of every anchor slug PATH's own ATX headings derive to, in order of
appearance, duplicates suffixed `-1', `-2', ... matching GitHub's own de-duplication. PATH
must already be known to exist -- callers check TARGET-EXISTS-P first."
  (let ((counts (make-hash-table :test 'equal))
        (slugs '()))
    (dolist (line (uiop:split-string (blank-fenced-code-blocks (uiop:read-file-string path))
                                      :separator '(#\Newline)))
      (let ((text (heading-text line)))
        (when text
          (let* ((base (slugify text))
                 (seen (gethash base counts 0)))
            (setf (gethash base counts) (1+ seen))
            (push (if (zerop seen) base (format nil "~A-~D" base seen)) slugs)))))
    (nreverse slugs)))

(defun strip-whitespace (string)
  "Return STRING with every space, tab, newline and carriage return removed -- normalizes a
link target that was itself hard-wrapped across a newline back into one contiguous path or
fragment. None of +SCANNED-FILES+ uses a link title (`(target \"title\")'), the one
construct where an internal space would be meaningful, so this is safe here."
  (remove-if (lambda (ch) (member ch '(#\Space #\Tab #\Newline #\Return))) string))

(defun external-target-p (target)
  "True when TARGET (already whitespace-stripped) starts with one of +EXTERNAL-PREFIXES+,
case-insensitively."
  (some (lambda (prefix)
          (and (<= (length prefix) (length target))
               (string-equal prefix target :end2 (length prefix))))
        +external-prefixes+))

(defun path-directory (path-string)
  "Return the directory portion of PATH-STRING, a POSIX-style relative path, or the empty
string if PATH-STRING has no `/'."
  (let ((slash (position #\/ path-string :from-end t)))
    (if slash (subseq path-string 0 slash) "")))

(defun normalize-relative-path (path-string)
  "Collapse `.' and `..' components out of PATH-STRING, a POSIX-style relative path, without
touching the filesystem. A leading `..' with nothing left to pop, or a leading `/', is not
expected in +SCANNED-FILES+'s own links and is not specially handled."
  (let ((components '()))
    (dolist (part (uiop:split-string path-string :separator '(#\/)))
      (cond ((or (string= part "") (string= part ".")) nil)
            ((string= part "..") (pop components))
            (t (push part components))))
    (format nil "~{~A~^/~}" (nreverse components))))

(defun resolve-target-file (source-file file-part)
  "Resolve FILE-PART -- a link target's portion before any `#fragment', possibly empty for a
same-file link -- against SOURCE-FILE's own directory. See this file's header for why this,
not repository-root resolution, is the convention every link in +SCANNED-FILES+ actually
needs."
  (if (string= file-part "")
      source-file
      (let ((dir (path-directory source-file)))
        (normalize-relative-path
         (if (string= dir "") file-part (concatenate 'string dir "/" file-part))))))

(defun target-exists-p (path)
  "True when PATH names an existing file or directory, relative to the current directory
(the repository root, since this script is always run from there)."
  (or (uiop:file-exists-p path)
      (uiop:directory-exists-p (if (and (plusp (length path))
                                         (char= (char path (1- (length path))) #\/))
                                    path
                                    (concatenate 'string path "/")))))

(defun line-number-at (content position)
  "Return the 1-based line number of POSITION within CONTENT."
  (1+ (count #\Newline content :end position)))

(defun check-file-links (source-file)
  "Scan SOURCE-FILE for every relative Markdown link and check that each one resolves.
Returns (VALUES CHECKED FAILURES): CHECKED is the count of relative (non-external) links
found, and FAILURES a list of human-readable failure descriptions, in the order found."
  (let* ((content (uiop:read-file-string source-file))
         (blanked (blank-fenced-code-blocks content))
         (checked 0)
         (failures '()))
    (cl-ppcre:do-scans (match-start match-end reg-starts reg-ends +link-scanner+ blanked)
      (let* ((raw-text (subseq blanked (aref reg-starts 0) (aref reg-ends 0)))
             (raw-target (subseq blanked (aref reg-starts 1) (aref reg-ends 1)))
             (target (strip-whitespace raw-target)))
        (unless (or (zerop (length target)) (external-target-p target))
          (incf checked)
          (let* ((hash-pos (position #\# target))
                 (file-part (if hash-pos (subseq target 0 hash-pos) target))
                 (fragment (if hash-pos (subseq target (1+ hash-pos)) ""))
                 (resolved (resolve-target-file source-file file-part))
                 (line (line-number-at blanked match-start)))
            (cond
              ((not (target-exists-p resolved))
               (push (format nil "~A:~D: [~A](~A) -> ~A does not exist"
                              source-file line raw-text raw-target resolved)
                     failures))
              ((and (plusp (length fragment))
                    (not (member resolved +excluded-fragment-targets+ :test #'string=))
                    (not (member fragment (file-heading-slugs resolved) :test #'string=)))
               (push (format nil "~A:~D: [~A](~A) -> #~A does not derive from any heading ~
                                  in ~A"
                              source-file line raw-text raw-target fragment resolved)
                     failures)))))))
    (values checked (nreverse failures))))

;;; ---- Main check ----

(let ((total-checked 0)
      (all-failures '()))
  (dolist (source-file +scanned-files+)
    (multiple-value-bind (checked failures) (check-file-links source-file)
      (incf total-checked checked)
      (setf all-failures (append all-failures failures))))
  (dolist (failure all-failures)
    (format *error-output* "~&FAIL ~A~%" failure))
  (when all-failures
    (die "~D of ~D link~:P checked failed -- see FAIL lines above."
         (length all-failures) total-checked))
  (format t "~&check-doc-links: ~D link~:P checked, 0 failed~%" total-checked))

(uiop:quit 0)
