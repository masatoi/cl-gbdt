;;;; check-support-matrix.lisp --- the README's CI-verified versions match what CI runs.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/check-support-matrix.lisp
;;;;
;;;; README.markdown's "Supported environments" table has a CI-verified column that names,
;;;; per backend, the exact library versions the functional suite actually runs against. That
;;;; claim had no checker of its own until now -- a version could be added to or dropped from
;;;; `.github/workflows/test.yml' without the README ever being told, and the table would keep
;;;; asserting whatever it last said.
;;;;
;;;; WHAT IS CHECKED. Per backend (LightGBM, XGBoost), the set of `X.Y.Z' version numbers the
;;;; README's CI-verified cell states must equal, exactly, the union of:
;;;;
;;;;   - the version `ffi-spec/VERSIONS' pins (one `<name> v<version>' line per backend --
;;;;     this is what `./tools/fetch-libs.sh' downloads for the ordinary `test' job's three
;;;;     platform legs, and is why the README calls this version "(pinned)");
;;;;   - every `lightgbm_version:'/`xgboost_version:' value named in
;;;;     `.github/workflows/test.yml''s `version-matrix' job, which exercises the recorded
;;;;     compatible range's lower endpoint against the *other* backend's pinned version (see
;;;;     that job's own extensive header comment for why only the endpoints are run, not
;;;;     every release in between).
;;;;
;;;; A version CI actually tests that the README fails to claim, and a version the README
;;;; claims that CI does not actually test, are BOTH failures, checked in both directions --
;;;;;; the second is the one a reader would never notice on their own, since it looks like
;;;; extra assurance rather than a false claim.
;;;;
;;;; READING `test.yml' WITHOUT A YAML PARSER. There is no YAML library in this project's
;;;; dependencies, and adding one for two keys would not be proportionate, so MATRIX-VERSIONS
;;;; reads `lightgbm_version:'/`xgboost_version:' with a regex instead. A first version of
;;;; that regex matched only the single-quoted scalar (`lightgbm_version: '4.0.0''); YAML
;;;; permits three ways to write the same value -- single-quoted, double-quoted
;;;; (`"4.0.0"') and bare (`4.0.0') -- and `test.yml' happening to use the first one made the
;;;; gap invisible rather than absent. The failure mode runs in both directions and the worse
;;;; one is silent: a version written the other two ways would either go undetected by
;;;; MATRIX-VERSIONS entirely -- so if the README also omitted it, this check would report
;;;; agreement while CI actually tested an unverified version, exactly backwards from this
;;;; check's purpose -- or, if the README did state it, fail this check for a reason that is
;;;; not real. MATRIX-VERSIONS' regex now accepts all three forms; see its own docstring.
;;;;
;;;; THE OTHER SIDE OF THAT SAME FIX. Accepting the bare scalar form widened MATRIX-VERSIONS'
;;;; own false-POSITIVE surface at the same time, because its regex carried no line anchor --
;;;; unlike PINNED-VERSION's, in the same file, which always has. Unanchored, `lightgbm_version'
;;;; sitting inside a `#' comment, inside a `run: |' shell block (`echo lightgbm_version:
;;;; 9.9.9'), or glued onto some other key's name (`not_lightgbm_version:',
;;;; `lightgbm_version_note:') could each match and feed a version into
;;;; EXPECTED-CI-VERIFIED-VERSIONS that `version-matrix' never actually names as a leg -- and
;;;; this direction fails the build on an ordinary, unrelated edit rather than staying silent.
;;;; MATRIX-VERSIONS is now anchored to the whole line (`(?m)^...$'), the same way
;;;; PINNED-VERSION already is, which rules out every case above: none of them is a standalone
;;;; line consisting of nothing but the key, its value, and surrounding whitespace. What
;;;; anchoring cannot rule out is a `run: |' block or heredoc whose body happens to contain a
;;;; STANDALONE line in exactly that shape -- indistinguishable from a real matrix entry to a
;;;; regex that has no notion of being inside a YAML scalar block. PINNED-VERSION accepts the
;;;; identically-shaped limitation for its own file; anchoring the two alike, rather than
;;;; trying to out-parse YAML with a larger regex, is the fix, not a claim of having none left.
;;;;
;;;; WHAT IS DELIBERATELY NOT CHECKED: the "Also measured" column. LightGBM 3.0.0 and XGBoost
;;;; 1.7.0 live there because they were measured BY HAND, once, and recorded rather than
;;;; re-run on every push -- XGBoost 1.7.0 in particular is a deliberately-red result (see
;;;; `.github/workflows/test.yml''s `version-matrix' header: the wheel does not even install
;;;; on the runner, so it cannot be a CI leg at all). That column exists precisely so those
;;;; measurements can be recorded WITHOUT claiming CI reproduces them on every push. A later
;;;; reader who "improves" this check by extending it to that column would be asking CI to
;;;; verify a claim the table's own lead-in sentence says is not CI's claim to verify -- see
;;;; README.markdown's "Supported environments" section, and do not do that.
;;;;
;;;; Also unchecked, for the same reason it is unchecked by the README itself: the ASDF row,
;;;; which states in its own CI-verified cell that no version is verified at all ("CI runs
;;;; `ros install asdf', taking whatever the current release is that day").
;;;;
;;;; WHAT THIS CANNOT CATCH
;;;;
;;;;   - Whether the versions actually agree for a reason, or by coincidence -- this compares
;;;;     two independently-maintained sources of truth (a table a human edits, a YAML file a
;;;;     human edits) and says only whether they currently state the same set.
;;;;   - A version-matrix leg that is present in `test.yml' but disabled, skipped or
;;;;     permanently red (`continue-on-error') -- see that job's XGBoost 1.7.0 story for why
;;;;     that distinction matters and is NOT this script's job to draw; it reads the matrix
;;;;     declaratively, not by asking whether the leg's last run actually passed.
;;;;   - A version written in YAML flow or block styles this regex does not anticipate (a
;;;;     folded or literal scalar, for instance) -- MATRIX-VERSIONS matches the three plain
;;;;     scalar forms `version-matrix' actually uses today, not the whole of YAML's scalar
;;;;     grammar.
;;;;   - A standalone line inside a `run: |' block or heredoc that happens to read exactly
;;;;     `KEY: value' with nothing else on it -- see "THE OTHER SIDE OF THAT SAME FIX" above;
;;;;     the line anchor rules out every OTHER false-positive shape, not this one.

(require :asdf)

(ql:quickload '("cl-ppcre") :silent t)

(defparameter +versions-path+ "ffi-spec/VERSIONS"
  "Path, relative to the repository root, of the file pinning each backend's vendored
version -- one `<name> v<version>' line per backend.")

(defparameter +test-workflow-path+ ".github/workflows/test.yml"
  "Path, relative to the repository root, of the workflow whose `version-matrix' job names
every additional version CI exercises beyond the pinned one.")

(defparameter +readme-path+ "README.markdown"
  "Path, relative to the repository root, of the file whose \"Supported environments\" table
this check verifies.")

(defparameter +backends+
  '((:versions-key "lightgbm" :matrix-key "lightgbm_version" :readme-row "LightGBM")
    (:versions-key "xgboost" :matrix-key "xgboost_version" :readme-row "XGBoost"))
  "One plist per backend. :VERSIONS-KEY is its name as +VERSIONS-PATH+ writes it,
:MATRIX-KEY the YAML key +TEST-WORKFLOW-PATH+'s `version-matrix' job uses for it, and
:README-ROW the first cell of its row in +README-PATH+'s environments table.")

(defparameter +version-number-scanner+
  (cl-ppcre:create-scanner "\\d+\\.\\d+\\.\\d+")
  "Matches one `X.Y.Z' version number, used against every text this check reads versions
out of.")

(defun die (format-control &rest arguments)
  "Print FORMAT-CONTROL/ARGUMENTS to *ERROR-OUTPUT* as a FAIL line and exit with status 1."
  (format *error-output* "~&FAIL ~?~%" format-control arguments)
  (uiop:quit 1))

(defun version-numbers-in (text)
  "Return the sorted, duplicate-free list of every `X.Y.Z' version number CL-PPCRE:ALL-
MATCHES-AS-STRINGS finds in TEXT."
  (sort (remove-duplicates (cl-ppcre:all-matches-as-strings +version-number-scanner+ text)
                            :test #'string=)
        #'string<))

(defun pinned-version (versions-key)
  "Return the version +VERSIONS-PATH+ pins for VERSIONS-KEY (\"lightgbm\" or \"xgboost\"),
or NIL if that file names no such backend."
  (cl-ppcre:register-groups-bind (version)
      ((format nil "(?m)^~A\\s+v(\\S+)\\s*$" (cl-ppcre:quote-meta-chars versions-key))
       (uiop:read-file-string +versions-path+))
    version))

(defun matrix-versions (matrix-key)
  "Return every version +TEST-WORKFLOW-PATH+'s `version-matrix' job names for MATRIX-KEY
(\"lightgbm_version\" or \"xgboost_version\"), duplicates included -- callers that want a set
should deduplicate. Matches all three YAML scalar forms a version can be written in -- single-
quoted, double-quoted and bare -- since YAML permits all three and nothing here parses YAML;
see this file's header for why matching only the single-quoted form was a real gap, not a
theoretical one. Anchored to the whole line, the same way PINNED-VERSION already anchors its
own regex, so a bare MATRIX-KEY sitting inside a comment, a `run:' shell line, or as a prefix
or suffix of some other key can no longer match -- only accepting the bare scalar form
widened this function's own surface for exactly that, which PINNED-VERSION's file format
never exposed it to."
  (let ((versions '())
        (scanner (cl-ppcre:create-scanner
                  (format nil "(?m)^[ \\t]*~A:\\s*[\"']?([\\d.]+)[\"']?[ \\t]*$"
                          (cl-ppcre:quote-meta-chars matrix-key)))))
    (cl-ppcre:do-register-groups (version) (scanner (uiop:read-file-string +test-workflow-path+))
      (push version versions))
    (nreverse versions)))

(defun readme-row-cells (readme-row)
  "Return the pipe-delimited cells (trimmed, leading/trailing empty cells dropped) of
+README-PATH+'s table row beginning `| READMEROW |', or NIL if no such row is found."
  (let ((line (cl-ppcre:register-groups-bind (whole)
                  ((format nil "(?m)^(\\|\\s*~A\\s*\\|.*)$"
                           (cl-ppcre:quote-meta-chars readme-row))
                   (uiop:read-file-string +readme-path+))
                whole)))
    (when line
      (mapcar (lambda (cell) (string-trim '(#\Space #\Tab) cell))
              (remove-if (lambda (cell) (zerop (length (string-trim '(#\Space #\Tab) cell))))
                         (uiop:split-string line :separator '(#\|)))))))

(defun readme-ci-verified-versions (readme-row)
  "Return the sorted, duplicate-free list of version numbers in the CI-verified cell (the
second cell, after the row label) of +README-PATH+'s READMEROW row. Deliberately reads only
that cell, never the \"Also measured\" cell that follows it -- see this file's header."
  (let ((cells (readme-row-cells readme-row)))
    (unless (>= (length cells) 2)
      (die "could not find a `| ~A | ... |' row in ~A" readme-row +readme-path+))
    (version-numbers-in (second cells))))

(defun expected-ci-verified-versions (backend)
  "Return the sorted, duplicate-free list of versions CI actually exercises for BACKEND (a
plist from +BACKENDS+): +VERSIONS-PATH+'s pin, union `.github/workflows/test.yml''s
`version-matrix' entries for it."
  (let ((pinned (pinned-version (getf backend :versions-key)))
        (matrix (matrix-versions (getf backend :matrix-key))))
    (unless pinned
      (die "~A pins no version for ~A" +versions-path+ (getf backend :versions-key)))
    (sort (remove-duplicates (cons pinned matrix) :test #'string=) #'string<)))

(defun check-backend (backend)
  "Check BACKEND (a plist from +BACKENDS+): the README's CI-verified versions must equal,
exactly, EXPECTED-CI-VERIFIED-VERSIONS. Returns T and prints an OK line on a match; prints a
FAIL line describing the mismatch in both directions and returns NIL otherwise."
  (let* ((readme-row (getf backend :readme-row))
         (readme-versions (readme-ci-verified-versions readme-row))
         (expected-versions (expected-ci-verified-versions backend))
         (readme-only (set-difference readme-versions expected-versions :test #'string=))
         (ci-only (set-difference expected-versions readme-versions :test #'string=)))
    (cond
      ((and (null readme-only) (null ci-only))
       (format t "~&~A: README and CI agree on ~{~A~^, ~}~%" readme-row expected-versions)
       t)
      (t
       (when readme-only
         (format *error-output*
                 "~&FAIL ~A: README claims ~{~A~^, ~} as CI-verified, but CI does not test ~
                  ~[no version~;that version~:;those versions~]~%"
                 readme-row readme-only (length readme-only)))
       (when ci-only
         (format *error-output*
                 "~&FAIL ~A: CI tests ~{~A~^, ~}, but the README's CI-verified cell omits ~
                  ~[no version~;that version~:;those versions~]~%"
                 readme-row ci-only (length ci-only)))
       nil))))

;;; ---- Main check ----

(let ((all-ok t))
  (dolist (backend +backends+)
    (unless (check-backend backend)
      (setf all-ok nil)))
  (unless all-ok
    (die "the README's CI-verified column does not match ffi-spec/VERSIONS + ~A's ~
          version-matrix -- see FAIL lines above."
         +test-workflow-path+))
  (format t "~&check-support-matrix: ~D backend~:P checked, README matches CI~%"
          (length +backends+)))

(uiop:quit 0)
