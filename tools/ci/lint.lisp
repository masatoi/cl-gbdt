;;;; lint.lisp --- Static checks over the hand-written Lisp sources.
;;;;
;;;; Usage:
;;;;   ros run -- --non-interactive --load tools/ci/lint.lisp
;;;;
;;;; Two checks, because neither covers the other:
;;;;
;;;;   mallet          style rules
;;;;   column width    mallet does NOT check line length -- a 132-column file passes
;;;;                   it without comment, which was verified. Every task in this
;;;;                   project's history treated a clean mallet run as covering the
;;;;                   100-column rule; it never did.
;;;;
;;;; The generated src/*/c-api.lisp are excluded: they are machine output and are not
;;;; held to the style guide.

(require :asdf)

(ql:quickload "mallet" :silent t)

(defparameter +maximum-columns+ 100
  "The column limit the Google Common Lisp Style Guide sets, per this project's
CLAUDE.md.")

(defparameter +source-patterns+
  '("src/*.lisp" "src/regen/*.lisp" "tests/*.lisp" "tests/functional/*.lisp"
    "tools/ci/*.lisp" "tools/regen.lisp")
  "Hand-written Lisp sources, relative to the repository root.")

(defun hand-written-sources ()
  "Return the pathnames of every hand-written Lisp source, generated files excluded."
  (remove-if (lambda (path) (string= "c-api" (pathname-name path)))
             (mapcan (lambda (pattern)
                       (directory (merge-pathnames pattern (uiop:getcwd))))
                     +source-patterns+)))

(defun over-long-lines (path)
  "Return a list of (LINE-NUMBER . WIDTH) for lines in PATH exceeding the limit."
  (with-open-file (in path :external-format :utf-8)
    (loop :for line := (read-line in nil)
          :for number :from 1
          :while line
          :when (> (length line) +maximum-columns+)
            :collect (cons number (length line)))))

(defun report-column-violations (sources)
  "Print every over-long line in SOURCES. Returns the number found."
  (let ((total 0))
    (dolist (path sources total)
      (dolist (violation (over-long-lines path))
        (incf total)
        (format *error-output* "~&~A:~D is ~D columns, over the ~D limit~%"
                (enough-namestring path (uiop:getcwd))
                (car violation) (cdr violation) +maximum-columns+)))))

(defun report-mallet-violations (sources)
  "Print every mallet violation in SOURCES. Returns the number found."
  (let ((results (mallet:lint-files sources))
        (total 0))
    (dolist (entry results total)
      (dolist (violation (cdr entry))
        (incf total)
        (format *error-output* "~&~A:~A:~A ~A [~A]~%"
                (enough-namestring (car entry) (uiop:getcwd))
                (mallet/violation:violation-line violation)
                (mallet/violation:violation-column violation)
                (mallet/violation:violation-message violation)
                (mallet/violation:violation-rule violation))))))

(let* ((sources (hand-written-sources))
       (columns (report-column-violations sources))
       (style (report-mallet-violations sources)))
  (format t "~&checked ~D hand-written source~:P~%" (length sources))
  (format t "~&column violations: ~D~%" columns)
  (format t "~&mallet violations: ~D~%" style)
  (uiop:quit (if (zerop (+ columns style)) 0 1)))
