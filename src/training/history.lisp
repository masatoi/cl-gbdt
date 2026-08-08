;;;; history.lisp --- Folding a training run's per-iteration evaluations into a report.
;;;;
;;;; Core and backend-neutral: nothing here holds a foreign pointer, calls a shared library,
;;;; or knows which backend produced its input. Both backends' `train' methods record the
;;;; same normalized entry shape -- (DATASET-INDEX METRIC-NAME VALUE) lists, which is what
;;;; each backend's own `%read-evaluation' exists to produce -- so the fold from a run's
;;;; worth of those into a `training-report' is the same code for both. It lives here once
;;;; rather than in each `train', in the same spirit as `%read-evaluation' itself: one code
;;;; path, so the two backends cannot drift apart in what they record.
;;;;
;;;; Under src/training/ rather than directly under src/, and so deliberately absent from
;;;; src/all.lisp's `use-reexport' list. That list is not optional for a direct child of
;;;; src/ -- `all-lisp-reexports-every-top-level-src-file' in tests/bindings.lisp asserts
;;;; that every src/*.lisp appears in it, and says why -- so a top-level file here would
;;;; necessarily publish `training-report-from-history' from `CL-GBDT'. A caller reads a
;;;; report, it never builds one, and Phase 3b's early stopping may well want a different
;;;; shape for this; publishing it now would commit to this one. `src/regen/' is the
;;;; existing precedent for a subdirectory of src/ that is core rather than backend-specific
;;;; and whose files are still imported directly by their consumers rather than reexported.
;;;; Phase 3b's own backend-neutral logic belongs beside this file.
;;;;
;;;; Consumers: `cl-gbdt/src/lightgbm/protocol' and `cl-gbdt/src/xgboost/protocol', each
;;;; from its own `train'.

(uiop:define-package #:cl-gbdt/src/training/history
  (:use #:cl)
  (:import-from #:cl-gbdt/src/training-report
                #:make-training-series
                #:make-training-report)
  (:export #:training-report-from-history))

(in-package #:cl-gbdt/src/training/history)

(defun training-report-from-history (history num-rounds)
  "Return a `training-report' over HISTORY, the record of a NUM-ROUNDS-iteration run.

HISTORY has one element per completed iteration, in iteration order, and each element is
that iteration's whole evaluation: a list of (DATASET-INDEX METRIC-NAME VALUE) lists, the
shape both backends' `%read-evaluation' returns. VALUE may be NIL, which is how a backend
reports a field it could not read as a real.

The result carries one `training-series' per distinct (DATASET-INDEX, METRIC-NAME) pair,
holding that pair's values across the run as a `simple-vector' in iteration order. A NIL
value takes its slot like any other, so a series stays aligned with the iteration
numbering instead of sliding every later value one iteration earlier -- see
`training-series-values' for why that matters and why the vector is a `simple-vector'.

Series come back in the order their pairs were FIRST SEEN, which is the order the first
iteration's entries arrived in, which is the backend's own evaluation order. Nothing is
sorted, deliberately: what makes the report usable is that `training-report-series' and
`evaluation' list the same pairs in the same order, so a caller who can read one can read
the other, and imposing any ordering of this function's own would destroy that.

NUM-ROUNDS is recorded as given rather than derived from HISTORY's length: a run with no
metric configured records an empty evaluation every iteration, and it still ran."
  (let ((values-by-key (make-hash-table :test #'equal))
        (keys '()))
    (dolist (entries history)
      (dolist (entry entries)
        (destructuring-bind (index metric-name value) entry
          (let ((key (cons index metric-name)))
            ;; `nth-value' 1 is presence, not truth: a pair whose first recorded value is
            ;; NIL has still been seen, and must not reach KEYS a second time.
            (unless (nth-value 1 (gethash key values-by-key))
              (push key keys))
            (push value (gethash key values-by-key))))))
    (setf keys (nreverse keys))
    (make-training-report
     :num-rounds num-rounds
     :series (loop :for key :in keys
                   :collect (make-training-series
                             :index (car key)
                             :metric (cdr key)
                             :values (coerce (reverse (gethash key values-by-key))
                                             'simple-vector))))))
