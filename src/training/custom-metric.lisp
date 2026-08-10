;;;; custom-metric.lisp --- Building and collision-checking a custom metric's recorded entry.
;;;;
;;;; Both functions here are pure: no handle, no pointer, no shared library, nothing that
;;;; requires `open-backend' at all. `custom-metric-entry' turns one call to a caller's
;;;; `:evaluation' function into the exact (DATASET-INDEX METRIC-NAME VALUE) shape both
;;;; backends' `%read-evaluation' already produce -- the shape `training-report-from-history'
;;;; (`cl-gbdt/src/training/history') folds into a report series and `observe-iteration'
;;;; (`cl-gbdt/src/training/early-stopping') reads to advance a watcher. `custom-metric-entry'
;;;; existing at all is what lets a caller's metric reach both of those readers for free, by
;;;; producing the one shape they already know how to fold in, rather than a shape of its own
;;;; either reader would need special-casing to recognize. `check-metric-name-collision' guards
;;;; the one way a caller's metric could corrupt them anyway: a name that collides with one the
;;;; library itself already reports for the same dataset index, which would fold two different
;;;; series into one under the (DATASET-INDEX, METRIC-NAME) key both readers key by.
;;;;
;;;; Under src/training/, beside history.lisp and early-stopping.lisp, rather than in
;;;; src/config/: those two are already the backend-neutral, layer-1-testable training helpers
;;;; this project keeps beside the recording and stopping logic they serve rather than with the
;;;; dataset- and parameter-shaping helpers under src/config/, and this is the same kind of
;;;; work -- a third pure step in the same per-iteration path, not a dataset or parameter
;;;; concern.
;;;;
;;;; Under src/training/ rather than directly under src/, and so deliberately absent from
;;;; src/all.lisp's `use-reexport' list, for the same reason `src/training/history.lisp' and
;;;; `src/training/early-stopping.lisp' are absent from it -- see their own headers. Tasks 2
;;;; and 3 are this file's only intended callers, each from its own backend's `train'.
;;;;
;;;; Consumers: `cl-gbdt/src/lightgbm/protocol' and `cl-gbdt/src/xgboost/protocol', each from
;;;; its own `train' (Tasks 2 and 3).

(uiop:define-package #:cl-gbdt/src/training/custom-metric
  (:use #:cl)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-argument)
  (:export #:custom-metric-entry
           #:check-metric-name-collision))

(in-package #:cl-gbdt/src/training/custom-metric)

(defun custom-metric-entry (backend-name name value dataset-index)
  "Return (list DATASET-INDEX NAME VALUE), the entry one call to a caller's `:evaluation'
function contributes to an iteration -- the exact (DATASET-INDEX METRIC-NAME VALUE) shape
both backends' `%read-evaluation' already produce, which is what lets
`training-report-from-history' and `observe-iteration' fold this entry in alongside the
library's own without either needing to know it came from somewhere else.

BACKEND-NAME is the keyword `train''s own backend reports through `backend-name' --
`:lightgbm' or `:xgboost' -- passed straight through, unexamined, to every
`unsupported-argument' this signals, exactly as `cl-gbdt/src/training/early-stopping''s
`train-early-stopping-watcher' and `make-early-stopping-watcher' take and pass their own, so
a caller sees the backend it actually called `train' on rather than a backend that does not
exist. Threading a keyword this way costs nothing: both `train' methods already have
`backend' in scope and can pass `(backend-name backend)'.

Signals `unsupported-argument' naming \"train's :evaluation\" unless NAME is a `string' and
VALUE is `(or real null)'. NAME must be a string outright, never merely coerced to one: both
readers key a series by (DATASET-INDEX, METRIC-NAME) under `equal', and `evaluation' reports
every metric name the library itself computes as a string, so a keyword or a symbol NAME
would key a series under a shape nothing else in the report could ever be looked up by. NIL
is an accepted VALUE, not a special case bolted on here: it is how both backends already
record a field they could not read as a real, and `observe-iteration' already treats it as no
improvement rather than an error, so a caller's own unreadable value is recorded the same
way."
  (unless (stringp name)
    (error 'unsupported-argument
           :backend backend-name
           :argument "train's :evaluation"
           :reason (format nil "a custom metric's name must be a string, got ~S" name)))
  (unless (or (realp value) (null value))
    (error 'unsupported-argument
           :backend backend-name
           :argument "train's :evaluation"
           :reason (format nil "a custom metric's value must be a real number or NIL, got ~S"
                            value)))
  (list dataset-index name value))

(defun check-metric-name-collision (backend-name name dataset-index library-entries)
  "Signal `unsupported-argument' naming \"train's :evaluation\" when LIBRARY-ENTRIES holds an
entry whose index is DATASET-INDEX and whose name is `string=' to NAME; return (values)
otherwise.

BACKEND-NAME is the keyword `train''s own backend reports through `backend-name' --
`:lightgbm' or `:xgboost' -- passed straight through, unexamined, to the
`unsupported-argument' this signals, the same convention `custom-metric-entry' above and
`cl-gbdt/src/training/early-stopping''s watcher functions follow: a caller sees the backend it
actually called `train' on rather than a backend that does not exist.

LIBRARY-ENTRIES is one iteration's worth of the library's OWN (DATASET-INDEX METRIC-NAME
VALUE) entries -- what `%read-evaluation' returns before a caller's own metric is appended to
it. A collision here would fold the caller's series into the library's own under
`training-report-from-history''s and `observe-iteration''s shared (DATASET-INDEX,
METRIC-NAME) key, so whichever of the two values arrived second for a given iteration would
silently overwrite the other every time, in a series a reader has no way to tell was ever
shared.

Checked against one iteration's actual entries rather than at `train''s own entry, for the
same reason `observe-iteration''s docstring gives for its own first-call metric check: what
the library reports for a given dataset index cannot be known before a booster has produced a
real evaluation to report it in, so there is nothing yet to compare NAME against before
then."
  (when (find-if (lambda (entry)
                    (and (eql (first entry) dataset-index)
                         (string= (second entry) name)))
                  library-entries)
    (error 'unsupported-argument
           :backend backend-name
           :argument "train's :evaluation"
           :reason (format nil "~S already names a metric the library reports for dataset ~
                                index ~D" name dataset-index)))
  (values))
