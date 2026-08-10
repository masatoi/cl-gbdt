;;;; custom-metric.lisp --- Building and collision-checking a custom metric's recorded entry.
;;;;
;;;; Every function here is pure: no handle, no pointer, no shared library, nothing that
;;;; requires `open-backend' at all. `custom-metric-entry' turns one call to a caller's
;;;; `:evaluation' function into the exact (DATASET-INDEX METRIC-NAME VALUE) shape both
;;;; backends' `%read-evaluation' already produce -- the shape `training-report-from-history'
;;;; (`cl-gbdt/src/training/history') folds into a report series and `observe-iteration'
;;;; (`cl-gbdt/src/training/early-stopping') reads to advance a watcher. `custom-metric-entry'
;;;; existing at all is what lets a caller's metric reach both of those readers for free, by
;;;; producing the one shape they already know how to fold in, rather than a shape of its own
;;;; either reader would need special-casing to recognize. `check-metric-name-collision' guards
;;;; one way a caller's metric could corrupt them anyway: a name that collides with one the
;;;; library itself already reports for the same dataset index, which would fold two different
;;;; series into one under the (DATASET-INDEX, METRIC-NAME) key both readers key by.
;;;; `make-metric-name-pin' and `pin-metric-name' guard the other, which is the caller's alone:
;;;; a name that CHANGES from one iteration to the next, which no per-iteration check can
;;;; catch because there is nothing wrong with either name on the iteration it appears in.
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
           #:check-metric-name-collision
           #:make-metric-name-pin
           #:pin-metric-name))

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
VALUE is `(or real null)'. NAME must be a string outright, never merely coerced to one:
`training-report-from-history' folds a series by (DATASET-INDEX, METRIC-NAME) under `equal',
and `evaluation' reports every metric name the library itself computes as a string, so a
keyword or a symbol NAME would key a series under a shape nothing else in the report could
ever be looked up by. NIL is an accepted VALUE, not a special case bolted on here: it is how
both backends already record a field they could not read as a real, and `observe-iteration'
already treats it as no improvement rather than an error, so a caller's own unreadable value
is recorded the same way."
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
it. A collision here would put two different values under the one (DATASET-INDEX,
METRIC-NAME) pair `training-report-from-history' and `observe-iteration' each use to find a
series, and each would mishandle it its own way: `training-report-from-history' would push
both values onto that one pair's accumulator within a single iteration, corrupting that
series' alignment with the run's iterations from that point on; `observe-iteration''s
`%find-watched-entry' would return only the first of the two entries `find-if' reaches for
that pair, so a watcher on it would silently read one value and never learn the other
existed.

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

(defun make-metric-name-pin ()
  "Return a fresh, empty pin for `pin-metric-name' below: state one `train' run keeps for the
whole of its loop, mapping each dataset index to the metric name that index's `:evaluation'
call returned the FIRST time it was called.

Created once per run rather than once per iteration -- a pin that forgot between iterations
would compare nothing -- and only for a run that actually has an EVALUATION, so a run without
one allocates nothing. A hash table under `eql' because a dataset index is an integer and
because the pin is written once per index and read once per index per iteration afterwards.

The pin is `train''s own state and never the caller's: nothing here is exported from
`CL-GBDT', and no argument of `train' reaches it. See `pin-metric-name' for what it is for."
  (make-hash-table :test #'eql))

(defun pin-metric-name (backend-name pin name dataset-index)
  "Record NAME as DATASET-INDEX's metric name in PIN the first time that index is seen, and
on every later call signal `unsupported-argument' naming \"train's :evaluation\" unless NAME
is `string=' to what was recorded then. Return (values).

BACKEND-NAME is the keyword `train''s own backend reports through `backend-name' --
`:lightgbm' or `:xgboost' -- passed straight through, unexamined, to the
`unsupported-argument' this signals, the same convention `custom-metric-entry' and
`check-metric-name-collision' above follow.

ONE NAME PER DATASET INDEX FOR THE WHOLE RUN is the contract this enforces, and it is what
`train''s generic-function docstring states as a requirement on EVALUATION. A caller's
function is free to return a different name at a different INDEX -- the pin is per index --
but not a different name at the same one from one iteration to the next, because a series
is keyed by the (DATASET-INDEX, METRIC-NAME) pair and a name that varies is asking for a
series nothing can align:

  - Varying WITHOUT ever colliding gives one short series per name it took. Each is pushed
    only on the iterations that name appeared in, so every one of them comes out shorter than
    the run -- breaking `train''s own \"every series is exactly that long\" guarantee in the
    ragged direction, and misaligning each value with the iteration it was measured at.
  - Varying INTO a name the library also reports gives a series LONGER than the run.
    `check-metric-name-collision' cannot catch that one: `train' runs it on the first
    iteration only, which is the first moment there is a real evaluation to compare against,
    and a caller returning a safe name then and a colliding one afterwards passes it. From
    the iteration the names meet, `training-report-from-history' pushes two values onto that
    one key per iteration and the series reaches `1 + 2(N-1)' elements over N rounds --
    longer than `training-report-num-rounds' says the run was.

Pinning closes both, and closes the second WITHOUT a second collision check: once every
index's name is fixed at the first iteration, the only name that can ever collide with the
library's is the one the first iteration already offered, which is exactly what
`check-metric-name-collision' is given. It also restores `%find-watched-entry''s stated
invariant that at most one entry matches a given (index, metric) pair -- `find-if' returns
the first of two and an early-stopping watcher would silently read one value per iteration
and never learn the other existed.

Refusing rather than renaming, or than recording under the new name: either would be this
library inventing a series key the caller never asked for, and policy section 7's rule against
silent fallbacks applies to a caller's own metric as much as to a capability. `string=' rather
than `equal' matches how both readers compare a metric name, so a name this accepts is a name
they will treat as the same one."
  (multiple-value-bind (pinned foundp) (gethash dataset-index pin)
    (cond ((not foundp)
           (setf (gethash dataset-index pin) name))
          ((not (string= pinned name))
           (error 'unsupported-argument
                  :backend backend-name
                  :argument "train's :evaluation"
                  :reason (format nil "the custom metric returned name ~S for dataset index ~
                                       ~D after returning ~S for it; one name per dataset ~
                                       index is required for the whole run"
                                  name dataset-index pinned)))))
  (values))
