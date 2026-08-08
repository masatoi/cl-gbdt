;;;; early-stopping.lisp --- Deciding whether a training run should stop.
;;;;
;;;; Backend-neutral and pure: no handle, no pointer, no shared library, nothing that
;;;; requires `open-backend' at all. `observe-iteration' sees one iteration's worth of
;;;; entries -- the exact (DATASET-INDEX METRIC-NAME VALUE) shape `%read-evaluation'
;;;; produces and `cl-gbdt:evaluation' returns -- and answers a boolean; it never touches a
;;;; booster, a dataset, or a C call. That purity is what makes this file testable at
;;;; layer 1 rather than only against the real libraries, which is why the stop logic lives
;;;; here rather than inside either backend's `train'.
;;;;
;;;; Under src/training/ rather than directly under src/, and so deliberately absent from
;;;; src/all.lisp's `use-reexport' list, for the same reason `src/training/history.lisp'
;;;; is absent from it -- see that file's own header. Task 3 (Phase 3b) is this file's
;;;; only intended caller, from each backend's own `train'; a caller reads the watcher's
;;;; result, it never needs to build a watcher of a different shape, so publishing this
;;;; from `CL-GBDT' would commit to a shape before there is a second caller to test it
;;;; against.
;;;;
;;;; Consumers: `cl-gbdt/src/lightgbm/protocol' and `cl-gbdt/src/xgboost/protocol', each
;;;; from its own `train' (Task 3).

(uiop:define-package #:cl-gbdt/src/training/early-stopping
  (:use #:cl)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-argument)
  (:export #:make-early-stopping-watcher
           #:train-early-stopping-watcher
           #:observe-iteration
           #:watcher-best-iteration
           #:watcher-best-score
           #:watcher-stopped-p))

(in-package #:cl-gbdt/src/training/early-stopping)

;;; ---------------------------------------------------------------------------
;;; The watcher

(defstruct (early-stopping-watcher (:conc-name watcher-)
                                    (:constructor %make-early-stopping-watcher))
  "The state of one early-stopping decision: which (dataset, metric) series to watch, in
which direction, for how many non-improving rounds -- all fixed at construction -- plus
the mutable running state `observe-iteration' updates as a run progresses.

INDEX and METRIC together select one series out of an iteration's ENTRIES, the same way
`training-report-from-history' keys a series by (DATASET-INDEX . METRIC-NAME). DIRECTION
is `:lower-is-better' or `:higher-is-better' -- see `make-early-stopping-watcher' for why
this is a required, caller-supplied field rather than something inferred. ROUNDS is how
many consecutive non-improving iterations `observe-iteration' tolerates before it reports
a stop.

BEST-ITERATION and BEST-SCORE record the best value seen so far and which iteration
produced it, both NIL until the first real value arrives. SINCE-IMPROVEMENT counts
consecutive iterations, including the current one, that did not improve on BEST-SCORE;
STOPPED-P latches true the first time it reaches ROUNDS and stays true afterward, since a
watcher that has already recommended stopping has nothing new to say on a later call."
  (index 0)
  (metric "")
  (direction :lower-is-better)
  (rounds 1)
  (best-iteration nil)
  (best-score nil)
  (since-improvement 0)
  (stopped-p nil))

;;; ---------------------------------------------------------------------------
;;; Construction: parsing and validating SPEC

(defun %signal-unsupported (backend-name argument-name reason)
  "Signal `unsupported-argument' naming BACKEND-NAME and ARGUMENT-NAME, with REASON.

The construction-time checks below -- `%require-metric', `%require-direction',
`%require-rounds', `%resolve-dataset' and `train-early-stopping-watcher''s own
:record-history check -- all raise the identical shape of condition and differed before
only in ARGUMENT-NAME and REASON; folding the boilerplate here is what keeps BACKEND-NAME
set correctly in one place rather than copied into eight near-identical `error' forms."
  (error 'unsupported-argument :backend backend-name :argument argument-name :reason reason))

(defun %require-metric (backend-name spec)
  "Return SPEC's :metric, the name of the series `observe-iteration' watches.

Signals `unsupported-argument' naming BACKEND-NAME when :metric is absent or not a string.
This is one of the four keys section 9 requires up front, so a malformed SPEC fails before
any training iteration runs rather than partway through one."
  (let ((metric (getf spec :metric)))
    (unless (stringp metric)
      (%signal-unsupported backend-name "train's :early-stopping :metric"
                            (format nil ":metric is required and must be a string, got ~S"
                                     metric)))
    metric))

(defun %require-direction (backend-name spec)
  "Return SPEC's :direction, `:lower-is-better' or `:higher-is-better'.

Signals `unsupported-argument' naming BACKEND-NAME when :direction is absent or is
neither. Neither backend's C API exposes whether a metric improves upward or downward --
grepping both vendored headers for `higher_better', `maximize', `direction' and
`greater_is' turns up nothing, and the only evaluation-related functions bound are the
four `%read-evaluation' already wraps -- so this cannot be read off the backend the way a
capability can. Policy section 9 permits deciding this from information the backend
provides or from an explicit caller value, and only the second exists here; inferring it
from the metric's name instead (a table of \"logloss is lower, auc is higher\") would be
the same guess with a lookup table in front of it, and section 9 forbids it for the same
reason it forbids the guess itself."
  (let ((direction (getf spec :direction)))
    (unless (member direction '(:lower-is-better :higher-is-better))
      (%signal-unsupported backend-name "train's :early-stopping :direction"
                            (format nil ":direction is required and must be ~
                                         :lower-is-better or :higher-is-better, got ~S"
                                     direction)))
    direction))

(defun %require-rounds (backend-name spec)
  "Return SPEC's :rounds, a positive integer: how many consecutive non-improving
iterations `observe-iteration' tolerates before it reports a stop.

Signals `unsupported-argument' naming BACKEND-NAME when :rounds is absent or is not a
positive integer -- zero or negative would report a stop before the run had any real
chance to improve, which is never a useful watcher."
  (let ((rounds (getf spec :rounds)))
    (unless (and (integerp rounds) (plusp rounds))
      (%signal-unsupported backend-name "train's :early-stopping :rounds"
                            (format nil ":rounds is required and must be a positive ~
                                         integer, got ~S" rounds)))
    rounds))

(defun %resolve-dataset (backend-name spec dataset-names)
  "Return the position in DATASET-NAMES that SPEC's :dataset selects.

Signals `unsupported-argument' naming BACKEND-NAME when :dataset is absent or is neither a
string nor a non-negative integer, when an integer is out of range for DATASET-NAMES, when
a string matches no element, or when a string matches more than one -- Phase 3a
deliberately allows two `:valid-sets' entries to share one name, since the index tells
them apart in the report, but here the name must pick exactly one series to watch, and
silently taking the first match would make which one it picked invisible to the caller."
  (let ((dataset (getf spec :dataset)))
    (cond
      ((integerp dataset)
       (unless (and (>= dataset 0) (< dataset (length dataset-names)))
         (%signal-unsupported backend-name "train's :early-stopping :dataset"
                               (format nil ":dataset ~D is out of range for ~D dataset~:P"
                                        dataset (length dataset-names))))
       dataset)
      ((stringp dataset)
       (let ((matches (loop :for name :in dataset-names
                             :for position :from 0
                             :when (equal name dataset)
                               :collect position)))
         (case (length matches)
           (0 (%signal-unsupported backend-name "train's :early-stopping :dataset"
                                    (format nil "no dataset is named ~S" dataset)))
           (1 (first matches))
           (t (%signal-unsupported
               backend-name "train's :early-stopping :dataset"
               (format nil "~S names more than one dataset, at indices ~{~D~^, ~}; pass ~
                            the index instead to pick one" dataset matches))))))
      (t (%signal-unsupported backend-name "train's :early-stopping :dataset"
                               (format nil ":dataset is required and must be a string or ~
                                            a non-negative integer, got ~S" dataset))))))

(defun make-early-stopping-watcher (backend-name spec dataset-names)
  "Parse SPEC into an `early-stopping-watcher' that `observe-iteration' advances.

BACKEND-NAME is the keyword the caller's own `train' method runs under -- `:lightgbm' or
`:xgboost' -- and is used only to name which backend an `unsupported-argument' condition
reports; nothing here compares it or otherwise interprets it.

SPEC is a plist and all four of :metric, :dataset, :direction and :rounds are required;
any one absent or malformed signals `unsupported-argument' -- see `%require-metric',
`%require-direction', `%require-rounds' and `%resolve-dataset' for what each accepts and
why. There is no optional fifth key and no default for any of the four: a caller who
wants early stopping states exactly what to watch, in which direction, and for how long,
rather than inheriting a guess.

:dataset selects which series to watch, either by name (a string, resolved against
DATASET-NAMES) or by position (a non-negative integer, checked against DATASET-NAMES'
length). DATASET-NAMES is the list `train' already builds for its report: `(cons nil
valid-set-names)', so position 0 is always the training set, with no name of its own, and
position N+1 is the Nth `:valid-sets' entry.

:metric naming a series the booster never actually reports is not caught here -- SPEC
alone cannot say what a booster will report before it runs a single iteration -- but by
`observe-iteration', the first time it looks for that series and does not find it."
  (let ((metric (%require-metric backend-name spec))
        (direction (%require-direction backend-name spec))
        (rounds (%require-rounds backend-name spec))
        (index (%resolve-dataset backend-name spec dataset-names)))
    (%make-early-stopping-watcher :index index :metric metric :direction direction
                                   :rounds rounds)))

(defun train-early-stopping-watcher (backend-name early-stopping record-history dataset-names)
  "Return the watcher EARLY-STOPPING asks for, or NIL when EARLY-STOPPING is NIL and the
run is therefore unwatched.

The entry point both backends' `train' calls, rather than `make-early-stopping-watcher'
directly: it holds the one rule that belongs to `train''s argument list rather than to a
spec on its own, so that neither backend has to carry a copy of it. Lives here, beside the
four validators it joins, because it is the same kind of work they do -- reject a
malformed request before any iteration runs -- and needs nothing a backend has beyond
BACKEND-NAME: NIL is passed through, RECORD-HISTORY is a boolean, and DATASET-NAMES is the
list `train' already built. Nothing here touches a handle, a pointer or a shared library,
so this file stays as testable at layer 1 as its header promises.

BACKEND-NAME is the keyword `train''s own backend reports through `backend-name' --
`:lightgbm' or `:xgboost' -- passed straight through, unexamined, to every
`unsupported-argument' this function or `make-early-stopping-watcher' signals, so a caller
sees the backend it actually called `train' on rather than a backend that does not exist.
Threading a keyword this way costs nothing: both `train' methods already have `backend' in
scope and can pass `(backend-name backend)', which adds no import and no dependency on the
backend protocol -- unlike a `backend' *object*, which would.

Signals `unsupported-argument' when EARLY-STOPPING is supplied together with
RECORD-HISTORY NIL. The two contradict each other: a watcher advances on the very
per-iteration evaluation RECORD-HISTORY NIL exists to skip, and reading that evaluation
costs the same whether one series is watched or every series is recorded, so there is no
cheaper middle path to offer a caller who asked for both -- accepting the pair and quietly
recording after all, or accepting it and never stopping, would each be a different answer
than the one asked for.

The condition names :BACKEND BACKEND-NAME here exactly as `%require-metric' and the other
three validators of the same spec do, so a caller reading that slot gets the backend it
actually called `train' on whichever of the five checks fired -- never a placeholder like
`:early-stopping', which names no backend `open-backend' ever returns.

Otherwise delegates to `make-early-stopping-watcher', which validates the spec's four
required keys against DATASET-NAMES -- see its docstring. Both backends call this before
their own booster constructor, so a rejected spec never leaves a raw booster handle behind
to unwind."
  (when early-stopping
    (unless record-history
      (%signal-unsupported backend-name "train's :early-stopping"
                            (format nil ":early-stopping needs the per-iteration ~
                                         evaluation :record-history NIL skips; pass ~
                                         :record-history T, or drop :early-stopping")))
    (make-early-stopping-watcher backend-name early-stopping dataset-names)))

;;; ---------------------------------------------------------------------------
;;; Advancing the watcher

(defun %comparator (watcher)
  "Return the strict two-argument predicate WATCHER's `watcher-direction' implies: `<' for
`:lower-is-better', `>' for `:higher-is-better'. Both are strict rather than `<=' / `>=' --
a plateau is not an improvement, so it must not reset `watcher-since-improvement'."
  (ecase (watcher-direction watcher)
    (:lower-is-better #'<)
    (:higher-is-better #'>)))

(defun %find-watched-entry (watcher entries)
  "Return the element of ENTRIES matching WATCHER's index and metric, or NIL.

ENTRIES is one iteration's (DATASET-INDEX METRIC-NAME VALUE) lists, in whatever order the
backend produced them; nothing about that order is assumed here, only that at most one
entry matches a given (index, metric) pair, which is exactly the (DATASET-INDEX,
METRIC-NAME) key `training-report-from-history' relies on for the same reason."
  (find-if (lambda (entry)
             (and (eql (first entry) (watcher-index watcher))
                  (string= (second entry) (watcher-metric watcher))))
           entries))

(defun observe-iteration (watcher entries iteration)
  "Advance WATCHER by one completed ITERATION and return true when the run should stop
after it.

ENTRIES is that iteration's whole evaluation, the exact (DATASET-INDEX METRIC-NAME VALUE)
shape `%read-evaluation' returns; only the one entry matching WATCHER's index and metric
is read; every other entry -- another metric, another dataset -- is ignored. Signals
`unsupported-argument' when no entry matches: SPEC's :metric could not be checked against
what the booster actually reports until a real evaluation existed to check it against, so
this is where that mismatch surfaces, on the first call, not at `make-early-stopping-watcher'
time, which would mean predicting what the library was about to report.

A NIL entry value -- Phase 3a's way of recording a field the backend could not read as a
real, rather than dropping it -- counts as no improvement: it cannot be compared against
WATCHER's `watcher-best-score', and treating it as an improvement would extend a run on a
value nobody can actually read. A real value only counts as an improvement when it is
strictly better than `watcher-best-score' by WATCHER's `watcher-direction' -- or when
`watcher-best-score' is still NIL, i.e. this is the first real value the watcher has ever
seen, which is unconditionally its best so far. Either way, an improvement updates
`watcher-best-iteration' and `watcher-best-score' and resets `watcher-since-improvement'
to zero; anything else -- a NIL value, or a real one that does not improve -- increments
it instead.

`watcher-stopped-p' latches true, and this returns true, once `watcher-since-improvement'
reaches `watcher-rounds' -- ROUNDS consecutive iterations, including this one, with no
improvement. It stays true on every later call: a watcher that has already recommended
stopping has nothing new to say."
  (let ((entry (%find-watched-entry watcher entries)))
    (unless entry
      (error 'unsupported-argument
             :backend :early-stopping
             :argument "make-early-stopping-watcher's :metric"
             :reason (format nil "the booster never reported metric ~S for dataset index ~D"
                              (watcher-metric watcher) (watcher-index watcher))))
    (let ((value (third entry)))
      (if (and value
               (or (null (watcher-best-score watcher))
                   (funcall (%comparator watcher) value (watcher-best-score watcher))))
          (setf (watcher-best-score watcher) value
                (watcher-best-iteration watcher) iteration
                (watcher-since-improvement watcher) 0)
          (incf (watcher-since-improvement watcher))))
    (when (>= (watcher-since-improvement watcher) (watcher-rounds watcher))
      (setf (watcher-stopped-p watcher) t))
    (watcher-stopped-p watcher)))
