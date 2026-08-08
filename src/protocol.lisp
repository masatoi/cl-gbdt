;;;; protocol.lisp --- Generic functions of the unified API.
;;;;
;;;; Declarations only. Methods are added by the individual backend systems.
;;;;
;;;; Both backends take string key/value parameters, so a plist is accepted and
;;;; stringified. Backend-specific parameters pass through untouched; differences
;;;; between the two training algorithms are not forced into one abstraction.

(uiop:define-package #:cl-gbdt/src/protocol
  (:use #:cl)
  (:import-from #:alexandria #:parse-body)
  (:import-from #:cl-gbdt/src/backend #:backend)
  (:export #:make-dataset
           #:dataset-num-rows
           #:dataset-num-features
           #:train
           #:update-one-iteration
           #:predict
           #:save-model
           #:load-model
           #:model-to-string
           #:feature-importance
           #:evaluation
           #:free-dataset
           #:free-booster
           #:with-dataset
           #:with-booster))

(in-package #:cl-gbdt/src/protocol)

(defgeneric make-dataset (backend matrix
                           &key label weight group feature-names parameters reference)
  (:documentation "Build a training dataset for BACKEND from MATRIX.

MATRIX is anything `with-foreign-matrix' accepts. LABEL is the target vector, WEIGHT
the per-sample weights, GROUP the group sizes for ranking, and FEATURE-NAMES a list
of feature name strings. PARAMETERS is a plist passed through to the backend.

REFERENCE, when supplied, is an existing dataset from the same backend whose bin mapper
the new dataset aligns to instead of computing its own. A validation dataset destined
for `train''s :VALID-SETS must be built with the training dataset as its REFERENCE, or
the backend will refuse to attach it: two independently-binned datasets are not
comparable, and LightGBM, for example, rejects LGBM_BoosterAddValidData outright when
the bin mappers differ.

Free the result with `free-dataset' or wrap it in `with-dataset'."))

(defgeneric dataset-num-rows (dataset)
  (:documentation "Return the number of rows in DATASET."))

(defgeneric dataset-num-features (dataset)
  (:documentation "Return the number of features in DATASET."))

(defgeneric train (backend dataset
                    &key valid-sets num-rounds parameters record-history early-stopping)
  (:documentation "Train a BACKEND model on DATASET and return two values: a booster and
a `training-report' of the run.

VALID-SETS is a list of validation sets, NUM-ROUNDS the number of boosting iterations,
and PARAMETERS a plist passed through to the backend. Each VALID-SETS element is either
a dataset, whose validation set gets no name, or a (NAME . DATASET) cons, where NAME is
a string that becomes that dataset's `training-series-name' in the report below; the two
forms may be freely mixed in one list. Two entries may legitimately share one NAME --
their index, not their name, is what tells them apart in the report, so this is accepted
rather than rejected as a duplicate. A cons whose car is not a string signals
`unsupported-argument' naming :VALID-SETS and the offending element; a cons whose cdr is
not this backend's own kind of dataset signals `wrong-backend-reference', the same
condition a bare wrong-backend dataset already signals. Both are checked before any
foreign call.

Free the booster with `free-booster' or wrap it in `with-booster'. `with-booster' binds
the primary value only, so a caller who wants the report has to use `multiple-value-bind'
and free the booster itself.

RECORD-HISTORY, T by default, decides whether the run is recorded at all. A caller who
ignores the secondary value receives the same booster, and the same conditions, either
way -- but not at the same price. With RECORD-HISTORY true, `train' reads the whole
evaluation once per iteration, for every dataset the booster holds, and that read is not
free: it measurably lengthens `train'. Measured over 500 rounds on 2000 rows x 20 columns
with two metrics configured, recording roughly DOUBLED LightGBM's wall-clock `train' time,
with and without a validation set, and added roughly 70-80% to XGBoost's with one
validation set; XGBoost with none stayed inside the measurement noise, that backend
evaluating every dataset in a single call rather than one call each. Those are orders of
magnitude on one machine, not precise figures, and they grow with how many datasets and
metrics there are to read.

RECORD-HISTORY NIL performs no evaluation read at all, so `train' costs what it cost
before it recorded anything. It still returns a `training-report' as its secondary value --
never NIL, so a caller destructuring two values never has to handle two shapes -- whose
`training-report-series' is empty and whose `training-report-num-rounds' is the run's
length, exactly as a run with no metric configured reports.

Recording also decides, on XGBoost, which :VALID-SETS entries `train' accepts at all. A
dataset the backend's own evaluation path cannot evaluate -- an unlabelled DMatrix is the
case this was found through, which `XGBoosterEvalOneIter' rejects while
`XGBoosterUpdateOneIter' trains on it happily -- now fails `train' itself with
`foreign-call-error', where before it trained normally and failed only a later `evaluation'
call. This is general rather than specific to that one input: any configuration whose
evaluation path errors while its update path does not now fails the whole run. Pass
RECORD-HISTORY NIL to train such a configuration, which is the behaviour a caller had
before `train' recorded anything.

EARLY-STOPPING, NIL by default, ends the run once a watched metric stops improving. It is
a plist and all four of its keys are required:

  :METRIC     a string, the metric to watch, spelled the way this backend spells it in
              `evaluation' -- LightGBM's \"binary_logloss\", XGBoost's \"logloss\".
  :DATASET    which dataset's copy of that metric to watch: a string naming a :VALID-SETS
              entry, or an integer index, 0 being the training set and N+1 the Nth
              :VALID-SETS entry. A name matching two entries signals `unsupported-argument'
              -- two entries may share a name, but a watcher has to watch exactly one, so
              pass the index to say which.
  :DIRECTION  `:lower-is-better' or `:higher-is-better'.
  :ROUNDS     a positive integer: how many consecutive iterations may fail to improve on
              the best value seen so far before the run stops.

Anything else -- a key missing, a metric that is not a string, :ROUNDS zero -- signals
`unsupported-argument' before the first iteration runs. So does a :METRIC this booster
never reports, at the end of the first iteration, which is the first moment there is a
real evaluation to check the name against.

:DIRECTION is required, and is not inferred, because neither library exposes it: nothing
in either C API says whether a metric improves upward or downward, and deciding it from
the metric's NAME would be a guess with a lookup table in front of it. Improvement is
strict -- a value equal to the best seen so far is not an improvement and counts toward
:ROUNDS -- and a value the backend reported but could not be read as a real counts as no
improvement either, since nothing can be compared against it.

EARLY-STOPPING together with RECORD-HISTORY NIL signals `unsupported-argument': early
stopping needs the watched series, and reading the evaluation costs the same whether one
series is watched or every series is recorded, so there is no cheaper middle path to
offer. Ask for early stopping and the history comes with it.

A run given EARLY-STOPPING fills `training-report-best-iteration', `-best-score' and
`-early-stopped-p', and the returned booster's `booster-best-iteration'. A run not given
it leaves all four NIL -- see below.

The secondary value is a `training-report'. Its `training-report-series' is a list of
`training-series', one per metric per dataset -- the same (DATASET-INDEX, METRIC-NAME)
pairs `evaluation' reports for the trained booster, in the same order, so a series can be
found by the index and metric name a caller already knows. Each series carries
`training-series-values', one element per completed iteration in order: a `double-float',
or NIL where the backend reported a value that could not be read as a real. The last
element of a series is what `evaluation' answers for that pair immediately after `train'
returns; the earlier ones are what it would have answered at each earlier iteration, and
are the only way to see them, since a trained booster no longer remembers them.

`training-report-num-rounds' is how many iterations actually ran, which is NUM-ROUNDS
unless EARLY-STOPPING cut the run short, and every series is exactly that long.

`training-report-best-iteration', `-best-score' and `-early-stopped-p' are filled only
when EARLY-STOPPING was supplied, and are NIL otherwise -- NIL meaning \"not determined\",
never \"iteration 0\". With it, BEST-ITERATION is the 1-based iteration at which the
watched series reached its best value, BEST-SCORE is that value, and EARLY-STOPPED-P says
whether the run ended because of the watcher rather than by reaching NUM-ROUNDS. All three
are filled whether or not the run was actually stopped: a watcher that never fired still
knows which iteration was best. The booster's own `booster-best-iteration' carries the
same iteration as the report's.

`training-report-series' is empty when the booster has no metric configured at all --
LightGBM's `metric=none', XGBoost's `disable_default_eval_metric=1' -- and when
RECORD-HISTORY is NIL. An empty series list is not an error and says nothing about whether
training succeeded; NUM-ROUNDS iterations still ran, and `training-report-num-rounds' still
says so.

Every series carries `training-series-index'; a series carries a non-NIL
`training-series-name' only for a dataset named through :VALID-SETS. The training set is
never a :VALID-SETS entry and so is always index 0 with a NIL name, and a :VALID-SETS
entry passed bare is NIL too -- nothing here invents a name for either, the same way
`evaluation' invents no name for the index it reports."))

(defgeneric update-one-iteration (booster)
  (:documentation "Advance BOOSTER by one boosting iteration.

Use this to drive the training loop yourself. Returns false when no further split was
possible and the backend can report that -- LightGBM does. XGBoost's booster protocol
has no equivalent signal, so its `update-one-iteration' always returns true after a
successful call; treat a true return as \"the call succeeded\", not as proof a split
happened, unless the backend is known to be LightGBM."))

(defgeneric predict (booster matrix &key kind num-iteration)
  (:documentation "Predict on MATRIX using BOOSTER.

KIND is `:normal' (default, transformed predictions), `:raw' (raw scores),
`:leaf-index' (leaf indices) or `:contrib' (feature contributions). NUM-ITERATION
limits how many trees are used; nil uses all of them.

Returns a `(simple-array double-float (* *))'."))

(defgeneric save-model (booster path &key num-iteration)
  (:documentation "Save BOOSTER's model to PATH.

NUM-ITERATION limits how many boosted rounds are saved on LightGBM, nil meaning all of
them. XGBoost has no such limit -- `XGBoosterSaveModel' always saves every round -- so
supplying NUM-ITERATION there signals `unsupported-argument' instead of being silently
ignored."))

(defgeneric load-model (backend path)
  (:documentation "Load a model from PATH and return a BACKEND booster."))

(defgeneric model-to-string (booster &key num-iteration)
  (:documentation "Return BOOSTER's model as a string.

NUM-ITERATION behaves as it does for `save-model': LightGBM honors it, nil meaning
every round; XGBoost has no iteration-limited variant of this call and signals
`unsupported-argument' when NUM-ITERATION is supplied."))

(defgeneric feature-importance (booster &key kind num-iteration)
  (:documentation "Return BOOSTER's feature importances as `(simple-array double-float (*))'.

KIND is `:split' (how often a feature was used to split) or `:gain' (total gain). The
result has one entry per feature, in column order, zero for a feature never used in a
split. LightGBM's own C call is already dense; XGBoost's reports only features actually
used, identified by name rather than column, so this backend's method reconstructs the
dense, per-column result from that.

NUM-ITERATION behaves as it does for `save-model': LightGBM limits the importance
calculation to that many rounds, nil meaning all of them; XGBoost has no such limit and
signals `unsupported-argument' when NUM-ITERATION is supplied.

Every result is one-dimensional -- one number per feature, full stop. XGBoost's
`gblinear' booster reports a per-class matrix instead of a single score per feature for
a multi-class model, which has no defined single-value reduction (summing signed linear
coefficients across classes can cancel a feature that matters to none near zero); rather
than invent one, that backend signals `unsupported-argument' instead of returning
anything for that combination. LightGBM's own call never reports that shape: it already
aggregates a multi-class model's per-class contributions into one number per feature
inside the library."))

(defgeneric evaluation (booster)
  (:documentation "Return BOOSTER's evaluation metrics as a fresh list of
(DATASET-INDEX METRIC-NAME VALUE) lists -- one entry per metric per dataset.

DATASET-INDEX identifies the dataset a value was computed on by its position among the
datasets BOOSTER retains: 0 is the training set BOOSTER was trained on, 1 is the first
entry of `train''s :VALID-SETS, 2 the second, and so on -- the order the caller supplied
them in, which is also LightGBM's own `data_idx' numbering. No name is invented for any
of them: LightGBM identifies a validation set by index and by nothing else, so there is
no upstream name to report and this API does not fabricate one.

METRIC-NAME is the backend's own name for the metric, exactly as that backend spells it.
LightGBM's \"binary_logloss\" and XGBoost's \"logloss\" name related quantities under
different names, and neither is translated into the other's vocabulary. Which metrics a
booster has at all is decided by what `train' was given (LightGBM's `metric', XGBoost's
`eval_metric') and by the objective's own default, not by this call.

VALUE is a `double-float', or NIL when the backend reported a value this library could
not read as one. Only XGBoost produces NIL: its values arrive as formatted text (see
:VALUE-SOURCE below) and its `std::ostream' spells a non-finite double \"inf\" or
\"nan\", which is not `double-float' syntax. LightGBM's values are already doubles and
are returned unmodified, a non-finite one included.

Entries are ordered by DATASET-INDEX, and within one dataset in the backend's own metric
order -- the same metrics, in the same order, for every dataset.

The result is empty when BOOSTER has no metrics configured (LightGBM's `metric=none') and
when BOOSTER retains no dataset to evaluate at all -- a `load-model' booster, which has
neither a training set nor validation sets.

The secondary value is a plist saying where the numbers came from, because the two
backends do not produce them the same way and a caller must not have to guess which kind
it is holding:

  :VALUE-SOURCE  `:library-doubles' when the backend handed over binary doubles
                 (LightGBM's `LGBM_BoosterGetEval'), or `:parsed-text' when this library
                 read them out of a string the backend formatted (XGBoost's
                 `XGBoosterEvalOneIter'). A `:parsed-text' number is this library's
                 reading of a text format upstream does not document as stable, not a
                 number the backend itself reported.
  :RAW           Present for `:parsed-text' only: the exact string the backend produced,
                 unmodified, so the parse loses nothing upstream said.
                 `cl-gbdt/xgboost:evaluate-one-iteration' returns that same string directly.

Signals `released-handle-error' when BOOSTER, or any dataset it retains, has already been
freed -- both backends read a retained validation set's own memory while evaluating, so
this is checked before any foreign call rather than left to crash -- and
`backend-not-open' when BOOSTER's backend has since been closed."))

(defgeneric free-dataset (dataset)
  (:documentation "Free DATASET. Does nothing if it was already freed."))

(defgeneric free-booster (booster)
  (:documentation "Free BOOSTER. Does nothing if it was already freed."))

(defmacro with-dataset ((var form) &body body)
  "Bind VAR to the dataset FORM returns, evaluate BODY, and always free it.

Explicit resource management is the first-class pattern; finalizers are only a
safety net, and that net reports and does not free: running the C free from
whatever thread the garbage collector chose would give no ordering guarantee
between a booster and the dataset it holds, and `with-booster' nested inside
this macro exists precisely to guarantee that order.

Declarations at the head of BODY are moved onto a fresh binding of VAR that
shadows the one FORM's value is stored in, scoped to BODY alone. Splicing them
onto the outer binding instead -- the one `unwind-protect''s cleanup clause also
reads to call `free-dataset' -- would put an `(ignore VAR)' declaration from
BODY in the same scope as that read, which SBCL flags as \"reading an ignored
variable\" (verified empirically; do not simplify this back to `progn' or a
single binding)."
  (multiple-value-bind (forms declarations) (alexandria:parse-body body)
    `(let ((,var ,form))
       (unwind-protect
            (let ((,var ,var))
              ,@declarations
              ,@forms)
         (free-dataset ,var)))))

(defmacro with-booster ((var form) &body body)
  "Bind VAR to the booster FORM returns, evaluate BODY, and always free it.

A booster holds a strong reference to the dataset it was trained on, so nesting this
inside `with-dataset' cannot invert the release order. Declarations at the head of
BODY are shadow-bound as in `with-dataset', for the same reason."
  (multiple-value-bind (forms declarations) (alexandria:parse-body body)
    `(let ((,var ,form))
       (unwind-protect
            (let ((,var ,var))
              ,@declarations
              ,@forms)
         (free-booster ,var)))))
