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
  (:import-from #:cl-gbdt/src/backend #:backend #:backend-name #:close-backend)
  (:import-from #:cl-gbdt/src/handle #:dataset #:booster #:handle-backend)
  (:import-from #:cl-gbdt/src/conditions #:backend-methods-not-loaded)
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
           #:with-booster
           #:with-backend))

(in-package #:cl-gbdt/src/protocol)

(defgeneric make-dataset (backend matrix
                           &key label weight group feature-names parameters reference
                             missing categorical-features)
  (:documentation "Build a training dataset for BACKEND from MATRIX.

MATRIX is either a dense matrix -- anything `with-foreign-matrix' accepts -- or a
`csr-matrix', the sparse compressed-sparse-row form `make-csr-matrix' builds. LABEL is the
target vector, WEIGHT the per-sample weights, GROUP the group sizes for ranking, and
FEATURE-NAMES a list of feature name strings. PARAMETERS is a plist passed through to the
backend.

A `csr-matrix' needs BACKEND's `:sparse-input' capability, which `make-dataset' re-checks
itself: a caller who never asked `backend-supports-p' gets `capability-unavailable' rather
than a missing-symbol crash, and no backend ever falls back to converting the matrix to a
dense one instead. Every other argument means exactly what it means for a dense matrix,
including PARAMETERS and REFERENCE -- a backend that refuses one of them refuses it either
way, and a backend that honours it honours it either way. The resulting dataset is an
ordinary dataset: nothing downstream of here, `train' included, can tell which form built
it.

The dataset's feature count is the `csr-matrix''s own NUM-COLUMNS, its declared width --
not the largest column index it happens to store. The two are different facts, and only
the caller knows the first; see `make-csr-matrix''s docstring for why it is required rather
than inferred.

What a `csr-matrix' MEANS, however, is not the same on both backends when it omits entries:
LightGBM reads an absent entry as `0.0' and XGBoost reads one as missing, so the dataset
built here is only the same data on both when every element is stored. Nothing signals when
it is not -- the trained numbers simply differ. See the `csr-matrix' struct's own docstring,
where that divergence is stated as the property of the value it is.

FEATURE-NAMES must be a proper list; a dotted list, a circular one, or a value that is no list
at all signals `unsupported-argument' naming `:feature-names'. `listp' is true of a dotted
list, so the shape has to be checked before the list is traversed: on SBCL 2.6.7 `(length
'(\"a\" . \"b\"))' signals a raw `type-error' and `(length circular)' does not return, both
measured. NIL, the default, is a proper list and means no names. Only the list's SHAPE is
checked, though: a proper list holding something other than a string is not caught here, and
fails later instead, in the foreign-string allocation that writes each name across, with a raw
`type-error' naming the value rather than `unsupported-argument'.

MISSING names the value in MATRIX that means *missing* -- the datum a caller wrote in place
of one they do not have, such as the -999.0 a CSV convention often uses. It is a VALUE, not
a policy: it says which number means missing and does not turn missing handling on or off,
nor make zero mean missing. A backend's own flags for those -- LightGBM's `use_missing' and
`zero_as_missing' -- stay where they are, reachable through PARAMETERS.

MISSING is a `real' or NIL, and anything else signals `unsupported-argument' naming
`:missing'. NIL, the default, is the backend's own default sentinel and is exactly what
every call got before this argument existed: a caller who passes nothing gets the same
dataset, and the same trained numbers, as before.

A non-NIL MISSING needs BACKEND's `:missing-value' capability, which `make-dataset'
re-checks itself: a caller who never asked `backend-supports-p' gets
`capability-unavailable' rather than an argument silently ignored, and no backend ever falls
back to its own sentinel instead. XGBoost provides the capability -- the sentinel is a key
in the config JSON its dataset-creation entry points read. LightGBM does not, and signals for
ANY non-NIL value including a NaN it would in fact honour: its C API has no missing-value key
at all, and a capability whose answer depended on which value was passed could not be stated
by `backend-supports-p'.

Measured, and worth knowing before choosing a sentinel: XGBoost compares MISSING against the
data at SINGLE precision, whatever MATRIX's own element type. Two `double-float's that round
to the same `single-float' therefore both count as missing -- probed against the vendored
library, the sentinel 16777217.0 matches the datum 16777216.0d0, a different double sharing
its float32, and does not match 16777224.0d0, which is a different float32.

CATEGORICAL-FEATURES is a list of 0-based column indices naming which columns of MATRIX hold
CATEGORIES rather than quantities. A category ordinal is a label, not a magnitude: a split on
a quantitative column can only ask whether the value is above some threshold, which is a
question about an order the ordinals do not have, while a split on a categorical one asks
which SUBSET of the categories a row falls in. The distinction is invisible in the data --
both are numbers in the same matrix -- so it is stated here or not at all. The order of the
list is the caller's and carries no meaning; the same column named twice is an error rather
than a duplicate to collapse.

NIL, the default, is the backend's own default -- every column read as a quantity -- and is
exactly what every call got before this argument existed. An index that is not an integer, is
negative, is beyond MATRIX's last column, or was named twice signals `unsupported-argument'
naming `:categorical-features'. The column count that range check is made against is MATRIX's
own, and MATRIX may be a `csr-matrix', whose DECLARED width it is: CATEGORICAL-FEATURES means
the same thing for either form.

A non-NIL CATEGORICAL-FEATURES needs BACKEND's `:categorical-features' capability, which
`make-dataset' re-checks itself: a caller who never asked `backend-supports-p' gets
`capability-unavailable' rather than an argument silently ignored, and no backend ever falls
back to reading the column as a quantity instead. XGBoost provides it -- the column list
becomes the `\"feature_type\"' field it attaches to a finished DMatrix. LightGBM provides it
through a different mechanism: the list becomes a `categorical_feature' entry in the
parameter string that builds the dataset.

That is the same channel PARAMETERS itself reaches LightGBM through, and so the one place
where naming CATEGORICAL-FEATURES changes what PARAMETERS may say. Supplying BOTH signals
`unsupported-argument' there, for `categorical_feature' and the four further spellings that
backend honours -- `cat_feature', `categorical_column', `cat_column' and
`categorical_features' -- because the entry `make-dataset' writes lands after the caller's
own and LightGBM keeps the first, so it is CATEGORICAL-FEATURES that would be silently
discarded.

PARAMETERS on its own is unaffected. A caller who names no categorical column and writes one
of those keys by hand is using PARAMETERS as the pass-through it has always been, and gets
exactly the dataset they got before CATEGORICAL-FEATURES existed. Every other key passes
through untouched either way, and no other backend refuses anything on this account.

`predict' takes no CATEGORICAL-FEATURES of its own, and deliberately: measured, a booster
trained from a dataset built with one predicts correctly from a plain matrix, the trained
trees carrying the category sets they split on rather than re-deriving them from what they are
asked about. Naming the columns again at prediction time would be a second place for the same
statement to be wrong.

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
                    &key valid-sets num-rounds parameters record-history early-stopping
                         objective evaluation)
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

A run given EARLY-STOPPING usually fills `training-report-best-iteration', `-best-score'
and `-early-stopped-p', and the returned booster's `booster-best-iteration' -- but not
always; see below for the two cases where they stay NIL regardless. A run not given
EARLY-STOPPING at all always leaves all four NIL.

OBJECTIVE, NIL by default, is a function that supplies the gradient and Hessian itself,
so the run boosts against the caller's own loss rather than the library's. It requires
the `:custom-objective' capability, which `train' re-checks and signals
`capability-unavailable' for. A non-NIL OBJECTIVE that is not a `function' -- a number, a
string, or a SYMBOL naming one -- signals `unsupported-argument' naming :OBJECTIVE, before
any foreign call and so before a booster exists to leak.

It is called once per iteration, before that iteration's update, with one argument: the
booster's current raw scores for its training set, as a (ROWS GROUPS) `double-float'
array -- the margin, before any sigmoid or softmax, and the same shape and element type
`predict' returns. GROUPS is 1 for regression and binary classification and `num_class'
for multiclass. It must return two values, the gradient and the Hessian, each a (ROWS
GROUPS) array. The SHAPE is what is checked: a wrong rank, wrong dimensions, or one value
where two were required signals `dimension-mismatch' before any foreign call. The element
type is not: `double-float', `single-float' and a general array whose elements are reals --
what `(make-array (list ROWS GROUPS))' gives, and the most natural thing to write -- are all
accepted and all train the same model, each element being coerced where the buffer is
written. An element that is not a real signals `unsupported-element-type' naming its type,
there at the write, before the library has been called.

The caller writes one array shape and it means the same thing on both backends: the two
libraries flatten it differently -- LightGBM group-major, XGBoost row-major -- and each
backend's own code absorbs that.

A handle the objective frees, or a backend it closes, is caught the moment it returns and
before any further foreign call: `train' re-runs its own dataset and backend checks there,
so freeing the training set from inside an objective signals `released-handle-error' rather
than faulting the process.

On LightGBM, OBJECTIVE **overrides** any `objective' in PARAMETERS, forcing it to
\"none\". LightGBM refuses a custom update while the booster holds an objective function
at all, so the combination the override replaces cannot run; every other parameter,
`num_class' included, passes through untouched. XGBoost's parameters are never rewritten,
and its configured objective goes on transforming predictions -- so with
`binary:logistic' still set there, `predict :kind :normal' returns probabilities while
LightGBM's returns the raw score. One custom-objective run, two meanings for :NORMAL.

A library metric configured through PARAMETERS relates to the library's objective, not to
the caller's, so what RECORD-HISTORY records and what EARLY-STOPPING watches may be
meaningless under a custom objective. Nothing signals; the caller decides.

The function runs inside `train''s foreign-float-trap mask, so the caller's own Lisp
arithmetic does not trap: `(/ 1.0d0 0.0d0)' yields infinity rather than signalling, on
x86-64 as well as on aarch64. The mask is what makes the two platforms agree here, and it
agrees on the masked convention the two C libraries are written against.

EVALUATION, NIL by default, is a function that turns one dataset's current predictions into
a named metric value, so the run records the caller's own measure of fit beside the
library's. It requires the `:custom-evaluation' capability, which `train' re-checks and
signals `capability-unavailable' for. BOTH backends provide it, and the re-check is not
thereby decorative: it is what a caller who withdrew the capability, or who moved to a
library lacking what LightGBM's read needs, meets instead of a run that quietly recorded the
library's metrics alone.

It also signals `unsupported-argument' naming :EVALUATION, on either backend, for a non-NIL
EVALUATION combined with RECORD-HISTORY NIL -- a custom metric's whole result is the
per-iteration series RECORD-HISTORY NIL exists not to build -- and for an EVALUATION that is
not a `function'. A SYMBOL naming a function is refused with the rest: `funcall' would resolve
it afresh at each iteration against whatever global definition was then in force, rather than
against what the caller passed. All three checks run before any foreign call.

It is called ONCE PER DATASET PER ITERATION, after that iteration's update, with two
arguments: SCORES, that dataset's current predictions as a (ROWS GROUPS) `double-float'
array, and the dataset's INDEX. It must return two values, a metric NAME -- a string -- and
a VALUE, a real number or NIL. A NAME that is not a string, or a VALUE that is neither,
signals `unsupported-argument' naming :EVALUATION.

A REAL VALUE IS RECORDED AS A `double-float', coerced when the entry is built rather than
stored as returned: `training-series-values' documents every element of every series as a
`double-float' or NIL, and a caller returning 1/3 reads 0.3333333333333333d0 back out of its
own series. A real too large for a `double-float' to hold records the signed infinity, on
every platform alike rather than signalling on the ones whose `:overflow' trap is enabled.
The name is COPIED at the same point, so a caller free to return one string object per
iteration and rewrite its characters cannot change what a completed run recorded, nor what
the pin below compares against.

INDEX numbers the datasets exactly as EARLY-STOPPING's :DATASET and the report's
`training-series-index' do: 0 is the training set, and N+1 is the Nth :VALID-SETS entry.
ROWS is THAT dataset's own row count -- a validation set shorter than the training set is
handed its own shorter array, not a padded or truncated view of the training set's.

SCORES is what `predict :kind :normal' returns for that dataset, and NOT the margin
OBJECTIVE is handed: with a classification objective configured, these are the transformed
probabilities. Measured on EACH backend separately, for a training set and for a validation
set alike, and the measurement means something different on each: on LightGBM SCORES comes
from a cached-prediction entry point, so agreeing with `predict' says two different C
functions agree; on XGBoost SCORES is itself a prediction pass over the dataset's own DMatrix,
so agreeing says that DMatrix and a fresh one built from the same rows answer alike. Neither
backend's figure stands in for the other's and the two are not compared.

Under a custom OBJECTIVE the two backends then part company, and it is the divergence
OBJECTIVE's own paragraphs above already describe rather than a new one. On LightGBM
`objective=none' leaves no transform to apply, so :NORMAL and :RAW become the same numbers and
SCORES coincides with what OBJECTIVE was handed. On XGBoost nothing is rewritten, so a
configured `binary:logistic' goes on transforming and the two stay apart: one run in which
EVALUATION reads probabilities while OBJECTIVE reads the margin behind them.

A NIL VALUE means \"not computable this iteration\" -- a fold whose denominator was zero,
a metric undefined before some minimum number of rows has a prediction. It is recorded in
its place in the series rather than dropped, exactly as a value the backend itself could not
report is, and it counts as NO IMPROVEMENT to an EARLY-STOPPING watcher, which cannot
compare against something it cannot read.

The values join the secondary value as SERIES OF THEIR OWN, one per (INDEX, NAME) pair,
indistinguishable in shape from the library's own -- so `training-report-series' answers
for them, and EARLY-STOPPING can watch one by giving :METRIC the name the function returns
and :DATASET the index it was returned for. Nothing else is needed to make a custom metric
watchable.

The library's own series remain exactly what they were, and come FIRST: EVALUATION's
entries are appended after every library entry of the same iteration, so the pairs
`evaluation' reports for the trained booster are a PREFIX of `training-report-series', in
the same order, and a caller who could find a library series before can still find it the
same way. `evaluation' itself never reports a custom metric: it asks the library what the
library computed, and the library never computed this one.

EVALUATION together with RECORD-HISTORY NIL signals `unsupported-argument': a custom
metric's whole result is the per-iteration series RECORD-HISTORY NIL exists not to build, so
the values would be computed at full cost and then dropped. This is the same pair, and the
same refusal, EARLY-STOPPING and RECORD-HISTORY NIL already make.

A NAME colliding with one the library itself reports for the SAME index signals
`unsupported-argument' too, at the end of the FIRST iteration -- the first moment there is a
real evaluation to compare the name against, exactly as a :METRIC no booster reports is
caught there. The pair (INDEX, NAME) is what a series is keyed by, so two different
quantities under one pair would corrupt the series rather than produce two. The same NAME at
a DIFFERENT index does not collide, and neither does a name the library does not report at
all -- what is checked is what this booster actually reported, not a list of well-known
metric names.

EVALUATION must return THE SAME NAME FOR A GIVEN INDEX ON EVERY ITERATION OF THE RUN. Two
indices may return two different names, and usually will not; what is refused is one index's
name changing between iterations. The first iteration's name is remembered per index, and any
later iteration returning a different one for that index signals `unsupported-argument'
naming :EVALUATION, mid-run, exactly where the differing name was returned. This is a
requirement rather than a convenience: a series holds one value per completed iteration, and
a name that varies asks for something no series can be -- each name it took would be pushed
only on the iterations it appeared in, giving several series all shorter than the run and
each misaligned with the iterations its values were measured at. Varying INTO the library's
own name for that index is worse still and is the case the collision check above cannot
reach, since that check runs on the first iteration only: from the iteration the two names
meet, one key would collect two values per iteration and its series would come out LONGER
than `training-report-num-rounds' says the run was. Pinning the name is what keeps \"every
series is exactly that long\", below, true of a caller's own series as well as the library's.
Returning ONE STRING OBJECT and rewriting it in place is refused on the same terms, and is not
a case the pin could have caught by itself: what is pinned, and what every recorded entry
holds, is a COPY taken at the moment the name was returned, so the comparison is against what
that iteration actually said rather than against an object the caller can still edit.

The function runs inside `train''s foreign-float-trap mask, on the same terms OBJECTIVE
does: the caller's own arithmetic does not trap, so `(/ 1.0d0 0.0d0)' yields infinity rather
than signalling `division-by-zero' on x86-64 as well as on aarch64. A handle it frees, or a
backend it closes, is caught the moment it returns and before the next dataset is read, the
same way OBJECTIVE's is.

The secondary value is a `training-report'. Its `training-report-series' is a list of
`training-series', one per metric per dataset -- the same (DATASET-INDEX, METRIC-NAME)
pairs `evaluation' reports for the trained booster, in the same order, so a series can be
found by the index and metric name a caller already knows. A run given EVALUATION carries
that function's own pairs as well, appended after all of those, which is why the
`evaluation' pairs are described above as a PREFIX of this list rather than the whole of it.

Each series carries `training-series-values', one element per completed iteration in order:
a `double-float', or NIL where the backend reported a value that could not be read as a real.
The last element of a series is what `evaluation' answers for that pair immediately after
`train' returns; the earlier ones are what it would have answered at each earlier iteration,
and are the only way to see them, since a trained booster no longer remembers them.

`training-report-num-rounds' is how many iterations actually ran, which is NUM-ROUNDS
unless EARLY-STOPPING cut the run short, and every series is exactly that long.

`training-report-best-iteration', `-best-score' and `-early-stopped-p' stay NIL when
EARLY-STOPPING was not supplied at all -- NIL meaning \"not determined\", never \"iteration
0\". When it was supplied, BEST-ITERATION is the 1-based iteration at which the watched
series reached its best real value, BEST-SCORE is that value, and EARLY-STOPPED-P says
whether the run ended because of the watcher rather than by reaching NUM-ROUNDS -- but
supplying EARLY-STOPPING does not by itself guarantee BEST-ITERATION or BEST-SCORE comes
back non-NIL:

  - NUM-ROUNDS zero or negative runs no iteration at all, so the watcher is never
    consulted and all three slots -- and the booster's `booster-best-iteration' -- stay
    NIL exactly as if EARLY-STOPPING had not been given.
  - A run every one of whose watched values was unreadable -- see EVALUATION's account of
    a value the backend reported but this library could not parse as a real -- never has a
    real value to call best, so BEST-ITERATION and BEST-SCORE stay NIL even though
    EARLY-STOPPED-P can still become T once ROUNDS consecutive unreadable iterations pass:
    an unreadable value counts toward ROUNDS the same as a non-improving real one, but
    never sets a best.

Outside those two cases, all three are filled whether or not the run was actually
stopped: a watcher that never fired still knows which iteration was best. The booster's
own `booster-best-iteration' carries the same iteration as the report's BEST-ITERATION,
staying NIL together with it in both cases above too -- this is what `predict',
`save-model' and `model-to-string''s NUM-ITERATION :BEST resolve against, and why they
signal `unsupported-argument' rather than assume a default when it is NIL.

`training-report-series' is empty when the booster has no metric configured at all --
LightGBM's `metric=none', XGBoost's `disable_default_eval_metric=1' -- and when
RECORD-HISTORY is NIL. An empty series list is not an error and says nothing about whether
training succeeded; NUM-ROUNDS iterations still ran, and `training-report-num-rounds' still
says so. A run given EVALUATION on a booster with no metric configured is the one case where
the first of those two no longer empties the list: the library contributes nothing and the
caller's function contributes every series there is. RECORD-HISTORY NIL cannot combine with
EVALUATION at all, so the second case is unchanged.

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

(defgeneric predict (booster matrix &key kind num-iteration missing)
  (:documentation "Predict on MATRIX using BOOSTER.

MATRIX is either a dense matrix -- a 2D array of `double-float' or `single-float', one row
per observation -- or a `csr-matrix' describing the same rows sparsely. A `csr-matrix'
requires the `:sparse-input' capability, which `predict' re-checks itself and signals
`capability-unavailable' for rather than falling back to a dense conversion; see
`backend-supports-p'. Its NUM-COLUMNS must be BOOSTER's own feature count, and when it is
not, the failure is the backend library's to report -- `foreign-call-error', in each
library's own words -- since neither this function nor the backends pre-empt a consistency
check the library already makes.

An entry the `csr-matrix' omits does not mean the same thing to the two libraries -- `0.0'
to LightGBM, missing to XGBoost -- so the rows predicted on here are only the same rows on
both backends when every element is stored. Nothing signals when they are not; the numbers
returned simply differ. See the `csr-matrix' struct's own docstring, where that divergence
is stated as the property of the value it is.

KIND is `:normal' (default, transformed predictions), `:raw' (raw scores),
`:leaf-index' (leaf indices) or `:contrib' (feature contributions). All four are available
for a dense MATRIX on both backends, and for a `csr-matrix' on LightGBM. On XGBoost, a
`csr-matrix' supports `:normal' and `:raw' only: its sparse entry point,
`XGBoosterPredictFromCSR', is that library's INPLACE prediction rather than a CSR spelling
of the dense call, and it refuses the other two outright -- so `predict' signals
`foreign-call-error' for them, the library's own refusal passed through rather than an
emulation invented here. The only way to get `:contrib' or `:leaf-index' out of XGBoost for
rows held as a `csr-matrix' is to materialise those rows as a dense matrix -- a 2D
`double-float' or `single-float' array -- and predict on that. Note that this is a real cost,
not a formality: MATRIX is the only thing `predict' takes, and a dataset is not one of its
accepted forms, so building the rows into a dataset with `make-dataset' does not lead to a
prediction at all.

NUM-ITERATION limits
how many trees are used: nil uses all of them, an integer uses that many, and `:best'
resolves to BOOSTER's own `booster-best-iteration' -- the iteration an `:early-stopping'
`train' run judged best -- before either of those. Signals `unsupported-argument' naming
NUM-ITERATION when `:best' is given and `booster-best-iteration' is NIL: BOOSTER was
never trained with `:early-stopping', or that run never determined a best iteration at
all -- see `train''s docstring for when that happens even with `:early-stopping'
supplied. NIL keeps meaning \"every round\" on every booster, including one with a best
iteration to resolve `:best' against; `:best' is an additional accepted value, never a
new default. It means the same thing for a `csr-matrix' as for a dense matrix on both
backends.

MISSING names the value in MATRIX that means *missing*, exactly as `make-dataset''s own
MISSING names it in a training matrix: a `real' or NIL, `unsupported-argument' naming
`:missing' for anything else, and NIL -- the default -- the backend's own sentinel, so a
caller who passes nothing gets the predictions they always got. A non-NIL MISSING needs
BOOSTER's backend's `:missing-value' capability, which `predict' re-checks ITSELF rather
than inheriting any check `make-dataset' made: the two operations check it separately, and
a backend could provide it for one and not the other. It says the same thing about a
`csr-matrix' as about a dense matrix -- a STORED value equal to it is missing -- and says
nothing about an entry a `csr-matrix' omits, which each library goes on reading its own way
as described above.

Nothing ties MISSING here to the MISSING the dataset BOOSTER was trained from was built
with. XGBoost, the backend that provides the capability, does not record a dataset's
sentinel on the booster trained from it, so predicting with a different sentinel than
training used -- or with none -- is a call the library accepts and never reports. Keeping
the two consistent is the caller's responsibility; nothing here detects that they are not.

Returns TWO values.

The FIRST is a `(simple-array double-float (* *))', with one row per input row either way --
a `csr-matrix''s row count is `csr-matrix-num-rows', one less than its INDPTR's length. It is
exactly what it has always been: the second value was added beside it and changed neither its
dimensions nor its elements, for any KIND, dense or sparse.

The SECOND is the SHAPE the backend states for that result -- a list of integers in
`array-dimensions' order -- or NIL where the backend states none. A NIL here is not an error
and nothing signals: the `:prediction-shape' capability (see `backend-supports-p') is what says
whether a backend states one, and NO OPERATION REFUSES ON IT. The three capabilities this
protocol's operations do refuse on -- `:sparse-input', `:missing-value' and
`:categorical-features', each documented above on the operation that checks it -- gate an
ARGUMENT, and refusing is how a caller learns the argument will not be honoured. There is no
argument asking for a shape, so a backend that answers false returns NIL here and predicts
exactly as it otherwise would.

A stated shape describes the same elements the first value holds, and may have MORE axes
than the first value's two. Measured on XGBoost: `:normal' and `:raw' report exactly the first
value's own `array-dimensions'; `:leaf-index' reports rows x iterations x output groups x
parallel trees, and `:contrib' rows x output groups x (features + 1), both of which the first
value folds into rows x (total element count / rows). Both stay multidimensional on a BINARY
model -- (rows 4 1 1) and (rows 1 4) for a four-round model over three features, its single
output group notwithstanding -- so the extra structure is not something only a multiclass
objective produces.

On LightGBM the same second value is DERIVED rather than reported, that library stating no
shape anywhere: `LGBM_BoosterCalcNumPredict' gives an element count and no axes at all.
`:normal' and `:raw' state the first value's own `array-dimensions'; `:contrib' states
rows x output groups x (features + 1), the three axes the element count, the row count and the
booster's feature count determine between them; and `:leaf-index' states NIL, its sub-layout
having no property this project can check and an unchecked ordering being a guess rather than a
measurement. `:contrib''s CLASS-MAJOR ordering is checked, and that is why it is stated at all:
contributions grouped that way sum to their output group's `:raw' score and grouped the other
way do not -- measured against both libraries, and asserted on LightGBM by
`lightgbm-s-derived-contrib-shape-is-the-one-the-numbers-support' in
tests/functional/prediction-shape.lisp, whose feature-major arm is the control. Both backends
therefore answer `:prediction-shape' true, and on this one the keyword says the mechanism is
present rather than that the library said anything."))

(defgeneric save-model (booster path &key num-iteration)
  (:documentation "Save BOOSTER's model to PATH.

NUM-ITERATION limits how many boosted rounds are saved on LightGBM: nil means all of
them, an integer that many, and `:best' resolves to BOOSTER's own `booster-best-iteration'
first -- see `predict' for exactly when that resolution itself signals
`unsupported-argument'. XGBoost has no such limit at all -- `XGBoosterSaveModel' always
saves every round -- so supplying NUM-ITERATION there signals `unsupported-argument'
whether it is an explicit integer or `:best' resolved to one; nothing about `:best' is
special-cased around that check. A caller who wants an XGBoost model file that stops at
the best iteration slices to it first with `cl-gbdt/xgboost:slice-model' and saves the
slice instead."))

(defgeneric load-model (backend path)
  (:documentation "Load a model from PATH and return a BACKEND booster."))

(defgeneric model-to-string (booster &key num-iteration)
  (:documentation "Return BOOSTER's model as a string.

NUM-ITERATION behaves as it does for `save-model', `:best' included: LightGBM honors it,
nil meaning every round; XGBoost has no iteration-limited variant of this call and signals
`unsupported-argument' when NUM-ITERATION is supplied, an explicit integer or `:best'
resolved to one alike."))

(defgeneric feature-importance (booster &key kind num-iteration)
  (:documentation "Return BOOSTER's feature importances as `(simple-array double-float (*))'.

KIND is `:split' (how often a feature was used to split) or `:gain' (total gain). The
result has one entry per feature, in column order, zero for a feature never used in a
split. LightGBM's own C call is already dense; XGBoost's reports only features actually
used, identified by name rather than column, so this backend's method reconstructs the
dense, per-column result from that.

NUM-ITERATION behaves as it does for `save-model' for NIL and an explicit integer:
LightGBM limits the importance calculation to that many rounds, nil meaning all of them;
XGBoost has no such limit and signals `unsupported-argument' when NUM-ITERATION is
supplied. Unlike `predict', `save-model' and `model-to-string', it does NOT accept
`:best' -- both backends signal `unsupported-argument' naming NUM-ITERATION when `:best'
is given, regardless of whether BOOSTER has a best iteration to resolve it against.

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
  (:documentation "Free DATASET. Does nothing if it was already freed.

Freeing the same DATASET from two threads at once can end the process outright
rather than signalling anything catchable -- see `docs/user-guide/threads.md'.

Freeing DATASET while any call is in flight, on another thread, on a booster
that holds it -- as its training set or as one of its validation sets -- can
end the process the same way. The `released-handle-error' an already-freed
handle signals on its next use is a single-threaded guarantee: it holds when
the free and the next call are sequential on one thread, not when a call on
that booster is already running elsewhere. See `docs/user-guide/threads.md'
for both backends' windows."))

(defgeneric free-booster (booster)
  (:documentation "Free BOOSTER. Does nothing if it was already freed.

Freeing the same BOOSTER from two threads at once can end the process outright
rather than signalling anything catchable -- see `docs/user-guide/threads.md'."))

;;; ---------------------------------------------------------------------------
;;; Fallbacks: the backend is loaded, its unified-API methods are not
;;;
;;; `cl-gbdt/lightgbm' is Layer 1 alone -- it registers the backend class and opens the
;;; library, and deliberately does not carry the methods that implement the generic
;;; functions above. That makes a state reachable which used to be impossible: a program
;;; holding an open backend and a generic function with no applicable method for it.
;;; Measured on SBCL, that produces `SB-PCL::NO-APPLICABLE-METHOD-ERROR', whose report
;;; names neither the backend nor the system to load. Policy section 7 asks for a typed
;;; condition instead, and these are it.
;;;
;;; Each is specialized on a base class -- `backend', `dataset', `booster' -- so a real
;;; backend's own method is strictly more specific and always wins. Nothing here fires in
;;; a program that loaded `cl-gbdt/<backend>/unified'.
;;;
;;; The six that take keyword arguments spell out the generic function's own keyword list
;;; rather than saying `&key &allow-other-keys', and that is load-bearing rather than
;;; cosmetic. Being specialized on a base class does not make a fallback INAPPLICABLE next
;;; to a backend's own method -- the base class is a superclass of the concrete one, so
;;; both are applicable and only the ORDER differs -- and CLHS 7.6.5 says a generic
;;; function accepts every keyword as soon as ANY applicable method has
;;; `&allow-other-keys'. Written that way, these fallbacks silently widened six of the
;;; thirteen generics for every caller: `(make-dataset backend matrix :totally-bogus 1)'
;;; stopped signalling and started returning a dataset, a typo'd `:parameters' or
;;; `:num-rounds' trained a differently configured model with nothing raised, and SBCL's
;;; compile-time warning for a misspelled keyword went quiet. Listing the keywords
;;; explicitly satisfies the congruence rules -- the list IS the generic function's own,
;;; which is also what every concrete method declares -- and moves the set of accepted
;;; keywords nowhere. Adding a keyword to one of these generics means adding it here too.

(defmethod make-dataset ((backend backend) matrix
                         &key label weight group feature-names parameters reference missing
                           categorical-features)
  "Fallback for a BACKEND whose unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `make-dataset' rather than building anything from MATRIX.
A Layer 1 caller who does not need the unified API builds a dataset directly with that
backend's own `create-dataset' instead."
  (declare (ignore matrix label weight group feature-names parameters reference missing
                   categorical-features))
  (error 'backend-methods-not-loaded
         :backend (backend-name backend) :generic-function 'make-dataset))

(defmethod train ((backend backend) dataset
                  &key valid-sets num-rounds parameters record-history early-stopping
                       objective evaluation)
  "Fallback for a BACKEND whose unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `train'. Layer 1 has no equivalent of this generic as a
whole -- its own training report, early stopping and `:objective'/`:evaluation' callbacks
are `train''s own concepts -- so recovering from this means loading the unified system, not
composing a Layer 1 substitute."
  (declare (ignore dataset valid-sets num-rounds parameters record-history early-stopping
                   objective evaluation))
  (error 'backend-methods-not-loaded
         :backend (backend-name backend) :generic-function 'train))

(defmethod load-model ((backend backend) path)
  "Fallback for a BACKEND whose unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `load-model'. A Layer 1 caller loads a model with that
backend's own `load-model' in `api.lisp' instead, which returns a handle without dispatching
through this generic at all."
  (declare (ignore path))
  (error 'backend-methods-not-loaded
         :backend (backend-name backend) :generic-function 'load-model))

(defmethod dataset-num-rows ((dataset dataset))
  "Fallback for a DATASET whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `dataset-num-rows'. A Layer 1 caller reads the same
count with that backend's own `dataset-num-rows' in `api.lisp', which needs no unified
system loaded."
  (error 'backend-methods-not-loaded
         :backend (backend-name (handle-backend dataset))
         :generic-function 'dataset-num-rows))

(defmethod dataset-num-features ((dataset dataset))
  "Fallback for a DATASET whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `dataset-num-features'. A Layer 1 caller reads the same
count with that backend's own `dataset-num-features' in `api.lisp', which needs no unified
system loaded."
  (error 'backend-methods-not-loaded
         :backend (backend-name (handle-backend dataset))
         :generic-function 'dataset-num-features))

(defmethod free-dataset ((dataset dataset))
  "Fallback for a DATASET whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `free-dataset' rather than freeing anything. A Layer 1
caller frees the same handle with that backend's own `free-dataset' in `api.lisp' instead."
  (error 'backend-methods-not-loaded
         :backend (backend-name (handle-backend dataset))
         :generic-function 'free-dataset))

(defmethod update-one-iteration ((booster booster))
  "Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `update-one-iteration' rather than advancing anything. A
Layer 1 caller drives the same loop with that backend's own `update-one-iteration' in
`api.lisp'. `train''s own loop does not call that function either: it calls `native.lisp''s
`%update-one-iteration' directly, because `api.lisp''s `update-one-iteration' would re-check
every handle on every iteration -- see that backend's `train' method for the same note in
its own words."
  (error 'backend-methods-not-loaded
         :backend (backend-name (handle-backend booster))
         :generic-function 'update-one-iteration))

(defmethod predict ((booster booster) matrix &key kind num-iteration missing)
  "Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `predict' rather than predicting anything on MATRIX. A
Layer 1 caller predicts with that backend's own `predict' in `api.lisp' instead."
  (declare (ignore matrix kind num-iteration missing))
  (error 'backend-methods-not-loaded
         :backend (backend-name (handle-backend booster))
         :generic-function 'predict))

(defmethod save-model ((booster booster) path &key num-iteration)
  "Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `save-model' rather than writing anything to PATH. A
Layer 1 caller saves the same model with that backend's own `save-model' in `api.lisp'
instead."
  (declare (ignore path num-iteration))
  (error 'backend-methods-not-loaded
         :backend (backend-name (handle-backend booster))
         :generic-function 'save-model))

(defmethod model-to-string ((booster booster) &key num-iteration)
  "Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `model-to-string' rather than returning anything. A
Layer 1 caller gets the same string with that backend's own `model-to-string' in `api.lisp'
instead."
  (declare (ignore num-iteration))
  (error 'backend-methods-not-loaded
         :backend (backend-name (handle-backend booster))
         :generic-function 'model-to-string))

(defmethod feature-importance ((booster booster) &key kind num-iteration)
  "Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `feature-importance' rather than computing anything. A
Layer 1 caller gets the same result with that backend's own `feature-importance' in
`api.lisp' instead."
  (declare (ignore kind num-iteration))
  (error 'backend-methods-not-loaded
         :backend (backend-name (handle-backend booster))
         :generic-function 'feature-importance))

(defmethod evaluation ((booster booster))
  "Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `evaluation' rather than reading anything. A Layer 1
caller gets the same metrics with that backend's own `evaluation' in `api.lisp' instead."
  (error 'backend-methods-not-loaded
         :backend (backend-name (handle-backend booster))
         :generic-function 'evaluation))

(defmethod free-booster ((booster booster))
  "Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `free-booster' rather than freeing anything. A Layer 1
caller frees the same handle with that backend's own `free-booster' in `api.lisp' instead."
  (error 'backend-methods-not-loaded
         :backend (backend-name (handle-backend booster))
         :generic-function 'free-booster))

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

(defmacro with-backend ((var form) &body body)
  "Bind VAR to the backend FORM returns, evaluate BODY, and always close it.

Nest this OUTSIDE `with-dataset' and `with-booster', never inside them. `close-backend'
calls `cffi:close-foreign-library', and once it has, `free-dataset' and `free-booster' on
that backend `warn' and leak rather than calling the C free -- deliberately, because the
older path called into a possibly-unmapped library. A `with-backend' nested inside a
handle's macro therefore closes the library first and leaks the handle second:

    (with-backend (backend (open-backend :xgboost))
      (with-dataset (train (make-dataset backend matrix :label labels))
        (with-booster (model (train backend train :num-rounds 10))
          (predict model matrix))))

Declarations at the head of BODY are moved onto a fresh binding of VAR that shadows the one
FORM's value is stored in, exactly as in `with-dataset' and for the same empirically-verified
reason: splicing them onto the outer binding, which `unwind-protect''s cleanup clause also
reads, puts an `(ignore VAR)' declaration from BODY in the same scope as that read, which
SBCL flags as reading an ignored variable. Do not simplify this to `progn' or a single
binding.

This macro takes no lock and makes no claim about concurrency. Closing a backend while
another thread is inside a call on it is one of the hazards
`docs/user-guide/threads.md' names -- the check `handle-live-pointer' makes and the C call
that follows it are not held together, so nothing here prevents a close landing between
them."
  (multiple-value-bind (forms declarations) (parse-body body)
    `(let ((,var ,form))
       (unwind-protect
            (let ((,var ,var))
              ,@declarations
              ,@forms)
         (close-backend ,var)))))
