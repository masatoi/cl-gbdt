# Training

What `train` returns as its training report, and how `:early-stopping` uses it.

## Training report

`train` returns two values: the booster, and a `training-report` of the run just
completed. Its `training-report-series` is a list of `training-series`, one per metric per
dataset -- the same (index, metric-name) pairs `evaluation` reports for the trained
booster, in the same order. Each series carries `training-series-index` (0 for the training
set, 1 for the first `:valid-sets` entry, 2 for the second, and so on),
`training-series-metric` (the backend's own name for the metric) and
`training-series-values` (one value per completed iteration, oldest first, `NIL` where a
value could not be read as a real).

`:valid-sets` accepts two element forms, freely mixed in one list: a bare dataset, whose
series carry no name, or a `(name . dataset)` cons, where `name` is a string that becomes
`training-series-name` for every series recorded at that dataset's index. A series always
carries an index; it carries a non-`NIL` name only when the validation set it came from was
given one. The training set is never a `:valid-sets` entry, so its own series are always
index 0 with a `NIL` name -- nothing here invents a name for it, or for a validation set
passed bare. Two validation sets may legitimately share one name; their index, not their
name, is what tells the two apart in the report.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *eval-matrix*
  (make-array '(8 2) :element-type 'double-float
                      :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                           (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                           (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *eval-label*
  (make-array 8 :element-type 'single-float
                 :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defun show-report (name backend dataset-parameters booster-parameters reference-p)
  (cl-gbdt:with-dataset (train-set (apply #'cl-gbdt:make-dataset backend *eval-matrix*
                                          :label *eval-label* dataset-parameters))
    (cl-gbdt:with-dataset (valid-set (apply #'cl-gbdt:make-dataset backend *eval-matrix*
                                            :label *eval-label*
                                            (append (when reference-p
                                                      (list :reference train-set))
                                                    dataset-parameters)))
      ;; NOT `with-booster': it binds the primary value only, so a caller who wants the
      ;; report -- `train''s secondary value -- uses `multiple-value-bind' and frees the
      ;; booster itself instead.
      (multiple-value-bind (booster report)
          (cl-gbdt:train backend train-set :num-rounds 5
                          ;; A named :valid-sets entry: a bare dataset would leave
                          ;; TRAINING-SERIES-NAME NIL, same as the training set's own.
                          :valid-sets (list (cons "valid" valid-set))
                          :parameters booster-parameters)
        (unwind-protect
             (progn
               (dolist (series (cl-gbdt:training-report-series report))
                 (format t "~A series: index=~S name=~S metric=~S last=~S~%"
                         name (cl-gbdt:training-series-index series)
                         (cl-gbdt:training-series-name series)
                         (cl-gbdt:training-series-metric series)
                         (aref (cl-gbdt:training-series-values series) 4)))
               (format t "~A best-iteration: ~S~%" name
                       (cl-gbdt:training-report-best-iteration report)))
          (cl-gbdt:free-booster booster))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (show-report "LightGBM" lgbm
        '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
        '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
          :verbose -1 :metric "binary_logloss,auc")
        t)
  (show-report "XGBoost " xgb '()
        '(:objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0
          :eval-metric "logloss" :eval-metric "error")
        nil)
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM series: index=0 name=NIL metric="binary_logloss" last=0.35374722486733523d0
LightGBM series: index=0 name=NIL metric="auc" last=1.0d0
LightGBM series: index=1 name="valid" metric="binary_logloss" last=0.35374722486733523d0
LightGBM series: index=1 name="valid" metric="auc" last=1.0d0
LightGBM best-iteration: NIL
XGBoost  series: index=0 name=NIL metric="logloss" last=0.4740770012140274d0
XGBoost  series: index=0 name=NIL metric="error" last=0.0d0
XGBoost  series: index=1 name="valid" metric="logloss" last=0.4740770012140274d0
XGBoost  series: index=1 name="valid" metric="error" last=0.0d0
XGBoost  best-iteration: NIL
```

Dataset 1's series carry the name `"valid"` on both backends; dataset 0's -- the training
set -- stay `NIL` regardless, and so would dataset 1's if `valid-set` had been passed bare
instead of as `(cons "valid" valid-set)`. `training-report-best-iteration`,
`training-report-best-score` and `training-report-early-stopped-p` are all `NIL` above
because that call gave `train` no `:early-stopping` -- see the next section for what fills
them in, and [`:num-iteration :best`](#num-iteration-best) below for what the booster's own
best iteration is then good for.

A malformed `:valid-sets` element signals one of two conditions, kept distinct because they
are different mistakes: a `(name . dataset)` cons whose `name` is not a string signals
`unsupported-argument`, naming `:valid-sets` and the offending element; a cons whose
`dataset` half is not this backend's own kind of handle signals `wrong-backend-reference` --
the same condition a bare wrong-backend dataset already signals elsewhere in this API. Both
are checked before any foreign call. Duplicate names are not one of these mistakes: two
validation sets sharing a name train and report normally, distinguished by index.

### Stopping early: `:early-stopping`

`train` takes `:early-stopping`, a plist that ends the run once a watched metric stops
improving. All four keys are required, with no default for any of them:

| Key | Meaning |
|---|---|
| `:metric` | A string, the metric to watch, spelled the way this backend spells it in `evaluation` -- LightGBM's `"binary_logloss"`, XGBoost's `"logloss"` |
| `:dataset` | Which dataset's copy of that metric to watch: a string naming a `:valid-sets` entry, or an integer index (0 the training set, N+1 the Nth `:valid-sets` entry). A name matching two entries signals `unsupported-argument` -- two entries may share a name, but a watcher has to watch exactly one, so pass the index instead |
| `:direction` | `:lower-is-better` or `:higher-is-better` |
| `:rounds` | A positive integer: how many consecutive non-improving iterations are tolerated before the run stops |

`:direction` is required, and is never inferred, because neither library's C API exposes
whether a metric improves upward or downward -- there is nothing to read it off, and
guessing from the metric's *name* (a lookup table mapping `"logloss"` to "lower" and
`"auc"` to "higher") would be the same guess with a table in front of it. The caller
already knows which way their own metric goes; this API does not pretend to.

`training-report-best-iteration`, `-best-score` and `-early-stopped-p`, and the returned
booster's own `booster-best-iteration`, are filled by a run given `:early-stopping` -- but
not unconditionally: `:num-rounds` zero or negative never lets the watcher see an iteration
at all, and a run every one of whose watched values came back unreadable (see `evaluation`'s
account of a value the backend reported but could not be parsed) never has a real value to
call best, so `-best-iteration` and `-best-score` stay `NIL` in both cases even though
`-early-stopped-p` can still turn `T` in the second one. `NIL` keeps meaning "not
determined" on the report, exactly as it does with no `:early-stopping` at all, never
"iteration 0".

`:early-stopping` together with `:record-history nil` signals `unsupported-argument`: early
stopping needs the very per-iteration evaluation `:record-history nil` exists to skip, and
reading it costs the same whether one series is watched or every series is recorded, so
there is no cheaper middle path to offer a caller who asks for both.

The example below builds a validation set from the training labels *inverted*, which is a
doc-only trick to force the watched metric to worsen from the very first iteration and
provoke a stop within a handful of rounds -- never something a real validation set is built
from. It also demonstrates [`:num-iteration :best`](#num-iteration-best), covered right
after, in the same run: `predict`, `save-model` and `model-to-string` accept `:best`
wherever they accept `:num-iteration`, an additional value alongside `NIL` (every round)
and an explicit integer, never a new default. `:best` resolves to the booster's own
`booster-best-iteration` before anything else runs, and a booster with no best iteration to
resolve against signals `unsupported-argument` -- see that section for the full contract.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *es-matrix*
  (make-array '(8 2) :element-type 'double-float
                      :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                           (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                           (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *es-label*
  (make-array 8 :element-type 'single-float
                 :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))
(defparameter *es-inverted-label*
  (map '(vector single-float) (lambda (x) (- 1.0 x)) *es-label*))

(defun train-early-stopped (backend dataset-parameters booster-parameters metric reference-p)
  "Return (values BOOSTER REPORT TRAIN-SET VALID-SET). The caller frees all four -- the two
datasets after BOOSTER, since BOOSTER retains both strongly for its own lifetime, exactly
the order `with-booster' nested inside `with-dataset' would enforce."
  (let* ((train-set (apply #'cl-gbdt:make-dataset backend *es-matrix* :label *es-label*
                            dataset-parameters))
         (valid-set (apply #'cl-gbdt:make-dataset backend *es-matrix*
                            :label *es-inverted-label*
                            (append (when reference-p (list :reference train-set))
                                    dataset-parameters))))
    (multiple-value-bind (booster report)
        (cl-gbdt:train backend train-set :num-rounds 1000
                        :valid-sets (list (cons "valid" valid-set))
                        :early-stopping (list :metric metric :dataset "valid"
                                               :direction :lower-is-better :rounds 3)
                        :parameters booster-parameters)
      (values booster report train-set valid-set))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (multiple-value-bind (booster report train-set valid-set)
      (train-early-stopped lgbm '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1
                                                 :verbose -1))
                            '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1
                              :min-data-in-bin 1 :verbose -1 :metric "binary_logloss")
                            "binary_logloss" t)
    (unwind-protect
         (progn
           (format t "LightGBM num-rounds=~S early-stopped-p=~S best-iteration=~S~%"
                   (cl-gbdt:training-report-num-rounds report)
                   (cl-gbdt:training-report-early-stopped-p report)
                   (cl-gbdt:training-report-best-iteration report))
           (format t "LightGBM predict :num-iteration :best differs from every round: ~S~%"
                   (not (equalp (cl-gbdt:predict booster *es-matrix*)
                                 (cl-gbdt:predict booster *es-matrix* :num-iteration :best))))
           (cl-gbdt:save-model booster "/tmp/lgbm-best.txt" :num-iteration :best)
           (format t "LightGBM save-model :num-iteration :best wrote /tmp/lgbm-best.txt: ~S~%"
                   (and (probe-file "/tmp/lgbm-best.txt") t)))
      (cl-gbdt:free-booster booster)
      (cl-gbdt:free-dataset valid-set)
      (cl-gbdt:free-dataset train-set)))
  (cl-gbdt:close-backend lgbm))

(let ((xgb (cl-gbdt:open-backend :xgboost)))
  (multiple-value-bind (booster report train-set valid-set)
      (train-early-stopped xgb '()
                            '(:objective "binary:logistic" :max-depth 2 :eta 0.5
                              :verbosity 0 :eval-metric "logloss")
                            "logloss" nil)
    (unwind-protect
         (progn
           (format t "XGBoost  num-rounds=~S early-stopped-p=~S best-iteration=~S~%"
                   (cl-gbdt:training-report-num-rounds report)
                   (cl-gbdt:training-report-early-stopped-p report)
                   (cl-gbdt:training-report-best-iteration report))
           ;; XGBoost's save-model asymmetry, unaffected by :best: XGBoosterSaveModel has no
           ;; iteration limit at all -- it always writes every round -- so :best resolves to
           ;; an integer first and then meets the exact `unsupported-argument' check an
           ;; explicit :num-iteration already does. No special case is written around it.
           (handler-case
               (cl-gbdt:save-model booster "/tmp/xgb-best.json" :num-iteration :best)
             (error (c) (format t "XGBoost  save-model :num-iteration :best SIGNALED ~A: ~A~%"
                                 (type-of c) c)))
           ;; The escape hatch: slice to the best iteration first, then save the slice, which
           ;; `save-model' accepts with no :num-iteration at all -- a sliced booster's every
           ;; round already is the range the caller wanted.
           (let ((sliced (cl-gbdt/xgboost:slice-model
                          booster :begin 0 :end (cl-gbdt:booster-best-iteration booster))))
             (unwind-protect
                  (progn
                    (cl-gbdt:save-model sliced "/tmp/xgb-sliced.json")
                    (format t "XGBoost  slice-model to the best iteration then save-model ~
                               wrote /tmp/xgb-sliced.json: ~S~%"
                            (and (probe-file "/tmp/xgb-sliced.json") t)))
               (cl-gbdt:free-booster sliced))))
      (cl-gbdt:free-booster booster)
      (cl-gbdt:free-dataset valid-set)
      (cl-gbdt:free-dataset train-set)))
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM num-rounds=4 early-stopped-p=T best-iteration=1
LightGBM predict :num-iteration :best differs from every round: T
LightGBM save-model :num-iteration :best wrote /tmp/lgbm-best.txt: T
XGBoost  num-rounds=4 early-stopped-p=T best-iteration=1
XGBoost  save-model :num-iteration :best SIGNALED UNSUPPORTED-ARGUMENT: save-model's :num-iteration is not supported by XGBOOST: XGBoosterSaveModel has no iteration limit; every boosted round is saved.
XGBoost  slice-model to the best iteration then save-model wrote /tmp/xgb-sliced.json: T
```

The LightGBM booster kept fitting the training data for three more rounds after its watched
validation metric stopped improving at iteration 1, which is why predicting from its best
iteration alone differs from predicting over the full, unstopped run.

### `:num-iteration :best`

`predict`, `save-model` and `model-to-string` accept `:best` wherever they accept
`:num-iteration`, demonstrated together with `:early-stopping` in the example just above --
an additional value alongside `NIL` (every round) and an explicit integer, never a new
default: `NIL` keeps meaning "every round" on every booster, including one that has a best
iteration to resolve `:best` against. `feature-importance` also accepts `:num-iteration` but
does not accept `:best`; only the three named above do.

`:best` resolves to the booster's own `booster-best-iteration` -- the same iteration
`training-report-best-iteration` named when `train` was given `:early-stopping` -- before
anything else runs: LightGBM's own `:num-iteration` resolution knows only `NIL` and an
integer, so `:best` must already be one by the time it gets there. A booster with no best
iteration to resolve against -- never trained with `:early-stopping`, or a run that hit one
of the two `NIL` cases the previous section describes -- signals `unsupported-argument`:
the question has no answer for that booster, and this API does not invent one.

**The save-model asymmetry is not smoothed over for `:best`.** LightGBM's `save-model`
honours `:num-iteration`, `:best` included, and writes a file limited to that many trees.
XGBoost's `XGBoosterSaveModel` has no iteration limit at all, so `save-model` there already
signals `unsupported-argument` for any non-`NIL` `:num-iteration`; `:best` resolves to an
integer first and then meets that exact check, with no special case written around it, as
the output above shows. The escape hatch is `cl-gbdt/xgboost:slice-model` (see [the
differences table](backend-differences.md#where-the-two-backends-genuinely-differ) and
[Backend-specific packages](backends.md#backend-specific-packages)), also shown above: slice
to the best iteration first, then save the slice.

### Turning recording off: `:record-history`

Recording is not free, and it is on by default. `train` reads the whole evaluation once per
iteration, for every dataset the booster holds. Measured here over 500 rounds on 2000 rows ×
20 columns with two metrics configured, that roughly **doubled** LightGBM's wall-clock
`train` time -- with and without a validation set -- and added roughly **70-80%** to
XGBoost's with one validation set attached. XGBoost with no validation set stayed inside the
measurement noise, that backend evaluating every dataset in one call rather than one call
each. Treat these as orders of magnitude on one machine: run-to-run variance on the same
code is easily ±15%.

`train` therefore takes `:record-history`, `t` by default:

```lisp
(multiple-value-bind (booster report)
    (cl-gbdt:train backend train-set :num-rounds 500 :record-history nil)
  ;; report is a training-report with no series, over 500 rounds.
  (cl-gbdt:free-booster booster))
```

With `:record-history nil` no evaluation is read at all, and `train` costs what it cost
before it recorded anything -- measured against the commit this branch started from, the two
agree within the noise on both backends. The secondary value is still a `training-report`,
never `nil`, so a caller destructuring two values never has to handle two shapes: its
`training-report-series` is empty and its `training-report-num-rounds` is the run's length,
exactly as a run with no metric configured reports.

Recording also decides, on XGBoost, which `:valid-sets` entries `train` accepts at all. An
unlabelled DMatrix is the case this was found through: `XGBoosterUpdateOneIter` trains on it
happily, while `XGBoosterEvalOneIter` refuses it (`label and prediction size not match`). So
with recording on -- the default -- such an entry now fails `train` itself with
`foreign-call-error`, where before this branch it trained normally and failed only a later
`evaluation` call. The general rule is that any configuration whose evaluation path errors
while its update path does not now fails the whole run. `:record-history nil` never reaches
the evaluation path and restores the older behaviour. LightGBM tolerates the same input,
recording finite values, so this is XGBoost-specific in practice.
