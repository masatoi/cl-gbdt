# Backend differences

Where LightGBM and XGBoost genuinely differ, for a caller moving code between them.

## Where the two backends genuinely differ

A caller moving code from one backend to the other needs this in one place -- checked
directly against both backends' source, not only the differences the design doc calls
out first:

| | LightGBM | XGBoost |
|---|---|---|
| `make-dataset`'s `:reference` | Aligns the new dataset's bin mapper to an existing one's (required for a `train` `:valid-sets` entry) | Signals `unsupported-argument` -- no bin-mapper concept |
| `make-dataset`'s `:parameters` | Configures the dataset's own binning (`max_bin` and friends) | Signals `unsupported-argument`: the vendored header (`ffi-spec/xgboost/include/xgboost/c_api.h`) documents only `missing`/`nthread`/`data_split_mode` for `XGDMatrixCreateFromDense`'s config JSON, none of which are LightGBM's dataset-level binning keys, and confirmed empirically, the library silently ignores any other key rather than rejecting it -- forwarding `:parameters` there regardless would just move today's silent drop one layer deeper instead of fixing it |
| `update-one-iteration`'s return value | `nil` once an iteration produces no further split -- a real signal | Always `t` after a successful call; XGBoost's booster protocol has no equivalent signal |
| `predict`'s `:kind` on a `csr-matrix` | All four kinds, `LGBM_BoosterPredictForCSR` serving each of them with the same values the dense path produces | `:normal` and `:raw` only. `XGBoosterPredictFromCSR` is that library's *inplace* prediction entry point, not a CSR spelling of the dense call, and it refuses `:contrib` and `:leaf-index` -- passed through as `foreign-call-error`. See [Sparse input](data-and-prediction.md#sparse-input-csr-matrices) for the measured matrix and the workaround |
| `predict`'s second value (shape) | States the result array's own `array-dimensions` for `:normal`/`:raw`, derives one for `:contrib`, and states `NIL` for `:leaf-index` -- there is no property this project can check for that kind's sub-layout | Reads `out_shape`/`out_dim` back from the library and states it verbatim, on both matrix forms and for every `:kind` that form serves (all four dense, `:normal`/`:raw` only on a `csr-matrix`, per the row above). See [Prediction shape](data-and-prediction.md#prediction-shape) for both derivations and the measured shapes |
| `save-model`'s `:num-iteration` | Limits how many trees are saved | Signals `unsupported-argument` -- `XGBoosterSaveModel` always saves every round |
| `model-to-string`'s `:num-iteration` | Limits the rounds serialized | Signals `unsupported-argument` -- no iteration-limited variant exists |
| `feature-importance`'s `:num-iteration` | Limits the importance calculation | Signals `unsupported-argument` -- no iteration-limited variant exists |
| `feature-importance`'s result shape | Always one number per feature | Signals `unsupported-argument` instead of returning a result when the model reports a multi-dimensional score shape -- a `gblinear` booster's importance on a multi-class model, whose scores are a per-class matrix with no single-value reduction this backend will invent |
| What `evaluation` evaluates | The datasets `train` attached, read back by index (`LGBM_BoosterGetEval`): the library computed these metrics during training and this reads them out | The booster's own retained training set and `:valid-sets` entries, which this backend hands to `XGBoosterEvalOneIter` explicitly -- that call evaluates whatever DMatrices it is given and consults nothing the booster was built with, so passing the retained ones is what makes the index mean the same thing on both backends |
| `evaluation`'s values | `LGBM_BoosterGetEval`'s own doubles, returned unmodified -- the secondary value says `:value-source :library-doubles` | Parsed out of the single formatted line `XGBoosterEvalOneIter` produces -- `:value-source :parsed-text`, with that line itself kept verbatim under `:raw`, and a value XGBoost spelled `inf`/`nan` coming back as `nil` rather than a number. The same line is `cl-gbdt/xgboost:evaluate-one-iteration`'s own primary value at Layer 1, for a caller who wants it without going through the portable API |
| Model slicing | No counterpart at all: LightGBM's C API has nothing that extracts a range of boosting rounds into a new model, so `(backend-supports-p backend :model-slicing)` is `nil` and there is no LightGBM function to call | `cl-gbdt/xgboost:slice-model` (Layer 1, XGBoost-only), over `XGBoosterSlice`. Returns a new booster holding a half-open `[begin, end)` range of the parent's layers, independent of it -- freeing the parent leaves the slice usable. Deliberately not part of the unified API: with no LightGBM counterpart a portable version could only signal for every caller of one backend, or emulate, and emulating is what [the capability model](backends.md#asking-a-backend-what-it-can-do) exists to rule out |
| `train`'s `:objective` | **Overrides** any `objective` in `:parameters` -- all five spellings this library honours, `objective_type`, `app`, `application` and `loss` included -- forcing it to `"none"`, since `LGBM_BoosterUpdateOneIterCustom` refuses to run while the booster holds an objective function at all | Never rewrites `:parameters`; a configured objective's own prediction transform stays in effect, so a custom-objective run's `predict :kind :normal` differs from `:raw` there, while LightGBM's are identical. See [Custom objective](custom-training.md#custom-objective) for both |
| `backend-version` | Always `nil` -- LightGBM's C API has no version entry point | A `"MAJOR.MINOR.PATCH"` string, e.g. `"3.3.0"` |
| Untested-version warning | Never signalled -- there is no version to compare, so `open-backend` never checks one | `open-backend` signals `untested-backend-version` (a warning, not an error) when the loaded version falls outside the recorded supported range |

`src/version.lisp` records that supported range as two distinguishable claims: a narrow
*verified* one (the versions the functional suite above has actually run against)
and a wider *inferred* one (the range across which `tools/check-upstream.lisp` confirms cl-gbdt's
imported C functions' declarations are unchanged). The warning gates on the wider inferred
range -- a version different from the exact tested one is the common case for a compatible
caller, not a signal of trouble.

A version matrix (task 4) turned part of each inferred range into a measured one by actually
running the functional suite -- not just comparing headers -- against the range's endpoints.
The counts below are what that suite had when the matrix was measured; it has grown since,
and the rows are left as the measurement recorded them rather than restated against a total
that run never saw:

| library | version | result |
|---|---|---|
| LightGBM | 3.0.0 | not tested -- no aarch64 wheel exists on PyPI for this release, confirmed directly (`pip download lightgbm==3.0.0 --only-binary=:all:` finds no candidate); permanently inferred-only on this platform |
| LightGBM | 4.0.0 | ✅ all 106 assertions pass |
| LightGBM | 4.7.0 (pinned) | ✅ all 106 assertions pass |
| XGBoost | 1.7.0 | ❌ 105 of 106 pass; the ranking round trip fails (see below) |
| XGBoost | 2.0.0 | ✅ all 106 assertions pass |
| XGBoost | 3.3.0 (pinned) | ✅ all 106 assertions pass |

XGBoost 1.7.0 is a real, measured incompatibility, not a gap in coverage: every assertion
`tools/check-upstream.lisp` cannot see -- plain classification and multiclass round trips,
`feature-importance`, save/load, every close-backend guard -- passes unchanged, but
`xgboost-api-ranking-round-trip-respects-group-boundaries` (tests/functional/xgboost-api.lisp)
does not. That test trains a deliberately low-capacity `rank:pairwise` booster and asserts
predictions increase strictly within each query group; at 1.7.0 the first two rows of each
group tie instead. `XGBoosterUpdateOneIter` and `XGDMatrixSetUIntInfo` are both still present
and both still return success -- this is `rank:pairwise`'s internal behavior differing between
releases, the exact kind of break header comparison cannot see because no function's
declaration changed. XGBoost 2.0.0 was tried next and passed everything 3.3.0 does, so the
recorded range's lower bound moved to 2.0.0, not 1.7.0 -- see `*xgboost-version-range*`'s
docstring in `src/version.lisp` for the full account, including why this pulled both the
*verified* and the *inferred* bound up together rather than leaving 1.7.0 covered by a
header-only claim the functional suite had, by then, already disproven.

The matrix runs on Linux x86_64 only, not all three platforms `.github/workflows/test.yml`
already covers for the pinned versions -- see
[Continuous integration](../../CONTRIBUTING.md#continuous-integration)
for why that is a deliberate restriction, not a gap.

Run together against both backends:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *matrix*
  (make-array '(4 2) :element-type 'double-float
                      :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0)
                                           (5.0d0 0.0d0) (5.0d0 1.0d0))))
(defparameter *label*
  (make-array 4 :element-type 'single-float :initial-contents '(0.0 0.0 1.0 1.0)))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (format t "LightGBM backend-version: ~S~%" (cl-gbdt:backend-version lgbm))
  (format t "XGBoost  backend-version: ~S~%" (cl-gbdt:backend-version xgb))

  (cl-gbdt:with-dataset (lgbm-ds1 (cl-gbdt:make-dataset lgbm *matrix* :label *label*
                                     :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                    :verbose -1)))
    (cl-gbdt:with-dataset (lgbm-ds2 (cl-gbdt:make-dataset lgbm *matrix* :label *label*
                                       :reference lgbm-ds1
                                       :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                      :verbose -1)))
      (format t "LightGBM :reference accepted; aligned dataset has ~D rows~%"
              (cl-gbdt:dataset-num-rows lgbm-ds2))))

  (cl-gbdt:with-dataset (xgb-ds1 (cl-gbdt:make-dataset xgb *matrix* :label *label*))
    (handler-case (cl-gbdt:make-dataset xgb *matrix* :label *label* :reference xgb-ds1)
      (error (c) (format t "XGBoost  :reference SIGNALED ~A: ~A~%" (type-of c) c)))
    (cl-gbdt:with-dataset (xgb-ds-grouped (cl-gbdt:make-dataset xgb *matrix* :label *label*
                                             :group '(2 2)))
      (format t "XGBoost  :group accepted; grouped dataset has ~D rows~%"
              (cl-gbdt:dataset-num-rows xgb-ds-grouped)))
    (handler-case (cl-gbdt:make-dataset xgb *matrix* :label *label* :parameters '(:max-bin 3))
      (error (c) (format t "XGBoost  :parameters SIGNALED ~A: ~A~%" (type-of c) c)))

    (cl-gbdt:with-booster (xgb-booster (cl-gbdt:train xgb xgb-ds1 :num-rounds 1
                                          :parameters '(:objective "binary:logistic"
                                                         :max-depth 2 :verbosity 0)))
      (format t "XGBoost  update-one-iteration => ~S~%"
              (cl-gbdt:update-one-iteration xgb-booster))
      (dolist (call (list (lambda () (cl-gbdt:save-model xgb-booster "/tmp/m.json"
                                                           :num-iteration 1))
                           (lambda () (cl-gbdt:model-to-string xgb-booster :num-iteration 1))
                           (lambda () (cl-gbdt:feature-importance xgb-booster :num-iteration 1))))
        (handler-case (funcall call)
          (error (c) (format t "SIGNALED ~A: ~A~%" (type-of c) c))))))

  (cl-gbdt:with-dataset (lgbm-ds (cl-gbdt:make-dataset lgbm *matrix* :label *label*
                                    :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                   :verbose -1)))
    (cl-gbdt:with-booster (lgbm-booster (cl-gbdt:train lgbm lgbm-ds :num-rounds 1
                                           :parameters '(:objective "binary" :num-leaves 2
                                                          :min-data-in-leaf 1 :min-data-in-bin 1
                                                          :verbose -1)))
      (format t "LightGBM update-one-iteration => ~S~%"
              (cl-gbdt:update-one-iteration lgbm-booster))))

  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM backend-version: NIL
XGBoost  backend-version: "3.3.0"
LightGBM :reference accepted; aligned dataset has 4 rows
XGBoost  :reference SIGNALED UNSUPPORTED-ARGUMENT: make-dataset's :reference is not supported by XGBOOST: XGBoost has no bin-mapper alignment; :reference is a LightGBM-only concept.
XGBoost  :group accepted; grouped dataset has 4 rows
XGBoost  :parameters SIGNALED UNSUPPORTED-ARGUMENT: make-dataset's :parameters is not supported by XGBOOST: XGDMatrixCreateFromDense's config JSON only recognizes missing/nthread/data_split_mode, none of which are LightGBM's dataset-level binning parameters, and the library silently ignores any other key rather than rejecting it.
XGBoost  update-one-iteration => T
SIGNALED UNSUPPORTED-ARGUMENT: save-model's :num-iteration is not supported by XGBOOST: XGBoosterSaveModel has no iteration limit; every boosted round is saved.
SIGNALED UNSUPPORTED-ARGUMENT: model-to-string's :num-iteration is not supported by XGBOOST: XGBoosterSaveModelToBuffer has no iteration limit.
SIGNALED UNSUPPORTED-ARGUMENT: feature-importance's :num-iteration is not supported by XGBOOST: XGBoosterFeatureScore has no iteration limit.
LightGBM update-one-iteration => T
```

And, separately, the `feature-importance` shape rejection above, which needs a linear
(`gblinear`), multi-class booster to trigger -- a `multi:softprob` objective over 9 rows
in 3 classes:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/xgboost/unified) :silent t)

(let* ((backend (cl-gbdt:open-backend :xgboost))
       (rows-per-class 3) (num-classes 3) (cols 3)
       (rows (* rows-per-class num-classes))
       (matrix (make-array (list rows cols) :element-type 'double-float))
       (label (make-array rows :element-type 'single-float)))
  (dotimes (row rows)
    (let ((class (floor row rows-per-class)) (offset (mod row rows-per-class)))
      (dotimes (col cols)
        (setf (aref matrix row col) (coerce (+ (* class 10) offset col) 'double-float)))
      (setf (aref label row) (coerce class 'single-float))))
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend matrix :label label))
    (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 5
                                     :parameters (list :booster "gblinear"
                                                        :objective "multi:softprob"
                                                        :num-class num-classes
                                                        :verbosity 0)))
      (handler-case (cl-gbdt:feature-importance booster :kind :split)
        (error (c) (format t "feature-importance SIGNALED: ~A: ~A~%" (type-of c) c)))))
  (cl-gbdt:close-backend backend))
```

Output:

```
feature-importance SIGNALED: UNSUPPORTED-ARGUMENT: feature-importance's booster is not supported by XGBOOST: XGBoosterFeatureScore reported a 2-dimensional shape (3 3) instead of one score per feature -- most likely a linear (gblinear) booster's :split importance on a multi-class model, whose scores are a per-class matrix; no single value per feature can be derived without inventing a reduction this backend does not vouch for.
```

And `evaluation`, whose two rows above are about how each backend produces the numbers
rather than about one backend refusing something. The same call, the same 8 rows, the same
one `:valid-sets` entry, on both backends -- dataset 0 is the training set, dataset 1 is
that validation set, and the metric names are each library's own:

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

(defun show (name backend dataset-parameters booster-parameters reference-p)
  (cl-gbdt:with-dataset (train-set (apply #'cl-gbdt:make-dataset backend *eval-matrix*
                                          :label *eval-label* dataset-parameters))
    (cl-gbdt:with-dataset (valid-set (apply #'cl-gbdt:make-dataset backend *eval-matrix*
                                            :label *eval-label*
                                            (append (when reference-p
                                                      (list :reference train-set))
                                                    dataset-parameters)))
      (cl-gbdt:with-booster (booster (cl-gbdt:train backend train-set :num-rounds 5
                                                     :valid-sets (list valid-set)
                                                     :parameters booster-parameters))
        (multiple-value-bind (entries provenance) (cl-gbdt:evaluation booster)
          (format t "~A entries:    ~S~%~A provenance: ~S~%" name entries name provenance))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (show "LightGBM" lgbm
        '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
        '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
          :verbose -1 :metric "binary_logloss,auc")
        t)
  (show "XGBoost " xgb '()
        '(:objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0
          :eval-metric "logloss" :eval-metric "error")
        nil)
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM entries:    ((0 "binary_logloss" 0.35374722486733523d0)
                      (0 "auc" 1.0d0)
                      (1 "binary_logloss" 0.35374722486733523d0)
                      (1 "auc" 1.0d0))
LightGBM provenance: (:VALUE-SOURCE :LIBRARY-DOUBLES)
XGBoost  entries:    ((0 "logloss" 0.4740770012140274d0) (0 "error" 0.0d0)
                      (1 "logloss" 0.4740770012140274d0) (1 "error" 0.0d0))
XGBoost  provenance: (:VALUE-SOURCE :PARSED-TEXT :RAW
                      "[5]	0-logloss:0.47407700121402740	0-error:0.00000000000000000	1-logloss:0.47407700121402740	1-error:0.00000000000000000")
```

Dataset 0 and dataset 1 agree here because the validation set is built over the same rows
as the training set. Nothing names those datasets in the call above: LightGBM knows a
validation set by its index and by nothing else, so the portable API reports the position
the caller supplied it in rather than inventing `"valid_0"`-style names on its own. The
`0-`/`1-` prefixes inside XGBoost's `:raw` line are the names this backend must pass
`XGBoosterEvalOneIter` -- that call demands one per DMatrix -- and are the indices
themselves for exactly that reason.
