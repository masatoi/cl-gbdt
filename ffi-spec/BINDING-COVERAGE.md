# Binding Coverage

Every C function in `src/lightgbm/c-api.lisp` and `src/xgboost/c-api.lisp` is `wrapped`,
`planned`, or `excluded`. `tools/ci/check-binding-coverage.lisp` fails the build when one is
none of the three, so a regenerated binding cannot arrive without the project taking a
position on it.

**`wrapped` is not written down here.** The checker derives it: a binding is wrapped when the
backend's own `native.lisp` imports its Lisp name from the c-api package. That is the same
computation `tools/ci/check-abi-blacklist.lisp` and `tools/check-upstream.lisp` each make.
Recording it here as well would give it a second home to go stale in.

Neither are counts or percentages. Coverage in this project is guaranteed by classification --
every function has a recorded position -- not by a number. A percentage cannot tell "40%
missing, all of it Arrow and distributed training we will never support" apart from "40%
missing, half of it useful"; the headings below can. The checker prints the live counts on
every run.

A function this file lists must also still exist as a `cffi:defcfun`; the checker fails on a
name that does not, which is how a typo or an upstream removal surfaces. And a function listed
here must NOT be wrapped -- a row that survives its own function being wrapped is the failure
this file is most likely to develop, so the checker fails on that too.

Each `## Excluded` heading carries its section's reason, so a row's Note is for what the
heading does not already say: a function-specific caveat, or which wrapped entry point
supersedes it. A section with no argument under its heading is a section that has not made one.

One naming decision, settled here rather than left to be re-made per row. This file is English
throughout, but `docs/cl-gbdt-layered-api-implementation-policy.md` is written in Japanese, and
this file cites one of its sections often enough to need a convention: **フォローアップ** --
"follow-up", the section recording work that has been identified and deliberately not
scheduled. Citations name it unromanised, because that is the string a reader has to search
that document for, and romanising it would help nobody find it. Everything this file says
*about* what that section decided is in English.

## Planned

Functions worth wrapping that nobody has wrapped yet. This is the work list for S5 of the
Layer 1 standalone-library programme (see the implementation policy document), and each Note
says what wrapping it would let a caller do -- which is what a priority order gets argued from.

Being here is not a promise of a date, and it is not a claim that every row is worth the same.
Some rows are two-argument accessors; some are whole capabilities with a design already
written down. The Note is what separates them.

Both backends' rows share this one table, LightGBM's first. The `LGBM_` and `XG` prefixes
tell them apart, so there is no backend column.

| Function | What wrapping it would give a caller |
|---|---|
| `LGBM_BoosterLoadModelFromString` | Load a model back from the string `model-to-string` returns, closing a round trip that is currently open at one end: `load-model` takes a pathname only, so a caller holding a model as a string -- out of a database row, an HTTP response, a `save-model` output read back -- has to write it to a temporary file first. |
| `LGBM_DatasetCreateFromFile` | Read a training set straight off disk in LightGBM's own text or binary format, without materialising it as a Lisp array first. This is the `:file-input` capability the implementation policy's フォローアップ section has already designed and left unimplemented; per its external-memory section the result is an ordinary in-memory dataset, so this is a convenience and a memory-peak win, not external memory. |
| `LGBM_DatasetSaveBinary` | Write an already-binned dataset to LightGBM's own binary format, so a second run over the same data pays for binning once rather than every time. Pairs with `LGBM_DatasetCreateFromFile` above; on its own it is write-only. |
| `LGBM_DatasetGetField` | Read back the `label`, `weight`, `init_score`, `group` or `position` a dataset actually holds. `make-dataset` sets these through `LGBM_DatasetSetField` and nothing can currently read them, so a caller cannot confirm that the labels the library holds are the labels they meant to pass. |
| `LGBM_DatasetGetFeatureNames` | Ask a dataset which columns it believes it holds, and in what order. `make-dataset :feature-names` sets them through `LGBM_DatasetSetFeatureNames` and nothing reads them back, so a caller assembling a dataset from several sources -- or handed one built elsewhere in the same image -- cannot check that column 7 is the feature they think before they train on it. It is also the dataset-side half of the pairing that makes `LGBM_BoosterValidateFeatureNames` useful: the names to validate against a model have to come from somewhere. |
| `LGBM_DatasetGetFeatureNumBin` | Ask how many bins a given feature was discretised into, which is the first thing to look at when a feature that should matter produces no splits -- a constant or near-constant column collapses to one bin. |
| `LGBM_DatasetGetSubset` | Build a row subset of an already-binned dataset. This is how cross-validation folds are made without re-binning the data once per fold, and how a caller subsamples a large training set cheaply. |
| `LGBM_DatasetAddFeaturesFrom` | Join another dataset's columns onto an existing one, so a wide training set can be assembled from separately built feature blocks instead of one array that must exist whole in memory. |
| `LGBM_DatasetDumpText` | Dump a constructed dataset to a text file, which is the only way to see what LightGBM's binning actually did to the caller's numbers. Upstream marks this debugging-only and the wrapper should repeat that; it is a diagnostic, not a persistence format (`LGBM_DatasetSaveBinary` is that). |
| `LGBM_DatasetCreateFromCSC` | Build a dataset from a column-major sparse matrix without transposing it to CSR first. Genuinely arguable: `make-csr-matrix` already covers sparse input and a caller holding CSC can convert in Lisp, so the gain is one avoided copy against the cost of a second sparse type on the public surface. Recorded as planned so the trade-off is argued rather than forgotten. |
| `LGBM_BoosterPredictForCSC` | Predict from a column-major sparse matrix. Same trade-off as `LGBM_DatasetCreateFromCSC`, and the same decision: a CSC type that a caller can build but not predict from would be the half-truth `*optional-symbols*`' `:sparse-input` entry already argues against. |
| `LGBM_BoosterPredictForFile` | Score a data file straight to a result file. Unlike every prediction path cl-gbdt has, the rows never pass through Lisp or through a foreign buffer this process sized, so a caller can score a dataset larger than the memory they have. |
| `LGBM_BoosterPredictSparseOutput` | Get SHAP feature contributions back as a sparse matrix. Dense `:contrib` output costs `num_class * num_data * (num_feature + 1)` doubles whether or not the model touches those features; on a wide model that is the difference between a prediction that fits in memory and one that does not. |
| `LGBM_BoosterFreePredictSparse` | Serves `LGBM_BoosterPredictSparseOutput` -- it is the only way to release the three buffers that call allocates -- and is classified with it. Without it, sparse contribution output leaks on every call. |
| `LGBM_BoosterPredictForMatSingleRow` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a one-row input goes through `LGBM_BoosterPredictForMat` when it is dense and `LGBM_BoosterPredictForCSR` when it is sparse -- both wrapped -- and either redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists that setup out of the scoring call entirely, thread-count configuration included. |
| `LGBM_BoosterPredictForMatSingleRowFastInit` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a one-row input goes through `LGBM_BoosterPredictForMat` when it is dense and `LGBM_BoosterPredictForCSR` when it is sparse -- both wrapped -- and either redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists that setup out of the scoring call entirely, thread-count configuration included. |
| `LGBM_BoosterPredictForMatSingleRowFast` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a one-row input goes through `LGBM_BoosterPredictForMat` when it is dense and `LGBM_BoosterPredictForCSR` when it is sparse -- both wrapped -- and either redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists that setup out of the scoring call entirely, thread-count configuration included. |
| `LGBM_BoosterPredictForCSRSingleRow` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a one-row input goes through `LGBM_BoosterPredictForMat` when it is dense and `LGBM_BoosterPredictForCSR` when it is sparse -- both wrapped -- and either redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists that setup out of the scoring call entirely, thread-count configuration included. |
| `LGBM_BoosterPredictForCSRSingleRowFastInit` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a one-row input goes through `LGBM_BoosterPredictForMat` when it is dense and `LGBM_BoosterPredictForCSR` when it is sparse -- both wrapped -- and either redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists that setup out of the scoring call entirely, thread-count configuration included. |
| `LGBM_BoosterPredictForCSRSingleRowFast` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a one-row input goes through `LGBM_BoosterPredictForMat` when it is dense and `LGBM_BoosterPredictForCSR` when it is sparse -- both wrapped -- and either redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists that setup out of the scoring call entirely, thread-count configuration included. |
| `LGBM_FastConfigFree` | Serves the two `*SingleRowFastInit` calls above -- it releases the `FastConfigHandle` they return -- and is classified with them. A `FastConfigHandle` is a third foreign resource kind alongside `dataset` and `booster`, so wrapping the fast path means giving `src/handle.lisp` a third handle class; that cost belongs to the family's decision, not to this row. |
| `LGBM_BoosterRollbackOneIter` | Undo the last boosting iteration, so a caller driving `update-one-iteration` themselves can step back over an iteration that made a watched metric worse. Named as an unimplemented Phase 2 item by the implementation policy's フォローアップ section; this row agrees with it. |
| `LGBM_BoosterRefit` | Recompute an existing model's leaf values from new data, keeping its tree structure -- the cheap way to keep a deployed model current without retraining it. Named as an unimplemented Phase 2 item by the implementation policy's フォローアップ section; this row agrees with it. |
| `LGBM_BoosterResetParameter` | Change a booster's parameters between iterations -- a learning rate schedule, for instance -- without building a new booster and losing the trees already fitted. Named as an unimplemented Phase 2 item by the implementation policy's フォローアップ section; this row agrees with it. |
| `LGBM_BoosterResetTrainingData` | Point an existing booster at a different training set and keep boosting, which is what incremental training on newly arrived data looks like. Same family as the three rows above; the policy's フォローアップ section names the other three and not this one, so it is this file's own addition rather than a restatement. |
| `LGBM_BoosterMerge` | Fold another booster's trees into this one, so models fitted separately -- per shard, per worker, per time window -- can be scored as one. |
| `LGBM_BoosterGetCurrentIteration` | Ask a booster how many iterations it has actually run. A Layer 1 caller driving `update-one-iteration` in its own loop currently has to count for itself, and a booster from `load-model` cannot be asked at all. |
| `LGBM_BoosterNumModelPerIteration` | Ask how many trees one iteration adds -- one for regression and binary classification, one per class for multiclass -- which is what turns a tree index into an (iteration, class) pair. |
| `LGBM_BoosterNumberOfTotalModel` | Ask how many trees a model holds in total, the bound on any tree index a caller passes to the leaf-value calls below. |
| `LGBM_BoosterGetLeafValue` | Read one leaf's output value out of a trained model, addressed by tree and leaf index -- the smallest unit of "what did this model actually learn" that does not require parsing a dumped model. |
| `LGBM_BoosterSetLeafValue` | Write one leaf's output value, which is how a caller clamps or hand-corrects a single pathological leaf without retraining. Pairs with `LGBM_BoosterGetLeafValue`; wrapping the getter alone would let a caller see the problem and not fix it. |
| `LGBM_BoosterDumpModel` | Get the trained trees as JSON -- split features, thresholds, leaf values, structure -- so a caller can inspect a model, diff two models, or re-implement scoring elsewhere. `model-to-string` returns LightGBM's own text format, which is for reloading, not for reading. |
| `LGBM_BoosterGetFeatureNames` | Ask a model which feature names it was trained with. A booster that came from `load-model` carries no dataset, so today there is no way to find out what columns it expects, in what order. |
| `LGBM_BoosterValidateFeatureNames` | Have LightGBM itself check that the columns about to be predicted on match the ones the model was trained with, turning a silent column-order mistake -- which otherwise produces plausible, wrong numbers -- into an error. |
| `LGBM_BoosterGetLoadedParam` | Recover, as JSON, the parameters a model was trained with. For a booster from `load-model` this is the only record of its objective, its number of classes, and its learning rate that exists in the process. |
| `LGBM_BoosterGetLinear` | Ask whether a model fits linear models at its leaves (`linear_tree`). A caller reading leaf values, or reimplementing scoring from a dumped model, gets the wrong answer if they assume constant leaves and the model does not have them. |
| `LGBM_BoosterGetUpperBoundValue` | Ask for the largest raw score the model can produce, over all leaves. Lets a caller size an output range, or detect that a model can never reach a threshold they are testing against. |
| `LGBM_BoosterGetLowerBoundValue` | Ask for the smallest raw score the model can produce. Pairs with `LGBM_BoosterGetUpperBoundValue`. |
| `LGBM_BoosterShuffleModels` | Reorder a trained model's trees within an iteration range. Niche, and the weakest row in this table: it matters to a caller who scores with a truncated `:num-iteration` and wants that prefix to be a different random subset of the trees each time, and to nobody else. Recorded as planned because it has a caller-facing meaning, not because it has a claim on anyone's time. |
| `LGBM_SetMaxThreads` | Bound LightGBM's CPU use for the whole process in one call. Today the only route is a `num_threads` entry in every dataset's and every booster's parameter string, which a caller has to remember at every construction site and cannot change afterwards. |
| `LGBM_GetMaxThreads` | Read back the process-wide thread limit. Pairs with `LGBM_SetMaxThreads`; on its own it answers a question a caller who never set it does not have. |
| `LGBM_DumpParamAliases` | Read the library's own parameter-to-alias map, as JSON, at run time. This project already depends on that map: `*objective-parameter-names*` in `src/config/objective.lisp` is the `objective` entry of it, and its docstring records that the five spellings were read off LightGBM 4.7.0 by hand. A hand-transcribed copy of a table the library will hand over on request is stale the moment `ffi-spec/VERSIONS` moves. |
| `XGDMatrixCreateFromURI` | Read a training set straight off disk, XGBoost reading the file itself, so a caller whose data is already in a file never materialises it as a Lisp array. This is the `:file-input` capability the implementation policy's フォローアップ section has already designed and left unimplemented, and it is the XGBoost half of the row `LGBM_DatasetCreateFromFile` is above -- one capability, both backends. It is also `ffi-spec/ABI-BLACKLIST.md`'s own named replacement for `XGDMatrixCreateFromFile`, which is excluded below, so this row is what makes that recommendation reachable rather than a dead pointer. Per the policy's external-memory section the result is an ordinary in-memory DMatrix, so this is a convenience and a memory-peak win, not external memory. |
| `XGDMatrixSaveBinary` | Write an already-constructed DMatrix to XGBoost's own binary format, so a second run over the same data pays for construction once rather than every time. Pairs with `XGDMatrixCreateFromURI` above; on its own it is write-only. Same row as `LGBM_DatasetSaveBinary`. The header excludes `QuantileDMatrix` and external-memory DMatrix, neither of which this project builds, so that restriction costs nothing today. |
| `XGDMatrixCreateFromCSC` | Build a DMatrix from a column-major sparse matrix without transposing it to CSR first. Genuinely arguable, and the same trade-off `LGBM_DatasetCreateFromCSC` records: `make-csr-matrix` already covers sparse input and a caller holding CSC can convert in Lisp, so the gain is one avoided copy against the cost of a second sparse type on the public surface. One difference from LightGBM's row is worth carrying: XGBoost has no CSC *prediction* entry point at all, so on this backend a CSC type would be constructible and not predictable from -- exactly the half-truth `LGBM_BoosterPredictForCSC` exists to avoid on the other. |
| `XGDMatrixCreateFromColumnar` | Build a DMatrix one column at a time, from a JSON list of array-interface descriptors, rather than from one row-major matrix. Two things a caller cannot get today follow. Features of *different* element types in one dataset: `make-dataset` takes a single Lisp array, so every column shares its type. And categorical features named by *string*: the header's columnar format carries category names as arrow-shaped offset/value buffers, where `make-dataset :categorical-features` can only name 0-based columns whose contents are already numeric codes. `src/xgboost/array-interface.lisp` already emits the descriptor one column needs; assembling them into the JSON list this call takes is the added work. |
| `XGBoosterPredictFromColumnar` | Predict from the same column-at-a-time layout `XGDMatrixCreateFromColumnar` above builds from, and classified with it -- a columnar input a caller can train on but not predict from would be the half-truth `*optional-symbols*`' `:sparse-input` entry already argues against. This is also one of XGBoost's inplace prediction calls, so it carries `XGBoosterPredictFromDense`'s no-DMatrix property below, and would have to be measured for the same prediction-kind limit. |
| `XGBoosterPredictFromDense` | Score a dense matrix without building a DMatrix for it. This is the dense counterpart of a call this backend already makes: `%predict-from-csr` (`src/xgboost/native.lisp`) reaches `XGBoosterPredictFromCSR`, XGBoost's inplace prediction, for a `csr-matrix`, while the dense path builds a transient DMatrix, attaches nothing to it, predicts, and frees it. Closing that asymmetry takes a whole construction-and-free pass out of every dense `predict`. One limit to measure before wrapping rather than assume: `%predict-from-csr`'s docstring records that this backend's inplace CSR prediction serves `:normal` and `:raw` only, refusing `:contrib` and `:leaf-index` with a clean nonzero return; whether the dense inplace path has the same limit has not been measured here, and a wrapper must not silently fall back to the DMatrix path to paper over it. |
| `XGDMatrixSliceDMatrix` | One decision, two rows: build a row subset of an existing DMatrix, which is how cross-validation folds are made without rebuilding each fold from Lisp arrays, and how a caller subsamples a large training set cheaply. Same row as `LGBM_DatasetGetSubset`. Not to be confused with `XGBoosterSlice`, which is wrapped and which `slice-model` calls: that slices a *model* by boosting round, these slice *data* by row. `XGDMatrixSliceDMatrixEx` is this function plus an `allow_groups` flag, so it is a strict superset and wrapping the family is one function, not two; both rows appear because both bindings exist and each needs a position. |
| `XGDMatrixSliceDMatrixEx` | One decision, two rows: build a row subset of an existing DMatrix, which is how cross-validation folds are made without rebuilding each fold from Lisp arrays, and how a caller subsamples a large training set cheaply. Same row as `LGBM_DatasetGetSubset`. Not to be confused with `XGBoosterSlice`, which is wrapped and which `slice-model` calls: that slices a *model* by boosting round, these slice *data* by row. `XGDMatrixSliceDMatrixEx` is `XGDMatrixSliceDMatrix` plus an `allow_groups` flag, so it is a strict superset and wrapping the family is one function, not two; both rows appear because both bindings exist and each needs a position. |
| `XGDMatrixGetStrFeatureInfo` | Ask a DMatrix which columns it believes it holds, in what order, and what type each is. `make-dataset :feature-names` sets `feature_name` through `XGDMatrixSetStrFeatureInfo` and `:categorical-features` sets `feature_type` through the same wrapped call, and nothing reads either back -- so a caller cannot confirm that column 7 is the feature they think, nor that the columns they marked categorical are the columns XGBoost marked. Same row as `LGBM_DatasetGetFeatureNames`, with the feature-type half added; LightGBM has no per-feature type string. |
| `XGDMatrixGetInfoRef` | Read back the `label`, `weight`, `base_margin` or `group` a DMatrix actually holds. `make-dataset` sets these through `XGDMatrixSetInfoFromInterface` and `XGDMatrixSetUIntInfo`, both wrapped, and nothing can read them, so a caller cannot confirm that the labels the library holds are the labels they meant to pass. Same row as `LGBM_DatasetGetField`. Of the three read-back entry points this is the one upstream recommends -- its own header comment says it replaces `XGDMatrixGetFloatInfo` and `XGDMatrixGetUIntInfo` below -- and the only one that can express a matrix-shaped field. It returns an array-interface JSON document rather than a pointer and a length, so the wrapper pays for parsing it. |
| `XGDMatrixGetFloatInfo` | One decision with `XGDMatrixGetInfoRef` above and `XGDMatrixGetUIntInfo` below: the three answer the same question and wrapping the read-back means choosing one, not three. Recorded as planned rather than superseded because this file's supersession section requires the superseding function to be *wrapped*, and `XGDMatrixGetInfoRef` is not -- so no such claim can honestly be made yet. What this one adds is a pointer and a length directly, where `XGDMatrixGetInfoRef` returns a JSON document to parse; what it gives up is matrix-shaped output. **Genuinely arguable which of the three should be wrapped**, and recorded as planned so that choice gets made once and deliberately rather than defaulted into. |
| `XGDMatrixGetUIntInfo` | Same decision as `XGDMatrixGetFloatInfo` above, and the same arguable status. This is the `unsigned` half, which is where a ranking dataset's `group` field lives: `%set-group-field` (`src/xgboost/native.lisp`) writes it through `XGDMatrixSetUIntInfo`, so a read-back that skips this one leaves `group` the single field `make-dataset` can set and nothing can check. |
| `XGDMatrixNumNonMissing` | Ask how many of a DMatrix's cells the library actually stored -- that is, how sparse the matrix it built really is. This is the direct check on `make-dataset :missing`, the `:missing-value` capability only this backend provides: a caller who passes `:missing 0.0` over a matrix of known density can confirm from this one number that the zeros were dropped, where today the only evidence is that training behaves differently. |
| `XGDMatrixGetDataAsCSR` | Read back the predictor values a DMatrix actually stored -- and, for a quantised DMatrix, the *quantised* values, which is the only way to see what XGBoost's binning did to the caller's numbers. Same role `LGBM_DatasetDumpText` fills for LightGBM, and with the same caveat: upstream marks this "for testing" and the wrapper should repeat that, since it is a diagnostic and not a persistence format (`XGDMatrixSaveBinary` above is that). Unusually for this C API the caller allocates the output buffers, so wrapping it means sizing them first -- from `XGDMatrixNumRow`, wrapped, and `XGDMatrixNumNonMissing` above. |
| `XGDMatrixGetQuantileCut` | Read the bin boundaries `hist` and `approx` chose for each feature. A caller on this project's categorical path is already on one of those two -- `make-dataset :categorical-features` requires it, since `exact` refuses categorical splits -- and has no way to see what the histogram method decided. The neighbouring, coarser question, how many bins a feature got, is what `LGBM_DatasetGetFeatureNumBin` answers for LightGBM; this answers the finer one, which cuts. |
| `XGDMatrixGetCategories` | One decision, four rows: read back the category sets a categorical DMatrix or a trained model actually holds. `make-dataset :categorical-features` names which columns hold categories, but nothing reads back *which* categories the library found in them, and `predict`'s own docstrings already rest on the claim that the trained trees carry the category sets they split on -- this family is what would let a caller check that claim rather than take it. The two `ExportToArrow` spellings are the ones with readable output; the plain `Get` pair returns an opaque `CategoriesHandle` that nothing else in these bindings can read. The Arrow exclusion below does not reach any of the four: none takes an Arrow C data interface structure, and the export is a JSON document describing arrow-shaped buffers, decodable without an Arrow dependency. Upstream marks all of them experimental, which argues for wrapping them late rather than for having no position on them. |
| `XGDMatrixGetCategoriesExportToArrow` | One decision, four rows: read back the category sets a categorical DMatrix or a trained model actually holds. `make-dataset :categorical-features` names which columns hold categories, but nothing reads back *which* categories the library found in them, and `predict`'s own docstrings already rest on the claim that the trained trees carry the category sets they split on -- this family is what would let a caller check that claim rather than take it. The two `ExportToArrow` spellings are the ones with readable output; the plain `Get` pair returns an opaque `CategoriesHandle` that nothing else in these bindings can read. The Arrow exclusion below does not reach any of the four: none takes an Arrow C data interface structure, and the export is a JSON document describing arrow-shaped buffers, decodable without an Arrow dependency. Upstream marks all of them experimental, which argues for wrapping them late rather than for having no position on them. |
| `XGBoosterGetCategories` | One decision, four rows: read back the category sets a categorical DMatrix or a trained model actually holds. `make-dataset :categorical-features` names which columns hold categories, but nothing reads back *which* categories the library found in them, and `predict`'s own docstrings already rest on the claim that the trained trees carry the category sets they split on -- this family is what would let a caller check that claim rather than take it. The two `ExportToArrow` spellings are the ones with readable output; the plain `Get` pair returns an opaque `CategoriesHandle` that nothing else in these bindings can read. The Arrow exclusion below does not reach any of the four: none takes an Arrow C data interface structure, and the export is a JSON document describing arrow-shaped buffers, decodable without an Arrow dependency. Upstream marks all of them experimental, which argues for wrapping them late rather than for having no position on them. |
| `XGBoosterGetCategoriesExportToArrow` | One decision, four rows: read back the category sets a categorical DMatrix or a trained model actually holds. `make-dataset :categorical-features` names which columns hold categories, but nothing reads back *which* categories the library found in them, and `predict`'s own docstrings already rest on the claim that the trained trees carry the category sets they split on -- this family is what would let a caller check that claim rather than take it. The two `ExportToArrow` spellings are the ones with readable output; the plain `Get` pair returns an opaque `CategoriesHandle` that nothing else in these bindings can read. The Arrow exclusion below does not reach any of the four: none takes an Arrow C data interface structure, and the export is a JSON document describing arrow-shaped buffers, decodable without an Arrow dependency. Upstream marks all of them experimental, which argues for wrapping them late rather than for having no position on them. |
| `XGBCategoriesFree` | Serves the four `*Categories*` calls above -- it releases the `CategoriesHandle` each of them creates, and is the only function in these bindings that consumes one -- and is classified with them. Without it every category read-back leaks, and the plain `Get` pair would be a call whose only legal follow-up does not exist. |
| `XGBoosterGetStrFeatureInfo` | One decision, two rows: the booster's own feature names and per-feature types, which are separate from the DMatrix's. A booster from `load-model` carries no dataset, so today there is no way to ask what columns it expects, in what order, or which of them it treats as categorical. Same row as `LGBM_BoosterGetFeatureNames`, with a setter half LightGBM's C API has no counterpart for and a feature-type half it has no concept of. |
| `XGBoosterSetStrFeatureInfo` | One decision, two rows with `XGBoosterGetStrFeatureInfo` above: make a model self-describing before it is saved, or repair one that was saved without names. This row carries a concrete interaction to settle first, not a hypothetical one: `%feature-score-index` (`src/xgboost/native.lisp`) requires every name `XGBoosterFeatureScore` reports to be XGBoost's default `f<index>` form and signals `foreign-call-error` on anything else, and its docstring records that this holds today only because *this backend never attaches names to the booster* -- a DMatrix built with `make-dataset :feature-names` still reports `"f1"`, measured against the vendored library. Wrapping this call would make that assumption false, so `feature-importance` has to learn the caller's own names in the same change. |
| `XGBoosterGetAttr` | One decision, three rows: XGBoost's per-model string attributes, a key/value map stored *inside* the saved model. A caller can stamp a model with the date it was trained, the revision of the code that produced it, or the identifier of the data it was fitted on, and read it back after `load-model` -- today nothing survives `save-model` but the trees and the parameters. `XGBoosterSetAttr` writes (and deletes, on a null value), `XGBoosterGetAttr` reads one key, `XGBoosterGetAttrNames` lists them, and the three are close to useless apart. LightGBM has no counterpart, so this would be a backend-specific Layer 1 addition rather than a unified-API one. |
| `XGBoosterSetAttr` | One decision, three rows: XGBoost's per-model string attributes, a key/value map stored *inside* the saved model. A caller can stamp a model with the date it was trained, the revision of the code that produced it, or the identifier of the data it was fitted on, and read it back after `load-model` -- today nothing survives `save-model` but the trees and the parameters. `XGBoosterSetAttr` writes (and deletes, on a null value), `XGBoosterGetAttr` reads one key, `XGBoosterGetAttrNames` lists them, and the three are close to useless apart. LightGBM has no counterpart, so this would be a backend-specific Layer 1 addition rather than a unified-API one. |
| `XGBoosterGetAttrNames` | One decision, three rows: XGBoost's per-model string attributes, a key/value map stored *inside* the saved model. A caller can stamp a model with the date it was trained, the revision of the code that produced it, or the identifier of the data it was fitted on, and read it back after `load-model` -- today nothing survives `save-model` but the trees and the parameters. `XGBoosterSetAttr` writes (and deletes, on a null value), `XGBoosterGetAttr` reads one key, `XGBoosterGetAttrNames` lists them, and the three are close to useless apart. LightGBM has no counterpart, so this would be a backend-specific Layer 1 addition rather than a unified-API one. |
| `XGBoosterSaveJsonConfig` | One decision, two rows: recover a booster's whole internal configuration as a JSON document, and put one back. The getter is this backend's answer to what `LGBM_BoosterGetLoadedParam` answers for LightGBM -- for a booster from `load-model` it is the only record in the process of the objective it was trained with, its number of classes and its learning rate. The setter is the half LightGBM has no counterpart for: it restores that configuration onto another booster, which is how a caller reproduces a run's settings without transcribing them by hand. Upstream calls the pair experimental and warns the signature may change. |
| `XGBoosterLoadJsonConfig` | One decision, two rows: recover a booster's whole internal configuration as a JSON document, and put one back. `XGBoosterSaveJsonConfig` is this backend's answer to what `LGBM_BoosterGetLoadedParam` answers for LightGBM -- for a booster from `load-model` it is the only record in the process of the objective it was trained with, its number of classes and its learning rate. This one is the half LightGBM has no counterpart for: it restores that configuration onto another booster, which is how a caller reproduces a run's settings without transcribing them by hand. Upstream calls the pair experimental and warns the signature may change. |
| `XGBoosterSerializeToBuffer` | One decision, two rows: a memory snapshot of a booster -- model *and* configuration *and* training state -- into a byte buffer, and back out of one. The header's own Serialization note is what makes this distinct from `save-model`: the "Model" functions deliberately strip the configuration, so a run interrupted after N iterations cannot be resumed from a saved model with its parameters intact. This pair is what check-pointing a long run would use. The header's other stated use, resuming a distributed task, is out of scope for the reason the distributed section below gives; check-pointing within one process is not. |
| `XGBoosterUnserializeFromBuffer` | One decision, two rows: a memory snapshot of a booster -- model *and* configuration *and* training state -- into a byte buffer, and back out of one. The header's own Serialization note is what makes this distinct from `load-model`: the "Model" functions deliberately strip the configuration, so a run interrupted after N iterations cannot be resumed from a saved model with its parameters intact. This pair is what check-pointing a long run would use. The header's other stated use, resuming a distributed task, is out of scope for the reason the distributed section below gives; check-pointing within one process is not. |
| `XGBoosterLoadModelFromBuffer` | Load a model back from the bytes `model-to-string` returns, closing a round trip that is currently open at one end: `load-model` takes a pathname only, so a caller holding a model as bytes -- out of a database row, an HTTP response, a `save-model` output read back -- has to write it to a temporary file first. The saving half, `XGBoosterSaveModelToBuffer`, is already wrapped and is what `model-to-string` calls. Same row as `LGBM_BoosterLoadModelFromString`, with one difference to carry into the wrapper: XGBoost's buffer is bytes in `json` or `ubj` encoding, not necessarily text. |
| `XGBoosterDumpModel` | One decision, four rows: get the trained trees as text, JSON or dot -- split features, thresholds, leaf values, structure -- so a caller can inspect a model, diff two models, or re-implement scoring elsewhere. `model-to-string` returns XGBoost's own save format, which is for reloading, not for reading. Same row as `LGBM_BoosterDumpModel`. The four differ only in arguments: `Ex` adds a `format`, `WithFeatures` takes per-feature names and types in place of an fmap file path, and `XGBoosterDumpModelExWithFeatures` takes both -- a strict superset of the other three, so wrapping the family is one function, not four. All four are listed because all four bindings exist and each needs a position; none is *superseded* in this file's sense, since that section requires the superseding function to be wrapped and none of these is. |
| `XGBoosterDumpModelEx` | One decision, four rows: get the trained trees as text, JSON or dot -- split features, thresholds, leaf values, structure -- so a caller can inspect a model, diff two models, or re-implement scoring elsewhere. `model-to-string` returns XGBoost's own save format, which is for reloading, not for reading. Same row as `LGBM_BoosterDumpModel`. The four differ only in arguments: `Ex` adds a `format`, `WithFeatures` takes per-feature names and types in place of an fmap file path, and `XGBoosterDumpModelExWithFeatures` takes both -- a strict superset of the other three, so wrapping the family is one function, not four. All four are listed because all four bindings exist and each needs a position; none is *superseded* in this file's sense, since that section requires the superseding function to be wrapped and none of these is. |
| `XGBoosterDumpModelWithFeatures` | One decision, four rows: get the trained trees as text, JSON or dot -- split features, thresholds, leaf values, structure -- so a caller can inspect a model, diff two models, or re-implement scoring elsewhere. `model-to-string` returns XGBoost's own save format, which is for reloading, not for reading. Same row as `LGBM_BoosterDumpModel`. The four differ only in arguments: `Ex` adds a `format`, `WithFeatures` takes per-feature names and types in place of an fmap file path, and `XGBoosterDumpModelExWithFeatures` takes both -- a strict superset of the other three, so wrapping the family is one function, not four. All four are listed because all four bindings exist and each needs a position; none is *superseded* in this file's sense, since that section requires the superseding function to be wrapped and none of these is. |
| `XGBoosterDumpModelExWithFeatures` | One decision, four rows: get the trained trees as text, JSON or dot -- split features, thresholds, leaf values, structure -- so a caller can inspect a model, diff two models, or re-implement scoring elsewhere. `model-to-string` returns XGBoost's own save format, which is for reloading, not for reading. Same row as `LGBM_BoosterDumpModel`. The four differ only in arguments: `Ex` adds a `format`, `WithFeatures` takes per-feature names and types in place of an fmap file path, and this one takes both -- a strict superset of the other three, so wrapping the family is one function, not four, and this is the one to wrap. All four are listed because all four bindings exist and each needs a position; none is *superseded* in this file's sense, since that section requires the superseding function to be wrapped and none of these is. |
| `XGBoosterReset` | Release the caches a booster accumulates while training -- gradients, predictions, leaf partitions -- which the header's Booster group states persist until either this call or `XGBoosterFree`. A caller who trains once and then keeps the booster for scoring holds training-sized memory for the life of that booster today, and the only way out is to `save-model` it and `load-model` it back. LightGBM has no counterpart. |
| `XGBSetGlobalConfig` | One decision, two rows: set and read XGBoost's process-wide parameters -- `verbosity`, `nthread`, `use_rmm` -- in one call, as a flat JSON object. Today the only route to any of them is a per-booster parameter a caller has to remember at every construction site, and `verbosity` has no real home there at all, since it governs the library's own logging rather than any one model. Same role `LGBM_SetMaxThreads` and `LGBM_GetMaxThreads` fill for LightGBM, over a wider set of keys. |
| `XGBGetGlobalConfig` | One decision, two rows with `XGBSetGlobalConfig` above: set and read XGBoost's process-wide parameters -- `verbosity`, `nthread`, `use_rmm` -- in one call, as a flat JSON object. On its own the getter answers a question a caller who never set anything does not have, which is why the two are one decision rather than two. |
| `XGBuildInfo` | Ask the loaded shared library what it was built with -- OpenMP, CUDA, federated learning, and its dependency versions -- as JSON. `backend-version`, over the wrapped `XGBoostVersion`, answers only the version number, and that does not say whether a build can do a thing: two libraries at the same version differ in exactly these flags. It is also the evidence a future capability probe would need before answering true for anything the CUDA section below rules out today, where the current answer rests on what `tools/fetch-libs.sh` fetches rather than on asking the library. |

## Excluded — needs the Arrow C data interface

These take `struct ArrowArray`, `struct ArrowSchema` or `struct ArrowArrayStream`: the Arrow C
data interface. cl-gbdt has no Arrow dependency and no way to build one of those structures
from Lisp data, which means it has no way to write the functional test that would prove a
wrapper correct -- a test would have to fabricate an Arrow buffer by hand, and would then be
testing this project's idea of the format rather than the format. A caller who has Arrow data
already has the numbers, and can hand them to `make-dataset` or `predict` as a matrix or a
`csr-matrix`. Revisit if a Common Lisp Arrow implementation this project is willing to depend
on appears; the decision is about the dependency and the test, not about the functions.

| Function | Note |
|---|---|
| `LGBM_DatasetCreateFromArrowStream` | |
| `LGBM_DatasetCreateFromArrow` | Upstream marks this deprecated in favour of `LGBM_DatasetCreateFromArrowStream`, which is excluded for the same reason. |
| `LGBM_DatasetSetFieldFromArrowStream` | |
| `LGBM_DatasetSetFieldFromArrow` | Upstream marks this deprecated in favour of `LGBM_DatasetSetFieldFromArrowStream`, which is excluded for the same reason. |
| `LGBM_BoosterPredictForArrowStream` | |
| `LGBM_BoosterPredictForArrow` | Upstream marks this deprecated in favour of `LGBM_BoosterPredictForArrowStream`, which is excluded for the same reason. |

## Excluded — streaming dataset construction

LightGBM's multi-threaded incremental construction flow. The header documents it as four
steps: build a Dataset "schema" (`LGBM_DatasetCreateFromSampledColumn` or
`LGBM_DatasetCreateByReference`), call `LGBM_DatasetInitStreaming`, push rows from N threads
each carrying the `tid` it was assigned, then call `LGBM_DatasetMarkFinished`.

`docs/cl-gbdt-layered-api-implementation-policy.md`'s フォローアップ section has already settled
what this is and is not, in the course of dropping external memory from Phase 4: it records
that `LGBM_DatasetInitStreaming` and its neighbours are a mechanism for *constructing* a
dataset by streaming rows in from several threads, and not a mechanism for keeping data out of
memory. This section agrees with that finding rather than re-deriving one.

With that motivation gone, what is left is a C-level concurrency contract the *caller* has to
uphold: N threads, each with its own `tid` in `0..N-1`, each writing a disjoint row range, none
of them finishing the dataset early. cl-gbdt has never put a threading obligation on a caller,
and a wrapper that gets this wrong corrupts memory instead of signalling -- there is no
`released-handle-error` shape available for "two threads claimed the same tid". `make-dataset`
already builds the same finished dataset from one matrix or one `csr-matrix`.

That contract is dispositive for seven of the rows below -- `LGBM_DatasetInitStreaming`, the
four `PushRows*`, `LGBM_DatasetMarkFinished` and `LGBM_DatasetSetWaitForManualFinish`. **The
other eight are excluded by membership in the flow rather than by the contract**: they are
single-threaded and take no `tid`, but each exists only to get this flow to its first pushed
row or to carry its schema between processes, and none of them builds a dataset a caller could
train on by itself. Wrapping any of them without the flow would publish a step of a procedure
whose remaining steps are excluded. Each row's Note names the step it is.

This is the exclusion in this file most likely to be revisited, and a reader revisiting it
should weigh both halves of the argument above: the seven fall if the thread contract is judged
ownable, the eight only fall with them.

**This section is LightGBM-only by construction, and Task 2 should not widen it.** XGBoost has
no counterpart flow: its incremental and external-memory construction
(`XGDMatrixCreateFromCallback`, `XGQuantileDMatrixCreateFromCallback`,
`XGExtMemQuantileDMatrixCreateFromCallback`) is driven by a caller-supplied C data iterator, so
those functions belong under "requires a C callback into Lisp" below, on that section's own
argument and not on this one's. Of those three only `XGDMatrixCreateFromCallback` is actually
classified there, because it is the only one in the generated bindings: `tools/regen.lisp` emits
only names beginning `XGB` or `XGD`, and the other two begin `XGQ` and `XGE`.

| Function | Note |
|---|---|
| `LGBM_DatasetInitStreaming` | |
| `LGBM_DatasetSetWaitForManualFinish` | |
| `LGBM_DatasetMarkFinished` | |
| `LGBM_DatasetPushRows` | |
| `LGBM_DatasetPushRowsWithMetadata` | |
| `LGBM_DatasetPushRowsByCSR` | |
| `LGBM_DatasetPushRowsByCSRWithMetadata` | |
| `LGBM_DatasetCreateByReference` | Step 1 of the flow: allocates a dataset whose bins are aligned to an existing one, for rows that arrive later. |
| `LGBM_DatasetCreateFromSampledColumn` | Step 1 of the flow, named as such in `LGBM_DatasetPushRowsWithMetadata`'s own header comment: allocates a dataset and picks its bin boundaries from sampled data, for rows that arrive later. |
| `LGBM_GetSampleCount` | Serves `LGBM_DatasetCreateFromSampledColumn`: it sizes the buffer that `LGBM_SampleIndices` then fills, and that call consumes. Classified with it. |
| `LGBM_SampleIndices` | Serves `LGBM_DatasetCreateFromSampledColumn`: it produces the `sample_indices` that call takes. Classified with it. |
| `LGBM_DatasetSerializeReferenceToBinary` | Carries a dataset's bin schema between processes so step 1 can happen in a different one. Classified with the flow it serves. |
| `LGBM_DatasetCreateFromSerializedReference` | The other end of `LGBM_DatasetSerializeReferenceToBinary`: step 1 of the flow, from a schema serialised elsewhere. |
| `LGBM_ByteBufferGetAt` | Serves `LGBM_DatasetSerializeReferenceToBinary`, the only function in the whole header that produces a `ByteBufferHandle`, and is classified with it -- there is nothing else in these bindings a ByteBuffer could come from. |
| `LGBM_ByteBufferFree` | Serves `LGBM_DatasetSerializeReferenceToBinary`, the only function in the whole header that produces a `ByteBufferHandle`, and is classified with it. |

## Excluded — requires a CUDA device

These take a `__cuda_array_interface__`: a pointer into GPU device memory together with its
shape and its stream. cl-gbdt has no CUDA dependency and no way to allocate or fill device
memory from Lisp, so it cannot produce an input for one of these at all -- the pointer would
have to come from something outside this project entirely, and without one there is no
functional test to write. Nor is there a machine to run such a test on: every runner in
`.github/workflows/test.yml`'s matrix -- `ubuntu-latest`, `ubuntu-24.04-arm` and
`macos-latest` -- is GPU-less, and a macOS runner cannot have a CUDA device at all.

A caller who has data on a GPU can copy it back to host memory and hand it to `make-dataset` or
`predict` as a matrix or a `csr-matrix`. What they give up by doing so is the transfer they were
avoiding, which is the entire point of these four entry points -- so this is a real capability
set aside, not a formality. Revisit if this project gains both a CUDA-capable CI machine and a
way to name device memory from Lisp; the decision is about the input and the test, not about the
functions. Note that it does not rest on how the vendored library was *built*: whether the wheel
`tools/fetch-libs.sh` extracts has CUDA support compiled in is not established here, and would
not change the answer if it did -- `XGBuildInfo`, planned above, is the call that would settle
that question if it ever mattered.

This section is XGBoost-only, and not by a scoping choice: LightGBM reaches its own CUDA support
through a `device_type` parameter on ordinary entry points, and its vendored header declares no
CUDA-specific function at all. Each row names the CPU spelling of the same input; the four cover
this backend's dense and columnar shapes, its CSR entry points having no CUDA counterpart.

| Function | Note |
|---|---|
| `XGDMatrixCreateFromCudaArrayInterface` | The CPU spelling of the same input, `XGDMatrixCreateFromDense`, is wrapped and is what `make-dataset` calls. |
| `XGDMatrixCreateFromCudaColumnar` | The CPU spelling, `XGDMatrixCreateFromColumnar`, is planned above. |
| `XGBoosterPredictFromCudaArray` | The CPU spelling, `XGBoosterPredictFromDense`, is planned above. |
| `XGBoosterPredictFromCudaColumnar` | The CPU spelling, `XGBoosterPredictFromColumnar`, is planned above. |

## Excluded — distributed training across machines

Both backends expose their own collective-communication layer, so that several processes on
several machines train one model together: LightGBM's is `LGBM_Network*` below, XGBoost's is
its `XGCommunicator*` and `XGTracker*` families. Wrapping either would mean this project taking
a position on process topology, port allocation and failure of a peer -- and proving it with a
functional test that needs more than one machine, which is not a test this repository's CI can
run. A caller who wants distributed training has each project's CLI and Python package, both of
which own that orchestration already. This is a scope decision about what cl-gbdt is -- an
in-process wrapper -- and it is expected to hold for both backends.

`LGBM_NetworkInitWithFunctions` is doubly out: its two arguments are C function pointers, which
the "requires a C callback into Lisp" section below rules out on its own terms.

XGBoost's two families are named above for the scope decision they illustrate, not because
either is classified here: **neither is in the generated bindings at all.** `tools/regen.lisp`
emits only names beginning `XGB` or `XGD`, and `XGCommunicator*` and `XGTracker*` begin with
neither, so nothing in `src/xgboost/c-api.lisp` can reach them and there is nothing to take a
position on -- a row for one would name a function the checker cannot find. What this section
does classify on the XGBoost side is the single row below, which is not a collective operation
itself but the trace distributed training leaves on an ordinary DMatrix call.

| Function | Note |
|---|---|
| `LGBM_NetworkInit` | |
| `LGBM_NetworkFree` | |
| `LGBM_NetworkInitWithFunctions` | Also takes two C function pointers (`reduce_scatter_ext_fun`, `allgather_ext_fun`), so the callback exclusion below applies independently. |
| `XGDMatrixDataSplitMode` | Reports whether a DMatrix was built with `data_split_mode` row or column -- a distinction that exists only because a distributed run may have split the columns across workers before any one process saw the data. `data_split_mode` is an optional config key of `XGDMatrixCreateFromDense` and `XGDMatrixCreateFromCSR`, both wrapped, and cl-gbdt never sets it, so this getter can only ever answer row. It becomes worth wrapping on the same trigger the rest of this section does, and not before. |

## Excluded — requires a C callback into Lisp

This project has never introduced a C→Lisp callback, and the implementation policy's
external-memory section records that decision for XGBoost's data iterator: reaching that API
requires the caller to supply a data iterator implemented as a C callback, which that section
concludes is not a shape the portable contract of its section 7 can carry. The same reasoning
applies here, and each of these four has an additional, specific problem recorded in its Note.

The two backends contribute two rows each, and the four split the same way: one log-callback
registration and one dataset constructor per backend.

| Function | Note |
|---|---|
| `LGBM_DatasetCreateFromCSRFunc` | Not wrappable even in principle from CFFI: the header documents `get_row_funptr` as "Pointer to `std::function<void(int idx, std::vector<std::pair<int, double>>& ret)>`" -- a C++ object with a C++ calling convention and C++ argument types, not a C function pointer. `cffi:defcallback` cannot produce one. |
| `LGBM_RegisterLogCallback` | The registered function is called by whichever thread LightGBM is logging from, including OpenMP worker threads the Lisp runtime has never seen. Calling into SBCL from a foreign thread it did not create is the case this project has the least ability to make safe, and log redirection is not worth being the first place it is attempted. |
| `XGDMatrixCreateFromCallback` | XGBoost's external-memory constructor, which the implementation policy's external-memory section already ruled out by name and for exactly this reason. It has a second, independent problem: the flow its own header documents needs `XGProxyDMatrixCreate` and the `XGProxyDMatrixSetData*` setters to hand each batch over, and none of those is in the generated bindings -- `tools/regen.lisp` emits only names beginning `XGB` or `XGD`, and they begin `XGP`. So even granted a callback mechanism, there is no way from these bindings to feed this call its data. |
| `XGBRegisterLogCallback` | The XGBoost spelling of `LGBM_RegisterLogCallback` above, and excluded with it -- though on its own terms the case is weaker, because this header claims the callback "will run on the thread that registered it" where LightGBM's makes no such promise. That claim is unverified here; it would have to be measured against a library that logs from inside OpenMP regions before anything could rest on it. Either way, log redirection is not worth being the first place this project attempts a C→Lisp callback. |

## Excluded — on the ABI blacklist

`ffi-spec/ABI-BLACKLIST.md` is this project's record of functions that must never be called,
each with the reason and a replacement. Anything in its "still present in the generated
bindings -- must not be called" table is excluded here by construction, and
`tools/ci/check-binding-coverage.lisp`'s check D enforces the agreement between the two files
rather than leaving it to whoever edits one of them.

| Function | Note |
|---|---|
| `LGBM_DatasetCreateFromMats` | See that file's own row for the silent `int` / `int*` `is_row_major` break. It names two replacements, and they do not fare alike here: `LGBM_DatasetCreateFromMat` is already wrapped and is the one to reach for, while the other -- the streaming API -- is itself excluded above, so a reader following that half of the recommendation should stop there rather than take it as licence. |
| `XGDMatrixCreateFromDataIter` | See that file's own row for the `float missing` argument gained upstream since the reference implementations were written. Its named replacement, `XGDMatrixCreateFromCallback`, is itself excluded above as a C callback -- so unlike `LGBM_DatasetCreateFromMats`, neither the blacklisted function nor its replacement is reachable, and this operation is unavailable rather than merely unwrapped. |
| `XGDMatrixCreateFromFile` | See that file's own row: already gone from XGBoost's `master`, one release after the tag `ffi-spec/VERSIONS` pins. Its named replacement, `XGDMatrixCreateFromURI`, is planned above, which is the recommendation this file can act on. Upstream's own header marks this deprecated since 2.0.0 in favour of that same function, so the blacklist and the header agree; the blacklist's reason is the stronger of the two and is the one that governs. |

## Excluded — superseded by a wrapped entry point

A function cl-gbdt does not call because it calls a better one. Every claim in this section must
name the superseding function, and that function must actually be wrapped -- checked against
the backend's own `native.lisp` `:import-from` clause, not assumed from the names. A
supersession claim that is wrong is worse than no classification, because it reads like a
decision someone made with evidence; and a claim that stays right only until someone unwraps
its superseder is why the checker fails on a row whose function has since been wrapped.

The rows below have been checked that way: the `LGBM_` row against `src/lightgbm/native.lisp`,
the `XG` rows against `src/xgboost/native.lisp`.

Note what does *not* land here, since the shape recurs on the XGBoost side. Where upstream
marks a function deprecated in favour of one this project does not call either, the claim
cannot be made -- `XGDMatrixGetFloatInfo` and the four `XGBoosterDumpModel*` spellings are
planned above for exactly that reason, not excluded.

| Function | Note |
|---|---|
| `LGBM_BoosterPredictForMats` | Superseded by `LGBM_BoosterPredictForMat`, which `src/lightgbm/native.lisp` imports and `predict` calls. `Mats` differs only in taking an array of pointers to individual rows instead of one buffer -- and `call-with-foreign-matrix` (`src/data.lisp`) always yields exactly one contiguous row-major buffer, whether it pins a `simple-array`'s storage vector or copies into a fresh foreign one. A wrapper would therefore have to build a row-pointer array on top of a contiguous buffer it already has, to reach a call that then walks it. There is no input shape this project can produce that `Mats` serves better. |
| `XGDMatrixCreateFromMat` | Superseded by `XGDMatrixCreateFromDense`, which `src/xgboost/native.lisp` imports and `%create-dmatrix` calls. `Mat` takes a bare `float*` with `nrow`, `ncol` and `missing` as C arguments; `Dense` takes the same buffer wrapped in the array-interface JSON descriptor `src/xgboost/array-interface.lisp` builds, with `missing` in its config. The descriptor is the strictly more capable of the two, because it carries the element type: a `double-float` matrix reaches XGBoost at its own precision, which `Mat`'s prototype makes impossible. |
| `XGDMatrixCreateFromMat_omp` | Superseded by `XGDMatrixCreateFromDense`, on `XGDMatrixCreateFromMat`'s reasoning above. Its one addition over that function, an `nthread` argument, is the `nthread` key of `XGDMatrixCreateFromDense`'s config, so the supersession gives up nothing. |
| `XGBoosterPredict` | Superseded by `XGBoosterPredictFromDMatrix`, which `src/xgboost/native.lisp` imports and `%predict-from-dmatrix` calls. Upstream marks this deprecated and names the same replacement. It is also strictly less informative: it reports one flat length where the replacement reports `out_shape` and `out_dim`, which is the very thing the `:prediction-shape` capability publishes as `predict`'s second value. |
| `XGBoosterBoostOneIter` | Superseded by `XGBoosterTrainOneIter`, which `src/xgboost/native.lisp` imports and `train :objective` calls. Upstream marks this deprecated since 2.1.0. It takes the gradient and the Hessian as a bare `float*` and a flat length, with no iteration number; the replacement takes array-interface descriptors and an `iter`, and the descriptor's shape is what carries the `[rows, groups]` layout a multiclass custom objective produces -- the wrapper writes exactly that, and a flat length could not express it. |
| `XGDMatrixSetFloatInfo` | Superseded by `XGDMatrixSetInfoFromInterface`, which `src/xgboost/native.lisp` imports and `%set-info-field` calls for `label` and `weight`. The replacement covers a strict superset -- any field, any element type -- because it takes an array-interface descriptor rather than a `float*` and a length. Unlike the two rows around it, upstream does *not* itself mark this one deprecated, so the claim here rests on the replacement's coverage and on this project calling it, not on a header annotation. (`XGDMatrixSetUIntInfo` is deprecated on the same terms and is nonetheless still the entry point this backend uses for `group`, for the reason `%set-group-field`'s docstring records; it has no row here because it is wrapped.) |
| `XGDMatrixSetDenseInfo` | Superseded by `XGDMatrixSetInfoFromInterface`, which `src/xgboost/native.lisp` imports and `%set-info-field` calls. Upstream marks this deprecated since 2.1.0 and names the same replacement. |

## Excluded — not exported by the shared library

There is nothing to call. The generated binding exists because `src/regen/emit.lisp` emits what
the vendored header declares, and this is declared in the header -- but as an `INLINE_FUNCTION`
definition, not a `LIGHTGBM_C_EXPORT` declaration like every other entry point in the file, so
the symbol lives in each translation unit that includes the header rather than in the library.
`nm -D --defined-only vendor/lightgbm/lib/lib_lightgbm.so` finds it nowhere; it is the only one
of LightGBM's unwrapped bindings of which that is true.

This is the row that makes the case for classifying rather than counting. A percentage would
have carried it forever as coverage still to be won.

| Function | Note |
|---|---|
| `LGBM_SetLastError` | Also meaningless to call across the boundary even where the symbol exists: it writes the *caller's own* thread-local `LastErrorMsg()` buffer, so it tells LightGBM nothing. `LGBM_GetLastError`, the half of the pair that reads the library's buffer, is exported, and is wrapped. |

## Excluded — validates an operation cl-gbdt does not offer

A backend's C API contains guards as well as operations: functions whose whole job is to
decide whether some *other* call is about to be made legally. A guard is only as wrappable as
the call it guards. Where cl-gbdt offers that call, the guard belongs with it and is not a
separate decision; where cl-gbdt does not, wrapping the guard alone would publish a validator
for something a caller has no way to attempt, which reads as a promise that the operation is
coming. These are excluded for as long as that is true, and each row names the operation it is
waiting on -- so a section this small stays a decision that can be reversed on a known trigger,
rather than a leftover.

| Function | Note |
|---|---|
| `LGBM_DatasetUpdateParamChecking` | Waits on a dataset-parameter-update operation. It takes two parameter strings and errors if the change between them is one an already-constructed dataset cannot accept; `make-dataset` fixes a dataset's parameters at construction and nothing reopens them. Note that `LGBM_BoosterResetParameter` is planned above and does not count: that changes a *booster*'s parameters, and this guard does not cover it. |
