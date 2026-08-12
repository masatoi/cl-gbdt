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

## Planned

Functions worth wrapping that nobody has wrapped yet. This is the work list for S5 of the
Layer 1 standalone-library programme (see the implementation policy document), and each Note
says what wrapping it would let a caller do -- which is what a priority order gets argued from.

Being here is not a promise of a date, and it is not a claim that every row is worth the same.
Some rows are two-argument accessors; some are whole capabilities with a design already
written down. The Note is what separates them.

| Function | What wrapping it would give a caller |
|---|---|
| `LGBM_BoosterLoadModelFromString` | Load a model back from the string `model-to-string` returns, closing a round trip that is currently open at one end: `load-model` takes a pathname only, so a caller holding a model as a string -- out of a database row, an HTTP response, a `save-model` output read back -- has to write it to a temporary file first. |
| `LGBM_DatasetCreateFromFile` | Read a training set straight off disk in LightGBM's own text or binary format, without materialising it as a Lisp array first. This is the `:file-input` capability the implementation policy's フォローアップ section has already designed and left unimplemented; per its external-memory section the result is an ordinary in-memory dataset, so this is a convenience and a memory-peak win, not external memory. |
| `LGBM_DatasetSaveBinary` | Write an already-binned dataset to LightGBM's own binary format, so a second run over the same data pays for binning once rather than every time. Pairs with `LGBM_DatasetCreateFromFile` above; on its own it is write-only. |
| `LGBM_DatasetGetField` | Read back the `label`, `weight`, `init_score`, `group` or `position` a dataset actually holds. `make-dataset` sets these through `LGBM_DatasetSetField` and nothing can currently read them, so a caller cannot confirm that the labels the library holds are the labels they meant to pass. |
| `LGBM_DatasetGetFeatureNames` | Read back the feature names a dataset holds, the counterpart of the `LGBM_DatasetSetFeatureNames` that `make-dataset :feature-names` already calls. |
| `LGBM_DatasetGetFeatureNumBin` | Ask how many bins a given feature was discretised into, which is the first thing to look at when a feature that should matter produces no splits -- a constant or near-constant column collapses to one bin. |
| `LGBM_DatasetGetSubset` | Build a row subset of an already-binned dataset. This is how cross-validation folds are made without re-binning the data once per fold, and how a caller subsamples a large training set cheaply. |
| `LGBM_DatasetAddFeaturesFrom` | Join another dataset's columns onto an existing one, so a wide training set can be assembled from separately built feature blocks instead of one array that must exist whole in memory. |
| `LGBM_DatasetDumpText` | Dump a constructed dataset to a text file, which is the only way to see what LightGBM's binning actually did to the caller's numbers. Upstream marks this debugging-only and the wrapper should repeat that; it is a diagnostic, not a persistence format (`LGBM_DatasetSaveBinary` is that). |
| `LGBM_DatasetCreateFromCSC` | Build a dataset from a column-major sparse matrix without transposing it to CSR first. Genuinely arguable: `make-csr-matrix` already covers sparse input and a caller holding CSC can convert in Lisp, so the gain is one avoided copy against the cost of a second sparse type on the public surface. Recorded as planned so the trade-off is argued rather than forgotten. |
| `LGBM_BoosterPredictForCSC` | Predict from a column-major sparse matrix. Same trade-off as `LGBM_DatasetCreateFromCSC`, and the same decision: a CSC type that a caller can build but not predict from would be the half-truth `*optional-symbols*`' `:sparse-input` entry already argues against. |
| `LGBM_BoosterPredictForFile` | Score a data file straight to a result file. Unlike every prediction path cl-gbdt has, the rows never pass through Lisp or through a foreign buffer this process sized, so a caller can score a dataset larger than the memory they have. |
| `LGBM_BoosterPredictSparseOutput` | Get SHAP feature contributions back as a sparse matrix. Dense `:contrib` output costs `num_class * num_data * (num_feature + 1)` doubles whether or not the model touches those features; on a wide model that is the difference between a prediction that fits in memory and one that does not. |
| `LGBM_BoosterFreePredictSparse` | Serves `LGBM_BoosterPredictSparseOutput` -- it is the only way to release the three buffers that call allocates -- and is classified with it. Without it, sparse contribution output leaks on every call. |
| `LGBM_BoosterPredictForMatSingleRow` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a 1-row matrix goes through `LGBM_BoosterPredictForMat`, which redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists the setup (including thread-count configuration) out of the scoring call entirely. |
| `LGBM_BoosterPredictForMatSingleRowFastInit` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a 1-row matrix goes through `LGBM_BoosterPredictForMat`, which redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists the setup (including thread-count configuration) out of the scoring call entirely. |
| `LGBM_BoosterPredictForMatSingleRowFast` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a 1-row matrix goes through `LGBM_BoosterPredictForMat`, which redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists the setup (including thread-count configuration) out of the scoring call entirely. |
| `LGBM_BoosterPredictForCSRSingleRow` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a 1-row matrix goes through `LGBM_BoosterPredictForMat`, which redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists the setup (including thread-count configuration) out of the scoring call entirely. |
| `LGBM_BoosterPredictForCSRSingleRowFastInit` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a 1-row matrix goes through `LGBM_BoosterPredictForMat`, which redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists the setup (including thread-count configuration) out of the scoring call entirely. |
| `LGBM_BoosterPredictForCSRSingleRowFast` | One decision, six rows: LightGBM's single-row prediction API, for a caller scoring one row at a time behind a request rather than a batch offline. `predict` on a 1-row matrix goes through `LGBM_BoosterPredictForMat`, which redoes predictor setup on every call; these reuse it, and the `FastInit`/`Fast` pair hoists the setup (including thread-count configuration) out of the scoring call entirely. |
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

This is the exclusion in this file most likely to be revisited. A caller who genuinely produces
rows incrementally, and is willing to own the thread contract, would gain something the wrapper
cannot offer today; the argument above is about cost and testability, not about the flow being
useless.

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

## Excluded — distributed training across machines

These set up and tear down LightGBM's own collective-communication layer so that several
processes on several machines train one model together. Wrapping them would mean this project
taking a position on process topology, port allocation and failure of a peer -- and proving it
with a functional test that needs more than one machine, which is not a test this repository's
CI can run. A caller who wants distributed LightGBM has the CLI and the Python package, both of
which own that orchestration already. `LGBM_NetworkInitWithFunctions` is doubly out: its two
arguments are C function pointers, which the "requires a C callback into Lisp" section below
rules out on its own terms.

| Function | Note |
|---|---|
| `LGBM_NetworkInit` | |
| `LGBM_NetworkFree` | |
| `LGBM_NetworkInitWithFunctions` | Also takes two C function pointers (`reduce_scatter_ext_fun`, `allgather_ext_fun`), so the callback exclusion below applies independently. |

## Excluded — requires a C callback into Lisp

This project has never introduced a C→Lisp callback, and the implementation policy's
external-memory section records that decision for XGBoost's data iterator: reaching that API
requires the caller to supply a data iterator implemented as a C callback, which that section
concludes is not a shape the portable contract of its section 7 can carry. The same reasoning
applies here, and each of these two has an additional, specific problem recorded in its Note.

| Function | Note |
|---|---|
| `LGBM_DatasetCreateFromCSRFunc` | Not wrappable even in principle from CFFI: the header documents `get_row_funptr` as "Pointer to `std::function<void(int idx, std::vector<std::pair<int, double>>& ret)>`" -- a C++ object with a C++ calling convention and C++ argument types, not a C function pointer. `cffi:defcallback` cannot produce one. |
| `LGBM_RegisterLogCallback` | The registered function is called by whichever thread LightGBM is logging from, including OpenMP worker threads the Lisp runtime has never seen. Calling into SBCL from a foreign thread it did not create is the case this project has the least ability to make safe, and log redirection is not worth being the first place it is attempted. |

## Excluded — on the ABI blacklist

`ffi-spec/ABI-BLACKLIST.md` is this project's record of functions that must never be called,
each with the reason and a replacement. Anything in its "still present in the generated
bindings -- must not be called" table is excluded here by construction, and
`tools/ci/check-binding-coverage.lisp`'s check D enforces the agreement between the two files
rather than leaving it to whoever edits one of them.

| Function | Note |
|---|---|
| `LGBM_DatasetCreateFromMats` | See that file's own row for the silent `int` / `int*` `is_row_major` break and the replacements it names. One of them, `LGBM_DatasetCreateFromMat`, is already wrapped. |

## Excluded — superseded by a wrapped entry point

A function cl-gbdt does not call because it calls a better one. Every claim in this section
names the wrapped function and has been checked against `src/lightgbm/native.lisp`'s
`:import-from` clause, not assumed from the names: a supersession claim that is wrong is worse
than no classification, because it reads like a decision someone made with evidence.

| Function | Note |
|---|---|
| `LGBM_BoosterPredictForMats` | Superseded by `LGBM_BoosterPredictForMat`, which `src/lightgbm/native.lisp` imports and `predict` calls. `Mats` differs only in taking an array of pointers to individual rows instead of one buffer -- and `call-with-foreign-matrix` (`src/data.lisp`) always yields exactly one contiguous row-major buffer, whether it pins a `simple-array`'s storage vector or copies into a fresh foreign one. A wrapper would therefore have to build a row-pointer array on top of a contiguous buffer it already has, to reach a call that then walks it. There is no input shape this project can produce that `Mats` serves better. |

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

| Function | Note |
|---|---|
| `LGBM_DatasetUpdateParamChecking` | Takes two parameter strings and errors if the change between them is one an already-constructed dataset cannot accept. It exists to guard mutating a live `Dataset`'s parameters, which cl-gbdt has no operation for: `make-dataset` fixes a dataset's parameters at construction and nothing reopens them. Wrapping it would publish a validator for a call that does not exist. Revisit only alongside a dataset-parameter-update operation, never before one. |
