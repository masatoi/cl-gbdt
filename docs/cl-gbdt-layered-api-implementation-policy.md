# cl-gbdt Layered API Restructuring: Implementation Policy

## 1. Purpose of this document

This document instructs the agent that will implement and revise `masatoi/cl-gbdt` from here on, covering the policy for integrating LightGBM and XGBoost, the boundary of the public API, the handling of backend-specific features, the migration procedure and the acceptance criteria.

When adding an individual feature, the implementing agent must give this document's rules priority, and must satisfy the following two at the same time.

1. For operations whose meaning can be guaranteed to be the same on both backends, provide a stable unified high-level API.
2. Do not delete, flatten or silently discard a backend-specific feature that cannot be unified; make it available through a safe backend-specific API.

The goal is not "presenting every feature of LightGBM and XGBoost as one homogeneous API". The goal is to achieve both the portability of shared operations and the availability of backend-specific features at once, through clear layers within the same library.

## 2. Current state

In the current `master`, the unified protocol's 12 generic functions are implemented for both LightGBM and XGBoost.

- Creating a dataset, and getting the number of rows and the number of features
- Training, and updating by one iteration
- Prediction
- Saving, loading and stringifying a model
- Feature importance
- Freeing a dataset / booster

The generated CFFI binding covers the upstream C API broadly, while the C functions the current high-level backend implementation uses are a part of it. Also, the current `src/lightgbm/backend.lisp` and `src/xgboost/backend.lisp` carry the following responsibilities in the same layer.

- Finding and initializing the shared library
- Calling the raw C API
- Converting C-API-specific representations for Lisp
- Handle ownership and freeing
- Implementing the methods of the unified generic functions
- Absorbing the differences between backends

This structure was reasonable at the stage where the basic features were being implemented, but as more backend-specific features are added from here on, the boundary between the unified API and the backend-specific API easily becomes unclear.

Note that since the first version of this document was written, the following mechanisms have gone into `master`. The design for making it three layers may take them as given.

- Build-time enforcement of the ABI blacklist (`tools/ci/check-abi-blacklist.lisp`). It fails when a backend imports an unstable C function. It also reports import names that cannot be resolved.
- Upstream drift detection (`tools/check-upstream.lisp`). It compares **only the functions a backend imports** against the vendored headers. Comparing the headers as a whole makes both upstreams look unstable, but restricted to the functions used there are 0 breaking changes across LightGBM v3.0.0-v4.7.0 and XGBoost v2.0.0-v3.3.0.
- A record of the supported version range and the `untested-backend-version` warning (`src/version.lisp`). It keeps verified and inferred distinct. Only XGBoost can be cross-checked at runtime.
- An exhaustiveness check for float trap masking (`tools/ci/check-float-traps.lisp`).
- An exhaustiveness assertion for `src/all.lisp`'s re-export list.
- A CI version matrix. It runs only on push-to-master and weekly.

This measurement also shows that **the wider the unified API is made, the more unstable it becomes, and the more the functions used are kept to a stable subset, the more portability rises**. Making it three layers is positioned as fixing that property into the structure.

In particular, the following problems need to be avoided.

- Losing the original shape or metadata in order to fit the unified API's return value.
- Making a feature that only one backend has look as though it exists on the other.
- Silently ignoring an argument that cannot be supported.
- The caller handling raw foreign pointers in order to use a backend-specific feature.
- The unified API and the backend-specific API implementing the same C operation separately, so that a fix is duplicated.

## 3. Basic policy

cl-gbdt adopts the following three-layer structure.

### Layer 0: Raw FFI binding

This is the generated CFFI binding, corresponding almost one-to-one to LightGBM's and XGBoost's C API.

Examples:

- `cl-gbdt/src/lightgbm/c-api`
- `cl-gbdt/src/xgboost/c-api`

This layer is an internal implementation, and shall not be a stable public API.

In Layer 0, the C API's function names, pointers, out parameters, buffer lengths, error codes and so on are as a rule kept in a form close to the original. The generated code must not be hand-edited. A change is made through the generation path: the binding generator, the vendored headers, the ABI blacklist and the like.

### Layer 1: Backend-specific safe API

This is the layer that converts LightGBM-specific or XGBoost-specific features into a form a Common Lisp caller can call safely.

The candidate public packages shall be the following.

- `cl-gbdt/lightgbm`
- `cl-gbdt/xgboost`

However, check for collisions with ASDF system names and for dependencies under package-inferred-system; if necessary, the internal packages may be split as follows.

- `cl-gbdt/src/lightgbm/native`
- `cl-gbdt/src/lightgbm/protocol`
- `cl-gbdt/src/lightgbm/all`
- `cl-gbdt/src/xgboost/native`
- `cl-gbdt/src/xgboost/protocol`
- `cl-gbdt/src/xgboost/all`

A public package may re-export the internal native package, but must not re-export the raw C API package.

Layer 1 may be close to the C API in feature granularity, but must not publish even C's calling conventions as they are. Process the following for Lisp.

- Convert out parameters into ordinary return values or multiple values.
- Copy foreign buffers into Lisp objects, and make their lifetime explicit.
- Convert error codes into the existing condition hierarchy.
- Accept the existing backend / dataset / booster handles rather than raw pointers.
- Verify the handle's released state and the backend's open state.
- Free a foreign resource whose ownership has been taken, on every abnormal exit path.
- Preserve the floating-point trap masking SBCL requires.
- Do not lose the metadata upstream returned — shape, feature names, metric names and the like.

Examples of Layer 1's API are envisaged to be something like the following.

```lisp
(cl-gbdt/xgboost:booster-slice booster :begin 0 :end 50)
(cl-gbdt/xgboost:feature-score booster :kind :total-cover)
(cl-gbdt/xgboost:evaluate-one-iteration booster datasets)

(cl-gbdt/lightgbm:rollback-one-iteration booster)
(cl-gbdt/lightgbm:reset-parameters booster parameters)
(cl-gbdt/lightgbm:refit booster dataset)
```

These are examples showing a direction, and the API must not be settled merely by converting an upstream C name to a Lisp name mechanically. Define the contract first, including the return value, ownership, conditions and shape.

### Layer 2: Unified portable API

This is the high-level API published from the `cl-gbdt` package, portable between LightGBM and XGBoost. The current `cl-gbdt/src/protocol` is the centre of this layer.

Each Layer 2 method delegates to Layer 1's backend-specific safe API as far as possible. Layer 2 and Layer 1 must not implement the same C API operation separately.

Conceptually, the dependency direction shall be the following.

```text
cl-gbdt portable API
        |
        v
backend-specific safe API
        |
        v
generated raw CFFI binding
```

The dependency direction must not be reversed. In particular, the core `cl-gbdt` system must not, as it does not today, depend on a specific backend system or on a shared library.

## 4. Criteria for inclusion in the unified API

A feature may be added to Layer 2 only when it satisfies one of the following.

1. Equivalent meaning and lifecycle can be guaranteed on both backends.
2. It can be implemented directly on one and, on the other, by a safe emulation that does not change the meaning.
3. It is worth publishing as an optional capability, and the behaviour when it is unsupported can be made explicit with a typed condition.

Where the following conditions apply, a feature must not be forced into Layer 2.

- The meaning of a value differs from backend to backend even under the same keyword.
- Converting to a common format loses axis, shape, metadata or precision.
- Implementing it requires a groundless aggregation, approximation or interpolation.
- There is no way to implement it other than ignoring it on one of the backends.
- Only the API name is common, but the effect the caller expects differs.

In that case, put it in Layer 1, or design a common result type that carries more information.

## 5. No loss of information

Information the upstream API returns must not be silently discarded for the sake of unification.

The following in particular shall be mandatory rules.

- When converting a multidimensional prediction into `(nrow flattened-width)`, also return metadata from which the original shape can be recovered.
- When aggregating a multiclass feature importance into a single value, prove that it is the aggregation upstream defined. An aggregation invented here must not be used.
- Do not reinterpret an XGBoost-specific importance kind as a common kind that means something different.
- Converting from single-float to double-float does not increase precision, so make the reason for the conversion and its cost explicit in the contract.
- Treat a model's string representation as an opaque string, and do not assume it is the same format across backends.

The existing `predict`'s return-value contract must not be broken immediately. Where shape preservation is needed, choose one of the following in design review.

- Keep the existing `predict`, and add a new API that preserves the shape.
- The primary value shall be the existing array, and the secondary value the shape metadata.
- Add a prediction result object, and migrate in stages.

In making that choice, give priority to backward compatibility.

## 6. Parameter handling

Keep `:parameters` at training time as an escape hatch that does not lose backend-specific settings. Translating every parameter into a common vocabulary must not be attempted.

However, distinguish the following clearly.

- portable parameter: one whose meaning can be guaranteed to be the same on both backends.
- backend parameter: one passed through to the backend transparently, with no guarantee of portability.
- dataset construction option: one that controls the form of the C API call itself.

Even a dataset option that is valid on the backend side, such as XGBoost's `missing`, `nthread` and `data_split_mode`, must not be rejected wholesale. Consider a scheme that accepts only recognisable keys and rejects an unknown key, or a key whose meaning differs, with `unsupported-argument`.

An implementation must not silently ignore an argument, nor depend on the C library silently discarding an unknown key.

## 7. Capability model

Implement the currently reserved `backend-capabilities`, and make at least the following queries possible.

```lisp
(backend-supports-p backend :sparse-input)
(backend-supports-p backend :evaluation-history)
(backend-supports-p backend :early-stopping)
(backend-supports-p backend :model-slicing)
(backend-supports-p backend :multidimensional-feature-score)
```

A capability is not merely a fixed table based on the backend name; it reflects the following as needed.

- Backend kind
- Runtime version
- Foreign symbols that could actually be resolved
- Build options and platform-dependent features

However, a capability query is not a substitute for verification at the actual call. At the feature's call site as well, always signal a typed condition when it is unsupported. Where a capability is false, there must not be a silent fallback to another feature.

## 8. Library availability and symbol probing

At the backend's initialization, check every foreign symbol its implementation requires.

At present, `*required-symbols*` in the XGBoost backend has **2** omissions. The first version of this document listed only `XGDMatrixSetUIntInfo`, but a mechanical cross-check gives the following.

```
lightgbm: import 18 / required 18 / omissions 0
xgboost:  import 20 / required 18 / omissions 2
    XGDMatrixSetUIntInfo      (sets the ranking group)
    XGBoosterGetNumFeature    (used to make feature importance a dense vector)
```

This document missed one because, at the time it was written, it did not have the mechanical check it itself requires in §8. Fix this inconsistency first. And the miss itself is evidence of the need to add the check.

Furthermore, to prevent a recurrence, add a mechanical check that verifies the correspondence between the foreign calls the backend source references and `*required-symbols*`. Even where a complete call graph analysis is difficult, check at least the following.

- A raw C function called from backend-specific source is classified as a required symbol or as an optional capability symbol.
- Where a required symbol is absent, `open-backend` signals `missing-foreign-symbols`.
- Where an optional symbol is absent, only the capability concerned is disabled, rather than the whole backend being made impossible to open.

LightGBM has no runtime version API, so a version must not be inferred and guaranteed. This asymmetry is already stated explicitly in `src/version.lisp` and the README.

The connection of the XGBoost version to the tested-version warning is **implemented** (`untested-backend-version`). The remaining task is the connection to capability decisions; design it together with §7.

## 9. Evaluation and early stopping

The next caller-facing features to prioritise are validation metric, evaluation history, best iteration and early stopping.

The current `valid-sets` registers or caches a dataset with the foreign backend, but the evaluation values cannot be obtained from the unified API. Complete this.

As a plan that preserves backward compatibility, the first candidate shall be a scheme in which `train`'s primary value is the booster as before and a training report is returned as the secondary value.

```lisp
(multiple-value-bind (booster report)
    (train backend dataset
           :valid-sets valid-sets
           :num-rounds 1000
           :early-stopping-rounds 30)
  ...)
```

A training report shall be able to express at least the following.

- The dataset name
- The metric name
- The value at each iteration
- The best iteration
- The best score
- Whether early stopping occurred

Base the determination of a metric's direction not on inference from the string name, but on information the backend provides or an explicit choice by the caller. When a callback API is added as well, do not force a backend-specific callback into the same function signature; separate the portable event object from the backend-specific extension.

## 10. Resource safety

Publishing Layer 1 must not weaken the current resource safety.

- Use the existing handle hierarchy for a dataset / booster.
- Hold a strong reference while the booster needs a training / validation dataset.
- Do not make a foreign call after the backend has been closed.
- A double free is a no-op.
- Warn about a resource that cannot be freed after backend close, and mark the Lisp-side handle released.
- A finalizer is a safety net; do not make it the primary path for a foreign free.
- Provide a `with-*` macro, or an equivalent explicit ownership API, for a new foreign resource.
- Ensure an error during cleanup does not overwrite the original condition.

As a rule, do not add an escape hatch that returns a raw pointer. Where it is genuinely necessary, limit it to an unstable, non-public package, and prefer a callback form that restricts the pointer's lifetime to dynamic extent.

## 11. Package and system boundaries

Keep the following.

- The `cl-gbdt` core can be loaded without a backend shared library.
- `cl-gbdt/lightgbm` and `cl-gbdt/xgboost` can be loaded independently.
- One backend system does not depend on the other backend system.
- Raw C API symbols are not re-exported from the `cl-gbdt` package.
- A backend-specific public symbol is published only from `cl-gbdt/lightgbm` or `cl-gbdt/xgboost`.
- The dependency declarations that let each package-inferred-system leaf load alone are kept.

Even where the placement of the unified API's methods is separated out, do not create an implicit ordering that depends only on the result of having loaded the methods. Make the existing leaf-system check follow the new source structure.

## 12. Phased implementation plan

### Phase 0: Fix the known availability inconsistency

1. Add `XGDMatrixSetUIntInfo` and `XGBoosterGetNumFeature` to XGBoost's `*required-symbols*`.
2. Add an exhaustiveness check for the required symbols. `tools/ci/check-abi-blacklist.lisp` already "reads a backend's import-from clauses and maps them to C names through c-api.lisp's defcfun", so it is enough to point its cross-check at `*required-symbols*`. No new mechanism is needed.
3. Adopt the check after confirming where it fails. This repository has twice shipped a check whose failure it had not seen.

The document drift in the README's assertion counts is **already resolved** (243 / 106, matching the current state), so it shall be out of scope for this phase.

This phase does not change the API structure.

### Phase 1: Separate responsibilities without changing behaviour

1. Extract the backend-specific functions that make raw calls safe.
2. Make the existing unified generic methods delegate to those functions.
3. Keep the existing public API, return values, conditions and test results.
4. Apply the same separation principle on both LightGBM and XGBoost.

The aim of this phase is not a new feature but the establishment of the three-layer structure with a single implementation path.

### Phase 2: Publish the backend-specific safe API

1. Define the contract of the functions to be published.
2. Export from the backend-specific package.
3. Do not publish raw pointers; use the existing handles.
4. Add documentation and backend-specific functional tests.
5. Implement the capability model.

Choose the first things to publish from the features that are clearly unavailable at present because of unification.

- XGBoost model slicing
- shape-preserving XGBoost feature score
- XGBoost evaluation
- LightGBM evaluation
- LightGBM rollback / refit / reset parameter

### Phase 3: Complete the unified training API

1. Obtain the evaluation history.
2. Design the training report.
3. Implement early stopping.
4. Connect the best iteration to prediction / persistence.
5. Define validation set naming.

### Phase 4: Data and prediction extensions — complete

Add the following as capability-gated APIs, according to priority and demand from real use.

- sparse input
- missing value option
- categorical metadata
- multidimensional prediction result
- custom objective / evaluation

Implementing all of them at once must not be attempted. For each feature, implement the Layer 1 contract, whether it is to be included in Layer 2, the capability and the functional test as one set.

This one-set unit was kept to the end, and each item in the list went in as its own separate PR. External memory was dropped from the list for the reason below, so **at this point Phase 4 has no unimplemented item left**.

| Item in the list | Implementation | capability | PR | functional test |
|---|---|---|---|---|
| sparse input | `make-dataset` and `predict` also accept a `csr-matrix` wherever they accept a dense matrix | `:sparse-input` (both backends) | #18 | `tests/functional/sparse-input.lisp` |
| missing value option | `:missing` on `make-dataset` and `predict` | `:missing-value` (true on XGBoost only) | #19 | `tests/functional/missing-value.lisp` |
| categorical metadata | `make-dataset`'s `:categorical-features` | `:categorical-features` (both backends) | #20 | `tests/functional/categorical-features.lisp` |
| multidimensional prediction result | the shape `predict` returns as its secondary value | `:prediction-shape` (both backends) | #22 | `tests/functional/prediction-shape.lisp` |
| custom objective / evaluation | `train`'s `:objective` and `:evaluation` | `:custom-objective` / `:custom-evaluation` (both backends) | #23 / #24 | `tests/functional/custom-objective.lisp` / `custom-evaluation.lisp` |

Do not add any further item to this list. Treat a new data / prediction feature as part of the Follow-up below rather than as a continuation of the phases.

### Why external memory was dropped from the list

The list originally included external memory as well, but reading the vendored headers showed that **neither backend provides, at the corresponding dataset-construction entry point, anything this wrapper could call external memory**, so it was dropped. The grounds are recorded here so that the same investigation is not repeated.

- **XGBoost**'s external memory is in the `Streaming` group. The header itself says "the experimental external-memory-based DMatrix, which reads data in batches during training", and to reach it the caller has to pass `XGDMatrixCreateFromCallback` or `XGExtMemQuantileDMatrixCreateFromCallback` a **data iterator implemented as a C callback** (`XGDMatrixCallbackNext`, `DataIterResetCallback`). `XGDMatrixCreateFromURI` is "load a data matrix", and makes an ordinary in-memory DMatrix.
- **LightGBM** has no corresponding feature. `LGBM_DatasetCreateFromFile` is "Load dataset from file (like LightGBM CLI version does)", and the `Dataset` it produces sits entirely in memory in binned form. `LGBM_DatasetInitStreaming` and what surrounds it are a mechanism for **construction**, feeding rows in from multiple threads, not a mechanism for putting the data outside memory.

So implementing it would mean introducing this project's first C→Lisp callback, riding on an API upstream itself calls experimental, and adding a capability that is true on only one of the backends. That is not a form that can be put on the portable contract §7 requires. To overturn this decision in the future, first confirm that the three points above have changed.

### Why file input was not put on the unified API

Both libraries' own file-reading entry points -- `LGBM_DatasetCreateFromFile` and `XGDMatrixCreateFromURI` -- were measured against the vendored LightGBM v4.7.0 and XGBoost v3.3.0 on Linux aarch64, over the same four-row fixture written as libsvm and as CSV. The transcripts are `docs/superpowers/specs/2026-08-13-file-input-measurements.md`; every figure below is one of them, not a restatement of an earlier design document. Six findings decided the shape.

| | Finding |
|---|---|
| 1 | **`:format` has no shared meaning.** XGBoost's `format` query key is mandatory for every text file and is checked before the file is even opened (record §5). LightGBM has no `format` key in its `parameters` vocabulary at all -- it infers CSV, TSV or libsvm from the file's own content, and refuses what it cannot infer with `Unknown format of training data. Only CSV, TSV, and LibSVM (zero-based) formatted text files are supported.` (record §5). |
| 2 | **The same headerless CSV reads differently.** LightGBM reads it 4 rows × 3 features, with the labels -- `label_column` defaults to 0 (record §5). XGBoost reads the identical file 4 × 4 with **no labels**, the label column read as a feature, unless `label_column=0` is added to the URI, which then reads 4 × 3 with labels (record §5). |
| 3 | **A header row needs `header=true` on LightGBM, and fails without it.** `header.csv` fails `Unknown token label in data file` until `header=true` is passed, and reads correctly once it is (record §5). Every `XGDMatrixCreateFromURI` config JSON in the record carried exactly two keys, `uri` and `silent` -- no header key ever appeared there. |
| 4 | **XGBoost does not check `format` against the file's contents, and the wrong direction is fatal.** `train.csv?format=libsvm`, and a binary DMatrix declared `?format=libsvm`, both **SIGSEGV** inside a thread dmlc creates for the parse -- outside any Lisp stack, so no `handler-case` anywhere can catch it (record §4). The other direction is silently wrong rather than fatal: `train.libsvm?format=csv` returns status 0, a 4-row **1-column** DMatrix, and no label (record §4). |
| 5 | **XGBoost is retiring the path.** Every text-file attempt, including a refused one, prints once per process to stderr: `WARNING: .../data.cc:963: Text file input has been deprecated since 3.1` (record §6), measured on the vendored 3.3.0. |
| 6 | **LightGBM's `parameters` and `reference` have no XGBoost counterpart.** LightGBM's file constructor also takes a `reference` dataset whose bin mapper this one aligns to -- a `NULL` reference confirmed to mean "build its own" (record §7) -- and every `XGDMatrixCreateFromURI` config observed, `uri` and `silent` only, has no argument that plays either role. |

Finding 4 is the one that decides the layer. A unified API with a required `:format` argument would let one caller's typo end the process, in a thread no Lisp handler can reach. The only case where the two libraries agreed exactly was **libsvm read with defaults**: LightGBM's `train.libsvm` with no parameters, and XGBoost's `train.libsvm?format=libsvm`, both read the same fixture as **4 rows × 4 features**, with identical labels (record §2, §5) -- the fixture's libsvm indices are 1, 2 and 3, and both libraries read libsvm indices as zero-based, so column 0 exists on both and is empty. Even that one point of exact agreement still prints the deprecation warning on XGBoost.

So: **two backend-specific Layer 1 functions, and no `:file-input` capability.** A capability name implies a portable contract behind it, and the measurement says there is none. `LGBM_DatasetCreateFromFile` is published as `cl-gbdt/lightgbm:create-dataset-from-file`, taking no format argument since LightGBM needs none. `XGDMatrixCreateFromURI` is published as `cl-gbdt/xgboost:create-dataset-from-file`, taking a **required** FORMAT argument and a wrapper-side check, built on `detect-file-format`, that refuses a declared format disagreeing with the file's own contents -- `file-format-mismatch` -- before the foreign call finding 4 shows is otherwise fatal in one direction and silently wrong in the other. `make-dataset` gains nothing.

### Follow-up

This records the outstanding items that came to light at the point Phase 4 completed and in the Layer 1 / Layer 2 separation work that followed. None of them is a completion criterion for any phase. Do not bundle them together into a single phase. From the second item on, take them up one at a time, at the point demand from real use appears, following §17's classification and §13's test policy.

- **Dataset / booster construction, model persistence and metadata queries in Layer 1 — complete** — Unlike the other three, this was not a new feature but debt the Layer 1 / Layer 2 split left behind. So it was taken up without waiting for demand from real use, as the first item of the next stage of the layer split, and closed. Both backends have placed six operations — `create-dataset`, `create-booster`, `update-one-iteration`, `predict`, `free-dataset`, `free-booster` — in `src/<backend>/api.lisp`, so a caller that has loaded only `cl-gbdt/lightgbm` or only `cl-gbdt/xgboost` can build a dataset, make a booster on top of it, advance it one iteration at a time, predict, and free both. That this holds with no unified API present in the image at all is shown by `tests/functional/lightgbm-standalone.lisp` and `tests/functional/xgboost-standalone.lisp`. What both files name is that backend's public package alone, and they list not one other system of this project (`rove` aside, there is no other declaration). Because `tools/ci/check-leaf-systems.lisp` loads each system alone in a separate fresh process, this claim is backed by the build rather than by prose. §3's requirement, "Each Layer 2 method delegates to Layer 1's backend-specific safe API as far as possible", was met by putting into `src/<backend>/api.lisp` on both backends the remaining seven operations — `save-model`, `load-model`, `model-to-string`, `feature-importance`, `evaluation`, `dataset-num-rows`, `dataset-num-features` — for which Layer 1 had no counterpart at that point, so that all twelve of the 13 methods other than `train` delegate their whole procedure to them. `train` delegates its construction only: it calls the corresponding Layer 1 `create-booster` to build the booster, but the loop itself keeps calling `native.lisp`'s functions directly rather than going through `api.lisp`'s `update-one-iteration` — so as not to have every handle re-checked on every iteration, and that reason is recorded in the comment at the `create-booster` call site in each backend's `train`. After the loop ends it writes only the best iteration, through `src/handle.lisp`'s internal writer `%set-booster-best-iteration`. The training report, early stopping and `train`'s `:objective` and `:evaluation` are Layer 2-specific concepts and have no corresponding Layer 1 operation, so they do not fall under this delegation.
- **file input — complete** — `LGBM_DatasetCreateFromFile` and `XGDMatrixCreateFromURI` are published as `create-dataset-from-file` on `cl-gbdt/lightgbm` and `cl-gbdt/xgboost` respectively. Both make an ordinary in-memory dataset, so this remains distinct from external memory above. What this bullet originally proposed -- `make-dataset`'s `MATRIX` also accepting a pathname, gated on a `:file-input` capability -- was designed, then measured against the vendored libraries, and abandoned: see "Why file input was not put on the unified API" above for the six findings that decided it, foremost among them that XGBoost does not check its own `format` argument against a file's contents, and a wrong declaration in one direction SIGSEGVs the process outside any Lisp handler. `make-dataset` gains nothing; the two backend-specific functions above are the whole of what was built.
- **shape-preserving XGBoost feature score** — listed under Phase 2's "the first things to publish" and still unimplemented. `:multidimensional-feature-score` is registered in `*known-capabilities*` but is false on every backend, stopped at the state where "the fact that it is unsupported can itself be answered".
- **LightGBM rollback / refit / reset parameter** — likewise an unimplemented item from Phase 2's list. `LGBM_BoosterRollbackOneIter`, `LGBM_BoosterRefit` and `LGBM_BoosterResetParameter` exist in the binding and are not published as Layer 1.

## 13. Test policy

Separate tests clearly into the following two kinds.

### Portable contract tests

The same test is applied to both backends, verifying that the unified API means the same thing on each.

At a minimum, verify the following against the real library.

- label / weight / group / feature names
- normal / raw / leaf-index / contribution prediction
- multiclass shape
- `num-iteration`
- save / load / model-to-string
- split / gain importance
- validation metric
- early stopping
- released handle / closed backend / wrong backend handle
- that an unsupported capability becomes a typed condition

Exact numerical agreement between backends is not required. Verify the properties that correspond to the contract, such as shape, ordering, meaning, monotonicity and reproducibility within the same backend after reloading.

### Backend-specific tests

For a backend-specific API, verify that what it returns does not lose the characteristics of the upstream C API.

- XGBoost multidimensional shape
- XGBoost-specific importance kind
- model slicing
- LightGBM reference dataset
- refit / rollback
- capability degradation when an optional symbol is absent

A new public function must have a functional test that calls the real shared library, not merely a mock.

## 14. Backward compatibility

A change that breaks existing callers must not be introduced all at once.

- The existing generic function names and primary return values must be kept.
- A condition must not be degraded into a plain `error` or an implementation-dependent CFFI error.
- An existing keyword must not be changed to a different meaning.
- Where the shape of a return value is changed, a new API or a staged migration must be used.
- The unified API must not be deleted on the grounds that a backend-specific API is being added.
- The raw C API package must not be promised as a stable public API.

Where a breaking change is unavoidable, a design document, a migration example and a deprecation period must be presented first, and the change must be handled as a separate PR.

## 15. Non-goals

The following are not goals of this policy.

- Translating every LightGBM and XGBoost parameter to a common name.
- Making the training results of the two backends agree numerically.
- Immediately publishing every function of the upstream C API as a high-level API.
- Making the raw CFFI API a stable API for ordinary callers.
- Reproducing by simulation, on one backend, a concept that only the other has.
- Overhauling every layer wholesale in a single PR.

## 16. Completion criteria

The initial completion of the layered API restructuring shall be the point at which all of the following are satisfied.

1. The dependency direction among raw FFI, the backend-specific safe API and the unified API is clear in the source structure.
2. The existing unified generic methods delegate to the backend-specific safe API.
3. The unified API and the backend-specific API do not implement the same C operation separately.
4. At least one backend-specific feature can be used from the backend-specific package without a raw pointer.
5. `backend-capabilities` returns the actual availability of features.
6. A classification and a mechanical check of required / optional foreign symbols exist.
7. All the existing layer 1 / layer 2 tests pass.
8. A new backend-specific API has a functional test that uses the real shared library.
9. The core system keeps the property that it can be loaded without a backend library.
10. The public API and the package boundary are described in the README or in a dedicated document.

## 17. Final instructions to the implementing agent

Before you begin implementing, classify the feature you are changing into one of the following.

- raw FFI concern
- backend-specific safe API
- unified portable API
- optional capability

You must not add a keyword to a unified generic while the feature remains unclassified.

Also, when you present an implementation proposal, you must always explain the following.

1. Which Layer to put that feature in.
2. Whether it means the same thing on both backends.
3. Whether it loses shape or metadata.
4. Which condition it signals on an unsupported backend / version.
5. Who holds the resource ownership.
6. How the existing API delegates to it.
7. Which functional test proves the contract.

When in doubt, choose to keep the complete information as a backend-specific safe API, rather than putting it into the unified API and reducing the information. After that, promote it to the unified API once you have confirmed a real use case and the meaning on both backends.

## 18. Layer 1 standalone-library programme (S1–S5)

Since 2026-08-11, a five-stage programme from S1 to S5 has been under way, to make
`cl-gbdt/lightgbm` and `cl-gbdt/xgboost` into independent libraries good enough for practical use
without loading the unified API (`cl-gbdt/lightgbm/unified`, `cl-gbdt/xgboost/unified`) at all.
Until now this definition existed only in
`docs/superpowers/specs/2026-08-11-layer1-standalone-design.md`, and because that directory is
covered by `.gitignore`, it had never once been recorded in a tracked document. This section is
that record.

This programme is a separate track from §12's phased plan (Phase 0–4), and began after Phase 4
completed. Where Phase 2 made "export from the backend-specific package" its subject and Phase 4
"data and prediction extensions", this programme makes its subject Layer 1 being self-sufficient
as a library on its own, without the unified API. Do not add items to §12's Phase list.

The decisions binding the whole programme are the following five.

1. Coverage is guaranteed by classification (the exhaustiveness of the classification), and is
   not guaranteed by percentage.
2. `cl-gbdt/<backend>` means Layer 1 alone. The unified API's 13 methods are carried by
   `cl-gbdt/<backend>/unified`.
3. It shall be one C function, one Lisp function. The C calling conventions are converted (out
   parameters into return values or multiple values, foreign buffers into Lisp objects, error
   codes into typed conditions), and where the C API provides init and free as a pair, a `with-*`
   macro is provided as well.
4. The docstring shall be the primary source of documentation. The API reference is generated from
   the docstrings and checked byte-for-byte, in the same way as `src/*/c-api.lisp`.
5. A public symbol is required to have a functional test that is checked mechanically.

The state of each stage is as follows.

- **S1 — separating the layers. Complete (PR #26).** The concrete classes and the library lifecycle
  (`initialize-backend`/`shutdown-backend`) were moved into Layer 1, `cl-gbdt/<backend>` was made a
  Layer 1-only system, and the ownership patterns were consolidated into a single
  `with-pointer-ownership`.
- **S2 — moving each operation's procedure into Layer 1. Complete (PR #27, #29, #30).** PR #27
  moved six operations on each backend (`create-dataset`, `create-booster`,
  `update-one-iteration`, `predict`, `free-dataset`, `free-booster`), PR #29 the remaining seven
  operations (`save-model`, `load-model`, `model-to-string`, `feature-importance`, `evaluation`,
  `dataset-num-rows`, `dataset-num-features`), and PR #30 `train`'s booster construction, each
  into `src/<backend>/api.lisp` on both backends. As a result, all 13 unified methods delegate at
  least part of their procedure to Layer 1. The reason the degree of delegation differs for
  `train` alone from the other 12 (the boosting loop itself does not go through
  `update-one-iteration`) is already recorded in the Follow-up above, and is not repeated here.
- **S3 — classification of the bindings. Complete (PR #31).** All 177 bindings that
  `src/*/c-api.lisp` generates are classified in `ffi-spec/BINDING-COVERAGE.md` as one of
  `wrapped`/`planned`/`excluded`, and `tools/ci/check-binding-coverage.lisp` fails the build on an
  unclassified binding.
- **S4-1 — a mechanism that generates the API reference from the docstrings and checks it
  byte-for-byte. Complete (PR #33).** A development-only emitter `src/docgen/`
  (`introspect.lisp`, `render.lisp`, `emit.lisp`, `all.lisp`), which introspects the loaded image
  and writes out Markdown, was added as the ASDF system `cl-gbdt/docgen`, and its driver
  `tools/gen-api-reference.lisp` generated a `docs/API-REFERENCE.md` covering all 174 symbols
  (141/88/89) that the three public packages `cl-gbdt`, `cl-gbdt/lightgbm` and `cl-gbdt/xgboost`
  export. `tools/ci/check-api-reference.lisp` checks in four stages: the existence of the
  introspection primitives, byte-for-byte agreement between the generated result and the committed
  file, a documentation floor over every public symbol (including the slots of classes and
  conditions), and a per-package export count floor. To satisfy this documentation floor,
  `:documentation` was added to twelve condition slots in `src/conditions.lisp`.
- **S4-2 — a mechanism that checks mechanically that a functional test exists for every public
  symbol. Complete (PR #34).**
  `docs/FUNCTIONAL-COVERAGE.md` gives a position to all 174 symbols that the three public packages
  `cl-gbdt`, `cl-gbdt/lightgbm` and `cl-gbdt/xgboost` export. `covered` is not written down in the
  file — `tools/ci/check-functional-coverage.lisp` reads each file in `tests/functional/*.lisp`
  and derives `covered` afresh each time, from the symbol name appearing in a top-level form other
  than the package form. This, like `wrapped` in `ffi-spec/BINDING-COVERAGE.md`, is a design for
  not holding a record that is maintained in two places. The remaining symbols are classified by
  hand into either `## Unproven` — symbols that should have a functional test and do not — (29 of
  them), or the group of `## Exempt` headings — one independent heading per reason — (58 of them). The
  checker checks both the floor on `covered` (`+minimum-covered+` = 87) and the ceiling on
  `unproven` (`+maximum-unproven+` = 29), and the latter works as a ratchet: if public symbols
  without a functional test increase, `unproven` goes above 29, and the build fails unless an edit
  raises that constant inside `tools/ci/check-functional-coverage.lisp`. However, what this
  mechanism guarantees is a "recorded position" per symbol, not a "proven contract". An
  `## Exempt` heading is recognised only by a prefix match — the literal `## Unproven`, and any
  heading beginning with `## Exempt` — so merely raising a new `## Exempt: ...` heading, or adding
  a line to one of the five existing `## Exempt` headings, is enough to house an untested symbol
  without moving either `covered` or `unproven` — what this mechanism actually prevents is "being
  published without a classification", and whether each `## Exempt`'s reason is legitimate is in
  the end for a reviewer to judge by reading the prose. The classification work itself made plain
  the amount of work involved: that 29 of the 174 symbols have no functional test.
- **S5 — in progress.** Publish the C functions nobody has published yet. The work list was the
  `## Planned` section of `ffi-spec/BINDING-COVERAGE.md`. Both backends' file input rows,
  `LGBM_DatasetCreateFromFile` and `XGDMatrixCreateFromURI`, are gone from that section now that
  both are wrapped, as `create-dataset-from-file` -- `check-binding-coverage.lisp`'s CHECK C fails
  the build on a name that is both wrapped and classified, so the rows were deleted rather than
  corrected. See "Why file input was not put on the unified API" above for the measurement that
  decided their shape, and the Follow-up's file-input item above for the record that this half of
  S5 is complete. Still `## Planned`, and still S5's remaining work: LightGBM's
  `LGBM_BoosterRollbackOneIter`/`LGBM_BoosterRefit`/`LGBM_BoosterResetParameter`. The Follow-up's
  "shape-preserving XGBoost feature score" is a separate thing. `XGBoosterFeatureScore` itself is
  already wrapped, and this item is an implementation problem of changing how an existing binding
  returns its result, so it does not appear in the list of bindings S5 should publish.

---

Target repository: <https://github.com/masatoi/cl-gbdt>

Basis commit (updated 2026-08-06): [`59d1979`](https://github.com/masatoi/cl-gbdt/commit/59d1979) (PR #9 merge)
