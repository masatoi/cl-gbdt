# API Reference

This file is GENERATED from the docstrings of every symbol `cl-gbdt`, `cl-gbdt/lightgbm` and
`cl-gbdt/xgboost` export. **Never edit it by hand** -- the same rule `src/*/c-api.lisp`
carries. Regenerate it with:

```
ros run -- --non-interactive --load tools/gen-api-reference.lisp
```

`tools/ci/check-api-reference.lisp` regenerates it into a temporary file and fails the build
when the committed copy differs, so a docstring edited without regenerating is caught rather
than silently shipped.

Docstrings appear verbatim inside `text` fences: they are hand-wrapped for the REPL, and they
hold Lisp conventions Markdown would render as something else.

## Packages

### `cl-gbdt` -- 145 symbols

- [`*known-capabilities*`](#cl-gbdt-known-capabilities)
- [`*lightgbm-version-range*`](#cl-gbdt-lightgbm-version-range)
- [`*xgboost-version-range*`](#cl-gbdt-xgboost-version-range)
- [`backend`](#cl-gbdt-backend)
- [`backend-capabilities`](#cl-gbdt-backend-capabilities)
- [`backend-error`](#cl-gbdt-backend-error)
- [`backend-error-backend`](#cl-gbdt-backend-error-backend)
- [`backend-info`](#cl-gbdt-backend-info)
- [`backend-library-load-failed`](#cl-gbdt-backend-library-load-failed)
- [`backend-library-load-failed-cause`](#cl-gbdt-backend-library-load-failed-cause)
- [`backend-library-load-failed-path`](#cl-gbdt-backend-library-load-failed-path)
- [`backend-library-not-found`](#cl-gbdt-backend-library-not-found)
- [`backend-library-not-found-searched`](#cl-gbdt-backend-library-not-found-searched)
- [`backend-library-path`](#cl-gbdt-backend-library-path)
- [`backend-methods-not-loaded`](#cl-gbdt-backend-methods-not-loaded)
- [`backend-methods-not-loaded-generic-function`](#cl-gbdt-backend-methods-not-loaded-generic-function)
- [`backend-name`](#cl-gbdt-backend-name)
- [`backend-not-open`](#cl-gbdt-backend-not-open)
- [`backend-open-p`](#cl-gbdt-backend-open-p)
- [`backend-supports-p`](#cl-gbdt-backend-supports-p)
- [`backend-version`](#cl-gbdt-backend-version)
- [`booster`](#cl-gbdt-booster)
- [`booster-best-iteration`](#cl-gbdt-booster-best-iteration)
- [`booster-training-set`](#cl-gbdt-booster-training-set)
- [`booster-validation-sets`](#cl-gbdt-booster-validation-sets)
- [`call-with-foreign-matrix`](#cl-gbdt-call-with-foreign-matrix)
- [`capability-unavailable`](#cl-gbdt-capability-unavailable)
- [`capability-unavailable-capability`](#cl-gbdt-capability-unavailable-capability)
- [`check-backend-version`](#cl-gbdt-check-backend-version)
- [`check-foreign-call`](#cl-gbdt-check-foreign-call)
- [`close-backend`](#cl-gbdt-close-backend)
- [`csr-matrix`](#cl-gbdt-csr-matrix)
- [`csr-matrix-indices`](#cl-gbdt-csr-matrix-indices)
- [`csr-matrix-indptr`](#cl-gbdt-csr-matrix-indptr)
- [`csr-matrix-num-columns`](#cl-gbdt-csr-matrix-num-columns)
- [`csr-matrix-num-rows`](#cl-gbdt-csr-matrix-num-rows)
- [`csr-matrix-values`](#cl-gbdt-csr-matrix-values)
- [`data-error`](#cl-gbdt-data-error)
- [`dataset`](#cl-gbdt-dataset)
- [`dataset-num-features`](#cl-gbdt-dataset-num-features)
- [`dataset-num-rows`](#cl-gbdt-dataset-num-rows)
- [`dimension-mismatch`](#cl-gbdt-dimension-mismatch)
- [`dimension-mismatch-expected`](#cl-gbdt-dimension-mismatch-expected)
- [`dimension-mismatch-given`](#cl-gbdt-dimension-mismatch-given)
- [`evaluation`](#cl-gbdt-evaluation)
- [`feature-importance`](#cl-gbdt-feature-importance)
- [`file-format-mismatch`](#cl-gbdt-file-format-mismatch)
- [`file-format-mismatch-declared`](#cl-gbdt-file-format-mismatch-declared)
- [`file-format-mismatch-detected`](#cl-gbdt-file-format-mismatch-detected)
- [`file-format-mismatch-path`](#cl-gbdt-file-format-mismatch-path)
- [`find-backend-class`](#cl-gbdt-find-backend-class)
- [`foreign-call-error`](#cl-gbdt-foreign-call-error)
- [`foreign-call-error-code`](#cl-gbdt-foreign-call-error-code)
- [`foreign-call-error-function-name`](#cl-gbdt-foreign-call-error-function-name)
- [`foreign-call-error-message`](#cl-gbdt-foreign-call-error-message)
- [`foreign-element-type`](#cl-gbdt-foreign-element-type)
- [`foreign-matrix`](#cl-gbdt-foreign-matrix)
- [`foreign-matrix-cols`](#cl-gbdt-foreign-matrix-cols)
- [`foreign-matrix-element-type`](#cl-gbdt-foreign-matrix-element-type)
- [`foreign-matrix-pointer`](#cl-gbdt-foreign-matrix-pointer)
- [`foreign-matrix-rows`](#cl-gbdt-foreign-matrix-rows)
- [`free-booster`](#cl-gbdt-free-booster)
- [`free-dataset`](#cl-gbdt-free-dataset)
- [`gbdt-error`](#cl-gbdt-gbdt-error)
- [`handle`](#cl-gbdt-handle)
- [`handle-backend`](#cl-gbdt-handle-backend)
- [`handle-live-pointer`](#cl-gbdt-handle-live-pointer)
- [`handle-pointer`](#cl-gbdt-handle-pointer)
- [`handle-released-p`](#cl-gbdt-handle-released-p)
- [`initialize-backend`](#cl-gbdt-initialize-backend)
- [`load-model`](#cl-gbdt-load-model)
- [`make-csr-matrix`](#cl-gbdt-make-csr-matrix)
- [`make-dataset`](#cl-gbdt-make-dataset)
- [`make-handle`](#cl-gbdt-make-handle)
- [`make-training-report`](#cl-gbdt-make-training-report)
- [`make-training-series`](#cl-gbdt-make-training-series)
- [`make-version-range`](#cl-gbdt-make-version-range)
- [`missing-foreign-symbols`](#cl-gbdt-missing-foreign-symbols)
- [`missing-foreign-symbols-names`](#cl-gbdt-missing-foreign-symbols-names)
- [`missing-training-set`](#cl-gbdt-missing-training-set)
- [`missing-training-set-booster`](#cl-gbdt-missing-training-set-booster)
- [`model-to-string`](#cl-gbdt-model-to-string)
- [`normalize-parameters`](#cl-gbdt-normalize-parameters)
- [`open-backend`](#cl-gbdt-open-backend)
- [`predict`](#cl-gbdt-predict)
- [`probe-capabilities`](#cl-gbdt-probe-capabilities)
- [`probe-foreign-symbols`](#cl-gbdt-probe-foreign-symbols)
- [`register-backend`](#cl-gbdt-register-backend)
- [`release-handle`](#cl-gbdt-release-handle)
- [`released-handle-error`](#cl-gbdt-released-handle-error)
- [`released-handle-error-object`](#cl-gbdt-released-handle-error-object)
- [`resolve-and-load-library`](#cl-gbdt-resolve-and-load-library)
- [`save-model`](#cl-gbdt-save-model)
- [`shutdown-backend`](#cl-gbdt-shutdown-backend)
- [`train`](#cl-gbdt-train)
- [`training-report`](#cl-gbdt-training-report)
- [`training-report-best-iteration`](#cl-gbdt-training-report-best-iteration)
- [`training-report-best-score`](#cl-gbdt-training-report-best-score)
- [`training-report-early-stopped-p`](#cl-gbdt-training-report-early-stopped-p)
- [`training-report-num-rounds`](#cl-gbdt-training-report-num-rounds)
- [`training-report-series`](#cl-gbdt-training-report-series)
- [`training-series`](#cl-gbdt-training-series)
- [`training-series-index`](#cl-gbdt-training-series-index)
- [`training-series-metric`](#cl-gbdt-training-series-metric)
- [`training-series-name`](#cl-gbdt-training-series-name)
- [`training-series-values`](#cl-gbdt-training-series-values)
- [`unfreed-handle-warning`](#cl-gbdt-unfreed-handle-warning)
- [`unfreed-handle-warning-kind`](#cl-gbdt-unfreed-handle-warning-kind)
- [`unknown-backend`](#cl-gbdt-unknown-backend)
- [`unknown-backend-registered`](#cl-gbdt-unknown-backend-registered)
- [`unknown-capability`](#cl-gbdt-unknown-capability)
- [`unknown-capability-capability`](#cl-gbdt-unknown-capability-capability)
- [`unknown-capability-known`](#cl-gbdt-unknown-capability-known)
- [`unsupported-argument`](#cl-gbdt-unsupported-argument)
- [`unsupported-argument-argument`](#cl-gbdt-unsupported-argument-argument)
- [`unsupported-argument-backend`](#cl-gbdt-unsupported-argument-backend)
- [`unsupported-argument-reason`](#cl-gbdt-unsupported-argument-reason)
- [`unsupported-element-type`](#cl-gbdt-unsupported-element-type)
- [`unsupported-element-type-given`](#cl-gbdt-unsupported-element-type-given)
- [`untested-backend-version`](#cl-gbdt-untested-backend-version)
- [`untested-backend-version-backend`](#cl-gbdt-untested-backend-version-backend)
- [`untested-backend-version-tested`](#cl-gbdt-untested-backend-version-tested)
- [`untested-backend-version-version`](#cl-gbdt-untested-backend-version-version)
- [`update-one-iteration`](#cl-gbdt-update-one-iteration)
- [`version-compare`](#cl-gbdt-version-compare)
- [`version-in-range-p`](#cl-gbdt-version-in-range-p)
- [`version-range`](#cl-gbdt-version-range)
- [`version-range-inferred-evidence`](#cl-gbdt-version-range-inferred-evidence)
- [`version-range-inferred-high`](#cl-gbdt-version-range-inferred-high)
- [`version-range-inferred-low`](#cl-gbdt-version-range-inferred-low)
- [`version-range-tested-description`](#cl-gbdt-version-range-tested-description)
- [`version-range-verified-evidence`](#cl-gbdt-version-range-verified-evidence)
- [`version-range-verified-high`](#cl-gbdt-version-range-verified-high)
- [`version-range-verified-low`](#cl-gbdt-version-range-verified-low)
- [`with-booster`](#cl-gbdt-with-booster)
- [`with-dataset`](#cl-gbdt-with-dataset)
- [`with-foreign-float-traps-masked`](#cl-gbdt-with-foreign-float-traps-masked)
- [`with-foreign-matrix`](#cl-gbdt-with-foreign-matrix)
- [`with-pointer-ownership`](#cl-gbdt-with-pointer-ownership)
- [`write-foreign-sequence`](#cl-gbdt-write-foreign-sequence)
- [`wrong-backend-reference`](#cl-gbdt-wrong-backend-reference)
- [`wrong-backend-reference-argument`](#cl-gbdt-wrong-backend-reference-argument)
- [`wrong-backend-reference-backend`](#cl-gbdt-wrong-backend-reference-backend)
- [`wrong-backend-reference-expected`](#cl-gbdt-wrong-backend-reference-expected)
- [`wrong-backend-reference-given`](#cl-gbdt-wrong-backend-reference-given)

### `cl-gbdt/lightgbm` -- 92 symbols

- [`*known-capabilities*`](#cl-gbdt-known-capabilities)
- [`backend-capabilities`](#cl-gbdt-backend-capabilities)
- [`backend-error`](#cl-gbdt-backend-error)
- [`backend-error-backend`](#cl-gbdt-backend-error-backend)
- [`backend-info`](#cl-gbdt-backend-info)
- [`backend-library-load-failed`](#cl-gbdt-backend-library-load-failed)
- [`backend-library-load-failed-cause`](#cl-gbdt-backend-library-load-failed-cause)
- [`backend-library-load-failed-path`](#cl-gbdt-backend-library-load-failed-path)
- [`backend-library-not-found`](#cl-gbdt-backend-library-not-found)
- [`backend-library-not-found-searched`](#cl-gbdt-backend-library-not-found-searched)
- [`backend-library-path`](#cl-gbdt-backend-library-path)
- [`backend-methods-not-loaded`](#cl-gbdt-backend-methods-not-loaded)
- [`backend-methods-not-loaded-generic-function`](#cl-gbdt-backend-methods-not-loaded-generic-function)
- [`backend-name`](#cl-gbdt-backend-name)
- [`backend-not-open`](#cl-gbdt-backend-not-open)
- [`backend-open-p`](#cl-gbdt-backend-open-p)
- [`backend-supports-p`](#cl-gbdt-backend-supports-p)
- [`backend-version`](#cl-gbdt-backend-version)
- [`booster`](#cl-gbdt-booster)
- [`booster-eval`](#cl-gbdt-lightgbm-booster-eval)
- [`booster-eval-names`](#cl-gbdt-lightgbm-booster-eval-names)
- [`booster-training-set`](#cl-gbdt-booster-training-set)
- [`booster-validation-sets`](#cl-gbdt-booster-validation-sets)
- [`capability-unavailable`](#cl-gbdt-capability-unavailable)
- [`capability-unavailable-capability`](#cl-gbdt-capability-unavailable-capability)
- [`close-backend`](#cl-gbdt-close-backend)
- [`create-booster`](#cl-gbdt-lightgbm-create-booster)
- [`create-dataset`](#cl-gbdt-lightgbm-create-dataset)
- [`csr-matrix`](#cl-gbdt-csr-matrix)
- [`csr-matrix-indices`](#cl-gbdt-csr-matrix-indices)
- [`csr-matrix-indptr`](#cl-gbdt-csr-matrix-indptr)
- [`csr-matrix-num-columns`](#cl-gbdt-csr-matrix-num-columns)
- [`csr-matrix-num-rows`](#cl-gbdt-csr-matrix-num-rows)
- [`csr-matrix-values`](#cl-gbdt-csr-matrix-values)
- [`data-error`](#cl-gbdt-data-error)
- [`dataset`](#cl-gbdt-dataset)
- [`dataset-num-features`](#cl-gbdt-lightgbm-dataset-num-features)
- [`dataset-num-rows`](#cl-gbdt-lightgbm-dataset-num-rows)
- [`dimension-mismatch`](#cl-gbdt-dimension-mismatch)
- [`dimension-mismatch-expected`](#cl-gbdt-dimension-mismatch-expected)
- [`dimension-mismatch-given`](#cl-gbdt-dimension-mismatch-given)
- [`evaluation`](#cl-gbdt-lightgbm-evaluation)
- [`feature-importance`](#cl-gbdt-lightgbm-feature-importance)
- [`file-format-mismatch`](#cl-gbdt-file-format-mismatch)
- [`file-format-mismatch-declared`](#cl-gbdt-file-format-mismatch-declared)
- [`file-format-mismatch-detected`](#cl-gbdt-file-format-mismatch-detected)
- [`file-format-mismatch-path`](#cl-gbdt-file-format-mismatch-path)
- [`foreign-call-error`](#cl-gbdt-foreign-call-error)
- [`foreign-call-error-code`](#cl-gbdt-foreign-call-error-code)
- [`foreign-call-error-function-name`](#cl-gbdt-foreign-call-error-function-name)
- [`foreign-call-error-message`](#cl-gbdt-foreign-call-error-message)
- [`free-booster`](#cl-gbdt-lightgbm-free-booster)
- [`free-dataset`](#cl-gbdt-lightgbm-free-dataset)
- [`gbdt-error`](#cl-gbdt-gbdt-error)
- [`handle-backend`](#cl-gbdt-handle-backend)
- [`handle-released-p`](#cl-gbdt-handle-released-p)
- [`lightgbm-backend`](#cl-gbdt-lightgbm-lightgbm-backend)
- [`load-model`](#cl-gbdt-lightgbm-load-model)
- [`make-csr-matrix`](#cl-gbdt-make-csr-matrix)
- [`missing-foreign-symbols`](#cl-gbdt-missing-foreign-symbols)
- [`missing-foreign-symbols-names`](#cl-gbdt-missing-foreign-symbols-names)
- [`missing-training-set`](#cl-gbdt-missing-training-set)
- [`missing-training-set-booster`](#cl-gbdt-missing-training-set-booster)
- [`model-to-string`](#cl-gbdt-lightgbm-model-to-string)
- [`open-backend`](#cl-gbdt-open-backend)
- [`predict`](#cl-gbdt-lightgbm-predict)
- [`released-handle-error`](#cl-gbdt-released-handle-error)
- [`released-handle-error-object`](#cl-gbdt-released-handle-error-object)
- [`save-model`](#cl-gbdt-lightgbm-save-model)
- [`unfreed-handle-warning`](#cl-gbdt-unfreed-handle-warning)
- [`unfreed-handle-warning-kind`](#cl-gbdt-unfreed-handle-warning-kind)
- [`unknown-backend`](#cl-gbdt-unknown-backend)
- [`unknown-backend-registered`](#cl-gbdt-unknown-backend-registered)
- [`unknown-capability`](#cl-gbdt-unknown-capability)
- [`unknown-capability-capability`](#cl-gbdt-unknown-capability-capability)
- [`unknown-capability-known`](#cl-gbdt-unknown-capability-known)
- [`unsupported-argument`](#cl-gbdt-unsupported-argument)
- [`unsupported-argument-argument`](#cl-gbdt-unsupported-argument-argument)
- [`unsupported-argument-backend`](#cl-gbdt-unsupported-argument-backend)
- [`unsupported-argument-reason`](#cl-gbdt-unsupported-argument-reason)
- [`unsupported-element-type`](#cl-gbdt-unsupported-element-type)
- [`unsupported-element-type-given`](#cl-gbdt-unsupported-element-type-given)
- [`untested-backend-version`](#cl-gbdt-untested-backend-version)
- [`untested-backend-version-backend`](#cl-gbdt-untested-backend-version-backend)
- [`untested-backend-version-tested`](#cl-gbdt-untested-backend-version-tested)
- [`untested-backend-version-version`](#cl-gbdt-untested-backend-version-version)
- [`update-one-iteration`](#cl-gbdt-lightgbm-update-one-iteration)
- [`wrong-backend-reference`](#cl-gbdt-wrong-backend-reference)
- [`wrong-backend-reference-argument`](#cl-gbdt-wrong-backend-reference-argument)
- [`wrong-backend-reference-backend`](#cl-gbdt-wrong-backend-reference-backend)
- [`wrong-backend-reference-expected`](#cl-gbdt-wrong-backend-reference-expected)
- [`wrong-backend-reference-given`](#cl-gbdt-wrong-backend-reference-given)

### `cl-gbdt/xgboost` -- 93 symbols

- [`*known-capabilities*`](#cl-gbdt-known-capabilities)
- [`backend-capabilities`](#cl-gbdt-backend-capabilities)
- [`backend-error`](#cl-gbdt-backend-error)
- [`backend-error-backend`](#cl-gbdt-backend-error-backend)
- [`backend-info`](#cl-gbdt-backend-info)
- [`backend-library-load-failed`](#cl-gbdt-backend-library-load-failed)
- [`backend-library-load-failed-cause`](#cl-gbdt-backend-library-load-failed-cause)
- [`backend-library-load-failed-path`](#cl-gbdt-backend-library-load-failed-path)
- [`backend-library-not-found`](#cl-gbdt-backend-library-not-found)
- [`backend-library-not-found-searched`](#cl-gbdt-backend-library-not-found-searched)
- [`backend-library-path`](#cl-gbdt-backend-library-path)
- [`backend-methods-not-loaded`](#cl-gbdt-backend-methods-not-loaded)
- [`backend-methods-not-loaded-generic-function`](#cl-gbdt-backend-methods-not-loaded-generic-function)
- [`backend-name`](#cl-gbdt-backend-name)
- [`backend-not-open`](#cl-gbdt-backend-not-open)
- [`backend-open-p`](#cl-gbdt-backend-open-p)
- [`backend-supports-p`](#cl-gbdt-backend-supports-p)
- [`backend-version`](#cl-gbdt-backend-version)
- [`booster`](#cl-gbdt-booster)
- [`booster-boosted-rounds`](#cl-gbdt-xgboost-booster-boosted-rounds)
- [`booster-training-set`](#cl-gbdt-booster-training-set)
- [`booster-validation-sets`](#cl-gbdt-booster-validation-sets)
- [`capability-unavailable`](#cl-gbdt-capability-unavailable)
- [`capability-unavailable-capability`](#cl-gbdt-capability-unavailable-capability)
- [`close-backend`](#cl-gbdt-close-backend)
- [`create-booster`](#cl-gbdt-xgboost-create-booster)
- [`create-dataset`](#cl-gbdt-xgboost-create-dataset)
- [`csr-matrix`](#cl-gbdt-csr-matrix)
- [`csr-matrix-indices`](#cl-gbdt-csr-matrix-indices)
- [`csr-matrix-indptr`](#cl-gbdt-csr-matrix-indptr)
- [`csr-matrix-num-columns`](#cl-gbdt-csr-matrix-num-columns)
- [`csr-matrix-num-rows`](#cl-gbdt-csr-matrix-num-rows)
- [`csr-matrix-values`](#cl-gbdt-csr-matrix-values)
- [`data-error`](#cl-gbdt-data-error)
- [`dataset`](#cl-gbdt-dataset)
- [`dataset-num-features`](#cl-gbdt-xgboost-dataset-num-features)
- [`dataset-num-rows`](#cl-gbdt-xgboost-dataset-num-rows)
- [`dimension-mismatch`](#cl-gbdt-dimension-mismatch)
- [`dimension-mismatch-expected`](#cl-gbdt-dimension-mismatch-expected)
- [`dimension-mismatch-given`](#cl-gbdt-dimension-mismatch-given)
- [`evaluate-one-iteration`](#cl-gbdt-xgboost-evaluate-one-iteration)
- [`evaluation`](#cl-gbdt-xgboost-evaluation)
- [`feature-importance`](#cl-gbdt-xgboost-feature-importance)
- [`file-format-mismatch`](#cl-gbdt-file-format-mismatch)
- [`file-format-mismatch-declared`](#cl-gbdt-file-format-mismatch-declared)
- [`file-format-mismatch-detected`](#cl-gbdt-file-format-mismatch-detected)
- [`file-format-mismatch-path`](#cl-gbdt-file-format-mismatch-path)
- [`foreign-call-error`](#cl-gbdt-foreign-call-error)
- [`foreign-call-error-code`](#cl-gbdt-foreign-call-error-code)
- [`foreign-call-error-function-name`](#cl-gbdt-foreign-call-error-function-name)
- [`foreign-call-error-message`](#cl-gbdt-foreign-call-error-message)
- [`free-booster`](#cl-gbdt-xgboost-free-booster)
- [`free-dataset`](#cl-gbdt-xgboost-free-dataset)
- [`gbdt-error`](#cl-gbdt-gbdt-error)
- [`handle-backend`](#cl-gbdt-handle-backend)
- [`handle-released-p`](#cl-gbdt-handle-released-p)
- [`load-model`](#cl-gbdt-xgboost-load-model)
- [`make-csr-matrix`](#cl-gbdt-make-csr-matrix)
- [`missing-foreign-symbols`](#cl-gbdt-missing-foreign-symbols)
- [`missing-foreign-symbols-names`](#cl-gbdt-missing-foreign-symbols-names)
- [`missing-training-set`](#cl-gbdt-missing-training-set)
- [`missing-training-set-booster`](#cl-gbdt-missing-training-set-booster)
- [`model-to-string`](#cl-gbdt-xgboost-model-to-string)
- [`open-backend`](#cl-gbdt-open-backend)
- [`predict`](#cl-gbdt-xgboost-predict)
- [`released-handle-error`](#cl-gbdt-released-handle-error)
- [`released-handle-error-object`](#cl-gbdt-released-handle-error-object)
- [`save-model`](#cl-gbdt-xgboost-save-model)
- [`slice-model`](#cl-gbdt-xgboost-slice-model)
- [`unfreed-handle-warning`](#cl-gbdt-unfreed-handle-warning)
- [`unfreed-handle-warning-kind`](#cl-gbdt-unfreed-handle-warning-kind)
- [`unknown-backend`](#cl-gbdt-unknown-backend)
- [`unknown-backend-registered`](#cl-gbdt-unknown-backend-registered)
- [`unknown-capability`](#cl-gbdt-unknown-capability)
- [`unknown-capability-capability`](#cl-gbdt-unknown-capability-capability)
- [`unknown-capability-known`](#cl-gbdt-unknown-capability-known)
- [`unsupported-argument`](#cl-gbdt-unsupported-argument)
- [`unsupported-argument-argument`](#cl-gbdt-unsupported-argument-argument)
- [`unsupported-argument-backend`](#cl-gbdt-unsupported-argument-backend)
- [`unsupported-argument-reason`](#cl-gbdt-unsupported-argument-reason)
- [`unsupported-element-type`](#cl-gbdt-unsupported-element-type)
- [`unsupported-element-type-given`](#cl-gbdt-unsupported-element-type-given)
- [`untested-backend-version`](#cl-gbdt-untested-backend-version)
- [`untested-backend-version-backend`](#cl-gbdt-untested-backend-version-backend)
- [`untested-backend-version-tested`](#cl-gbdt-untested-backend-version-tested)
- [`untested-backend-version-version`](#cl-gbdt-untested-backend-version-version)
- [`update-one-iteration`](#cl-gbdt-xgboost-update-one-iteration)
- [`wrong-backend-reference`](#cl-gbdt-wrong-backend-reference)
- [`wrong-backend-reference-argument`](#cl-gbdt-wrong-backend-reference-argument)
- [`wrong-backend-reference-backend`](#cl-gbdt-wrong-backend-reference-backend)
- [`wrong-backend-reference-expected`](#cl-gbdt-wrong-backend-reference-expected)
- [`wrong-backend-reference-given`](#cl-gbdt-wrong-backend-reference-given)
- [`xgboost-backend`](#cl-gbdt-xgboost-xgboost-backend)

## Symbols

<a id="cl-gbdt-known-capabilities"></a>

## `cl-gbdt:*known-capabilities*`

- **Kind** variable
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Every capability name `backend-supports-p' will answer for.

Policy section 7 named five as a MINIMUM -- it asks that at least those be answerable, not
that they be the whole list -- so this grows as new questions become worth asking.
`:missing-value' is the first name added past that floor: whether the backend lets a caller
say which value in the data means *missing*, which XGBoost does through its creation and
prediction config JSONs and LightGBM has no C-API key for at all. A name is registered here
as soon as it is a question worth asking, whether or not any backend answers true to it yet --
`:multidimensional-feature-score' is registered and false everywhere, which says "not
supported yet" rather than "never heard of it". Registering the name is what makes a
misspelling distinguishable from a real answer.

`:categorical-features' is the second name past that floor: whether the backend lets a caller
name which COLUMNS of the data hold categories rather than quantities, so a split on one of
them partitions the category set instead of thresholding an ordinal that has no order. XGBoost
answers it through the `"feature_type"' field its `XGDMatrixSetStrFeatureInfo' attaches to a
finished DMatrix, and LightGBM through a `categorical_feature' key in the parameter string that
builds the dataset -- a key rather than a C symbol, so nothing a symbol probe can decide. It is
a question about `make-dataset' alone: `predict' takes no such argument, the trained trees
already carrying the category sets they split on.

`:prediction-shape' is the third name past that floor: whether the backend states the SHAPE of
a result it just predicted, which `predict' returns as its second value -- a list of integers
in `array-dimensions' order, or NIL where the backend states none. Its answer says what that
second value CONTAINS rather than whether a call will be accepted, and NO OPERATION REFUSES ON
IT, which is deliberate.

`:custom-objective' is the fourth name past that floor: whether `train' accepts an
`:objective' function that turns the current raw scores into a gradient and a Hessian, so a
run boosts against the caller's own loss rather than one built into the library.

`:custom-evaluation' is the fifth name past that floor: whether `train' accepts an
`:evaluation' function that turns one dataset's current predictions into a named metric
value, recorded per iteration alongside the library's own metrics. Both backends answer it
true, and it is the ONE name here whose two true answers come out of DIFFERENT LISTS --
LightGBM probes it, XGBoost declares it. See the paragraph on where a re-checked name's answer
comes from, below, which that shape is the sole exception to.

It is not the odd member of an otherwise uniform list. SIX of the ten names here are
re-checked by the operation they gate, each signalling `capability-unavailable' for an argument
it cannot honour: `:missing-value', `:categorical-features', `:custom-objective' and
`:custom-evaluation', in both backends' `protocol.lisp' -- the last two
by `%check-custom-objective' and `%check-custom-evaluation', off `train''s :OBJECTIVE and
:EVALUATION -- plus `:sparse-input' and `:model-slicing', which are re-checked a layer down
in `api.lisp': `%check-sparse-input', in both backends' copy of that file, gates
`create-dataset' and `predict', the two operations that choose a sparse entry point, and
`:model-slicing' gates `slice-model' in XGBoost's. Those six are the whole of
it -- `:evaluation-history', `:early-stopping' and `:multidimensional-feature-score' are
re-checked nowhere (the third has no backend answering it true at all, and
`%check-feature-score-dim', which is where a multidimensional score is rejected, signals
`unsupported-argument' off what the library reported at runtime rather than off the
capability).
`:prediction-shape' has no argument to refuse at all: nothing asks for a shape, so a false
answer means the second value is always NIL while `predict' otherwise behaves exactly as it
always did. That is not the silent fallback policy section 7 forbids -- section 7 protects a
caller who asked for something and would otherwise have it quietly dropped, and here nothing was
asked for -- and a re-check added for symmetry with those six would make `predict' signal
outright on a backend that simply has less to say.

Where a re-checked name's answer COMES from varies and does not affect that split: a backend
declares it in `*provided-capabilities*' when nothing is left to look up, and names its C
functions in `*optional-symbols*' when a library that opens perfectly well can still lack
them -- `:sparse-input', `:model-slicing' and `:custom-objective' are probed that way on every
backend providing them, `:missing-value' and `:categorical-features' declared.

`:custom-evaluation' is the one name that is BOTH, and that is a fact about the two LIBRARIES
rather than a disagreement between the two backends about what the capability means. Each
backend's per-dataset prediction read needs different C functions, and they fall on opposite
sides of required: LightGBM's needs three (`LGBM_BoosterGetPredict',
`LGBM_BoosterGetNumPredict', `LGBM_BoosterGetNumClasses'), not one of them in that backend's
`*required-symbols*', so a LightGBM lacking any of the three opens perfectly well and cannot
serve a custom metric -- exactly the state a probe exists to detect. XGBoost's needs one
(`XGBoosterPredictFromDMatrix'), which IS in its `*required-symbols*', so no XGBoost this
library will open is in the corresponding state and a probe would have nothing left to decide.
Naming it in both lists on ONE backend is what would be wrong: `probe-capabilities' records
PROVIDED entries ahead of probed ones, so the probe's answer would be unreachable, which that
function's own docstring calls a contradiction in the backend's declarations.

A false answer needs the same re-check whichever list it came from, and a backend naming a
capability in NEITHER list reads false for a third reason again -- the absence of any
declaration, which is LightGBM's `:missing-value' answer.

`:evaluation-history' is true on both backends: `train' records one, and each backend names
the capability in its own `*provided-capabilities*' rather than in `*optional-symbols*',
because the C functions it needs are already in that backend's `*required-symbols*' and a
probe therefore has nothing left to decide. See `probe-capabilities''s PROVIDED.
```

<a id="cl-gbdt-lightgbm-version-range"></a>

## `cl-gbdt:*lightgbm-version-range*`

- **Kind** variable
- **Exported from** `cl-gbdt`

```text
LightGBM's recorded compatible-version range.

Never compared against a loaded version at runtime, unlike *XGBOOST-VERSION-RANGE* --
LightGBM's C API has no version entry point at all (`grep -c Version
src/lightgbm/c-api.lisp' reports 0), so `backend-version' is always NIL on this
backend and there is nothing to compare it against. See
`cl-gbdt/src/lightgbm/classes''s `initialize-backend' and `check-backend-version'
below. Recorded here anyway, for the same documentation purpose the README's
backend-differences table serves.

VERIFIED-LOW moved from "4.7.0" to "4.0.0" once task 4 actually ran the functional
suite against it -- confirmed clean, same 106 assertions as the pinned "4.7.0".

INFERRED-LOW is pinned at "3.0.0" and can never move to VERIFIED: `lightgbm==3.0.0'
predates aarch64 wheels on PyPI -- confirmed directly by task 4 (`pip download
lightgbm==3.0.0 --only-binary=:all:' finds no candidate at all on this platform) -- so
the CI version matrix this range's header comment describes cannot install it to
actually test against. This lower bound stays inferred permanently, not just until the
next matrix run.
```

<a id="cl-gbdt-xgboost-version-range"></a>

## `cl-gbdt:*xgboost-version-range*`

- **Kind** variable
- **Exported from** `cl-gbdt`

```text
XGBoost's recorded compatible-version range -- see *LIGHTGBM-VERSION-RANGE*'s
docstring for what VERIFIED and INFERRED each mean and this file's header comment for
why the runtime check gates on INFERRED, not VERIFIED. Unlike LightGBM, this range is
actually compared against a loaded version: XGBoost's C API does expose one, read by
`cl-gbdt/src/xgboost/native''s `%read-version'.

Both bounds moved up from task 3's "1.7.0": task 4 ran the functional suite against
`xgboost==1.7.0' and its ranking round trip failed --
`xgboost-api-ranking-round-trip-respects-group-boundaries' in
tests/functional/xgboost-api.lisp requires predictions to increase strictly within each
query group, and 1.7.0 instead produced a tie between the first two rows of each group
where 3.3.0 keeps all four strictly increasing. Every other assertion (105 of 106,
including the plain classification and multiclass round trips, feature-importance,
save/load, and every close-backend guard) passed unchanged at 1.7.0 -- this is a real
function returning different numbers, not a symptom of a missing symbol or a crash, so
`probe-foreign-symbols' and `tools/check-upstream.lisp''s header comparison could never
have caught it. `xgboost==2.0.0' was tried next and passed everything the pinned
"3.3.0" does, so INFERRED-LOW moved up to meet VERIFIED-LOW at "2.0.0" rather than
leave the disproven "1.7.0" claim in place under a wider, ABI-only label. Nothing
between 1.7.0 and 2.0.0 was tested, so this range makes no claim about it either.
```

<a id="cl-gbdt-backend"></a>

## `cl-gbdt:backend`

- **Kind** class
- **Superclasses** `standard-object`
- **Exported from** `cl-gbdt`

```text
A connection to a gradient boosting implementation.

Each backend specializes this class and implements `initialize-backend',
`shutdown-backend', and the unified API methods.
```

### Slots

#### `name`

- **Readers** `backend-name`

```text
Backend name, either `:lightgbm' or `:xgboost'.
```

#### `library-path`

- **Readers** `backend-library-path`

```text
Path of the shared library actually loaded.
```

#### `capabilities`

- **Readers** `backend-capabilities`

```text
Plist of capability keyword to T/NIL, as returned by
`probe-capabilities' at `open-backend' time. Read through `backend-supports-p' rather
than directly, so an unregistered keyword signals `unknown-capability' instead of
silently reading as NIL.
```

#### `version`

- **Readers** `backend-version`

```text
Library version; an inferred value or nil when unavailable.
```

#### `openp`

```text
Whether the shared library is currently open.
```

<a id="cl-gbdt-backend-capabilities"></a>

## `cl-gbdt:backend-capabilities`

- **Kind** generic function
- **Signature** `(backend-capabilities object)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:backend`'s `capabilities` slot. See `cl-gbdt:backend`.

<a id="cl-gbdt-backend-error"></a>

## `cl-gbdt:backend-error`

- **Kind** condition
- **Superclasses** `gbdt-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Base type for errors during backend initialization or connection.
```

### Slots

#### `backend`

- **Readers** `backend-error-backend`

```text
Backend name, either `:lightgbm' or `:xgboost'.
```

<a id="cl-gbdt-backend-error-backend"></a>

## `cl-gbdt:backend-error-backend`

- **Kind** generic function
- **Signature** `(backend-error-backend condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:backend-error`'s `backend` slot. See `cl-gbdt:backend-error`.

<a id="cl-gbdt-backend-info"></a>

## `cl-gbdt:backend-info`

- **Kind** function
- **Signature** `(backend-info backend)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Return BACKEND's state as a plist.

Keys are `:name', `:version', `:capabilities', `:library-path' and `:open'.
```

<a id="cl-gbdt-backend-library-load-failed"></a>

## `cl-gbdt:backend-library-load-failed`

- **Kind** condition
- **Superclasses** `backend-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
The shared library was found but could not be loaded.
```

### Slots

#### `path`

- **Readers** `backend-library-load-failed-path`

```text
The candidate path `cffi:load-foreign-library' was given and
rejected -- an explicit `:path', one read from the backend's environment-variable
override, or one found by searching the vendored library directory.
```

#### `cause`

- **Readers** `backend-library-load-failed-cause`

```text
The condition `cffi:load-foreign-library' itself signalled,
kept here rather than reported directly so its own message and type both survive.
```

<a id="cl-gbdt-backend-library-load-failed-cause"></a>

## `cl-gbdt:backend-library-load-failed-cause`

- **Kind** generic function
- **Signature** `(backend-library-load-failed-cause condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:backend-library-load-failed`'s `cause` slot. See `cl-gbdt:backend-library-load-failed`.

<a id="cl-gbdt-backend-library-load-failed-path"></a>

## `cl-gbdt:backend-library-load-failed-path`

- **Kind** generic function
- **Signature** `(backend-library-load-failed-path condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:backend-library-load-failed`'s `path` slot. See `cl-gbdt:backend-library-load-failed`.

<a id="cl-gbdt-backend-library-not-found"></a>

## `cl-gbdt:backend-library-not-found`

- **Kind** condition
- **Superclasses** `backend-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
The backend's shared library could not be located.
```

### Slots

#### `searched`

- **Readers** `backend-library-not-found-searched`

```text
List of paths that were searched.
```

<a id="cl-gbdt-backend-library-not-found-searched"></a>

## `cl-gbdt:backend-library-not-found-searched`

- **Kind** generic function
- **Signature** `(backend-library-not-found-searched condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:backend-library-not-found`'s `searched` slot. See `cl-gbdt:backend-library-not-found`.

<a id="cl-gbdt-backend-library-path"></a>

## `cl-gbdt:backend-library-path`

- **Kind** generic function
- **Signature** `(backend-library-path object)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:backend`'s `library-path` slot. See `cl-gbdt:backend`.

<a id="cl-gbdt-backend-methods-not-loaded"></a>

## `cl-gbdt:backend-methods-not-loaded`

- **Kind** condition
- **Superclasses** `backend-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Signalled when a unified-API generic function is called on a backend
whose methods for it were never loaded.

`cl-gbdt/lightgbm' registers the backend class and opens the library; `cl-gbdt/lightgbm/unified'
is what adds the methods implementing `cl-gbdt''s portable generic functions. A program that
loaded the first and not the second can open a backend and then find no method to call, and
this condition is that state said plainly. Without it the caller sees the implementation's own
no-applicable-method error, which names neither the backend nor the system to load.

Never signalled while both systems are loaded: each fallback method is specialized on a base
class, so a backend's own method is always more specific.
```

### Slots

#### `generic-function`

- **Readers** `backend-methods-not-loaded-generic-function`

```text
The unified-API generic function that was called.
```

<a id="cl-gbdt-backend-methods-not-loaded-generic-function"></a>

## `cl-gbdt:backend-methods-not-loaded-generic-function`

- **Kind** generic function
- **Signature** `(backend-methods-not-loaded-generic-function condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:backend-methods-not-loaded`'s `generic-function` slot. See `cl-gbdt:backend-methods-not-loaded`.

<a id="cl-gbdt-backend-name"></a>

## `cl-gbdt:backend-name`

- **Kind** generic function
- **Signature** `(backend-name object)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:backend`'s `name` slot. See `cl-gbdt:backend`.

<a id="cl-gbdt-backend-not-open"></a>

## `cl-gbdt:backend-not-open`

- **Kind** condition
- **Superclasses** `backend-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
An operation was attempted on a backend that is not open.
```

<a id="cl-gbdt-backend-open-p"></a>

## `cl-gbdt:backend-open-p`

- **Kind** function
- **Signature** `(backend-open-p backend)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Return true when BACKEND's shared library is open.
```

<a id="cl-gbdt-backend-supports-p"></a>

## `cl-gbdt:backend-supports-p`

- **Kind** function
- **Signature** `(backend-supports-p backend capability)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Return true when BACKEND provides CAPABILITY, NIL when it does not.

CAPABILITY must be one of `*known-capabilities*'; anything else signals
`unknown-capability' rather than answering NIL, so a typo cannot be mistaken for a
supported-but-absent feature.

A true answer means the shared library actually loaded resolved every foreign symbol the
capability needs, as probed at `open-backend' -- not that the headers cl-gbdt was built
against declared them -- or that the backend declared the capability unconditionally,
having nothing left to probe because everything it needs is in `*required-symbols*' and the
backend opened at all (see `probe-capabilities''s PROVIDED). A false answer means the
feature is unavailable here, and is never a licence to fall back to something else silently:
the operation itself signals `capability-unavailable' (policy section 7).
```

<a id="cl-gbdt-backend-version"></a>

## `cl-gbdt:backend-version`

- **Kind** generic function
- **Signature** `(backend-version object)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:backend`'s `version` slot. See `cl-gbdt:backend`.

<a id="cl-gbdt-booster"></a>

## `cl-gbdt:booster`

- **Kind** class
- **Superclasses** `handle`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
A trained, or in-progress, backend model.

Retains its training set strongly, which is what makes `with-booster''s docstring --
that nesting it inside `with-dataset' cannot invert the release order -- true.
```

### Slots

#### `training-set`

- **Readers** `booster-training-set`

```text
The dataset this booster was trained on, or NIL for a
`load-model' booster, which has none.
```

#### `validation-sets`

- **Readers** `booster-validation-sets`

```text
The validation datasets passed to `train' via
`:valid-sets', or NIL when none were given. `LGBM_BoosterAddValidData' stores each
one's pointer inside the booster exactly as `LGBM_BoosterCreate' does for the
training set, so these are retained strongly for the same liveness-checking reason
as TRAINING-SET: freeing one out from under a live booster is the same hazard.
```

#### `best-iteration`

- **Readers** `booster-best-iteration`

```text
The iteration an early-stopped `train' run judged
best, or NIL.

Set only when `train' was given `:early-stopping', and only when its watcher actually
determined a best iteration -- NIL for a `load-model' booster, for one trained without
`:early-stopping', and for one trained with it whose watcher never got the chance to see
one (see `train''s docstring for the two ways that happens even with `:early-stopping'
supplied). Writable only through `%set-booster-best-iteration', below, whose one caller is
`train' itself, after its loop ends. `predict', `save-model' and `model-to-string''s
`:num-iteration :best' resolve against this slot, signalling `unsupported-argument' rather
than assuming a default when it is NIL.
```

<a id="cl-gbdt-booster-best-iteration"></a>

## `cl-gbdt:booster-best-iteration`

- **Kind** generic function
- **Signature** `(booster-best-iteration object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:booster`'s `best-iteration` slot. See `cl-gbdt:booster`.

<a id="cl-gbdt-xgboost-booster-boosted-rounds"></a>

## `cl-gbdt/xgboost:booster-boosted-rounds`

- **Kind** function
- **Signature** `(booster-boosted-rounds booster)`
- **Exported from** `cl-gbdt/xgboost`

```text
Return the number of boosting rounds BOOSTER holds, via `XGBoosterBoostedRounds'.

This is the count `slice-model''s BEGIN and END are indices into, and the only way for a
caller to learn the valid range before asking for a slice: `XGBoosterSlice' signals
`foreign-call-error' rather than clamping when END exceeds this number. It is also the
round index `update-one-iteration' boosts at, read from the same C function, so a booster
`train' ran for N rounds reports N here.

Takes a booster handle, not a pointer, and reads it through `%check-xgboost-booster' --
policy section 10: a Layer 1 entry point a caller reaches directly must validate the handle
it is given, since no `defmethod' specializer has ruled out a foreign one first.

Signals `wrong-backend-reference' when BOOSTER was not built by the XGBoost backend,
`released-handle-error' when it has been freed, `backend-not-open' when its backend has
been closed, and `foreign-call-error' when the call itself fails.
```

<a id="cl-gbdt-lightgbm-booster-eval"></a>

## `cl-gbdt/lightgbm:booster-eval`

- **Kind** function
- **Signature** `(booster-eval booster data-index)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Return BOOSTER's evaluation metric values for the dataset at DATA-INDEX, as a fresh
`(simple-array double-float (*))', via `LGBM_BoosterGetEvalCounts' and
`LGBM_BoosterGetEval'. Entry N corresponds to entry N of `booster-eval-names' -- call that
separately for the metric names, since LightGBM reports them independently of DATA-INDEX.

DATA-INDEX is LightGBM's own numbering, passed straight through to `LGBM_BoosterGetEval'
unmodified: 0 means the training set BOOSTER was built from; 1 means the first dataset in
`train''s :VALID-SETS, 2 the second, and so on. Nothing here assigns a name to any of
them -- see design policy section 4's rule against inventing dataset names for LightGBM.

Signals `foreign-call-error' when DATA-INDEX is negative or exceeds the number of datasets
BOOSTER actually has attached -- confirmed directly against the vendored library, which
rejects both with a descriptive `LGBM_GetLastError' message rather than reading out of
bounds. Also signals `wrong-backend-reference' when BOOSTER was not built by the LightGBM
backend, `released-handle-error' when BOOSTER itself or any dataset it retains has already
been freed, and `backend-not-open' when its own backend has since been closed.

`%check-booster-datasets-live' runs before any foreign call here, and covers every dataset
BOOSTER retains rather than only the one DATA-INDEX addresses: `LGBM_BoosterGetEval'
evaluates through metric objects built over each attached dataset's own label and weight
arrays, none of which `LGBM_DatasetFree' clears from the booster. Evaluating after any of
them was freed reads memory that is no longer ours -- it does not reliably crash, which is
what makes it worth a guard rather than a warning. This is the same check
`update-one-iteration' has always made, and the one the `evaluation' method makes at
Layer 2; it is repeated here because this function is public in `cl-gbdt/lightgbm' and so
is reachable without going through that method at all.

It runs after `%check-lightgbm-booster' rather than before it, unlike in the `evaluation'
method, because that method reaches its body only for an argument CLOS already dispatched
as a `lightgbm-booster' while this one accepts whatever the caller passes: reading
`booster-training-set' out of a dataset would fail as a slot type error instead of the
`wrong-backend-reference' this promises. Neither check makes a foreign call, so both still
precede every one of them.
```

<a id="cl-gbdt-lightgbm-booster-eval-names"></a>

## `cl-gbdt/lightgbm:booster-eval-names`

- **Kind** function
- **Signature** `(booster-eval-names booster)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Return the names of BOOSTER's configured evaluation metrics, as a fresh list of
strings, via `LGBM_BoosterGetEvalCounts' and `LGBM_BoosterGetEvalNames'.

The returned list belongs to the caller outright; cl-gbdt keeps no reference to it. Entry
N here names entry N of the value list `booster-eval' returns for any DATA-INDEX -- metric
names are configured once for the whole booster, not per dataset, which is why this takes
no DATA-INDEX argument of its own, unlike `booster-eval'. NIL when BOOSTER has no metrics
configured -- trained with `metric=none', or returned by `load-model', which never had
metrics attached in the first place.

Signals `wrong-backend-reference' when BOOSTER was not built by the LightGBM backend,
`released-handle-error' when it has already been freed, and `backend-not-open' when its
own backend has since been closed.
```

<a id="cl-gbdt-booster-training-set"></a>

## `cl-gbdt:booster-training-set`

- **Kind** generic function
- **Signature** `(booster-training-set object)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:booster`'s `training-set` slot. See `cl-gbdt:booster`.

<a id="cl-gbdt-booster-validation-sets"></a>

## `cl-gbdt:booster-validation-sets`

- **Kind** generic function
- **Signature** `(booster-validation-sets object)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:booster`'s `validation-sets` slot. See `cl-gbdt:booster`.

<a id="cl-gbdt-call-with-foreign-matrix"></a>

## `cl-gbdt:call-with-foreign-matrix`

- **Kind** generic function
- **Signature** `(call-with-foreign-matrix matrix function)`
- **Exported from** `cl-gbdt`

```text
Make MATRIX available as foreign memory and call FUNCTION.

FUNCTION is called with four arguments: (POINTER NROW NCOL ELEMENT-TYPE). POINTER
addresses the elements laid out row-major. ELEMENT-TYPE is the symbol `double-float'
or `single-float'.

POINTER is valid only for the duration of FUNCTION and must not escape it. Add a
method here to support a new input format.
```

### Methods

#### `(call-with-foreign-matrix (matrix foreign-matrix) (function t))`

```text
MATRIX is already foreign memory, so this method is pure forwarding: FUNCTION receives
MATRIX's own pointer, dimensions and element type unchanged, with no copy and no pin.
```

#### `(call-with-foreign-matrix (matrix array) (function t))`

```text
MATRIX is a Lisp array, which is not foreign memory yet: this method validates it is
2D, normalizes its element type to `double-float' or `single-float', and then pins or
copies its storage -- a simple-array is pinned in place on SBCL, anything else is copied,
see `%call-with-pinned-matrix' and `%call-with-copied-matrix' -- before calling FUNCTION
with a pointer into it.
```

<a id="cl-gbdt-capability-unavailable"></a>

## `cl-gbdt:capability-unavailable`

- **Kind** condition
- **Superclasses** `backend-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Signalled when an operation needs a capability this backend does not have.

Distinct from `unknown-capability': the question was well formed and the answer is no.
Policy section 7 requires the operation itself to signal this rather than relying on the
caller having asked `backend-supports-p' first, and forbids falling back to some other
behaviour instead.
```

### Slots

#### `capability`

- **Readers** `capability-unavailable-capability`

```text
The capability the operation needed.
```

<a id="cl-gbdt-capability-unavailable-capability"></a>

## `cl-gbdt:capability-unavailable-capability`

- **Kind** generic function
- **Signature** `(capability-unavailable-capability condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:capability-unavailable`'s `capability` slot. See `cl-gbdt:capability-unavailable`.

<a id="cl-gbdt-check-backend-version"></a>

## `cl-gbdt:check-backend-version`

- **Kind** function
- **Signature** `(check-backend-version backend-name version range)`
- **Exported from** `cl-gbdt`

```text
Signal `untested-backend-version' -- a WARNING, not an ERROR; see its own docstring
for why -- when VERSION, BACKEND-NAME's loaded library version, falls outside RANGE's
*inferred* bounds, or does not parse as a "MAJOR.MINOR.PATCH" string at all, VERSION
= NIL included. Does nothing when VERSION is confirmed within range.

Gates on RANGE's inferred bounds rather than its narrower verified ones -- see this
file's header comment: warning on every difference from the exact version the
functional suite happens to run against would fire on nearly every compatible caller.

Only `cl-gbdt/src/xgboost/classes''s `initialize-backend' calls this. It is never
called for LightGBM: with `backend-version' always NIL there, this would signal on
every single open, a check that can never actually confirm compatibility -- see
*LIGHTGBM-VERSION-RANGE*'s docstring for the fuller explanation of that asymmetry.
```

<a id="cl-gbdt-check-foreign-call"></a>

## `cl-gbdt:check-foreign-call`

- **Kind** function
- **Signature** `(check-foreign-call code function-name last-error)`
- **Exported from** `cl-gbdt`

```text
Signal `foreign-call-error' when CODE reports failure, otherwise return CODE.

Both wrapped libraries use the same idiom: 0 on success, nonzero on failure, with the
detail behind a `*GetLastError' entry point returning a `char *'. LAST-ERROR is a
function of no arguments returning that message as a string, or NIL -- it is the only
part that differs between backends, so it is the only part passed in. FUNCTION-NAME
identifies which C function reported CODE, for the condition's report.
```

<a id="cl-gbdt-close-backend"></a>

## `cl-gbdt:close-backend`

- **Kind** function
- **Signature** `(close-backend backend)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Close BACKEND. Does nothing if it is already closed.
```

<a id="cl-gbdt-lightgbm-create-booster"></a>

## `cl-gbdt/lightgbm:create-booster`

- **Kind** function
- **Signature** `(create-booster backend dataset &key parameters valid-sets)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Create a booster over DATASET on BACKEND via `LGBM_BoosterCreate', returning a
`lightgbm-booster'.

The result is UNTRAINED: `LGBM_BoosterCreate' allocates the model and fixes its parameters,
and every boosting iteration comes from a later `update-one-iteration'. Free it with
`free-booster'.

PARAMETERS is a plist in LightGBM'S OWN vocabulary, rendered by `%parameter-string' and
handed to the creation call verbatim; nothing here translates a key or a value, and no key
is added. VALID-SETS is a list of `lightgbm-dataset's -- bare datasets, not the
(NAME . DATASET) entries `cl-gbdt''s `train' accepts, a name being a training-report concept
with no meaning at this layer -- attached afterward with `LGBM_BoosterAddValidData' in the
order given. The booster retains DATASET and a COPY of VALID-SETS, which keeps them alive
for its lifetime and lets `update-one-iteration' notice a dataset freed out from under it.
The copy is what makes that promise hold: were the caller's own list object stored, a later
`delete' or `(setf (cdr ...))' on it would remove an entry from the booster's view while
LightGBM still held that dataset's pointer.

Signals `wrong-backend-reference' when BACKEND is not a `lightgbm-backend' -- the other
backend's object, or not a backend at all -- before anything else is read from it and ahead of
the openness check, which for another backend's object would answer about the wrong shared
library; see `%check-object-class'. Signals `backend-not-open' before any foreign call
when BACKEND is not open -- see `%check-backend-open' -- `wrong-backend-reference' when
DATASET or a VALID-SETS entry is not a `lightgbm-dataset', and `released-handle-error' or
`backend-not-open' when one is but has already been freed or had its own backend closed; see
`%check-lightgbm-dataset', which is what rules out `LGBM_BoosterCreate' being handed a
booster's own pointer as its training set. This function dispatches on nothing, so those two
checks are the only thing standing between a wrong-kind handle -- or another backend's object
where this one belongs -- and a segfault. Signals `foreign-call-error' when creation reports
success but writes a null handle -- that check lives in `%create-booster', beside the call it
guards, and is not repeated here.

Every check runs before the creation call, so a rejected VALID-SETS entry leaves no booster
in existence at all. The raw handle then exists in C from the moment that call returns and
nothing in Lisp references it until `make-handle' runs -- `with-pointer-ownership' spans
exactly that gap, so a validation set that fails to attach frees the booster rather than
orphaning it.

PARAMETERS cannot fail in that gap here, and this is where the two backends' ownership forms
differ. `%parameter-string' is an ARGUMENT to `%create-booster', so a malformed plist -- an
odd-length one, which `normalize-parameters' signals `data-error' for -- signals before any
handle exists; measured, `:parameters '(:eta)' frees no booster at all, because there was
none. On `cl-gbdt/src/xgboost/api' the same plist frees exactly one: that library applies
parameters AFTER creation, so its `%set-parameters' runs inside its ownership form. See its
`create-booster', which carries both measurements.
```

<a id="cl-gbdt-xgboost-create-booster"></a>

## `cl-gbdt/xgboost:create-booster`

- **Kind** function
- **Signature** `(create-booster backend dataset &key parameters valid-sets)`
- **Exported from** `cl-gbdt/xgboost`

```text
Create a booster over DATASET on BACKEND via `XGBoosterCreate', returning an
`xgboost-booster'.

The result is UNTRAINED: `XGBoosterCreate' allocates the model and every boosting iteration
comes from a later `update-one-iteration'. Free it with `free-booster'.

VALID-SETS is a list of `xgboost-dataset's -- bare datasets, not the (NAME . DATASET) entries
`cl-gbdt''s `train' accepts, a name being a training-report concept with no meaning at this
layer. Unlike LightGBM's `LGBM_BoosterCreate', which takes the training set alone and gains
validation sets afterward through `LGBM_BoosterAddValidData', `XGBoosterCreate' takes the
whole array of DMatrix handles a booster will ever reference AT ONCE -- the training set
first, then each validation set in the order given -- and this library has no "add valid
data" entry point at all. Nothing is attached after creation here because there is nothing to
attach it with: a validation set left out of the creation call could never be added later.

PARAMETERS reaches the booster the other way round for the same reason of asymmetry: a plist
in XGBOOST'S OWN vocabulary, applied AFTER creation by `%set-parameters', one
`XGBoosterSetParam' call per pair, this library having no bulk-parameter argument the way
`LGBM_BoosterCreate''s parameter string is. Nothing here translates a key or a value, and no
key is added. Nor does anything REJECT one: measured against the vendored library,
`XGBoosterSetParam' validates nothing at all -- an unknown key, a non-numeric `eta', an
objective that does not exist and a bogus `tree_method' were each accepted here without a
status code, and the last three surfaced later, as a `foreign-call-error' out of the first
call that makes XGBoost configure its learner, which is the `XGBoosterBoostedRounds' inside
`update-one-iteration'. The unknown key never surfaced at all. So this function returns a
booster for parameters it cannot honour, and the error, when there is one, names a C function
the caller never wrote.

The booster retains DATASET and a COPY of VALID-SETS, which keeps them alive for its lifetime
and lets `update-one-iteration' notice a dataset freed out from under it. The copy is what
makes that promise hold: were the caller's own list object stored, a later `delete' or
`(setf (cdr ...))' on it would remove an entry from the booster's view while XGBoost still
held that dataset's pointer.

Signals `wrong-backend-reference' when BACKEND is not an `xgboost-backend' -- the other
backend's object, or not a backend at all -- before anything else is read from it and ahead of
the openness check, which for another backend's object would answer about the wrong shared
library; see `%check-object-class'. Signals `backend-not-open' before any foreign call when
BACKEND is not open -- see `%check-backend-open' -- `wrong-backend-reference' when DATASET or
a VALID-SETS entry is not an `xgboost-dataset', and `released-handle-error' or
`backend-not-open' when one is but has already been freed or had its own backend closed; see
`%check-xgboost-dataset'. That check runs on EVERY element of the array this backend hands
`XGBoosterCreate', not on DATASET alone, and it is what rules out a booster's own pointer
arriving there as a DMatrix handle. This function dispatches on nothing, so those two checks
are the only thing standing between a wrong-kind handle -- or another backend's object where
this one belongs -- and a segfault. Signals `foreign-call-error' when creation reports success
but writes a null handle -- that check lives in `%create-booster', beside the call it guards,
and is not repeated here.

Every check runs before the creation call, so a rejected VALID-SETS entry leaves no booster in
existence at all. The raw handle then exists in C from the moment that call returns and
nothing in Lisp references it until `make-handle' runs -- `with-pointer-ownership' spans
exactly that gap, so anything signalling in between frees the booster rather than orphaning
it. `%set-parameters' is the whole of what runs in that gap, and it does signal, just not for
any of the parameters the measurement above covers: it renders PARAMETERS through
`normalize-parameters', which signals `data-error' for an ODD-LENGTH plist rather than let a
final key go silently missing. Measured -- `:parameters '(:eta)' signals `data-error' from
inside this form and `%free-booster-unchecked' runs exactly once. That is this form's live
failure mode, not a prospective one, and it is what a value XGBoost itself would have
tolerated cannot produce.

`cl-gbdt/src/lightgbm/api''s `create-booster' looks symmetrical here and is not. The same odd
plist signals the same `data-error' there, but with NO booster freed -- measured, zero calls
to that backend's `%free-booster-unchecked' -- because `%parameter-string' is an ARGUMENT to
`LGBM_BoosterCreate' and so runs before any handle exists. What its own ownership form catches
instead is `%add-valid-data' refusing a mismatched bin mapper. The asymmetry is the one this
docstring opens with, reaching all the way down: parameters go in at creation there and after
creation here.
```

<a id="cl-gbdt-lightgbm-create-dataset"></a>

## `cl-gbdt/lightgbm:create-dataset`

- **Kind** function
- **Signature** `(create-dataset backend matrix &key label weight group feature-names parameters reference)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Build and return a `lightgbm-dataset' from MATRIX on BACKEND.

MATRIX is a dense matrix -- built via `LGBM_DatasetCreateFromMat' -- or a `csr-matrix', built
via `LGBM_DatasetCreateFromCSR'. Nothing else about this function varies with which of the two
it is: see `%dataset-pointer' above, the only form here that branches on it.

LABEL and WEIGHT are attached to the finished dataset with `LGBM_DatasetSetField' under those
LightGBM field names, GROUP with the same call under `group', and FEATURE-NAMES with
`LGBM_DatasetSetFeatureNames'. Each is attached only when supplied; a NIL one is not written
as an empty field. REFERENCE is another `lightgbm-dataset' whose bin mapper this one should
align to -- what `LGBM_DatasetCreateFromMat' calls its `reference' argument, and what a
validation set needs to be binned the same way as the training set it will be scored against
-- or NIL for none.

PARAMETERS is a plist in LightGBM'S OWN vocabulary, rendered by `%parameter-string' and handed
to the creation call verbatim; nothing here translates a key or a value, and no key is added.
`:categorical-feature' is one such key among the rest -- this function does not know it as a
concept, and a caller who wants columns read as categories writes that key here, already
rendered as the string LightGBM expects. `cl-gbdt/src/lightgbm/protocol''s `make-dataset' is
what turns the portable :CATEGORICAL-FEATURES argument into exactly that entry before calling
this.

Signals `wrong-backend-reference' when BACKEND is not a `lightgbm-backend' -- the other
backend's object, or not a backend at all -- before anything else is read from it and ahead of
the openness check below, which for another backend's object would answer about the wrong
shared library; see `%check-object-class'. Signals `backend-not-open' before any foreign
call when BACKEND is not open -- see `%check-backend-open'. Signals `capability-unavailable'
naming `:sparse-input' when MATRIX is a `csr-matrix' and that capability reads false,
`wrong-backend-reference' when REFERENCE is supplied and is not a `lightgbm-dataset',
`released-handle-error' when it has already been freed, and `backend-not-open' when its own
backend has since been closed -- see `%reference-pointer'. Signals `foreign-call-error' when
the creation call reports success but writes a null handle: a library-contract violation, but
one every later call through this handle would otherwise dereference blindly.

The raw dataset handle exists in C from the moment the creation call returns, but
`make-handle' does not take ownership of it until the very end -- attaching LABEL, WEIGHT,
GROUP or FEATURE-NAMES can each signal first (a wrong-length LABEL is the commonest way).
`with-pointer-ownership' spans exactly that gap: the pointer is owned by nobody inside its
body, and any exit that has not called TAKE-OWNERSHIP frees the raw dataset here instead of
orphaning it.
```

<a id="cl-gbdt-xgboost-create-dataset"></a>

## `cl-gbdt/xgboost:create-dataset`

- **Kind** function
- **Signature** `(create-dataset backend matrix &key label weight group feature-names missing feature-types)`
- **Exported from** `cl-gbdt/xgboost`

```text
Build and return an `xgboost-dataset' -- a DMatrix -- from MATRIX on BACKEND.

MATRIX is a dense matrix -- built via `XGDMatrixCreateFromDense' -- or a `csr-matrix', built
via `XGDMatrixCreateFromCSR'. Nothing else about this function varies with which of the two it
is: see `%dataset-pointer' above, the only form here that branches on it.

LABEL and WEIGHT are attached to the finished DMatrix with `XGDMatrixSetInfoFromInterface'
under those XGBoost field names, GROUP with `XGDMatrixSetUIntInfo', and FEATURE-NAMES and
FEATURE-TYPES with `XGDMatrixSetStrFeatureInfo' under its `"feature_name"' and
`"feature_type"' fields. Each is attached only when supplied; a NIL one is not written as an
empty field, so a caller who passes nothing gets a DMatrix with no such field at all rather
than a vector of defaults.

MISSING is the value in MATRIX that means *missing* -- a real, or NIL for this library's own
default, the IEEE NaN. It becomes the `"missing"' key of whichever creation config JSON
MATRIX's own form reaches, so it is honoured identically on both paths. The comparison the
library then makes is at SINGLE precision, whatever MATRIX's element type: two `double-float's
that share a `single-float' both count as missing against a sentinel that narrows to it.

FEATURE-TYPES is a list of strings, one per column, in XGBoost'S OWN vocabulary: `"c"' for a
categorical column and `"q"' for a quantitative one. Nothing here renders it, range-checks it
or knows what a category is -- `cl-gbdt/src/xgboost/protocol''s `make-dataset' is what turns
the portable :CATEGORICAL-FEATURES column list into exactly this, by
`categorical-feature-types', before calling this. A list of the wrong length is XGBoost's own
to refuse.

There is no :PARAMETERS and no :REFERENCE, and their absence is not a refusal: this library's
creation config JSON documents three keys -- `"missing"', which has its own argument above,
`"nthread"' and `"data_split_mode"' -- and has no concept resembling LightGBM's bin-mapper
alignment at all, so there is nothing at this layer for either argument to name.
`cl-gbdt/src/xgboost/protocol''s `make-dataset' does signal `unsupported-argument' for both,
because the unified generic promises the arguments and a caller moving a working call across
backends has to be told rather than silently ignored; see that method's docstring for the
measurement behind it.

Signals `wrong-backend-reference' when BACKEND is not an `xgboost-backend' -- the other
backend's object, or not a backend at all -- before anything else is read from it and ahead of
the openness check below, which for another backend's object would answer about the wrong
shared library; see `%check-object-class'. Signals `backend-not-open' before any foreign
call when BACKEND is not open -- see `%check-backend-open'. Signals `capability-unavailable'
naming `:sparse-input' when MATRIX is a `csr-matrix' and that capability reads false -- see
`%check-sparse-input'. Signals `foreign-call-error' when the creation call reports success but
writes a null handle: a library-contract violation, but one every later call through this
handle would otherwise dereference blindly.

The raw DMatrix handle exists in C from the moment the creation call returns, but `make-handle'
does not take ownership of it until the very end -- attaching LABEL, WEIGHT, GROUP,
FEATURE-NAMES or FEATURE-TYPES can each signal first (a wrong-length LABEL is the commonest
way). `with-pointer-ownership' spans exactly that gap: the pointer is owned by nobody inside
its body, and any exit that has not called TAKE-OWNERSHIP frees the raw DMatrix here instead of
orphaning it.
```

<a id="cl-gbdt-csr-matrix"></a>

## `cl-gbdt:csr-matrix`

- **Kind** structure
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
A sparse matrix in compressed sparse row (CSR) format: for row ROW, the pairs
INDICES[k] and VALUES[k], for INDPTR[ROW] <= k < INDPTR[ROW+1], give that row's non-zero
columns and their values.

NUM-COLUMNS is a required argument to `make-csr-matrix' rather than inferred from the
largest index INDICES actually holds, because a matrix's declared width and its largest
stored index are different facts: the last columns can legitimately hold nothing, and the
stored indices alone cannot distinguish that from a matrix that simply is not that wide.

NUM-ROWS is derived from INDPTR's length (`(1- (length indptr))'), not stored as its own
slot -- INDPTR already fixes the row count, so a separate slot would only be a second copy
of the same fact for validation to keep in sync.

INDPTR and INDICES are `(simple-array (signed-byte 32) (*))'; VALUES is a `(simple-array
double-float (*))'. All three are already coerced to what a backend hands to the C API, so
a backend method only needs to pin them -- see `with-foreign-matrix' in this same file for
the dense equivalent.

Every slot is `:read-only t', so there is no `setf' expander for any of the four
accessors. `make-csr-matrix' is the only way to build one and it validates everything a
backend later relies on; a writable slot would make that validation defeatable after the
fact, and a matrix whose INDPTR had been replaced by a list would reach the C API as a raw
`type-error' from inside the pinning code rather than as one of this file's own
conditions. `foreign-matrix' above is reader-only for the same reason: a `csr-matrix' that
exists is one both backends can be handed.

An entry a row does not store is *absent*, not zero, and the two libraries read absence
differently: LightGBM reads an absent entry as `0.0' (its own `zero_as_missing' is off by
default) while XGBoost reads one as missing, and no config key on either changes that --
it is what CSR means to each library. So a `csr-matrix' that omits entries describes
different data to the two backends and changes trained numbers silently rather than
signalling; store every element, zeros included, when the same matrix has to mean the same
thing on both. See README.markdown's "An absent entry is not a zero, and the two libraries
disagree about it" for the measured runs on each.
```

### Slots

#### `indptr`

- **Readers** `csr-matrix-indptr`


#### `indices`

- **Readers** `csr-matrix-indices`


#### `values`

- **Readers** `csr-matrix-values`


#### `num-columns`

- **Readers** `csr-matrix-num-columns`


<a id="cl-gbdt-csr-matrix-indices"></a>

## `cl-gbdt:csr-matrix-indices`

- **Kind** function
- **Signature** `(csr-matrix-indices instance)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:csr-matrix`'s `indices` slot. See `cl-gbdt:csr-matrix`.

<a id="cl-gbdt-csr-matrix-indptr"></a>

## `cl-gbdt:csr-matrix-indptr`

- **Kind** function
- **Signature** `(csr-matrix-indptr instance)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:csr-matrix`'s `indptr` slot. See `cl-gbdt:csr-matrix`.

<a id="cl-gbdt-csr-matrix-num-columns"></a>

## `cl-gbdt:csr-matrix-num-columns`

- **Kind** function
- **Signature** `(csr-matrix-num-columns instance)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:csr-matrix`'s `num-columns` slot. See `cl-gbdt:csr-matrix`.

<a id="cl-gbdt-csr-matrix-num-rows"></a>

## `cl-gbdt:csr-matrix-num-rows`

- **Kind** function
- **Signature** `(csr-matrix-num-rows matrix)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Return MATRIX's row count, one less than the length of its INDPTR array.

See the struct's own docstring for why NUM-ROWS is derived rather than stored.
```

<a id="cl-gbdt-csr-matrix-values"></a>

## `cl-gbdt:csr-matrix-values`

- **Kind** function
- **Signature** `(csr-matrix-values instance)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:csr-matrix`'s `values` slot. See `cl-gbdt:csr-matrix`.

<a id="cl-gbdt-data-error"></a>

## `cl-gbdt:data-error`

- **Kind** condition
- **Superclasses** `gbdt-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Base type for errors in the supplied data.
```

<a id="cl-gbdt-dataset"></a>

## `cl-gbdt:dataset`

- **Kind** class
- **Superclasses** `handle`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
A backend's training or validation dataset.
```

<a id="cl-gbdt-dataset-num-features"></a>

## `cl-gbdt:dataset-num-features`

- **Kind** generic function
- **Signature** `(dataset-num-features dataset)`
- **Exported from** `cl-gbdt`

```text
Return the number of features in DATASET.
```

### Methods

#### `(dataset-num-features (dataset dataset))`

```text
Fallback for a DATASET whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `dataset-num-features'. A Layer 1 caller reads the same
count with that backend's own `dataset-num-features' in `api.lisp', which needs no unified
system loaded.
```

#### `(dataset-num-features (dataset lightgbm-dataset))`

```text
Return DATASET's feature count, read via `LGBM_DatasetGetNumFeature'. Delegates wholly, as
`dataset-num-rows' above does and for the same reason.
```

#### `(dataset-num-features (dataset xgboost-dataset))`

```text
Return DATASET's feature count, read via `XGDMatrixNumCol'. Delegates wholly, as
`dataset-num-rows' above does and for the same reason.
```

<a id="cl-gbdt-lightgbm-dataset-num-features"></a>

## `cl-gbdt/lightgbm:dataset-num-features`

- **Kind** function
- **Signature** `(dataset-num-features dataset)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Return DATASET's feature count, read via `LGBM_DatasetGetNumFeature'.

Signals what `dataset-num-rows' above signals, on the same terms and through the same two
calls.
```

<a id="cl-gbdt-xgboost-dataset-num-features"></a>

## `cl-gbdt/xgboost:dataset-num-features`

- **Kind** function
- **Signature** `(dataset-num-features dataset)`
- **Exported from** `cl-gbdt/xgboost`

```text
Return DATASET's feature count, read via `XGDMatrixNumCol'.

Signals what `dataset-num-rows' above signals, on the same terms and through the same two
calls.
```

<a id="cl-gbdt-dataset-num-rows"></a>

## `cl-gbdt:dataset-num-rows`

- **Kind** generic function
- **Signature** `(dataset-num-rows dataset)`
- **Exported from** `cl-gbdt`

```text
Return the number of rows in DATASET.
```

### Methods

#### `(dataset-num-rows (dataset dataset))`

```text
Fallback for a DATASET whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `dataset-num-rows'. A Layer 1 caller reads the same
count with that backend's own `dataset-num-rows' in `api.lisp', which needs no unified
system loaded.
```

#### `(dataset-num-rows (dataset lightgbm-dataset))`

```text
Return DATASET's row count, read via `LGBM_DatasetGetNumData'.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/lightgbm/api''s `dataset-num-rows', which is where
the class guard the specializer above used to provide now lives too.
```

#### `(dataset-num-rows (dataset xgboost-dataset))`

```text
Return DATASET's row count, read via `XGDMatrixNumRow'.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/xgboost/api''s `dataset-num-rows', which is where
the class guard the specializer above used to provide now lives too.
```

<a id="cl-gbdt-lightgbm-dataset-num-rows"></a>

## `cl-gbdt/lightgbm:dataset-num-rows`

- **Kind** function
- **Signature** `(dataset-num-rows dataset)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Return DATASET's row count, read via `LGBM_DatasetGetNumData'.

Signals `wrong-backend-reference' when DATASET is not a `lightgbm-dataset' -- a booster, the
other backend's dataset, or not a handle at all. This function dispatches on nothing, so
`%check-object-class' is the only thing between a wrong-kind pointer and the foreign call.
Unlike the frees, which use that same check and then tolerate a released handle, this requires
a LIVE one and reads it through `handle-live-pointer' immediately after: `released-handle-error'
for a freed DATASET, `backend-not-open' for a closed backend.
```

<a id="cl-gbdt-xgboost-dataset-num-rows"></a>

## `cl-gbdt/xgboost:dataset-num-rows`

- **Kind** function
- **Signature** `(dataset-num-rows dataset)`
- **Exported from** `cl-gbdt/xgboost`

```text
Return DATASET's row count, read via `XGDMatrixNumRow'.

Signals `wrong-backend-reference' when DATASET is not an `xgboost-dataset' -- a booster, the
other backend's dataset, or not a handle at all. This function dispatches on nothing, so
`%check-object-class' is the only thing between a wrong-kind pointer and the foreign call.
Unlike the frees, which use that same check and then tolerate a released handle, this requires
a LIVE one and reads it through `handle-live-pointer' immediately after.
```

<a id="cl-gbdt-dimension-mismatch"></a>

## `cl-gbdt:dimension-mismatch`

- **Kind** condition
- **Superclasses** `data-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
An array's rank or size differs from what was expected.
```

### Slots

#### `expected`

- **Readers** `dimension-mismatch-expected`

```text
What the dimension should have been, almost always a short
phrase in words -- e.g. "a 2D array" or "INDPTR to start at 0" -- except
`check-objective-result', where it is `(rows groups)', the one shape both checked arrays
were required to match exactly.
```

#### `given`

- **Readers** `dimension-mismatch-given`

```text
What was actually supplied -- an integer, a list of integers, or
a string, depending on the check -- rarely the same shape as EXPECTED: EXPECTED usually
names what was required in words, while GIVEN carries the actual value or shape, so the
two are not parallel. E.g. `(:gradient DIMS :hessian DIMS)' when a custom objective's
gradient and Hessian are checked together, so the report can say which of the two was
wrong.
```

<a id="cl-gbdt-dimension-mismatch-expected"></a>

## `cl-gbdt:dimension-mismatch-expected`

- **Kind** generic function
- **Signature** `(dimension-mismatch-expected condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:dimension-mismatch`'s `expected` slot. See `cl-gbdt:dimension-mismatch`.

<a id="cl-gbdt-dimension-mismatch-given"></a>

## `cl-gbdt:dimension-mismatch-given`

- **Kind** generic function
- **Signature** `(dimension-mismatch-given condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:dimension-mismatch`'s `given` slot. See `cl-gbdt:dimension-mismatch`.

<a id="cl-gbdt-xgboost-evaluate-one-iteration"></a>

## `cl-gbdt/xgboost:evaluate-one-iteration`

- **Kind** function
- **Signature** `(evaluate-one-iteration booster datasets names)`
- **Exported from** `cl-gbdt/xgboost`

```text
Evaluate BOOSTER against DATASETS via `XGBoosterEvalOneIter', labeling each entry of
DATASETS with the corresponding entry of NAMES, at BOOSTER's own current boosted-round
count (read fresh via `XGBoosterBoostedRounds', the same source `update-one-iteration'
reads its own `iter' argument from).

Unlike LightGBM's `booster-eval', which reads the validation sets `train' already
attached to BOOSTER by index, XGBoost's `XGBoosterEvalOneIter' evaluates whatever
DMatrices the caller passes here and does not consult BOOSTER's own `:valid-sets' at all
-- this is the asymmetry that
docs/superpowers/specs/2026-08-06-evaluation-api-design.md section 2 documents between
the two libraries' C APIs; a portable Layer 2 method built on top of this still has to
decide what to pass as DATASETS, which is exactly what that design leaves for the layer
above this one to settle, not this function.

Returns two values:

  RAW    The exact string `XGBoosterEvalOneIter' wrote to `out_result', e.g.
         "[5]\ttrain-logloss:0.1\tvalid-logloss:0.2". This is the only value this
         function can promise is faithful to what XGBoost actually produced -- its
         format is not documented as stable. RAW is authoritative; PARSED below is
         derived from it and must never be substituted for it.

  PARSED A fresh list of (LABEL . VALUE) conses, this function's own interpretation of
         RAW -- see `%parse-eval-result' for exactly how, and why it does not attempt to
         split LABEL back into a dataset name and a metric name. VALUE is a `double-float'
         for ordinary numeric text, or NIL when XGBoost wrote something `%read-eval-value'
         cannot read as one -- its own `"inf"'/`"nan"' spellings for a non-finite
         metric are the known case; the LABEL stays regardless, only VALUE goes missing.
         PARSED is empty when DATASETS is empty. A field PARSED cannot make sense of never
         costs RAW: this always returns RAW once the foreign call itself has succeeded, no
         matter what PARSED does with it afterward. If RAW and PARSED ever disagree, RAW is
         what XGBoost said and PARSED is a bug in this function, not the other way around.

DATASETS and NAMES must be the same length, checked in `%eval-one-iter' before either
foreign array is built; see that function's docstring for the hazard an unchecked
mismatch would reach. Every entry of DATASETS, and BOOSTER itself, is read through
`handle-live-pointer' and confirmed built by the `:xgboost' backend before any foreign
call -- passing a released or wrong-backend handle straight to `XGBoosterEvalOneIter'
would be a segfault, not a catchable condition, exactly as for every other
caller-supplied handle in this backend. Signals `wrong-backend-reference' when BOOSTER or
any DATASETS entry was not built by `:xgboost', `released-handle-error' for an
already-freed one, `backend-not-open' when its own backend has since been closed, and
`dimension-mismatch' when DATASETS and NAMES differ in length.
```

<a id="cl-gbdt-evaluation"></a>

## `cl-gbdt:evaluation`

- **Kind** generic function
- **Signature** `(evaluation booster)`
- **Exported from** `cl-gbdt`

```text
Return BOOSTER's evaluation metrics as a fresh list of
(DATASET-INDEX METRIC-NAME VALUE) lists -- one entry per metric per dataset.

DATASET-INDEX identifies the dataset a value was computed on by its position among the
datasets BOOSTER retains: 0 is the training set BOOSTER was trained on, 1 is the first
entry of `train''s :VALID-SETS, 2 the second, and so on -- the order the caller supplied
them in, which is also LightGBM's own `data_idx' numbering. No name is invented for any
of them: LightGBM identifies a validation set by index and by nothing else, so there is
no upstream name to report and this API does not fabricate one.

METRIC-NAME is the backend's own name for the metric, exactly as that backend spells it.
LightGBM's "binary_logloss" and XGBoost's "logloss" name related quantities under
different names, and neither is translated into the other's vocabulary. Which metrics a
booster has at all is decided by what `train' was given (LightGBM's `metric', XGBoost's
`eval_metric') and by the objective's own default, not by this call.

VALUE is a `double-float', or NIL when the backend reported a value this library could
not read as one. Only XGBoost produces NIL: its values arrive as formatted text (see
:VALUE-SOURCE below) and its `std::ostream' spells a non-finite double "inf" or
"nan", which is not `double-float' syntax. LightGBM's values are already doubles and
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
`backend-not-open' when BOOSTER's backend has since been closed.
```

### Methods

#### `(evaluation (booster booster))`

```text
Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `evaluation' rather than reading anything. A Layer 1
caller gets the same metrics with that backend's own `evaluation' in `api.lisp' instead.
```

#### `(evaluation (booster lightgbm-booster))`

```text
Return BOOSTER's evaluation metrics via `%read-evaluation', the pointer-level reader this
backend shares between this method and `train''s per-iteration recording loop, so the two
can never disagree -- see the `evaluation' generic function's docstring for the portable
contract this satisfies.

Reads one `LGBM_BoosterGetEval' result per dataset BOOSTER retains, in the order
`train' attached them, and pairs entry N of each with entry N of the single metric-name
list `LGBM_BoosterGetEvalNames' reports for the whole booster -- LightGBM configures
metrics once per booster, not per dataset, so the names are read once and reused for
every dataset rather than re-read per index.

DATASET-INDEX is passed straight through to `LGBM_BoosterGetEval' as its `data_idx':
this backend's own numbering already is the portable contract's numbering, 0 for the
training set and 1 upward for each `:VALID-SETS' entry in order, with no renumbering
along the way. The datasets are counted from BOOSTER's own retained handles rather
than asked of the library, which has no entry point reporting how many are attached; a
`load-model' booster retains none, and is the case that count is 0 for. LightGBM does
answer `data_idx' 0 for such a booster -- with an empty result, confirmed against the
vendored library, since a model file carries no metrics -- but there is no dataset behind
that index for the portable contract to name, so it is not evaluated at all rather than
reported as index 0.

The values are `LGBM_BoosterGetEval''s own doubles, returned unmodified, which is what
the secondary value's `:value-source :library-doubles' says; unlike XGBoost's, none of
it is parsed from text, so there is no :RAW to keep and no VALUE is ever NIL.

This method's whole body was procedure, so all of it -- the per-dataset reading, the
`data_idx' passthrough and the value-source plist alike -- is
`cl-gbdt/src/lightgbm/api''s `evaluation'. That function checks BOOSTER's own kind and
pointer through `%check-lightgbm-booster' first, then calls `%check-booster-datasets-live'
before any foreign call, including inside `%read-evaluation': `LGBM_BoosterGetEval'
evaluates each attached validation set through the metric objects built over that
dataset's own label and weight arrays, none of which `LGBM_DatasetFree' clears from the
booster, so evaluating after one of them was freed is a use-after-free rather than a
catchable condition -- the identical hazard `update-one-iteration' guards against with
the same call.

Two things changed with the move. First, the kind check now runs before
`%check-booster-datasets-live' rather than after, so a value that is not a booster at
all is answered with `wrong-backend-reference' where it used to reach
`booster-training-set' and produce a bare CLOS `no-applicable-method'. Second, that same
reorder means a booster that is ITSELF released and also still retains a released
dataset now signals `released-handle-error' naming the BOOSTER -- from
`%check-lightgbm-booster''s own live-pointer check, which now runs first -- where the
Layer 2 method's old order, `%check-booster-datasets-live' before the booster's own
pointer, named the DATASET instead. Same condition type either way, so nothing a caller
dispatches on changed; every other order is as it was.
```

#### `(evaluation (booster xgboost-booster))`

```text
Return BOOSTER's evaluation metrics via `%read-evaluation', the pointer-level reader this
backend shares between this method and `train''s per-iteration recording loop, so the two
can never disagree -- see the `evaluation' generic function's docstring for the portable
contract this satisfies.

`XGBoosterEvalOneIter' evaluates whatever DMatrices it is handed and consults nothing the
booster was built with, so the datasets this evaluates are BOOSTER's own retained
handles: its training set first, then each `train' :VALID-SETS entry in the order the
caller supplied them. That is what makes DATASET-INDEX mean the same thing here as it
does on LightGBM, which can only evaluate what training attached -- measured before this
method was written: for one booster, one set of handles and one iteration,
`XGBoosterEvalOneIter' called directly and this path through `%read-evaluation' produce
byte-identical result strings, and both agree with the logloss and error rate computed
independently from `predict' on the same data. A `load-model' booster retains no dataset
at all, which is the case an empty result comes from.

Each dataset is named to `XGBoosterEvalOneIter' by its own decimal index -- "0" for the
training set, "1" for the first validation set -- because the call requires one name per
DMatrix and builds each result label by joining that name to the metric's name with a
hyphen. `%split-eval-label' takes the label back apart against those same names, which is
the only way to recover the metric name: nothing in the result string alone marks where
one half ends and the other begins. Those names are an argument to a C call, never a
dataset name this API reports -- the caller sees the index, exactly as on LightGBM.

The values are `%parse-eval-result''s reading of `XGBoosterEvalOneIter''s formatted
output, which is what the secondary value's `:value-source :parsed-text' says, and its
`:raw' carries that output unmodified so nothing the library actually wrote is lost to
the parse. A field whose value the parser could not read as a `double-float' -- XGBoost
spells a non-finite one "inf" or "nan" -- keeps its entry with VALUE NIL rather than
disappearing from the result.

This method's whole body was procedure, so all of it -- the per-dataset pointer resolution,
the decimal naming, the parse and the provenance plist alike -- is
`cl-gbdt/src/xgboost/api''s `evaluation'. That function reads BOOSTER and every dataset it
evaluates through `handle-live-pointer' before calling `%read-evaluation', so a freed
booster or a freed retained dataset signals `released-handle-error' there; unlike
`cl-gbdt/src/lightgbm/api''s `evaluation', this backend needs no separate
`%check-booster-datasets-live', since every dataset it evaluates is one the delegate
resolves and checks explicitly, by its own handle, before any foreign call. The one thing
that changed with the move is that the booster's kind is now checked before its pointer is
read, so a value that is not a booster gets `wrong-backend-reference' rather than whatever
`handle-live-pointer' made of it. Unlike LightGBM's twin, the booster was already checked
before its retained datasets before the move, so a booster that is itself released and
also retains a released dataset has always signalled `released-handle-error' naming the
BOOSTER here, not the dataset.
```

<a id="cl-gbdt-lightgbm-evaluation"></a>

## `cl-gbdt/lightgbm:evaluation`

- **Kind** function
- **Signature** `(evaluation booster)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Return BOOSTER's evaluation metrics as two values: a list of (DATASET-INDEX METRIC-NAME
VALUE) entries, and a plist stating where the values came from.

Reads one `LGBM_BoosterGetEval' result per dataset BOOSTER retains, in the order they were
attached, through `%read-evaluation' -- the same pointer-level reader `train''s per-iteration
recording loop uses, so the two can never disagree. DATASET-INDEX is 0 for the training set
and 1 upward for each validation set, which is `LGBM_BoosterGetEval''s own `data_idx'; nothing
here renumbers anything. The datasets are counted from BOOSTER's own retained handles rather
than asked of the library, which has no entry point reporting how many are attached. A
`load-model' booster retains none, and is the case the count is 0 for.

The second value is always `(:value-source :library-doubles)': these are
`LGBM_BoosterGetEval''s own doubles, returned unmodified. Unlike `cl-gbdt/src/xgboost/api''s
`evaluation', nothing here parses text, so there is no :RAW to keep and no VALUE is ever NIL.

The kind check runs FIRST and `%check-booster-datasets-live' second, which is the opposite of
the order the Layer 2 method used. It has to be: `%check-booster-datasets-live' reads
`booster-training-set' off whatever it is handed, so a DATASET reaching it produced a bare
CLOS `no-applicable-method' error instead of a typed condition -- the same defect
`update-one-iteration' was measured to have and fixed the same way. That check is still owed
before any foreign call, `LGBM_BoosterGetEval' evaluating each attached validation set through
metric objects built over that dataset's own arrays, none of which `LGBM_DatasetFree' clears
from the booster.
```

<a id="cl-gbdt-xgboost-evaluation"></a>

## `cl-gbdt/xgboost:evaluation`

- **Kind** function
- **Signature** `(evaluation booster)`
- **Exported from** `cl-gbdt/xgboost`

```text
Return BOOSTER's evaluation metrics as two values: a list of (DATASET-INDEX METRIC-NAME
VALUE) entries, and a plist stating where the values came from.

`XGBoosterEvalOneIter' evaluates whatever DMatrices it is handed and consults nothing the
booster was built with, so what this evaluates is BOOSTER's own retained handles: its training
set first, then each validation set in the order it was given. That is what makes
DATASET-INDEX mean here what it means on LightGBM, which can only evaluate what training
attached. A booster from `load-model' retains none, and is the case an empty result comes
from.

Each dataset is named to the C call by its own decimal index -- "0", "1" -- because the
call requires one name per DMatrix and builds each result label by joining that name to the
metric's with a hyphen; `%split-eval-label' takes the label back apart against those same
names, which is the only way to recover the metric name. Those names are an argument to a C
call, never a dataset name this API reports.

The second value is `(:value-source :parsed-text :raw TEXT)': VALUE is `%parse-eval-result''s
reading of formatted output, and :RAW carries that output unmodified so nothing the library
wrote is lost to the parse. A field the parser could not read as a `double-float' -- XGBoost
spells a non-finite one "inf" or "nan" -- keeps its entry with VALUE NIL rather than
disappearing.

The kind check runs first and every retained dataset is then read through
`handle-live-pointer', so a freed booster or a freed retained dataset signals
`released-handle-error' here; unlike `cl-gbdt/src/lightgbm/api''s `evaluation', this needs no
separate `%check-booster-datasets-live', every dataset it evaluates being one it resolves and
checks explicitly before any foreign call.
```

<a id="cl-gbdt-feature-importance"></a>

## `cl-gbdt:feature-importance`

- **Kind** generic function
- **Signature** `(feature-importance booster &key kind num-iteration)`
- **Exported from** `cl-gbdt`

```text
Return BOOSTER's feature importances as `(simple-array double-float (*))'.

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
inside the library.
```

### Methods

#### `(feature-importance (booster booster) &key kind num-iteration)`

```text
Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `feature-importance' rather than computing anything. A
Layer 1 caller gets the same result with that backend's own `feature-importance' in
`api.lisp' instead.
```

#### `(feature-importance (booster lightgbm-booster) &key (kind :split) num-iteration)`

```text
Return BOOSTER's per-feature importances via `LGBM_BoosterFeatureImportance'.

The result has one entry per feature. The width comes from
`LGBM_BoosterGetNumFeature', which works whether BOOSTER came from `train' or
`load-model' -- unlike a booster's training set, which `load-model' leaves
unbound.

NUM-ITERATION does not accept :BEST, unlike `predict', `save-model' and
`model-to-string' -- `%reject-best-num-iteration' signals `unsupported-argument' for it
rather than letting it reach `LGBM_BoosterFeatureImportance' as raw, uninterpreted data.

The procedure is `cl-gbdt/src/lightgbm/api''s `feature-importance', and so is the :BEST
refusal: unlike `save-model' and `model-to-string', this operation never RESOLVED :BEST, and a
refusal is something Layer 1 can make for itself with the same argument name and the same
condition. Nothing is left here.
```

#### `(feature-importance (booster xgboost-booster) &key (kind :split) num-iteration)`

```text
Return BOOSTER's per-feature importances via `XGBoosterFeatureScore'.

Signals `unsupported-argument' when NUM-ITERATION is supplied: `XGBoosterFeatureScore''s
config JSON has no iteration-limiting key, only `importance_type', `feature_map' and
`feature_names' -- honoring it would require slicing the booster first, which this
backend does not do, so this refuses rather than silently scoring every round instead of
the requested subset.

The result has one entry per feature, indexed by column, matching
`cl-gbdt/src/lightgbm/api''s `feature-importance' -- zero for a feature never used
in a split. `XGBoosterFeatureScore' itself reports the opposite: `out_n_features' and
`out_scores' cover only features that appear in at least one split, so a feature never
split on is absent from its report, not present with a zero -- confirmed directly
against the vendored library and documented upstream. Left as `XGBoosterFeatureScore'
returns it, the result's length would be the number of *used* features, not the
dataset's column count, and its indices would not correspond to column positions --
sparse where LightGBM's equivalent is always dense. This builds a dense vector of
`%booster-num-features' entries instead, initialized to zero, and scatters each
reported score into the column `%feature-score-index' recovers from its feature name.

Signals `unsupported-argument' instead of returning a result at all when
`XGBoosterFeatureScore' reports more than one score per feature -- see
`%check-feature-score-dim'. In practice this is a linear (`gblinear') booster's `:split'
importance on a multi-class model: its scores are a per-class matrix, not one number per
feature, and there is no single value this backend can derive from that matrix without
inventing a reduction XGBoost itself does not define.

NUM-ITERATION does not accept :BEST, unlike `predict', `save-model' and
`model-to-string' -- `%reject-best-num-iteration' signals `unsupported-argument' for it
explicitly, ahead of the blanket rejection just below that would otherwise catch it only
incidentally, as any other non-NIL value.

The procedure is `cl-gbdt/src/xgboost/api''s `feature-importance', which takes no
:NUM-ITERATION at all. Both refusals stay here, in this order: :BEST explicitly, ahead of the
blanket refusal that would otherwise catch it only incidentally as any other non-NIL value.
```

<a id="cl-gbdt-lightgbm-feature-importance"></a>

## `cl-gbdt/lightgbm:feature-importance`

- **Kind** function
- **Signature** `(feature-importance booster &key (kind :split) num-iteration)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Return BOOSTER's per-feature importances via `LGBM_BoosterFeatureImportance', as a fresh
`(simple-array double-float (*))' with one entry per feature, indexed by column.

The width comes from `LGBM_BoosterGetNumFeature', which answers whether BOOSTER came from
`create-booster' or from `load-model' -- unlike a booster's training set, which `load-model'
leaves unbound. A feature never used in a split is reported as zero rather than omitted.

KIND is `:split' or `:gain', mapped onto LightGBM's own `C_API_FEATURE_IMPORTANCE_*' constant
by `%feature-importance-type', which signals for anything else. NUM-ITERATION is a positive
integer or NIL for every iteration; :BEST is refused by `%reject-best-num-iteration', as it is
in `save-model' and `predict' and for the same reason. KIND is checked BEFORE NUM-ITERATION,
preserving the order the Layer 2 method's own `let*' established.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend, and
`released-handle-error' or `backend-not-open' from the `handle-live-pointer' inside that
check, which runs before anything else is read.
```

<a id="cl-gbdt-xgboost-feature-importance"></a>

## `cl-gbdt/xgboost:feature-importance`

- **Kind** function
- **Signature** `(feature-importance booster &key (kind :split))`
- **Exported from** `cl-gbdt/xgboost`

```text
Return BOOSTER's per-feature importances via `XGBoosterFeatureScore', as a fresh
`(simple-array double-float (*))' with one entry per feature, indexed by column -- zero for a
feature never used in a split.

`XGBoosterFeatureScore' itself reports the opposite: `out_n_features' and `out_scores' cover
only features that appear in at least one split, so a feature never split on is absent from
its report rather than present with a zero. Left as it comes back, the result's length would
be the number of USED features and its indices would not correspond to columns -- sparse where
LightGBM's equivalent is always dense. This builds a dense vector of `%booster-num-features'
entries instead and scatters each reported score into the column `%feature-score-index'
recovers from its feature name.

KIND is `:split' or `:gain', mapped by `%feature-importance-type' onto `"weight"' and
`"total_gain"' -- the latter deliberately, XGBoost's own `"gain"' being an average where
LightGBM's `:gain' and this project's contract mean the total.

Takes no :NUM-ITERATION, unlike `cl-gbdt/src/lightgbm/api''s `feature-importance':
`XGBoosterFeatureScore''s config JSON has no iteration-limiting key. The argument is absent
here rather than refused; `cl-gbdt/src/xgboost/protocol''s method is where
`unsupported-argument' is signalled, that refusal existing only because LightGBM honours the
argument.

Signals `unsupported-argument' instead of returning a result at all when
`XGBoosterFeatureScore' reports more than one score per feature -- see
`%check-feature-score-dim'. In practice that is a `gblinear' booster's `:split' importance on
a multi-class model, whose scores are a per-class matrix with no single value to derive from
it.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend, and
`released-handle-error' or `backend-not-open' from the `handle-live-pointer' inside that
check, which runs before anything else is read.
```

<a id="cl-gbdt-file-format-mismatch"></a>

## `cl-gbdt:file-format-mismatch`

- **Kind** condition
- **Superclasses** `data-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
A file's contents do not match the format the caller declared for it.

Signalled by `cl-gbdt/xgboost''s `create-dataset-from-file' before any foreign call. Unlike
`unsupported-argument', the argument here is supported and well-formed -- it is the data that
disagrees with it.

This check exists because XGBoost does not make it. Measured against the vendored 3.3.0:
`train.csv?format=libsvm', and a binary DMatrix declared as libsvm, both segfault inside dmlc's
non-Lisp parser thread, where no Lisp handler can run and no condition can be signalled. The
reverse mismatches return success and garbage. Refusing before the call is therefore the only
point at which this failure can be reported at all, which is why it is a requirement of that
function's contract rather than a convenience.
```

### Slots

#### `path`

- **Readers** `file-format-mismatch-path`

```text
The file whose contents disagree with the declared format.
```

#### `declared`

- **Readers** `file-format-mismatch-declared`

```text
The format keyword the caller declared, e.g. :LIBSVM.
```

#### `detected`

- **Readers** `file-format-mismatch-detected`

```text
The format the file's first non-empty line was classified as, or
:UNKNOWN when it matched no format this wrapper can recognise.
```

<a id="cl-gbdt-file-format-mismatch-declared"></a>

## `cl-gbdt:file-format-mismatch-declared`

- **Kind** generic function
- **Signature** `(file-format-mismatch-declared condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:file-format-mismatch`'s `declared` slot. See `cl-gbdt:file-format-mismatch`.

<a id="cl-gbdt-file-format-mismatch-detected"></a>

## `cl-gbdt:file-format-mismatch-detected`

- **Kind** generic function
- **Signature** `(file-format-mismatch-detected condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:file-format-mismatch`'s `detected` slot. See `cl-gbdt:file-format-mismatch`.

<a id="cl-gbdt-file-format-mismatch-path"></a>

## `cl-gbdt:file-format-mismatch-path`

- **Kind** generic function
- **Signature** `(file-format-mismatch-path condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:file-format-mismatch`'s `path` slot. See `cl-gbdt:file-format-mismatch`.

<a id="cl-gbdt-find-backend-class"></a>

## `cl-gbdt:find-backend-class`

- **Kind** function
- **Signature** `(find-backend-class name)`
- **Exported from** `cl-gbdt`

```text
Return the class name registered for the backend NAME, or nil.
```

<a id="cl-gbdt-foreign-call-error"></a>

## `cl-gbdt:foreign-call-error`

- **Kind** condition
- **Superclasses** `gbdt-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
A C API call returned a non-zero status.

MESSAGE holds whatever LGBM_GetLastError or XGBGetLastError reported.
```

### Slots

#### `function-name`

- **Readers** `foreign-call-error-function-name`

```text
The C function whose return value became CODE, e.g.
"LGBM_BoosterCreate" or "XGBoosterCreate".
```

#### `code`

- **Readers** `foreign-call-error-code`

```text
The nonzero status FUNCTION-NAME returned. Both libraries document
-1 for failure and 0 for success, but this is only ever guaranteed to be nonzero here, not
exactly -1.
```

#### `message`

- **Readers** `foreign-call-error-message`

```text
The library's own account of the failure, read from
`LGBM_GetLastError' or `XGBGetLastError' at the moment CODE came back, or NIL when neither
library had one to report.
```

<a id="cl-gbdt-foreign-call-error-code"></a>

## `cl-gbdt:foreign-call-error-code`

- **Kind** generic function
- **Signature** `(foreign-call-error-code condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:foreign-call-error`'s `code` slot. See `cl-gbdt:foreign-call-error`.

<a id="cl-gbdt-foreign-call-error-function-name"></a>

## `cl-gbdt:foreign-call-error-function-name`

- **Kind** generic function
- **Signature** `(foreign-call-error-function-name condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:foreign-call-error`'s `function-name` slot. See `cl-gbdt:foreign-call-error`.

<a id="cl-gbdt-foreign-call-error-message"></a>

## `cl-gbdt:foreign-call-error-message`

- **Kind** generic function
- **Signature** `(foreign-call-error-message condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:foreign-call-error`'s `message` slot. See `cl-gbdt:foreign-call-error`.

<a id="cl-gbdt-foreign-element-type"></a>

## `cl-gbdt:foreign-element-type`

- **Kind** function
- **Signature** `(foreign-element-type element-type)`
- **Exported from** `cl-gbdt`

```text
Return the CFFI type keyword corresponding to ELEMENT-TYPE.

Signals `unsupported-element-type' for anything else.
```

<a id="cl-gbdt-foreign-matrix"></a>

## `cl-gbdt:foreign-matrix`

- **Kind** class
- **Superclasses** `standard-object`
- **Exported from** `cl-gbdt`

```text
A matrix already laid out row-major in foreign memory.

Use this to pass data without going through a Lisp array. Ownership stays with the
caller; cl-gbdt never frees it.
```

### Slots

#### `pointer`

- **Readers** `foreign-matrix-pointer`

```text
CFFI pointer to the first element.
```

#### `rows`

- **Readers** `foreign-matrix-rows`

```text
Number of rows.
```

#### `cols`

- **Readers** `foreign-matrix-cols`

```text
Number of columns.
```

#### `element-type`

- **Readers** `foreign-matrix-element-type`

```text
Element type, `double-float' or `single-float'.
```

<a id="cl-gbdt-foreign-matrix-cols"></a>

## `cl-gbdt:foreign-matrix-cols`

- **Kind** generic function
- **Signature** `(foreign-matrix-cols object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:foreign-matrix`'s `cols` slot. See `cl-gbdt:foreign-matrix`.

<a id="cl-gbdt-foreign-matrix-element-type"></a>

## `cl-gbdt:foreign-matrix-element-type`

- **Kind** generic function
- **Signature** `(foreign-matrix-element-type object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:foreign-matrix`'s `element-type` slot. See `cl-gbdt:foreign-matrix`.

<a id="cl-gbdt-foreign-matrix-pointer"></a>

## `cl-gbdt:foreign-matrix-pointer`

- **Kind** generic function
- **Signature** `(foreign-matrix-pointer object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:foreign-matrix`'s `pointer` slot. See `cl-gbdt:foreign-matrix`.

<a id="cl-gbdt-foreign-matrix-rows"></a>

## `cl-gbdt:foreign-matrix-rows`

- **Kind** generic function
- **Signature** `(foreign-matrix-rows object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:foreign-matrix`'s `rows` slot. See `cl-gbdt:foreign-matrix`.

<a id="cl-gbdt-free-booster"></a>

## `cl-gbdt:free-booster`

- **Kind** generic function
- **Signature** `(free-booster booster)`
- **Exported from** `cl-gbdt`

```text
Free BOOSTER. Does nothing if it was already freed.
```

### Methods

#### `(free-booster (booster booster))`

```text
Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `free-booster' rather than freeing anything. A Layer 1
caller frees the same handle with that backend's own `free-booster' in `api.lisp' instead.
```

#### `(free-booster (booster lightgbm-booster))`

```text
Free BOOSTER via `LGBM_BoosterFree'. Does nothing if it was already freed.

See `free-dataset''s docstring for why this does not signal `backend-not-open' when
BOOSTER's backend has already been closed -- the same `with-booster' cleanup-form
reasoning applies here.

This method's whole body was procedure too, and is `cl-gbdt/src/lightgbm/api''s
`free-booster' entirely, where the closed-backend branch now lives beside the identical one
`free-dataset' takes.
```

#### `(free-booster (booster xgboost-booster))`

```text
Free BOOSTER via `XGBoosterFree'. Does nothing if it was already freed.

See `free-dataset''s docstring for why this does not signal `backend-not-open' when
BOOSTER's backend has already been closed -- the same `with-booster' cleanup-form
reasoning applies here.

This method's whole body was procedure too, and is `cl-gbdt/src/xgboost/api''s
`free-booster' entirely, where the closed-backend branch now lives beside the identical one
`free-dataset' takes.
```

<a id="cl-gbdt-lightgbm-free-booster"></a>

## `cl-gbdt/lightgbm:free-booster`

- **Kind** function
- **Signature** `(free-booster booster)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Free BOOSTER via `LGBM_BoosterFree'. Does nothing if it was already freed, and returns no
useful value.

Signals `wrong-backend-reference' when BOOSTER is not a `lightgbm-booster' -- a dataset, a
booster built by another backend, or not a handle at all -- before anything is read from it
and before any foreign call, for the reason `free-dataset' above states and by the same
`%check-object-class'.

See `free-dataset' above for why this does not signal `backend-not-open' when BOOSTER's
backend has already been closed, but marks the handle released and `warn's the foreign
memory leaked instead -- the same cleanup-form reasoning applies here, whether the
`unwind-protect' is the caller's own or the one inside `cl-gbdt''s `with-booster'.
```

<a id="cl-gbdt-xgboost-free-booster"></a>

## `cl-gbdt/xgboost:free-booster`

- **Kind** function
- **Signature** `(free-booster booster)`
- **Exported from** `cl-gbdt/xgboost`

```text
Free BOOSTER via `XGBoosterFree'. Does nothing if it was already freed, and returns no
useful value.

Signals `wrong-backend-reference' when BOOSTER is not an `xgboost-booster' -- a dataset, a
booster built by another backend, or not a handle at all -- before anything is read from it
and before any foreign call, for the reason `free-dataset' above states and by the same
`%check-object-class'.

See `free-dataset' above for why this does not signal `backend-not-open' when BOOSTER's
backend has already been closed, but marks the handle released and `warn's the foreign memory
leaked instead -- the same cleanup-form reasoning applies here, whether the `unwind-protect'
is the caller's own or the one inside `cl-gbdt''s `with-booster'.
```

<a id="cl-gbdt-free-dataset"></a>

## `cl-gbdt:free-dataset`

- **Kind** generic function
- **Signature** `(free-dataset dataset)`
- **Exported from** `cl-gbdt`

```text
Free DATASET. Does nothing if it was already freed.
```

### Methods

#### `(free-dataset (dataset dataset))`

```text
Fallback for a DATASET whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `free-dataset' rather than freeing anything. A Layer 1
caller frees the same handle with that backend's own `free-dataset' in `api.lisp' instead.
```

#### `(free-dataset (dataset lightgbm-dataset))`

```text
Free DATASET via `LGBM_DatasetFree'. Does nothing if it was already freed.

Unlike every other operation in this file -- `free-booster' below excepted, which
takes this same path for this same reason -- this does not go through
`handle-live-pointer' and so does not signal `backend-not-open' when DATASET's
backend has already been closed. `free-dataset' runs from `with-dataset''s
`unwind-protect' cleanup form, and a non-local exit is exactly when that cleanup
runs; signalling there would replace whatever condition is already unwinding the
stack instead of letting it propagate. It `warn's instead, the foreign memory being
genuinely unreclaimable by then.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/lightgbm/api''s `free-dataset', which is where the
closed-backend branch and the measurements behind it now live.
```

#### `(free-dataset (dataset xgboost-dataset))`

```text
Free DATASET via `XGDMatrixFree'. Does nothing if it was already freed.

Unlike every other operation in this file -- `free-booster' below excepted, which takes this
same path for this same reason -- this does not go through `handle-live-pointer'
and so does not signal `backend-not-open' when DATASET's backend has already been closed
-- see `cl-gbdt/src/lightgbm/protocol''s `free-dataset' for why: this runs from
`with-dataset''s `unwind-protect' cleanup form, and signalling there would replace whatever
condition is already unwinding the stack instead of letting it propagate. It `warn's instead,
the foreign memory being genuinely unreclaimable by then.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/xgboost/api''s `free-dataset', which is where the
closed-backend branch and the reasoning behind it now live.
```

<a id="cl-gbdt-lightgbm-free-dataset"></a>

## `cl-gbdt/lightgbm:free-dataset`

- **Kind** function
- **Signature** `(free-dataset dataset)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Free DATASET via `LGBM_DatasetFree'. Does nothing if it was already freed, and returns no
useful value.

Signals `wrong-backend-reference' when DATASET is not a `lightgbm-dataset' -- a booster, a
dataset built by another backend, or not a handle at all -- before anything is read from it
and before any foreign call. This function dispatches on nothing, so `%check-object-class' is
the only thing between a wrong-kind pointer and `LGBM_DatasetFree' dereferencing it; see that
function for why the frees cannot use the same check the rest of this file uses, and for what
the two wrong handles measured against the vendored library did without it.

Unlike every other operation that reads an existing handle -- `free-booster' below excepted,
which takes this same path for this same reason -- this does not go through
`handle-live-pointer' and so does not signal `backend-not-open' when DATASET's backend has
already been closed. A free runs from a cleanup form -- the `unwind-protect' a Layer 1 caller
writes for itself, as tests/functional/lightgbm-standalone.lisp does, or the one inside
`cl-gbdt''s `with-dataset', which reaches this function through the method that delegates to
it and which a caller of `cl-gbdt/lightgbm' alone does not have -- and a non-local exit is
exactly when that cleanup runs; signalling there would replace whatever condition is already
unwinding the stack instead of letting it propagate. So when
the backend is closed, the handle is instead marked released without calling
`LGBM_DatasetFree' -- the shared library may no longer be mapped into the process, so that
call cannot be trusted not to crash -- and a `warn' reports the foreign memory as leaked,
since it is genuinely unreclaimable at that point.
```

<a id="cl-gbdt-xgboost-free-dataset"></a>

## `cl-gbdt/xgboost:free-dataset`

- **Kind** function
- **Signature** `(free-dataset dataset)`
- **Exported from** `cl-gbdt/xgboost`

```text
Free DATASET via `XGDMatrixFree'. Does nothing if it was already freed, and returns no
useful value.

Signals `wrong-backend-reference' when DATASET is not an `xgboost-dataset' -- a booster, a
dataset built by another backend, or not a handle at all -- before anything is read from it
and before any foreign call. This function dispatches on nothing, so `%check-object-class' is
the only thing between a wrong-kind pointer and `XGDMatrixFree' dereferencing it; see that
function for why the frees cannot use the same check the rest of this file uses.

Unlike every other operation that reads an existing handle -- `free-booster' below excepted,
which takes this same path for this same reason -- this does not go through
`handle-live-pointer' and so does not signal `backend-not-open' when DATASET's backend has
already been closed. A free runs from a cleanup form -- the `unwind-protect' a Layer 1 caller
writes for itself, as tests/functional/xgboost-standalone.lisp does, or the one inside
`cl-gbdt''s `with-dataset', which reaches this function through the method that delegates to it
and which a caller of `cl-gbdt/xgboost' alone does not have -- and a non-local exit is exactly
when that cleanup runs; signalling there would replace whatever
condition is already unwinding the stack instead of letting it propagate. So when the backend
is closed, the handle is instead marked released without calling `XGDMatrixFree' -- the shared
library may no longer be mapped into the process, so that call cannot be trusted not to crash
-- and a `warn' reports the foreign memory as leaked, since it is genuinely unreclaimable at
that point.
```

<a id="cl-gbdt-gbdt-error"></a>

## `cl-gbdt:gbdt-error`

- **Kind** condition
- **Superclasses** `error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Base type for every error cl-gbdt signals.
```

<a id="cl-gbdt-handle"></a>

## `cl-gbdt:handle`

- **Kind** class
- **Superclasses** `standard-object`
- **Exported from** `cl-gbdt`

```text
A CLOS wrapper around a foreign dataset or booster pointer.

Build instances with `make-handle', never directly.
```

### Slots

#### `pointer`

- **Readers** `handle-pointer`

```text
CFFI pointer to the underlying foreign object.
```

#### `released`

```text
A one-element list shared with this handle's finalizer
closure. Its CAR is true once the handle has been released.

This cannot be a plain slot: a finalizer that closed over the handle itself, to read
a slot through it, would keep the handle reachable forever and therefore never run.
The cons cell is reachable from both the handle and the closure without either one
keeping the other alive.
```

#### `backend`

- **Readers** `handle-backend`

```text
The `backend' instance that owns this handle's foreign
pointer. Read through this, never assumed still open, so `handle-live-pointer' can
tell whether that backend has since been closed -- a pointer into a backend whose
shared library `close-backend' has unmapped is exactly as dangerous as one already
freed.
```

<a id="cl-gbdt-handle-backend"></a>

## `cl-gbdt:handle-backend`

- **Kind** generic function
- **Signature** `(handle-backend object)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:handle`'s `backend` slot. See `cl-gbdt:handle`.

<a id="cl-gbdt-handle-live-pointer"></a>

## `cl-gbdt:handle-live-pointer`

- **Kind** function
- **Signature** `(handle-live-pointer handle)`
- **Exported from** `cl-gbdt`

```text
Return HANDLE's foreign pointer, or signal `released-handle-error' when HANDLE has
already been released, or `backend-not-open' when HANDLE's backend has since been
closed.

Every operation that reaches into a backend's C API must read the pointer through this
function, never `handle-pointer' directly: passing an already-freed pointer back to C
is a segfault, which is not a catchable Lisp condition -- it kills the process
outright, skipping `unwind-protect' and leaking everything else. A pointer into a
backend whose shared library `close-backend' has since closed is the same hazard --
nothing clears it -- so that is checked here too, turning it into a catchable
condition instead of a possible crash.
```

<a id="cl-gbdt-handle-pointer"></a>

## `cl-gbdt:handle-pointer`

- **Kind** generic function
- **Signature** `(handle-pointer object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:handle`'s `pointer` slot. See `cl-gbdt:handle`.

<a id="cl-gbdt-handle-released-p"></a>

## `cl-gbdt:handle-released-p`

- **Kind** function
- **Signature** `(handle-released-p handle)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Return true when HANDLE has already been released.
```

<a id="cl-gbdt-initialize-backend"></a>

## `cl-gbdt:initialize-backend`

- **Kind** generic function
- **Signature** `(initialize-backend backend &key path)`
- **Exported from** `cl-gbdt`

```text
Locate and load BACKEND's shared library.

When PATH is supplied it takes precedence over the search. On failure this signals
`backend-library-not-found', `backend-library-load-failed', or
`missing-foreign-symbols'. Expected to populate `backend-capabilities' with the plist
`probe-capabilities' returns, so `backend-supports-p' has something to read. Implemented
by each backend.
```

### Methods

#### `(initialize-backend (backend lightgbm-backend) &key path)`

```text
Load LightGBM's shared library and record its capabilities on BACKEND.

Discovery order: PATH, then *library-env-var*, then the vendored directory under
*vendor-library-directory*, then CFFI's system library search for
*default-library-name* -- see `resolve-and-load-library' for the exact rules and
the conditions each failure mode signals.

Once a library is loaded, every name in *required-symbols* must resolve via
`probe-foreign-symbols', passed the `cffi:foreign-library' just loaded as
:LIBRARY -- see that function's docstring for the SBCL caveat: it validates
the library argument but, on this platform, cannot actually scope the symbol
search to it -- or this signals `missing-foreign-symbols' -- the
version-mismatch check that function exists for. Only once that required check
has passed does this probe *optional-symbols* via `probe-capabilities' and
record the result on `backend-capabilities' -- unlike a missing required
symbol, a missing optional one never signals; it only makes
`backend-supports-p' answer NIL for the capability that symbol backs.
*provided-capabilities* goes to the same call as :PROVIDED, recording the
capabilities this backend provides unconditionally -- nothing is probed for
them, because the C functions they need are in *required-symbols* and the probe
above has already passed.

LightGBM's C API has no runtime version query, so `backend-version' is left
NIL rather than guessed --
and, unlike `cl-gbdt/src/xgboost/classes''s `initialize-backend', this never
calls `cl-gbdt/src/version''s `check-backend-version': with nothing to read, a
call here could never confirm compatibility, only ever warn on every single
open, which is not a check worth leaving in. See `*lightgbm-version-range*''s
docstring for the fuller explanation of this asymmetry between the backends.

`open-backend' only marks a backend open -- and so only calls `close-backend' on
it -- once this method returns normally. So if the symbol probe (or anything
else after the library loads) signals, the library is closed right here before
the condition propagates; otherwise it would stay mapped into the process with
BACKEND dropped and nothing left able to close it.
```

#### `(initialize-backend (backend xgboost-backend) &key path)`

```text
Load XGBoost's shared library and record its capabilities on BACKEND.

Discovery order: PATH, then *library-env-var*, then the vendored directory under
*vendor-library-directory*, then CFFI's system library search for
*default-library-name* -- see `resolve-and-load-library' for the exact rules and the
conditions each failure mode signals.

Once a library is loaded, every name in *required-symbols* must resolve via
`probe-foreign-symbols', passed the `cffi:foreign-library' just loaded as :LIBRARY -- see
that function's docstring for the SBCL caveat: it validates the library argument but,
on this platform, cannot actually scope the symbol search to it -- or this signals
`missing-foreign-symbols'.

Only once that required check has passed does this probe *optional-symbols* via
`probe-capabilities' and record the result on `backend-capabilities' -- unlike a missing
required symbol, a missing optional one never signals; it only makes `backend-supports-p'
answer NIL for the capability that symbol backs. *provided-capabilities* goes to the same
call as :PROVIDED, recording the capabilities this backend provides unconditionally --
nothing is probed for them, because the C functions they need are in *required-symbols* and
the probe above has already passed.

Once `backend-version' is read, `check-backend-version' compares it against
`*xgboost-version-range*' and signals `untested-backend-version' -- a warning, not an
error -- when it falls outside that range's recorded bounds. This is the only backend
this runs for: LightGBM's C API has no version entry point, so `backend-version' stays
NIL there and there is nothing to compare -- see
`cl-gbdt/src/lightgbm/classes''s `initialize-backend'.

`open-backend' only marks a backend open -- and so only calls `close-backend' on it --
once this method returns normally. So if the symbol probe (or anything else after the
library loads) signals, the library is closed right here before the condition propagates;
otherwise it would stay mapped into the process with BACKEND dropped and nothing left able
to close it.
```

<a id="cl-gbdt-lightgbm-lightgbm-backend"></a>

## `cl-gbdt/lightgbm:lightgbm-backend`

- **Kind** class
- **Superclasses** `backend`
- **Exported from** `cl-gbdt/lightgbm`

```text
A connection to the LightGBM shared library, implementing
cl-gbdt's unified backend protocol.
```

### Slots

#### `foreign-library`

```text
The `cffi:foreign-library' `initialize-backend'
loaded, kept so `shutdown-backend' can close exactly this one.
```

<a id="cl-gbdt-load-model"></a>

## `cl-gbdt:load-model`

- **Kind** generic function
- **Signature** `(load-model backend path)`
- **Exported from** `cl-gbdt`

```text
Load a model from PATH and return a BACKEND booster.
```

### Methods

#### `(load-model (backend backend) (path t))`

```text
Fallback for a BACKEND whose unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `load-model'. A Layer 1 caller loads a model with that
backend's own `load-model' in `api.lisp' instead, which returns a handle without dispatching
through this generic at all.
```

#### `(load-model (backend lightgbm-backend) (path t))`

```text
Load a LightGBM model from PATH via `LGBM_BoosterCreateFromModelfile' and
return a new booster.

The returned booster has no training set -- see the `booster' class'
documentation -- since PATH names a model, not a dataset.

The raw booster handle exists in C from the moment `LGBM_BoosterCreateFromModelfile'
returns, but `make-handle' does not take ownership of it until it also succeeds --
mirroring `cl-gbdt/src/xgboost/protocol''s `load-model', which reaches for the same
`with-pointer-ownership' macro for the same reason: nothing here guarantees
`make-handle' cannot signal, and a raw handle it never took ownership of would
otherwise be orphaned rather than freed.

Signals `backend-not-open' before the foreign call when BACKEND is not open --
see `%check-backend-open'.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/lightgbm/api''s `load-model', which is where the
ownership window, the null-handle check and the backend guard now live.
```

#### `(load-model (backend xgboost-backend) (path t))`

```text
Load an XGBoost model from PATH and return a new booster.

Unlike LightGBM's `LGBM_BoosterCreateFromModelfile', which allocates the booster and
loads the model in a single call, XGBoost splits the two: `XGBoosterCreate' first builds
a booster with no DMatrix handles at all -- see `%create-booster' -- and only then does
`XGBoosterLoadModel' populate it from PATH.

The returned booster has no training set -- see the `booster' class' documentation --
since PATH names a model, not a dataset.

The raw booster handle exists in C from the moment `XGBoosterCreate' returns, but
`make-handle' does not take ownership of it until `XGBoosterLoadModel' has also
succeeded. `with-pointer-ownership' spans exactly that gap: the pointer is owned by
nobody inside its body, and any exit that has not called TAKE-OWNERSHIP -- a failing
`XGBoosterLoadModel' the likeliest -- frees the raw booster here instead of orphaning it.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/xgboost/api''s `load-model', which is where the
two-call construction, the ownership window and the backend guard now live.
```

<a id="cl-gbdt-lightgbm-load-model"></a>

## `cl-gbdt/lightgbm:load-model`

- **Kind** function
- **Signature** `(load-model backend path)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Load a LightGBM model from PATH via `LGBM_BoosterCreateFromModelfile' and return a new
booster built against BACKEND.

The returned booster has no training set -- see the `booster' class' documentation -- since
PATH names a model, not a dataset. `evaluation' on it therefore reports nothing, and
`update-one-iteration' signals `missing-training-set'.

The raw booster handle exists in C from the moment `LGBM_BoosterCreateFromModelfile' returns,
but `make-handle' does not take ownership of it until it also succeeds; `with-pointer-ownership'
spans exactly that gap, freeing the raw handle on any exit that has not taken ownership rather
than orphaning it.

Signals `wrong-backend-reference' when BACKEND is not a `lightgbm-backend' -- the other
backend's, a handle, or not a backend at all -- checked FIRST, ahead of `%check-backend-open',
which asks only whether the object is open and answers that truthfully for the wrong library.
Signals `backend-not-open' when BACKEND is closed, and `foreign-call-error' when the library
reports success but returns a null handle.
```

<a id="cl-gbdt-xgboost-load-model"></a>

## `cl-gbdt/xgboost:load-model`

- **Kind** function
- **Signature** `(load-model backend path)`
- **Exported from** `cl-gbdt/xgboost`

```text
Load an XGBoost model from PATH and return a new booster built against BACKEND.

Unlike LightGBM's `LGBM_BoosterCreateFromModelfile', which allocates the booster and loads the
model in one call, XGBoost splits the two: `XGBoosterCreate' first builds a booster with no
DMatrix handles at all, and only then does `XGBoosterLoadModel' populate it from PATH.

The returned booster has no training set -- see the `booster' class' documentation -- since
PATH names a model, not a dataset. `evaluation' on it therefore reports nothing, and
`update-one-iteration' signals `missing-training-set'.

`with-pointer-ownership' spans exactly the window in which the raw booster is owned by
nobody: any exit that has not called TAKE-OWNERSHIP -- a failing `XGBoosterLoadModel' the
likeliest -- frees it here instead of orphaning it.

Signals `wrong-backend-reference' when BACKEND is not an `xgboost-backend', checked FIRST,
ahead of `%check-backend-open', which asks only whether the object is open and answers that
truthfully for the wrong library. Signals `backend-not-open' when BACKEND is closed.
```

<a id="cl-gbdt-make-csr-matrix"></a>

## `cl-gbdt:make-csr-matrix`

- **Kind** function
- **Signature** `(make-csr-matrix &key indptr indices values num-columns)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Return a `csr-matrix' holding INDPTR, INDICES and VALUES in standard CSR layout, one
NUM-COLUMNS wide.

INDPTR, INDICES and VALUES may each be any sequence -- list or vector -- and come back
already coerced to the specialized arrays `csr-matrix-indptr', `csr-matrix-indices' and
`csr-matrix-values' store; see the struct's own docstring for exactly which types, and
for why NUM-COLUMNS is required rather than inferred.

Signals `dimension-mismatch' when NUM-COLUMNS is not a positive integer; when INDICES
and VALUES have different lengths; when INDPTR does not start at 0, decreases anywhere,
or disagrees with INDICES/VALUES' shared length; or when an element of INDICES falls
outside [0, NUM-COLUMNS). Signals `unsupported-element-type' when an element of VALUES
is not a real number, or when an element of INDPTR or INDICES is not representable as
`(signed-byte 32)' -- the type both are stored as, so neither a non-integer nor an
integer too large for it reaches `%coerce-index-vector' as a raw `type-error'. Every
check runs against INDPTR, INDICES and VALUES before any of the three is coerced, so a
malformed matrix is rejected without paying for the copy a valid one needs.
```

<a id="cl-gbdt-make-dataset"></a>

## `cl-gbdt:make-dataset`

- **Kind** generic function
- **Signature** `(make-dataset backend matrix &key label weight group feature-names parameters reference missing categorical-features)`
- **Exported from** `cl-gbdt`

```text
Build a training dataset for BACKEND from MATRIX.

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
'("a" . "b"))' signals a raw `type-error' and `(length circular)' does not return, both
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
becomes the `"feature_type"' field it attaches to a finished DMatrix. LightGBM provides it
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

Free the result with `free-dataset' or wrap it in `with-dataset'.
```

### Methods

#### `(make-dataset (backend backend) (matrix t) &key label weight group feature-names parameters reference missing categorical-features)`

```text
Fallback for a BACKEND whose unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `make-dataset' rather than building anything from MATRIX.
A Layer 1 caller who does not need the unified API builds a dataset directly with that
backend's own `create-dataset' instead.
```

#### `(make-dataset (backend lightgbm-backend) (matrix t) &key label weight group feature-names parameters reference missing categorical-features)`

```text
Build a LightGBM dataset from MATRIX -- a dense matrix via `LGBM_DatasetCreateFromMat',
a `csr-matrix' via `LGBM_DatasetCreateFromCSR' -- attaching LABEL, WEIGHT and GROUP with
`LGBM_DatasetSetField' and FEATURE-NAMES with `LGBM_DatasetSetFeatureNames' when supplied.
See the `make-dataset' generic function's docstring for what each argument means,
including REFERENCE, and for what a `csr-matrix' changes about none of them.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `cl-gbdt/src/lightgbm/api''s
`%dataset-pointer', which checks it. Every other argument behaves identically either way:
PARAMETERS and REFERENCE reach the sparse entry point as the same two C parameters they
reach the dense one as, and LABEL, WEIGHT, GROUP and FEATURE-NAMES are attached to the
finished dataset by `create-dataset', which never sees which entry point built it.

Signals `capability-unavailable' naming `:missing-value' for a non-NIL MISSING, whatever the
value is and whatever form MATRIX takes -- this backend has no C-API route for a missing-value
sentinel at all. See `%check-missing-value' above, which carries the reasoning, including why
a NaN LightGBM would in fact honour is refused with the rest. MISSING NIL, the default, is
this backend's own default and reaches no check: every call that does not name a sentinel
behaves exactly as it did before the argument existed.

CATEGORICAL-FEATURES names which columns of MATRIX hold categories rather than quantities,
and becomes a `categorical_feature' entry in the parameter string -- this backend's own name
for the list, rendered by `categorical-feature-string' from the caller's own MATRIX. It is
appended after PARAMETERS' own entries, and reaches whichever creation entry point MATRIX's
form selects without a branch of its own: the string is built once, and neither
`LGBM_DatasetCreateFromMat' nor `LGBM_DatasetCreateFromCSR' can tell which of the two it was
built for.

CATEGORICAL-FEATURES NIL, the default, adds nothing to that string -- not an empty entry --
so a call that names no categorical column builds exactly the dataset it built before this
argument existed. A non-NIL value signals `capability-unavailable' naming
`:categorical-features' when the capability reads false (see `%check-categorical-features'
above, which reads the capability rather than this backend's name) and `unsupported-argument'
naming `:categorical-features' for an index that is not an integer, is negative, is beyond
MATRIX's last column, or was named twice.

Signals `unsupported-argument' naming "make-dataset's :parameters" when CATEGORICAL-FEATURES
is supplied AND PARAMETERS carries any of the five spellings LightGBM honours for that key --
`categorical_feature', `cat_feature', `categorical_column', `cat_column' or
`categorical_features'. Only the two together: the entry this method appends would land after
the caller's and LightGBM keeps the FIRST occurrence of a duplicated key, so the argument the
caller explicitly named would be the one silently discarded.

PARAMETERS alone is untouched by any of this. A caller who names no categorical column and
writes `categorical_feature' there by hand is on policy section 6's escape hatch for a
backend's own vocabulary, and gets it honoured exactly as they did before this argument
existed. `cat_features' is never refused either way, the library not honouring it as an alias.
See `%check-categorical-parameter-keys' above for the measurements behind all of that.

Signals `foreign-call-error' when dataset creation reports success but writes a
null handle -- a library-contract violation, but one every later call through
this handle would otherwise dereference blindly. Signals `wrong-backend-reference'
when REFERENCE is supplied but is not a `lightgbm-dataset', `released-handle-error'
when it has already been freed, and `backend-not-open' when its backend has since
been closed -- see `%reference-pointer'.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'.

The procedure itself is Layer 1 and lives in `cl-gbdt/src/lightgbm/api''s `create-dataset':
building the pointer, attaching LABEL, WEIGHT, GROUP and FEATURE-NAMES in that order, and the
ownership dance that frees the raw dataset when one of those signals. What is left here is
the portable contract -- the three checks above, and rendering CATEGORICAL-FEATURES into the
one parameter-string key this backend states it as. Everything the paragraphs above promise
about a null handle, about REFERENCE and about the raw handle's ownership is that function's
doing; see its own docstring.
```

#### `(make-dataset (backend xgboost-backend) (matrix t) &key label weight group feature-names parameters reference missing categorical-features)`

```text
Build an XGBoost dataset (a DMatrix) from MATRIX -- a dense matrix via
`XGDMatrixCreateFromDense', a `csr-matrix' via `XGDMatrixCreateFromCSR' -- attaching LABEL
and WEIGHT with `XGDMatrixSetInfoFromInterface', GROUP with `XGDMatrixSetUIntInfo', and
FEATURE-NAMES with `XGDMatrixSetStrFeatureInfo' when supplied. See the `make-dataset'
generic function's docstring for what each argument means.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `cl-gbdt/src/xgboost/api''s `%dataset-pointer',
which checks it. LABEL, WEIGHT, GROUP and FEATURE-NAMES behave identically either way: they
are attached to the finished DMatrix by `create-dataset', which never sees which entry point
built it. REFERENCE and PARAMETERS are refused for a `csr-matrix' exactly as they are for a
dense matrix, and for the same reasons, spelled out below.

MISSING, the value that means *missing*, becomes the `"missing"' key of whichever creation
config JSON MATRIX's form reaches. It needs this backend's `:missing-value' capability, which
`%check-missing-value' re-checks below rather than trusting the caller to have asked, and it
signals `unsupported-argument' for anything that is neither a `real' nor NIL -- see
`missing-value-json', which renders it. NIL, the default, sends the IEEE NaN this backend
sent unconditionally before the argument existed, so a caller who passes nothing gets exactly
what they got before. The comparison the library then makes is at SINGLE precision, whatever
MATRIX's own element type: two `double-float's that share a `single-float' both count as
missing against a sentinel that narrows to it.

CATEGORICAL-FEATURES, a list of 0-based column indices, is attached with the same
`XGDMatrixSetStrFeatureInfo' FEATURE-NAMES uses, under the `"feature_type"' field instead of
`"feature_name"' -- one string per column, `"c"' for a named column and `"q"' for every
other, as `categorical-feature-types' renders them. It needs this backend's
`:categorical-features' capability, which `%check-categorical-features' re-checks below rather
than trusting the caller to have asked, and it signals `unsupported-argument' naming
`:categorical-features' for an index that is not an integer, is negative, is beyond MATRIX's
last column, or was named twice. NIL, the default, attaches no `"feature_type"' at all --
exactly what every call sent before the argument existed, not a vector of `"q"'.

The list is rendered from the CALLER's MATRIX, before `create-dataset' builds anything, so a
bad index signals with no DMatrix yet allocated and the range check is made against the same
count `cl-gbdt/src/lightgbm/protocol''s `make-dataset' checks against. The attachment then has
to wait until after creation, `XGDMatrixSetStrFeatureInfo' needing a handle -- which is also
why a `csr-matrix' needs nothing of its own here: the two creation branches have converged by
the time it runs, and the renderer reads a `csr-matrix''s declared column count where it reads
a dense matrix's second dimension.

Measured, and the reason a dataset that builds here can still fail later: `tree_method exact'
refuses categorical features at `train', not at `make-dataset'. The DMatrix is built and the
types attached without complaint, and `XGBoosterUpdateOneIter' then returns -1 with
`Updater `grow_colmaker` or `exact` tree method doesn't support categorical features'. That is
:PARAMETERS' business, not this method's -- `hist' and `approx' both work -- and nothing here
pre-validates an updater it is not given.

REFERENCE and PARAMETERS both signal `unsupported-argument' rather than being silently
dropped: REFERENCE is a LightGBM-only concept -- aligning a new dataset's bin mapper to an
existing one's, which XGBoost has nothing resembling. PARAMETERS is more subtle: the
vendored header (`ffi-spec/xgboost/include/xgboost/c_api.h') documents exactly three keys
for `XGDMatrixCreateFromDense''s config JSON -- `"missing"', which now has its own
:MISSING argument above and so is not what a caller reaches for PARAMETERS to set,
`"nthread"' and `"data_split_mode"' -- none of which correspond to what a caller moving
a working call from LightGBM actually means by dataset-level PARAMETERS there: binning knobs
such as
`max_bin' and `min_data_in_bin'. Forwarding `normalize-parameters''s output into that
config JSON regardless would not raise anything either: confirmed empirically against the
vendored library, `XGDMatrixCreateFromDense' returns success and silently ignores an
unrecognized config key rather than rejecting it, which would just move today's silent
drop one layer deeper, into C, instead of fixing it. The same holds for a `csr-matrix': that
header documents `XGDMatrixCreateFromCSR''s config by cross-reference to
`XGDMatrixCreateFromDense', so it is the same three keys either way, and the refusal below
names whichever of the two the caller's own MATRIX would have reached -- see
`cl-gbdt/src/xgboost/api''s `%creation-function-name', which words that name where the calls
it names are made. Either PARAMETERS or REFERENCE accepted and discarded here would let a
caller move a working `make-dataset' call from LightGBM to XGBoost and get a dataset that
looks fine but was not built the way the caller asked, which is exactly the failure mode this
project keeps finding.

Signals `foreign-call-error' when dataset creation reports success but writes a null
handle -- a library-contract violation, but one every later call through this handle would
otherwise dereference blindly.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'.

The procedure itself is Layer 1 and lives in `cl-gbdt/src/xgboost/api''s `create-dataset':
building the pointer, attaching LABEL, WEIGHT, GROUP, FEATURE-NAMES and the rendered feature
types in that order, and the ownership dance that frees the raw DMatrix when one of those
signals. What is left here is the portable contract -- the three capability checks above, the
two refusals, and rendering CATEGORICAL-FEATURES into the feature-type strings this backend
states them as. Everything the paragraphs above promise about a null handle and about the raw
handle's ownership is that function's doing; see its own docstring.
```

<a id="cl-gbdt-make-handle"></a>

## `cl-gbdt:make-handle`

- **Kind** function
- **Signature** `(make-handle class-name pointer backend kind &key training-set validation-sets best-iteration)`
- **Exported from** `cl-gbdt`

```text
Create an instance of CLASS-NAME wrapping POINTER for BACKEND, with a finalizer
attached that warns `unfreed-handle-warning' (naming KIND) if the instance is
garbage-collected before `release-handle' runs on it.

TRAINING-SET, VALIDATION-SETS and BEST-ITERATION, when supplied, become the new
instance's `training-set', `validation-sets' and `best-iteration' initargs; only
`booster' has slots for them. Free the result with `release-handle'.
```

<a id="cl-gbdt-make-training-report"></a>

## `cl-gbdt:make-training-report`

- **Kind** function
- **Signature** `(make-training-report &key series num-rounds best-iteration best-score early-stopped-p)`
- **Exported from** `cl-gbdt`

```text
Return a `training-report' over SERIES, recorded across NUM-ROUNDS iterations.

BEST-ITERATION, BEST-SCORE and EARLY-STOPPED-P default to NIL, which is the honest value for
a run that was not given :EARLY-STOPPING; a caller that was given it passes what its watcher
found. See `training-report-best-iteration', `-best-score' and `-early-stopped-p' for why NIL
means "not determined" rather than an invented default.
```

<a id="cl-gbdt-make-training-series"></a>

## `cl-gbdt:make-training-series`

- **Kind** function
- **Signature** `(make-training-series &key index name metric values)`
- **Exported from** `cl-gbdt`

```text
Return a `training-series' for METRIC on the dataset at INDEX, optionally called NAME.

VALUES is one element per completed iteration; see the slot's own documentation for why a NIL
element is legal.
```

<a id="cl-gbdt-make-version-range"></a>

## `cl-gbdt:make-version-range`

- **Kind** function
- **Signature** `(make-version-range &key ((:verified-low verified-low) nil) ((:verified-high verified-high) nil) ((:verified-evidence verified-evidence) nil) ((:inferred-low inferred-low) nil) ((:inferred-high inferred-high) nil) ((:inferred-evidence inferred-evidence) nil))`
- **Exported from** `cl-gbdt`

Constructor of the `cl-gbdt:version-range` structure. See `cl-gbdt:version-range`.

<a id="cl-gbdt-missing-foreign-symbols"></a>

## `cl-gbdt:missing-foreign-symbols`

- **Kind** condition
- **Superclasses** `backend-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
The loaded library lacks functions cl-gbdt needs.

Detected by symbol probing. This is the most reliable signal of a version mismatch.
```

### Slots

#### `names`

- **Readers** `missing-foreign-symbols-names`

```text
List of C function names that were not found.
```

<a id="cl-gbdt-missing-foreign-symbols-names"></a>

## `cl-gbdt:missing-foreign-symbols-names`

- **Kind** generic function
- **Signature** `(missing-foreign-symbols-names condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:missing-foreign-symbols`'s `names` slot. See `cl-gbdt:missing-foreign-symbols`.

<a id="cl-gbdt-missing-training-set"></a>

## `cl-gbdt:missing-training-set`

- **Kind** condition
- **Superclasses** `data-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
UPDATE-ONE-ITERATION was called on a booster with no training set.

A `load-model' booster has no training set -- see the `booster' class' documentation --
so there is no dataset to advance it against. XGBoost's `XGBoosterUpdateOneIter' takes
the training DMatrix as an explicit argument, unlike LightGBM's, which reads its own
internal training-set pointer implicitly; a null DMatrixHandle would not come back as a
status code the way a bad parameter does, but as a null-pointer dereference inside
XGBoost's own implementation. This is signalled before that foreign call runs, for the
same reason `%check-booster-datasets-live' checks the pointers it does.
```

### Slots

#### `booster`

- **Readers** `missing-training-set-booster`

```text
The booster UPDATE-ONE-ITERATION was called on.
```

<a id="cl-gbdt-missing-training-set-booster"></a>

## `cl-gbdt:missing-training-set-booster`

- **Kind** generic function
- **Signature** `(missing-training-set-booster condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:missing-training-set`'s `booster` slot. See `cl-gbdt:missing-training-set`.

<a id="cl-gbdt-model-to-string"></a>

## `cl-gbdt:model-to-string`

- **Kind** generic function
- **Signature** `(model-to-string booster &key num-iteration)`
- **Exported from** `cl-gbdt`

```text
Return BOOSTER's model as a string.

NUM-ITERATION behaves as it does for `save-model', `:best' included: LightGBM honors it,
nil meaning every round; XGBoost has no iteration-limited variant of this call and signals
`unsupported-argument' when NUM-ITERATION is supplied, an explicit integer or `:best'
resolved to one alike.
```

### Methods

#### `(model-to-string (booster booster) &key num-iteration)`

```text
Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `model-to-string' rather than returning anything. A
Layer 1 caller gets the same string with that backend's own `model-to-string' in `api.lisp'
instead.
```

#### `(model-to-string (booster lightgbm-booster) &key num-iteration)`

```text
Return BOOSTER's model as a string via `LGBM_BoosterSaveModelToString'.

NUM-ITERATION's :BEST is resolved by `%resolve-best-num-iteration' before
`%resolve-num-iteration' ever sees it, exactly as `predict' and `save-model' resolve it.

The procedure is Layer 1 and lives in `cl-gbdt/src/lightgbm/api''s `model-to-string'. What is
left here is resolving :BEST, exactly as `save-model' above.
```

#### `(model-to-string (booster xgboost-booster) &key num-iteration)`

```text
Return BOOSTER's model as a JSON string via `XGBoosterSaveModelToBuffer'.

Signals `unsupported-argument' when NUM-ITERATION is supplied: `XGBoosterSaveModelToBuffer''s
config JSON has no iteration-limiting key, only `"format"' -- see `save-model' for the
same guard on the sibling entry point, and for why silently ignoring it is not an option.
:BEST is resolved by `%resolve-best-num-iteration' first, into an integer, which then
meets this same check exactly as an explicit integer would.

`out_dptr' is XGBoost's own memory, copied out via `foreign-string-to-lisp' with an
explicit `:count' from `out_len' rather than trusted to be null-terminated at the right
place.

The procedure is Layer 1 and lives in `cl-gbdt/src/xgboost/api''s `model-to-string', which
takes no :NUM-ITERATION at all. What is left here is the refusal above, which exists because
the unified API promised a portable argument LightGBM honours and this library has no route
for.
```

<a id="cl-gbdt-lightgbm-model-to-string"></a>

## `cl-gbdt/lightgbm:model-to-string`

- **Kind** function
- **Signature** `(model-to-string booster &key num-iteration)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Return BOOSTER's model as a string via `LGBM_BoosterSaveModelToString'.

NUM-ITERATION means what it means for `save-model' above, :BEST refused on the same terms and
by the same call. The text this returns is the text `save-model' writes, so it can be written
to a file and handed back to `load-model'.

Signals `wrong-backend-reference', `released-handle-error' and `backend-not-open' exactly as
`save-model' does, and for the same reason: this function dispatches on nothing.
```

<a id="cl-gbdt-xgboost-model-to-string"></a>

## `cl-gbdt/xgboost:model-to-string`

- **Kind** function
- **Signature** `(model-to-string booster)`
- **Exported from** `cl-gbdt/xgboost`

```text
Return BOOSTER's model as a JSON string via `XGBoosterSaveModelToBuffer'.

Takes no iteration limit, for the reason `save-model' above states: that entry point's config
JSON has only a `"format"' key. The text this returns is a complete model document and can
be written to a `.json' file and handed back to `load-model'.

`out_dptr' is XGBoost's own memory, copied out via `foreign-string-to-lisp' with an explicit
`:count' from `out_len' rather than trusted to be null-terminated at the right place.

Signals `wrong-backend-reference', `released-handle-error' and `backend-not-open' exactly as
`save-model' does, and for the same reason: this function dispatches on nothing.
```

<a id="cl-gbdt-normalize-parameters"></a>

## `cl-gbdt:normalize-parameters`

- **Kind** function
- **Signature** `(normalize-parameters plist)`
- **Exported from** `cl-gbdt`

```text
Return PLIST as a list of (NAME . VALUE) string pairs, in order.

Nothing is validated or filtered: a backend-specific parameter passes through untouched,
which is what `make-dataset' and `train' promise. Signals `data-error' when PLIST has an
odd length, because a silently dropped final key is how a parameter goes missing.
```

<a id="cl-gbdt-open-backend"></a>

## `cl-gbdt:open-backend`

- **Kind** function
- **Signature** `(open-backend name &key path)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Open the backend NAME and return a `backend' instance.

PATH, when supplied, takes precedence over the shared library search. Signals a
condition when NAME is unregistered or initialization fails. Close a successful
instance with `close-backend'.
```

<a id="cl-gbdt-predict"></a>

## `cl-gbdt:predict`

- **Kind** generic function
- **Signature** `(predict booster matrix &key kind num-iteration missing)`
- **Exported from** `cl-gbdt`

```text
Predict on MATRIX using BOOSTER.

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
supplied. NIL keeps meaning "every round" on every booster, including one with a best
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
present rather than that the library said anything.
```

### Methods

#### `(predict (booster booster) (matrix t) &key kind num-iteration missing)`

```text
Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `predict' rather than predicting anything on MATRIX. A
Layer 1 caller predicts with that backend's own `predict' in `api.lisp' instead.
```

#### `(predict (booster lightgbm-booster) (matrix t) &key (kind :normal) num-iteration missing)`

```text
Predict on MATRIX with BOOSTER -- a dense matrix via `LGBM_BoosterPredictForMat', a
`csr-matrix' via `LGBM_BoosterPredictForCSR'.

KIND and NUM-ITERATION are as the `predict' generic function documents. NUM-ITERATION's
:BEST is resolved HERE, by `%resolve-best-num-iteration', and the integer it produces is what
reaches the procedure: `booster-best-iteration' is written by `train' and by nothing else, so
:BEST is a Layer 2 concept and `cl-gbdt/src/lightgbm/api''s `predict' takes an integer or NIL,
refusing the keyword itself with `unsupported-argument' -- see its docstring, which says so
from the other side. Predictions start from iteration 0 -- the protocol exposes no
start-iteration override.

Signals `capability-unavailable' naming `:missing-value' for a non-NIL MISSING, whatever the
value is and whatever form MATRIX takes -- this backend has no C-API route for a
missing-value sentinel at all, on the prediction path any more than on the ingestion one. See
`%check-missing-value' above, which carries the reasoning; `make-dataset' calls that same
function, and this is a second call site rather than a copy of it, because policy section 7
asks each operation to re-check the capability for itself. MISSING NIL, the default, reaches
no check: every prediction that names no sentinel behaves exactly as it did before the
argument existed.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `cl-gbdt/src/lightgbm/api''s
`%check-sparse-input', which checks it before any foreign call. Everything else means exactly
what it means for a dense matrix, `csr-matrix' or not: KIND and NUM-ITERATION are honoured
identically on either path -- all four KINDs included, unlike `cl-gbdt/src/xgboost/api''s
`predict', whose sparse entry point is XGBoost's inplace prediction and covers only two of
them.

Returns the result array and, as a second value, the SHAPE this backend states for it -- a
list of integers in `array-dimensions' order, or NIL where it states none. Nothing here
reports axes the way XGBoost's `out_shape'/`out_dim' pair does, so that value is DERIVED
rather than read back: `:normal' and `:raw' get the result array's own dimensions, `:contrib'
the three axes `contrib-shape' divides the element count into (NIL for any of the four cases
that function's own docstring enumerates), and `:leaf-index' NIL. `%prediction-shape', beside
the procedure in `cl-gbdt/src/lightgbm/api', is where that happens and carries the
measurements. This backend declares `:prediction-shape' in `*provided-capabilities*' to say
the mechanism is here; nothing re-checks that declaration, there being no argument to refuse.

The procedure itself is Layer 1 and lives in `cl-gbdt/src/lightgbm/api''s `predict': the
choice of entry point, the buffer sized from `LGBM_BoosterCalcNumPredict', the OUT-LEN
assertion, the copy-out and the derived shape, together with the `:sparse-input' gate and the
deliberate absence of any NaN or infinity scan over the result. Everything the paragraphs
above promise about those is that function's doing; see its own docstring. What is left here
is the portable contract: the :MISSING gate this backend answers with a refusal, and
resolving :BEST.

Refusing a KIND this backend has no prediction type for moved BELOW BOTH of those, and is
the one thing about this method a caller can observe changing: `%predict-type''s `ecase'
now runs inside the procedure rather than in the same `let' that read the pointer, so a
call wrong in two ways at once is answered by whichever check still runs first. Measured
through `cl-gbdt:predict' against the vendored library, on a booster trained without
:EARLY-STOPPING and so with no best iteration: a bad KIND together with a non-NIL :MISSING
signalled `sb-kernel:case-failure' before the split and signals `capability-unavailable'
now; a bad KIND together with `:num-iteration :best' signalled `sb-kernel:case-failure' and
signals `unsupported-argument'. A bad KIND alone is `sb-kernel:case-failure' either way.
Both changes put a typed `cl-gbdt' condition where an untyped one used to escape, and the
old order could not be restored without calling `%predict-type' here purely for effect,
duplicating a check the procedure already makes.
```

#### `(predict (booster xgboost-booster) (matrix t) &key (kind :normal) num-iteration missing)`

```text
Predict on MATRIX with BOOSTER -- a dense matrix via `XGBoosterPredictFromDMatrix', a
`csr-matrix' via `XGBoosterPredictFromCSR'.

KIND and NUM-ITERATION are as the `predict' generic function documents. NUM-ITERATION's
:BEST is resolved HERE, by `%resolve-best-num-iteration', and the integer it produces is what
reaches the procedure: `booster-best-iteration' is written by `train' and by nothing else, so
:BEST is a Layer 2 concept and `cl-gbdt/src/xgboost/api''s `predict' takes an integer or NIL,
refusing the keyword itself with `unsupported-argument' -- see its docstring, which says so
from the other side and measures what the keyword did before that refusal existed.
Predictions start from iteration 0 -- the protocol exposes no start-iteration override.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `cl-gbdt/src/xgboost/api''s
`%check-sparse-input', which checks it before any foreign call.

MISSING, the value in MATRIX that means *missing*, needs this backend's `:missing-value'
capability, which `%check-missing-value' re-checks below before any foreign call rather than
inheriting the check `make-dataset' made on the dataset BOOSTER was trained from -- policy
section 7 asks each operation to check for itself. That CAPABILITY GATE is the whole of what
this method does with the argument; the value itself is passed down unexamined and untouched.
It signals `unsupported-argument' for anything that is neither a `real' nor NIL, see
`missing-value-json', and NIL -- the default -- sends the IEEE NaN this backend sent
unconditionally before the argument existed, so a caller who passes nothing predicts exactly
what they predicted before. Which config JSON the sentinel then reaches depends on MATRIX's
own form and is the procedure's business, not this method's -- see
`cl-gbdt/src/xgboost/api''s `predict', which words that split from the other side.

Nothing here relates MISSING to the sentinel BOOSTER's training dataset was built with:
XGBoost does not record a DMatrix's sentinel on the booster, so the two are independent and
their disagreement is undetectable. See the `predict' generic function's docstring, where
that is stated as the caller's responsibility.

Returns the result array and, as a second value, the SHAPE this backend states for it -- a
list of integers in `array-dimensions' order. It is never NIL here, unlike
`cl-gbdt/src/lightgbm/protocol''s `predict': this library REPORTS a shape, through the
`out_shape'/`out_dim' pair both entry points write, and the procedure hands that report back
verbatim rather than deriving anything. `cl-gbdt/src/xgboost/api''s `predict' is where the
read-back lives and carries the measurements, the `out_dim' values per KIND among them. This
backend declares `:prediction-shape' in `*provided-capabilities*' to say the mechanism is
here; nothing re-checks that declaration, there being no argument to refuse.

The procedure itself is Layer 1 and lives in `cl-gbdt/src/xgboost/api''s `predict': the
choice of entry point, the transient DMatrix the dense path builds and frees, the sparse
path's restriction to `:normal' and `:raw', the shape read-back, the copy-out of
`out_result', together with the `:sparse-input' gate and the deliberate absence of any NaN or
infinity scan over the result. Everything the paragraphs above promise about those is that
function's doing; see its own docstring. What is left here is the portable contract: the
:MISSING capability gate, and resolving :BEST.

Refusing a KIND this backend has no prediction type for moved BELOW BOTH of those, and is
the one thing about this method a caller can observe changing: `%predict-type''s `ecase' now
runs inside the procedure rather than in the same `let' that read the pointer, so a call
wrong in two ways at once is answered by whichever check still runs first. Measured through
`cl-gbdt:predict' against the vendored library, on a booster trained without :EARLY-STOPPING
and so with no best iteration: a bad KIND together with `:num-iteration :best' signalled
`sb-kernel:case-failure' before the split and signals `unsupported-argument' now, putting a
typed `cl-gbdt' condition where an untyped one used to escape. A bad KIND alone is
`sb-kernel:case-failure' either way, and so is a bad KIND together with a non-NIL :MISSING --
the gate above never refuses while `:missing-value' reads true, which is the one row where
this backend differs from `cl-gbdt/src/lightgbm/protocol''s `predict', whose gate always
refuses and which therefore changed on that pair too. The old order could not be restored
without calling `%predict-type' here purely for effect, duplicating a check the procedure
already makes.
```

<a id="cl-gbdt-lightgbm-predict"></a>

## `cl-gbdt/lightgbm:predict`

- **Kind** function
- **Signature** `(predict booster matrix &key (kind :normal) num-iteration)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Predict on MATRIX with BOOSTER, returning two values: the result array and the SHAPE this
backend states for it.

MATRIX is a dense matrix -- predicted through `LGBM_BoosterPredictForMat' -- or a `csr-matrix',
through `LGBM_BoosterPredictForCSR'. KIND is `:normal', `:raw', `:leaf-index' or `:contrib',
mapped onto LightGBM's own `C_API_PREDICT_*' constant by `%predict-type', which signals for
anything else. Predictions start from iteration 0; nothing here exposes a start-iteration
override.

NUM-ITERATION is a positive integer, or NIL for every iteration -- which LightGBM spells as 0,
and `%resolve-num-iteration' is what writes it that way. :BEST is REFUSED, with
`unsupported-argument' naming this backend and "predict's :num-iteration": only `train' writes
a booster's `best-iteration', and a booster built by `create-booster' has none, so at this layer
the keyword would name an empty slot. `%reject-best-num-iteration' is what refuses it, and its
own docstring measures why the refusal has to be explicit -- `%resolve-num-iteration' is
`(or num-iteration 0)', so :BEST would otherwise reach a foreign call expecting an integer as
uninterpreted data and come back a raw CFFI `type-error' rather than a `cl-gbdt' condition. The
refusal is invisible to Layer 2: `cl-gbdt/src/lightgbm/protocol''s `predict' method resolves
:BEST first -- by `%resolve-best-num-iteration', which signals `unsupported-argument' when the
booster has no best iteration to resolve it against -- and calls this with the integer that
resolution produced, so the keyword itself never arrives from there.

Signals `capability-unavailable' naming `:sparse-input' when MATRIX is a `csr-matrix' and that
capability reads false -- see `%check-sparse-input' above, which checks it before any foreign
call. Everything else means exactly what it means for a dense matrix: both entry points take
the same PREDICT-TYPE, the same START-ITERATION/NUM-ITERATION pair and the same parameter
string, and both fill the same buffer in the same row-major order, so KIND and NUM-ITERATION
are honoured identically on either path -- all four KINDs included, unlike
`cl-gbdt/src/xgboost/api''s `predict', whose sparse entry point is XGBoost's inplace
prediction and covers only two of them. A `csr-matrix' whose NUM-COLUMNS is not BOOSTER's own
feature count is LightGBM's own mistake to catch, and it does, with a clean nonzero return this
reports as `foreign-call-error' ("The number of features in data (N) is not the same as it was
in training data (M)."); nothing here pre-empts that check.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, an XGBoost booster, or not a handle at all. This function dispatches on nothing, so
`%check-lightgbm-booster' is the only thing between such a handle's pointer and
`LGBM_BoosterPredictForMat' or `LGBM_BoosterPredictForCSR'. In the measurement that found this
defect, an `xgboost-booster' with the check absent produced an SBCL corruption warning and
`Signal 7' -- a bus error, which kills the process rather than signalling anything a caller
could handle; in another it returned without faulting at all. See `%check-object-class' above
on why the spread, and not either outcome, is the argument for checking in Lisp.

Signals `released-handle-error' for a freed BOOSTER, and `backend-not-open' when its backend
has since been closed -- both from the `handle-live-pointer' inside `%check-lightgbm-booster',
which is read before anything is allocated, and before NUM-ITERATION is examined, so a freed
booster handed :BEST is reported as freed rather than as having no best iteration.

The output buffer's element count comes from `LGBM_BoosterCalcNumPredict', not from the row
count alone: the row count is only correct for a single-class objective. That count is read the
same way for either matrix kind -- it depends on BOOSTER, the row count, KIND and
NUM-ITERATION, and on nothing about how the rows are laid out. The second array dimension is
that count divided by the row count, guarded by `%predict-ncol'. Whichever entry point ran also
writes its own element count back through OUT-LEN; this is asserted equal to
`LGBM_BoosterCalcNumPredict''s count rather than trusted silently, since the buffer was sized
from the latter and a mismatch would mean either an under-filled result or a write past the
allocated buffer going unnoticed.

No prediction call here ever states the result's SHAPE, and that count is the whole of what one
reports bearing on it: `LGBM_BoosterCalcNumPredict' returns a number and nothing else, and
neither entry point reports axes the way XGBoost's `out_shape'/`out_dim' pair does -- so the
SECOND value is DERIVED rather than reported. `%prediction-shape' above is where that happens,
from the element count, the row count and BOOSTER's own feature count -- the last read by a
further library call, `LGBM_BoosterGetNumFeature', which runs inside this function's own
`with-foreign-float-traps-masked' body wrap like every other call it makes. It states a shape
only for the KINDs those three determine one for: `:normal' and `:raw' get the result array's
own dimensions, `:contrib' the three axes `contrib-shape' divides the count into (NIL for any
of the four cases that function's own docstring enumerates), and `:leaf-index' NIL. This
backend declares `:prediction-shape' in `*provided-capabilities*' to say the mechanism is here;
nothing re-checks that declaration, there being no argument to refuse. The first value is
untouched by all of it -- same dimensions, same elements, every KIND, either entry point.

Deliberately does not scan the result for NaN or infinity -- see
`cl-gbdt/src/xgboost/api''s `predict' for the identical reasoning, which applies here
unchanged: `with-foreign-float-traps-masked' restores the C calling convention around this
call, it does not and should not decide what counts as a valid model output.
```

<a id="cl-gbdt-xgboost-predict"></a>

## `cl-gbdt/xgboost:predict`

- **Kind** function
- **Signature** `(predict booster matrix &key (kind :normal) num-iteration missing)`
- **Exported from** `cl-gbdt/xgboost`

```text
Predict on MATRIX with BOOSTER, returning two values: the result array and the SHAPE this
backend states for it.

MATRIX is a dense matrix -- predicted through `XGBoosterPredictFromDMatrix' -- or a
`csr-matrix', through `XGBoosterPredictFromCSR'. KIND is `:normal', `:raw', `:leaf-index' or
`:contrib', mapped onto XGBoost's own prediction-type number by `%predict-type', which signals
for anything else. Predictions start from iteration 0; nothing here exposes a start-iteration
override.

NUM-ITERATION is a positive integer, or NIL for every iteration -- which XGBoost spells as an
`"iteration_end"' of 0, and `%resolve-num-iteration' is what writes it that way. :BEST is
REFUSED, with `unsupported-argument' naming this backend and "predict's :num-iteration":
only `train' writes a booster's `best-iteration', and a booster built by `create-booster' has
none, so at this layer the keyword would name an empty slot. `%reject-best-num-iteration' is
what refuses it, and its own docstring says why the refusal has to be explicit --
`%resolve-num-iteration' is `(or num-iteration 0)', so the keyword passes straight through it
as uninterpreted data. MEASURED at this layer with the refusal removed, on both entry points:
:BEST reaches `%predict-config-json''s `~D' directive and renders into the config JSON as the
bare token `BEST' -- `{...,"iteration_end":BEST,"strict_shape":true}' -- so the call comes
back as a `foreign-call-error' quoting XGBoost's own JSON parser ("Unknown construct, around
character position: 63"). That is a `cl-gbdt' condition, unlike the raw CFFI `type-error'
`%reject-best-num-iteration''s docstring records for LightGBM's integer-typed call, but it
names a parse failure at a character offset rather than the argument the caller got wrong, and
its wording is XGBoost's to change. The refusal is invisible to Layer 2:
`cl-gbdt/src/xgboost/protocol''s `predict' method resolves :BEST first -- by
`%resolve-best-num-iteration', which signals `unsupported-argument' when the booster has no
best iteration to resolve it against -- and calls this with the integer that resolution
produced, so the keyword itself never arrives from there.

MISSING is the value in MATRIX that means *missing* -- a real, or NIL for this library's own
default, the IEEE NaN -- and it reaches the library through a DIFFERENT config for each of
MATRIX's two forms, neither of them the one `create-dataset' fills. A dense MATRIX becomes a
transient DMatrix, so its sentinel is a key in THAT DMatrix's creation config --
`%create-dmatrix', exactly as for a dataset that outlives the call. A `csr-matrix' builds no
DMatrix at all, so its sentinel is a key in the INPLACE PREDICT config instead --
`%predict-from-csr', which needs the key anyway. Same argument, same meaning, two config
strings built by two functions: see `%predict-config-json', whose own docstring says why the
dense path leaves the key out of the predict config rather than sending the sentinel twice.
`missing-value-json' signals `unsupported-argument' for a value that is neither a real nor
NIL, and the comparison the library then makes is at SINGLE precision, whatever MATRIX's own
element type, as it is on the ingestion path. The `:missing-value' CAPABILITY is not checked
here: it gates the portable :MISSING argument of `cl-gbdt/src/xgboost/protocol''s `predict',
which checks it there, and MISSING reaches both branches alike so there would be no branch for
such a check to belong to -- the same division `%dataset-pointer' above records for the
ingestion path.

Signals `capability-unavailable' naming `:sparse-input' when MATRIX is a `csr-matrix' and that
capability reads false -- see `%check-sparse-input' above, which checks it before any foreign
call.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, a LightGBM booster, or not a handle at all. This function dispatches on nothing, so
`%check-xgboost-booster' is the only thing between such a handle's pointer and
`XGBoosterPredictFromDMatrix' or `XGBoosterPredictFromCSR'.

Signals `released-handle-error' for a freed BOOSTER, and `backend-not-open' when its backend
has since been closed -- both from the `handle-live-pointer' inside `%check-xgboost-booster',
which is read before anything is allocated, and before NUM-ITERATION is examined, so a freed
booster handed :BEST is reported as freed rather than as having passed a keyword this layer
refuses.

A dense MATRIX is built into a transient DMatrix via `%create-dmatrix' first --
`XGBoosterPredictFromDMatrix' takes a DMatrix handle, unlike LightGBM's
`LGBM_BoosterPredictForMat', which predicts straight off a raw pointer and row/column
counts. It is built first, before anything else here, so a MISSING that
`%dense-matrix-config-json' refuses signals with nothing pinned and no foreign allocation
held -- the property `%create-dmatrix''s own docstring claims. The transient DMatrix is
freed before this returns, on every exit path, since nothing else retains it. Its free is
checked with `check-xgb', not discarded outright: a failure there is reported with `warn'
rather than an error, matching `free-dataset''s own reasoning for warning instead of
signalling, since raising an error from cleanup would replace whatever condition is already
propagating on an unwinding exit -- but on the ordinary success path, a failed free still
leaks foreign memory and is worth reporting rather than passing over in silence.

A `csr-matrix' builds no DMatrix at all: `XGBoosterPredictFromCSR' is XGBoost's INPLACE
prediction and reads the three vectors where they lie, so there is nothing transient to
free and no `unwind-protect' around it. That saves a copy, and it is the entry point the
`:sparse-input' capability declares -- but it is a different code path from
`XGBoosterPredictFromDMatrix', not a CSR spelling of it, and **it covers only `:normal' and
`:raw'**. `:contrib' and `:leaf-index' on a `csr-matrix' signal `foreign-call-error'
("Unsupported prediction type:2" and ":6" respectively), measured against the vendored
library. Both work on a dense matrix, and materialising the rows as one -- a 2D
`double-float' or `single-float' array, or a `foreign-matrix' -- is the only way to reach
either KIND for rows a caller holds sparsely: those are the only other forms
`call-with-foreign-matrix' has a method for, and a dataset is not among them, so routing the
rows through `create-dataset' leads nowhere `predict' can be called on.
That refusal is the library's own and is left to it, exactly as a `csr-matrix' whose
NUM-COLUMNS is not BOOSTER's feature count is ("Number of columns in data must equal to the
trained model"). NUM-ITERATION is honoured identically on both paths -- the same
`iteration_begin'/`iteration_end' pair reaches the same config JSON, which additionally
carries the `"missing"' key inplace prediction requires; see `%predict-config-json'.

The output buffer's total element count comes from the C call's own `out_shape'/`out_dim'
report, not from the row count alone -- the row count is only
correct for a single-class objective. The second array dimension is that total divided
by the row count, guarded by `%predict-ncol' -- the same derivation
`cl-gbdt/src/lightgbm/native''s `%predict-ncol' makes for its own row-count-alone pitfall,
and the one that
tells a three-class `multi:softprob' model's predictions apart from a binary model's. Both
entry points report it the same way, `"strict_shape":true' being set for both.

That same report is also RETURNED, as this function's second value: `%reported-shape' reads
`out_shape' back as a list of integers instead of only multiplying it out, and neither entry
point interprets or reshapes it. Reading the shape BACK is what parts this backend from
`cl-gbdt/src/lightgbm/api''s `%prediction-shape', which has no such call to read and derives
what it can from an element count instead, stating NIL for `:leaf-index'. It is never NIL
here: `out_dim' was measured 2 for `:normal'
and `:raw', 3 for `:contrib' and 4 for `:leaf-index' on both entry points, so
`%reported-shape''s empty-loop case -- a zero DIM -- does not arise. This backend declares
`:prediction-shape' in `*provided-capabilities*' to say so, and nothing re-checks that
declaration: there is no argument to refuse, and what the declaration says is that the
mechanism is present, not that the shape is non-NIL. Measured against the vendored library,
the shape is RICHER than the first value's own dimensions for two kinds: a four-round
three-class model over four features reports (rows 4 3 1) for `:leaf-index' where the array is
rows x 12, and
(rows 3 5) for `:contrib' where the array is rows x 15. A four-round BINARY model over three
features reports (rows 4 1 1) and (rows 1 4) -- multidimensional there too, its one output
group notwithstanding -- so this is not a multiclass-only difference. The first value is
untouched by any of it.

`out_result' is XGBoost's own memory, valid only until the next call into this booster,
so every element is copied out, coerced from `single-float' to `double-float', before
this returns.

Deliberately does not scan the result for NaN or infinity. `with-foreign-float-traps-masked'
around this body stops SBCL from turning an intermediate invalid operation inside
XGBoost's own computation -- e.g. `multi:softprob''s softmax normalization -- into a signal;
it does not, and cannot, stop XGBoost from legitimately returning a non-finite value as a
final result (`:raw' scores in particular are not bounded the way a transformed prediction
is). Rejecting or flagging one here would be a policy this wrapper does not otherwise
impose on any other operation's output, invented for this fix rather than driven by a
reported failure -- a caller that cannot tolerate a non-finite prediction should check for
one itself.
```

<a id="cl-gbdt-probe-capabilities"></a>

## `cl-gbdt:probe-capabilities`

- **Kind** function
- **Signature** `(probe-capabilities optional-symbols &key provided (library :default))`
- **Exported from** `cl-gbdt`

```text
Return the capability plist for OPTIONAL-SYMBOLS, probed against LIBRARY, with every
capability in PROVIDED recorded true ahead of them.

OPTIONAL-SYMBOLS is an alist of a capability keyword and the C function names that capability
needs: ((:model-slicing "XGBoosterSlice") ...). A capability is true when every one of its
names resolves.

PROVIDED is a list of capability keywords the backend provides unconditionally, recorded as
true without being probed. A probe cannot express "always true": it derives every answer
from a symbol lookup, and a capability whose C functions are all in `*required-symbols*' has
nothing left to look up -- `open-backend' has already refused to open a library missing any
of them. `:evaluation-history' is the case this exists for; leaving it out of the plist
entirely would make `backend-supports-p' answer NIL, which that function documents as the
feature being unavailable and the operation signalling `capability-unavailable', and neither
is true of a capability both backends ship.

PROVIDED's entries come first, so a capability named in both PROVIDED and OPTIONAL-SYMBOLS
reads true through `getf' whatever the probe found. That combination is a contradiction in
the backend's own declarations rather than a case with a useful meaning: a capability is
either unconditional or probed, and `tools/ci/check-abi-blacklist.lisp' checks both lists
against `*known-capabilities*' but cannot tell which list a name belongs in.

Every declared capability appears in the result, true and false alike, rather than only the
true ones -- `backend-info' should be able to report what was asked as well as what was
answered.

**This never signals for a missing symbol**, which is the whole difference between an optional
symbol and a required one: policy section 8 says an optional symbol's absence disables that one
capability rather than preventing the backend from opening. Callers wanting the required
behaviour use `probe-foreign-symbols' directly and signal `missing-foreign-symbols'
themselves.

LIBRARY is passed through to `probe-foreign-symbols'; see its docstring for the SBCL caveat
about scoping a probe to one library.
```

<a id="cl-gbdt-probe-foreign-symbols"></a>

## `cl-gbdt:probe-foreign-symbols`

- **Kind** function
- **Signature** `(probe-foreign-symbols names &key (library :default))`
- **Exported from** `cl-gbdt`

```text
Return the C function names in NAMES that are absent from LIBRARY.

Returns nil when all are present. This is how version mismatches are detected. It
cannot catch a function whose name stayed the same while its signature changed;
those are avoided by design instead (see ffi-spec/ABI-BLACKLIST.md).

LIBRARY is passed straight through to `cffi:foreign-symbol-pointer' and defaults
to its own default, `:default', which searches every foreign library the image
currently has loaded. Pass the `cffi:foreign-library' object
`cffi:load-foreign-library' returns for the library just opened to scope the
probe to it -- on CFFI backends that honor the argument. SBCL, the only backend
this project runs on, is not one of them: `cffi-sbcl.lisp''s
`%foreign-symbol-pointer' takes LIBRARY only to validate it against
`cffi::get-foreign-library' (an unregistered designator still signals, which is
real and worth having) and then ignores the handle, resolving through
`sb-sys:find-foreign-symbol-address' -- SBCL's global linkage table -- exactly
as `:default' would. Verified directly: with the vendored LightGBM and XGBoost
libraries both loaded, probing "XGBoosterCreate" with LIBRARY bound to
LightGBM's own `foreign-library' object still reports it found. So on this
platform LIBRARY cannot by itself prove a probe came from a specific library --
the caller must still trust that PATH (or the search order above it) named the
right file; only a LightGBM already loaded by something else, providing every
name in *required-symbols* under the same names, defeats that, and no argument
to this function can detect it here.
```

<a id="cl-gbdt-register-backend"></a>

## `cl-gbdt:register-backend`

- **Kind** function
- **Signature** `(register-backend name class-name)`
- **Exported from** `cl-gbdt`

```text
Associate the backend name NAME with the class CLASS-NAME.

Each backend system calls this when it loads.
```

<a id="cl-gbdt-release-handle"></a>

## `cl-gbdt:release-handle`

- **Kind** function
- **Signature** `(release-handle handle free-function)`
- **Exported from** `cl-gbdt`

```text
Call FREE-FUNCTION on HANDLE's pointer exactly once, then mark HANDLE released.

FREE-FUNCTION is a function of one argument, HANDLE's pointer. A second call on an
already-released HANDLE does nothing -- this is what lets `free-dataset' and
`free-booster' promise that freeing an already-freed handle is a no-op. Also cancels
HANDLE's finalizer, since that finalizer exists only to report a free that never
happened, and this one just did.
```

<a id="cl-gbdt-released-handle-error"></a>

## `cl-gbdt:released-handle-error`

- **Kind** condition
- **Superclasses** `gbdt-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
A handle was used after it had been freed.
```

### Slots

#### `object`

- **Readers** `released-handle-error-object`

```text
The handle instance itself -- a `dataset' or `booster' -- that
was already released when this call tried to use it.
```

<a id="cl-gbdt-released-handle-error-object"></a>

## `cl-gbdt:released-handle-error-object`

- **Kind** generic function
- **Signature** `(released-handle-error-object condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:released-handle-error`'s `object` slot. See `cl-gbdt:released-handle-error`.

<a id="cl-gbdt-resolve-and-load-library"></a>

## `cl-gbdt:resolve-and-load-library`

- **Kind** function
- **Signature** `(resolve-and-load-library backend &key path env-var directory pattern default-name)`
- **Exported from** `cl-gbdt`

```text
Locate and load BACKEND's shared library, returning the `cffi:foreign-library' loaded
and the path it came from as a second value.

Search order: PATH, then ENV-VAR, then DIRECTORY searched for PATTERN, then CFFI's own
system search for DEFAULT-NAME. ENV-VAR is honored strictly -- when it is set but names
a path that does not exist this signals `backend-library-not-found' rather than falling
through, because a typo in an override that silently loads a different library is worse
than a failure.

Signals `backend-library-not-found' (with `:searched') when no candidate is found
anywhere -- PATH, ENV-VAR and DIRECTORY all failing to name one, and CFFI's own system
search for DEFAULT-NAME failing too -- and `backend-library-load-failed' (with `:path'
and `:cause') when a candidate was found but `cffi:load-foreign-library' rejects it.
```

<a id="cl-gbdt-save-model"></a>

## `cl-gbdt:save-model`

- **Kind** generic function
- **Signature** `(save-model booster path &key num-iteration)`
- **Exported from** `cl-gbdt`

```text
Save BOOSTER's model to PATH.

NUM-ITERATION limits how many boosted rounds are saved on LightGBM: nil means all of
them, an integer that many, and `:best' resolves to BOOSTER's own `booster-best-iteration'
first -- see `predict' for exactly when that resolution itself signals
`unsupported-argument'. XGBoost has no such limit at all -- `XGBoosterSaveModel' always
saves every round -- so supplying NUM-ITERATION there signals `unsupported-argument'
whether it is an explicit integer or `:best' resolved to one; nothing about `:best' is
special-cased around that check. A caller who wants an XGBoost model file that stops at
the best iteration slices to it first with `cl-gbdt/xgboost:slice-model' and saves the
slice instead.
```

### Methods

#### `(save-model (booster booster) (path t) &key num-iteration)`

```text
Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `save-model' rather than writing anything to PATH. A
Layer 1 caller saves the same model with that backend's own `save-model' in `api.lisp'
instead.
```

#### `(save-model (booster lightgbm-booster) (path t) &key num-iteration)`

```text
Save BOOSTER's model to PATH via `LGBM_BoosterSaveModel'.

NUM-ITERATION limits how many trees are saved, :BEST resolved by
`%resolve-best-num-iteration' first; nil saves all of them, which LightGBM spells as 0.
Returns PATH.

The procedure is Layer 1 and lives in `cl-gbdt/src/lightgbm/api''s `save-model'. What is left
here is resolving :BEST, which reads `booster-best-iteration' -- a slot `train' writes and
nothing else does, so the keyword has no meaning below this layer.
```

#### `(save-model (booster xgboost-booster) (path t) &key num-iteration)`

```text
Save BOOSTER's model to PATH via `XGBoosterSaveModel'.

Signals `unsupported-argument' when NUM-ITERATION is supplied: unlike LightGBM's
`LGBM_BoosterSaveModel', `XGBoosterSaveModel' takes no iteration limit -- it always
saves every boosted round -- and silently ignoring the argument would be exactly the
failure mode `unsupported-argument' exists to prevent, per `%check-unsupported'. :BEST is
resolved by `%resolve-best-num-iteration' first, into an integer, which then meets this
same check exactly as an explicit integer would -- not special-cased around it. A caller
who wants a file that stops at the best iteration slices to it first with
`cl-gbdt/xgboost:slice-model' and saves the slice instead.

Returns PATH.

The procedure is Layer 1 and lives in `cl-gbdt/src/xgboost/api''s `save-model', which takes no
:NUM-ITERATION at all. What is left here is the refusal above, which exists because the
unified API promised a portable argument LightGBM honours and this library has no route for.
```

<a id="cl-gbdt-lightgbm-save-model"></a>

## `cl-gbdt/lightgbm:save-model`

- **Kind** function
- **Signature** `(save-model booster path &key num-iteration)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Save BOOSTER's model to PATH via `LGBM_BoosterSaveModel', and return PATH.

NUM-ITERATION is a positive integer limiting how many trees are written, or NIL for all of
them -- which LightGBM spells as 0, and `%resolve-num-iteration' is what writes it that way.
:BEST is REFUSED, with `unsupported-argument' naming this backend and "save-model's
:num-iteration": only `train' writes a booster's `best-iteration', so at this layer the
keyword would name an empty slot. The refusal is invisible to Layer 2, whose method resolves
:BEST first and calls this with the integer that resolution produced.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, an XGBoost booster, or not a handle at all. This function dispatches on nothing, so
`%check-lightgbm-booster' is the only thing between such a handle's pointer and
`LGBM_BoosterSaveModel'; see `%check-object-class' above on what wrong-kind pointers did when
measured against the vendored library. It also signals `released-handle-error' for a freed
BOOSTER and `backend-not-open' for a closed backend, both from the `handle-live-pointer'
inside that check, which runs before NUM-ITERATION is examined.
```

<a id="cl-gbdt-xgboost-save-model"></a>

## `cl-gbdt/xgboost:save-model`

- **Kind** function
- **Signature** `(save-model booster path)`
- **Exported from** `cl-gbdt/xgboost`

```text
Save BOOSTER's model to PATH via `XGBoosterSaveModel', and return PATH.

Takes no iteration limit, unlike `cl-gbdt/src/lightgbm/api''s `save-model':
`XGBoosterSaveModel' has no such parameter and always writes every boosted round. The
argument is therefore ABSENT here rather than refused -- a Layer 1 caller who names it gets
Common Lisp's own unknown-keyword error, which is the right report for a keyword that does
not exist. `cl-gbdt/src/xgboost/protocol''s method is where `unsupported-argument' is
signalled, because that refusal exists only because the unified API promised a portable
:NUM-ITERATION that LightGBM honours.

XGBoost selects its serialization format from PATH's extension -- `.json' and `.ubj' are the
current ones -- and reports an unrecognized extension itself, as `foreign-call-error'.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, a LightGBM booster, or not a handle at all. This function dispatches on nothing, so
`%check-xgboost-booster' is the only thing between such a handle's pointer and
`XGBoosterSaveModel'; see `%check-object-class' above on what wrong-kind pointers did when
measured against the vendored library. `released-handle-error' and `backend-not-open' come
from the `handle-live-pointer' inside that same check.
```

<a id="cl-gbdt-shutdown-backend"></a>

## `cl-gbdt:shutdown-backend`

- **Kind** generic function
- **Signature** `(shutdown-backend backend)`
- **Exported from** `cl-gbdt`

```text
Close BACKEND's shared library and release its resources.

Implemented by each backend.
```

### Methods

#### `(shutdown-backend (backend lightgbm-backend))`

```text
Close LightGBM's shared library.

`cffi:close-foreign-library' drops cl-gbdt's own reference and, on platforms
where the C loader honors `dlclose' reference counting, may unmap the library;
POSIX does not guarantee an actual unload, so this cannot promise the library's
code and data are gone from the process afterward -- only that cl-gbdt no
longer holds it open.
```

#### `(shutdown-backend (backend xgboost-backend))`

```text
Close XGBoost's shared library.

`cffi:close-foreign-library' drops cl-gbdt's own reference and, on platforms where the C
loader honors `dlclose' reference counting, may unmap the library; POSIX does not
guarantee an actual unload, so this cannot promise the library's code and data are gone
from the process afterward -- only that cl-gbdt no longer holds it open.
```

<a id="cl-gbdt-xgboost-slice-model"></a>

## `cl-gbdt/xgboost:slice-model`

- **Kind** function
- **Signature** `(slice-model booster &key (begin 0) end (step 1))`
- **Exported from** `cl-gbdt/xgboost`

```text
Return a new booster holding BOOSTER's layers from BEGIN to END, taken STEP at a time.

The interval is HALF-OPEN, `[BEGIN, END)': END names the first layer left out, so slicing a
ten-round booster with `:begin 0 :end 5' gives five rounds, not six. Measured against the
vendored libxgboost, whose header documents no interval semantics at all; XGBoost's own
rejection of `:begin 5 :end 5' as "Empty slice is not allowed" is the same reading from
the other side.

END defaults to NIL, meaning through the last layer, and is passed to `XGBoosterSlice' as
its own 0. NIL rather than 0 in Lisp because a caller writing `:END 0' means "nothing",
and silently reading that as "everything" is the kind of translation policy section 5
exists to prevent -- so an explicit `:END 0' signals `unsupported-argument' rather than
being forwarded to a C 0 that would mean the opposite. Every other out-of-range request is
XGBoost's own to refuse, and it does, with `foreign-call-error': END past the last layer,
BEGIN below zero, STEP below one, and a STEP that does not divide the interval evenly.

The returned booster belongs to the caller, who frees it with `free-booster'. It is
INDEPENDENT of BOOSTER: `XGBoosterSlice' copies the layers it selects, so freeing BOOSTER
first is legitimate, and the slice keeps predicting the same values afterward -- verified
against the vendored library with BOOSTER and the DMatrix it was trained on both freed. It
therefore retains no parent, exactly as a `load-model' booster retains no training set;
retaining one anyway would make freeing BOOSTER signal `released-handle-error' on correct
code. For the same reason the slice has no training set of its own, so `evaluation' and
`update-one-iteration' on it behave as they do for a `load-model' booster.

Signals `wrong-backend-reference' when BOOSTER was not built by the XGBoost backend,
`released-handle-error' when it has been freed, `backend-not-open' when its backend has
been closed, `capability-unavailable' when the loaded library has no `XGBoosterSlice',
`unsupported-argument' for an explicit `:END 0', and `foreign-call-error' when the slice
itself fails.

The capability is re-checked here rather than assumed: policy section 7 requires the
operation to signal for itself, so a caller who never asked `backend-supports-p' gets a
typed condition instead of a missing-symbol crash. The handle check runs first, before the
capability check, so handing this a LightGBM booster reports the wrong handle rather than
the true-but-irrelevant news that the backend it came from cannot slice.
```

<a id="cl-gbdt-train"></a>

## `cl-gbdt:train`

- **Kind** generic function
- **Signature** `(train backend dataset &key valid-sets num-rounds parameters record-history early-stopping objective evaluation)`
- **Exported from** `cl-gbdt`

```text
Train a BACKEND model on DATASET and return two values: a booster and
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
              `evaluation' -- LightGBM's "binary_logloss", XGBoost's "logloss".
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
"none". LightGBM refuses a custom update while the booster holds an objective function
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

A NIL VALUE means "not computable this iteration" -- a fold whose denominator was zero,
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
than `training-report-num-rounds' says the run was. Pinning the name is what keeps "every
series is exactly that long", below, true of a caller's own series as well as the library's.
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
EARLY-STOPPING was not supplied at all -- NIL meaning "not determined", never "iteration
0". When it was supplied, BEST-ITERATION is the 1-based iteration at which the watched
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
`evaluation' invents no name for the index it reports.
```

### Methods

#### `(train (backend backend) (dataset t) &key valid-sets num-rounds parameters record-history early-stopping objective evaluation)`

```text
Fallback for a BACKEND whose unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `train'. Layer 1 has no equivalent of this generic as a
whole -- its own training report, early stopping and `:objective'/`:evaluation' callbacks
are `train''s own concepts -- so recovering from this means loading the unified system, not
composing a Layer 1 substitute.
```

#### `(train (backend lightgbm-backend) (dataset t) &key valid-sets (num-rounds 100) parameters (record-history t) early-stopping objective evaluation)`

```text
Train a LightGBM booster on DATASET for up to NUM-ROUNDS boosting iterations, and
return it and a `training-report' of the run.

Builds the booster with `LGBM_BoosterCreate' from PARAMETERS, attaches each of
VALID-SETS with `LGBM_BoosterAddValidData', then drives
`LGBM_BoosterUpdateOneIter' NUM-ROUNDS times -- or fewer, when EARLY-STOPPING
ends the run first. See the `train' generic function's docstring for what each
argument means, and for what the secondary value holds; NUM-ROUNDS defaults to
100 when not supplied.

Each VALID-SETS element is either a dataset, whose series carry no name, or a
(NAME . DATASET) cons, where NAME is a string that reaches `training-series-name' for
every series recorded at that dataset's index -- see `%valid-set-name' and
`%valid-set-dataset', which split VALID-SETS into two parallel lists, of datasets and of
names, once at the top of this method; everything below reads the datasets list under
the name VALID-SETS, exactly as before this method accepted names at all. Two entries
may legitimately share one NAME: their index, not their name, is what a caller uses to
tell them apart in the report, so this is accepted rather than rejected as a duplicate.
The training set is never a VALID-SETS entry and is always index 0 with a NIL name.

When RECORD-HISTORY is true -- the default -- this reads the whole evaluation after each
iteration through `%read-evaluation': the same function the `evaluation' method calls, on
the same booster pointer and the same dataset count, which is what keeps the history and
what `evaluation' answers afterward from being able to disagree. `training-report-from-history'
folds the run's worth of them into the report once the loop is done; that fold is backend-
neutral and shared with `cl-gbdt/src/xgboost/protocol''s `train', so what the two backends
record cannot drift apart either. It orders series by the (DATASET-INDEX, METRIC-NAME)
pair's first appearance, which for this backend is `%read-evaluation''s own dataset-major
order, so the report's series arrive in exactly the order `evaluation' reports its entries
in without anything being sorted.

RECORD-HISTORY NIL skips that read entirely -- one `LGBM_BoosterGetEval' call per dataset
per iteration, which is what makes recording cost real wall-clock time (see the `train'
generic's docstring for the measured figures). The loop is then exactly the
`LGBM_BoosterUpdateOneIter' loop this method ran before it recorded anything, and the
report it still returns as its secondary value has an empty series list over the same
NUM-ROUNDS -- `training-report-from-history' over an empty history, the same shape a run
with `metric=none' produces.

A read that fails propagates, freeing the booster through the `unwind-protect' below rather
than returning a report whose series are shorter than the run: a short series is
indistinguishable from one a buggy loop recorded, and "one value per iteration" is the
invariant a caller reading the report relies on.

EARLY-STOPPING watches one of those recorded series and ends the loop once it has stopped
improving -- see the `train' generic function's docstring for the spec's four required
keys, and `train-early-stopping-watcher' for why it cannot be combined with
RECORD-HISTORY NIL.
The watcher sees each iteration's entries exactly as the history records them, off the one
`%read-evaluation' call this loop already makes, so what stopped the run and what the
report shows can never be two different readings. `training-report-num-rounds' needs
nothing extra to report the shortened run: it has counted actual iterations since Phase 3a.

OBJECTIVE replaces `LGBM_BoosterUpdateOneIter' with `LGBM_BoosterUpdateOneIterCustom' for
every iteration of the loop, driven by the gradient and Hessian the caller's own function
returns -- see the `train' generic function's docstring for what that function is called with
and what it must return. Signals `capability-unavailable' naming `:custom-objective' for a
non-NIL OBJECTIVE when the capability reads false, before any foreign call: see
`%check-custom-objective' above, which reads the capability rather than this backend's name,
and `*optional-symbols*' for why the answer here is probed rather than declared. OBJECTIVE
NIL, the default, reaches no check and runs exactly the `LGBM_BoosterUpdateOneIter' loop this
method has always run.

A non-NIL OBJECTIVE also OVERRIDES any `objective' entry in PARAMETERS, forcing it to
"none" through `objective-parameters' before `LGBM_BoosterCreate' ever sees the string.
That is not a convenience: `LGBM_BoosterUpdateOneIterCustom' refuses to run while the booster
holds an objective function at all -- `Check failed: objective_function_ == nullptr', a
non-zero return this method would surface as `foreign-call-error' -- so the combination the
override replaces has no working form to preserve. Every other parameter passes through
untouched and in its original order, `num_class' included, which is still what tells LightGBM
how many output groups a multiclass custom objective has.

Each iteration reads the booster's current raw scores with `%booster-predictions' --
`LGBM_BoosterGetPredict' at `data_idx' 0, the scores LightGBM already holds for the training
data, rather than a fresh `predict' over the training matrix, which this method does not have
and which would cost a full prediction pass per iteration. It hands them to OBJECTIVE as a
(ROWS GROUPS) `double-float' array, where ROWS comes from `%dataset-num-rows' on the training
set's own pointer and GROUPS from `LGBM_BoosterGetNumClasses'. What comes back is checked by
`check-objective-result' -- `cl-gbdt/src/config/objective''s, which is backend-neutral pure
code and names no library, so a second backend refuses the same shapes with the same
`dimension-mismatch' by calling it rather than by restating it -- and only then flattened into
the C buffers, GROUP-MAJOR on this backend (row I of group K at `(+ (* K ROWS) I)') and
converted to `single-float', which is what `LGBM_BoosterUpdateOneIterCustom''s `const float*'
parameters admit. Both the flattening and the score layout are measured; see
`%update-one-iteration-custom' and `%booster-predictions' for the measurements. The flattening is
this method's business and not the caller's: OBJECTIVE is handed, and returns, a (ROWS GROUPS)
array whichever order the library underneath wants it in.

OBJECTIVE is funcalled inside this method's own `with-foreign-float-traps-masked' body wrap,
so the caller's Lisp arithmetic runs under the masked convention on x86-64 as well as on
aarch64 -- `(/ 1.0d0 0.0d0)' yields infinity there rather than signalling
`division-by-zero'. Nothing about that is specific to a custom objective; it is simply where
in `train' the caller's code now runs. A condition the caller's function does signal
propagates out of `train' through the `unwind-protect' below, freeing the booster handle
rather than orphaning it, exactly as a mid-loop foreign failure does.

An objective that frees a handle this loop depends on, or closes BACKEND, is caught rather
than crashed on: `%recheck-train-datasets' re-runs this method's own opening checks the
moment the `funcall' returns, and TRAIN-DATA-POINTER is reassigned from what it returns, so
nothing after the caller's code uses a pointer read before it. See that function for what
each of the three re-checks is for. This is the only place the loop needs it -- the
OBJECTIVE NIL branch beside it runs no caller code at all.

Such a run also leaves this method's own cleanup to run against a closed BACKEND:
`free-booster' then emits an unfreed-handle warning instead of signalling, and the
foreign booster is genuinely leaked, since the library may already be unmapped by the
time that cleanup runs -- see the comment on the `unwind-protect' below for why letting
that warning escape is correct.

Neither RECORD-HISTORY nor EARLY-STOPPING is disabled by OBJECTIVE, and neither is made
meaningful by it: a metric configured through PARAMETERS relates to the library's own
objective, not to the caller's, and this method neither signals nor warns about that -- see
the `train' generic function's docstring, which states it as the caller's decision.

EVALUATION adds the caller's own metric to what each iteration records, one call per dataset
per iteration -- see the `train' generic function's docstring for what that function is
called with and what it must return. Signals `capability-unavailable' naming
`:custom-evaluation' when the capability reads false, and `unsupported-argument' naming
"train's :evaluation" for RECORD-HISTORY NIL or for a non-function, all three before any
foreign call: see `%check-custom-evaluation' above, and `*optional-symbols*' for why the
answer here is probed rather than declared. EVALUATION NIL, the default, reaches no check and
records exactly what this method has always recorded.

The calls happen after this iteration's own `%read-evaluation' and BEFORE the history push
and the watcher, in `%custom-evaluation-entries' -- so the entries the history keeps and the
entries the watcher sees are one list, and `:early-stopping' can watch a custom metric with
nothing here to arrange it. The custom entries are APPENDED after every library entry, which
is what makes `training-report-from-history''s first-seen ordering put the library's series
first, as a prefix, exactly where `evaluation' reports them; a custom metric never reaches
`evaluation' at all, that method reading only `LGBM_BoosterGetEval'.

Each dataset's predictions are read with `%booster-predictions' at that dataset's own
`data_idx' -- `LGBM_BoosterGetNumPredict' then `LGBM_BoosterGetPredict', the values LightGBM
already holds, rather than a fresh `predict' over a matrix this method does not have. They
are `predict :kind :normal''s numbers and not the margin OBJECTIVE is handed, measured on
both datasets; see `%booster-predictions', which records that measurement and the per-dataset
length it rests on. ROW-COUNTS is read once before the loop, from the same pointers this
method already validated, and only when EVALUATION is non-NIL, so a run that asks for no
custom metric makes no extra foreign call at all.

An EVALUATION that frees a handle this loop depends on, or closes BACKEND, is caught the same
way an OBJECTIVE is: `%custom-evaluation-entries' calls `%recheck-train-datasets' the moment
each `funcall' returns -- between two consecutive datasets' reads, not once per iteration --
and TRAIN-DATA-POINTER is reassigned from what it returns.

DATASET and every VALID-SETS entry's dataset half are each run through
`%check-lightgbm-dataset' before any foreign call. `train' dispatches on
BACKEND, not on DATASET, so unlike `dataset-num-rows' or `free-dataset' there
is no CLOS specializer here to rule out the wrong kind of handle first --
without this, `handle-live-pointer' would happily hand `LGBM_BoosterCreate' a
booster's own pointer to use as its training-set `DatasetHandle'. Signals
`wrong-backend-reference' when DATASET or a VALID-SETS entry's dataset half is
not a `lightgbm-dataset', and `released-handle-error' or `backend-not-open'
when one is but has already been freed or had its own backend closed. A
VALID-SETS entry that is a cons with a non-string car never reaches this check
at all: `%valid-set-name' signals `unsupported-argument' for it first, which is
the different mistake a malformed name is, kept distinct from a wrong dataset
handle.

The returned booster retains DATASET as its training set and a fresh copy of
VALID-SETS as its validation sets, keeping all of them alive for the booster's
lifetime and letting `update-one-iteration' notice if any is freed out from
under it -- see `%check-booster-datasets-live'. The copy matters: VALID-SETS is
the caller's own list, and `make-handle' would otherwise store that exact list
object rather than a snapshot of it. A caller who destructively removes an
entry from VALID-SETS after `train' returns -- `delete', `(setf (cdr ...))',
reusing the list elsewhere with `nconc' -- would silently remove it from the
booster's view too, since both would be the same cons cells; the dataset
`LGBM_BoosterAddValidData' already attached would then go unchecked by
`%check-booster-datasets-live' even though LightGBM still holds its pointer.
Free the result with `free-booster' or wrap it in `with-booster'.

BOOSTER is bound to a full handle already: `create-booster' manages the raw-pointer window
between `LGBM_BoosterCreate' and its own `make-handle' call internally -- see its docstring
-- and this method never touches a pointer that let does not already own. What the
`unwind-protect' below manages instead is what happens to that handle from here: any exit
from the loop or the report construction that does not reach the final `setf' frees BOOSTER
rather than orphaning it.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'.
```

#### `(train (backend xgboost-backend) (dataset t) &key valid-sets (num-rounds 100) parameters (record-history t) early-stopping objective evaluation)`

```text
Train an XGBoost booster on DATASET for up to NUM-ROUNDS boosting iterations, and
return it and a `training-report' of the run.

Builds the booster with `XGBoosterCreate' over DATASET and every VALID-SETS entry's
DMatrix handle together -- see `%create-booster' for why XGBoost takes the whole set up
front rather than adding validation data afterward. Applies PARAMETERS one at a time via
`XGBoosterSetParam', then drives `XGBoosterUpdateOneIter' NUM-ROUNDS times -- or fewer,
when EARLY-STOPPING ends the run first. See the `train' generic function's docstring for
what each argument means, and for what the secondary value holds; NUM-ROUNDS defaults to
100 when not supplied.

Each VALID-SETS element is either a dataset, whose series carry no name, or a
(NAME . DATASET) cons, where NAME is a string that reaches `training-series-name' for
every series recorded at that dataset's index -- see `%valid-set-name' and
`%valid-set-dataset', which split VALID-SETS into two parallel lists, of datasets and of
names, once at the top of this method; everything below reads the datasets list under
the name VALID-SETS, exactly as before this method accepted names at all. Two entries
may legitimately share one NAME: their index, not their name, is what a caller uses to
tell them apart in the report, so this is accepted rather than rejected as a duplicate.
The training set is never a VALID-SETS entry and is always index 0 with a NIL name.

When RECORD-HISTORY is true -- the default -- this reads the whole evaluation after each
iteration through `%read-evaluation': the same function the `evaluation' method calls, over
the same DMatrix pointers in the same order, which is what keeps the history and what
`evaluation' answers afterward from being able to disagree.
`training-report-from-history' folds the run's worth of them into the report once the loop
is done; that fold is backend-neutral and shared with `cl-gbdt/src/lightgbm/protocol''s
`train', so what the two backends record cannot drift apart either. It orders series by the
(DATASET-INDEX, METRIC-NAME) pair's first appearance, which for this backend is the order
`XGBoosterEvalOneIter' formatted its own result in, so the report's series arrive in exactly
the order `evaluation' reports its entries in without anything being sorted. A field the
parse could not read as a `double-float' is recorded as NIL, keeping its place in the series
rather than shortening it -- see `training-series-values'.

RECORD-HISTORY NIL skips that read entirely -- one `XGBoosterEvalOneIter' call per
iteration, plus the parse of the line it formats, which is what makes recording cost real
wall-clock time (see the `train' generic's docstring for the measured figures). The loop is
then exactly the `XGBoosterUpdateOneIter' loop this method ran before it recorded anything,
and the report it still returns as its secondary value has an empty series list over the
same NUM-ROUNDS -- `training-report-from-history' over an empty history, the same shape a
run with `disable_default_eval_metric=1' produces.

Skipping the read also widens what this method accepts, which matters here more than it
does on LightGBM: `XGBoosterEvalOneIter' evaluates every DMatrix it is handed, and refuses
one it cannot evaluate -- an unlabelled DMatrix passed in VALID-SETS is the case this was
found through, which `XGBoosterUpdateOneIter' trains on without complaint while the
evaluation call signals `foreign-call-error' ("label and prediction size not match"). With
RECORD-HISTORY true that failure now propagates out of `train' itself, through the
`unwind-protect' below, where before this backend recorded anything it surfaced
only at a later `evaluation' call. RECORD-HISTORY NIL never reaches the evaluation path and
so trains such a configuration exactly as before.

A read that fails propagates, freeing the booster through the `unwind-protect' below rather
than returning a report whose series are shorter than the run: a short series is
indistinguishable from one a buggy loop recorded, and "one value per iteration" is the
invariant a caller reading the report relies on.

EARLY-STOPPING watches one of those recorded series and ends the loop once it has stopped
improving -- see the `train' generic function's docstring for the spec's four required
keys, and `train-early-stopping-watcher' for why it cannot be combined with
RECORD-HISTORY NIL.
The watcher sees each iteration's entries exactly as the history records them, off the one
`%read-evaluation' call this loop already makes, so what stopped the run and what the
report shows can never be two different readings. `training-report-num-rounds' needs
nothing extra to report the shortened run: it has counted actual iterations since Phase 3a.

OBJECTIVE replaces `XGBoosterUpdateOneIter' with `XGBoosterTrainOneIter' for every iteration
of the loop, driven by the gradient and Hessian the caller's own function returns -- see the
`train' generic function's docstring for what that function is called with and what it must
return. Signals `capability-unavailable' naming `:custom-objective' for a non-NIL OBJECTIVE
when the capability reads false, before any foreign call: see `%check-custom-objective'
above, which reads the capability rather than this backend's name, and `*optional-symbols*'
for why the answer here is probed rather than declared. OBJECTIVE NIL, the default, reaches
no check and runs exactly the `XGBoosterUpdateOneIter' loop this method has always run.

PARAMETERS is passed through untouched, unlike `cl-gbdt/src/lightgbm/protocol''s `train',
which forces `objective' to "none" because `LGBM_BoosterUpdateOneIterCustom' refuses to run
while the booster holds an objective function. `XGBoosterTrainOneIter' has no such
restriction -- measured, a custom update is accepted with any objective set -- so there is
nothing here to override, and this method calls `objective-parameters' nowhere. What that
costs the caller is that the configured objective's PREDICTION TRANSFORM stays in effect:
with `binary:logistic' still set, `predict :kind :normal' on the resulting booster returns
probabilities of a margin the caller's own loss produced, while `:raw' returns that margin.
The generic function's docstring states this as the caller's decision; nothing here signals
or warns about it. `num_class' is likewise just another parameter, and 3 of it alone gives
three output groups -- no `multi:*' objective is needed for a multiclass custom-objective run.

Each iteration reads the booster's current raw scores with `%booster-predictions' at `:raw' --
an `XGBoosterPredictFromDMatrix' margin prediction over the training DMatrix, this library
having no counterpart to LightGBM's `LGBM_BoosterGetPredict' that hands back scores it
already holds. It costs a prediction pass per iteration, which that backend's loop does not
pay. The result reaches OBJECTIVE as a (ROWS GROUPS) `double-float' array, where ROWS comes
from `%dataset-num-rows' on the training set's own pointer and GROUPS is divided out of the
prediction's reported element count by `%predict-ncol' rather than read from a parameter.
What comes back is checked by `check-objective-result' -- `cl-gbdt/src/config/objective''s,
the same backend-neutral pure code LightGBM's `train' calls, so both backends refuse the same
shapes with the same `dimension-mismatch' -- and only then flattened into the C buffers,
ROW-MAJOR on this backend (row I of group K at `(+ (* I GROUPS) K)', which is what an
`__array_interface__' of shape `[ROWS, GROUPS]' means) and converted to `single-float'. Both
the flattening and the score layout are measured; see `%train-one-iteration-custom' and
`%booster-predictions'. The flattening is this method's business and not the caller's:
OBJECTIVE is handed, and returns, a (ROWS GROUPS) array whichever order the library underneath
wants it in -- LightGBM wants the other one.

OBJECTIVE is funcalled inside this method's own `with-foreign-float-traps-masked' body wrap,
so the caller's Lisp arithmetic runs under the masked convention on x86-64 as well as on
aarch64 -- `(/ 1.0d0 0.0d0)' yields infinity there rather than signalling
`division-by-zero'. Nothing about that is specific to a custom objective; it is simply where
in `train' the caller's code now runs. A condition the caller's function does signal
propagates out of `train' through the `unwind-protect' below, freeing the booster handle
rather than orphaning it, exactly as a mid-loop foreign failure does.

An objective that frees a handle this loop depends on, or closes BACKEND, is caught rather
than crashed on: `%recheck-train-datasets' re-runs this method's own opening checks the
moment the `funcall' returns, and TRAIN-DATA-POINTER, VALID-SET-POINTERS and
DATASET-POINTERS are all reassigned from what it returns, so nothing after the caller's code
uses a pointer read before it. See that function for what each of the three re-checks is
for. This is the only place the loop needs it -- the OBJECTIVE NIL branch beside it runs no
caller code at all.

Such a run also leaves this method's own cleanup to run against a closed BACKEND:
`free-booster' then emits an unfreed-handle warning instead of signalling, and the
foreign booster is genuinely leaked, since the library may already be unmapped by the
time that cleanup runs -- see the comment on the `unwind-protect' below for why letting
that warning escape is correct.

Neither RECORD-HISTORY nor EARLY-STOPPING is disabled by OBJECTIVE, and neither is made
meaningful by it: a metric configured through PARAMETERS relates to the library's own
objective, not to the caller's, and this method neither signals nor warns about that -- see
the `train' generic function's docstring, which states it as the caller's decision.

The argument is accepted by this lambda list rather than being absent from it: `train' is one
generic function, so a method that did not take the keyword at all would answer a caller who
named it with SBCL's `unknown-keyword-argument' rather than with the typed condition every
other unavailable capability on this backend answers with.

EVALUATION adds the caller's own metric to what each iteration records, one call per dataset
per iteration -- see the `train' generic function's docstring for what that function is called
with and what it must return. Signals `capability-unavailable' naming `:custom-evaluation'
when the capability reads false, and `unsupported-argument' naming "train's :evaluation" for
RECORD-HISTORY NIL or for a non-function, all three before any foreign call: see
`%check-custom-evaluation' above, and `*provided-capabilities*' for why the answer here is
DECLARED where LightGBM's is probed. EVALUATION NIL, the default, reaches no check and records
exactly what this method has always recorded.

The calls happen after this iteration's own `%read-evaluation' and BEFORE the history push
and the watcher, in `%custom-evaluation-entries' -- so the entries the history keeps and the
entries the watcher sees are one list, and `:early-stopping' can watch a custom metric with
nothing here to arrange it. The custom entries are APPENDED after every library entry, which
is what makes `training-report-from-history''s first-seen ordering put the library's series
first, as a prefix, exactly where `evaluation' reports them; a custom metric never reaches
`evaluation' at all, that method reading only `XGBoosterEvalOneIter'.

Each dataset's predictions are read with `%booster-predictions' at `:normal', over that
dataset's OWN DMatrix pointer and OWN row count -- a fresh `XGBoosterPredictFromDMatrix' pass
per dataset per iteration, this library having nothing that hands back predictions it already
holds. They are `predict :kind :normal''s numbers and not the `:raw' margin OBJECTIVE is
handed, which on this backend are genuinely different numbers in the same run: `train'
rewrites no parameter, so a configured objective's transform stays in effect for both. Both
facts are measured; see `%booster-predictions'. ROW-COUNTS is read once before the loop, from
the same pointers this method already validated, and only when EVALUATION is non-NIL, so a run
that asks for no custom metric makes no extra foreign call at all.

An EVALUATION that frees a handle this loop depends on, or closes BACKEND, is caught the same
way an OBJECTIVE is: `%custom-evaluation-entries' calls `%recheck-train-datasets' the moment
each `funcall' returns -- between two consecutive datasets' reads, not once per iteration --
and TRAIN-DATA-POINTER, VALID-SET-POINTERS and DATASET-POINTERS are all reassigned from the
list it returns, exactly as they are on the OBJECTIVE path.

DATASET and every VALID-SETS entry's dataset half are each run through
`%check-xgboost-dataset' before any foreign call. `train' dispatches on BACKEND, not on
DATASET, so unlike `dataset-num-rows' or `free-dataset' there is no CLOS specializer here
to rule out the wrong kind of handle first -- without this, `handle-live-pointer' would
happily hand `XGBoosterCreate' a booster's own pointer to use as one of its DMatrix
handles. Signals `wrong-backend-reference' when DATASET or a VALID-SETS entry's dataset
half is not an `xgboost-dataset', and `released-handle-error' or `backend-not-open' when
one is but has already been freed or had its own backend closed. A VALID-SETS entry that
is a cons with a non-string car never reaches this check at all: `%valid-set-name' signals
`unsupported-argument' for it first, which is the different mistake a malformed name is,
kept distinct from a wrong dataset handle.

The returned booster retains DATASET as its training set and a fresh copy of VALID-SETS
as its validation sets, keeping all of them alive for the booster's lifetime and letting
`update-one-iteration' notice if any is freed out from under it -- see
`%check-booster-datasets-live'. The copy matters: VALID-SETS is the caller's own list,
and `make-handle' would otherwise store that exact list object rather than a snapshot of
it. A caller who destructively removes an entry from VALID-SETS after `train' returns --
`delete', `(setf (cdr ...))', reusing the list elsewhere with `nconc' -- would silently
remove it from the booster's view too, since both would be the same cons cells; the
DMatrix `XGBoosterCreate' already attached would then go unchecked by
`%check-booster-datasets-live' even though XGBoost still holds its pointer -- the same
hazard `cl-gbdt/src/lightgbm/protocol''s `train' guards against, for the identical reason.
Free the result with `free-booster' or wrap it in `with-booster'.

BOOSTER is bound to a full handle already: `create-booster' manages the raw-pointer window
between `XGBoosterCreate' and its own `make-handle' call internally -- see its docstring
-- and this method never touches a pointer that let does not already own. What the
`unwind-protect' below manages instead is what happens to that handle from here: any exit
from the loop or the report construction that does not reach the final `setf' frees BOOSTER
rather than orphaning it.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'.
```

<a id="cl-gbdt-training-report"></a>

## `cl-gbdt:training-report`

- **Kind** class
- **Superclasses** `standard-object`
- **Exported from** `cl-gbdt`

```text
What a training run recorded, returned as `train''s secondary value.

Policy section 9 asks a training report to express the dataset, the metric, the per-iteration
values, the best iteration, the best score, and whether early stopping happened. All six are
here; the last three read NIL unless `train' was given :EARLY-STOPPING -- see each slot's own
documentation for why NIL is "not determined" rather than an invented default. A class rather
than a plist so a caller can rely on each field saying what it means regardless of whether
this run used early stopping.
```

### Slots

#### `series`

- **Readers** `training-report-series`

```text
A list of `training-series', one per metric per dataset. Empty when
the booster has no metrics configured -- LightGBM's `metric=none'.
```

#### `num-rounds`

- **Readers** `training-report-num-rounds`

```text
How many boosting iterations actually ran. Equal to `train''s
:NUM-ROUNDS in this phase; Phase 3b's early stopping is what can make it smaller, which is why
it is recorded rather than left for the caller to assume.
```

#### `best-iteration`

- **Readers** `training-report-best-iteration`

```text
Which iteration produced `best-score', or NIL.

Filled only when `train' was given :EARLY-STOPPING; every other run -- including one that
completes normally with :EARLY-STOPPING never given at all -- leaves this NIL. Finding the
best iteration means knowing whether a metric improves upward or downward, and policy
section 9 forbids inferring that from the metric's name; :EARLY-STOPPING is what supplies
that direction explicitly, which is what makes filling this slot possible at all. NIL means
"not determined", never "iteration 0" -- a run's best iteration can genuinely be iteration
0, and NIL must stay distinguishable from that answer.
```

#### `best-score`

- **Readers** `training-report-best-score`

```text
The best value `best-iteration' achieved, or NIL.

Filled under exactly the same condition as `training-report-best-iteration' and for the same
reason: only when `train' was given :EARLY-STOPPING. NIL here means "not determined", the
same as for `best-iteration' -- not "no improvement was ever seen", which would be a real,
reportable outcome once :EARLY-STOPPING is in use.
```

#### `early-stopped-p`

- **Readers** `training-report-early-stopped-p`

```text
True when the run stopped before `num-rounds' iterations
because :EARLY-STOPPING's patience was exhausted.

NIL in two cases this slot does not distinguish: :EARLY-STOPPING was given but the run
completed on its own before triggering it, and :EARLY-STOPPING was not given at all. Neither
case stopped early for a reason this slot needs to report, which is why both read the same
as "not determined" reads for `best-iteration' and `best-score'.
```

<a id="cl-gbdt-training-report-best-iteration"></a>

## `cl-gbdt:training-report-best-iteration`

- **Kind** generic function
- **Signature** `(training-report-best-iteration object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:training-report`'s `best-iteration` slot. See `cl-gbdt:training-report`.

<a id="cl-gbdt-training-report-best-score"></a>

## `cl-gbdt:training-report-best-score`

- **Kind** generic function
- **Signature** `(training-report-best-score object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:training-report`'s `best-score` slot. See `cl-gbdt:training-report`.

<a id="cl-gbdt-training-report-early-stopped-p"></a>

## `cl-gbdt:training-report-early-stopped-p`

- **Kind** generic function
- **Signature** `(training-report-early-stopped-p object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:training-report`'s `early-stopped-p` slot. See `cl-gbdt:training-report`.

<a id="cl-gbdt-training-report-num-rounds"></a>

## `cl-gbdt:training-report-num-rounds`

- **Kind** generic function
- **Signature** `(training-report-num-rounds object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:training-report`'s `num-rounds` slot. See `cl-gbdt:training-report`.

<a id="cl-gbdt-training-report-series"></a>

## `cl-gbdt:training-report-series`

- **Kind** generic function
- **Signature** `(training-report-series object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:training-report`'s `series` slot. See `cl-gbdt:training-report`.

<a id="cl-gbdt-training-series"></a>

## `cl-gbdt:training-series`

- **Kind** class
- **Superclasses** `standard-object`
- **Exported from** `cl-gbdt`

```text
One metric's values over one dataset, across a training run.
```

### Slots

#### `index`

- **Readers** `training-series-index`

```text
The dataset's position among the datasets the booster retains: 0 is
the training set, 1 the first `train' :VALID-SETS entry, and so on. This is each backend's own
identifier -- LightGBM knows a validation set by index and by nothing else -- so it is always
present, whether or not the caller supplied a name.
```

#### `name`

- **Readers** `training-series-name`

```text
The name the caller gave this dataset in `train''s :VALID-SETS, or NIL.

NIL for the training set, which is never a :VALID-SETS entry and so has no name a caller could
supply, and NIL for any validation set passed bare rather than as a (NAME . DATASET) cons.
Nothing here invents one.
```

#### `metric`

- **Readers** `training-series-metric`

```text
The metric's name, as the backend spells it -- LightGBM's
"binary_logloss" and XGBoost's "logloss" are different strings for the same idea, and
neither is translated.
```

#### `values`

- **Readers** `training-series-values`

```text
One element per completed iteration, in order: a `double-float', or
NIL where the backend reported a value that could not be read as a real.

A `simple-vector' rather than a `(vector double-float)' precisely so NIL can appear. XGBoost
formats metric values through `std::ostream', which writes `nan' and `inf' for non-finite
doubles; policy section 5 says such a field is reported as unreadable rather than dropped or
replaced by an invented number, and dropping it would also slide every later value one
iteration earlier.

`double-float' holds for a `train' :EVALUATION's own values too, and holds by COERCION rather
than by luck: a caller's function may return any real, and `custom-metric-entry'
(`cl-gbdt/src/training/custom-metric') coerces what it returns before the entry is built, so
a 1/3 is recorded as 0.3333333333333333d0 rather than as a `ratio', and a real too large for
a `double-float' as the signed infinity rather than as a signalled overflow. That is what
keeps this slot's promise the same one for every series in a report, whoever produced it.
```

<a id="cl-gbdt-training-series-index"></a>

## `cl-gbdt:training-series-index`

- **Kind** generic function
- **Signature** `(training-series-index object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:training-series`'s `index` slot. See `cl-gbdt:training-series`.

<a id="cl-gbdt-training-series-metric"></a>

## `cl-gbdt:training-series-metric`

- **Kind** generic function
- **Signature** `(training-series-metric object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:training-series`'s `metric` slot. See `cl-gbdt:training-series`.

<a id="cl-gbdt-training-series-name"></a>

## `cl-gbdt:training-series-name`

- **Kind** generic function
- **Signature** `(training-series-name object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:training-series`'s `name` slot. See `cl-gbdt:training-series`.

<a id="cl-gbdt-training-series-values"></a>

## `cl-gbdt:training-series-values`

- **Kind** generic function
- **Signature** `(training-series-values object)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:training-series`'s `values` slot. See `cl-gbdt:training-series`.

<a id="cl-gbdt-unfreed-handle-warning"></a>

## `cl-gbdt:unfreed-handle-warning`

- **Kind** condition
- **Superclasses** `warning`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
A handle was collected while still holding a live foreign pointer.

Signalled from a finalizer, which reports and does **not** free: running the C free from
whatever thread the GC chose would give no ordering guarantee between a booster and the
dataset it holds, and `with-booster' nested inside `with-dataset' exists precisely to
guarantee that order.
```

### Slots

#### `kind`

- **Readers** `unfreed-handle-warning-kind`

```text
The keyword naming which sort of handle this was -- `:dataset'
or `:booster' -- passed straight through from the `make-handle' call that built it.
```

<a id="cl-gbdt-unfreed-handle-warning-kind"></a>

## `cl-gbdt:unfreed-handle-warning-kind`

- **Kind** generic function
- **Signature** `(unfreed-handle-warning-kind condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:unfreed-handle-warning`'s `kind` slot. See `cl-gbdt:unfreed-handle-warning`.

<a id="cl-gbdt-unknown-backend"></a>

## `cl-gbdt:unknown-backend`

- **Kind** condition
- **Superclasses** `backend-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
OPEN-BACKEND was called with a name no backend system has registered.

Distinct from `backend-not-open', which is about an operation attempted on a
backend instance that exists but has not been opened yet; this is about a name
that `find-backend-class' does not know at all.
```

### Slots

#### `registered`

- **Readers** `unknown-backend-registered`

```text
List of currently registered backend names.
```

<a id="cl-gbdt-unknown-backend-registered"></a>

## `cl-gbdt:unknown-backend-registered`

- **Kind** generic function
- **Signature** `(unknown-backend-registered condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:unknown-backend`'s `registered` slot. See `cl-gbdt:unknown-backend`.

<a id="cl-gbdt-unknown-capability"></a>

## `cl-gbdt:unknown-capability`

- **Kind** condition
- **Superclasses** `gbdt-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
Signalled when a capability keyword is not in `*known-capabilities*'.

A programming error, not a backend limitation: returning NIL for a misspelled capability
would be indistinguishable from "this backend does not support it", and a caller that
believes the second silently skips the feature. Policy section 7 forbids exactly that
silent path.
```

### Slots

#### `capability`

- **Readers** `unknown-capability-capability`

```text
The keyword the caller asked about.
```

#### `known`

- **Readers** `unknown-capability-known`

```text
Every capability name this build knows.
```

<a id="cl-gbdt-unknown-capability-capability"></a>

## `cl-gbdt:unknown-capability-capability`

- **Kind** generic function
- **Signature** `(unknown-capability-capability condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:unknown-capability`'s `capability` slot. See `cl-gbdt:unknown-capability`.

<a id="cl-gbdt-unknown-capability-known"></a>

## `cl-gbdt:unknown-capability-known`

- **Kind** generic function
- **Signature** `(unknown-capability-known condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:unknown-capability`'s `known` slot. See `cl-gbdt:unknown-capability`.

<a id="cl-gbdt-unsupported-argument"></a>

## `cl-gbdt:unsupported-argument`

- **Kind** condition
- **Superclasses** `data-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
A caller supplied an argument that BACKEND has no way to honor.

Unlike `wrong-backend-reference', which is about a caller-supplied *handle* built by the
wrong backend, this is about an argument naming a concept the backend does not have at all
-- `make-dataset''s :REFERENCE on XGBoost, for example, which has no bin-mapper to align to.
Signalling here, instead of silently discarding the argument, is deliberate: this project has
repeatedly found a silently-dropped argument to be the failure mode where a caller moves
working code from one backend to the other and gets different, wrong behaviour with no
indication anything changed.
```

### Slots

#### `backend`

- **Readers** `unsupported-argument-backend`

```text
Name of the backend the argument was passed to.
```

#### `argument`

- **Readers** `unsupported-argument-argument`

```text
Description of which caller-supplied argument is unsupported,
e.g. "make-dataset's :reference", for the report.
```

#### `reason`

- **Readers** `unsupported-argument-reason`

```text
Why BACKEND cannot honor the argument.
```

<a id="cl-gbdt-unsupported-argument-argument"></a>

## `cl-gbdt:unsupported-argument-argument`

- **Kind** generic function
- **Signature** `(unsupported-argument-argument condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:unsupported-argument`'s `argument` slot. See `cl-gbdt:unsupported-argument`.

<a id="cl-gbdt-unsupported-argument-backend"></a>

## `cl-gbdt:unsupported-argument-backend`

- **Kind** generic function
- **Signature** `(unsupported-argument-backend condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:unsupported-argument`'s `backend` slot. See `cl-gbdt:unsupported-argument`.

<a id="cl-gbdt-unsupported-argument-reason"></a>

## `cl-gbdt:unsupported-argument-reason`

- **Kind** generic function
- **Signature** `(unsupported-argument-reason condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:unsupported-argument`'s `reason` slot. See `cl-gbdt:unsupported-argument`.

<a id="cl-gbdt-unsupported-element-type"></a>

## `cl-gbdt:unsupported-element-type`

- **Kind** condition
- **Superclasses** `data-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
An array with an unsupported element type was supplied.
```

### Slots

#### `given`

- **Readers** `unsupported-element-type-given`

```text
Element type of the array that was supplied.
```

<a id="cl-gbdt-unsupported-element-type-given"></a>

## `cl-gbdt:unsupported-element-type-given`

- **Kind** generic function
- **Signature** `(unsupported-element-type-given condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:unsupported-element-type`'s `given` slot. See `cl-gbdt:unsupported-element-type`.

<a id="cl-gbdt-untested-backend-version"></a>

## `cl-gbdt:untested-backend-version`

- **Kind** condition
- **Superclasses** `warning`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
A library version outside the tested list was loaded.

This is a warning rather than an error so that a new upstream release does not
immediately render cl-gbdt unusable.
```

### Slots

#### `backend`

- **Readers** `untested-backend-version-backend`

```text
The keyword naming the backend whose loaded library this is
about -- in practice always `:xgboost', since LightGBM's C API has no version entry
point for `check-backend-version' to call.
```

#### `version`

- **Readers** `untested-backend-version-version`

```text
VERSION as `check-backend-version' was given it, unchanged --
`%parse-version' is used only for the internal range comparison and never rewrites this
slot. Three cases: a parseable "MAJOR.MINOR.PATCH" string that fell outside RANGE, an
unparseable string stored verbatim -- e.g. "not-a-version" -- or NIL, a separate case
from an unparseable string: the backend had no version entry point to read at all.
```

#### `tested`

- **Readers** `untested-backend-version-tested`

```text
Two strings describing what is actually known: the narrower
verified range with its evidence, then the wider inferred range with its own -- see
`version-range-tested-description'.
```

<a id="cl-gbdt-untested-backend-version-backend"></a>

## `cl-gbdt:untested-backend-version-backend`

- **Kind** generic function
- **Signature** `(untested-backend-version-backend condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:untested-backend-version`'s `backend` slot. See `cl-gbdt:untested-backend-version`.

<a id="cl-gbdt-untested-backend-version-tested"></a>

## `cl-gbdt:untested-backend-version-tested`

- **Kind** generic function
- **Signature** `(untested-backend-version-tested condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:untested-backend-version`'s `tested` slot. See `cl-gbdt:untested-backend-version`.

<a id="cl-gbdt-untested-backend-version-version"></a>

## `cl-gbdt:untested-backend-version-version`

- **Kind** generic function
- **Signature** `(untested-backend-version-version condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:untested-backend-version`'s `version` slot. See `cl-gbdt:untested-backend-version`.

<a id="cl-gbdt-update-one-iteration"></a>

## `cl-gbdt:update-one-iteration`

- **Kind** generic function
- **Signature** `(update-one-iteration booster)`
- **Exported from** `cl-gbdt`

```text
Advance BOOSTER by one boosting iteration.

Use this to drive the training loop yourself. Returns false when no further split was
possible and the backend can report that -- LightGBM does. XGBoost's booster protocol
has no equivalent signal, so its `update-one-iteration' always returns true after a
successful call; treat a true return as "the call succeeded", not as proof a split
happened, unless the backend is known to be LightGBM.
```

### Methods

#### `(update-one-iteration (booster booster))`

```text
Fallback for a BOOSTER whose backend's unified-API methods are not loaded: signals
`backend-methods-not-loaded' naming `update-one-iteration' rather than advancing anything. A
Layer 1 caller drives the same loop with that backend's own `update-one-iteration' in
`api.lisp'. `train''s own loop does not call that function either: it calls `native.lisp''s
`%update-one-iteration' directly, because `api.lisp''s `update-one-iteration' would re-check
every handle on every iteration -- see that backend's `train' method for the same note in
its own words.
```

#### `(update-one-iteration (booster lightgbm-booster))`

```text
Advance BOOSTER by one boosting iteration via `LGBM_BoosterUpdateOneIter'.

Returns false when the iteration just run produced no further split, per the generic
function's contract -- about that call, not a latch for the rest of the run; see
`cl-gbdt/src/lightgbm/api''s function of the same name, which spells that
distinction out. Signals `released-handle-error' when BOOSTER's training set,
or any of its validation sets, has already been freed -- see
`%check-booster-datasets-live'.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/lightgbm/api''s `update-one-iteration', which is
where the liveness check and the `produced_empty_tree' inversion now live.
```

#### `(update-one-iteration (booster xgboost-booster))`

```text
Advance BOOSTER by one boosting iteration via `XGBoosterUpdateOneIter'.

Unlike LightGBM's `LGBM_BoosterUpdateOneIter', which reads the booster's internal
training-set pointer implicitly, XGBoost's version takes the DMatrix handle explicitly,
so this reads it back from `booster-training-set' rather than being able to omit it. A
`load-model' booster's training set is NIL by design -- see the `booster' class'
documentation -- and handing `XGBoosterUpdateOneIter' a null DMatrixHandle would not
come back as a status code the way a bad parameter does: it is a null-pointer dereference
inside XGBoost's own implementation. That case is rejected here, before the foreign call,
for the same reason `%check-booster-datasets-live' exists for the pointers it does check.

XGBoost also reports no `produced_empty_tree'-style signal from this call, unlike
LightGBM -- there is nothing for this backend to report a false return for, so unlike
`cl-gbdt/src/lightgbm/protocol''s method of the same name, this always returns true after
a successful call; the generic function's "returns false when no further split was
possible" applies only insofar as a backend can report it, which this one cannot.

Signals `released-handle-error' when BOOSTER's training set, or any of its validation
sets, has already been freed -- see `%check-booster-datasets-live'. Signals
`missing-training-set' when BOOSTER has no training set at all -- a `load-model'
booster, which never went through `train' -- since handing `XGBoosterUpdateOneIter' a
null DMatrixHandle in that case is a null-pointer dereference, not something it can
reject with a status code.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/xgboost/api''s `update-one-iteration', which is
where the two checks and the explicit training-set pointer now live.
```

<a id="cl-gbdt-lightgbm-update-one-iteration"></a>

## `cl-gbdt/lightgbm:update-one-iteration`

- **Kind** function
- **Signature** `(update-one-iteration booster)`
- **Exported from** `cl-gbdt/lightgbm`

```text
Advance BOOSTER by one boosting iteration via `LGBM_BoosterUpdateOneIter'.

Returns false when THIS iteration produced no further split -- LightGBM's own
`produced_empty_tree' out parameter, read by `%update-one-iteration' and inverted here, so
that a true return means the iteration did produce one. Per call, and about that call only:
the C function recomputes the flag every time and promises nothing about the next one, so a
false return is not a latch and a loop that stops on it is making its own decision, not
reading one the library has made. `cl-gbdt/src/protocol''s generic says the same in the
portable words "returns false when no further split was possible".

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, an XGBoost booster, or not a handle at all -- and this function dispatches on
nothing, so `%check-lightgbm-booster' is the only thing between such a handle's pointer and
`LGBM_BoosterUpdateOneIter'. Measured against the vendored library with the check absent, an
`xgboost-booster' came back as a `foreign-call-error' rather than a fault, but only because
LightGBM happened to return -1 for that particular pointer: nothing in the C API examines a
handle it is given, so that is a fact about one measurement, not a contract.

Signals `released-handle-error' when BOOSTER's training set, or any of its validation sets,
has already been freed -- see `%check-booster-datasets-live', which runs before any foreign
call because `LGBM_BoosterUpdateOneIter' dereferences those datasets' pointers itself and a
freed one is a segfault rather than a catchable condition. Signals `released-handle-error'
for a freed BOOSTER, and `backend-not-open' when its backend has since been closed -- both
from the `handle-live-pointer' inside `%check-lightgbm-booster', which is why the kind check
is first: a handle this backend never built is the wrong handle whatever its state, and
`%check-booster-datasets-live' would otherwise read slots off it before anything questioned
what it was.

PRECEDENCE, when more than one of those is true at once. BOOSTER's own state is examined
before its datasets', so a fault in the booster or its backend WINS over a freed training set.
Measured against the vendored library, before and after the kind check moved ahead of
`%check-booster-datasets-live':

  booster freed + training set freed   was `released-handle-error' naming the DATASET,
                                       is `released-handle-error' naming the BOOSTER
  training set freed + backend closed  was `released-handle-error' naming the dataset,
                                       is `backend-not-open'
  training set freed alone             `released-handle-error' naming the dataset, unchanged

Both changes are deliberate and neither is a widening: a released handle, or a shared library
`close-backend' has unmapped, is a more fundamental fault than the dataset a still-usable
booster points at, and reporting the dataset while the booster itself was unusable named the
lesser of the two. `cl-gbdt/src/xgboost/api''s `update-one-iteration' records the same
precedence and one further row of its own.
```

<a id="cl-gbdt-xgboost-update-one-iteration"></a>

## `cl-gbdt/xgboost:update-one-iteration`

- **Kind** function
- **Signature** `(update-one-iteration booster)`
- **Exported from** `cl-gbdt/xgboost`

```text
Advance BOOSTER by one boosting iteration via `XGBoosterUpdateOneIter'. Always returns
true after a successful call.

Unlike LightGBM's `LGBM_BoosterUpdateOneIter', which reads the booster's internal
training-set pointer implicitly, XGBoost's version takes the DMatrix handle explicitly, so
this reads it back from `booster-training-set' rather than being able to omit it. A
`load-model' booster's training set is NIL by design -- see the `booster' class'
documentation -- and handing `XGBoosterUpdateOneIter' a null DMatrixHandle would not come back
as a status code the way a bad parameter does: it is a null-pointer dereference inside
XGBoost's own implementation. That case is rejected here, before the foreign call, for the
same reason `%check-booster-datasets-live' exists for the pointers it does check.

XGBoost also reports no `produced_empty_tree'-style signal from this call, unlike LightGBM, so
there is nothing here to return false for -- where `cl-gbdt/src/lightgbm/api''s function of
the same name returns false once an iteration produced no further split, this one cannot tell,
and says so by always returning true.

Signals `released-handle-error' when BOOSTER's training set, or any of its validation sets,
has already been freed -- see `%check-booster-datasets-live', which runs before any foreign
call and whose docstring measures why the two kinds are not the same hazard: a freed TRAINING
set faults inside the library, since its handle is what this call passes to C, while a freed
VALIDATION set is refused on contract rather than for safety, that array not being consulted
here at all. Signals `missing-training-set' when
BOOSTER has no training set at all.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, a LightGBM booster, or not a handle at all -- and this function dispatches on
nothing, so `%check-xgboost-booster' is the only thing between such a handle's pointer and
`XGBoosterUpdateOneIter'. Signals `released-handle-error' for a freed BOOSTER, and
`backend-not-open' when its backend has since been closed -- both from the
`handle-live-pointer' inside `%check-xgboost-booster', which is why the kind check is first: a
handle this backend never built is the wrong handle whatever its state, and both
`%check-booster-datasets-live' and the `booster-training-set' read below would otherwise take
slots off it before anything questioned what it was.

PRECEDENCE, when more than one of those is true at once. BOOSTER's own state is examined
before its datasets' and before the question of whether it has one, so a fault in the booster
or its backend WINS over both. Measured against the vendored library, before and after the
kind check moved ahead of the other two:

  booster freed + training set freed   was `released-handle-error' naming the DATASET,
                                       is `released-handle-error' naming the BOOSTER
  training set freed + backend closed  was `released-handle-error' naming the dataset,
                                       is `backend-not-open'
  freed booster with NO training set   was `missing-training-set',
                                       is `released-handle-error'
  training set freed alone             `released-handle-error' naming the dataset, unchanged

All three changes are deliberate and none is a widening: a released handle, or a shared
library `close-backend' has unmapped, is a more fundamental fault than the dataset a
still-usable booster points at, and the third row was reading a RELEASED booster's slots in
order to discover that it had no training set -- answering a question about a handle that
should not have been read at all. The third row has no LightGBM counterpart, that backend
having no `missing-training-set' guard; `cl-gbdt/src/lightgbm/api''s `update-one-iteration'
records the other two.
```

<a id="cl-gbdt-version-compare"></a>

## `cl-gbdt:version-compare`

- **Kind** function
- **Signature** `(version-compare a b)`
- **Exported from** `cl-gbdt`

```text
Compare two "MAJOR.MINOR.PATCH" version strings A and B component by component.

Returns `:less', `:equal' or `:greater' when both parse. Returns NIL when either does
not -- including either being NIL itself. NIL deliberately does not sort as the
smallest possible version: something this project cannot even parse is not
comparable at all, not merely low.
```

<a id="cl-gbdt-version-in-range-p"></a>

## `cl-gbdt:version-in-range-p`

- **Kind** function
- **Signature** `(version-in-range-p version low high)`
- **Exported from** `cl-gbdt`

```text
Return true when VERSION falls within [LOW, HIGH], inclusive, all three
"MAJOR.MINOR.PATCH" strings compared with `version-compare'.

Returns NIL -- "cannot confirm this is in range", not "confirmed out of range" --
when VERSION, LOW or HIGH fails to parse. VERSION = NIL needs no special case:
`version-compare' already returns NIL for it, and `member' on a NIL first argument
returns NIL in turn.
```

<a id="cl-gbdt-version-range"></a>

## `cl-gbdt:version-range`

- **Kind** structure
- **Exported from** `cl-gbdt`

```text
What is known about a backend's compatible library versions, split into the
VERIFIED and INFERRED claims this file's header comment distinguishes.

VERIFIED-LOW and VERIFIED-HIGH bound the versions the functional suite has actually
trained and predicted against; today a single point (VERIFIED-LOW = VERIFIED-HIGH)
for both backends. VERIFIED-EVIDENCE names what backs it, for the warning's report.

INFERRED-LOW and INFERRED-HIGH bound the wider span `tools/check-upstream.lisp'
argues is ABI-compatible from C header comparison alone. INFERRED-EVIDENCE names
that argument. VERIFIED-LOW/HIGH always fall within [INFERRED-LOW, INFERRED-HIGH].
```

### Slots

#### `verified-low`

- **Readers** `version-range-verified-low`


#### `verified-high`

- **Readers** `version-range-verified-high`


#### `verified-evidence`

- **Readers** `version-range-verified-evidence`


#### `inferred-low`

- **Readers** `version-range-inferred-low`


#### `inferred-high`

- **Readers** `version-range-inferred-high`


#### `inferred-evidence`

- **Readers** `version-range-inferred-evidence`


<a id="cl-gbdt-version-range-inferred-evidence"></a>

## `cl-gbdt:version-range-inferred-evidence`

- **Kind** function
- **Signature** `(version-range-inferred-evidence instance)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:version-range`'s `inferred-evidence` slot. See `cl-gbdt:version-range`.

<a id="cl-gbdt-version-range-inferred-high"></a>

## `cl-gbdt:version-range-inferred-high`

- **Kind** function
- **Signature** `(version-range-inferred-high instance)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:version-range`'s `inferred-high` slot. See `cl-gbdt:version-range`.

<a id="cl-gbdt-version-range-inferred-low"></a>

## `cl-gbdt:version-range-inferred-low`

- **Kind** function
- **Signature** `(version-range-inferred-low instance)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:version-range`'s `inferred-low` slot. See `cl-gbdt:version-range`.

<a id="cl-gbdt-version-range-tested-description"></a>

## `cl-gbdt:version-range-tested-description`

- **Kind** function
- **Signature** `(version-range-tested-description range)`
- **Exported from** `cl-gbdt`

```text
Return RANGE's evidence as a list of two strings -- the verified point/range, then
the wider inferred one -- for `untested-backend-version''s :TESTED initarg.

Kept as two separate entries rather than one blended string, so the condition's
report -- "Tested: ~{~A~^, ~}" -- states each claim's own strength instead of
implying both are the same kind of evidence.
```

<a id="cl-gbdt-version-range-verified-evidence"></a>

## `cl-gbdt:version-range-verified-evidence`

- **Kind** function
- **Signature** `(version-range-verified-evidence instance)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:version-range`'s `verified-evidence` slot. See `cl-gbdt:version-range`.

<a id="cl-gbdt-version-range-verified-high"></a>

## `cl-gbdt:version-range-verified-high`

- **Kind** function
- **Signature** `(version-range-verified-high instance)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:version-range`'s `verified-high` slot. See `cl-gbdt:version-range`.

<a id="cl-gbdt-version-range-verified-low"></a>

## `cl-gbdt:version-range-verified-low`

- **Kind** function
- **Signature** `(version-range-verified-low instance)`
- **Exported from** `cl-gbdt`

Reader of `cl-gbdt:version-range`'s `verified-low` slot. See `cl-gbdt:version-range`.

<a id="cl-gbdt-with-booster"></a>

## `cl-gbdt:with-booster`

- **Kind** macro
- **Signature** `(with-booster (var form) &body body)`
- **Exported from** `cl-gbdt`

```text
Bind VAR to the booster FORM returns, evaluate BODY, and always free it.

A booster holds a strong reference to the dataset it was trained on, so nesting this
inside `with-dataset' cannot invert the release order. Declarations at the head of
BODY are shadow-bound as in `with-dataset', for the same reason.
```

<a id="cl-gbdt-with-dataset"></a>

## `cl-gbdt:with-dataset`

- **Kind** macro
- **Signature** `(with-dataset (var form) &body body)`
- **Exported from** `cl-gbdt`

```text
Bind VAR to the dataset FORM returns, evaluate BODY, and always free it.

Explicit resource management is the first-class pattern; finalizers are only a
safety net, and that net reports and does not free: running the C free from
whatever thread the garbage collector chose would give no ordering guarantee
between a booster and the dataset it holds, and `with-booster' nested inside
this macro exists precisely to guarantee that order.

Declarations at the head of BODY are moved onto a fresh binding of VAR that
shadows the one FORM's value is stored in, scoped to BODY alone. Splicing them
onto the outer binding instead -- the one `unwind-protect''s cleanup clause also
reads to call `free-dataset' -- would put an `(ignore VAR)' declaration from
BODY in the same scope as that read, which SBCL flags as "reading an ignored
variable" (verified empirically; do not simplify this back to `progn' or a
single binding).
```

<a id="cl-gbdt-with-foreign-float-traps-masked"></a>

## `cl-gbdt:with-foreign-float-traps-masked`

- **Kind** macro
- **Signature** `(with-foreign-float-traps-masked &body body)`
- **Exported from** `cl-gbdt`

```text
Evaluate BODY with the :INVALID, :DIVIDE-BY-ZERO and :OVERFLOW floating-point traps
masked, restoring whatever trap set was in effect once BODY returns or unwinds.

Both wrapped libraries are C code, written and tested against the ordinary C
floating-point environment, where these three IEEE-754 exceptions are masked by
default: an intermediate NaN or infinity produced partway through a computation is
data to keep working with, not a signal. SBCL does not honor that convention
uniformly -- confirmed empirically: x86-64 SBCL enables all three traps by default,
aarch64 SBCL enables none of them -- so a call into LightGBM's or XGBoost's own
numeric code (the softmax normalization behind an XGBoost `multi:softprob' prediction
is the case that surfaced this) can take an intermediate value the library was written
to tolerate and, only on x86-64, turn it into an uncaught SBCL
FLOATING-POINT-INVALID-OPERATION in the middle of a foreign call.

Masking here restores the environment every call into these libraries was always
supposed to run in; it does not suppress a real error. SBCL enabling these traps is
the anomaly this macro corrects for, not the library's own arithmetic. It is a
correction, not a general license to ignore floating-point exceptions: callers that
mask a *caller-supplied* computation, not just a foreign call, would be hiding their
own bugs instead of restoring someone else's calling convention.

Every entry point in either backend wraps its entire body in this macro, not just the
specific call this was first found through: every `defmethod' in `src/<backend>/protocol.lisp'
and `src/<backend>/classes.lisp', and every `defun' in `api.lisp', `classes.lisp' or
`native.lisp' that the backend's public package exports and so is reached with no `defmethod'
to inherit a mask from. `api.lisp' is where most of them now are -- it holds each backend's
finished Layer 1 operations, every one of which a caller reaches directly. See those files'
commentary for the enumeration, and
`tools/ci/check-float-traps.lisp', which fails the build when one of them is missing it.

Expands to a plain PROGN under #-SBCL: this project's foreign-call boundary is
SBCL-specific throughout already (see `cl-gbdt/src/data''s `%call-with-pinned-matrix'
for the same split), and no other implementation this project targets is known to
enable these traps by default.
```

<a id="cl-gbdt-with-foreign-matrix"></a>

## `cl-gbdt:with-foreign-matrix`

- **Kind** macro
- **Signature** `(with-foreign-matrix (pointer nrow ncol element-type) matrix &body body)`
- **Exported from** `cl-gbdt`

```text
Make MATRIX available as foreign memory and evaluate BODY.

POINTER, NROW, NCOL and ELEMENT-TYPE are bound within BODY. POINTER becomes invalid
once BODY returns.
```

<a id="cl-gbdt-with-pointer-ownership"></a>

## `cl-gbdt:with-pointer-ownership`

- **Kind** macro
- **Signature** `(with-pointer-ownership (pointer free-function ownership-operator) &body body)`
- **Exported from** `cl-gbdt`

```text
Evaluate BODY with POINTER owned by nobody, freeing it unless BODY takes ownership.

OWNERSHIP-OPERATOR names a local function BODY calls to hand POINTER to a Lisp handle:

  (with-pointer-ownership (raw #'%free-dataset-unchecked take-ownership)
    (%set-info-field raw "label" label)
    (take-ownership 'lightgbm-dataset backend :dataset))

It takes `make-handle''s arguments minus POINTER -- CLASS-NAME, BACKEND, KIND and
`make-handle''s own keyword arguments -- returns the handle it made, and records that
ownership moved. BODY's values are this form's values, so a body ending in
`(values handle report)' returns both.

The gap this closes: the foreign resource exists from the moment the creation call
returns, but nothing in Lisp references it until `make-handle' runs, so nothing will
free it and no finalizer will ever fire for it. A BODY that leaves without calling
OWNERSHIP-OPERATOR -- by signalling, by `throw', by `return-from', or simply by
returning -- would orphan it. FREE-FUNCTION, a function of one argument, is called on
POINTER in exactly that case.

Any error FREE-FUNCTION itself signals is discarded, so a failing cleanup cannot
replace the condition that caused the unwind (policy section 10). POINTER is evaluated
once, and after FREE-FUNCTION: an error from evaluating the free function would leave a
pointer this form had already taken and could no longer free, so the form that decides
HOW to free is evaluated while there is still nothing to lose.
```

<a id="cl-gbdt-write-foreign-sequence"></a>

## `cl-gbdt:write-foreign-sequence`

- **Kind** function
- **Signature** `(write-foreign-sequence pointer cffi-type sequence coercer)`
- **Exported from** `cl-gbdt`

```text
Copy SEQUENCE into the foreign array at POINTER, each element passed through
COERCER before being stored as CFFI-TYPE.
```

<a id="cl-gbdt-wrong-backend-reference"></a>

## `cl-gbdt:wrong-backend-reference`

- **Kind** condition
- **Superclasses** `data-error`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

```text
A caller-supplied dataset or booster argument did not belong to the same
backend as the operation it was passed to, or was the wrong kind of handle outright -- or,
for a Layer 1 operation that takes a BACKEND rather than a handle, that argument was another
backend's object.

Several entry points funnel through this same condition: `make-dataset''s :REFERENCE,
`train''s DATASET argument, and each entry of `train''s :VALID-SETS all expect a dataset,
EXPECTED's default; `cl-gbdt/lightgbm''s `booster-eval' and `booster-eval-names' expect a
booster instead, the first callers to pass `:expected "booster"' -- without that slot,
reusing this condition unmodified for them produced a self-contradictory report, e.g.
"must be a dataset built by LIGHTGBM itself, not LIGHTGBM-DATASET" when a dataset was
rejected because a *booster* was required. Every one of these entry points hands a
caller-supplied handle straight to a foreign function that expects a specific handle kind,
without a CLOS specializer to rule out the wrong one first -- `make-dataset' and `train'
both dispatch on the backend, not on the handle, and `booster-eval'/`booster-eval-names'
are plain functions with no CLOS dispatch at all, so a handle built by a different
backend, of the wrong kind, or not a handle at all, would otherwise reach the foreign call
unexamined. `cl-gbdt/src/lightgbm/api''s and `cl-gbdt/src/xgboost/api''s `create-dataset' and
`create-booster' are here for the same reason one step earlier: each was a `defmethod'
specialized on its own backend class, and as a plain `defun' takes whatever backend object it
is given -- which decides which library's entry point runs and which backend the resulting
handle records its liveness against.

A handle is an opaque pointer as far as the underlying C API is concerned: handing it the
wrong one is undefined behaviour once it crosses the FFI boundary, not something the C
library can reject on its own. This is checked here, before any foreign call, so the
failure is a condition instead of a crash or silent corruption.
```

### Slots

#### `backend`

- **Readers** `wrong-backend-reference-backend`

```text
Name of the backend the caller was operating against.
```

#### `given`

- **Readers** `wrong-backend-reference-given`

```text
The class of the object actually passed where EXPECTED was wanted:
a handle of that kind built by BACKEND, or -- when EXPECTED is "backend" -- BACKEND's own
backend object.
```

#### `argument`

- **Readers** `wrong-backend-reference-argument`

```text
Description of which caller-supplied argument received the wrong
value, e.g. "make-dataset's :reference" or "train's dataset argument", for the report.
```

#### `expected`

- **Readers** `wrong-backend-reference-expected`

```text
What kind of thing the argument should have been, as a noun
for the report -- "dataset" (the default, and every caller before `booster-eval' and
`booster-eval-names'), "booster", or "backend", the noun the two backends'
`%check-object-class' passes for a Layer 1 `create-dataset' or `create-booster' handed the
OTHER backend's object, or any non-backend value. Defaulting to "dataset" means the
two oldest callers, `%check-lightgbm-dataset' and `%check-xgboost-dataset', need no change
of their own to keep reporting exactly what they always have.
```

<a id="cl-gbdt-wrong-backend-reference-argument"></a>

## `cl-gbdt:wrong-backend-reference-argument`

- **Kind** generic function
- **Signature** `(wrong-backend-reference-argument condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:wrong-backend-reference`'s `argument` slot. See `cl-gbdt:wrong-backend-reference`.

<a id="cl-gbdt-wrong-backend-reference-backend"></a>

## `cl-gbdt:wrong-backend-reference-backend`

- **Kind** generic function
- **Signature** `(wrong-backend-reference-backend condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:wrong-backend-reference`'s `backend` slot. See `cl-gbdt:wrong-backend-reference`.

<a id="cl-gbdt-wrong-backend-reference-expected"></a>

## `cl-gbdt:wrong-backend-reference-expected`

- **Kind** generic function
- **Signature** `(wrong-backend-reference-expected condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:wrong-backend-reference`'s `expected` slot. See `cl-gbdt:wrong-backend-reference`.

<a id="cl-gbdt-wrong-backend-reference-given"></a>

## `cl-gbdt:wrong-backend-reference-given`

- **Kind** generic function
- **Signature** `(wrong-backend-reference-given condition)`
- **Exported from** `cl-gbdt`, `cl-gbdt/lightgbm`, `cl-gbdt/xgboost`

Reader of `cl-gbdt:wrong-backend-reference`'s `given` slot. See `cl-gbdt:wrong-backend-reference`.

<a id="cl-gbdt-xgboost-xgboost-backend"></a>

## `cl-gbdt/xgboost:xgboost-backend`

- **Kind** class
- **Superclasses** `backend`
- **Exported from** `cl-gbdt/xgboost`

```text
A connection to the XGBoost shared library, implementing cl-gbdt's
unified backend protocol.
```

### Slots

#### `foreign-library`

```text
The `cffi:foreign-library' `initialize-backend'
loaded, kept so `shutdown-backend' can close exactly this one.
```

