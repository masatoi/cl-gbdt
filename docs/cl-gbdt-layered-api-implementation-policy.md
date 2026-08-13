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

The dependency direction must not be reversed. In particular, the core `cl-gbdt` system must not depend on a specific backend system or on a shared library, as before.

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
- The primary value shall be the existing array and the secondary value shape metadata.
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

## 12. 実装の段階計画

### Phase 0: 既知のavailability不整合を修正

1. XGBoost `*required-symbols*` に `XGDMatrixSetUIntInfo` と `XGBoosterGetNumFeature` を追加する。
2. required symbol網羅checkを追加する。`tools/ci/check-abi-blacklist.lisp` が既に「backendのimport-from句を読み、c-api.lisp のdefcfunでC名へ写す」処理を持つので、その照合対象を `*required-symbols*` に向けるだけでよい。新規機構は不要である。
3. checkが落ちるところを確認してから採用する。本repositoryは落ちるところを見ていないcheckを二度出荷している。

README内のassertion件数のdocument driftは **解消済み** (243 / 106 で現状と一致) のため、本phaseの対象外とする。

このphaseではAPI構造を変更しない。

### Phase 1: 責務分離を行うが挙動を変えない

1. raw callを安全化するbackend固有functionを抽出する。
2. 既存の共通generic methodを、そのfunctionへ委譲させる。
3. 既存public API、戻り値、condition、テスト結果を維持する。
4. LightGBMとXGBoostの両方で同じ分離原則を適用する。

このphaseの目的は新機能ではなく、実装経路を一本化した三層構造を確立することである。

### Phase 2: Backend-specific safe APIを公開

1. 公開対象functionのcontractを定義する。
2. backend固有packageからexportする。
3. raw pointerを公開せず、既存handleを使用する。
4. documentationとbackend固有functional testを追加する。
5. capability modelを実装する。

最初の公開対象は、共通化によって現在利用できないことが明確な機能から選ぶ。

- XGBoost model slicing
- shapeを保持するXGBoost feature score
- XGBoost evaluation
- LightGBM evaluation
- LightGBM rollback / refit / reset parameter

### Phase 3: 共通training APIを完成

1. evaluation historyを取得する。
2. training reportを設計する。
3. early stoppingを実装する。
4. best iterationをprediction / persistenceと接続する。
5. validation set namingを定義する。

### Phase 4: Dataとpredictionの拡張 — 完了

優先度と実利用要求に応じて、以下をcapability-gated APIとして追加する。

- sparse input
- missing value option
- categorical metadata
- multidimensional prediction result
- custom objective / evaluation

すべてを同時に実装しようとしてはならない。各機能について、Layer 1 contract、Layer 2に含める可否、capability、functional testを一組として実装する。

この一組という単位は最後まで守られ、一覧の各項目がそれぞれ独立したPRとして入った。external memoryは下記の理由で一覧から外したため、**この時点でPhase 4に未実装の項目はない**。

| 一覧の項目 | 実装 | capability | PR | functional test |
|---|---|---|---|---|
| sparse input | dense matrixを受ける位置で `make-dataset` と `predict` が `csr-matrix` も受ける | `:sparse-input` (両backend) | #18 | `tests/functional/sparse-input.lisp` |
| missing value option | `make-dataset` と `predict` の `:missing` | `:missing-value` (XGBoostのみ真) | #19 | `tests/functional/missing-value.lisp` |
| categorical metadata | `make-dataset` の `:categorical-features` | `:categorical-features` (両backend) | #20 | `tests/functional/categorical-features.lisp` |
| multidimensional prediction result | `predict` が第二値として返すshape | `:prediction-shape` (両backend) | #22 | `tests/functional/prediction-shape.lisp` |
| custom objective / evaluation | `train` の `:objective` と `:evaluation` | `:custom-objective` / `:custom-evaluation` (両backend) | #23 / #24 | `tests/functional/custom-objective.lisp` / `custom-evaluation.lisp` |

以後この一覧へ項目を足さない。新しいdata / prediction機能は、phaseの続きではなく下記のフォローアップとして扱う。

### external memoryを一覧から外した理由

当初この一覧にはexternal memoryも含めていたが、vendoredヘッダを読んだ結果、**両backendとも、このwrapperがexternal memoryと呼べるものを、対応するdataset構築の入口では提供していない**ことが分かったため外した。同じ調査を繰り返さないよう、根拠を残す。

- **XGBoost**のexternal memoryは`Streaming`グループにある。ヘッダ自身が "the experimental external-memory-based DMatrix, which reads data in batches during training" と書き、到達するには`XGDMatrixCreateFromCallback`または`XGExtMemQuantileDMatrixCreateFromCallback`に、呼び出し側が**Cコールバックとして実装したdata iterator**（`XGDMatrixCallbackNext`、`DataIterResetCallback`）を渡す必要がある。`XGDMatrixCreateFromURI`は "load a data matrix" であって、通常のin-memory DMatrixを作る。
- **LightGBM**には該当する機能がない。`LGBM_DatasetCreateFromFile`は "Load dataset from file (like LightGBM CLI version does)" で、できあがる`Dataset`はbin化された形で全量メモリに載る。`LGBM_DatasetInitStreaming`とその周辺は、複数スレッドから行を流し込む**構築**の機構であって、データをメモリ外に置く機構ではない。

したがって実装するとすれば、このプロジェクト初のC→Lisp callbackを導入し、上流自身がexperimentalと呼ぶAPIに乗り、片側のbackendでのみ真となるcapabilityを足すことになる。§7が要求するportable contractに載せられる形ではない。将来この判断を覆すなら、上記の三点が変わったことを先に確認すること。

### フォローアップ

Phase 4完了時点、およびその後のLayer 1 / Layer 2分離作業で判明している残件を記録する。いずれもどのphaseの完了条件でもない。まとめて一つのphaseに束ねない。二件目以降は、実利用要求が出た時点で、§17の分類と§13のテスト方針に従い一件ずつ着手する。

- **Layer 1でのdataset / booster構築、model永続化、metadata照会 — 完了** — 他の三件と違い、これは新機能ではなくLayer 1 / Layer 2分離が残した負債であった。したがって実利用要求を待たず、層分離の次段の最初の項目として着手し、閉じた。両backendとも `src/<backend>/api.lisp` に六つの操作 — `create-dataset`、`create-booster`、`update-one-iteration`、`predict`、`free-dataset`、`free-booster` — を置いたので、`cl-gbdt/lightgbm` あるいは `cl-gbdt/xgboost` だけを読み込んだcallerが、datasetを構築し、その上にboosterを作り、一反復ずつ進め、予測し、両方を解放できる。unified APIがimageに一切存在しない状態でそれが成り立つことは、`tests/functional/lightgbm-standalone.lisp` と `tests/functional/xgboost-standalone.lisp` が示す。両fileが名指しするのは、そのbackendのpublic packageのみであって、本projectの他のsystemは一つも挙げない(`rove` を除けば、ほかに宣言は無い)。`tools/ci/check-leaf-systems.lisp` が個別の新規processで各systemを単独loadするため、この主張は文章ではなくbuildが担保する。§3の「Layer 2の各methodは、可能な限りLayer 1のbackend-specific safe APIへ委譲する」という要求は、この段階でLayer 1に対応物がなかった残り七つの操作 — `save-model`、`load-model`、`model-to-string`、`feature-importance`、`evaluation`、`dataset-num-rows`、`dataset-num-features` — を両backendとも `src/<backend>/api.lisp` へ置いたことで、13 methodのうち `train` を除く十二すべてが手続き全体をこれらへ委譲する形で果たされた。`train` は構築だけの委譲で、Layer 1に対応する `create-booster` を呼んでboosterを構築するが、loop自体は `api.lisp` の `update-one-iteration` を経由せず `native.lisp` の関数を直接呼び続ける — 毎反復ごとに全handleを再検査させないためで、その理由は両backendの `train` それぞれの `create-booster` 呼び出し箇所のコメントに記してある。loopが終わった後にbest iterationだけを `src/handle.lisp` のinternalなwriter `%set-booster-best-iteration` で書く。training report、early stopping、`train` の `:objective` と `:evaluation` はLayer 2固有の概念であって対応するLayer 1操作を持たないので、この委譲には入らない。
- **file input** — `make-dataset` の `MATRIX` がpathnameも受け、libraryが自らファイルを読む形 (`LGBM_DatasetCreateFromFile`、`XGDMatrixCreateFromURI`)。どちらも通常のin-memory datasetを作るので、これはexternal memoryではなく、別capability `:file-input` になる。設計のみ済み、未実装。
- **shapeを保持するXGBoost feature score** — Phase 2の「最初の公開対象」に挙げたまま未実装。`:multidimensional-feature-score` は `*known-capabilities*` に登録済みだが全backendでfalseであり、「未対応であること自体は答えられる」状態で止まっている。
- **LightGBM rollback / refit / reset parameter** — 同じくPhase 2の一覧の未実装項目。`LGBM_BoosterRollbackOneIter`、`LGBM_BoosterRefit`、`LGBM_BoosterResetParameter` はbindingには存在し、Layer 1として公開していない。

## 13. テスト方針

テストを次の二種類に明確に分ける。

### Portable contract tests

同一のテストを両backendに適用し、共通APIの意味が一致することを検証する。

最低限、次を実ライブラリで検証する。

- label / weight / group / feature names
- normal / raw / leaf-index / contribution prediction
- multiclass shape
- `num-iteration`
- save / load / model-to-string
- split / gain importance
- validation metric
- early stopping
- released handle / closed backend / wrong backend handle
- unsupported capabilityが型付きconditionになること

数値がbackend間で完全一致することは要求しない。shape、順序、意味、単調性、再読込後の同一backend内再現性など、contractに対応した性質を検証する。

### Backend-specific tests

固有APIについて、上流C APIの特徴を失わず返すことを検証する。

- XGBoost multidimensional shape
- XGBoost固有importance種別
- model slicing
- LightGBM reference dataset
- refit / rollback
- optional symbol欠落時のcapability低下

新しい公開functionには、mockだけでなく実shared libraryを呼ぶfunctional testを必須とする。

## 14. 後方互換性

既存利用者を壊す変更を、一括して導入してはならない。

- 既存generic function名とprimary return valueを維持する。
- conditionを単なる `error` や実装依存CFFI errorへ退化させない。
- 既存keywordを別の意味へ変更しない。
- 戻り値shapeを変更する場合は新APIまたは段階的移行を用いる。
- backend固有APIの追加を理由に、共通APIを削除しない。
- raw C API packageを公開安定APIとして約束しない。

破壊的変更が不可避な場合は、設計文書、移行例、deprecation期間を先に提示し、別PRとして扱う。

## 15. 非目標

次は本方針の目標ではない。

- LightGBMとXGBoostの全parameterを共通名へ翻訳すること。
- 両backendの学習結果を数値的に一致させること。
- 上流C APIの全関数を直ちに高水準公開すること。
- 生CFFI APIを一般利用者向けの安定APIにすること。
- 一方のbackendにしかない概念を他方で擬似的に再現すること。
- 単一PRで全レイヤーを全面改修すること。

## 16. 完了条件

レイヤードAPI再編の初期完了は、次をすべて満たした時点とする。

1. raw FFI、backend-specific safe API、unified APIの依存方向がsource構造上明確である。
2. 既存共通generic methodがbackend-specific safe APIへ委譲している。
3. 共通APIと固有APIが同じC処理を重複実装していない。
4. backend固有packageから、少なくとも一つの固有機能をraw pointerなしで利用できる。
5. `backend-capabilities` が実際の機能可用性を返す。
6. required / optional foreign symbolの分類と機械的検査が存在する。
7. 既存layer 1 / layer 2 testがすべて成功する。
8. 新しい固有APIに実shared libraryを使うfunctional testがある。
9. core systemがbackend libraryなしでloadできる性質を維持する。
10. 公開APIとpackage境界がREADMEまたは専用documentに記載されている。

## 17. 実装エージェントへの最終指示

実装を始める前に、変更対象の機能を次のいずれかへ分類すること。

- raw FFI concern
- backend-specific safe API
- unified portable API
- optional capability

分類できないまま共通genericへkeywordを追加してはならない。

また、実装案を提示する際は必ず以下を説明すること。

1. その機能をどのLayerへ置くか。
2. 両backendで意味が一致するか。
3. shapeやmetadataを失わないか。
4. 未対応backend/versionでどのconditionを送出するか。
5. resource ownershipを誰が持つか。
6. 既存APIからどのように委譲するか。
7. どのfunctional testでcontractを証明するか。

迷った場合は、共通APIへ入れて情報を減らすより、backend-specific safe APIとして完全な情報を保持する方を選ぶこと。その後、実利用例と両backendの意味を確認してから共通APIへ昇格させる。

## 18. Layer 1 standalone-library化プログラム (S1〜S5)

2026-08-11以降、`cl-gbdt/lightgbm` と `cl-gbdt/xgboost` を、統合API
(`cl-gbdt/lightgbm/unified`、`cl-gbdt/xgboost/unified`) を一切loadしなくても実用に足る独立した
libraryにする、S1からS5までの五段階のプログラムが進行している。この定義はこれまで
`docs/superpowers/specs/2026-08-11-layer1-standalone-design.md` にのみ存在し、同ディレクトリは
`.gitignore` の対象であるため、trackedな文書には一度も記録されていなかった。本節がその記録である。

このプログラムは§12の段階計画 (Phase 0〜4) とは別系統であり、Phase 4完了後に始まった。Phase 2は
「backend固有packageからexportする」ことを、Phase 4は「data / predictionの拡張」を主題としたのに
対し、このプログラムはLayer 1が統合APIなしで単体のlibraryとして自足していることを主題とする。§12
のPhase一覧には項目を足さない。

プログラム全体を拘束する決定は次の五つである。

1. coverageはclassification (分類の網羅) で保証し、percentageでは保証しない。
2. `cl-gbdt/<backend>` はLayer 1のみを意味する。統合APIの13 methodは
   `cl-gbdt/<backend>/unified` が担う。
3. one C function, one Lisp functionとする。C呼出し規約は変換し (out parameterを戻り値または
   multiple valuesへ、foreign bufferをLisp objectへ、error codeを型付きconditionへ)、C APIが
   initとfreeを対で提供する箇所には `with-*` macroも用意する。
4. docstringを文書の一次情報源とする。API referenceはdocstringから生成し、`src/*/c-api.lisp` と
   同様byte-for-byteで検査する。
5. 公開symbolには機械的に検査されるfunctional testを要求する。

各段階の現状は次のとおりである。

- **S1 — 層の分離。完了 (PR #26)。** concrete classとlibrary lifecycle
  (`initialize-backend`/`shutdown-backend`) をLayer 1へ移し、`cl-gbdt/<backend>` をLayer 1のみの
  systemとし、所有権パターンを `with-pointer-ownership` 一つへ集約した。
- **S2 — 各操作の手続きをLayer 1へ移す。完了 (PR #27、#29、#30)。** PR #27で両backend各6操作
  (`create-dataset`、`create-booster`、`update-one-iteration`、`predict`、`free-dataset`、
  `free-booster`)、PR #29で残り7操作 (`save-model`、`load-model`、`model-to-string`、
  `feature-importance`、`evaluation`、`dataset-num-rows`、`dataset-num-features`)、PR #30で
  `train` のbooster構築を、それぞれ両backendの `src/<backend>/api.lisp` へ移した。結果、13の
  unified methodはすべて、その手続きの少なくとも一部をLayer 1へ委譲する。委譲の程度が `train` の
  みほかの12と異なる理由 (boosting loop自体は `update-one-iteration` を経由しない) は既に上記の
  フォローアップに記録済みであり、ここでは繰り返さない。
- **S3 — bindingのclassification。完了 (PR #31)。** `src/*/c-api.lisp` が生成する177 bindingすべてを
  `ffi-spec/BINDING-COVERAGE.md` で `wrapped`/`planned`/`excluded` のいずれかに分類し、
  `tools/ci/check-binding-coverage.lisp` が未分類のbindingでbuildを失敗させる。
- **S4-1 — docstringからAPI referenceを生成しbyte-for-byteで検査する仕組み。完了 (PR #33)。**
  loadされたimageをintrospectしてMarkdownを書き出すdevelopment-onlyのemitter `src/docgen/`
  (`introspect.lisp`、`render.lisp`、`emit.lisp`、`all.lisp`) をASDF system `cl-gbdt/docgen`
  として追加し、そのdriverである `tools/gen-api-reference.lisp` が、`cl-gbdt`・
  `cl-gbdt/lightgbm`・`cl-gbdt/xgboost` の三つの公開packageがexportする174 symbol
  (141/88/89) すべてを覆う `docs/API-REFERENCE.md` を生成した。`tools/ci/check-api-reference.lisp`
  は四段階で検査する: introspection primitiveの存在、生成結果とcommit済みfileの
  byte-for-byte一致、公開symbol全てへのdocumentation floor (class/conditionのslotを含む)、
  package毎のexport数floor。このdocumentation floorを満たすため、`src/conditions.lisp` の
  condition slot十二個に `:documentation` を追加した。
- **S4-2 — 公開symbolすべてにfunctional testが存在することを機械的に検査する仕組み。完了。**
  `docs/FUNCTIONAL-COVERAGE.md` が、`cl-gbdt`・`cl-gbdt/lightgbm`・`cl-gbdt/xgboost` の三つの
  公開packageがexportする174 symbolすべてに位置を与える。`covered`はfileに書き下さない —
  `tools/ci/check-functional-coverage.lisp` が `tests/functional/*.lisp` の各fileを読み、
  package-formを除くtop-level formにsymbol名が現れれば`covered`とその都度導出する。これは
  `ffi-spec/BINDING-COVERAGE.md` の `wrapped` と同じく、二重管理される記録を持たないための設計
  である。残るsymbolは、functional testを書くべきだが書かれていない `## Unproven`(29件)か、
  理由ごとに独立した `## Exempt` 見出し群(58件)かのいずれかへ人手で分類する。checkerは
  `covered`の下限(`+minimum-covered+` = 87)と`unproven`の上限(`+maximum-unproven+` = 29)を
  両方検査し、後者がratchetとして働く: functional testを持たない公開symbolが増えれば
  `unproven`が29を超え、この定数を `tools/ci/check-functional-coverage.lisp` 内で引き上げる
  編集をしない限りbuildが失敗する。ただし
  この機構が保証するのはsymbolごとの「記録された位置」であり、「証明されたcontract」ではない。
  `## Exempt`見出しは、literalな `## Unproven` と、`## Exempt` で始まる任意の見出しという前方
  一致でしか認識されないため、新しい `## Exempt: ...` 見出しを立てるか、既存の五つの `## Exempt`
  見出しのいずれかへ行を追加するだけでも、`covered` も `unproven`
  も動かさずに未testのsymbolを収められる — この機構が実際に防いでいるのは「無分類のまま公開
  されること」であり、各 `## Exempt` の理由が正当かどうかは最終的にreviewerが文章を読んで判断
  する。分類作業そのものが、174 symbol中29件にfunctional testが存在しないという作業量を明らか
  にした。
- **S5 — 未着手。** まだ誰も公開していないC functionを公開する。作業一覧は
  `ffi-spec/BINDING-COVERAGE.md` の `## Planned` 節である。同節にはLightGBMの
  `LGBM_BoosterRollbackOneIter`/`LGBM_BoosterRefit`/`LGBM_BoosterResetParameter`、および両
  backendのfile input (`LGBM_DatasetCreateFromFile`/`XGDMatrixCreateFromURI`) の行があり、これら
  は上記のフォローアップが既に個別に記録している未実装項目と同じものを指す。フォローアップの
  「shapeを保持するXGBoost feature score」は別である。`XGBoosterFeatureScore` 自体は既にwrapped
  済みであり、この項目は既存bindingの戻し方を変える実装課題であって、S5が公開すべきbindingの一覧
  には現れない。

---

対象リポジトリ: <https://github.com/masatoi/cl-gbdt>

基準commit (2026-08-06 更新): [`59d1979`](https://github.com/masatoi/cl-gbdt/commit/59d1979) (PR #9 merge)
