# cl-gbdt Layered API Restructuring: Implementation Policy

## 1. Purpose of this document

This document instructs the agent that will implement and revise `masatoi/cl-gbdt` from here on, covering the policy for integrating LightGBM and XGBoost, the boundaries of the public API, the handling of backend-specific features, the migration procedure and the acceptance criteria.

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

The generated CFFI bindings cover the upstream C API broadly, while the C functions the current high-level backend implementation uses are a part of it. Also, the current `src/lightgbm/backend.lisp` and `src/xgboost/backend.lisp` carry the following responsibilities in the same layer.

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

## 4. 共通APIへ含める判断基準

機能をLayer 2へ追加してよいのは、次のいずれかを満たす場合である。

1. 両backendで同等の意味とライフサイクルを保証できる。
2. 一方では直接実装、他方では意味を変えない安全なemulationによって実装できる。
3. optional capabilityとして公開する価値があり、未対応時の挙動を型付きconditionで明示できる。

次の条件に該当する場合、無理にLayer 2へ入れてはならない。

- 同じkeywordでもbackendごとに値の意味が異なる。
- 共通形式へ変換するとaxis、shape、metadataまたは精度が失われる。
- 実装のために根拠のない集約、近似、補間が必要になる。
- 一方のbackendで無視する以外に実装方法がない。
- API名だけ共通だが、利用者が期待する効果が異なる。

その場合はLayer 1へ置くか、より情報量の多い共通result typeを設計すること。

## 5. 情報を失わないこと

共通化のために、上流APIが返す情報を黙って捨ててはならない。

特に次を必須規則とする。

- 多次元predictionを `(nrow flattened-width)` に変換する場合、元shapeを復元可能なmetadataも返す。
- multiclass feature importanceを単一値へ集約する場合、上流が定義した集約であることを証明する。独自集約は禁止する。
- XGBoost固有のimportance種別を、意味の違う共通種別へ読み替えない。
- single-floatからdouble-floatへの変換は精度を増やさないため、変換理由とコストをcontractに明示する。
- model文字列表現はopaqueな文字列として扱い、backend間で同一formatであると仮定しない。

既存 `predict` の戻り値contractを直ちに破壊してはならない。shape保持が必要な場合は、以下のいずれかを設計レビューで選択する。

- 既存 `predict` を維持し、shapeを保持する新APIを追加する。
- primary valueを既存配列、secondary valueをshape metadataとする。
- prediction result objectを追加し、段階的に移行する。

選択に際しては後方互換性を優先する。

## 6. Parameterの扱い

学習時の `:parameters` は、backend固有設定を失わないescape hatchとして維持する。すべてのparameterを共通語彙へ翻訳しようとしてはならない。

ただし、次を明確に区別すること。

- portable parameter: 両backendで同じ意味を保証できるもの。
- backend parameter: backendへ透過的に渡すが、可搬性を保証しないもの。
- dataset construction option: C API呼出し形式自体を制御するもの。

XGBoostの `missing`、`nthread`、`data_split_mode` のように、backend側で有効なdataset optionまで一括して拒否してはならない。認識可能なkeyのみを受理し、未知または意味の異なるkeyを `unsupported-argument` で拒否する方式を検討する。

引数を黙って無視すること、およびC libraryが未知keyを黙殺することに依存する実装は禁止する。

## 7. Capability model

現在予約されている `backend-capabilities` を実装し、少なくとも次の問い合わせを可能にする。

```lisp
(backend-supports-p backend :sparse-input)
(backend-supports-p backend :evaluation-history)
(backend-supports-p backend :early-stopping)
(backend-supports-p backend :model-slicing)
(backend-supports-p backend :multidimensional-feature-score)
```

capabilityは単なるbackend名による固定表ではなく、必要に応じて次を反映する。

- backend種別
- runtime version
- 実際に解決できたforeign symbol
- build optionやplatform依存機能

ただしcapability queryは実際の呼出し時検証の代替ではない。機能呼出し側も、未対応の場合は必ず型付きconditionを送出すること。capabilityが偽の場合に黙って別機能へfallbackしてはならない。

## 8. Library availabilityとsymbol probe

backendの初期化時には、その実装が必須とする全foreign symbolを検査する。

現在、XGBoost backendの `*required-symbols*` には **2件** の漏れがある。本書初版は `XGDMatrixSetUIntInfo` のみを挙げていたが、機械的に照合すると次のとおりである。

```
lightgbm: import 18 / required 18 / 漏れ 0
xgboost:  import 20 / required 18 / 漏れ 2
    XGDMatrixSetUIntInfo      (ranking group設定)
    XGBoosterGetNumFeature    (feature importanceの密ベクトル化で使用)
```

本書自身が §8 で要求している機械的checkを、本書を書いた時点では持っていなかったために1件取りこぼしている。この不整合を最初に修正すること。そしてこの取りこぼし自体が、check追加の必要性を示す証拠である。

さらに再発防止として、backend sourceが参照するforeign callと `*required-symbols*` の対応を検証する機械的checkを追加する。完全なcall graph解析が困難な場合でも、少なくとも以下を検査する。

- backend固有sourceから呼ばれるraw C functionがrequiredまたはoptional capability symbolとして分類されている。
- required symbol欠落時は `open-backend` が `missing-foreign-symbols` を送出する。
- optional symbol欠落時はbackend全体を開けなくするのではなく、該当capabilityだけを無効にする。

LightGBMはruntime version APIを持たないため、versionを推測して保証してはならない。この非対称性は `src/version.lisp` と READMEに明記済みである。

XGBoost versionのテスト済みversion警告への接続は **実装済み** である (`untested-backend-version`)。残る課題はcapability判断への接続であり、§7 と併せて設計すること。

## 9. Evaluationとearly stopping

次に優先すべき利用者向け機能は、validation metric、evaluation history、best iteration、early stoppingである。

現在の `valid-sets` はforeign backendへdatasetを登録またはcacheするが、共通APIから評価値を取得できない。これを完成させる。

後方互換性を維持する案として、`train` のprimary valueは従来どおりboosterとし、secondary valueにtraining reportを返す方式を第一候補とする。

```lisp
(multiple-value-bind (booster report)
    (train backend dataset
           :valid-sets valid-sets
           :num-rounds 1000
           :early-stopping-rounds 30)
  ...)
```

training reportは少なくとも次を表現できるものとする。

- dataset名
- metric名
- iterationごとの値
- best iteration
- best score
- early stoppingが発生したか

metric方向の判定は文字列名から推測せず、backendが提供する情報または明示的な利用者指定に基づくこと。callback APIを追加する場合も、backend固有callbackを無理に同一function signatureへ変換せず、portable event objectと固有extensionを分ける。

## 10. Resource safety

Layer 1を公開しても、現在のresource safetyを弱めてはならない。

- dataset / boosterは既存handle hierarchyを使用する。
- boosterがtraining / validation datasetを必要とする間はstrong referenceを保持する。
- backendを閉じた後にforeign callを行わない。
- 二重解放はno-opとする。
- backend close後の解放不能resourceは警告し、Lisp側handleをrelease済みにする。
- finalizerは安全網であり、foreign freeの主経路にしない。
- 新しいforeign resourceには `with-*` macroまたは同等の明示的所有権APIを用意する。
- cleanup中のerrorが元のconditionを上書きしないようにする。

raw pointerを返すescape hatchは原則として追加しない。どうしても必要な場合は、非安定・非公開packageに限定し、pointerの寿命をdynamic extentに制限するcallback形式を優先する。

## 11. Packageとsystemの境界

次を維持する。

- `cl-gbdt` coreはbackend shared libraryなしでloadできる。
- `cl-gbdt/lightgbm` と `cl-gbdt/xgboost` は独立してloadできる。
- 一方のbackend systemが他方のbackend systemへ依存しない。
- raw C API symbolsを `cl-gbdt` packageからre-exportしない。
- backend固有の公開symbolは `cl-gbdt/lightgbm` または `cl-gbdt/xgboost` からのみ公開する。
- package-inferred-systemの各leafが単独loadできる依存宣言を維持する。

共通API methodの配置を分離する場合でも、methodをloadした結果だけに依存する暗黙の順序を作らないこと。既存のleaf-system checkを新しいsource構造へ追従させる。

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
