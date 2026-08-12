# cl-gbdt レイヤードAPI再編 方針説明書

## 1. この文書の目的

本書は、`masatoi/cl-gbdt` を今後実装・改修するエージェントに対し、LightGBM と XGBoost の統合方針、公開APIの境界、バックエンド固有機能の扱い、移行手順および受け入れ条件を指示するものである。

実装エージェントは、個別機能を追加する際に本書の規則を優先し、次の二つを同時に満たさなければならない。

1. 両バックエンドで同一の意味を保証できる処理には、安定した共通高水準APIを提供する。
2. 共通化できない固有機能を削除・平坦化・黙殺せず、安全なバックエンド固有APIから利用可能にする。

目標は「LightGBM と XGBoost の全機能を一つの均質なAPIに見せること」ではない。目標は、共通処理の可搬性と固有機能の可用性を、同一ライブラリ内の明確なレイヤーによって両立することである。

## 2. 現状認識

現在の `master` では、LightGBM と XGBoost の両方について、共通protocolの12 generic functionが実装済みである。

- dataset作成、行数・特徴量数取得
- 学習、1 iteration更新
- prediction
- model保存・読込・文字列化
- feature importance
- dataset / booster解放

生成済みCFFI bindingは上流C APIを広く含む一方、現在の高水準backend実装が利用するC関数はその一部である。また、現在の `src/lightgbm/backend.lisp` と `src/xgboost/backend.lisp` は、次の責務を同じ層で担っている。

- shared library探索と初期化
- raw C API呼出し
- C API固有表現のLisp向け変換
- handleの所有権と解放
- 共通generic functionのmethod実装
- バックエンド差分の吸収

この構造は基本機能を実装する段階では合理的だったが、今後固有機能を増やすと、共通APIとbackend固有APIの境界が不明瞭になりやすい。

なお本書の初版作成後、`master` には次の機構が入っている。三層化の設計はこれらを前提としてよい。

- ABI blacklistのbuild時強制 (`tools/ci/check-abi-blacklist.lisp`)。backendが不安定なC関数をimportすると失敗する。解決できないimport名も報告する。
- 上流drift検出 (`tools/check-upstream.lisp`)。**backendがimportする関数だけ**をvendored headerと比較する。header全体の比較では両上流が不安定に見えるが、使用関数に限ればLightGBM v3.0.0-v4.7.0、XGBoost v2.0.0-v3.3.0で破壊的変更は0件である。
- 対応version範囲の記録と `untested-backend-version` 警告 (`src/version.lisp`)。verifiedとinferredを区別して保持する。XGBoostのみruntime照合可能。
- float trap maskingの網羅check (`tools/ci/check-float-traps.lisp`)。
- `src/all.lisp` のre-exportリスト網羅assertion。
- CI version matrix。push-to-masterと週次のみ実行する。

またこの計測は、**共通APIを広げるほど不安定になり、使用関数を安定した部分集合に保つほど可搬性が上がる**ことを示している。三層化はこの性質を構造として固定するものと位置づける。

特に、次の問題を避ける必要がある。

- 共通APIの戻り値に合わせるため、本来のshapeやmetadataを失う。
- 一方のbackendにしかない機能を、他方に存在するように見せる。
- 対応できない引数を黙って無視する。
- backend固有機能を使うために、利用者が生のforeign pointerを扱う。
- 共通APIと固有APIが別々に同じC処理を実装し、修正が二重化する。

## 3. 基本方針

cl-gbdtは、次の三層構造を採用する。

### Layer 0: Raw FFI binding

LightGBM / XGBoostのC APIにほぼ1対1で対応する、生成済みCFFI bindingである。

例:

- `cl-gbdt/src/lightgbm/c-api`
- `cl-gbdt/src/xgboost/c-api`

この層は内部実装であり、安定した公開APIとはしない。

Layer 0では、C APIの関数名、pointer、out parameter、buffer長、error codeなどを原則として原形に近い形で保持する。生成コードを手編集してはならない。変更はbinding generator、vendored header、ABI blacklist等の生成経路を通して行う。

### Layer 1: Backend-specific safe API

LightGBMまたはXGBoost固有の機能を、Common Lisp利用者が安全に呼び出せる形へ変換する層である。

公開packageの候補は次のとおりとする。

- `cl-gbdt/lightgbm`
- `cl-gbdt/xgboost`

ただしASDF system名との衝突やpackage-inferred-system上の依存関係を確認し、必要なら内部packageを以下のように分割してよい。

- `cl-gbdt/src/lightgbm/native`
- `cl-gbdt/src/lightgbm/protocol`
- `cl-gbdt/src/lightgbm/all`
- `cl-gbdt/src/xgboost/native`
- `cl-gbdt/src/xgboost/protocol`
- `cl-gbdt/src/xgboost/all`

公開packageは内部native packageをre-exportしてよいが、raw C API packageをre-exportしてはならない。

Layer 1はC APIの機能粒度に近くてよいが、Cの呼出し規約までそのまま公開してはならない。次をLisp向けに処理すること。

- out parameterを通常の戻り値またはmultiple valuesへ変換する。
- foreign bufferをLisp objectへコピーし、寿命を明確化する。
- error codeを既存condition hierarchyへ変換する。
- raw pointerではなく既存のbackend / dataset / booster handleを受け取る。
- handleの解放済み状態とbackendのopen状態を検証する。
- 所有権を取得したforeign resourceは、あらゆる異常経路で解放する。
- SBCLで必要なfloating-point trap maskingを維持する。
- 上流が返したshape、feature name、metric name等のmetadataを失わない。

Layer 1のAPI例は次のようなものを想定する。

```lisp
(cl-gbdt/xgboost:booster-slice booster :begin 0 :end 50)
(cl-gbdt/xgboost:feature-score booster :kind :total-cover)
(cl-gbdt/xgboost:evaluate-one-iteration booster datasets)

(cl-gbdt/lightgbm:rollback-one-iteration booster)
(cl-gbdt/lightgbm:reset-parameters booster parameters)
(cl-gbdt/lightgbm:refit booster dataset)
```

これらは方向性を示す例であり、上流C API名を機械的にLisp名へ変換するだけでAPIを確定してはならない。戻り値、所有権、condition、shapeを含むcontractを先に定義すること。

### Layer 2: Unified portable API

`cl-gbdt` packageから公開する、LightGBM / XGBoost間で可搬な高水準APIである。現在の `cl-gbdt/src/protocol` がこの層の中心となる。

Layer 2の各methodは、可能な限りLayer 1のbackend-specific safe APIへ委譲する。Layer 2とLayer 1が同じC API処理を別々に実装してはならない。

概念的には次の依存方向とする。

```text
cl-gbdt portable API
        |
        v
backend-specific safe API
        |
        v
generated raw CFFI binding
```

依存方向を逆転させてはならない。特にcore `cl-gbdt` systemは、従来どおり特定backend systemやshared libraryへ依存してはならない。

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

- **Layer 1でのdataset / booster構築、model永続化、metadata照会 — 完了** — 他の三件と違い、これは新機能ではなくLayer 1 / Layer 2分離が残した負債であった。したがって実利用要求を待たず、層分離の次段の最初の項目として着手し、閉じた。両backendとも `src/<backend>/api.lisp` に六つの操作 — `create-dataset`、`create-booster`、`update-one-iteration`、`predict`、`free-dataset`、`free-booster` — を置いたので、`cl-gbdt/lightgbm` あるいは `cl-gbdt/xgboost` だけを読み込んだcallerが、datasetを構築し、その上にboosterを作り、一反復ずつ進め、予測し、両方を解放できる。unified APIがimageに一切存在しない状態でそれが成り立つことは、`tests/functional/lightgbm-standalone.lisp` と `tests/functional/xgboost-standalone.lisp` が示す。両fileが名指しするのは、そのbackendのpublic packageのみであって、本projectの他のsystemは一つも挙げない(`rove` を除けば、ほかに宣言は無い)。`tools/ci/check-leaf-systems.lisp` が個別の新規processで各systemを単独loadするため、この主張は文章ではなくbuildが担保する。§3の「Layer 2の各methodは、可能な限りLayer 1のbackend-specific safe APIへ委譲する」という要求は、この段階でLayer 1に対応物がなかった残り七つの操作 — `save-model`、`load-model`、`model-to-string`、`feature-importance`、`evaluation`、`dataset-num-rows`、`dataset-num-features` — を両backendとも `src/<backend>/api.lisp` へ置いたことで、13 methodすべてが手続き全体をこれらへ委譲する形で果たされた。`train` も例外ではなく、Layer 1に対応する `create-booster` を呼んでboosterを構築したうえで、loopが終わった後にbest iterationだけを `src/handle.lisp` のinternalなwriter `%set-booster-best-iteration` で書く。training report、early stopping、`train` の `:objective` と `:evaluation` はLayer 2固有の概念であって対応するLayer 1操作を持たないので、この委譲には入らない。
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

---

対象リポジトリ: <https://github.com/masatoi/cl-gbdt>

基準commit (2026-08-06 更新): [`59d1979`](https://github.com/masatoi/cl-gbdt/commit/59d1979) (PR #9 merge)
