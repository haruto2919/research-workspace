---
project: sequential-video-lora-analysis
spec_type: implementation
status: draft
title: 50Salads時系列逐次LoRA fine-tuning基盤
created: 2026-09-09
last_updated: 2026-09-09
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
workspace_base_commit: 8f60ff667b7e846620eec9208c078cab2efa0c72
implementation_repository: tamaki-lab/2026_04_ishikawa_simple-MeMViT
implementation_base_branch: main
implementation_base_commit: e0deb093694d367ed9b02065e6d4cd38802093d6
loader_repository: tamaki-lab/sequential_loader
loader_base_branch: master
loader_base_commit: cef09aa12560127451a5f569d86d5d51671e6986
---

# 50Salads時系列逐次LoRA fine-tuning基盤 spec

> **Status: draft**
>
> 本specは内容確認中であり、研究コードの実装、config変更、test追加、launcher変更、
> dependency追加、学習・評価runを許可しない。`approved` へ変更されるまで実装に使用しない。

## 1. 目的

第一段階として、50Saladsを時系列順に逐次入力し、Kinetics-400 pretrained MeMViTを
backboneとして、全16 blockのattention `q` / `v` LoRAと新規51-class frame-level headだけを
学習できる、再現可能なsequential fine-tuning基盤を定義する。

この基盤は、後続研究で「LoRAに動画由来の時間情報・動作情報が蓄積されるか」を解析・評価する
ための前提を作る。LoRAが有意な時間・動作情報を獲得したこと自体は、本specの成功条件ではない。

## 2. Authorityと基準revision

### 2.1 Authority

本specの判断根拠は、優先順に次のとおりとする。

1. 2026-09-09のユーザー指示と、同指示に列挙された確定事項・技術検証Evidence
2. [`2026-09-03-mtg.md`](../meetings/2026-09-03-mtg.md) の決定事項
3. [`README.md`](../README.md) の研究目的と現在方針
4. 下記基準revisionの現行コード・public contract

Evidence A〜OはユーザーがChatGPT上の手順をローカル実行し、出力を確認済みと報告した結果である。
本spec作成時には研究コード変更、dataset scan、model smoke、学習runを再実行していない。

### 2.2 固定するrepositoryとrevision

| 役割 | repository | branch | 基準commit |
|---|---|---|---|
| 研究文脈SSOT | `haruto2919/research-workspace` | `main` | `8f60ff667b7e846620eec9208c078cab2efa0c72` |
| 実装対象 | `tamaki-lab/2026_04_ishikawa_simple-MeMViT` | `main` | `e0deb093694d367ed9b02065e6d4cd38802093d6` |
| 外部loader | `tamaki-lab/sequential_loader` | default branch `master` | `cef09aa12560127451a5f569d86d5d51671e6986` |

実装開始時には各remoteの先端を再確認する。基準commitから進んでいる場合は、差分が本specの前提・
interface・成功条件を変えないことをread-onlyで確認してから実装する。別branchやローカルdirty stateを
権威ある入力として使用しない。

### 2.3 現行main調査で確認したパス差分

- 実装対象のGitHub `main` には `AGENTS.md` が存在しない。ローカル作業ツリーにだけある未追跡
  `AGENTS.md` は本specの根拠に使用していない。
- 指定された `model/memvit/mvit_classification_only.py` はGitHub `main` に存在しない。
  現行のclassification-only MeMViT実装は `model/memvit/memvit_model.py` であり、ファイル先頭に
  旧ファイル名のコメントが残っている。本specでは現行パスを基準にする。

## 3. Decision Contract

### 3.1 採用する学習条件

| 項目 | 採用内容 |
|---|---|
| dataset | 50Salads |
| task | fine 51-class frame-level classification |
| chunk | 16 frames、sampling interval 1、non-overlapping |
| initial baseline | `num_epochs = 1` のstrict online one-pass |
| implementation | `num_epochs > 1` にも対応し、各epoch内の順序を維持 |
| model | MeMViT、K400 pretrained lineageを維持 |
| trainable parameters | 全16 blockのattention `q` / `v` LoRAと新規51-class headのみ |
| causality | strict frame-level causal |
| optimization | Lightning Manual Optimization、教師ありchunkごとに1 step |
| optimizer | AdamW、learning rate `1e-4`、weight decay `0.01` |
| scheduler | なし |
| baseline seed | 既存50Salads configの `RNG_SEED = 0` を維持 |
| confirmed environment | `torch == 2.14.0+cu130`、`torchvision == 0.29.0+cu130`、`lightning == 2.6.5`、`peft == 0.20.0` |
| numerical precision | Lightning `precision = "32-true"` |
| distributed | 対象外。strict pathはsingle process / single GPU |
| resume | epoch境界のみ保証 |

### 3.2 明示的な対象外

本specでは次を実装・実験しない。

- full training run、およびその結果を使った科学的結論
- multi-seed experiment
- hyperparameter tuning
- LoRA rank ablation
- optimizer比較、OrthogonalAdamW
- LoRA成分解析、静的/動的情報分離、LoRA orthogonalization、LoRA空間変換
- VLM downstream評価、動画生成評価
- mid-sequence resume
- distributed / multi-GPUでのsequential state同期
- sequence-consistent random augmentationの新設
- external `sequential_loader` のpublic contract変更

## 4. 現行mainと要求状態の差分

| Component | 基準mainの状態 | 本specで必要な状態 |
|---|---|---|
| 50Salads入力 | 実装repository内のlegacy sequential datasetを使用 | `tamaki-lab/sequential_loader` のpublic APIをConsumerとして使用 |
| legacy annotation | indexed labelを `frame_id - 1` で配置し、backgroundを追加 | direct index対応、fineなしはignore、background classなし |
| preprocessing | sequential trainでtemporal subsample、ImageNet mean/std、random scale/crop/flip | temporal resamplingなし、K400 mean/std、決定的resize/center crop |
| PatchEmbed | `Conv3d(kernel_t=3, padding_t=1)` のsymmetric temporal convolution | pretrained weightを維持したstateful causal temporal convolution |
| attention memory | `MemViT.clear_memory()` がattention cacheをreset | attention cacheに加えてPatchEmbed historyもreset |
| optimization | Lightning automatic optimization | Manual Optimization |
| zero-supervision chunk | zero lossでも通常のautomatic step対象になり得る | forwardのみでoptimizer/global optimization stepを進めない |
| scheduler | scheduler無効時もdummy `ConstantLR`を返す | scheduler object/stepなし |
| LoRA | PEFT依存・LoRA構成なし | PEFT q/v LoRAを明示構成 |
| checkpoint load | configにK400 pathはあるが、mainのmodel構築経路にload処理がない | 検証済み互換loadと新規head skipを実装 |
| eval output | 現行headはeval時にsoftmaxを適用 | lossへ渡す値はtrain/valともraw logits |
| tests | dataloader確認用scriptのみ | 下記success criteriaに対応する短時間test/smoke |

legacy 50Salads datasetのlabel parsing、repeat-last tail処理、internal collate contractは、外部loaderを
使う本経路の仕様として再利用しない。

## 5. データ契約

### 5.1 Canonical split

論理splitと固定sequence順は次のとおりとする。

```text
train (14 sequences)
01-1, 01-2,
02-1, 02-2,
03-1, 03-2,
04-1, 04-2,
05-1, 05-2,
06-1, 06-2,
07-1, 07-2

val (4 sequences)
09-1, 09-2, 10-1, 10-2
```

物理splitは次のように扱う。

- trainはphysical `train` だけからcanonical 14 sequencesを選ぶ。
- physical `train1` は使わない。特に `01-1` の重複sourceを混入させない。
- valはphysical `val` と `val1` のsourceを論理的に合成し、上記canonical順へ並べる。
- physical splitごとに `Salads50Adapter.sequence_sources(split)` を呼び、返された
  `SequenceSource` を `sequence_id` で選択・合成する。
- canonical対象の欠落、canonical対象内の重複、train/val overlap、video欠落、annotation欠落は
  fail-fastとする。canonical外のphysical sourceは本経路では選択しない。
- annotation参照は、各sourceを発見したphysical splitに対応するものを保持する。

実装時のpreflightで、少なくとも次を確認する。

```text
train count = 14
val count = 4
duplicate sequence_id = 0
train/val overlap = 0
missing video = 0
missing annotation = 0
```

### 5.2 External sequential loader契約

`tamaki-lab/sequential_loader@cef09aa12560127451a5f569d86d5d51671e6986` のtop-level
`sequential_loader` packageだけをConsumerから参照する。内部 `src.*` moduleへ依存しない。

strict pathは次を満たす。

```text
DataLoaderConfig.strict = True
batch_size = 1
shuffle = False
num_workers = 0
drop_last = False
in_order = True
SequenceOrderingConfig.shuffle_sequences = False
collate = unbatched SequentialSample passthrough
```

`SequentialSample` の主要契約は次のとおりである。

| field | type / shape |
|---|---|
| `frames` | CPU `torch.uint8 [T,3,H,W]` |
| `frame_indices` | `torch.int64 [T]` |
| `timestamps` | `torch.float64 [T]` |
| `valid_mask` | `torch.bool [T]` |
| lifecycle | `sequence_id`, `sequence_index`, `is_first`, `is_last` |
| metadata | `source_id`, `source_metadata`, `dataset_metadata`, `evaluation_reference` |

tail paddingはloader標準をそのまま使う。

```text
frames = zero
frame_indices = -1
timestamps = NaN
valid_mask = False
```

Consumer側でrepeat-last補正を導入しない。Loaderはannotationをparseせず、
`evaluation_reference` / metadata内のannotation pathをopaque referenceとして渡す。
fine label parsingとalignment検証はConsumer側の責務とする。

### 5.3 Frameとannotationのalignment

- validなdecoded video frame index `i` はannotation frame ID `i` と直接対応させる。
- `-1` shiftを行わない。
- annotationをframe IDで索引化し、同じIDの重複はfail-fastとする。
- video実体に存在しないannotation末尾のextra rowは、対応するdecoded frameがないため使用しない。
- loader padding positionはannotation lookupを行わない。

annotation rowのfine supervisionは次のように解釈する。

| row | 解釈 |
|---|---|
| 3列: `frame_id coarse fine` | `fine` を教師labelにする |
| 2列: `frame_id coarse` | fine taskではignore |
| 1列: `frame_id` | no labelとしてignore |
| padding | ignore |

targetは `torch.int64 [T]` とし、fine supervisionがない位置は `ignore_index = -100` とする。
loss / metricのeffective maskは常に次で定義する。

```text
effective_supervision_mask = sample.valid_mask AND fine_label_available
```

canonical train+val annotationから得たfine label文字列のunique集合を辞書順に並べ、`0..50` を
割り当てる。background classは作らない。実装時にmappingをrun artifactへ保存し、unique数が51で
なければfail-fastとする。ユーザー確認済み統計は次のとおりである。

```text
unique fine = 51
fine supervised entries = 199802
coarse only = 11151
no label = 52025
```

## 6. Input preprocessing

初期baselineではtrain / valともrandom augmentationを使わず、同一frameへ同一transformを適用する。
chunk内・chunk間で座標系を変えない。

処理順を次に固定する。

1. `uint8` から `float32` へ変換
2. `/ 255`
3. bilinear interpolation、`antialias = True` でshort sideを256へresize
4. 224 x 224 center crop
5. K400 mean/stdでnormalize
   - mean = `[0.45, 0.45, 0.45]`
   - std = `[0.225, 0.225, 0.225]`
6. `[T,C,H,W]` を `[1,C,T,224,224]` へ変換

temporal resampling、`UniformTemporalSubsample`、random resize/crop/flipは使用しない。
`frames_per_clip = 16`、`sampling_rate = 1` とし、loaderのvalid frame順を維持する。
resizeの数値policyはtrain / valとも `interpolation = bilinear`、`antialias = True` に固定する。

決定的transformが必要な理由は、stateful causal PatchEmbedが前chunkの実frameを引き継ぐためである。
chunkごとに異なるrandom cropを適用すると、chunk境界の空間対応が崩れる。

## 7. Model契約

### 7.1 Backbone、checkpoint、head

- modelは現行 `model/memvit/memvit_model.py` のMeMViTを基準にする。
- checkpoint lineageは `models/Kinetics/MeMViT_16L_16x4_K400.pyth` とする。
- checkpointのSHA-256は
  `3c61adbb7e045d8cc2435d6f26b3f8d74460786dfcde97a9579d44922eb4bc0d` とする。
- load前にSHA-256を検証し、不一致ならfail-fastする。
- K400 pretrained PatchEmbedを含むbase weightを再初期化しない。
- temporal relative position parameterは必要な32 tensorだけ既存の検証済み方法でinterpolateする。
- pretrained classification headの2 tensorはshape不一致としてskipし、新規51-class headを使う。
- その他のunresolved keyを許容しない。
- headは各frameのspatial tokenを集約してraw logits `[1,16,51]` を返す。
- train / valともCrossEntropyLossへsoftmax前のraw logitsを渡す。
- outputがfiniteであることを確認する。

ユーザー確認済みcheckpoint smokeの期待値は次のとおりである。

```text
direct load = 571
temporal relative position interpolation = 32
skipped head = 2
unresolved = 0
missing = new classification head only
```

### 7.2 Strict frame-level causality

frame `t` のlogitへframe `t+1` 以降のvisual inputが影響してはならない。

現行PatchEmbedは `Conv3d(kernel_t=3, padding_t=1)` のsymmetric temporal convolutionであり、
attention側をcausalにしてもfuture frameが1つ前のPatchEmbed出力とlogitへ混入する。本specでは
PatchEmbedを次のように変更する。

```text
temporal kernel = 3
left temporal padding = 2
right temporal padding = 0
Conv3d temporal padding = 0
spatial padding = 3（現行維持）
temporal stride = 1（現行維持）
```

処理は明示的なleft padと既存 `Conv3d` weightで実現し、pretrained projection weightのshape、値、
checkpoint key lineageを維持する。

### 7.3 Stateful causal PatchEmbed

- 同一sequenceでは前chunkの最後の実frame 2枚をhistoryとして保持し、次chunkのleft contextに使う。
- sequence先頭はhistoryなしとし、不足するleft contextをzeroで埋める。
- history更新では `valid_mask=True` の実frameだけを使い、loader paddingを実frameとして保存しない。
- historyはgradientを前chunkへ遡らせない状態として保持する。
- public forward outputのtemporal長は入力chunkと同じ16を維持する。
- repeat-last paddingやchunk overlapを新設しない。

naiveなchunk単位zero-left-padは、連続32-frame入力に対するcausal convolutionと、16+16 chunk処理の
chunk 2先頭2frameを不一致にするため禁止する。

### 7.4 Online state lifecycle

online stateは次の2種類である。

1. MeMViT attention memory
2. Stateful causal PatchEmbedの2-frame history

次のreset規則を両方へ原子的に適用する。

- `sample.is_first is True` のforward前
- train epoch開始時
- validation epoch開始時
- epoch境界checkpointからresumeした直後、最初のsampleを処理する前

同一sequence内では両stateを保持し、別sequenceへ一切引き継がない。既存 `clear_memory()` 相当の
model-level public reset経路から両方をclearできるようにする。

## 8. LoRA契約

`peft == 0.20.0` のPEFT方式を使い、MeMViT全16 blockのattention projectionだけを対象にする。

```text
target_modules = ["q", "v"]
r = 8
lora_alpha = 16
lora_dropout = 0.0
```

現行50Salads configの `MVIT.POOL_FIRST = True` では各blockに個別の `q` / `v` Linearが存在する。
実装後はtarget moduleを列挙して、意図しない同名moduleを含まず次を満たすことをassertする。

```text
target modules = 32
LoRA tensors = 64
head tensors = 2
```

checkpoint load後にbase pretrained parametersをfreezeし、trainable parameterを次だけに限定する。

- q/v LoRA parameters
- 新規51-class classification headのweight / bias

optimizerには `requires_grad=True` のparameterだけを渡す。LoRA Bのzero initializationにより、
初回stepでLoRA Aが変化しないことはfailureとしない。少なくともLoRA Bとheadが更新され、frozen baseが
不変であることを確認する。

## 9. Loss、metric、optimization semantics

### 9.1 Lossとmetric

- frame-wise CrossEntropyLossを使う。
- lossはeffective supervision maskがTrueのlogit / targetだけで計算する。
- padding、coarse-only、no-label位置をlossとmetricから除外する。
- validationの最低限のmetricは、全validation sequenceのsupervised frameを分母とするglobal
  frame-wise top-1 accuracyとする。
- 既存 `compute_topk_accuracy` 等を局所計算へ再利用してよいが、tail長の違うchunkを同じ重みで
  平均せず、correct countとsupervised frame countを集約する。
- 科学的評価の最終metric、VLM・動画生成による評価は本specで定義しない。

### 9.2 Manual Optimization

Lightningの `automatic_optimization` を無効化し、strict sequential pathではgradient accumulationを
使わない。baselineの `grad_accum = 1` を固定し、1 supervised chunkを1 optimization stepとする。

各chunkを次の順序で処理する。

1. sequence開始なら両online stateをresetする。
2. forwardを必ず1回実行し、attention memoryとPatchEmbed historyを進める。
3. effective supervision maskと `supervised_count` を計算する。
4. `supervised_count == 0` の場合:
   - backwardしない。
   - optimizer.stepしない。
   - scheduler stepしない。
   - Lightningのglobal optimization stepを進めない。
5. `supervised_count > 0` の場合:
   - optimizerのgradientをclearする。
   - masked CEを計算する。
   - manual backwardを1回行う。
   - optimizer.stepを1回行う。
   - Lightningのglobal optimization stepを1だけ進める。

unsupervised chunkでzero lossをautomatic optimizationへ返す方式は採用しない。

### 9.3 Optimizerとscheduler

初期baselineは次に固定する。

```text
optimizer = AdamW
learning rate = 1e-4
weight_decay = 0.01
scheduler = none
```

現行のscheduler無効時のdummy `ConstantLR`は本経路で作らない。OrthogonalAdamW、optimizer比較、
hyperparameter studyは対象外である。

## 10. Epoch、validation、checkpoint / resume

### 10.1 Epoch semantics

実装はmulti-epoch対応とし、初期baselineは `num_epochs = 1` とする。

各epochで次を維持する。

- canonical sequence順を固定する。
- sequence内はabsolute decoded frame index順、chunk index順とする。
- shuffleしない。
- epoch開始時にonline stateをresetする。
- `num_epochs > 1` では同じsequenceを次epochで再度見ることを許可する。

したがって、結果報告では次を区別する。

- `num_epochs = 1`: strict online one-pass baseline
- `num_epochs > 1`: sequential fine-tuning

validationもcanonical val順に逐次処理し、sequence境界で両stateをresetする。validation stateをtrainの
最終sequenceから引き継がない。

### 10.2 Checkpointとresume

mid-sequence resumeは保証しない。epoch境界checkpointだけをresume対象にする。

epoch境界checkpointは少なくとも次を再現できる情報を保持する。

- LoRAと51-class headのstate
- optimizer state
- completed epochとglobal optimization step
- 採用config、label mapping、基準repository commit
- K400 checkpointの識別情報

attention memoryとPatchEmbed historyはepoch境界で空であるため永続化せず、resume時にclearする。
loader positionの途中復元は行わず、次epochのcanonical先頭sequenceから開始する。

既存 `TRAIN.AUTO_RESUME` がepoch境界を保証できない場合、sequential baseline configでは無効化する。
明示resume時もepoch完了checkpoint以外は拒否する。overwriteやcheckpoint選択を暗黙に行わない。

## 11. 互換性と変更範囲

### 11.1 維持するもの

- MeMViT K400 pretrained checkpoint lineage
- pretrained PatchEmbed weightの値とshape
- external `sequential_loader` のpublic APIと責務境界
- loaderはvisual / temporal sample生成、Consumerはfine label解釈という分担
- 既存non-sequential baselineの既定挙動
- 再利用可能な既存model、optimizer、metric helper

### 11.2 実装承認後に想定する影響範囲

現行main基準では、少なくとも次のcomponentが候補になる。正確な局所配置は実装開始時のmainに合わせる。

| Component | 必要な責務 |
|---|---|
| dependency definition | PEFTとexternal loader revisionを再現可能に固定 |
| `dataset/dataloader_factory.py` またはConsumer-side bridge | canonical split合成、public loader構築、Sample受け渡し |
| Consumer-side annotation helper | direct-index fine label parsing、mask、51-class mapping |
| `dataset/transforms.py` またはsequential専用transform | deterministic K400 preprocessing |
| `model/memvit/stem_helper.py` | stateful causal PatchEmbed |
| `model/memvit/memvit_model.py` | 両online stateのmodel-level reset、raw frame logits |
| checkpoint helper / model construction | K400互換load、relative position interpolation、head skip |
| LoRA setup | q/v PEFT注入、freeze、trainable parameter audit |
| `model/simple_lightning_model.py` | Manual Optimization、masked loss、state lifecycle、metric集約 |
| `main_pl.py` / sequential config | single-process strict設定、multi-epoch、epoch-boundary resume |
| tests | Success Criteria 1〜15の短時間検証 |

既存baselineを不要に変えないため、external loader、causal PatchEmbed、LoRA、Manual Optimizationは
50Salads sequential LoRA経路で明示的に有効化する。既存interfaceを変更する必要が生じた場合は、
breaking change、影響、移行方法を示して本specを再承認する。

## 12. Success Criteria

### 12.1 実装成功条件

| ID | 観測可能な成功条件 | 推奨検証レベル |
|---|---|---|
| SC-01 | canonical splitがtrain 14、val 4、duplicate 0、overlap 0、missing video/annotation 0 | dataset preflight |
| SC-02 | source / frame / chunk順がchronologicalでshuffleなし | loader integration test |
| SC-03 | decoded frame `i` とannotation frame ID `i` がdirect一致し、`-1` shiftしない | alignment test |
| SC-04 | input `[1,3,16,224,224]` からfinite raw logits `[1,16,51]` を得る | model smoke |
| SC-05 | future frame perturbationによる過去frame logitsの差が0 | strict causality test |
| SC-06 | continuous referenceとchunked stateful causal PatchEmbedがchunk境界を含め一致 | unit test |
| SC-07 | zero / repeat-lastのtest-only tail variantでvalid frame logitsの差が0 | padding invariance test |
| SC-08 | sequence boundary reset後の先頭chunkがclean startと一致し、attention cache 0、PatchEmbed history `None` | state reset test |
| SC-09 | trainable parameterがq/v LoRA 64 tensorと51-class head 2 tensorだけ | parameter audit |
| SC-10 | optimizer update後もfrozen pretrained baseが不変 | update smoke |
| SC-11 | `supervised_count == 0` でforwardあり、backward/optimizer.stepなし、global optimization step不変 | Manual Optimization test |
| SC-12 | `supervised_count > 0` でmasked CE、backward、optimizer.stepが各1回、global optimization stepが+1 | Manual Optimization test |
| SC-13 | `num_epochs = 1` でcanonical trainをchronologicalに1回だけ走査できる | 別途許可されたbaseline run |
| SC-14 | tiny fixtureで `num_epochs > 1` でも各epochの順序を維持し、epoch境界で両stateをresetする | multi-epoch integration test |
| SC-15 | padding、coarse-only、no-label frameがloss / metricの分母・分子から除外される | mask / metric test |

SC-05〜SC-08の比較はmodelをeval modeにし、同じweight・state・dtype・入力prefixを使う。ユーザー確認済み
EvidenceではSC-05〜SC-08に対応する `max abs diff = 0.0` が得られているため、実装testでも厳密一致を
期待値とする。

SC-13は「実装がone-passを支える」というacceptance conditionだが、実dataset全体を用いるfull training
runは本specの実装・短時間検証scope外であり、別途ユーザー許可後に実行する。承認前および実装だけの
依頼からfull runの許可を推定しない。

### 12.2 科学的成功条件との分離

本specでは次を成功条件にしない。

- LoRAに時間・動作情報が有意に蓄積されたこと
- downstreamでbaselineを上回ること
- static / dynamic成分が分離可能であること

本specの科学的役割は、後続解析へ渡せる再現可能な逐次学習基盤を成立させることである。上記主張は
別specでbaseline、metric、比較条件、seed、統計判定を定義して検証する。

## 13. Reproducibility記録

初期baselineで使用する、確認済みの実行環境と数値条件を次に固定する。

```text
torch == 2.14.0+cu130
torchvision == 0.29.0+cu130
lightning == 2.6.5
peft == 0.20.0
Lightning precision = "32-true"
resize interpolation = bilinear
resize antialias = True
K400 checkpoint SHA-256 = 3c61adbb7e045d8cc2435d6f26b3f8d74460786dfcde97a9579d44922eb4bc0d
```

将来の許可済みrunでは、既存logging / output構造に合わせて少なくとも次を保存する。

- 実装repository commit、external loader commit、dirty state
- 実効configとCLI引数
- Python、PyTorch、Lightning、PEFT、CUDAのversion
- device、precision、seed
- K400 checkpoint pathとSHA-256
- canonical sequence ID一覧とphysical split対応
- label-to-index mapping
- supervision統計とdataset file manifestまたは同等のdataset識別情報
- optimizer step数、supervised / unsupervised chunk数
- epoch境界checkpointとmetric集計の分母

同じrun directoryを暗黙に上書きしない。

## 14. Ambiguity Gate

### 14.1 Blocking: なし

2026-09-09のユーザー指示により、従来のblocking項目はすべて次のとおり解消した。

| 従来のblocking項目 | 確定内容 |
|---|---|
| PEFT version / dependency pin | `peft == 0.20.0` |
| K400 checkpoint fingerprint | SHA-256 `3c61adbb7e045d8cc2435d6f26b3f8d74460786dfcde97a9579d44922eb4bc0d` |
| Deterministic resizeの数値policy | bilinear interpolation、`antialias = True` |
| Baseline numerical precision | Lightning `precision = "32-true"` |

Gate再評価の結果、`approved` を妨げるblockingな未決事項は0件である。ただし、本更新ではユーザーが
`status: draft` の維持と実装停止を明示しているため、statusはdraftのままとし、実装権限は発生しない。

### 14.2 Non-blocking: 既存styleに従ってよい

- 新しいhelper / testの正確なファイル名と局所配置
- 既存loggerへ追加する内部metric key名
- run artifactの具体的なdirectory名。ただし上書き禁止と必須記録項目は維持する。
- reset helperの内部method名。ただしmodel-levelから両stateを1回でclearできることを維持する。

## 15. 技術検証Evidenceのトレーサビリティ

ユーザー報告済みEvidenceと本spec要件の対応は次のとおりである。

| Evidence | 確認済み内容 | 対応箇所 |
|---|---|---|
| A, B | direct frame-label alignment、fine 51 classes | §5.3、SC-03 |
| C | K400 checkpoint互換、forward finite | §7.1、SC-04 |
| D | q/v LoRA + headのみ更新 | §8、SC-09、SC-10 |
| E, F | loader→model bridge、mask連携、unsupervised forward | §5、§9 |
| G | Manual Optimizationのstep semantics | §9.2、SC-11、SC-12 |
| H, I | original causality violationとPatchEmbed原因 | §7.2 |
| J, K | causal PatchEmbedとfull MeMViTでpast diff 0 | §7.2、SC-05 |
| L | 2-frame historyでcontinuous/chunked一致 | §7.3、SC-06 |
| M | valid logitsのtail padding不変性 | §5.2、SC-07 |
| N | attention + PatchEmbed state resetでclean start一致 | §7.4、SC-08 |
| O | canonical split整合性 | §5.1、SC-01 |

## 16. Implementation Handoff

本節は索引であり、`status: draft` の間はinactiveである。

- approved spec: なし。このdraftを承認後、frontmatterを `approved` へ変更したrevisionを使用する。
- 実装目的: 50Salads strict sequential one-pass / multi-epoch LoRA fine-tuning基盤
- 基準repository/commit: `tamaki-lab/2026_04_ishikawa_simple-MeMViT@e0deb093694d367ed9b02065e6d4cd38802093d6`
- external loader: `tamaki-lab/sequential_loader@cef09aa12560127451a5f569d86d5d51671e6986`
- 変更scope: §11.2
- 対象外・維持条件: §3.2、§11.1
- success criteria: SC-01〜SC-15
- 許可される短時間検証: 承認後のunit test、synthetic/tiny fixture integration、checkpoint/model smoke
- 長時間runの許可状態: 未許可
- 未検証予定: SC-13の実dataset one-pass full training、科学的評価全般
