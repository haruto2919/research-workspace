---
project: sequential-video-lora-analysis
spec_type: implementation
status: implemented
title: Stage 3 ActivityNetからfrozen ViTとmasked meanによるclip feature生成
created: 2026-09-23
last_updated: 2026-09-23
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
workspace_base_commit: 58d30cdaea4d701ef54db753681c6d6485378670
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_base_branch: dev
implementation_base_commit: 3c0e40e86924ceb38931c2d5214ecf3251a8f99d
implementation_work_branch: dev
sequential_loader_repository: tamaki-lab/2026_09_ishikawa_sequential_loader
sequential_loader_branch: ActivityNet
sequential_loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
hf_checkpoint: google/vit-base-patch16-224
dataset: ActivityNet v1.3
dataset_split: training
frames_per_chunk: 16
feature_size: 768
---

# Stage 3 ActivityNetからfrozen ViTとmasked meanによるclip feature生成 spec

> **Status: implemented**
>
> 本specは、ActivityNet v1.3の1 chunkを既存Sequential Loaderとfrozen ViTで処理し、
> valid frameだけを用いたmasked meanによってorder-invariantな
> `clip_feature [768]` を生成するStage 3の実装契約である。
>
> 2026-09-23のユーザー指示により承認し、同日に実装・必須の短時間検証を完了した。
> 詳細は[実装・検証記録](../experiments/2026-09-23-activitynet-clip-feature-verification.md)を参照。
> 学習・長時間runは許可範囲に含めない。

## 1. 目的

既に次の基盤は成立している。

- `ViTFrameEncoder` によるsingle-frame CLS feature `[B,768]` 抽出。
- 50Salads `SequentialSample` からfrozen ViTへ接続し、
  `frame_features [T,768]` を得るStage 2 bridge。
- ActivityNet v1.3 Adapterによる
  `training / validation / testing` source列挙とSequential Loader Coreへの接続。
- training設定基盤のargparseからHydra/YAMLへの移行。

一方、ActivityNetを研究code側のViT経路へ接続し、
動画1 chunkをfixed-sizeのclip representationへ変換する経路は未実装である。

本specでは次を成立させる。

```text
ActivityNet v1.3 training subset
  -> ActivityNetAdapter
  -> SequenceSource
  -> SequentialDataset
  -> SequentialSample [T,3,H,W]
  -> valid frame extraction
  -> AutoImageProcessor
  -> frozen ViTFrameEncoder
  -> valid_features [N_valid,768]
  -> scatter
  -> frame_features [T,768]
  -> MaskedMeanClipAggregator
  -> clip_feature [768]
```

このStageの目的は、
LoRA / MoCo実装前に **ActivityNetからvideo-level featureを得る最小baseline** を固定することである。

## 2. Authority / Evidence

### 2.1 現在の方針

直近のresearch directionでは、

1. image-pretrained ViTを利用する。
2. 動画をSequential Loaderから時間順に入力できる基盤を作る。
3. 最初のclip representationは、時間順序を使わない単純baselineとしてmasked meanを用いる。
4. dataset integration、aggregation、LoRA、MoCoを一度に実装しない。
5. LoRAはStage 3完了後の別specで扱う。
6. temporal learningの主張は、LoRA/MoCo統合後のcontrol比較まで行わない。

とする。

### 2.2 implementation repository

基準revision:

```text
tamaki-lab/2026_09_ishikawa_sequential-video-lora
dev@3c0e40e86924ceb38931c2d5214ecf3251a8f99d
```

実装branchは既存 `dev` を使用する。

### 2.3 Sequential Loader

ActivityNet Adapterの基準revision:

```text
tamaki-lab/2026_09_ishikawa_sequential_loader
ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
```

このrevisionでは、ActivityNet v1.3について、

- `training=10024`
- `validation=4926`
- `testing=5044`
- `.mp4/.mkv/.webm`
- 1 physical video = 1 whole-video sequence
- annotation subsetをsplit SSOTとして使用
- action segmentをtraining targetやcropへ使用しない

というAdapter contractが成立している。

### 2.4 既存ViT contract

現在の `ViTFrameEncoder` は、

```text
google/vit-base-patch16-224
ViTModel(..., add_pooling_layer=False)
CLS token
input  [B,3,224,224]
output [B,768]
backbone frozen
```

を使用する。

本specではこのcontractを変更しない。

## 3. Decision Contract

### 3.1 Dataset

固定:

```text
ActivityNet v1.3
split = training
```

Stage 3のsmokeではtraining subsetの決定的な先頭sourceから
先頭chunkを取得する。

validation / testing比較は本specでは行わない。

### 3.2 Sequential Loader revision

固定:

```text
tamaki-lab/2026_09_ishikawa_sequential_loader
ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
```

research codeへloader implementationをcopy / vendorしない。

local cloneの絶対pathは実装へhard-codeしない。

### 3.3 chunk contract

固定:

```text
frames_per_chunk = 16
sampling = contiguous frames
B = 1
```

Stage 3ではframe stride、時間秒ベースsampling、random temporal crop等を追加しない。

期待するsampleの中心contract:

```text
frames         [16,3,H,W]
valid_mask     [16]
frame_indices  [16]
timestamps     [16]
```

validでないtail位置が存在する場合は、
そのframeをViTへ入力しない。

### 3.4 Annotation handling

ActivityNetのlabel / temporal segmentは、

- model input
- feature aggregation
- training target
- sequence crop

へ使用しない。

annotationはActivityNet Adapter側のsplit / evaluation referenceとしてのみ扱う。

### 3.5 preprocessing

既存Stage 1 / Stage 2と同じ、

```python
AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)
```

を使用する。

valid frameだけをprocessorへ渡す。

期待shape:

```text
pixel_values [N_valid,3,224,224]
```

独自resize / normalizeを再実装しない。

### 3.6 frame feature extraction

既存 `ViTFrameEncoder` を変更せず使用する。

```text
valid_features [N_valid,768]
```

を取得後、元のT位置へscatterし、

```text
frame_features [16,768]
```

を作る。

`valid_mask=False` の行はzero vectorとする。

frame order / frame_indices / timestampsをsortし直さない。

### 3.7 MaskedMeanClipAggregator

新しい小さい `nn.Module` として、

```text
MaskedMeanClipAggregator
```

を追加する。

概念interface:

```python
class MaskedMeanClipAggregator(nn.Module):
    def forward(
        self,
        frame_features: torch.Tensor,  # [T,D]
        valid_mask: torch.Tensor,      # [T]
    ) -> torch.Tensor:                 # [D]
        ...
```

計算:

```text
clip_feature
= sum(valid frame_features) / number_of_valid_frames
```

数式:

```text
z_clip = (Σ_t m_t z_t) / (Σ_t m_t)
```

ここで、

- `z_t`: frame feature
- `m_t ∈ {0,1}`: valid mask

とする。

Module自体はtrainable parameterを持たない。

### 3.8 mask validation

最低限、次を検査する。

- `frame_features.ndim == 2`
- `valid_mask.ndim == 1`
- T dimensionが一致する。
- valid frame数が1以上。

all-invalid mask:

```text
valid_mask.sum() == 0
```

は明示的なerrorとする。

zero除算やNaNへ進ませない。

### 3.9 order invariance

masked meanは意図的にorder-invariantとする。

同じvalid feature集合について、

```text
[A, B, C]
[C, B, A]
[B, C, A]
```

の出力は一致しなければならない。

これはtemporal modelingの評価ではなく、
**baselineが時間順序を使わないことの仕様確認**である。

### 3.10 gradient semantics

`MaskedMeanClipAggregator.forward()` 内へ、

- `torch.no_grad()`
- `.detach()`
- CPUへの強制転送

を埋め込まない。

Stage 3 smokeでは外側で、

```python
encoder.eval()

with torch.no_grad():
    ...
```

を使用する。

これによりStage 4以降でLoRAを追加した際、
同じaggregationをgradient path上で再利用できる状態を維持する。

### 3.11 ViT freeze

Stage 3ではViT backboneの全parameterをfreezeしたままとする。

```text
trainable backbone parameters = 0
```

をsmokeで確認する。

LoRAはまだ追加しない。

### 3.12 Hydraとの境界

本specでは、

```text
conf/dataset/activitynet.yaml
```

を追加しない。

また、

- `main.py`
- `main_pl.py`
- `TrainValDataModule`
- `SimpleLightningModel`

へActivityNet SequentialSample trainingを統合しない。

理由:
現行training pathはclassification用 `(data, labels)` contractであり、
Stage 3でActivityNet SequentialSampleを混ぜると、
dataset integrationだけでなくtraining interface変更まで同時に発生するため。

Stage 3はstandalone integration smokeとして成立させる。

## 4. Implementation Design

### 4.1 implementation branch

固定:

```text
dev
```

基準:

```text
3c0e40e86924ceb38931c2d5214ecf3251a8f99d
```

実装開始時にdev HEADが基準revisionから進んでいる場合、
差分が本spec contractへ影響するか確認する。

### 4.2 aggregator

推奨配置:

```text
model/masked_mean_clip_aggregator.py
```

必要に応じて `model/__init__.py` からexportする。

大きなVideoEncoder hierarchy、aggregation registry、
汎用temporal frameworkは作らない。

### 4.3 ActivityNet integration smoke

新規entrypoint:

```text
smoke_activitynet_vit_clip_feature.py
```

責務:

1. ActivityNet dataset rootをruntime引数で受け取る。
2. pinned ActivityNet Adapterから `training` sourcesを取得する。
3. `FixedChunkConfig(frames_per_chunk=16)` でSequentialDatasetを構築する。
4. strict sequential dataloaderから先頭1 sampleを取得する。
5. valid frameだけをprocessorへ渡す。
6. frozen ViTでvalid featureを生成する。
7. `frame_features [16,768]` へscatterする。
8. `MaskedMeanClipAggregator` で `clip_feature [768]` を作る。
9. shape / finite / padding / freeze / metadataを検証する。
10. diagnosticを標準出力へ出す。
11. reader lifecycleを正しくcloseする。

### 4.4 Stage 2 helper reuse

既存 `smoke_50salads_vit_bridge.py` から、
branch / dataset / revision checkまで含むscript全体をimportしない。

必要であれば、

```text
valid frame extraction
processor
ViT
scatter
```

だけをdataset非依存の小さいhelperへ切り出してよい。

ただし本specで大きな抽象化を新設しない。

既存50Salads smokeの外部挙動を壊さない。

## 5. Smoke Diagnostic

最低限、次を確認・表示する。

```text
implementation branch / revision
sequential_loader revision
checkpoint ID
resolved device

dataset = ActivityNet v1.3
split = training
sequence_id
sequence_index
is_first
is_last

frames shape
frames dtype
valid count / T
valid_mask
frame_indices
timestamps

pixel_values shape
valid_features shape
frame_features shape
clip_feature shape

frame feature finite
clip feature finite
padding rows zero

total backbone parameters
trainable backbone parameters
```

期待中心値:

```text
frames              [16,3,H,W]
pixel_values        [N_valid,3,224,224]
valid_features      [N_valid,768]
frame_features      [16,768]
clip_feature        [768]
trainable backbone parameters = 0
clip feature finite = True
```

## 6. Unit Test Plan

### 6.1 MaskedMeanClipAggregator

synthetic tensorで最低限次を検証する。

1. all-validで通常meanと一致。
2. invalid rowを平均から除外する。
3. padding featureが大きな値でもvalid maskがfalseなら出力へ影響しない。
4. reverseしても出力が一致する。
5. 任意permutationでも出力が一致する。
6. all-invalid maskをerrorにする。
7. T dimension mismatchをerrorにする。
8. output shapeが `[D]`。
9. outputがinputと同じdevice上に残る。
10. gradientを妨げない。

gradient checkでは、例えば `frame_features.requires_grad=True` で
outputからbackwardできることまでを短時間で確認してよい。

これはStage 3でparameter updateを行うことを意味しない。

### 6.2 frame-to-clip integration helper

helperを切り出す場合のみ、
synthetic/mocked inputで次を確認する。

- valid frameだけprocessor/encoder対象になる。
- scatter後のinvalid rowがzero。
- frame alignmentを変更しない。

helperを切り出さない場合はActivityNet smokeと既存Stage 2 testで担保する。

## 7. Real ActivityNet Smoke

研究サーバ上の実ActivityNet v1.3を使った、
先頭1 chunk程度の短時間smokeを必須とする。

dataset rootはruntime引数とし、
コードへ絶対pathをhard-codeしない。

確認項目:

1. ActivityNet Adapterをimportできる。
2. `training` sourceを取得できる。
3. 先頭sourceから `SequentialSample` を得られる。
4. T=16 contractを満たす。
5. valid frameが1枚以上ある。
6. processor後shapeが正しい。
7. ViT featureが `[N_valid,768]`。
8. scatter後が `[16,768]`。
9. padding rowがzero。
10. masked meanが `[768]`。
11. frame / clip featureがfinite。
12. trainable backbone parameter数が0。
13. backward / optimizer / parameter updateを実行していない。
14. annotation label / segmentをmodel inputへ渡していない。

full ActivityNet全動画forwardは実施しない。

## 8. Success Criteria

本specの実装成功は次を全て満たすこととする。

1. implementation repositoryが
   `tamaki-lab/2026_09_ishikawa_sequential-video-lora`。
2. implementation branchが `dev`。
3.基準revisionが
   `3c0e40e86924ceb38931c2d5214ecf3251a8f99d`
   またはその後継で、本spec contractへの影響が確認済み。
4. Sequential Loaderは
   `ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`
   を基準とする。
5. ActivityNet v1.3 `training` sourceを取得できる。
6. contiguous 16 frame chunkを取得できる。
7. valid frameだけをprocessorへ渡す。
8. processor後が `[N_valid,3,224,224]`。
9. frozen `ViTFrameEncoder` から `[N_valid,768]` を取得する。
10. 元T位置へscatterして `frame_features [16,768]` を作る。
11. invalid rowがzero。
12. `MaskedMeanClipAggregator` が存在する。
13. valid rowのみから `clip_feature [768]` を生成する。
14. paddingを平均分母に含めない。
15. all-invalid maskをerrorにする。
16. clip featureがfinite。
17. reverse / permutationでmasked mean出力が一致する。
18. ViT backboneのtrainable parameter数が0。
19. Stage 3 smokeで `eval() + torch.no_grad()` を使用する。
20. aggregator内部ではgradientをdetachしない。
21. annotation label / segmentをmodel inputに使用しない。
22. frame order / metadata alignmentを変更しない。
23. `main.py` / `main_pl.py` のclassification training contractを変更しない。
24. HydraへActivityNet training configを追加しない。
25. LoRA / MoCo / optimizer / backward / training loopを混入していない。
26. 既存50Salads bridgeとHydra config testsに新規回帰を入れない。

## 9. Explicit Out of Scope

本specでは次を実装・検証しない。

- LoRA。
- LoRA target / rank / alpha / dropout。
- LoRA parameter update。
- MoCo / BYOL / MAE等のSSL objective。
- query encoder / key encoder。
- EMA。
- projector。
- InfoNCE。
- queue。
- optimizer。
- training loss。
- training loop。
- sequential vs shuffle学習比較。
- reverseを使ったtemporal評価。
- GRU / LSTM / temporal Transformer。
- order-aware attention。
- frame delta modeling。
- CLIP-ViT。
- ActivityNet annotationを使うdownstream task。
- Hydra training pathへのActivityNet integration。
- `B>1` video batch semantics。
- final temporal sampling policy。
- full ActivityNet feature extraction。
- long training / GPU experiment。

## 10. Compatibility

本変更は既存classification training pathに対してadditiveとする。

維持する:

- `main.py`
- `main_pl.py`
- Hydra config groups
- existing dataset factory
- existing model factory
- `SimpleLightningModel`
- `ViTFrameEncoder` contract
- 50Salads Stage 2 smoke
- utility / smoke CLI方針

Stage 3のために、
既存classification用 `dataset=activitynet` を見かけ上追加しない。

ActivityNet SequentialSampleをtraining systemへ統合する責務は後続specへ残す。

## 11. Failure / Stop Conditions

次の場合は条件を黙って変更せず停止・報告する。

- pinned ActivityNet Adapter revisionを利用できない。
- ActivityNet full layoutがAdapter contractと一致しない。
- `training` sourceを取得できない。
- SequentialSample contractが想定と異なる。
- valid frameが0枚。
- processor後shapeが `[N_valid,3,224,224]` にならない。
- ViT output hidden sizeが768でない。
- frame feature / clip featureにNaN / Infがある。
- padding rowをzeroに保てない。
- masked meanでpaddingを除外できない。
- `MaskedMeanClipAggregator` を成立させるためViT contract変更が必要になる。
- Stage 3成立に既存classification training interface変更が必要になる。
- ActivityNet対応のためSequential Loader Core変更が必要になる。
- LoRA / MoCoを先に入れないとclip feature生成が成立しない。

この場合、別sampling、別backbone、label supervision、
training integration等へ勝手にscopeを拡張しない。

## 12. Ambiguity Gate

### 12.1 Blocking

現時点で技術的なblocking ambiguityはない。

本specでは次を固定する。

- implementation branch: `dev`
- implementation base: `3c0e40e86924ceb38931c2d5214ecf3251a8f99d`
- loader: `ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`
- dataset: ActivityNet v1.3
- split: `training`
- chunk: contiguous 16 frames
- B=1
- checkpoint: `google/vit-base-patch16-224`
- frame feature: CLS / 768 dim
- ViT frozen
- invalid frameはViTへ入力しない
- invalid feature rowはzero
- aggregation: parameter-free masked mean
- all-invalidはerror
- order-invariant baseline
- standalone smoke
- Hydra training pathへは未統合
- LoRA / MoCoは対象外

### 12.2 Non-blocking

既存styleに従って決めてよい。

- aggregator fileのprivate helper構成。
- exact exception type / error message。
- test function名。
- ActivityNet smoke CLI option名。
- diagnostic printの細かな表示形式。
- frame encoding logicをsmall helperへ切り出すか、smoke内に置くか。
- `model/__init__.py` からaggregatorをexportするか。
- smoke実行deviceをCUDA優先/CPU fallbackにする細かな表現。

これらはinput/output、mask semantics、order invariance、
freeze、scopeを変更してはならない。

## 13. Reproducibility

Research Workspace基準:

```text
haruto2919/research-workspace
main@58d30cdaea4d701ef54db753681c6d6485378670
```

research code基準:

```text
tamaki-lab/2026_09_ishikawa_sequential-video-lora
dev@3c0e40e86924ceb38931c2d5214ecf3251a8f99d
```

loader基準:

```text
tamaki-lab/2026_09_ishikawa_sequential_loader
ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
```

model checkpoint:

```text
google/vit-base-patch16-224
```

dataset rootはruntimeで渡し、repositoryへ絶対pathをhard-codeしない。

## 14. このStageで主張できること

Stage 3が成功した場合、次だけを主張できる。

- ActivityNet動画をSequential Loader経由で既存ViT feature pathへ接続できる。
- 1 chunkのvalid frameからfixed-size `clip_feature [768]` を生成できる。
- paddingをclip representationから除外できる。
- ViT backboneをfrozenのまま利用できる。
- masked mean baselineがorder-invariantである。

次はまだ主張できない。

- LoRAが動画情報を学習した。
- LoRAが時間情報を学習した。
- clip featureが時間順序を表現している。
- 動的情報を獲得した。
- sequential inputがshuffleより優れている。
- ActivityNet上で性能が向上した。

## 15. 実装後の次段階

Stage 3 implemented後は別specでStage 4へ進む。

```text
ActivityNet frames
 -> ViT + LoRA
 -> frame_features
 -> masked mean
 -> clip_feature
```

Stage 4では、

- base ViT frozen
- LoRAのみtrainable
- LoRA target
- rank
- alpha
- dropout
- 1-step update smoke
- base parameter不変
- LoRA parameter更新

を独立して固定する。

MoCoはさらにその後のStage 5とする。

## 16. Spec Gate

本specは2026-09-23のユーザー指示「このspecをapprovedに変更して実装してください」により `approved` へ更新した。

目的、dataset、revision、input/output、aggregation、
mask semantics、freeze、Hydraとの責務境界、
成功条件、対象外は定義済みで、blocking ambiguityは残っていない。

コード実装、unit / regression tests、実ActivityNetの先頭1 chunk smokeを完了し、
現在は `implemented`。新規25件・50Salads 1件・Hydra 38件が成功した。
既存ViT testの期待引数不一致1件は実装前後で同一で、新規回帰はない。
詳細は[実装・検証記録](../experiments/2026-09-23-activitynet-clip-feature-verification.md)を参照。
実装開始時に、implementation `dev` とloader `ActivityNet` のHEADが
それぞれ本specの基準revisionと一致し、両worktreeがcleanであることを確認した。
学習・長時間run・commit・pushは実施範囲に含めない。

# Implementation Handoff

- approved spec: 本spec
- 実装目的: ActivityNetの1 chunkからorder-invariantな `clip_feature [768]` を生成する
- implementation repository: `tamaki-lab/2026_09_ishikawa_sequential-video-lora`
- branch: `dev`
- base: `3c0e40e86924ceb38931c2d5214ecf3251a8f99d`
- loader: `tamaki-lab/2026_09_ishikawa_sequential_loader@ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`
- change scope: ActivityNet integration smoke + MaskedMeanClipAggregator + required small tests/helpers
- must preserve: existing Hydra/classification path / ViTFrameEncoder / 50Salads bridge
- success criteria: 8章
- allowed short verification: unit tests / synthetic aggregation tests / ActivityNet先頭1 chunk smoke
- long run permission: なし
- excluded: LoRA / MoCo / optimizer / training / temporal evaluation
