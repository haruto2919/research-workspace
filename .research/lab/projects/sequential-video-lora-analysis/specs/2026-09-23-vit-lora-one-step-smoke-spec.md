---
project: sequential-video-lora-analysis
spec_type: implementation
status: implemented
title: Stage 4 ActivityNet clip経路へのViT LoRA注入と1-step更新smoke
created: 2026-09-23
last_updated: 2026-09-24
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_base_branch: dev
implementation_base_commit: b6385d87e2e3e82a719d8f4b686b44aa293b1135
implementation_work_branch: dev
sequential_loader_repository: tamaki-lab/2026_09_ishikawa_sequential_loader
sequential_loader_branch: ActivityNet
sequential_loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
hf_checkpoint: google/vit-base-patch16-224
transformers_version_observed: 5.17.0
peft_version: 0.21.0
dataset: ActivityNet v1.3
dataset_split: training
frames_per_chunk: 16
feature_size: 768
---

# Stage 4 ActivityNet clip経路へのViT LoRA注入と1-step更新smoke spec

> **Status: implemented**
>
> 本specは、Stage 3で成立した
> `ActivityNet -> ViT frame feature -> masked mean clip feature`
> の経路へLoRAを追加し、base ViTを更新せずLoRAだけを1 step更新できることを
> 検証するための実装契約である。
>
> 本Stageはengineering verificationであり、LoRAが時間情報を学習したこと、
> 動画理解性能が向上したこと、MoCoが成立したことを示すものではない。

## 1. 目的

Stage 3では次が成立している。

```text
ActivityNet v1.3 training
  -> Sequential Loader
  -> contiguous 16 frames
  -> AutoImageProcessor
  -> frozen ViT
  -> frame_features [16,768]
  -> MaskedMeanClipAggregator
  -> clip_feature [768]
```

Stage 4では、この既存video input / aggregation contractを維持したまま、
ViT attentionのQ/V projectionへLoRAを注入する。

```text
ActivityNet v1.3 training
  -> Sequential Loader
  -> contiguous 16 frames
  -> AutoImageProcessor
  -> ViT base frozen + Q/V LoRA trainable
  -> frame_features [16,768]
  -> MaskedMeanClipAggregator
  -> clip_feature [768]
  -> engineering-only scalar loss
  -> backward
  -> AdamW
  -> 1 optimizer step
```

確認したいことは次だけである。

1. base ViTはfreezeされたままである。
2. optimizer対象はLoRA parameterだけである。
3. clip-level scalar lossからLoRAまでgradientが流れる。
4. 1 step後に少なくとも1つのLoRA parameterが変化する。
5. base ViT parameterは1 step前後で変化しない。
6. Stage 3までの既存feature / mask / metadata contractを壊さない。

## 2. Authority / Evidence

### 2.1 研究方針

2026-09-17 MTGおよび2026-09-23の壁打ちでは、
最初に動画LoRAの学習・観測経路を成立させ、その後にMoCoやtemporal objectiveへ進む方針としている。

masked meanは意図的にorder-invariantなnon-temporal baselineである。
したがってStage 4では時間情報獲得を主張しない。

関連brainstorm:

- `.research/secretary/notes/brainstorm/2026-09-23-late-fusion-lora-temporal-learning.md`
- `.research/secretary/notes/brainstorm/2026-09-23-stage4-vit-lora-one-step-next.md`

### 2.2 implementation repository

GitHub上で確認した基準revision:

```text
tamaki-lab/2026_09_ishikawa_sequential-video-lora
dev@b6385d87e2e3e82a719d8f4b686b44aa293b1135
```

このcommitでは `ViTFrameEncoder` が
`ViTModel.from_pretrained(checkpoint_id)` を使用し、poolerを生成する状態である。

本specでは後述のとおり、CLS tokenのみをfeatureとして用いる現在の研究契約に合わせて
`add_pooling_layer=False` を明示し、poolerを生成しない構成へ戻す。

### 2.3 Stage 3 Evidence

Stage 3ではActivityNet trainingの実データ1 chunkについて、

- input: contiguous 16 frames
- `frame_features [16,768]`
- `clip_feature [768]`
- finite feature
- padding除外
- masked meanのorder invariance
- frozen ViT
- gradientをdetachしないaggregator

が確認済みである。

### 2.4 LoRA / PEFT事前確認

研究サーバの現行venvでユーザーが確認した実測値:

```text
PyTorch      2.14.0+cu130
Transformers 5.17.0
PEFT         0.21.0
```

`google/vit-base-patch16-224` のattention linear moduleは各12 blockについて、

```text
q_proj
k_proj
v_proj
o_proj
```

である。

Q/Vだけを抽出した結果:

```text
q_proj: 12
v_proj: 12
total target modules: 24
```

PEFT `LoraConfig` のdry-runでは、

```text
target_modules = ["q_proj", "v_proj"]
r = 8
lora_alpha = 8
lora_dropout = 0.0
bias = "none"
```

でadapter注入に成功し、

```text
trainable parameters = 294,912
trainable tensor count = 48
unexpected trainable = []
CLS feature shape = [2,768]
feature finite = True
```

を確認した。

その後、研究サーバのローカルworktreeで
`ViTModel.from_pretrained(checkpoint, add_pooling_layer=False)` とPEFT 0.21.0を組み合わせた
preflightもユーザーが実行し、次を確認した。

```text
pooler = None
target count = 24
trainable parameters = 294,912
trainable tensor count = 48
unexpected trainable = []
CLS feature shape = [2,768]
feature finite = True
```

loading reportではpoolerのMISSINGは消え、checkpoint側のclassifier weight/biasのみUNEXPECTEDとして残った。
また、ローカルの `ViTFrameEncoder` についても `encoder.vit.pooler is None` を確認した。

これらは研究サーバ上のローカル実行Evidenceであり、GitHub上のresearch codeへ反映済みとは扱わない。

既存testについてはユーザー環境で

```text
65 passed, 2 warnings
```

を確認している。

## 3. Decision Contract

### 3.1 model checkpoint

固定:

```text
google/vit-base-patch16-224
```

### 3.2 frame feature

固定:

```text
outputs.last_hidden_state[:, 0, :]
```

shape:

```text
[B,768]
```

classification headおよびpooler出力は使用しない。

### 3.3 pooler

Stage 4以降のViT feature encoderではpoolerを生成しない。

```python
ViTModel.from_pretrained(
    checkpoint_id,
    add_pooling_layer=False,
)
```

期待:

```text
model.pooler is None
```

理由:

- 現在のfeature contractは `last_hidden_state[:,0,:]` のCLS tokenのみを使う。
- poolerはforward outputとして利用しない。
- checkpointに存在しないpooler parameterをランダム初期化して保持する必要がない。
- Stage 4の「base ViT frozen + LoRAのみtrainable」というparameter contractを簡潔にする。

既存frozen baselineの `ViTFrameEncoder` も同じpoolerなし契約へ合わせる。

### 3.4 LoRA implementation

Hugging Face PEFTを使用する。

固定version:

```text
peft==0.21.0
```

自前LoRA実装は行わない。

### 3.5 LoRA target

conceptual target:

```text
ViT self-attention Query / Value projection
```

実module名:

```text
q_proj
v_proj
```

PEFT config:

```python
target_modules=["q_proj", "v_proj"]
```

期待target数:

```text
q_proj x 12
v_proj x 12
total = 24
```

K projectionおよびoutput projectionへLoRAを追加しない。

### 3.6 LoRA hyperparameters

Stage 4 engineering smokeでは次に固定する。

```yaml
r: 8
lora_alpha: 8
lora_dropout: 0.0
bias: none
```

scalingは通常のLoRA設定として `alpha / r = 1`。

これらは最終的な研究hyperparameterではなく、
Stage 4の動作確認用固定値である。

### 3.7 trainable parameter contract

期待:

```text
LoRA trainable parameters = 294,912
LoRA trainable tensors = 48
unexpected trainable tensors = 0
```

計算:

- target linear: 24個
- each projection: 768 -> 768
- rank: 8
- each targetのLoRA parameter:
  `768*8 + 8*768 = 12,288`
- total:
  `12,288 * 24 = 294,912`

base ViTのparameterはoptimizerへ含めない。

### 3.8 encoder structure

既存 `ViTFrameEncoder` はfrozen baselineとして維持する。

LoRA pathは別クラスとして追加する。

推奨配置:

```text
model/vit/vit_lora_frame_encoder.py
```

推奨class:

```text
ViTLoRAFrameEncoder
```

外部forward contractは既存ViTFrameEncoderと揃える。

input:

```text
pixel_values [B,3,224,224]
```

output:

```text
CLS feature [B,768]
```

大きなencoder factoryやLoRA registryは本Stageでは作らない。

### 3.9 existing video path reuse

Stage 3の次を再利用する。

- ActivityNet Adapter
- Sequential Loader
- contiguous 16-frame chunk
- valid frame extraction
- AutoImageProcessor
- scatter to `frame_features [T,768]`
- `MaskedMeanClipAggregator`

mask、padding、frame alignment、metadata contractを変更しない。

### 3.10 engineering-only smoke loss

MoCo導入前のgradient path確認として次を用いる。

```python
loss = clip_feature.float().pow(2).mean()
```

このlossの数値や減少量には研究的意味を持たせない。

用途は、

```text
clip_feature
 -> scalar loss
 -> backward
 -> masked mean
 -> frame features
 -> ViT Q/V LoRA
```

のgradient path確認だけである。

### 3.11 optimizer

Stage 4 smokeではAdamWを使用する。

```yaml
optimizer: AdamW
lr: 1.0e-3
weight_decay: 0.0
steps: 1
```

optimizerへ渡すparameter集合は
`requires_grad=True` のLoRA parameterだけとする。

schedulerは使用しない。

このlearning rateは性能最適化値ではなく、
1-step updateを検出するengineering smoke用固定値である。

## 4. Implementation Design

### 4.1 dependency

`requirements.txt` にPEFTを追加する。

```text
peft==0.21.0
```

Transformersは再現性とQ/V module name contractの固定を優先し、
`requirements.txt` で次へpinする。

```text
transformers[torch]==5.17.0
```

PEFTも同様に次へpinする。

```text
peft==0.21.0
```

Stage 4のimplementation / verificationでは、このdependency contractを使用する。

### 4.2 frozen encoder pooler contract

`model/vit/vit_frame_encoder.py` をpoolerなしにする。

同時に既存unit testのmock expectationも

```python
ViTModel.from_pretrained(
    "google/vit-base-patch16-224",
    add_pooling_layer=False,
)
```

へ合わせる。

これは既存feature output `[B,768]` を変えず、
未使用poolerだけを除去する変更である。

### 4.3 LoRA encoder

新しい `ViTLoRAFrameEncoder` は、

1. poolerなしViTをloadする。
2. PEFT `LoraConfig` を作る。
3. `get_peft_model` でQ/VへLoRAを注入する。
4. CLS token featureを返す。

LoRA configは3章の固定値を使用する。

### 4.4 Stage 4 smoke entrypoint

Stage 3 smokeはfreeze / no-grad確認用として保持し、変更しない。

Stage 4専用の新規entrypointを追加する。

推奨:

```text
smoke_activitynet_vit_lora_one_step.py
```

責務:

1. runtime引数からActivityNet dataset rootを受け取る。
2. pinned ActivityNet Adapterのtraining sourceを取得する。
3. strict sequential loaderから先頭1 chunkを取得する。
4. valid frameだけをprocessorへ渡す。
5. ViT+LoRAでvalid featureを作る。
6. `frame_features [16,768]` へscatterする。
7. masked meanで `clip_feature [768]` を作る。
8. engineering-only lossを計算する。
9. optimizer構築前にtrainable parameter集合を監査する。
10. base / LoRA parameterをstep前にsnapshotする。
11. `loss.backward()`。
12. gradientを監査する。
13. `optimizer.step()`。
14. base / LoRA parameter差分を監査する。
15. diagnosticを標準出力へ出す。
16. reader lifecycleを正しくcloseする。

### 4.5 gradient mode

Stage 4 smokeでは、

- `torch.no_grad()` でforward全体を囲まない。
- LoRA pathをdetachしない。
- MaskedMeanClipAggregatorは既存gradient semanticsを維持する。

base ViTは `requires_grad=False` により更新対象外とする。

## 5. Verification Plan

### 5.1 poolerなし + PEFT preflight

2026-09-23に研究サーバのローカルworktreeで実測済み。

```text
ViTModel(add_pooling_layer=False)
 -> pooler is None
 -> PEFT q_proj/v_proj injection succeeds
 -> target count = 24
 -> trainable params = 294,912
 -> trainable tensor count = 48
 -> unexpected trainable = []
 -> CLS feature [2,768]
 -> finite = True
```

さらにローカル `ViTFrameEncoder` でも `pooler is None` を確認した。

このpreflightは成功済みであり、Ambiguity Gateのblocking Aは解消した。
ただしローカル変更のGitHub反映状態は別途implementation開始時に確認する。

### 5.2 unit tests

最低限次を検証する。

1. LoRA encoderがpoolerなしViTをloadする。
2. target modulesが `q_proj/v_proj` だけで24個。
3. `q_proj=12`, `v_proj=12`。
4. LoRA trainable parameter数が294,912。
5. trainable tensor数が48。
6. trainable parameter名に想定外がない。
7. base ViT parameterが `requires_grad=False`。
8. forward output shapeが `[B,768]`。
9. forward outputがfinite。
10. eval時のforward contractを維持する。

### 5.3 one-step synthetic / mocked update test

実データに依存しない短時間testで次を確認する。

1. scalar lossがfinite。
2. backwardが成功。
3. 少なくとも1つのLoRA parameterにfiniteかつnon-zero gradientがある。
4. base parameterにgradientがない。
5. optimizer parameter集合がLoRA trainable parameter集合と一致。
6. optimizer step後、base parameterはstep前と完全一致。
7. optimizer step後、少なくとも1つのLoRA parameterが変化。
8. step後parameterがfinite。

LoRA初期化の性質上、
「全LoRA tensorが1 stepで変化する」は成功条件にしない。

### 5.4 existing regression tests

最低限、現在確認済みの次の範囲を再実行する。

```text
test/model/test_vit_frame_encoder.py
test/model/test_masked_mean_clip_aggregator.py
test/model/test_activitynet_vit_clip_feature.py
test/model/test_50salads_vit_bridge.py
test/config
```

Stage 4開始前にユーザー環境では65 tests passedしているため、
この範囲へ新規回帰を入れない。

### 5.5 real ActivityNet one-chunk smoke

研究サーバ上のActivityNet v1.3 training先頭1 chunk程度を使う。

full dataset trainingは行わない。

確認:

- 16 frames contract。
- valid frame count > 0。
- pixel values `[N_valid,3,224,224]`。
- valid features `[N_valid,768]`。
- frame features `[16,768]`。
- clip feature `[768]`。
- feature / loss / gradient finite。
- target LoRA modules = 24。
- trainable params = 294,912。
- base unchanged。
- LoRA at least one parameter updated。

## 6. Diagnostic Output

Stage 4 smokeでは最低限次を表示する。

```text
implementation branch / HEAD
sequential_loader branch / HEAD
checkpoint
PyTorch version
Transformers version
PEFT version
device

dataset / split
sequence_id
sequence_index
frame_indices
timestamps
valid count

pixel_values shape
valid_features shape
frame_features shape
clip_feature shape

pooler is None
LoRA target names / count
q_proj count
v_proj count

total model parameters
trainable parameters
trainable tensor count
unexpected trainable names

loss
loss finite

LoRA finite gradient count
LoRA nonzero gradient count
base gradient count

base changed parameter count
LoRA changed parameter count
all changed parameters finite
```

## 7. Success Criteria

本specの実装成功は次を全て満たすこととする。

1. implementation repositoryが
   `tamaki-lab/2026_09_ishikawa_sequential-video-lora`。
2. implementation branchが `dev`。
3. base revisionが
   `b6385d87e2e3e82a719d8f4b686b44aa293b1135`
   またはその後継で、差分影響を確認済み。
4. PEFT 0.21.0を使用する。
5. model checkpointが `google/vit-base-patch16-224`。
6. ViT poolerを生成しない。
7. frame featureがCLS `[B,768]`。
8. LoRA targetが `q_proj/v_proj` のみ。
9. target countが24。
10. q_proj 12 / v_proj 12。
11. r=8 / alpha=8 / dropout=0 / bias=none。
12. LoRA trainable parameter数が294,912。
13. LoRA trainable tensor数が48。
14. unexpected trainable parameterが0。
15. base ViTがoptimizer対象外。
16. ActivityNet 16-frame pathを維持する。
17. masked mean contractを維持する。
18. clip featureが `[768]` かつfinite。
19. engineering-only lossがfinite。
20. backwardが成功。
21. 少なくとも1つのLoRA parameterにfinite non-zero gradientがある。
22. base parameterにgradientがない。
23. optimizer対象がLoRA trainable parameter集合と一致する。
24. 1 step後にbase parameterが変化しない。
25. 1 step後に少なくとも1つのLoRA parameterが変化する。
26. 更新後LoRA parameterがfinite。
27. 既存Stage 1-3 / config testsへ新規回帰を入れない。
28. 実ActivityNet先頭1 chunk smokeが終了code 0。
29. long trainingを実行していない。
30. temporal learningを成功条件としていない。

## 8. Explicit Out of Scope

本specでは次を実装・評価しない。

- MoCo。
- BYOL / MAE等の別SSL objective。
- query/key encoder。
- momentum encoder。
- EMA。
- projector / predictor。
- InfoNCE。
- queue。
- full ActivityNet training。
- multi-step training性能。
- scheduler。
- checkpoint save / resume。
- sequential vs shuffle比較。
- reverse評価。
- static-repeat評価。
- future prediction。
- temporal offset prediction。
- GRU / LSTM。
- temporal Transformer。
- video attention。
- LoRA rank / alpha / dropout比較。
- Q/V以外のLoRA target比較。
- Orthogonal Gradients。
- CLIP-ViT。
- downstream VLM / video generation。
- LoRAが時間情報を獲得したという主張。

## 9. Compatibility

維持する:

- Stage 3 ActivityNet input contract。
- `SequentialSample` alignment。
- valid-mask semantics。
- padding row zero semantics。
- `MaskedMeanClipAggregator`。
- frozen `ViTFrameEncoder` の外部forward contract。
- existing 50Salads bridge。
- Hydra / classification training path。
- `main.py` / `main_pl.py` の既存責務。

変更を許可する候補:

- `requirements.txt`: PEFT dependency追加。
- `model/vit/vit_frame_encoder.py`: poolerなし契約へ統一。
- `test/model/test_vit_frame_encoder.py`: poolerなしload contractへ更新。
- 新規LoRA encoder。
- 新規LoRA unit tests。
- 新規Stage 4 smoke。
- 必要最小限のexport。

本Stageのために既存classification trainerへActivityNet LoRA trainingを統合しない。

## 10. Reproducibility

Research Workspace:

```text
haruto2919/research-workspace
main
```

research code baseline:

```text
tamaki-lab/2026_09_ishikawa_sequential-video-lora
dev@b6385d87e2e3e82a719d8f4b686b44aa293b1135
```

Sequential Loader:

```text
tamaki-lab/2026_09_ishikawa_sequential_loader
ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
```

runtime observed:

```text
Python: 3.12.x
PyTorch: 2.14.0+cu130
Transformers: 5.17.0
PEFT: 0.21.0
```

model:

```text
google/vit-base-patch16-224
add_pooling_layer=False
```

dataset rootはruntime引数で渡し、repositoryへ絶対pathをhard-codeしない。

短時間verificationはCPUで実行可能な範囲を基本とする。
CUDAでのtraining性能は本specの成功条件に含めない。

## 11. Failure / Stop Conditions

次の場合は条件を黙って変更せず停止・報告する。

- poolerなしViTへPEFT 0.21.0を注入できない。
- `q_proj/v_proj` target数が24でない。
- Q/V以外が意図せずtrainableになる。
- trainable parameter数が294,912と一致しない。
- feature shapeが768でない。
- feature / loss / gradient / updated parameterにNaN/Infがある。
- base ViT parameterが1 step後に変化する。
- LoRAにgradientが到達しない。
- LoRA parameterが1 step後に1つも変化しない。
- Stage 4成立にMoCoやtemporal moduleが必要になる。
- Stage 4成立にclassification training pathの大幅変更が必要になる。
- Sequential Loader Core変更が必要になる。

この場合、target module、rank、loss、dataset、backboneを勝手に変更しない。

## 12. Ambiguity Gate

### 12.1 Blocking

以下はdraft時点の確認事項。Aは2026-09-23のpreflightで成功済み、
Bは`requirements.txt`をTransformers 5.17.0へpinする方針で確定済み。
両項目は解消済みであり、現在のblocking事項はない。

#### A. poolerなし + PEFT preflight

次を実環境で確認する。

```text
ViTModel(..., add_pooling_layer=False)
pooler = None
PEFT injection succeeds
target count = 24
trainable params = 294,912
unexpected trainable = []
CLS feature [B,768]
finite = True
```

#### B. Transformers version pin

選択肢:

1. `requirements.txt` のTransformersを `5.17.0` にpinする。
2. requirementsは現状維持し、Stage 4 implementation / verification環境だけ
   5.17.0を必須として記録する。

推奨は、Q/V module nameとPEFT compatibilityの再現性を優先し、
`transformers[torch]==5.17.0` へpinすること。

### 12.2 Non-blocking

既存styleに従って決めてよい。

- LoRA encoderのdocstring。
- private helper構成。
- exception type / message。
- test function名。
- smoke CLI option名。
- diagnostic printの細かな整形。
- `model/__init__.py` / `model/vit/__init__.py` のexport位置。

## 13. このStageで主張できること

Stage 4成功後に主張できるのは次まで。

- ActivityNet video clipを既存ViT pathへ入力できる。
- ViT Q/V 24 projectionへLoRAを追加できる。
- base ViTを固定したままLoRAだけをtrainableにできる。
- clip-level scalar lossからLoRAへgradientを流せる。
- 1 optimizer stepでbaseを変えずLoRAを更新できる。
- Stage 3のmasked-mean video representation pathをLoRA付きでも維持できる。

主張できないこと:

- LoRAが時間順序を学習した。
- LoRAがmotionを表現した。
- sequential inputがshuffleより優れる。
- 動画domain adaptationが成功した。
- downstream性能が改善した。
- MoCoが有効である。

## 14. Stage 4後の次段階

Stage 4のGate通過後、別specでStage 5へ進む。

Stage 5候補:

```text
MoCo mechanics
  -> query encoder
  -> key / momentum encoder
  -> projector
  -> EMA
  -> contrastive loss
  -> 必要ならqueue
```

その後、

```text
non-temporal:
ViT+LoRA -> masked mean -> ordinary SSL

temporal:
ViT+LoRA -> temporal relationを明示したobjective
```

を分離して比較する。

## 15. Spec Gate

本specは現在 `approved`。

2026-09-23のユーザー指示「このスペックをapprovedに変更し，実装してください」により、
本spec全体（scope / LoRA config / smoke loss / optimizer / success criteria / dependency pin）
とコード実装、unit tests / 実ActivityNet one-chunk smokeを承認済み。

poolerなし + PEFT preflight、およびTransformers 5.17.0 pin方針は確定済み。

長時間runはStage 4 approved後でも別途許可が必要。

# Implementation Handoff

- approved spec: 本spec（2026-09-23にユーザー承認済み）
- 実装目的: ActivityNet clip経路でbase ViTを固定し、Q/V LoRAだけを1 step更新可能にする
- 基準repository/commit: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@b6385d87e2e3e82a719d8f4b686b44aa293b1135`
- change scope: poolerなしViT contract / PEFT dependency / LoRA encoder / unit tests / ActivityNet one-step smoke
- 対象外: MoCo / temporal objective / long training / scientific performance evaluation
- success criteria: 7章
- 許可されている短時間検証: unit tests / one-chunk smoke
- 長時間run: 未許可
- 未検証予定: temporal learning / downstream evaluation / multi-step training

## 実装・必須検証の完了記録

2026-09-24、指定`dev`へ実装し、新規22件・既存65件のテストと実ActivityNet先頭1 chunkの
CPU one-step smokeが成功した。詳細は[実装・検証記録](../experiments/2026-09-24-vit-lora-one-step-verification.md)を参照。
statusはユーザー指定の`approved`を維持する。長時間run・commit・pushは行っていない。
