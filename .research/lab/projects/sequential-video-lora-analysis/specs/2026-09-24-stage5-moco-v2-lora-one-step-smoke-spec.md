---
project: sequential-video-lora-analysis
spec_type: implementation
status: approved
title: Stage 5 ActivityNet ViT-LoRA MoCo v2-style 1-step mechanics smoke
created: 2026-09-24
last_updated: 2026-09-24
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_base_branch: dev
implementation_base_commit: 3b980c8a90e8cad08e78a5ba13cbb3629ec5ae58
implementation_work_branch: dev
sequential_loader_repository: tamaki-lab/2026_09_ishikawa_sequential_loader
sequential_loader_branch: ActivityNet
sequential_loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
hf_checkpoint: google/vit-base-patch16-224
transformers_version: 5.17.0
peft_version: 0.21.0
dataset: ActivityNet v1.3
dataset_split: training
frames_per_chunk: 16
clip_feature_size: 768
projection_size: 128
queue_capacity: 4096
momentum: 0.999
temperature: 0.07
---

# Stage 5 ActivityNet ViT-LoRA MoCo v2-style 1-step mechanics smoke spec

> **Status: approved**
>
> 2026-09-24の壁打ちで、Stage 5をMoCo v2型のmechanicsを現在の
> ActivityNet -> ViT+LoRA -> masked mean経路へ接続し、短時間の1-stepで検証する方針を
> ユーザーが承認した。
>
> 本Stageは自己教師あり学習機構のengineering verificationであり、長時間学習、
> representation性能、LoRAが時間情報・motionを獲得したことを示すものではない。

## 1. 目的

Stage 4では次が成立している。

```text
ActivityNet v1.3 training
  -> Sequential Loader
  -> contiguous 16 frames
  -> ViT base frozen + Q/V LoRA
  -> frame_features [16,768]
  -> MaskedMeanClipAggregator
  -> clip_feature [768]
  -> engineering-only loss
  -> LoRA 1-step update
```

Stage 5ではengineering-only lossをMoCo v2-styleのcontrastive objectiveへ置き換え、
Query / Momentum-Key / Projector / EMA / FIFO Queue / InfoNCEのmechanicsを
実ActivityNetの異なる2 sequenceで短時間検証する。

```text
warm-up sequence A
  -> Key branch
  -> normalized projected key
  -> FIFO Queue

training sequence B
  -> same 16-frame clip
       |-> Query view -> Query ViT+LoRA -> masked mean -> Query Projector -> q
       \-> Key view   -> Key ViT+LoRA   -> masked mean -> Key Projector   -> k+
  -> positive: q vs k+
  -> negative: Queue内のdifferent sequence_id keyのみ
  -> InfoNCE
  -> backward
  -> Query LoRA + Query Projectorを1 step更新
  -> Key LoRA + Key ProjectorをEMA更新
  -> current k+をQueueへenqueue
```

確認したいことは次である。

1. supervised class labelを使わずcontrastive lossを計算できる。
2. Query base ViTはfreezeされたまま、Query LoRAとQuery Projectorだけがgradient updateされる。
3. Key branchにgradientがなく、Key LoRA / Key ProjectorはEMAだけで更新される。
4. Queueは過去keyとsequence metadataを保持し、same-sequence keyをnegativeから除外できる。
5. positive / negative logits、InfoNCE loss、gradient、更新後parameterがfinite。
6. Stage 3/4の16-frame、mask、CLS、masked mean、LoRA contractを壊さない。

## 2. Authority / Evidence

### 2.1 研究方針

2026-09-17 MTGでは、最終方式を先に固定せず、
自己教師ありの動画LoRA学習を一度動かして観測・比較できる開発ループを優先すると決定した。

2026-09-23〜24のbrainstormでは次へ収束した。

- Stage 3のmasked meanはnon-temporal baselineとして維持する。
- Stage 4はLoRA gradient / update経路だけを確認する。
- Stage 5ではMoCo v2型Queueありmechanicsを導入する。
- ordinary MoCoだけでtemporal orderを学習したとは主張しない。
- positiveは同一clipの2 view。
- negativeはdifferent `sequence_id`だけ。
- same-video clipは距離に関係なくStage 5ではnegativeにしない。
- Queue capacityは4096をengineering baselineとして固定する。
- Query Projectorはtrainable、Key ProjectorはEMA追従とする。
- LoRA情報の後続評価はProjector前のclip feature / LoRA parameterを中心にする。

関連:

- `.research/secretary/notes/brainstorm/2026-09-24-stage5-moco-v3-online-suitability.md`
- `.research/secretary/notes/brainstorm/2026-09-23-late-fusion-lora-temporal-learning.md`
- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-17-mtg.md`
- `.research/lab/projects/sequential-video-lora-analysis/experiments/2026-09-24-vit-lora-one-step-verification.md`

### 2.2 現行実装

GitHub上で確認したStage 5の基準revision:

```text
tamaki-lab/2026_09_ishikawa_sequential-video-lora
dev@3b980c8a90e8cad08e78a5ba13cbb3629ec5ae58
```

確認済み構造:

- `ViTLoRAFrameEncoder`: `google/vit-base-patch16-224`、poolerなし、Q/V LoRA。
- LoRA: q_proj 12 + v_proj 12 = 24 targets。
- LoRA trainable: 294,912 parameters / 48 tensors。
- `MaskedMeanClipAggregator`: valid rowだけをmeanしgradientを保持。
- `sequential_vit_bridge.encode_chunk`: valid frameをprocessorへ渡し、`[T,768]`へscatter。
- Stage 4 real ActivityNet smoke: base変更0、LoRA更新、22 new + 65 regression tests成功。
- 現行repositoryにMoCo moduleは存在しない。
- PyTorch / Transformers / PEFT以外のMoCo専用dependencyは不要。

Sequential Loaderは次を維持する。

```text
tamaki-lab/2026_09_ishikawa_sequential_loader
ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
```

## 3. Decision Contract

### 3.1 MoCo variant / scope

Stage 5は **MoCo v2-style mechanics** とする。

採用:

- Query encoder。
- Momentum / Key encoder。
- 2-layer nonlinear projection MLP。
- L2 normalization。
- InfoNCE。
- momentum EMA。
- FIFO Queue。
- different-sequence negative masking。

採用しない:

- MoCo v3 predictor。
- in-batch-negative-only設計。
- distributed all-gather / DDP shuffle。
- full MoCo v2 pretraining recipe。
- long training。

### 3.2 Backbone / LoRA

Stage 4 contractをそのまま維持する。

```text
checkpoint: google/vit-base-patch16-224
pooler: None
frame feature: last_hidden_state[:,0,:] -> [B,768]
LoRA targets: q_proj / v_proj
target count: 24
r: 8
alpha: 8
dropout: 0.0
bias: none
```

Query branch:

- base ViT: frozen。
- Query LoRA: trainable。

Key branch:

- Query encoderから初期stateを完全copyする。
- base ViT: frozen。
- Key LoRA: `requires_grad=False`。
- Key branch全体へbackwardしない。

Stage 5では明示性と検証容易性を優先し、Query / Key encoderは独立instanceとして持つ。
frozen base weight共有によるmemory最適化は対象外。

### 3.3 Clip representation

両branchとも次を固定する。

```text
valid frames
  -> ViT+LoRA CLS frame features
  -> scatter [16,768]
  -> MaskedMeanClipAggregator
  -> h [768]
```

masked meanはorder-invariantであり、Stage 5はnon-temporal baselineである。

### 3.4 Two-view engineering transform

Stage 5はaugmentation性能を評価せず、2-view mechanicsを決定的に検証する。

同一16-frame clipから、同一frame indices / timestampsを保ったまま次を作る。

```text
Query view:
  raw valid frames as-is
  -> existing AutoImageProcessor

Key view:
  raw valid framesをwidth方向にhorizontal flip
  -> existing AutoImageProcessor
```

Key viewのflipはclip内の全valid frameへ同一に適用する。
frameごとに異なるrandom transformをかけない。
temporal crop、frame drop、reverse、shuffleは行わない。

このtransformはengineering smoke専用であり、
MoCo v2のfull stochastic augmentation recipeや最終training augmentationではない。

### 3.5 Projector

Query / Keyの両branchに同一architectureのProjectorを置く。

```text
input:  768
Linear(768, 768)
ReLU
Linear(768, 128)
output: 128
```

BatchNorm、Dropoutは使わない。

この構成はMoCo v2の「backbone representation -> 2-layer nonlinear projection -> low-dimensional contrastive representation」
という構造をViTの768-dim clip featureへ最小適応したengineering固定値である。

parameter count per projector:

```text
Linear(768,768): 768*768 + 768
Linear(768,128): 768*128 + 128
total = 689,024 parameters
4 parameter tensors
```

Query Projector:

- `requires_grad=True`
- optimizer対象。

Key Projector:

- Query Projectorから初期stateを完全copy。
- `requires_grad=False`
- EMAだけで更新。

### 3.6 Contrastive representation

Projector outputへL2 normalizationを適用する。

```python
q = F.normalize(query_projection, dim=-1)
k = F.normalize(key_projection, dim=-1)
```

期待shape:

```text
q:  [128]
k+: [128]
queue key: [128]
```

finiteであり、normは数値誤差内で1。

### 3.7 Positive

positiveは、同一 `sequence_id` / `sequence_index` / frame setから作った2 viewである。

```text
q(current clip, query view)
vs
k+(same current clip, key view)
```

ActivityNetのaction class label、annotation labelは使用しない。

### 3.8 Negative policy

negativeはQueue内のkeyのうち、

```text
queued.sequence_id != current.sequence_id
```

を満たすものだけ。

Stage 5では次をnegativeにしない。

- 同一動画の隣接clip。
- 同一動画の遠距離clip。
- current positive key。

`sequence_id` はsupervised labelではなくデータ由来metadataとしてmaskにのみ使用する。

valid negativeが0個の場合、contrastive updateを黙って実行せず明示的に失敗させる。
Stage 5 real smokeはwarm-upにより少なくとも1 negativeを保証する。

### 3.9 Queue

FIFO metadata queueを使用する。

capacity:

```text
K = 4096
```

entry contract:

```text
normalized key [128]
sequence_id
sequence_index or clip-start metadata
```

初期queueは**empty**とする。
random featureで事前充填しない。

Queueへ保存するkeyはdetach済みでgradientを保持しない。

capacity超過時は最古entryから削除する。

same-sequence keyはQueue自体から削除する必要はなく、
current queryのnegative選択時にmaskする。

### 3.10 Queue warm-up / real smoke sample

real ActivityNet smokeではtraining sourceの先頭2 sequenceを使う。

```text
source A = training sources[0]
source B = training sources[1]
```

各sourceについて、独立したsingle-source SequentialDatasetを構成し、
先頭16-frame chunkだけを読む。
source Aを最後まで走査してsource Bへ到達する方式にはしない。

Warm-up:

```text
source A first chunk
 -> Key view
 -> Key branch
 -> normalized key
 -> Queue enqueue
 -> optimizer updateなし
 -> EMAなし
```

Training step:

```text
source B first chunk
 -> Query / Key views
 -> q / k+
 -> Queue内source A keyをnegativeとしてInfoNCE
 -> 1 optimizer step
 -> EMA
 -> source B k+をenqueue
```

これにより、source B update時のnegativeは過去にenqueue済みのsource Aだけであり、
future keyを使用しない。

### 3.11 InfoNCE

positive logit:

```text
l_pos = q dot k+
```

negative logits:

```text
l_neg_i = q dot k_i-
```

logits:

```text
[l_pos, l_neg_1, ..., l_neg_N] / T
```

temperature:

```text
T = 0.07
```

target index:

```text
0
```

PyTorch cross entropyでscalar lossを計算する。

成功条件:

- at least 1 valid negative。
- logits finite。
- loss scalar / finite。
- supervised action label不使用。

### 3.12 Optimizer

Stage 5 mechanics smokeではStage 4と同じengineering optimizerを使う。

```yaml
optimizer: AdamW
lr: 1.0e-3
weight_decay: 0.0
steps: 1
scheduler: none
```

optimizer対象:

```text
Query LoRA:      294,912 parameters / 48 tensors
Query Projector: 689,024 parameters / 4 tensors
total:           983,936 parameters / 52 tensors
```

optimizer対象外:

- Query base ViT。
- Key base ViT。
- Key LoRA。
- Key Projector。
- Queue。

このlrは性能最適化値ではなく1-step mechanics検出用。

### 3.13 EMA

momentum coefficient:

```text
m = 0.999
```

EMA対象:

- Query LoRA -> Key LoRA。
- Query Projector -> Key Projector。

base ViTは両branchともfrozenなのでEMA対象外。

式:

```text
theta_key = m * theta_key + (1-m) * theta_query
```

1-step training sequenceの順序:

1. Query branch forwardでqを作る。
2. Key branchをno-gradでforwardしk+を作る。
3. **更新前Queue**でInfoNCEを計算する。
4. backward。
5. optimizer.step()でQuery LoRA / Query Projectorを更新する。
6. no-gradでKey LoRA / Key ProjectorをEMA更新する。
7. このstepで計算済みのdetached k+をQueueへenqueueする。

このEMA timingは、次stepのKey forward前に最新Queryへmomentum追従した状態を用意する。
current k+をcurrent query自身のnegativeへ混ぜない。

## 4. Implementation Design

### 4.1 既存コードを維持する

変更しない契約:

- `ViTFrameEncoder`。
- `ViTLoRAFrameEncoder` の外部forward contract。
- `MaskedMeanClipAggregator`。
- `sequential_vit_bridge.encode_chunk` の既存call contract。
- Stage 3 / Stage 4 smoke。
- 50Salads bridge。
- Hydra classification path。
- Sequential Loader Core / ActivityNet Adapter。

必要ならStage 5用helperを追加し、既存bridgeへbreaking changeを入れない。

### 4.2 新規MoCo component

推奨配置:

```text
model/moco/
  __init__.py
  vit_lora_moco.py
```

責務:

- Query / Key encoder初期化。
- Query / Key projector。
- key-side gradient disable。
- L2 normalization。
- metadata FIFO queue。
- different-sequence negative mask。
- InfoNCE logits / loss。
- EMA update。

class名やprivate helper分割は既存styleに合わせてよいが、
上記contractを1箇所から検査可能にする。

### 4.3 Two-view helper

Stage 5用のview生成は既存 `encode_chunk` を壊さず追加する。

必須contract:

- query / keyでframe indicesは同一。
- valid maskは同一。
- query raw framesは変更しない。
- key raw framesだけwidth flip。
- invalid/padding frameをprocessorへ送らない。
- input sample / metadataをin-place変更しない。

### 4.4 Stage 5 smoke entrypoint

新規entrypoint候補:

```text
smoke_activitynet_vit_lora_moco_one_step.py
```

責務:

1. implementation / loader provenanceを表示・監査。
2. dependency versionを監査。
3. ActivityNet training sources[0:2]を取得。
4. source A / Bの先頭chunkをそれぞれ1つだけ読む。
5. Query / Key encoder + projectorの初期一致を監査。
6. warm-up source A keyをQueueへenqueue。
7. source Bで2 viewを作る。
8. q / k+ / valid negativesを作る。
9. InfoNCEを計算。
10. trainable / optimizer集合を監査。
11. gradientを監査。
12. Queryを1 step更新。
13. KeyをEMA更新。
14. current detached k+をQueueへenqueue。
15. parameter差分 / finite / queue状態を監査。
16. readerを成功・失敗の両方でcloseする。

## 5. Verification Plan

### 5.1 Unit tests: Projector / normalization

確認:

- Projector input `[B,768]` -> output `[B,128]`。
- Query Projector 689,024 params / 4 tensors。
- normalized q/k finite。
- q/k norm ~= 1。
- Query Projector trainable、Key Projector frozen。

### 5.2 Unit tests: Query / Key initialization and EMA

確認:

- 初期Query LoRA == Key LoRA。
- 初期Query Projector == Key Projector。
- Query base / Key baseはfrozen。
- Key側全parameterにgradientなし。
- Query update後、EMA前にQuery / Keyの対象parameter差が生じる。
- EMA後、少なくとも1つのKey LoRAまたはProjector tensorが変化する。
- EMA値が式と一致する。
- Key base ViTはEMA前後で変化しない。

### 5.3 Unit tests: Queue / masking

synthetic keyで確認:

- empty start。
- enqueueでcount / FIFO orderが進む。
- K超過でoldestを削除。
- stored keyはdetach。
- different sequence_idだけnegativeに選択。
- same sequence_idは隣接 / 遠距離に関係なく除外。
- valid negative 0ならfail fast。
- current positiveをnegativeへ入れない。
- key / metadata alignmentが崩れない。

capacity境界のunit testでは小さいKを注入可能にしてよい。
production defaultは4096。

### 5.4 Unit tests: InfoNCE / gradient

syntheticまたは軽量ViT configurationで確認:

- positive index 0。
- logits shape `[1, 1+N_neg]`。
- T=0.07。
- loss scalar / finite。
- Query LoRAにfinite non-zero gradient。
- Query Projectorにfinite non-zero gradient。
- Query base gradient 0。
- Key branch gradient 0。
- optimizer集合 = Query LoRA + Query Projector exactly。
- optimizer step後Query LoRAの少なくとも1 tensorが変化。
- Query Projectorの少なくとも1 tensorが変化。
- Query base / Key base不変。
- step / EMA後全parameter finite。

### 5.5 Existing regression tests

最低限Stage 4で成功済みの次を再実行する。

```text
test/model/test_vit_lora_frame_encoder.py
test/model/test_activitynet_vit_lora_one_step.py
test/model/test_vit_frame_encoder.py
test/model/test_masked_mean_clip_aggregator.py
test/model/test_activitynet_vit_clip_feature.py
test/model/test_50salads_vit_bridge.py
test/config
```

Stage 5によりStage 1〜4の既存contractへ新規回帰を入れない。

### 5.6 Real ActivityNet two-sequence smoke

CPUで短時間実行可能な範囲を基本とする。

確認:

- source A != source B。
- 両方first chunk / 16-frame contract。
- Query / Key viewのframe metadata同一。
- Query / Key frame / clip feature finite。
- clip feature `[768]`。
- projected q / k `[128]`。
- warm-up後queue count = 1。
- source B step前 valid negative count >= 1。
- negativeのsequence_id != source B。
- InfoNCE finite。
- Query LoRA / Projectorにgradient。
- Query base / Key全体にbackward gradientなし。
- Query LoRA / Projector更新あり。
- Key LoRA / Projector EMA更新あり。
- 両base ViT変更なし。
- step後queue count = 2。
- all parameters / queue keys finite。
- exit code 0。

## 6. Diagnostic Output

real smokeでは最低限次を表示する。

```text
implementation branch / HEAD
sequential_loader branch / HEAD
checkpoint
PyTorch / Transformers / PEFT versions
device

dataset / split
warm-up sequence_id / sequence_index
training sequence_id / sequence_index
frame_indices / timestamps / valid count

query/key pixel_values shape
query/key frame_features shape
query/key clip_feature shape
query/key projected shape

LoRA targets / count
query LoRA params / tensors
query projector params / tensors
optimizer total params / tensors
unexpected optimizer params

queue capacity / count before warm-up
queue count after warm-up
valid negative count
negative sequence_ids
queue count after training enqueue

positive similarity
negative logits min/max
temperature
loss
loss finite

Query LoRA finite/nonzero gradient count
Query Projector finite/nonzero gradient count
Query base gradient count
Key gradient count

Query base changed count
Query LoRA changed count
Query Projector changed count
Key base changed count
Key LoRA EMA changed count
Key Projector EMA changed count

all parameters finite
all queue keys finite
```

loss値そのものや1-stepでの減少は成功条件にしない。

## 7. Success Criteria

本specの実装成功は次を全て満たすこと。

1. implementation repository / branchが指定どおり。
2. Stage 4 HEAD `3b980c8...` またはその後継で差分影響を確認済み。
3. pinned Sequential Loaderを使用。
4. checkpoint / Transformers / PEFT contractを維持。
5. Query / Key base ViTがfrozen。
6. Query / Key LoRAがQ/V 24 targets、同一config。
7. Query / Key初期LoRA stateが一致。
8. Query / Key Projector初期stateが一致。
9. Projectorが768 -> 768 -> 128、689,024 params。
10. Query optimizer対象が983,936 params / 52 tensors exactly。
11. optimizer対象がQuery LoRA + Query Projector以外を含まない。
12. Key全parameterにbackward gradientがない。
13. masked mean `[768]` contractを維持。
14. projected representationが`[128]`、finite、L2 normalized。
15. positiveがsame clip two-view。
16. supervised ActivityNet class labelを使用しない。
17. negativeがdifferent sequence_idだけ。
18. same sequence_id keyをnegativeから除外。
19. FIFO Queue capacityが4096、empty initialization。
20. warm-up source A keyだけでqueue count=1。
21. source Bのloss時にfuture/current keyをnegativeへ使わない。
22. valid negative >= 1。
23. T=0.07、InfoNCE finite。
24. Query LoRAにfinite non-zero gradient。
25. Query Projectorにfinite non-zero gradient。
26. Query base gradient=0。
27. Query LoRAが1 stepで少なくとも1 tensor更新。
28. Query Projectorが1 stepで少なくとも1 tensor更新。
29. Query baseが不変。
30. m=0.999 EMA後、Key LoRA / Projectorの少なくとも1 tensorが期待式どおり変化。
31. Key baseが不変。
32. current detached k+ enqueue後queue count=2。
33. 全parameter / queue keyがfinite。
34. 指定のStage 1〜4 regression testsへ新規回帰なし。
35. 実ActivityNet two-sequence CPU smokeがexit code 0。
36. long trainingを実行していない。
37. temporal learning / performance改善を成功条件にしていない。

## 8. Explicit Out of Scope

本Stageでは実装・評価しない。

- MoCo v3。
- Predictor。
- in-batch-negative-only training。
- full MoCo v2 stochastic augmentation recipe。
- random resized crop / color jitter / blur等のtraining recipe最適化。
- multi-step / epoch training。
- full ActivityNet training。
- scheduler / warmup scheduler。
- checkpoint save / resume。
- distributed training / DDP / cross-GPU queue。
- batch shuffle for BatchNorm。
- queue size ablation。
- momentum / temperature tuning。
- projector dimension tuning。
- projectorあり/なしablation。
- same-video distant negative。
- temporal exclusion window。
- sequence-balanced queue。
- queue temporal decay。
- sequential vs random/shuffle comparison。
- reverse / static-repeat評価。
- temporal objective。
- future prediction。
- temporal Transformer / GRU / video attention。
- Orthogonal Gradients。
- downstream VLM / video generation。
- 「LoRAが時間情報を獲得した」という研究主張。

## 9. Compatibility / Non-breaking Requirements

維持する:

- existing Stage 1〜4 smoke / tests。
- existing `ViTLoRAFrameEncoder` forward interface。
- existing `MaskedMeanClipAggregator`。
- ActivityNet Adapter / Sequential Loader Core。
- 50Salads path。
- Hydra classification path。
- `main.py` / `main_pl.py` の既存責務。
- current dependency pins。

既存classification trainerへMoCoを押し込まない。
Stage 5は独立MoCo component + smokeとして追加する。

## 10. Reproducibility

Research Workspace:

```text
haruto2919/research-workspace@main
```

implementation baseline:

```text
tamaki-lab/2026_09_ishikawa_sequential-video-lora
dev@3b980c8a90e8cad08e78a5ba13cbb3629ec5ae58
```

Sequential Loader:

```text
tamaki-lab/2026_09_ishikawa_sequential_loader
ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
```

runtime baseline observed:

```text
Python 3.12.x
PyTorch 2.14.0+cu130
Transformers 5.17.0
PEFT 0.21.0
```

dataset rootはruntime argumentで受け取りhard-codeしない。

Stage 5はexact loss値の再現を成功条件にしない。
Projector / LoRAのrandom initialization seed値ではなく、shape、parameter集合、gradient、
update、EMA式、queue metadata、finite性をcontractとする。

## 11. Failure / Stop Conditions

次の場合は条件を黙って変更せず停止・報告する。

- Query / Key初期stateを一致させられない。
- Query / Key LoRA target contractがStage 4と異なる。
- Query baseまたはKey branchに意図しないgradientが入る。
- optimizerへKey / base parameterが入る。
- Projector parameter countが契約と一致しない。
- valid negativeが0。
- same-sequence keyがnegativeへ混入する。
- current positive keyがcurrent negativeへ混入する。
- q / k / logits / loss / gradient / updated parameter / queue keyにNaN/Inf。
- Query LoRAまたはQuery Projectorが1 stepで1つも更新されない。
- Query / Key baseが変化する。
- EMAがm=0.999式と一致しない。
- Queue key / metadata alignmentが崩れる。
- Stage 1〜4の既存testに新規回帰が出る。
- Stage 5成立にSequential Loader Core変更が必要になる。
- Stage 5成立にclassification trainerの大幅変更が必要になる。
- mechanics smokeにfull trainingが必要になる。

## 12. Ambiguity Gate

### 12.1 Blocking

**なし。**

2026-09-24のユーザー承認により、次は確定済み。

- Stage 5をMoCo v2-style 1-step mechanics smokeに限定。
- trainable Query Projector / EMA Key Projector。
- same-clip two-view positive。
- video-consistent view transform。
- different sequence_id-only negatives。
- empty FIFO Queue / warm-up。
- K=4096。
- m=0.999。
- T=0.07。
- q/k L2 normalization。

read-only実装調査を踏まえ、Projectorの具体形は
`768 -> 768 -> 128` の2-layer MLPとして固定した。
Stage 5がperformance experimentではなくmechanics smokeであるため、
このprojection dimensionはengineering contractであり研究結論ではない。

two-view transformも、full augmentation recipeを導入せず
`raw vs clip-consistent horizontal flip` の決定的engineering transformへ固定した。

### 12.2 Non-blocking

既存styleに従って決めてよい。

- MoCo class / helperの細かな命名。
- Queue内部をring buffer / dequeのどちらで実装するか。
- metadata内部表現。
- exception class / message。
- private helper分割。
- diagnostic printの細かな整形。
- export位置。
- test function名。

ただし外部contract、parameter count、mask semantics、更新順序は変更しない。

## 13. このStageで主張できること

成功後に主張できる:

- ActivityNet clipから2 viewを作り、ラベルなしでMoCo-style contrastive lossを計算できる。
- Base ViTを固定したままQuery LoRA + Projectorをcontrastive lossで更新できる。
- Key LoRA + ProjectorをgradientではなくEMAで追従できる。
- 過去different-video keyをQueue negativeとして利用できる。
- same-video keyをnegativeから除外できる。
- Stage 4のvideo / LoRA pathをMoCo mechanicsへ接続できる。

主張できない:

- MoCo学習でrepresentation性能が向上した。
- LoRAがtemporal order / motionを獲得した。
- sequential orderがshuffleより有効。
- K=4096 / m=0.999 / T=0.07が最適。
- Projectorありの方が研究目的に最適。
- full ActivityNet trainingが安定する。

## 14. 次段階

Stage 5 Gate通過後に別specで扱う候補:

```text
Stage 6:
ActivityNet + ViT-LoRA + MoCo のmulti-step / training loop
  -> stochastic video-consistent augmentations
  -> checkpoint / logging
  -> 実用batch / GPU memory確認

その後:
sequential vs shuffle
temporal objective
ordered / reversed / static-repeat controls
LoRA / pre-projector feature解析
```

Stage 5でfull trainingへ自動移行しない。

## 15. Spec Gate

本specは **approved**。

2026-09-24のユーザー指示
「推奨を承認します．specを作成してください」
により、直前に提示したStage 5推奨方針を承認済み入力としてspec化した。

実装コードの変更は本spec作成では行わない。
実装を開始する場合は `engineering-task` へ引き継ぐ。

短時間のunit tests / real two-sequence CPU smokeは実装検証として許可される。
multi-step / long training / GPU full runは別途ユーザー許可を必要とする。

# Implementation Handoff

- approved spec: 本spec。
- 実装目的: Stage 4のActivityNet ViT-LoRA clip経路へMoCo v2-style Query/Key/Projector/EMA/Queue/InfoNCEを追加し、1-step mechanicsを検証する。
- 基準repository/commit: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@3b980c8a90e8cad08e78a5ba13cbb3629ec5ae58`
- change scope: 新規MoCo component / two-view helper / unit tests / ActivityNet two-sequence one-step smoke / 必要最小限のexport。
- 対象外: long training / MoCo v3 / temporal objective / performance evaluation / sequential-vs-shuffle。
- success criteria: 7章。
- 許可されている短時間検証: unit tests / Stage 1〜4 regression / real ActivityNet two-sequence CPU smoke。
- 長時間run: 未許可。
- 未検証予定: representation性能、temporal learning、GPU memory、multi-step stability、projector / queue hyperparameter比較。
