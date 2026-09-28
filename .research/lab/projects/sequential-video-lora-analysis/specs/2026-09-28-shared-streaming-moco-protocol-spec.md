---
project: sequential-video-lora-analysis
record_type: implementation-spec
status: approved
created: 2026-09-28
last_updated: 2026-09-28
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_base_commit: 4835b5736f0b1dcc9962cbeffd85e880010311ea
supersedes:
  - 2026-09-28-stage6b-strict-online-moco-xtrans-spec.md
---

# Shared Streaming MoCo Protocol Spec

## 1. 目的

Stage 6Aの4-stream round-robin MoCoと、現在採用するstrict-online MoCoを別々のtraining loopとして実装せず、
**共通Streaming MoCo engineの異なるprotocol設定**として扱う。

共通化する目的は次の3点である。

1. Stage 6Aの既存baselineを再現可能なまま保持する。
2. strict online / x_trans / same-video past negativesを同じ更新契約上で実装する。
3. 今後のablationでstream、augmentation、negative policyを1軸ずつ変更できるようにする。

本specはengineering mechanicsの実装契約であり、representation性能やtemporal learningの科学的成功を主張しない。

## 2. Authority / 入力

現在のユーザー指示（2026-09-28）を最優先入力とする。

- x_transは各valid RGB frameを **RGB -> GBR** へchannel permutationし、その後horizontal flipする。
- same-videoの過去clipをnegativeとして利用する。
- 4-stream round-robinだけに固定せず、strict onlineを選択可能にする。
- Stage 6A / Stage 6Bを別loopとして複製せず、stream等を選択可能な共通実装へ寄せる。
- Stage 6Aはhistorical / comparison baselineとして再現可能に残す。

関連文脈:

- `.research/secretary/notes/brainstorm/2026-09-28-moco-xtrans-online-constraints.md`
- `.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-24-stage6a-moco-multistep-canary-spec.md`
- `.research/lab/projects/sequential-video-lora-analysis/experiments/2026-09-24-stage6a-moco-multistep-verification.md`

旧 `2026-09-28-stage6b-strict-online-moco-xtrans-spec.md` は本specで置き換える。

## 3. Current baseline

implementation repository:

- repository: `tamaki-lab/2026_09_ishikawa_sequential-video-lora`
- branch: `dev`
- baseline: `4835b5736f0b1dcc9962cbeffd85e880010311ea`

維持するmodel / optimizer条件:

- dataset: ActivityNet v1.3 training
- frames_per_chunk: 16
- checkpoint: `google/vit-base-patch16-224`
- Base ViT: frozen
- LoRA target: 12 layersのQ/V、計24 target
- LoRA: r=8、alpha=8、dropout=0、bias=none
- clip representation: frame-wise CLS -> Masked Mean -> [768]
- projector: 768 -> 768 -> 128
- projected feature: L2 normalized
- Queue: metadata付きFIFO、capacity K=4096
- momentum: 0.999
- temperature: 0.07
- optimizer: AdamW、lr=1e-3、weight_decay=0
- optimizer対象: Query LoRA + Query Projectorのみ
- Key branch: gradientなし、EMA追従
- MoCo model内部へdataset iteration / multi-step training loopを入れない

## 4. Protocol model

Streaming MoCoの研究条件を、次の3軸へ分離する。

### 4.1 stream_mode

許可する値:

- `round_robin`
- `strict_single`

#### round_robin

ActivityNet trainingの先頭4 sourceを使う。

sample order:

```text
A1 -> B1 -> C1 -> D1 -> A2 -> B2 -> C2 -> D2 -> A3 -> ...
```

各source内のsequence_indexとabsolute frame orderを維持する。

#### strict_single

ActivityNet trainingの先頭1 sourceだけを使う。

sample order:

```text
A1 -> A2 -> A3 -> A4 -> ...
```

future chunkを明示的に先読みしてloss、positive、negative、warm-upへ使用しない。

本specのstrict-online検証scopeは**1動画内**に限定する。
動画A終了後に動画Bへ移る際のQueue / optimizer / EMA stateのkeep/resetは本specでは決めない。

### 4.2 key_transform

許可する値:

- `horizontal_flip`
- `gbr_horizontal_flip`

Query viewは常にraw current clipとする。

#### horizontal_flip

valid frameをwidth方向へhorizontal flipする。
Stage 6A互換。

#### gbr_horizontal_flip

各valid frameについて次の順序で変換する。

1. channel orderを `[R,G,B] -> [G,B,R]`、index `[1,2,0]` へ変更。
2. width方向へhorizontal flip。

clip内の全valid frameへ同じtransform semanticsを適用する。
invalid / padding frameは変更しない。
original `SequentialSample` をmutationしない。
metadataはQuery / Key間で完全に維持する。

### 4.3 negative_policy

許可する値:

- `different_sequence`
- `all_past`

#### different_sequence

old Queueのうち `entry.sequence_id != current.sequence_id` のentryだけをnegativeにする。
Stage 6A互換。

#### all_past

**loss計算時点でFIFO Queue K=4096に保持されている全past key**をnegativeにする。
sequence_idでは除外しない。

「all past」は無制限履歴を意味しない。
Queue capacityを超えた古いkeyは既存FIFO semanticsに従ってevictされる。

current positive keyはcurrent stepのloss計算前にenqueueしないため、
同一stepのnegativeには含まれない。

## 5. Protocol preset

Stage名は独立algorithm実装ではなく、上記3軸のpresetとして扱う。

### Stage 6A preset

```text
stream_mode      = round_robin
key_transform    = horizontal_flip
negative_policy  = different_sequence
source_count     = 4
warmup_count     = 4
```

Stage 6Aの既存挙動を再現する。

### Stage 6B preset

```text
stream_mode      = strict_single
key_transform    = gbr_horizontal_flip
negative_policy  = all_past
source_count     = 1
warmup_count     = 1
```

現在採用するstrict-online protocolとする。

### Ablation

3軸はStage presetとは独立に組み合わせ可能とする。
これにより、例えば次を比較できる。

- streamだけ変更
- transformだけ変更
- negative policyだけ変更

未知の将来方式向けplugin systemは作らず、現時点で実在する2値 x 3軸だけを扱う。

## 6. Warm-up

warm-upは共通training engineで一般化する。

**active streamごとの最初の1 sampleをKey-onlyでenqueueする。**

- `round_robin`: A1 / B1 / C1 / D1をwarm-up
- `strict_single`: A1だけをwarm-up

warm-up中は次を実行しない。

- Query forward
- InfoNCE loss
- backward
- optimizer step
- EMA update

warm-up完了後にoptimizerを作成し、training updateを開始する。

warm-up数は `source_count` と一致する。
`max_steps` はtraining update数だけを数え、warm-upは含まない。

## 7. Common Streaming MoCo engine

training loop本体はprotocolに依存しない共通実装とする。

各training stepの順序:

1. ordered sample streamからcurrent sampleを1件取得。
2. protocolの `key_transform` でQuery raw / Key transformed viewを作る。
3. Query forward。
4. Key forward。
5. **更新前old Queue**から `negative_policy` に従ってnegativeを選択。
6. InfoNCE lossを計算。
7. backward。
8. Query LoRA + Query ProjectorをAdamW update。
9. Key LoRA + Key Projectorをm=0.999でEMA update。
10. forward済みのdetached current positive keyをQueueへenqueue。
11. audit / logging。

model componentは上記multi-step orchestrationを持たない。

## 8. 責務分離

### Stream scheduler

担当:

- source選択
- Reader lifecycle
- ordered `SequentialSample` の供給
- round-robin / strict-single scheduling
- EOF時の資源解放

担当しない:

- MoCo loss
- optimizer
- EMA
- negative selection
- augmentation

### View transform

担当:

- raw Query / transformed Key viewの生成
- valid frameだけの変換
- original sample / metadata保全

### Negative selector

担当:

- old Queue entriesからprotocolに従ってnegativeを選択

Queue自体はpast representationとmetadataのFIFO保存を担当し、
研究条件としてのsame/different sequence判定を固定的に抱えない。

### Common training engine

担当:

- warm-up
- model forward
- loss
- backward
- optimizer lifecycle
- EMA
- enqueue
- audit
- max_steps

### MoCo model

担当:

- Query / Key encoder
- Projector
- Queue primitive
- InfoNCE primitive
- EMA primitive

dataset iterationやstream schedulingを持たない。

## 9. Interface / compatibility

3軸は共通protocol objectまたは同等の明示的な引数として選択可能にする。
具体的なdataclass / Literal / Enumの選択はnon-blocking implementation detailとする。

既存Stage 6A smokeの実行経路は壊さない。
既存CLIを残す場合は薄いcompatibility wrapperとしてStage 6A presetを共通engineへ渡す。

新しいstrict-online smoke/canaryはStage 6B presetを共通engineへ渡す。

共通engine内部でStage名による条件分岐を増やさず、
挙動差は原則として3軸のprotocol値から決める。

## 10. Strict-online expected behavior

Stage 6B presetでは次を満たす。

warm-up:

```text
A1 -> Key(x_trans) -> Queue
Queue = [A1]
```

A2:

```text
positive = A2 raw vs A2 x_trans
negative = [A1]
training後 Queue = [A1, A2]
```

A3:

```text
positive = A3 raw vs A3 x_trans
negative = [A1, A2]
training後 Queue = [A1, A2, A3]
```

Queue capacityを超える場合はFIFOで最古entryをevictする。

## 11. Success Criteria

### Protocol / architecture

1. `stream_mode`, `key_transform`, `negative_policy` を独立に指定できる。
2. Stage 6A / Stage 6Bが3軸のpresetとして表現される。
3. Stage名による別training loopを複製しない。
4. MoCo model内部にmulti-step training loopを追加しない。

### Stream

5. `round_robin` がA1/B1/C1/D1/A2/B2/C2/D2...を維持する。
6. `strict_single` がA1/A2/A3/A4...を維持する。
7. 各source内のsequence_indexが1ずつ進む。
8. strict-singleでfuture chunkを明示的に先読みしない。
9. EOF / exception / early stopで全Readerを解放する。

### Warm-up

10. round-robinではactive 4 streamのfirst sampleをKey-only warm-upする。
11. strict-singleではA1だけをKey-only warm-upする。
12. warm-up中にQuery forward / loss / backward / optimizer / EMAがない。
13. max_stepsにwarm-upを含めない。

### Transform

14. Queryはraw sampleと完全一致する。
15. `horizontal_flip` はvalid frameのみwidth flipする。
16. `gbr_horizontal_flip` はvalid frameをGBR化してからwidth flipする。
17. padding frameは変更しない。
18. metadataを変更しない。
19. original sampleをmutationしない。

### Negative

20. `different_sequence` はsame sequence_idを除外する。
21. `all_past` はold Queueの全entryをnegativeにする。
22. Stage 6B A2ではA1だけがnegativeになる。
23. Stage 6B A3ではA1/A2がnegativeになる。
24. current positive keyを同stepのnegativeに含めない。
25. Queue capacity K=4096を維持し、overflow時は既存FIFO semanticsを維持する。

### Model update

26. loss / logitsがfinite。
27. Query LoRA / Projectorにfinite nonzero gradientがある。
28. Query Base gradient count=0、run前後で不変。
29. Key branch gradient count=0。
30. Query LoRA / Projectorがoptimizerで更新される。
31. Key LoRA / Projectorがm=0.999のEMA式で追従する。
32. parameter / Queue keyがfinite。
33. update orderが loss(old queue) -> backward -> optimizer -> EMA -> enqueue。

### Regression

34. Stage 6A presetで既存4-stream / different-sequence behaviorを再現できる。
35. Stage 6A関連主要unit / integration testへ新規回帰がない。
36. Stage 5 / ViT-LoRA / masked mean / 50Salads / config主要testへ新規回帰がない。
37. Stage 6B strict-single / GBR+flip / all-pastの新規unit / integration testが成功する。

## 12. 短時間検証

実装時に許可する範囲:

- protocol selection unit tests
- stream ordering / Reader cleanup tests
- transform tests
- negative selector tests
- common engine synthetic multi-step tests
- Stage 6A regression
- Stage 6B synthetic / lightweight canary
- 実ViT / PEFTを使う短時間integration test

実ActivityNetの長時間/full pretraining、GPU長時間run、downstream評価は本spec作成だけでは起動しない。

## 13. Explicit Out of Scope

- full ActivityNet pretraining
- 複数動画を連続処理するstrict-online full-dataset policy
- 動画境界でのQueue / optimizer / EMA reset or keep
- checkpoint / resume
- DDP / multi-GPU
- queue size / momentum / temperature tuning
- temporal exclusion window
- negative age weighting / queue temporal decay
- RGB->GBR以外のaugmentation ablationの実験実行
- linear probe
- reverse / static-repeat evaluation
- VideoMAE-style objective
- Orthogonal Gradients
- LoRA SVD / motion correlation
- representation改善・temporal learningの研究主張

## 14. Ambiguity Gate

### Blocking

なし。

今回固定済み:

- all-past = K=4096 FIFO Queueに現在保持されている全past key
- protocol差分 = stream / transform / negative policyの独立3軸
- Stage 6A / 6B = preset
- strict-online初期scope = ActivityNet 1動画内
- warm-up = active streamごとのfirst sampleをKey-only enqueue

### Non-blocking

既存styleに従って実装者が決めてよい。

- protocol objectをdataclass / Literal / Enumのどれで表現するか
- 新規共通module / helperの具体名
- private helper分割
- logging JSON fieldの細かな整形
- test function名
- Stage 6A compatibility wrapperの内部関数名

上記でprotocol semanticsや既存Stage 6A再現性を変更してはならない。

## 15. Spec Gate

本specは **approved**。

2026-09-28のユーザー指示により、
Stage 6A / Stage 6Bを別loopで保持せず、stream等を選択可能な共通Streaming MoCo実装へ寄せる方針、
およびspec作成前に提示した以下の推奨条件が採用された。

- negativeはK=4096 FIFO Queue内に現在保持されている全past key
- stream / transform / negative policyを独立選択可能にする
- Stage 6A / Stage 6Bをpresetとして扱う
- strict-onlineの今回scopeはActivityNet 1動画内

本spec作成はコード変更、commit、push、長時間実験runの実行指示ではない。

# Implementation Handoff

- approved spec: 本spec
- 実装目的: Stage 6A/6Bを共通Streaming MoCo engine + 3軸protocol設定へ統合する
- 基準repository/commit: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@4835b5736f0b1dcc9962cbeffd85e880010311ea`
- change scope: stream scheduler / key-view transform / negative selector / common training engine / preset / smoke-canary / tests
- 維持条件: Stage 6A再現性、ViT+Q/V LoRA、Masked Mean、Projector、EMA、K=4096 FIFO Queue、model/training責務分離
- success criteria: 11章
- 許可されている短時間検証: 12章
- 長時間run: 未許可
- 未検証予定: full ActivityNet、動画境界policy、downstream性能、temporal learning
