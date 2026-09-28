---
project: sequential-video-lora-analysis
record_type: implementation-spec
status: superseded
created: 2026-09-28
last_updated: 2026-09-28
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_base_commit: 4835b5736f0b1dcc9962cbeffd85e880010311ea
superseded_by: 2026-09-28-shared-streaming-moco-protocol-spec.md
---

# Stage 6B: x_trans + same-video past negatives + strict online MoCo

## 1. 目的

Stage 6Aの4-stream round-robin / different-sequence-only negativeを比較baselineとして残したまま、
現在の研究方針として次の3点を導入する。

1. positive key viewを、current clip `x` の各valid RGB frameへ
   **RGB -> GBR channel permutation + horizontal flip** を同一変換として適用した `x_trans` とする。
2. current clipより前に観測した**同一動画のpast clip keyをnegative**として使う。
3. 4-stream round-robinをcurrent pathでは使わず、1本の動画を
   `A1 -> A2 -> A3 -> ...` と未来を先読みせず処理するstrict online pathを追加する。

本Stageはengineering mechanicsの変更・検証であり、representation性能やtemporal learningの
科学的成功を主張しない。

## 2. Authority

現在のユーザー指示（2026-09-28）を最優先入力とする。

- `x_trans`をRGB変換 + 水平変換にする。
- 同じ動画の過去clipをnegativeにする。
- 4-stream round-robinをやめ、strict onlineにする。

RGB変換は直前の会話で定義済みの **RGB -> GBR** を用いる。

関連brainstorm:
- `.research/secretary/notes/brainstorm/2026-09-28-moco-xtrans-online-constraints.md`
- `.research/secretary/notes/brainstorm/2026-09-25-video-lora-learning-and-evaluation-framework.md`

既存baseline:
- Stage 6A spec / experiment
- current implementation: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@4835b5736f0b1dcc9962cbeffd85e880010311ea`

## 3. 現在の実装との差分

### Stage 6A current baseline

- positive: raw current clip vs clip-consistent horizontal flip
- negative: Queueのdifferent sequence_id only
- streams: ActivityNet training先頭4動画
- warm-up: A1 / B1 / C1 / D1
- train order: A2 / B2 / C2 / D2 / A3 ...
- valid negative minimum: 3

### Stage 6B

- positive: raw current clip `x_t` vs `x_trans_t`
- `x_trans_t`: 各valid frameを channel order `[R,G,B] -> [G,B,R]` に並べ替え、その後width方向へhorizontal flip
- padding frameは変更しない
- clip内の全valid frameへ同じchannel permutation / flipを適用する
- negative: loss計算前からQueueに存在する**全past keys**
- Stage 6Bの初期検証では1 sourceのみなので、negativeは同じsequence_idのpast clipsになる
- stream: ActivityNet trainingの先頭1動画
- warm-up: A1をKey-onlyでenqueue
- train order: A2 -> A3 -> A4 -> ...
- future chunkをpositive / negative / warm-upへ使わない
- first training step A2ではA1が1件のvalid negativeとなる

## 4. 実装設計

### 4.1 Two-view transform

`integration/sequential_moco.py` のview生成を拡張する。

Query:
- raw `SequentialSample`

Key:
- sampleをclone
- valid frameのみchannel index `[1, 2, 0]` へ並べ替える
- その結果をwidth dimensionでflip
- invalid / padding rowsは元frameのまま
- sequence_id / sequence_index / frame_indices / timestamps / valid_mask / evaluation_referenceは変更しない

transform後もCPU uint8 `[T,3,H,W]` を維持する。

### 4.2 Negative policy

MoCo componentはStage 6Aの再現経路を壊さない。

negative policyを次の2種類として明示的に扱える最小変更を行う。

- `different_sequence`: Stage 6A互換。current sequence_idと同じQueue entryを除外。
- `all_past`: Stage 6B。old Queueに存在するentryをsequence_idに関係なく全てnegativeにする。

Stage 6B entry pointでは `all_past` を必ず使用する。

current positive keyはloss後・optimizer後・EMA後までenqueueせず、
同一stepのnegativeには入れない。

### 4.3 Strict online consumer

Stage 6Aの4-stream implementationを削除せず、比較再現用に保持する。

Stage 6B用consumerを新規追加または責務が明確な既存training moduleへ追加し、
1 sourceだけを既存Sequential Loader public APIで開く。

処理順:
1. A1を読む。
2. A1の`x_trans`をKey encoderへ通してenqueueする。Query forward / loss / backward / optimizer / EMAは行わない。
3. A2を読む。
4. raw A2 -> Query、x_trans A2 -> Key。
5. old Queue（A1）をnegativeとしてInfoNCE。
6. backward。
7. Query LoRA + Query ProjectorをAdamW update。
8. Key LoRA + Key ProjectorをEMA。
9. pre-EMAで計算済みのA2 positive keyをenqueue。
10. A3, A4...も同じ手順。
11. `max_steps`到達またはsource EOFで終了。

明示的なfuture prefetchは行わない。

### 4.4 Queue

- capacity: 4096を維持。
- metadata `sequence_id / sequence_index` は維持。
- Stage 6B短時間canaryではQueue evictionを起こさない。
- warm-up後Queue count=1。
- training step N完了後Queue count=1+N。
- Queue entryのsequence_indexは0,1,2,...のchronological orderを維持する。

### 4.5 Optimizer / model contract

変更しない:
- checkpoint: `google/vit-base-patch16-224`
- Base ViT frozen
- LoRA: Q/V, r=8, alpha=8, dropout=0, bias=none
- masked mean clip feature
- projector 768 -> 768 -> 128
- L2 normalization
- momentum=0.999
- temperature=0.07
- optimizer: AdamW, lr=1e-3, weight_decay=0
- optimizer target: Query LoRA + Query Projector only
- Key branch gradientなし
- model内部へtraining loopを入れない

## 5. 既存baselineへの影響

Stage 6Aを削除・破壊しない。

- 4-stream round-robin pathはhistorical/comparison baselineとして残す。
- Stage 6Aのdifferent-sequence negative behaviorを再現可能にする。
- Stage 6Bは別entry point / explicit negative policyで起動する。
- existing classification / 50Salads / Stage 1-5 pathを変更しない。

## 6. Success Criteria

### Transform

1. Query frameはrawと完全一致する。
2. Key valid frameは `RGB -> GBR` の後にhorizontal flipされる。
3. padding frameは変更されない。
4. metadataはQuery / Keyで同一。
5. original sampleをmutationしない。
6. 全valid frameへ同じtransform semanticsを適用する。

### Negative policy

7. Stage 6B `all_past` では同じsequence_idのpast keysもnegativeへ含む。
8. A2ではA1だけがnegative。
9. A3ではA1 / A2がnegative。
10. current positive keyはcurrent stepのnegativeへ含まれない。
11. Stage 6A `different_sequence` policyの既存挙動を再現できる。

### Strict online order

12. ActivityNet training先頭1 sourceを使用する。
13. warm-upはA1だけ。
14. training orderはA2 -> A3 -> A4 -> ...。
15. sequence_indexは1ずつ増加する。
16. future chunkを明示的に先読みしてlossへ使わない。
17. max_stepsはtraining updateだけを数え、warm-upを含まない。
18. warm-up後Queue=1。
19. training step N後Queue=1+N。
20. EOF / exception / early stopでReaderを解放する。

### Model update

21. Query Base gradient=0 / parameter不変。
22. Key branch gradient=0。
23. Query LoRA / Projectorにfinite nonzero gradientがある。
24. Query LoRA / Projectorが更新される。
25. Key LoRA / Projectorがm=0.999でEMA追従する。
26. loss / logits / parameters / Queue keysがfinite。
27. update orderは loss(old queue) -> backward -> Query optimizer -> Key EMA -> enqueue。

### Regression

28. Stage 6B unit / integration testが成功。
29. Stage 6Aの主要negative-policy / orchestration testが再現可能。
30. Stage 5 / ViT-LoRA / masked mean関連主要testへ新規回帰がない。

## 7. Explicit Out of Scope

- full ActivityNet pretraining
- GPU長時間run
- checkpoint / resume
- queue size / momentum / temperature tuning
- temporal exclusion window
- negative age weighting / temporal decay
- RGB->GBR以外のchannel permutation ablation
- augmentation recipe比較
- linear probe
- reverse / static-repeat評価
- VideoMAE-style
- Orthogonal Gradients
- LoRA SVD / motion correlationの科学評価
- 動画A終了後に動画Bへ移るfull-dataset stream policyの確定
- representation改善・temporal learningの主張

Stage 6Bの初期strict-online検証は、**1動画内**で完結させる。
複数動画境界でQueueを保持するかresetするかは、full-dataset training specで別途決める。

## 8. Ambiguity Gate

### Blocking

なし。

今回の初期実装は1動画内strict-onlineに限定するため、
複数動画境界のQueue policyを未決のまま実装へ持ち込まない。

### Non-blocking

- 新規training function / fileの具体名
- test function名
- logging JSON fieldの整形
- internal helper分割

既存repository styleへ合わせてよい。

## 9. 実行境界

現在依頼はコード変更の要求として扱う。

許可される短時間検証:
- unit tests
- integration tests
-既存回帰tests
- synthetic / lightweight canary

許可されていないもの:
- full ActivityNet training
- GPU長時間run
- downstream evaluation

Git commit / push / PRはengineering-taskの実行境界に従い、別途明示許可がない限り行わない。

# Implementation Handoff

- approved spec: 本spec
- 実装目的: x_trans（GBR+flip）、all-past same-video negatives、1-stream strict-online MoCoを追加
- 基準repository/commit: tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@4835b5736f0b1dcc9962cbeffd85e880010311ea
- change scope: two-view transform / MoCo negative policy / strict-online consumer / smoke-canary / tests
- 維持条件: Stage 6A再現経路、ViT+Q/V LoRA、masked mean、Projector、EMA、Queue、既存training責務分離
- success criteria: 6章
- 短時間検証: 許可
- 長時間run: 未許可
- 未検証予定: downstream性能、temporal learning、full-dataset sequence-boundary policy


## Superseded

2026-09-28、Stage 6A / Stage 6Bを別consumerとして保持する設計を撤回し、共通Streaming MoCo engineへ統合する方針を採用したため、本specはsupersededとする。

後継spec:
- `2026-09-28-shared-streaming-moco-protocol-spec.md`

本specを新規実装のAuthorityとして使用しない。
