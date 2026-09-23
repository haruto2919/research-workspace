
---
project: sequential-video-lora-analysis
spec_type: implementation
status: implemented
title: Stage 6A ActivityNet ViT-LoRA MoCo multi-step canary
created: 2026-09-24
last_updated: 2026-09-24
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_base_commit: 1cbaa4a0fde2fd196a36feb3eeb0074709b52cd9
implementation_commit: 8304d033b2cf2da7e6636842ef250ed45a5693bb
sequential_loader_repository: tamaki-lab/2026_09_ishikawa_sequential_loader
sequential_loader_branch: ActivityNet
sequential_loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
---

# Stage 6A ActivityNet ViT-LoRA MoCo multi-step canary

## 1. 目的

Stage 5で成立したActivityNet + ViT-LoRA + MoCo v2-styleの1-step mechanicsを、
学習条件をできるだけ変更せず複数stepへ拡張する。

本Stageの中心目的は、4本のActivityNet動画streamを各動画内chronologicalのまま
round-robinで処理し、10-step smokeから100-step canaryまでMoCoの更新契約が
破綻せず継続できることを確認することである。

本Stageはengineering stabilityの確認であり、representation性能、temporal order、
motion獲得、sequential順序の優位性は評価しない。

## 2. Authority / 入力

本specは次を入力とする。

- 2026-09-24のユーザー承認
  - ActivityNet training先頭4 sequence
  - 4 stream round-robin
  - A1 / B1 / C1 / D1をKey-only warm-up
  - 10-step smoke -> 100-step canary
  - Stage 5のMoCo条件を維持
  - Sequential Loader Coreを変更しない
  - MoCo model内部へtraining loopを入れない
- Stage 5 approved spec
  - .research/lab/projects/sequential-video-lora-analysis/specs/2026-09-24-stage5-moco-v2-lora-one-step-smoke-spec.md
- Stage 5 implementation verification
  - .research/lab/projects/sequential-video-lora-analysis/experiments/2026-09-24-stage5-moco-one-step-verification.md
- Stage 6A brainstorm
  - .research/secretary/notes/brainstorm/2026-09-24-stage6a-moco-multistep-canary.md
- current implementation
  - tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@1cbaa4a0fde2fd196a36feb3eeb0074709b52cd9

brainstormは探索記録であり、実装時の正本は本approved specとする。

## 3. 現在の前提

Stage 5では次が成立済みである。

- checkpoint: google/vit-base-patch16-224
- Base ViT frozen
- LoRA target: 12 layersのq_proj / v_proj、計24 target
- LoRA: r=8、alpha=8、dropout=0、bias=none
- Query / Keyは独立parameter storage、初期state一致
- Projector: 768 -> 768 -> 128、Query trainable / Key EMA
- clip representation: frame-wise CLS feature -> Masked Mean -> [768]
- projected representation: [128]、L2 normalized
- Queue: metadata付きFIFO、K=4096、empty initialization
- positive: same clip two-view
- negative: different sequence_id only
- temperature: 0.07
- momentum: 0.999
- optimizer: AdamW、Query LoRA + Query Projectorのみ、lr=1e-3、weight_decay=0
- Stage 5 view: Query raw / Key valid-frame clip-consistent horizontal flip
- update order
  1. Query / Key forward
  2. 更新前QueueでInfoNCE
  3. backward
  4. Query optimizer step
  5. Key EMA
  6. 既に計算済みのdetached current positive keyをenqueue

current ViTLoRAMoCo はoptimizer / EMA / enqueueをcaller責務としており、
Stage 6Aでもこの責務境界を維持する。

## 4. 採用するStage 6A設計

### 4.1 Dataset / source

ActivityNet v1.3 training splitを使用する。

Adapterが返すsource列の先頭4件を固定使用する。

- A = sources[0]
- B = sources[1]
- C = sources[2]
- D = sources[3]

4件のsequence_idは互いに異なることをpreflightで検証する。
random source samplingは行わない。

### 4.2 Loader責務

Sequential Loader Coreは変更しない。

各sourceごとに既存public APIで独立したstreamを作る。

- A stream: A1 -> A2 -> A3 -> ...
- B stream: B1 -> B2 -> B3 -> ...
- C stream: C1 -> C2 -> C3 -> ...
- D stream: D1 -> D2 -> D3 -> ...

各streamは既存のSequentialDataset、SequentialVideoReader、
FixedChunkConfig(frames_per_chunk=16)、build_sequential_dataloader、
sequential_sample_streamを使用する。

各動画内のchunk順、absolute decode frame index順を維持する。

Round-robinはSequential Loaderの機能として追加せず、研究repositoryの
consumer / training orchestration側で実装する。

### 4.3 Warm-up

4 streamのfirst chunkを順にKey branchだけへ通し、Queueへenqueueする。

- A1 -> Key -> enqueue
- B1 -> Key -> enqueue
- C1 -> Key -> enqueue
- D1 -> Key -> enqueue

warm-upでは以下を行わない。

- Query forward
- loss計算
- backward
- optimizer step
- EMA update

warm-up完了時のQueue countはexactly 4とする。
各warm-up keyはdetached、finite、L2 normalizedでなければならない。

### 4.4 Training round-robin

warm-up後は各streamの次chunkからtrainingを開始する。

- step 0: A2
- step 1: B2
- step 2: C2
- step 3: D2
- step 4: A3
- step 5: B3
- 以後同様

max_steps はtraining update数だけを数え、4件のwarm-upは含めない。

各sequence内で観測するsequence_indexは単調に1ずつ増加し、
同一sequenceについてfuture chunkを先読みしてtrainingへ渡さない。

### 4.5 max_steps

同一実装で少なくとも次を実行可能にする。

- 10-step smoke
- 100-step canary

実行時にmax_stepsを指定可能にする。
初期smokeの標準値は10とする。

Stage 6Aにはcheckpoint / resumeを含めないため、各起動はfresh model / optimizer /
Queue / stream stateから開始する。10-step runのstateを100-step runへ引き継がない。

### 4.6 EOF

いずれか1 streamが終了したら4-stream round-robinを終了する。

max_steps到達前にEOFへ達した場合は停止して理由を報告し、
当該target step数のGate通過を主張しない。

EOF、正常終了、exceptionのいずれでも全streamのReader資源を解放する。

### 4.7 MoCo / optimizer契約

Stage 5条件を変更しない。

- checkpoint / processor
- Base freeze
- LoRA target / r / alpha / dropout / bias
- Projector architecture
- Masked Mean
- Query raw / Key horizontal flip
- K=4096
- m=0.999
- T=0.07
- different-sequence-only negative
- AdamW lr=1e-3 / weight_decay=0
- optimizer対象 = Query LoRA + Query Projectorのみ
- Query -> loss -> backward -> optimizer -> EMA -> enqueueの更新順序

100 training step + 4 warm-upはQueue capacity未満なので、
Stage 6Aの正常系ではQueue evictionを期待しない。

### 4.8 Training loopの責務境界

MoCo model内部へtraining loopを実装しない。

self_supervised/moco/ はMoCo algorithm componentの責務を維持する。

含めてよいもの:
- Query / Key encoder管理
- Projector
- InfoNCE
- MetadataQueue
- EMA primitive
- Query parameter取得等のMoCo固有helper

含めないもの:
- multi-step for / while loop
- dataset / stream iteration
- optimizer lifecycle
- max_steps orchestration
- logging cadence
- run termination
- checkpoint / resume
- epoch管理

Stage 6Aのmulti-step orchestrationはconsumer / training側へ置く。
current repositoryにtraining専用packageがないため、必要最小限の新しいtraining責務を
追加してよい。具体的なfile名・private helper分割は既存styleに合わせる
non-blocking implementation detailとする。

既存classification main.py / main_pl.py / SimpleLightningModelへ
MoCo training loopを押し込まない。

## 5. 診断 / logging contract

Stage 6Aでは少なくとも次を観測可能にする。

各training step:
- step
- sequence_id
- sequence_index
- valid frame count
- loss
- positive similarity
- negative similarityまたはnegative logitsの要約
- valid negative count
- queue count
- queue unique sequence_id count
- Query LoRA gradient finite / nonzero
- Query Projector gradient finite / nonzero
- Query LoRA gradient norm
- Query Projector gradient norm
- Query base gradient count
- Key gradient count
- all parameters finite
- all queue keys finite

開始前:
- Query / Key initial state contract
- optimizer parameter set
- Base snapshotまたは同等の不変性監査用state

終了後:
- Query Base不変
- Key Base不変
- Query LoRAが更新されている
- Query Projectorが更新されている
- Key LoRA / ProjectorがEMAにより追従している
- parameter / Queueがfinite

loss、similarity、gradient normのexact値や単調減少は成功条件にしない。

## 6. 実装scope

実装時に許可される変更はStage 6Aへ直接必要な最小範囲とする。

候補:
- 4つのsingle-source SequentialSample streamを安全にopen / closeするconsumer helper
- 4-stream round-robin scheduler / iterator
- Stage 5 update contractを複数step呼び出すtraining orchestration
- Stage 6A用のstandalone smoke / canary entry point
- unit / integration tests
- 必要最小限のexport

既存Stage 5 MoCo componentを再利用し、同等機能を重複実装しない。

ViTLoRAMoCoのalgorithm semanticsをStage 6A都合で変更する必要が生じた場合は、
変更理由とStage 5回帰影響を示して停止する。training loopをmodelへ移すことで
解決してはならない。

## 7. Success Criteria

Stage 6Aの実装成功は次を全て満たすこと。

1. implementation repository / branchが tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev。
2. baseline 1cbaa4a0fde2fd196a36feb3eeb0074709b52cd9 またはその後継で、baseline以降の差分影響を確認済み。
3. Sequential Loaderが ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0。
4. Sequential Loader Coreを変更していない。
5. ActivityNet trainingの先頭4 sourceを使用し、4つのsequence_idが異なる。
6. 1 source = 1 chronological streamとして既存public APIを使用する。
7. warm-up順がA1 -> B1 -> C1 -> D1。
8. warm-up中にoptimizer step / EMA / backwardが発生しない。
9. warm-up完了時Queue count=4。
10. warm-up keyがdetached、finite、L2 normalized。
11. 最初のtraining sampleがA2、以後B2 -> C2 -> D2 -> A3のround-robinになる。
12. 各sequenceのsequence_indexがchronologicalに1ずつ進む。
13. max_stepsはtraining updateのみを数える。
14. Stage 5のcheckpoint / LoRA / Projector / Masked Mean / view / optimizer / K / m / Tを維持する。
15. optimizer parameter setがQuery LoRA + Query Projector exactly。
16. 各training stepでsame sequence_id keyをnegativeから除外する。
17. 各training stepでvalid different-sequence negativeが少なくとも3件存在する。
18. current positive keyを同じstepのnegativeとして使用しない。
19. 各training stepでInfoNCE loss / logitsがfinite。
20. 各training stepでQuery LoRAにfiniteかつ少なくとも1つnon-zero gradientがある。
21. 各training stepでQuery Projectorにfiniteかつ少なくとも1つnon-zero gradientがある。
22. 各training stepでQuery Base gradient count=0。
23. 各training stepでKey branch gradient count=0。
24. update順序がloss(old queue) -> backward -> Query optimizer -> Key EMA -> enqueue。
25. 各step後にcurrent detached keyがQueueへ1件追加される。
26. Stage 6A範囲ではQueue evictionが起きず、training step N後のQueue countが4 + N。
27. Queue key / metadata alignmentが維持される。
28. 全training stepでmodel parameter / gradient / queue keyにNaN / Infがない。
29. run開始前後でQuery Base ViTが不変。
30. run開始前後でKey Base ViTが不変。
31. run終了時にQuery LoRAの少なくとも1 tensorが更新されている。
32. run終了時にQuery Projectorの少なくとも1 tensorが更新されている。
33. run終了時にKey LoRA / ProjectorがEMAによって初期stateから変化している。
34. EOF / early stop / exceptionで全Reader資源を解放できる。
35. 10-step real ActivityNet smokeがtarget 10 stepsへ到達しexit code 0。
36. 10-step Gate通過後、別fresh runの100-step real ActivityNet canaryがtarget 100 stepsへ到達しexit code 0。
37. Stage 5 MoCo unit / integration testsへ新規回帰がない。
38. Stage 1〜4主要回帰およびconfig testsへ新規回帰がない。
39. 既存classification trainer / 50Salads pathへ新規回帰がない。
40. performance改善、loss低下、temporal learningをStage 6A成功条件として主張しない。

## 8. Failure / Stop Conditions

次の場合は条件を黙って変更せず停止・報告する。

- 4 sourceを一意に取得できない
- 4 stream内のchronological orderが崩れる
- A1〜D1 warm-up後にQueue countが4でない
- warm-upでoptimizer / EMA / backwardが実行される
- training順がround-robin contractから外れる
- target step前にいずれかstreamがEOF
- valid different-sequence negative < 3
- same-sequence keyがnegativeへ混入する
- current positive keyがcurrent negativeへ混入する
- Query BaseまたはKey branchへbackward gradientが入る
- optimizerへKey / Base parameterが入る
- Query LoRA / Projectorのgradientが非finite、または両方のどちらかが全てzero
- loss / logits / parameter / gradient / queue keyへNaN / Inf
- Query / Key Baseがrun前後で変化する
- EMA更新順序またはenqueue順序がStage 5 contractと異なる
- Queue metadata / key alignmentが崩れる
- Readerを確実にcloseできない
- Stage 6A成立にSequential Loader Core変更が必要になる
- Stage 6A成立にMoCo model内部training loopが必要になる
- 既存classification trainerへの大幅統合が必要になる
- Stage 1〜5関連testに新規回帰が出る

## 9. Explicit Out of Scope

本Stageでは実装・評価しない。

- full MoCo v2 stochastic augmentation
- RandomResizedCrop / ColorJitter / Blur等のtraining recipe
- scheduler / warmup scheduler
- LR / weight decay tuning
- queue size ablation
- momentum / temperature tuning
- projector ablation
- checkpoint save / resume
- epoch training
- full ActivityNet training
- distributed training / DDP / cross-GPU queue
- sequence-balanced queue
- queue temporal decay
- same-video distant negatives
- strict one-video-at-a-time streaming
- sequence shuffle / random trainingとの比較
- temporal objective
- reverse / static-repeat control
- future prediction
- temporal Transformer / GRU / attention
- Orthogonal Gradients
- representation/downstream性能評価
- VLM / video generation
- 「LoRAが時間情報・motionを獲得した」という研究主張

## 10. Compatibility / Non-breaking Requirements

維持する:
- model/backbones/ / model/aggregators/ / self_supervised/moco/ の現行責務分離
- ViTLoRAFrameEncoder interface
- MaskedMeanClipAggregator interface
- ViTLoRAMoCoのStage 5 algorithm contract
- sequential_moco_bridge.pyのStage 5 view / encode contract
- ActivityNet Adapter / Sequential Loader Core
- 50Salads path
- Hydra classification path
- main.py / main_pl.py / SimpleLightningModelの既存責務
- dependency pins

Stage 6A training orchestrationは既存classification trainerへ混ぜない。

## 11. Reproducibility

Research Workspace:
- haruto2919/research-workspace@main

implementation baseline:
- tamaki-lab/2026_09_ishikawa_sequential-video-lora
- dev@1cbaa4a0fde2fd196a36feb3eeb0074709b52cd9

Sequential Loader:
- tamaki-lab/2026_09_ishikawa_sequential_loader
- ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0

runtime baseline:
- Python 3.12.x
- PyTorch 2.14.0+cu130
- Transformers 5.17.0
- PEFT 0.21.0

dataset:
- ActivityNet v1.3
- split = training
- sources = first 4 returned by the pinned Adapter
- frames_per_chunk = 16

dataset rootはruntime argumentで受け取りhard-codeしない。

Stage 5と同様、random initialization seedを新たな成功条件へ固定しない。
exact loss / similarity / gradient normの再現をGateにしない。
ordering、source選択、shape、parameter集合、update semantics、finite性をcontractとする。

## 12. Ambiguity Gate

### 12.1 Blocking

なし。

ユーザーが2026-09-24に次を明示承認済み。

- 4 streams
- ActivityNet training先頭4 sequence
- round-robin
- A1 / B1 / C1 / D1 warm-up
- 10-step smoke -> 100-step canary
- Stage 5 MoCo条件維持
- EOF時はstream終了、target未達ではGate通過扱いにしない
- Sequential Loader Coreを変更しない
- MoCo model内部へtraining loopを入れない

### 12.2 Non-blocking

既存styleに従って実装者が決めてよい。

- training package / scriptの具体的file名
- round-robin helperをfunction / classのどちらにするか
- diagnostic outputの整形
- gradient norm計算helperのprivate分割
- test function名
- exception class / message
- CLI optionの細かな命名。ただしmax_steps相当の意味を維持する
- CPU / single GPUを選択するruntime device argumentの具体的表現

上記で研究主張、ordering、parameter update contractを変えてはならない。

## 13. このStageで主張できること

成功後に主張できる:
- 4本のActivityNet streamを各動画内chronologicalのままround-robinで処理できる
- Sequential Loader Coreを変更せずconsumer側でmulti-stream schedulingできる
- 4-way warm-up後、different-video negativesを維持して10〜100 stepのMoCo更新を継続できる
- Base ViTを固定し、Query LoRA + Projectorをmulti-step更新できる
- Key LoRA + Projectorをmulti-step EMA追従できる
- Queue / metadata / gradient / parameter finite性を100 step範囲で維持できる

主張できない:
- representation性能が向上した
- lossが収束した
- 100 stepが十分な学習量である
- sequential orderがshuffleより優れる
- round-robinがstrict online streamingと等価である
- LoRAがtemporal order / motionを獲得した
- K=4096 / m=0.999 / T=0.07 / lr=1e-3が最適

## 14. Spec Gate

本specは **implemented**。

2026-09-24のユーザー指示
「推奨を承認します．また，MoCo model内部へtraining loopを入れないことも実装方針に入れます．specが作成可能だったら作成してください」
により、直前のStage 6A推奨条件と追加の責務境界が承認済み入力となっている。

本spec作成は研究コードの実装許可や実験run許可ではない。
実装を開始する場合はengineering-taskへ引き継ぐ。

10-step / 100-step実ActivityNet runやGPU実行は、実装後にユーザーが実行を依頼した時点で
実行境界を再確認する。本spec保存だけではrunを開始しない。

# Implementation Handoff

- approved spec: 本spec
- 実装目的: Stage 5のMoCo 1-step mechanicsを、4 ActivityNet chronological streamsのconsumer-side round-robinで10〜100 training stepへ拡張する
- 基準repository/commit: tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@1cbaa4a0fde2fd196a36feb3eeb0074709b52cd9
- Sequential Loader: tamaki-lab/2026_09_ishikawa_sequential_loader@ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
- change scope: consumer-side 4-stream lifecycle / round-robin / multi-step orchestration / canary entry point / tests / minimal exports
- 維持条件: Stage 5 MoCo semantics、Sequential Loader Core、現行model責務分離、classification trainer、Stage 1〜5 regression
- 禁止: MoCo model内部training loop、Sequential Loader Core拡張、temporal objective、full augmentation、checkpoint / resume、long/full training
- success criteria: 7章
- 許可されている短時間検証: 実装依頼時にunit / integration /既存回帰を実施可能。real 10-step / 100-step runは実行時にユーザー指示を確認する
- 長時間run: 未許可
- 未検証予定: representation性能、temporal learning、strict streaming、sequential-vs-shuffle、augmentation recipe、GPU memory、long-run stability

## 実装・短時間検証の記録

2026-09-24、指定`dev`へconsumer側orchestrationとentrypointを実装した。
新規42件・既存125件のテストが成功し、人工データ10/100-stepと実ViT/PEFT接続を検証済み。
さらに実ActivityNet 10-step smokeと別fresh processの100-step canaryが両方PASSし、
成功条件35・36を含む本specの必須実装検証が完了した。

実装commitは `8304d033b2cf2da7e6636842ef250ed45a5693bb`。
本specのstatusを`implemented`へ更新した。
詳細は[実装・検証記録](../experiments/2026-09-24-stage6a-moco-multistep-verification.md)を参照。
