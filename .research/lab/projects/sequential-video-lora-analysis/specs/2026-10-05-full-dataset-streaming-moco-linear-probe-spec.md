---
project: sequential-video-lora-analysis
record_type: implementation-spec
status: approved
created: 2026-10-05
last_updated: 2026-10-06
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_base_commit: 8301ebb2903d467848f2d74b830b13a7f2a090b2
sequential_loader_repository: tamaki-lab/2026_09_ishikawa_sequential_loader
sequential_loader_branch: ActivityNet
sequential_loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
depends_on:
  - 2026-09-28-shared-streaming-moco-protocol-spec.md
---

# Full-dataset single-pass Streaming MoCo + ActivityNet segment Linear Probe Spec

## 1. 目的

既存のStage 6B Streaming MoCoをActivityNet v1.3 `training`全10,024動画へ拡張し、
1回だけ決定的な順序で逐次学習した最終Query LoRAが、元のBase ViTよりも
ActivityNet action分類に有用な表現を作るかをsegment-level Linear Probeで比較できる
再現可能な実行系を実装する。

本specが固定するend-to-end contractは次である。

```text
ActivityNet training 10,024 videos / deterministic single pass
    -> final Query LoRA evaluation snapshot
    -> common ActivityNet annotation-segment manifest
    -> Base ViT / Base ViT + final Query LoRA feature artifacts
    -> Linear Probe x seeds 0, 1, 2
    -> Top-1 / Macro class accuracy mean +/- sample std
    -> local artifacts + Comet experiment/artifact lineage
```

実装成功は、上記pipelineが指定条件で再現・監査できることで判定する。
MoCo-LoRAがBase ViTを上回ることは科学的評価結果であり、実装成功条件には含めない。

## 2. Authorityと根拠

### 2.1 Authority順

1. 2026-10-05までの現在のユーザー指示と、会話で明示的に採用された判断。
2. [Shared Streaming MoCo Protocol Spec](2026-09-28-shared-streaming-moco-protocol-spec.md)。
3. Research Workspaceの実装・検証Evidence。
4. 現行コード `tamaki-lab/2026_09_ishikawa_sequential-video-lora@main@8301ebb2903d467848f2d74b830b13a7f2a090b2`。
5. 探索中brainstorm。brainstorm単独では実装Authorityにしない。

対象project READMEは現在のメイン実装を
`tamaki-lab/2026_09_ishikawa_sequential-video-lora`としているため、本specも同repositoryを使う。

関連記録:

- [学習・評価framework brainstorm](../../../../secretary/notes/brainstorm/2026-09-25-video-lora-learning-and-evaluation-framework.md)。
- [x_trans / online制約 brainstorm](../../../../secretary/notes/brainstorm/2026-09-28-moco-xtrans-online-constraints.md)。
- [Stage 6A multi-step verification](../experiments/2026-09-24-stage6a-moco-multistep-verification.md)。
- [ActivityNet Adapter verification](../experiments/2026-09-23-activitynet-adapter-verification.md)。
- [ActivityNet clip feature verification](../experiments/2026-09-23-activitynet-clip-feature-verification.md)。

### 2.2 確認済み事実

- 現行mainは共通Streaming MoCo engineと、`stream_mode`、`key_transform`、
  `negative_policy`の独立3軸を実装済みである。
- Stage 6B presetは `strict_single / gbr_horizontal_flip / all_past` である。
- current model contractは、frozen `google/vit-base-patch16-224`、Q/V LoRA、
  Masked Mean `[768]`、Projector `768 -> 768 -> 128`、K=4096 FIFO Queue、
  momentum 0.999、temperature 0.07である。
- current optimizerはQuery LoRA + Query Projectorのみを対象にしたAdamW、
  `lr=1e-3`、`weight_decay=0`である。
- current production候補loopは1動画のfresh runだけを扱い、複数動画境界、checkpoint、
  resume、Linear Probe、Comet Artifact lineageは未実装である。
- ActivityNet AdapterのResearch Workspace Evidenceでは、v1.3 split件数は
  `training=10,024 / validation=4,926 / testing=5,044`、training sourceは
  normalized video ID順である。
- Stage 6A Evidenceは実ActivityNetのfresh 10-step / 100-stepまでであり、
  full-dataset性能やtemporal information獲得を示すものではない。

### 2.3 承認済み判断

- full-dataset学習はActivityNet `training`のsingle passとする。
- 動画内はchronological、動画間はAdapterが返す決定的な固定順序とする。
- 動画境界でQuery / Key / optimizer / EMA / FIFO Queueを保持し、再warm-upしない。
- 評価用Query LoRA snapshotは1,000 processed videosごととfinalに保存する。
- resume checkpointは100 processed videosごとに`latest.pt`へrolling保存する。
- Linear ProbeはActivityNet annotation segment単位とし、Base ViTと最終Query LoRAを比較する。
- Base / LoRAは同一manifest、同一segment、同一preprocessingを使う。
- Linear Probeはraw pre-projector `[768]` segment featureを使い、追加正規化しない。
- Probeは3 seeds、Top-1をprimary、Macro class accuracyをsecondaryとする。
- local artifactを再現可能な実体、Cometをtracking / comparison / lineageとして使う。

## 3. Scope

### 3.1 In scope

- ActivityNet `training`全動画を順番に1回処理するproduction Streaming MoCo path。
- 動画境界をまたぐstate保持と、video-boundary exact resume。
- Query LoRA evaluation snapshotとMoCo resume checkpointの分離。
- ActivityNet annotation segment manifestとcanonical label mapping。
- Base / final Query LoRAのsegment feature事前抽出。
- frozen feature上のLinear Probe 6 runsと3-seed集計。
- local artifact、metadata、hash、Comet experiment / Artifact lineage。
- unit / integration / lightweight smokeと既存Stage 6A / 6B回帰。
- full runを起動できるCLI/configと、実行前Gate。

### 3.2 Explicit out of scope

- ActivityNet `validation`や`testing`をMoCo pretrainingへ使用すること。
- 複数epoch、shuffle、data replay、DDP、DataParallel、multi-GPU training。
- validation性能によるMoCo snapshotまたはProbe epochのbest選択。
- Queue size、momentum、temperature、LoRA rank、augmentation、negative policyのtuning。
- temporal exclusion、negative age weighting、queue temporal decay。
- Projector feature `[128]`によるLinear Probe。
- fine-tuning、nonlinear head、class weighting、scheduler、early stopping、PCA、whitening。
- VideoMAE、Orthogonal Gradients、reverse / static-repeat、LoRA SVD、motion correlation。
- temporal orderやmotionを獲得したという主張。
- full-dataset学習またはfull downstream evaluationの、このspec作成時点での実行。

## 4. 維持する既存contract

次を変更しない。

| 項目 | 固定値 |
|---|---|
| base checkpoint | `google/vit-base-patch16-224` |
| frame encoder | frame-wise CLS、poolerなし |
| base ViT | frozen |
| LoRA target | 12 layersの`q_proj` / `v_proj`、計24 modules |
| LoRA | `r=8`, `alpha=8`, `dropout=0`, `bias=none` |
| chunk | 16 frames |
| clip representation | valid frame CLSのMasked Mean `[768]` |
| Query/Key Projector | `768 -> 768 -> 128` |
| projected feature | L2 normalized |
| Queue | metadata付きFIFO、K=4096 |
| momentum / temperature | `0.999 / 0.07` |
| MoCo optimizer | AdamW、`lr=1e-3`、`weight_decay=0` |
| optimizer target | Query LoRA + Query Projectorのみ |
| Key branch | gradientなし、EMAのみ |
| Stage 6B positive | raw Query vs valid-frame `RGB -> GBR` + horizontal flip Key |
| Stage 6B negatives | loss前old Queueの`all_past` entries |
| update order | loss(old Queue) -> backward -> optimizer -> EMA -> enqueue |

既存Stage 6A / Stage 6B preset、現行smoke CLI、Stage 1-5、50Salads、通常classification経路は
再現可能なまま維持する。full-dataset pathを既存canaryの暗黙モードへせず、明示的なproduction入口にする。

## 5. Full-dataset single-pass MoCo protocol

### 5.1 Datasetと順序

- dataset: ActivityNet v1.3。
- split: `training`のみ。
- source count: 厳密に10,024。異なる場合は開始前Gateで停止する。
- source order: `ActivityNetAdapter.sequence_sources("training")`が返す順序をそのまま使う。
- 現行Adapter contractではnormalized video ID順であり、追加shuffleを行わない。
- ordered source ID listを保存し、そのcanonical SHA-256をrun metadataへ記録する。
- MoCo初期化seedはproduction CLIの必須引数とし、値を暗黙defaultにしない。model / projector生成前に
  Python、NumPy、Torch CPU、Torch CUDAへ適用し、resolved config、checkpoint、snapshotへ記録する。
  resume時はfresh runと同じseedを要求する。これはLinear Probeのseeds `0, 1, 2`とは別の条件である。
- 各動画内は`sequence_index=0,1,2,...`とabsolute frame orderを維持する。
- future video / future chunkを明示的に先読みしてloss、positive、negative、warm-upへ使わない。

### 5.2 初回warm-upと動画境界

全runでKey-only warm-upは最初の動画の最初のchunkだけに行う。

```text
video A:
  A1 -> Key-only warm-up
  A2, A3, ... -> training updates

video B:
  B1, B2, ... -> training updates
  （再warm-upなし）
```

動画境界では次をkeepする。

- Query LoRA。
- Query Projector。
- Key LoRA。
- Key Projector。
- AdamW optimizer state。
- EMAの継続状態。
- FIFO Queueのkeysとmetadata。
- global update step。

動画境界でreset、Queue clear、optimizer再作成、Key再初期化、再warm-upを行わない。
最初の動画が1 chunkだけの場合、その動画はwarm-upだけで完了し、次動画の最初のchunkから
training updateを開始する。

### 5.3 1-pass完了条件

- 各sourceをEOFまで処理してから`processed_videos`を1増やす。
- `processed_videos=10,024`かつ次のsourceが存在しない状態をfinalとする。
- `global_update_step`はwarm-upを含まず、optimizer update完了後に1増やす。
- 最後の有効chunkは通常どおり処理する。
- decode / integrity / non-finite / checkpoint restore不一致ではfail-fastで停止する。
- sourceやchunkを自動skipして10,024-video claimを維持したように見せない。
- 失敗時は最後に正常保存された`latest.pt`からのみ再開する。

### 5.4 Process model

- single process、single GPU、one active video stream。
- 1 chunkごとに1 optimizer update。
- DDP / DataParallelは使用しない。
- device、PyTorch / Transformers / PEFT / Sequential Loader version、Git SHA、dataset root、
  ordered source hashをmetadataへ記録する。

## 6. Checkpoint contract

評価用snapshotと学習再開用checkpointは別artifactとする。

### 6.1 Evaluation snapshot: Query LoRA only

保存タイミング:

- `processed_videos`が1,000の倍数になったvideo境界。
- single pass完了時のfinal（10,024 videos）。

保存形式はPEFT-nativeとする。

```text
log/moco/<run_id>/evaluation_snapshots/
  videos-001000_step-<global_step>/
    adapter_model.safetensors
    adapter_config.json
    metadata.json
  ...
  videos-010024_step-<global_step>_final/
    adapter_model.safetensors
    adapter_config.json
    metadata.json
```

snapshotへ含めるmodel weightはQuery encoderのLoRAだけで、Query Projector、Key branch、
optimizer、Queueは含めない。Linear ProbeのMoCo条件はfinal Query LoRA snapshotだけを使用する。
途中snapshotは診断用であり、validationによるbest selectionには使わない。

`metadata.json`の必須項目:

- schema / protocol version。
- implementation repository / branch / commit / dirty-state check。
- Sequential Loader branch / commit。
- base model ID。
- dataset / split / source count / ordered source SHA-256。
- processed videos / global update step / final flag。
- stream mode / key transform / negative policy。
- queue capacity / momentum / temperature。
- LoRA config。
- MoCo optimizer config。
- creation time、local relative path、Comet upload status / artifact version。
- MoCo初期化seed、実device identity、base ViT tensor fingerprint、依存version。

### 6.2 Resume checkpoint: complete training state

保存タイミング:

- 100 processed videosごとのvideo境界。
- final video完了時。

保存先はrunごとに1個のrolling fileとする。

```text
log/moco/<run_id>/resume/latest.pt
```

必須state:

- Query LoRA state。
- Query Projector state。
- Key LoRA state。
- Key Projector state。
- AdamW optimizer state。
- FIFO Queueのcapacity、順序付きkey、`sequence_id`、`sequence_index`。
- `processed_videos`、`next_video_index`、`global_update_step`。
- ordered source IDsまたはそのhashと、次source ID。
- protocol / model / optimizer / dependency / repository provenance。
- Python、NumPy、Torch CPU、Torch CUDAのRNG state（存在するもの）。

resume checkpointは一時fileへ完全保存して検証後にatomic replaceし、途中書き込みの
`latest.pt`を有効扱いしない。保存はvideo境界だけで行い、再開時は`next_video_index`の動画から開始する。
完了済み動画の途中へ戻らず、次動画をskipしない。

restore時は、repository / protocol / base model / LoRA / Queue capacity / optimizer config /
ordered source hash / next source ID / seed / clean code state / device identityを照合する。
untracked fileを含むdirty checkoutや不一致をwarningだけで継続せず停止する。
fresh runとresume runの両方で、既存のbase frozen / parameter set / finite性監査を維持する。

### 6.3 Snapshotとresumeの関係

- evaluation snapshotは小さく、下流評価へ渡す公開契約。
- resume checkpointは同一runを正確に継続する内部契約。
- evaluation snapshotからMoCo学習をresumeしない。
- `latest.pt`をLinear Probeへ直接渡さない。
- 両者のmetadataは同じ`run_id`、`processed_videos`、`global_update_step`でlineageを照合可能にする。

## 7. ActivityNet segment manifest

### 7.1 Annotation source

- ActivityNet Adapterの`evaluation_reference`から、evaluation側helperでv1.3 annotationを解決する。
- Sequential Loader coreへaction label / segment semanticsを追加しない。
- `training` annotationをProbe学習、`validation` annotationを最終評価へ使う。
- `testing`はlabelがないため使わない。
- Loaderが返すraw timestampsをそのまま使い、offset補正、FPS再推定、並べ替えを行わない。

### 7.2 Canonical label mapping

- training / validationの全annotation labelの集合が同じ200 classesであることを確認する。
- 200 labelを文字列昇順で一度だけ並べ、`0..199`へ割り当てる。
- mappingは`label_mapping.json`へ保存し、canonical JSON bytesのSHA-256を記録する。
- Base / LoRA / seeds 0,1,2で同じmappingを使う。

### 7.3 Segment sampleとchunk inclusion

1 annotation instanceを1 sampleとする。重複するlabel / temporal rangeが存在しても、
annotation indexが異なるものは独立sampleとして扱う。

安定したsample ID:

```text
<split>:<normalized_video_id>:<annotation_index>
```

chunkをsegmentへ採用する条件:

- `valid_mask=True`のtimestampが1件以上ある。
- 全valid timestampがfiniteである。
- annotationを`[segment_start, segment_end]`の閉区間として、
  全valid timestampが `segment_start <= t <= segment_end` を満たす。
- padding / invalid frameのtimestampとpixelは判定に使わない。
- 1 frameでもsegment外のvalid timestampがあるchunkは採用しない。
- partial overlapを自動採用しない。

採用chunkが0件のannotationはmanifestからskipし、split / class別のskip countをmetadataへ残す。

### 7.4 Manifest files

```text
log/linear_probe/manifest/<manifest_id>/
  activitynet_linear_probe_manifest.jsonl
  label_mapping.json
  metadata.json
```

manifest rowの必須field:

```json
{
  "segment_id": "training:video-id:0",
  "split": "training",
  "video_id": "video-id",
  "annotation_index": 0,
  "label": "Action label",
  "label_id": 0,
  "segment_start": 1.25,
  "segment_end": 4.75,
  "chunk_indices": [3, 4],
  "chunk_count": 2
}
```

manifest row順は `split -> Adapter source order -> annotation_index`で固定する。
完成file bytesのSHA-256を`segment_manifest_sha256`として保存する。

### 7.5 Dataset Integrity Gate

以下をすべて満たすまでfeature extractionへ進まない。

1. training labels = 200 classes。
2. validation labels = 200 classes。
3. 各classのtraining sample >= 1。
4. 各classのvalidation sample >= 1。
5. duplicate `segment_id` = 0。
6. training / validation `segment_id` overlap = 0。
7. 全rowで`chunk_count == len(chunk_indices) >= 1`。
8. label mapping hashとmanifest metadataが一致する。
9. 同じmanifestを再生成したとき、同一入力・環境でSHA-256が一致する。

Gate FAIL時は停止して結果とskip countsを保存する。chunk包含条件、class集合、splitを
自動的に緩和しない。条件変更は別のユーザー判断とspec更新を必要とする。

## 8. Feature extraction

### 8.1 比較条件

| condition | encoder |
|---|---|
| `base_vit` | frozen `google/vit-base-patch16-224` |
| `moco_query_lora_final` | 同じBase ViT + final Query LoRA snapshot |

MoCo Projectorは使用しない。両conditionで同じmanifest、chunk、processor、valid mask、
Masked Mean、segment aggregation、dtypeを使う。

### 8.2 Feature definition

```text
valid RGB frames
  -> ViT frame CLS [T,768]
  -> valid rowsのMasked Mean
  -> chunk feature [768]
  -> manifestで採用された全chunkの単純平均
  -> segment feature [768]
```

segment featureはpre-projectorのraw float32値をそのまま保存・利用する。
次を行わない。

- L2 normalization。
- z-score standardization。
- centering。
- PCA / whitening。
- conditionごとに異なる後処理。

### 8.3 Feature artifact

```text
log/linear_probe/features/<condition>/<feature_id>/
  training/features.pt
  validation/features.pt
  metadata.json
```

各`features.pt`は最低限次を持つ。

```python
{
    "features": Tensor[N, 768],  # float32, CPU
    "labels": Tensor[N],         # int64, CPU
    "segment_ids": list[str]
}
```

training / validationそれぞれで、`segment_ids`がmanifestの該当splitと順序・件数まで一致することを
load時にも検証する。metadataにはmanifest / label mapping / encoder snapshot / source codeのhashと
provenanceを保存する。

Base / LoRAのmetadataには、dataset root、annotation hash、split別source count / order、manifest / mapping、
base ViT tensor fingerprint、processor config、feature definition、抽出code / dependency provenance、device identityを
共有契約として保存する。Linear Probe開始前に共有契約のcanonical SHA-256と内容が完全一致することを検証し、
意図したQuery LoRA以外の比較軸が異なる場合は停止する。

feature extractionはProbe seedに依存せず、各condition・splitにつき1回だけ実行する。

## 9. Linear Probe protocol

### 9.1 固定hyperparameters

| 項目 | 値 |
|---|---|
| classifier | `Linear(768, 200, bias=True)` |
| encoder / features | 完全freeze、事前抽出済み |
| loss | Cross Entropy、class weightなし |
| optimizer | AdamW |
| learning rate | `1e-3` |
| weight decay | `1e-4` |
| batch size | `256` |
| epochs | `100` |
| scheduler | なし |
| train shuffle | あり |
| validation shuffle | なし |
| early stopping | なし |
| seeds | `0, 1, 2` |

seedが変えるものはclassifier初期値とtraining featureのshuffle順だけとする。
manifest、label mapping、Base / LoRA feature artifactはseed間で共有する。

各seedはfresh classifier / optimizerから100 epochsを完了する。validationはepoch選択へ使わず、
epoch 100終了時のclassifierを1回だけ最終評価する。

### 9.2 Metrics

primary:

- `Top-1 Accuracy (%)`。

secondary:

- `Macro class accuracy (%)` = 200 classそれぞれの正解率の算術平均。

詳細artifact:

- 200 x 200 confusion matrix。
- classごとのaccuracyとsupport。
- training loss / Top-1 history。

conditionごとに3 seedsの算術平均とsample standard deviation（`ddof=1`）を報告する。
個別seed結果を隠さず、集計値と同時に保存する。

### 9.3 Result artifacts

各run:

```text
log/linear_probe/results/<condition>/seed-<seed>/
  probe_classifier.pt
  summary.json
  history.csv
  per_class_accuracy.csv
  confusion_matrix.npy
```

集計:

```text
log/linear_probe/results/comparison/
  aggregate_summary.json
  aggregate_summary.csv
  metadata.json              # mutable Comet status sidecar
```

`aggregate_summary.json`と`aggregate_summary.csv`はimmutable payloadとし、Comet upload statusを
payload自身へ追記しない。両payloadのSHA-256とretry情報は`metadata.json`へ分離する。

`summary.json`はcondition、seed、hyperparameters、Top-1、Macro class accuracy、
sample counts、manifest / mapping / feature hashes、code commit、Comet experiment keyを含む。

## 10. Comet trackingとartifact lineage

### 10.1 原則

- workspace: repositoryの`.comet.config`に従う。
- project: `2026-09-ishikawa-sequential-video-lora`。
- API keyをcode、config artifact、metadata、logへ書かない。
- local artifactを再現可能な実体とし、Cometだけを唯一の保存先にしない。
- local保存完了後にCometへ登録する。
- upload失敗を黙って無視せず、local metadataへstatus / error / retry-neededを記録する。
- network失敗で正常なlocal artifactを削除しない。

### 10.2 Experiments

MoCo pretrainingは1 experimentとして、最低限次を100 optimizer updatesごとに記録する。

- `moco/loss`。
- `moco/positive_similarity`。
- `moco/valid_negative_count`。
- `moco/query_lora_grad_norm`。
- `moco/processed_videos`。
- `moco/global_update_step`。
- Queue count / unique sequence count。

100-step間の値をどう集約して送るかは既存logging styleに合わせてよいが、local audit logは
復旧・診断に必要な粒度を失わない。

Linear Probe experiment名:

```text
lp-v1__base-vit__seed-0
lp-v1__base-vit__seed-1
lp-v1__base-vit__seed-2
lp-v1__moco-query-lora-final__seed-0
lp-v1__moco-query-lora-final__seed-1
lp-v1__moco-query-lora-final__seed-2
```

Probeはepochごとにtrain loss / train Top-1を記録し、final validationのTop-1 / Macro class accuracyを
1回記録する。200 class個別値は比較画面のmetricへ大量展開せず、artifactへ保存する。

### 10.3 Artifact lineage

最低限次のlineageを辿れること。

```text
MoCo experiment
  -> versioned Query LoRA snapshots (`activitynet-moco-query-lora`)
  -> final Query LoRA identifier
  -> common manifest (`activitynet-linear-probe-manifest`)
  -> Base features / LoRA features
  -> six Probe experiments
  -> aggregate comparison artifact
```

推奨artifact名:

- `activitynet-moco-query-lora`。
- `activitynet-linear-probe-manifest`。
- `activitynet-segment-features-base-vit`。
- `activitynet-segment-features-moco-query-lora`。
- `activitynet-linear-probe-results`。

同名artifactはversionを進め、final snapshotをmetadata / alias相当で一意に識別する。
downstream runは入力artifact version、local SHA-256、upstream experiment keyをparameterとして記録する。

## 11. 実装責務と影響範囲

### 11.1 Full-dataset orchestration

`training/`にproduction責務を置き、次を分離する。

- ordered multi-video source scheduler。
- existing per-chunk MoCo updateの再利用。
- video completion accounting。
- evaluation snapshot writer。
- resume checkpoint writer / loader。
- local audit / Comet logging。

現行`training.moco_canary.run_streaming_moco`の「fresh single-video canary」contractを
resume対応へ暗黙変更しない。共通primitiveを抽出する場合も、既存canary wrapperとtestを維持する。

### 11.2 Evaluation

evaluation側へ次の責務を追加する。

- ActivityNet annotation resolution。
- canonical label mapping / manifest generation / integrity Gate。
- Base / Query LoRA feature extraction。
- Linear Probe training / final evaluation / aggregation。
- local / Comet artifact registration。

Sequential Loader core、ActivityNet Adapterのsource semantics、通常classification pipelineへ
label-aware責務を混在させない。

### 11.3 CLI / config

少なくとも次を別entry pointとして明示する。

1. full-dataset MoCo train / resume。
2. manifest build / audit。
3. feature extraction。
4. Linear Probe 3-seed run / aggregate。

具体的なfile名、dataclass / Hydraの採否、private helper名はnon-blockingとする。
ただし全CLIはresolved config、Git SHA、artifact path、hash、Comet keyを標準出力またはmetadataへ残す。

## 12. Failure、overwrite、resume policy

- 既存run directoryを暗黙overwriteしない。
- fresh runとresumeを明示的に分ける。
- `--resume`相当が指定された場合だけ`latest.pt`を読む。
- final済みrunを通常resumeしない。
- hash / protocol / source順不一致なら停止する。
- production MoCo / feature extraction / Probeはuntracked fileを含むclean implementation checkoutだけで行う。
- corrupted snapshot / checkpointをfallbackで部分loadしない。
- manifest / feature / result artifactは同一IDの内容が一致する場合だけ再利用する。
- 内容が異なる既存artifactを同一versionとして上書きしない。
- Comet upload retryはlocal artifactのhashを再検証してから行う。

## 13. Verification plan

### 13.1 Unit tests

- multi-video order: `A1(warm-up), A2..., B1, B2...`。
- 動画境界でQueue / optimizer / Query / Key stateが保持される。
- 2本目以降で再warm-upしない。
- processed video / global step accounting。
- 100-video rolling resume、1,000-video evaluation snapshot、final保存trigger。
- Queue keys / metadataの順序付きserialize / restore。
- evaluation snapshotがQuery LoRAだけをPEFT-nativeで保存・再loadできる。
- resume不一致・破損・mid-video checkpointを拒否する。
- manifestのinclusive containment、padding除外、0-chunk skip、duplicate annotation独立性。
- 200-class Gate、overlap / duplicate / missing class failure。
- Base / LoRA featureのsegment ID・順序一致。
- raw `[768]` featureに追加normalizeがない。
- Probeでencoder / featureが更新されず、classifierだけが更新される。
- seedがclassifier初期値とtrain shuffleだけへ作用する。
- Top-1、Macro class accuracy、confusion matrix、3-seed mean / sample std。
- Comet disabled / upload failureでもlocal artifactが保持される。

### 13.2 Resume equivalence integration test

小さい決定的synthetic multi-video streamで、次を比較する。

```text
uninterrupted run
vs
video boundaryでsave -> process再生成 -> resume
```

最終Query / Key LoRA、両Projector、optimizer state、Queue key / metadata順、
processed videos、global stepが同一になることを確認する。CPU決定的条件ではtensorをexact比較する。

### 13.3 Lightweight smoke

- 現行dependency / cached checkpoint / pinned Sequential Loaderで短いActivityNet prefixを処理する。
- 少なくとも動画境界を1回またぎ、B1がtraining updateになることを確認する。
- video-boundary checkpointからfresh process resumeする。
- 少数annotationのmanifest / Base / LoRA feature / 1-epoch Probe plumbingを確認する。
- smokeは200-class性能やfull run成功を主張しない。

### 13.4 Regression

- Stage 6A presetとhistorical smoke。
- Stage 6B single-video smoke。
- shared protocol / transform / negative selector / Queue / EMA。
- ViT LoRA / Masked Mean / ActivityNet bridge / 50Salads。
- existing config、classification、checkpoint tests。

## 14. Success Criteria

### 14.1 Implementation success

1. ActivityNet training 10,024動画をdeterministic fixed orderでsingle passできる。
2. 最初のA1だけがwarm-upで、動画境界state keepと再warm-upなしが監査できる。
3. 100-video rolling resumeがvideo境界からexact continuationできる。
4. 1,000-video + finalのQuery LoRA snapshotがPEFT-nativeでloadできる。
5. final snapshotはvalidationを使わず、10,024 videos完了で一意に決まる。
6. 共通manifestがIntegrity GateをPASSし、hashで固定される。
7. Base / LoRA featureが同一segment IDsで事前抽出される。
8. 6 Probe runsが固定hyperparametersで完了し、final validationだけを報告する。
9. seed別結果、mean +/- sample std、詳細artifactが再計算可能である。
10. local artifactとComet lineageが相互にhash / version / experiment keyで追跡できる。
11. 既存Stage 6A / 6B、Stage 1-5、通常training経路へ新規回帰がない。
12. failure時に条件を自動緩和・sample skip・best checkpoint選択しない。

### 14.2 Scientific outcome

次は成功・失敗どちらでも正しい研究結果として報告する。

- `moco_query_lora_final`のTop-1がBaseを上回る、同等、下回る。
- Macro class accuracyがTop-1と異なる傾向を示す。
- seed varianceが大きい。
- manifest Gateがstrict containment下でFAILする。
- full runが数値不安定性またはdata integrity errorで停止する。

結果を見て条件を事後変更した場合は、同じprotocol versionの結果として混在させない。

## 15. Compatibility / breaking changes

- publicな既存MoCo presetとsmoke CLIを削除しない。
- current Stage 6A / 6B semanticsを変更しない。
- Sequential Loader coreを変更しない。
- existing `utils/checkpoint.py`のclassification checkpoint contractをMoCo用に流用して壊さない。
- current `.comet.config`を使い、secretをcodeへ移さない。
- 新artifact schemaにはversionを付け、未対応schemaを黙ってloadしない。

本修正で追加したseed / code / device identityと共有feature契約を必須化するため、MoCo protocol、resume、
snapshot、segment feature schemaは`v2`へ上げる。旧`v1` artifactをproductionへ混在・自動migrationせず、
未対応schemaとして停止して新契約で再生成する。既存Stage 6A / 6Bの学習semantics自体は変更しない。

## 16. Ambiguity Gate

### Blocking

なし。

研究主張・比較条件・外部挙動を変える判断は、2026-10-05までの会話で採用済みである。

### Non-blocking

既存styleに合わせて実装者が決めてよい。

- production module / function / CLIの具体名。
- dataclass、Hydra config、plain argparseの使い分け。
- local JSONのindentやprivate helper分割。
- Comet SDKの具体的なupload API呼び出し。
- 100-step Comet metric interval内の送信時点。
- test function名とfixture分割。

これらの選択で、dataset順、checkpoint内容、manifest sample集合、feature値、Probe条件、
metrics、artifact lineageを変更してはならない。

## 17. 実行境界

本specは実装契約の作成であり、研究コードの変更、commit、push、学習runの開始指示ではない。

実装時に許可対象として想定する短時間検証:

- unit tests。
- synthetic integration tests。
- existing regression tests。
- lightweight ActivityNet prefix smoke。
- checkpoint save/loadとComet disabled modeのsmoke。

別途明示承認が必要:

- ActivityNet 10,024 videosのfull MoCo run。
- full manifest / Base / LoRA feature extraction。
- 6本のfull Linear Probe run。
- GPU長時間run、tmux起動、server上のprocess管理。
- 研究コードのcommit / push / PR。

## 18. Spec Gate

本specは **approved**。

根拠:

- full-dataset protocol、checkpoint / resume、segment評価、Probe hyperparameters、artifact、
  Comet運用、failure Gateはユーザーが会話で採用済みである。
- 現在依頼で正式implementation specの作成とResearch Workspace `main`への保存が明示された。
- blockingな未決事項はない。
- current code、既存approved spec、関連Evidenceとの整合を確認した。

approvedは研究コード実装または長時間runの自動許可を意味しない。

# Implementation Handoff

- approved spec: 本spec
- 実装目的: full-dataset single-pass Streaming MoCo、Query LoRA snapshot / exact resume、ActivityNet segment-level Base-vs-LoRA Linear Probe、Comet lineageを実装する
- 基準repository/commit: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@main@8301ebb2903d467848f2d74b830b13a7f2a090b2`
- dependency baseline: `tamaki-lab/2026_09_ishikawa_sequential_loader@ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`
- 変更scope: production multi-video orchestration、checkpoint / resume、manifest、feature extraction、Linear Probe、local / Comet artifact tracking、tests / CLI
- 対象外・維持条件: 3.2章と4章。Stage 6A / 6B既存preset、Loader core、classification pathを維持する
- success criteria: 14章
- 許可されている短時間検証: 17章の短時間検証
- 長時間runの許可状態: 未許可。別途明示承認が必要
- 未検証予定: full-dataset numerical stability、strict manifestの200-class PASS、Base-vs-LoRA downstream性能、temporal information獲得
