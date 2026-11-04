---
project: sequential-video-lora-analysis
date: 2026-11-04
record_type: implementation-cpu-verification
status: cpu-tests-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_base_commit: 50ff715773137388ba51850532660d26daa4501a
loader_repository: 2026_09_ishikawa_sequential_loader
loader_branch: feature-stride-ddp
loader_base_commit: 85a39c2
code_state: committed
research_commit: eb2e8247a5c4f230472288e88d5a977135a57351
loader_commit: 4026b06f684dc7762b37dd06bba9ab869c479d9e
spec: ../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md
---

# sequential-loader（stride 16）でのclip MoCo LoRA学習：実装とCPU検証

[承認済みspec](../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md)を実装した。specの指示どおり、新しいbranchは切らず、次の2つのbranchの上に積んでいる。

- 研究コード：`feature-vit-moco@50ff715`
- loader：`feature-stride-ddp@85a39c2`

結果は次のとおり。

- **CPUテストS1〜S8はすべてPASS**した。
- 研究コードの全テストは36 failed / 1003 passedで、失敗したテストの集合は変更前と**完全に一致**した。
- loaderの全テストは214件すべてOK。`sample_transform` を指定しないときの出力は、`85a39c2` と**bitwise一致**した。
- 2026-11-04にcommitした：研究コード `feature-vit-moco@eb2e824`、loader `feature-stride-ddp@4026b06`（pushはしていない。内容はCPU検証時と同じ）。GPU（G1・G2）は実行していない。

## 変更したファイル

### loader（`feature-stride-ddp`）

| ファイル | 内容 |
|---|---|
| `src/sequential/multistream.py`（+87／−11） | L1：`build_multistream_loader(..., sample_transform=None)` を追加した。<br>・W>0ではworkerの中で、W=0ではmain processで、各chunkに変換をかける。<br>・結果は `MultiStreamBatch.outputs`（長さB）に入れる。`samples` は、指定しない場合と同じ内容のまま。<br>L2：workerは、epochの最初に `transform_seed(seed, epoch, rank, worker)`（sha256から作る63 bitの値）でtorch・random・numpyの乱数を初期化する。使ったseedは `last_epoch_summary["transform_seeds"]` に記録する。<br>修正（下の「実装中に見つけた問題」）：途中で止めたepochの `steps` を、返したbatchの数にした |
| `tests/test_multistream_transform.py`（新規、7件） | ・変換なしなら `outputs` はNoneで、`samples` は変わらない<br>・`outputs` は各sampleの変換結果<br>・workerの乱数はseed・epoch・rank・workerで決まる（手で再現した値と一致する）<br>・seedの値と、記録したseed<br>・途中で止めたepochのsteps<br>・呼び出せないtransformは拒否する |

### 研究コード（`feature-vit-moco`）

| ファイル | 内容 |
|---|---|
| `scripts/vit_moco/activitynet_sequential.py`（新規、79行） | I2：`SequentialClipTransform`。chunkを (view q, view k, 埋めたフレーム数) に変える。4枚に満たないclipは、最後の有効フレームで埋めてから、今のaug-plus（`ClipTransform`）をかける。<br>`video_sources`：`sl.ActivityNetAdapter` の `training`。CPUテストだけは `video_dir` のmp4を使う。<br>`build_loader`：stride 16、T=4、B＝local batch、W＝`data.num_workers`、loader seed＝`training.seed`。<br>I5：data記録に、loaderのcommit、stride、B、W、prefetch、動画数、source一覧のsha256、埋め方、epochの終わり方を残す |
| `conf/moco_vit_seq_activitynet_clip.yaml`（新規） | I1：`moco_vit_lr_activitynet` に、次を加えたpreset。<br>・`data.source=activitynet_sequential_clips`、`data.clip={frames: 4, frame_stride: 16}`、`data.sequential`（dataset_root、split、video_dir、prefetch_factor 2）<br>・gradient checkpointing、LR 0.0003<br>・checkpoint `[1000,…,6000,6255]`、probe `[0,1,2,10,100,256,1000,…,6000,6255]`<br>・profile `seq_activitynet_clip`、出力先 `log/moco_vit_seq_activitynet_clip/…` |
| `scripts/vit_moco/moco_vit_ddp_smoke.py`（+183／−82） | I1：新しいsourceは、このpresetからだけ受け付ける。`data.clip` と `data.sequential` があるか検査する。clipのモデル・aug・identityは、両方のclip sourceで同じにした。<br>I3：sequentialのときは、`DistributedSampler` と `steps_per_epoch` を使わない `train_sequential` で回す。loaderが尽きたら次のepochへ進む。<br>I4：LRは3e-4で一定にした（identityの `lr_schedule` も一定と記録する）。<br>I5：metricsに、rankごとの `padded_frames`・`data_wait_sec` を加えた。epochごとのloader summaryを `loader_epochs.rank{r}.jsonl` に、epochの記録をrank reportに残す。<br>I7：checkpointに、epochとepoch内のstepをそのまま保存する。resumeでは、そのepochを最初から読み直し、消費済みのstepを捨ててからRNGを戻す。捨てるのにかかった時間を `resume_skip` に記録する。resumeしたrunでは、start stepより後のcheckpointだけを保存する。<br>1stepの処理（data errorの共有、更新、Queueの検査、metrics、checkpoint、probe）は、既存の経路と共通の関数 `run_step` に移した（既存の経路の挙動は変えていない） |
| `scripts/vit_moco/query_collapse_diagnostic.py`（+51／−5） | I6：profile `seq_activitynet_clip` を追加した。<br>・GPUで受け付けるのはG1（10step）とG2（6,255step）だけで、それぞれ決めたprobeとcheckpointでなければ止める。<br>・LoRA、LR 3e-4、T=4、stride 16、gradient checkpointing、ActivityNet training、`video_dir` なし、prefetch 2以外は止める。<br>・resumeを受け付けるのはこのprofileだけで、GPUでは6,255stepのrunだけ。<br>probeは、両方のclip sourceで静止clipにする。<br>resumeしたrunのsummaryは、step 0とstart stepより後のprobe・updateだけを数える |
| `test/model/test_vit_moco_seq_activitynet_clip.py`（新規、31件） | S1〜S7 |
| `test/model/test_vit_moco_lr_activitynet_clip.py`（+2／−1） | `PRESETS` の一覧をそのまま比べているテストに、新しいpresetを加えた |

### 実装中に見つけた問題（specの範囲内で直した）

1. **diagnosticsはresumeを受け付けない作りだった**（「probe_step 0は最適化の前」のため）。
   - D8（再開できるようにする）を満たすため、このprofileだけresumeを受け付けるようにした。
   - resumeしたrunでは、**組み立てたばかりのモデル（学習前の状態）でstep 0のprobeを先に取り直し**、そのあとでcheckpointを読み込む。こうすると、step 0からの変化量（`from_step0`）が、元のrunと同じ基準で計算できる。
   - summaryで数えるprobe・updateは、step 0とstart stepより後のものだけにした。
2. **loaderのsummaryの数え間違い**：途中で止めたepochの `steps` が、返したbatchの数より1少なかった。stepを数える位置を、batchを返す前に移した（loaderにテストを1件追加）。

## CPUテスト

| # | 内容 | 結果 |
|---|---|---|
| S1 | ・各stepは、streamごとに1 clip<br>・clipのframe indexは k, k+s, k+2s, k+3s で、同じ枠の次のclipは k+4s から始まる<br>・viewは `[4, 3, H, W]`（batchでは `[B, 4, 3, H, W]`）<br>・2 rankで違う動画を読む（S4のloader summary） | PASS |
| S2 | ・最後のclipの有効フレームが1・2・4枚の場合と、4フレーム分の長さがない動画（4フレーム）で、足りない位置は最後の有効フレームと同じ画素・同じaugになる<br>・埋めた数を返し、runのmetricsに記録する | PASS |
| S3 | ・同じフレームを4回並べたclipでは、1つのviewの4フレームがすべて同じになる（1 viewに1回の乱数）<br>・W=2のloaderの出力は、2回作っても同じ | PASS |
| S4 | ・7stepのrun（2 rank×2 stream）で、epochの数・steps・終わり方（最後以外は `exhausted`）が両rankで一致し、metricsのepochとも合う<br>・rankごとにtransform seedが違い、読む動画が重ならない<br>・epochが変わると割り当てが変わる | PASS |
| S5 | 全stepでLRは3e-4のまま（epochをまたいでも） | PASS |
| S6 | ・epochの境目（step 2）と途中（step 5）のcheckpointから再開した<br>・その後のstepのmetrics（入力のsha256、loss、Queueなど。時間の項目を除く）と、最終状態のsha256が、止めずに学習した場合と**bitwise一致**した<br>・summaryはPASSで、step 0とstep 7のprobe特徴も一致した<br>・checkpointはstart stepより後だけを保存する | PASS |
| S7 | ・presetの内容と出力先が正しい<br>・GPUの長さ（10、6,255）、probe、checkpoint、LR、T、stride、学習方式、split、`video_dir`、prefetch、batch、workerが違えば止める<br>・ほかのpresetではこのsourceを拒否する。ほかのdiagnosticsは、今までどおりresumeを拒否する<br>・GPUのresumeは6,255stepのrunだけ | PASS |
| S8 | ・研究コードの全テスト（`.venv/bin/python -m pytest -rf -q test`、GPUが見える状態）：**36 failed / 1003 passed**。変更前は36 failed / 972 passed（`50ff715` と同じ内容のときの記録）。増えた31件は新しいテストで、失敗IDの集合は**完全に一致**<br>・loaderの全テスト：214件OK（変更前は207件、追加は7件）<br>・`sample_transform` なしの出力（batch・summary）を、`85a39c2` を `git archive` で取り出したものと比べた。W=0／2 × stride 1／3 × epoch 0／1で**bitwise一致** | PASS |

CPUでの小さな試し実行（7step）では、tiny ViTのlossは0.24→2.20になった（K=8、batch 4）。これは配線の確認のためで、lossの値に意味はない。

## まだ行っていないこと（それぞれ別に許可が必要）

- 研究コード（`eb2e824`）とloader（`4026b06`）はcommit済み。GPU runは、研究コードがcleanなら実行できる。loaderのcommitは、data記録（`sequential_loader` のgit情報）に残る。
- **G1**：10step（VRAM、データ待ち、decodeの速さ）。
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_seq_activitynet_clip \
      model.train_mode=lora training.max_steps=10 'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'
  ```
- **G2**：6,255step（約19時間の見込み）。

## 互換性

- 既存のpreset、data source、学習経路、checkpoint形式は変えていない。既存テストの結果も変わらない（S8）。
- 既存の経路の1stepの処理は `run_step` に移したが、処理の順番と中身は同じである。resumeのテスト（bitwise一致）も、変わらずPASSしている。
- 依存パッケージは追加していない。
