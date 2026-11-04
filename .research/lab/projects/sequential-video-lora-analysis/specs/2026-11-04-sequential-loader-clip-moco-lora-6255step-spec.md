---
project: sequential-video-lora-analysis
date: 2026-11-04
spec_type: implementation-experiment
status: implemented
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
base_commit: 50ff715773137388ba51850532660d26daa4501a
loader_repository: 2026_09_ishikawa_sequential_loader
loader_branch: feature-stride-ddp
loader_base_commit: 85a39c2
source_brainstorm: ../../../../secretary/notes/brainstorm/2026-11-04-sequential-loader-long-training-loss-check.md
---

# sequential-loader（stride 16）でclip MoCo LoRAを6,255step学習し、lossが下がるかを確かめるspec

**Status: implemented（2026-11-04。CPUテストS1〜S8はPASS、研究コード `feature-vit-moco@eb2e824`・loader `feature-stride-ddp@4026b06` としてcommit済み。G1は工学PASS（データ待ち約3.2秒/step）。§10aの追記（D12・D13）は実装し、CPUテストS1〜S9はPASS。研究コード `f16a485`・loader `b114c17` としてcommit済み。G1の再実行は工学PASS（データ待ち約2.3秒/step）。G2はstep 1,601（epoch 0の終わり）で失敗した。§10bの追記（D14、loader workerをforkserverで起動する）を実装し、CPUテストS10〜S12はPASS。研究コード `1a37860`・loader `ce4dcbc` としてcommit済み。G1b・G1の再実行と、G2のやり直しは未実施。[実装記録](../experiments/2026-11-04-sequential-loader-clip-moco-lora-implementation-cpu-verification.md)）**。**Status: approved（2026-11-04、ユーザーが承認）**。blockingな未決事項はない（B1・B2は2026-11-04に決定、§10）。実装は、別に指示を受けてから行う。研究コードの実装とGPU runは、specが承認された後に、それぞれ別に許可を得てから行う（§9）。

## 1. 目的と背景

- これまでの学習では、動画からclipを取り出すときに、manifestの時刻ごとにseekする方法を使っていた（`ActivityNetClipDataset`）。
- 今回は、sequential-loader（動画を先頭から順番に読む）でclipを作り、MoCo LoRAを学習する。目的は、**lossがきちんと下がるか**を確かめること。
- 学習量は、公式MoCoとの比較（E5、ResNet50、5epoch＝640万枚でlossは9.718→7.608）と**フレーム数**をそろえる：256 clip×4フレーム×6,255step＝6,405,120フレーム。
- 10/22 MTGの「たくさんのフレームを通さないとlossは下がりきらない」という話を受けたもの。数え方は2026-11-04にユーザーが決めた（[壁打ち](../../../../secretary/notes/brainstorm/2026-11-04-sequential-loader-long-training-loss-check.md)）。

## 2. 前提（コード調査で確認済み）

研究コード `feature-vit-moco@50ff715`（clean）：

- 学習は `scripts/vit_moco/moco_vit_ddp_smoke.py`。
  - map-styleのdatasetを `DistributedSampler` と `DataLoader`（16 worker）で読む。`steps_per_epoch = len(loader)` を前提にしている。
  - データのエラーは、毎stepに全rankで集めて止める。
- clipの入力は `[frames, 3, 224, 224]`。2つのviewは、それぞれ全フレームに同じ乱数でaug-plusをかける（`ClipTransform`、spec 2026-11-03 E7）。Early Fusionのモデル（`ViTMoCoClipBackbone`）はそのまま使える。
- Queue（65,536）は、毎stepに一定数のkeyを入れる作りである（`queue_size % (world × local_batch) == 0` の検査、`concat_all_gather`）。そのため、batchは毎step 128（1 rankあたり）に固定する必要がある。
- 学習率は「epochごとのcosine（5epochの期間、warmupなし）」。これまでのA2は1epochだけだったので、実際には3e-4で一定だった。
- checkpointとresumeの仕組みはある。resumeでは、そのepochの消費済みbatchを読み直して捨てる（`steps_per_epoch` から位置を計算する）。
- GPU runの長さ・probe・checkpointは、presetごとに許可された値だけを受け付ける（`query_collapse_diagnostic.py` の `ACTIVITYNET_GPU_RUNS` など）。
- probeはImageNetの固定64枚で、clipの場合は4フレームに繰り返す（spec 2026-11-03 E11）。

sequential-loader `feature-stride-ddp@85a39c2`（研究コードの `.venv` から使っている）：

- `build_multistream_loader(sources, reader, chunk_config=FixedChunkConfig(frames_per_chunk, frame_stride), stream_config=MultiStreamConfig(streams_per_rank, num_workers, prefetch_factor, seed, rank, world_size))`。
- 1stepで、各rankのB個の枠から1 chunkずつ返す（`MultiStreamBatch.samples`、長さB）。
  - chunkは `frames` uint8 `[T, 3, H, W]`（元の解像度）、`valid_mask`（有効なフレームが先頭から並ぶ）、`frame_indices`、`timestamps`、`is_first`、`is_last` を持つ。
- 動画の割り当ては、epochごとに「sequence_id順 → seed＋epochで並べ替え → rank、枠の順に配る」。
- **どれか1つの枠の担当動画が尽きたstepで、そのepochは終わる**（全rankで合意して終わる）。終わったときの各枠の記録は `last_epoch_summary` に残る。
- workerは、枠 i を worker `i mod W` が読む。workerはdecodeだけを行い、変換（aug）をかける場所はない。
- stride kでも、decodeは全フレームに対して行う。RGBに変換するのは、取り出すフレームだけ。

データ：

- ActivityNet v1.3 training。`sl.ActivityNetAdapter(dataset_root).sequence_sources("training")` で10,024本。
- 全体で約326時間（約3,400万フレーム）。stride 16で取ると、1周で約210万フレーム（約53万clip）。
- 6,255stepは160万clipなので、約3周分になる。

## 3. Decision Contract（2026-11-04にユーザーが採用）

| # | 項目 | 採用 |
|---|---|---|
| D1 | データの読み方 | sequential-loader（`feature-stride-ddp`）。ActivityNet training全体、`frame_stride=16` |
| D2 | 入力 | **4フレームのclip**（loaderの1 chunk＝1 clip、`frames_per_chunk=4`）。フレームは k, k+16, k+32, k+48。同じ動画の次のclipは k+64 から始まる |
| D3 | batch | **1 rankあたり128本の動画を同時に1 clipずつ進める**（`streams_per_rank=128`）。2 rank、global batchは256 clip |
| D4 | 動画の最後で4枚に満たないclip | **足りないフレームを、そのclipの最後の有効フレームの繰り返しで埋める**（batchは常に128） |
| D5 | 学習方式 | **LoRAのみ**（Q/V、rank 8、alpha 8）。モデル、MoCo、augは、clip A2と同じ |
| D6 | 学習率 | **3e-4で一定**（epochごとのcosineは使わない） |
| D7 | 学習量 | **6,255step**（6,405,120フレーム、1,601,280 clip） |
| D8 | checkpoint | **1,000stepごと（1,000〜6,000）と6,255step**。途中で止まっても再開できるようにする。再開は、そのepochの最初から読み直して、消費済みのstepを捨てる（B1） |
| D9 | 比べる基準 | 追加の学習はしない。既存のclip A2 LoRA（ランダム順、0.25秒間隔、459step）と見比べる |
| D10 | 進め方 | G1（10step）→ G2（6,255step）。それぞれ別に許可を得る |
| D11 | augをかける場所 | **loaderに `sample_transform` を加え、workerの中でかける**（B2、§4 L1・L2） |

### 変えないもの（clip A2と同じ）

- `google/vit-base-patch16-224`、token結合のEarly Fusion（T=4、固定sin-cosの時間位置埋め込み）、gradient checkpointing。
- projection 128、K=65,536、momentum 0.999、T=0.2、`meta_einsum`。
- AdamW（weight decay 1e-4、betas (0.9, 0.999)、eps 1e-8）、FP32、seed 0、2 GPU（DDP）、local batch 128。
- aug-plus（RandomResizedCrop 224・scale (0.2, 1)、ColorJitter、Grayscale、GaussianBlur、Flip、ViTの正規化0.5）を、clipの全フレームに同じ乱数でかける。positiveは同じclipの2 view。
- 固定probe（ImageNet 64枚を4フレームに繰り返す）。

### 対象外

- Full FT、ランダム順のclip学習を同じ量だけ回すこと、stride・T・Bの比較、学習率の探索。
- 下流タスクでの評価、時間情報の評価（必要なら、終わった後に別途決める）。
- loaderの読み込み方式そのものの変更（seekの追加など）。

## 4. 実装要件（最小差分）

### 研究コード（`feature-vit-moco`）

| # | 内容 |
|---|---|
| I1 | 新しいdata source `activitynet_sequential_clips` と、新しいpreset（`moco_vit_seq_activitynet_clip`、profile名も新しくする）。preset以外では、このsourceを受け付けない |
| I2 | sequential-loaderからbatchを作る経路。各rankで `build_multistream_loader`（stride 16、T=4、B=128、loader seed＝`training.seed`）を作り、epochごとに `set_epoch(epoch)` して回す。各stepの128個のchunkから、`im_q`・`im_k` `[128, 4, 3, 224, 224]` を作る。<br>・D4の埋め方を行う。<br>・aug-plusは今の `ClipTransform` をそのまま使い、D4の埋め方とあわせて、loaderの `sample_transform` としてworkerの中でかける（D11） |
| I3 | epochの長さが前もって分からないことに対応する。学習ループは「loaderが尽きたら次のepoch」で進め、`steps_per_epoch` を使わない。checkpointには、epochとepoch内のstepを直接保存する |
| I4 | D6：このpresetでは、学習率を3e-4で一定にする |
| I5 | 記録。metricsに、stepごとの埋めたclipの数と、データ待ちの時間を加える。epochごとに `last_epoch_summary`（読み終えた動画・途中の動画・未読の動画）を保存する。run_reportに、loaderのcommit、stride、B、W、動画数、source一覧のsha256を記録する |
| I6 | GPU runは、G1（10step、probe `[0,1,2,10]`、checkpointなし）と、G2（6,255step、probe `[0,1,2,10,100,256,1000,2000,3000,4000,5000,6000,6255]`、checkpoint `[1000,2000,3000,4000,5000,6000,6255]`）だけを受け付ける。resumeはG2のcheckpointからだけ |
| I7 | resume：checkpointのepochで `set_epoch` し、そのepochの最初から読み直して、epoch内の消費済みstep数だけbatchを捨てる（D8）。resumeした後のstepの入力と状態は、止めずに学習した場合と一致させる。読み直しにかかった時間を記録する |

### sequential-loader（`feature-stride-ddp`、D11）

| # | 内容 |
|---|---|
| L1 | `build_multistream_loader` に、任意の `sample_transform`（chunk 1つを受け取って値を返す関数）を加える。W>0ならworkerの中で、W=0ならmain processでかける。指定しない場合の出力は、変更前と同じにする |
| L2 | workerの中の乱数は、epoch・rank・workerごとに決まる値で初期化する（seedが同じなら同じaugになる）。何を元に決めたかを記録できるようにする |

## 5. 互換性

- 既存のpreset、data source（`imagenet`、`activitynet_frames`、`activitynet_clips`）、学習経路、診断、評価、checkpoint形式は変えない。既存の学習は、bitwiseに同じ挙動のまま。
- loaderは、`sample_transform` を指定しなければ、変更前（`85a39c2`）と同じ出力にする。
- 依存パッケージは追加しない。

## 6. 検証（実装時、CPUのみ）

小さなViTと合成の動画を使う（既存のテストfixtureにならう）。

| # | 内容 |
|---|---|
| S1 | batchの形が `[128, 4, 3, 224, 224]`×2 view（テストでは小さいBとサイズ）。各clipのframe indexは k, k+16, k+32, k+48。同じ枠の次のstepは k+64 から始まる。2 rankで違う動画を読む |
| S2 | D4：4枚に満たない最後のclipは、最後の有効フレームで埋まる。埋めたclipの数が記録と一致する。4フレーム分の長さがない動画も同じように扱う |
| S3 | aug：1つのviewの全フレームに同じ変換がかかる（`ClipTransform` と同じ）。seedが同じなら、augも同じになる |
| S4 | epochの終わり：どれかの枠が尽きたら全rankで終わり、次のepochは割り当てが変わる。`last_epoch_summary` が保存される |
| S5 | 学習率は3e-4のまま一定（epochをまたいでも） |
| S6 | resume：途中のcheckpoint（epochの途中と、epochの境目の両方）から再開すると、その後のstepの入力のsha256と最終状態が、止めずに学習した場合とbitwise一致する |
| S7 | presetの制約：G1・G2以外の長さ、probe、checkpointを拒否する。ほかのpresetでは、このsourceを拒否する |
| S8 | 非回帰：研究コードの全テストで、失敗IDの集合が変更前と同じ（GPUが見える状態で実行）。loaderを変える場合は、loaderの全テストがOKで、`sample_transform` なしの出力が `85a39c2` と同じ |

## 7. GPU run（それぞれ別に許可が必要）

```bash
# G1: 10step。VRAM、データ待ち、decodeの速さを測る
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_seq_activitynet_clip \
    model.train_mode=lora training.max_steps=10 'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'
# G2: 6,255step。G1が工学PASSで、データ待ちが小さいことを確かめた後
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_seq_activitynet_clip \
    model.train_mode=lora training.max_steps=6255
```

- tmuxセッション `vit-moco-a1`。研究コードはclean commitの状態で実行する。
- G2は約19時間の見込み（clip A2 LoRAの約10.7秒/step）。G1でデータ待ちが大きい場合は、G2の前に報告する（Wなどの設定を勝手に変えない）。

## 8. 成功条件

### 工学的な成功（PASS / FAIL）

- G1・G2が最後のstepまで完走する。データのエラー、OOM、NCCLのエラー、非有限の値がない。
- Queueのpointerと実際のkeyの数が、毎stepで期待どおり。最後の状態確認（全rankが同じ状態、凍結した部分が変わっていない）がPASS。
- GPU1のVRAM headroomが1,024 MiB以上。
- G2で、すべてのcheckpointが保存される。

### 科学的な観測（記述的に報告し、閾値でPASS / FAILを決めない）

- Queueが埋まった後（step 256より後）のlossの推移。250stepごとの平均で、最初の窓（257〜506）と最後の窓（6,006〜6,255）を比べる。
- clip A2 LoRA（ランダム順、459step）の同じstepとの比較。条件が違う（フレームの間隔、順番に読むこと）ので、数値の差は参考として読む。
- collapseの兆候（`query_feature_std`、probeの統計）。
- 埋めたclipの割合、epochの数と長さ、データ待ちの時間。

### 主張しないこと

- 表現の品質、時間情報の獲得、公式MoCoと同じ性能になったこと、stride 16が最適であること。

## 9. 実行の境界

- このspecの承認は、研究コード（とloader）の実装とCPUの検証までを許可する。
- G1・G2は、それぞれユーザーの明示的な許可を得てから実行する。
- commitとpushは、明示的に依頼された場合だけ行う。

## 10. 未決事項

### blocking

なし。2026-11-04に、ユーザーが次のとおり決めた。

| # | 論点 | 採用 | 採用しなかった案 |
|---|---|---|---|
| B1 | resumeの方法 | **(a) そのepochの最初から読み直して、消費済みのstepを捨てる**。結果が厳密に一致し、実装が小さい。再開には数十分〜1時間程度かかり得る | (b) loaderに各枠の動画と位置を保存し、そこから読み始める（loaderの大きな変更になる） |
| B2 | augをかける場所 | **(a) loaderに `sample_transform` を加え、workerの中でかける**（§4 L1・L2） | (b) 研究コードのmain processでかける（数秒/stepの遅れの見込み） |

### non-blocking（既存に合わせる、または実装時に決める）

- loaderのworker数Wは、1 rankあたり16から始める（今の `num_workers` と同じ）。B=128を割り切れる値にする。G1の速さを見て報告する。
- 出力先は、既存と同じく `log/<preset>/lora/<source>/lr_0.0003/<日時>/`。
- epochの終わり方は、loaderの既存の仕様（どれかの枠が尽きたら終わる）に従う。1epochで、全フレームの一部は読まれずに次のepochへ進む（epochごとに割り当ては変わる）。

## 10a. 追記（2026-11-04、G1の後）：main processでの作業を減らす

G1（10step）では、1stepの計算が約8.7秒だったのに対し、データ待ちが約3.2秒/stepあった（[G1 Evidence](../experiments/2026-11-04-sequential-loader-clip-moco-lora-g1-gpu-evidence.md)）。CPUだけで測ると、原因はworkerの数ではなかった（W=32ではCPUを使い切り、かえって遅くなった）。原因は、main processでの次の2つの作業だった。

- 学習に使わない元の解像度のフレームを受け取ること（1 rankあたり毎step約1.28 GB）
- 入力全体（約0.6 GB）のsha256を計算すること（約1秒）

ユーザーが次の変更を採用した（2026-11-04）。

| # | 内容 |
|---|---|
| D12 | `sample_transform` を指定したときは、loaderは元のフレームをmain processへ返さない。返すのは管理用の情報（動画のID、chunkの番号、最初か最後か、frame index、timestamp、有効位置）と、変換の結果だけにする |
| D13 | 入力のsha256は、workerの中でclipごと・viewごとに計算する。main processでは、128 clip分のsha256を並べた一覧から、view q・view kそれぞれのsha256を1つずつ作り、`input_sha256` として記録する。今のrun（G1）とは値が違うが、同じ設定のrunどうしの比較（resumeの一致）には同じように使える |

実装要件：

- **L3**（loader）：変換したときの `MultiStreamBatch.samples` は、管理用の情報だけを持つ軽いデータ（`ChunkInfo`）にする。変換しない場合の出力は、変更前と同じにする。
- **I8**（研究コード）：`SequentialClipTransform` は、view q・view k、埋めたフレームの数、各viewのsha256を返す。main processは、viewをまとめて、一覧のsha256を作る。data記録に、この作り方を残す。

変えないもの：学習の内容（画像、aug、batch、Queue、LR、6,255step、W=16）。GPUでの制約（G1・G2の長さ、probe、checkpoint、決めた値）も変えない。

検証：

- **S9**：変換したときに返る管理用の情報が、変換しない場合のsampleの情報と一致する。clipごとのsha256が、viewから計算し直した値と一致する。S1〜S8を、この変更の後で再びPASSさせる。
- CPUだけで、main processの1stepあたりの時間を変更の前後で比べる（G1の追記と同じ方法）。
- G1（10step）をもう一度行い、データ待ちを測る（別に許可を得る）。

## 10b. 追記（2026-11-04、G2の失敗の後）：loader workerをforkserverで起動する【approved、2026-11-05】

G2は、step 1,600を終えた直後、epoch 0の終わり（step 1,601）で失敗した。両rankのloader workerが、片付けの途中で `SIGABRT` を受けて止まった（[G2 Evidence](../experiments/2026-11-04-sequential-loader-clip-moco-lora-g2-gpu-evidence.md)）。GPU 0・1で、次のことを確かめた。

- CUDAとNCCLを使っているrankのプロセスから、workerをforkで作る（今の方式）。この場合、epochの終わりにworkerを片付けるとabortする（48秒で再現）。
- CPUだけ（CUDAもNCCLもなし）では再現しない。
- workerをforkserverで作ると、同じ条件で3epochとも正常に終わる。1epochの時間も遅くならない。

### 採用案（D14）

| # | 内容 |
|---|---|
| D14 | loaderのworkerを、**forkserver** で起動する。効かせるのはloaderの `DataLoader` だけにする。rankのプロセス（`mp.spawn`）とDDPの起動方式、プロセス全体の既定値は変えない |

### 実装要件

- **L4**（loader）：`MultiStreamConfig` に、任意の `multiprocessing_context`（`None`、または `"fork"`・`"spawn"`・`"forkserver"`）を加える。`_worker_parts` の `DataLoader(multiprocessing_context=...)` に、その値をそのまま渡す。
  - 既定値は `None`。この場合は、今と同じ既定の方式（Linuxではfork）になり、出力も変更前（`b114c17`）と同じにする。
  - W=0で `None` 以外を指定した場合は、`ValueError` で拒否する（`persistent_workers` と同じ扱い）。
- **I9**（研究コード）：
  - presetに `data.sequential.multiprocessing_context: forkserver` を加え、`build_loader` から `MultiStreamConfig` へ渡す。
  - data記録の `streams` に、`multiprocessing_context` を残す。
  - `SEQUENTIAL_ADOPTED` に `"data.sequential.multiprocessing_context": "forkserver"` を加える。GPU runでは、ほかの値を拒否する。

### 変えないもの

学習の内容（画像、aug、batch、Queue、LR、6,255step、W=16、prefetch 2）。workerの乱数の初期化（`transform_seed`）。epochの終わり方。G1・G2の長さ、probe、checkpoint、そのほかの決めた値。

### 検証

- **S10**（CPU）：W>0のとき、forkserverで作った各batchの出力とtransform seedが、fork（`None`）のときと一致する（loaderのテスト）。
- **S11**（CPU）：loaderの全テストがOK。`multiprocessing_context=None` の出力は `b114c17` と同じ。
- **S12**（CPU）：研究コードのS1〜S9が再びPASSし、全テストの失敗IDの集合が変わらない。presetが `forkserver` 以外を拒否する。
- **G1b**（GPU、別に許可が必要）：scratchpadの作業用スクリプト `repro_epoch_end_ddp.py` で、epochの終わりが正常に済むことを確かめる。今回の確認と同じ条件（GPU 0・1、CUDAとNCCLを使った状態、256本、3epoch）で、プロセス全体の切り替えではなく、新しいpresetの設定（`build_loader` 経由）でforkserverにする。
- **G1**（GPU、別に許可が必要）：10stepを再び実行する。工学PASSであることと、データ待ちが今と同じ程度であることを確かめる。

### 未決事項

blocking：なし。2026-11-05に、ユーザーが次のとおり決めた。

| # | 論点 | 選択肢 | 採用 |
|---|---|---|---|
| B3 | G2をどこから始めるか | **(a) step 0から、最初からやり直す**。約20.5時間（約11.8秒/step）。checkpointの系統が1本で済む<br>**(b) G2のstep 1,000のcheckpointから再開する**。checkpointの「設定の同一性」（`config_identity`）にはloaderのcommitが入っている。そのため、今のままでは再開が拒否される。再開するには、このprofileだけ同一性の比べ方を変える必要がある（loaderのcommitとworkerの起動方式を比べないなど）。節約できるのは、約3.3時間からepoch 0の1,000step分の読み飛ばし（40〜50分）を引いた、約2.5時間 | **(a)**（2026-11-05、ユーザーが採用）。step 1,000のcheckpointは使わない |
| B4 | 起動方式を指定する場所 | **(a) loaderの設定項目にする（L4）**。既定値は今の方式のままで、研究コードのpresetからforkserverを指定する<br>**(b) loaderの中で、W>0なら常にforkserverにする**。設定は増えないが、loaderのほかの利用者（ほかのテストなど）の挙動も変わる。また、workerに渡すものすべてが、送れる形（pickle）でなければならなくなる | **(a)**（2026-11-05、ユーザーが採用） |

non-blocking：

- forkserverのworkerの起動は、epochごとに行う（`persistent_workers` は使わない、今と同じ）。今回の確認では、1epochの時間は遅くならなかった。
- 今回の再現では、forkの方式で片方のrankが止まると、もう片方のrankのworkerが親のない状態で残った。forkserverで同じことが起きるかは、G1bで、終わったあとに残ったプロセスを確かめて報告する（それ以上の対策は、このspecの範囲外）。
- G1で、途中停止（`stopped`）の片付けが正常だった理由は、このspecでは調べない。

## 11. 承認記録

- 2026-11-04：draftを作成した。D1〜D10は、ユーザーが壁打ちで採用した内容である。
- 2026-11-04：B1・B2とも(a)を、ユーザーが採用した（D8・D11に反映）。
- 2026-11-04：ユーザーがspecを承認した。実装するbranchは、新しく切らずに今のbranch（研究コード `feature-vit-moco@50ff715`、loader `feature-stride-ddp@85a39c2`）とする。研究コードの `.venv` はloaderの作業ツリーを直接読み込むため、loaderの変更は研究コードからすぐに見える。
- 2026-11-04：実装した（研究コード `feature-vit-moco@50ff715`、loader `feature-stride-ddp@85a39c2` の上、未commit）。CPUテストS1〜S8はPASS。specの範囲内で、2つの点を直した。1つ目は、このprofileだけresumeを受け付けるようにし、resumeしたrunではstep 0のprobeを取り直すようにしたこと。2つ目は、loaderが途中で止めたepochのstepsを、1つ少なく数えていたこと。
- 2026-11-04：研究コードを `feature-vit-moco@eb2e824`、loaderを `feature-stride-ddp@4026b06` としてcommitした（pushはしていない）。
- 2026-11-04（G1の後）：§10aのD12・D13（元のフレームを返さない、sha256をworkerの中でclipごとに計算する）を、ユーザーが採用した。
- 2026-11-04：§10aのD12・D13を実装した（未commit）。CPUテストS1〜S9はPASSし、全テストの失敗ID集合は変わらない。main processの1stepあたりの時間は、CPUのみの測定で約1.65秒から約0.45秒になった。
- 2026-11-04：研究コード `feature-vit-moco@f16a485`、loader `feature-stride-ddp@b114c17` としてcommitし、G1を再実行した（工学PASS、データ待ち約2.3秒/step）。
- 2026-11-04（G2の失敗の後）：§10b（D14、loader workerをforkserverで起動する）をdraftとして追記した。B3・B4は未決のため、まだ承認されていない。
- 2026-11-05：B3・B4とも(a)を、ユーザーが採用した。G2は、修正後にstep 0からやり直す。起動方式は、loaderの設定項目（L4）にして、研究コードのpresetからforkserverを指定する。§10bの承認は、まだ受けていない。
- 2026-11-05：ユーザーが§10bを承認した。実装するbranchは今のまま（loader `feature-stride-ddp@b114c17`、研究コード `feature-vit-moco@f16a485`）。GPU run（G1b・G1・G2）は、それぞれ別に許可を得る。
- 2026-11-05：§10bのL4・I9を実装した（loader `b114c17`・研究コード `f16a485` の上）。研究コード `1a37860`・loader `ce4dcbc` としてcommitした（pushはしていない）。S10〜S12はPASS（loaderの全テスト215件OK、変換なしの出力は `b114c17` とbitwise一致、研究コードの全テストの失敗ID集合は変わらない）。G1b・G1・G2はまだ行っていない。
