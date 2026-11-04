---
project: sequential-video-lora-analysis
date: 2026-11-04
record_type: gpu-run-evidence
status: g2-failed-at-epoch-end
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: f16a485
loader_commit: b114c17
spec: ../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md
---

# sequential-loader（stride 16）clip MoCo LoRA：GPU G2（6,255step予定）

[spec](../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md) §7のG2を、ユーザーの許可を得て実行した。**step 1,600を終えた直後、epoch 0の終わり（step 1,601のデータ読み込み）で、両rankのloader workerが `SIGABRT` で止まり、runは失敗した**。学習そのもの（step 1〜1,600）には異常はなかった。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_seq_activitynet_clip \
    model.train_mode=lora training.max_steps=6255
```

- 研究コード `f16a485`（clean）、loader `b114c17`（clean）。tmuxセッション `vit-moco-a1`。
- 時刻：12:07:40開始 → 17:23:33終了。exit 1。
- 出力：`log/moco_vit_seq_activitynet_clip/lora/activitynet_sequential_clips/lr_0.0003/2026-11-04_12-07-47-617094/`
  - `run_report.json`：`status: failed`、`last_global_step` 1,600
  - checkpoint：`checkpoint_step001000.pt`（＋rankごとの位置ファイル）
  - `loader_epochs.rank{0,1}.jsonl` は空（epoch 0のsummaryを書く前に止まったため）
- Cometには記録していない（このrunnerはrun directoryを記録の正本とする）。

## 止まった状況

| 項目 | 内容 |
|---|---|
| 止まった場所 | `stage: data`、epoch 0、step 1,601、両rank |
| エラー | `RuntimeError: DataLoader worker (pid …) is killed by signal: Aborted.`（rank 0・rank 1とも） |
| 直前のログ | `terminate called without an active exception` が11行 |
| 流れ（traceback） | rank 0：loaderのworkerの1つが「担当する枠に次の動画がない」（`_worker_parts` の `yield None`）を返した＝**epoch 0の終わり**<br>rank 1：全rankの合意（`all_reduce(MIN)`）でepochの終わりを受け取った（`yield part` の位置で `GeneratorExit`）<br>→ 両rankとも `parts.close()` → DataLoaderの `_shutdown_workers()` でworkerの終了を待つ間に、workerが `SIGABRT` で落ち、PyTorchのsignal handlerが例外にした |
| 学習の値 | step 1,600まで全stepで有限、lossの異常なし |

- 推定：**epochが終わったときのworkerの後片付け（DataLoaderの終了処理）で、workerがabortしている**。G1では、max_stepsで途中停止したとき（`stopped`）の終了処理は2回とも正常だった。epochが最後まで終わる場合（`exhausted`）は、GPU runでは今回が初めてだった。
- worker内で何がabortしたか（PyAV／FFmpegのdecode、あるいはworkerのプロセス終了処理）は、まだ確認していない。

### CPUだけでの再現の試み（2026-11-04 17:40ごろ）

- 方法：presetと同じloader（B=128、W=16、stride 16、T=4、aug-plusはworkerの中）を1 rank分だけ作り、ActivityNet trainingの先頭128本（1枠に1本）を、epochが終わるまで読んだ。GPU、DDP、学習の計算は使っていない。スクリプトはscratchpadの `seqtrain/repro_epoch_end.py`（研究コード・loaderは変えていない）。
- 結果：3stepでepochが終わり（`exhausted`）、**workerの終了処理は正常だった**（abortなし、exit 0）。
- **CPUだけ・短い読み込みでは再現しなかった**。G2との違いは、(1) workerが、CUDAとNCCLを使っている学習プロセスからforkされていること（loaderのDataLoaderは既定のforkで起動している）、(2) workerが5時間以上動いていたこと。どちらが効いているかは確認していない。

### GPU 0・1での再現（2026-11-04 23:28、ユーザーの許可を得て実行）

- 1回目の試み（23:25）：学習スクリプトを、動画を絞った設定（`data.sequential.video_dir`）で起動した。しかし、runnerの検査で起動直後に止まった（step数は10か6,255だけ。`video_dir` も、GPU runでは採用値の `null` から変えられない）。GPUでの計算はしていない。空に近い出力 `…/lr_0.0003/2026-11-04_23-25-34-202902/` が残っている。
- 2回目：scratchpadの作業用スクリプト `seqtrain/repro_epoch_end_ddp.py` を使った（研究コード・loaderは変えていない）。
  - runnerと同じく `mp.spawn` で2つのrankを立てた。各rankで `cuda.set_device`、NCCLの `init_process_group`、cuBLASの行列積、`all_reduce` を済ませ、CUDAとNCCLを使っている状態にした。
  - そのうえで、`activitynet_sequential.build_loader` でpresetと同じloader（B=128/rank、W=16、stride 16、T=4、aug-plusはworkerの中）を作った。ActivityNet trainingの先頭256本（mp4）へのシンボリックリンクのフォルダから読み、1枠に1本とした。
  - `train_sequential` と同じ手順（`set_epoch` → `iter` → StopIterationまで `next` → `close`）で、3epoch分を読む予定にした。モデルと学習は動かしていない。
  - 研究コード `f16a485`、loader `b114c17`、tmux `vit-moco-a1`。
- 結果：**起動から48秒で再現した**。
  - `terminate called without an active exception` が5行出た。
  - rank 0はepoch 0（2step、`exhausted`）を終えた。rank 1は、epoch 0の終わりの `_shutdown_workers()` で `RuntimeError: DataLoader worker (pid …) is killed by signal: Aborted.` となった。G2と同じtracebackである。
  - 出力は `seq-epochend-ddp.log`（scratchpad）。
- 結論：**長時間動かさなくても、CUDAとNCCLを使っているプロセスからforkしたworkerを、epochの終わりに片付けるとabortする**。CPUだけ（1プロセス、CUDAもNCCLもなし）では再現しなかった（上記）。原因は「workerがCUDAとNCCLを使っているプロセスからforkで作られていること」と強く推定できる。ただし、worker内のどのスレッドが `std::terminate` を呼んだかまでは確認していない。
- G1で、max_stepsの途中停止（`stopped`）の片付けが正常だった理由は確認していない。
- 片付けの問題：この再現では、異常終了したrank（rank 1）が止められたあと、**rank 0側のloader worker 16個が親のいない状態で残った**。これらがログのパイプを開いたままにしたので、tmuxの次のコマンドが始まらなかった。ユーザーの許可を得て、23:49にこの16個を `kill` した。G2の失敗のあとには、このような残りは見つかっていない。

### workerをforkserverで起動した場合（2026-11-04 23:49、ユーザーの許可を得て実行）

- 方法：上と同じ作業用スクリプトを、同じ条件（GPU 0・1、CUDAとNCCLを使った状態、256本、B=128/rank、W=16）で実行した。違いは1点だけで、loaderを作る前に、rankのプロセスで `torch.multiprocessing.set_start_method("forkserver", force=True)` を呼んだ。loaderの `DataLoader` は `multiprocessing_context` を指定していないので、この既定の方式でworkerが起動する。研究コード・loaderは変えていない。
- 結果：**3epochとも、両rankで正常に終わった**（各epochは2step、`exhausted`）。
  - `terminate called` は出ず、exit 0。終わったあとに残ったプロセスもない。
  - 1epochあたり約25秒で、forkのとき（epoch 0が約30秒）と比べて遅くはなっていない。
  - workerに渡す設定（`SequentialClipTransform`、reader、動画の割り当て）は、forkserverでもそのまま送れた。
  - 出力は `seq-epochend-fs.log`（scratchpad）。
- 結論：workerをforkserverで起動すれば、epochの終わりの異常終了は起きない。修正の候補として有力である。ただし、6,255stepの学習そのもので確かめたわけではない。また、forkserverで起動したworkerの乱数の種や、作るbatchがforkのときと同じになるかは確認していない。

## 学習の値（step 1〜1,600）

250stepごとの平均（両rankを合わせた値）。

| step | loss | 正例のcos | 最も似たnegativeのcos | 特徴のばらつき（query std） | 勾配のnorm |
|---|---|---|---|---|---|
| 1〜10 | 7.662 | 0.742 | 0.735 | 0.0513 | 1.313 |
| 11〜256 | 8.488 | 0.642 | 0.717 | 0.0806 | 1.102 |
| 257〜506 | 8.348 | 0.522 | 0.601 | 0.0869 | 1.056 |
| 507〜756 | 7.990 | 0.622 | 0.693 | 0.0870 | 0.986 |
| 757〜1,006 | 7.780 | 0.699 | 0.756 | 0.0874 | 0.909 |
| 1,007〜1,256 | 7.632 | 0.746 | 0.798 | 0.0876 | 0.853 |
| 1,257〜1,506 | 7.528 | 0.776 | 0.818 | 0.0878 | 0.795 |
| 1,351〜1,600 | **7.491** | 0.786 | 0.826 | 0.0878 | 0.773 |

- spec §8の比較窓のうち、最初の窓（257〜506）の平均は **8.348**。step 1,351〜1,600の平均は **7.491** で、0.86下がっている。Queueが埋まった（step 256）後は、lossは下がり続けている。
- ただし、spec §8が判定に使う最後の窓（6,006〜6,255）には届いていない。**「lossが下がるか」の判定はできていない**。
- 正例のcosと、最も似たnegativeのcosは、どちらも上がっている（0.52→0.79、0.60→0.83）。

固定probe（ImageNet 64枚、rank 0、`q`）：

| probe step | 次元ごとのstd | 64枚どうしのcos（平均） | cos（p95） |
|---|---|---|---|
| 0 | 0.0727 | 0.317 | 0.446 |
| 10 | 0.0561 | 0.593 | 0.683 |
| 100 | 0.0797 | 0.178 | 0.349 |
| 256 | 0.0835 | 0.097 | 0.287 |
| 1,000 | 0.0829 | 0.110 | 0.328 |

- step 10で一時的に特徴が似る方向へ寄ったが、step 100以降は戻り、step 1,000でも崩壊の兆候（std低下、cosの上昇、uniformity warning）はない。

## データと時間

| 項目 | 値 |
|---|---|
| 1stepの計算（`step_time_sec`、step 2〜1,600の平均） | 8.84秒 |
| データ待ち（遅い方のrank、step 2〜1,600） | 平均2.67秒、中央値2.79秒 |
| 1stepあたりの実時間（開始から止まるまで） | 約11.8秒（1,600stepで約5時間16分） |
| 埋めたフレーム | 1,600step×256 clip×4フレームのうち11,549フレーム（約0.7％） |
| epoch 0の長さ | 1,600step（この時点で、ある枠の動画がなくなった） |

- epoch 0が1,600stepで終わったため、6,255stepの学習では、epochの終わりが少なくとも3回ある見込み。

## まだ行っていないこと

- epoch末でworkerがabortする原因の特定と修正。
- G2のやり直し（step 1,000のcheckpointから再開するか、最初からやり直すか）。どちらも、epoch末の問題を直してからでないと、step 1,601付近で同じ失敗を繰り返す見込み。

## 訂正（2026-11-05）：workerはforkではなくspawnで起動していた

上の「GPU 0・1での再現」の結論では、原因を「CUDAとNCCLを使っているプロセスからworkerをforkで作ったこと」としていた。**この説明は誤りだった**。

- G1b（下記）のログで確かめたところ、rankのプロセスでの既定の起動方式は **spawn** だった。rankは `mp.spawn` で起動しているので、その子プロセスの既定の方式もspawnになる。loaderの `DataLoader` は `multiprocessing_context` を指定していなかった。そのため、G2と最初の再現では、**loaderのworkerはspawnで起動していた**。
- 残ったworker 16個のコマンド行（`spawn_main(tracker_fd=4, pipe_handle=…)`、pipe_handleはプロセスごとに違う）も、spawnで起動されたプロセスのものだった。
- CPUだけで確かめ直した（2026-11-05 00:55ごろ、scratchpadの `seqtrain/repro_epoch_end_method.py`）。1 rank、B=128、W=16、先頭128本で、プロセス全体の既定の方式だけを変えた。
  - **spawn**：epochの終わりにworkerが `SIGABRT` で止まった（`DataLoader worker … is killed by signal: Aborted.`）。**GPUもNCCLもなしで再現した**。
  - **fork**：3stepでepochが終わり、正常に終了した（最初のCPUの試しと同じ）。
- 訂正した結論：**spawnで起動したworkerが、終了するときにabortする**。CUDAとNCCLは関係ない。最初のCPUの試しで再現しなかったのは、1つのプロセスで動かしたため、workerがforkで起動していたからである。
- なぜspawnのworkerだけが終了時にabortするのかは確認していない。推定では、spawnの子プロセスはPythonを普通に終了する（C++の後片付けが動く）のに対し、forkとforkserverの子プロセスは `os._exit` で終了する。この違いによると考えられる。
- 修正（forkserver、spec §10b）は、この訂正の後でも有効である。forkserverで起動したworkerは、G1bでも正常に終わった。
- 誤った説明の残り：loader `ce4dcbc` と研究コード `1a37860` のcommitメッセージには「CUDAとNCCLを使うrankからforkしたworker」と書かれている（commitは書き換えていない）。研究コードの冒頭のコメントは `9285a47` で訂正し、そのcommitメッセージに `1a37860` の誤りも記した。loaderのコード・コメントには、誤った説明はない。

## G1b：presetの設定のforkserverで、epochの終わりを確かめる（2026-11-05 00:49、ユーザーの許可を得て実行）

- 研究コード `1a37860`（clean）、loader `ce4dcbc`（clean）。GPU 0・1、tmux `vit-moco-a1`。
- 上と同じ作業用スクリプト `repro_epoch_end_ddp.py` を使った。プロセス全体の既定の方式は変えず（spawnのまま）、`build_loader` に渡るpresetの設定（`multiprocessing_context: forkserver`）だけでworkerを起動した。2 rank、CUDAとNCCLを使った状態、256本、3epochである。
- 結果：**3epochとも、両rankで正常に終わった**（各2step、`exhausted`）。
  - `terminate called` は出ず、exit 0。
  - 終わったあとに残ったプロセスもない。
  - 1epochあたり約26〜27秒。
  - 出力は `seq-g1b-epochend.log`（scratchpad）。
