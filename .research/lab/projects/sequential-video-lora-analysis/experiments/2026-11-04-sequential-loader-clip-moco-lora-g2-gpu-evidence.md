---
project: sequential-video-lora-analysis
date: 2026-11-04
record_type: gpu-run-evidence
status: g2-engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: f16a485
loader_commit: b114c17
spec: ../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md
---

# sequential-loader（stride 16）clip MoCo LoRA：GPU G2（6,255step予定）

[spec](../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md) §7のG2を、ユーザーの許可を得て実行した。**step 1,600を終えた直後、epoch 0の終わり（step 1,601のデータ読み込み）で、両rankのloader workerが `SIGABRT` で止まり、runは失敗した**。学習そのもの（step 1〜1,600）には異常はなかった。

> **追記（2026-11-09）**：loaderのworkerをforkserverで起動するように直し（spec §10b）、G2をstep 0からやり直した。step 2,114で一度止め、step 2,000のcheckpointから再開して、**6,255stepを完走し、工学PASS**となった。結果は末尾の「G2の完走」にある。

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

## G2のやり直し（2026-11-05、step 0から、spec §10b B3）

- G1の3回目（上のG1 Evidenceの追記）が工学PASSだったので、ユーザーの許可どおり、G2をstep 0から始めた。
- 研究コード `9285a47`（clean）、loader `ce4dcbc`（clean）。`CUDA_VISIBLE_DEVICES=0,1`、tmux `vit-moco-a1`。コマンドはspec §7のG2と同じ。
- 開始：2026-11-05 02:08:12。出力：`log/moco_vit_seq_activitynet_clip/lora/activitynet_sequential_clips/lr_0.0003/2026-11-05_02-08-19-443327/`
- step 3まで、lossは1回目のG2と同じ値で進んだ（7.588、7.447、7.538）。GPU 0は13,375 MiB、GPU 1は13,155 MiBを使っている。
- 見込み：1stepあたり約11.5〜11.8秒で、約20〜20.5時間。2026-11-05の22時〜23時ごろに終わる見込み。最初のepochの終わりは、約5時間後（step 1,600付近）の見込み。

### ユーザーの指示で停止（2026-11-05 09:19）

- ユーザーの指示で、tmuxにCtrl-Cを送って止めた。最後に記録されたstepは2,114（epoch 1）。
  - 両rankとも、step 2,115で `KeyboardInterrupt` を記録した（`rank*_report.json` の `status: failed`）。
  - 1回目のCtrl-Cでは、rankのプロセスが終わりきらなかった。約2分後に2回目のCtrl-Cを送ると、全プロセスが終わった。GPU 0・1は空いた。2回目で実行スクリプトも止まったため、`seq-g2c.log` に `exit=` の行はない。
- 残ったもの：checkpointはstep 1,000と2,000。
- **epoch 0の終わりは正常に越えた**。両rankとも `loader_epochs` にepoch 0（1,600step、`exhausted`）が記録された（07:38）。そのあと、epoch 1のworkerが起動して学習が続いた（step 1,601のloss 7.366）。
- 参考の値（判断はしない）：
  - lossの平均は、step 257〜506で8.348、step 1,865〜2,114で7.351。
  - `query_feature_std` は、step 1で0.0686、step 2,114で0.0879。

### step 2,000から再開（2026-11-08 17:25、ユーザーの指示）

- 研究コード `9285a47`（clean）、loader `ce4dcbc`（clean）。停止したときと同じcommitである。`CUDA_VISIBLE_DEVICES=0,1`、tmux `vit-moco-a1`。
- コマンド：G2と同じコマンドに、`training.resume=<上の出力>/checkpoint_step002000.pt` を足した（scratchpadの `seq-g2d.sh`）。
- 出力：`log/moco_vit_seq_activitynet_clip/lora/activitynet_sequential_clips/lr_0.0003/2026-11-08_17-26-03-055153/`
- 開始から約28分でstep 2,001が出た。この間に、checkpointの読み込み、epoch 1の最初の400 batchの読み直し、step 0のprobeを行った。
- **step 2,001と2,002は、止める前のrunと一致した**。
  - 各rankの `input_sha256`（view q・view k）が、bitwiseで一致した。
  - lossも一致した（7.33260…、7.26620…）。
- GPUの使用量は、GPU 0が13,352 MiB、GPU 1が13,132 MiB。
- 見込み：残りは4,255step。1stepあたり約11.7秒なので、2026-11-09の朝（8時ごろ）に終わる見込み。

## G2の完走（2026-11-09 07:58）

step 2,000から再開したrunは、step 6,255まで完走した（exit 0、終了は2026-11-09 07:58:01）。step 1〜2,000はやり直したrun（`2026-11-05_02-08-19-443327`）の記録、step 2,001〜6,255は再開したrun（`2026-11-08_17-26-03-055153`）の記録を合わせて読む。
- sha256（先頭、再開したrun）：`run_report.json` `a84b755f259d`、`metrics.jsonl` `ac04412225c2`、`diagnostic_summary.json` `b819d6231ff2`、`probe_metrics.jsonl` `d87ba2d447b8`

### 工学的な確認（spec §8）

| 項目 | 結果 |
|---|---|
| 完走 | **PASS**。`status: completed`、`last_global_step` 6,255、failuresなし。2つのrunを合わせて、step 1〜6,255が1つずつ記録されている。全stepでfinite、LRは全stepで3e-4 |
| Queue | queue_ptrは毎step 256ずつ進み、step 256で0に戻った（K=65,536） |
| 最後の状態確認 | **PASS**。全rankが同じ状態（`state_sha256` が一致）、凍結した部分は変わっていない、keyの凍結した部分はqueryと同じ |
| diagnostic summary | `engineering_status: PASS`、reasonsなし。再開したrunのprobe `[0, 3000, 4000, 5000, 6000, 6255]` がすべて記録された。step 0のprobeの特徴は、やり直したrunとbitwise一致した |
| VRAM | GPU 0：max reserved 12,784 MiB、headroom 19,434 MiB<br>GPU 1：max reserved 12,784 MiB、**headroom 7,312 MiB**（基準は1,024 MiB以上）→ **PASS** |
| checkpoint | **すべて保存された**。step 1,000・2,000はやり直したrun、step 3,000・4,000・5,000・6,000・6,255は再開したrunにある |
| resume | step 2,000のcheckpoint（sha256 `3c88ddcc…`）から再開した。epoch 1の最初の400 batchを読み直して捨てるのに1,505秒かかった。step 2,001・2,002の入力とlossは、止める前のrunと一致した |

- `ranks_feature_identical` は `false` だった。これまでのG1（3回とも）とclip A1でも `false` で、`engineering_status` の判定には入っていない。GPU 0とGPU 1は機種が違うので、probeの特徴がbitwiseで一致しないためと考えられる（確認はしていない）。

### loss（250step窓の平均、記述のみ）

| 窓（step） | loss | 正例のcos | 最も似たnegativeのcos | `query_feature_std` |
|---|---|---|---|---|
| **257〜506（最初の窓）** | **8.348** | 0.522 | 0.601 | 0.0869 |
| 1,007〜1,256 | 7.632 | 0.746 | 0.798 | 0.0876 |
| 2,007〜2,256 | 7.308 | 0.825 | 0.851 | 0.0878 |
| 3,007〜3,256 | 7.140 | 0.858 | 0.874 | 0.0879 |
| 4,007〜4,256 | 7.078 | 0.868 | 0.883 | 0.0879 |
| 5,007〜5,256 | 6.991 | 0.886 | 0.894 | 0.0879 |
| **6,006〜6,255（最後の窓）** | **6.964** | 0.889 | 0.896 | 0.0879 |

- Queueが埋まった後のlossは、**最初の窓の8.348から最後の窓の6.964へ下がった**（−1.38）。step 6,255のlossは6.979。
- 下がり方は後半ほど小さい。257stepから区切った250step窓ごとの低下は、最初の3つで0.36・0.21・0.15だった。step 4,007以降は、1窓あたり−0.004〜0.031だった。2つの窓は、1つ前の窓よりわずかに高かった。
  - step 5,257〜5,506：6.995（1つ前は6.991）
  - step 5,757〜6,006：6.978（1つ前は6.976）
- 平均のnegativeのcosは、step 2,000以降はほぼ0（−0.0007〜−0.0002）だった。
- **最も似たnegativeのcosは、全期間で正例のcosより少し高い**（最後の窓で0.896と0.889）。同じ動画の少し前のclipがQueueに入っているため、と考えられる（spec §2の注意点、未確認）。
- **clip A2 LoRA（ランダム順、0.25秒間隔）との比較**：同じstep（410〜459の窓）のlossは、A2が8.309、今回が8.242だった。条件（フレームの間隔、順番に読むこと）が違うので、差は参考にとどめる（spec §8）。A2は459stepで終わっているので、それより後は比べられない。

### collapseの兆候（記述のみ）

`query_feature_std`（学習batchのqの各次元のばらつき）は、step 256より後は0.0869〜0.0880の範囲だった。上限の目安1/√128≈0.0884に近いまま下がっていない。

固定probe（ImageNet 64枚の静止clip、rank 0）：

| step | z_q cos p50 | z_q cos p95 | q mean_dim_std | z_q：step 0とのcos p50 | CLS cos p50 | CLS：step 0とのcos p50 |
|---|---|---|---|---|---|---|
| 0 | 0.318 | 0.446 | 0.0727 | — | 0.032 | — |
| 10 | 0.594 | 0.683 | 0.0561 | 0.948 | 0.032 | 1.000 |
| 256 | 0.095 | 0.288 | 0.0835 | 0.643 | 0.029 | 0.985 |
| 1,000 | 0.102 | 0.327 | 0.0829 | 0.581 | 0.022 | 0.960 |
| 2,000 | 0.095 | 0.339 | 0.0831 | 0.550 | 0.020 | 0.934 |
| 3,000 | 0.094 | 0.350 | 0.0830 | 0.509 | 0.018 | 0.905 |
| 4,000 | 0.092 | 0.360 | 0.0829 | 0.460 | 0.018 | 0.878 |
| 5,000 | 0.091 | 0.365 | 0.0830 | 0.425 | 0.020 | 0.856 |
| 6,000 | 0.092 | 0.381 | 0.0827 | 0.393 | 0.021 | 0.842 |
| 6,255 | 0.096 | 0.381 | 0.0826 | 0.365 | 0.023 | 0.825 |

- **6,255stepまで、collapseは観測されなかった**。
  - z_qの方向集中（p50）は、step 10で0.59まで上がった後、step 256から最後まで約0.09〜0.10のままだった。
  - q mean_dim_stdは約0.083のままだった。
  - `uniformity_warnings` は空だった。
- p95はstep 256の0.288から、step 6,255の0.381へ少しずつ上がった。一部のprobe画像どうしが近づいている。
- 特徴は、学習の後半も動き続けている。step 0とのcos p50は、z_qで0.365、CLSで0.825まで下がった（clip A2 LoRAのstep 459ではCLSが0.978）。
- probeはImageNetの静止clipなので、ActivityNetでの表現品質は示さない。

### epochと読み込み

| epoch | step数 | 終わり方 | 1 rankあたりの完了した動画（例：rank 0のslot 0） |
|---|---|---|---|
| 0 | 1,600 | `exhausted` | 27本 |
| 1 | 1,580 | `exhausted` | 31本 |
| 2 | 1,413 | `exhausted` | 27本 |
| 3 | 1,419 | `exhausted` | 27本 |
| 4 | 243 | `stopped`（max_steps） | 4本 |

- 両rankとも同じstep数で、合計は6,255step。epochの終わり（4回）は、どれも正常に越えた。
- epochの長さは1,413〜1,600stepだった。どこかのslotの動画が尽きた時点でepochが終わる。そのため、1 epochで全動画を読み終えるわけではない。
- 埋めたフレームは、45,087 / 6,405,120フレーム（**約0.70％**）。

### 時間

| 項目 | 値 |
|---|---|
| 1stepの計算（`step_time_sec`、step 2〜6,255の平均） | 8.85秒 |
| データ待ち（遅い方のrank） | step 2〜2,000：平均3.16秒、中央値3.29秒<br>step 2,002〜6,255：平均2.91秒、中央値2.86秒 |
| 再開したrunの本体 | 52,316秒（約14.5時間、resumeの読み直し約25分を含む） |
| やり直したrun（step 0〜2,114） | 2026-11-05 02:08〜09:19（約7.2時間） |

### 主張しないこと（spec §8）

表現の品質、時間情報の獲得、公式MoCoと同じ性能になったこと、stride 16が最適であること。

## Cometへの記録（2026-11-10、ユーザーの指示）

学習中のrunはCometを使わない（runのディレクトリが記録の正本）。ユーザーの指示で、学習の後から、2つのrunの記録をつないでCometに送った。
- 実験：`haruto2919/2026-09-ishikawa-sequential-video-lora`、名前 `seq-loader-clip-lora-g2__moco__6255step`
  - URL：https://www.comet.com/haruto2919/2026-09-ishikawa-sequential-video-lora/d7089226ac604f169c491873800299c3
  - tag：moco、lora、clip、sequential-loader、stride-16、seed-0、g2
- 送ったもの：
  - **stepごとの値（step 1〜6,255）**：step 1〜2,000は最初のrun（`2026-11-05_02-08-19-443327`）、step 2,001〜6,255は再開したrun（`2026-11-08_17-26-03-055153`）から取った。最初のrunのstep 2,001〜2,114は送っていない。
  - stepごとの値は、loss、正例・平均negative・最も似たnegativeのcos、lr、`trainable_grad_norm`、`query_feature_std`、`step_time_sec`、データ待ち（遅い方のrank）、埋めたフレーム数。epochも付けた。
  - **固定probe（rank 0）**：cls・q・z_qの統計と、step 0とのcos p50。step 0〜2,000は最初のrun、3,000〜6,255は再開したrunから取った。
  - **設定**：`config_identity`、研究コード・loaderのcommit、2つのrunのディレクトリ、再開に使ったcheckpointとそのsha256、epochごとのstep数。
  - **ファイル**：両runの `resolved_config.yaml`・`rank0_report.json`・`loader_epochs`、再開したrunの `run_report.json`・`diagnostic_summary.json`。checkpointは送っていない。
- 送るのに使ったスクリプト：scratchpadの `seqtrain/comet_upload_g2.py`（runのディレクトリを読むだけ。研究コードは変えていない）。
- 確認：CometのAPIで読み直した。stepごとの10個の値は、どれもstep 1〜6,255がそろっていた。lossは手元の記録と全stepで一致した。probeは全13 stepがあり、ファイルは10個あった。

## checkpointを1つのフォルダにまとめた（2026-11-13、ユーザーの指示）

checkpointが2つのrunに分かれていたので、1つのフォルダから全stepを参照できるようにした。
- フォルダ：`log/moco_vit_seq_activitynet_clip/lora/activitynet_sequential_clips/lr_0.0003/combined_seq_clip_lora_step1000-6255/`
- 中身：step 1,000・2,000（やり直したrun `2026-11-05_02-08-19-443327`）と、step 3,000・4,000・5,000・6,000・6,255（再開したrun `2026-11-08_17-26-03-055153`）の `checkpoint_stepXXXXXX.pt` と `.rank{0,1}.pt` への**相対シンボリックリンク**。由来を書いた `README.txt` も置いた。
- 元のrunのファイルは移動も変更もしていない。上の記録のパスはそのまま使える。
- 確認：リンク切れはなし。7個のcheckpointをすべてCPUで読み込めた。`global_step` はファイル名と一致し、どのcheckpointにもquery側のLoRA tensorが48個ずつ入っていた。
