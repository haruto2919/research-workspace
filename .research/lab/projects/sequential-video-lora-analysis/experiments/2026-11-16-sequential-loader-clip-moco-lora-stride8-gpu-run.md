---
project: sequential-video-lora-analysis
date: 2026-11-16
record_type: gpu-run-evidence
status: engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: 59bade2
base_spec: ../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md
base_run: 2026-11-04-sequential-loader-clip-moco-lora-g2-gpu-evidence.md
---

# sequential-loader clip MoCo LoRA：フレーム間隔を8に変えた6,255step学習

前回の6,255step学習（stride 16、[記録](2026-11-04-sequential-loader-clip-moco-lora-g2-gpu-evidence.md)）と比べるため、**フレーム間隔（`data.clip.frame_stride`）だけを16から8に変えた**学習を、ユーザーの指示で始めた。学習率（3e-4）を含め、それ以外の設定は前回と同じである。

- このrunのための専用specはない。前回の設定（[spec](../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md)、`conf/moco/experiment/g2.yaml`）に、コマンドの上書きでstrideだけを変えた。
- 開始時点の記録に、完走後（2026-11-17）の結果を末尾の「完走（2026-11-17）」として追記した。**6,255stepを完走し、工学PASS**。

## 実行

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python main_moco.py +experiment=g2 data.clip.frame_stride=8 experiment_name=g2_stride8
```

- 研究コード `59bade2`（feature-vit-moco、clean）。GPU 0・1、tmuxセッション `vit-moco-a1`。
- 開始：2026-11-16 13:29:23。
- 出力：`log/moco/g2_stride8/lora/2026-11-16_13-29-30-869086/`
  - `resolved_config.yaml` に `frame_stride: 8`、`experiment_name: g2_stride8` が記録されている。
- `experiment_name=g2_stride8` は、前回の結果と保存先を分けるために付けた（出力先の名前だけで、計算には関係しない）。
- 実行前に、学習を動かさずに設定だけを表示し（Hydraの `--cfg job`）、strideと実験名以外が前回と同じであることを確かめた。

## 設定（前回との違い）

| 項目 | 前回（stride 16） | 今回 |
|---|---|---|
| フレーム間隔 `frame_stride` | 16 | **8** |
| 実験名（出力先） | g2 | g2_stride8 |

前回と同じ設定：

- モデル：ViT-B/16（`google/vit-base-patch16-224`）、LoRA（Q・V、rank 8、alpha 8.0、dropout 0）、projector 768→768→128、gradient checkpointingあり
- MoCo：K 65,536、momentum 0.999、temperature 0.2
- データ：ActivityNet v1.3 trainingを順番に読む（1 GPUあたり128 stream、worker 16、prefetch 2、forkserver）。4フレームをtoken結合したclip、画像224
- 学習：全体のbatch 256、学習率3e-4（一定）、AdamW（weight decay 1e-4）、6,255step、seed 0、FP32
- checkpoint：1,000・2,000・3,000・4,000・5,000・6,000・6,255
- 固定probe：ImageNet 64枚、step 0, 1, 2, 10, 100, 256, 1,000〜6,000, 6,255

## 開始直後の状態（2026-11-16 13:35ごろ）

| 項目 | 値 |
|---|---|
| loss（step 1・2・3） | 7.492、7.446、7.549 |
| 値 | すべて有限、学習率は3e-4 |
| 1stepの計算（`step_time_sec`、step 2・3） | 約8.6秒（前回の平均は8.84秒） |
| GPUの使用量 | GPU 0：13,375 MiB、GPU 1：13,155 MiB（前回とほぼ同じ） |

## 見込み（未確認）

- 1clipが読む範囲は、前回の64フレームから32フレームに半減する（4フレーム×stride）。そのため、同じ6,255stepで読む動画の時間は、前回の約半分になる。
- 前回は、epoch 0が1,600stepで終わった。今回は約3,200stepで終わる見込みで、6,255stepの中でepochの終わりは約1回になる（前回は3回）。
  - **訂正（2026-11-17）**：前回のepochの終わりは4回だった（epochの長さは1,600・1,580・1,413・1,419step、最後は243stepで停止）。
- 計算の重さは前回と同じなので、1stepあたり約11.7秒（データ待ちを含む）で、全体は約20時間の見込み。終わるのは2026-11-17の朝9時〜10時ごろの見込み。
- 前回と比べるときの注意：step数をそろえているので、読んだ動画の量はそろっていない。動画の量でそろえるなら、約12,500stepが必要になる。

## 完走（2026-11-17 09:27）

途中で止めることなく、6,255stepを完走した（exit 0、終了は2026-11-17 09:27:13、所要は約19.96時間）。

### 工学的な確認（前回と同じ項目）

| 項目 | 結果 |
|---|---|
| 完走 | **PASS**。`status: completed`、`last_global_step` 6,255、failuresなし。step 1〜6,255が1つずつ記録されている。全stepで値は有限、学習率は全stepで3e-4 |
| 上書きの記録 | `run_report.json` の `overrides` は `+experiment=g2`、`data.clip.frame_stride=8`、`experiment_name=g2_stride8`。データの記録も `frame_stride: 8`、動画10,024本 |
| 最後の状態確認 | **PASS**。両rankの `state_sha256` が一致した。凍結した部分は変わっておらず、keyの凍結した部分はqueryと同じ |
| diagnostic summary | `engineering_status: PASS`、reasonsなし、`uniformity_warnings` は空。probeは予定の13点（step 0〜6,255）がすべて記録された |
| VRAM | GPU 0：max reserved 12,784 MiB、headroom 19,434 MiB<br>GPU 1：max reserved 12,784 MiB、**headroom 7,312 MiB**（基準は1,024 MiB以上）→ **PASS**。前回と同じ値 |
| checkpoint | step 1,000・2,000・3,000・4,000・5,000・6,000・6,255がすべて保存された |
| epochの終わり | epoch 0は **3,184step** で終わり（`exhausted`、両rank）、workerの異常終了はなかった。epoch 1は3,071stepでmax_stepsにより止まった（`stopped`） |

- `ranks_feature_identical` は `false` だった。前回と同じで、`engineering_status` の判定には入っていない（GPU 0と1は機種が違う）。
- 見込みとの比較：epoch 0の長さ（約3,200stepの見込み → 3,184step）、所要時間（約20時間の見込み → 19.96時間）は、見込みどおりだった。

### loss（250step窓の平均、記述のみ）

前回（stride 16、[記録](2026-11-04-sequential-loader-clip-moco-lora-g2-gpu-evidence.md)）の値を並べた。

| 窓（step） | loss（今回） | loss（前回） | 正例のcos（今回） | 最も似たnegativeのcos（今回） | `query_feature_std`（今回） |
|---|---|---|---|---|---|
| 1〜10 | 7.638 | 7.662 | 0.746 | 0.743 | 0.0517 |
| **257〜506（最初の窓）** | **8.412** | 8.348 | 0.509 | 0.603 | 0.0869 |
| 1,007〜1,256 | 7.651 | 7.632 | 0.745 | 0.814 | 0.0876 |
| 2,007〜2,256 | 7.335 | 7.308 | 0.823 | 0.867 | 0.0879 |
| 3,007〜3,256 | 7.170 | 7.140 | 0.857 | 0.891 | 0.0878 |
| 4,007〜4,256 | 7.114 | 7.078 | 0.867 | 0.893 | 0.0879 |
| 5,007〜5,256 | 7.043 | 6.991 | 0.880 | 0.907 | 0.0880 |
| **6,006〜6,255（最後の窓）** | **6.998** | 6.964 | 0.887 | 0.909 | 0.0880 |

- Queueが埋まった後のlossは、**最初の窓の8.412から最後の窓の6.998へ下がった**（−1.41）。step 6,255のlossは7.083。
- 下がり方は前回と同じく後半ほど小さい。
- どの窓でも、今回のlossは前回より0.02〜0.06高い。ただし、1 seedで、読んだ動画の量も違う（下記）ので、差は記述にとどめる。
- **最も似たnegativeのcosは、全期間で正例のcosより高く、その差は前回より大きい**（最後の窓で0.909と0.887、前回は0.896と0.889）。stride 8では、Queueに入っている同じ動画の少し前のclipが、時間的にさらに近くなるため、と考えられる（未確認）。

### collapseの兆候（記述のみ）

`query_feature_std` は、step 256より後は0.0869〜0.0880で、前回と同じ範囲だった。

固定probe（ImageNet 64枚の静止clip、rank 0）：

| step | z_q cos p50 | z_q cos p95 | q mean_dim_std | z_q：step 0とのcos p50 | CLS cos p50 | CLS：step 0とのcos p50 |
|---|---|---|---|---|---|---|
| 0 | 0.318 | 0.446 | 0.0727 | — | 0.032 | — |
| 10 | 0.599 | 0.685 | 0.0558 | 0.945 | 0.032 | 1.000 |
| 256 | 0.095 | 0.288 | 0.0835 | 0.651 | 0.028 | 0.985 |
| 1,000 | 0.101 | 0.329 | 0.0829 | 0.558 | 0.021 | 0.960 |
| 2,000 | 0.083 | 0.326 | 0.0835 | 0.539 | 0.019 | 0.926 |
| 3,000 | 0.077 | 0.330 | 0.0837 | 0.506 | 0.019 | 0.900 |
| 4,000 | 0.084 | 0.347 | 0.0833 | 0.442 | 0.020 | 0.874 |
| 5,000 | 0.074 | 0.354 | 0.0836 | 0.410 | 0.021 | 0.847 |
| 6,000 | 0.086 | 0.370 | 0.0832 | 0.383 | 0.022 | 0.829 |
| 6,255 | 0.081 | 0.368 | 0.0833 | 0.373 | 0.024 | 0.815 |

- **6,255stepまで、collapseは観測されなかった**。z_qの方向集中（p50）はstep 256以降0.07〜0.10、q mean_dim_stdは約0.083のままだった。
- 前回（step 6,255：z_q p50 0.096、p95 0.381、CLSのstep 0とのcos 0.825）とほぼ同じ推移だった。
- probeはImageNetの静止clipなので、ActivityNetでの表現品質は示さない。

### データと時間

| 項目 | 今回 | 前回 |
|---|---|---|
| 1stepの計算（`step_time_sec`、step 2以降の平均） | 8.84秒 | 8.84秒 |
| データ待ち（遅い方のrank、step 2以降） | 平均2.46秒、中央値2.59秒 | 平均2.67秒、中央値2.79秒（step 1,600まで） |
| 全体の所要時間 | 19.96時間（中断なし） | 中断・再開を含む |
| epoch 0の長さ | 3,184step | 1,600step |

- 同じ6,255stepで、今回はepoch 0（全動画を1周）＋epoch 1の約96％を読んだ。前回は全動画を約4.2周した（epochの長さは1,600・1,580・1,413・1,419step＋停止までの243step）。学習に入れたフレーム数（6,255step×256 clip×4フレーム＝6,405,120フレーム）は同じだが、カバーした動画の時間（周回数）がそろっていないので、前回との比較はこの点に注意が必要。
  - **訂正（2026-11-17）**：最初は「前回は約3.9周」と書いていた。epoch 0の長さ（1,600step）だけから計算した誤りで、その後のepochはもっと短かった。

### まだ行っていないこと

- 並び順への反応の評価（前回のrunと同じ評価）を、このrunにも行うかは未定。
- LoRAの重みの解析（2026-11-12 MTGの方針）にこのrunを含めるかは未定。

## Cometへの記録（2026-11-17、ユーザーの指示）

学習中のrunはCometを使わない（runのディレクトリが記録の正本）。ユーザーの指示で、学習の後からCometに送った。送り方は、stride 16のrun（[記録](2026-11-04-sequential-loader-clip-moco-lora-g2-gpu-evidence.md)の「Cometへの記録」）とそろえた。

- 実験：`haruto2919/2026-09-ishikawa-sequential-video-lora`、名前 `seq-loader-clip-lora-stride8__moco__6255step`
  - URL：https://www.comet.com/haruto2919/2026-09-ishikawa-sequential-video-lora/aabfeb370e944828bb4333a8e4f27676
  - tag：moco、lora、clip、sequential-loader、stride-8、seed-0
  - stride 16のrun（`seq-loader-clip-lora-g2__moco__6255step`）と同じproject・同じ値の名前なので、Comet上で並べて比べられる。
- 送ったもの：
  - **stepごとの値（step 1〜6,255）**：loss、正例・平均negative・最も似たnegativeのcos、lr、`trainable_grad_norm`、`query_feature_std`、`step_time_sec`、データ待ち（遅い方のrank）、埋めたフレーム数。epochも付けた。
  - **固定probe（rank 0）**：cls・q・z_qの統計と、step 0とのcos p50（13 step）。
  - **設定**：`config_identity`、コマンドの上書き（`overrides`）、研究コード・loaderのcommit、runのディレクトリ、epochごとのstep数と終わり方。
  - **ファイル**：`run_report.json`、`rank0_report.json`、`diagnostic_summary.json`、`resolved_config.yaml`、`loader_epochs.rank{0,1}.jsonl`。checkpointは送っていない。
- 送るのに使ったスクリプト：scratchpadの `comet_upload_stride8.py`（runのディレクトリを読むだけ。研究コードは変えていない）。
- 確認：CometのAPIで読み直した。
  - stepごとの10個の値は、どれもstep 1〜6,255がそろっていた。
  - lossは手元の記録と全stepで一致した。
  - probeは全13 stepがあり、ファイルは6個あった。
