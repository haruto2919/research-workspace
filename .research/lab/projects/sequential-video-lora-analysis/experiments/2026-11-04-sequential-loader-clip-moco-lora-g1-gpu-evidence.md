---
project: sequential-video-lora-analysis
date: 2026-11-04
record_type: gpu-run-evidence
status: g1-engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: eb2e8247a5c4f230472288e88d5a977135a57351
loader_commit: 4026b06f684dc7762b37dd06bba9ab869c479d9e
spec: ../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md
---

# sequential-loader（stride 16）clip MoCo LoRA：GPU G1（10step）

[spec](../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md) §7のG1を、ユーザーの許可を得て実行した。**10stepを完走し、工学PASS**となった。ただし、**データ待ちが毎step約3.2秒ある**。そのため、G2は約21時間の見込みになる（specの見込みは約19時間）。spec §7に従い、Wなどの設定は変えずに報告する。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_seq_activitynet_clip \
    model.train_mode=lora training.max_steps=10 'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'
```

- 研究コード `eb2e824`（clean）、loader `4026b06`（clean）。tmuxセッション `vit-moco-a1`。
- GPU 0はRTX PRO 4500 Blackwell、GPU 1はRTX A4500。開始前は、どちらも空いていた。
- 時刻：10:48:04開始 → 10:55:13終了（run本体は420秒）。exit 0。
- 出力：`log/moco_vit_seq_activitynet_clip/lora/activitynet_sequential_clips/lr_0.0003/2026-11-04_10-48-11-706805/`
  - sha256（先頭）：`run_report.json` `ec569cd4e961`、`metrics.jsonl` `a0b772419360`、`diagnostic_summary.json` `725595e1720f`

## 工学的な確認

| 項目 | 結果 |
|---|---|
| 完走 | 10/10step、`status: completed`、failuresなし |
| Queueの検査、最後の状態確認 | PASS（全rankが同じ状態、凍結した部分は変わっていない） |
| diagnostic summary | PASS（probe 0・1・2・10） |
| VRAM | GPU 0：max reserved 12,784 MiB、headroom 19,434 MiB<br>GPU 1：max reserved 12,784 MiB、**headroom 7,312 MiB**（clip A1 LoRAと同じ）<br>→ PASS |
| データ | ActivityNet training 10,024本（`sources_sha256` `b2219038…`）、stride 16、T=4、1 rankあたり128 stream、W=16、prefetch 2、loader seed 0 |
| epoch | 両rankとも、epoch 0の10stepで `stopped`（max_stepsで止めた）。rankごとにtransform seedが記録された |
| 埋めたフレーム | 10step×256 clip×4フレームのうち30フレーム（約0.3％） |
| LR | 全stepで3e-4 |

## 時間

| 項目 | 値 |
|---|---|
| 起動 → step 1のデータが届くまで | 約130秒（128本×2 rankの動画を開き、16 workerが最初のclipを作るまで） |
| 1stepの計算（`step_time_sec`、step 2〜10） | 平均8.67秒（clip A1 LoRAの約8.7秒と同じ） |
| **データ待ち**（`data_wait_sec`、遅い方のrank、step 2〜10） | **平均3.16秒、中央値3.19秒**（1.9〜4.6秒） |
| probe 1回 | 約3秒 |

- 1stepあたりは、約8.7秒＋約3.2秒＝**約11.8秒**になる。G2（6,255step）は、起動、probe、checkpointを含めて**約21時間**の見込み（specでは約19時間と見込んでいた）。
- データ待ちは、loader（W=16、各workerが8 streamを担当し、stride 16でも全フレームをdecodeする）が、計算に追いついていないことを示す。10stepだけなので、定常状態の値としては参考にとどまる。

## 最初の10stepの値（参考）

| step | loss | 最も似たnegativeとのcos | 正例のcos |
|---|---|---|---|
| 1 | 7.588 | 0.369 | 0.720 |
| 5 | 7.676 | 0.782 | 0.737 |
| 10 | 7.791 | 0.794 | 0.746 |

- clip A1 LoRA（ランダム順、0.25秒間隔）のstep 10は、loss 7.741、最も似たnegative 0.766、正例 0.756だった。Queueが埋まっていく途中なので、lossが上がるのは、これまでのrunと同じである。
- 最も似たnegativeのcosは、ランダム順よりわずかに高い（0.794と0.766）。同じ動画の少し前のclipがQueueに入ることによる、と考えられる（spec §2の注意点）。10stepだけなので、判断はしない。

## まだ行っていないこと

- G2（6,255step）。データ待ちを含めて約21時間の見込み。ユーザーの許可を得てから行う。

## 追記（2026-11-04）：workerの数と、データ待ちの内訳（CPUのみ）

ユーザーが「GPU 2枚のままworkerを増やす」案を選んだ。そこで、specを直す前に、GPUを使わずに読み込みの速さを測った。

- 方法：presetと同じ設定（B=128、stride 16、T=4、aug-plus）で、1 rank分のloaderを作る。2つのプロセスを同時に動かし（seed 0と1）、2 rankを模した。NASは読み取りだけ。
- 1stepごとに、3つの時間を分けて測った。
  - `next()`：loaderからbatchを受け取る時間
  - `stack`：main processで、128 clipを1つのtensorにまとめる時間
  - `sha`：入力のsha256を計算する時間
- スクリプト：scratchpadの `seqtrain/bench_loader.py`（研究コードは変えていない）

| 条件 | `next()` | stack | sha256 | 1stepの合計 | 25stepの所要時間 | CPU（load average） |
|---|---|---|---|---|---|---|
| W=16、待ちなし | 2.4秒 | 0.17秒 | 1.05秒 | 約3.6秒 | 約95秒 | 約16〜25 |
| W=32、待ちなし | 0.8秒 | 2.7秒 | 0.92秒 | 約4.4秒 | 約117秒 | **約70〜80**（64スレッドを超える） |
| W=16、毎step 8.7秒待つ（学習の計算を模した） | **0.5秒** | 0.13秒 | **1.0秒** | 約1.65秒 | — | — |

- **W=32にしても速くならなかった**。CPUが使い切られて、全体ではかえって遅くなった（25stepで117秒と95秒）。
- loaderが先にbatchを作り終えていても、main processで**毎step約1.65秒**かかる。
  - 内訳は、受け取りに約0.5秒、sha256に約1.0秒、まとめるのに約0.13秒。
  - さらに、workerは変換後のviewのほかに、**元の解像度のフレーム（1 rankあたり毎step約1.28 GB）**もmain processへ送っている。
- G1の学習中のデータ待ち（約3.2秒）は、この1.65秒より長い。学習中はCPUやメモリの帯域を取り合うため、受け取りが遅くなっていると考えられる（推定、未確認）。
- 推定：データ待ちの主な原因は、decodeを行うworkerの数ではない。main processでの「元のフレームの受け取り」と「入力全体のsha256」である。

## 追記（2026-11-04）：G1の再実行（spec §10a、D12・D13の後）

元のフレームを返さないようにし、sha256をworkerの中でclipごとに計算するようにした。その後、ユーザーの許可を得て、同じコマンドでG1を再実行した。

- 研究コード `f16a485`（clean）、loader `b114c17`（clean）。tmuxセッション `vit-moco-a1`。
- 時刻：11:58:06開始 → 12:05:01終了（run本体は405秒）。exit 0。
- 出力：`log/moco_vit_seq_activitynet_clip/lora/activitynet_sequential_clips/lr_0.0003/2026-11-04_11-58-14-565464/`
  - sha256（先頭）：`run_report.json` `abb558cd6493`、`metrics.jsonl` `7909ab4f2038`、`diagnostic_summary.json` `6bcca0042f4f`

| 項目 | 1回目（`eb2e824`／`4026b06`） | 再実行（`f16a485`／`b114c17`） |
|---|---|---|
| 工学的な確認（完走、Queue、最後の状態確認、summary、VRAM） | PASS | PASS（headroom：GPU 0は19,434 MiB、GPU 1は7,312 MiB） |
| 各stepのloss | 7.588 → 7.791 | **1回目と全stepで同じ**（入力は同じ。変わったのは記録の仕方だけ） |
| 1stepの計算（step 2〜10の平均） | 8.67秒 | 8.72秒 |
| **データ待ち**（遅い方のrank、step 2〜10） | 平均3.16秒、中央値3.19秒 | **平均2.30秒、中央値2.68秒**（0.5〜3.3秒） |
| G2（6,255step）の見込み | 約21時間 | **約19.5時間** |

- データ待ちは約0.9秒/step減った。ただし、CPUだけで測ったときの見込み（約0.45秒）までは下がらなかった。
- 長い待ち（約3秒）と短い待ち（約0.6秒）が交互に出る。CPUだけで測ったときは、loaderは1stepを約3.5秒で作れていた。学習と同時に動かすと、loaderの作る速さが落ちているか、一部のworkerが遅れている（たとえば解像度の高い動画を受け持ったworker）と考えられる。**原因は確認していない**。
