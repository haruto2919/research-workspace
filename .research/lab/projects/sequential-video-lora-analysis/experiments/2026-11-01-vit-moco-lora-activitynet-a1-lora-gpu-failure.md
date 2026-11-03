---
project: sequential-video-lora-analysis
date: 2026-11-01
record_type: gpu-run-evidence
status: failed-data-decode-error-at-step3-fixed-uncommitted
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: 167f1c1f6b03e68bdd7d4fae3e5e05d893559d62
spec: ../specs/2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md
---

# ViT＋MoCo ActivityNet単一フレーム：GPU A1（LoRA・10step）失敗記録

[ActivityNet単一フレームspec](../specs/2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md)のA1のうち、LoRAの10step smokeをユーザーの許可を得て実行した。**step3のデータ読み込みで失敗**した。Full FTのA1は実行していない。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet model.train_mode=lora training.max_steps=10 training.learning_rate=0.0003 'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'
```

- 研究リポジトリ：`feature-vit-moco@167f1c1`、clean（`dirty=False`）。
- GPU構成はL4/L5と同じ（GPU0：RTX PRO 4500 Blackwell、GPU1：RTX A4500）。開始前は両GPUとも空き。
- 出力：`log/moco_vit_lr_activitynet/lora/activitynet_frames/lr_0.0003/2026-11-01_17-24-22-020829/`
- 時刻：17:24開始 → 18:12終了（約48分。うち30分はNCCLのtimeout待ち）。

## 結果

| 項目 | 内容 |
|---|---|
| status | `failed`、engineering `BLOCKED` |
| 完了step | 2 / 10（probe 0・1・2は記録済み、probe 10は欠落） |
| step1 | loss 7.689、positive cos 0.700、negative cos −0.0003、grad norm 3.13、q std 0.0693、step_time 2.93 s |
| step2 | loss 7.578、positive cos 0.729、negative cos 0.0015、grad norm 1.65、q std 0.0587、step_time 1.45 s |
| 失敗箇所 | rank 0、stage `data`、step 3。DataLoader worker 2の `RuntimeError` |
| エラー | `v1-3/train_val/v_5kBKAfEX7XA.mp4: no frame at or before 45.000 s after seeking` |
| その後 | rank 1はstep3の `all_gather_object`（入力ハッシュの集約）で待ち続け、30分（1,800,000 ms）でNCCL watchdogがtimeoutし、プロセスが終了した |

step1〜2の値は2stepだけのものであり、学習の挙動について何も主張しない。

## 原因（read-onlyで確認）

`scripts/vit_moco/activitynet_frames.py` の `decode_frame`（2026-10-21の `9348efe` から変更なし）は、目標時刻へkeyframe seek（`backward=True`）してから前向きにdecodeし、「pts ≤ 目標時刻の最後のframe」を返す。このとき、**seek後の最初のframeが目標時刻より後だと、エラーで止まる**。

問題の動画（長さ45.58 s、29.97 fps、1080p h264、manifest上のframe数は5）のkeyframeは5.005 s間隔だった。最後のkeyframeだけ次のようになっている。

- pts 45.045 s
- dts 44.945 s（他のkeyframeは pts−0.033 s 程度）

目標45.0 sに対してdts ≤ 45.0 のこのkeyframeへseekしたため、最初にdecodeされたframeのpts（45.045）が目標を超え、条件を満たすframeがなかった。同じ方法で55 sへseekした場合も45.045に着地した。

- この経路は、ImageNetを使ったL4/L5では通らない。今回の実装変更（`167f1c1`）にも含まれない、既存のデータ読み込みコードの問題である。
- 117,522 frameのうち、同じ理由で失敗するものが何件あるかは未確認。

## 副次的な問題

1つのrankのデータ読み込みエラーが他のrankへ伝わらず、もう一方のrankは集約の通信で30分待ってから終了した。結果として失敗はすべて記録されたが、エラーの検出まで30分かかる。

## データ読み込み速度（参考、未確定）

固定probeの作成が終わったのは17:26頃で、step2のmetricsが書かれたのは17:36頃だった。最初のbatch（rankあたり128 frame）を1つのworkerが作るのに、最大で約10分かかったと考えられる。16 workerで並列に処理しても、459stepで数時間かかる可能性がある。ただしこれは2stepだけからの粗い推定で、step3以降の定常的な速度は測れていない。

## 2026-11-02 追記：修正と全frame確認

ユーザーが修正1＋2＋3を承認した。研究コードは `feature-vit-moco@167f1c1` の上で修正し、**未commit**である。

### 修正1：`decode_frame`（`scripts/vit_moco/activitynet_frames.py`）

最初のseekで目標時刻以前のframeが見つからなかった場合、seek位置を目標の1・2・4…秒前と下げて読み直す。動画の先頭まで下げても見つからなければ、エラーにする。最初のseekで見つかった場合は、従来と同じ処理・同じ画像になる。frameの定義（pts ≤ 目標時刻の最後のframe）は変えていない。

### 修正3：rank間のデータエラー共有（`scripts/vit_moco/moco_vit_ddp_smoke.py`）

各stepの入力ハッシュの集約（`all_gather_object`）に、データ読み込みのエラーも載せるようにした。どれか1つのrankでエラーが起きると、すべてのrankが同じstepで止まる。エラーを起こしたrankは元のエラーを、他のrankは `data loading failed on another rank: ...` を記録する。正常時の入力ハッシュと記録内容は変えていない。

### テスト（`test/model/test_moco_vit_ddp_smoke.py` に4件追加）

| テスト | 修正前 | 修正後 |
|---|---|---|
| seekが目標を越えた場合に、手前から読み直して同じ画像を返す（合成動画） | FAIL | PASS |
| 先頭まで下げても見つからなければエラーになる | FAIL | PASS |
| A1で失敗した実動画（`v_5kBKAfEX7XA`）の5 frameが、先頭からdecodeした画像とbitwise一致する（NASがある場合のみ） | FAIL（45 s） | PASS |
| 1つのrankのデータエラーで、両rankが同じstep・stage `data` で止まる（CPU・gloo、60秒以内） | FAIL（rank 1が `train` 段で失敗） | PASS |

全テストの結果：36 failed / **907 passed**。失敗IDの集合は、ActivityNet単一フレームの実装後（36 failed / 903 passed）と**完全に一致**した。passedの増加分4件は、新規テストの数と一致する。

### 修正2：全frameの確認（CPU、読み取りのみ）

修正後の `decode_frame` で、manifestの全frameを1枚ずつ取り出した（32プロセス、2026-11-02 13:08開始、約33分）。

| 項目 | 結果 |
|---|---|
| 対象 | 117,522 frame（9,998動画。frameのない26動画を除く）。indexは0〜117,521が欠けも重複もなく揃った |
| 取り出しエラー | **0件** |
| 読み直しが必要だったframe | **43 frame（42動画）**。修正前のコードなら、この43 frameで失敗していた。目標時刻は5 sが多く、A1の `v_5kBKAfEX7XA` 45 sも含む |
| 1 frameあたりの時間（1プロセス） | p50 0.26 s、p90 1.10 s、p99 2.88 s、最大 26.0 s、平均 0.53 s |

- 確認に使った `activitynet_frames.py` のsha256は `1917c6aa…8d807`（修正後・未commit）。
- 確認用スクリプトと全frameの結果（frameごとのseek回数・時間・画像のsha256）はscratchpadにあり、研究リポジトリには入れていない。

### 読み込み速度についての補足

平均0.53 s/frameなら、1 workerが128 frameのbatchを作るのに約70秒かかる。16 workerで並列に処理すれば、1stepあたりのデータ待ちは数秒程度と見込まれる。A1で最初のbatchに最大約10分かかったように見えた原因は、まだ分かっていない。再実行するA1で実測する。

## 未実施

- 修正した研究コードのcommit。runnerはdirtyな状態でのGPU runを拒否するため、A1の再実行前に必要。
- A1の再実行（LoRA → Full FT）とA2。
