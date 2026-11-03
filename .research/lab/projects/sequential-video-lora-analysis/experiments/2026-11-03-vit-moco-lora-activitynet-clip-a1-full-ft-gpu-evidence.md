---
project: sequential-video-lora-analysis
date: 2026-11-03
record_type: gpu-run-evidence
status: a1-full-ft-engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: fe58fc37ef28901184980ccd5f9e90e40c3d836d
spec: ../specs/2026-11-03-vit-moco-lora-activitynet-clip-early-fusion-spec.md
---

# ViT＋MoCo ActivityNet clip Early Fusion：GPU A1（Full FT・10step）

[clip A1 LoRA](2026-11-03-vit-moco-lora-activitynet-clip-a1-lora-gpu-evidence.md)が工学PASSとなったのを受けて、ユーザーの許可を得て、Full FTのA1をtmuxセッション `vit-moco-a1` から実行した。**10stepを完走し、工学PASS**となった。LoRAとの比較器の結果も `comparable=true` で、**clipのA1は両方式とも工学的な成功条件を満たした**。

GPU1のheadroomは **6,984 MiB** で、単一フレームのFull FT（1,750 MiB）より大きかった。1stepの計算時間は約9.6秒だった。A2は未実施。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet_clip model.train_mode=full_ft training.max_steps=10 training.learning_rate=0.0003 'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'
```

- 研究リポジトリ：`fe58fc3`（clean）。GPU構成はLoRAと同じ。開始前、両GPUとも空いていた。
- 出力：`log/moco_vit_lr_activitynet_clip/full_ft/activitynet_clips/lr_0.0003/2026-11-03_15-12-06-149049/`
- 時刻：15:11:59開始 → 15:20:34終了（run本体は507秒）。

## 工学チェック

| 項目 | 結果 |
|---|---|
| run / engineering | `completed` / **PASS**（理由なし、`failures=[]`） |
| VRAM gate | PASS。peak reserved：GPU0 13,112 MiB / 32,218（headroom 19,106）、GPU1 13,112 MiB / 20,096（headroom **6,984**、閾値1,024） |
| final checks | 全rankがrank0と同じ状態（state sha256 `ccb37c99…`）。Full FTなので、frozen parameterは0 |
| 学習対象 | 86,487,680 parameter（query全体） |
| 固定probe | 元の64枚の入力ハッシュ `6af50745…08d8bf`（一致）。静止clipの入力ハッシュ `457fabe0…cd1f`（**LoRAと一致**） |
| probe step | 0・1・2・10がすべて記録された |
| queue | step10でqueue_ptr 2,560 |

- 単一フレームA1 Full FTのpeak reservedは、GPU0が18,124 MiB、GPU1が18,346 MiBだった。clipでは785 tokenに増えたが、reservedは約5 GiB少なくなった。gradient checkpointingで活性化の保持が減ったためと見られる（内訳は測っていない）。
- `ranks_feature_identical` は `false`。clip A1 LoRAと単一フレームのrunでも `false` で、これは成功条件の項目ではない。

### LoRAとの比較（`--comparison-axis train_mode`）

LoRA（`2026-11-03_15-01-35-356834`）とFull FT（本run）を比較器で比べた。結果は標準出力にだけ出し、ファイルには保存していない。

| 項目 | 結果 |
|---|---|
| `comparable` | **true** |
| 学習入力（全10step） | 一致（違うstepはない） |
| probe入力 | 一致 |
| step0の特徴（CLS / z_q / q） | すべてbitwise一致 |

## 学習中の値（参考）

| step | loss | positive cos | negative cos | hardest neg cos | grad norm | q std | step_time |
|---|---|---|---|---|---|---|---|
| 1 | 7.627 | 0.712 | −0.0002 | 0.367 | 3.28 | 0.0698 | 18.75 s |
| 2 | 7.561 | 0.733 | 0.0017 | 0.724 | 2.04 | 0.0491 | 9.52 s |
| 5 | 7.661 | 0.740 | 0.0078 | 0.755 | 1.43 | 0.0441 | 9.61 s |
| 10 | 7.731 | 0.752 | 0.0167 | 0.778 | 1.25 | 0.0540 | 9.74 s |

固定probe（rank0）の `z_q` 方向集中（offdiag cos p50）：

| step | clip LoRA | clip Full FT | 参考：単一フレームFull FT |
|---|---|---|---|
| 0 | 0.318 | 0.318 | 0.319 |
| 1 | 0.452 | 0.524 | 0.518 |
| 2 | 0.536 | 0.611 | 0.610 |
| 10 | 0.602 | 0.573 | 0.583 |

- `q` のuniformity警告は出ていない。
- step1のlossは、LoRAとFull FTで同じ値（7.626703262329102）だった。単一フレームのA1では、小数第7位がずれていた。この2方式の間のbitwise一致は成功条件に含まれない。
- **10stepの時点では、Queueの65,536件のうち本物のkeyはまだ2,560件しかない。上の値から、学習の良し悪し、collapse、lossが下がるか、LoRAとFull FTのどちらが優れているかについては何も主張しない。**

## 時間

ファイルの更新時刻から見た目安：

| 区間 | 時間 |
|---|---|
| 起動 → 固定probeの作成終了 | 約2分27秒 |
| → step1の終了（最初のbatchの待ちと、probe step0・1を含む） | 約3分43秒 |
| step1 → step10 | 約104秒（step_timeの合計は87秒。probe 2を含む） |

- **1stepの計算時間（`step_time_sec`）は、step2〜10で9.51〜9.74秒（約9.6秒）**。単一フレームA1 Full FTは約1.74秒で、約5.5倍だった。
- LoRAと同じく、10stepでは定常状態のデータ読み込み速度は測れていない。見積もりでは1 batchあたり約3〜4秒で、計算時間より短い（[A1 LoRA Evidence](2026-11-03-vit-moco-lora-activitynet-clip-a1-lora-gpu-evidence.md)）。
- **A2の所要時間の見積もり（未測定）**：
  - LoRA：459step × 約8.8秒 ＋ 起動約4分 ＝ **約70分**
  - Full FT：459step × 約9.6秒 ＋ 起動約4分 ＝ **約77分**
  - どちらも、probe（100・256・459）とcheckpointの時間は含んでいない。単一フレームのA2は、LoRAが約35分、Full FTが約25分だった。

## 未実施

- A2（LoRA → Full FT、各459step）。別に許可が必要。spec §7のとおり、1stepの時間が単一フレームA2（LoRA約3.8秒、Full FT約2.5秒）より大幅に長いため、A2を実行するかはユーザーが判断する。
