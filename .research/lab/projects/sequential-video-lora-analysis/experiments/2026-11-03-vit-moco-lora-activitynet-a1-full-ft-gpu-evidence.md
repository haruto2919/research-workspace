---
project: sequential-video-lora-analysis
date: 2026-11-03
record_type: gpu-run-evidence
status: a1-full-ft-engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: 9155c2baa59c06f8204b81d7d466bf4fc4e955d6
spec: ../specs/2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md
---

# ViT＋MoCo ActivityNet単一フレーム：GPU A1（Full FT・10step）

[A1 LoRAの再実行](2026-11-02-vit-moco-lora-activitynet-a1-lora-gpu-evidence.md)が工学PASSとなったのを受けて、ユーザーの許可を得てFull FTのA1をtmuxセッション `vit-moco-a1` から実行した。**10stepを完走し、工学PASS**となった。LoRAとの比較器も `comparable=true` で、**A1は両方式とも工学的な成功条件を満たした**。A2は未実施。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet model.train_mode=full_ft training.max_steps=10 training.learning_rate=0.0003 'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'
```

- 研究リポジトリ：`9155c2b`、clean。GPU構成はL4/L5と同じ（GPU0：RTX PRO 4500 Blackwell、GPU1：RTX A4500）。開始前は両GPUとも空き。
- 出力：`log/moco_vit_lr_activitynet/full_ft/activitynet_frames/lr_0.0003/2026-11-03_10-31-07-443680/`
- 時刻：10:31:00開始 → 10:38:06終了（run本体は418秒）。

## 工学チェック

| 項目 | 結果 |
|---|---|
| run / engineering | `completed` / **PASS**（理由なし、`failures=[]`） |
| VRAM gate | PASS。peak reserved：GPU0 18,124 MiB / 32,218（headroom 14,094）、GPU1 18,346 MiB / 20,096（headroom **1,750**、閾値1,024） |
| final checks | 全rankがrank0と同じ状態（state sha256 `1045b512…`）。Full FTなのでfrozen parameterは0 |
| 学習対象 | 86,487,680 parameter（query全体） |
| データ | 117,522 frame、459 step/epoch、frames_sha256 `c9cc4aec…105c` |
| 固定probe | 入力ハッシュ `6af50745…08d8bf`。**L4/L5と一致** |
| probe step | 0・1・2・10がすべて記録された |
| queue | step10でqueue_ptr 2,560 |

- GPU1のheadroomは1,750 MiBで、閾値（1,024 MiB）は満たしているが余裕は小さい。以前のImageNet Full FT（G3）の1,749.875 MiBとほぼ同じ値である。
- `ranks_feature_identical` は `false`。LoRAのA1とL5（ImageNet、両方式）でも同じく `false` であり、成功条件の項目ではない。

### LoRAとの比較（`--comparison-axis train_mode`）

LoRA（`2026-11-02_13-57-26-045737`）とFull FT（本run）を比較器で比べた。結果はstdoutだけで、ファイルには保存していない。

| 項目 | 結果 |
|---|---|
| `comparable` | **true** |
| 学習入力（全10step） | 一致（違うstepなし） |
| probe入力 | 一致 |
| step0の特徴（CLS / z_q / q） | すべてbitwise一致 |

## 学習中の値（参考）

| step | loss | positive cos | negative cos | grad norm | q std | step_time |
|---|---|---|---|---|---|---|
| 1 | 7.689 | 0.700 | −0.0003 | 3.52 | 0.0693 | 9.41 s |
| 2 | 7.549 | 0.736 | 0.0016 | 1.89 | 0.0492 | 1.72 s |
| 5 | 7.720 | 0.729 | 0.0079 | 1.50 | 0.0436 | 1.74 s |
| 10 | 7.775 | 0.745 | 0.0168 | 1.20 | 0.0532 | 1.74 s |

固定probe（rank0）の `z_q` 方向集中（offdiag cos p50）：

| step | LoRA | Full FT |
|---|---|---|
| 0 | 0.319 | 0.319 |
| 1 | 0.455 | 0.518 |
| 2 | 0.541 | 0.610 |
| 10 | 0.625 | 0.583 |

- `q` のuniformity警告は出ていない。
- step1のlossは、LoRA（7.68887996…）とFull FT（7.68887948…）で小数第7位がずれた。step0の特徴と学習入力は一致している。この2方式の間のbitwise一致は成功条件に含まれない。ずれの原因は調べていない。
- **10stepでQueueはまだ2,560 / 65,536しか本物のkeyで埋まっていない。上の値から、学習の良し悪し、collapse、LoRAとFull FTの優劣については何も主張しない。**

## データ読み込み時間

ファイルの更新時刻から見た目安：

| 区間 | 時間 |
|---|---|
| 起動 → 固定probeの作成終了 | 約2分35秒 |
| → step1（最初のbatchの待ちとprobe step0・1を含む） | 約3分10秒 |
| step2 → step10 | 約25秒（step_time 1.72〜1.75 s、probe 2・10を含む） |

- 最初のbatchの待ちは約3分で、LoRAの再実行とほぼ同じだった。
- 1stepの計算はFull FTで約1.74 s、LoRAで約1.45 sだった。A2の所要時間はデータ読み込みの速さで決まると見られ、見積もり（1方式あたり約35分〜1.5時間）は変えない（未確定）。

## 未実施

- A2（LoRA → Full FT、各459step）。別に許可が必要。
