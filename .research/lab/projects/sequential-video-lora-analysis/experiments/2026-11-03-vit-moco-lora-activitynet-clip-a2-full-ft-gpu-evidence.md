---
project: sequential-video-lora-analysis
date: 2026-11-03
record_type: gpu-run-evidence
status: a2-full-ft-engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: fe58fc37ef28901184980ccd5f9e90e40c3d836d
spec: ../specs/2026-11-03-vit-moco-lora-activitynet-clip-early-fusion-spec.md
---

# ViT＋MoCo ActivityNet clip Early Fusion：GPU A2（Full FT・459step＝1epoch）

[clip A2 LoRA](2026-11-03-vit-moco-lora-activitynet-clip-a2-lora-gpu-evidence.md)が工学PASSとなったのを受けて、ユーザーの許可を得てA2のFull FTをtmuxセッション `vit-moco-a1` から実行した。**459stepを完走し、工学PASS**となった。LoRAとの比較器も `comparable=true` である。**Queueが埋まった後（256stepより後）、50step窓のlossは単調に減少し、collapseは観測されなかった（暫定）**。所要時間は約87分だった。これで、**clip Early Fusion specの工学的な成功条件（A1・A2、両方式）がすべて揃った**。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet_clip model.train_mode=full_ft training.max_steps=459 training.learning_rate=0.0003
```

- spec §7のコマンドをそのまま使った。probe stepとcheckpointはpresetの既定値（probe `[0,1,2,10,100,256,459]`、checkpoint `[459]`）。
- 研究リポジトリ：`fe58fc3`、clean。GPU構成はA2 LoRAと同じ。開始前、両GPUとも空いていた。
- 出力：`log/moco_vit_lr_activitynet_clip/full_ft/activitynet_clips/lr_0.0003/2026-11-03_16-46-46-287044/`
- 時刻：16:46:39開始 → 18:13:46終了（run本体は5,218秒＝約87分）。

## 工学チェック

| 項目 | 結果 |
|---|---|
| run / engineering | `completed` / **PASS**（理由なし、`failures=[]`、rank reportの欠落なし） |
| step | 459 / 459、全stepでfinite |
| checkpoint | `checkpoint_step000459.pt`（約1.4 GiB）と、rankごとのRNG sidecar 2個 |
| VRAM gate | PASS。peak reserved：GPU0 13,112 MiB（headroom 19,106）、GPU1 13,112 MiB（headroom **6,984**、閾値1,024）。A1 Full FTと同じ値 |
| final checks | 全rankがrank0と同じ状態。Full FTなので、frozen parameterは0 |
| 固定probe | 元の64枚の入力ハッシュ `6af50745…08d8bf`（一致）。静止clipの入力ハッシュ `457fabe0…cd1f`（A2 LoRAと一致）。probe stepは7個とも記録された |
| 警告ヒューリスティック | 一度も出ていない |
| queue | step459でqueue_ptr 51,968 |

- `ranks_feature_identical` は `false`。他のrunと同じで、成功条件の項目ではない。
- 終了時に、torchのUserWarning（`barrier(): using the device under current context`）が1回出た。A2 LoRAと同じもので、runの判定には影響しない。

### LoRAのA2との比較（`--comparison-axis train_mode`）

| 項目 | 結果 |
|---|---|
| `comparable` | **true** |
| 学習入力（全459step） | 一致（違うstepはない） |
| probe入力 | 一致 |
| step0の特徴（CLS / z_q / q） | すべてbitwise一致 |

### A1 Full FT（10step）とのprefix比較（`--comparison-axis prefix`、記録用）

| 項目 | 結果 |
|---|---|
| 学習入力（共通10step）・probe入力 | 一致 |
| step0の特徴 | bitwise一致 |
| step1〜2のloss | **bitwise一致**（step1は7.626703262329102） |
| step3〜10のloss | ずれた。最大3.1e-3（step 9） |
| step1以降のprobe特徴 | 不一致（比較器の判定は `comparable=false`） |

spec §8はrun間のbitwise一致を求めておらず、この比較は記録用である。Full FTのずれがLoRA（最大1.9e-6）より大きくなる傾向は、単一フレームのA2（Full FTで最大3.3e-3）やL5と同じである。

比較器の結果はどれも標準出力に出しただけで、ファイルには保存していない。

## 学習batchの値（rank0、記述のみ）

約50stepの窓ごとの平均。LoRAの値は[clip A2 LoRA](2026-11-03-vit-moco-lora-activitynet-clip-a2-lora-gpu-evidence.md)から引いた。

| 窓（step） | loss Full FT | loss LoRA | positive cos Full FT | negative cos Full FT | hardest neg cos Full FT | q std Full FT |
|---|---|---|---|---|---|---|
| 1–50 | 7.945 | 7.947 | 0.736 | 0.036 | 0.761 | 0.0640 |
| 51–100 | 8.339 | 8.373 | 0.706 | 0.083 | 0.749 | 0.0771 |
| 101–150 | 8.508 | 8.580 | 0.671 | 0.101 | 0.719 | 0.0821 |
| 151–200 | 8.570 | 8.677 | 0.631 | 0.089 | 0.682 | 0.0853 |
| 207–256 | 8.520 | 8.685 | 0.569 | 0.022 | 0.617 | 0.0872 |
| 257–306 | 8.396 | 8.599 | 0.518 | −0.056 | 0.561 | 0.0871 |
| 307–356 | 8.259 | 8.496 | 0.538 | −0.065 | 0.576 | 0.0870 |
| 357–406 | 8.124 | 8.399 | 0.563 | −0.070 | 0.593 | 0.0869 |
| 410–459 | 8.010 | 8.309 | 0.590 | −0.069 | 0.617 | 0.0868 |

- **256stepより前**：lossは上昇した（50step窓の最大は151–200窓の8.570、step 256で8.593）。この期間のlossからは、学習の良し悪しを判断できない。
- **256stepより後**：50step窓のlossは**単調に減少**した（8.396→8.010、step459では7.914）。459stepの時点でも、減少は止まっていない。positive cosは0.52→0.59に上がった。
- 同じ窓で比べると、Full FTのlossはLoRAより低く、差は256step以降に約0.20→0.30と広がった。単一フレームのA2でも同じ傾向だった（約0.17→0.26）。**単一seed・1epochの記述であり、LoRAとFull FTの優劣は主張しない**（spec §8）。

## 固定probe（ImageNet 64枚の静止clip、rank0、記述のみ）

| step | z_q cos p50 Full FT | z_q cos p50 LoRA | z_q：step0とのcos p50 Full FT | CLS：step0とのcos p50 Full FT | CLS：step0とのcos p50 LoRA |
|---|---|---|---|---|---|
| 0 | 0.318 | 0.318 | — | — | — |
| 1 | 0.524 | 0.452 | 0.923 | 0.948 | 1.000 |
| 2 | 0.611 | 0.536 | 0.902 | 0.940 | 1.000 |
| 10 | 0.572 | 0.602 | 0.896 | 0.921 | 1.000 |
| 100 | 0.180 | 0.182 | 0.840 | 0.910 | 0.994 |
| 256 | 0.086 | 0.090 | 0.562 | 0.891 | 0.985 |
| 459 | 0.126 | 0.109 | 0.494 | 0.872 | 0.978 |

- **両方式とも、459stepまでcollapseは観測されなかった**。z_qの方向集中は、step2〜10で約0.6まで上がった後、step256以降は約0.1まで下がった。
- Full FTでは、CLS特徴がstep0から大きく動いた（step459で0.872。LoRAは0.978）。単一フレームのA2（Full FT 0.856、LoRA 0.977）と同じ傾向である。
- このprobeは静止clipで、ActivityNetでの表現品質は示さない。

## 時間

| 区間 | 時間 |
|---|---|
| 起動 → 固定probeの作成終了 | 約2分29秒 |
| → step1の終了（`update_metrics` の書き込み時刻） | 約3分44秒 |
| step1 → step459 | 約80分29秒（4,829秒、**平均10.5秒/step**） |

- 1stepの計算時間（`step_time_sec`）は、中央値9.80秒、最大10.13秒だった。実際の1stepあたりの時間との差は約0.75秒/stepで、これにはprobeとcheckpointの書き込みも含まれる。したがって、データ読み込みの待ちは多くても約0.75秒/stepで、**計算で律速された**。
- 所要時間は約87分で、見積もり（約87分）どおりだった。単一フレームのA2 Full FTは約25分だった。

## spec §8の工学的な成功条件のまとめ

| 条件 | A1 LoRA | A1 Full FT | A2 LoRA | A2 Full FT |
|---|---|---|---|---|
| completed・failures無し・finite・rank一致・VRAM PASS・engineering PASS | ✓ | ✓ | ✓ | ✓ |
| frozen baseが不変（LoRA） | ✓ | — | ✓ | — |
| 元の64枚のprobe入力ハッシュが `6af50745…` | ✓ | ✓ | ✓ | ✓ |
| 静止clipの入力ハッシュが、同じ長さのLoRAとFull FTで一致 | ✓ | ✓ | ✓ | ✓ |
| 同じ長さのLoRAとFull FTで `comparable=true` | ✓（A1同士） | ✓ | ✓（A2同士） | ✓ |
| 459stepで、checkpoint（step 459）と全probe stepが揃う | — | — | ✓ | ✓ |
| prefix比較（記録用）：step0の特徴とstep1のlossがbitwise一致 | — | — | ✓ | ✓ |

## 言えないこと

- 表現品質（kNNやlinear probeは未実施）。ActivityNetの固定probeもない。
- 時間情報・動き情報を獲得したかどうか。MoCoのpositiveが同じclipの2拡張なので、フレームを平均するだけでも解ける（spec §8）。
- 1epochより長い学習での安定性。
- LoRAとFull FTの優劣、最適なLR、0.25秒・T=4という設定の妥当性。
- 単一フレームのA2との違いの原因。入力が違うため、記述にとどめる。
