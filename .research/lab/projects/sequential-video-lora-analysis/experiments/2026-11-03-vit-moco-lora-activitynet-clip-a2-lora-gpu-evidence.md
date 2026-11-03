---
project: sequential-video-lora-analysis
date: 2026-11-03
record_type: gpu-run-evidence
status: a2-lora-engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: fe58fc37ef28901184980ccd5f9e90e40c3d836d
spec: ../specs/2026-11-03-vit-moco-lora-activitynet-clip-early-fusion-spec.md
---

# ViT＋MoCo ActivityNet clip Early Fusion：GPU A2（LoRA・459step＝1epoch）

clipのA1が両方式とも工学PASSとなった（[LoRA](2026-11-03-vit-moco-lora-activitynet-clip-a1-lora-gpu-evidence.md)、[Full FT](2026-11-03-vit-moco-lora-activitynet-clip-a1-full-ft-gpu-evidence.md)）。そこで、ユーザーの許可を得て、A2のLoRAをtmuxセッション `vit-moco-a1` から実行した。**459stepを完走し、工学PASS**となった。**Queueが埋まった後（256stepより後）、50step窓のlossは単調に減少し、collapseは観測されなかった（暫定）**。所要時間は約82分だった。Full FTのA2は未実施。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet_clip model.train_mode=lora training.max_steps=459 training.learning_rate=0.0003
```

- spec §7のコマンドをそのまま使った。probe stepとcheckpointはpresetの既定値である（probe `[0,1,2,10,100,256,459]`、checkpoint `[459]`）。
- 研究リポジトリ：`fe58fc3`、clean。GPU構成はA1と同じ（GPU0：RTX PRO 4500 Blackwell、GPU1：RTX A4500）。開始前は両GPUとも空いていた。
- 出力：`log/moco_vit_lr_activitynet_clip/lora/activitynet_clips/lr_0.0003/2026-11-03_15-22-36-985580/`
- 時刻：15:22:29開始 → 16:44:13終了（run本体は4,895秒＝約82分）。

## 工学チェック

| 項目 | 結果 |
|---|---|
| run / engineering | `completed` / **PASS**（理由なし、`failures=[]`、rank reportの欠落なし） |
| step | 459 / 459、全stepでfinite |
| checkpoint | `checkpoint_step000459.pt`（約702 MiB）と、rankごとのRNG sidecar 2個 |
| VRAM gate | PASS。peak reserved：GPU0 12,784 MiB（headroom 19,434）、GPU1 12,784 MiB（headroom 7,312）。A1と同じ |
| final checks | 全rankがrank0と同じ状態。frozen baseは変化なし。frozenのkeyはqueryと一致 |
| 固定probe | 元の64枚の入力ハッシュは `6af50745…08d8bf`（一致）。静止clipの入力ハッシュは `457fabe0…cd1f`（A1と一致）。probe step 0・1・2・10・100・256・459をすべて記録した |
| 警告ヒューリスティック | 一度も出ていない |
| queue | step256以降はすべて本物のkey。step459のqueue_ptrは51,968（= 459 × 256 mod 65,536） |

- `ranks_feature_identical` は `false`。A1と単一フレームのrunでも同じ値で、成功条件の項目ではない。
- 終了時に、`barrier(): using the device under current context` というtorchのUserWarningが1回出た。単一フレームのA2 LoRAでも同じものが出ており、runの判定には影響しない。

### A1（10step）とのprefix比較（`--comparison-axis prefix`、記録用）

clip A1 LoRA（`2026-11-03_15-01-35-356834`）と本runで、共通する最初の10stepを比較した。結果は標準出力にだけ出し、ファイルには保存していない。

| 項目 | 結果 |
|---|---|
| 学習入力（共通10step） | 一致（違うstepはない） |
| probe入力 | 一致 |
| step0の特徴（CLS / z_q / q） | bitwise一致 |
| step1のloss | **bitwise一致**（7.626703262329102） |
| step2〜10のloss | step 4・9・10で4.8e-7〜1.9e-6ずれた。ほかのstepは一致 |
| step1以降のprobe特徴 | 不一致。そのため比較器の判定は `comparable=false` |

spec §8では、run間のbitwise一致を求めていない。この形（forwardは再現し、最初のbackward以降でわずかにずれる）は、単一フレームのA2やL5と同じである。

## 学習batchの値（rank0、記述のみ）

約50stepの窓ごとの平均：

| 窓（step） | loss | positive cos | negative cos | hardest neg cos | q std |
|---|---|---|---|---|---|
| 1–50 | 7.947 | 0.735 | 0.036 | 0.746 | 0.0643 |
| 51–100 | 8.373 | 0.698 | 0.082 | 0.735 | 0.0773 |
| 101–150 | 8.580 | 0.655 | 0.100 | 0.702 | 0.0822 |
| 151–200 | 8.677 | 0.610 | 0.090 | 0.663 | 0.0854 |
| 207–256 | 8.685 | 0.536 | 0.023 | 0.593 | 0.0874 |
| 257–306 | 8.599 | 0.476 | −0.056 | 0.530 | 0.0872 |
| 307–356 | 8.496 | 0.490 | −0.064 | 0.540 | 0.0871 |
| 357–406 | 8.399 | 0.506 | −0.069 | 0.551 | 0.0870 |
| 410–459 | 8.309 | 0.527 | −0.068 | 0.571 | 0.0869 |

- **256stepより前**は、lossが上昇した（step 256で8.836）。Queueの初期値が本物のkeyに置き換わり、negativeが難しくなっていく期間である。単一フレームのA2やL4/L5と同じく、この期間のlossからは学習の良し悪しを判断できない。
- **256stepより後**（Queueはすべて本物のkey）は、50step窓のlossが**単調に減少**した（8.599→8.309、step459で8.240）。459stepの時点でも、減少は止まっていない。positive cosは0.48から0.53に上がった。
- `query_feature_std` は0.087で、上限の1/√128 ≈ 0.0884に近い。
- 学習率は全期間3e-4で一定だった（cosineの更新はepoch境界でだけ行う）。
- 窓ごとの推移の形は、単一フレームのA2 LoRA（256step以降は8.671→8.382）とほぼ同じだった。入力が違うので、lossの値の大小からは何も言わない（spec §8）。

## 固定probe（ImageNet 64枚の静止clip、rank0、記述のみ）

| step | z_q cos p50 | z_q cos p95 | q mean_dim_std | z_q：step0とのcos p50 | CLS cos p50 | CLS：step0とのcos p50 |
|---|---|---|---|---|---|---|
| 0 | 0.318 | 0.446 | 0.0727 | — | 0.032 | — |
| 1 | 0.452 | 0.563 | 0.0652 | 0.986 | 0.032 | 1.000 |
| 2 | 0.536 | 0.634 | 0.0599 | 0.967 | 0.032 | 1.000 |
| 10 | 0.602 | 0.687 | 0.0555 | 0.948 | 0.032 | 1.000 |
| 100 | 0.182 | 0.348 | 0.0795 | 0.925 | 0.031 | 0.994 |
| 256 | 0.090 | 0.287 | 0.0838 | 0.661 | 0.028 | 0.985 |
| 459 | 0.109 | 0.307 | 0.0828 | 0.601 | 0.025 | 0.978 |

- **459stepまで、collapseは観測されなかった**。z_qの方向集中は、step10までに0.60まで上がり、その後step100で0.18、step256で0.09まで下がった。step459では0.11だった。この推移は、単一フレームのA2 LoRA（0.625→0.199→0.099→0.129）とほぼ同じ形である。
- CLSのstep0とのcos p50は0.978で、大きくは変わっていない。
- probeはImageNet画像を4回繰り返した静止clipで、学習データ（ActivityNetの動画clip）とは分布が違う。この値は「同じ入力の表現がどう動いたか」を見るためのもので、ActivityNetでの表現品質は示さない。

## 時間

| 区間 | 時間 |
|---|---|
| 起動 → 固定probeの作成終了 | 約2分27秒 |
| → step1の終了（`update_metrics` の書き込み時刻） | 約3分35秒 |
| step1 → step459 | 約75分17秒（4,517秒、**平均9.86秒/step**） |

- 1stepの計算時間（`step_time_sec`）は、中央値8.88秒、最大9.06秒だった。実際の1stepあたりの時間は約9.9秒である。差の約1.0秒/stepには、probe（10・100・256・459）とcheckpointの書き込みも含まれる。そのため、**データ読み込みの待ちは多くても約1秒/step**である。単一フレームの約2.3秒/stepより短く、A1での見積もりどおり、**計算で律速された**。
- 所要時間は約82分で、見積もり（約70分）より長かった。見積もりに、probe・checkpoint・データ待ちの分を入れていなかったためである。
- Full FTのA2は、計算時間9.6秒/step＋約1秒で、**約87分（約1.5時間）**と見積もる（未測定）。

## 言えないこと

- 表現品質（kNNやlinear probeは未実施）。ActivityNetの固定probeもない。
- 時間情報・動き情報の獲得。MoCoのpositiveが同じclipの2拡張なので、フレームを平均するだけでも解ける（spec §8）。
- 1epochより長い学習での安定性。
- LoRAとFull FTの優劣（Full FTのA2は未実施）。
- 単一フレームのA2との違いの原因。入力が違うため、記述にとどめる。

## 未実施

- A2のFull FT（459step）。別途、許可が必要。
