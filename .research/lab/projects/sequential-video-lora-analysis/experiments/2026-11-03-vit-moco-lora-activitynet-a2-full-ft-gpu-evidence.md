---
project: sequential-video-lora-analysis
date: 2026-11-03
record_type: gpu-run-evidence
status: a2-full-ft-engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: 9155c2baa59c06f8204b81d7d466bf4fc4e955d6
spec: ../specs/2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md
---

# ViT＋MoCo ActivityNet単一フレーム：GPU A2（Full FT・459step＝1epoch）

[A2 LoRA](2026-11-03-vit-moco-lora-activitynet-a2-lora-gpu-evidence.md)が工学PASSとなったのを受けて、ユーザーの許可を得てA2のFull FTをtmuxセッション `vit-moco-a1` から実行した。**459stepを完走し、工学PASS**となった。LoRAのA2との比較器は `comparable=true` である。**これで、spec §8の工学的な成功条件（A1・A2、両方式）がすべて揃った。**

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet model.train_mode=full_ft training.max_steps=459 training.learning_rate=0.0003
```

- spec §7のコマンドそのまま。probe step `[0,1,2,10,100,256,459]`、checkpoint `[459]`（presetの既定値）。
- 研究リポジトリ：`9155c2b`、clean。GPU構成はL4/L5と同じ。開始前は両GPUとも空き。
- 出力：`log/moco_vit_lr_activitynet/full_ft/activitynet_frames/lr_0.0003/2026-11-03_11-49-57-818631/`
- 時刻：11:49:50開始 → 12:14:42終了（run本体は1,483秒＝約25分）。

## 工学チェック

| 項目 | 結果 |
|---|---|
| run / engineering | `completed` / **PASS**（理由なし、`failures=[]`、rank reportの欠落なし） |
| step | 459 / 459、全stepでfinite |
| checkpoint | `checkpoint_step000459.pt`（約1.32 GiB）とrankごとのRNG sidecar 2個 |
| VRAM gate | PASS。peak reserved：GPU0 18,124 MiB（headroom 14,094）、GPU1 18,346 MiB（headroom **1,750**、閾値1,024）。A1 Full FTと同じ値 |
| final checks | 全rankがrank0と同じ状態。Full FTなのでfrozen parameterは0 |
| 固定probe | 入力ハッシュ `6af50745…08d8bf`（**L4/L5と一致**）。probe step 7個がすべて記録された |
| 警告ヒューリスティック | 一度も出ていない |

- `ranks_feature_identical` は `false`。他のrunと同じで、成功条件の項目ではない。

### LoRAのA2との比較（`--comparison-axis train_mode`）

| 項目 | 結果 |
|---|---|
| `comparable` | **true** |
| 学習入力（全459step） | 一致（違うstepなし） |
| probe入力 | 一致 |
| step0の特徴（CLS / z_q / q） | すべてbitwise一致 |

### A1 Full FT（10step）とのprefix比較（`--comparison-axis prefix`、記録用）

| 項目 | 結果 |
|---|---|
| 学習入力（共通10step）・probe入力 | 一致 |
| step0の特徴 | bitwise一致 |
| step1〜3のloss | **bitwise一致** |
| step4〜10のloss | ずれた。最大3.3e-3（step 8） |
| step1以降のprobe特徴 | 不一致（比較器の判定は `comparable=false`） |

spec §8ではrun間のbitwise一致を求めておらず、この比較は記録用である。L5でも、Full FTのずれはLoRAより速く大きくなった。今回も同じ傾向で、LoRAのprefix比較のずれ（最大4.8e-7）より大きい。

比較器の結果は、いずれもstdoutだけで、ファイルには保存していない。

## 学習batchの値（rank0、記述のみ）

約50stepの窓ごとの平均（LoRAは[A2 LoRA](2026-11-03-vit-moco-lora-activitynet-a2-lora-gpu-evidence.md)から）：

| 窓（step） | loss Full FT | loss LoRA | positive cos Full FT | negative cos Full FT | q std Full FT |
|---|---|---|---|---|---|
| 1–50 | 7.999 | 8.012 | 0.728 | 0.037 | 0.0633 |
| 51–100 | 8.417 | 8.450 | 0.695 | 0.085 | 0.0765 |
| 101–150 | 8.587 | 8.645 | 0.661 | 0.104 | 0.0817 |
| 151–200 | 8.659 | 8.746 | 0.619 | 0.093 | 0.0851 |
| 207–256 | 8.613 | 8.753 | 0.554 | 0.025 | 0.0873 |
| 257–306 | 8.498 | 8.671 | 0.497 | −0.057 | 0.0870 |
| 307–356 | 8.375 | 8.580 | 0.515 | −0.067 | 0.0869 |
| 357–406 | 8.240 | 8.480 | 0.539 | −0.072 | 0.0868 |
| 410–459 | 8.123 | 8.382 | 0.567 | −0.072 | 0.0867 |

- **256stepより前**：lossは上昇した（50step窓の最大は151–200窓の8.659）。LoRA・L4/L5と同じく、この期間のlossから学習の良し悪しは判断できない。
- **256stepより後**：50step窓のlossは**単調に減少**した（8.498→8.123）。459step時点でも減少は止まっていない。
- 同じ窓で、Full FTのlossはLoRAより低く、差は256step以降に約0.17→0.26と広がった。**単一seed・1epochの記述であり、LoRAとFull FTの優劣は主張しない**（spec §8）。

## 固定probe（ImageNet 64枚、rank0、記述のみ）

| step | z_q cos p50 Full FT | z_q cos p50 LoRA | z_q：step0とのcos p50 Full FT | CLS：step0とのcos p50 Full FT | CLS：同 LoRA |
|---|---|---|---|---|---|
| 0 | 0.319 | 0.319 | — | — | — |
| 1 | 0.518 | 0.455 | 0.944 | 0.970 | 1.000 |
| 2 | 0.610 | 0.541 | 0.904 | 0.945 | 1.000 |
| 10 | 0.584 | 0.625 | 0.869 | 0.906 | 1.000 |
| 100 | 0.177 | 0.199 | 0.830 | 0.908 | 0.995 |
| 256 | 0.105 | 0.099 | 0.544 | 0.879 | 0.985 |
| 459 | 0.107 | 0.129 | 0.488 | 0.856 | 0.977 |

- **collapseは459stepまで観測されなかった**（両方式）。z_qの方向集中は、step2〜10で約0.6まで上がった後、step256以降は約0.1に下がった。
- Full FTでは、CLS特徴がstep0から大きく動いた（step459で0.856）。LoRAは0.977だった。Full FTはbackbone全体を更新するので、この差は方式の違いとして予想される範囲である。
- probeはImageNet画像で、ActivityNetでの表現品質は示さない。

## データ読み込みと時間

| 区間 | 時間 |
|---|---|
| 起動 → 固定probeの作成終了 | 約2分35秒 |
| → step1（`update_metrics` の書き込み時刻） | 約3分10秒 |
| step2 → step459 | 約19分（1,132秒、**平均2.5秒/step**） |

- 1stepの計算時間（`step_time_sec`）は中央値1.78秒だった。データ読み込みの待ちは約0.7秒/stepで、直前のLoRA A2（約2.3秒/step）より短かった。
- 理由は確認していない。同じseedなので、LoRAとFull FTは同じframeを同じ順番で読む（比較器で学習入力の一致を確認済み）。そのため、LoRAで読んだ動画がOSのファイルキャッシュに残っていた可能性がある（NASはCIFSで `cache=strict`、終了後のbuff/cacheは約87 GiB）。LoRAとFull FTのA2で、データ待ちの条件は同じではない。

## spec §8の工学的な成功条件のまとめ

| 条件 | A1 LoRA | A1 Full FT | A2 LoRA | A2 Full FT |
|---|---|---|---|---|
| completed・failures無し・finite・rank一致・VRAM PASS・engineering PASS | ✓ | ✓ | ✓ | ✓ |
| probe入力ハッシュがL4/L5と一致 | ✓ | ✓ | ✓ | ✓ |
| 同じ長さのLoRAとFull FTで `comparable=true` | ✓（A1同士） | ✓ | ✓（A2同士） | ✓ |
| 459stepでcheckpoint（step 459）と全probe stepが揃う | — | — | ✓ | ✓ |
| prefix比較（記録用）：step0の特徴・step1のlossがbitwise一致 | — | — | ✓ | ✓ |

## 言えないこと

- 表現品質（kNN、linear probeは未実施）。ActivityNetの固定probeもない。
- 時間情報・動き情報の獲得（単一フレーム学習なので対象外）。
- 1epochより長い学習での安定性。
- LoRAとFull FTの優劣、最適LR。
- L5（ImageNet）との違いの原因（データが違うため記述にとどめる）。
