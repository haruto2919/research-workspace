---
project: sequential-video-lora-analysis
date: 2026-11-03
record_type: gpu-run-evidence
status: a2-lora-engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: 9155c2baa59c06f8204b81d7d466bf4fc4e955d6
spec: ../specs/2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md
---

# ViT＋MoCo ActivityNet単一フレーム：GPU A2（LoRA・459step＝1epoch）

A1が両方式とも工学PASSとなった（[LoRA](2026-11-02-vit-moco-lora-activitynet-a1-lora-gpu-evidence.md)、[Full FT](2026-11-03-vit-moco-lora-activitynet-a1-full-ft-gpu-evidence.md)）のを受けて、ユーザーの許可を得てA2のLoRAをtmuxセッション `vit-moco-a1` から実行した。**459stepを完走し、工学PASS**となった。Full FTのA2は未実施。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet model.train_mode=lora training.max_steps=459 training.learning_rate=0.0003
```

- spec §7のコマンドそのまま。probe stepとcheckpointはpresetの既定値（probe `[0,1,2,10,100,256,459]`、checkpoint `[459]`）。
- 研究リポジトリ：`9155c2b`、clean。GPU構成はL4/L5と同じ（GPU0：RTX PRO 4500 Blackwell、GPU1：RTX A4500）。開始前は両GPUとも空き。
- 出力：`log/moco_vit_lr_activitynet/lora/activitynet_frames/lr_0.0003/2026-11-03_10-48-36-408110/`
- 時刻：10:48:29開始 → 11:23:30終了（run本体は2,093秒＝約35分）。

## 工学チェック

| 項目 | 結果 |
|---|---|
| run / engineering | `completed` / **PASS**（理由なし、`failures=[]`、rank reportの欠落なし） |
| step | 459 / 459、全stepでfinite |
| checkpoint | `checkpoint_step000459.pt`（約702 MiB）とrankごとのRNG sidecar 2個 |
| VRAM gate | PASS。peak reserved：GPU0 12,190 MiB（headroom 20,028）、GPU1 12,338 MiB（headroom 7,758） |
| final checks | 全rankがrank0と同じ状態、frozen baseは不変、frozen keyはqueryと一致 |
| 固定probe | 入力ハッシュ `6af50745…08d8bf`（**L4/L5と一致**）。probe step 0・1・2・10・100・256・459がすべて記録された |
| 警告ヒューリスティック | 一度も出ていない |
| queue | step256以降はすべて本物のkey。step459でqueue_ptr 51,968（= 459 × 256 mod 65,536） |

- `ranks_feature_identical` は `false`。A1とL5でも同じで、成功条件の項目ではない。

### A1（10step）とのprefix比較（`--comparison-axis prefix`、記録用）

A1 LoRAの再実行（`2026-11-02_13-57-26-045737`）と本runの共通10stepを比較した。結果はstdoutだけで、ファイルには保存していない。

| 項目 | 結果 |
|---|---|
| 学習入力（共通10step） | 一致（違うstepなし） |
| probe入力 | 一致 |
| step0の特徴（CLS / z_q / q） | bitwise一致 |
| step1のloss | **bitwise一致**（7.68887996673584） |
| step2〜10のloss | step 2・5・7で4.77e-7（float32の1 ulp程度）ずれた。他のstepは一致 |
| step1以降のprobe特徴 | 不一致。そのため比較器の判定は `comparable=false` |

spec §8は、run間のbitwise一致を成功条件にしていない。このprefix比較は記録用で、step0とstep1のlossのbitwise一致を確かめ、それ以降はずれの大きさを報告するものである。この結果は、L5で分かったGPU backwardのrun間非決定性と同じ形をしている（forwardは再現し、最初のbackward以降でわずかにずれる）。

## 学習batchの値（rank0、記述のみ）

約50stepの窓ごとの平均：

| 窓（step） | loss | positive cos | negative cos | hardest neg cos | q std |
|---|---|---|---|---|---|
| 1–50 | 8.012 | 0.726 | 0.037 | 0.747 | 0.0632 |
| 51–100 | 8.450 | 0.687 | 0.084 | 0.733 | 0.0768 |
| 101–150 | 8.645 | 0.647 | 0.103 | 0.701 | 0.0820 |
| 151–200 | 8.746 | 0.600 | 0.093 | 0.662 | 0.0852 |
| 207–256 | 8.753 | 0.524 | 0.024 | 0.588 | 0.0874 |
| 257–306 | 8.671 | 0.461 | −0.057 | 0.523 | 0.0872 |
| 307–356 | 8.580 | 0.472 | −0.066 | 0.533 | 0.0870 |
| 357–406 | 8.480 | 0.489 | −0.072 | 0.546 | 0.0869 |
| 410–459 | 8.382 | 0.511 | −0.071 | 0.566 | 0.0868 |

- **256stepより前**：lossは上昇した（step 256で8.833）。Queueの初期値が本物のkeyに置き換わり、negativeが難しくなっていく期間で、L4/L5と同じく、この期間のlossから学習の良し悪しは判断できない。
- **256stepより後**（Queueがすべて本物のkey）：50step窓のlossは**単調に減少**した（8.671→8.382）。459step時点でも減少は止まっていない。positive cosは0.46→0.51と上がった。
- `query_feature_std` は0.087で、上限1/√128 ≈ 0.0884に近い。
- 平均negative cosは256step以降、−0.06〜−0.07だった（L5のImageNetでは−0.015〜−0.05）。
- 学習率は全期間3e-4で一定（cosineの更新はepoch境界のみ）。

## 固定probe（ImageNet 64枚、rank0、記述のみ）

| step | z_q cos p50 | z_q cos p95 | q mean_dim_std | z_q：step0とのcos p50 | CLS cos p50 |
|---|---|---|---|---|---|
| 0 | 0.319 | 0.449 | 0.0726 | — | 0.032 |
| 1 | 0.455 | 0.564 | 0.0651 | 0.985 | 0.032 |
| 2 | 0.541 | 0.639 | 0.0598 | 0.966 | 0.032 |
| 10 | 0.625 | 0.706 | 0.0540 | 0.940 | 0.032 |
| 100 | 0.199 | 0.364 | 0.0787 | 0.923 | 0.030 |
| 256 | 0.099 | 0.306 | 0.0831 | 0.664 | 0.027 |
| 459 | 0.129 | 0.332 | 0.0818 | 0.602 | 0.026 |

- **collapseは459stepまで観測されなかった**。step10までに0.63まで上がったz_qの方向集中は、step100で0.20、step256で0.10まで下がった。step459では0.13だった。
- CLSは、step0とのcos p50が0.977で、大きくは変わっていない（LoRA rank 8、459step）。
- probeはImageNet画像で、学習データ（ActivityNetのフレーム）とは別の分布である。この値は「同じ画像の表現がどう動いたか」を見るためのもので、ActivityNetでの表現品質は示さない。

## データ読み込みと時間

| 区間 | 時間 |
|---|---|
| 起動 → 固定probeの作成終了 | 約2分35秒 |
| → step1（`update_metrics` の書き込み時刻） | 約3分 |
| step2 → step459 | 約29分（1,749秒、**平均3.8秒/step**） |

- 1stepの計算時間（`step_time_sec`）は中央値1.48秒、最大3.7秒だった。実際の1stepあたりの時間は約3.8秒で、**約2.3秒/stepはデータ読み込みの待ち**である。1 workerが1 batchを作るのに約61秒（16 worker × 3.8秒）かかった計算になる。これは全frame確認で測った平均0.53秒/frame（約68秒/batch）とほぼ合う。
- spec §7の「L5のImageNetより大幅に遅い場合は報告」に当たる。L5のLoRA（ImageNet）は計算時間の中央値1.50秒/step、起動を含む全体で2.03秒/stepだった。A2の実際の時間（約3.8秒/step）はその約1.9倍で、主な原因はActivityNetのフレームのdecodeである。
- Full FTのA2も、データ読み込みで決まると見られる。所要時間は約35分と見積もる（Full FTの計算時間は1.74秒/stepで、データ待ちより短いため）。

## 言えないこと

- 表現品質（kNN、linear probeは未実施）。ActivityNetの固定probeもない。
- 時間情報・動き情報の獲得（単一フレーム学習なので、そもそも対象外）。
- 1epochより長い学習での安定性。
- LoRAとFull FTの優劣（Full FTのA2は未実施）。
- L5（ImageNet）との違いの原因。データが違うため、記述にとどめる。

## 未実施

- A2のFull FT（459step）。別に許可が必要。
