---
project: sequential-video-lora-analysis
date: 2026-11-01
record_type: gpu-diagnostic-local-artifact-evidence
status: l5-both-modes-engineering-pass-prefix-bitwise-fail-gpu-backward-nondeterminism
evidence_level: local-run-report-diagnostic-summary-metrics-and-comparator-output-read-directly-on-research-server
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
local_execution_commit: 387456875e40f7e1ad9037b43bf631abc5221e63
spec: ../specs/2026-10-31-vit-moco-low-lr-long-run-spec.md
previous_evidence: 2026-10-31-vit-moco-low-lr-l4-imagenet-gpu-evidence.md
---

# ViT＋MoCo 低LR長めの学習（L5）：ImageNet 2GPU・各1,000step（LR 3e-4）

## 実行内容

ユーザーが研究コードを `feature-vit-moco@3874568`（"Add long-run query collapse diagnostics"、spec実装の5ファイル）としてcommitし、研究サーバで次の2 runを実行した（2026-11-01）。出力はAIが同じサーバ上で直接読んだ。

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_long model.train_mode=lora training.max_steps=1000 training.learning_rate=0.0003
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_long model.train_mode=full_ft training.max_steps=1000 training.learning_rate=0.0003
```

- LoRA：`log/moco_vit_lr_long/lora/imagenet/lr_0.0003/2026-11-01_00-11-06-779221`（33.8分）
- Full FT：`log/moco_vit_lr_long/full_ft/imagenet/lr_0.0003/2026-11-01_09-37-00-940051`（37.6分）
- 同じディレクトリの `2026-11-01_09-30-10-257648`（Full FT）は、step 31で `KeyboardInterrupt`（手動停止）により `status=failed` となったrunで、結果としては使わない。
- GPU構成はL4と同じ（GPU0：RTX PRO 4500 Blackwell、GPU1：RTX A4500）。

## 工学Gate

| 項目 | LoRA | Full FT |
|---|---|---|
| status / failures | completed / [] | completed / [] |
| commit / dirty | `3874568` / false | `3874568` / false |
| 学習step | 1〜1,000（欠落・重複なし、全step finite、LRは全step 3e-4、epoch 0のみ） | 同左 |
| 両rankのstate一致 | true | true |
| frozen baseが不変 | true | —（frozenなし） |
| 診断の `engineering_status` / profile | PASS / `lr_long` | PASS / `lr_long` |
| probe step | `[0,1,2,10,100,256,500,1000]` 全記録 | 同左 |
| checkpoint | `checkpoint_step001000.pt`（＋rankごとのsidecar） | 同左 |
| peak reserved（rank0 / rank1） | 12,190 / 12,338 MiB | 18,124 / 18,346 MiB（L4と同じ） |
| VRAM Gate | PASS | PASS |
| `uniformity_warnings` | 0件 | 0件 |

**spec §8の決定性条件（L4とのbitwise一致）は不合格**（次節）。

## 決定性：L4とのprefix比較（`--comparison-axis prefix`）

| 比較 | comparable | 一致したもの | 一致しなかったもの |
|---|---|---|---|
| LoRA：L4 vs L5 | **false** | 学習入力（step 1〜10）、probe入力、step0のCLS / z_q / q | probe step 1・2・10の全stage |
| Full FT：L4 vs L5 | **false** | 同上 | 同上 |
| L5：LoRA vs Full FT（`train_mode`軸） | **true** | 学習入力、probe入力、step0特徴 | — |

ずれの大きさ（rank0、fixed probeの統計量の差の絶対値）：

| | step 1 | step 2 | step 10 |
|---|---|---|---|
| LoRA z_q cos p50 | 6.2e-8 | 8.1e-8 | 1.5e-7 |
| Full FT z_q cos p50 | 4.5e-8 | 3.8e-5 | **2.5e-3** |
| Full FT CLS row norm中央値 | 1.8e-7 | 1.7e-4 | 7.3e-3 |

学習batch側（`metrics.jsonl`）では、**step 1のlossは両方式ともbitwise一致**（7.361582279205322）した。一方、step 1の勾配normは13桁目で異なった（LoRA 2.185853998092884 vs …2319）。LoRAではその後もずれはほとんど増えなかったが、Full FTでは急速に増え、step 10のlossが4桁目で異なった（7.41582 vs 7.41598）。

読み取れること：

- 同じハードウェア・同じ入力では、**forward（step0のprobe、step1のloss）はbitwiseで再現**した。したがって、L4の記録でcross-LR比較のstep0不一致をGPU0の機種変更によるものと推定したことと矛盾しない。
- **backwardはrun間でbitwise再現しない**。GPUの学習経路は、同じcommit相当の学習コード・同じハードウェアでもrun-to-runの再現性がない。GPUで同じ設定を2回実行して比べたのは今回が初めてであり、そのため今回初めて見えた。
- `resolved_config` では `cudnn_deterministic=True` だが、`deterministic_algorithms=False`、`CUBLAS_WORKSPACE_CONFIG` は未設定である。原因の候補は、attentionのbackward（SDPAのmemory-efficient系kernel）やcuBLASなどの非決定的なkernelだが、**特定はしていない**。
- 3874568で変わったのはpreset、検証部分、比較器、テストだけで、学習のstep計算は変わっていない（コード差分による説明は考えにくい）。

spec §8は「一致しない場合は報告し、原因を調べるまで科学的な解釈を保留する」と定めている。以下の科学的観測は**記述として記録するが、解釈は暫定**とする。ただし、ずれの大きさ（最大でもstep10のcos p50で2.5e-3）は、次に示す変化（cos p50 0.45→0.005など）より2桁以上小さい。

## 科学的観測（記述）

### 学習loss（rank0、100step窓の平均 loss / positive cosine）

| 窓 | LoRA | Full FT |
|---|---|---|
| 1–100 | 7.793 / 0.761 | 7.788 / 0.763 |
| 101–200 | 8.138 / 0.701 | 8.120 / 0.705 |
| 201–300 | 8.141 / 0.607 | 8.087 / 0.617 |
| 301–400 | 7.985 / 0.598 | 7.896 / 0.614 |
| 401–500 | 7.830 / 0.631 | 7.721 / 0.651 |
| 501–600 | 7.699 / 0.666 | 7.574 / 0.691 |
| 601–700 | 7.607 / 0.695 | 7.490 / 0.720 |
| 701–800 | 7.503 / 0.725 | 7.387 / 0.751 |
| 801–900 | 7.433 / 0.746 | 7.328 / 0.771 |
| 901–1000 | **7.365** / 0.767 | **7.282** / 0.787 |

- **256stepより前**：lossは上昇した（MA100の最大はLoRA 8.177（step 241で終わる窓）、Full FT 8.145（step 235））。Queueの初期値（本物のkeyでないベクトル）が本物のkeyに置き換わり、negativeが難しくなっていく期間である。この期間のlossは「学習で下がったか」の判断には使えない。
- **256stepより後**（Queueがすべて本物のkey）：100step窓のlossは**単調に減少**した。257–356窓→901–1000窓で、LoRAは8.059→7.365、Full FTは7.991→7.282。1,000step時点でも減少は止まっていない。
- positive cosineは、Queueが埋まる頃に0.59〜0.60まで下がり、その後0.77〜0.79まで戻った。hardest negative cosineも同様に推移し、最後の窓ではpositiveと同程度かやや上だった（LoRA 0.770、Full FT 0.810）。
- 平均negative cosineは256step以降、−0.015〜−0.05とほぼ0だった。学習batchの `query_feature_std` は0.0878で、128次元の単位ベクトルが取りうる上限 1/√128 ≈ 0.0884 にほぼ等しい。
- 学習率は全期間3e-4で一定（cosineの更新はepoch境界のみ）。lossの減少はLRの減衰によるものではない。

### 固定64画像probe（rank0、画像間cos p50 / mean_dim_std）

| step | LoRA CLS | LoRA z_q | LoRA q std | Full FT CLS | Full FT z_q | Full FT q std |
|---|---|---|---|---|---|---|
| 0 | 0.0319 | 0.319 | 0.0726 | 0.0319 | 0.319 | 0.0726 |
| 1 | 0.0319 | 0.453 | 0.0651 | 0.0380 | 0.495 | 0.0625 |
| 2 | 0.0319 | 0.530 | 0.0604 | 0.0365 | 0.564 | 0.0582 |
| 10 | 0.0319 | 0.455 | 0.0649 | 0.0289 | 0.394 | 0.0684 |
| 100 | 0.0316 | 0.106 | 0.0830 | 0.0325 | 0.109 | 0.0829 |
| 256 | 0.0293 | 0.0028 | 0.0876 | 0.0357 | 0.0097 | 0.0872 |
| 500 | 0.0282 | 0.0065 | 0.0873 | 0.0332 | 0.0074 | 0.0874 |
| 1000 | 0.0274 | **0.0046** | **0.0874** | 0.0322 | **0.0004** | **0.0876** |

- **遅れて起きるcollapseは、1,000stepまで観測されなかった**。step 1〜2で0.45〜0.56まで上がったz_qの方向集中は、step 100で0.11、step 256以降はほぼ0まで下がった。q stdは上限付近（0.087〜0.088）まで広がった。警告ヒューリスティック（q std < 1e-4 かつ cos p50 > 0.999）は一度も出ていない。
- CLSの画像間cosは、両方式とも事前学習時の値（0.03前後）からほとんど動かなかった。step0からの変化（cosine_to_step0 p50）は、step 1,000でLoRA 0.960、Full FT 0.852だった。Full FTの方がbackboneの表現が大きく動いている。
- Projector出力 z_q のrow norm中央値は、LoRAで2.27→3.73、Full FTで2.27→4.18と緩やかに増えた（LR 0.03では10stepで655 / 5,394に達していた）。

## 判定と非主張

- **工学**：L5は両方式とも完走し、工学Gate PASS。ただし、spec §8の「L4とのbitwise一致」は**不合格**。原因はGPU backwardのrun間非決定性と推定しているが、kernelは特定していない。
- **記述的な結果（暫定）**：LR 3e-4・1,000stepでは、Queueが埋まった後のMoCo lossが両方式で単調に減少し、遅れて起きるcollapseは観測されなかった。Projectorとqueryの出力は、むしろ方向が分散する方向に進んだ。
- **言えないこと**：表現品質（kNN、linear probeは未実施）、1epoch以上での安定性、LoRAとFull FTの優劣（Full FTのlossが約0.08低いのは単一seedの記述にすぎない）、最適LR、0.03との厳密な因果比較、lossの減少が学習とQueue key一貫性のどちらによるものか。

## 次の候補（未承認）

1. **GPU決定性の切り分け**：同じpreset（L4、10step）を同じcommitで2回実行し、run間のずれの基準値（noise floor）を測る。必要に応じて、`torch.use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG` / eager attention で再現性が回復するかを確認する。設定を変える場合は別specとする。
2. **表現品質の評価spec**：1,000stepのcheckpointに対して、kNNまたはlinear probeを行う（spec B6で対象外としたもの）。
3. **1epoch（5,004step）以上への延長**：spec B1により、別途許可が必要。
4. （以前からの候補）同じハードウェアで0.03 baselineを再実行する。ただし1により、比較はbitwiseではなく、許容誤差付きの判定にする必要がある可能性がある。
