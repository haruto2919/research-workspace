---
project: sequential-video-lora-analysis
date: 2026-10-21
record_type: gpu-smoke-user-console-evidence
status: console-completed-awaiting-run-report-and-feature-diagnostic
evidence_level: user-supplied-imageNet-2gpu-ten-step-console-only
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: dev
github_branch_head_observed: 2c269915f0c5036fa2a460959fdf401d54456026
local_execution_commit: unverified
spec: ../specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md
---

# ViT DDP MoCo Full FT / LoRA：ImageNet 2GPU 10step（ユーザー提出コンソール）

## Evidence境界と暫定結論

2026-10-21、ユーザーは `CUDA_VISIBLE_DEVICES=0,1`、`training.max_steps=10`、`model.train_mode=lora` / `full_ft` の2本の独立runのコンソールログを提出。両方とも10/10 stepの `finite=true` 行があり、`Saved run artifacts in ...` と表示された。**コンソール上の処理完走を確認**。

**G3 10step VRAM/Final Checkの正式PASSは保留**。run_report.json、rank別report、input_sha256の10step全範囲、actual local HEADは未受領。前回のG3 1step PASSは本runのGate PASSを自動保証しない。

主要な観測：両方式でstep1 lossは7.361582279205322、step1 Query feature stdは0.07224058359861374と一致。一方、step2からQuery feature stdが急落し、Full FTではstep10で6.056817710486939e-06に至る。LoRAもstep10で0.01095892395824194。**normalized MoCo Queryの局所バッチ内表現が強く均一化する可能性がある**。LR 0.03による大きいAdamW更新は有力な仮説だが未証明。学習の成功、特徴品質、Full FT vs LoRAの優劣はここから判断不可。

## 実行コマンド・出力先

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke model.train_mode=lora training.max_steps=10
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke model.train_mode=full_ft training.max_steps=10
```

- LoRA: `log/moco_vit_ddp_smoke/lora/imagenet/2026-10-21_18-53-11-012950`
- Full FT: `log/moco_vit_ddp_smoke/full_ft/imagenet/2026-10-21_18-58-33-409858`
- 最新のGitHubコード確認先: [runner](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/vit_moco/moco_vit_ddp_smoke.py)、[spec](../specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)。ローカル実行時のGit状態はコンソールから未確認。

## ユーザー提供のStep別主要数値（rank0表示）

| Step | LoRA loss | FT loss | LoRA query_feature_std | FT query_feature_std |
|---:|---:|---:|---:|---:|
| 1 | 7.361582 | 7.361582 | 0.072240584 | 0.072240584 |
| 2 | 8.604421 | 8.629482 | 0.006344094 | 0.000811591 |
| 3 | 8.386229 | 8.324354 | 0.005215700 | 0.000200799 |
| 4 | 8.331672 | 8.225353 | 0.005222943 | 0.000031072 |
| 5 | 8.211179 | 8.088413 | 0.006043598 | 0.000013974 |
| 6 | 8.045090 | 7.930787 | 0.008103114 | 0.000008883 |
| 7 | 8.015432 | 7.894660 | 0.008813216 | 0.000006841 |
| 8 | 7.952779 | 7.797044 | 0.006368445 | 0.000006045 |
| 9 | 7.887098 | 7.776325 | 0.009141508 | 0.000005483 |
| 10 | 7.843848 | 7.755354 | 0.010958924 | 0.000006057 |

| 指標 | LoRA step1 → step10 | Full FT step1 → step10 |
|---|---|---|
| grad norm | 2.185854 → 0.0661925 | 2.591367 → 0.0284488 |
| positive cosine | 0.765250 → 0.790720 | 0.765250 → 0.829998 |
| hardest negative cosine | 0.365829 → 0.839814 | 0.365829 → 0.873532 |
| negative cosine mean | -0.0001779 → 0.0224565 | -0.0001779 → 0.0236779 |
| queue_ptr | 256 → 2560 | 256 → 2560 |
| step1 train_step time | 3.215s | 4.591s |
| step10 train_step time | 1.443s | 1.713s |
| finite | true all 10 console steps | true all 10 console steps |

- `query_feature_std` の定義は runnerで `out.q.detach().std(dim=0).mean()`。 `out.q` はMoCoのProjector出力を `F.normalize(...,dim=1)` した**128次元Query表現**であり、ViT CLS直接の標準偏差ではない。rank0のlocal batch 128について算出された指標。
- Step10 Queue置換済みキーは globalB256 × 10 = 2560 / 65536（3.90625%）。大部分のQueueは初期random keysであり、Loss曲線だけで表現品質を測れない。
- Hardest negative cosineがstep10でpositive cosineを超えるが、これだけでは必ずしも異常と断言できない。ただしfeature stdの急減と合わせて診断が必要。
- Lossがstep2以降やや下がることも、学習済み特徴の非collapseを保証しない。
- 全step lr=0.03は採用5epoch cosineでepoch0を維持しているため期待値。ResNet50 MoCo v2のSGD由来数値をAdamWに転用したレシピであり、ViTへの最適性は未検証。

## 現時点での判定・次の最優先作業

**分離判定**：
- 工学（コンソールで10step計算・queue_ptr更新・finite）：観測された範囲では成立。
- 工学（rank間一致・headroom/VRAMの10step正式PASS）：run_report未受領で保留。
- 研究（表現学習の成功）：未成立。むしろ **Full FTのnormalized Queryに非常に強い表現均一化の警戒シグナル**。

1. 両runの`run_report.json`で`status`、`vram_gate.status`、rank別headroom、`final_checks`、failure空、input hash等を確認。
2. 20step継続を機械的に実行する前に、**Query normalized projection上のcollapse兆候**を診断。補助指標として、pre-projector ViT CLSの分散、Projector pre/post normalizationの分散、画像間cosine平均・std・min/max、Query更新量とlayer別grad/update、Keyとの距離を別の診断設計で切り分ける。
3. 原因候補は (a) 事前学習済みViTへのAdamW lr0.03、warmupなし（第一候補）、(b) Projectorのみのcollapse、(c) InfoNCE Query/Key EMAの初期非定常性およびqueue未置換、(d) 同一入力やaugment異常。ログだけで単独犯を確定しない。
4. specで固定されたLR、batch、精度、Kはユーザー再承認なしに変えず、**低LR診断や追加メトリクス実装は別spec/許可の下で行う**。既存ログの値を再解釈して元specを無断に修正しない。20step、ActivityNet frames、長期学習、Linear Probe、時間情報獲得は未確認。

## 関連記録

- [G3 1step Evidence](2026-10-21-vit-ddp-moco-g3-imagenet-1step-console-evidence.md)
- [承認済みViT DDP spec](../specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)
- [2026-10-15 MTG](../meetings/2026-10-15-mtg.md)
