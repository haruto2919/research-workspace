---
project: sequential-video-lora-analysis
date: 2026-10-19
status: completed
evidence_level: controlled-2gpu-fp32-synthetic-16step
scope: A1-16step-numerical-parity
spec: ../specs/2026-10-18-meta-moco-a1-infonce-16step-parity-spec.md
repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
branch: dev
implementation_commit: 1ce9badcbae3e2df08fa400944e4c385785063f9
reference_repository: facebookresearch/moco
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
source: user-shared-terminal-output
---

# Meta公式MoCo v2と自作MoCoのA1・16step・2GPU正式数値一致 — 実験Evidence

> **判定：PASS**（user-provided `judgement.md`、2026-10-19）。本記録はユーザーが研究サーバで実行したterminal outputの転記・解釈であり、ChatGPTがGPU/NCCLを直接起動したり、SSH上のraw trace・JSONを独立に閲覧したものではない。**今回PASSした範囲だけ**を正式Evidenceとする。

## 判定と観測事実

- 総合：`status=PASS`。
- `formal_fp32_parity (M vs S1)=PASS`。許容差`rtol=1e-5, atol=1e-6, threshold_status=fixed`（2026-10-19に[spec](../specs/2026-10-18-meta-moco-a1-infonce-16step-parity-spec.md)で事前承認）。floatは`abs(self-meta)<=atol+rtol*abs(meta)`、discreteは完全一致。
- **M vs S1 の比較は judged=16,256、bitwise equal=16,256、exact failures=0、float failures=0、first failureなし**。これは比較器が採取した**tensor/trace比較エントリ数**であって「16,256 scalar要素」の意味ではない。full tensorはstep 1,8,9,16、small traceは全16step、2rank。
- 再実行M/Mb、S0/S0b、S1/S1bはすべて比較エントリ16,256/16,256でbitwise、`rerun_reproducibility={'M':'BITWISE','S0':'BITWISE','S1':'BITWISE'}`。
- 負対照M vs S0：`FAIL`（fixed超過のfloat tensor/trace entries=9,542/16,256、bitwise=5,268/16,256、exact failures=0）、`s0_control_outcome=DIVERGED`。**総合PASSの要件から独立**。
- S0 vs S1は`DIAGNOSTIC`、制御契約valid。「modeだけが違い、step1のlogits入力まではbitwise一致、全stepのmode非依存discrete一致」、step2以降の自然な計算経路分岐は契約違反と判定しない。初回bitwise差 `step1 rank0 after_logits_before_enqueue/logits/logits`。
- 6runすべて `status=completed`、16/16steps、full steps `[1,8,9,16]`、threshold fixed、ユーザー貼付けのrun verificationで`Failed checks: {}`、runner `ALL SIX RUNS COMPLETED`。

## full-stepのQuery gradient、SGD parameter/momentum比較

Meta M と S1（各rank同一）：

| step | ranks | Query grad exceeded | SGD Query params exceeded | SGD momentum exceeded | max_abs / relative L2 |
|---|---|---|---|---|---|
| 1 | 0,1 | 0/27,966,656 | 0/27,966,656 | 0/27,966,656 | 0/0（すべての3群） |
| 8 | 0,1 | 0/27,966,656 | 0/27,966,656 | 0/27,966,656 | 0/0（すべての3群） |
| 9 | 0,1 | 0/27,966,656 | 0/27,966,656 | 0/27,966,656 | 0/0（すべての3群） |
| 16 | 0,1 | 0/27,966,656 | 0/27,966,656 | 0/27,966,656 | 0/0（すべての3群） |

Meta M と従来S0（rank0とrank1で数値一致）：

| step | grad exceed/27,966,656 | grad max_abs | grad relative L2 | SGD momentum exceed | SGD Query param exceed |
|---|---:|---:|---:|---:|---:|
| 1 | 961 | 1.55e-05 | 2.153e-06 | 961 | 0 |
| 8 | 27,029,675 | 1.435 | 1.468 | 27,413,886 | 26,955,519 |
| 9 | 27,001,145 | 1.228 | 1.374 | 27,426,546 | 27,035,427 |
| 16 | 26,900,847 | 0.4886 | 1.485 | 27,509,083 | 27,295,772 |

最初の**fixed tolerance超過**は `step1 rank0 after_backward/query_grads/encoder_q.conv1.weight`。当該tensorは`torch.float32`, shape `[64,3,7,7]`、9,408 scalar中852 scalar超過、`max_abs=1.55e-05`、`mean_abs=2.802e-06`、`relative L2=2.134e-06`。これは上記の「最初のbitwise差（logits）」とは別概念。

S0 vs S1の各step比較は、step1からstep16まで差あり。step2以降は以前の更新差が重み、特徴、EMA、Queue、BNへ波及し得る。データは今回のsyntheticの制御runであり、長時間の学習安定性・表現品質の失敗を示さない。

## 実験条件・来歴

- Research code HEAD：`dev@1ce9badcbae3e2df08fa400944e4c385785063f9`、ユーザーが直前にgit cleanを確認。公式clone HEAD：`a419ef282fc9885015674e77f58c59447206a27f`、clean。
- Fixture SHA256（`judgement.md` M/S0 provenance）：`c7d769ee5eeaa554713b5a5cadc9a34be12df6296f13732ba66b6cb275d54fd1`。
- 合成A1 fixture、2GPU/NCCL、ResNet50+MLP、K64、global batch8（4/GPU）、64×64、SGD lr0.0075/momentum0.9/weight_decay1e-4、EMA0.999、temperature0.2、seed0、FP32/TF32 OFF、Comet OFF、freshでresume/checkpointなし。
- Queue pointerは各runでstep8→0、step9→8、step16→0（K64/global8でQueue二巡に相当）。
- 全runの `verification.threshold_status=fixed`、正式閾値はrun前の2026-10-19のspec承認で固定。
- 先行CPU非回帰は201 passed / 693.95s（別のユーザー提出実行、実装検証記録に保存済み）。

## ローカル成果物のパス（GitHubにはアップロードしていない）

研究サーバroot：

```text
/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/manual_20261017_191704/a1_16step_fixed_20261019_101429_votN9Ji8/
```

- 6run: `M/`, `Mb/`, `S0/`, `S0b/`, `S1/`, `S1b/`（各`run_report.json`, `resolved_config.yaml`, `provenance.json`, `trace/`, `verification_summary.md`等）。
- 6 comparisons: `cmp_M_S0/`, `cmp_M_S1/`, `cmp_S0_S1/`, `cmp_M_rerun/`, `cmp_S0_rerun/`, `cmp_S1_rerun/`（各`parity_report.json`、`verification_summary.md`）。
- 判定：`judgement/judgement.md`・`judgement/judgement.json`（2番目の存在はjudgeの現行コードによる保存仕様。ChatGPTから実体確認はしていない）。

## 実験から主張できること／できないこと

**Confirmed within recorded scope**：同一A1 synthetic・2GPU・16step条件で、Meta公式Mと自作の`meta_einsum`実装S1はすべての**記録・比較対象**でbitwise一致した。Query勾配、post-SGD Query重み、SGD momentumのフル保存step 1,8,9,16で両rankともmax absolute error 0、相対L2 0。各方式はfresh再実行でもbitwise。

**Control evidence**：従来`self`算術のS0はstep1で少数の勾配scalar超過から始まり、継続後に差が増大する一方、Meta式に一致させたS1では今回の差は解消。今回の制御介入（算術mode）で説明可能であることを強く支持。

**Not established**：全16stepの**全tensor full保存**（fullは1,8,9,16だけ）、K65536や1025step、ImageNet実画像、10epoch学習、Line​ar Probe/表現品質、Frozen ViT/LoRA/Sequential Loader、一般的なGPUや異環境での完全一致。Meta式を通常MoCoの`default`へ切り替える許可・変更も行われていない。

**Evidence boundary**：この記録はユーザー貼り付けのテキストから`judgement.md`、6comparisonのCLI要約、6runのverify内容を確認して生成。GitHubから研究サーバのuncommitted変更、raw logs、GPU driver、fixture実体等は直接参照できない。偽陽性対策のため正式判定の元JSON/trace/provenanceはユーザーの研究サーバで保管を継続。

## 次のGate（未承認の提案）

本16step A1検証は完了した。通常`DistributedMoCo` defaultの`self`から`meta_einsum`への変更は別spec・承認・非回帰検証を経て検討する。K65536/Queue長期、ImageNet実画像、10epoch/linear-probeや動画パイプラインは別のEvidence level・別Gateで計画する。今回の6runを理由に自動実行・設定変更しない。

## 関連

- [Approved 16step spec](../specs/2026-10-18-meta-moco-a1-infonce-16step-parity-spec.md)
- [CPU implementation verification](2026-10-18-meta-moco-a1-infonce-16step-parity-implementation-verification.md)
- [1step diagnostic evidence](2026-10-18-meta-moco-a1-infonce-arithmetic-gpu-one-step-evidence.md)
