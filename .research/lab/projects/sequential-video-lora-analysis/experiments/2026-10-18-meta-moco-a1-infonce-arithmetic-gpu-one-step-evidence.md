---
project: sequential-video-lora-analysis
record_type: gpu-diagnostic-evidence
status: completed
date: 2026-10-18
evidence_level: controlled-1step-2gpu
spec: ../specs/2026-10-18-meta-moco-a1-infonce-arithmetic-diagnostic-spec.md
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: 13a962b3beffa15807be2e13472aee3e6f734d09
reference_repository: facebookresearch/moco
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
---

# Meta MoCo v2 A1：InfoNCE算術1step制御診断（2GPU・ユーザー提供実測）

> **診断結果：SUPPORTED AS SUFFICIENT AT STEP 1**。本ファイルのstatus=completedは1step診断のみを指す。親Spec AのA1 PASS、正式FP32閾値の確定、16step以降のparity・表現品質を意味しない。ChatGPTはGPUを直接実行せず、ユーザー提供のjudgement.mdを記録した。

## 目的・対照条件

- M：既存Meta公式の`meta_reference_a1`を参照。固定公式commit a419ef2。
- S0：自作`diagnostic.info_nce_impl=self`。S1：自作`diagnostic.info_nce_impl=meta_einsum`。S0b/S1bはそれぞれfresh再実行。
- 自作branch dev、実装commit 13a962b3beffa15807be2e13472aee3e6f734d09（ユーザー直前のローカル確認）。
- 2GPU/NCCL、A1 synthetic fixture、K64、local batch4、64px、FP32、Comet OFF、1step、full trace step1。唯一の介入はInfoNCE算術mode。
- 暫定閾値：rtol=1e-5、atol=1e-6。判定run用に正式固定していない。

## ログ所在（研究サーバ。GitHubへraw traceは転送しない）

共通prefix：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/manual_20261017_191704/`

- Meta M：`meta_reference_a1/`。
- GPU診断root：`a1_infonce_20261018_142211/`。
- run：`S0/`, `S1/`, `S0b/`, `S1b/`。
- comparison：`cmp_M_S0/`, `cmp_M_S1/`, `cmp_S0_S1/`, `cmp_S0_rerun/`, `cmp_S1_rerun/`。
- 判定：`a1_infonce_20261018_142211/judgement/judgement.md`。

## 自動判定と有効性

ユーザー提供Markdownに記載されたjudge出力：

- category：**SUPPORTED AS SUFFICIENT AT STEP 1**。
- run validity (§7.3)：**VALID**。
- reproducibility (§7.2)：**BITWISE**（同mode fresh rerun）。
- 理由：S0ではMeta対Query勾配が暫定閾値を超え、S1では全rankのQuery勾配・SGD後Query parameter・SGD momentumに暫定閾値超過がなく、first_failureもない。
- 制御比較S0対S1のstatus：`DIAGNOSTIC`。初回bitwise差：`step1 rank0 after_logits_before_enqueue/logits/logits`。
- M対S0・M対S1・各rerunの通常比較status：`BLOCKED`。これは誤差閾値が`provisional`だからであり、judgeの有効性・再現性判定と矛盾しない。

## 主要数値（Meta Mを基準）

| 指標 | S0 rank0 / rank1 | S1 rank0 / rank1 |
|---|---:|---:|
| 全Query勾配の超過scalar数 / 要素数 | 961 / 27,966,656 | **0 / 27,966,656** |
| global relative L2 | 2.153e-06 | **0** |
| global max_abs_error | 1.55e-05 | **0** |
| SGD後Query parameter超過 | 0 | **0** |
| SGD momentum超過 | 961 | **0** |
| 最初の閾値超過tensor | encoder_q.conv1.weight | None |

| Query grad tensor | S0 vs M: relative L2 / max abs / exceed | S1 vs M |
|---|---|---|
| `encoder_q.fc.2.weight` | 2.289e-07 / 1.676e-08 / 0 | **0 / 0 / 0** |
| `encoder_q.layer1.0.conv1.weight` | 2.167e-06 / 1.788e-06 / 2 | **0 / 0 / 0** |
| `encoder_q.conv1.weight` | 2.134e-06 / 1.55e-05 / 852 | **0 / 0 / 0** |

両rankに同じ数値が出た。S1は許容差内に収まっただけではなく、**Query勾配についてmax_abs_error=0**、すなわちStep1の全Query勾配が要素単位で一致したことを意味する。Query以外の全状態・全比較tensorのbitwise一致を主張しない。

## 解釈・境界

**確認できた研究上の結論**：A1の2GPU・Step1条件で、InfoNCE算術をMeta式に揃えるだけで観測済みのQuery勾配差をゼロまで解消できた。差の原因候補がInfoNCE算術経路であることを強く支持する。S0ではmomentum差が残り、複数stepへの波及を調べる価値がある。

**未確認**：16step以降の一致、A1正式PASS、FP32正式閾値、長時間のEMA/Queue/DDP状態、ImageNet性能。ChatGPTはjudgement.jsonやraw parity_report、GPUログ本体を研究サーバから直接読んでおらず、ユーザーの貼り付けとGitHub実装の照合に基づく。

## 次の判断（未承認の提案）

通常のInfoNCE defaultを自動変更しない。次は結果をレビューし、同じ演算方式での複数step比較と、正式判定runより前の閾値固定を別spec/承認で進める。現在の`meta_einsum`診断modeはA1/1stepに制限されており、そのまま16step実行しない。既存artifact、Meta公式clone、CIFARの既定経路を保護する。

## 関連文書

- [追加spec](../specs/2026-10-18-meta-moco-a1-infonce-arithmetic-diagnostic-spec.md)
- [実装・CPU検証](2026-10-18-meta-moco-a1-infonce-arithmetic-diagnostic-implementation-verification.md)
- [勾配差の事前解析](../code-analysis/2026-10-17-meta-moco-a1-gradient-divergence-explain.md)
