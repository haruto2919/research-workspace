---
date: 2026-10-21
project: sequential-video-lora-analysis
record_type: experiment-evidence
status: completed
gate: E5
experiment: imagenet1k-meta-moco-v2-five-epoch-pair
campaign: moco_v2_e5_20261020_220032
experiment_started: 2026-10-20
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: 7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62
meta_reference_repository: facebookresearch/moco
meta_reference_commit: a419ef282fc9885015674e77f58c59447206a27f
spec: ../specs/2026-10-20-meta-moco-v2-imagenet-5epoch-sequential-comet-comparison-spec.md
evidence_source: "User pasted SSH server summary/pair_summary.md to ChatGPT; remote raw files/checkpoints not directly inspected from GitHub"
launcher_verdict: PASS
comparison_curve_agreement: bitwise
checkpoint_validation: "launcher reportedly checked all epoch checkpoints, rank sidecars, recorded checkpoint sha256 before PASS"
comet_status: "two independent experiment URLs and [] errors in launcher report"
scientific_representation_quality: UNVERIFIED
---

# Meta公式 MoCo v2 → 自作 MoCo：ImageNet-1K 5epoch E5 実験結果（PASS）

## 概要とEvidence boundary

2026-10-21、ユーザーがSSH研究サーバの`log/moco_imagenet_5epoch/moco_v2_e5_20261020_220032/summary/pair_summary.md`の全内容をChatGPTへ提示した。ユーザー提示の**ランチャー生成最終サマリーを一次出力として転記・分析**した実験Evidenceであり、ChatGPTがSSH上の`pair_manifest.json`、`comparison_report.json`、run report、checkpoint、Comet Web UIへ直接アクセスして検証したものではない。

この記録は承認済み[5epoch逐次比較spec](../specs/2026-10-20-meta-moco-v2-imagenet-5epoch-sequential-comet-comparison-spec.md)の結果であり、以前の10epoch E3案を実行したものではない。学習runnerはMeta公式モデルを読み込む共通比較用runnerであり、公式の200epoch `main_moco.py`全設定を再演したという主張ではない。

## 実行identifiers

- Campaign: `moco_v2_e5_20261020_220032`
- Local campaign root（サーバ上）: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_5epoch/moco_v2_e5_20261020_220032`
- Summary: `summary/pair_summary.md`
- Comparison: `comparison/comparison_report.json`
- Research code: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`, `7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62`
- Meta official model source: `facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f`
- 既存仕様に基づく固定条件：ImageNet-1K train、ResNet50 + MLP MoCo v2、DDP 2 GPU × local128 = global256、queue K65536、MoCo momentum 0.999、temperature 0.2、SGD LR0.03/momentum0.9/weight decay1e-4、5epoch cosine horizon、seed0、FP32 strict、同一canonical initial fixture、各run fresh・別Comet Experiment。**実行時の全Resolved Configは本ユーザー貼付summaryには含まれない**が、ランチャーの比較契約判定がPASS。

## 成果物と処理完了

全体ランチャー判定：**PASS**、reason `every step completed and every check holds`。次の4段がすべてexit code 0かつ `completed` と報告。

| Step | 状態 | 時間 [秒] | 検証 |
|---|---|---:|---|
| `initial_state` | completed | 5 | 同一初期Fixture生成 |
| `meta_reference` | completed | 20055 | fresh 5epoch/epoch checkpoint＋Comet検証 |
| `self` | completed | 20225 | fresh 5epoch/epoch checkpoint＋Comet検証 |
| `comparison` | completed | 3 | 比較レポート生成・Gate合格 |

- 合計elapsed: **40,288秒（11時間11分28秒）**。Meta 20,055秒（5時間34分15秒）、自作 20,225秒（5時間37分5秒）、差170秒（自作が約0.85%長い）。単一連続実験であり性能差の一般化はしない。
- 両runの完了step：**25,020 / 25,020**、完了epoch `[0,1,2,3,4]`。
- Meta VRAM `max reserved 13028/20096 MiB` (rank0/rank1とも)、Self `13038/20096 MiB` (rank0/rank1とも)。残りは各7068〜7058MiB程度。ただし別プロセスを含む余裕を保証する数値ではない。
- 各epoch checkpoint5個とrank0/rank1 sidecar各epoch、内部checkpoint hashに関する検査は、ランチャーの`check_run`が通過したことから**サマリー上の成功として観測**。個々のファイル内容/ロード・resume等を今回直接確認したものではない。
- 同一FixtureからのMeta→自作の逐次実行、後続自動起動、Comet独立送信というユーザーの運用要件がランチャー判定上満たされた。

## Comet（同一Project内で別Experiment）

| 実装 | Experiment key/URL | errors |
|---|---|---|
| Meta reference | [441624a0a35f49198ef5c97d95f41bc1](https://www.comet.com/haruto2919/moco-imagenet-v2-reproduction/441624a0a35f49198ef5c97d95f41bc1) | `[]` |
| Self | [ca87df41f91c45a0a0449e9e9d6efeed](https://www.comet.com/haruto2919/moco-imagenet-v2-reproduction/ca87df41f91c45a0a0449e9e9d6efeed) | `[]` |

双方 `workspace=haruto2919`, `project=moco-imagenet-v2-reproduction` の独立したExperiment ID。コンソールのsummaryで`errors=[]`を確認。Comet UI表示の最新状態や実際のメトリクス系列をChatGPTが直接読み取ったものではない。

## 公式↔自作 5epoch比較

```text
contract_valid: True
both_completed: True
inputs_identical: True
metrics_bitwise_equal: True
epoch_states_identical: True
curve_agreement: bitwise
common logged steps: 2503
input hashes identical: 251/251
first metric differences: none
```

`inputs_identical`は採取された251地点のinput hash一致、`metrics_bitwise_equal`はログ化指標に対する一致、`epoch_states_identical`は各epochの**canonical model state SHA**一致。**全25,020stepの生画像tensorやoptimizer/RNG sidecarの全値の同一性は未検証**。

| Epoch index | 完了 | LR一致 | 平均InfoNCE Loss (Meta = Self) | Canonical state一致 |
|---:|---|---|---:|---|
| 0 | 両方True | True | 9.71783674 | True |
| 1 | 両方True | True | 8.46133620 | True |
| 2 | 両方True | True | 7.93711850 | True |
| 3 | 両方True | True | 7.72709281 | True |
| 4 | 両方True | True | 7.60787190 | True |

初回epoch平均`9.71783674`→最終epoch平均`7.60787190`。差分`2.10996484`（初回比**約21.7%低下**）、**各epochで平均Lossが前epochより低下**。5epochの最終到達Lossは初期のepoch0平均より小さく、以前の100step損失増加だけでは学習停止を論じられないことを示す。ただし同じImageNet train上のInfoNCE Loss低下を、semantic representation quality/線形評価/kNN/動画動的情報取得と同一視しない。

## Gate判定・未確立事項

**確定（ユーザー提示summaryに基づく）**：
- 5epoch × 2実装、fresh・同一比較条件の逐次学習がランチャー上**PASS**。
- 記録指標はbitwise一致、サンプル入力hash 251/251一致、各epoch canonical state一致、ログ上Comet送信errors=[]、2つの独立Experiment作成。
- インフラ条件：2GPUでVRAMの採用済みheadroom gateを満たし、学習完了まで大規模停止/OOMなし。

**これだけでは未確立**：
- すべてのstepの全入力Tensor・中間状態の完全bitwise、native checkpoint/optimizer/RNG sidecarの全値一致、checkpointロード／resume検証。
- Meta公開200epoch MoCo v2事前学習品質、kNN/Linear Probeの精度、feature collapseがないこと、難しい負例との分離の適正性。
- 動画上での時間的情報獲得、Frozen ViT / LoRA / Sequential Loaderでの本研究タスク成立。

**次の研究判断候補（未承認・未実施）**：既存Comet時系列を使ってPositive/Negative/Hardest Negativeの推移・Lossの後半改善・gradient norm/learning rateの推移を整理し、必要なら固定サンプル上でFeature variance/類似度分布、軽量kNN等の表現診断を別途設計する。公式／自作の追加の長時間学習は不要と決めた方針を保持し、動画向け段階的統合に戻る検討を優先。提案を実装指示やGPU実行許可として扱わない。

## SSOTと実行境界

- Research Workspace上の本ドキュメントが、ユーザー提示の5epoch最終summaryに基づく**結果記録**。
- ソースの研究サーバに残る`pair_summary.md`, `pair_manifest.json`, `comparison_report.json`, checkpoint/sidecars, Comet SDK logsが実体。GitHub上でSSH cloneへpull済み、ローカル実験を再現実行済み、Checkpointをリモート保存済みとは主張しない。
- この記録作業はGitHub Research Workspaceの文章保存のみ。**研究コード変更、GPU実行、Comet変更は行わない**。
