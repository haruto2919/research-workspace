---
date: 2026-10-18
project: sequential-video-lora-analysis
source_todo: "2026-10-15 MTG MoCoの公式参照と自作を比較する課題（16step parityへの移行相談）"
topic: moco-a1-infonce-16step-parity-decisions
status: exploratory
tags: [brainstorm, moco, infonce, ddp, fp32, numerical-parity]
---

# A1 InfoNCE算術一致の16step拡張：決めるべき事項（探索記録）

## 現在地と根拠

ユーザーは2026-10-18にA1 1stepの `SUPPORTED AS SUFFICIENT AT STEP 1`（run VALID、S0/S1再演 BITWISE）を報告。Meta公式と自作 `meta_einsum` のstep1全Query勾配は両rankでglobal relative L2=0、max_abs_error=0、暫定超過0/27,966,656、SGD momentum超過0。従来自作`self`は勾配超過961要素/Rank、momentum超過961。制御比較の差分初出はlogits、他条件の差は有効性checkで否定。

証拠の正本：`.research/lab/projects/sequential-video-lora-analysis/experiments/2026-10-18-meta-moco-a1-infonce-arithmetic-gpu-one-step-evidence.md`。追加spec：`.../specs/2026-10-18-meta-moco-a1-infonce-arithmetic-diagnostic-spec.md`（GitHub frontmatter `draft`）。親Spec A `2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md`もdraftで、FP32具体閾値は未正式化。現研究コード正本は `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`。`self_supervised/moco/ddp_moco.py`に両演算方式が実装済み。`scripts/parity/moco_imagenet_parity.py::validate_info_nce_diagnostic`は`meta_einsum`をA1 1step・trace[1]等に制限。`compare_moco_parity.py`は`implementation`,`rerun`,`a1_info_nce`を扱うが`a1_info_nce`は1step限定。`judge_moco_a1_info_nce.py`も明示的にStep1のみ判定。現状ではconfigを16stepにするだけで拒否される。

## 中心の問い

公式Metaと算術を合わせた自作MoCoのstep1完全一致が、同一初期stateと2GPU-DDPの学習を16step継続しても維持されるか。その際、SGD momentum、EMA Key、BN buffers、Queueとpointer（K64/global batch8でstep8、16に一巡）およびShuffle-BNに不整合が現れないか。

## 要決定項目・推奨候補（未採用）

| ID | 決める内容 | 推奨候補 | 理由と要点 |
| --- | --- | --- | --- |
| B1 | 研究主張・Gate | A1 small-K 16stepのconfirmatory数値照合に限定。A3 K65536 1025step/画像学習とは別 | 1step原因診断の一般化範囲を限定 |
| B2 | 比較run | M（Meta）, S1（自作meta_einsum）を必須。S0（自作従来式）は同一commitでの反例対照として推奨。M/S1 fresh rerunを優先し、S0 rerunも可能なら追加 | 1step差と16step差の因果比較と再現性 |
| B3 | Fixtureと条件 | 既存同一A1 synthetic fixture、K64、B4×2、64px、seed0、ResNet50+MLP、SGD LR0.0075/momentum0.9/wd1e-4、EMA0.999、T0.2、FP32/TF32 OFF、plain BN、NCCL/2GPU、固定rank/device、16step、Comet OFF、no resume/checkpoint | 単一変更軸だけを残す |
| B4 | Queue境界とtrace | full trace `[1,8,9,16]` + 既存small trace全16step、enqueue前/後のQueue/ptrとrank順、Key EMAとBN broadcast時点を確認。K64/global8は8stepで一巡 | step8→9、16でwraparound・古いKey参照の検証が必要 |
| B5 | 計算modeと後方互換 | `diagnostic.info_nce_impl=meta_einsum`の診断許可プロファイルを「A1 1step, full[1]」または「A1 16step, full[1,8,9,16]」に限定。default `self`を維持し、Meta clone/通常Standalone/CIFARを不変更 | 本番動作を変えない |
| B6 | 比較契約と判定器 | `implementation`と`rerun`既存機構を流用。16stepのS0-vs-S1変更軸検査は新たなpair profile（または専用小check）として1step契約とは別にする。1step専用judgeの名称/意味を変更せず、16step用判定を分離 | 1step診断やpassとの混同回避 |
| B7 | 判定基準 | **正式判定前に** `rtol=1e-5, atol=1e-6`を候補としてレビュー・固定。discrete/hashはexact、float全対象は公式基準でallclose。bitwise一致は特に強い補助的goal/別指標として全項目数を報告し、FP32 parity PASSと区別。NaN/Inf・条件差・欠損はFAIL/INVALID/BLOCKEDで止める | 親Spec Aの要求はFP32許容差一致であり、bitwiseを事後導入して別目的へすり替えない。旧Meta同条件再演16stepはbitwise一致という参考測定。判定対象16step結果を見て閾値を緩和しない |
| B8 | 合否の対象範囲 | 全16stepで保存されるsmall/discrete/rank/phaseを照合。full tensorは事前指定の1/8/9/16でQuery grad/SGD・momentum/EMA/Queue/BN/logits/loss等を判定。全step全tensor一致と過大主張しない | 記録コストと精度を両立 |
| B9 | 実行手順・証拠 | 実装後commit/clean、Meta固定clone clean、同一CUDA/cuDNN/NCCL、fixture SHA/state hash、resolved configs、各run fresh独立dir、run内checks、compare及びjudgeを保存。CPUのモード許可・拒否・非回帰テストを先に実施。Mの旧run再利用は機械的provenanceが合う場合のみ | 再現性・比較契約 |
| B10 | 停止と実行境界 | CPU testは実装scope、GPU 16step run/再実行は改めて明示許可。OOM、NCCL hang、nonfinite、phase差、Queue mismatch時に自動修復や条件変更はしない。Comet OFF、10epoch/A2/A3/A5非対象 | 既存データ・研究主張を守る |

## 代案・反例・保留

- **bitwise必須を公式の唯一のPASS条件にする案**：step1で完全一致したため妥当な高い目標だが、親Spec Aが採用した「FP32許容差内一致」より厳しい。目的変更が必要なので別指標とし、bitwise全一致した場合は強い成功として報告する案を優先。
- **S0を走らせない案**：M対S1の数値同等だけなら十分でGPUコストは減るが、同一commit上の従来式による差との対照は弱くなる。因果説明を重視するならS0追加が有力。
- **全16stepで全tensorをfull記録**：網羅性は高いがデータ量が大きい。まず既存の1/8/9/16を推奨、問題発生時のみ局所診断追加。
- **A3/K65536やImageNet本番同時実施**：研究目的とGPU/VRAMコストが変わるため今回のscope外。
- **Meta式を通常moco_logitsのdefaultへ変更**：CIFAR baseline・後方互換・既存性能に影響し得る。16step実測と別途の採用承認までは行わない。

## 未決とresearch-spec引き継ぎ

blocking候補は (1) 16stepの厳密な成功範囲、(2) M/S0/S1とrerunの構成、(3) fixed FP32閾値・bitwiseの位置付け、(4) 16step比較器に必要な最小変更範囲、(5) GPU実行の許可と資源上限。どれも本メモ時点では推奨であって採用決定ではない。

**research-spec handoff候補**：A1 small-K16stepの同一Fixture・2GPU-DDP、Meta式を診断のみ許可するprofile拡張、固定したtrace/比較契約、事前登録した許容差でのRUN/REPRO診断とreport。参照commit・fixture・モデル・optimizer・no-Cometなど既存spec条件を維持。採用後にspec化を明示依頼するまではspec/README/TODO/コードは変更しない。

## 研究上の根拠

- [Step1 GPU evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-18-meta-moco-a1-infonce-arithmetic-gpu-one-step-evidence.md)
- [Step1 diagnostic spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-18-meta-moco-a1-infonce-arithmetic-diagnostic-spec.md)
- [Parent Spec A](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md)
