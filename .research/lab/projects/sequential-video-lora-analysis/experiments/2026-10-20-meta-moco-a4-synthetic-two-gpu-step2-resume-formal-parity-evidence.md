---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: experiment-evidence
status: completed
gate: A4
judgement: PASS
evidence_level: "synthetic-two-gpu-fp32-step2-checkpoint-step3-resume-formal-parity"
source: "user-provided-terminal-output of stage_summary.json and compare/parity_report.json, plus earlier excerpt of comparison entries; SSH artifacts not independently accessed"
experiment_spec: ../specs/2026-10-20-meta-moco-a3-a4-a5-integrated-execution-spec.md
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: 914e736c2699e2e129211a888d1f19013dd1dd1a
meta_reference_commit: a419ef282fc9885015674e77f58c59447206a27f
run_label: a345_20261020_114641
stage: A4
comparison_pair: formal_resume
rtol: 1.0e-5
atol: 1.0e-6
threshold_status: fixed
comparison_entries: 3608
bitwise_equal_entries: 3608
gpu_gates: "A5-pilot PASS; A4 PASS; A3 NOT_RUN; A5-pair NOT_RUN"
---

# A4 Formal PASS：MoCo 2GPU 3-step checkpoint/resume 数値再現性

> **Formal A4 = PASS（ユーザー提供の研究サーバ上のstage summaryと比較器reportの表示内容）**。synthetic固定fixtureを用いて、自作MoCo S1の中断なしstep3と、step2 checkpointから再開したstep3を比較した結果、**judged 3,608エントリ / 3,608エントリがbitwise一致、exact_failures 0、close_failures 0、first_difference/failureなし、formal contract_valid true**。これは固定設定・2GPU・この3step再生の比較範囲に限る。ChatGPTはSSH上のtensorやcheckpoint本体を独立取得していない。

## 1. Authority / 実験の条件

- 採用済み[A3/A4/A5統合spec](../specs/2026-10-20-meta-moco-a3-a4-a5-integrated-execution-spec.md)、gate A4。
- 研究コード：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、A5-pilot/直前A4 planでcleanと確認したcommit `914e736c2699e2e129211a888d1f19013dd1dd1a`。
- 参照clone（このA4では比較器の直接比較対象ではない）：`facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f`、A4 plan時点でclean。
- A5-pilot実GPU 20step PASSは前段Gateであり、[別のEvidence](2026-10-20-meta-moco-a5-pilot-real-imagenet-two-gpu-20step-evidence.md)に記録済み。
- Synthetic fixture、224px、2GPU、ResNet50+MLP128、queue K65536、local batch 32/各GPU、global 64、FP32/AMP OFF、normal`moco.info_nce_impl=meta_einsum`、`diagnostic.info_nce_impl=null`、Comet OFF。条件はユーザーが先に提示したA4 planと採用済みspecに基づく。実行runの`resolved_config.yaml`原本は未取得。
- Run A：`S1_uninterrupted`をstep1〜3まで実行、step2でnative checkpointとrank sidecarを保存、full traceはstep3。
- Run B：`S1_resumed`は同一checkpointから復帰してstep3まで実行、同一fixture、full trace step3。
- Compare：`--pair-kind formal_resume --gate A4 --threshold-status fixed --rtol 1e-5 --atol 1e-6`、離散状態exact。A4は**公式Mと自作Sの比較ではなく、自作S1 uninterrupted対同一自作S1 resumedの比較**。

## 2. SSH上のcampaign/artifact path（ユーザー提示）

Campaign root:

`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/gates/a345_20261020_114641`

- A4 stage：`A4/stage_summary.json`、`A4/stage_summary.md`。
- Fixture：`A4/fixture/fixture.pt`、metadata/checksumを記録するfixture outputs。
- 中断なしrun：`A4/runs/S1_uninterrupted/`、step2 checkpoint `checkpoint_step000002.pt`とrank0/rank1 sidecars（ランチャーはmanifest/hashを照合する設計だが、現メッセージにhash値そのものはない）。
- Resume run：`A4/runs/S1_resumed/`。
- 正式比較report：`A4/compare/parity_report.json`。

## 3. ユーザー提供terminalによるStage最終判定

```text
=== A4 STAGE ===
Status: PASS
Reason: every step completed and every check holds
Issues: []
fixture state: completed exit: 0
S1_uninterrupted state: completed exit: 0
S1_resumed state: completed exit: 0
compare state: completed exit: 0
```

ユーザーが報告したランチャーの総合Gate判定：**PASS**、stage issuesなし、全4substep正常終了（exit0）。

## 4. Formal resumeの比較内容・判定

```text
status : PASS
reason : all compared tensors within the fixed tolerance; within-run checks PASS
pair_kind : formal_resume
tolerance : {'rtol': 1e-05, 'atol': 1e-06, 'status': 'fixed',
             'rule': '|candidate - reference| <= atol + rtol * |reference|; NaN never matches',
             'run_consistency_issues': [], 'consistency_enforced': True}
counts : {'judged': 3608, 'exact_failures': 0, 'close_failures': 0, 'bitwise_equal': 3608}
steps_compared : [3]
full_tensor_steps : [3]
first_failure : None
first_difference : None
contract_valid: True
bitwise: {'judged': 3608, 'bitwise_equal': 3608, 'fraction': 1.0,
          'first_difference': 'none (all compared values bitwise equal)'}
```

| メトリクス | 観測 | 判定 |
|---|---:|---|
| formal comparator status | PASS | PASS |
| formal contract_valid | true | PASS |
| pair_kind | formal_resume | 正しい比較 |
| steps compared | [3] | 計画どおり |
| full_tensor_steps | [3] | 計画どおり |
| judged comparison entries | 3608 | 全比較対象を計上 |
| bitwise_equal | 3608/3608 (100%) | 今回の比較対象は完全一致 |
| exact_failures | 0 | PASS |
| close_failures | 0 | PASS |
| first_failure | None | PASS |
| first_difference | None | PASS |
| threshold status | fixed (1e-5/1e-6) | 事前固定 |

> 重要：`3608`は「rank/phase/tensorなどの**比較エントリ数**」であり、Tensorのscalar総数ではない。実際のtorch tensor全要素のbitwise一致を独立に数え直した記録ではない。また`formal_resume`の性質はA2のMeta公式対自作`formal_implementation`と区別する。

直前のユーザー添付report末尾（`compare/parity_report.json`の部分転載）では、step3/rank1/`after_sgd`/`query_params`の複数重みtensorが`bitwise_equal=true`、`max_abs_error=0`、`mismatch_count=0`となっていた。今回の総合reportは**全比較対象**に拡張して`3608/3608`の一致と報告している。

## 5. 残る未検証・次の段階

- 今回確認できたこと：self S1のstep2 native checkpoint＋rank別sidecarを使うResume経路について、指定synthetic fixture・2GPU・固定FP32設定のstep3で、既存compare harnessの比較対象が全件bitwise一致した。
- **未検証**：step4以降の長時間Resume、異なるcommitやcheckpoint format、実ImageNet DataLoaderのshuffle/prefetchを含むResume、外部環境での再現性、representation quality、LoRA/ViT/動画逐次学習。
- **A3 = NOT_RUN**。次は同じcampaign rootのA3 `plan`を出し、前段A4 PASS、git/Meta clone clean、GPU競合・出力未作成を確認。**A3本番実行はMeta 1025step＋Self S1 1025step（full trace step1,1024,1025）、毎step small trace**のため、実行前に短いparity traceの時間/ログ容量の実測見積りが必要。A5-pilot `bench`の0.79秒/stepはparity trace込みの時間推定に流用不可。
- **A5-pair = NOT_RUN**。A3 PASSが前段条件。OFF/Lightの各100step監視overhead測定は別成果物。
- 研究コード・Meta clone・SSH outputをChatGPTから変更・実行していない。GitHubのこのMarkdown Evidenceのみ保存する。
