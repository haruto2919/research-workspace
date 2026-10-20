---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: user-provided-terminal-evidence
status: meta-reference-m0-completed-m0r-not-run
evidence_level: gpu-diagnostic-one-step
gate: A2
run_kind: parity
reference: meta_reference
reference_repository: facebookresearch/moco
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: dc502e416aa26ed4e0791767fa5a72f7843e399d
a2_formal_comparison: NOT_RUN
---

# A2 Meta公式 M0：ImageNet 2GPU 1step診断runのユーザー報告

## Result / Evidence Scope

ユーザー提供の研究サーバ標準出力で、固定Meta公式（外部cloneを使う研究側parity runner）の**M0 2GPU/NCCL、ImageNet実画像、224px、local32/global64、K65536、FP32、step1 full traceの1step診断run完了**を確認した。

- 現地出力run directory：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/runs/a2_M0_20261020_090238`
- 元fixture：`.../log/moco_imagenet_parity/fixtures/a2_imagenet_20261020_084625/fixture.pt`、SHA256 `9edede9195eccef13e7ec8446880d6bfc6a550b11e96b466dbf047723d24b4ac`（[fixture整合性PASS](2026-10-20-meta-moco-a2-imagenet-fixture-generation-evidence.md)）。
- 事前check：自作 `dev@dc502e416aa26ed4e0791767fa5a72f7843e399d` / Meta clone `a419ef282fc9885015674e77f58c59447206a27f` を`test`で検査、`git status --porcelain`空のassertは通過。
- 起動時GPU free：0,1とも20,082 MiB。GPUはNVIDIA RTX A4500。
- `run_report.json`の表示：`status=completed`、`completed_steps=1`。
- `info_nce`：`selector_source=meta_reference`、`effective=meta_reference`、`per_rank=['meta_reference','meta_reference']`、`consistent=true`。
- rank0 step1 metrics（ユーザー標準出力）：

| Metric | Observed |
|---|---:|
| Loss | 6.47951602935791 |
| positive_cosine_mean | 0.9418870925903321 |
| negative_cosine_mean | -0.0001348661258816719 |
| queue_ptr | 64 |
| LR | 0.0075 |
| query_grad_norm | 19.536392104956256 |
| query_update_norm | 0.1465230765670673 |
| step_time_sec_with_tracing | 10.245887390999997 |

GPU各rank `resources`：

| rank | max_memory_allocated MiB | max_memory_reserved MiB | traced step wall seconds |
|---:|---:|---:|---:|
| 0 | 4016.37939453125 | 4340.0 | 10.245887390999997 |
| 1 | 4016.37939453125 | 4340.0 | 10.247478796001815 |

**この条件のMeta公式1stepのメモリ可用性・NCCL分散実行は支持される**。ただし本値はper-step全相trace込みであり、定常の通常学習速度・長時間安定性・自作S1側のVRAM適合を立証しない。

## Console assertionの偽陽性

最後のユーザー側確認スクリプトで：

```text
AssertionError: Within-run checks: {'max_errors': None}
```

が発生した。ただし実験runnerからは先に`Saved run artifacts`、`status=completed`、`completed_steps=1`が出ている。原因は確認スクリプトの

```python
if isinstance(v, dict) and v.get("status") not in ("PASS", "INFO")
```

が、必須合否チェックでなく診断値辞書`verification.max_errors`（`status`キーなし）まで不合格として集計したこと。上記出力の`bad`は`{'max_errors': None}`だけなので、ユーザーが入力した確認コードの挙動に照らす限り、実際の他の`verification` dict statusはPASS/INFOだったと判断できる。**MoCo GPU runの失敗ではない。**

再確認は`max_errors`やstatusを持たない診断dictを除外して`status`のある本当のcheckを列挙して行う。pytest/testコード・学習コードの修正は不要。

## 次のGate

- M0 raw `run_report.json`のwithin-run statusを正しいfilterで再表示。全必須`status=PASS`と`INFO`のみを確認。
- M0成功を受け、**同じfixture、同じprovisional thresholdとbackend、2GPU、同じstep 1、fresh processとfresh run directory**でMeta公式M0rを実行する。失敗なら自動batch/Queue/precision変更はせず診断中断。
- 比較器 `compare_moco_parity.py --pair-kind rerun --threshold-status provisional` でM0/M0rを照合。`provisional`の`BLOCKED`は正式判定がまだできないことを意味し、`counts`と`first_difference`、`summary_by_phase_group`、`per_step`、`preconditions`と`within_run_checks`を観察。2回のbitwise一致・数値ばらつきを基にFP32正式閾値をM1/S1実行**前**にユーザー承認で固定する。
- A2の正式M1 vs S1 `formal_implementation`は**NOT_RUN**。1stepのloss値だけで性能やMetaとの数値一致は主張できない。

## References

- [A2実験spec（基本方針採用、formal閾値未決）](../specs/2026-10-20-meta-moco-a2-imagenet-one-step-parity-experiment-spec.md)
- [A2 fixture生成・整合性PASS](2026-10-20-meta-moco-a2-imagenet-fixture-generation-evidence.md)

本記録はユーザー出力由来であり、研究サーバの実体、run artifactsファイルの内容全体をChatGPTが直接検査したものではない。研究サーバのcloneはGitHub書込みで同期されない。
