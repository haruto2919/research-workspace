---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: user-provided-terminal-evidence
status: meta-reference-rerun-bitwise-reproducible-formal-not-run
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

---

## 2026-10-20 M0r・M0 vs M0r 再現性診断（追加Evidence）

この追記はユーザーが研究サーバで実行したログの転記・分析であり、ChatGPTがSSH上のraw artifactを直接閲覧して独立検証した結果ではない。

### 事前チェックとM0rの成功

前回の確認コードで誤ってエラー扱いになった `verification.max_errors`（辞書にstatus fieldがないため）をスキップしてM0のstatus付きcheckのみを再評価した。16項目が`PASS`、BN rank差に関する2項目が`INFO`、`META M0 VERIFICATION PASS`。

- Meta M0r run：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/runs/a2_M0r_20261020_090749`
- Git preflightは前回固定commit・clean検査と整合。GPU freeは0/1とも20082 MiB（ユーザー出力）。
- M0r `Saved run artifacts`、`META M0r VERIFICATION PASS`。
- M0r step 1 rank 0：loss `6.47951602935791`、positive_cosine_mean `0.9418870925903321`、negative_cosine_mean `-0.0001348661258816719`、queue_ptr `64`、lr `0.0075`、query_grad_norm `19.536392104956256`、query_update_norm `0.1465230765670673`。M0の報告と一致。
- M0r `step_time_sec_with_tracing=2.034807861004083`、M0の値は10.245887390999997で異なる。wall timeは初回ウォームアップ・I/Oなど条件依存があるため**計算値の非再現性とみなさず**、A5性能結論にも使わない。

### CPUオフライン comparatorの実測

`--pair-kind rerun --gate A2 --threshold-status provisional`にてM0をreference、M0rをcandidateとした。

- 比較出力：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/compare/a2_M0_M0r_20261020_090802/parity_report.json`
- `Comparison status: BLOCKED`：`threshold is provisional (spec §5.3): diagnostic only; 0 float tensors outside the provisional tolerance`。
- `counts={'judged':3926,'exact_failures':0,'close_failures':0,'bitwise_equal':3926}`。
- `first_difference: None`、`first_failure: None`。

**Evidence判定：Meta公式同一fixture・同一環境での2独立fresh runの比較対象3,926件は全件bitwise一致。再現性診断の数値契約はクリア。比較器の`BLOCKED`は暫定閾値ゆえの正式判定保留であって再現性失敗ではない。**

**3,926はtrace上の比較entry数であり、画像枚数や浮動小数scalar要素数を意味しない。** 壁時計測値が異なることはこの記録対象の再現性違反とはみなさない。

### A2正式判定の閾値案（まだ未承認）

- `rtol=1e-5, atol=1e-6`を事前固定候補として推奨（A1の過去採用値と同じ）。離散量/入力hash/permutation/pointerはexact。floatは`abs(Self-Meta) <= 1e-6 + 1e-5 * abs(Meta)`。bitwise一致率は別指標として記録。
- M0/M0rで誤差が観測されなかった事実は**零許容差をA2の科学的必要条件にする根拠にはならない**。A2で計算方式の異なる独立実装同士の浮動小数比較を行うため、現行compareの共通thresholdとA1の確定値を維持するのが比較可能性上の推奨。再現性floorは0だが選んだ非ゼロ閾値は「実装差を許容する設計基準」であり自動最適化された統計区間ではない。
- **この推奨はユーザー承認前**。正式M1/S1はまだ`NOT_RUN`、A2 specは`draft`のまま。閾値のユーザー採用と正式runへの別実行指示が必要。先行するM0/M0rをformal判定へ転用しない。

### Next Gate

1. この閾値とexact条件をユーザーが決定し、[A2実験spec](../specs/2026-10-20-meta-moco-a2-imagenet-one-step-parity-experiment-spec.md)に`fixed`として事前記録する。
2. 事前承認後、既存fixtureを用い**新規fresh M1 と新規fresh 自作S1**を各2GPU/NCCL 1stepで独立実行。両方のresolved configに同じfixed閾値・flagsを記録する。
3. `compare_moco_parity.py --pair-kind formal_implementation`でM vs S1を判定し、runと比較の成果物からPASS/FAIL/INVALID/BLOCKEDを区別してResearch Workspaceへ記録する。OOM、rank不一致、NaN/Infの場合は条件自動変更なしで停止。
