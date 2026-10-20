---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: experiment-evidence
status: partial
experiment: moco-v2-imagenet-10epoch-e2
campaign: moco_v2_e2_20261020_173541
gate: E2
official_reference_commit: a419ef282fc9885015674e77f58c59447206a27f
research_code_commit_expected: d6f7acc75ee92adcf7e6e3e65831d46759b9ed14
source: user-pasted-ssh-compare-output
sampled_metrics: BITWISE_EQUAL
sampled_input_hashes: 51_OF_51_IDENTICAL
epoch_end_state: NOT_CONFIRMED
overall_gate: PENDING_CANONICAL_STATE_AND_CHECKPOINT_VERIFICATION
---

# E2 実ImageNet 1epoch Meta公式 vs 自作：比較ログ部分結果

2026-10-20、ユーザーが研究サーバでMeta公式対自作の`compare_moco_10epoch.py`を実行した出力を貼り付けた。**ChatGPTはSSHサーバのrun report/比較JSON/Checkpoint現物へ接続していない**。このファイルは提示ログの確認済み範囲のみ記録する。

## Runと比較条件

- Campaign: `moco_v2_e2_20261020_173541`（自作runのパス断片より）
- Self run: `2026-10-20_19-38-02-387616_self_E2`
- 目的：ImageNet-1K、2GPU×local128/global256、FP32、MoCo v2 ResNet50、10epoch cosine scheduleの最初の1epoch（5004step）について、公式adapter vs 自作の入力・ログ化指標・1epoch canonical stateを比較。
- E2実験の詳細recipeは[採用済みspec](../specs/2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md)に依拠。ユーザー貼付断片自体にfull resolved_configはない。

## ユーザー提示の比較器ログ

```text
curve_agreement: bitwise
Contract issues: []
Input hashes: {'hashed_steps': 51, 'identical_steps': 51, 'first_differing_step': None}
loss: 502/502 matched, max_abs_diff=0.0
positive_cosine_mean: 502/502 matched, max_abs_diff=0.0
negative_cosine_mean: 502/502 matched, max_abs_diff=0.0
hardest_negative_cosine_mean: 502/502 matched, max_abs_diff=0.0
lr: 502/502 matched, max_abs_diff=0.0
queue_ptr: 502/502 matched, max_abs_diff=0.0
finite: 502/502 matched, max_abs_diff=0.0
query_grad_norm: 52/52 matched, max_abs_diff=0.0

=== EPOCH STATES ===
Epoch: 0
Completed: [True, True]
Global steps: [5004, 5004]
Loss difference: 0.0
```

貼付内容では`canonical_state_equal`の行は欠けている。同様に`epoch_states_identical`、`both_completed`、`contract_valid`の値とcheckpoint/sidecar有無は**この貼付断片から直接確認できない**。Contract issues=[] は判定器に契約指摘がないことを示す。

## 直接成立する主張

- 1epochに必要な5004stepが両実装のepoch記録上は完了している（`completed=[True,True]`）。
- 51個の採取された入力ハッシュが両実装で一致（全5004step分のtensor同一は未証明）。
- 各502地点のLoss、positive/negative/hardest-negative cosine、LR、Queue pointer、finite、および52地点のquery_grad_normの各比較値はbitwise一致（表示上`max_abs_diff=0.0`）。epoch mean loss差は0.0。
- 既存比較器コード`scripts/parity/compare_moco_10epoch.py:126–131`はepoch statesが未記録の場合も`all([])==True`により`curve_agreement=bitwise`と表示可能。よって**`curve_agreement`だけからcanonical state SHAの一致を断定しない**。

## 残る正式Gate

研究サーバで比較report`comparison_report.json`を読み、`verdicts.epoch_states_identical`と`epochs[0].canonical_state_equal`が両方`True`か、`verdicts.contract_valid/both_completed/inputs_identical/metrics_bitwise_equal`がすべて`True`か確認。両runの`run_report.json`からcheckpointと2rankのsidecar実在、Comet記録と異常なしを確認すること。

状態SHAは**モデル全体のcanonical stateのみ**の照合であり、native checkpointファイルのバイト列、optimizer/RNG sidecarの全状態が一致するとは主張しない。これらを確認する前にE2全体PASS/E3承認と扱わない。

次回のユーザー提示結果で必要項目がPASSと確認されたら、本ファイルに追記してstatusを更新する。E3の10epochは別承認が必要。
