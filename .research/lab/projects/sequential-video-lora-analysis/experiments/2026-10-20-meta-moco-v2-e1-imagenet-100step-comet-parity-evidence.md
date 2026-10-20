---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: experiment-evidence
status: completed
experiment: moco-v2-imagenet-10epoch-e1-100step
campaign: moco_v2_e1_20261020_165430
gate: E1
official_reference: facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
research_code: tamaki-lab/2026_09_ishikawa_sequential-video-lora@d6f7acc75ee92adcf7e6e3e65831d46759b9ed14
evidence_source: user-provided-ssh-console-and-comparison-report-printout
pair_status: PASS_SAMPLED_INPUT_AND_METRICS_BITWISE
epoch_state_status: NOT_EVALUATED_PARTIAL_EPOCH
execution_authority: user-ssh-run-not-chatgpt
---

# Meta公式 MoCo v2 vs 自作：実ImageNet E1（100step + Comet）比較結果

> 2026-10-20、ユーザーがSSH研究サーバのコンソール・比較器出力をアップロードしたログから採録。ChatGPTはSSHサーバ上の`run_report.json`、`metrics.jsonl`、`comparison_report.json`やGPU jobの現物を直接読んでいない。従って**表示された結果の範囲**に限定して主張する。E1は10epoch全体の学習や特徴品質検証ではない。

## Provenance / run identity

- 採用spec：[2026-10-20 Meta MoCo v2 ImageNet 10epoch Comet reproduction](../specs/2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md)（E1個別実行承認済み、E2/E3は別承認）
- code：`tamaki-lab/2026_09_ishikawa_sequential-video-lora`, `dev@d6f7acc75ee92adcf7e6e3e65831d46759b9ed14`（自作Cometの出力`self_commit`=`d6f7acc...`, `self_dirty=False`）
- reference：`facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f`。
- Campaign：`moco_v2_e1_20261020_165430`。
- Local artifact root（研究サーバのみ）：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_10epoch/moco_v2_e1_20261020_165430`
- 自作run：`2026-10-20_17-10-59-608242_self_E1`。
- 比較出力：`comparisons/e1_20261020_171802`。
- Comet：Meta公式 `https://www.comet.com/haruto2919/moco-imagenet-v2-reproduction/1f99cdba1f74470391f804121a8d5db2` ／自作 `https://www.comet.com/haruto2919/moco-imagenet-v2-reproduction/54fefa983b55467398fc79e84c3404a1`。
- 共通設定（仕様・自作Comet metadata）：ImageNet-1K train, ResNet50+MLP, world_size=2, local_batch128/global256, FP32/AMP off, SGD LR 0.03 momentum0.9 wd1e-4, K65536, MoCo m0.999, T0.2, num_workers16 per rank, seed0, deterministic cuDNN, 10epoch Cosine LR *schedule horizon*。E1では100stepで停止し、10epochの学習を実施していない。
- Cometの自作側 `initial_state_sha256=02accfd5efd1144d2a81e64f24478302c6d443f25080ef830cb2e1a918875807`。Meta側と同じ初期fixtureの契約成立は比較器`contract_valid=True`で裏付けられるが、Meta側の独立実ハッシュ値は貼付ログでは省略。

## Comparator verdict（ユーザー提示のactual output）

| 項目 | 観測値 | Evidenceの解釈 |
|---|---|---|
| contract_valid | true | 比較対象のrun契約に問題なし |
| both_completed | true | Meta/selfともrun report上completed |
| inputs_identical | true | **2/2個所の**採取済み入力ハッシュ一致 |
| metrics_bitwise_equal | true | 比較対象の記録scalar一致 |
| epoch_states_identical | null | E1はepoch途中終了、モデルepoch状態の比較なし |
| curve_agreement | bitwise | **ログ化された曲線に関する判定のみ** |
| contract_issues | [] | 指摘なし |

記録step比較はすべて `max_abs_diff=0.0`：
- `loss`: 11/11
- `positive_cosine_mean`: 11/11
- `negative_cosine_mean`: 11/11
- `hardest_negative_cosine_mean`: 11/11
- `lr`: 11/11
- `queue_ptr`: 11/11
- `finite`: 11/11
- `query_grad_norm`: 2/2

入力SHAの詳細：`hashed_steps=2`, `identical_steps=2`, `first_differing_step=null`。仕様上Step1/100の2地点であり、100stepの全入力一致を証明するものではない。比較器は`epoch_states_identical=None`でもログ指標が一致すれば`curve_agreement=bitwise`と表示するため、**Query/Key/Queue/Optimizer全テンソルのbitwise証明とは区別**する。

## Self runの観測値（E1最後の出力）

| 項目 | 観測値 |
|---|---:|
| Step完了 | 100（比較器はMeta/selfともcompleted） |
| step1 loss | 6.441793441772461 |
| step100 loss | 9.806382179260254 |
| 平均loss（100step） | 9.126792376041413 |
| step100 positive cosine | 0.7111494064331055 |
| step100 negative cosine | 0.24109346866607667 |
| step100 Query gradient norm | 0.12952577874793422 |
| step100 Queue pointer | 25600 |
| 100step duration | 166.59917097601283 sec |
| global throughput | 153.66222922973563 images/sec |
| data wait rank0 / rank1 | 2.6756847890501376 / 4.351743691004231 sec |
| peak allocated (both ranks) | 12520.07861328125 MiB |
| peak reserved (both ranks) | 13038.0 MiB |

- Comet self Experimentは`COMET INFO: Experiment is live`から開始され、終了Summaryで`status=completed`と記録され、`loss[11]`、cosine[11]、`query_grad_norm[2]`ほかにmetric値とrole/campaignのメタデータが記録された。公式側も比較レポートにComet URLが含まれる。**Web UI上での曲線オーバーレイを目視したという報告はまだない**。
- Comet警告：`comet.auto_log.weights`deprecated（推奨名`comet.auto_log.histogram_weights`）、`comet_ml`のtorchより先のimport推奨。コンソール上は明示的なmetric記録と完了を確認できた。自動loggingの完全性は警告により保証されないが、今回の明示記録の失敗を示すものではない。
- Lossはstep1 `6.4418`→step100 `9.8064`と増加。初期Queueの内容が負例で入れ替わる期間の短時間観測であり、学習品質や表現獲得の成否を意味しない。K65536に対し100×256=25,600個のkeyがenqueueされ、Queue一巡前（約39.1%相当）。同じ値になることが今回の**実装再現**の目的であり、Lossが単調減少することはE1合格条件ではない。
- `epoch_end=0`はpartial epoch metricsの出力で、`completed_epochs`やepoch checkpointを意味しない。E2での1epoch完走時にcanonical state hash・checkpoint保存・resume可否を別評価する。

## Evidence level / next gate

**E1 status：PASS（100stepの共通契約・2地点入力hash・ログ化した全metricのbitwise一致・双方完了・Comet実験URL）**。E1だけで全ステップの入力/全テンソル状態parity、Queue一巡後、1epoch checkpoint、10epochの長期学習安定性や自己教師あり特徴の品質を確認したとは言えない。

**Next**：E2はImageNet **1epoch**（10epoch Cosine horizonのうち1epochを実行し終了）を、Metaと自作の独立runで実施し、データ順序/入力hash/epoch checkpointのcanonical model state一致とComet結果を検証する段階。約5004step/epochのためE1の100stepより計算時間・NAS read・checkpoint容量が大きくなる。E2のGPU実行は**現時点で未承認**であり、E1のPASSによる自動移行をしない。失敗した場合は修正・設定変更・10epoch full runに進まない。
