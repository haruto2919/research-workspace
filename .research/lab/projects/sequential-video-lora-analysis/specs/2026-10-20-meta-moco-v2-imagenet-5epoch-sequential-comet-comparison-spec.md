---
project: sequential-video-lora-analysis
spec_type: experiment-and-engineering
status: implemented
created: 2026-10-20
last_updated: 2026-10-21
title: "Meta MoCo v2 ImageNet-1K five-epoch sequential official/self reproduction with independent Comet experiments"
authority: "2026-10-20 user instruction: do not perform the former 10-epoch comparison because of runtime; replace it with 5 epochs per implementation, auto-run Meta reference first and self second, keep Comet records separate"
supersedes_e3_of: 2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md
source_code_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_code_branch_at_review: dev
source_code_commit_at_review: d6f7acc75ee92adcf7e6e3e65831d46759b9ed14
official_reference: facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
implementation_status: IMPLEMENTED_CPU_VERIFIED_PUSHED
implementation_commit: 7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62
implementation_record: ../experiments/2026-10-20-meta-moco-v2-5epoch-pair-launcher-implementation-verification.md
five_epoch_gpu_run_status: PASS_USER_SUPPLIED_FINAL_SUMMARY
five_epoch_gpu_evidence: ../experiments/2026-10-21-meta-moco-v2-e5-imagenet-5epoch-pair-parity-evidence.md
five_epoch_campaign: moco_v2_e5_20261020_220032
five_epoch_gpu_run_authority: NOT_GRANTED_BY_THIS_SPEC
---

# MoCo v2 ImageNet-1K 5epoch 連続比較（Meta公式→自作、Comet別Experiment）

## 1. 現在のユーザー決定・変更理由

2026-10-20、ユーザーは旧10epoch Meta/Self比較の所要時間が長すぎるため**10epoch検証は行わず、5epochへ縮小**することを指定。さらに**Meta公式5epochが終了したら、自作5epochを自動で連続起動**し、**Cometの記録を分離**することを要求。

この方針は旧[10epoch spec](2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md)の**未来のE3 10epoch実行計画だけを置き換える**。旧V1 30step / E1 100step / E2 1epochの既存Evidence、比較コード、初期状態/Queue/optimizerの結果とCheckpointは変更・削除しない。

2026-10-20のE2結果：Meta/Self各5004step、sampled input hashes 51/51、ログ指標502/502、gradient norm52/52のbitwise一致、epoch-end canonical SHA `2dc5e59947677ffcb9cadc9006947e415b992f2b42f09517f0f3b19b7e44abbc`一致、checkpoint/両rank RNG sidecar実在、Comet URL/errorsなしをユーザーSSHログで確認。**E2に使用したcheckpointや初期fixtureを新5epochの学習開始に利用しない**。

## 2. 固定した5epoch学習条件

| 軸 | 5epoch比較の契約 |
|---|---|
| Dataset | ImageNet-1K training（従来と同じroot, class order） |
| Backbone / head | torchvision ResNet50 + MoCo v2 MLP 2048→2048→128 |
| DDP | GPU0,1・world_size=2・local128/global256、NCCL、plain BN + shuffle BN |
| MoCo | K65536, m0.999, T0.2、Meta builderと自作`meta_einsum` |
| Optimizer | SGD LR0.03, momentum0.9, WD1e-4, nesterov=false |
| Augment | MoCo v2 aug-plus、224px・two-crop |
| Epoch/scheduler | **training.epochs=5, training.cos=true、cosine horizon=5epoch**、epoch0..4すべて実行 |
| Resume | **null**；E2や過去実験からの継続ではなく、公式/自作それぞれ新規fresh run |
| Precision | FP32、AMP/TF32 off、cudnn deterministic true・benchmark false |
| Seed/input | seed0；同一canonical fixtureとepoch/rank seed policy（sampler/augmentation/Shuffle-BN） |
| DataLoader | 16 workers/rank、drop_last true |
| Recording | Comet live rank0、同一workspace/project内で**公式と自作に独立した2 Experiment** |
| Artifact | 各runのmetrics.jsonl / epochs.jsonl / run_report.json / resolved config / provenance / epoch checkpoints+rank0/1 sidecars |
| 目安step数 | 5004 step/epoch（実reportを正とする）、**25,020step/run×2=50,040 optimizer step相当** |

5epochは公式公開MoCo v2の200epoch事前学習や旧10epochの最初5epochとは異なる**新しい独立recipe**。Cosineのperiod/horizonを5に変更するとepoch1以降のLRが変わる。E2は10epoch horizonのepoch0のみのEvidenceであるため、E2のCheckpointからのresumeは比較同一性を壊す。これを明示的に禁止する。

## 3. 連続実行・ログ分離の契約

ユーザーが後から明示的にGPU開始を承認し、サーバで専用ランチャーを`--execute`したときだけ起動。**コード実装の許可とGPUの長時間学習実行許可は別**。

```text
CPU / preflight（repo commit/dirty、Meta clone、GPU空き、ImageNet、保存先、Comet）
  ↓ PASSのみ
新規campaign moco_v2_e5_<timestamp> と共通canonical fixture（1個）
  ↓
Meta公式builderから 5epoch（fresh） → meta_reference 専用Checkpoint・Comet Experiment A
  ↓ status=completed、25,020step、epoch0..4、全checkpoint/sidecar、Comet URL/Errors等を確認
自作DistributedMoCoから 5epoch（fresh・同fixture） → self 専用Checkpoint・Comet Experiment B
  ↓ 同様に完了・成果物・Cometを確認
offline comparator → loss/cosines/lr/grad/queue/inputs・epoch別canonical state SHA
  ↓
保存済み報告を集約（PASS / FAIL / BLOCKED、両Experiment URL、比較report、時間・VRAM）
```

- **Comet**：`workspace=haruto2919`, `project=moco-imagenet-v2-reproduction`は従来の設定どおり。2つの別Projectを増設せず、実装・roleが異なる**独立Experiment 2件**を生成する。既存runnerの`CometLive`は各fresh runで別`comet_ml.Experiment`を作るため、その機能を再利用する。`COMET_CONFIG`、API keyは従来ルール通り外部管理、送信先preflightとAPI key非表示の運用を保つ。
- **出力**：旧`log/moco_imagenet_10epoch/*`は一切上書きせず、例として`log/moco_imagenet_5epoch/moco_v2_e5_<timestamp>/`以下にinitial_state / meta_reference / self / comparison / logs / summaryを別々に保存。実際のrun dir名は各role識別可能で一意、両runは同一campaign ID。
- **順序**：MetaとSelfは同時GPU実行しない。Metaがexit=0でもreport・checkpoint・Cometの検証がFAILなら**Selfは起動しない**。Self失敗時は比較しない。自動リトライ・黙ったbatch縮小/AMP変更/worker変更/Queue変更は禁止。失敗時は成果物保全し停止。
- **比較**：現在の`compare_moco_10epoch.py`を原則再利用。入力hash各記録地点、全ログ化scalar、epoch0..4のcanonical model state SHAを比較する。**`curve_agreement=bitwise`だけで合格にせず**`contract_valid/both_completed/inputs_identical/metrics_bitwise_equal/epoch_states_identical`が全てTrue、epoch0..4全件がcompletedかつcanon SHA一致でPASSとする。途中step全入力、native Checkpoint/optimizer/RNGの全値のbitwise主張はしない。

## 4. 実装で必要な最小変更（まだ未実施）

現行`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`は`scripts/parity/moco_imagenet_10epoch.py:66-73`の`ADOPTED_RECIPE['training.epochs']=10`と`prepare_config()`のGPU Gate検査により、単純な`training.epochs=5`のoverrideを拒否する。**既存10epoch/E1/E2の検証条件を黙って5epochに置換しない**。

推奨する最小設計：

1. 現行runnerに**5epoch専用の新Gate識別子（候補 `E5`）**を追加し、このGateでのみ`training.epochs=5`を許可。既存`V1/E1/E2/E3`の10epoch固定・旧比較再現経路は維持する。MoCo model/InfoNCE/optimizer/data transformの計算意味・関数定義は変更しない。新Gateの正式名は実装時の局所詳細だが、過去の`E3=10epoch`と同名で新5epochを記録しない。
2. 既存のHydra configとfixture builder・run report・比較器を使用する**狭い用途の逐次ランチャー**を作成する（候補 `scripts/parity/run_moco_5epoch_pair.py`）。`--execute`なしの既定は計画を表示するのみ。実行時は`gate=E5 training.epochs=5 training.max_steps=null training.stop_after_epoch=null training.resume=null logging.comet_enabled=true logging.online_check=true`を両runへ渡し、Hydra output dirを独立設定する。
3. `test/model/test_moco_imagenet_10epoch.py`の旧recipeの回帰、5epoch Gate受理/不整合拒否、CPU/mockの公式→自作順序・失敗時停止・Comet別experiment記録・Epoch state比較・旧artifact非破壊を検証。既存`run_moco_gates.py`の仕様（旧A3/A4/A5）へ拡大・流用しない。
4. 実装やtestをGitHubへcommit/pushする場合は、別途明示されたGit操作の許可が必要。実際の長時間GPU job/Comet real sendは実装完了後の別の実行判断とする。

## 5. 成功条件とEvidenceの境界

- **実装Done**：5epoch設定が新Gateでのみ受理され、計画モードがGPU/Cometへアクセスせず、モックで順序/停止/両Experiment分離/比較を自動テスト可能。旧V1/E1/E2の適用条件は非回帰。実GPU runや外部Comet real送信が未実施ならその旨を報告する。
- **GPU run Done**（別承認）：公式/自作でepoch0..4を完走、同じ初期state、独立ExperimentのURL、両側Checkpoint・sidecars、5epoch comparison PASS（主張の範囲は記録指標とepoch canonical state）。不一致の場合は原因と最初の差分を保存してFAIL、成功に読み替えない。
- **研究上の限界**：今回の5epochはMoCo mechanics/learning dynamicsの公式コードとの再現性検証であり、Meta公開200epoch性能やSemantic Representation Qualityの証明ではない。kNN/Linear Probeは既存B4判断に従い任意。動画LoRA/Sequential loaderの学習・解析は今回の作業に含めない。

## 6. Authority・変更境界

- **ユーザー採用**：10epochは実行しない、5epoch×2、順番はMeta→Self、Cometを分離、時間節約を優先。
- **既存specとE2から継承**：画像/モデル/SGD/Queue/augmentation/FP32/deterministic/入力fixture/epoch checkpoint/safety Gate/Comet projectを維持。
- **新しい設計選択**：5epoch Cosine horizon、E2からresumeしないfresh、同一Project内の別Experiment、fail-closed sequential launcher。いずれも従来の比較可能性を維持する解釈。2つの別Projectを欲する場合、追加の明示指示で再審査する。
- **未実施**：この文書作成は実装、GPU実行、Comet送信、研究コードのcommit/pushを伴わない。長時間runの許可はない。

## 7. 実装記録（2026-10-20）

ユーザー依頼「このスペックに従って実装してください」により、§4の1〜3を`dev@d6f7acc`の上に実装した（**未commit**）。

- runner：`gate=E5`を追加し、GPU runでは`E5`だけが`training.epochs=5`を受理する（V1/E1/E2/E3は10epochのまま）。モデル・InfoNCE・optimizer・transform・比較器は無変更。
- ランチャー：`scripts/parity/run_moco_5epoch_pair.py`。既定は計画表示のみ、`--execute`でpreflight → fixture → Meta公式5epoch → 検証 → 自作5epoch → 検証 → 比較 → 集約。保存先は`log/moco_imagenet_5epoch/moco_v2_e5_<timestamp>/`。
- CPU検証：関連3テストファイルで**108 passed**（新規：E5 Gate 8件、ランチャー25件）。計画モードを研究サーバで実行し、何も書き込まれないことを確認。
- **未実施**：研究コードのcommit/push、`--execute`、5epoch GPU学習、Comet実送信。§3・§5の「GPU run Done」は別承認のまま。
- **追記（同日）**：ユーザー許可により`dev@7fc1f56`（`dd0d8b3` feat＋`7fc1f56` test）へcommitし、`origin/dev`へpush。clean commitで計画モードのissuesなしを確認。`--execute`・5epoch GPU学習・Comet実送信は引き続き未実施・未承認。

詳細は[実装・CPU検証記録](../experiments/2026-10-20-meta-moco-v2-5epoch-pair-launcher-implementation-verification.md)。

## 8. E5 5epoch×2 GPU実験完了（2026-10-21）

**過去の§4「まだ未実施」、§6「未実施」、§7「GPU未実施」は当時の状態の履歴**であり、現状は以下のとおり。ユーザー提示の研究サーバ`summary/pair_summary.md`より、**E5 5epoch sequential pair = PASS**。この結果をもって、5epoch専用Gate/ランチャー実装と受入実験について`status: implemented`とする。

- Campaign `moco_v2_e5_20261020_220032`、`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` commit `7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62`、Meta参照clone commit `a419ef282fc9885015674e77f58c59447206a27f`。
- `initial_state -> meta_reference -> self -> comparison` が全て`completed` / exit 0。両run 5epochs（epoch0..4）, 25,020/25,020step、ランチャーによるcheckpoint 5個／各rank sidecar／config・Comet等の保存検査PASS。全工程40,288秒（11時間11分28秒）。
- 入力SHA hash251/251地点一致、共通ログstep 2503地点でmetric bitwise、epoch0〜4のcanonical stateはすべて一致。比較Verdict `contract_valid, both_completed, inputs_identical, metrics_bitwise_equal, epoch_states_identical=True`、`curve_agreement=bitwise`、最初のmetric差分なし。
- Epoch平均Loss（両実装同値）：`9.71783674 -> 8.46133620 -> 7.93711850 -> 7.72709281 -> 7.60787190`（初回epoch平均→最終epoch平均で約21.7%低下）。
- Cometは同一Project内の独立Experiment。Meta公式: https://www.comet.com/haruto2919/moco-imagenet-v2-reproduction/441624a0a35f49198ef5c97d95f41bc1、自作: https://www.comet.com/haruto2919/moco-imagenet-v2-reproduction/ca87df41f91c45a0a0449e9e9d6efeed。両者errors `[]`。
- 元の実験結果と未確立事項：[E5実験Evidence](../experiments/2026-10-21-meta-moco-v2-e5-imagenet-5epoch-pair-parity-evidence.md)。

**科学的な区別**：実装の数値再現性は高いEvidenceで確認できたが、Semantic Representation Quality・ImageNetのdownstream accuracy・特徴崩壊の詳細・動画におけるLoRAの動的情報獲得は未検証。仕様の実装/受入実験が完了したことを、研究全体の完了と混同しない。

**確認の出所**：ChatGPTはサーバ上のrun artifactやComet内部履歴には直接アクセスせず、ユーザーが貼付したランチャーの最終summaryを採用。5epoch以外の10epoch長時間実行は引き続き対象外で、ユーザー依頼により新しい研究コード/Comet/job変更は行っていない。
