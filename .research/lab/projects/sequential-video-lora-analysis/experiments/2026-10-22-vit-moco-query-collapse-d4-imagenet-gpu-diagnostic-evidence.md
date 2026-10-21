---
project: sequential-video-lora-analysis
date: 2026-10-22
record_type: gpu-diagnostic-user-console-evidence
status: lora-d4-engineering-pass-full-ft-pending
evidence_level: user-pasted-local-console-plus-run-report-and-diagnostic-summary
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: dev
github_dev_head_observed: 312c121975d68e860d5f89bf38ed26aab23b0981
local_execution_commit: unverified
spec: ../specs/2026-10-22-vit-moco-query-collapse-diagnostic-spec.md
---

# ViT＋MoCo Query Collapse：D4 ImageNet 2GPU 診断記録（LoRAコンソールのみ）

## 2026-10-22 04:02 JST LoRA run（ユーザー提出）

2026-10-22 04:02:15 JST `nvidia-smi -i 0,1` はGPU0/1とも NVIDIA RTX A4500で約15MiBの使用、GPU util 0%、ドライバ615.71.09を表示。これは**起動前の瞬間的snapshot**であり、run中のVRAM Gate・GPU安定性を保証しない。

以下のユーザー実行コマンドで新しいGPU 10step診断を開始：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke \
 --config-name moco_vit_query_collapse model.train_mode=lora training.max_steps=10
```

- 出力先（ユーザーのコンソールに表示）：`log/moco_vit_query_collapse/lora/imagenet/2026-10-22_04-02-22-441192`
- 10stepすべてについて `finite=true`、`Saved run artifacts in ...` が表示され、コンソール上は**正常完走**。
- HF Hub未認証警告はログにあるが、この断片ではcheckpoint loadそのものは完了。分類headの `classifier.weight/bias UNEXPECTED` は分類モデルcheckpointからpooler/classifierなしの`ViTModel`を構築する時に想定される差異であり、単独では失敗原因ではない。
- 研究コードGitHub `dev`には診断実装commit `312c121975d68e860d5f89bf38ed26aab23b0981` が存在することを別途直接確認した。一方**研究サーバの実行時HEAD・dirty / remote追従はこのコンソールでは照合できない**。

## 通常学習側のログ（更新前forwardの学習batch指標）

| global_step | loss | query_feature_std | trainable_grad_norm | positive_cosine_mean | hardest_negative_cosine_mean | queue_ptr |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.361582279 | 0.0722405836 | 2.185853930 | 0.7652500153 | 0.3658290386 | 256 |
| 2 | 8.604420662 | 0.00634409394 | 0.213800279 | 0.5247993469 | 0.6709590435 | 512 |
| 3 | 8.386228561 | 0.00521569978 | 0.108960548 | 0.5793584824 | 0.6973463058 | 768 |
| 4 | 8.331671715 | 0.00522294268 | 0.0700124934 | 0.6031340599 | 0.7131292343 | 1024 |
| 5 | 8.211178780 | 0.00604359806 | 0.0539189536 | 0.6403270721 | 0.7629656792 | 1280 |
| 6 | 8.045089722 | 0.00810311362 | 0.0615466870 | 0.6868749619 | 0.7704276085 | 1536 |
| 7 | 8.015432358 | 0.00881321635 | 0.0537477480 | 0.7079388618 | 0.7749258518 | 1792 |
| 8 | 7.952778816 | 0.00636844523 | 0.108406811 | 0.7357276917 | 0.8238543510 | 2048 |
| 9 | 7.887098312 | 0.00914150756 | 0.0620990559 | 0.7662922859 | 0.8251033783 | 2304 |
| 10 | 7.843848228 | 0.01095892396 | 0.0661924993 | 0.7907202721 | 0.8398139954 | 2560 |

- 全10step LRは `0.03`、epoch0、すべて finite。
- 以前の非診断LoRA 10stepの提供ログと表示されたloss/std/grad/queue_ptrの値が各stepで一致。**診断による学習攪乱が無いという方向のEvidence**だが、run同士のall-rank input hash、checkpoint identity、最終model stateの完全比較まで確認したとは主張しない。
- Step2の学習batch `query_feature_std` はstep1から大幅に低下（0.07224058→0.00634409）。ただし異なるtrain minibatchの**更新前forward**の比較であり、collapseの発生箇所／Optimizerの影響は未特定。
- 10stepでは初期random Queue K65536のうち2560（3.90625%）が実Keyに置換。Lossがstep2以降下がることを表現学習成立とは解釈しない。

## D4 Gate：現時点の未確認と次作業

本コンソールには**診断の核心となる固定probe**の内容が表示されていない。したがってLoRAについてさえ**D4正式PASS保留**。次のローカル成果物が必要：

1. 上記runの`run_report.json`の`status`、`verdicts.smoke_completed`、`vram_gate.status/per_rank`、`final_checks`、`failures`。
2. `diagnostic_summary.json`の`engineering_status`、`reasons`、`ranks_feature_identical`、`probe_memory`、`trajectory_rank0`（各stage `cls/z_q/q`×step `0/1/2/10`）、`uniformity_warnings`。
3. `update_metrics.jsonl`のstep1、LoRA `lora_A`/`lora_B`/`projector` ごとの `grad_norm`、`param_norm_before/after`、`update_norm`、`relative_update`。入力hash、probe state unchanged。
4. 工学GateがPASSした時点でFull FT fresh10stepを同一recipeで実施するかユーザー判断。FTの過去10step VRAM headroomは1749.875 MiBで最低1024MiBとの差が少ない。GPU診断で不足・OOM等があればBLOCKEDで停止し、recipeを無断変更しない。
5. 両modeの結果があれば比較CLIで`comparable=true`と**各mode`engineering_status=PASS`**をそれぞれ確認。前者は同一入力・step0初期特徴の比較可能性であり、Gate PASSを代替しない。次にD5解釈へ進む。

本記録は**GPUをChatGPTが実行していない**ユーザーコンソールEvidenceであり、原本は研究サーバローカル。Loss/Feature collapseの因果特定、学習率の適否、Full FT/LoRAの表現品質優劣は未検証。

## 関連

- [採用済み診断spec](../specs/2026-10-22-vit-moco-query-collapse-diagnostic-spec.md)
- [診断実装CPU Evidence](2026-10-22-vit-moco-query-collapse-diagnostic-implementation-cpu-verification.md)
- [非診断G3 10step](2026-10-21-vit-ddp-moco-g3-imagenet-10step-console-evidence.md)


## 2026-10-22追記：LoRAのrun_report＋固定probeでD4工学PASS

ユーザーが研究サーバ上で前述のLoRA runの`run_report.json`、`diagnostic_summary.json`を読み込んだPython出力を提示。これは**ユーザーが提示した集計値**であり、GitHubからローカルJSON原本を直接検証した結果ではない。前節「LoRA D4正式PASS保留」は旧時点の状態として維持し、本追記で更新する。

### 工学判定

| key | ユーザー提示値 |
| --- | --- |
| `run_report.status` | `completed` |
| `verdicts.smoke_completed` / `vram_feasible` | `true` / `true` |
| `vram_gate.status` | `PASS` |
| GPU0/1 `max_reserved_mib` | 12,338 MiB / 12,338 MiB |
| GPU0/1 `headroom_mib` | 7,757.875 MiB / 7,757.875 MiB（最低1,024 MiBを満たす） |
| `final_checks.ranks_hold_rank0_state` | `true`; hash `7bd1a7584c3f37ff9c2d64ecbaff3252b7c2945c95e541289ba740e3a6c2841a` 両rank一致 |
| `frozen_base_unchanged` / `frozen_key_equals_query` | `true` / `true`。Frozen Query/Keyの開始・終了hashとも `4e564e2a800d722d52ea60ca91fe6a1999ce336abeea932e42496020a75e574f` |
| `failures` | `[]` |
| `diagnostic_summary.engineering_status` / `reasons` | `PASS` / `[]` |
| `ranks_feature_identical` | `true` |
| `uniformity_warnings` | `[]`（後述：保守的条件の非該当であり無異常の証明でない） |

`probe_memory`では、両rankとも更新0で予約1282MiB、1で12038MiB、2・10で12338MiB（累積peak）。追加probeを含めてもLoRAのVRAM Gateを満たしている。ただし、これはFull FTのVRAMを保証しない。

### 同一固定64画像・rank0：CLS、Projector raw、Queryの比較

統計の各stepは**当該回数のoptimizer更新後**（0は更新前）のfixed probe。同じ64枚、同じ変換、同じモデル内Query側で算出。

| stage | step | mean_dim_std | offdiag_cosine_p50 | offdiag_cosine_p95 | row_norm_median |
| --- | ---: | ---: | ---: | ---: | ---: |
| cls | 0 | 0.8460224305 | 0.0319388916 | 0.1217781055 | 24.40650433 |
| cls | 1 | 0.8442022905 | 0.0319166785 | 0.1231291608 | 24.35245159 |
| cls | 2 | 0.8445384282 | 0.0318935304 | 0.1234027378 | 24.50394986 |
| cls | 10 | 0.5341880390 | 0.2686978235 | 0.9722146388 | 15.53567600 |
| z_q | 0 | 0.1666485849 | 0.3187901585 | 0.4487528806 | 2.266190648 |
| z_q | 1 | 3.0172653364 | 0.9931685850 | 0.9964761794 | 153.1438831 |
| z_q | 2 | 7.0842982426 | 0.9952332546 | 0.9978128403 | 375.8004762 |
| z_q | 10 | 40.37776154 | 0.9940623221 | 0.9994618682 | 655.4058866 |
| q | 0 | 0.0726317575 | 0.3187901598 | 0.4487528827 | 1.000000001 |
| q | 1 | 0.0071525748 | 0.9931685849 | 0.9964761795 | 1.000000003 |
| q | 2 | 0.0058179530 | 0.9952332545 | 0.9978128402 | 0.999999996 |
| q | 10 | 0.0076429267 | 0.9940623221 | 0.9994618681 | 1.000000005 |

**重要な観測**：
- CLSの`mean_dim_std`とcos p50はstep0→1でほぼ不変だが、**Projector rawのoffdiag cosine p50が0.31879→0.99317**、Queryのcos p50も同様に急上昇。極端な**出力方向の集中**は正規化`q`の前の`z_q`で生じる。したがってL2正規化そのものはこの方向の集中を作っていない。
- `z_q`のraw channel stdは **0.1666→3.0173→7.0843→40.3778とむしろ上昇**。raw出力の大きさも中央値2.266→153.144→375.800→655.406と著増。`raw feature variance collapse`ではなく、**raw出力norm拡大と特徴方向の均一化**として正確に記述する。QueryのL2-normalized stdは0.07263→0.00715へ急減。
- CLSはstep10でstd 0.534、cos p95 0.972と変化する。projectorだけが**唯一の原因**とは断定できず、CLSの個々のembedding drift、LoRA更新、Q/K/Queue相互作用、最初のOptimizer更新に伴う両者の同時変化が残る。
- `uniformity_warnings=[]`はwarning ruleが`q.mean_dim_std < 1e-4` **AND** `offdiag_cosine_p50>0.999`を同時に必要とするため。step1 q std0.00715, cos p50 0.99317は両方未達。これは**警告未発火**であり高い同方向性が無いという意味ではない。採用spec B5に従いheuristicを研究上のPASS/FAILにしない。

### Step1のtrainableパラメータ更新量（LoRA rank0）

| group | grad_norm | update_norm | relative_update |
| --- | ---: | ---: | ---: |
| `lora_A` | 0.0 | 2.3868588582e-05 | 2.9803465415e-06 |
| `lora_B` | 0.0309570922 | 11.501004806 | 1.1501004806e13（初期parameter norm=0のためε分母） |
| `projector` | 2.1856347045 | **24.900610885** | **1.439908435** |

解釈：
- LoRA Aの初回grad 0はLoRA Bの初期値0と整合。Aの非ゼロupdateはAdamW decoupled weight decayによる微小変化（`lr×wd≈3e-6`）。学習信号の勾配とは混同しない。
- LoRA Bの`relative_update≈1.15e13`は**ゼロ初期化した分母にepsilonを使った数式上の巨大値**であり、更新倍率が本当に1e13倍という意味ではない。絶対 `update_norm=11.501` と初期norm0を分けて報告する。
- Projector `relative_update≈1.44` はstep1 updateのL2ノルムが**更新前Projectorパラメータnormの約1.44倍**という非常に大きい変更を示す。これはraw `z_q` normの約67.6倍増や方向集中と整合する。学習率AdamW 0.03が過大という仮説を強めるが、**原因の因果確定ではない**（同じstepでLoRA Bも更新、対応するablation/低LR対照なし）。

### 現時点の判定・次アクション

1. **LoRA単独のD4 GPU工学Gate：PASS**。計測・学習・DDP rank整合とVRAMについてユーザー提示集計から確認。
2. **D4全体：未完了**。Full FT fresh10step runと`diagnostic_summary.engineering_status`、VRAM/headroom、各stage trajectories、`compare_runs(...).comparable`は未受領。ユーザーが実行する際は過去Full FT headroom 1,749.875 MiBと閾値1,024 MiB差の小ささに注意。自動batch/AMP変更不可。
3. **D5研究的解釈は暫定**：LoRAはstep1でCLS aggregate statisticsを概ね保ちつつ、Projectorのraw特徴方向が急激に集中することを観測。Projector原因の有力仮説である一方、LoRA B更新とその他要因を隔離した実験ではない。Full FT結果と合わせて優先仮説を絞る。
4. 完全なGit provenance、raw JSONやprobe manifestの実物、以前のG3 runとの全rank hash比較はGitHub上から直接確認しておらず、必要な場合はローカルから照合。
5. 低LR・Projector-only/LoRA-only ablationは**次の別research decision/spec**。現行D4の採用recipeは維持。
