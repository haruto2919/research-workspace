---
project: sequential-video-lora-analysis
date: 2026-10-22
record_type: gpu-diagnostic-user-console-evidence
status: d4-both-modes-engineering-pass-d5-descriptive-analysis-causal-unverified
evidence_level: user-pasted-local-console-run-report-diagnostic-summary-and-compare-runs
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


## 2026-10-22追記：Full FTの診断10stepコンソール完走（工学Gate正式判定待ち）

ユーザーが2026-10-22に、以下のFull FT診断runの学習コンソールをテキストとして提出した。runは**10/10stepで`finite=true`、保存完了メッセージあり**。ただし`run_report.json`・`diagnostic_summary.json`の出力内容（Full FTのVRAM Gate、state整合、固定64画像のCLS/z_q/q、step1更新量）はこの提出には**含まれていない**。したがって **Full FTのD4正式Gateは現時点で未判定**。

- 研究サーバ側実行：`CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_query_collapse model.train_mode=full_ft training.max_steps=10`
- 保存先：`log/moco_vit_query_collapse/full_ft/imagenet/2026-10-22_04-20-32-327570`
- 元の診断なしのFull FT G3 10stepログとユーザー提出のloss/feature std/gradient値は、表示桁で全step一致。instrumentationが学習軌跡を変えないという方向のEvidenceだが、全runのinput SHAや最終model state完全一致まで確認したとは主張しない。
- HFの匿名Hub access warningと事前学習分類headの `UNEXPECTED classifier.weight/bias` メッセージは出たが、consoleではViT weightsのロード・全stepの処理を完了している。単独ではrun失敗と解釈しない。

### Full FT GPU実行の学習batch側ログ（固定probeの結果ではない）

| step | loss | query_feature_std | trainable_grad_norm | positive_cosine_mean | hardest_negative_cosine_mean | queue_ptr |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.361582279 | 0.0722405836 | 2.591366860 | 0.7652500153 | 0.3658290386 | 256 |
| 2 | 8.629482269 | 0.00081159134 | 0.3004990527 | 0.5198947906 | 0.6799616814 | 512 |
| 3 | 8.324354172 | 0.00020079943 | 0.1448920660 | 0.5928415298 | 0.6964627266 | 768 |
| 4 | 8.225353241 | 0.00003107241 | 0.0830285434 | 0.6274086952 | 0.7302149773 | 1024 |
| 5 | 8.088413239 | 0.00001397400 | 0.0655057280 | 0.6697249413 | 0.7738025188 | 1280 |
| 6 | 7.930787086 | 0.00000888288 | 0.0502022392 | 0.7176298141 | 0.7848484039 | 1536 |
| 7 | 7.894659996 | 0.00000684135 | 0.0483050908 | 0.7426160812 | 0.8284255981 | 1792 |
| 8 | 7.797043800 | 0.00000604500 | 0.0378403839 | 0.7810830593 | 0.8412216187 | 2048 |
| 9 | 7.776324749 | 0.00000548305 | 0.0324205727 | 0.8054279327 | 0.8581251144 | 2304 |
| 10 | 7.755354404 | 0.00000605682 | 0.0284487542 | 0.8299976349 | 0.8735323906 | 2560 |

- 全step `epoch=0`、`lr=0.03`、`finite=true`。step10のQueue pointer=2560はK65536の3.90625%分が実Keyで置換されたことに相当する（queue内の実Keyの種類や代表性は確認していない）。
- 同じ非診断条件でのLoRA step10 `query_feature_std=0.01095892395824194`に対して、Full FT `6.056817710486939e-06` は約 **1/1,809**。これは**異なる訓練batchに対する更新前forward**のQuery分散であり、固定probeのCLS／Projector局在化の結果ではない。表現品質の順位付けやMoCo成功判定には使わない。
- Full FTの`positive_cosine_mean`は0.830、`hardest_negative_cosine_mean`は0.874（step10）。positive平均とhardest-negative平均は異なる統計であるため、単純な相対大小はInfoNCE不成立の証明ではない。

### D4への残作業・観測の優先順

1. 保存済みFull FTの `run_report.json`（`status`,`vram_gate`,`final_checks`,`failures`,`missing_rank_reports`）と`diagnostic_summary.json`（`engineering_status`,`reasons`,`ranks_feature_identical`,`probe_memory`,`uniformity_warnings`）を読み、**Full FTの正式工学Gate**と最低headroom≥1,024MiBを判定する。以前の非診断G3 Full FTではheadroom 1749.875MiBだが、新しい診断runで同値だったと推定しない。
2. 同じ `diagnostic_summary.json` の `trajectory_rank0`の`cls`,`z_q`,`q` × 更新`0/1/2/10`を取り、LoRAの既知の固定probe結果と比較する。特にstep0→1でCLSから均一化するか、CLSが残りProjector段で出力方向が集中するかを特定。raw`z_q`の`mean_dim_std`と`row_norm_median`を必ず併記し、raw magnitude増加とnormalized Query std低下を区別する。
3. `update_rank0`（step1）の`backbone`と`projector`の `grad_norm`,`update_norm`,`relative_update`を確認。LoRA側のProjector相対update約1.44がFull FTでも同程度か調べる。AdamW LR0.03原因説は現段階で仮説。
4. 両modeの `engineering_status=PASS` を**個別**に確認したうえで、比較CLI `python -m scripts.vit_moco.query_collapse_diagnostic <lora_dir> <full_ft_dir>` の `comparable`、全10stepの学習view SHA、probe入力、step0 feature bitwise一致を確認する。CLIの`comparable`は工学Gate PASSを保証しない。
5. これらが揃ってから研究的解釈D5。今のログのみでFull FTのCLS Collapse・Projector原因・学習率原因を断定しない。

**Evidence境界**：今回の添付はユーザーが研究サーバで取得したコンソール出力であり、ChatGPTはFull FT GPU run、診断summary原本の閲覧、ローカルdirty/commitの確認を行っていない。後続のrun report集計受領までD4全体は正式PASSではない。


## 2026-10-22追記：D4両方式工学PASSとD5記述的解析

### 受領した新しいEvidenceと範囲

2026-10-22、ユーザーが研究サーバのローカル`run_report.json`／`diagnostic_summary.json`（LoRA/Full FT）を読み、`compare_runs`で比較したPythonコンソール出力を提示した。数値の正本は研究サーバローカルartifactであり、ChatGPTはJSON原本の存在・git実行時HEAD・dirty状態を直接検証できていない。前節の「Full FT工学Gate正式判定待ち」はその時点の記録であり、以下の最新判定が優先する。

実行run：
- LoRA：`log/moco_vit_query_collapse/lora/imagenet/2026-10-22_04-02-22-441192`
- Full FT：`log/moco_vit_query_collapse/full_ft/imagenet/2026-10-22_04-20-32-327570`

### D4 工学Gate確定（両モードPASS）

| Gate項目 | LoRA | Full FT |
| --- | --- | --- |
| `run_report.status` | completed | completed |
| `vram_gate.status` | PASS | PASS |
| `diagnostic_summary.engineering_status` | PASS | PASS |
| `diagnostic_summary.reasons` | [] | [] |
| `diagnostic_summary.ranks_feature_identical` | true | true |
| `final_checks.ranks_hold_rank0_state` | true | true |
| `max_reserved_mib`、rank0/1 | 12338 / 12338 | 18346 / 18346 |
| `headroom_mib`、rank0/1 | 7757.875 / 7757.875 | 1749.875 / 1749.875 |
| `uniformity_warnings` | [] | **step10 rank0/rank1のみ警告** |

2GPUいずれも、採用spec閾値の`headroom_mib >= 1024`を満たす。Full FTは余裕が **725.875MiB** しかなく、条件を増やしてのGPU runへ無断延長しない。Full FT `frozen_base_unchanged=true`は `frozen_start.parameters=0` のため**ViTの凍結を意味しない**（Full FTではBackbone学習対象）。

両run比較CLI結果：
- `train_inputs_identical=true`（全比較step）、`probe_inputs_identical=true`
- `step0_features_bitwise_equal={cls:true,z_q:true,q:true}`
- `comparable=true`
- `Both engineering gates PASS: True`、`D4 ready for D5: True`

**判断**：D4のengineering Gateは両modeでPASS。比較の同一初期特徴・同一probe画像・各stepの学習入力は支持された。一方これは**表現品質の成功Gateではない**。

### D5 記述的解析（固定64画像・rank0、更新前0 / 更新後1,2,10）

全stageで同じ64画像を測定。Cos値は異なる画像ペア（対角なし）の中央値/p95。CLSおよびProjector raw `z_q`とL2正規化Query `q`を区別する。

| mode | stage | step | mean_dim_std | offdiag_cos_p50 | offdiag_cos_p95 | row_norm_median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LoRA | cls | 0 | 0.84602243 | 0.03193889 | 0.12177811 | 24.406504 |
| LoRA | cls | 1 | 0.84420229 | 0.03191668 | 0.12312916 | 24.352452 |
| LoRA | cls | 2 | 0.84453843 | 0.03189353 | 0.12340274 | 24.503950 |
| LoRA | cls | 10 | 0.53418804 | 0.26869782 | 0.97221464 | 15.535676 |
| LoRA | z_q | 0 | 0.16664858 | 0.31879016 | 0.44875288 | 2.266191 |
| LoRA | z_q | 1 | 3.01726534 | 0.99316858 | 0.99647618 | 153.143883 |
| LoRA | z_q | 2 | 7.08429824 | 0.99523325 | 0.99781284 | 375.800476 |
| LoRA | z_q | 10 | 40.37776154 | 0.99406232 | 0.99946187 | 655.405887 |
| LoRA | q | 0 | 0.07263176 | 0.31879016 | 0.44875288 | 1.0 |
| LoRA | q | 1 | 0.00715257 | 0.99316858 | 0.99647618 | 1.0 |
| LoRA | q | 2 | 0.00581795 | 0.99523325 | 0.99781284 | 1.0 |
| LoRA | q | 10 | 0.00764293 | 0.99406232 | 0.99946187 | 1.0 |
| Full FT | cls | 0 | 0.84602243 | 0.03193889 | 0.12177811 | 24.406504 |
| Full FT | cls | 1 | 0.11770424 | 0.98495289 | 0.99659101 | 25.351545 |
| Full FT | cls | 2 | 0.04551157 | 0.99830853 | 0.99984338 | 24.479593 |
| Full FT | cls | 10 | 0.00186125 | 0.99999748 | 0.99999949 | 28.272040 |
| Full FT | z_q | 0 | 0.16664858 | 0.31879016 | 0.44875288 | 2.266191 |
| Full FT | z_q | 1 | 0.30611520 | 0.99994166 | 0.99998633 | 155.822460 |
| Full FT | z_q | 2 | 0.57858195 | 0.99999757 | 0.99999979 | 586.084100 |
| Full FT | z_q | 10 | 0.14518664 | approx 1.0 | approx 1.0 | 5394.1868 |
| Full FT | q | 0 | 0.07263176 | 0.31879016 | 0.44875288 | 1.0 |
| Full FT | q | 1 | 0.00071494 | 0.99994166 | 0.99998633 | 1.0 |
| Full FT | q | 2 | 0.00016083 | 0.99999757 | 0.99999979 | 1.0 |
| Full FT | q | 10 | **4.51817047e-6** | **0.9999999985225709**（warningレコードのより高精度値） | approx 1.0 | 0.99999999 |

注：上表の`approx 1.0`はユーザーのprintが`:.8g`で丸めた値であり、floatのbitwise完全一致を主張しない。

**結果A：Full FT CLSは最初の更新直後に強く方向が集中した。** CLS cos p50は0.03194→0.98495→0.99831→0.99999748、stdは0.8460→0.1177→0.0455→0.00186。raw CLS normはstep0 24.41→step1 25.35→step10 28.27で**零ノルム化ではない**。ViT Backboneの表現方向の多様性が失われた兆候を、fixed probeにより直接確認。

**結果B：LoRA CLSは初期2stepでは方向多様性維持、Projector rawでは方向が集中した。** LoRA CLS cos p50 0.03194→0.03192→0.03189、std 0.8460→0.8442→0.8445。一方`z_q` cos p50 0.31879→0.99317、raw std 0.16665→3.0173、raw norm中央値2.266→153.144。L2 normalize前の`z_q`で既に方向集中が成立するので、正規化単独の原因ではない。

**結果C：両modeともProjector出力の方向集中、Full FTでより極端。** Full FT `z_q` cos p50 0.31879→0.99994166、q std 0.07263→0.00071494（step1）→4.518e-6（step10）。LoRAではq step1 std0.007153、step10 0.007643で、強い方向集中はあるがFull FTほどではない。警告Rule（q std<1e-4 AND cos p50>0.999）はFull FT step10のみ真、LoRAは偽。heuristicが偽でもLoRAの同方向性0.993を軽視しない。

### Step1 optimizer更新監査

| mode | group | grad_norm | update_norm | relative_update |
| --- | --- | ---: | ---: | ---: |
| LoRA | lora_A | 0 | 0.00002386859 | 2.98034654e-6 |
| LoRA | lora_B | 0.0309570922 | 11.50100481 | 1.15010048e13（B初期norm=0で分母epsilon） |
| LoRA | projector | 2.1856347045 | 24.90061088 | **1.439908435** |
| Full FT | backbone | 1.3921865068 | **277.44090050** | **0.3600864710** |
| Full FT | projector | 2.1856346752 | 24.90060874 | **1.439908310** |

- **Projectorのstep1更新は両方式でほぼ同一の大きさ**（絶対norm約24.90、更新前normに対し相対約1.44）。同条件のAdamW `lr=0.03`で非常に大きいパラメータ更新という有力な機構的説明。
- **Full FT Backboneの相対更新0.3601**はViTの全trainable parameter群に対するaggregateノルム比。これがCLSの急速な方向均一化に関係する可能性が高いが、更新normの大きさだけでは因果断定しない。
- LoRA B `relative_update=1.15e13`はゼロ初期化の分母epsilonによる機械的値で、更新倍率の研究的尺度に使用しない。A grad0と微小updateは初期LoRAとAdamW weight decayで説明可能。
- Full FTとLoRAはBackboneもProjectorも各stepで同時に更新するため、**「Projectorだけが原因」「ViTだけが原因」「lr0.03が唯一の原因」とは言えない**。この観測からの**局在化**と更新要因の**因果同定**を分離。

### D5時点の推論と次の実験候補（未承認）

- **最有力候補**：AdamW初期LR0.03（warmupなし）がViTとProjectorに対して過大で、初回に各群の大きなupdateを生み、方向集中へ寄与した可能性。SGD用のLR値とAdamW LRの数値は最適条件として互換とみなせない。ただし**LRを減らした対照実験が未実施なので未検証仮説**。
- **反例・別説明**：Projectorのoutputが特定方向へ写像する初期過渡、ViTの更新・正規化、EMA Key/QueueとInfoNCEの相互作用、stochastic training minibatchに対する短期過学習、AdamWの最初のstep特性。初期QueueはK65536のうちstep10で2560だけが実Keyになっており、後期表現学習・下流性能を10stepから判定できない。
- **次の比較設計候補**：元の`lr=0.03`を保存したまま、**(i) AdamW低LRを両group同時に変える**対照、**(ii) ViT/LoRAとProjectorでLRを分離する**群別ablation、必要なら **(iii) projector-frozen／backbone-frozen**の短時間probe ablation。最初はLoRA＋Full FTの同一64画像・同一比較条件・更新1/2/10回・CPU/GPU Gateを対にして観測。学習率候補・実行回数・GPU枠は別specでユーザー判断を必要とする。実行条件を黙って変更しない。
- **現段階では**20stepや5epochへの延長、ActivityNet/Sequential Loader本統合、Full FT対LoRAの性能ランキングを推奨しない。まず低LRまたは分離学習率の因果切り分けを先行。

### Status と非主張

- **D4 GPU工学・モード比較：PASS**（ユーザーが表示したlocal summary集計に基づく）。
- **D5記述的局在化：支持あり**（Full FTではViT CLS側から、LoRAではCLS初期多様性を保ちつつProjector側から、強い方向集中）。`D5 causal attribution`は**未検証**。
- **未成立**：低LRがCollapseを防ぐ、表現品質が向上する、online videoで動的情報を学習できる、LoRAがFull FTより優れた表現を獲得した、など。
- **Research Workspace記録のみ**。ChatGPTはGPU新run・研究コード編集・研究コードcommit/pushを行っていない。次の実験条件をこの文書だけで承認したことにはしない。
