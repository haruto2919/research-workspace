---
date: 2026-10-21
project: sequential-video-lora-analysis
record_type: experiment-evidence
status: g3-1step-pass-user-reported-summary
evidence_level: user-provided-terminal-output-and-local-json-summary-two-1step-gpu-runs
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: dev
github_branch_head_observed: 2c269915f0c5036fa2a460959fdf401d54456026
local_execution_commit: unverified-from-terminal-output
spec: ../specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md
---

# G3 ImageNet 2GPU・1step：LoRA / Full FT ユーザー提供コンソールEvidence

## Evidenceと判定境界

2026-10-21、ユーザーが研究サーバ `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora` の `.venv` から、`CUDA_VISIBLE_DEVICES=0,1` により `model.train_mode=lora` と `full_ft` をそれぞれ `training.max_steps=1` で実行したコンソール全文をチャットに提供した。

双方で `global_step=1` のJSONメトリクス、`finite=true`、`queue_ptr=256`、`Saved run artifacts in ...` が表示された。今回の可視コンソール範囲ではtracebackなし。これにより **1stepの処理経路が完走し保存処理に達したという限定的Evidence** がある。

**正式なG3 PASSとはまだ判定しない**。保存先の `run_report.json`、`rank0_report.json`、`rank1_report.json`、`metrics.jsonl` の内容は未受領。特に `vram_gate.status` は `Saved run artifacts` の出力だけでは分からず、runnerはVRAM BLOCKEDでも保存メッセージを出し得る。ローカルGPU状態・VRAM peak・final checks・入力ハッシュ・local HEAD SHAはGitHubから直接確認できない。

## 実行条件と出力先

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke model.train_mode=lora training.max_steps=1
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke model.train_mode=full_ft training.max_steps=1
```

- LoRA: `log/moco_vit_ddp_smoke/lora/imagenet/2026-10-21_18-24-37-810425`
- Full FT: `log/moco_vit_ddp_smoke/full_ft/imagenet/2026-10-21_18-29-51-042240`
- Config上のGPUレシピ: 2GPU × local128、global256、FP32、AdamW LR0.03、Queue K65536、m0.999、T0.2。ただし実行後のresolved configによる個別検証は未受領。
- Hugging Faceのunauthenticated Hub warningは表示されたが、双方のrankでcheckpoint loadは続行。`ViTModel`の `classifier.weight`/`classifier.bias` はUNEXPECTEDと表示された。分類Headを持たないpretrained `ViTModel`としては想定内。

## ユーザー提供の1step数値（rank0コンソール）

| 指標 | LoRA | Full FT |
| --- | ---: | ---: |
| global_step | 1 | 1 |
| epoch | 0 | 0 |
| loss | 7.361582279205322 | 7.361582279205322 |
| positive_cosine_mean | 0.7652500152587891 | 0.7652500152587891 |
| negative_cosine_mean | -0.0001778690959326923 | -0.0001778690959326923 |
| hardest_negative_cosine_mean | 0.36582903861999516 | 0.36582903861999516 |
| lr | 0.03 | 0.03 |
| trainable_grad_norm | 2.1858539299619006 | 2.59136685995681 |
| query_feature_std | 0.07224058359861374 | 0.07224058359861374 |
| queue_ptr | 256 | 256 |
| step_time_sec | 3.39038220999646 | 4.143081277026795 |
| finite | true | true |

## 解釈（事実と分離）

- 学習前のfull-FT ViTとLoRA-Bゼロ初期化ViTが一致し、ProjectorとQueueも共通seedから作られる実装なので、同じ入力viewなら初回出力とlossの一致は設計上期待される。**二つの実runが同一のinput hashかはmetrics.jsonlを照合するまで未確定**。
- trainable-grad normはパラメータ集合が異なるため大小を直接比較しない。
- 初期Queueの大部分はランダムkeyのまま（256 / 65536のみ置換済）であり、step1のcosine分離やlossから学習性能は論じられない。
- step_timeは一回のtrain_stepの計測であり、データ読込と安定した長時間スループットを示さない。

## 次の判定と停止条件

1. `run_report.json`で双方 `status=completed`, `verdicts.smoke_completed=true`, `vram_gate.status=PASS`, `verdicts.vram_feasible=true` を確認し、rank別peak reserved / headroom ≥1024 MiBを検査する。
2. `final_checks`のrank state一致、Frozen Base不変、Query/Key frozen一致を確認する。failuresが空か、missing rank reportsが空かを確認。
3. `metrics.jsonl`の二つのrunで同じstepのinput_sha256（rank0とrank1、im_q/im_k）を比較し、初期Loss一致の条件を明確にする。
4. 両runがG3 1step PASSなら次は同じレシピでImageNetの10step、続いて20step（specのGPU許可/Gateを遵守）。一方でもBLOCKEDならレシピを勝手に変更せず、VRAMまたは数値不安定のEvidenceに基づき別判断とする。
5. ActivityNet framesはその後の別Gate。現時点で非逐次の表現性能比較、Loss低下、LoRA優位、時間情報獲得は主張しない。

## Sources

- [approved spec](../specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)
- [CPU implementation verification](2026-10-21-vit-ddp-moco-full-ft-lora-implementation-cpu-verification.md)
- [GitHub runner dev](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/vit_moco/moco_vit_ddp_smoke.py)

## 追記（2026-10-21）：run_reportとmetricsの集計提示によるG3 1step正式判定

ユーザーは研究サーバ上で両runの `run_report.json` と `metrics.jsonl` を読み取るPythonスクリプトを実行し、その集計コンソールを提供した。以下は **ユーザー提示のローカルファイル集計結果** であり、GitHub connectorからローカルJSONの原本を直接開いた結果ではない。初版の「正式なPASS判定は保留」はその時点の履歴であり、今回の追記によって **ImageNet / 2GPU・1step G3は両方式PASS** と判断を更新する。

| ゲート項目 | LoRA | Full FT |
| --- | --- | --- |
| `status` | completed | completed |
| `last_global_step` | 1 | 1 |
| `verdicts.smoke_completed` | true | true |
| `verdicts.vram_feasible` | true | true |
| `vram_gate.status` | PASS | PASS |
| `vram_gate.reasons` | [] | [] |
| `failures` | [] | [] |
| `missing_rank_reports` | [] | [] |
| `ranks_hold_rank0_state` | true | true |
| `frozen_base_unchanged` | true | true |
| `frozen_key_equals_query` | true | true |
| `metrics.jsonl` first-step `input_sha256` | 両方式で一致 | 両方式で一致 |

GPU別メモリ値は各runでrank0/rank1同一:

| 単位: MiB / rank | LoRA rank0 | LoRA rank1 | Full FT rank0 | Full FT rank1 |
| --- | ---: | ---: | ---: | ---: |
| peak reserved | 12038.0 | 12038.0 | 16862.0 | 16862.0 |
| headroom | 8057.875 | 8057.875 | 3233.875 | 3233.875 |

- 各rankのtotal GPU memoryは `20095.875 MiB`（上記peakとheadroomの和）。VRAMゲート閾値1024MiBに対し両方式PASS。Full FTはLoRAより **4824 MiB/rank** 多く予約した。
- Full FTでは学習対象のViTが全てtrainableのため、`frozen_base_unchanged=true` および `frozen_key_equals_query=true` は非適用集合に関する真偽（空集合に対する確認）であり、**Full FTの全重みが凍結されていたという意味ではない**。実更新の確認はG2 CPU Gateと今後の追加GPU計測を参照する。
- 入力hashの一致は同じrank順の `im_q` / `im_k` 入力テンソルが両runで一致したという集計結果に基づく。したがって初回forward/lossの一致を比較条件として支持する。
- これは **G3の1step資源・工学チェックのみのPASS** であり、10step/20step、ActivityNet GPU、長期安定性、Loss低下、下流表現品質、時間情報の獲得、Full FTとLoRAの性能順位は未検証のまま。

### 次アクション候補（別run。追加学習未実施）

1. GPUの空きVRAMとcleanな研究コードHEADを再確認。既存runは上書きしない。
2. 採用済みrecipeを変えず、ImageNetで `training.max_steps=10` をLoRA・Full FTそれぞれfreshに実行し、run_reportのVRAM Gate、最終checks、stepごとのfinite loss/gradient/feature stdとQueue更新を確認する。
3. 10stepが両方式PASSした場合のみ `training.max_steps=20` をそれぞれfreshに実行する。AdamW LR0.03はpretrained ViTで安定性未検証の高い候補であるため、failureを観測したときに無断でLR/batch/precisionを変更しない。
4. G3短時間PASSの後、ActivityNet framesへの接続、および別specで表現学習の評価方法・学習率探索を設計する。

**未確認**：実験サーバの現在のdirty状態およびHEAD実値、run JSON全文、長時間のCUDA/VRAM安全性。
