---
date: 2026-10-21
project: sequential-video-lora-analysis
record_type: experiment-evidence
status: provisional-console-evidence-awaiting-run-report
evidence_level: user-provided-terminal-output-two-1step-gpu-runs
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
