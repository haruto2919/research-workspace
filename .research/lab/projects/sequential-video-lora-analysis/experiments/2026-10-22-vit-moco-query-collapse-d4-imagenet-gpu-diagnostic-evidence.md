---
project: sequential-video-lora-analysis
date: 2026-10-22
record_type: gpu-diagnostic-user-console-evidence
status: lora-console-completed-awaiting-diagnostic-summary-and-full-ft
evidence_level: user-pasted-local-console-only
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
