---
project: sequential-video-lora-analysis
date: 2026-10-31
record_type: gpu-diagnostic-local-artifact-evidence
status: l4-both-modes-engineering-pass-cross-lr-strict-comparability-fail-hardware-changed
evidence_level: local-run-report-diagnostic-summary-and-comparator-output-read-directly-on-research-server
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
local_execution_commit: 46ad9833f764ed7d9bb8df45f2687345b7610f2a
spec: ../specs/2026-10-22-vit-moco-configurable-learning-rate-collapse-control-spec.md
---

# ViT＋MoCo 低LR対照（L4）：ImageNet 2GPU・各10step（LR 3e-4）

## 実行内容

ユーザーが研究サーバで以下の2 runを実行した（2026-10-31）。出力ファイルは、AIが同じサーバ上で直接読んだ。

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_query_collapse_lr model.train_mode=lora training.max_steps=10 training.learning_rate=0.0003
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_query_collapse_lr model.train_mode=full_ft training.max_steps=10 training.learning_rate=0.0003
```

- LoRA：`log/moco_vit_query_collapse_lr/lora/imagenet/lr_0.0003/2026-10-31_22-32-03-330318`
- Full FT：`log/moco_vit_query_collapse_lr/full_ft/imagenet/lr_0.0003/2026-10-31_22-38-39-863638`
- 同じディレクトリにある `2026-10-31_22-27-12-279286`（LoRA）にはlogしかない。これはworking treeがdirtyで起動を拒否されたときの残骸で、結果としては使わない。
- 拒否の原因は、未追跡ファイル `AGENT_RESEARCH.md` と `CLAUDE.md` だった。研究リポジトリのローカル設定 `.git/info/exclude` にこの2ファイルを登録して解消した（commitはしていない）。

## 工学Gate（両方式PASS）

| 項目 | LoRA 3e-4 | Full FT 3e-4 |
|---|---|---|
| status / failures | completed / [] | completed / [] |
| commit / dirty | `46ad983` / false | `46ad983` / false |
| 両rankのstate一致 | true | true |
| frozen baseが不変 | true | true |
| 診断の `engineering_status` | PASS | PASS |
| peak reserved（rank0 / rank1） | 12,190 / 12,338 MiB | 18,124 / 18,346 MiB |
| 最小headroom | 7,757.875 MiB（rank1） | 1,749.875 MiB（rank1、閾値1,024） |
| VRAM Gate | PASS | PASS |

**ハードウェア変更**：10/22のD4 runではGPU0/1が2台ともRTX A4500だった。今回は**GPU0がRTX PRO 4500 Blackwell（32GB）、GPU1がRTX A4500（20GB）**という異種構成になっている。torch 2.14.0+cu130、cuDNN 92400、NCCL 2.30.7は同じ。

## 比較器の結果（出力JSONはAIのscratchpadにのみ保存）

| 比較 | comparable | 理由 |
|---|---|---|
| LoRA：0.03（D4） vs 3e-4 | **false** | step0のCLS / z_q / q特徴がbitwiseで一致しない |
| Full FT：0.03（D4） vs 3e-4 | **false** | 同上 |
| LR 3e-4：LoRA vs Full FT | **true** | step0がbitwise一致、学習入力・probeも一致 |

cross-LRの比較では、学習入力、probe、LR以外のconfig identityはすべて一致した。step0の統計量は小数4〜5桁目まで同じ値（例：CLS cos p50は0.0319389 vs 0.0319389）。bitwise不一致の原因は、**GPU0の機種が変わったことによる浮動小数点の差**と推定している（未検証）。厳密な比較Gateとしては不合格であり、以下の0.03との比較は**参考の記述**にとどまる。

## 固定64画像・rank0の推移（画像間cos p50 / mean_dim_std）

| run | CLS s1 | CLS s10 | z_q s1 | z_q s10 | q std s1 | q std s10 |
|---|---|---|---|---|---|---|
| LoRA 0.03（D4） | 0.0319 | 0.269 | 0.993 | 0.994 | 0.00715 | 0.00764 |
| **LoRA 3e-4** | 0.0319 | 0.0319 | **0.453** | **0.455** | **0.0651** | **0.0649** |
| Full FT 0.03（D4） | 0.985 | 1.000 | 0.9999 | 1.000 | 0.000715 | 4.5e-6 |
| **Full FT 3e-4** | **0.0381** | **0.0290** | **0.495** | **0.392** | **0.0625** | **0.0685** |

step0は全run共通で、CLS 0.0319、z_q / q 0.319、q std 0.0726。z_qのrow norm中央値は、3e-4では2.27から2.8前後にとどまった（0.03ではs10で655（LoRA）、5,394（Full FT））。`uniformity_warnings` は3e-4の両runとも0件。

## Step1の更新量（rank0）

| group | grad norm | 相対更新 0.03 → 3e-4 |
|---|---|---|
| Projector | 2.186（両LRで同じ） | 1.44 → 0.0144 |
| Full FT backbone | 1.392（両LRで同じ） | 0.360 → 0.00360 |
| LoRA B（初期値0） | 0.0310 | 更新norm 11.5 → 0.115 |

step1の勾配はLRに依存せず同じ値で、更新量はLRにほぼ比例した（1/100）。Projectorの更新normは `0.03 × √689,024 ≈ 24.9` と一致する。つまりAdamWの初回更新は、勾配の大きさにほぼ関係なく、各parameterをおよそ±LRだけ動かしていることになる。LR 0.03は、Projector重みの典型的な大きさ（約17/830 ≈ 0.02）より大きい。これは数値から読み取れる機構の解釈であり、追加実験による検証はしていない。

## 学習batch側（metrics.jsonl、参考）

- `query_feature_std` のs10：LoRA 0.0618、Full FT 0.0657（0.03では0.0110と6.06e-6）
- lossは10stepを通して7.34〜7.50で、明確な減少はない。positive cosineは0.765から約0.80へ緩やかに上昇。hardest negative cosineは0.37から0.75へ上昇したが、これはQueueが埋まっていく途中の値である。

## 判定と非主張

- **工学**：L4は両方式PASS。
- **記述的な結果**：LR 3e-4では、10step以内にProjectorの方向集中、Full FTのCLS同方向化、Query stdの崩壊のいずれも起きなかった。0.03で見られた初回更新直後のcollapseとは明確に異なる。
- **因果**：「LR 0.03がcollapseの主因である」という仮説を強く支持する結果ではある。ただし、cross-LR比較はハードウェア変更のため厳密Gateを満たしていない。単一seed・10stepの結果でもある。
- **言えないこと**：表現品質、loss減少、長期安定性、LoRAとFull FTの優劣、最適LR。

## 次の候補（未承認）

1. 現在のGPU構成で、0.03を新presetで再実行する（LoRA・Full FTの各10step）。同じcommit・同じハードウェアのbaselineができ、cross-LRの比較を `comparable=true` で確定できる。
2. 3e-4で10stepを超える学習（数百step〜1epoch）を行い、lossが減るか、遅れてcollapseしないかを確認する。presetの `max_steps=10` 固定を外す必要があるので、別specとする。
3. 必要に応じて、1e-4や1e-3などのLR感度を見る。

2026-10-31追記：2はユーザーが選び、[長めの学習spec（1,000step、approved）](../specs/2026-10-31-vit-moco-low-lr-long-run-spec.md)として保存した。
