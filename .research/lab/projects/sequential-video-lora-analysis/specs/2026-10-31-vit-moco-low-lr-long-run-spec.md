---
project: sequential-video-lora-analysis
spec_type: experiment-and-implementation
status: implemented
created: 2026-10-31
approved: 2026-10-31
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
base_branch: feature-vit-moco
base_commit: 46ad9833f764ed7d9bb8df45f2687345b7610f2a
source_evidence: ../experiments/2026-10-31-vit-moco-low-lr-l4-imagenet-gpu-evidence.md
parent_spec: 2026-10-22-vit-moco-configurable-learning-rate-collapse-control-spec.md
---

# ViT＋MoCo 低LR（3e-4）長めの学習：1,000step 診断run spec

**Status: implemented（2026-10-31。CPU Gate T0〜T3はPASS。2026-11-01に `3874568` でGPU L5を実行し、工学PASS。L4とのbitwise一致は不合格。§11参照）**。**Status: approved（2026-10-31。B1〜B7はユーザーが推奨案のとおり承認）**。研究コードの実装とGPU runは、それぞれ別に許可を得てから行う（§9）。

## 1. 目的と背景

[L4 Evidence](../experiments/2026-10-31-vit-moco-low-lr-l4-imagenet-gpu-evidence.md)では、LR 3e-4にすると、0.03で起きていた初回更新直後のcollapseが10step以内では起きなかった。具体的には、Projectorの方向集中、Full FTのCLS同方向化、Query stdの崩壊のいずれも見られなかった。ただし、10stepではQueueが本物のkeyで埋まっておらず、lossもほとんど動かなかった。

このspecで答えたい問いは次の2つ：

1. LR 3e-4のとき、Queueが埋まった後（256stepより後）にMoCo lossは下がるか。
2. 10stepを超えて学習したとき、遅れてcollapseが起きるか。

表現品質を評価することは、このspecの目的に含めない（§3）。

## 2. 前提（コード調査で確認済み、`46ad983`）

- runnerはGPUでは `max_steps ∈ {1, 10, 20}` しか受け付けない（`moco_vit_ddp_smoke.py`、`GPU_MAX_STEPS`）。診断側もGPUでは10stepに固定され、probe `[0,1,2,10]` 以外は拒否する（`query_collapse_diagnostic.py`、`GPU_MAX_STEPS = 10`、`check_config`）。**このため、1,000stepのrunには新しいpresetとprofileが必要**。
- Queueは K=65,536、global batch 256 なので、**256step目で全体が本物のkeyに置き換わる**（`real_key_count`）。
- LRは5epochを1周期とするcosineで、epochの区切りごとに更新される。warmupはない。1epochは5,004stepなので、**1,000stepの間、LRは3e-4のまま一定**。
- 診断runはresumeを禁止している（probe step 0 は「optimizer stepが0回」の状態を意味するため）。
- 1stepあたりの実測時間（L4）：LoRAが約1.45秒、Full FTが約1.72秒。1,000stepならそれぞれ約25分、約30分（probeの時間は含まない）。

## 3. Decision Contract（承認済み）

| # | 論点 | 採用 |
|---|---|---|
| B1 | 学習の長さ | **新規に1,000step**。1epoch（5,004step）以上は、結果を見たうえで別に許可を得る（別specまたはこのspecの改訂） |
| B2 | LRとschedule | **LR 3e-4**。既存のepoch単位cosine（`epochs=5`、warmupなし）をそのまま使う。LR以外のrecipeはL4と同じ |
| B3 | 学習方式と順番 | **LoRAを先に実行し、工学PASSを確認してからFull FT** |
| B4 | probeを取るstep | **`[0, 1, 2, 10, 100, 256, 500, 1000]`**。固定64画像のprobe（seed 0、Resize256→CenterCrop224）、microbatch 8、更新量の計測（`update_steps=[1]`）は既存と同じ |
| B5 | checkpoint | **1,000step目にだけ保存する**（`checkpoint_steps=[1000]`）。途中では保存しない |
| B6 | 表現品質の評価 | **対象外**。kNNやlinear probeは別spec |
| B7 | GPU | **`CUDA_VISIBLE_DEVICES=0,1`**（GPU0がRTX PRO 4500 Blackwell、GPU1がRTX A4500。L4と同じ構成） |

### 対象外

- 1epoch以上の学習、LRの探索、複数seed、Backbone / Projectorで別々のLRを使うこと
- 表現品質、LoRAとFull FTの優劣、最適なLRに関する主張
- 同じハードウェアでの0.03 baselineの再実行（厳密な因果の確認は未完了として残す）
- 既存のG3/D4/L4 preset、0.03のrecipe、既存の出力namespaceの変更
- Comet連携（既存方針どおり、runディレクトリを記録の正本とする）

## 4. 実装要件（最小差分）

1. 新しいpreset `conf/moco_vit_lr_long.yaml`
   - `moco_vit_query_collapse_lr` を継承する。
   - `diagnostics.profile` を新しい値（例：`lr_long`）にする。
   - `diagnostics.probe_steps: [0, 1, 2, 10, 100, 256, 500, 1000]`、`training.checkpoint_steps: [1000]`
   - `training.learning_rate` は従来どおり**必須**とする。省略時に値を黙って補わない。
   - 出力先は `log/moco_vit_lr_long/${model.train_mode}/${data.source}/lr_${training.learning_rate}/<timestamp>`
2. GPUでのstep数の例外は、**新しいprofileかつ新しいpresetのときだけ** `max_steps=1000` を許可する。新しいprofileでの許可値は1,000だけ。既存の `GPU_MAX_STEPS`（runnerの1/10/20、診断の10）は、他の経路ではそのまま維持する。
3. 新しいprofileのGPU runでは、B4のprobe step、B5のcheckpoint、固定probeの設定以外の値をすべて拒否する（fail-closed）。LRについては既存の `lr_sensitivity` と同じ扱い（正の有限値のみ受け付ける）。
4. 既存の比較器（`query_collapse_diagnostic`）で、新しいrunとL4 runの step 0/1/2/10 を比較できるようにする。L4と新runを同じmode・同じLRで比べる場合と、新runのLoRAとFull FTを比べる場合の両方が対象。既存の比較器で足りるなら変更しない。
5. 1,000stepでの `metrics.jsonl` は、既存どおり毎stepを記録する。lossの移動平均などの集計は解析時に行い、runnerには集計機能を追加しない。

## 5. 互換性

- 既存presetを使ったCPUとGPUの経路は、変更前とbitwiseで一致しなければならない（診断のON/OFF、0.03、`lr_sensitivity`の各経路）。
- 全テストの失敗ID集合が、変更前後で同じであること。

## 6. 検証（実装時、CPUのみ）

| Gate | 内容 |
|---|---|
| T0 | 新しいpresetとprofileに対する単体テスト。LR未指定・`max_steps≠1000`・probe stepやcheckpointの変更を拒否すること、他のpresetでは新profileを拒否すること、出力先が別になること |
| T1 | 既存経路の非回帰。D0〜D3およびL0〜L3相当のbitwise一致テストが通ること。失敗ID集合が変更前後で同じであること |
| T2 | tiny ViT・2rank・glooで、新profileを短いstep数に置き換えて完走させること。probe・checkpoint・summaryが揃うこと |
| T3 | 比較器が新runのprobe step（0〜1000）を読めること。L4 runとstep 0/1/2/10だけを突き合わせられること |

## 7. GPU run（L5、別に許可が必要）

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_long model.train_mode=lora training.max_steps=1000 training.learning_rate=0.0003
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_long model.train_mode=full_ft training.max_steps=1000 training.learning_rate=0.0003
```

- 研究リポジトリをclean commitにしてから実行する。runnerがdirtyな状態でのGPU runを拒否する。
- LoRAがPASSしてからFull FTを実行する。

## 8. 成功条件

### 工学的な成功（PASS / FAIL）

- 両方式とも `status=completed`、`failures=[]`、全stepがfinite、rank間でstateが一致、frozen baseが不変（LoRA）、VRAM Gate PASS、診断の `engineering_status=PASS` であること。
- probe step 0/1/2/10 の CLS / z_q / q が、同じmodeのL4 runとbitwiseで一致すること（決定性の確認）。一致しない場合は報告し、原因を調べるまで科学的な解釈を保留する。
- 1,000step目のcheckpointと、全probe stepの記録が揃っていること。

### 科学的な観測（記述的に報告し、閾値でPASS / FAILを決めない）

- loss、positive cosine、hardest negative cosineの推移。**256stepより前と後を分けて**、移動平均（例：100step窓）で示す。
- 固定probeの各stepにおける CLS / z_q / q の画像間cos p50・p95、mean_dim_std、row normの推移。
- 既存の警告ヒューリスティック（`q mean_dim_std < 1e-4` かつ `cos p50 > 0.999`）が出るかどうか。これは警告であり、科学的な固定閾値としては扱わない。
- LoRAとFull FTの違いは記述するが、どちらが優れているかの判断はしない。

loss が下がらない場合も、遅れて collapse が起きた場合も、そのまま結果として報告する。

## 9. 実行の境界

- このspecの承認は、**研究コードの実装を許可するものではない**。実装は `engineering-task` で、ユーザーが改めて指示してから行う。
- GPU run（§7）は、実装とCPU検証の後に**別に許可を得る**。1epoch以上の延長も別に許可が必要。
- commit、push、ブランチ操作は、ユーザーの指示があったときだけ行う。

## 10. 承認記録

- 2026-10-31：ユーザーがB1〜B7を推奨案のとおり承認した。

## 11. 実装記録

- 2026-10-31：`feature-vit-moco@46ad983` の上に未commitで実装した。CPU Gate T0〜T3はすべてPASS。全テストの失敗ID集合は、変更前後で同じ36件だった。GPU L5は未実施で、実行前に研究コードのcommitが必要。比較器には `--comparison-axis prefix`（`compare_prefix`）を追加した。§8の「L4 runとのbitwise一致」の確認にはこれを使う。詳細は[実装・CPU検証記録](../experiments/2026-10-31-vit-moco-low-lr-long-run-implementation-cpu-verification.md)。
- 2026-11-01：ユーザーが研究コードを `feature-vit-moco@3874568` としてcommitし、GPU L5（LoRA→Full FT、各1,000step）を実行した。両方式とも工学Gate PASS。§8のL4とのbitwise一致は**不合格**（step0とstep1のlossは一致するが、step1の勾配から先がずれる。GPU backwardのrun間非決定性と推定）。そのため、科学的観測は暫定の記述として記録した：Queueが埋まった後、lossは単調に減少し、遅れて起きるcollapseは観測されなかった。詳細は[L5 Evidence](../experiments/2026-11-01-vit-moco-low-lr-l5-1000step-imagenet-gpu-evidence.md)。
