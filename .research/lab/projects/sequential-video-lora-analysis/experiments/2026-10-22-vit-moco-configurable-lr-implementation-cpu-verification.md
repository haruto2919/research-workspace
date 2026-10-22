---
project: sequential-video-lora-analysis
date: 2026-10-22
record_type: implementation-verification
status: cpu-pass-gpu-not-run
evidence_level: cpu-unit-integration-regression-plus-real-d4-artifact-read
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: dev
base_commit: 312c121975d68e860d5f89bf38ed26aab23b0981
code_state: uncommitted (1 new file + 3 modified files on dev@312c121); not pushed
spec: ../specs/2026-10-22-vit-moco-configurable-learning-rate-collapse-control-spec.md
source: ai-executed-on-research-server (Claude Code)
---

# ViT＋MoCo 任意LR指定（LR可変診断profile）：実装・CPU検証（L0〜L3）

## 結論・Evidence境界

- **CPU Gate L0〜L3：PASS**。承認済みspec（B1〜B6）に従い、学習率を実行時に任意の正の有限値で指定できる診断preset、GPU recipeの「LRだけを例外にする」処理、同mode・異LRの比較モード、CPUテストを実装した。
- AIが研究サーバ上で実行した結果である。全テストは `CUDA_VISIBLE_DEVICES=""` で**CPUのみ**で実行した。GPUは使っていない。
- **L4（ImageNet 2GPU）は未実施**。B7は未許可のまま。**L5（解釈）も未実施**。低LRでCollapseが改善するか、LR 0.03が原因かは、この記録では何も主張しない。
- 研究コードは `dev@312c121` 上の**未commit差分**。commit/pushはしていない。
- 実行環境はD0〜D3の記録と同じ（Python 3.12.3、torch 2.14.0+cu130、transformers 5.17.0、hydra-core 1.3.7）。

## 変更ファイル（研究コード）

| file | 種別 | 内容 |
| --- | --- | --- |
| `conf/moco_vit_query_collapse_lr.yaml` | 新規 | `defaults: [moco_vit_query_collapse, _self_]`。`training.learning_rate: null`（指定必須）、`diagnostics.profile: lr_sensitivity`、出力先`log/moco_vit_query_collapse_lr/<mode>/<source>/lr_<LR>/<timestamp>/` |
| `scripts/vit_moco/query_collapse_diagnostic.py` | 変更（+136/−3行） | `lr_profile`、`learning_rate`（値の検査）、`check_lr_preset`、`check_config`へのprofile検査とLRのfloat化、summaryへの`profile`/`learning_rate`/`first_step_lr`の追加、`compare_learning_rates`、CLIの`--comparison-axis {train_mode,learning_rate}`（既定は従来どおり`train_mode`） |
| `scripts/vit_moco/moco_vit_ddp_smoke.py` | 変更（+9/−1行） | `prepare_config`のGPU recipe照合で、LR profileのときだけ`training.learning_rate`を照合から外す。`run()`で、GPUかつLR profileのときはHydraのconfig名が`moco_vit_query_collapse_lr`であることを要求する。docstringに使用例を追加 |
| `test/model/test_query_collapse_diagnostic.py` | 変更（+216行） | L1/L2のテスト36件を追加（既存33件は変更なし） |

変更していないもの：`conf/moco_vit_ddp_smoke.yaml`、`conf/moco_vit_query_collapse.yaml`（旧G3/D4のpreset）、`ADOPTED_RECIPE`の値、MoCo本体（loss・EMA・Queue）、ViT backbone、データローダ、optimizer（単一param groupのAdamW）とscheduler（`epoch_cosine_lr`）、`compare_runs`、統計・probeの計算、`scripts/parity/*`。

## 実装の要点（specとの対応）

| spec | 実装 |
| --- | --- |
| B1/B2 値域 | `training.learning_rate`をそのまま使う。正の有限なint/floatだけを受理し（上限・下限や候補一覧はない）、`None`/0/負値/NaN/±Inf/文字列/boolは拒否する。受理した値はfloatに直して保存するので、`3e-4`、`0.0003`、整数のLRがすべて同じ1つの値として記録される |
| B3 単一LR | optimizerは従来どおり`AdamW(model.trainable_query_parameters(), lr=training.learning_rate)`の単一group。schedulerも同じ`epoch_cosine_lr`。LoRA A/B、ViT、Projectorはすべて同じLRになる |
| B4 GPU Gate | GPU recipe照合でLRが外れるのは`diagnostics.enabled`かつ`diagnostics.profile=lr_sensitivity`のときだけ。それ以外の採用値、固定probe、fresh 10step、resume禁止はそのまま。さらにGPUでは、Hydraのconfig名が新presetであることを要求する（旧presetに`+diagnostics.profile=...`を付けても通らない）。旧presetとbase presetのGPU runは、従来どおり0.03以外を拒否する |
| B5 比較 | `compare_learning_rates(a, b)`：mode違いはValueErrorで拒否。`comparable`の条件は、両runがPASSかつ完走、LRが異なる、全stepの学習view hashが一致、probe（selection、transform、64枚の各tensor hash、入力hash、dataset hash）が一致、step0のcls/z_q/q特徴がbitwise一致、`optimizer.learning_rate`以外のconfig identityが一致すること。外れた項目は`reasons`に残す。出力には両runのLR・比率・profile・engineering_status・commit/dirty、stageとstepごとのcos p50/p95・mean_dim_std・row norm（raw z_qのnormを含む）、groupごとのstep1更新量（lr、grad、更新前後のparameter norm、update、relative）を含める。commitが異なる場合はその旨を注記する。既存の`compare_runs`と、CLIの既定動作は変えていない |
| B6 必須・出力先 | 新presetではLRの指定が必須で、未指定ならrun成果物を作る前に`ValueError`で止まる。出力先にはLR別のdirectory（`lr_0.0003`など。Hydraは`3e-4`も`0.0003`も同じfloatとして解釈する）を使う。既存の成果物は上書きしない（既存のfail-closed） |

### 実装中に見つかった問題（specの記述からの変更）

spec §3は新presetで`training.learning_rate: ???`とする案だった。しかしHydra/OmegaConfでは、presetを合成するときに**`???`が継承元の値を上書きしない**。このため、そのままでは**LRを指定しなくても0.03が黙って引き継がれ**、LR profileのままGPUで0.03が流れてしまう。これは同じ§3にある「0.03を黙って継承してGPUを動かさない」に反する。そこで`null`で上書きし、検査で必須エラー（`training.learning_rate is required ...`）にした。意図（指定必須・0.03を黙って継承しない）はspecどおりで、仕組みだけを変えている。テストでは、合成後の値が`None`になること（0.03ではないこと）を確認している。

## Gate別の結果

| Gate | 結果 | 観測 |
| --- | --- | --- |
| L0：設計・境界 | PASS | 新presetは旧の診断presetと比べて、`training.learning_rate`（null）、`diagnostics.profile`、hydraの出力先だけが異なる（Hydra composeで確認）。LRの例外はLR profileのときだけ。optimizerとschedulerの追加・変更はない |
| L1：config unit | PASS（24件） | ・GPU（device数をmonkeypatch）：両modeで3e-4、3e-3、1e-4、0.03、1e-9、10.0、1（int）を受理し、floatとして保存。未指定は「required」で拒否。<br>・旧の診断preset・base preset・profileだけ付けて診断を無効にした設定は、0.0003を`adopted recipe`で拒否。未知のprofileは拒否。<br>・LR profileでも、weight decay、local batch、betas、epochs、K、fp32_precision、probe数、max_steps=20は拒否。<br>・`None`/0/0.0/−3e-4/NaN/±Inf/"0.0003"/"abc"/True/Falseは、何も書く前に拒否。<br>・GPUでconfig名が新preset以外（旧preset、base、不明）なら、何も書く前に拒否 |
| L2：tiny ViT／2-rank gloo | PASS（12件） | ・LoRA（3e-4、3e-3）とFull FT（3e-4）のLR profile runが、すべて完走・finite・summary PASS。probeは0/1/2/3 stepの全記録がある。<br>・LRがresolved config、config identity、summary（`learning_rate`、`first_step_lr`）、各stepの`lr`（＝`epoch_cosine_lr(LR, epoch, 5)`）、update記録の`lr`に反映されている。<br>・0.03の診断runと比べて、identityのSHAは異なるが、LRを除いたidentityは一致。学習viewのhashは全stepで一致。<br>・**step1のgradient normはLRに依存せず同一**（同じ初期状態からの1 step目）で、**update normはLRとともに増える**（LoRA projector：3e-4 < 3e-3 < 0.03。Full FTのbackboneとprojector：3e-4 < 0.03）。<br>・LR profile（3e-4）と、診断なし・同じLRのrunで、学習trace（metrics）、final state、identityがbitwise一致した。<br>・既存のrun dirへの出力は拒否。<br>・LR比較：0.03と3e-4はLoRA・Full FTとも`comparable=True`（比率0.01、step0特徴一致、identity一致）。mode違いはValueError。同じLRどうしは「equal」の理由付きで`comparable=False`。runのコピーを改ざんすると（identity、学習view、probe、step0、status）、それぞれ1件の理由として検出された。CLIは`--comparison-axis learning_rate`で動き、出力の上書きは拒否。既定の軸は従来のcross-mode比較のまま |
| L3：回帰 | PASS（新規の失敗なし） | ・同じvenv・`CUDA_VISIBLE_DEVICES=""`で全テストを実行した。<br>・変更前（`312c121`のclean clone）：598 passed / 197 failed / 18 errors。<br>・変更後：**634 passed** / 197 failed / 18 errors（増えた36件は新規テスト）。<br>・**失敗・エラーのtest ID（215件）の集合は変更前後で完全に一致**（D3の記録と同じ既存の失敗：GPU前提、NASのKinetics、`test_moco_5epoch_pair`の1件）。<br>・旧0.03経路のbitwise回帰は、下の「追加確認」1を参照 |

### 追加確認

1. **旧0.03経路のbitwise回帰（`312c121`のclean clone 対 変更後）**：同じtiny CPU条件で、診断OFF（両mode、5 step、step3 checkpoint）と旧診断preset（両mode、3 step、probe 0〜3）を実行した。metrics、final state、config identity、resolved_config、**checkpointのバイト列**、`probe_metrics.jsonl`、`update_metrics.jsonl`、run reportとrank reportのキー構成が、すべて一致した。差分は2点だけ：manifestの`repository`（commitとdirtyの状態）と、summaryに追加した3キー（`profile=None`、`learning_rate=0.03`、`first_step_lr=0.03`）。既存キーの値で変わったのは`output_dir`と`repository`だけ。
2. **実際の旧0.03 D4成果物をbaselineとして読めるか**（読み取りのみ、出力なし）：`log/moco_vit_query_collapse/{lora,full_ft}/imagenet/…`（どちらも`312c121`・clean・PASS）を`compare_learning_rates`で自分自身と比べると、学習view・probe・step0（cls/z_q/q）・LR以外のidentityがすべて一致し、外れたのは「LRが同じ」だけだった。probe step [0,1,2,10]とupdateのgroupも読める。mode違い（LoRA 対 Full FT）は拒否された。→ L4の後、旧D4 runをそのまま同modeのbaselineとして使える。
3. **実CLI入口**（CPU、tiny、出力はscratchpad）：`--config-name moco_vit_query_collapse_lr`でLRを指定しないと`ValueError`で止まり、run成果物は作られない（Hydraの`.hydra/`とログだけが作られる。これは既存の不正configのときと同じ）。`training.learning_rate=3e-4`では完走し、resolved configは`learning_rate: 0.0003`と`profile: lr_sensitivity`。既定の出力先は`log/moco_vit_query_collapse_lr/<mode>/imagenet/lr_0.0003/<ts>`。比較CLIは、以前の0.03のCLI runに対して`comparable=True`、step1のprojector updateは0.433（0.03）対0.00433（3e-4）。commitが異なる2 runなので注記も出た。
4. **ミューテーション確認**：「LRを全GPU runで照合から外す」「GPUのconfig名検査を外す」の2通りの改変は、それぞれL1テストで検出された（確認後に元へ戻し、`cmp`で一致を確認）。
5. lint：flake8のE/F系で新しい違反はない（runnerの既存4件は変更前と同じ）。

## L4（GPU）に必要な前提と手順（未実行・別許可）

- runnerはdirtyなrepositoryからのGPU runを拒否するので、今回の差分のcommitが先に必要。
- spec §10の順序：LoRA（LRを指定）→工学PASSの確認→Full FT（同じLR）。Full FTのheadroomは旧基準で1,749.875 MiBしかない。

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_query_collapse_lr \
    model.train_mode=lora training.max_steps=10 training.learning_rate=<LR>
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_query_collapse_lr \
    model.train_mode=full_ft training.max_steps=10 training.learning_rate=<LR>
# 同mode・異LR（旧0.03 D4 runをbaselineに）
python -m scripts.vit_moco.query_collapse_diagnostic log/moco_vit_query_collapse/lora/imagenet/2026-10-22_04-02-22-441192 \
    log/moco_vit_query_collapse_lr/lora/imagenet/lr_<LR>/<ts> --comparison-axis learning_rate --output <new.json>
# 同LR・異mode（従来の比較）
python -m scripts.vit_moco.query_collapse_diagnostic <lora lr run> <full_ft lr run> --output <new.json>
```

## 未確認点

1. GPUでの実測（VRAM、finite、DDP整合）は未実施。LRを変えてもメモリ量は変わらない見込みだが、大きなLRで非finiteになればBLOCKEDとして記録し、自動でLRを下げることはしない。
2. 旧D4 runは`312c121`のコードによるもの。新runは別commitになるため、比較の妥当性は上記1の0.03経路のbitwise回帰に依拠している（比較出力にも注記される）。
3. 低LRで方向集中が減るか、update normが小さすぎて「学習していないだけ」になっていないかは、L4/L5で判断する（spec §7）。

## 関連

- [承認済みspec（任意LR）](../specs/2026-10-22-vit-moco-configurable-learning-rate-collapse-control-spec.md)
- [D4/D5 Evidence（0.03 baseline）](2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md)
- [D0〜D3 実装・CPU検証](2026-10-22-vit-moco-query-collapse-diagnostic-implementation-cpu-verification.md)
- [低LR対照の探索メモ](../../../../secretary/notes/brainstorm/2026-10-22-vit-moco-post-d4-learning-rate-causal-diagnosis.md)
