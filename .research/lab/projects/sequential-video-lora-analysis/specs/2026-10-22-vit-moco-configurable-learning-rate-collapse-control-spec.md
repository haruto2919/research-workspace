---
date: 2026-10-22
project: sequential-video-lora-analysis
record_type: experiment-and-implementation-spec
topic: vit-moco-configurable-learning-rate-collapse-control
status: draft
approval: pending
user_decided: "LRを固定候補に限定せず実行時に任意指定できる仕様"
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
code_review_commit: 312c121975d68e860d5f89bf38ed26aab23b0981
source_brainstorm: ../../../../secretary/notes/brainstorm/2026-10-22-vit-moco-post-d4-learning-rate-causal-diagnosis.md
related_spec: 2026-10-22-vit-moco-query-collapse-diagnostic-spec.md
related_evidence: ../experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md
---

# ViT＋MoCo：任意Learning Rate指定とCollapse低LR対照実験spec

## 1. 目的・状態・Authority

**Status: draft（2026-10-22作成、ユーザーによる詳細設計の承認待ち）。今回のユーザー要求で「Learning Rateを任意に指定できること」は確定要件。** 以下の入力検証、preset方式、比較軸、実験回数などの具体的な実装契約は提案であり、§11のBlockingを明示採用するまで`approved`に昇格しない。

目的：ImageNet 2GPUのMoCo（ViT Full Fine-Tuning／Frozen ViT＋Q/V LoRA）の10step固定probeで観測された特徴量の方向集中が、高い初期AdamW LR（現在`0.03`）によるものかを、**LR以外の条件を固定**した実験で検証できるようにする。

このspecは旧G3 / D4の承認済み`lr=0.03` recipeを**変更・supersedeしない**。新しい明示的な「可変LR診断experiment」経路を追加し、研究者がCLIで任意のLR値を選べるようにする。新しい学習率の「最適値」をこのspecで固定しない。

対象：`sequential-video-lora-analysis`。研究コード（GitHub）は`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、read-only確認時のHEADは`312c121975d68e860d5f89bf38ed26aab23b0981`。研究サーバの未commit変更・実行時HEADはGitHubから未確認。

## 2. Evidence（確認済み事実と研究仮説を区別）

[D4/D5 Evidence](../experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md)のユーザー提示ローカル集計：

- `lr=0.03`、ImageNet 2GPU、Full FT/LoRA 10stepは両方式のengineering GateがPASS、全stepの学習入力、64枚固定probe、step0初期特徴が一致。
- Full FTのCLS固定probe off-diagonal cosine p50：`0.03194 → 0.98495`（step0→step1）、step10では約`0.999997`。ViT CLSの方向の均一化が初回から顕著。
- LoRAのCLSはstep1/2でcos p50約`0.032`を維持したが、Projector raw出力`z_q`のcos p50は`0.31879 → 0.99317`。Full FTの同段はstep1`0.99994`。
- Projectorの初回`relative_update`は両方式約`1.4399`。Full FT Backboneの初回`relative_update`は約`0.3601`。
- raw `z_q`の出力normは増大し、単なるゼロベクトル化やL2正規化のみの現象ではない。
- **仮説**：AdamWの高い初期LRとwarmup無しによる大きな初回更新が方向集中を招いた可能性。**低LR対照の実測がまだ無いので因果未確定**。工学Gate PASSと表現品質PASSを混同しない。

## 3. ユーザー確定要件と第一推奨設計

### 確定した要件

1. LRを`0.03`・`0.003`・`0.0003`などの**事前に列挙した値に制限しない**。任意の学習率をユーザーが実行時にCLIから指定できる。
2. 診断の意味と既存baselineを維持しながら、高LR／低LRの特徴軌跡・更新量を比較できるようにする。
3. 「spec作成」は研究コードの変更・GPU run・commit/pushの許可ではない。

### 推奨する最小のインターフェース（承認待ち）

- 新規Hydra preset候補：`conf/moco_vit_query_collapse_lr.yaml`。既存`moco_vit_query_collapse.yaml`を継承し、そのpresetを指定した場合だけ「LR可変の診断プロファイル」を明示的にONにする。
- **入力は既存フィールド`training.learning_rate`を再使用する。** `lr`や`backbone_lr`の別フィールドを冗長に増やさない。CLI例（**承認・実装後のみ有効、現コードでは通らない**）：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke \
  --config-name moco_vit_query_collapse_lr \
  model.train_mode=lora training.max_steps=10 \
  training.learning_rate=0.0003

# 例：別の値をユーザーが直接指定（Full FT）
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke \
  --config-name moco_vit_query_collapse_lr \
  model.train_mode=full_ft training.max_steps=10 \
  training.learning_rate=0.003
```

- LR未指定の場合は実行前に明確な`ValueError`／Hydra required-value errorで停止（新presetで`training.learning_rate: ???`とし、`0.03`を黙って継承してGPUを動かさない）。
- 入力検証の推奨：**有限の正の実数のみ**（`lr > 0`かつ`isfinite(lr)`）。特定の「採用値一覧」も恣意的な上限・下限も設けない。`0`（更新しない対照）、負数、`NaN`、`Inf`、非数値、boolはこの実験では拒否する。`0`更新対照を後で使う場合は別の明示条件として扱う。巨大な正のLRが有限でも実行中に非finite／OOMとなればGateはBLOCKED。
- LRは**Backbone（Full FT）またはLoRA A/BとProjectorへ同じ1つのスカラー**として適用。初回の科学的変数は「共有LR 1個」のみ。**group別LR、Projector-only freeze、warmup有無の切替は対象外**（次の別spec）。
- 検証可能な既定候補は`3e-4`とするが、これを唯一の合法値・ハードコードする値にはしない。`3e-3`、`1e-4`等も入力可能にする。

## 4. GPU Fail-closedと既存の研究条件の保護

現在の`scripts/vit_moco/moco_vit_ddp_smoke.py`の`ADOPTED_RECIPE`には`training.learning_rate=0.03`があり、GPUでの他値overrideは拒否される。また`scripts/vit_moco/query_collapse_diagnostic.py`は固定probeとfresh10stepを検証する。

**推奨実装契約**：

1. 元`conf/moco_vit_ddp_smoke.yaml`、元`conf/moco_vit_query_collapse.yaml`、既存G3/D4経路の`lr=0.03`固定とすべての値を維持する。元presetに`training.learning_rate=0.0003`を渡した場合は**従来どおり拒否**する。
2. 新presetの明示的診断profile（例`diagnostics.profile: lr_sensitivity`）のみ、GPUの`ADOPTED_RECIPE`比較で**例外的に`training.learning_rate`だけを可変フィールドとして扱う**。他のすべての`ADOPTED_RECIPE`フィールドは旧来と同じ厳密一致を必須とする。
3. 新profileを旧presetへ誤用できないよう、Hydraの現在のconfig nameとprofileを整合検査する（GPUでは明示的な新preset選択のみ許可）。直接`runner.run(cfg)`を呼ぶCPU unit/smokeでは、同一のprofile/入力値契約がconfigで検証可能な構造にする。単にCLIでflagを足せば旧baselineの制約が無条件で解除される仕組みにしない。
4. `query_collapse.check_config`の固定probe、`max_steps=10`、ImageNet train、fresh開始／resume禁止、Full FT/LoRA、DDP=2／NCCL、global batch256、seed0、FP32／AMP off／TF32 off、ViT-B/16 pretrained、Projector128、LoRA Q/V rank8 alpha8、MoCo InfoNCE/EMA/QueueK65536/T0.2、AdamW wd1e-4/betas/eps、epoch-wise cosine horizon5（warmupなし）等を**変更しない**。
5. Optimizierは引き続き現行の`torch.optim.AdamW(model.trainable_query_parameters(), lr=training.learning_rate, ...)`の単一param groupを使用。既存`epoch_cosine_lr(training.learning_rate, epoch, training.epochs)`を同じ式で使う。新しいscheduler／群別param-group／optimizerの追加を行わない。
6. baseline`0.03`の診断有効／無効、metrics、Queue、Key EMA、state、checkpointの結果が変更前後でbitwise一致することを小型CPU smokeで検証。GPU runの新許可はレシピ値を変えた実験に限定して保護する。

## 5. 再現性・identity・出力

- **別namespace**：例 `log/moco_vit_query_collapse_lr/<mode>/imagenet/lr_<resolved LR>/<timestamp>/`。GPU実行時は既存runとの上書きを例外なく拒否。LRの文字列表現／指数表記に依存しないcanonically resolved numerical LR（`0.0003`と`3e-4`は同値）を`resolved_config.yaml`、`run_report.json`のconfig identity、診断summaryに記録。情報損失／filename不正を避ける形式は実装時に一意に定義。
- config identityには元から`optimizer.learning_rate`が含まれる。新値がconfig identity SHAに反映され、既存0.03との混同・異LR checkpoint resumeは拒否（本診断はresume自体禁止）。
- full record：`resolved_config.yaml`、`metrics.jsonl`、`rank*_report.json`、`run_report.json`、`probe_manifest.json`、`probe_metrics.jsonl`、`update_metrics.jsonl`、`diagnostic_summary.json`、`comparison JSON`（比較後）。サンプルハッシュ、pretrained model hash、code commit、dirty状態、GPU/driver、実際のLR、実行時の先頭epoch LRも記録する。
- 旧0.03 runを**新コードから必ず再実行する要求はしない**。保存済みの2026-10-22 0.03 D4のLoRAとFull FT runをbaselineとして使用可。ただし旧runが残っていて生データ比較が可能であること、sample hash・pretrained identity・dataset・学習view SHA、同一step0 CLS/z_q/qを直接確認する。既存runが欠損・比較不能なら、baselineの再測定は**別GPU実行許可**が必要。
- LoRAとFull FTの低LRは**別々のfresh初期化**で実行し、同LRのcross-mode比較と、同modeでのcross-LR比較を区別する。

## 6. 比較ツール（既存の意味を壊さない）

現在の`scripts/vit_moco/query_collapse_diagnostic.py`の`compare_runs(a,b)`は学習view一致、固定probe一致、step0 CLS/z_q/q特徴bitwise一致を調べる。現在の`comparable=true`は「LR以外の条件が同じ」や「工学Gate PASS」を保証しないため、このままLR因果比較の合格判定に使わない。

- 最小の追加比較モード案：既存CLIを壊さず、明示的な`--comparison-axis learning_rate`（または別の明示的な小helper）を追加。**既存`compare_runs`の標準動作と旧D4結果を変更しない**。
- LR比較では、`train_mode`が同一、両run工学Gate PASS、10step完走、全学習view hash一致、固定probe全64画像・transform・input SHA一致、step0全3stageのfeature SHA bitwise一致、dataset/pretrained/seed/optimizer/scheduler/precision/MoCo条件は **LR以外すべて一致**を要求する。
- 互換性確認では、`config_identity`を正規化して`optimizer.learning_rate`だけを比較対象外にした値が一致することを検証する。ただし新診断profileや保存ディレクトリは比較対象の科学変数に含めず、記録上は明示する。commitが異なる場合は双方のcode identityを報告し、baseline pathのbitwise regressionにより差分の影響がないことを要求する。
- 違うmode同士は`LR効果`の比較対象として拒否。同LRでのLoRA vs Full FT比較は従来のcross-mode pathを別に使用。出力`comparable`と双方の`engineering_status`を別々に扱う。
- 比較結果に両runのactual LR、比率、対象mode、固定probe各stageのstep0/1/2/10、cosine p50/p95、mean_dim_std、raw z_q norm、step1 update（absolute/relative）を含める。比較結果は新規JSON、上書き禁止。差が出たとき数値と理由を保存し、成功と自動解釈しない。

## 7. 計測指標とSuccess/Failure

**観測単位は同じ固定64画像での更新前0・更新後1/2/10**。旧診断の計測位置・統計実装・step0 drift、Key/Queueを触らないQuery-only probeを維持。

| Gate | PASS条件 | Failure / BLOCKED |
| --- | --- | --- |
| L0 設計・code read-only | LR可変の新preset/旧presetの不変境界とCPU/GPU条件、比較軸が分離 | 旧recipeチェックを全runで解除、新しいoptimizerを無条件で導入 |
| L1 設定CPU unit | 新profileで任意の正有限LRを受理、旧profileは0.03以外を拒否、未指定／0／負値／NaN／Inf／非数値／boolを拒否 | 無指定を0.03と黙認、旧profileからGPU LRチェック迂回、他recipe要素の変更を許す |
| L2 tiny ViT/2rank CPU smoke | Full FT/LoRA・複数LR値でfinite、step1更新量が値を反映、probe0/1/2/10全記録、input/state不変、LRによるidentity差、LRのみ異なる比較が成立 | 不一致隠蔽、診断の有無で学習軌跡変化、別軸の変更をLR差と誤認 |
| L3 regression | 旧G3/D4の0.03経路（診断ON/OFF）のtiny CPU学習trace、final state、checkpointsなどbitwise一致。既存CPU関連testsに新規失敗なし | 旧レシピ性能/出力のサイレント変更、旧CLI互換を破る |
| L4 ImageNet 2GPU（別許可） | LoRA→Full FTで各10step fresh、両rank finite・DDP整合、probe入力/step0一致、VRAM headroom >=1024MiB、artifact保存・比較成立 | nonfinite、OOM、NCCL、VRAM不足、hash不一致、probe state変更などはBLOCKED。batch/AMP/K/解像度等を自動変更しない |
| L5 科学的解釈 | baseline0.03と新LRでCLS・z_q・qのcos分布、std、raw norm、更新normを同mode比較し、LR低減への感度と未解決要因を明示 | q std維持だけで「正常学習」断言、low LRによる単なるupdate停止を成功扱い、最適LR・因果を単一runから確定 |

- 数値の判断は**まず連続量の変化**で行う。旧`q.mean_dim_std <1e-4 AND cosine_p50 >0.999`はheuristic警告のままで、科学的な固定閾値へ昇格しない。
- LRを下げてcosineが低下しても、`update_norm`がほぼ0で学習できていない場合は「方向維持」と「表現学習成功」を区別する。
- 10stepのloss非単調や特定LRでの改善は、それだけで表現品質、online動画への汎化、Full FTとLoRAの優劣を示さない。後続のLonger run/LPは別Gate。
- 新LR値が大きすぎて非finiteになった場合はそのままEvidenceとしてBLOCKED。自動でLRを下げたり、条件を変更して「PASS」を作らない。

## 8. 変更対象と明示的な対象外

**実装候補（最小差分）**：
- 新規`conf/moco_vit_query_collapse_lr.yaml`（明示可変LR profile、独立出力先、LR required）。
- 局所`scripts/vit_moco/moco_vit_ddp_smoke.py`の`prepare_config`で、旧recipe完全一致と新profileの「LRのみ例外」を分ける。
- 局所`scripts/vit_moco/query_collapse_diagnostic.py`のprofile/値の検査とLR比較機能を追加（既存statisticsやprobeの算術は触れない）。
- `test/model/test_query_collapse_diagnostic.py`等に正負入力・GPU fail-closed configuration・tiny ViT 2rank・旧bitwise regression・比較軸を追加する。
- **編集禁止**：MoCo loss/EMA/Queue本体、ViT Backbone、既存G3/D4 preset、データローダ、既存出力JSONの意味、ResNet50 Meta parity経路。コードdiffは要求に直接必要な範囲だけ。

**対象外**：BackboneとProjectorの別LR、warmup/Cosine方式変更、Optimizer SGD切替、LoRAのrank/対象module変更、batch size/precision変更、ActivityNet/Sequential Loader、20step/5epoch、多seed confirmatory、評価用Linear Probe、研究コードの無承認commit/push、未承認GPU実行。

## 9. Research Workspace / code / local境界

- 本specの正本は`haruto2919/research-workspace@main`。
- 新LRのコード実装候補は`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`（調査時HEAD`312c121975d68e860d5f89bf38ed26aab23b0981`、実装着手前に最新HEAD/dirtyと局所指示を再確認）。
- 研究サーバのNAS、worktree、GPU状態、dirty/uncommitted、現行のJSON原本はGitHub経由では直接確認できない。
- spec作成／承認と研究コード実装・commit/push・2GPU学習は別の行為。**今回の依頼では研究コード、GPU、checkpoint、Comet通信を変更・実行しない**。GPU L4実行はユーザーの新しい明示許可が必要。

## 10. 再現実験の推奨順序（すべて承認・実装後）

1. 既存`0.03`のLoRA/Full FT結果を不変baselineとして確保し、provenance/raw report/input SHAを確認。
2. **LR値をユーザーが自由に選ぶ**（研究上の第一候補は`0.0003`、これは固定採用値ではない）。
3. LoRAのLR可変fresh10step L4→工学PASSと収束傾向を確認してからFull FT同LR fresh10step L4。Full FTは旧基準でGPU 1台あたりheadroom 1,749.875MiBと狭い。
4. 旧LoRA0.03 vs 新LoRA(LR x)、旧Full FT0.03 vs 新Full FT(LR x)を**同mode cross-LR**で比較。次に新LoRA vs 新Full FTを**cross-mode**で比較する。
5. L5にて「学習率に応じてCLSがどう変わるか」「両modeに共通するProjector方向集中は改善したか」「更新がちゃんと起きているか」を検証。単一seed・10stepの暫定Evidenceとして報告。
6. 低LR改善が観測された後に、必要ならBackbone/Projectorの独立LRを新しい別specで検討。

## 11. Ambiguity Gate（blocking / non-blocking）

| ID | 分類・現在の状態 | 推奨 / 判断理由 |
| --- | --- | --- |
| B1 | **ユーザー確定**：LR任意指定 | ハードコード候補への制限はしない。`training.learning_rate`をCLI入力として再使用 |
| B2 | **Blocking：推奨待ち**：任意LRの値域 | 正の有限実数（任意の正値）を許可。0や非finiteは学習成功と誤認しやすいので拒否。0 LRの対照が必要なら別条件 |
| B3 | **Blocking：推奨待ち**：一律 vs module別LR | 今回は**1つのLRを全trainable Query群（Backbone/LoRA+Projector）へ適用**。module別は次段階 |
| B4 | **Blocking：推奨待ち**：旧GPU Gateとの両立 | 新preset選択時だけLR例外。他recipe/fresh10step/probeはfail-closed。旧presetのGPU値と既存成果物はそのまま |
| B5 | **Blocking：推奨待ち**：baselineと比較軸 | 旧D4の0.03を再利用（raw artifactが比較可能な場合）、同mode cross-LRの新検証モードを付与。既存cross-mode CLIは不変 |
| B6 | **Blocking：推奨待ち**：LR省略・出力先 | 新presetはLRの明示指定必須。異なるLRを確実に区別できる別output namespace、上書き不可、resolved LR記録 |
| B7 | **独立の実行許可Gate（未許可）** | L4のGPU runは研究コード実装・CPU検証と分けて、その都度ユーザーの許可を得る。今回のspec作成は実行許可ではない |

**Non-blocking**：関数名、内部JSON helperの分割、テスト名、LRディレクトリのcanonically safe string、error messageの細部など、科学的な意味やCLIを変えない実装詳細は既存コードstyleに従う。テンプレート`.company/templates/requirements-spec-template.md`が現行mainに無い場合は既存採用specと同形式で保全する。

**Spec readiness**：ユーザー確定のB1以外は上記推奨設計の明示採用が未済のため、**draft**。B2〜B6をユーザーが採用した場合、research-specで`approved`へ移行可能。B7のGPU許可はその後も別Gate。

## 12. Engineering Handoff（承認後用）

- 実装目的：CLIから任意の正有限LRを指定できる、独立したImageNet 2GPU Query-collapse診断プロファイルと、旧0.03対照比較機能。
- 基準：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、調査時`312c121975d68e860d5f89bf38ed26aab23b0981`。ローカルHEAD/dirtyは開始時確認。
- 変更scope：新preset、runner strict recipeのLRのみの例外、diagnostic profile、同mode LR比較、局所CPU tests。
- 旧G3/D4の既定値・結果非変更、MoCo algorithm無変更。
- Verification：L0→L1→L2→L3 CPUを先行、L4 GPUは新規許可後、L5解釈は結果取得後。
- 実装・commit/push・GPUは**本draft作成から自動承認されない**。

## 13. 関連文書

- [D4/D5：両mode工学PASSとCLS/Projector方向集中](../experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md)
- [採用済み固定LR D4 spec](2026-10-22-vit-moco-query-collapse-diagnostic-spec.md)
- [低LR対照の探索メモ](../../../../secretary/notes/brainstorm/2026-10-22-vit-moco-post-d4-learning-rate-causal-diagnosis.md)
- [旧ViT Full FT / LoRA採用spec](2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)
- [現在runner](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/vit_moco/moco_vit_ddp_smoke.py)
- [現在diagnostic helper](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/vit_moco/query_collapse_diagnostic.py)
