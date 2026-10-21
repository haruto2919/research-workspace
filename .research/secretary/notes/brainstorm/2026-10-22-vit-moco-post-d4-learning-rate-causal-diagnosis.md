---
date: 2026-10-22
project: sequential-video-lora-analysis
source_todo: "D4・D5によりCLS/Projectorの方向集中が観測されたため次にすべき研究手順を相談"
topic: vit-moco-post-d4-learning-rate-causal-diagnosis
status: exploratory
tags: [brainstorm, research, moco, vit, lora, learning-rate, collapse, ablation]
---

# D4後の研究判断：AdamW学習率とBackbone/Projectorの方向集中の因果切り分け

## 現在地と中心の問い（2026-10-22）

**中心の問い**：AdamW `lr=0.03`（warmupなし）が、MoCo Queryの特徴方向集中を引き起こしているのか。それともBackbone / Projector自体・MoCo機構・入力／queue等の要因か。元G3/D4を温存して、最小の対照実験でどこまで検証できるか。

**確認済みEvidence（ユーザー提示の研究サーバJSON集計による、正本JSONをChatGPTが直接読んだものではない）**：

- [2026-10-22 D4/D5 Evidence](../../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md)では、ImageNet固定64枚の `probe_step=0/1/2/10` に対し2GPU×10stepのLoRA/Full FTの工学Gateが両方PASS。訓練の全step入力hash、probe入力、step0のCLS/z_q/q特徴も比較コマンドで一致・bitwise一致。
- Full FT CLSのoffdiagonal cosine p50は`0.03194 → 0.98495`（更新前→1回更新後）、step10には`0.99999748`。CLS std`0.846 → 0.1177 → 0.001861`。Full FTには**Backbone出力段から方向集中**が観測された。
- LoRA CLSのcos p50は更新前→1→2で`0.03194 → 0.03192 → 0.03189`、初期2stepの分布を概ね維持。LoRA z_qのcos p50は`0.31879 → 0.99317`と、**Projector raw出力段で方向集中**が観測された。
- Full FT z_q cos p50は`0.31879→0.99994166`（step1）。両modeともL2正規化前に方向集中があるため、`F.normalize`のみの問題ではない。Raw z_qの標準偏差・normは増大しており、zero-vector collapseとは区別する。
- Projectorのstep1更新normは両mode`≈24.9006`、相対更新norm`≈1.4399`。Full FT Backboneは更新norm`≈277.4409`、相対`≈0.3601`。LoRA Bの相対更新値は分母ゼロ（epsilon代用）のため大きさの比較指標には使用できない。
- 工学的D4 PASSは表現品質PASSではない。警告未発火のLoRAもProjector方向集中（cos p50約0.993）が顕著。Queueは10step後でも65536中2560のみ実Keyに置換。

**現在実装（GitHub read-only確認）**：

- `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`, `scripts/vit_moco/moco_vit_ddp_smoke.py` は `torch.optim.AdamW(model.trainable_query_parameters(), lr=training.learning_rate, ...)`。BackboneとProjectorは**単一の学習率**、epochごとのCosine scheduler、**warmupなし**、`training.learning_rate: 0.03`。
- 元`conf/moco_vit_ddp_smoke.yaml`はG3基準条件を固定、診断専用`conf/moco_vit_query_collapse.yaml`でのみ64画像診断ON。`scripts/vit_moco/query_collapse_diagnostic.py`はGPU診断の`max_steps=10`、学習レシピ・probeの採用値をfail-closedで検証。runner自体もGPUの採用レシピを固定。したがって **`training.learning_rate=0.0003` のCLI overrideだけで次のGPU診断を実行する設計にはなっていない**。明示的な新specと新recipe identity、CPU非回帰検証が必要。
- GitHubは研究サーバのlocal dirtyや当時実行したcode SHA、raw JSONなどを直接確認できない。変更開始時に最新HEADと旧実験のprovenanceを照合する。

## 原因候補と反例

| 仮説 | 支持する観測 | まだ説明しきれない点 |
| --- | --- | --- |
| H1 AdamWの初回高LR更新が過大（最有力） | Projector相対更新≈1.44、Full FT Backbone≈0.36、2モード共通。AdamW初回は非ゼロ勾配の各座標でおおむね`−lr·sign(g)`になる | 低LR実験未実施。LR以外の差を除去した対照が必要。更新normの大きさはそれ単独で原因確定しない |
| H2 Projectorの初回更新が方向集中の主因 | LoRAのCLSほぼ不変でz_q cosが0.993、Projector更新約24.9 | 同じstepにLoRA Bも更新。Projectorだけ更新OFFならどうなるか未検証 |
| H3 Full FTのViT Backbone更新が追加悪化を引き起こす | CLS自体cos0.985へ急変、Backbone更新norm277.44 | Projector学習がMoCoの勾配・次stepのBackbone更新を変えるため単純な切り分けではない |
| H4 学習損失/Queueの非定常性・初期化が原因 | 10step時点でQueue大半が初期random状態、EMA Keyも相互作用 | Query-only fixed probeではKey/Queue寄与を独立操作していない |
| H5 測定上の見かけ/極小ノルム | normalized stdのみでは大きさに依存 | raw z_qとCLSのcosも高い、normは増大しており純粋なL2 artifactでは説明しにくい |

AdamW初回更新の概算：通常のbias correction後、十分に非ゼロの勾配座標では`Δθ_i ≈ −lr·g_i/(|g_i|+eps)`（plus decoupled decay）。Projector約68.9万parameterなら `0.03×sqrt(689024)≈24.9`、Full FT ViT約8579万parameterなら`0.03×sqrt(85.8e6)≈278`。観測24.90と277.44に非常に近い。**高LRが大規模な初回更新を生む機構的な説明と整合**するが、特徴集中の因果証明には実験が必要。

## 発散：候補実験と比較

| 候補 | 何を変更するか | 得られる区別 | コスト/限界 |
| --- | --- | --- | --- |
| A：両groupの一律低LR比較（**第一推奨**） | `lr=0.03`を既存baselineとして維持し、両modeの別runで`lr=3e-4`（100分の1の候補） | 初回大更新を抑えるとCLS/z_q/qの方向集中が緩和されるか | Backbone/Projectorの寄与はまだ混ざる。10stepで単に動かないだけの結果を成功扱いしない |
| B：中間LRスキャン | `lr=3e-3`（候補）、必要に応じ`3e-4`との間 | LR responseの量依存性 | 追加GPU run、依然因果的なmodule分離にならない |
| C：2×2 group-LR factorial | Backbone/LoRAとProjectorそれぞれ`high=0.03`/`low=3e-4`（候補） | Backbone更新LRとProjector更新LRの効果・交互作用 | 実装とrunが増える。初期段階ではA後に条件を絞る方が小さい |
| D：frozen moduleの短時間ablation | freeze Projectorまたはfreeze Backbone | どちらのmoduleの更新が必要条件か | AdamW対象・MoCo Key EMA・勾配フローと研究上の比較意味が変わるので別specが要る |
| E：warmup試験 | 初期LRを0.03までゆっくり上げる | 初回突発更新との関連 | LR scheduleとoptimizer初期の影響が混合、現段階ではAより識別力が低い |
| F：20step/ActivityNet直行 | 現行0.03のまま先へ進む | 技術的pipeline確認 | すでに大きな方向集中が起きているので表現品質の課題を切り分けない（優先度低） |

**推奨判断**：まずAとして、新たな`learning_rate=3e-4`（未承認候補）を両modeへ適用する**10step D4相当の診断対照**をspec化し、CPU Gate後に明示GPU許可を得て実施する。既存`lr0.03`のD4 runはbaselineとして利用。中間`3e-3`は、LR100分の1条件で出力維持は確認できても更新が小さすぎるなど判断不能な場合に候補とする。もしAで明らかな改善が出た場合、C（Backbone/Projector LR 2×2切り分け）を次のspec候補として優先する。

## Aの実験契約候補（研究spec未承認）

### 固定

- Model pretrained identity、ViT-B/16、train_mode full_ft/lora、Q/V LoRA rank8 alpha8、MoCo InfoNCE・EMA・K65536・T0.2、AdamW（wd1e-4、betas=(.9,.999)、eps1e-8）、epoch horizon=5、初回epochのLR定義を現行recipeに整合。
- ImageNet train（同じsample-list hash）・seed0・同じ学習view train input hash、DDP2rank（GPU0,1）、localB128/global256、FP32/AMP off/TF32 off、画像224、同じ学習拡張。
- Query-only fixed-probe ImageNet trainからの決定的64画像、Resize256/CenterCrop224、microbatch8、probe_step0/1/2/10（最初の更新直後を最優先）。
- 診断metricsは`cls`、`z_q`、`q`のmean_dim_std、row norm、画像間offdiag cosine p50/p95、step0 drift、step1更新量（Backbone/LoRA A・B/Projector）、finite、VRAM。既存計算方式を変更しない。
- 工学`engineering_status=PASS`、両rank state一致、probe一致、少なくとも1024MiB headroom、失敗時は自動降格せずBLOCKED。Full FTの既存headroomは1749.875MiBで狭い。

### 変更する唯一の実験変数（候補）

- baselineは現在保存済み`lr=0.03`。新規に`lr=0.0003`を**Backbone/LoRAとProjector両方に同値で適用**。Optimizer本体やその他の学習設定は変更しない。
- 別のHydra preset名/出力namespace/config identityで新規保存し、既存G3/D4preset・report・既存smokeを一切上書きしない。
- `training.learning_rate`の値変更は現在のGPU fail-closed契約に抵触するため、`runner.py`と`diagnostics.py`のレシピチェックに対する**明示的な新しい検証済み契約**が必要。現行`--config-name moco_vit_query_collapse training.learning_rate=0.0003`をそのままGPUで実行しようとしても拒否されるため、ユーザーに無承認overrideを案内しない。
- 実装時にbaseline `lr0.03`の計算とconfigが以前とbitwise一致することをCPU regressionで検証。新モードでは異なるlrなのに同一`config_identity`と記録されないこと。
- GPUを伴うrun、GitHub code commit/pushは、engineering-taskとGPU Gateの承認境界を守る。

### 事前に定義するSuccess / Failure（提案）

- **工程Gate**：両mode新runのfinite、DDP、probe/rank state、VRAM headroom、入力hash一致、出力非上書きがPASS。失敗ならBLOCKED。
- **現象Gate**：同一固定probeで、新`3e-4`のstep1・2・10においてCLS/z_q/qの画像間cos分布がbaselineより「どれだけ」低下したか、normalized q stdとnorm・update_normを数値で比較。**科学的な固定のcollapse PASS/FAIL閾値は自動採用しない**（既存B5の趣旨）。
- **学習停止の偽陽性防止**：`mean_dim_std`が初期値に保たれた理由が、単にupdateが極端に小さいだけではないか。実際にパラメータが更新される、gradがfinite、step10の学習量、queue埋まり具合を並記。これは10stepで有用な表現を獲得したという意味ではない。
- **原因解釈**：低LRで強い方向集中が緩和されれば「LRへの感度・高LR関与の支持」。改善しなければLRだけでは説明できない、もしくはもっと適切なLR・別のloss/queue設計が必要。単一seed・10stepだけで長時間表現品質は評価しない。

## 次の順序と検討すべき判断

1. このbrainstormをResearch Workspace `main`へ**探索メモ**として保存。仕様・実装・GPU許可ではない。
2. ユーザーが希望すればresearch-specで「一律低LR対照（0.03 vs 候補3e-4）」について数値、対象mode、実験seed、baseline再run要否、出力namespace、CPU Gate、VRAM Gate、GPU許可、停止条件のblockingを確定する。
3. approved specの後に限りengineering-taskで診断preset/recipeチェック・CPU testsを最小差分追加。現行baselineの実験を変えない。
4. GPU runは別許可があった場合だけ2GPU×各10step、LoRA→Full FTを順番に実施する。既存baselineとtrain view hash／step0 CLS,z_q,qを比較し、原因解釈を実験記録に残す。
5. 仮説が支持されたらBackboneとProjectorのLRを切り分ける2×2（group LR）条件のspec化を次候補。LRの最適化・multi-seedやLPはさらに後段。
6. 研究上の課題「長時間動画から動的情報を獲得したか」は本診断では扱わない。再びFeature Collapseが疑われない学習条件を確立してからActivityNet単一frame→Sequential Loaderへ段階統合する。

### research-spec handoff候補（未承認）

- 対象: `sequential-video-lora-analysis`のViT＋MoCo Full FT / Frozen ViT＋LoRA
- 目的: AdamW高LRがCLS/Projectorの方向集中へ与える影響を、採用済みD4基準条件と比べて識別
- 採用候補: 一律低LRを先に測定、その後必要なら群別LR実験
- Candidate LR: `3e-4`、追加候補`3e-3`。**ユーザー未採用**
- 変更対象（候補）: 同一研究コードrepositoryのrunner recipe/diagnostic preset checks、専用config、小規模CPU tests、別output
- 不変: ImageNet、seed、ViT pretrained、MoCo mechanics、EMA/Queue、DDP2、globalB256、FP32、probe64／steps0,1,2,10、既存baseline
- Evidence: [2026-10-22 D4/D5](../../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md)
- 未確定: LR候補の正式採用、GPU実行/コードcommitの許可、baseline再実行要否、条件数、後続因果ablation
