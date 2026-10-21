---
date: 2026-10-22
project: sequential-video-lora-analysis
record_type: diagnostic-implementation-and-experiment-spec
topic: vit-moco-query-collapse-diagnostic
status: approved
approval: approved
approved: 2026-10-22
approved_scope: B1-B5 diagnostic design and CPU engineering contract
gpu_run_authorized: false
code_change_requested_in_this_turn: false
source_brainstorm: ../../../../secretary/notes/brainstorm/2026-10-22-vit-moco-g3-collapse-diagnostic-roadmap.md
related_spec: 2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md
evidence: ../experiments/2026-10-21-vit-ddp-moco-g3-imagenet-10step-console-evidence.md
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_reference_commit: 2c269915f0c5036fa2a460959fdf401d54456026
---

# ViT＋MoCo：Full FT / LoRAの正規化Query特徴量Collapse原因診断 spec

## 1. 結論、目的、承認状態

**Status: approved（2026-10-22、B1〜B5の採用をユーザーが明示承認）**。本書は元のdraftを2026-10-22の「採用します」により承認済みに変更した。B1〜B5の計測契約は実装用に採用したが、**この会話では研究コード変更・GPU診断run・GitHubコードcommit/pushは実行しない**。GPU実験はB6として引き続き実行時の個別承認を必要とする。

目的は、2026-10-21のImageNet-1K・2GPU・10stepで観測された **L2正規化後のMoCo Query出力のバッチ内分散急減**を、(a)事前学習ViTのCLS、(b)Projector正規化前出力、(c)正規化後Queryのどこで、(d)どのoptimizer更新時点で生じるか、同一固定入力で再現・局在化すること。

**このspecの成功は「原因の切り分けに必要な再現可能な測定が得られること」であり、学習成功・LoRA優位・Collapse治療・ViTの時間情報獲得を意味しない。**

## 2. Authority、現状のEvidence、研究仮説を分離

### 確認済み事実（研究記録とGitHubコード）

- [前段spec](2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)は現行ViT DDP MoCoエンジニアリングの採用済み契約。G3の短時間GPU smokeはImageNet、`model.train_mode=full_ft|lora`、ViT-B/16 CLS [B,768]、Projector `Linear(768,768) → ReLU → Linear(768,128)`、L2-normalized Query、AdamW lr0.03、2GPU×local128、FP32、K65536、m0.999、T0.2を採用する。
- [G3 10step Evidence](../experiments/2026-10-21-vit-ddp-moco-g3-imagenet-10step-console-evidence.md)はユーザー提示の研究サーバログに基づく。LoRA・Full FTとも10step完了、VRAM/最終DDPチェックPASS、全10stepのモード間入力SHA256一致。GPUあたりpeak reserved＝LoRA 12,338 MiB、Full FT 18,346 MiB、FT headroom＝1,749.875 MiB（下限1,024 MiB）。
- 既存runnerの `query_feature_std` は `float(out.q.detach().std(dim=0).mean())`。 `out.q` はQuery Projector `z_q` を `F.normalize(z_q, dim=1)` した128次元特徴。**ViT CLS直接の分散ではない**。
- Step1の `query_feature_std=0.0722405836`（両mode）、step2はLoRA 0.00634409 / FT 0.00081159、step10はLoRA 0.01095892 / FT 0.00000605682。full-FTで正規化Query表現の均一化が非常に強い疑い。これは数値有限性Gateの失敗を意味しない。
- [現在のコード](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/tree/dev)の観測点：`model/vit/vit_moco_backbone.py`の `forward` は `last_hidden_state[:,0,:]`。 `self_supervised/moco/standalone_moco.py` は `backbone_q` / `projector_q` と `encode_query` を持つ。 `self_supervised/moco/ddp_moco.py` のMoCo `forward` はEMA・Queueを**変更**するため、評価probeにそのまま呼び出してはならない。
- `scripts/vit_moco/moco_vit_ddp_smoke.py` は学習時に `DDP forward → loss.backward → optimizer.step` を実行。ログの`global_step=1`特徴は**step1の更新前forward**で計測される。ログのstep2特徴は前回更新後の**別バッチ**のforwardなので、既存ログだけで「最初のupdateが原因」と因果断定できない。
- [Hydra preset](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/conf/moco_vit_ddp_smoke.yaml)の既定 `training.checkpoint_steps: []`。既存10step GPU runの学習後重みを保存・復元できるというEvidenceはなく、診断は新しいfresh runを必要とする。
- GitHub上`dev` HEADを2026-10-22に確認：`2c269915f0c5036fa2a460959fdf401d54456026`。研究サーバの未コミット/未push changeやGPU状態はGitHubから直接確認できない。

### 解釈・仮説（未検証）

- H1: **AdamW lr0.03、warmupなし**がViT/Projectorに対して大きすぎ、最初の更新で表現が変化した可能性。旧ResNet50＋SGDの学習率数値を移植したことが動機だが、lrが原因であることは**未実証**。
- H2: CLSの多様性は残るが、共通学習対象Projectorで崩壊。
- H3: FTではpretrained ViT CLS自体の多様性も低下し、LoRAは更新制約により別の挙動をとる。
- H4: MoCoのKey/Queue初期状態やEMAの非定常性が損失・更新方向に関与する。
- H5: 画像内容・augmentation・featureのnorm／正規化が影響。入力hashが両mode間で同じことは適切なaugmentationや十分な画像多様性の証明にはならない。

## 3. Decision Contract：今回の対象・対象外

### 対象（採用済み）

1. **固定probe画像でのfeature診断**：2つの学習方式について同一画像・同一前処理の出力を、更新前後で観測する。
2. **3つの観測位置**：`cls = backbone_q(pixel_values)` [B,768]、`z_q = projector_q(cls)` [B,128]、`q = F.normalize(z_q,dim=1)` [B,128]。
3. **時点**：`probe_step=0`（fresh init、いかなるoptimizer stepより前）、`1`（1回更新完了後）、`2`（2回更新完了後）、`10`（10回更新完了後）。既存 `metrics.jsonl.global_step=n` のforward-time指標と更新後probeを**別指標・別時点**として保存する。
4. **比較条件の維持**：G3の2GPU、学習用ImageNet train、same pretrained checkpoint／Projector seed／Queue seed、AdamW lr0.03、weight decay、学習batch、MoCo設定、augmentation、FP32、epoch schedulerを維持。両mode間の各step学習viewのSHA一致も照合する。
5. **optimizerの作用点の補助監査**：Query側のBackbone（FTではViT／LoRAではQ,VのLoRAパラメータ）とProjectorの勾配・パラメータ更新を区別。実効更新量と勾配ノルムは別物として扱う。
6. **工程別停止**：CPUでtest・人工データ診断の正しさを先に確認、GPU実行前は新しい明示許可、VRAM Gate・finite・DDP整合が破れれば停止。

### 対象外（このspecでは実施しない）

- 既存G3 10stepログや採用済みspecの上書き、G3結果の遡及的な仕様変更。
- LR `0.03` の自動変更。**低LR比較は次段階の別判断・別spec**（候補として `3e-4` 等はbrainstormにのみ記載）。
- ActivityNet、Sequential Loader、動画chunk、strict online、Long-run／5epoch、Linear Probe、表現品質の優劣や動的情報獲得の主張。
- Meta公式ResNet50のparityコード・runner・E5結果・Comet経路の変更。
- GPUのメモリGateを通すための暗黙のbatch縮小・AMP有効化・解像度変更・Queue縮小・GPU数変更。
- 元のMoCo forward（EMA/Queue enqueue）、Optimizer更新数式やLoRA injectionの意味論の変更。

## 4. 計測対象と統計の計算契約（B1/B2/B4/B5で採用済み）

**固定probe dataset**：ImageNet-1K `train` を読み取り専用で使用。通常のImageFolderのstable sample listとseed0から決定的に画像indexを選び、**計64枚**を採用する案。同じ64枚を両modeと両rankで用い、index・相対path・クラス・選択seed・データ集合の`samples_sha256`・実入力テンソルSHA256を出力。これは診断用の固定サンプルであり、ImageNet validationや下流性能評価ではない。低多様性の偶然を避けるため、同じ画像重複indexは禁止する。

**固定probe transform**：ランダムaugmentationを含めない。画像 `Resize(256) → CenterCrop(224) → ToTensor → Normalize`（正規化mean/stdは既存HF processorの値）を候補とする。RGBへの変換を固定し、拡張・出力サイズをprobe manifestへ記録。**学習DataLoader側の既存2-view stochastic transformは変更しない**。

**probe計算**：`model.backbone_q` と `model.projector_q` を直接呼び（`model.forward`・DDP forward・MoCo forwardは呼ばない）、`torch.inference_mode()` / `no_grad()` と固定evalモードで評価。前後のtraining/evalフラグとCPU/CUDA RNG stateを正確に復元し、optimizer、EMA Key、Queue、Sampler cursorを一切変更しない。ViTのdropout挙動がmodeによって変化する可能性を明示するため、診断用evalモードを全時点で統一する。既存Projectorは`Linear → ReLU → Linear`でBatchNormなし。

**VRAM配慮**：学習のglobal/local batch 256/128は固定。probeのみ**microbatch 8枚**で64画像を推論し、出力CLS/Projector/qの小tensorをCPUへdetach・集計。評価用の追加optimizerやfull-batch activation graphを作らない。各rankのGPU peak/headroomを旧G3と同じ算出式で評価し、閾値`>=1024 MiB`を守る。probeでGPU OOMなら自動設定変更せずBLOCKED記録。固定probeを各rankで同一計算する場合は、rank間の結果の一致可否・probe input hashを記録する。

**各feature段の必須メトリクス**（`cls` / `z_q` / `q`、それぞれprobe64枚の特徴matrix `X[N,D]`）：

| 指標 | 定義 | 用途 |
| --- | --- | --- |
| `mean_dim_std` | `torch.std(X, dim=0, correction=1).mean()` | 既存`query_feature_std`に対応。ただしprobe用固定64枚なので数値を直接同一条件とみなさない |
| `median_dim_std` | channel別sample stdの中央値 | 一部channelのみ残る退化との区別 |
| `row_norm_min/median/max` | `torch.linalg.vector_norm(X,dim=1)` | 出力norm消失、L2正規化の影響 |
| `offdiag_cosine_mean/p50/p95` | **異なる画像同士**の`F.normalize(X,dim=1) @ F.normalize(X,dim=1).T`の対角を除外した分布 | 全画像が同一方向に近づく場合の直接的な指標 |
| `finite`, `shape`, `zero_norm_count` | 非有限率／shape／normが0（またはEPS近傍）のサンプル数 | 測定数値・前処理の妥当性 |

**時点間指標**：同じprobe index順について `probe_step=0` と各`1/2/10`のsamplewise cosineとL2差の分布を、CLS/z_q/qごとに計算する。初期の各modeの全probe特徴について、初期値一致（FT対LoRA、許容差は既存CPU baseline G0準拠）を検証。

**更新量の補助指標**：`optimizer.step()`前後のtrainable `backbone_q` と `projector_q`を区別し、少なくともstep1の相対更新量（`||Δθ||2 / max(||θ_before||2, ε)`）、更新後`parameter_norm`、`gradient_norm`を記録。LoRAの`A`と`B`は別集計。更新snapshotは必要なCPU側へ移し、**Full FTの全parameterのGPU上二重cloneを禁止**。全parameter追跡でメモリ・処理が厳しい場合は、対象moduleの範囲と計測値の種類を固定してから実装する（§12 Blocking）。grad normが小さいこととupdateが小さいことを同一視しない。

**Key/Queue**：現行MoCoの更新・最終state監査を再利用。第一段階のprobeはQuery-onlyであることを明示し、EMA Keyなどの診断を追加の必須工程にしない。Key/Queue由来の要因を完全に排除したと主張しない。

## 5. Runと時点の厳密な定義

1. 両modeは初期化をfreshとし、**既存10step runのcheckpointからresumeしない**（現在のG3出力では`checkpoint_steps=[]`）。
2. `probe_step=0`はoptimizer更新0回、probe inference前後でQuery/Key/Queue/optimizer/全RNG/samplerの状態が変わらないことをtestする。
3. 初回train stepは従来と同じ`two_crops → DDP forward → backward → AdamW optimizer.step`。その後、**学習バッチと独立の同じ64枚probe**で `probe_step=1` を測る。
4. 同様に `probe_step=2`、`probe_step=10`。probeはMoCo loss・EMA・Queueに関与しない。計測による学習state変化が起きたらdiagnostic runnerはFAILとする。
5. FTとLoRAの各train stepで学習view `im_q/im_k`のsha256（全rank）を対応比較し、違いがあれば**同一入力でのmode比較不可**として記録する。Probe64枚のinput tensor SHAは全probe_step・rank・modeで同一を必須とする。
6. Step毎の既存run `metrics.jsonl`（train batchに対する更新前forward指標）と新規 `probe_metrics.jsonl`（固定probeに対する更新後forward指標）を別ファイルにし、`global_step`の意味を混ぜない。
7. 対応するrankが同じprobeを得るので、診断ヘッダにはprobe selection policy・actual sample SHA・checkpoint hash・Query初期param hash・seed・git HEAD／dirty状態を保存する。

## 6. 実装契約と影響範囲（B3採用済み。コード変更は別依頼）

**最小差分方針**：既存MoCo数式・DDP runner通常経路を変更せず、既存runnerに「診断有効時だけ呼ばれる計測hook」を挿入する。既定の診断OFF時は現行G3の同じ入力／損失／checkpoint／出力の意味を保つ。新しい汎用学習エンジン、MoCo再実装、独立した大規模training loopを追加しない。

| ファイル候補 | 最小変更案 |
| --- | --- |
| `scripts/vit_moco/moco_vit_ddp_smoke.py` | `diagnostics.enabled`（既定false/省略可）の分岐をstage0と指定step後に追加。callbackはQuery-only、Dataloader/Optimizer/Queueの既存数式には触れない。診断出力のwrite/exceptionをrank別reportに反映 |
| `scripts/vit_moco/query_collapse_diagnostic.py`（新規） | fixed probe index/transform、Feature統計、before/after query stage測定、CPU集計、元state復元検証。可能なら既存`build_model` / dataset helper再利用 |
| Hydra既存preset | 診断OFF時の既定挙動を保持するため、`+diagnostics.enabled=true`のような明示overrideを候補に。定義の詳細は実装時に独自判断して研究条件を変えない |
| `tests/`に新規の診断専用unit / CPU integration test | 正規化出力の統計、probe snapshot副作用なし、同一入力、境界条件、元G3挙動非回帰 |

**禁止**：`self_supervised/moco/{standalone_moco,ddp_moco,vit_ddp_moco}.py`の損失、EMA、Queue動作への介入。ResNet50側 `scripts/parity/*` の変更。元G3既存report schemaを破壊する変更。診断がOFFの状態で余分なforwardが動く設計。

## 7. 出力と追跡可能性（採用済み）

診断runは元G3と**別output namespace**（例：`log/moco_vit_query_collapse/<mode>/imagenet/<timestamp>/`）に新規保存し、過去の`log/moco_vit_ddp_smoke/`を上書きしない。Hydraのoverrideによるoutput dir変更で衝突が起きないこと、同名dirはfail-closedであることを確認。

必須artifact:

- `resolved_config.yaml`、`run_report.json`、`rank0_report.json` / `rank1_report.json`、従来の `metrics.jsonl`：既存G3の工学証跡（形式互換）。
- `probe_manifest.json`：64枚のindex/relative_path/class、sample list hash、固定transform、probe batchsize、入力tensor hash、pretrained identity、git commit/dirty。
- `probe_metrics.jsonl`：`mode`、`rank`、`probe_step`、`probe_timing=after_optimizer_step`（0はbefore any optimizer）、feature stage、sample size、統計、初期時点からのdistance。
- `update_metrics.jsonl`：step1でのbackbone / projector（LoRAはA/Bを分離）のgradient norm / parameter norm / update norm（測定できた範囲を明示）。
- `diagnostic_summary.json`：engineering status、観測された特徴統計の時系列、強い均一化の**兆候フラグ**、未成立の研究主張、failure/VRAM gateと関連run ID。

`mean_dim_std`やpairwise cosineは浮動小数の閾値を事前に科学的な「Collapse成功/失敗」と決め打ちせず、分布と比率を報告。例えば `q.mean_dim_std<1e-4` かつ offdiag cosine p50`>0.999` は**警告候補**に過ぎず、研究上の確定判定には使用しない（B5承認：科学的な絶対閾値は採用せず、heuristicは警告に限定）。

Checkpointの追加保存は本diagnostic scopeの必須要件にしない（既存`checkpoint_steps=[]`維持）。本結果は保存時点の統計を主Evidenceとし、モデル重みの追加解析・再利用には別設計が必要。

## 8. Gate・受入条件（工学 vs 研究）

| Gate | 成功条件 | 失敗／停止条件 |
| --- | --- | --- |
| D0 現行コード・計測位置の静的監査 | ViT CLS/Projector/raw q/qの計測pathが現在コードと一致、`MoCo.forward`副作用回避を明文化 | QueryとKeyを混同、post-updateとpre-updateを混同 |
| D1 CPU unit（実データ不要） | 人工 `X`（全一致、互いに異なる、ゼロベクトル、非finite）でfeature statsと警告を検証。正規化前後、dim、diagonal除外の計算が期待値通り | 非finite計測の見逃し、shape不一致、統計未定義の黙殺 |
| D2 CPU tiny ViT／2-rank gloo | `probe_step=0/1/2`、両mode、同一input SHA、probeがRNG/Queue/EMA/optimizer/sampler stateを変更しない、diagnostic disabledで既存指標が再現。tinyモデルでは確定的に特徴をCollapseさせるfixtureも使う | 追加probeで学習state、loss trace、rank stateが変化、ファイル上書き、rank単独hang |
| D3 CPU existing regression | 元G3のunit、LoRA、MoCo、parity関係の関連testに退行がない（元の既知失敗と新規失敗を区別） | 新規テスト失敗、新規回帰 |
| D4 GPU実測（別許可） | 採用済みGPU基準レシピを保持しFT/LoRA各10step freshでprobe0/1/2/10を記録、各rank finite・固定probe同一、VRAM headroom ≥1024MiB、DDP最終state PASS、入力ハッシュ整合、全probeファイル整合 | OOM・NCCL error・nonfinite・VRAM gate不足・probeのstate副作用、どれか1つでもあればBLOCKED |
| D5 解釈 | どのfeature stage・どの更新タイミングで分散/方向が変わったかを根拠付きでまとめ、CLS/Projector/正規化の残存仮説を明示 | 小さいstdだけで原因を断定、未実験のLR効果を既証明扱い |

**D4が工学PASSしても、「MoCo表現品質が良い」「Collapseが解消した」とは判定しない**。特徴の均一化が確認された場合は、現象の再現が診断成果になる。研究段階の次の設計判断は、その結果から行う。

## 9. 再現性・比較条件・資源ガード

- Research code repository／branch／Git HEADをrun直前に記録。現在GitHubで確認したbase `dev@2c269915f0c5036fa2a460959fdf401d54456026`は設計調査の基準であり、実装後commitは別。GPU runはcleanな研究コードからのみ行う（既存G3の前提）。
- ImageNet train、HF `google/vit-base-patch16-224` identity、sample-list SHA、固定probe64画像のrelative pathとimage tensor SHAを報告。新規ダウンロードはしない。
- 既存`RandomResizedCrop/ColorJitter/Grayscale/GaussianBlur/Flip`の学習側2-view、dataset sampler、DDP NCCL×2、local128、seed0、FP32/no AMP/TF32 off、AdamW `(lr=0.03, wd=1e-4, betas=(.9,.999), eps=1e-8)`、K65536/m0.999/T0.2、epoch horizon5を基準として固定。診断の目的で値を変えない。
- fixed probe画像の評価はCPU常駐 + microbatch8で `no_grad`、学習tensorを長期間GPU保持しない。step1直後のピーク/headroomを必ず保存。FT 10step既存headroom1750MiBは余裕が限られるため、Probe中に実際の最高peakがGateを上回るか測定する。
- Rank別failureには段階／step／rank／例外／OOM memory summary等を保存、どちらか一rankに問題があれば全体をFAIL扱い。自動batch縮小・AMP・余分なGPU追加は禁止。
- 古い10step runの特徴統計と新しいprobeを直接数値的に同列比較しない（学習バッチとfixed probe、forward時点と更新時点が異なる）。同一seed・input hash・実装commitが揃う部分だけ比較。

## 10. 実装・実験の順序と実行境界

1. **完了（2026-10-22）**：ユーザーがB1〜B5を採用し、spec statusを`approved`へ更新。B6のGPU実行は別許可として維持。
2. 別途許可された範囲で研究コード `dev` に最小差分実装（新規feature stats、probe hook、CPU tests）。コードを編集した場合、対応するengineering-task workflowとcode commit/pushの承認境界を守る。
3. CPU artificial/tiny2rank gloo regressionを実行し、研究Workspaceに検証結果を記録。実装完了を示すstatus移行は必須Gate通過後のみ。
4. GPU runは**明示的な別許可**が出たときのみ、研究サーバでGPU ID/空き/clean HEAD/データのpreflight後に実行。LoRAとFTを順番にfresh 10stepで診断し、同一サンプル・入力hash・rank整合・VRAMを比較。
5. run resultを`experiments/`へEvidenceとして保存。観測箇所が特定できたら、低LRによる因果仮説検証の別spec、または現行G3 20step継続の可否を改めて決定。

**承認済みspecは後続のengineering-taskに引き継げるが、今回のユーザー発話は設計採用のみである。研究コード変更・Git code commit/push・GPU run・データdownload・Comet通信は今回実施しない。**

## 11. 既存baselineとの互換性・非回帰

- 元`moco_vit_ddp_smoke`で診断がOFFならrun artifacts、Loss、Queue、EMA、Grad、DDP state、checkpoint/resume identityの意味を維持。出力ディレクトリの衝突による既存結果上書きは禁止。
- 公式MoCo vs自作MoCo E5 ResNet50 parityを再実行せず、既存test/契約を破壊しない。既存`ddp_moco.py`等のMoCoエンジン本体を診断に合わせて改造しない。
- 新規probe実行によって学習データの巡回順・augmentation乱数・同一stepの学習viewsが変わらないことをチェック。計測実行のあり／なしで**同一seedと入力の学習トレース**が一致するようにする（bitwiseが成立しない場合は誤差の実測と根拠を示し、無断で許容差を拡大しない）。
- `+diagnostics.enabled=true`等のCLI項目と出力namespaceは新規に追加しても、既存実行引数は変更しない。追加診断の再現性をcheckpoint／resumeの暗黙な改造で実現しない。

## 12. Ambiguity Gateの解消と承認範囲（2026-10-22）

**設計BlockingはB1〜B5の採用により解消済み。** ユーザーが本draftのB1〜B5の推奨条件に対して「採用します」と明示した。下表を診断実装用の**確定契約**として扱う。B6は「実行時の許可Gate」として残るが、specをapprovedにすることを妨げる未解決の設計項目ではない。**GPU診断runの許可が出たことにはならない**。

| ID | 判断 | 採用した値・境界 |
| --- | --- | --- |
| B1 | **採用済**：固定probeと前処理 | ImageNet trainの決定的64枚（seed0、重複なし）、Resize256/CenterCrop224/HF processor正規化、RGB固定。固定probeは学習側random augmentationと切り離す |
| B2 | **採用済**：観測点と学習条件 | Full FT / LoRA fresh各10step、同じ固定probeでoptimizer更新0/1/2/10回時、CLS768・Projector raw128・normalized Query128。元のAdamW lr0.03、global batch256、2GPU、FP32を維持。低LR比較はscope外 |
| B3 | **採用済**：実装・互換性 | 元runnerの診断OFFをデフォルトに保ち、診断有効時のみoptional hook＋専用小moduleで測定。MoCoコアの損失・Queue・EMAには介入しない。別namespaceに成果物保存 |
| B4 | **採用済**：optimizer更新量 | step1でFull FT ViT / LoRA A・B / Projectorのgrad・parameter・update量を分離測定。CPU snapshotを優先し、GPU上の全parameter二重cloneは禁止。余分な測定範囲を追加する場合は別判断 |
| B5 | **採用済**：Collapse評価 | 特徴std、行ノルム、画像間offdiagonal cosine等の原値・分布を正本とする。heuristicなCollapse疑い表示は警告に限定し、科学的な成功/失敗の絶対閾値にはしない |
| B6 | **実行許可Gate（未許可）** | specの承認とGPU利用は別。GPUの2rank×10step診断runは、ユーザーが実行を明示許可するまで実施しない。今回の承認はB1〜B5の仕様だけ |

**Non-blocking（設計上残件なし）**：局所helper名、テストファイル分割、JSONキー衝突のない追加フィールド順などは既存styleに従う。spec template参照`.company/templates/requirements-spec-template.md`はGitHubのmainでは未発見のため、既存採用specの形式で記述。これによる研究条件の曖昧さはない。

## 13. Spec Gate（approved）とImplementation Handoff

- **Readiness: approved（2026-10-22）**。§12 B1〜B5のblocking設計項目をユーザーが採用し、観測対象・scope・再現性・success/fail/stop条件が確定した。B6はGPU実行時の別許可であり、現在 **NOT_AUTHORIZED**。
- **Breaking changes: なし**。従来G3のCLI・正常系・checkpoint/Queue/EMAの算術・出力schemaは維持し、診断OFFが既定。2026-10-21の採用specをsupersedeしない。
- **コード変更境界**：承認済み仕様は後続のengineering-taskで実装可能な契約。**今回の「採用します」はB1〜B5の設計承認であり、GitHubコードの変更・commit/pushまで実行する依頼ではない**。研究コードの編集は後続の明示実装依頼に従う。
- **GPU検証境界**：実装後もD4 GPUの実行は明示の個別許可が必要。D0-D3のCPU engineering確認とD4 GPUとを混同しない。
- **Implementation handoff**:
  - approved spec: 本ファイル（2026-10-22承認）。
  - purpose: CLS/Projector/raw q/normalized qの固定probeによる、更新0/1/2/10回時のCollapse局在化。
  - code: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、2026-10-22の調査基準`2c269915f0c5036fa2a460959fdf401d54456026`。ローカルの現在HEAD・dirtyは実装開始時に別途確認。
  - allowed design scope: optional diagnostics hook、fixed-probe helper、CPU tests、診断用JSON sidecars。元runner/G3の後方互換維持。
  - verification: D0→D1→D2→D3 CPUを先行。D4 GPUは別許可、D5解釈は観測後。
  - excluded: 学習率の変更、5epoch/20step/ActivityNet本実験、ResNet50 E5、性能優劣の主張。
  - unresolved **research questions** (not spec blockers): LRが崩壊原因か、CLS vs Projector、Query vs Key影響、表現品質の改善策。

## 14. 根拠・関連文書

- [G3 10step工学PASS・Collapse疑い](../experiments/2026-10-21-vit-ddp-moco-g3-imagenet-10step-console-evidence.md)
- [元のViT DDP Full FT/LoRA切替spec](2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)
- [Collapse診断の探索メモ](../../../../secretary/notes/brainstorm/2026-10-22-vit-moco-g3-collapse-diagnostic-roadmap.md)
- [2026-10-15 MTG](../meetings/2026-10-15-mtg.md)
- [Runner](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/vit_moco/moco_vit_ddp_smoke.py)
- [MoCo Query/Key/Projectorの元コード](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/self_supervised/moco/standalone_moco.py)

## 15. Decision history（2026-10-22）

- 依頼：「Collapse診断用のspecを作成したい」→ B1〜B6の案を含むdraftをResearch Workspace mainへ作成。
- 承認：「採用します」→ **直前の会話で確認したB1〜B5の推奨条件を採用**。2026-10-22、statusを`approved`へ移行。
- **B6は独立したGPU許可Gateとして未許可のまま**。元G3で行われた過去の10step実行承認は、この新診断runへの包括許可にはならない。
- コードの変更、CPUテスト、GPU run、コードcommit/push、Comet通信は今回未実施。specを`implemented`へ移行するのは、後続で実装・必須GateのEvidenceを確認したときだけ。
