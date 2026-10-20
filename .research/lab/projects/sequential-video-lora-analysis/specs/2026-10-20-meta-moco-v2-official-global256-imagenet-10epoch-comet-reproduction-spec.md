---
project: sequential-video-lora-analysis
spec_type: experiment-and-engineering
status: approved
created: 2026-10-20
last_updated: 2026-10-20
title: "Meta MoCo v2 Official Global Batch 256 ImageNet-1K 10-Epoch Reproduction and Comet Comparison"
authority: "User approved B1-B5, staged V1/E1 execution and E2 one-epoch Meta/self GPU+Comet execution on 2026-10-20; E3 10epoch remains unauthorized. E1 success is user-shared experiment evidence, not Github-observed GPU runs."
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch_at_read: dev
implementation_commit_at_read: 914e736c2699e2e129211a888d1f19013dd1dd1a
official_repository: facebookresearch/moco
official_commit: a419ef282fc9885015674e77f58c59447206a27f
source_brainstorm: ../../../../secretary/notes/brainstorm/2026-10-20-meta-moco-comet-learning-curve-and-reproduction.md
related_evidence:
  - ../experiments/2026-10-20-meta-moco-a3-synthetic-two-gpu-1025step-formal-implementation-evidence.md
  - ../experiments/2026-10-20-meta-moco-a5-pilot-real-imagenet-two-gpu-20step-evidence.md
approval: USER_APPROVED_B1_B2_B3_B4_B5_2026-10-20
b4_feature_evaluation: ADOPTED_LIGHTWEIGHT_HELDOUT_DIAGNOSTICS_KNN_OPTIONAL_2026-10-20
comet_project_name: moco-imagenet-v2-reproduction
comet_project_policy: ADOPTED_NEW_DEDICATED_PROJECT_AND_COMET_DOT_CONFIG_MANAGED_2026-10-20
comet_project_creation: NOT_VERIFIED
comet_destination_config: "repository .comet.config (project_name=moco-imagenet-v2-reproduction, user decision 2026-10-20); exported as COMET_CONFIG; API key stays in ~/.comet.config"
b2_runtime_mode: ADOPTED_STRICT_DETERMINISTIC_BENCHMARK_OFF_TF32_OFF_2026-10-20
b3_input_alignment: ADOPTED_SAME_SEED_PER_EPOCH_RANK_WITH_SAMPLED_INPUT_HASHES_2026-10-20
b3_comet_mode: ADOPTED_LIVE_RANK0_LOCAL_JSONL_FIRST_2026-10-20
b5_workers_checkpoints: ADOPTED_16_WORKERS_PER_RANK_EVERY_EPOCH_CHECKPOINT_KEPT_2026-10-20
b1_stop_rule: ADOPTED_FAIL_CLOSED_THRESHOLD_1024MIB_2026-10-20
implementation: IMPLEMENTED_CPU_VERIFIED_COMMITTED_d6f7acc_PUSHED_2026-10-20
implementation_record: ../experiments/2026-10-20-meta-moco-v2-10epoch-runner-implementation-verification.md
v1_partial_evidence: ../experiments/2026-10-20-meta-moco-v2-v1-global256-gpu-feasibility-partial.md
vram_gate: META_AND_SELF_30STEP_BOTH_PASS_2026-10-20
v1_offline_comparison: INPUT_HASH_1_OF_1_AND_LOGGED_METRICS_BITWISE_PASS_2026-10-20
e1_execution_authority: USER_APPROVED_META_AND_SELF_100STEP_WITH_COMET_ON_2026-10-20
e1_execution_status: META_SELF_100STEP_COMPLETE_INPUT_HASH_2OF2_METRICS_BITWISE_2026-10-20
e2_execution_authority: USER_APPROVED_META_AND_SELF_ONE_EPOCH_WITH_COMET_2026-10-20
e2_execution_status: META_SELF_EPOCH_COMPLETED_5004_STEPS_SAMPLED_METRICS_BITWISE_STATE_CHECK_PENDING_2026-10-20
e2_partial_evidence: ../experiments/2026-10-20-meta-moco-v2-e2-imagenet-one-epoch-pair-partial-evidence.md
e1_evidence: ../experiments/2026-10-20-meta-moco-v2-e1-imagenet-100step-comet-parity-evidence.md
comet_destination_verified: E1_META_AND_SELF_EXPERIMENT_URLS_PRESENT_2026-10-20
ten_epoch_training: NOT_RUN
comet_upload: E1_META_AND_SELF_EXPERIMENT_LINKS_REPORTED_AND_SELF_METRICS_CONFIRMED_2026-10-20
representation_evaluation: NOT_RUN
---

# Meta公式MoCo v2条件でImageNet 10epochを実行し自作とComet比較するspec（approved）

> **現行採用契約と実施済みGate（2026-10-20）**：Meta公式MoCo v2のglobal256（2GPU×local128）、初期LR0.03、FP32、K65536でImageNet-1K 10epochを最終的に比較する。B1〜B5承認済み。研究コード`dev@d6f7acc`はCPU52/52 PASS。**V1（両実装各30step）のVRAM Gate PASS**、**E1（両実装各100step）の入力hash2/2一致・記録メトリクスbitwise一致・Comet実験URL報告済み**。**E2 1epochとE3 10epochは未実行・別途実行許可が必要**。E1はpartial epochで状態hash未評価、学習品質未実証。

## 1. 背景と検証する問い

2026-10-15 MTGでは、MoCo単体の数値的mechanics確認と、「標準的な条件で学習が成立し有用な特徴が得られるか」を分離し、Meta公式をベースラインにする方針が明示された。

既存Evidence：
- A2：実ImageNet・2GPUの1step Meta対自作 formal comparisonで3,926/3,926比較エントリbitwise一致。
- A3：固定synthetic入力、2GPU、1025step×2runで58,790/58,790比較エントリbitwise一致、Queue一巡点を含む。これは**既存条件での数値等価性**。
- A5-pilot：実ImageNet・2GPU・local32/global64の20step自作benchが正常完了。**今回採用するlocal128/global256のVRAM・性能を確認した証拠ではない**。

研究上の3つの異なる主張を区別する：

1. **Implementation parity**：同じ初期状態/入力ならMetaと自作は同じ計算をするか（A3が対象）。
2. **Learning dynamics reproduction**：実ImageNetで10epochのLoss・類似度・勾配/Queue推移と結果がMetaと自作で整合するか（本specの主目的）。
3. **Representation learned**：学習後の画像特徴が意味的に有用か。**今回の10epoch本体Gateでは証明対象にしない**。固定held-out画像の特徴診断でcollapse等の異常兆候を調べるが、ImageNet kNN/Linear Probeは独立した任意評価とする（B4採用）。Lossの低下/一致や特徴分散の存在だけでは有用性を証明できない。

この実験は「Meta公式MoCo v2の**200epoch標準結果**を再現した」とは表現しない。**MoCo v2レシピの10epoch短縮run**である。

## 2. 参照公式コードと採用済みの実験パラメータ

公式固定clone `facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f`の`README.md`、`main_moco.py`、`moco/builder.py`を正本とする。公式READMEの8GPU例は`--lr 0.03 --batch-size 256`、MoCo v2向け必須フラグ`--mlp --moco-t 0.2 --aug-plus --cos`。通常の公式epochデフォルトは200であり、今回の10epochはユーザー希望による変更。

| Axis | Fixed by user + official reference | 補足 |
|---|---|---|
| Data | ImageNet-1K `/mnt/NAS-TVS872XT/dataset/ImageNet`、split=train | 過去の実測：train 1,281,167枚/1000 classes。Datasetを移動/変更しない |
| Epochs | **10** | `--epochs 10`、epoch 0..9。200epochの最初の10epochを取り出すのとは異なる |
| Model | **ResNet-50**＋MoCo v2 MLP head、128-d | 2D ImageNet backbone、非ViT |
| Queue K / EMA m / temperature T | **65536 / 0.999 / 0.2** | 公式MoCo v2レシピ |
| GPUs | **2GPU DDP**、NCCL、各GPU=RTX A4500 20GiB級 | 公式の8GPU例からGPU数だけ変更、full-batchの数値は保持 |
| Global batch | **256** | 公式`--batch-size 256`、今回は**local128 ×2**。accumulationで見かけ上の256に置き換えない |
| Optimizer | **SGD、initial LR 0.03、momentum 0.9、weight decay 1e-4、nesterov=false** | A3のglobal64/LR0.0075とは**新規別実験** |
| LR schedule | **`--cos`、horizon 10epoch** | 公式`lr(epoch)=0.03×0.5×(1+cos(pi×epoch/10))`をepoch開始時に適用。warmupの追加なし |
| Crop/augmentation | **MoCo v2 `--aug-plus`** | 224px RandomResizedCrop(.2,1), jitter(.4,.4,.4,.1) p=.8, gray p=.2, GaussianBlur(.1,2) p=.5, HFlip, ToTensor, ImageNet normalize。Q/K独立のtwo-crop |
| MoCo arithmetic | Meta公式builderに対して自作`moco.info_nce_impl=meta_einsum` | 既存A3のformal結果を参照 |
| Training precision | FP32、AMPなし（採用したA2/A3の検証精度を暫定推奨） | **cudnn.benchmark/TF32などの運用整合はBlocking B2** |
| Metadata/seed | seed0を初期候補 | hash/commit/cuda/torch/package/worker RNGを記録 |
| Comet | **新規専用project `moco-imagenet-v2-reproduction`（ユーザー採用済み）**。同一project内にMeta公式/自作の**別experiment**を作り、metric名・step/epoch軸を統一 | projectはまだ作成・存在確認していない。**送信方法と入力同期方針はBlocking B3** |

**step数の予算**：rankごとの`DistributedSampler`が`ceil(1,281,167/2)=640,584`件を扱い、`drop_last=True`、local batch128で`floor(640,584/128)=5004`step/epoch。**10epochで50,040 optimizer updates/実装**、2本で**100,080 updates**。サンプラpadding/末尾dropにより各epochですべての画像をちょうど1回学習する契約ではない。実際のsampler/loader lengthを両rankで事前検証する。

公式の`--workers`デフォルト32は`main_worker`でGPU数2に分配すると**16 worker/rank**。これも厳密公式動作への候補とするが、旧A5は8 worker/rank。NAS/CPU/workerメモリを実測し、worker変更を勝手に隠さない。

## 3. 最初の独立Gate：VRAM feasibility（全量10epochより前・必須）

**Gate V0**：まずGPU2枚の状態、CUDA/NCCL、データと出力、両repo clean commit、Meta外部shim importを検査。既存のA2/A3で検証されたのは**local32**でありlocal128は**未確認**。

**Gate V1**（新規GPU実行の別承認が必要）：
- Meta公式Mと自作S1を**別々に**、同一local128/global256・224px・ResNet50＋MLP＋K65536・FP32で、実ImageNet two-crop DataLoaderを使用し、forward → backward → SGD、EMA、Queue更新まで含めて短い実機pilotで確認。
- 最初の1〜数stepの正常性と、warmup後の少なくとも複数stepで、**GPU0/1それぞれのmax allocated/reserved VRAM、device total、データ待ち、step throughput、CUDA/NCCLエラー、発生した非有限値**を記録。2実装は独立fresh output。比較用のLoss/metricsと実験条件を残す。
- 公式`main_moco.py`を含む参照cloneは改変しない。大きい学習runより先に最低限の短いpilot専用実行経路を用意する。
- **PASS条件**：両実装が指定されたglobal256/FP32・同一モードで全pilot stepを完走、エラーなし、OOMなし、両GPUのVRAMピークが総量を超えない（実運用に足る余裕も観測し、余裕ゼロに近い場合はB1再判定）、破綻/不正Queueなし。実測のRAM/workerやGPU空きも確認。
- **BLOCKED条件**：OOM、NCCL/ドライバ異常、十分なメモリ余裕なし、NaN/Inf、重要な入力/設定不一致。**local batchの減少、AMP、解像度削減、gradient accumulation、K変更、GPU追加を自動選択しない**。発生理由と実測値を記録し、ユーザーの別の設計判断を待つ。
- いきなり10epochを開始しない。V1 PASS後に100step canary→1epoch pilot→正式10epochの条件付き順序に進む。

## 4. 最小限の専用学習経路（現行コード調査）

現在の研究コード`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、HEAD観測`914e736c2699e2e129211a888d1f19013dd1dd1a`：

- `scripts/parity/moco_imagenet_parity.py` は `MetaReferenceAdapter`（公式builderからmodel構築、公式`adjust_learning_rate`、公式checkpoint保存形式）、`SelfAdapter`（ResNet50/DistributedMoCo＋`meta_einsum`）の両方を持ち、rank0から`CometRun.log_metrics`で同じ指標を送れる。
- ただし現行`bench_loop`は`adapter.build(None)`で**初期重みが共有されず**、`sampler.set_epoch(0)`、`set_epoch_lr(0)`、`iter(loader)`を**1epoch分しか消費せず**、resume/checkpointを拒否する。`training.max_steps`を50,040等へ増やしても10epochトレーニングにはならない。
- `run_kind=parity`はcanonical fixtureとtrace保存に特化し、10epochの全step trace取得は莫大な保存/計測コストになるため転用しない。
- `scripts/parity/run_moco_gates.py`はA5-pairで**自作だけ**を対象としComet OFFを強制。ここを無断で公式10epoch比較にすり替えない。
- official importは既存shimで解決し、official source patchを不要にする。既存CIFAR/LoRA/ViT実装、A1〜A5と旧artifactには影響を与えない。

**実装提案（承認待ち）**：既存adapter、transforms、model state mapping、report/provenance/comet logger、checkpoint helperを再利用した**10epoch専用runner**を別のconfig/entrypointとして追加する。学習ループはepoch0..9、DistributedSampler.set_epoch、公式1epoch全batch、cosine LR、DDP/EMA/Queue、rank0 metrics記録、epoch checkpoint、RNG/sampler sidecar、epoch end/final manifestを実装。new CLIの正式名や責務分離は既存コードstyleに沿う。**Meta公式コード本体や元parity実験のロジックを編集しない**。

## 5. 比較条件とComet計測契約（当初候補・採用内容は後続記録を正とする）

**2実装共通で必須**：
- model / Query-Key weights / BN buffers / Queue / optimizer初期状態のcanonical mappingからの作成と検証、開始時SHA（Meta公式由来固定fixture/stateを推奨）。
- ImageFolderクラス順序、DistributedSampler per-epoch index、rank-local batch/label (labelはcontrastiveで未使用)、ランク間同期、transform pipeline/param/正規化、seed/worker init、epoch boundary、LR、データを読み込んだ総サンプル数。
- 再現性を強く主張するなら**MetaとSelfに同一の画像IDだけでなく同じ2-view画像Tensorを与える仕組み**とhash/stream alignmentを追加する。公式multi-workerの自然なランダム拡張のまま独立runする場合、同じseedだけでは画像ごとのaugmentを同一と断言できないので、Lossカーブの一致は定量的な再現傾向に留め、bitwiseの合否をA3から推定しない。
- `cudnn.benchmark=True`は公式`main_moco.py`の学習設定の一つ。既存parityはdeterministic/benchmark OFF/TF32 OFFを固定。10epochでは、公式忠実運用と厳密再現性のどちらを優先するか決め、**両実装で同条件**とする。

**Cometに比較表示する共通キー（候補）**：
- 共通横軸：`global_step`、`epoch`、optional `images_seen`、epoch内step。
- 10stepごと：`loss`（学習のmini-batch InfoNCE）、`positive_cosine_mean`、`negative_cosine_mean`、`lr`、`queue_ptr`、finiteなど。可能なら`hardest_negative_cosine_mean`。
- 100stepごと：`query_grad_norm`。full Query param cloneを必要とする`query_update_norm`はoverhead/必要性を測ってから別途選択し、全step計測しない。
- epochごと：sample-weighted mean loss、例数、global step、duration、GPU peak allocated/reserved、checkpoint SHA、再開有無。
- Comet projectは**今回の専用新規project `moco-imagenet-v2-reproduction`（ユーザー採用済み）**とし、従来のActivityNet/動画LoRA等のprojectへ混在させない。公式と自作は**同一project内の別experiment**・同一groupとし、role=`meta_reference`/`self_meta_einsum`、image resolution、epoch数、global batch、LR、K、seed、Git commit、state/input policyを明記する。グラフは2系列オーバーレイ＋差分、epoch-wise summaryを作る。metricの定義とサンプリングstepを統一する。**認証情報・workspace・project_nameの接続設定は研究サーバの`~/.comet.config`の`[comet]`に一元管理する**。`cfg.logging.comet_project=null`を既存パスで保持し、Comet SDKへ`project_name=None`を渡すことで設定を参照する。**初回送信前に設定ファイル・環境変数等を考慮した実際の宛先project/workspaceが期待値と一致するか検証し、不一致なら送信しない**。experiment名/タグ/科学メタデータはコード側で管理する。projectの作成・所有workspace・権限は実環境で確認する。
- **local artifactがSSOT**：各runの軽量history（JSONL等）、checkpoint、epoch summary/manifestを先に原子保存。Comet通信の失敗でGPU学習を破綻させず、イベントidとsource hashを使って再送/二重送信を抑える。credential/API keyはログ/Repo/Comet parameterに記録しない。
- Cometライブ送信 vsローカルに保存して後からアップロードの運用選択はユーザー承認待ち。比較可視化とscientific Gateを混同しない。

## 6. Gate順序・成功条件の候補

| Gate | 範囲（順番） | 合格／記録したいEvidence |
|---|---|---|
| E0 | 新runnerとCPU testだけのengineering検証 | epoch0..9のsampler/LR、DataLoader/augmentation/mappingの条件テスト、checkpoint/resume再現、既存CIFAR/parity非回帰、独立出力/Comet送信mock。**GPU実験ではない** |
| **V1** | **Meta＋Self local128/global256実画像数step VRAM feasibility（必須）** | 両rankピークVRAM・OOM/NCCL・step有限・Queue正常、条件一意。FAIL/BLOCKEDならSTOP |
| E1 | 短いpaired canary（例100step） | 画像順序・初期state・augmentation policyの一致とLoss/grad/Queueを比較、Comet logging/ローカル保存試験。**数値一致の合否を設定に応じて定義** |
| E2 | 1epoch pilot（2run） | sampler epoch進行・1epoch全batch・checkpoint/resume、実データ負荷、GPU/CPU/ディスク・時間再見積り。修正は明示的な設定改訂のみ |
| **E3** | **Meta公式、Self各10epoch**（独立fresh run） | 10epoch完了（各50,040stepを目安）、同一recipe、epoch LR、Loss/pos/neg/grad/Queueの履歴、NaNなし、epoch checkpoint+SHA、Comet series・差分表示、失敗時はartifact保全。**学習曲線の類似性だけでbitwise成功としない** |
| E4 | **軽量held-out特徴診断（B4採用済み）** | ImageNet valから固定した少数の画像をラベルを使わず選び、**学習前・公式10epoch後・自作10epoch後**の凍結ResNet50 backbone特徴を共通の前処理/eval modeで測る。**Feature variance・feature norm・異なる画像間cosine similarity分布**の有限性とcollapse兆候を記録し、公式/自作を比較する。**kNN/Linear Probeは今回の必須Gateには含めない**。数値しきい値とkNN精度による成功判定は設けない |

**区別する判定**：`engineering_pass`、`vram_feasible`、`training_completed`、`curve_agreement`、`lightweight_feature_diagnostics_completed`（異常兆候を含む観測）を別々に保存し、一つのPASSへ無条件にまとめない。`representation_quality_evaluated`は別途kNN/Linear Probe等の評価を実施した場合だけ記録する。

**結果をどのように解釈するか**：
- 同じ入力/初期値で両者が近い曲線を示しても、実画像10epochで学習に十分な特徴を獲得したとは限らない。E4の軽量診断はcollapseや数値異常の兆候を確認するだけで、意味的特徴の有用性を証明しない。
- 10epoch後にLossが単調減少しないことだけで失敗扱いしない（negative分布・Queueの非定常性・augmentationが影響）。
- 公式の200epoch/Linear Probe Accuracy（MoCo v2の論文レベルの数字）は**10epochの合格基準に用いない**。
- 比較で差が出たら、同一state/augmentation順序、worker seed、GPU演算モード、optimizer、BN/Shuffle、Cometでのサンプリング時点を切り分ける。

## 7. 再現性・artifact・安全境界

- 適用repo/commitを実装時にGitHub current stateから再取得し、clean codeで固定。固定Meta cloneを改変しない。SSH worktree/dirty、ImageNet/NAS、GPU実体はGitHubから検証不能。実機開始時のcheckが必要。
- 各実装のoutputは同じcampaign root下の**別ディレクトリ**、既存artifactを上書きしない。設定、CLI、環境、split/count/classes、source commit、dataset transform/seed、sampler and epoch indices digest、initial state/hash、worker RNG policy、run ID、checkpoint SHA、Comet experiment IDを保存する。
- 10epochの途中失敗や再開はnative format（Meta:公式形式、Self:既存形式）とrankごとのRNG/sampler sidecarを使い、resume後のepoch・global step・DataLoader位置・LR・Queueの一致を試験してから正式運用。
- 期間が長いためNAS cache/worker wait、GPU使用・温度、ログ容量、checkpoint retentionをpilotで評価。tmux/session、長時間スケジュール、停止/再開の責務を明確にする。
- **No unauthorized action**：このspecの保存をもってコード実装/テストrun/GPU実験/Comet通信許可としない。V1 pilot、E1、E2、E3はそれぞれ別の実行合意が必要。

## 8. Ambiguity Gate（当初の提案履歴・採用記録を優先）

**以下の表は当初の提案として保存している歴史的記述です。2026-10-20の採用決定は末尾の「B2・B3残部・B5採用記録」と「V1 GPU短時間検証への進行承認」、およびfrontmatterを正とします。B1〜B5の承認は現在完了しています。**

| ID | Blocking question | 推奨する選択と理由 |
|---|---|---|
| **B1** | **V1でlocal128がOOMまたはVRAM安全余裕不足になった場合** | 公式Batch256/LR0.03を維持してSTOP。追加GPU等の別条件を提案、**global64やAMPへ自動降格しない**（今回のユーザー採用を守る）。この停止規則は提案で、V1実測までfeasibility自体は不明 |
| **B2** | **公式の`cudnn.benchmark=True`などの実行モードとA3の厳密再現設定のどちらを優先するか** | 公式条件忠実を第一候補に両runで揃え、modeをprovenanceに保存。bitwiseはA3 formal Gateの既存証拠として別立て。ただし再現性優先なら両方benchmark OFF等で固定する代替案 |
| **B3（一部採用済み）** | **入力再生とComet記録方式** | **Comet保存先はユーザー決定済み：新規project `moco-imagenet-v2-reproduction`、同一project内でMeta/Selfを別experiment。**残りは**公式M/Selfを同一state＋同じImageNetサンプル順＋同じ2-view拡張で比べる**方法、ライブ送信か再送可能な事後アップロードか、入力再生コストの判断。非同期/事後アップロードで原runを壊さず、2系列・epoch/step・source hashを残す方式を推奨。augmentationの再生は公式transform分布を変えない実装で検証する |
| **B4（採用済み／非blocking）** | **本研究での特徴評価の範囲** | **ユーザーが新推奨を採用：**固定少数のImageNet val画像で、学習前・公式10epoch後・自作10epoch後のResNet50 backbone特徴のvariance、norm、画像間cosine分布を**軽量診断**する。画像ラベルを使う正式なkNN/Linear Probeは**任意の別Gateとして延期**。分類能力・意味的な特徴品質の獲得を今回の10epoch結果の必須の主張にしない |
| **B5** | **worker/Checkpoint/失敗時の運用細部** | 公式`--workers 32`相当（2GPU各16）を第一候補にし、VRAMだけでなくCPU/RAM/NASを測る。epoch毎native checkpoint＋rank sidecar、差分のないresume試験、失敗時はfresh outputを維持。保存上限・Comet命名/タグは実装前に固定 |

`non-blocking`：CLI file名、local JSONL file名、Comet display nameの表記、既存helperの呼び出し構造等は既存repo規約に従う。ただしmetricキー、同一step同期、checkpoint/experiment identityを曖昧にしない。

**Spec readiness: approved（2026-10-20）**。B1：VRAM安全余裕1024MiB（各GPU）・OOM/異常時STOP・無断降格なし。B2：strict cuDNN/TF32。B3：同seed/入力hash＋Cometライブ（V1ではOFF）。B4：軽量特徴診断・kNN任意。B5：16 workers/rank＋epoch checkpoint。**blockingな設計判断なし**。ただし実機V1は未検証であり、E1/E2/E3は先行Gate PASSと別許可が必要。

## 9. Handoff（approved後に使用）

- **先に実装するのは**専用epoch-aware runner + config + minimal tests + Comet/local logging adapter + local128 VRAM short pilot gate。既存公式cloneは非改変。
- **非対象**：A3 parityハーネスの演算変更、既存A5-pairの設計変更、ViT/LoRA/Sequential Loader、Meta公式200epoch checkpoint精度の再現主張。
- **必要な検証**：CPU deterministic canary/epoch schedule/mapping/resume + 既存非回帰。短いGPU V1は後日別許可。
- **大規模実験の開始条件**：V1 PASS、E1/E2を段階的PASS、同条件+clean immutable commit、資源見積り承認。10epochの2run、Comet送信は**NOT_RUN**。
- **結果の報告**：公式/自作の条件差がないこと、全epoch history、Comet URL/experiment keys、Loss/cosine差、GPU/CPU/NAS/VRAM、学習前・両学習後の**固定held-out軽量特徴診断**、確立したclaim/未確立のclaimを分ける。kNN/Linear Probeの未実施は隠さず明記する。

### B4採用記録（2026-10-20）

ユーザー指示「新しい推奨案を採用します」を、直前の**B4：ImageNetの本格kNNを必須から外し、固定held-out画像による軽量特徴量診断を行う**という提案への明示承認として反映した。**この承認の対象はB4だけ**であり、B1/B2/B3/B5の承認、spec全体の`approved`化、コード変更、GPU実験、Comet送信の許可ではない。

E4の再現性契約：同じ固定val画像index・同じ前処理・同じ2048次元backbone特徴位置・同じnormalization方針で、同一初期stateとMeta公式/自作の10epochモデルを比較する。測定対象は有限性、サンプル間のvariance、L2 norm分布、非同一画像対のcosine similarity分布。可能なら有効次元の補助診断も記録するが、測定が重い場合は必須化しない。ImageNetのvalidationクラスラベルを指標計算に使用しない。run reportに評価画像のindex/前処理/hash、checkpoint lineage、集計定義、結果を保存し、異常兆候・確立しない主張を明示する。固定サンプル数や実装上のファイル名はengineering review時に事前決定して記録する。

### Comet専用project採用記録（2026-10-20）

ユーザーは「既存のComet projectを使わず新規に作成する」案を採用。正式な**project name**を`moco-imagenet-v2-reproduction`と固定する。Meta公式と自作MoCoを別々のExperimentとして同じprojectへ送信し、比較画面ではLoss・Positive/Negative cosine・Grad等を同一step/epoch軸で重ねる。

今回の決定は**送信先の研究仕様**であり、Cometサービスでprojectを実際に作成したという証拠ではない。Comet workspace/権限とproject存在は初回送信前に確認し、存在しない場合は適切なworkspaceに新規作成する。**Comet APIキー、workspace、project_nameは研究サーバの`~/.comet.config`で管理し、認証情報をコード・GitHubへ置かない。**既存コードの`logging.comet_project=null`を尊重し、Comet SDK設定から宛先を取得する。送信前に`project_name=moco-imagenet-v2-reproduction`かつ意図したworkspaceかを検証し、相違・設定欠落ならfail-closedで送信しない。全体のhome default設定の変更が既存実験に及ぼす影響も確認する。**B3の共通入力再生方式・ライブ送信か事後登録かは未承認のまま**。本specは引き続き`draft`。

### .comet.config中心の管理方針（2026-10-20、ユーザーの運用認識を反映）

ユーザー確認：「コメットは.comet.configで管理する想定ですがその認識であってますか？」。Comet公式SDKの設定仕様と研究コード`conf/moco_imagenet_parity.yaml`、`scripts/parity/moco_imagenet_parity.py`を照合。`[comet] api_key / workspace / project_name`は研究サーバの`~/.comet.config`で管理可能。`logging.comet_project: null`で`Experiment(project_name=None)`に到達する既存実装はSDKの設定を参照する。本研究の採用project名は`moco-imagenet-v2-reproduction`。

- 実験間で変えない接続先・秘密情報：`~/.comet.config`（GitHub非追跡）。API keyの値をrun reportやChatGPTへ表示しない。
- 実験ごとに変わる実験名、tags、run provenance、metricsのキー/step定義：研究コードの設定とComet logへ記録。
- **preflight必須**：存在・認証・Cometの実際のworkspace/project識別子を検証し、環境変数やコードの明示overrideで別projectへ誤送信されないよう停止。ユーザー指定projectの存在/作成は未確認。
- **状態境界**：接続方式の整理は`B3`の入力stream完全一致やオンライン/事後送信方式の決定ではない。`status=draft`、未承認B1/B2/B3残部/B5、GPU/Comet run未実行を維持。

Comet SDK公式ドキュメント：`https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/`。

### B2・B3残部・B5採用記録と実装（2026-10-20）

ユーザーは本specに従った実装を依頼し（「このスペックに従って実装してください」）、実装前の確認で次の推奨案を選択した。**この選択の対象はB2・B3残部・B5であり、GPU実行（V1/E1/E2/E3）とComet送信の実行許可ではない。**

- **B2（実行モード）**：厳密再現モード。両実装とも`cudnn.deterministic=True`・`cudnn.benchmark=False`・TF32 off（`fp32_precision=ieee`）で、A2/A3と同じ設定。公式`main_moco.py`の`main_worker`（benchmark=True）とは異なるため、provenanceに記録する。
- **B3（入力）**：同一seedで入力を揃える。epoch・rankごとに、DataLoader専用generator（workerのbase seed）、CPU global RNG（shuffle-BNのrandperm）、Python `random`を`derive_seed(seed, 名前, epoch, rank)`で固定する。DistributedSamplerは公式と同じseed0とset_epochを使う。拡張の分布は公式のままで、worker数が同じならMetaと自作が同じ2-view Tensorを受け取る。1step目と100stepごとに全rankの入力sha256を記録し、E1で照合する。
- **B3（Comet）**：ライブ送信。rank0がローカルJSONLを先に書いてから送る。Cometの障害では学習を止めず、記録して送信だけを停止する。resumeは親runと同じexperimentを継続するため、クラッシュ直前のstepが重複して表示されることがある。事後再送ツールは作らない。
- **B5**：16 worker/rank（公式`--workers 32`を2GPUで分割）。epoch終了ごとにnative checkpointとrank sidecarを保存し、全epochを保持する。

**B1採用履歴の更新（2026-10-20）**：前段の実装時点では未確認だったが、CPU52/52 PASSの報告と「B1の安全余裕を1024MiBで採用してV1（各30step）へ進めてよいか」の明示質問に対して、ユーザーは「次に進みます」と回答。**B1の閾値1024MiB・失敗時STOP/無断降格なしと、V1 Meta/Self各30stepの実行だけ採用・許可**。headroomはコード上「総量−当該プロセスのpeak reserved」であり、他ユーザーのGPU使用量を含まないため実行前の`nvidia-smi`確認が必要。

**Comet宛先（最終形）**：研究repo直下の追跡済み`.comet.config`は、SDKの優先順位で`~/.comet.config`より強い。そのため、「`~/.comet.config`だけで宛先を管理する」という本specの想定どおりでは新projectへ送信されない。当初はユーザー提示のテンプレート形式の専用ファイルを`COMET_CONFIG`で指定する設計で実装した。しかしユーザーのcommit `3cade3d`で直下`.comet.config`の`project_name`が`moco-imagenet-v2-reproduction`へ変更され、専用ファイルとrunner用yamlは含まれなかった。確認のうえ、ユーザーは**直下`.comet.config`で管理する**方式を選択した（2026-10-20）。runner（`logging.comet_config: .comet.config`）は、このファイルを`COMET_CONFIG`として指定する。**この設定が続く間は、このrepoから明示的なproject指定なしで起動する他のComet実験（`train.py`、既存parity runなど）も`moco-imagenet-v2-reproduction`へ送られる**（§5の「既存projectと混在させない」と緊張関係にあり、ユーザー判断として記録する）。APIキーは引き続き`~/.comet.config`で管理する。送信前のpreflightでは、環境変数・`.env`を含めてSDKが解決する宛先が期待値と一致すること、APIキーが解決されることを確認する。一致しなければ学習もせずにfail-closedとする。オンライン確認では、workspaceへの到達性とprojectの事前存在も記録する。

実装は`dev`の`3cade3d`（ユーザーcommit）と`d6f7acc`（yamlの復元とComet宛先の修正）としてorigin/devへpush済み。実装と検証の詳細は[実装・CPU検証記録](../experiments/2026-10-20-meta-moco-v2-10epoch-runner-implementation-verification.md)を参照。V1/E1/E2/E3・Comet送信・Comet project作成はいずれも**NOT_RUN**。E4軽量特徴診断の実装は今回の範囲外（§9 Handoffの「先に実装するもの」に含まれない）。


### V1 GPU短時間検証への進行承認（2026-10-20）

- ユーザー指示：「次に進みます」。直前のB1閾値1024MiBとV1 30step×2runの具体的な確認質問に対する同意。
- 採用設定：ImageNet train、ResNet50+MLP128、K65536、m0.999/T0.2、2GPU×local128/global256、SGD LR0.03/mom0.9/wd1e-4、FP32、strict deterministic、16 workers/rank、V1は`gate=V1 training.max_steps=30 logging.comet_enabled=false`。初期state fixtureをMeta公式と自作に共通利用する。
- V1の合格判断：Meta/Selfそれぞれ予定30stepを完了、loss有限でQueue/EMA動作に異常なし、両rankで`total_memory_mib − max_reserved_mib >= 1024`。`nvidia-smi`で他GPUプロセスの占有を事前確認する。OOM/NCCL異常/余裕不足ではSTOPし、BatchやAMPは変更しない。
- V1後に両runの`run_report.json`からstatus・training_completed・vram_gate.status・per-rank headroom・GPUピーク・throughput・data waitを確認する。1件でもBLOCKEDなら停止。E1 100step、E2 1epoch、E3 10epoch、Comet実送信はV1の許可に含めない。
- 事前の実機CPU testはユーザー提示ログで52/52 PASS、GitHub code `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` `d6f7acc75ee92adcf7e6e3e65831d46759b9ed14`がclean。今回のGitHubでの作業はspecのみで、GPU job・Comet・研究コード変更は行わない。
- **`approved`は仕様承認であって、V1/E1/E2/E3や表現品質が実証されたことを意味しない**。E4軽量特徴診断は**未実装**で、別の評価Gateとする。


### V1途中結果：Meta公式のみ30step PASS、自作はpreflight停止（2026-10-20）

ユーザーが実行したcampaign `moco_v2_v1_20261020_162206`のMeta公式V1は30/30step、両rankのpeak reserved 13026.0MiB、headroom 7069.875MiB、`vram_gate.status=PASS`。しかし自作のV1はMeta cloneに生成された未追跡Python `.pyc` 2件が原因で`repository_preflight`が停止、学習は未開始。**V1全体はINCOMPLETE、E1/E2/E3へ進めない**。

また、Meta公式の起動コマンドで本specの`logging.comet_enabled=false`が指定されておらず、YAML defaultは`true`。Cometの実送信有無・URLは未確認。厳密にComet OFFのV1ペアを揃える場合は、初回Metaのresource PASSを残しつつMeta OFFの別runを取り直す必要がある。追加の証拠と安全な対応方法は[partial V1実験記録](../experiments/2026-10-20-meta-moco-v2-v1-global256-gpu-feasibility-partial.md)を参照。

この更新は**Gate状態・Evidenceリンクのみ**であり、approvedなB1〜B5契約・研究コード・10epoch実験の許可範囲は変更しない。

### V1 pair合格・元のMeta Comet懸念解消（2026-10-20）

直前の「V1途中結果」は履歴として維持し、本追記を**最新のGate判定**とする。ユーザーがSSH研究サーバから`run_report.json`の集約情報とself V1学習ログを再提示したため、**MetaとSelfの30step両方PASS**を確認した。

- campaign: `moco_v2_v1_20261020_162206`、同一fixture、FP32、2GPU×local128/global256、K65536、30step。
- Meta: status completed / training_completed true / `vram_gate.status=PASS` / peak reserved 13026MiB（rank0/1）/ headroom 7069.875MiB（rank0/1）。
- Self: status completed / training_completed true / `vram_gate.status=PASS` / peak reserved 13038MiB（rank0/1）/ headroom 7057.875MiB（rank0/1）。
- **双方ともComet preflight enabled=false、experiment created=false、error count0**。前段の「Meta Comet ON可能性」は実reportにより否定され、同一V1 OFF契約を満たす。
- 両実装のstep1/10/20/30でLoss等の出力値は一致。**全step入力一致・full state bitwiseではなく、観測したscalar一致の主張のみ**。offline compareのinput hash判定は別途必要。
- Data waitがMeta側約75秒/rank、Self側約5秒/rankと大差。性能比較の主張は保留。
- 正本Evidence:[V1両実装PASSの実験記録](../experiments/2026-10-20-meta-moco-v2-v1-global256-gpu-feasibility-partial.md)。この記録は研究サーバの現物をGitHub connectorで取得したものではなく、ユーザーが提示したログ由来。
- **V1 PASSED only**。E1 100step・E2 1epoch・E3 10epoch・Comet real send・E4軽量特徴診断は未実施。次のGPU実験実行には別の明示許可が必要。

### V1 offline comparisonの採用Evidence追記（2026-10-20 16:44 JST）

同一campaign・同一初期fixtureでMeta/Selfそれぞれ30step、GPU V1が両者PASSした後、ユーザーが研究サーバで`compare_moco_10epoch.py`を実行。比較器の`contract_valid=True`、`both_completed=True`、`inputs_identical=True`（**ハッシュした1stepのみ一致**）、`metrics_bitwise_equal=True`、`curve_agreement=bitwise`、`contract_issues=[]`を確認。Loss/positive-negative-hardest cosine/LR/queue_ptr/finiteは各4/4step、query_grad_normは2/2stepでscalar値が完全一致（max_abs_diff=0）。ただし`epoch_states_identical=None`で**終了epochのstate比較は未実施**。比較器の`bitwise`表示はfull stepのモデル全状態bitwise一致を示さない。V1は**計算資源+サンプル学習指標の再現性Gate PASS**であり、正式E1/E2/E3/E4の代替ではない。

研究サーバの出力：`log/moco_imagenet_10epoch/moco_v2_v1_20261020_162206/comparisons/v1_20261020_164421/comparison_report.json`（GitHub connectorでは実体未確認）。[V1 Evidence](../experiments/2026-10-20-meta-moco-v2-v1-global256-gpu-feasibility-partial.md)参照。**E1の100step×2run、Cometライブ送信はV1実行許可に含まれない**ため、先行Gate結果を受けた別のユーザー実行合意を必要とする。

### E1（各100step）実行承認と実行境界（2026-10-20）

ユーザーはV1の比較器による結果（Meta/Self両実装completed、契約valid、記録指標bitwise一致、入力hash 1/1一致、epoch state未評価）の報告後、**「次に進みたいです」**と回答。直前の明示質問「E1としてMeta公式・自作それぞれ100stepのGPU学習とCometへの記録を開始してよいか」に対する回答であるため、**E1 GPU×2runおよび当該2runのCometライブ送信のみを承認**した。コード変更やE2 1epoch/E3 10epochへの実行許可を含まない。

**共通固定条件**：新規campaignを起こし、同campaign内にMeta公式builderから生成した単一のcanonical初期state fixtureを置く。両runnerに`gate=E1 training.max_steps=100 logging.comet_enabled=true`と同一の`initial_state.fixture_path`を指定する。その他の採用recipe（ImageNet train、ResNet50 MoCo v2 MLP、2GPU×local128/global256、SGD LR0.03、FP32 strict、K65536、MoCo momentum0.999/temperature0.2、16worker/rank、seed0）は変更しない。V1のfixtureを改変/上書きしない。E1はV1からのresumeではなく**fresh initial state**から開始する。

**安全Gate**：研究コード`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`のcheckoutがcleanでcommit固定、公式clone`facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f`もclean、GPU0/1の他job有無、NAS/ImageNet存在、`~/.comet.config`等の秘密情報・認証をユーザー環境で確認。`PYTHONDONTWRITEBYTECODE=1`で前段の未追跡`.pyc`生成を抑止し、dirtyなら開始しない。Comet SDK preflightはrepository `.comet.config`とホームAPI keyから`haruto2919/moco-imagenet-v2-reproduction`へ届くことを確認する。Project未存在の場合はオンラインSDKの自動作成・権限結果を確認する。API keyは一切出力・commitしない。

**実行順**：Meta公式100step → status/steps/Comet experiment key・URL・errorsを確認 → 正常なら自作100step → 両方完了時にoffline comparatorで同一campaign、sampler+fixture contract、1step/100step input SHAと10step毎のLoss/Cosine/LR/Queueおよび1/100stepのgradを比較。各runはHydraの独立日時付きdirectoryに保存し、JSONL/ArtifactはローカルSSOTとする。プロセスのエラー、Comet API接続不可、誤送信可能性があれば続行/再試行前に停止・調査する。

**成功・報告**：Meta/Selfの`run_report.json`から`status=completed`・`training_completed=true`・100/100step・失敗なし、`preflight.comet.enabled=true`、意図したworkspace/project、Comet実験の`key/url`と`errors`、CPU/NAS data wait、GPU peakを確認。比較器は`contract_valid/both_completed/inputs_identical/metrics_bitwise_equal/curve_agreement/contract_issues`を報告する。入力hashは1stepと100stepの**計2箇所**であり、100stepすべてのtensor一致ではない。100stepはepoch未完了なので`epoch_states_identical=None`、checkpoint/state full bitwiseは**未検証**。Queue K65536は100stepで一巡せず、V1/E1のLoss曲線は特徴品質を示さない。

**実行状況**：この決定記録の保存はGPU jobやCometへの実送信ではない。ユーザー研究サーバ側で起動して結果を返す。E2/E3/E4・コード変更・full runは未承認／未実行のまま。

### E1実測結果の採用Evidence（2026-10-20、ユーザー添付ログ）

ユーザーがSSHサーバからE1の自作100stepコンソールと比較器出力を提示したため、**E1完了状態**へ更新。実験root：`log/moco_imagenet_10epoch/moco_v2_e1_20261020_165430`、比較report：`comparisons/e1_20261020_171802/comparison_report.json`。GitHubからサーバ上のraw artifactsを直接確認したものではない。

- 両run `completed`、`contract_valid=true`、`contract_issues=[]`。同一config/初期状態条件で100step完走。
- sampled input hash **2/2一致**、loss・各cosine・lr・queue_ptr・finiteが**各11/11完全一致**（max_abs_diff=0）、query_grad_normが**2/2完全一致**。比較器 `metrics_bitwise_equal=True,curve_agreement=bitwise`。
- `epoch_states_identical=None`。これは100stepがepoch途中であるためで、**epoch checkpoint/canonical state未比較**。全100stepの入力全テンソルや学習state全体のbitwise一致と混同しない。
- Comet：Meta公式 `https://www.comet.com/haruto2919/moco-imagenet-v2-reproduction/1f99cdba1f74470391f804121a8d5db2`、自作 `https://www.comet.com/haruto2919/moco-imagenet-v2-reproduction/54fefa983b55467398fc79e84c3404a1`。自作はSDK終了Summaryで明示記録され、公式は比較reportのURLとして観測。UI上での同時曲線表示は未確認。
- Lossはstep1 6.441793→step100 9.806382だが、Queue置換途中・短時間runのため特徴学習成功/失敗の主張には使わない。
- [E1原本Evidence](../experiments/2026-10-20-meta-moco-v2-e1-imagenet-100step-comet-parity-evidence.md)参照。
- **本更新でE2/E3の実行許可は生じない**。E2では1epoch完走・epoch checkpoint・canon state SHA一致・データ待ち/長期安定性を判断する。E2の実行には別のユーザー承認を要する。E3 10epochはさらに別Gate。

### E2（Meta公式・自作それぞれ1epoch）個別実行承認（2026-10-20）

ユーザーはE1結果（100step双方完了、入力hash 2/2地点、記録scalar bitwise、Comet両URLあり）の報告と直前の質問「E2としてMeta公式・自作それぞれ1epochの実験へ進めてよいか」に対し、**「次に進めます」**と回答した。よって**E2 GPU×2本＋同runのComet送信を許可**、**E3 10epoch・E4評価・研究コード改変は許可しない**。

- **実験recipeは固定**：ImageNet-1K train / ResNet50 MoCo v2 MLP / world_size2 / local_batch128（global256）/ SGD LR0.03,momentum0.9,weight_decay0.0001 / MoCo K65536,m0.999,T0.2 / FP32 strict deterministic, TF32 off / 16 workers/rank / seed0 / 10epoch cosine horizon。CLIは `gate=E2 training.stop_after_epoch=1 training.max_steps=null training.resume=null logging.comet_enabled=true logging.online_check=true`。**training.epochs=10のまま**1epochで止める。E1のrun/fixtureを再利用しない。
- 単一の**新規**初期state fixture（`build_moco_parity_fixture fixture.kind=synthetic training.local_batch_size=128`で公式builder由来の初期重みのみ生成）を新キャンペーン `log/moco_imagenet_10epoch/moco_v2_e2_<timestamp>/initial_state/fixture.pt`に固定。実学習のImageNet画像はrunner DataLoaderから読み込む。公式と自作に**同じfixture**を指定する。
- **preflight**：研究コード `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` が公式の現行commit `d6f7acc75ee92adcf7e6e3e65831d46759b9ed14` でclean、Meta公式clone `a419ef282fc9885015674e77f58c59447206a27f` clean、`PYTHONDONTWRITEBYTECODE=1`、GPU0/1に競合処理がないこと、NAS train 1000-class/1,281,167 sample、`df`容量、Comet専用project preflightを確認。キャッシュdirty、OOM、コミット違い、Comet到達不可、Checkpoint保存失敗時は他の実装を起動しない。設定を黙って変更しない。
- **実行順**：Meta公式1epoch → `run_report.json` status, full epoch, `checkpoint_epoch001.pt`・各rank sidecar、canonical state SHA、Comet key/URL/errors、peak memoryを確認 → 自作1epoch → offline comparator。記録は各runner固有の日時run dirへ保存しE1/V1は上書きしない。長時間接続切断に備えてSSH上でtmuxなどの運用を推奨（未導入なら強制しない）。
- 現行`train_epochs()`は `stop_after_epoch=1,max_steps=null`でImageNet1epoch全stepを処理し、完了時のみ`checkpoint_epoch001.pt`と`canonical_state_sha256`を`epochs.jsonl`へ保存する。`compare_moco_10epoch.py`で`epoch_states_identical=True`を得るには双方のcanonical state SHA一致が必要。**CheckpointファイルSHAは実装固有の書式・metadataがあるため、ファイルバイト列同士が等しいことを条件にしない**。オプティマイザ・Queue・RNGの内部項目はこの比較器の包括的bitwise確認範囲に含まれない。
- dataset=1,281,167、DistributedSampler world2→rank640,584、DataLoader`drop_last=True`/local batch128により**5004step/epoch（想定）**。実数はrun_report内の`data.steps_per_epoch`を正とする。Hashを記録するStep1と以後100stepごと（5000まで）を比較、Loss等はStep1と10stepごと/lastを比較。サンプリング比較であり5004step全入力bitwise保証ではない。
- **E2成果判定**：Meta/self両方`status=completed`、`last_global_step=data.steps_per_epoch`、`completed_epochs=[0]`、epochs[0]`completed=true`、checkpoint実在、rank sidecar各2、canonical state SHA non-null、comet key/URL、errors空。比較で`contract_valid=True,both_completed=True,inputs_identical=True,metrics_bitwise_equal=True,epoch_states_identical=True,curve_agreement=bitwise`が理想値。差分が出た場合は具体的な最初のstep、metric、epoch状態、入力hashを調べてE3には進まない。
- **Evidenceの境界**：E2で同一初期条件の1epoch再現が確立してもMoCoの学習済み特徴品質やMoCo v2標準200epochの再現は確立しない。E3 10epochの実行は新しいユーザー承認を必要とする。
- **今回のGitHub操作**はResearch Workspaceに承認を保存するだけ。実際のGPU run、Comet送信、SSH local checkoutの更新はChatGPTから実施していない。

### E2 1epoch比較ログの途中Evidence（2026-10-20、ユーザー提示）

Meta公式・自作の比較器出力で`curve_agreement=bitwise`、`contract_issues=[]`、入力ハッシュ`51/51`一致、Loss・Positive/Negative/Hardest Cosine・LR・Queue Pointer・finiteはそれぞれ`502/502`一致（max_abs_diff=0.0）、Query Grad Norm`52/52`一致、epoch0は両者`completed=true,global_step=5004`、mean loss差0.0。

ただし貼付が末尾で途切れ、**`epochs[0].canonical_state_equal`および`verdicts.epoch_states_identical`は未確認**。比較器はepoch SHA未記録時にも`curve_agreement=bitwise`を表示し得るため、現時点での承認済みE2全体の正式判定は**保留**する。`run_report.json`とCheckpoint/sidecar/Comet状態も必要に応じて照合。原本: [E2暫定実験結果](../experiments/2026-10-20-meta-moco-v2-e2-imagenet-one-epoch-pair-partial-evidence.md)。E3 10epoch実行は未承認であり、本追記はE3実行を許可しない。
