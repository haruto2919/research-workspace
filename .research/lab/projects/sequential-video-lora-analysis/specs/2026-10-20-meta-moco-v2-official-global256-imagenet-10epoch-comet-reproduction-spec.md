---
project: sequential-video-lora-analysis
spec_type: experiment-and-engineering
status: approved
created: 2026-10-20
last_updated: 2026-10-20
title: "Meta MoCo v2 Official Global Batch 256 ImageNet-1K 10-Epoch Reproduction and Comet Comparison"
authority: "User adopted official-example global batch 256 / 2 GPUs x local 128 / initial LR 0.03, mandatory VRAM feasibility, ImageNet 10 epochs, Meta vs self Comet loss comparison on 2026-10-20. Remaining choices are proposed, not approved."
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
vram_gate: NOT_RUN
ten_epoch_training: NOT_RUN
comet_upload: NOT_RUN
representation_evaluation: NOT_RUN
---

# Meta公式MoCo v2条件でImageNet 10epochを実行し自作とComet比較するspec（approved）

> **現行承認済み契約（2026-10-20）**：Meta公式MoCo v2のGlobal Batch256・初期LR0.03を2GPU・local128/GPUで維持し、Meta公式と自作を各10epoch学習してComet比較する。**先行V1 VRAM feasibility必須**。B1〜B5を採用済み（B4のkNNは任意）。実装は`dev@d6f7acc`、CPU test 52/52 PASS。**今回許可されたGPU実行はV1（30step×2run）のみ**。E1/E2/E3/Comet送信には別途実行合意が必要。V1の実機結果は未確認。

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

## 5. 比較条件とComet計測契約（Blocking B3で最終選択）

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

## 8. Ambiguity Gate（未承認の推奨条件）

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
- **`approved`は仕様承認であって、V1/E1/E2/E3や表現品質が実証されたことを意味しない**。実装済みE4軽量特徴診断は別扱い。
