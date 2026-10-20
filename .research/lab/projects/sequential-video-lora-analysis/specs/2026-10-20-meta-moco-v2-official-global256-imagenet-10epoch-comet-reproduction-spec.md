---
project: sequential-video-lora-analysis
spec_type: experiment-and-engineering
status: draft
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
approval: PENDING_REVIEW
implementation: NOT_STARTED
vram_gate: NOT_RUN
ten_epoch_training: NOT_RUN
comet_upload: NOT_RUN
representation_evaluation: NOT_RUN
---

# Meta公式MoCo v2条件でImageNet 10epochを実行し自作とComet比較するspec（draft）

> **今回採用されたこと**：公式MoCo v2の「Global Batch 256／初期LR 0.03」の8GPU例に含まれる数値設定を、利用可能なRTX A4500 **2GPU・local batch 128/GPU**で維持し、Meta公式と自作の**それぞれ10epoch**をImageNet-1Kで行い、Comet上のLoss等を比較する。**VRAM feasibilityは必須の事前Gate**。現在のcommitにこの専用10epoch経路はなく、研究コード編集・GPU実験・Comet送信は**このspec作成時点では一切行わない**。未解決の設計判断があるためstatusは`draft`で、approved/implementedではない。

## 1. 背景と検証する問い

2026-10-15 MTGでは、MoCo単体の数値的mechanics確認と、「標準的な条件で学習が成立し有用な特徴が得られるか」を分離し、Meta公式をベースラインにする方針が明示された。

既存Evidence：
- A2：実ImageNet・2GPUの1step Meta対自作 formal comparisonで3,926/3,926比較エントリbitwise一致。
- A3：固定synthetic入力、2GPU、1025step×2runで58,790/58,790比較エントリbitwise一致、Queue一巡点を含む。これは**既存条件での数値等価性**。
- A5-pilot：実ImageNet・2GPU・local32/global64の20step自作benchが正常完了。**今回採用するlocal128/global256のVRAM・性能を確認した証拠ではない**。

研究上の3つの異なる主張を区別する：

1. **Implementation parity**：同じ初期状態/入力ならMetaと自作は同じ計算をするか（A3が対象）。
2. **Learning dynamics reproduction**：実ImageNetで10epochのLoss・類似度・勾配/Queue推移と結果がMetaと自作で整合するか（本specの主目的）。
3. **Representation learned**：学習後の画像特徴が初期値より有用か（最低限のheld-out評価を別のEvidenceとして必要。Lossの低下/一致だけでは証明できない）。

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
| Comet | 同一Comet projectで公式/自作を別experiment、metric名/step・epoch軸を統一 | **送信方法と同期方針はBlocking B3** |

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
- 公式と自作は**別experiment**で同一project・同一groupとし、role=`meta_reference`/`self_meta_einsum`、image resolution、epoch数、global batch、LR、K、seed、Git commit、state/input policyを明記する。グラフは2系列オーバーレイ＋差分、epoch-wise summaryを作る。metricの定義とサンプリングstepを統一する。
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
| E4 | held-out特徴評価（研究上の確認） | 学習前・公式学習後・自作学習後の凍結backbone特徴を同じ評価protocolで比較。ImageNet val kNNを候補、Linear Probeは別確認。初期値より改善したか、公式と自作の差はどの程度か |

**区別する合否**：`engineering_pass`、`vram_feasible`、`training_completed`、`curve_agreement`、`representation_quality_evaluated` を別々に保存し、一つのPASSへ無条件にまとめない。

**結果をどのように解釈するか**：
- 同じ入力/初期値で両者が近い曲線を示しても、実画像10epochで学習に十分な特徴を獲得したとは限らない。
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
| **B3** | **どこまで完全に同じ学習入力を再生し、Cometをライブ送信するか** | **公式M/Selfを同一state＋同じImageNetサンプル順＋同じ2-view拡張で比べる**方式を推奨。非同期/事後アップロードで原runを壊さず、Cometには2系列・epoch/step・source hashを残す。augmentationの再生は公式transform分布を変えない実装で検証する。完全同期がコスト過大なら独立runの曲線類似と区別 |
| **B4** | **10epochの表現品質を最低限どこで測るか** | 最初は固定ImageNet val kNN（学習前・公式後・自作後）を別Gateとして推奨。Linear Probeは必要なら次段階。10epochなので公式200epoch最終性能の数値閾値は指定しない |
| **B5** | **worker/Checkpoint/失敗時の運用細部** | 公式`--workers 32`相当（2GPU各16）を第一候補にし、VRAMだけでなくCPU/RAM/NASを測る。epoch毎native checkpoint＋rank sidecar、差分のないresume試験、失敗時はfresh outputを維持。保存上限・Comet命名/タグは実装前に固定 |

`non-blocking`：CLI file名、local JSONL file名、Comet display nameの表記、既存helperの呼び出し構造等は既存repo規約に従う。ただしmetricキー、同一step同期、checkpoint/experiment identityを曖昧にしない。

**Spec readiness: draft / BLOCKED FOR APPROVAL**。global256/local128/LR0.03/10epoch/公式MoCo v2/Comet比較/VRAM Gate必須は採用済み。B1のfail-closedはユーザー要求からほぼ導かれるが、B2〜B5の比較方式・副作用・品質評価はまだ採用待ち。**ユーザーが推奨方針を採用した後にapprovedへ昇格可能**。その時点でも実際のV1/E1/E2/E3のGPU実行は別許可。V1実機成功までは10epochを開始しない。

## 9. Handoff（approved後に使用）

- **先に実装するのは**専用epoch-aware runner + config + minimal tests + Comet/local logging adapter + local128 VRAM short pilot gate。既存公式cloneは非改変。
- **非対象**：A3 parityハーネスの演算変更、既存A5-pairの設計変更、ViT/LoRA/Sequential Loader、Meta公式200epoch checkpoint精度の再現主張。
- **必要な検証**：CPU deterministic canary/epoch schedule/mapping/resume + 既存非回帰。短いGPU V1は後日別許可。
- **大規模実験の開始条件**：V1 PASS、E1/E2を段階的PASS、同条件+clean immutable commit、資源見積り承認。10epochの2run、Comet送信は**NOT_RUN**。
- **結果の報告**：公式/自作の条件差がないこと、全epoch history、Comet URL/experiment keys、Loss/cosine差、GPU/CPU/NAS/VRAM、freeze feature eval、確立したclaim/未確立のclaimを分ける。
