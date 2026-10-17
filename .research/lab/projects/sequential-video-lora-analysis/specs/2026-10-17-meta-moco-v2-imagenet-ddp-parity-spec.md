---
project: sequential-video-lora-analysis
spec_type: implementation
status: draft
title: Meta MoCo v2 ImageNet-1K 2-GPU DDP / Shuffle-BN reference parity (Spec A)
created: 2026-10-17
last_updated: 2026-10-17
authority: user-adopted comparison design and Spec A decisions, pending spec approval
workspace_repository: haruto2919/research-workspace
workspace_branch: main
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: pin-at-preflight
reference_repository: facebookresearch/moco
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
dataset: ImageNet-1K
dataset_root: /mnt/NAS-TVS872XT/dataset/ImageNet
---

# Spec A：Meta公式MoCo v2と自作MoCoのImageNet-1K・2GPU DDP数値一致検証（draft）

> **Status: draft（実装不可・未承認）**。2026-10-17までのユーザー採用事項を反映した実装・検証仕様の初稿。未解決事項は第11節に記載する。このspec作成は、GitHub上の研究コード変更、研究サーバでのclone、GPU実験、commit/push、10 epoch学習の実行承認を意味しない。

## 1. 目的・成功と主張できる範囲

2026-10-15 MTGでは、CIFAR-10＋ResNet18のStandalone MoCoの64/500stepはQueue・EMA・checkpointなどの工学的GateでPASSしたが、良い特徴を獲得した証拠ではないと確認された。これに対し、**Meta公式MoCo v2を独立した比較基準にして、自作MoCoが同じ計算・パラメータ更新・DDP通信を行うかを直接検証する**。

Spec Aの目的は、固定した初期パラメータ、BN buffer、Queue、Optimizer、2-view画像、rank割当て、Shuffle-BN permutationのもと、1stepおよび複数stepのForward、InfoNCE、Gradient、SGD、Key EMA、Queue、BatchNormを数値的に比較すること。

- **工学的成功**：事前に確定した誤差閾値で各tensor/状態の参照実装との一致を示す。
- **工学的未成功**：値・時点・rankの差が説明できない、あるいはDDP/Queue/Optimizerが不整合。
- **本specが証明しないもの**：ImageNet表現の有用性、10epoch学習の統計的再現、Metaの200epoch公開精度、動画の時間情報獲得。

別specの**Spec B**は公式・自作それぞれ独立したfresh 10 epoch、Cosine Schedulerも10epochで完結させ、固定validation画像集合の特徴評価を必須とする。Linear Probeは別Gate。Spec Bの学習・評価実行はこのSpec Aの対象外。

## 2. Authority・固定参照・コード現況

### 2.1 ユーザーが採用したDecision Contract
1. Spec AとSpec Bを分離。Spec Aでは公式コードと自作コードを独立ディレクトリに保持して比較し、公式コードを自作へコピー・置換しない。
2. 公式MoCo v2のアルゴリズムを変更しない。公式側に追加してよいのはComet・計測などの非侵襲的変更のみ（import等の互換対応は第11節のGate）。
3. ImageNet-1K＋ランダム初期化ResNet50＋224px＋2GPU DDP/Shuffle-BNを採用。
4. 比較基準の初期設定はglobal batch64＝32/GPU、Queue K65536、SGD initial LR0.0075、momentum0.9、weight decay1e-4、m0.999、T0.2、dim128、MLP。BatchとQueueは実機に収まるか測定してから最終的な実行値とする。
5. FP32で1step・複数stepの数値一致と、DDP/Queueの正常性を必須確認する。結果はComet、構造化ローカルartifact、checkpoint、比較reportに残す。
6. データルートはユーザー提供の /mnt/NAS-TVS872XT/dataset/ImageNet 。パスの提供は実体の確認ではない。

### 2.2 GitHubで確認したMeta固定版
- Repo: facebookresearch/moco
- Commit: a419ef282fc9885015674e77f58c59447206a27f
- moco/builder.py：公式はResNet50をnum_classes=128で構築。MLP有効時、最終fcを2048→2048→128（Linear,ReLU,Linear）とする。Query/Keyは同一初期重みでKey gradientを止める。forward中はQuery→Key EMA→global batch shuffle→Key→unshuffle→旧QueueからInfoNCE logits→当step Keyをglobal gatherしてenqueueの順序。
- main_moco.py：DDP、ImageFolder(train)、DistributedSampler＋set_epoch、224px aug-plus、SGD、epoch毎Cosine更新。MoCo v2フラグは --mlp --moco-t 0.2 --aug-plus --cos 。**公式のCosine更新はepochごと**であり、既存自作CIFAR smokeのstepごとCosineとは異なる。Spec Aの固定step比較時はLRの更新時点を公式と揃える。
- **確認済みの互換性懸念**：このcommitのmain_moco.pyは deeplearning.cross_image_ssl.moco.builder / loader をimportしており、通常の独立cloneでそのまま動くとは限らない。非侵襲的な外部Pythonパス配置・import shimを優先検討する。公式ソースをpatchする場合はimport/packaging限定の変更を別途レビュー・承認し、diffを保存する。**公式MoCo計算処理の変更は不可**。
- **ライセンス表記に食い違い**：当該commitのLICENSEとbuilder.pyヘッダはMIT、READMEの文末はCC-BY-NC 4.0と記述。いずれかの表記が一致しているとは断言せず、公開/コピー方針の判断前に原本を確認。公式コードを研究コードrepositoryへ転載しない。

### 2.3 GitHubで確認した自作側
- Repo: tamaki-lab/2026_09_ishikawa_sequential-video-lora、対象branch: dev（実装前にHEAD commitを再取得してpin）。
- self_supervised/moco/standalone_moco.py：CIFAR-10用ResNet18、単一device、Separate MLP、EMA、FIFO、InfoNCE、rand unit Queue。all_gatherを使わずDDP Shuffle-BN未実装。
- scripts/smoke/smoke_moco_cifar10.py および conf/moco_standalone.yaml：CIFAR専用・単一epoch内step・Comet ONを未実装として拒否。
- 過去の承認済みCIFAR spec、smoke、LoRA regressionを維持する。既存READMEにLightningのDDPがあってもMoCo DDPの実装完了とはみなさない。
- 研究サーバのdirty changes/unpushed commits、CUDA、NAS内の実データ、checkpointはGitHubから未確認。

## 3. Scopeと実装方針

### 3.1 公式参照側
- 研究コードrepoの外側にMeta公式の**独立clone**を置き、固定commitをcheckoutする。絶対clone先は研究サーバで確定・記録し、元の固定commitを変更不能な参照状態として保持する。
- 非侵襲的なComet instrumentation、rank0記録、parity用の外部harness・hook・state注入・dataset adapterのみを許容候補とする。元commitからの差分、実際にロードしたmodule pathとfile hashを保存。
- 公式本体を改変して同じ値になった結果は、その差分が解決されるまで「公式との一致」と称さない。ライセンス表記とimport問題をpreflightで監査する。

### 3.2 自作側
- **CIFAR専用経路を破壊しない**。ImageNet ResNet50用の独立Hydra config/entrypointと、DDP・Shuffle-BN・cross-rank gather・比較fixture・結果保存を追加する案を第一候補とする。
- Query/Key ResNet50はtorchvision random init、通常の7×7 stem/maxpool、pool後2048-d。MLPは2048→2048→128。公式ではfc内、自作では独立Projectorというmodule名の差があり得るため、parameter＋bufferの**明示mapping表**を設けて初期stateを同値化する。
- モデル側はencoder/EMA/Queue/InfoNCEのみ、trainer側がOptimizer、DDP起動、データ、Comet、Checkpointを担当。一般的な汎用frameworkは新規作成しない。
- 既存CIFAR-10 smoke、Hydra設定、LoRA、旧Stage5/6のモデル/出力を既定で変更しない。

### 3.3 Distributed契約
- world_size=2、GPUごと1process、NCCL（実環境で検証）、local batch32/global64、drop_last=True、DistributedSampler、set_epoch、step＝2rank合計の1optimizer update。
- Key EMAはparameterのみ、KeyのBN running bufferはEMAの対象外。公式同様のShuffle-BN（Key画像のglobal gather→rank0発broadcast permutation→rankごとのKey forward→inverse permutation）、Keyのglobal all_gather→Queue enqueue。SyncBatchNormへの代替は禁止。
- Queueとptrは**全rank一致**、Queryの更新・DDP gradientsは対応パラメータで整合すること。
- **BN bufferを無条件に全rank一致と要求しない**。Key側BN running statisticsはlocal shuffled minibatchとDDP buffer broadcast timingでrank差が生じ得る。同一rankの公式 vs 自作を、同じforward時点で比較する。DDP broadcast_buffers・Key EMA時点は検証値として明記する。

## 4. 採用条件と事前確認

| 項目 | 共通設定 | 検証境界 |
|---|---|---|
| Dataset | ImageNet-1K train、SSL lossにラベルを用いない | rootのtrain/val構造・class inventory・読み取りは実機preflight |
| Dataset root | /mnt/NAS-TVS872XT/dataset/ImageNet | ユーザー提供、アクセス実体未確認 |
| Input/2-view | 224×224、Meta MoCo v2のaug-plus、ImageNet normalization | 同じaugmentation済み2viewをfixtureとして共有 |
| Backbone | ResNet50、ランダム初期化 | 初期weights＋BN buffersを共通コピー |
| MLP | 2048→2048→128 | Key/Queryとも対応state照合 |
| GPU | 2×RTX A4500（各約19GB VRAM） | 両rankの実際の割当てとpeak VRAMを測定 |
| Batch | local32 ×2/global64 | 初期試験値、OOMなら停止・再協議 |
| MoCo | K65536、m0.999、T0.2、L2 normalize、all K negatives | 初期random Queue、FIFO、old Queue logits |
| Optimizer | SGD LR0.0075, momentum0.9, wd1e-4, nesterov=false | 公式と同じパラメータ対象・step timing |
| Precision | FP32、parity中AMPなし | TF32、CUDA/backend flagsとtorch/torchvision versionを記録 |
| Seed | seed0を初回候補 | RNGだけでなく実初期state/input hashを共有 |
| Scheduler | 固定stepテストは同じLRを供給。10epoch版はSpec Bで公式と同じepoch単位Cosine | 10epoch学習は本specで実行しない |

ImageNet実機preflightはsplitの存在とクラス・画像数・読み込み/デコード可否を確認して、その結果をartifact化する。データを無断download・移動・再構成しない。

**メモリ不足時の挙動**：Queue本体はK65536・dim128・FP32で32MiB/rank。自動でKやbatchを変えず、GPUごとのpeak allocated/reserved、step時間、OOM箇所を記録して停止。local batch16/global32などの代替とLR調整は**別条件としてユーザーへ再提示**し、両実装に同じ変更を適用する。片方だけAMPやK変更は不可。

## 5. 同一入力・状態の数値比較契約

### 5.1 Canonical fixture（候補）
1. **初期state**：公式ResNet50のquery/key backbone、MLP、BN running stats、Queue K65536とptr、optimizer param groups/momentum bufferをsnapshot化し、自作の対応名へマッピング。各tensorのshape/dtype/hash、Key grad無効、storage非共有を確認。公式にない診断用real_key_countを別に管理しても、同等性判定は公式queue/ptrに基づく。
2. **入力**：Synthetic固定2view→ImageNet実画像の固定2viewの順に検証。各rankのサンプルindexと前処理後tensor [32,3,224,224]、global order、入力hash、dtypeを固定。両実装で別々にRandomResizedCropを実行した結果を「同じ入力」と見なさない。
3. **Shuffle-BN**：公式のrank0作成global permutationとunshuffle indexを記録・再利用し、各rankのpermutation/RNGを一致。公式ロジックを変更しない方法で共有できるかをpreflight。公式vs自作のrank対応を固定。
4. **比較タイミング**：before forward、after query、after Key EMA、after Key forward/unshuffle、after logits-before enqueue、after enqueue、after backward、after SGDをそれぞれ記録。観測hookでtensorやRNGを変更しない。

### 5.2 比較項目
| 比較対象 | 期待 |
|---|---|
| Query/Key/MLP/BN/Queue初期state | Shape/dtype/値が同じ、明示parameter mapping成功 |
| q/k features | L2 normalize済み値を同一rank/同一sample順で比較 |
| Positive/Negative logits・temperature・CE | **enqueue前の旧Queue**に対する値を比較 |
| targets/形状 | 全0、dtype long、sample数／rank順／ptrは厳密一致 |
| SGD gradients・更新後Query | DDP allreduce、momentum buffer、weight decay、LRを一致 |
| Key EMA | Gradientなし、EMA式と更新順序を一致 |
| Queue | 各rankで同一global key順にenqueue、Queue値／ptr／wraparoundが一致 |
| Shuffle-BN・BN buffers | shuffle/unshuffle、rank別running_mean/var/counter、broadcast時点とQuery gradientの整合 |
| Checkpoint/resume | save/loadでstateが復元し、次stepの状態差が記録される |

### 5.3 許容誤差
- **暫定候補：FP32 rtol=1e-5、atol=1e-6**。ユーザーは「FP32で許容誤差内の数値一致」を採用したが、**この具体値は実機測定前なので未確定**。torch.testing.assert_close等で値・差を検査。
- 各tensor/step/rank/phaseのmax_abs_error、max_rel_error（小分母への注意）、mean_abs_error、mismatch_count、NaN/Infを保存。discrete valuesは誤差許容なく一致。
- 同じ公式fixtureの再実行間誤差と、公式vs自作の差を測定し、正式閾値を**判定用runより前**にレビュー・確定。結果をPASSにするため事後的に閾値を緩めない。
- CPU/GPU・CUDA演算はbitwise一致を要求しない。独立seedだけのLoss曲線一致はparityの根拠にならない。

## 6. Verification Gates（実行には別許可）

| Gate | 内容 | 必須成功条件 |
|---|---|---|
| A0 Preflight | 公式固定commit checkout/import、ライセンス/コードdiff、Data root、GPU/NCCL、学習環境、CIFAR既存tests確認 | 全必須のprovenanceを記録。欠損はBLOCKED |
| A1 Mathematics | 小さいK・Bのsyntheticで公式式・EMA・Queue・2rank Shuffle-BN/all_gatherを検証 | 演算・状態・rank順の計算が一致 |
| A2 One step | 固定2GPU、初期state、view、shuffleによるforward/backward/SGD照合 | §5.2の全必須項目が事前固定閾値内 |
| A3 Multi step / Queue cycle | 小Kでwraparound、実K65536・globalB64では**1024step**で初期ランダムQueueが完全置換し、次step以降も検証 | SGD/EMA/Queue/BN/DDPが各phaseで継続整合 |
| A4 Checkpoint/non-regression | 保存→新インスタンス復元→同じ次stepで同等性。既存CIFAR/LoRA testsも検査 | Model/Optimizer/BN/Queue/必要なRNG/samplerが復元し回帰なし |
| A5 Real-data GPU benchmark | **ImageNet実画像100〜300step**で両rankのVRAMとthroughputを測定 | 最大VRAM、step秒、images/sec、data wait、failure有無が記録され実行可否を判定 |

- A1は小Kの数学テストで、**K65536のQueue一巡の代替ではない**。A3の1024stepは長めのrunなので、所要時間・GPU状態を示し別途実行許可を確認する。
- A5は**別途ユーザー実行許可が必要**。100〜300step完了とVRAMが収まることを自動で断定しない。
- 状態はPASS、FAIL、BLOCKED、NOT_RUNを区別。未実行をPASSにしない。NaN/Inf、OOM、NCCLハング、Queue pointer差、公式元コードからの未承認の機構変更があれば原因・first failing step/rank/tensorを記録して停止。
- **Spec Bでの10epoch学習・固定validation features・Linear Probeは本specのGateではない**。

## 7. Artifact・Comet・checkpoint契約

独立したrun IDごとに、公式か自作かを明示した専用ディレクトリへ次を保存する。具体的な出力先の絶対パスとlogging cadenceは実装前の確認事項。

| Artifact | 内容 |
|---|---|
| resolved_config.yaml | dataset、2GPU/rank、batch/K、MLP、SGD、seed、FP32/backend、Meta v2 flags、env |
| provenance.json | 双方のrepo/commit/dirty、公式patch diff/checksum、環境version、GPU/NCCL、Dataset inventory/manifest |
| fixture_manifest.json | 初期weight/key/BN/Queue/Optimizer hash、rank・2view/sampler/shuffle順とtensor hashes |
| parity_report.json | Gate、step/rank/phase/tensor、誤差と正式閾値、fail reason、Queue/BN/DDP検査、未実行 |
| checkpoint.pt + manifest | Query/Key/Queue/ptr、Optimizer、必要なScheduler、RNG、Sampler、epoch/step、config identity |
| verification_summary.md | 判定と根拠、既知制約、未実行項目、再現手順、比較図への相対path |
| resource_benchmark.json | 各GPUのmax_memory_allocated/reserved、after warmup step時間・images/sec・DataLoader wait等 |

**公式checkpoint制約**：固定commitのmain_moco.pyはepoch、model state、Optimizerを中心に保存している。完全なrank別RNG/Shuffle-BN/DataLoader再開状態まで標準で保存するとは確認できない。必要分は非侵襲的なsidecar等で管理し、公式の保存仕様を無断で変更しない。exact resumeの可否はA4で実測する。

**Comet**：公式・自作でimplementation=meta_reference/self、run_kind=parity/bench、commit、K、global/local B、seed、precision、rank構成、Gate、statusを共通設定として記録。Loss、cosines、grad/update norm、LR、Queue pointer、step timeを同じ定義・step単位（Optimizer更新）でログに残す。rank0中心に集約し、ログ書き込みで学習結果を変えない。Cometが不調でもローカルのfailure reportを保存する。API keyや機密情報は保存しない。

**GitHub保存境界**：巨大checkpoint、ImageNet、Queue tensor dump、資格情報はresearch-workspaceへpushしない。実行後にResearch Workspaceのexperiments/へ条件・測定値と観測に基づくEvidenceを追記する。研究サーバのcloneとGitHubは別環境であり同期を推定しない。

## 8. 互換性・実装境界

- 既存のCIFAR ResNet18 20/64/500step smokeと批准済みCIFAR spec、LoRA/ViT既存テストを非回帰とする。既存CIFAR configをImageNet用に置換しない。
- 公式MoCoは別clone。ライセンス・importの整合を確認し、基準commitとローカルinstrumentation diffを追跡する。
- 自作MoCoのtraining loopに不要な動作変更を入れず、最小のImageNet-DDP経路を作る。新しい専用runner名は既存命名規則に合わせ実装前に確定。
- Meta標準条件と今回のglobal batch64・10epochは同じではない。Metaの200epoch精度を再現したとは主張しない。
- batch/precision/Queueなど一方だけの変更があれば同一条件比較として扱わない。失敗時のauto-fallback・勝手なresumeは禁止。

## 9. 将来の実装・確認の順序

1. **Spec A本文の承認後**、研究サーバで自作repoのHEAD/dirtyと元公式commitを固定して独立clone。import/ライセンス/依存のpreflightを実施。
2. 既存CIFAR非回帰を保ち、ImageNet ResNet50 DDP＋Shuffle-BNの新runnerと最小Comet hook・parity fixture/比較器を整備。
3. A1小K→A2固定2GPU 1step→A3複数stepとQueue循環→A4保存復元/非回帰の順に検査し、差異の初出位置を記録。
4. A5のGPU実行許可を別途得た場合、100〜300stepの実機VRAM/速度を測定。global64/K65536を維持できるか判断。
5. 実測Evidenceと差分をResearch Workspaceに保存し、**Spec B（fresh10epoch×公式・自作、Cosine to10、固定検証集合、Linear Probeは別Gate）**を別途作成・承認する。

## 10. 不採用・保留案

- 公式MoCo本体のEMA/Queue/InfoNCE/DDPを自作側に合わせて改変：不採用。
- 公式ソースの全体コピーを自作repo内に保存：不採用。
- Queue容量を計測前に4096へ削減：現時点では不採用（Queue本体はK65536でも32MiB/rank）。
- CIFAR single-device smokeを無理にImageNet DDP runnerへ改造：不採用。
- 独立10epoch Lossのbitwise一致を実装成功条件にする：不採用。
- Linear ProbeをSpec A必須Gateに含める：不採用。

## 11. Ambiguity Gate・approvedへの条件

### 11.1 2026-10-17 B1〜B6 推奨方針をユーザーが明示採用

ユーザーは本日、従来B1〜B6で提案した方針を**すべて採用**した。以下は採用済みDecision Contractであり、未採用案ではない。一方、**設計方針の採用と研究サーバでの検証PASSは異なる**。実測結果がない項目を「確認済み」としない。

| ID | 採用済みの決定方針 | 実機・実装時の検証Gate／不成立時の扱い |
|---|---|---|
| **B1** | 公式固定commitの学習ロジックを変更せず、まず外部shim/環境・Pythonパス設定など非侵襲的手段でimportを解決する。原ライセンス表示を保持し、Metaコードを自作repoにコピーしない。 | 固定commitのclone/importと原LICENSE/READMEの差を記録する。**source import patchが必要な場合のみ**変更diffを示し、適用前に追加承認。アルゴリズム改変は不可。 |
| **B2** | FP32 tensorの初期許容誤差候補に `rtol=1e-5, atol=1e-6` を採用し、公式fixture再演・公式対自作の診断差分を測って、**合否判定用runより前**に閾値を固定する。 | この数値が妥当であることは**未検証**。基準を満たさなければfirst-failureを記録し、必要な変更は理由を添えて再承認。shape/index/Queue ptr等は完全一致。run結果を見て都合よく閾値を緩めない。 |
| **B3** | 公式・自作へ固定2-view tensor、全query/key/optimizer/Queue/BN初期状態、全rankのdata order、Shuffle-BN permutationとDDP buffer broadcastの時点を共通に与えるcanonical fixture＋専用比較器を設ける。 | 2GPUで同一fixtureを確実に再演できるか実装・テストで確認。同じrank/同じphaseのBN bufferを比較。seedのみの一致は成功条件にしない。 |
| **B4** | 最初はFP32、local batch32×2（global64）、K65536で短時間ベンチマークし、VRAM・step time・DataLoader待ち時間を実測する。 | **まだVRAM peakは未計測**。OOM等があれば自動でK/precision/batchを変えず停止・報告。batch16等は両実装の比較条件を改めて提案してユーザー承認後に採用。**100〜300stepのGPU実験は別途実行許可が必要**。 |
| **B5** | Meta標準checkpoint形式は維持し、足りないrank別RNG/Sampler等の再現状態をsidecar等へ保存する。固定fixtureで保存→別プロセス復元→次step一致を試験する。Cometとローカルartifactのmetric定義・step単位を揃える。 | 公式checkpointだけで完全再開可能と主張しない。Sampler/DataLoader worker乱数やGPU rank別stateの復元可否を測定し、再現不能なら未対応として記録。ログ頻度・保存場所は既存実装との整合をレビューして最終決定。 |
| **B6** | ユーザー提供の `/mnt/NAS-TVS872XT/dataset/ImageNet` は**読み取り専用**でinventory・構造・サンプルdecode・I/Oを監査する。 | `train/`、`val/`、ImageFolder class構造、画像件数・欠損・読取権限の結果を保存。構造不適合なら停止し、データ移動・再編成・downloadは自動で行わない。GitHub経由でNAS実体を確認したとは主張しない。 |

### 11.2 承認状態の区別

- **確定済み（Decision）**：上表のB1〜B6の方針、元の4つの基本方針、Spec Aの8項目。改めて同じ判断を質問しない。
- **実装・実機検証待ち（Evidence）**：B1のimport、B2の基準誤差、B3のDDP fixture、B4のVRAM、B5のcheckpoint resume、B6のNAS inventory。Gate A0〜A5の必要なrunの結果が得られるまでPASS/FAILは未確定。
- **条件付きの追加意思決定**：Meta source import patch、数値閾値の変更、VRAM不足による学習batch・precision・Kの変更、データセット構造の変更が必要になった場合は、承認済み条件を無断変更せずユーザーへ再提示する。
- **Spec本文の承認**：今回はB1〜B6の推奨案採用であり、**Spec A全文をapprovedにする明示承認と同一視しない**。このファイルは引き続き `status: draft`。Spec A本文のレビュー・承認が完了したらstatus更新する。
- **実行権限**：承認済みspecでも、研究コード変更、commit/push、短いテスト、Queue一巡、100〜300step実験、10 epoch本学習は別々の行為として扱う。特にA3長めのrunとA5のGPU benchは別途許可を必要とする。

### 11.3 Non-blocking

コードの局所命名、reportの視覚表現、既存形式に合わせた保存先の細部は既存規則に従う。ただし変化すると入力順や比較条件に影響するDataLoader workers、rank別seed、logging side effects等は記録し、比較の前に両実装で条件を一致させる。

## 12. 関連文書・コード

- 2026-10-17 brainstorm：../../../../secretary/notes/brainstorm/2026-10-17-meta-moco-v2-reference-parity-validation.md
- 2026-10-15 MTG：../meetings/2026-10-15-mtg.md
- CIFAR standalone approved spec：2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md
- CIFAR GPU 64/500step Evidence：../experiments/2026-10-15-standalone-moco-gpu-64step-and-gate-c-500step-evidence.md
- Meta公式builder：https://github.com/facebookresearch/moco/blob/a419ef282fc9885015674e77f58c59447206a27f/moco/builder.py
- Meta公式main_moco：https://github.com/facebookresearch/moco/blob/a419ef282fc9885015674e77f58c59447206a27f/main_moco.py
- 自作Standalone MoCo：https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/self_supervised/moco/standalone_moco.py

## 13. Implementation Handoff（未承認・実行不可）

対象：Spec Aのみ。公式固定commitの独立clone、非侵襲的計測、自作ResNet50/ImageNet DDP＋Shuffle-BN、canonical fixture＋parity comparator、CIFAR非回帰。

成功条件：A0〜A4の事前合意された数値/状態GateがPASS、A5の実機結果が別途許可のもと記録されること。10epoch本学習・表現品質はSpec B。

**現状：draft。Meta clone、研究コード変更、GPU run、checkpoint生成、テスト実行はいずれも未実施。**
