---
project: sequential-video-lora-analysis
spec_type: implementation
status: approved
title: Standalone MoCo v2-style CIFAR-10 mechanics and diagnostic pilot
created: 2026-10-15
last_updated: 2026-10-15
workspace_repository: haruto2919/research-workspace
workspace_branch: main
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_baseline_commit: 7dcd194f439c7e28d796e2986857ab0df1982cbe
reference_repository: facebookresearch/moco
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
dataset: CIFAR-10
encoder: resnet18-cifar-random-init
moco_variant: v2-style
authority: user-approved MoCo design and B1-B5 details
---

# Standalone MoCo v2型・CIFAR-10独立検証 実装spec（approved）

> **Status: approved（2026-10-15）。** Queue random初期化・Key-forward前EMA・logits後かつoptimizer前enqueueを含む基本設計、およびB1–B5の全推奨値がユーザーに承認された。**本specは今後の実装基準となるが、研究コード変更の実施依頼、commit/push、500step pilotやGPU実験の実行承認を自動的に意味しない。**

## 1. 目的・証明したいこと

2026-10-08の教授MTGで問題となった、既存ActivityNet統合MoCoのLoss振動・Query LoRA勾配急増に対して、MoCoの正しさを独立に検証できるbaselineを構築する。**LoRA・ViT・Sequential Loader・動画**を使わず、通常のCIFAR-10の画像2-viewで以下を独立証明する。

1. 2-view → Query/Key encoder → 非線形Projector → L2 normalize → Positive/Queue Negative → temperature付きInfoNCEの数値契約。
2. KeyはQuery初期コピー・独立storage・gradientなし・EMAでのみ**パラメータ**を更新し、Query/ProjectorはSGDで学習する。
3. 初期random Queue・FIFOと、Key-forward前EMA／logits計算後・backward前enqueueの順序が**固定された公式参照コード**に整合する。
4. 実画像20step smokeでの成立と、500stepの統制pilotでLoss・類似度・特徴量ノルム・勾配・更新量・Queue置換状況を記録できる。
5. 設定・seed・revision・checkpoint・検証値を再利用可能なartifactに残し、後段のFrozen ViT移行における比較基準を持つ。

**工学的PASSと研究的性能を区別する。** NaN/Infなし・更新経路成立は工学的検証であり、Loss単調減少、CIFAR-10線形評価精度向上、時間情報獲得の証明ではない。

## 2. Authority・現在コード・参照の確定範囲

### 2.1 ユーザー採用事項

- Stage1はCIFAR-10 train、ResNet18（CIFAR向けsmall stem、初期学習済み重みなし）、MoCo v2型、2層Projector 512→512→128、通常のDataLoader、LoRAなし、Sequential Loaderなし、batch32、seed0、Queue K=1024、EMA m=0.999、T=0.2、SGD momentum0.9、20step smoke→500step診断候補。
- **公式方式へ変更したユーザーの明示指示**：Queueはrandom unit-vectorで満たして開始、warm-upなし。Queryを更新する前にKeyをEMAで更新してからKey forward。Queueへのenqueueは旧Queueによるlogits構築後・Backward/SGD前。
- 公式のMoCo v2を小規模に適応した**機構検証**であり、ImageNet精度やDDPによる公式pretraining再現を主張しない。
- 保存はJSON/YAML/グラフ/Markdown/checkpoint、Comet opt-inを第一候補とする。
- 2026-10-15にユーザーがB1–B5の推奨値を一括採用した。4.2・4.5・4.6の詳細設定、Queue全件Negative、データ実行範囲は承認済みの契約とする。

### 2.2 調査済み基準

- GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、閲覧時HEAD `7dcd194f439c7e28d796e2986857ab0df1982cbe`。GitHubには`model/vit/`、`model/resnet/resnet18.py`、`dataset/cifar10.py`、`conf/config.yaml`、`conf/lora_standalone.yaml`、CIFAR LoRA smokeがある。tree上にMoCoの本体は存在しない。ルート`AGENTS.md`/ `CLAUDE.md` は確認できなかった。
- 既存`model/resnet/resnet18.py`は`torchvision.models.resnet18(weights=IMAGENET1K_V1 or None)`の**通常の画像分類wrapper**であり、分類Headを置換する構成。今回のCIFAR small stem＋対照学習用MoCoへの無条件流用はせず、新規MoCo専用entrypointから`torchvision.models.resnet18(weights=None)`を使う。既存classification経路を壊さない。
- `dataset/cifar10.py`はdownload=Trueの学習/評価DataLoaderを構築する。**今回の既定download=false、train labels未使用、TwoCropsTransformを確実にするため別のMoCo用Dataset/transform配線を使う**。既存ファイルを変更しない。
- `conf/config.yaml`は`main.py`等の汎用Hydra設定、`conf/lora_standalone.yaml`は別学習経路。今回も**専用Hydra設定**を新規作成する。
- Meta公式`facebookresearch/moco`の現mainは2026-02-03削除commit `8976944da9c6b94cbd9158d7ebe50912aef807ef`で空。**過去固定commit** `a419ef282fc9885015674e77f58c59447206a27f` の`moco/builder.py`, `main_moco.py`, `README.md`, `LICENSE` を参照する。公式READMEのMoCo v2指定は`--mlp --moco-t 0.2 --aug-plus --cos`。公式はResNet50/ImageNet/DDP前提、今回の単一device/ResNet18/CIFAR10で公式精度を再現したとは呼ばない。公式コードの複製が必要なら該当ライセンス表記を維持する。
- [2026-10-08 MTG](../meetings/2026-10-08-mtg.md)、[Stage5旧spec](2026-09-24-stage5-moco-v2-lora-one-step-smoke-spec.md)、[Stage5旧実験](../experiments/2026-09-24-stage5-moco-one-step-verification.md)、[今回のユーザー決定メモ](../../../../secretary/notes/brainstorm/2026-10-15-standalone-moco-spec-decision-points.md)、[段階的統合メモ](../../../../secretary/notes/brainstorm/2026-10-15-moco-loader-lora-staged-causal-debugging.md)を参照。
- 旧Stage5のPEFT/動画用MoCoは「空Queue＋warm-up、optimizer後EMA＋enqueue」だったが、**今回の明示指示が優先**。そのコードの数学・テストを参考にできても、今回の正本動作として復活させない。
- 研究サーバのdirty/untracked・データ実体・GPU健全性はGitHubから確認できない。実装着手時の`git status --porcelain`、actual HEAD、dataset cacheをpreflightする。

## 3. Scope・互換性境界

### 実装scope（予定）

- 通常画像用のMoCo Query/Key Backbone＋Projector、EMA、FIFO Queue、InfoNCE、2-view augmentation、SGD更新を独立モジュール化。
- 独立したCIFAR-10のHydra config/entrypoint、人工入力による単体・短時間real-data smoke、検証レポート、checkpoint保存/復元の検査。
- 後段のViTへ移行可能な最小interface：Backbone（feature dimensionalityを属性または明示引数で取得）、Projector、MoCo計算/Queueを分離する。ただし汎用トレーニングフレームワークや任意encoder factoryは作らない。

### 対象外

- ViT・自作LoRA・PEFT・Sequential Loader・動画chunk・ActivityNet・negative metadata/sequence masking・strict-online・Linear Probe・MoCo v3。
- ImageNetでのDDP/multi-GPU再現、公式ResNet50 ImageNet精度、長時間full pretraining、本格hyperparameter sweep、データ自動ダウンロード、実運用Comet常時ON。
- 既存`main.py`、`main_pl.py`、`model/resnet/resnet18.py`、`model/vit/*`、`dataset/cifar10.py`、`conf/config.yaml`の挙動変更。既存出力schema、CIFAR LoRA検証を維持。
- GitHub上に記録されていない以前のSSH MoCo実装を「現行実装」として強制マージしない。

## 4. モデル・データ・計算契約

### 4.1 EncoderとProjection

- Backbone：`torchvision.models.resnet18(weights=None)`。CIFAR10 input `[B,3,32,32]`に対して最初のconvを `Conv2d(3,64,3,stride=1,padding=1,bias=False)` へ変更、`maxpool=Identity()`、global average pooling後の512次元をbackbone outputに使用。元分類FCは使わない。**必ずrandom initialization**、学習前重みとseedを記録。
- Query/Keyは同一構造・初期state完全一致、Tensor storageは別。初期Key全parameterを`requires_grad=False`、Query全parameterとQuery ProjectorだけOptimizer対象。
- 両branchのProjector：`Linear(512,512)` → `ReLU` → `Linear(512,128)`。初期Query→Key stateを完全copy。L2 normalization: `F.normalize(output, dim=1)`、default dim128。
- Key Backbone＋Key ProjectorともパラメータEMA対象。Query/KeyのBatchNorm bufferは上のweight EMA式には含めない（BN-specific policyは5.5で指定）。
- Query/Keyの中間表現の次元、Unit norm、finite、`grad_fn`境界を検証する。

### 4.2 CIFAR-10と2-view

- `torchvision.datasets.CIFAR10(train=True, download=False)`を使う。downloadが必要なら**明示的overrideでのみ**許す。画像ラベルは返る場合もcontrastive loss、positive/negative、train/validation評価に使用しない。
- 同じsampleから独立random変換で`im_q`と`im_k`を生成。`TwoCropsTransform`相当の小さなwrapperを利用し、同一indexの2viewを対応づける。
- **採用した画像拡張（公式MoCo v2の32×32適応）**：
  `RandomResizedCrop(32,scale=(0.2,1.0),ratio=(0.75,1.3333333))` →
  `RandomApply([ColorJitter(0.4,0.4,0.4,0.1)],p=0.8)` →
  `RandomGrayscale(p=0.2)` →
  `RandomApply([GaussianBlur(kernel_size=3,sigma=(0.1,2.0))],p=0.5)` →
  `RandomHorizontalFlip(p=0.5)` →
  `ToTensor` →
  `Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])`。
  PILから変換し、2viewが確実に独立random drawであること、view shape `[32,3,32,32]`を確認。公式が224画像向けに使ったBlurカーネルを盲流用せず、kernel=3とする点は**公式との差分**。
- full CIFAR-10 train（50,000枚）からseed0で1エポック分の固定シャッフル順。batch32、drop_last=true、num_workers=0、shuffleに使ったseedとsubset policyを永続化。pilot 500stepは最大16,000入力であり初回epoch範囲（再読込による同一画像の再出現なし）。`dataset root`はconfigで指定。
- 2view対応のsample indexとinput shape・rangeのsanityをunit検証。ただし学習時の元画像やcrop RNGを毎step全保存する義務はない。再実行・resume時に必要なRNGとsampler stateは保存する。

### 4.3 Queue/InfoNCE（公式に近い方式を採用）

- Queue buffer `[128,K]`、K=1024、`torch.randn(128,1024)`を**列単位** `F.normalize(dim=0)`。初期Queueには実画像由来ではないrandom unit vectorが**K個**入っている。`queue_ptr=0`、`real_key_count=0`。空Queue/Key-only warm-upは**採用しない**。
- 現在の`q,k`はそれぞれ`[B,128]`でunit norm。Positive logits `l_pos=(q*k).sum(dim=1,keepdim=True)`。`l_neg=q @ queue_snapshot` (`[B,K]`)。logitsを`concat([l_pos,l_neg],dim=1)/T`、T=0.2、targetsはbatch要素すべてclass index0。Loss=`F.cross_entropy(logits,targets)`。
- **現在計算したPositive keyは当stepのNegativeには使わない。** Queueのsnapshot（`clone().detach()`または同等のautograd安全な手段）を先に作り、その後enqueueしてもbackward計算が変化しないことをtest。Key/Queueともgradientなし。
- Meta方式に合わせ、最初のspecではQueue内の**すべてのKキーをNegativeとして扱い、sample-ID/クラスlabelによるmaskをしない**。ランダム初期キーは別サンプルとしての保証がないこと、後段の動画metadata policyとは異なることを明記。学習用ラベルは使用しない。
- `enqueue`は正規化済み`k.detach()`で、32keyをpointer位置へ書き込み、`ptr=(ptr+32)%1024`。capacityがbatchの整数倍という前提で実画像runnerはbatch32固定。単体テストはK=8,B=4などでwraparound/oldest上書きを確認。`real_key_count=min(K,steps*batch32)`を別管理し、32step後に全keyがrealになる。Queueのunit norm/finite、pointer、FIFO、初期RNG再現性を監査。
- 本仕様のnon-DDP単一deviceにおける`all_gather`は恒等操作。DP/DDPはサポートしない。

### 4.4 Query/Key/Queueの1step順序（ユーザー採用）

0. run開始時、Query/Key encoder＋projectorの初期stateを完全一致させ、別storageで保持、random normalized Queueを登録（warm-upなし）。
1. 現在のQuery view `im_q`をQuery Encoder＋Projectorでforwardし、`q`を正規化。
2. `torch.no_grad()`内で、Optimizer更新**前**のQueryパラメータ`theta_q`を用い`theta_k <- m*theta_k+(1-m)*theta_q`（m=0.999）。続けてEMA後のKey encoder/projectorで`im_k`をforwardし`k`を正規化。Key paramsにgradientなし。
3. EMA/Key-forwardの後、**enqueue前のQueue snapshot**で`l_pos,l_neg,logits,loss`を作る。
4. forward内またはtrainer境界で、現在のdetached`k`をQueueへenqueue。**Backward/optimizerより前**。ステップ3のlogitsは変更されない。
5. `optimizer.zero_grad`・`loss.backward`・SGD`optimizer.step`でQuery Backbone＋Projectorを更新。Key parameterはOptimizer対象外。
6. next batchへ。step2に戻る。

**EMAをOptimizer後に移す・enqueueをOptimizer後へ移す・random Queueを空Queueへ置き換えるのは今回のspec違反。** 初回EMAは初期Query/Keyが同一ならparameter値を保つ。BN bufferの更新とbatch shuffleは別契約で扱う。

### 4.5 BatchNorm／batch shuffle（採用済み詳細契約）

- Query/Keyとも`train()`でforwardし、各branchのBatchNormの`running_mean`,`running_var`,`num_batches_tracked`は**各branchのforwardの通常規則でのみ更新する**。KeyのBN parameter `weight/bias`はEMA対象、BN running bufferをQueryからコピー・EMAしない。`torch.no_grad()`はKey側BN running bufferの自動更新を止めない点を明示する。
- Key側の特定BN bufferをOptimizer/EMA対象として誤更新しない。両branchのBN bufferはstate_dict・checkpoint対象とする。
- **single-deviceでDDP batch-shuffleを実装しない**。公式のDDP用`_batch_shuffle_ddp`/all-gatherと同等の分散BatchNorm behaviorは主張しない。batch構成の統計効果に注意し、num_workers、batch size、BatchNorm modeを固定する。
- Key batch input順序はQuery viewとsame sample indexを対応づけて保持。将来DDPを追加するときは別specでBN/shuffleを再検討。

### 4.6 Optimizer / Scheduler（採用済み数値）

- SGD: `momentum=0.9, weight_decay=1e-4, nesterov=false`。初期LR `0.00375` は**公式ImageNetのlr0.03 / batch256をbatch32へ比例換算した初期候補として採用した値であって、CIFAR10最適値の根拠ではない**。公式の性能再現は求めない。
- **Gate B 20step smoke**: `max_steps=20`、LR `0.00375`固定、scheduler disabled。乱数seed0、batch32、drop_last。新規run、fresh Query/Key/Queue。forward/backward/step/EMA/Queue/ログとcheckpointの成立を確認。
- **Gate C 500step pilot**: `max_steps=500`、LR `0.00375`開始、stepに応じたcosine decay（`lr(t)=0.00375*0.5*(1+cos(pi*t/500))`, tは完了済みoptimizer steps、最後のstepに向けて0へ近づく）。**20step smokeから重みを引き継がず新規run**。20stepと500stepのloss曲線は学習率履歴が違うため無条件に数値比較しない。
- Gradient accumulationなし、mixed precisionなし、gradient clippingは既定OFF。NaN/Inf検出時はfail-fastしてfailed artifactを残す。比較のためloss、grad、実parameter update（SGD後）とLRを全step記録。
- CPU unit/syntheticは実行環境を問わず使える。GPUは`device=cuda`を**明示した場合だけ**使用し、利用不能・ドライバ応答不能なら明示エラー。勝手にGPU device indexを選ばず`CUDA_VISIBLE_DEVICES`などで外部制御する。CPU→GPUのbitwise完全一致は合格基準にしない。

## 5. 変更対象・ファイル配置案

既存構成を維持し、新規だけで完結させる。実装前に最新dev/head/dirtyと既存類似ファイルを再確認し、二重実装を防ぐ。**名称・配置は局所的な実装候補であり、役割・挙動・artifact契約は固定**。

```text
self_supervised/
  moco/
    __init__.py
    standalone_moco.py              # Q/K backbone+projector, EMA, queue, InfoNCE
scripts/
  smoke/
    smoke_moco_cifar10.py            # Hydra entrypoint, 2-view, SGD, 20/500 presets
    moco_verification_reporting.py   # JSON, PNG, Markdown
conf/
  moco_standalone.yaml               # standalone defaults; existing configs untouched
test/
  model/
    test_moco_standalone.py          # InfoNCE, EMA, queue, BN, step order
    test_moco_cifar10_reporting.py   # dataset fake, 20step fake-smoke, artifacts/reload
```

- `torchvision`（ResNet18、CIFAR10、transforms）、`torch`、`hydra-core`、`matplotlib`、`numpy`は既存requirementsにある。新しいMoCo専用外部dependency、torch hub download、PEFTは不要。
- `main.py`、既存`model/resnet/`・`dataset/cifar10.py`・`model/vit/`・`conf/config.yaml`・CIFAR LoRA smokeを変更しない。独立`conf/moco_standalone.yaml`を`@hydra.main(config_path="../../conf", config_name="moco_standalone",version_base=None)`など既存styleで読む。
- 標準run directory:`log/moco_standalone/<YYYY-MM-DD>/<HH-MM-SS-ffffff>/`（Hydra`job.chdir=false`）。一意なディレクトリにfresh runを作成し、既存成果物を上書きしない。手動resumeは`resume`明示時のみ。
- 実装コードに全Encoder固定名を埋め込まず、次段のViT encoderを載せられる最小の128-d normalized projected-feature contractを残す。一方、今回のscriptはResNet18/CIFAR10以外を正式対応として宣言しない。

## 6. Verification Gate（測定できる合格基準）

### Gate A：数学・構成（CPU unit、必須）

| No | 検証 | PASS条件 |
| --- | --- | --- |
| A01 | Query/Key初期 | state一致、storage非共有、Key全params requires_grad=false、OptimizerにKeyなし |
| A02 | 2-view | 対応sample IDが同じ、2つのrandom drawが独立、形状一致、label未使用 |
| A03 | Positive / Negative | 人工q/k/Queueのdot-productと`/T` logitsが独立手計算と許容誤差内一致、target=0 |
| A04 | InfoNCE | 小さい人工float64ケースで`cross_entropy` reference一致（rtol 1e-8/atol 1e-10を目安） |
| A05 | 初期Queue | 全K列ノルム≈1、finite、pointer=0、random keyが同seedで再現、warmup=0 |
| A06 | Queue FIFO | K=8/B=4・実K1024/B32でpointer、上書き、連続enqueue、wraparound一致、real_key_count遷移検証 |
| A07 | future/current leakage | current kをenqueueしても現在のlogits不変、同step positiveがnegativeに混入しない、autograd version errorなし |
| A08 | EMA | EMA前後の全Keyパラメータが独立に計算した`m*oldkey+(1-m)*oldquery`と一致、**Key forward前**の適用をイベント順序で検査 |
| A09 | Grad/optimizer | Query Backbone/Projector finiteかつ少なくとも一部nonzero gradient/update、Key grad tensor数0、Optimizer parameter ID集合一致 |
| A10 | BatchNorm | Key BN bufferはKey forwardでのみ規則どおり更新、Key BN affine weight/biasはEMA、bufferはコピー/EMAされない |
| A11 | 異常入力 | Kやfeature dim不整合、nonfinite、wrong dtype、Queue pointer破損で適切に失敗し、無言スキップしない |

浮動小数点許容誤差はテストケースのdtypeに応じて設定し、lossの手計算とEMAを「厳密一致」と言うときにbit-exactではなく数値許容誤差内と明示する。

### Gate B：実CIFAR10・20step smoke（必須の短時間検証）

- cache済みCIFAR-10 trainを読み、seed0、batch32、20step。2viewの入力shape `[32,3,32,32]`、Query/Key特徴`[32,128]`、logits`[32,1025]`をすべて有限として確認。
- 20 optimizer updates完了、Key gradient0、EMA error許容内、Query Backbone/Projectorのupdate少なくとも一部nonzero、Queue ptr=640、real_key_count=640、random_key_remaining=384、初期random Keyを含む全Queue unit norm/finite。
- Loss、pos/neg/hardest similarity、feature norm/variance、global grad/update norm、LR、Queue real/random countsを**全step**記録する。`Lossが単調に下がる`は要求しない。
- Query/Key parameter/BN buffers、queue/pointer/counters、Optimizer、scheduler（存在時）、step、seed/RNG、configを保存し、読み込み検証が成功。ラベルを分類Lossとして使っていない。
- データが欠ける・CUDA不可・nan/inf・テスト不合格ならPASS扱いせず、失敗した場所と理由を記録。

### Gate C：500step controlled pilot（**別途実行許可が必要**）

- 独立fresh run、CIFAR10 full train順序とseed固定、batch32、cosine LRで500 optimizer updates。実例総数=16,000、Queueのrandom leftoverは32stepで0。異常停止もstep/log/provenanceを保存する。
- 毎step実測のloss、正規化前Projector `||z_q||_2,||z_k||_2` のmin/mean、正規化後feature各次元のbatch variance、positive cosine、mean/hardest negative cosine、Query Backbone/Projector grad norm、実update norm、LR、Queue real/random countを記録し、`loss.png`等で可視化。
- NaN/Inf、EMA式違反、Queue内容・pointer破綻、Optimizer対象誤り、Key gradient発生はFAIL。勾配急増／pre-normalization norm縮小／variance崩壊の兆候は**診断要**として記録し、原因未解明なら安易な研究的PASSとしない。絶対loss値や毎step下降は成功閾値にしない。
- **Stage1の工学PASS ≠ 表現品質の成功**。CIFAR10分類精度や動画時間情報を獲得したと主張しない。

### Gate D：保存復元・非回帰（必須）

- 人工決定的バッチを固定し、checkpoint save→別インスタンスload→同一評価入力のforward、Queue/key/BN buffers、Optimizer state、次step更新の同等性を指定float許容範囲で検査。Queue全要素とpointer、RNG・data cursorもresume時に再現。
- 同じseed＋同じCPU synthetic conditionsでLoss・Key EMA・Queue内容を数値一致。GPU別環境でのbitwise保証は不要。
- 新規tests PASS、および既存`test/model/test_lora_linear.py`、`test/model/test_vit_lora_frame_encoder.py`、`test/model/test_vit_lora_reporting.py`などCPU実行可能なLoRA/ViT回帰tests PASS。
- GPU依存の既知問題がある他testや、実ActivityNet full runは本specの合否と混同しない。

## 7. Report/Artifact contract

成功runまたは測定済み失敗runで、同じrun directoryへ以下を保存。メトリクスは**観測した実値**だけを記載し、固定`PASS`を出力しない。

```text
log/moco_standalone/<date>/<time>/
  resolved_config.yaml
  report.json
  loss.png
  similarity.png
  grad_update.png
  queue.png
  verification_summary.md
  checkpoint.pt
```

- `report.json`：`schema_version`、`status`、`provenance` (repo/branch/HEAD/dirty, Python/PyTorch/torchvision/CUDA)、dataset split/root・sample count・seed、backbone・projection、Queue/T/EMA、optimizer・LR/scheduler・step数、`history` (global_step、sample_count、loss、cosine/negative stats、grad/update norm、pre-L2 norm、feature variance、queue_ptr、real_key_count、random_remaining_count)、`checks` (各PASS/FAIL/NOT_RUN)、`failure` (あれば原因)。欠損値を0で代用しない。
- `loss.png`：step-Loss曲線、32step時点（K/BS=32）で初期random Queueの全置換を注記。`similarity.png`：positive/mean-negative/hardest-negative。`grad_update.png`：Query Backbone/Projector勾配ノルムと実update norm（同じ単位とは限らないのでlegend/軸明確化）。`queue.png`：random残数・実画像key数・pointer遷移。
- `verification_summary.md`：run条件、Gate別PASS/FAIL/NOT_RUN、公式参照commit、**公式との差分（ResNet18/CIFAR10/single-device/BN policy/blur）**、学習と表現性能に関する限定的結論、chart/checkpointへの相対パス。
- `checkpoint.pt`：Query/Key（parameterとBN buffer）、Projector、Queue/pointer/counts、Optimizer、Scheduler、global_step、Sampler位置、Python/numpy/torch/CUDA RNG必要部分、config identity。**Key/Queueからgradient graphを保存しない**。保存時にGPU TensorをCPUへ適切に移せる形式を設計する。
- Cometは`logging.comet_enabled=false`を既定。必要ならMoCo stageを示す別tags、`global_step`の単位=optimizer updateと明記し、既存の動画stage6bと同じexperiment名を使わない。Comet側欠落時もローカルartifactは完成。
- checkpoint/dataのGitHub pushは行わない。実測が取れたらResearch Workspace`experiments/`にはrun provenanceと主要結果を別途Evidenceとして記録する。

## 8. Reproducibility / execution boundaries

- GitHub対象repo：`tamaki-lab/2026_09_ishikawa_sequential-video-lora`、実装候補branch `dev`。基準HEADはspec調査時`7dcd194f439c7e28d796e2986857ab0df1982cbe`。実装時に現在HEADを取り直し、差分があれば変更影響を再評価。ローカルdirty/untrackedを勝手にreset/checkoutしない。
- 既存の`requirements.pytorch.txt`はtorch/torchvisionをバージョン無固定で取得する。実行環境の正確なpackage version、CUDA等をレポートへ記録し、後からversion pinを加えるなら互換性Gateを通す。
- CIFAR10 root既定は`./downloaded_data`。事前cacheがなければ無許可ダウンロードや外部fetchにfallbackしない。offlineでもunit/合成検証を独立実行できるようにする。
- 新コード実装、既存コードのread-only調査、CPU unit/短いsmoke、GPU pilot、commit/push、500step以上の長時間runは**個別の実行境界**。本specがapprovedになっても500step実行やGitHub pushが自動許可されたとは扱わない。
- 本文の`python -m scripts.smoke.smoke_moco_cifar10`は将来のentrypoint**候補**。実際にコードが存在するまで実行済み・利用可能とは主張しない。Hydra override例：`data.root=/path/to/CIFAR10 training.device=cpu training.max_steps=20`。
- 正式にStage2へhandoffする際はこのsame MoCo coreのQueue/EMA/logit順序を維持し、BackboneのみFrozen ViTへ変更して一致・差分を検証する。

## 9. Ambiguity Gate / レビュー対象

### ユーザーの判断が採用済みで、再質問不要

- MoCo v2型＋CIFAR10＋ResNet18単体検証、K1024・m0.999・T0.2、batch32/seed0、SGD momentum、20stepと500step段階、JSON/PNG/Markdown/checkpoint、独立Hydra・Comet OFF、現在のViT/LoRA/Loaderを変更しない。
- **公式準拠でrandom Queue、Key forward前EMA、logits後/optimizer前enqueue**。warm-up、optimizer後EMAを選ばない。

### B1–B5：2026-10-15にユーザーが推奨値を一括採用（blockingなし）

| ID | 論点 | 承認済み固定値 | 別案・影響（今回非採用） |
| --- | --- | --- | --- |
| B1 | ResNet18 Key BatchNorm | Query/Key `train()`、Key forwardでBN buffersを独立更新、parameterのみEMA。single GPU batch shuffleなし | buffers copy/EMA、Key eval、batch shuffle導入は数値結果が変わる |
| B2 | CIFAR10用augmentation | 上記4.2のcrop/color/gray/blur3/flip、ImageNet normalize、seed0 | blurなし、CIFAR normalize、別crop範囲でPositive難度が変わる |
| B3 | Learning Rate・scheduler・weight decay | SGD lr=0.00375、WD=1e-4、smoke20固定LR、pilot500 cosine | 参照論文/既存`conf/optimizer/sgd.yaml`の他値、LR policy差で勾配・曲線が変わる |
| B4 | Queue Negativeのsample-id mask | 公式同様、全K keyをNegativeに採用しID maskingなし | 同じsampleの過去keyをmaskするならメタデータQueueが必要になり、公式との差も生じる |
| B5 | dataset execution scope | 50k trainから初回500step以内、同seedのfresh pilot、num_workers0、drop_last | 固定subset/複数epochで再登場頻度やQueue対照が変化 |

**2026-10-15のユーザー発言「採用します」によりB1–B5の推奨値は全件採用済み。Blocking ambiguity: なし。** 実行環境のGPU・データcacheやローカルdirtyはコード実装・実行前に確認する運用事項であり、未決の研究設定ではない。

### Non-blocking

- Python file/helper/testの局所命名、図の色、Markdown表の罫線、内部APIのprivate helper名、state_dict中の非公開key配置などは既存のコード規約に合わせてよい。**ただし外部出力ファイル名、logging metric key、Queue/EMAの公開契約を勝手に変更しない。**
- 将来のViT/LoRA/Sequential Loader接続は別spec。旧Stage5のdifferent-sequence maskや空Queue動作を現在のStage1へ暗黙に混入しない。

## 10. Approval / engineering handoff

- **現在のspec status: approved（2026-10-15）**。ユーザーによるB1–B5の一括採用でblocking ambiguityは解消。**status=approvedと、実際のコード実装・短時間smoke・長時間pilot・pushの作業指示は区別する。**
- Handoff候補：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`の最新HEAD/dirty確認 → 新規MoCo core・CIFAR10 two-view runner・Hydra・reporting・unit tests → Gate A/B/D → 観測結果をユーザーへ報告。500step Gate C、GPU pilot、pushは別途明示許可が必要。
- 成功時も「MoCo mechanics/CIFAR10 short controlled learning engineering verification」と明記し、時間順序・表現学習性能の証明には拡大しない。


## 11. Approval record（2026-10-15）

- **ユーザーによる明示承認**：直前のレビューで提示された5項目すべての推奨値について「採用します」。B1 Key BatchNorm、B2 CIFAR10 augmentation、B3 SGD LR/weight decay/scheduler、B4 Queue全件Negative、B5 dataset/seed/batch/500step fresh-run設定を承認。既に採用済みのランダムQueue初期化、Key forward前EMA、logits構築後・backward/optimizer前enqueueは変更しない。
- **Spec Gate**：目的、scope、既存互換性、実装対象、数値条件、Gate A/B/C/Dの観測可能な合否、artifacts、再現性を記載済み。Blocking 0件、Non-blockingは局所実装命名やグラフ装飾のみ。
- **Implementation handoff**：承認済みspecは本ファイル。対象`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`（調査時HEAD `7dcd194f439c7e28d796e2986857ab0df1982cbe`）。次に実装を明示依頼された場合、最新dev・ローカルdirtyを再確認し、独立MoCo core・CIFAR-10 two-view runner・Hydra・レポート・単体テストを新規実装、Gate A/B/Dを検証。既存ViT/LoRA/Sequential Loader/分類経路は変更しない。
- **実行境界**：今回の採用は仕様の承認であり、研究コードの変更・commit/push・GPU使用・500step Gate C pilotの実行まで依頼したものではない。実行結果がない段階で`implemented`へ昇格させず、実測値は`experiments/`で別途記録する。

## 12. 実装進捗記録（2026-10-15、Gate C未実施）

ユーザーの実装依頼（2026-10-15）に基づき、ローカルの実装repository（branch `dev`、開始時HEAD `7dcd194`で基準と一致、clean）へ**新規ファイルのみ**で実装した。既存ファイルは変更していない。変更は未コミット。

| ファイル | 役割 |
| --- | --- |
| `self_supervised/moco/standalone_moco.py`, `__init__.py` | CIFAR ResNet18 backbone、projector、Q/K・EMA・FIFO Queue・InfoNCE |
| `conf/moco_standalone.yaml` | 独立Hydra設定（本specの数値。Gate Cは`training.max_steps=500 training.scheduler=cosine`） |
| `scripts/smoke/smoke_moco_cifar10.py` | entrypoint・2-view・SGD・checkpoint/resume・失敗記録 |
| `scripts/smoke/moco_verification_reporting.py` | 7章のartifact（JSON・PNG 4枚・Markdown） |
| `test/model/test_moco_standalone.py`, `test/model/test_moco_cifar10_reporting.py` | Gate A 23件、runner/Gate D 19件 |

- **Gate A**: PASS（CPU、23件）。spec違反を6種類入れるmutationでも、それぞれテストが失敗することを確認した
- **Gate B**: PASS（実CIFAR-10 50,000枚・20 step・CPU、`log/moco_standalone/2026-10-15/01-47-30-253616/`）
- **Gate D**: PASS（checkpoint roundtrip、resumeと中断しなかったrunのbit-exact一致、同seedでの再現、LoRA/ViT回帰32件。`pytest test`全体の失敗・エラーID集合も変更前後で一致）
- **Gate C**: 未実行（別途の実行許可が必要）。GPU・commit/pushも未実施
- 裁量内の判断・測定値・限界は[実装・検証記録](../experiments/2026-10-15-standalone-moco-v2-cifar10-mechanics-verification.md)に記録した。Comet opt-inは未実装で、`comet_enabled=true`は拒否する

Gate Cが未実施のため、statusは`approved`のままとする。
