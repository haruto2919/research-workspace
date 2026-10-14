---
project: sequential-video-lora-analysis
date: 2026-10-14
analysis_mode: explain
detail: full
level: junior
source_repository: /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
source_commit: cbc303c05502e914798e25717c4a9b29e98c8285
source_branch: dev
source_dirty: false
targets:
  - main.py
  - main_pl.py
  - train.py
  - val.py
  - conf/
  - dataset/
  - model/
  - setup/
  - logger/
  - callback/
  - utils/
  - integration/
  - scripts/smoke/
related_spec: specs/2026-10-13-standalone-vit-lora-cifar10-verification-spec.md
related_experiment: experiments/2026-10-13-standalone-vit-lora-implementation-verification.md
---

# Code Deep Dive: current dev branch codebase

## 対象と解析の範囲

- 目的: 現在の`dev`ブランチに実在するコードを、各クラス・関数の役割、Tensor shape、処理順序、背景数学、採用理由まで初学者向けに説明する。
- ソース: `tamaki-lab/2026_09_ishikawa_sequential-video-lora`のローカルcheckout、`dev`、commit `cbc303c05502e914798e25717c4a9b29e98c8285`。
- Git状態: `HEAD`、ローカル`dev`、`origin/dev`は同じcommitで、tracked/untrackedの変更はなかった。したがって、本レポートはcommitだけで解析元を特定できる。
- 研究プロジェクト: `sequential-video-lora-analysis`。関連specは`approved`、実装検証記録は`in-progress`である。
- 解析方法: ソース、設定、主要テストの静的読解。`research-code-insight`の境界に従い、import、pytest、学習、smoke、データ取得は実行していない。
- 外部依存: `sequential_loader`、Transformers、Torchvision、PyTorchVideo、Lightning、Comet内部の実装は解析対象外。呼び出し側が要求するinterfaceだけを確認した。
- 関数網羅の範囲: 実行コード、モデル、データ、utility、smoke/audit CLIに定義されたクラス・関数・メソッドを扱う。`test/`の各test関数は逐語解説せず、検証している契約をまとめる。

### 最初に理解すべきbranchの違い

2026-10-11の旧レポート`2026-10-11-dev-branch-full-explain.md`はcommit `65dedb4...`を解析しており、Streaming MoCo、Queue、Linear Probe、full pipelineを含んでいた。しかし、そのcommitを含む現在のbranchは`archive-dev-before-rework-20261012`であり、現在の`dev`には含まれない。旧commitと現`dev`のmerge-baseは`b6385d8...`である。

現在の`dev`に実在する研究経路は次の2本である。

1. frozen ViTによるSequential Loader chunkの特徴抽出。
2. Sequential LoaderやMoCoから切り離した、自作Q/K/V LoRAのCIFAR-10教師あり短時間検証。

`self_supervised/`、Streaming MoCoの`training/`、`evaluation/`、Linear Probe、`run_full_pipeline.sh`は現在の`dev`から削除されている。したがって、このレポートはMoCoの現在実装を説明しない。研究READMEに残るMoCo記録は過去の研究文脈であり、現commitの機能ではない。

## Overview: What / Why

このrepositoryには、目的が異なる3つの経路が同居している。

### 1. 通常の画像・動画分類基盤

Hydra設定からDataset、DataLoader、分類モデル、Optimizerを組み立て、Cross Entropyで学習する。通常PyTorch版`main.py`とLightning版`main_pl.py`の2入口がある。

```text
Hydra config
  ├─ dataset: CIFAR10 / ImageFolder / VideoFolder / ZeroImages
  ├─ model: ResNet / ViT / X3D / ABN / dummy
  └─ optimizer: SGD / Adam
          ↓
DataLoader → model(data, labels) → logits [B, C] + scalar loss
          ↓
backward → optimizer.step → scheduler.step
          ↓
validation → checkpoint → optional Comet upload
```

これは研究用LoRAの本体ではなく、元になった一般分類テンプレートである。

### 2. frozen ViTによる動画chunk特徴抽出

外部`sequential_loader`が返す`SequentialSample`からvalid frameだけを取り出し、画像ViTでframeごとにCLS特徴へ変換する。padding位置はゼロ行として復元し、必要ならvalid frameだけの平均を取る。

```text
frames [T, 3, H, W], uint8 CPU
valid_mask [T], bool
  ↓ validだけ選択、V = valid frame数
valid_frames [V, 3, H, W]
  ↓ AutoImageProcessor
pixel_values [V, 3, 224, 224]
  ↓ frozen ViT-B/16
valid_features [V, 768]
  ↓ 元のT位置へscatter、paddingは0
frame_features [T, 768]
  ↓ MaskedMeanClipAggregator
clip_feature [768]
```

ViTは各frameを独立に処理し、平均も順序不変である。この経路単体にはframe間attentionや時間順序モデルがない。

### 3. 自作LoRAの独立CIFAR-10検証

画像事前学習済みViTを固定し、各Transformer blockのQ/K/Vから選んだ線形射影だけに低ランク補正を加える。CIFAR-10の少量画像でLoRAと新しい10クラスHeadだけを更新する。

```text
CIFAR-10 PIL image [32, 32, 3]
  ↓ checkpoint固有AutoImageProcessor
image [3, 224, 224]
  ↓ DataLoader
images [B, 3, 224, 224], labels [B]
  ↓ frozen ViT + trainable LoRA
CLS features [B, 768]
  ↓ trainable Linear(768, 10)
logits [B, 10]
  ↓ Cross Entropy
loss []
  ↓ AdamW
LoRA A/B + classifierだけ更新
```

この独立経路は、LoRAの挿入、勾配、更新、保存・再読込を分かりやすく検証するためのものだ。動画、Sequential Loader、MoCo、時間情報学習は接続されていない。

## Tensor shape早見表

| 場所 | 入力shape | 出力shape | 主なdtype/device | 意味 |
|---|---:|---:|---|---|
| 通常画像Dataset | PIL/HWC画像 | `[3,224,224]` | float32 / CPU | crop・正規化後の画像 |
| 通常画像batch | `[B,3,224,224]` | model依存 | float32 / GPU | ResNet・ViT・ABN入力 |
| VideoFolder transform | clip `[3,T,H,W]` | `[3,16,224,224]` | float / CPU | 時間subsample済みclip |
| VideoFolder batch | `[B,3,16,224,224]` | `[B,C]` | float / GPU | X3D分類入力とlogits |
| 分類labels | `[B]` | scalar loss | int64 / GPU | class index `0..C-1` |
| 分類logits | `[B,C]` | top-k accuracy | float / GPU | softmax前score |
| SequentialSample frames | `[T,3,H,W]` | valid frames `[V,3,H,W]` | uint8 / CPU | `V≤T`、padding除外 |
| ViT processor | `[V,3,H,W]` | `[V,3,224,224]` | 通常float32 / CPU | resize・rescale・normalize |
| ViT token列 | `[V,3,224,224]` | 概念上`[V,197,768]` | float / encoder device | 196 patch + 1 CLS |
| ViT frame feature | token列 | `[V,768]` | float / encoder device | token index 0のCLS |
| restored frame feature | `[V,768]` | `[T,768]` | 同上 | padding行は0 |
| masked mean | `[T,768]`, mask `[T]` | `[768]` | 同上 | valid行だけの平均 |
| LoRA対象projection | `x [...,768]` | `[...,768]` | baseと同dtype/device | Q/K/Vの線形写像 |
| LoRA A | `[...,768]` | `[...,r]` | trainable | 低次元へ圧縮 |
| LoRA B | `[...,r]` | `[...,768]` | trainable | 元次元へ復元 |
| standalone Head | `[B,768]` | `[B,10]` | trainable | CIFAR-10 logits |

`B`はbatch size、`C`はclass数、`T`はchunkの総frame数、`V`はvalid frame数、`r`はLoRA rankである。既定値は`T=16`、`r=8`。

## Concepts and Design Decisions

### 1. 分類とCross Entropy

モデルは各classのscoreである`logits [B,C]`を出す。Cross Entropyは1 sampleについて

\[
L=-\log\frac{\exp(z_y)}{\sum_{c=1}^{C}\exp(z_c)}
\]

である。正解class `y`のlogitを他より大きくするとlossが下がる。`ClassificationBaseModel.forward`は、labelなしならlogitsだけ、labelありならlogitsとlossを返す（`model/base_model.py:27-45`）。この共通interfaceにより、ResNet、X3D、dummyを同じtrain loopで扱える。

Top-k accuracyは、最大scoreからk個に正解labelが含まれる割合である。`compute_topk_accuracy`は`logits.topk`で予測indexを取り、batch内の正解数を百分率へ変換する（`utils/accuracy.py:30-43`）。

### 2. ViTのpatchとSelf-Attention

ViT-B/16は224×224画像を16×16 patchへ分ける。

\[
14\times14=196\text{ patches}
\]

画像全体を表すためのCLS tokenを1個足すため、token数は197になる。各tokenは768次元である。

Self-Attentionの中心式は

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
\]

である。

- Q（Query）: 各tokenが何を探すか。
- K（Key）: 各tokenがどんな手がかりを持つか。
- V（Value）: attention重みによって実際に混ぜる内容。

`ViTFrameEncoder.forward`と`ViTLoRAFrameEncoder.forward`は`last_hidden_state[:,0,:]`だけを返すので、shapeは`[B,768]`になる（`model/vit/vit_frame_encoder.py:14-15`、`model/vit/vit_lora_frame_encoder.py:84-85`）。

### 3. LoRAの数学

元の線形層を

\[
y=Wx+b
\]

とする。LoRAは`W`を更新せず、低ランク行列`A`,`B`による補正を加える。

\[
y=(Wx+b)+\frac{\alpha}{r}BAx
\]

コードではdropoutも含み、実際には次である。

```python
return self.base(x) + self.scaling * self.lora_b(
    self.lora_a(self.dropout(x))
)
```

根拠: `model/vit/lora_linear.py:33-34`。

PyTorchの`nn.Linear(in,out)`のweight shapeは`[out,in]`なので、hidden size 768、rank 8なら

- `A.weight`: `[8,768]`
- `B.weight`: `[768,8]`
- 1 projectionのparameter数: `8×768 + 768×8 = 12,288`

ViT-Bは12 blockで、既定のQ/V 2 projectionへ入れるため、

\[
12\text{ blocks}\times2\text{ targets}\times12,288
=294,912
\]

がLoRAのtrainable parameter数になる。10クラスHeadは`768×10+10=7,690`なので、standalone pilotの合計trainable parameterは`302,602`である。

`A`はKaiming uniform、`B`は0で初期化される（`model/vit/lora_linear.py:30-31`）。開始時は`BA=0`なのでLoRA挿入前と出力が完全に同じになる。事前学習済みモデルの初期挙動を壊さずに追加学習を始められる。

初回backwardでは、概念的に

\[
\frac{\partial L}{\partial A}
\propto B^\top\frac{\partial L}{\partial y}x^\top
\]

であり、`B=0`なので`A`の勾配は0になる。一方、

\[
\frac{\partial L}{\partial B}
\propto\frac{\partial L}{\partial y}(Ax)^\top
\]

は一般に0ではない。1 step目でBが更新され、2 step目以降にAへも勾配が流れる。`test/model/test_lora_linear.py:10-38`はこの順序を明示的に検査している。

### 4. なぜQ/Vが既定なのか

Qを変えると「どこを見るか」、Vを変えると「何を取り出すか」を調整できる。Kを固定すると、対応付けの基準側を保ったまま適応する構成になる。承認済みspecが初期条件としてQ/Vを選んでおり、コードはQ/K/Vの任意の非空組合せを許す。

ただし、Q/Vがこの研究課題に最適だという性能証拠は現コードや未実行のStage Bからは得られない。現在確認できるのは「選択可能な実装になっている」ことだけである。

### 5. なぜBase ViTをfreezeするのか

`ViTLoRAFrameEncoder`は最初に`self.vit.requires_grad_(False)`を呼び、その後にtrainableなLoRA wrapperを挿入する（`model/vit/vit_lora_frame_encoder.py:75-82`）。これにより、Optimizer対象はLoRAとHeadに限定される。

利点:

- 更新parameterが少なく、必要メモリが減る。
- 変化の原因をLoRAへ限定しやすい。
- Baseとadapterを分けて保存できる。

代償:

- Base表現そのものを大きく作り替えられない。
- adapterだけではcheckpoint IDとライブラリ側の同じBase復元が必要になる。
- parameter効率が、表現性能や時間情報獲得を自動的に保証するわけではない。

### 6. valid frameのscatterとmasked mean

`encode_chunk`はpadding frameをViTへ通さず、valid frameだけを処理する（`integration/sequential_vit.py:12-34`）。例として

```text
valid_mask = [True, True, True, False, False]
```

ならViT入力は3 frameだけで、`valid_features [3,768]`を`frame_features [5,768]`の先頭3行へ戻し、残り2行を0にする。

`MaskedMeanClipAggregator`は

\[
c=\frac{1}{V}\sum_{t:m_t=1}f_t
\]

をboolean indexingと`mean(dim=0)`で実装する（`model/masked_mean_clip_aggregator.py:14-27`）。paddingは分子にも分母にも入らない。

平均は単純で安定し、可変長frame列を固定長へ変換できる。一方で順序不変なので、frameを逆順にしても結果は同じである。ViTもframeを独立処理するため、このfeatureだけから「時間順序をモデル化した」とは言えない。

### 7. Optimizer、scheduler、勾配蓄積

- SGD: 現在の勾配方向へ更新し、momentumで過去の更新方向を蓄積する。
- Adam/AdamW: parameterごとに勾配の1次・2次momentを持ち、scaleを調整する。
- StepLR: 10 epochごとにlearning rateを0.1倍にする。
- ConstantLR: factor 1.0なので実質的にlearning rateを変えない。

通常PyTorch loopの勾配蓄積は、複数batchで`loss.backward()`を重ねてから`optimizer.step()`する（`train.py:100-122`）。ただしlossを`grad_accum_interval`で割っていないため、「batch平均」ではなく「勾配の和」になる。さらにepoch末のbatch数がintervalで割り切れない場合、端数の勾配はstepされない。これはコードから直接読める現在の挙動である。

## Code Walkthrough

## A. 通常PyTorchの入口と学習loop

### `main.py`

#### `TqdmEpoch.__init__`（`main.py:28-37`）

resume後の`start_epoch+1`から`num_epochs`までを`tqdm`で表示する。checkpointの`current_epoch`を再実行しないための範囲設定である。

#### `prepare_training`（`main.py:40-122`）

設定から次を順番に作る。

1. Comet logger。
2. train/validation DataLoader。
3. CUDA device。`torch.cuda.is_available()`をassertするので、この通常経路はCPU fallbackを持たない。
4. class数を渡した分類モデル。
5. 任意の`nn.DataParallel` wrapper。
6. Optimizerとscheduler。
7. 勾配蓄積・logging間隔を持つ`TrainConfig`。
8. resume指定時はmodel、optimizer、scheduler、step counterをcheckpointから復元。

返り値が多いのは、`main`をorchestrationへ限定し、個別部品の作成を一か所へ集めるためである。

#### `ValidationChecker.__init__` / `should_validate`（`main.py:125-134`）

`current_epoch % interval == 0`または最終epochならvalidationを行う。最終epochを必ず評価するため、intervalの倍数でなくても実行される。

#### `main`（`main.py:137-205`）

Hydra decoratorが`conf/config.yaml`を合成する。各epochで

```text
train → 必要ならvalidation → checkpoint保存 → Comet保存
```

の順に処理する。checkpointはvalidationを行ったepochだけ保存される。

### `train.py`

#### `TrainConfig` / `TrainOutput`（`train.py:16-26`）

`TrainConfig`は勾配蓄積間隔とlog間隔、`TrainOutput`はepoch平均loss、top-1、次のglobal train stepを運ぶdataclassである。

#### `LoggerChecker.__init__` / `should_log`（`train.py:29-34`）

global stepが`log_interval_steps`の倍数かを判定する。

#### `OptimizerChecker.__init__` / `should_zero_grad` / `should_update`（`train.py:37-48`）

batch indexは1始まりである。intervalが3ならbatch 1でzero、batch 1〜3でbackward、batch 3でstepする。

#### `train`（`train.py:51-136`）

1 epochの処理を担当する。

```python
outputs = model(data, labels=labels)
loss = outputs.loss.mean()
loss.backward()
...
optimizer.step()
```

根拠: `train.py:103-122`。

- 画像は`[B,3,224,224]`、動画は`[B,3,T,224,224]`を想定する。
- `ModelOutput.logits [B,C]`からtop-1/top-5を計算する。
- `loss.mean()`はDataParallelでdeviceごとのlossが集約された場合にもscalar化するためである。
- schedulerはbatchごとではなくepoch末に1回進む。
- epoch metricはsample数で重み付けした平均である。

### `val.py`

#### `ValidationOutput`（`val.py:15-19`）

validationの平均loss、top-1、次のvalidation global stepを保持する。

#### `validation`（`val.py:22-88`）

`model.eval()`と`torch.no_grad()`で、dropout・BatchNormを評価modeにし、autograd graphを作らない。各batchで`[B,C]` logits、scalar loss、top-kを集計する。Optimizer更新は行わない。

## B. Lightning経路

### `main_pl.main`（`main_pl.py:17-71`）

CUDAを確認し、Comet logger、`TrainValDataModule`、`SimpleLightningModel`、callbackを作る。`pl.Trainer`へGPU、epoch数、勾配蓄積、logging間隔、SyncBatchNorm pluginを渡し、`trainer.fit`を呼ぶ。

通常PyTorch版との違いは、device移動、backward、distributed同期、checkpoint loopをLightningへ委譲する点である。

### `SimpleLightningModel`

#### `__init__`（`model/simple_lightning_model.py:26-62`）

Hydra設定を保存し、`configure_model`で実分類モデルを作る。`save_hyperparameters`によりcheckpointへconstructor引数を記録できる。

#### `configure_optimizers`（`model/simple_lightning_model.py:64-81`）

共通factoryでOptimizerとschedulerを作り、Lightningが理解するdictで返す。

#### `configure_callbacks`（`model/simple_lightning_model.py:83-103`）

実験名ごとの保存directoryを作り、`val_top1`が高い上位2 checkpointを保存する`ModelCheckpoint`を返す。

#### `log_train_loss_top15`（`model/simple_lightning_model.py:105-131`）

train loss、top-1、top-5をstep/epoch両方で記録する。`sync_dist=True`によりDDP worker間で同期する。

#### `training_step`（`model/simple_lightning_model.py:133-163`）

batchをmodelへ渡し、lossとtop-kを記録してlossを返す。backwardとstepはLightningが行う。

#### `log_val_loss_top15`（`model/simple_lightning_model.py:165-178`）

validation metricをepoch単位で同期・記録する。

#### `validation_step`（`model/simple_lightning_model.py:180-205`）

validation batchのlossとtop-kを計算する。`eval`、device移動、`no_grad`はLightningが管理する。

### `TrainValDataModule`

- `__init__`（`dataset/dataset_pl.py:9-22`）: 共通factoryでDataLoaderを一度作る。
- `train_dataloader`（同:24-25）: train loaderを返す。
- `val_dataloader`（同:27-28）: validation loaderを返す。
- `n_classes` property（同:30-32）: model Head作成に必要なclass数を公開する。

### `callback.configure_callbacks`（`callback/callback_pl.py:5-16`）

model階層をdepth 4まで表示する`ModelSummary`だけを返す。精度best checkpointは`SimpleLightningModel.configure_callbacks`側が追加する。

## C. Datasetとtransform

### `DataloadersInfo` / `configure_dataloader`

`DataloadersInfo`（`dataset/dataloader_factory.py:22-32`）はtrain loader、validation loader、class数をまとめる。

`configure_dataloader`（同:35-117）は`dataset_cfg.name`で分岐するfactoryである。

| name | Dataset | transform | 主なmodel入力 |
|---|---|---|---|
| `CIFAR10` | Torchvision CIFAR-10 | image transform | `[B,3,224,224]` |
| `ImageFolder` | directory class構造 | image transform | `[B,3,224,224]` |
| `VideoFolder` | PyTorchVideo | video transform | `[B,3,16,224,224]` |
| `ZeroImages` | 人工ゼロ画像 | train image transform | `[B,3,224,224]` |

### `Cifar10Info` / `cifar10`（`dataset/cifar10.py`）

`Cifar10Info`はroot、batch size、worker数、train/val transformを持つ。`cifar10`はtrain/test splitを作り、通常経路では`download=True`で自動取得する。trainはshuffleと`drop_last=True`、validationは順序固定で端数batchも残す。class数は10。

### `ImageFolderInfo` / `image_folder`（`dataset/image_folder.py`）

train/val directoryの存在を確認し、`ImageFolder`を作る。directory名からclassを決め、両splitのclass集合が同じことを検査する。返り値は2 DataLoaderとclass数。

### `VideoFolderInfo`（`dataset/video_folder.py:23-33`）

root、split名、batch設定、transform、clip秒数、validation時の1動画あたりclip数を持つ。

### `collate_for_video`（同:36-38）

PyTorchVideo sample dictをdefault collateし、共通train loopが期待する`(video, label)` tupleへ変換する。video batchは`[B,C,T,H,W]`。

### `video_folder`（同:41-105）

- train: `RandomClipSampler`と`RandomSampler`で動画・時間位置をランダム化する。
- validation: `ConstantClipsPerVideoSampler`と`SequentialSampler`で再現しやすい順序にする。
- audioはdecodeしない。
- class directory名をtrain/valで照合する。
- `LimitDataset`でiterable datasetをmap-style風に包む。

### `LimitDataset`

- `__init__`（同:119-124）: underlying iterableを2回連結したiteratorを作る。decode失敗でsample数が減る場合にも`num_videos`件を取り出せる可能性を上げる設計である。
- `__getitem__`（同:126-127）: indexは使わず、iteratorから次sampleを返す。
- `__len__`（同:129-130）: 元datasetの動画数をepoch長として返す。

これはstatefulなwrapperであり、通常のrandom-access Datasetとは性質が違う。workerごとにiterator stateを持つ点や、2回分を超えるdecode失敗には対応できない点がtrade-offである。

### `ZeroImageInfo` / `ZeroImageDataset` / `zero_images`

- `ZeroImageInfo`（`dataset/zero_images.py:13-20`）: 既定256×256、1000 sample、10 classの人工条件。
- `ZeroImageDataset.__init__`（同:24-28）: size、image size、class数、transformを保存。
- `__len__`（同:30-31）: dataset sizeを返す。
- `__getitem__`（同:33-38）: HWCゼロ画像を作り、labelを`index % n_classes`にする。transform後はCHW Tensorになる。
- `zero_images`（同:41-63）: 同じ人工Datasetからtrain/val loaderを作る。

目的はデータ内容に依存せず、pipelineのshapeや配線を確認すること。分類性能を測るデータではない。

### `TransformVideoInfo` / `transform_video`

設定dataclass（`dataset/transforms.py:9-15`）はframe数、resize範囲、crop sizeを持つ。

`transform_video`（同:32-86）はsample dictの`video` keyだけを処理する。

train順序:

1. 時間方向を16 frameへuniform subsample。
2. `[0,255]`を`[0,1]`へscale。
3. mean 0.45、std 0.225でnormalize。
4. short sideを256〜320へrandom resize。
5. 224 random crop。
6. random horizontal flip。
7. audio keyを削除。

validationではresize sizeを256に固定し、center cropを使う。trainだけrandom augmentationを入れるのは過学習を抑え、validationを決定的にするためである。

### `TransformImageInfo` / `transform_image`

設定dataclass（同:18-21）はresize 256、crop 224を持つ。

`transform_image`（同:89-121）はtrainでRandomResizedCropとhorizontal flip、validationでResizeとCenterCropを使う。どちらもImageNetのmean/stdでnormalizeする。

### `identity`（同:24-29）

値を変更せず返すdebug用hookである。

## D. 共通model interfaceとfactory

### `ModelConfig`（`model/model_config.py:15-20`）

model名、pretrained使用flag、Torch cache path、class数を持つdataclassである。

### `ModelOutput`（`model/base_model.py:10-12`）

`logits`とoptional `loss`を持つNamedTuple。train/validation/model種別間の共通interfaceになる。

### `ClassificationBaseModel`

- `__init__`（同:17-22）: config、CrossEntropyLoss、仮の`nn.Module`を保持。
- `__create_dummy_model`（同:24-25）: subclassが本物へ置換するまでのplaceholder。private methodなのでoverride前提ではない。
- `forward`（同:27-45）: `self.model(pixel_values)`で`[B,C]` logitsを作り、labelがあればCross Entropyを加える。

### `get_device`（同:48-59）

通常modelなら最初のparameter、DataParallelなら`model.module`の最初のparameterからdeviceを取得する。train loopがCPU/GPU文字列を別管理しないためのhelperである。

### `set_torch_home`（`model/model_factory.py:15-21`）

`TORCH_HOME`環境変数を設定し、Torchvision/Torch Hubのpretrained weight保存先を変える。

### `configure_model`（同:24-63）

model名からwrapperを選ぶfactory。未知名は`ValueError`。この分岐によりtrain loopは具体的architectureを知らなくてよい。

## E. 各model

### `ResNet18` / `ResNet50`

両者は同じ構造である。

- `__init__`: 共通baseを初期化し、model準備、Head交換の順に呼ぶ。
- `prepare_model`: `use_pretrained`がtrueならImageNet-1K weight、falseならrandom初期化。
- `replace_pretrained_head`: 最終`fc`を`Linear(in_features,n_classes)`へ交換。

根拠: `model/resnet/resnet18.py:12-25`、`model/resnet/resnet50.py:13-26`。

ResNetはresidual connection

\[
y=F(x)+x
\]

により深いCNNでも勾配を流しやすくする。入力`[B,3,224,224]`、出力`[B,C]`。

### `X3DM`

- `__init__`（`model/x3d/x3d.py:9-12`）: model準備とHead交換。
- `prepare_model`（同:14-20）: Torch Hubから`x3d_m`を読み、最終Softmaxを外してlogitsを直接出す。
- `replace_pretrained_head`（同:22-27）: video分類Headをclass数へ合わせる。

X3Dは空間だけでなく時間方向にも畳み込むvideo modelで、入力`[B,3,T,H,W]`、出力`[B,C]`。Softmaxを外すのは`CrossEntropyLoss`が内部でlog-softmax相当を行うためである。

### `ViTb`

- `__init__`（`model/vit/vision_transformer.py:12-14`）: model準備。
- `prepare_model`（同:16-21）: `ViTForImageClassification.from_pretrained`でViT-B/16を読み、Headをclass数へ合わせる。shape不一致Headは再初期化を許す。
- `forward`（同:23-37）: Transformers modelへpixelとlabelを渡し、共通`ModelOutput`へ変換。

注意: `ModelConfig.use_pretrained=false`でも、このclassは常に`from_pretrained`を呼ぶ。READMEの「scratch学習」説明はResNet/X3Dには対応するが、現`ViTb`では成立しない。

### `ViTFrameEncoder`

- `__init__`（`model/vit/vit_frame_encoder.py:9-12`）: `ViTModel`を読み、全parameterをfreeze。
- `forward`（同:14-15）: CLS feature `[B,768]`を返す。

これは分類Headを持たない特徴抽出器である。`add_pooling_layer=False`を指定していないためpoolerも読み込むが、forward返り値ではpooler outputを使わない。

### `LoRALinear`

- `__init__`（`model/vit/lora_linear.py:12-31`）:
  - baseが`nn.Linear`であることを要求。
  - rank、alpha、dropoutを検証。
  - 元Linear objectを保持してfreeze。
  - 同じdtype/deviceでA/Bを作る。
  - `scaling=alpha/rank`。
  - Aをrandom、Bをzero初期化。
- `forward`（同:33-34）: base出力へLoRA residualを加える。入力の先頭次元数を問わないため、`[B,D]`にも`[B,N,D]`にも使える。

元Linear objectをwrapper内部へ保持するため、weight値は失われない。ただしstate_dict keyは`q_proj.weight`から`q_proj.base.weight`のように変わるので、生ViTのstate_dictとそのまま同形ではない。

### `validate_targets`（`model/vit/vit_lora_frame_encoder.py:16-25`）

Q/K/Vの非空sequenceだけを受け付け、未知名、重複、文字列単体を拒否する。YAMLで指定した順序は保持する。

### `inject_attention_lora`（同:28-63）

重要な処理順は「全対象を検査してから、一括置換」である。

```python
for block in layers:
    # q_proj/k_proj/v_projの存在・型・shapeをslotsへ記録
...
for attention, attr, linear in slots:
    setattr(attention, attr, LoRALinear(...))
```

あるblockで失敗したとき、前半blockだけ置換済みという半端なmodelを残さないためである。次を検査する。

- `vit.layers`と`config.num_hidden_layers`が一致。
- 各blockに`attention`が存在。
- 対象が未wrapの`nn.Linear`。
- in/out featuresがどちらもhidden size。

実装は現在のTransformersが提供する`vit.layers[i].attention.q_proj/k_proj/v_proj`構造へ依存する。`requirements.txt`はTransformers versionを固定していないため、将来module pathが変わるとfail-fastする。

### `ViTLoRAFrameEncoder`

- `__init__`（同:69-82）: poolerなしViTを読み、Baseをfreezeし、targetを検証し、`enabled=true`ならLoRAを注入。
- `forward`（同:84-85）: CLS `[B,hidden_size]`を返す。
- `lora_state_dict`（同:87-94）: LoRA A/BだけをCPU cloneとして抽出。Base約86M parameterを保存しない。
- `load_lora_state_dict`（同:96-111）: key集合と各shapeの完全一致を検査してから、現在parameterのdevice/dtypeへ変換してcopyする。

完全一致を要求する理由は、target違いやlayer不足をsilent skipすると、意図と違うadapterで評価しても気づきにくいためである。

### `MaskedMeanClipAggregator`

`forward`（`model/masked_mean_clip_aggregator.py:14-27`）は`frame_features [T,D]`とbool mask `[T]`を検証し、valid行だけを平均して`[D]`を返す。all-false maskを拒否するため、空集合の平均によるNaNを防ぐ。boolean indexingはvalid featureへのgradient pathを維持する。

### `ZeroOutputModel`

- `__init__`（`model/dummy_models/zero_outout_model.py:11-17`）: parameterを1つ持たせるためのLinearを置く。
- `forward`（同:19-49）: batch sizeに合わせたゼロlogits `[B,C]`と、labelありならgradient可能な0 lossを返す。

model本体の正しさではなく、loop、logging、checkpoint配線を切り分けるdummyである。logitsは保持したLinearを通らない。

### `ABNResNet50`

- `__init__`（`model/abn/attention_branch_network.py:13-15`）: 共通baseと内部ABNを作る。
- `prepare_model`（同:17-21）: `ABNResNet50Model`を設定。
- `forward`（同:23-44）: perception branchとattention branchの2 logitsを受け、labelありなら2つのCross Entropyを加算する。推論時に返すlogitsはperception側。

名前は`ABNResNet50`だが、内部で実際に読み込むbackboneはTorch Hubの`cifar100_resnet56`である（同:55-59）。現コード上はResNet50ではない。

### `ABNResNet50Model`

- `__init__`（同:52-73）: CIFAR ResNetをfeature extractorとperception branchへ分け、attention branchを作り、最新attention map用slotを初期化。
- `__create_attention_branch`（同:75-103）: feature mapからclass別map、1-channel sigmoid attention map、補助class logits用mapを作る。
- `__split_r50_into_half`（同:105-127）: conv〜layer3をfeature extractor、avgpool〜新Headをperception branchにする。
- `__get_last_num_features`（同:129-130）: extractor末尾moduleのchannel数を得る。
- `get_attention_map`（同:132-133）: 最後のforwardで保存した`[B,1,H',W']` mapを返す。
- `__gap`（同:135-139）: Global Average Poolingで`[B,C,H',W']→[B,C]`。
- `attention_branch`（同:141-149）: class feature mapからattention mapと補助logitsを作る。
- `forward`（同:151-161）: feature `x1`へattention mapをelement-wiseに掛け、perception branchへ渡す。

数式では

\[
x_2=x_1\odot M(x_1),\quad M\in[0,1]
\]

で、注目領域を強弱付けする。補助branchにも分類lossを課すため、attention mapがclass識別に関係するよう学習される狙いがある。

## F. Optimizer、scheduler、logger、metric、checkpoint

### `configure_optimizer`（`setup/optimizer.py:10-50`）

`SGD`または`Adam`を生成する。SGDだけmomentumを使う。未知名を拒否する。

### `configure_scheduler` / `dummy_scheduler`（`setup/scheduler.py`）

- `configure_scheduler`（行9-30）: 有効なら10 epochごとに0.1倍の`StepLR`、無効ならdummy。
- `dummy_scheduler`（行33-40）: factor 1.0の`ConstantLR`。train loop側で常に`scheduler.step()`を呼べるよう、`None`分岐をなくす。

### `configure_logger`（`logger/logger.py:5-33`）

通常PyTorch用Comet Experimentを作り、日時名、model tag、解決済み設定を記録する。`disabled=true`でも同じmethod interfaceを使える。

### `configure_logger_pl`（`logger/logger_pl.py:8-39`）

Lightning用`CometLogger`と実験名を返す。model名をtagへ追加する。

### `AverageMeter`

- `__init__`（`utils/average_meter.py:16-22`）: current、sum、count、averageを0で初期化。
- `update`（同:24-40）: `sum += value*n`、`count += n`、`avg=sum/count`。batch sizeをnにすることで、最終端数batchを含むsample加重平均になる。

### `AvgMeterLossTopk`

- `__init__`（同:48-62）: loss用1個、各k用のmeterを作る。
- `update`（同:64-79）: 1 batchのlossとtop-kをbatch size重みで更新。
- `get_meters`（同:81-82）: mixinが必要なmeter群を返す。
- `get_mode_name`（同:84-85）: `train`または`val`を返す。

### `GetMetricsDictMixin`

- `get_meters` / `get_mode_name`（`utils/mixin/average_meter_mixin.py:5-9`）: subclass実装を要求する抽象的hook。
- `get_step_metrics_dict`（同:11-26）: current batch値をComet keyへ変換。
- `get_epoch_metrics_dict`（同:28-43）: epoch平均値をComet keyへ変換。

### `TqdmLossTopK`

- `__init__`（`utils/tqdm_loss_topk.py:8-10`）: postfix文字列を初期化。
- `set_postfix_str_loss_topk`（同:12-27）: step、loss、各top-kを組み立ててtqdmへ設定。
- `init_postfix_str`（同:29-30）: 文字列reset。
- `add_step_to_postfix_str`（同:32-33）: global step追加。
- `add_loss_to_postfix_str`（同:35-36）: scientific notationのloss追加。
- `add_topk_to_postfix_str`（同:38-44）: 各accuracy追加。

### checkpoint helper（`utils/checkpoint.py`）

- `remove_layer_prefix_from_state_dict`（行16-34）: DataParallelが付ける`module.`を外す。
- `add_layer_prefix_to_state_dict`（行37-54）: 通常modelのkeyへ`module.`を付け、DataParallelへload可能にする。
- `save_to_checkpoint`（行57-108）: epoch、global steps、model、optimizer、scheduler stateをlocal `.pt`へ保存。DataParallel prefixは保存時に外す。
- `save_to_comet`（行111-124）: 同じcheckpoint dictをComet modelとして記録。
- `load_from_checkpoint`（行127-164）: local fileまたは`experiment:`で始まるComet参照から読み、model/optimizer/scheduler/counterを復元する。

通常checkpointは学習再開用なのでOptimizerとschedulerを含む。standalone LoRA checkpointは配布・比較用の最小artifactなのでadapterとHeadだけを含む。この2種類は目的が違う。

## G. Sequential LoaderとViTの接続

### `encode_chunk`（`integration/sequential_vit.py:10-38`）

処理順:

1. `valid_mask [T]`から`valid_count=V`を数え、0なら停止。
2. `sample.frames[valid_mask]`で`[V,3,H,W]`を抽出。
3. processorで`[V,3,224,224]`へ変換し、shapeを検査。
4. encoder deviceへ送り、`[V,768]`を得る。
5. finiteを検査。
6. `[T,768]`のゼロTensorを同dtype/deviceで作る。
7. valid位置へfeatureをscatter。
8. padding位置が0か検査。
9. processor出力、valid feature、restored featureを返す。

返り値を3段階とも残すのは、smokeで前処理・encoder・scatterのどこが壊れたかを切り分けやすくするためである。

## H. Runnable smoke / audit scripts

### `smoke_vit_frame_encoder.main`（`scripts/smoke/smoke_vit_frame_encoder.py:15-54`）

local画像1枚をRGBへ変換し、processor後`[1,3,224,224]`、frozen ViT後`[1,768]`、finite、trainable backbone parameter 0を検査・表示する。

### `smoke_50salads_vit_bridge`

- `git_output`（`scripts/smoke/smoke_50salads_vit_bridge.py:26-29`）: 指定repositoryでGit commandを実行し、文字列を返す。
- `main`（同:32-116）: 50Salads `train1`の最初の16-frame chunkを読み、frozen ViT featureへ変換し、shape、dtype、padding、commit pinを検査する。

`EXPECTED_BRANCH`は`feature-50salads-loder`であるため、現在の`dev`でそのまま起動するとbranch checkで停止する。過去の特定branchを再現するsmokeとして残っている。

### `smoke_activitynet_vit_clip_feature`

- `git_output`（`scripts/smoke/smoke_activitynet_vit_clip_feature.py:27-30`）: Git出力helper。
- `main`（同:33-142）: ActivityNet trainingの先頭chunkを読み、`[16,768]` frame featureと`[768]` masked meanを作る。frame index/timestampが処理中に変化していないことも検査する。

これは現在の`dev`を期待するが、使うencoderは`ViTFrameEncoder`でありLoRA付きではない。

### standalone CIFAR-10 LoRA script

#### `_image_transform` / inner `convert`（`scripts/smoke/smoke_vit_lora_cifar10.py:27-30`）

AutoImageProcessorをTorchvision Dataset transformとして使えるclosureへ変換する。processorのbatch次元`[1,3,224,224]`を`squeeze(0)`し、1 sample `[3,224,224]`へする。

#### `_resolved_device`（同:33-40）

`cpu`、`auto`、利用可能な`cuda`だけを受け付ける。通常`main.py`と違いCPU実行を明示的に支援する。

#### `save_adapter_head_checkpoint`（同:43-49）

metadata、LoRA A/B、classifier stateだけを保存する。frozen ViT全体を複製しない。

#### `load_adapter_head_checkpoint`（同:52-58）

metadataのdict完全一致を要求してからadapterとHeadをstrict loadする。checkpoint ID、target、rankなどが異なる誤読込を防ぐ。

#### `_git_provenance`（同:61-74）

repositoryのHEADとdirty状態を取得する。Gitが使えなければ`unavailable`を返し、学習そのものは継続可能にする。

#### `run`（同:77-214）

現在の自作LoRA検証の中心である。

1. CIFAR-10、AdamW、正のsubset/batch/step、Comet無効を検査。
2. seedとdeviceを設定。
3. AutoImageProcessorとCIFAR-10 train splitを作る。standalone既定は`download=false`。
4. seed付き`randperm`で64 sampleを選び、batch 16のDataLoaderを作る。
5. frozen ViT + LoRAと`Linear(768,10)` Headを作る。
6. trainable parameterだけでAdamWを作る。
7. 更新前のHead、adapter、全frozen parameterをclone。
8. 20 step、`images→CLS→logits→CE→backward→step`を行う。
9. loss、batch accuracy、adapter勾配normの和、Head勾配normの和を記録。
10. Base完全不変、adapter更新、Head更新を検査。
11. adapter + Headを保存。
12. eval modeの1 sample logitsを保存前後で比較。
13. `report.json`と解決済みconfigを保存。

64 sample、batch 16なので1巡は4 batchである。20 stepではDataLoaderを5巡する。目的は高精度化ではなく、実画像と教師labelを通して有限な勾配・更新が成立するかを見ること。

#### `main`（同:217-219）

Hydraの`conf/lora_standalone.yaml`を読み、`run`へ渡す薄い入口である。

### ActivityNet inventory audit

#### `normalize_video_id`（`scripts/smoke/audit_activitynet_inventory.py:20-38`）

local filename stemの先頭`v_`だけを取り、annotation JSONのvideo IDへ合わせる。

#### `scan_root`（同:41-81）

root以下を再帰走査し、video file一覧、normalized ID→path一覧、全extension件数を返す。rootがなければ空結果。

#### `main`（同:84-350）

annotation JSONと3 local rootを比較し、root別件数、JSON一致、missing、local-only、duplicate、subset別利用可能数、例を表示する。データを変更しないinventory toolである。

## I. Hydra設定の関係

### 通常分類: `conf/config.yaml`

既定はCIFAR-10、ResNet18、SGD、batch 8、25 epoch、scheduler無効、Comet有効、CUDA必須である。Hydra overrideによりDataset・model・optimizerを交換する。

### LoRA単体: `conf/lora_standalone.yaml`

通常設定から独立している。既定値は次の通り。

| 項目 | 値 |
|---|---|
| checkpoint | `google/vit-base-patch16-224` |
| target | Q/V |
| rank / alpha / dropout | 8 / 8 / 0 |
| data | CIFAR-10 train 64枚 |
| download | false |
| batch / steps | 16 / 20 |
| optimizer | AdamW, lr 1e-3, wd 0 |
| device | CPU |
| Comet | false |

独立presetにした理由は、既存分類configやSequential Loader/MoCo条件を暗黙に変えず、LoRA機構だけを小さく検証するためである。

## Mathematical / Research Context

### 研究上、現在確認できる範囲

承認済みspecの主眼は、統合MoCoで見つかったloss振動・勾配急増からLoRA機構を切り離し、説明可能なA/B実装を検査することである。現コードは次を静的に満たす。

- Q/K/Vの任意の非空組合せ。
- Base freeze。
- A random / B zero。
- `alpha/r` scaling。
- adapter-only state。
- CIFAR-10の通常教師ありCross Entropy経路。
- LoRA + HeadだけのOptimizer。
- Baseの更新前後bit-exact比較。

これらは工学的な配線と更新対象の条件である。以下は示さない。

- 動画や時間順序の表現を獲得したこと。
- MoCo lossが安定したこと。
- PEFTやfull fine-tuningより性能が高いこと。
- CIFAR-10で十分な汎化性能があること。
- ActivityNetで有用な動画featureになったこと。

### spec・実験記録との対応差

| 項目 | 研究文書 | 現commitで観測したコード |
|---|---|---|
| Q/V、r=8、alpha=8 | approved | 一致 |
| 64枚、batch16、20step、AdamW | approved | 一致 |
| Base freeze・adapter+Head更新 | approved | 実装あり |
| adapter+Head checkpoint | approved | 実装あり |
| Stage B実CIFAR-10 run | 未実施 | entrypointあり、結果は静的解析では未確認 |
| `enabled=false`をStage Bで拒否 | 実験記録には拒否と記載 | 現`run`に拒否条件はなく、Head-only学習を許す |
| loading infoのmissing/mismatch検査 | 実験記録に記載 | 現`ViTLoRAFrameEncoder`には`output_loading_info`検査がない |
| 実装は未コミット | 2026-10-13記録時点 | 現在はcommit `cbc303c...`に含まれ、`origin/dev`と一致 |

この差は現コードの実行失敗を意味するとは限らないが、実験記録の一部が現在のmerge後sourceを正確には表していないことを意味する。

## Alternatives and Trade-offs

### LoRA wrapper vs PEFT

自作wrapperは式とparameterを追いやすく、Q/K/Vのどこへ何が入るかが明示的である。一方、Transformers内部module pathの変更、distributed/quantization対応、adapter mergeなどを自前で管理する必要がある。PEFTは対応範囲が広いが、今回の「A/Bの初期化と勾配を一行ずつ追う」目的には自作の方が教育的である。

### adapter-only保存 vs full model保存

adapter-onlyは小さく、Base不変という研究条件を表しやすい。一方、同じcheckpoint内容、Transformers実装、前処理を再現できなければ完全なmodelを復元できない。現metadataはcheckpoint IDを持つが、Base weightのhashやライブラリversionはcheckpoint内に保存しない。

### masked mean vs temporal model

masked meanはparameter不要で、padding処理とshapeが明快である。時間順序を捨てるため、動きや前後関係を扱うにはtemporal attention、RNN、temporal convolution、差分featureなどが必要になる。

### 通常PyTorch版 vs Lightning版

通常版は更新順序を直接読みやすい。Lightning版はDDP、device、logging、checkpointをframeworkへ任せられる。2経路があることで学習実装が重複し、細かな挙動がずれる可能性もある。

## 現実装を読むときの重要な注意

1. 現`dev`にMoCo、Queue、EMA、Linear Probeはない。旧レポートを現コードの説明として使わない。
2. frozen frame encoderとLoRA encoderは別classであり、ActivityNet/50Salads smokeは前者を使う。
3. `ViTb`は`use_pretrained=false`でもpretrainedを読む。
4. `ABNResNet50`の実backboneはCIFAR-100 ResNet56である。
5. 通常`main.py`と`main_pl.py`はCUDA必須。LoRA standaloneはCPU既定。
6. 通常CIFAR-10 loaderは自動downloadするが、standaloneは明示opt-inである。
7. Transformers versionが未固定で、LoRA injectionは特定のmodule構造へ依存する。
8. 通常train loopの勾配蓄積はlossをintervalで割らず、epoch端数をstepしない。
9. `encoder.train()`はBase parameterをfreezeしてもmoduleをtrain modeにする。Base内にdropout等があれば動作modeは変わり得る。parameter更新の有無とは別問題である。
10. `run`の`adapter_grad_norm_sum`は各parameter Tensorのnormを足した値であり、全勾配を連結したglobal normではない。
11. 通常train loopはDataParallelの複数lossを`mean()`するが、validation loopは`outputs.loss`をそのまま`AverageMeter.update`へ渡す。複数GPUでlossが複数要素Tensorになる場合、そこで使う`.item()`はscalarだけを受け付けるため、`trainer.use_dp=true`のvalidation経路には静的に不整合がある。

## 主要テストが保証しようとしている契約

テストは実行していないが、静的に次を確認した。

- `test_lora_linear.py`: B=0初期等価、Base freeze、1 step目B勾配、2 step目A勾配、不正hyperparameter拒否。
- `test_vit_lora_frame_encoder.py`: Q/K/V全7組合せ、12 blockへの注入数、Base不変、adapter round-trip、target不正、注入失敗時の非部分変更。
- `test_vit_lora_checkpoint.py`: adapter + Headの保存・復元、metadata不一致拒否。
- `test_vit_frame_encoder.py`: frozen CLS featureのshape。
- `test_masked_mean_clip_aggregator.py`: padding除外、shape・dtype・all-invalidのvalidation、gradient。
- `test_activitynet_vit_clip_feature.py` / `test_50salads_vit_bridge.py`: SequentialSampleからframe/clip featureへのbridge契約。
- Dataset/model/setup/utils/config tests: factory分岐、shape、metric、checkpoint prefix、Hydra compositionの回帰。

テストコードが存在することや過去記録のPASSは、この静的解析時点で現環境のPASSを意味しない。

## Learning Resources

このrepository内では、次の順に読むと理解しやすい。

1. `model/base_model.py` — 分類modelの共通interface。
2. `train.py` — forward、loss、backward、stepの最小loop。
3. `model/vit/vit_frame_encoder.py` — ViTのCLS特徴。
4. `model/vit/lora_linear.py` — LoRAの数式がほぼそのままコードになった箇所。
5. `model/vit/vit_lora_frame_encoder.py` — 12 blockへの注入とadapter I/O。
6. `scripts/smoke/smoke_vit_lora_cifar10.py` — 実画像、Head、Optimizer、保存までの全体。
7. `integration/sequential_vit.py`と`model/masked_mean_clip_aggregator.py` — 動画chunkを固定長featureへする経路。

前提概念として、`nn.Linear`のweight shape、autogradのchain rule、Cross Entropy、Self-Attention、train/eval mode、state_dictを押さえるとよい。

## Next Learning Steps

1. `LoRALinear`の4次元程度の手計算例で、B=0時の出力と初回A/B勾配を追う。
2. ViT 1 blockの`QK^T→softmax→V`を小さいtoken数で計算し、Q/K/V変更の違いを見る。
3. `smoke_vit_lora_cifar10.run`を上から追い、どのTensorだけが`requires_grad=True`か表にする。
4. `encode_chunk`の`valid_mask=[T,T,F,T]`例で、抽出とscatter後の行対応を紙に書く。
5. 実行検証を行う場合は、Stage Aのunit、Stage Bの実CIFAR-10、Sequential Loader接続を別々の主張として記録する。

## 根拠・未確認事項

### Observed

- branch、commit、clean状態、`origin/dev`一致。
- 現`dev`の全production/runnable Python symbol、Hydra設定、主要test契約。
- MoCo/Linear Probe/full pipelineが現treeに存在しないこと。
- Base freeze、LoRA式、target injection、adapter-only保存、通常分類とSequential ViT bridgeの静的data flow。
- 旧解析commitがarchive branchだけに含まれること。

### Aligned

- approved specのQ/V、rank 8、alpha 8、B zero、Base freeze、CIFAR-10 64枚・batch16・20step、AdamW、adapter+Head保存と、現コードの主要構造は論理的に対応する。

### Inferred

- Q/V既定は「注目先と集約内容を変え、Kを基準として残す」設計として解釈できる。ただしrepository内に性能比較による採用根拠はない。
- 自作LoRAは統合問題から機構を分離し、更新対象を説明しやすくするための実装と解釈できる。これはspecの目的と整合する。

### Unverified

- import可能性、pytest結果、CUDA/CPU runtime、CIFAR-10実画像20 step、loss・accuracy・gradientの実測値。
- Hugging Face checkpointの現cache内容と、外部`sequential_loader`の実装・revision・データ実体。
- LoRAが動画情報・時間情報を獲得すること、MoCo統合時の安定性、下流性能。
- 現在インストールされたTransformersで、実ViTのmodule pathがコードの期待どおりであること。
