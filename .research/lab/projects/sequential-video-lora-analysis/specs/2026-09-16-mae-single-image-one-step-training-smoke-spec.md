---
project: sequential-video-lora-analysis
spec_type: implementation
status: implemented
title: MAE単一画像1-step学習 smoke
created: 2026-09-16
last_updated: 2026-09-17
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_base_branch: mae
implementation_base_commit: bcbee263d50bb11c4c7be13decca9a6bd750102a
hf_checkpoint: facebook/vit-mae-large
---

# MAE単一画像1-step学習 smoke spec

> **Status: implemented**
>
> 本specは、単一画像forward / reconstruction loss計算まで成立したMAE実装を次段階へ進め、
> PyTorch Lightningの通常training pathを用いて1 optimization stepだけ実行し、
> reconstruction lossからMAE parameterが実際に更新されることを確認するための実装契約である。
>
> 2026-09-17のユーザー指示により承認され、同日に実装と必須検証を完了した。

## 1. 目的

`facebook/vit-mae-large`へ単一RGB画像1枚を入力し、Hugging Face MAE標準の
reconstruction lossを用いてPyTorch Lightning `Trainer.fit()`で**1 optimization stepだけ**学習し、
backwardとAdamWによるparameter updateが正常に成立することを確認する。

今回の確認経路は次である。

```text
RGB画像1枚
  -> AutoImageProcessor
  -> pixel_values [1, 3, 224, 224]
  -> one-sample Dataset / DataLoader
  -> MAELightningModule.training_step()
  -> ViTMAEForPreTraining
  -> MAE reconstruction loss
  -> Lightning automatic backward
  -> AdamW optimizer.step()
  -> encoder parameterのbefore / after差分を確認
```

今回の成功は「MAEが高品質に学習した」「lossが改善した」という科学的成功ではない。
主張できるのは、**現在のMAE実装において単一画像から標準lossを計算し、Lightningのtraining pathを通して
1 stepの勾配更新を実行できる**ことまでとする。

## 2. Authorityと基準revision

### 2.1 Authority

本specの判断根拠は、優先順に次のとおりとする。

1. 2026-09-16のユーザー指示
   - 次段階として「MAE 1-step学習確認」を行う。
   - Lightning `Trainer.fit(max_steps=1)`を使用する。
   - optimizerはAdamWとする。
   - learning rateは`1e-4`とする。
   - MAE全trainable parameterを通常更新する。
2. `2026-09-11-mae-single-image-reconstruction-smoke-spec.md`
   - `facebook/vit-mae-large`、単一RGB画像、seed 0、single GPU、FP32等の前提。
3. 実装repository `mae` branchのcommit
   `bcbee263d50bb11c4c7be13decca9a6bd750102a`。
   - `model/mae_lightning_model.py` にMAE専用LightningModuleが存在する。
   - `smoke_mae_single_image.py` に単一画像forward smokeが存在する。
4. プロジェクトREADMEと関連研究文脈。

### 2.2 実装基準

| 役割 | repository | branch | 基準commit |
|---|---|---|---|
| 研究文脈SSOT | `haruto2919/research-workspace` | `main` | spec作成時の`main` |
| 実装対象 | `tamaki-lab/2026_09_ishikawa_sequential-video-lora` | `mae` | `bcbee263d50bb11c4c7be13decca9a6bd750102a` |

本specは、前段MAE実装が存在する`mae` branchの上に追加実装することを前提とする。
実装開始時にbranch先端が基準commitから進んでいる場合は、現在コードをread-onlyで確認し、
本specのinput/output、training semantics、成功条件へ影響する差分がないことを確認する。

GitHubから確認できないlocal dirty change、未commit変更、未push commitは確認済み事実として扱わない。

## 3. 現行実装から確認できる前提

基準commitでは次が実装済みである。

```text
model/mae_lightning_model.py
  - MAELightningModule
  - facebook/vit-mae-largeのロード
  - forward(pixel_values, noise=None)

smoke_mae_single_image.py
  - RGB画像読み込み
  - AutoImageProcessor
  - seed = 0
  - single GPU / FP32
  - torch.no_grad()でのforward
  - loss / logits / mask / ids_restoreのsanity check
```

一方、`MAELightningModule`には現時点で次がない。

```text
training_step
configure_optimizers
```

したがって本specでは、既存forwardを維持したままLightningの最小training contractを追加する。

## 4. Decision Contract

### 4.1 採用条件

| 項目 | 固定値 / 方針 |
|---|---|
| checkpoint | `facebook/vit-mae-large` |
| model class | `ViTMAEForPreTraining` |
| input | 任意のローカルRGB画像1枚 |
| preprocessing | checkpoint対応`AutoImageProcessor` |
| batch size | 1 |
| seed | `0` |
| mask ratio | checkpoint標準`0.75` |
| precision | FP32 |
| device | single GPU |
| training framework | PyTorch Lightning |
| training entry | `Trainer.fit()` |
| optimization steps | **1** |
| trainable parameters | **MAE全parameter** |
| optimizer | **AdamW** |
| learning rate | **`1e-4`** |
| weight decay | **`0.0`** |
| Adam betas | `(0.9, 0.999)` |
| Adam eps | `1e-8` |
| scheduler | なし |
| gradient accumulation | なし（1 batch = 1 optimization step） |
| gradient clipping | なし |
| checkpoint saving | なし |
| external logger | なし |
| validation | なし |

### 4.2 `weight_decay = 0.0`の理由

本smokeでは「parameterが変化した」ことをreconstruction loss由来の勾配更新のEvidenceとして扱う。
AdamWのdecoupled weight decayによる変化を混在させないため、`weight_decay = 0.0`へ固定する。

この値は将来の本学習hyperparameterを決定するものではなく、1-step診断条件である。

### 4.3 learning rateの位置づけ

`1e-4`は1-step smokeでparameter updateを観測するための固定値であり、
MAE fine-tuningまたは動画学習で最適なlearning rateであるという主張はしない。

## 5. 実装設計

### 5.1 `MAELightningModule`の拡張

既存`model/mae_lightning_model.py`を拡張する。

既存の責務:

```text
__init__
forward
```

へ、次の最小責務を追加する。

```text
training_step(batch, batch_idx)
configure_optimizers()
```

#### `training_step`

役割:

1. batchから`pixel_values`を取得する。
2. 既存`forward()`を使用する。
3. `outputs.loss`を取得する。
4. lossがscalarかつfiniteであることを満たす。
5. `train_loss`としてLightningへ記録する。
6. lossをreturnし、Lightning automatic optimizationへ渡す。

MAE標準lossを再実装しない。`ViTMAEForPreTraining`が返す`outputs.loss`をそのまま利用する。

#### `configure_optimizers`

次のoptimizerだけを構築する。

```text
AdamW(
  MAE全trainable parameters,
  lr=1e-4,
  weight_decay=0.0,
  betas=(0.9, 0.999),
  eps=1e-8,
)
```

schedulerは追加しない。

### 5.2 trainable parameter契約

LoRAやfreezeはまだ導入しない。

`MAELightningModule`内の`ViTMAEForPreTraining`について、通常のpretrained load後のparameterを
すべてtrainableのままとし、今回のsmokeでは全parameterをoptimizer対象とする。

実行前に少なくとも次を確認可能にする。

```text
trainable_parameter_count == total_parameter_count
```

本specのために個別layerをfreezeしない。

### 5.3 one-sample Dataset / DataLoader

`Trainer.fit()`へ単一画像を渡すため、1 sampleだけを返す最小Dataset / DataLoaderを用意する。

データ契約:

```text
RGB image
  -> AutoImageProcessor
  -> pixel_values [1, 3, 224, 224]
  -> Datasetでは1 sample [3, 224, 224]
  -> DataLoader(batch_size=1)
  -> training_stepでは [1, 3, 224, 224]
```

Dataset / DataLoaderは今回のsmoke専用でよく、50Saladsや既存DataModuleへ統合しない。

固定条件:

```text
batch_size = 1
shuffle = False
num_workers = 0
```

DataLoaderは1 batchだけを返す。

### 5.4 1-step training smoke entrypoint

前段のforward smokeを壊さず、1-step training専用entrypointを追加する。

第一候補:

```text
smoke_mae_single_image_train.py
```

処理順:

1. CUDA availabilityを確認する。
2. `seed = 0`を固定する。
3. RGB画像1枚を読み込む。
4. `AutoImageProcessor`で`pixel_values`を作る。
5. input shape / dtypeを確認する。
6. one-sample Dataset / DataLoaderを構築する。
7. `MAELightningModule`を構築する。
8. 全parameterがtrainableであることを確認する。
9. encoderの代表parameterをbefore状態としてcloneする。
10. `Trainer(max_steps=1, ...)`で`fit()`する。
11. `trainer.global_step == 1`を確認する。
12. 同じ代表parameterのafter状態を取得する。
13. parameter deltaを確認する。
14. train lossとparameter更新結果を標準出力へ表示する。

### 5.5 Trainer契約

Trainerは少なくとも次の意味を満たす。

```text
accelerator = gpu
devices = 1
max_steps = 1
precision = 32-bit true precision
logger = disabled
checkpointing = disabled
validation = none
```

`max_steps=1`をoptimization step数のauthorityとする。

本smokeのためにepoch学習、validation loop、checkpoint callback、Comet/W&B等を追加しない。

## 6. parameter updateの検証

### 6.1 観測対象

MAE encoderに属する代表parameterを1つ固定してbefore / afterを比較する。
第一候補はViTMAE encoderのpatch projection weightとする。

概念上の対象:

```text
ViTMAEForPreTraining
  -> vit
  -> embeddings
  -> patch_embeddings
  -> projection.weight
```

実装時には現在使用中のTransformers versionで実際のnamed parameterをread-only確認し、
選択した**正確なparameter nameを標準出力または実装報告へ記録する**。

decoder-only parameterを代表parameterには選ばない。

### 6.2 before / after

1-step前に:

```text
before = target_parameter.detach().clone()
```

1-step後に:

```text
after = target_parameter.detach()
delta = after - before
```

を評価する。

成功条件:

```text
torch.equal(before, after) == False
||delta|| > 0
||delta|| is finite
```

parameter全体をcloneして比較することは必須としない。ViT-MAE-Largeで不要なメモリ消費を避ける。

### 6.3 gradient確認の扱い

本specではLightning automatic optimizationを使用し、

- `training_step`がfinite lossをreturnする。
- `Trainer.fit()`が1 global step完了する。
- weight decayを0としたAdamWでencoder parameterが変化する。

ことを必須Evidenceとする。

これらを満たせば1-step backward / optimizer update成立の確認とする。
gradient tensorを全parameterについて保存・比較することは要求しない。

## 7. 入出力契約

### 7.1 入力

```text
1 local RGB image
```

processor後:

```text
pixel_values: [1, 3, 224, 224], FP32
```

画像内容そのものは研究条件ではないため固定しない。

### 7.2 training output / evidence

最低限次を観測可能にする。

```text
checkpoint ID
resolved device
transformers version
pixel_values shape
trainable parameter count
total parameter count
training loss
loss finite
trainer global_step
observed parameter name
parameter changed: True
delta norm
delta norm finite
```

lossの絶対値、および1 step前後のloss低下はpass/fail条件にしない。

## 8. 変更範囲

原則として次だけを変更・追加する。

```text
model/mae_lightning_model.py          # training_step / optimizer追加
smoke_mae_single_image_train.py       # 新規: 1-step training smoke
model/__init__.py                     # 必要な場合のみimport/export整理
```

既存style上、smoke専用の小さなDataset helperを別fileへ置く明確な理由がある場合は許容するが、
汎用dataset abstractionやDataModuleを新設しない。

## 9. 原則変更しない範囲

```text
main_pl.py
model/simple_lightning_model.py
model/model_factory.py
model/memvit/**
dataset/**
configs/**
setup/**
smoke_mae_single_image.py  # 前段forward smokeの既存契約は維持
```

既存MeMViT classification経路、50Salads経路、sequential loader経路の意味を変更しない。

## 10. Success Criteria

本specの実装成功は、以下をすべて満たすことである。

1. 前段の`MAELightningModule.forward()`契約を維持したまま`training_step`を追加できる。
2. `configure_optimizers()`が指定条件のAdamWを返す。
3. `facebook/vit-mae-large`を正常にロードできる。
4. RGB画像1枚から`pixel_values [1, 3, 224, 224]`を生成できる。
5. DataLoaderがbatch size 1の1 batchを生成する。
6. MAE全parameterがtrainableであり、trainable countとtotal countが一致する。
7. `Trainer.fit()`がsingle GPU / FP32で開始できる。
8. `training_step`のreconstruction lossがscalarかつfiniteである。
9. `max_steps=1`で学習が終了し、`trainer.global_step == 1`である。
10. AdamWによるoptimizer stepが例外なく完了する。
11. 代表encoder parameterについてbeforeとafterが同一でない。
12. parameter delta normが`> 0`かつfiniteである。
13. model parameterにNaN / Infが発生していないことを、少なくとも更新対象parameterのsanity checkで確認する。
14. scheduler、validation、checkpoint保存、外部loggerを使用していない。
15. 既存forward smokeおよびMeMViT分類経路に不要なbreaking changeを導入していない。

## 11. 科学的成功との区別

本specでは次を主張しない。

- lossが改善した。
- MAEが1画像を学習したことでgeneralizationが向上した。
- 最適なoptimizer / learning rateを得た。
- 動画情報を学習した。
- 時間情報を獲得した。
- LoRAが有効である。
- downstream performanceが改善した。

1 step前後でlossが下がらなくても、それだけでは本spec失敗としない。

## 12. 明示的な対象外

本specでは次を実装・検証しない。

- LoRA injection
- Base MAE freeze
- LoRA-only update
- 50Salads
- `sequential_loader`
- 動画入力
- 複数frame入力
- temporal modeling
- online / continual learning
- scheduler
- warmup
- learning-rate tuning
- optimizer比較
- weight decay tuning
- gradient clipping tuning
- mixed precision
- multi-GPU / DDP
- validation
- checkpoint保存 / resume
- epoch単位の学習
- multi-step学習
- loss curve評価
- reconstruction品質評価
- scientific experiment

## 13. 再現性条件

固定:

```text
checkpoint = facebook/vit-mae-large
seed = 0
batch_size = 1
mask_ratio = 0.75
optimizer = AdamW
lr = 1e-4
weight_decay = 0.0
betas = (0.9, 0.999)
eps = 1e-8
max_steps = 1
precision = FP32
devices = 1
scheduler = none
```

実行時には可能な範囲で次を記録する。

```text
implementation commit
transformers version
resolved Hugging Face checkpoint revision
CUDA device
observed parameter name
training loss
parameter delta norm
```

## 14. Failure / Stop Conditions

次の場合は、条件を黙って変更して続行せず失敗または未検証として報告する。

- CUDAが利用できない。
- `facebook/vit-mae-large`をロードできない。
- FP32 / single GPUでOOMする。
- lossがNaN / Infになる。
- backward / optimizer stepで例外が発生する。
- `global_step`が1にならない。
- weight decay 0にもかかわらず代表encoder parameterが変化しない。
- parameter deltaにNaN / Infがある。
- current Transformers APIが本spec前提と非互換である。

特にOOM時に、勝手にMAE-Baseへ変更したりmixed precisionへ変更したりしない。
条件変更が必要ならspecを再検討する。

## 15. Ambiguity Gate

### 15.1 Blocking

**なし。**

2026-09-16のユーザー判断により、以下を採用済みとする。

- Lightning `Trainer.fit(max_steps=1)`
- AdamW
- learning rate `1e-4`
- MAE全parameter更新
- 単一RGB画像 / batch size 1
- seed 0
- single GPU / FP32
- schedulerなし
- loss低下を成功条件にしない

parameter update診断を明確にするため、本draftでは`weight_decay=0.0`を固定した。

### 15.2 Non-blocking

既存styleに従って決めてよい。

- training smoke scriptの細かなファイル名
- one-sample Dataset class/helperの名前
- CLI argument名
- `train_loss`の表示書式
- parameter deltaの表示桁数
- local RGB画像の具体的内容
- import/exportの局所的配置

これらはscope、training semantics、成功条件を変更してはならない。

## 16. 実装後の次段階

本specがimplementedになった後、次段階は別specで扱う。

第一候補:

```text
pretrained MAE encoderへLoRAを注入
  -> Base MAEをfreeze
  -> LoRAだけをtrainableにする
  -> 単一画像1 step
  -> Base不変 / LoRA更新を確認
```

その後に、`sequential_loader + 50Salads`による実動画frame入力、temporal design、online video SSLへ進む。

後続要件を本specへ先取りして実装しない。

## 17. Spec Gate

本specは、目的、基準implementation、training path、optimizer条件、更新対象、1-step条件、
parameter updateの観測方法、scope、対象外、再現性条件、success criteria、failure conditionsを固定した。

blocking ambiguityは残っていない。

statusは`implemented`である。
2026-09-17に`approved`へ変更後、`engineering-task`による実装と必須検証を完了した。
