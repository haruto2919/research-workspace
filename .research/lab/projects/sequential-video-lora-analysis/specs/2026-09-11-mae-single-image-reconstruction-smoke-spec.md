---
project: sequential-video-lora-analysis
spec_type: implementation
status: draft
title: MAE事前学習モデル統合と単一画像reconstruction loss smoke
created: 2026-09-11
last_updated: 2026-09-11
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
workspace_base_commit: 723e4c462f03befc0e3493114c7c500b2b851f2d
implementation_repository: tamaki-lab/2026_04_ishikawa_simple-MeMViT
implementation_base_branch: main
implementation_base_commit: e0deb093694d367ed9b02065e6d4cd38802093d6
hf_checkpoint: facebook/vit-mae-large
---

# MAE事前学習モデル統合と単一画像reconstruction loss smoke spec

> **Status: draft**
>
> 本specは、MAEを現在の研究実装基盤へ追加する最初の実装契約を定義する。
> `approved` へ変更されるまでは、研究コードの変更・実装開始を許可しない。
> 本specは既存の逐次LoRA draft specを自動的に置き換えたり、承認したりしない。

## 1. 目的

Hugging Faceで公開されているImageNet事前学習済みMAE
`facebook/vit-mae-large` を現在の研究コードへ最小構成で導入し、MAE専用の
PyTorch Lightning `LightningModule` を通して、**単一RGB画像1枚から標準MAE forwardを実行し、
reconstruction lossを正常に取得できること**を確認する。

今回の成功条件は「MAEで学習できること」ではなく、次の最小経路が成立することである。

```text
RGB画像1枚
  -> checkpoint対応image processor
  -> pixel_values [1, 3, 224, 224]
  -> MAE専用LightningModule
  -> ViTMAEForPreTraining
  -> 75% patch masking
  -> decoder reconstruction
  -> finite scalar reconstruction loss
```

この確認を、後続のLoRA注入、動画frame入力、sequential learning、temporal modelingへ進む前の
独立した基盤確認とする。

## 2. Authorityと基準revision

### 2.1 Authority

本specの判断根拠は、優先順に次のとおりとする。

1. 2026-09-11のユーザー指示
   - MAEを研究方針として採用する。
   - `facebook/vit-mae-large` を使用する。
   - MAE専用LightningModuleを作る。
   - 単一画像でreconstruction lossを確認する。
   - `seed = 0`、single GPU、Lightning `Trainer`を使わず`forward`を直接smokeする。
   - 今回はbackward / optimizer step / parameter updateを行わない。
2. `.research/secretary/notes/brainstorm/2026-09-11-mae-vs-clip-direction.md`
   - MAEを主軸とし、最初にpretrained MAE integrationと単一画像reconstruction loss確認を行う方針。
3. プロジェクト `README.md` と既存の研究文脈。
4. 下記基準revisionの現行コード。

brainstormは探索記録であり、それ単独では実装許可ではない。本specが現在要求を実装可能な契約へ
固定する役割を持つ。

### 2.2 固定するrepositoryとrevision

| 役割 | repository | branch | 基準commit |
|---|---|---|---|
| 研究文脈SSOT | `haruto2919/research-workspace` | `main` | `723e4c462f03befc0e3493114c7c500b2b851f2d` |
| 実装対象 | `tamaki-lab/2026_04_ishikawa_simple-MeMViT` | `main` | `e0deb093694d367ed9b02065e6d4cd38802093d6` |

実装開始時にはremote `main` の先端を再確認する。基準commitから進んでいる場合は、本specの
前提・public interface・変更範囲・成功条件へ影響しないことをread-onlyで確認してから実装する。
GitHubから確認できないローカルdirty changeや未push commitは、本specの権威ある入力として扱わない。

### 2.3 外部checkpoint

使用するモデルIDを次で固定する。

```text
facebook/vit-mae-large
```

Transformersの公開APIから次を使用する。

```text
ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-large")
AutoImageProcessor.from_pretrained("facebook/vit-mae-large")
```

checkpointの現行configで本specが依存する契約は次である。

| 項目 | 値 |
|---|---:|
| architecture | `ViTMAEForPreTraining` |
| image size | 224 |
| patch size | 16 |
| channels | 3 |
| mask ratio | 0.75 |
| encoder hidden size | 1024 |
| encoder layers | 24 |
| decoder layers | 8 |
| norm pixel loss | false |

Hugging Face repositoryの完全なrevision SHAは本specでは固定しない。実装・smoke実行時には、
実際に解決されたcheckpoint revisionと`transformers` versionを実装報告へ記録する。
モデルIDや上記主要configが変化している場合は、黙って別条件で実行せず本specとの差分として報告する。

## 3. 現行実装から確認できる前提

基準commitの現行コードでは、通常のLightning実行は `main_pl.py` から
`SimpleLightningModel` を構築し、MeMViTをclassificationモデルとして扱う構成である。
`SimpleLightningModel` はCrossEntropy、Top-k metric、frame-wise supervision、MeMViTのonline memoryなど、
既存分類タスクの責務を持つ。

また、`model/model_factory.py` の現行factoryは実質的にMeMViTを構築する経路であり、MAE用経路はない。
`requirements.txt` には既に `transformers[torch]` が含まれている。

したがって本specでは、既存`SimpleLightningModel`へMAE固有分岐を追加せず、MAEの最小責務を持つ
専用LightningModuleを独立させる。

## 4. Decision Contract

### 4.1 採用する内容

| 項目 | 採用内容 |
|---|---|
| model checkpoint | `facebook/vit-mae-large` |
| model class | `ViTMAEForPreTraining` |
| preprocessing | checkpoint対応 `AutoImageProcessor` |
| input | ローカルの任意RGB画像1枚 |
| batch size | 1 |
| expected input | `pixel_values [1, 3, 224, 224]` |
| mask ratio | checkpoint標準 `0.75` |
| seed | `0` |
| precision | FP32 |
| execution | single process / single GPU |
| Lightning | MAE専用 `LightningModule` を新設 |
| smoke path | `LightningModule`を直接instantiateし、`forward`を直接呼ぶ |
| Trainer | 使用しない |
| optimization | 行わない |
| backward | 行わない |
| parameter update | 行わない |
| scientific claim | 行わない。最小forward成立確認のみ |

### 4.2 明示的な対象外

本specでは次を実装・検証しない。

- `training_step`を用いた学習loop
- `configure_optimizers`
- optimizer / scheduler
- `loss.backward()`
- `optimizer.step()`
- parameter更新確認
- `Trainer.fit()` / `Trainer.validate()`
- checkpoint保存・resume
- LoRA注入
- Base MAE freeze / LoRA-only update
- 50Salads
- `sequential_loader`
- 動画decode・動画frame入力
- 複数frame入力
- temporal modeling
- causal past fusion / cache / sequence reset
- MeMViTとの統合
- classification head / action label / frame-level metric
- reconstruction画像の視覚品質評価
- reconstruction lossの性能比較
- full training / long run / multi-seed experiment
- CPU動作保証
- multi-GPU / DDP

## 5. 実装設計

### 5.1 MAE専用LightningModule

新しいLightningModuleは、既存`SimpleLightningModel`から分岐せず独立して実装する。
第一候補の配置は次とする。

```text
model/mae_lightning_model.py
```

責務は本specで必要な最小限に限定する。

```text
MAELightningModule
  __init__
    - checkpoint IDを保持
    - ViTMAEForPreTraining.from_pretrained(...)でpretrained MAEを構築

  forward(pixel_values, noise=None)
    - Hugging Face MAEへ入力を委譲
    - ViTMAEForPreTrainingOutputを返す
```

`forward`の中で`torch.no_grad()`を強制しない。将来の学習拡張で同じforwardを再利用できるよう、
勾配を切る責務はsmoke側に置く。

本specでは`training_step`、`validation_step`、`configure_optimizers`を要求しない。
Lightning `Trainer`へ渡して学習可能であることも成功条件に含めない。

必要であれば`model/__init__.py`から新moduleをexportする。ただし既存MeMViTのfactory contractを変更するためだけに
`model_factory.py`へMAE分岐を追加しない。

### 5.2 単一画像smoke entrypoint

単一画像確認専用の短いentrypointを用意する。第一候補の配置は次とする。

```text
smoke_mae_single_image.py
```

entrypointはローカル画像パスを受け取り、次の処理だけを行う。

1. CUDAが利用可能であることを確認する。
2. seedを`0`へ固定する。
3. ローカル画像を読み込み、RGBへ変換する。
4. `AutoImageProcessor.from_pretrained("facebook/vit-mae-large")` で前処理する。
5. `pixel_values` が `[1, 3, 224, 224]` であることを確認する。
6. `MAELightningModule` を構築し、1 GPUへ移す。
7. moduleを`eval()`へ切り替える。
8. `torch.no_grad()`下で`forward(pixel_values)`を1回だけ実行する。
9. `loss`、`logits`、`mask`、必要に応じて`ids_restore`のshapeと有限性を確認する。
10. 確認結果を標準出力へ表示し、成功時に正常終了する。

Comet等の外部logger、dataset、DataLoader、Lightning `Trainer`は使用しない。

### 5.3 seed契約

maskingの乱数を再現しやすくするため、smoke開始時にseed `0`を設定する。
Lightningを利用しているため、既存環境で利用可能であれば `pl.seed_everything(0, workers=True)` 相当を使用してよい。
重要なのは、**同一環境・同一checkpoint・同一入力画像でmasking乱数を固定して再実行可能にすること**である。

成功条件として特定のloss数値そのものは固定しない。

### 5.4 single GPU契約

- 1 process / 1 GPUだけを使用する。
- 実際のCUDA device indexは既存の`CUDA_VISIBLE_DEVICES`運用を尊重する。
- smoke内部では可視GPUの`cuda:0`を使用してよい。
- DDP、distributed sampler、SyncBatchNorm等は使用しない。
- CPU fallbackは設けない。CUDAが使用できない場合はfail-fastする。
- mixed precisionは使用せず、FP32で確認する。

## 6. 入出力契約

### 6.1 入力

```text
1 RGB image
  -> AutoImageProcessor
  -> pixel_values: torch.Tensor [1, 3, 224, 224]
```

入力画像の内容は研究条件ではないため固定しない。JPEG/PNG等、PILでRGBとして読み込めるローカル画像を
1枚使用する。画像ファイル自体をrepositoryへ追加することは必須としない。

### 6.2 MAE出力

224×224画像を16×16 patchへ分割するため、patch数は

```text
(224 / 16) * (224 / 16) = 196
```

である。

本checkpointではpatchごとのreconstruction target dimensionは

```text
16 * 16 * 3 = 768
```

であるため、batch size 1で次を期待する。

```text
loss: scalar tensor
logits: [1, 196, 768]
mask: [1, 196]
ids_restore: [1, 196]
```

`mask_ratio = 0.75`では、196 patchのうち147 patchがmasked、49 patchがvisibleとなることをsanity checkする。

## 7. 変更範囲

### 7.1 変更を許可する範囲

実装時に必要な変更は原則次へ限定する。

```text
model/mae_lightning_model.py   # 新規: MAE専用LightningModule
model/__init__.py              # 必要な場合のみexport追加
smoke_mae_single_image.py      # 新規: 単一画像forward smoke
```

既存style上より適切な同等配置が明確な場合、ファイル名・import整理はnon-blocking実装詳細として調整してよい。
ただし責務とscopeは変えない。

### 7.2 原則変更しないもの

```text
main_pl.py
model/simple_lightning_model.py
model/model_factory.py
model/memvit/**
dataset/**
configs/**
setup/**
```

また、現行`requirements.txt`には`transformers[torch]`が存在するため、今回の機能のためだけのdependency追加は
原則行わない。基準環境で必要classをimportできない場合は、勝手に大きなdependency migrationを行わず、
実装報告でversion差分を示す。

## 8. Success Criteria

本specの**実装成功**は、以下をすべて満たすこととする。

1. `MAELightningModule` を例外なくimport / instantiateできる。
2. `facebook/vit-mae-large` pretrained weightをロードできる。
3. checkpoint対応processorでRGB画像1枚から `pixel_values [1, 3, 224, 224]` を生成できる。
4. single GPU / FP32で `MAELightningModule.forward()` を直接1回実行できる。
5. `outputs.loss` が0次元scalar tensorである。
6. `outputs.loss` がNaNでもInfでもなくfiniteである。
7. `outputs.logits.shape == [1, 196, 768]` を確認できる。
8. `outputs.mask.shape == [1, 196]` を確認できる。
9. maskされたpatch数が147、visible patch数が49であることを確認できる。
10. `outputs.ids_restore.shape == [1, 196]` を確認できる。
11. smoke中にbackward、optimizer step、parameter updateを実行していない。
12. Lightning `Trainer`、DataLoader、50Salads、sequential_loaderを使用していない。
13. 既存MeMViT分類経路へ不要なbreaking changeを導入していない。

### 8.1 科学的成功とは区別する

以下は本specの成功条件ではない。

- reconstruction lossが低いこと
- reconstruction画像が人間にとって高品質であること
- MAEが動画情報を獲得したこと
- LoRAが有効であること
- 時間情報を学習したこと
- downstream精度が向上したこと

本specで主張できるのは、**事前学習済みMAEを研究コード上でロードし、単一画像に対する標準MAE
reconstruction loss計算経路が成立した**ことまでである。

## 9. 検証手順

実装後は、長時間runではなく1回の短いsmokeだけを行う。

概念的な実行は次とする。

```text
CUDA_VISIBLE_DEVICES=<one_gpu> \
python smoke_mae_single_image.py --image-path <local_rgb_image>
```

標準出力には少なくとも次を確認できる情報を出す。

```text
checkpoint
resolved device
pixel_values shape
loss
loss finite check
logits shape
mask shape
masked patch count
visible patch count
ids_restore shape
```

lossの絶対値をpass/fail閾値にはしない。

## 10. 互換性と非回帰

本specは既存MeMViT経路と並列の最小MAE経路を追加するものであり、既存の
`SimpleLightningModel`、MeMViT classification、50Salads dataloader、逐次学習挙動を変更しない。

そのため今回の実装で既存classification baselineを再設計しない。既存経路のimportを壊していないことを
最低限確認する。

## 11. Ambiguity Gate

### 11.1 Blocking

**なし。**

以下は2026-09-11のユーザー判断で確定した。

- `facebook/vit-mae-large`
- MAE専用LightningModule
- 単一画像forward / reconstruction loss確認
- `seed = 0`
- single GPU
- Lightning `Trainer`を使わず直接`forward`
- backward / optimizer / parameter updateは対象外

### 11.2 Non-blocking

以下は研究主張や成功条件を変えない局所的な実装詳細として、既存styleに合わせてよい。

- MAE module / smoke scriptの細かなファイル名
- import/export配置
- CLI argumentの細かな命名
- smokeに使うRGB画像の内容
- 標準出力のラベル名
- checkpoint cacheの実体パス

これらの調整で本specの責務境界、input/output契約、対象外、success criteriaを変更してはならない。

## 12. 実装後の次段階

本specがimplementedになった後、次は別のspecとして段階的に進める。

```text
本spec
MAE load + single-image reconstruction loss
  -> 次段階1: 1 step backward / parameter updateの確認
  -> 次段階2: MAE encoderへのLoRA注入 + Base freeze + LoRA-only update
  -> 次段階3: sequential_loader + 50Salads frameをMAEへ入力するsmoke
  -> 次段階4: temporal designの確定
  -> 次段階5: online video self-supervised learning
```

後続段階の要件を先取りして本specへ実装しない。

## 13. 外部参照

- Hugging Face model: `https://huggingface.co/facebook/vit-mae-large`
- Hugging Face config: `https://huggingface.co/facebook/vit-mae-large/blob/main/config.json`
- Hugging Face preprocessor config: `https://huggingface.co/facebook/vit-mae-large/blob/main/preprocessor_config.json`
- Transformers ViTMAE documentation: `https://huggingface.co/docs/transformers/model_doc/vit_mae`

## 14. Spec Gate

本specは、目的、scope、対象外、外部checkpoint、実装責務、input/output契約、再現性条件、
GPU条件、成功条件、非回帰条件まで固定済みであり、blocking ambiguityは残っていない。

ただし現在のstatusは`draft`である。ユーザーが本spec本文を確認し、`approved`への変更を明示した後にのみ、
`engineering-task`へ引き継いで研究コード実装を開始できる。
