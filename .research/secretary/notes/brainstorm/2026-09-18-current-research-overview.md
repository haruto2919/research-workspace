---
date: 2026-09-18
project: sequential-video-lora-analysis
source_todo: null
topic: 現在の研究全体像の再整理
status: exploratory
tags: [brainstorm, research, overview, vit, lora, moco, sequential-loader, framework]
---

# 現在の研究全体像の再整理

## 研究の中心目的

画像で事前学習されたViT系modelをbaseとして固定し、動画を自己教師ありで学習するときにLoRAへどのような動画由来情報、特に時間的・動的情報が入るかを解析する。

最終的には、学習済みLoRAに保持された情報をparameter / feature / downstream probe等で解析し、静的情報と動的情報の違いを理解する方向を目指す。

## 最新MTGで確定していること

2026-09-17 MTGでは、旧MeMViT教師あり逐次LoRAを主方針とせず、画像事前学習済みViT + 動画自己教師ありLoRAへ研究の中心を移した。

重要な決定:
- MAEを最終方式として固定しない。
- まず1方式で「動画LoRAを学習 -> 観測 -> 比較」まで一巡できる開発loopを作る。
- 時系列順入力とshuffle入力を比較可能にしたい。
- LoRAが時間情報を獲得したかの厳密評価は、学習loop成立後にparameter / feature / control入力を用いて設計する。
- 既存frameworkは必要部分のみ再利用し、品質が不明な過去コードをそのまま継承しない。

READMEは2026-09-13更新で旧MeMViT中心の記述が残っているため、現在方針は2026-09-17 MTGと2026-09-18 brainstormを優先して読む。

## 現時点で最有力の最初のbaseline

### Base encoder

最初はCLIP-ViTではなく、ImageNetで通常の画像分類事前学習をしたViT-B/16級を使う候補が有力。

理由:
- 最初の開発loopでCLIP固有のimage-text alignment等の変数を増やさない。
- MoCo / LoRA / sequential inputの挙動へ集中できる。
- 後から同じframeworkでCLIP-ViTへ差し替え、pretraining priorの違いを比較できる。

CLIP-ViTは棄却ではなく第2段階の比較候補。特にtext encoderをsemantic probeとして使う価値がある。

### 動画自己教師ありobjective

MoCo系を第一baseline候補とする。

加藤卒論で用いられた
- query encoder
- momentum encoder
- EMA
- queue
- InfoNCE
- sequential vs random clip比較

を研究上の参考にする。

ただし、MoCo queueが過去featureを保持すること自体はtemporal understandingを意味しない。

### 最初の主張範囲

最初のbaselineでは
「ImageNet-pretrained ViTを固定し、sequential videoをMoCo系objectiveで学習してLoRAだけを更新できる」
ところまでを成立させる。

この段階では「LoRAが時間情報を学習した」とは主張しない。

## 実装の段階

現在有力な順序:

```text
Stage 1
ImageNet-pretrained ViT
single-image feature extraction

Stage 2
external sequential_loader
 -> video clip
 -> frozen ViT
 -> frame features [B,T,D]

Stage 3
non-temporal clip representation
 -> masked mean等
 -> clip feature [B,D]

Stage 4
LoRA追加
 -> Base ViT freeze
 -> LoRAのみ1-step update確認

Stage 5
MoCo mechanics
 -> query/key
 -> EMA
 -> projector
 -> InfoNCE
 -> queue（採用variant次第）

Stage 6
sequential video + ViT + LoRA + MoCo統合

Stage 7
sequential vs random/shuffle比較

Stage 8
temporal extension
 -> reverse
 -> shuffle
 -> static-repeat
 -> past-future prediction
 -> order-aware aggregation等
```

LoRAより先にsequential_loaderとViTの接続を確認する理由は、現在の最大integration riskが「2D ViTへvideo clipをどう通すか」にあるため。

## framework方針

current Ishikawa repository / simple_cnn系の小さい構造を骨格として維持する案が有力。

### simple_cnnから維持したいもの
- model / dataset / trainerの単純な責務分離
- model_factory
- Lightning training基盤
- optimizer / logger / checkpointの基本構造
- 小さく理解しやすいdirectory構成

### Hayashi repoから参考にするもの
- external `tamaki-lab/sequential_loader` をpublic APIだけで使うconsumer boundary
- SequentialSampleの
  - frames
  - frame_indices
  - timestamps
  - valid_mask
  - is_first / is_last
- frame continuity / padding / EOF smoke

datasetやloader内部実装そのものはcopyしない。

### Saeki repoから参考にするもの
- bare `ViTModel` を用いた `ViTFrameEncoder`
- BCHW -> [B,D] のframe feature extraction
- image encoderとvideo/temporal処理の責務分離

GRU/LSTM/sliding transformer等のtemporal headは初期baselineへ持ち込まない。

## 有力なcode architecture

```text
sequential_loader
      ↓
SequentialLoaderAdapter
      ↓
SequentialBatch
      ↓
VideoEncoder
      ↓
ViTFrameEncoder
      ↑
   LoRA adapter
      ↓
ClipAggregator
      ↓
clip feature
      ↓
MoCoModel
      ↓
MoCoLightningModule
```

責務:
- LoaderAdapter: external loaderとの境界のみ
- ViTFrameEncoder: 1 frame -> featureのみ
- VideoEncoder: [B,T,C,H,W] -> [B,T,D]
- ClipAggregator: frame features -> clip feature
- LoRA: adapter注入 / freeze / parameter audit
- MoCoModel: query/key/EMA/projector/queue/loss
- MoCoLightningModule: optimization/logging/checkpoint

既存classification用SimpleLightningModelへMoCoのif分岐を大量追加しない。

## temporal learningについて

ImageNet ViTでもCLIP ViTでも、各frameを独立encodeしてmean poolingするだけではframe orderを区別できない。

したがって初期mean poolingは「時間情報を使わないvideo baseline」として意図的に使う。

その後、
- ordered vs shuffled
- ordered vs reversed
- static repeat
- order-aware temporal aggregation
- past -> future prediction

等を入れて、単なるvideo domain adaptationとtemporal information acquisitionを分離する。

## CLIPの位置付け

CLIP-ViTは第2段階の比較候補。

同じframeworkを使って

```text
ImageNet-pretrained ViT-B/16 + LoRA + MoCo
vs
CLIP-pretrained ViT-B/16 + LoRA + MoCo
```

を比較する。

CLIPではfrozen text encoderをprobeとして利用し、LoRA学習前後でsemantic representationがどう変わったかを見る候補がある。

## 現在まだ未決の項目

- 最初のViT checkpoint。
- MoCo variant: queueありMoCo v1/v2型を加藤方式に寄せるか、ViT向けMoCo v3へ寄せるか。
- clip aggregationの最初のexact仕様。
- LoRA挿入位置 / rank / alpha / dropout。
- EMA対象をLoRAのみとするかprojectorも含めるか。
- dataset。
- temporal objective。
- 最終評価metric。
- CLIPへ移るタイミング。

## 今いちばん次に決めるべきこと

研究全体の最終方式ではなく、最初のimplementation sliceを固定する。

有力な最小slice:

```text
external sequential_loader
 -> 1 video clip [T,C,H,W]
 -> ViTFrameEncoder
 -> frame features [T,D]
```

ここではLoRA / MoCo / temporal moduleはまだ入れない。

このinterfaceが成立した後、
LoRA -> MoCo -> sequential/shuffle比較
の順に追加する。

このメモは現在方針を整理した探索記録であり、specまたは実装許可ではない。
