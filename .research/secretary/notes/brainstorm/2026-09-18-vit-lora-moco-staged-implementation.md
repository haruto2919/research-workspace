---
date: 2026-09-18
project: sequential-video-lora-analysis
source_todo: null
topic: ImageNet-pretrained ViT-B16 + LoRA + MoCo の段階的実装
status: exploratory
tags: [brainstorm, research, vit, lora, moco, video, implementation-plan, baseline]
---

# ImageNet-pretrained ViT-B/16 + LoRA + MoCo の段階的実装

## 出発点

2026-09-17 MTGでは、最終方式を先に固定するより、まず1方式で動画LoRAの「学習・観測・比較」まで一巡できる開発ループを成立させることが最優先とされた。

2026-09-18の壁打ちでは、最初のbaselineとしてCLIP-ViTよりも通常のImageNet-pretrained ViT-B/16から始め、MoCo + LoRAの学習系を単純な条件で成立させた後、CLIP-ViTへ差し替える案を有力候補とした。

現在の実装repo mainでは、model_factoryの `vit_b` は `memvit` へ読み替えられ、`ClassificationBaseModel` / `SimpleLightningModel` はCrossEntropyLoss・top-k分類評価を前提としている。そのためMoCoを既存分類trainerへ直接押し込むより、自己教師あり用の独立したモデル / LightningModuleとして段階的に作る方が責務を分離しやすい。

## 基本原則

一度に追加する新要素は原則1つにする。

```text
ViT
 -> LoRA
 -> MoCo
 -> LoRA + MoCo
 -> clip化
 -> sequential loader
 -> sequential vs shuffle
 -> temporal extension
```

各段階にGateを置き、Gateを満たしてから次へ進む。

## Stage 0: 実装契約を固定する

決めるもの:

- ViT-B/16 checkpoint
- 入力解像度 / normalization
- featureとして使う位置（例: classification head前のrepresentation）
- 最初に使うMoCo variant
- queueを使うか
- LoRA挿入位置の初期候補
- SSL用LightningModuleを分類trainerから分離するか

最初のcheckpoint候補として、純粋にImageNet-1K supervised weightsを使いたい場合はTorchVisionの `ViT_B_16_Weights.IMAGENET1K_V1` が候補になる。

Gate:

- checkpointとpreprocessを再現可能な形で明記できる
- MoCo v1/v2型queueありかMoCo v3型かを区別できる
- どのtensorをcontrastive representationへ渡すか説明できる

## Stage 1: pretrained ViT単体

実装:

- ViT-B/16をロード
- 1枚の画像をforward
- classifierではなくencoder featureを取り出せるようにする
- pretrained weightsをfreezeできるようにする

確認:

- input / feature shape
- output finite
- 同一入力でeval時に再現する
- trainable parameter数が意図通り

Gate:

- ViT単体のfeature extractionが安定して動く
- 既存MeMViT分類pathと分離して動作する

## Stage 2: ViT + LoRA単体

実装:

- ViT attentionの候補箇所へLoRA注入
- Base ViT freeze
- LoRAのみtrainable
- 最小のdummy lossまたは簡単なfeature lossで1 step更新

確認:

- Base parameter: updateなし
- LoRA parameter: gradientあり / updateあり
- loss / gradient / parameterがfinite
- trainable parameter一覧を出せる

Gate:

- 「ViTは固定、LoRAだけ更新」がparameter比較で証明できる

この段階ではMoCoも動画も入れない。

## Stage 3: MoCo単体を静止画で成立

実装:

- Query encoder
- Momentum/Key encoder
- projector
- predictor（採用するMoCo variantによる）
- EMA update
- InfoNCE等のcontrastive loss
- queueを使うvariantならenqueue/dequeue

入力:

- 同じ画像に異なるaugmentationを与えた2 view

確認:

- q / k shape
- positive pair / negativeの定義
- InfoNCE finite
- Queryはgradient update
- Keyはbackwardでは更新されずEMAのみ
- queueを使う場合は内容・pointerが進む

Gate:

- 1 stepだけでなく数十step程度連続してMoCo機構が壊れず動く

この段階ではLoRAを外し、MoCoロジック自身を先に切り分けてもよい。

## Stage 4: ViT + LoRA + MoCo統合

構成:

```text
View A
 -> Query ViT(base frozen) + Query LoRA
 -> projector/predictor
 -> q

View B
 -> Key ViT(base frozen) + Key LoRA
 -> key projector
 -> k

Query LoRA: gradient update
Key LoRA: EMA follow
Base ViT: frozen
```

重要な設計点:

- EMA対象をLoRAだけにするかprojectorも含めるか
- frozen baseをquery/keyで共有するか複製するか
- key側LoRA初期値をquery側と同一にする

確認:

- Query LoRAだけoptimizerに入る
- Key側にgradientが付かない
- Key LoRAがEMAで変化する
- Base ViTは変化しない
- contrastive lossがfinite

Gate:

- 「動画なし」の状態で、LoRA + MoCoの学習機構を完全に説明・検証できる

## Stage 5: 1 video clipへ拡張

最初は時間学習を狙わず、動画を入力できることだけを確認する。

最小baseline候補:

```text
T frames
 -> 各frameを同じViT + LoRAでencode
 -> frame features
 -> order-invariant pooling（例: mean）
 -> clip feature
 -> MoCo
```

これは時間順序を失うが、意図的に「非temporal baseline」として使える。

確認:

- clip input shape
- frame -> clip featureへの変換
- 2 viewのclip augmentation
- 1 clipでcontrastive lossが計算可能

Gate:

- 静止画MoCoから動画clip MoCoへ入力契約だけを変えて学習できる

この段階では「時間情報を学習した」と主張しない。

## Stage 6: sequential loader統合

実装:

- 長時間動画からclipを時系列順に取得
- DataLoaderのsample / sub_id / frame rangeを追跡
- 将来frameへアクセスしていないことを確認
- sequence切替を明示

比較条件:

- sequential order
- random/shuffle order

確認:

- 実際のbatch順をログで検証
- 同じclip生成条件で順序だけ変えられる
- MoCo loss / LoRA updateが両条件で動く
- sequence跨ぎで不正な状態共有がない

Gate:

- 加藤研究に対応する「逐次clip vs random/shuffle clip」の最小比較が可能

## Stage 7: baseline実験

比較:

```text
A. Frozen ImageNet ViT
B. ImageNet ViT + LoRA + MoCo / random
C. ImageNet ViT + LoRA + MoCo / sequential
```

この段階の問い:

- sequential条件でも学習ループが成立するか
- random/shuffleとの差はどの程度か
- LoRA parameter trajectoryに違いがあるか

まだ「時間情報を獲得した」とは断定しない。

## Stage 8: temporal extension

baseline成立後に初めて時間順序をmodel/objectiveへ導入する。

候補:

- order-aware temporal aggregation
- ordered pair / clip objective
- past -> future feature prediction
- reverse / shuffle hard negative
- temporal offset discrimination

control:

- ordered
- shuffled
- reversed
- static-repeat

ここで初めて「時間構造を使うLoRA」と「単なるvideo domain adaptation」を比較する。

## 現時点の有力な進め方

最初の実装単位は以下の4つに切ると切り分けやすい。

1. ViT feature extractor
2. ViT + LoRA 1-step update
3. MoCo image smoke
4. ViT + LoRA + MoCo image smoke

その後にvideo clip / sequential loaderへ進む。

特に、いきなり `sequential_loader -> ViT + LoRA + MoCo` を実装すると、loss異常時にloader / clip aggregation / LoRA / EMA / queueのどこが原因か切り分けにくい。

## 現コードとの関係

現行mainの分類trainerはCrossEntropyLossとtop-kを前提にしているため、MoCo用には独立したSSL LightningModuleを作る方向が有力。

既存のMeMViT pathを壊さず、新しいViT/MoCo系を別pathで成立させる方が比較とrollbackが容易。

## 未決事項

- TorchVision ViT-B/16を使うか別実装を使うか
- MoCo v1/v2型queueありを加藤研究に合わせるか、ViT向けMoCo v3へ寄せるか
- LoRA対象をq/vにするかqkv/projまで広げるか
- projector / predictorの次元
- key側EMA対象
- clip-level baselineのpooling方法
- sequential loaderで使う動画dataset

## spec引き継ぎ候補

最初のspecは全Stageを一括にせず、Stage 1-2またはStage 1のみの小さい契約にする方が安全。

候補:

- Spec A: ImageNet-pretrained ViT-B/16 feature extraction smoke
- Spec B: ViT-B/16 + LoRA one-step update smoke
- Spec C: image-level MoCo smoke
- Spec D: ViT + LoRA + MoCo integration smoke

このメモは探索記録であり、specまたは実装許可ではない。
