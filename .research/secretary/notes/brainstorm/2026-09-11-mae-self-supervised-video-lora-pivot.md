---
date: 2026-09-11
project: sequential-video-lora-analysis
source_todo: null
topic: ImageNet事前学習MAEを用いた自己教師あり動画LoRAへの方針変更
status: exploratory
tags: [brainstorm, research, mae, lora, self-supervised-learning, video, temporal-modeling]
---

# ImageNet事前学習MAEを用いた自己教師あり動画LoRAへの方針変更

## 読み込んだ文脈

- `.research/lab/projects/sequential-video-lora-analysis/README.md`
- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-10-mtg.md`
- `.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-09-sequential-lora-finetuning-spec.md`
- `.research/secretary/notes/brainstorm/2026-09-09-video-lora-temporal-knowledge-analysis.md`
- 実装repo `tamaki-lab/2026_04_ishikawa_simple-MeMViT` の `model/model_factory.py`, `model/base_model.py`, `model/simple_lightning_model.py`, `main_pl.py`

## 方針変更

2026-09-11のユーザー指示により、従来の「50Saladsを時系列順にMeMViTへ入力し、教師ありでLoRAを逐次更新する基盤」を第一目標とする方向から、次の方向へ変更された。

1. 画像で事前学習済みのViTを出発点とする。
2. 動画の動的性を学習させることを目的に、動画を自己教師あり学習する。
3. 出発点としてHugging FaceのImageNet事前学習MAEを利用する。
4. MAE本体を基本的に保持し、LoRAを追加して動画から学習する構成を検討する。
5. 実装は段階ごとにPull Requestを分ける。

このユーザー指示は、2026-09-09のdraft specより新しく、研究方針の検討ではこちらを優先する。旧draft specは現方針を表さないため、そのまま実装契約として用いない。ただしbrainstormスキルの責務範囲ではspec自体は更新しない。

## 現在想定している作業フロー

### PR 1: Hugging Face MAEをmodel factoryへ追加

- ImageNet pretrained MAEをロードできるようにする。
- 画像のみを入力したMAE pretraining相当のforward/lossが動作することを最初のbaselineとする案。
- 既存MeMViT trainerとMAE用trainerを分けるか、共通trainerへ抽象化するかは未決。
- 動画入力時のtemporal fusionをどこに導入するかは未決。

### PR 2: LoRA追加

- Hugging Face MAEを直接改変するのではなく、wrapper/subclass等で拡張する案を検討する。
- どのprojectionへLoRAを付けるか、encoderのみかdecoderも対象にするかは未決。
- Base MAEのどこまでをfreezeするかも要決定。

### PR 3: sequential loaderの契約調査

- 辞書形式の各fieldが何を意味するかをコードとtest/outputから確認する。
- MAE/動画自己教師あり学習に実際に必要なfieldを切り分ける。

### 別途確認: クリップで学習したViT

- 緒方さんが利用している/作成した「クリップで学習したViT」の実体を確認する。
- pretraining dataset、objective、architecture、checkpoint入手方法、parameter sizeを確認する。
- ImageNet MAE baselineとの比較対象または初期値候補になり得るかを検討する。

## コード調査で確認できた事実

### model factory

現在の `model/model_factory.py` は、`vit_b`を`memvit`へ読み替えた上で `MemViT`を生成する構成で、MAEは未登録である。

### base model / trainer契約

`ClassificationBaseModel` は `CrossEntropyLoss` と `logits` を前提にしている。`SimpleLightningModel` も分類loss、top-k accuracy、frame-wise supervision、multi-head classificationを中心に構成されている。

したがって、Hugging Face `ViTMAEForPreTraining` が返すreconstruction loss/mask/reconstruction logitsをそのまま既存分類trainerへ載せることは設計上不自然であり、MAE用LightningModuleを分離するか、task abstractionを導入する必要がある。

## 外部資料から確認したこと

### Hugging Face ViTMAE

`facebook/vit-mae-base` はImageNet-1KでMAE事前学習されたViTであり、MAE pretrainingでは画像patchの高い割合をmaskし、decoderがmasked patchのpixelを再構成する。Hugging Faceでは `ViTMAEForPreTraining` として利用できる。

- https://huggingface.co/facebook/vit-mae-base
- https://huggingface.co/docs/transformers/model_doc/vit_mae

### VideoMAE

VideoMAEは画像MAEを動画へ拡張し、tube maskingと非常に高いmask ratioを用いて動画のmasked reconstructionを行う。公式実装は `[B,C,T,H,W]` の動画を入力し、spatiotemporal tokenを扱う。VideoMAE自体はvideo datasetからscratchでself-supervised pretrainingできることを主要な特徴としている。

- https://arxiv.org/abs/2203.12602
- https://github.com/MCG-NJU/VideoMAE

### image-to-video parameter-efficient adaptationの関連研究

「画像事前学習モデルをfreezeし、少数の追加parameterでvideo temporal modelingを獲得させる」という研究方向には先行研究がある。

- AIM: Adapting Image Models for Efficient Video Action Recognition (ICLR 2023)
  - 画像事前学習ViTをfreezeし、spatial/temporal/joint adaptersでvideo understandingへ適応する。
  - temporal dimensionへself-attentionを適用する構成を持つ。
  - https://arxiv.org/abs/2302.03024
- ST-Adapter: Parameter-Efficient Image-to-Video Transfer Learning
  - 画像事前学習モデルへspatio-temporal adapterを追加し、少数parameterでvideo dynamicsを扱う。
  - https://arxiv.org/abs/2206.13559

これらはMAEのmasked reconstructionそのものではないが、「image-pretrained model + parameter-efficient temporal adaptation」という研究位置づけを考える上で重要な比較対象になり得る。

## 中心となる研究上の問い

### Q1. 何をもって「動画の動的性を自己教師ありで学習した」とするか

ImageNet MAEへ動画frameを単純に1枚ずつ入力して画像再構成するだけでは、時間方向の関係を利用する必要がない。この構成でLoRAが更新されても、appearance/domain adaptationを学んだだけという代替説明が残る。

そのため、動画から動的情報を学ばせたいなら、objectiveまたはarchitectureのどちらかで複数frame間の関係を使わせる必要がある。

### Q2. temporal modelingをどこで行うか

候補:

A. Early / joint spatiotemporal modeling
- frameをまとめてtoken化し、space-time token間でattentionする。
- VideoMAEに近い。
- 動的性を直接扱いやすいが、ImageNet MAEからの構造変更が大きい。

B. Late temporal fusion
- 各frameをImage MAE encoderで処理する。
- frame-level representationを後段のtemporal moduleで融合する。
- 元のimage-pretrained encoderを保ちやすい。
- ただしMAE reconstruction objectiveとtemporal moduleをどう接続するかを新しく設計する必要がある。

C. ViT内部のattentionをtemporal directionでも再利用する
- AIMに近い考え方。
- spatial pathをimage-pretrainedのまま保ちながら、temporal pathだけLoRA/adapterで学習する候補。

現時点ではB/Cを優先して比較する価値が高い。理由は「画像で事前学習した知識を保持し、追加parameterへ動画由来情報を載せる」という研究目的と整合しやすいため。

### Q3. MAEでlate fusionをするときlossを何にするか

単純なframe-wise MAE lossではtemporal moduleを使わなくてもlossが下がる可能性がある。

候補:

- temporal moduleを経由しないとmasked frame/patchを再構成できない設計
- neighboring framesをcontextとしてcurrent frameのmasked patchを予測
- future/next-frame feature prediction
- temporal order reconstruction/prediction
- cross-frame masked modeling

ただし、これらは標準MAEから研究objectiveを変更するため、まずbaselineと研究提案を分ける必要がある。

## 初期の収束案

現時点では次の順序が最も検証しやすい。

1. **MAE integration baseline**
   - Hugging Face ImageNet-pretrained ViTMAEをmodel factoryからロードできる。
   - まず1枚の画像で標準MAE lossが計算できる。
   - このPRでは動画化・LoRA化を混ぜない。
2. **LoRA baseline**
   - MAE encoderのattentionへLoRAを追加し、Baseをfreezeして画像MAE lossでLoRAのみ更新できることを確認。
   - temporal knowledgeの主張はしない。
3. **Video temporal design**
   - late fusion / internal temporal adaptation / VideoMAE-like joint modelingを比較して、動画の時間関係をどこで使わせるか決める。
4. **Video self-supervised LoRA**
   - 選んだtemporal designで動画を自己教師あり学習し、LoRA/temporal parameterのみ更新。
5. **Evaluation**
   - normal videoとframe shuffle/static-repeat等を比較し、time order/motion利用を検証する。

## 保留・未解決事項

- MAE用trainerを既存 `SimpleLightningModel` から完全分離するか、task abstractionで共通化するか。
- Hugging Face `ViTMAEForPreTraining` をwrapperで保持するか、subclassするか。
- LoRA対象をencoder q/vだけにするか、q/k/v/proj、decoderも含めるか。
- 画像のみMAE baselineで使うdataset。
- 動画自己教師ありobjectiveの定義。
- late fusionの具体構造。
- sequential loaderを今回の動画SSLでstrict onlineに使うのか、単純clip samplingに使うのか。
- 緒方さんのclip-trained ViTの詳細。

## 次アクション候補

- trainer設計を壁打ちし、PR 1の責務境界を確定する。
- MAEにおけるlate fusionの設計候補と関連研究を詳しく調査する。
- LoRAをMAEへ追加する場合のwrapper/subclass/PEFT利用方針を比較する。
- sequential loaderの辞書出力をコード・test・具体例で調査する。

このメモは探索記録であり、specまたは実装許可ではない。
