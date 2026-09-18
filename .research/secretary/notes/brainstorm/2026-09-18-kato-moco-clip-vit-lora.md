---
date: 2026-09-18
project: sequential-video-lora-analysis
source_todo: null
topic: 加藤卒論のMoCoをCLIP-ViT+LoRAへ再利用する案
status: exploratory
tags: [brainstorm, research, moco, clip, vit, lora, video, online-learning, contrastive-learning]
---

# 加藤卒論のMoCoをCLIP-ViT+LoRAへ再利用する案

## 出発点

2026-09-17 MTGでは、MAEを最終方式として固定せず、MoCo/BYOL系/CLIP ViT等を候補として動画LoRAの自己教師あり学習方式を検討する方針になった。特に、加藤さんの研究パターンを参考にすることが議論された。

今回、加藤さんの卒論発表スライド「長時間動画のオンライン事前学習方法の検討」に記載されたMoCo構成を、CLIP事前学習済みViT + LoRAへ活用できるかを検討した。

## スライドから確認できたMoCo構成

スライドp.9では次の構成が示されている。

- 入力は1クリップ。
- 同一クリップへ異なるデータ変換を適用。
- 一方を通常Encoder、もう一方をMomentum Encoderへ入力。
- 両者の現在特徴をpositive pairとする。
- Momentum Encoderから過去に得た特徴をqueueへ保持し、negativeとして利用。
- InfoNCEを最小化。
- Momentum EncoderはEMAで更新。

スライドp.11では、対照学習encoderとしてX3D-M、momentum=0.99、queue size=256、temperature=0.3が記載されている。

スライドp.14では、長時間動画からrandom clipとsequential clipを作る比較が定義されている。p.16ではcontrastive learningでrandom/sequentialのpretraining lossに大きな差がなく、stride 64のsequential clipがdownstreamでrandomと同程度。p.18では3方式の中でcontrastive learningが最も効果的だったとまとめられている。

## 現研究への適合性

このMoCoの「学習機構」は現在の研究へ再利用しやすい。

候補構成:

```text
sequential video clip
  ├─ augmentation A -> Query: CLIP ViT + LoRA -> projector -> q
  └─ augmentation B -> Key:   CLIP ViT + LoRA -> projector -> k
                                               ^
                                               EMA

past keys -> queue -> negatives
q / k / queue -> InfoNCE
```

- CLIP base weightsはfreeze。
- Query側LoRAをgradient update。
- Key側LoRAはQuery側をEMA追従。
- projectorは必要に応じて学習。
- sequential loaderを用いて長時間動画を時系列順に供給する。

これは「画像事前学習済みモデルを保持し、動画だけでLoRAへ自己教師あり適応を行う」という研究目的と整合する。

## 重要な差分: X3D-MとCLIP ViT

加藤さんの構成ではX3D-Mがencoderであり、X3D-M自身が複数frameを動画clipとして処理して時間方向をモデル化できる。

一方、標準CLIP ViTは2D image encoderである。各frameを独立にCLIP ViTへ入れて単純平均すると、frame orderを入れ替えても同じrepresentationになり得る。

したがって、MoCo部分だけをそのまま移植しても、CLIP ViT + LoRAが時間順序を学ぶとは限らない。

必要な追加条件の候補:

1. frame featuresへorder-aware temporal aggregationを追加する。
2. 複数frame tokenを同一attentionで扱えるvideo adaptationを導入する。
3. ordered pair / future predictionなど、時間方向をlossへ入れる。
4. sequential update順序がLoRA parameter trajectoryへ与える影響を研究対象として分離する。

## MoCo queueに関する注意

queueに「過去の特徴量」が入ること自体は、過去動画をtemporal contextとして理解していることを意味しない。元のMoCoではqueueはnegative dictionaryであり、過去との時間関係を予測するmemoryではない。

したがって、

- online/sequentialに学習できる
- temporal relationをrepresentationとして学習する

を分けて評価する必要がある。

## 最小baselineとしての有力案

まず加藤さんの方式に近い構成をbaselineとして再現する価値が高い。

```text
A. Frozen CLIP ViT
B. CLIP ViT + LoRA + Kato-style MoCo
C. B + temporal/order-aware extension
```

Bでは「動画を逐次的にMoCoで自己教師あり学習し、LoRAのみ更新できる」ことまでを確認し、時間情報獲得は主張しない。

Cでordered/shuffled/reversed/static-repeat等を用い、時間情報を獲得したかを検証する。

## 現時点の方向性

### 有力

- 加藤さんのMoCoのquery/momentum encoder + queue + InfoNCEという学習機構を、CLIP ViT + LoRAへ置き換えて最初の動画LoRA baselineにする。
- sequential clipとrandom/shuffle clipの比較を引き継ぐ。
- ViT実装の参考としてMoCo v3も確認する。

### 未決

- clip representationをCLIP ViTでどう作るか。
- temporal moduleを入れるか、CLIP ViT内部をvideo対応させるか。
- EMA更新対象をLoRAだけにするかprojectorも含めるか。
- queue size / momentum / temperatureを加藤設定から引き継ぐか再調整するか。
- Bをbaselineとして実装した後、Cのtemporal objectiveを何にするか。

## 関連文書

- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-17-mtg.md`
- `.research/secretary/notes/brainstorm/2026-09-17-clip-vit-moco-video-lora.md`

このメモは探索記録であり、specまたは実装許可ではない。
