---
date: 2026-09-17
project: sequential-video-lora-analysis
source_todo: null
topic: CLIP-ViTとMoCoを用いた自己教師あり動画LoRA方針
status: exploratory
tags: [brainstorm, research, clip, vit, moco, lora, video, self-supervised-learning, temporal-modeling]
---

# CLIP-ViTとMoCoを用いた自己教師あり動画LoRA方針

## 出発点

2026-09-17 MTGで、MAEを最終方式として固定せず、MAE・MoCo・BYOL系・CLIP ViTなどを候補として、動画LoRAに適した自己教師あり学習方式を検討する方針になった。

今回の相談では、CLIP系ViTを画像事前学習済みbackboneとして用い、MoCo型の自己教師あり学習で動画からLoRAを更新する案が、現在の研究目的に合うかを検討した。

## 研究目的との対応

現在の中心目的は、画像で事前学習されたモデルの知識を基本的に保持しながら、動画由来の時間的・動的情報をLoRAへ追加し、その内部表現・parameter変化を解析することである。

CLIP-ViT + LoRA + MoCo型動画学習は、次の点で目的と整合する。

- CLIP ViTを画像側のsemantic priorとして固定できる。
- 動画学習時にはclass labelやtext labelを使わず、自己教師ありcontrastive objectiveを使える。
- trainable parameterをLoRAへ限定すれば、動画学習で増えた情報をLoRAへ帰属させやすい。
- CLIP text encoderをfreezeして残せば、学習後のsemantic shiftやaction-direction sensitivityを評価するprobeとして利用できる。
- MoCoのmomentum encoderは、2026-09-17 MTGで議論された「遅れて追従するencoder」「移動平均」と直接対応する。

## モデル名に関する注意

`timm/vit_base_patch16_clip_224.openai_ft_in1k` のようなcheckpointを使う場合、これは「ImageNet-1KでCLIP事前学習されたViT」ではない。

- CLIP pretraining: OpenAIのWIT-400M image-text pairs
- その後: ImageNet-1Kでsupervised fine-tuning

したがって研究文書では「CLIP-pretrained ViT, optionally ImageNet-1K fine-tuned」などと区別する必要がある。

純粋なOpenAI CLIP image encoderを初期値にしたい場合は、`vit_base_patch16_clip_224.openai`のようなCLIP pretrained weightsを候補にする。

## MoCoをそのまま使う場合の問題

標準MoCo / MoCo v3は、基本的に同じ画像の異なるaugmentationをpositive pairとして扱い、augmentationに不変なrepresentationを学習する。

そのため、frameを独立画像として通常のMoCoへ入れるだけでは、時間順序を利用する必要がなく、LoRAが時間情報を学習したという主張には弱い。

さらに、近接frameを単純にpositive pairへすると、`t` と `t+1` の違いを小さくする方向に学習するため、動き・方向の差を消す可能性がある。これは「temporal consistency」を学ぶには使えるが、「temporal direction」を学ぶこととは別である。

## 有力な構成

### Stage A: 開発baseline

```text
CLIP ViT image encoder
  + LoRA
  + MoCo v3 style query / momentum-key encoder
  + projection head
```

- Base CLIP weightsはfreeze。
- Query側LoRAのみgradient update。
- Key側はquery側のLoRA / headをEMAで追従。
- まず学習ループ、EMA更新、LoRA更新、loss計算を成立させる。

この段階ではtemporal knowledgeの主張をしない。

### Stage B: temporal objectiveを導入

標準MoCoとの差分として、時間構造をlossへ入れる。

候補:

1. 同一clipのordered viewをpositiveにする。
2. shuffled / reversed clipをhard negativeまたはorder-controlとして使う。
3. past -> future feature predictionをMoCo形式のcontrastive predictionにする。
4. temporal offsetを変えたpairを利用し、近傍・遠方・reverseを区別する。
5. normal / shuffle / static-repeatを同じ学習済みLoRAへ入力してtemporal sensitivityを比較する。

特に「動作方向」まで扱う場合、隣接frameをpositiveにするだけでは不十分で、ordered pair/clipを区別できる非対称objectiveやorder-aware moduleが必要になる可能性が高い。

## 重要な反例

CLIP ViTへ各frameを独立に通し、そのfeatureを単純平均するだけでは、frame順序を入れ替えても同じ集合のfeatureになるため、ordered / shuffled / reversedを区別できない。

この場合、LoRAをViT内部に追加していても、1 frameのforwardだけでは前後関係へアクセスできない。時間順序を直接表現したいなら、少なくとも次のどれかが必要になる。

- 複数frameを同時に扱うorder-aware temporal module
- ordered frame pair / clipを対象にしたobjective
- past -> futureの非対称prediction
- sequential online update自体を研究対象にし、学習順序がLoRA parameter trajectoryへ与える影響を解析する

## MAE案との比較

### CLIP + MoCoの利点

- pixel reconstruction decoderが不要でfeature-spaceだけで自己教師あり学習を構成できる。
- MoCo v3にはViTの自己教師あり学習実装・知見がある。
- momentum encoderという明確な状態を持ち、online / sequential条件との関係を研究しやすい。
- CLIP text encoderを評価probeとして利用でき、LoRAが獲得したsemantic情報を調べやすい。

### CLIP + MoCoの注意点

- CLIP自体は純粋なlabel-free image SSLではなくimage-text supervisionで学習されている。
- ImageNet-1K fine-tuned checkpointを使う場合はsupervised label情報も初期値に含む。
- 標準MoCoではtemporal directionを保証できない。
- trainableなtemporal moduleを大きくすると、動画情報がLoRAではなくそのmoduleへ入った可能性が残る。

## 現在の方向性

### 有力候補

CLIP-ViTを画像semantic priorとしてfreezeし、LoRAのみをtrainableにし、MoCo v3型のquery / momentum-key学習を動画へ適用する方針は、現在の研究目的と整合性が高い。

ただし研究の主張は「CLIP + MoCoを使った」ではなく、

> 画像semantic priorを保持したCLIP ViTへ、動画自己教師あり学習を通じてLoRAだけを適応させたとき、時間的・動的情報がどのようにLoRAへ獲得されるか

とする方が研究目的を表しやすい。

### 必須条件

MoCoをそのままframe-levelで適用するだけで終わらず、ordered / shuffled / reversed / future prediction等を利用し、時間構造を使わないと解けない比較・objectiveを設ける。

## 最小比較案

```text
A. Frozen CLIP base
B. CLIP + LoRA + standard image-style MoCo
C. CLIP + LoRA + temporal MoCo
```

評価:

- normal order
- shuffled order
- reversed order
- static-repeat
- LoRA ON/OFF
- CLIP text prompt ranking / linear probe / feature similarity
- LoRA parameter・featureの変化

BとCを分けることで、「単なる動画domain adaptation」と「時間構造を利用したadaptation」を切り分けられる。

## 未解決事項

- raw OpenAI CLIP weightsとImageNet-1K fine-tuned CLIP weightsのどちらを初期値にするか。
- MoCo v3をどこまで再利用するか。
- LoRA挿入位置とrank / alpha / dropout。
- momentum encoderでbaseとLoRAをどのようにEMA更新するか。
- temporal positive / negativeの具体的な定義。
- temporal moduleを追加するか、LoRAだけで成立させるか。
- 使用動画datasetとsampling interval。
- online sequential updateを主研究条件にするか、clip-level temporal SSLを主条件にするか。

## 関連文献・実装

- OpenAI CLIP: https://openai.com/index/clip/
- MoCo v3: https://arxiv.org/abs/2104.02057
- MoCo v3 PyTorch: https://github.com/facebookresearch/moco-v3
- VideoMoCo: https://openaccess.thecvf.com/content/CVPR2021/html/Pan_VideoMoCo_Contrastive_Video_Representation_Learning_With_Temporally_Adversarial_Examples_CVPR_2021_paper.html
- Video Contrastive Learning With Global Context: https://openaccess.thecvf.com/content/ICCV2021W/CVEU/html/Kuang_Video_Contrastive_Learning_With_Global_Context_ICCVW_2021_paper.html
- CACL: https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Cross-Architecture_Self-Supervised_Video_Representation_Learning_CVPR_2022_paper.html

## 関連Research Workspace文書

- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-17-mtg.md`
- `.research/secretary/notes/brainstorm/2026-09-11-clip-pretrained-vit-for-video-lora.md`
- `.research/secretary/notes/brainstorm/2026-09-17-clip-mae-reverse-motion-direction-evaluation.md`

このメモは探索記録であり、specまたは実装許可ではない。
