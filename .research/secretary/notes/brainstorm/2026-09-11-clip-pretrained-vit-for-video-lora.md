---
date: 2026-09-11
project: sequential-video-lora-analysis
source_todo: null
topic: CLIP事前学習ViTの動画LoRA研究への活用
status: exploratory
tags: [brainstorm, research, clip, vit, lora, video, temporal-modeling, vision-language]
---

# CLIP事前学習ViTの動画LoRA研究への活用

## 出発点

先生とのMTGで、CLIPで事前学習済みのViTを利用する案が出た。
現在の研究では、画像で事前学習したモデルを基本的に固定し、LoRAを用いて動画から時間的・動的情報を追加学習し、LoRAに何が保持されるかを解析する方向を検討している。
直近の探索ではImageNet事前学習MAEを初期モデルとし、動画自己教師あり学習、causal temporal modeling、ActivityNet / EPIC-KITCHENS比較を検討している。

## 確認済み事実

### CLIPの事前学習

CLIPは画像エンコーダとテキストエンコーダを、対応する画像・テキストpairが近く、非対応pairが遠くなるようにcontrastive learningで事前学習する。
CLIPのViT画像エンコーダは、単なる固定クラス分類器ではなく、テキストと比較可能な意味的visual representationを持つ。

重要な区別として、CLIPはMAEと同じ意味での「ラベルなし画像自己教師あり学習」ではない。クラスラベルは使わないが、画像と自然言語のpairをsupervisionとして利用するnatural-language supervision / multimodal contrastive pretrainingである。

### image-to-video adaptationの関連研究

- AIM: Adapting Image Models for Efficient Video Action Recognition
  - image-pretrained transformerをfreezeし、少数のadapterでspatial / temporal adaptationを行う。
  - temporal modelingではimage-pretrained self-attentionを時間軸へ再利用する。
- Frozen CLIP Models are Efficient Video Learners (EVL)
  - frozen CLIP image encoderのframe-level spatial featureを使い、軽量decoderとtemporal moduleでvideo recognitionへ適応する。
- ActionCLIP
  - CLIPのimage-text semantic spaceをvideo action recognitionへ拡張し、video-text matchingとしてaction recognitionを扱う。

したがって、「CLIP画像encoderを固定して、小さな追加parameterだけでvideoへ適応する」方向には明確な先行例がある。

## MAEとの違い

### MAE

- pretraining input: image
- supervision: mask前の元pixel
- objective: masked patch reconstruction
- 強み: ラベルなしvisual self-supervised learningを構成しやすい
- decoderが存在するため、動画でcurrent-frame masked reconstructionなどへ拡張しやすい
- テキスト意味空間は持たない

### CLIP

- pretraining input: image-text pair
- supervision: pair correspondence
- objective: image-text contrastive alignment
- 強み: action / object / sceneなどを自然言語と比較できるsemantic space
- 標準CLIP image encoderにはMAE decoderがないため、MAE reconstruction objectiveはそのまま使えない
- 動画へ使うにはtemporal moduleまたはvideo-level aggregation/objectiveが必要

## 今回の研究で使える方向

### Candidate A: CLIPを初期画像モデルとして使い、動画だけでLoRAを自己教師あり適応する

構成例:

```text
frame sequence
  -> frozen CLIP ViT + LoRA
  -> frame features
  -> causal temporal modeling
  -> video self-supervised objective
```

動画学習時にはテキストを使わず、例えば次をobjective候補とする。

- past frame featuresからcurrent / next frame CLIP featureを予測
- masked temporal feature prediction
- temporal consistency / contrastive learning
- normal orderと時間的に近いframeをpositiveとして扱うcontrastive objective

この場合、CLIPのsemantic image priorを初期値として保持しつつ、LoRAへvideo-specific / temporal informationを追加できるかを調べられる。
ただしこれはMAE lossではなく、別のvideo self-supervised objectiveを設計する必要がある。

### Candidate B: CLIPのtext encoderを評価用probeとして使う

これは今回の研究と特に相性が良い。

```text
video
 -> CLIP ViT + learned LoRA + temporal aggregation
 -> video embedding

text prompt
 -> frozen CLIP text encoder
 -> text embedding

cosine similarity
```

例えばEPIC-KITCHENSなら、

- "cutting an onion"
- "opening a drawer"
- "taking a cup"

などのpromptとvideo representationの類似度を見る。

学習前CLIP、ActivityNetで学習したLoRA、EPIC-KITCHENSで学習したLoRAを同じtext spaceで比較できるため、「LoRA追加後にどのsemantic conceptへ近づいたか」を解析しやすい。

さらに normal / shuffled / reversed / static-repeat を比較し、action promptへの類似度が時間構造を壊したときにどの程度変わるかを見ることで、temporal sensitivityを評価する候補になる。

### Candidate C: video-text contrastive learningでLoRAを学習する

動画側をCLIP ViT + temporal moduleでencodeし、captionやaction textとcontrastive learningする。

```text
video -> video encoder -> z_video
text  -> CLIP text encoder -> z_text
             similarity loss
```

意味的なaction representationを直接獲得しやすい一方、動画学習時にもtext supervisionを利用するため、現在の「動画データを自己教師ありで学習し、動画の動的性をLoRAへ獲得させる」という問いからは少し変わる。
したがって、採用する場合は研究目的を再定義する必要がある。

### Candidate D: MAEとCLIPを初期モデルとして比較する

同程度のViT backboneを用意し、

```text
Image MAE pretrained ViT -> video LoRA
CLIP pretrained ViT      -> video LoRA
```

として同じvideo dataset / temporal objective / LoRA条件で比較する。

これにより、

- reconstruction-based image pretraining
- language-aligned image pretraining

という事前学習方法の違いが、LoRAへ追加される動画情報へどう影響するか、という別の研究問いを作れる。
ただし比較変数が増えるため、現在の主研究を早く成立させる目的ではsecondary experiment候補とするのが安全。

## CLIPを使う最大の利点

今回の研究で特に価値が高いのは、text encoderを固定した「意味の物差し」が最初から存在することである。

MAEではLoRAを動画で学習した後、LoRAが何を獲得したかを調べるためにprobeやdownstream classifierを別途設計する必要がある。
CLIPでは、video representationをtext promptと比較することで、LoRA適応前後のsemantic shiftを直接観察できる可能性がある。

これは「LoRAに動画由来の何が追加されたかを解析する」というプロジェクト目的と整合しやすい。

## 主要な懸念

1. CLIPは標準状態ではframe-level image encoderであり、temporal reasoningを持たない。
2. video adaptation用のtemporal pathをどう設計するかはMAEと同様に必要。
3. LoRA学習によってCLIP image-text alignment自体が崩れる可能性がある。
4. temporal moduleを別途trainableにすると、動画情報がLoRAではなくtemporal moduleへ入る交絡が起こる。
5. CLIPを動画学習時にもtextで教師する場合、純粋なvideo self-supervised learningという現在の問いから変わる。
6. MAEのpixel reconstructionをそのままCLIPへ移植することはできず、decoder追加またはfeature-space objectiveが必要。

## 現時点の方向性

### 有力

- CLIPを「semantic image priorを持つ初期ViT」として利用する考え方は今回の研究と相性が良い。
- 特にfrozen text encoderを評価probeとして使い、LoRA学習前後のvideo-text similarityを解析する案は有力。
- image backboneをfreezeし、小さなtemporal adaptationだけを学習する考え方にはAIM / EVL等の先行研究がある。

### 現時点では保留

- MAEをCLIPへ完全に置き換えること。
- 動画学習時にvideo-text contrastive objectiveを主objectiveとすること。

理由は、現在の中心課題が「動画データから時間的・動的情報を自己教師ありでLoRAへ追加できるか」であり、MAEはこの問いを最も単純に構成しやすい。一方CLIPを主モデルにするとobjectiveも研究主張も変更されるため。

## 推奨する整理

現段階では次の役割分担が分かりやすい。

1. MAE branch
   - video self-supervised learningの主baseline / 初期研究系。
   - masked reconstructionを利用し、temporal情報獲得の仕組みを検証する。
2. CLIP branch
   - semantic image priorを持つ別初期モデル候補。
   - frozen text encoderをLoRA解析・zero-shot semantic evaluationへ利用する。
   - 将来的にはMAEとの比較またはCLIPを使ったvideo semantic adaptationへ展開する。

ただし、先生の意図が「MAEではなくCLIPを主軸にする」ことであれば、次のMTGでCLIPを挙げた目的（初期画像encoder / 評価probe / video-text training / downstream VLM接続のどれか）を確認する必要がある。

## 次に確認すべき論点

- 先生がCLIPを挙げた意図は、MAEの代替初期モデルか、比較baselineか、text spaceを利用した評価か。
- 動画学習時にもテキストを使う想定か。
- 純粋なvideo self-supervised learningを研究条件として維持するか。
- CLIPのimage-text alignmentをLoRA学習後も保持する必要があるか。
- LoRAをCLIP image encoder内に入れるか、temporal pathway側に入れるか。

## 関連文献

- Radford et al., Learning Transferable Visual Models From Natural Language Supervision, 2021.
- Wang et al., ActionCLIP: A New Paradigm for Video Action Recognition, 2021.
- Lin et al., Frozen CLIP Models are Efficient Video Learners, 2022.
- Yang et al., AIM: Adapting Image Models for Efficient Video Action Recognition, 2023.

このメモは探索記録であり、specまたは実装許可ではない。
