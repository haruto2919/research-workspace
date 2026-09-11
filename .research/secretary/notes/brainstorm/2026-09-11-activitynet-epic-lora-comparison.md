---
date: 2026-09-11
project: sequential-video-lora-analysis
source_todo: null
topic: ActivityNetとEPIC-KITCHENSで学習したLoRAの比較設計
status: exploratory
tags: [brainstorm, research, mae, lora, video, activitynet, epic-kitchens, evaluation]
---

# ActivityNetとEPIC-KITCHENSで学習したLoRAの比較設計

## 相談の出発点

ユーザーは、ImageNet等の画像データセットで自己教師あり事前学習されたMAEにLoRAを追加し、動画データセットで自己教師あり学習した後、ActivityNetとEPIC-KITCHENSそれぞれで学習したLoRAを比較することで、今回比較したい動画由来情報の差を評価できるかを検討している。

## 確認したデータセット特性

### ActivityNet

- 幅広い人間活動を対象とした大規模動画ベンチマーク。
- 200 classes、各classあたり約100本のuntrimmed video、総計約648時間と公式サイトに記載されている。
- global classification、trimmed classification、temporal activity detection等に利用される。
- Source: https://activity-net.org/

### EPIC-KITCHENS-100

- 一人称（egocentric）視点の、家庭のキッチン内で収録された非スクリプトの日常活動。
- head-mounted camera、100 hours、20M frames、約90K action segments、97 verb classes、300 noun classes。
- 人物の手と物体の細かな相互作用、fine-grained action、時間的変化が強い。
- Source: https://epic-kitchens.github.io/

## 現時点の結論

ActivityNetとEPIC-KITCHENSで別々にLoRAを学習して比較する方法は、

> 「異なる動画分布で学習したLoRAが、どのように異なる動画知識を獲得するか」

を調べるには有力である。

一方で、データセット間では以下も同時に変化するため、差分をそのまま

> 「時間・動的情報を学習した量の差」

とは解釈できない。

- 一人称 / 三人称・web動画中心というviewpoint
- kitchen限定 / wide-domainというscene・domain
- hand-object interactionの密度
- object identity・背景
- camera motion
- video duration、編集・カットの有無
- データ量
- action granularity

したがって、dataset comparisonは「dataset-specific video knowledge」の比較には適するが、「temporal informationそのもの」の因果的検証としては単独では不足する。

## 比較したい主張ごとの適否

### 1. DatasetごとにLoRAの獲得情報が異なるか

適している。

- ActivityNet-trained LoRA
- EPIC-KITCHENS-trained LoRA

を同一base MAE、同一LoRA構成、同一self-supervised objectiveで作れば、異なるvideo distributionによるLoRA差分を調べられる。

### 2. LoRAが時間・動的情報を学んだか

dataset間比較だけでは不十分。

同一dataset内で時間構造を壊すcontrolを追加する必要がある。

候補:

- normal order
- shuffled order
- reversed order
- static-repeat
- no temporal context

時間構造を壊したときに性能・loss・表現がどの程度悪化するかを測れば、temporal dependencyへの依存を評価しやすい。

### 3. ActivityNetとEPIC-KITCHENSのどちらがより動的情報をLoRAへ与えるか

可能だが、raw scoreを直接比較してはいけない。

各dataset内でbase MAEとの差、normal-vs-shuffle差などの相対効果量を計算し、その効果量同士を比較する方が妥当。

## 推奨する比較条件

ActivityNet学習とEPIC-KITCHENS学習で、少なくとも以下を固定する。

- 同一ImageNet pretrained MAE checkpoint
- 同一LoRA target module
- 同一rank / alpha / dropout
- 同一causal temporal fusion構造
- 同一mask ratio / reconstruction objective
- 同一optimizer / LR
- 同一resolution / augmentation
- 同一temporal sampling interval
- 同一training frame数またはtoken budget
- 同一optimization step数

特に、ActivityNetの総量はEPIC-KITCHENS-100より大きいため、全量をそのまま使うと「dataset特性」ではなく「学習量」の差が混ざる。まずはtraining frame/token budgetを揃えた比較が望ましい。

## 評価の考え方

### A. 共通評価条件を使う

ActivityNet-trained LoRAとEPIC-trained LoRAを、それぞれ別々のdataset固有metricだけで評価すると比較できない。

例:

- ActivityNet LoRAをActivityNet accuracyで評価
- EPIC LoRAをEPIC verb/noun accuracyで評価

この2値を直接比較しても意味がない。

両LoRAへ同じdiagnostic evaluationを適用する必要がある。

### B. Base MAEからの改善量を使う

各評価set Eについて、

Delta(E, LoRA) = metric(E, Base+LoRA) - metric(E, Base)

のようにbaseとの差を見る。

自己教師ありreconstructionなら、loss reduction率などのrelative metricにする。

### C. Temporal perturbation sensitivityを測る

同じclipに対してnormalとshuffledを用意し、

Temporal sensitivity = metric(normal) - metric(shuffled)

のような差を算出する。

これをActivityNet-trained LoRAとEPIC-trained LoRAで比較すると、単純なdataset固有accuracyよりも「時間構造利用」の差へ近づく。

### D. 共通の第三者評価setも候補

より厳密に動的情報を比較したい場合は、学習にはActivityNet / EPIC-KITCHENSを使い、評価には両者とは別のtemporal reasoningが必要な共通datasetを用いる案もある。

例としてSomething-Something系は、物体操作の時間関係を認識する必要が強いbenchmarkとして候補になる。ただし研究scopeと追加実装コストを考えて採否を決める。

## 現時点の推奨実験構造

まずは2 LoRAを作る。

- L_ActivityNet: ActivityNetで自己教師ありonline video learning
- L_EPIC: EPIC-KITCHENSで同一条件の自己教師ありonline video learning

その上で評価を次の二層に分ける。

### Layer 1: dataset-specific information

- parameter update magnitude
- layer-wise LoRA norm
- LoRA subspace similarity / cosine / principal angle等
- reconstruction improvement
- cross-dataset transfer

### Layer 2: temporal information

- normal vs shuffled
- normal vs reversed
- normal vs static-repeat
- temporal contextあり vs なし

この構成なら、

1. datasetによってLoRAがどう違うか
2. その差のうちtemporal structure利用に関係する部分があるか

を分離して議論しやすい。

## 採用候補

ActivityNet vs EPIC-KITCHENSの比較は採用候補。

ただし主張は最初から「どちらがより動的情報を学習した」ではなく、まず

> 異なるvideo distributionで学習したLoRAの獲得情報・temporal sensitivityを比較する

と置く方が安全。

## 未解決事項

- LoRA比較の主評価をparameter space、representation、reconstruction loss、downstream probeのどれに置くか。
- ActivityNet / EPICのtraining budgetをどの単位で一致させるか（frames / clips / tokens / optimizer steps）。
- online時のtemporal sampling rateとmemory length。
- 共通第三者評価datasetを導入するか。
- ActivityNet動画内の編集・scene cutをonline temporal signalとしてどう扱うか。
