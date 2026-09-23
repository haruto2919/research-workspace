---
date: 2026-09-23
project: sequential-video-lora-analysis
source_todo: null
topic: masked mean late fusion と LoRA の時間特徴学習可能性
status: exploratory
tags: [brainstorm, research, vit, lora, late-fusion, temporal-learning, moco]
---

# masked mean late fusion と LoRA の時間特徴学習可能性

## 出発点

現在のStage 3では、ActivityNetの連続16 frameを各frame独立にfrozen ViTへ通し、
`frame_features [16,768]` を得た後、valid frameのmasked meanによって
`clip_feature [768]` を作るところまで実装済みである。

次段階ではViT内部へLoRAを追加する予定だが、
「late fusionを先に作り、その後LoRAを追加する構成で、LoRAに時間特徴が入るのか」
という疑問を整理する。

## 確認済み事実

- Stage 3のmasked meanは意図的にorder-invariantなnon-temporal baselineとして設計されている。
- 現行 `ViTFrameEncoder` は各frameを独立にencodeする2D ViTであり、frame間attentionを持たない。
- Stage 4でLoRAを追加する場合、実行時の想定data flowは
  `frame -> ViT+LoRA -> frame feature -> masked mean -> clip feature`
  であり、LoRAはmasked meanの後ろに置く想定ではない。
- 現行Stage 3のaggregatorはgradientをdetachしないため、将来LoRAを入れた場合はclip-level lossからLoRAへgradientを流せる。

## 中心となる問い

`frame-wise ViT+LoRA -> masked mean -> self-supervised loss`
という構成だけで、LoRAが動画の時間順序・動的関係を学習できるか。

## 分析

### 1. masked meanだけではframe順序の情報が消える

各frame featureを

`z_t = f_theta(x_t)`

とし、clip featureを

`c = (1/T) sum_t z_t`

とする。

任意のframe permutation `pi` に対して

`(1/T) sum_t f_theta(x_pi(t)) = (1/T) sum_t f_theta(x_t)`

なので、同じframe集合なら

- ordered
- reversed
- shuffled

でclip featureは同じになる。

したがってlossがこのclip featureだけに依存する場合、lossもLoRAへのgradientもframe permutationに対して不変である。

この条件では、LoRAに
「Aの後にBが起きた」
「前進と逆再生が違う」
といった時間順序情報を教える信号が存在しない。

### 2. それでもLoRAが学べるものはある

時間順序は学べなくても、動画frameを大量に使うことでLoRAが次を学ぶ可能性はある。

- ActivityNet特有のappearance / domain
- 動作中に頻出する姿勢・物体配置
- 単一frameから推測できるmotion state
- video frame特有のblurや撮影条件
- clip内で共通して現れるsemantic content

ただし、これらは「動画由来情報」ではあっても、
厳密な意味でのtemporal relation / temporal orderとは区別する必要がある。

### 3. sequential update順序は別の意味で影響し得る

clipを

`clip_1 -> optimizer step -> clip_2 -> optimizer step -> ...`

のように逐次学習する場合、SGD/Adamの更新は一般に可換ではないため、
clip順序をshuffleすると最終LoRA parameterが変わる可能性はある。

これは研究上興味深いが、

- 学習parameter trajectoryが入力順に依存する
- representationそのものが時間順序を理解する

は別の主張である。

sequential vs shuffleでLoRAが変わるだけでは、
LoRAがclip内部の時間関係を表現したとは言えない。

### 4. late fusion自体が問題なのではない

late fusionでも、frame featureの後に

- temporal Transformer
- GRU/LSTM
- 1D temporal convolution
- positional encoding付きattention
- ordered pair module

などを置けばorder-awareにできる。

したがって本質的な問題はlate fusionそのものではなく、
現在のfusionがmasked meanというpermutation-invariant operatorであること。

## 有力な研究設計

### A. masked meanはbaselineとして残す

現在のStage 3は捨てず、

`ViT+LoRA -> masked mean -> MoCo`

をnon-temporal baselineとして残す。

これは
「動画domainへadaptしたLoRA」
と
「時間構造を利用するLoRA」
を分離するために重要なcontrolになる。

### B. temporal signalを別条件として追加する

有力候補は次。

#### B1. order-aware temporal aggregator

`frame features -> positional encoding -> temporal Transformer -> clip feature`

等に変更する。

長所:
- ordered / reversed / shuffledを区別できる。
- 現在のframe encoderを維持できる。

注意:
- aggregator自体がtrainableだと、時間情報がLoRAではなくaggregatorに入った可能性がある。
- LoRA内部の時間情報を主張するにはparameter freezeやablationが必要。

#### B2. temporal self-supervised objective

masked meanを維持しても、lossをclip間の時間関係に依存させる。

例:
- current clip -> future clip feature prediction
- temporal offset prediction
- ordered pair / reverse discrimination
- adjacent clip positive + non-adjacent negative

特にpast -> futureのような非対称objectiveは、
単なる「近いframeを似せる」よりtime directionを学ばせやすい。

#### B3. multiple-frameを直接見るmoduleへLoRAを入れる

LoRA自体にcross-frame relationを持たせたい場合は、
LoRAをframe-wise image ViTだけでなく、

- temporal Transformer
- video attention
- space-time attention

など複数frameを同時に見るmoduleへ入れる方法が最も直接的。

この場合は「temporal moduleのLoRA」という形になり、
LoRA parameter自体が時間方向のattention変換を担える。

## MoCoとの関係

通常の

`same clip -> augmentation A/B -> masked mean -> MoCo`

だけでは、frame順序を変えても同じ表現になり得るためtemporal signalは弱い。

一方、

`past clip -> query`
`future clip -> positive key`

のようにpositive pairの定義を時間方向へ変更すれば、
各clip内部がmasked meanでもclip間の時間関係をlossへ入れられる。

ただし、隣接clipが似る理由は動作の時間連続性だけでなく
scene / object / identityの連続性でも説明できる。
そのためshuffle / distant / static-repeat等のcontrolが必要。

## 現在の収束

### 採用候補

- Stage 3 masked meanはnon-temporal baselineとして維持する。
- Stage 4のLoRA 1-step smokeも現在のmasked mean経路上で実施してよい。
  ただし目的は「LoRAだけ更新できること」のengineering verificationであり、
  temporal learningのEvidenceとは扱わない。
- MoCo統合後、temporal learningを主張する前に、
  order-aware objectiveまたはorder-aware aggregationを別条件として追加する。

### 棄却寄り

- `frame-wise ViT+LoRA -> masked mean -> ordinary clip-level MoCo` だけで
  「LoRAが時間順序を学習した」と主張すること。
- sequential vs shuffleでparameter差が出ただけで
  temporal representation獲得と結論すること。

## 比較案

最低限、次を分けると解釈しやすい。

1. Frozen ViT + masked mean
2. ViT+LoRA + masked mean + ordinary SSL
3. ViT+LoRA + masked mean + temporal objective
4. 必要なら ViT+LoRA + order-aware temporal aggregator + temporal objective

評価control:

- ordered
- shuffled
- reversed
- static-repeat
- temporally distant clips

特にreverseを評価に使う場合、masked mean baselineはorderedとreverseが理論上同じになるため、
「reverseを区別できないbaseline」として明確な対照になる。

## 未解決事項

- temporal signalを最初にobjective側へ入れるか、aggregator側へ入れるか。
- LoRAが時間情報を保持したと主張するために、temporal moduleをfreezeすべきか。
- MoCoのpositive pairをsame-clip augmentationにするか、past/future clipにするか。
- 「時間情報」をorder、direction、motion continuity、future predictabilityのどれとして定義するか。

## 次アクション候補

次のStage 4 specでは、まずLoRAのparameter update smokeだけを固定する。
temporal learningの方式は同specへ混ぜず、Stage 5/6へ進む前に
「non-temporal MoCo baseline」と「temporal condition」の比較設計を別途固める。

## 関連

- `.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-23-activitynet-clip-feature-spec.md`
- `.research/lab/projects/sequential-video-lora-analysis/experiments/2026-09-23-activitynet-clip-feature-verification.md`
- `.research/secretary/notes/brainstorm/2026-09-18-vit-lora-moco-staged-implementation.md`
- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-17-mtg.md`

このメモは探索記録であり、specまたは実装許可ではない。
