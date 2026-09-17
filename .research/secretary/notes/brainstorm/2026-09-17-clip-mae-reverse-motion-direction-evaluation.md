---
date: 2026-09-17
project: sequential-video-lora-analysis
source_todo: null
topic: CLIPとMAE+LoRAにおけるReverse・動作方向・自己教師あり評価
status: exploratory
tags: [brainstorm, research, mae, clip, lora, temporal-modeling, evaluation, self-supervised]
---

# CLIPとMAE+LoRAにおけるReverse・動作方向・自己教師あり評価

## 中心となる問い

Reverse評価や動作方向分類を、CLIP事前学習ViT系とMAE+LoRA系でそれぞれどのように使い分ければよいか。また、動画学習自体を自己教師ありで行う場合に、どのような評価結果が「時間情報を獲得した」Evidenceになるかを整理する。

## 重要な整理

Reverse評価は両者で意味が異なる。

- CLIP系: 時間を反転したときに、意味的な方向（例: pick up / put down）のtext similarityが適切に反転するかを見る。semantic temporal directionの評価に向く。
- MAE+LoRA系: target frameとmaskを固定し、past contextの順序だけを反転してreconstruction / prediction errorが悪化するかを見る。semantic labelではなく、時間順序依存性の評価に向く。

したがって、CLIPは「何の動作へ意味が変わったか」を直接評価しやすい。一方MAEは「正しい時間順序を使わないと自己教師あり目的を解きにくいか」を評価しやすい。

## CLIPでのReverse評価案

動画表現 `z_v` と、固定CLIP text encoderで作る方向対のprompt embeddingを比較する。

例:

```text
original video: person picks up a cup
reverse video:  person puts down a cup に近い時間方向

prompts:
- a person picks up a cup
- a person puts down a cup
```

評価候補:

```text
m(V) = sim(z(V), text_pickup) - sim(z(V), text_putdown)
```

期待結果:

- original: `m(V) > 0`
- reversed: `m(reverse(V)) < 0`
- Temporal LoRAでこのmarginと方向swap consistencyがBase CLIP / frame-independent LoRAより大きい。

Reverse時に単純に性能が落ちることよりも、正しい逆方向promptへrankingが入れ替わることが強いEvidenceになる。

ただし、全てのactionがreverseで別の自然なactionになるわけではないため、方向対が意味的に成立するclipを選ぶ必要がある。

## MAE+LoRAでのReverse評価案

causal current-frame reconstructionを例にする。

```text
ordered past:  x_{t-k}, ..., x_{t-2}, x_{t-1}
reversed past: x_{t-1}, x_{t-2}, ..., x_{t-k}
target:        同じ current frame x_t
mask:          同一
```

loss:

```text
Delta_reverse = L_reversed_past - L_ordered_past
```

期待結果:

- temporal LoRA: `Delta_reverse > 0`
- frame-independent LoRA: `Delta_reverse ≈ 0`
- Base MAE / LoRA disabled: temporal LoRAより小さい

これにより「同じframe集合でも正しい順序の方がcurrent reconstructionへ有用」という時間順序依存性を示せる。

Reverse sensitivityだけでLoRA固有とは言えないため、LoRA ON/OFF、frame-independent LoRA、temporal module controlを併用する。

## 動作方向分類とラベル

意味的な「pick up / put down」「open / close」のような動作方向分類をMAE representationで評価する場合、人手またはdataset annotationのlabelが必要になる。

ただし、自己教師あり学習で重要なのはpretraining / adaptation時にsemantic labelを使わないことであり、評価時にlabelを使ったlinear probeを行っても、自己教師あり表現学習の評価として成立する。

標準的な評価案:

1. ラベルなし動画でMAE+LoRAを自己教師あり学習。
2. Base MAE + LoRAをfreeze。
3. 小さなlinear classifierだけを方向ラベル付きtrain splitで学習。
4. test splitでdirection-sensitive action accuracyを測る。
5. pretrained MAE / frame-independent LoRA / temporal LoRAを同条件で比較する。

期待結果はTemporal LoRAのlinear probe精度がcontrolより高く、特に時間方向に依存するclass pairで差が大きいこと。

## 完全にラベルなしで評価したい場合

意味カテゴリそのものは評価できないため、semantic action directionではなくtemporal sensitivityをproxyとして評価する。

候補:

- original vs reverse verification
- ordered vs shuffled verification
- future feature prediction error
- current-frame reconstruction with ordered vs reversed past
- no-past / static-repeat ablation

`original / reverse`の2値は人手semantic labelではなく、動画変換から自動生成できるpseudo-labelである。これは自己教師ありのtemporal order verificationとして利用できる。

ただし、original-vs-reverse classifierだけでは「pick upを理解した」ことは示せない。示せるのは時間方向を区別できるrepresentationを持つことまで。

## 性能と時間情報を分けて評価する

大きなReverse degradationだけでは良いモデルとは限らない。評価では次の3軸を分ける。

1. Normal性能
   - MAE: normal reconstruction / prediction lossが低い。
   - CLIP: original/reversed双方で対応する正しい方向promptを高くrankできる。
2. Temporal sensitivity
   - MAE: orderを壊したときだけlossが増える。
   - CLIP: reverseに応じてdirection prompt rankingが適切にswapする。
3. LoRA attribution
   - LoRA OFFやframe-independent LoRAでは上記効果が弱まり、temporal LoRAで強い。

## 最小比較案

```text
Model A: pretrained Base
Model B: frame-independent LoRA
Model C: temporal LoRA
```

MAE評価:

```text
normal past / reversed past / shuffled past / no-past
-> same current target and same mask
-> reconstruction or prediction loss
```

CLIP評価:

```text
original / reversed video
-> paired direction prompts
-> similarity margin / pairwise accuracy / semantic swap rate
```

追加評価として、MAE+LoRAではlabel付きlinear probeを用いてdirection-sensitive action recognitionを行う。これは学習時にlabelを使わない限り、自己教師ありpretrainingの評価と矛盾しない。

## 現時点の解釈

- Reverse評価によるsemantic directionの説明力はCLIPの方が高い。
- MAE+LoRAではReverseを「時間順序依存性」のlabel-free intrinsic testとして使うのが自然。
- MAE+LoRAでsemanticな動作方向まで主張するには、評価時のlinear probeなど外部labelを用いるか、別のsemantic modelをprobeとして接続する必要がある。
- 完全label-free評価では、時間情報の有無は示せても、人間が解釈するaction semanticsまで特定することはできない。

このメモは探索記録であり、specまたは実装許可ではない。
