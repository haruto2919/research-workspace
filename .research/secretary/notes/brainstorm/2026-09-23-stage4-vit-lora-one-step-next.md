---
date: 2026-09-23
project: sequential-video-lora-analysis
source_todo: null
topic: Stage 4 ViT LoRA one-step update の次実装案
status: exploratory
tags: [brainstorm, research, vit, lora, activitynet, smoke-test]
---

# Stage 4 ViT LoRA one-step update の次実装案

## 現在地

- research code: `tamaki-lab/2026_09_ishikawa_sequential-video-lora`
- current implementation branch: `dev`
- current GitHub HEAD確認時点: `b6385d87e2e3e82a719d8f4b686b44aa293b1135`
- Stage 3: ActivityNet 16 frames -> frozen ViT -> frame features -> masked mean -> clip feature [768] が成立済み。
- current `ViTFrameEncoder` は `google/vit-base-patch16-224` を読み込み、base ViT全parameterをfreezeする。
- `MaskedMeanClipAggregator` はgradientをdetachしない。
- temporal learningについては、masked meanのみではorder情報を失うため、Stage 4では時間情報獲得を主張しない。

## 次に実装する目的

Stage 4では研究objectiveをまだ実装せず、

`ActivityNet frames -> ViT + LoRA -> frame features -> masked mean -> scalar smoke loss`

のgradient pathを成立させ、

- Base ViTは変化しない。
- LoRAだけがtrainable。
- clip-level lossからLoRAへgradientが届く。
- optimizer stepで少なくとも1つのLoRA parameterが変化する。
- base parameterはstep前後で不変。

を確認する。

これはtemporal learningのEvidenceではなく、LoRA学習経路のengineering verificationである。

## 実装構造候補

既存のfrozen baselineを壊さないため、`ViTFrameEncoder` を直接LoRA専用へ変更するより、
新しい `ViTLoRAFrameEncoder` を追加する案を有力とする。

例:

- `model/vit/vit_lora_frame_encoder.py`
- `ViTFrameEncoder`: Stage 1/3のfrozen baselineとして維持。
- `ViTLoRAFrameEncoder`: LoRA付き学習path。

LoRA実装はPEFT利用を第一候補とする。

初期engineering baseline候補:

- target modules: attentionの `query`, `value`
- 対象: 全encoder blockで実際にmatchしたmodule
- rank `r=8`
- `lora_alpha=8`
- `lora_dropout=0.0`
- bias: `none`
- base ViT: frozen
- trainable: LoRAのみ
- classifier等の追加trainable module: なし

これらは最終研究hyperparameterではなく、1-step smokeを単純化する初期値。

PEFTを使用する場合は、実環境のTransformersとの互換versionをspecでpinし、
実際にmatchしたLoRA target module一覧・個数をtestで監査する。

## 既存pathとの接続

Stage 3の

`encode_chunk -> frame_features -> MaskedMeanClipAggregator`

を再利用する。

ただしStage 3 smokeの `torch.no_grad()` は変更せず、
Stage 4用の別smoke entrypointを作る。

候補:

`smoke_activitynet_vit_lora_one_step.py`

Stage 4ではgradientを有効にして、

1. ActivityNet先頭1 chunkを読む。
2. valid frameだけpreprocess。
3. ViT+LoRAでframe featuresを生成。
4. masked meanでclip featureを生成。
5. engineering用scalar lossを作る。
6. backward。
7. gradient監査。
8. optimizer.step。
9. base/LoRA parameter差分を監査。

を行う。

## smoke loss候補

MoCo実装前なので、研究上のobjectiveと混同しない単純な微分可能lossを使う。

候補:

`loss = clip_feature.float().pow(2).mean()`

このlossの値自体には研究上の意味を持たせない。
目的はclip-level scalarからLoRAまでgradient pathが成立するかだけを確認すること。

LoRA初期化方式によっては1 step目に全LoRA matrixが同時に変化しない可能性があるため、
Success Criteriaは「全LoRA parameterが変化」ではなく、

- LoRA側にfinite gradientが存在する。
- optimizer対象がLoRAのみ。
- 少なくとも1つのLoRA parameterがstep後に変化する。

とする。

## 必須test候補

1. Base ViT parameterが全て `requires_grad=False`。
2. LoRA parameterだけ `requires_grad=True`。
3. 想定外のtrainable parameterが0件。
4. target moduleが0件でない。
5. feature shapeが `[N_valid,768]`。
6. masked mean後が `[768]`。
7. loss finite。
8. backward後、LoRAにfinite gradientが存在。
9. base parameterにgradientなし、または更新対象外。
10. optimizer parameter集合がtrainable LoRA集合と一致。
11. step後、base parameterがbitwise/厳密比較で不変。
12. step後、少なくとも1つのLoRA parameterが変化。
13. padding除外・metadata保持などStage 3 contractを維持。
14. 既存frozen ViT / ActivityNet / masked mean testsを壊さない。

## Stage 4でやらないこと

- MoCo。
- query/key encoder。
- EMA。
- InfoNCE。
- queue。
- full training。
- sequential vs shuffle比較。
- reverse評価。
- temporal Transformer。
- past -> future prediction。
- temporal情報獲得の主張。
- LoRA rank/alpha等の科学的比較。

## Stage 4の次

Stage 4のGate通過後に、Stage 5としてMoCo mechanicsを独立実装する。

その後、

- non-temporal condition: ViT+LoRA -> masked mean -> ordinary MoCo
- temporal condition: temporal relationをlossへ明示的に入れる

を分けて設計する。

temporal conditionの有力候補は
past/current clipとfuture clipの関係を使うobjectiveであり、
ordered / shuffled / reversed / static-repeat等のcontrolを後続で比較する。

## 現在の方向性

次の実装単位としてはStage 4 LoRA one-step updateが最も切り分けやすい。
masked meanを用いるが、このStageの目的は時間学習ではなくLoRA学習経路の成立確認に限定する。

このメモは探索記録であり、specまたは実装許可ではない。
