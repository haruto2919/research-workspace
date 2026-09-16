---
date: 2026-09-17
project: sequential-video-lora-analysis
source_todo: null
topic: MAE方針の再整理
status: exploratory
tags: [brainstorm, research, mae, lora, video, temporal-modeling]
---

# MAE方針の再整理

## 読み込んだ文脈

- `.research/lab/projects/sequential-video-lora-analysis/README.md`
- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-10-mtg.md`
- `.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-11-mae-single-image-reconstruction-smoke-spec.md`
- `.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-16-mae-single-image-one-step-training-smoke-spec.md`
- `.research/secretary/notes/brainstorm/2026-09-11-mae-self-supervised-video-lora-pivot.md`
- `.research/secretary/notes/brainstorm/2026-09-17-mae-encoder-lora-targeting.md`

## 現在の研究方針

画像で事前学習されたMAEを出発点とし、base MAEが持つ画像・空間表現をなるべく保持したまま、動画を自己教師ありで学習することでLoRAまたは追加temporal moduleへ時間的・動的情報を獲得させる。

概念的な流れは次である。

```text
ImageNetで事前学習済みMAE
  -> MAE encoderへLoRAを追加
  -> base MAEをfreeze
  -> 複数frameからなる動画を入力
  -> frame間の時間関係を利用するtemporal modeling / objective
  -> LoRA・temporal parameterのみ更新
  -> 学習済みLoRAが保持する時間・動作情報を評価・解析
```

## 重要な注意

標準Image MAEへ動画frameを1枚ずつ独立に入力し、各frameのmasked patch reconstructionだけでLoRAを更新しても、時間方向の関係を利用する必要がない。この場合、LoRAが学んだものは動画のappearance/domain adaptationであり、時間情報を獲得したとは主張できない。

したがって、動画段階ではarchitectureまたはobjectiveのどちらかで複数frame間の関係を必須にする必要がある。

## 現在までに確認済みの段階

### 1. MAE integration baseline

`facebook/vit-mae-large` を研究コードへ導入し、単一RGB画像で標準MAE forwardを実行し、reconstruction lossを取得できることを確認済み。

### 2. MAE 1-step training baseline

単一RGB画像1枚を用い、Lightning `Trainer.fit()`の通常training pathでMAE全parameterを1 optimization step更新できることを確認済み。

これらは基盤動作確認であり、動画情報・時間情報の獲得を示すEvidenceではない。

## 次段階の最有力候補

### 3. MAE encoder LoRA-only smoke

MAE encoder全24 blockのattention Q/VへLoRAを追加する。候補は合計48 modules。

```text
vit.layers.0-23.attention.q_proj
vit.layers.0-23.attention.v_proj
```

base encoderとdecoderをfreezeし、LoRAのみがMAE reconstruction lossから1 step更新されることを確認する。PEFT利用が第一候補。

この段階も単一画像でよく、時間情報の主張は行わない。

## その後に決める中心課題

### 4. 動画のtemporal modeling設計

主な候補は次。

- Late temporal fusion: 各frameをImage MAE encoderで処理し、frame representationを後段temporal moduleで融合する。
- Internal temporal adaptation: image-pretrained ViT内部へtemporal attention / adapter / LoRA経路を追加する。
- VideoMAE-like joint modeling: space-time tokenを直接扱う。

現時点では、image-pretrained encoderを保ちやすいlate fusionまたはinternal temporal adaptationを優先候補とするが、まだ確定ではない。

### 5. 動画自己教師ありobjective

単純なframe-wise MAE lossだけでは不十分。temporal moduleを使わないと解けない学習目標が必要。

候補:

- neighboring framesをcontextとしてcurrent frameのmasked patchを予測
- cross-frame masked modeling
- future / next-frame feature prediction
- temporal order prediction

## 評価方針候補

「時間情報を使っているか」を検証するため、通常時系列だけでなく時間関係を壊した入力と比較する。

```text
normal ordered video
vs.
frame shuffle
vs.
static-repeat / duplicated frame
```

正常な時系列でのみ性能・表現が有利になるかを見ることで、appearanceだけでなくtime order / motionを利用しているかを評価する。

## 現在の研究ストーリー

```text
画像だけで学習したMAE
  ↓
静的・空間的な画像表現を持つ
  ↓
baseをできるだけ固定
  ↓
動画由来の追加学習をLoRA / temporal moduleへ限定
  ↓
時間変化・動作に関する情報を追加parameterへ獲得させる
  ↓
LoRA内部を解析し、何が獲得されたかを評価する
```

## 未解決事項

- MAE encoder LoRA smokeのhyperparameter固定。
- late fusionとinternal temporal adaptationのどちらを主軸にするか。
- temporal moduleの具体構造。
- 動画自己教師ありobjective。
- sequential_loaderをstrict online学習として使うか、単純な時系列video inputに使うか。
- 時間情報獲得の最終評価指標・下流タスク。

## 次アクション候補

1. MAE encoder Q/V LoRA-only 1-step smokeをspec化する。
2. その後、late fusionとinternal temporal adaptationを比較し、動画段階のarchitectureを決める。
3. temporal modelingが必須となる自己教師ありobjectiveを決める。

このメモは方針再整理の探索記録であり、specまたは実装許可ではない。
