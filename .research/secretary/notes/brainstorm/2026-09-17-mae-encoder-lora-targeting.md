---
date: 2026-09-17
project: sequential-video-lora-analysis
source_todo: null
topic: MAE encoder LoRA targeting
status: exploratory
tags: [brainstorm, research, mae, lora, peft]
---

# MAE encoder LoRA targeting

## 出発点

MAE単一画像1-step学習 smoke が完了し、`facebook/vit-mae-large` をPyTorch Lightningのtraining pathで1 optimization step更新できることを確認した。次段階として、base MAEを固定し、LoRAだけを更新できる最小構成を作る前に、LoRAの注入対象と実装方法を整理する。

## 確認済みEvidence

2026-09-17に `ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-large")` の `named_modules()` を確認した。

encoder側には24 blockあり、各blockに次のmoduleが存在する。

```text
vit.layers.<0-23>.attention.q_proj
vit.layers.<0-23>.attention.v_proj
```

したがって、encoderのQ/V候補は24 block × 2 = 48 modulesである。

一方、decoder側にも8 blockあり、次のmoduleが存在する。

```text
decoder.decoder_layers.<0-7>.attention.q_proj
decoder.decoder_layers.<0-7>.attention.v_proj
```

したがって、単純に `target_modules=["q_proj", "v_proj"]` とするとdecoder側も対象になり得るため、encoder-onlyを保証するtarget指定が必要である。

## 現時点の最有力方針

### LoRA対象

MAE encoderの全24 blockについて、attentionのQ/V projectionのみをLoRA対象とする。

```text
vit.layers.0.attention.q_proj
vit.layers.0.attention.v_proj
...
vit.layers.23.attention.q_proj
vit.layers.23.attention.v_proj
```

合計48 modules。

理由:

- 研究目的は、画像事前学習済みencoderへ動画由来の追加情報をLoRAとして獲得させることにある。
- decoderを更新対象へ含めると、reconstruction lossの変化がencoder LoRAだけによるものか分かりにくくなる。
- 過去のMeMViT最小LoRA確認でもattention Q/Vを出発点としていたが、今回の対象は現在のViT-MAE実装に合わせて明示的に再確認した。

### LoRA実装

Hugging Face PEFTを第一候補とする。

理由:

- LoRA自体の実装を研究対象にせず、LoRAへどのような動画情報が獲得されるかを研究対象にしたい。
- PEFTでtrainable parameter、targeted modules、adapter stateを確認できる。
- 自前LoRA実装による初期化・scaling・保存・freeze制御の追加不確実性を避けられる。

encoder-only targetingは、`named_modules()`から次の条件で48個のfull module nameを列挙し、その一覧を `LoraConfig.target_modules` に渡す方法を第一候補とする。

```python
encoder_targets = [
    name
    for name, module in model.named_modules()
    if name.startswith("vit.layers.")
    and (name.endswith(".q_proj") or name.endswith(".v_proj"))
]
```

LoRA適用後はPEFTの `targeted_module_names` などを使い、48 moduleだけが対象でdecoderが含まれていないことを必須確認とする。

## LoRA hyperparameter候補

1-step smoke用の第一候補:

```text
r = 8
lora_alpha = 8
lora_dropout = 0.0
bias = none
```

これは本学習の最適設定ではなく、LoRA-only update成立を確認するための診断条件候補である。

ViT-MAE-Large encoder hidden size 1024を前提とすると、Q/V 48 modulesへrank 8 LoRAを入れた場合、LoRA A/Bのtrainable parameterは概算で次となる。

```text
48 × (8×1024 + 1024×8) = 786,432 parameters
```

実装後は実際のPEFT出力でtrainable parameter数を確認し、概算と整合することをsanity check候補とする。

## freeze方針候補

```text
MAE base encoder weights: frozen
MAE decoder weights: frozen
LoRA A/B parameters: trainable
```

`modules_to_save`等でbase layerを追加学習しない。

## 次のsmokeで確認したいEvidence

単一RGB画像1枚を使い、MAE reconstruction lossで1 optimization stepだけ実行する。

必須候補:

- LoRA targetがencoder Q/Vの48 modulesのみ。
- decoder側にLoRAが入っていない。
- trainable parametersがLoRA parameterのみ。
- representative base encoder parameter: before == after。
- representative decoder parameter: before == after。
- representative LoRA parameter: before != after。
- LoRA delta norm > 0 かつfinite。
- lossがscalarかつfinite。
- `trainer.global_step == 1`。

この段階ではloss改善、reconstruction品質、動画情報・時間情報の獲得は主張しない。

## 保留・未決事項

- `r=8`, `lora_alpha=8`, `lora_dropout=0.0` を次specの固定値として採用するか。
- LoRA optimizer / learning rateを前段smokeのAdamW `1e-4`から継続するか、LoRA用に別値へ固定するか。
- PEFT dependencyのversion固定方法。

## Research Spec Handoff候補

- 対象プロジェクト: `sequential-video-lora-analysis`
- 実装目的: MAE encoder Q/VへLoRAを注入し、base MAEをfreezeした状態でLoRAのみが1 step更新されることを確認する。
- 採用方向: encoder 24 blocksのQ/V、計48 modulesを対象。decoderは対象外。PEFT利用を第一候補。
- 対象外: 50Salads、sequential_loader、動画入力、temporal modeling、full training、性能評価。
- 主要未決事項: LoRA hyperparameter、optimizer/lr、PEFT version固定。
