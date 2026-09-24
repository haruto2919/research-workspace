---
date: 2026-09-24
project: sequential-video-lora-analysis
source_todo: null
topic: ImageNet事前学習ViTをVideoMAE型encoderへ移植する案
status: exploratory
tags: [brainstorm, research, videomae, vit, imagenet, lora, temporal-learning]
---

# ImageNet事前学習ViTをVideoMAE型encoderへ移植する案

## 出発点

現在の研究では `google/vit-base-patch16-224` を画像事前学習済みViTとして使い、frame-wise ViT + Q/V LoRA + masked mean + MoCoをbaselineとしている。

このViTをVideoMAEのencoder側へ流用し、VideoMAE型のmasked video modelingによってLoRAへ時間的・動的情報を入れられるかを検討する。

## 確認済み事実

- 現行checkpoint `google/vit-base-patch16-224` はImageNet-21kでpretrainされ、ImageNet-1kでfine-tuneされたViT-B/16。
- 現行ViTは2D patch embeddingで、各frameを独立に処理している。
- VideoMAE-Baseはhidden size 768、12 layers、12 heads、patch size 16で、ViT-B/16とTransformer本体の主要dimensionは近い。
- VideoMAEは通常、複数frameを3D tubelet embeddingで時空間tokenへ変換し、それらをTransformer encoderへ同時入力する。
- original VideoMAEのtubelet sizeの標準値は2。
- VideoMAE pretraining encoderはvisible tokenだけを処理し、decoderがmasked tokenを復元する。

## 結論

**実装可能。**

ただし「Hugging FaceのImageNet ViTをそのままVideoMAEForPreTrainingへ差し替える」ことはできない。
Transformer block重みはかなり流用できるが、次の差分を変換する必要がある。

1. 2D patch embedding -> 3D/tubelet patch embedding
2. 2D positional embedding -> spatiotemporal positional embedding
3. CLS tokenの扱い
4. encoder outputとVideoMAE decoderの接続
5. state_dictのparameter naming / qkv表現

したがって実態は「ImageNet-pretrained ViTで初期化したVideoMAE型encoder」を新しく構成することになる。

## 推奨構成

### 案A: tubelet_size=1でImageNet ViTを動画化

最初の検証として最も扱いやすい。

```text
16 frames [B,T,3,H,W]
  -> 2D ViT patch embedding weightsをConv3d temporal kernel=1へ変換
  -> spatiotemporal patch tokens
  -> tube masking
  -> ImageNet ViTから初期化したTransformer blocks
       + Q/V LoRA
       + base weights freeze
  -> visible video tokens
  -> VideoMAE decoder
  -> masked patch reconstruction loss
```

2D patch projection `[D,3,16,16]` は temporal dimensionをunsqueezeして
`[D,3,1,16,16]` にできるため、画像checkpointをほぼそのまま初期化できる。

利点:
- patch embedding変換が最小。
- ImageNet ViTの初期表現を壊しにくい。
- Transformer self-attentionに全frame tokenを入れるため、frame間attentionは発生可能。
- Q/V LoRAがcross-frame attentionの変化を担える。

欠点:
- tubelet=2よりtoken数が増え、計算量が大きい。

### 案B: original VideoMAEに近いtubelet_size=2

2D patch weight `W` を時間方向へinflateする。

初期化候補:

```text
W_video[:,:,0,:,:] = W_image / 2
W_video[:,:,1,:,:] = W_image / 2
```

これにより2 frameの情報を平均的に使ってpatch tokenを作る。

利点:
- VideoMAE標準に近い。
- token数を半分にできる。
- tube maskingとの整合性が高い。

注意:
- temporal patch embedding自体はImageNet checkpointに存在しないため、inflate方法は研究上の追加設計になる。

## Transformer block

ViT-B/16とVideoMAE-Baseは主要dimensionが一致するため、Transformer層の

- attention Q/K/V
- output projection
- MLP
- LayerNorm

のweightは原理上かなり再利用できる。

ただしHugging Face ViTとoriginal VideoMAE/timmではstate_dict namingやQKV実装が異なるため、weight conversion codeが必要。

最初からHugging FaceのViT blockを再利用し、その前後だけvideo token化・masking・decoderを実装する方法も候補。

## Positional embedding

ImageNet ViTのlearned 2D position embeddingはvideo token数と一致しない。

候補:
1. spatial position embeddingを各frameへ複製し、temporal embeddingを新規追加。
2. VideoMAEと同様に固定sin-cos spatiotemporal positional encodingを使用。
3. spatial componentだけImageNetから移植し、temporal componentを0初期化。

研究上は、LoRAへ時間情報を入れたいなら、固定spatiotemporal位置埋め込みを使うと「時間情報がtrainable temporal embeddingだけに入った」という交絡を減らせる。

## CLS token

original VideoMAE pretrainingではpatch/tubelet reconstructionが中心でCLS tokenは本質的ではない。

そのためImageNet ViTのCLS tokenはpretraining時には外し、
Transformer block weightを主に引き継ぐ方が単純。

下流評価で必要ならmean poolingまたはCLSを別途検討する。

## Decoder

ImageNet ViTにはVideoMAE decoderがないため、decoderは新規追加が必要。

```text
encoder hidden 768
  -> encoder_to_decoder
  -> decoder hidden 384
  -> masked patch prediction
```

decoderをtrainableにするのが標準的。

このとき「全学習parameterがLoRAだけ」とはならない。
より正確には、

> ImageNet-pretrained encoder本体をfreezeし、encoder Q/V LoRAとVideoMAE decoderを自己教師ありで学習する

という構成になる。

学習後にdecoderを破棄し、LoRA付きencoderを解析すればよい。

ただしreconstruction loss改善の一部はdecoderが担えるため、「時間情報がLoRAへ入った」ことは別評価が必要。

## 現研究で重要な意味

現在の

```text
frame-wise ImageNet ViT + LoRA
  -> CLS [T,768]
  -> masked mean
  -> MoCo
```

ではclip内frame順序を使わない。

ImageNet-initialized VideoMAE型へ変えると、

```text
video frames
  -> spatiotemporal tokens
  -> Transformer self-attention across frames
  -> Q/V LoRA
  -> masked video reconstruction
```

となり、LoRAがcross-frame relationへ直接関与できる。

したがって研究目的
「画像で獲得した静的表現を固定し、LoRAへ動画の時間的・動的情報を追加する」
には、現行masked-mean MoCoよりも構造的に時間情報を与えやすい候補。

## 重要な注意

この方式はoriginal VideoMAEそのものではない。

呼び方としては、

- ImageNet-initialized VideoMAE
- image-pretrained ViT initialized masked video autoencoder
- VideoMAE-style masked video modeling with an ImageNet-pretrained ViT encoder

などが正確。

また、現行checkpoint `google/vit-base-patch16-224` は画像側が教師ありImageNet pretrainingである。
画像側も自己教師ありに揃えたい場合はMAE/DINO等で事前学習されたViTを同様に初期化する比較も考えられる。

## 最小実装案

最初は次の順序が安全。

1. ImageNet ViT-B/16のTransformer block weightを読み込めるVideoMAE-style encoderを作る。
2. tubelet_size=1でvideo input forwardを成立させる。
3. fixed spatiotemporal positional encodingを追加。
4. VideoMAE decoderを接続してreconstruction lossを計算。
5. ImageNet baseをfreeze。
6. Q/V LoRAを追加。
7. 1-stepでbase不変、LoRA更新、decoder更新を確認。
8. ordered / shuffled / reversedでreconstructionやfeature感度を比較。
9. 必要ならtubelet_size=2 inflationへ進む。

## 比較baseline

研究上は次の比較が解釈しやすい。

1. 現行: ImageNet ViT + LoRA + masked mean + MoCo
2. ImageNet ViT -> VideoMAE-style, base freeze + LoRA + decoder
3. VideoMAE from scratch / video-pretrained VideoMAE
4. 必要なら image-self-supervised ViT -> VideoMAE-style

これにより、

- image pretrainingを保持する効果
- temporal architecture/objectiveの効果
- LoRAによるparameter-efficient video adaptationの効果

を分離しやすい。

## 未解決事項

- tubelet_size=1か2か。
- fixedかlearnableかのspatiotemporal positional encoding。
- reconstruction decoderのdepth/width。
- decoderをどこまでtrainableにするか。
- Q/V LoRAだけで十分か。
- mask ratioをVideoMAE標準の90%付近にするか。
- ordered/shuffled/reversedのどれを学習条件・評価条件にするか。

このメモは探索記録であり、specまたは実装許可ではない。
