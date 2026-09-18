---
date: 2026-09-18
project: sequential-video-lora-analysis
source_todo: null
topic: 最初のMoCo+LoRA実装で通常ViTとCLIP-ViTのどちらを使うか
status: exploratory
tags: [brainstorm, research, vit, clip, moco, lora, baseline, self-supervised-learning]
---

# 最初のMoCo+LoRA実装で通常ViTとCLIP-ViTのどちらを使うか

## 出発点

加藤さんの卒論発表で用いられているMoCo型のオンライン対照学習を、現在の「画像事前学習済みViT + LoRA + 動画自己教師あり学習」へ応用する案を検討している。

今回の問いは、最初の開発baselineとして次のどちらを使うかである。

- ImageNet等で通常の画像分類事前学習を行ったViT
- CLIPでimage-text contrastive pretrainingを行ったViT image encoder

ここで「通常ViT」は、random initializationではなく、ImageNet-1K等でsupervised pretraining済みのViTを想定する。

## 現在の研究目的

2026-09-17 MTGでは、最終方式を先に固定せず、まず1方式で動画LoRAの学習・観測・比較まで一巡できる開発ループを成立させることが最優先とされた。

そのため最初のbaselineでは、新規性よりも以下を優先する。

- 実装の単純さ
- MoCo / LoRA / EMA / queueの挙動を切り分けやすいこと
- 時系列順 vs shuffle等を比較しやすいこと
- 後からbackbone事前学習の違いだけを比較できること

## 通常ViTを最初に使う利点

### 1. MoCo v3との整合性が高い

MoCo v3はViTを自己教師あり学習するbaselineとして検証されており、通常のViT backboneでの学習知見がある。

CLIP固有のimage-text projectionやtext encoderを考えずに、visual backbone + LoRA + projection head + momentum encoderへ集中できる。

### 2. 研究変数を減らせる

最初からCLIPを使うと、少なくとも以下の要素が追加される。

- CLIP image projectionをどこまで使うか
- CLIP固有のpreprocess / normalization
- LoRA更新後にimage-text alignmentが崩れていないか
- text encoderを評価に使うか
- MoCo featureをCLIP shared embedding spaceで取るか、ViT内部featureで取るか

通常ViTなら、「MoCo + LoRAが動画で動くか」の検証へ集中しやすい。

### 3. CLIPとの比較が後からきれいになる

同じViT-B/16級のarchitectureを使い、

```text
A. ImageNet-pretrained ViT-B/16 + LoRA + MoCo
B. CLIP-pretrained ViT-B/16     + LoRA + MoCo
```

として、可能な限り同じLoRA位置、rank、optimizer、video sampling、MoCo objectiveを使えば、「初期事前学習の違い」が動画LoRAへ与える影響を比較しやすい。

## CLIP-ViTを最初に使う利点

- semantic image priorが強い。
- frozen text encoderをsemantic probeとして使える。
- 最終的に「LoRAが何を獲得したか」をaction / object / sceneのtext promptで解析しやすい。
- 研究目的の「LoRA内部の獲得情報解析」には相性が良い。

ただし、これは学習ループ成立後に特に価値が高い利点であり、最初のsmoke/baseline段階では追加の設計変数にもなる。

## 現時点の収束

### 最初の実装候補

通常のImageNet-pretrained ViT-B/16級を使う。

```text
ImageNet-pretrained ViT
  + LoRA
  + Kato-style / MoCo-style query-key learning
  + momentum encoder
  + queue or chosen MoCo variant
  + InfoNCE
```

Base ViTはfreezeし、まずLoRA更新、momentum/EMA更新、queue、InfoNCE、sequential loaderとの接続を確認する。

この段階では時間情報獲得を主張しない。

### 次の比較

同じコードパスを維持してCLIP-pretrained ViTへbackboneを交換する。

その後、

- ordinary ViT + LoRA + MoCo
- CLIP ViT + LoRA + MoCo

を比較し、CLIP text encoderを評価probeとして追加する。

## 理由

現在の最優先が「最も早く、解釈可能な学習ループを成立させること」であるため、通常ViTの方が初期baselineとして因果を切り分けやすい。

一方、最終的な研究主張やLoRA semantic analysisではCLIP-ViTの価値が高いため、CLIPを棄却するのではなく第2段階の比較条件として残す。

## 実装段階案

```text
Stage 1
ImageNet ViT + LoRA + MoCo
-> 1 step / EMA / queue / InfoNCE / LoRA update確認

Stage 2
sequential vs random/shuffle
-> online条件の最小比較

Stage 3
CLIP-ViTへ同じframeworkを差し替え
-> pretraining priorの違いを比較

Stage 4
CLIP text encoderをprobeとして利用
-> LoRA学習前後のsemantic / temporal sensitivity解析
```

## 未解決事項

- 最初の通常ViT checkpointをどれにするか。
- MoCo v1/v2型queueを使うか、ViT向けMoCo v3構成へ寄せるか。
- 1 clipを2D ViTでどうrepresentするか。
- temporal moduleをbaseline時点で入れるか。
- ordinary ViTとCLIP ViTで同一architecture条件をどこまで揃えられるか。

## 関連文書

- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-17-mtg.md`
- `.research/secretary/notes/brainstorm/2026-09-17-clip-vit-moco-video-lora.md`
- `.research/secretary/notes/brainstorm/2026-09-18-kato-moco-clip-vit-lora.md`

このメモは探索記録であり、specまたは実装許可ではない。
