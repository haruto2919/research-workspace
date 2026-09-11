---
date: 2026-09-11
project: sequential-video-lora-analysis
source_todo: null
topic: sequential_loaderから50SaladsをImageNet事前学習MAEへ通す事前確認
status: exploratory
tags: [brainstorm, research, mae, sequential-loader, 50salads, smoke-test]
---

# sequential_loaderから50SaladsをImageNet事前学習MAEへ通す事前確認

## 相談

LoRA追加や動画向けtemporal designへ進む前に、一度 `sequential_loader -> 50Salads -> pretrained MAE` のforwardが動くことを確認する必要があるか。

## 結論

研究主張そのものに必須ではないが、実装を段階的に切り分けるためのintegration smoke testとしては実施する価値が高い。

この確認で検証するのは「MAEが動画の時間情報を学習できるか」ではなく、以下のinterfaceが成立するかである。

- `sequential_loader` から50Salads frameを取得できる。
- loader出力をMAEが要求する画像tensorへ変換できる。
- pretrained MAEへforwardできる。
- MAE reconstruction lossが有限値として計算できる。
- sequence順やframe metadataを壊さず後続の動画設計へ渡せる。

## この段階で主張してはいけないこと

標準Image MAEは画像入力を前提とするため、`sequential_loader` が時系列順にframeを返していても、各frameを独立にMAEへ入れるだけならtemporal relationをモデルが利用しているとは言えない。

したがってこのsmoke testは次を示さない。

- online temporal learningが成立したこと。
- 動画の動的性を学習したこと。
- LoRAが時間情報を獲得したこと。
- late fusionやtemporal attentionの設計が妥当であること。

## 推奨する最小確認

大規模学習は不要で、1〜数clipで十分。

1. `sequential_loader` から50Saladsの1 sequenceを取得する。
2. 返された `frames` と lifecycle / frame index等の主要metadataを確認する。
3. MAE用のresize・normalizeへ変換する。
4. Image MAEは `[B,C,H,W]` を前提とするため、最初は各frameを画像batchとして扱う。
5. ImageNet-pretrained `ViTMAEForPreTraining` へ入力する。
6. reconstruction `loss`、出力shape、NaN/Infがないことを確認する。

## 実装順序への提案

現在の作業順では、この確認をLoRA追加より前に置くのが分かりやすい。

1. Hugging Face ImageNet-pretrained MAEをmodel factoryへ追加。
2. 単一画像で標準MAE forward/lossを確認。
3. **`sequential_loader -> 50Salads -> MAE` integration smoke test。**
4. MAEへLoRAを追加し、Base freeze / LoRA updateを確認。
5. 動画temporal design（late fusion等）を決める。
6. ActivityNet / EPIC-KITCHENS等で動画自己教師あり学習へ進む。

この順序なら、LoRA追加後に問題が起きても、loader/MAE接続とLoRA実装を切り分けやすい。

## 50Saladsを使う意味

50Saladsを最終実験datasetとして採用する必要はない。既に扱ってきた動画datasetとsequential loaderを使って、最小の接続確認を行うためのengineering fixtureとして使えばよい。

教師ラベルはこのMAE smoke testでは原則不要であり、自己教師ありreconstructionの入力frameとしてのみ使用する。

## 未解決

- MAE入力時に時系列次元を単純にbatchへ畳むか、1 frameずつforwardするか。
- loaderのどのmetadataを後続のonline temporal designで保持するか。
- このsmoke testをPR 1（MAE integration）へ含めるか、loader integration用PRへ分けるか。

このメモは探索記録であり、specまたは実装許可ではない。
