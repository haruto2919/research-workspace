---
date: 2026-09-18
project: sequential-video-lora-analysis
source_todo: null
topic: ViT-LoRA-MoCo実験フレームワーク候補比較
status: exploratory
tags: [brainstorm, research, framework, vit, lora, moco, sequential-loader, lightning]
---

# ViT-LoRA-MoCo実験フレームワーク候補比較

## 出発点

現在の初期実装順は、次を有力候補としている。

```text
ImageNet-pretrained ViT single-image smoke
 -> sequential_loader video forward smoke
 -> clip feature contract
 -> LoRA update smoke
 -> MoCo mechanics
 -> LoRA + MoCo + sequential video integration
```

この実装を最も効率よく始めるため、次の3 repositoryを比較した。

1. `tamaki-lab/2026_09_hayashi_streaming_video_qa@feat/streaming-text-memory`
2. `tamaki-lab/2026_06_saeki_sequential@dev`
3. `tamaki-lab/simple_cnn_training@main`

調査時branch head:
- Hayashi: `1b030974193406fa338eb5bb235a3aeb54c7cf6f`
- Saeki: `3110853dc4f7fb7f8fea699d59c78713189bcb60`
- simple_cnn_training: `e541dd98f825eb58c193ccd55cffa858392b89fe`

## 確認済み事実

### 1. Hayashi streaming_video_qa

READMEでは、実50Saladsを外部 `sequential_loader` のpublic APIだけで1 frameずつ読み、Qwen3-VLへ渡す最小構成とされている。学習loopは含まれない。

`streaming_text_memory/source.py` では次を使用している。

- `Salads50Adapter`
- `SequentialDataset`
- `SequentialVideoReader`
- `FixedChunkConfig`
- `build_sequential_dataloader`
- `sequential_sample_stream`

`scripts/smoke_sequential_loader.py` では、`SequentialSample` の次を明示的に検証している。

- `frames: [T,3,H,W]`
- `frame_indices`
- `timestamps`
- `valid_mask`
- `is_first / is_last`
- chunk間frame index連続性
- tail padding
- EOF
- public APIのみを使うこと

強み:
- 現在使いたい外部 `sequential_loader` のconsumer例として非常に直接的。
- loader内部へ依存せずpublic APIだけを使う境界が明確。
- streaming smokeが充実している。

弱み:
- 学習loopなし。
- ViT trainerなし。
- LoRA / MoCoなし。
- 主目的はVLM inference / text memory。

結論:
- 学習frameworkのbaseには向かない。
- `sequential_loader` bridgeとacceptance smokeの参照元として非常に有用。

### 2. Saeki sequential

READMEではCNN/ViT用training frameworkとして、Hydra + PyTorch Lightning経路、uv、config群、testsが整備されている。

特に現在研究と近い実装として `online_vit` が存在する。

`src/simple_cv_training/model/online_vit/frame_encoder.py`:
- Hugging Face bare `ViTModel` を使用。
- `ViTForImageClassification` ではなくtask headを持たないframe feature extractor。
- BCHW -> `[B,D]` feature。
- feature sourceは `cls`, `pooler`, `mean_patch`。
- `freeze_backbone` を設定可能。

`docs/how-to/use_online_vit.md`:
- inputはBCTHW。
- frameごとにViT featureを抽出。
- temporal logicをViTの外へ分離。
- `frame_only`, GRU/LSTM, sliding transformerを持つ。
- online inferenceとonline updateを明確に区別。
- streaming validationを持つ。

`configs/model/online_vit.yaml`:
- ViT model ID、feature source、freeze、temporal head等をconfig化。
- 現在のdefault temporal typeはGRU。

training framework:
- Hydra config。
- Lightning runner。
- optimizer / trainer / logger / checkpointの分離。
- smoke trainer config。
- tests。
- uv / pyprojectでdependencyを管理。

sequential data:
- 50Salads用 `Salads50SequentialDataset`、causal clip、framewise/target label contract、streaming evaluationを持つ。
- ただし外部 `tamaki-lab/sequential_loader` をconsumerとして使う構成ではなく、Saeki repo内独自dataset実装。

強み:
- 現在ほしい「2D ViTをframe encoderとして動画へ使う」設計がすでに存在。
- model / data / trainer / config / testが分離されている。
- current projectの旧simple frameworkと設計上近く、移植対象を理解しやすい。
- Lightning + Hydraで今後LoRA/MoCo configを追加しやすい。

弱み:
- external sequential_loader接続はそのままでは使えない。
- default online_vitにGRUが入るため、初期non-temporal baselineでは `frame_only` またはframe encoder単体へ制限する必要がある。
- LoRA / MoCoは未実装。
- 現在configのViT checkpointは研究で最終採用するcheckpointとは別途選び直す必要がある。

結論:
- 3候補の中で主training frameworkの参照元として最有力。
- 丸ごと研究手法を採用するのではなく、frame encoder、config、runner、test構造を再利用するのがよい。

### 3. simple_cnn_training

古いsimple training framework。

確認した構造:
- argparse。
- PyTorch Lightningの `SimpleLightningModel`。
- model factory。
- `ViTForImageClassification` を使う `vit_b`。
- PyTorchVideoを使うVideoFolder。
- image/video classification中心。
- DP/DDP、Comet、checkpoint等の基礎機能。

強み:
- 単純で読みやすい。
- current Ishikawa repoの構造と近い。
- ViT / video / Lightningの基本部品はある。

弱み:
- ViTがclassification head込みで、feature extractorとしてはSaekiより使いにくい。
- Hydra/config compositionがない。
- external sequential_loaderなし。
- causal / online stateの設計なし。
- LoRA / MoCoなし。
- Saeki repoがこの系統をより研究向けに拡張した形に近い。

結論:
- 最初からここを主baseにするメリットはSaekiより小さい。
- current code lineageを理解する補助としては有用。

## 比較

| 観点 | Hayashi | Saeki | simple_cnn |
|---|---|---|---|
| ViT feature extractor | なし | 強い | 画像分類head込み |
| BCTHW動画入力 | VLM frame stream | 強い | VideoFolderあり |
| sequential / online設計 | external loader利用が強い | 強いが独自dataset | 弱い |
| external sequential_loader | 最も強い | なし | なし |
| Lightning training | なし | 強い | あり |
| Hydra/config | なし | 強い | argparse |
| tests/smoke | loader/VLM smokeが強い | training/model/data testsが強い | 基本test |
| LoRA | なし | なし | なし |
| MoCo | なし | なし | なし |
| 現研究への主framework適合 | 補助 | 最有力 | 低〜中 |

## 現時点の収束

一つのrepoを丸ごと選ぶのではなく、役割分担するのが最も効率的。

### 主framework

`2026_06_saeki_sequential@dev` を主な設計参照にする。

特に借りたい考え方:
- `ViTFrameEncoder`
- bare `ViTModel` からframe featureを取る責務分離
- BCTHWをframe-wiseに処理するmodel contract
- Hydra config
- Lightning runner
- smoke config
- model/data/runner/testの分離

ただし初期baselineではSaekiのGRU等temporal headを入れず、`frame_only` またはframe feature抽出だけにする。

### sequential_loader接続

`2026_09_hayashi_streaming_video_qa@feat/streaming-text-memory` をconsumer実装の参照元にする。

特に借りたいもの:
- public APIのみでloaderを構築する方式
- `SequentialSample` contract
- frame index / valid mask / EOF / padding / sequence boundary smoke
- datasetを研究repositoryへ複製しない境界

### simple_cnn_training

主baseとしては使わない。
current repositoryとの構造的な祖先・最小training loopの参考に限定する。

## 実装への落とし込み候補

current implementation repositoryは維持し、frameworkそのものを別repoへ乗り換えるより、Saeki/Hayashiから必要部分を段階的に移植する方が変更範囲を制御しやすい。

```text
Stage 1
Saeki ViTFrameEncoder型
 -> single image -> [B,D]

Stage 2
Hayashi sequential_loader consumer型
 -> SequentialSample [T,C,H,W]
 -> preprocess
 -> ViT frame encoder
 -> [T,D] / [B,T,D]

Stage 3
non-temporal clip representation
 -> frame_only / mean pooling

Stage 4
LoRA
 -> ViT内部だけへ注入
 -> Base freeze / LoRA update smoke

Stage 5
MoCo用SSL LightningModule
 -> query/key
 -> projector
 -> EMA
 -> queue/InfoNCE（採用variantに応じる）

Stage 6
sequential_loader + ViT + LoRA + MoCo統合
```

分類用 `SimpleLightningModel` にMoCoを無理に追加するより、Saeki runnerの責務分離を参考に、MoCo用LightningModuleを別に持つ方向が有力。

## 反例・注意点

- Saeki repoをそのままcopyすると、独自Salads50 datasetやGRU temporal headまで持ち込み、外部 sequential_loaderを使うという現在方針と重複する。
- Hayashi repoをbaseにするとtraining infrastructureをほぼ新規作成する必要がある。
- simple_cnnをbaseにすると、現在必要なonline/sequential設計を再度作り直す割合が大きい。
- 3 repoともLoRA/MoCoは現時点で実装されていないため、その部分は公式MoCo実装等を別途参照して新設する必要がある。

## 未解決事項

- current implementation repoへSaeki型src/Hydra構成まで導入するか、既存directory構成のまま考え方だけ移植するか。
- 最初のViT checkpoint。
- MoCo v1/v2型queueありかMoCo v3型か。
- LoRA libraryをPEFTにするか独自実装にするか。
- `SequentialSample[T,C,H,W]` をframe encoderへloopで渡すか `B*T` batchへreshapeして一括forwardするか。

このメモは探索記録であり、specまたは実装許可ではない。
