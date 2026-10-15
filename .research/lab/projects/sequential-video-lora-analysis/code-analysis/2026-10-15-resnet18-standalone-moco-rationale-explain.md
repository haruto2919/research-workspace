---
project: sequential-video-lora-analysis
date: 2026-10-15
analysis_mode: explain
detail: compact
level: junior
source_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_branch: dev
source_commit: 8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a
source_dirty: unknown (GitHub cannot observe SSH worktree)
targets:
  - self_supervised/moco/standalone_moco.py
  - conf/moco_standalone.yaml
  - scripts/smoke/smoke_moco_cifar10.py
related_spec: ../specs/2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md
related_evidence: ../experiments/2026-10-15-standalone-moco-gpu-64step-and-gate-c-500step-evidence.md
---

# Standalone MoCoでResNet18を採用した理由（実装解説）

## 対象と解析の範囲

2026-10-15時点のGitHub公開 `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`（HEAD `8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a`）にあるStandalone MoCoのEncoder選択を説明する。現在のSSHサーバ上の未commit変更やGPUプロセスは未確認。対象はResNet18採用理由・実コードと承認spec・短時間実験結果の関係。旧ActivityNet統合版の最新ローカルコードやViT移行実装は検証範囲外。

## Overview: What / Why

**結論**：ResNet18は最終研究モデルの決定ではなく、MoCoのQueue・Key EMA・InfoNCE・正負ペアなどの**機構をViT/LoRA/Sequential Loaderから切り離して独立検証するための一時的なbackbone**である。

- 2026-10-08 MTG後、旧ActivityNet統合MoCoで見られたLoss振動・Query LoRA勾配急増の原因を切り分ける方針となった。
- [承認済みspec](../specs/2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md)はCIFAR-10＋ResNet18、LoRA/ViT/Sequential LoaderなしでMoCo v2型mechanicsを検証する設計と明記している。
- Meta公式のMoCo v2はResNet50/ImageNet/DDPを基準とする。今回のResNet18/CIFAR-10/single-deviceは公式の精度再現ではなく、公式参考コードに近い処理順を小規模に確認する実装である。
- ResNet18は比較的軽量でtorchvisionにあり、32×32画像へのstem調整が容易。ViT固有のPatch設定やLoRA更新の問題を持ち込まず、問題の所在を限定できる。一方、「ResNet18だからMoCoの表現が良くなる」とは確認されていない。

## Key Components and Data Flow

| ファイル | コード上で確認した役割 |
| --- | --- |
| `self_supervised/moco/standalone_moco.py` | `CifarResNet18`: `resnet18(weights=None)`、3×3 stride1 convへの変更、`maxpool=Identity`、`fc=Identity`、512次元特徴量を出力 |
| 同上 | `projection_head`: `512→512→128`の2層MLP（中間ReLU） |
| 同上 | `StandaloneMoCo`: QueryとKeyを別storageに保持し、KeyはEMA、QueueはFIFO、正例/負例でInfoNCEを計算 |
| `scripts/smoke/smoke_moco_cifar10.py` | CIFAR-10の同じ画像から独立に2 view生成、ラベルはcontrastive trainingに使わない |
| `conf/moco_standalone.yaml` | `resnet18_cifar`、Queue1024、m0.999、T0.2、batch32、seed0、初期LR0.00375 |

処理の概略：
`CIFAR10画像 → 独立な2つの画像拡張 → ResNet18 Query/Key → MLP Projector → L2正規化128次元 → 現在画像2viewをpositive、FIFO Queueをnegative → InfoNCE → Query側だけSGD更新`。Key側パラメータはQueryからEMA更新。

## Concepts and Design Decisions

- ResNetは**特徴抽出器（Encoder）**であり、MoCoそのものではない。MoCoは「同じ画像の2つのviewの特徴を近づけ、Queueに入った特徴と区別する」という**学習機構**。
- `weights=None`なのでImageNet事前学習済みResNetではない。random initから学習する。
- ResNet18の分類Headを使わない。512次元をprojectorに渡し、128次元でcosine類似度を求める。
- 32×32の小画像でImageNet版の大きいstem（7×7 stride2）とmaxpoolをそのまま使うと初期から空間解像度を強く落とす。3×3 stride1・maxpoolなしはCIFAR向け調整。
- `StandaloneMoCo(backbone)`のようにEncoderを差し替えられる構造とし、次段でFrozen ViTを接続する余地を残している。ただし今回の実行スクリプトはResNet18/CIFAR-10のみ正式対応。

## Alternatives and Trade-offs

| 候補 | 利点 | 本段階での制約 |
| --- | --- | --- |
| ResNet18（採用） | 小規模・標準実装・32×32対応、MoCo機構だけを検証しやすい | 最終目的のViT/LoRA/動画を直接検証できない |
| ResNet50 | 公式MoCo v2により近い | CIFAR-10単体mechanicsの検証には計算負荷が増える |
| ViT-Base | 最終研究のEncoder構造に近い | 画像Patch/学習安定性/事前学習/LoRA等の影響とMoCoを再び混同しやすい |

推奨される次段階の方向性（*未検証の提案*）：まず同じMoCo coreのEncoderをViTへ交換してEncoder差を分離し、その後LoRA・Sequential Loaderを段階的に追加。比較する際はdatasetや学習率等の変更軸を明記し、単純なLoss比較をしない。

## Evidence and Limits

- 2026-10-15のユーザー提出のGPUサマリーを整理した[Evidence](../experiments/2026-10-15-standalone-moco-gpu-64step-and-gate-c-500step-evidence.md)では、64/500stepの両runで14工学チェックPASS、Key勾配ゼロ、EMA・FIFO・Query更新・checkpoint復元を報告。
- 同じEvidenceでは、500step終盤Lossがstep450で約6.26、step500で約6.64、positiveとhardest negativeの関係は十分でない。工学PASSを良い表現学習、ViTへの適合、時間情報の獲得と同一視しない。
- これらはユーザー提供の実験サマリーに基づく記録であり、ChatGPTが実験サーバの生データやcheckpointを直接検証したものではない。

## 根拠

- [GitHub code: `standalone_moco.py`](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/self_supervised/moco/standalone_moco.py)
- [GitHub code: `smoke_moco_cifar10.py`](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/smoke/smoke_moco_cifar10.py)
- [GitHub config: `moco_standalone.yaml`](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/conf/moco_standalone.yaml)
- [approved spec](../specs/2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md)
- [GPU evidence](../experiments/2026-10-15-standalone-moco-gpu-64step-and-gate-c-500step-evidence.md)
