---
project: sequential-video-lora-analysis
spec_type: implementation
status: approved
title: ViT Frame Encoderと単一画像feature extraction smoke
created: 2026-09-18
last_updated: 2026-09-22
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
workspace_base_commit: 7dc46ba54ced621da5cf426ca73704d43ad8fad7
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_base_branch: main
implementation_base_commit: e1c1715135d5f43fbaf700bbc3533ada1b367a59
implementation_work_branch: feature-vit-frame-encoder
implementation_work_branch_base_commit: e1c1715135d5f43fbaf700bbc3533ada1b367a59
hf_checkpoint: google/vit-base-patch16-224
---

# ViT Frame Encoderと単一画像feature extraction smoke spec

> **Status: approved**
>
> 本specは、simple_cnn clean baselineを起点として、動画研究で再利用できる
> frame-level ViT encoderを追加し、単一RGB画像からCLS featureを取得できることを確認する
> Stage 1の実装契約である。
>
> 2026-09-22にユーザーが承認し、Stage 1のコード実装を依頼した。

## 1. 目的

現在のresearch codeにはsimple_cnn由来の分類用ViT `ViTb` が存在し、
`ViTForImageClassification` からclass logitsを返す。

今後の動画研究ではclassification logitsではなく、各video frameを再利用可能なfeatureへ変換する
encoderが必要になる。

本specでは次の最小経路を成立させる。

```text
local RGB image
  -> AutoImageProcessor
  -> pixel_values [B, 3, 224, 224]
  -> ViTFrameEncoder
       -> ViTModel
       -> pretrained backbone
       -> CLS token extraction
  -> frame_features [B, 768]
```

このStageでは動画、LoRA、MoCo、temporal modelingは扱わない。

## 2. Authorityと基準revision

### 2.1 Authority

本specの判断根拠は、優先順に次のとおりとする。

1. 2026-09-18のユーザー指示
   - checkpointは `google/vit-base-patch16-224` を採用する。
   - feature sourceはCLS tokenを採用する。
   - 既存 `ViTb` は残し、`ViTFrameEncoder` を新規追加する。
   - Stage 1ではViT backboneをfreezeする。
   - smokeは実RGB画像 + `AutoImageProcessor` を使用する。
2. 2026-09-17 MTG
   - 画像事前学習済みViT + 動画自己教師ありLoRAへ研究中心を移す。
   - まず1方式で再現可能な開発loopを作る。
3. 2026-09-18 research overview / architecture brainstorm。
4. 現在のimplementation repository。

### 2.2 基準revision

| 役割 | repository | branch | commit |
|---|---|---|---|
| Research Workspace | `haruto2919/research-workspace` | `main` | `7dc46ba54ced621da5cf426ca73704d43ad8fad7` |
| implementation base | `tamaki-lab/2026_09_ishikawa_sequential-video-lora` | `main` | `e1c1715135d5f43fbaf700bbc3533ada1b367a59` |
| implementation work | 同上 | `feature-vit-frame-encoder` | `e1c1715135d5f43fbaf700bbc3533ada1b367a59` |

`feature-vit-frame-encoder` はspec作成時点で `main` と同じcommitを指しており、
Stage 1固有変更はまだ入っていない。

## 3. 現行実装

現在の `model/vit/vision_transformer.py` は次のclassification modelを持つ。

```python
ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    ...
)
```

現行contract:

```text
image [B,C,H,W]
  -> ViTb
  -> classification logits [B,n_classes]
```

本specではこの既存pathを変更・置換しない。

研究用に別責務として次を追加する。

```text
image/frame [B,C,H,W]
  -> ViTFrameEncoder
  -> frame feature [B,D]
```

## 4. Decision Contract

### 4.1 checkpoint

固定:

```text
google/vit-base-patch16-224
```

Stage 1では別checkpointへのfallbackを設けない。

### 4.2 model class

新しいencoderではHugging Faceのbare encoderを使用する。

```text
ViTModel
```

`ViTForImageClassification` のclassification headは研究用feature pathでは使用しない。

### 4.3 feature source

固定:

```text
CLS token
```

具体的には、

```python
outputs.last_hidden_state[:, 0, :]
```

をframe featureとする。

checkpointのhidden sizeは768であるため、Stage 1の出力contractは次とする。

```text
[B, 768]
```

mean-patchやpoolerとの比較は本specでは行わない。

### 4.4 freeze

ViT backboneの全parameterをfreezeする。

```text
requires_grad = False
```

Stage 1ではoptimizer、backward、parameter updateを行わない。

class自体に常時evalを強制する特殊なtrain/eval overrideは追加しない。
PyTorch標準の `.train()` / `.eval()` semanticsを維持する。

smoke実行時には明示的に

```python
encoder.eval()

with torch.no_grad():
    ...
```

を使用する。

### 4.5 preprocessing

smokeではcheckpoint対応のHugging Face processorを使用する。

```python
AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)
```

独自normalize / resize値をStage 1で再実装しない。

processorは `ViTFrameEncoder` の外側で使用し、
encoder自体はpreprocessed `pixel_values` を受け取る。

## 5. 実装設計

### 5.1 新規file

```text
model/vit/vit_frame_encoder.py
```

新規class:

```text
ViTFrameEncoder
```

責務は1 frame batchをViT featureへencodeすることだけとする。

### 5.2 class contract

概念interface:

```python
class ViTFrameEncoder(nn.Module):
    def __init__(
        self,
        checkpoint_id: str = "google/vit-base-patch16-224",
    ):
        ...

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

入力:

```text
pixel_values
shape: [B, 3, 224, 224]
dtype: floating point
```

出力:

```text
frame_features
shape: [B, 768]
```

forwardではclassification logits、loss、labelsを扱わない。

### 5.3 model load

```text
ViTModel.from_pretrained(checkpoint_id)
```

を使う。

checkpointのencoder weightを利用し、classification headを新設しない。

### 5.4 freeze audit

初期化後、少なくとも次を満たす。

```text
all(parameter.requires_grad is False
    for parameter in encoder.vit.parameters())
```

Stage 1ではtrainable parameterを追加しない。

### 5.5 existing classification path

以下は維持する。

```text
model/vit/vision_transformer.py
  -> ViTb
  -> ViTForImageClassification
```

`ViTb` のclassification contract、
`model_factory.py` の `vit_b` path、
既存classification testsを変更する必要はない。

新しい `ViTFrameEncoder` をclassification `ModelConfig` / `model_factory` へ無理に統合しない。

### 5.6 export

必要最小限として、

```text
model/vit/__init__.py
model/__init__.py
```

から `ViTFrameEncoder` をimport可能にしてよい。

既存 `ViTb` exportは維持する。

## 6. Single-image smoke

Stage 1のacceptance用に単一画像feature smokeを追加する。

推奨entrypoint:

```text
smoke_vit_frame_encoder.py
```

処理順:

1. local RGB画像pathを受け取る。
2. PIL等で画像をRGBとして読み込む。
3. `AutoImageProcessor` をcheckpointから構築する。
4. processorで `pixel_values` を生成する。
5. `ViTFrameEncoder` を構築する。
6. encoderがfrozenであることを確認する。
7. encoderをeval modeへする。
8. `torch.no_grad()` でforwardする。
9. input / output shapeを確認する。
10. featureがfiniteであることを確認する。
11. 最低限のdiagnosticを標準出力へ出す。

最低限記録するもの:

```text
checkpoint ID
resolved device
pixel_values shape
pixel_values dtype
feature shape
feature dtype
feature finite
total backbone parameters
trainable backbone parameters
```

期待値:

```text
pixel_values shape = [1, 3, 224, 224]
feature shape = [1, 768]
trainable backbone parameters = 0
feature finite = True
```

feature値そのものは固定値としてpass/failに使用しない。

## 7. Test方針

Stage 1の検証はfeature encoderの責務に限定する。

最低限、次を確認する。

1. `ViTFrameEncoder` をimportできる。
2. checkpointをロードできる。
3. processor後のsingle-image入力をforwardできる。
4. output shapeが `[1,768]`。
5. outputにNaN / Infがない。
6. backbone parameterが全てfrozen。
7. `ViTb` の既存classification pathを変更していない。

既存classification testの全面改修は行わない。

## 8. 変更scope

想定する変更は原則次だけ。

```text
model/vit/vit_frame_encoder.py       # 新規
model/vit/__init__.py                # export
model/__init__.py                    # 必要ならexport
smoke_vit_frame_encoder.py           # 新規
test/model/test_vit_frame_encoder.py # 必要な範囲の短時間test
```

実装時に上記以外の変更が必要になった場合、
Stage 1の責務に直接必要かを確認する。

## 9. 明示的な対象外

本specでは次を実装・検証しない。

- `sequential_loader`
- video clip入力
- `[B,T,D]` reshape
- clip mean pooling
- temporal aggregation
- temporal modeling
- LoRA
- Base ViT + LoRA parameter audit
- MoCo
- query / key encoder
- EMA
- InfoNCE
- queue
- optimizer
- backward
- fine-tuning
- full training
- dataset選定
- CLS vs mean-patch比較
- CLIP-ViT
- downstream classification評価
- temporal information評価

## 10. Success Criteria

本specの実装成功は次を全て満たすこととする。

1. implementation branchが `feature-vit-frame-encoder` である。
2. branchの基準commitが `e1c1715135d5f43fbaf700bbc3533ada1b367a59` である。
3. `google/vit-base-patch16-224` を使用する。
4. bare `ViTModel` を用いた `ViTFrameEncoder` が存在する。
5. 既存 `ViTb` を削除・置換していない。
6. `ViTFrameEncoder` がpreprocessed BCHW tensorを受け取る。
7. CLS tokenをfeatureとして返す。
8. RGB画像1枚のprocessor後shapeが `[1,3,224,224]`。
9. feature shapeが `[1,768]`。
10. featureがfinite。
11. ViT backboneのtrainable parameter数が0。
12. smokeでeval mode + `torch.no_grad()` を使用する。
13. Stage 2以降のvideo / LoRA / MoCo logicを混入していない。
14. 既存classification pathに不要なbreaking changeを導入していない。

## 11. Failure / Stop Conditions

次の場合は条件を黙って変更せず停止・報告する。

- checkpointをロードできない。
- `AutoImageProcessor` をcheckpointから構築できない。
- `ViTModel.from_pretrained` で想定外のweight incompatibilityが発生する。
- processor後shapeが期待contractと一致しない。
- output shapeが `[B,768]` にならない。
- outputにNaN / Infが含まれる。
- backboneにtrainable parameterが残る。
- Stage 1成立のために既存classification interface変更が必要になる。

この場合、別checkpoint、mean-patch、独自transform等へ勝手に切り替えない。

## 12. Ambiguity Gate

### 12.1 Blocking

**なし。**

2026-09-18のユーザー判断により次を採用済み。

- checkpoint: `google/vit-base-patch16-224`
- feature source: CLS token
- existing `ViTb`: 維持
- new class: `ViTFrameEncoder`
- new file: `model/vit/vit_frame_encoder.py`
- backbone: frozen
- smoke: local RGB image + `AutoImageProcessor`
- smoke inference: eval + no_grad

### 12.2 Non-blocking

既存styleに従って決めてよい。

- smoke CLI argument名
- local RGB画像の具体的内容
- diagnostic printの表示形式
- test function名
- exportを `model/vit/__init__.py` のみにするか `model/__init__.py` にも出すか
- device選択の細かなCLI表現

これらはinput/output contract、checkpoint、freeze、feature sourceを変更してはならない。

## 13. 実装後の次段階

本specがimplementedになった後、Stage 2を別specとして扱う。

```text
external sequential_loader
  -> video clip [T,C,H,W]
  -> preprocess
  -> ViTFrameEncoder
  -> frame features [T,768]
```

Stage 2でもまずframe-level feature extractionの成立を確認し、
clip aggregation / LoRA / MoCoはさらに後段へ分離する。

## 14. Spec Gate

本specでは、checkpoint、feature source、freeze、class責務、
既存classification pathとの互換性、single-image smoke、Success Criteria、対象外を固定した。

blocking ambiguityは残っていない。

2026-09-22にユーザーが承認したため、statusを `approved` とし、engineering-taskへ引き継ぐ。

# Implementation Handoff

- approved spec: 本spec
- 実装目的: pretrained ViTからsingle-frame CLS featureを取得する研究用encoderを追加
- 基準repository/commit: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@e1c1715135d5f43fbaf700bbc3533ada1b367a59`
- work branch: `feature-vit-frame-encoder`
- checkpoint: `google/vit-base-patch16-224`
- 変更scope: `ViTFrameEncoder` + export + single-image smoke +最小test
- 対象外: video / sequential_loader / LoRA / MoCo / training
- success criteria: 10章
- 許可されている短時間検証: import / single-image forward smoke /短時間test
- 長時間runの許可状態: 未許可
