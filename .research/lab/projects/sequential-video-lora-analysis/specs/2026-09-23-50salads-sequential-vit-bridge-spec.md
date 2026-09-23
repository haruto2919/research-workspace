---
project: sequential-video-lora-analysis
spec_type: implementation
status: draft
title: Stage 2 50Salads sequential_loaderとViTFrameEncoder接続
created: 2026-09-23
last_updated: 2026-09-23
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
workspace_base_commit: f5fe76684d763dff7cf272333fdecc06fcb1604a
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_base_branch: main
implementation_base_commit: e7e037a9191f36b26e87f80f48caccfb35b6166d
implementation_work_branch: feature-50salads-loder
implementation_work_branch_base_commit: e7e037a9191f36b26e87f80f48caccfb35b6166d
sequential_loader_repository: tamaki-lab/sequential_loader
sequential_loader_branch: master
sequential_loader_commit: cef09aa12560127451a5f569d86d5d51671e6986
hf_checkpoint: google/vit-base-patch16-224
dataset_fixture: 50Salads
dataset_split: train1
frames_per_chunk: 16
---

# Stage 2 50Salads sequential_loaderとViTFrameEncoder接続 spec

> **Status: draft**
>
> 本specは、Stage 1で実装済みの `ViTFrameEncoder` と、
> external `tamaki-lab/sequential_loader` の `SequentialSample` を接続し、
> 50Saladsの1 video chunkをframe feature列 `[T,768]` へ変換できることを確認する
> Stage 2の実装契約候補である。
>
> 本specはユーザーの「specを作成してください」という依頼に基づくdraftであり、
> コード実装の許可ではない。ユーザー承認後に `approved` へ変更する。

## 1. 目的

Stage 1では、`google/vit-base-patch16-224` のpretrained ViTを用いて、
単一RGB画像をCLS feature `[B,768]` へ変換する `ViTFrameEncoder` を実装し、
実checkpointを用いたsingle-image smokeを確認した。

Stage 2では、動画を時間順chunkとして返す `sequential_loader` と
`ViTFrameEncoder` の接続だけを成立させる。

最小経路は次とする。

```text
50Salads
  -> Salads50Adapter
  -> SequentialDataset
  -> SequentialSample
       frames [T,3,H,W]
       valid_mask [T]
       frame_indices [T]
       timestamps [T]
  -> valid frame extraction
  -> AutoImageProcessor
  -> pixel_values [N_valid,3,224,224]
  -> frozen ViTFrameEncoder
  -> valid_features [N_valid,768]
  -> scatter back to T positions
  -> frame_features [T,768]
```

このStageでは学習・勾配更新を行わない。

## 2. Authorityと基準revision

### 2.1 Authority

本specの判断根拠は、優先順に次のとおりとする。

1. 2026-09-23のユーザー指示
   - Stage 2 specを作成する。
   - 実装先branchは既存の `feature-50salads-loder` とする。
2. 2026-09-23 Stage 2 brainstorm
   - 50Saladsをengineering fixtureとしてgeneric loader-to-ViT bridgeを先に確認する。
   - ActivityNetは最終dataset候補だが、ActivityNet Adapterは別作業単位とする。
   - `ViTFrameEncoder` 自体を動画対応へ変更せず、複数frameをimage batchとして渡す。
   - Stage 2では勾配更新を行わない。
3. Stage 1 implemented spec
   - `google/vit-base-patch16-224`
   - CLS feature
   - backbone freeze
   - `ViTFrameEncoder` のinput/output contract
4. `tamaki-lab/sequential_loader` のpublic contract。

### 2.2 基準revision

| 役割 | repository | branch | commit |
|---|---|---|---|
| Research Workspace | `haruto2919/research-workspace` | `main` | `f5fe76684d763dff7cf272333fdecc06fcb1604a` |
| implementation base | `tamaki-lab/2026_09_ishikawa_sequential-video-lora` | `main` | `e7e037a9191f36b26e87f80f48caccfb35b6166d` |
| implementation work | 同上 | `feature-50salads-loder` | `e7e037a9191f36b26e87f80f48caccfb35b6166d` |
| sequential loader | `tamaki-lab/sequential_loader` | `master` | `cef09aa12560127451a5f569d86d5d51671e6986` |

spec作成時点で `feature-50salads-loder` は `main` と同一commitを指し、
Stage 2固有変更はまだ入っていない。

## 3. 現行contract

### 3.1 ViTFrameEncoder

既存 `model/vit/vit_frame_encoder.py` は次のcontractを持つ。

```text
input:
pixel_values [N,3,224,224]

output:
CLS feature [N,768]

backbone:
google/vit-base-patch16-224
frozen
```

Stage 2ではこのclass contractを変更しない。

`N` は画像batch数であり、1 video chunkのvalid frame数 `N_valid` をそのまま
image batchとして与えられる。

### 3.2 SequentialSample

pinned `sequential_loader` のpublic contractでは、

```text
frames         CPU uint8 [T,3,H,W], RGB
frame_indices  int64 [T]
timestamps     float64 [T]
valid_mask     bool [T]
sequence_id
sequence_index
is_first
is_last
```

を持つ。

strict DataLoaderは `batch_size=1` であり、
`frames` へbatch dimensionを追加しない。

tail paddingは、

```text
frames        -> zero
frame_indices -> -1
timestamps    -> NaN
valid_mask    -> False
```

で表現される。

## 4. Decision Contract

### 4.1 Stage 2 dataset fixture

Stage 2のintegration smokeには50Saladsを使用する。

```text
adapter: Salads50Adapter
split: train1
frames_per_chunk: 16
```

50SaladsはStage 2のengineering fixtureであり、
最終研究datasetとして採用したことを意味しない。

最終dataset候補のActivityNet対応は本specの対象外とする。

### 4.2 dataset root

50Saladsの絶対pathは環境依存のためspecへ固定しない。

smoke entrypointへ `dataset_root` をruntime引数として渡す。

### 4.3 source selection

smokeでは `Salads50Adapter.sequence_sources("train1")` が
決定的に列挙するsource順をそのまま使用する。

最初のintegration checkでは先頭sequenceの先頭chunkを対象としてよい。

source順を独自にshuffleしない。

### 4.4 chunk size

固定:

```text
T = 16
```

`FixedChunkConfig(frames_per_chunk=16)` を用いる。

この値はStage 2 smokeのengineering条件であり、
最終研究条件として最適と主張しない。

### 4.5 preprocessing

Stage 1と同じcheckpoint対応processorを使用する。

```python
AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)
```

独自resize / normalizeを再実装しない。

processorは `ViTFrameEncoder` の外側で使用する。

### 4.6 valid frameとpadding

`valid_mask=True` のframeだけをprocessor / ViTへ渡す。

```text
frames [T,3,H,W]
valid_mask [T]
  -> select valid positions
valid_frames [N_valid,3,H,W]
  -> processor
pixel_values [N_valid,3,224,224]
  -> ViTFrameEncoder
valid_features [N_valid,768]
```

padding位置をzero imageとしてViTへ入力しない。

encode後は元のT位置へfeatureを戻し、

```text
frame_features [T,768]
```

を作る。

`valid_mask=False` の行はzero vectorとする。

```text
frame_features[~valid_mask] == 0
```

これにより、

```text
frame_features[i]
frame_indices[i]
timestamps[i]
valid_mask[i]
```

が同じframe位置を表し続ける。

### 4.7 temporal order

frame順を変更しない。

特にRAW timestampは単調増加を保証しないため、
`timestamps` を用いたsortを行わない。

順序は `SequentialSample` が保持するsource / chunk / frame-index orderをそのまま使う。

### 4.8 metadata

最低限、Stage 2 smokeで次を観測・保持する。

```text
valid_mask
frame_indices
timestamps
sequence_id
sequence_index
is_first
is_last
```

`evaluation_reference` はmodel inputへ渡さない。

Stage 2ではannotationを学習labelとして使用しない。

### 4.9 ViTFrameEncoder

既存 `ViTFrameEncoder` の動画専用化は行わない。

追加しないもの:

- temporal dimensionの特別処理
- valid_mask処理
- padding処理
- frame間attention
- temporal positional embedding
- mean pooling
- sequential_loader import
- `torch.no_grad()` のhard-code

Stage 2のvideo/chunk semanticsはencoder外側で扱う。

### 4.10 gradient / update

Stage 2では学習しない。

smoke実行時は、

```python
encoder.eval()

with torch.no_grad():
    ...
```

を使用する。

次を行わない。

```text
loss.backward()
optimizer
optimizer.step()
parameter update
```

backboneはStage 1と同じく全parameter frozenを維持する。

ただし `ViTFrameEncoder.forward()` 自体へ `no_grad` を埋め込まない。
後のLoRA stageで同じforward pathをgradient付きで利用できる状態を保つ。

### 4.11 external sequential_loader dependency

research codeへ `sequential_loader` の実装をcopy / vendorしない。

pinned repository revision、

```text
tamaki-lab/sequential_loader
master@cef09aa12560127451a5f569d86d5d51671e6986
```

のpublic APIのみを利用する。

local cloneの絶対pathは環境依存でありspecへ固定しない。

環境構築時は当該revisionのexternal repositoryを利用側venvへ
editable installできることを前提とする。

本specでは `requirements.txt` へGit dependencyを新規追加することを必須としない。

## 5. 実装設計

### 5.1 implementation branch

固定:

```text
feature-50salads-loder
```

branch名の `loder` 表記は既存branch名としてそのまま使用する。

spec作成時点のbase:

```text
e7e037a9191f36b26e87f80f48caccfb35b6166d
```

### 5.2 Stage 2 entrypoint

Stage 2では、まずintegration smokeを追加する。

推奨entrypoint:

```text
smoke_50salads_vit_bridge.py
```

責務:

1. dataset rootをruntime引数で受け取る。
2. `Salads50Adapter` から `train1` sourceを得る。
3. `SequentialDataset` と `FixedChunkConfig(frames_per_chunk=16)` を構築する。
4. strict sequential DataLoaderから1 `SequentialSample` を取得する。
5. resource lifecycleを守り、途中停止時もreaderをcloseする。
6. valid frameを抽出する。
7. `AutoImageProcessor` でbatch preprocessingする。
8. frozen `ViTFrameEncoder` へvalid frame batchをforwardする。
9. featureをT位置へscatterする。
10. shape / dtype / finite / padding / metadata / freezeを検証する。
11. diagnosticを標準出力へ出す。

### 5.3 reusable abstraction

Stage 2では新しい `VideoEncoder`、
`SequentialLoaderAdapter`、`SequentialBatch` 等を必須追加しない。

まず `SequentialSample` public APIを直接consumeして
integration contractを確認する。

同じ変換処理をStage 3以降で再利用する必要が明確になった場合、
別specで薄いwrapperへ切り出すことを検討する。

### 5.4 複数batchへの将来拡張

本specではstrict loaderの `batch_size=1` に合わせて、

```text
[T,C,H,W] -> [T,768]
```

だけを実装・検証する。

ただし現在の設計は将来、

```text
[B,T,C,H,W]
 -> [B*T,C,H,W]
 -> ViTFrameEncoder
 -> [B*T,768]
 -> [B,T,768]
```

へ拡張可能である。

`B>1` のtraining semanticsは本specでは決めない。

## 6. Smoke diagnostic

最低限、次を標準出力またはassertionで確認する。

```text
implementation branch / expected base
sequential_loader revision（確認可能な範囲）
checkpoint ID
resolved device

sequence_id
sequence_index
is_first
is_last

frames shape
frames dtype
valid count / T
frame_indices
timestamps

pixel_values shape
pixel_values dtype

valid feature shape
frame_features shape
frame_features dtype
feature finite
padding rows zero

total backbone parameters
trainable backbone parameters
```

期待する中心shape:

```text
sample.frames        = [16,3,H,W]
sample.valid_mask    = [16]
pixel_values         = [N_valid,3,224,224]
valid_features       = [N_valid,768]
frame_features       = [16,768]
trainable backbone parameters = 0
feature finite = True
```

## 7. Test / verification方針

Stage 2では実50Saladsを用いたintegration smokeを必須とする。

最低限確認すること:

1. pinned `sequential_loader` public APIをimportできる。
2. `train1` から `SequentialSample` を取得できる。
3. `frames` がCPU `uint8 [T,3,H,W]`。
4. validな `frame_indices` が入力順のまま保持される。
5. valid frameだけをprocessorへ渡せる。
6. processor後shapeが `[N_valid,3,224,224]`。
7. `ViTFrameEncoder` outputが `[N_valid,768]`。
8. T alignmentを復元した `frame_features` が `[T,768]`。
9. valid featureがfinite。
10. invalid rowがzero。
11. backbone trainable parameter数が0。
12. backward / optimizer stepを実行していない。
13. `evaluation_reference` をmodelへ渡していない。

先頭chunkがpaddingを含まない場合、
padding scatter logicは短時間のsynthetic/mocked checkで検証してよい。
padding確認だけを目的に長い動画全体をViT forwardする必要はない。

## 8. 変更scope

想定する最小変更:

```text
smoke_50salads_vit_bridge.py   # 新規 integration smoke
```

padding / scatter等のロジックを短時間test可能にするため、
小さいhelperまたはtest fileが直接必要になった場合は追加してよい。

ただし、Stage 2のためだけに大きなmodel hierarchyやdataset abstractionを新設しない。

既存 `model/vit/vit_frame_encoder.py` は原則変更しない。

## 9. 明示的な対象外

本specでは次を実装・検証しない。

- ActivityNet Adapter
- ActivityNet integration smoke
- final dataset selection
- clip mean pooling
- clip representation
- temporal aggregation
- temporal modeling
- frame間attention
- LoRA
- LoRA target / rank / alpha / dropout
- MoCo
- query / key encoder
- EMA
- InfoNCE
- queue
- loss計算
- optimizer
- backward
- parameter update
- training loop
- sequential vs shuffle比較
- B>1 training
- downstream classification
- temporal information評価
- CLIP-ViT比較

## 10. Success Criteria

本specの実装成功は次を全て満たすこととする。

1. 実装branchが `feature-50salads-loder` である。
2. branch baseが `e7e037a9191f36b26e87f80f48caccfb35b6166d` である。
3. `sequential_loader` は `cef09aa12560127451a5f569d86d5d51671e6986` のpublic APIを基準とする。
4. 50Salads `train1`、`frames_per_chunk=16` で `SequentialSample` を取得できる。
5. `SequentialSample.frames` を `[T,3,H,W]` のframe列として扱う。
6. valid frameだけをcheckpoint-compatible processorへ入力する。
7. processor後のvalid image batchが `[N_valid,3,224,224]`。
8. 既存 `ViTFrameEncoder` から `[N_valid,768]` のCLS featureを得る。
9. featureを元のT位置へ戻し `frame_features [T,768]` を得る。
10. `valid_mask=False` のfeature rowがzero。
11. valid featureがfinite。
12. `frame_indices / timestamps / valid_mask` の位置対応を変更しない。
13. timestampによるreorderを行わない。
14. `sequence_id / sequence_index / is_first / is_last` を観測できる。
15. `evaluation_reference` をmodel inputへ渡さない。
16. ViT backboneのtrainable parameter数が0。
17. smokeで `eval()` + `torch.no_grad()` を用いる。
18. backward / optimizer / parameter updateを行わない。
19. 既存 `ViTFrameEncoder` contractを壊さない。
20. ActivityNet / pooling / LoRA / MoCo logicを混入していない。

## 11. Failure / Stop Conditions

次の場合は条件を黙って変更せず停止・報告する。

- pinned `sequential_loader` revisionのpublic APIを利用できない。
- 50Salads dataset layoutが `Salads50Adapter` の想定と一致しない。
- `train1` が存在しない、またはsourceを取得できない。
- decoded sampleがpublic `SequentialSample` contractと一致しない。
- valid frameが0枚になる。
- processor後shapeが `[N_valid,3,224,224]` にならない。
- ViT outputのhidden sizeが768でない。
- valid featureにNaN / Infが含まれる。
- padding rowを0として復元できない。
- frame order / metadata alignmentを保つためにloader internalsの改変が必要になる。
- Stage 2成立のために `ViTFrameEncoder` のtemporal化が必要になる。

この場合、独自loader、別checkpoint、timestamp sort、全padding encode等へ勝手に切り替えない。

## 12. Ambiguity Gate

### 12.1 Blocking

draft作成時点では、実装開始を妨げる技術的blocking ambiguityは置かない案とする。

以下をdraftの採用候補として固定した。

- work branch: `feature-50salads-loder`
- loader revision: `cef09aa12560127451a5f569d86d5d51671e6986`
- engineering fixture: 50Salads
- split: `train1`
- chunk size: 16
- `SequentialSample` public APIを直接consume
- valid frameだけencode
- output alignment: `[T,768]`
- invalid feature row: zero
- timestamp sortなし
- `ViTFrameEncoder` は変更しない
- Stage 2はforward smokeのみで学習なし
- ActivityNet対応は別spec

ただし本specは `draft` であり、
これらを実装契約として有効化するにはユーザーの明示承認が必要である。

### 12.2 Non-blocking

既存styleと実行環境に合わせて決めてよい。

- local 50Salads dataset root
- local `sequential_loader` clone path
- smoke CLI optionの細かな名前
- diagnostic print形式
- CUDA / CPUの自動device選択表現
- padding check用helper / test function名
- feature tensorをsmoke終了時にCPUへ移すかどうか
- exact test file名

これらはdataset semantics、frame順、shape、padding contract、freeze条件を変更してはならない。

## 13. Stage 2完了後の次段階

本specがimplementedになった後、最終dataset候補のActivityNetについて
dataset-specific Adapterを別作業単位で設計・実装する。

目標は、

```text
ActivityNet
 -> ActivityNet Adapter
 -> same SequentialSample public contract
 -> same Stage 2 bridge
 -> frame_features [T,768]
```

を成立させること。

50Salads smoke成功だけをActivityNet pipeline完成とは扱わない。

ActivityNet integrationを確認してから、
clip representation / aggregationのStageへ進む。

## 14. Spec Gate

本draftでは、Stage 2の目的、dataset fixture、loader revision、
input/output shape、padding、frame order、metadata、freeze、
勾配更新なし、ActivityNetとの責務分離、Success Criteriaを定義した。

現在statusは `draft`。
ユーザーが内容を明示承認した時点で `approved` へ変更し、
`engineering-task` へ引き継ぐ。

# Implementation Handoff（承認後に有効）

- approved spec: 本spec
- 実装目的: 50Salads SequentialSampleをfrozen ViT frame feature列へ変換するStage 2 bridge smoke
- 基準repository/commit: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@e7e037a9191f36b26e87f80f48caccfb35b6166d`
- work branch: `feature-50salads-loder`
- external loader: `tamaki-lab/sequential_loader@cef09aa12560127451a5f569d86d5d51671e6986`
- 変更scope: 50Salads integration smoke + 直接必要な最小helper/test
- 対象外: ActivityNet / aggregation / LoRA / MoCo / training
- success criteria: 10章
- 許可される短時間検証: import / 1 chunk integration smoke / padding短時間check
- 長時間runの許可状態: 未許可
