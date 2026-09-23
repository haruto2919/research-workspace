---
project: sequential-video-lora-analysis
spec_type: implementation
status: implemented
title: ActivityNet v1.3 Adapter for sequential_loader
created: 2026-09-23
last_updated: 2026-09-23
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential_loader
implementation_base_branch: master
implementation_base_commit: cef09aa12560127451a5f569d86d5d51671e6986
implementation_work_branch: ActivityNet
implementation_work_branch_base_commit: cef09aa12560127451a5f569d86d5d51671e6986
dataset: ActivityNet v1.3
annotation_json: json/activity_net.v1-3.min.json
---

# ActivityNet v1.3 Adapter for sequential_loader spec

> **Status: implemented**
>
> 本specは、既存 `sequential_loader` Coreを変更せず、
> 50Saladsと同じ責務分離でActivityNet v1.3を `SequenceSource` へ変換する
> dataset-specific Adapterを追加するための実装契約である。
>
> 実装先は既存branch `ActivityNet` とする。
> 2026-09-23にユーザーが本specの承認とコード実装を明示的に依頼した。

## 1. 目的

ActivityNet v1.3を既存Sequential Loaderへ追加し、Consumerがdataset固有構造を意識せず、

```text
ActivityNet
 -> ActivityNetAdapter
 -> SequenceSource
 -> existing SequentialDataset
 -> existing SequentialVideoReader
 -> existing SequentialSample
```

という既存contractを利用できるようにする。

今回の変更では、50Saladsで成立している

```text
Dataset Adapter -> SequenceSource -> Core
```

という責務境界を維持する。

## 2. Authority / Evidence

### 2.1 現在のユーザー指示

- ベースのSequential Loader Coreは変更しない。
- 50Saladsと同様の設計方針でActivityNet用loaderを作る。
- strictなdataset不整合処理を採用する。
- 実装先は既存 `ActivityNet` branchとする。

### 2.2 implementation repository

```text
repository:
tamaki-lab/2026_09_ishikawa_sequential_loader

base:
master@cef09aa12560127451a5f569d86d5d51671e6986

work branch:
ActivityNet@cef09aa12560127451a5f569d86d5d51671e6986
```

spec作成時点で `ActivityNet` branchは存在し、`master` と同じcommitを指す。

### 2.3 既存設計

現行 `Salads50Adapter` はdataset固有の

- file discovery
- video ID normalization
- split selection
- metadata / evaluation reference
- deterministic `SequenceSource` enumeration

だけを担当し、chunking / decode / DataLoader / trainingは担当しない。

ActivityNetでもこの責務分離を維持する。

### 2.4 ActivityNet full-data inventory

研究サーバ上のfull ActivityNet v1.3について、ユーザー実行のinventoryで以下を確認した。

```text
annotation JSON IDs: 19,994

training:   10,024
validation:  4,926
testing:     5,044

v1-3/train_val local IDs: 14,950
v1-3/test local IDs:       5,044

training ∪ validation == v1-3/train_val: True
testing == v1-3/test:                     True
train_val ∩ test:                          0

missing: 0
extra:   0
```

これらのlocal実行結果はユーザー報告Evidenceであり、GitHub connectorから研究サーバ実体を
直接確認したものではない。

### 2.5 container decode Evidence

ユーザー実行の短時間smokeで現行 `SequentialVideoReader` が次をdecodeできた。

- `.mp4`
- `.mkv`
- `.webm`

各形式で先頭16 frameを `torch.uint8 [16,3,H,W]` として取得し、
absolute frame index `0..15` を確認した。

したがってActivityNet Adapterはcontainer/codec別decode logicを持たず、
既存ReaderへPathを渡すだけとする。

## 3. Decision Contract

### 3.1 Dataset scope

対象はfull ActivityNet v1.3とする。

annotation metadataの正本:

```text
<dataset_root>/json/activity_net.v1-3.min.json
```

Adapterはpartial / truncated annotation JSONを研究用full datasetとして扱わない。

### 3.2 Local video roots

通常探索対象は次の2箇所だけとする。

```text
<dataset_root>/v1-3/train_val
<dataset_root>/v1-3/test
```

`manual_crawling_from_youtube/video` は探索対象へ含めない。

理由:
研究サーバ上ではprimary rootsだけでfull 19,994 IDsが揃い、
manual crawling側を加えると多数のduplicate IDが発生することが実測されているため。

fallback rootは実装しない。

### 3.3 Split semantics

外部APIのsplit名はannotation JSONの `subset` と一致させる。

```text
training
validation
testing
```

物理directory `train_val` をsplit名としてpublic APIへ露出しない。

想定API:

```python
adapter = sl.ActivityNetAdapter(dataset_root=Path("/path/to/ActivityNet"))

training_sources = adapter.sequence_sources("training")
validation_sources = adapter.sequence_sources("validation")
testing_sources = adapter.sequence_sources("testing")
```

`available_splits()` を提供する場合は上記3 subsetを決定的順序で返す。

### 3.4 Training subset policyとの責務分離

研究baselineでは

```text
training   -> self-supervised learning
validation -> held-out evaluation / analysis
testing    -> trainingには使用しない
```

を採用方向とする。

ただし、この選択はAdapterへhard-codeしない。

Adapterは3 subsetすべてを同じpublic contractで提供し、
どのsubsetをtrainingに使うかは後続training spec / consumerが決める。

### 3.5 Video ID normalization

local filename stemのleading `v_` だけを取り除く。

例:

```text
v_-1EC1ZP6aC4.mp4 -> -1EC1ZP6aC4
v__-4ngMPCA9A.mp4 -> _-4ngMPCA9A
```

leading `v_` 以外のunderscore / hyphenは変更しない。

normalized IDをannotation JSONの `database` keyとして扱う。

### 3.6 Supported extensions

対象video extension:

```text
.mp4
.mkv
.webm
```

extension判定はlowercase化して行ってよい。

それ以外のfileはvideo sourceとして列挙しない。

### 3.7 Sequence definition

```text
1 physical video = 1 sequence
```

各 `SequenceSource` は、

```text
sequence_id = normalized video ID
source_id   = normalized video ID
start_frame = 0
stop_frame  = None
```

とし、動画全体をEOFまでのunknown-length sequenceとして渡す。

ActivityNet annotationのaction segmentではsequenceを分割しない。

### 3.8 Annotation handling

Adapterはannotation JSONを

- subset判定
- video metadata参照
- evaluation reference

のために読む。

action label / temporal segmentを

- training target
- frame-aligned target
- sequence crop
- chunk boundary

へ変換しない。

ActivityNet用のopaque参照として
`ActivityNetAnnotationReference` を追加する。

最低限保持する候補:

```text
annotation_path
video_id
subset
frame_alignment_verified = False
```

厳密な秒→frame alignmentは本specの対象外。

### 3.9 SequenceSource metadata

最低限、次を保持する。

`source_metadata`:

```text
video_path
video_stem
video_extension
```

`dataset_metadata`:

```text
dataset = "activitynet"
version = "1.3"
split
video_id
annotation_path
annotation_frame_alignment = "unverified"
```

annotation内の全label / segmentをmetadataへ複製しない。

### 3.10 Deterministic ordering

`sequence_sources(split)` はnormalized video ID順で決定的に返す。

filesystem enumeration順へ依存しない。

Adapterはshuffleしない。

### 3.11 Strict integrity policy

ActivityNet Adapterはfull v1.3 layoutをstrictに検証する。

対象となるsupported video filesについて、次をsilentに補正しない。

- annotation JSONにあるexpected video IDのlocal fileがない
- supported local video fileのnormalized IDがannotation JSONにない
- 同じnormalized IDに複数のsupported local video fileが対応する
- JSON subsetとphysical rootの対応が矛盾する

これらは明示的なerrorとする。

正常な研究サーバlayoutではmissing / extra / duplicateが0であることが既に確認されているため、
fallbackやavailable-intersection modeは本specでは実装しない。

### 3.12 Physical root / subset consistency

strict baselineでは次を期待する。

```text
training ∪ validation IDs == v1-3/train_val IDs
testing IDs              == v1-3/test IDs
train_val IDs ∩ test IDs == empty
```

この不変条件が崩れる場合はerrorとし、
directory名だけを信頼してsubsetを書き換えない。

## 4. Public API

外部consumerは内部 `src.*` ではなくtop-level packageだけを利用する。

追加するpublic names:

```text
ActivityNetAdapter
ActivityNetAnnotationReference
```

想定:

```python
from pathlib import Path
import sequential_loader as sl

adapter = sl.ActivityNetAdapter(
    dataset_root=Path("/path/to/ActivityNet")
)

sources = adapter.sequence_sources("training")
```

既存public APIの名前・意味を変更しない。

## 5. Implementation Scope

想定する最小変更:

```text
src/adapters/activitynet.py
sequential_loader/__init__.py
tests/test_activitynet_adapter.py
tests/test_public_api.py
```

必要なら、50Saladsの利用例と同程度の最小consumer exampleとして

```text
examples/read_activitynet.py
```

を追加してよい。

`src/adapters/__init__.py` は現在のstyle上必要な場合だけ最小変更する。

## 6. Must Not Change

今回のActivityNet対応のために、次を変更しない。

```text
src/sequential/source.py
src/sequential/dataset.py
src/sequential/reader.py
src/sequential/sample.py
src/sequential/chunking.py
src/sequential/dataloader.py
src/config.py
```

既存contract:

- `SequenceSource`
- `SequentialDataset`
- `SequentialVideoReader`
- `SequentialSample`
- tail padding
- RAW timestamp semantics
- strict DataLoader semantics
- resource lifecycle

を維持する。

ActivityNet対応のためにCoreへdataset-specific分岐を追加しない。

## 7. Explicit Out of Scope

本specでは次を実装しない。

- ViT preprocessing / feature extraction
- ActivityNet -> ViT integration
- LoRA
- MoCo / BYOL / MAE等のSSL objective
- optimizer / backward / training loop
- frame stride
- temporal subsampling
- seconds-based sampling
- clip duration policy
- frames_per_chunkの研究条件決定
- video間shuffle / sequential比較
- per-video chunk budget
- annotation segment crop
- label parsingを用いたtraining target生成
- segment秒→frame index alignment
- downstream evaluation
- manual_crawling fallback
- missing videoを黙ってskipするrelaxed mode

これらは必要に応じて後続specで扱う。

## 8. Test / Verification Plan

### 8.1 Unit tests

temp directory + synthetic JSON / dummy file layoutで少なくとも次を検証する。

1. `training / validation / testing` をJSON subsetで列挙できる。
2. normalized video ID順で決定的に返す。
3. leading `v_` だけを除去する。
4. `.mp4/.mkv/.webm` をsource候補として扱う。
5. unrelated suffixをvideoとして扱わない。
6. `SequenceSource` がwhole-video contract
   `start_frame=0, stop_frame=None` を持つ。
7. source / dataset metadataが期待値を持つ。
8. `ActivityNetAnnotationReference` がopaque referenceとして付与される。
9. annotation referenceをmodel targetへ変換しない。
10. invalid splitを拒否する。
11. annotation JSON欠損 / malformed structureを拒否する。
12. expected local video missingを拒否する。
13. JSONにないlocal video IDを拒否する。
14. duplicate normalized IDを拒否する。
15. subsetとphysical rootの矛盾を拒否する。
16. existing `Salads50Adapter` testが回帰しない。
17. top-level public APIから新Adapter / referenceをimportできる。

### 8.2 Short real-data smoke

実装後、研究サーバ上の

```text
/mnt/NAS-TVS872XT/dataset/ActivityNet
```

を用いた短時間smokeを許可する。

最低限:

```text
len(sequence_sources("training"))   == 10024
len(sequence_sources("validation")) == 4926
len(sequence_sources("testing"))    == 5044
```

加えて任意sourceについて、

```text
sequence_id == normalized video ID
start_frame == 0
stop_frame is None
```

を確認する。

Reader統合確認では、既存Reader / Datasetを変更せず、
`.mp4/.mkv/.webm` 各1本から先頭chunkを取得できることを確認してよい。

長時間の全動画decode、ViT forward、trainingは本specの短時間検証に含めない。

## 9. Success Criteria

本specの実装成功は、次を全て満たすこととする。

1. 実装先branchが `ActivityNet`。
2. branch baseが `cef09aa12560127451a5f569d86d5d51671e6986`。
3. existing Sequential Loader Coreを変更していない。
4. `ActivityNetAdapter` をtop-level public APIから利用できる。
5. `ActivityNetAnnotationReference` をtop-level public APIから利用できる。
6. full `activity_net.v1-3.min.json` をsubset SSOTとして扱う。
7. primary rootsは `v1-3/train_val`, `v1-3/test` のみ。
8. `manual_crawling_from_youtube/video` を探索しない。
9. `training / validation / testing` をJSON subsetで選択できる。
10. normalized IDはleading `v_` だけを除去する。
11. `.mp4/.mkv/.webm` を扱える。
12. 1 video = 1 whole-video `SequenceSource`。
13. sequence source orderがdeterministic。
14. annotation segmentをsequence分割 / targetへ利用しない。
15. missing / extra / duplicate / root-subset mismatchをstrict errorにする。
16. 実データでsource countが
    `10024 / 4926 / 5044` と一致する。
17. 既存 `Salads50Adapter` とpublic Core contractが回帰しない。
18. ActivityNet固有logicがAdapter層からCoreへ漏れていない。

## 10. Failure / Stop Conditions

次の場合、条件を黙って変更せず停止・報告する。

- `ActivityNet` branchのbaseがspecの基準revisionから意図せず変わっている。
- full annotation JSONのschemaが想定と異なる。
- full dataset inventoryが既確認の `19994 / 10024 / 4926 / 5044` と一致しない。
- primary rootsにmissing / extra / duplicateが生じる。
- `.mp4/.mkv/.webm` の利用にCore変更が必要になる。
- ActivityNet対応のため既存 `SequenceSource` contract変更が必要になる。
- annotation segmentをinput contractへ入れないとAdapterが成立しない。
- 既存50Salads/public API testが回帰する。

この場合、fallback root、silent skip、Core改変等へ勝手に切り替えない。

## 11. Ambiguity Gate

### 11.1 Blocking

本specの実装を妨げるblocking ambiguityはない。

採用済み:

- implementation repository:
  `tamaki-lab/2026_09_ishikawa_sequential_loader`
- work branch: `ActivityNet`
- base commit: `cef09aa12560127451a5f569d86d5d51671e6986`
- Coreは変更しない
- 50Saladsと同じAdapter responsibility
- full ActivityNet v1.3
- JSON subsetをsplit SSOTにする
- primary rootsは `v1-3/train_val`, `v1-3/test`
- `manual_crawling` は使わない
- `.mp4/.mkv/.webm`
- leading `v_` normalization
- 1 video = 1 sequence
- whole-video `start_frame=0, stop_frame=None`
- annotation segmentをsequence分割に使わない
- strict missing / extra / duplicate / root mismatch error
- sampling / ViT / LoRA / SSL trainingは対象外

### 11.2 Non-blocking

既存styleに従って決めてよい。

- helper function名
- private index/cacheの内部構造
- error messageの細かな文言
- JSON load helperの分割方法
- metadata Mappingの順序
- example fileを追加するかどうか

ただしpublic behavior、strictness、責務境界を変更してはならない。

## 12. Compatibility

本変更はadditiveとする。

- 既存50Salads APIを維持。
- existing Core classes / config / Reader semanticsを維持。
- package version変更は本specでは必須としない。
- research repository側の既存Stage 2 bridgeを変更しない。

ActivityNet対応は新Adapterを追加するだけであり、
既存Consumerが50Saladsを読む経路にbreaking changeを入れない。

## 13. Reproducibility

実装基準:

```text
tamaki-lab/2026_09_ishikawa_sequential_loader
ActivityNet branch
base cef09aa12560127451a5f569d86d5d51671e6986
```

dataset validation Evidence:

```text
dataset_root:
研究サーバ /mnt/NAS-TVS872XT/dataset/ActivityNet

annotation:
json/activity_net.v1-3.min.json

expected:
training=10024
validation=4926
testing=5044
total=19994
missing=0
extra=0
```

absolute dataset pathはAdapter実装へhard-codeせず、runtimeの `dataset_root` として渡す。

## 14. Spec Gate

本specは現在 `implemented`。

ユーザーはActivityNet Adapterの方向性、
strict error policy、
既存 `ActivityNet` branchを使うことを採用した。

2026-09-23にユーザーが「このスペックをapprovedに変更し，実装してください」と
明示的に依頼したため、承認済みとして実装と8章の短時間検証を実施した。

# Implementation Handoff（承認済み）

- approved spec: 本spec
- implementation repository: `tamaki-lab/2026_09_ishikawa_sequential_loader`
- work branch: `ActivityNet`
- base commit: `cef09aa12560127451a5f569d86d5d51671e6986`
- implementation purpose: ActivityNet v1.3 dataset-specific Adapter追加
- change scope: ActivityNet Adapter / annotation reference / public export / tests / optional minimal example
- must preserve: Sequential Loader Core / Salads50Adapter / public existing contract
- success criteria: 9章
- allowed short verification: unit tests / source-count inventory / 3-format one-chunk smoke
- long run permission: なし
- excluded: ViT / LoRA / MoCo / training / sampling policy

## 15. Implementation / Verification

2026-09-23: 指定の`ActivityNet` branch / 基準commitから実装し、必須検証を完了した。
remote `ActivityNet` branchの実装commitは `19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`。
ActivityNet専用22 tests、全183 tests、実ActivityNet source count `10024 / 4926 / 5044`、
3形式のReader smokeが成功したため、spec statusを `implemented` とする。
同日、ユーザーの明示指示によりstatusを`implemented`へ更新した。検証結果と実装差分の識別情報は
[実装・短時間検証記録](../experiments/2026-09-23-activitynet-adapter-verification.md)を参照。
