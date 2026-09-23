---
date: 2026-09-23
project: sequential-video-lora-analysis
source_todo: null
topic: ActivityNet sequential loader設計
status: exploratory
tags: [brainstorm, research, activitynet, sequential-loader, dataset]
---

# ActivityNet sequential loader設計 壁打ち

## 出発点

Stage 2では50Saladsをengineering fixtureとして、

```text
50Salads
 -> SequentialSample
 -> AutoImageProcessor
 -> frozen ViTFrameEncoder
 -> frame_features [T,768]
```

のintegration smokeとpadding unit testを確認した。

次は最終dataset候補のActivityNetを、同じ `SequentialSample` public contractへ接続するための
dataset-specific Adapterを検討する。

今回、ユーザーがActivityNet dataset構造の抜粋ZIPを共有した。
`v1-3/test` と `v1-3/train_val` の動画本数は意図的に削減された抜粋である。

## 添付データから確認できたこと

### ディレクトリ構造

添付には少なくとも次がある。

```text
ActivityNet/
├── json/
│   ├── activity_net.v1-2.min.json
│   ├── activity_net.v1-2.min.json.formatted
│   ├── activity_net.v1-3.min.json
│   └── activity_net.v1-3.min.json.formatted
├── v1-3/
│   ├── test/
│   └── train_val/
├── official_tarball_baidu/
├── official_C3D_feature/
└── manual_crawling_from_youtube/
```

抜粋内の動画数:

- `v1-3/test`: 154 files
  - mp4: 137
  - mkv: 17
- `v1-3/train_val`: 64 files
  - mp4: 61
  - mkv: 3

ユーザーが本数を削減した抜粋なので、これらを本来のActivityNet本数とは扱わない。

### filenameとannotation ID

testの例:

```text
video file:
v_-1EC1ZP6aC4.mp4

JSON database key:
-1EC1ZP6aC4
```

添付のpartial JSONでは、このIDの `subset` が `testing` と確認できた。

したがってActivityNet Adapterでは、filename stem先頭の `v_` を取り除いたIDを
annotation database keyとして扱う方向が自然。

ただしtrain_val側のサンプルIDは、添付JSON自体が途中で切られているため
今回のZIPだけではmetadataとの対応を全件検証できない。

### annotation JSONの形

添付から、database entryには少なくとも次が存在することを確認できる。

```text
subset
duration
url
resolution
annotations:
  - label
  - segment [start_sec, end_sec]
```

annotationは秒単位segmentであり、50Saladsのframe labelとは性質が異なる。

### v1.3の構成上の注意

添付内 `official_tarball_baidu/tarball/README.md` では、

- `v1-3_train_val` はActivityNet 1.3で追加されたtrain/val動画
- `v1-3_test` はActivityNet 1.3で追加されたtest動画
- v1.3はv1.2のextensionなので、full v1.3にはv1.2データもmergeする

と説明されている。

したがって、

```text
ActivityNet/v1-3/train_val
ActivityNet/v1-3/test
```

だけを読むことと、full ActivityNet v1.3を使うことは同義ではない。

これはdataset compositionを変えるためspec前のblocking decision候補。

## 現行sequential_loaderから確認できたこと

使用するloader repository:

```text
tamaki-lab/2026_09_ishikawa_sequential_loader
master@cef09aa12560127451a5f569d86d5d51671e6986
```

現時点でActivityNet Adapterは存在しない。

### Adapterの責務

既存 `Salads50Adapter` は、

- dataset固有のfile探索
- video ID normalization
- splitごとのsource列挙
- metadata / evaluation reference付与

だけを担当する。

chunking / decode / batch / trainingは担当しない。

ActivityNetでもこの責務分離を維持するのが有力。

### SequenceSource

generic coreへ渡すcontract:

```text
sequence_id
source_id
source: Path
start_frame
stop_frame
source_metadata
dataset_metadata
evaluation_reference
```

ActivityNetでも「1 physical video = 1 SequenceSource」とし、

```text
start_frame = 0
stop_frame = None
```

でwhole videoをEOFまで読む案が最も既存設計に近い。

### chunk / sampling制約

現行generic coreの `FixedChunkConfig` は、

```text
frames_per_chunk
```

だけを持つ。

chunk plannerは連続frameを16枚ずつ読む設計であり、
frame stride / temporal subsamplingは現行public coreには確認できない。

したがって、

```text
16 contiguous frames
```

はActivityNet Adapter smokeにはそのまま利用できるが、
最終trainingでどの時間幅を1 clipとして扱うかは別途決める必要がある。

sampling policyはdataset Adapterの責務へ入れず、
Core / Reader / orchestration側の別設計として扱うのが有力。

## 中心となる設計候補

### 候補A: whole-video Adapter

```text
ActivityNet metadata JSON
        +
local video files
        |
        v
ActivityNetAdapter
        |
        v
1 video = 1 SequenceSource
start_frame=0
stop_frame=None
        |
        v
SequentialDataset
        |
        v
SequentialSample [T,C,H,W]
```

これを有力候補とする。

理由:

- 50Salads Adapterと責務が揃う。
- 長時間動画のsequential inputを保てる。
- annotation segmentでtraining inputをcropしないため、
  self-supervised inputにlabel由来境界を持ち込まない。
- 後でsame public contractを研究repo側から利用できる。

## splitの扱い

物理directoryは `train_val` だが、trainingとvalidationの区別はmetadata JSONの
`database[video_id]["subset"]` をSSOTにする候補。

有力:

```text
sequence_sources("training")
sequence_sources("validation")
sequence_sources("testing")
```

Adapterがannotation JSONをparseし、
physical `train_val` directoryからtraining / validationをmetadataでfilterする。

`train_val` directory名そのものを1 splitとしてtrainingへ使わない。

研究baselineでは、

- self-supervised training: `training` のみ
- evaluation / held-out check: `validation`
- `testing`: 原則trainingへ混ぜない

を有力候補とする。

## annotationの扱い

Stage 2相当のinput bridgeではannotationをmodel inputへ使わない。

候補:

```text
ActivityNetAnnotationReference
  annotation_path
  video_id
  subset
```

を `evaluation_reference` に置き、
label / segmentの解釈は評価側へ遅延する。

Adapterはsplit判定のためJSONを読むが、
annotation segmentをframe-aligned training targetへ変換しない。

理由:

- 現在の研究目的はself-supervised video adaptation。
- ActivityNetのsegmentは秒単位であり、
  decode frameとの厳密alignmentは別問題。
- dataset ingestionとdownstream evaluationを分離できる。

## file format / ID mapping

ActivityNet抜粋では `.mp4` と `.mkv` が混在する。

Adapterは単一suffixに固定せず、

```text
.mp4
.mkv
```

の両方を対象候補とする。

IDはfilename stemからleading `v_` のみ除去する。

例:

```text
v_-1EC1ZP6aC4.mp4
 -> -1EC1ZP6aC4
```

同一normalized IDに複数fileが対応する場合は黙って選ばずerror候補。

## missing / unavailable video

ActivityNetではmetadataとlocal downloadの完全一致を前提にしない可能性がある。
添付にもmissing/unavailable関連listが存在する。

候補方針:

- local fileが存在しannotation entryも存在するものを利用可能sourceとする。
- local fileがあるのにannotation entryがなくsplitを決められない場合はerror。
- annotation entryがあるがlocal videoがない場合は、黙って無視せずcount / IDを報告する。
- scientific runでは利用video ID manifestを固定し、
  sequential vs shuffle等の比較で同一集合を必ず使う。

「missing videoを1件でもfatalにする」か
「available intersectionを明示的に使う」かは最終spec前に決める必要がある。

## ActivityNet Adapter smoke候補

### Adapter単体

確認:

- full annotation JSONをparseできる。
- `training / validation / testing` を列挙できる。
- `train_val` physical directoryからmetadata subsetでtraining / validationを分離できる。
- mp4 / mkv双方を扱える。
- normalized IDとJSON database keyが一致する。
- source orderが決定的。
- duplicate normalized IDを拒否する。
- missing/unmatched件数を確認できる。

### Loader integration

最初の1 video / 1 chunkで、

```text
ActivityNetAdapter
 -> SequenceSource
 -> SequentialDataset
 -> SequentialSample
 -> frames [16,3,H,W]
```

を確認する。

### ViT integration

Stage 2で確立した同じbridgeへ渡し、

```text
SequentialSample
 -> AutoImageProcessor
 -> ViTFrameEncoder
 -> frame_features [16,768]
```

を確認する。

ここでも学習・LoRA・MoCoはまだ行わない。

## 今回の重要な未決事項

### Blocking候補

1. **full ActivityNet v1.3を使うか、v1-3追加分だけを使うか**
   - full v1.3ならv1.2 dataも必要。
   - dataset compositionそのものが変わる。

2. **self-supervised trainingに使うsubset**
   - trainingのみ。
   - training + validation。
   - testも含める。
   - 比較・評価設計へ影響するため明示する必要がある。

3. **missing video policy**
   - strict complete datasetを要求するか。
   - available intersectionを固定manifestとして使うか。

### Adapter smokeではnon-blockingだがtraining前にblocking

4. **temporal sampling policy**
   - 16 contiguous frames。
   - frame strideを入れる。
   - 秒ベースsampling。
   - ActivityNetはFPSが動画ごとに異なり得るため、
     contiguous 16 framesではclipの実時間幅が一定にならない。

5. **video間の順序**
   - Adapterではdeterministic ID sort候補。
   - 実際のonline trainingではvideo order自体もupdate trajectoryへ影響し得るため、
     training specでseed / orderを固定する必要がある。

## 現時点の有力方向

Adapter自体は次の最小責務にする。

```text
ActivityNet root + annotation JSON
  -> local video discovery
  -> filename ID normalization
  -> metadata subset mapping
  -> deterministic SequenceSource enumeration
```

そして、

- 1 video = 1 sequence
- whole video
- annotation segmentでinputをcropしない
- training / validation / testingはJSON subsetで分ける
- labels / segmentsはevaluation_reference側
- mp4 / mkv対応
- chunkingは既存Coreへ任せる
- ViT bridgeはStage 2実装を再利用

を有力候補とする。

sampling / strideはActivityNet Adapterへ入れない。

## 次にユーザーと決めること

最優先は次の3点。

1. 研究でいう「ActivityNet」はfull v1.3か、手元の `v1-3` additional videosだけか。
2. self-supervised学習ではtraining subsetだけを使うか。
3. full metadataに存在してもlocal videoが欠損している場合、available filesのみを固定manifest化して利用してよいか。

この3点が決まれば、ActivityNet Adapterのspec化に向けてかなり具体化できる。

temporal sampling policyはAdapter smokeとは分離し、
実際のvideo SSL training条件を決める段階で別途詰める。


## 2026-09-23 17:08 追記: manual_crawling_from_youtube/video のrelease解釈

添付ActivityNet構造と公式release情報を照合した結果、
`manual_crawling_from_youtube/video` を「v1.2動画だけの保存先」とみなすのは不適切。

確認できた根拠:

- `manual_crawling_from_youtube/video_list_all_id.txt` は19,994 IDを含む。
- ActivityNet v1.3公式構成は training 10,024 + validation 4,926 + testing 5,044 = 19,994 videos。
- 同directoryには
  - `missing_list.1.2.txt`
  - `missing_list.1.3.txt`
  の両方が存在する。
- `video_list_downloaded_id.txt` は18,226 ID、
  `video_list_unavailable_id.txt` は1,768 IDで、
  合計19,994となる。
- downloaded IDは `missing_list.1.2` と `missing_list.1.3` の双方に交差する。

したがって、このmanual crawling領域はrelease 1.2専用ではなく、
full ActivityNet v1.3の19,994 video ID集合をYouTubeから取得しようとした結果
（downloaded / unavailableを記録したもの）と解釈するのが最も整合的。

一方、release別の確実な区分は `official_tarball_baidu/tarball/README.md` にあり、

- `v1-2_train.tar.gz`, `v1-2_val.tar.gz`, `v1-2_test.tar.gz`
  = ActivityNet release 1.2 data
- `v1-3_train_val.tar.gz`, `v1-3_test.tar.gz`
  = release 1.3で追加されたvideos

と明記されている。

ActivityNet Adapterを設計するときはdirectory名だけでv1.2/v1.3を推定せず、
annotation JSONのvideo ID / subsetをSSOTとしてlocal filesと照合する方針が有力。
