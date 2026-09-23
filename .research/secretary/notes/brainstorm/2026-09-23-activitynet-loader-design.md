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


## 2026-09-23 17:20 追記: ActivityNet Adapter spec前チェックリスト

ActivityNet Adapterをspec化する前に、
「研究上の意味が変わるためユーザーが決める項目」と
「実装前にデータ実体を確認すればよい項目」を分離した。

### A. spec前に決めるべきblocking項目

#### A1. dataset scope

研究で使うActivityNetの集合を明示する。

候補:
- full ActivityNet v1.3
- v1.3 additional videosのみ
- 手元でdownload済みのavailable subset

有力:
full v1.3を論理datasetとし、実際に利用可能なlocal videoとのintersectionをmanifest化する。

理由:
directory名ではなくannotation JSONのvideo IDをSSOTにでき、
release別保存場所やmanual crawlの重複を吸収できる。

#### A2. annotation JSONの正本

ActivityNet Adapterがsplit / metadata判定に使うJSONを1つ固定する。

候補:
- full `activity_net.v1-3.min.json`

有力:
v1.3のfull annotation JSONを正本とする。

partial / truncated JSONはsmoke fixtureには使えても研究runのSSOTにはしない。

#### A3. training / validation / testing policy

有力:
- self-supervised training: `training`
- held-out check / evaluation: `validation`
- `testing`: trainingへ混ぜない

physical `train_val` directory名ではなくJSONの `subset` を使う。

#### A4. local video discovery rootsと重複優先規則

手元では少なくとも
- `manual_crawling_from_youtube/video`
- `v1-3/train_val`
- `v1-3/test`
に動画が存在し得る。

同じnormalized video IDが複数rootに存在する場合のpolicyが必要。

候補:
1. duplicateは常にerror。
2. root priorityを固定して1つを選ぶ。
3. binary content / decode可能性を比較して選ぶ。

有力:
まず全rootをscanし、同一IDのduplicateを報告する。
specでは優先順位を明示的に固定するか、duplicateが存在しないことを確認してから進む。
黙って先に見つかったfileを採用しない。

#### A5. missing video policy

annotation JSONにはあるがlocal videoが存在しないケースの扱い。

有力:
available intersectionを利用可能datasetとして明示し、
使用video ID manifestを固定する。
missing count / IDsを必ず記録する。

比較実験では同じmanifestを共有する。

#### A6. 1 sequenceの定義

有力:
1 physical video = 1 sequence。

```text
start_frame = 0
stop_frame = None
```

annotation segmentでsequenceを分割しない。

#### A7. annotationの責務

有力:
Adapterはannotation JSONをsplit判定 / metadata参照には使うが、
segment / labelをtraining inputやframe targetへ変換しない。

annotation情報は `evaluation_reference` / `dataset_metadata` に保持し、
厳密なtime-to-frame alignmentは評価specへ分離する。

### B. 実装前に確認すべきデータ実体

#### B1. full JSONがvalid JSONとしてparseできるか

確認:
- top-level keys
- `database`
- video entryの `subset`, `duration`, `annotations`
- training / validation / testing件数

#### B2. filename -> video ID normalization

確認:
- leading `v_` のみを除去するルールでJSON keyへ一致する割合
- underscoreやhyphenを含むIDのedge case
- extension違いでも同じIDとして扱えるか

#### B3. extension集合

添付では `.mp4` / `.mkv` を確認済み。

full datasetで他extensionが存在するかをscanし、
対応extensionをspecへ固定する。

#### B4. rootごとのfile count / normalized ID count

各rootについて:
- file数
- normalized unique ID数
- JSON keyとのintersection
- JSONにないlocal files
- JSONにあるがlocalにないfiles
- duplicate IDs

を確認する。

#### B5. duplicate実態

`manual_crawling_from_youtube/video` と `v1-3/*` の間で
同一video IDが重複しているかを実データで確認する。

重複がある場合:
- path
- extension
- file size
- decode可能性
を確認した上でpriority policyを決める。

#### B6. subset-directory整合

JSON `subset=training/validation` のIDが
physical `train_val` またはmanual crawlのどこに存在するか、
`subset=testing` がtest等に存在するかを確認する。

directory名をsubsetのSSOTにしない。

#### B7. decode smoke

各subsetから少数videoを選び、
`SequentialVideoReader` で
- open可能
- 先頭16 contiguous frames decode可能
- output dtype / shape
- EOF / padding
を確認する。

mp4とmkvを少なくとも1本ずつ含める。

#### B8. duration / FPSのばらつき

Adapter実装そのものには不要だが、
training sampling設計のため確認する。

- duration分布
- FPS分布
- resolution分布

特にActivityNetは動画ごとにFPSが異なり得るため、
16 contiguous framesの実時間幅が一定でない点を後続specで扱う。

### C. Adapterの責務として固定したい候補

```text
ActivityNetAdapter
  input:
    dataset root / discovery roots
    annotation JSON

  responsibility:
    local video discovery
    video ID normalization
    annotation metadata lookup
    subset filtering
    duplicate / missing reporting
    deterministic SequenceSource enumeration

  output:
    SequenceSource(
      sequence_id=video_id,
      source_id=video_id,
      source=video_path,
      start_frame=0,
      stop_frame=None,
      source_metadata=...,
      dataset_metadata=...,
      evaluation_reference=...
    )
```

Adapterに入れない候補:
- frame stride
- temporal subsampling
- chunk size
- shuffle
- batch size
- ViT preprocessing
- LoRA / MoCo
- annotation segment crop
- frame-level target生成

### D. Adapter smokeのsuccess候補

Adapter単体:
- full annotation JSONをparseできる。
- training / validation / testingをJSON subsetで列挙できる。
- source orderがdeterministic。
- normalized IDがannotation keyへ対応する。
- duplicate / missingをsilentに無視しない。
- mp4 / mkvを扱える。

Loader統合:
- ActivityNet sourceから `SequentialSample` が得られる。
- `frames [16,3,H,W]`, `valid_mask [16]`。
- frame orderを壊さない。
- tail paddingを既存core contractで扱える。

研究repo統合:
- Stage 2と同じbridgeを再利用し、
  `SequentialSample -> ViTFrameEncoder -> [16,768]`
  が成立する。

### E. 現時点でspec前に最優先で確認・決定する順序

1. full v1.3 annotation JSONを正本にするか。
2. local video discovery rootsを列挙する。
3. root横断のnormalized video ID inventoryを作る。
4. duplicate IDの実態を確認し、priority policyを決める。
5. missing policyと使用manifest方針を決める。
6. training / validation / testing policyを固定する。
7. 1 video = 1 sequence / whole-video policyを固定する。
8. その後ActivityNet Adapter specへ昇格する。

sampling / strideはAdapter specとは分離し、
ActivityNetでの動画自己教師あり学習spec前に決める。


## 2026-09-23 17:30 追記: ActivityNet inventory確認方法

Adapter spec前のデータ監査は、full ActivityNet rootを走査し、
annotation JSONのdatabase keyとlocal video IDを集合比較する方針とする。

確認対象root候補:
- `manual_crawling_from_youtube/video`
- `v1-3/train_val`
- `v1-3/test`

normalized video ID:
- file stemのleading `v_` のみ除去
- extensionはIDに含めない

集計する値:
- rootごとのvideo file数
- rootごとのnormalized unique ID数
- rootごとのJSON key一致ID数
- JSONにはあるがlocal unionにないmissing ID数
- localにはあるがJSONにないunmatched ID数
- root横断duplicate ID数
- extension別file数
- JSON subset（training / validation / testing）ごとのmetadata総数、local available数、missing数

重要:
- duplicateはpath数2以上のnormalized IDとして検出し、path一覧を出す。
- missingは各root単体ではなく、まず全local rootのunionに対して判定する。
- subsetはdirectory名ではなくannotation JSONの `subset` をSSOTにする。
- full dataset auditではvalidなfull `activity_net.v1-3.min.json` を使う。
  添付ZIP内のJSONは抜粋で途中までのためfull auditの正本には使わない。

このinventory結果をEvidenceにして、
local root priority、missing policy、利用manifest、ActivityNet Adapterのsplit contractをspec化する。


## 2026-09-23 17:40 追記: full ActivityNet inventory実測結果

ユーザーが研究サーバ上のfull datasetに対してinventory scriptを実行した。

### JSON

- annotation: `/mnt/NAS-TVS872XT/dataset/ActivityNet/json/activity_net.v1-3.min.json`
- valid JSONとしてparse可能
- JSON video IDs: 19,994

subset:
- training: 10,024
- validation: 4,926
- testing: 5,044

### local roots

#### manual_crawling_from_youtube/video

- video files: 18,226
- normalized unique IDs: 18,226
- JSON matched IDs: 18,226
- local-only: 0
- extension: mp4 18,226

#### v1-3/train_val

- video files: 14,950
- normalized unique IDs: 14,950
- JSON matched IDs: 14,950
- local-only: 0
- extensions:
  - mp4: 13,545
  - mkv: 1,386
  - webm: 19

#### v1-3/test

- video files: 5,044
- normalized unique IDs: 5,044
- JSON matched IDs: 5,044
- local-only: 0
- extensions:
  - mp4: 4,637
  - mkv: 403
  - webm: 4

### union

- local unique IDs: 19,994
- JSON matched IDs: 19,994
- missing IDs: 0
- local-only IDs: 0
- duplicate IDs: 18,226

subset別local available:
- training: 10,024 / 10,024
- validation: 4,926 / 4,926
- testing: 5,044 / 5,044
- missing: all 0

### 重要な解釈更新

`v1-3/train_val` と `v1-3/test` の合計は

```text
14,950 + 5,044 = 19,994
```

で、annotation JSONのfull ActivityNet v1.3全ID数と一致する。

また、

```text
training + validation
= 10,024 + 4,926
= 14,950
```

で `v1-3/train_val` のunique ID数と一致し、
`testing = 5,044` は `v1-3/test` と一致する。

したがって、この研究サーバ上のlocal layoutでは
`v1-3/train_val` と `v1-3/test` がfull ActivityNet v1.3を完全に保持している
と考えるのが最も整合的。

以前の「v1-3 directoryはadditional videosだけ」という解釈は、
official tarballの命名説明には当てはまっても、
現在の研究サーバ上の展開済みdirectoryの実体には当てはまらない。
Adapter設計では実測されたlocal layoutを優先する。

### manual_crawling rootの位置付け

18,226 IDがすべてfull v1.3集合内にあり、
global duplicate IDsも18,226であることから、
`manual_crawling_from_youtube/video` は
`v1-3/train_val` / `v1-3/test` に存在する動画のsubset copyと考えられる。

このためAdapterの通常探索rootとして両方を同時利用すると
18,226件のduplicateを必ず発生させる。

現時点の有力方針:
- primary local rootsは
  - `v1-3/train_val`
  - `v1-3/test`
- `manual_crawling_from_youtube/video` は通常探索対象から除外
- fallbackとして使う必要は、primary rootsが0 missingなので現時点ではない

これによりduplicate priority policy自体をAdapterへ持ち込まずに済む。

### 次に確認したい最小事項

spec化前の残確認は主に次。

1. JSON subsetとphysical directoryの整合を集合演算で確認する。
   - training ∪ validation == v1-3/train_val IDs
   - testing == v1-3/test IDs
   - train_valとtestのintersection == 0
2. mp4 / mkv / webm各形式を少数decode smokeする。
3. 1 video = 1 sequence / whole-video方針を正式に採用する。
4. trainingのみをself-supervised learningに使うかを決定する。

missing policyはfull primary rootsでmissing=0のため、
今回のAdapter baselineでは複雑なfallbackを持たせない方向が有力。
