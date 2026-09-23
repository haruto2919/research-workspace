---
date: 2026-09-23
project: sequential-video-lora-analysis
source_todo: null
topic: Stage 2 sequential_loaderとViTFrameEncoder接続
status: exploratory
tags: [brainstorm, research, stage2, sequential-loader, vit, video]
---

# Stage 2 sequential_loaderとViTFrameEncoder接続 壁打ち

## 出発点

Stage 1では `google/vit-base-patch16-224` を用いた `ViTFrameEncoder` を実装し、
real RGB image -> AutoImageProcessor -> frozen ViTModel -> CLS feature `[1,768]`
の経路を実環境で確認した。

Stage 2ではexternal `tamaki-lab/sequential_loader` から受け取った1 video chunkを、
Stage 1の `ViTFrameEncoder` へ接続し、frame feature列を得ることを目的候補とする。

## 確認済み文脈

### research code

- repository: `tamaki-lab/2026_09_ishikawa_sequential-video-lora`
- current main: `e1c1715135d5f43fbaf700bbc3533ada1b367a59`
- Stage 1 branch: `feature-vit-frame-encoder@e7e037a9191f36b26e87f80f48caccfb35b6166d`
- Stage 1実装はGitHub上ではまだmainへ未反映。

### sequential_loader

- repository: `tamaki-lab/sequential_loader`
- reference candidate: `master@cef09aa12560127451a5f569d86d5d51671e6986`
- public consumer contractは `SequentialSample`。
- `frames`: CPU `torch.uint8 [T,3,H,W]`, RGB。
- `frame_indices`: `int64 [T]`。
- `timestamps`: `float64 [T]`。
- `valid_mask`: `bool [T]`。
- `sequence_id`, `sequence_index`, `is_first`, `is_last` 等を保持する。
- strict DataLoaderは `batch_size=1, shuffle=False, num_workers=0` で、sampleへbatch次元を追加しない。
- tail paddingはframes=zero, frame_indices=-1, timestamps=NaN, valid_mask=False。
- validなframe_indicesはstrictly increasing。
- RAW timestampは単調性を保証せず、timestampを根拠にframeをsortしてはいけない。
- `evaluation_reference` はcausal inputへ混ぜない。

## Stage 2の中心目的候補

```text
SequentialSample
  frames [T,3,H,W] uint8 RGB
  valid_mask [T]
  frame_indices [T]
  timestamps [T]
        |
        v
checkpoint-compatible preprocessing
        |
        v
ViTFrameEncoder
        |
        v
frame_features [T,768]
```

Stage 2で主張するのは、
「sequential_loaderが返した時系列chunkを、順序・padding情報を保持して
pretrained ViT feature列へ変換できる」
ところまで。

時間情報を学習した、clip表現を学習した、LoRAが更新された、とは主張しない。

## 確認・決定が必要な項目

### 1. Stage 2のbase revision

候補:
- Stage 1をmainへ反映してから、そのmain commitをStage 2 baseにする。
- Stage 1 feature branchからStage 2 branchを直接作る。

推奨:
Stage 1をmainへ反映後、そのcommitをStage 2 baseにする。
各Stageを独立した作業単位として追跡しやすい。

### 2. sequential_loader revision

推奨候補:
`tamaki-lab/sequential_loader@cef09aa12560127451a5f569d86d5d51671e6986`

branch名だけでなくcommitをspecに固定する。
consumerはpublic `sequential_loader` APIのみ使用し、`src/` 内部moduleへ依存しない。

### 3. smoke用dataset / split

候補:
- 50Saladsをengineering fixtureとして使う。
- 最終研究dataset選定とは分離する。

推奨候補:
`Salads50Adapter` + `train1`。
Stage 2の目的はloader/ViT接続なので、datasetの科学的選定は行わない。

未決:
- dataset root。
- exact sequence IDを固定するか、split先頭sequenceでよいか。

### 4. chunk size T

推奨候補:
`frames_per_chunk=16`

理由:
既存の研究文脈とloader quick-startに整合し、十分小さいengineering smokeになる。

ただしStage 2では16を科学的最適値とは扱わない。

### 5. padding frameのfeature化

選択肢A:
全T frameをViTへ通し、後からvalid_maskで無視する。

選択肢B:
valid frameだけprocessor/ViTへ通し、featureを元のT位置へscatterし、
invalid positionをzero featureにする。

推奨:
B。

理由:
- padding zero-imageから意味のないViT featureを作らない。
- 不要なcomputeを避ける。
- `frame_indices/timestamps/valid_mask` とfeatureのT alignmentを保てる。
- Stage 3のmasked aggregationへ自然につながる。

候補contract:

```text
frame_features [T,768]
invalid rows = 0
valid_mask [T] preserved
```

### 6. preprocessing boundary

推奨:
- Stage 1と同じcheckpoint対応 `AutoImageProcessor` を使う。
- processorは `ViTFrameEncoder` の外側。
- input `uint8 RGB [T,3,H,W]` からvalid framesをbatch処理して
  `float [N_valid,3,224,224]` を作る。
- 独自resize/normalizeを再実装しない。

### 7. device boundary

推奨:
- `SequentialSample` とmetadataはCPUのまま。
- preprocessingはCPU側。
- processor後の `pixel_values` のみencoder deviceへ移す。
- featureをCPUへ戻すかGPUに保持するかはStage 3以降の構成と合わせる必要があるため未決。
  Stage 2 smokeではshape/finite確認だけならどちらでもよい。

### 8. gradient / eval policy

Stage 2 smokeではbackbone frozenのため `eval()` + `torch.no_grad()` を使う。

ただし再利用module内部に `torch.no_grad()` をhard-codeしない。
Stage 4でLoRAを追加すると、base parameterはfreezeのままでもLoRAへgradientが必要になるため。

### 9. metadata preservation

最低限保持・確認する候補:
- `valid_mask`
- `frame_indices`
- `timestamps`
- `sequence_id`
- `sequence_index`
- `is_first`
- `is_last`

重要:
- timestamp RAW modeの値でframeをsortしない。
- source/chunk/frame-index orderをそのまま維持する。
- `evaluation_reference` をmodel inputへ渡さない。

### 10. adapter / internal dataclassを作るか

選択肢A:
Stage 2では `SequentialSample` public APIを直接consumerし、余分なadapterを作らない。

選択肢B:
`SequentialLoaderAdapter` / `SequentialBatch` を導入し、内部contractへ変換する。

現時点の推奨:
Aから開始。

理由:
`SequentialSample` 自体が既に明示的なpublic contractであり、
Stage 2で実在するinterface mismatchはまだ確認されていない。
必要性が発生した時点でadapterを導入する方が最小である。

ただし将来、複数loader、batch dimension追加、training pipeline用の独自contractが必要になれば
adapter導入を再検討する。

## Stage 2 success criteria候補

- pinned sequential_loader revisionからpublic APIを利用できる。
- 50Saladsの1 chunkを時系列順に取得できる。
- `frames` inputが `uint8 [T,3,H,W]`。
- valid frameをcheckpoint-compatible processorへ渡せる。
- valid frame featureが `[N_valid,768]`。
- T alignmentを復元した場合 `frame_features [T,768]`。
- featureがfinite。
- invalid/padding位置がmodel inputとして有効frame扱いされない。
- `valid_mask/frame_indices/timestamps/sequence metadata` を壊さない。
- Stage 1のViT backboneはfrozenのまま。
- Stage 2 smokeはeval + no_grad。
- timestampでframe reorderingをしない。
- `evaluation_reference` をmodelへ入力しない。
- clip aggregation / LoRA / MoCoを追加しない。

## 今は決めなくてよいもの

- clip featureのaggregation方法。
- masked meanのexact実装。
- LoRA target/rank/alpha/dropout。
- MoCo variant / queue / temperature / EMA。
- sequential vs shuffle experiment。
- temporal objective。
- final dataset。
- final evaluation metric。
- CLIP-ViT比較。

## 反例・注意点

- 全padding frameをViTへ通してもfinite featureは出る可能性があるため、
  finiteだけではpadding handlingの正しさを保証しない。
- RAW timestampは逆行し得るので、timestamp昇順sortはloader contractを壊す。
- Stage 2でmean poolingまで入れると、loader/ViT bridgeのfailureとaggregation failureを分離しづらくなる。
- reusable moduleに `no_grad` を埋め込むとStage 4のLoRA gradientを阻害する可能性がある。
- 50Saladsをsmoke fixtureに使うことと、最終研究datasetとして採用することを混同しない。

## 現時点の方向性

有力:
- Stage 1をmainへ反映してからStage 2 branchを作る。
- sequential_loaderは `master@cef09aa...` をpin。
- 50Salads + train1 + T=16をengineering smoke候補。
- `SequentialSample` public APIを直接consume。
- valid frameだけprocessor/ViTへ通す。
- outputはT alignmentを保った `[T,768]` + original `valid_mask` 候補。
- metadataを保持し、timestamp sortはしない。
- no_gradはsmoke側だけ。
- aggregation / LoRA / MoCoはStage 2対象外。

## 次にユーザーが決める項目

1. Stage 1をmainへ反映してからStage 2を開始するか。
2. Stage 2 smokeを50Salads / train1 / T=16で固定するか。
3. paddingは「valid frameのみencode -> invalid feature rowは0」で進めるか。
4. Stage 2では `SequentialSample` を直接consumeし、adapter/dataclass追加を保留するか。
5. outputを `frame_features [T,768]` + `valid_mask` で固定するか。

これらが固まれば、research-specへ渡せる材料になる。


## 2026-09-23 10:47 追記: ActivityNetを最終datasetとする場合のStage 2確認順

ユーザーは最終的な動画datasetとしてActivityNetを利用する予定。

GitHub上の `tamaki-lab/sequential_loader@master` を確認したところ、
現時点のpublic example / adapterは50Salads向けであり、
repository code searchではActivityNet対応は確認できなかった。

ここで重要なのは、現在のloader設計では
「50Salads専用SequentialDataset」が中心なのではなく、
generic `SequentialDataset` にdataset固有Adapterが `SequenceSource` を供給する構造であること。

したがって推奨する切り分けは次。

```text
Step A: 既存50Salads adapterでgeneric bridgeを検証
Salads50Adapter
 -> SequentialDataset
 -> SequentialSample
 -> preprocessing
 -> ViTFrameEncoder
 -> [T,768]

Step B: ActivityNet adapterをsequential_loader側へ追加
ActivityNet
 -> ActivityNetAdapter候補
 -> same SequentialDataset
 -> same SequentialSample contract

Step C: ActivityNetで同じbridgeを再検証
ActivityNet SequentialSample
 -> same preprocessing / ViTFrameEncoder path
 -> [T,768]
```

### 推奨理由

50Saladsを先に使う目的は科学的dataset選定ではなく、
既に動くAdapterを使って
「SequentialSample consumer -> preprocessing -> ViTFrameEncoder」
だけを独立に検証するため。

ActivityNet adapterを先に作ると、
- ActivityNet file/split/source mapping
- dataset adapter
- chunk/read semantics
- loader -> ViT bridge
を同時に新規実装することになり、失敗時に原因を切り分けにくい。

50Salads bridgeが先に通っていれば、ActivityNet導入後のfailureは
ActivityNet adapter / data mapping側へかなり絞れる。

### 注意

50Salads smoke成功だけでActivityNet pipeline完成とはしない。
最終研究datasetがActivityNetなら、Stage 3 / LoRA / MoCoへ進む前に
ActivityNetでも同一public contractとfeature extraction smokeを通す。

ActivityNet側では少なくとも次を別途決める必要がある候補:
- dataset root / video file mapping
- train/val split mapping
- 1 sequenceを何として表すか
- start/stop frame policy
- sampling / chunking policy
- annotationをcausal inputから分離し、evaluation_referenceへどう持つか
- missing/corrupt videoの扱い

### 現時点の収束

有力:
1. 50Saladsをengineering fixtureとしてStage 2 generic bridge smokeに使う。
2. bridgeのinterfaceが固定できたらActivityNet adapterを別作業単位で作る。
3. ActivityNetで同じStage 2 smokeを通す。
4. その後Stage 3のclip representationへ進む。

50Saladsを最終datasetとして採用する判断ではない。
