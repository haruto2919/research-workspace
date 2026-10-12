---
project: sequential-video-lora-analysis
date: 2026-10-11
analysis_mode: explain
detail: full
level: junior
source_repository: /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
source_commit: 65dedb4ebc26c48d23606b9bc674044f64e64c06
source_branch: dev
source_dirty: false
targets:
  - all 135 Python files
related_analysis: code-analysis/2026-10-11-dev-branch-full-explain.md
---

# Pythonファイル・クラス・関数ガイド

## 0. この資料の目的と読み方

この資料は、`dev` ブランチの **各Pythonファイルが何を担当し、各クラス・関数が何をするか** を、人が確認しやすい索引として整理したものである。

- 対象: Pythonファイル135個（実装・CLI・補助94個、テスト41個）
- ソース: `dev` / `65dedb4ebc26c48d23606b9bc674044f64e64c06`
- Git状態: `dev` と `origin/dev` が同じcommitを指し、作業ツリーはclean
- 解析方法: 静的読解のみ。コード、テスト、学習、smokeは実行していない
- 詳細なTensor shape、数式、設計理由: [既存のFull Deep Dive](2026-10-11-dev-branch-full-explain.md)を参照

表の読み方:

| 表記 | 意味 |
|---|---|
| **研究コア** | 現在のActivityNet → ViT-LoRA → Streaming MoCo研究経路 |
| **評価** | Manifest、feature抽出、Linear Probe |
| **CLI** | 上記処理を設定から組み立てて実行する入口 |
| **旧分類系** | リポジトリ雛形由来の教師あり画像・動画分類経路 |
| **補助** | 設定、hash、checkpoint、loggerなど |
| **テスト** | 実装契約を自動確認するコード。研究処理の本体ではない |

`file.py:42` は、そのファイルの42行目付近に定義があるという意味である。`__init__` はインスタンス生成時、`forward` はPyTorch moduleを `module(x)` と呼んだときに実行される。

## 1. まず全体像

### 1.1 研究経路のファイル間データフロー

```text
scripts/moco/train_full_streaming_moco.py       設定を読み、学習全体を開始
  ├─ integration/activitynet_source_selection.py  使用動画を決定
  ├─ model/backbones/vit/vit_lora_frame_encoder.py Query/Key encoder
  ├─ self_supervised/moco/vit_lora_moco.py         MoCo本体
  └─ training/streaming_moco_full.py               全動画を逐次処理
       ├─ integration/sequential_stream.py          chunkを時系列供給
       ├─ integration/sequential_moco.py            Query/Key viewとencode
       ├─ integration/sequential_vit.py             frame → CLS feature
       ├─ model/aggregators/masked_mean_clip_aggregator.py
       ├─ training/streaming_moco.py                 1 update
       └─ training/moco_checkpoint.py                resume/snapshot保存

scripts/linear_probe/build_manifest.py          評価segmentを確定
scripts/linear_probe/extract_features.py        Base/LoRA feature抽出
scripts/linear_probe/run_probe.py               線形分類で比較
  ├─ evaluation/activitynet_manifest.py
  ├─ evaluation/segment_features.py
  └─ evaluation/linear_probe.py
```

### 1.2 主要Tensorだけ先に確認

| 名前 | shape | 作られる場所 | 意味 |
|---|---:|---|---|
| `sample.frames` | `[T,3,H,W]`, 通常 `T=16` | Sequential Loader | paddingを含む動画frame列 |
| `valid_mask` | `[T]` bool | Sequential Loader | 本物のframeだけTrue |
| `pixel_values` | `[V,3,224,224]` | processor | valid frameだけをViT入力へ変換 |
| `valid_features` | `[V,768]` | ViT encoder | 各frameのCLS feature |
| `frame_features` | `[T,768]` | `encode_chunk` | 元のframe位置へscatter。padding行は0 |
| `clip_feature` | `[768]` | masked mean | 1 chunkの表現 |
| `query`, `key` | `[128]` | MoCo projector | L2正規化済みcontrastive表現 |
| `negatives` | `[N,128]`相当 | FIFO Queue | 過去のKey表現 |
| probe logits | `[B,200]` | Linear Probe | ActivityNet 200 classのscore |

## 2. ルート直下のPythonファイル

### `main.py` — 旧PyTorch教師あり学習の入口

研究コアのStreaming MoCoではなく、画像・動画分類を通常のPyTorch loopで学習する旧経路である。

| シンボル | 役割 |
|---|---|
| `TqdmEpoch.__init__` (`main.py:29`) | `start_epoch+1` から最終epochまでを表示する進捗barを作る。resume後のepoch番号を保つ。 |
| `prepare_training` (`:40`) | logger、DataLoader、model、CUDA/DataParallel、optimizer、scheduler、checkpoint resumeをまとめて準備する。 |
| `ValidationChecker.__init__` (`:126`) | validation間隔と総epoch数を保持する。 |
| `ValidationChecker.should_validate` (`:130`) | 指定間隔、または最終epochならvalidationする。 |
| `main` (`:138`) | Hydra設定を受け、epochごとに `train` → 必要なら `validation` → checkpoint保存を行う。 |

### `main_pl.py` — 旧Lightning教師あり学習の入口

| シンボル | 役割 |
|---|---|
| `main` (`main_pl.py:18`) | Hydra設定からLightning logger、DataModule、`SimpleLightningModel`、callback、Trainerを構築し `fit` を呼ぶ。 |

### `train.py` — 旧PyTorch版の1 epoch学習

| シンボル | 役割 |
|---|---|
| `TrainConfig` (`train.py:17`) | gradient accumulation間隔とlog間隔を持つdataclass。 |
| `TrainOutput` (`:23`) | 1 epoch後のstep数とloss/top-k meterを返す入れ物。 |
| `LoggerChecker.__init__` / `should_log` (`:30`, `:33`) | 現在stepがlog間隔に当たるかを判定する。 |
| `OptimizerChecker.__init__` (`:38`) | gradient accumulation間隔を保持する。 |
| `OptimizerChecker.should_zero_grad` (`:41`) | accumulation windowの先頭でのみgradientを0にする。 |
| `OptimizerChecker.should_update` (`:47`) | 指定batch数ごとにoptimizer stepする。 |
| `train` (`:51`) | `model.train()`、forward、loss backward、Top-k集計、optimizer step、epoch末scheduler stepを行う。入力はbatch画像/動画、出力logitsは `[B,C]`。 |

注意: accumulation時にlossを間隔で割らないため、「平均」ではなくgradientの和になる。端数batchの最後をstepする処理もない。

### `val.py` — 旧PyTorch版のvalidation

| シンボル | 役割 |
|---|---|
| `ValidationOutput` (`val.py:16`) | validationのloss/top-k meterを返す入れ物。 |
| `validation` (`:22`) | `eval()` と `no_grad()` で全batchを評価し、loss・Top-1・Top-5を集計する。 |

### `sequential_vit_bridge.py` — ViT bridgeの互換import

関数定義はない。`sys.modules[__name__]` を `integration.sequential_vit` に置き換えることで、古い `import sequential_vit_bridge` を新しい実装へ転送する。

### `sequential_moco_bridge.py` — MoCo bridgeの互換import

関数定義はない。古いimport pathを `integration.sequential_moco` へ転送する。

## 3. `model/` — modelと表現生成

### 3.1 研究コアのViTと集約器

#### `model/backbones/vit/vit_frame_encoder.py` — frozen ViT frame encoder

| シンボル | 入出力 | 役割 |
|---|---|---|
| `ViTFrameEncoder.__init__` (`:18`) | checkpoint ID → module | Hugging Face `ViTModel` をpoolerなしで読み、hidden sizeを確認し、全parameterをfreezeする。 |
| `ViTFrameEncoder.forward` (`:34`) | `[V,3,224,224]` → `[V,768]` | `last_hidden_state[:,0,:]` を返す。0番tokenはCLS tokenで、各frame全体の特徴を表す。 |

重要箇所:

```python
self.vit = ViTModel.from_pretrained(checkpoint_id, add_pooling_layer=False)
self.vit.requires_grad_(False)
return self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
```

#### `model/backbones/vit/vit_lora_frame_encoder.py` — LoRA付きViT frame encoder

| シンボル | 入出力 | 役割 |
|---|---|---|
| `ViTLoRAFrameEncoder.__init__` (`:22`) | checkpoint/LoRA設定 → module | Base ViTをfreezeし、PEFTで全12 blockの `query` と `value` projectionへLoRAを注入する。Base重みは更新しない。 |
| `ViTLoRAFrameEncoder.forward` (`:49`) | `[V,3,224,224]` → `[V,768]` | LoRA適用済みViTのCLS featureを返す。Query側ではgradientあり、Key側では呼び出し元が `no_grad` にする。 |

#### `model/aggregators/masked_mean_clip_aggregator.py` — frame列をchunkへ集約

| シンボル | 入出力 | 役割 |
|---|---|---|
| `MaskedMeanClipAggregator.forward` (`:14`) | `[T,D]`, bool `[T]` → `[D]` | shape、mask型、最低1 valid frameを検証し、True行だけを平均する。paddingは分子・分母の両方から除外する。 |

```python
return frame_features[valid_mask.to(frame_features.device)].mean(dim=0)
```

例えば `valid_mask=[True, True, False]` なら、3行目がどんな値でも1・2行目だけの平均になる。平均なのでframe順を逆にしても同じ表現になる。

#### `model/backbones/vit/vision_transformer.py` — 旧教師ありViT分類器

| シンボル | 役割 |
|---|---|
| `ViTb.__init__` (`:13`) | 共通 `ClassificationBaseModel` を初期化し、分類用ViTを準備する。 |
| `ViTb.prepare_model` (`:17`) | `ViTForImageClassification` をpretrained有無とclass数に応じて作る。 |
| `ViTb.forward` (`:24`) | 画像 `[B,3,H,W]` とoptional labelをHF modelへ渡し、共通 `ModelOutput(logits, loss)` に変換する。研究コアのCLS抽出器とは別物。 |

#### `model/backbones/vit/__init__.py`

`ViTb`、`ViTFrameEncoder`、`ViTLoRAFrameEncoder` をpackage外へ再公開する。処理関数はない。

### 3.2 共通model基盤

#### `model/model_config.py`

| シンボル | 役割 |
|---|---|
| `ModelConfig` (`:16`) | model名、pretrained利用、`TORCH_HOME`、class数を保持するdataclass。旧分類系model factoryで使う。 |

#### `model/base_model.py`

| シンボル | 役割 |
|---|---|
| `ModelOutput` (`:10`) | `logits` とoptional `loss` を統一形式で返す `NamedTuple`。 |
| `ClassificationBaseModel.__init__` (`:17`) | configとCrossEntropyLossを保存し、仮modelを置く。 |
| `ClassificationBaseModel.__create_dummy_model` (`:24`) | subclassが実modelを置くまでの空 `nn.Module` を作るprivate helper。 |
| `ClassificationBaseModel.forward` (`:27`) | `self.model(pixel_values)` でlogitsを作り、labelsがあればcross entropyも計算する。 |
| `get_device` (`:48`) | DataParallelなら内部module、それ以外なら先頭parameterからdeviceを返す。 |

#### `model/model_factory.py`

| シンボル | 役割 |
|---|---|
| `set_torch_home` (`:12`) | pretrained weight cacheの環境変数 `TORCH_HOME` を設定する。 |
| `configure_model` (`:21`) | configのmodel名からResNet18/50、ABN、ViT、X3D、ZeroOutputModelのいずれかを構築するfactory。必要ならDataParallel化する。 |

#### `model/simple_lightning_model.py`

| シンボル | 役割 |
|---|---|
| `SimpleLightningModel.__init__` (`:26`) | model factoryから分類modelを作り、optimizer/scheduler/log設定を保存する。 |
| `configure_optimizers` (`:64`) | `setup` packageのfactoryでoptimizerとschedulerを作りLightning形式で返す。 |
| `configure_callbacks` (`:83`) | validation Top-1を監視するcheckpoint callbackを返す。 |
| `log_train_loss_top15` (`:105`) | train loss・Top-1・Top-5をLightning loggerへ記録する。関数名の `top15` はTop-1/5の意味。 |
| `training_step` (`:133`) | 1 train batchをforwardし、metricをlogしてlossを返す。backward/stepはLightning担当。 |
| `log_val_loss_top15` (`:165`) | validation metricを同期付きで記録する。 |
| `validation_step` (`:180`) | 1 validation batchをforwardし、loss・Top-1・Top-5を記録する。 |

#### `model/__init__.py`

model config、base model、全backbone、aggregator、dummy model、factory、Lightning wrapperを一つのimport面へ再公開する。

#### `model/backbones/__init__.py`

backbone packageの説明docstringだけを持つ。再公開は各subdirectoryの `__init__.py` が担当する。

#### `model/aggregators/__init__.py`

`MaskedMeanClipAggregator` を再公開する。

### 3.3 旧分類backbone

#### `model/backbones/resnet/resnet18.py`

| シンボル | 役割 |
|---|---|
| `ResNet18.__init__` (`:13`) | Base classを初期化しResNet18を準備する。 |
| `ResNet18.prepare_model` (`:18`) | torchvision ResNet18をImageNet pretrained有無に応じて作る。 |
| `ResNet18.replace_pretrained_head` (`:23`) | 最終 `fc` を対象class数のLinearへ交換する。 |

#### `model/backbones/resnet/resnet50.py`

`ResNet50.__init__` (`:14`)、`prepare_model` (`:19`)、`replace_pretrained_head` (`:24`) はResNet18版と同じ責務で、backboneだけResNet50である。

#### `model/backbones/resnet/__init__.py`

`ResNet18` と `ResNet50` を再公開する。

#### `model/backbones/x3d/x3d.py`

| シンボル | 役割 |
|---|---|
| `X3DM.__init__` (`:10`) | 動画分類用X3D-Mを準備する。 |
| `X3DM.prepare_model` (`:15`) | `torch.hub` からX3D-Mを読み、末尾Softmaxを外してlogitsを扱える形にする。 |
| `X3DM.replace_pretrained_head` (`:23`) | X3Dの分類headを対象class数へ交換する。入力は通常 `[B,3,T,H,W]`。 |

#### `model/backbones/x3d/__init__.py`

`X3DM` を再公開する。

#### `model/backbones/abn/attention_branch_network.py`

| シンボル | 役割 |
|---|---|
| `ABNResNet50.__init__` (`:14`) | 共通分類baseを初期化する。 |
| `ABNResNet50.prepare_model` (`:18`) | 内部 `ABNResNet50Model` を作る。実際に読むbackboneはCIFAR100 ResNet56で、クラス名と異なる点に注意。 |
| `ABNResNet50.forward` (`:24`) | attention branchとperception branchのlogits/lossを共通 `ModelOutput` にまとめる。 |
| `ABNResNet50Model.__init__` (`:53`) | feature extractor、attention branch、perception branchを構築する。 |
| `__create_attention_branch` (`:76`) | class activation mapと1-channel attention map、補助分類logitsを作る層を構築する。 |
| `__split_r50_into_half` (`:106`) | 読み込んだnetworkをfeature extractorと後段classifierへ分割する。 |
| `__get_last_num_features` (`:130`) | module列末尾からfeature channel数を得る。 |
| `get_attention_map` (`:133`) | 直近forwardで保存したattention mapを返す。 |
| `__gap` (`:136`) | Global Average Poolingで `[B,C,H,W]` を `[B,C]` にする。 |
| `attention_branch` (`:142`) | featureからattention mapと補助logitsを計算する。 |
| `forward` (`:152`) | attentionでfeatureをgateし、perception branchで最終分類する。 |

#### `model/backbones/abn/__init__.py`

`ABNResNet50` を再公開する。

#### `model/dummy_models/zero_outout_model.py`

ファイル名の `outout` は綴り違いだがimport pathとして固定されている。

| シンボル | 役割 |
|---|---|
| `ZeroOutputModel.__init__` (`:11`) | 配管試験用のparameterなしmodelを初期化する。 |
| `ZeroOutputModel.forward` (`:19`) | batch sizeに合わせた全0 logits `[B,C]` と0 lossを返す。実学習用ではない。 |

#### `model/dummy_models/__init__.py`

`ZeroOutputModel` を再公開する。

## 4. `integration/` — loader出力とmodelを接続

### `integration/sequential_vit.py` — 1 chunkをframe feature列へ変換

| シンボル | 入出力 | 役割 |
|---|---|---|
| `encode_chunk` (`:15`) | sample → `[T,D]` | valid frame抽出 → processor → encoder → 元のT位置へscatterの順で処理し、padding行を0にする。shape、finite値、metadata不変も検査する。 |

重要な処理:

```python
valid_frames = sample.frames[valid_mask]
pixel_values = processor(images=valid_frames, return_tensors="pt")["pixel_values"]
valid_features = encoder(pixel_values.to(device))
frame_features = torch.zeros((sample.frames.shape[0], feature_size), ...)
frame_features[valid_mask.to(frame_features.device)] = valid_features
```

`T=16, V=13` なら、encoderが処理するのは13枚だけで、戻り値は `[16,768]`。最後の3行は0である。

### `integration/sequential_moco.py` — MoCoの2 viewとencoding

| シンボル | 役割 |
|---|---|
| `make_two_views` (`:11`) | Queryにはraw sample、Keyには指定変換を加えたsampleを作る。`horizontal_flip` または `gbr_horizontal_flip`。paddingとmetadataは保つ。 |
| `encode_query_view` (`:23`) | `encode_chunk` → masked mean → Query projector。Query LoRA/projectorへgradientを流す。 |
| `encode_key_view` (`:30`) | Key encoder側で同じ処理を `no_grad` で行い、正規化済みKeyを返す。 |

### `integration/sequential_stream.py` — chunkを順序どおり供給

| シンボル | 役割 |
|---|---|
| `source_count` (`:17`) | `strict_single` なら1、`round_robin` なら指定stream数を返す。 |
| `validate_sources` (`:25`) | 必要本数、重複、source contractを学習開始前に検証する。 |
| `ordered_samples` (`:34`) | Readerを開き、strict順またはround-robin順でsampleを遅延yieldする。prefetchせず、終了・例外時はReaderを閉じる。 |
| nested `iterate` (`:56`) | 実際のgenerator本体。各streamのframe index連続性、mask、terminal paddingなどを確認してyieldする。 |

### `integration/activitynet_source_selection.py` — full/reduced動画集合を決定

| シンボル | 役割 |
|---|---|
| `_uint32` (`:15`) | seedが符号なし32-bit範囲内か検証する。 |
| `SourceSelectionConfig.from_mapping` (`:29`) | plain dictからprofile、strategy、seed、splitごとの件数を検証してdataclass化する。 |
| `SourceSelectionConfig.metadata` (`:59`) | configをartifactに記録できる安定したdictへ変換する。 |
| `SourceSelection` (`:67`) | 選択済みsource群、ID、hash、metadataをまとめるdataclass。 |
| `approved_profile_name` (`:73`) | repositoryが認めるexact full/reduced設定にだけ正式profile名を与える。 |
| `_score` (`:88`) | strategy、seed、split、sequence IDをSHA-256へ入れ、再現可能なranking scoreを作る。 |
| `_ordered_ids` (`:95`) | full sourceの件数とID一意性を検証し、Adapter順のID列を返す。 |
| `select_activitynet_sources` (`:106`) | hash score上位を選ぶが、実処理順は元のAdapter順へ戻す。選択IDと順序hashも作る。 |
| `validate_selection_metadata` (`:154`) | artifactのselection metadataが期待値と完全一致するか確認する。 |
| `validate_selection_record` (`:161`) | record schema、件数、hash、profileなど内部整合性を検証する。 |

### `integration/__init__.py`

loaderとmodelのadapter packageであることを示すdocstringだけを持つ。

## 5. `self_supervised/` — MoCo model primitive

### `self_supervised/moco/negative_selection.py`

| シンボル | 役割 |
|---|---|
| `select_negatives` (`:4`) | Queue entryから `different_sequence` なら現在と異なる動画だけ、`all_past` なら全過去entryを順序保持して返す。空なら明示的に失敗する。 |

### `self_supervised/moco/vit_lora_moco.py`

| シンボル | 役割 |
|---|---|
| `lora_parameters` (`:25`) | encoder内で `requires_grad=True` のLoRA parameterだけを列挙する。 |
| `require_normalized` (`:32`) | vector shape、有限性、L2 norm≈1を検証する。Queueへの壊れたkey混入を防ぐ。 |
| `QueueEntry` (`:40`) | `key [128]`、`sequence_id`、`sequence_index` を束ねるdataclass。 |
| `MetadataQueue.__init__` (`:49`) | `deque(maxlen=capacity)` を作る。既定4096件。 |
| `MetadataQueue.__len__` (`:58`) | 現在entry数を返す。 |
| `MetadataQueue.entries` (`:62`) | FIFO順のcopy/listを返す。 |
| `MetadataQueue.enqueue` (`:65`) | keyを検証して `detach().clone()` し、metadataと一緒に末尾へ追加する。満杯なら最古を落とす。 |
| `MetadataQueue.negatives` (`:73`) | 後方互換用に `different_sequence` policyでnegativeを選ぶ。 |
| `ViTLoRAMoCo.__init__` (`:87`) | Query encoder、deep copyしたKey encoder、両projector、aggregator、Queue、temperature、momentumを構築する。Keyはfreeze。 |
| `ViTLoRAMoCo.query_parameters` (`:118`) | optimizer対象であるQuery LoRAとQuery projector parameterを返す。 |
| `project_query` (`:121`) | clip `[768]` をMLPで `[128]` へ落とし、L2正規化する。gradientあり。 |
| `project_key` (`:127`) | Key projector版。呼び出しは `no_grad`。 |
| `contrastive_loss` (`:132`) | positiveをlogit 0、過去Queueをnegativeとしてtemperature-scaled InfoNCEを計算する。Key側はdetach。 |
| `update_key` (`:147`) | Key LoRA/projectorを `theta_k = m*theta_k + (1-m)*theta_q` でEMA更新する。 |

InfoNCEの出力はscalar。negativeがN件なら内部logitsは `[1,1+N]` である。

### `self_supervised/moco/__init__.py`

`MetadataQueue` と `ViTLoRAMoCo` を再公開する。

### `self_supervised/__init__.py`

self-supervised method packageであることを示すdocstringだけを持つ。

## 6. `training/` — Streaming MoCoの学習制御

### `training/moco_protocol.py` — 3軸protocol定義

| シンボル | 役割 |
|---|---|
| `StreamingMoCoProtocol.__post_init__` (`:20`) | stream mode、Key変換、negative policy、stream数の組合せを検証する。dataclass生成直後に動く。 |
| `source_count` property (`:32`) | active stream数を返す。 |
| `warmup_count` property (`:36`) | 各active streamで最初にKey-only投入する件数を返す。strictなら1、4-streamなら4。 |

### `training/moco_audit.py` — parameter/view契約の静的・実行時監査

| シンボル | 役割 |
|---|---|
| `audit_encoder_parameters` (`:9`) | trainable parameterがQ/V LoRAだけか、個数294,912・48 tensorsか、Baseがfreezeかを確認する。 |
| `parameter_groups` (`:47`) | Query LoRA、Query projector、Query base、Key LoRA、Key projector、Key baseへparameterを分類する。 |
| `audit_parameters` (`:62`) | optimizer対象、Query/Key初期一致、trainable/frozen状態をまとめて検証する。 |
| `audit_initial_state` (`:114`) | 初期Queueが空、Query/Key対応parameterが同一値などrun開始条件を検証する。 |
| `audit_views` (`:130`) | Query/Key sampleのshape、padding、metadata不変と、指定された変換だけが起きたことを確認する。 |

### `training/streaming_moco.py` — 1 updateの正本

| シンボル | 役割 |
|---|---|
| `report_json` (`:17`) | eventと値を1行JSONとして標準出力へ出す。smoke/testで機械的に読める。 |
| `cpu_snapshot` (`:21`) | parameterをdetach・CPU cloneし、更新前比較用snapshotを作る。 |
| `count_changed` (`:25`) | snapshotと現在値を比べ、変わったtensor数を数える。 |
| `audit_bases` (`:30`) | Query/KeyのBase ViTが一切変化していないことを確認する。 |
| `audit_finite` (`:36`) | MoCo parameterにNaN/Infがないことを確認する。 |
| `audit_queue` (`:41`) | Queue長、capacity、FIFO metadata、key shape/正規化を確認する。 |
| `enqueue_key` (`:64`) | sample metadata付きでcurrent KeyをQueueへ追加し、期待Queue状態を同期する。 |
| `_gradient_diagnostics` (`:74`) | parameter groupごとのgradient有限性、norm、非zero tensor数を集計する。 |
| `train_step` (`:92`) | 2 view作成、Query/Key forward、old Queue negative、InfoNCE、backward、Query optimizer、Key EMA、current Key enqueueをこの順で行う。 |

```text
forward(q,k) → old queueでloss → backward → Query更新 → Key EMA → kをenqueue
```

この順序は重要で、current Keyが自分自身のnegativeになることを避ける。

### `training/moco_canary.py` — 短時間multi-step確認

| シンボル | 役割 |
|---|---|
| `round_robin_samples` (`:27`) | 4 sourceをround-robinで読む互換wrapper。 |
| `run_streaming_moco` (`:32`) | protocolに沿ってwarm-upと複数 `train_step` を回し、Queue/parameter/reader cleanupを監査するbounded engine。 |
| nested `take_sample` (`:61`) | iteratorから次sampleを取り、EOFを統一的に扱う。 |
| `run_canary` (`:114`) | Stage 6A既定protocolで `run_streaming_moco` を呼ぶ便利wrapper。 |

### `training/moco_config.py` — typed設定とcanonical設定

| シンボル | 役割 |
|---|---|
| `_positive_int` (`:16`) | 値がboolではない正整数かを確認する共通validator。 |
| `ActivityNetConfig.from_mapping` (`:32`) | dataset root、annotation、split件数を検証してdataclass化する。 |
| `LoRAConfig.from_mapping` (`:58`) | rank、alpha、dropout、target modules等を検証する。 |
| `LoRAConfig.metadata` (`:71`) | LoRA設定をsnapshot/identity用dictにする。 |
| `EncoderConfig.from_mapping` (`:87`) | checkpoint ID、feature size、LoRAをまとめる。 |
| `SequentialConfig.from_mapping` (`:105`) | frames per chunkなどloader接続条件を検証する。 |
| `OptimizerConfig.from_mapping` (`:121`) | AdamWのlr、weight decay等を検証する。 |
| `IntervalConfig.from_mapping` (`:142`) | resume/snapshot/metric intervalを検証する。 |
| `MoCoConfig.from_mapping` (`:162`) | projection size、queue、momentum、temperature、3軸protocolを検証する。 |
| `RuntimeConfig.from_mapping` (`:199`) | run dir、resume、stop_after、production等の運用設定を検証する。 |
| `FullMoCoConfig.from_mapping` (`:228`) | 上記全groupを一つのtyped configへ統合する。 |
| `FullMoCoConfig.production_groups_match` (`:253`) | 科学条件がrepository canonical groupと一致するか判定する。tracking等は除外。 |
| `FullMoCoConfig.effective_protocol_identity` (`:270`) | 実際に使うstream/view/negative条件を安定したidentity dictにする。 |
| `canonical_activitynet_config` (`:276`) | repository既定ActivityNet設定を読む。 |
| `canonical_encoder_config` (`:280`) | 既定ViT/LoRA設定を読む。 |
| `canonical_sequential_config` (`:284`) | 既定chunk設定を読む。 |
| `canonical_moco_config` (`:288`) | 既定MoCo設定を読む。 |

各 `*Config` class自体は設定値を保持するimmutable dataclassである。

### `training/moco_checkpoint.py` — resumeと評価snapshot

| シンボル | 役割 |
|---|---|
| `ordered_source_sha256` (`:51`) | 順序込みsource ID列のhashを作る。並び替えも別runとして検出する。 |
| `seed_all` (`:55`) | Python、NumPy、Torch、CUDAのRNG seedを揃える。 |
| `base_fingerprint` (`:66`) | encoder Base重みの内容hashを作る。 |
| `encoder_base_fingerprint` (`:79`) | plain/PEFT wrapperの違いを吸収して同じBaseのfingerprintを得る。 |
| `validate_production_snapshot_metadata` (`:103`) | production snapshotのschema、commit、selection、LoRA/config契約を厳密に検証する。 |
| `lora_config` (`:191`) | encoderから実際のPEFT LoRA設定を抽出する。 |
| `optimizer_config` (`:199`) | optimizerのclass名とparam group hyperparameterを記録する。 |
| `_cpu` (`:209`) | nested tensor構造を再帰的にCPUへ移す。 |
| `rng_state` (`:219`) | Python/NumPy/Torch/CUDAのRNG stateをまとめる。 |
| `restore_rng_state` (`:232`) | 保存RNG stateを復元する。 |
| `training_state` (`:246`) | Query/Key LoRA、projector、optimizer、Queue、RNGをresume payloadへまとめる。 |
| `save_resume_checkpoint` (`:266`) | identity/counter/stateをchecksum付きでatomic保存する。 |
| `load_resume_checkpoint` (`:290`) | payload hashとschemaを検証してresume dataを読む。 |
| `validate_resume_identity` (`:314`) | 保存runと現在runの科学条件・source順等が同じか確認する。 |
| `restore_training_state` (`:321`) | model、optimizer、Queue、RNGを完全復元する。 |
| `snapshot_name` (`:343`) | processed video数/update数/final flagから一意なdirectory名を作る。 |
| `query_lora_state` (`:347`) | Query encoderからLoRA重みだけを抽出する。 |
| `write_evaluation_snapshot` (`:351`) | PEFT native adapter filesと厳密metadata/hashをatomicに公開する。projectorやQueueは含まない。 |
| `load_query_lora_snapshot` (`:393`) | snapshot契約とhashを検証しQuery LoRAへ読み込む。 |

### `training/streaming_moco_full.py` — full/reduced datasetの本学習loop

| シンボル | 役割 |
|---|---|
| `RunPaths.__init__` (`:50`) | run rootからevent log、resume、snapshot等の固定pathを作る。 |
| `_Recorder.__init__` (`:62`) | JSONL event file、Comet experiment、metric intervalを保持する。 |
| `_Recorder.event` (`:69`) | 即時eventをlocal JSONLへ記録する。 |
| `_Recorder.step` (`:75`) | update metricをwindowへ蓄積する。 |
| `_Recorder.flush_metrics` (`:79`) | intervalごとの平均metricをlocal/Cometへ流す。 |
| `_Recorder.close` (`:99`) | 残metricをflushしてfileを閉じる。 |
| `mean_grad` (`:103`) | step rowsのgradient metricを平均する。 |
| `run_identity` (`:107`) | commit、config、source selection、Base fingerprint等からresume比較用identityを作る。 |
| `_new_optimizer` (`:146`) | Query LoRA + Query projectorだけを対象にAdamWを作る。 |
| `_device_snapshot` (`:161`) | parameterの更新前copyをdevice上に保持する。 |
| `validate_full_sources` (`:166`) | 選択source件数・重複・空を検証する。 |
| `run_full_streaming_moco` (`:173`) | sourceを一度だけ順番に処理し、最初の1 chunkをwarm-up、その後update、動画境界resume、定期snapshot、stop/finalを統括する。 |
| nested `identity_now` (`:276`) | 現在model/configからidentityを再構成する。 |
| nested `snapshot_metadata` (`:282`) | evaluation snapshotへ持たせるcounter、selection、provenance等を作る。 |
| nested `save_snapshot` (`:308`) | 重複を避けながらevaluation snapshotを保存する。 |
| nested `snapshot_due` (`:327`) | 現在processed video数がsnapshot intervalに当たるか判定する。 |
| nested `save_resume` (`:330`) | 動画境界の完全stateを `latest.pt` へ保存する。 |

### `training/__init__.py`

model primitiveとは別のconsumer-side training orchestration packageであることを示すdocstringだけを持つ。

## 7. `evaluation/` — 下流評価

### `evaluation/activitynet_manifest.py` — どのsegment/chunkを評価するか固定

| シンボル | 役割 |
|---|---|
| `AnnotationIndex.__init__` (`:41`) | 期待するActivityNet annotation versionと、読込済みdatabase cacheを準備する。 |
| `AnnotationIndex.database` (`:47`) | JSONを読み、version・database構造を検証して返す。 |
| `AnnotationIndex.annotations` (`:57`) | source referenceに対応するannotationを正規化して返す。 |
| `canonical_label_mapping` (`:74`) | training/validation両方のlabel集合が一致することを確認し、alphabetical順に0..C-1を割り当てる。 |
| `chunk_in_segment` (`:91`) | chunk timestampがclosed interval `[start,end]` に完全包含されるかを判定する。 |
| `video_chunk_timestamps` (`:95`) | 1動画をSequential Loaderで読み、各chunkのvalid timestamp列を記録する。 |
| `contained` (`:111`) | timestampが有限・非空で、全値がsegment範囲内か確認する。 |
| `video_rows` (`:116`) | 1動画のannotationごとに、完全包含chunk indexを持つmanifest rowを作る。 |
| `build_manifest` (`:134`) | split → Adapter source順 → annotation index順で全rowを作る。 |
| `manifest_bytes` (`:152`) | row列を決定的JSONL bytesへ変換する。hashの正本。 |
| `integrity_gate` (`:156`) | class coverage、重複、split overlap、mapping hash、chunk数などをGate 1–8として検証する。 |
| `gate_status` (`:184`) | gate結果をproduction/reproducible条件込みの状態へまとめる。 |
| `write_manifest` (`:196`) | manifest、label mapping、metadataをhash付きでatomic保存する。 |
| `load_manifest` (`:204`) | file hashとGate状態を再確認してartifactを読む。 |

### `evaluation/segment_features.py` — manifest segmentを768次元へ変換

| シンボル | 入出力・役割 |
|---|---|
| `validate_feature_definition` (`:36`) | featureが「raw CLS → valid-frame masked mean → chunk mean」という実装済み定義か確認する。 |
| `shared_contract_sha256` (`:42`) | Base/LoRAで共有すべきfeature contractのhashを作る。 |
| `validate_shared_contract` (`:46`) | metadataの共有contract本体とhashが整合するか確認する。 |
| `validate_feature_pair` (`:58`) | BaseとLoRAのmanifest、segment ID/order、label、feature定義が同じか確認する。 |
| `feature_size_from_metadata` (`:69`) | metadataから正のfeature次元を読む。通常768。 |
| `chunk_feature` (`:77`) | `encode_chunk [T,768]` をmasked meanし、1 chunk `[768]` を返す。 |
| `video_segment_features` (`:86`) | manifest指定chunkを読み、chunk featureを単純平均して各segment `[768]` を作る。 |
| `extract_split` (`:114`) | split内全rowについてsourceを読み、feature `[Nsegment,768]` とlabels/IDsを作る。 |
| `save_split` (`:141`) | 1 splitのtensor payloadをatomic保存する。 |
| `verify_split` (`:148`) | feature shape、finite、dtype、labels、segment orderがmanifestと合うか確認する。 |
| `load_features` (`:163`) | 保存splitとmetadata/hashを読み、conditionも含めて検証する。 |
| `write_feature_artifact` (`:187`) | training/validation tensorとmetadata、checksumを一つのartifact directoryへ公開する。 |

segment表現は `mean(chunk_feature)` であり、chunk内valid frame数による重み付けはしない。

### `evaluation/linear_probe.py` — 固定feature上の線形分類

| シンボル | 役割 |
|---|---|
| `ProbeConfig.from_mapping` (`:53`) | protocol名と設定dictからepochs、batch、lr、weight decay、seeds等を検証して作る。 |
| `ProbeConfig.__post_init__` (`:76`) | dataclass直接生成時にも値域・許可値を検証する。 |
| `ProbeConfig.with_epochs` (`:106`) | 他条件を保ちepochsだけ替えたcopyを作る。smoke用。 |
| `ProbeConfig.hyperparameters` (`:109`) | feature size/class数を含む実験記録用dictを返す。 |
| `experiment_name` (`:137`) | protocol/condition/seedから一貫したrun名を作る。 |
| `train_probe` (`:144`) | `Linear(D,C)` だけをAdamW + cross entropyで学習する。feature extractorは存在せず、入力tensorも変更しない。 |
| `evaluate` (`:191`) | validation logits `[N,C]` からTop-1、class別精度、Macro精度、confusion matrix `[C,C]` を計算する。 |
| `mean_std` (`:217`) | seed間の平均とsample標準偏差 (`ddof=1`) を返す。 |
| `write_result` (`:224`) | classifier state、history、metrics、summary、CSV、各hashを保存する。 |
| `_csv` (`:246`) | field順を固定したCSV bytesを作る。 |
| `load_summary` (`:254`) | summary JSONとhashを検証して読む。 |
| `aggregate` (`:263`) | conditionごとの複数seed結果をmean/stdへ集約する。 |
| `aggregate_csv` (`:283`) | aggregate結果を人間確認用CSVへ変換する。 |

### `evaluation/__init__.py`

label-awareな下流評価をSequential Loader coreから分離するpackage docstringだけを持つ。

## 8. `scripts/` — 人が起動するCLI

CLIファイルは「科学計算の本体」よりも、設定読込、provenance、artifact path、失敗時停止、Comet連携を担当する。実際のmodel計算は前節までのmoduleへ委譲する。

### 8.1 本学習CLI

#### `scripts/moco/train_full_streaming_moco.py`

| シンボル | 役割 |
|---|---|
| `run` (`:35`) | Hydra設定をplain/typed config化し、provenance、ActivityNet Adapter inventory、source selection、processor、Query/Key encoder、MoCo、Cometを構築して `run_full_streaming_moco` を呼ぶ。 |
| `main` (`:162`) | Hydra decorator付きcommand entry。設定合成後に `run` へ渡す。 |

#### `scripts/moco/__init__.py`

production Streaming MoCo entrypoint packageであることを示すdocstringだけを持つ。

### 8.2 Linear Probe pipeline CLI

#### `scripts/linear_probe/configuration.py`

| シンボル | 役割 |
|---|---|
| `_science_view` (`:17`) | path、tracking、実行時だけのfieldを除き、科学条件としてhashすべき部分へ射影する。 |
| `resolved_science` (`:39`) | 指定config group群を完全resolveした科学設定dictにする。 |
| `science_contract` (`:47`) | resolved値、canonical一致、有効条件、SHA-256をまとめる。 |
| `feature_science_contract` (`:61`) | probe optimizer/epochなどを除き、feature artifactまでで共有する契約を作る。 |

#### `scripts/linear_probe/build_manifest.py`

| シンボル | 役割 |
|---|---|
| `selected_sources` (`:37`) | Adapter full inventoryを読み、full/reduced設定に従い各splitのsourceを確定する。 |
| `resolved_manifest_id` (`:67`) | base manifest IDにreduced selection hash suffixを必要に応じて付ける。 |
| `resolve_snapshot_manifest_id` (`:75`) | LoRA snapshot記録のselectionと現在の再計算selectionを照合し、正しいmanifest IDを先に決める。 |
| `progress` (`:110`) | 一定間隔でsplit進捗をJSON出力する。 |
| `build` (`:115`) | label mapping、chunk timestamp、rows、Gate 1–8、metadataを作ってmanifest artifactを保存する。 |
| `audit` (`:181`) | 保存manifestを独立に再生成し、bytes/hashが一致するGate 9を確認する。 |
| `run` (`:250`) | config検証後、build/audit modeを振り分ける。 |
| `main` (`:285`) | Hydra command entry。 |

#### `scripts/linear_probe/extract_features.py`

| シンボル | 役割 |
|---|---|
| `full_sources_and_selection` (`:40`) | Adapter full sourcesと現在selectionを再構成する。 |
| `dataset_sources_and_identity` (`:65`) | manifest selectionとdataset inventoryを照合し、mount pathではなくcontent/orderに基づくdataset identityを作る。 |
| `extraction_provenance` (`:106`) | feature比較に影響するprovenanceだけを抽出する。 |
| `preprocessing_identity` (`:115`) | processor class/config hash、checkpoint、frames per chunk、出力dtypeを記録する。 |
| `build_encoder` (`:125`) | `base` ならfrozen ViT、`lora` ならViT-LoRAを作りsnapshot contract/hashを検証してadapterを読む。 |
| `run` (`:190`) | manifestを読み、training/validationを抽出し、既存artifactは完全一致時だけreuseする。 |
| nested `progress` (`:312`) | 現在splitを束縛し、抽出進捗をJSON表示するcallback。 |
| `main` (`:344`) | Hydra command entry。 |

#### `scripts/linear_probe/run_probe.py`

| シンボル | 役割 |
|---|---|
| `aggregate_metadata` (`:37`) | 集約artifact用のimmutable metadataを作り、可変なComet状態を分離する。 |
| `run_seed` (`:49`) | 1 condition × 1 seedについてprobe学習、評価、artifact保存、Comet登録を行う。 |
| `run` (`:114`) | Base/LoRA feature pairを検証し、通常2 conditions × 3 seedsを実行してmean/stdを集約する。 |
| `main` (`:308`) | Hydra command entry。 |

#### `scripts/linear_probe/__init__.py`

manifest、feature、probeの3 entrypoint packageであることを示すdocstringだけを持つ。

### 8.3 Comet再送CLI

#### `scripts/retry_comet_artifact.py`

| シンボル | 役割 |
|---|---|
| `main` (`:18`) | 既存artifactのimmutable file hashを再検証し、以前disabled/failedだったComet uploadだけを再試行する。科学artifact本体は変更しない。 |

### 8.4 smoke / audit CLI

#### `scripts/smoke/smoke_vit_frame_encoder.py`

| シンボル | 役割 |
|---|---|
| `main` (`:17`) | 1画像をprocessor→frozen ViTへ通し、`[1,3,224,224] → [1,768]`、finite、全parameter freezeを手動確認する。 |

#### `scripts/smoke/smoke_50salads_vit_bridge.py`

| シンボル | 役割 |
|---|---|
| `git_output` (`:29`) | 指定repositoryでgit commandを実行し文字列を返すprovenance helper。 |
| `main` (`:35`) | pinned Sequential Loader revisionを確認し、50Salads最初のchunkをfrozen ViT bridgeへ通してshape/padding/metadataを表示する。 |

#### `scripts/smoke/smoke_activitynet_vit_clip_feature.py`

| シンボル | 役割 |
|---|---|
| `git_output` (`:32`) | provenance用git helper。 |
| `main` (`:38`) | ActivityNet先頭chunkをfrozen CLS `[T,768]` とmasked-mean `[768]` へ変換し、reader cleanupも確認する。 |

#### `scripts/smoke/smoke_activitynet_vit_lora_one_step.py`

| シンボル | 役割 |
|---|---|
| `git_output` (`:37`) | provenance用git helper。 |
| `run_one_step` (`:43`) | clip featureに工学的なmean-square lossを置き、LoRA gradient/update、Base不変、finiteを1 stepだけ確認する。研究目的lossではない。 |
| `main` (`:87`) | provenance、ActivityNet sample、LoRA encoder、parameter audit、`run_one_step` を統括する。 |

#### `scripts/smoke/smoke_activitynet_vit_lora_moco_one_step.py`

| シンボル | 役割 |
|---|---|
| `audit_provenance` (`:32`) | consumer/loader repository、branch、commit、依存pinを確認する。 |
| `read_first_chunk` (`:62`) | 1 sourceだけReaderで読み、最初のchunk contractを確認して必ず閉じる。 |
| `report_sample` (`:88`) | sequence/frame/mask/timestamp等を人間向けに表示する。 |
| `report_features` (`:95`) | frame/clip/projected featureのshape・finite・normを表示検証する。 |
| `run_one_step` (`:114`) | old QueueのnegativeでInfoNCE、backward、Query更新、Key EMA、enqueueを1回行い監査する。 |
| `main` (`:192`) | 2動画の先頭chunkを使い、1つをwarm-up、1つをupdateへ使うsmoke全体を組む。 |

#### `scripts/smoke/smoke_activitynet_vit_lora_moco_multistep.py`

| シンボル | 役割 |
|---|---|
| `main` (`:29`) | ActivityNet先頭4 sourceでStage 6A互換のround-robin multi-step canaryを実行する。 |

#### `scripts/smoke/smoke_activitynet_streaming_moco.py`

| シンボル | 役割 |
|---|---|
| `main` (`:31`) | Stage 6A/6B presetまたは3軸overrideをCLI引数から作り、bounded `run_streaming_moco` を起動する。 |

#### `scripts/smoke/audit_activitynet_inventory.py`

| シンボル | 役割 |
|---|---|
| `normalize_video_id` (`:20`) | filename stem先頭の `v_` だけを除きannotation ID形式へそろえる。 |
| `scan_root` (`:41`) | rootを再帰走査し、動画ID、extension別件数、重複を集計する。 |
| `main` (`:84`) | annotation databaseと複数local rootを照合し、missing、local-only、duplicateを報告する。 |

#### `scripts/smoke/audit_activitynet_timestamp_alignment.py`

| シンボル | 役割 |
|---|---|
| `NearestFrame` (`:24`) | annotation境界・中点など各target秒に最も近いdecoded frame情報を保持するdataclass。 |
| `parse_args` (`:32`) | split、対象動画数、許容誤差、保存先などを読む。 |
| `load_activitynet_database` (`:90`) | ActivityNet annotation JSONのversion/database構造を検証する。 |
| `parse_annotations` (`:106`) | label、start、end、midpointを有限な秒へ正規化する。 |
| `seconds_from_av_time` (`:170`) | PyAV timestamp/time_base等を安全に秒へ変換し、不正ならNoneにする。 |
| `probe_video` (`:180`) | container/streamのstart、duration、fps、time base、frame数を調べる。 |
| `sanitize_filename` (`:221`) | 出力画像名に使えない文字を安全な文字へ変換する。 |
| `save_frame` (`:225`) | uint8 CHW frameをHWC画像へ変換してPNG保存する。 |
| `inspect_video` (`:245`) | decode時刻の逆行、annotation target最近傍frame、chunk overlap/完全包含を調べ、代表frameと詳細reportを作る。 |
| `main` (`:492`) | 対象source選択、動画ごとの `inspect_video`、全体summary保存を統括する。 |

#### `scripts/smoke/__init__.py`

手動smoke/audit用packageであることを示すdocstringだけを持つ。

#### `scripts/__init__.py`

実行可能project utility packageであることを示すdocstringだけを持つ。

## 9. `dataset/` — 旧教師あり分類用DataLoader

研究コアのActivityNet逐次読込は外部 `sequential_loader` を使う。ここは主に旧分類経路用である。

### `dataset/dataloader_factory.py`

| シンボル | 役割 |
|---|---|
| `DataloadersInfo` (`:23`) | train loader、validation loader、class数を束ねるdataclass。 |
| `configure_dataloader` (`:35`) | dataset名からCIFAR10/ImageFolder/VideoFolder/ZeroImagesと対応transformを選ぶfactory。 |

### `dataset/dataset_pl.py`

| シンボル | 役割 |
|---|---|
| `TrainValDataModule.__init__` (`:9`) | `configure_dataloader` を呼び、loader群を保持する。 |
| `train_dataloader` (`:24`) | Lightningへtrain loaderを返す。 |
| `val_dataloader` (`:27`) | Lightningへvalidation loaderを返す。 |
| `n_classes` (`:31`) | class数を返す。 |

### `dataset/transforms.py`

| シンボル | 役割 |
|---|---|
| `TransformVideoInfo` (`:10`) | temporal/spatial size、mean/std等の動画transform設定。 |
| `TransformImageInfo` (`:19`) | image size、mean/std等の画像transform設定。 |
| `identity` (`:24`) | 入力をそのまま返すno-op。transform pipelineのdebug用。 |
| `transform_video` (`:32`) | temporal subsample、0–1化、normalize、train時random scale/crop/flip、val時fixed scale/center crop、audio除去を組む。出力videoは通常 `[C,T,H,W]`。 |
| `transform_image` (`:89`) | CHW float化、ImageNet normalize、train random crop/flip、val resize/center cropを組む。 |

### `dataset/video_folder.py`

| シンボル | 役割 |
|---|---|
| `VideoFolderInfo` (`:24`) | root、batch size、worker、clip duration等の設定。 |
| `collate_for_video` (`:36`) | PyTorchVideoのdict batchから `(video, label)` tupleを作る。 |
| `video_folder` (`:41`) | train random clip sampler、val uniform sampler、class対応確認、DataLoader構築を行う。 |
| `LimitDataset.__init__` (`:119`) | iterable datasetを2回chainし、decode failureがあっても所定epoch長を取りやすくする。 |
| `LimitDataset.__getitem__` (`:126`) | 内部iteratorから次のsampleを返す。index自体は使わない。 |
| `LimitDataset.__len__` (`:129`) | 元動画数に固定したepoch長を返す。 |

### `dataset/image_folder.py`

| シンボル | 役割 |
|---|---|
| `ImageFolderInfo` (`:11`) | root、batch、worker等の設定。 |
| `image_folder` (`:21`) | `train/` と `val/` のdirectory class構造を読み、class対応を検証してDataLoaderを作る。trainだけshuffle。 |

### `dataset/cifar10.py`

| シンボル | 役割 |
|---|---|
| `Cifar10Info` (`:9`) | root、batch、worker等の設定。 |
| `cifar10` (`:17`) | torchvision CIFAR10をdownload可能な形で作り、train/validation loaderとclass数10を返す。 |

### `dataset/zero_images.py`

| シンボル | 役割 |
|---|---|
| `ZeroImageInfo` (`:14`) | synthetic datasetの件数、画像size、class数等を持つ。 |
| `ZeroImageDataset.__init__` (`:24`) | 全0 HWC画像を返すdebug dataset設定を保存する。 |
| `ZeroImageDataset.__len__` (`:30`) | 指定dataset sizeを返す。 |
| `ZeroImageDataset.__getitem__` (`:33`) | 全0画像と `index % n_classes` labelを返し、必要ならtransformする。 |
| `zero_images` (`:41`) | train/validation用ZeroImageDatasetとDataLoaderを作る。 |

### `dataset/__init__.py`

4 dataset factory、transform、`configure_dataloader`、Lightning DataModuleを再公開する。

## 10. `setup/`, `callback/`, `logger/` — 旧分類系の構築補助

### `setup/optimizer.py`

| シンボル | 役割 |
|---|---|
| `configure_optimizer` (`:10`) | configに応じてSGDまたはAdamを作る。Streaming MoCoのAdamWはここを使わない。 |

### `setup/scheduler.py`

| シンボル | 役割 |
|---|---|
| `configure_scheduler` (`:9`) | StepLRまたはdummy schedulerを選ぶ。 |
| `dummy_scheduler` (`:33`) | learning rateを変えないfactor=1のConstantLRを返す。 |

### `setup/__init__.py`

`configure_optimizer` と `configure_scheduler` を再公開する。

### `callback/callback_pl.py`

| シンボル | 役割 |
|---|---|
| `configure_callbacks` (`:5`) | Lightningのmodel summary callbackを返す。 |

### `callback/__init__.py`

`configure_callbacks` を再公開する。

### `logger/logger.py`

| シンボル | 役割 |
|---|---|
| `configure_logger` (`:5`) | 旧PyTorch loop用Comet experimentを作り、tag/configを記録する。 |

### `logger/logger_pl.py`

| シンボル | 役割 |
|---|---|
| `configure_logger_pl` (`:8`) | Lightning用CometLoggerを構築する。 |

### `logger/comet_lineage.py`

| シンボル | 役割 |
|---|---|
| `scope_tag` (`:15`) | `production=True/False` をproduction/smoke tagへ変換し、bool以外を拒否する。 |
| `display_tag` (`:22`) | 表示用値をlowercase/hyphen形式へ正規化する。 |
| `start_experiment` (`:27`) | Comet experimentを開始または再接続する。無効・network failureはstatusとして返しlocal処理を止めない。 |
| `experiment_key` (`:48`) | experimentがあればkey、なければNoneを返す。 |
| `log_metrics` (`:52`) | experimentが存在するときだけmetricを送る。 |
| `end_experiment` (`:61`) | experimentが存在するときだけ終了処理する。 |
| `_flatten` (`:71`) | nested dictを `a.b=value` 形式へ再帰展開する。 |
| `artifact_files` (`:82`) | artifact内relative pathごとのSHA-256を計算する。 |
| `log_artifact` (`:88`) | upload直前にlocal hashとimmutable payloadを再検証し、Cometへartifact登録する。 |
| `record_comet_status` (`:124`) | mutable sidecar metadataだけにupload statusをatomic記録する。 |

### `logger/__init__.py`

旧loggerの `configure_logger` と `configure_logger_pl` を再公開する。

## 11. `utils/` — 共通補助

### `utils/artifact_io.py`

| シンボル | 役割 |
|---|---|
| `canonical_json_bytes` (`:9`) | key sort、空白なし、UTF-8、NaN禁止で決定的JSON bytesを作る。 |
| `sha256_bytes` (`:14`) | bytesのSHA-256 hex digestを返す。 |
| `sha256_file` (`:18`) | fileをchunk読込してSHA-256を返す。 |
| `fsync_directory` (`:26`) | rename後のdirectory entryまで永続化する。 |
| `write_bytes_atomic` (`:34`) | 同directoryのtemporary fileへ書いてfsyncし、`os.replace` で公開する。 |
| `write_json_atomic` (`:46`) | canonical JSONをatomic保存するwrapper。 |
| `read_json` (`:50`) | UTF-8 JSONを読む。 |

### `utils/configuration.py`

| シンボル | 役割 |
|---|---|
| `plain_config` (`:19`) | OmegaConfをresolveし、deep-copyされたplain Python dict/listへ変換する。 |
| `load_config_group` (`:29`) | repository `configs/<group>/<name>.yaml` を読み、simple nameだけ許す。 |
| `validate_path_component` (`:43`) | `/`, `..` 等を拒否し、artifact directory componentのpath traversalを防ぐ。 |

### `utils/provenance.py`

| シンボル | 役割 |
|---|---|
| `git_output` (`:34`) | 指定repositoryでgit commandを実行しtrim済みstdoutを返す。 |
| `utc_now` (`:38`) | timezone付きUTC ISO timestampを返す。 |
| `device_identity` (`:42`) | CPUまたは選択CUDA device、visible CUDA群を記録する。 |
| nested `cuda_device` (`:51`) | 1 CUDA indexのname/capability/memory等をdict化する。 |
| `collect_provenance` (`:63`) | origin、branch、commit、dirty/diff hash、loader pin、Python/library/device情報を検証・収集する。 |

### `utils/checkpoint.py` — 旧分類系checkpoint

| シンボル | 役割 |
|---|---|
| `remove_layer_prefix_from_state_dict` (`:16`) | DataParallel由来の先頭 `module.` を外す。 |
| `add_layer_prefix_to_state_dict` (`:37`) | 逆に `module.` を付ける。 |
| `save_to_checkpoint` (`:57`) | model、optimizer、scheduler、epoch/step等を旧形式checkpointへ保存する。 |
| `save_to_comet` (`:111`) | 旧checkpoint fileをComet modelとして登録する。 |
| `load_from_checkpoint` (`:127`) | CPU/single GPU/DataParallel差を吸収して旧stateを復元する。MoCo resumeとは別形式。 |

### `utils/accuracy.py`

| シンボル | 役割 |
|---|---|
| `compute_topk_accuracy` (`:6`) | logits `[B,C]` の上位k予測に正解labelが含まれるsample割合をpercentで返す。 |

### `utils/average_meter.py`

| シンボル | 役割 |
|---|---|
| `AverageMeter.__init__` (`:16`) | current value、sum、count、averageを0初期化する。 |
| `AverageMeter.update` (`:24`) | `sum += value*n`, `count += n`, `avg=sum/count` とsample-weighted更新する。 |
| `AvgMeterLossTopk.__init__` (`:48`) | lossと指定Top-kごとのAverageMeterを作る。 |
| `AvgMeterLossTopk.update` (`:64`) | 1 batchのloss/top-kをbatch size重みで更新する。 |
| `AvgMeterLossTopk.get_meters` (`:81`) | metric名→meterのmappingを返す。 |
| `AvgMeterLossTopk.get_mode_name` (`:84`) | train/validation等のprefix名を返す。 |

### `utils/mixin/average_meter_mixin.py`

| シンボル | 役割 |
|---|---|
| `GetMetricsDictMixin.get_meters` (`:5`) | subclassが実装すべきmeter mapping interface。 |
| `GetMetricsDictMixin.get_mode_name` (`:8`) | subclassが実装すべきmode名interface。 |
| `get_step_metrics_dict` (`:11`) | 各meterのcurrent valueをlogger用keyへまとめる。 |
| `get_epoch_metrics_dict` (`:28`) | 各meterのepoch averageをlogger用keyへまとめる。 |

### `utils/tqdm_loss_topk.py`

| シンボル | 役割 |
|---|---|
| `TqdmLossTopK.__init__` (`:8`) | tqdm本体とpostfix bufferを初期化する。 |
| `set_postfix_str_loss_topk` (`:12`) | step、loss、Top-kを1本の表示文字列へ組み立てる。 |
| `init_postfix_str` (`:29`) | postfix文字列を空に戻す。 |
| `add_step_to_postfix_str` (`:32`) | step表示を追加する。 |
| `add_loss_to_postfix_str` (`:35`) | loss表示を追加する。 |
| `add_topk_to_postfix_str` (`:38`) | 各Top-k精度表示を追加する。 |

### `utils/__init__.py`

AverageMeter、accuracy、旧checkpoint、tqdm表示helperを再公開する。artifact/config/provenance utilityは明示module pathからimportする設計。

### `utils/mixin/__init__.py`

`GetMetricsDictMixin` を再公開する。

## 12. `test/` — 各テストファイルと関数が確認すること

この節は「テストを実行した結果」ではなく、テストコードを静的に読んで分かる**意図された保証**である。`pytest` fixtureはテスト用データや偽物modelを準備し、`test_*` 関数が期待する契約をassertする。テスト関数内のlocal関数は、optimizer、Reader、Comet、失敗処理などを差し替えるtest doubleである。

### 12.1 設定・CLI構成

#### `test/config/conftest.py`

- `compose_config`: Hydraをtest用に初期化し、内側の `make` をfixtureとして返す。
- nested `make`: overrideと `return_hydra_config` を受け、実際のrepository configを合成する。

#### `test/config/test_composition.py`

- `test_legacy_defaults`: 旧分類入口のdefault configが期待どおり合成されるか。
- `test_group_choices`: dataset/model/optimizer等の各config groupを選び替えられるか。
- `test_representative_overrides`: 代表的なCLI overrideが最終値へ届くか。
- `test_hydra_paths_and_cwd`: Hydra output pathとworking directory policyが保たれるか。

#### `test/config/test_cli.py`

- `run_python`: subprocessでCLIを起動する共通helper。
- `test_config_print_without_training`: config表示modeが学習を始めず終了するか。
- `test_old_training_cli_is_rejected`: 廃止された旧CLI書式を明示的に拒否するか。
- `test_runtime_artifacts_and_resolved_config`: run directoryへresolved config等が作られるか。
- `stop_before_training`: monkeypatchで本学習直前に停止するlocal helper。
- `test_research_entrypoints_expose_hydra_config_without_running`: MoCo/Linear Probe entrypointが処理実行なしに合成設定を表示できるか。

#### `test/config/test_interfaces.py`

- `test_dataset_subtrees_reach_factory`: dataset別設定がDataLoader factoryへ正しく渡るか。
- `test_lightning_subtrees_preserve_factories`: Lightning経路でもmodel/data/logger/callback設定が失われないか。
- `test_plain_training_config_wiring`: 通常PyTorch入口でmodel、optimizer、resume設定が正しく配線されるか。
- `test_plain_training_intervals_and_checkpoint`: train/validation/checkpoint intervalが入口から下位関数へ届くか。
- `test_lightning_entrypoint_wiring`: device数を含むLightning Trainer構成が期待どおりか。
- `test_training_components_have_no_argparse_dependency`: 下位training componentがCLI parserに依存していないか。

#### `test/config/test_pipeline_composition.py`

- `compose_pipeline`: pipeline root configを合成するhelper。
- `test_pipeline_roots_compose_shared_science_defaults`: MoCo/manifest/feature/probeが共通科学設定を共有するか。
- `test_pipeline_roots_declare_and_resolve_mandatory_values`: 必須値が宣言され、override後に未解決値が残らないか。
- `test_pipeline_root_overrides`: root別overrideが期待値になるか。
- `test_pipeline_hydra_keeps_launch_working_directory`: launch時working directoryが変わらないか。
- `test_tracking_overrides_do_not_change_scientific_identity`: Comet等のtracking変更がscience hashを変えないか。
- `test_probe_override_is_hashed_and_noncanonical`: probe科学条件の変更がhashへ入り、canonical扱いされないか。
- `test_moco_production_match_excludes_tracking_but_includes_science`: production判定がtrackingを無視し、科学条件変更を検出するか。

### 12.2 dataset・setup・共通utility

#### `test/dataset/test_zero_images.py`

- `test_zero_images`: synthetic datasetの件数、shape、label、batch、class数が設定どおりか。

#### `test/dataset/test_video_folder.py`

- `test_video_folder`: train/val動画folderから期待shape・label・class数のDataLoaderを作れるか。

#### `test/dataset/test_cifar10.py`

このファイルはmodule-level skip設定を持ち、環境依存のCIFAR10 download/DataLoader契約を確認するテストを定義する。独立したtop-level helperはない。

#### `test/setup/conftest.py`

- `model`: optimizer/scheduler test用の小さなmodule fixture。

#### `test/setup/test_optimizer.py`

- `test_optimizer`: 設定ごとにSGD/Adamの型とhyperparameterが正しいか。

#### `test/setup/test_scheduler.py`

- `optimizer`: scheduler test用optimizer fixture。
- `test_scheduler`: StepLR/dummyの型とlearning-rate変化が期待どおりか。

#### `test/utils/test_accuracy.py`

- `test_accuracy`: 用意したlogits/labelsに対するTop-k percentが手計算値と一致するか。

#### `test/utils/test_average_meter.py`

- `test_average_meter`: current/sum/count/weighted averageの更新式が正しいか。
- `test_average_meters_loss_topk`: lossと複数Top-k meterの更新およびlogger dictが正しいか。

#### `test/utils/test_checkpoint.py`

- `DummyLogger`: Comet保存呼び出しを観測するtest double。
- `test_checkpoint_save_load`: DataParallel prefix差、model/optimizer/scheduler/counterのsave/load、Comet連携がround-tripするか。

### 12.3 model・bridge

#### `test/model/test_vit_frame_encoder.py`

- `FakeViT.__init__` / `forward`: token列を決定的に返す小型ViT test double。
- `TestViTFrameEncoder.test_cls_feature_and_frozen_backbone`: token 0だけが `[B,768]` として返り、backboneがfreezeされるか。

#### `test/model/test_vit_lora_frame_encoder.py`

- `encoder`: 実PEFT adapter付きencoder fixture。
- `test_real_peft_targets_config_and_frozen_base`: Q/VだけがLoRA対象で、設定・parameter数・Base freezeが契約どおりか。
- `test_cls_forward_shape_finite_and_gradient_mode`: train/evalそれぞれでCLS shape、finite、gradient pathが正しいか。

#### `test/model/test_masked_mean_clip_aggregator.py`

- `test_all_valid_matches_mean_and_preserves_shape_dtype_device`: 全validなら通常meanと一致し、shape/dtype/deviceを保つか。
- `test_invalid_rows_do_not_contribute_values_or_denominator`: padding値が結果へ全く影響しないか。
- `test_reverse_and_permutation_preserve_output`: 平均が順序不変であることを明示する。
- `test_invalid_shapes_and_empty_mask_raise`: shape不整合やvalid 0件を拒否するか。
- `test_non_boolean_mask_is_rejected_instead_of_integer_indexing`: int maskをboolean maskとして誤用しないか。
- `test_backward_only_reaches_valid_rows_with_correct_scaling`: gradientがvalid行へだけ `1/V` で流れるか。

#### `test/model/test_50salads_vit_bridge.py`

- `RecordingProcessor.__call__`: processorへ渡されたframeを記録して簡単なtensorを返す。
- `ValueEncoder.__call__`: 入力値から決定的featureを返す。
- `Test50SaladsViTBridge.test_padding_is_skipped_and_features_keep_frame_positions`: paddingをencoderへ渡さず、valid featureを元位置へ戻すか。

#### `test/model/test_activitynet_vit_clip_feature.py`

- `RecordingProcessor.__call__`: valid frameだけ受けたか記録するtest processor。
- `make_sample`: padding込みActivityNet風sampleを作る。
- `test_shared_bridge_skips_padding_preserves_metadata_and_aggregates`: bridgeのscatter、metadata不変、masked meanをまとめて確認する。
- `test_bridge_rejects_all_invalid_before_processor_or_encoder`: valid 0件を外部処理前に拒否するか。
- `test_bridge_rejects_invalid_processor_or_encoder_output`: processor/encoderのshapeやfinite違反を検出するか。
- `test_smoke_uses_strict_loader_closes_reader_and_runs_frozen_eval`: smokeが1 sourceをstrictに読み、readerを閉じ、encoderをeval/frozenで使うか。
- `test_smoke_rejects_wrong_provenance_before_reading_data`: provenance違反時にdatasetへ触れる前に止まるか。
- local `git_output` / `record_output`: git応答とforward出力を差し替え・記録するtest double。

#### `test/model/test_activitynet_vit_lora_one_step.py`

- `test_one_step_through_valid_scatter_and_masked_mean`: 実LoRA encoderへmasked mean経由でgradientが届き、Baseは変わらないか。
- `test_parameter_audit_rejects_contract_violation_before_optimizer`: trainable parameter契約違反をoptimizer生成前に止めるか。
- `test_one_step_rejects_invalid_gradients_or_updates`: NaN gradient、Base更新、LoRA未更新等を検出するか。
- `test_smoke_strict_loader_one_chunk_and_reader_cleanup`: 1 chunkだけ読み、どの終了経路でもReaderを閉じるか。
- `test_smoke_rejects_wrong_provenance_before_reading_data`: provenance failureがdataset accessより先か。
- local `make_optimizer`, `bad_step`, `git_output`, `before_forward`: optimizer更新やfailureを観測・注入するtest double。

#### `test/model/test_activitynet_vit_lora_moco_one_step.py`

- `asymmetric_sample`: flip/色変換を観測しやすい非対称sampleを作る。
- `test_two_views_preserve_source_and_metadata_and_skip_padding`: 2 viewがvalid frameだけを変換しmetadata/paddingを保つか。
- `test_real_peft_one_step_gradient_update_ema_and_enqueue_order`: 実PEFTでQuery更新 → Key EMA → enqueue順と各parameter変化を確認する。
- `test_initial_audit_rejects_wrong_contract`: Queue非空、Base trainable等の初期違反を拒否するか。
- `test_step_audit_rejects_invalid_gradients_updates_and_ema`: gradient/Query/Base/EMA/Queueの各破損を検出するか。
- `valid_git_output`: 正常provenance応答を返すhelper。
- `test_smoke_two_independent_sources_first_chunks_and_reader_cleanup`: 2 sourceの先頭chunkを独立に読み、必ず閉じるか。
- `test_provenance_rejected_before_dataset_read`: provenance違反をdata access前に止めるか。
- local `step`, `ema`, `append`, `loss`, `forward_hook`, `git_output`: 更新順・引数・failureを記録するtest double。

#### `test/model/test_model_factory.py`

- `test_model_factory`: model名ごとに正しいclassと出力shapeを作るか。
- `test_model_data_parallel`: 指定時にDataParallel wrapperを構成するか。
- `test_model_methods`: 共通model interfaceとdevice取得が成立するか。

#### `test/model/test_image_models.py`

- `test_image_model_output`: 各画像backboneが `[B,3,H,W]` から `[B,C]` とscalar lossを返すか。
- `test_image_model_loss_backward`: lossからparameterへ有限gradientが流れるか。

#### `test/model/test_videomodels.py`

- `test_video_model_output`: 動画backboneが `[B,3,T,H,W]` から期待logits/lossを返すか。
- `test_video_model_loss_backward`: 動画modelのloss backwardが成立するか。

### 12.4 MoCo primitive・protocol・streaming

#### `test/self_supervised/test_vit_lora_moco.py`

- `moco`: 実LoRA encoderからMoCo fixtureを作る。
- `unit_key`: indexに応じた正規化済み128次元keyを作る。
- `test_projector_counts_batched_shape_normalization_and_independent_initial_state`: projector parameter数、batch shape、L2 norm、Query/Keyが別objectだが同じ初期値か。
- `test_queue_empty_fifo_overflow_detach_and_metadata_alignment`: 空Queue、capacity overflow、detach/clone、metadata FIFO整合性。
- `test_queue_masks_same_sequence_at_any_distance_and_fails_without_negatives`: same-sequence除外が位置に依存せず、候補0件を拒否するか。
- `test_queue_rejects_nonfinite_wrong_shape_or_unnormalized_keys`: Queue input契約違反を拒否するか。
- `test_infonce_positive_zero_temperature_detached_keys_and_old_queue`: positive位置、temperature>0、Key detach、明示old negativesだけを使うか。
- `test_ema_matches_formula_changes_only_adapters_and_projector`: EMA式どおりでBaseを変えないか。

#### `test/training/test_moco_protocol.py`

- `unit_key`: protocol test用正規化keyを作る。
- `test_axes_are_independent_and_warmup_matches_active_sources`: stream/view/negativeの3軸を独立に選べ、warm-up数がstream数に合うか。
- `test_presets_defaults_and_immutability`: Stage 6A/6B presetとfrozen dataclass性。
- `test_protocol_rejects_unknown_axes`: 未知の各軸値を拒否するか。
- `test_views_transform_all_valid_rgb_frames_and_preserve_padding_and_every_metadata_field`: 全valid frameだけ指定変換し他fieldを保つか。
- `test_view_default_is_stage5_horizontal_flip_and_unknown_transform_fails`: default互換と未知変換拒否。
- `test_view_audit_rejects_transform_padding_and_metadata_changes`: view破損の種類ごとにauditが止めるか。
- `test_negative_selection_preserves_old_entries_order_and_policy_independence`: Queue順を保ちpolicyがstream/view軸へ依存しないか。
- `test_negative_selection_rejects_empty_queue`: policyごとにempty negativesを拒否するか。
- `test_strict_single_different_sequence_fails_clearly_but_all_past_includes_same_video`: strict singleではall-pastが必要になる理由を確認する。
- `test_default_capacity_retains_only_4096_most_recent_keys_with_fifo_metadata`: capacityとFIFO。
- `test_infonce_uses_only_explicit_negatives_and_detaches_all_keys`: lossが渡されたnegativeだけを使いKeyへgradientを流さないか。

#### `test/training/test_streaming_moco.py`

- `test_strict_order_has_no_prefetch_and_closes_on_early_stop`: A1,A2…順で先読みせず、早期終了時もReaderを閉じるか。
- `test_strict_reader_closes_on_every_exit`: 正常、例外、generator closeの全経路でcleanupするか。
- `test_stage6b_key_only_warmup_past_negatives_and_update_order`: 1 warm-up、all-past negative、optimizer→EMA→enqueue順。
- `test_engine_axes_are_independent`: 3軸組合せがengineで混線しないか。
- `test_shared_engine_fifo_audit_accepts_capacity_overflow`: Queueが満杯後も期待FIFOを正しく監査するか。
- `test_strict_eof_does_not_claim_unreached_target`: source枯渇を目標step達成と誤報しないか。
- `test_strict_training_exception_closes_reader`: forward/loss/enqueue等の失敗時もReaderを閉じるか。
- `test_stage6b_real_vit_peft_integration`: 実ViT+PEFTを共通engineへ接続できるか。
- local `consume`, `optimizer`, `update_key`, `compute_loss`, `append`, `loss`, `fail`: cleanup・順序・引数・failure用test double。

#### `test/training/test_moco_multistep_canary.py`

- `sources` / `reader`: 4-stream用fake source/Reader fixture。
- `SmallEncoder.__init__` / `forward`: 高速にgradient/updateを観測できる小型encoder。
- `small_moco`: 小型encoderでQuery/Key初期化を揃えたMoCo fixture。
- `records`: JSON outputを解析するhelper。
- `test_round_robin_no_prefetch_and_early_consumer_close`: A1,B1,C1,D1…順とno-prefetch/cleanup。
- `test_all_open_readers_close_on_every_exit`: 全4 Readerのcleanup。
- `test_scheduler_rejects_corrupt_order_before_yield`: frame index/order破損をyield前に拒否するか。
- `test_multistep_real_moco_semantics_with_small_encoder`: 複数stepのgradient、optimizer、EMA、negative、Queue順。
- `test_real_vit_peft_and_public_loader_integration`: 実ViT/PEFTとpublic loader interfaceの接続。
- `test_eof_never_claims_unreached_target`: EOF時の成否report。
- `test_training_exception_closes_four_readers`: 各処理位置の例外で全Reader cleanup。
- `test_audits_stop_invalid_training`: NaN gradient、Base更新、Queue破損等で停止。
- `test_cli_rejects_invalid_steps_before_provenance`: 不正step数を外部確認前に拒否。
- `test_cli_rejects_invalid_sources_before_model`: source本数/重複をmodel作成前に拒否。
- `test_cli_first_four_sources_default_steps_and_fresh_runs`: CLI既定source選択とrun間state非共有。
- local `initial`, `consume`, `corrupt`, `generate`, `optimizer`, `update_key`, `compute_loss`, `append`, `fail`, `forward`, `enqueue`, `hook`, `step`, `loss`: 意図的破損とcall order観測用test double。

#### `test/training/test_streaming_moco_cli.py`

- `cli`: dataset/model/engineをfakeへ差し替えたCLI fixture。
- `invoke`: CLI optionを組み立てて起動するhelper。
- `test_presets_select_first_sources_and_log_resolved_protocol`: presetが必要source本数と解決済み3軸を選ぶか。
- `test_axis_override_changes_only_selected_axis`: 1軸overrideが他2軸を変えないか。
- `test_all_axes_can_override_a_preset_together`: 3軸同時override。
- `test_fresh_runs_and_steps_beyond_queue_capacity`: runごとにstateがfreshでcapacity超stepも渡せるか。
- `test_invalid_options_fail_before_provenance_or_dataset`: option validation順。
- `test_invalid_source_count_or_identity_fails_before_model`: source validation順。
- `test_provenance_failure_precedes_dataset_access`: provenance gate順。
- `test_engine_failure_is_not_reported_as_pass`: 例外をPASS表示しないか。

### 12.5 full run・checkpoint・source selection

#### `test/training/test_full_streaming_moco.py`

- `make_sources`: 動画ごとのchunk数を指定したfake source列を作る。
- `videos`: fake Reader/Loader fixture。
- `small`: 小型MoCo、snapshot writer、model builderを束ねるfixture。
- `run`: full engineを共通既定値で呼ぶhelper。
- `events`: JSONL eventを読むhelper。
- `full_state`: Query/Key/projector/Queueを比較可能な形で取得する。
- `test_single_warmup_and_state_kept_across_video_boundaries`: run全体でwarm-upは1回、動画境界でmodel/Queue/optimizerをresetしないか。
- `test_one_chunk_first_video_is_only_warmup`: 最初の動画が1 chunkなら更新0で次動画へ続くか。
- `test_resume_and_snapshot_triggers`: 指定動画間隔でresume/snapshotを保存するか。
- `test_metric_window_every_100_updates`: update単位metric window集約。
- `test_video_boundary_resume_is_exact`: 中断・resume後の最終stateが連続runと一致するか。
- `test_stop_after_at_or_beyond_selected_count_finishes_at_source_exhaustion`: stop_afterのpause/final判定。
- `test_resume_rejects_mismatch_corruption_mid_video_and_final`: identity違反、破損、非boundary、final resumeを拒否。
- `test_resume_rejects_source_selection_mismatch`: selection metadata差を拒否。
- `test_dirty_full_run_is_rejected_before_creating_output`: dirty sourceを出力作成前に拒否。
- `test_decode_failure_stops_and_keeps_last_boundary_checkpoint`: decode失敗時に未完動画をcommitせず最後の境界checkpointを保つ。
- `test_missing_snapshot_is_written_on_resume`: checkpoint済みsnapshotが欠落していればresume時に補う。
- local `initial`, `snapshot`, `build`, `crash`: 小型model初期化・snapshot代替・故障注入。

#### `test/training/test_full_streaming_moco_cli.py`

- `cli`: full CLI用fake Adapter/source/model/engine fixture。内側の `make_sources` と `invoke` がsource作成とHydra override起動を担う。
- `test_fresh_run_uses_all_training_sources_in_adapter_order`: full profileが全training sourceをAdapter順で渡すか。
- `test_resume_is_explicit_and_reads_saved_experiment`: resume flagなしに暗黙resumeせず、指定時に保存Comet keyを読むか。
- `test_fresh_run_comet_experiment_labels`: production/smoke scopeをComet名/tagへ反映するか。
- `test_resume_reconnects_saved_comet_experiment`: resumeで同一experimentへ再接続するか。
- `test_scientific_override_is_explicitly_nonproduction`: canonical科学条件を変えたrunをproduction扱いしないか。
- `test_reduced_profile_selects_adapter_order_and_separates_protocol`: reduced selectionとprofile identityが正しいか。
- `test_invalid_options_stop_before_work`: 不正runtime設定を外部作業前に拒否。
- `test_wrong_source_count_stops_before_model`: inventory件数違反をmodel構築前に拒否。

#### `test/training/test_moco_checkpoint.py`

- `key`: seedから決定的な正規化keyを作る。
- `test_queue_order_and_metadata_round_trip`: resume後もQueue tensor・ID・index・FIFO順が一致するか。
- `test_rng_state_round_trip`: RNG保存→乱数消費→復元で同じ列へ戻るか。
- `test_seed_all_reproduces_python_numpy_and_torch`: 3 RNGが同じseedで再現するか。
- `test_identity_mismatch_and_unsupported_schema_are_rejected`: resume identity/schema差を拒否。
- `snapshot_metadata`: snapshot test用正規metadataを作るhelper。
- `test_snapshot_is_query_lora_only_and_reloads_exactly`: evaluation snapshotがQuery LoRAだけで、exact reloadできるか。
- `test_snapshot_load_rejects_tampering_and_config_mismatch`: file改竄・LoRA config差を拒否。
- `test_base_fingerprint_is_plain_peft_wrapper_independent`: 同じBaseならwrapper有無でfingerprintが同じか。
- `test_production_snapshot_contract_is_strict`: production metadataの各必須field違反を拒否。
- `test_real_provenance_survives_weights_only_checkpoint`: weights-only安全読込でもprovenance情報を検証できるか。

#### `test/test_activitynet_source_selection.py`

- `sources`: Adapter順のfake training/validation sources。
- `reduced`: reduced configを簡単に作るhelper。
- `select`: 既定full sourcesでselectionを呼ぶhelper。
- `test_same_inputs_reproduce_ids_hash_and_adapter_order`: 同入力でID/hashが再現し、選択後もAdapter順か。
- `test_sha256_ranking_v1_has_a_fixed_known_result`: ranking algorithmが既知結果から変わっていないか。
- `test_seed_count_or_strategy_changes_selection_identity`: 選択条件変更がidentityを変えるか。
- `test_model_seed_is_not_an_input_to_source_selection`: model RNGとselection RNGが独立か。
- `test_duplicate_count_and_metadata_mismatch_are_rejected`: 重複、件数、metadata差を拒否。
- `test_selection_record_is_required_but_excluded_from_stable_hash`: recordは必須だが、記録時刻等の可変情報をstable hashへ含めないか。
- `test_invalid_config_is_rejected`: profile/seed/count/strategy不正を拒否。
- `test_only_exact_repository_profiles_are_approved`: exact full/reducedだけに正式profile名を与えるか。

### 12.6 Manifest・feature・probe・Comet

#### `test/evaluation/anet_synthetic.py`

- `chunk_times`: fake chunkごとのtimestamp列を作る。
- `ColorEncoder.__init__` / `forward`: frameの色値から決定的featureを作る小型encoder。
- `RecordingProcessor.to_dict` / `__call__`: preprocessing identityを公開し、受取画像を単純tensor化するtest processor。

#### `test/evaluation/conftest.py`

- `videos`: sourceごとのfake chunkを返すReader fixture。
- `dataset`: temporary annotation JSONとsourceを作るfixture。
- nested `make`: database内容とtimestampからsynthetic dataset一式を生成する。

#### `test/evaluation/test_activitynet_manifest.py`

- `entry`: annotation database entryを作るhelper。
- `two_classes`: 2-class synthetic dataset fixture。
- `test_closed_interval_containment_padding_and_partial_overlap`: fully-contained chunkだけ採用しpaddingを無視、部分overlapを落とすか。
- `test_contained_requires_valid_finite_timestamps`: 空・NaN/Inf timestampを拒否。
- `test_duplicate_annotations_are_independent_and_order_is_split_source_index`: 重複annotationを別segmentとして保持し、順序を固定するか。
- `test_label_mapping_requires_same_label_set`: train/validation label集合差を拒否。
- `row`: Gate test用manifest row helper。
- `test_gate_failures`: class coverage、chunk空、mapping hash等の各Gate違反を検出するか。
- `test_split_overlap_fails`: 同一sourceのsplit overlapを拒否。
- `test_manifest_files_round_trip_and_reject_tampering`: artifact round-tripと改竄検出。

#### `test/evaluation/test_segment_features.py`

- `entry`: annotation helper。
- `setup`: synthetic dataset/manifest/processorを準備するfixture。
- `extract`: 条件をそろえて `extract_split` を呼ぶhelper。
- `test_segment_feature_is_plain_mean_of_raw_chunk_masked_means`: 数値例でmasked frame mean→chunk mean定義を確認する。
- `test_base_and_lora_conditions_share_segment_ids_and_labels`: 2条件のsample/order/labelが同じか。
- `test_artifact_round_trip_and_rejections`: 保存読込、hash、shape、condition、manifest差の拒否。
- `test_shared_feature_contract_pair_validation`: Base/LoRA共有contract hashの一致。
- `test_changed_timestamps_are_rejected_at_extraction`: manifest作成後のtimestamp変化を検出。
- `test_only_the_implemented_feature_definition_is_accepted`: 未実装feature定義を拒否。

#### `test/evaluation/test_linear_probe.py`

- `epochs`: 少数epoch用ProbeConfig helper。
- `data`: 再現可能なsynthetic feature/labelを作る。
- `test_only_classifier_is_trained_and_features_are_untouched`: Linearだけを更新し入力featureを変更しないか。
- `test_seed_controls_only_init_and_shuffle`: seedが初期化とtrain shuffleだけを制御するか。
- `test_metrics_confusion_and_macro_accuracy`: 手作り予測でTop-1、class精度、macro、confusionを確認。
- `test_mean_and_sample_std_and_aggregate`: mean、`ddof=1` std、condition集約。
- `test_result_files_round_trip_and_tamper_detection`: result artifactのhash検証。
- `test_optimizer_is_built_from_the_recorded_config`: 実際のAdamW hyperparameterと記録値が一致するか。
- local `record`: callback/optimizer生成引数を捕捉するtest double。

#### `test/evaluation/test_linear_probe_cli.py`

- `entry`: annotation entry helper。
- `cli`: synthetic ActivityNetとfake encoderを組むfixture。内側の `encoder` と `invoke` がcondition別encoderと各CLI起動を担う。
- `test_pipeline_gates_hashes_and_reuse`: manifest→feature→probe全段のGate/hash/reuse条件。
- `test_feature_definition_override_is_rejected_before_extraction`: feature定義overrideを計算前に拒否。
- `test_reduced_selection_auto_separates_manifest_features_and_results`: selection違いでartifact namespaceが分離するか。
- local `materialize`: profile/seed別artifact一式を作る。
- `test_snapshot_manifest_resolution_recomputes_and_validates_exact_selection`: snapshot記録selectionを再計算照合してからmanifestを決めるか。
- `test_lora_snapshot_selection_mismatch_is_rejected_even_for_smoke`: smokeでもsnapshot/selection不一致を許さないか。
- `comet_calls`: Comet start callを記録するfixture。
- `test_pipeline_comet_experiment_labels`: 各stageのexperiment名/tag/scopeが一貫するか。

#### `test/evaluation/test_comet_lineage.py`

- `artifact`: immutable fileとmetadataを作るfixture。
- `test_disabled_comet_records_retry_and_keeps_files`: Comet無効時もlocal artifactを残しretry状態を記録するか。
- `test_upload_failure_is_recorded_not_raised`: upload失敗が科学処理を失敗にせずstatus化されるか。
- `test_successful_upload_records_version_and_experiment`: 成功時version/keyを記録するか。
- `test_changed_local_file_blocks_upload`: hash記録後に変わったfileをuploadしないか。
- `test_mutable_metadata_cannot_be_an_artifact_payload`: 可変sidecarをimmutable payloadへ入れないか。
- `test_experiment_start_failure_returns_status`: start失敗を例外ではなくstatusで返すか。
- `test_nested_parameters_are_flattened`: nested configのdotted-key化。
- `test_scope_tag_requires_bool`: production scopeの型を厳密にする。
- `test_display_tag_hyphenates`: tag文字列の正規化。

### 12.7 pipeline launcher

#### `test/test_full_pipeline_launcher.py`

- `test_launcher_keeps_full_default_and_propagates_one_profile_to_every_stage`: shell launcherのdefaultがfullで、1つのsource profile/selection設定を全stageへ渡すか。
- `test_launcher_validates_exact_snapshot_selection_before_manifest_stage`: snapshotと現在selectionの照合がmanifest buildより先に置かれているか。

## 13. 人間が確認するときの推奨順序

### 最短コース: 研究の1 updateを理解する

1. `integration/sequential_vit.py` — paddingを除き `[T,768]` を作る
2. `model/aggregators/masked_mean_clip_aggregator.py` — `[T,768] → [768]`
3. `integration/sequential_moco.py` — Query/Key viewを作る
4. `self_supervised/moco/vit_lora_moco.py` — projector、InfoNCE、EMA、Queue
5. `training/streaming_moco.py` — 更新順序

### full runまで理解する

6. `integration/sequential_stream.py` — 時系列sample供給
7. `training/streaming_moco_full.py` — 全動画loop
8. `training/moco_checkpoint.py` — resume/snapshot
9. `scripts/moco/train_full_streaming_moco.py` — CLIからの組立

### 評価まで理解する

10. `evaluation/activitynet_manifest.py`
11. `evaluation/segment_features.py`
12. `evaluation/linear_probe.py`
13. `scripts/linear_probe/*.py`

## 14. よく混同しやすいファイル

| 混同しやすい組 | 違い |
|---|---|
| `vit_frame_encoder.py` vs `vision_transformer.py` | 前者は研究用frozen CLS抽出器。後者は旧教師あり分類器。 |
| `moco_checkpoint.py` vs `utils/checkpoint.py` | 前者はMoCoのQueue/RNG/LoRAを含む厳密resume。後者は旧分類系。 |
| `streaming_moco.py` vs `streaming_moco_full.py` | 前者は1 updateと短い共通engineの部品、後者は動画全体・resume・snapshotを統括。 |
| `moco_canary.py` vs `scripts/smoke/*.py` | 前者は再利用可能な短時間engine、後者は人が起動するCLI。 |
| `evaluation/linear_probe.py` vs `scripts/linear_probe/run_probe.py` | 前者は学習・評価計算、後者はartifact/設定/Cometを含む実行管理。 |
| resume checkpoint vs evaluation snapshot | resumeは学習全state。snapshotはLinear Probe用Query LoRAだけ。 |
| Base/LoRA feature vs MoCo projector | Linear Probeはprojector後128次元ではなく、projector前CLS集約768次元を使う。 |

## 15. 重要な実装上の結論

1. 現在の動画表現は **frame-wise ViT + masked mean** で、frame順序に対して不変である。
2. 学習対象は **Query Q/V LoRA + Query projector**。Base ViTとKey branchはoptimizerでは更新しない。
3. KeyはEMA、Queueはdetached FIFO、lossはold Queueだけをnegativeに使う。
4. full runは動画境界でもmodel/optimizer/Queueを保持し、resumeだけ動画境界で保存する。
5. Linear ProbeはBase/LoRAの同一manifest・同一768次元feature定義を比較する。
6. これらはコード構造として観測した事実であり、時間情報・動作情報を実際に獲得した証拠ではない。

## 16. 根拠と未確認事項

### Observed

- 全Pythonファイル135個を列挙し、実装/CLI/補助94個とtest 41個へ分類した。
- class/function定義、import再公開、主要呼び出し関係を `dev@65dedb4e` から静的確認した。
- branchは `dev`、`origin/dev` と同commit、作業ツリーはcleanだった。

### Inferred

- Q/V LoRA、GBR+flip、all-past negative、各hyperparameterがこの研究課題に最適であることは、コードからは証明できない。
- private helperやtest doubleの目的は呼び出し箇所とassertから推定した部分を含む。

### Unverified

- 現commitでのtest PASS、実データfile、decoder、GPU実行時挙動。
- loss振動・gradient急増の原因。
- Linear Probeの数値結果と時間・動作情報の獲得。
- 外部 `sequential_loader` 内部実装。ここではこのrepositoryが要求するinterfaceだけを扱った。
