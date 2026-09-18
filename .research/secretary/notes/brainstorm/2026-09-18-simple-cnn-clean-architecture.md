---
date: 2026-09-18
project: sequential-video-lora-analysis
source_todo: null
topic: simple_cnnを基盤にしたViT-sequential-LoRA-MoCoのクリーンな実装構造
status: exploratory
tags: [brainstorm, research, architecture, simple-cnn, vit, sequential-loader, lora, moco]
---

# simple_cnnを基盤にしたViT-sequential-LoRA-MoCoのクリーンな実装構造

## 出発点

実験frameworkについて、`tamaki-lab/simple_cnn_training` の単純なmodel/data/trainer構造を基盤にし、
Saeki / Hayashi repositoryから必要部分だけを参照・追加する方が、研究コード全体を理解しやすく保てるのではないかという案を検討した。

現在の実装順の有力候補は次。

```text
ViT single-image smoke
 -> sequential_loader video forward smoke
 -> clip feature contract
 -> LoRA update smoke
 -> MoCo mechanics
 -> LoRA + MoCo + sequential video integration
```

## 確認済み事実

### simple_cnn_training

- `model/model_factory.py` でmodel選択を集約。
- `model/simple_lightning_model.py` にLightning training loop。
- `dataset/dataloader_factory.py` にdataset選択を集約。
- ViT実装は `ViTForImageClassification` を利用する画像分類model。
- VideoFolderはPyTorchVideoベース。
- 構造が小さく、entry point / model / dataset / optimizer / loggerが追いやすい。

### Saeki

- bare `ViTModel` を使う `ViTFrameEncoder` があり、BCHW -> [B,D] を明示。
- video側はBCTHWをframeへ分解し、frame featureとtemporal logicを分離。
- GRU/LSTM/sliding transformer等は別責務。
- config/test/runnerもsimple_cnn系から拡張されている。

### Hayashi

- external `sequential_loader` のpublic APIだけをconsumerとして利用。
- `SequentialSample` の frames / frame_indices / timestamps / valid_mask / is_first / is_last を明示的に扱う。
- loader内部moduleへ依存しない。
- EOF / padding / frame continuityをsmokeで検証。

### 現在の石川repo

- simple_cnn系のmodel/dataset/Lightning構造を残している。
- 一方、旧MeMViT / classification / legacy sequential datasetの責務が多く残っている。
- current model factoryでは `vit_b` が実質MeMViTへ読み替えられる状態であり、新しいViT-MoCo pathは分離して作る必要がある。

## 現時点の方向性

simple_cnnを「骨格」として使う案は有力。

ただし、Saeki/Hayashiのコードを大きな塊でコピーするのではなく、次の責務単位で再構成する。

1. Data source / sequential contract
2. Frame encoder
3. Clip encoder / aggregation
4. Parameter-efficient adapter (LoRA)
5. Self-supervised objective (MoCo)
6. Training orchestration
7. Experiment configuration / smoke

各layerは下位layerだけに依存し、dataset固有情報をmodelへ漏らさない。

## 推奨依存関係

```text
main_moco.py / runner
        |
        v
MoCoLightningModule
        |
        v
MoCoModel
        |
        v
VideoEncoder
  |           |
  v           v
FrameEncoder  ClipAggregator
  |
  v
ViTModel
  ^
  |
LoRA injection

Data path:
sequential_loader
   |
   v
SequentialLoaderAdapter
   |
   v
SequentialBatch
   |
   v
preprocess
   |
   +---------------------> VideoEncoder
```

重要なのは、`sequential_loader` をViT内部から呼ばない、LoRAをDataLoaderから意識しない、
MoCoをViT classへ直接埋め込まないこと。

## 推奨directory案

現行simple_cnn系の理解しやすさを維持するため、大規模なsrc-layout移行は初期段階では行わない候補。

```text
.
├── main_pl.py                    # 既存classification pathは維持
├── main_moco.py                  # 新しいSSL/MoCo entry point
│
├── dataset/
│   ├── dataloader_factory.py     # dataset選択だけ
│   ├── transforms.py
│   └── sequential/
│       ├── __init__.py
│       ├── loader_adapter.py     # Hayashiを参考: external sequential_loader public APIのみ
│       ├── batch.py              # modelへ渡す共通batch contract
│       ├── collate.py
│       └── transforms.py         # uint8 frame -> ViT preprocess
│
├── model/
│   ├── model_factory.py
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── vit_frame_encoder.py  # Saekiを参考: BCHW -> [B,D]
│   │   └── video_encoder.py      # [B,T,C,H,W] -> [B,T,D] -> clip feature
│   │
│   ├── aggregation/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── mean_pool.py          # 初期non-temporal baseline
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   └── lora.py               # LoRA注入・freeze・parameter auditのみ
│   │
│   ├── ssl/
│   │   ├── __init__.py
│   │   ├── moco.py               # query/key encoder, EMA
│   │   ├── projector.py
│   │   ├── queue.py              # queue版を採用する場合
│   │   └── losses.py             # InfoNCE等
│   │
│   ├── moco_lightning_model.py   # training orchestrationのみ
│   └── simple_lightning_model.py # 既存classification path
│
├── config/
│   ├── vit.py
│   ├── sequential.py
│   ├── lora.py
│   └── moco.py
│
├── scripts/
│   ├── smoke_vit.py
│   ├── smoke_sequential_loader.py
│   ├── smoke_sequential_vit.py
│   ├── smoke_lora.py
│   └── smoke_moco.py
│
└── test/
    ├── dataset/
    │   └── test_sequential_loader_adapter.py
    ├── model/
    │   ├── test_vit_frame_encoder.py
    │   ├── test_video_encoder.py
    │   └── test_lora.py
    └── ssl/
        └── test_moco.py
```

これは候補構造であり、specではない。

## 各moduleの責務

### dataset/sequential/loader_adapter.py

Hayashiの考え方を採る。

担当:
- external `sequential_loader` のpublic API利用。
- SequentialSampleを研究側の共通batch contractへ変換。
- frame index / timestamps / valid mask / sequence境界の保持。

担当しない:
- ViT normalization。
- LoRA。
- MoCo augmentation。
- model forward。

外部loaderの内部実装をcopyしない。

### dataset/sequential/batch.py

research code内部の安定したinterfaceを定義する候補。

例:

```text
frames:       [B,T,C,H,W]
valid_mask:   [B,T]
frame_indices:[B,T]
timestamps:   [B,T]
sequence_id
is_first
is_last
```

外部 `SequentialSample` の変更をmodel全体へ直接波及させないための境界。

### model/encoders/vit_frame_encoder.py

Saekiの責務分離を採る。

担当:
- pretrained bare ViTのload。
- BCHW -> [B,D]。
- feature source (CLS等)。
- freeze設定。

担当しない:
- 動画順序。
- clip aggregation。
- LoRA学習ロジック。
- MoCo。
- classifier。

### model/encoders/video_encoder.py

動画とframe encoderの橋渡し。

```text
[B,T,C,H,W]
 -> [B*T,C,H,W]
 -> FrameEncoder
 -> [B*T,D]
 -> [B,T,D]
 -> ClipAggregator
 -> [B,D]
```

Python loopより、初期候補としてB*Tへflattenして一括forwardする方が単純かつ効率的。

`valid_mask` をaggregationへ渡す。

### model/aggregation/

初期baselineは `MaskedMeanPool`。

これはorder-invariantであり、「時間情報を使わないvideo baseline」として意図的に残す。

将来:
- temporal transformer
- GRU/LSTM
- future-prediction module

等を追加してもVideoEncoderのinterfaceを変えない。

Saekiのtemporal headは初期実装では持ち込まない。

### model/adapters/lora.py

LoRAをViT classへ直接書き込まない。

担当:
- target module指定。
- LoRA注入。
- base freeze。
- trainable parameter audit。
- LoRA on/off。

これにより、将来CLIP-ViTへframe encoderを差し替えてもadapter layerを再利用しやすくする。

### model/ssl/moco.py

MoCoをVideoEncoderへ埋め込まない。

担当:
- query encoder。
- key/momentum encoder。
- EMA。
- projector/predictor。
- queue（採用variantなら）。
- q/k生成。

`VideoEncoder` は「clip -> feature」だけを知り、MoCoを知らない。

### moco_lightning_model.py

既存 `SimpleLightningModel` をMoCo対応へ肥大化させない。

新しいmoduleは:
- batch受取。
- two-view augmentation呼出。
- MoCo forward。
- loss logging。
- optimizer。
- checkpoint。

だけを担当。

classification top1/top5の前提を持ち込まない。

## config方針

simple_cnnの単純さを保つため、最初からSaekiのHydra全体を移植する必要はない。

ただし、LoRA/MoCoでargument数が増えるため、1つの巨大なargparse Namespaceへ全設定を追加するのも避けたい。

候補:
- `ViTConfig`
- `SequentialConfig`
- `LoRAConfig`
- `MoCoConfig`

をdataclassとして分離し、entry pointで組み立てる。

Hydra導入は複数baseline/ablationを大量に回す段階で再評価できる。

## 追加順

### Phase A: framework cleanupを最小化

既存classification / MeMViT codeを大規模削除しない。
新しいpathを横に追加する。

### Phase B: ViT frame encoder

```text
image -> ViTFrameEncoder -> [B,D]
```

### Phase C: external sequential_loader bridge

```text
SequentialSample
 -> SequentialBatch
 -> preprocess
 -> VideoEncoder
 -> [B,T,D]
```

この段階ではaggregation前後をログで確認する。

### Phase D: non-temporal clip baseline

MaskedMeanPoolを追加。

### Phase E: LoRA

VideoEncoder interfaceを変えず、FrameEncoder内部へadapterを注入。

### Phase F: MoCo

MoCoModelとMoCoLightningModuleを追加。

### Phase G: temporal extension

新しいAggregator/Objectiveとして追加し、mean baselineを壊さない。

## cleanに保つための禁止事項候補

- `model_factory.py` の中へtraining logicを書かない。
- `SimpleLightningModel` にclassification / MAE / MoCoの巨大なif分岐を追加しない。
- dataset classからmodelをimportしない。
- modelから `sequential_loader` をimportしない。
- ViTFrameEncoderへGRU / queue / EMAを追加しない。
- LoRA target module名を複数fileに重複させない。
- `sequential_loader` の内部moduleをimportしない。
- Hayashi / Saeki repoのdataset/modelを丸ごとcopyしない。
- 旧MeMViT pathを新しいViT pathのために無理に再利用しない。

## smoke / testをarchitecture境界に対応させる

1. `test_vit_frame_encoder`
   - BCHW -> [B,D]
2. `test_sequential_loader_adapter`
   - order / valid_mask / EOF
3. `test_video_encoder`
   - B,T,C,H,W -> B,T,D -> B,D
4. `test_lora`
   - base unchanged / LoRA changed
5. `test_moco`
   - query gradient / key no-grad / EMA / finite loss
6. integration smoke
   - sequential_loader -> ViT -> LoRA -> MoCo

各stageのfailure原因を責務単位で切り分ける。

## 代替案との比較

### Saeki repoを丸ごとbaseにする

実装量は減る部分もあるが、Hydra、独自50Salads dataset、temporal head、streaming evaluation等、
現在不要な責務まで持ち込みやすい。

### Hayashi repoをbaseにする

sequential_loaderは綺麗だがtraining frameworkを新設する割合が大きい。

### simple_cnn骨格 + 必要部分追加

新規module数は増えるが、各責務を理解しながら増やせる。
今後のCLIP-ViT、LoRA、MoCo、temporal objectiveの差し替えにも向く。

## 現時点の収束

有力候補:

- current Ishikawa repo / simple_cnn系の骨格を維持。
- Hayashiからはexternal sequential_loaderのconsumer boundaryだけ採る。
- Saekiからはbare ViT frame encoderとframe/video責務分離だけ採る。
- LoRAはadapter layer。
- MoCoはSSL layer。
- temporal moduleはaggregation/objective layerとして後付け。
- 既存classification / MeMViT pathは初期段階では壊さず、新しいpathを並行追加する。

これにより「データ」「画像encoder」「動画化」「追加学習parameter」「自己教師ありobjective」を独立して差し替えられる。

## 未解決事項

- configをdataclass + argparseのまま行くかHydraを早期導入するか。
- new entry point名を `main_moco.py` にするか汎用 `main_ssl.py` にするか。
- `SequentialBatch` をdataclassにするかdictにするか。
- clip aggregationの初期仕様。
- MoCo variant。
- LoRA library / target module。
- current legacy codeを将来どの段階で整理・削除するか。

このメモは探索記録であり、specまたは実装許可ではない。


## 2026-09-18 15:32 追記: まずsimple_cnnと同等状態へ揃える案

ユーザーから、現在の石川repoを一度 `tamaki-lab/simple_cnn_training@main` と同等の状態へ戻してから、新しい研究実装を始める案が提示された。

現行repoを確認すると、README、pyproject、requirements、logger/callback等はsimple_cnn系と共通部分が多い一方で、MeMViT、旧sequential dataset、追加config、分類用state/debug処理等が上乗せされている。

現研究の新規pathは
`ViT -> sequential_loader -> LoRA -> MoCo`
であり、旧MeMViT/classification固有処理への依存を必要としない。

そのため、実装開始前にsimple_cnn相当のbaselineへ揃える方針は有力。

ただし、Git historyを消すようなforce resetではなく、現在状態をcommit/branch等で参照可能に残し、
「simple_cnn baselineを復元した1 commit」を明示的な研究実装の起点にする方が安全。

推奨イメージ:

```text
current legacy state
   |
   | preserve in history / branch
   v
simple_cnn baseline sync commit
   |
   v
Stage 1: ViT feature extractor
   |
   v
Stage 2: sequential_loader bridge
   |
   v
Stage 3: clip feature
   |
   v
Stage 4: LoRA
   |
   v
Stage 5: MoCo
```

baseline同期後は、まずsimple_cnn由来の既存smoke/testが通ることをGateとする。
その後、新しい機能は各Stageごとに別commitで追加する。

この方式の利点:
- どの変更が研究固有かGit diffで明確になる。
- 旧MeMViT由来の副作用を切り離せる。
- Hayashi/Saekiから持ち込む要素を必要最小限にできる。
- 失敗時にsimple_cnn baselineまで容易に戻せる。
- 後から「simple_cnnから何を追加したか」を説明しやすい。

注意:
- READMEやproject固有のGit metadataまで機械的に完全一致させる必要はない。
- 研究repoとして必要なrepository name、READMEの研究説明、gitignore等は必要に応じて保持する。
- 「コード基盤をsimple_cnnと同等にする」と「repositoryそのものをsimple_cnnのcloneにする」は分けて考える。


## 2026-09-18 15:32 追記: 石川repoをsimple_cnn baselineへ揃える具体差分

比較対象:
- Ishikawa: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@main`
  - commit: `e0deb093694d367ed9b02065e6d4cd38802093d6`
- simple_cnn: `tamaki-lab/simple_cnn_training@main`
  - commit: `e541dd98f825eb58c193ccd55cffa858392b89fe`

recursive tree比較結果:
- Ishikawa files: 72
- simple_cnn files: 72
- 内容同一: 25
- 同じpathだが内容変更: 17
- Ishikawa側のみ: 30
- simple_cnn側のみ: 30

### そのまま保持する25ファイル

両repoでblob SHAまで一致している。

```text
.flake8
.mypy.ini
.pep8
.pylintrc
.pytest.ini
.vscode/extensions.json
.vscode/tasks.json
README.md
args/__init__.py
callback/__init__.py
callback/callback_pl.py
logger/__init__.py
logger/logger.py
logger/logger_pl.py
model/base_model.py
pyproject.toml
requirements.pytorch.txt
requirements.txt
setup/__init__.py
setup/scheduler.py
utils/average_meter.py
utils/checkpoint.py
utils/mixin/__init__.py
utils/mixin/average_meter_mixin.py
utils/tqdm_loss_topk.py
```

### Ishikawa側のみで、baseline化時に削除する候補30ファイル

すべて旧MeMViT / old sequential dataset / Orthogonal Gradient / old run/debug系に由来し、現在のsimple_cnn baselineには存在しない。

```text
configs/MeMViT_16_50Salads_frame.yaml
configs/MeMViT_16_K400.yaml
configs/MeMViT_16_K400_multi_classes.yaml
configs/parser.py
dataset/sequential/base_sequential_video_dataset.py
dataset/sequential/epic_kitchens/epic_kitchens_sequential_data_folder.py
dataset/sequential/epic_kitchens/epic_kitchens_sequential_dataset.py
dataset/sequential/salads50/__init__.py
dataset/sequential/salads50/salads50_sequential_data_folder.py
dataset/sequential/salads50/salads50_sequential_dataset.py
dataset/sequential/use_50Salads_webdataset.py
dataset/sequential_video_dataset.py
dataset/sequential_video_folder.py
model/memvit/__init__.py
model/memvit/attention.py
model/memvit/build.py
model/memvit/common.py
model/memvit/config/__init__.py
model/memvit/config/custom_config.py
model/memvit/config/defaults.py
model/memvit/distributed.py
model/memvit/logging.py
model/memvit/memvit_model.py
model/memvit/stem_helper.py
model/memvit/utils.py
run_epic_frame_1gpu_memvit.sh
run_sequential_video_folder_1gpu.sh
setup/orthogonalAdamW.py
test_dataloader.py
utils/validation_evaluator.py
```

これらはGit historyからは消さず、baseline commitでworking treeから削除する。

### simple_cnn側から復元する30ファイル

現在Ishikawa側に存在せず、simple_cnn baselineの標準機能・testsを構成する。

```text
dataset/cifar10.py
dataset/image_folder.py
dataset/video_folder.py
dataset/zero_images.py
main.py
model/abn/__init__.py
model/abn/attention_branch_network.py
model/dummy_models/__init__.py
model/dummy_models/zero_outout_model.py
model/resnet/__init__.py
model/resnet/resnet18.py
model/resnet/resnet50.py
model/vit/__init__.py
model/vit/vision_transformer.py
model/x3d/__init__.py
model/x3d/x3d.py
test/dataset/test_cifar10.py
test/dataset/test_video_folder.py
test/dataset/test_zero_images.py
test/model/test_image_models.py
test/model/test_model_factory.py
test/model/test_videomodels.py
test/setup/conftest.py
test/setup/test_optimizer.py
test/setup/test_scheduler.py
test/utils/test_accuracy.py
test/utils/test_average_meter.py
test/utils/test_checkpoint.py
train.py
val.py
```

### simple_cnn版へ戻す15ファイル

同pathだがIshikawa側で旧研究向けに拡張されているため、baselineではsimple_cnn版へ戻す候補。

```text
.vscode/launch.json
.vscode/settings.json
args/arg_parse.py
dataset/__init__.py
dataset/dataloader_factory.py
dataset/dataset_pl.py
dataset/transforms.py
main_pl.py
model/__init__.py
model/model_config.py
model/model_factory.py
model/simple_lightning_model.py
setup/optimizer.py
utils/__init__.py
utils/accuracy.py
```

主な差分:
- `args/arg_parse.py`: sequential / EPIC / 50Salads / MeMViT / Orthogonal optimizer等の引数が追加。
- `dataset/*`: old sequential datasetへ拡張。
- `main_pl.py`: MeMViT configと追加device/state処理。
- `model/*`: model factoryが実質MeMViT中心、LightningModuleにframewise/state/debug処理を追加。
- `setup/optimizer.py`: AdamW / OrthogonalAdamW等を追加。
- `utils/*`: framewise metric / validation evaluator追加。
- `.vscode/settings.json`: 旧 `2026_04_ishikawa_simple-MeMViT/.venv-memvit` へのhard-coded interpreter pathが残っている。

### repo固有として機械的にsimple版へ戻さない2ファイル

#### `.comet.config`

simple_cnn側は `workspace=tttamaki`, `project_name=test-20220222`。
Ishikawa側は `workspace=haruto2919`, `project_name=2026-04-ishikawa-simple-memvit`。

simple側へ戻すのではなく、研究repo固有設定としてcurrent project名へ更新する候補。
少なくとも旧 `2026-04-ishikawa-simple-memvit` 名は現状と不整合。

#### `.gitignore`

Ishikawa側だけに
- `.codex/`
- `test/`
- `batch_debug.csv`
- `data/`

が追加されている。

baseline testsを復元するため `test/` ignoreは削除すべき。
一方、`.codex/`, `batch_debug.csv`, `data/` はlocal/generated artifact除外として保持候補。

したがってsimple_cnn版を丸ごとcopyするより、
`simple_cnn .gitignore + 必要なIshikawa固有ignore`
へ整理するのが安全。

## baseline化の推奨操作単位

1. 現在のIshikawa main状態をGit historyで参照可能なことを確認。
2. 上記30 Ishikawa-only filesを削除。
3. 上記30 simple-only filesをsimple_cnn `e541dd98...` から復元。
4. 上記15 modified filesをsimple_cnn版へ戻す。
5. `.comet.config` をcurrent project用に整理。
6. `.gitignore` はsimple baselineを基準に `test/` ignoreを外し、必要なlocal artifact ignoreだけ残す。
7. tests / import / Lightning smokeを実行。
8. 「simple_cnn baseline sync」として1 commitにまとめる。
9. 以降のViT / sequential_loader / LoRA / MoCoは別commit・別作業単位で追加。

この追記は探索記録であり、実装許可やspecではない。
