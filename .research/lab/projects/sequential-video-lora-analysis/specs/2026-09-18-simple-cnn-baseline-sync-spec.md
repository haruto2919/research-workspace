---
project: sequential-video-lora-analysis
spec_type: implementation
status: draft
title: simple_cnn baseline同期
created: 2026-09-18
last_updated: 2026-09-18
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
workspace_base_commit: 162755bf97ddf789226665bb7be8677e028f0436
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_base_branch: main
implementation_base_commit: e0deb093694d367ed9b02065e6d4cd38802093d6
implementation_work_branch: chore/simple-cnn-baseline-sync
reference_repository: tamaki-lab/simple_cnn_training
reference_branch: main
reference_commit: e541dd98f825eb58c193ccd55cffa858392b89fe
---

# simple_cnn baseline同期 spec

> **Status: draft**
>
> 本specは、現在の石川research code repositoryを、
> `tamaki-lab/simple_cnn_training@e541dd98f825eb58c193ccd55cffa858392b89fe`
> を基準とするclean baselineへ揃え、その後の
> `ViT -> sequential_loader -> LoRA -> MoCo`
> 実装を差分として追跡しやすくするための実装契約である。
>
> 本specはコード実装の許可ではない。ユーザー承認後にapprovedへ変更し、
> implementationは専用branchで行う。

## 1. 目的

現在の `tamaki-lab/2026_09_ishikawa_sequential-video-lora` には、
simple_cnn由来のtraining frameworkに加えて、旧研究方針で用いたMeMViT、
legacy sequential dataset、50Salads / EPIC-Kitchens固有処理、
OrthogonalAdamW、frame-wise state/debug処理等が混在している。

現在の研究方針では、まずsimple_cnn相当の小さいtraining frameworkを基準点とし、
その上へ研究固有機能を次の順で追加する。

```text
simple_cnn baseline
  -> ImageNet-pretrained ViT
  -> external sequential_loader
  -> clip representation
  -> LoRA
  -> MoCo
```

本specの目的は、研究固有の新規実装を始める前に、
**現在repositoryをsimple_cnn由来のclean baselineへ同期し、
「ここから何を追加したか」をGit diffで明確にできる0地点を作ること**である。

本specではViTの新規feature extractor、external sequential_loader、LoRA、MoCoはまだ実装しない。

## 2. Authorityと基準revision

### 2.1 Authority

本specの判断根拠は、優先順に次のとおりとする。

1. 2026-09-18のユーザー指示
   - まずsimple_cnnと同じ状態に近づけてから研究実装を開始する。
   - baseline同期作業自体をspec化する。
   - branchを適切に分離する。
2. 2026-09-17 MTG
   - 最終方式を先に固定せず、まず再現可能な開発loopを成立させる。
   - 既存frameworkは必要部分だけ利用し、品質不明な過去コードをそのまま継承しない。
   - 作業単位でbranch / commitを整理する。
3. 2026-09-18 brainstorm
   - `.research/secretary/notes/brainstorm/2026-09-18-simple-cnn-clean-architecture.md`
   - current Ishikawa repoとsimple_cnnのrecursive tree差分を確認済み。
4. 現在のGitHub上のimplementation repository。
5. reference repository `tamaki-lab/simple_cnn_training`。

### 2.2 基準revision

| 役割 | repository | branch | commit |
|---|---|---|---|
| Research Workspace | `haruto2919/research-workspace` | `main` | `162755bf97ddf789226665bb7be8677e028f0436` |
| 実装対象 | `tamaki-lab/2026_09_ishikawa_sequential-video-lora` | `main` | `e0deb093694d367ed9b02065e6d4cd38802093d6` |
| baseline参照元 | `tamaki-lab/simple_cnn_training` | `main` | `e541dd98f825eb58c193ccd55cffa858392b89fe` |

本specの差分分類は上記commit同士の比較に基づく。

実装開始時にremote `main` が進んでいても、
本specのbaseline sourceは上記commitへ固定する。
基準commit以降の変更を黙って取り込まない。

## 3. Git branch / worktree contract

### 3.1 作業branch

implementationは必ず次のbranchで行う。

```text
chore/simple-cnn-baseline-sync
```

branch種別を `chore/` とする理由は、本作業が新機能追加ではなく、
既存research repositoryをclean baselineへ同期する構造整理だからである。

### 3.2 branchの起点

branchは次のcommitから作る。

```text
tamaki-lab/2026_09_ishikawa_sequential-video-lora
main@e0deb093694d367ed9b02065e6d4cd38802093d6
```

次を禁止する。

- `mae` branchから作る。
- `mae` branchをmergeする。
- `main`上で直接作業する。
- force reset / history rewriteで過去実装を消す。
- 過去branchを削除する。

`mae` branchは過去のMAE実装Evidenceとして保持し、本specでは一切変更しない。

### 3.3 local dirty state

実装開始時のlocal working treeがdirtyである場合、
未commit変更を自動削除・stash・上書きしない。

clean working treeまたは新しいworktreeで
`chore/simple-cnn-baseline-sync` を用意できない場合は停止し、状況を報告する。

### 3.4 commit

必須検証が通った後、baseline同期を1つの論理commitとして記録する。

推奨commit message:

```text
chore: restore simple_cnn baseline
```

push、Pull Request、mainへのmergeは本specの実装許可から自動推定しない。
必要な場合は別途ユーザー指示を得る。

## 4. 現行差分の確認結果

基準commit同士のrecursive tree比較結果は次のとおり。

| 区分 | 件数 |
|---|---:|
| Ishikawa files | 72 |
| simple_cnn files | 72 |
| 同一内容 | 25 |
| 同pathだが内容変更 | 17 |
| Ishikawa側のみ | 30 |
| simple_cnn側のみ | 30 |

本specでは、この差分を
「削除」「復元」「simple_cnn版へ戻す」「repository固有として保持」
へ分類する。

## 5. 変更scope

### 5.1 Ishikawa側のみ: 削除する30ファイル

次はsimple_cnn baselineに存在せず、
旧MeMViT / legacy sequential / old dataset / Orthogonal Gradient / debug経路由来であるため、
working treeから削除する。

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

削除はGit historyを書き換える意味ではない。
これらは過去commit / branchから参照可能な状態を維持する。

### 5.2 simple_cnn側から復元する30ファイル

次はreference commitに存在し、
現在Ishikawa mainには存在しないsimple_cnn標準構成である。
内容はreference commitと一致する状態へ復元する。

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

### 5.3 simple_cnn版へ戻す15ファイル

次は同pathだが、現在Ishikawa側で旧研究向け変更が入っている。
本specではreference commitの内容へ戻す。

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

これにより、少なくとも次の旧研究固有責務をbaselineから外す。

- MeMViT専用model factory経路
- 50Salads / EPIC-Kitchens legacy sequential dataset経路
- frame-wise state / debug / validation evaluator処理
- OrthogonalAdamWを含む旧optimizer拡張
- 旧MeMViT config解析
- 旧local venvへのhard-coded VS Code interpreter path

### 5.4 内容をそのまま保持する25ファイル

次は両repositoryでblob SHAまで一致しているため変更しない。

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

### 5.5 repository固有としてsimple_cnnへ機械的に戻さないもの

#### `.comet.config`

simple_cnnのworkspace / test projectへ戻さない。

次の方針にする。

```text
workspace = haruto2919
project_name = 2026-09-ishikawa-sequential-video-lora
```

API keyはrepository fileへ追加しない。

#### `.gitignore`

simple_cnn版を基準にするが、
Ishikawa repository固有のlocal/generated artifact除外として次は保持してよい。

```text
.codex/
batch_debug.csv
data/
```

一方、`test/` ignoreは削除する。
simple_cnn標準testをGit管理対象へ戻すためである。

## 6. 実装方法

### 6.1 file-level sync

reference repositoryのGit historyをcurrent repositoryへmergeしない。

`simple_cnn_training@e541dd98...` の各対象file内容をreferenceとして、
現在repository側でfile-levelに同期する。

目的は「simple_cnn repositoryのGit lineageを取り込む」ことではなく、
「simple_cnn baseline相当のcode stateを作る」ことである。

### 6.2 過剰なcleanupをしない

本specに列挙していないfile、dependency version、formatter設定、
repository historyは不要に変更しない。

simple_cnn側に存在するという理由だけで、
現在要求と無関係な新しい抽象化やmodernizationを追加しない。

### 6.3 新しい研究機能を混ぜない

baseline同期commitには次を入れない。

- 新しいViTFrameEncoder
- external `sequential_loader`
- new video encoder / clip aggregation
- LoRA
- MoCo
- CLIP
- Hydra
- dataclass config再設計
- temporal model
- new experiment launcher

これらはbaseline同期完了後の別spec / 別branch・commitで扱う。

## 7. Breaking changeと互換性

本specは、現在mainに存在する旧MeMViT / legacy sequential経路を
work branchから意図的に除去するため、**現行mainに対してbreaking cleanup**である。

ただし次を維持する。

- Git historyから旧実装を参照できる。
- `mae` branchを変更・削除しない。
- main historyを書き換えない。
- simple_cnn由来の基本classification frameworkを復元する。

本specの目的は旧MeMViT baselineとのruntime互換を維持することではない。
旧実装を新frameworkへcompatibility shimで残すこともしない。

## 8. 検証

### 8.1 構造検証

必須。

1. 5.1の30ファイルがworking treeから存在しない。
2. 5.2の30ファイルが存在する。
3. 5.2 / 5.3の対象file内容が
   `simple_cnn_training@e541dd98...` と一致する。
4. 5.4の25ファイルが不要に変更されていない。
5. `.gitignore` が `test/` をignoreしていない。
6. `.comet.config` にAPI keyがない。
7. `.vscode/settings.json` に旧
   `2026_04_ishikawa_simple-MeMViT/.venv-memvit`
   hard-coded interpreter pathが残っていない。

### 8.2 syntax / import smoke

必須。

少なくとも次が例外なく通ること。

```text
python -m compileall args callback dataset logger model setup utils main.py main_pl.py train.py val.py
```

および主要package import smoke。

```text
import args
import dataset
import model
import setup
import utils
```

環境dependency不足でimportできない場合、
dependencyを勝手に追加して合わせず、現在環境の未検証事項として報告する。
reference `requirements*.txt` は本specでは変更しない。

### 8.3 CPU / local-data-independent tests

必須候補として、外部datasetやGPUを要求しない範囲を実行する。

```text
pytest -q   test/utils/test_accuracy.py   test/utils/test_average_meter.py   test/dataset/test_zero_images.py
```

これらが現在環境上で実行不能の場合は、理由を記録し、
構造同期成功とtest未実行を区別する。

### 8.4 GPU / external dataset依存tests

次はreference testとして復元するが、本specの必須pass条件にはしない。

- `test/model/test_model_factory.py`
- `test/model/test_image_models.py`
- `test/model/test_videomodels.py`
- `test/setup/test_optimizer.py`
- `test/setup/test_scheduler.py`
- `test/utils/test_checkpoint.py`
- `test/dataset/test_video_folder.py`

理由:

- 一部は `torch.cuda.is_available()` を必須とする。
- pretrained model downloadを伴いうる。
- VideoFolder testは研究室固有dataset pathを前提とする。

GPU / datasetが利用可能な場合は追加smokeとして実行してよいが、
実験datasetの大規模runは行わない。

## 9. Success Criteria

本specの実装成功は、次をすべて満たすこととする。

1. implementation branchが
   `chore/simple-cnn-baseline-sync` であり、
   `main@e0deb093...` を起点としている。
2. `main` と `mae` を直接変更していない。
3. 5.1の旧研究固有30ファイルが削除されている。
4. 5.2のsimple_cnn標準30ファイルが復元されている。
5. 5.3の15ファイルがreference commitと同内容である。
6. 5.4の既存同一25ファイルへ不要な変更がない。
7. `.gitignore` で `test/` が追跡可能になっている。
8. `.comet.config` がcurrent research project用であり、secretを含まない。
9. syntax / import smoke結果を記録している。
10. CPU / local-data-independent test結果を記録している。
11. GPU / external dataset依存testを実行したかどうかと理由を記録している。
12. baseline同期以外のViT / sequential_loader / LoRA / MoCo新機能が混入していない。
13. baseline同期が1つの論理commitとして回収されている。
14. force reset、history rewrite、branch削除を行っていない。

## 10. 明示的な対象外

本specでは次を行わない。

- new ImageNet-pretrained ViT feature extraction実装
- CLIP-ViT
- `tamaki-lab/sequential_loader` integration
- 50Saladsの新しいloader bridge
- video clip representation
- LoRA injection
- LoRA hyperparameter決定
- MoCo / BYOL / MAE実装
- temporal modeling
- sequential vs shuffle experiment
- scientific training run
- hyperparameter tuning
- dataset download
- old MeMViT codeの新frameworkへの移植
- backward compatibility layer
- `mae` branch整理
- push / Pull Request / merge
- main branchのhistory rewrite

## 11. Failure / Stop Conditions

次の場合は黙って条件変更せず停止・報告する。

- implementation base commit `e0deb093...` を取得できない。
- reference commit `e541dd98...` を取得できない。
- local dirty changeがあり、安全なbranch/worktreeを作れない。
- simple_cnn reference fileと本specのfile inventoryが一致しない。
- implementation中に基準file以外の変更が必要になる。
- reference dependencyのままimport不能で、dependency変更が必要になる。
- current repo固有secret / credentialを発見する。
- branch作成時に既存同名branchがあり、その内容が本specと無関係である。

この場合、force overwriteやbranch削除で続行しない。

## 12. Ambiguity Gate

### 12.1 Blocking

**なし。**

本specでは次を固定した。

- baseline reference repository / commit
- implementation base repository / commit
- work branch名
- 削除 / 復元 / revert対象
- repository固有exception
- 必須検証
- 対象外
- merge / pushを自動実行しない境界

### 12.2 Non-blocking

実装者が既存Git / Python styleに従って決めてよい。

- file copyに使う具体的なGit command
- clean worktreeを作るdirectory名
- test実行順
- compileall / pytestのlog表示形式
- commit前のdiff確認command

ただし、branch起点、file内容、scope、Success Criteriaを変えてはならない。

## 13. 実装後の次段階

本specがimplementedになった後、
次の研究実装は別specで扱う。

第一候補:

```text
simple_cnn baseline
  -> ImageNet-pretrained ViT
  -> single-image feature extraction smoke
```

その後、

```text
ViT
  -> external sequential_loader
  -> video clip frame features
  -> LoRA
  -> MoCo
```

と段階的に追加する。

## 14. Spec Gate

本draftでは、目的、reference revision、implementation base revision、
branch、変更scope、breaking change、検証、Success Criteria、対象外を固定した。

blocking ambiguityは現時点でない。

ただし、現在の依頼は「specを作る」までであり、
コード実装の明示承認ではないためstatusは `draft` とする。

ユーザーが本内容を承認した場合、
statusを `approved` へ変更し、engineering-taskへ引き継ぐ。

# Implementation Handoff

- approved spec: 本specがapprovedになった後に使用
- 実装目的: current Ishikawa repoをsimple_cnn clean baselineへ同期する
- 基準repository/commit:
  - implementation: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@e0deb093694d367ed9b02065e6d4cd38802093d6`
  - reference: `tamaki-lab/simple_cnn_training@e541dd98f825eb58c193ccd55cffa858392b89fe`
- work branch: `chore/simple-cnn-baseline-sync`
- 変更scope: 5章の削除30 / 復元30 / revert15 / repo固有2設定
- 対象外: ViT新規実装、sequential_loader、LoRA、MoCo、full experiment
- success criteria: 9章
- 許可されている短時間検証: syntax/import smoke、CPU/local-data-independent pytest、利用可能なら追加GPU smoke
- 長時間runの許可状態: 未許可
- push / PR / merge: 未許可
