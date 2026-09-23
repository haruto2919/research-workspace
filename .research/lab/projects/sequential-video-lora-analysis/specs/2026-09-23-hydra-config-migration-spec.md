---
project: sequential-video-lora-analysis
spec_type: implementation
status: implemented
title: argparseからHydra/YAMLへの設定基盤移行
created: 2026-09-23
last_updated: 2026-09-23
workspace_repository: haruto2919/research-workspace
workspace_base_branch: main
workspace_base_commit: e7842199e34cafa3ab1c3489fb33f782e5cadc7c
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_base_branch: main
implementation_base_commit: 7705e2159678fa516a8f17a0633e4712377b6eb9
implementation_work_branch: dev
hydra_version: 1.3.7
---

# argparseからHydra/YAMLへの設定基盤移行 spec

> **Status: implemented**
>
> 本specは、研究コードのtraining設定基盤を、flatな `argparse.Namespace` から
> YAML + Hydraへ移行するための実装契約である。
>
> 2026-09-23にユーザーが、training configのみHydra化し、`main.py` / `main_pl.py` の両方を移行、
> 旧training CLI互換layerを残さず、dataset / model / optimizerをconfig group化する方針を採用した。
> 2026-09-23に `dev` へ実装し、短時間検証を完了した。学習・実験runは未実施。
> 検証結果と既存テスト不一致1件は[実装・検証記録](../experiments/2026-09-23-hydra-config-migration-verification.md)を参照。

## 1. 目的

現在の研究コードでは `args/arg_parse.py` が、dataset、model、video、training、optimizer、
GPU、logging、checkpoint設定を1つのflatな `argparse.Namespace` として生成し、
`main.py` / `main_pl.py` から各componentへ渡している。

今後はActivityNet、ViT、LoRA、MoCo、sampling、control条件など設定軸が増えるため、
training設定をYAMLとしてversion controlし、Hydraでcompose / overrideできる基盤へ移行する。

本変更の中心目的は次の2点である。

1. 現在のtraining挙動・default値を変えずに設定入口だけをHydraへ置き換える。
2. 今後の `dataset × backbone × LoRA × SSL × sampling/control` 比較に耐える設定構造を先に整える。

研究アルゴリズム、dataset semantics、model behaviorを変更するspecではない。

## 2. Authority / Evidence

### 2.1 現在のユーザー判断

2026-09-23に次を採用した。

- YAML + Hydraへ移行する。
- 中央のtraining configのみHydra化する。
- `main.py` と `main_pl.py` の両方を移行する。
- 旧 `-b`, `-m`, `-d`, `-lr` 等のtraining CLI互換layerは残さない。
- dataset / model / optimizerをconfig group化する。
- utility / smoke script固有の `argparse` は維持する。
- Hydra導入だけを独立specとし、ActivityNet integration、clip aggregation、LoRA、MoCoを混ぜない。
- 実装branchは既存の `dev` を使用する。

### 2.2 implementation基準revision

| 役割 | repository | branch | commit |
|---|---|---|---|
| Research Workspace | `haruto2919/research-workspace` | `main` | `e7842199e34cafa3ab1c3489fb33f782e5cadc7c` |
| implementation base | `tamaki-lab/2026_09_ishikawa_sequential-video-lora` | `main` | `7705e2159678fa516a8f17a0633e4712377b6eb9` |

implementation work branchは既存の `dev` とする。
2026-09-23確認時点で `dev` は `main` と同じ `7705e2159678fa516a8f17a0633e4712377b6eb9` を指している。

### 2.3 現行実装から確認した依存関係

- `main.py` / `main_pl.py` は `ArgParse.get()` をentrypointとして使用する。
- `dataset/dataloader_factory.py` と `dataset/dataset_pl.py` はflat Namespaceからdataset / loader / video設定を参照する。
- `model/simple_lightning_model.py` はflat Namespaceからmodel / optimizer / scheduler / checkpoint設定を参照する。
- `main.py` は `vars(args)` をlogging parameterとして渡す。
- `model/model_config.py` には既存component-level dataclass `ModelConfig` がある。
- `requirements.txt` にはHydraはまだ含まれていない。
- utility / smoke scriptsには個別用途の `argparse` が存在する。

## 3. Decision Contract

### 3.1 Hydra対象範囲

Hydraへ移行する対象は、研究training entrypointの設定だけとする。

対象:

```text
main.py
main_pl.py
args/arg_parse.py が現在提供しているtraining設定
それをconsumeするdataset / model / training周辺の必要最小限のinterface
```

対象外の `argparse`:

```text
audit_activitynet_inventory.py
smoke_vit_frame_encoder.py
smoke_50salads_vit_bridge.py
その他、単発utility / smoke専用CLI
```

リポジトリから `argparse` 自体を完全排除することは目的としない。

### 3.2 entrypoint

`main.py` と `main_pl.py` の両方をHydra entrypointへ変更する。

概念形:

```python
@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="config",
)
def main(cfg: DictConfig):
    ...
```

`version_base` は明示する。

### 3.3 Hydra version

stable系を使用し、次をpinする。

```text
hydra-core==1.3.7
```

1.4 development releaseは使用しない。
`omegaconf` を別に再実装・vendorしない。

### 3.4 primary config

primary config:

```text
conf/config.yaml
```

最低限の構成:

```text
conf/
├── config.yaml
├── dataset/
│   ├── cifar10.yaml
│   ├── imagefolder.yaml
│   ├── videofolder.yaml
│   └── zero_images.yaml
├── model/
│   ├── resnet18.yaml
│   ├── resnet50.yaml
│   ├── x3d.yaml
│   ├── abn_r50.yaml
│   ├── vit_b.yaml
│   └── zero_output_dummy.yaml
└── optimizer/
    ├── sgd.yaml
    └── adam.yaml
```

LoRA / MoCo / ActivityNet / aggregation用config groupは本specで先回りして作らない。

### 3.5 defaults composition

`conf/config.yaml` のdefaultsは、現行 `ArgParse` のdefault behaviorを維持する。

```yaml
defaults:
  - dataset: cifar10
  - model: resnet18
  - optimizer: sgd
  - _self_
```

config group名はCLI選択用にlowercase / snake_caseを使用してよいが、
内部で既存factoryへ渡す `name` は既存literal値を維持する。

例:

```yaml
# conf/dataset/cifar10.yaml
name: CIFAR10
root: ./downloaded_data
```

```yaml
# conf/optimizer/sgd.yaml
name: SGD
lr: 1.0e-4
weight_decay: 5.0e-4
momentum: 0.9
```

### 3.6 nested config

config groupにしない共通設定は、最低限次のnested sectionへ整理する。

```text
loader
video
trainer
scheduler
logging
checkpoint
```

概念contract:

```yaml
loader:
  batch_size: 8
  num_workers: 2

video:
  frames_per_clip: 16
  clip_duration: 2.6666666666666665
  clips_per_video: 1

trainer:
  num_epochs: 25
  val_interval_epochs: 1
  log_interval_steps: 1
  grad_accum: 1
  use_dp: false
  devices: "-1"

scheduler:
  enabled: false

logging:
  comet_log_dir: ./comet_logs/
  tf_log_dir: ./tf_logs/
  disable_comet: false

checkpoint:
  save_dir: ./log
  resume: null
```

`model` groupは少なくとも次を保持する。

```text
name
use_pretrained
torch_home
```

dataset groupはdatasetごとに必要な既存fieldだけを保持する。

### 3.7 旧argparse fieldからHydra pathへの対応

| 現行field | Hydra path | 現行default |
|---|---|---|
| `root` | `dataset.root` | `./downloaded_data` |
| `dataset_name` | `dataset.name` | `CIFAR10` |
| `train_dir` | `dataset.train_dir` | `train` |
| `val_dir` | `dataset.val_dir` | `val` |
| `torch_home` | `model.torch_home` | `./pretrained_models` |
| `model_name` | `model.name` | `resnet18` |
| `use_pretrained` | `model.use_pretrained` | `true` |
| `frames_per_clip` | `video.frames_per_clip` | `16` |
| `clip_duration` | `video.clip_duration` | `80/30`相当 |
| `clips_per_video` | `video.clips_per_video` | `1` |
| `batch_size` | `loader.batch_size` | `8` |
| `num_workers` | `loader.num_workers` | `2` |
| `num_epochs` | `trainer.num_epochs` | `25` |
| `val_interval_epochs` | `trainer.val_interval_epochs` | `1` |
| `log_interval_steps` | `trainer.log_interval_steps` | `1` |
| `optimizer_name` | `optimizer.name` | `SGD` |
| `grad_accum` | `trainer.grad_accum` | `1` |
| `lr` | `optimizer.lr` | `1e-4` |
| `momentum` | `optimizer.momentum` | `0.9` |
| `weight_decay` | `optimizer.weight_decay` | `5e-4` |
| `use_scheduler` | `scheduler.enabled` | `false` |
| `use_dp` | `trainer.use_dp` | `false` |
| `devices` | `trainer.devices` | `"-1"` |
| `comet_log_dir` | `logging.comet_log_dir` | `./comet_logs/` |
| `tf_log_dir` | `logging.tf_log_dir` | `./tf_logs/` |
| `save_checkpoint_dir` | `checkpoint.save_dir` | `./log` |
| `checkpoint_to_resume` | `checkpoint.resume` | `null` |
| `disable_comet` | `logging.disable_comet` | `false` |

datasetに不要なfieldを全dataset YAMLへ無理に複製する必要はない。
例えば `train_dir / val_dir` はImageFolder / VideoFolderに必要な範囲で持たせる。

### 3.8 CLI override

Hydra標準overrideを使用する。

例:

```bash
python main_pl.py \
  dataset=imagefolder \
  model=vit_b \
  optimizer=adam \
  loader.batch_size=8 \
  optimizer.lr=1e-4 \
  trainer.devices=0
```

boolean変更例:

```bash
python main_pl.py model.use_pretrained=false
python main_pl.py scheduler.enabled=true
python main_pl.py logging.disable_comet=true
```

旧training CLI形式

```text
-d / -m / -b / -w / -e / -lr / --devices / --scratch / --no_comet ...
```

との互換parser / translation layerは実装しない。

### 3.9 working directory / Hydra output

Hydra導入により既存relative pathの意味を変更しない。

明示的に:

```yaml
hydra:
  job:
    chdir: false
```

とする。

Hydra自身のrun artifactは既存 `.gitignore` 対象である `log/` 配下へ置く。

概念形:

```yaml
hydra:
  run:
    dir: log/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: log/hydra/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

multirunの研究利用自体は本specの対象外だが、Hydra metadataの保存先だけを衝突しない形で定義する。

### 3.10 resolved config

実行時に使用されたresolved configをHydraのrun artifactとして残す。
独自YAML copy機構を別途実装しない。

既存loggerへ設定を渡す必要がある場合は `OmegaConf.to_container(..., resolve=True)` 等により
通常のPython mappingへ変換してよい。

### 3.11 componentへの設定受け渡し

flatな `argparse.Namespace` をHydra `DictConfig`へ単純置換して全componentへ丸ごと渡すだけの設計は避ける。

責務ごとに必要なsubtreeを渡す。

例:

```text
dataset:
  dataset cfg + loader cfg + video cfg

model:
  model cfg

optimizer:
  optimizer cfg

scheduler:
  scheduler cfg

checkpoint:
  checkpoint cfg
```

既存 `ModelConfig` のようなcomponent-level dataclassは維持・再利用してよい。

一方で、本specでは全YAML設定をStructured Config/dataclassとして二重定義しない。

### 3.12 current behavior preservation

Hydra移行前後で、同じ意味の設定を与えた場合に次を維持する。

- dataset選択。
- model選択。
- pretrained / scratch指定。
- loader batch size / worker数。
- video clip関連値。
- epoch / validation / logging interval。
- optimizer名とhyperparameter。
- scheduler on/off。
- DP / Lightning devices指定。
- Comet on/offとlog path。
- checkpoint save / resume path。

Hydra移行を理由にoptimizer、scheduler、model、datasetの既存algorithmを変更しない。

## 4. Implementation Scope

想定する主な変更:

```text
requirements.txt
conf/config.yaml
conf/dataset/*.yaml
conf/model/*.yaml
conf/optimizer/*.yaml
main.py
main_pl.py
dataset/dataloader_factory.py
dataset/dataset_pl.py
model/simple_lightning_model.py
README.md
```

training pathから不要になった場合:

```text
args/arg_parse.py
args/__init__.py
```

は削除してよい。

必要な最小testを追加する。

実装時に上記以外のfile変更が必要なら、Hydra config migrationへ直接必要かを確認する。

## 5. Explicit Out of Scope

本specでは次を実装しない。

- ActivityNet Adapter変更。
- ActivityNet -> ViT integration。
- masked mean / late fusion / clip representation。
- LoRA。
- LoRA target / rank / alpha / dropout。
- MoCo / BYOL / MAE等のSSL objective。
- query / key encoder。
- EMA。
- projector / InfoNCE / queue。
- temporal modeling。
- sequential / shuffle / reverse / static-repeat比較。
- parameter sweepの研究実行。
- Structured Configによる全config schemaのdataclass化。
- config-driven object instantiation (`hydra.utils.instantiate`) への全面移行。
- model / dataset factoryの全面設計変更。
- utility / smoke scriptのHydra化。
- research algorithmの変更。

## 6. Compatibility / Breaking Change

### 6.1 breaking change

training entrypointのCLI syntaxは意図的にbreaking changeとする。

旧:

```bash
python main_pl.py -d ImageFolder -m vit_b -b 8 -lr 0.0001 --devices 0
```

新:

```bash
python main_pl.py dataset=imagefolder model=vit_b loader.batch_size=8 optimizer.lr=0.0001 trainer.devices=0
```

旧CLI互換layerは持たない。

### 6.2 preserve

次は維持する。

- `main.py` と `main_pl.py` のentrypoint filename。
- existing dataset factoryが提供するdataset choices。
- existing model choices。
- existing optimizer choices。
- existing default behavior。
- utility / smoke script CLI。
- checkpoint file formatそのもの。

READMEの使用例は新Hydra CLIへ更新する。

## 7. Test / Verification Plan

### 7.1 config composition tests

少なくとも次を自動testする。

1. default configをcomposeできる。
2. defaultが `dataset=CIFAR10`, `model=resnet18`, `optimizer=SGD` 相当になる。
3. old ArgParse default値と3.7節のmappingが一致する。
4. `dataset=imagefolder` を選択できる。
5. `dataset=videofolder` を選択できる。
6. `model=vit_b` を選択できる。
7. `optimizer=adam` を選択できる。
8. scalar override `loader.batch_size=...` が反映される。
9. `model.use_pretrained=false` が反映される。
10. `scheduler.enabled=true` が反映される。
11. `checkpoint.resume=null` を表現できる。
12. `hydra.job.chdir` がfalseである。

### 7.2 interface tests

必要最小限で次を確認する。

- `main.py` / `main_pl.py` が `ArgParse.get()` に依存しない。
- training componentが `argparse.Namespace` 型を要求しない。
- dataset factoryへ必要なdataset / loader / video設定が渡る。
- Lightning modelへmodel / optimizer / scheduler / checkpoint設定が渡る。
- existing `ModelConfig` contractを壊さない。

### 7.3 CLI/config smoke

長時間trainingを行わず、Hydraのconfig表示/composeでentrypoint設定を確認する。

例:

```bash
python main.py --cfg job --resolve
python main_pl.py --cfg job --resolve
python main_pl.py dataset=imagefolder model=vit_b optimizer=adam --cfg job --resolve
```

Hydra CLIがconfig表示だけで終了できない等の実装差異がある場合、同等のcompose testで代替してよい。

### 7.4 regression

既存の短時間unit testsを実行し、Hydra移行と無関係なmodel / dataset logicの回帰がないことを確認する。

実dataset full trainingは本specの必須verificationにしない。

## 8. Success Criteria

本specの実装成功は次を全て満たすこととする。

1. implementation work branchが `dev` であり、実装開始時点の基準が `7705e2159678fa516a8f17a0633e4712377b6eb9` である。
2. `hydra-core==1.3.7` がdependencyとして追加される。
3. `conf/config.yaml` がprimary configとして存在する。
4. dataset / model / optimizerがconfig groupとして存在する。
5. `main.py` と `main_pl.py` の両方がHydra entrypointを使う。
6. training pathで `ArgParse.get()` を使用しない。
7. 旧training CLI互換layerを実装していない。
8. utility / smoke専用 `argparse` は不要に変更していない。
9. 3.7節の既存default値が維持される。
10. Hydra CLI overrideでconfig groupとscalar値を変更できる。
11. `hydra.job.chdir=false` によりruntime cwdを変更しない。
12. Hydra run artifactが `log/hydra/` 配下に保存される。
13. componentが必要なconfig subtreeを受け取り、flat Namespace依存を解消している。
14. 全configをStructured Configへ二重定義していない。
15. READMEのtraining起動例がHydra CLIへ更新される。
16. default / representative overrideのconfig composition testが通る。
17. 既存のHydra移行と無関係な短時間testsが回帰しない。
18. ActivityNet integration / aggregation / LoRA / MoCoを混入していない。

## 9. Failure / Stop Conditions

次の場合は条件を黙って変えず停止・報告する。

- Hydra導入のためにdataset / model / optimizerの既存algorithm変更が必要になる。
- Hydra導入だけでは既存relative path semanticsを維持できない。
- current mainからbase revisionが大きく変わり、設定contract自体を再調査する必要がある。
- existing config fieldの意味を変えないと移行できない。
-旧CLIとHydra CLIの二重SSOTを残さないと既存重要workflowが成立しないことが判明する。
- utility / smoke scriptsまで全面Hydra化しないとpackage importが成立しない。
- config migrationのためにLoRA / MoCo / ActivityNet等の未実装研究機能を先に追加する必要が生じる。

この場合、互換layer追加や研究scope拡張を独断で行わない。

## 10. Ambiguity Gate

### 10.1 Blocking

なし。

採用済み:

- Hydra + YAML。
- training configのみ移行。
- `main.py` / `main_pl.py` 両方。
- 旧training CLI互換なし。
- dataset / model / optimizerのconfig group化。
-その他はnested config。
- utility / smoke argparse維持。
- `hydra.job.chdir=false`。
- current behavior / default値維持。
-研究機能追加なし。

### 10.2 Non-blocking

既存styleと本specのcontract内で決めてよい。

- private helper名。
- config test file名。
- YAML keyの並び順。
- `DictConfig` subtreeから既存dataclassへ変換するhelperの有無。
- `log/hydra/` 以下の秒以下のdirectory表現。
- args package directoryを空で残すか完全削除するか。
- READMEの説明文の細かな表現。

ただし、外部CLI、config path、default値、責務境界を変更してはならない。

## 11. Reproducibility

implementation base:

```text
tamaki-lab/2026_09_ishikawa_sequential-video-lora
main@7705e2159678fa516a8f17a0633e4712377b6eb9
```

work branch:

```text
dev
```

dependency:

```text
hydra-core==1.3.7
```

Hydra output:

```text
log/hydra/...
```

本specはconfig migrationの実装成功だけを判定する。
研究結果の再現性評価やfull training結果の一致は後続実験specで扱う。

## 12. 実装後の次段階

本specがimplementedになった後、研究実装は次へ進む。

```text
ActivityNet
 -> SequentialSample
 -> frozen ViT
 -> frame_features [T,768]
 -> masked mean
 -> clip_feature [768]
```

その後、LoRA、MoCo、full SSL loop、control比較を別specで段階的に扱う。

## 13. Spec Gate

本specは承認済み契約に基づき、2026-09-23に `implemented` とした。

目的、scope、breaking change、config構造、既存default mapping、
working directory、dependency、verification、対象外を固定し、blocking ambiguityは残っていない。

2026-09-23にユーザーが上記方針を採用し、本specの作成を明示的に依頼した。
同日、実装branchを既存の `dev` に変更する指示を受け、本specへ反映した。
承認時点ではコード実装およびtraining runは未実施だった。
同日、コード実装と必須の短時間検証を完了した。Hydra新規テスト38件成功、既存CPUテスト21件成功・既存不一致1件で、新たな回帰はない。training runは未実施。

# Implementation Handoff

- approved spec: 本spec
- implementation repository: `tamaki-lab/2026_09_ishikawa_sequential-video-lora`
- base: `main@7705e2159678fa516a8f17a0633e4712377b6eb9`
- work branch: `dev`
- implementation purpose: flat argparse training configをYAML + Hydraへ移行
- change scope: config files / training entrypoints / config consumer interface / dependency / README / tests
- breaking change: 旧training CLI syntaxは廃止
- must preserve: current training semantics / defaults / utility smoke CLI / research algorithms
- success criteria: 8章
- allowed short verification: config compose / config print / unit tests /既存短時間tests
- long run permission: なし
- excluded: ActivityNet integration / clip aggregation / LoRA / MoCo / training experiment
