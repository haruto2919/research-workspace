---
date: 2026-10-06
project: sequential-video-lora-analysis
source_todo: null
topic: comet-tag-and-experiment-naming
status: exploratory
tags: [brainstorm, research, comet, experiment-tracking]
---

# Cometタグ・Experiment命名規則の整理

## 出発点

ActivityNet full runに入る前に、Comet上でBase ViTとMoCo Query LoRAの結果を比較しやすくするため、
タグとExperiment名の設計を整理した。

現在の実装ではCometへの記録自体は行われているが、比較軸として使いたい
production / smoke、feature condition、seed、protocol等の一部がparameter側にしかなく、
タグだけで十分に絞り込みにくい。

## 現在実装から確認したこと

対象repository:

- `tamaki-lab/2026_09_ishikawa_sequential-video-lora`
- 調査branch: `dev`

主な確認箇所:

- `logger/comet_lineage.py`
- `scripts/moco/train_full_streaming_moco.py`
- `scripts/linear_probe/build_manifest.py`
- `scripts/linear_probe/extract_features.py`
- `scripts/linear_probe/run_probe.py`
- `scripts/retry_comet_artifact.py`

現状の代表例:

- MoCo: `tags=('moco', 'full-dataset')`
- Manifest: `tags=('linear-probe', 'manifest')`
- Feature: `tags=('linear-probe', 'features')`
- Probe: `tags=('linear-probe', condition)`
- Aggregate: `tags=('linear-probe', 'aggregate')`
- retry: `tags=('retry',)`

## 採用した設計原則

Comet上の責務を次のように分ける。

- **Tags**: 比較・filterに使う低カーディナリティの実験軸
- **Experiment name**: 一覧を見たときにrunの役割を識別する名前
- **Parameters / metadata**: run ID、commit SHA、artifact SHA、device、snapshot step等の厳密な再現・lineage情報

高カーディナリティな値はタグに増やさない。

## 採用したタグ一覧

### 共通カテゴリ

- `moco`
- `linear-probe`

### stage

- `manifest`
- `features`
- `probe`
- `aggregate`

### scope

- `production`
- `smoke`

### condition

- `base-vit`
- `moco-query-lora-final`

### seed

- `seed-0`
- `seed-1`
- `seed-2`

MoCoにもseed tagを付ける。

### protocol

- `stage6b-v2`
- `lp-v1`

意味:

- `stage6b-v2`: 現行Stage 6B Streaming MoCo protocolのversion識別子
- `lp-v1`: 現行Linear Probe評価protocolのversion識別子

コード/config側では `stage6b_v2` の表記を使うが、
Comet tagでは可読性のため `stage6b-v2` とする。

## 各Experimentに付けるタグ

### MoCo full run

```text
moco
production
stage6b-v2
seed-<N>
```

smoke時は `production` を `smoke` に置き換える。

### Manifest

```text
linear-probe
manifest
production
lp-v1
```

### Base ViT feature

```text
linear-probe
features
base-vit
production
lp-v1
```

### MoCo Query LoRA feature

```text
linear-probe
features
moco-query-lora-final
production
lp-v1
```

### Base ViT Linear Probe

```text
linear-probe
probe
base-vit
seed-<N>
production
lp-v1
```

### MoCo Query LoRA Linear Probe

```text
linear-probe
probe
moco-query-lora-final
seed-<N>
production
lp-v1
```

### Aggregate

```text
linear-probe
aggregate
production
lp-v1
```

smoke時は各Experimentで `production` を `smoke` に置き換える。

## 採用したExperiment命名規則

### MoCo

```text
stage6b-v2__moco__<run-id>
```

例:

```text
stage6b-v2__moco__activitynet-full-seed0
```

### Manifest

```text
lp-v1__manifest__<manifest-id>
```

### Feature

```text
lp-v1__features__<condition>
```

例:

```text
lp-v1__features__base-vit
lp-v1__features__moco-query-lora-final
```

### Probe

```text
lp-v1__probe__<condition>__seed-<N>
```

例:

```text
lp-v1__probe__base-vit__seed-0
lp-v1__probe__moco-query-lora-final__seed-0
```

### Aggregate

```text
lp-v1__aggregate
```

## タグにしない情報

以下はparameter / metadataとして保持し、タグにはしない。

- run_id
- Git commit SHA
- snapshot SHA
- processed_videos
- global_update_step
- device / GPU identity
- artifact SHA
- dataset root path

Dataset名・versionも現時点ではタグ化しない。
将来、ActivityNetとEPIC-KITCHENS等を同一Comet project内で直接比較する必要が出た場合に、
dataset tagの追加を再検討する。

## 比較の想定

例えば本番Linear Probeだけを見たい場合:

```text
linear-probe + probe + production
```

同一seedでBase ViTとMoCo Query LoRAを比較する場合:

```text
linear-probe + probe + production + seed-0
```

MoCo Query LoRA側だけを3 seed横断で見る場合:

```text
linear-probe + probe + production + moco-query-lora-final
```

## 保留・注意点

- `retry_comet_artifact.py` が `ExistingExperiment` に再接続する場合、
  現行 `start_experiment()` は新しいtagを追加しない。
  retryはartifact再登録の運用目的であり、比較用tagの主経路とはしない。
- Dataset tagは現時点では追加しない。
- このメモはbrainstorm記録であり、実装specではない。

## Research Spec Handoff候補

- 実装目的: Comet上でfull run / smoke、Base / LoRA、seed、protocolを容易に比較できるようにする。
- 変更候補:
  - 各 `start_experiment()` 呼び出しのtag生成
  - Experiment名の統一
  - production / smoke tagの自動判定
  - condition / seed / protocol tagの追加
- 維持するもの:
  - run_id、commit、artifact hash等はparameter / metadataへ保持
  - 現行artifact lineageとhash検証
- 次段階: ユーザーがspec化を明示した場合、`research-spec` でscope・互換性・成功条件を確認する。
