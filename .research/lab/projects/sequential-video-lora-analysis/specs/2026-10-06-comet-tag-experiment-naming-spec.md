---
project: sequential-video-lora-analysis
record_type: implementation-spec
status: draft
created: 2026-10-06
last_updated: 2026-10-06
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_base_commit: f5d480b9928fd3f79efd8b5dfca2a2caffc7aad0
depends_on:
  - 2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md
  - ../../../../secretary/notes/brainstorm/2026-10-06-comet-tag-and-experiment-naming.md
---

# Comet Tag Taxonomy and Experiment Naming Spec

## 1. 目的

ActivityNet full runを開始する前に、Comet上で以下を容易にfilter・比較できるよう、
既存のComet experiment tagとexperiment nameを統一する。

- production run と smoke run。
- Base ViT と final MoCo Query LoRA。
- Linear Probeのseed 0 / 1 / 2。
- Stage 6B MoCo protocol と Linear Probe protocol。
- manifest / feature extraction / probe / aggregate の処理段階。

本specはtracking UI上の比較可能性を改善するための実装契約であり、
MoCo学習条件、Linear Probe科学条件、artifact payload、hash、metric値、学習結果を変更しない。

## 2. Authorityと根拠

Authority順:

1. 2026-10-06のユーザー採用判断。
2. [Cometタグ・Experiment命名規則 brainstorm](../../../../secretary/notes/brainstorm/2026-10-06-comet-tag-and-experiment-naming.md)。
3. [Full-dataset single-pass Streaming MoCo + ActivityNet segment Linear Probe Spec](2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md)。
4. 現行実装 `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`。
5. 既存test。

実装調査時点の `dev` headは
`f5d480b9928fd3f79efd8b5dfca2a2caffc7aad0` である。

確認した主な実装:

- `logger/comet_lineage.py`
- `scripts/moco/train_full_streaming_moco.py`
- `scripts/linear_probe/build_manifest.py`
- `scripts/linear_probe/extract_features.py`
- `scripts/linear_probe/run_probe.py`
- `evaluation/linear_probe.py`
- `scripts/retry_comet_artifact.py`
- `test/evaluation/test_comet_lineage.py`
- `test/training/test_full_streaming_moco_cli.py`
- `test/evaluation/test_linear_probe_cli.py`

## 3. 現在実装

現在の代表的なComet tagは次である。

| 処理 | 現在のexperiment name | 現在のtags |
|---|---|---|
| MoCo | `moco-full__<run_id>` | `moco`, `full-dataset` |
| Manifest | `<manifest_id>__manifest` | `linear-probe`, `manifest` |
| Feature | `lp-v1__features__<condition>` | `linear-probe`, `features` |
| Probe | `lp-v1__<condition>__seed-<N>` | `linear-probe`, raw condition |
| Aggregate | `lp-v1__aggregate` | `linear-probe`, `aggregate` |
| Retry | `retry__<artifact_name>` | `retry`（新規experimentの場合） |

現状でもrun identity、scientific config、provenance、artifact hash等はparameter / metadataに記録される。
本specではそれらをtagへ重複コピーせず、比較軸だけをtag化する。

## 4. 設計原則

Comet上の責務を次のように固定する。

### 4.1 Tags

低カーディナリティで、UI filter・比較に直接使う軸だけを持つ。

### 4.2 Experiment name

一覧上でそのexperimentの役割を人間が識別するために使う。
厳密なidentityはexperiment名へ詰め込まない。

### 4.3 Parameters / metadata

厳密な再現性・lineageのSSOTとする。

以下はtagへ追加しない。

- `run_id`
- Git commit SHA
- snapshot SHA / artifact SHA
- `processed_videos`
- `global_update_step`
- device / GPU identity
- dataset root
- local artifact path

dataset名・versionも本specではtag化しない。
将来、同一Comet project内で複数datasetを直接比較する要件が出た場合に別specで扱う。

## 5. Canonical tag vocabulary

tag spellingは次をcanonicalとする。

### 5.1 category / stage

- `moco`
- `linear-probe`
- `manifest`
- `features`
- `probe`
- `aggregate`

### 5.2 scope

- `production`
- `smoke`

同一experimentへ両方を付けてはならない。

### 5.3 condition

- `base-vit`
- `moco-query-lora-final`

コード上のcondition
`base_vit` / `moco_query_lora_final`
をComet tagへ出すときだけunderscoreをhyphenへ正規化する。
科学的conditionの内部表現自体は変更しない。

### 5.4 seed

- `seed-0`
- `seed-1`
- `seed-2`

Linear Probeでは各seed experimentへ対応seedだけを付ける。
MoCoではruntime seedから `seed-<N>` を生成し、seed 0以外も同じ規則を使う。

### 5.5 protocol

- `stage6b-v2`
- `lp-v1`

MoCo protocol tagはversion付きHydra preset名
`settings.moco.name`（現行 `stage6b_v2`）からunderscoreをhyphenへ正規化して導出する。

Linear Probe protocol tagは
`resolved['linear_probe']['id']` / `ProbeConfig.protocol`
（現行 `lp-v1`）から導出する。

protocol tag文字列を別configへ重複定義してはならない。

## 6. Scope判定

### 6.1 MoCo

MoCoのscope tagは次で決める。

```text
production
  iff settings.production_groups_match() is True
      and runtime.stop_after_videos is None

smoke
  otherwise
```

理由:

- `production_groups_match()`だけでは、canonical science configを使った
  `stop_after_videos`付き短時間runをproductionと誤tagする可能性がある。
- full runは10,024 sourceのsingle passを完走するproduction pathを意図するため、
  explicit stop条件があるrunはtracking上smokeとする。

既存の `run_config.production_config` の意味は変更しない。
scope tag判定のために別のboolを必要とする場合は局所計算し、
scientific config identityを変更しない。

### 6.2 Manifest

保存済み / audit済みmetadataの `production` を使う。

- `True` -> `production`
- `False` -> `smoke`

### 6.3 Feature extraction

既存の `production` 判定を使う。

```text
manifest_metadata['production'] and feature science canonical
```

MoCo-LoRA conditionのproduction featureでは既存のfinal snapshot validationを維持する。

### 6.4 Probe / Aggregate

`run_probe.py` が現在計算している既存 `production` boolをそのまま使う。
production条件の意味は変更しない。

## 7. Experiment name contract

### 7.1 MoCo

```text
<stage6b-protocol-tag>__moco__<run-id>
```

現行例:

```text
stage6b-v2__moco__activitynet-full-seed0
```

### 7.2 Manifest

```text
<linear-probe-protocol>__manifest__<manifest-id>
```

現行例:

```text
lp-v1__manifest__activitynet-full
```

### 7.3 Features

```text
<linear-probe-protocol>__features__<condition-tag>
```

現行例:

```text
lp-v1__features__base-vit
lp-v1__features__moco-query-lora-final
```

### 7.4 Probe

```text
<linear-probe-protocol>__probe__<condition-tag>__seed-<N>
```

現行例:

```text
lp-v1__probe__base-vit__seed-0
lp-v1__probe__moco-query-lora-final__seed-0
```

`evaluation.linear_probe.experiment_name()` をこのcontractへ更新し、
aggregate内の `probe_experiment_keys` も同じhelperの結果を使う。

### 7.5 Aggregate

```text
<linear-probe-protocol>__aggregate
```

現行例:

```text
lp-v1__aggregate
```

## 8. Experiment tag contract

tag tupleの順序もdeterministicに固定する。
Comet上では順序に意味を持たせないが、testしやすくするため呼び出し順を統一する。

### 8.1 MoCo

production:

```text
('moco', 'production', 'stage6b-v2', 'seed-<N>')
```

smoke:

```text
('moco', 'smoke', 'stage6b-v2', 'seed-<N>')
```

既存 `full-dataset` tagは削除する。
`production`がfull runを、`smoke`がpartial runを表すため、
二重のscope表現を残さない。

### 8.2 Manifest

```text
('linear-probe', 'manifest', '<scope>', 'lp-v1')
```

### 8.3 Base ViT feature

```text
('linear-probe', 'features', 'base-vit', '<scope>', 'lp-v1')
```

### 8.4 MoCo Query LoRA feature

```text
('linear-probe', 'features', 'moco-query-lora-final', '<scope>', 'lp-v1')
```

### 8.5 Base ViT Probe

```text
('linear-probe', 'probe', 'base-vit', 'seed-<N>', '<scope>', 'lp-v1')
```

### 8.6 MoCo Query LoRA Probe

```text
('linear-probe', 'probe', 'moco-query-lora-final', 'seed-<N>', '<scope>', 'lp-v1')
```

### 8.7 Aggregate

```text
('linear-probe', 'aggregate', '<scope>', 'lp-v1')
```

## 9. 実装方針

### 9.1 共通helper

`logger/comet_lineage.py` に、Comet表示用tokenの最小helperを追加してよい。

必須責務:

- boolからscope tagを返す。
- config / condition valueのunderscoreをhyphenへ正規化する。

例示interface:

```python
def scope_tag(production: bool) -> str:
    ...

def display_tag(value: str) -> str:
    ...
```

正確なfunction名は既存styleへ合わせてよいが、同じ変換を各scriptへコピーしない。

制約:

- `scope_tag` はbool以外を黙ってtruthy/falsy変換しない。
- `display_tag` はscientific configそのものを書き換えず、Comet表示用stringだけを返す。
- 新しいYAML taxonomy configは追加しない。
- tag taxonomyを別のmutable sourceへ複製しない。

### 9.2 MoCo

`scripts/moco/train_full_streaming_moco.py` で:

- `settings.moco.name`からprotocol tagを導出。
- runtime seedからseed tagを導出。
- Section 6.1のscope判定を行う。
- experiment nameとtagsをSection 7 / 8へ変更。
- fresh run / resumeのComet experiment key継続契約は変更しない。

resume時は `ExistingExperiment` を使用する現行動作を維持し、
既存experimentへtagを再追加しない。
本spec実装後に開始したfresh runには初回作成時点で必要tagが揃うため、
resumeでbackfillする必要はない。

### 9.3 Manifest

`scripts/linear_probe/build_manifest.py` のaudit時に:

- protocolは `resolved['linear_probe']['id']` 相当のversioned configから取得できる形にする。
- metadata `production`からscope tagを導出。
- name / tagsをSection 7.2 / 8.2へ変更。

manifest science contract、Gate、artifact alias、hashは変更しない。

### 9.4 Feature extraction

`scripts/linear_probe/extract_features.py` で:

- condition tagをruntime conditionから表示用正規化。
- scopeは既存 `production` boolから導出。
- protocolはversioned Linear Probe config IDから導出。
- tagsへcondition / scope / protocolを追加する。

feature_id、artifact name、artifact alias、feature identity SHAは変更しない。

### 9.5 Probe

`evaluation/linear_probe.py::experiment_name()` をSection 7.4へ変更する。

`scripts/linear_probe/run_probe.py::run_seed()` で:

- raw condition tagをhyphen表記へ正規化。
- `probe` stage tagを追加。
- `seed-<N>`を追加。
- scope tagを既存production boolから追加。
- protocol tagを追加。

aggregateの `probe_experiment_keys` keyも更新済み `experiment_name()`を使うため、
Comet experiment nameとの不一致を作らない。

### 9.6 Aggregate

`run_probe.py` のaggregate experimentへscope / protocol tagを追加する。
experiment name `lp-v1__aggregate` は維持する。

## 10. Retry behavior

`scripts/retry_comet_artifact.py` の責務はartifact再登録であり、
比較用tag taxonomyの再構築ではない。

本specでは次を維持する。

- 新規retry experimentのtagは `retry`。
- upstream experiment keyがある場合は `ExistingExperiment`へ再接続する。
- `start_experiment()` のexisting experiment pathで新tagをbackfillしない。
- retry成功可否はartifact metadataの `status / retry_needed / artifact_version / experiment_key`
  で追跡する。

過去experimentへのtag migration / backfillはout of scopeとする。

## 11. Backward compatibility

変更してよいもの:

- 新規Comet experimentの表示名。
- 新規Comet experimentのtags。
- aggregate内 `probe_experiment_keys` のkey文字列
  （新しいexperiment naming contractへ追従）。

変更してはならないもの:

- local artifact directory layout。
- artifact payload file。
- artifact SHA-256。
- artifact name。
- artifact alias。
- Comet project名。
- metric key。
- scientific config。
- production判定の科学的意味。
- manifest / feature / probe identity SHAの入力。
- resume checkpoint identity。
- evaluation snapshot contract。
- existing local artifact reuse判定。

本変更だけを理由に既存local artifactを無効化・再計算しない。

過去Comet experimentsのname/tagをmigrationしない。
本spec実装後に新規作成されるexperimentから新contractを適用する。

## 12. Explicit out of scope

- MoCo / Probe hyperparameter変更。
- full runの実行。
- full run bash launcher作成。
- Comet projectの変更・分割。
- dashboard / panel自動作成。
- dataset tag追加。
- run_idやcommit SHAのtag化。
- Comet APIによる過去runの一括tag backfill。
- artifact naming / versioning方針変更。
- retry experimentへ元experimentの全tagを複製する機能。

## 13. Test requirements

### 13.1 Comet helper unit tests

`test/evaluation/test_comet_lineage.py` または責務に合う既存testへ追加する。

最低限:

- `True -> production`。
- `False -> smoke`。
- bool以外を拒否。
- `stage6b_v2 -> stage6b-v2`。
- `moco_query_lora_final -> moco-query-lora-final`。
- `lp-v1 -> lp-v1`。

### 13.2 MoCo CLI

`test/training/test_full_streaming_moco_cli.py` で
`start_experiment`をmockし、fresh runについて以下を確認する。

canonical full run:

```text
name = stage6b-v2__moco__run1
tags = ('moco', 'production', 'stage6b-v2', 'seed-7')
```

`runtime.stop_after_videos=2`:

```text
scope tag = smoke
```

scientific override:

```text
scope tag = smoke
```

resumeはsaved `moco_experiment_key`をexisting keyとして渡す既存testを維持する。

### 13.3 Manifest

auditで新experimentを作るtestを追加し、production / smokeそれぞれについて
name / tagsを確認する。

smoke例:

```text
name = lp-v1__manifest__m
tags = ('linear-probe', 'manifest', 'smoke', 'lp-v1')
```

### 13.4 Feature extraction

Base / LoRA双方でname / tagsを確認する。

smoke Base:

```text
name = lp-v1__features__base-vit
tags = ('linear-probe', 'features', 'base-vit', 'smoke', 'lp-v1')
```

smoke LoRA:

```text
name = lp-v1__features__moco-query-lora-final
tags = ('linear-probe', 'features', 'moco-query-lora-final', 'smoke', 'lp-v1')
```

### 13.5 Probe experiment naming

`evaluation.linear_probe.experiment_name()`について:

```text
lp-v1, base_vit, 0
-> lp-v1__probe__base-vit__seed-0

lp-v1, moco_query_lora_final, 2
-> lp-v1__probe__moco-query-lora-final__seed-2
```

### 13.6 Probe tags

seed runで次を確認する。

```text
('linear-probe', 'probe', 'base-vit', 'seed-0', 'smoke', 'lp-v1')
```

LoRA / seed 1,2も同じ生成規則を使う。

### 13.7 Aggregate

smoke aggregate:

```text
name = lp-v1__aggregate
tags = ('linear-probe', 'aggregate', 'smoke', 'lp-v1')
```

production pathではscopeだけが `production` へ変わる。

### 13.8 Regression

既存以下のcontractを維持する。

- Comet disabled時にlocal artifactが消えない。
- Comet upload failureがlocal scientific workをfailさせない。
- hash不一致artifactをuploadしない。
- local feature / probe artifact reuseが維持される。
- full MoCo fresh / resume contractが維持される。
- Linear Probeのmetric / aggregate値はtag変更の影響を受けない。

## 14. Acceptance criteria

実装完了には以下をすべて満たすこと。

1. 新規MoCo experimentがSection 7.1 / 8.1のname・tagsを持つ。
2. `stop_after_videos`付きMoCo runが `smoke` になり、`production`を持たない。
3. Manifest / Feature / Probe / AggregateがSection 7 / 8のname・tagsを持つ。
4. Base / LoRA featureをtag filterだけで区別できる。
5. Base / LoRA Probeを `probe + production + seed-N` で同一seed比較できる。
6. `stage6b-v2` と `lp-v1` がそれぞれversioned configから導出され、別hard-code sourceを持たない。
7. scientific config、metrics、artifact payload/hash/name/alias、local reuse identityが変更されない。
8. resume時に既存MoCo experiment keyを継続する。
9. retry workflowが既存どおりartifact再登録できる。
10. relevant unit / CLI / end-to-end smoke testsが全件PASSする。
11. 実ActivityNet full runはこの変更の検証として実行しない。短時間 / synthetic testで実装検証を完了できる。

## 15. Ambiguity Gate

### Blocking

なし。

ユーザー採用済みの内容により、tag vocabulary、scope、condition、seed、protocol、
Experiment命名規則、dataset tagを追加しない方針まで確定している。

### Non-blocking

- 共通helperの具体的function名。
- tag helper testを既存test fileへ置くか責務別fileへ分けるか。
- caller内の局所変数名。

これらは既存styleに従ってよく、Comet外部挙動を変更してはならない。

## 16. Implementation handoff

本specがapprovedになった後、`engineering-task`で実装する。

想定変更箇所:

```text
logger/comet_lineage.py
scripts/moco/train_full_streaming_moco.py
scripts/linear_probe/build_manifest.py
scripts/linear_probe/extract_features.py
scripts/linear_probe/run_probe.py
evaluation/linear_probe.py
test/evaluation/test_comet_lineage.py
test/training/test_full_streaming_moco_cli.py
test/evaluation/test_linear_probe_cli.py
（必要に応じて既存Linear Probe unit test）
```

実装はComet tracking表示層に限定し、full run launcher作成とは分離する。
