---
project: sequential-video-lora-analysis
spec_type: implementation
status: implemented
title: Meta InfoNCE arithmetic opt-in for the normal ResNet50 DDP MoCo path
created: 2026-10-19
last_updated: 2026-10-19
authority: user-adopted B1-B12 including B6-prime; spec approved by the 2026-10-19 user requests to implement it and to record it as approved
approved_on: 2026-10-19
implemented_on: 2026-10-19
implementation_committed: false
implementation_verified_diff_sha256: 13bf95635d84eb441d9ad0e48fe950e0600f48990feb92b0511cb48eff3cca76
engineering_cpu_gate: PASS
gpu_gates: NOT_RUN
implementation_record: ../experiments/2026-10-19-meta-infonce-normal-moco-opt-in-implementation-verification.md
workspace_repository: haruto2919/research-workspace
workspace_branch: main
workspace_base_commit: 845cc7b432d11df5621ccac890fac449a4713030
workspace_dirty_before_spec: true
workspace_preexisting_untracked: [.agents/.DS_Store, .agents/skills/.DS_Store]
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_base_commit: 1ce9badcbae3e2df08fa400944e4c385785063f9
implementation_dirty_at_spec_review: false
reference_repository: facebookresearch/moco
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
predecessor_evidence: ../experiments/2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md
---

# Meta公式InfoNCE演算を通常ResNet50 DDP MoCoで明示利用する実装仕様（draft）

> **Status: implemented（2026-10-19承認・実装、Engineering/CPU Gate PASS、研究コード未コミット、GPU Gate未実施）**。本specは、採用済みB1〜B5・B6'・B7〜B12を、実装者が追加の研究判断をせず実装・CPU検証できる契約へ変換したもの。承認と実装は研究コードのcommit/push、GPU run、A2〜A5の正式実験を許可しない。A1の既存Evidenceとartifactは変更しない。実装・検証の結果は[実装・CPU検証記録](../experiments/2026-10-19-meta-infonce-normal-moco-opt-in-implementation-verification.md)、実装時の解釈は第18節。
>
> （作成時の状態：draft・実装不可・未承認。spec作成の依頼は研究コード変更、CPU/GPU run、commit/pushを含まなかった。）

## 1. 目的と主張範囲

### 1.1 目的

A1 synthetic、ResNet50＋MLP、2GPU DDP、K64、16stepでは、Meta公式Mと自作`meta_einsum`方式S1が、比較対象16,256件すべてでbitwise一致した。自作の従来`self`方式S0はMとの差を蓄積した。

本specの目的は、A1専用の診断selectorに閉じている`meta_einsum`を、ResNet50＋2GPU DDPの通常A2〜A5経路で**明示的に選択可能**にし、次を保証することである。

1. 既定の`self`と既存A1契約を維持する。
2. 設定値、構築モデル、全rank、provenance、Checkpoint、Resume、比較reportが同じeffective modeを指す。
3. 通常実行ではMeta公式を追加実行せず、full trace・全parameter copy・不要なGPU同期を持ち込まない。
4. 正式Parityだけを独立run＋保存traceのオフライン比較として実施する。

### 1.2 このspecの成功で主張できること

実装GateがPASSした場合、「対象commitのResNet50 DDP runnerで`meta_einsum`を明示opt-inでき、mode identity、旧A1互換、Checkpoint/Resume拒否、記録schema、軽量監視のCPU検証が通った」と主張できる。

### 1.3 主張できないこと

- A2〜A5のGPU実験がPASSしたこと。
- A1の`rtol=1e-5, atol=1e-6`がA2以降にも妥当であること。
- ImageNet実画像、K65536、1025step、長期間学習でMeta公式と一致すること。
- ImageNet 10epoch、Linear Probe、表現品質、Frozen ViT、動画Streaming MoCo、Sequential Loader、LoRAで有効であること。
- `meta_einsum`を将来の既定値へ変更してよいこと。

## 2. Authority、Evidence、現行実装

### 2.1 Authority

1. 本spec作成を依頼した現在のユーザー指示と採用済みB1〜B12（B6はB6'に置換）。
2. [通常経路導入の設計判断](../../../../secretary/notes/brainstorm/2026-10-19-moco-reference-infonce-normal-path-design-decisions.md)。
3. [A1・16step GPU parity Evidence](../experiments/2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md)。
4. [approved A1・16step spec](2026-10-18-meta-moco-a1-infonce-16step-parity-spec.md)。
5. [parent Spec A draft](2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md)、[2026-10-15 MTG](../meetings/2026-10-15-mtg.md)。

親Spec Aは`draft`であり、そのA2〜A5や長時間runを本specの承認だけで実行可能にしない。

### 2.2 repositoryの固定点

2026-10-19 11:04 JSTに次を確認した。

| repository | branch | GitHub HEAD | local state |
|---|---|---|---|
| `haruto2919/research-workspace` | `main` | `845cc7b432d11df5621ccac890fac449a4713030` | HEAD一致。spec作成前から`.DS_Store` 2件がuntracked |
| `tamaki-lab/2026_09_ishikawa_sequential-video-lora` | `dev` | `1ce9badcbae3e2df08fa400944e4c385785063f9` | HEAD一致、clean |

実装は上記implementation commitを基準にする。実装開始時にbranch、HEAD、dirtyを再確認し、異なる場合は差分が本specに影響しないことをレビューする。

### 2.3 現行コードで確認した事実

| component | 現行状態 | 本specで必要なこと |
|---|---|---|
| `self_supervised/moco/ddp_moco.py` | `DistributedMoCo(info_nce_impl='self'|'meta_einsum')`を実装済み。既定`self`。modeはplain attributeでstate_dict外 | 算術本体は変更しない。通常利用に合わせてdocstringだけ更新可能 |
| `standalone_moco.py` | `moco_logits`はsum/matmul。Standalone/CIFAR/動画系から利用 | 変更しない |
| `conf/moco_imagenet_parity.yaml` | `diagnostic.info_nce_impl: self`のみ | 通常selector、nullableなlegacy診断selector、監視cadenceを追加 |
| `moco_imagenet_parity.py` | `meta_einsum`をA1の2profileだけに制限。`config_identity`にmodeなし。Self checkpoint/sidecarにもmodeなし | resolver、Gate検証、全rank照合、mode-aware checkpoint、監視負荷分離、schema拡張 |
| `bench_loop` | 毎step・全rankで全Query parameterをcloneし、grad/update normを計算。step前後にCUDA sync | cadence制御し、既定でupdate cloneを無効化。bench timingは窓単位で集計 |
| `compare_moco_parity.py` | old `diagnostic` modeは読むが、通常formal比較でcandidateが`meta_einsum`か必須検査しない | effective mode readerとformal pairを追加。旧pair/A1 schemaは維持 |
| A1 judge 2種 | 1step/16stepの既存A1契約を判定 | 変更しない。新comparison schemaで既存testが通ることだけ確認 |
| `utils/checkpoint.py` | 一般分類runner用。Parity runnerは独自checkpointを使用 | 変更しない |

## 3. 採用済みB1〜B12との対応

| ID | 本specの固定契約 |
|---|---|
| B1 | ResNet50＋2GPU DDPの`DistributedMoCo`と専用runnerだけを対象にする |
| B2 | effective defaultは`self`。`meta_einsum`は明示opt-inのみ |
| B3 | `moco.info_nce_impl`を通常selector、`diagnostic.info_nce_impl`をA1互換selectorとする。第5節の衝突規則でfail-fast |
| B4 | 通常selectorはA2/A3/A4 parityとA5 benchで利用可能。A1は既存診断契約を維持 |
| B5 | modeをconfig identity、native checkpoint metadata、rank sidecar、manifest、provenanceへ保存。不一致Resume禁止 |
| B6' | 通常runはMetaを起動せず、traceを保存せず、軽量監視だけ。正式Parityは独立run＋オフライン比較 |
| B7 | A1閾値を他Gateへ継承しない。Gateごとのfixed閾値なしではformal PASS/FAILを出さない |
| B8 | 実画像fixture、2-view、初期state、rank、Shuffle-BN、Queue、GPU/VRAMをA2/A5の事前Gateで確認。OOM時に条件を変えない |
| B9 | A1、Standalone/CIFAR、state_dict key、旧artifactを維持。schemaはversion付きで拡張 |
| B10 | selector、伝搬、rank、Checkpoint、CPU DDP、比較器、A1/CIFAR非回帰を自動test化 |
| B11 | Engineering→CPU→A2→A5→別承認後A3/A4の順。長時間runへ自動進行しない |
| B12 | spec承認、実装、CPU test、commit/push、各GPU Gateを別権限とし、異常時停止・無上書きを守る |

## 4. Scopeと対象外

### 4.1 実装scope

- 通常用とA1診断用selectorの解決・拒否。
- `DistributedMoCo`へのeffective mode伝搬と全rank一致確認。
- mode-aware config identity、Self native checkpoint、全ranksidecar、Resume。
- run/provenance/Comet/checkpoint/comparisonのversion付きmetadata。
- A2〜A4の正式comparisonでcandidateのeffective modeを必須検査するpair。
- `bench_loop`の軽量監視cadenceと高コスト計測のopt-in化。
- 関連CPU unit/integration/gloo testsと既存201件の非回帰。

### 4.2 明示的な対象外

- `StandaloneMoCo`、CIFAR-10、動画Streaming MoCo、Frozen ViT、LoRA、Sequential Loaderの挙動変更。
- InfoNCE算術の再実装、Meta cloneの変更、SyncBatchNorm、AMP/TF32の導入。
- 新しい10epoch training runner、ImageNet学習、Linear Probe、表現品質評価。
- A2〜A5の許容差、full trace step、実験回数の確定とGPU実行。
- 既存artifact/checkpointの書換え、巨大trace/checkpointのResearch Workspaceへの保存。

本specで「通常学習」と呼ぶものは、現行runnerの実画像`bench_loop`が行うResNet50 DDPの学習stepと、その将来の再利用可能な軽量監視部分である。10epoch等の新しい長期学習entrypointは別specとする。

## 5. Config selectorの決定契約

### 5.1 新しい設定

```yaml
moco:
  queue_size: 65536
  momentum: 0.999
  temperature: 0.2
  info_nce_impl: self  # self | meta_einsum。通常A2〜A5用。既定はself

diagnostic:
  info_nce_impl: null  # null | self | meta_einsum。A1互換selector。旧configのselfは限定的に再生可能
```

`moco.info_nce_impl=self`は明示されたYAML既定値であり、未指定時にもeffective modeは`self`になる。`diagnostic.info_nce_impl`を`null`へ変更しても、旧artifactは変更せず、旧A1 CLIの明示override `diagnostic.info_nce_impl=self|meta_einsum`を受理する。旧A2〜A5の`resolved_config.yaml`には従来defaultの`diagnostic.info_nce_impl=self`が含まれるため、下表の同値なlegacy aliasとして再生可能にする。

### 5.2 単一resolver

runner内に副作用のない`resolve_info_nce_selection(cfg)`を1つ置き、validation、adapter、report、Comet、checkpoint、comparator testの期待値をそこへ集約する。戻り値は次の形とする。

```json
{
  "configured_normal": "self",
  "configured_diagnostic": null,
  "selector_source": "moco",
  "effective": "self",
  "legacy_compat": false
}
```

Meta参照runはモデル側selectorを使わないため、`selector_source=meta_reference`、`effective=meta_reference`として記録する。設定ファイル上の`moco.info_nce_impl`は`self`のままにし、Meta公式の実装方式を偽って`meta_einsum`と記録しない。

### 5.3 解決・拒否表

| implementation / Gate | normal selector | diagnostic selector | 結果 |
|---|---|---|---|
| `self`, A2/A3/A4 parity | `self`または`meta_einsum` | `null` | normal値をeffective modeとして許可 |
| `self`, A5 bench | `self`または`meta_einsum` | `null` | normal値をeffective modeとして許可 |
| `self`, A2〜A5 | `self` | `self` | 旧resolved config再生用の同値aliasとして許可。effective=`self`、`legacy_compat=true` |
| `self`, exact A1 1step/16step profile | `self` | `self`または`meta_einsum` | diagnostic値をeffective modeとして許可。既存A1契約 |
| `self`, A1 | `meta_einsum` | 任意 | 拒否。A1 S1はlegacy診断selectorを使う |
| `self`, A2〜A5 | `meta_einsum` | non-null | 拒否。通常metaとlegacy診断を混在させない |
| `self`, A2〜A5 | `self` | `meta_einsum` | 拒否。診断metaはA1限定 |
| `self`, A1 profile | `self`以外 | non-null | 値が同じでも拒否。selector sourceを一意にする |
| `meta_reference` | `self` | `null` | effective=`meta_reference` |
| `meta_reference` | `self` | `self` | 旧resolved config再生用aliasとして許可し、effective=`meta_reference`、`legacy_compat=true` |
| `meta_reference` | `meta_einsum`またはdiagnostic `meta_einsum` | 任意 | 拒否。自作meta selectorを公式runへ付与しない |
| 任意 | 未知の文字列 | 任意 | spawn前に`ValueError` |

`selector_source`は通常値で`moco`、A1互換値で`diagnostic`、Selfの旧`self/self` aliasで`legacy_diagnostic_self`、Meta参照で`meta_reference`とする。legacy aliasはmodeを変えず旧resolved configの再生だけを許すもので、新規configの推奨形にはしない。

拒否は`resolved_config.yaml`保存後、process spawn、モデル構築、Queue/EMA更新、Comet開始より前に行う。errorには両selector、implementation、run_kind、gate、許可された組合せを含める。

### 5.4 Hydra例

```bash
# 既定の従来方式（A5）
python -m scripts.parity.moco_imagenet_parity run_kind=bench gate=A5

# 通常A5でMeta算術を明示opt-in
python -m scripts.parity.moco_imagenet_parity \
  run_kind=bench gate=A5 moco.info_nce_impl=meta_einsum

# 既存A1 S1。normal selectorは既定selfのまま
python -m scripts.parity.moco_imagenet_parity \
  implementation=self run_kind=parity gate=A1 \
  diagnostic.info_nce_impl=meta_einsum \
  moco.queue_size=64 training.local_batch_size=4 data.image_size=64 \
  training.max_steps=16 trace.full_steps='[1,8,9,16]' \
  verification.threshold_status=fixed verification.rtol=1e-5 verification.atol=1e-6 \
  logging.comet_enabled=false

# 拒否例：通常selectorとA1診断selectorの二重指定
python -m scripts.parity.moco_imagenet_parity \
  gate=A1 moco.info_nce_impl=meta_einsum diagnostic.info_nce_impl=meta_einsum
```

## 6. モデル・全rank・実行時identity

### 6.1 伝搬

`SelfAdapter.build()`はresolverの`effective`だけを`DistributedMoCo(..., info_nce_impl=...)`へ渡す。呼出側がconfigの2fieldを個別に読み、別々の優先規則を持つことを禁止する。

`DistributedMoCo`の算術、default、`INFO_NCE_IMPLS`、state_dict keyは変更しない。通常利用へ拡張されたことが分かるようmodule docstringの「A1 diagnostic only」は更新してよい。

### 6.2 全rankの整合性

model構築後、最初のforward・Comet metric・checkpoint loadより前に、各rankが次をall-gatherする。

- resolverが返したeffective mode。
- `adapter.model.info_nce_impl`（Meta参照は`meta_reference`）。
- selector source。

全rankの3値が同じで、config effectiveとmodel attributeが一致する場合だけ続行する。不一致はrankを列挙した設定エラーとして全rankで停止する。run reportの`info_nce.per_rank`に観測値を保存する。

### 6.3 config identity v2

`config_identity()`をversion付きにし、既存fieldに次を追加する。

```json
{
  "schema_version": "moco_imagenet_config_identity/v2",
  "info_nce": {
    "configured_normal": "meta_einsum",
    "configured_diagnostic": null,
    "selector_source": "moco",
    "effective": "meta_einsum",
    "legacy_compat": false
  }
}
```

`training.max_steps`、trace、logging、gateは従来どおりResume identityから除外できる。modeとselector sourceは除外してはならない。同じ演算でもnormal/diagnostic sourceが異なるcheckpointを通常Resumeで混ぜない。

## 7. Checkpoint・Resume契約

### 7.1 新規schema

- `SELF_CHECKPOINT_FORMAT = moco_imagenet_ddp_checkpoint/v2`
- `SIDECAR_FORMAT = moco_imagenet_parity_sidecar/v2`
- config identityは`moco_imagenet_config_identity/v2`

Self native checkpointはmodel/optimizerを従来どおり保持し、top-level `metadata`へ次を追加する。

```json
{
  "effective_info_nce_impl": "meta_einsum",
  "selector_source": "moco",
  "config_identity_schema": "moco_imagenet_config_identity/v2",
  "config_identity_sha256": "..."
}
```

各rank sidecarにも同じ4fieldを保存する。Meta公式native checkpoint形式は変更せず、公式のeffective modeとidentityはsidecarへ保存する。checkpoint manifestはnative/sidecarのschema、mode、hashを記録する。

### 7.2 新形式Resume

model state、optimizer、RNGを変更する前に次を検証する。

1. native metadata、当該rank sidecar、現在config identityのschema/hashが一致。
2. native/sidecar/current resolverのeffective modeとselector sourceが一致。
3. 全rank sidecarのeffective modeが一致。
4. implementation、fixture SHA、global step、rank、checkpoint formatが一致。

どれか1つでも違えば通常Resumeを拒否する。`load_state_dict`、optimizer load、RNG restoreへ進まない。rankごとの検査エラーはcollective-safeに集約し、片rankだけがraiseして他rankをbarrier待ちに残さない。

### 7.3 mode未記録のlegacy Self checkpoint

新設定を追加する。

```yaml
training:
  legacy_resume_info_nce_impl: null  # null | self | meta_einsum
```

既定`null`では、mode metadataを持たないSelf checkpoint/sidecarを**拒否**する。再開を許すのは次をすべて満たす場合だけである。

1. `legacy_resume_info_nce_impl`が明示され、現在のeffective modeと一致する。
2. checkpointと同じrun directoryの旧`resolved_config.yaml`が存在する。
3. 旧configの`diagnostic.info_nce_impl`から読めるmodeが明示値と一致する。未知・欠損・矛盾は拒否する。
4. 旧v1アルゴリズムで計算したconfig identity hashと旧sidecarのhashが一致する。
5. fixture、implementation、rank、step等の従来条件も一致する。

許可時も通常Resumeと区別し、`run_report.resume.legacy_mode_assertion`、provenance、summaryへ明示値と旧config SHA-256を記録する。値を`self`と自動推定しない。Meta公式の旧native checkpointは演算方式が固定なので、旧sidecar identityをv1規則で検査し、`effective=meta_reference`として扱う。

### 7.4 state_dict互換

`DistributedMoCo.state_dict()`のkeyを増減しない。InfoNCE modeをbuffer/parameterへ入れない。したがって既存model stateのstrict loadを維持し、実験identityはcheckpoint top-level metadataとsidecarで保証する。

## 8. B6'：通常run、軽量監視、Parityの負荷分離

### 8.1 責務

| 経路 | Meta追加実行 | full trace | model/grad比較 | 主目的 |
|---|---:|---:|---:|---|
| 通常A5 bench / 将来の学習 | しない | しない | しない | 自作1runの学習と軽量監視 |
| A2/A3/A4 parity run | 各implementationを別process・別directoryで実行 | Gate specで指定したstepだけ | 各run内trace採取 | 公式/自作の比較材料生成 |
| offline comparator | GPU modelを実行しない | 保存済みtraceを読む | run終了後にCPU比較 | formal判定report生成 |

`parity_loop`の`PhaseTracer`はparity専用のままとし、benchへ登録しない。通常runからMeta adapterを同時構築したり、公式parameterを毎stepコピーしたりしない。

### 8.2 軽量監視の初期値

```yaml
monitor:
  enabled: true
  scalar_every_steps: 10
  grad_norm_every_steps: 100
  update_norm_every_steps: 0
  log_first_step: true
  log_last_step: true

bench:
  warmup_steps: 10
  measurement_window_steps: 10
```

- scalar監視：rank0でloss、positive/negative cosine mean、Queue pointer、LRを記録する。Selfでは`real_key_count`も記録できるが、Metaとの共通metricとは分離する。
- grad norm：rank0のQuery gradient normだけを指定stepで計算する。既定は100stepごと。
- update norm：全Query parameterの更新前cloneが必要なため既定無効。正のintervalを明示したstepだけ、rank0でclone・計算する。
- `enabled=false`はscalar/grad/update metricとComet metric送信をすべて止める。ただしrun開始時のmode/provenance、終了status、failure artifact、モデル既存の有限性・Queue invariant checkは止めない。
- interval `0`はその高コストmetricだけを無効にする。`log_first_step`/`log_last_step`はenabled時に適用する。

### 8.3 `bench_loop`の変更

- 現行の毎step・全rank `query_before = clone(all query params)`を削除し、update norm対象stepのrank0だけで行う。
- grad/update normを`step_metrics`の必須fieldからoptional fieldへ変える。未計測を0で偽装せず、fieldを省略する。
- console historyとCometは監視対象stepだけ記録する。学習step数は別counterで常に正確に保持する。
- A5 timingは`measurement_window_steps`単位でCUDA eventまたは同等の窓計測を行い、毎stepのhost `_sync`を必須にしない。窓末で同期し、step time、images/sec、data waitを集計する。
- peak allocated/reservedはrun単位で記録する。通常metric cadenceのためだけに毎step取得しない。

### 8.4 維持する異常検出

`DistributedMoCo`が現在行う入力shape/dtype、Queue shape/finite/unit norm/pointer、q/k/logits/loss finite、rank間failure伝搬は本specで緩和しない。loss非finite、OOM、NCCL error、Queue不整合は監視intervalに関係なく停止する。

勾配norm・update normは診断metricであって唯一の安全検査にしない。run終了時にはQuery parameter、Queue、optimizer momentumのfiniteを確認し、最終step直後の破損を見逃さない。

### 8.5 overheadの測定

A5のGPU許可後、同一commit・hardware・data order・seed・modeで次のfresh pairを実行する。

1. `monitor.enabled=false`
2. 上記既定light monitor

両runでwarmup、measurement window、step数を同じにし、median/p90 step time、global images/sec、data wait、peak allocated/reservedを比較する。次をreportする。

```text
overhead_percent = 100 * (light_median_step_time / off_median_step_time - 1)
```

実測前に許容overhead値を捏造しない。本specの実装PASSは「off時に高コストclone/normが呼ばれず、paired reportが生成できること」までとし、数値的なoverhead採否はA5の事前承認済みGateで決める。

## 9. 正式Parity、比較器、Evidence

### 9.1 独立runとoffline comparison

Meta Mと自作S1は同一processで並走させない。同一fixture・条件から別directoryへ独立実行し、両run終了後に`compare_moco_parity.py`が保存traceをCPUで比較する。比較器はmodelを構築せず、GPUを使用しない。

### 9.2 effective mode readerとschema互換

run schemaを`moco_imagenet_parity_run/v3`へ上げ、次を追加する。

```json
"info_nce": {
  "configured_normal": "meta_einsum",
  "configured_diagnostic": null,
  "selector_source": "moco",
  "effective": "meta_einsum",
  "per_rank": ["meta_einsum", "meta_einsum"],
  "consistent": true,
  "legacy_compat": false
}
```

既存`diagnostic` blockはA1 reader互換のため削除しない。新readerは`info_nce.effective`を優先し、旧v1/v2 artifactでは`diagnostic.info_nce_impl`へfallbackする。Meta旧artifactはimplementationから`meta_reference`と解決する。fallback使用はreportに`mode_source=legacy_fallback`として残す。

comparison schemaはversionを上げても、既存の`reference.info_nce_impl`、`candidate.info_nce_impl`、A1 control fieldを維持する。A1 judge 2種を変更せず既存synthetic report testがPASSすることを必須とする。

### 9.3 formal pair

既存`implementation`、`rerun`、`resume`、A1 pairの意味を変えず、新規pair kindを追加する。

- `formal_implementation`：Meta reference対Self `meta_einsum`。A2/A3用。
- `formal_resume`：同じSelf `meta_einsum`のuninterrupted対resumed。A4用。

formal pairは次をpreconditionとする。

1. `info_nce.consistent=true`、全rank mode一致。
2. candidate effective modeが`meta_einsum`。`formal_resume`は両runとも`meta_einsum`。
3. config/provenance/checkpoint identityのmodeがrun reportと一致。
4. fixture、commit/dirty、environment、rank、backend、入力、初期state等の既存preconditionが成立。
5. Gate別specで事前固定したrtol/atol/statusが両runとCLIで一致。

mode不明、不一致、`self`方式はformal PASS/FAILを出さず`INVALID`とする。通常`implementation` pairでは従来どおりS0を診断比較できる。

### 9.4 判定の区別

- shape、dtype、step/rank/phase、hash、targets、Queue pointer等の離散状態：exact。
- float：Gateごとの事前固定`atol + rtol * abs(reference)`。
- bitwise：許容差判定と別fieldで件数・割合・first differenceを表示。
- 欠損、NaN/Inf、mode/provenance違反：数値を読む前にINVALID。
- provisional閾値：診断reportは作れるがformal PASS/FAILはBLOCKED。

A1用fixed `rtol=1e-5, atol=1e-6`はA1にだけ適用する。A2/A3/A4は判定run前の別specで閾値・full trace step・比較対象を固定する。

### 9.5 artifact

各runは既存artifactに加えてmode/schemaを記録する。

- `resolved_config.yaml`
- `run_report.json` v3
- `provenance.json`（selector、effective、per-rank、implementation file hash）
- `fixture_manifest.json`
- `resource_benchmark.json`（monitor profile、cadence、timing window）
- `checkpoint_manifest.json`（作成時のみ。mode/schema/hash/legacyなし）
- `verification_summary.md`
- parity時だけ`trace/rank*/step_*.pt`

Comet parameterにはeffective mode、selector source、monitor enabled/cadence、run schemaを記録する。資格情報は保存しない。旧artifactは更新せず、schema/fallbackにより識別する。出力directoryは常にfreshとし、既存fileがあれば停止する。

## 10. 変更予定ファイル

### 10.1 変更する

| file | 変更 |
|---|---|
| `conf/moco_imagenet_parity.yaml` | normal/diagnostic selector、legacy resume指定、monitor cadence、bench timing window |
| `scripts/parity/moco_imagenet_parity.py` | resolver、Gate validation、SelfAdapter伝搬、rank照合、schema v3、checkpoint/sidecar v2、legacy Resume、monitor/bench負荷分離、Comet/provenance |
| `scripts/parity/compare_moco_parity.py` | effective mode reader、旧schema fallback、`formal_implementation`/`formal_resume`、mode precondition、schema拡張 |
| `test/model/test_moco_parity_harness.py` | selector表、artifact、Checkpoint、monitor、comparator、A1互換test |
| `test/model/test_moco_ddp.py` | 両modeのCPU gloo伝搬・rank整合、既存数式/Queue/DDP非回帰に必要な最小test |
| `self_supervised/moco/ddp_moco.py` | 機能変更なし。A1限定とする古いdocstringだけ通常opt-inの説明へ更新 |

### 10.2 変更しない

- `self_supervised/moco/standalone_moco.py`
- `conf/moco_standalone.yaml`およびCIFAR smoke/report。
- `scripts/parity/judge_moco_a1_info_nce.py`
- `scripts/parity/judge_moco_a1_info_nce_16step.py`
- `utils/checkpoint.py`とそのtest。
- Meta clone、動画Streaming MoCo、ViT、LoRA、Sequential Loader関連コード。

上記「変更しない」fileの変更が必要と判明した場合は、単なる局所styleでない限りscope拡張として停止し、specを再レビューする。

## 11. 自動テスト契約

### 11.1 selectorと伝搬

- 第5.3節の全caseをparameterized testにする。
- 未知値、A1でnormal meta、A2〜A5でdiagnostic metaまたはnormal meta＋diagnostic self、Meta runへのmeta selectorをspawn前に拒否。
- 旧A2〜A5およびMeta resolved configの`normal=self, diagnostic=self`はlegacy aliasとして受理し、`legacy_compat=true`を記録する。
- defaultはeffective `self`。
- A1 S0/S1の既存1step/16step profileだけを受理し、既存profile以外の`meta_einsum`を拒否。
- SelfAdapterと`DistributedMoCo.info_nce_impl`がresolver値と一致。
- CPU gloo 2rankでmode全rank一致。意図的なrank差は全rankで停止。
- ImageNet fixtureのA2〜A4 normal `meta_einsum`を受理し、旧A1診断metaはsynthetic限定を維持。

### 11.2 Checkpoint/Resume

- config identity hashはeffective modeまたはselector sourceが変わると変化する。
- v2同mode Resumeはmodel/optimizer/Queue/RNGを復元し、次step比較を通す。
- current config、native metadata、rank sidecarのどれかのmodeが違えばstate load前に拒否。
- rank間sidecar mode不一致をdeadlockせず全rankで拒否。
- legacy Selfは既定拒否。
- legacy明示値、旧resolved config、旧identity hashがすべて一致する場合だけ許可。
- legacy明示`meta_einsum`に対して旧configが`self`なら拒否。
- `DistributedMoCo.state_dict()`の既存key集合が変更前と同じ。

### 11.3 監視と負荷

- `monitor.enabled=false`でQuery parameter clone、grad norm、update norm、Comet metricsが呼ばれない。
- light既定ではscalarがstep 1、10の倍数、最終step、grad normがstep 1、100の倍数、最終stepで記録され、update normは記録されない。
- update intervalが正の対象stepだけ、rank0で更新前cloneする。
- optional metric欠損を0やPASSへ変換しない。
- timing windowごとのstep数、elapsed、throughput集計が正しい。
- NaN/Inf、Queue不整合、OOM相当failureはmonitor disabledでも停止・artifact化する。

### 11.4 比較器・Evidence

- new v3 artifactからeffective modeを読み、旧v1/v2はdiagnostic fallbackで読む。
- `formal_implementation`はMeta対Self metaだけを許可し、Self `self`、unknown、rank不一致をINVALID。
- `formal_resume`は両runのmode、Checkpoint identity、resume evidenceを必須とする。
- provisional閾値ではformal PASS/FAILを出さない。
- fixed tolerance内/外、bitwise/non-bitwise、discrete mismatch、missing trace、NaNを区別。
- 既存A1 1step/16step comparator/judgeのPASS・FAIL・BLOCKED・INVALID・NOT_RUN synthetic testsを変更せず通す。
- comparison directoryの上書きを拒否する。

### 11.5 非回帰基準

実装前commitと同じ環境で、少なくとも次の既存201件と新規testを実行する。

```text
test/model/test_moco_standalone.py
test/model/test_moco_cifar10_reporting.py
test/model/test_moco_ddp.py
test/model/test_moco_parity_harness.py
```

受入は「旧201件すべてPASS＋新規testすべてPASS、unexpected skip/xpassなし」。件数増加後も`201 passed`という古い件数を新結果として使わない。環境由来の既存failを除外する場合は、変更前commit・同一環境でも同じfailであるEvidenceが必要である。

## 12. 受入状態と停止条件

### 12.1 Engineering/CPU Gate

| status | 定義 |
|---|---|
| `PASS` | 第10節のscope内で実装され、第11節の旧201＋新規testが全PASS。研究コードclean commitは別権限 |
| `FAIL` | 有効な同一環境testでselector、mode identity、Resume、非回帰の必須条件が不一致 |
| `BLOCKED` | dependency、CPU distributed環境、既存repository差分等により有効なtestを完了できない |
| `NOT_RUN` | 実装または必須testを開始していない |
| `INVALID` | 誤commit、dirty混入、test選択漏れ、改変されたfixture等で結果を受入判定に使えない |

### 12.2 即時停止

- mode、selector source、config identity、native/sidecar、rank間の不一致。
- NaN/Inf、OOM、NCCL error/timeout、rank loss、Queue invariant違反。
- fixture/input/rank/permutation/commit/dirty/threshold契約違反。
- 既存artifactへの上書き、Meta cloneの変更、対象外fileの機能変更が必要になった場合。

batch size、K、precision、backend、許容差、monitor cadenceを失敗回避のため無断変更しない。失敗・部分artifactはfresh directoryに保存し、条件変更は別承認へ戻す。

## 13. Verification Gateの順序と権限

1. **S0 Spec approval**：ユーザーが本specを明示承認。**2026-10-19完了**（実装依頼と、承認状態の記録依頼）。
2. **E1 Implementation**：別の明示依頼後、`engineering-task`で第10節だけを実装。研究コードのcommit/pushは含まない。**2026-10-19完了**（第10.1節の6ファイルのみ、未コミット）。
3. **E2 CPU verification**：旧201＋新規test。GPUを使わない。結果をResearch Workspace `experiments/`へ記録するのは別依頼または承認後。**2026-10-19 PASS**（279 passed、[記録](../experiments/2026-10-19-meta-infonce-normal-moco-opt-in-implementation-verification.md)）。
4. **G2 A2 short parity**：ImageNet inventory、固定実画像2-view、初期state、GPU/VRAM、Gate固有閾値・traceを別specで事前固定し、GPU実行を別承認。
5. **G5 A5 overhead/VRAM**：monitor off/lightのpaired fresh bench。別GPU許可。OOM時に条件を変えない。
6. **G3/G4 A3・A4**：A2/A5のEvidence後、K65536長期QueueとResume formal runを別承認。順序や同時実行を自動化しない。
7. **将来Gate**：10epoch、表現品質、Linear Probe、Frozen ViT、動画Loader、LoRAは別spec。

Gate番号は親Spec AのA2〜A5を維持する。A3/A4を後段に回すのは採用済みB11の実行順序であり、parent Spec Aの未承認部分を承認済みに変えない。

## 14. 後方互換性

- `DistributedMoCo` constructor defaultと算術`self`を維持。
- Standalone/CIFAR/動画経路を変更しない。
- model state_dict keyを維持。
- A1の2つのdiagnostic profile、A1 comparator/judge、A1 Evidence、固定閾値を維持。
- 旧run/comparison schemaをread-onlyで読めるfallbackを持ち、旧artifactを再保存しない。
- 旧mode未記録Self checkpointは暗黙に`self`とみなさない。明示・旧config・旧hashの3重確認でのみlegacy Resume。
- 新fieldを追加しても既存A1 judgeが参照するfield名を削除・意味変更しない。
- comparison/report/benchmark outputはfresh directoryだけに作る。

## 15. Ambiguity Gateと未検証事項

### Blocking ambiguity

**実装契約上はなし。** selector優先順位、衝突拒否、legacy Resume、監視初期値、schema、変更file、受入条件は現在の採用済み方針と現行コードから固定できた。

本specはblocking ambiguityがないため承認可能と判断し、2026-10-19に承認・実装された。実装中に確定した解釈は第18節に記録する。

### Non-blocking

- helper/test関数の局所名、JSON fieldの並び、Markdownの表示順は既存styleに合わせる。
- CUDA eventの小さな実装詳細は、`measurement_window_steps`ごとの同期と同じmetric定義を満たせばよい。
- error classは既存styleに合わせられるが、エラー理由と拒否時点は本specどおりにする。

### 未検証・後続Gateで固定する事項

- A2/A3/A4のformal許容差、full trace step、正式run数。
- ImageNet実画像fixture、GPU割当、A4500上のVRAM/throughput。
- monitor off/lightの実測overhead。
- K65536のQueue一巡、A4 exact Resume、長期安定性。
- 10epoch、表現品質、Linear Probe、Frozen ViT、LoRA、Sequential Loader。

これらは本spec承認のblockingではなく、対応するGPU/研究Gateの`NOT_RUN`事項である。

## 16. Implementation Handoff（2026-10-19承認・実装済み）

- approved spec: 本ファイル（2026-10-19承認、`status: implemented`）
- 実装目的: A1で一致した`meta_einsum`をResNet50 DDPのA2〜A5通常経路で安全に明示opt-inする
- 基準repository/commit: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` / `1ce9badcbae3e2df08fa400944e4c385785063f9`
- 変更scope: 第10.1節の6file。`ddp_moco.py`はdocstringのみ
- 対象外・維持条件: 第4.2節、第10.2節、第14節
- success criteria: 第11節の旧201＋新規CPU test全PASS、mode/schema/Resume/monitor契約成立
- 許可されている短時間検証: 第11節のCPU test（2026-10-19実施済み、279 passed）
- 長時間runの許可状態: 不許可
- GPU runの許可状態: A2〜A5すべて不許可。本spec承認でも許可されない
- 未検証予定: 第15節の未検証事項

## 17. 保存・承認記録

- Research Workspace target: `haruto2919/research-workspace@main`
- base commit: `845cc7b432d11df5621ccac890fac449a4713030`
- spec path: `.research/lab/projects/sequential-video-lora-analysis/specs/2026-10-19-meta-infonce-normal-moco-opt-in-spec.md`
- 作成時点ではlocal working treeへの保存のみ。commit/pushは本依頼の権限外であり、spec自身のcommit SHAは未作成。
- 承認する場合は、ユーザーが本specの内容を明示承認した後に`status: approved`、`approved_on`、必要なら実装基準commitを更新する。承認と実装・test・commit/push・GPU実験を一括扱いしない。
- 2026-10-19: ユーザーの「このスペックに従って実装してください」で実装し、その後の「spec更新とexperiments記録まで行ってください」を受けて、`approved_on: 2026-10-19`を記録した。第10節の実装と第11節のCPU検証を完了したため`status: implemented`とした。実装基準は`1ce9badcbae3e2df08fa400944e4c385785063f9`に対する未コミット差分（`git diff`のsha256は`13bf9563…`、frontmatter参照）。研究コードとResearch Workspaceのcommit/pushは行っていない。

## 18. 実装時の解釈と既知の影響（2026-10-19）

実装は第5〜第9節の契約に従った。仕様文から一意に読めなかった点と、実装で生じた影響を記録する（詳細は[実装・CPU検証記録](../experiments/2026-10-19-meta-infonce-normal-moco-opt-in-implementation-verification.md)）。

1. **既定値変更に伴う既存test 1件の更新**: 第5.1節の`diagnostic.info_nce_impl: null`は、既存testの「既定は`self`」というassertionと直接矛盾する。このassertionを「`null`であり、effectiveは`self`」の確認へ置き換えた。第11.5節の「旧201件PASS」はこの1 assertionの更新を含む。
2. **A1 judgeとschema v3**: 変更しないA1 16step judgeは、runnerとcomparatorのschema定数をimportしている。run・comparison schemaをv3へ上げたため、新commitのjudge16はv3 artifactだけを受け付け、既存のA1 v2 artifactは再判定できない。旧commitでの判定とA1 Evidenceは変わらない。
3. **`formal_resume`のrank>0 BN buffer**: 既存`resume` pairと同じく、resume後最初のforward前のrank>0 BN bufferはinformationalとする（checkpointはrank 0だけが保存し、DDPが次のforwardで再broadcastするため）。
4. **benchのloss有限性検査**: 第8.4節に従い、loss非有限は監視間隔に関係なく毎step検査する。Meta公式モデルは内部に検査を持たないため、lossの完了を待つ小さなhost同期が毎step残る。第8.3節で除いたのは、毎stepの全device同期、Query clone、normである。
5. **A1の`diagnostic.info_nce_impl=self`**: 第5.3節の「exact A1 profile」はmeta_einsumに対する制限と解釈した。selfは既存A1契約どおり、A1の任意の設定で受理する（source `diagnostic`）。A1で`diagnostic`がnullならsource `moco`の`self`とする。
6. **legacy指定の範囲**: `training.legacy_resume_info_nce_impl`の非null値はimplementation=selfに限る（Metaは拒否）。v2 checkpointの再開ではこの値を使わない。
7. **formal pairの前提**: 両runにv3の`info_nce`（全rank一致）を要求する。ほかに、config・provenance・run reportのmode一致、clean・同一commit、`max_steps`一致、比較範囲のfull trace step一致を要求する。`formal_resume`ではreferenceがresumeしていないこと、candidateの全rankのresume証跡とcheckpoint identityの一致も要求する。違反は値を読む前に`INVALID`。
