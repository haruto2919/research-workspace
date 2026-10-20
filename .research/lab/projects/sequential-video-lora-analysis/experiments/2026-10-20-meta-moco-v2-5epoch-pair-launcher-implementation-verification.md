---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: completed
created: 2026-10-20
last_updated: 2026-10-20
spec: ../specs/2026-10-20-meta-moco-v2-imagenet-5epoch-sequential-comet-comparison-spec.md
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_base_commit: d6f7acc75ee92adcf7e6e3e65831d46759b9ed14
implementation_committed: true
implementation_commits:
  - dd0d8b3 feat(parity): add the E5 five-epoch gate and the sequential Meta/self pair launcher
  - 7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62 test(parity): cover the E5 gate and the 5-epoch pair launcher
implementation_pushed: origin/dev@7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62
implementation_new_files:
  scripts/parity/run_moco_5epoch_pair.py: c3f41f536b711f5a6034408a2ee42bd2e0673202418f2f1d31179069344e1798
  test/model/test_moco_5epoch_pair.py: 7fcfe3c48ab182322547be44fcbb57131f5eda519f993d0c974048fe0c7399be
existing_files_changed:
  - scripts/parity/moco_imagenet_10epoch.py
  - conf/moco_imagenet_10epoch.yaml (comments only)
  - test/model/test_moco_imagenet_10epoch.py
engineering_cpu_gate: PASS
cpu_tests: "108 passed, 1 warning in 95.43s (test_moco_imagenet_10epoch 60 + test_moco_gate_launcher 23 + test_moco_5epoch_pair 25)"
five_epoch_gpu_run: NOT_RUN
comet_real_send: NOT_RUN
reference: facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
---

# MoCo v2 ImageNet 5epoch逐次ペア（Meta公式→自作）：実装とCPU検証

2026-10-20のユーザー依頼「このスペックに従って実装してください」に基づき、[5epoch逐次比較spec](../specs/2026-10-20-meta-moco-v2-imagenet-5epoch-sequential-comet-comparison-spec.md)§4の最小変更（E5 Gate、逐次ランチャー、テスト）を研究コードへ実装し、CPUで検証した。**`--execute`、5epoch GPU学習、Comet実送信は行っていない。**

**追記（同日）**：ユーザーの「commit/pushして良いです」を受け、`dev`へ`dd0d8b3`（feat）と`7fc1f56`（test）をcommitし、`origin/dev`へpushした（`d6f7acc..7fc1f56`）。新規2ファイルのSHA-256はfrontmatterの値と一致。clean な`7fc1f56`で計画モードを再実行し、**issuesなし**（研究repo clean、公式clone `a419ef2` clean、config契約・ImageNet train確認PASS）、何も書き込まれないことを確認した。GPU空き確認とComet preflightは`--execute`時にだけ行うため未確認。以下の本文中の「未commit」「repoがdirty」は実装直後の状態の記録として残す。

## 環境と基準

- 実装root：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`、branch `dev`
- 開始時HEAD：`d6f7acc75ee92adcf7e6e3e65831d46759b9ed14`（specの`source_code_commit_at_review`と一致、clean）
- Python 3.12.3、torch 2.14.0+cu130、hydra 1.3.7、omegaconf 2.3.1、pytest 9.1.1、comet_ml 3.58.6（`.venv`）
- 公式clone：`a419ef2`、`git status`差分なし
- 変更前ベースライン：`test_moco_imagenet_10epoch.py`＋`test_moco_gate_launcher.py`で**75 passed**

## 変更内容

| ファイル | 変更 |
|---|---|
| `scripts/parity/moco_imagenet_10epoch.py` | `GATES`に`E5`を追加。`GATE_EPOCHS={"E5": 5}`と`adopted_recipe(gate)`を追加し、GPU runの採用recipe照合をgate別にした（E5だけ`training.epochs=5`、V1/E1/E2/E3は従来どおり10）。`verification_summary.md`の見出しと「短縮run」表記を実際の`training.epochs`から出すように変更（10epochのrunでは出力が従来と同一） |
| `conf/moco_imagenet_10epoch.yaml` | コメントのみ（gateの説明にE5、epochsの説明にE5=5）。値は無変更 |
| `scripts/parity/run_moco_5epoch_pair.py`（新規、540行） | 逐次ランチャー |
| `test/model/test_moco_imagenet_10epoch.py` | E5 Gateのテスト8件を追加、CPU runテストに要約見出しの確認を追加 |
| `test/model/test_moco_5epoch_pair.py`（新規、521行） | ランチャーのCPU/mockテスト25件 |

MoCoモデル、InfoNCE、optimizer、data transform、比較器（`compare_moco_10epoch.py`）、`run_moco_gates.py`は変更していない。新しい依存関係もない。ランチャーは`run_moco_gates.py`の`gpu_check`（nvidia-smiだけで「CUDA_VISIBLE_DEVICESが空きGPUちょうど2枚」を確認）などの関数をimportして使うが、A3/A4/A5のstage定義には手を入れていない。

### ランチャーの動作

- **既定（引数なし）**：計画を表示するだけ。何も書き込まず起動もせず、GPU（nvidia-smi）にもCometにもアクセスしない
- **`--execute`**：新しいcampaign `moco_v2_e5_<YYYYmmdd_HHMMSS>`を作り、次の順で既存CLIを1つずつsubprocessで実行する

```text
<root>/moco_v2_e5_<timestamp>/        # 既定root: log/moco_imagenet_5epoch（git-ignore済み）
├── initial_state/    build_moco_parity_fixture fixture.kind=synthetic training.local_batch_size=128
├── meta_reference/   moco_imagenet_10epoch implementation=meta_reference（Comet Experiment A）
├── self/             moco_imagenet_10epoch implementation=self（同じoverrideとfixture、Comet Experiment B）
├── comparison/       compare_moco_10epoch --reference meta_reference/ --candidate self/
├── logs/             各stepのstdout/stderr
├── pair_manifest.json
└── summary/pair_summary.{json,md}
```

- 両runに渡すoverride：`gate=E5 training.epochs=5 training.max_steps=null training.stop_after_epoch=null training.resume=null logging.comet_enabled=true logging.online_check=true`、共通の`initial_state.fixture_path`、runごとの`hydra.run.dir`
- 子プロセスには`PYTHONDONTWRITEBYTECODE=1`を渡す（V1でMeta cloneに`__pycache__`が残り、自作runがpreflightで止まった事象の再発防止）

### Fail-closedの条件

| 時点 | BLOCKED/FAILになる条件 |
|---|---|
| `--execute`開始時（何も書き込まない） | rootがHydra override値に使えない文字を含む、またはrepo内でgit-ignoreされていない／campaign dirが既に存在／計画configが契約（E5 recipe、fresh、Comet ON、16 workers、seed0、NCCLなど）と違う、またはMeta/自作の差がimplementation以外にもある／研究repoがdirty／公式cloneがpinned commitでないかdirty／ImageNet trainがない／GPUが「空きのちょうど2枚」でない／Comet preflight（宛先・API key・online workspace確認）が失敗 |
| 各GPU run直前 | GPUが空いていない、研究repoのcommitがcampaign開始時から変わった・dirty、公式cloneがdirty → BLOCKED（そのrunは起動しない） |
| Meta run後 | exit≠0、またはexit 0でも：resolved configが計画と違う、fresh開始でない（resume、開始epoch/step、fixture SHA）、`5×steps_per_epoch`stepに達していない、epoch0..4のいずれかが未完了か canonical state SHAなし、checkpoint 5個・各rank sidecarが揃わないかcheckpoint SHAが記録と違う、InfoNCEモード不一致、commit不一致、Comet errorsあり・送信停止・新規Experimentでない・宛先違い → FAIL、**自作runは起動しない** |
| 自作run後 | 上と同じ確認に加え、CometのExperiment keyがMetaと同じならFAIL → **比較しない** |
| 比較後 | `contract_valid / both_completed / inputs_identical / metrics_bitwise_equal / epoch_states_identical`のいずれかがTrueでない、epoch行が0..4でない、いずれかのepochでcompleted・LR一致・canonical state一致が揃わない → FAIL。**`curve_agreement=bitwise`だけでは合格にしない** |

自動リトライ、条件変更（batch・AMP・worker・Queue）はしない。止まった場合も成果物は残す。

## Specとの対応

| spec | 対応 | 確認したテスト |
|---|---|---|
| §4.1 E5 Gateだけが5epochを受理、旧V1/E1/E2/E3は10epochのまま | `adopted_recipe(gate)` | 全5 gate×両実装で、E5は5のみ受理・10を拒否、他gateは10のみ受理・5を拒否。旧gateのrecipeが`ADOPTED_RECIPE`と同一。E5との差は`training.epochs`だけ |
| §2 cosine horizon=5 | 両実装とも`training.epochs`からhorizonを取る | 公式`main_moco.adjust_learning_rate`（args.epochs=5）のLRがepoch0..4で`epoch_cosine_lr(0.03, e, 5)`と一致。epoch0は10epoch scheduleと同じ0.03、epoch1以降は低い |
| §2 E2 checkpointからresumeしない | 既存`resume()`のidentity照合 | E2設定とE5設定のconfig identityの差が`training.epochs`だけで、identityが異なる（E2 checkpointの再開は既存の照合で拒否される） |
| §4.2 計画モードはGPU/Cometにアクセスしない | 既定は`plan_pair`のみ | nvidia-smi・`scripts.parity`のsubprocess・GPU確認・Comet確認・起動が一度も呼ばれず、rootも作られない |
| §4.2 両runの条件 | `RUN_OVERRIDES`と`PAIR_CONFIG` | 計画configが契約と一致、runnerの`prepare_config`が両方を受理、Meta/自作の差はimplementationだけ、fixtureはlocal batch128 |
| §3 順序・失敗時停止 | 逐次実行と各step後の確認 | Meta失敗→自作・比較なし／Meta exit0でもComet error・sidecar欠落→自作なし／自作失敗→比較なし／自作前のGPU使用中・commit変更・clone dirty→BLOCKED |
| §3 Comet別Experiment | 既存`CometLive`をrunごとに使用 | 2件の別Experiment（key別、名前`meta_reference-E5-<campaign>`と`self_meta_einsum-E5-<campaign>`）、各runの10stepが自分のExperimentにだけ記録、両URLがsummaryに入る |
| §3 Epoch state比較 | 既存比較器＋ランチャー判定 | 自作のepoch3 canonical stateだけ変えると、記録scalarがbitwise一致でもFAILし、epoch3が名指しされる。比較reportを書き換えて`epoch_states_identical=None`等にすると、`curve_agreement=bitwise`でもFAIL |
| §3 旧artifact非破壊 | 既定rootを`log/moco_imagenet_5epoch`に分離 | 既存campaignと旧10epoch artifactのSHAが不変、実repoの`log/moco_imagenet_10epoch`のファイル一覧・サイズ・mtimeも不変 |

テストでは、fixture builderと比較器の実際の`main()`、runnerの実際の`run()`（config検査・rank report集約・CometLive）を通す。偽物にしたのはrank worker（モデル・データなし）、fixtureのテンソル、Comet SDKだけ。

## 実行した検証

- `.venv/bin/python -m pytest -q -p no:cacheprovider test/model/test_moco_imagenet_10epoch.py test/model/test_moco_gate_launcher.py test/model/test_moco_5epoch_pair.py`：**108 passed, 1 warning in 95.43s**。warningは既存のComet preflightテストでcomet_ml内部が出す`SentryHubDeprecationWarning`で、今回の変更とは無関係
- `flake8 --max-line-length 120`（変更・新規4ファイル）：命名規則プラグイン（FNE系、既存コード全体に同種あり）以外の指摘なし
- 計画モードを研究サーバで実際に実行（`python -m scripts.parity.run_moco_5epoch_pair`）：exit 0、4 stepの計画を表示。`log/moco_imagenet_5epoch`は作られなかった。出たissueは「研究repoがdirty（今回の未commit変更）」の1件だけで、config契約・Meta/自作の差分・ImageNet train・公式cloneの確認は通った。空きディスク2746.8GiB

## 実施していないこと

- 研究コードのcommit/push（別途Git操作の許可が必要）。**repoがdirtyのままでは`--execute`は必ずBLOCKEDになる**
- `--execute`、5epochのGPU学習（Meta・自作）、Comet実送信、本物のMeta builderを使うfixture生成のランチャー経由実行
- `test_moco_parity_harness.py`などparity全体のテスト（今回の変更が触れないファイル）

## 実行時の目安と注意

- E2実測（1epoch）：Meta 5,092秒、自作 4,009秒、peak reserved 13,028MiB（Meta）／13,038MiB（自作）/GPU。5epoch×2runは**合計で約13時間**が目安（実reportを正とする）
- checkpointはE2実測で1個約370MB。10個とfixtureで約3.9GB
- ランチャーは途中再開機能を持たない（自動リトライ禁止のため）。端末切断で止まらない方法で起動する
- `--execute`の前に、変更をcommitしてrepoをcleanにする必要がある（ランチャーはcampaign中のcommitが変わると次のrunをBLOCKEDにする）
