---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: completed
created: 2026-10-20
last_updated: 2026-10-20
spec: ../specs/2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_base_commit: 914e736c2699e2e129211a888d1f19013dd1dd1a
implementation_committed: true
implementation_commits:
  - "3cade3dc9c6e474e0c81595690cbee5a494e4cb2 (user: runner, comparator, tests; root .comet.config project_name)"
  - "d6f7acc (restore conf/moco_imagenet_10epoch.yaml; root .comet.config comment; Comet tests)"
implementation_pushed: origin/dev
implementation_files_at_d6f7acc:
  .comet.config: 233c05e1f9e427a29f90cb7fde91b0e822d11fb66ce9615a8a9688224ed1b7f4
  conf/moco_imagenet_10epoch.yaml: 9997481f0988a944f905db57d7c8dfd8901238661e373052de34e2910f47f8b9
  scripts/parity/moco_imagenet_10epoch.py: e36333032ef09dee8f9522321343dc489628bf3de8d3cbcac9f687da830245ad
  scripts/parity/compare_moco_10epoch.py: 1ddb98580a57cc78e2ec7683bd5c1e66537624971a20df4cd59af2f18be24108
  test/model/test_moco_imagenet_10epoch.py: f2368f222af3d93f6631a5de21a612421e6ffee71d7dba7cc677c6b74c84b80c
existing_files_changed: ".comet.config (project_name -> moco-imagenet-v2-reproduction, user decision)"
engineering_cpu_gate: PASS
cpu_tests: "new 52 passed; existing non-regression 382 passed / 3 failed (HEAD moved to 3cade3d mid-run), the 3 rerun on a fixed HEAD: passed"
gpu_gates: {V1: NOT_RUN, E1: NOT_RUN, E2: NOT_RUN, E3: NOT_RUN, E4: NOT_IMPLEMENTED}
comet_send: NOT_RUN
comet_project_creation: NOT_VERIFIED
reference: facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
reference_clone: /mnt/HDD12TB-1/ishikawa/facebookresearch-moco
---

# Meta MoCo v2 10epoch runner：実装とCPU検証

2026-10-20のユーザー依頼「このスペックに従って実装してください」（`.comet.config`はユーザー提示のテンプレートに従う）に基づき、[10epoch再現spec](../specs/2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md)の§9 Handoffにある「専用epoch-aware runner + config + minimal tests + Comet/local logging + local128 VRAM pilot gate」を研究コードへ追加し、CPUで検証した。実装前に、ユーザーはB2・B3残部・B5で推奨案を選択した（specの「B2・B3残部・B5採用記録と実装」）。**GPU実行・Comet送信は行っていない。**

## 経緯（commit）

1. 914e736（clean）を基準に、runner・比較器・テスト・10epoch用yaml・専用Comet設定を追加した（未commit）。
2. 非回帰テストの実行中（15:34）に、ユーザーがcommit `3cade3d`を作成してpushした。このcommitにはrunner・比較器・テストが含まれ、直下`.comet.config`の`project_name`が`moco-imagenet-v2-reproduction`へ変更されていた。一方、10epoch用yamlと専用Comet設定は含まれず、作業ツリーからも消えていた。
3. ユーザーは「直下`.comet.config`で管理」「commitしてpush」を選択した。yamlを`logging.comet_config: .comet.config`として復元し、直下`.comet.config`へ写り込んでいた専用ファイル用のコメントを実態に合う説明へ差し替え、テストを直下ファイル前提に更新した。`d6f7acc`としてcommitし、origin/devへpushした。

## 環境と基準

- 実装root：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`、branch `dev`
- Python 3.12.3、torch 2.14.0+cu130、torchvision 0.29.0+cu130、comet_ml 3.58.6、pytest 9.1.1（`.venv`）
- 公式clone：`a419ef2`のまま、`git status`差分なし

## 追加・変更したファイル

| ファイル | 内容 |
|---|---|
| `conf/moco_imagenet_10epoch.yaml`（91行） | 10epoch用config。採用recipe、B2 strict、B3入力整合、B5 workers16、V1閾値、Comet宛先の期待値 |
| `scripts/parity/moco_imagenet_10epoch.py`（850行） | 10epoch runner（Hydra entrypoint） |
| `scripts/parity/compare_moco_10epoch.py`（200行） | M/Sのオフライン比較器（E1〜E3用） |
| `test/model/test_moco_imagenet_10epoch.py`（523行） | CPUテスト52件 |
| `.comet.config`（既存） | `project_name=moco-imagenet-v2-reproduction`（ユーザー判断）、説明コメント |

新しい依存関係はない。`MetaReferenceAdapter`／`SelfAdapter`、拡張処理、state mapping、`step_metrics`、`BenchMonitor`、`WindowTimer`、RNG退避、fixture生成（`build_moco_parity_fixture`）は既存のものをimportして再利用し、既存のparity／A5経路のロジックには手を入れていない。

## 実装した契約（specとの対応）

### 学習ループ（§2・§4）

- epoch 0..9：`DistributedSampler(shuffle=True, seed=0).set_epoch(epoch)`、epoch開始時の`--cos` LR（Metaは公式`adjust_learning_rate`、自作は`epoch_cosine_lr`）、1epochの全batch（`drop_last=True`）、DDP・EMA・Queue、rank0のmetrics記録、epoch終了時のcheckpoint
- 初期状態：campaignごとに1回`build_moco_parity_fixture fixture.kind=synthetic training.local_batch_size=128`で作る公式builder由来のcanonical state。両実装がこれを読み込み、構築後のcanonical state sha256がfixtureと一致しなければ停止する
- 自作は`moco.info_nce_impl=meta_einsum`、Metaは公式builderの演算（`null`から実装ごとに自動設定。不一致の明示値は拒否）
- **GPU runは採用recipe（global256＝2×128、LR0.03、momentum0.9、wd1e-4、10epoch cos、ResNet50+MLP128、K65536、m0.999、T0.2、224px、FP32/AMPなし、strict）以外を起動前に拒否**する（B1：自動で条件を下げない）。CPUはgloo＋自作のテスト専用
- 停止区分：`training.max_steps`（V1/E1、途中epochはcheckpointなし）、`training.stop_after_epoch`（E2、10epochのスケジュールは維持）
- 起動前preflight：公式cloneがcleanな固定commitであること（常に）、研究repoがcleanなcommitであること（GPU run）

### 入力整合（B3）

- epoch・rankごとに、DataLoader専用generator（workerのbase seed）、CPU global RNG（shuffle-BNのrandperm）、Python `random`を`derive_seed(seed, 名前, epoch, rank)`で設定する
- 1step目と`input_alignment.hash_every_steps`（既定100）ごとに、全rankの`im_q`/`im_k`のsha256を`metrics.jsonl`へ記録する。epochごとのsampler index digestとseedも`epochs.jsonl`へ記録する

### Checkpoint・Resume（§7・B5）

- epoch終了ごとに、rank0がnative形式（Meta：`main_moco.save_checkpoint`、自作：既存形式）、全rankがsidecar（RNG、completed epochs、global step、config identity sha256）を`.tmp`経由で原子的に保存する。全epochを保持する
- `training.resume=<run>/checkpoint_epochNNN.pt`：新しいrun dirで次のepochから再開する。identity・実装・rank・epoch範囲を全rankで照合してから読み込む（collectiveのため、片方のrankだけが落ちることはない）。履歴は親segmentとつないで読む（`stitched_records`）

### ローカル記録とComet（§5・B3）

- rank0：`metrics.jsonl`（10stepごとにloss・positive/negative/hardest-negative cosine・lr・queue_ptr・finite、100stepごとにQuery grad norm、入力hash）、`epochs.jsonl`（全rank・全stepの平均loss、duration、throughput、rankごとのデータ待ち・VRAMピーク・sampler digest、checkpoint sha256、epoch終了時のcanonical state sha256）。1行ごとにflushし、Cometへ送るのはその後
- Comet preflight（spawn前・fail-closed）：`logging.comet_config`（直下`.comet.config`）を`COMET_CONFIG`として指定し、SDKが解決するworkspace/projectが`haruto2919`/`moco-imagenet-v2-reproduction`であること、APIキーが解決されることを確認する。オンライン確認では、workspaceへの到達性とprojectの事前存在を記録する。`logging.comet_project`はnullのまま（非nullなら拒否）
- ライブ送信：rank0のみ。experiment名は`<role>-<gate>-<campaign>`、tagは`moco_imagenet_v2_reproduction`・role・gate・`campaign:<id>`。作成後にも宛先を照合する。auto metric/param/weight/histogram/co2 loggingは明示的にoff（metric定義をrunのものに統一するため）。env・graph・CLI引数のlogは`.comet.config`の`[comet_auto_log]`に従う。Cometの例外は記録して送信を止めるだけで、学習は続ける。resumeは親の`comet_experiment.json`のkeyで同じexperimentを継続する

### V1 Gate（§3）

- `gate=V1 training.max_steps=30 logging.comet_enabled=false`で、両rankのmax allocated/reserved、総量、headroom、warmup後のstep時間・データ待ち・throughput、host MemAvailableの最小値を記録する
- PASS：全予定stepの完了、failureなし、両GPUのheadroom ≥ `vram_gate.min_headroom_mib`（既定1024、ユーザー未確認）。それ以外はBLOCKEDとし、理由（OOM等の例外文を含む）を記録する
- 判定は`verdicts.training_completed`・`verdicts.vram_feasible`、比較器側の`curve_agreement`等に分け、一つのPASSにまとめない

## 検証結果

### 新規CPUテスト（`test_moco_imagenet_10epoch.py`、52件すべてPASS）

- 構成：Hydraのcomposeで採用recipeを確認、parity用configは不変。`.comet.config`はテンプレートどおりでAPIキーなし。不正runの拒否（12通り）、GPU runでの条件引き下げ拒否（6通り）、InfoNCEの実装別解決
- 公式との照合（公式cloneを使用）：`reference_args`で`batch 128／workers 16／lr 0.03／epochs 10／cos／K 65536`、公式`adjust_learning_rate`と`epoch_cosine_lr`が10epochすべてで完全一致。ImageNet train 1,281,167枚・2rank・local128で**640,584件/rank、5,004 step/epoch、10epochで50,040 step**
- 実CPU学習（ResNet50、gloo 2rank、32px、K8、local2、8枚→2 step/epoch、3epoch、worker2）：
  - 3epoch完走、epoch LRがcosine値と一致、checkpoint3個とsidecar6個、sha256一致、開始時canonical state＝fixture
  - **rerunで`metrics.jsonl`の全行と各epochのloss・canonical state sha256がbitwise一致**（入力hashも全step一致）。比較器の判定は`curve_agreement=bitwise`
  - **epoch1のcheckpointからresumeすると、親とつないだ履歴が中断なしのrunとbitwise一致**
  - 既存run dirへの上書き拒否、identityの異なるresumeを全rankで読み込み前に拒否
- 比較器：最初の差分（loss step4、入力hash step5）の特定、contract違反（campaign・identity・sampler digest）で`not_comparable`、CLIの上書き拒否
- Comet：宛先解決（repo直下から起動・別ディレクトリから起動の両方）、APIキーの値がrecordに含まれないこと。fail-closed 5通り（環境変数による別project、別projectの設定ファイル、APIキーなし、既存の別`COMET_CONFIG`、ファイルなし）、オンライン確認（fake API）。ライブ送信（fake SDK）：auto logging off、名前・tag・parameter、bool→int、送信失敗時に例外を出さず停止、宛先不一致ならend、resume時の同一experiment継続、rank0のみ
- V1判定：PASS／headroom不足でBLOCKED／OOMでBLOCKED

### 非回帰

- 既存の非回帰一式（`preflight_moco_imagenet`のNON_REGRESSION_TESTS 8ファイル＋新規）：**382 passed / 3 failed**（834.98秒）
- 失敗3件はすべて`test_moco_parity_harness.py`のA1 16-step系で、理由は「self repository commit differs」。テスト実行中にユーザーのcommit `3cade3d`でHEADが`914e736`から動き、S0とS1のrun間でcommitが変わったため。**HEADを固定した状態での再実行では3件ともPASS**（新規テストと合わせて56 passed）
- flake8（repo設定、120文字）：新規ファイルに指摘なし（既存コードと同種のFNE命名警告を除く）

## 実環境での事前確認（GPU・通信なし）

- Hydraで`campaign=dryrun initial_state.fixture_path=x`をcomposeし、`prepare_config`がPASS（採用recipeを満たす。InfoNCEは`meta_einsum`に解決）
- Comet宛先の解決（オフライン）：直下`.comet.config`から`haruto2919 / moco-imagenet-v2-reproduction`、APIキーは`~/.comet.config`から解決済み（値は表示していない）
- `repository_preflight`は、未commitの新規ファイルがある状態ではGPU runを正しく拒否した。現在は`d6f7acc`でclean

## 未実施・未確立・注意

- V1/E1/E2/E3のGPU実行、Comet送信、Comet projectの作成・存在確認（オンライン）：**NOT_RUN**（それぞれ別の実行許可が必要）
- Meta adapter経路（CUDA専用）・NCCL・pin memory・VRAM計測の実機動作はCPUテストでは検証できない。V1で初めて確認する
- local128のVRAM・throughput・RAM：不明（A5-pilotはlocal32の結果）
- B1のheadroom閾値1024MiB：ユーザー未確認
- E4軽量特徴診断の実装：今回の範囲外
- **直下`.comet.config`の変更により、このrepoから明示的なproject指定なしで起動する他のComet実験（`train.py`、既存parity runなど）も`moco-imagenet-v2-reproduction`へ送られる**（ユーザー判断。spec §5の「既存projectと混在させない」と緊張関係にある）
