---
project: sequential-video-lora-analysis
record_type: experiment-evidence
status: partial
date: 2026-10-20
experiment: meta-moco-v2-imagenet-10epoch-v1
campaign: moco_v2_v1_20261020_162206
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_commit_expected: d6f7acc75ee92adcf7e6e3e65831d46759b9ed14
official_reference_commit: a419ef282fc9885015674e77f58c59447206a27f
source_of_evidence: user-pasted-terminal-and-report-summary
official_meta_v1: PASS_WITH_COMET_CONFIG_DEVIATION
self_v1: PREFLIGHT_BLOCKED_NOT_RUN
pair_v1: INCOMPLETE
comet_delivery: UNKNOWN_META_ENABLED_BY_DEFAULT
---

# V1 VRAM feasibility：Meta公式PASS / 自作preflight停止（途中経過）

> 本記録はユーザーが2026-10-20にSSH研究サーバから貼り付けた**学習コンソール・実験reportの抜粋**に基づく。ChatGPTはSSHサーバ上のrun artifactsを直接検証しておらず、コンソールに存在しない情報を補完しない。**Meta公式のV1はPASS、自作は学習開始前に公式clone cleanチェックで停止。V1全体は未完了。**

## 承認済み契約とrun identity

- [spec](../specs/2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md)のV1：ImageNet-1K `train`、ResNet50 MoCo v2 MLP、2GPU×local128/global256、SGD initial LR0.03、FP32、K65536、Queue/EMA、`training.max_steps=30`、`logging.comet_enabled=false`、両rankのheadroom >=1024MiB、失敗時停止。
- campaign: `moco_v2_v1_20261020_162206`
- fixture: `log/moco_imagenet_10epoch/moco_v2_v1_20261020_162206/initial_state/fixture.pt`（ユーザー実行ログに示された同fixtureをMeta/Selfが参照）。
- 研究コードcommit：直前のユーザー確認で`dev@d6f7acc75ee92adcf7e6e3e65831d46759b9ed14` clean。今回のrun時点の追加local状態までは直接未確認。

## Meta公式V1：計算資源上のPASS（user-provided observed evidence）

- run: `log/moco_imagenet_10epoch/moco_v2_v1_20261020_162206/2026-10-20_16-22-44-972339_meta_reference_V1`
- `run_report.json`をユーザーが抽出した表示：`status=completed`、`verdicts.training_completed=true`、`vram_gate.status=PASS`、`last_global_step=30`、`planned_last_step=30`、`issues=[]`。
- rank 0・1とも `max_reserved_mib=13026.0`、`headroom_mib=7069.875`。計算上のGPU総メモリ約20095.875MiB。headroomの定義はこのrunプロセスのpeak reserved差引であり、他プロセスのGPU使用量を含めない。
- rank 0・1とも `epoch_max_allocated_mib=12525.76513671875`、`epoch_max_reserved_mib=13026.0`。
- 実行時間 `epoch_duration_sec=205.15626996199717`、global throughput `37.434878307266125 images/sec`（30stepのpilot実測に限る）。DataLoader待ち時間：rank0 `74.57521226906101` 秒、rank1 `75.89892828995653` 秒。**10epochの長期throughputや安定性は未確認**。
- local SGD step 1/10/20/30にloss `6.44179344/7.93906975/8.57916832/8.94080162`、finite true、queue_ptr `256/2560/5120/7680` を確認。Queue warmup中30stepのloss上昇を学習失敗や性能改善の根拠にしない。step30 `query_grad_norm=0.16113631400013403`。
- **契約差分／要確認**：Meta公式を起動したコマンドは`logging.comet_enabled=false`を指定していなかった。最新`conf/moco_imagenet_10epoch.yaml`のdefaultは`logging.comet_enabled=true`。したがって**Meta runではCometが有効だったはず**であり、Comet実experiment作成・送信成功・URLは**ユーザー提示ログから確認できない**。`run_report.json`の`preflight.comet.enabled`、`comet.url`、`comet.errors`、`resolved_config.yaml`を確認。V1条件はComet OFFなので、実行負荷を含む形式的一致が必要ならMetaをOFFで再試行する（個別run pathを保護）。

## 自作V1：学習なしでpreflight停止

- 自作の起動条件は`implementation=self campaign=同一 gate=V1 initial_state.fixture_path=同一 training.max_steps=30 logging.comet_enabled=false`。同じコマンドを2回試し、両方とも同じpreflightエラー。
- 例外：`scripts/parity/moco_imagenet_10epoch.py:189` `repository_preflight`、`RuntimeError: reference clone must be clean pinned commit`。指定cloneのcommitは**正しく**`a419ef282fc9885015674e77f58c59447206a27f`、しかし `dirty=True`。
- 唯一報告されたdirty paths：`?? moco/__pycache__/__init__.cpython-312.pyc` および `?? moco/__pycache__/loader.cpython-312.pyc`。Python import時に生成されたuntracked bytecode cacheで、ソースコードの変更ではない。Gitの`status --porcelain --untracked-files=all`でdirtyと判定する設計によりGPU trainingの前でブロック。
- `self`に`run_report.json`が存在しないとユーザーが報告。これはpreflight時点で失敗するrunnerの挙動と整合。**自作のVRAM/学習可能性をこのエラーで判断しない**。
- `PYTHONDONTWRITEBYTECODE=1`を次回runner起動時に環境へ設定すると、新たなPython bytecodeキャッシュの生成を防げる。既存の上記2ファイルは消えないので、`git -C <reference clone> status --short --untracked-files=all`で他の差分がないことを確認してから**この2ファイルだけを削除**。広域`git clean -fdx`を使わず、ソース・公式commitを触らない。削除後に`git status` cleanを再検証。
- 同一campaign/fixtureで`self` V1を再実行し、`logging.comet_enabled=false`を明記。Hydraの新しいrun directoryへ保存。fail/passのreport、per-rank VRAM headroom、queue/finite、CPU/NAS wait、throughputを確認。

## 現時点の判定／Next action

| Gate | 判定 | 注意 |
|---|---|---|
| Meta V1 GPU feasibility | **PASS** | 30step完了、headroom 7069.875MiB/rank。Comet OFF契約との差異あり、ログ確認要 |
| Self V1 GPU feasibility | **NOT_RUN/PREFLIGHT_BLOCKED** | untracked Python cacheを解消するまでGPU測定なし |
| Combined V1 | **INCOMPLETE** | Meta/Self両方PASSが条件 |
| Comet | **POSSIBLY_ENABLED_IN_META** | 実送信有無・experiment URLはrun_reportで要確認 |

次のGate E1/E2/E3は**未承認・未実行**。Meta公式pilotでVRAM余裕があっても自作pilotの代わりにはならない。もしMetaのComet-ONが確認されたら、V1のComet-OFFプロトコルとの差分を明示し、strictly identical pilot data/throughput comparisonの際は別のMeta OFF runを取得する。ここでは既存artifactを削除・上書きしない。

**Authority**：ユーザー提供のrun terminal+summary > GitHub上で未確認の実機現物。GitHub codeの静的実装はrun結果そのものではない。記録は今後の自作成功後に続報追記する。