---
project: sequential-video-lora-analysis
record_type: experiment-evidence
status: completed
date: 2026-10-20
experiment: meta-moco-v2-imagenet-10epoch-v1
campaign: moco_v2_v1_20261020_162206
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_commit_expected: d6f7acc75ee92adcf7e6e3e65831d46759b9ed14
official_reference_commit: a419ef282fc9885015674e77f58c59447206a27f
source_of_evidence: user-pasted-terminal-and-report-summary
official_meta_v1: PASS
self_v1: PASS
pair_v1: PASS_RESOURCE_FEASIBILITY_ONLY
comet_delivery: DISABLED_BOTH_PER_RUN_REPORT
---

# V1 VRAM feasibility：Meta公式・自作ともに30step PASS（最終結果）

> 本記録はユーザーが2026-10-20にSSH研究サーバから貼り付けた**学習コンソール・run_report抜粋**に基づく。ChatGPTはSSHサーバ上のrun artifacts自体を直接検証していない。**最終結果はMeta公式・自作のV1ともにPASS**。以下の途中経過に続き末尾の「V1最終追記」が新しい判定である。

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

---

## V1最終追記：2026-10-20 16:39 JST受領、両実装PASS

**前段の「self NOT_RUN・combined INCOMPLETE・Comet有効可能性」は新しい証拠によって解消した歴史的記述であり、現在の判定ではない。** ユーザーが研究サーバのrun summaryを再確認したことに基づく。

### 自作V1（新しい実測ログ）

- run：`log/moco_imagenet_10epoch/moco_v2_v1_20261020_162206/2026-10-20_16-34-13-790251_self_V1`
- 環境：`PYTHONDONTWRITEBYTECODE=1`設定、`CUDA_VISIBLE_DEVICES=0,1`。Metaと**同じcampaign/fixture**で再実行。
- `run_report`から`status=completed`、`verdicts.training_completed=true`、`vram_gate.status=PASS`、完了/計画 step `30/30`、`issues=[]`。
- 両rankのVRAM: peak reserved `13038.0 MiB`、headroom `7057.875 MiB`、peak allocated `12520.07861328125 MiB`。
- 実行時間 `121.4313339979999 sec`、global throughput `63.245619949514 images/s`、rank0 data wait `4.645293254929129 sec`、rank1 `4.888334309027414 sec`。
- Step 1/10/20/30：Metaとlossは`6.441793441772461` / `7.939069747924805` / `8.579168319702148` / `8.940801620483398`で一致。同じstepのpositive/negative cosine、hardest-negative cosine、queue_ptr（256,2560,5120,7680）、LR 0.03、有限性、step1/30のquery grad normの表示値も一致。**サンプリングしたscalarログ値の一致**であって全パラメータbitwise一致の証明ではない。
- 自作reportのComet preflight enabled=false、Experiment keyなし、error count0。

### Meta公式のComet状態についての訂正

- 前段ではMeta起動コマンドに`logging.comet_enabled=false`が視覚上省略されているため送信可能性を示したが、ユーザーが**実際のrun_reportから確認**したところ`preflight.comet.enabled=false`、Experiment keyなし、error count0。
- **両実装でComet OFFが正式な実行結果により確認された。** V1のComet-OFF契約を満たし、前段で示した「Comet OFFのためMetaを必ず取り直す」は**不要**。
- なぜMetaがOFFであったか（実際のoverride/環境の反映過程）は提示ログのみでは不明。V1を成立させる必要条件であるOFF自体は確認済み。

### V1最終比較

| 計測指標 | Meta公式 | 自作 |
|---|---:|---:|
| `status`, `training_completed`, `vram_gate.status` | `completed / true / PASS` | `completed / true / PASS` |
| completed/planned steps | 30/30 | 30/30 |
| rank0 peak reserved | 13026.0 MiB | 13038.0 MiB |
| rank1 peak reserved | 13026.0 MiB | 13038.0 MiB |
| rank0/1 headroom | 7069.875 MiB | 7057.875 MiB |
| rank0/1 peak allocated | 12525.765 MiB | 12520.079 MiB |
| total elapsed epoch partial | 205.156 s | 121.431 s |
| global images/sec | 37.435 | 63.246 |
| data wait rank0 | 74.575 s | 4.645 s |
| data wait rank1 | 75.899 s | 4.888 s |
| Comet preflight / Experiment / errors | OFF / None / 0 | OFF / None / 0 |
| issues | [] | [] |

### Evidence範囲と次のGate

- **V1 resource feasibility PASS（Meta + self）**：global256、2GPU、FP32、224pxで30stepが完走。両rankともheadroom≥1024MiB。旧自作preflightブロックは`__pycache__`由来で、現runでは解除された。
- **学習演算再現性の補助根拠**：提示scalarログ値が一致。同一サンプル/2view入力が等しかったかは、`metrics.jsonl`の入力sha256と`compare_moco_10epoch.py`の実行結果による**別途検証が必要**。現実の全step state bitwiseまで確立したと主張しない。
- **処理速度は未確定**：MetaのDataLoader待ち約75秒/rank vs 自作約5秒/rank。NAS/page cache・初回読み込み・worker起動など差があり得る。V1の数値から自作が1.69倍高速であるという性能主張をしない。
- **E1はまだ未許可**：次に実施するなら、公式と自作のV1 report/metricsをoffline comparatorで照合し、V1 Gate成立の最終チェック後、別のE1（Meta/Self 100step）の実行合意を取得する。E1ではComet API実送信とsampled input hash、Queue一巡有無、metric整合を検証する。E2/E3/E4も未実施。
- raw evidenceは研究サーバの`log/moco_imagenet_10epoch/moco_v2_v1_20261020_162206`。SSHローカルの物理ファイルやlogsのSHAはGitHubから直接確認不能。本Research Workspace記録はユーザー提示値の要約である。

**最終状態：pair V1 = PASS_RESOURCE_FEASIBILITY_ONLY。V1までは成功、10epoch学習や表現品質を実証していない。**

---

## V1オフライン比較の追加Evidence（2026-10-20 16:44 JST）

ユーザーが研究サーバ上で`python -m scripts.parity.compare_moco_10epoch`を実行した結果を貼り付けた。これは**GitHubから実行ファイルを読んだ結果ではなく、ユーザー提示した比較出力**の転記である。前節のV1 resource feasibility判定に、独立した**ログ比較Evidence**を加える。

- campaign: `moco_v2_v1_20261020_162206`
- Meta run: `2026-10-20_16-22-44-972339_meta_reference_V1`
- Self run: `2026-10-20_16-34-13-790251_self_V1`
- comparison output dir: `log/moco_imagenet_10epoch/moco_v2_v1_20261020_162206/comparisons/v1_20261020_164421`
- output reports: `comparison_report.json` と `comparison_summary.md`（SSHローカル。GitHubへ生ログ本体は未反映）。

### 比較器の出力

| 判定名 | 実際の値 | 解釈 |
|---|---|---|
| `contract_valid` | `True` | 比較器でのconfig、campaign、sampler等の契約照合がPASS（問題リストなし） |
| `both_completed` | `True` | 双方のrun_report statusはcompleted |
| `inputs_identical` | `True` | SHA-256を記録した1stepで2実装の入力hashが同一 |
| `metrics_bitwise_equal` | `True` | 比較対象の記録scalar値が完全一致 |
| `epoch_states_identical` | `None` | 学習途中30stepでepoch checkpoint/state digestなしのため未評価 |
| `curve_agreement` | `bitwise` | 比較器の**記録曲線上**の判定。全パラメータや全step tensorへのbitwise証明ではない |
| `contract_issues` | `[]` | 契約上の問題なし |

`metrics.jsonl`由来の一致（全てmax_abs_diff=0.0）：loss 4/4、positive_cosine_mean 4/4、negative_cosine_mean 4/4、hardest_negative_cosine_mean 4/4、lr 4/4、queue_ptr 4/4、finite 4/4、query_grad_norm 2/2。入力hashは `hashed_steps=1, identical_steps=1, first_differing_step=null`。

### Evidence解釈と境界

- 実ImageNet + 2GPU + FP32 + global256での**両実装V1 resource feasibility PASSとサンプル観測した学習scalar完全一致**は支持される。これは単なる合成画像parityより実運用条件に近い補助Evidence。
- **入力については1stepのみハッシュ照合**。残り29stepの入力tensorが同一だったことまではこのレポートは保証しない。今回の`curve_agreement=bitwise`はサンプリングしたログに対する判定であり、全step/全stateの形式的完全一致と同義ではない。
- **epoch状態は未比較**。`compare_moco_10epoch.py`の`states = [... for epoch in epochs if canonical_state_equal is not None]`が空の場合、`all(states)`は真となり得るため、`curve_agreement=bitwise`は`epoch_states_identical=None`の状況でも表示される。これを超える主張をしない。
- ImageNetの表現品質・分類精度、10epoch学習の成功、E2/E3の長時間安定性、Cometへの実送信、速度差の有意性は未検証。
- 次の研究判断は`E1`（別承認を要するMeta/Self各100step）で、`1 + every100`の追加input-hashやComet実接続、Loss/cosine/gradの再現性を確認するかどうか。E1へは**V1の承認から自動移行しない**。

**更新後の本記録の最新判定**：V1 Meta+Self GPU feasibility **PASS**、V1 sampled-input+logged-scalar comparison **PASS**、epoch-state/tensor-level formal parity **NOT_EVALUATED**。経緯にある`partial`と`INCOMPLETE`は当時の履歴として保持し、現在状態の主張と混同しない。
