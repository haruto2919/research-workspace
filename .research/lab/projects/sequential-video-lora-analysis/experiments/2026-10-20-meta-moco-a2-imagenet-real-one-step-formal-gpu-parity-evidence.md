---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: experiment-evidence
status: completed
gate: A2
judgement: PASS
evidence_level: "one-step-real-imagenet-two-gpu-fp32-formal-parity"
source: user-provided-terminal-output
experiment_spec: "../specs/2026-10-20-meta-moco-a2-imagenet-one-step-parity-experiment-spec.md"
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: dev
research_commit: dc502e416aa26ed4e0791767fa5a72f7843e399d
meta_reference_repository: facebookresearch/moco
meta_reference_commit: a419ef282fc9885015674e77f58c59447206a27f
run_label: formal_a2_20261020_091930
comparison_pair: formal_implementation
rtol: 1.0e-5
atol: 1.0e-6
threshold_status: fixed
---

# A2正式Evidence：ImageNet-1K実画像・Meta MoCo v2 vs Self meta_einsum・2GPU・1step

> **Formal A2 = PASS（ユーザー提供GPU実験ログ）**。自作S1はMeta公式M1と**比較器がjudgedした3,926エントリすべてbitwise一致**。exact failures 0、float threshold exceedances 0、first difference/failure None。これはユーザーが研究サーバ上で実施した**一つのfresh M1/S1 pairのone-step結果**であり、ChatGPTがSSH上のtraceやparity_report.jsonの実体を独立に取得・検査したわけではない。

## 1. Authority・再現性の前提

- [A2承認済み実験spec](../specs/2026-10-20-meta-moco-a2-imagenet-one-step-parity-experiment-spec.md)で正式GPU判定に先立ち、ユーザーが`verification.rtol=1e-5`、`verification.atol=1e-6`、`verification.threshold_status=fixed`、整数・ハッシュ等exactを明示採用。
- [固定ImageNet fixture生成・検査](2026-10-20-meta-moco-a2-imagenet-fixture-generation-evidence.md)：64枚・2rank×localB32・224px・FP32・初期state/input hash検査がPASS。
- [Meta M0/M0r診断](2026-10-20-meta-moco-a2-reference-m0-gpu-one-step-evidence.md)：same-fixture 2 fresh 2GPU/1step rerunでjudged3,926/bitwise3,926。こちらは`provisional`の診断であり、正式M1/S1判定へ流用していない。
- 研究コード：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` `dc502e416aa26ed4e0791767fa5a72f7843e399d`。Meta公式固定clone：`facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f`。
- ユーザーのrunスクリプトで両`git rev-parse HEAD`と`git status --porcelain`のassert、fixture SHA照合、CUDA device count、GPU空きがpreflight PASSとなった。今回貼付けコマンドの一部はterminalの文字表示が崩れているが、全工程の成功出力を確認。

## 2. 固定した実験契約

| Axis | Fixed value |
|---|---|
| Dataset | ImageNet-1K train、同一fixtureの実画像64枚、各画像から固定2-view |
| Backbone | ResNet50 + MLP projector、canonical Meta初期重み・BN state・Queue |
| Image | 224×224、FP32、AMP OFF、TF32 OFF、seed=0 |
| Distributed | NVIDIA RTX A4500×2、NCCL、2rank、local batch32、global batch64、Shuffle-BN + DDP broadcast |
| MoCo | Queue K65536、EMA momentum 0.999、temperature 0.2 |
| Optimizer | SGD、初期LR0.0075、momentum0.9、weight decay0.0001、epoch0 cosine schedule |
| Steps | 1 step、step1のFull trace、fresh開始、Comet OFF |
| M1 | `implementation=meta_reference`、外部無改変Meta clone |
| S1 | `implementation=self`、通常`moco.info_nce_impl=meta_einsum`、`diagnostic.info_nce_impl=null` |
| Compare | `--pair-kind formal_implementation --gate A2 --threshold-status fixed --rtol 1e-5 --atol 1e-6` |
| 判定 | `abs(S1-M1) <= 1e-6 + 1e-5 * abs(M1)`、離散/hash/順序等exact；bitwise率は別集計 |

Fixture（ローカルSSH上）：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/fixtures/a2_imagenet_20261020_084625/fixture.pt`

Fixture SHA-256：`9edede9195eccef13e7ec8446880d6bfc6a550b11e96b466dbf047723d24b4ac`。ユーザー実行の`sha256sum -c`結果は`OK`。

## 3. 実行run・出力ディレクトリ

ユーザー提供コンソール出力：

- Root：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/formal_a2_20261020_091930`
- M1（Meta公式）：`.../formal_a2_20261020_091930/M1_meta_reference`
- S1（Self meta_einsum）：`.../formal_a2_20261020_091930/S1_meta_einsum`
- Comparator：`.../formal_a2_20261020_091930/compare`
- Result：`.../formal_a2_20261020_091930/compare/parity_report.json`
- CPU parsed outcome: `M1_DIR: VERIFIED`、`S1_DIR: VERIFIED`、`BOTH FORMAL RUNS COMPLETED`。ユーザー実行したassert群により、それぞれ`status=completed`、`completed_steps=1`、`info_nce.effective/per_rank/consistent`、`tolerance fixed`、status付きwithin-run `PASS/INFO`、rank0/rank1 full trace file存在が確認された。
- 公式M1 step1 rank0: `loss=6.47951602935791`、`positive_cosine_mean=0.9418870925903321`、`negative_cosine_mean=-0.0001348661258816719`、`queue_ptr=64`、`lr=0.0075`、`query_grad_norm=19.536392104956256`、`query_update_norm=0.1465230765670673`、`step_time_sec_with_tracing=2.0903532589945826`。
- 自作S1同一stepのmetrics：loss/cosine/queue_ptr/LR/gradient norm/update normは上記と同値、`step_time_sec_with_tracing=2.1236261960002594`。これらwall timeはtraceの複製・同期コストを含み、1step測定から速度優劣やA5性能を主張しない。

## 4. 正式比較の判定

```text
PASS: all compared tensors within the fixed tolerance; within-run checks PASS
FORMAL STATUS: PASS
Reason: all compared tensors within the fixed tolerance; within-run checks PASS
Counts: {'judged': 3926, 'exact_failures': 0, 'close_failures': 0, 'bitwise_equal': 3926}
First difference: None
First failure: None
```

| Metric | Observed | 判定 |
|---|---:|---|
| judged comparison entries | 3,926 | 有効な比較 |
| bitwise_equal | 3,926 / 3,926（100%） | PASS、判定対象全件一致 |
| exact_failures | 0 | PASS |
| close_failures | 0 | PASS |
| first_difference | None | 差分は観測されない |
| first_failure | None | 失敗は観測されない |
| Formal comparator status | PASS | **A2 PASS** |
| Run within-run checks | M1・S1両方PASS/INFOのみ | PASS |

**重要**：3,926はrank/step/phase/tensorなどの**比較エントリ数**であって、パラメータscalarの総数、ImageNetサンプル数、学習step数を意味しない。bitwise完全一致は`rtol/atol`以内という結果より強い観測だが、その一致が認められるのは**今回記録・比較されたエントリの範囲のみ**。M1/S1の実際の`resource_benchmark.json`ピークVRAMはコンソールに貼付けられておらず、本EvidenceではM0の前回VRAMを今回の測定値として代用しない。

## 5. Gate判定・未検証の範囲

### 確認された主張

- 指定したImageNet実画像64枚、224px、ResNet50、2GPU NCCL、localB32、K65536、初期fixture、FP32、1stepについて、Meta公式と自作`meta_einsum`の**既存harnessが記録・判定したForward／Backward／SGD／EMA／Queue／BN／DDPの状態**が事前固定基準内で一致した。
- 現実には固定許容差以内だけでなく、**比較エントリ3,926/3,926がbitwise一致**。正式比較器のwithin-run、形式条件・数値判定が`PASS`。
- 以前のCPU Engineering 279/279 PASS、A1 synthetic 16step bitwise PASSに、**より現実的なImageNet 224px・K65536・localB32でのA2 1step Evidence**が加わった。

### まだ主張できないこと

- 2step以上の実画像連続学習、Queue一巡の前後、長時間の勾配/損失安定性、Checkpoint/Resume exact、複数seedの再現性。
- A5の通常`run_kind=bench`のmonitor overhead・throughput・peak VRAM feasibility（今回はparity full-traceでありbenchではない）。
- MoCoによって学習した特徴量の分類識別性、ImageNet 10epochでの改善、ViT・LoRA・sequential-videoへの移植後の挙動、時間的/動的表現の獲得。
- 公式コードの全構成／一般的な環境での普遍的一致。

## 6. 次の行動（未承認の新GPU Gate）

- **第一候補：A5準備**。meta_einsum通常ベンチ実行の`monitor.enabled=false`対light条件で100〜300step程度のGPU VRAM/処理量/monitor overheadを比較するための独立A5実験specを作成し、事前Gate/同一条件・ユーザー承認後に実施する。A2 traceの2.09秒対2.12秒をA5代わりに用いない。
- **別候補：A3/A4**。Queue一巡の複数stepとGPU resume exactは別spec/別実行許可。A2 PASSだけで長時間runへ進まない。
- Research Workspace `main`への本記録作成はGitHub上のSSOTのみ更新し、研究サーバclone上の学習・checkpointやGit statusを変更しない。

本記録は、ユーザー提供の1回の正式run＋固定条件/比較器によるEvidenceであり、SSHの実際のtrace/report filesの独立閲覧はできない。
