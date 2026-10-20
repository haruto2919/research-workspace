---
project: sequential-video-lora-analysis
spec_type: experiment
status: draft
created: 2026-10-20
last_updated: 2026-10-20
title: "A5 ImageNet ResNet50 2GPU Normal MoCo GPU Feasibility and Monitor OFF vs Light Benchmark"
authority: "A2 formal PASS is approved Evidence; A5 is a distinct GPU Gate requiring separate user approval"
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: dc502e416aa26ed4e0791767fa5a72f7843e399d
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
parent_spec: 2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md
a2_evidence: ../experiments/2026-10-20-meta-moco-a2-imagenet-real-one-step-formal-gpu-parity-evidence.md
a5_gpu_gate: NOT_RUN
run_permission: NOT_GRANTED_BY_THIS_DRAFT
---

# A5 実画像 GPU feasibility・監視負荷比較実験spec（draft）

> **実験計画draft・GPU run未許可。** A2はMeta公式対自作 `meta_einsum`の実画像2GPU 1step正式数値一致（3,926/3,926 bitwise）の確認を完了した。A5はA2の重いphase traceを使わない**通常`run_kind=bench`**で、ImageNetのreal DataLoaderを使ったGPUリソース・処理速度と、monitor OFF/Lightの増分負荷を調べる。**A5を行うこと自体は、Queue一巡A3・Resume A4・10epoch・Frozen ViT等の実験許可にはならない。**

## 1. 目的・研究上の問い

ImageNet-1K（train）、ResNet50＋MLP、2GPU DDP/NCCL、local batch32/global batch64、K65536、FP32において、ユーザー環境（NVIDIA RTX A4500×2）で、自作MoCoの通常`meta_einsum`経路を長めの範囲で**VRAM内・finite値を保ちながら継続実行できるか**。また**Light monitor**を有効化したとき、monitor OFFに対してどの程度のVRAM/step time/throughput増加・低下が観測されるかを測る。

A2のM1/S1は`run_kind=parity`でtraceがあり、各1stepのstep time（約2秒）は通常学習速度の根拠として使用できない。本A5の成否は公式実装との数値一致でも、表現学習の品質でもない。

## 2. Authority・現行コードの確認

- [A2正式PASS Evidence](../experiments/2026-10-20-meta-moco-a2-imagenet-real-one-step-formal-gpu-parity-evidence.md)：同一fixtureのMeta M1/自作S1が1step終了、固定許容差内かつ3,926/3,926比較エントリbitwise一致。ただしDataLoader通常経路、速度、長期安定性、S1の実測peak GPU memoryは未検証。
- [通常InfoNCE opt-in spec](2026-10-19-meta-infonce-normal-moco-opt-in-spec.md) §8/B6'は`bench`と`parity`を分離。通常`self`はデフォルト維持、`meta_einsum`は明示opt-in。Light監視のdefaultはscalar 10step、grad norm100step、update norm 0、first/last true、warmup10、window10。
- [親Spec A（draft）](2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md)はA5をImageNet 100〜300stepのGPU資源・throughput Gateと定義し、実機runには別承認を必要とする。親文書の記載順はA3/A4→A5だが、[10/19優先順位検討](../../../../secretary/notes/brainstorm/2026-10-19-post-a1-16step-moco-parity-next-priorities.md)はA2→A5小規模feasibility→費用の大きいA3/A4検討を提案。**本draftは優先度としてA5先行を提案するだけで、親Spec A全体を変更・承認しない**。
- Research code `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` `dc502e416aa26ed4e0791767fa5a72f7843e399d` の`conf/moco_imagenet_parity.yaml`、`scripts/parity/moco_imagenet_parity.py`、`test/model/test_moco_parity_harness.py`をread-only確認済み（GitHub上の状態）。Repo直下とscripts/parityのAGENTS.md/CLAUDE.mdは見つからなかった。
- 実装の`bench_loop`は`DistributedSampler`、`ImageFolder(data.root/data.split)`、adapterのtwo-crop augmentation、rank当たり`num_workers=8`、`drop_last=true`で通常データを逐次取得する。**parityの固定画像fixtureは使用しない**。
- `run_bench_steps`は全stepでForward/Backward/SGD、finite loss check、終了時Query/Queue/momentumのfinite checkを実施。monitor off時、rank0のstep metrics/historyは空でも`completed_steps`で完了判定する。
- `WindowTimer`はwarmup後のwindow境界でだけCUDA syncしてstep timeを測る。GPUのpeak allocated/reserved、measurement windows、DataLoader wait、global images/sec、完了stepとmodeはrun artifactに保存。benchにはphase Full Traceはない。独立公式Metaを並行runしない。

## 3. Stageと候補条件（draft：採用要確認）

### Phase A5-0：preflight・短いfeasibility pilot

**提案：`20step`の`monitor.enabled=false`の自作S1 bench 1本を先に実施。** 1〜10stepのwarmup後、11〜20stepの1 windowだけを計測し、全rank completed、GPU OOM/NCCL hang/DataLoader error/NaN/Infなし、GPU peak memoryがdevice内、1 window metricsを取得できるか確認。20stepは**A5の100〜300step正式Evidenceではない**ため成功をA5 PASSとしない。承認前に実行しない。

### Phase A5-1：100step monitor OFF/Light比較（提案）

| 設定 | 値 | 固定理由 |
|---|---|---|
| Implementation | `self`＋`moco.info_nce_impl=meta_einsum`、`diagnostic.info_nce_impl=null` | A2で公式と一致した算術を通常runで使用。default`self`へ戻さない |
| run kind/Gate | `bench` / `A5` | traceなしの通常学習負荷 |
| Dataset | ImageNet-1K train、NAS read-only、ImageFolder + current two-crops | 毎step real DataLoader/augmentation |
| Backbone | random-init ResNet50＋MLP projector、dim128 | A2/親Spec Aと同じ |
| GPUs | 2 × RTX A4500、world_size2、NCCL、plain BN＋Shuffle BN | 自作S1の実機測定 |
| Data | image224、local batch32/global64、num_workers8 **per rank**、pin_memory true、drop_last true、seed0、sampler epoch0 | 2条件間で一致 |
| MoCo/SGD | K65536、m0.999、T0.2、SGD LR0.0075、momentum0.9、weight decay1e-4、cos10epoch horizonのepoch0 | A2と同一初期設計 |
| Precision | FP32、AMP OFF、TF32 OFF、cudnn_deterministic true / benchmark false | 同一backendで比較 |
| 実行 | **独立fresh 100step × 2run（OFF → Light）**、resumeなし、checkpointなし、Comet OFF | 100〜300stepの下限・短いコストで feasibility確認 |
| Timing | warmup10、window10（計9window/run、11〜100stepで90step計測） | 既存測定器仕様 |
| OFF | `monitor.enabled=false`、その他monitor cadence fieldはLightと同じ既定値のまま | 開始・終了・エラー/finite確認は維持。step metricsは無出力でも正常 |
| Light | `monitor.enabled=true`、scalar10、grad100、update0、first/last true | 現行default light profile、update normの全parameter cloneを避ける |
| Artifact | 各run fresh directory、`resolved_config.yaml`、`run_report.json`、`provenance.json`、`resource_benchmark.json`、`verification_summary.md` | 条件と出力の事後監査 |
| 設定変更禁止 | batch/K/image/precision/worker/seed/温度/EMAを、片方だけやOOM回避のために変更しない | paired runの比較可能性 |

**この100step pairはA5の「実機で動く」「監視のコストを初回推定する」ための最小Evidence。** OFF→Lightの順序効果、GPUクロック、NAS cache、DataLoader待ち、GPU競合による時間変動を排除できないので、1 pairの差だけから安定した監視overheadの一般的数値閾値PASSを断定しない。

### Phase A5-2：overhead再現性を主張する場合の条件付き追加実行（提案）

必要なら別のfresh 100step pairを逆順（Light → OFF）で実施し、run order効果を調べる。**Phase A5-2は追加200stepであり別実行許可を要する。** A5-1だけでGPU予算・メモリに十分な判断ができればA5-2を必須としない。具体的な許容overheadパーセントは現時点で恣意的に作らず、必要ならユーザーが正式run前に定義する。値を採択する基準なしに「監視overhead 5%未満でPASS」などと主張しない。

## 4. 比較するMetricsと計算

全rankのrun status、completed steps、finite/failure reason、`info_nce.effective`と`per_rank`、backend/seed/worker/Queue/opt/hparam整合、GPU別`max_memory_allocated_mib`・`max_memory_reserved_mib`、window count/step span/latency/distribution、`data_wait_sec_after_warmup`、`images_per_sec_global_after_warmup_mean`を保存。

| Metrics | 定義/解釈 |
|---|---|
| Peak allocated/reserved MiB | 両rank別のPyTorch CUDAピーク。GPU全体の使用率やOOMの代替ではない |
| Mean step (s) | measurement windowの合計wall time / measured steps。Data wait/monitor/sync込み |
| Median/p90 step (s) | **Windowごとの平均step timeを並べたmedian/p90**（100step時9window）。各個別stepのp90ではない |
| Global throughput (images/sec) | global batch64 × measured steps / window集計の経過時間。単一rankに64枚処理を二重計上しない |
| DataLoader wait (s/step) | next(iterator)のwait時間。デコード、prefetch、NAS latencyの影響を受ける |
| Monitor overhead | 同一metricについて`100*(Light - OFF)/OFF`をstep latencyとpeak VRAMに使用。throughput低下率は`100*(OFF - Light)/OFF`。値をprofile/run/rank/mean/windowに紐付けて報告 |
| Metric cadence | OFFではinterval計算を回避。Lightはrank0 scalars/grad_normのみ。両runのHistory件数が異なるのは想定どおり |

**同期タイミング**：WindowTimerはwindow境界でのみCUDA syncし、`step_time_sec_after_warmup.mean`は時間基準に使える。ただしA2の1step trace時間と混同しない。rank別step timingが若干違う場合はglobal throughputの整合のため、どのrankの値から算出したか明記し、両rankを足さない。

## 5. A5 Gate・停止条件

### A5-0 pilot 判定

- PASS（pilotのみ）：both ranks completed 20/20、`info_nce.effective=meta_einsum`全rank一致、finite/資源/measurement windowあり、batch32/K65536/224px/2GPUを維持。
- BLOCKED：GPU/VRAM不足、NCCL、NAS dataset/read、PyTorch DataLoader workers、rank失敗。OOM時の自動batch/K/image/precision/num_workers削減なし。fail report保存・診断対象を区別。
- pilot成功をもって100step pairへ**無承認で自動進行しない**。

### A5-1 100step pair 判定

- **A5 feasibility PASS**：両runとも`status=completed`、`completed_steps=100`、`info_nce.effective=meta_einsum`両rank一致、失敗なし、各rankのpeak GPU allocated/reserved有限で容量以内、warmup10後9window/90 measured steps、データ待ちとthroughputの記録があり、OFFとLightの設定差がmonitor.enabledだけである（Cometは両方OFF）。
- **A5 monitor overhead MEASURED**：両runのmean/window timing・VRAM・throughputから増減値を算出し、run順序と限界を添えて記録。**明示された許容上限がない限り「数値的にoverhead合格」とは判定しない**。
- **INVALID/BLOCKED/FAIL**：異なるmodel/mode/commit/seed/data setting、欠損artifact、non-finite、OOM/長いhang、rank未完了、0 windowsは理由を切り分け。自動fallbackなし。
- A5ではOFFでmetric historyが空になる可能性が正常なため、`len(history)==100`は要求しない。必ず`completed_steps`を判定する。benchの`verification={}`も正常であり、parityのwithin-run項目を要求しない。

### Safety / 実行境界

- 最初にユーザー側で両Git commit/dirty、ImageNet file access、CUDA_VISIBLE_DEVICES=0,1、GPU空き/他ジョブ、NAS負荷・出力先容量を再確認する。
- 研究サーバでGPU実験はユーザーが別途指示/実行した場合のみ。**本draftのGitHub保存からGPU run許可を推定しない**。
- pilotでメモリに収まっても100stepの安定性はまだ未確認。OOM、non-finite、NCCL/driver問題が起きたら新たな設定を勝手に決めず停止する。
- 測定中はGPUを可能な範囲で専有。外部ジョブや不定のNAS負荷によるtiming撹乱をログに記録する。
- 公式Meta cloneを編集・GPU benchで動かす必要はない。今回は**自作の監視コスト評価**でありMeta対Selfの正式数値比較のやり直しはしない。

## 6. 現行runnerによる実行手順案（まだGPU実行しない）

対象workdir: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`。研究コード`dev@dc502e416aa26ed4e0791767fa5a72f7843e399d`とcleanなMeta公式cloneを保持。新しい`hydra.run.dir`で各runを独立保存する。Comet OFFとし、研究サーバの`log/`以下をGit管理しない。

A5-0 pilot例（**spec・GPU run別承認後のみ**）：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m scripts.parity.moco_imagenet_parity \
  implementation=self run_kind=bench gate=A5 \
  moco.info_nce_impl=meta_einsum diagnostic.info_nce_impl=null \
  training.max_steps=20 monitor.enabled=false \
  logging.comet_enabled=false \
  bench.warmup_steps=10 bench.measurement_window_steps=10 \
  hydra.run.dir=<fresh-pilot-output-dir>
```

A5-1 OFF/Light (各runの`hydra.run.dir`は別fresh)：

```bash
# OFF
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m scripts.parity.moco_imagenet_parity \
  implementation=self run_kind=bench gate=A5 \
  moco.info_nce_impl=meta_einsum diagnostic.info_nce_impl=null \
  training.max_steps=100 monitor.enabled=false \
  logging.comet_enabled=false \
  bench.warmup_steps=10 bench.measurement_window_steps=10 \
  hydra.run.dir=<fresh-off-dir>

# Light
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m scripts.parity.moco_imagenet_parity \
  implementation=self run_kind=bench gate=A5 \
  moco.info_nce_impl=meta_einsum diagnostic.info_nce_impl=null \
  training.max_steps=100 monitor.enabled=true \
  monitor.scalar_every_steps=10 monitor.grad_norm_every_steps=100 \
  monitor.update_norm_every_steps=0 \
  monitor.log_first_step=true monitor.log_last_step=true \
  logging.comet_enabled=false \
  bench.warmup_steps=10 bench.measurement_window_steps=10 \
  hydra.run.dir=<fresh-light-dir>
```

生成reportは`run_report.json`の`resources[0]`、`resources[1]`、`completed_steps`、`ranks`、`info_nce`と`resource_benchmark.json`を読む。GPU・NAS実行環境とrun pathはユーザー側で確認する。

## 7. Ambiguity Gate / 現在の承認待ち

### Blocking（本draftをapprovedにする前）

1. **A5 Scopeと順序**：親Spec Aの旧順序 A3/A4→A5ではなく、直近のA2 PASSを受けA5 feasibilityを優先してよいか。**推奨：Yes**（1,025stepのQueue一巡より先に100stepで本番リソースの基準を取る）。A3/A4は別Gateとして維持。
2. **GPU run budget**：A5-0 pilot 20step、A5-1 OFF/Light各100step、追加逆順再現性 A5-2を必須にするか。**推奨：20step pilot→100stepペアまでを第一承認scope**。逆順2runは結果のnoiseと研究目的に応じて別判断。
3. **数値的monitor overhead合格基準**：今回まず負荷を測ることが目的か、例えば許容overhead %上限を設定するか。**推奨：feasibility PASSとoverhead実測報告を分離し、上限は設定しない**（根拠なしに閾値を作らない）。
4. **ImageNet/worker/測定条件**：`num_workers=8`/rank・NAS I/O・FP32/local32/K65536を固定し、GPU実行時にメモリ/IOが足りない場合は停止して再議論。**推奨：既存configを変えず初回実測**。
5. **実験実行の承認**：このspecを作ることと、研究サーバで2GPU×20/100/100 step runを開始することを別の行為として確認する。

### Non-blocking

- ディレクトリ名・日時suffix、terminal consoleログのファイル名は既存の命名・fresh保証を守れば実験主張に影響しない。
- 結果文書の小数点桁数は実測raw valuesの保持を前提に読みやすく丸めてよい。

### Handoff

- **採用待ちspec**：本ファイル。承認されるまでは`status=draft`のまま。
- **コード変更scope**：なし。既存runner/config/testを利用。
- **成功定義**：20step pilot成立 → 新規OFF/Light各100stepのnormal benchでVRAM/step mean・window distribution・Data wait・images/secを取得し、過度な一般化なしで比較。
- **非対象**：A3 1025step、A4 checkpoint/resume、10epoch、Meta/自作の長時間数値一致、Linear Probe、Frozen ViT/LoRA/sequential loader。
- **GPU status**：NOT_RUN、別承認待ち。
