---
date: 2026-10-08
project: sequential-video-lora-analysis
source_todo: null
topic: streaming-moco-multi-gpu-feasibility
status: exploratory
tags: [brainstorm, research, streaming-moco, multi-gpu, performance]
---

# Streaming MoCoの複数GPU化が難しい理由と代替案（探索記録）

## 出発点・中心の問い

2026-10-08の相談：「複数gpuで実験をしたいときになぜ難しいか考えてください」。

対象は現行のActivityNet v1.3／Stage 6B-v2 strict-single Streaming MoCo、最終Query LoRA取得、その後のBase/LoRA特徴抽出・Linear Probeまでのfull pipeline。特に「既存 `run_full_pipeline.sh` と科学的に同じ実験」のまま高速化することを重視する。現時点でユーザーからmulti-GPU実装、採用方針、spec化は指示されていない。

## 確認した文脈・Evidence（GitHubでの読み取り）

- Research Workspace: `.research/lab/projects/sequential-video-lora-analysis/README.md`、`specs/2026-09-28-shared-streaming-moco-protocol-spec.md`、`specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md`、`.research/secretary/notes/brainstorm/2026-10-07-full-pipeline-throughput-optimization.md`、`materials/2026-10-08-current-streaming-moco-experiment-overview-mtg-materials.md`。
- 実装: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、調査時GitHub HEAD `65dedb4ebc26c48d23606b9bc674044f64e64c06`。
- `training/streaming_moco_full.py`: 動画を1本ずつ、chunk順に完了させる。最初のchunkだけKey-only warm-up。Query/Key/Optimizer/EMA/FIFO Queueは動画境界でも維持、動画境界でresume checkpoint。
- `training/streaming_moco.py`: 1chunkにつきQuery/Key forward → old QueueでInfoNCE → backward → optimizer.step → EMA → current Keyをenqueue。グラデーション・base・Queue・EMA等を毎step監査。
- `self_supervised/moco/vit_lora_moco.py`: Query/Key encoder、Query/Key Projector、metadata FIFO Queue、EMA更新。QueueはPythonのdequeベースで共有分散実装ではない。
- `run_full_pipeline.sh`: GPU指定は数字1個だけ許容。MoCoの後にBaseとQuery LoRAの特徴を順番に抽出、Linear Probeを実行する。
- `training/moco_checkpoint.py`: resume identityに実装commit、依存、source order、seed、device identityなどを含む。
- 汎用の `main_pl.py` DDP対応というREADME説明は、別入口の通常学習であり、専用Streaming MoCoのmulti-GPU対応を意味しない。
- Research Workspaceの2026-10-07速度最適化brainstormでは、同一Science Contractを求める初期scopeからMoCoのDDP/DataParallel・chunk分割・batch変更・Query/Key別GPU分割等を除外し、profilingとaudit最適化を先に行う候補としている。
- GitHub情報のみ確認済み。SSHサーバ上のdirty worktree、実GPU構成、処理時間内訳、最新full run実測は未確認。

## なぜ難しいか（事実からの分析）

1. **逐次依存**: chunk t+1 は chunk t のoptimizer/EMA/Queue更新後stateを使用する。GPU0=A2、GPU1=A3を同時学習すると更新前stateを共用するか同期点を変えるため、従来と異なるオンライン最適化になる。
2. **Queue整合**: Negativeは旧Queue内の全past Key。各rankローカルQueueではNegativeが異なる。分散共有するならentryのglobal order、metadata、Key計算時点、現在positiveの投入禁止を保証する必要がある。
3. **EMAとOptimizer stepの意味**: 分散batch化するとstep数・順序・勾配・EMA対象theta_qが変わる。普通のDDPへ置くだけではScience Contractと一致しない。
4. **Resume/再現性**: world size、rankごとのRNG、distributed Queue、同期位置、device identityを含む新契約と検証が必要。単純GPU数変更で既存checkpointのresume同等性は主張できない。
5. **性能**: 1chunk/global stepで厳密に順次同期するならDataParallel/DDPが扱う有効batchは小さく、通信同期コストで逆に遅くなる場合がある。毎step auditも頻繁にGPU/CPU同期するため、多GPUだけでは根本解決しない。実測profile未確認。
6. **Query/Key 2GPU**: 別GPUで各branchのforward並行化は理論上可能だが、Query/Key両出力が揃って初めてloss計算、EMAのcross-device更新が必要。現在のコードは逐次呼び出しなので非自明な実装変更が必要。厳密な数値一致は別途評価。

## 候補と比較（いずれも未採用）

| 候補 | 元のScience Contract | 実現難度 | 主な論点 |
|---|---|---|---|
| A. GPUごとに独立run（別run_id/seed/条件） | 各runでは維持しやすい | 低 | 実験1本の所要時間は短縮しない。NAS/CPU競合 |
| B. MoCoは1GPU、Base/LoRA feature extractionを別GPU jobで並列化 | 維持しやすいがfeature同等性要検証 | 中 | 学習終了後の完全独立条件。既存launcherは逐次実行 |
| C. 現行1GPUのprofiling→audit頻度/同期削減→ordered prefetch | 原理上維持しやすいがexact equivalence要検証 | 中 | 最初の20とevery100等は候補、未採用のまま |
| D. Query/Key encoderを別GPUで実行 | sequential objectiveは維持し得る、数値同等性未確認 | 高 | forward overlap、EMA/transport、GPU同期、速度改善の実測 |
| E. 16フレームを複数GPUでframe-wise処理しmasked meanを再合成 | 数学的には同一演算を目指せるがbitwise差とautograd配線あり | 高 | tiny per-GPU work、フレーム間演算非依存性、通信・コスト |
| F. chunk/動画を分割してDDP/DP学習 | 従来と同一にはならない | 高（別protocol） | バッチ・optimizer回数・EMA・Queue更新の意味が変化 |

## 現時点の収束（推奨候補・未承認）

- 目的が「同じMoCo runを同じ条件で速く」であれば、先に1GPU profiling、audit overhead、decode、forward/backward、GPU utilizationを計測する。
- GPUが複数ある利点はまず「独立runを各GPUで実施」「MoCo後のBase/LoRA feature extractionを条件同等のまま並列化」の方向で活用する。
- MoCo自体の2GPU化を検討するならDP/DDPで動画を分割せず、Query/Key分離やframe-wise並列を別設計として検証する。速度向上は保証しない。
- 既存Science Contractの変更を許容するならmulti-GPU mini-batch MoCoは新たな研究protocol、独立したbaseline/結果として扱う。

## 反例・保留・未解決事項

- GPU時間の大半がQuery forwardとKey forwardに占められ、複数GPUが余っている場合、Query/Key分割は有益となる可能性がある。計測せずに無効と断言しない。
- GPUの空き容量・機種・PCIe/NVLink、CPU decode速度、NAS帯域が不明。実際のボトルネックは未確認。
- CUDA kernelや異種GPUによる演算順序・精度差によりbitwise一致を要求できるかは別途決定が必要。
- ネガティブ共有、rank間Queue重複、分散checkpoint、途中resumeは未設計・未検証。
- ユーザーが優先するのが「実験1本のwall-clock短縮」か「複数条件のtotal throughput改善」か、今後の意思決定点。

## 次のアクション候補（TODOには未追加）

1. 固定prefixで1GPUの時間内訳・GPU利用率・NAS使用量を計測する。
2. MoCoを保持しつつ後段のBase/LoRA特徴抽出を2GPUで独立実行できるか、artifact値とmanifest identityを検査する。
3. Query/Key 2GPU分割の実装可能性・通信量・性能を別specで検討する（必要時のみ）。
4. 1GPUとmulti-GPUの同一Science Contractの範囲を明示的に決める。

## Authority

このメモは探索的brainstormであり、spec・実装許可・実験結果ではない。明示的なユーザーの採用はまだない。
