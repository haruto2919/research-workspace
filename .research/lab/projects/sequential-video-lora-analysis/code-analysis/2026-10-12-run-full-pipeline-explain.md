---
project: sequential-video-lora-analysis
date: 2026-10-12
analysis_mode: explain
detail: full
level: mid
source_repository: https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_branch: dev
source_commit: 65dedb4ebc26c48d23606b9bc674044f64e64c06
source_dirty: false (Comet provenance at the run; current server state unknown)
targets:
  - run_full_pipeline.sh
  - scripts/moco/train_full_streaming_moco.py
  - training/streaming_moco_full.py
  - training/streaming_moco.py
  - scripts/linear_probe/build_manifest.py
  - scripts/linear_probe/extract_features.py
  - scripts/linear_probe/run_probe.py
  - evaluation/linear_probe.py
related_spec:
  - specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md (approved)
  - specs/2026-10-07-configurable-reduced-video-full-pipeline-spec.md (implemented)
comet_experiment_key: da975c40965747a499aa08465f7546db
comet_experiment_name: stage6b-subset-v1__moco__activitynet-reduced-v1-seed7
---

# Code Deep Dive: run_full_pipeline.sh と ActivityNet reduced run

## 対象と解析の範囲

- 目的: `run_full_pipeline.sh` が起動する全パイプラインの実行順、入出力、設定、Cometで観測されたMoCo実験との対応を説明する。
- 実装の根拠: `tamaki-lab/2026_09_ishikawa_sequential-video-lora` のコミット `65dedb4ebc26c48d23606b9bc674044f64e64c06`。このコミットは指定Comet実験の `provenance.implementation.commit` に一致する。
- 運用の根拠: Research Workspace `main` の `AGENTS.md`、`research-code-insight/SKILL.md`、対象project README、上記specを確認。
- 観測の根拠: Comet Research Readerのexperiment summaryおよびmetric history。サーバのローカル出力、完了ログ、running process、checkpoint実体にはアクセスしていない。
- 対象外: コード変更、学習実行、ローカルcheckpointの検証、Linear Probe の未取得成績を推測すること。

## Overview: What / Why

`run_full_pipeline.sh` は単なるMoCo学習ランチャーではなく、選択したActivityNet動画で単一パスMoCoを実施し、確定したfinal Query LoRAをBase ViTと同じ下流データで比較する、5段階のエンドツーエンド実行入口である。

1. Streaming MoCo: 選択したtrainingソースを時系列順に学習し、Query LoRA final snapshotを作る。
2. Manifest: training / validationの同じsource-selection profileを使ってannotation segmentを列挙し、class coverageおよび再生成hashのIntegrity Gateを監査する。
3. Feature extraction: Base ViTとBase ViT + final Query LoRAのpre-projector segment特徴 [768] を、同一manifestに基づいて抽出する。
4. Linear Probe: 各特徴をfreezeし、200クラス線形分類器をseeds 0/1/2で学習・評価する（計6 runs）。
5. Aggregate: Top-1 AccuracyとMacro Class Accuracyの平均およびsample stdを比較結果として保存する。

## Key Components and Data Flow

### Launcher, arguments and preflight

`run_full_pipeline.sh` の引数は
`<activitynet_root> <run_id> <moco_seed> [gpu_id] [moco_mode] [source_selection_profile]`。
省略時のGPUは0、モードはfresh、profileは`activitynet_full_v1`なので、縮小実験は第6引数に`activitynet_reduced_v1`を明示する必要がある。

- `dev` branchを要求。untrackedを含めGit checkoutのclean stateを要求する。
- `fresh` は新規run directoryのみ許可し、`resume` は既存の`resume/latest.pt`から厳密照合して再開。`skip` はMoCo学習だけを省略し、final snapshotのvalidationから下流へ進む。
- エラーでは `set -Eeuo pipefail` により失敗を伝播し、後続Stageを実行しない。
- 各stageのstdout/stderrを`log/full_pipeline/<run_id>/`に記録する。

### Stage 1: Streaming MoCo

対象ファイル:
- `scripts/moco/train_full_streaming_moco.py`: Hydra設定、dataset full inventory照合、選択集合の決定、Comet初期化。
- `training/streaming_moco_full.py`: 逐次動画ループ、初回warm-up、video-boundary resume checkpoint、Query LoRA snapshot、集計ロギング。
- `training/streaming_moco.py`: 1 chunkの Query / Key encode、InfoNCE、勾配、optimizer、EMA、Key enqueue、監査。

`strict_single` で1動画をアクティブに処理する。chunkサイズは16 frames。最初の動画の最初のchunkはKey-only warm-up、その後のchunkは各1 optimizer step。動画間でQuery/Key、optimizer、EMA、Queueは保持する。Queryは元のview、KeyはRGB→GBR+horizontal flip。負例はold Queueの`all_past`。現行方式はフレーム間の明示的時間融合を持たず、frame-wise ViT CLSをmasked meanで集約したclip特徴を利用する。

順序: 旧Queueを負例としてloss計算 → backward → Query LoRA+Query ProjectorをAdamWで更新 → Key EMA → KeyをQueueへ追加。

resume checkpoint: `log/moco/<run_id>/resume/latest.pt`、100動画ごとの境界とfinalで保存。評価用snapshot: `log/moco/<run_id>/evaluation_snapshots/videos-*_step-*_final/`、final時にQuery LoRAのみをPEFT形式で保存する。finalと1000動画ごとのsnapshotは別タイミング判定。単一プロファイルが1000本なら同じ境界のfinal snapshotを1つ作る。

Cometへの`moco/loss`、`moco/positive_similarity`、`moco/query_lora_grad_norm`は100 updatesの平均、Queue件数と負例件数、processed_videosは記録時の値。`processed_videos`は動画全chunk処理後に加算されるので、最後の定期記録が999でも直後に1000へ到達し得る。

### Stage 2: Manifest and Integrity Gate

`scripts/linear_probe/build_manifest.py`がsource selectionに整合するannotation-segment manifestと200クラスのlabel mappingを構築。さらにauditでSHA-256再生成一致とtrain/validationのclass coverage等を検証する。production manifestに`Gate PASS`が必要。失敗時は後段に進まない。

selected source数はtraining=1000、validation=1000。学習にvalidationは使用しない。training selectionはMoCo学習とProbe training manifestで共用する。`activitynet_reduced_v1`はSHA-256でソース集合を選んでからAdapter順に並べ直す。source selection seed=0とMoCo初期化seed=7は別変数。

### Stage 3: Segment features

`scripts/linear_probe/extract_features.py`では`base_vit`と`moco_query_lora_final`の2条件を同じmanifestで抽出。valid frame CLS [768]をmasked meanでchunk特徴 [768]にし、対象segmentに属するchunk特徴を平均してsegment特徴 [768]にする。いずれもraw pre-projector float32特徴で追加正規化なし。Key encoder / MoCo ProjectorはLinear Probeで使用しない。

### Stages 4 and 5: Linear Probe / Aggregate

`scripts/linear_probe/run_probe.py`、`evaluation/linear_probe.py`。事前抽出した特徴をfreezeし、`Linear(768,200)`の線形headだけを各conditionにつきseed 0/1/2で100 epochs学習（AdamW, LR 1e-3, WD 1e-4, batch=256）。学習中にvalidationによるbest epoch選択をせず、最後のepochでTop-1 / Macro class accuracyを計算する。6 runsの後、conditionごとに平均と標本標準偏差を集計する。

## Comet から直接観測された reduced MoCo実験

- Experiment name: `stage6b-subset-v1__moco__activitynet-reduced-v1-seed7`
- Experiment key: `da975c40965747a499aa08465f7546db`
- Tags: `moco`, `production`, `seed-7`, `stage6b-subset-v1`
- Implementation: `tamaki-lab/2026_09_ishikawa_sequential-video-lora` / `dev` / `65dedb4ebc26c48d23606b9bc674044f64e64c06`; recorded dirty=false.
- Dataset: ActivityNet v1.3 training、full inventory 10024件、selected 1000件。validation選択は1000件。
- Source selection: `activitynet-reduced-v1`、`sha256_rank_preserve_adapter_order_v1`、selection seed=0、selection SHA-256=`1f91c280f09a1aa1b6b22a0bc130c2ed19a7888695b0a82f8e22afb793da5217`。
- MoCo protocol `activitynet-selected-single-pass-streaming-moco/v1`、`strict_single`、`gbr_horizontal_flip`、`all_past`、frame chunk=16。
- Optimizer: AdamW lr 0.001、WD 0、seed=7。GPU: NVIDIA RTX PRO 4500 Blackwell。
- Latest recorded point (not necessarily final video boundary): update_step=209100、processed_videos=999、loss=4.5137730038、positive_similarity=0.721314975、Query LoRA grad norm=0.59782618、queue_count=4096、valid_negative_count=4096、unique sequence count=15。
- Entire reported loss range: min=2.6376058835、max=8.9345104790。
- Entire 100-update-average Query LoRA gradient norm range: min=0.0000210837、max=5456.4088952 at step 145400。大規模な中途勾配スパイクであり、学習安定性を未検証とする合理的根拠。最大は1 stepの単発grad normではなく100更新の平均。
- Comet指定プロジェクトの実験一覧として取得できたものは2件のMoCo実験。今回のLinear Probe実験、manifest PASS、feature artifacts、aggregate resultsは直接確認できず。

## Mathematical / Research Context

MoCoのInfoNCEは正例Keyと過去Keyの対照で最適化され、学習時の損失値は下流のaction recognition精度を直接表さない。Sequence chunkは時系列順に学習へ投入されるが、masked mean自体はframe順序に不変である。よって、Linear Probeで改善しても時間情報・動き情報の獲得を単独で証明しない。

2026-10-08 MTGでは、loss振動・Query LoRA勾配急増の原因が未確定として、Linear Probeの解釈を保留し、LoRA単体 / MoCo単体から検証する方針を記録している。現在のscientific statusの記述と本件の観測は区別する。

## Alternatives and Trade-offs

縮小profileは全量実験に比べて学習・特徴抽出の処理量を削減するが、validationを含めた200クラスcoverageは事前には保証されない。正式profileと恣意的なoverrideを分離し、選択identityを全artifactへ伝播することで別条件の取り違えを防止する。resume stateとfinal evaluation snapshotを分離し、再開可能性と評価目的を区別する。

## 次に確認すべきこと

1. サーバ上の`log/full_pipeline/activitynet-reduced-v1-seed7/01_moco.log`末尾で`Result: {"status":"complete", ...}`などの終了イベントを確認。
2. `log/moco/activitynet-reduced-v1-seed7/evaluation_snapshots/`の`*_final/metadata.json`で`final: true`、`processed_videos: 1000`と selection identityを確認。
3. `log/full_pipeline/activitynet-reduced-v1-seed7/02_manifest_audit.log`と `log/linear_probe/manifest/<manifest_id>/metadata.json`のGate PASSを確認。
4. `03_features_*.log`、`04_linear_probe.log`、`log/linear_probe/results/<result_id>/comparison/aggregate_summary.json`から評価実施・成績を確認。
5. 問題切り分けのため、`log/moco/activitynet-reduced-v1-seed7/audit.jsonl`のstep 145400周辺の勾配とlossの生記録を確認。

## 根拠・未確認事項

- Observed: 上記GitHub固定コミットのbash / Pythonコード、Comet experiment metadataと時系列metricは直接確認した。
- Aligned: 2026-10-05 approved specと2026-10-07 implemented addendumに対して、選択source、single-pass、final Query LoRA、downstream compareの経路は整合する。
- Inferred: `processed_videos=999`は最終1000本目の途中で最後の100-update metricが送られた可能性がある。完了の証拠ではない。
- Unverified: 研究サーバの現在dirty状態、完走・checkpoint実在、final snapshot、manifest gate、Linear Probe出力、表現性能、時系列情報の獲得、勾配スパイクの根本原因。


## 追補: Linear Probe実行有無の判定（2026-10-12）

### Cometの再確認と結論

- Comet Research Readerを使い、設定済みプロジェクトで全実験（limit=100）と`linear-probe`タグを検索。取得できた全実験はMoCo 2件だけで、`linear-probe`タグ実験は0件。
- 対象 MoCo 実験 `da975c40965747a499aa08465f7546db` の最終周期metricは 2026-10-09 13:42:38 JST、`global_update_step=209100`、`processed_videos=999`。process duration約32時間23分。
- `processed_videos`は**動画終了後**に更新されるが、Cometのperiodic metricは**100 updatesごと**に更新される。このため`processed_videos=999`は1000本目の完了を否定する証拠にならない。
- `scripts/linear_probe/run_probe.py::run_seed`は新規seed結果について`start_experiment`を行い、毎epoch`train/loss`/`train/top1`、最終評価で`val/top1`/`val/macro_class_accuracy`をCometへ送る。aggregateもComet experimentを作る。従ってLinear ProbeがComet無効設計なのではない。
- `scripts/linear_probe/build_manifest.py::audit`はIntegrity Gate不合格時にComet experimentを作る前にreturnするため、Manifest実験が0件であることはMoCo直後の停止を証明しない。reduced 1000/1000の200-class coverageはGitHub/Cometから未検証。
- `logger/comet_lineage.py::start_experiment`は通信エラーを捕捉しlocal scientific workを停止させない。`extract_features.py`と`run_probe.py`は一致する既存local artifactを再利用した場合に新規Comet実験を生成しない。したがってComet実験不在だけで「Linear Probeが実行されなかった」と断定してはいけない。
- **確実な結論:** 現在アクセスできるComet範囲では、今回のLinear Probe成功と成績は確認できない。最も直接的な確認先は研究サーバの`log/full_pipeline/activitynet-reduced-v1-seed7/`、final snapshot metadata、manifest Gate、probe aggregateである。

### ローカル診断コマンド（読み取り専用）

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
RUN_ID=activitynet-reduced-v1-seed7
for f in 01_moco.log 01_moco_resume.log 02_manifest_build.log 02_manifest_audit.log 03_features_base_vit.log 03_features_moco_query_lora_final.log 04_linear_probe.log; do
  p="log/full_pipeline/$RUN_ID/$f"
  if [[ -f "$p" ]]; then echo "===== $p ====="; tail -n 15 "$p"; else echo "MISSING: $p"; fi
done
find "log/moco/$RUN_ID/evaluation_snapshots" -maxdepth 1 -type d -name '*_final' -print 2>/dev/null || true
```

判定: `01_moco.log`に`status=complete`か`Result: {"status": "complete", ...}`がありfinal snapshotが存在すればMoCoは完了。 `02_manifest_audit.log` のGate FAILなら後段停止。 `03_features_*.log`があれば特徴抽出実行を確認。`04_linear_probe.log`中の`probe_result`または`aggregate`でprobe進行・完了を確認する。`start_experiment`失敗などのComet障害はstage logの`Comet: ...`または`Comet metric logging failed`で調べる。


## 追補: ユーザー提供ログに基づくManifest Gate停止の確定（2026-10-12）

以下はユーザーがチャットへ貼り付けたSSHサーバ上の読み取りログに基づく観測である。GitHubコネクタからサーバ上のファイル実体を直接検証したものではない。

### ユーザー提供ログから確認された事実

- `log/full_pipeline/activitynet-reduced-v1-seed7/01_moco.log`の末尾: `Result: {"final":true,"global_update_step":209144,"next_source_id":null,"next_video_index":1000,"processed_videos":1000,"queue_count":4096,"status":"complete"}`。MoCo 1000件 single-passは正常終了した。
- 同じログにComet artifact `haruto2919/activitynet-moco-query-lora:3.0.0` のアップロード成功・delivery確認が表示された。これはMoCo側snapshot artifactのアップロードについてであり、Linear Probe結果ではない。
- `02_manifest_build.log`の末尾: `{"event":"manifest_built","path":"log/linear_probe/manifest/lp-v1__sel-1f91c280f09a","status":"FAIL","failed":["1_training_labels_200","2_validation_labels_200","3_training_sample_per_class","4_validation_sample_per_class"],"rows":{"training":1480,"validation":1542},"skipped":{"training":14,"validation":16}}`。
- training / validation各1000本についてsegment manifestを構築したが、200クラス存在・全クラス1件以上という4条件がFAIL。`skipped`は動画数ではなく除外されたannotation segment数を示す（segmentがchunk包含条件を満たさない等）。
- `02_manifest_audit.log`、`03_features_base_vit.log`、`03_features_moco_query_lora_final.log`、`04_linear_probe.log`が存在しないとユーザーは報告。したがって、当該launcher実行はManifest build失敗直後に停止し、Linear Probeまで到達していないと判断できる。
- `run_full_pipeline.sh`は `set -Eeuo pipefail` を使用する。Manifest build非0終了で後段を実行しない。Manifest Gateは200クラス網羅性の問題を検出しており、Comet側にLinear Probe結果が見えない直接原因はLinear Probe未実行。
- `moco/processed_videos=999`、`global_update_step=209100`のComet最終定期記録と`1000/209144`の実際の完了は両立する。最終44 updatesは100-updateごとのメトリクス送信タイミング外。
- クラスの具体的な欠損一覧・各クラスの実サンプル数は貼付ログにないため未確認。

### 考えられる対応と制約

1. MoCo結果をそのまま保持し、欠損クラスとGate結果の詳細をlocal manifest metadataとmapping、annotation-derived rowsから監査する。既存MoCoは再実行不要。
2. 現行の200クラス/共通training selectionというapproved実験契約を維持するなら、selection profileの改訂またはtraining/validation選択動画数の増加など、class coverageを満たす新たなsource選択が必要。**training selectionが変われば既存の最終LoRAのselection identityと合わないため、現行lineageのままでdownstreamを流用できない**。新MoCo runと新artifactが必要。
3. 現在の1,000本・最終LoRAを維持して評価する案は、クラス網羅Gateおよび200-way評価の契約を変更する新たなspec/評価protocolとして検討する。今回の失敗を無視して強制的に`PASS`に書き換えるべきではない。
4. 単に`run_full_pipeline.sh ... skip activitynet_reduced_v1`を再実行しても、同じscientific config / source selectionのManifestが同じ理由で失敗する。既存のFAIL manifestを自動で書き換えない。


## 追補: 全量Stage 6B-v2と縮小runの停止原因比較（2026-10-12）

### 比較対象・ソース同一性

- Comet full-run key: `e1df2d65b1d64b94b5c80a48473d017f`、name `stage6b-v2__moco__run1`。source code `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@605f2afb95e1dda541208a46d69ce46acd83b335`、dirty=false（Comet出力）。selected sourceはActivityNet training 10024、MoCo seed=0、stop_after_videos=null、resume=false。
- Comet reduced-run key: `da975c40965747a499aa08465f7546db`、name `stage6b-subset-v1__moco__activitynet-reduced-v1-seed7`。source code `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@65dedb4ebc26c48d23606b9bc674044f64e64c06`、dirty=false（Comet出力）。selected training=1000、MoCo seed=7。
- GitHub compare `605f2afb..65dedb4e`で4 commits ahead。変更対象に`run_full_pipeline.sh`、`scripts/linear_probe/build_manifest.py`、`evaluation/activitynet_manifest.py`、`training/streaming_moco_full.py`、`scripts/moco/train_full_streaming_moco.py`、source selection実装追加などが含まれる。したがって「両runのコードが完全に同じ」とは言えない。

### Cometの直接観測

- Full: 最終定期記録 `global_update_step=605900`、`processed_videos=2910`（期待10024）。最後のmetric timestampは2026-10-09 17:25:54 JST。Comet上のgrad normの最大記録は約`1107159.54`（100-update集計の最大）で、大規模な数値不安定性の兆候がある。ただし停止原因と因果関係は未確証。
- Reduced: Cometの最後の定期記録は999/1000・209100 step、しかしユーザー提供の`01_moco.log`で1000/1000・209144 step・`status=complete`とfinal snapshotのComet artifactアップロード成功を確認。
- Reducedの`02_manifest_build.log`で`status=FAIL`、failed=`1_training_labels_200`、`2_validation_labels_200`、`3_training_sample_per_class`、`4_validation_sample_per_class`。後続ログなし。縮小runがManifest Gateで停止した原因は確定。
- FullはComet記録上まだMoCo途中（2910/10024）に見える。研究サーバのfull-run local completion status、エラー、後続stageは現在未提示。従って「full runもManifest Gateで止まった」と主張できない。Comet実験が終了したか、loggingだけ止まったかをsummaryだけから断定しない。

### 共通障害の可能性と優先順位

1. 第一優先は **full-run `run1` のローカル `01_moco.log`末尾と`audit.jsonl`末尾を確認**し、明示エラー/ユーザー中断/OS kill/突然のログ途絶/正常完了を識別すること。
2. `training/streaming_moco_full.py`は例外時に`audit.jsonl`へ`event="stopped"`とreasonを記録してraiseするが、OSによるSIGKILL・電源断などではこのeventは残らない場合がある。
3. 正常完走していればfinal snapshotや`run_metadata`を確認した上で`log/full_pipeline/run1/02_manifest_build.log`を確認する。full runでもmanifest FAILならクラス網羅、chunk包含条件、annotation/timestampの共有部分を追加監査する。
4. Full runが途中で落ちているならManifest Gateとは別の問題で、データdecode、OOM、nonfinite、数値不安定、OS/プロセス終了等をローカルログと環境証拠で切り分ける。Comet grad spike自体は停止原因を示さない。
5. `run1`の実行commitと現行commitは異なる。resumeではcommit/provenance/selectionなどのidentity照合があるため、環境を変更して無条件にresumeしない。旧checkpointは保持し、原因調査してから再開方針を決める。

### ローカル診断（読み取り専用）

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
RUN_ID=run1
for f in 01_moco.log 01_moco_resume.log 02_manifest_build.log 02_manifest_audit.log 03_features_base_vit.log 03_features_moco_query_lora_final.log 04_linear_probe.log; do
  p="log/full_pipeline/$RUN_ID/$f"
  echo "===== $p ====="
  if [[ -f "$p" ]]; then tail -n 25 "$p"; else echo "なし"; fi
done
echo '===== local audit tail ====='
if [[ -f "log/moco/$RUN_ID/audit.jsonl" ]]; then tail -n 8 "log/moco/$RUN_ID/audit.jsonl"; else echo "なし"; fi
echo '===== snapshots ====='
if [[ -d "log/moco/$RUN_ID/evaluation_snapshots" ]]; then find "log/moco/$RUN_ID/evaluation_snapshots" -mindepth 1 -maxdepth 1 -type d -name '*_final' -print; else echo "なし"; fi
echo '===== resume checkpoint presence ====='
if [[ -f "log/moco/$RUN_ID/resume/latest.pt" ]]; then ls -lh "log/moco/$RUN_ID/resume/latest.pt"; else echo "なし"; fi
```

判定: `stopped` eventとreasonがあればPython例外の候補、正常完了 `status=complete` + final snapshotがあれば下流の問題を確認、どちらも無くログが途絶していればOS/プロセス側の終了痕跡を追加調査する。ユーザーの手元での追加コマンド実行は提案のみで、ChatGPT側で実行していない。


## 追補: 全量runのCUDA 719停止確定とActivityNet split訂正（2026-10-12）

### Source・Authority

- SSHサーバの情報はユーザーがチャットに貼付した`01_moco.log`末尾と`log/moco/run1/audit.jsonl`末尾に基づく。GitHubコネクタからサーバログ実体に直接アクセスしたものではない。
- 対象コードはCometに記録された`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@605f2afb95e1dda541208a46d69ce46acd83b335`、dirty=false、ActivityNet AdapterはResearch Workspaceで検証済み`tamaki-lab/2026_09_ishikawa_sequential_loader@ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`。
- 参照コード: `conf/activitynet/v1_3.yaml`、`scripts/moco/train_full_streaming_moco.py`、`run_full_pipeline.sh`、`scripts/linear_probe/build_manifest.py`、`training/streaming_moco_full.py`。
- 承認済みspec `specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md` は**MoCo pretrainingをActivityNet trainingのみ**と定める（validationやtestingは明示的out of scope）。

### 2つの停止原因は別

- **full run `stage6b-v2__moco__run1`**: user-provided local `01_moco.log`に`torch.AcceleratorError: CUDA error: unspecified launch failure`、`CUDA_ERROR_LAUNCH_FAILED`コード`719`、`cuMemcpyHtoDAsync_v2`報告がある。user-provided `audit.jsonl`最終event=`stopped`、UTC `2026-10-09T08:26:09.628491+00:00`（JST同日17:26:09）、`processed_videos=2910`、`next_video_index=2910`（0起算、次に処理中の動画は2911本目）、`global_update_step=605937`、`next_source_id=IGcsVPa34Hc`、`final=false`。エラー発生前の監査step605936の`all_parameters_finite`/`all_queue_keys_finite`はいずれもtrue。MoCo学習中にCUDAが失敗し、ManifestおよびLinear Probeへ到達していないことが確定。
- **reduced run `stage6b-subset-v1__moco__activitynet-reduced-v1-seed7`**: user local `01_moco.log`で1000本`status=complete`、209144 updatesを確定。その後`02_manifest_build.log`で200 class coverageのIntegrity Gate FAILにより停止。2つのrunsが同じ段階・同じ理由で止まったわけではない。
- Full runのComet定期記録が605900/2910で切れていたのは、100-update periodic logging中断のため（監査ログでは605937まで到達）。

### 10,024本のsplitの訂正

- `conf/activitynet/v1_3.yaml`: `expected_source_counts.training=10024`、`validation=4926`。
- Research Workspaceの実データAdapter検証`experiments/2026-09-23-activitynet-adapter-verification.md`では`training=10024`、`validation=4926`、`testing=5044`、**総数19994**。10,024は全splitの合計ではなく**trainingだけ**。
- `scripts/moco/train_full_streaming_moco.py`は`settings.activitynet.splits['training']`から`ActivityNetAdapter.sequence_sources(split)`を取得し、件数10024を検証してsingle-pass streaming MoCoへ渡す。Comet `config.dataset.split=training`も一致。
- 同commit `scripts/linear_probe/build_manifest.py`は`SPLITS=('training','validation')`双方からsegment manifestを構築し、Base/LoRA frozen feature/Linear Probeへ引き渡す。testingはこのpipelineでは使用しない。MoCo学習とLinear Probe supervised trainingは役割が異なる。

### CUDA 719をどう読むか / 未検証

- NVIDIA公式CUDA documentation: `CUDA_ERROR_LAUNCH_FAILED (719)`はdevice kernelで例外発生、誤ったデバイスポインタや範囲外メモリアクセス等も候補で、CUDAコンテキストは継続使用できない。同期されないCUDAエラーは`pixel_values.to(device)`など後続APIで初めて報告されることがあるため、stack traceだけで転送操作や特定動画を根本原因に断定しない。
- データ固有再現、GPUハード/ドライバ、PyTorch/ CUDA kernel、直前の非同期kernel、過負荷等はまだ切り分けられていない。既存loss/gradの大きな変動とCUDA 719の因果は未確認。
- NVIDIAの公式Xid docsは、Linux kernel logの`NVRM: Xid`を調査するよう案内している。ユーザーのSSHサーバに直結できないため、下記コマンドの実行結果未確認。

### 読み取り専用診断と判断

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
echo '=== 実行環境 ==='
git branch --show-current
git rev-parse HEAD
nvidia-smi
echo '=== 2026-10-09 17:26 JST前後のGPUドライバログ ==='
journalctl -k --since '2026-10-09 17:15:00' --until '2026-10-09 17:35:00' --no-pager | grep -Ei 'NVRM|Xid|NVIDIA|GPU|pcie|nvlink' || true
echo '=== run1のresume checkpoint ==='
if [[ -f log/moco/run1/resume/latest.pt ]]; then ls -lh log/moco/run1/resume/latest.pt; else echo 'checkpointなし'; fi
echo '=== 直近checkpoint/snapshotの監査記録 ==='
grep -E '"event": "(resume_checkpoint|evaluation_snapshot|stopped)"' log/moco/run1/audit.jsonl | tail -n 8
```

- `journalctl`が権限不足・ログローテーションで表示されない場合は、許可された管理者に同期間のkernel journalやsyslogを確認してもらう。
- `resume_interval=100 videos`なので、最終保存は正常なら2900 videos地点付近と推定するが、ファイルの内容・mtimeは未検証。動画境界からのresumeであり、2911本目のfailed chunkから厳密に再開するものではない。
- 全量runはcommit`605f2afb`、後の縮小runはcommit`65dedb4e`で4 commits差がある。resume時のcode identityやprovenance照合があるので**現行チェックアウトのまま無条件resumeを試さない**。研究実装やcheckpointを変更せず、Xidと実際のcheckpoint存在を優先確認する。
- CUDA fault後に同一Python processで繰り返すのでなく、新しいプロセスでの安全な切り分けが必要。実験継続・コード変更はこのコード解析skillの権限外。

### 補足する設計問題

- Full runで停まった理由を「training完了後validationへ移る段階でのエラー」と説明するのは誤り。MoCoはtrainingのみ10,024本を対象にし、**2910本（約29%）**で停止した。
- 縮小runのManifest Gate FAILと、全量runのCUDA 719停止は分離して扱い、下流評価方法の変更をCUDA failureの修正手段と混同しない。


## 追補: GPU2識別障害・2900本checkpoint・診断優先順位（2026-10-12）

### 今回ユーザーが提示したSSHサーバ上の出力

- 現在のローカルGit: `dev@65dedb4ebc26c48d23606b9bc674044f64e64c06`（full-runを実行した `605f2afb95e1dda541208a46d69ce46acd83b335`とは異なる）。
- `nvidia-smi`が`Unable to determine the device handle for GPU2: 0000:B1:00.0: Unknown Error`と報告。2026-10-12 17:36 JST、driver 580.178.04 / CUDA 13.0。表示できたGPU0,1はともにNVIDIA RTX A4500で、GPU2の正常表示なし。
- Comet full-run`e1df2d65b1d64b94b5c80a48473d017f`のtraining device identityは`NVIDIA RTX PRO 4500 Blackwell`、UUID`8a059266-7ede-0427-ff30-08ba7a9b5224`、CUDA可視デバイスindex0。現在の正常表示された2枚のA4500とは違う。**物理GPU2がComet記録のBlackwellである可能性は高いが、PCIe bus ID↔UUIDの直接照合はまだなく、確定はしない**。
- `journalctl -k`を一般ユーザーで実行した結果、`You are currently not seeing messages from other users and the system`と出ており、Xid/PCIeログは見えない。出力が空だからXidが無いとは言えない。
- `log/moco/run1/resume/latest.pt`は18MBで2026-10-09 17:00 JSTに存在。
- audit`resume_checkpoint` event: UTC 2026-10-09 08:00:40.247、`processed_videos=2900`、`next_video_index=2900`、`global_update_step=603379`、`next_source_id=IC-Rd7ouQmE`、sha256`4e684a7cc15cc6f676abc5e87854be4546802cf9a354ff4847492ec24549d037`。
- audit`stopped` event: UTC 2026-10-09 08:26:09.628、CUDA 719、`processed_videos=2910`、`next_video_index=2910`、`global_update_step=605937`、`next_source_id=IGcsVPa34Hc`。checkpointからは2901本目から再処理が必要で、エラー地点のchunkからはresumeできない。
- ユーザーは`sha256sum`自体はまだ実行していないため、現存する`latest.pt`がaudit SHAと一致するか未検証。ファイルの存在のみからcheckpointの中身の完全性を断定できない。
- `training/moco_checkpoint.py`の`RESUME_IDENTITY`はrepository/branch/**commit**/**device_identity**/seed/ordered sources/version他を完全一致で要求。`training/streaming_moco_full.py`は`validate_resume_identity`を呼ぶ。現在checkoutのcommitと違うことが確認済みで、GPUも異なりうるため、**現在の状態での`resume`は同一実験再開要件を満たさない**。
- 縮小runのManifest Gate FAILはGPU2故障と独立し、GPUを復旧しても同じsource選択ならGateが自動解消するわけではない。

### 評価・推論（事実とは区別）

- GPU2ハンドル取得不可と、当該学習の`CUDA_ERROR_LAUNCH_FAILED`には関連が疑われる。GPU・PCIe/電源・ドライバ等の障害を優先して調べるべき。ただしGPU2が失われた時刻や一次障害はまだ未確定で、どちらが先か、GPU2=Blackwellか、原因がハードかソフトかは現段階で断定しない。
- NVIDIA公式Xid Catalog https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html では、Xid79がPCIe経由でGPUがアクセス不能になった状態を表すと記載。ただしユーザーの今回のシステムでXid79が出たとは証明できない。

### 次の読み取り専用調査

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
echo '=== checkpoint checksum ==='
sha256sum log/moco/run1/resume/latest.pt
echo '=== device list ==='
nvidia-smi -L
echo '=== PCI device presence ==='
lspci -s b1:00.0 -nn
echo '=== PCI sysfs (if available) ==='
ls -ld /sys/bus/pci/devices/0000:b1:00.0 2>&1
```

- checksum期待値: `4e684a7cc15cc6f676abc5e87854be4546802cf9a354ff4847492ec24549d037`。合致はraw file integrityの補助証拠だが、resumeとhardware/reproducibility contractは別途検証が必要。
- 許可を受けた管理者に、`sudo journalctl -k --since '2026-10-09 17:15:00' --until '2026-10-09 17:40:00' --no-pager | grep -Ei 'NVRM|Xid|B1:00|pcie|aer|nvidia'`と障害現在時点周辺ログの確認を依頼。NVRM Xid、PCIe AER、reset、GPU認識断が記録されているか見る。Xid79以外のコードもありうる。
- ユーザー所有でないGPU/サーバ資源があるため、`sudo nvidia-smi --gpu-reset`、driver卸し、PCIデバイスremove/rescan、マシンrebootをユーザー単独で試さない。ログ採取・管理者への連絡を優先。
- GPUの健全性とcheckpointを確認した後、元のcommit、device identity、dataset source ordering、seed等と照合した上で再開計画を立てる。現在の`dev@65dedb4e`からそのまま`run_full_pipeline.sh ... resume`は実行しない。
