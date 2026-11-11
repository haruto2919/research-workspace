---
project: sequential-video-lora-analysis
date: 2026-11-11
spec_type: implementation-refactoring
status: implemented
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
base_commit: 559073f517a082e41844ba495e90d57035fe2bdf
loader_repository: 2026_09_ishikawa_sequential_loader
loader_branch: feature-stride-ddp
loader_base_commit: ce4dcbc
source_brainstorm: ../../../../secretary/notes/brainstorm/2026-11-10-research-code-refactoring-plan.md
source_analysis: ../code-analysis/2026-11-10-implementation-structure-explain.md
---

# ViT＋MoCo＋LoRAの本流のリファクタリングspec

**Status: implemented（2026-11-11。研究コード `feature-vit-moco@c478269`（正解データ）と `@59bade2`（本体）。CPUでの確認 V-C・V-T・V-P・V-I はすべて満たした。[実装記録](../experiments/2026-11-11-vit-moco-pipeline-refactoring-implementation-cpu-verification.md)）**。**Status: approved（2026-11-11、ユーザーが承認）**。GPUでの確認（§8の V-G）は、実装とは別に許可を取る。

## 1. 目的

本流の計算の中身を変えずに、次の3つを実現する。

1. **GPUの数を簡単に切り替えられる**：設定1つで、CPU・GPU 1個・GPU 2個を切り替える。
2. **学習の設定を1か所で変えられる**：stride、読み方（ランダム順か逐次か）、step数などを、1つのexperimentファイルとコマンドラインの上書きで変える。
3. **本流がテストや検証のコードに依存しない**：本流だけで動くようにする。

あわせて、ファイル構造を旧フレームワークの形（上位フォルダに `__init__.py` を置いて公開する形）にそろえ、自分が読んで分かる構造にする。

## 2. 前提（コード調査で確認済み）

基準：研究コード `feature-vit-moco@559073f`（clean）。詳しい構造は[構造解析](../code-analysis/2026-11-10-implementation-structure-explain.md)を参照。

- 本流の中心は次のファイル。
  - `scripts/vit_moco/moco_vit_ddp_smoke.py`（769行）：runner
  - `query_collapse_diagnostic.py`（901行）：診断、presetのルール、比較CLI
  - `activitynet_frames.py`、`activitynet_sequential.py`：データ
  - `temporal_order_eval.py`：並び順評価
  - `self_supervised/moco/`：MoCoの3段の継承
  - `model/vit/`：ViTのbackboneと自作LoRA
- presetは7つ（`conf/moco_vit_*.yaml`）で、`defaults` で最大5段につながっている。
- GPUでの制約：
  - `prepare_config` が `world_size >= 2` を必須にしている。
  - `ADOPTED_RECIPE` と `query_collapse_diagnostic` の許可リスト（`ACTIVITYNET_GPU_RUNS`、`CLIP_ADOPTED`、`SEQUENTIAL_ADOPTED`、`SEQUENTIAL_GPU_RUNS` など）が、presetごとの値とstep数を固定している。
- 本流が借りている道具：
  - `scripts/parity/moco_imagenet_parity.py`：`PILGaussianBlur`、`_free_port`、`_global_norm`、`_sha256_json`、`_sync`、`apply_backend_flags`、`epoch_cosine_lr`、`step_metrics`
  - `scripts/parity/moco_imagenet_10epoch.py`：`JsonLines`、`epoch_seeds`、`gpu_snapshot`、`peak_memory`、`seed_epoch`、`write_json_atomic`、`read_jsonl`
  - `scripts/parity/moco_parity_state.py`：`REPOSITORY_ROOT`、`environment`、`file_sha256`、`git_info`、`state_sha256`、`tensor_sha256`
  - `scripts/smoke/smoke_moco_cifar10.py`：`TwoCropsTransform`、`capture_rng_state`、`restore_rng_state`
- 本流の中の「テスト専用」の分岐：
  - `training.device=cpu`（gloo）
  - `data.sequential.video_dir`（GPUでは拒否される）
- checkpointの同一性（`config_identity`）に入っているもの：
  - 設定の値（`moco` 節の全体、`precision` 節の全体、学習の一部のキー）
  - `IDENTITY_SCHEMA`、`LR_SCHEDULE`、`CONSTANT_LR`、`CLIP_FUSION` の文字列
  - データの記録（augmentationのrepr、`num_workers`、データのダイジェスト、sequential-loaderのgit情報）
  - モジュールのパスはreprに入っていない（G2の `run_report.json` で確認）。
- `temporal_order_eval` は、既存のrunの `resolved_config.yaml`（古い形式）から `build_model` を呼ぶ。
- GPUでは、同じコードでもstep 2以降はrun間でbitwise一致しない（L5、A2）。
- テスト：
  - 本流に関わるテストは9ファイル（約2,800行）。
  - 一部は互いにimportしている（`test_vit_moco_lr_activitynet.py` → `test_moco_vit_ddp_smoke.py` など）。
  - `CUDA_VISIBLE_DEVICES=""` での全テストは 197 failed / 826 passed / 18 errors（559073f）。失敗はすべて旧フレームワークのテストと、`test_moco_5epoch_pair.py` の1件。

## 3. Decision Contract（2026-11-10〜11にユーザーが採用）

| # | 項目 | 採用 |
|---|---|---|
| R1 | 互換性 | **同じ設定なら、数値がbitwise一致する**。確認する範囲は、CPU（gloo）は全step、GPUはstep 0のprobe特徴とstep 1のloss |
| R2 | 進め方 | **まとめて作り直す**。作り直す前に、今のコードの出力を正解データとして保存する |
| R3 | 旧フレームワーク | **中身は触らない**（`main.py`、`train.py`、`val.py`、`main_pl.py`、`model/` の分類モデル、`dataset/`・`setup/`・`utils/`・`logger/`・`callback/` の既存ファイル）。`__init__.py` への書き足しだけは行う（R6） |
| R4 | presetの許可リスト | **無くす**。代わりに、experiment名とコマンドラインの上書きを `run_report.json` に残す。gitがclean（GPU）、成果物を上書きしない、のガードは残す |
| R5 | GPUの数を変えたときのbatch | **global batch 256を保つ**（local = global ÷ プロセス数） |
| R6 | 公開のしかた | **既存の `__init__.py` に書き足す**。新しいファイルは**既存パッケージの直下**に置く（サブフォルダは作らない） |
| R7 | parityとsmoke | **本流の道具をimportするように書き換える**。道具は本流に1つだけ置く |
| R8 | 古い入口とpreset | **消す**（`scripts/vit_moco/`、`conf/moco_vit_*.yaml`）。過去の実験は、記録にあるcommitをcheckoutして再現する |
| R9 | シャッフル | 既存の2つの読み方（manifestのランダム順、sequential-loaderの逐次読み込み）を、**`data.order` 1つで切り替えるだけ**。新しい読み方は足さない |
| R10 | テスト | **新しい構造へ移す**。preset・許可リストを確かめるテストは消す。テスト同士のimportは、`test/` の共通ヘルパーにまとめる |
| R11 | GPUの正解 | **既存のrunの記録と比べる**（§8 V-G）。一致しないときだけ、基準commitでGPU runをやり直して原因を切り分ける |
| R12 | 出力先 | **`log/moco/<experiment名>/<train_mode>/<日時>`** に統一する。過去の出力は今の場所に残し、移動も上書きもしない |

### 対象外

- 旧フレームワークのファイルの中身の変更、旧テストの修正。
- Lightningなど、新しいフレームワークへの移行。新しいdependencyの追加。
- 学習の計算の変更：loss、augmentation、seedの順番、乱数の消費順、Queue、LoRA、モデル構造、正例・負例の定義。
- 新しいデータの読み方（sequential-loaderのclipをシャッフルする、など）。
- linear probeの追加、新しい評価。
- gradient accumulation、AMP。
- MoCoの公式一致検証（parity）のロジックの変更。importの書き換えだけを行う。
- sequential-loaderリポジトリの変更。
- リポジトリの `CLAUDE.md` が存在しない `AGENT.md` を参照している件（別に扱う）。

## 4. ファイル構造（変更後）

```text
main_moco.py                        学習の入口（Hydra、config_path="conf/moco"）
conf/moco/
  config.yaml                       defaults：trainer、model、data、moco、diagnostics、training、precision
  trainer/  cpu.yaml  cpu_ddp2.yaml  gpu1.yaml  gpu2.yaml
  model/    lora.yaml  full_ft.yaml
  data/     imagenet.yaml  anet_frames.yaml  anet_clips.yaml
  diagnostics/  off.yaml  probe.yaml
  experiment/   l4.yaml  l5.yaml  a1.yaml  a2.yaml  clip_a1.yaml  clip_a2.yaml  g1.yaml  g2.yaml
model/vit/
  vit_moco_backbone.py、lora_linear.py、vit_lora_frame_encoder.py   （中身はそのまま）
  __init__.py                       ViTMoCoBackbone、ViTMoCoClipBackbone、inject_attention_lora、TRAIN_MODES を書き足す
self_supervised/moco/
  standalone_moco.py、ddp_moco.py、vit_ddp_moco.py   （中身はそのまま）
  __init__.py                       ViTDistributedMoCo を書き足す。説明文の「ViTなし」を直す
dataset/
  moco_transforms.py                TwoCropsTransform、ClipTransform、SequentialClipTransform、PILGaussianBlur、two_crops
  activitynet_frames.py             scripts/vit_moco/ から移す（manifest作成のCLIを含む）
  activitynet_sequential.py         scripts/vit_moco/ から移す（build_loader、video_sources）
  moco_data.py                      build_dataset / build_loader の入口。data.source と data.order で切り替える
  __init__.py                       上の公開名を書き足す
utils/
  hashing.py                        file_sha256、tensor_sha256、state_sha256、sha256_json
  provenance.py                     REPOSITORY_ROOT、git_info、environment
  jsonio.py                         JsonLines、read_jsonl、write_json_atomic
  rng.py                            capture_rng_state、restore_rng_state、epoch_seeds、seed_epoch
  runtime.py                        free_port、sync、gpu_snapshot、peak_memory、apply_backend_flags
  __init__.py                       上の公開名を書き足す
training/                           新しいパッケージ（今は中身のない既存ディレクトリ）
  __init__.py
  config.py                         設定の検査（能力の検査だけ。§5.3）、config_identity（旧形式を保つ）、古い設定の読み替え（§5.6）
  model_factory.py                  build_model、pretrained_identity、frozen_state、optimizer_record
  step.py                           train_step、step_metrics、global_norm
  loop.py                           epoch型のループ、sequential型のループ（train_sequential）、epoch_cosine_lr
  checkpoint.py                     save_checkpoint、resume、checkpoint_file、sidecar_file（形式 v1 のまま）
  probe.py                          学習中の固定probe（QueryCollapseProbe、UpdateMonitor、統計関数、write_summary）
  report.py                         final_checks、vram_gate、run_report
  launcher.py                       run()、_worker()、プロセスの起動（trainer.accelerator と trainer.devices）
evaluation/                         新しいパッケージ（今は中身のない既存ディレクトリ）
  __init__.py
  temporal_order.py                 今の temporal_order_eval.py（argparseのCLIのまま）
  collapse_compare.py               今の query_collapse_diagnostic の比較CLI（compare_runs、compare_learning_rates、compare_prefix）
scripts/parity/、scripts/smoke/     ロジックはそのまま。上の道具をimportするように書き換える（R7）
```

- 消すもの：`scripts/vit_moco/`（6ファイル）、`conf/moco_vit_*.yaml`（7ファイル）。
- `main_moco.py` の位置づけは、旧フレームワークの `main.py` と同じく、リポジトリ直下の入口とする。
- `utils/__init__.py` と `dataset/__init__.py` は、旧フレームワークのモジュール（Comet ML、pytorchvideo、Lightning）も一括でimportする。新しい本流もこれを経由するので、読み込みが遅くなることは受け入れる（R6）。ただし、これらが入っていない環境では本流も動かない点に注意する。

## 5. 実装要件

### 5.1 設定の構造

| 節 | 主なキー | 備考 |
|---|---|---|
| `trainer` | `accelerator: cpu｜cuda`、`devices`（プロセス数）、`backend`（cpuならgloo、cudaならnccl）、`timeout_minutes`、`broadcast_buffers` | `cpu.yaml`＝1プロセス、`cpu_ddp2.yaml`＝2プロセス（テストと正解データ用）、`gpu1.yaml`、`gpu2.yaml`。GPUの選び方は今と同じく `CUDA_VISIBLE_DEVICES` |
| `model` | `train_mode`、`checkpoint_id`、`projection_dim`、`lora.*`、`gradient_checkpointing` | 今と同じ値 |
| `data` | `source: imagenet｜activitynet_frames｜activitynet_clips`、`order: shuffle｜sequential`、`clip.frames`、`clip.interval_sec`（shuffleで使う）、`clip.frame_stride`（sequentialで使う）、`clip.anchor`、`activitynet_manifest`、`sequential.*`、`root`、`split`、`image_size`、`num_workers` | `order: sequential` は `source: activitynet_clips` のときだけ有効。内部では、今の `activitynet_sequential_clips` に対応する |
| `moco` | 今の `moco` 節と同じキーと値 | 同一性ハッシュにそのまま入るので、キーを変えない |
| `training` | `seed`、`global_batch_size`（256）、`learning_rate`、`lr_schedule: epoch_cosine｜constant`、`epochs`、`weight_decay`、`betas`、`eps`、`max_steps`、`checkpoint_steps`、`resume` | `local_batch_size` は `global_batch_size ÷ trainer.devices` で計算する。割り切れなければ拒否する |
| `precision` | 今の `precision` 節と同じ | 同一性ハッシュにそのまま入る |
| `diagnostics` | `enabled`、`probe_steps`、`probe.*`（今の値） | `off.yaml` と `probe.yaml` |
| `experiment_name` | 文字列（既定は `custom`） | 出力先 `log/moco/${experiment_name}/${model.train_mode}/<日時>` |

- `experiment/*.yaml` は、上のgroupの選択と値の上書きだけを書く。冒頭のコメントに、元にした実験のspecとEvidenceへのリンクを書く。
- 8つのexperimentファイルの値は、それぞれ今のpresetで解決される値と同じにする。ただし、出力先、`trainer` の表し方、`global_batch_size` の表し方、許可リストに関わる項目は新しい形にする。
- 使い方の例：

  ```bash
  python main_moco.py +experiment=g2 trainer=gpu1 data.clip.frame_stride=32
  ```

  experimentの指定方法（`+experiment=` か `experiment=` か）は、Hydraの書き方に合わせて実装時に決める。

### 5.2 計算を変えないこと（bitwise一致の条件）

- seedの順番：seed → backbone → もう一度seed → projectorとQueue。
- 乱数を使う順番：augmentation（`ClipTransform` で乱数状態を戻す処理を含む）、shuffle BN、epochごとのseed、loaderのworkerのseed。
- forwardの計算順、`meta_einsum`、Queueのコピーとenqueueのタイミング、EMA。
- DDPの包み方、`DistributedSampler`（shuffle、seed 0）、`drop_last`、`generator`、epochごとのLRの設定タイミング。
- クラス名と `__repr__` の文字列（`ClipTransform(one draw per clip)`、`PILGaussianBlur(sigma=...)` など）。
- パラメータ名（state_dictのキー）。

### 5.3 設定の検査（能力の検査だけを残す）

**残す**検査：

- `train_mode`、`data.source`、`data.order` の値が正しいこと。`order: sequential` は `activitynet_clips` のときだけ。
- 必要なキーがあること（manifest、clip、sequential）。
- FP32でAMPなし（今のコードはこれ以外に対応していない）。
- `broadcast_buffers: true`。
- `moco.queue_size % global_batch_size == 0`。`global_batch_size % devices == 0`。
- `max_steps >= 1`。`checkpoint_steps` が `[1, max_steps]` に入っていること。
- `accelerator` と `backend` の組み合わせ（cpuならgloo、cudaならnccl）。見えるGPUの数が `devices` 以上であること。
- 診断ありのresumeは、`order: sequential` のときだけ。
  - 理由：今のコードでは、epoch型のresumeだとprobe step 0の意味が変わるため。今の挙動を保つ。
- GPU runはgitがcleanであること。出力先に既存の成果物があれば拒否すること。
- `data.sequential.video_dir`：フォルダの `*.mp4` を読むdata sourceとして、CPUでもGPUでも使えるようにする（テスト専用の扱いを外す）。

**消す**検査：

- `ADOPTED_RECIPE`、`GPU_MAX_STEPS`。
- presetごとの許可リスト（`*_GPU_RUNS`、`*_ADOPTED`、`LONG_*`）。
- 「このdata sourceはこのpresetだけ」の検査。
- `world_size >= 2`。
- `check_lr_preset`。

### 5.4 GPUの数の切り替え

- `trainer.devices` 個のプロセスを `mp.spawn` で起動する（1でもprocess groupを作る）。今のDDPとMoCoの経路を1本のまま使う。
- 1プロセスで `concat_all_gather` などが動くことは、CPUテストで確かめる。
- 1GPUでlocal 256がメモリに載らないときは、OOMのエラーで止まる。今と同じく、`rank*_report.json` に記録する。batchの分割やAMPに自動で切り替えることはしない。

### 5.5 記録

- `run_report.json` に次を加える。
  - `experiment_name`
  - コマンドラインの上書き（Hydraの `overrides.task` の一覧）
  - `trainer`（accelerator、devices）
- 今ある記録（`config_identity`、`data`、`provenance`、`vram_gate`、`verdicts` など）は、同じキーで残す。
- `config_identity` は、新しい設定から**今と同じ形の辞書**を組み立てる。
  - `local_batch_size`、`world_size` などのキー名もそのまま。
  - data sourceの名前も今の名前（`activitynet_sequential_clips` など）。
  - こうすれば、同じ実験なら、既存checkpointの `identity_sha256` と一致する。

### 5.6 既存のrunとの互換

- `evaluation/temporal_order.py` と `training/checkpoint.py` は、**既存のrunの `resolved_config.yaml`（古い形式）**も読めるようにする。
  - `training/config.py` に、古い形式を新しい形式へ読み替える関数を1つ置く。
  - 古い形式のrunは、R1・G2の評価と、G2の続きのresumeで使う。
- `evaluation/temporal_order.py` の `--clip-config` は、古い形式と新しい形式のどちらのclip runの設定でも受け付ける。

### 5.7 parityとsmokeの書き換え

- `scripts/parity/`、`scripts/smoke/` の中で、§4で移した道具を定義している箇所は、新しい場所からimportするようにする。
- 移した道具の定義は、元のファイルから消す。parity側の他のコードと、テストが使う名前は、importで引き続き見えるようにする。
- parity固有のもの（`import_reference`、state mapping、adapter、比較器、launcherなど）は動かさない。

### 5.8 テスト

- 本流に関わる9ファイルを、新しいimport先・experiment名・設定の構造に合わせて移す。
  - 確かめている性質は残す：shape、数値、resume、rank一致、epochの終わり、padding、probe、評価の集計など。
- 消すテスト：presetの許可値、GPUのstep数、presetだけのdata source・resumeを確かめるもの。
  - 消したテストの一覧は、実装記録に残す。
- テスト同士のimport（`write_video`、`workspace`、`load_anet_cfg` など）は、`test/` の共通ヘルパー（例：`test/moco_helpers.py` と `conftest.py`）にまとめる。
- 追加するテスト：
  - `trainer` の cpu（1プロセス）・cpu_ddp2 で、小さなrunが完走すること。
  - `data.order` の切り替え。
  - 古い形式の `resolved_config.yaml` の読み替え。
  - 正解データとの一致（§8 V-C）。

## 6. 手順

1. **正解データを取る（基準commit `559073f` のまま、CPUのみ）**
   - 正解データを取るスクリプトを1つ作り、`test/refactor_golden/` に置く。
   - 入力は固定のパスに作る：小さなViT、小さなImageFolder、合成動画、manifest。`log/refactor_golden/fixtures/` に置く。一時ディレクトリだと、データのダイジェストと同一性ハッシュが毎回変わってしまうため。
   - 次のCPU（gloo、2プロセス）runを今のコードで実行する。

     | データ | 学習方式 | 備考 |
     |---|---|---|
     | ImageNet | LoRA、Full FT | 診断あり |
     | ActivityNet単一フレーム | LoRA | |
     | ActivityNet clip | LoRA、Full FT | |
     | sequential clip | LoRA | 途中のcheckpointと、epochの境界・epochの途中からのresumeを含む |

   - 保存するもの：`metrics.jsonl`、`run_report.json`、`diagnostic_summary.json`、`probe_metrics.jsonl`、checkpoint、`loader_epochs`。保存先は `log/refactor_golden/559073f/`。
   - スクリプトは別のcommitにする。
2. **作り直す**：§4と§5のとおりに作る。
3. **CPUで確かめる**：§8の V-C、V-T、V-P。
4. **commitする**（pushはしない）。
5. **GPUで確かめる**（別に許可を取る）：§8の V-G。

## 7. 互換性と既存の実験への影響

- 同じ設定なら、計算はbitwise一致する（R1）。既存の結果との比較はそのまま有効。
- 既存checkpointの互換：
  - G2などの既存checkpointは、新しいコードでresumeできる（同一性ハッシュが一致すること）。
  - 既存checkpointを評価に読み込める（strictで読み込めること）。
- 実行コマンドは変わる（R8）。過去の記録にある古いコマンドは、記録にあるcommitで再現する。
- 出力先は変わる（R12）。過去の出力はそのまま残す。
- 1GPU・CPUで学習した結果は、2GPUの結果とbitwise一致しない（shuffle BNの単位が変わるため）。これは新しい実験条件として扱う。

## 8. 成功条件

### 実装時（CPUのみ、この承認の範囲で実行してよい）

- **V-C（正解データとの一致）**：§6の1で取った各runについて、新しいコードで同じ設定のrunを実行し、次がbitwise一致すること。
  - `metrics.jsonl` の全step（`step_time_sec` と `data_wait_sec` などの時間の値を除く）
  - `final_state_sha256`
  - `config_identity_sha256`
  - probe特徴のsha256
  - checkpointのmodel state
  - resumeしたrunが、resumeしなかったrunと一致すること
- **V-T（テスト）**：
  - 移したテストと追加したテストがすべて通る。
  - 旧フレームワーク側のテスト（今の失敗ID：image models、model factory、checkpoint、video models、video folder）の失敗IDの集合が変わらない。
  - parityとsmokeのテストは、`test_moco_5epoch_pair.py` の今の失敗1件を除いて通る。
- **V-P（既存のrunとの互換、CPUで読み取りのみ）**：
  - G2のcheckpoint（step 6,255）を、新しいコードで `experiment=g2` から作ったモデルにstrictで読み込める。
  - G2の設定から新しいコードで計算した `config_identity_sha256` が、G2の記録（`9fde5def…`）と一致する。
  - 既存のR1・G2のrunの古い形式の `resolved_config.yaml` を、評価コードが読み込める。
- **V-I（独立していること）**：
  - 本流（`main_moco.py`、`training/`、`evaluation/`、`dataset/`・`utils/` の新しいファイル）が、`test/`、`scripts/parity/`、`scripts/smoke/` をimportしない（grepで確認）。

### GPUでの確認（実装の後、別に許可を取る）

- **V-G（正解との一致）**：新しいコードで、次の各設定をGPU 2枚（`CUDA_VISIBLE_DEVICES=0,1`）で1 stepずつ実行し、既存のrunの記録とstep 1のlossとstep 0のprobe特徴のsha256がbitwise一致すること。

  | 設定 | 既存のrun |
  |---|---|
  | L4 LoRA | `moco_vit_query_collapse_lr/lora/imagenet/lr_0.0003/2026-10-31_22-32-03-330318` |
  | L4 Full FT | `moco_vit_query_collapse_lr/full_ft/imagenet/lr_0.0003/2026-10-31_22-38-39-863638` |
  | A1 LoRA | `moco_vit_lr_activitynet/lora/activitynet_frames/lr_0.0003/2026-11-02_13-57-26-045737` |
  | A1 Full FT | `moco_vit_lr_activitynet/full_ft/activitynet_frames/lr_0.0003/2026-11-03_10-31-07-443680` |
  | clip A1 LoRA | `moco_vit_lr_activitynet_clip/lora/activitynet_clips/lr_0.0003/2026-11-03_15-01-35-356834` |
  | clip A1 Full FT | `moco_vit_lr_activitynet_clip/full_ft/activitynet_clips/lr_0.0003/2026-11-03_15-12-06-149049` |
  | G1 | `moco_vit_seq_activitynet_clip/lora/activitynet_sequential_clips/lr_0.0003/2026-11-05_02-00-34-914687` |

  - 一致しないときは、基準commit `559073f` で同じ1 stepを実行して、記録との差がコードの変更によるものか、記録した時点からの環境やコードの違いによるものかを切り分ける（R11）。
- **V-D（GPUの数の切り替え）**：`trainer=gpu1` で、ImageNet LoRAの1〜2 stepが完走するか、OOMのエラーとして正しく止まって記録されること（どちらになったかを記録する）。

### 主張しないこと

- 1GPU・CPUでの学習結果が2GPUと同じになること。
- 新しい構造での性能の向上（計算は変えない）。

## 9. 実行の境界

- 実装の承認は、§6の1〜4（CPUでの正解データ取り、作り直し、CPUでの確認、commit）までを含む。
- GPUでの確認（V-G、V-D）は、それぞれ別に許可を取る。学習のrunは行わない。
- pushは行わない。

## 10. 未決事項

### blocking

- なし（2026-11-11時点）。

### non-blocking（実装時に、既存のstyleに合わせて決める）

- experimentの指定方法（`+experiment=` か `experiment=` か、Hydraの `@package _global_` の使い方）。
- `utils/runtime.py` などの細かいファイル名と、関数の振り分け。ただし §4 の層の分け方（utils・dataset・training・evaluation）は変えない。
- 正解データのrunのstep数（小さくてよい。epochの境界とresumeを含むこと）。
- `_sha256_json` などの非公開名を公開名にするときの名前。同一性ハッシュには入らない。

## 11. 承認記録

- 2026-11-11：draftを作成した（R1〜R12は、2026-11-10〜11の壁打ちと質問への回答でユーザーが採用）。
- 2026-11-11：ユーザーが承認し、実装を指示した（approved）。GPUでの確認（V-G、V-D）は別に許可を取る。
- 2026-11-11：実装した（`c478269` 正解データ、`59bade2` 本体、pushなし）。V-C：30 passed（10本の正解runとbitwise一致。mainプロセスのPython/numpyのRNGは一度もseedされず使われないので比較から外し、使われないことを確かめた）。V-T：全テストの失敗・エラーのIDは変更前に含まれるものだけ（196 failed / 778 passed / 18 errors）。V-P：G2の同一性ハッシュ一致・strict読み込み可。V-I：本流は test/・scripts をimportしない。GPUでの確認（V-G、V-D）は未実施。
- 2026-11-11：GPUでの確認を、ユーザーの許可を得て実行した。V-G：7設定すべてで、step 1のlossとstep 0のprobeが既存のrunの記録とbitwise一致した（勾配ノルムは約1e-13のずれで、既知のGPU非決定性）。V-D：`trainer=gpu1` でImageNet LoRAの2 stepが完走した（GPU 0、headroom 8,718 MiB）。
