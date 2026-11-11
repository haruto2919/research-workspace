---
project: sequential-video-lora-analysis
date: 2026-11-10
analysis_mode: explain
detail: full
level: mid
source_repository: /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
source_commit: 9285a474c41922891f73095d040e4c4e61fd0272
source_branch: feature-vit-moco
source_dirty: false
external_dependency: /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential_loader (feature-stride-ddp@ce4dcbc)
targets:
  - scripts/vit_moco/
  - self_supervised/moco/
  - model/vit/
  - scripts/parity/
  - scripts/smoke/
  - conf/
  - main.py, main_pl.py, train.py, val.py, model/, dataset/, setup/, utils/, logger/, callback/, integration/
related_spec: null
---

# Code Deep Dive: リファクタリング前の実装全体の構造

## 対象と解析の範囲

- 目的：リファクタリングに向けて、研究コードのリポジトリが今どういう構造になっているかを把握する。
- 対象：`2026_09_ishikawa_sequential-video-lora` の追跡ファイル全体（約2.5万行、うちテスト約1.1万行）。外部の `sequential_loader` リポジトリは、研究コードとの接点だけを扱う。
- 除外：`log/`（実行結果）、`hub/`（torch.hubのキャッシュ）、`.venv`、loaderリポジトリの内部実装。
- 方法：本流（`scripts/vit_moco/`、`self_supervised/moco/`、`model/vit/`）は直接読んだ。旧フレームワークとparity群は、読み取り専用のサブエージェント2つに調べさせた。主要な主張（PEFTを使っていないこと、`model/__init__.py` が一括でimportすること）は直接確認した。
- コードの実行、テスト、学習は行っていない。

## Overview：一言でいうと

このリポジトリには、**時期の違う3つの層が地層のように重なっています**。

| 層 | 時期 | 中心のファイル | 今の状態 |
|---|---|---|---|
| ① 汎用の分類フレームワーク | 〜9月 | `main.py`、`main_pl.py`、`train.py`、`model/`、`dataset/`、`setup/`、`utils/`、`logger/` | **使われていない**（テストでだけ動く） |
| ② MoCoの公式一致検証 | 10月中旬 | `scripts/parity/`（約5,800行） | 検証は完了。ただし**本流が部品を借りている** |
| ③ 本流：ViT＋MoCo＋LoRA | 10月下旬〜 | `scripts/vit_moco/`、`self_supervised/moco/`、`model/vit/` | **今の研究はすべてここ** |

これに加えて、CIFAR-10などで動作を確かめた `scripts/smoke/` があり、外部の `sequential_loader` リポジトリも使っています。

## Key Components and Data Flow

### 1. 実際に学習を動かすと、何がどの順で呼ばれるか

G2（sequential-loaderでのclip LoRA学習）を例にします。

```text
python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_seq_activitynet_clip model.train_mode=lora training.max_steps=6255
│
├─ Hydraが conf/ から設定を合成（下の「設定の継承」参照）
│
└─ main() → run()                                   moco_vit_ddp_smoke.py:702
     ├─ prepare_config()  設定の検査（GPUで許す値・step数）   :115
     ├─ 出力先に既存の成果物があれば中止、GPU runはgitがcleanでないと中止
     ├─ mp.spawn(_worker, nprocs=2)  GPUごとに1プロセス      :721
     │    └─ _worker(rank)                                  :637
     │         ├─ dist.init_process_group（NCCL）
     │         └─ train()                                    :489
     │              ├─ build_model()  ViT＋LoRA＋MoCoを作る    :197
     │              ├─ AdamW（学習するパラメータだけ）
     │              ├─ two_crops()  aug-plus（2つのview）     :262
     │              ├─ データ：sequentialなら activitynet_sequential.build_loader()
     │              │          それ以外は build_dataset() ＋ DistributedSampler
     │              ├─ QueryCollapseProbe（固定probeでの診断、有効なときだけ）
     │              ├─ resume（指定されたとき）
     │              └─ 学習ループ
     │                   ├─ sequential：train_sequential()     :433
     │                   └─ それ以外：epochごとのループ          :585〜
     │                        └─ run_step()  1 stepの処理         :554
     │                             ├─ 全rankのデータエラーを集める
     │                             ├─ train_step()               :395
     │                             │    └─ ViTDistributedMoCo.forward()
     │                             ├─ Queueのpointerを検査
     │                             ├─ metrics.jsonl に1行書く
     │                             ├─ checkpoint（決めたstepで）
     │                             └─ probe（決めたstepで）
     │              └─ final_checks()  全rankが同じ状態か、凍結部分が不変か
     └─ rankごとの報告をまとめて run_report.json、vram_gate、診断のsummary
```

### 2. 1 stepの中身（MoCoのforward）

`self_supervised/moco/ddp_moco.py:128` の `DistributedMoCo.forward`：

1. 全rankのbatchサイズが同じか、入力が正しいかを検査する。
2. query側：`backbone_q`（ViT＋LoRA）→ `projector_q`（768→768→128）→ L2正規化 → `q`
3. key側（勾配なし）：
   - key encoderをEMAで更新する
   - batchをrank間で混ぜて（shuffle BN）key encoderに通し、元の順に戻す → `k`
4. Queueのコピーを使って、logits（正例1個＋negative 65,536個）とInfoNCE lossを計算する（`meta_einsum` 実装）。
5. 今回のkeyを全rank分集めてQueueに入れる（次のstepからnegativeになる）。

`train_step`（`moco_vit_ddp_smoke.py:395`）が、この後に `backward` と `optimizer.step` を行い、loss・正例cos・最も似たnegativeのcos・`query_feature_std` などを記録します。

### 3. 部品ごとの説明

#### モデル（`model/vit/`）

| ファイル | 中身 |
|---|---|
| `vit_moco_backbone.py` | `ViTMoCoBackbone`：HuggingFaceの `google/vit-base-patch16-224` を読み込み、CLS特徴（768次元）を返す。`train_mode=lora` ならViTを凍結してLoRAを入れ、`full_ft` ならViT全体を学習する。`ViTMoCoClipBackbone`：4フレームのpatch tokenを1列に並べてViTに通す（token結合のEarly Fusion）。時間の位置埋め込みは固定のsin-cos（`temporal_embedding`、t=0で0なので、1フレームなら画像ViTと一致する） |
| `vit_lora_frame_encoder.py` | `inject_attention_lora`：ViTの各blockのQ/V（設定次第でK）の `nn.Linear` を `LoRALinear` に置き換える。置き換える前に全対象を検査する |
| `lora_linear.py` | **自作のLoRA**（34行）：`base(x) + (alpha/rank) * B(A(x))`。Aはkaiming初期化、Bは0初期化 |

> 注：READMEの9月の記録には「PEFT LoRA」とあるが、今のコードにPEFTは見当たらない（`peft` という文字列は .py/.txt/.toml のどこにもない）。今使われているLoRAは自作の `LoRALinear` だけ。

#### MoCo（`self_supervised/moco/`）：3段の継承

```text
StandaloneMoCo        standalone_moco.py:79   1GPU版。query/key encoder、projector、EMA、FIFO Queue、InfoNCE
  └ DistributedMoCo   ddp_moco.py:88          複数GPU版。shuffle BN、全rankのkeyをQueueへ、meta_einsum
      └ ViTDistributedMoCo  vit_ddp_moco.py:20  ViT用。凍結パラメータはEMAしない、clip入力 [B,T,C,H,W] の検査
```

- `StandaloneMoCo` の作成時に、query側をdeepcopyしてkey側を作り、Queueをランダムな単位ベクトルで埋める（`standalone_moco.py:96-104`）。
- `DistributedMoCo` は、Meta公式とbitwise一致することを確認した実装（parity群で検証済み）。

#### データ：3つの経路

| `data.source` | 中身 | ファイル |
|---|---|---|
| `imagenet` | torchvisionの `ImageFolder` | `moco_vit_ddp_smoke.py:300` |
| `activitynet_frames` / `activitynet_clips` | 事前に作ったmanifest（10秒間隔の時刻）を、1件ずつseekしてdecodeする（PyAV）。clipは0.25秒間隔の4フレーム | `scripts/vit_moco/activitynet_frames.py` |
| `activitynet_sequential_clips` | 外部の `sequential_loader` で、動画を先頭から順番に読む。1 rankあたり128本の動画を同時に1 clipずつ進める。augはloaderのworkerの中でかける | `scripts/vit_moco/activitynet_sequential.py` |

- 前の2つは普通の `DataLoader` ＋ `DistributedSampler` で、epochの長さが前もって分かる。
- sequentialは、epochの長さがepochの終わりまで分からない。そのため学習ループが別関数（`train_sequential`）になっている。

#### 診断と評価

| ファイル | 中身 |
|---|---|
| `scripts/vit_moco/query_collapse_diagnostic.py`（901行） | 固定のImageNet 64枚（probe）で、CLS・`z_q`・`q` の特徴の散らばりを決めたstepごとに測る。1 stepのパラメータ更新量も測る。**それに加えて、各presetで「GPUで許す値・step数」のルールもここに置かれている**（`ACTIVITYNET_GPU_RUNS`、`SEQUENTIAL_ADOPTED` など） |
| `scripts/vit_moco/temporal_order_eval.py`（405行） | 学習済みモデルで、clipの並び順（normal・shuffle・reverse・static）による特徴の変化を測る評価（独立したCLI、argparse） |

#### 設定（`conf/`）：presetの継承

Hydraの `defaults` で、次のように1本の鎖になっています。

```text
moco_vit_ddp_smoke                 基本（ImageNet、LR 0.03、5epoch cosine）
 └ moco_vit_query_collapse         ＋固定probeの診断
    └ moco_vit_query_collapse_lr   ＋LRを指定できる
       ├ moco_vit_lr_long          1,000step
       └ moco_vit_lr_activitynet   ActivityNet単一フレーム
          ├ moco_vit_lr_activitynet_clip    4フレームclip
          └ moco_vit_seq_activitynet_clip   sequential-loader（G2）
```

新しい実験を足すたびに、presetが1段ずつ下に増えてきた構造です。どのpresetも、上の段の設定を全部引き継ぎます。

### 4. 本流が他の層から借りているもの

本流（`scripts/vit_moco/`）は、③の層だけで完結していません。

| 借りている先 | 借りているもの | どこで |
|---|---|---|
| `scripts/parity/moco_imagenet_10epoch.py` | `JsonLines`、`epoch_seeds`、`gpu_snapshot`、`peak_memory`、`seed_epoch`、`write_json_atomic`、`read_jsonl` | smoke `:64`、diagnostic `:58` |
| `scripts/parity/moco_imagenet_parity.py` | `PILGaussianBlur`、`_free_port`、`_global_norm`、`_sha256_json`、`_sync`、`apply_backend_flags`、`epoch_cosine_lr`、`step_metrics` | smoke `:67`、diagnostic `:59`、eval `:39`、sequential `:19` |
| `scripts/parity/moco_parity_state.py` | `REPOSITORY_ROOT`、`environment`、`file_sha256`、`git_info`、`state_sha256`、`tensor_sha256` | ほぼ全ファイル |
| `scripts/smoke/smoke_moco_cifar10.py` | `TwoCropsTransform`、`capture_rng_state`、`restore_rng_state` | smoke `:72`、diagnostic `:61` |
| `model/vit/` | `ViTMoCoBackbone` など、`inject_attention_lora`、`LoRALinear` | smoke `:63` |

借りているのは、ハッシュ、git情報、JSONの書き出し、乱数、LR schedule、blur、metricsといった**汎用の道具だけ**です。Meta公式との比較のためのadapter、比較器、launcherは使っていません。

### 5. 使われていない層（①）の中身

- `main.py` / `main_pl.py` / `train.py` / `val.py`：Hydraで分類を学習するループ（DataParallel版とLightning版）。テスト（`test/config/`）からしか呼ばれない。
- `model/`：ResNet、X3D、ABN、分類用ViT（`ViTb`）、`configure_model` などのモデル集。
- `dataset/`、`setup/`（optimizer・scheduler）、`utils/`（精度、AverageMeter、Cometのcheckpoint）、`logger/`、`callback/`：すべて `main*.py` 用。
- `integration/sequential_vit.py`：古いsequential-loaderとframe encoderの橋渡し。今は `activitynet_sequential.py` が後を継いでいると推測。
- `model/vit/vit_frame_encoder.py`、`model/masked_mean_clip_aggregator.py`：9月のsmoke用。

## Concepts and Design Decisions

コードから読み取れる設計の方針です（理由は推測を含みます）。

1. **再現性を最優先している**
   - seedの順番が決まっている：seed → backbone（LoRA Aの初期化）→ もう一度seed → projectorとQueue（`build_model`、`:197`）。こうすると、LoRAとFull FTで同じprojector・Queueから始まる。
   - 入力のsha256を毎stepで記録している。checkpointには設定の同一性ハッシュが入っていて、resumeのときに照合する。
2. **GPU runに強いガードがある**
   - gitがcleanでないとGPU runを始めない（`run`、`:713`）。
   - presetごとに、許されたstep数・LR以外は拒否する（`prepare_config`、`:149-161`）。
   - 出力先の成果物を上書きしない。
3. **失敗は全rankで同時に止める**：データエラーや非有限値は、全rankで集めてから止める（タイムアウトで固まらないようにするため）。
4. **実行記録をファイルに残す**：Cometは使わず、run directoryの `run_report.json`、`metrics.jsonl`、`rank*_report.json` が記録の正本。

## Code Walkthrough：リファクタリングで気をつける点

### 構造上の問題（事実）

| # | 問題 | 根拠 |
|---|---|---|
| P1 | **本流が、完了した検証スクリプトに依存している**。parityの1,565行のファイルから、`_` で始まる非公開の関数まで借りている | 上の「借りているもの」の表 |
| P2 | **`model` をimportすると、使っていないLightningとComet MLまで読み込まれる**。`model/__init__.py` が `SimpleLightningModel` を一括でimportし、そこから `utils` → `utils/checkpoint.py` → `comet_ml` とつながる | `model/__init__.py:15`、`simple_lightning_model.py`、`utils/__init__.py`、`utils/checkpoint.py` |
| P3 | **`moco_vit_ddp_smoke.py` は名前が「smoke」なのに本番のrunner**で、769行に設定検査、モデル作成、データ作成、2種類の学習ループ、checkpoint/resume、プロセス管理、レポート作成が全部入っている | ファイル全体 |
| P4 | **診断モジュールが実験のルールも持っている**。`prepare_config` は「このpresetか」「GPUで何stepを許すか」を `query_collapse.*` に問い合わせる | `moco_vit_ddp_smoke.py:126-158`、`query_collapse_diagnostic.py:89-118` |
| P5 | **runnerの骨組みが3回コピーされている**（`_worker`、`run`、`train_step`、`resume`）：parity、10epoch、ViT smoke | parityの調査（中身のdiffはViT smoke分は未確認） |
| P6 | **checkpointの仕組みが5種類、設定の方式が4種類ある**（Hydraの合成、Hydraの単独yaml、argparse、OmegaConf.load） | 旧フレームワークの調査 |
| P7 | presetの鎖が深い（最大5段）。G2の設定を知るには、5つのyamlを上から順に読む必要がある | `conf/` |
| P8 | 小さなもの：`zero_outout_model.py`（typo）、中身が `__pycache__` だけの空ディレクトリ（`evaluation/`、`training/`、`scripts/linear_probe/`、`scripts/moco/`）、`self_supervised/moco/__init__.py` の説明文が「ViTなし」のまま、`conf/moco_vit_seq_activitynet_clip.yaml` の冒頭コメントが2回重複 | 各ファイル |

### リファクタリングで壊してはいけない性質

- **数値の同一性**：seedの順番、乱数の消費順、`meta_einsum` のlogits計算。ここを変えると、これまでのrun（L4/L5、A1/A2、G2）と比べられなくなる。
- **checkpointの形式**（`moco_vit_ddp_smoke_checkpoint/v1`）と設定の同一性ハッシュ：変えると既存のcheckpointからresumeできなくなる。temporal_order_evalも、既存のcheckpointを読み込んで使う。
- **GPUのガード**：clean commitでないと動かないこと、presetごとの許可。
- **テストの基準**：11/04時点で「36 failed / 972 passed」がこれまでの基準（既知の失敗を含む）。リファクタリングの前後で、失敗するテストIDの集合が変わらないことを確かめる。

## Alternatives and Trade-offs（整理の方向性の候補）

以下は候補で、決定ではありません（spec化するなら `research-spec` で決める）。

1. **共通の道具を1か所にまとめる**：`moco_parity_state.py` と parity・smokeから借りている関数を、`common/`（仮）のような共有モジュールに移す。parityと本流の両方がそこを使うようにする。
   - 利点：依存の向きが「古い検証 → 本流」ではなくなる。
   - 注意：parity側のテスト（約3,000行）も一緒に直す必要がある。
2. **`model/__init__.py` の一括importをやめる**：P2を解消する一番小さい変更。
3. **runnerを分割する**：設定検査、データ作成、学習ループ、checkpoint、レポートを別ファイルにする。
4. **実験のルール（preset）を診断から分ける**：「どのpresetで何を許すか」を、`query_collapse_diagnostic.py` から独立させる。
5. **使っていない層①を片付ける**：消す、`legacy/` に移す、別branchに残す、のどれか。テスト（`test/config/` など）も一緒に扱う。

## 根拠・未確認事項

- Observed（直接読んで確認）：
  - 学習の呼び出し順と1 stepの処理
  - MoCoの3段の継承
  - ViT/clip backboneと自作LoRA
  - データの3経路
  - presetの継承の鎖
  - 本流のimport
  - `model/__init__.py` が一括でimportすること
  - PEFTが見当たらないこと
- Inferred（推測）：
  - 設計の理由（再現性の優先など）
  - `integration/sequential_vit.py` は使われなくなった
  - Lightning/Cometの読み込みは実行時に害がない（読み込みが遅くなるだけと推測）
- Unverified（未確認）：
  - P5のViT smoke分は関数の形だけで判断しており、中身のdiffは取っていない
  - テストは実行していない
  - 外部の `sequential_loader` リポジトリの内部構造は対象外
