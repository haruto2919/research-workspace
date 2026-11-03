---
project: sequential-video-lora-analysis
date: 2026-11-04
record_type: implementation-cpu-verification
status: cpu-tests-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
base_commit: fe58fc37ef28901184980ccd5f9e90e40c3d836d
code_state: committed
implementation_commit: 50ff715773137388ba51850532660d26daa4501a
spec: ../specs/2026-11-03-temporal-order-sensitivity-evaluation-spec.md
---

# 時間情報の評価（並び順への反応）：実装とCPU検証

[承認済みspec](../specs/2026-11-03-temporal-order-sensitivity-evaluation-spec.md)を、研究コード `feature-vit-moco@fe58fc3` の上に実装した。

- **CPUテストT1〜T6はすべてPASS**した。全テストは36 failed / 972 passedで、失敗したテストの集合は変更前と**完全に一致**した（passedが増えた15件は、新しく作ったテストの数と一致）。
- 研究コードは `feature-vit-moco@50ff715` にcommitした（2026-11-04、3ファイル。内容はCPU検証時と同じ）。
- R0（validation manifestの作成）は2026-11-04に実行した（4,926本、58,163時刻）。R1（GPUでの評価）も2026-11-04に実行した：[R1 Evidence](2026-11-04-temporal-order-sensitivity-evaluation-r1-gpu-evidence.md)

## 変更したファイル

| ファイル | 内容 |
|---|---|
| `scripts/vit_moco/activitynet_frames.py` | I1：`--split` で `validation` を選べるようにした。既定は `training` のまま。docstringにvalidationの使い方を1行加えた（+3／−1行） |
| `scripts/vit_moco/temporal_order_eval.py`（新規、405行） | I2：spec §3〜§7を実装した評価CLI。<br>I3：fixed probeの特徴が、学習時の記録と一致するかを確かめる |
| `test/model/test_temporal_order_eval.py`（新規、196行） | I4：T1〜T6の15件 |

### 実装の要点

- **モデルの組み立て**：どのモデルも、`--clip-config`（clip runの `resolved_config.yaml`）と同じclipの構造で組み立てる。train_modeだけはモデルごとに指定する（`NAME=MODE:PATH|init`）。
- **checkpointの読み込み**：`model` のstate dictをstrictに読み込む。読み込む前に、checkpointのidentity（train_mode、checkpoint_id、projection_dim）が、組み立てたモデルと同じかを確かめる。
- **特徴の取り出し**：既存の `stage_features` を、そのまま使う（query側をevalモードにして、CLSとqを取り出す）。
- **clipの入力**：
  - clipは `ActivityNetClipDataset`（4フレーム、0.25秒間隔、manifestの時刻を最後のフレームにする）で取り出す。
  - 前処理はfixed probeと同じで、Resize(256) → CenterCrop(224)。いったんuint8で保持し、`/255`（ToTensorと同じ計算）とNormalize(0.5, 0.5)はあとでかける。
  - 4つの並べ方とsame-videoの入力は、256 clipずつ組み立てて、メモリの使用量を抑える。
- **選び方**：1つのgeneratorをseedで固定し、次の順に決める。
  1. 動画（ID順に並べた列を `randperm` で並べ替え、先頭から選んでから並べ直す）
  2. 各動画のclip
  3. same-video（10秒後、なければ10秒前）
  4. other-video（巡回させて、自分以外の動画と組み合わせる）
  5. shuffleの並び（恒等以外の23通りから1つ）
- **集計**：
  - CLSとqのそれぞれで、normalとのcos類似度をfloat64で計算する。
  - 動きの大きさは、0〜1の画像で隣り合うフレームの差の絶対値を平均し、中央値を境に2つに分ける。
  - bootstrapは1,000回（seed 0）。
  - モデル間の差は、C−A、C−B、B−Aを、同じclipどうしで比べる。
- **読み込みの確認（I3）**：各runの `resolved_config.yaml` から、そのrun自身の構造（単一フレームのrunなら単一フレームのbackbone）を組み立てる。同じfixed probeの特徴を計算し、run自身が記録したsha256（`diagnostic_summary.json` の `trajectory_rank0`）と比べる。initモデルは `--init-reference` のstep 0と比べる。
- **出力と実行の条件**：
  - 出力先は新しいディレクトリだけで、すでにあれば止める。
  - `split` がvalidationでないmanifestは受け付けない。
  - GPUで実行する場合、リポジトリがdirtyなら止める。
  - 出力は `summary.json`（集計、条件の定義、選んだ動画とclip、外した数、各モデルのcheckpointのsha256とI3の結果、provenance）と、`per_clip.jsonl`。

## CPUテスト

| # | 内容 | 結果 |
|---|---|---|
| T1 | 並べ方（normal、shuffle、reverse、static、same-video）の入力が正しい。clipが1つしかない動画にはsame-videoがない | PASS |
| T2 | 選び方：<br>・1動画から1 clipを選び、動画は重ならない<br>・seedで再現でき、seedを変えると変わる<br>・same-videoは10秒後を優先し、なければ10秒前、どちらもなければNone<br>・other-videoは自分以外<br>・shuffleは恒等にならない<br>・clip数がmanifestの時刻を持つ動画数を超えると拒否する | PASS |
| T3 | 動きの大きさが手計算と一致する。bootstrapはseedで再現でき、信頼区間が平均を含む。除外したclip（NaN）は数えない。モデル間の差の組み合わせ（C−A、C−B、B−A）が正しい。`--model` の書き方の誤りを拒否する | PASS |
| T4 | 小さなViTのclip run（LoRA、Full FTのstep 3）、単一フレームのFull FT run（step 3）、init（LoRA runのstep 0）で、計算し直したprobeの特徴のsha256が、runの記録と**bitwise一致**した（CLS／z_q／q）。checkpointを読み込まずにstep 3と比べると、一致しないことも確かめた。単一フレームのcheckpointは、clip用のモデルにstrictに読み込める。train_modeが違う場合は拒否する | PASS |
| T5 | 合成のvalidation manifest（5本、そのうち1本は時刻が1つだけ）で、4モデル（A、B-FT、C-LoRA、C-FT）を評価した。<br>・`summary.json` と `per_clip.jsonl` の項目がそろう<br>・I3の結果がすべて一致<br>・same-videoから外した動画が1本<br>・cosの値は[−1, 1]に収まる<br>・既存のディレクトリには書かずに止まる<br>・training splitのmanifestは拒否する | PASS |
| T6 | manifest CLIの既定は `training` のまま、`validation` も受け付け、ほかの値は拒否する。全テストの失敗IDの集合が、変更前と同じ | PASS |

### 全テスト（`.venv/bin/python -m pytest -rf -q test`、GPUが見える状態）

| | 結果 |
|---|---|
| 変更前（`fe58fc3`、新しいテストは除外） | 36 failed / 957 passed |
| 変更後 | 36 failed / 972 passed（+15件は新しいテスト） |
| 失敗IDの集合 | 変更前と**完全に一致**。以前の記録（`9155c2b` の修正後、clip段階のT7）の36件とも一致 |

全テストの最初の実行は、GPUが見えない設定（`CUDA_VISIBLE_DEVICES=''`）だったため、197 failed / 18 errorsになった。GPUを使うパラメータのテストが失敗したためで、変更とは関係ない。以前の基準と同じ条件（GPUが見える状態）で実行し直し、上の比較を行った。

## 実データの作業（それぞれ別に許可が必要）

- **R0**：validation manifestの作成（CPU、NASは読み取りのみ）。**2026-11-04に実行した**（結果は下の「R0の結果」）。
  ```bash
  python -m scripts.vit_moco.activitynet_frames /mnt/NAS-TVS872XT/dataset/ActivityNet --split validation --workers 16 --output-dir log/activitynet_frames/2026-11-04_validation_interval10s_offset5s
  ```
- **R1**（2026-11-04に実行。[R1 Evidence](2026-11-04-temporal-order-sensitivity-evaluation-r1-gpu-evidence.md)）：評価の実行（GPU 1枚、tmux）。研究コードは `50ff715` にcommit済み（dirtyな状態ではGPUの評価が止まる）。
  - モデルは、A=lora:init、B-LoRA、B-FT、C-LoRA、C-FTの5つ。
  - `--clip-config` と `--init-reference` には、clip LoRA A2のrunを使う。

### R0の結果（2026-11-04 05:25:53〜05:30:38、tmux `vit-moco-a1`）

- 研究コード `feature-vit-moco@50ff715`、dirtyではない。16 thread、所要4分45秒、exit 0。
- 出力：研究リポジトリの `log/activitynet_frames/2026-11-04_validation_interval10s_offset5s/`

| 項目 | 値 |
|---|---|
| `split` | validation |
| 動画数 | 4,926本（読めずに止まった動画はない。読めない動画があるとmanifestは書かれない） |
| manifestの時刻（clip数） | 58,163（1本あたり 最小0・最大75・平均11.8） |
| 時刻が1つもない動画 | 18本（使える動画は4,908本） |
| 時刻が2つ以上ある動画 | 4,735本 |
| 長さの取得元 | stream 4,483本、container 443本 |
| 最初の時刻 | 5.0秒（4フレームのclipの先頭は4.25秒になり、0秒より前にはならない） |
| trainingのmanifest（2026-10-21）と重なる動画 | 0本 |
| `frames.jsonl` のsha256 | `c62474962bd21f670edb7ac4a0bc0f283e2c527e5aa45b8af575da0d36595978`（`manifest.json` に記録された値と一致） |
| `manifest.json` のsha256 | `e60715c605c10391f1de86d8e1314aa564d8aa59e2b1c3083781e6cd27afac59` |

- 追加の確認（CPU、動画は読まない）：このmanifestで `select_clips` を2,000 clip・seed 0で呼ぶと、2,000本の動画から1 clipずつ選べた。same-videoのclipがない（時刻が1つだけの）動画は67本。

## 互換性

- 既存の学習経路、診断、比較器、presetは変えていない。
- manifest CLIの既定（`training`）も変えていない。
- 依存パッケージは追加していない。
