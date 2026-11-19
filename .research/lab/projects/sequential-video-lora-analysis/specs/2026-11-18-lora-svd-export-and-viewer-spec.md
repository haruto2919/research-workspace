---
project: sequential-video-lora-analysis
date: 2026-11-18
spec_type: implementation-analysis-tool
status: approved
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-lora-svd（feature-vit-moco から切る）
base_commit: 59bade2
source_meeting: ../meetings/2026-11-12-mtg.md
design_reference: https://claude.ai/artifact/JZrtPHYNqRZe5XSbVcZMsx （画面のデモ。数値は仮データ）
---

# LoRA の特異値分解：CSV 書き出しと HTML ビューアの spec

**Status: approved（2026-11-18 ユーザー承認。実装中）**

## 1. 目的

2026-11-12 の MTG の決定事項「今の学習済みモデルの LoRA を特異値分解し、事前学習の重みとの比較・初期値からの距離で現状を把握する」「結果を実験ごとに管理し、層・モジュールを選んで表示できる解析アプリを作る」を実現する。

次の2つを作る。

1. **書き出しスクリプト（Python）**：学習済みの checkpoint から、特異値分解の結果を決まった形の CSV に書き出す。計算はすべてここで行う。
2. **ビューア（HTML）**：書き出した CSV を読み、表示するだけ。計算はしない。

同じ CSV からは、いつでも同じ表示になる。どの checkpoint・どのコードから作った数値かを、書き出しフォルダの記録からたどれる。

## 2. 前提（コード調査で確認済み、2026-11-18）

基準：研究コード `feature-vit-moco@59bade2`（clean）。

- **LoRA の実装**：`model/vit/lora_linear.py` の `LoRALinear`。
  - 出力 = `base(x) + scaling · lora_b(lora_a(x))`、`scaling = alpha / rank`。
  - `lora_a` は `kaiming_uniform`、`lora_b` は 0 で初期化する。
  - 凍結した元の重みは `base.weight` として checkpoint に入っている。
- **LoRA を付ける場所**：`model/vit/vit_lora_frame_encoder.py`。
  - ViT（`ViTModel`）の `layers[i].attention` の `q_proj`・`v_proj`（768×768）を `LoRALinear` に置き換える。
  - 今の設定（`conf/moco/model/lora.yaml`）は target `[q, v]`、rank 8、alpha 8、dropout 0。
- **checkpoint**：`training/checkpoint.py` が `checkpoint_stepNNNNNN.pt` に次を保存する。
  - `model`（state_dict 全体。query 側 `backbone_q` と key 側を含む）、`optimizer`、`identity`、`identity_sha256`、`global_step`。
  - `identity["model"]["lora"]` に rank・alpha が入る。
- **使える既存の関数**：`evaluation/temporal_order.py` と `training/` に、次の部品がそろっている。
  - `load_run_config`：run の `resolved_config.yaml` を、2026-11-11 より前の形も含めて読む。
  - `build_model`：設定からモデルを作る。seed を固定するので、学習前（step 0）と同じ初期値になる。
  - checkpoint を strict に読み込む処理（`load_query`）。
  - ファイルの sha256（`file_sha256`）。
- **出力先**：`.gitignore` に `log/` が入っているので、書き出した CSV はリポジトリに入らない。
- **ブランチ**：`dev` は ResNet の MoCo 段階（`7fc1f56`）に戻してあり、ViT＋MoCo のコードは `feature-vit-moco` にしかない。そのため、`feature-vit-moco` から切る。
- **解析する checkpoint**：次の7 run・14 個。ファイルと `resolved_config.yaml` の存在を確認した。
  - 順に読む学習は、step 1,000〜6,255 をまとめた `combined_seq_clip_lora_step1000-6255/` にあり、中身は元の run へのシンボリックリンク。そのため、`resolved_config.yaml` はリンク先の run から読む。

| run_id（案） | 学習 | 方式 | step | run のフォルダ（`log/` 以下） | 実験記録 |
|---|---|---|---|---|---|
| imagenet-lora | ImageNet・画像、学習率 3e-4 | LoRA | 1000 | `moco_vit_lr_long/lora/imagenet/lr_0.0003/2026-11-01_00-11-06-779221` | `2026-11-01-vit-moco-low-lr-l5-1000step-imagenet-gpu-evidence.md` |
| imagenet-ft | 同上 | Full FT | 1000 | `moco_vit_lr_long/full_ft/imagenet/lr_0.0003/2026-11-01_09-37-00-940051` | 同上 |
| frame-lora | ActivityNet・単一フレーム | LoRA | 459 | `moco_vit_lr_activitynet/lora/activitynet_frames/lr_0.0003/2026-11-03_10-48-36-408110` | `2026-11-03-vit-moco-lora-activitynet-a2-lora-gpu-evidence.md` |
| frame-ft | 同上 | Full FT | 459 | `moco_vit_lr_activitynet/full_ft/activitynet_frames/lr_0.0003/2026-11-03_11-49-57-818631` | `2026-11-03-vit-moco-lora-activitynet-a2-full-ft-gpu-evidence.md` |
| clip-lora | ActivityNet・4フレームclip（シャッフル） | LoRA | 459 | `moco_vit_lr_activitynet_clip/lora/activitynet_clips/lr_0.0003/2026-11-03_15-22-36-985580` | `2026-11-03-vit-moco-lora-activitynet-clip-a2-lora-gpu-evidence.md` |
| clip-ft | 同上 | Full FT | 459 | `moco_vit_lr_activitynet_clip/full_ft/activitynet_clips/lr_0.0003/2026-11-03_16-46-46-287044` | `2026-11-03-vit-moco-lora-activitynet-clip-a2-full-ft-gpu-evidence.md` |
| seq16-lora | 動画を先頭から順に読む・stride 16 | LoRA | 1000, 2000, 3000, 4000, 5000, 6000, 6255 | `moco_vit_seq_activitynet_clip/lora/activitynet_sequential_clips/lr_0.0003/combined_seq_clip_lora_step1000-6255` | `2026-11-04-sequential-loader-clip-moco-lora-g2-gpu-evidence.md` |

## 3. Decision Contract（2026-11-16〜18 にユーザーが採用）

### 採用

- **構成**：書き出しスクリプト（Python）と、CSV を読むだけのビューア（HTML）に分ける。
- **範囲**：差分 ΔW の解析（必須の3ファイル）に加え、「そのまま W₀+ΔW」の解析と特異ベクトル（任意の3ファイル）も含める。
  - 実装は、必須 → 任意の順に段階的に進める。
- **SVD をかける行列**：次の2通りを両方出す。
  - **差分**：ΔW（LoRA は `scaling·B·A`、Full FT は `W_FT − W₀`）
  - **そのまま**：W₀+ΔW。W₀ の分解と比べる。
- **LoRA の rank は実験ごとに変わりうる**：rank は各 run の設定から読む。CSV とビューアは、rank が run ごとに違っても扱える。
- **Full FT の ΔW の特異値**：上位 32 個を残す。
- **特異ベクトル**：残す本数は「その学習の LoRA の rank」本。Full FT は、同じグループの LoRA の rank 本。
- **向きのずれ**：「最も重なる組」で定義する（5.3 節）。
- **ビューアの読み込み**：次の2つに対応する。
  - ローカルサーバー（`python -m http.server`）で、書き出しフォルダを自動で読む。
  - HTML を直接開いたときは、ファイル選択で読む。
- **元の実験へのリンク**：runs.csv に、実験記録のファイル名と comet の URL を入れる。ビューアに表示する。

### 対象外

- **学習前の A₀ と、A・B を別々に見る解析**
  - `build_model` で A₀ を作り直せる見込みはあるが、今回は扱わない。
- **k・MLP の重み、MoCo の key 側の解析**
- **動画を入力して反応を見る解析**：LoRA あり／なしで反応する場所のヒートマップなど。
- **CSV を埋め込んだ共有用の HTML 1ファイル**
- **複数の書き出しフォルダをまとめる処理（collect）**
  - 解析する run が増えたら、登録ファイルに足して全部を書き出し直す（5.6 節）。
- **外部ライブラリ（CDN）の読み込み**
- **MTG 資料用の図の作成、Comet への送信**
- **学習コード・設定・既存の checkpoint の変更**

### 棄却した案

- **ビューアの中で特異値分解を計算する**：数値の正しさを確かめにくくなるため。
- **ビューアの仮データ表示を残す**：実データと取り違えるおそれがあるため、仮データの生成部分は入れない。

## 4. ファイル構成（追加するファイル）

```text
evaluation/lora_svd.py              # 計算と書き出し（python -m evaluation.lora_svd）
evaluation/lora_svd_viewer.html     # ビューア（外部ライブラリなし）
conf/analysis/lora_svd_runs.yaml    # 解析する run の登録（人が書く）
test/evaluation/test_lora_svd.py    # CPU のテスト
```

既存のファイルは変更しない。

## 5. 実装要件

### 5.1 run の登録ファイル（`conf/analysis/lora_svd_runs.yaml`）

人が書くのは、run ごとに次の項目だけとする。

```yaml
runs:
  - run_id: seq16-lora
    group: seq16
    label: 動画を先頭から順に読む・stride 16（6,255step）
    checkpoints:
      - log/.../combined_seq_clip_lora_step1000-6255/checkpoint_step001000.pt
      # ...
    record: 2026-11-04-sequential-loader-clip-moco-lora-g2-gpu-evidence.md
    comet_url: https://www.comet.com/...
```

- 方式（LoRA／Full FT）・rank・alpha・データ・clip のフレーム数・stride・読み方は、手で書かない。
  - checkpoint の実体（シンボリックリンクは `resolve()` する）と同じフォルダの `resolved_config.yaml` から、`load_run_config` で読む。
  - checkpoint の `identity["model"]` とも照合し、食い違えばエラーにする。
- 最初の登録内容は、2 節の表の7 run・14 個とする。
- 同じ `group` には、LoRA の run が1つだけ入る（Full FT は0〜1つ）。

### 5.2 計算（checkpoint ごと）

- **CPU のみで計算する**：`torch.set_num_threads` は既定 4 にし、引数で変えられるようにする。GPU で動いている学習の、CPU を使う読み込みを遅くしないため。
- **読み込み**：`torch.load(map_location="cpu", mmap=True, weights_only=False)` で読み、`model` を `build_model(cfg)` に strict で読み込む。使うのは `backbone_q` だけ。
- **W₀**
  - 各 run の設定で `build_model(cfg)` を作り、学習前の `q_proj`・`v_proj` の重みを W₀ とする（`checkpoint_id` は全 run で `google/vit-base-patch16-224`）。
  - 全 run の W₀ が bitwise に一致することを確認し、違えばエラーにする。
  - LoRA の checkpoint では、`base.weight` が W₀ と bitwise に一致することを確認する。
- **層とモジュール**：`layer` は `vit.layers` の番号に1を足した1〜12。`module` は `q`（`q_proj`）と `v`（`v_proj`）。
- **ΔW**（float64）
  - LoRA：`scaling × lora_b.weight @ lora_a.weight`（出力×入力 = 768×768）
  - Full FT：`W_FT − W₀`
- **SVD**：`torch.linalg.svd(ΔW, full_matrices=False)`（float64）。
- **CSV に書く値**
  - `sigma_1 … sigma_k`：LoRA は k = rank、Full FT は k = 32。
  - `fro` = ‖ΔW‖_F
  - `head_norm_h` = ‖ΔW の 64(h−1)+1〜64h 行目‖_F（h = 1〜12。行 = 出力の次元）
  - `fro_step_diff` = ‖ΔW_t − ΔW_前‖_F
    - 「前」は、同じ run で登録した直前の checkpoint。
    - 最初の checkpoint では `fro` と同じ値にする（学習前の ΔW は、LoRA・Full FT とも 0 のため）。
- **W₀ の値**（1回だけ）
  - 特異値 768 個、`fro`、`head_norm_1…12`。
  - `svd_summary.csv` の `run_id = pretrained`・`step = 0` の行と、`w0_spectrum.csv` に書く。
- **そのまま**
  - `torch.linalg.svdvals(W₀ + ΔW)` の 768 個 → `merged_spectrum.csv`
  - 左特異ベクトルの上位 16 本どうしの |cos| の表（5.3 節）→ `merged_overlap.csv`
- **特異ベクトル**
  - ΔW の左右の特異ベクトルの上位 r 本（r は 3 節の本数）→ `svd_vectors.csv`
  - **符号の決め方**：各成分 c で、u_c の絶対値が最大の要素（同じ値なら番号が小さい方）が正になるように、u_c と v_c の符号をそろえて反転する。step ごとに色が反転して見えないようにするため。

### 5.3 向きのずれ（「最も重なる組」）

- U₀ を W₀ の上位 16 本の左特異ベクトル、U′ を W₀+ΔW の上位 16 本の左特異ベクトルとし、M = |U₀ᵀ U′|（16×16）とする。
- `merged_overlap.csv` には、M の全 256 要素を書く。
- `top16_cos_mean` = (1/16) Σᵢ maxⱼ M[i, j] を `svd_summary.csv` に書く。
  - ビューアは「向きのずれ」= 1 − `top16_cos_mean` を表示する。

### 5.4 CSV の形式（schema `lora_svd/1`）

| ファイル | 必須 | 1行の単位 | 列 |
|---|---|---|---|
| `runs.csv` | 必須 | 1 run | `run_id, group, label, mode, rank, alpha, data_source, clip_frames, frame_stride, order, record, comet_url, run_dir` |
| `svd_summary.csv` | 必須 | run × step × 層 × q/v（＋pretrained） | `run_id, step, layer, module, sigma_1 … sigma_K, fro, fro_step_diff, head_norm_1 … head_norm_12, top16_cos_mean` |
| `w0_spectrum.csv` | 必須 | 層 × q/v × 番号 | `layer, module, idx, sigma` |
| `merged_spectrum.csv` | 任意 | run × step × 層 × q/v × 番号 | `run_id, step, layer, module, idx, sigma` |
| `svd_vectors.csv` | 任意 | 1本のベクトル | `run_id, step, layer, module, kind, component, x_1 … x_768`（`kind` は `U` か `V`） |
| `merged_overlap.csv` | 任意 | 1要素 | `run_id, step, layer, module, i, j, cos` |

- **`mode`**：`lora` か `full_ft`（設定の `train_mode` のまま）。
- **Full FT の `rank`・`alpha`**：空欄にする。
- **K**：全 run の最大（今は 32）。k 個より後ろの `sigma` は空欄にする。
- **設定にない値**：clip でない run の `clip_frames`、sequential でない run の `frame_stride` は空欄にする。
- **数値の書式**：`%.10g` で書く。

### 5.5 書き出しフォルダの記録

- **出力先**：`log/analysis/lora_svd/<書き出しID>/`
  - 書き出しID は、`YYYY-MM-DD_HH-MM-SS` 形式の開始時刻とする。
  - 既存のフォルダには上書きしない。
- **`manifest.json` に記録するもの**
  - `schema`（`lora_svd/1`）、作成時刻
  - 研究コードの commit（`git rev-parse HEAD`）
  - 登録ファイルの sha256
  - checkpoint ごとに：実体のパス、sha256、`global_step`、`identity_sha256`
  - W₀ の出どころ（`checkpoint_id`）
  - 書き出したファイルの一覧
- **clean でないと実行しない**：研究コードに未 commit の変更があるときは実行を止める（`evaluation/temporal_order.py` と同じ）。
- **ファイル名と step の確認**：checkpoint の `global_step` が、ファイル名の step と違えばエラーにする。

### 5.6 コマンド

```bash
python -m evaluation.lora_svd --runs conf/analysis/lora_svd_runs.yaml [--only RUN_ID ...] [--threads 4] [--skip-optional]
```

- `--only`：テストや確認用に、一部の run だけを書き出す。
- `--skip-optional`：任意の3ファイルを書かない。
- 解析する run が増えたら、登録ファイルに足して、全部を新しいフォルダへ書き出し直す。

### 5.7 ビューア（`evaluation/lora_svd_viewer.html`）

- **元にするもの**：デザイン参照のデモ（第2版）の画面構成・操作を引き継ぐ。
  - 画面：全体マップ、最大特異値の比較、特異値スペクトル、特異値分解の行列、学習の推移、ヘッドごとのマップ。
  - 操作：「差分／そのまま」の切り替え、rank の切り替え。
  - **仮データの生成部分は入れない。**
- **読み込み**
  - **サーバー経由**：`?data=<フォルダ>` を付けて開くと、その中の `manifest.json` と、そこに書かれた CSV を読む。
  - **直接開いたとき**：`?data` がないとき、または読めないときは、ファイル選択（複数選択）で読む。
  - **形式の確認**：ファイルの種類は列名で判断する。必須の列が足りなければ、ファイル名と足りない列名を画面に出す。
  - **任意のファイルがないとき**：該当する画面に「〜がありません」と出し、ほかの画面は表示する。
- **記録の表示**：`manifest.json` を読んだときは、書き出しID と研究コードの commit を画面上部に出す。
- **外部ライブラリなし**：CDN を読まない。オフラインで動く。
- **使い方の例**（研究室のサーバーで書き出し、手元の PC で見る場合）

```bash
cd <研究コード>; python -m http.server 8000
ssh -L 8000:localhost:8000 <サーバー>   # 手元の PC で
# ブラウザで http://localhost:8000/evaluation/lora_svd_viewer.html?data=/log/analysis/lora_svd/<書き出しID>/
```

### 5.8 テスト（`test/evaluation/test_lora_svd.py`、CPU、外部データなし）

| ID | 確認すること |
|---|---|
| T1 | 小さな A・B・scaling から作った ΔW の特異値が、`numpy.linalg.svd` の値と一致する（相対 1e-10）。個数は rank。 |
| T2 | `head_norm` の2乗の和が `fro`² に一致する。 |
| T3 | 符号をそろえた後も U Σ Vᵀ が ΔW に戻る。同じ入力なら、何度計算しても符号が同じ。 |
| T4 | ΔW = 0 のとき、そのままの特異値が W₀ の特異値と一致し、`top16_cos_mean` = 1。 |
| T5 | 最初の checkpoint の `fro_step_diff` が `fro` と一致し、2つ目以降が ‖ΔW_t − ΔW_前‖_F と一致する。 |
| T6 | 書き出した CSV の列名が 5.4 節と完全に一致し、読み戻した値が元の値と一致する（`%.10g` の範囲）。 |
| T7 | rank が違う run（例：4 と 16）を同じ書き出しに入れても、K と空欄が 5.4 節どおりになる。 |
| T8 | Full FT の ΔW が `W_FT − W₀` になり、特異値が 32 個になる。 |

## 6. 手順

1. `feature-vit-moco@59bade2` から `feature-lora-svd` を切る。
2. **必須の3ファイル**（差分）の計算・書き出しと、T1〜T3・T5〜T8 を実装する。
3. **任意の3ファイル**（そのまま・特異ベクトル）と、T4 を実装する。
4. ビューアを実装する（必須だけ／任意ありの両方で表示できること）。
5. CPU の確認（8 節の V-1〜V-4）を行い、commit する（push はしない）。
6. 別に許可を得て、全14個の書き出し（V-5）を行い、実験記録を書く。

## 7. 互換性と既存の実験への影響

- 既存のコード・設定・テスト・checkpoint は変更しない。追加するのは 4 節の4ファイルだけ。
- checkpoint は読むだけ。`log/analysis/` に新しいフォルダを作る以外は、何も書かない。
- 学習中の run（stride 8、GPU 0・1）には触れない。CPU のスレッド数を絞る（5.2 節）。

## 8. 成功条件

### 実装時（CPU のみ。この spec の承認の範囲で実行してよい）

- **V-1**：T1〜T8 がすべて通る。
- **V-2**：既存のテスト全体を実行し、失敗するテストの集合が変更前と同じ。
- **V-3**：1 run（`imagenet-lora`、checkpoint 1個）だけを `--only` で書き出し（数分以内）、次を満たす。
  - `base.weight` と W₀ が bitwise に一致する。
  - `sigma` が、同じ B・A から別に計算した `svdvals` と一致する（相対 1e-10）。
  - `manifest.json` の項目がそろっている。
- **V-4**：V-3 の出力をビューアで開いたとき（サーバー経由とファイル選択の両方）、全画面がエラーなく表示される。
  - この環境にはブラウザがないので、V-4 はユーザーが手元で確認する。
  - Claude は、ビューアのスクリプトの構文チェックと、DOM を差し替えた描画テストまでを行う。

### 全体の書き出し（実装の後、別に許可を取る）

- **V-5**：2 節の14個を書き出す（CPU、10〜20 分の見込み）。次を満たす。
  - 全 checkpoint で、5.2 節の一致確認（W₀、`base.weight`、`identity`、step）が通る。
  - 6つの CSV と `manifest.json` がそろう。
- 結果は `experiments/` に実験記録として残す。記録には、書き出しID、commit、対象の checkpoint、ビューアの開き方を書く。

### 主張しないこと

- 特異値やノルムの大きさからは、LoRA が静的・動的のどちらの情報を持つかは言えない。この spec の成功は「正しく計算して表示できる」ことだけを意味する。
- 「0.1% 程度なら学習していないのと同じ」（MTG での例え）を判定の基準として使わない。

## 9. 実行の境界

- **この spec の承認で行ってよいこと**
  - 実装
  - CPU のテスト
  - V-3 の小さな書き出し
- **別に許可を取ること**
  - V-5 の全体の書き出し
  - push、`dev` への反映
- **GPU は使わない。**

## 10. 未決事項

### blocking

なし（2026-11-18 に、範囲・Full FT の特異値の数・向きのずれの定義・ビューアの読み込み方をユーザーが決定）。

### non-blocking（実装時に、既存の style に合わせて決める）

- CLI の細かい引数名、ログの出し方。
- ビューアの見た目の細部（デモの第2版に合わせる）。
- `run_id`・`label` の表記（2 節の表の案をもとに、登録ファイルで直せる）。

## 11. 承認記録

- 2026-11-18：draft 作成。
- 2026-11-18：ユーザーが承認し、実装を指示（「承認して実装に進めて」）。9 節の「この spec の承認で行ってよいこと」まで。
- 2026-11-18：実装し、`feature-lora-svd@1e0816e` に commit した（push はしていない）。
  - V-1〜V-3 は満たした。V-4（ブラウザでの表示）はユーザーの確認待ち。V-5 は未実施。
  - 詳細は [実装と CPU での確認](../experiments/2026-11-18-lora-svd-export-and-viewer-implementation-cpu-verification.md)。
  - この spec から変えた点：
    - テストは `test/model/test_lora_svd.py` に置いた（既存の評価コードのテストの置き場所に合わせた）。
    - 書き出し先を指定する引数 `--output-dir` を加えた（テスト用。省略時は 5.6 節どおり）。
  - V-3 の「相対 1e-10」は、丸める前の計算値で満たした（1.2e-15）。CSV の値は `%.10g` の丸めの分だけずれる（最大 4.6e-10）。
  - status は、V-4 の確認が済むまで `approved` のままにする。
- 2026-11-18：ユーザーの指示（「ほかの結果についても見れるようにして」）で、V-5 を行った。
  - 2 節の「14 個」は数え間違いで、表にあるのは 13 個（6 run × 1 個と、順番読みの 7 個）。
  - 表の 13 個に加え、完走していた stride 8 の run（7 個）を登録ファイルに足した（`feature-lora-svd@c46e3b4`）。
  - 書き出しID は `2026-11-18_18-25-20`（8 run・20 個）。全 checkpoint で一致確認が通った。
  - 詳細は [全 run の書き出し](../experiments/2026-11-18-lora-svd-full-export.md)。
  - 残りは V-4（ブラウザでの表示）だけ。
- 2026-11-19：ユーザーの指示（「完走後に追加」）で、完走した 8 フレームの run（7 個）を登録に足した（`2002345`）。
  - 書き出しID は `2026-11-19_04-19-36`（9 run・27 個）。
  - 行数が増えて、ビューアの最小・最大の求め方が JavaScript の引数の上限を超えたので、ループに直した（`a549e34`）。
  - 残りは V-4 だけ。
- 2026-11-19：追補 [向きの比べ方と部分空間の比較](2026-11-19-lora-svd-direction-and-subspace-addendum-spec.md) を draft で作成。承認されると、5.3 節と 5.4 節の一部（`top16_cos_mean`、`merged_overlap.csv`、schema）が置き換わる。
