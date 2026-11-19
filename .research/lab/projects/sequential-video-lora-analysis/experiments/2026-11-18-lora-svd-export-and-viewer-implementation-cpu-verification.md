---
project: sequential-video-lora-analysis
date: 2026-11-18
record_type: implementation-cpu-verification
status: verified-except-browser
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-lora-svd
base_commit: 59bade2
research_commits: [1e0816e]
spec: ../specs/2026-11-18-lora-svd-export-and-viewer-spec.md
export_id: 2026-11-18_06-56-48
---

# LoRA の SVD 解析：CSV の書き出しとビューアの実装、CPU での確認

[承認済み spec](../specs/2026-11-18-lora-svd-export-and-viewer-spec.md) に従って実装し、`feature-lora-svd@1e0816e` に commit した（push はしていない）。

- CPU での確認 V-1〜V-3 は満たした。
- V-4（ブラウザでの表示）は、ユーザーが手元で確認する。
- V-5（14 個すべての書き出し）は、まだ行っていない。別に許可を取る。

## 追加したファイル（研究コード）

| ファイル | 役割 |
|---|---|
| `evaluation/lora_svd.py` | checkpoint を読み、ΔW の SVD などを計算して CSV 6 個と `manifest.json` を書く（`python -m evaluation.lora_svd`） |
| `conf/analysis/lora_svd_runs.yaml` | 書き出す 7 run・14 checkpoint の登録 |
| `evaluation/lora_svd_viewer.html` | CSV を読んで表示するビューア（外部ライブラリなし） |
| `test/model/test_lora_svd.py` | CPU のテスト 14 件 |

## spec から変えた点

- **テストの置き場所**：
  - spec では `test/evaluation/test_lora_svd.py` としていた。
  - 実際は `test/model/test_lora_svd.py` に置いた。
  - 理由：`test/evaluation/` にはテストがなく、評価コードのテスト（`test_temporal_order_evaluation.py`）も `test/model/` にあるので、それに合わせた。
- **引数 `--output-dir` を追加した**：
  - 書き出し先を指定できる。省略時は spec どおり `log/analysis/lora_svd/<日時>/`。
  - テストで一時フォルダへ書くために使う。
- ほかの引数（`--runs`、`--only`、`--threads`、`--skip-optional`）は spec どおり。

## 確認の結果

### V-1：テスト

- `test/model/test_lora_svd.py` の 14 件がすべて通った（CPU、106 秒）。
- 内容は spec の T1〜T8 に対応する。小さな ViT で作った LoRA（rank 4 と 2）と Full FT の run を使う。

### V-2：既存のテスト全体

- `CUDA_VISIBLE_DEVICES=""` で全テストを実行した（1 時間 4 分）。

  | | 失敗 | 成功 | エラー |
  |---|---|---|---|
  | 2026-11-11 の記録（リファクタリング後） | 196 | 778 | 18 |
  | 今回 | 196 | 792 | 18 |

- 成功が増えた 14 件は、今回追加したテスト。
- 失敗とエラーは、どちらも旧フレームワークのテストだけ。
  - 失敗：image models、model factory、video models、checkpoint、video folder
  - エラー：optimizer、scheduler
- 既存のファイルは変更していない（追加だけ）。

### V-3：1 run の書き出し（`imagenet-lora`）

コマンド（CPU、4 スレッド）：

```bash
.venv/bin/python -m evaluation.lora_svd --runs conf/analysis/lora_svd_runs.yaml --only imagenet-lora --threads 4
```

- 出力：`log/analysis/lora_svd/2026-11-18_06-56-48/`（commit `1e0816e`、dirty なし）
- 実行時間は 30 秒、最大メモリは 2.7 GB。
- 行数：
  - `svd_summary.csv`：48 行（事前学習の重みと LoRA、それぞれ 12 層 × q・v）
  - `svd_vectors.csv`：384 行
  - `w0_spectrum.csv`、`merged_spectrum.csv`：各 18,432 行
  - `merged_overlap.csv`：6,144 行

書き出しのコードを使わずに、checkpoint と transformers の事前学習済みモデルから計算し直して照合した（スクリプトは会話の作業用で、リポジトリには入れていない）。

- **`base.weight` と W₀**：
  - checkpoint の 24 個の `base.weight` は、transformers から読み込んだ `google/vit-base-patch16-224` の q・v の重みと bitwise に一致した。
- **特異値（`sigma`）**：
  - 書き出し側で丸める前の値は、numpy で別に計算した値と最大で相対 1.2e-15 の差だった。spec の基準（相対 1e-10）を満たす。
  - CSV に書かれた値との差は最大で相対 4.6e-10。`%.10g`（有効数字 10 桁）で丸めたことによる差で、上限は 5e-10。
- **ほかの値**：CSV の値と計算し直した値の最大差は、どれも `%.10g` の丸めの範囲に収まった。
  - フロベニウスノルム：相対 2.4e-10
  - ヘッドごとのノルム：相対 4.3e-10
  - W₀ の特異値：相対 5.0e-10
  - 「そのまま」の特異値：相対 4.9e-10
  - 向きの重なり（cos）：絶対 5.0e-11
  - top16_cos_mean：絶対 4.1e-11
  - 特異ベクトル（ΔW v = σ u）：5.6e-11
- 符号の決め方（u の絶対値が最大の成分を正にする）も守られていた。
- 最初の checkpoint なので、`fro_step_diff` は `fro` と同じ値になっている。
- **`manifest.json`**：spec 5.5 節の項目がすべてそろっていた（schema、作成日時、commit、環境、登録ファイルの sha256、事前学習の次元、残す個数、スレッド数、checkpoint ごとの sha256 と identity、ファイル一覧）。

### V-4 の準備：ビューア

Claude が行った確認：

- スクリプトの構文チェックが通った（`node --check`）。
- DOM を差し替えた描画テストを V-3 の出力で回し、120 通りの描画がすべて例外なく通った。対象は 12 層 × q・v、2 つの比較（差分・そのまま）と各指標。
- ローカルのサーバー経由（`?data=`）で、V-3 の出力の 7 ファイルを読み込めた。

ブラウザでの表示は、この環境ではできない。ユーザーの手元で次の手順で確認する。

1. 研究リポジトリの直下で、サーバーを立てる。

   ```bash
   cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora && .venv/bin/python -m http.server 8000 --bind 127.0.0.1
   ```

2. 手元の PC から、ポートを転送する。

   ```bash
   ssh -L 8000:localhost:8000 <サーバー>
   ```

3. ブラウザで次の URL を開く。

   `http://localhost:8000/evaluation/lora_svd_viewer.html?data=/log/analysis/lora_svd/2026-11-18_06-56-48/`

- ファイル選択でも確認する場合：HTML を手元にコピーして開き、出力フォルダの 7 ファイルを選ぶ。
- この書き出しは `imagenet-lora` だけなので、次の 2 点が出るのは想定どおり。
  - グループは ImageNet のみ。
  - Full FT 側は「なし」。

## 主張しないこと

- この記録は、「正しく計算でき、表示できる」ことの確認だけ。
- 特異値の大きさから、LoRA が静的な情報と動的な情報のどちらを持つかは判断していない。
