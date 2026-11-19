---
project: sequential-video-lora-analysis
date: 2026-11-18
record_type: analysis-export
status: verified-except-browser
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-lora-svd
research_commit: a549e34
spec: ../specs/2026-11-18-lora-svd-export-and-viewer-spec.md
export_id: 2026-11-19_04-19-36
last_updated: 2026-11-19
---

# LoRA の SVD 解析：全 run の書き出し（V-5）

ユーザーの指示（「ほかの結果についても見れるようにして」）で、登録したすべての checkpoint を書き出した。

- **最新の書き出しは `2026-11-19_04-19-36`**（9 run・27 checkpoint、6 グループ）。8 フレームの run を完走後に足した（末尾の「追加（2026-11-19）」）。
- 下の「書き出したもの」と「確認」は、2026-11-18 の書き出し（`2026-11-18_18-25-20`、8 run・20 個）の記録。

- 実装と 1 run での確認は [実装の記録](2026-11-18-lora-svd-export-and-viewer-implementation-cpu-verification.md) にある。

## 書き出したもの

- 書き出しID：`2026-11-18_18-25-20`
- 場所：`log/analysis/lora_svd/2026-11-18_18-25-20/`（101 MB）
- 研究コード：`feature-lora-svd@c46e3b4`（dirty なし）

| グループ | run_id | 方式 | step | 実験記録 |
|---|---|---|---|---|
| imagenet | imagenet-lora / imagenet-ft | LoRA / Full FT | 1000 | `2026-11-01-vit-moco-low-lr-l5-1000step-imagenet-gpu-evidence.md` |
| frame | frame-lora / frame-ft | LoRA / Full FT | 459 | `2026-11-03-vit-moco-lora-activitynet-a2-*-gpu-evidence.md` |
| clip | clip-lora / clip-ft | LoRA / Full FT | 459 | `2026-11-03-vit-moco-lora-activitynet-clip-a2-*-gpu-evidence.md` |
| seq16 | seq16-lora | LoRA | 1000〜6000（1000 ごと）、6255 | `2026-11-04-sequential-loader-clip-moco-lora-g2-gpu-evidence.md` |
| seq8 | seq8-lora | LoRA | 1000〜6000（1000 ごと）、6255 | `2026-11-16-sequential-loader-clip-moco-lora-stride8-gpu-run.md` |

- 合計 8 run・20 checkpoint。
- **spec から増やしたもの**：stride 8 の run（`seq8-lora`）。
  - spec の表にはなかったが、2026-11-17 に完走していたので、登録ファイルに足した（commit `c46e3b4`）。
- **まだ入っていないもの**：8 フレーム・GPU 3 枚の run（`2026-11-17-sequential-loader-clip-moco-lora-8frames-3gpu-gpu-run.md`）。
  - 書き出した時点で学習中だった（checkpoint は step 4,000 まで）。
- **途中の書き出し**：`2026-11-18_18-20-29` は、spec の表どおりの 13 個だけを書き出したもの。
  - 中身は今回の書き出しに含まれている。

コマンド（CPU、4 スレッド。GPU は学習が使っていたので使っていない）：

```bash
.venv/bin/python -m evaluation.lora_svd --runs conf/analysis/lora_svd_runs.yaml --threads 4
```

- 実行時間は 3 分 24 秒、最大メモリは 3.2 GB。
- 全 checkpoint で、5.2 節の一致確認（W₀、`base.weight`、`identity`、step）が通った。1 つでも外れると止まる作りになっている。

## 確認

書き出しのコードを使わずに、checkpoint と transformers の事前学習済みモデルから計算し直して照合した。

- **対象**：7 checkpoint
  - 今回の書き出しで照合したもの：imagenet-ft、seq8 の step 1000 と 6255
  - 途中の書き出し（`2026-11-18_18-20-29`）で照合したもの：imagenet-ft、frame-lora、clip-ft、seq16 の step 2000 と 6255
  - imagenet-ft は両方で照合した。同じコードで同じ checkpoint から計算している。
- **結果**：
  - 特異値、フロベニウスノルム、1 つ前の checkpoint との距離、ヘッドごとのノルム、「そのまま」の特異値は、すべて相対 5e-10 以内で一致した。これは CSV の有効数字 10 桁の範囲。
  - 特異ベクトルは、ΔW v = σ u の関係を満たしていた。
  - LoRA の `base.weight` は、事前学習済みの重みと bitwise に一致した。
- **ビューア**：
  - DOM を差し替えた描画テストで、5 グループ・計 2,040 通りの描画が、すべて例外なく通った。
  - ローカルのサーバー経由でも、7 ファイルを読み込めた。
- **ブラウザでの表示（V-4）**：ユーザーの手元での確認待ち。

## ビューアの開き方

1. 研究リポジトリの直下で、サーバーを立てる。

   ```bash
   cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora && .venv/bin/python -m http.server 8000 --bind 127.0.0.1
   ```

2. 手元の PC で、ポートを転送する。

   ```bash
   ssh -L 8000:localhost:8000 150-home
   ```

3. ブラウザで次の URL を開く。

   `http://localhost:8000/evaluation/lora_svd_viewer.html?data=/log/analysis/lora_svd/2026-11-19_04-19-36/`

## 追加（2026-11-19）：8 フレームの run

ユーザーの指示（「完走後に追加」）で、8 フレーム・GPU 3 枚の run が完走するのを待ってから、登録に足した。

- **学習の完走**：`run_report.json` の書き込みは 2026-11-19 04:16:07。
  - `run_report.json`：`status: completed`、`last_global_step` 6255、failures なし。
  - `diagnostic_summary.json`：`engineering_status: PASS`、reasons なし。
- **登録**：

  | run_id | グループ | 学習 | step |
  |---|---|---|---|
  | seq16-t8-lora | seq16-t8 | 順に読む・8 フレーム clip・stride 16 | 1000〜6000（1000 ごと）、6255 |

  - run：`log/moco/g2_t8_3gpu/lora/2026-11-17_15-55-31-413149/`
  - 記録：`2026-11-17-sequential-loader-clip-moco-lora-8frames-3gpu-gpu-run.md`
  - Comet の URL はまだないので空にしている。
  - commit `2002345`
- **書き出し**：
  - 書き出しID：`2026-11-19_04-19-36`（9 run・27 checkpoint、137 MB）
  - 研究コード：commit `2002345`、dirty なし
  - CPU、4 スレッドで 3 分 37 秒。最大メモリは 3.2 GB。
  - 全 checkpoint で一致確認が通った。
- **照合**：書き出しのコードを使わずに計算し直した。
  - 対象：seq16-t8 の step 1000 と 6255
  - すべての値が相対 5e-10 以内（CSV の有効数字 10 桁の範囲）で一致した。
  - `base.weight` は、事前学習済みの重みと bitwise に一致した。
- **ビューアの不具合と修正**：
  - 不具合：この書き出しを読むと、「Maximum call stack size exceeded」で止まった。
  - 原因：最小・最大を `Math.max(...配列)` で求めていた。`merged_overlap.csv` が 165,888 行あり、配列を引数として展開すると、JavaScript の引数の上限を超える。
  - 修正：12 か所すべてをループで求める形に変えた（commit `a549e34`）。
  - 修正後の確認：描画テストを、テスト用の小さなデータと 3 つの書き出し（1 run、20 個、27 個）で回し、すべて例外なく通った。27 個では 6 グループ・2,880 通り。
  - サーバー経由での読み込みも通った。
  - 書き出しのデータ（CSV）は、この修正の影響を受けない。

## 主張しないこと

- この記録は、「全 run を正しく計算して表示できる」ことの確認だけ。
- 特異値の大きさから、LoRA が静的な情報と動的な情報のどちらを持つかは判断していない。
