---
project: sequential-video-lora-analysis
date: 2026-11-19
record_type: implementation-cpu-verification
status: verified-except-browser
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-lora-svd
base_commit: a549e34
research_commits: [0aa523f]
spec: ../specs/2026-11-19-lora-svd-direction-and-subspace-addendum-spec.md
export_id: 2026-11-19_06-56-19
---

# LoRA の SVD 解析：向きの比べ方と部分空間の比較（追補の実装と CPU での確認）

[承認済みの追補](../specs/2026-11-19-lora-svd-direction-and-subspace-addendum-spec.md)に従って実装し、`feature-lora-svd@0aa523f` に commit した（push はしていない）。

- 登録した全 27 checkpoint を、新しい形式（`lora_svd/2`）で書き出し直した。
- CPU での確認 A-1〜A-4 の結果は下のとおり。
- A-5（ブラウザでの表示）は、ユーザーが手元で確認する。

## 変えたこと

| ファイル | 変更 |
|---|---|
| `evaluation/lora_svd.py` | schema `lora_svd/2`。`svd_summary.csv` の `top16_cos_mean` を、`top16_span_overlap`（φ）と `pa_cos_1…16`（主角の cos）に置き換えた。`merged_overlap.csv` に `w0_index` 列と、i = 17 の行（W₀ の 17〜768 番で最も近いもの）を足した。 |
| `evaluation/lora_svd_viewer.html` | 次の 6 点を入れた。 |
| `test/model/test_lora_svd.py` | T9〜T13 を足し、T4 と列名の確認を新しい形式に合わせた（14 → 16 件）。 |

ビューアの変更：

- **形式の確認**：`lora_svd/2` だけを読む。古い形式は、manifest があってもなくても「書き出し直してください」と出す。
- **全体マップ**：「向きのずれ」を 1 − φ にした。有効ランクの式を説明に書いた。
- **「そのまま」の行列**：今の 16×16 のヒートマップの形はそのままにして、次を足した。
  - 下に「17〜768」の 1 行。
  - 新しい向きの列の ▼ 印と、しきい値のスライダー（既定 0.5）。
  - 主角の cos の棒グラフ。
- **スペクトルの統計**：stable rank と、3 つの式の説明を足した。
- **新しい画面「run どうしの比較」**：A と B を選んで φ(i, j) のヒートマップを出す。全 run の組の表も出し、U と V を切り替えられる。
- **学習の推移**：「特異値」「有効ランク」「最後との向き」を足した。

## 確認の結果

### A-1：テスト

`test/model/test_lora_svd.py` の 16 件がすべて通った（CPU、89 秒）。

### A-2：既存のテスト全体

- `CUDA_VISIBLE_DEVICES=""` で全テストを実行した（50 分）。

  | | 失敗 | 成功 | エラー |
  |---|---|---|---|
  | 2026-11-18（追補の前、`1e0816e`） | 196 | 792 | 18 |
  | 今回（`0aa523f`） | 196 | 794 | 18 |

- 失敗とエラーのテスト ID（214 個）が、前回と完全に一致した。どれも旧フレームワークのテスト。
- 成功が 2 件増えたのは、このファイルのテストが 14 件から 16 件になったため。

### A-3：全 27 checkpoint の書き出し直し

```bash
.venv/bin/python -m evaluation.lora_svd --runs conf/analysis/lora_svd_runs.yaml --threads 4
```

- 書き出しID：`2026-11-19_06-56-19`
- 研究コード：commit `0aa523f`、dirty なし。schema は `lora_svd/2`。
- CPU で 3 分 10 秒、最大メモリ 3.2 GB。全 checkpoint で、元 spec 5.2 の一致確認が通った。
- 書き出しのコードを使わずに、checkpoint と事前学習の重みから計算し直して照合した。

| checkpoint | φ | 主角の cos | 16×16 の \|cos\| | 17〜768 番の行 | `w0_index` |
|---|---|---|---|---|---|
| imagenet-ft step 1000（Full FT） | 4.9e-11 | 1.2e-10 | 5.0e-11（絶対） | 4.2e-10 | 24 行列すべて一致 |
| clip-lora step 459 | 5.0e-11 | 5.1e-11 | 5.0e-11（絶対） | 4.6e-10 | 同上 |
| seq8-lora step 3000（順番読みの途中） | 5.1e-11 | 1.3e-10 | 5.0e-11（絶対） | 4.0e-10 | 同上 |
| seq16-t8-lora step 6255（順番読みの最後） | 4.9e-11 | 4.1e-10 | 5.0e-11（絶対） | 4.3e-10 | 同上 |

（数値は最大の相対誤差。どれも CSV の有効数字 10 桁の範囲 5e-10 以内）

### A-4：ビューア

- 構文チェックが通った。
- DOM を差し替えた描画テストを、A-3 の書き出しで回した。
  - 対象：6 グループの全 step、「差分 / そのまま」と全指標、LoRA / Full FT、推移の 6 つの指標、U / V、しきい値 3 通り。
  - 結果：計 984 通りがすべて例外なく通り、表示に NaN・undefined も出なかった。
  - 必須ファイルだけの場合も、該当する画面に「〜がありません」と出た。
- ビューアで計算した φ を、`svd_vectors.csv` から numpy で計算した値と比べた。
  - φ(i, j)：3 組（U と V、LoRA と Full FT を含む）で、最大の相対誤差は 7.0e-16。
  - 推移の「最後との向き」：seq16-lora の層 5・q の全 7 step で、7.8e-16。
- 古い形式の書き出し（`2026-11-19_04-19-36`）を読ませると、manifest がある場合もない場合も、案内の文が出た。

### A-5：ブラウザでの表示

ユーザーの確認待ち。開き方は前回と同じで、URL の `data=` だけを新しい書き出しに変える。

`http://localhost:8000/evaluation/lora_svd_viewer.html?data=/log/analysis/lora_svd/2026-11-19_06-56-19/`

## 互換性

- 古い書き出し（`lora_svd/1`）は、消さずに残している。新しいビューアでは読めない（追補 3.4 のとおり）。
- 計算に使う checkpoint・設定・登録ファイルは変えていない。
- `lora_svd/1` にもあった値の計算は変えていない。前回の書き出し（`2026-11-19_04-19-36`）と、文字列として完全に一致することを確かめた。
  - 一致したもの：`svd_summary.csv` の共通の列すべて、`w0_spectrum.csv`、`merged_spectrum.csv`、`svd_vectors.csv`、`runs.csv`、`merged_overlap.csv` の 16×16 の部分。

## 主張しないこと

- 「新しい向き」の本数は、仮のしきい値しだいで変わる。
- run どうしの φ が高い理由は、この確認からは分からない。
