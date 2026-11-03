---
project: sequential-video-lora-analysis
date: 2026-11-03
record_type: loader-verification
status: pass-on-sample
loader_repository: tamaki-lab/2026_09_ishikawa_sequential_loader
loader_branch: ActivityNet
loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
research_commit: fe58fc37ef28901184980ccd5f9e90e40c3d836d
brainstorm: ../../../../secretary/notes/brainstorm/2026-11-03-activitynet-sequential-loader-verification.md
---

# ActivityNet用sequential-loaderの動作確認（training 259本、A1〜A8・B1）

[壁打ちで決めた確認項目](../../../../secretary/notes/brainstorm/2026-11-03-activitynet-sequential-loader-verification.md)に従って確認した。対象はActivityNet trainingの259本（無作為200本＋問題が出やすい59本）で、sequential-loader（`19a0ed7`、clean）で**先頭からEOFまで**読んだ。**A1〜A8はすべて問題なし**だった。

- 確認したこと：エラーはなかった。chunkの並び・フレーム数・`decode_frame` との画像一致（2,483か所）・再現性・資源の解放も、すべて正しかった。
- 元ファイルの性質：timestampの重複と飛びが、40本で動画の**最後の1〜2フレームだけ**に見つかった。ffprobeでも同じ値が出るので、loaderの誤りではなく動画ファイル側の性質である。
- 読み込み速度：1プロセスで平均45.5フレーム/秒（1080pでは22）だった。**strict設定（1プロセス）では、training 1周に約190時間かかる**見積もりになる。
- 変更：sequential-loaderにも研究コードにも手を入れていない。CPUのみで、GPUは使っていない。

## 実行

- スクリプト：scratchpad の `seqloader_check.py`（sha256 `d9ddfe4384ff…`）。研究リポジトリには入れていない。研究リポジトリの `.venv` を使い、`import sequential_loader` で公開APIだけを呼んだ。
- 経路：
  1. `ActivityNetAdapter(dataset_root=/mnt/NAS-TVS872XT/dataset/ActivityNet).sequence_sources("training")`（inventoryの検証は2.0秒、10,024本）
  2. `SequentialDataset(reader=SequentialVideoReader(), chunk=FixedChunkConfig(16))`
  3. `build_sequential_dataloader`（strictの既定値）
  4. `sequential_sample_stream`
- 1本ごとに独立したdatasetで読んだ（A6のみ、3本を続けて読むdataset）。並列は動画単位の32プロセス。
- 時刻：21:50:15開始 → 22:08:34終了（18分）。読んだ有効フレームは合計854,532枚。

### 対象の動画（seed 0）

| 種類 | 本数 | 内容 |
|---|---|---|
| 無作為 | 200 | 下の特殊な動画を除いたtrainingから抽出 |
| 5秒未満 | 26 | 単一フレームmanifestで1枚も取れなかった全本（1.4〜5.0秒） |
| webm | 10 | trainingの全本 |
| mkv | 5 | 無作為 |
| 最長 | 5 | 最長755秒 |
| 59.94fps／15fps／端数fps（30.01など） | 各3 | 無作為 |
| durationをcontainerから取った動画 | 3 | 無作為 |
| 11/02にseekの問題が出た動画 | 1 | `v_5kBKAfEX7XA`（1080p、45.6秒） |

## 結果

| # | 確認内容 | 結果 |
|---|---|---|
| A1 | エラーなくEOFまで読める | **259/259本、エラー0件**（特殊な動画もすべて） |
| A2 | chunkの並び：`sequence_index` が0から1ずつ増える。`is_first` は先頭だけ、`is_last` は最後だけ。paddingは最後のchunkだけで、中身はゼロ。frame indexは0から途切れずに続く。T=16 | **問題0件**（全259本） |
| A3 | 有効フレームの総数が、`ffprobe -count_frames` の値と一致する | **259/259本で一致** |
| A4 | timestampの逆行 | **0件** |
| A4 | timestampの重複 | 19本で各1か所。**すべて動画の最後の1〜2フレーム**（最後のBフレームが直前と同じPTSを持つ） |
| A4 | 約1.5フレームより大きい飛び | 21本で各1か所。**すべて最後のフレーム**（直前から約2フレーム分、0.06〜0.08秒の間隔。Pフレーム） |
| A4 | 最初のtimestamp | 全本で0.0 |
| A5 | loaderのi番目の画像と、そのtimestampで `decode_frame` が返す画像がbitwise一致する | **2,483か所すべて一致**（先頭・末尾・無作為4か所＝1,548、keyframe＝469、keyframeの直前＝466） |
| A6 | 3本を1つのdatasetで続けて読む：動画の順番、切り替わりでのリセット、`is_last` から次の動画への移行 | 2組（5秒未満の3本、短い無作為の3本）とも問題なし。各動画の画像ハッシュは、1本ずつ読んだ結果と一致 |
| A7 | 同じ動画を2回読むと、全フレームのハッシュが一致する | **10/10本で一致** |
| A8 | `sequential_sample_stream` を1 sampleで抜ける：開いているファイル数 | 3本とも、開始前4 → 読み込み中5 → 抜けた後4（元に戻った） |

- A4の重複と飛びは、ffprobe（`-show_entries frame=pts_time`）でも同じ位置・同じ値が出た。loaderはPTSをそのまま返している（RAW mode）。動画ファイルの末尾の性質で、loaderの誤りではない。
- 重複したtimestampで `decode_frame` を呼ぶと、同じPTSを持つ2枚のうち後ろの1枚が返る。今回のA5では、比較した位置にこの重複フレームは含まれていなかった。学習でtimestampを使う場合は、末尾にこの重複があることに注意する。

## B1：読み込み速度（参考）

1プロセスあたり、decodeとRGB変換だけの速さ（resizeなし）。32プロセスを同時に動かした条件で測った。

| 解像度 | 本数 | フレーム/秒/プロセス |
|---|---|---|
| 1280×720 | 76 | 44.9 |
| 1920×1080 | 48 | 21.9 |
| 640×480 | 40 | 119.4 |
| 320×240 | 23 | 283.9 |
| 640×360 | 10 | 149.2 |
| 854×480 | 9 | 93.2 |
| 全体 | 259 | 45.5 |

- 無作為200本では、1プロセスが1秒の処理で、動画1.72秒分を読んだ。training全体（約326時間＝117万秒）を1周すると、**約68万プロセス秒（約190時間）**になる。
- strict設定（batch 1、worker 0）は1プロセスで動くので、そのままでは1 epochに**約8日**かかる見積もりになる。動画単位で32並列にできれば、約6時間になる。ただし、並列化はloaderに実装されていない。
- この見積もりは、モデルの計算・resize・データ拡張を含まない。

## 言えること・言えないこと

- 言えること：抽出した259本の範囲で、sequential-loaderは次の点で正しく動いた。
  - 動画を先頭からEOFまで、正しい順番・正しいフレーム数で返した。
  - 画像は、seekを使う別経路（`decode_frame`）で取り出したものとbitwise一致した。
  - 再現性があり、途中で抜けても資源は解放された。
- 言えないこと：
  - training全10,024本で問題がないこと（今回は約2.6%）。
  - validation・testingの動画での挙動。
  - strideでの間引き・batch化・並列化（loaderに機能がない）。
  - ラベル・区間とフレームの対応。
  - 学習に使ったときの速度（モデル計算を含む）。

## 出力（scratchpad、研究ワークスペースには入れていない）

- `seqcheck-full/selection.json`（選んだ動画とtags）
- `seqcheck-full/videos.jsonl`（1本ごとの結果）
- `seqcheck-full/extra.json`（A6・A8）
- `seqcheck-full.log`
- 集計：`seqcheck_summary.py`、A4の位置の確認：`seqcheck_ts_where.py`
- 試運転：4本（5秒未満・webm・mkv・`v_5kBKAfEX7XA`）で全項目問題なし（`seqcheck-dry/`）
