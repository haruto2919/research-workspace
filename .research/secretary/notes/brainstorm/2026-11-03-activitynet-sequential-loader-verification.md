---
date: 2026-11-03
project: sequential-video-lora-analysis
source_todo: null
topic: activitynet-sequential-loader-verification
status: exploratory
tags: [brainstorm, research, sequential-loader, activitynet, verification]
---

# ActivityNet用sequential-loaderが正常に機能するかの確認項目（探索）

## 出発点

2026-11-03、clip Early FusionのA1・A2が両方式で工学PASSになった。その後、時間情報の評価方法（順序への感度、A/B/Cの比較）を壁打ちしている途中で、ユーザーが次の意向を示した。

- 「その前に、sequential-loader（ActivityNet用）が正常に機能することを確認したい」

時間情報の評価の壁打ちは、ここで**一時中断**した。中断時点での候補は、ラベルなしの順序感度（通常・シャッフル・逆再生・静止 × A：step0、B：単一フレームA2、C：clip A2）を見て、そのあとSSv2などでlinear probeを行う、という2段構えだった。評価データ（ActivityNet val／SSv2）はまだ選んでいない。

## 読み込んだ文脈

- [sequential-loader解説（code-analysis）](../../../lab/projects/sequential-video-lora-analysis/code-analysis/2026-11-03-sequential-loader-activitynet-explain.md)：`2026_09_ishikawa_sequential_loader@19a0ed7`（ActivityNet branch、clean）の構成。
  - Adapterが動画全体を1 sequenceとして渡す。
  - Readerは先頭から前向きにdecodeし、連続16フレームのchunkを作る。frame indexはdecode順で、timestampはRAWのPTS。
  - strictのDataLoaderのみ（batch 1、worker 0）。strideはない。
- [2026-09-23 Adapter検証](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-09-23-activitynet-adapter-verification.md)で確認済みの範囲：全183テストOK、実データinventory 19,994本、3形式各1本の先頭chunk。
- [2026-09-23 clip feature検証](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-09-23-activitynet-clip-feature-verification.md)で確認済みの範囲：training先頭1 chunkをfrozen ViTに通した。
- 単一フレームmanifest（2026-10-21、training 10,024本）から確認した値：
  - 合計の長さは約326時間（中央値114秒、最長755秒）。
  - 拡張子はmp4 9,062本、mkv 952本、webm 10本。
  - fpsは29.97／30／25／23.98／15／59.94などで、30.01や29.98のような端数もある。
  - 解像度は320×240から1920×1080まである。`start_time` が0でない動画はない。
  - フレームが取れなかった26本は、すべて5秒未満の動画。
- 11/02の `decode_frame` 修正後、manifestの全117,522枚をエラー0件で取り出せた。ただし、これはseekを使う別経路である。

## 問い

sequential-loaderは、実際のActivityNet動画を、先頭から最後まで、正しいフレーム・正しい順番・正しい時刻で、エラーなく返すか。読み込みの速さはどの程度か。

## ユーザーが採用した方針（2026-11-03）

| 論点 | 採用 |
|---|---|
| 確認の中心 | 「正しく読めているか」（A1〜A8）。「学習に使えるか」は速度（B1）だけを参考に測る |
| 確認する動画 | **trainingから無作為に約200本＋問題が出やすい動画** |
| 速度（B1） | **測る**（A1を回すついでに記録する） |

## 確認項目（ユーザーが一覧を確認し、足りない・不要の指摘はなかった）

| # | 確認内容 | 方法の候補 |
|---|---|---|
| A1 | エラーなくEOFまで読める | 対象の全動画を最後まで回し、例外を記録する |
| A2 | chunkの並び：frame indexが0から途切れない、`sequence_index` が1ずつ増える、`is_first` は先頭だけ、`is_last` は最後だけ、paddingは最後のchunkだけ | 全sampleを機械的にチェックする |
| A3 | 有効フレームの総数が、別の方法で数えた数と一致する | ffprobeで数える（`-count_packets` は速いが、packet数とframe数は一致しない場合がある。`-count_frames` は全decodeなので遅い。未決） |
| A4 | timestampが逆行しない・重複しない・約1/fpsより大きく飛ばない | 件数と最大の飛びを記録する |
| A5 | **中身が正しい**：i番目の画像が、そのtimestampで `decode_frame` が返す画像とbitwise一致する | 1本につき数か所（先頭・途中・末尾・keyframeの前後）。**最重要**（A1〜A4は形しか見ない） |
| A6 | 動画が切り替わると、番号と印がリセットされ、別の動画の画像になる | 連続する2本の境目を確認する |
| A7 | 同じ動画を2回読むと、結果がまったく同じになる | 画像のハッシュを比べる |
| A8 | 途中で抜けても、ファイルが開いたまま残らない | 抜ける前後で、開いているファイル数を比べる |
| B1 | 読み込み速度（フレーム/秒、1本あたりの時間）と、training 1周の見積もり | A1と同時に記録する |

### 問題が出やすい動画の候補（AIの推薦、ユーザー未確認）

- `v_5kBKAfEX7XA`（11/02にseekの問題が出た動画、1080p）
- フレームが取れなかった5秒未満の26本（すべて）
- webm 10本（すべて）と、mkvの数本
- 最長の数本（約750秒）
- fpsが59.94／15の動画、端数のfps（30.01など）の動画
- durationをcontainerから取った動画の数本

### 対象外

- strideでの間引き、batch化・並列化・DDP、ラベルとフレームの対応。loaderに機能がないか、学習の設計で決めることである。
- sequential-loader本体の変更。

## 反例・注意点

1. A1〜A4がすべて通っても、画像の中身が正しいとは限らない。そのためA5を必須にする。
2. A5の比較相手 `decode_frame` も、ffmpegのdecodeに依存している。両方が同じように間違えている場合は検出できない。ただし、seekありの経路と前向きdecodeの経路という、読み方の違う2つが一致すれば、強いEvidenceになる。
3. `decode_frame` は「timestamp以下で最後のフレーム」を返す。同じPTSのフレームが複数あったり、PTSが逆行したりすると、一致しない可能性がある。その場合はA4の結果と合わせて読む。
4. 1080pの動画が多いので、decodeとRGB変換が重い。速度は解像度ごとに分けて記録したほうが読みやすい。
5. 無作為の200本は、10,024本の約2%である。0件だったとしても「全動画で問題がない」とは言えない。言えるのは「抽出した範囲で問題がなかった」までである。

## 未決事項（実行前に決めること）

- A3の数え方（`-count_packets` か `-count_frames` か）。
- A5で、1本あたり何か所をどう選ぶか。
- 無作為抽出のseedと、特殊な動画の最終的なリスト。
- 確認スクリプトの置き場所（研究リポジトリには入れず、scratchpadに置く案。clip decodeの確認と同じ扱い）と、結果の記録先（`experiments/`）。
- CPUのプロセス数（32プロセスを想定）。

## 次アクション候補（未採用・TODO未追加）

1. 未決事項を決めて、確認スクリプトを作る。sequential-loaderと研究コードは変更しない。
2. CPUで実行し、結果を `experiments/` に記録する。
3. 確認が終わったら、時間情報の評価の壁打ちに戻る。

## 関連ファイル

- [sequential-loader解説](../../../lab/projects/sequential-video-lora-analysis/code-analysis/2026-11-03-sequential-loader-activitynet-explain.md)
- [clip Early Fusionの壁打ち](2026-11-03-activitynet-clip-early-fusion-loss-check.md)
- [ActivityNet Adapter spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-09-23-activitynet-adapter-spec.md)

## 追記（2026-11-03 22:20）：確認の実行

ユーザーが推薦案（259本、`-count_frames`、1本あたり約10か所、32プロセス）を採用し、CPUで実行した。A1〜A8はすべて問題なしだった。timestampの重複・飛びは、末尾の1〜2フレームだけにあった（動画ファイル側の性質）。B1は45.5フレーム/秒/プロセスだった。結果：[sequential-loader確認記録](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-activitynet-sequential-loader-verification.md)
