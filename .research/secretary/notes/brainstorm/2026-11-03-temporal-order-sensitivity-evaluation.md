---
date: 2026-11-03
project: sequential-video-lora-analysis
source_todo: null
topic: temporal-order-sensitivity-evaluation
status: exploratory
tags: [brainstorm, research, evaluation, temporal, early-fusion, activitynet]
---

# 時間情報の評価：clip特徴の並び順への反応（ラベルなし、探索）

## 出発点

2026-11-03、ActivityNet clip Early FusionのA1・A2が、両方式（LoRA・Full FT）で工学PASSした。そこでユーザーが「LoRA（モデル）が時間情報を獲得したかを評価する方法を決めたい」と考えた。lossが下がっただけでは、時間情報を獲得したとは言えない。

壁打ちの途中で、sequential-loaderの確認と、strideとmulti-stream DDPの実装を先に行い、そのあとこの壁打ちに戻った。

## 読み込んだ文脈

- [clip Early Fusion spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-11-03-vit-moco-lora-activitynet-clip-early-fusion-spec.md)、clip A2の[LoRA](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-vit-moco-lora-activitynet-clip-a2-lora-gpu-evidence.md)と[Full FT](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-vit-moco-lora-activitynet-clip-a2-full-ft-gpu-evidence.md)。
  - clipは4フレーム、0.25秒間隔で、manifestの時刻を最後のフレームにする。
  - token結合で785 token。固定sin-cosの時間位置埋め込み（t=0でゼロ）を使い、CLSを出力にする。
  - MoCoのpositiveは、同じclipの2拡張。
- 過去の評価の壁打ち（[9/17 MAE LoRA時間情報評価](2026-09-17-mae-lora-temporal-information-evaluation.md)、[10/21 静的・動的情報の3論文](2026-10-21-static-dynamic-information-goyal-choi-kowal-review.md)、[11/01 early fusion](2026-11-01-early-fusion-vit-lora-temporal.md)）：
  - 時間構造だけを壊して比べる（normal、shuffle、reverse、static-repeat）。
  - 事前学習モデル、フレームごとに学習したLoRA、時間を含めて学習したLoRAの3つを比べる。
- 使えるcheckpoint（研究リポジトリ `log/` 内、すべて459step、LR 3e-4）：
  - 単一フレーム LoRA：`moco_vit_lr_activitynet/lora/activitynet_frames/lr_0.0003/2026-11-03_10-48-36-408110/`
  - 単一フレーム Full FT：`…/full_ft/activitynet_frames/lr_0.0003/2026-11-03_11-49-57-818631/`
  - clip LoRA：`moco_vit_lr_activitynet_clip/lora/activitynet_clips/lr_0.0003/2026-11-03_15-22-36-985580/`
  - clip Full FT：`…/full_ft/activitynet_clips/lr_0.0003/2026-11-03_16-46-46-287044/`
- ActivityNet validationは4,926本で、ラベルもある（今回は使わない）。manifest作成ツール（`scripts/vit_moco/activitynet_frames.py`）は、`--split` に `training` しか受け付けない。

## 問い

clip特徴は、フレームの**並び順**（と動きの有無）によって変わるか。変わるなら、それは**clipでの学習で身についたもの**か。

## ユーザーが採用した方針（2026-11-03）

| 論点 | 採用 |
|---|---|
| 評価の範囲 | **並び順への反応だけ**（ラベルなし）。ラベルありの課題（SSv2などのlinear probe）は、今回は行わない |
| 評価に使う動画 | **ActivityNet validation**（学習と同じ種類の動画で、学習には使っていない） |
| E1 clipの形 | **学習と同じ**（4フレーム、0.25秒間隔、manifestの時刻を最後のフレームにする） |
| E2 clipの数 | validationのmanifest（10秒ごと、約57,000 clipの見込み）から、**seedで固定して無作為に2,000 clip** |
| E3 並べ方 | 通常（1→2→3→4）、**シャッフル（恒等以外の並びを、clipごとにseedで1つ選ぶ）**、逆再生（4→3→2→1）、静止（最後のフレームを4回繰り返す） |
| E4 比べる特徴 | **CLS（backboneの出力）とq（正規化したMoCoの特徴）の両方** |
| E5 ものさし | **基準になる類似度も測る**：同じ動画で10秒離れたclipとの類似度、別の動画のclipとの類似度 |
| E6 動きの大きさ | **clipごとにフレーム間の画素の差を記録**し、動きの大小で分けて集計する |

比べるモデル（追加の学習はしない）：

| 記号 | 中身 |
|---|---|
| A | 学習前（事前学習ViT＋時間位置埋め込み、step0と同じ初期値） |
| B-LoRA／B-FT | 単一フレームA2のcheckpointを、clip用のbackboneに読み込んだもの |
| C-LoRA／C-FT | clip A2のcheckpoint |

## 結果の読み方（事前に決めておく解釈）

| 結果のパターン | 解釈 |
|---|---|
| A・B・Cとも、cosがほぼ1 | どのモデルも見た目しか使っていない。今のclip学習では、時間情報は入っていない |
| Cだけcosが下がる（特にシャッフル・逆再生で） | clipでの学習で、並び順を区別するようになった |
| A・B・Cが同じくらい下がる | 反応は構造（時間位置埋め込みとjoint attention）から来ていて、学習で身についたものではない |
| 静止だけ大きく下がる | 動きの有無、またはフレームの見た目の違いには反応するが、順番は区別していない |

予想：今のclip学習は、同じclipの2拡張をpositiveにしているので、時間を使わなくても解ける。そのため、「時間情報はほとんど入っていない」（CがA・Bとほぼ同じ）になる可能性が高い。その場合も、学習方法を変えたときに比べるための基準として使える。

## 反例・注意点

1. **シャッフル・逆再生**は、使うフレームは通常と同じで、順番だけが違う。そのため、純粋に「順番への反応」を測れる。**静止**はフレームの中身も変わる（3枚分の画像がなくなる）ので、「動きの有無への反応」と「見た目が変わったことへの反応」が混ざる。
2. **Aでも反応は0にならない**。時間位置埋め込みがあるので、並べ替えると入力が変わる。したがって、「変わったかどうか」ではなく、**AやBとの差**で読む。
3. **Bのモデル**は、時間位置埋め込みとフレームをまたぐattentionを、学習で一度も見ていない。clip用の構造に載せること自体が、学習時とは違う使い方になる。
4. **qの段階**では、Aのprojectorはseed 0のランダムな初期値である。qでのAの値は、「学習していないprojectorを通した値」として読む。
5. **clipが0.75秒しかない**。動きの小さいclipでは、並べ替えても入力がほとんど変わらない。そのためE6で分けて読む。動きの指標（画素の差）は、カメラの動きや照明の変化も拾う。
6. 2,000 clip・seed 1つ・各条件1回の結果である。平均の差には、clip単位のpairedなbootstrapで信頼区間を付ける案がある（未決）。

## 未決事項（spec前に決めること）

- 評価時の画像の前処理：fixed probeと同じにするか（resizeとcenter crop、正規化0.5）。データ拡張はしない。
- 動きの指標の定義（隣り合うフレームの画素の平均絶対差など）と、大小で分ける境目（中央値で2つに分ける、など）。
- 「別の動画のclip」の選び方（clipごとに無作為に1つ）と、「10秒離れたclip」が動画の範囲外になる場合の扱い。
- 集計方法：平均と中央値、pairedなbootstrapの信頼区間の有無。
- 実装の場所：研究コードのbranch（`feature-vit-moco` から新しいbranchを切るか）。validation manifestへの対応（`--split validation` を加える）と、評価用スクリプトを新しく作る。
- 出力形式と保存先（研究リポジトリの `log/` の下に新しいディレクトリを作るか）。
- 計算資源：推論だけで済む。GPUを使う場合は、別に許可を取る。

## 次アクション候補（未採用・TODO未追加）

1. 未決事項を決めて、spec化する（`research-spec`）。
2. 実装し、CPUで検証する。
3. 許可を得てから、評価を実行する。

## 関連ファイル

- [clip Early Fusionの壁打ち](2026-11-03-activitynet-clip-early-fusion-loss-check.md)
- [early fusionの壁打ち（11/01）](2026-11-01-early-fusion-vit-lora-temporal.md)
- [MAE LoRA時間情報評価（9/17）](2026-09-17-mae-lora-temporal-information-evaluation.md)

## 追記（2026-11-03）：spec化

ユーザーが、spec前の細部P1〜P10をすべて推薦案で採用した。
- P1：2,000本の別々の動画から、1 clipずつ選ぶ。
- P2：10秒後のclip（なければ10秒前）。
- P3：別の動画のclipを組み合わせる。
- P4：隣り合うフレームの画素の差の平均で、中央値を境に2つに分ける。
- P5：pairedなbootstrapで95%信頼区間を出す。
- P6：fixed probeと同じ前処理。
- P7：query側のCLSとq。
- P8：`feature-vit-moco` で実装する。
- P9：テストはCPU、評価はGPU 1枚（別に許可を取る）。
- P10：probe特徴のsha256で、読み込みが正しいかを確認する。

draftのspecは次のとおり：[時間情報の評価spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-11-03-temporal-order-sensitivity-evaluation-spec.md)

## 追記（2026-11-04）：実装

specが承認され、実装した（CPUテストはPASS、未commit、R0・R1は未実施）：[実装記録](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-04-temporal-order-sensitivity-evaluation-implementation-cpu-verification.md)
