---
project: sequential-video-lora-analysis
date: 2026-11-03
spec_type: evaluation
status: implemented
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_base_commit: fe58fc37ef28901184980ccd5f9e90e40c3d836d
source_brainstorm: ../../../../secretary/notes/brainstorm/2026-11-03-temporal-order-sensitivity-evaluation.md
---

# 時間情報の評価：clip特徴の並び順への反応（ActivityNet validation）

**Status: implemented（2026-11-04。CPUテストT1〜T6はPASS、失敗ID集合は変更前と同じ。研究コードは `feature-vit-moco@50ff715`。R0・R1は2026-11-04に実施（[R1 Evidence](../experiments/2026-11-04-temporal-order-sensitivity-evaluation-r1-gpu-evidence.md)）。[実装記録](../experiments/2026-11-04-temporal-order-sensitivity-evaluation-implementation-cpu-verification.md)）**。**Status: approved（2026-11-03、ユーザーが承認）**。実装は、別に指示を受けてから行う。manifestの作成（R0）と評価の実行（R1、GPU）は、それぞれ別に許可が必要。

## 1. 目的

学習が終わったモデルについて、clipの特徴が**フレームの並び順**（と動きの有無）によって変わるか、変わるならそれが**clipでの学習で身についたものか**を、ラベルを使わずに測る。

今回は、今あるcheckpoint（459stepの短い学習）にあてはめて**基準値**を得る。同じ評価を、あとで行う学習（sequential入力、stride、学習目標の変更）にも使う。

## 2. 前提（読み取りで確認した内容）

- 研究コード：`feature-vit-moco@fe58fc3`（clean）。
- clip Early Fusionの経路：
  - 4フレームを0.25秒間隔で取り、manifestの時刻を最後のフレームにする。
  - `ViTMoCoClipBackbone` でtoken結合し（785 token）、固定sin-cosの時間位置埋め込み（t=0でゼロ、bufferなので保存されない）を加える。出力はCLS。
  - clipは `decode_clip` で取り出す（実動画2,005 clipで `decode_frame` とbitwise一致を確認済み）。
- 特徴の取り出し：既存の `stage_features` は、query側（`backbone_q`、`projector_q`）をevalモードにして、CLS → projector（z_q）→ L2正規化（q）の順に取り出す。
- checkpoint：`checkpoint_step000459.pt` の `model` に、`ViTDistributedMoCo` 全体のstate dictが入っている。単一フレームのcheckpointも、clip用のモデルにstrictで読み込める（parameterの名前が同じで、時間位置埋め込みは保存されないため）。
- fixed probeの前処理：Resize(256) → CenterCrop(224) → ToTensor → Normalize(0.5, 0.5)。学習のときに、各probe stepの特徴のsha256を `diagnostic_summary.json` に記録している。
- manifest作成ツール `scripts/vit_moco/activitynet_frames.py` は、`--split` に `training` しか受け付けない。`sequential_loader.ActivityNetAdapter` はvalidation（4,926本）に対応している。
- 研究コードの `.venv` が読み込むsequential-loaderは、今は `feature-stride-ddp@85a39c2` である。ActivityNet Adapterは `19a0ed7` から変わっていない。

## 3. 評価するモデル（追加の学習はしない）

| 記号 | 中身 | 重み |
|---|---|---|
| A | 学習前 | clip presetで組み立てたモデルの初期値（seed 0）。LoRAとFull FTでstep0の特徴はbitwise一致するので、1つだけ評価する |
| B-LoRA | 単一フレームで学習 | `log/moco_vit_lr_activitynet/lora/activitynet_frames/lr_0.0003/2026-11-03_10-48-36-408110/checkpoint_step000459.pt` を、clip用のモデルに読み込む |
| B-FT | 同上（Full FT） | `…/full_ft/activitynet_frames/lr_0.0003/2026-11-03_11-49-57-818631/checkpoint_step000459.pt` |
| C-LoRA | clipで学習 | `log/moco_vit_lr_activitynet_clip/lora/activitynet_clips/lr_0.0003/2026-11-03_15-22-36-985580/checkpoint_step000459.pt` |
| C-FT | 同上（Full FT） | `…/full_ft/activitynet_clips/lr_0.0003/2026-11-03_16-46-46-287044/checkpoint_step000459.pt` |

- どのモデルも、clipのpresetと同じ構造（T=4、clip backbone）にする。checkpointは `model` をstrictで読み込み、query側だけを使う。
- 各checkpointのsha256と、元のrunの `config_identity_sha256` を記録する。

## 4. データ

| # | 内容 |
|---|---|
| D1 | ActivityNet v1.3 **validation**のmanifestを、trainingと同じ規則（10秒間隔、t = 5 + 10n、動画の長さ未満）で作る。出力先は新しいディレクトリ（`log/activitynet_frames/<日付>_validation_interval10s_offset5s/`）。 |
| D2 | clipが1つ以上ある動画から、**2,000本**をseed 0で選ぶ。各動画から、seed 0でclipを**1つ**選ぶ。 |
| D3 | 各clipは、学習と同じ形（4フレーム、0.25秒間隔、manifestの時刻tを最後のフレームにする：t−0.75, t−0.5, t−0.25, t）。`decode_clip` で取り出す。 |
| D4 | 前処理は、fixed probeと同じにする（Resize(256) → CenterCrop(224) → ToTensor → Normalize(0.5, 0.5)）。4フレームすべてに同じ処理をかけ、データ拡張はしない。 |

## 5. 入力の条件

元のclip（フレーム1, 2, 3, 4）から、次を作る。

| 条件 | 中身 |
|---|---|
| normal | 1, 2, 3, 4 |
| shuffle | 恒等以外の23通りの並べ方から、clipごとにseed 0で1つを選ぶ |
| reverse | 4, 3, 2, 1 |
| static | 4, 4, 4, 4（最後のフレームを繰り返す） |
| same-video | 同じ動画の、10秒後のclip（なければ10秒前のclip）。normalの並び。どちらもない動画は、この条件だけから外し、外した数を記録する |
| other-video | 選んだ2,000 clipの中から、別の動画のclipをseed 0で1つずつ組み合わせる（normalの並び）。特徴はnormalのものを使うので、再計算しない |

- 並べ方と組み合わせは、seedだけで決まる。全モデルで同じものを使う。

## 6. 測る値

- 各モデル・各clipで、**CLS**と**q**それぞれについて、normalの特徴と各条件の特徴とのcos類似度を計算する。
- **動きの大きさ**：前処理後で正規化する前（ToTensorの後、0〜1）の画像について、隣り合うフレームの画素の差の絶対値を平均し、3つの差の平均をとる。全2,000 clipの中央値を境に、「動き大」と「動き小」に分ける。

## 7. 集計

- モデル × 条件 × 特徴（CLS／q）× 動き（全体／大／小）ごとに、cos類似度の平均・中央値と、clip単位のbootstrap（1,000回、seed 0）による平均の95%信頼区間を出す。
- モデル間の差（C−A、C−B、B−A。LoRAとFull FTそれぞれ）は、同じclipでの差の平均をとり、pairedなbootstrapで95%信頼区間を付ける。
- 閾値によるPASS／FAILは決めない。結果は記述的に報告する（§9）。

## 8. 実装の範囲

| # | 内容 |
|---|---|
| I1 | `scripts/vit_moco/activitynet_frames.py`：`--split` の選択肢に `validation` を加える。ほかの動き（規則、出力、既存のtraining manifest）は変えない。 |
| I2 | 新しいスクリプト `scripts/vit_moco/temporal_order_eval.py`（CLI）を作る。§3〜§7を実装する。<br>・入力：validation manifest、モデルの一覧（名前と、checkpointのパスまたは `init`）、seed、clip数、device、出力ディレクトリ（新しいものだけ。既にあれば止める）。<br>・出力：`summary.json`（集計、条件の定義、選んだ動画・clipの一覧、外した数、provenance）と、`per_clip.jsonl`（clipごとの各cos類似度と動きの大きさ）。<br>・provenance：研究コードのcommit、dirtyかどうか、checkpointのsha256、manifestのsha256、loaderのgit情報、環境。 |
| I3 | 正しく読み込めたかの確認（P10）：各モデルで、学習時と同じ固定probe（ImageNet 64枚の静止clip）の特徴を同じ方法で計算する。元のrunの `diagnostic_summary.json` に記録された特徴のsha256（Aはstep0、B・Cはstep459）と比べ、一致したかどうかと、一致しない場合は最大の絶対差を記録する。 |
| I4 | テスト `test/model/test_temporal_order_eval.py`（CPU、小さなViT、合成データ）を作る。§10のT1〜T6。 |

- 既存の学習経路・診断・比較器は変えない。新しいdependencyも追加しない。
- 出力先は `log/temporal_order_eval/<日時>/` とする。

## 9. 結果の読み方（記述的な観測。主張の範囲）

| パターン | 読み方 |
|---|---|
| どのモデルもcosがほぼ1 | 見た目しか使っていない。今の学習では、時間情報は入っていない |
| Cだけ（shuffle・reverseで）下がる | clipでの学習で、並び順を区別するようになった |
| A・B・Cが同じくらい下がる | 構造（時間位置埋め込み、joint attention）から来る反応で、学習によるものではない |
| staticだけ大きく下がる | 動きの有無か見た目の違いには反応するが、順番は区別していない |

- 変化の大きさは、same-videoとother-videoの類似度と並べて読む。
- **主張しないこと**：動作の理解、表現の品質、長い学習での結果、0.25秒・T=4の妥当性、LoRAとFull FTのどちらが優れているか。
- Bは、時間位置埋め込みとフレームをまたぐattentionを学習で見ていない。Aのqは、学習していないprojectorを通した値である。

## 10. 成功条件

### CPUテスト（実装の成功）

| # | 内容 |
|---|---|
| T1 | 入力の条件：shuffleは恒等にならず、seedで再現できる。reverseとstaticの並びが正しい。same-videoは10秒後を優先し、なければ10秒前を選び、どちらもない動画は外す。other-videoは別の動画を選ぶ。全モデルで同じ条件が使われる。 |
| T2 | 選び方：2,000本（テストでは小さな数）が別々の動画で、seedで再現できる。 |
| T3 | cos類似度と動きの大きさの計算が、手で計算した値と一致する。bootstrapがseedで再現できる。 |
| T4 | checkpointの読み込み：単一フレームとclipのcheckpoint（小さなViTの合成run）をclip用のモデルにstrictで読み込める。query側の特徴が、元のrunで記録したprobeの特徴と一致する。 |
| T5 | 出力：既存のディレクトリなら止める。`summary.json` と `per_clip.jsonl` の項目がそろう。 |
| T6 | 非回帰：`--split training` の動きが今と同じ。今の失敗IDの集合（36件）が変わらない。 |

### 実データ（実行は別に許可を取る）

| # | 内容 |
|---|---|
| R0 | validation manifestの作成（CPU、NASは読み取りのみ）。動画数、clip数、読めなかった動画の数を記録する。 |
| R1 | 評価の実行（GPU 1枚、tmux）。全モデルで、I3の確認（probe特徴のsha256の一致、または最大の絶対差）を記録する。 |

## 11. 再現性

- 選び方・並べ方・組み合わせ・bootstrapは、すべてseed 0で決まる。
- 研究コードはclean commitの状態で実行する（GPUで評価する場合）。checkpoint・manifest・コードのハッシュを記録する。
- 評価はGPU 1枚で行う。同じGPU・同じbatchの大きさなら、forwardの結果は再現すると考えているが、確認はI3による。

## 12. 対象外

- ラベルを使う評価（linear probe、kNN、SSv2）。
- 学習（新しいrunや、学習の続き）。
- sequential-loader（multi-stream）を使った評価。
- 間隔やフレーム数を変えたclip。

## 13. 履歴

- 2026-11-03：brainstormのE1〜E6とP1〜P10（すべて推薦案）をもとに、draftを作成した。
- 2026-11-03：MoCoの正例・負例（正例は同じclipの2拡張、負例はqueueの65,536 key、同じ動画のclipも負例から外さない）を確認したあと、ユーザーが承認した（approved）。
- 2026-11-04：`feature-vit-moco@fe58fc3` の上に、未commitで実装した（I1〜I4）。CPUテストT1〜T6はPASS。全テストは36 failed / 972 passedで、失敗IDの集合は変更前と同じ。probe特徴の再計算は、小さなViTのrunの記録とbitwise一致した。specからのずれはない。R0・R1は未実施。
