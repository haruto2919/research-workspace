---
date: 2026-10-08
project: null
related_research_project: sequential-video-lora-analysis
source_todo: null
topic: streaming-moco-lab-tour-demo
status: exploratory
record_type: outreach-demo-brainstorm
tags: [brainstorm, outreach, lab-tour, interactive, streaming-moco, non-research]
---

# 研究室見学用 Streaming MoCo 体験アプリ：B案のブラッシュアップ

## スコープとAuthority

2026-10-08、ユーザーが前回の候補「B. AIに動画を順番に覚えさせよう！」を選び、さらにブラッシュアップすることを依頼。関連する一般企画は [lab-tour-video-ai-demo](2026-10-08-lab-tour-video-ai-demo.md)。

これは見学者向け教育・広報デモの**探索案**。研究上の判断・spec・実装承認・性能評価ではない。研究コード、研究README、spec、experiment、TODOは変更しない。実デモアプリもまだ未実装。

## 確認した研究文脈（GitHub、2026-10-08）

- 研究SSOT: `.research/lab/projects/sequential-video-lora-analysis/README.md` と2026-10-05 approved full-dataset Streaming MoCo/Linear Probe spec、2026-09-24 MTG。
- 実装: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` HEAD `65dedb4ebc26c48d23606b9bc674044f64e64c06`。
- コード: `conf/moco/stage6b_v2.yaml`、`training/moco_protocol.py`、`training/streaming_moco.py`、`training/streaming_moco_full.py`、`self_supervised/moco/vit_lora_moco.py`。
- 16フレーム/chunk。現在chunkのraw Query viewとGBR色変換＋水平反転のKey viewがpositive pair。
- 比較に使用するnegativesは更新前のQueueの過去Key特徴。Stage6B-v2は`all_past`のため同じ動画の過去chunkも含む。違う行動であるとは限らない。
- 正しい学習順: `old queueと対比してloss → backward → Query LoRA＋Query Projector更新 → Key LoRA＋Key ProjectorをEMA更新 → 比較時に計算済みの現在Keyをenqueue`。
- Queue容量は4096（128次元の投影後特徴＋sequence metadata）。古いものから押し出すFIFO。
- 最初の動画の最初のchunkはKey-only warm-up。学習更新はせずQueueに初期Keyのみを入れる。動画間でモデル・Queue・optimizerを初期化し直さない。
- Base ViTは凍結。Query側のLoRAとprojectorのみをoptimizerで更新。現状の学習ログだけで動画動作・順序情報の獲得や性能向上は未証明。

## 展示コンセプト（提案）

仮タイトル **「AI Video Learning Lab — 動画を順番に見せて、学習をのぞこう」**。

重要な言い換え: 「過去の動画を記憶して行動を理解する」ではなく、**「過去のchunkの特徴を保存し、現在のchunkを変形したもの・過去の特徴と比較し、少量の追加パラメータを更新する」**。

目標はMoCoの式を暗記させるのではなく、見学者が操作を通じて「同じ場面の別の見え方と過去の場面を比較して学習する」構造を理解すること。

### 1画面UI案

- **上段の主役**: 自作/権利処理済みの短い動画、タイムライン、現在の16-frame chunkハイライト。最初は動画A、途中で動画Bへ移行。
- **中央左（青）**: 元chunk → Query encoder + Projector。元の映像から現在の特徴を生成する視点。
- **中央右（緑）**: 同じchunkにGBR色変換＋左右反転をかける → Key encoder + Projector。これが同じchunkのpositive。
- **中央下**: Queueの過去Keyカードを横に並べ、赤の比較線をQueryから伸ばす。昔の特徴の順序を可視化。カードをタップすると動画名、chunk番号、保存順を見られる。
- **下段**: `比較→Query LoRA+Projector更新→Key EMA→現在Key保存`の処理順表示。Base ViTは鍵マーク。重要処理だけ点滅。
- **画面上の短い説明**: 1ステップ1文。「今の映像を少し変えて同じものを作った」「前に見た映像の特徴と比べている」など。
- 常に `教育用シミュレーション／実際の研究実験の結果ではありません` と表示。

### 最小のインタラクション

1. `次の16フレームへ`（メイン操作）：一押しで1chunkの処理順アニメーションが進む。`戻す`、`最初から`も用意。
2. `同じ場面はどれ？`：元chunkに対応する変形Keyをカードから選ぶミニクイズ。研究の「positive」を説明。
3. `過去の特徴をタップ`：同じ動画の過去chunkもnegative比較対象になり得ることを見せる。
4. `詳しく見る`：初学者向けの言い換えと、Query/Key/InfoNCE/EMA/LoRA/FIFOという研究用語表示を切り替える。

### 90秒で説明できる台本案

- 0〜15秒: 「AIに動画を1つずつ見せてみます」と動画Aを選ぶ。
- 15〜30秒: chunk A1の最初のKeyだけをQueueへ入れ、初回準備を見せる。
- 30〜60秒: chunk A2に進み、元映像と変形版のpositive、Queue中のA1を比較。Query側LoRAとprojectorが更新され、KeyはEMAで追従した後、新しいKeyがQueueへ入る。
- 60〜75秒: A3、B1へ進めて過去のAがQueueに残ることを見せる。
- 75〜90秒: 「動画の順番・過去情報を使いながら、AIの小さな追加部品を学習しています。何が本当に学べるかを研究しています」と締める。
時間は提案上の仮定で確定していない。

## MVP / 拡張 / 対象外

**MVP推奨**: 完全オフライン動作するブラウザの教育用リプレイ。デモ用のサンプル動画、2視点画像、Queuing、Query/Key/LoRA更新を決定的に再生。重いGPU学習は不要。

**拡張候補**:
- Queue 2D特徴マップ：初期は模式図であると明示。実測モードを作るならデータの由来と抽出方法を示す。
- ごく小さなtoy contrastive learnerの実際のパラメータ更新。ただし研究ViT/MoCoとは異なると必ず明示。
- 事前に検証した実ViT特徴や研究トレースをリプレイする「実測モード」。性能・権利・実験整合性を検証した後に検討。
- 再生速度、自動モード、全画面表示。

**初回対象外**: 研究full runのライブ学習、根拠のない学習成功率・理解度メーター、検証されていないLoRAの動作理解の主張、研究用データセットの無許可公開。

## 注意・反例・未決

- Queueは元動画そのものの保存ではなく、埋め込み表現のFIFO。UIに数件しか表示しない場合、**研究設定のK=4096と表示件数を区別**する。
- 「negative = 似ていない行動」とは言わない。同じ動画のすぐ前のchunkもnegative候補なので、誤った分離を促すfalse negativeの可能性がある。
- 最初のwarm-upで学習を描かない。
- 現在Keyはold queueとのloss計算後に登録する。Keyを登録してから現在自分自身をnegativeとして使う描写は誤り。
- 更新するのはQuery LoRAとQuery Projector。KeyのLoRA/ProjectorはEMAのみ。Base ViTは固定。
- 動画の**逐次提示順**を利用することと、各16フレーム内の**時間方向が表現に入ること**は別。現行Masked Meanはclip内frame順に不変。
- 未決: 見学者の人数・学年・時間、オフライン表示機器、サンプル動画、シミュレーションかtoy実学習か、研究用語の深さ。

## 次の判断候補（まだTODO化・spec化しない）

B案の初回MVPを「chunk送り＋変形Positive表示＋過去Queue比較＋LoRA更新アニメーション＋動画境界の記憶保持」に絞るか。その後に独立デモ用のspec化を依頼するか。

**Authority: exploratory / outreach-only。研究用の実装・実験・README・spec・TODOに影響しない。**
