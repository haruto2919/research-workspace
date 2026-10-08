---
date: 2026-10-08
project: sequential-video-lora-analysis
type: meeting-materials
topic: 現行Streaming MoCo-LoRA実験の全体像・評価と研究上の論点
status: draft
target_meeting: null
notion_url: null
notion_export: null
tags: [meeting-materials, research, activitynet, streaming-moco, lora, linear-probe]
---

# MTG資料：現在実装しているStreaming MoCo-LoRA実験の全体像
2026-10-08 作成／現行コード dev（65dedb4ebc26c48d23606b9bc674044f64e64c06）を基準

## 0. 今日のMTGで先生に相談したいこと（Ask）

1. **現在のLinear Probeを「LoRAの有用性の一次評価」と位置付けてよいか。** Base ViTとMoCoで学習したQuery LoRAを、同一のActivityNet動作区間・学習条件で比較する。
2. **「時間情報の獲得」を示すための次の対照実験をどう設計するか。** 現行のframe-wise ViT＋masked meanはchunk内フレーム順に不変であるため、まず「動画内chunkの更新順を保持 vs shuffle」を比較する案を相談したい。
3. **同じ動画の過去chunkをMoCoのNegativeにする妥当性をどう評価するか。** 近接chunkには似た動作が含まれ得るため、false negativeの影響を測る必要がある。
4. **縮小版で事前確認する範囲と、全動画版に進む条件をどう定めるか。** コードには1,000本／1,000本の縮小profileがあるが、実データの200-class coverage・性能は未確認。

## 1. 現時点の結論

- **実装した実験の目的：** 画像事前学習済みViTを固定し、時間順に入力される動画を用いてAttention Q/VのLoRAをMoCoで自己教師あり学習する。得られた特徴表現が動作分類に役立つかを、Base ViTとのLinear Probeで比較する。
- **現在比較できる主張：** 「逐次MoCoで学習したLoRAを追加すると、ActivityNet segment動作分類に有用な特徴が得られるか」。
- **現行実験だけでは言えない主張：** 「LoRAが時間順序や動きを直接理解した」。segment分類性能の差は静的な見た目の変化でも説明できる。
- **実装と実測は別：** Full Pipeline・縮小profile・計測処理はGitHubで確認できた。一方、最新コードでのfull/reduced end-to-end完走、Base対LoRAの最終評価値は今回確認できない。

## 2. 研究目的と処理の全体像

**研究目的：** 画像で学習したViTに動画を逐次入力し、小さい追加パラメータ（LoRA）を通じて動的な情報を獲得・解析できるか調べる。最終的には静的／動的情報の切り分けや下流活用を検討する。

~~~text
ActivityNet v1.3（動画）
  │  Sequential Loader：動画ごとに時系列順、16 frames/chunk
  ▼
画像事前学習済みViT + Q/V LoRA
  │  各frameから CLS特徴 [768] → valid frame平均 [768]
  ▼
Stage 6B Streaming MoCo（動作ラベルは不使用）
  │  Query/Key + Projector [128] + InfoNCE + EMA + FIFO Queue
  ▼
学習済み Query LoRA の最終Snapshot
  │
  ├──────────────────────────────┐
  ▼                              ▼
Base ViT（LoRAなし）       ViT + 学習済みQuery LoRA
  │                              │
  └──── 同一segment manifest ────┘
                 │  frame特徴 → chunk平均 → segment平均 [768]
                 ▼
         各条件のLinear Probe
          200クラス分類 × 3 seeds
                 ▼
        Top-1 / Macro class accuracy 比較
~~~

**注意：** 下流評価で使用するのはProjectorの128次元ではなく、ViT/Query LoRAから得るProjector前の768次元特徴である。

## 3. MoCo学習の仕組み（現在のStage 6B）

### 3.1 入力・表現

| 項目 | 現在の設定 |
|---|---|
| データ | ActivityNet v1.3 training（全動画設定：10,024本） |
| Encoder | google/vit-base-patch16-224（画像事前学習済み） |
| 動画入力 | 16 frames/chunk；1動画を最後まで処理して次の動画へ（strict_single） |
| 特徴量 | 各frameのCLS [768] → valid frameのmasked mean → chunk [768] |
| 学習対象 | Query側ViTのQ/V LoRA（rank 8、alpha 8）とQuery Projector |
| 固定／追従 | Query/KeyのBase ViTは固定；Key LoRA/ProjectorはEMAで追従 |

### 3.2 Positive・Negative・学習更新

- **Query View：** 元のRGB chunk。
- **Positive Key View：** 同じchunkの有効フレームにRGB→GBRのチャンネル入替＋左右反転を適用。
- **Negative：** Queue内の過去のKey全件（同じ動画の過去chunkを含む；現在のPositiveは未投入）。
- **目的関数：** InfoNCE。Positiveとの類似度を、過去のNegativeとの類似度より相対的に高くする。
- **初回のみ：** 最初の動画の最初のchunkをKey-only warm-upとしてQueueへ入れ、LoRA更新は行わない。

~~~text
各chunk：
  2つのView生成 → Query/Keyの特徴・Projector出力
  → 旧QueueをNegativeにInfoNCE計算
  → backward → Query LoRA + Query ProjectorをAdamW更新
  → Key LoRA + Key ProjectorをEMA更新
  → 計算済みPositive KeyをFIFO Queueに追加
  → 次のchunkへ（動画をまたいでも学習状態・Queueは保持）
~~~

| MoCo設定 | 値 |
|---|---:|
| Projection | 128次元 |
| Queue容量 | 4,096 |
| EMA momentum | 0.999 |
| Temperature | 0.07 |
| Optimizer / LR | AdamW / 0.001 |
| Weight decay | 0 |
| データ周回 | 選択したtraining動画を1回ずつ（single pass） |

## 4. Downstream評価：ActivityNet segment Linear Probe

- **評価単位：** 動画全体ではなく、ActivityNet annotationの動作区間（segment）を1サンプルとする。
- **採用chunk：** 有効フレームのtimestampがすべてsegment開始〜終了内に入るchunkだけ。対応chunkがないsegmentは除外し、件数を記録する。
- **特徴：** 各frameのCLSをchunk内で平均し、採用chunkをさらに平均した**768次元segment特徴**。両条件とも同じ作り方で、明示的な正規化はしない。
- **比較条件：** ①Base ViT（LoRAなし） ②Base ViT＋最終Query LoRA。両条件のmanifest、segment ID、データ分割、特徴定義を一致させる。
- **Probe：** 特徴抽出器は固定し、768→200の線形分類器だけ学習。Cross Entropy／AdamW（LR 0.001、weight decay 0.0001）／batch 256／100 epoch／seed 0, 1, 2。
- **評価：** 最終epochのvalidationでTop-1 Accuracy（主指標）、Macro Class Accuracy（副指標）。条件ごとに3 seedの平均と標本標準偏差を集計する。Validationによるearly stoppingは行わない。

**評価できること：** LoRAによって、線形分類で取り出せる動作識別情報が増えたか。

**評価できないこと：** 改善の原因が時間順序の理解か、人物・物体・背景など静的特徴の変化か。

## 5. 全動画版・縮小版と再現性

| source_selection profile | MoCo training | Probe training | Probe validation |
|---|---:|---:|---:|
| activitynet_full_v1（標準） | 10,024本 | 10,024本 | 4,926本 |
| activitynet_reduced_v1（実装済み） | 1,000本 | 1,000本 | 1,000本 |

- 縮小版は動画IDのSHA-256 rankingと専用のselection seedで再現可能に選ぶ。学習の処理順は元のAdapter順を維持する。
- 動画選択seed、MoCoのモデルseed、Linear Probe seedを区別し、選択した動画集合の同一性をhashで検証する。
- **縮小版の評価Gate：** training/validation双方の200クラスcoverage、重複なし、manifest再生成一致などをproduction候補に要求する。1,000本profileがこれを実際に満たすかは未確認。
- launcherはfresh（新規MoCo）、resume（動画境界から再開）、skip（完了済みMoCoを利用して評価）を区別する。
- Resume checkpointは100動画ごと＋最終、評価用Query LoRA snapshotは1,000動画ごと＋最終。Cometへ学習metrics・実験条件・成果物の対応関係を記録する。
- **Authority上の注意：** 2026-10-07の縮小実験addendumはResearch Workspace上ではdraftだが、GitHubのdevには縮小profileとlauncher引数が存在する。コードへの実装と、科学条件の正式承認・実データでの検証を混同しない。

## 6. Evidence：確認されたこと／未確認のこと

| 項目 | 状態・根拠 |
|---|---|
| Stage 6A→6Bの共通MoCo処理 | 2026-09-28の記録で既存167件＋新規75件のCPUテスト成功。Stage 6Aの人工10-step数値一致、Stage 6Bの人工データでのViT/PEFT接続を確認 |
| Hydra設定一元化 | 2026-10-06記録：全563件のうち527件成功・36件失敗。36件は基準コードでも再現した環境依存失敗。関連する設定合成・テストを確認 |
| 現行pipeline | devコードでMoCo→manifest→Base/LoRA特徴抽出→2条件×3 seed Probe→aggregate、Cometの導線を確認 |
| 動画数縮小 | devコードにsource selectionの設定・検証処理とlauncher引数あり |
| 実ActivityNetでの最終研究結果 | **今回GitHubからは未確認**：最新版でのfull/reduced完走、200-class Gate PASS、Top-1/Macroの実測比較 |
| 時間情報の獲得 | **未検証**。MoCoの損失やLoRA更新の検査だけでは根拠にならない |

## 7. Interpretation：現状の解釈と限界

1. 現在のベースラインは**frame-wise ViT＋late fusion（masked mean）**であり、chunk内のframe順序を直接表現しない。そのため「動画を時間順に入力すること」と「モデルが時間順序を理解すること」を区別する必要がある。
2. 同じ動画の過去chunkをNegativeにすることで、時間的に近い似た場面を意図せず遠ざける**false negative**の可能性がある。
3. Baseを上回るProbe結果が得られても、静的特徴が改善した可能性は残る。時間方向の学習効果を主張するには、更新順を制御した対照実験などが追加で必要。
4. 縮小版で確認できた結果を全10,024本のfull結果として扱わず、選択動画数・クラスcoverage・MoCo seed・selection seedを明示して比較する。

## 8. 次の進め方の候補（MTGで決めたい）

1. **まず一次評価を成立させる：** 縮小設定のmanifest Gateとend-to-end実行を確認し、Base対LoRAの3-seed Probe結果を取得。Gateに失敗した場合は条件を変更したことを明示して再設計する。
2. **時間方向の対照実験：** 同じ動画・同じMoCo条件で「動画内chunkの提示／更新順のみ」を変更して比較する。変える単位（chunk順／動画間順／frame順）を区別し、最初の比較を一つに絞る。
3. **MoCo負例の検証：** Queue内Negativeとの類似度、時間差、近接chunkの扱いを観測し、現行all_pastの妥当性を検討する。
4. **LoRAの中身の解析：** 層別の更新量、特異値、静止／動作区間での反応差を評価設計へ追加する。

**MTGで得たい判断：** ①Linear Probeを一次評価として先行するか、②時間順序を評価する最初のcontrol条件、③縮小実験をどのGateで本実験に昇格させるか。

## 9. 参照資料（実装・仕様・研究記録）

- [研究プロジェクトREADME](../README.md)
- [2026-09-24 MTG議事録](../meetings/2026-09-24-mtg.md)
- [Full-dataset Streaming MoCo + Linear Probe approved spec](../specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md)
- [2026-10-07 縮小版addendum（draft）](../specs/2026-10-07-configurable-reduced-video-full-pipeline-spec.md)
- [Stage 6B共有MoCo検証（2026-09-28）](../experiments/2026-09-28-shared-streaming-moco-verification.md)
- [Hydra一元化検証（2026-10-06）](../experiments/2026-10-06-full-pipeline-hydra-config-verification.md)
- [実装：run_full_pipeline.sh](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/run_full_pipeline.sh)
- [実装：Streaming MoCo更新](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/training/streaming_moco.py)
- [実装：Segment features](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/evaluation/segment_features.py)
- [実装：Linear Probe](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/evaluation/linear_probe.py)

注：これはMTG用の説明・相談資料であり、未計測の精度や実験成功を補って記載していない。図は別ファイルを新規作成せず、資料内のテキスト概念図で示した。
