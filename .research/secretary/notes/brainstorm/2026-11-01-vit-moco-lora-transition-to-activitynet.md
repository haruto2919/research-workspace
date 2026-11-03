---
date: 2026-11-01
project: sequential-video-lora-analysis
source_todo: ViT＋MoCo／LoRAの未解決点を整理し、事前学習を適切に動かせる経路の確認を進める。（2026-10-23 TODO）
topic: vit-moco-lora-transition-to-activitynet
status: exploratory
tags: [brainstorm, research, vit, moco, lora, activitynet, data-domain]
---

# ImageNetでのViT＋MoCo＋LoRA成立後、動画入力（ActivityNet）へ進むか（探索）

## 出発点

2026-11-01、L5（LR 3e-4、1,000step）の結果を受けて、ユーザーが「動画入力（ActivityNet）に進もうと思うがどう思うか」と相談した。

## 読み込んだ文脈

- [L5 Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-01-vit-moco-low-lr-l5-1000step-imagenet-gpu-evidence.md)：両方式とも工学PASS。Queueが埋まった後、lossは単調に減少し、遅れて起きるcollapseはなかった（暫定）。GPU backwardのrun間非決定性を確認した。表現品質は未評価。
- [2026-10-22 MTG](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-10-22-mtg.md) 決定4：「まず事前学習経路の成立を優先し、その後、strideなどの条件によってLoRAがどう変わるかを解析する」。学習量の数え方（「20万回」「ImageNet×200epoch」）は未整理。
- 10/22 MTG資料の手順案：「学習条件と表現評価の進め方を定めた後、ActivityNet単一フレーム→Sequential Loaderへ段階的に再統合する」。
- [10/21 段階計画brainstorm](2026-10-21-full-vit-vs-frozen-vit-lora-moco-before-sequential-loader.md)：画像→P4 動画固定chunk（Loaderなし）→P5 Sequential Loader差分導入→P6 strict online＋研究的評価。一因子ずつ変える方針（2026-10-15 MTG）。
- 研究コード `feature-vit-moco@3874568`：runnerはすでに `data.source=activitynet_frames`（`scripts/vit_moco/activitynet_frames.py` のmanifest）を受け付ける。既存manifestは10秒間隔・offset 5秒で、10,024動画・117,522枚（10/21記録）。この間隔はsmoke用の固定値で、研究用に妥当かは未検証。
- 10/08 MTG：本題の評価は、ActivityNet segment特徴のLinear Probe（元のViT＋LoRAが動作情報を獲得したか）。

## 中心となる問い

ImageNetで「学習が回り、collapseしない」ことは確認できた。この状態で、入力のドメインを動画（ActivityNet）に変えてよいか。変えるなら、何を一度に変えるか。

## 候補

| 案 | 内容 | 評価 |
|---|---|---|
| A | いきなり動画chunk＋Sequential Loaderへ進む | **棄却寄り**。データ・入力形式・順序・状態管理が同時に変わり、失敗時に原因を切り分けられない（10/15方針に反する） |
| B | ActivityNet**単一フレーム**（`activitynet_frames`）に入力だけ差し替え、LoRA・LR 3e-4・その他recipeは固定 | **採用候補（最有力）**。既存runnerで対応済み。変わるのはデータだけで、10/22資料の手順案とも一致する |
| C | 先にImageNetで表現品質（kNN／linear probe）を評価する | **保留**。現在のcheckpointはImageNet分類でfine-tune済み（`google/vit-base-patch16-224`）のため、ImageNetで評価しても「壊していないか」しか分からない。本命の評価（10/08 MTG）はActivityNetなので、評価系はActivityNet側で作る方が研究に直結する |
| D | 先にGPUのゆらぎ（noise floor）を測る | **並行可**。10stepを2回実行するだけで済む。動画移行のblockerではない |

## 案Bで注意すべき点（反例・失敗条件）

1. **データ量とQueueの関係**：117,522枚 ÷ global batch 256 ≈ 1epochあたり約459step。Queue K=65,536はデータ全体の約56%にあたる。
   - 1epoch以内でも、同じ動画の別フレーム（平均約11.7枚/動画、10秒間隔）がQueueに入り、似たフレームがnegativeとして扱われる（false negative）。
   - 459stepを超えると、**同じフレーム自身の前のepochのkey**がQueueに残っている可能性がある。L5と同じ1,000stepは約2.2epochにあたり、これが起きる。
   - 判断が必要：step数を1epoch以内に抑えるか、Kを小さくするか、そのまま実行して挙動を観察するか。
2. **ImageNetとの結果比較は記述のみ**：データが違うので、L5とのbitwise比較やprefix比較は成立しない。比べられるのは推移の形（Queueが埋まった後にlossが減るか、collapseしないか）だけ。
3. **固定probe**：今のprobeはImageNet 64枚。続けて使うと、ImageNet runとの連続性は保てる。ActivityNetのprobeを別に用意するかは未決。
4. **時間情報は扱っていない**：単一フレームなので、この段階で動的情報や時間情報が得られたとは主張できない。動画としての意味が出てくるのはP4（固定chunk）以降。
5. **10秒間隔の妥当性**：smoke用の値のまま。フレーム類似度の診断（5/10/20/30秒の比較）は未実施。

## 現在の方向性（AIの推薦。ユーザー未採用）

- 動画へ進むこと自体には賛成。ただし**案B（ActivityNet単一フレーム、入力だけ差し替え）から始める**。
- 最初は短いrun（10step）で工学チェックを通し、その後、長さ（1epoch以内か、2epoch以上か）を決めて実行する。
- noise floor（案D）は、空いたGPU時間に並行して取る。
- 表現評価は、ActivityNetのsegment linear probeを本命として、P4以降に別specで設計する。

## 未解決事項

- run長とQueue Kの扱い（上記1）。
- 固定probeをImageNetのままにするか、ActivityNetのprobeを追加するか。
- `moco_vit_lr_long` presetのGPUチェックが `activitynet_frames` をどう扱うか（研究コードで未確認）。
- 学習量の数え方（10/22 MTGの宿題）。

## 次アクション候補（TODO未追加）

- 案Bをspec化する（`research-spec`）。
- noise floor run（L4 presetで10stepを2回）。

## 追記（2026-11-01）

ユーザーは、early fusionの設計より先に単一フレームの段階を進めることを選んだ。案Bは[ActivityNet単一フレームspec（draft）](../../../lab/projects/sequential-video-lora-analysis/specs/2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md)に昇格した。C1〜C6は承認待ち。
