---
date: 2026-11-01
project: sequential-video-lora-analysis
source_todo: null
topic: early-fusion-vit-lora-temporal
status: exploratory
tags: [brainstorm, research, vit, lora, moco, early-fusion, late-fusion, temporal, activitynet]
---

# LoRAが時間情報を学習できるよう、late fusionからearly fusionへ変える（探索）

## 出発点

2026-11-01、ActivityNet単一フレーム段階のspec準備中に、ユーザーが次の意向を示した。「現状はlate fusionだと思うが、LoRAが時間情報を学習できるようにするため、early fusionに変更したい」。

## 読み込んだ文脈

- [2026-09-23 late fusionとLoRAの時間特徴](2026-09-23-late-fusion-lora-temporal-learning.md)：問題の本質は、late fusionそのものよりも、集約に順序を区別しない平均（masked mean）を使っていることだと整理した。masked mean経路は、時間を使わないbaselineとして残す案（A）。
- [2026-09-24 ImageNet ViT→VideoMAE初期化](2026-09-24-imagenet-vit-videomae-initialization.md)：tubelet 1（2Dのpatch重みをそのまま時間方向kernel 1へ変換）／tubelet 2（重みを半分ずつ複製）。位置埋め込みは、固定sin-cosの時空間埋め込みにすれば「時間情報が学習可能な埋め込みだけに入った」という交絡を減らせる。
- [2026-09-24 MTG](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-09-24-mtg.md)：画像MAE/ViTの重みをVideoMAEへ移してearly fusionを試す案が出た。late fusionとの「どちらもやる」方向を確認したが、比較条件と優先順位は固定していない。「どちらが適切かは評価で判断する」。
- 現在のViT＋MoCo経路（`feature-vit-moco@3874568`）は**1枚の画像（またはActivityNetの1フレーム）を入力するだけで、fusionはまだない**。late fusion（frame-wise CLS＋masked mean）は、以前のActivityNet Stage 5経路の構成である。
- 2026-10-22 MTG：まず事前学習経路を成立させ、その後strideなどの条件でLoRAがどう変わるかを解析する。

## 整理

### early fusionにする意味

- late fusion（各フレームを別々にViTへ通し、最後に平均）では、LoRAが付いたattentionは常に1フレームの中しか見ない。LoRAがフレーム間の関係を表現する場所がない。
- early fusion（複数フレームのtokenを1つの系列にしてViTへ入れる）なら、Q/VのLoRAがフレーム間のattentionを直接変えられる。「LoRAに時間情報を持たせる」という目的に、構造が合っている。
- late fusionでも、後ろに順序を区別する集約器（temporal Transformerなど）を置けば順序を扱える。ただし、その場合に時間情報が入るのは集約器で、LoRAではない。本研究の目的とはずれる。

### early fusionにしただけでは足りない条件（反例・失敗条件）

1. **時間方向の位置情報**：同じ空間位置埋め込みを全フレームに付けただけだと、attentionはフレームの並び順を区別できない。並べ替えても出力は変わらず、順序不変のままになる。時間方向の位置埋め込みが必要。固定sin-cosにすれば学習可能なパラメータが増えず、「時間を学んだのはLoRA」と言いやすい。
2. **学習目標が時間情報を必要とするか**：MoCoのpositiveが「同じclipの2つの拡張」だけだと、見た目（背景・物体）だけで解けてしまう（静的情報への近道）。early fusionにしても、LoRAは静的情報しか学ばない可能性がある。時間に敏感なpositive/negativeの設計（例：時間をシャッフル・逆再生したclipをnegativeにする、時間をずらしたclipの扱い）が別途必要。**これが最も結果を左右する論点**。
3. **計算量**：ViT-B/16で16フレーム・tubelet 1にすると、token数は3,136（単一画像の16倍）になる。attentionの計算量は約256倍、MLPは16倍。今の1GPUあたり128枚（LoRAで約12GB）は不可能で、batchを大きく減らすか、フレーム数を減らす（4/8）、tubelet 2、解像度を下げる、gradient checkpointingを使う、などが必要。
4. **事前学習重みとのずれ**：画像ViTはフレーム間のattentionを見たことがない。学習前（step0）の時点で、単一画像のときとは出力が変わる。T=1（1フレーム）なら単一フレーム経路と一致するように作ると、前の段階からの連続性を確認できる。
5. **評価**：時間情報を得たと言うには、順序に敏感な評価（シャッフル・逆再生・同じフレームの繰り返しとの比較）が必要。lossの低下だけでは言えない。

## 候補

| 案 | 内容 | 評価 |
|---|---|---|
| E1 | 時空間joint attention、tubelet 1（2Dのpatch重みをそのまま使う）、固定sin-cosの時間位置埋め込み、Q/V LoRA | **採用候補（最有力）**。重みの変換が最小で、T=1なら今の経路と一致させやすい。計算量が最大の課題 |
| E2 | E1をtubelet 2にする（patch重みを半分ずつ複製） | 候補。token数が半分になる。複製の方法が研究上の追加の設計判断になる |
| E3 | 空間と時間のattentionを分ける（TimeSformer型）。時間attentionを新しく追加 | 保留。学習可能なモジュールが新しく増えるので「LoRAが時間を学んだ」と言いにくい。時間attentionに固定した空間重みとLoRAを流用する変形ならあり得る |
| E4 | late fusion＋順序を区別する集約器 | 棄却寄り（主経路としては）。時間情報が集約器に入る |
| E5 | masked meanのlate fusion | **対照として残す**（時間を使わないbaseline、9/23のA） |
| E6 | 動画で事前学習済みのbackbone（VideoMAEのcheckpointやMeMViT） | 棄却寄り。「画像で事前学習したモデルにLoRAで時間情報を足す」という研究の前提と合わない |

## ActivityNet単一フレーム段階との関係

- 単一フレームはT=1の場合なので、fusionの種類とは関係しない。単一フレームspecのC1〜C6は、early fusionを採用しても無駄にならない（データのドメイン変更だけを切り分ける段階として残る）。
- early fusionは、その次の「動画clip」段階（以前の段階計画のP4）の設計を置き換える。

## 現在の方向性（AIの推薦。ユーザー未採用）

- early fusionへの変更に賛成。主経路はE1を推奨し、E5（masked mean）を対照として残す。
- 進め方：単一フレーム段階（既存コードで実行可能）を先に行い、並行してearly fusion clip段階の設計を詰める。
- 9/24 MTGでは「どちらが適切かは評価で判断」としていたため、主経路をearly fusionに切り替えることは次回MTGで先生に共有する。

## early fusion clip段階のspec前に決めること（未決）

1. フレーム数Tと、フレーム間隔（stride。10/22 MTGの64/32/16/8候補との対応）
2. tubeletのサイズ（1か2）
3. 時間位置埋め込み（固定sin-cosか、学習可能か）
4. MoCoのpositive/negativeを、時間に敏感にするかどうか、どう設計するか
5. 計算資源：batch、解像度、gradient checkpointing、Queueの大きさ
6. clipの表現：CLSか、tokenの平均か
7. 時間情報の評価方法（シャッフル・逆再生・静止の対照）
8. late fusion対照（E5）をどこまで実装・実行するか

## 関連ファイル

- [ActivityNet移行の壁打ち](2026-11-01-vit-moco-lora-transition-to-activitynet.md)
- [L5 Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-01-vit-moco-low-lr-l5-1000step-imagenet-gpu-evidence.md)
