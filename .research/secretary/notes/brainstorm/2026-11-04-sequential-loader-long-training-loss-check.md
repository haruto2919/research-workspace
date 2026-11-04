---
date: 2026-11-04
project: sequential-video-lora-analysis
source_todo: null
topic: sequential-loader-long-training-loss-check
status: exploratory
tags: [brainstorm, research, sequential-loader, moco, lora, activitynet]
---

# sequential-loader（stride 16）で長めに学習し、lossが下がるかを確かめる

## 出発点

2026-11-04、時間情報の評価（R1）が終わった後で、ユーザーが次のことを確かめたいと考えた。sequential-loaderを使い、stride 16で学習しても、MoCoのlossがきちんと下がるか。10/22のMTGで「公式の設定でも、たくさんのフレームを通さないとlossは下がりきらない」という話が出たので、それと同じくらいの量で学習したい。

## 読み込んだ文脈

- [10/22 MTG](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-10-22-mtg.md)：
  - 当日メモに「20万回画像を通さないとmocoのもとの設定にならない（1k imagenet×200epoch）」とある。
  - 文字起こしでは「20万フレーム」。
  - 議事録の未確定事項では、学習量の数え方は曖昧なので再確認する、とされていた。
  - strideの候補は64／32／16／8。
- 公式MoCo v2：ImageNet 128万枚×200epoch、batch 256（約1,000,000step）。
- 自作と公式の比較（[E5](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-21-meta-moco-v2-e5-imagenet-5epoch-pair-parity-evidence.md)）：ResNet50、5epoch＝25,020step（640万枚）で、epoch平均lossは9.718→7.608と下がった。
- sequential-loader：
  - フレーム数のstrideとmulti-stream DDPは、loader `feature-stride-ddp@85a39c2` に実装済み（[実装記録](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-sequential-loader-frame-stride-multistream-implementation-verification.md)）。研究コードの学習には、まだつないでいない。
  - strideにしても、decodeは全フレームに対して行う。
- ActivityNet trainingは、全部で約326時間の動画（約3,400万フレーム）。stride 16で取ると、1周は約210万フレーム。
- 計算速度：
  - 単一フレームA2 LoRAの計算は、中央値1.48秒/step。
  - clip A2 LoRA（T=4、0.25秒間隔）は、459stepで約82分（約10.7秒/step）。

## ユーザーが採用した方針（2026-11-04、途中）

| 論点 | 採用 |
|---|---|
| 学習量 | **(b-2) フレーム数を公式比較の5epochとそろえる**：256 clip×4フレーム×6,255step＝約640万フレーム（160万clip）。clip LoRAの速度で約19時間、ActivityNet（stride 16）で約3周 |
| 入力の単位 | **複数フレームのclip**（stride 16で取ったフレームを、Early Fusionでまとめて入れる） |
| 学習方式 | **LoRAのみ** |
| batchの組み方 | **(i) 1GPUあたり128本の動画を同時に1 clipずつ進める**（loaderの1 chunk＝1 clip、B=128） |

## 反例・注意点

1. 順番に読むので、Queueに同じ動画の少し前のclipがたくさん入る。それがよく似たnegativeになり、lossが下がりにくい可能性がある。lossが下がらなくても、すぐにバグとは判断できない。ランダム順で学習した既存のrunと比べて読む。
2. stride 16でも、decodeは全フレームに対して行う。データの読み込みが学習に間に合うかは、短い試し実行で測る。

## 未決事項

- 細部はすべて決まった（下の追記）。残りは、spec化の可否と、spec時のコード調査で確かめる実装上の点。

## 次アクション候補（未採用・TODO未追加）

1. 未決事項を決めて、spec化する。
2. 研究コードにsequential-loaderをつなぐ実装と、CPUでの検証。
3. 短いGPU確認（読み込み速度を含む）の後、長い学習を行う（それぞれ別に許可を取る）。

## 追記（2026-11-04）：batchの組み方と学習量の数え方

- batchの組み方は、(i) 128本×1 clip（多様性が高い、Queueの偏りが小さい、loaderをそのまま使える）を採用した。(ii) 32本×4 clipは採用しなかった。
- 学習量は、(b-2) フレーム数をそろえる（6,255step）を採用した。(b-1) step数をそろえる（25,020step、約3日）は採用しなかった。
- 研究コードの制約：Queueの大きさは、global batchで割り切れる必要があり、毎stepのbatchは固定にする必要がある（`moco_vit_ddp_smoke.py` の検査、`concat_all_gather`）。動画の最後で4フレームに満たないclipの扱いを決める必要がある。
- 今の学習率scheduleは「epochごとのcosine（5epochの期間、warmupなし）」である。A2（459step＝1epoch）では、LRは3e-4のまま一定だった。sequential-loaderでの「epoch」の意味（1周）と合わせて決める必要がある。

## 追記（2026-11-04）：細部の決定

| # | 論点 | 採用 |
|---|---|---|
| 1 | clipのフレーム数T | **4枚**（stride 16で、フレーム 0・16・32・48。約1.6秒分） |
| 2 | 動画の最後で4枚に満たないclip | **足りないフレームを、そのclipの最後の有効フレームの繰り返しで埋める**（batchは常に128）。静止に近い入力が混ざることは承知の上で採用。各動画の最後の1 clip（1周で約1万本分、全clipの約2％の見込み）と、4フレーム分の長さがない動画が対象 |
| 3 | 学習率 | **3e-4で一定**（これまでのA2の実際の条件と同じ） |
| 4 | checkpoint | **1,000stepごとと最後（6,255step）に保存し、再開できるようにする** |
| 5 | 比べる基準 | **今回は追加しない**。既存のclip A2（ランダム順、0.25秒間隔）の最初の459stepと見比べ、必要かは結果を見てから決める |
| 6 | 進め方 | **短いGPU確認（10step程度、読み込み速度とメモリ）→ 6,255stepの本番**。それぞれ別に許可を取る |

注意：2で埋めたclipは、最後の数フレームが静止している。この割合は、学習中に記録する案がある（spec時に決める）。

## 追記（2026-11-04）：spec化

draftのspecは次のとおり：[sequential-loader clip MoCo LoRA 6,255step spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md)。blockingのB1（resumeの方法）・B2（augをかける場所）を確認中。

## 追記（2026-11-04）：承認と実装

specを承認し、今のbranchのまま実装した（CPUテストはPASS、未commit、G1・G2は未実施）：[実装記録](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-04-sequential-loader-clip-moco-lora-implementation-cpu-verification.md)
