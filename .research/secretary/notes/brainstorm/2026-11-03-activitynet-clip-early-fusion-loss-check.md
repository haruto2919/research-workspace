---
date: 2026-11-03
project: sequential-video-lora-analysis
source_todo: null
topic: activitynet-clip-early-fusion-loss-check
status: exploratory
tags: [brainstorm, research, vit, lora, moco, early-fusion, activitynet, clip]
---

# ActivityNetの普通の動画clip入力（Early Fusion）で学習できるか・lossが下がるかの確認（探索）

## 出発点

2026-11-03、ActivityNet単一フレームのA1・A2が両方式で工学PASSした後、ユーザーが次の段階として次の意向を示した。

- 時間情報の獲得は目的にしない。
- ActivityNetを使い、**逐次（シーケンシャル）ではない普通の動画clip入力**で学習できるか、**lossがちゃんと下がるか**を確認したい。
- 複数フレームは**Early Fusionで結合してからViTへ渡す**想定。

## 読み込んだ文脈

- [README](../../../lab/projects/sequential-video-lora-analysis/README.md)：A2（LoRA・Full FT、各459step）は工学PASSで、Queueが埋まった後のlossは両方式で単調に減少し、collapseは観測されなかった（暫定）。データ読み込みの待ちは、LoRAで約2.3秒/step、Full FTで約0.7秒/stepだった。
- [A2 LoRA Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-vit-moco-lora-activitynet-a2-lora-gpu-evidence.md)、[A2 Full FT Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-vit-moco-lora-activitynet-a2-full-ft-gpu-evidence.md)：VRAMのpeak reservedは、LoRAで約12.3 GiB/rank、Full FTで約18.3 GiB/rank（GPU1のheadroomは1,750 MiB）。いずれもbatch 128/rank、197 token。
- [11/01 early fusionの壁打ち](2026-11-01-early-fusion-vit-lora-temporal.md)：token結合・tubelet 1・固定sin-cosの時間位置埋め込み・Q/V LoRA（E1）を最有力とした。spec前の未決事項を8つ挙げた。
- [10/22 MTG](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-10-22-mtg.md)：まず事前学習経路を成立させ、その後strideの候補（64/32/16/8、未確定）でLoRAの変化を調べる。
- 研究コード（`feature-vit-moco@9155c2b`、read-only）：`ViTMoCoBackbone` は `ViTModel` に画像1枚を渡し、CLSを返す。`ActivityNetFrameDataset` は1件＝1フレーム（manifestの10秒間隔の時刻、117,522件）。clipやearly fusionの経路はまだない。

## 問い

ActivityNetのT枚のフレームからなるclipを、token結合のearly fusionでViTへ入れたとき、ViT＋MoCo（LoRA／Full FT、LR 3e-4）は工学的に問題なく学習でき、Queueが埋まった後にlossが下がるか。

- 時間情報を獲得したかは問わない（評価もしない）。
- この確認は、後で時間情報を扱う段階（LoRAにフレーム間の関係を持たせる）と同じ経路で行い、作業を無駄にしない。

## ユーザーが採用した方針（2026-11-03）

| 論点 | 採用 | 理由・補足 |
|---|---|---|
| Early Fusionの形 | **token結合（F2）**。各フレームを別々にpatch化し、T×196個のtokenを1つの系列にする（tubelet 1）。時間位置埋め込みを足す | LoRAが付いたattentionがフレーム間を直接見られ、後の時間情報の段階に直結する |
| MoCoのpositive | **同じclipに2通りの拡張**。拡張（crop・色）はclip内の全フレームで同じものを使う | 時間を問わない今回の目的に最も単純 |
| フレーム数T | **T=4** | 約785 token。メモリとdecodeの負荷が比較的軽い |

## 比較した案

| 案 | 内容 | 判断 |
|---|---|---|
| F1 チャンネル結合 | patch埋め込みのカーネルを時間方向T枚分に広げる（tubelet = T）。token数は196のまま | 棄却寄り。計算は単一画像並みで手軽だが、LoRA方式ではpatch埋め込みが固定なので、フレームは固定された重みで混ざるだけ。後の時間段階でLoRAに時間情報を持たせる経路にならない。**再検討条件**：token結合でVRAMやデータ読み込みが成立しない場合の代替 |
| **F2 token結合（tubelet 1）** | 上記 | **採用** |
| F3 token結合（tubelet 2） | 2フレームずつpatch化 | 保留。T=8以上にしたいときの候補。重みの複製方法が、研究上の新しい設計判断として増える |
| 同じ動画の別clipをpositiveにする | CVRL・ρBYOL型 | 保留。時間をまたいだ不変性を学ぶ設計で、今回の目的から一歩踏み込む |
| T=8／T=16 | — | 保留。T=4で成立を確認した後の候補 |

## 反例・失敗条件

1. **計算量**：785 tokenでは、attentionが約16倍、ほかの計算が約4倍になる。batch 128/rankのままでは、LoRAでもVRAMが足りない可能性が高い（未測定）。batchを下げるとMoCoの条件（global batch、Queueが埋まるstep）が単一フレームA2から変わり、lossを比べにくくなる。
2. **データ読み込み**：単一フレームでも、待ちが最大2.3秒/stepあった。clipではdecodeするフレームが増える。keyframeからの前向きdecodeでclipの範囲を読むので、単純にT倍ではないが、さらに遅くなる可能性が高い。
3. **事前学習重みとのずれ**：画像ViTはフレーム間のattentionを見たことがない。step0の出力は、単一画像のときと変わる。T=1で単一フレーム経路と一致するように作れば、前の段階からの連続性を確認できる。
4. **時間位置埋め込みとT=1一致**：固定sin-cosは、t=0で0にならない（cos成分が1）。T=1一致を工学チェックに使うなら、t=0がゼロになる形にするなどの工夫が要る。
5. **lossの比較**：clipとフレームではタスクの難しさが違う。lossの値そのものを単一フレームA2と比べても、優劣は言えない。言えるのは、同じrunの中で「Queueが埋まった後に下がったか」である。
6. **「時間を問わない」の確認**：MoCoのpositiveが同じclipの2拡張なので、モデルはフレームを平均するだけでも解ける。lossが下がっても、フレーム間の関係を使った証拠にはならない（今回は問わない）。

## 未決事項（spec前に決めること）とAIの推薦

| # | 論点 | 推薦（ユーザー未採用） |
|---|---|---|
| U1 | clipの取り方 | 既存manifestの117,522個の時刻（10秒間隔）を、それぞれclipの起点にする。sample数・459step/epochが単一フレームA2と同じになる。範囲が動画の末尾を越えるclipの扱い（除外か、前にずらすか）も決める |
| U2 | フレーム間隔（stride） | 時間（秒）ではなくフレーム数で数えるか、10/22 MTGの候補（64/32/16/8）の単位とどう揃えるかを確認する。fpsは動画ごとに違う（25/30など） |
| U3 | 時間位置埋め込み | 固定sin-cos（学習しない）で、t=0がゼロになる形。T=1で単一フレーム経路と一致する |
| U4 | batchとメモリ | gradient checkpointingでglobal batch 256・K 65,536を保てるかを、短いsmokeでVRAMを測ってから決める。保てない場合は、batchを下げることと、それに伴う条件の変化を明記する |
| U5 | データ読み込み | まずは今のon-the-fly decodeのまま、A1（10step）で時間を測る。遅すぎれば、clipを縮小してローカルディスクへ書き出す案を別に検討する |
| U6 | clipの表現 | CLS（今の経路と同じ）。tokenの平均は使わない |
| U7 | 固定probe | ImageNetの64枚をT=4回繰り返した「静止clip」にする案。L4/L5と同じ画像を使えるが、入力の形が違うので、probe入力ハッシュはL4/L5と一致しなくなる |
| U8 | 実行順序と成功条件 | 単一フレーム段階と同じ構成：A1（10step、LoRA→Full FT）→A2（459step）。工学的成功（completed・finite・rank一致・VRAM PASS・T=1一致）と、記述的な観測（256step以降の窓ごとのlossが下がるか）を分ける |
| U9 | Full FTのVRAM | 197 tokenでも、GPU1のheadroomは1,750 MiBしかなかった。785 tokenでは、checkpointingがあっても足りない可能性がある。LoRAを先に行い、Full FTは測定してから判断する |

## 次アクション候補（未採用・TODO未追加）

1. U1〜U9を決め、spec化する（`research-spec`）。
2. spec承認後に実装し、CPUで検証する。その後、GPU A1（LoRA）でVRAMと読み込み時間を測る。

## 関連ファイル

- [11/01 early fusionの壁打ち](2026-11-01-early-fusion-vit-lora-temporal.md)
- [11/01 ActivityNet移行の壁打ち](2026-11-01-vit-moco-lora-transition-to-activitynet.md)
- [ActivityNet単一フレームspec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md)
- [A2 LoRA Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-vit-moco-lora-activitynet-a2-lora-gpu-evidence.md)
- [A2 Full FT Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-vit-moco-lora-activitynet-a2-full-ft-gpu-evidence.md)

## 追記（2026-11-03）：spec化

ユーザーがU1〜U9の推薦案を採用し、フレーム間隔0.25秒とclipの位置（manifestの時刻＝最後のフレーム）を選んだ。承認済みspec：[ActivityNet clip Early Fusion spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-11-03-vit-moco-lora-activitynet-clip-early-fusion-spec.md)。
