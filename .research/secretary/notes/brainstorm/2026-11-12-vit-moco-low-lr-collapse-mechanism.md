---
date: 2026-11-12
project: sequential-video-lora-analysis
source_todo: null
topic: vit-moco-low-lr-collapse-mechanism
status: exploratory
tags: [brainstorm, research, moco, lora, learning-rate, representation-collapse]
---

# ViT＋MoCo：学習率を下げると特徴方向の集中が緩和された理由

## 出発点と問い

ユーザー質問：「学習率0.03では固定64画像の特徴が1方向に集中したが、3e-4ではその現象が消えた。なぜ解決するのか？」

本メモは既存実験Evidenceを用いた機構の教育的説明と未検証点の整理であり、新たな実験結果・確定的な因果証明ではない。

## 関連Evidence（ユーザー報告およびResearch Workspaceの記録）

- D4（2026-10-22、LR 0.03）：固定64画像・全2,016ペアのcos中央値で、更新1回後のProjector生出力 z_q は LoRA 0.99317、Full FT 0.99994。CLSは LoRA 0.03192、Full FT 0.98495。初回Projector更新normは約24.90、更新前パラメータnorm比は約1.44。LoRAのz_q row norm中央値は2.266から153.14に増大。
- L4（2026-10-31、LR 3e-4、各10step）：更新1回後のz_q cos中央値は LoRA 0.453、Full FT 0.495。Full FTのCLS cosは0.038。初回勾配normは旧LRと一致、Projector更新norm／相対更新は約1/100（相対更新約0.0144）。ただし旧D4とL4ではGPU0の機種が変わり、step0のbitwise一致は確認されずcross-LR比較器は comparable=false。
- L5（2026-11-01、LR 3e-4、各1,000step）：z_q cos中央値はstep1000で LoRA 0.0046、Full FT 0.0004。CLSは約0.027／0.032。qの平均次元stdは0.0874／0.0876。Queueが実keyで満たされた後、lossの100step移動窓平均は減少。L4とのprefix比較はGPU backwardのrun間非決定性と整合するbitwise不一致であり、科学的解釈は暫定。
- 64枚probeは固定入力の**方向集中**を測定するもので、表現品質・時間情報の獲得そのものを検証していない。

## もっとも整合する機構の説明（解釈）

1. AdamWの更新は学習率に概ね比例する。初回stepではモーメントがゼロから始まるため、epsilonが小さくgradientが非ゼロな成分では、更新の絶対値がパラメータごとに約LRとなる（decoupled weight decayは別項）。単純化すると \(\Delta\theta_i \simeq -\eta\,\mathrm{sign}(g_i)\)。
2. Projectorは約689,024パラメータを持つ。多数の成分を一斉に約0.03動かすと、全体更新のL2ノルムが \(0.03\sqrt{689024}\approx24.9\) になる。これは更新前Projector normの約1.44倍。0.0003では約0.249、相対約0.0144に留まる。したがって**「勾配方向が同じでも1stepのジャンプ幅が大きく異なる」**。
3. 大きな共同の出力成分が各画像の差を支配すると、\(z_i = c + \delta_i\) （\(\|c\| \gg \|\delta_i\|\)）となり、異画像のcos類似度が1に近づく。これは「1方向に集中する」現象の**説明用モデル**であり、Projectorのバイアス項が原因と特定したわけではない。
4. LoRAでは初期CLSのcosを保ったままProjector出力が急激に同方向化したため、特にProjector段の大きな更新と整合する。Full FTではCLSも同時に方向集中し、Backboneの大幅更新も候補となる。L2正規化前のz_qですでに現象が生じているため、正規化そのものが主原因とは言いにくい。
5. 3e-4では初回の特徴空間破壊が抑えられ、その後の学習でz_qのcos中央値が低下した。ただし「学習率を下げると必ず識別的表現が得られる」ことは意味しない。

## 比較・反例・未解決事項

- **最有力な説明**：AdamWのLR 0.03による過大な初回更新が強い方向集中に寄与。とりわけProjector更新が初期重みスケールに対して大きい。
- **未分離**：Projector更新、LoRA B更新、Full FTのViT更新、EMA Key、Queue、InfoNCEの相互作用のどれが必要条件・十分条件なのか。
- **比較制限**：LR以外の学習条件を揃えても、cross-LRの旧／新runはGPU構成差がありbitwise比較Gate未達。同一GPUのrunでもbackwardの完全決定性は未達。単一seed、0.03は10step／3e-4は1000stepなので長期比較に非対称性がある。
- **反例**：低LRでcosが低くても、ランダムで情報価値のない特徴かもしれない。ゼロ付近のcosやloss減少だけで有用性を証明しない。
- **次に検証可能な候補（未承認）**：同一GPU環境・複数seedでLR sweep、Backbone／ProjectorのLR分離または一方のみ固定するablation、特徴のkNN・linear probe、固定probeでuniformity・positive/negative・ノルムを同時評価。どれも本メモでは実行・spec化しない。

## 現在の方向性

LR 3e-4はこの設定で初期方向集中を抑え、1,000stepまで同様の強い方向集中が再発しなかったという暫定Evidenceがある。原因については「過大な初回更新」と説明するのが最もデータに整合するが、最適LR・一般的な安定性・表現品質を断定しない。

## 関連ファイル

- `.research/lab/projects/sequential-video-lora-analysis/README.md`
- `.research/lab/projects/sequential-video-lora-analysis/experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md`
- `.research/lab/projects/sequential-video-lora-analysis/experiments/2026-10-31-vit-moco-low-lr-l4-imagenet-gpu-evidence.md`
- `.research/lab/projects/sequential-video-lora-analysis/experiments/2026-11-01-vit-moco-low-lr-l5-1000step-imagenet-gpu-evidence.md`
