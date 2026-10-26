---
date: 2026-10-26
project: sequential-video-lora-analysis
source_todo: null
topic: static-dynamic-disentanglement-needs-survey
status: exploratory
tags: [brainstorm, research, background, literature-survey, static-dynamic-information, disentanglement, lora, presentation]
---

# 研究背景の壁打ち：「動的情報を分離したい」というニーズの先行研究調査

## 出発点

- ユーザーの研究背景の流れ（2026-10-26 ユーザー発言）：**「いろんな研究で動的情報を分離したいというニーズがある → 自分の手法でそれをやる」**。
- 前ターンでAIが示した3本の動機（A: オンライン学習，B: 静的バイアス／動的情報，C: LoRA解析）のうち，ユーザーは「分離のニーズ」を背景の主軸に選んだ。
- 依頼：そのニーズを示す研究を探す。
- 直近の文脈：中間発表は2026-10-30。10/21の発表ストーリー案は「オンライン学習 → 静的バイアス → Frozen ViT＋LoRA → MoCo」の順。動画でのEvidenceはまだない（READMEは2026-10-22まで，ViT＋MoCoのcollapse診断の段階）。

## 読み込んだ文脈

- `.research/lab/projects/sequential-video-lora-analysis/README.md`（2026-10-22まで）
- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-10-15-mtg.md`
- `.research/lab/projects/sequential-video-lora-analysis/materials/2026-10-21-online-video-lora-contrastive-research-presentation-mtg-materials.md`
- `.research/secretary/notes/brainstorm/2026-10-19-research-presentation-related-work-gap-survey.md`
- `.research/secretary/notes/brainstorm/2026-10-21-static-dynamic-information-goyal-choi-kowal-review.md`
- `.research/secretary/notes/brainstorm/2026-10-08-continual-lora-paper-relevance.md`
- 日次TODO：2026-10-26は未作成（最新は2026-09-25）。

## 調査方法と確認の範囲

- 2026-10-26にWeb検索で，各論文の公式ページ・arXiv・会議ページを確認した。
- 確認できたのは，abstractと公式要約の範囲の主張・手法の要点。本文中の数値や表は今回確認していないため，数値は書いていない。
- 既存メモで扱い済みの論文（Goyal 2017，Choi 2019，Kowal 2022，StillMix 2023，ST-Adapter，MotionMAEなど）は，ここでは再掲を最小限にした。

## 調査結果：分野ごとの「分離したい」ニーズ

### 1. 動画生成・編集：見た目と動きを別々に制御したい

| 論文 | 何を分離したか | 要点 |
|---|---|---|
| MCnet (Villegas+, ICLR 2017) https://arxiv.org/abs/1706.08033 | motion / content | 動きと内容を別のエンコーダで扱い，次フレームを予測する |
| DRNet (Denton & Birodkar, NeurIPS 2017) https://proceedings.neurips.cc/paper/2017/hash/2d2ca7eedf739ef4c3800713ec482e1a-Abstract.html | content（clip内で不変）/ pose（時間変化） | 敵対的lossで，pose特徴にclipの同一性が含まれないようにする |
| MoCoGAN (Tulyakov+, CVPR 2018) https://openaccess.thecvf.com/content_cvpr_2018/html/Tulyakov_MoCoGAN_Decomposing_Motion_CVPR_2018_paper.html | content / motion の潜在変数 | 内容を固定して動きだけ変える，またはその逆の生成ができる |
| Disentangled Sequential AE (Li & Mandt, ICML 2018) https://arxiv.org/abs/1803.02991 | static / dynamic の潜在変数 | 系列VAEで時間不変な成分と時間変化する成分を分ける |
| **MotionDirector** (Zhao+, ECCV 2024) https://arxiv.org/abs/2310.08465 | **空間LoRA（見た目）/ 時間LoRA（動き）** | dual-path LoRAとappearance-debiased temporal loss。普通のLoRA tuningでは動きが見た目と結びついてしまう問題に対処 |
| **Customize-A-Video** (Ren+, ECCV 2024) https://arxiv.org/abs/2402.14780 | **時間attentionのLoRA（動き）/ appearance absorber（見た目）** | 見た目を吸収するモジュールは**順序を崩したフレームで学習**する。空間層にLoRAを置くと配置を記憶し，動きの学習が悪化すると報告 |
| DreamVideo (Wei+, CVPR 2024) https://arxiv.org/abs/2312.04433 | identity adapter / motion adapter | 被写体と動きを別々に学習してから組み合わせる |
| Still-Moving (Chefer+, TOG 2024) https://arxiv.org/abs/2407.08674 | spatial adapter / motion adapter | 静止した「frozen video」で見た目だけを学習し，動きの事前知識は保つ |
| **Dynamic Concepts Personalization / Set-and-Sequence** (Abdal+, SIGGRAPH 2025) https://arxiv.org/abs/2502.14844 | **順序なしフレームで学習した見た目LoRA basis / 系列で学習した動きの成分** | フレームを1枚ずつ入力すると時間の信号がない。だからそのLoRAは見た目の基底になる，という設計 |
| Separate Motion from Appearance (Liu+, arXiv 2025) https://arxiv.org/abs/2501.16714 | 時間attentionの純化 | MotionDirectorなどでも時間側に見た目が漏れる，と指摘 |
| VideoJAM (Chefer+, ICML 2025) https://arxiv.org/abs/2502.02492 | 見た目と動きの同時表現 | ピクセル再構成の目的関数は，動きより見た目を優先させてしまうと主張 |

### 2. 動画認識・自己教師あり表現学習：静的なショートカットを除き，動きで判断させたい

| 論文 | 要点 |
|---|---|
| Two-Stream (Simonyan & Zisserman, NeurIPS 2014) https://arxiv.org/abs/1406.2199 | 静止フレーム（見た目）とoptical flow（動き）を別のstreamで扱う古典 |
| Choi+ (NeurIPS 2019) | scene biasの緩和（既存メモ参照） |
| DSM (Wang+, AAAI 2021) https://arxiv.org/abs/2009.05757 | 自己教師あり学習で，動きを保って背景を乱したpositiveと，動きを壊したnegativeを使う。sceneとmotionを切り離す |
| FAME (Ding+, CVPR 2022) https://arxiv.org/abs/2109.15130 | 前景を別の背景へ貼り付け，contrastive学習に動きを使わせる。**MoCoベース**の事前学習 |
| Static and Dynamic Concepts (Qian+, ECCV 2022) https://arxiv.org/abs/2207.12795 | 単一フレームを静的概念，フレーム差分を動的概念として分けて学習する |
| StillMix (Li+, ICCV 2023) | 静的バイアスの緩和と評価（既存メモ参照） |

### 3. 動画×言語・VLM：1フレームで解けてしまい，時間を理解していない

| 論文 | 要点 |
|---|---|
| Revealing Single Frame Bias (Lei+, ACL 2023) https://aclanthology.org/2023.acl-long.29 | 単一フレームで学習したモデルが複数フレームの手法に勝つことがあり，データに静的な見た目のバイアスがあると指摘 |
| Test of Time (Bagad+, CVPR 2023) https://arxiv.org/abs/2301.02074 | 既存の動画言語モデルは前後関係（before/after）すら苦手。時間順序の後学習TACTを提案 |
| TempCompass (Liu+, ACL Findings 2024) https://arxiv.org/abs/2403.00476 | 静的な内容が同じで時間的側面だけ違う動画で評価。Video LLMの時間知覚は弱く，Image LLMを上回らないと報告 |

### 4. モデル内部の解析：静的情報と動的情報を定量化したい

- Kowal+ (CVPR 2022) と拡張版 "Quantifying and Learning Static vs. Dynamic Information in Deep Spatiotemporal Networks" (arXiv 2211.01783)：多くの時空間モデルは静的情報に偏る。channelは静的・動的・両方に分かれる（既存メモ参照。拡張版の掲載先は今回未確認）。

### 5. 応用領域：動きの成分だけを使いたい

| 領域 | 論文 | 分離の目的 |
|---|---|---|
| 歩容認識 | GaitNet (Zhang+, CVPR 2019) https://arxiv.org/abs/1904.04925 | 服装などの見た目とposeを分け，pose系列で個人を識別する |
| 顔の偽造検出 | FTCN (Zheng+, ICCV 2021) https://arxiv.org/abs/2108.06693 | 空間カーネルを1にして時間的な一貫性だけを見る。汎化のため |
| 異常検知 | AMMC-Net (Cai+, AAAI 2021) https://ojs.aaai.org/index.php/AAAI/article/view/16177 | 見た目と動きの対応関係を，正常パターンとして学習する |
| 3D・シーン再構成 | D²NeRF (Wu+, NeurIPS 2022) https://arxiv.org/abs/2205.15838 | 静的背景と動く物体を2つの場に分ける。動的な側が静的な側を吸収してしまう問題を報告 |
| 世界モデル・ロボット | Genie (Bruce+, ICML 2024) https://arxiv.org/abs/2402.15391 | ラベルなし動画から，フレーム間の「潜在行動」を取り出す |

## 解釈：背景のストーリーに対して何が言えるか

### 強く言えること

- **「動的情報を分離したい」というニーズは，生成・認識・VLM・解析・応用の5つの領域で独立に確認できる。** 背景の1段目「いろんな研究で分離したいニーズがある」は，十分な文献で支えられる。
- **生成分野では，LoRAやadapterで見た目と動きを分けるやり方がすでに主流。** LoRAが動きの情報の容れ物になり得ることの傍証になる。

### 新規性への脅威

- 「LoRAで静的情報と動的情報を分ける」というアイデア自体は，生成分野ですでに提案されている（MotionDirector，Customize-A-Video，Set-and-Sequence）。そのため，**「LoRAを使って分離する」こと自体は新規性にならない**。
- 本研究との差分になりうる候補（まだ確定していない）：
  - 生成ではなく認識・表現の側である
  - 時間層を持たない画像事前学習モデルを使う
  - 自己教師ありで学習する
  - 逐次ストリームで学習する
  - 分離を設計で強制するのではなく，「何が加わったか」を測る
  - 個別動画のカスタマイズではなく，大規模動画からの汎用的な適応である

### 最も重要な反例：現在の構成は，生成分野の知見では「見た目側」のLoRAに当たる

- Customize-A-VideoとSet-and-Sequenceは，**フレーム単位（順序なし）で学習したLoRAを，見た目を吸収・表現する側として使っている。** 動きは，時間attentionに置いたLoRAや系列での学習で取っている。
- 現在の構成は次のとおり。
  - frame-wise ViTの空間attentionのQ/VにLoRAを付けている
  - 集約はmasked mean（順序に依存しない）
  - positiveは同じclipの色入替＋反転
- この構成では，1回のforwardの中で**LoRAが計算できるのは1フレームから得られる関数だけ**になる。Kowalの定義（static = 1フレームから抽出できる情報）に従うと，LoRAが表現できる情報は定義上staticになる。
  - 逐次学習によって重みが入力順に依存することはありうる。また，1フレームから動きを示唆する手がかり（姿勢やmotion blurなど，いわば「暗示的な動き」）に敏感になることもありうる。ただし，どちらも複数フレームから初めて分かる動的情報ではない。
- したがって，背景を「分離のニーズ → 自分の手法で分離する」とする場合，**今のままでは手法が分離を実現する構造になっていない**と指摘される可能性が高い。

### 選択肢（候補であり，採用は未決）

| 案 | 内容 | 長所 | 短所・リスク |
|---|---|---|---|
| (i) 時間的な相互作用を入れる | temporal attention / temporal tokenにLoRAを置く，またはpositive・negativeを時間に基づいて作る（DSMやFAME型） | 「分離する手法」として筋が通る | 構成の変更が大きい。今の段階的検証の計画とずれる |
| (ii) 2つのLoRAで分離を設計する | フレーム単位のLoRA（静的）と時間側のLoRA（動的）を分ける（Set-and-Sequence / MotionDirector型を認識側へ持ち込む） | 生成分野と対応がとれ，比較しやすい | 新規性の差分を，認識側・自己教師あり・ストリームに置く必要がある |
| (iii) 主張を「測る」側に寄せる | 「画像モデル＋逐次動画の自己教師あり適応で，LoRAに何が加わるか（静的か動的か）を測る」 | 現在の構成のまま進められる。「静的しか入らない」も結果になる | 背景の「自分の手法で分離する」とは主張の強さが変わる |

## 未解決の問い

- 本研究でいう「分離」とは何か。
  - (a) Base＝静的，LoRA＝動的差分という役割分担
  - (b) 静的用と動的用の2つのパラメータ群に分けること
  - (c) 表現空間で静的成分と動的成分が分かれること
- 中間発表（10/30）では，背景の「分離のニーズ」と，手法の「現在の構成」の間のずれをどう扱うか。
- 認識側で「LoRAによる静的・動的の分離」を直接扱った先行研究は，今回の検索では見つからなかった。ただし網羅的な調査ではないので，「先行研究がない」とは主張しない。

## 次アクション候補（未採用，TODOには追加していない）

- 「分離」の定義を(a)〜(c)から選び，それに合わせて手法の選択肢(i)〜(iii)を比べる壁打ち。
- 中間発表の背景スライドを「5領域のニーズ → 生成分野ではLoRAで分離済み → 認識・自己教師あり・ストリームでは未整理」という流れで組めるか，検討する。
- MotionDirector，Customize-A-Video，Set-and-Sequenceの本文で，LoRAの配置と学習方法の詳細を確認する。
