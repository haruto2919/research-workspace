---
date: 2026-10-21
project: sequential-video-lora-analysis
source_todo: null
topic: static-dynamic-information-goyal-choi-kowal-review
status: exploratory
tags: [brainstorm, research, literature-survey, temporal-modeling, lora, scene-bias, static-dynamic-information, intermediate-representation]
---

# 画像モデルの静的情報依存と動画の動的情報を区別する意義 — 3論文の原典調査

## 出発点・現在の研究文脈

- ユーザー提示スクリーンショットの3文献 Goyal+ ICCV2017、Choi+ NeurIPS2019、Kowal+ CVPR2022 の中身を一次資料に基づき確認し、研究発表で「静的情報と動的情報を区別したい理由」へつなげる。
- 対象project: `.research/lab/projects/sequential-video-lora-analysis/`。
- 2026-10-20までのREADME上、Meta MoCo公式対自作のImageNet ResNet50再現性を調査中で、Frozen ViT+LoRA+Sequential Loaderの動的情報獲得は未確認。LoRA解析は研究の中心目標だが実験結果を主張できない。
- 2026-10-15 MTGでは中間発表のLoRA解析方針を重視。日次TODO `2026-10-21.md` は未作成。
- 本メモは論文内容と発表上の位置付けに関する探索記録であり、spec・実験承認ではない。

## 1. Goyal et al., ICCV 2017（動的情報が必要な課題）

- 原題: *The "Something Something" Video Database for Learning and Evaluating Visual Common Sense*
- 公式: https://openaccess.thecvf.com/content_iccv_2017/html/Goyal_The_Something_Something_ICCV_2017_paper.html
- PDF: https://openaccess.thecvf.com/content_ICCV_2017/papers/Goyal_The_Something_Something_ICCV_2017_paper.pdf
- 問い: 物体や場面だけでなく、行為・物理的相互作用・時間順の区別を要求する動画理解課題をどう用意するか。
- 達成: 人手で撮影・ラベル付けした10万本超、174種の動作テンプレートから成るSomething-Somethingデータセットを構築（2017原論文はV1であり後続のV2と混同しない）。
- 結果: 原論文Table4の40カテゴリでは pretrained 2D CNN+Avg top1 error79.2%（accuracy20.8%）、pretrained 2D CNN+LSTM 77.8%（22.2%）、pretrained 3D CNN+Avg 66.2%（33.8%）、2D+3D特徴融合63.8%（36.2%）。設定依存で単純な普遍的な「2Dでは不可能」の証拠ではない。
- 境界: データセット提案・教師ありアクション予測が中心。online SSLやLoRA解析ではない。
- 研究との接続: 静止画だけから識別しづらい方向・順序などを動画表現に加える意義の導入。発表例は「持ち上げる／置く」など概念例としてよいが元論文の正確なクラス名として断定しない。

## 2. Choi et al., NeurIPS 2019（静的背景への過度な依存を緩和）

- 原題: *Why Can't I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition*
- 公式: https://proceedings.neurips.cc/paper/2019/hash/ab817c9349cf9c4f6877e1894a1faa00-Abstract.html
- PDF: https://proceedings.neurips.cc/paper_files/paper/2019/file/ab817c9349cf9c4f6877e1894a1faa00-Paper.pdf
- 問い: 動作の識別に無関係な背景・場面とアクションラベルとの偶然の相関を学習し、異なる場面やタスクへの転移が弱くなる問題。
- 達成: Mini-Kinetics-200のラベル付き行動分類のCEに、(a) scene adversarial loss（pretrained scene classifierのpseudo soft labels/gradient reversal）、(b) human-mask confusion/entropy loss（人物を隠した入力ではアクションを予測できないように）を追加。
- 原論文 Table1 3D-ResNet-18 transfer top1：HMDB51 53.6→56.7%、UCF101 83.5→84.5%、Diving48 18.0→20.5%。scene classifier（pseudo label）29.7→2.9%。両損失が相補的とablation。
- 境界: 静的情報すべてを排除する研究ではない（scene biasに焦点）。背景を捨てればmotion理解が必ず成立するわけでもない。教師ありpretrainingでありstreaming/LoRA内部分離ではない。著者自身が最終性能はSOTAに届いていないと明記。
- 研究との接続: 動画で学習してもシーン・物体・背景への依存だけで正答できるリスクの根拠。LoRA表現の場面差/動作差を切り分けて評価する必要性を支持。

## 3. Kowal et al., CVPR 2022（モデル内部の静的・動的情報を定量化）

- 原題: *A Deeper Dive Into What Deep Spatiotemporal Networks Encode: Quantifying Static vs. Dynamic Information*
- 公式: https://openaccess.thecvf.com/content/CVPR2022/html/Kowal_A_Deeper_Dive_Into_What_Deep_Spatiotemporal_Networks_Encode_Quantifying_CVPR_2022_paper.html
- arxiv/pdf: https://arxiv.org/pdf/2206.02846
- 区別: static = 単一フレームから抽出可能な外観/テクスチャ等、dynamic = 複数フレームから初めて分かる動き・変化等（動きmotionはdynamicの部分集合）。
- 方法: dynamic pair = 同じ動画に異なる動画style（外観変更）を適用して動き保持、static pair = 同じstyleの動画でフレームをシャッフルし時間相関を変え外観保持。中間層特徴/各channelの相関（mutual-information関連の代理量）を計測し、static/dynamic/joint/residualの4種類に分類。VOSではoptical flow jitterを使用。
- 結果: 検証した多数のCNN/Transformer動画モデルはstatic偏重。SlowFast等のfast stream/一部cross-connectionがdynamicを取りやすい。SSv2はDiving48/Kineticsより動的情報を評価しやすいと報告。例 Fig5 SlowOnly Kinetics dynamic21.4% static38.4% vs SSv2 dynamic28.2% static30.7%；FastOnly SSv2 dynamic31.6% static21.9%（同論文条件依存）。中間channelは静的・動的・両方・残差に分かれる。
- 境界: これは学習済み動画モデルの中間activationの解析であり、LoRA weightの特異値そのものを分析して「動き」を識別する方法ではない。frame-wise ViT+masked meanはフレーム順序に不変なため、入力を並び替えるだけではclip表現の変化を測れない。style処理も完全に静的/動的を独立制御しているわけではないため指標は操作依存。
- 研究との接続: 明示的なcontrol動画ペアで frozen ViT vs ViT+LoRA の中間channel/CLS表現を比較する研究案（現時点では未採用）。LoRAのlayer別SVD・rank分析は更新量や容量の指標で、static/dynamicを直接識別しない。入力表現の変化/ablationと対応づけることを提案。

## 3本の論理的な接続

1. Goyal (2017): 時系列の差異が認識結果を変える課題を体系化した。
2. Choi (2019): 動画モデルが正解しても、実際には背景等のstatic shortcutへ頼る場合がある。特定のscene biasを減らすことで転移が改善する。
3. Kowal (2022): 内部channelには静的・動的・混合型があり、モデルやデータセットによって分布が異なる。出力性能だけでなく中間表現を測定する意義を支持。
4. 本研究の問い（未検証）: ImageNet事前学習ViTをfreezeし、逐次動画の自己教師あり学習でLoRAだけを（ただしMoCo Projectorも学習）適応したとき、LoRAによる追加表現は静的か動的か。既存のKowal解析を直接LoRA重みに適用できるとは断定しない。

## 比較候補・反例・未解決

- 候補: 同場面・異動作、同動作・異背景、ordered vs shuffled、static-repeat、reverse、style changeのcontrolペア。LoRA ON/OFF・pre-projector特徴・層別feature・channelsと併用。
- frame-wise CLSの単純masked meanは順序不変なので、order-sensitive downstreamを計測するにはtemporal architecture/objective変更または時系列順のparameter trajectory変化解析が必要。これは現行baselineの制約。
- Choiの「背景抑制」は動き情報そのものの獲得の直接証明ではない。
- Kowalのstatic/dynamic unit割合もモデル依存・スタイル変換依存の推定値。LoRA特異値だけで時間情報の所在を確定しない。
- 比較手法の本採用・実験runは未決。spec作成・TODO追加は本メモでは行わない。

## 関連文脈
- .research/lab/projects/sequential-video-lora-analysis/README.md
- .research/lab/projects/sequential-video-lora-analysis/meetings/2026-10-15-mtg.md
- .research/secretary/notes/brainstorm/2026-10-08-strongly-related-paper-survey.md

## 次アクション候補（未採用）
- 3本を導入1～2スライドへまとめ、原図の引用出典を明記。
- Kowalのcontrolled video pairsをLoRA機能評価へ転用できるか、frame-wise baselineのorder-invarianceと合わせて検討。
