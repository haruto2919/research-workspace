---
date: 2026-10-19
project: sequential-video-lora-analysis
source_todo: null
topic: research-presentation-related-work-gap-survey
status: exploratory
tags: [brainstorm, research, literature-survey, presentation, streaming-video, self-supervised-learning, temporal-information, lora, moco]
---

# 中間発表向け関連研究の位置付けとResearch Gapの調査

## 出発点・研究文脈
- ユーザーの依頼：オンライン学習、動的情報を取り出すモチベーション、自己教師あり学習の三観点で、各論文が「できたこと→未解決・未検証→自分の研究の価値」に結び付く文献を選定する。
- Research WorkspaceのREADMEでは、画像事前学習済みモデルに逐次動画を入力し、LoRAで追加学習して動的情報が保持されるかを内部解析することが中心目的。
- 2026-10-15 MTGの合意：LoRA・MoCo・Loaderを分けて検証し、ResNet/MoCoの基準確認→ViT/LoRA/Sequential Loaderへ段階統合。2026-10-30中間発表ではLoRA解析方針を明示。
- 最新の2026-10-19 Evidence：A1 synthetic/2GPU/16stepで公式Meta MoCoと診断S1の記録対象がbitwise一致。実画像ImageNet長期学習や動画LoRAの時間情報獲得は未検証。
- **基本的な論理線**：既存研究はオンライン更新・動画動的表現・動画SSL・画像モデルの省パラメータ適応をそれぞれ進めている。これらをstrict sequential self-supervised adaptation + frozen image-ViT + LoRA temporal analysisという交差領域で実証・解析することが本研究の候補上の貢献。交差領域そのものに先行研究が無いとは主張しない。

## 用語の厳密な区別
- online inference/online recognition: 未来フレームなしに逐次推論できる。必ずしも逐次的なパラメータ学習ではない（TSM/MeMViTなど）。
- streaming / online model learning: 観測順序に従ってパラメータを更新する。sample replayとsingle-pass、滑り窓のoverlapは別条件で明示する。
- MoCoのdynamic queue/dictionaryはself-supervised representation learningの計算方式であり、それ自体で「strict streaming video learning」を実現するものではない。
- 動画の入力と動的情報の獲得は異なる。frame-wise ViT + masked meanはフレーム順に不変。InfoNCE aloneは時間変換の理解を直接強制しない。

## 主要研究（全て原論文・公式会議ページにアクセス確認済み）

### A. オンライン・逐次学習
1. **Learning from One Continuous Video Stream** (Carreira et al., CVPR 2024)
   - https://openaccess.thecvf.com/content/CVPR2024/html/Carreira_Learning_from_One_Continuous_Video_Stream_CVPR_2024_paper.html
   - 達成：単一連続動画からのオンライン学習設定と適応・汎化評価。future prediction pretrainingやupdate cadenceを分析、条件下でbatch 1のi.i.d.学習に匹敵。
   - 本研究との違い：pixel-to-pixel予測が中心で、image-ViT/LoRA、MoCo、LoRA内部の動的情報評価は主題でない。
2. **Learning from Streaming Video with Orthogonal Gradients** (Han et al., CVPR 2025)
   - https://openaccess.thecvf.com/content/CVPR2025/html/Han_Learning_from_Streaming_Video_with_Orthogonal_Gradients_CVPR_2025_paper.html
   - 達成：動画をshuffledから順序入力にするとrepresentation learningが落ちることをDoRA/VideoMAE/future predictionで確認。gradientの相関へのorthogonal optimizerを提案し改善。
   - 残る問い：optimizerの問題を解くのが中心。frozen image-pretrained ViTを自己教師ありLoRA更新した際の動的情報・LoRA内部解析は直接対象外。
   - 関係：chronological vs shuffled、およびgradient相関のbaselineを要請。
3. **I Have a Stream: Making Self-Supervised Learning Work on Continuous Video** (Martinović et al., arXiv 2026-09-30、StreamMAE)
   - https://arxiv.org/abs/2609.40333 / https://martinovicivan.github.io/StreamMAE/
   - 達成：strict sliding-window single-pass動画フレーム学習におけるcontrastive/distillationの困難を検証。intra-batch near-duplicatesの重要性を分析、StreamMAEを提案。
   - **危険な過剰主張を避ける**：「MoCoは動画streamで使えない」という一般定理ではない。論文の比較はMoCo-v3・from-scratch frame stream等で、ユーザーが試すMoCo-v2式/LoRA/clip negativeとは条件が異なる。
   - 関係：same-video past negatives、near-duplicate、batch構成、MoCo-vs-MAEの検証理由を強める。成功前提ではなく失敗可能性のEvidence。
4. **Memory Storyboard: Leveraging Temporal Segmentation for Streaming Self-Supervised Learning from Egocentric Videos** (Yang & Ren, CoLLAs/PMLR 2026)
   - https://proceedings.mlr.press/v330/yang26a.html
   - 達成：長尺実動画のストリーミングSSLで短期・長期メモリとtemporal segmentationによるreplayを提案、semantic representationsを改善。
   - 残る問い：raw/segment memory replayを活用する手法で、replay無しとの公平比較やLoRA動的情報解析は本論文の中心でない。したがって「streaming SSL自体が未実現」とは言えない。
5. **Online-LoRA: Task-Free Online Continual Learning via Low Rank Adaptation** (Wei et al., WACV 2025)
   - https://openaccess.thecvf.com/content/WACV2025/html/Wei_Online-LoRA_Task-Free_Online_Continual_Learning_via_Low_Rank_Adaptation_WACV_2025_paper.html
   - 達成：pretrained ViTに対するonline-LoRAと忘却を抑えるregularization、task-free classification streamsで評価。
   - 未対象：連続動画での自己教師あり動的表現とそのLoRA解析ではない。LoRAがonlineに使える点の比較。

### B. 動的情報の必要性と獲得
6. **The Something Something Video Database for Learning and Evaluating Visual Common Sense** (Goyal et al., ICCV 2017)
   - https://openaccess.thecvf.com/content_iccv_2017/html/Goyal_The_Something_Something_ICCV_2017_paper.html
   - 達成：フレーム間の物理的相互作用を識別する動画ベンチマークの確立。動画理解には静的外観のみで足りない課題がある。
   - 論文はdataset提案であり新規の連続的学習法は提示しない。動きの情報が必要という導入材料。
7. **ST-Adapter: Parameter-Efficient Image-to-Video Transfer Learning** (Pan et al., NeurIPS 2022)
   - https://proceedings.neurips.cc/paper_files/paper/2022/hash/a92e9165b22d4456fc6d87236e04c266-Abstract.html
   - 達成：画像事前学習モデルに小型時空間adapterを追加し、少数更新parameterで動的動画認識に適応。
   - 未対象：strict streaming single-pass self-supervision、LoRA内部に何が入るかの解析。反論ポイント：parameter-efficient temporal learning自体は既に実証されているため、新規性にしてはならない。
8. **MotionMAE: Self-supervised Video Representation Learning with Motion-Aware Masked Autoencoders** (Yang et al., BMVC 2024)
   - https://bmvc2024.org/proceedings/499/
   - 達成：masked appearance reconstructionに加え、近接frameのtemporal difference由来のmotion structureを予測し、静的・動的情報を学習。
   - 未対象：single-pass sequential LoRA adaptationやLoRA内部の静動分離。現行masked mean+MoCoにはexplicit motion targetがないため実験設計の比較として重要。
9. **TimeBridge: Self-Supervised Video Representation Learning via Start-End Joint Embedding and In-Between Frame Prediction** (Wang et al., CVPR 2026)
   - https://openaccess.thecvf.com/content/CVPR2026/html/Wang_TimeBridge_Self-Supervised_Video_Representation_Learning_via_Start-End_Joint_Embedding_and_CVPR_2026_paper.html
   - 達成：start/end間の中間frame復元を通じ動的変換そのものを自己教師ありで学習し、dense predictionで評価。
   - 未対象：LoRAによるstrict streaming adaptationやLoRA情報解析。「既存SSLは動きが学べない」と言わないための重要な対照。

### C. 自己教師あり動画学習
10. **Momentum Contrast for Unsupervised Visual Representation Learning** (He et al., CVPR 2020)
    - https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html
    - 達成：EMA key encoder+FIFO queueによる画像SSLと転移性能。
    - 未対象：逐次動画入力・明示的な時間順序学習・LoRAの内部解析。研究のMoCo選定理由とvalidation基準。
11. **VideoMoCo: Contrastive Video Representation Learning with Temporally Adversarial Examples** (Pan et al., CVPR 2021)
    - https://openaccess.thecvf.com/content/CVPR2021/html/Pan_VideoMoCo_Contrastive_Video_Representation_Learning_With_Temporally_Adversarial_Examples_CVPR_2021_paper.html
    - 達成：video SSLのためtemporal frame dropoutとqueue内のkey agingへtemporal decay。
    - 未対象：厳密なsingle-passストリーミングLoRA更新と、その動的知識の内部解析。むしろMoCo類似法が動画にも適応済みという強い先行研究。
12. **Spatiotemporal Contrastive Video Representation Learning (CVRL)** (Qian et al., CVPR 2021)
    - https://openaccess.thecvf.com/content/CVPR2021/html/Qian_Spatiotemporal_Contrastive_Video_Representation_Learning_CVPR_2021_paper.html
    - 達成：動画のtime-consistent augmentationとclip samplingを用いる自己教師ありcontrastive pretraining。
    - 未対象：single-pass strict chronological LoRA learningと解析。
13. **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training** (Tong et al., NeurIPS 2022)
    - https://arxiv.org/abs/2203.12602
    - 達成：tube maskingと高mask率による動画自己教師あり表現を示す。
    - 未対象：本研究のLoRA+strict streaming/LoRA情報解析。MoCo以外の重要なobjective比較候補。

## 収束：中間発表で優先して取り上げたい6〜8本
- **オンライン**：Han 2025、StreamMAE 2026。余裕あればCarreira 2024 / Memory Storyboard 2026。
- **動的情報**：ST-Adapter 2022、MotionMAE 2024またはTimeBridge 2026。導入にSomething-Something 2017。
- **SSL**：MoCo 2020、VideoMoCo 2021、対比としてVideoMAE 2022。
- 横断補助：Online-LoRA 2025（LoRA更新の実現可能性と未扱いの動画dynamic informationを接続）。

## 提案するスライドの論理
1. 連続動画では非IID・近接frame相関が学習を難しくする：Han 2025、StreamMAE 2026。
2. 動画理解では時間変化の表現が不可欠で、画像モデルの少数parameter適応も可能：Something-Something / ST-Adapter、MotionMAE/TimeBridge。
3. MoCo/VideoMoCo/VideoMAEによりラベル無しの動画表現学習は進展した。しかし基本的にこれらだけでは、strict streamingにおけるLoRAの動的情報の所在・時系列順理解を直接証明しない。
4. 本研究：Frozen image ViT + Sequential Loader + self-supervised learning + Q/V LoRAを段階検証、linear probe / temporal shuffle-reversal / static-vs-dynamic controls / LoRA-layer SVDなどで獲得情報を評価する（評価案は未承認）。

## 反例・代替説明・未決
- 動画を入力して学習が進んでも、appearance、scene ID、augmentation shortcutだけを獲得した可能性がある。
- frame-wise CLS + masked meanはpermutation invariant。shuffle/reversal sensitivityの欠如は現行architecture由来の可能性があり、LoRA学習不全と切り分ける。
- contrastive negativesとしてsame-video近接clipを使うとfalse negativesの可能性がある。queueがあるからstreaming MoCoが必ず成立するとは言えない。
- LoRAの少ない更新parameterそのものやimage-to-video adapterは既知。新規性は組合せの検証と内部情報の説明に置くが、先行研究全探索は未完了であり新規性確定ではない。
- 比較案：chronological vs shuffled、LoRA ON/OFF、image vs video、MoCo vs video-specific SSL、same-video negative設定、時間順序課題、固定下流評価＋LoRA SVD。実行計画は別Gate/承認が必要。
- 2026-10-19の数値一致は機械的parityの範囲で、動画表現品質・LoRA動的知識は実験Evidenceが無い。

## 次アクション候補（未採用）
- 発表スライドの掲載論文を6〜8本に絞り、3節に分割。
- 先行研究ごとの原図使用許諾・出典表示を確認し、発表用の比較表に落とす。
- Temporal order controlを必須評価に含めるかのresearch-spec前の検討。
- MoCo-v2式とStreamMAEとの直接比較を要件にするかは学習コストと研究scopeを踏まえて選ぶ。

## 関連Research Workspace
- `.research/lab/projects/sequential-video-lora-analysis/README.md`
- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-10-15-mtg.md`
- `.research/lab/projects/sequential-video-lora-analysis/experiments/2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md`
- `.research/secretary/notes/brainstorm/2026-10-08-strongly-related-paper-survey.md`
- `.research/secretary/notes/brainstorm/2026-10-19-post-a1-16step-moco-parity-next-priorities.md`
