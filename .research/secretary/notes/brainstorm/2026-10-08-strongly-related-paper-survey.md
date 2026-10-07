---
date: 2026-10-08
project: sequential-video-lora-analysis
source_todo: null
topic: 現在研究と関連の強い論文の再調査
status: exploratory
tags: [brainstorm, research, literature-survey, streaming-video, self-supervised-learning, lora, moco, temporal-modeling]
---

# 現在研究と関連の強い論文の再調査

## 現在研究の検索軸

Research Workspaceの現行README・approved full-dataset spec・既存brainstormを基に、次の6軸で論文を再調査した。

1. continuous / streaming videoを時系列順にsingle-passで学習
2. self-supervised video representation learning
3. MoCo / contrastive learningとpast representation queue
4. image-pretrained ViTをfreezeしてparameter-efficientにvideoへ適応
5. static appearanceではなくmotion / temporal dynamicsを学習するobjective
6. sequential non-IID入力とcorrelated gradients

現在の中心目的は標準的なtask/class-incremental continual learningではなく、画像事前学習ViTを動画streamへLoRAで自己教師あり適応し、動画由来の有用な情報を追加できるか、その内部を解析すること。

## 最重要論文

### S1. I Have a Stream: Making Self-Supervised Learning Work on Continuous Video
- Martinović, Knobel, Asano, arXiv 2026
- https://arxiv.org/abs/2609.40333
- temporally ordered continuous video、global reshuffleなし、multi-epoch replayなしでSSLを比較。
- MoCo v3、DINO、MAEを同一streaming条件で比較し、contrastive/distillation系が苦戦、MAEがよりrobustと報告。
- near-duplicate framesによる高いintra-batch similarityとfalse negativeを主要問題として分析。
- 現行Streaming MoCoの妥当性を直接問い直す最重要先行研究。ただし論文はfrom-scratch frame-stream、現研究はpretrained ViT + LoRA + clip-level queueなので同一条件ではない。

### S2. Learning from Streaming Video with Orthogonal Gradients
- Han et al., CVPR 2025
- https://arxiv.org/abs/2504.01961
- continuous videoをself-supervisedに逐次学習し、shuffleからsequentialへ変えると性能が落ちることをDoRA、VideoMAE、future predictionで検証。
- 原因としてcorrelated gradientsに着目し、SGD/AdamWへOrthogonal Gradientを導入。
- 現研究のsequential non-IID、AdamW、chronological vs shuffled、将来のOrthogonal Gradients比較に直接関係する。

### S3. VideoMoCo: Contrastive Video Representation Learning with Temporally Adversarial Examples
- Pan et al., CVPR 2021
- https://arxiv.org/abs/2103.05905
- MoCoをvideo representationへ拡張。
- temporal frame dropoutによるtemporal robustnessと、queue内のold keyにtemporal decayを入れる。
- 現行研究のMoCo queue staleness、past key、temporal augmentationに直接関係する。
- offline video SSLでありstrict single-pass streamingではない点は違う。

### S4. Learning from One Continuous Video Stream
- Carreira et al., CVPR 2024
- https://arxiv.org/abs/2312.00598
- single continuous video、shuffleなし、batch-size 1に近いonline learningを扱い、consecutive framesの高相関とadaptation/generalizationを分析。
- future prediction pretrainingとupdate paceを検討。
- 現行strict streamの研究背景として非常に近い。ただし目的関数・モデルはMoCo/LoRAではない。

## temporal / motion情報獲得に強く関係

### A1. TimeBridge
- Wang et al., CVPR 2026
- start/end frameからin-between framesを復元し、temporal transformationそのものをself-supervisedに学習。
- 現行MoCo + masked meanがframe order/motionをloss成立条件にしていない問題への強い比較対象。
- https://heisen-wang.github.io/timebridge.github.io/

### A2. MotionMAE
- Yang et al., BMVC 2024
- static appearanceだけでなく、近接frame差分からmotion structureを明示的に予測。
- 「動画を入力した」ことと「動的情報を学習した」ことを区別する現在研究に直接有用。
- https://arxiv.org/abs/2210.04154

## image-pretrained modelをparameter-efficientにvideoへ適応する点で強く関係

### A3. ST-Adapter
- Pan et al., NeurIPS 2022
- https://arxiv.org/abs/2206.13559
- pre-trained image modelをfreezeし、lightweight spatio-temporal adapterだけでvideo understandingへ適応。
- 「image modelにはtemporal knowledgeがないため、parameter-efficient moduleでdynamic reasoningを追加する」という研究仮説が現研究に近い。
- supervised action recognitionでありself-supervised streamingではない。

### A4. AIM: Adapting Image Models for Efficient Video Action Recognition
- Yang et al., ICLR 2023
- https://arxiv.org/abs/2302.03024
- pretrained image transformerをfreezeし、spatial / temporal / joint adapterだけを学習。
- frame-wise image featuresの平均だけではtemporal modelingが弱く、explicit temporal adaptationが重要であることを示す。
- 現行frame-wise CLS + masked mean baselineの限界を考える比較研究として重要。

## streaming architectureの比較候補

### A5. Learning Streaming Video Representation via Multitask Training (StreamFormer)
- Yan et al., ICCV 2025
- https://arxiv.org/abs/2504.20041
- pretrained vision transformerへcausal temporal attentionを追加し、未来frameを使わずstreaming representationを形成。
- 現行late-fusion masked meanと対照的なexplicit temporal architecture。
- self-supervised MoCo/LoRAではなくmultitask visual-language trainingなので、直接baselineというよりarchitecture comparison。

## 基礎的関連

### B1. The Challenges of Continuous Self-Supervised Learning
- Purushwalkam et al., ECCV 2022
- https://arxiv.org/abs/2203.12710
- continuous non-IID SSLでtemporal correlation、inefficiency、forgettingを分析しMinRed replay bufferを提案。
- 現行scopeはraw replayなしなので方法は異なるが、continuous SSLの問題設定の基礎文献。

## 前回のcontinual-LoRA論文の位置付け

- Online-LoRA: online ViT+LoRAという設定面では関連するが、supervised continual classification / forgettingが中心。
- EBLoRA: LoRA SVD/effective-rank解析フェーズでは関連が強い。
- InfLoRA / Janus-LoRA: forgetting/order-dependenceのEvidence取得後にorthogonalityを導入する段階で重要。
- 現在のStreaming MoCo + video information acquisitionのコア先行研究としては、上記S/A群より優先度を下げる。

## 現時点の読書優先順位

1. I Have a Stream
2. Learning from Streaming Video with Orthogonal Gradients
3. VideoMoCo
4. AIM / ST-Adapter
5. TimeBridge / MotionMAE
6. Learning from One Continuous Video Stream
7. StreamFormer
8. Continuous SSL
9. continual-LoRA群

## 研究への具体的な示唆

- 現行MoCoを「streaming SSL baseline」として維持する価値はあるが、2026年のI Have a StreamはMoCo v3がcontinuous streamで苦戦するEvidenceを示すため、full runの結果解釈では必ず対照に置く。
- same-video all-past negativesは近接clipをfalse negativeにする可能性があり、VideoMoCoのtemporal decayやI Have a Streamのnear-duplicate分析が直接参考になる。
- current masked meanではcross-frame relationを直接表現しないため、temporal claimを行うにはTimeBridge、MotionMAE、AIM、StreamFormerのようなexplicit temporal mechanismとの比較が重要。
- chronological input自体の難しさはOrthogonal Gradients論文が最も直接的に扱う。まずAdamW baselineを成立させ、その後のablationとして位置付けるのが自然。
- image-pretrained ViTをfreezeして少数parameterでvideoへ適応する研究仮説はST-Adapter/AIMによって既に強く支持される。現在研究の新規性は、この発想をsingle-pass self-supervised streaming + LoRA内部解析へ持ち込む点に置く方が整理しやすい。

## 未解決

- I Have a StreamのMoCo v3 failureが、clip-level + FIFO queue + pretrained LoRAの現行Stage 6Bでも再現するか。
- all-past negative policyとtemporal decay / exclusion windowの比較をいつ行うか。
- temporal-information claim用にTimeBridge/MotionMAE系objectiveを実装比較するか。
- AIM/ST-Adapterのexplicit temporal moduleと、現行masked mean + Q/V LoRAの差をどの評価で切り分けるか。
