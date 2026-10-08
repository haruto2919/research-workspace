---
date: 2026-10-08
project: sequential-video-lora-analysis
source_todo: null
topic: 英論文紹介向け候補選定
status: exploratory
tags: [brainstorm, research, literature-survey, paper-presentation, streaming-video, lora, temporal-modeling]
---

# 英論文紹介向け候補選定

## 選定基準

現在のResearch Workspace上の研究との関連性に加え、英論文紹介としての新規性と説明しやすさを重視した。

評価軸:
1. 2025-2026年の新しさ
2. 現在研究との直接関連
3. 新規性を一言で説明しやすいか
4. 方法図・実験結果を発表で説明しやすいか
5. peer-reviewed venueの有無

## 最有力候補

### 1. I Have a Stream: Making Self-Supervised Learning Work on Continuous Video
- Ivan Martinović, Lukas Knobel, Yuki M. Asano
- arXiv:2609.40333
- 2026-09-30
- arXivコメントおよびNeurIPS 2026公式Downloadsに掲載を確認
- continuous videoをtemporal orderのまま、global shuffleなし、multi-epoch replayなしでself-supervised learningする。
- contrastive/distillation系がcontinuous streamで苦戦することを分析し、主因をintra-batch near-duplicate redundancyと特定。
- StreamMAEとしてstream-aware regularization、DataDrop、motion-biased cropを提案。
- 現研究のsingle-pass streaming SSL、MoCoのnear-duplicate / false-negative問題に非常に直接的。
- LoRAは扱わない点が差分。
- 英論文紹介としては問題設定→失敗分析→提案→結果が明快で、2026年9月末と非常に新しい。

### 2. TimeBridge: Self-Supervised Video Representation Learning via Start-End Joint Embedding and In-Between Frame Prediction
- Wang et al., CVPR 2026
- start/end frameから中間frameを再構成してtemporal transformationを明示的に学習。
- 現研究の「動画を学習したLoRA」と「時間情報を学習したLoRA」を区別する問いに直接関係。
- CVPR 2026でpeer-reviewed、図示しやすく、発表向け。
- streaming/LoRAではない。

### 3. TemporalDoRA: Temporal PEFT for Robust Surgical Video Question Answering
- Carlini et al., arXiv:2603.09696, 2026
- standard PEFTがframe間 interactionをadaptation pathwayで扱わない問題を指摘。
- low-rank bottleneck内部にtemporal MHAを入れたvideo-specific DoRA。
- frozen backbone + low-rank adaptation + explicit temporal mixingで、現在研究との構造的関連は非常に強い。
- surgical VideoQAというdomain差があり、現時点ではarXiv preprint。

## 有力候補

### 4. TM-Adapter: Temporal Merge Adapter for Efficient Global Temporal Modeling
- Hahm et al., WACV 2026
- image-to-video PETL。
- consecutive frame redundancyをmerge-unmergeで抑え、local/global temporal branchを持つ。
- pretrained image modelをfreezeして少数parameterでvideo temporal representationを追加する点が近い。
- supervised action recognition、AdapterでありLoRA/self-supervised/streamingではない。

### 5. Learning from Streaming Video with Orthogonal Gradients
- Han et al., CVPR 2025
- continuous video SSLでshuffle→sequentialによる性能低下をcorrelated gradientsとして分析。
- SGD/AdamWへOrthogonal Gradientsを導入。
- 現研究のchronological single-pass、AdamW、sequential non-IID問題へ非常に直接的。
- 新規性重視では2026論文より一段下。

### 6. D2ST-Adapter
- Pei et al., ICCV 2025
- image-to-video PEFTでspatial/temporal featureをadapter内でdisentangle。
- 現研究の将来的なstatic/dynamic information separationと強く関連。
- few-shot supervised action recognitionであり現在のself-supervised streamingとは距離がある。

### 7. LiON-LoRA
- Zhang et al., ICCV 2025
- video diffusionでLoRAのspatial / temporal controlをorthogonality・norm consistency・linear scalabilityから扱う。
- LoRA内部のstatic/dynamic分離やorthogonalityという将来方向と関連。
- video generation domainで現在のrepresentation learningとは距離がある。

## 英論文紹介としての推奨順位

1. I Have a Stream
2. TimeBridge
3. TemporalDoRA
4. TM-Adapter
5. Learning from Streaming Video with Orthogonal Gradients

## 推奨判断

### 新規性と現在研究の両方を最大化
I Have a Stream。

理由:
- 2026-09-30と極めて新しい。
- continuous / temporally ordered / no global shuffle / no multi-epoch replay / self-supervisedという研究設定が現在研究に非常に近い。
- 現在採用しているMoCo系contrastive objectiveがstreamで苦戦するという結果があり、直接研究上の問いにつながる。
- 問題分析が明快で、発表ストーリーを作りやすい。

### 「時間情報をどう学ぶか」を中心に発表
TimeBridge。

理由:
- CVPR 2026。
- temporal transformationを明示的に学習するという新規性が明快。
- 現在のmasked mean + MoCoがtemporal orderを直接要求しないという研究課題と対比しやすい。

### 「LoRA/PEFT × 動画」を中心に発表
TemporalDoRA。

理由:
- low-rank adaptation内部へtemporal mixingを入れるため、現在のQ/V LoRA研究と構造的に近い。
- ただしpreprintでありdomainがsurgical VQA。

## 現時点の結論

英論文紹介の題材として1本選ぶなら、最有力は I Have a Stream。
研究との関連性だけでなく、「既存SSLはcontinuous videoでなぜ失敗するか」という明快な問題設定と、StreamMAEという提案、最新性、現在のStreaming MoCoへの直接的な示唆が揃っている。

TimeBridgeはaccepted conference paperとしての安心感と、temporal representationの説明しやすさで強い。
TemporalDoRAはLoRAとの直接性が最も強いが、preprintとapplication domainの差を許容できる場合の候補。
