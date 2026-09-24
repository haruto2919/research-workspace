---
date: 2026-09-24
project: sequential-video-lora-analysis
source_todo: null
topic: オンライン逐次動画学習におけるMoCo以外の学習方法
status: exploratory
tags: [brainstorm, research, online-learning, streaming-video, self-supervised-learning, lora]
---

# オンライン逐次動画学習におけるMoCo以外の学習方法

## 出発点

現行baselineは、画像事前学習済みViTへQ/V LoRAを追加し、ActivityNetを時系列順に入力しながらMoCo v2-styleのInfoNCE + EMA Key Encoder + FIFO Queueで更新する構成である。

今回の問いは、MoCo以外にオンライン／ストリーミング動画からLoRAを学習する方法として何が考えられるか、そのうち現在の研究目的である「画像情報だけでは得にくい時間的・動的情報をLoRAへ獲得させる」ことにどの方式が適するか、である。

ここでいうオンライン学習は、MoCo論文のonline encoderという意味ではなく、動画clipが時系列順に到着し、基本的にその順序で逐次更新するstreaming / sequential learningを指す。

## 重要な整理

MoCoの代替を考えるときは、次の2層を分ける必要がある。

1. **自己教師あり学習の目的関数**
   - 何を予測・一致・再構成させることで表現を学ぶか。
2. **オンライン／継続学習の安定化方法**
   - 逐次入力による勾配相関、忘却、分布変化をどう扱うか。

たとえばVideoMAEは自己教師目的、Orthogonal Gradientsは最適化方法であり、競合ではなく組み合わせ可能である。

## 自己教師あり学習の主要候補

### 1. BYOL系

BYOLは2つのaugmentation viewを使い、online networkがtarget networkの表現を予測する。target networkはonline networkのEMAで更新される。negative sampleを必要としない。

現在のMoCoとの違いは、Queueとnegativeを使わず、同一sampleの2 view間の予測だけで学習する点。

逐次動画への利点:
- small batchでもnegative不足が問題になりにくい。
- same-video clipをnegativeにしてしまうfalse negative問題を避けられる。
- 現行MoCoのEMA Query/Key構造をかなり再利用しやすい。

注意:
- 標準BYOLのobjectiveだけでは「clip Aとclip Bの時間順序」そのものを必ず学ぶとは限らない。
- 同一clipのaugmentation invarianceだけでは静的特徴に寄る可能性があるため、temporal positive設計が必要。

参考:
- Grill et al., Bootstrap Your Own Latent, NeurIPS 2020
  https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html

### 2. SimSiam

SimSiamは2 viewをSiamese networkへ入れ、一方にpredictor、他方にstop-gradientを入れて表現一致を学ぶ。negative sampleもmomentum encoderも不要。

利点:
- Queue不要。
- EMA encoder不要。
- 構造が非常に単純で、MoCo以外の最小baselineとして比較しやすい。

注意:
- strict sequential動画での性能は元論文の主対象ではない。
- temporal signalを明示しなければ、BYOL同様に「時間情報を学習した」とは直接言えない。

参考:
- Chen & He, Exploring Simple Siamese Representation Learning, CVPR 2021
  https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html

### 3. Barlow Twins / VICReg系

2 viewの表現を一致させつつ、各feature dimensionが同じ情報へcollapseしないよう分散・共分散を制約する。

MoCoと異なり、positive/negativeの分類そのものを使わない。

利点:
- Queue不要。
- negative mining不要。
- false negative問題を避けやすい。

注意:
- batch statisticsを使う方式は極小batchのstrict online条件と相性を検証する必要がある。
- 現行の1 clip/少数stream更新にそのまま持ち込めるとは限らない。

参考:
- Zbontar et al., Barlow Twins, ICML 2021
  https://proceedings.mlr.press/v139/zbontar21a.html

### 4. DINO / Self-Distillation

DINOはstudentとEMA teacherを使い、異なるviewの出力分布を一致させるself-distillation方式。contrastive negativeを使用しない。

利点:
- ViTとの相性が非常に強く研究されている。
- 現在のViT + LoRA構成に概念的に載せやすい。
- EMA teacherを使うためMoCo実装から一部構造を再利用できる。

注意:
- DINO自体は画像SSLとして設計されており、frame-wiseに使うだけでは時間情報は保証されない。
- multi-crop等のaugmentation設計が複雑になりやすい。

参考:
- Caron et al., Emerging Properties in Self-Supervised Vision Transformers, ICCV 2021
  https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html

### 5. VideoMAE / masked video modeling

時空間tokenを高い割合でmaskし、欠損部分を復元する。VideoMAEではtube maskingを用いる。

利点:
- 入力そのものが複数frameであり、時間方向を利用する自己教師目的を作りやすい。
- 2025年のStreaming Video with Orthogonal Gradients論文でも、streaming条件におけるVideoMAEが評価対象になっている。

注意:
- 現行実装はframe-wise ViT CLS -> masked meanであり、clip内順序を直接扱わない。そのままではVideoMAE objectiveを導入できない。
- temporal token interactionを可能にするarchitecture変更が必要。

参考:
- Tong et al., VideoMAE, 2022
  https://arxiv.org/abs/2203.12602

### 6. CPC / future feature prediction

現在までの表現から未来の表現を予測する。CPCでは未来表現をcontrastiveに予測する。

利点:
- 時間方向がobjectiveに直接入る。
- 「過去から未来を予測できる情報をLoRAが獲得した」という解釈がしやすい。
- streaming順序と自然に整合する。

注意:
- CPC原型はnegative samplingを使うため、MoCoと同様にnegative設計問題が残る。
- predictorやcontext encoderを追加する必要がある。

参考:
- van den Oord et al., Representation Learning with Contrastive Predictive Coding
  https://arxiv.org/abs/1807.03748

### 7. V-JEPA系 latent prediction

V-JEPAはpixelを再構成する代わりに、maskされた／未来の部分のlatent representationを予測する。

利点:
- pixel reconstructionより「意味のある動的表現」に集中させやすい。
- negative sampleを必要としない。
- videoの時間的理解を直接狙った現代的なself-supervised objective。
- V-JEPA 2 / 2.1ではmotion understandingやaction anticipationで強い結果が報告されている。

注意:
- target encoder / predictor / masking設計が必要で、現行MoCoより実装変更は大きい。
- strict online single-passでの学習安定性は別途検証が必要。

参考:
- Bardes et al., V-JEPA, ICLR 2024
  https://openreview.net/forum?id=WFYbBOEOtv
- Assran et al., V-JEPA 2, 2025
  https://arxiv.org/abs/2506.09985
- Mur-Labadia et al., V-JEPA 2.1, 2026
  https://arxiv.org/abs/2603.14482

### 8. temporal pretext task

動画固有の時間情報を直接pretext taskにする方式。

例:
- speed prediction
- temporal order / shuffle detection
- temporal cycle consistency
- start/end frameから中間frameを推定

SpeedNetは再生速度判定から時空間表現を学習する。TCCは複数動画間の時間対応をcycle consistencyで学習する。CVPR 2026のTimeBridgeは開始・終了frameから中間frameを復元することでframe-to-frame dynamicsを学ぶ。

利点:
- 「時間情報」を何として学ぶかが明確。
- shuffle/reverse評価との研究ストーリーを作りやすい。

注意:
- pretext task固有のshortcutを学ぶ可能性がある。
- ActivityNetのような多様な動画でどのtaskが汎用的か検討が必要。

参考:
- Benaim et al., SpeedNet, CVPR 2020
  https://openaccess.thecvf.com/content_CVPR_2020/html/Benaim_SpeedNet_Learning_the_Speediness_in_Videos_CVPR_2020_paper.html
- Dwibedi et al., Temporal Cycle-Consistency Learning, CVPR 2019
  https://openaccess.thecvf.com/content_CVPR_2019/html/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.html
- Wang et al., TimeBridge, CVPR 2026
  https://openaccess.thecvf.com/content/CVPR2026/html/Wang_TimeBridge_Self-Supervised_Video_Representation_Learning_via_Start-End_Joint_Embedding_and_CVPR_2026_paper.html

## オンライン／継続学習側の候補

### Replay / rehearsal

過去sampleまたは過去featureを小さなbufferへ保存し、新しいsampleと混ぜて再学習する。

これはMoCo queueとは目的が異なる。MoCo queueは主にnegative dictionaryだが、Experience Replayは過去知識の忘却を抑えるために過去sampleを再学習する。

現在の研究では、strict onlineで「過去動画を再び入力してよいか」という研究条件の定義が必要。

### Distillation / feature preservation

以前のmodelやfeatureをteacherとして保持し、現在modelが大きくずれないようdistillation lossを加える。

CaSSLeはself-supervised lossをcontinual learning用distillationへ変換し、BYOL、MoCoV2、SimCLR、Barlow Twins、SwAV、VICRegなど複数objectiveへ適用可能であることを示した。

参考:
- Fini et al., Self-Supervised Models Are Continual Learners, CVPR 2022
  https://openaccess.thecvf.com/content/CVPR2022/html/Fini_Self-Supervised_Models_Are_Continual_Learners_CVPR_2022_paper.html

### Gradient projection / Orthogonal Gradients

新しいbatchのgradientが直前のbatchと強く相関する問題に対し、gradient方向を直交化・射影する。

CVPR 2025のLearning from Streaming Video with Orthogonal Gradientsでは、streaming videoの非IID・高冗長性による性能低下を、optimizerへ直交勾配処理を加えることで改善し、DoRA、VideoMAE、future predictionの3方式でAdamWを上回ると報告している。

重要なのは、Orthogonal GradientsはMoCoの代替objectiveではなく、MoCo/BYOL/VideoMAE/future prediction等へ追加できるoptimizer側の手法であること。

参考:
- Han et al., Learning from Streaming Video with Orthogonal Gradients, CVPR 2025
  https://openaccess.thecvf.com/content/CVPR2025/html/Han_Learning_from_Streaming_Video_with_Orthogonal_Gradients_CVPR_2025_paper.html

### Parameter isolation / PEFT

base encoderをfreezeし、LoRAやadapter、promptだけを更新することで過去表現の破壊を抑える。

これは現在の研究がすでに採用している考え方に近い。ただし単一LoRAをずっと上書きする場合、LoRA内部でのforgettingは残る。複数LoRA、動的routing、過去LoRA固定などは後続候補。

CVPR 2026にもLoRAをcontinual learningへ使う研究が出ているが、現在の研究ではまず単一LoRAの逐次学習特性を測る方が解釈しやすい。

## 現研究との比較

| 方法 | Negative不要 | EMA/Teacher | 時間情報をobjectiveへ直接入れやすい | small-batch streaming適合性 | 現行実装からの変更量 |
|---|---:|---:|---:|---:|---:|
| MoCo v2 | × | ○ | △ | ○ queueあり | 基準 |
| BYOL | ○ | ○ | △〜○ | ○ | 小〜中 |
| SimSiam | ○ | × | △〜○ | ○ | 小 |
| Barlow Twins/VICReg | ○ | × | △ | △ batch依存 | 中 |
| DINO | ○ | ○ | △〜○ | △ | 中 |
| VideoMAE | ○ | × | ○ | △ | 大 |
| CPC/future prediction | 一部× | 任意 | ◎ | ○ | 中〜大 |
| V-JEPA | ○ | ○系 | ◎ | 未検証要素あり | 大 |
| temporal pretext | 任意 | 任意 | ◎ | ○ | 中〜大 |

## 現時点の収束

MoCo以外をすぐ比較baselineへ追加するなら、**BYOLまたはSimSiam**が実装距離の短い候補。

理由:
- 現行ViT + LoRA + clip featureを維持できる。
- MoCo特有のqueue/negative設計を外せる。
- small-batch sequential条件でも成立させやすい。
- MoCoとの違いが「negative + queueが必要か」という明確な比較になる。

一方、「LoRAが本当に時間情報を学ぶ」こと自体を強く検証したいなら、BYOL/SimSiamだけでは不十分な可能性が高い。より研究目的へ近いのは、**future feature prediction / V-JEPA / temporal pretext task**である。これらは時間方向をlossの成立条件に直接組み込める。

そのため、今後の比較軸としては、

- Baseline objective: MoCo vs BYOL/SimSiam
- Temporal objective: future prediction / V-JEPA系
- Streaming optimization: AdamW vs Orthogonal Gradients

のように、objectiveとoptimizerを分離して比較する設計が解釈しやすい。

## 反例・注意点

- 「動画を時系列順に入力している」だけでは、モデルが時間情報を利用したとは言えない。
- BYOL/DINO/MoCoで同一clipのaugmentation一致だけを学習する場合、静的semanticsだけでlossを下げられる可能性がある。
- VideoMAEでも、architectureとmask設計によっては見た目の補完へ偏り得る。
- future predictionは時間方向を使うが、低レベルpixel変化だけを学ぶshortcutを避ける必要がある。latent predictionはその対策候補。
- replayはforgettingには有効でも「完全single-pass」というオンライン条件を弱めるため、採用時はprotocolを明示する必要がある。

## 未解決事項

- online条件を「過去sample再利用禁止」と定義するか。
- 1 clip更新か、4-stream mini-batchを許すか。
- clip内frame順を使うtemporal moduleを追加するか。
- LoRAだけでtemporal predictionを成立させるのか、軽量predictorを追加してよいか。
- MoCoの次の比較baselineをBYOL、SimSiam、future predictionのどれにするか。
- Orthogonal Gradientsをobjective比較とは独立した第2軸として導入するか。

このメモは探索記録であり、specまたは実装許可ではない。
