---
date: 2026-09-17
project: sequential-video-lora-analysis
source_todo: null
topic: 現在研究方針に近いMAE・CLIP・late fusion関連文献調査
status: exploratory
tags: [brainstorm, research, literature-review, mae, clip, lora, video, late-fusion, self-supervised, online-learning]
---

# 現在研究方針に近いMAE・CLIP・late fusion関連文献調査

## 出発点

現在の研究主軸は、画像で自己教師あり事前学習されたMAEを出発点とし、Base MAEを基本的に固定しながらencoder attentionのQ/VへLoRAを追加し、動画の自己教師あり学習によって時間的・動的情報をLoRAへ追加し、その獲得情報を解析することである。

現在のtemporal design候補は、frameごとにImage MAE encoderへ入力した後、past/currentのpatch tokenを因果的に融合し、current frameまたはfuture frameのmasked patch reconstructionへ利用する構成である。CLIPは主軸ではないが、画像事前学習ViTから動画へ適応する比較系と、学習後のsemantic probe候補として重要である。

本メモでは、2026-09-17時点で現在方針に近い先行研究を、特に次の3軸で調査した。

- image-pretrained modelからvideoへのparameter-efficient / self-supervised transfer
- CLIP事前学習ViTのvideo adaptation
- MAEとlate / delayed temporal fusion、causal / streaming temporal modeling

## 結論

現在方針に最も近い文献群は、単一の完全一致論文ではなく、次の複数研究の交差領域にある。

1. **SiamMAE (NeurIPS 2023)**
   - frameを独立にshared encoderで処理し、decoder側のcross-attentionで過去frameとfuture frameを融合する。
   - future frameを95% maskし、past frameをほぼそのまま与えてfuture patch reconstructionを行う。
   - 「MAE + frame独立encoder + 後段cross-attention + temporal reconstruction」という意味で、現在検討中のMAE + late fusionに最も直接的に近い。

2. **Recurrent Video Masked Autoencoders (RVM, CVPR 2026)**
   - frameごとのViT表現をrecurrent coreへ逐次入力し、状態を時間方向へ伝播する。
   - asymmetric maskingとpixel reconstructionのみで動画表現を学習する。
   - streaming / causal temporal stateという観点で現在方針に非常に近いが、LoRAへ情報を閉じ込める研究ではない。

3. **From Static to Dynamic / Co-Settle (CVPR 2026)**
   - image-pretrained encoderをfreezeし、軽量projectionだけを動画自己教師あり学習する。
   - MAE、CLIPを含む複数image-pretrained ViTで検証している。
   - 「画像モデルを保持し、小さい追加parameterだけを動画で自己教師あり適応する」という高レベル設定が現在研究と非常に近い。
   - ただし追加parameterはLoRAではなくprojectionで、目的もtemporal consistencyとsemantic separabilityの両立である。

4. **Remembering by Reconstructing (2026)**
   - MAE headとdomain-specific LoRAを組み合わせ、video stream上でincremental / online test-time adaptationを行う。
   - 「MAE + LoRA + streaming video」というキーワードの組合せが既に存在するため、今後の新規性主張では注意が必要。
   - 一方、目的はdomain incremental learning / domain retrievalであり、「画像MAE encoderのLoRAへ動画由来の時間情報を獲得させ、その中身を解析する」こととは異なる。

5. **Towards Data-Efficient Video Pre-training with Frozen Image Foundation Models (CVPR 2026 Workshop)**
   - image foundation modelをfreezeし、streaming videoを扱うrecurrent temporal moduleだけを学習する。
   - 「空間表現をimage modelに残し、時間表現のみを小さい追加部分へ学習する」という発想は現在研究に近い。
   - ただし動画情報の保存先が明示的なtemporal moduleであり、LoRA attributionとは異なる。

したがって、現段階で新規性を置くなら「MAEを動画へ使うこと」「MAEとLoRAを組み合わせること」「late fusionを使うこと」単独では弱い。より狭く、**image-pretrained MAEを保持し、causal / onlineなvideo self-supervised objectiveでLoRAに時間情報を追加し、どの時間的・動的情報がLoRAへ保持されたかを明示的に解析すること**が中心候補になる。ただし完全な先行研究網羅性を保証するものではなく、正式な新規性主張には追加の系統的検索が必要である。

## 特に重要な論文

| 論文 | 年/会議 | 画像事前学習 | temporal modeling / fusion | 学習信号 | LoRA / PEFT | 現在方針との関係 |
|---|---|---|---|---|---|---|
| Siamese Masked Autoencoders | NeurIPS 2023 | MAE系 | frame独立encoder後、decoder cross-attention | future masked-patch reconstruction | なし | MAE + late fusionの最重要参照 |
| Recurrent Video Masked Autoencoders | CVPR 2026 | image model利用ではなくvideo SSL | recurrent Transformer/RNN state | asymmetric masked pixel reconstruction | なし | causal / streaming temporal stateの最重要参照 |
| From Static to Dynamic (Co-Settle) | CVPR 2026 | MAE/CLIP等をfreeze | lightweight projection | temporal cycle consistency + separability | 軽量projection | frozen image model + video SSLが非常に近い |
| Remembering by Reconstructing | arXiv 2026 | pretrained model + MAE head | video stream online TTT | MAE reconstruction | domain-specific LoRA | MAE + LoRA + streamという強い近接例 |
| Towards Data-Efficient Video Pre-training with Frozen Image Foundation Models | CVPRW 2026 | frozen image foundation model | recurrent temporal module | video task / temporal learning | temporal moduleのみ | frozen spatial + trainable temporalという比較対象 |
| MAE-ST: Masked Autoencoders As Spatiotemporal Learners | NeurIPS 2022 | 原則video MAE training | spacetime tokensをjoint処理 | pixel reconstruction | なし | 標準的なvideo-MAE対照。late fusionではない |
| VideoMAE | NeurIPS 2022 | 原則video SSL | tube masking + video ViT | pixel reconstruction | なし | video masked modeling baseline |
| Masked Video Distillation | CVPR 2023 | image/video teacher | video student | masked feature prediction | なし | image spatial / video temporalの差のEvidence |
| ST-Adapter | NeurIPS 2022 | image-pretrained model | spatio-temporal adapter | supervised video task | Adapter | image-to-video PEFTの代表 |
| AIM | ICLR 2023 | image-pretrained ViT/CLIP系 | temporal/spatial/joint adapters | supervised action recognition | Adapter | frozen image modelへtemporal capacityを追加 |
| EVL | ECCV 2022 | frozen CLIP | frame spatial featuresをTransformer decoderで後段集約 | supervised video recognition | lightweight decoder | CLIP版late temporal fusionに近い |
| ViFi-CLIP | CVPR 2023 | CLIP | framewise CLIP + feature pooling | video-text/action supervision | full FT中心 | 最小構成のCLIP video baseline |
| X-CLIP | ECCV 2022 | CLIP | cross-frame attention | video recognition supervision | lightweight module | CLIPへ明示temporal modelingを追加 |
| CLIP4Clip | Neurocomputing 2022 | CLIP | parameter-free / sequential / tight aggregation | video-text contrastive | 非LoRA | frame-level CLIP後段aggregationの代表 |
| DiST | ICCV 2023 | CLIP等 | spatial encoderとtemporal encoderを分離 | supervised video task | lightweight temporal encoder | static/dynamic分離の構造的比較対象 |
| TM-Adapter | WACV 2026 | image-pretrained model | local/global temporal merge adapter | supervised video task | Adapter | 最新のvideo PETL比較候補 |
| Test-Time Training on Video Streams | JMLR 2025 | image model | recent frame windowを逐次利用 | MAE等のself-supervised TTT | full/TTT | online video self-supervisionの重要参照 |

## CLIPで事前学習済みのViT

### CLIPの意味

CLIPは、image encoderとtext encoderを約4億image-text pairでcontrastiveに事前学習し、画像と自然言語を共通embedding空間へ整列する。CLIPで事前学習済みViTとは、このcontrastive image-text objectiveで学習されたViT image encoderを指す。

MAEとの主な差は、MAEがmasked patchのpixel reconstructionを自己教師信号とするのに対し、CLIPは画像全体とtextの意味的対応を強く学習する点である。そのため、CLIPには標準MAE decoderや自然なpixel reconstruction objectiveは存在しないが、学習後のrepresentationをtext promptと比較して意味解釈しやすい利点がある。

### CLIPをvideoへ拡張する主要パターン

#### ViFi-CLIP

frameごとにCLIP image encoderを適用し、feature poolingしてtext embeddingとのsimilarityを計算する比較的単純な構成。full fine-tuningによってvideo domain gapを埋める。明示的な重いtemporal moduleなしでも動画性能が成立することを示すbaselineとして重要。

#### EVL

CLIP image encoderをfreezeし、frame-level spatial featuresを軽量Transformer decoderへ集約する。decoder内にlocal temporal moduleを置く。現在研究の「image encoderを維持し、後段で時間統合する」という構造のCLIP版に近い。

#### ST-Adapter / AIM

image-pretrained backboneをほぼ保持し、軽量なspatio-temporal adapterを挿入してvideoへ適応する。現在研究と同じparameter-efficient image-to-video transferの文脈だが、動画情報がadapter側に保存される設計である。

#### X-CLIP

CLIP内にcross-frame attentionを追加し、frame間情報を明示的に交換する。late fusionよりencoder内部に近いtemporal interactionの比較候補になる。

#### CLIP4Clip

frame-level CLIP representationをparameter-free / sequential / tightな方法で後段集約する。video-text retrievalが目的だが、「CLIP frame encoder + late temporal aggregation」の設計比較として参考になる。

### 現在研究への意味

CLIPを主軸にすると、動画学習後にtext encoderを固定probeとして使い、LoRA適応前後でaction/state-change promptとのsimilarityがどう変わったかを分析しやすい。一方、学習時のvideo self-supervised objectiveを別途設計する必要があり、text supervisionを使うと「動画そのものだけからLoRAへ時間情報を追加する」という現在の問いが変わる。

そのため、現行の「まずMAEを主軸、必要ならCLIPをsemantic evaluation / comparisonへ使う」という方針は、目的と手段の分離という点で合理的である。

## MAE + late fusion

### 重要: late fusionの定義

動画の文脈では、各frameを独立またはほぼ独立にencoderで処理し、encoder後のfeature / patch token / predictionを時間方向に統合する構成を広くlate / delayed temporal fusionとして扱える。ただし論文によって名称は異なり、SiamMAEは自らを「late fusion」と呼ぶことが中心ではない。

### SiamMAEが最重要

SiamMAEは2 frameをshared encoderで独立に処理した後、cross-attention decoderで融合する。past frameを保持しfuture frameを95% maskする非対称設計により、future reconstructionを解くためにcorrespondence / motionを利用しやすくする。

現在候補の

```text
past frame tokens ----+
                       +--> causal cross-attention / temporal fusion --> decoder --> reconstruction
current frame tokens -+
```

に近い。特に、current/future側tokenをquery、past側tokenをcontext (K/V) とする設計は直接参考になる。

### RVMはonline / causal側の重要参照

RVMはshared ViTで各frameを処理し、その表現をrecurrent coreへ逐次入力してstateを更新する。asymmetric masking + pixel reconstructionで学習し、長時間へ線形コストでstateを伝播できる。

現在研究でpast cacheを単純保存する案と比較し、明示的state updateを持つrecurrent temporal operatorという別案を与える。ただしtrainable recurrent coreをそのまま導入すると、時間情報がLoRAではなくcoreへ保存されるため研究目的上の交絡になる。

### 標準VideoMAE / MAE-STとの違い

VideoMAEやMAE-STはclip全体をspatiotemporal tokensとしてjointに処理する方向であり、framewise image encoderを保持して後段fusionする現在案とは異なる。

したがって、現在案の比較baselineとしては有用だが、architectureをそのまま採用すると「image pretrained representationを保持しLoRAへ何が追加されたか」という分析の分離が難しくなる。

### Co-Settleの示唆

Co-SettleはMAEとCLIPの双方をfrozen image encoderとして使い、その上の軽量projectionだけをKinetics-400で5 epoch自己教師あり適応して改善している。これは、Base image representationを大きく変更せず、少量の動画学習parameterだけでdynamic informationを追加できる可能性を直接支持する。

一方、その追加情報はprojection側に保存されるため、現在研究では「projectionをLoRAへ置き換えると何が変わるか」「LoRAだけでtemporal consistencyを得られるか」が比較論点になり得る。

## 現在設計へ取り込める知見

### 1. asymmetric maskingを強く検討する

current frameを通常MAEの75% maskで復元するだけでは、current frame自身のvisible patchから解けてpastを無視するshortcutが生じる可能性がある。

SiamMAE / RVMはfuture/target側を95%程度まで強くmaskし、past contextの利用を促している。現在研究でも、まず75%をbaselineとしつつ、temporal pathway依存性を見るために90〜95% target maskingを比較する価値が高い。

### 2. current/future query -> past K/V のcross-attentionは有力

patch tokenのspatial identityを保ったまま、current/future tokenをquery、past tokenをK/Vとするcross-attentionはSiamMAEおよびRVM系のdecoderと整合し、global poolingよりreconstruction objectiveへ接続しやすい。

### 3. trainable temporal moduleはLoRA解析を交絡させる

EVL、ST-Adapter、AIM、DiST、RVM等は、時間情報を学習する専用moduleを置くことで性能を出している。しかし現在研究では「LoRAが何を獲得したか」が中心であるため、強いtrainable temporal moduleを追加すると情報保存先の帰属が曖昧になる。

そのため少なくとも次のcontrolが必要になる。

- frame-independent MAE + LoRA
- temporal fusionありだがfusionはfixed / frozen
- trainable fusion + LoRA
- LoRAなし、trainable fusionのみ

これにより、動画性能改善やtemporal sensitivityがLoRA由来かfusion module由来かを切り分ける。

### 4. temporal objective利用をablationで証明する

normal orderだけでなく、no-past、shuffle、reverse、static-repeat等を使い、loss / representation / downstream probeが時間構造に依存しているかを確認する。単に動画frameで学習しただけでは「時間情報を学習した」とは言えない。

### 5. online cacheのstalenessは実験契約として扱う

LoRAを各stepで更新しながらpast featureをcacheすると、過去featureは古いLoRA parameterで生成されたstale representationになる。strict onlineでcacheをdetachするか、window内をcurrent LoRAで再encodeするかで計算量と学習意味が変わるため、temporal specで明示する必要がある。

## 新規性についての暫定整理

現時点の調査では、次の組合せを完全に同じ目的で行う代表論文は確認できていない。

```text
ImageNet pretrained MAE
  + Base encoder/decoder freeze
  + encoder Q/V LoRA only
  + causal / online video self-supervised learning
  + patch-token-level late temporal fusion
  + LoRA内部に追加されたtemporal/dynamic informationの解析
```

ただし、近接研究として少なくともCo-Settle、SiamMAE、RVM、Remembering by Reconstructingが存在するため、「MAE + LoRA + video」「frozen image model + video SSL」「MAE + late fusion」自体を新規性として主張しない。

今後の新規性確認では、特に

- video self-supervised LoRA adaptation
- LoRA parameter / subspace analysis for temporal knowledge
- image-to-video transferでのLoRAとadapter/projectionの比較
- online / streaming MAE adaptation with parameter-efficient tuning

を追加検索する。

## 主要文献URL

- MAE: https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html
- VideoMAE: https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html
- MAE-ST: https://arxiv.org/abs/2205.09113
- SiamMAE: https://proceedings.neurips.cc/paper_files/paper/2023/hash/7ffb9f1b57628932518505b532301603-Abstract-Conference.html
- RVM: https://openaccess.thecvf.com/content/CVPR2026/html/Zoran_Recurrent_Video_Masked_Autoencoders_CVPR_2026_paper.html
- Co-Settle: https://arxiv.org/abs/2603.26597
- Remembering by Reconstructing: https://arxiv.org/abs/2605.31108
- Frozen Image Foundation Models for Video: https://openaccess.thecvf.com/content/CVPR2026W/CV4Smalls/html/Orlova_Towards_Data-Efficient_Video_Pre-training_with_Frozen_Image_Foundation_Models_CVPRW_2026_paper.html
- Masked Video Distillation: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Masked_Video_Distillation_Rethinking_Masked_Feature_Modeling_for_Self-Supervised_Video_CVPR_2023_paper.html
- CLIP: https://proceedings.mlr.press/v139/radford21a.html
- ViFi-CLIP: https://openaccess.thecvf.com/content/CVPR2023/html/Rasheed_Fine-Tuned_CLIP_Models_Are_Efficient_Video_Learners_CVPR_2023_paper.html
- EVL: https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6715_ECCV_2022_paper.php
- X-CLIP: https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1398_ECCV_2022_paper.php
- ST-Adapter: https://proceedings.neurips.cc/paper_files/paper/2022/hash/a92e9165b22d4456fc6d87236e04c266-Abstract-Conference.html
- AIM: https://arxiv.org/abs/2302.03024
- CLIP4Clip: https://www.sciencedirect.com/science/article/pii/S0925231222008876
- DiST: https://openaccess.thecvf.com/content/ICCV2023/html/Qing_Disentangling_Spatial_and_Temporal_Learning_for_Efficient_Image-to-Video_Transfer_Learning_ICCV_2023_paper.html
- TM-Adapter: https://openaccess.thecvf.com/content/WACV2026/html/Hahm_TM-Adapter_Temporal_Merge_Adapter_for_Efficient_Global_Temporal_Modeling_WACV_2026_paper.html
- Test-Time Training on Video Streams: https://www.jmlr.org/beta/papers/v26/24-0439.html

## status / Authority

本メモは文献調査に基づくexploratory brainstormであり、specでも実装許可でもない。late fusion方式、mask ratio、temporal moduleのtrainability、current reconstruction / future predictionの採否は未決定である。