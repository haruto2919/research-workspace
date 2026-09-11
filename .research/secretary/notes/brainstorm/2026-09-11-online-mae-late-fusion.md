---
date: 2026-09-11
project: sequential-video-lora-analysis
source_todo: null
topic: オンラインMAEにおけるlate fusionの妥当性と設計
status: exploratory
tags: [brainstorm, research, mae, lora, video, online-learning, late-fusion, temporal-modeling]
---

# オンラインMAEにおけるlate fusionの妥当性と設計

## 出発点

画像データセットで自己教師あり事前学習されたMAEを出発点とし、Base MAEを基本的に固定した上でLoRAを追加し、動画データセットを時系列に入力して動画由来の動的・時間的情報を学習させたい。

候補として、各frameを画像MAE encoderで処理した後に時間方向へ統合するlate fusionが挙がった。今回の論点は、late fusionが研究目的に対して妥当か、またオンライン学習かつMAEで実現可能かである。

## 確認済みの関連知見

- AIMは画像事前学習ViTをfreezeし、軽量なadaptationでvideoへ拡張するimage-to-video transferの例である。
- ST-Adapterも画像事前学習モデルへspatio-temporal adapterを追加し、少数parameterでvideoへ適応する。
- VideoMAEおよびMasked Autoencoders As Spatiotemporal Learnersは、MAEをvideoのspatiotemporal masked modelingへ拡張する代表例である。
- これらは「画像モデルを保持しつつ時間方向の能力を追加する」ことや「videoでMAE型自己教師あり学習を行う」ことの妥当性を支持するが、今回の「online + image-pretrained MAE + LoRA + late fusion」と完全に同一ではない。

## 現時点の解釈

### Late fusionは有力だが最適解とは未確定

late fusionには、ImageNet事前学習済みMAE encoderを大きく変更せず利用できる利点がある。各frameを既存2D encoderで処理し、後段だけにtemporal pathwayを追加できるため、画像由来の表現と動画由来の追加処理を構造上分離しやすい。

一方で、fusionをdecoder後や最終embedding後に置くなど遅すぎる構成では、MAE reconstruction lossがtemporal pathwayを必要としない可能性がある。時間情報を学習させるには、fusion後の表現がmasked reconstructionまたは別のtemporal objectiveへ実際に寄与する必要がある。

### 研究目的との重要な衝突

今回の研究では「動画学習でLoRAに何が追加されたか」を解析することが重要である。late fusion moduleを別のtrainable moduleとして追加すると、動画の動的情報がLoRAではなくfusion module側へ格納される可能性がある。

したがって、単純な「trainable temporal fusion + trainable LoRA」は、動画性能を出す目的には合理的でも、「LoRAが動画由来情報を保持するか」を調べる研究では交絡になる。

## オンライン学習との両立

onlineを「未来frameを参照せず、時刻tではx_1...x_tだけを利用して更新する」と定義すれば、MAE + late fusionは実現可能である。

概念構成:

```text
x_t
 -> Image-MAE encoder
 -> z_t

past cache: z_{t-K}, ..., z_{t-1}
             + z_t
             -> causal temporal fusion
             -> decoder / prediction head
             -> self-supervised loss
             -> parameter update
```

必要条件:

- temporal fusionはcausalにする。future featureをattention key/valueへ入れない。
- 過去featureを一定長cacheする。
- sequence/video切り替え時にcacheをresetする。
- sequential loaderはshuffleせず時系列順を維持する。
- strict streamingならpast cacheはdetachして保持し、各stepで過去までの巨大な計算graphを残さない。
- past featuresを現在parameterで再計算しない場合、cacheは過去parameterで計算されたstale featureになる。この近似を許容するかは実験契約で明示する。

## MAE lossの候補

### Candidate A: current-frame masked reconstruction

現在frameのpatchをmaskし、現在のvisible patchと過去frame featureをfusionして現在frameのmasked patchを再構成する。

利点:
- 時刻tでtargetが得られるためstrict online updateが可能。
- future leakageがない。

懸念:
- current frameだけで復元できるとtemporal pathwayを無視する可能性がある。
- 高mask率、structured mask、past-only context ablationなどで時間情報への依存を確認する必要がある。

### Candidate B: future-frame prediction

過去〜現在からfuture frameまたはfuture featureを予測する。

利点:
- motion/state transitionを使う必要性が高く、dynamic learningの解釈はしやすい。

懸念:
- target frameが到着するまでoptimizer updateできないため1 step以上の遅延が生じる。
- strict immediate online updateとは異なるが、future frameを入力として先読みせず、到着後に更新するstreaming learningとしては成立し得る。

### Candidate C: temporal contrast / order objectiveをMAE lossへ追加

frame reconstructionに加え、近接frameの表現関係、順序、変化量などを自己教師ありで学習する。

利点:
- standard MAEが静的appearanceだけで解ける問題を補える。

懸念:
- objectiveが増え、何がLoRAへ入ったかの解釈が難しくなる。

## Late fusionの位置候補

1. encoder後のframe-level pooled featureを融合
   - 最軽量。
   - spatial情報を落とすためpixel reconstructionとの接続が弱い。

2. encoder後のpatch tokenを融合
   - MAE decoderへつなぎやすく、masked reconstructionとtemporal reasoningを両立しやすい。
   - token数と計算量が大きい。

3. encoder後にcurrent queryからpast patch tokensへcausal cross-attention
   - onlineと相性がよく、current reconstructionがpastを使う経路を明示しやすい。
   - 新しいtemporal moduleが主に学習するとLoRA解析の交絡になる。

## 現時点の有力方向

単純なtrainable late fusionを最終解とするより、次の比較が有力。

### Baseline 1: Image-MAE + LoRA, frame independent

動画frameを独立画像としてMAE学習する。時間情報を使わないcontrolとして用いる。

### Baseline 2: Causal late-fusion MAE

Image-MAE encoderをframeごとに利用し、past/current encoder tokensをcausal temporal fusionしてcurrent masked patchを再構成する。

### Candidate 3: temporal operatorのbase weightを固定しLoRAだけを学習

画像attentionの重みを時間軸にも再利用する、または固定したtemporal operatorへLoRAを付けることで、時間適応parameterをLoRA側へ寄せる。これにより「動画由来情報がLoRAに保持されたか」という研究目的とlate fusionを整合させられる可能性がある。

## 現時点の推奨

late fusionは、pretrained image MAEを保護しながらonline video adaptationを追加する初期候補として合理的。ただし、独立したtrainable temporal fusionを追加するだけでは研究目的に対して最適とは言えない。

第一に、frame-independent MAE+LoRAをcontrolとして成立させる。その後、causal late fusionを追加し、normal order / temporal shuffle / static repeat / no-pastなどのablationでtemporal pathwayの利用を確認する。

研究主張をLoRAへ集中させたい場合は、temporal fusionの学習parameterを別adapterへ逃がさず、固定temporal operator + LoRA、あるいはtemporal attention自体のLoRA adaptationを候補として比較する。

## 未解決事項

- onlineの定義をframe-by-frame update、chunk-by-chunk update、one-pass trainingのどこまで厳密に要求するか。
- current reconstructionとfuture predictionのどちらを主objectiveにするか。
- temporal fusionをframe-levelで行うかpatch-token levelで行うか。
- temporal fusionのparameterを学習対象に含めるか。含める場合LoRA解析との交絡をどう扱うか。
- LoRAをimage encoder q/vへ付けるだけでよいか、temporal pathway側にもLoRAを置くか。
- cache featureのdetachとstalenessをどのように実験契約へ定義するか。

## 関連文献

- AIM: Adapting Image Models for Efficient Video Action Recognition, ICLR 2023.
- ST-Adapter: Parameter-Efficient Image-to-Video Transfer Learning, NeurIPS 2022.
- VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training, NeurIPS 2022.
- Masked Autoencoders As Spatiotemporal Learners, 2022.

## 2026-09-11 13:46 JST 追記: late fusion後のMAE lossの意味

MAEはクラスラベルを教師にしない。maskした入力patchの元pixel値をtargetとして、decoderの再構成値との誤差を自己教師ありlossとして使う。標準MAEでは主にmaskされたpatchについてpixel-spaceのMSEを計算する。

16 frameをencoderに通してlate fusionする場合、fusionそのものではlossは決まらず、「decoderに何を復元させるか」を別途定義する必要がある。

候補は次の通り。

1. 16 frameすべてのmasked patchを復元する。
   - 各frameの元pixel patchがtarget。
   - lossは全frameのmasked patch reconstruction errorの平均。
   - ただし各frame自身のvisible patchだけで復元できると、temporal fusionを使わない可能性がある。

2. current frameだけを復元する。
   - 過去15 frame + current frameのvisible patchをfusionし、current frameのmasked patchをdecoderで復元する。
   - targetはcurrent frameの元pixel patch。
   - strict onlineと相性がよく、今回の有力候補。

3. future frameを予測・復元する。
   - 過去/currentからfuture frameやfuture featureを予測する。
   - temporal learningを強く要求できるが、標準MAEからobjectiveが変わり、target到着までupdateが遅延する。

重要なのは、ラベルがなくても元動画frame自体が教師信号になることである。例えばcurrent frameの75%をmaskした場合、encoder側にはその75%を見せないが、loss計算時には元のcurrent frameを保持しておき、decoderが復元したmasked patchと比較する。

また、MAE decoderはpatch token列と位置情報を前提にするため、16 frameを1本のglobal vectorへ単純平均してから標準decoderへ渡す設計は自然ではない。reconstructionを維持するなら、patch tokenのspatial identityを保ったままtemporal fusionする、あるいはcurrent-frame patch tokenをquery、past-frame patch tokenをcontextとしてcross-attentionする設計の方が接続しやすい。
