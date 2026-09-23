---
date: 2026-09-24
project: sequential-video-lora-analysis
source_todo: null
topic: Stage 5 MoCo v3 とオンライン学習の適合性
status: exploratory
tags: [brainstorm, research, moco, moco-v2, moco-v3, online-learning, sequential-video, lora]
---

# Stage 5 MoCo v3 とオンライン学習の適合性

## 出発点

Stage 4でActivityNet 16 frames -> ViT Q/V LoRA -> masked mean -> one-step LoRA updateまで成立した。
Stage 5でMoCo baselineを検討するにあたり、MoCo v3がmomentum / EMAを使うか、
また逐次動画を時系列順に処理するオンライン学習に適するかを整理する。

## 確認済み事実

- MoCo v3でもmomentum encoderを使う。
- Query / online encoderはoptimizerで更新し、Key / momentum encoderは
  `theta_k <- m theta_k + (1-m) theta_q` のEMAで追従する。
- MoCo v3はv1/v2型のmemory queueを外す。
- MoCo v3論文ではqueueの利得がbatchが十分大きい場合（例: 4096）に小さくなるとしてqueueを除去し、
  current batch内のkeysをnegativeとして用いる。
- Query側にはprediction headがあり、key側にはない。lossは2 viewを入れ替えた対称形。

## オンライン学習との関係

ここでいうオンライン学習はMoCo文献のonline/query encoderではなく、
動画clipを時系列順に到着させ、その都度更新するstreaming / sequential learningを意味する。

### v3の利点

- queue管理が不要で実装が単純。
- 過去featureを長期間保持しないため、古いfeatureのstalenessを管理しなくてよい。
- momentum encoderによる安定化は残る。
- 複数動画から十分大きく多様なmini-batchを作れるなら逐次学習でも原理上利用可能。

### v3の弱点

strict onlineで1 clipまたは非常に小さいbatchを逐次処理する場合、
negativeがcurrent batchに限られるためcontrastive signalが弱い。
batch size 1ではstandardなin-batch negativeが存在しない。

また、連続clipのみでbatchを作ると互いに内容が近く、
意味的にはpositiveに近いsampleをnegativeとして扱うfalse negativeが増える可能性がある。

### v1/v2 queue型の利点

- current batchが小さくても過去keyをnegative dictionaryとして再利用できる。
- streamingで到着したpast clipsを自然に蓄積できる。
- momentum encoderによりpast keyとcurrent keyのrepresentation spaceの変化を緩やかにできる。

### v1/v2 queue型の注意

- 同一動画の近接clipをnegativeとして入れるとfalse negativeになり得る。
- distribution shiftやLoRA更新が大きい場合、古いqueue featureがstaleになる可能性がある。
- queueが「過去情報」でもあるため、オンライン条件の定義とfuture information非利用を明確にする必要がある。

## 今回のViT+LoRAへの解釈

有力候補:

```text
Query:
Frozen ViT + Query LoRA
  -> masked mean
  -> Query projector
  -> q
  -> gradient update

Key:
Frozen ViT + Key LoRA
  -> masked mean
  -> Key projector
  -> k
  -> gradientなし

Query LoRA / projector
  -> EMA
Key LoRA / projector

past key
  -> FIFO queue
```

base ViTは既にfrozenなので、EMA対象をLoRAとprojectorへ限定する案が自然。
これはoriginal MoCoそのものではなく、本研究用のadaptation候補。

## 現在の方向性

strict / small-batch sequential video learningを重視するなら、
MoCo v3が「EMAを使わない」から不向きなのではなく、
「queueを使わずcurrent batch negativesへ依存する」点が主な不適合要因になり得る。

そのためStage 5 baselineではv2型queueありMoCoが有力候補。
ただしqueue内negative policyを先に詰める必要がある。

## 未解決事項

- online learningをbatch size 1のstrict streamingとするか、time-ordered mini-batchとするか。
- 複数動画streamをparallelにbatch化してよいか。
- queueへ同一videoの近接clipを入れるか。
- temporal exclusion windowを設けるか。
- negativeをdifferent sequence_id中心にするか。
- EMA対象をLoRAだけにするか、projectorも含めるか。
- queue size / momentum coefficient / projector dimension。

このメモは探索記録であり、specまたは実装許可ではない。


## 2026-09-24 00:33 追記: Stage 5 queue negative policyの収束案

### 中心となる問い

Stage 5のnon-temporal MoCo baselineで、

- 同じ動画の過去clipをnegativeにするか
- 隣接clipを除外するか
- different videoだけをnegativeにするか
- queue sizeをどうするか

を整理した。

### 有力方針

Stage 5 baselineでは、**positiveは同一clipの2 view、negativeはdifferent sequence_idのみ**とする案を有力候補とする。

理由:

- 同一動画の近接clipをnegativeにすると、意味的・時間的に近いclipを強制的に離すfalse negativeが起きやすい。
- 同一動画の遠いclipもActivityNetのuntrimmed videoでは意味が同じ/異なるの両方があり、ラベルなしでは安全にnegativeと断定しにくい。
- Stage 5はtemporal objectiveを入れないnon-temporal baselineなので、same-video temporal distanceに依存したnegative policyを混ぜない方が解釈しやすい。
- CVRLの「same short videoの2 clipsをpositive、different videosをnegative」とする考え方とも整合する。

したがってStage 5では、同一sequence_idのkeyは距離に関係なくnegativeから除外する。
この方針では「隣接clip exclusion window」は不要で、same-video全体除外に包含される。

### same-video distant negativeの扱い

same-videoの十分離れたclipをnegativeとして使う案は棄却ではなく、後続ablation候補とする。

比較候補:

1. different-video-only negatives
2. same-video distant negatives（temporal exclusion windowあり）
3. same-video all negatives

これにより、negative policy自体がLoRA表現へ与える影響を後で測れる。

### Queue実装候補

Queue entryに最低限、

`(key, sequence_id, sequence_index / clip_start)`

を保持し、loss計算時にcurrent queryと同じ`sequence_id`をmaskする。

Stage 5初期候補としてqueue sizeは **K=4096** を採用候補とする。

理由:

- original MoCoの65536より小さく、逐次更新下で古いkeyのstalenessを抑えやすい。
- 小batchでも数千negativeを確保できる。
- queue memoryはprojected keyのみなので実装コストが小さい。
- 最終値ではなくengineering baselineとし、後で1024 / 4096 / 16384等をablation可能にする。

VideoMoCoが古いqueue keyの劣化をtemporal decayで扱っていることからも、長すぎるqueueを無条件に採用しない方がよい。

### strict sequential時の注意

1本の長い動画を連続処理すると、その動画のkeyがFIFO queueを占有し、
different-video-only mask後のvalid negativesが減る可能性がある。

Stage 5ではMoCo mechanicsの成立を優先し、複数sequence_idを含むsample streamで検証する。
後続のstrict sequential stageでは、

- valid negative countのログ
- sequenceごとのqueue占有率
- 必要ならper-sequence cap / enqueue間引き

を別途検討する。

### 現在の収束

Stage 5 baseline候補:

```text
Positive:
same clip, two augmentations

Negative:
different sequence_id only

Same-video keys:
queueには保持可能だがcurrent queryのnegativeからmask

Adjacent same-video clips:
negativeにしない

Same-video distant clips:
Stage 5ではnegativeにしない
後続ablation候補

Queue:
FIFO, metadata付き
K=4096を初期候補
```

これは探索上の有力候補であり、specではない。


## 2026-09-24 01:38 追記: Projectorを学習する意味とLoRAへの影響

### 確認

- 標準的なMoCo v2ではprojection MLPはtrainable。
- Query projectorはcontrastive lossからgradient updateされる。
- Key projectorはQuery projectorをEMAで追従する。
- ProjectorはMoCo成立の数学的必須条件ではない。identity / fixed projectionでもlossからLoRAへgradientを流すことは可能。
- ただし標準MoCo v2からは外れ、representation qualityやoptimization特性が変わる。

### LoRAへのgradient

Query側を
`h = f_theta(x)`（thetaはLoRA）、
`z = p_phi(h)`（phiはProjector）、
`L = InfoNCE(z,...)`
とすると、

`dL/dtheta = dL/dz * dz/dh * dh/dtheta`

となる。

したがってProjectorのJacobian `dz/dh` がLoRAへ流れるgradientの方向・大きさを決める。
Projectorを更新すると、このgradient変換もstepごとに変化する。

### 研究上の意味

trainable Projectorには2つの側面がある。

利点:
- MoCo v2として標準に近いbaselineになる。
- contrastive objective専用空間をProjector側へ分離でき、pre-projector featureを過度にcontrastive taskへ特化させにくい。
- 画像SSLではlearnable nonlinear projection headがpre-projection representation qualityを改善するEvidenceがある。

注意:
- loss改善の一部をProjectorが担えるため、「学習による変化はLoRAだけに蓄積された」とは言えない。
- Projectorが変わることでLoRAへ流れるgradient自体も変わるため、ProjectorはLoRA学習 dynamicsの一部になる。

### Stage 5候補

目的が「標準MoCo baselineの成立」なら、
- Query LoRA + Query Projectorをgradient update
- Key LoRA + Key ProjectorをEMA
- Base ViTはfreeze
が有力。

目的が「LoRAだけにcontrastive objectiveを直接担わせる」なら、
- Projectorなし（identity）
- または固定Projector
が解釈しやすいが、標準MoCo v2 baselineではなくなる。

現段階では、まず標準寄りのtrainable Projectorを採用し、
LoRAの情報評価はProjectorを捨てたpre-projector clip featureとLoRA parameterで行う案が有力。
必要なら後続ablationで `trainable projector vs identity/no-projector` を比較する。

この追記は探索記録であり、specではない。
