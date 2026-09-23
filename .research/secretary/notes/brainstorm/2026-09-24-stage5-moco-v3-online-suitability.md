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
