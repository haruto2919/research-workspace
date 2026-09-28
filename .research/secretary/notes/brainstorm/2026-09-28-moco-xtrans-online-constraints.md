---
date: 2026-09-28
project: sequential-video-lora-analysis
source_todo: null
topic: MoCo x_trans augmentation and online constraints
status: exploratory
tags: [brainstorm, research, moco, augmentation, online-learning, streaming-video, lora]
---

# MoCoのx_trans augmentationでonline制約をどこまで緩和できるか

## 提案

current clip `x_t` に対し、全frameへ同じ変換を適用した

- RGB -> GBR channel permutation
- horizontal flip

を `x_trans` とする。

MoCoでは、

- query: `x_t`
- positive key: `x_trans`
- negatives: past clip keys in queue

とする。

## 結論

この設計はonline / single-pass条件をかなり扱いやすくするが、online学習固有の制約を全て消すわけではない。

### 緩和できるもの

1. future clipをpositiveに使う必要がない。
2. 別videoからpositiveを探す必要がない。
3. current clipだけでpositive pairを生成できる。
4. 1-stream sequential処理でもpositive constructionが成立する。
5. all-past negativesを許すなら、different-video-only policyによるnegative枯渇も回避できる。

### 残るもの

1. adjacent past clipsをnegativeにした際のfalse negative。
2. sequential inputのnon-IID / gradient correlation。
3. queue feature staleness。
4. EMA key encoderの追従遅れ。
5. forgetting / order dependence。
6. current masked-mean architectureではtemporal orderを直接学ばないこと。

## x_trans固有の注意

GBR channel permutationはImageNet-pretrained ViTに対して強い色分布shiftとなる可能性がある。
positiveとして固定的に使うと、モデルはchannel identityを捨てる方向へ強く学習する。

horizontal flipも左右方向を不変にするため、motion directionそのものを保持したい場合は目的と衝突する可能性がある。

同一clip内の全frameには同じ変換を適用し、frameごとに異なるaugmentationを入れて人工的なmotionを作らないことが重要。

## 比較候補

- raw vs standard temporally-consistent augmentation
- raw vs GBR only
- raw vs horizontal flip only
- raw vs GBR + horizontal flip
- fixed GBR permutation vs random channel permutation

評価はlinear probeだけでなく、

- positive similarity
- adjacent-clip similarity
- LoRA gradient / update norm
- motion scoreとの相関

を見る。

## 研究上の位置付け

x_transは「online制約を避けるaugmentation」というより、

> current sampleからpositiveを自己完結に作り、future dataを使わずcontrastive learningを成立させる設計

と表現する方が正確。

ただし、motion / temporal dynamicsをLoRAへ入れることは別問題なので、
VideoMoCo、CVRL、temporal contrastive objective等との比較が必要。

このメモは探索記録であり、specまたは実装許可ではない。


## 2026-09-28 specへの昇格

現在のユーザー指示により、x_trans（RGB→GBR + horizontal flip）、same-video past negatives、1-stream strict onlineの実装契約を次のapproved specへ昇格した。

- `.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-28-stage6b-strict-online-moco-xtrans-spec.md`

以後、この変更の実装契約は上記specを正本とする。


## 2026-09-28 追記: Stage 6A / 6Bを共通streaming MoCo実装へ寄せる案

### 問い

Stage 6Aを別training loopとして残すより、stream modeを選択可能にして共通化した方が実装としてきれいではないか。

### 現時点の有力案

Stage 6A / 6Bの差分は、MoCo本体ではなく主に次の3軸として表現する。

1. sample stream / scheduling
   - Stage 6A: 4-source round-robin
   - Stage 6B: 1-source strict chronological
2. key-view transform
   - Stage 6A: horizontal flip
   - Stage 6B: RGB->GBR + horizontal flip
3. negative policy
   - Stage 6A: different-sequence only
   - Stage 6B: all past keys

training loop本体は、ordered SequentialSample stream、warm-up key数、view transform、negative policyを受け取る共通関数へ寄せる。

### 推奨責務分離

- stream builder / scheduler:
  SequentialSampleをどの順番で供給するかだけを担当。
- view transform:
  current sampleからQuery / Key viewを作る。
- negative policy:
  old Queueからどのentryをnegativeとして読むかを決める。
- common MoCo training loop:
  warm-up -> Query/Key forward -> loss(old queue) -> backward -> optimizer -> EMA -> enqueue を担当。
- MoCo model:
  encoder / projector / queue / InfoNCE / EMA primitiveのみを保持し、stream iterationは持たない。

### Stage 6Aの残し方

Stage 6A専用algorithm実装を複製して保持するのではなく、
共通実装へ以下の設定を与えることで再現する。

- stream = round_robin
- source_count = 4
- warmup_count = 4
- transform = horizontal_flip
- negative_policy = different_sequence

Stage 6Bは同じ共通実装へ、

- stream = strict_single
- source_count = 1
- warmup_count = 1
- transform = gbr_horizontal_flip
- negative_policy = all_past

を与える。

### 実装上の注意

過度に汎用的なframeworkへしない。
今回実在する差分3軸だけを明示的にparameterizeし、未知の将来方式向けのplugin systemまでは作らない。

historical Stage 6A smoke CLIを残す必要がある場合は薄いcompatibility wrapperとして保持し、
内部では共通training logicを呼ぶ。
これにより既存baselineの再現性とcode duplication削減を両立できる。

### 現時点の判断

「Stage 6Aを別loopとして残す」より、
**Stage 6A / 6Bを同じstreaming MoCo engineの異なるprotocol設定として扱う**方が、
比較可能性、保守性、testの明確さの点で優れる。

これは探索中の設計案であり、既存approved Stage 6B specの変更はまだ行わない。
