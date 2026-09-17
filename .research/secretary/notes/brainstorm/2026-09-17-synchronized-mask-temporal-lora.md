---
date: 2026-09-17
project: sequential-video-lora-analysis
source_todo: null
topic: MAE+LoRAにおける時間方向同期マスクの有効性
status: exploratory
tags: [brainstorm, research, mae, lora, video, temporal-modeling, masking]
---

# MAE+LoRAにおける時間方向同期マスクの有効性

## 読み込んだ文脈

- `.research/lab/projects/sequential-video-lora-analysis/README.md`
- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-10-mtg.md`
- `.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-16-mae-single-image-one-step-training-smoke-spec.md`
- `.research/secretary/notes/brainstorm/2026-09-17-mae-direction-recap.md`
- `.research/secretary/notes/brainstorm/2026-09-17-mae-encoder-lora-targeting.md`
- `.research/secretary/notes/brainstorm/2026-09-11-online-mae-late-fusion.md`
- VideoMAE (Tong et al., NeurIPS 2022)

## 出発点

VideoMAEのTube Maskingのように、連続frame間で同じ空間patch座標をmaskすることで、現在検討中のImage MAE + LoRAでもLoRAが時間的・動的情報を学びやすくなるかを検討する。

## 確認済み事実

VideoMAEでは、動画の時間的冗長性・相関により、隣接frameの同じ場所にmasked contentのコピーが残るとreconstruction shortcutが生じ得ると説明されている。Tube Maskingは時間方向に対応する空間位置をまとめてmaskし、このinformation leakageを抑える目的を持つ。

現在の研究方針では、標準Image MAEへframeを1枚ずつ独立入力するだけでは、各frameのreconstruction lossが他frameに依存しないため、LoRAが時間関係を学んだとは主張できない。動画段階ではarchitectureまたはobjectiveで複数frameの関係を必要にする方針が既に整理されている。

## 中心的な整理

### 1. 同一clip内でmask座標を同期する案は有力な補助策

連続frameからなる1 window / clipの中で、同じ2D patch maskを各frameへ適用する方法は、VideoMAEのTube Maskingに近い発想として合理的である。

ただし、ここで同期するのは「同一clip内の時間方向」であり、学習全体を通して常に同じ絶対座標だけをmaskすることではない。後者は特定位置を恒常的に見せない空間biasを生み得るため避ける。mask patternはclip / stepごとに再サンプリングし、そのclip内ではframe間で共有する形が候補となる。

### 2. frame-independent MAEでは、同期maskだけではtemporal learningにならない

各frameを独立処理する場合、概念的なlossは次のように分解できる。

```text
L = sum_t L_MAE(x_t, M)
```

同じmask `M`を全frameに使っても、`L_MAE(x_t, M)`はそのframe `x_t`だけから計算できる。`x_{t-1}`や`x_{t+1}`への依存がなく、frame間の情報交換もない。

そのため、連続frameで似たgradientが生じる可能性はあっても、それはappearanceの類似・動画domainへのadaptationでも説明できる。これだけを根拠に「LoRAが時間情報を学んだ」とは言えない。

### 3. 同期maskが意味を持つのはcross-frame contextを使える構造と組み合わせた場合

例えばcausal late fusionを使う場合、

```text
x_{t-K}, ..., x_t
  -> 同じ2D mask Mをclip内で共有
  -> 各frameをImage MAE encoder + LoRAでencode
  -> past/current tokenをtemporal fusion
  -> current frameのmasked patchをreconstruct
```

とすれば、current-frame reconstructionのgradientがpast-frame contextを経由してLoRAへ流れる可能性がある。

概念的には、

```text
z_t = E_base+LoRA(x_t, M)
c_t = F(z_{t-K}, ..., z_t)
L_t = reconstruction(x_t[M], Decoder(c_t))
```

となり、`L_t`が複数frameのrepresentationへ依存するため、LoRA updateへ時間contextが反映される条件が生まれる。

## 重要な反例・注意点

### 同期maskは「時間を使わないと解けない」ことを保証しない

current frameのvisible patchだけで十分にreconstructionできる場合、temporal fusionがあってもモデルはpastを無視できる。そのため、同期maskはtemporal learningの十分条件ではない。

VideoMAEにおけるTube Maskingの主目的は、隣接frameの同一位置から単純copyするshortcutを抑えることであり、それ自体が時間順序・時間方向を直接教師するobjectiveではない。

したがって、同期maskで期待できるのは主に「単純copyを減らし、spatiotemporal contextを使う必要性を高める」ことであり、`before -> after`の順序や因果を学んだことまで保証しない。

### LoRAへ時間情報を帰属させるにはtrainable temporal moduleが交絡になる

別のtemporal fusion moduleもtrainableにすると、動画情報がLoRAではなくfusion moduleへ格納される可能性がある。

LoRAそのものを研究対象にするなら、候補として次を比較する。

- fixed / frozen temporal operator + Image MAE encoder LoRA
- temporal attention / cross-attention側へLoRAを置く
- trainable fusionを使う場合はLoRA-onlyとfusion-onlyのablationを行う

## 候補設計の比較

| 設計 | 時間情報をLoRAへ要求する強さ | 解釈 |
|---|---:|---|
| frame-independent MAE + frameごとrandom mask | 弱い | appearance/domain adaptation baseline |
| frame-independent MAE + clip内同期mask | 弱い | mask相関はあるがlossはframeごとに独立 |
| causal temporal fusion + random mask | 中 | cross-frame contextは使えるがmask shortcutが残る可能性 |
| causal temporal fusion + clip内同期mask | 中〜強候補 | same-position copyを抑え、past/current contextを使う圧力を高められる |
| temporal pathway自体にLoRA + clip内同期mask | 強候補 | temporal interactionのparameterをLoRAへ直接寄せやすい |

この表は探索上の評価であり、実験済みEvidenceではない。

## 現時点の有力仮説

> 連続frame内でmask座標を同期することは、Image MAE + LoRAのtemporal learningを補助する有力な方法である。ただし、frame間の情報交換がない構成では十分ではない。causal cross-frame modelingと組み合わせることで初めて、reconstruction lossが時間contextへ依存し、LoRAが時間的・動的情報を学習する圧力を作れる可能性が高い。

さらに、「固定mask」は全学習で同じ座標へ固定するのではなく、clipごとにmaskを再サンプリングし、そのclip内のframe間だけで座標を同期するのが妥当な候補である。

## 検証案

同じarchitecture・optimizer・データで、少なくとも次を比較する。

```text
A. frame-independent + random masks
B. frame-independent + synchronized mask
C. temporal fusion + random masks
D. temporal fusion + synchronized mask
```

その上でDについて入力時系列を壊す。

```text
normal ordered video
vs. frame shuffle
vs. static repeat
vs. no-past
```

Dが通常時系列で有利で、shuffle / static repeat / no-pastでその利点が失われるなら、単なるmask patternやappearance adaptationではなく、temporal context利用を示すEvidence候補になる。

LoRAへの帰属を評価する場合は、LoRA-only / temporal-module-only / bothのablationも必要である。

## 未解決事項

- temporal fusionをpatch-token levelで行うか、frame-levelで行うか。
- current frameのみをreconstructするか、window内の複数frameをreconstructするか。
- synchronized mask ratioをImage MAE標準75%から変えるか。
- temporal operatorをfreezeするか、LoRAをtemporal pathway側へ置くか。
- 「時間情報」の評価をmotion、temporal order、causal predictionのどの粒度で定義するか。

## 次アクション候補

1. `clip内同期mask`をtemporal objective候補として残す。
2. late fusion / temporal pathwayの具体構造を決める際に、同期maskとの組み合わせを比較候補へ入れる。
3. normal order / shuffle / static repeat / no-past ablationを、temporal informationの評価候補としてspec化前に詰める。

このメモは探索記録であり、specまたは実装許可ではない。
