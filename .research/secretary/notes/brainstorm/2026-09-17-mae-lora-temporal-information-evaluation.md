---
date: 2026-09-17
project: sequential-video-lora-analysis
source_todo: null
topic: MAE LoRAが時間情報を学習したことの評価
status: exploratory
tags: [brainstorm, research, mae, lora, video, temporal-modeling, evaluation]
---

# MAE LoRAが時間情報を学習したことの評価

## 読み込んだ文脈

- `.research/lab/projects/sequential-video-lora-analysis/README.md`
- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-10-mtg.md`
- `.research/secretary/notes/brainstorm/2026-09-11-online-mae-late-fusion.md`
- `.research/secretary/notes/brainstorm/2026-09-11-mae-self-supervised-video-lora-pivot.md`
- `.research/secretary/notes/brainstorm/2026-09-17-mae-encoder-lora-targeting.md`
- 2026-09-17時点では日次TODO `.research/secretary/todos/2026-09-17.md` は未作成。

## 中心となる問い

Base MAEを基本的に固定し、動画学習で更新したLoRAについて、単なる動画ドメインへのappearance adaptationではなく、frame順序、過去文脈、動き、状態遷移などの時間的・動的情報を実際に利用・保持していることをどう示すか。

## 重要な考え方

reconstruction lossが下がるだけでは、時間情報を学習したEvidenceにはならない。current frame自身のvisible patchだけで復元できる可能性があるためである。

評価では、同じframe集合・同じtarget・同じmaskを保ったまま時間構造だけを壊し、性能が悪化するかを見る必要がある。さらに「動画モデル全体」ではなく「LoRAが担っている」と主張するには、LoRAを無効化したcontrolとの比較が必要である。

## 第一段階: temporal perturbation test

同一clipに対して次を比較する。

1. `normal`: 正しい時間順序。
2. `shuffle`: 同じframe集合をランダムに並べ替える。
3. `reverse`: 時間順序を逆転する。
4. `static-repeat`: 1 frameを繰り返し、見た目を残しつつ動きを消す。
5. `no-past`: current frameだけを使い、過去contextを除く。

current-frame masked reconstructionを使う場合、各条件でcurrent targetとmask乱数を同一にして、時間構造以外の差をできるだけ除く。

reconstruction lossを `L` とすると、例えば

```text
TemporalSensitivity(shuffle) = L_shuffle - L_normal
```

と定義できる。正しい順序を利用しているなら、平均的に `L_normal < L_shuffle` が期待される。

ただしこの差だけではLoRA由来とは言えない。

## 第二段階: LoRA attribution

同じモデル構造・同じ入力条件で少なくとも次を比較する。

- Base MAEのみ / LoRA disabled
- Base MAE + 学習済みLoRA
- 可能なら frame-independent MAE+LoRA control
- temporal learningを行ったMAE+LoRA

LoRAあり・なしで temporal sensitivity を比較する。

```text
Delta_temporal_with_lora
  = L_shuffle(with LoRA) - L_normal(with LoRA)

Delta_temporal_without_lora
  = L_shuffle(without LoRA) - L_normal(without LoRA)

LoRA_temporal_gain
  = Delta_temporal_with_lora - Delta_temporal_without_lora
```

`LoRA_temporal_gain > 0` が複数clip・複数seedで安定するなら、LoRA追加によって正しい時間構造への依存が増えたEvidenceになる。

## 第三段階: frame-independent LoRA control

同じ動画frameを使うが、各frameを独立画像としてMAE lossで学習したLoRAをcontrolとして用意する。

```text
A: pretrained MAE
B: frame-independent MAE + LoRA
C: temporal MAE + LoRA
```

BとCで画像ドメイン自体は近づくため、Cだけがshuffle / reverse / no-pastへ強く反応するなら、appearance adaptationだけでは説明しにくくなる。

## 第四段階: temporal probe / downstream evaluation

reconstruction lossだけでなく、学習済み表現をfreezeして時間依存タスクを解けるか確認する。

候補:

- temporal order prediction
- motion direction prediction
- before / after state classification
- action recognitionのうち、静止画だけでは区別しづらいクラス

特に `pick up` / `put down` のような逆向き操作や、Something-Something系の時間関係に依存するactionは、temporal representationのprobeとして分かりやすい候補。

probeを使う場合は、Base/LoRAをfreezeし、軽量なlinear probeだけを学習することで、表現に含まれる情報を比較しやすくする。

## 第五段階: LoRA固有性を強める追加ablation

- LoRAをゼロ化または無効化するとtemporal sensitivityが減るか。
- temporal moduleが別に存在する場合、そのparameterを固定した条件でもLoRAが時間依存性を持つか。
- LoRAを別datasetで学習したものへswapしたとき、temporal sensitivityの特徴が変わるか。
- layerごとにLoRAを無効化し、どの層がtemporal sensitivityへ寄与するかを見る。

## temporal moduleとの交絡

独立したtrainable temporal fusion moduleを追加し、そのmoduleとLoRAを同時に学習すると、時間情報がfusion moduleへ保存されている可能性がある。

したがって研究主張を「LoRAが時間情報を学習した」とするなら、次のどれかが必要。

- temporal operatorを固定し、LoRAだけをtrainableにする。
- temporal pathway自体をLoRAでadaptする。
- trainable temporal moduleを使う場合、LoRA on/off、temporal module on/offのfactorial ablationで寄与を分離する。

これを行わない場合、安全な主張は「システム全体が時間情報を利用した」までであり、「LoRAに保持された」とは断定しにくい。

## 最小評価セット候補

最初の診断としては以下が実装負荷と解釈のバランスが良い。

```text
Model A: pretrained MAE, LoRAなし
Model B: frame-independent LoRA
Model C: temporal LoRA

Input conditions:
- normal
- shuffle
- no-past
- static-repeat

Metric:
- current-frame masked reconstruction loss
- 各clipで同一maskを使用
```

期待するEvidenceは、Model Cでのみ `normal` が明確に有利で、時間構造を壊すと悪化すること。Model Bが同様に悪化しないことが重要。

## 統計的な確認

1 clipや1 seedだけでは科学的主張に弱い。

- 多数clipでpaired comparisonを行う。
- 同じclipのnormalとperturbedを対応付けて比較する。
- 平均差だけでなくconfidence intervalを出す。
- 学習seedを複数用意し、傾向が再現するか確認する。

## 外部研究との対応

- VideoMAEはvideo masked modelingで動画表現を学習し、高いmask ratioやtube maskingを利用する。
- `Masked Autoencoders As Spatiotemporal Learners` はMAEをspatiotemporal patch reconstructionへ拡張している。
- MotionMAEは、通常のvideo MAEがstatic appearanceに偏る可能性を問題として、motion structure predictionを追加している。この点は、reconstruction loss低下だけではdynamic information獲得を十分に示せないという今回の懸念と整合する。
- `Shuffle and Learn` はframeの正しいtemporal order verificationを自己教師あり信号として利用しており、時間順序を壊す比較がtemporal representationの検証として妥当であることを支持する。
- Something-Somethingは、日常物体への細かな操作を扱い、時間関係に依存するactionを含むため、強いtemporal probe候補になる。

## 現時点の有力方針

最初からVLMや動画生成だけで評価せず、まず因果を切り分けやすい以下を優先する。

1. normal / shuffle / no-past / static-repeatによるtemporal perturbation。
2. pretrained MAE / frame-independent LoRA / temporal LoRAの比較。
3. LoRA on/offによる寄与分離。
4. その後にtemporal linear probeやaction/state-change benchmarkで表現の意味を確認。
5. 最後にVLMや動画生成などの下流利用へ進む。

## 未解決事項

- 主objectiveをcurrent-frame reconstructionにするかfuture-feature predictionにするか。
- temporal fusionをpatch token levelで行うか別方式にするか。
- temporal operatorを固定するかtrainableにするか。
- 主評価datasetを何にするか。
- reconstruction loss以外の最初のprobeを何にするか。

このメモは探索記録であり、specまたは実装許可ではない。
