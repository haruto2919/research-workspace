---
date: 2026-10-08
project: sequential-video-lora-analysis
source_todo: null
topic: stage6b-v2-learning-curve-interpretation
status: exploratory
tags: [brainstorm, research, streaming-moco, stage6b-v2, comet, stability, evaluation]
---

# Stage 6B-v2 MoCo: 学習途中の結果と研究上の解釈（2026-10-08）

## 相談の出発点とScope

ユーザー依頼: `stage6b-v2__moco__run1` の現状の結果から分かることと考察。

- 対象 Comet experiment: `stage6b-v2__moco__run1` / `e1df2d65b1d64b94b5c80a48473d017f`。
- 実装: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`, commit `605f2afb95e1dda541208a46d69ce46acd83b335`, Comet provenance dirty=false。
- 科学設定: ActivityNet v1.3 training 10,024動画、16 frames/chunk、seed 0、AdamW lr=1e-3、weight_decay=0、strict_single、gbr_horizontal_flip、all_past、Queue 4096、EMA 0.999、temperature 0.07、ImageNet pretrained `google/vit-base-patch16-224` + Q/V LoRA。
- 根拠コード:
  - `conf/moco/stage6b_v2.yaml`
  - `training/streaming_moco.py`
  - `training/streaming_moco_full.py`
  - `self_supervised/moco/vit_lora_moco.py`
  - `integration/sequential_moco.py`
  - `training/moco_protocol.py`
- 参照 spec: `.research/lab/projects/sequential-video-lora-analysis/specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md`。
- これは**学習途中のCometログからの探索的解釈**であり、実装妥当性の保証、成果物検査、学習性能の確証ではない。

## 観測済み事実（Comet読み出し）

Comet `get_experiment_summary` は最大 step 413,000、処理済み動画 1,988、loss 4.736603、positive similarity 0.660228、Query LoRA grad norm 24,267.943 を返した。一方、同日に取得した履歴の方が後まであり、loss / similarity / gradはstep 414,600、processed videosはstep 414,700で動画1,996を記録した。要約と履歴の更新タイミング差があるため、**一つの同期した「最新値」とはみなさない**。

`moco/loss` 等は 100 optimizer updatesごとにログされ、loss/positive similarity/LoRA勾配ノルムは直前100 stepsの平均。4,146点（100～414,600）を解析した。以下、lossとsimilarityは区間平均、勾配は記録値の区間中央値。

| updates | loss平均 | positive similarity平均 | Query LoRA grad norm中央値 |
|---|---:|---:|---:|
| 20k–60k | 4.144 | 0.505 | 0.00996 |
| 60k–120k | 4.228 | 0.700 | 10.06 |
| 120k–220k | 4.369 | 0.677 | 67.64 |
| 220k–320k | 4.544 | 0.670 | 809.96 |
| 320k–400k | 4.564 | 0.662 | 3659.99 |
| 400k–415k | 4.812 | 0.673 | 41837.58 |

- Queueはstep 4,100時点で4,096に達し、その後の記録ではすべて4096。valid negativesも4096。
- 最大の100-step平均勾配ノルムはstep 318,400で約876,657（そのときの100-step平均lossは約4.16、positive similarityは約0.599）。step 361,500は約773,993。
- 後半で高い勾配ノルムの記録が持続的に増加している。単発の外れ値だけではない。
- 直近のQueue unique sequence IDsは概ね20件前後で、直近の動画がnegativeを多く占め得る。ただし同一動画のKeyの割合自体はこの指標から判別できない。
- 実装では勾配ノルムをQuery LoRA parameter gradientsのglobal L2 normとして測定、100-update平均でCometに記録する。finite/nonzeroチェックは通るかぎり継続する。
- 損失はステップを通じて有限であり、少なくともログされた点では明白なNaN/Infは観測されない。
- Cometで確認できた本projectのexperiment一覧には MoCo full run と別seedのsubset runの計2件。今回対象のfinal Query LoRA vs Base ViTのLinear Probe比較結果は含まれていない。

## 解釈・推論

1. **処理系の継続動作の支持**: 10万step超の入力、更新、Queue運用が実行された証拠ではあるが、研究上の正しさと表現品質の検証ではない。
2. **InfoNCE lossの意味**: 初期Queue充填までloss値はnegative数変化の影響を受ける。一方、Queue充填後も20–60kの平均4.14から400k以降平均4.81に増加する。非定常なストリーム・負例難易度・モデル内部変化で説明可能で、単調減少を成功条件にできない。
3. **positive similarity**: 初期約0.5から後半約0.65–0.7へ上昇。同一chunkのRGBとGBR+HFlipを近づける傾向と整合。しかしnegative分離、非崩壊性、静的/動的情報、下流精度は不明。
4. **勾配ノルムの持続的増大は最優先で解明**: AdamWでは勾配ノルムと実更新量が一対一対応しないため、勾配の大きさだけでモデル発散を断定できない。だが中央値まで桁単位で増えており数値的健全性への懸念が強い。
5. **時間情報の獲得は未証明**: frame-wise ViT + masked meanはclip内順序の置換に不変。same-chunk transformed viewsがpositive、same-video pastもnegative。順序・動きの表現学習を学習指標だけから示せない。

## 未検証仮説（採否未決）

### H1: 正規化前Projector特徴のノルム縮小
Query特徴は `F.normalize(self.query_projector(clip_feature), dim=-1)`。正規化前のベクトルがゼロ近傍に近づくと逆伝播の微分が大きくなり得る。有限のlossと巨大な勾配が共存している点と整合するが、実測ノルム未確認。対立仮説は勾配経路の他の層/LoRA変化。

### H2: 同一動画の過去chunkをnegativeにするfalse-negative / hard-negative作用
strict_single + all_pastでは時間的近接chunkを否定例として押し離し、同種の場面・動作の表現を分離させる可能性。だが実際のsame-video negative比率/negative similarityが未記録のため機序未確証。

### H3: 高い学習率または長時間オンライン更新によるパラメータドリフト
AdamW lr=1e-3、schedulerなし、単一通過。下流特徴劣化やLoRA weight norm増加の懸念。ただしパラメータ値・AdamW実更新ノルムの比較がないため未検証。

### H4: データ内容・動画境界による非定常性
特定動画や背景/動作構成の変化によりloss・gradientが揺らぐ。step 318,400 / 361,500等で動画IDやclip indexとの紐付けを調べ、他のstepとの差を見る必要がある。

反例/代替説明:
- 勾配が大きくてもAdamWによりパラメータは有限かつ有用な特徴が得られる可能性。
- Positive similarityが高いだけではcollapseも成功も断定できない。
- 非定常データではlossの上昇は即座に悪化を意味しない。

## 次に得るべきEvidence（推奨順、未承認）

1. **破壊的変更なしのdiagnostics**: 既存checkpoint/snapshotとローカルauditログを調べ、ステップ別/層別Query LoRA grad・weight norm、Query Projector出力のL2 norm（正規化前）、正規化後variance、Query/Key cosine、actual AdamW update normを確認。Comet ReaderだけではcheckpointやSSH上ファイルを取得できない。
2. **negative分析**: same-video / different-video negative個数、cosine similarity分布、pos-vs-hardest-negative margin、Queue keyの生成step（staleness）を見る。
3. **表現評価**: 固定のActivityNet評価setでBase ViT、途中Query LoRA（保存済みsnapshotが実際に存在するなら）、最終LoRAを同一Linear Probe protocolで比較する。正規specのfinal評価はTop-1とMacro class accuracyの3-seed比較。
4. **時間順序のcontrol**: 同一frame集合のforward/reverse/shuffleの比較ではmasked meanの順序不変性という表現上の限界がある。stream update order control、frame orderを利用できるarchitectureとの比較などを区別する。
5. **必要な場合の独立したcontrolled run**: データ順とseedをそろえた比較にてlr減少、gradient clipping、negative policyの変更等を段階的に検証。既存production runへ途中で科学設定を適用しない。

## 現状の方向性

- **最有力**: 勾配急増の原因究明と表現品質評価を別トラックで先行し、長時間の途中状態をそのまま「収束」「動き学習成功」と報告しない。
- **保留**: learning rate変更、gradient clipping、negative policy変更、別architectureへの移行。新しい実験の条件と判断指標の確定が必要。
- **根拠なしに棄却すべき主張**: LoRAが時間情報を確実に獲得した、MoCoが発散した、positive similarity上昇が表現品質向上を示した、など。

## 関連URL

- Git commit: https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/commit/605f2afb95e1dda541208a46d69ce46acd83b335
- YAML: https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/605f2afb95e1dda541208a46d69ce46acd83b335/conf/moco/stage6b_v2.yaml
- Gradient and train step: https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/605f2afb95e1dda541208a46d69ce46acd83b335/training/streaming_moco.py
- Query normalization: https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/605f2afb95e1dda541208a46d69ce46acd83b335/self_supervised/moco/vit_lora_moco.py
- Approved spec: https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md
