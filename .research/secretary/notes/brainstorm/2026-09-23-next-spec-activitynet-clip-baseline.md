---
date: 2026-09-23
project: sequential-video-lora-analysis
source_todo: null
topic: 次specのscope - ActivityNetからorder-invariant clip featureまで
status: exploratory
tags: [brainstorm, research, activitynet, vit, late-fusion, clip-feature, next-spec]
---

# 次specのscope - ActivityNetからorder-invariant clip featureまで

## 出発点

- frozen `ViTFrameEncoder` による `[B,768]` CLS feature抽出は実装済み。
- 50Salads `SequentialSample` → valid frame → ViT → `frame_features [T,768]` は実装済み。
- ActivityNet v1.3 AdapterはSequential Loaderの `ActivityNet` branchに実装済みで、full inventory・183 tests・実動画decode smokeまで確認済み。
- ActivityNet → ViT接続とclip aggregationは未実施。

今回の問いは、次のimplementation specでどこまでを1つの変更単位にするか。

## 候補

### A. ActivityNet → ViT接続だけ

`ActivityNet -> SequentialSample -> ViTFrameEncoder -> frame_features [T,768]`。

最小だが、50Saladsですでにbridge自体は実証済みであり、次のspecですぐaggregationが必要になるため作業単位が細かすぎる。

### B. ActivityNet → ViT → masked mean clip featureまで

`ActivityNet -> SequentialSample -> frame_features [T,768] -> masked mean -> clip_feature [768]`。

masked meanはvalid frameのみを平均し、frame順を使わない order-invariant late-fusion baseline とする。

利点:
- 次のLoRA / MoCoへ渡す `clip_feature [768]` のcontractまで固定できる。
- ActivityNet integrationと最初のvideo-level representationを1つの目的にまとめられる。
- temporal moduleを入れず、順序を使わないbaselineを明示的に作れる。

### C. LoRAまで含める

採用しない。dataset integration、aggregation、LoRA target/rank/alpha/dropout/freezeを同時に増やし、failure sourceの切り分けが悪くなる。

### D. MoCoまで含める

採用しない。query/key、EMA、projector、InfoNCE、queue等の独立した設計判断が多すぎる。

## 現在の収束

次specは候補Bを最有力とする。

目的: ActivityNet v1.3 training subsetの1 chunkを、既存Sequential Loaderとfrozen ViTで処理し、valid frameだけからorder-invariantな `clip_feature [768]` を生成する。

想定pipeline:
`ActivityNet -> ActivityNetAdapter -> SequentialSample [T,3,H,W] -> ViTFrameEncoder -> frame_features [T,768] -> masked mean -> clip_feature [768]`。

## 次specに入れる候補

- ActivityNet Adapter実装revisionを明示的にpinする。
- splitは `training`。
- smokeは `frames_per_chunk=16` のcontiguous frames。
- annotation label / segmentはmodel inputへ使用しない。
- valid frameだけViTへ入力し、padding feature rowはzero。
- masked meanはvalid rowのみ平均。
- `clip_feature.shape == [768]`、finiteを確認。
- all-invalid maskはerror。
- synthetic testでpaddingを無視した平均が期待値と一致することを確認。
- 同じvalid feature集合をpermutation / reverseしてもmasked mean出力が同じであることを確認し、order-invariant baselineであることを明示。
- ViT backboneはfrozen。
- smokeはeval + no_grad。backward / optimizer / parameter updateなし。
- 実ActivityNetの先頭1 chunk程度の短時間smokeを必須とする。

## 対象外

- LoRAとそのhyperparameter。
- MoCo、query/key、EMA、projector、InfoNCE、queue。
- optimizer / backward / training loop。
- sequential vs shuffle学習比較。
- GRU / LSTM / temporal Transformer / order-aware attention。
- CLIP-ViT。
- ActivityNet annotationを使うdownstream評価。
- B>1 training semantics。
- 最終temporal sampling policy。

## 実装構造の有力候補

`frame_features [T,D] + valid_mask [T] -> MaskedMeanClipAggregator -> clip_feature [D]` の小さい責務を切り出す。

大きなVideoEncoder hierarchyや汎用temporal frameworkはまだ作らない。Stage 2 bridgeの処理は必要最小限だけ再利用する。

## Success Criteria候補

1. pinned ActivityNet Adapterからtraining sourceを取得できる。
2. 先頭chunkを `SequentialSample [16,3,H,W]` として取得できる。
3. valid frameのみをprocessor / ViTへ渡せる。
4. `frame_features [16,768]` を得られる。
5. invalid rowはzero。
6. masked meanで `clip_feature [768]` を得られる。
7. clip featureがfinite。
8. paddingを平均分母へ含めない。
9. synthetic testで期待平均値と一致する。
10. permutation / reverseでclip featureが一致する。
11. all-invalid maskをerrorにする。
12. ViT backbone trainable parameter数が0。
13. backward / optimizer / parameter updateを実行しない。
14. annotation label / segmentをmodel inputへ使わない。

## 次段階

- Stage 4: ViT + LoRA、base freeze、LoRAのみtrainable、1-step update smoke。
- Stage 5: MoCo mechanics。
- Stage 6: ActivityNet + sequential input + ViT + LoRA + MoCo統合。
- その後にsequential / shuffle / reverse / static-repeat等のcontrolを設計する。

## 未解決事項

- ActivityNet Adapterを `ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0` として直接pinするか、master統合後のcommitを使うか。
- implementation branch名とbase commit。
- aggregationを小さい `nn.Module` とするかhelperとするか。LoRA/MoCoでgradient pathを再利用するなら `nn.Module` が有力。
- Stage 2 bridge処理をどこまでhelperへ切り出すか。大きな抽象化は避ける。

このメモは探索記録であり、specまたは実装許可ではない。

## 2026-09-23 追記: Hydra移行完了後の次段階

### 最新状態

- training configのargparse -> Hydra/YAML移行は実装・短時間検証まで完了したものとして扱う。
- research codeの現在のGitHub基準は `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` の
  `3c0e40e86924ceb38931c2d5214ecf3251a8f99d`。
- ActivityNet Adapterの現在の基準は
  `tamaki-lab/2026_09_ishikawa_sequential_loader@ActivityNet` の
  `19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`。
- 50Saladsでは `SequentialSample -> frozen ViT -> frame_features [T,768]` のbridgeを確認済み。
- ActivityNet -> ViT接続とclip-level representationは未実装。

### 次に進める最有力Stage

次の独立specは、Hydra移行とは切り離して次を対象とする。

```text
ActivityNet training subset
  -> ActivityNetAdapter
  -> SequentialSample [T,3,H,W]
  -> valid frame extraction
  -> AutoImageProcessor
  -> frozen ViTFrameEncoder
  -> frame_features [T,768]
  -> valid_maskを使ったmasked mean
  -> clip_feature [768]
```

初期chunkは `frames_per_chunk=16` のcontiguous framesを用いる候補を維持する。

### なぜここまでを先に行うか

- ActivityNet Adapterは完成しているが、研究code側のViT経路へまだ接続していない。
- MoCo等のcontrastive SSLを実装する前に、query/keyへ渡すvideo-level feature contractが必要。
- masked meanなら時間順序を使わないため、最初のorder-invariant late-fusion baselineとして解釈しやすい。
- LoRA、MoCo、dataset integrationを一度に入れず、failure sourceを分離できる。

### このStageで主張できること / できないこと

確認できる:
- ActivityNet動画を既存Sequential Loader経由でViT featureへ変換できる。
- valid frameだけから1 chunkのfixed-size feature `[768]` を生成できる。
- paddingをmeanへ混入しない。
- ViT backboneをfrozenのまま利用できる。

まだ主張できない:
- LoRAが動画情報を学習した。
- 時間順序を表現した。
- 動的情報を獲得した。
- sequential inputがshuffleより優れている。

masked meanはpermutation invariantなので、この段階は明示的にtemporal baselineではなく
order-invariant clip representation baselineとする。

### その後の順序

```text
Stage 3
ActivityNet -> ViT -> masked mean -> clip_feature [768]
        ↓
Stage 4
ViTへLoRA注入
base frozen / LoRA only trainable / 1-step update smoke
        ↓
Stage 5
MoCo mechanics
query / key / EMA / projector / InfoNCE / queue等
        ↓
Stage 6
ActivityNet + sequential input + ViT + LoRA + MoCoを統合
        ↓
Stage 7
ordered vs shuffle等のcontrol
        ↓
Stage 8
temporal-specific control / LoRA parameter・feature解析
```

LoRAやMoCoをStage 3へ混ぜない方針を維持する。
