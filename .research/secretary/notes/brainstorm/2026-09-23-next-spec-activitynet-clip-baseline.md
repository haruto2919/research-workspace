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