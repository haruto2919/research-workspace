---
date: 2026-10-21
project: sequential-video-lora-analysis
source_todo: null
topic: full-vit-vs-frozen-vit-lora-moco-before-sequential-loader
status: exploratory
tags: [brainstorm, research, vit, moco, full-fine-tuning, lora, representation-evaluation, sequential-loader]
---

# Full ViT＋MoCo vs Frozen ViT＋LoRA＋MoCoを先に性能比較し、後でSequential Loaderを導入する段階計画（探索）

## 相談とAuthority

2026-10-21、ユーザーが「ViTの重み可変+mocoとFrozen ViT-LoRA+ mocoで性能比較を一旦してみて，その後シーケンシャルローダを導入の順で行おうと考えていたのですが，どの手順で行っていけばよいかを考えてください」と依頼した。本メモは**現在のユーザーによる比較・統合順序の方針案に対する壁打ち**であり、正式spec、GPU実験承認、GitHub研究コード変更承認ではない。今後のspec化に必要な成功条件と不確実性を明示する。

## 読み込んだ根拠

- [2026-10-15 MTG](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-10-15-mtg.md)：ResNetでMoCo基準確認→ViT置換→LoRA追加→Sequential Loader導入という**一因子ずつ段階統合**の方針。異常原因（MoCo loss振動／LoRA勾配急増）が追跡できる対照が必要。ユーザーの新しい順序案は、画像段階でFull ViT vs ViT-LoRAを先に比較する点で追加的に具体化されている。
- [2026-10-08 MTG](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-10-08-mtg.md)：本題はActivityNet segmentの768-dim特徴Linear Probe（validation Top-1）、元のViT+LoRAが動作情報を獲得したか。以前の統合結果は不安定で、Linear Probe性能の研究上の解釈は保留。Frame-wise CLS＋masked meanはchunk内フレーム順序を保持する集約ではない。
- [E5 5epoch Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-21-meta-moco-v2-e5-imagenet-5epoch-pair-parity-evidence.md)：ImageNet-1K ResNet50 MoCo v2のMeta参照 vs 自作が各25,020step、logged metrics bitwise、251/251 input hash、全epoch canonical state一致、Loss epoch平均9.7178→7.6079。**MoCo計算機構の実装再現性であり、ViTへの転用や表現品質の証明ではない**。
- [過去の五段階案](./2026-10-15-five-stage-lora-moco-research-task-plan.md)、[MoCo統合ロードマップ](./2026-10-15-post-moco-gate-c-integration-roadmap.md)：以前はprojector-only Frozen ViT＋MoCoを最初の工程対照とする提案。正式採用されたFull vs LoRAの画像性能比較specはこの資料にはない。
- 2026-10-21 GitHubの研究コード`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` commit `7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62`を静的確認：`self_supervised/moco/ddp_moco.py`はResNet50 `ImageNetResNet50`と汎用`DistributedMoCo(backbone)`を持つ。`standalone_moco.py`は`backbone.feature_dim`を要求し、Query/backbone+MLP ProjectorとKey EMAコピーを持つ。`model/vit/vit_lora_frame_encoder.py`は事前学習済み`ViTModel`のQ/V LoRA、base frozen、CLS 768、`vit_frame_encoder.py`はFrozen ViT CLS 768。自作LoRA単体のsupervised CIFAR10 smokeは既存。ResNet画像MoCo正式runはResNet固定、今回必要なFull ViTおよびLoRA ViTの共通ImageNet画像MoCo学習・評価の完全な既存runはGitHub上で未確認。研究サーバのdirty/unpushed状態は未確認。
- 旧Stage5 ActivityNet ViT-LoRA MoCoの1step工学実験はResearch Workspaceの履歴として存在するが、現在のGitHub `dev`のコードパスを同じ完成済み学習・性能比較コードと扱わない。

## 中心となる研究の問い

1. 同一初期事前学習済みViTと同一画像MoCo設定から、**Full Fine-Tuning**と**Frozen Backbone＋Q/V LoRA Fine-Tuning**は、学習後の表現品質・計算コストでどう異なるか。
2. 初期の高い事前学習特徴品質と、MoCo後の学習による改善をどう区別するか。
3. その後動画Chunkへ入力を変え、Sequential Loaderを導入しても、追加LoRAとMoCoの工程が維持できるか。
4. 画像性能が得られても時間順序・動作情報を学習できたとは限らない。将来のTemporal evidenceをどの対照で取るか。

## 最有力の段階計画（採用候補）

| 段階 | 内容 | 最小合格条件・Evidence |
|---|---|---|
| P0 比較契約/評価先行 | データのsplit、同一pretrained ViT checkpoint、CLS位置、224px/two-view augment、MoCo queue/EMA/T/Projector共通、optimizer対象・学習量、eval前後のtrain/val定義、seed、保存形式を確定 | 2群間の差分は原則学習可能範囲、完全同一入力・同一初期状態に対応付けられる。性能指標を先に決定 |
| P1 共通実装とCPU/smoke | 2群（必要なら3群）を同じMoCoエンジンで動作。A: Full ViT＋Query Projector更新/Key全学習対象をEMA。B: pretrained Base完全freeze＋Q/V LoRAとQuery Projectorのみ更新/Key LoRA＋ProjectorをEMA | Forward CLS[batch,768]→MLP→128、InfoNCE/Queue/EMA、optimizer IDs/gradients/updates、AのViT基盤更新、BのViT基盤bitwise不変、Key grad0、全有限、ckpt roundtrip。LoRA Bゼロ初期では最初のA勾配0を許容 |
| P2 GPU pilot・資源診断 | 両群で同一固定small image stream、1step→短時間20–100step→100–500step程度。GPU 20GiB×2でBatch/VRAM/速度/OOMを実測。十分なstep/epochの本番条件はpilot後固定 | 勾配非finiteなし、学習・保存・復元、2実装とも目標のmemory headroom。Queue初期置換にかかるstepをK/globalBatchで記録し、短期Lossの上下で学習成功/失敗を判断しない |
| P3 通常画像で公平な比較 | 固定データ（ImageNet train subset＋valなど、予算が厳しければCIFAR10 smoke後に小規模固定subset）、同一入力two-view・sampler・augmentation seed、同一optimizer step budget、同一評価条件で2群学習。学習率など実装間の適切な範囲は**対等な探索予算**で選ぶ | 主評価は**同じheldout評価におけるLinear Probe Top-1**、補助kNN／正例retrieval rank・margin／feature variance・effective rank。CLS 768のbackbone表現とMoCo Projector出力128の評価は別々に記録。各群のpretrained-before vs afterとFull vs LoRAを併記 |
| P3' 原因切り分け用推奨対照 | C: Frozen ViT＋Projector-only MoCoを低予算の補助対照として追加 | LoRA適用による改善と、Projectorだけでも生じる改善を区別。追加run予算が厳しければスモークや固定少量subsetの比較だけでも価値がある |
| P4 動画固定Chunk（Loaderなし） | 画像で検証したモデルを使い、ActivityNet固定16-frame chunk→frame CLS[16,768]→masked mean [768]→MLP Projector128→MoCoへ変換。2-viewの動画拡張と学習初期化・checkpoint継続/新規の扱いを別specで固定 | valid_mask/pad除外、chunk source ID/timestamp、CLS/grad/Query LoRA update・EMA/Queue・loss有限、オフライン固定入力のartifact記録。**時系列学習の主張はしない** |
| P5 Sequential Loader差分導入 | P4と同じ固定raw chunks・初期state・seed・two-view・queueをSequential Loader経由で再演 | Raw frames/indices/timestamps/valid maskが一致、CLS・loss・grad・update・EMA・Queueが数値契約内一致。stream order、動画境界reset、no-future leakage／past queue監査。差分原因をLoaderだけに特定 |
| P6 Strict Online＋研究的評価 | 少量multi-chunk→短時間pilot→online継続学習、最終的にActivityNetの固定segment splitでBase/LoRAのLinear Probe、順逆・frame shuffle/static repeatで時間依存性を検証 | LoRA update/freeze/EMA安定、same-video negatives/false negatives方針、過去能力保持。学習器・online protocol・表現品質・temporal sensitivityを別々に判定。長時間実験は別承認 |

## 公平な比較の注意点と反例

- **初期チェックポイントの公平性**：A/Bとも`google/vit-base-patch16-224`系など同一のpretrained weightから初期化する。Fullをrandom初期化、LoRAをpretrained初期化する比較は無効。Query projector初期状態も合わせる。初期B=0のLoRA branchはFull基盤と出力一致を確認できる。
- **比較できる部分**：A/Bの主差分はtrainable scopeであり、学習後のweightをbitwise一致させることではない。EMA対象範囲は可変学習部分だけに対応、Frozen Baseの不変性を検証する。
- **主評価**：MoCoのtrain Lossを`性能`の代用にしない。正解ラベルをMoCo自己教師ありtrainに渡さず、評価時のみLinear Probe/kNNを固定protocolで使う。train/val/test leak防止。pretrained ViTはもともとの教師あり・画像知識を持ち得るため、**学習前との差分**を測る。必要なら事前学習ドメインと別の画像datasetで評価する。
- **Projector confound**：学習済Projectorの128次元とEncoder CLS768次元は別指標。研究のLoRA表現解析ではまずCLS768を主評価、128も補助。2群とも評価時には同じfeature definitionを保つ。
- **Batch/Optimizer confound**：GPUメモリが足りずFullのbatchだけ違うとInfoNCE/Queue/Batch統計も変わる。global batch・Kに対するqueue turnover、augmentation policyとoptimizer stepsを同じ条件で固定。Gradient accumulationは単純なglobal batch変更の代替とはならず、MoCo enqueue/EMA timingを保てる仕様が必要。Full/LoRAで同じLRが適正とは限らないため、等しい学習率探索回数・計算予算を報告する。
- **GPUコスト**：ResNet50 E5の11時間11分をViT runの所要時間とみなさない。ViT Fullはactivation・optimizer state負担が大きい。LoRAのtrainable params削減＝同率のGPU VRAM削減とは限らない。GPU 2台×約20GiBのpreflight pilotとVRAM/step time計測が先。
- **ChunkとTemporalの差**：frame-wise ViT＋masked meanはchunk内フレーム順序の置換に不変。時間的・動的情報の主張は別のorder-sensitive対照、学習の順序依存性、positive設計、過去queueとfuture leakageの監査が必要。

## 最有力の現時点判断／保留／棄却寄り

- **最有力**：同一画像2-viewのMoCo共通学習器を、Full ViT と Frozen ViT+Q/V LoRAで動かす。検証をsmoke→短期pilot→同一画像予算の学習→heldout Linear Probe/kNNに分ける。低コストならProjector-only Frozen ViTを追加。
- **後段**：動画raw固定chunk→Sequential Loader一致→strict online→temporal/LoRA解析。
- **保留**：本番dataset（CIFAR10 vs fixed ImageNet subset vs ActivityNet静止frame）、学習epoch/optimizer/LR予算、2GPU 20GiB VRAMでのbatch実現可能性、Linear Probe学習量・validation分割、性能の妥当性、画像学習checkpointを動画開始に流用するか再初期化するか。
- **棄却寄り**：ResNet E5が通ったためViT＋LoRAの動画full runに直行する／LossだけでFull対LoRA優劣を決める／画像MoCo比較でtemporal learningを主張する／Fullのみ極端にbatchやaugmentationを変更して同条件比較と称す。
- **反証候補**：Frozen ViTの事前学習特徴が既に強く、短期MoCoで両群ともLinear Probe改善しない（これはLoRA固有の失敗とは限らない）。Projector-onlyがLoRA同等になりLoRA追加寄与が薄い。LoRAが同等性能でもComputeが減らない。Static-repeated入力でも高性能だと動的情報の根拠が弱い。

## Specへの引き継ぎ候補と次の判断

最初のspecに含める範囲は**P0〜P2（ViT Full/Frozen+LoRA MoCo共通の通常画像1step〜pilot）**を推奨。P3性能比較のdataset・LRと評価指標の具体値は、それより前にP0で固定する。P4以降のvideo Chunk、Loader parity、strict onlineは独立spec/gateにする。

研究上Blockingなのは「性能比較を何のdataset・split・feature・metric・予算で決めるか」。**推奨デフォルト**はCIFAR10 224pxの短時間sanity→既存ImageNet trainから固定subsetを使ったMoCo→学習で使わないheldout validationでCLS768のlinear probe Top-1（必要なら小規模kNN）、同じfeature extractor before/after。独立した下流ImageNet以外への転移は追加候補。データサイズや予算の最終選択はユーザーによるspec段階で確定する。

**未実施**：spec作成/昇格、研究コード変更、GPU run、Comet送信、README/TODO変更。探索記録を実行許可とみなさない。
