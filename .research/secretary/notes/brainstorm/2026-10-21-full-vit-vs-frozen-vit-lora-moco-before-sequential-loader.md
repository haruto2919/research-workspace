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

## Specへの接続（2026-10-21）

ユーザーがB2（自作DistributedMoCo利用）・B10（初期共通Optimizer/LR、後段方式別LR探索）・B11（複数GPU DDP）を採用したため、[ViT DDP MoCo Full FT / LoRA切替・Smokeのdraft spec](../../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)を作成。未決の数値・データ条件があるため、specは`draft`のまま。研究コード変更、GPU実行は未許可。

## 追記：Optimizerの客観的再評価（2026-10-21 12:13 JST）

### 新たな相談とEvidence

ユーザーは、公式ResNet50 MoCo v2/E5のSGD設定をViT Full FT / Frozen ViT＋LoRAにも引き継ぐべきか、AdamWとSGDのどちらが総合的に適するかを客観評価するよう依頼した。この相談は**推薦を求めたものであり、AdamWの採用承認ではない**。既存[ViT DDP切替draft spec](../../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)のB10（両方式で初期Optimizer種・LRを揃える）は維持し、最終Optimizer種はまだ未確定とする。

- Meta [MoCo v2公式 main_moco.py](https://github.com/facebookresearch/moco/blob/a419ef282fc9885015674e77f58c59447206a27f/main_moco.py)ではResNet50/ImageNetのOptimizerはSGD（既存E5もSGD）である。
- Meta [MoCo v3公式CONFIG.md](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md)はViTモデルのpretrainingをAdamW（ViT-B例：LR 1.5e-4、WD 0.1、warmup 40epoch）で実施している。**MoCo v3と今回のMoCo v2式Queue付き・事前学習済みViTのFine-Tuningは別条件**なので、その数値を流用しない。
- [Chen et al., ICCV 2021, An Empirical Study of Training Self-Supervised Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_An_Empirical_Study_of_Training_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)はViTの自己教師あり学習でoptimizer/LR/patch embeddingなどを検証し、安定性がaccuracyへ影響することを示す。MoCo v3のViTをAdamW baselineとした。
- [Kumar et al., ICLR 2024, How to Fine-Tune Vision Models with SGD](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e9cbf616dac568a9cb3342761125db24-Abstract-Conference.html)は複数のpretrained視覚モデルFine-TuningタスクでAdamWがSGDより良い条件がある一方、最初のembedding層をfreezeするとSGDの差が解消・逆転し得ることとメモリ利点を報告。ただし本研究のMoCo v2/LoRAと同一実験ではない。
- 現行コード `scripts/smoke/smoke_vit_lora_cifar10.py` の独立LoRA単体（分類head付き）ではAdamWを使用済み。ただしこれはMoCo＋ViTでの性能検証ではない。
- `self_supervised/moco/ddp_moco.py`はMoCo演算に特定Optimizerを要求しない。一方、ResNet50 parity専用runnerはSGDを固定しているため、ViT専用runを分離してResNet50 E5の再現契約を維持する。

### 比較・推論・推薦（未採用）

| 評価軸 | SGD | AdamW |
| --- | --- | --- |
| 既存ResNet50 E5設定の連続性 | 高い | 異なる |
| ViT MoCoの公式先例 | v2はResNet中心 | v3 ViTで公式採用 |
| pretrained ViT Full FTの初期候補 | LRやembedding層への配慮が必要な場合あり | 安定性・Fine-Tuning実績から有力 |
| LoRA＋Projector | LR調整で可能、現研究では未検証 | 既存LoRA単体smokeで利用済み、MoCo時は未検証 |
| optimizer stateメモリ | momentum SGDは約1個分/学習parameter | AdamWは1次・2次momentで約2個分/学習parameter |
| 科学的な公平性 | どちらも双方の方式で同じoptimizerを使えば成立 | 同左。方法ごとのLR最適性は別に検証 |

**第一候補：AdamWをViT Full FT・LoRA共通の初期Optimizerにする。** ただし、FullFTとLoRAで同一初期LR・schedulerを用いる条件統制pilotと、後続の等しいLR探索予算での性能比較を区別する。AdamWに一律高WDを設定せず、対象parameter group/decay除外、warmup、LR、seed、global batch、AMP、特徴evalを事前に固定する。AdamWがVRAMに収まらない場合はBatch・mixed precision・SGDを比較対象として再検討し、FullFTとLoRAで異なるglobal batchにした結果を公平な同条件比較として扱わない。

**SGDの価値：** MoCo v2/E5とのOptimizer連続性、メモリ削減、ViT+MoCoで同等の表現評価を達成できる可能性。SGDの低性能を最初から断言しない。実際にどちらが本研究で良いかは、同一画像・seed・初期ViT/Projector・DDP global batch・学習量を使用し、両Optimizerに適切なLR探索を与え、有限Loss/勾配、更新norm、学習速度・ピークVRAM、heldout CLS Linear Probe/kNNなどで判断する必要がある。LoRA/FullFTへの適合性やベストLRは**未実測・未採用**。

### research-specへの引き継ぎ候補

B10 numerical blocking：AdamWかSGDの採用、同一初期LRの値、Weight Decayのparam-group規則、warmup・cosine horizon、GPU/Batch/AMP条件を次のspec Gateで決める。新しい提案を現行draftの承認やGPU実行許可とみなさない。

## 追記：AdamWの明示採用（2026-10-21）

直前のOptimizer比較への回答に対し、ユーザーが「採用します」と明示したため、**ViT＋MoCo Full FT / Frozen ViT＋LoRAの共通OptimizerにAdamWを採用**する。初期の条件統制比較は同じ初期LR・同じschedulerで行い、将来的な表現性能比較では方式別LR探索を可能とする既決の方針を維持する。以前の節にある「未採用」は、その時点の探索状態を記した履歴であり、本追記以降はAdamWの採用が優先する。

保存先の[DDP ViT-MoCo Full FT / LoRA切替のdraft spec](../../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)のB10とAmbiguity Gateを同期した。**LR具体値・weight decay対象と値・scheduler/warmup・GPU Batchは未決**。ResNet50公式parityのSGDは変更しない。spec全体のapproved、研究コードの変更、GPU実行は依然として未許可。

## 追記：ActivityNetフレームの低類似度を狙った時間間隔抽出（2026-10-21）

### ユーザー指示として確定した方向性

ユーザーは、Full ViT＋MoCo対Frozen ViT＋LoRA＋MoCoの**通常DataLoader・単一フレーム入力の段階**では、同一動画から抽出したフレーム同士の類似度が「なるべく低くなることが想定される時間間隔」を設けたいと明示した。目的は近接・重複フレームを避けて視覚的多様性を高めることと解釈する。ただし**具体秒数・フレーム上限数・類似度閾値は未採用**であり、現行の[ViT DDP切替draft spec](../../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)はB6/7/8を未決のまま保持する。

### 確認済み実装・研究文脈

- 研究コード`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`の`scripts/smoke/smoke_activitynet_vit_clip_feature.py`はSequentialDatasetの**最初の16連続frame**を読む経路であり、今回の「疎間隔・通常DataLoader学習」が既に実装された証拠ではない。
- `dataset/video_folder.py`には`RandomClipSampler`/`RandomSampler`があるが、動画clip抽出であって**ActivityNet単一frameを一定秒数で疎抽出する確定済み経路とは別**。
- 現行E5はImageNet画像学習、2GPU×local128、K=65536。新ViT用のActivityNet frame Manifest/selectionは別設計。
- 動画の冗長性によるsampling最適化は先行研究でも問題とされる：[MGSampler (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Zhi_MGSampler_An_Explainable_Sampling_Strategy_for_Video_Action_Recognition_ICCV_2021_paper.html)、[Exploring Video Frame Redundancies (CVPRW 2023)](https://openaccess.thecvf.com/content/CVPR2023W/VDU/html/Yoon_Exploring_Video_Frame_Redundancies_for_Efficient_Data_Sampling_and_Annotation_CVPRW_2023_paper.html)。ただしActivityNet上で最適秒数が既知という根拠ではない。

### 案の比較・懸念

| 案 | 長所 | 短所・留意点 | 評価 |
| --- | --- | --- | --- |
| 固定秒数間隔（例10秒）＋全体順序をshuffle | 仕様・Manifest・再現性が簡潔。時間的に近すぎる画像を抑制 | 静止カメラなどでは間隔が空いても高類似度。短い動画と長い動画で枚数不均衡 | 初期候補 |
| 長い固定秒数（例30秒以上） | 見かけの時間隔は大きくなる | フレーム総数低下、行動区間の見落とし、同一Frameのepochまたぎ再出現とK=65536との競合 | 根拠なしに固定しない |
| 動画長で等間隔（動画あたり固定16/32frame） | 動画別の学習重みを揃えやすい | 長短動画で実際の時間差が違い、短い動画は近接フレームになり得る | 最小時間隔制約と組み合わせる候補 |
| 画像特徴が最も異なるframeを動的選択 | 直接類似度を下げられる | pretrained encoder依存の選択bias、アクション重要frameの脱落、追加推論計算、代表性損失 | 現段階の初期方針には採らない |
| 短い診断で固定秒数間隔を選定 | 実データに即した根拠と再現性が得られる | 実際の動画試料・読込コストが要る。実行は別許可 | **最有力** |

### 最有力案（未承認・未実施）

1. ActivityNet training splitの事前に固定した多様な動画（例えば100〜200本）で、**候補時間差5/10/20/30秒**のフレームペアを抽出する。総数・速度の実測前には具体間隔を正式固定しない。
2. 学習に使う前の**同一pretrained ViT-B/16 CLS[768]**をeval modeで計算しL2 normalize。フレームペアcosine similarityを時間差bin別に比較する。cosine medianだけでなくp90（似たまま残る上側tail）、有効抽出枚数、カバレッジ（時間範囲/アクション区間は診断に留める）を記録し、必要ならSSIM・thumbnail目視を補助。CLS cosineが視覚的冗長性や動作の本質を完全に表すわけではない。
3. **類似度低下が頭打ちになる中で、保持フレーム総数が十分な最小時間間隔**を選ぶ。極端に長くして全フレーム数を犠牲にしない。秒数は結果を見て固定、Manifest（video ID、path、duration、timestamp、frame index、sampling policy、seed、file hash、split）を凍結する。
4. 本学習でのフレーム抽出は固定し、**DDPのDistributedSamplerで読み出し順序だけをepoch毎にshuffle**。Full FTとLoRAは同じManifest、同じtwo-view augmentation policy、同じglobal batch、同じinitial pretrained/projectorと学習量を比較契約とする。
5. 通常画像MoCo Positiveは**同一フレームの2-view**。別時間フレームはPositiveにしない。K=65536と動画ごとの保持枚数のバランス、sample ID・video ID別Queue監査、same-video/semantic false negativeの有無を記録する。**同一動画のNegative除外は未採用**であり、後段の別比較方針とする。
6. 全10,024 training動画が使用可能と仮定した場合、K65536を上回る画像数には平均7枚/動画以上、2Kを上回るには平均14枚/動画程度が必要な計算になる。実際の有効動画数・フレーム数はローカルデータ実体でのみ確認可能。これは厳密なMoCo理論要件ではなく、epoch跨ぎの同一画像Negativeが残ることを避けるための診断目安。
7. この段階は**frame-level画像表現学習**であり、時間隔を広げただけで動的/時間順序特徴を学んだとは主張しない。Sequential Loader、16-frame chunk、時間順序感度は後の独立Gate。

### 未検証の仮説と反例

- 仮説：5秒より10/20/30秒の方が同一動画frameの高類似度pair比率が下がる。**ActivityNet全般で単調低下するかは未検証**。
- 反例：固定カメラ／長い同場面では30秒離しても類似。場面転換後では数秒で低類似。間隔を伸ばしすぎるとdatasetがQueueサイズを下回ったり学習対象の行動部分を落とす。
- 追加確認：異動画間で同じ行為をするframeも高類似になりうるため、時間間隔だけでMoCoのfalse negativesを解消できない。

### research-specへ渡す事項

**ユーザー採用の目的**：近接フレームを避け、同一動画内のフレームの視覚的類似度を低くする。**未決**：具体的な秒数、許容する抽出枚数分布、類似度診断を必須とするか、抽出間隔確定の基準、video ID由来negative診断。実際の診断GPU実行・コード変更は未許可。現行specに自動昇格させない。
