---
project: sequential-video-lora-analysis
date: 2026-10-11
analysis_mode: explain
detail: full
level: junior
source_repository: /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
source_commit: 65dedb4ebc26c48d23606b9bc674044f64e64c06
source_branch: dev
source_dirty: false
targets:
  - model/
  - integration/
  - self_supervised/
  - training/
  - evaluation/
  - scripts/
  - dataset/
  - setup/
  - logger/
  - utils/
  - main.py
  - main_pl.py
  - train.py
  - val.py
  - run_full_pipeline.sh
related_spec:
  - specs/2026-09-28-shared-streaming-moco-protocol-spec.md
  - specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md
  - specs/2026-10-07-configurable-reduced-video-full-pipeline-spec.md
---

# Code Deep Dive: dev branch full explanation

## 対象と解析の範囲

- 目的: devブランチの研究コードを、各クラス・関数の役割、Tensor shape、処理順序、背景数学、採用理由まで初学者向けに説明する。
- ソース: tamaki-lab/2026_09_ishikawa_sequential-video-lora の dev、commit 65dedb4ebc26c48d23606b9bc674044f64e64c06。
- Git状態: HEADは dev と origin/dev に一致し、作業ツリーはclean。したがって本レポートは上記commitだけでソースを特定できる。
- 中心対象: ActivityNet → Sequential Loader → ViT Q/V LoRA → Streaming MoCo → Query LoRA snapshot → segment feature → Linear Probe。
- 併記対象: テンプレート由来の教師あり分類系、smoke/audit CLI、artifact・checkpoint・Comet utility。
- テスト: テストコードの契約は静的に読んだが、research-code-insightの境界に従いテスト、学習、smoke、CLIは実行していない。
- 外部依存: sequential_loader本体のソースは別リポジトリなので、本レポートはこのconsumer側が要求する公開interfaceだけを説明する。
- 除外: test/ の各test関数を一つずつ逐語解説すること、dog.jpg、生成済みlog/data、外部ライブラリ内部実装。

関連文脈は、Shared Streaming MoCo specが implemented、full-dataset specが approved、source-selection addendumが implemented である。2026-10-08 MTGではloss振動とQuery LoRA勾配急増が未解決で、Linear Probeの科学的解釈は保留されている。よって以下は「コードが何をしているか」の説明であり、「時間情報を学習できた」という主張ではない。

## Overview: What / Why

このリポジトリには大きく2系統が同居している。

1. 研究の中心経路
   - 動画を16 frameずつ時系列に読む。
   - 各frameを画像ViTで独立に768次元へ変換する。
   - valid frameだけを平均してchunkを1本の768次元vectorへする。
   - Query側のViT Q/V LoRAとprojectorをInfoNCEで更新する。
   - Key側はbackpropagationせず、Queryの指数移動平均で更新する。
   - 過去Keyを4096件までFIFO Queueに保持する。
   - 最終Query LoRAをBase ViTと同じActivityNet segmentで比較する。

2. テンプレート由来の教師あり分類経路
   - CIFAR10、ImageFolder、VideoFolderなどをResNet、ViT、X3D、ABNで分類する。
   - 通常PyTorch版とLightning版の2入口を持つ。
   - 研究のStreaming MoCoとはoptimizer、checkpoint、data flowが別である。

中心経路の重要な設計は「model primitiveとtraining orchestrationの分離」である。ViTLoRAMoCoはQuery/Key、projector、Queue、InfoNCE、EMAだけを持ち、動画iteration、warm-up、optimizer step、resumeはtraining側が担当する。これにより、同じmodel primitiveを短時間canaryとfull runで共有し、更新順序を監査しやすくしている。

## Key Components and Data Flow

~~~text
ActivityNet video
  │ Sequential Loader: chronological chunks, T=16
  ▼
SequentialSample
  frames       [16, 3, H, W], CPU uint8
  valid_mask   [16], bool
  frame_indices/timestamps/metadata
  │
  ├─ Query view: raw valid frames
  └─ Key view: valid frames only RGB→GBR, then horizontal flip
          padding and metadata stay unchanged
  │
  ▼ AutoImageProcessor
pixel_values [V, 3, 224, 224], V = valid frame count
  │
  ▼ ViT-B/16, frameごと
valid CLS features [V, 768]
  │ scatter back
  ▼
frame features [16, 768], padding rows = 0
  │ masked mean
  ▼
clip feature [768]
  │ MLP 768→768→128, L2 normalize
  ▼
q [128] / k+ [128]
  │ old FIFO Queue: N past keys [N, 128]
  ▼
InfoNCE logits [1, 1+N] → scalar loss
  │
  ├─ AdamW: Query LoRA + Query projector
  ├─ EMA: Key LoRA + Key projector
  └─ enqueue detached pre-EMA current key

final Query LoRA
  │
  ├─ Base ViT condition
  └─ Base ViT + Query LoRA condition
        ↓ same manifest
segment features [Nsegment, 768]
        ↓ frozen features
Linear(768, 200) × seeds 0,1,2
        ↓
Top-1 / Macro class accuracy
~~~

## Tensor shape早見表

| 段階 | shape | dtype / device | 意味 |
|---|---:|---|---|
| sample.frames | [T, 3, H, W], T=16 | uint8 / CPU | padding込みのRGB frame |
| sample.valid_mask | [T] | bool / 通常CPU | 本物のframeかpaddingか |
| valid_frames | [V, 3, H, W] | uint8 / CPU | validだけをsource順で抽出 |
| pixel_values | [V, 3, 224, 224] | processor依存、通常float32 / CPU→encoder device | ViT前処理後 |
| ViT token列 | 概念上 [V, 197, 768] | float / encoder device | 196 patch + CLS |
| valid_features | [V, 768] | float / encoder device | token 0、つまりCLS |
| frame_features | [16, 768] | 同上 | 元位置へ戻しpadding行は0 |
| clip_feature | [768] | 同上 | valid frameのmasked mean |
| projected q, k | [128] | float / model device | L2 norm = 1 |
| Queue keys | [N, 128], 1≤N≤4096 | detached float / model device | 過去Key |
| InfoNCE logits | [1, 1+N] | float / model device | 先頭がpositive |
| segment features | [Nsegment, 768] | float32 / CPU artifact | chunk featureの単純平均 |
| labels | [Nsegment] | int64 / CPU | 0..199 |
| probe logits | [B, 200] | float / probe device | 200-class score |
| confusion matrix | [200, 200] | int64 / CPU | 行=true、列=prediction |

Vは1〜16である。processorのdtypeはコードでassertしていないため、「通常float32」は推定であり、直接保証されるのはshapeである。

## Concepts and Design Decisions

### 1. ViTが1 frameを768次元にする

model/backbones/vit/vit_frame_encoder.py:15-35 のViTFrameEncoderは、poolerなしのHugging Face ViTModelを読み、全backbone parameterをfreezeする。

~~~python
self.vit = ViTModel.from_pretrained(checkpoint_id, add_pooling_layer=False)
self.vit.requires_grad_(False)

def forward(self, pixel_values):
    return self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
~~~

224×224画像を16×16 patchへ切ると14×14=196 patchになる。CLS tokenを先頭へ足すためtoken数は197。各tokenは768次元で、コードは位置0だけを取り出すため出力は [V,768] になる。

CLSは「画像全体を代表するよう学習されたtoken」である。ただし、このViTは各frameを独立処理する。16 frameを同時にvideo transformerへ入れているわけではないので、forwardそのものにframe間attentionはない。

ViTLoRAFrameEncoderは同じViTをfreezeしてから、PEFTのLoRAをq_projとv_projへ注入する（model/backbones/vit/vit_lora_frame_encoder.py:19-50）。

### 2. LoRAの数学

通常の線形層を y = Wx とする。LoRAはWそのものを更新せず、

ΔW = α/r × BA

を足して、

y = (W + ΔW)x

とする。ここでAは高次元からrank rへ圧縮し、Bはrank rから元の次元へ戻す。r=8、α=8なのでscale α/r=1である。

このコードはViTの12層それぞれのqueryとvalue、合計24 linear moduleへLoRAを付ける。hidden size 768なら1 moduleあたりのparameterは

2 × 8 × 768 = 12,288

で、全体は

12 layers × 2 targets × 12,288 = 294,912 parameters

となる。A/Bで48 tensorである。この数はtraining/moco_audit.py:38-43がruntimeに式から検証する。

Query projectorはLinear(768,768) → ReLU → Linear(768,128)で、689,024 parameters / 4 tensors。したがってoptimizer対象は合計983,936 parameters / 52 tensorsである。Base ViTとKey branchはoptimizer対象外。

なぜQ/Vか:

- Queryは「何に注目するか」、Valueは「集約する内容」を変えられる。
- Key projectionを変えないことでattentionの対応軸を比較的保つ、という一般的なLoRA設定に近い。
- ただし、この研究でQ/Vが最適という性能根拠はコードからは確認できない。これは設定として固定されているだけである。

### 3. valid frameだけを処理し、元位置へ戻す

integration/sequential_vit.py:15-70 のencode_chunkはpaddingをViTへ渡さない。

~~~python
valid_frames = sample.frames[valid_mask]
pixel_values = processor(images=valid_frames, return_tensors="pt")["pixel_values"]
valid_features = encoder(pixel_values.to(device))

frame_features = torch.zeros((sample.frames.shape[0], feature_size), ...)
frame_features[valid_mask.to(frame_features.device)] = valid_features
~~~

具体例としてvalid_maskが

[True, True, True, False, ..., False]

なら、ViTへ渡すのは3 frameだけである。valid_features [3,768]をframe_features [16,768]の先頭3行へ戻し、残り13行を0にする。

このscatter代入はvalid_featuresへのgradient pathを維持する。したがってQuery側ではmasked meanからLoRAへbackpropagationできる。padding行はそもそもencoderを通らない。

### 4. Masked Mean

MaskedMeanClipAggregator（model/aggregators/masked_mean_clip_aggregator.py:7-27）は

c = 1/V × Σ m_t f_t

を実装する。m_tはvalidなら1、paddingなら0。実装はboolean indexing後にmeanするので、paddingは分子にも分母にも入らない。

~~~python
return frame_features[valid_mask.to(frame_features.device)].mean(dim=0)
~~~

長所:

- frame数が違っても固定長 [768] になる。
- parameterがなく、安定で説明しやすい。
- padding処理が明示的。

重要な限界:

- 平均は順序不変で、frameを逆順にしても同じ値になる。
- ViTもframeを独立処理するため、1 chunkの表現自体は時間順序を直接表さない。
- よってLinear Probeが改善しても、それだけで「動き」や「時間順序」を理解したとは言えない。

segment featureもchunk featureの単純平均である。chunkごとのvalid frame数で重み付けしないので、短いchunkも完全な16-frame chunkも1票として扱う。

### 5. Query/Keyの2 view

integration/sequential_moco.py:11-20 のmake_two_viewsはQueryをrawのまま返し、Keyだけを変換する。

Stage 6Bではvalid frameごとに

1. channel [R,G,B] → [G,B,R]、index [1,2,0]
2. width方向にhorizontal flip

を行う。padding frameとmetadataは変更しない。dataclasses.replaceを使うため、元sampleはmutationしない。

具体例: あるpixelが [R,G,B]=[10,20,30] なら、色channelは [20,30,10] になる。その後、画像の左右位置が反転する。

採用意図は「同じchunkの内容を保ちながら見え方を変え、同一性をpositiveとして学習する」こと。ただしGBR変換は標準的な自然画像augmentationより強く、色意味を壊す可能性もある。その科学的妥当性は未検証。

### 6. ProjectorとL2 normalization

clip [768]はMLPを通り [128]へ射影され、F.normalizeで長さ1にされる（self_supervised/moco/vit_lora_moco.py:108-130）。

q̂ = z_q / ||z_q||₂,  k̂ = z_k / ||z_k||₂

単位vector同士のdot productはcosine similarityになる。値域は理論上[-1,1]。projectorを使う理由は、contrastive objectiveが使う空間と、下流評価に使う生の768次元空間を分離するためである。Linear Probeはprojector後の128次元ではなく、projector前の768次元を使う。

### 7. FIFO Queueとnegative policy

MetadataQueueは各entryを

- key [128]
- sequence_id
- sequence_index

としてdeque(maxlen=4096)へ保存する。enqueue時にdetach().clone()するため、過去graphを保持しない。capacityを超えると最古entryが自動で落ちる。

negative policy:

- different_sequence: currentと異なるsequence_idだけ。Stage 6A。
- all_past: old Queueの全entry。Stage 6B。

Stage 6Bの具体例:

~~~text
A1: Key-only warm-up → Queue [A1]
A2: positive=A2 raw vs A2 transformed, negatives=[A1]
    update後 Queue [A1,A2]
A3: positive=A3 raw vs A3 transformed, negatives=[A1,A2]
    update後 Queue [A1,A2,A3]
~~~

same-videoの過去chunkもnegativeになる。これは時系列で異なる状態を区別させる可能性がある一方、意味的に似た隣接chunkをfalse negativeとして引き離す可能性もある。

### 8. InfoNCE

self_supervised/moco/vit_lora_moco.py:132-144:

~~~python
negative_keys = torch.stack([entry.key.detach() for entry in entries])
similarities = torch.cat(((query @ positive_key.detach()).reshape(1),
                          negative_keys @ query))
logits = similarities.unsqueeze(0) / self.temperature
loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=query.device))
~~~

N個のnegativeがあるとlogitsは [1,1+N]。class 0をpositiveと見なすcross entropyなので、

L = -log exp(q·k⁺/τ) / {exp(q·k⁺/τ) + Σ exp(q·kᵢ/τ)}

となる。τ=0.07はsimilarity差を鋭くする。positive similarityを上げ、negative similarityを下げる方向にQueryだけへgradientが流れる。positive Keyとnegative Keyはdetachされている。

例としてsimilarityが [0.8, 0.2, -0.1] なら、温度で割ったlogitsは約 [11.43,2.86,-1.43]。positiveが明確に最大なのでlossは小さい。

### 9. Query更新、Key EMA、enqueueの順序

training/streaming_moco.py:92-144 が1 updateの正本である。

~~~text
1. raw Query / transformed Keyを作る
2. Query forward
3. Key forward（no_grad）
4. old Queueを監査
5. old Queueからnegativeを選ぶ
6. InfoNCE
7. backward
8. Query LoRA + Query ProjectorへAdamW
9. Key LoRA + Key ProjectorをEMA
10. forward済みcurrent KeyをQueueへenqueue
~~~

EMAは

θ_k ← m θ_k + (1-m) θ_q,  m=0.999

である。KeyがQueryをゆっくり追うので、Queue内の過去表現と現在のencoderが急にずれにくい。Base ViTは両branchとも更新しない。

current keyをloss前にenqueueしない理由は、自分自身がnegativeへ混ざるのを防ぐため。さらにenqueueするkeyはEMA更新前にforwardした値なので、そのstepのpositiveと保存値が一致する。

### 10. Warm-upとstream

StreamingMoCoProtocolは3軸を独立に持つ。

| preset | stream | Key変換 | negative | warm-up |
|---|---|---|---|---:|
| Stage 6A | 4-stream round-robin | horizontal flip | different sequence | 4 |
| Stage 6B | strict single | GBR + flip | all past | 1 |

warm-upはactive streamの最初のsampleをKey-onlyでQueueへ入れる。Query forward、loss、backward、optimizer、EMAはしない。Queueが空だとInfoNCEのnegativeが0件になるため、最初のnegativeを準備する役割である。

ordered_samplesはprefetchせず、ReaderをExitStackで必ず閉じる。strict_singleではA1,A2,A3...、round_robinではA1,B1,C1,D1,A2...。frame indexの連続性、terminalだけのpadding、CPU uint8、bool maskを毎sample検証する。

### 11. full-dataset single pass

training/streaming_moco_full.py:173-417 は選択されたtraining動画を一度だけAdapter順で処理する。

- warm-upは全runで「最初の動画の最初のchunk」だけ。
- 動画境界でQueue、optimizer、Query、Keyをresetしない。
- processed_videosは動画をEOFまで読み切った後だけ増える。
- resume checkpointは動画境界だけ。
- stop_after_videosはpauseでありfinalではない。
- 100動画ごとにresume、1000動画ごととfinalにevaluation snapshot、100 updateごとにComet集約がcanonical設定。

full profileはtraining 10,024 / validation 4,926。reduced profileはSHA-256 rankingでtraining/validation各1,000本を選ぶが、実際の処理順はAdapter順へ戻す。selection seedとmodel seedは独立。

### 12. Resume stateとevaluation snapshotは別物

training/moco_checkpoint.pyは2種類を明確に分ける。

resume/latest.pt:

- Query LoRA、Query projector
- Key LoRA、Key projector
- AdamW state
- Queue keys + metadata
- Python/NumPy/Torch/CUDA RNG
- countersと厳密なrun identity

を持つ。payload SHA-256を外側へ持たせ、temporary write → reload検証 → replace → directory fsyncで公開する。

evaluation snapshot:

- Query LoRAだけ
- PEFT native adapter_model.safetensors / adapter_config.json
- metadataと各file hash

を持つ。projectorやQueueを含めないのでMoCo resumeには使えない。Linear ProbeのLoRA conditionへ読み込むためのartifactである。

### 13. Manifest

ActivityNet annotation 1件を1 sampleとする。chunk採用条件は「valid timestampが1個以上あり、有限で、すべてclosed interval [segment_start, segment_end] 内」である。部分的にoverlapするchunkは不採用。

同じ動画に重複annotationがあってもannotation_indexが違うため別segment_idになる。

manifest順序は

split → Adapter source order → annotation_index

で固定。labelはtraining/validationの共通200 labelsをalphabetical sortし0..199へ割り当てる。Integrity Gateはclass coverage、重複、split overlap、chunk count、mapping hash、再生成hashを検証する。

### 14. Segment feature

各採用chunk:

f_chunk = 1/V × Σ frame_CLS

各segment:

f_segment = 1/M × Σ f_chunk

結果は [768] float32 CPU。normalization、centering、projectorは使わない。BaseとLoRAの共有contract hash、manifest hash、segment ID/orderを一致させる。

### 15. Linear Probe

Linear Probeはfeature extractorを一切更新せず、線形分類器だけを学習する。

y = Wx + b

Wは [200,768]、bは [200]、入力batch xは [B,768]、logitsは [B,200]。

設定:

- Cross Entropy、class weightなし
- AdamW、lr=1e-3、weight decay=1e-4
- batch size 256
- 100 epochs
- seed 0,1,2
- trainingだけshuffle
- scheduler、early stoppingなし
- validationは最終epoch後に1回、model selectionへ使わない

Top-1:

正解予測数 / 全validation sample数 × 100

Macro class accuracy:

各classの正解率をclass間で単純平均

である。後者はsample数の多いclassに支配されにくい。3 seedの標準偏差はsample standard deviation、ddof=1。

## Code Walkthrough

### A. 研究中心モジュールの全クラス・関数

#### model/backbones/vit

- ViTFrameEncoder.__init__（vit_frame_encoder.py:18）: checkpointをpoolerなしで読み、hidden sizeを照合し、全parameterをfreeze。
- ViTFrameEncoder.forward（同:34）: [V,3,224,224] → CLS [V,768]。
- ViTLoRAFrameEncoder.__init__（vit_lora_frame_encoder.py:22）: frozen ViTへPEFT LoRAを注入。dictまたはtyped configを受ける。
- ViTLoRAFrameEncoder.forward（同:49）: LoRA適用後のCLS [V,768]。
- MaskedMeanClipAggregator.forward（masked_mean_clip_aggregator.py:14）: [T,D]とbool [T]を検証し、validだけを平均して[D]。

#### integration

- encode_chunk（sequential_vit.py:15）: valid frame抽出、processor、encoder、padding位置へのscatter、finite/shape監査。
- make_two_views（sequential_moco.py:11）: raw Queryと変換Keyを非破壊で作る。
- encode_query_view（同:23）: Query encoder → masked mean → Query projector。gradientあり。
- encode_key_view（同:30）: Key側の同処理。no_grad。
- source_count（sequential_stream.py:17）: stream modeから必要source数を返す。
- validate_sources（同:25）: source数とsequence_idの一意性を確認。
- ordered_samples（同:34）: Reader lifecycle、chronological order、shape/dtype/index/padding contractを守るcontext manager。
- SourceSelectionConfig.from_mapping（activitynet_source_selection.py:29）: all_sourcesまたはSHA-256 ranking設定を型・範囲検証。
- SourceSelectionConfig.metadata（同:59）: 安定したselection設定dict。
- SourceSelection（同:66）: selected sources、identity、SHAを束ねるimmutable result。
- approved_profile_name（同:73）: repository内version付きpresetとの完全一致だけをapprovedと認定。
- _uint32（同:15）: seed範囲検証。
- _score（同:88）: strategy/seed/split/idをNUL区切りでSHA-256化。
- _ordered_ids（同:95）: full inventory countとID一意性を検証。
- select_activitynet_sources（同:106）: hash rankで集合を選び、処理順はAdapter順へ戻し、content identityを作る。
- validate_selection_metadata（同:154）: artifactのselection SHA/identity一致を検証。
- validate_selection_record（同:161）: hash外の作成時provenance recordを検証。

#### self_supervised/moco

- select_negatives（negative_selection.py:4）: old Queue snapshotからpolicyどおりのentry tupleを選ぶ。空ならfail。
- lora_parameters（vit_lora_moco.py:25）: parameter nameに.lora_A.または.lora_B.を含むものだけ抽出。
- require_normalized（同:32）: [128]、finite、L2 norm≈1を検証。
- QueueEntry（同:39）: key、sequence_id、sequence_indexのfrozen dataclass。
- MetadataQueue.__init__（同:49）: capacity付きdequeを作る。
- MetadataQueue.__len__（同:58）: entry数。
- MetadataQueue.entries（同:61）: 外部mutationを避けるtuple snapshot。
- MetadataQueue.enqueue（同:65）: normalized keyとmetadataを検証し、detach cloneしてFIFO追加。
- MetadataQueue.negatives（同:73）: 旧Stage 5用different-sequence shortcut。
- ViTLoRAMoCo.__init__（同:87）: Queryから独立Keyをdeepcopyし、2 projector、Queue、momentum、temperatureを構築。
- ViTLoRAMoCo.query_parameters（同:118）: Query LoRA + Query projectorだけを返す。
- project_query / project_key（同:121/127）: 768→128とL2 normalize。Keyはno_grad。
- contrastive_loss（同:132）: positiveをclass 0にしたInfoNCEを返す。
- update_key（同:147）: Key LoRA/projectorへEMA。

#### training protocol / audit / per-step

- StreamingMoCoProtocol.__post_init__（moco_protocol.py:20）: 3軸とround-robin数を検証。
- source_count / warmup_count property（同:31/35）: active stream数。
- audit_encoder_parameters（moco_audit.py:9）: Q/V target数、poolerなし、LoRA tensor/parameter数、Base freezeを監査。
- parameter_groups（同:47）: query/key × base/LoRA/projectorへnamed parameterを分類。
- audit_parameters（同:62）: Query/Key config、projector構造、trainable集合、Queue次元を照合。
- audit_initial_state（同:114）: Query/Key値一致、storage独立、Queue emptyを検証。
- audit_views（同:130）: metadata不変、transform semantics、padding不変を検証。
- report_json（streaming_moco.py:17）: JSON eventをstdoutへ出す。
- cpu_snapshot（同:21）: parameterをdetached CPU clone。
- count_changed（同:25）: snapshotと異なるtensor数。
- audit_bases（同:30）: Query/Key Base不変を確認。
- audit_finite（同:36）: 全model parameterのNaN/Inf検査。
- audit_queue（同:41）: FIFO順、metadata、detach、shape、norm、expected key完全一致を検証。
- enqueue_key（同:64）: independent expected dequeと実Queueの両方へ追加して再監査。
- _gradient_diagnostics（同:74）: Query LoRA/projectorだけにfinite nonzero gradientがあり、他groupにないことを検証。
- train_step（同:92）: forward → old negatives → loss → backward → optimizer → EMA → enqueueの正本。
- round_robin_samples（moco_canary.py:27）: Stage 6A互換入口。
- run_streaming_moco（同:32）: fresh bounded run、active stream warm-up、optimizer構築、max_steps実行。
- run_canary（同:114）: Queue evictionなしというStage 6A上限を追加。

#### typed config

- _positive_int（moco_config.py:16）: boolを除外する正整数検証。
- ActivityNetConfig.from_mapping: split、count、class countをtyped dataclass化。
- LoRAConfig.from_mapping / metadata: target、rank、alpha、dropoutを検証・serialize。
- EncoderConfig.from_mapping: checkpointとshape contract。
- SequentialConfig.from_mapping: frames/chunkとround-robin stream数。
- OptimizerConfig.from_mapping: AdamWだけを許可しlr、decay、betas、epsを検証。
- IntervalConfig.from_mapping: resume/snapshot/metric間隔。
- MoCoConfig.from_mapping: protocol、projection、Queue、EMA、temperature、optimizerを統合。
- RuntimeConfig.from_mapping: dataset path、run ID、seed、device、resume/stop設定。
- FullMoCoConfig.from_mapping: Hydra合成値を全体のimmutable configへ変換。
- FullMoCoConfig.production_groups_match: 実行値をrepository canonical YAMLと比較。trackingは科学条件から除外。
- FullMoCoConfig.effective_protocol_identity: fullとselected subsetのprotocol identityを分離。
- canonical_activitynet_config / canonical_encoder_config / canonical_sequential_config / canonical_moco_config: repository SSOT YAMLをtyped objectとして読む。

#### full run

- RunPaths（streaming_moco_full.py:49）: resume、snapshot、audit、metadata、source listのpathを一元化。
- _Recorder.__init__: JSONLをappend openし、metric windowを初期化。
- _Recorder.event: local auditへ必ず1行保存。step以外はstdoutにも出す。
- _Recorder.step: contextを付けてevent保存し、windowへ追加。
- _Recorder.flush_metrics: 100 updateなど指定間隔で平均loss/similarity/grad normをCometへ送る。
- _Recorder.close: fileを閉じる。
- mean_grad: window内Query LoRA grad normの平均。
- run_identity: resumeで完全一致させる科学条件・code・dependency・device・source identityを構築。
- _new_optimizer: AdamWをQuery LoRA + projectorだけで作り、集合一致を検証。
- _device_snapshot: frozen Baseをdevice上cloneし、毎stepのhost copyを避ける。
- validate_full_sources: expected countとID一意性。
- run_full_streaming_moco: fresh/resume、single pass、warm-up、動画境界checkpoint、snapshot、pause/finalを統括。
- 内部identity_now: 現在optimizerを含むidentityを再構築。
- 内部snapshot_metadata: downstream snapshotのlineageを構築。
- 内部save_snapshot: Query LoRA保存とComet artifact登録。
- 内部snapshot_due: finalまたはinterval境界を判定。
- 内部save_resume: Base/finite/Queue監査後に完全stateをatomic保存。

#### checkpoint

- ordered_source_sha256: ordered IDsのcanonical JSON hash。
- seed_all: Python/NumPy/Torch/CUDA RNGを同じseedへ。
- base_fingerprint: PEFT wrapper込みencoderの非LoRA tensorsをhash。
- encoder_base_fingerprint: .base_layer.を正規化し、plain/PEFTで同じBase hashを得る。
- validate_production_snapshot_metadata: final、dataset、protocol、code、loader、version、shape、LoRA、optimizer、selectionを厳密照合。
- lora_config / optimizer_config: 実objectからmetadataを抽出。
- _cpu: nested training stateをCPU clone。
- rng_state / restore_rng_state: 全RNG stateの保存・復元。
- training_state: Query/Key adapters/projectors、optimizer、Queueを保存形式へ。
- save_resume_checkpoint: self-hash、temporary verification、atomic replace。
- load_resume_checkpoint: outer/payload hash、schema、video-boundary countersを検証。
- validate_resume_identity: RESUME_IDENTITY全fieldの完全一致。
- restore_training_state: tensor name/shapeを検証し、adapter/projector/optimizer/Queueを復元。
- snapshot_name: videos/update/finalからdirectory名を作る。
- query_lora_state: PEFT native LoRA tensorだけを取得。
- write_evaluation_snapshot: LoRA-only directoryを作り、in-memoryとの完全一致を検証。
- load_query_lora_snapshot: schema/hash/config/selection/keyを検証してencoderへ厳密load。

#### manifest / features / probe

- AnnotationIndex.__init__: expected annotation versionとcacheを初期化。
- AnnotationIndex.database: JSONをpath単位で一度だけ読みversion検証。
- AnnotationIndex.annotations: evaluation referenceからlabel/segmentを正規化。
- canonical_label_mapping: split間で同じ200 labelsを要求し、sort順IDを作る。
- chunk_in_segment / contained: valid timestampsのclosed interval完全包含。
- video_chunk_timestamps: 1動画を順に走査しchunkごとのvalid timestampsを返す。
- video_rows: annotationごとの採用chunk indices、segment row、skip labelを作る。
- build_manifest: split/source/annotation順で全rowを構築。
- manifest_bytes: 1 row 1 canonical JSON line。
- integrity_gate / gate_status: coverage・重複・hash・再生成のPASS/SMOKE_PASS/FAIL判定。
- write_manifest / load_manifest: atomic保存、hash・Gate・selection identity検証。
- validate_feature_definition: 実装済み定義だけを許可。
- shared_contract_sha256 / validate_shared_contract / validate_feature_pair: Base/LoRAの比較軸を同一化。
- feature_size_from_metadata: artifactから正のDを取得。
- chunk_feature: no_gradで [T,768] → [768] float32 CPU。
- video_segment_features: 必要chunkを一度だけencodeし、rowごとにmean。
- extract_split: manifest row順を保ちfeatures [N,768]、labels [N]、IDsを作る。
- save_split / verify_split: split artifactを保存しshape/dtype/orderを検証。
- load_features / write_feature_artifact: hash、manifest、contractを照合して読み書き。
- ProbeConfig.from_mapping / __post_init__: probe科学条件を読み、未実装overrideをfail-fast。
- with_epochs: immutable configのepochだけを置換。smoke用。
- hyperparameters: classifierを含む記録用dict。
- experiment_name: protocol/condition/seedの安定名。
- train_probe: seed固定、fresh Linear、epoch shuffle、CE/AdamW training。
- evaluate: Top-1、confusion、per-class、macroを計算。
- mean_std: arithmetic meanとsample std。
- write_result / load_summary: classifier、history、metrics、hashのartifact化と検証。
- _csv: deterministic CSV bytes。
- aggregate / aggregate_csv: conditionごと3 seedを集約。

### B. CLIの全関数

- scripts/moco/train_full_streaming_moco.run: Hydra解決、provenance、ActivityNet inventory、source selection、model構築、full engine呼び出し。
- 同main: Hydra decorator付き入口。
- scripts/linear_probe/configuration._science_view: 運用fieldを科学identityから除外。
- resolved_science: 指定groupのresolved config。
- science_contract: resolved/canonical比較とSHA。
- feature_science_contract: probe training条件を除くfeature用identity。
- build_manifest.selected_sources: Adapter full inventory、annotation hash、selection。
- resolved_manifest_id: reduced selectionならSHA suffixを付ける。
- resolve_snapshot_manifest_id: final snapshotと再計算selectionを先に照合。
- progress: interval progress JSON。
- build: manifest新規作成とGate 1-8。
- audit: 独立再生成でGate 9を確認しartifact登録。
- run/main: 入力検証とHydra入口。
- extract_features.full_sources_and_selection: full sourceとselectionを再構築。
- dataset_sources_and_identity: mount pathでなくcontent/orderへ紐付け。
- extraction_provenance: feature比較に必要なprovenanceへ射影。
- preprocessing_identity: processor class/config hash、frame数、dtypeを記録。
- build_encoder: BaseまたはLoRA conditionを構築しsnapshot contractを検証。
- run/main: 同一manifestから両splitを抽出し、既存artifactは完全一致時だけreuse。
- run_probe.aggregate_metadata: aggregate payloadと可変Comet sidecarを分離。
- run_seed: 1 condition × 1 seedのclassifier training/evaluation/artifact。
- run/main: Base/LoRA pair検証、6 run、mean/std集約。
- scripts/retry_comet_artifact.main: local hashを再検証し、失敗/disabledのComet登録だけ再試行。
- run_full_pipeline.sh のusage/die/run_logged/extract_feature_dir: 引数説明、fail-fast、tee logging、feature artifact path解決。shell全体はMoCo→snapshot検証→manifest build/audit→2条件feature→6 probes→aggregateの順。

### C. smoke / audit関数

- smoke_vit_frame_encoder.main: 1画像で [1,3,224,224] → [1,768] とfreezeを確認。
- smoke_50salads_vit_bridge.git_output/main: pinned revisionを確認し、50Salads最初のchunk bridgeを表示。
- smoke_activitynet_vit_clip_feature.git_output/main: ActivityNet最初のchunkでfrozen CLS、padding zero、masked meanを確認。
- smoke_activitynet_vit_lora_one_step.git_output: Git情報helper。
- 同run_one_step: 科学的意味のないmean-square lossでLoRA gradient/updateとBase freezeだけを確認。
- 同main: provenance、sample、shape、metadata不変を統括。
- smoke_activitynet_vit_lora_moco_one_step.audit_provenance: repository/branch/loader/dependency pin。
- read_first_chunk: 指定sourceのfirst chunk contract。
- report_sample / report_features: metadataと4段階shapeを表示・検証。
- run_one_step: 1回のInfoNCE、gradient、optimizer、EMA、enqueueを独立監査。
- main: 2動画の先頭chunkでwarm-up + 1 update。
- smoke_activitynet_streaming_moco.main: Stage 6A/B presetと3軸overrideのbounded run。
- smoke_activitynet_vit_lora_moco_multistep.main: Stage 6A互換4-stream canary。
- audit_activitynet_inventory.normalize_video_id: filenameの先頭v_だけ除去。
- scan_root: 動画file、normalized ID、extension countを再帰集計。
- main: annotationと複数local rootを照合しmissing/local-only/duplicateを表示。
- NearestFrame（timestamp audit）: annotation targetに最も近いdecoded frameを保持。
- parse_args: split、動画数、許容誤差、出力先。
- load_activitynet_database: annotation v1.3 databaseを検証。
- parse_annotations: label/start/end/midpointを正規化。
- seconds_from_av_time: PyAV時間を秒へ安全変換。
- probe_video: container/stream start、duration、fps、time baseを取得。
- sanitize_filename: 画像保存名を安全な文字へ。
- save_frame: uint8 CHW tensorをHWC PNGへ。
- inspect_video: timestamp逆行、annotation最近傍frame、overlap/完全包含chunkを監査し画像を保存。
- main: 対象sourceを選びreportを統括。

### D. テンプレート由来の教師あり分類

#### model

- ModelConfig: model名、pretrained、TORCH_HOME、class数。
- ModelOutput: logits [B,C] とoptional scalar loss。
- ClassificationBaseModel.__init__: CrossEntropyLossとplaceholder model。
- __create_dummy_model: subclass構築前の空Module。
- forward: model(pixel_values)後、labelがあればcross entropy。
- get_device: DataParallelならmodule側、それ以外は先頭parameter device。
- set_torch_home: pretrained weight cache先を環境変数へ。
- configure_model: ResNet18/50、ABN、ViT、X3D、dummyのfactory。
- ResNet18/50.__init__: torchvision modelを作り分類headを交換。
- prepare_model: ImageNet pretrained有無を選択。
- replace_pretrained_head: fcを [feature,C] へ。
- ViTb.prepare_model/forward: ViTForImageClassificationをC classesで読み、HF outputをModelOutputへ。
- X3DM.prepare_model: torch.hubのx3d_mを読みSoftmaxを外す。
- X3DM.replace_pretrained_head: video headをC classesへ。
- ZeroOutputModel.forward: [B,C]全0 logitsと独立な0 loss。配管debug専用。
- ABNResNet50.prepare_model/forward: attention branchとperception branchのCEを足す。
- ABNResNet50Model.__init__: CIFAR100 ResNet56をfeature/perceptionへ分割しattention branchを複製構築。クラス名と実際にloadするbackbone名は一致しない点に注意。
- __create_attention_branch: class activation [B,C,H,W]、1-channel attention [B,1,H,W]、aux logits path。
- __split_r50_into_half: feature extractorとavgpool+Linearへ分割。
- __get_last_num_features: 最後moduleのnum_features。
- get_attention_map: 直近forwardのmap。
- __gap: [B,C,H,W] → [B,C]。
- attention_branch: attention mapとaux logits。
- forward: x1をattentionで乗算しperception分類。maskは1を足さないため0〜1のgate。

入力shapeは画像modelが [B,3,H,W]、X3Dが [B,3,T,H,W]、出力は共通 [B,C]。

#### dataset

- DataloadersInfo: train/val loaderとclass数。
- configure_dataloader: dataset名からtransformとfactoryを選ぶ。
- TrainValDataModule.__init__: loaderを一度構築。
- train_dataloader / val_dataloader / n_classes: Lightningへ公開。
- TransformVideoInfo / TransformImageInfo: transform hyperparameter dataclass。
- identity: transform debug breakpoint用no-op。
- transform_video: temporal subsample、0〜1化、Kinetics系normalize、train random scale/crop/flip、val fixed scale/center crop、audio除去。出力videoは通常 [C,T,H,W]。
- transform_image: CHW float32化、ImageNet normalize、train random crop/flip、val resize/center crop。
- VideoFolderInfo、ImageFolderInfo、Cifar10Info、ZeroImageInfo: loader設定。
- collate_for_video: dict batchを(video,label) tupleへ。
- video_folder: PyTorchVideo dataset、train random clip、val fixed clips、class照合、DataLoader。
- LimitDataset.__init__/__getitem__/__len__: decode failureへ備えiteratorを2回連結し、epoch長をnum_videosへ固定。
- image_folder: class directory構造からtrain shuffle/val fixed loader。
- cifar10: downloadを許す標準CIFAR10 loader、C=10。
- ZeroImageDataset.__init__/__len__/__getitem__: zero HWC imageとindex mod C label。
- zero_images: 同じsynthetic datasetでtrain/val loader。

#### training

- TrainConfig / TrainOutput: gradient accumulation/log間隔とepoch結果。
- LoggerChecker.should_log: global step modulo interval。
- OptimizerChecker.should_zero_grad: accumulation windowの先頭。
- OptimizerChecker.should_update: batch indexがintervalの倍数。
- train: model.train、forward、loss.backward、top-k、interval optimizer step、epoch scheduler。
- ValidationOutput / validation: no_grad eval、全batch logging、平均metric。
- TqdmEpoch: start_epoch+1からnum_epochsまで。
- prepare_training: logger、data、CUDA model、optional DataParallel、optimizer/scheduler、resumeを構築。
- ValidationChecker.should_validate: intervalまたはfinal epoch。
- main.py main: epoch train→条件付きval→local/Comet checkpoint。
- main_pl.py main: Lightning logger/data/model/callback/Trainer構築とfit。
- SimpleLightningModel.__init__: factory modelとconfigを保持。
- configure_optimizers: optimizer/scheduler。
- configure_callbacks: val_top1上位2 checkpoint。
- log_train_loss_top15 / log_val_loss_top15: distributed sync metric logging。
- training_step / validation_step: forward、top1/top5、logging。backward/device/no_gradはLightningへ委譲。
- configure_optimizer: SGDまたはAdam。
- configure_scheduler: StepLR(10 epochs, ×0.1)またはdummy。
- dummy_scheduler: factor 1のConstantLR。
- configure_callbacks: ModelSummary。
- configure_logger / configure_logger_pl: Comet experimentとtag。

gradient accumulationはlossをintervalで割らず「gradientの和」をstepする。またepoch末のbatch数がintervalで割り切れない場合、残りgradientをstepする処理はない。これはコード上の挙動で、意図は明記されていない。

### E. 共通utility

- canonical_json_bytes: key sort、空白なし、UTF-8、NaN禁止のhash用JSON。
- sha256_bytes / sha256_file: memory/file SHA-256。
- fsync_directory: rename後のdirectory metadataを永続化。
- write_bytes_atomic / write_json_atomic: sibling temporary → fsync → replace。
- read_json: UTF-8 JSON。
- plain_config: OmegaConfをdeep-copied plain dictへ。
- load_config_group: simple nameだけ許しrepository YAMLを読む。
- validate_path_component: directory componentのpath traversalを防ぐ。
- git_output: git subprocessのtrimmed stdout。
- utc_now: UTC ISO timestamp。
- device_identity: CPUまたはvisible CUDA群とselected device。
- collect_provenance: origin/branch/commit/dirty/diff hash、pinned loader、versionsを検証・記録。
- scope_tag / display_tag: Cometのproduction/smokeと表示token。
- start_experiment: 新規/既存Cometを開始。network失敗はlocal処理を止めずstatus化。
- experiment_key / log_metrics / end_experiment: null-safe Comet操作。
- _flatten: nested dictをdotted parameter名へ。
- artifact_files: relative file→SHA。
- log_artifact: upload直前にlocal hashを再検証しmetadataへstatus。
- record_comet_status: mutable sidecarへComet状態をatomic更新。
- remove_layer_prefix_from_state_dict / add_layer_prefix_to_state_dict: DataParallelのmodule.変換。
- save_to_checkpoint / load_from_checkpoint: 旧教師ありpathのmodel/optimizer/scheduler/counters保存復元。
- save_to_comet: 旧checkpointをComet modelとして登録。
- compute_topk_accuracy: logits [B,C]のtop-kに正解が含まれる割合。
- AverageMeter.__init__/update: sample-weighted running mean。
- AvgMeterLossTopk.__init__/update/get_meters/get_mode_name: loss/top-k meter集合。
- GetMetricsDictMixin.get_meters/get_mode_name: subclass contract。
- get_step_metrics_dict / get_epoch_metrics_dict: Comet用metric dict。
- TqdmLossTopK.__init__: postfix buffer。
- set_postfix_str_loss_topk: step/loss/top-k表示を組立。
- init_postfix_str / add_step... / add_loss... / add_topk...: 表示文字列helper。

## Mathematical / Research Context

### この実装が学習できるもの

Observed:

- 更新されるのはQuery Q/V LoRAとQuery projector。
- objectiveは同一chunkのraw/変換viewを近づけ、過去chunkを遠ざける。
- parameter stateとQueueは動画境界を越えて保持される。

Inferred:

- 逐次的に変わるデータ分布へLoRAを適応させたい。
- same-video past negativesにより、異なる時刻のchunkを識別させたい。
- EMA KeyとFIFO Queueでsingle-clipでも多くのnegativeを利用したい。

このobjectiveだけでは保証されないもの:

- frame順序を復元・予測できること。
- motion directionを表現すること。
- LoRAが静的背景ではなく動的情報を保持すること。
- Base ViTより下流性能が高いこと。

特に、frame encodingと2段のmean aggregationが順序不変なので、「時間情報」は入力演算から直接生まれるのではなく、逐次optimizer state、negative history、データ分布を通じてparameterへ残る可能性に依存する。この仮説は別評価が必要。

### 計算・memoryの概算

- 1 updateはQuery 16 frame + Key 16 frame、最大32 image forward相当。
- ViTはvalid frameだけ処理するためterminal padding分は節約。
- Queue tensor本体は4096×128×4 bytes ≈ 2 MiB（float32想定）。metadataとPython objectは別。
- Base ViTはQuery/Keyの2 copyを持つため、Queueよりmodel memoryが支配的。
- full loopはmixed precision、DDP、gradient accumulation、prefetchを実装していない。canonical launcherはvisible GPUを1枚に限定する。

## Alternatives and Trade-offs

| 現在の選択 | 利点 | 代替 / 注意 |
|---|---|---|
| frame-wise ViT + mean | 単純、画像pretrain再利用、shape固定 | VideoMAE/TimeSformer等なら時間attentionを持てるが計算と移植が重い |
| Q/V LoRA r=8 | Base固定、更新parameterを約98万に限定 | full fine-tune、QKV全部、MLP LoRA。どれが良いかはablationが必要 |
| PEFT | adapter保存/loadと対象module管理が標準化 | libraryを使っても対象名、freeze、gradientの単体検証は必要 |
| 128-d projector | contrastive空間と評価空間を分離 | projectorなしは単純だが768-d Queueとobjective干渉が増える |
| EMA m=0.999 | Keyが滑らかでQueue consistencyを保ちやすい | momentum調整が必要。大きすぎると追従が遅い |
| FIFO 4096 | bounded memory、実装が明快 | batch negatives、memory bank、age weighting |
| all-past | strict onlineでfutureを見ない | 隣接chunkのfalse negative、古いencoder表現とのstaleness |
| GBR + flip | 強いview差でinvarianceを要求 | 色意味破壊。標準crop/color jitterとの比較が必要 |
| fully-contained chunk | label leakage/曖昧境界を減らす | 境界chunkと短segmentを多く捨てる |
| mean of chunk means | 実装が単純、各chunk同重み | frame-count weighted meanなら全frame同重み |
| final-epoch probe | validationでepoch選択しない | early stoppingより公平性は明快だがunder/overfitを見逃す |
| strict artifact hash | 比較条件と再利用が安全 | metadata/schemaが重く、変更時のmigration負担が大きい |

## 重要な読み方と注意点

1. ViTLoRAMoCoのdocstringにStage 5とあるが、canonical configはStage 6B v2を読み、full engineでも同classを使う。名前・docstringより呼び出し元とconfigを優先して読む。
2. evaluation snapshotは「Query LoRAだけ」。MoCo projectorの性能をprobeしているのではない。
3. Base/LoRA comparisonは同じmanifestと768-d pre-projector featureを使うので、意図した差分はLoRAの有無。
4. reduced profileは動画集合をhash rankで選ぶが順序をrandomizeしない。オンライン順序はAdapter順のまま。
5. code内の多数のRuntimeErrorは冗長ではなく、研究artifactのsilent mismatchを早期停止するための実験契約である。
6. 2026-10-08 MTGの未解決事項により、静的に正しい更新順序が見えてもloss振動・勾配急増の原因は確定しない。

## テストから読み取れる意図

テストは実行していない。静的に確認した主要契約:

- ViT/LoRA: target、Base freeze、CLS shape、gradient mode。
- Masked Mean: padding除外、順序不変、validだけへのgradient。
- protocol: 3軸独立、metadata不変、FIFO 4096、explicit negatives。
- streaming: no-prefetch、Reader cleanup、warm-up、update order、capacity overflow。
- full run: 動画境界state保持、exact resume、stop/final、dirty拒否、decode failure。
- checkpoint: Queue/RNG round-trip、snapshot LoRA-only、tamper rejection、Base fingerprint。
- manifest: closed interval、duplicate annotation、200-label Gate、tamper rejection。
- feature: mean定義、Base/LoRA ID一致、timestamp再照合。
- probe: classifierだけ更新、seed scope、confusion/macro、sample std。
- selection/config: selection seedとmodel seed分離、stable hash、trackingを科学identityから除外。

## Learning Resources

このコードを順に学ぶなら、次のローカルsourceが最短経路である。

1. integration/sequential_vit.py
2. model/aggregators/masked_mean_clip_aggregator.py
3. model/backbones/vit/vit_lora_frame_encoder.py
4. self_supervised/moco/vit_lora_moco.py
5. training/streaming_moco.py
6. training/streaming_moco_full.py
7. evaluation/activitynet_manifest.py
8. evaluation/segment_features.py
9. evaluation/linear_probe.py

背景概念として、次をこの順で復習すると理解しやすい。

- Tensor indexing、boolean mask、autograd
- ViT patch embedding、CLS、self-attentionのQ/K/V
- low-rank matrix factorizationとLoRA
- cosine similarity、softmax、cross entropy
- InfoNCE、momentum encoder、FIFO negative queue
- optimizer stateとEMA
- immutable experiment identity、SHA-256、atomic write
- confusion matrix、Top-1、macro accuracy、sample standard deviation

## Next Learning Steps

1. 1 chunkのshapeを紙に書き、V=13のterminal chunkでpaddingがどこで消えるか追う。
2. 小さいD=4、projection=2、Queue 2件の手計算InfoNCEを作る。
3. LoRAのA/B shapeと294,912 parametersを自分で再計算する。
4. train_stepの10段階を、parameter/gradient/Queueが変わる箇所に色分けする。
5. frame順をreverseしてもmasked meanが同じになることを数式で確認する。
6. 「Linear Probe改善」と「時間順序理解」が同義でない理由を説明できるようにする。
7. 次の実装判断へ進む前に、MTG方針どおりLoRA単体と標準MoCo単体の期待curve・gradient・similarityを別々に確認する。

## 根拠・未確認事項

### Observed

- dev@65dedb4e、clean、origin/dev一致。
- PEFT LoRAはq_proj/v_proj、r=8、alpha=8、dropout=0、bias=none。
- Base freeze、Query LoRA+projectorだけAdamW。
- Keyはno_gradとEMA。
- Queue 4096、projection 128、momentum 0.999、temperature 0.07。
- strict_single、GBR+flip、all_past。
- 16 frame/chunk、frame CLS 768、masked mean。
- Base/LoRA featureはraw pre-projector 768-d。
- Linear Probeは200 class、3 seeds、100 epochs。

### Aligned

- implemented Shared Streaming MoCo specの更新順、warm-up、3軸protocolとコードは構造上対応する。
- approved full-dataset specのsingle pass、video-boundary resume、LoRA-only snapshot、segment Linear Probeとコードは構造上対応する。
- implemented source-selection addendumのSHA rank、Adapter順復元、selection identity伝播とコードは構造上対応する。

### Inferred

- Q/V LoRA、GBR transform、all-past negativeが研究仮説に有効である理由。
- EMA 0.999、Queue 4096、temperature 0.07がこのdatasetに最適であること。
- source Adapter順がオンライン学習に望ましい順序であること。

### Unverified

- テストの現在commitでのPASS。
- 実ActivityNet file、annotation、decoder挙動。
- 実行時のdtype、GPU memory、throughput。
- loss振動、勾配急増の原因。
- snapshot/manifest/features/resultsの既存artifact状態。
- Base ViTに対するLinear Probe結果。
- temporal/motion informationの獲得。
- 外部論文や参照実装との完全一致。
