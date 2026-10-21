---
date: 2026-10-21
project: sequential-video-lora-analysis
record_type: implementation-spec
status: implemented
approved: 2026-10-21
implemented: 2026-10-21
topic: vit-ddp-moco-full-ft-lora-switch-smoke
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_reference_commit: 7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62
reference_moco_commit: a419ef282fc9885015674e77f58c59447206a27f
source_brainstorm: ../../../../secretary/notes/brainstorm/2026-10-21-full-vit-vs-frozen-vit-lora-moco-before-sequential-loader.md
evidence: ../experiments/2026-10-21-meta-moco-v2-e5-imagenet-5epoch-pair-parity-evidence.md
---

# ViT＋自作DistributedMoCo：Full FT / Frozen ViT＋LoRA切替・DDP短時間検証 spec（draft）

## 1. 目的とstatus

**status: implemented（2026-10-21、§13）**。残っていたBlockingは§12で解消して承認した。以下の§1〜§11は承認前の経緯として残し、§12と矛盾する箇所は§12を優先する。

既存の自作`DistributedMoCo`（Meta公式MoCo v2とResNet50・ImageNet-1Kの5 epoch比較で確認した実装）の基本機構を再利用し、画像事前学習済みViTに対して次の2方式をHydra設定で切り替え、**複数GPUのDDPで短時間の工学検証**を行えるようにする。

- `full_ft`: Query ViT本体＋Query Projectorを学習する。
- `lora`: pretrained Base ViTを凍結し、Query側の指定LoRA A/B＋Query Projectorだけを学習する。
- 両方式は共通のMoCo Query/Key、Key EMA、Key all-gather、InfoNCE、FIFO Queue、Checkpoint検証およびログ契約を用いる。
- 先に通常画像の基本経路を確かめ、動画フレーム、複数フレームchunk、Sequential Loaderは段階的に導入する。

**現時点はdraft（実装不可）**。2026-10-21にB2（自作DistributedMoCoでDDP）、B11（複数GPU）、B10（初期は両方式で同一Optimizer/LR）、AdamW共通採用を確認。さらにユーザーが**初期LR、Weight Decayの値と適用対象、Warmup/Cosine、DDP Global BatchをImageNet E5公式vs自作比較と合わせる**と明示採用したため、§3・§11に数値を固定する。その他のBlockingが残るためspec全体は未承認。

## 2. Authority / Evidence と実装境界

- ユーザー採用事項（本spec第3節）が最優先。
- [2026-10-15 MTG](../meetings/2026-10-15-mtg.md)：ResNet MoCo確認→ViT→LoRA→Sequential Loaderという段階的統合。
- [2026-10-21 E5 Evidence](../experiments/2026-10-21-meta-moco-v2-e5-imagenet-5epoch-pair-parity-evidence.md)：Meta公式vs自作MoCo、ImageNet-1K / ResNet50 / 2GPU / 各5epoch、記録指標bitwise・各epoch canonical model state一致。**これはViT/LoRAの正しさや表現品質の証明ではない**。元データはユーザー提示のSSH実験summaryであり、GitHubから実験の実体を直接読んだものではない。
- [探索brainstorm](../../../../secretary/notes/brainstorm/2026-10-21-full-vit-vs-frozen-vit-lora-moco-before-sequential-loader.md)：同一pretrained ViT、通常画像→固定動画chunk→Sequential Loaderの比較案。exploratoryであり、本specへ自動昇格しない。
- 基準コード：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、調査時HEAD `7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62`。GitHub上の状態のみ確認。SSH側dirty/unpushedやGPU実機の現在状態は未確認。
- 公式参照：`facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f`。
- 既存`scripts/parity/`のResNet50公式比較経路・E5採用recipe・Checkpoint形式・Comet識別子は**変更しない**。このViT作業を既存E5 Gateに混入させない。

## 3. 採用済みDecision Contract（ユーザー明示承認）

| ID | 採用内容 | 境界 |
| --- | --- | --- |
| B2 | `self_supervised/moco/ddp_moco.py`の自作`DistributedMoCo`を基盤にする。ViT Full FT / LoRAを同じDDP MoCoエンジンで切り替える | 独立の新MoCo計算エンジンを作らない。公式比較済みResNet50経路を壊さない |
| B10 | **OptimizerはAdamW**。Full FT / LoRA共通で**初期LR=0.03、weight decay=0.0001（各モードのQuery側trainable全parameterへ同じ規則で適用）、LR warmupなし、5epoch horizonのepoch-wise cosine**を採用。後段は等しい探索予算で方式別LR探索可 | これらの数値はE5のSGD用recipeの再利用であり、AdamWでのViT最適性・SGDと同一更新・公式parityを意味しない。変更が必要なら別承認 |
| B11 | **複数GPU DDP、2 GPU×local batch 128=global batch 256**を設定として採用 | E5 ResNet50では実行済みだが、ViT Full FT＋AdamW/LoRAでのVRAM適合は未実測。AMP/precision、実GPUの使用許可は別決定。OOM時の自動batch縮小はしない |
| 比較順序 | ViTの通常画像MoCoのFull FT対LoRAを先に比較可能にし、その後にSequential Loaderを導入する | 動画chunkやオンライン制約は今回の実装scopeへ混入させない |

## 4. 現行コード調査と設計制約

1. `StandaloneMoCo`は外部`backbone.feature_dim`を要求し、Query/KeyのBackbone＋MLP Projector、EMA、初期random normalized Queue、InfoNCE、enqueueを持つ。
2. `DistributedMoCo(StandaloneMoCo)`はDDP下でKey-input global shuffle/unshuffle、全rank Keyのgatherとenqueue、`info_nce_impl=self|meta_einsum`を提供する。E5の自作側は`meta_einsum`を使った。
3. `model/vit/vit_lora_frame_encoder.py`はViT CLS出力`[B,768]`、Frozen Base、Q/K/V選択と自作`LoRALinear`を持つ。一方、`ViTModel`ロード直後に`vit.layers[*].attention.q_proj`等を期待する実装は、実際のTransformers `ViTModel`でのpath適合を**未検証**。FakeViT unit PASSだけを実checkpointへの正常注入の証明にしない。
4. `model/vit/vit_frame_encoder.py`はViTを凍結するので、`enabled=false`をFull FT切替として流用できない。
5. `StandaloneMoCo.query_parameters()`はQuery全parameterを列挙する。LoRA時のOptimizerへFrozen Baseを渡さず、`requires_grad`と対象ID集合を照合する。一方、EMAのQuery/Key対応関係が壊れないよう、Optimizer用のtrainable listと全parameter順序を混同しない。
6. Frozen Baseをbitwise不変とするには、EMAでfrozen parameterを算術更新しない設計とそのtestを検討する。Key初期state一致、LoRA Key/Projectorの追従、DDP rank間一致を必須とする。
7. 既存`scripts/parity/moco_imagenet_10epoch.py`はResNet50と採用recipeを固定するため、新しいViT DDP runは別のHydra preset/entrypointで実装し、既存parity runnerは変更しない方針。

## 5. 実装scope（候補、未承認事項は第7節）

- 同じチェックポイント由来のViT CLS`[B,768]`を返すBackbone interfaceをFull FT / LoRAで統一し、`feature_dim=768`を提供。
- Hydraの明示的モード`model.train_mode=full_ft|lora`（名称は採用候補）で切替。初期状態が同じこと、無関係なclassifier headがないことを確認。
- **両方式共通OptimizerはAdamW（採用済み）**。Full FTではQuery ViT本体＋Projectorを更新し、LoRA方式ではBase完全freezeでLoRA A/B＋Projectorだけを更新する。AdamWには`requires_grad=True`のQuery学習対象のみ登録し、Key gradientなし。初期共通LR=0.03、weight decay=1e-4（Query trainable全parameter）、LR warmup=0、cosine epoch horizon=5を同じ規則で用いる。
- Projector候補：`768 -> 768 -> 128`、L2 normalization。MoCoのQueue/EMA/InfoNCE計算順は既存共通機構を再利用。
- DDPでは全rankの初期Query/Key、broadcast/gradient all-reduce、Key shuffle/unshuffle、Queue pointer・buffer一致をテストする。LoRAのdropout・未使用parameterによるDDP不具合も検出。
- checkpoint candidate：モデルQuery/Key＋Projector＋Queue/ptr、optimizer/scheduler、RNG・sampler cursor、モード/LoRA構成/ViT checkpoint identityを記録し、別インスタンスへ厳密restore。異なるモードのresumeを拒否。
- データ用preset候補はImageNet-1K少量画像の2-view smoke、次にActivityNetの単一フレームを通常DataLoaderで供給。split、frame index、video ID、timestampとサンプリングManifestを保存できる形を想定する。
- 回帰：既存MoCo DDP parity / ResNet50 E5受入契約、ViT/LoRA standalone unit、Config testを変更前後で比較。E5長時間GPU再実行を自動で要求しない。

## 6. 検証Gateの設計候補（性能成功とは別）

| Gate | 観測する合格条件 |
| --- | --- |
| G0：ViT/LoRA注入 | 実checkpointの層構造・Q/V注入箇所・初期CLS shape・initial-equivalence、LoRA A/B初期化の検査 |
| G1：2-rank CPU synthetic DDP | mode切替、all-rank Query初期同期、Key gradientなし、all-reduce、Key gather、FIFO pointer、Queue完全一致、InfoNCE finite |
| G2：trainability | Full FTのViTとProjectorが更新、LoRAのBase（Query/Key）がbitwise不変、LoRA Bおよび複数step後のAが更新、Projector更新、EMA数式と順序が一致 |
| G3：GPU短時間pilot | 2GPUで1step→少数step、同一設定で両モードを実行。各rankのVRAM peak/headroom、step time、loss/勾配、Queue/K整合、OOM fail-fast |
| G4：save / resume | 2rankで新規保存と復元、モード/設定不一致の拒否、同一入力での特徴・次step状態の許容誤差内一致。失敗原因も記録 |
| G5：既存非回帰 | ResNet50 DDP unit、MoCo standalone、ViT-LoRA単体、Config契約が回帰しない |

Lossの単調下降、公式ResNet50とのViT数値bitwise一致、Linear Probe改善、動画の動的情報獲得は実装GateのPASS要件にしない。K=65536では最初の20stepでQueue一巡しないことを明示する。

## 7. Ambiguity Gate / 正式承認前に必要な判断

### Blocking：未確定（本draftをapprovedにしない理由）

（2026-10-21：下表のBlockingはすべて§12で解消済み）

| ID | 未決事項 | 現時点の候補・論点 |
| --- | --- | --- |
| B3/4/5 | ViT checkpoint、LoRA injection path、切替名 | `google/vit-base-patch16-224`、Q/V・rank8・alpha8・dropout0、`model.train_mode`を候補。実Transformers構造を調査・検証し、既存注入関数の扱いを決定 |
| B6/7/8 | Smokeで使う入力データと正式scope | ImageNet-1K固定実画像を初期接続診断、ActivityNet単一フレーム（通常Loader）を追加するか。ActivityNetはサンプリング枚数/interval/重複/manifest/split/negative policyを固定 |
| B9 | MoCo設定 | 公式v2/E5基準は`K=65536, m=0.999, T=0.2, projection=128`。実データpilotも同値で固定するか、Syntheticだけ縮小Kを許すか |
| B10-numeric | **解消済み：AdamW / 初期LR=0.03 / WD=0.0001・Query trainable全parameter / LR warmupなし / 5epoch cosine** | ユーザーの明示採用によりE5数値を採用。ResNet50 E5のSGDからAdamWへ変わるため、Optimizer挙動そのもののparityは主張しない。実験で不安定なら設定を自動変更せず結果を記録し別判断とする |
| B11-resource | **GPU数・batchは解消済み：2 GPU×local128=global256**。残件はGPU ID / peak VRAM・headroom / FP32かAMPか / OOM時の実行停止境界 | ViT Full FT＋AdamWのGPUメモリ適合は未検証。変更が必要なら新しい比較条件として別判断し、無断でbatchを縮小しない |
| B12 | Checkpoint粒度、resumeの合格誤差と方式変更時扱い | MoCo全state＋RNG＋sampler保存、cross-mode resume拒否候補 |
| B13 | GPU Smoke steps・プロベナンス・Comet | 1/10/20step等の候補、seed、input hash、モード別run/Comet保存先、rank別error処理を確定 |
| architecture | DDPモデル初期化・Key EMAのfrozen-parameter挙動 | Freeze対象、KeyのEMA対象、実HFモデルのLoRA構造、DDP unused parameter/通信順を検証した上で承認 |

### Non-blocking

helper名、testファイル分割、ログ可視化の内部実装、Hydra新presetのファイル名、local fixtureの一時出力先は、既存の構造を壊さない範囲で実装時に調整可能。

## 8. Scope外 / 実行境界

- 今回はSequential Loader、ActivityNet 16-frame chunk、strict-online学習、動画時間情報解析、Linear Probe・長時間のFull vs LoRA性能比較を実装・実行しない。
- E5既存ResNet50比較、公式clone、別repositoryのSequential Loaderを変更しない。
- GitHubへのspec保存は研究文書の更新のみ。**研究コードの実装、CPU/GPUテスト実行、データダウンロード、長時間GPU学習、Comet送信、code commit/pushは未許可**。
- 本draftの`status`が`approved`になるまでは、engineering-taskによる研究コードの変更を開始しない。承認時にはユーザーがBlockingへの判断とspec全体の内容を明示承認する。
- GPU runtime/cuda、データ実体、ローカルdirty/uncaptured checkpointはGitHubから検証不能。実行直前に研究サーバ上でpreflightする。

## 9. Implementation Handoff（未承認・実行不可）

- target: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、調査時 `7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62`
- planned change: ViT backbone adapter、Full FT / LoRA factory、DDP compatible trainable optimizer filter、専用Hydra preset/runner、checkpoint/metrics、tests
- preserve: `self_supervised/moco/standalone_moco.py` / `ddp_moco.py`のMoCo意味論とResNet50 E5数値比較の再現性、旧ログschema
- verification: 第6節、ただし第7節のBlocking解消まで数値契約は未確定
- authorization: **draftであり、コード変更・実行の許可なし**
- 2026-10-21更新：approved。変更scope・検証・実行許可は§12.3〜§12.5を正とする

## 10. Decision history（2026-10-21：B10 Optimizer追認）

- 2026-10-21、ユーザーがSGDとAdamWの客観的な総合比較を依頼。比較ではMeta公式MoCo v2のResNet50＋SGD、Meta MoCo v3のViT＋AdamW、pretrained ViTのFine-Tuningおよび現行LoRA単体SmokeのAdamW使用実績を区別した。
- アシスタントは、**今回のViT Full FT / LoRAに共通してAdamWを用いる方針を推薦**。ユーザーが続く「採用します」で明示承認したため、B10のOptimizer種を**確定（AdamW）**へ更新した。
- **当時未採用だった項目**：LR・WD、weight decayの除外規則、warmup/cosine horizon、Batch Size、精度、GPU runの許可。LR・WD・scheduler・batchに関する新しいユーザー採用は§11を優先し、精度・実行許可などは未決のまま。
- 既存ResNet50 E5はSGDを維持。AdamW採用はViT用の別runnerに限定し、公式ResNet50とのbitwise parity成果を一般化しない。

## 11. 追加の明示採用：E5 ImageNet-1K比較レシピから4項目を引き継ぐ（2026-10-21）

ユーザーの「初期Learning Rateの具体値、Weight Decayの値と適用対象、Warmup・Cosine Schedulerの設定、DDPのGlobal Batch Sizeは、公式MoCo vs 自作MoCoをImageNet-1Kで比較したときと合わせます」という明示指示により、次を固定する。基準は[2026-10-21 E5 Evidence](../experiments/2026-10-21-meta-moco-v2-e5-imagenet-5epoch-pair-parity-evidence.md)と研究コード`dev/conf/moco_imagenet_10epoch.yaml`のE5 Gate（実験時間軸の5epoch条件）。

| 決定項目 | 採用値 | 適用/区別 |
| --- | --- | --- |
| Optimizer | AdamW | 2026-10-21に別途採用済み。**E5の実際のSGDは変更しない** |
| 初期LR | `0.03` | 両方式共通。E5のSGD値をAdamWに移すため、高すぎる可能性あり。実験前後の観測で判断し、自動変更なし |
| Weight Decay | `0.0001` | Query側の**学習可能なすべてのパラメータ**に同じWDを適用。Full FTはViT本体＋Projector、LoRAはLoRA A/B＋Projector。KeyとFrozen Baseはoptimizerから除外。E5のSGD同様、trainable bias/normalizationパラメータも除外しない |
| Weight Decayの意味 | AdamWのdecoupled weight decay | E5のSGD `weight_decay`（勾配にL2正則化を加える方法）とは数式上異なる。**同じ数値は同じ更新挙動を意味しない** |
| LR Warmup | `0 epoch`（なし） | E5 configの`timing.warmup_steps: 10`はVRAM/時間benchmarkの**計測窓のwarmup**であり、LR warmupではない。誤用禁止 |
| Cosine LR | `cos=true`、5epoch horizon | epoch開始時に`lr(e)=0.03 * 0.5 * [1+cos(pi*e/5)]`（eは0始まり）。E5と同じepoch単位。短時間Smoke中でも総期間を20step等へ自動短縮しない |
| DDP world size | 2 | NCCL・各rank1GPU、初期構成 |
| Local / Global Batch | 128 / **256** | `2 × 128 = 256`。Full FTとLoRAで同値。Queue `K=65536`に対し256stepで初期random keyの一巡に相当 |

### 検証とリスク・停止条件

1. **SGDからAdamWへのOptimizer変更は意図的な差分**。Meta公式MoCo v2と自作ResNet50 E5のbitwise同一性はこのViT学習には適用できない。
2. pretrained ViT Full FT＋LoRAにAdamW LR0.03を使った学習安定性は**未検証**。NaN/Inf、極端なgradient・update、Loss・feature collapseの兆候はログ化し、有限性・Guardに違反したら止める。比較方法・学習率を無断変更しない。
3. **2 GPU×128でViT Full FT＋AdamWがメモリに収まるか未検証**。実行前にVRAM、trainable parameterとoptimizer-state、activationを診断し、OOM時はfail-closedで停止。Batchサイズを自動縮小、AMP有効化、gradient accumulationでの代替は行わない。変更はユーザー再判断と実験レシピの更新が必要。
4. 今回のscopeは**実装＋短時間DDP接続検証**まで。5epochはSchedulerのhorizonとして採用したものであり、ViTの5epoch学習runを許可したという意味ではない。実GPU実行と長時間runは別途許可が必要。
5. 同一Weight Decay規則は両方式で適用するが、ViT本体とLoRA行列で正則化されるパラメータは異なるため、**同一数値＝同一の正則化効果**ではない。後段の科学的比較ではこの差を明記する。
6. この明示採用によりB10数値とB11 batchは**解消済み**。第7節の他のBlocking（実ViT module path、dataset/manifest/augmentation、精度とGPU実行可否、checkpoint、smoke条件、EMA/DDP対応）は継続。**spec statusはdraft**。

## 12. 承認（2026-10-21）：Blocking候補の一括採用と追加決定

2026-10-21、ユーザーが「specの候補どおりで承認して進めてください」と指示した。候補に値がなかった4点（ViT入力の正規化、ActivityNet単一フレームの扱い、Pos/Neg方針、manifest生成の実行）とGPU実行範囲は、同日の追加質問への回答で決定した。これにより本specを`approved`とする。

### 12.1 承認前のread-only調査（2026-10-21、研究サーバ）

- 研究コード：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`、`dev`、HEAD `7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62`、clean（§2の基準commitと一致）。
- `.venv`：Python 3.12.3、torch 2.14.0+cu130、transformers 5.17.0、PyAV 18.1.0。
- HF cacheの`google/vit-base-patch16-224`を`ViTModel.from_pretrained(..., add_pooling_layer=False)`でofflineに読込んだ結果：
  - top-levelは`embeddings / layers / layernorm`。各blockは`layers[0..11].attention.{q,k,v,o}_proj`（768×768）。
  - pooler・classifierはない（checkpoint側の`classifier.*`は未使用）。bufferは0、`hidden_dropout_prob=attention_probs_dropout_prob=0.0`、parameter数85,798,656、attentionは`sdpa`。
  - 既存`inject_attention_lora`が期待する`layers[i].attention.q_proj/v_proj`の経路は実checkpointに存在する。§4.3の未検証点はG0 testで確定させる。
- checkpointのpreprocessorはmean=std=[0.5, 0.5, 0.5]、size 224。リポジトリ内の既存ViTコードはこれを`AutoImageProcessor`で使っている。一方、E5のaug-plusはImageNetのmean/stdを使う。
- ActivityNet v1.3のannotation JSONではtrainingが10,024本、durationの中央値は114.0秒。10秒間隔・中央開始ならJSON duration基準の見積もりで117,652枚（約459 step/epoch @global 256）。実manifestは実際のstream長で再計算する。
- GPUはRTX A4500（20,470 MiB）×2（調査時idle）。この調査shellでは`CUDA_VISIBLE_DEVICES`が空のため、torchからGPUは見えない。
- 実装前のpytest baselineでは`test/dataset/test_video_folder.py`（NAS上のKinetics400を読む）が既に失敗していた。本作業とは無関係で、G5では変更前後の差分で判定する。

### 12.2 Blockingの解消（採用値）

| ID | 採用内容（2026-10-21承認） |
| --- | --- |
| B3 checkpoint | `google/vit-base-patch16-224`を`ViTModel(add_pooling_layer=False)`で読み、CLS＝`last_hidden_state[:, 0]`、`feature_dim=768`。hub commit hashと、LoRAを除くpretrained ViT stateのsha256をrunに記録する |
| B4 LoRA | 既存の`inject_attention_lora`と`LoRALinear`を再利用する。対象はQ/V（`layers.{0..11}.attention.q_proj/v_proj`の24箇所）、rank 8、alpha 8、dropout 0。初期値はAがkaiming-uniform、Bがzero。Full FTは同じcheckpointをLoRA注入なしで全parameter学習する |
| B5 切替名 | `model.train_mode=full_ft\|lora`。既定値は置かず、未指定は拒否する |
| B6 入力正規化 | E5 self側のaug-plus（RandomResizedCrop scale (0.2, 1)、ColorJitter(0.4, 0.4, 0.4, 0.1) p=0.8、Grayscale p=0.2、PIL Gaussian blur σ∈(0.1, 2) p=0.5、HFlip）を使う。Normalizeだけはcheckpointのpreprocessor値（mean=std=0.5）にする。両データ源・両モードで共通 |
| B7 ImageNet | `/mnt/NAS-TVS872XT/dataset/ImageNet/train`をImageFolderとして読み取り専用で使う。DistributedSampler（shuffle、seed 0）、drop_last、per-(epoch, rank) seeding、16 workers/rankはE5と同じ。「少量」は`training.max_steps`で消費stepを制限する意味である。epoch長はfull ImageNet（5004 step）のままなので、5epoch cosine horizonを短縮しない |
| B8 ActivityNet単一フレーム | **今回のscopeに含める**。`data.source=imagenet\|activitynet_frames`で切り替える。<br>・split：training。v1.3 JSONのsubsetを`sequential_loader.ActivityNetAdapter`で検証した10,024本。<br>・時刻：**10秒間隔、t = 5 + 10n 秒（n ≥ 0）、t < 実video streamの長さ、動画あたりの上限なし**。<br>・フレーム：「時刻tに表示されているフレーム」＝pts ≤ tの最後のフレーム。PyAVで直前のkeyframeへseekし、前方へdecodeして得る。<br>・Manifest：`manifest.json`＋`frames.jsonl`を新規directoryへ1回生成して凍結する。内容はvideo ID、相対path、split、timestamp、推定frame index、stream長・平均fps、JSON duration、policy、各sha256、adapter/loader commit。<br>・読めない動画が1本でもあれば、manifestを書かずに停止する（silent skipなし）。<br>・この10秒は**smoke用manifestの間隔**である。正式な学習間隔はbrainstormの類似度診断の後に別途判断する |
| B8 Pos/Neg | Positiveは同一フレームの2-viewだけとする。同一動画由来のnegativeは除外しない（MoCo本体は変更しない）。監査のためvideo IDをmanifestに残す |
| B9 MoCo | runnerの既定値とGPU run：K=65536、m=0.999、T=0.2、Projector 768→768→128、InfoNCEは`meta_einsum`（E5 self側と同じ演算経路）。K・batch・画像サイズ・ViTの縮小は、CPUのsynthetic/unit testでだけ許す（GPU runでは拒否する） |
| B10 detail | `torch.optim.AdamW`、betas (0.9, 0.999)、eps 1e-8、amsgrad false（PyTorch既定値）、lr 0.03、wd 1e-4。単一param group＝Query側のtrainable全parameter（Full FT：ViT＋Projector、LoRA：LoRA A/B＋Projector） |
| B11-resource | FP32、AMPなし、TF32 off（`fp32_precision=ieee`）、cuDNN deterministic、benchmark off（E5と同じ）。VRAM headroom gateは1024 MiB（E5 V1と同じ）で、下回ればBLOCKED。OOM・NCCL error・非有限値・headroom不足では停止し、batch・AMP・解像度・accumulation・K・GPU数を自動で変更しない。GPU IDは実行時の`CUDA_VISIBLE_DEVICES`で指定し、run recordに残す |
| B12 Checkpoint/resume | **保存**：`training.checkpoint_steps`で指定したglobal stepの終わりに保存する。rank 0がmodel全state（Query/KeyのViT・LoRA・Projector、Queue、queue_ptr、real_key_count）、AdamW state、epoch/step_in_epoch/global_step、config identity（train_mode、checkpoint identity、LoRA構成、データ識別子を含む）を書く。各rankはRNG sidecarを書く。<br>**resume**：新しいrun directoryで行う。identityの不一致（別train_modeなど）は、全rankでstateを読む前に拒否する。sampler cursorは、同じepoch seedでiteratorを作り、消費済みのbatch数を読み飛ばして再現する。<br>**合格基準（CPU、gloo）**：復元直後のstateが保存時とbitwise一致し、resume後の継続がuninterrupted runとbitwise一致すること。GPUでの許容誤差はG3を承認するときに決める |
| B13 Smoke条件 | G3のstep数は1/10/20とし、GPU runでは`max_steps`∈{1, 10, 20}以外を拒否する。seed 0。入力hashは全stepで記録する（各rankのim_q/im_k sha256）。run directoryは`log/moco_vit_ddp_smoke/<train_mode>/<data.source>/<timestamp>`でモード別に分ける。**Cometには送らず**、local artifactだけを記録とする。rankごとのfailure（stage、step、例外、OOM時のmemory summary）を`rank{r}_report.json`に残し、どのrankが失敗してもrun全体を失敗とする |
| architecture | Query/Keyは既存どおり別storageのdeepcopyにする。EMAは`requires_grad=True`のQuery parameterに対応するKeyだけを更新する。これは新subclassで実装し、`standalone_moco.py`と`ddp_moco.py`は変更しない。Frozen BaseのKeyはQueryとbitwise一致を保つ。Optimizerには`trainable_query_parameters()`（requires_grad=Trueのみ）を渡し、EMA用の全parameter順序とは分ける。DDPは`find_unused_parameters=False`、`broadcast_buffers=True`（E5の既定）。初期化はseed→LoRA A、再度seed→Projector/Queueの順に行い、両モードのProjectorとQueueの初期値を一致させる |

### 12.3 実装scope（approved）

1. `model/vit/vit_moco_backbone.py`：`ViTMoCoBackbone(checkpoint_id, train_mode, LoRA設定)`。CLS `[B, 768]`と`feature_dim`を返す。
2. `self_supervised/moco/vit_ddp_moco.py`：`DistributedMoCo`のsubclass。追加・変更は、frozen-awareなEMAと`trainable_query_parameters()`だけ。
3. `scripts/vit_moco/activitynet_frames.py`：manifest生成CLI、timestamp規則、単一フレームのdecode、manifest dataset。
4. `scripts/vit_moco/moco_vit_ddp_smoke.py`と`conf/moco_vit_ddp_smoke.yaml`：DDP runner。build、data、AdamW、epoch cosine、step metrics（loss、正負cosine、最難negative、lr、queue_ptr、trainable grad norm、Query特徴std、step time）、VRAM/headroom、checkpoint/resume、終了時のrank間state一致とFrozen Base不変の確認、run report。
5. tests：G0（実checkpoint）、G1/G2（2-rank CPU、実Transformers `ViTModel` classを小configで使用）、G4（runnerをCPUで実行）、config契約、ActivityNetの時刻規則・decode・manifest。

**変更しないもの**：`self_supervised/moco/standalone_moco.py`、`ddp_moco.py`、`scripts/parity/*`、`conf/moco_imagenet_*.yaml`、`model/vit/vit_lora_frame_encoder.py`、既存test。

### 12.4 検証Gateの実施区分

| Gate | 今回の扱い |
| --- | --- |
| G0 | 実施：CPU pytest（HF cacheの実checkpoint）。注入箇所、CLS shape、pooler/classifierがないこと、LoRA A/Bの初期値、両モードのinitial-equivalence |
| G1 | 実施：2-rank CPU（gloo）、小configの実`ViTModel`。両モードについて、初期同期、Key gradientなし、勾配all-reduce、shuffle/unshuffle、Key gather、FIFO pointer、Queueのrank間一致、InfoNCEの有限性 |
| G2 | 実施：同じ構成で、Full FT更新、LoRAのBase（Query/Key）のbitwise不変、step 1でのLoRA B勾配、step 2以降のLoRA A勾配、Projector更新、EMAの数式と順序 |
| G3 | **未許可**（2026-10-21回答：実装とCPU検証まで）。runnerは用意するが、GPUでは実行しない |
| G4 | 実施：2-rank CPU（gloo）でrunnerを通す。小ViT、小ImageFolder、合成動画を使う。GPUでのG4はG3と一緒に別途承認する |
| G5 | 実施：既存pytest全体を変更前後で比較する |
| ActivityNet manifest | 実施を許可（2026-10-21回答）：実データから生成する（CPU、NAS読み取りのみ、新規directoryへ）。少数フレームの実decode確認を含む |

### 12.5 実行境界（承認後）

- 許可されているもの：研究コードの実装、CPUでのtest/smoke、ActivityNet manifestの生成（読み取りのみ）。
- 未許可のもの：GPU run（G3とGPUでのG4）、Comet送信、長時間学習、データのダウンロード、code commit/push。これらには別途の明示許可が必要。
- 学習の成功（Loss低下、表現品質、Full FTとLoRAの優劣）は、今回の実装Gateでは主張しない。

## 13. 実装記録（2026-10-21）

- §12.3のscopeを、研究コード`dev` `7fc1f56`の上に新規ファイルだけで実装した（既存ファイルの差分0、**未commit**）。
- §12.4のG0、G1、G2、G4、G5はCPUでPASSし、ActivityNet manifestの生成（10,024本、117,522枚）と実decode確認（64/64）も完了した。
- **G3とGPUでのG4は未実施**（未許可）。runnerはGPU runをcleanなcommitからしか起動しないため、G3の前に研究コードのcommit許可が必要。
- 詳細は[実装・CPU検証記録](../experiments/2026-10-21-vit-ddp-moco-full-ft-lora-implementation-cpu-verification.md)を参照。
- 2026-10-21追記：ユーザーの指示により`dev`へcommit（`9348efe` feat、`2c26991` test）し、`origin/dev`へpushした。GPU runに必要なcleanなcommitは満たした。G3の実行は引き続き別途の許可が必要。
