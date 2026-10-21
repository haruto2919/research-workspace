---
date: 2026-10-21
project: sequential-video-lora-analysis
record_type: implementation-spec
status: draft
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

既存の自作`DistributedMoCo`（Meta公式MoCo v2とResNet50・ImageNet-1Kの5 epoch比較で確認した実装）の基本機構を再利用し、画像事前学習済みViTに対して次の2方式をHydra設定で切り替え、**複数GPUのDDPで短時間の工学検証**を行えるようにする。

- `full_ft`: Query ViT本体＋Query Projectorを学習する。
- `lora`: pretrained Base ViTを凍結し、Query側の指定LoRA A/B＋Query Projectorだけを学習する。
- 両方式は共通のMoCo Query/Key、Key EMA、Key all-gather、InfoNCE、FIFO Queue、Checkpoint検証およびログ契約を用いる。
- 先に通常画像の基本経路を確かめ、動画フレーム、複数フレームchunk、Sequential Loaderは段階的に導入する。

**現時点はdraft（実装不可）**。2026-10-21のユーザー発言「B2はDDPを使う」「B11は複数GPU」「B10はLRをなぜ揃えないか」および、続く「採用します」により、以下のB2/B10/B11方針が採用された。他の候補設定・実験条件を一括承認されたとは解釈しない。数値・データなどのBlockingが残るため、spec全体は未承認。

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
| B10 | **初期の接続検証・条件統制比較では両方式に同じOptimizer種、同じ初期LRとスケジュールを使用する**。将来の性能比較では方式別LR探索もできる設定設計にする | 同じLRが両方式の最適LRという主張はしない。Optimizer種・初期LR数値・scheduler horizonは未決 |
| B11 | **複数GPUのDDP**で実行する。現在の再現実験と同じ2 GPUを初期構成の候補・検証基準とする | GPU ID、local/global batch、FP32/AMP、VRAM headroom、実GPU smokeの実行許可は別途確定 |
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
- Full FTのOptimizer：Query ViT本体＋Projector。LoRA方式：Base完全freeze、LoRA A/B＋Projectorのみ。Key gradientなし。
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

| ID | 未決事項 | 現時点の候補・論点 |
| --- | --- | --- |
| B3/4/5 | ViT checkpoint、LoRA injection path、切替名 | `google/vit-base-patch16-224`、Q/V・rank8・alpha8・dropout0、`model.train_mode`を候補。実Transformers構造を調査・検証し、既存注入関数の扱いを決定 |
| B6/7/8 | Smokeで使う入力データと正式scope | ImageNet-1K固定実画像を初期接続診断、ActivityNet単一フレーム（通常Loader）を追加するか。ActivityNetはサンプリング枚数/interval/重複/manifest/split/negative policyを固定 |
| B9 | MoCo設定 | 公式v2/E5基準は`K=65536, m=0.999, T=0.2, projection=128`。実データpilotも同値で固定するか、Syntheticだけ縮小Kを許すか |
| B10-numeric | **Optimizer種、初期共通LR、weight decay、schedulerとhorizon** | AdamWはViT候補だが**未採用**。SGDを使えば公式基準に近いがViTに適するとは限らない。両方式の初回比較の数値を固定する |
| B11-resource | GPU数、local/global batch、VRAM制限、precision | 2GPUを初期候補にする。Full FTとLoRAで同じglobal batchを採用できるか、実測で判定。予算不足の自動変更を禁止するか決定 |
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
