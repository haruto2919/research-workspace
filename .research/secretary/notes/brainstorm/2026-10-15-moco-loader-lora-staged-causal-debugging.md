---
date: 2026-10-15
project: sequential-video-lora-analysis
source_todo: null
topic: moco-loader-lora-staged-causal-debugging
status: exploratory
tags: [brainstorm, moco, sequential-loader, custom-lora, isolation, validation, ablation]
---

# MoCo → Loader → LoRA の段階的統合・原因切り分け計画（探索）

## 発端と中心の問い

ユーザーは、自作LoRAとSequential Loaderの接続確認のためだけに検証用MSEを新規導入するより、先に本来の研究対象であるMoCoを単体検証し、Sequential Loader＋MoCo、さらにSequential Loader＋MoCo＋自作LoRAへ進んだ方が、異常時の原因を特定しやすいのではないかと提案。**それぞれの段階で何を一つだけ変えるか、どの対照で責任範囲を切り分けるか**が論点。

## 根拠と現在確認できること

- [2026-10-08教授MTG](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-10-08-mtg.md)：統合Stage6bでMoCo lossが振動、Query LoRA勾配が後半急増。Sequential Loader・LoRA・MoCoの役割を独立検証した上で再統合する方針が明示。
- [2026-09-24 Stage5実験](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-09-24-stage5-moco-one-step-verification.md)、[2026-09-28 Stage6実験](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-09-28-shared-streaming-moco-verification.md)：従来PEFT LoRA＋動画用MoCoの1-step / syntheticのengineering確認あり。ただし現在の自作LoRAと同一実装ではない。
- [2026-10-08 loss/gradient探索](./2026-10-08-stage6b-v2-learning-curve-interpretation.md)：Projector正規化前norm、same-video negative、onlineの非定常入力、AdamW driftなどが未検証仮説。lossの単調減少を合格条件にしない。
- [2026-10-14次段階メモ](./2026-10-14-post-lora-standalone-next-moco-validation.md)：自作LoRAのCIFAR10 standalone確認、Loader＋自作LoRAのMSE 2-step検査は候補だったが、今回の相談により「先にMoCoを実装・単体検証し、その損失で接続を確かめる」という別の順序を比較対象とする。
- GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、2026-10-15閲覧時HEAD `7dcd194f439c7e28d796e2986857ab0df1982cbe` のtreeでは `model/moco`、`training/moco` 等のMoCo実装は確認できない。現行で確認できるのは `integration/sequential_vit.py`、`model/vit/vit_lora_frame_encoder.py`、`model/masked_mean_clip_aggregator.py` など。SSH側の未コミット・未push worktreeはGitHubから確認不能。

## 代替案・トレードオフ

### A. 自作LoRA＋Loaderの検証用MSE → MoCo単体 → 統合

長所：2chunk/2stepで自作LoRAまでのautograd/更新を比較的小コストで確かめられる。短所：本来の学習目的InfoNCEではないテスト専用経路が増える。MoCoによる正負例とQueue条件の問題には触れない。テストとして残す余地はあるが**本線の先行条件にはしない**。

### B. ユーザーの提案：MoCo単体 → Loader＋MoCo → Loader＋MoCo＋自作LoRA

長所：研究に用いる目的関数だけで前進し、接続の失敗箇所を順序立てて絞れる。短所：画像1枚→動画16frame、dataset変更、Aggregator追加、通常Loader→Sequential Loaderを同時に入れると「Loaderを加えた」という比較にならない。MoCo単体でtrainable ResNet等を使い、Loader＋MoCoでfrozen ViTへ変更すると、モデルとoptimizer/学習対象も同時変更となる。**方針は推奨、間に対照ステージが必要**。

### C. 最初から全結合したActivityNet full run

時間・GPUが重く原因が分からない。過去の問題を再発しかねないため非推奨。

## 推奨する段階設計：比較軸を1つに固定する

| Stage | Encoder / 対象 | 入力経路 | 更新対象 | このステージで検証する差分 |
|---|---|---|---|---|
| 0（確認済） | Frozen ViT＋自作Q/V LoRA＋分類Head | CIFAR10通常DataLoader | LoRA A/B＋Head | 単体LoRA実装 |
| 1a | 独立MoCo core（InfoNCE, Queue, EMA）＋小さいtrainable encoder | 人工feature / 通常画像Loader | Query encoder＋projector | MoCoメカニクス、参照値一致、no future leakage |
| 1b（必要なら） | 参照MoCo baseline（trainable ResNetなど） | CIFAR10通常Loader | encoder＋projector | 数値・学習曲線の参考、実装の独立照合。最終ViTの挙動保証ではない |
| 2a | **最終構成に近いfrozen ViT（LoRA無効）**＋MoCo Projector | 通常Loader、画像と同等の入力 | Query projectorのみ（KeyはEMA） | backbone/凍結・feature 768→128適応、MoCo依存を独立に確認 |
| 2b | 同一frozen ViT＋Projector・MaskedMean | 事前固定したActivityNet **16-frame chunk**（Loaderなし） | Projectorのみ | 動画化、valid/padding、aggregatorでのgradient・数値監査 |
| 3 | 2bと同じモデル、学習設定、**同一raw frame/timestamp/augmentation/入力順** | **Sequential Loader**経由 | Projectorのみ | Loaderによる接続・前処理・順序差分を独立検証 |
| 4 | 3へ自作Q/V LoRAを注入、Base frozen | 同一Sequential Loader / chunk | Query LoRA A/B＋Projector（Key LoRA/ProjectorはEMA） | MoCoのInfoNCEからLoRAへ勾配・更新・Base不変、2step以上 |
| 5 | 4で安定した構成 | strict-single逐次stream（研究対象の時系列順） | 同上 | online順序、negative policy、streaming distribution shift、数値安定性 |

**段階2b→3が最重要の比較**：同じrawフレーム列と固定transform seed・同一初期state・同一MoCo config・同一negative queue state・同じ更新順序を維持。出力、Loss、勾配、optimizer差分、EMA、Queue key/metadataの対応が所定精度内で一致することを検証。Loaderが順序を変えれば、それ自体を設計違反として検出。事前decodeは短時間の対照用であり、Stage5本番のstrict-online future-data不利用条件とは区別。

**段階3→4の注意**：LoRA無効のViTまでtrainableにしてしまうと、stage4へ進むときBase凍結／学習対象変更とLoRA注入が同時に発生する。Stage2a以降はBaseをfrozen、Query Projectorのみをtrainableに統一する方が切り分けやすい。このMoCo適応は標準的な全面学習MoCoそのものではなく、mechanics/Projectorの対照条件と明記。

**段階4の追加の対照（失敗時必須）**：同じ固定chunkに対し`Loaderなし＋MoCo＋LoRA`を実施してLoaderあり版と比較。両方で失敗→MoCo/LoRA結合・optimizer/EMA・autogradの可能性を優先。Loaderありだけ失敗→入力変換/metadata/順序/readerの交互作用を疑う。Stage4とStage3のLoss値一致はLoRAの追加により通常成り立たないので、数値同等性を成功条件にしない。LoRA ON初期B=0ではeval/同一stateの初期出力がLoRA OFFと一致するべき。

## 合格条件と診断ログ候補

- **Stage1**：人工InfoNCEが参照数値と一致、positiveはcurrent same sample、negativeはpast keysのみ、Queue FIFOとcapacity、Key勾配0、EMA式一致、Query/Projector grad finite/nonzero、実画像multi-stepで有限かつ非崩壊の基本診断。Lossが毎step低下することは要求しない。
- **Stage2**：同一trained/frozen modelの参照値・shape・finite、CLS frame `[16,768]`、clip `[768]`、Projected key/query `[128]`、padding除外。same-image vs video chunkの比較は同じseed設定でも値が異なるのが自然であり、グラフ値のみでバグ断定しない。
- **Stage3**：2bと同一raw inputの`torch.equal`、feature/masked mean/logits/loss/gradと更新結果の許容誤差内一致。Loaderがmetadata/future orderingを壊さない。Pairwise regressionを自動テストにする。
- **Stage4**：Query base changed0、Base gradients0、optimizer expected ID set一致、A/B非ゼロ勾配と changed tensors（Bゼロ初期につきA step1=0でよい）、Query projector更新、Key branch全体gradient0、EMA式、Queueへのdetached pre-EMA key登録、finite。MoCoを同一動画内で回す場合Negative policyとの整合が必要。
- **Stage5**：ステップ別MoCo loss（queue size同時記録）、positiveとnegative類似度（hardest neg・margin含む）、pre-L2 query/key Projector feature norm、feature variance/embedding collapse兆候、Query LoRA A/B global&層別 grad、AdamW実update norm、LR、Queue occupancy、有効negativeとsame-video比率、動画ID/chunk/timestamp、gradient急増step。先に短いpilot、次に固定条件での複数step、問題解消後にfull runへ。

## 注意と未決

- 入力channel GBR＋flipとMoCo標準augmentationを一度に変えない。Stage1では標準2-view、Stage2以降は比較する2 viewを同じ変換・seedへ固定し、最後のablationで変換だけ変更する。
- strict-single + different-sequence only ではcurrent動画のfresh queueに有効negativeがない場合がある。Stage1/2/3/4では有効negativeが確保できる小さい複数sample/sequence条件を設計し、Stage5のnegative policy（all-past等）を研究設定として別Gateで確認する。
- 2b→3でoffline chunkを事前作成して同一の生フレームを共有するか、Loaderのdecodeを厳密照合するかは実装/データアクセス制約を見てspecで固定する。
- Stage1bで使うreference MoCo/ResNet等、Stage2aのfrozen ViT pretraining checkpoint、MoCo projector architecture、negative policy、2-view、ステップ数、test toleranceは正式なspec化時に確定。今は推奨。
- 自作LoRAのCIFAR10 checkpointのtrained adapterはここへ持ち込む必要はなく、ImageNet ViT上にzero-B LoRAを初期挿入して差分を確認する（既学習Headは不要）。
- 既存GitHub dev上にはMoCo本体がないため「既存コードをそのまま変更する」か「SSH worktreeに残る古いMoCoを安全に移す」かはread-onlyで所在確認が必要。旧PEFTベースのMoCo codeを無監査に新自作LoRAへ差し込まない。

## Research Spec handoff（ユーザーの明示spec化待ち）

- 研究目的：以前のMoCo loss振動・勾配急増の原因特定を容易にするため、入力データ・Encoder・Loader・LoRA・時系列onlineを1因子ずつ導入し、MoCo mechanicsの単体検証から逐次動画LoRAへ到達する。
- 第一spec候補：`MoCo mechanics + standalone image baseline`、Source参照と単体check契約・ログ・短時間pilotまで。次いで`ViT/Video Offline Bridge`→`Loader parity`→`Custom LoRA Integration`→`Strict Online controlled diagnostics`。
- 主要成功条件：段階ごとのfinite/checksに加えStage2b→3の同一入力の数値一致、Stage3→4のQuery/Key gradient/EMA境界とA/B更新、Stage5の数値安定性・異常発生時の因子帰属。
- 対象外：新規MSE学習ハーネス、すぐのActivityNet full run・Linear Probe研究上の結論・時間情報獲得の証明。
- Authority：全内容は探索的提案。ユーザーは順序を比較検討しているのみで、spec作成・コード改変・学習run・TODO追加を指示していない。
