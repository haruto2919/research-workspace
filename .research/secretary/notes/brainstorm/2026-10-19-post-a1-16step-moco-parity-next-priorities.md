---
date: 2026-10-19
project: sequential-video-lora-analysis
source_todo: null
topic: post-a1-16step-moco-parity-next-priorities
status: exploratory
tags: [brainstorm, research, moco, imagenet, parity, lora, vit, sequential-loader, roadmap]
---

# Meta MoCo A1・16step一致後の次課題 — 優先順位の探索記録

## 現在地（GitHub mainの最新Evidenceに基づく）

- 2026-10-19にA1 synthetic、ResNet50+MLP、K64、2GPU/NCCL、16stepのM vs S1（meta_einsum）がformal FP32 PASS、judged 16,256/16,256の全比較entry bitwise一致、M/S0/S1それぞれの再実行bitwise。S0従来式はfixed tolerance超過が9,542/16,256 entries。観測範囲は全16step small/step1,8,9,16 fullであり、ImageNet実画像・長期Queueの代替ではない。
- Research Workspace: `.research/lab/projects/sequential-video-lora-analysis/experiments/2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md`。
- 親Spec A `2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md`はGitHub上`draft`。A1の専用16step specは`approved`。親Spec AのA2（one-step、特に実画像224pxと比較条件）、A3（K65536 Queue cycle 1024step以降）、A4（checkpoint/resume）、A5（実画像100-300step VRAM/throughput）は未検証。Spec Bの10epoch、固定validation特徴評価、Linear Probeも未着手。
- 実装GitHub dev HEAD `1ce9badcbae3e2df08fa400944e4c385785063f9`。`DistributedMoCo.info_nce_impl`既定は`self`。`meta_einsum`はA1の1step/16step2つの診断profileに制限され、一般的なA2/A3/A5や通常のvideo pipelineにそのまま指定できない。`StandaloneMoCo`やCIFARを勝手に変更しない。
- 2026-10-15 MTGではLoRA/MoCo/Loaderを一度に統合せず差分を分ける方針。MoCo単体GPU64/500stepの工学チェックはPASSだが表現品質は未評価。段階統合の推奨は通常画像Frozen ViT+MoCo→固定16frame動画chunk→同一入力のSequential Loader parity→Q/V LoRA→strict online。ユーザーの大枠はLoRA単体→MoCo単体→Frozen ViT+MoCo+Loader→+LoRA→LoRA解析。

## 研究上の問い

制御された16step数値一致を、通常のMoCo基盤とその実画像・長期学習へどう安全に接続し、その後のFrozen ViT/動画Loader/LoRA統合に誤った工学的前提を持ち込まないか。

## 推奨順序（未採用の探索候補）

| 順 | 次タスク | 合格/退出の目安 | 証拠の限界・境界 |
|---|---|---|---|
| 1（最優先） | **InfoNCE公式演算を通常検証・学習経路で使う設計を決め、別specを作る**。既存`meta_einsum`のA1限定制約をどう扱うか。推奨はResNet50/DDPの公式parityを優先した明示opt-inとし、従来`self`を非回帰の基準として維持する方法を第一候補に比較。将来本番default変更は別判断 | デフォルト、trace/比較契約、scope、A2/A3/A5/ViTで使う演算の一貫性をspec化。CPU testと既存1/16step A1回帰 | 「A1だけがMeta式」で以降の旧算術へ戻っては先行parityの保証が引き継げない。いきなり全経路default変更はCIFAR・過去artifactの意味を変える |
| 2 | **実画像/224pxでのA2比較とA5小規模preflight**：ImageNet read-only inventory、実際のGPU/VRAM測定、固定実画像2viewでの公式vs自作1step、100-300step benchの可否判断 | 内容一致したfixture SHA/精度/grad/EMA/Queue、GPU peak、step time、I/O、OOMなし。小規模benchで先にVRAMを測定、localB32/K65536が収まらなければ勝手に変更せず再承認 | 16step K64/64px syntheticの成功はImageNet実画像/224px/実Kでの成功を保証しない。A5も個別run許可必要 |
| 3 | **A3・A4の状態継続検証**：K65536（globalB64のとき1024stepで一巡）のQueue長期/境界、checkpoint→復元の厳密な再現性 | 1024→1025の境界でQueue・ptr・EMA・optimizer・BN/DDPが一致。resume vs uninterrupted比較 | K65536/1025stepを実行する前に実機コスト・計測頻度・ログ容量・停止条件を別承認 |
| 4 | **MoCoの表現品質評価（Spec B候補）**：公式/自作fresh 10epochを同条件で比較、固定validation feature・retrieval/varianceを必須、Linear Probeは別Gate。予算が重いなら固定評価集合の短時間診断から開始 | 工学PASSとは別に、固定画像集合で表現が学習されたことを検証。複数seedや線形評価は後段 | loss減少・positive vs negativeだけでは良い表現の証拠でない。10epochで公式長時間精度の再現は主張できない |
| 5 | **研究主題へ段階統合**：Frozen ImageNet-pretrained ViT + MoCo (LoRA OFF、通常画像)→固定16frame ActivityNet chunk/MaskedMean→Sequential Loader parity→Q/V LoRA→strict online→LoRA時間解析 | 一度に変更する因子を一つに絞り、Query/Key/Projector/EMA/Queue、Base凍結、入力index/time/mask、LoRA A/B更新を確認 | 3→4の一致と特徴品質を分離。frame-wise CLS+masked meanはフレーム内順序不変なので時間順序理解の証明には別評価が必要 |

## 優先順位の競合と推奨

- **A（公式整合性優先）**：演算方式の利用範囲を先にspec化し、小規模な実画像A2＋A5 preflight、A3/A4、Spec Bの順で進む。Pros: 今回の厳密なparity成功を本番検証に接続できる。Cons: Frozen ViTへの統合開始が遅くなる。
- **B（研究統合を早める）**：まずMeta式の採用境界だけ決め、通常画像Frozen ViT＋MoCoの小規模engineering gateを先に進める。A2/A3/A5/表現品質は独立Gateとして並行計画。Pros: 2026-10-15 MTGで推奨した段階統合を早く進める。Cons: ImageNet ResNet50実画像での公式parityを先に確認していないことを明記する必要。
- **C（非推奨）**：S0既定のまま従来の動画full pipelineや10epoch本学習へ直行。Pros: 手元のコード再利用。Cons: 16stepの負対照で差が累積した算術を『Metaと一致した実装』と誤称し、異常の切り分けも困難。

**探索上の推奨**：最優先は『検証済み算術をどの経路に正式採用するか』のspec化。その後はMoCo公式検証（実画像A2とA5 feasibility）を短いGateとして通し、A3大規模1025step・Spec B 10epochへ無条件に突進せず、Frozen ViT接続を進めることを有力案とする。A3の要否は研究室が求めるMeta parity完全性とGPU予算を踏まえてユーザーが判断する。

## 次にユーザーが決めること

1. `meta_einsum`を特定のparity/train経路へ明示opt-inで解禁するか、一般的な`DistributedMoCo` defaultを変更するか。推奨は前者を先に比較し、CIFAR/Standalone/過去結果を保護。
2. 公式ImageNet ResNet50実画像A2/A5 feasibilityを先に進めるか、Frozen ViT接続仕様を優先するか。推奨は実画像の短時間preflightとFrozen ViT接続設計の並行。
3. 大規模Queue/K65536・1025stepと10epochを今回の必須Gateとするか、研究主題に合わせ独立Gateで段階化するか。

## research-spec引き継ぎ候補

- 目的：A1でbitwise一致を達成した公式InfoNCE算術を、通常のMoCo比較/評価経路に適用可能にし、後方互換性を確保。
- 比較：Meta固定commit/公式式・自作S1・旧自作S0。既存A1 1/16stepとCIFAR非回帰を必須。
- 範囲：`DistributedMoCo`およびparity runner/configの診断profile／explicit mode指定境界を扱い、必要最小限の改変を精査。通常default変更は本specの候補だが未承認。
- 未決：opt-inの適用範囲（A2/A3/A5、Frozen ViT integrationのどこまで）、schemaとCLI、実画像・Queue長期runの許可と資源見積り。

## 関連

- [A1 16step PASS Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md)
- [Approved A1 16step spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-18-meta-moco-a1-infonce-16step-parity-spec.md)
- [Parent Spec A draft](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md)
- [Five-stage research plan](./2026-10-15-five-stage-lora-moco-research-task-plan.md)
- [Post-MoCo integration roadmap](./2026-10-15-post-moco-gate-c-integration-roadmap.md)
- [A5 ImageNet 2GPU bench監視OFF/Light実験spec（2026-10-20 draft）](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-20-meta-moco-a5-imagenet-gpu-bench-monitor-overhead-experiment-spec.md)
