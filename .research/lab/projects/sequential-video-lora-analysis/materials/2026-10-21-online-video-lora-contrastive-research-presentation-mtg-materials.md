---
date: 2026-10-21
project: sequential-video-lora-analysis
type: meeting-materials
topic: 長時間動画オンライン学習・静的バイアス・画像ViT+LoRA・MoCoの研究発表ストーリー
status: draft
target_meeting: null
notion_url: null
notion_export: null
tags: [meeting-materials, presentation, online-learning, static-bias, video-lora, moco]
---

# 2026-10-21 研究発表スライドのストーリーと根拠

## 0. 今回レビュー・相談したいこと（Ask）

1. 発表の因果的な順序「長時間動画の逐次・オンライン学習 → 動画認識の静的バイアス → 画像ViTを凍結しLoRA追加学習分を解析 → MoCoによる自己教師あり学習」は、現在の研究の中心的な問いに整合するか。
2. 現行のframe-wise CLS + masked meanが順序不変である制約を、発表のどの位置で強調するか。
3. 動的情報の実証に必要な対照実験（順逆再生、static-repeat、背景変換、LoRA SVDなど）を、実行計画に昇格する前にどう絞るか。

## 1. 現時点の結論／研究目的

画像で事前学習されたViTのBase重みを凍結し、時間順に入力される動画に対してAttention Q/VへのLoRAを自己教師ありのMoCoで更新する。**LoRAを更新できること**と、**LoRAに動画の時間的・動的な情報が意味的に分離されたこと**は異なる。研究の核心は追加更新を観測・解析し、Baseモデルとは異なる動画由来情報を示せるか検証する点に置く。

## 2. 根拠となる関連研究と研究位置づけ

| 関連研究 | 原論文で確認できる貢献 | 本研究との区別 |
| --- | --- | --- |
| MeMViT (Wu et al., CVPR 2022) | キャッシュした過去特徴で長期文脈を利用して動画認識 | 主題は長期動画認識。画像事前学習済みViTの固定とLoRA追加更新の科学解析は別の問い |
| TSM (Lin et al., ICCV 2019) | 低コストな時間方向特徴交換、online recognitionも扱う | 効率的な時間表現と、逐次データでのLoRA重み更新を区別 |
| StillMix (Li et al., ICCV 2023) | 背景だけでなく前景の静的ショートカットを検証；SCUBA/SCUFOを提供 | 静的バイアスの存在を示す。すべての動画モデルが静的情報だけを使用するとは主張しない |
| VideoMAE (Tong et al., NeurIPS 2022) | tube maskingを用いた自己教師あり動画事前学習 | 動画専用の時間情報学習が有力な別経路。比較・発展候補 |
| VideoMoCo (Pan et al., CVPR 2021) | temporal frame dropoutとQueueのtemporal decayを導入 | MoCoへの動画拡張が既に存在する。まず公式MoCoを再現し、構成要素を切り分ける理由を説明 |
| MoCo (He et al., CVPR 2020) | momentum encoder + negative queueによる自己教師あり対照学習 | 本研究の学習基盤。これだけで動的情報の獲得を保証しない |
| LoRA (Hu et al., ICLR 2022) | 既存重みを固定し、低ランク追加パラメータを学習 | パラメータ更新の区分と、静的／動的情報の意味的分離は異なる |

論文URL:
- MeMViT: https://openaccess.thecvf.com/content/CVPR2022/html/Wu_MeMViT_Memory-Augmented_Multiscale_Vision_Transformer_for_Efficient_Long-Term_Video_Recognition_CVPR_2022_paper.html
- TSM: https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html
- StillMix: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Mitigating_and_Evaluating_Static_Bias_of_Action_Representations_in_the_ICCV_2023_paper.html
- VideoMAE: https://arxiv.org/abs/2203.12602
- VideoMoCo: https://openaccess.thecvf.com/content/CVPR2021/html/Pan_VideoMoCo_Contrastive_Video_Representation_Learning_With_Temporally_Adversarial_Examples_CVPR_2021_paper.html
- MoCo: https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html
- LoRA: https://mlanthology.org/iclr/2022/hu2022iclr-lora/

## 3. 現在の動画MoCo構成（設計文書／コードから）

```text
ActivityNet v1.3（training、16 frame/chunk）
  → strict_single逐次入力
  → 画像事前学習済みFrozen ViT + Attention Q/V LoRA（rank 8, alpha 8）
  → frame-wise CLS [16,768] → valid frame Masked Mean [768]
  → Query/Key Projector（768→768→128）
  → 同clipの色チャネル入替＋左右反転をPositive
  → 古いFIFO Queue（K4096、同じ動画の過去chunkも含む）をNegative
  → InfoNCE → Query LoRA/ProjectorをAdamW更新
  → Key branchはEMA追従（momentum 0.999）
```

MaskedMeanClipAggregatorはvalid行の`mean(dim=0)`であり、frameの並べ替えに対して値が変わらない。順次学習によりパラメータが入力順に依存する可能性と、各clipの特徴が時間順序を直接符号化することを混同しない。

## 4. Evidence: 直接確認できた範囲

- **LoRA単体：** 2026-10-15 MTG記録ではCIFAR-10分類20stepで、LoRA A/Bの勾配・更新、ViT Base凍結、分類loss低下を確認。これは画像分類による工学的確認。
- **MoCo単体：** 2026-10-20 E2 Evidenceでは、Meta公式／自作のImageNet1epoch各5004stepで、記録指標・採取input hash・epoch canonical model stateの比較PASS。Checkpoint/sidecarファイル存在、Comet local-report checkも確認。表現品質は未評価。
- **5epochへの方針変更：** 10epoch比較は中止。公式5epoch→自作5epochの逐次ランチャーがdev@7fc1f56へpush済み。最新Research Workspace時点で長時間GPU実行は未実施。
- **動画統合：** ActivityNetのstage smokeや学習機構の工学チェックはあるが、MoCo lossの振動・勾配等の問題があり、動作分類の有効性や動的情報獲得は現時点で結論にできない。

## 5. Interpretation／現在言えること・言えないこと

言えること：
- 最新記録の範囲でBase凍結・LoRA更新・MoCo単体の数値再現性にはEvidenceがある。
- 静的バイアスの問題は既存研究で実証されている。
- 現行frame-wise CLS＋masked meanはフレーム順序不変という構造的制約がある。

まだ言えないこと：
- LoRAが時間順序や運動情報を獲得済み。
- LoRAのみが動的情報、Baseのみが静的情報を保持する。
- ImageNetでMoCoの正規レシピと同等のrepresentation qualityが得られた。
- ActivityNetの最終Linear Probe等で有意な改善が確認された。

## 6. スライド構成（16:9／加藤氏の卒業発表フォーマット参照）

1. タイトル
2. 研究ストーリー
3. 長時間動画のオンライン学習
4. 長時間動画関連研究（MeMViT / TSM）
5. 静的情報バイアス
6. 静的バイアス関連研究（StillMix）
7. 動画自己教師あり学習（VideoMAE / VideoMoCo）
8. 研究課題
9. 研究目的
10. Frozen ViTとLoRAの役割
11. ActivityNet Sequential Loader
12. 現行モデル全体
13. MoCo機構
14. MoCoとVideoMoCoの比較理由
15. frame-wise meanの順序不変性
16. 確認済みEvidenceと未確認
17. 段階的検証計画
18. 動的情報獲得の解析案（未検証）
19. まとめ
20. 参考文献
21–22. 質疑応答用補足

PowerPoint本体は会話の成果物として作成した編集可能な`.pptx`であり、GitHubの研究repoや実験結果を変更していない。

## 7. 次に進む候補（未採用、TODO追加なし）

- 研究発表用の重要項目：見た目の静的バイアスを問題にする先行研究と、時間的順序を学べない現行アーキテクチャの両方を説明。
- LoRA分析はSVD/effective rankだけでは不十分。Base対LoRA、順逆、static-repeat、背景・服装変換等の対照を組み合わせる。
- 必要なら時間依存性を持つ新しいアーキテクチャ（temporal aggregation / temporal token interaction）を比較候補として別途壁打ち・spec Gateへ進める。

## 8. Research Workspace参照先

- [Project README](../README.md) — 2026-10-20最新status/研究目的。
- [2026-10-15 MTG](../meetings/2026-10-15-mtg.md) — 中間発表と段階的統合の決定。
- [2026-10-08 MTG](../meetings/2026-10-08-mtg.md) — 動画MoCoの研究主張・順序不変の課題。
- [Streaming MoCo/Linear Probe spec](../specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md)。
- [Meta E2 Evidence](../experiments/2026-10-20-meta-moco-v2-e2-imagenet-one-epoch-pair-partial-evidence.md)。
- [5epoch sequential comparison spec](../specs/2026-10-20-meta-moco-v2-imagenet-5epoch-sequential-comet-comparison-spec.md)。

