---
date: 2026-11-12
project: sequential-video-lora-analysis
topic: static-dynamic-evaluation-literature-survey
status: survey
tags: [survey, lora-analysis, static-dynamic, evaluation, temporal-probe]
---

# LoRAが動的情報を学んだかを評価する方法：文献調査（2014–2026）

- 解説ページ（Artifact、非公開）：https://claude.ai/artifact/JAmkhCe6u18iHRCGZaqJ62
- 関連する壁打ちメモ：[2026-11-12-lora-dynamic-information-evaluation-13-year-survey.md](../../../../secretary/notes/brainstorm/2026-11-12-lora-dynamic-information-evaluation-13-year-survey.md)

## 調査の範囲

4分野を並行して調べた（約130本）。各分野の詳細メモ（論文ごとの評価手法、この研究への当てはめ案、検証済みの図URL）は次のファイルにある。

| 分野 | 本数 | 詳細メモ |
| --- | --- | --- |
| A. LoRA・fine-tuningの更新と表現の解析 | 30 | [track1](2026-11-12-static-dynamic-survey-track1-lora-analysis.md) |
| B. 動画の静的バイアスと時間理解の評価・ベンチマーク | 34 | [track2](2026-11-12-static-dynamic-survey-track2-static-bias-benchmarks.md) |
| C. 時間・動きを狙う動画自己教師あり学習、画像→動画PEFT、ストリーミングSSL | 41 | [track3](2026-11-12-static-dynamic-survey-track3-temporal-ssl-adaptation.md) |
| D. 生成・系列潜在モデルの静動分離と、動画拡散のmotion LoRA | 27 | [track4](2026-11-12-static-dynamic-survey-track4-generative-disentanglement.md) |

採録先は arXiv comment・CVF・ECVA・OpenReview・PMLR・proceedings・ACL Anthology で確認し、確認できなかったものは各メモで「未確認」と書いた。図は arXiv HTML 版から取得した（2026-11-12時点）。

## 要点

1. **cos≈0.998 は「感度が低い」ことを示すだけ**で、順序情報が「ない」ことの証明にはならない。凍結特徴から順序・向き・速度を線形probeで読み出せるか（能力）を別に測る必要がある。「感度／能力」の区別は本調査での整理。
2. **最も近い先行例は MViT（ICCV 2021）のテスト時シャッフル実験**。時間位置埋め込みを持つ ImageNet-21K 事前学習 ViT-B は、Kinetics-400 でシャッフル前後とも 79.3%（MViT-B は 77.2→70.1）。TIME（ICML 2022）は、ブロックごとの順序probeで深い層ほど順序情報が消えることを示した。
3. **目的関数の構造**：正例が「同じclipの2つのaugmentation」だと、順序を区別する圧力がない。DSM・TE・C-DSVAE・Test of Time などは、時間を崩したclipを負例や別の正例として明示的に入れている。
4. **Kowal et al.（CVPR 2022）の静的ペア／動的ペア**は Stylized ActivityNet を使っており、この研究の評価に最も直接移植しやすい。
5. **Q/V LoRA について**：生成モデルでは、V や FFN に入れた LoRA は見た目を運びやすく、Q/K に入れると漏れにくいという報告がある（Separate Motion from Appearance, ACM MM 2025、Motion Inversion, SIGGRAPH 2025）。この研究の ViT に当てはまるかは未確認の仮説。
6. **LoRAの重み解析（SVD、intruder dimension、ΔM/ΔD、タスクベクトル）**は「どこが・どれだけ・どの向きに変わったか」を示すが、意味の判断には入力介入とprobeが必要。

## 証拠の5段（本調査での整理）

| 段 | 問い | 主な手法 |
| --- | --- | --- |
| 1 感度 | 順序を崩すと特徴が変わるか | cos、4×4フレーム間類似度行列、ΔW側だけの出力の比較 |
| 2 能力 | 凍結特徴から順序・向き・速度を読めるか | 層ごとの順列probe、順/逆probe、stride probe、chiral pair、SSv2 vs K400 |
| 3 選択性 | 静的な手がかりを変えても保たれるか | 静的／動的ペアの相互情報量、静止clip、背景差し替え、差分特徴＋Leakage Gap |
| 4 因果 | LoRAへの介入で動的能力が選択的に落ちるか | alpha sweep、Q/V・層knockout、特異成分除去＋ランダム対照、活性patching |
| 5 学習条件の対照 | 順序を壊して学習したLoRAと差が出るか | シャッフルclip LoRA、静止clip LoRA、ストリーム順序の比較 |

具体的な実験の並べ方と結果の読み方は、解説ページの「この研究への当てはめ」「結果の読み方」の節にまとめた。いずれも未実施の提案である。

## 未確認・注意

- 採録先が未確認：Vinoground、Side4Video、Weight Spectra（2505.23099）、Soligo et al.（2506.11618）、Goel et al.（2510.05092）、TCLR の CVIU 巻号年、Alain & Bengio の本会議／WS区分。
- 著者側ページ・公式リポジトリのみで確認：Motion Inversion（SIGGRAPH 2025）、Separate Motion from Appearance（ACM MM 2025）。
- ワークショップ論文：TemporalBench、VBenchComp、Punzo et al.。
- 「この研究への当てはめ」はすべて調査者の提案であり、各論文の主張ではない。
