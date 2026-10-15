---
date: 2026-10-15
project: sequential-video-lora-analysis
source_todo: null
topic: post-moco-gate-c-integration-roadmap
status: exploratory
tags: [brainstorm, research, moco, lora, vit, integration, online-learning, staged-verification]
---

# MoCo Gate C後の段階的統合に関する研究相談（探索）

## 位置づけ

2026-10-15、ユーザーはLoRA単体20step／Standalone MoCo GPU64/500stepの結果を見た後、今後の方針の確認を希望した。今回の相談は方針の再整理であり、ユーザーによる新方針の採用・正式spec化・コード実装・実験runの許可を意味しない。

## 参照したAuthority・Evidence

1. [2026-10-08 MTG（明示された方針）](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-10-08-mtg.md)：従来ActivityNet統合系のMoCo loss振動・Query LoRA勾配急増を原因切り分けするまでLinear Probe結果を研究上の結論としない。LoRA、MoCo、Sequential Loaderの検証を分ける。
2. [2026-10-15 GPU64/500step Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-15-standalone-moco-gpu-64step-and-gate-c-500step-evidence.md)：CIFAR10/ResNet18、LoRAなし、Loaderなし。64step/500stepとも工学check14件PASS、Gate C工学PASS。500step最終Loss 6.64295、Positive .146564、mean negative .010121、hardest negative .483434。特徴の識別性・汎化・collapseの有無は未検証。64/500のschedulerはそれぞれ固定LR/cosineであり直接Loss比較は避ける。
3. [2026-10-15段階統合brainstorm](./2026-10-15-moco-loader-lora-staged-causal-debugging.md)：変更因子を一つずつ導入するStage0〜5案。これはexploratoryでありapproved specではない。
4. [プロジェクトREADME](../../../lab/projects/sequential-video-lora-analysis/README.md)：研究の最終目的は動画逐次学習でLoRAが静的／動的情報をどう保持するかの評価。LoRA／MoCo単体の工学検証と動画情報の獲得は別。
5. [2026-10-14次段検証brainstorm](./2026-10-14-post-lora-standalone-next-moco-validation.md)：CIFAR分類のLoRA学習はLoRA＋分類Headの検証でありMoCo/動画学習の成功ではない。MaskedMeanはchunk内順序不変。

情報境界：研究サーバのローカルworktree・GPU runそのものはGitHubから直接確認できない。500stepのEvidenceはユーザー提出の自動サマリーから記録されたもの。2026-10-15時点で名指しのFrozen ViT＋MoCo Stage2 draft specのGitHub main上のパスは取得できなかったため、**現行approved specが存在するとは断定しない**。

## 解くべき中心問題

- 学習が工学的に成立することと、情報的に有用な表現を獲得することを明確に分ける。
- 過去のMoCo loss振動・LoRA勾配増大を、Encoder、MoCo、動画入力、Sequential Loader、自作LoRA、strict-onlineのいずれの因子に帰属できるか。
- 16-frame CLS masked meanを使う限り、chunk内順序への感度を主張できない。オンラインで時間情報の獲得を目指す長期研究課題と、当面の工学安定化を分ける。

## 今回の推奨候補（未承認）

| 段階 | 固定する要素／導入する差分 | 証明したいこと |
| --- | --- | --- |
| 0 完了 | ViT自作Q/V LoRA＋CIFAR supervised、独立ResNet18＋CIFAR MoCo | 各基本部品の工程上の動作。特徴品質ではない |
| 1a 次の最有力 | Frozen ImageNet-pretrained ViT（LoRA OFF）＋MoCo、通常画像Loader。Query ProjectorのみSGD等で学習、Key Projector EMA、Base固定 | ViTへの置換後もfeatures 768→projector128→InfoNCE/Queue/EMAが成立 |
| 1b 並行評価候補 | Standalone ResNet18-MoCo checkpointを使い、固定検証集合のPositive retrieval順位／margin、variance・effective rank、学習前後・ラベル付きheld-out linear evaluation | MoCo単体で識別的な表現が獲得されたか。追加評価はengineering Gateの合否とは区別 |
| 2a | 上記Frozen ViT＋Projectorの入力を事前固定16-frame ActivityNet chunkに変更し、masked mean導入（Loaderなし） | Video chunk、padding mask、集約処理の数値・gradient接続 |
| 2b | 2aと同じ実データraw frame/順序/transform/モデル・Queue・seedのままSequential Loader経由へ変更 | Loaderの有無で特徴、loss、gradient、update、EMA、Queueが許容差内一致 |
| 3 | 2bへ自作Q/V LoRAを追加、Base ViT frozen、Query LoRA A/B＋Projectorを学習、Key側EMA | MoCoのInfoNCEからA/Bにgradientと更新が伝わり、Base不変。Bゼロ初期ゆえA初step0を許容 |
| 4 | Strict-single逐次動画stream、negative policyとaugmentationを明示し順序・記憶制約の評価 | Online条件でのLoss、Grad/Update、Queue、EMA、表示統計、安定性 |
| 5 | Base ViT vs MoCo-LoRA評価、時間反転・順序入替等の対照 | 動作分類などの表現品質と時間順序への感度を分けて判定 |

段階1aは最終構成に向けた**工程対照**であり、ResNet18の標準MoCoと同条件のモデル学習ではない。ResNet18 trainable backboneとFrozen ViT＋Projector onlyをLoss絶対値で直接比較しない。

## 選択肢とトレードオフ

- A: MoCo表現品質評価を全段階の前提Gateにする。強みは低品質のMoCoを持ち込むリスクの低減、弱みはViT統合工程の停止・参照学習期間の追加。
- B: **推奨** Frozen ViT＋MoCoの短時間engineering統合と、ResNet18 MoCo表現品質の固定評価を独立・並行実施。機構検証を進めつつ、表現品質の懸念を残件として明示できる。
- C: ActivityNet＋MoCo＋自作LoRA full runに直行。複数変更の交絡、GPUコスト、従来異常の原因不明が継続するため非推奨。

## 各Gateで残す主な統計と失敗条件

- 再現性：repository/ref/commit/dirty、config・input index・transform seed、artifact/report/checkpoint。
- MoCo工程：Query/Keyの更新境界、Key gradient0、EMA式差、FIFOとno-future-key、Queue contents / negative policy、nonfinite有無。
- 学習接続：LoRA A/B layer-wise grad・update norm、Base完全一致、Projectorのgrad/update、pre-L2 feature norm、LR。
- 特徴品質：固定検証集合でpositive retrieval rank、mean・hardest negative margin、feature covariance/effective rank、heldout linear evaluation（結果は未取得）。
- 動画：chunkのraw frameとtimestamp、mask、valid count、順序、same-video negatives占有率、Loader parity。
- 不合格時の対照：Loaderなし動画chunkとの差分、LoRA ON/OFFで同じ入力・初期状態を比較。最初のB=0 LoRA出力一致と、その後のA/B学習を確認。

## 未決・別specで確定すべき論点

1. 次のFrozen ViT＋MoCoについて初期入力・2-view transform・学習率・optimizer/step数・チェック閾値・artifactsを確定する。参照 MoCo のQueue/EMA/Tを機構検証では固定し、既存ActivityNetのQueue4096/T0.07とは同一視しない。
2. ResNet18 MoCoの表現品質追加評価の範囲（固定retrievalだけか、Linear Probeまで行うか）。短時間engineer Gateと表現学習Gateを独立に設計する。
3. 動画Chunkなし／あり、Loaderなし／あり、LoRAなし／ありの各対照で、比較可能な同一入力・学習状態をどこまで固定できるか。
4. 長期的に時間順序の獲得を検証するタスク：masked meanで順序不変である限界、順序に敏感なencoder/aggregatorの導入可能性。研究的な変更は工学安定化の段階に混ぜない。
5. 本相談の方針が採用かどうか、次の正式spec化範囲は**ユーザー未確定**。

## 次アクション候補（TODO未追加）

最初に、Frozen ViT＋MoCo（LoRA無効、通常のCIFAR-10 Loader）の限定specを作るかをユーザーと確認する。並行してResNet18単体MoCoの固定評価設計を別の成果物で検討する。単体工程の検証後に、動画Chunk→Sequential Loader parity→自作LoRA→strict-onlineの順に移る。現時点で新しいresearch codeの編集・run・spec昇格は実施しない。
