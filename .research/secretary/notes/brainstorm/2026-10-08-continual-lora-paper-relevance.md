---
date: 2026-10-08
project: sequential-video-lora-analysis
source_todo: null
topic: continual-LoRA論文と現在研究の関連整理
status: exploratory
tags: [brainstorm, research, lora, continual-learning, online-learning, streaming-video, svd, orthogonality]
---

# Continual-LoRA論文と現在研究の関連整理

## 出発点

次の4論文について、現在のResearch Workspace上の研究と何が本当に関連し、何が異なるかを整理した。

- Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation (arXiv:2602.00722)
- Janus-LoRA: A Balanced Low-Rank Adaptation for Continual Learning (arXiv:2605.28495)
- Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation (arXiv:2411.05663)
- InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning (arXiv:2404.00228)

## 読み込んだ現在文脈

- `.research/lab/projects/sequential-video-lora-analysis/README.md`
- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-24-mtg.md`
- `.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-28-shared-streaming-moco-protocol-spec.md`
- `.research/lab/projects/sequential-video-lora-analysis/specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md`
- `.research/lab/projects/sequential-video-lora-analysis/experiments/2026-10-06-full-pipeline-hydra-config-verification.md`
- `.research/secretary/notes/brainstorm/2026-09-25-video-lora-learning-and-evaluation-framework.md`
- `.research/secretary/notes/brainstorm/2026-09-28-moco-xtrans-online-constraints.md`

2026-10-08の日次TODOは未作成。

## 確認済みの現在研究

現在の中心目的は、画像事前学習済みViTへ動画を逐次入力し、Base ViTをfreezeしたままQ/V LoRAを自己教師ありMoCoで更新して、画像のみでは得にくい動画由来の動的情報をLoRAへ獲得できるかを調べること。その後、LoRAの内部構造と下流有用性を解析する。

承認済みfull-dataset specでは、ActivityNet training 10,024動画をdeterministicな順序で1回だけ処理し、動画境界でもQuery/Key/optimizer/EMA/Queue stateを保持するsingle-pass Streaming MoCoを採用している。学習後はBase ViTと最終Query LoRAをsegment-level Linear Probeで比較する。

現在のspecは、複数epoch、shuffle、raw-data replay、Orthogonal Gradients、LoRA SVD、motion correlation、temporal-order claimを明示的にout of scopeとしている。一方、brainstormではchronological vs shuffled、forgetting/order dependence、LoRA singular spectrum/effective rank、将来の直交化を研究候補として保持している。

## 重要な区別

現在研究は標準的なclass/task-incremental continual learningそのものではない。

- supervisedな新クラス・新taskを順番に学ぶことが主目的ではない。
- 過去task精度を維持することがprimary objectiveではない。
- self-supervised MoCoで動画表現を逐次適応し、そのLoRAに何が追加されたかを調べることが主目的。
- task boundaryを前提としない。
- Queueにはpast representationを保持するが、approved full-dataset scopeではraw-data replayをしない。

したがって4論文の関連は「continual learningだから全部直接関連」ではなく、どの問題設定・解析・最適化原理が共通するかで分ける必要がある。

## 各論文との関連

### 1. Online-LoRA

最も近い点:
- pretrained ViTをLoRAでonlineに適応する。
- non-stationary / task-free streamを扱う。
- sequential更新でforgettingを問題にする。

現在研究との違い:
- Online-LoRAの中心はsupervised online continual classificationとcatastrophic forgetting抑制。
- 現在研究はself-supervised MoCoで動画LoRAを作り、下流Linear Probeでrepresentation utilityを見る。
- 現在研究のprimary questionは「昔のtaskを忘れないか」ではなく「動画から有用/動的な情報がLoRAへ追加されるか」。
- Online-LoRAはhard bufferを利用する構成も持つが、現在のapproved full-dataset scopeはraw-data replayを明示的に使わない。

位置付け:
関連研究としては「task-free/onlineなViT+LoRA適応」の先行例として重要。ただし提案手法をそのままbaselineとして採用するより、研究設定の近い先行研究として引用する性格が強い。

### 2. Spectral Imbalance / EBLoRA

最も近い点:
- LoRAの `Delta W` をSVDしてsingular spectrumを見る。
- singular valueの偏り/effective rankをLoRAの内部構造として解釈する。

これはResearch Workspaceで既に予定している、
- singular values
- effective/stable rank
- layer別spectrum
- performanceとSVDの対応
と非常に直接的に接続する。

現在研究との違い:
- EBLoRAはtask間forgettingの原因としてspectral imbalanceを扱い、balanced updateを学習方法として提案する。
- 現在研究ではまだ「spectral imbalanceがforgettingを起こす」が研究仮説ではなく、まず動画LoRAが何を学んだかをSVDで観察する段階。
- task-specific updatesや過去task performanceを直接評価する現在設計ではない。

位置付け:
現在の学習法そのものより、LoRA解析フェーズに直接使える重要な関連研究。まず現行MoCo-LoRAのspectrumを観察し、その後にspectral imbalanceとorder dependence/forgettingが結び付くEvidenceが出れば、EBLoRA型制約を検討するのが自然。

### 3. InfLoRA

最も近い点:
- pretrained modelをfreezeし、低ランクsubspace内だけで新しい学習を行う。
- 過去知識へ干渉しない方向へ更新を制限する。
- stability-plasticityをsubspace/orthogonalityで扱う。

Research Workspaceでは、
- sequential非IID入力でのforgetting/order dependence
- Orthogonal Gradientsをoptimizer側比較候補
- 将来のLoRA直交化
を既に候補としているため、この部分と関連する。

現在研究との違い:
- InfLoRAの中心は複数task/classを順次学ぶcontinual-learning protocol。
- 現在研究はActivityNetを一つのself-supervised streamとしてsingle passする。
- 現行MoCoでは明示的な「過去task subspace」を作っていない。
- 時間/動作情報獲得そのものをInfLoRAは目的にしていない。

位置付け:
現時点のbaselineへ入れる方法ではなく、もしchronological learningで過去表現の劣化や強いorder dependenceが確認されたときの干渉抑制候補。さらに将来、静的/動的LoRA subspaceを分離する研究へ進む場合に概念的に重要。

### 4. Janus-LoRA

最も近い点:
- LoRAのA/B因子を独立更新した結果、最終的なcomposite update `Delta W` が意図した直交方向からずれる問題を扱う。
- historical subspaceに対するorthogonalityをLoRAでどう正しく実現するかを扱う。

これは、現在Workspaceで将来候補になっているOrthogonal Gradients/LoRA直交化を実装する場合に重要。

現在研究との違い:
- 現行approved pipelineではOrthogonal Gradientsはout of scope。
- 現在のMoCo学習は過去task subspaceを明示的に保護していない。
- Janus-LoRAのfeature-level marginはclass-incremental learningのplasticity維持を目的としており、self-supervised MoCoへそのまま対応しない。

位置付け:
現在の主baselineとの直接関連は弱い。ただし将来「LoRA更新を過去gradient/feature subspaceと直交させる」方法を採るなら、InfLoRAより一歩具体的な実装上の注意を与える重要論文。

## 現時点の関連度

研究設定への近さ:
1. Online-LoRA
2. InfLoRA / Janus-LoRA
3. EBLoRA

現在の予定している解析への近さ:
1. EBLoRA
2. InfLoRA / Janus-LoRA
3. Online-LoRA

現在のapproved full-dataset MoCo pipelineへ直接手法として入れる必要性:
- 4本とも現時点では低い。
- 現在はまずBase ViT vs MoCo-LoRAのLinear Probeを成立させることが優先。
- その後、chronological/shuffled、LoRA SVD、forgetting/order dependenceのEvidenceを取得してから、orthogonalityやbalanced spectrumを学習法として導入する方が研究主張を分離しやすい。

## 現時点の有力な使い方

- Online-LoRA: Related Workで「task-free/online ViT+LoRA適応」の位置付けに使う。
- EBLoRA: LoRA内部解析のSVD/effective-rank設計と、spectral imbalance仮説の参考に使う。
- InfLoRA: sequential update interferenceをsubspace制約で抑える将来比較の参考に使う。
- Janus-LoRA: Orthogonal LoRAを実装するとき、A/B個別更新とcomposite updateのずれを避ける設計根拠に使う。

## 反例・注意

「動画を逐次入力している」だけで標準的continual learningと同じ研究問題になるわけではない。
現在研究でcatastrophic forgettingを主要主張にするには、過去時点/過去動画群に対する表現性能低下を明示的に測る評価系が必要。現行approved Linear Probeだけでは、この4論文と同じforgetting問題を評価したことにはならない。

## 未解決

- 現行full run後に、checkpointごとの固定probe/representation driftを測ってforgettingを研究対象に昇格するか。
- chronological vs shuffledをいつ実施するか。
- SVD loggingをfull runへ追加するか、まずfinal/checkpoint解析だけにするか。
- Orthogonal Gradients / InfLoRA / Janus型制約を比較するのは、forgetting/order-dependenceのEvidence取得後にするか。
