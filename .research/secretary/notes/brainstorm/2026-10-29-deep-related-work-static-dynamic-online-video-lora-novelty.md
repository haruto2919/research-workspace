---
date: 2026-10-29
project: sequential-video-lora-analysis
source_todo: null
topic: deep-related-work-static-dynamic-online-video-lora-novelty
status: exploratory
tags: [brainstorm, research, literature-survey, static-dynamic-information, online-learning, lora, novelty]
---

# 静的・動的情報と逐次動画LoRA：近接研究の原論文調査と新規性の条件

調査日：2026-10-29 JST。これは先行研究調査と仮説整理であり、spec、実装許可、実験結果ではない。

## 結論

「画像事前学習ViTを動画へ適応」「凍結した画像モデルに少数パラメータを足す」「動画でLoRAを学習」「オンライン動画学習」「静的・動的情報を分ける」は、それぞれ先行例がある。特にPapalampidiら（CVPR 2024）は画像事前学習ViTをjoint時空間attentionへ拡張し、動画と言語の対照学習でLoRAを試している。従って画像ViT＋時空間attention＋LoRA＋動画学習を単独の新規性とは主張できない。

現時点で狙える研究上の差分は、(1) 厳密な時間順・single-passの自己教師あり更新、(2) 画像Baseを固定してフレーム間計算経路のLoRAを学習、(3) LoRAの実効差分ΔWが単一フレーム由来の静的手掛かりでは得られない情報を増やしたか、入力介入・LoRA介入・下流probeで因果的に調べる、という組合せと実証である。ただし今回の探索で「同一研究が存在しない」とは保証できず、比較実験前に新規性を確定しない。

## 研究の現在地と定義

Research WorkspaceのREADME（last_updated 2026-10-22）と10/22 Evidenceでは、現行経路はフレームごとのViT特徴とmasked meanを中心とし、ViT＋MoCoのProjector出力が初回更新で方向集中する。10stepの工学確認はあるが、動画から動的情報を得た証拠はない。M1（複数フレームのtokenを画像事前学習ViTのattentionへ同時投入し、固定の時間位置情報を与え、Q/V LoRAを学ぶ）は10/26〜28の探索案で、実装済みとは扱わない。研究コードのSSH側dirty差分・GPU成果物はGitHubから直接確認していない。

ここで静的情報は単一フレームから抽出できる外観・背景・物体等、動的情報は複数フレームを見て初めて分かる順序・状態変化・運動等とする（Kowalらの定義に対応）。「時間的一貫性」「隣接フレームの対応」「順序の識別」「行為の理解」は異なる能力であり、論文タイトルのdynamicだけで同一視しない。オンラインは「クリップ内のフレームが時刻順」ではなく、学習サンプルの更新順が時刻順で、将来サンプルを使わず、可能ならsingle-passという条件として別に定義する。

## 主要先行研究の比較

| 論文・年 | 原論文で確認したこと | 本研究との重なり | 残る差分・注意 |
|---|---|---|---|
| [Papalampidi et al., CVPR 2024](https://arxiv.org/html/2312.07395) | 画像事前学習ViTをjoint時空間attentionへ転用し、動画・テキストの対照事前学習でLoRAを比較。LoRAはQKV、attention出力、FFNに適用し、動画用3D patch convolutionも学習。LoRAは同論文設定でMLP adapterより劣った。 | 画像ViT、early fusion、LoRA、動画の対照学習まで重なる。**最優先の近接研究**。 | 動画・テキストペアを使うオフライン学習、動画用patch層も更新。主題は長動画encoderの性能・メモリで、ΔWの静的・動的情報への帰属や厳密なsingle-pass逐次学習ではない。単純な構成差だけでは弱い。 |
| [Co-Settle, CVPR 2026](https://arxiv.org/html/2603.26597) | 画像encoderを凍結し、projection layerのみをKinetics-400で5 epoch自己教師あり学習。temporal cycle consistencyとsemantic separabilityを両立させ、Chiral SSV2の時間順識別も評価。 | 画像→動画、凍結Base、軽量な差分、ラベルなし動画学習、時間評価が重なる。 | 学習部位は後段projectionでLoRAではなく、frame間対応・一貫性が中心。LoRA重みΔWの層別帰属と時刻順single-passは扱わない。ただし「時間を評価していない」と言ってはいけない。 |
| [TimeTuning, ICCV 2023](https://arxiv.org/html/2308.11796) | 画像事前学習encoderをラベルなし動画で時間整合の自己教師あり目的によりfine-tuneし、dense表現を改善。 | 画像表現を動画の時間信号で更新する発想。 | encoder本体を更新し、LoRA差分の所在・静動分離・single-passが主題ではない。時間整合は順序識別と同義ではない。 |
| [AIM, ICLR 2023](https://arxiv.org/html/2302.03024)、[ST-Adapter, NeurIPS 2022](https://arxiv.org/html/2206.13559) | 凍結画像ViTへ小型adapterを足し、時間方向を処理して行動認識。AIMは画像attentionの時間方向再利用、ST-Adapterは時空間adapter。 | 画像知識を残して時間能力を追加する設計が重なる。 | いずれも主にラベル付き下流タスクで学習。追加部品はLoRA差分ではなく、厳密なオンライン自己教師あり更新とΔW解析は主題ではない。 |
| [Orlova et al., arXiv 2026](https://arxiv.org/html/2605.19137) | 凍結画像モデルの特徴に再帰時間モジュールを置き、streaming推論条件で複数の教師あり下流タスクを検証。本文は「時間モジュールの動画事前学習はまだ行っていない」と明記。 | 凍結画像Baseと因果的な時間処理。 | 動的能力の所在は新しい再帰モジュール。**自己教師あり動画事前学習の先行達成例として引用しない**。 |
| [Carreira et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Carreira_Learning_from_One_Continuous_Video_Stream_CVPR_2024_paper.html)、[StreamMAE, arXiv 2026](https://arxiv.org/html/2609.40333) | 連続ストリームでの逐次学習を研究。前者は将来フレーム予測等、後者は時刻順・再shuffleなしのMAEで、近接frameのbatch内冗長性を分析。 | 「オンライン／逐次」は既に研究課題であり、非i.i.d.の難しさを支持。 | LoRAによる静動分離は主題ではない。StreamMAEは画像encoderを学び、動きは主にcrop選択に利用する。 |
| [VideoMSN, arXiv 2026](https://arxiv.org/html/2609.40347) | 画像事前学習ViTに動画frameをsuper-imageとして配置し、空間patch・時間frame maskingのSiamese自己教師あり学習。 | 画像ViTをラベルなし動画へ効率的に転用。 | BaseをLoRAだけで固定する設定やsingle-pass、ΔW解析が主題ではない。評価の中心がKinetics/UCF/HMDBの精度である点にも注意。 |
| [Qian et al., arXiv 2022](https://arxiv.org/html/2207.12795) | static frameとframe differenceを使い、静的・動的conceptを自己教師ありで分ける。 | 静動分離そのものも既出。 | 入力とconcept表現での分離であり、凍結画像Baseと動画LoRA ΔWの差分帰属ではない。 |
| [Kowal et al., CVPR 2022](https://arxiv.org/html/2206.02846) | 外観と時間構造を変えるvideo pairで、中間表現の静的・動的biasを定量化。多くの動画モデルで静的biasが強い。 | 評価設計の最重要先行研究。 | 重みΔWが何を加えたかという介入は未対象。Kowalのchannel相関指標をLoRAの重みSVDへ直接読み替えない。 |
| [Set-and-Sequence, arXiv 2025](https://arxiv.org/html/2502.14844) | 動画生成DiTで、順序を崩したframeからidentity LoRA基底を学び、動画系列でmotion residualを追加。 | LoRAで静的・動的成分を分ける着想も既出。 | 生成モデルの単一動画personalizationであり、認識用画像ViTの逐次自己教師あり表現とΔWの検証ではない。 |
| [S³T, arXiv 2026](https://arxiv.org/html/2609.04203) | 凍結VLMにLoRAを付け、密なframe viewをteacherとして自己蒸留。主設定ではLoRAは言語モデル側、別設定では画像encoderにもLoRA。 | 無ラベル動画からLoRAで時間能力を改善する例。 | state trackingのVLMで、時刻順single-passの画像ViT学習、静動差分の解析ではない。 |

補助の近接例：[I2V-Adapter（Pattern Recognition 2026）](https://www.sciencedirect.com/science/article/pii/S0031320326001937) は凍結画像ViT＋後段adapterをラベルなし動画のframe間整合で学び、追跡・動画物体分割に使う。Co-Settleと同じく、画像Base固定＋自己教師あり動画適応を新規とは言えない根拠になる。

## 新規性の見積もり

### 成立しない主張

- 「画像で静的、動画で動的」という分担が初めて。
- 「画像ViTを凍結し、小型部品で動画から時間情報を獲得」が初めて。
- 「画像ViTのjoint時空間attentionにLoRAを付けて動画を対照学習」が初めて（Papalampidiらがある）。
- 「LoRAで外観と動きを分ける」が初めて（生成分野のSet-and-Sequence等がある）。
- 「オンライン学習だから動的情報を学べる」。クリップ内の時間関係と学習サンプルの更新順は別軸であり、オフラインでも時間関係は学べる。

### 条件付きで強い研究主張

「凍結した画像ViTのフレーム間計算経路におけるLoRAを、将来データを使わない時刻順single-passの無ラベル動画から学習し、ΔWが追加する複数フレーム依存情報を、静的な動画ドメイン適応から分離して検証する。さらに同一データ・同一更新予算のshuffled学習と比べ、逐次学習がその情報の獲得・保持に与える影響を明らかにする。」

これは今回の主要原論文群との**調査範囲内の差分**であり、実験で成立するかは未確認。「初めて」とはまだ書かない。学習可能なProjectorがあるMoCoでは、Projector出力の改善をLoRA獲得と混同せず、評価はProjector前のencoder特徴、LoRA on/off、Projector固定または同一条件で行う。

### 採用候補・保留・棄却寄り

- 採用候補：M1のフレーム間Q/V LoRA、固定時間位置情報、画像Base凍結、時刻順single-pass、ΔWの介入とstatic/dynamic対照。方法論としての貢献を狙うなら、学習目的に時間を使わないと解けない信号を含める。
- 保留：LoRAの部分空間から「純粋な動き成分」を抽出できるという主張。SVDの向きだけでは意味は決まらず、同じノルム・rankのランダム除去等との機能比較が必要。
- 棄却寄り：通常の同一clip 2-view MoCoだけで「LoRAは動きを学ぶ」と予告すること。場面・物体等の静的shortcutでlossを下げられる。

## 新規性を実証する最小比較

1. 同じ動画入力経路でBase（ΔW=0）と動画LoRAを比較。Baseも時間位置情報とjoint attentionで時間に反応し得るため、Baseの値を0と仮定しない。
2. 動画LoRAの対照として、同じM1構造・同じ更新予算で、frame順序を無作為化して学んだLoRAと、静的frame中心で学んだLoRAを置く。frame-wise学習LoRAをM1にそのまま移す比較は入力分布差を生むので補助扱いにする。
3. 「clip内順序」と「clip更新順」を別々に操作。後者だけを時刻順single-pass対shuffledに変え、目的関数・見た動画集合・update数・rank・Projector・評価器を揃える。stream設定では未来clipをQueueやpositiveに入れない。
4. 時間の向きだけを変える同一frame集合のペア、外観だけを変えて動きを維持するペア、動き依存の下流probeを組み合わせる。単なる1−cos差やattention量は補助で、正しい方向判断・静的変化への頑健性・LoRA除去での性能低下を主根拠にする。
5. ΔWを層別/Q-V別に除去し、時間依存の性能が落ちるかを確認する。特異値や主角度は機能の証明ではない。学習済みProjectorを含む場合は帰属を明確にする。
6. 1seed pilotと、複数seed・独立動画での確認を分ける。Papalampidi、Co-Settle、Kowal、StreamMAEを目的に応じた主要baseline／評価先行例とする。

## 反例と未解決点

- 時間順で学んだLoRAと順序を崩して学んだLoRAが同等なら、LoRAは動画ドメインの静的差分を主に吸収している可能性がある。
- Baseや固定時間位置情報だけで逆再生を判別できるなら、LoRAの増分を示す必要がある。
- LoRA on/offで時間probeが改善しても、色・背景の変更で同時に崩れるなら「動的専用」とは言えない。
- O1/O2のreverse negativeは逆再生対称動作でfalse negativeとなり、逆再生時の不自然な低レベル手掛かりを学ぶおそれがある。
- 「動的情報がLoRAにのみ存在する」は強すぎる。確かめられるのは「LoRAがBaseに加える増分のうち、複数フレーム依存の機能があるか」である。
- データセットに実際の長い連続ストリーム、動き依存評価、外観介入の実行環境があるかはGitHubから未確認。
- 文献探索は主要原論文・近接論文を中心にしたもので、全査読誌・全preprint・全コードを網羅した新規性証明ではない。

## 以前の探索メモへの補正

2026-10-26メモはPapalampidiらを挙げていなかったため、「画像ViT＋joint attention＋LoRA＋動画対照学習」を新規性候補から外す。Orlovaらは自己教師あり動画事前学習を実施した論文ではなく、本文で未実施と明記する。Co-Settleは時間順の評価も行っており、単に時間を調べない先行研究とは言えない。これらの補正は先行研究の読み直しによるもので、既存の探索メモや採用specを変更しない。

## 関連Research Workspace

- プロジェクトREADME: https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/README.md
- 最新のGPU診断記録: https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md
- 提案手法と解析: https://github.com/haruto2919/research-workspace/blob/main/.research/secretary/notes/brainstorm/2026-10-27-early-fusion-lora-dynamic-information-analysis-plan.md
- 動的情報の確認案: https://github.com/haruto2919/research-workspace/blob/main/.research/secretary/notes/brainstorm/2026-10-28-dynamic-information-verification-proposals.md

次アクション候補：最初にPapalampidiらのLoRA条件を再現可能なbaselineとして確定し、M1の学習前Baseの時間probe・VRAM・特徴分布を測るためのspec readinessを検討する。TODOとspecには自動昇格しない。
