---
date: 2026-11-12
project: sequential-video-lora-analysis
source_todo: "静的・動的情報の分離とLoRA解析方法を調査する"
topic: lora-dynamic-information-evaluation-13-year-survey
status: exploratory
tags: [brainstorm, research, literature-survey, lora, video, static-dynamic, evaluation]
---

# LoRAが動的情報を学習したかを検証する方法：2014–2026年の主要学会調査

調査日：2026-11-12 JST。対象期間は2014年から2026年までの13暦年。CVPR、ICCV、ECCV、NeurIPS、ICML、ICLR、AAAI、WACV、TMLRの原論文を中心に、LoRA解析、表現解析、動画の静的・動的情報分離、オンライン／継続学習評価を横断した。これは探索メモであり、spec、実装許可、実験結果ではない。網羅的なシステマティックレビューではなく、現在の研究判断に効く代表研究の構造化調査である。

## 結論

LoRAが動的情報を学習したという主張は、LoRA重みのSVD、順序を変えたときのcos類似度、ActivityNet分類精度のどれか1つでは成立しない。最低でも次の三角測量が必要である。

1. **機能**：同じフレーム集合で順序・速度・方向だけを変えた課題において、BaseよりLoRAありの方が正しく識別できる。
2. **分離**：動的課題の改善が、背景・物体・sceneなど静的手掛かりの改善だけでは説明できない。appearanceを変えてmotionを保つ介入と、motionを変えてappearanceを保つ介入の両方を使う。
3. **帰属**：LoRAを無効化、縮小、層別／Q-V別に除去、特異成分を除去すると動的能力が選択的に落ちる。matched random ablationでは同じ落ち方をしない。

したがって、最も強い設計は **入力介入 × LoRA介入 × 下流probe** である。SVD、CKA、主角度、勾配、attentionは「どこに・どのように変化が入ったか」を説明する補助証拠とし、動的情報の機能的証明には使わない。

現在のG2結果は否定的だが重要である。ActivityNet validationの2,000 clip、4フレーム・0.25秒間隔で、G2-LoRAのqはshuffleとのcosが0.9982、reverseが0.9971、staticが0.9331だった。並び順を区別する情報が身についた観測はなく、長い逐次学習後のモデルはR1のモデルより順序変化に鈍かった。ただしこれは1 seed・1データセット・短いclip・cos感度の記述的結果であり、「動的情報を全く持たない」とはまだ断定できない。

## 1. 最初に定義を固定する

本研究での作業定義は次とする。

- **静的情報**：単一フレームから推定できる物体、scene、背景、identity、色、形、姿勢の一部。
- **動的情報**：複数フレーム間の関係を必要とする方向、速度、順序、状態遷移、周期、接近／離反、作用の前後関係。
- **時間感度**：入力順を変えると特徴が変わること。正しい時間理解とは限らない。
- **時間能力**：方向・速度・順序・状態遷移などを正しく予測できること。
- **分離**：静的因子を変えても動的予測が保たれ、動的因子を変えると予測が対応して変わること。
- **オンライン効果**：clip内部の時間構造ではなく、学習サンプルが未来を使わず時刻順・single-passで到着することによる差。

この区別をしないと、「動画で学習した」「reverseで特徴が変わった」「時刻順に入力した」をすべて動的情報獲得と誤解する。

## 2. 13年間の研究の流れ

| 年 | 代表研究・学会 | 評価上の意味 |
|---|---|---|
| 2014 | [Two-Stream ConvNets, NeurIPS](https://papers.nips.cc/paper_files/paper/2014/hash/ca007296a63f7d1721a2399d56363022-Abstract.html) | RGBのspatial streamとoptical-flowのtemporal streamを分ける古典。静的／動的を別入力で比較する発想の起点。 |
| 2015 | [Modeling Video Evolution, CVPR](https://openaccess.thecvf.com/content_cvpr_2015/html/Fernando_Modeling_Video_Evolution_2015_CVPR_paper.html) | frameの時間順をrank poolingで表現し、順序を持つ動画表現を評価。 |
| 2016 | [Dynamic Image Networks, CVPR](https://openaccess.thecvf.com/content_cvpr_2016/html/Bilen_Dynamic_Image_Networks_CVPR_2016_paper.html) | 動画の時間発展を1枚のdynamic imageへ集約。単一frame baselineとの差が重要。 |
| 2017 | [Something-Something, ICCV](https://openaccess.thecvf.com/content_iccv_2017/html/Goyal_The_Something_Something_ICCV_2017_paper.html)、[DRNET, NeurIPS](https://papers.nips.cc/paper/2017/hash/2d2ca7eedf739ef4c3800713ec482e1a-Abstract.html) | 時間関係が必要なデータと、stationary／time-varying latentの分離。identity・poseのswapや生成が分離検証になる。 |
| 2018 | [Arrow of Time, CVPR](https://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Learning_and_Using_CVPR_2018_paper.html)、[MoCoGAN, CVPR](https://openaccess.thecvf.com/content_cvpr_2018/html/Tulyakov_MoCoGAN_Decomposing_Motion_CVPR_2018_paper.html)、[RESOUND/Diving48, ECCV](https://openaccess.thecvf.com/content_ECCV_2018/html/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.html) | 正逆判定、content固定motion交換、静的biasを抑えたデータセットという三つの評価軸。 |
| 2019 | [Clip Order Prediction, CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Self-Supervised_Spatiotemporal_Learning_via_Video_Clip_Order_Prediction_CVPR_2019_paper.html)、[SlowFast, ICCV](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)、[CKA, ICML](https://proceedings.mlr.press/v97/kornblith19a.html) | 順序予測を自己教師信号にし、slow spatial／fast motion経路を比較。表現差はCKAで層別比較可能。 |
| 2020 | [SpeedNet, CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Benaim_SpeedNet_Learning_the_Speediness_in_Videos_CVPR_2020_paper.html)、[FineGym, CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Shao_FineGym_A_Hierarchical_Video_Dataset_for_Fine-Grained_Action_Understanding_CVPR_2020_paper.html) | 再生速度判定と細粒度の時間構造を、動的能力の機能テストとして使える。 |
| 2021 | [Only Time Can Tell, WACV](https://openaccess.thecvf.com/content/WACV2021/html/Sevilla-Lara_Only_Time_Can_Tell_Discovering_Temporal_Data_for_Temporal_Modeling_WACV_2021_paper.html)、[BE, CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Removing_the_Background_by_Adding_the_Background_Towards_Background_Robust_CVPR_2021_paper.html)、[Context-Motion Decoupling, CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Self-Supervised_Video_Representation_Learning_by_Context_and_Motion_Decoupling_CVPR_2021_paper.html)、[CORP, ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_Contrast_and_Order_Representations_for_Video_Self-Supervised_Learning_ICCV_2021_paper.html) | shuffleで人間認識が落ちるtemporal class、背景置換、motion vector予測、contrast＋orderの併用。通常の同一clip contrastiveだけでは時間関係を使わないという問題が明確化。 |
| 2022 | [LoRA, ICLR](https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/)、[Kowal et al., CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Kowal_A_Deeper_Dive_Into_What_Deep_Spatiotemporal_Networks_Encode_Quantifying_CVPR_2022_paper.html)、[FAME, CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Motion-Aware_Contrastive_Video_Representation_Learning_via_Foreground-Background_Merging_CVPR_2022_paper.html)、[ST-Adapter, NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a92e9165b22d4456fc6d87236e04c266-Abstract.html) | ΔWのrank・subspace解析、latent channelのstatic/dynamic bias定量化、foreground-motionとbackgroundの交換、画像モデルを凍結した動画適応。 |
| 2023 | [AdaLoRA, ICLR](https://arxiv.org/abs/2303.10512)、[AIM, ICLR](https://openreview.net/forum?id=CIoSZ_HKHS7)、[TimeTuning, ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Salehi_Time_Does_Tell_Self-Supervised_Time-Tuning_of_Dense_Image_Representations_ICCV_2023_paper.html)、[Task Arithmetic, ICLR](https://mlanthology.org/iclr/2023/ilharco2023iclr-editing/) | LoRAの特異値・重要度、画像モデル＋軽量部品で時間適応、LoRA／差分重みの加減算・scale介入の根拠。 |
| 2024 | [DoRA, ICML](https://proceedings.mlr.press/v235/liu24bn.html)、[Video-First Encoder, CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Papalampidi_A_Simple_Recipe_for_Contrastively_Pre-training_Video-First_Encoders_Beyond_16_CVPR_2024_paper.html)、[One Continuous Video Stream, CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Carreira_Learning_from_One_Continuous_Video_Stream_CVPR_2024_paper.html)、[LoRA Learns Less and Forgets Less, TMLR](https://mlanthology.org/tmlr/2024/biderman2024tmlr-lora/) | LoRAとFTのweight magnitude/direction差、動画ViTでのLoRA近接例、single-streamの適応と汎化、低rank制約と忘却の評価。 |
| 2025 | [LoRA vs Full FT, NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ff541950d1e885af90f523571564a401-Abstract-Conference.html)、[KnOTS, ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0d4f8a5109c5083b5307fcd0bddae7a7-Abstract-Conference.html) | 特異方向への介入で機能・忘却との因果を検証。複数LoRAのSVD整列からsubspace比較の重要性を確認。 |
| 2026 | [From Static to Dynamic, CVPR](https://openaccess.thecvf.com/CVPR2026?day=2026-06-07)、[OPLoRA, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40703)、[Towards Understanding LoRA Dynamics, ICML](https://proceedings.mlr.press/v306/ding26k.html) | temporal-order discriminationを含むimage-to-video評価、Base主要特異部分空間とLoRA更新の干渉、LoRA更新ダイナミクスの理論化。 |

## 3. 一般的なLoRA解析方法

LoRAを \(W'=W_0+\Delta W\), \(\Delta W=sBA\) とする。AとBを個別に解釈せず、原則として実効更新 \(\Delta W\) を解析する。任意の可逆行列Rに対し、BとAの間へRとRの逆行列を入れても同じΔWになるため、A/B単体の方向には一意な意味がない。

| 方法 | 具体的に測るもの | 分かること | 単独では分からないこと |
|---|---|---|---|
| 更新量監査 | \(\|\Delta W\|_F\)、\(\|\Delta W\|_2\)、\(\|\Delta W\|_F/\|W_0\|_F\)、層別Q/K/V比 | どこがどれだけ変わったか | 変化がmotionかどうか |
| SVD・rank | \(\Delta W=U\Sigma V^T\)、累積energy、effective rank、stable rank | 少数方向への集中、層ごとの容量利用 | 高特異値＝動作概念、とは言えない |
| Baseとの干渉 | W0の上位特異部分空間へのΔW energy、principal angles | 既存知識方向を上書き／回避しているか | 動的意味の有無 |
| LoRA間比較 | seed、ordered、shuffled、static学習LoRA間のFrobenius cosine、主角度、Grassmann distance | 共通方向と条件固有方向 | 同じsubspace＝同じ機能、とは限らない |
| 表現比較 | 層ごとのCKA、SVCCA、PWCCA、RSA | 入力に対する内部表現がどの層で変わったか | 変化の正しさ・必要性 |
| residual解析 | \(r_l(x)=h_l^{LoRA}(x)-h_l^{Base}(x)\) をprobe | LoRAが追加した情報をBase本体と分ける | probeのshortcutを除かない限り因果ではない |
| scale介入 | LoRA scale αを0→1→1.5等で掃引 | 効果の用量反応、LoRAへの帰属 | α増加が分布外挙動を作る可能性 |
| module knockout | 層、Q/V、head、特異成分を一つずつ除去 | 必要な部位・成分 | 冗長成分があると重要性を過小評価 |
| sufficiency test | 一部層／上位特異成分だけを残す | その成分だけで機能を再現できるか | 相互作用があると単純加算できない |
| matched random control | 同rank・同Frobenius normのランダム部分空間を除去／追加 | 単なる容量・ノルム効果を除く | データ交絡 |
| 時系列解析 | checkpointごとのSVD、主角度、probe、forgetting | 学習中にいつ能力・干渉が生じたか | 最終機能の正しさ |

### 推奨するSVD指標

- 層lの相対更新量：\(R_l=\|\Delta W_l\|_F/\|W_{0,l}\|_F\)
- 上位k成分のenergy：\(E_l(k)=\sum_{i\le k}\sigma_i^2/\sum_i\sigma_i^2\)
- effective rank：正規化特異値のentropyから算出。ただしLoRAの設定rankを超えない。
- Base主要部分空間との重なり：W0上位k左／右特異ベクトルへΔWを射影したenergy比。
- 条件間の主角度：ordered-LoRAとshuffled-LoRAの左・右特異部分空間を別々に比較。

### SVDの正しい使い方

1. まず各ΔWのスペクトルを記述する。
2. 上位成分を除去／縮小し、dynamic probeが落ちるか測る。
3. 同じenergyを持つランダム成分除去と比較する。
4. static probeも同時に測り、dynamicだけが選択的に落ちるか確認する。

「上位特異値が大きい」「rank 8のうち2方向が支配的」だけでは、動的情報を学んだ証拠ではない。

## 4. 静的・動的情報を分けて評価する方法

### 4.1 入力介入

| 介入 | 保つもの | 壊すもの | 主な解釈 |
|---|---|---|---|
| reverse | frame集合・appearance | 時間方向 | 正逆・状態遷移への感度／能力 |
| shuffle | frame集合 | 局所・大域順序 | 順序依存性 |
| static repeat | 1 frameのappearance | frame間変化 | motionの有無。ただしframe集合も変わる |
| speed change | 空間内容・おおよその軌跡 | 速度 | 速度表現 |
| frame drop／stride | 内容の一部 | 時間分解能 | 時間scaleへの頑健性 |
| background swap | 前景motion | scene・背景 | 静的shortcutへの依存 |
| foreground／identity swap | 背景・軌跡の一部 | 対象appearance | motionとidentityの分離 |
| motion transfer | content | motion | motion codeの移植可能性 |
| content transfer | motion | content | content codeの分離 |

reverse・shuffleは「特徴が変わるか」だけでなく、正しいラベルへ変わるかを測る。背景置換は元動画との特徴一致だけでなく、dynamic predictionが保たれるかを測る。

### 4.2 probe

- **Static probe**：object、scene、background、identity、color／shape。単一frameで学習・評価できるもの。
- **Dynamic probe**：forward/reverse、順序、方向、速度bin、state transition、phase、temporal action class。
- **Residual probe**：full featureではなくLoRA residual r(x)だけから同じラベルを予測する。
- **Cross-factor probe**：訓練時に見ていないappearance×motionの組合せへ一般化する。
- **Cross-dataset probe**：ActivityNetで学んだ表現をSSv2 temporal subset、Diving48、FineGym等で評価する。

linear probeはencoderを凍結し、pre-projector CLSを主に使う。qはProjectorが情報を吸収できるため補助とする。全モデルでprobeの学習データ、正則化、探索範囲、seedを揃える。

### 4.3 disentanglement指標

合成データなどground-truth factorがある場合は、DCI、MIG、SAP、FactorVAE scoreを参考にできる。ただし、[Locatello et al., ICML 2019](https://proceedings.mlr.press/v97/locatello19a.html)が示すように、教師なしdisentanglementは帰納バイアスなしには同定できず、指標間の順位も安定しない。実動画では、これらを主指標にせず、factorを制御したcounterfactual pairと機能テストを優先する。

## 5. 現在の研究への適用

### 確認済み事実

- モデルは画像事前学習ViTを基盤とし、4フレームをtoken結合するearly fusion、固定時間位置埋め込み、Q/V LoRA、MoCo Projectorを持つ。
- 現行positiveは同じclipの2 augmentationであり、静的な物体・背景だけでもcontrastive lossを下げられる。
- G2はsequential-loader、stride 16、6,255stepを完走し、lossは低下、collapseは観測されなかった。
- しかしG2のshuffle／reverse cosはほぼ1で、staticだけが相対的に大きく下がった。順序を区別する情報を得た観測はない。

### 解釈

最も自然な仮説は、現在の目的関数がLoRAに「同一clipの見た目をaugmentation間で一致させる」ことを主に学ばせ、順序を使う必要がないことである。別の可能性として、LoRAが動画domainへの適応としてframe間の変化を抑え、よりinvariantな表現を作ったことも考えられる。cosだけでは両者を分けられない。

### 未検証

- reverse／shuffleを分類できないのか、それとも特徴差が小さくてもlinear/nonlinear probeで読めるのか。
- ΔW residualにmotion情報があるか。
- 動的能力が特定層・Q/V・特異成分に局在するか。
- ordered学習とshuffled学習の差があるか。現在のR1 C-LoRAとG2は学習量・loader・frame間隔が一致せず、因果比較ではない。

## 6. 推奨する評価プロトコル

### Tier 1：既存checkpointだけでできる低コスト解析

1. **LoRA α sweep**：A、B-LoRA、C-LoRA、G2-LoRAでα={0, 0.25, 0.5, 0.75, 1.0, 1.25}。normal/shuffle/reverse/staticのCLS、residual、dynamic probeを測る。
2. **residual probe**：各層のCLS residualと最終CLS residualから、reverse、shuffle、速度、motion magnitudeを予測する。同時にscene／video identity probeを置く。
3. **層別・Q/V knockout**：1層ずつ、block群ごと、Qのみ、Vのみを無効化。dynamic scoreとstatic scoreの差を測る。
4. **ΔW SVD＋介入**：上位1成分ずつ除去、下位成分除去、同energyランダム除去。dynamicが選択的に落ちる成分があるか。
5. **部分空間比較**：C-LoRA対G2-LoRA、seed間、将来作るordered対shuffledで主角度・CKAを測る。

Tier 1はLoRA内部の帰属を強めるが、現在のデータだけでは静的shortcutを十分に除けない。

### Tier 2：静動を制御した機能評価

1. **合成factorial動画**を作る。shape、色、texture、背景、identityをstatic因子、方向、速度、軌跡、加速度、順序をdynamic因子とする。
2. 訓練と評価でappearance×motionの組合せを分ける。未見組合せへのdynamic predictionを主指標にする。
3. 実動画では、SSv2のtemporal class／Chiral SSv2、Diving48、FineGymを小規模probeに使う。
4. BE/FAME型のbackground swapでmotionを保持し、dynamic predictionが保たれるか測る。
5. normal対reverseを同じframe集合のpaired sampleとして扱い、paired bootstrap 95% CIを出す。

### Tier 3：オンライン学習が効いたかの因果比較

学習サンプル集合、総update数、rank、optimizer、augmentation、positive/negative規則、Queue長、seedを揃え、次を比較する。

| 条件 | clip内部 | stream更新順 | 調べるもの |
|---|---|---|---|
| O-N | normal | ordered single-pass | 提案条件 |
| S-N | normal | globally shuffled | オンライン順序の効果 |
| R-N | normal | reverse stream | stream方向の効果 |
| O-FS | frame-shuffled | ordered | clip内部の時間信号の必要性 |
| S-FS | frame-shuffled | shuffled | ほぼ静的な対照 |

評価は一定stepごとに固定held-out setで行い、dynamic probe、static probe、LoRA spectrum、representation drift、過去区間の保持、未来区間へのforward transfer、wall-clock／memoryを記録する。Queueの実装は固定し、順序に伴うQueue構成の差はオンライン条件の一部として扱う。Queueだけの寄与を分けたい場合は、別のfactorとしてqueue-free／fixed-bank対照を置く。

## 7. 主張を認める判定条件

「LoRAが動的情報を学習した」は、少なくとも次が複数seedで成立した場合に限定する。

1. BaseよりLoRAありでdynamic taskが改善し、paired CIが0をまたがない。
2. ordered-LoRAが同一予算のshuffled-stream LoRAよりdynamic taskで改善する。
3. background／appearanceを変えても改善が保たれ、static-only baselineでは説明できない。
4. LoRAを外す、αを下げる、重要層／特異成分を除くとdynamic改善が減る。
5. 同じ介入でstatic taskが同程度以上に落ちるだけ、という結果ではない。
6. pre-projector CLSまたはLoRA residualで再現し、Projectorだけの効果ではない。
7. 少なくとも合成factorialデータ＋実動画の2系統で再現する。

より正確な論文表現は「LoRAがBaseに追加する表現に、静的shortcutだけでは説明できない複数フレーム依存の機能が含まれた」であり、「動的情報がLoRAにのみ保存された」ではない。

## 8. 失敗結果の読み方

| 結果 | 解釈 |
|---|---|
| dynamic probeもreverse判定も改善しない | 現行目的関数／clip設定では動的情報を獲得していない可能性が高い |
| 特徴cosはほぼ1だがprobeは改善 | 差は小さいが線形に読める方向へ整理された可能性 |
| staticとdynamicが同時に改善 | 一般的なvideo domain adaptation。分離の証拠ではない |
| background swapで崩れる | scene shortcut依存 |
| SVD上位成分除去で全taskが落ちる | 一般容量成分。motion特異的とは言えない |
| orderedとshuffledが同等 | 時刻順学習の優位性なし。clip内容または総学習量の効果 |
| G2がさらに順序不変になる | objectiveが時間変化を捨てるinvarianceを強めた可能性 |

## 9. 優先順位

### 最優先

1. 既存G2でpre-projector CLSとLoRA residualのreverse／shuffle／speed probe。
2. LoRA α sweepとQ/V・層別knockout。
3. ΔW SVD＋上位成分のcausal ablation＋matched random control。

### 次点

4. 小さなfactorial moving-shapesデータでstatic/dynamic crossed split。
5. SSv2 temporal subsetまたはChiral SSv2でlinear probe。
6. ordered対shuffled-streamのmatched training comparison。

### 補助

7. CKA／SVCCA／主角度、attention可視化、gradient norm、motion magnitude相関。

## 10. 保留・棄却寄り

- **特異値だけで動作概念数を数える**：棄却寄り。高特異値は大きい線形変換方向であり、1方向＝1動作ではない。
- **ActivityNet action linear probeだけで判断**：棄却寄り。scene・object biasで解ける。
- **qだけを見る**：棄却寄り。Projectorの学習とLoRA獲得を分けられない。
- **reverseをnegativeにするだけ**：保留。逆再生対称動作、編集artifact、低レベル手掛かりがfalse negative／shortcutになる。
- **1 seedのcos差に閾値を置く**：棄却。paired CI、機能課題、複数seedが必要。

## 11. 主要な解析手法の原論文

- [SVCCA, NeurIPS 2017](https://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-understanding-and-improvement)
- [PWCCA, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/a7a3d70c6d17a73140918996d03c014f-Abstract.html)
- [CKA, ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html)
- [FactorVAE score, ICML 2018](https://proceedings.mlr.press/v80/kim18b.html)
- [Unsupervised disentanglementの非同定性, ICML 2019](https://proceedings.mlr.press/v97/locatello19a.html)
- [弱教師ありdisentanglement, ICML 2020](https://proceedings.mlr.press/v119/locatello20a.html)
- [CaSSLe, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Fini_Self-Supervised_Models_Are_Continual_Learners_CVPR_2022_paper.html)

## 12. Research Spec Handoff候補

- 対象プロジェクト：sequential-video-lora-analysis
- 中心の問い：G2-LoRAがBaseに追加した表現に、静的shortcutでは説明できない複数フレーム依存機能があるか。
- 採用方向：入力介入 × LoRA介入 × static/dynamic probeの三角測量。
- 最小scope候補：既存checkpointのみを使うα sweep、residual probe、Q/V・層別knockout、SVD成分介入。
- 比較対象：A、B-LoRA、C-LoRA、G2-LoRA、必要に応じFull FT。将来のmatched ordered／shuffled LoRA。
- 主指標：dynamic probe差、static leakage、paired bootstrap CI、LoRA on/off差、介入による選択的性能低下。
- 対象外：このメモからの実装、GPU run、目的関数変更、TODO追加。
- 主要未決：最初のdynamic taskをreverse/order/speedのどれにするか、合成データを先にするか、SSv2を先にするか、seed数、計算予算。

## 関連Research Workspace

- プロジェクトREADME：`.research/lab/projects/sequential-video-lora-analysis/README.md`
- 現行の時間順序評価spec：`.research/lab/projects/sequential-video-lora-analysis/specs/2026-11-03-temporal-order-sensitivity-evaluation-spec.md`
- G2評価Evidence：`.research/lab/projects/sequential-video-lora-analysis/experiments/2026-11-11-temporal-order-sensitivity-evaluation-g2-gpu-evidence.md`
- 先行する解析計画：`.research/secretary/notes/brainstorm/2026-10-27-early-fusion-lora-dynamic-information-analysis-plan.md`
- 近接研究と新規性：`.research/secretary/notes/brainstorm/2026-10-29-deep-related-work-static-dynamic-online-video-lora-novelty.md`

次アクション候補：Tier 1を実験specへ落とす前に、dynamic taskを「reverse二値分類＋shuffle permutation分類＋speed bin」のどこまで含めるかを決める。TODOとspecへは自動昇格しない。
