---
date: 2026-11-12
project: sequential-video-lora-analysis
source_todo: null
topic: lora-static-dynamic-analysis-literature-survey
status: exploratory
tags: [brainstorm, research, lora, video, static-dynamic, evaluation, literature-survey]
---

# LoRAが動的情報を学習したかを検証する解析法：2014–2026年の関連研究整理

## 相談の出発点

画像事前学習ViTを凍結し、動画を逐次入力してLoRAを学習したとき、LoRAが背景・人物・物体などの静的情報ではなく、時間変化・運動・順序などの動的情報を追加的に獲得したかを、どのように検証すべきかを整理する。

対象は、主に2014〜2026年のCVPR、ICCV、ECCV、WACV、NeurIPS、ICLR、ICMLの論文とし、LoRAの一般的解析、ニューラル表現比較、動画の静的・動的分離、時間情報の自己教師あり学習、データセットバイアス評価を横断した。TMLRのLoRA分析も補助的に含める。

本メモは文献調査に基づく探索記録であり、実験specや実装許可ではない。

## 現在のプロジェクト文脈とEvidence

Research Workspaceの2026-11-11時点の記録では、次が確認済みである。

- ActivityNetの4フレームclipをEarly FusionしてLoRAまたはFull FTで459 step学習する経路は、工学的には完走している。
- sequential-loaderでLoRAを6,255 step学習したG2も完走し、Queue充填後のlossは8.348から6.964へ低下し、collapseは観測されなかった。
- しかし、ActivityNet validationの2,000 clipでnormalとshuffle／reverseを比較すると、459 stepのclip LoRAもG2 LoRAもcos類似度はほぼ1だった。
- G2 LoRAのCLSはshuffle 0.9974、reverse 0.9956、static-repeat 0.9280、qはshuffle 0.9982、reverse 0.9971、static-repeat 0.9331だった。
- 学習前より学習後のほうがshuffle／reverseに鈍く、G2は10秒離れた同一動画clipとの類似度も高い。別動画とのqのcosは0.0082まで下がった。
- よって、現時点で確認できるのは「動画識別・同一動画内で持続する特徴が強くなった可能性」であり、「LoRAがフレーム順序や動作を学習した」とは言えない。
- 現評価は1 seed、4フレーム、評価間隔0.25秒、学習時G2間隔は約0.53秒で、動的表現の品質を十分に判定するEvidenceではない。

関連する正本:

- [プロジェクトREADME](https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/README.md)
- [時間順序感度評価spec](https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/specs/2026-11-03-temporal-order-sensitivity-evaluation-spec.md)
- [459 stepモデルの時間順序感度Evidence](https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/experiments/2026-11-04-temporal-order-sensitivity-evaluation-r1-gpu-evidence.md)
- [6,255 step G2の時間順序感度Evidence](https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/experiments/2026-11-11-temporal-order-sensitivity-evaluation-g2-gpu-evidence.md)

## 文献調査から得られる中心的な結論

**LoRAが動的情報を学んだことを、SVD、cos類似度、linear probeのどれか1つだけで示すことは難しい。**

最も説得力のある設計は、次の4種類のEvidenceを組み合わせることである。

1. **LoRA重みの構造**  
   ΔWのSVD、有効ランク、元重み・別seed・別学習条件との部分空間比較を行う。
2. **LoRAが表現へ加えた情報**  
   Base、Base+LoRA、LoRAによる差分表現を比較し、動的因子と静的因子のどちらが読み出せるかを測る。
3. **入力因子を独立に変える反実仮想評価**  
   見た目を変えて動きを保つ、または動きを壊して見た目を保つ動画対を作り、どちらに反応するかを測る。
4. **因果的アブレーションと動的下流課題**  
   特定layer・Q/V・特異方向を除去したとき、動的課題だけが選択的に悪化するかを確認する。

この4つが同じ結論を示したとき、初めて「LoRAが主として動的情報を追加した」という主張が強くなる。

## 1. 一般的なLoRA解析

### 1.1 ΔWを解析対象にする

LoRAでは各対象行列に対して

[
W' = W_0 + sBA = W_0 + ΔW
]

となる。AとBには、任意の可逆行列Rについて

[
BA = (BR)(R^{-1}A)
]

という再パラメータ化の曖昧性がある。そのため、AとBの要素・方向をそのまま意味解釈するより、まず実際に重みへ加わる **ΔW=sBA** を解析するのが安全である。

例外として、Aの出力 (Ax) とBの出力 (B(Ax)) をactivationとして調べる、A固定/B固定の因果的アブレーションを行う、という解析は有効である。ICML 2024のAsymmetry LoRAは、Aを入力特徴抽出、Bを出力方向への写像として捉え、両者の役割が非対称であることを示している。

### 1.2 SVD・スペクトル解析

各layer・Q/K/Vごとに

[
ΔW = USigma V^	op
]

を計算し、少なくとも次を記録する。

| 指標 | 意味 | 注意 |
|---|---|---|
| (|ΔW|_F / |W_0|_F) | 相対的な更新量 | 大きいだけでは有用とは限らない |
| (|ΔW|_2 / |W_0|_2) | 最大方向の強さ | 1方向への集中を検出しやすい |
| 特異値スペクトル | rank内で何方向が使われたか | rank=8でも実質1〜2方向の場合がある |
| cumulative energy@k | 上位k方向が占める二乗エネルギー | 上位方向アブレーションのk選択に使う |
| stable rank | (|ΔW|_F^2/|ΔW|_2^2) | 実効的な広がり |
| entropy effective rank | 正規化特異値のエントロピー | layer間比較に使える |
| principal angles | 2つの更新部分空間の重なり | seed、ordered/shuffle、単一frame/clip間で比較 |
| (U,V)と(W_0)の主特異空間の重なり | 元モデルの既存方向を使うか、新方向を作るか | 新方向=動的情報とは限らない |

LoRA原論文はrank-deficiencyと更新部分空間を調べた。TMLR 2024の“LoRA Learns Less and Forgets Less”はFull FTの更新が一般的なLoRA rankより10〜100倍高rankになりうると報告した。NeurIPS 2025の“LoRA vs Full Fine-tuning”は、LoRAで元重みの主特異空間と異なる高順位の“intruder dimensions”が生じること、特異値を介入的に縮小すると忘却が軽減することを示した。したがって、**SVDは記述だけで終えず、特異方向を除去・縮小して性能変化を測る**ところまで行うと因果的Evidenceになる。

### 1.3 重みの方向と大きさ

ICML 2024のDoRAは、重みをmagnitudeとdirectionに分けるとLoRAとFull FTの差を説明しやすいとした。本研究でも、各列またはattention head単位で、

- weight normの変化
- cosine方向変化
- QとVのどちらで変化が強いか
- head別の変化量

を分けて測る価値がある。ただし、方向変化が大きいこと自体は動的情報の証明ではない。後述する動的・静的タスクの因果アブレーションと結び付ける必要がある。

### 1.4 layer・target moduleの因果的アブレーション

以下を同じcheckpointから作る。

- 全LoRA
- LoRAなし
- Qのみ、Vのみ
- 前半layerのみ、後半layerのみ
- layerを1つずつ除去
- 各ΔWの上位特異方向のみ
- 各ΔWから上位特異方向を除去
- 同じFrobenius normを除去するランダム方向
- LoRA scale (gammain{0,0.25,0.5,1,1.5,2})

動的課題の性能が、静的課題よりも特定のlayer・方向・scaleに強く依存するなら、動的情報との因果的対応を示せる。

## 2. 表現解析

### 2.1 Base、Base+LoRA、差分表現を分ける

同じ入力xに対して、

[
h_0(x)=f_{W_0}(x), quad
h_L(x)=f_{W_0+ΔW}(x), quad
Delta h(x)=h_L(x)-h_0(x)
]

を保存する。

評価対象は最低限、次の3つである。

- BaseのCLS
- Base+LoRAのCLS
- ΔCLS

さらにlayerごとのhidden state、各LoRA branch出力、Q/K/Vのattention outputも保存すると、どの段階で動的情報が現れるかを調べられる。

**重要な注意:** Base+LoRA全体のCKAやcosだけを見ると、凍結Baseの大きな共通成分に埋もれてLoRA差分が見えにくい。したがって、全表現の比較に加えて Δh またはlayer内のLoRA寄与を解析する。

### 2.2 表現類似度

- **CKA**: layer間、seed間、学習条件間の表現類似度。高次元・サンプル数より特徴次元が大きい場合にも比較的扱いやすい。
- **SVCCA / PWCCA**: 表現の主要部分空間を比較。SVCCAはSVDで分散99%程度の方向へ圧縮してからCCAを行う。
- **RSA**: サンプル間距離行列の相関。motion pair／appearance pairの幾何を比較しやすい。
- **principal-angle analysis**: ΔhやΔWの部分空間同士の重なりを直接比較。
- **nearest-neighbor retrieval**: actionが同じでappearanceが異なる動画を近く、identityやsceneだけが同じ動画を遠くできるか。

CKA/SVCCAは「似ている／違う」を示すが、何の情報が違うかは示さない。必ず次のprobeまたは反実仮想対と組み合わせる。

### 2.3 probe matrixとcross-leakage

同一のfrozen featureに小さなprobeを学習し、次の行列を作る。

| representation | 動的ラベル | 静的ラベル |
|---|---:|---:|
| Base CLS | action/verb/order/speed/direction | scene/object/person/background |
| Base+LoRA CLS | 同上 | 同上 |
| ΔCLS | 同上 | 同上 |
| 各layerのLoRA寄与 | 同上 | 同上 |

望ましい結果は、Base+LoRAとΔCLSで動的ラベルのdecodabilityが増え、静的ラベルの増加が小さいことである。

ただしlinear probeは、分離された情報だけでなく「線形に読みやすくなった情報」を測る。linearと小型MLPの両方を用い、probe容量を固定し、複数seedで評価する。

cross-leakageとして、

- motion representationからidentity/sceneを予測できてしまう精度
- static representationからaction/orderを予測できてしまう精度

を報告する。VDSM、DiViD、DEVIASなどの静的・動的分離研究は、factor classification、swap、cross-leakage、未知のaction-scene組合せを用いている。

## 3. 静的情報と動的情報を独立に変える評価

### 3.1 Kowal et al. CVPR 2022の考え方

Kowal et al.は、動画対を次のように作り、中間表現の相関から静的・動的biasを定量化した。

- **static pair**: 同じフレーム集合をshuffleし、見た目を保ちながら時間構造を壊す。
- **dynamic pair**: 同じ動きを保ちながらstyleを変え、見た目を変える。
- **identical pair**: 同一動画。

layerまたはchannelごとにペアのactivation相関を計算し、static、dynamic、joint、residual unitを推定する。相関をjoint Gaussian仮定の下でmutual informationの下界として扱う。

本研究では、Kowalの手順をそのままCLSだけに適用するより、次のようにLoRA差分へ拡張する。

[
G_{mathrm{dyn}} =
I_{mathrm{dyn}}(Base+LoRA)-I_{mathrm{dyn}}(Base)
]

[
G_{mathrm{stat}} =
I_{mathrm{stat}}(Base+LoRA)-I_{mathrm{stat}}(Base)
]

主張したいのは単なる (I_{mathrm{dyn}}>I_{mathrm{stat}}) ではなく、**LoRAを足した増分として (G_{mathrm{dyn}}>G_{mathrm{stat}})** である。

### 3.2 推奨する反実仮想入力

| 条件 | 保つもの | 変えるもの | 検証すること |
|---|---|---|---|
| normal vs reverse | フレーム集合・見た目 | 時間方向 | arrow of time |
| normal vs nontrivial shuffle | フレーム集合・見た目 | 局所順序 | 順序感度 |
| normal vs static-repeat | 最終frameの見た目 | 動き・他frame | 動きの有無。ただし情報量も減る |
| normal vs speed change | 内容・順序 | 速度 | 速度表現 |
| same motion, style changed | motion | texture/color/style | appearance不変性 |
| same motion, background replaced | foreground motion | scene/background | 背景bias |
| same action, different person/object/scene | action | identity/content | semantic motion retrieval |
| same scene/object, different action | appearance | motion | 静的shortcutの排除 |
| time-shifted same video | video identity | phase/action state | 動画内時間変化 |

static-repeatは動きだけでなく3枚分のappearance情報も失うため、単独では動的情報の証拠にならない。style変換、background replacement、same-motion/different-appearance対を併用する。

### 3.3 変換が自明な手掛かりを作らないための確認

- reverseやshuffle前後で使うフレーム集合を完全一致させる。
- compression artifact、frame duplication、境界の不連続だけで解けないかを確認する。
- optical-flow magnitude、camera motion、cutの有無で層別化する。
- 人間が順・逆を判定できないclipを主要評価から除くか、別群として報告する。
- style変換前後でflow・trajectoryが保たれているかを確認する。
- クラスやclipの採用規則は結果を見る前に固定する。

## 4. 動的情報を必要とする下流評価

### 4.1 データセット・課題

| 評価 | 長所 | 本研究での用途 |
|---|---|---|
| Something-Something V2 | 物体操作と時間関係を必要とするclassが多い | 主たるdynamic linear probe候補 |
| Temporal classes / Temporal-50系 | shuffleすると人間認識が落ちるclassを人手で選ぶ | 順序が必要なclassだけに絞る |
| AFD | 1 frameの静的情報からactionを解けないよう設計 | appearance-free control |
| Diving48 |一般的action datasetより静的biasを抑える | 補助。Kowal et al.は完全にdynamic-onlyではないと指摘 |
| SCUBA / SCUFO | 背景またはforegroundの静的shortcutを合成 | 静的漏洩の定量化 |
| HAT | 人物・背景を操作しbackground biasを評価 | counterfactual OOD評価 |
| SimMotion | appearanceを越えたsemantic motion similarity | motion retrieval候補 |
| ActivityNet validation | 現学習分布と同じで実装しやすい | sanity check。dynamic主評価には単独使用しない |

### 4.2 ラベルなしでも可能な課題

- forward/reverse 2値分類
- permutation/order classification
- playback speed classification
- temporal distance prediction
- phase retrieval、video alignment、cycle consistency
- same motion/different appearance retrieval
- future motion/trajectory/optical-flow prediction
- motion-direction classification

Arrow-of-timeは、不可逆な物理変化を含むclipに限定しないとchance付近でも異常ではない。4フレーム・0.75秒では動きが弱い可能性が高いため、8/16フレームまたは複数のtemporal spanを併用する。

## 5. 学習条件そのものを使った因果対照

動的情報がLoRAに入ったかを最も直接的に示すには、入力順序だけを変えた対照学習runを用意する。

| training condition | 内容 |
|---|---|
| ordered | 現在の時間順 |
| shuffled-within-clip | 同じframe集合を毎clip固定shuffle |
| reverse | 逆順 |
| static-repeat | 同じframeをT回 |
| single-frame | 現在のB-LoRA対照 |
| random-video order | 動画内順序も壊す |
| motion-focused objective | order/speed/flow/temporal negativeを追加 |

データ量、frame集合、optimizer、LR、step数、Queue、augmentationを合わせる。

たとえば動的probe性能 (P) に対して、

[
mathrm{Temporal Gain}
=
[P(Base+LoRA_{ordered})-P(Base)]
-
[P(Base+LoRA_{shuffled})-P(Base)]
]

を求める。orderedだけが動的課題で改善し、静的課題では差が小さいなら、時間順序による学習の因果効果を示せる。

現在のpositiveが「同じclipの2拡張」である限り、背景・人物・物体を使ってInfoNCEを解ける。CVPR 2022のFAMEは、naiveなvideo contrastive learningが共通する静的背景をshortcutとして使う問題を明示している。ICCV 2021のCORPは、通常のcontrastive objectiveだけではframe/clip間の時間関係を明示的に学ばないと指摘し、contrastとorderを分けた。したがって、現在の結果は実装失敗というより、**objectiveが時間情報を要求していないために生じる予想可能な解**と解釈できる。

## 6. 推奨する評価バッテリー

### Tier 0：工学sanity

- Base凍結、LoRA更新、checkpoint読込を確認。
- 同一入力で再現可能。
- norm、rank、gradient、updateがfinite。
- LoRA scale=0でBaseと一致。
- 評価データ・frame index・transformのhashを保存。

### Tier 1：既存checkpointだけで行える低コスト解析

1. 全Q/V・layerのΔW SVD、stable/effective rank、energy@k。
2. Base、Base+LoRA、ΔCLSを全layerで保存。
3. normal／reverse／shuffle／staticに加え、style/background interventionを追加。
4. CKA、SVCCA、pair correlationを計算。
5. motion/appearanceのsmall probe matrixを作る。
6. paired video-level bootstrapで95% CIを出す。

### Tier 2：因果的LoRA解析

1. layer、Q/V、head、特異方向のアブレーション。
2. top-k除去とenergy-matched random除去を比較。
3. LoRA scale sweep。
4. 動的課題と静的課題の性能低下を同時に測る。
5. ordered LoRAとsingle-frame/shuffled LoRAの部分空間・表現差を比較。

### Tier 3：再学習を含む確認実験

1. ordered vs shuffled-within-clipを同一frame集合・同一computeで3〜5 seed。
2. 時間を必要とするpositive/negativeまたはorder/speed auxiliary lossを追加。
3. ActivityNetは学習に使い、評価はSSv2/Temporal classes/AFD等へ転移。
4. unseen appearance、unseen scene、unseen actor/objectでcross-combination generalizationを測る。

## 7. 主要指標

単一の「動的スコア」に潰さず、最低限以下を並べる。

- Dynamic probe gain: (P_{dyn}(Base+LoRA)-P_{dyn}(Base))
- Static probe gain/leakage: (P_{stat}(Base+LoRA)-P_{stat}(Base))
- ΔCLS dynamic/static decodability
- Kowal型 (G_{dyn},G_{stat})
- reverse/shuffle/staticのpaired effect size
- unseen appearance/sceneでのdynamic task performance
- ablationによるdynamic task dropとstatic task drop
- layer/Q/V/特異方向別寄与
- seed平均、標準偏差、95% CI
- video単位のhierarchical bootstrap

複数clipを同一videoから取る場合、clipを独立サンプルとしてbootstrapすると疑似反復になる。まずvideoをresampleし、その内部でclipを集計する。

## 8. 何が得られれば「動的情報を学んだ」と言えるか

### 弱いEvidence

- reverse/shuffleでcosが少し下がる。
- ΔWに非ゼロ特異値がある。
- lossが下がる。

これらだけでは不十分である。

### 中程度のEvidence

- BaseよりBase+LoRAでorder/speed/action probeが改善。
- ΔCLSからdynamic labelを読み出せる。
- style/backgroundを変えてもmotion retrievalが維持される。
- 静的labelの読み出し増加がdynamic labelより小さい。

### 強いEvidence

- ordered学習だけがshuffled学習よりdynamic taskを改善。
- 特定のLoRA layer/特異方向を除くとdynamic taskだけが大きく落ちる。
- energy-matched random方向除去では同じ低下が起きない。
- unseen appearance/scene/personでも同じmotionを近く検索できる。
- 結果が複数seed・複数datasetで再現する。

### 最も強い主張に近いEvidence

- 動的情報の増加、静的漏洩の小ささ、因果アブレーション、OOD generalizationが同時に成立する。
- Base、Full FT、single-frame LoRA、shuffled LoRAより明確に優れる。
- 事前登録したclass/clip選択と統計手順で再現する。

## 9. 本研究への推奨順位

### 最優先：既存checkpointの解析を拡張

現在のreverse/shuffle/static結果はbaselineとして有用だが、単独では「時間情報がない」と断定し切れない。まず次を追加する。

1. **Kowal型のsame-motion/different-appearance対**  
   style/backgroundを変え、motionを保つ。
2. **ΔCLSとlayer-wise LoRA contribution**  
   Base成分に埋もれないようにする。
3. **dynamic/static probe matrix**  
   motion/order/actionとscene/object/video identityを対にして測る。
4. **ΔW SVD + top singular direction ablation**  
   重みの記述と性能への因果効果を接続する。
5. **G2の学習間隔に合わせた評価**  
   0.53秒間隔と、より長い8/16 frame spanを追加する。

### 次点：objectiveの対照

- orderedとshuffleを同一frame集合で学習。
- temporal order、speed、future motion、flowのいずれかをauxiliary targetにする。
- negativeを「別動画」だけにせず、同一video内で意味のある時間差・誤順序を使う。
- ただし近接clipをすべてnegativeにすると、同じaction phaseをfalse negativeにする可能性がある。時間距離とannotationを使って除外規則を決める。

### 主評価データ

- ActivityNetは学習・sanity・一般活動認識の対照として残す。
- 「動的情報」の主評価はSSv2、Temporal classes、AFD、またはappearanceを制御した独自paired setに置く。
- Diving48は補助に使い、dynamic-only datasetとはみなさない。

## 10. 反例・限界

- ΔWの高特異方向は、新しい情報を示しても動的情報とは限らない。
- ΔCLSにactionが入っても、背景からactionを予測している可能性がある。
- reverseに反応しなくても、速度・軌跡・変形など別の動的情報を持つ可能性がある。
- reverseに反応しても、編集artifactを検出しただけの可能性がある。
- 「静的情報が少ない」ことと「動的情報が多い」ことは同値ではない。
- 完全なunsupervised disentanglementは、data/modelの帰納バイアスなしには同定できないという理論的限界がある。
- LoRAがFull FTより低rankであるため、動的情報が獲得できないのか、objectiveが要求していないのかを分ける必要がある。
- 1 seedのLoRA部分空間一致・不一致は結論に使わない。
- LoRAのA/B単独のnorm比較は再パラメータ化に依存する。

## 11. 文献マップ（2014–2026）

### 時間・運動表現と評価の流れ

- 2014, NeurIPS: [Two-Stream Convolutional Networks for Action Recognition in Videos](https://proceedings.neurips.cc/paper_files/paper/2014/hash/ca007296a63f7d1721a2399d56363022-Abstract.html) — appearanceとoptical-flow motionの分離。
- 2015, CVPR: [Modeling Video Evolution for Action Recognition](https://openaccess.thecvf.com/content_cvpr_2015/html/Fernando_Modeling_Video_Evolution_2015_CVPR_paper.html) — frameの時間順序をrank poolingで表現。
- 2016, ECCV: [Shuffle and Learn](https://publications.ri.cmu.edu/shuffle-and-learn-unsupervised-learning-using-temporal-order-verification) — 時間順序verification。
- 2016, CVPR: [Dynamic Image Networks](https://openaccess.thecvf.com/content_cvpr_2016/html/Bilen_Dynamic_Image_Networks_CVPR_2016_paper.html) — temporal evolutionを1枚のdynamic imageへ集約。
- 2017, ICCV: [Something-Something](https://openaccess.thecvf.com/content_iccv_2017/html/Goyal_The_Something_Something_ICCV_2017_paper.html) — 物理的・時間的common senseを要求するdataset。
- 2017, CVPR: [Odd-One-Out Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Fernando_Self-Supervised_Video_Representation_CVPR_2017_paper.html) — 誤ったframe順序を見抜くpretext。
- 2017, NeurIPS: [Unsupervised Learning of Disentangled Representations from Video](https://proceedings.neurips.cc/paper/2017/hash/2d2ca7eedf739ef4c3800713ec482e1a-Abstract.html) — stationary contentとtime-varying poseの分離。
- 2018, CVPR: [Learning and Using the Arrow of Time](https://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Learning_and_Using_CVPR_2018_paper.html) — forward/reverse判定。
- 2018, ECCV: [RESOUND / Diving48](https://openaccess.thecvf.com/content_ECCV_2018/html/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.html) — static representation biasを減らすdataset設計。
- 2018, CVPR: [MoCoGAN](https://openaccess.thecvf.com/content_cvpr_2018/html/Tulyakov_MoCoGAN_Decomposing_Motion_CVPR_2018_paper.html) — content固定/motion可変および逆のlatent swap。
- 2018, ECCV: [Temporal Relation Network](https://openaccess.thecvf.com/content_ECCV_2018/html/Bolei_Zhou_Temporal_Relational_Reasoning_ECCV_2018_paper.html) — multi-scale temporal relation。
- 2019, ICCV: [SlowFast Networks](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html) — slow semanticsとfast motionの二経路。
- 2019, CVPR: [Temporal Cycle-Consistency Learning](https://openaccess.thecvf.com/content_CVPR_2019/html/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.html) — phase alignmentとtemporal correspondence。
- 2019, NeurIPS: [Why Can't I Dance in the Mall?](https://proceedings.neurips.cc/paper_files/paper/2019/hash/ab817c9349cf9c4f6877e1894a1faa00-Abstract.html) — scene biasのadversarial抑制。
- 2020, ECCV: [Recognizing Temporal Transformations](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730426.pdf) — forward-backward、random skip、uniform skip。
- 2020, CVPR: [SpeedNet](https://openaccess.thecvf.com/content_CVPR_2020/html/Benaim_SpeedNet_Learning_the_Speediness_in_Videos_CVPR_2020_paper.html) — natural speedの判定。
- 2021, WACV: [Only Time Can Tell](https://openaccess.thecvf.com/content/WACV2021/html/Sevilla-Lara_Only_Time_Can_Tell_Discovering_Temporal_Data_for_Temporal_Modeling_WACV_2021_paper.html) — 人がshuffleで認識困難になるtemporal classを選定。
- 2021, ICCV: [CORP](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_Contrast_and_Order_Representations_for_Video_Self-Supervised_Learning_ICCV_2021_paper.html) — contrastとorderの分離。
- 2021, CVPR: [Context and Motion Decoupling](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Self-Supervised_Video_Representation_Learning_by_Context_and_Motion_Decoupling_CVPR_2021_paper.html) — key frame contextとmotion vector予測。
- 2021, ICCV: [Motion-Focused Contrastive Learning](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Motion-Focused_Contrastive_Learning_of_Video_Representations_ICCV_2021_paper.html) — flowに基づくsamplingとgradient alignment。
- 2021, CVPR: [VDSM](https://openaccess.thecvf.com/content/CVPR2021/html/Vowels_VDSM_Unsupervised_Video_Disentanglement_With_State-Space_Modeling_and_Deep_Mixtures_CVPR_2021_paper.html) — identity/dynamics transferとfactor classification。
- 2022, CVPR: [Kowal et al., Static vs Dynamic](https://openaccess.thecvf.com/content/CVPR2022/html/Kowal_A_Deeper_Dive_Into_What_Deep_Spatiotemporal_Networks_Encode_Quantifying_CVPR_2022_paper.html) — style変換、shuffle、MI/相関によるbias定量化。
- 2022, CVPR: [FAME](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Motion-Aware_Contrastive_Video_Representation_Learning_via_Foreground-Background_Merging_CVPR_2022_paper.html) — contrastive learningのbackground shortcutをforeground-background mergeで抑制。
- 2022, NeurIPS Datasets: [HAT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ff52407b80dde0f0f45814db2738464c-Abstract-Datasets_and_Benchmarks.html) — 74モデルのbackground biasを合成動画で評価。
- 2022, ECCV: [Is Appearance Free Action Recognition Possible?](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2545_ECCV_2022_paper.php) — appearance-free datasetでmotion依存を評価。
- 2023, ICCV: [StillMix / SCUBA / SCUFO](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Mitigating_and_Evaluating_Static_Bias_of_Action_Representations_in_the_ICCV_2023_paper.html) — 背景とforeground双方のstatic bias。
- 2023, CVPR: [Masked Motion Encoding](https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Masked_Motion_Encoding_for_Self-Supervised_Video_Representation_Learning_CVPR_2023_paper.html) — appearance再構成だけでなくdense motion trajectoryを再構成。
- 2024, ECCV: [MotionDirector](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7327_ECCV_2024_paper.php) — spatial/temporal LoRA二経路とappearance-debiased temporal loss。
- 2024, CVPR: [VMC](https://openaccess.thecvf.com/content/CVPR2024/html/Jeong_VMC_Video_Motion_Customization_using_Temporal_Attention_Adaption_for_Text-to-Video_CVPR_2024_paper.html) — consecutive latent residualをmotion referenceとして利用。
- 2024, ECCV: [DEVIAS](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8652_ECCV_2024_paper.php) — action/scene分離と未知action-scene組合せ。
- 2025, NeurIPS: [DisMo](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e88870ec82f2469b0ddf32c817920c68-Abstract-Conference.html) — identityを越えたsame-action retrievalとzero-shot action。
- 2025, ICCV: [Bitrate-Controlled Diffusion](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Bitrate-Controlled_Diffusion_for_Disentangling_Motion_and_Content_in_Video_ICCV_2025_paper.html) — bitrate bottleneckによるmotion/content分離。
- 2026, CVPR Findings: [SemanticMoments / SimMotion](https://openaccess.thecvf.com/content/CVPR2026F/html/Huberman_SemanticMoments_Training-Free_Motion_Similarity_via_Third_Moment_Features_CVPRF_2026_paper.html) — controlled synthetic＋人手実動画でsemantic motion similarityを評価。
- 2026, ICLR: [MoAlign](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c700d47679e325215a3edb323b63f069-Abstract-Conference.html) — optical-flow予測でmotion subspaceを定義し、appearance-entangled featureとの違いを扱う。

### LoRA・表現解析

- 2017, NeurIPS: [SVCCA](https://proceedings.neurips.cc/paper/2017/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html)
- 2018, NeurIPS: [PWCCA](https://proceedings.neurips.cc/paper/2018/hash/a7a3d70c6d17a73140918996d03c014f-Abstract.html)
- 2018, ICLR: [DCI: quantitative evaluation of disentangled representations](https://openreview.net/forum?id=By-7dz-AZ)
- 2018, NeurIPS: [MIG / β-TCVAE](https://proceedings.neurips.cc/paper_files/paper/2018/hash/1ee3dfcd8a0645a25a35977997223d22-Abstract.html)
- 2019, ICML: [CKA](https://proceedings.mlr.press/v97/kornblith19a.html)
- 2019, ICML: [Limits of unsupervised disentanglement](https://proceedings.mlr.press/v97/locatello19a.html)
- 2022, ICLR: [LoRA](https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/)
- 2024, ICML: [DoRA](https://proceedings.mlr.press/v235/liu24bn.html)
- 2024, ICML: [Asymmetry in Low-Rank Adapters](https://proceedings.mlr.press/v235/zhu24c.html)
- 2024, ICLR: [The Expressive Power of LoRA](https://proceedings.iclr.cc/paper_files/paper/2024/hash/154926e0b66e2b2a8c1120852f31a12d-Abstract-Conference.html)
- 2024, NeurIPS: [PiSSA](https://papers.nips.cc/paper/2024/hash/db36f4d603cc9e3a2a5e10b93e6428f2-Abstract-Conference.html)
- 2024, TMLR: [LoRA Learns Less and Forgets Less](https://mlanthology.org/tmlr/2024/biderman2024tmlr-lora/)
- 2025, NeurIPS: [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://proceedings.nips.cc/paper_files/paper/2025/hash/ff541950d1e885af90f523571564a401-Abstract-Conference.html)

## 12. 現時点の方向性

### 最有力

1. ΔW SVDと因果的特異方向アブレーション。
2. Base／Base+LoRA／ΔCLSのlayer-wise解析。
3. Kowal型のsame-motion/different-appearance対とsame-appearance/broken-motion対。
4. dynamic/static probeのcross-leakage matrix。
5. ordered vs shuffled学習のcompute-matched対照。
6. SSv2または人手で確認したtemporal classへの転移評価。

### 保留

- MIG/DCIをそのまま実動画に適用する案。ground-truth factorが明確でないため、合成またはpaired datasetを用意できる場合に限る。
- motion LoRAの生成・swap評価。現在の認識モデルではdecoderがないため、そのままは使えない。
- per-channel static/dynamic unit数。ViTのhidden channelとLoRA rank方向の意味がCNNほど明瞭でないため、layer/rank directionを主にする。

### 棄却寄り

- SVDの特異値が大きいだけで「動的情報」と結論する。
- reverse/shuffleのcosだけで最終判断する。
- ActivityNetのaction accuracyだけで動的情報を主張する。
- AとBのraw normだけを比較する。
- 1 seed・1 checkpoint・1 datasetだけで分離を主張する。

## 未解決の問い

- 動的情報の中心定義を、順序、速度、軌跡、action semanticsのどれに置くか。
- SSv2、Temporal classes、AFD、独自paired setのどれを主評価にするか。
- style/background interventionをどの方法で作り、motion preservationをどう検証するか。
- ΔCLSを最終層だけで扱うか、layer-wise LoRA contributionを実装するか。
- ordered vs shuffled再学習を、現在の6,255 step条件で行うか、短いpilotから始めるか。
- temporal auxiliary objectiveをorder、speed、flow、same-motion contrastのどれにするか。

## 次アクション候補（TODOには追加しない）

1. まず既存G2 checkpointを使うTier 1解析の具体条件を決める。
2. dynamic/staticの主評価datasetを1つ、補助datasetを1つ選ぶ。
3. style/background interventionの生成と検証方法を決める。
4. その後、評価specへ引き継ぐ。
