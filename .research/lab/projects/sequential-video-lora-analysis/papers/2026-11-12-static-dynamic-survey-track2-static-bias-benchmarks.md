# Track 2: 動画モデルの「静的バイアス vs 時間・動的理解」をどう測るか（2014–2026）

調査日: 2026-11-12
対象: 診断手法・ベンチマーク・定量化手法（CVPR/ICCV/ECCV/NeurIPS/ICLR/ICML/WACV/BMVC/TPAMI/IJCV と主要な video-LLM 時間ベンチマーク）
背景: 画像事前学習ViTのQ/VにLoRAを入れ、ActivityNetの4フレームclipでstreaming MoCoを学習している。frame shuffle/reverseをかけても特徴がほぼ変わらない（cos ≈ 0.998）。この状態で「順序・方向・速度・状態変化を学べたか」をどう評価すればよいかを調べた。

## 検証方針

- 採録venueは、arXivのcomment欄、CVF Open Access、NeurIPS/ICLR/PMLR/ACL Anthology/BMVC proceedings、IJCVのDOI、mlanthologyで確認した。確認できなかったものは「未確認」と書いた。
- 図URLは arXiv HTML 版の `<img src>` から絶対URLを組み立て、`curl -sI -m 20` で **HTTP 200 / content-type image/png** を全件確認した（2026-11-12）。PDF由来の図は載せていない。
- 数値は、自分でHTML/PDF本文から確認できたものだけを書いた。アブストラクトや二次情報からの数値はその旨を明記した。
- 「temporal sensitivity と temporal capability の区別」を一つの論文の定義として述べたものは見つからなかった。§2 の整理は、下記論文の評価プロトコルの違いから**本調査で組み立てた枠組み**である。

---

## 1. 全体マップ: 評価手法の6つの系統

| 系統 | 何をするか | 代表論文 | 何がわかるか |
|---|---|---|---|
| A. 入力摂動テスト | テスト時にフレームをshuffle/reverseしたり、1フレームを繰り返したりして、出力や精度の変化を見る | MViT（ICCV21）、Only Time Can Tell、TIME（ICML22）、STEP（ICRA26）、TVBench、VBenchComp | モデルが順序に**反応するか**（sensitivity） |
| B. 時間依存サブセット・データセット | 時間情報がないと解けないクラスや問題だけを集める | Something-Something、Diving48、Only Time Can Tell の Temporal set、Lei et al. の SSv2-label/template | ベンチマークの点数が本当に時間理解を反映しているか |
| C. 外観を消す・矛盾させる OOD テスト | 背景を差し替える、外観をノイズにする、マイム動画を使う | Mimetics、AFD（ECCV22）、SCUBA/SCUFO（ICCV23）、SDN（NeurIPS19） | 静的手がかりに頼らずに認識できるか |
| D. 表現内部の静的/動的定量化 | 静的ペア・動的ペアを作り、中間表現との相互情報量やconceptを測る | Kowal et al.（CVPR22 / TPAMI24）、VTCD（CVPR24）、RESOUND | 層やチャネルごとに、静的情報と動的情報のどちらを持っているか |
| E. 時間的に対になるペア判別 | 開ける/閉める、before/after、逆再生など、見た目が同じで時間だけ違うペアを区別させる | Chirality in Action（NeurIPS25）、Test of Time（CVPR23）、Arrow of Time（CVPR18）、Paxion ActionBench、Vinoground、TempCompass | 時間方向の情報を**取り出せるか**（capability） |
| F. video-LLM 時間ベンチマーク | 多肢選択QAやcaption照合で時間の側面を問う | Perception Test、MVBench、TempCompass、VITATECS、TemporalBench、TOMATO、TVBench | LLM系モデルの時間理解（本研究には主にプロトコルを借りる形で参考になる） |

---

## 2. 「temporal sensitivity」と「temporal capability」を分けて考える（本調査での整理）

- **Sensitivity（順序に反応するか）**: 入力のフレーム順を壊したときに、出力（特徴・ロジット・精度）がどれだけ変わるか。例: MViT論文のshuffle時の精度低下、STEPの逆再生時の精度低下、本研究の cos(f(x), f(shuffle(x)))。
- **Capability（時間情報を取り出せるか）**: 凍結した特徴から、順序・方向・before/after を（線形）分類器で**読み出せるか**。例: Chirality in Action の chiral pair に対する線形分離可能性、Arrow of Time の順再生/逆再生分類、TIME の temporal order prediction を層ごとに見る分析。
- **分ける理由**: cos ≈ 0.998 は「特徴ベクトル全体のうち、順序で変わる成分のノルムが小さい」ことを意味する。しかし、その小さい成分が**一貫した方向**を持っていれば、線形probeで順方向と逆方向を高精度に分けられる可能性は残る。逆に、cosが下がってもそれがノイズなら、時間情報を読み出すことはできない。
  - したがって、「cosが高い＝時間を学んでいない」とまでは言えない。Sensitivity（cos・Δ精度）に加えて、Capability（順方向/逆方向やchiralペアに対する線形probe精度）と、Kowal流の静的/動的相互情報量を併せて報告するのがよい。
- **注意点**: 摂動テストには分布外入力の問題がある。shuffleした動画は自然な動画の分布から外れるため、精度が落ちても「時間を使っていた」のか「OOD入力で壊れた」のかを区別しにくい。Test of Time が synthetic video に対して control task を設けているのはこの問題への対処であり、参考になる。

---

## 3. 論文リスト（31本）

凡例: **[図あり]** は §4 に検証済みの図URLがある論文。

### 3.1 原点・データセット系（appearance と motion の分離、時間依存データ）

#### (1) Two-Stream Convolutional Networks for Action Recognition in Videos [図あり]
- Simonyan & Zisserman. **NeurIPS（当時NIPS）2014**（NeurIPS proceedings 2014 の一覧で確認）
- arXiv: https://arxiv.org/abs/1406.2199
- **評価・分析の方法**: RGBの1フレームを入力とする spatial stream と、積み重ねたoptical flowを入力とする temporal stream を別々に学習し、late fusionで統合する。各streamを単独で評価したときの精度を比べると、「外観だけでどこまで解けるか」と「動きを足すとどれだけ伸びるか」が分かる。これが、その後の appearance/motion の二分法の出発点になった。
- **本研究での使い方と限界**: 「1フレーム（または同一フレーム×4）だけを入れたモデル」の性能を静的ベースラインとして必ず並べる、という考え方をそのまま借りられる。ただし、optical flowは「動き」の一部しか表さないので、dynamicの定義としては狭い。

#### (2) The "something something" video database for learning and evaluating visual common sense
- Goyal et al. **ICCV 2017**（CVF Open Access ICCV2017 一覧に `Goyal_The_Something_Something_ICCV_2017_paper` があることを確認）
- arXiv: https://arxiv.org/abs/1706.04261
- **評価・分析の方法**: 「Pushing something from left to right」のようなテンプレート文をクラスにし、物体カテゴリではなく**動作の様式**（方向・接触・状態変化）でラベル付けした。物体がクラス間で共通なので、外観だけでは解きにくい。後続研究で「時間理解の標準的な試験場」として使われるようになった。
- **本研究での使い方と限界**: SSv2（またはその Temporal subset）で linear probe や few-shot を行い、ActivityNet/Kinetics での結果と比較するのが、最も標準的な「動的情報を学んだか」の外部評価になる。ただし、手持ち撮影のためカメラの動きや手の外観といった別のバイアスが残ること、Kowal TPAMI 版では SSv2 でも静的情報がかなり使われると報告されていることに注意する。

#### (3) RESOUND: Towards Action Recognition without Representation Bias（Diving48）
- Yingwei Li, Yi Li, Nuno Vasconcelos. **ECCV 2018**（CVF/ECCV Open Access PDF、DOI 10.1007/978-3-030-01231-1_32）
- arXiv: なし（CVF: https://openaccess.thecvf.com/content_ECCV_2018/papers/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.pdf ）
- **評価・分析の方法**: データセット D に対して、ある表現 φ（物体・シーン・人物の静的特徴）で学習した最良分類器の精度 M(D,φ) を測り、**representation bias を B(D,φ) = log( M(D,φ) / M_rnd )** と定義する（M_rnd はチャンスレベル。PDF本文の式(9)で確認）。B が大きいほど、静的表現だけで解けてしまう偏ったデータセットということになる。この基準に沿って、静的バイアスの小さい Diving48（48種類の飛込み）を構築した。
- **本研究での使い方と限界**: 「凍結した画像ViT（LoRAなし）の1フレーム特徴＋線形分類器」の精度をチャンスレベルとの対数比で表せば、評価に使うデータセットそのものの静的バイアスを数値で示せる。ただし、Kowal et al.（CVPR22）は Diving48 も実際には動的バイアスが強くないと報告しているので、Diving48 を「純粋な時間ベンチマーク」として扱うのは避けたほうがよい。

#### (4) What Makes a Video a Video: Analyzing Temporal Information in Video Understanding Models and Datasets
- De-An Huang et al. **CVPR 2018**（CVF Open Access、pp. 7366–7375）
- arXiv: なし（CVF: https://openaccess.thecvf.com/content_cvpr_2018/html/Huang_What_Makes_a_CVPR_2018_paper.html ）
- **評価・分析の方法**: 動画から「動き」を取り除いた入力を作り、精度がどれだけ落ちるかで動きの寄与を見積もる。単純なフレーム削除だと分布が変わってしまうため、class-agnostic temporal generator と motion-invariant frame selector を使い、より公平な ablation を設計した。アブストラクトによると、動きの寄与の推定値は UCF101 で 25%→6%、Kinetics で 15%→5% に下がり、従来の見積もりが過大だったことを示した。
- **本研究での使い方と限界**: shuffle で精度が落ちたとしても、その一部は分布外入力の影響である、という注意喚起として引用できる。手法そのもの（生成器を学習する部分）は再現の負担が大きい。

#### (5) Video Time: Properties, Encoders and Evaluation [図あり]
- Ghodrati, Gavves, Snoek. **BMVC 2018**（arXiv comment: "14 pages, BMVC 2018"）
- arXiv: https://arxiv.org/abs/1807.06980
- **評価・分析の方法**: 時間の性質を (a) 非対称性、(b) 連続性、(c) 因果性 の3つに分け、それぞれに対応するタスクとして (a) Arrow of Time 予測（順再生か逆再生か）、(b) Future Frame Selection（続くフレームの選択）、(c) Action Template Classification（SSv2のテンプレート分類）を設計した。C3D や LSTM などのエンコーダを、この3タスクで比較する。
- **本研究での使い方と限界**: 4フレームclipでも (a) の順/逆分類と (b) の続きフレーム選択（例: clipの次に来るフレームを候補から選ぶ）は凍結特徴の上で簡単に実装でき、「時間の性質ごとのcapability」を報告する枠組みになる。ただし、Arrow of Time は撮影上の手がかり（重力、カメラワーク）で解けてしまうことがある（(6) を参照）。

#### (6) Learning and Using the Arrow of Time
- Donglai Wei, Joseph Lim, Andrew Zisserman, William Freeman. **CVPR 2018**（CVF Open Access、pp. 8052–8060）
- arXiv: 未確認（arXiv API のタイトル検索ではヒットしなかった。CVF: https://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Learning_and_Using_CVPR_2018_paper.html ）
- **評価・分析の方法**: 動画が順再生か逆再生かを当てるタスクを大規模に学習・評価した。そのうえで、映像編集の慣習などの「人工的な手がかり」が学習に混入する影響を調べ、それを取り除いたデータセットも用意した。
- **本研究での使い方と限界**: 「順/逆の2値分類を凍結特徴でlinear probeする」ことは、本研究の reverse テストを capability 評価に格上げする最も簡単な方法である。ただし、ActivityNet にはテロップやカメラワークといった撮影手がかりがあるため、それで解けていないかを確認する必要がある（例: カメラが固定のclipだけで評価する）。

#### (7) Temporal Relational Reasoning in Videos（TRN）
- Bolei Zhou et al. **ECCV 2018**（arXiv comment: "camera-ready version for ECCV'18"）
- arXiv: https://arxiv.org/abs/1711.08496
- **評価・分析の方法**: TRN を「順序どおりのフレーム」と「relation module 内でシャッフルしたフレーム」の2条件で学習して比べた。Something-Something では大きな差が出た一方、UCF101 では差がなかった（本文 Figure 6a 周辺で確認）。データセットによって順序が必要かどうかが違うことを示した、初期の代表的な実験である。
- **本研究での使い方と限界**: 「shuffleして学習したモデル」を対照条件に置くという設計（学習時の順序を壊すablation）は、MoCo+LoRA でもそのまま実装できる。正しい順序で学習したモデルと順序を壊して学習したモデルの差を、時間情報の寄与として示せる。ただし、これはテスト時のshuffleではなく学習時のshuffleなので、比較するときに条件を混同しないようにする。

#### (8) Why Can't I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition [図あり]
- Jinwoo Choi et al. **NeurIPS 2019**（arXiv comment）
- arXiv: https://arxiv.org/abs/1912.05534
- **評価・分析の方法**: シーン分類器の擬似ラベルを使った scene adversarial loss と、人物を隠した動画では予測が自信なさげになるべきだとする human mask confusion loss で、シーンバイアスを抑える（SDN）。評価は、ダウンストリーム（UCF101、HMDB51、Diving48 など）への転移性能と、CAM 可視化で行った。
- **本研究での使い方と限界**: 「人物領域をマスクした入力で特徴がどれだけ保たれるか」は、静的な背景にどれだけ依存しているかの診断として、MoCo特徴にも転用できる。ただし、人物検出が必要になり、ActivityNet には人物が小さく写る場面が多い。

#### (9) Mimetics: Towards Understanding Human Actions Out of Context [図あり]
- Weinzaepfel & Rogez. **IJCV 2021**（vol. 129, issue 5、DOI 10.1007/s11263-021-01446-y）
- arXiv: https://arxiv.org/abs/1912.07249
- **評価・分析の方法**: Kinetics の50クラスについて、**物体もシーンもないマイム（パントマイム）動画**を集め、評価専用セットにした。Kinetics で学習したモデルを Mimetics でテストすると、3D CNN の精度は大きく落ち、姿勢ベースのモデルのほうが文脈バイアスに強いことが示された。
- **本研究での使い方と限界**: Kinetics/ActivityNet と重なるクラスについてゼロショット検索や線形probeを行えば、外観に頼らない汎化を確認できる。ただし、クラス数が少なく、ActivityNet のクラスとの対応付けは手作業になる。

#### (10) Only Time Can Tell: Discovering Temporal Data for Temporal Modeling [図あり]
- Sevilla-Lara et al. **WACV 2021**（CVF Open Access WACV2021、pp. 535–544）
- arXiv: https://arxiv.org/abs/1907.08340
- **評価・分析の方法**: **人間**に「元の順序の動画」と「フレームをshuffleした動画」を見せてクラスを答えさせ、shuffleで正答率が大きく落ちるクラスを「temporal classes」として選んだ（本文で確認）。モデルの性能でクラスを選ぶと評価対象のモデルに引きずられるため、人間の知覚を基準にしている。Kinetics と Something-Something から Temporal Dataset と Static Dataset を作り、両者の精度差やテスト時shuffleの影響を比較した。
- **本研究での使い方と限界**: 評価を「Temporal subset」と「Static subset」に分けて報告すると、全体平均では見えない時間理解の差を示せる。本研究の cos 指標も、ActivityNet の全clipではなく、時間が重要なクラスに絞って再計算するとよい。ただし、人間アノテーションに基づくクラス一覧は Kinetics/SSv2 用なので、ActivityNet にそのまま使うことはできない。

#### (11) Multiscale Vision Transformers（MViT）の frame shuffling ablation
- Haoqi Fan et al. **ICCV 2021**（CVF Open Access ICCV2021 に `Fan_Multiscale_Vision_Transformers_ICCV_2021_paper` があることを確認）
- arXiv: https://arxiv.org/abs/2104.11227
- **評価・分析の方法**: テスト時に入力フレームの順序をランダムにshuffleし（学習時はshuffleなし、いずれのモデルも temporal embedding あり）、Kinetics-400 の精度を比べた。MViT-B は 77.2→70.1（−7.1%）と大きく落ちたのに対し、**ViT-B は shuffle にほぼ頑健で、temporal positional embedding が事実上無視されている**と述べている。ImageNet-21K 事前学習の ViT-B でも、shuffle 前後で同じ 79.3% だった（本文 "Frame shuffling." 段落で確認）。
- **本研究での使い方と限界**: 本研究（画像ViT＋時間位置埋め込み＋トークン連結）が cos ≈ 0.998 になるという現象は、この「素朴な ViT は Kinetics で bag-of-frames として振る舞う」という既知の結果と同じ構図で説明できる。最も直接的な先行例として引用できる。ただし、これは教師ありの Kinetics 分類での結果であり、MoCo 自己教師学習や ActivityNet で同じことが起きるかは、自分で確かめる必要がある。

#### (12) Time Is MattEr: Temporal Self-supervision for Video Transformers（TIME） [図あり]
- Sukmin Yun et al. **ICML 2022**（arXiv comment、PMLR v162）
- arXiv: https://arxiv.org/abs/2207.09067
- **評価・分析の方法**: SSv2 で、(a) 元動画と shuffle 動画の精度を Static/Temporal クラス別に比較し、(b) **各 Transformer ブロックの特徴から temporal order を予測する精度**を測った。その結果、shuffle 後も高い精度が保たれること、深いブロックほど時間順序の情報が消えることを示した（Figure 1 のキャプションで確認）。これに対し、shuffle 動画を低確信度に押しやる debias 損失、順序予測損失、トークンの時間的な流れの予測損失を提案した。
- **本研究での使い方と限界**: (b) の「層ごとの順序予測probe」は、LoRA を入れた各層の特徴に対してそのまま実装でき、時間情報が何層目で失われるかを示せる。shuffle を負例にする損失は MoCo の負例設計（shuffle/reverse を hard negative にする）にも直結する。ただし、これは教師ありの fine-tuning が前提の手法である。

#### (13) Is Appearance Free Action Recognition Possible?（AFD） [図あり]
- Filip Ilic, Thomas Pock, Richard Wildes. **ECCV 2022**（ECVA の論文ページ、DOI 10.1007/978-3-031-19772-7_10）
- arXiv: https://arxiv.org/abs/2207.06261
- **評価・分析の方法**: UCF101 の各動画からoptical flowを計算し、ランダムノイズ画像をそのflowに沿ってワープさせることで、**1フレームだけでは何も分からず、動画として見たときだけ動きが分かる**「Random Dot Cinematogram」形式の動画（AFD）を作った。11種類のアーキテクチャを RGB と AFD で比較すると、モデルは AFD で大きく性能が落ちる一方、人間はどちらでもほぼ同等だった。
- **本研究での使い方と限界**: AFD 版の clip で特徴の近傍検索や線形probeを行えば、「外観ゼロでも動きを符号化しているか」の極端なテストになる。ただし、画像事前学習の ViT にとってノイズ動画は強い分布外入力なので、性能が出ないのは当然とも言える。最終評価ではなく、補助的な診断として使うのが現実的である。

#### (14) How Severe is Benchmark-Sensitivity in Video Self-Supervised Learning?（SEVERE）
- Thoker et al. **ECCV 2022**（arXiv comment）
- arXiv: https://arxiv.org/abs/2203.14221
- **評価・分析の方法**: 動画の自己教師あり事前学習モデルを、(i) ドメインシフト、(ii) サンプル数、(iii) 動作の細かさ（FineGym などの fine-grained）、(iv) タスクシフトの4軸で評価し、UCF101/HMDB51 での順位が他の条件では維持されないことを示した。そこから代表的なサブセットを SEVERE-benchmark として提案した。
- **本研究での使い方と限界**: 自己教師あり（MoCo）の動画表現を評価する際、UCF/HMDB だけでは不十分であり、細かい動作（SSv2、FineGym）を含めるべきだ、という根拠として引用できる。ただし、静的/動的を直接分けて測る指標ではない。HTML版に図がなかったため、図は載せていない。後継の SEVERE++（arXiv:2504.05706）は arXiv comment が "Under Review" で、採録は未確認。

#### (15) Mitigating and Evaluating Static Bias of Action Representations in the Background and the Foreground（SCUBA / SCUFO） [図あり]
- Haoxin Li et al. **ICCV 2023**（CVF Open Access ICCV2023、pp. 19911–19923）
- arXiv: https://arxiv.org/abs/2211.12883
- **評価・分析の方法**: 動画の前景（動いている人物）を残して背景を別の画像（Places365、VQGAN-CLIP 生成画像、正弦波の縞模様）に差し替えた **SCUBA** と、SCUBA から1フレームを取り出して時間方向に積み重ねた**動きのない動画 SCUFO** を作った（Figure 3 で確認）。SCUBA で精度が高く、SCUFO で精度が低い（チャンスに近い）ほど、背景にも前景の静的手がかりにも頼っていない良い表現だと判断する。前景の静的手がかりもバイアスになりうることを示した。
- **本研究での使い方と限界**: 「1フレームを4回繰り返した clip」は SCUFO と同じ発想であり、本研究で最も手軽に作れる対照である。元の clip と繰り返し clip の特徴が近い（cos が高い）ほど、静的情報に支配された表現だと言える。背景差し替えには人物セグメンテーションが必要で、ActivityNet ではコストがかかる。

### 3.2 表現の内部を定量化・解釈する系

#### (16) A Deeper Dive Into What Deep Spatiotemporal Networks Encode: Quantifying Static vs. Dynamic Information [図あり]
- Matthew Kowal et al. **CVPR 2022**（arXiv comment）
- arXiv: https://arxiv.org/abs/2206.02846
- **評価・分析の方法**: static を「1フレームから得られる情報」、dynamic を「複数フレームから得られる情報」と定義した。**静的ペア**（同じスタイルで、フレームをshuffleして動きだけ変えた2本）と**動的ペア**（同じ動きで、video stylization により外観だけ変えた2本）を作り、ペア間で中間表現の**相互情報量**（相関に基づく推定）を層ごと・チャネルごとに測る。これにより、各ユニットが static / dynamic / joint / residual のどれに当たるかを分類する（本文 §3.1 と Figure 2 で確認）。ペアの作成には **Stylized ActivityNet** の検証セットを使っており、本研究のデータと一致する。結論として、ほとんどのネットワークは強く静的に偏っていた。また、Diving48 は実は動的バイアスが強くなく、SSv2 のほうが動的能力の評価に向くと報告している。
- **本研究での使い方と限界**: 本研究の「shuffle で cos ≈ 0.998」は、実質的にこの「静的ペア」に対する類似度にあたる。これに「動的ペア（同じclipを style 変換・色変換したもの）」の類似度を加えて、両者の比（static/dynamic バイアス）を LoRA の有無・層ごとに比べれば、そのまま論文化できる形の評価になる。ただし、stylization の生成には計算コストがかかる。簡易版としては、色ジッタやグレースケール化で外観を変える方法もあるが、その場合は「外観をどれだけ変えたか」の妥当性を説明する必要がある。

#### (17) Quantifying and Learning Static vs. Dynamic Information in Deep Spatiotemporal Networks（Kowal TPAMI 拡張版） [図あり]
- Matthew Kowal et al. **TPAMI 2024**（arXiv comment: "TPAMI 2024"）
- arXiv: https://arxiv.org/abs/2211.01783
- **評価・分析の方法**: CVPR22 の手法を、video instance segmentation にも拡張した。さらに「学習中のどの時点で static と dynamic が獲得されるか」を追跡し、バイアスを制御する手段として neuron removal と **StaticDropout** を提案した。StaticDropout は、元動画 v1 と、そのフレームをshuffleした v2 を同じモデルに通し、中間表現を使って静的な情報への依存を落とす手法である（Fig. 12 のキャプションで確認）。
- **本研究での使い方と限界**: 「学習エポックごとの static/dynamic 比の推移」は、streaming MoCo の学習曲線と並べて見せるのに向いている。StaticDropout は、shuffle clip を使った正則化として MoCo 損失に追加する候補になる。ただし、教師ありの設定で検証された手法なので、自己教師ありへの移植は自分で確かめる必要がある。

#### (18) Understanding Video Transformers via Universal Concept Discovery（VTCD） [図あり]
- Matthew Kowal et al. **CVPR 2024 (Highlight)**（arXiv comment）
- arXiv: https://arxiv.org/abs/2401.10831
- **評価・分析の方法**: video transformer の中間特徴を時空間の tubelet に分け、クラスタリングで「concept」を教師なしで見つける。次に、各 concept のトークンをマスクしたときに出力がどれだけ変わるかで、その concept の重要度を測る（Figure 3 の concept masking）。複数のモデルで共通に現れる「Rosetta concept」として、浅い層では時空間の位置、中間層では物体追跡、深い層では細かい動作の概念が現れることを示した。
- **本研究での使い方と限界**: LoRA の有無で各層の concept を比べ、「時間変化を追う concept が増えたか」を定性的に示す分析として使える。ただし、実装が重く、4フレームでは tubelet の時間的な広がりが小さいため、効果が見えにくい可能性がある。

### 3.3 時間的に対になるペアで「時間方向」を問う系

#### (19) Test of Time: Instilling Video-Language Models with a Sense of Time [図あり]
- Piyush Bagad, Makarand Tapaswi, Cees Snoek. **CVPR 2023**（arXiv comment）
- arXiv: https://arxiv.org/abs/2301.02074
- **評価・分析の方法**: 色付きの図形が順番に出現する合成動画（6色×3形状×before/after の 180 ペア）を作り、「赤い丸が黄色い丸の**前に**出る」という caption と、順序を入れ替えた caption のどちらが動画と合うかを判定させる **time-order consistency** を評価した。合成動画が分布外として扱われていないかを確かめる control task も用意している（Figure 2 で確認）。改善手法の TACT では、時間を逆転させた動画とテキストを、同じサンプル内の負例として対照学習に追加した。
- **本研究での使い方と限界**: 「reverse した clip を同一サンプル内の負例にする」という TACT の発想は、MoCo の負例設計にそのまま移せる。control task 付きの合成動画テストは、分布外入力の問題を切り分ける手本になる。ただし、video-language 用のテストなので、画像特徴だけのモデルでは「2本の動画のどちらが順方向か」という形に作り直す必要がある。

#### (20) Chirality in Action: Time-Aware Video Representation Learning by Latent Straightening（CiA / LiFT） [図あり]
- Piyush Bagad & Andrew Zisserman. **NeurIPS 2025**（NeurIPS 2025 proceedings の abstract ページ、Oxford ORA の記録で確認）
- arXiv: https://arxiv.org/abs/2509.08502
- **評価・分析の方法**: 「開ける/閉める」「畳む/広げる」のように、**時間的に逆の関係にあり、1フレームでは区別しにくい行動ペア（chiral pair）**を SSv2（16グループ）、EPIC-Kitchens（66グループ）、Charades（28グループ）から抽出し、Chirality in Action（CiA）として整備した。評価プロトコルは、**凍結した特徴の上で chiral グループごとに線形分類器を学習し、テスト精度をグループ間で平均する**というもの（本文 "Evaluation protocol." で確認）。提案手法の LiFT は、DINOv2 のフレーム特徴系列を static トークンと dynamic トークンに分解し、perceptual straightening の考え方で潜在空間での軌跡を直線化する。
- **本研究での使い方と限界**: 本研究の capability 評価として**最も直接的に使えるプロトコル**である。「凍結画像ViT（LoRAなし）」「LoRA+MoCo」「既存の動画モデル」の CiA 線形probe精度を比べれば、cos ≈ 0.998 の状態でも方向の情報が読み出せるかを判定できる。画像特徴を時間方向に適応させるという LiFT の問題設定も本研究に近い。ただし、SSv2/EPIC/Charades の動画を取得する必要があり、4フレームではサンプリング間隔の選び方が結果に影響する。

#### (21) Paxion: Patching Action Knowledge in Video-Language Foundation Models（ActionBench）
- Zhenhailong Wang et al. **NeurIPS 2023 (spotlight)**（arXiv comment）
- arXiv: https://arxiv.org/abs/2305.10683
- **評価・分析の方法**: ActionBench は2つの probing task からなる。**Action Antonym**（caption の動詞を反意語に置き換えた文と正しい文を区別する）と、**Video Reversal**（元動画と逆再生動画を区別する）である。対照として Object Replacement（物体名の置き換え）を用意し、物体は区別できるのに動作は区別できないことを示した。既存の video-language モデルは、2つの probing task でほぼチャンスレベルだった（アブストラクトで確認）。
- **本研究での使い方と限界**: 「物体の置き換え（静的な対照）」と「逆再生（動的なテスト）」を並べるという2本立ての設計は、画像側だけでも「別の clip との判別」と「逆再生 clip との判別」の比として再現できる。ただし、テキスト側のタスクは本研究の設定には直接は使えない。

#### (22) Order Matters: On Parameter-Efficient Image-to-Video Probing for Recognizing Nearly Symmetric Actions（STEP） [図あり]
- Thinesh Thiyakesan Ponbagavathi & Alina Roitberg. **ICRA 2026**（arXiv comment: "Accepted to ICRA 2026"）
- arXiv: https://arxiv.org/abs/2503.24298
- **評価・分析の方法**: 凍結した画像基盤モデル（DINOv2/CLIP）の上に attentive probe を置く方式は**フレームの並べ替えに対して不変（permutation-invariant）**であり、逆再生しても予測が変わらないことを示した。一方、PEFT（adapter/prompt）や提案手法の STEP（フレームごとの学習可能な位置埋め込み、全フレーム共通の CLS トークン、簡略化した attention）は、テスト時に逆再生すると大きく精度が落ちる（例: STEP は HRI-30 で −44.8%。Table II で確認）。つまり「逆再生での精度低下」を、時間順序を符号化している証拠として使っている。
- **本研究での使い方と限界**: 画像から動画への適応という、本研究と最も近い2025–26年の研究である。「正順・逆順でのテスト精度」の表は、そのまま本研究の評価表の雛形になる。ただし、位置埋め込みがあっても attention がそれを使うとは限らない（MViT 論文の ViT-B の例）。本研究で位置埋め込みを入れているのに変化が小さいことは、「位置埋め込みがあっても、それが使われているとは限らない」という論点になる。

### 3.4 video-language / video-LLM の静的バイアス・時間ベンチマーク

#### (23) Revisiting the "Video" in Video-Language Understanding（ATP） [図あり]
- Shyamal Buch et al. **CVPR 2022 (Oral)**（arXiv comment）
- arXiv: https://arxiv.org/abs/2206.01720
- **評価・分析の方法**: 凍結した画像-言語エンコーダの上で、**順序を使わずに1フレームだけを選んで答える atemporal probe（ATP）**を学習し、既存の video QA や検索ベンチマークがどこまで「画像だけ」で解けるかを上限として測った。ATP で解けてしまう問題を除くと、本当に動画の理解が必要な部分集合が得られる（NExT-QA の分析）。
- **本研究での使い方と限界**: 「最良の1フレームを選べば解けるか」という強い静的ベースラインの考え方は、本研究で静的ベースラインを作るときの基準になる（平均プーリングより強い基準を置く）。ただし、テキストQAのない設定では、「最良フレーム選択器」を検索タスク用に作り直す必要がある。

#### (24) Revealing Single Frame Bias for Video-and-Language Learning（Singularity） [図あり]
- Jie Lei, Tamara Berg, Mohit Bansal. **ACL 2023**（long paper、pp. 487–507、ACL Anthology 2023.acl-long.29）
- arXiv: https://arxiv.org/abs/2206.03428
- **評価・分析の方法**: 学習時に1フレームしか使わないモデルでも、MSRVTT などの検索・QA で既存の動画モデルと同等の性能が出ることを示し、既存データセットの static appearance bias を明らかにした。代わりに、SSv2 の **label** と **template** をテキストクエリにした2つの検索タスク（SSv2-label / SSv2-template）を提案した（Figure 2 で確認）。template は物体名を伏せたテキストなので、動作そのものの理解が必要になる。
- **本研究での使い方と限界**: SSv2-template 検索は「物体名なしで動作の様式を当てる」タスクとして、MoCo 特徴の評価（テキストがない場合は template ラベルによる kNN 分類）に転用しやすい。ただし、テキストエンコーダとの組み合わせを前提とした指標である。

#### (25) Perception Test: A Diagnostic Benchmark for Multimodal Video Models
- Viorica Pătrăucean et al. **NeurIPS 2023 Datasets and Benchmarks Track**（arXiv comment）
- arXiv: https://arxiv.org/abs/2305.13786
- **評価・分析の方法**: 参加者が台本に沿って撮影した実写動画を使い、記憶・抽象化・物理・意味という「スキル」と、記述・説明・予測・反実仮想という「推論の型」を、多肢選択 video QA、物体/点の追跡、時間的な行動・音の区間検出などで診断する。動画は11.6k本、平均23秒で、知覚的に興味深い状況が写るように設計されている（アブストラクトで確認）。
- **本研究での使い方と限界**: 点追跡や行動区間検出のタスクは、凍結特徴の評価にも使える。ただし、ベンチマーク全体は大きく、4フレーム clip のモデルには時間スケールが合わないことが多い。

#### (26) MVBench: A Comprehensive Multi-modal Video Understanding Benchmark
- Kunchang Li et al. **CVPR 2024 (Highlight)**（arXiv comment）
- arXiv: https://arxiv.org/abs/2311.17005
- **評価・分析の方法**: 画像向けの静的なタスクを時間方向に拡張する（static-to-dynamic）方針で、行動の順序、移動方向、状態変化、反実仮想などを含む20の時間タスクを、既存データセットから自動で多肢選択QAに変換した。
- **本研究での使い方と限界**: タスク分類（順序・方向・状態変化など）は、本研究の評価軸を整理するときの語彙として参照できる。ただし、TVBench（下記）が、MVBench には1フレームやテキストだけで解ける問題が多いと批判している点に注意する。

#### (27) TempCompass: Do Video LLMs Really Understand Videos? [図あり]
- Yuanxin Liu et al. **Findings of ACL 2024**（ACL Anthology 2024.findings-acl.517）
- arXiv: https://arxiv.org/abs/2403.00476
- **評価・分析の方法**: 時間の側面（action、speed、direction、attribute change、event order）ごとに、**静的な内容が同じで特定の時間的な側面だけが異なる「conflicting videos」**を作った（Figure 3、本文で確認）。作り方は3通りある。direction と attribute change は「元動画と逆再生動画」のペア、speed は「加速/減速した動画と元動画を上下に並べた（spatial concatenation）」3つ組、event order は「2つの動画を時間方向につなげる順序を入れ替えた（temporal concatenation）」ペアである。これにより、1フレームのバイアスや言語の事前知識では解けないようにしている。
- **本研究での使い方と限界**: speed（倍速 vs 等速）と direction（順再生 vs 逆再生）の conflicting pair を自前の ActivityNet clip で作り、それぞれを判別できるか線形probeで測る、という形に移植しやすい。4フレームでも、フレーム間隔（stride）を変えることで速度条件を作れる。ただし、逆再生の動画は不自然な物理を含むため、分布外入力の注意は同じく必要である。

#### (28) VITATECS: A Diagnostic Dataset for Temporal Concept Understanding of Video-Language Models [図あり]
- Shicheng Li et al. **ECCV 2024**（arXiv comment）
- arXiv: https://arxiv.org/abs/2311.17404
- **評価・分析の方法**: 正しい caption と、時間的な側面（方向、強度、順序、局所性、合成性、型）を1つだけ変えた**反事実的な caption** のペアを、NLI モデルと人手を組み合わせたパイプラインで作った。モデルがどちらの caption を動画に合うと判定するかを測る。
- **本研究での使い方と限界**: 時間概念を6つの側面に分ける整理は、本研究で「順序・方向・速度・状態変化」の評価軸を定義するときに引用できる。ただし、テキスト側の反事実なので、画像のみのモデルでは直接評価できない。

#### (29) Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos
- Jianrui Zhang, Mu Cai, Yong Jae Lee. **venue 未確認**（arXiv のみ確認。v1、2024-10）
- arXiv: https://arxiv.org/abs/2410.02763
- **評価・分析の方法**: Winoground を動画に拡張したもの。同じ単語を違う順序で並べた2つの caption と、それに対応する2本の短い自然動画を組にした **temporal counterfactual** を作った。**text score**（動画に対して正しい caption を選ぶ）、**video score**（caption に対して正しい動画を選ぶ）、**group score**（両方を同時に正解する）で評価し、多くのモデルは video score でチャンスレベル程度だった（本文で確認）。
- **本研究での使い方と限界**: 「video score（2本の動画のどちらがクエリに合うか）」は、テキストなしでも「クエリ clip に対して、順方向の clip と逆方向の clip のどちらが近いか」という形で再現できる。ただし、採録 venue は確認できなかった。

#### (30) TemporalBench: Benchmarking Fine-grained Temporal Understanding for Multimodal Video Models
- Mu Cai et al. **NeurIPS 2024 Workshop（Video-Language Models）として mlanthology に記録あり。本会議等の採録は未確認**
- arXiv: https://arxiv.org/abs/2410.10818
- **評価・分析の方法**: 約2Kの人手による詳細な時間的 caption から、約10Kの QA を作った。多肢選択QAでは、LLM が「他の選択肢と最も共通部分の多い選択肢」を手がかりに正解できてしまうという落とし穴を指摘し、**Multiple Binary Accuracy（MBA）**（正解と各負例を1対1の2択に分け、すべてに正解した場合だけ正解とする）を提案した（本文で確認）。
- **本研究での使い方と限界**: MBA の「全ての2択に正解して初めて正解とする」という考え方は、本研究で複数の摂動（shuffle、reverse、1フレームの繰り返し）を同時に区別できるかを評価する指標として使える。ただし、採録状況には注意する。

#### (31) TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models
- Ziyao Shangguan et al. **ICLR 2025**（ICLR 2025 proceedings のページで確認）
- arXiv: https://arxiv.org/abs/2410.23266
- **評価・分析の方法**: 既存のベンチマークの多くが、1フレーム・少数フレーム・順序の崩れたフレームで解けてしまうことを指摘した。そのうえで、action count、direction、rotation、shape & trend、velocity & frequency、visual cues の6タスク、1,484問を人手で作成した。動画の多くは自分たちで撮影・生成したものである。最良モデルでも人間との差は 57.3% だった。
- **本研究での使い方と限界**: 方向・回転・速度・頻度という分類は、「dynamic とは何か」を具体的に定義するときに役立つ。ただし、LLM 向けの多肢選択QAなので、画像特徴のモデルではタスクを作り直す必要がある。

#### (32) Lost in Time: A New Temporal Benchmark for VideoLLMs（TVBench） [図あり]
- Daniel Cores et al. **BMVC 2025**（BMVC 2025 proceedings のページ）
- arXiv: https://arxiv.org/abs/2410.07752
- **評価・分析の方法**: 既存のベンチマーク（MVBench など）では、**入力フレームを shuffle しても SOTA モデルの精度がほとんど落ちない（time-invariant）**ことを示した（Figure 1）。原因を、1フレームで解ける問題（spatial bias）、問題文が情報を与えすぎていること（textual bias）、世界知識だけで解けることに分けて分析した。TVBench では、時間的に紛らわしい選択肢と情報を与えすぎない問題文を設計し、「shuffle するとチャンスレベルまで落ちる」ことをベンチマークの妥当性の条件にしている。
- **本研究での使い方と限界**: 「shuffle すればチャンスまで落ちるべき」という基準は、本研究で**評価タスクそのものの妥当性を確かめる手順**として使える。例えば、自前の順序判別タスクで、凍結した画像ViTに shuffle 入力を与えたときにチャンスになることを先に確認しておく。ただし、評価対象は LLM 系である。

#### (33) Breaking Down Video LLM Benchmarks: Knowledge, Spatial Perception, or True Temporal Understanding?（VBenchComp） [図あり]
- Bo Feng et al. **NeurIPS 2025 Workshop（"Evaluating the Evolving LLM Lifecycle"）のポスター**（neurips.cc virtual のページで確認。本会議ではない）
- arXiv: https://arxiv.org/abs/2505.14321
- **評価・分析の方法**: ベンチマークの各問題を、動画なしで解ける **LLM-Answerable**、フレームを2回 shuffle しても2つのモデルが正解する **Semantic**、正しい順序が必要な **Temporal**、その他、に自動で分類するパイプラインを作った（Figure 4）。shuffle しても多くのモデルの総合スコアが変わらないことを示した（Figure 3）。
- **本研究での使い方と限界**: 評価サンプルを「shuffle しても解ける clip」と「shuffle すると解けなくなる clip」に分け、後者だけで指標を報告するという手順は、ActivityNet での評価にそのまま使える。ただし、ワークショップ論文である。

#### （補足）How Do Video Foundation Models Encode Intuitive Physics? Probing Across Pretraining Paradigms
- Samuele Punzo et al. **NeurIPS 2026 Workshop on World Models in Physical AI に採録**（arXiv comment）
- arXiv: https://arxiv.org/abs/2606.09646
- **評価・分析の方法**: 凍結した動画基盤モデル（V-JEPA、VideoMAE、LTX-Video）の層ごとの特徴に、linear / MLP / temporal attentive probe をかけて直観物理を予測させる。対照条件として、**frame-shuffled control**（フレーム順をランダムに並べ替える）と **single-frame control**（1フレームを動画長だけ繰り返す）を入力側で適用し、backbone と probe は凍結したまま性能の変化を見る（付録 A.6 で確認）。
- **本研究での使い方と限界**: 「shuffle 対照」と「1フレーム繰り返し対照」の2本立てを probe 評価に組み込むという、2026年時点での標準的な書き方の例として参照できる。ただし、ワークショップ論文で、対象はタスクも直観物理に限られる。

---

## 4. 検証済みの図（arXiv HTML の直リンク、全件 HTTP 200 / image/png を確認）

| # | 論文 | 図 | URL | 図の内容と読み方 |
|---|---|---|---|---|
| 1 | Two-Stream（NeurIPS14） | Fig. 1 | https://arxiv.org/html/1406.2199v2/arch.png | 1枚のRGBフレームを入れる spatial stream と、複数フレームのoptical flowを積み重ねて入れる temporal stream が並列に走り、最後に class score を fusion する構成図。「外観」と「動き」を別の経路で扱うという二分法の原点として示す。 |
| 2 | Only Time Can Tell（WACV21） | Fig. 1 | https://arxiv.org/html/1907.08340v2/sneezing_vs_crying.png | 「あくび・くしゃみ・泣く」の静止フレームを並べ、1枚では区別できないことを示す導入図。フレームの shuffle で時間情報を除くと、時間が本当に必要な行動が浮かび上がる、という発想を一目で伝えられる。 |
| 3 | Only Time Can Tell | Fig. 5(a) | https://arxiv.org/html/1907.08340v2/Acc_Temp_Static.png | 各ネットワークの Temporal Dataset と Static Dataset での絶対精度を比べた棒グラフ。R2D は両者でほぼ同じだが、時間方向の畳み込みを持つ R(2+1)D は Temporal 側で相対的に強い。「時間をモデル化できるほど Temporal set で差がつく」ことの読み取り方を示す。 |
| 4 | Only Time Can Tell | Fig. 6(b) | https://arxiv.org/html/1907.08340v2/shuffle.png | テスト時に shuffle したときの精度低下を、Temporal クラスで学習したモデルと Static クラスで学習したモデルで比較した図。Temporal 側で学習したモデルのほうが shuffle に敏感になる（特徴が時間変化に反応するようになる）。 |
| 5 | Kowal CVPR22 | Fig. 1 | https://arxiv.org/html/2206.02846v1/teaser.png | 任意の中間表現について、static（赤）と dynamic（青）への偏りを層ごと・ユニットごとに定量化するという手法の概要図。 |
| 6 | Kowal CVPR22 | Fig. 2 | https://arxiv.org/html/2206.02846v1/main_method2.png | (1) stylization と frame shuffle（VOS では flow jitter）で静的ペアと動的ペアを作り、(2) 凍結モデルの中間表現でペア間の相互情報量を計算する、という2段階の手順図。本研究の「shuffle の cos」がどの位置づけになるかを説明するのに最適。 |
| 7 | Kowal TPAMI24 | Fig. 12 | https://arxiv.org/html/2211.01783v2/Staticdropout_v4.png | StaticDropout の概要。元動画 v1 とそのフレームを shuffle した v2 を同じモデルに通し、中間表現を比べて静的な情報への依存を落とす。shuffle clip を学習信号として使う一例。 |
| 8 | VTCD（CVPR24） | Fig. 2 | https://arxiv.org/html/2401.10831v3/MainMethod8.png | 動画の特徴を時空間の tubelet に分け、クラスタリングで concept を見つけ、マスクによって重要度を測る VTCD の流れ図。 |
| 9 | AFD（ECCV22） | Fig. 1 | https://arxiv.org/html/2207.06261v1/teaser1.png | 3枚1組で、RGBフレーム、そのoptical flow、外観を消した AFD フレーム（ノイズ）を並べた図。AFD の1枚だけでは何も分からないが、動画として再生すると動きが見える、という設計思想を示す。 |
| 10 | AFD | Fig. 2 | https://arxiv.org/html/2207.06261v1/flowchart.png | AFD の作り方: 1) ノイズ R0 を初期化、2) フレーム対ごとにoptical flowを計算、3) flow で R0 をワープ、を繰り返す。 |
| 11 | SCUBA/SCUFO（ICCV23） | Fig. 1 | https://arxiv.org/html/2211.12883v3/bg_fg_bias.png | (a) 通常の HMDB51 テスト、(b) 背景を縞模様に置き換えた SCUBA、(c) SCUBA にランダムな静的前景を挿入した矛盾動画、の3条件で予測がどう変わるか。背景だけでなく前景の静的手がかりもバイアスになることを示す。 |
| 12 | SCUBA/SCUFO | Fig. 3 | https://arxiv.org/html/2211.12883v3/illu_dataset_construction.png | OOD ベンチマークの作り方。前景の行動を残して背景を差し替えたものが SCUBA、その1フレームを積み重ねて動きをなくしたものが SCUFO。本研究の「1フレームを4回繰り返す対照」の根拠になる図。 |
| 13 | Test of Time（CVPR23） | Fig. 2 | https://arxiv.org/html/2301.02074v2/syn-task-v4.png | before/after 関係を持つ合成動画とテキストのペアで time-order consistency を評価するタスクの概要と、合成動画が分布外扱いされていないかを確かめる control task。 |
| 14 | Test of Time | Fig. 3 | https://arxiv.org/html/2301.02074v2/method-v8.png | TACT の概要。通常のバッチ内負例に加えて、同じサンプルの時間反転版（動画・テキスト）を追加の負例にする。MoCo で reverse clip を hard negative にする設計の参考図。 |
| 15 | Chirality in Action（NeurIPS25） | Fig. 1 | https://arxiv.org/html/2509.08502v2/teaser-v4.png | (a) chiral action（時間的に逆の行動ペア）の考え方と、SSv2/EPIC/Charades から作ったメタデータセット、(b) LiFT が既存の動画・画像モデルより chiral 判別に強いこと、(c) SSv2 での linear probe の結果。 |
| 16 | Chirality in Action | Fig. 4 | https://arxiv.org/html/2509.08502v2/examples-dataset-v4.png | 3つのデータセットからの chiral ペアの例。各ペアを区別するには、空間的な文脈を割り引いて「何がどう変化したか」を見る必要がある。 |
| 17 | TVBench（BMVC25） | Fig. 1 | https://arxiv.org/html/2410.07752v3/figs/ims/fig1_rel.png | 既存のベンチマークでは、入力を shuffle しても SOTA モデルの精度がほとんど落ちない（time-invariant）のに対し、TVBench では shuffle するとチャンスレベルまで落ちることを示す図。「良い時間ベンチマークは shuffle で壊れるべき」という基準を示す。 |
| 18 | VBenchComp（NeurIPS25 WS） | Fig. 3 | https://arxiv.org/html/2505.14321v1/Figure/gpt_4o.png | GPT-4o などのスコアが、フレームを shuffle してもほぼ変わらないことを示すグラフ（Fig. 3 は複数モデルのパネルからなり、これはその1枚）。 |
| 19 | VBenchComp | Fig. 4 | https://arxiv.org/html/2505.14321v1/Figure/pipeline_v3.png | 問題を LLM-Answerable / Semantic（shuffle しても解ける）/ Temporal / Other に振り分けるパイプライン。評価サンプルを時間依存性で分けるという手順の図。 |
| 20 | TempCompass（ACL Findings 24） | Fig. 3 | https://arxiv.org/html/2403.00476v3/conflicting_videos.png | 時間の側面ごとの conflicting video ペア・3つ組の例。逆再生（direction・attribute change）、速度を変えた動画の空間連結（speed）、時間方向の連結順の入れ替え（event order）によって、静的な内容が同じまま時間の側面だけが違う動画を作る。 |
| 21 | Lei et al.（ACL23） | Fig. 2 | https://arxiv.org/html/2206.03428v1/ssv2_example.png | SSv2 の例。時間順に並べた3フレームと、template（物体名を伏せた文）・label の注釈。これを基にした SSv2-template / SSv2-label 検索タスク。 |
| 22 | Mimetics（IJCV21） | Fig. 1 | https://arxiv.org/html/1912.07249v3/splashijcv.png | 文脈が部分的な例（上）、ない例（中、マイム）、誤解を招く例（下）。シーンや物体という文脈に頼った認識の弱点を示す。 |
| 23 | Video Time（BMVC18） | Fig. 1 | https://arxiv.org/html/1807.06980v1/tasks_cc.png | 時間の3つの性質（非対称性・連続性・因果性）と、それに対応する3つの評価タスクの図。 |
| 24 | ATP（CVPR22） | Fig. 3 | https://arxiv.org/html/2206.01720v1/figures/fig3.png | 凍結した画像-言語エンコーダの上で、順序を使わずに1フレームを選ぶ atemporal probe の構成。「画像だけでどこまで解けるか」という強い静的ベースライン。 |
| 25 | STEP（ICRA26） | Fig. 1 | https://arxiv.org/html/2503.24298v2/ICRA_teaser_v4.png | 凍結した画像基盤モデルの上の probe を、フレーム順序に敏感にして「拾う/置く」のようなほぼ対称な行動を区別できるようにする、という概要。PEFT との比較も含む。 |
| 26 | TIME（ICML22） | Fig. 2 | https://arxiv.org/html/2207.09067v1/0615_fig2.png | 3種類の時間的自己教師: (a) shuffle 動画を使った空間バイアスの除去、(b) 深いブロックでも時間順序を保つための順序予測、(c) フレーム間のトークンの流れの予測。 |
| 27 | SDN（NeurIPS19） | Fig. 3 | https://arxiv.org/html/1912.05534v1/overview_v8.png | scene adversarial loss と human mask confusion loss でシーンバイアスを抑えた事前学習の全体像。 |
| 28 | VITATECS（ECCV24） | Fig. 2 | https://arxiv.org/html/2311.17404v2/main/figures/example.png | 6つの時間的な側面ごとに、正しい caption と1点だけ変えた反事実 caption のペアの例。 |

（SEVERE は arXiv HTML 版に図が含まれていなかったため、図はない。RESOUND、Huang CVPR18、Arrow of Time は arXiv HTML 版が確認できず、図は載せていない。）

---

## 5. 本研究（LoRA + streaming MoCo、4フレーム、ActivityNet）への評価レシピ案

文献からそのまま組み立てられる評価を、手軽なものから順に挙げる。

1. **Sensitivity の分解（既存の cos 指標の拡張）** — MViT 論文、Kowal、SCUFO、Punzo et al.
   - cos(f(x), f(T(x))) を T ∈ {shuffle, reverse, 1フレーム×4の繰り返し, 外観変換（色ジッタ・stylization）} で測る。「shuffle/reverse の cos（静的ペアの類似度）」と「外観変換の cos（動的ペアの類似度）」を並べ、Kowal 流に static/dynamic のどちらに偏っているかとして報告する。
   - これを LoRA なし（凍結画像ViT）と LoRA あり、さらに層ごとに比べる。
   - 特徴ベクトル全体の cos だけでなく、順序で変わる差分 f(x) − f(reverse(x)) のノルムを、clip 間のばらつきと比べた相対値でも示す。
2. **Capability（線形probe）** — Arrow of Time、Video Time、Chirality in Action、STEP
   - 凍結特徴を入力にして、「順方向か逆方向か」「正しい順序か shuffle か」を当てる線形分類器を、clip を分割した train/test で学習・評価する。cos ≈ 0.998 でも高精度なら「時間情報は弱いが読み出せる」、チャンス程度なら「時間情報が表現に入っていない」と判定できる。
   - 外部ベンチマークとしては、CiA（chiral pair の線形分離可能性）と SSv2（Temporal subset、template ラベル）を使う。
3. **評価タスクの妥当性チェック** — TVBench、Only Time Can Tell、VBenchComp
   - 凍結した画像ViTを1フレームまたは平均プーリングで使った場合に、そのタスクがチャンス近くまで落ちることを先に確認する（静的に解けるタスクでは時間理解を測れないため）。
   - ActivityNet の評価 clip を、「shuffle しても判別できる clip」と「shuffle すると判別できなくなる clip」に分けて報告する。
4. **分布外入力の問題への対処** — Huang CVPR18、Test of Time の control task
   - shuffle 入力で性能が落ちたときに、それが時間を使っていた証拠なのか、分布外入力で壊れただけなのかを区別するため、「順序はそのままで同程度の画質劣化を加えた clip」などの対照を置く。
5. **学習側への示唆**（評価ではないが、文献に根拠がある） — TACT、TIME、StaticDropout
   - reverse/shuffle した clip を MoCo の hard negative にする、または順序予測の補助損失を足す、といった改善策は複数の論文で有効性が報告されている。評価で「時間情報がない」と分かった場合の次の一手になる。

---

## 6. 注意点・未確認事項

- **venue 未確認**: Vinoground（arXiv のみ）、Arrow of Time の arXiv 版（CVPR 2018 であることは CVF で確認済み）。
- **ワークショップ論文**: TemporalBench（mlanthology では NeurIPS 2024 Workshop として記録）、VBenchComp（NeurIPS 2025 Workshop のポスター）、Punzo et al.（NeurIPS 2026 Workshop、arXiv comment による）。本会議論文と同じ重みで引用しないこと。
- **arXiv 版がない論文**: RESOUND（ECCV18）、Huang et al.（CVPR18）。CVF のリンクを記載した。
- **数値の出典**: Huang CVPR18 の 25%→6% / 15%→5% はアブストラクトから引用した（本文は未確認）。その他の数値（MViT の −7.1% と 79.3%、STEP の −44.8%、TOMATO の 57.3% など）は本文・proceedings ページで確認した。
- **2024–2026年の「画像→動画適応モデルの時間情報probing」**: STEP（ICRA 2026）、LiFT/CiA（NeurIPS 2025）、Punzo et al.（2026 WS）以外に、LoRA や adapter で適応した画像ViTの時間的な感度を正面から分析した論文は、今回の検索では見つからなかった。ST-Adapter（NeurIPS 2022）や AIM（ICLR 2023）は、手法論文としての SSv2 評価や attention の可視化はあるが、shuffle/reverse による診断は確認できなかった。この部分が、本研究の新規性を主張できる余地になりうる。
- **「temporal sensitivity vs capability」の区別**は本調査で行った整理であり、特定の論文が定義した用語ではない。資料で使う場合は「本研究での整理」と明記する。
