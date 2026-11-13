# Track 3: 時間・動き情報を狙う動画自己教師あり学習 / scene-motion分離 / 画像→動画PEFT / ストリーミングSSL
## 「時間・動きを学習できたこと」を各論文がどう評価しているか

調査日: 2026-11-12

### このメモの前提（研究文脈）

- 対象の設定: 画像で事前学習したViTのattention Q/VにLoRAを入れ、ActivityNetの4フレームクリップ（フレームをトークンとして連結し、時間位置埋め込みを付与）を、ストリーミング形式のMoCo対照学習で学習している。正例ペアは「同じクリップに2種類のaugmentationをかけたもの」。
- 困っていること: 学習後の特徴が、フレームのシャッフル・逆再生でほとんど変わらない。つまり、学習したのが動き（順序・方向・速さ）なのか、見た目（静的appearance）だけなのかを評価したい。

### 検証方法と凡例

- 採録先（venue）は次のいずれかで確認した。確認できなかったものは「未確認」と書いた。
  - arXiv APIの `arxiv:comment` 欄（著者が書いた採録情報）
  - CVF Open Access の掲載ページ（CVPR / ICCV / WACV）
  - ECVA の論文一覧（ECCV）
- arXiv ID は arXiv API でタイトルと照合した。arXiv版が見つからない論文は「arXiv: 見つからず」と書いた。
- 図のURLは arXiv HTML 版の `<img src>` から組み立てた。2026-11-12 に `curl -sI` で **HTTP 200 かつ `content-type: image/png`** であることを全件確認した。PDFからの切り出しは使っていない。
- 図は著者の著作物なので、MTG資料に使うときは出典（論文名・図番号）を必ず書くこと。
- 「評価」の説明は、アブストラクトと arXiv HTML 本文で確認できた範囲に限った。数値は本文で確認できたものだけを載せた。

---

## 0. 先に結論: 評価方法の型と、この研究での使いどころ

各論文の評価方法は、大きく次の8つの型に分けられる。

| 型 | 中身 | 代表論文 | この研究での使いどころ |
|---|---|---|---|
| ① 下流の行動認識（fine-tune / linear probe） | UCF101 / HMDB51 / Kinetics での精度 | ほぼ全論文 | 見た目だけでも高い精度が出るため、**動きを学んだ証拠としては弱い** |
| ② 時間が重要なデータセットでの評価 | SSv2、Diving48、FineGym、Sevilla-Laraらが選んだ時間クラス | TCLR, TE, FAME, BE, AIM, EVL, Tubelet | K400 と SSv2/Diving48 の **差（gap）** を見る。EVL では時間モジュールの効果が K400 で +0.5%未満、SSv2 で +13.8% |
| ③ 時間を壊した入力への感度 | シャッフル・逆再生・静止画をコピーした「static video」を入れたときの精度・確信度・特徴の変化 | Time Is MattEr, Only Time Can Tell, DSM, BE, MME | **今回の「特徴がほぼ変わらない」という観察は、まさにこの型の否定的な結果**。クラス別・確信度差・層別に分解すると解釈しやすい |
| ④ 時間タスクのprobe | 凍結した特徴の上に小さな分類器を置き、順序・ずれ量・前後関係・速度を当てさせる | Time Is MattEr（層ごとの順序予測）, Jenni+ 2020（同期・前後判定）, TE | **LoRA前と後で probe 精度を比べるのが最も直接的**。ブロックごとに測れる |
| ⑤ 近傍検索（retrieval） | クエリ動画に近い動画を R@k で評価 | VCOP, Pace, PRP, TE, TCLR, SpeedNet | 「同じ見た目・違う動き」を検索できるかを測れるよう、ペアを自分で設計すると有効 |
| ⑥ 可視化 | CAM / Grad-CAM で動いている領域を見ているか | DSM, BE, FAME, MCL, Tubelet, DualPath | 補助的な証拠。定量評価の代わりにはならない |
| ⑦ 表現の内部統計 | 時間ステップ間の特徴のコサイン類似度（TCLR）、時間カーネル重みの分布（FAME） | TCLR, FAME | 4フレームのトークン特徴について、**フレーム間類似度行列**を学習前後で比べる |
| ⑧ 「時間を壊したデータで学習」する対照実験 | シャッフルした動画で事前学習しても下流性能が変わらなければ、時間を使っていない | MME（VideoMAEについて） | 同じ設定で「シャッフルしたクリップで学習した LoRA」を作り、評価差を見る |

**構造上の注意（この研究向け）**

正例ペアが「同じクリップの2つのaugmentation」で、どちらも同じフレーム順のままなら、MoCo の目的関数に順序を区別する圧力は働かない。シャッフル版は正例でも負例でもないからである。このため、特徴がシャッフルで変わらないのは、目的関数から予想される結果である可能性が高い。

文献では、次のように**時間を壊したものを明示的に負例にする、または区別させる**設計が使われている。

- DSM: 時間を局所的に乱したクリップを負例にする
- Time Is MattEr: シャッフル入力に低い確信度を出させる
- TE: 時間変換に対して等変（equivariant）にする
- TCLR: 同じ動画の別時刻クリップを区別させる
- Jenni+ 2020: 時間変換を分類させる

評価を設計するときは、この「学習目的が何に対して不変・敏感になるよう作られているか」とセットで書くと、先生への説明が通りやすい。

---

## A. 時間に関するpretext task（2015–2020）

### A1. Shuffle and Learn: Unsupervised Learning using Temporal Order Verification
- 著者: Ishan Misra et al.
- 採録: **ECCV 2016**（arXiv comment「Accepted at ECCV 2016」）
- arXiv: https://arxiv.org/abs/1603.08561
- **方法と評価**
  - 3フレームの組が正しい時間順か、シャッフルされているかを当てる二値分類で、CNN を学習する。
  - 学習用の組は、オプティカルフローで選んだ「動きの大きい区間」からサンプルする。動きのない区間ではシャッフルしても見分けられないからである。
  - 評価は、UCF101 / HMDB51 の行動認識への事前学習効果、fc7特徴での最近傍検索、FLIC / MPII での姿勢推定。姿勢推定を使うことで、「時間で変化する情報（人の姿勢）を捉えている」ことを示した。
- **この研究での再利用と限界**
  - ActivityNet の4フレームクリップは動きの小さい区間を多く含む可能性がある。**フロー量や差分画像の大きさでクリップを層別して、シャッフル感度を測る**と、「動きがある区間でも鈍いのか」を切り分けられる。
  - 限界: 3フレーム・CNN時代の設定なので、数値は直接比較できない。
- **図**
  - Fig. 2: 動きの大きい窓から正例（正しい順）と負例（順序を崩した組）を作り、重みを共有した3本のネットワークで判定する構成図。
  - ![Shuffle and Learn Fig2](https://arxiv.org/html/1603.08561v2/network_arch_sampling.png)

### A2. Self-Supervised Video Representation Learning With Odd-One-Out Networks（O3N）
- 著者: Basura Fernando et al.
- 採録: **CVPR 2017**（arXiv comment）
- arXiv: https://arxiv.org/abs/1611.06646
- **方法と評価**
  - 複数の部分系列のうち、1本だけ時間順を崩した「仲間外れ」を当てる課題で学習する。
  - 各部分系列は、dynamic image やフレーム差分の和などでまとめてから入力する。
  - 評価は UCF101 / HMDB51 の行動分類（UCF101のみで事前学習して 60.3%）。サンプリング戦略の違いを比較するアブレーションもある。
- **この研究での再利用と限界**
  - 「正順のクリップ2本 + 崩したクリップ1本から崩れた方を選ぶ」は、凍結特徴の上で動く **zero-shot の順序判別テスト**にできる。例えば、特徴距離が最も離れたものを仲間外れとみなせばよい。
  - 限界: フレーム差分などで入力段階から動きを強調しており、生のRGBトークンを使う ViT とは前提が異なる。

### A3. Unsupervised Representation Learning by Sorting Sequences（OPN）
- 著者: Hsin-Ying Lee et al.
- 採録: **ICCV 2017**（arXiv comment）
- arXiv: https://arxiv.org/abs/1708.01246
- **方法と評価**
  - シャッフルしたフレーム列の元の順序を当てる（並べ替える）課題で学習する。全フレームペアの特徴をまとめて順序を予測する。
  - 評価は行動認識（UCF101 / HMDB51）に加えて、画像分類と物体検出（PASCAL VOC）への転移。
- **この研究での再利用と限界**
  - 順序の「二値判定」より細かい「順列の分類」は、4フレームなら 4! = 24 クラス（逆順を同一視すれば12クラス）の linear probe にそのまま使える。
  - 限界: 画像タスクへの転移で評価しているため、学習した特徴が「動き」を捉えた証拠にはならない。

### A4. Learning and Using the Arrow of Time
- 著者: Donglai Wei, Joseph J. Lim, Andrew Zisserman, William T. Freeman
- 採録: **CVPR 2018**（CVF Open Access で確認）
- arXiv: 見つからず（arXiv API のタイトル検索で該当なし）。CVF: https://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Learning_and_Using_CVPR_2018_paper.html
- **方法と評価**
  - 動画が順再生か逆再生か（arrow of time）を判定する ConvNet を、3つの大規模データセットで学習する。
  - 評価は次の4つ。
    1. 順・逆の判定精度
    2. 撮影上の慣習などの**人工的な手がかり（artificial cues）**が学習に与える影響の分析
    3. クラスタ分析と重要領域の局在化による、判定に使った視覚的手がかりの確認
    4. 応用（行動認識の自己教師あり事前学習、映画クリップの逆再生検出）
- **この研究での再利用と限界**
  - 「逆再生への感度」を測るときは、**物理的に逆にしても不自然に見えない動作**（往復運動など）と、不可逆な動作（落下・注ぐなど）を分けて集計すべきだ、という視点が得られる。
  - 限界: arXiv HTML版がないため、図は掲載していない。

### A5. Self-Supervised Spatiotemporal Learning via Video Clip Order Prediction（VCOP）
- 著者: Dejing Xu et al.
- 採録: **CVPR 2019**（CVF Open Access で確認）
- arXiv: 見つからず。CVF: https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Self-Supervised_Spatiotemporal_Learning_via_Video_Clip_Order_Prediction_CVPR_2019_paper.html
- **方法と評価**
  - 1本の動画から複数のクリップを取り出し、シャッフルした順序を 3D CNN で予測する。フレームよりクリップの方が順序の曖昧さが小さいことが動機。
  - 評価は、最近傍検索と行動認識の fine-tune。C3D / R3D / R(2+1)D の3つのバックボーンで試している。
- **この研究での再利用と限界**
  - 4フレームを「2フレーム × 2クリップ」に分けて前後を当てる probe にすれば、ActivityNet の長い動画内での順序も評価できる。
  - 限界: 検索評価（UCF101 split1 の R@k）は見た目の類似でも高くなるため、それ単体では動きの証拠にならない。

### A6. SpeedNet: Learning the Speediness in Videos
- 著者: Sagie Benaim et al.
- 採録: **CVPR 2020 (oral)**（arXiv comment）
- arXiv: https://arxiv.org/abs/2004.06130
- **方法と評価**
  - 動画が通常速度か早送りかを二値分類するネットワークを、Kinetics で自己教師あり学習する。
  - **人工的な手がかり（shortcut）を避ける工夫**として、ランダムなリサイズ（圧縮ノイズのぼかし）、空間・時間のaugmentation、同じクリップの通常版と早送り版を同じバッチに入れる same-batch training を導入した。
  - 評価には、Kinetics のテスト集合に加えて、高フレームレートの NFS データセットを使った。対策なしでは Kinetics での精度が「誤解を招くほど高い」が、NFS では崩れることをアブレーションで示している。
  - そのほか、行動認識（UCF101 / HMDB51）、動画検索、CAM による可視化。
- **この研究での再利用と限界**
  - 「速度」の probe を作るときは、**フレーム間隔を変える以外の手がかり（圧縮ノイズ、ブレ、解像度）で解けていないか**を、別ドメインのデータで確認する必要がある。同じクリップの速度違いを同じバッチに入れる設計は、MoCo の負例設計にも使える。
  - 限界: 2値の速度判定は「速さ」の一部しか測らない。
- **図**
  - Fig. 2: 「速さらしさ（speediness）は動きの大きさとは違う」ことを示す図。人がカメラに近づくと動きの大きさ（中段）は大きく変わるが、SpeedNet の出力は通常速度のまま安定している。2倍速にすると、歩いている区間だけが「速い」と判定される。読み方: 下段の曲線が、動きの量ではなく再生速度に反応しているかを見る。
  - ![SpeedNet Fig2](https://arxiv.org/html/2004.06130v2/speediness_neq_motion_magnitude_compressed.png)
  - Fig. 3: SpeedNet の構成図（通常速度か早送りかの二値分類器）。
  - ![SpeedNet Fig3](https://arxiv.org/html/2004.06130v2/figures/diagram_post5.png)

### A7. Video Playback Rate Perception for Self-supervised Spatio-Temporal Representation Learning（PRP）
- 著者: Yuan Yao et al.
- 採録: **CVPR 2020**（arXiv comment）
- arXiv: https://arxiv.org/abs/2006.11476
- 依頼文では「PRP pace prediction (ECCV 2020)」と1件にまとめられていたが、実際は**別の2本**である。PRP は Yao らの CVPR 2020 論文、Pace Prediction は次の A8（Wang らの ECCV 2020 論文）。
- **方法と評価**
  - 間引き（dilated）サンプリングで再生速度ラベルを作る。早送り倍率を分類する識別側と、motion-attention 付きでクリップを再構成する生成側を組み合わせる。
  - 評価は行動認識と動画検索。
- **この研究での再利用と限界**
  - 4フレームの取得間隔（stride）を変えて「どの stride で取ったか」を当てる linear probe は、すぐ作れる速度評価になる。
  - 限界: 速度ラベルには「元動画の動作自体の速さ」が混ざる。この問題は RSPNet（A9）で指摘されている。

### A8. Self-supervised Video Representation Learning by Pace Prediction
- 著者: Jiangliu Wang, Jianbo Jiao, Yun-Hui Liu
- 採録: **ECCV 2020**（ECVA 一覧で eccv_2020 として確認）
- arXiv: https://arxiv.org/abs/2008.05861
- **方法と評価**
  - 超スロー・スロー・通常・速い・超速いの5段階のペースでクリップをサンプルし、ペースを分類する。さらに、同じ内容の動画を近づける対照学習を加える。
  - 評価は、複数の3D CNN での行動認識と動画検索。conv5 の attention 可視化もある。
- **この研究での再利用と限界**
  - 5段階のペース分類は、「凍結特徴にペース情報がどれだけ線形に入っているか」を測る **probe のラベル設計の参考**になる。
  - 限界: 対照学習で「同じ内容・違うペース」を正例にすると、ペースに対して不変になる方向に引っ張られうる。目的関数の設計と評価が矛盾しないか注意が必要。
- **図**
  - Fig. 2: 5種類のペース（super slow〜super fast）でサンプルする方法。開始フレームの色が濃いほど速い。
  - ![Pace Fig2](https://arxiv.org/html/2008.05861v2/sample_rate.png)

### A9. RSPNet: Relative Speed Perception for Unsupervised Video Representation Learning
- 著者: Peihao Chen et al.
- 採録: **AAAI 2021**（arXiv comment）
- arXiv: https://arxiv.org/abs/2011.07949
- **方法と評価**
  - 絶対的な速度ラベルは、元動画の動作速度と食い違う（content-label inconsistency）。そこで、**2クリップ間の相対速度**を学習信号にする。
  - 見た目を学習させる課題（A-VID）も別のヘッドで併用し、動きと見た目を分担させる。
  - 評価は、UCF101 / HMDB51 / Something-V2 の行動認識と、動画検索。
- **この研究での再利用と限界**
  - 速度の probe は、「同じ動画から速度を変えて取った2本のどちらが速いか」という**相対判定**にすると、内容の違いによる交絡を減らせる。
  - 限界: 動きと見た目を別ヘッドに分けて学習しているため、単一の埋め込みを評価する場合とは解釈が異なる。
- **図**
  - Fig. 2: 構成図。共有エンコーダの後に、相対速度を学ぶヘッド g_m と、見た目の識別を学ぶヘッド g_a を置く。
  - ![RSPNet Fig2](https://arxiv.org/html/2011.07949v2/framework.png)

### A10. Video Representation Learning by Recognizing Temporal Transformations
- 著者: Simon Jenni, Givi Meishvili, Paolo Favaro
- 採録: **ECCV 2020**（arXiv comment）
- arXiv: https://arxiv.org/abs/2007.10730
- **方法と評価**
  - 次の時間変換のどれがかかっているかを 3D CNN に判別させる。
    - 速度（何フレームおきに取るか）
    - ランダムな並べ替え
    - 往復（forward-backward）
    - 不規則なフレーム飛ばし（順序は保つ warp）
  - 設計原則は「多くのフレームを見ないと解けない変換を選ぶ」こと。ランダムな並べ替えは近くの数フレームで見破れるため、転移性能が低いと報告している。
  - 評価で特に重要なのは **「時間に関する疑似タスク」への転移**である。
    - **同期**: 重なりのある2系列のずれ量（-6〜+6フレーム）を当てる。
    - **前後判定**: 重ならない2系列のどちらが先かを当てる。
    - 事前学習済み C3D の conv5 特徴の上に、新しく初期化した全結合層を学習させる。精度と MAE で、教師あり事前学習や他の自己教師あり手法より良いことを示した。
  - そのほか、UCF101（畳み込み層を凍結）/ HMDB51（全体を fine-tune）、最近傍検索、Guided Backprop による可視化。
- **この研究での再利用と限界**
  - **同期（ずれ量）と前後判定の probe は、4フレームのトークン入力でもそのまま作れる**ので、LoRA前後の比較に最も使いやすい評価の1つ。
  - 限界: probe の分類器に3層の全結合層を使っているので、線形 probe より「情報が取り出しやすい形で入っているか」の判定は甘くなる。
- **図**
  - Fig. 2: 1つのミニバッチに、速度・ランダム・往復・warp の4種類の変換を混ぜて入力し、変換の種類と速度クラスを同時に当てる学習の図。
  - ![Jenni2020 Fig2](https://arxiv.org/html/2007.10730v1/time_warps_v3.png)

### A11. Modeling Video Evolution for Action Recognition（rank pooling）
- 著者: Basura Fernando et al.
- 採録: **CVPR 2015**（CVF Open Access で確認）
- arXiv: CVPR版は見つからず。TPAMI 拡張版「Rank Pooling for Action Recognition」は https://arxiv.org/abs/1512.01848 （arXiv comment に IEEE TPAMI と記載）。
- **方法と評価**
  - フレーム特徴を時間順に並べられるような ranking 関数を動画ごとに学習し、その**パラメータ自体を動画表現にする**。順序が変われば表現も変わる仕組みである。
  - 評価は、Hollywood2 / HMDB51（一般行動）、MPII cooking（細かい行動）、ChaLearn（ジェスチャー）。平均プーリングより 7〜10% 向上した。
- **この研究での再利用と限界**
  - **「順序に敏感な非学習ベースライン」**として使える。凍結 ViT のフレーム特徴に rank pooling をかけたものと、学習した LoRA 特徴とで、順序 probe の精度を比べればよい。
  - 限界: 自己教師あり学習ではなく、表現を作る手法である。

### A12. Dynamic Image Networks for Action Recognition
- 著者: Hakan Bilen et al.
- 採録: **CVPR 2016**（CVF Open Access で確認）
- arXiv: CVPR版は見つからず。拡張版「Action Recognition with Dynamic Image Networks」は https://arxiv.org/abs/1612.00738 （arXiv comment に掲載先の記載なし → 掲載先は未確認）。
- **方法と評価**
  - rank pooling を生の画素に直接かけて、動画全体を1枚の「dynamic image」に要約する。画像用の CNN をそのまま動画に使えるようにする。
  - 近似 rank pooling 層によって、中間特徴マップにも適用できるよう一般化した。
  - 評価は標準的な行動認識ベンチマーク（RGB との組み合わせを含む）。
- **この研究での再利用と限界**
  - 4フレームから dynamic image を作り、それを画像 ViT に入れた特徴は、「時間順序を1枚の画像に焼き込んだ」入力の**比較対象**として使える。
  - 限界: 動きの評価手法そのものではない。

---

## B. 対照学習で scene（見た目）と motion（動き）を分ける研究と、時間に敏感な対照学習

### B1. Enhancing Unsupervised Video Representation Learning by Decoupling the Scene and the Motion（DSM）
- 著者: Jinpeng Wang et al.
- 採録: **AAAI 2021**（arXiv comment「AAAI2021 camera-ready」）
- arXiv: https://arxiv.org/abs/2009.05757
- **方法と評価**
  - **正例**: 空間的な局所撹乱（Spatial Local Disturbance、空間warp など）で、動きはそのまま・見た目を崩したクリップ。
  - **負例**: 時間的な局所撹乱（Temporal Local Disturbance）で、見た目はそのまま・動きを崩したクリップ。
  - 正例を近づけ、負例を遠ざける。こうして「時間を壊したものを負例にする」ことで、時間への感度を上げる。
  - 評価で重要なのは、**1フレームを複製した「static video」に対する通常動画の相対精度向上をクラスごとに散布図にした分析**（Fig. 1）。点が赤線の付近にあるクラスは、「動画にしても精度が上がらない ＝ 見た目だけで解けている」と解釈する。
  - そのほか、静止画入力・別人物の静止画の貼り付け・静止フレームをノイズとして重ねる、という3条件での注目領域の可視化（Fig. 6）、Kinetics のラベル 1% / 10% / 100% での半教師あり評価。
- **この研究での再利用と限界**
  - **「4フレームとも同じフレームにした static clip」に対する特徴距離や probe 精度**を基準にすると、シャッフル感度の数値に意味のある目盛りが付く。
  - 負例に時間撹乱を入れる学習設計は、今のMoCoにそのまま足せる候補である。
  - 限界: static video を基準にする評価は、クラスラベルのある下流データセットが前提になる。
- **図**
  - Fig. 1: 各点が1クラス。通常動画を入れたときの、static video に対する相対精度向上を示す。(a) では多くのクラスが赤線付近にあり、見た目だけで解けている。DSM の (b)(d) では全体が持ち上がる。(c) は MoCo のベースライン。
  - ![DSM Fig1](https://arxiv.org/html/2009.05757v3/figs/teaser.png)
  - Fig. 6: 「本当に動いている領域を見ているか」を確認する図。静止画入力、別人物の静止画の貼り付け、静止フレームをノイズとして重ねた入力の3条件で注目領域を比べる。
  - ![DSM Fig6](https://arxiv.org/html/2009.05757v3/figs/where_to_look.png)

### B2. Removing the Background by Adding the Background: Towards Background Robust Self-supervised Video Representation Learning（Background Erasing, BE）
- 著者: Jinpeng Wang et al.
- 採録: **CVPR 2021**（arXiv comment）
- arXiv: https://arxiv.org/abs/2009.05769
- **方法と評価**
  - 同じ動画のランダムな静止フレームを全フレームに足して「気をそらす動画（distracting video）」を作り、元の動画と特徴を一致させる。背景（静的な見た目）に頼らないようにする。
  - MoCo に追加すると、UCF101 で +16.4%、HMDB51 で +19.1%、Diving48 で +14.5% 向上した。
  - 評価で重要なのは、**クラスごとの「static video での分類精度」と「BE による改善量」が強い負の相関を持つ**こと（Fig. 4）。つまり、見た目で解けないクラスほど改善している。
  - そのほか、人物が主体の actor-HMDB51、未知クラスへの汎化の可視化、静止画入力・人物の貼り付けへの反応（Fig. 7）。
- **この研究での再利用と限界**
  - LoRA前後の改善量を、「static clip でどれだけ当たるか」というクラスの性質に対してプロットすれば、**改善が動き由来か見た目由来か**を1枚の図で説明できる。
  - 限界: ActivityNet にはクラスラベルがあるが、4フレームの短いクリップで static 精度を測ると、ほとんどのクラスが見た目だけで解ける可能性がある。
- **図**
  - Fig. 1: 背景で答えてしまう「background cheating」の概念図。
  - ![BE Fig1](https://arxiv.org/html/2009.05769v4/teaser.png)
  - Fig. 4: 各点が HMDB51 の1クラス。横軸が static video での精度、縦軸が BE による相対的な改善。右下がりで、見た目に頼れないクラスほど改善が大きい。
  - ![BE Fig4](https://arxiv.org/html/2009.05769v4/relative_bias.png)

### B3. Motion-aware Contrastive Video Representation Learning via Foreground-background Merging（FAME）
- 著者: Shuangrui Ding et al.
- 採録: **CVPR 2022**（arXiv comment「CVPR2022 camera ready」）
- arXiv: https://arxiv.org/abs/2109.15130
- **方法と評価**
  - 同じ動画の2クリップを正例として近づけると、共通する静的背景が近道（background bias）になる。
  - そこで、フレーム差分と色の統計で前景（動いている領域）を取り出し、**別の動画の背景に合成**したクリップを正例にする。検出器は使わない。
  - 評価は、UCF101 / HMDB51 / Diving48 の下流性能、クラスに依存しない活性化マップ（CAAM）、CAM、検索結果の可視化。
  - 面白い分析として、**R(2+1)D の時間方向カーネル重みの分布を、教師あり・対照学習・FAME で比べたバイオリンプロット**（Fig. 3）がある。学習が「時間方向の重み」を使っているかを、重みの統計で見ている。
- **この研究での再利用と限界**
  - 今の「正例 ＝ 同じクリップの2つのaugmentation」は、まさに背景という近道を許す構図である（Fig. 2 の議論）。
  - 評価としては、**LoRA 差分（ΔW = BA）の大きさを層ごと・時間位置埋め込みとの関係で見る**のが、Fig. 3 に相当する内部統計になる。
  - 限界: 前景の抽出はフレーム差分に頼っており、4フレームの短いクリップでは前景がうまく取れない可能性がある。
- **図**
  - Fig. 2: 飛び込みの例。同じ背景（緑の点線）で動き（赤枠）が違う2クリップを正例にすると、静的バイアスに傾くことを説明する図。
  - ![FAME Fig2](https://arxiv.org/html/2109.15130v3/Figure/teaser.png)
  - Fig. 4: FAME の対照学習の全体像。前景を他の動画の背景に合成してから、既存の対照学習に入れる。
  - ![FAME Fig4](https://arxiv.org/html/2109.15130v3/Figure/moco.png)
  - Fig. 3: R(2+1)D の全層の時間カーネル重みの分布（左から、教師あり・対照学習・FAME）。
  - ![FAME Fig3](https://arxiv.org/html/2109.15130v3/violinplot.png)

### B4. Motion-Focused Contrastive Learning of Video Representations（MCL）
- 著者: Rui Li et al.
- 採録: **ICCV 2021 (oral)**（arXiv comment）
- arXiv: https://arxiv.org/abs/2201.04029
- **方法と評価**
  - オプティカルフローを使って、動きの大きい tubelet（時空間の小領域）をaugmentationとしてサンプルする。
  - さらに、畳み込み層の勾配マップをフローの地図（空間・時間・時空間の3種類）に揃える損失を加える。
  - 評価は、UCF101 の linear 評価（81.91%、ImageNet 教師あり事前学習より +6.78%）、Kinetics-400 の linear 評価（66.62%）、Grad-CAM と動きの地図の比較。
- **この研究での再利用と限界**
  - **「勾配マップや attention がオプティカルフローの大きい領域とどれだけ重なるか」**を数値化すれば、可視化を定量評価に変えられる。
  - 限界: オプティカルフローの計算コストがかかる。

### B5. Static and Dynamic Concepts for Self-supervised Video Representation Learning
- 著者: Rui Qian et al.
- 採録: **ECCV 2022**（arXiv comment）
- arXiv: https://arxiv.org/abs/2207.12795
- **方法と評価**
  - 静止フレームとフレーム差分を使って、「静的概念」と「動的概念」のプロトタイプを分けて学習する。その後、cross-attention で概念ごとの局所特徴を集めて対照学習する。
  - 評価は、UCF-101 / HMDB-51 / Diving-48。加えて、**静的・動的・両方の局所特徴を使ったときのクラス別精度比較**、概念コードのクラス間類似度、概念ごとの attention 可視化。
- **この研究での再利用と限界**
  - 「フレーム差分から作った特徴」と「静止フレームから作った特徴」に分けてクラス別に比べる方法は、ViT の**フレーム差分トークンを入力した場合と比べる**という簡単な対照実験に応用できる。
  - 限界: 専用の概念モジュールを前提にした分析である。
- **図**
  - Fig. 2: 全体の枠組み。概念プロトタイプとの類似度で概念コードを作り、静的・動的に分けて揃える。
  - ![StaticDynamic Fig2](https://arxiv.org/html/2207.12795v1/fig/framework.png)

### B6. TCLR: Temporal Contrastive Learning for Video Representation
- 著者: Ishan Dave et al.
- 採録: **Computer Vision and Image Understanding (CVIU)**（arXiv comment で採録を確認。巻・号・年は未確認）
- arXiv: https://arxiv.org/abs/2101.07974
- **方法と評価**
  - 通常のインスタンス対照学習に、次の2つの損失を加える。
    - **local-local 損失**: 同じ動画の重ならないクリップ同士を区別させる。
    - **global-local 損失**: 1クリップの特徴マップの各時間ステップ同士を区別させる。
  - 評価は、UCF101 / HMDB51 の行動認識、最近傍検索、少ラベル学習、Diving48（細かい行動）。
  - 重要な分析は、**特徴マップの時間ステップ間のコサイン類似度行列を損失ごとに並べた図**（Fig. 13）。インスタンス対照学習だけでは時間ステップ間がほぼ同じ特徴になり、TCLR では対角成分以外の類似度が下がる。
- **この研究での再利用と限界**
  - **4フレーム分のトークン（フレームごとに平均したもの）の 4×4 コサイン類似度行列**を、LoRA前・LoRA後・シャッフル入力で並べれば、Fig. 13 と同じ形の証拠になる。「特徴がシャッフルで変わらない」現象の原因（時間方向に特徴が均一化しているのか）を確かめるのに直接使える。
  - 限界: 時間ステップが区別できることは、順序・方向を理解していることと同じではない。
- **図**
  - Fig. 2: 3つの損失（インスタンス対照、local-local、global-local）の構成図。
  - ![TCLR Fig2](https://arxiv.org/html/2101.07974v4/Fig1_TCLR.png)
  - Fig. 13: 時間ステップ間のコサイン類似度。(b) はインスタンス対照損失のみで、時間ステップ間の類似度が高い（区別できていない）。(e) は TCLR で、類似度が下がる（時間的に区別できる）。
  - (b) インスタンス対照損失のみ: ![TCLR Fig13b](https://arxiv.org/html/2101.07974v4/IC.png)
  - (e) TCLR: ![TCLR Fig13e](https://arxiv.org/html/2101.07974v4/TCLR.png)

### B7. Time-Equivariant Contrastive Video Representation Learning（TE）
- 著者: Simon Jenni, Hailin Jin
- 採録: **ICCV 2021 (oral)**（arXiv comment）
- arXiv: https://arxiv.org/abs/2112.03624
- **方法と評価**
  - 時間方向のaugmentationに「不変」にするのではなく、**「等変（equivariant）」**にする。2つの動画に同じ相対時間変換（速度・再生方向・時間クロップのずれ）をかけたとき、その相対変換を表すベクトル同士が一致するよう対照学習する。
  - 補助課題として、2クリップが「重なっている / 正しい順で重なっていない / 逆順で重なっていない」の3クラス分類も入れる。
  - 評価は、UCF101 / HMDB51 の linear 評価・fine-tune・最近傍分類と検索（R@k）、Diving48。アブレーションでは、速度予測・再生方向予測・順序重なり予測をそれぞれ外したときの効果を見ている。
  - バッチサイズを変えて、等変性の学習と「時間augmentation同士を区別するだけ」の学習を比べた分析（Fig. 5）もある。
- **この研究での再利用と限界**
  - 今の MoCo は「同じ順序の2つのaugmentation」を正例にしているので、時間変換に対して**不変でも等変でもない（何の圧力もない）**状態である。TE の考え方は「正例ペアの間に時間変換を入れ、その変換を当てさせる」という改善案にも、評価案にもなる。
  - 評価案の例: 凍結した特徴の差分から、相対変換（同じ・逆再生・2倍速）を当てる probe。
  - 限界: 3D CNN 前提の実験である。
- **図**
  - Fig. 2: 時間変換（速度・方向）と、3クラスの順序・重なり課題の説明図。2本の動画で、同じ色のクリップ間の相対変換が同一であることを認識させる。
  - ![TE Fig2](https://arxiv.org/html/2112.03624v1/fig2_v1_reduced.png)
  - Fig. 3: 時間等変な対照学習モデルの全体像。
  - ![TE Fig3](https://arxiv.org/html/2112.03624v1/equi_model.png)

### B8. Contrast and Order Representations for Video Self-Supervised Learning（CORP）
- 著者: Kai Hu et al.
- 採録: **ICCV 2021**（CVF Open Access で確認）
- arXiv: 見つからず（タイトル検索で該当なし）。CVF: https://openaccess.thecvf.com/content/ICCV2021/html/Hu_Contrast_and_Order_Representations_for_Video_Self-Supervised_Learning_ICCV_2021_paper.html
- **方法と評価**
  - 2つのクリップについて、まず同じ動画由来かを判定し、同じ動画ならその**時間順序**も予測する。
  - 対称な類似度（contrast）と反対称なパターン（order）を分けて学習する decoupling attention を提案した。
  - 評価は、Kinetics-400 と Something-something V2 の行動認識。
- **この研究での再利用と限界**
  - 「同一性」と「順序」を同じ埋め込みで同時に学ぶと、対称な類似度だけでは順序を表現できない、という指摘は重要である。**コサイン類似度（対称）でシャッフル感度を測ること自体の限界**として、評価設計の注意点に書ける。
  - 限界: arXiv HTML がないため図はない。

### B9. Self-supervised Video Representation Learning by Uncovering Spatio-temporal Statistics
- 著者: Jiangliu Wang et al.
- 採録: **TPAMI**（arXiv comment「Accepted by TPAMI」。年は未確認）。前身は CVPR 2019「Self-supervised Spatio-temporal Representation Learning for Videos by Predicting Motion and Appearance Statistics」（https://arxiv.org/abs/1904.03597 、arXiv comment で CVPR 2019 を確認）。
- arXiv: https://arxiv.org/abs/2008.13426
- **方法と評価**
  - 「最も大きな動きがある大まかな位置とその主な方向」「時間方向に色が最も変化する位置と主な色」などの統計量を、空間分割パターンで大まかな位置として回帰させる。
  - 評価は、C3D / 3D-ResNet / R(2+1)D / S3D-G の4つのバックボーンで、行動認識、動画検索、**動的シーン認識**、行動類似度ラベリングの4つの下流タスク。
- **この研究での再利用と限界**
  - 「主な動きの方向（8方向など）を当てる probe」は、**方向**を評価したいというこの研究の目的に合う。ラベルはオプティカルフローや motion boundary から自動で作れる。
  - 限界: カメラの動きと物体の動きを区別しない統計量である。

### B10. Time Is MattEr: Temporal Self-supervision for Video Transformers（TIME）
- 著者: Sukmin Yun et al.
- 採録: **ICML 2022**（arXiv comment）。ICML は今回の会議リストの対象外だが、評価方法がこの研究に最も近いので含めた。
- arXiv: https://arxiv.org/abs/2207.09067
- **方法と評価**
  - 動画 Transformer が空間に偏っていることを、次の3つの実験で示した。
    1. **SSv2 で、元の動画とフレームをシャッフルした動画の精度を比較**した。Sevilla-Lara らの「静的クラス / 時間クラス」に分けても、シャッフル後の精度が高いままだった。
    2. **各ブロックの凍結出力に線形分類器を付けてフレーム順を当てさせた**。例として Motionformer では、3番目のブロックで 99.5% だった精度が、最終ブロックでは 45.6% に下がった（**深い層ほど時間情報が消える**）。
    3. 元の動画とシャッフル動画の**確信度の差の分布**を見た。
  - 対策として、次の3つの自己教師ありタスクを追加した。
    - シャッフル入力に低い確信度を出させる（debias）
    - フレーム順を当てさせる（order）
    - トークンの時間方向の流れの向き（8方向 + 中心）を当てさせる（flow）
  - Kinetics-400 でも、元の動画とシャッフル動画の精度を比較している。
- **この研究での再利用と限界**
  - **最も直接的に流用できる評価セット**である。具体的には次の3つ。
    1. ViT の各ブロックで「4フレームの順列」を線形 probe し、LoRA前後で曲線を比べる。
    2. シャッフル入力での特徴変化や確信度の差を、ヒストグラムで示す。
    3. 時間が重要なクラスと静的なクラスに分けて集計する。
  - 「最終層では順序情報が消えているが、中間層には残っている」可能性も、これで切り分けられる。
  - 限界: 教師ありの fine-tune での分析なので、自己教師ありの埋め込みでは、確信度の代わりに特徴距離を使う必要がある。
- **図**
  - Fig. 1: (a) SSv2 で、元動画とシャッフル動画の精度（静的クラス / 時間クラス別）。(b) ブロックごとの順序予測の線形 probe 精度（深い層ほど下がる）。(c) 元動画とシャッフル動画の確信度差の分布。
  - ![TIME Fig1](https://arxiv.org/html/2207.09067v1/0612_fig1c_add.png)
  - Fig. 2: 3つの時間自己教師ありタスク（debias / order / flow）の説明図。
  - ![TIME Fig2](https://arxiv.org/html/2207.09067v1/0615_fig2.png)

### B11. Masked Motion Encoding for Self-Supervised Video Representation Learning（MME）
- 著者: Xinyu Sun et al.
- 採録: **CVPR 2023**（arXiv comment）
- arXiv: https://arxiv.org/abs/2210.06096
- **方法と評価**
  - VideoMAE のように見た目（画素）を復元させるのではなく、マスクした領域の**動きの軌跡（位置変化と形状変化）**を復元させる。
  - 評価で重要なのは、**シャッフルした動画で VideoMAE を事前学習しても、画素の復元誤差は低く収束し、下流の行動認識性能も元の動画で学習した場合とほぼ同じ**ことを示した対照実験（Fig. 3）。つまり、画素の復元は各フレーム内で完結していて、時間を使っていない。
- **この研究での再利用と限界**
  - **「シャッフルしたクリップで学習した LoRA」と「正しい順序のクリップで学習した LoRA」を作り、下流評価や probe に差が出るか**を見る。これは、今の学習が時間を使っているかを判定する最も強い対照実験の1つである。
  - 限界: 学習をもう1回（シャッフル条件で）回すコストがかかる。
- **図**
  - Fig. 3: (a) シャッフル動画でも画素の復元誤差が低く収束する。(b) 下流の精度で、シャッフル学習の VideoMAE と通常の VideoMAE に差がない。一方、MME は時間情報があると上回る。
  - ![MME Fig3](https://arxiv.org/html/2210.06096v2/toy-exp.png)
  - Fig. 1: 動きの軌跡（位置変化と形状変化）を復元する、という考え方の図。
  - ![MME Fig1](https://arxiv.org/html/2210.06096v2/fig1.png)

### B12. MGMAE: Motion Guided Masking for Video Masked Autoencoding
- 著者: Bingkun Huang et al.
- 採録: **ICCV 2023**（arXiv comment + CVF で確認）
- arXiv: https://arxiv.org/abs/2308.10794
- **方法と評価**
  - オンラインのオプティカルフローでマスクをワープし、時間方向に一貫したマスク（動きに沿った tube）を作る。これにより、時間方向の情報漏れを減らす。
  - 評価は、Something-Something V2 と Kinetics-400 の fine-tune、**VideoMAE とのクラス別精度比較**（Fig. 4）、マスクの可視化。
- **この研究での再利用と限界**
  - 「時間方向の情報漏れ（隣のフレームから写せば解ける）」という視点は、4フレームが互いに似すぎている ActivityNet のクリップにも当てはまる。
  - 限界: MAE 系の手法なので、対照学習と直接比べることはできない。
- **図**
  - Fig. 4: クラス別に、MGMAE と VideoMAE の精度を比べた図。
  - ![MGMAE Fig4](https://arxiv.org/html/2308.10794v1/figures/acc_vs_partial.png)

### B13. Tubelet-Contrastive Self-Supervision for Video-Efficient Generalization
- 著者: Fida Mohammad Thoker, Hazel Doughty, Cees Snoek
- 採録: **ICCV 2023**（arXiv comment + CVF で確認）
- arXiv: https://arxiv.org/abs/2303.11003
- **方法と評価**
  - 別の動画に、同じ人工的な動きの軌跡（tubelet）を貼り付ける。見た目は違うが局所的な動きだけが共通する2本を正例にする。
  - tubelet には、線形・非線形の動きと、拡大縮小・回転・せん断を加えて、事前学習データにない動きも作る。
  - 評価は、Something-Something、FineGym、SEVERE ベンチマークなど10の下流設定。事前学習データを 25% に減らしても性能を保つというデータ効率も示した。
  - **UCF の小規模設定で、クラス別の精度差を「最も上がった10クラス・最も下がった10クラス」として示した図**（Fig. 7）がある。動きが必要なクラスは上がり、空間的なクラスは少し下がる、というトレードオフまで報告している。
- **この研究での再利用と限界**
  - **「人工的に動きを合成した、見た目は同じで動きだけが違うクリップ」**は、評価用の統制刺激としても使える。例えば、同じ背景に同じパッチを貼り、左向き・右向き・速い・遅いと動かして、特徴で区別できるかを見る。
  - 限界: 合成した動きは自然な動作とは分布が違う。
- **図**
  - Fig. 1: 上段は通常の時間対照ペアで、空間バイアスが大きい。下段は tubelet 対照ペアで、tubelet 内の動きだけを共有する。
  - ![Tubelet Fig1](https://arxiv.org/html/2303.11003v2/concept-1.png)
  - Fig. 7: ベースラインに対するクラス別の精度差（上位・下位10クラス）。
  - ![Tubelet Fig7](https://arxiv.org/html/2303.11003v2/figures/ucf101_bar_graph.png)

### B14. Time Does Tell: Self-Supervised Time-Tuning of Dense Image Representations（TimeTuning, TimeT）
- 著者: Mohammadreza Salehi, Efstratios Gavves, Cees G. M. Snoek, Yuki M. Asano
- 採録: **ICCV 2023**（CVF Open Access で確認）
- arXiv: https://arxiv.org/abs/2308.11796
- **方法と評価**
  - DINO などの**画像で事前学習した ViT**を、動画の時間方向に揃えるクラスタリング損失で fine-tune する。設定としてはこの研究に近い。
  - 評価は、YTVOS で学習し、DAVIS17 / YTVOS での教師なし動画セマンティックセグメンテーション（クラスタリング・過剰クラスタリング）、Pascal VOC への転移。DINO の特徴が時間方向に「跳ねる」ことも可視化している。
- **この研究での再利用と限界**
  - 「画像で事前学習した ViT に動画で追加学習する」という設定は同じだが、**狙っているのは時間方向の一貫性（不変性）であり、動きへの感度ではない**。この研究の「シャッフルしても変わらない」は、TimeT 的な観点からは望ましい性質になりうる。目的の違いを先生に説明する際の対比材料になる。
  - 限界: 密な（ピクセル単位の）特徴を評価するものなので、クリップレベルの評価には直接使えない。

### B15. Self-supervised Video Transformer（SVT）
- 著者: Kanchana Ranasinghe et al.
- 採録: **CVPR 2022**（arXiv comment）
- arXiv: https://arxiv.org/abs/2112.01514
- **方法と評価**
  - フレームレートや視野の異なる global / local ビューを作り、それらを一致させる（負例なし、DINO 風）。
  - 評価は、Kinetics-400 / UCF-101 / HMDB-51 / SSv2 の行動認識。
- **この研究での再利用と限界**
  - 「異なるフレームレートのビューを一致させる」設計は、**速度に対して不変**になる方向の学習である。速度 probe の精度が下がるのは仕様どおりと解釈すべきことを示す例になる。
  - 限界: 時間情報を直接評価する専用の実験は、アブストラクトの範囲では確認できなかった。

---

## C. 評価方法そのものを扱った研究（時間が本当に必要か）

### C1. Only Time Can Tell: Discovering Temporal Data for Temporal Modeling
- 著者: Laura Sevilla-Lara et al.
- 採録: **WACV 2021**（CVF Open Access で確認）
- arXiv: https://arxiv.org/abs/1907.08340
- **方法と評価**
  - Kinetics と Something-something について、**フレームをシャッフルした動画を人に見せ、認識精度が落ちたクラスを「時間クラス」として選ぶ**。計算モデルで選ぶとバイアスがかかるため、人による判定にした。
  - 時間クラスと静的クラスでの性能差が、ネットワークの時間モデリング能力（R2D と R(2+1)D の違いなど）と相関することを示した。
  - 時間クラスで学習すると未知クラスへの転移が良くなり、テスト時にシャッフルしたときの性能低下も大きくなる。
- **この研究での再利用と限界**
  - 公開されているクラス分割（Kinetics / SSv2 の時間クラス）を使えば、**「時間クラスでのシャッフル感度」と「静的クラスでのシャッフル感度」の差**を報告できる。TIME もこの分割を使っている。
  - 限界: ActivityNet のクラスに対する分割はないので、使うなら Kinetics / SSv2 で評価するか、自分で同様の分割を作る必要がある。
- **図**
  - Fig. 1: 「あくび・くしゃみ・泣く」はフレームを見るだけでは区別しにくい、という時間の必要性の例。
  - ![OTCT Fig1](https://arxiv.org/html/1907.08340v2/sneezing_vs_crying.png)
  - Fig. 6(b): テスト時にシャッフルしたときの影響。時間クラスで学習したネットワークほど、シャッフルで大きく落ちる。
  - ![OTCT Fig6b](https://arxiv.org/html/1907.08340v2/shuffle.png)

### C2. How Severe is Benchmark-Sensitivity in Video Self-Supervised Learning?（SEVERE）
- 著者: Fida Mohammad Thoker, Hazel Doughty, Piyush Bagad, Cees Snoek
- 採録: **ECCV 2022**（arXiv comment）
- arXiv: https://arxiv.org/abs/2203.14221
- **方法と評価**
  - 動画の自己教師あり学習の評価が、UCF101 / HMDB51 での fine-tune という慣例に偏っていることを、4つの観点（ドメイン、サンプル数、行動の細かさ、タスク）で検証した。7つのデータセット、9つの手法、6つのタスク、500以上の実験を行った。
  - その中から、汎化性を測る SEVERE ベンチマークを抽出した。
- **この研究での再利用と限界**
  - 「UCF101 で良い ＝ 動きを学んだ」とは言えない、という主張の根拠になる。細かい行動や別ドメインの評価を入れる理由付けに使える。
  - 限界: arXiv HTML 版の本文がほぼ取得できなかったため（参考文献部分のみ）、データセットの具体的な構成はこのメモでは確認していない。図もない。

### C3. Revisiting the "Video" in Video-Language Understanding（ATP）
- 著者: Shyamal Buch et al.
- 採録: **CVPR 2022 (oral)**（arXiv comment）
- arXiv: https://arxiv.org/abs/2206.01720
- **方法と評価**
  - 画像レベルの理解だけで動画ベンチマークをどこまで解けるかの上限を測る「atemporal probe（ATP）」を提案した。ビデオQAや検索の多くは、時間を理解しなくても高い性能が出ることを示した。
  - ATP を使って、時間的に難しいサブセットを抽出する手法も示した。
- **この研究での再利用と限界**
  - 「**時間を使わないベースライン**（1フレームの特徴、またはフレーム特徴の平均）」を必ず評価表に入れるべき、という根拠になる。
  - 限界: 動画と言語のタスク向けの研究である。

### C4. Test of Time: Instilling Video-Language Models with a Sense of Time
- 著者: Piyush Bagad, Makarand Tapaswi, Cees G. M. Snoek
- 採録: **CVPR 2023**（arXiv comment）
- arXiv: https://arxiv.org/abs/2301.02074
- **方法と評価**
  - 「before / after」の関係について、7つの既存の動画言語モデルが時間順序の一貫性を理解できていないことを示した。
  - VideoCLIP を少量のデータで事後学習して時間の感覚を持たせ、時間認識の必要度が異なる3つのタスク・6つのデータセットで zero-shot 評価した。
- **この研究での再利用と限界**
  - 「既存の大規模モデルに、少ないデータで追加学習して時間の感覚を足す」という発想は、LoRA の位置付けと近い。**評価は時間の必要度が異なるタスクに分けて行うべき**という点を参考にできる。
  - 限界: テキストを使う評価なので、この研究にはそのまま使えない。

---

## D. 画像→動画のパラメータ効率的な適応（PEFT）と、時間モデリングの評価

共通する評価の型: K400（見た目で解ける）と SSv2 / Diving48（時間が必要）の両方で評価し、時間モジュールのアブレーションの差が SSv2 側で大きいことを示して、時間モデリングの証拠とする。

### D1. ST-Adapter: Parameter-Efficient Image-to-Video Transfer Learning
- 著者: Junting Pan et al.
- 採録: **NeurIPS 2022**（arXiv comment）
- arXiv: https://arxiv.org/abs/2206.13559
- **方法と評価**
  - 凍結した画像モデル（CLIP ViT など）の各ブロックに、depth-wise の時空間畳み込みを含む軽量アダプタを挿入する。パラメータの約 8% で、時間について推論できるようにする。
  - 評価は K400 / SSv2 など。アブレーションでは、ボトルネック幅、挿入するブロック、ブロック内の位置、**畳み込みカーネルの形**を変えた。**時間方向のカーネル幅が最も性能に効く**ことから、時間構造のモデリングが重要だと主張している。SSv2 での attention 可視化もある。
- **この研究での再利用と限界**
  - LoRA は「トークン同士の混ぜ方を変えない」適応なので、ST-Adapter のように**時間方向の演算を明示的に入れる適応との比較**が、構造上の限界を議論する材料になる。
  - 限界: 教師あり分類で評価しており、自己教師あり学習ではない。
- **図**
  - Fig. 1: (a) 時間モジュールを設計してから全体を fine-tune する従来法と、(b) 軽量な ST-Adapter だけを学習する方法の比較。
  - ![ST-Adapter Fig1](https://arxiv.org/html/2206.13559v3/fig1.png)

### D2. AIM: Adapting Image Models for Efficient Video Action Recognition
- 著者: Taojiannan Yang et al.
- 採録: **ICLR 2023**（arXiv comment）
- arXiv: https://arxiv.org/abs/2302.03024
- **方法と評価**
  - 凍結した ViT の空間 attention を時間方向にも再利用し、アダプタを3か所（空間・時間・MLP並列）に加える。
  - 評価は K400 / K700 / SSv2 / Diving-48。
  - SSv2 での段階的なアブレーション（Table 1）では、次の3つの基準と比べた。
    - 凍結した space-only モデルの linear probe: **15.1%**
    - space-only モデルの full fine-tune: **36.2%**
    - TimeSformer の full fine-tune: **59.5%**
  - これらに対して、空間・時間・joint の適応を順に足していく効果を見ている。attention の可視化もある。
- **この研究での再利用と限界**
  - 「凍結した画像 ViT をフレームごとに処理して平均したもの（space-only）」の **SSv2 linear probe は大幅に低い**、という数値は、自分のモデルを SSv2 で linear probe したときの比較基準として使える（バックボーンは IN-21K の ViT-B/16）。
  - 限界: 教師ありの適応である。
- 図: 方法の図（Fig. 2）は今回は掲載しない。HTML では Fig. 1 と Fig. 4 の画像のみ確認した。

### D3. Dual-path Adaptation from Image to Video Transformers（DualPath）
- 著者: Jungin Park, Jiyoung Lee, Kwanghoon Sohn
- 採録: **CVPR 2023**（arXiv comment）
- arXiv: https://arxiv.org/abs/2303.09857
- **方法と評価**
  - 空間パスと時間パスを分ける。時間パスでは、連続する低解像度フレームを**格子状に並べた1枚の画像（grid-like frameset）**にして画像 ViT に入れる。それぞれのパスに軽量なアダプタを置く。
  - 評価は K400 / HMDB51 / SSv2 / Diving-48。
  - 「格子状のフレーム集合は本当に時間文脈を捉えているか」という節で、SSv2 の attention を時間適応あり・なしで比べている。
- **この研究での再利用と限界**
  - この研究の「4フレームをトークンとして連結」と、DualPath の「格子状に並べる」は、どちらも**画像 ViT の空間 attention にフレーム間の関係を担わせる**設計である。時間位置埋め込みがどの程度効いているかを、埋め込みを入れ替える・取り除くアブレーションで確認することを勧める。
  - 限界: 時間文脈の検証は可視化が中心である。
- **図**
  - Fig. 3: DualPath の全体構成（空間パスと、格子状フレーム集合を使う時間パス）。
  - ![DualPath Fig3](https://arxiv.org/html/2303.09857v1/fig3.png)
  - Fig. 4(b): 時間適応ありの attention（SSv2「回して、すぐ止まる」の例）。
  - ![DualPath Fig4b](https://arxiv.org/html/2303.09857v1/figures/qual_w.png)

### D4. Frozen CLIP Models are Efficient Video Learners（EVL）
- 著者: Ziyi Lin et al.
- 採録: **ECCV 2022**（arXiv comment）
- arXiv: https://arxiv.org/abs/2208.03550
- **方法と評価**
  - 凍結した CLIP 画像エンコーダの多層特徴を、軽量な Transformer デコーダで集約する。デコーダには局所時間モジュール（時間畳み込み、時間位置埋め込み、時間 cross-attention）を置く。
  - 評価は Kinetics-400 と Something-Something-v2 の2つ。
  - アブレーション（Table 8）で、**時間モジュールの効果が K400 では +0.5% 未満、SSv2 では +13.8%**と大きく異なることを示した。また、CLIP ViT-B/16 の SSv2 linear probe は約 20% と述べている。
- **この研究での再利用と限界**
  - **「K400 と SSv2 での時間モジュールの効果の差」は、時間モデリングの効果を示す最も分かりやすい証拠の型**である。LoRA あり・なし × K400 / SSv2 の linear probe の 2×2 表にすれば、先生にも伝わりやすい。
  - 限界: 教師あり学習である。また、ActivityNet は K400 に近い「見た目で解ける」データセットである可能性が高い。

### D5. Side4Video: Spatial-Temporal Side Network for Memory-Efficient Image-to-Video Transfer Learning
- 著者: Huanjin Yao, Wenhao Wu, Zhiheng Li
- 採録: **未確認**（arXiv comment「Technical report」、Web 検索でも採録先は確認できなかった）
- arXiv: https://arxiv.org/abs/2311.15769
- **方法と評価**
  - 凍結した大規模画像モデル（ViT-E 4.4B まで）の多層特徴を、横に置いた軽量な時空間サイドネットワークで処理する。重いバックボーンへの逆伝播を避けて、メモリを削減する。
  - 評価は、**時間に関係するデータセット（SSv1 / SSv2）とシーン中心のデータセット（K400）を明示的に分けて**行い、テキスト動画検索も含む。
- **この研究での再利用と限界**
  - データセットを時間型とシーン型に分けて評価するという書き方を参考にできる。
  - 限界: 採録先が未確認なので、引用するときは「arXiv の技術報告」と書くこと。

### D6. ZeroI2V: Zero-Cost Adaptation of Pre-trained Transformers from Image to Video
- 著者: Xinhao Li, Yuhan Zhu, Limin Wang
- 採録: **ECCV 2024**（arXiv comment）
- arXiv: https://arxiv.org/abs/2310.01324
- **方法と評価**
  - 次の2つで、推論時の追加コストをゼロにする。
    - 一部の attention ヘッドの K/V を別時刻のフレームから取る「時空間二重ヘッド attention（STDHA）」
    - 学習後に元の重みへ統合できる**線形アダプタ**（LoRA と同じく統合可能な線形の低コスト適応）
  - 評価は SSv2 など。STDHA のアブレーション（時間ヘッドの数、時間の受容野）や、空間ヘッド・時間ヘッドごとの Grad-CAM がある。
- **この研究での再利用と限界**
  - **統合可能な線形アダプタだけでは時間モデリングは不十分で、トークンの混ぜ方（STDHA）を変える必要がある**、という設計判断は、Q/V の LoRA だけの限界を議論する材料として最も近い。
  - 限界: 教師ありの fine-tune である。
- **図**
  - Fig. 2: 線形適応と STDHA の説明図。
  - ![ZeroI2V Fig2](https://arxiv.org/html/2310.01324v2/method.png)

### D7. MotionDirector: Motion Customization of Text-to-Video Diffusion Models（動画分野の LoRA 分析の例）
- 著者: Rui Zhao et al.
- 採録: **ECCV 2024**（ECVA 一覧で eccv_2024 として確認）
- arXiv: https://arxiv.org/abs/2310.08465
- **方法と評価**
  - テキストから動画を生成する拡散モデルに、特定の動きを LoRA で学習させる。素朴な LoRA では、動きが学習動画の見た目と結び付いてしまう。
  - そこで、**空間（見た目）用の LoRA と時間（動き）用の LoRA を分けた dual-path 構成**にし、見た目の影響を取り除く時間損失（appearance-debiased temporal loss）を導入した。
  - 評価は、学習した動きを別の見た目で生成できるか（見た目の多様性と動きの忠実性）。
- **この研究での再利用と限界**
  - 「LoRA が動きを学ぶと見た目と結び付く」という観察は、認識モデルの LoRA でも起こりうる問題として引用できる。
  - 限界: 生成モデルなので評価方法は流用しにくい。今回の調査では、**認識用の LoRA の時間モデリング能力を専門に分析した論文は確認できなかった**（今後の調査課題）。

---

## E. ストリーミング・連続動画からの自己教師あり学習

### E1. The Challenges of Continuous Self-Supervised Learning
- 著者: Senthil Purushwalkam, Pedro Morgado, Abhinav Gupta
- 採録: **ECCV 2022**（ECVA 一覧で eccv_2022 として確認）
- arXiv: https://arxiv.org/abs/2203.12710
- **方法と評価**
  - 連続したデータストリームで自己教師あり学習（SimSiam、ResNet-18）を行い、3つの課題を示した。
    1. 計算効率とデータ効率の悪さ
    2. 時間的に相関した非 IID データによる表現の劣化
    3. 非定常な分布での忘却
  - 評価は ImageNet などの linear 分類。相関の強いストリームとして、Kinetics の動画から取ったフレームをつなげたストリームと、KrishnaCam（1人の大学院生の9か月分の一人称動画）を使った。
  - 対策として、リプレイバッファと、相関したサンプルを捨てる Minimum Redundancy バッファを提案した。
- **この研究での再利用と限界**
  - ストリーミング MoCo では、**キュー内の負例が同じ動画の近い時刻のクリップで埋まり、見た目の識別すら難しくなる / 偏る**可能性がある。評価に「バッファの相関度」や「キュー内の同一動画率」を記録しておくとよい。
  - 限界: 画像レベルの表現評価で、時間情報の評価ではない。
- **図**
  - Fig. 2: 連続自己教師あり学習の3つの課題（データが繰り返されない、非 IID、非定常）と対策の概観。
  - ![ContSSL Fig2](https://arxiv.org/html/2203.12710v2/overview.png)

### E2. Learning from One Continuous Video Stream
- 著者: João Carreira et al.
- 採録: **CVPR 2024**（CVF Open Access + arXiv comment で確認）
- arXiv: https://arxiv.org/abs/2312.00598
- **方法と評価**
  - ミニバッチもaugmentationもシャッフルもなしに、1本の連続した動画ストリームからオンラインで学習する枠組み。
  - 評価は、**ストリーム内での適応（in-stream）**と、**別ストリームへの汎化（out-of-stream）**の2軸。ScanNet / Ego4D からストリームを作り、ピクセルからピクセルへの予測（未来予測・深度・セグメンテーション）で統一的に評価した。
  - 連続ストリームでは連続する勾配同士のコサイン類似度が強く相関することを示し、モメンタムが悪影響を与えること、重み更新のペースが重要なことを報告した。
  - 事前学習中の Kinetics linear 精度を、未来予測の時間的なずれ幅ごとに比べ、ずれが長いほど良いという結果も出している。
- **この研究での再利用と限界**
  - ストリーミング学習の評価を **「適応」と「汎化」の2軸**で書くという枠組みは、そのまま使える。また、**未来予測の時間的なずれが長いほど表現が良い**という結果は、4フレームの間隔（stride）を広げる動機付けになる。
  - 「モメンタムが悪影響」という報告は、MoCo のモメンタムエンコーダとの関係で要注意（ただし、この報告はオプティマイザのモメンタムの話であり、MoCo の EMA とは別物）。
  - 限界: ピクセル予測の枠組みで、対照学習ではない。
- **図**
  - Fig. 1: 上段は単一の連続ストリームでの学習（IID学習との違い）。下段は、ピクセルからピクセルへのモデルで適応（in-stream）と汎化（out-of-stream）を測る評価の枠組み。
  - ![OneStream Fig1](https://arxiv.org/html/2312.00598v2/figures/ssl5.png)

---

## F. この研究向けの評価メニュー案（文献からの組み立て）

以下は文献から組み立てた**提案**であり、どれもまだ実施していない。

| # | 評価 | 元になった論文 | 何が分かるか | 注意点 |
|---|---|---|---|---|
| 1 | ブロックごとの順列の線形 probe（4! または 12クラス）を、LoRA前後で比較 | Time Is MattEr, OPN | 順序情報がどの層に、どれだけ線形に入っているか | 凍結した ViT でも時間位置埋め込みから順序が解ける可能性がある。**トークンの位置を入れ替えるのか、内容を入れ替えるのか**を明確に区別すること |
| 2 | 同期（ずれ量）・前後判定の probe | Jenni+ 2020 | 時間の向きと時間差の情報 | 分類器の容量（線形か MLP か）を明記する |
| 3 | 相対変換（同じ / 逆再生 / 2倍速）の probe | TE, RSPNet | 方向と速さを、内容の影響を除いて測る | 同じ動画から作るので、内容の交絡が少ない |
| 4 | 4×4 のフレーム間コサイン類似度行列 | TCLR | 時間方向に特徴が均一化しているか | 区別できることは順序を理解していることとは別 |
| 5 | static clip（同じフレーム×4）を基準にした感度と、クラス別の散布図 | DSM, BE | 見た目だけで解けるクラスとの関係 | 4フレームでは、ほとんどのクラスが見た目で解ける恐れ |
| 6 | シャッフルしたクリップで学習した LoRA との比較 | MME | 学習が時間を使っているかの対照実験 | 学習コストが1回分増える |
| 7 | K400 型 vs SSv2 / Diving48 型の linear probe（LoRA あり・なし） | EVL, AIM, Only Time Can Tell | 時間が必要なタスクでの差 | ActivityNet での学習結果が SSv2 に転移するとは限らない |
| 8 | 時間を使わないベースライン（1フレーム、フレーム平均）を必ず併記 | ATP | 評価の目盛り | — |
| 9 | 動きの大きさで層別した集計 | Shuffle and Learn, SpeedNet | 動きのある区間でも鈍いのか | 圧縮ノイズなどの shortcut に注意 |

---

## 注意点・確認できなかったこと

- **採録先が未確認のもの**
  - Side4Video（arXiv の技術報告のみ）
  - TCLR の CVIU の巻・号・年
  - Uncovering Statistics の TPAMI 掲載年
  - Dynamic Image の拡張版（arXiv 1612.00738）の掲載誌
- **arXiv 版が見つからなかったもの**（タイトル検索で該当なし）: Arrow of Time（CVPR 2018）、VCOP（CVPR 2019）、CORP（ICCV 2021）、Modeling Video Evolution（CVPR 2015 版）。これらは CVF のページで採録を確認したが、図は掲載していない。
- 依頼文の「PRP pace prediction (ECCV 2020)」は**別々の2本**として扱った（PRP ＝ Yao らの CVPR 2020、Pace Prediction ＝ Wang らの ECCV 2020）。
- Time Is MattEr は ICML 2022 で、指定された会議リストの外だが、評価方法がこの研究に最も近いので含めた。
- 認識タスク用の LoRA が時間を表現できるかを専門に分析した論文は、この調査では確認できなかった。近い議論として ZeroI2V（統合可能な線形アダプタ + attention の変更）と MotionDirector（生成モデルの dual LoRA）を載せた。
- SEVERE は arXiv HTML 版の本文がほぼ取得できなかったため、アブストラクトの範囲のみで記述した。
- 図の URL は 2026-11-12 時点で HTTP 200 かつ image/png であることを確認した（計34枚、21本の論文）。arXiv HTML の版番号が変わると URL が変わる可能性がある。
