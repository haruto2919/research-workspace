# Track 4: 生成・系列潜在モデルにおける static/dynamic（content/motion）分離と、動画拡散の motion/appearance LoRA・アダプタ — 「分離をどう評価しているか」の調査

調査日: 2026-11-12
対象: 2017–2026（ICLR/ICML/NeurIPS/CVPR/ICCV/ECCV ほか）
収録: 27本（系列分離・一般分離指標 13本 + 動画拡散の motion カスタマイズ/LoRA 分離 14本）
図: 18本の論文について、arXiv HTML の画像URLを33枚 `curl -sI` で確認済み（すべて HTTP 200、content-type image/png または image/jpeg）

## 0. この調査の読み方（研究文脈との対応）

- 学生側の設定: 画像事前学習 ViT の attention Q/V に LoRA を入れ、4フレームクリップでストリーミング MoCo（識別的な対照学習）を行う。知りたいのは、LoRA が **static（見た目・シーン・ID）とは別に dynamic（動き・順序・速度）** を学習したかどうか。
- ここで扱う論文はほとんどが **生成モデル**（VAE/GAN/拡散）で、評価は「潜在を入れ替えて**デコード**し、学習済み判定器（judge）で判定する」形をとる。学生のモデルにはデコーダがないため、そのまま使える評価は少ない。本文では各論文ごとに、**入力側を入れ替える / 特徴に probe をかける / 特徴の差分を見る** といった識別モデル向けの言い換えと、その限界を書いた。
- 会場（venue）の確認方法は各論文に書いた（OpenReview / CVF / PMLR / ECVA / proceedings / arXiv comments）。確認できなかった点は「未確認」と明記した。

---

## 1. 標準評価プロトコルのまとめ（先に全体像）

### 1.1 系列分離（sequential disentanglement）の標準プロトコル

系列 $x_{1:T}$ を static 潜在 $s$（$f$ や $z_f$ と書く論文もある）と dynamic 潜在 $z_{1:T}$（または $d_{1:T}$）に分ける。

1. **条件付き生成（fix-one, sample-the-other）**: $s$ を固定して $z_{1:T}$ を事前分布から引き直す（またはその逆）。生成した系列 $\tilde{x}$ を、ラベル付き実データで**教師あり学習した判定器 $C$（judge）**に入れ、固定した側の因子（例: Sprites の動作、MUG の表情）が保たれているかを判定する（DSVAE → S3VAE → C-DSVAE → SPYL → DBSE と受け継がれた形）。
2. **スワップ（swap）**: 2本の系列 a, b から $\mathrm{dec}(s^a, z^b_{1:T})$ を作る。結果は「a の見た目で b の動き」になるはず。定性的に示す論文が多いが、2025年の MSD ベンチマークでは判定器か VLM で定量化している（2-Swap）。
3. **指標**（判定器の出力 $p(y|x)$ から計算）
   - **Acc**: 固定した因子のラベルが判定器の予測と一致する割合（高いほど良い）。
   - **IS** $=\exp\big(\mathbb{E}_x\,\mathrm{KL}[p(y|x)\,\|\,p(y)]\big)$（高いほど良い。生成が判定器にとって明確で、かつ多様）。
   - **Intra-entropy** $H(y|x)=-\sum_y p(y|x)\log p(y|x)$ のサンプル平均（低いほど判定器が自信をもって分類できている）。
   - **Inter-entropy** $H(y)=-\sum_y p(y)\log p(y)$, $p(y)=\frac1N\sum_i p(y|x_i)$（高いほど多様）。
   - **EER（音声 TIMIT）**: 話者照合。$s$ どうし（または時間平均した $z$ どうし）のコサイン類似度をしきい値で判定し、誤拒否率＝誤受理率となる点の値。分離がうまくいっていれば、static EER は低く、dynamic EER は高くなる。
4. **潜在の分類（latent classification / probe）**: SPYL 以降で定着した。テスト集合の $s$ と $z$ を取り出して 80/20 に分け、「$s$→static ラベル」「$s$→dynamic ラベル」「$z$→static ラベル」「$z$→dynamic ラベル」の4つの分類器（SVM/RF/KNN）を学習する。理想は「$s$→static は完全に当たり、$s$→dynamic はチャンスレベル」。DBSE はこれを **Leakage Gap**（＝本来の因子の正解率 − 漏れ因子の正解率）としてまとめた。
5. **データセット**: Sprites（キャラクタ。動作3種×向き3種＝9クラスが dynamic、肌・上着・ズボン・髪の色 各6種が static。8フレーム）、MUG（顔表情6種が dynamic、人物IDが static。15フレーム）、SM-MNIST（数字が static、動きは連続値）、TIMIT（話者が static、言語内容が dynamic。200ms＝20フレームの区間）。後に Jester（手ジェスチャ）、Letters、時系列（PhysioNet、Air Quality）が加わり、2025年には dSprites/3D Shapes の系列版、dMelodies-WAV、VoxCeleb の VLM ラベル版が入った。
6. **相互情報量（MI）**: S3VAE と C-DSVAE は $I(s;z_{1:T})$ を最小化し、$I(s;x)$ と $I(z;x)$ を最大化する。ただし**評価指標**としては MI を直接使わず、上の probe/判定器で代わりに測ることが多い。DBSE は「情報漏れ」を $I_q(s,X_d)>0$ または $I_q(d,X_s)>0$ と定義したうえで、実際には probe の正解率で測っている。

### 1.2 一般の分離指標（非系列、Locatello et al. ICML 2019 が整理したもの）

- **BetaVAE score**: ある1因子を固定したサンプル対の潜在差から、どの因子を固定したかを線形分類器で当てる正解率。
- **FactorVAE score**: 上の改良版。正規化した潜在の分散が最小の次元を使い、多数決分類器で判定する。
- **MIG（Mutual Information Gap）**: 各因子について、相互情報量が1位の潜在次元と2位の潜在次元の差を正規化したもの。
- **DCI Disentanglement**: 因子を予測するときの各潜在次元の重要度を正規化し、そのエントロピーから計算する（Completeness / Informativeness と組で使う）。
- そのほか Modularity と SAP。
- Locatello らの要点は2つ。(1) 帰納バイアスも教師もなしに分離を同定することは理論上できない。(2) 指標どうしの相関はデータセットによって変わる（BetaVAE と FactorVAE、MIG と DCI は互いに似ている）。
- 学生への含意: **正解の因子ラベル（static/dynamic）がある評価データ**が要る。

### 1.3 動画拡散の motion カスタマイズの標準評価

- **テキスト整合（CLIP-T / Text Align.）**: 生成フレームと「新しい見た目」を書いたプロンプトの CLIP 類似度の平均。参照動画の見た目に引きずられていないかを間接的に測る（MotionDirector はこれを appearance diversity と呼ぶ）。
- **時間的一貫性**: 連続フレーム間の CLIP（または DINO）特徴のコサイン類似度の平均。
- **Motion Fidelity Score（Yatim et al. CVPR 2024）**: CoTracker で入力動画と出力動画のトラックレット集合を求め、Chamfer 型の最近傍相関で比べる。ピクセル位置が揃っていない動画どうしでも動きを比べられる。式は §3 の Yatim の項に記載。
- **Hybrid motion fidelity（DeT, ICCV 2025）**: 軌跡の Fréchet 距離（点の**順序**も考慮する）と、速度方向のコサインを混ぜたもの。
- **Appearance leakage の直接測定（Separate Motion from Appearance, 2025）**: 参照動画の被写体を表すテキスト（例 "a person"）と生成動画の CLIP 類似度を **App. Leak**（低いほど良い）とし、動詞を除いた新プロンプトとの類似度を **App. Align.**（高いほど良い）、既製の動作分類器の正解率を **Motion** とする。Q/K/V/FFN のどれに LoRA を入れたかで比較している。
- **ユーザースタディ**: motion fidelity（参照と動きが同じか）、appearance diversity（参照の見た目に縛られていないか）、temporal consistency、text alignment を、2手法比較または5段階評価で問う。
- **アブレーション**: spatial（見た目）側のパスや absorber を外すとどうなるか、LoRA をどの層・どの射影に入れるか、を比べる。

---

## 2. 系列分離（static/dynamic）と一般分離指標

### 2.1 MCnet: Decomposing Motion and Content for Natural Video Sequence Prediction
- 著者: Ruben Villegas et al.
- 会場: **ICLR 2017**（arXiv comment「International Conference on Learning Representations (ICLR) 2017」で確認）
- arXiv: https://arxiv.org/abs/1706.08033
- 方法と評価: 動画の未来フレーム予測で、**motion encoder にはフレーム差分 $x_t-x_{t-1}$ の系列**を、**content encoder には最後のフレーム1枚**を入力し、2経路を合わせて次フレームを生成する。評価は主に予測フレームの PSNR/SSIM（KTH, Weizmann, UCF-101）で、分離そのものを直接測る指標は使っていない。付録ではカメラが大きく動く場合を分析している。
- 学生への翻訳: 「動き＝フレーム差分」という帰納バイアスは、ViT 特徴の差分 $\Delta f_t=f_t-f_{t-1}$（LoRA あり/なしそれぞれ）を dynamic の代理として probe する、という形で使える。限界として、カメラが動くと差分は背景の見た目も含んでしまい、4フレームでは差分が3つしかない。

### 2.2 DRNet: Unsupervised Learning of Disentangled Representations from Video
- 著者: Emily Denton, Vighnesh Birodkar
- 会場: **NeurIPS (NIPS) 2017**（proceedings.neurips.cc の 2017 hash ページで確認）
- arXiv: https://arxiv.org/abs/1705.10915
- 方法と評価: 各フレームを content（クリップ内でほぼ一定）と pose（時間変化）の2つのエンコーダで表す。**scene discriminator** が「2つの pose ベクトルが同じ動画から来たか」を当てようとし、pose encoder はその判定をできなくするように学習する（敵対的損失）。pose に動画固有の見た目情報が残らないようにするための仕組み。評価は3つある。(1) NORB のグリッド（行から pose、列から content を取って生成）。(2) content 特徴と pose 特徴それぞれに分類器を学習する（敵対項があると content 特徴で物体IDがよく当たる）。(3) KTH で pose 系列だけから少数ショットで行動分類する。
- 学生への翻訳: scene discriminator は、そのまま**事後の漏れテスト**として使える。LoRA 由来の特徴（例: LoRA あり特徴 − LoRA なし特徴）のペアから「同じ動画か」を当てる小さな分類器を学習し、正解率がチャンスを大きく超えれば static（動画ID）の情報が漏れている。限界として、MoCo は同じ動画のクリップを正例にするので、学習目的そのものが「同じ動画か」を当てる方向に働く。漏れが出やすいのは前提として受け止め、**LoRA なし ViT との比較**で読む必要がある。
- 図（検証済み）:
  - Fig. 4: https://arxiv.org/html/1705.10915v1/norb_all.png
    - 図の内容: 左は NORB での分解例。グリッドの各画像は「上の行の画像の pose」と「左の列の画像の content」を組み合わせて生成したもの。中央は敵対項なしで学習した場合で、pose ベクトルに content が入り込み、生成が pose 側に引っ張られる。右は比較手法（Mathieu et al.）。
    - 読み方: 行方向に見て物体が同じ、列方向に見て向きが同じなら、うまく分離できている。

### 2.3 FHVAE: Unsupervised Learning of Disentangled and Interpretable Representations from Sequential Data
- 著者: Wei-Ning Hsu et al.
- 会場: **NIPS 2017**（arXiv comment「Accepted to NIPS 2017」で確認。proceedings ページは未参照）
- arXiv: https://arxiv.org/abs/1709.07902
- 方法と評価: 音声を対象に、系列レベルの潜在 $z_2$（話者・雑音など）と区間レベルの潜在 $z_1$（音素など）を階層 VAE で分ける。評価は2つ。(1) 系列レベルの平均 **s-vector** で話者照合を行い、**EER** を i-vector と比べる。(2) $z_1$ を使った雑音条件の異なる ASR で単語誤り率を下げられるかを見る。後の DSVAE と C-DSVAE が使う TIMIT の EER プロトコルの元になった論文。
- 学生への翻訳: 「同じ動画か」の照合 EER は、デコーダなしで計算できる**最も素直な指標**。クリップ特徴のコサイン類似度で同一動画（あるいは同一シーンや同一人物）を照合し、static 用の特徴は EER が低く、dynamic 用の特徴（LoRA の残差など）は EER が高い、という形で評価できる。限界として、学生のモデルには static 特徴と dynamic 特徴の明示的な分割がないので、どの特徴を比べるか（LoRA なし特徴 / LoRA あり特徴 / 差分 / 時間平均を引いた残差）を自分で決める必要がある。

### 2.4 MoCoGAN: Decomposing Motion and Content for Video Generation
- 著者: Sergey Tulyakov et al.
- 会場: **CVPR 2018**（CVF Open Access で確認）
- arXiv: https://arxiv.org/abs/1707.04993（arXiv HTML 変換なし、ar5iv も変換失敗）
- 方法と評価: 動画の潜在を「動画全体で固定の content $z_C$」と「RNN が生成する motion 系列 $z_M^{(t)}$」に分け、画像判別器と動画判別器の両方で学習する GAN。評価指標は2つ。**ACD（Average Content Distance）**は生成動画のフレーム間で content 特徴（顔は OpenFace の特徴、形状は平均色）の平均ペアワイズ L2 距離をとったもので、小さいほど同一人物・同一物体が保たれている。**MCS（Motion Control Score）**は、時空間 CNN の行動分類器が、条件として指定した行動を生成動画から正しく認識できる割合。ユーザー選好テストも行っている。
- 学生への翻訳: ACD は「クリップ内で static 特徴が一定か」の指標として、そのまま特徴空間で計算できる（フレームごとの特徴のクリップ内ペアワイズ距離）。LoRA あり特徴が ViT 単体より大きくクリップ内で変化していれば、時間変化に反応する成分が増えたと解釈できる。限界として、ノイズや手ブレでも距離は増えるので、「動きによる変化」かどうかはフレーム順シャッフルなどの対照条件と比べる必要がある。

### 2.5 DSVAE: Disentangled Sequential Autoencoder
- 著者: Yingzhen Li, Stephan Mandt
- 会場: **ICML 2018**（PMLR v80, `yingzhen18a` で確認）
- arXiv: https://arxiv.org/abs/1803.02991
- 方法と評価: 系列全体で1つの static 潜在 $f$ と、フレームごとの dynamic 潜在 $z_t$（確率的 RNN 事前分布）をもつ VAE。$z_t$ の次元を小さくして static 情報が入りにくくしている。評価は3種類。
  1. Sprites で、$f$ を推論値に固定して $z_t$ をランダムに引き直す（逆も行う）。判定器の予測が元フレームと一致しない割合（disagreement）と $\mathrm{KL}[p_{recon}\|p_{data}]$ を報告し、一様分布との KL と比べて有意かを示す。
  2. 同じ $f$ で 200 系列を生成し、属性分類のフレームごとの推移を見て、static 属性が時間方向に保たれるかを確認する。
  3. TIMIT で $\mu_f$ と時間平均した $\mu_z$ による話者照合 EER を比べる。
- 学生への翻訳: (2) の「判定の時間推移」は、フレームごとの ViT 特徴に static probe（シーン・ID の分類器）をかけ、LoRA を入れても 4 フレームで予測が一定かを見る、という形で使える。(3) は 2.3 の EER と同じ考え方。限界として、判定器は生成画像ではなく実クリップの特徴に対して学習することになるので、「判定器が分離を見ている」のではなく「特徴が判定器にとって線形に読める」ことを測る評価になる。

### 2.6 S3VAE: Self-Supervised Sequential VAE for Representation Disentanglement and Data Generation
- 著者: Yizhe Zhu et al.
- 会場: **CVPR 2020**（CVF Open Access と arXiv comment「to appear in CVPR2020」で確認）
- arXiv: https://arxiv.org/abs/2005.11437
- 方法: 正則化を3つ加える。
  1. **Static consistency**: **フレーム順をシャッフルした系列**の $z_f$ を正例、別の動画を負例とする triplet loss で、$z_f$ がフレーム順に依存しないようにする。
  2. **Dynamic factor prediction**: $z_t$ から、FlowNet2 の光学フローで最も動きの大きいグリッド位置（顔データでは顔ランドマークの距離）を予測させる。
  3. **MI 最小化**: $I(z_f;z_{1:T})$ を小さくする。
- 評価: 実動画2本の $z_f$ と $z_{1:T}$ を入れ替えるスワップ（定性）と、判定器による Acc/IS/$H(y|x)$/$H(y)$。SMMNIST は $z_f$ を固定して数字を判定し、MUG は $z_{1:T}$ を固定して表情を判定し、Sprites は両方を判定して平均する。アブレーションでは、動き予測の項を除くと Acc が大きく下がる。
- 学生への翻訳:
  - static consistency の考え方は、評価にそのまま使える。**同じクリップをシャッフル・逆再生したときの特徴の変化量**を測り、static 用の特徴は変わらず、LoRA が順序を学んでいれば LoRA 側の特徴が変わるはず。
  - 光学フローの最大位置を予測させるタスクは、**dynamic probe の教師ラベル**としてそのまま使える。
  - 限界として、ViT の [CLS] 特徴や平均プール特徴は構造上ほぼ順序不変になりうる。時間方向の位置埋め込みや時間方向の注意がない設計だと、順序の情報はそもそも特徴に入らない。構造を確認しておく必要がある。
- 図（検証済み）:
  - Fig. 2: https://arxiv.org/html/2005.11437v1/framework-crop.png
    - 図の内容: フレームごとのエンコーダ → LSTM → $z_f$ と $z_t$ → デコーダ、という全体構成に、3つの正則化（シャッフル triplet、フロー由来の擬似ラベル予測、MI 最小化）がどこにかかるかを示している。
  - Fig. 4: https://arxiv.org/html/2005.11437v1/all_swapping-crop.png
    - 図の内容: SMMNIST / Sprite / MUG で、外観動画 $V_a$ と動き動画 $V_m$ の表現を入れ替えた結果を DSVAE と MonkeyNet と比べたもの。
    - 読み方: 各列で「$V_a$ の見た目のまま $V_m$ の動き」になっているかを見る。DSVAE は数字が変わったりぼやけたりする。

### 2.7 R-WAE: Disentangled Recurrent Wasserstein Autoencoder
- 著者: Jun Han et al.
- 会場: **ICLR 2021**（arXiv comment「ICLR 2021」で確認）
- arXiv: https://arxiv.org/abs/2101.07496（arXiv HTML なし）
- 方法と評価: 系列 VAE の KL 正則化を Wasserstein 距離（MMD 版と GAN 版）に置き換え、static と dynamic の分離と MI の制御を図る。評価は DSVAE/S3VAE と同じ判定器ベースの Acc/IS/エントロピー（Sprites、MUG、SM-MNIST）と、TIMIT の EER（付録Cで定義）。C-DSVAE 以降の比較表に必ず出てくるベースライン。
- 学生への翻訳: 方法そのものは生成モデル向けなので直接は使えない。比較表を読むときの基準値（例: MUG Acc 約71%）として知っておけば足りる。

### 2.8 C-DSVAE: Contrastively Disentangled Sequential Variational Autoencoder
- 著者: Junwen Bai et al.
- 会場: **NeurIPS 2021**（arXiv comment「Accepted by NeurIPS 2021」で確認）
- arXiv: https://arxiv.org/abs/2110.12091
- 方法: ELBO に MI 項 $+I_q(s;x_{1:T})+I_q(z_{1:T};x_{1:T})-I_q(z_{1:T};s)$ を加え、MI の推定を**対照学習（InfoNCE）**で行う。正例は増強で作る。**content 増強＝フレーム順のシャッフル・逆順**（中身は同じで動きだけ変わる）を $s$ の正例にし、**motion 増強＝色の歪み・ノイズ・拡大**（動きは同じで見た目だけ変わる）を $z_{1:T}$ の正例にする。
- 評価: 判定器（judge）による Acc/IS/$H(y|x)$/$H(y)$（Sprites/MUG/SM-MNIST）と TIMIT の content EER（低いほど良い）・motion EER（高いほど良い）。Sprites の Acc は 99.99%、MUG の Acc は 81.16% で、これが当時の SOTA。
- 学生への翻訳: 学生の MoCo と構造がいちばん近い論文。「どの増強で正例を作るかが、どの因子に不変になるかを決める」という対応関係は、**評価用のクリップ対の設計**にそのまま使える。
  - 「同じ見た目・違う順序（シャッフル/逆再生）」の対では、static 用の特徴は近く、dynamic 用の特徴は遠いはず。
  - 「違う見た目（色の変換など）・同じ動き」の対では、その逆になるはず。
  - LoRA あり/なしのそれぞれで、この2種類の対の距離比を測る。
  - 限界として、色の変換は static の一部（色）しか変えず、シーンや人物の変化は再現できない。
- 図（検証済み）:
  - Fig. 2（左・右パネル）: https://arxiv.org/html/2110.12091v1/figs/smmnist_aug.png 、 https://arxiv.org/html/2110.12091v1/figs/sprite_aug.png
    - 図の内容: 1行目が元の系列、2行目が content 増強（順序を逆転・ランダム化）、3行目が motion 増強（引き伸ばしと色強調、または色の歪みとガウスノイズ）。
    - 読み方: 2行目は「中身が同じで動きが違う」、3行目は「動きが同じで見た目が違う」。MoCo の評価用対照ペアの作り方の見本として読める。
  - Fig. 3（左・右パネル）: https://arxiv.org/html/2110.12091v1/figs/sprites_fixed_content.png 、 https://arxiv.org/html/2110.12091v1/figs/sprites_fixed_motion.png
    - 図の内容: Sprites で static と dynamic の一方を固定して他方を入れ替えた生成。本文によれば、左は $s$ を固定した場合（キャラクタIDが保たれる）、右は $z$ を固定した場合（動きが保たれる）。
    - 読み方: 1・2・4行目が入力、3・5行目が操作後の生成。

### 2.9 SPYL: Sample and Predict Your Latent: Modality-free Sequential Disentanglement via Contrastive Estimation
- 著者: Ilan Naiman et al.（第1・第2著者が同等貢献）
- 会場: **ICML 2023**（PMLR v202 `naiman23a` と arXiv comment で確認）
- arXiv: https://arxiv.org/abs/2305.15924
- 方法: C-DSVAE がモダリティ固有の増強に頼っていた点を改め、**VAE の事後分布から**正例・負例を作る。KL 距離でバッチ内の近い分布と遠い分布を選んでサンプリングし、さらに予測サンプリングの工夫を加える。
- 評価:
  - 従来の判定器プロトコル（Acc/IS/$H$、EER）に加えて、**潜在分類ベンチマーク**を提案した。$s$ と $d_{1:T}$ それぞれから static ラベルと dynamic ラベルを当てる4つの分類器を学習する（SVM、RF/KNN でも頑健性を確認。MUG/Letters/Jester）。
  - 正例ビューの質も、「static 正例の $s^+$ で static ラベルは当たり、dynamic ラベルはチャンス（MUG では 16.66%）に近いか」で評価している。
- 学生への翻訳: **4分類器 probe は、デコーダのない学生のモデルにいちばん移しやすい評価**。
  - 特徴 A（LoRA なし ViT のクリップ特徴）、特徴 B（LoRA あり）、特徴 C（B − A の残差、または時間平均を引いたフレーム特徴）を用意する。
  - それぞれから static ラベル（シーン・人物）と dynamic ラベル（順序・速度・行動）を線形 probe で当てる。
  - 「C は dynamic で高く static でチャンスに近い」なら、LoRA が見た目とは別に動きを学んだことの証拠になる。
  - 限界として、線形で読めないことは「情報がない」ことを意味しない（非線形 probe も併用する）。また static ラベルと dynamic ラベルがデータ上で相関していると（例: 特定のシーンでしか起きない行動）、漏れと区別できない。
- 図（検証済み）:
  - Fig. 3: https://arxiv.org/html/2305.15924v1/views_analysis_lr.png
    - 図の内容: (A) MUG に SimCLR の増強をかけたビューの例（色や切り抜きは変わるが、表情と人物はそのまま）。(B) 「static 正例」の分類正解率をエポックごとに示したもので、濃い線（static ラベル、↑）は高く、薄い線（dynamic ラベル、↓）はチャンスに近いほど良い。C-DSVAE は dynamic 側が約70%と高く、漏れが大きい。(C) 「dynamic 正例」の場合の逆の検査。
    - 読み方: 「当たるべきラベルは高く、当たってはいけないラベルはチャンスに近く」という2本の線の**差**を見る。
  - Fig. 2: https://arxiv.org/html/2305.15924v1/swap_sprites_mug_lr.png
    - 図の内容: Sprites と MUG のスワップ。4枚1組で、1 が source、2 が target、3 が content（static）のスワップ、4 が pose（dynamic）のスワップ。

### 2.10 SKD: Multifactor Sequential Disentanglement via Structured Koopman Autoencoders
- 著者: Nimrod Berman et al.（第1・第2著者が同等貢献）
- 会場: **ICLR 2023（notable top-25%, Spotlight）**（arXiv comment で確認）
- arXiv: https://arxiv.org/abs/2303.17264
- 方法: 潜在の時間発展を線形の **Koopman 行列 $C$** で近似する（$C=Z_p^+Z_f$）。**固有値が1の固有ベクトルは時間不変＝static**、それ以外は dynamic とみなせる、という観察に基づき、固有値に対するスペクトル罰則（static 固有値を1に近づけ、dynamic 固有値を1から離す）をかける。固有ベクトルごとに static を細かく分けられるので、2因子を超える多因子分離ができる。
- 評価: 固有部分空間ごとの係数を入れ替える「factorial swap」（髪・肌・上着の色を1つずつ入れ替え、判定器でどの因子が変わったかを正解率で測る）、static 部分空間の t-SNE（36通りの組み合わせが格子状に分かれる）、MUG で dynamic 部分空間を段階的に入れ替える incremental swap、そして2因子の標準ベンチマーク（Acc/IS/EER）。
- 学生への翻訳: 4フレームの特徴系列 $f_{1:4}$（パッチごとの特徴、または [CLS]）に最小二乗で線形遷移 $C$ を当てはめ、**固有値が1付近のモードに乗るエネルギーの割合**を LoRA あり/なしで比べられる。LoRA で dynamic モードの割合が増えれば、時間変化をとらえる成分が増えたと読める。限界として、4フレームでは遷移の組が3つしかなく、$C$ の推定が不安定になる。多数のクリップをまとめて当てはめるなどの工夫が要る。
- 図（検証済み）:
  - Fig. 1: https://arxiv.org/html/2303.17264v1/kae_scheme.png
    - 図の内容: エンコーダ → 潜在系列 $Z$ を過去 $Z_p$ と未来 $Z_f$ に分割 → 最小二乗で Koopman 行列 $C$ → デコーダ、という構成と、固有値へのスペクトル罰則 $\mathcal{L}_{eig}$。
  - Fig. 2: https://arxiv.org/html/2303.17264v1/sprites_factorial.png
    - 図の内容: source のキャラクタの static 因子を、target のものへ1つずつ（髪 → 肌 → 上着の順など）入れ替えていく様子。
    - 読み方: 1段ごとに、指定した因子だけが変わり、動きは変わらないかを見る。

### 2.11 DBSE: Sequential Disentanglement by Extracting Static Information From A Single Sequence Element
- 著者: Nimrod Berman et al.（第1〜4著者が同等貢献）
- 会場: **ICML 2024**（PMLR v235 `berman24a` と arXiv comment で確認）
- arXiv: https://arxiv.org/abs/2406.18131
- 方法: static は時間不変なので **1フレームから取り出せる**という仮定を置く。static 事後分布を $g_1$（1フレームの特徴）だけに条件付け、dynamic 経路には **$g_{2:T}-g_1$（1フレーム目を引いた差分）**を LSTM に入れる。この「引き算の帰納バイアス」で情報漏れを抑えるので、MI 項がいらない。
- 評価:
  - 従来の判定器プロトコル（MUG Acc 86.90%）。
  - **Leakage Gap**: 生成版と潜在版がある。
    - 生成版: static を引き直したときは「dynamic の正解率は高く、static の正解率はランダムに近い」はずで、その差を測る。dynamic を引き直した場合も同様に測る。
    - 潜在版: SPYL 型の probe で、本来の因子の正解率 − 漏れ因子の正解率を測る（MUG の static 特徴で 29.37% → 54.29%）。
  - t-SNE と時系列の下流タスクでも評価している。
- 学生への翻訳: 「1フレーム目を引く」は、学生の評価に**そのまま使える特徴変換**になる。クリップの各フレームの ViT 特徴 $g_t$ について $g_t-g_1$（または $g_t-\bar g$）を作り、そこに static probe と dynamic probe をかける。差分特徴で static が当たらず dynamic が当たるなら分離が成り立っていて、LoRA がその差を広げるかを比べる。Leakage Gap $=\mathrm{Acc}_{\text{target}}-\mathrm{Acc}_{\text{leak}}$ は報告用の1つの数値として使いやすい。限界として、ViT の特徴は線形に足し引きできるとは限らず、差分にも見た目に依存する成分（見た目に依存する動きの大きさなど）が残る。
- 図（検証済み）:
  - Fig. 2: https://arxiv.org/html/2406.18131v1/tsne_mug_qual_reb.png
    - 図の内容: MUG の static 潜在を人物IDで色分けした t-SNE（左）と、dynamic 潜在を表情で色分けした t-SNE（右）。
    - 読み方: 左は ID ごと、右は表情ごとにまとまれば、それぞれの因子をとらえている。学生の場合は、残差特徴を「シーンIDで色分け」と「順序/速度で色分け」の2通りで描けば同じ見方ができる。
  - Fig. 4: https://arxiv.org/html/2406.18131v1/mug_swaps_lr.png
    - 図の内容: A が source、B が target、C が static を入れ替えた結果、D が dynamic を入れ替えた結果。

### 2.12 MSD Benchmark: Disentanglement Beyond Static vs. Dynamic: A Benchmark and Evaluation Framework for Multi-Factor Sequential Representations
- 著者: Tal Barami et al.
- 会場: **NeurIPS 2025 Datasets and Benchmarks Track**（arXiv comment と papers.nips.cc 2025 の D&B ページで確認）
- arXiv: https://arxiv.org/abs/2510.17313
- 方法と評価: static/dynamic の2因子を超える**多因子**の系列分離ベンチマーク。動画・音声・時系列の6データセットを含む。10の指標を3群に分けている。
  1. **2-Swap / 2-GSample / M-Swap / M-GSample**: 入れ替えや引き直しの結果を判定器か **VLM** で判定する。
  2. **DCI-M/C/E**: 潜在の軌跡を平坦化して系列向けに適用する。
  3. 動画専用の一貫性指標 **C-Swap / C-Sample / GC-Sample**: static 属性が系列全体で保たれるかを見る。
  さらに、ラベルを使って潜在次元と因子を事後的に対応づける **Latent Exploration Stage**（予測器ベースとスワップベース）と、実データ（VoxCeleb）に VLM で自動ラベルを付けて判定させる仕組みを提案している。VLM 判定は教師あり評価と高い順位一致を示した。
- 学生への翻訳: 実動画には static/dynamic の正解ラベルがないことが多いが、**VLM でシーン・人物属性・動作・カメラ動作をタグ付けして probe の教師にする**という方針が使える。DCI は、学生の特徴（例えば LoRA 残差の各次元）がどの因子に効いているかを見る道具として、デコーダなしで計算できる。限界として、VLM のラベルは照明や服の色などで不正確になり（論文の検証でも 0.63〜0.72 程度）、速度や順序のような細かい dynamic 因子をどこまで正しく付けられるかは未検証。
- 図（検証済み）:
  - Fig. 1: https://arxiv.org/html/2510.17313v3/pipeline_lr.png
    - 図の内容: データセット選択 → （ラベルがなければ）VLM タガー → 教師なし学習 → Latent Exploration（予測器かスワップで潜在と因子を対応づける）→ 評価（Swap/GSample を判定器か VLM で判定、C-Swap などの一貫性、DCI）の流れ全体。
    - 読み方: 下段の「Evaluation Metrics」が評価の3つの群に対応する。学生にとっては「判定器を VLM で置き換える」部分と「DCI」部分が参考になる。
  - Fig. 5: https://arxiv.org/html/2510.17313v3/ranking_correlation_mean.png
    - 図の内容: VLM による評価と、教師あり判定器による評価のモデル順位の一致。

### 2.13 Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations
- 著者: Francesco Locatello et al.
- 会場: **ICML 2019**（PMLR v97 `locatello19a` で確認）
- arXiv: https://arxiv.org/abs/1811.12359
- 方法と評価: 12,000 以上のモデルを学習し、BetaVAE/FactorVAE/MIG/Modularity/DCI/SAP を比較した大規模研究（定義は §1.2）。結論は3つ。(1) 帰納バイアスなしの教師なし分離は理論上同定できない。(2) 分離がよいことが下流タスクの少サンプル学習の効率に結びつくとは限らない。(3) 教師なしのスコアではモデル選択ができず、乱数シードとハイパーパラメータの影響が大きい。
- 学生への翻訳: 分離を主張するには、**ラベル付きの評価データ（static と dynamic を別々に操作したクリップ）**と**複数シードでのばらつき**の報告が必要になる。限界として、これらの指標は因子ごとの正解値があることを前提にしており、自然動画ではそのまま使えない。合成データか VLM ラベル（2.12）で補う必要がある。

---

## 3. 動画拡散の motion カスタマイズと LoRA / アダプタによる見た目と動きの分離

### 3.1 AnimateDiff（MotionLoRA）
- 著者: Yuwei Guo et al.
- 会場: **ICLR 2024**（OpenReview の PDF 1ページ目に「Published as a conference paper at ICLR 2024」とあることを確認。Spotlight は公式 GitHub の表記によるもので、OpenReview の forum では未確認）
- arXiv: https://arxiv.org/abs/2307.04725
- 方法:
  - T2I モデルに**時間方向の motion module**を足して動画化する。
  - 学習時は、動画データの見た目の偏りを**domain adapter**（空間 self/cross-attention に入れる LoRA。**静止フレームで学習**）に吸収させ、推論時には外す（または強さ $\alpha$ を下げる）。
  - **MotionLoRA** は motion module の self-attention に入れる LoRA で、ズーム・パン・回転などのカメラ動作をクロップのルールベース増強で作った 20〜50 本の参照動画から学習する。
- 評価: ユーザースタディ（平均順位）と CLIP 指標の2本立て。CLIP 指標は、テキスト整合、ドメイン類似度（パーソナライズ T2I の参照画像との CLIP 類似度）、滑らかさの3つ。domain adapter の強さを 1 から 0 まで変えるアブレーションも行っている。
- 学生への翻訳:
  - 「静止フレームで学習した LoRA に見た目を吸収させ、動き用の LoRA には時間方向だけを学ばせる」という分担は、**static 専用の対照モデル（1フレームを複製した静止クリップで学習）**との比較の基準として使える。
  - zoom-in と zoom-out は**同じフレームを逆順にしたもの**なので、見た目を完全に揃えたまま**順序だけを変えるテスト**になる。LoRA 特徴でズームの方向（＝順序）が当たるかを確かめられる。
  - 限界として、クロップで作った合成の動きは実世界の物体の動きと分布が違う。
- 図: 図版は x1.png などの汎用名で、本文中の位置との対応を確かめられなかったため掲載を見送った。

### 3.2 MotionDirector: Motion Customization of Text-to-Video Diffusion Models
- 著者: Rui Zhao et al.
- 会場: **ECCV 2024（Oral）**（ECVA の論文ページと ECCV 2024 virtual oral ページで確認）
- arXiv: https://arxiv.org/abs/2310.08465
- 方法:
  - **Dual-path LoRA**: spatial path では、spatial transformer（self-attention と FFN。cross-attention には入れない）の LoRA を**ランダムに選んだ1フレーム**で学習し、見た目を覚えさせる。temporal path では、spatial LoRA を共有したまま凍結し、temporal transformer の LoRA だけを全フレームで学習する。推論時は spatial LoRA を外す。
  - **Appearance-debiased temporal loss**: 同じ動画のフレームのノイズに対して $\phi(\epsilon_i)=\sqrt{\beta^2+1}\,\epsilon_i-\beta\,\epsilon_{anchor}$ とアンカーフレームを差し引き（$\beta=1$、アンカーはランダムなフレーム）、見た目のオフセットを除いた損失を加える。
- 評価:
  - UCF Sports Action（12種の動作、95本）と LOVEU-TGVE-2023（76本、532プロンプト）を使う。
  - 自動指標: appearance diversity（多様なプロンプトと全フレームの CLIP 類似度）、時間一貫性（フレーム間の CLIP 類似度）、PickScore。
  - MTurk の2手法比較（1本あたり5人）: appearance diversity / temporal consistency / motion fidelity（参照動画と動きが同じか）。
  - アブレーション: 結合 LoRA / dual-path / dual-path＋debias loss。結合した LoRA は見た目の多様性を損なう。
- 学生への翻訳:
  - Fig. 4 の観察（**軌跡の形が動きを、軌跡の位置が見た目を表す**）は、学生の特徴空間でそのまま確かめられる。4フレーム分の特徴を PCA/t-SNE で線分として描き、「動き同じ・見た目違い」「動き違い・見た目同じ」の動画対を比べる。そのうえで、**各フレーム特徴からクリップ平均（またはアンカーフレーム）を引いた後**に動き同じの対が重なるかを見る。
  - 「1フレームで学習した LoRA＝見た目担当」は、2.11 の DBSE（1フレームから static を取る）と同じ考え方。
  - 限界として、この観察は拡散の潜在に対するもので、ViT の識別的な特徴でも同じ幾何になるかは確かめる必要がある。
- 図（検証済み）:
  - Fig. 3: https://arxiv.org/html/2310.08465v1/images/method.png
    - 図の内容: dual-path の構成。上が spatial path（spatial LoRA を1フレームで学習）、下が temporal path（spatial LoRA を共有・凍結し、temporal LoRA を学習）。
    - 読み方: 推論では spatial LoRA を捨て、temporal LoRA だけを残す。「見た目を別の LoRA に吸わせてから動きの LoRA を学習する」構造を示している。
  - Fig. 4: https://arxiv.org/html/2310.08465v1/images/loss.png
    - 図の内容: (a) 4本の車の動画の関係（1と2、3と4は動きが同じで見た目が違う。2と3は動きが違い見た目が同じ）。(b) 逆拡散過程の潜在の軌跡。(c) 時刻0の潜在を、フレーム順に線で結んだもの。(d) debias 後の潜在。
    - 読み方: (c) では、見た目が違うと点群の**位置**が離れ、動きが同じだと**線の形**が似ている。(d) で見た目のオフセットを除くと、動きの構造が残る。学生の特徴の可視化でそのまま真似できる。

### 3.3 DreamVideo: Composing Your Dream Videos with Customized Subject and Motion
- 著者: Yujie Wei et al.
- 会場: **CVPR 2024**（CVF Open Access で確認）
- arXiv: https://arxiv.org/abs/2312.04433
- 方法: 被写体は textual inversion と **identity adapter**（cross-attention に入れる）で、動きは temporal transformer の全層に入れる **motion adapter** で学習する。motion adapter には、学習動画から**ランダムに選んだ1フレームの CLIP 画像埋め込み**を「appearance guidance」として全フレームに与える。見た目は条件として与えられるので、adapter は見た目を覚える必要がなく、動きの学習に集中できる。
- 評価:
  - 被写体: CLIP-T（テキスト整合）、CLIP-I と DINO-I（被写体の類似度）。
  - 時間一貫性: 連続フレームの CLIP 類似度。
  - 動きのカスタマイズの自動評価は CLIP-T と時間一貫性だけで、動きの忠実度は主にユーザースタディで見ている。
  - 層ごとの**重み変化量**の分析（Fig. 4）。
- 学生への翻訳: appearance guidance の発想は、評価では「**見た目の情報（LoRA なし ViT の1フレーム特徴）を条件として与えた後に、LoRA 特徴が dynamic ラベルについて追加の予測力をもつか**」という条件付き probe として使える。具体的には、$[f_{\text{base}}(x_1), f_{\text{LoRA}}]$ と $f_{\text{base}}(x_1)$ 単独の probe 正解率の差を見る。層ごとの $\|\Delta W\|$ 分析は、Q/V LoRA のどの層が大きく更新されたかを調べる形でそのまま使える。限界として、重みの変化量は「何を学んだか」までは教えてくれない。
- 図（検証済み）:
  - Fig. 2: https://arxiv.org/html/2312.04433v1/framework.png
    - 図の内容: 2段階の学習。被写体学習（textual identity → ID adapter）と、動き学習（ランダムな1フレームの CLIP 埋め込みを appearance guidance として motion adapter に入れる）。
  - Fig. 4: https://arxiv.org/html/2312.04433v1/weight_change.png
    - 図の内容: 空間と時間の重みをすべて微調整したときの層ごとの重み変化量。被写体学習では cross-attention 層の変化が大きく、動き学習では層ごとの差が小さい。
    - 読み方: 学生の LoRA の層ごと・Q/V ごとの $\|BA\|$ を同じ形の図にすると比較しやすい。

### 3.4 VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models
- 著者: Hyeonho Jeong et al.
- 会場: **CVPR 2024**（CVF Open Access で確認）
- arXiv: https://arxiv.org/abs/2312.00845
- 方法:
  - **Motion distillation**: 連続フレームの残差 $\delta v^n=v^{n+1}-v^n$（潜在空間ではノイズの残差 $\delta\epsilon^n_t$）を「動きベクトル」とみなし、予測と正解の残差のコサイン類似度を最大化するように、**temporal attention 層だけ**を微調整する。
  - **Appearance-invariant prompt**: 背景や見た目の語を除いたプロンプト（例: "A bird is flying"）で学習する。
- 評価: CLIP によるテキスト整合とフレーム一貫性、27人のユーザースタディ（appearance diversity、つまり「入力動画の見た目にどれだけ縛られていないか」を含む）。微調整する層の比較（temporal attention と self/cross-attention）のアブレーションもある。
- 学生への翻訳: 残差のコサインは、**特徴の残差 $f_{t+1}-f_t$ の向き**として評価に使える。「見た目違い・動き同じ」の2クリップで、フレーム残差どうしのコサインが LoRA で上がるかを測る。微調整する層の比較は、学生の Q/V（spatial attention）の LoRA が temporal attention の LoRA と同じ役割を果たしうるかを考えるときの参照点になる。限界として、学生の ViT が**時間方向の注意をもたない**場合は、LoRA が時刻ごとに独立に作用するので、順序情報は MoCo の目的とクリップレベルの集約の仕方からしか入らない。
- 図（検証済み）:
  - Fig. 2: https://arxiv.org/html/2312.00845v1/figures/figure-overview.jpg
    - 図の内容: キーフレーム生成モデルの temporal attention だけを、フレーム残差（動きベクトル）を揃えるように学習し、その後の補間・超解像の段階につなぐ全体の流れ。

### 3.5 Customize-A-Video: One-Shot Motion Customization of Text-to-Video Diffusion Models
- 著者: Yixuan Ren et al.
- 会場: **ECCV 2024**（arXiv comment「Accepted by ECCV 2024」で確認）
- arXiv: https://arxiv.org/abs/2402.14780
- 方法:
  - **Appearance Absorber**: 1段目で、temporal 層を素通しにした T2V モデルの空間 attention に S-LoRA または textual inversion を付け、**順序をばらばらにしたフレーム**で学習して見た目を吸収させる。
  - **T-LoRA**: 2段目で、absorber を凍結したまま temporal attention の LoRA を学習する。推論時は absorber を外して T-LoRA だけを使う。
- 評価: CLIPScore（テキスト整合）、連続フレーム間の LPIPS（時間一貫性）、同じプロンプトで乱数を変えた生成どうしの LPIPS（多様性）、ユーザースタディ。アブレーションでは、LoRA を入れる attention の種類と absorber の種類（なし / S-LoRA / TextInv / Dual）を比べる。
- 学生への翻訳: 「順序をばらばらにしたフレームで見た目を先に吸収させる」は、評価ではシャッフルしたクリップと元のクリップで LoRA 特徴の差を見る、という操作に対応する（2.6 S3VAE と同じ）。LoRA を入れる attention の比較は、Q/V や層の位置の比較実験の設計の参考になる。限界として、生成モデルの「見た目の漏れ」は画像で見えるが、識別特徴の漏れは probe を通してしか見えない。
- 図（検証済み）:
  - Fig. 2: https://arxiv.org/html/2402.14780v3/archs.png
    - 図の内容: (1) temporal 層を素通しにして、空間 attention の absorber を順序のないフレームで学習する。(2) absorber を凍結して T-LoRA を学習する。(3) 推論では T-LoRA だけを使う。
  - Fig. 4: https://arxiv.org/html/2402.14780v3/ablations.png
    - 図の内容: 左は LoRA を入れる attention の比較で、S-LoRA は室内の家具や壁の飾りまで覚えてしまう。右は absorber の種類の比較で、absorber なしだと参照動画の見た目（眼鏡やロゴ）が残り、Dual absorber がいちばんきれいに見た目を除ける。
    - 読み方: 「動きの LoRA だけを学習すると見た目が漏れる」ことを目で見せる典型的な図。

### 3.6 DMT: Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer
- 著者: Danah Yatim et al.（CVPR virtual ページでは著者順が異なるが、CVF の論文ページでは Yatim が第1著者）
- 会場: **CVPR 2024**（CVF Open Access で確認）
- arXiv: https://arxiv.org/abs/2311.17009
- 方法: 学習なしの手法。T2V モデルの中間特徴 $f\in\mathbb{R}^{F\times M\times N\times D}$ を空間方向に平均した **SMM（Spatial Marginal Mean）**$=\frac{1}{MN}\sum_{i,j}f_{i,j}$ をフレームごとの記述子にする。SMM は物体の姿勢や位置・レイアウトを保ちつつ、見た目の細部には鈍い。それでも SMM そのものを合わせると見た目まで引きずられるので、**フレーム間のペアワイズ差 $\Delta_{(i,j)}=\phi_i-\phi_j$** を参照動画と揃える損失 $\sum_{i,j}\|\Delta_{(i,j)}-\tilde\Delta_{(i,j)}\|^2$ で生成を誘導する。
- 評価: **Motion Fidelity Score**（下の式）と、編集の忠実度（目標プロンプトとの CLIP 類似度）の2軸で比較する（54組、21本の動画）。
  - CoTracker で入力動画と出力動画のトラックレット集合 $\mathcal{T}$, $\tilde{\mathcal{T}}$ をとる。
  - スコア $=\frac1m\sum_{\tilde\tau}\max_{\tau}\mathrm{corr}(\tau,\tilde\tau)+\frac1n\sum_{\tau}\max_{\tilde\tau}\mathrm{corr}(\tau,\tilde\tau)$。
  - $\mathrm{corr}$ は、各フレームの変位ベクトル $(v^x_k,v^y_k)$ どうしのコサインのフレーム平均。
- 学生への翻訳:
  - **「平均プールした特徴のフレーム間差分行列」は、ViT でもそのまま作れる**（パッチ特徴の平均、または [CLS]）。4フレームなら $4\times4\times D$。この差分行列のクリップ間の類似度が、見た目違い・動き同じの対で LoRA によって上がるか、見た目同じ・順序違いの対で下がるかを測る。
  - Motion Fidelity Score は、**クリップ対の「動きの類似度」の外部の物差し**として使える。CoTracker で求めた動きの類似度と、LoRA 特徴の類似度の相関（Spearman）を、LoRA なし ViT や CLIP の見た目類似度と比べる。
  - 限界として、平均プールは空間の情報を落とすので局所的な動きに弱い。4フレームのトラックは短く、相関の推定はノイズが大きい。
- 図（検証済み）:
  - Fig. 2: https://arxiv.org/html/2311.17009v2/analysis_nn.png
    - 図の内容: 時空間特徴をそのまま再構成すると元の見た目に縛られる。SMM で再構成すると、姿勢や位置は保ちつつ見た目は自由になる。SMM で最近傍のフレームを検索すると、見た目が違っても同じ姿勢のフレームが返る。
    - 読み方: 「空間平均した特徴でも、見た目に頑健な動き・姿勢の情報が残る」ことを示す証拠。
  - Fig. 3: https://arxiv.org/html/2311.17009v2/pipeline.png
    - 図の内容: (a) DDIM 逆変換 → 時空間特徴 → フレームごとの SMM（$F\times D$）→ ペアワイズ差（$F\times F\times D$）。(b) 生成中の動画の差分行列を参照動画のものに合わせるよう誘導する。
  - Fig. 8: https://arxiv.org/html/2311.17009v2/metrics.png
    - 図の内容: 横軸が CLIP テキスト類似度（編集への忠実さ）、縦軸が Motion-Fidelity score。TokenFlow は動きは高いが編集が弱く、Tune-A-Video はその逆。
    - 読み方: **右上ほど「見た目は変えつつ動きは保つ」ことに成功している**。学生の評価も「static への不変性（横軸）× dynamic への感度（縦軸）」の2軸で描くと伝わりやすい。

### 3.7 MotionClone: Training-Free Motion Cloning for Controllable Video Generation
- 著者: Pengyang Ling et al.
- 会場: **ICLR 2025**（proceedings.iclr.cc の 2025 ページと ICLR virtual poster で確認）
- arXiv: https://arxiv.org/abs/2406.05338
- 方法と評価: temporal attention の重みのうち**値が大きい少数の成分（各空間位置で top-k、既定は $k=1$）**が主要な動きを決めている、という観察に基づく。参照動画の疎な temporal attention マップ（$t_\alpha=400$ の1回のデノイズで取り出す）に生成動画の注意を合わせるよう、デノイズの初期ステップだけ誘導する。評価はテキスト整合（CLIP）、時間一貫性（連続フレームの CLIP）、20人のユーザースタディ（motion preservation と appearance diversity、1〜5点）。
- 学生への翻訳: 学生の ViT に**フレームをまたぐ注意**がある場合（時空間トークンを一緒に扱う構成）は、LoRA の有無でフレーム間の注意の **top-1 の対応先**がどう変わるかを可視化・定量化できる（例: 光学フローとの一致率）。限界として、フレームごとに独立に処理する ViT では、この分析はできない。

### 3.8 Motion Inversion for Video Customization
- 著者: Luozhou Wang et al.
- 会場: **SIGGRAPH 2025**（Adobe Research の論文ページで確認。ACM DL の掲載ページは未確認）
- arXiv: https://arxiv.org/abs/2403.20193
- 方法: temporal transformer に足す学習可能な **Motion Embedding** を2種類学習する。
  - **Motion Query-Key Embedding** $m^{QK}\in\mathbb{R}^{1\times N\times C}$ は**空間次元をもたない**。空間次元をもたせると、注意マップに物体の形（見た目）が写り込むため。
  - **Motion Value Embedding** $m^V\in\mathbb{R}^{(H\times W)\times N\times C}$ は空間次元をもつが、**前フレームとの差分**をとって static な見た目を除く（光学フローに似た発想）。
- 評価: Motion Fidelity Score（Yatim と同じ CoTracker ベース）、テキスト類似度、画質、121人のユーザースタディ（66組、22本の動画）。
- 学生への翻訳: 「**Q/K は空間を潰せば見た目に依存しにくい。V は差分をとらないと見た目を運ぶ**」という論点は、Q/V LoRA を使う学生の設計を見直す直接の材料になる。評価としては、Q だけの LoRA と V だけの LoRA を別々に有効にして static 漏れの probe を比べる（3.11 と合わせて読むとよい）。限界として、論文は temporal attention（フレーム間）を扱っている。spatial attention の Q/V にも同じ役割分担が当てはまるかは仮説にとどまる。
- 図（検証済み）:
  - Fig. 3: https://arxiv.org/html/2403.20193v2/method-2.png
    - 図の内容: 左は $m^{QK}$ に空間次元をもたせると（×）注意マップに戦車の形が写り込むが、空間次元を除くと（✓）フレーム間の $N\times N$ の関係だけになる様子。右は $m^V$ をそのまま使うと元の人物の見た目が残るが、差分をとると新しいプロンプトの見た目に置き換わる様子。
    - 読み方: 「どの成分が見た目を運ぶか」を構造面から整理した図。

### 3.9 DiTFlow: Video Motion Transfer with Diffusion Transformers
- 著者: Alexander Pondaven et al.
- 会場: **CVPR 2025**（arXiv comment と CVF Open Access ページで確認）
- arXiv: https://arxiv.org/abs/2412.07776
- 方法と評価: DiT の全フレーム間の注意から、各パッチがフレーム $i$ からフレーム $j$ のどのパッチへ移るかを argmax でとり、変位マップ **AMF（Attention Motion Flow）**を作る。生成の潜在（または位置埋め込み）を、AMF の L2 損失で最適化する。DAVIS の50本の動画それぞれに **Caption / Subject / Scene の3段階のプロンプト**（元の動画から離れていく順）を用意し、CLIPScore による画質（IQ）と、Yatim 型の Motion Fidelity で評価する。
- 学生への翻訳: 「元の動画にどれだけ近い見た目で試すか」を段階的に変える設計は、評価データの作り方に使える。例えば (1) 同じ動画の別区間、(2) 同じ動作で別の人物、(3) 同じ動作で別のシーン、の順に static の距離を広げ、LoRA 特徴の動きの類似度がどこまで保たれるかを測る。AMF は、ViT のパッチ特徴の最近傍でフレーム間の対応をとれば、注意を使わずに近いものを作れる。限界として、最近傍の対応は見た目の類似度に依存するので、見た目が大きく変わると対応がとれない。

### 3.10 DeT: Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer
- 著者: Qingyu Shi et al.
- 会場: **ICCV 2025**（arXiv comment「ICCV 2025」で確認。CVF のページは URL を推測したところ 404 になり、未確認）
- arXiv: https://arxiv.org/abs/2503.17350
- 方法:
  - DiT の 3D full attention では前景と背景の特徴が混ざり、見た目と動きを分けにくいことを可視化で示す。
  - **共有の時間方向 1D カーネル**で特徴を時間方向に平滑化して分離を促し、潜在上の dense point tracking loss で前景の軌跡を揃える。
  - **MTBench** を提案する。軌跡のクラスタ数で難易度を easy（1〜3）/ medium（4〜6）/ hard（7以上）に分ける。
- 評価:
  - **Hybrid motion fidelity**: $\mathcal{M}=\frac1N\sum_n\big[\alpha e^{-d_F(\mathcal{T}^n_i,\mathcal{T}^n_j)}+(1-\alpha)\bar c_n\big]$（$\alpha=0.5$）。$d_F$ は軌跡の Fréchet 距離で、**点の順序**も考慮する。$\bar c_n$ は速度 $\Delta\mathcal{T}(t)$ のコサインの平均。
  - 編集の忠実度（CLIP）と、時間一貫性（連続フレームの DINO 特徴のコサイン）。
- 学生への翻訳:
  - Fréchet 距離は時間を逆にすると値が変わるので、**順序を区別する動きの物差し**になる（MFS のコサインも符号が反転する）。順序を逆にしたクリップ対で「外部の物差しの距離」と「LoRA 特徴の距離」が同じ方向に動くかを検証できる。
  - 動きの複雑さによる難易度分けは、評価クリップを層別して結果を報告するときに使える。
  - 限界として、4フレームの軌跡は短く、Fréchet 距離はほとんど端点の差で決まってしまう。
- 図（検証済み）:
  - Fig. 2: https://arxiv.org/html/2503.17350v2/figs/trade_off.png
    - 図の内容: 動きの忠実度と編集の忠実度のトレードオフで、各手法の位置を比べたもの。
  - Fig. 7: https://arxiv.org/html/2503.17350v2/figs/MTBench_new.png
    - 図の内容: (a) 評価プロンプトの動作語の頻度分布。(b) 軌跡のクラスタ数による難易度分け（平均クラスタ数は easy 2.7、medium 5.8、hard 10.4）。

### 3.11 Separate Motion from Appearance: Customizing Motion via Customizing Text-to-Video Diffusion Models
- 著者: Huijie Liu et al.
- 会場: **ACM MM 2025（公式 GitHub README の BibTeX「Proceedings of the 33nd ACM international conference on multimedia」, 2025 による。公式 proceedings ページは未確認）**
- arXiv: https://arxiv.org/abs/2501.16714
- 方法:
  - 先行手法（MotionDirector）では、参照動画の背景（例: 窓）が生成に紛れ込む「**appearance leakage**」が起きることを指摘する。
  - **TAP（Temporal Attention Purification）**: 事前学習済みの **Value はそのまま**にして、temporal LoRA は **Query か Key だけ**に入れ、注意の形だけを変えて動きを作る。最終的には Key だけに入れる。
  - **AH（Appearance Highway）**: U-Net のスキップ接続の始点を、LoRA を入れた temporal ブロックから、LoRA を入れていない spatial ブロックに付け替える。
- 評価:
  - **Q/K/V/FFN のどれに LoRA を入れるかの探索実験**（Tab. 1）がある。指標は、Motion（既製の動作分類器の正解率）、App. Align.（動詞を除いた新プロンプトとの CLIP 類似度、↑）、**App. Leak.（参照動画の被写体のテキスト "a person" との CLIP 類似度、↓）**。
  - 結果: Q/K に入れると動きを得つつ見た目を保てる（App. Align. 約25.5）。V や FFN に入れると見た目が悪化する（V は 22.88、FFN と MotionDirector は約21.2）。
  - ほかに CLIP / ViCLIP / VBench / 美的スコアと、ユーザースタディ（appearance diversity、motion fidelity、temporal consistency）。
- 学生への翻訳:
  - **学生の Q/V LoRA にいちばん直接関係する論文**。「V への更新は見た目を運びやすく、Q/K への更新は注意の配分だけを変える」という仮説を、学生の設定で**Q だけの LoRA / V だけの LoRA / Q+V の LoRA**を学習して比べれば検証できる。比べるのは static 漏れの probe（シーン・ID の正解率、↓が望ましい）と dynamic probe（順序・速度、↑）。
  - App. Leak は、**「参照（学習）データに特有の static 属性が、無関係なクリップの特徴にも出てくるか」**という形に言い換えられる。例えば学習ストリームのシーンを zero-shot で分類する probe を、別ドメインのクリップに当てる。
  - 限界として、論文の Q/K/V は**時間方向の注意**のもので、学生の ViT（多くは空間方向の注意）の V は**パッチの中身そのもの**を変える。この違いのため、「V＝見た目」という結論がさらに強く出る可能性も、逆に Q でも見た目が変わる可能性もある。
- 図（検証済み）:
  - Fig. 1: https://arxiv.org/html/2501.16714v2/fig1-8.png
    - 図の内容: 参照動画の背景にある「窓」が、先行手法（2行目）の生成に紛れ込む appearance leakage の例と、提案手法（最終行）の結果。
    - 読み方: 「動きの LoRA が見た目も覚えてしまう」という問題を1枚で伝える図。
  - Fig. 2: https://arxiv.org/html/2501.16714v2/framework-8.png
    - 図の内容: (a) U-Net 全体と Appearance Highway（黄色の破線）。(b) TAP で、Q と V の Linear はそのまま、K の Linear にだけ LoRA を付ける。(c) スキップ接続の始点を temporal ブロックから spatial ブロックに付け替える。
    - 読み方: (b) が「LoRA をどの射影に入れるか」を設計の対象にしている点に注目する。

### 3.12 VideoMage: Multi-Subject and Motion Customization of Text-to-Video Diffusion Models
- 著者: Chi-Pin Huang et al.（arXiv API で確認）
- 会場: **CVPR 2025**（arXiv comment「CVPR 2025」と CVF Open Access の PDF の存在で確認）
- arXiv: https://arxiv.org/abs/2503.21781
- 方法: 動きの LoRA が被写体の見た目を拾ってしまう appearance leakage を、**負の classifier-free guidance**で抑える。参照動画の1フレームに textual inversion をかけて被写体トークンを作り、「A static video of ⟨被写体⟩」という見た目だけの条件 $c_{ap}$ で予測したノイズを差し引いた「見た目を除いた目標」$\epsilon_{ap\text{-}free}=(1+\omega)\epsilon-\omega\,\epsilon_\theta(x_t,c_{ap},t)$ に回帰させる。
- 評価: CLIP テキスト整合、時間一貫性、PickScore、ユーザースタディ（motion fidelity と motion diversity の選好率）。
- 学生への翻訳: 評価の面では「**静止動画（1フレームを複製したクリップ）に対する応答を、参照として差し引く**」という操作がそのまま使える。LoRA 特徴から、同じフレームを4枚並べた静止クリップの特徴を引いた残差 $f(\text{clip})-f(\text{static clip})$ に dynamic probe をかける。限界として、識別特徴は線形に足し引きできるとは限らない。
- 図: 主要図の検証は今回は見送った。

### 3.13 B-LoRA: Implicit Style-Content Separation using B-LoRA
- 著者: Yarden Frenkel et al.
- 会場: **ECCV 2024**（ECVA と LNCS の DOI 10.1007/978-3-031-72684-2_11、ECCV virtual poster で確認）
- arXiv: https://arxiv.org/abs/2403.14572
- 方法と評価: SDXL の各 transformer ブロックに別々のプロンプトを流し込む **prompt injection** の分析で、**ブロック2と4が内容（content）、ブロック5が色・スタイル**を担っていることを突き止めた。$\mathrm{sim}(C_I(I_{\hat p\to i}),C_T(\hat p))$ のように、生成画像と流し込んだプロンプトの CLIP 類似度で測る。そのうえで、その2ブロックだけに LoRA（B-LoRA）を学習し、1枚の画像からスタイルと内容を暗黙に分ける。評価は定性比較とユーザースタディ。
- 学生への翻訳: 「**層ごとに一部だけ介入して、どの層がどの因子を担っているかを測る**」という方法は、学生の LoRA の**層ごとの有効化・無効化**の実験にそのまま使える。各層の LoRA だけを有効にしたときの static probe と dynamic probe の変化を表にすればよい。限界として、ViT の層の役割は SDXL の UNet とは異なるので、同じ層番号の対応は期待できない。
- 図（検証済み）:
  - Fig. 4: https://arxiv.org/html/2403.14572v2/vis_inject_generation2.png
    - 図の内容: 左は「tiger」をブロック2か4だけに流し込むと内容が虎に変わり、他のブロックに流し込んでも変化しない様子。右はブロック5が色を支配している様子。
    - 読み方: 層単位で介入して因子との対応をとる見本。

### 3.14 ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs
- 著者: Viraj Shah et al.
- 会場: **ECCV 2024**（ECVA の論文ページと ECCV virtual poster で確認）
- arXiv: https://arxiv.org/abs/2311.13600
- 方法と評価: 被写体の LoRA とスタイルの LoRA を合成するとき、2つの LoRA の $\Delta W$ の**列ベクトルどうしのコサイン類似度**が大きいと干渉する、という観察に基づく。列ごとの混合係数を学習し、各 LoRA の再現性を保ちつつ列間のコサインを小さくする。評価は、スタイル整合（CLIP-I）、被写体整合（DINO）、テキスト整合（CLIP-T）、ユーザースタディ。失敗例として、スタイル画像の内容が漏れるケースも示している。
- 学生への翻訳: **LoRA の重み空間での比較**として使える。(a) 静止クリップ（1フレームの複製）だけで学習した「static LoRA」と、通常の LoRA を用意する。(b) 両者の $\Delta W$（列空間または左右の特異ベクトル）の主角度やコサインを比べる。重なりが大きければ、通常の LoRA も static な適応を多く含むと読める。限界として、重み空間で近いことと、機能として同じことは一致しない。
- 図（検証済み）:
  - Fig. 3(b): https://arxiv.org/html/2311.13600v2/fig6.png
    - 図の内容: 列ベクトルが強く揃った LoRA どうしを単純に足すと合成がうまくいかないことと、ZipLoRA が層をまたいで列間のコサイン類似度の平均を下げていることを示す。
    - 読み方: 「2つの適応の重なり具合を、重み空間でどう定量化するか」の例。

---

## 4. 学生（識別的な対照学習 ViT＋LoRA）向けの評価の言い換え: まとめ表

| 元の評価（論文） | 識別モデルでの言い換え | 主な限界 |
|---|---|---|
| 条件付き生成＋判定器 Acc（DSVAE/S3VAE/C-DSVAE/SPYL/DBSE） | 入力側で static と dynamic を操作したクリップ対を作る。同じ見た目でシャッフル・逆再生・速度変更したものと、色変換・別人物・別シーンで同じ動きのもの。各特徴でどちらに反応するかを測る | 自然動画で「同じ動き・違う見た目」の対を集めるのが難しい（合成、同じ動作クラス、AnimateDiff 型のズームなどで補う） |
| 潜在の4分類器 probe と Leakage Gap（SPYL/DBSE） | LoRA なし特徴、LoRA あり特徴、その残差、時間平均を引いた特徴に static probe と dynamic probe をかけ、Gap $=\mathrm{Acc}_{target}-\mathrm{Acc}_{leak}$ を報告する | 線形 probe の限界。static と dynamic のラベルがデータ上で相関している場合 |
| EER による照合（FHVAE/DSVAE/C-DSVAE） | 同じ動画（同じシーン・同じ人物）かの照合の EER を特徴ごとに測る。static 用は低く、dynamic 用は高いのが望ましい | MoCo の目的が同一動画の判別に近いので、漏れが起きるのは前提として読む |
| scene discriminator（DRNet） | LoRA 由来特徴のペアから同一動画かを当てる事後分類器 | 同上 |
| シャッフルへの不変性（S3VAE）、content 増強（C-DSVAE） | シャッフル・逆再生したクリップでの特徴の変化量（static は不変、dynamic は変化） | 構造上順序に不変な特徴（時間方向の位置情報がない場合）では、自明に変化ゼロになる |
| 1フレーム目を引く（DBSE）、アンカーを引く（MotionDirector）、差分 V（Motion Inversion）、静止条件を引く（VideoMage） | $g_t-g_1$、$g_t-\bar g$、$f(\text{clip})-f(\text{静止 clip})$ を dynamic 特徴として probe する | 特徴が線形に足し引きできるとは限らない |
| SMM のペアワイズ差（Yatim） | 平均プール特徴の $4\times4$ 差分行列のクリップ間類似度 | 局所的な動きに弱い |
| Motion Fidelity（Yatim/Motion Inversion/DiTFlow）、Hybrid MF（DeT） | CoTracker で求めた動きの類似度と、特徴の類似度の順位相関（LoRA あり/なし、CLIP と比較） | 4フレームの軌跡は短い |
| App. Leak と Q/K/V の探索（Separate Motion） | Q だけ / V だけ / Q+V の LoRA を学習し、static 漏れと dynamic の probe を比べる | 時間方向の注意と空間方向の注意で、射影の役割が違う |
| 層ごとの介入（B-LoRA）、重み変化（DreamVideo） | 層ごとに LoRA を有効化して probe を比べる。層ごとの $\|BA\|$ を見る | 層の役割はモデルごとに違う |
| LoRA 列のコサイン（ZipLoRA） | static 専用 LoRA と通常 LoRA の部分空間の重なり | 重み空間で近くても、機能が同じとは限らない |
| Koopman 固有値（SKD） | 特徴系列に線形遷移を当てはめ、固有値1付近のモードのエネルギー割合を比べる | 4フレームでは推定が不安定 |
| ACD/MCS（MoCoGAN） | クリップ内のフレーム特徴の平均距離（static の一貫性）と動作 probe の正解率 | ノイズでも距離が増える |
| DCI と VLM 判定・タグ付け（MSD 2025, Locatello） | VLM で付けた static/dynamic ラベルで DCI と probe を計算する | VLM ラベルの誤り。速度や順序のラベル付けは未検証 |

---

## 5. 注意点（検証状況）

- 会場の確認状況:
  - 公式ページで確認: MoCoGAN、S3VAE、DreamVideo、VMC、Yatim（DMT）、DiTFlow（以上 CVF）。DSVAE、Locatello、SPYL、DBSE（以上 PMLR）。DRNet（NeurIPS proceedings）。MotionClone（ICLR proceedings）。MotionDirector、B-LoRA、ZipLoRA（以上 ECVA）。MSD（NeurIPS 2025 D&B ページ）。AnimateDiff（OpenReview の PDF）。
  - arXiv comment だけで確認: MCnet、FHVAE、R-WAE、C-DSVAE、SKD、Customize-A-Video、DeT、VideoMage。
  - 二次情報で確認: Motion Inversion（Adobe Research ページで SIGGRAPH 2025）、Separate Motion from Appearance（GitHub の BibTeX で ACM MM 2025）。どちらも公式 proceedings は未確認。
  - AnimateDiff の Spotlight は GitHub の表記によるもので、OpenReview の forum では未確認。
- 図はすべて arXiv HTML の `<img src>` から絶対 URL を組み立て、`curl -sI` で HTTP 200 と image/* を確認した（33枚）。PDF からの図は使っていない。MoCoGAN と R-WAE は arXiv HTML がない（ar5iv も変換失敗）ため図はない。AnimateDiff は図のファイル名が汎用名で、図番号との対応を確かめられなかったため掲載していない。
- 数値（例: MUG Acc 86.90%、Separate Motion の Tab. 1 の値）は各論文の HTML 本文から転記した。学生の設定での「翻訳」はすべて本調査者の提案で、論文が主張していることではない。
