# Track 1: LoRA / ファインチューニング更新と表現を解析する一般的手法（2016–2026）

作成日: 2026-11-12
対象の問い: 画像事前学習ViTの attention Q/V に入れた LoRA（4フレーム連結・時間位置埋め込み・streaming MoCo）が、**静的な見た目ではなく動的情報（順序・方向・速度）を学んだか**を、LoRAそのものと表現の両面からどう解析するか。
現状の観測: フレームshuffle / reverse で特徴がほぼ変わらない（cos ≈ 0.998）。

## 調査方法と信頼度について

- 会議・年は、arXiv の Comments 欄、PMLR / NeurIPS proceedings / CVF / ECVA / ACL Anthology / iclr.cc・neurips.cc の virtual ページ、mlanthology のいずれかで確認した。確認元を各項目に書いている。一次ソースで確認できなかったものは「未確認」と書いた。
- 図URLはすべて arXiv HTML 版（`https://arxiv.org/html/<id>vN/...`）の `<img src>` / `<object data>` から組み立て、`curl -sI` で **HTTP 200 かつ `content-type: image/png` または `image/svg+xml`** を確認したものだけを載せている（2026-11-12 時点）。
  - 補足: HTML中の相対パスは `<id>vN/xxx.png` の形だが、実際の配信URLは `https://arxiv.org/html/<id>vN/xxx.png`（バージョンディレクトリは1回だけ）だった。
- 「生徒の問いへの適用」は、各論文の手法を本研究設定に当てはめた**提案**であり、論文が主張している結果ではない。

---

## 一覧表

| # | 論文 | 会議・年 | 解析手法のカテゴリ | 図 |
|---|---|---|---|---|
| 1 | LoRA (Hu et al.) | ICLR 2022 | ΔWの部分空間類似度・増幅率 | 2 |
| 2 | AdaLoRA (Zhang et al.) | ICLR 2023 | SVD型パラメータ化・感度ベース重要度 | - |
| 3 | DoRA (Liu et al.) | ICML 2024 (Oral) | 重みの大きさ/方向分解 | 2 |
| 4 | Asymmetry in LoRA (Zhu et al.) | ICML 2024 | A/B 行列の役割（CCA類似度） | 1 |
| 5 | LoRA Learns Less and Forgets Less (Biderman et al.) | TMLR 2024 (Featured Cert.) | Full-FT の ΔW スペクトル・必要ランク | 2 |
| 6 | LoRA vs Full FT: An Illusion of Equivalence (Shuttleworth et al.) | NeurIPS 2025 | intruder dimensions（特異ベクトル対応） | 2 |
| 7 | PiSSA (Meng et al.) | NeurIPS 2024 (Spotlight) | 主成分特異成分での初期化 | 1 |
| 8 | MiLoRA (Wang et al.) | NAACL 2025 | 副成分特異成分での初期化・部分空間類似度 | 1 |
| 9 | Measuring the Intrinsic Dimension (Li et al.) | ICLR 2018 | 内在次元 | - |
| 10 | Intrinsic Dimensionality … LM Fine-Tuning (Aghajanyan et al.) | ACL 2021 | 内在次元（ファインチューニング版） | - |
| 11 | Weight Spectra Induced Efficient Model Adaptation (Si et al.) | 未確認（arXiv 2025） | Full-FT の特異値・特異ベクトル変化 | - |
| 12 | Editing Models with Task Arithmetic (Ilharco et al.) | ICLR 2023 | タスクベクトル・コサイン類似度・否定 | 2 |
| 13 | TIES-Merging (Yadav et al.) | NeurIPS 2023 | 符号衝突・上位k%トリミング | 1 |
| 14 | KnOTS (Stoica et al.) | ICLR 2025 | LoRA更新の SVD 整列・ΔW出力のCKA | 2 |
| 15 | Task Singular Vectors (Gargiulo et al.) | CVPR 2025 | 層ごとのタスク行列SVD・タスク干渉スコア | 1 |
| 16 | SVCCA (Raghu et al.) | NeurIPS (NIPS) 2017 | 表現類似度（SVD+CCA） | 1 |
| 17 | CKA (Kornblith et al.) | ICML 2019 | 表現類似度（CKA） | 1 |
| 18 | Linear classifier probes (Alain & Bengio) | ICLR 2017 (Workshop と認識、区分は未確認) | 線形プローブ | - |
| 19 | Network Dissection (Bau et al.) | CVPR 2017 (Oral) | ユニットと概念の対応（IoU） | - |
| 20 | Revisiting Model Stitching (Bansal et al.) | NeurIPS 2021 | モデルスティッチング | - |
| 21 | What is being transferred in transfer learning? (Neyshabur et al.) | NeurIPS 2020 | 特徴再利用・loss basin・モジュール臨界性 | - |
| 22 | Fine-Tuning can Distort Pretrained Features (Kumar et al.) | ICLR 2022 (Oral) | 特徴歪み（ID/OOD部分空間） | - |
| 23 | On the Surprising Effectiveness of Attention Transfer for ViTs (Li et al.) | NeurIPS 2024 | attention map と特徴の切り分け・CKA・JSD | 2 |
| 24 | Fine-Tuning Enhances Existing Mechanisms (Prakash et al.) | ICLR 2024 | 回路解析・DCM・CMAP（モデル間パッチング） | 2 |
| 25 | Overcoming Sparsity Artifacts in Crosscoders (Minder et al.) | NeurIPS 2025 | crosscoder によるモデル差分 | - |
| 26 | Convergent Linear Representations of Emergent Misalignment (Soligo et al.) | 未確認（arXiv 2025） | rank-1 LoRA を方向×スカラーとして解釈 | - |
| 27 | Learning to Interpret Weight Differences (Goel et al.) | 未確認（arXiv 2025） | 重み差分を言語で説明させる（DIT） | - |
| 28 | Concept Sliders (Gandikota et al.) | ECCV 2024 | LoRA = 解釈可能な概念方向 | 1 |
| 29 | B-LoRA (Frenkel et al.) | ECCV 2024 | ブロックごとの役割分析（style/content） | 1 |
| 30 | Heart-LoRA (Zhong et al.) | WACV 2025 | ViT LoRA のヘッド単位重要度（Taylor展開） | - |

合計 30 本（うち会議・ジャーナル確認済み 27 本、未確認 3 本）。図は 25 枚・27 URL（17 本分。Asymmetry の Figure 1 は (a)(b)(c) の3 URL）。

---

## A. LoRA の重み更新 ΔW = BA そのものを解析する手法

### 1. LoRA: Low-Rank Adaptation of Large Language Models
- 著者: Edward J. Hu et al.
- 会議: **ICLR 2022**（mlanthology `iclr/2022/hu2022iclr-lora` と OpenReview forum `nZeVKeeFYf9` で確認。OpenReview 本体ページは閲覧していない）
- arXiv: [2106.09685](https://arxiv.org/abs/2106.09685)
- 解析手法:
  - LoRA は事前学習重み W0 を凍結し、更新を ΔW = BA（B ∈ R^{d×r}, A ∈ R^{r×k}）に限定する。論文の Section 7 は「学習された ΔW がどんな部分空間にあるか」を調べている。
  - **部分空間類似度**: ランク r=8 と r=64 で学習した A の右特異ベクトル U をとり、上位 i 本と上位 j 本の重なりを
    φ(A1, A2, i, j) = ‖U_{A1}^{i⊤} U_{A2}^{j}‖_F² / min(i, j) ∈ [0,1]
    で測る。1 なら同じ部分空間、0 なら直交。r=8 の上位方向は r=64 にも含まれることから、「本当に効いている方向はごく少数」と論じている。
  - **増幅率 (amplification factor)**: ΔW の上位 r 特異ベクトル U, V に W を射影し、‖ΔW‖_F / ‖U^⊤ W V^⊤‖_F を計算する。GPT-3 の48層目 W_q、r=4 では 6.91/0.32 ≈ 21.5。結論は「ΔW は W で強調されていない既存方向を大きく増幅している」。
- 生徒の問いへの適用（提案）: 各層の ΔW_q, ΔW_v で φ（シード間・ランク間）と増幅率を出すと、「LoRAが少数の安定した方向を学んだか」「その方向は W0 で弱かった方向か」がわかる。さらに、その上位方向を**時間位置埋め込みの差分 (p_t − p_t')** や**フレーム差分トークン**に当てて応答を見れば、時間に関係する方向かを確かめられる。
- 限界: どの方向が強化されたかはわかるが、その方向が「順序」「速度」を表しているかは、入力側の操作（shuffle 等）や probe と組み合わせないと判断できない。
- 図:
  - **Figure 3** — `https://arxiv.org/html/2106.09685v2/qv8_qv64.png`
    r=8 と r=64 の A の列ベクトル部分空間類似度のヒートマップ（ΔW_q と ΔW_v）。横軸・縦軸はそれぞれ何本目までの特異方向を使ったか。左下が明るいほど上位方向が共有されている。右の2枚は左下三角の拡大。
  - **Figure 8** — `https://arxiv.org/html/2106.09685v2/w_vs_delta_w.png`
    W_q の特異方向と ΔW_q の特異方向の正規化部分空間類似度を、r を変えた場合とランダム基準とで比べた図。ΔW はランダムより W と相関するが、W の最上位方向そのものとは重ならない。「W で強調されていない方向を増幅する」という主張の根拠。

### 2. AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
- 著者: Qingru Zhang et al.
- 会議: **ICLR 2023**（arXiv Comments: "The 11th International Conference on Learning Representations (ICLR 2023)"）
- arXiv: [2303.10512](https://arxiv.org/abs/2303.10512)
- 解析手法:
  - 更新を SVD の形 ΔW = P Λ Q でパラメータ化し、直交性の正則化 ‖P^⊤P − I‖_F² + ‖QQ^⊤ − I‖_F² を加える。
  - 各特異値（とそれに対応する P, Q の成分）の重要度を、感度 I(w) = |w · ∇_w L| を指数移動平均で平滑化し、その不確かさも考慮したスコアで測る。重要度の低い特異値を落として、層や行列（Q/K/V/FFN）ごとにランクを配分し直す。
  - 結果として「どの層・どの行列にランクが残ったか」（Figure 3 のランク分布）自体が、どこが重要だったかの解析になる。
- 生徒の問いへの適用（提案）: Q と V、浅い層と深い層のどこにランクが残るかを見れば、LoRA が働いている場所の地図になる。特異値ごとの感度を MoCo 損失で計算し、**通常クリップとshuffleクリップで感度が変わる方向**を探す使い方もある。
- 限界: 重要度は学習損失に対する感度なので、MoCo 損失が順序を必要としていなければ、順序に関わる方向は重要と出ない。損失に依存した指標である。
- 図: HTML版はあるが、手法の説明に向く図が少ないため載せない。

### 3. DoRA: Weight-Decomposed Low-Rank Adaptation
- 著者: Shih-Yang Liu et al.
- 会議: **ICML 2024 (Oral)**（arXiv Comments: "ICML2024(Oral)"）
- arXiv: [2402.09353](https://arxiv.org/abs/2402.09353)
- 解析手法:
  - 重みを列ごとの**大きさ m** と**方向 V/‖V‖_c** に分ける: W = m · V/‖V‖_c。
  - 学習ステップ t での変化を、列ごとに平均した
    ΔM^t = (1/k) Σ_n |m^{n,t} − m_0^n|（大きさの変化）、
    ΔD^t = (1/k) Σ_n (1 − cos(V^{n,t}, W_0^n))（方向の変化）
    で測る。横軸 ΔD・縦軸 ΔM で散布図を描くと、Full FT は**負の相関**（方向を大きく変えるときは大きさを少し変える）、LoRA は**正の相関**（大きさと方向を一緒に変える）になる、と報告している。
- 生徒の問いへの適用（提案）: ΔW_q と ΔW_v で ΔM/ΔD を層ごとに出せば、LoRA が「既存特徴のスケールを変えただけ」なのか「方向まで回したのか」を見分けられる。スケールが中心なら、既存の静的特徴を強めただけという解釈と整合する（ただし、それで確定するわけではない）。
- 限界: 列単位の幾何量なので、時間情報か外観情報かという**意味**はわからない。
- 図:
  - **Figure 2** — `https://arxiv.org/html/2402.09353v6/qproj_ft_vs_lora_vs_MADoRA_pattern.svg`
    query 行列について、(a) FT、(b) LoRA、(c) DoRA の ΔD（横）と ΔM（縦）を層・学習ステップごとにプロットした図。マーカーは学習ステップ、色は層。傾きの符号（FT は負、LoRA は正）を見る。
  - **Figure 3** — `https://arxiv.org/html/2402.09353v6/img/q_delta_m_d.png`
    LoRA/DoRA と事前学習重みとの (a) 大きさの差、(b) 方向の差を層ごとに示した図。どの層で何がどれだけ変わったかを読む。

### 4. Asymmetry in Low-Rank Adapters of Foundation Models
- 著者: Jiacheng Zhu et al.
- 会議: **ICML 2024**（PMLR v235, pp. 62369–62385, `proceedings.mlr.press/v235/zhu24c.html` で確認）
- arXiv: [2402.16842](https://arxiv.org/abs/2402.16842)
- 解析手法:
  - ΔW = BA のうち、**A は入力から特徴を取り出し、B はその特徴から出力を作る**という役割の違いを主張する。
  - 「同じタスク・別シード」「同じ初期化・別タスク」「別初期化・別タスク」で学習した A 同士、B 同士の類似度を層ごとに CCA で比べる。B は**同じタスクなら似ていて、タスクが違えば似ない**。A は**初期化が同じなら似ていて、タスクにはあまり依存しない**。
  - そこから「B だけ学習し、A はランダムのまま固定してもほぼ同じ性能」になることを示し、情報理論的な汎化バウンドでも説明している。実験は RoBERTa, BART, LLaMA-2, ViT。
- 生徒の問いへの適用（提案）: 学習された情報は主に **B（出力側の部分空間）**にあると考え、B の列空間を解析対象の中心にする。例えば「A をランダムに固定して B だけ学習した LoRA」と比べて時間感度が変わるかを見るのは、軽い対照実験になる。
- 限界: A/B の役割分担は言えるが、学習した内容が時間情報かどうかは別に検証が必要。
- 図:
  - **Figure 1 (a)(b)(c)** —
    `https://arxiv.org/html/2402.16842v2/figures/sametask_diffseed_cca_mean.png`（(a) 別初期化・同タスク）
    `https://arxiv.org/html/2402.16842v2/figures/difftask_sameseed_cca_mean.png`（(b) 同初期化・別タスク）
    `https://arxiv.org/html/2402.16842v2/figures/difftask_diffseed_cca_mean.png`（(c) 別初期化・別タスク）
    RoBERTa の各層で、学習後の A 同士・B 同士の類似度を描いた図。B は (a) で高く (b)(c) で低い。A は (b) で高く (a)(c) で低い。「B はタスクを、A は初期化を反映する」と読む。

### 5. LoRA Learns Less and Forgets Less
- 著者: Dan Biderman et al.
- 会議: **TMLR 2024（Featured Certification）**（arXiv Comments: "accepted to Transactions on Machine Learning Research, August 2024 (Featured Certification)"。OpenReview `aloEru2qCG`）
- arXiv: [2405.09673](https://arxiv.org/abs/2405.09673)
- 解析手法:
  - コード・数学ドメインで LoRA と Full FT を比べ、ターゲット性能（learning）と元ドメイン性能の劣化（forgetting）を同時に測る。
  - **Full FT の ΔW を SVD し、分散の 90% を説明するのに必要なランク**を層・行列ごとに数える。必要ランクは典型的な LoRA 設定の 10〜100倍で、ΔW は低ランクではないと論じる（例: LLaMA-2-7B の4096×4096 W_q で >1500）。
  - 生成のトークン多様性、正則化（dropout, weight decay）との比較も行う。
- 生徒の問いへの適用（提案）: 小規模な Full FT（または高ランク LoRA）を同じ MoCo 設定で回し、ΔW のスペクトルを LoRA と比べると、「時間情報を学ぶには低ランク更新では足りないのか」を間接的に確かめられる。forgetting 側として、ImageNet 線形 probe 精度の維持度も測れる。
- 限界: ランクは「表現力の容量」の指標であり、何を学んだかはわからない。LLM での結果なので、ViT＋動画で同じ傾向になるとは限らない。
- 図:
  - **Figure 6** — `https://arxiv.org/html/2405.09673v2/figures/svd_summary_starcoder_all.png`
    StarCoder での継続事前学習における、層（横軸）ごとの「分散90%を説明するのに必要なランク」（縦軸、最大4096）。色は学習トークン量で、薄いほど長く学習している。学習が進むほど必要ランクが上がる様子を読む。
  - **Figure S7** — `https://arxiv.org/html/2405.09673v2/figures/single_layer_spectrum_starcoder_cropped.png`
    26層目 W_q の特異値（基礎重み、FT後の重み、その差分）と累積説明分散。差分 ΔW も 90% の説明にランク >1500 が必要なことを示す。

### 6. LoRA vs Full Fine-tuning: An Illusion of Equivalence
- 著者: Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma
- 会議: **NeurIPS 2025**（`neurips.cc/virtual/2025/poster/115207` で確認。arXiv Comments 欄には記載なし）
- arXiv: [2410.21228](https://arxiv.org/abs/2410.21228)
- 解析手法:
  - 事前学習重み W0 と学習後重み W_t をそれぞれ SVD し、**学習後の各特異ベクトル y_j と事前学習の全特異ベクトル x_i のコサイン類似度**の行列を描く。
  - **intruder dimension の定義（Definition 3.1）**: 学習後の上位 k 本の特異ベクトルのうち、max_i cos(y_j, x_i) < ε となるもの（事前学習のどの方向とも似ていない新しい方向）。Algorithm 1 で個数を数える。
  - LoRA は上位特異方向に intruder dimension を作り、Full FT はほとんど作らない。intruder dimension の数は事前学習分布の性能劣化（forgetting）や継続学習での劣化と相関し、その特異値を縮めると忘却が減る。B だけ学習すると intruder dimension が減る、という結果も報告している。
- 生徒の問いへの適用（提案）: 各層の W_q + ΔW_q, W_v + ΔW_v について intruder dimension を数え、**intruder 方向が時間位置埋め込み・フレーム差分にどれだけ反応するか**を測ると、「LoRA が作った新方向」が時間的な方向か外観的な方向かを調べられる。0.998 という結果と合わせ、新方向が「フレーム共通（静的）」入力にしか反応しないなら、静的特徴を学んだ裏付けになる。
- 限界: intruder dimension は**新しい方向がある**ことを示すだけで、その方向の意味は示さない。ε と k の選び方に結果が依存する。
- 図:
  - **Figure 1** — `https://arxiv.org/html/2410.21228v3/figures/magicoder_full_vs_lora_r64.png`
    LLaMA2-7B で、事前学習と学習後の特異ベクトルのコサイン類似度行列。Full FT はほぼ対角（構造が保たれる）で、LoRA は対角がずれて、類似度の低い列（intruder）が現れる。
  - **Figure 2** — `https://arxiv.org/html/2410.21228v3/figures/figure_1.png`
    (a) 事前学習重みの SVD が学習でどう変わるかを測る方法の模式図。(b) Full FT と LoRA の行列の比較。(c) intruder dimension は事前学習のどの特異ベクトルともコサイン類似度がほぼ0になる。

### 7. PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models
- 著者: Fanxu Meng et al.
- 会議: **NeurIPS 2024 (Spotlight)**（arXiv Comments: "NeurIPS 2024 spotlight"）
- arXiv: [2404.02948](https://arxiv.org/abs/2404.02948)
- 解析手法:
  - W = U Σ V^⊤ の**上位 r 個の主成分**で A, B を初期化し（A = U_r Σ_r^{1/2}, B = Σ_r^{1/2} V_r^⊤ の形）、残差 W_res = W − AB を凍結する。
  - 「W の主要方向を直接動かす」と「LoRA のように 0 から別方向を学ぶ」の違いを、収束速度と性能で比べている。手法の論文だが、「どの特異部分空間を更新させるか」を制御変数にできる点で解析にも使える。
- 生徒の問いへの適用（提案）: PiSSA 型（主成分）と MiLoRA 型（副成分）の初期化で、時間感度（shuffle 時の特徴変化）がどう変わるかを比べると、「時間情報が W0 のどの部分空間で表現されやすいか」の手がかりになる。
- 限界: 初期化を変えた比較は介入実験なので、性能差の原因（時間情報なのか単に最適化のしやすさなのか）は切り分けが必要。
- 図:
  - **Figure 1(c)** — `https://arxiv.org/html/2404.02948v4/lora_pissa.png`
    Full FT・LoRA・PiSSA の比較図のうち PiSSA の部分。青は凍結、橙は学習するパラメータ。W を主成分 (A, B) と残差に分け、主成分側だけを学習する構造を示す。

### 8. MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning
- 著者: Hanqing Wang et al.
- 会議: **NAACL 2025**（arXiv Comments: "accepted at NAACL 2025"。ACL Anthology `2025.naacl-long.248`）
- arXiv: [2406.09044](https://arxiv.org/abs/2406.09044)
- 解析手法:
  - PiSSA とは逆に、W の**下位（副）特異成分**で LoRA を初期化し、主成分は凍結する。「事前学習知識は主成分にあるので、副成分を使えば壊しにくい」という仮説。
  - 学習後の更新と W の特異部分空間の**部分空間類似度**を層・モジュールごとに比べる（Figure 2）。
- 生徒の問いへの適用（提案）: PiSSA と同様、どの部分空間を動かすと時間感度が出るかの対照実験に使える。
- 限界: LLM の結果なので、ViT＋MoCo で同じ傾向になる保証はない。
- 図:
  - **Figure 1** — `https://arxiv.org/html/2406.09044v3/overview.png`
    (a) 重み行列を副特異成分（U_m, Σ_m, V_m）と主成分に分け、副成分で A_m, B_m を初期化する図。(b) 学習中は A_m, B_m だけを更新し、主成分 W_p は凍結する。

### 9. Measuring the Intrinsic Dimension of Objective Landscapes
- 著者: Chunyuan Li et al.
- 会議: **ICLR 2018**（arXiv Comments: "Published in ICLR 2018"）
- arXiv: [1804.08838](https://arxiv.org/abs/1804.08838)
- 解析手法: パラメータを θ = θ_0 + P θ_d（P はランダム射影、θ_d は d 次元）に制限して学習し、通常学習の 90% 性能に届く最小の d（d_90）を「内在次元」と呼ぶ。タスクの難しさや必要な自由度の目安になる。
- 生徒の問いへの適用（提案）: 「静的特徴のみで達成できる MoCo 損失」と「時間情報が必要なタスク（順序分類など）」で d_90 を比べれば、後者に必要な自由度を見積もれる。
- 限界: 自由度の量はわかるが、何を学んだかはわからない。

### 10. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
- 著者: Armen Aghajanyan et al.
- 会議: **ACL-IJCNLP 2021（Outstanding Paper）**（ACL Anthology `2021.acl-long.568` で確認）
- arXiv: [2012.13255](https://arxiv.org/abs/2012.13255)
- 解析手法: #9 の内在次元をファインチューニングに使う。RoBERTa は約200個のパラメータ（ランダム射影で全空間へ戻す）だけで MRPC の 90% 性能に届く。事前学習が進むほど、またモデルが大きいほど内在次元が下がると報告し、これが LoRA の「更新は低ランクでよい」という仮説の根拠の一つになっている。
- 生徒の問いへの適用（提案）: LoRA のランクを振ったとき、時間的タスクの性能が飽和するランクを測るのは、内在次元の簡易版になる。
- 限界: #9 と同じ。

### 11. Weight Spectra Induced Efficient Model Adaptation（2025）
- 著者: Chongjie Si et al.
- 会議: **未確認**（arXiv のみ確認。Comments 欄に会議名なし）
- arXiv: [2505.23099](https://arxiv.org/abs/2505.23099)
- 解析手法（abstract に基づく）: Full FT 前後の重み行列を SVD で比べ、**上位の特異値が主に増幅され、残りはほぼ変わらない**こと、**上位の特異ベクトルがタスク固有の方向へ回転し、それ以外の部分空間は安定している**ことを報告する。そこから上位特異方向を学習可能に再スケールする手法を提案している。
- 生徒の問いへの適用（提案）: W0 の上位特異方向ごとに「特異値の変化」と「特異ベクトルの回転角」を分けて測ると、更新が「スケール変更」か「回転」かを分けられる（DoRA の分解を特異部分空間に移したもの）。
- 限界: 査読前。図は確認していない。

---

## B. タスクベクトル・モデルマージ系（ΔW を「ベクトル」として比べる）

### 12. Editing Models with Task Arithmetic
- 著者: Gabriel Ilharco et al.
- 会議: **ICLR 2023**（arXiv Comments: "In Proceedings of ... ICLR 2023"）
- arXiv: [2212.04089](https://arxiv.org/abs/2212.04089)
- 解析手法:
  - **タスクベクトル** τ = θ_ft − θ_pre。足すとタスク能力が加わり、引く（否定）とその能力が忘れられ、類推（A − B + C）もできる。
  - 異なるタスクのタスクベクトル間のコサイン類似度はほぼ0（ほぼ直交）で、似たタスク同士だけ類似度が上がる（Figure 5）。否定では、ランダム方向を引いた場合を対照に置く。
- 生徒の問いへの適用（提案）: (i) 通常クリップで学習した LoRA、(ii) フレームをshuffleしたクリップで学習した LoRA、(iii) 単一フレーム（静止画）で学習した LoRA を作り、ΔW 同士のコサイン類似度を比べる。(i) と (iii) がほぼ同じなら、(i) は静的情報しか学んでいない可能性が高い。(i) − (ii) を「時間ベクトル」とみなして足し引きし、順序 probe の精度が動くかを見る、という使い方もできる。
- 限界: パラメータ空間のコサインは粗い指標で、値が近くても機能が違うことはある（#6 の主張）。
- 図:
  - **Figure 1** — `https://arxiv.org/html/2212.04089v3/task_vectors.svg`
    (a) タスクベクトル＝学習後重み − 事前学習重み、(b) 否定、(c) 加算、(d) 類推、の模式図。
  - **Figure 5** — `https://arxiv.org/html/2212.04089v3/vision_cossim.svg`
    CLIP で、8つの画像分類タスクのタスクベクトル間コサイン類似度の行列。ほとんどほぼ0で、数字系（MNIST/SVHN/GTSRB）や衛星画像系（EuroSAT/RESISC45）だけ高い。

### 13. TIES-Merging: Resolving Interference When Merging Models
- 著者: Prateek Yadav et al.
- 会議: **NeurIPS 2023**（arXiv Comments: "Published at NeurIPS 2023"）
- arXiv: [2306.01708](https://arxiv.org/abs/2306.01708)
- 解析手法: タスクベクトルの (1) 大きさ上位 k% だけ残す（Trim）、(2) パラメータごとに符号を多数決で決める γ = sgn(Σ_t τ_t)（Elect）、(3) その符号に合う値だけ平均する（Disjoint merge）。解析として、上位数%のパラメータだけで性能がほぼ保たれること、**符号衝突**の割合がモデル数とともに増えることを示している。
- 生徒の問いへの適用（提案）: 時間情報ありの LoRA となしの LoRA の ΔW で、上位パラメータの重なりと符号衝突率を出せば、「時間情報のために追加で使われたパラメータ」があるかを粗く見られる。
- 限界: パラメータ単位の統計なので、意味の解釈には使えない。
- 図:
  - **Figure 1** — `https://arxiv.org/html/2306.01708v2/ties-merging12.svg`
    パラメータを四角、タスクベクトルを矢印（向き＝符号、長さ＝大きさ）で表した Trim → Elect sign → Disjoint merge の手順図。

### 14. Model Merging with SVD to Tie the KnOTS
- 著者: George Stoica et al.
- 会議: **ICLR 2025**（`iclr.cc/virtual/2025/poster/30908`、mlanthology `iclr/2025` で確認）
- arXiv: [2410.19735](https://arxiv.org/abs/2410.19735)
- 解析手法:
  - 複数の LoRA 更新を横に並べて SVD する: [ΔW^(1), …, ΔW^(n)] = U Σ [V^(1), …, V^(n)]。共通の基底 U Σ の上でタスクごとの係数 V^(i) を比べ、TIES などでマージする。
  - 解析として、**ΔW だけが作る出力**（ΔW_j^(i) x）同士の CKA を、別タスクで学習したモデル間で計算する。Full FT の更新同士は CKA が高く揃っているが、LoRA の更新同士は大きくずれる（KnOTS で整列させると揃う）。
- 生徒の問いへの適用（提案）: 「ΔW だけを通した出力」（ΔW_q x, ΔW_v x）を取り出して CKA を測る、という考え方は直接使える。例えば同じクリップの通常順とshuffle順で、**W0 側の出力ではなく ΔW 側の出力だけ**を比べると、LoRA の寄与が順序に反応しているかがわかる。最終特徴の cos ≈ 0.998 は W0 側の大きな静的成分に埋もれている可能性もあるので、ΔW 側だけを見る価値がある。
- 限界: CKA は線形な構造の類似しか見ない。
- 図:
  - **Figure 1** — `https://arxiv.org/html/2410.19735v1/figures/KnOTS_225ppi.png`
    層 j の各モデルの更新 ΔW_j^(i) を連結して SVD し、共通の U, Σ と各モデルの V^(i) に分けて、V 空間でマージする手順。
  - **Figure 2** — `https://arxiv.org/html/2410.19735v1/figures/ckas_225ppi.png`
    各 attention 層で、別タスクで学習した更新だけが作る出力の平均 pairwise CKA。(a) Full FT は高い、(b) LoRA は大きく低い、(c) KnOTS で整列させると高くなる。

### 15. Task Singular Vectors: Reducing Task Interference in Model Merging
- 著者: Antonio Andrea Gargiulo et al.
- 会議: **CVPR 2025**（arXiv Comments: "In Proceedings of ... CVPR 2025"）
- arXiv: [2412.00081](https://arxiv.org/abs/2412.00081)
- 解析手法:
  - タスクベクトルをモデル全体の平らなベクトルではなく、**層ごとの行列 Δ_i** として扱い SVD する（Task Singular Vectors, TSV）。各タスクの更新はごく少数の特異方向で表せる（低ランク性）。
  - タスク間干渉を **STI = ‖(U^⊤U − I) Σ (V^⊤V − I)‖_1** で定義する（U, Σ, V は全タスクの SVD を連結したもの）。異なるタスクの特異ベクトルが重なるほど値が大きい。ViT-B/32 では浅い層で STI が高く、深い層で低いと報告している。
- 生徒の問いへの適用（提案）: 「静止画 LoRA」と「動画 LoRA」の層ごとの特異ベクトルの重なり（STI 的な指標）を出せば、**どの層で動画 LoRA が静止画 LoRA と違う方向を持つか**を特定できる。違いが出る層に絞って probe や patching をかけられる。
- 限界: 重なり＝機能の重なりとは限らない。
- 図:
  - **Figure 3** — `https://arxiv.org/html/2412.00081v3/similarity_matrices_Layer0.png`
    ViT-B/32 の第1 attention 層で、8タスクの特異ベクトル同士の類似度行列。対角ブロックはタスク内、非対角ブロックはタスク間の類似度。非対角が明るいほど干渉が大きい。

---

## C. 表現（活性）を比較・解析する手法

### 16. SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability
- 著者: Maithra Raghu et al.
- 会議: **NeurIPS (NIPS) 2017**（arXiv Comments: "Accepted to NIPS 2017"）
- arXiv: [1706.05806](https://arxiv.org/abs/1706.05806)
- 解析手法: 2つの層の活性行列（データ点×ニューロン）について、(1) それぞれ SVD で分散の大半（例: 99%）を占める方向だけ残し、(2) その部分空間同士で CCA を行い、正準相関の平均 ρ̄ を類似度にする。層×層の類似度行列を学習ステップごとに描くと、**下の層から先に固まる**という学習ダイナミクスが見える。
- 生徒の問いへの適用（提案）: 層ごとに「LoRA なし vs あり」「通常順 vs shuffle」の SVCCA を出すと、LoRA の影響や順序の影響が出る層がわかる。ただし現在の推奨は #17 の CKA。
- 限界: CCA は可逆線形変換に不変すぎて、ネットワーク間比較で直感と合わないことがある（#17 の指摘）。
- 図:
  - **Figure 4** — `https://arxiv.org/html/1706.05806v2/dynamics_plots_crop.png`
    CIFAR-10 の conv net（上）と ResNet（下）で、学習の各時点の層と最終層の SVCCA 類似度を層×層で並べた図。下の層ほど早く最終表現に近づく。

### 17. Similarity of Neural Network Representations Revisited（CKA）
- 著者: Simon Kornblith et al.
- 会議: **ICML 2019**（arXiv Comments: "ICML 2019"）
- arXiv: [1905.00414](https://arxiv.org/abs/1905.00414)
- 解析手法:
  - 列を中心化した活性 X ∈ R^{n×p1}, Y ∈ R^{n×p2}（同じ n 個の入力）に対し、
    **線形CKA(X, Y) = ‖Y^⊤X‖_F² / (‖X^⊤X‖_F · ‖Y^⊤Y‖_F)**。
  - 直交変換と等方スケーリングには不変で、任意の可逆線形変換には不変ではない。そのため、ランダム初期化が違うネットワークの対応する層を正しく見つけられる（CCA・SVCCA・線形回帰はできない）。
- 生徒の問いへの適用（提案）: cos ≈ 0.998 は「1本の特徴ベクトルの比較」だが、CKA は「多数クリップ全体での表現構造の比較」。例えば **CKA(通常順の特徴群, shuffle 順の特徴群)** を層ごと・LoRA あり/なしで比べる。さらにトークン単位（時刻ごとのトークン）でも計算すると、最終プーリングで消える差を拾える可能性がある。
- 限界: CKA が高いことは「情報が同じ」を意味しない（attention transfer 論文 #23 も、CKA では拾えない類似があると注意している）。順序が線形に読めるかどうかは、probe で直接測る必要がある。
- 図:
  - **Figure 2** — `https://arxiv.org/html/1905.00414v4/tiny_cifar_comparison.png`
    初期化だけ違う CNN 同士の層×層類似度を、CKA・CCA・線形回帰・SVCCA で比べた図。CKA だけが「同じ層同士が最も似ている」という対角構造をはっきり示す。

### 18. Understanding Intermediate Layers Using Linear Classifier Probes
- 著者: Guillaume Alain, Yoshua Bengio
- 会議: **ICLR 2017（Workshop Track と認識。mlanthology は "ICLR 2017" と記載。本会議かワークショップかは一次ソースで未確認）**
- arXiv: [1610.01644](https://arxiv.org/abs/1610.01644)
- 解析手法: 凍結したネットワークの各中間層の特徴に、モデルとは独立に**線形分類器（probe）**を学習させ、その精度で「その層に、その情報が線形に読める形で存在するか」を測る。深い層ほど線形分離性が単調に上がることを示した。
- 生徒の問いへの適用（提案）: 最も直接的な道具。**順序（正順 vs 逆順、24通りの並べ替えの識別）、方向（左右移動）、速度（フレーム間隔 1/2/4 の識別）**を、層ごと・LoRA あり/なし（W0 だけのモデル）で probe する。「LoRA ありで probe 精度が上がる」ことが、LoRA が動的情報を足したことの最もわかりやすい証拠になる。対照として、静的外観で解ける課題（シーン分類など）でも probe する。
- 限界: probe が成功しても、モデルがその情報を**使っている**とは限らない（相関であって因果ではない）。probe の容量が大きすぎると probe 自身が学習してしまうので、線形・正則化つきにし、ランダム初期化モデルや LoRA なしモデルをベースラインに置く。
- 図: HTML 版は確認していない（古い論文のため図の選定は省略）。

### 19. Network Dissection: Quantifying Interpretability of Deep Visual Representations
- 著者: David Bau et al.
- 会議: **CVPR 2017 (Oral)**（arXiv Comments: "Oral presentation at CVPR 2017"）
- arXiv: [1704.05796](https://arxiv.org/abs/1704.05796)
- 解析手法: 各ユニット（チャネル）の活性マップを上位分位で二値化し、概念ラベル付きのセグメンテーションデータ（Broden）との **IoU** を計算する。IoU が閾値（0.04）を超えたユニットを、その概念の「検出器」と数える。ネットワーク全体で何種類の概念検出器があるかで、解釈可能性を定量化する。
- 生徒の問いへの適用（提案）: 動画版としては、LoRA の B の列（出力方向）や ΔW_v の上位特異方向ごとに「どのクリップ・どのトークンで強く発火するか」を集め、**動きの大きい領域（オプティカルフローの大きさ）との IoU** を測る、という形で応用できる。外観セグメントとの IoU が高く、動き領域との IoU が低ければ、静的な方向と解釈できる。
- 限界: 概念ラベルが用意されている概念しか見つからない。動き・時間の概念辞書は自前で用意する必要がある。

### 20. Revisiting Model Stitching to Compare Neural Representations
- 著者: Yamini Bansal, Preetum Nakkiran, Boaz Barak
- 会議: **NeurIPS 2021**（`proceedings.neurips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c` で確認）
- arXiv: [2106.07682](https://arxiv.org/abs/2106.07682)
- 解析手法: モデル A の下位層とモデル B の上位層を、学習可能な小さな層（1×1 conv や線形層）でつなぎ、性能がどれだけ落ちるか（stitching penalty）で表現の互換性を測る。CKA では見えない違いも出ると主張している。
- 生徒の問いへの適用（提案）: 「LoRA なし ViT の下位層 → LoRA あり ViT の上位層」をつないで、時間 probe や MoCo 検索の性能を比べれば、時間情報（もしあれば）が**どの層の LoRA で作られているか**を切り分けられる。
- 限界: stitching 層の学習自体が情報を補ってしまう可能性があるので、stitching 層は線形に限る。

### 21. What is being transferred in transfer learning?
- 著者: Behnam Neyshabur, Hanie Sedghi, Chiyuan Zhang
- 会議: **NeurIPS 2020**（NeurIPS proceedings PDF・mlanthology で確認）
- arXiv: [2008.11687](https://arxiv.org/abs/2008.11687)
- 解析手法: (1) 入力画像をブロック単位でシャッフルして、特徴再利用と低レベル統計の寄与を分ける。(2) 事前学習から始めたモデルは同じ loss basin にとどまるか（重み空間で補間して性能の谷を見る）。(3) **module criticality**（ある層だけを初期値側へ戻すと性能がどれだけ落ちるか）で層の重要度を測る。
- 生徒の問いへの適用（提案）: 「入力をシャッフルして性能がどれだけ落ちるか」というやり方は、フレーム順シャッフルの実験と同じ発想。module criticality は、**層ごとに LoRA をオフにして**時間 probe の精度がどれだけ落ちるかを測る形に置き換えられる。
- 限界: 画像分類での知見で、時間情報を直接扱ってはいない。

### 22. Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution
- 著者: Ananya Kumar et al.
- 会議: **ICLR 2022 (Oral)**（arXiv Comments: "ICLR (Oral) 2022"）
- arXiv: [2202.10054](https://arxiv.org/abs/2202.10054)
- 解析手法: 線形モデルでの理論と実験で、Full FT は学習データが張る部分空間の特徴は変えるが、その直交補空間（OOD 方向）の特徴はほぼ変えないため、ヘッドと特徴の不整合で「特徴が歪む」ことを示す。先に線形 probe を学習してから全体を学習する **LP-FT** を提案している。
- 生徒の問いへの適用（提案）: 「学習データに含まれる変動の方向しか特徴は変わらない」という見方は、MoCo の正例ペアに時間的な違いがほとんどなければ、時間方向の特徴が更新されない、という仮説の参考になる（仮説であり、この論文の結果ではない）。
- 限界: 理論は線形モデルでの話。

### 23. On the Surprising Effectiveness of Attention Transfer for Vision Transformers
- 著者: Alexander C. Li et al.
- 会議: **NeurIPS 2024**（arXiv Comments: "NeurIPS 2024"）
- arXiv: [2411.09702](https://arxiv.org/abs/2411.09702)
- 解析手法:
  - 事前学習 ViT の **attention map だけ**を、ランダム初期化の生徒に (a) そのままコピー（Attention Copy）または (b) 蒸留（Attention Distillation）し、それ以外の重みは一から学習する。これでトークン間の情報の流れ（inter-token, attention が決める）とトークン内の特徴（intra-token）を切り分ける。
  - CKA で層ごとの特徴の類似度、**Jensen–Shannon divergence (JSD)** でヘッドごとの attention map の類似度を測る。attention 転移モデルは、特徴は fine-tuned モデルとあまり似ていない（CKA はスクラッチと同程度）が、性能は同等になる。
- 生徒の問いへの適用（提案）: LoRA を Q/V に入れているので、**Q の LoRA は attention map（どのフレームのどのトークンを見るか）を変え、V の LoRA は運ばれる中身を変える**、と分けて考えられる。具体的には、(i) LoRA あり/なしで**フレーム間 attention の割合**（時刻 t のトークンが別の時刻のトークンにどれだけ注意するか）を比べる、(ii) shuffle 入力で attention map の JSD を測る、(iii) LoRA ありモデルの attention をなしモデルに入れ替える（またはその逆）、ことで、時間情報が「見る場所」と「運ぶ中身」のどちらに入ったかを切り分けられる。
- 限界: attention map は情報の流れの一部しか表さない（残差経路や MLP は見ない）。
- 図:
  - **Figure 2** — `https://arxiv.org/html/2411.09702v1/attention_transfer_v3.png`
    Attention Copy（教師の attention map を生徒にそのまま貼る）と Attention Distillation（生徒が自分の attention を計算し、教師の map に近づける損失を加える）の2方式の図。H はヘッド数、L は層数。
  - **Figure 5** — `https://arxiv.org/html/2411.09702v1/cka_smaller.svg`
    fine-tuned MAE との層ごとの CKA。attention 転移モデルの特徴は fine-tuned モデルとあまり似ておらず、スクラッチモデルと同程度。「性能が同じでも表現は違う」ことを示す。

---

## D. 因果的介入・モデル差分（model diffing）

### 24. Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking
- 著者: Nikhil Prakash et al.
- 会議: **ICLR 2024**（arXiv Comments: "ICLR 2024"）
- arXiv: [2402.14811](https://arxiv.org/abs/2402.14811)
- 解析手法:
  - Llama-7B で entity tracking を担う回路（ヘッドの集合）を **path patching** で特定し、ファインチューニング版（Vicuna, Goat, FLoat）でも同じ回路が働くかを調べる。
  - **DCM (Desiderata-based Component Masking)**: 「値だけ変えた反実仮想文」「位置だけ変えた反実仮想文」などの desiderata を用意し、各ヘッド群がどの情報（値/位置）を運んでいるかをマスク学習で特定する。
  - **CMAP (Cross-Model Activation Patching)**: ファインチューニング後モデルのあるヘッド群の活性を元のモデルに貼り付け、性能向上がどの部品から来るかを特定する。結論は「ファインチューニングは新しい回路を作るより、既存の回路（特に Value Fetcher ヘッド）を強化する」。
- 生徒の問いへの適用（提案）: とても相性が良い。(i) **CMAP の動画版**: LoRA ありモデルの第 l 層の出力（または Q/V だけ）を LoRA なしモデルに貼り付け、順序 probe や順序検索の性能がどの層で回復するかを見る。(ii) **DCM 的 desiderata**: 「同じフレーム集合で順序だけ違う」「同じ順序で外観だけ違う」反実仮想クリップのペアを作り、ヘッドごとに「順序の違いに反応するか、外観の違いに反応するか」を分類する。0.998 という結果を、ヘッド単位まで分けて調べられる。
- 限界: 解析のコストが高い。対象とする振る舞い（タスク）を明確に定義する必要がある。
- 図:
  - **Figure 1** — `https://arxiv.org/html/2402.14811v1/circuit_flow.png`
    Llama-7B の entity tracking 回路。4つのヘッド群（A, B, C, D）が、最終トークン・クエリラベル・前のクエリラベルの位置にある。情報がどの位置からどの位置へ流れるかを読む。
  - **Figure 4** — `https://arxiv.org/html/2402.14811v1/cmap_eval.png`
    CMAP の結果。Goat-7B / FLoat-7B の回路部品の活性を Llama-7B に貼り付けると、Value Fetcher ヘッドだけで性能がファインチューニング後モデル並みに上がる。どの部品が改善の原因かを読む。

### 25. Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning
- 著者: Julian Minder et al.
- 会議: **NeurIPS 2025**（arXiv Comments: "Accepted at 39th Conference on Neural Information Processing Systems (NeurIPS 2025)"）
- arXiv: [2504.02922](https://arxiv.org/abs/2504.02922)
- 解析手法: base モデルと chat モデルの活性を一緒に再構成する辞書（**crosscoder**、SAE の2モデル版）を学習し、各潜在因子のデコーダノルムの比 Δ_norm から「base にしかない因子」「chat にしかない因子」を見つける。L1 損失では見かけ上 chat 専用に見える因子が出る（sparsity artifact）ことを示し、**Latent Scaling** という指標と BatchTopK crosscoder で、本当に chat 固有の因子を見つける。
- 生徒の問いへの適用（提案）: LoRA なし ViT と LoRA あり ViT の活性で crosscoder（または SAE）を学習し、「LoRA ありにしかない因子」を集めて、それがどんなクリップ（動きの大きいクリップか、特定の外観か）で発火するかを見る。モデル差分を因子単位で解釈する、現在いちばん新しい手法の一つ。
- 限界: 学習コストが高く、アーティファクトへの注意が必要（論文の主題そのもの）。ViT での実例は本論文にはない。

### 26. Convergent Linear Representations of Emergent Misalignment（2025）
- 著者: Anna Soligo, Edward Turner, Senthooran Rajamanoharan, Neel Nanda
- 会議: **未確認**（arXiv のみ確認）
- arXiv: [2506.11618](https://arxiv.org/abs/2506.11618)
- 解析手法: **rank-1 LoRA** を9個だけ使ってモデルを変化させ、各 rank-1 アダプタを「入力から読む方向 a と書き込む方向 b、その間のスカラー」として直接解釈する（ΔW x = b (a^⊤ x)）。スカラー a^⊤ x がどの入力で大きくなるかを調べてアダプタの役割を分類し、活性から抽出した方向でアブレーションして因果的に確かめる。
- 生徒の問いへの適用（提案）: Q/V に **rank-1（または少ランク）LoRA** を入れて学習し直すと、各アダプタのスカラー a^⊤ x を「時刻ごと・トークンごと」に記録できる。このスカラーが**フレーム順やフレーム間の差に応じて変わるか**を見るのは、とても解釈しやすい解析になる。
- 限界: 査読前。rank-1 に制限すると表現力が落ちるので、通常ランクの LoRA と同じ振る舞いになるとは限らない。

### 27. Learning to Interpret Weight Differences in Language Models（2025）
- 著者: Avichal Goel, Yoon Kim, Nir Shavit, Tony T. Wang
- 会議: **未確認**（arXiv のみ確認。OpenReview に投稿版 PDF はあるが採択は未確認）
- arXiv: [2510.05092](https://arxiv.org/abs/2510.05092)
- 解析手法: ラベル付きの合成重み差分で **DIT アダプタ**を学習させ、ファインチューニング済みモデルに付けると、そのモデルが「自分がどう変わったか」を自然言語で説明できるようにする（Diff Interpretation Tuning）。
- 生徒の問いへの適用（提案）: 直接は使いにくい（言語モデル用）。「重み差分をラベル付きの合成例から読み解くメタモデル」という考え方の参考程度。
- 限界: LLM 限定・査読前。

---

## E. LoRA を「解釈可能な方向」「ブロックごとの役割」として扱う例

### 28. Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models
- 著者: Rohit Gandikota et al.
- 会議: **ECCV 2024**（ECVA `papers_ECCV/html/5660_ECCV_2024_paper.php`、DOI 10.1007/978-3-031-73661-2_10）
- arXiv: [2311.12092](https://arxiv.org/abs/2311.12092)
- 解析手法: 拡散モデルで、対になる概念（c+ と c−、例: 若い/年老いた）について、ターゲット概念 c_t のスコアを
  ε*(x, c_t) = ε(x, c_t) + η Σ_p [ε(x, (c+, p)) − ε(x, (c−, p))]
  に近づけるよう LoRA を学習する（p は年齢を変えるときの人種・性別など、保持したい属性）。学習した LoRA は**1つの概念方向**になり、推論時のスケール α で連続的に強弱を変えられる。低ランク制約と保持概念によって、他の属性との絡み合いが減ることを示している。
- 生徒の問いへの適用（提案）: 逆向きの発想として、「**正順 vs 逆順**」「**速い vs 遅い**」のような対概念を正例/負例にして LoRA を学習すれば、時間方向を明示的に表す LoRA を作れる。それを今の MoCo LoRA の ΔW と比べ（部分空間類似度 #1、コサイン #12）、今の LoRA に時間方向がどれだけ含まれているかを測る「物差し」にできる。
- 限界: 生成モデルでの手法なので、識別モデル（ViT＋MoCo）にそのまま持ち込めるわけではない。
- 図:
  - **Figure 2** — `https://arxiv.org/html/2311.12092v2/method.png`
    属性 c+ を強め c− を弱める guided score で LoRA を学習する仕組み。ガウスノイズから時刻 t までを部分的にデノイズし、そのサンプル x_t 上で損失をとる。

### 29. Implicit Style-Content Separation using B-LoRA
- 著者: Yarden Frenkel, Yael Vinker, Ariel Shamir, Daniel Cohen-Or
- 会議: **ECCV 2024**（ECVA paper 01549、DOI 10.1007/978-3-031-72684-2_11）
- arXiv: [2403.14572](https://arxiv.org/abs/2403.14572)
- 解析手法:
  - SDXL の各 transformer ブロックに、**そのブロックだけ別のテキストプロンプト p̂ を注入**し、生成画像がどう変わるかを CLIP 類似度で測る（400組のプロンプトで平均）。これで「content を決めるブロック（W_0^2, W_0^4）」と「style／色を決めるブロック（W_0^5）」を特定する。
  - その2ブロックだけに LoRA を学習させると、1枚の画像から style と content を分けて取り出せる。
- 生徒の問いへの適用（提案）: 「層（ブロック）ごとに入力を差し替えて役割を調べる」方法は、そのまま ViT に使える。例えば**第 l 層にだけ shuffle したクリップのトークンを入れ、残りの層には通常のクリップを入れる**（またはその逆）と、どの層が順序の影響を受けるかがわかる。また、LoRA を特定の層だけに入れた学習で時間感度がどう変わるかも、B-LoRA 的な層選択の分析になる。
- 限界: 生成モデルでの分析。ブロックの役割はモデルによって違う。
- 図:
  - **Figure 3** — `https://arxiv.org/html/2403.14572v2/style_injection.png`
    SDXL の構造と、テキストベースの分析方法の図。i 番目のブロックにだけ別のプロンプト p̂ を入れ、他のブロックには p を入れて、生成画像の変化からブロックの役割を推定する。

### 30. Rethinking Low-Rank Adaptation in Vision: Exploring Head-Level Responsiveness across Diverse Tasks（Heart-LoRA）
- 著者: Yibo Zhong et al.
- 会議: **WACV 2025**（CVF Open Access `WACV2025/papers/Zhong_Rethinking_Low-Rank_Adaptation_in_Vision_...` で確認。依頼の対象会議外だが ViT＋LoRA で直接関係するので載せた）
- arXiv: [2404.08894](https://arxiv.org/abs/2404.08894)
- 解析手法: ViT の LoRA をヘッド単位に分け、**Taylor 展開による重要度スコア（TIS）**を各ヘッドで計算・累積し、タスクに反応するヘッドだけを有効にする。ヘッドの一部（例: Caltech101 で25%、Flowers102 で50%）を止めても attention map がほぼ変わらない、というヘッド冗長性の分析も含む。
- 生徒の問いへの適用（提案）: Q/V の LoRA を**ヘッド単位**で見て、時間 probe の損失（または「通常順 vs shuffle の特徴差」）に対する Taylor 重要度を出せば、時間に効くヘッドがあるかどうかを安く探せる。#24 のヘッド単位の解析の前段に使える。
- 限界: 重要度は一次近似。

---

## F. 本研究への当てはめ: 解析の組み立て案（提案）

以下は上の論文の手法を組み合わせた**提案**であり、どの論文もこの設定での結果は報告していない。

1. **まず「情報があるか」を probe で確かめる（#18）**
   順序（正/逆、24通りの並べ替え）、方向、速度（フレーム間隔）を、層ごと・トークン単位とプーリング後の両方で probe する。比較は必ず **W0 のみ（LoRA なし）／LoRA あり／ランダム初期化**の3つ。LoRA で精度が上がらなければ、LoRA は動的情報を足していない、と言いやすくなる。
2. **cos ≈ 0.998 を正しく読む（#14, #17）**
   1本のベクトルのコサインは、大きな静的成分に支配されやすい。(a) クリップ間の通常の変動に対して shuffle による変化がどれくらいか（例: 変化量 / クリップ間距離）、(b) **ΔW 側だけの出力**（ΔW_q x, ΔW_v x）で shuffle 前後を比べる、(c) CKA で集団として比べる、の3つで見直す。
3. **ΔW の幾何を見る（#1, #3, #5, #6, #11）**
   層ごとの ΔW_q, ΔW_v について、特異値スペクトル、増幅率、ΔM/ΔD、intruder dimension を出す。そのうえで上位特異方向に「時間位置埋め込みの差 p_t − p_t'」「フレーム差分トークン」「フレーム平均トークン」を通したときの応答の比を測る。時間成分にほとんど反応しないなら、静的な方向と解釈できる。
4. **対照 LoRA と比べる（#12, #15, #28）**
   静止画（1フレームを4回）LoRA、shuffle クリップ LoRA、通常 LoRA を学習し、ΔW のコサイン・部分空間類似度・層ごとの STI 的な重なりを比べる。通常 LoRA ≈ 静止画 LoRA なら、時間情報は学ばれていない。
5. **Q と V、attention を分けて見る（#23, #30）**
   フレーム間 attention の割合、shuffle 時の attention map の JSD、ヘッドごとの Taylor 重要度を見る。
6. **因果的に確かめる（#24, #20, #21, #29）**
   層ごとに LoRA をオン/オフ、LoRA ありの活性をなしのモデルへパッチング（CMAP）、層ごとに shuffle トークンを注入、で順序 probe の性能がどこで変わるかを特定する。
7. **より細かい解釈（#25, #26）**
   必要なら、rank-1 LoRA での再学習（スカラー a^⊤x の時間変化）や crosscoder による「LoRA ありにしかない因子」の抽出まで進める。

全体を通した注意: 重み空間の解析（3, 4）は「どこが・どれだけ・どの向きに変わったか」はわかるが、「何の情報か」はわからない。意味の判断は、入力の反実仮想操作（shuffle・reverse・速度変更）と probe／パッチング（1, 2, 6）で行う必要がある。

---

## 図URLの検証ログ（2026-11-12, `curl -sI -m 20`）

| URL | HTTP | content-type |
|---|---|---|
| https://arxiv.org/html/2106.09685v2/qv8_qv64.png | 200 | image/png |
| https://arxiv.org/html/2106.09685v2/w_vs_delta_w.png | 200 | image/png |
| https://arxiv.org/html/2402.09353v6/qproj_ft_vs_lora_vs_MADoRA_pattern.svg | 200 | image/svg+xml |
| https://arxiv.org/html/2402.09353v6/img/q_delta_m_d.png | 200 | image/png |
| https://arxiv.org/html/2402.16842v2/figures/sametask_diffseed_cca_mean.png | 200 | image/png |
| https://arxiv.org/html/2402.16842v2/figures/difftask_sameseed_cca_mean.png | 200 | image/png |
| https://arxiv.org/html/2402.16842v2/figures/difftask_diffseed_cca_mean.png | 200 | image/png |
| https://arxiv.org/html/2405.09673v2/figures/svd_summary_starcoder_all.png | 200 | image/png |
| https://arxiv.org/html/2405.09673v2/figures/single_layer_spectrum_starcoder_cropped.png | 200 | image/png |
| https://arxiv.org/html/2410.21228v3/figures/magicoder_full_vs_lora_r64.png | 200 | image/png |
| https://arxiv.org/html/2410.21228v3/figures/figure_1.png | 200 | image/png |
| https://arxiv.org/html/2404.02948v4/lora_pissa.png | 200 | image/png |
| https://arxiv.org/html/2406.09044v3/overview.png | 200 | image/png |
| https://arxiv.org/html/2212.04089v3/task_vectors.svg | 200 | image/svg+xml |
| https://arxiv.org/html/2212.04089v3/vision_cossim.svg | 200 | image/svg+xml |
| https://arxiv.org/html/2306.01708v2/ties-merging12.svg | 200 | image/svg+xml |
| https://arxiv.org/html/2410.19735v1/figures/KnOTS_225ppi.png | 200 | image/png |
| https://arxiv.org/html/2410.19735v1/figures/ckas_225ppi.png | 200 | image/png |
| https://arxiv.org/html/2412.00081v3/similarity_matrices_Layer0.png | 200 | image/png |
| https://arxiv.org/html/1706.05806v2/dynamics_plots_crop.png | 200 | image/png |
| https://arxiv.org/html/1905.00414v4/tiny_cifar_comparison.png | 200 | image/png |
| https://arxiv.org/html/2411.09702v1/attention_transfer_v3.png | 200 | image/png |
| https://arxiv.org/html/2411.09702v1/cka_smaller.svg | 200 | image/svg+xml |
| https://arxiv.org/html/2402.14811v1/circuit_flow.png | 200 | image/png |
| https://arxiv.org/html/2402.14811v1/cmap_eval.png | 200 | image/png |
| https://arxiv.org/html/2311.12092v2/method.png | 200 | image/png |
| https://arxiv.org/html/2403.14572v2/style_injection.png | 200 | image/png |

（Asymmetry の Figure 1 の3枚を1つの図として数えると、図は 25 枚 / 27 URL）

## 未確認・注意事項

- **Alain & Bengio (2017)**: ワークショップトラックと認識しているが、OpenReview は自動アクセスが拒否された（challenge 403）。mlanthology は "ICLR 2017" とだけ記載している。
- **LoRA (ICLR 2022)**: mlanthology と OpenReview forum ID で確認。OpenReview 本体は閲覧していない。
- **Illusion of Equivalence**: arXiv の Comments には会議名がない。NeurIPS 2025 は neurips.cc virtual のポスターページで確認した。
- **#11 Weight Spectra, #26 Soligo et al., #27 Goel et al.**: 会議は未確認（arXiv のみ）。#11 と #26 の内容は abstract に基づく。
- 図のキャプション要約は、arXiv HTML の figcaption を訳して要約したもの。数値の読み取りは行っていない。
