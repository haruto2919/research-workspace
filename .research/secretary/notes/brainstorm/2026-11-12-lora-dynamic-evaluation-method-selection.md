---
date: 2026-11-12
project: sequential-video-lora-analysis
source_todo: null
topic: lora-dynamic-evaluation-method-selection
status: exploratory
tags: [brainstorm, research, lora-analysis, static-dynamic, evaluation, probe]
---

# 自分のLoRAが動的情報を学んだかを確かめる手法の選定

## 出発点

文献調査（[解説ページ](https://claude.ai/artifact/JAmkhCe6u18iHRCGZaqJ62)、[調査の要点](../../../lab/projects/sequential-video-lora-analysis/papers/2026-11-12-static-dynamic-evaluation-literature-survey.md)）の約130本から、今の設定で「LoRAが動的情報を学んだ」ことを確かめるのに使える手法を選び、根拠とやり方の案を整理する。

## 読み込んだ文脈

- README（last_updated 2026-11-11）、直近MTG（2026-10-22：LoRAが動的情報を保持するかは今後の解析で確認する。strideはLoRAの変化を調べる比較条件）。
- 並び順への反応の評価（[spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-11-03-temporal-order-sensitivity-evaluation-spec.md)、[逐次clip LoRAを加えた結果](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-11-temporal-order-sensitivity-evaluation-g2-gpu-evidence.md)）。
- 今日の壁打ちメモ（[13年分の調査](2026-11-12-lora-dynamic-information-evaluation-13-year-survey.md)）。
- 今日のTODOファイルは未作成。

## 前提として使える既存の資産

| 名前（このメモ内） | 中身 | 記録上の記号 |
| --- | --- | --- |
| 学習前 | LoRA初期状態＝元のViT | A |
| 単一フレームLoRA | 単一フレームでMoCo学習、459step | B-LoRA |
| ランダム順clip LoRA | 4フレームclip（0.25秒間隔）、459step | C-LoRA |
| 逐次clip LoRA | sequential-loader、stride 16、6,255step（途中checkpointあり） | G2-LoRA |
| Full FT版 | 単一フレーム・clipのFull FT、各459step | B-FT、C-FT |

- 評価器 `temporal_order_eval` がある：ActivityNet validationの2,000 clip、shuffle・reverse・static・same_video・other_video、CLS（backbone出力）とqの両方、per-clip出力とpaired bootstrap。
- 時間位置埋め込みは固定のsin-cos（学習されない）。LoRAはQ/Vのみ。したがって「時間位置をどう使うか」はQ/VのLoRAだけが変えられる。
- 重要な事実：**学習前のモデルでもshuffleのcosは0.98（CLS）**。LoRAの寄与は、学習前のモデルとの差として測る必要がある。

## 評価軸

1. 「動的情報を学んだ」という主張への寄与（証拠の5段のどこを埋めるか）
2. 既存checkpointでできるか（追加の学習やデータが要るか）
3. 代わりの説明（静的な手がかり、分布外、学習量の違い）を排除できるか
4. 結果が出なかったときにも情報が得られるか

## 採用候補（優先順）

### 1. 層ごとの時間probe（能力）

- 根拠：TIME（ICML 2022）は、ブロックごとの凍結特徴から順序を線形probeし、深い層ほど順序情報が消えることを示した。Arrow of Time（CVPR 2018）、OPN（ICCV 2017）、Pace Prediction（ECCV 2020）、Chirality in Action（NeurIPS 2025）も凍結特徴の上の線形分類で評価している。cos≈1でも、順序が小さな成分に一貫して入っていれば読み出せるので、今の結果と矛盾せずに「ある／ない」を判定できる。
- やり方の案：
  - タスク：(a) 順再生／逆再生の2値、(b) 4フレームの並べ方の分類（24通り、または先頭フレームの位置の4値）、(c) フレーム間隔の分類（例：stride 8／16／32。同じ開始時刻から取る）。
  - 特徴：各ブロックのCLSと、フレームごとのトークン平均を4つ並べたもの。
  - データ：validationの動画を**動画単位で**train/testに分ける（同じ動画のclipが両方に入ると、動画の見た目で解けてしまう）。1動画から複数clipを取って本数を増やす。
  - probe：ロジスティック回帰（正則化を固定）。5-fold、精度とbootstrap区間。
  - 比較：学習前／単一フレームLoRA／ランダム順clip LoRA／逐次clip LoRA。逐次clip LoRAは学習時の間隔（約0.53秒）でも評価する。
  - 判定：LoRAありが学習前と単一フレームLoRAの両方を上回るか。中間層だけで読めて最終層で消えるか。
- 注意：フレームごとのトークンを並べたprobeは、元のViTでもフレーム間の差から解ける可能性がある。だから学習前との差で見る。

### 2. 陽性対照（評価手順が動的情報を検出できることの確認）

- 根拠：今の目的関数は順序を区別する圧力を持たない（FAME、DSM、TE、C-DSVAEの議論）ので、「検出されない」結果が出やすい。Test of Time（CVPR 2023）が対照タスクで評価そのものの妥当性を確かめたように、「ない」と言うには、ある場合に検出できる手順であることを示す必要がある。
- やり方の案：同じViT＋Q/V LoRAを、順再生／逆再生や並べ方の分類で**直接**短く学習した「時間を教えたLoRA」を作る（または公開されている動画モデルを使う）。そのモデルで採用候補1・4・5の評価が反応することを先に確認する。
- 注意：時間を教えたLoRAをprobeと同じタスクで学習すると自明に高くなる。評価は別のタスク（例：学習は順／逆、評価は間隔分類）で行う。

### 3. 静的probeと差分特徴（選択性、Leakage Gap）

- 根拠：DBSE（ICML 2024）とSPYL（ICML 2023）は、潜在に「静的ラベル」「動的ラベル」の分類器を両方かけ、漏れを Leakage Gap で測る。DBSEは1フレーム目を引いた差分を動的な経路にした。動的probeの改善が「動画ドメインへの一般的な適応」でないことを示すのに使える。
- やり方の案：
  - 静的probe：動画ID（同じ動画の別clipかどうか）、ActivityNetの行動クラス（要確認：validationのラベルの取り出し）。
  - 動的probe：採用候補1と同じ。
  - 入力する特徴：(i) CLS、(ii) LoRAの差分 r = h<sup>LoRA</sup> − h<sup>学習前</sup>、(iii) フレームごとの差分 g<sub>t</sub> − g<sub>1</sub>。
  - 判定：差分特徴で静的probeが下がり動的probeが残るか。LoRAで動的probeだけが上がるか、両方上がるか。

### 4. 静的ペアと動的ペアの比較（選択性）

- 根拠：Kowal et al.（CVPR 2022）は、shuffleで作る「静的ペア」と、stylizationで作る「動的ペア」の類似度（相互情報量）を層ごとに比べた。データはStylized ActivityNetで、この研究と同じ。今のshuffleのcosは、このうち静的ペア側だけを見ている。
- やり方の案：既存の評価器に「見た目だけ変えた条件」（色変換、グレースケール、可能ならstylization）を足す。other_videoのcosで正規化した類似度（(cos − cos<sub>other</sub>) / (1 − cos<sub>other</sub>)）で、静的ペアと動的ペアを並べる。層ごと、LoRAあり／なしで比べる。
- 注意：色変換は「見た目をどれだけ変えたか」が弱い。Stylized ActivityNetが入手できるか要確認。

### 5. LoRAへの介入（因果）

- 根拠：Illusion of Equivalence（NeurIPS 2025）は特異値の倍率を連続的に変えて機能を測り、上位の特異方向でも能力に必要とは限らないことを示した。Separate Motion from Appearance（ACM MM 2025）とMotion Inversion（SIGGRAPH 2025）は、V側の更新が見た目を運びやすいと報告している。Neyshabur et al.（NeurIPS 2020）は層を初期値に戻して重要度を測った。
- やり方の案：
  - LoRAの出力に係数 α（0, 0.25, 0.5, 0.75, 1.0, 1.25）を掛け、採用候補1・3を測り直す。
  - Qだけ、Vだけ有効にする。前半・中間・後半の層だけ外す。
  - ΔWの上位特異成分を1つずつ除き、同じエネルギーのランダムな向きの除去と比べる。
  - 判定：動的probeだけが選択的に、αに応じて段階的に変わるか。
- 注意：採用候補1で動的probeの改善が出ない場合、この介入は「何も落ちない」になる。採用候補1の結果を見てから行う。

### 6. ΔWの比較（既存checkpoint、CPUのみ）

- 根拠：Task Arithmetic（ICLR 2023）、LoRA（ICLR 2022）の部分空間類似度、KnOTS（ICLR 2025）。単一フレームLoRAとclip LoRAが既にあるので、追加の学習なしで「clipで学習したLoRAは単一フレームLoRAと何が違うか」を見られる。
- やり方の案：層ごとのΔW = sBA（AとBを別々に見ない）について、cos類似度、上位特異部分空間の類似度 φ、特異値スペクトル。違いが大きい層を、採用候補1・5で重点的に調べる。
- 注意：459stepと6,255stepで学習量が違う。同じ条件でのseed間のばらつき（Asymmetry in LoRA, ICML 2024：Aは初期化に強く依存）を基準に置かないと、違いの大きさを解釈できない。

### 7. シャッフルしたclipで学習したLoRA（学習条件の対照、再学習あり）

- 根拠：MME（CVPR 2023）は、シャッフル動画でVideoMAEを学習しても下流性能が変わらないことから「時間を使っていない」と示した。TRN（ECCV 2018）も順序どおり／シャッフルで学習して比べた。最も強い対照実験。
- やり方の案：ランダム順clip LoRA（459step）と同じ設定で、clip内のフレームを毎回シャッフルして学習する。2〜3 seedずつ。採用候補1・3・6で比べる。
- コスト：459stepのclip学習は約82分だった。seedを含めて数回分。
- 判定：正しい順のLoRAとシャッフル学習のLoRAが同等なら、今の学習は時間を使っていない。

## 保留

- **SSv2・chiral pairでの外部評価**（Chirality in Action、EVL/AIMの2×2表）：外部妥当性は高いが、データの取得と前処理のコストがある。採用候補1で差が出たら行う。
- **見た目と動きを独立に変えた合成動画**（Tubelet-Contrast、Test of Time）：完全に統制できるが、画像ViTにとって分布外。補助として。
- **背景差し替え**（FAME、SCUBA）：人物のセグメンテーションが必要。採用候補4の色変換で代用する。
- **attentionの解析**（Attention Transfer, NeurIPS 2024）：別の時刻のトークンへのattentionの割合、shuffle時のattention mapの距離。Q側LoRAの寄与を見るのに有用。採用候補5と合わせて検討。
- **ストリームの順序の効果の比較**（Carreira et al., CVPR 2024）：clip内の順序の効果が確認できてから。

## 棄却寄り

- 特異値の大きさだけで動作概念の数を数える（Illusion of Equivalenceの結果に反する）。
- ActivityNet全体の行動分類の線形probeだけで判断する（静的に解ける。Only Time Can Tell、DSM）。
- Projector後のqだけで判断する（CLSで見る）。
- AFD（外観を消した動画）を主評価にする（画像ViTにとって分布外が強い）。
- video-LLMベンチマーク、VTCD、crosscoder（設定に合わない、または重い）。

## 反例・代わりの説明

- 学習前のモデルも順序に鈍い（shuffleのcos 0.98）。LoRAが「鈍さを強めた」可能性もあり、能力が下がる結果もあり得る。
- 逆再生の判別は、圧縮や撮影の手がかりで解ける可能性がある（Arrow of Time）。動きの大きいclipや固定カメラのclipで分けて集計する。
- 逐次clip LoRAとランダム順clip LoRAの違いは、学習量・フレーム間隔・読み方が混ざっている（既存記録の注意と同じ）。

## 未解決の問い

- ActivityNet validationの行動ラベルと区間を、評価clipに対応づけられるか。
- probeの学習に使うclip数（1動画から何本取るか）。
- 陽性対照を自作するか、公開モデルを使うか。
- 層ごとの特徴をどこで取り出すか（ブロック出力のCLS、フレームごとのトークン平均）。

## 次にユーザーが決めること（候補）

- 採用候補1〜3（既存checkpointだけでできる）を、まとめて1つの評価specにするか。
- 採用候補7（シャッフルclipで学習したLoRA）を、評価と並行して先に回すか。

## spec引き継ぎ候補（まだspecではない）

- 中心の問い：LoRAが元のViTに追加した表現から、順序・向き・速度を読み出せるか。それは静的な手がかりの改善で説明できないか。
- 対象モデル：学習前、単一フレームLoRA、ランダム順clip LoRA、逐次clip LoRA（＋陽性対照）。
- 評価：層ごとの時間probe（順／逆、並べ方、間隔）、静的probe（動画ID、行動クラス）、差分特徴、Leakage Gap。
- 未決：データ分割の単位と本数、probeの形と正則化、特徴の取り出し位置、陽性対照の作り方。

---

## 追記（2026-11-12）：小林さんの卒論を反映

- 対象：小林優斗「動作認識における静的バイアス低減のための静的情報と動的情報の分離」（2024年度卒論）。
- 原本のPDFはワークスペースにない。内容は [2026-10-26 中間発表の背景メモ](2026-10-26-midterm-background-issue-purpose.md) の要約（PDFから抽出して読んだもの）に基づく。数値は原本と照合していない。
- 上の選定では考慮していなかった。

### この評価選定に効く点

1. **「特徴が離れた」は「動的情報を得た」ではない（手法1の教訓）**
   - GRLで1フレームでは動作を当てられないようにした結果、画像入力と動画入力の出力のcosはほぼ0になった。しかしバイアス指標は改善しなかった。
   - 今回は逆向き（cos≈1）だが、同じ教訓になる。cosの大小は、動的情報の有無の判定に使えない。機能で測る採用候補1・3を主評価にする根拠を、研究室の先行事例で補強できる。
   - LoRAの差分 r = h^LoRA − h^学習前 が学習前の特徴と「違う」ことも、それだけでは動的の証拠にならない。差分特徴には静的probeもかけ、Leakage Gapで見る（採用候補3）。
2. **仮定が本研究と同じ**
   - 「画像と動画で共通な特徴＝静的、動画に固有な特徴＝動的」。本研究の「元の画像ViT＝静的、動画で学習したΔW＝動的」と同じ発想である。
   - その仮定が特徴空間の分離ではうまくいかなかった、という結果は、本研究で「ΔWに入ったものを対照実験で確かめる」必要性の根拠になる。
3. **評価に使った指標を借りられる（新しい採用候補）**
   - 1フレーム入力と複数フレーム入力の精度の比（ベースラインは 54.80% ／ 70.93%、約8割を1フレームで出せる）。
   - BOR（背景だけの動画での性能比、73.47%）と HOR（人物だけの動画での性能比、18.79%）。
   - 福沢さんの修論の SHAcc ／ SBErr（ActionSwapの背景入れ替え動画）も同じ系統。
   - 本研究での使い方の案：ActivityNetの行動クラス、またはKinetics400で線形probeを学習し、通常clip・1枚×4のclip・背景だけ・人物だけ・背景入れ替えで精度を比べる。比較対象は学習前とLoRAあり。LoRAで「1フレーム比」と「BOR」が下がり「HOR」が上がれば、静的な手がかりへの依存が減った（段3の証拠）と言える。
   - 先に「保留」とした背景差し替えは、**研究室内に人物マスクや背景入れ替えのデータ・コードがあれば、低コストで採用候補に上がる**。要確認。
4. **評価データで効果が変わった（手法2）**
   - Kinetics400では効果がなく、Temporal-32（時間が必要なクラスの集合、Only Time Can Tellと同系統）では効果があった。
   - 本研究でも、ActivityNet全体ではなく、時間が必要なclipやクラスに絞った評価を並べるべき、という根拠になる。Temporal-32を研究室で使える状態か確認したい。
5. **ベースラインの構成が本研究の静的な対照になる**
   - ImageNet-21k ViT＋フレームごとの特徴の平均（late fusion）。平均なので並べ替えに対して構造的に不変である。
   - 採用候補1の時間probeで、この構成（学習前ViTのフレームごとCLSの平均）がチャンスになることを、評価の妥当性チェックとして最初に確かめる。

### 注意

- 小林さんの結果は、Kinetics400の教師あり学習で、LoRAも逐次学習も使っていない。数値を本研究と直接は比べない。
- 学位論文で査読付きではない。研究室内の先行研究として引用する。

### 確認したいこと

- 小林さん・福沢さんの、背景だけ・人物だけ・背景入れ替え動画の生成コードやデータ（Kinetics400のマスク等）を借りられるか。
- Temporal-32のクラス一覧とデータが研究室にあるか。

---

## 追記（2026-11-12）：小林さんの論文（arXiv:2509.23009）を原文で読んで反映

- 対象：Masato Kobayashi, Ning Ding, Toru Tamaki, "Disentangling Static and Dynamic Information for Reducing Static Bias in Action Recognition", arXiv:2509.23009v1（2025-09-27）。arXivのComments：ICCV 2025 Workshop and Challenge on Disentangled Representation Learning for Controllable Generation (DRL4Real)。
- 読んだもの：arXiv版PDFから抽出した本文（v1、全文）。
- **訂正**：直前の追記は、2026-10-26のメモにある卒論の要約に基づいていた。この論文とは別の文書で、数値（1フレーム 54.80%、16フレーム 70.93% など）はこの論文の表にはない。以下はこの論文の原文に基づく。卒論とこの論文の関係（同じ研究の別版か）は確認していない。

### 論文の中身（評価に関係する部分）

- **2つのstream**
  - unbiased stream：TimeSformer（Kinetics400で事前学習）。行動を予測する特徴 f_u。
  - biased stream：時間情報を使えず、静的な手がかりだけに頼る特徴 f_b。2つの版がある。
    - extractor-based：フレームごとの画像特徴の時間平均（late-fusion ViT）。フレームの順序が結果に影響しない。
    - input-based：unbiasedと同じエンコーダに、フレームをシャッフルした動画と、1フレームを複製した動画を交互に入れる。
- **分離の損失**：f_u と f_b の HSIC（統計的な独立性の指標。小さいほど独立）を、min-maxで最適化する。
- **シーン予測の敵対損失**：Places365で事前学習して凍結したViTに、ランダムな1フレームを入れてシーンのsoft labelを作る（シーンのアノテーションが不要）。unbiased側はGRLでシーンを当てられないように、biased側は当てられるように学習する。
- **評価**
  - データ：Temporal-32（Only Time Can TellのTemporal-50のうちKinetics400の32クラス）、UCF101、UCF-Temporal13とUCF-Static13（著者が作った、時間が重要なクラスと静的に解けるクラスの部分集合。付録にクラス一覧）。
  - 指標：top-1、2つのstream間のHSIC、HAT（Chung et al., NeurIPS 2022）の4指標。
    - BOR：人物を消した背景だけの動画の精度を、元の動画の精度で割った値（↓）
    - HOR：背景を単色で塗った人物だけの動画の同様の比（↑）
    - SHAcc：別クラスの動画の背景に入れ替えた動画の精度（↑）
    - SBErr：入れ替え動画で間違えたとき、背景側のクラスを答えた割合（↓）
- **結果（Temporal-32、表1）**：TimeSformerだけのベースラインは top-1 80.9、BOR 77.0、HOR 23.4、SHAcc 9.61、SBErr 36.3。input-based＋シーン予測（t0=0）で BOR 71.0、SBErr 27.5 に下がった（top-1 は 69.5）。
- **論文自身の注意**
  - 「HSICが小さいことは、静的バイアスが小さいことを直接は意味しない」。HSICが最も小さかった設定（extractor-based、シーン予測なし）は、BORが最も大きかった。
  - HORは、単色背景の人物だけの動画にシーンのheadが合わない（ドメインギャップ）ため、バイアスの正しい評価に向かない可能性がある。
  - シーン予測損失の効果はデータセットと設定に大きく依存する。

### この評価選定に効く点

1. **「静的な参照特徴」の作り方がそのまま使える**
   - extractor-based：学習前の画像ViTのフレームごとCLSの平均。順序に対して構造的に不変。
   - input-based：自分のモデルにシャッフルしたclip、1枚×4のclipを入れた特徴。今の評価器の shuffle と static の条件と同じ。
   - これらを f_static として、自分のLoRAの特徴（またはLoRAの差分 r）がどれだけ f_static で説明できるかを測る。採用候補3・4の「静的な対照」を、この論文の定義で書ける。
2. **HSICを分離の指標として足せる。ただし主指標にしない**
   - HSIC(f_LoRA, f_static) や HSIC(r, f_static) を、LoRAあり／なし・層ごとに出す。
   - 論文自身が「HSICが小さくても静的バイアスは小さいとは限らない」と示しているので、HSICは補助の指標にし、機能の指標（採用候補1の時間probe、下のBOR・SBErr）と必ず並べる。「特徴が離れた≠動的情報を得た」という教訓を、研究室の論文で裏付けられる。
3. **シーンのsoft labelで静的probeを作れる（ラベル問題の解決）**
   - 採用候補3の静的probeは、動画IDか行動クラスを想定していた。Places365で学習したViTをランダムな1フレームに当てたsoft labelなら、ActivityNetのどのclipにもアノテーションなしでシーンのラベルを付けられる。
   - 自分の特徴・LoRAの差分・フレーム間の差分から、このシーン分布をどれだけ予測できるか（KLまたはtop-1一致）を、静的probeとして使う。動的な成分にシーンが漏れていないか（Leakage Gap）を測れる。
4. **HATの4指標で選択性（段3）を測れる**
   - 自分のモデルは自己教師ありで行動分類器を持たないので、凍結特徴の上に行動の線形probeを学習してから、HATの動画（背景だけ・人物だけ・背景入れ替え）で BOR・SHAcc・SBErr を測る。比較は学習前／LoRAあり。
   - LoRAで BOR と SBErr が下がり、SHAcc が上がれば、背景への依存が減った証拠になる。HORは論文の注意どおり参考値にする。
   - HATはKinetics400とUCF101向けで、研究室で使われている。ActivityNetには直接使えないので、評価はUCF101やTemporal-32で行うことになる。
5. **時間が重要なクラスと静的なクラスを分けて比べられる**
   - UCF-Temporal13 と UCF-Static13 で行動の線形probeを学習し、LoRAによる改善が Temporal13 側で大きいかを見る。動的情報を学んだなら、改善は Temporal13 に偏るはず。
   - 注意：付録の一覧では BasketballDunk と FrontCrawl が両方のリストに載っている。使う前に著者側の定義を確認する。
6. **評価ではなく学習の次の一手として**
   - この論文の構造（静的なstreamとunbiased streamをHSICで独立にする）は、本研究の「凍結した画像ViT＝静的、LoRA＝動的」とよく対応する。将来、LoRAの出力を学習前のフレーム平均特徴から独立にする損失を加える、という改善案になり得る。今回の評価の範囲外。

### 採用候補への反映

| 採用候補 | 変更 |
| --- | --- |
| 3. 静的probe＋差分特徴 | 静的ラベルに「Places365 ViTのシーンsoft label」を追加。静的な参照特徴を extractor-based／input-based の2通りで定義。HSICを補助指標として追加 |
| 保留だった背景差し替え | HATの BOR・SHAcc・SBErr（UCF101／Temporal-32、行動の線形probe経由）として採用候補に上げる。HORは参考値 |
| 外部評価（保留） | UCF-Temporal13／UCF-Static13 と Temporal-32 を、SSv2より先に使う候補にする |

### 確認したいこと

- 研究室にあるHATの動画（Kinetics400／UCF101）と、Temporal-32・UCF-Temporal13／Static13 のデータを使えるか。
- 両方のリストに載っている2クラスの扱い。
- Places365で事前学習したViTのcheckpointの入手先。
