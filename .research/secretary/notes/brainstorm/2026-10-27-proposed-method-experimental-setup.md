---
date: 2026-10-27
project: sequential-video-lora-analysis
source_todo: null
topic: proposed-method-experimental-setup
status: exploratory
tags: [brainstorm, research, experimental-design, early-fusion, lora, online-learning, static-dynamic-information]
---

# 提案手法が実現できたときの実験設定（想定）

## 出発点

- ユーザーの依頼（2026-10-27）：現在の実装ではなく、提案手法を実現できたときの実験設定（想定）を考えてほしい。
- このメモは想定の実験設計である。spec、実装許可、実験結果のいずれでもない。数値（フレーム数、rank、データ量、計算時間）はすべて仮置きである。

## 読み込んだ文脈

- `.research/lab/projects/sequential-video-lora-analysis/README.md`（last_updated 2026-10-22）
  - 現在の実装：ViT-B/16、Q/V LoRA、MoCo v2型、ActivityNet 16フレーム、late fusion（masked mean）。ViT＋MoCoのQuery collapseを診断中である。
  - 計算資源の実績：RTX A4500×2（各20GB級）、DDP。
- `meetings/2026-09-17-mtg.md`：時間順入力とshuffle入力の比較、同一フレームの繰り返しの対照。
- `meetings/2026-09-24-mtg.md`：early fusion（VideoMAE型）を候補にする。MoCoの過去Keyをオンライン条件でnegativeにする妥当性。
- [2026-10-27 解析方針（early fusion前提）](2026-10-27-early-fusion-lora-dynamic-information-analysis-plan.md)
  - 比較条件C0〜C3、解析A〜F、中心となる問いQ1〜Q3。
- [2026-10-27 オンラインとオフライン](2026-10-27-online-vs-offline-dynamic-information-lora.md)
  - H1（構造）とH2（学習の順序）への再定式化。
  - MoCo v2型の正例は静的な手がかりで解けるという指摘。
- [2026-10-26 仮説の正しさと新規性](2026-10-26-image-static-lora-dynamic-hypothesis-validity-novelty.md)
  - 動的情報の定義D1。分離の定義(a)。構成M1、E1、M2。

## 前提（このメモでの仮定）

- 提案手法はM1とする。
  - 凍結した画像ViTのattentionを、全フレームのtokenにまとめて適用する（joint space-time attention、tubelet＝1）。
  - 学習するのはQ/VのLoRA（ΔW）だけである。
- 動的情報＝D1（複数フレームを見て初めて分かる情報）。分離の定義＝(a)（W₀＝静的、ΔW＝動的な差分）。
- 「実現できた」とは、M1＋LoRA＋自己教師あり学習が崩壊せずに学習でき、Baseより表現が悪化しないことを指す。

## 検証する仮説

| 記号 | 仮説 | 主な比較 |
|---|---|---|
| H1（構造） | 時間順のフレームを、フレームをまたぐ経路で学習したΔWは、D1の動的情報を持つ | C3とC1・C2 |
| H2（学習の順序） | 時間順のストリームで学習したΔW（オンライン）と、clipをshuffleして学習したΔW（オフライン）では、獲得する情報が違う | C3とC4 |
| H3（目的関数） | ΔWに動的情報が入るかは、自己教師ありの目的関数が静的な手がかりで解けるかに依存する | C3とC5 |

- 主張の中心はH1である。H2は研究の設定（オンライン学習）に関する問いで、どちらの結果になっても報告できる。H3は「VideoMoCoの仕組みのおかげではないか」という懸念への回答になる。

## 1. モデル設定（想定）

| 項目 | 想定 | 理由・注意 |
|---|---|---|
| Base | ViT-B/16（現在と同じ画像事前学習済みの重み）、凍結 | 現在の実装と連続させる。CLIP・DINOなど別のBaseは汎化確認として後段で行う |
| 入力 | T＝8フレーム、stride 4（30fpsで約1秒）、224px | joint attentionはtoken数T×196に対して二乗で重くなるため。T＝16は計算量の目安を見て後段で行う |
| フレームの順序 | パラメータを持たない時間位置埋め込み（sinusoidal）を各フレームのtokenに加える | **必須**。時間位置がないとjoint attentionはフレームの並べ替えに対して不変になり、late fusionと同じく順序を持てない |
| CLS | 動画全体で1つのCLS（全フレームのtokenを参照する） | フレームごとのCLSを平均するとlate fusionに戻る |
| LoRA | 全12 blockのQ/V、rank 8、α 16 | 現在の実装（24 module、294,912 parameters＝rank 8相当）とそろえる。挿入位置とrankは後段のablationで変える |
| 学習するもの | ΔWと、自己教師あり学習用のProjectorだけ | 評価時はProjectorを外す |

## 2. 自己教師あり学習の目的関数（想定）

- **O1：MoCo v2型（現在と同じ）**
  - 同じclipに2通りのaugmentationをかけたものを正例にする。
  - 静的な手がかりだけで解ける。そのためH3の対照として使う。
- **O2：時間を使わないと解けないMoCo（提案側の主条件）**
  - 正例：同じ動画の、時刻の違う2つのclip。ρMoCo（Feichtenhofer+ CVPR2021）の考え方である。
  - 追加のnegative：同じclipを逆再生したもの、shuffleしたもの。見た目はまったく同じなので、時間の向きや順序を見ないと区別できない。逆再生を手がかりに使う考え方は、Arrow of Time（Wei+ CVPR2018）に近い。
  - 注意：時間的に対称な動作（手を振る、往復運動など）では、逆再生は本当は同じものなのにnegativeとして扱われる（false negative）。動きの大きいclipだけに適用する、重みを下げる、などの対策が必要になりうる。
- **O3：VideoMAE型の再構成（後段の候補）**
  - tube maskingをした再構成を目的にする。小さなdecoderの学習が必要になるので後段に回す。

## 3. 学習データと学習プロトコル

### 学習データ

- 主：ActivityNet v1.3 training（約1万本のuntrimmed動画、loaderは整備済み）。
  - 計算量を抑えるために、動画の部分集合（例：2,000本）から始めることを想定する。
- 後段の候補：Walking Tours（DoRA）のように、1本が長い連続動画。オンライン設定をより厳密にできる。

### 学習の順序（H2の軸）

| プロトコル | clipの与え方 | 備考 |
|---|---|---|
| オンライン（時間順） | B本の動画を並行に読み、各動画の中では時刻順に連続したclipを与える。1パスで、replayもshuffleもしない | 現在の4-stream round-robinを一般化したもの |
| オフライン（i.i.d.） | オンラインと**同じclipの集合**をすべてshuffleして与える。更新回数もオンラインと同じにする | 違いを順序だけに絞る。Han+ CVPR2025などでも、更新回数をそろえた比較が使われている |
| オフライン（上限の参考） | 複数epoch学習する | 「よく学習した場合」の参考値。主比較には使わない |

- clipの中のフレームは、どのプロトコルでも時間順に並べる（C2を除く）。

## 4. 比較条件

評価時の構造はすべてM1にそろえる。違うのはΔWの学習のさせ方だけである。

| ID | 学習時の構造 | clip内のフレーム | clipの与え方 | 目的関数 | 役割 |
|---|---|---|---|---|---|
| C0 | ― | ― | ― | ―（ΔW＝0） | 基準。凍結したBaseもフレーム間attentionを計算するので、時間感度は0とは限らない |
| C1 | フレーム単位（late fusion） | 時間順 | オンライン | O2 | 静的な動画ドメイン差分の対照 |
| C2 | M1 | 学習時にshuffle | オンライン | O2 | フレームの共起は学べるが順序は学べない対照 |
| **C3** | **M1** | **時間順** | **オンライン** | **O2** | **提案** |
| C4 | M1 | 時間順 | オフライン（i.i.d.） | O2 | H2：学習の順序の効果 |
| C5 | M1 | 時間順 | オンライン | O1 | H3：目的関数の効果 |

- 参考として置ける条件（必須ではない）
  - M1のFull fine-tuning：容量の上限の参考。
  - AIM型の構成（凍結したattentionを時間方向に使い回し、adapterを足す）：既存の設計との比較。
- seedは各条件3つを想定する。

## 5. 評価

### 評価データ

| 目的 | データ（候補） | 注意 |
|---|---|---|
| 時間への感度（解析A） | ActivityNet validation、SSv2 validationの部分集合 | 逆再生、shuffle、1フレームの繰り返しを作って入力する |
| 時間の向きの判定 | SSv2またはUCF101の部分集合で、順再生と逆再生を線形分類する | ΔWの有無で正解率を比べる |
| 時間の順序が重要な行動認識 | SSv2（pick upとput downなど、向きだけが違うクラス）、Temporal-32（小林卒論）、Temporal-50 | Temporal-50はSevilla-Lara+ WACV2021の可能性が高いが、書誌は未確認 |
| 静的バイアス（解析B） | HAT（背景の入れ替え、人だけ残す）、Kowal+ CVPR2022型の対照ペア | BOR/HOR、SHAcc/SBErr、静的・動的の割合の推定 |
| 下流タスク（解析F） | 線形分類：SSv2（動きが中心）とActivityNet・Kinetics（場面が中心） | 「SSv2で大きく上がり、場面中心のデータではあまり変わらない」なら動的寄りと解釈できる |

### 指標（2026-10-27解析方針メモの記号）

- 時間感度の増分：ΔS = S(Cx) − S(C0)。S = 1 − cos(f(x), f(介入後のx))。
- ΔWの寄与：r(x) = f_{Base+ΔW}(x) − f_{Base}(x)。r(x)に含まれる動的情報の割合を、Kowal型の推定で求める。
- フレーム間attention比（解析D）：各tokenのattentionのうち、他のフレームのtokenに向かう割合。動くpatchと静止patchで分けて集計する。
- ΔWを外す・強さαを変える介入（解析C-a）：上の指標がどう変わるかを見る。

### 仮説ごとの予測（判定基準の案）

| 比較 | 仮説どおりの場合 | 仮説が誤りの場合 |
|---|---|---|
| H1：C3とC1・C2 | 逆再生への感度の増分、時間の向きの判定の正解率、SSv2の正解率で、C3 > C2 ≈ C1 | C3 ≈ C1。ΔWには静的な動画ドメインの差分しか入っていない |
| H1：仕組み（解析D） | C3のフレーム間attention比がC0より上がり、特に動くpatchで上がる | C3のフレーム間attention比が下がる。ΔWがフレームごとの処理に戻っている（反例） |
| H2：C3とC4 | どちらかの方向に差が出る（向きは事前に決めない） | 差がない。動的情報の獲得に学習の順序は効かない（これも結論として報告できる） |
| H3：C5とC3 | C5 ≈ C1、C3 > C5。静的に解ける目的関数ではΔWに動的情報が入らない | C5 ≈ C3。目的関数によらず構造だけで動的情報が入る |

- 差の判断には、3 seedの平均とばらつき、評価サンプルに対するbootstrapの信頼区間を使うことを想定する。

## 6. 計算量の目安（推論、粗い見積もり）

- T＝8のM1はtoken数が1,568になる。線形層はフレーム単位の約8倍、attentionは二乗で増える。
  - ViT-Bのforwardは1 clipあたり200 GFLOPs前後と見積もった。
  - MoCoではQueryのforwardとbackward、Keyのforwardが必要なので、1 clipあたり約1 TFLOP程度になる。
- 凍結しているBaseでも、LoRAへの勾配を通すために全層の活性値を保持する必要がある。bf16、flash attention、gradient checkpointingを使う前提で、GPU 1枚あたりのbatchは8〜16程度と想定する。
- ActivityNet全体を重ならないclipで1パスすると、約100万clipになる。A4500×2では1パスに丸1日前後かかる可能性がある。
- 条件6つ × seed 3つ＝18 runになるので、全量では重い。**動画の部分集合（例：2,000本、1 runあたり数時間）から始める**ことを推奨する。
- 実測していない粗い見積もりである。最初に短いrunでスループットを測ってから決める。

## 7. 実施の順序（案）

1. **P0 評価系の確認（学習なし）**
   - C0だけで解析A・B・Dを回す。
   - 評価の仕組みが動くこと、凍結したBaseがどの程度時間に反応するか、T×196 tokenでBaseの表現がどれだけ崩れるかを先に知る。
2. **P1 H1（オフラインで先に行う）**
   - C1・C2・C3をオフライン（i.i.d.）で学習して比べる。H1にはオンラインは必要ないので、扱いやすいオフラインで先に確かめる。
3. **P2 H2**
   - C3（オンライン）とC4（オフライン）を、同じclipの集合・同じ更新回数で比べる。
4. **P3 H3とablation**
   - C5（O1）。LoRAの挿入層、rank、T、E1（tubelet埋め込みLoRA）。
5. **P4 汎化**
   - 別のBase（CLIP、DINO）、別のストリームデータ（長い連続動画）。

## 8. 反例・リスク

- **凍結したattentionの分布のずれ**
  - 画像の196 tokenで学習したattentionを1,568 tokenに適用すると、attentionが薄まり、Baseの表現が崩れる可能性がある。
  - P0で確かめる。崩れが大きい場合は、AIM型（同じ空間位置どうしの時間方向attention）を代わりの構成として検討する。
- **ΔWがフレーム間のやり取りを止める**
  - 分布のずれを直すために、ΔWが「他のフレームを見ない」方向に学習する可能性がある。解析Dで検出する。
- **O2のfalse negative**
  - 時間的に対称な動作では、逆再生を本当は同じものとして扱うべきである。
- **データの偏り**
  - ActivityNetは場面の手がかりが強い。学習データが場面中心でも、評価には動きが中心のデータ（SSv2、Temporal系）を必ず含める。
- **前提が未成立**
  - 現在はViT＋MoCoのcollapse診断中である。このメモの設定は、学習が成立した後を想定している。

## 収束

| 区分 | 内容 |
|---|---|
| 中心 | M1＋Q/V LoRA、目的関数O2、比較条件C0〜C3（H1）、評価A・D・時間の向きの判定・SSv2 |
| 次点 | C4（H2）、C5（H3） |
| 後段 | O3（VideoMAE型）、E1、別のBase、長い連続動画、VLM・生成への応用 |
| 保留 | M2（MeMViT型memory）。オンライン推論が本当に必要になった場合に検討する |

## 未解決の問い（ユーザーが決めること）

1. 主の目的関数をO2（時間を使わないと解けないMoCo）にするか。現在のO1のまま進めて、H3の結果として見せるか。
2. 中間発表で示す比較条件を、C0〜C3（H1だけ）にするか、C4・C5まで含めるか。
3. T（フレーム数）とstrideをどう決めるか。P0の計算量とBaseの崩れを見て決めるか。
4. 評価データとして、SSv2やHATを実際に用意できるか（研究サーバ上のデータの有無はGitHubからは確認できない）。

## 次アクション候補（未採用、TODOには追加していない）

- 中間発表の「今後の設計」スライドに、比較条件の表（C0〜C3、必要ならC4・C5）と実施の順序（P0〜P3）を入れる。
- O2の正例・負例の作り方の先行研究（ρMoCo、Arrow of Time、時間の対照学習の系統）を本文で確認する。
- P0（C0だけの評価系）をspec化するかの判断。spec化はユーザーが明示した場合に限る。

## 関連

- [2026-10-27 解析方針（early fusion前提）](2026-10-27-early-fusion-lora-dynamic-information-analysis-plan.md)
- [2026-10-27 オンラインとオフライン](2026-10-27-online-vs-offline-dynamic-information-lora.md)
- [2026-10-26 仮説の正しさと新規性](2026-10-26-image-static-lora-dynamic-hypothesis-validity-novelty.md)
- [2026-10-27 小林卒論の背景としての重要点](2026-10-27-kobayashi-thesis-background-relevance.md)
- `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-17-mtg.md`、`2026-09-24-mtg.md`

---

## 追記 2026-10-27 19:55 JST：本実験に向けた比較設計の補正（exploratory）

### 比較解釈の注意

- **主要な主張**：時間情報が学習済みLoRAにあると言うためには、同じ時空間Attention構造M1に対し、学習前Base（LoRA無し）と学習後Base＋LoRAを比較し、Projectorを除いた表現で動き依存の評価を行う。単なるMoCo Loss低下や特異値の変化だけでは証明できない。
- **C0の解釈**：凍結した画像ViTでも、joint attentionと時間位置情報を加えたM1は時間に反応する可能性がある。「Baseに静的情報しかない」と断言せず、LoRAによる時間依存性・タスク性能の増分を測る。
- **旧C1（Late Fusion）**：フレームごとに特徴抽出して平均する構造はフレーム順入替に不変。逆再生をhard negativeとするO2はこの構造では区別不能になり得る。C1は順序不変性の補助対照とし、比較可能な通常MoCo O1で学習する別条件を設ける。Late Fusionで学習したLoRAをM1に転写して評価するなら、アーキテクチャ転移という別の実験になる。
- **旧C2（学習中のclip内shuffle＋O2）**：逆再生／shuffleをnegativeとするO2と、入力側で順序を破壊する条件を同時適用すると、教師信号の整合性を損ねるおそれがある。初期の主比較から外し、まず評価時だけoriginal/reverse/shuffle/static-repeatを与える介入として実施する。旧C2の学習群は保留。
- **オンライン対オフラインの交絡**：MoCoのQueueとEMAは投入順に依存する。C3とC4で同じclip集合・更新回数を揃えてもQueue構成は変わるため、「サンプル投入順＋Queue相互作用」を含む学習プロトコル差として報告する。Queue中の同一動画Key比・Key年齢・Hard Negativeを監査する。

### 最小の主要比較

| 条件 | 構造 | 目的関数 | サンプル投入 | 主な役割 |
| --- | --- | --- | --- | --- |
| C0 | M1／LoRAなし | 学習なし | ― | 学習前のBase特性 |
| C3 | M1／Q,V LoRA | 時間区別を要求するO2 | 時間順・single-pass | 提案条件、H1 |
| C4 | M1／Q,V LoRA | C3と同じO2 | 同一clip集合をshuffle・single-pass | H2 |
| C5 | M1／Q,V LoRA | 通常MoCo O1 | 時間順・single-pass | H3 |
| C1補助 | frame-wise＋masked mean／LoRA | 整合的なO1 | 時間順・single-pass | 順序不変の対照 |

### O2の具体化で先に決めること

- Positiveは、現在までに観測した**同じclipの2 view**を基本とする。異時刻の同一動画clipは動作が異なる可能性があり、無条件にpositiveにはしない。
- 時間方向の追加学習信号には、同じフレーム集合を逆順にしたものをhard negativeにする案を置く。reverseは対称的な動作ではfalse negativeとなり得るため、適用対象、重み、mask、augmentationの整合性を事前に固定する。通常MoCoとO2を別目的として比較する。
- cross-clip positiveやQueueには未来のclipを使わない。1clipは観測したフレームを全部読み終わった段階で処理する。
- 同じclipの2 viewでも元の見た目だけで解ける可能性は残るため、時間方向の追加損失が必要かを実測する。

### 初期数値の候補（未確定）

| 項目 | Pilot候補 | 事前確認 |
| --- | --- | --- |
| Base | ImageNet pretrained ViT-B/16、凍結、M1 joint attention | フレーム数増加による表現の崩れ |
| 動画入力 | 224px、T=8、約1秒窓、固定時間位置情報 | 動画fpsの違いを考慮した時刻基準抽出、動き量 |
| LoRA | 全12層Attention Q/V、rank8、alpha8、dropout0 | 既存単体検証と揃える案。alpha16は感度実験 |
| Optimizer | AdamW、LoRA/Projector別LRの候補 3e-5／1e-4／3e-4 | 既知のLR0.03 collapse疑いを固定probeで再評価 |
| MoCo | momentum0.999、temperature0.2、Queue4096（仮） | Queue一巡、同一動画negative、学習安定性 |
| データ | ActivityNet training 2,000動画pilot→全10,024動画を候補 | 動画IDと時刻のmanifest固定、validationの漏洩防止 |
| 学習 | 1pass、0/1/2/10/100step固定probe、再開可 | EMA/Queue/LoRA更新、CLSとProjector別のcollapse |
| 計算 | RTX A4500×2、mixed precision等は候補 | global/local batchはT=8の実測VRAMとthroughputで確定 |
| 再現性 | pilot 1seed、確認的実験3seed以上 | 比較群で同じclip数・update数・評価器・探索予算 |

- 上記LR、alpha、K、batch、入力フレーム、実行期間はすべて**未承認の仮置き**。GPUでの資源実測・学習成立・別途spec化が必要であり、既存のLR0.03本体条件は変更しない。
- H1の主な評価は、Projectorを外したencoder特徴によるSomething-Something V2等の動き依存クラスのLinear Probeと、original/reverse/shuffle/static-repeatの感度。単なるcosine差に加え、動作の向きの対照ペアで性能を確認する。ActivityNetではaction分類のmAPも補助測定する。
- LoRAの内部は、個別のA/Bではなく実効更新重み ΔW=(alpha/r)BA の層別SVD、effective rank、Q/V寄与、層ごと除去・LoRA倍率操作で解析する。ただし特異値単独では動的情報の根拠にならない。
- 順序は P0 M1 Baseの評価・VRAM → P1 O1で学習安定化 → P2 O2でC3学習 → P3 C3対C4/C5 → P4補助構造対照・full data・multi-seed とする。

この追記はbrainstorm段階であり、spec・コード改変・実験実行・GPU利用の承認ではない。
