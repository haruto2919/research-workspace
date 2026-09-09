---
date: 2026-09-09
project: sequential-video-lora-analysis
source_todo: null
topic: 動画逐次学習でLoRAに追加される情報の解析と動的・時間的情報への特化可能性
status: exploratory
tags: [brainstorm, research, lora, video, temporal-information, external-chat]
source: chatgpt
---

# 動画逐次学習でLoRAに追加される情報の解析と動的・時間的情報への特化可能性

## Source

- ChatGPTで行ったMeMViT、LoRA、`sequential_loader`に関する研究相談を、2026-09-09に構造化して取り込んだ。
- 対象範囲は、LoRAの基礎、画像由来知識と動画由来知識の区別、MeMViTへのLoRA導入候補、LoRAが保持する情報の評価・分離に関する議論である。
- 外部チャット内のAI説明・提案は、ユーザーが明示的に採用した研究方針や確認済みEvidenceとしては扱わない。
- 会話全文の複製ではなく、問い、候補、判断理由、保留事項を再現できる探索記録として保存する。

## 読み込んだ既存研究文脈

- [`README.md`](../../../lab/projects/sequential-video-lora-analysis/README.md)
  - プロジェクトはactive。
  - 共用シーケンシャルデータローダとLoRA逐次学習基盤を先に整備し、その後にLoRAが保持する動画情報を評価する段階と記録されている。
- [`2026-09-03-mtg.md`](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-09-03-mtg.md)
  - 最初の技術目標として、逐次入力に対してLoRAのみを学習するfine-tuningを動作させることが決定されている。
  - 動画情報の評価を次段階とし、分離・直交化・変換は基本パイプライン成立後に検討することが決定されている。
- [`2026-09-09-sequential-lora-finetuning-spec.md`](../../../lab/projects/sequential-video-lora-analysis/specs/2026-09-09-sequential-lora-finetuning-spec.md)
  - 50Salads、K400 pretrained MeMViT、全16 blockのattention `q` / `v` LoRA、新規51-class headを用いるbaselineがdraftとして具体化されている。
  - `r = 8`、`lora_alpha = 16`、`lora_dropout = 0.0`、`peft == 0.20.0`、AdamWなどがdraft内で固定されている。
  - LoRA成分解析、静的/動的情報分離、VLM・動画生成評価、科学的結論は対象外である。
  - statusは`draft`であり、実装・config変更・test追加・学習runを許可していない。
- 2026-09-09の日次TODOは未作成。
- 対象project配下にexperiment記録とreferenceファイルは見つからなかった。
- 関連する既存brainstormは見つからなかった。

## 相談の出発点

画像で事前学習されたモデルを固定し、動画を逐次入力してLoRAだけを追加学習すれば、画像だけでは得にくい時間順序、動き、状態遷移などを、元の静的知識と分離してLoRAへ保持できるのではないか、という問題意識が出発点である。

同時に、LoRAに関する基礎知識がない状態から、以下を理解・整理することが相談の目的だった。

- LoRAの仕組みと学習対象
- 画像から得られる知識と、動画から追加され得る知識の違い
- 現在のMeMViT実装でLoRAを導入できる箇所
- LoRAに実際に何が学習されたかを検証する方法
- 「LoRAに時間情報だけが入った」という過剰な解釈を避けるための比較条件

## 中心となる問い

1. Base modelを固定して動画でLoRAを学習したとき、LoRAには何が追加されるのか。
2. 時間順序、motion、state transition、duration、temporal relationを、背景、物体、人物、姿勢などの静的手掛かりから操作的に区別できるか。
3. LoRAが動作を学習したのか、背景・物体・人物・dataset biasを利用しただけなのかを、どの実験で切り分けられるか。
4. MeMViTのattention projectionへLoRAを入れた場合、どこまでを時間情報の獲得と解釈できるか。
5. 通常LoRAをbaselineとして先に測り、その後にmotion-focusedな制約やmodule限定を導入する順序が妥当か。
6. 「LoRAが動画特有情報を保持した」と主張するために、task accuracy以外にどのEvidenceが必要か。

## 会話内で整理された基礎概念

以下は外部チャット内のAI説明であり、本メモでは研究上の確定事項ではなく、議論の前提となった概念整理として保持する。

### LoRA

学習済み重みを $W$ とすると、LoRAでは重み差分を低rank行列で表す。

$$
W' = W + \Delta W, \qquad \Delta W = BA
$$

forwardは次のように解釈できる。

$$
y = Wx + BAx
$$

通常はBase weight $W$ をfreezeし、LoRAの $A$ と $B$ を更新する。新規task headがランダム初期化なら、headも学習対象にする必要がある可能性がある。

### 静的情報と動的・時間的情報の暫定整理

| 種別 | 会話内で挙がった例 |
|---|---|
| 画像から得やすい静的情報 | 人物、物体、色、形、質感、背景、場所、姿勢、空間的位置関係、単一frame内の状態 |
| 動画で追加され得る情報 | 時間順序、動く方向、速度、動作の開始・終了、継続時間、状態遷移、frame間関係、周期性、時間的依存関係 |

外部チャットでは、後者を暫定的に `Order + Motion + State Transition + Duration + Temporal Relation` と整理した。ただし、これはAIによる概念化であり、研究上の正式定義ではない。

### MeMViT memoryとLoRAの区別

- MeMViT memoryは、forward時に過去clipの特徴を保持・参照するonline stateである。
- LoRAは、gradientで更新され、学習後に保存される追加parameterである。
- 両者は役割も寿命も異なり、混同しない。

## Workspaceとの照合結果

| 外部チャット時点の内容 | 最新Workspace文脈 | 本メモでの扱い |
|---|---|---|
| 画像事前学習Baseを固定して動画LoRAを学習する仮説 | 現在のdraft specはKinetics-400 pretrained MeMViTをbaselineに採用 | 研究仮説と現行baselineに差がある。K400 baselineだけでは「画像だけではない追加知識」を直接切り分けられないため、要確認事項として残す |
| `sequential_loader` branch、commit、現在configは要確認 | draft specは実装repo `main@e0deb093...` と外部loader `master@cef09aa...` を基準revisionとして記録 | 外部チャットのbranch認識を現在の正本とみなさず、draft specのrevisionを最新記録として参照する |
| target module、head、rank、alpha、dropout、実装方式は未決 | draft specは`q/v`、head学習、`r=8`、`alpha=16`、dropout 0、PEFTを具体化 | 外部チャット上の決定としては扱わない。現行draft baselineの選択として区別して記録する |
| まず通常LoRAを作り、その後に解析・分離する案はAI提案 | READMEと2026-09-03 MTGでは、基盤成立後に評価・分離へ進む順序が決定済み | 会話由来の採用事項ではないが、Workspace上の既定方向と整合する |
| normal / shuffle / static-repeatやprobeを評価候補とした | 現行draft specは科学的評価と成分分離を明示的に対象外としている | 後続の評価研究候補として保持し、現行implementation specへ混入させない |
| LoRA-only時のoptimizer filterやshort smokeを候補とした | draft specはtrainable parameter audit、frozen base不変、短時間testをSuccess Criteriaに含む | 現行draft baselineでは具体化済み。ただしspec未承認のため実行権限はない |

## 仮説・候補

### H1: Baseをfreezeし、LoRAへ動画による追加変化を保持する

- 期待: Base weightsと追加学習分をparameter上で分離でき、full fine-tuningより追加成分を解析しやすい。
- 反例・代替説明: LoRAはloss低減に役立つ静的appearance、背景、物体、人物、dataset biasも学習できる。
- 現在地: プロジェクト全体の研究仮説と整合するが、「LoRA = temporal knowledge」は未確認。

### H2: 時間方向に関係するmoduleへ対象を限定すれば、motion寄りになる

- 候補: temporal attention、inter-frame relation、attentionの`q` / `k` / `v` / `qkv` / `proj`。
- 懸念: 現在のMeMViT attentionは時空間tokenを扱い、spatialとtemporalが完全分離されているとは限らない。
- 解釈上の制約: `qkv`または`q/v`へLoRAを入れただけで「temporal-only LoRA」とは呼べない。
- 現在地: draft implementation baselineは全16 blockの`q/v`を選んでいるが、これはmotion-specificityを保証する選択ではない。

### H3: 静止clipでLoRA差分を抑制すれば、motion-focusedになる

同一frameを繰り返した静止clip $x_{static}$ に対し、LoRA追加出力を

$$
\Delta f(x) = f_{LoRA}(x) - f_{base}(x)
$$

と定義し、候補lossを次のように置く案が出た。

$$
L = L_{task} + \lambda L_{static}, \qquad
L_{static} = \|\Delta f(x_{static})\|^2
$$

- 期待: 時間変化がない入力でLoRAが働きにくい性質を誘導する。
- 懸念: task性能を損なう可能性があり、「静止clipで差分ゼロ」が望ましいという根拠も未確認。
- 現在地: 通常LoRAに静的情報が混入するEvidenceを得た後に再検討する保留案。

### H4: 時間構造だけを変えたcontrolでLoRAの依存情報を測る

候補条件:

- normal: 正しい時間順序の動画
- shuffled: 同じframe集合を並べ替えた動画
- static-repeat: 1 frameを複製した静止動画

期待する役割:

- normalとshuffledの比較で、appearanceを概ね揃えたまま時間順序利用を測る。
- normalとstatic-repeatの比較で、動きの有無に対するLoRA挙動を測る。

主な交絡:

- shuffleは不自然な入力分布を作り、時間順序以外の差も生む。
- static-repeatは実動画分布から大きく外れる可能性がある。
- 差が出ても、どの層・表現・taskで測るかにより解釈が変わる。

### H5: LoRA由来表現をprobeし、static/dynamic情報を別軸で測る

候補probe target:

- dynamic: 動作、時間順序、速度、移動方向、状態遷移
- static: 背景、物体identity、人物identity

期待: task accuracyだけでは分からない情報内容を定量化する。

未決:

- 「LoRA由来表現」をどのactivationまたは差分として定義するか。
- どの層をprobeするか。
- annotationとcontrol baselineをどう用意するか。
- probeの予測可能性を、LoRA固有の情報保持とどこまで解釈するか。

## 実装・実験案の整理

| ID | 案 | 外部チャット時点 | Workspace照合後の位置づけ |
|---|---|---|---|
| A | Baseをfreezeし、通常LoRAで動画を追加学習 | 有力候補 | MTGとREADMEの第一段階に整合 |
| B | 時間方向moduleだけへLoRAを導入 | 候補 | temporal-only moduleを特定できるまで保留 |
| C | static-repeatでLoRA差分を抑制するloss | 候補 | 通常LoRAの静的情報混入を測定した後の改善候補 |
| D | normalとshuffledを比較 | 候補 | 後続の科学的評価候補 |
| E | normalとstatic-repeatを比較 | 候補 | 後続の科学的評価候補 |
| F | LoRA表現へdynamic/static probeを適用 | 候補 | 後続の科学的評価候補 |
| G | 通常LoRA baseline→情報解析→motion-focused化 | AI提案として有力 | 進行順はMTGの既定方向と整合。ただし各評価法は未採用 |
| H | attention projectionを主対象にする | 候補 | draft specは`q/v`を採用 |
| I | `q`と`v`だけへLoRAを導入 | 候補 | draft baselineで具体化済みだがspecは未承認 |
| J | `q/k/v`すべてへLoRAを導入 | 候補 | 将来のtarget ablation候補 |
| K | `qkv + proj + MLP`へ広く導入 | 保留寄り | 解析困難性が増すため、狭いtargetで不足した場合に再検討 |
| L | Baseをfreezeし、LoRAと新規task headを学習 | 候補 | draft baselineで具体化済みだがspecは未承認 |
| M | `requires_grad=True`だけをoptimizerへ渡す | 候補 | draft specのparameter auditとoptimizer契約に反映済み |
| N | 1 batch / short runでgradient、freeze、save/loadを検証 | 候補 | draft specの短時間Success Criteriaと整合 |
| O | 小さいrankで追加変化の自由度を制約 | 候補 | draft baselineは`r=8`。rankと動的情報純度の関係は未検証 |
| P | 画像事前学習backboneをBaseにする | 有力論点 | 現行K400 video-pretrained baselineとの役割分担が必要 |

## 判断に使う比較軸

- 動的情報と静的情報を操作的に切り分けられるか。
- 「時間情報を学習した」という主張が反証可能か。
- appearance、背景、物体、人物、dataset biasによるshortcutを測定できるか。
- Base modelがすでに持つ動画知識と、LoRAが追加した情報を区別できるか。
- LoRAが保持する情報をparameterまたはactivationから解析しやすいか。
- target moduleを広げたときの適応能力と解釈可能性のtrade-off。
- task accuracy以外のEvidenceを得られるか。
- 通常LoRAと改善案を同一条件で比較できるか。
- 実現可能性、計算コスト、失敗時に得られる情報。

## 現在の方向性

### 外部チャット内で明示的に採用された研究設計

- なし。
- ユーザーはLoRAを研究実装へ追加する前提で、LoRAの基礎を理解したい意図を明示した。
- これはtarget module、loss、control、rank、head、motion-focused化の採用を意味しない。

### Workspaceで確認済みの進行順

1. 共用シーケンシャルデータローダと通常LoRA逐次学習基盤を成立させる。
2. 学習済みLoRAが動画由来情報を保持するか評価する。
3. Evidenceに基づいて、静的/動的分離、直交化、変換、motion-focused化を検討する。

### 現行draft implementation specとの関係

- 50Salads + K400 pretrained MeMViT + `q/v` LoRA + 51-class headという具体baselineは、既存draft specに記録済みである。
- そのspecは未承認であり、実装やrunの許可を与えない。
- そのspecは基盤成立のみを扱い、LoRAが時間情報を保持したという科学的結論を成功条件にしていない。
- 本brainstormのnormal / shuffled / static-repeat、probe、static suppressionは、後続の評価・分離検討として分離して扱う。

## 保留・棄却寄りの案

### 保留

- temporal-only moduleへの限定
  - MeMViT内でspatial / temporalを明確に分離できるmoduleが特定できた場合に再検討する。
- `q/k/v`、`qkv + proj + MLP`までtargetを拡張
  - draft baselineの`q/v`だけでは学習能力が不足するEvidenceが得られた場合に再検討する。
- static suppression loss
  - 通常LoRAに静的情報が多く混入するEvidenceと、妥当なstatic controlが得られた場合に再検討する。
- rankとtarget moduleのablation
  - 基盤成立後、科学的評価のmetricと計算予算を定義してから検討する。

### 棄却寄り

- 「Baseをfreezeして動画でLoRAを学習すれば、自動的にLoRAは時間情報になる」という解釈
  - LoRAは背景、物体、人物、姿勢、dataset biasなど、loss低減に有用な静的情報も学習し得る。
- 「MeMViTのattention `qkv`または`q/v`へLoRAを付ければtemporal LoRAになる」という解釈
  - attention projectionは時空間tokenを処理し、module名だけでは情報内容を保証できない。
- action recognition accuracyの向上だけを、時間情報獲得の十分なEvidenceとすること
  - 1 frameの静的cueだけでもactionを推測できる反例がある。

## 反例・失敗条件

- 包丁、玉ねぎ、人物姿勢などの単一frame cueだけで「切る」を推測できるため、action精度だけではmotion利用を示せない。
- 同じ中間姿勢でも「座る」と「立ち上がる」は時間順序で意味が変わるため、方向性・順序を明示的に評価する必要がある。
- normalとshuffleの差は、時間順序利用だけでなく、shuffleによるout-of-distribution入力への弱さでも説明できる。
- static-repeatの差は、motion利用だけでなく、実動画分布からの逸脱でも説明できる。
- Baseが動画事前学習済みなら、LoRAが新たに獲得した動画情報とBaseが既に持つ情報を分離しにくい。
- 新規headを学習する場合、headとLoRAのどちらに情報が保持されたかを分けて評価する必要がある。

## 未解決事項（9件）

1. 「動画で追加された知識」を、時間順序、motion、state transition、duration、temporal relationのどこまでとして操作的に定義するか。
2. 画像事前学習Baseという研究仮説と、現行K400 video-pretrained baselineをどう位置づけ、何を比較するか。
3. LoRAが動画特有情報を保持したかを測る一次評価taskとprimary metricを何にするか。
4. normal / shuffled / static-repeatのどれをcontrolとして採用し、各control固有の分布交絡をどう扱うか。
5. 「LoRA由来表現」をどのlayer、activation、Baseとの差分として定義し、probeするか。
6. 背景、物体、人物などの静的情報混入を、どのlabel・metric・baselineで測るか。
7. LoRAを「時間情報を獲得した」と呼ぶために必要なEvidence水準をどこに置くか。
8. 通常LoRAの測定後に、target module制限やstatic suppression lossなどのmotion-focused手法を導入する判断条件を何にするか。
9. VLMまたは動画生成を使う将来評価の目的、接続方法、対象モデルをどう選ぶか。

## 外部チャット時点では未決だったが、現行draft specで具体化された事項

以下は外部チャットの採用事項ではない。現在のimplementation baseline案として、既存draft specが具体化した内容である。

- dataset: 50Salads fine 51-class frame-level classification
- Base: K400 pretrained MeMViT
- LoRA target: 全16 blockのattention `q` / `v`
- trainable: `q/v` LoRAと新規51-class head
- LoRA: `r=8`、`alpha=16`、dropout 0、PEFT 0.20.0
- optimizer: AdamW、`lr=1e-4`、weight decay 0.01
- Base freezeとtrainable parameter audit
- short testによるcausality、state reset、gradient、weight不変性の確認

これらはspec statusが`draft`の間は実装契約としてactiveではなく、科学的評価設計も確定させない。

## 次アクション候補

TODOへは追加しない。ユーザーが採用または別途依頼した場合の候補として保持する。

1. 現行draft implementation specを基盤仕様として承認するかを、別途判断する。
2. 画像事前学習BaseとK400 video-pretrained Baseの役割を整理し、科学的主張に必要な比較条件を定める。
3. 通常LoRAが保持するstatic / dynamic情報の操作的定義とprimary metricを決める。
4. normal / shuffled / static-repeat controlの交絡と生成方法を比較する。
5. LoRA由来表現の抽出位置とprobe protocolを設計する。
6. 通常LoRAのEvidenceが得られた後に、static suppressionやtarget module制限の必要性を判断する。
7. 評価方針が固まった場合、ユーザーの明示依頼により、現行implementation specとは別のresearch specへ引き継ぐ。

## Research Specへ将来引き継げる事項

### 中心となる問い

- Baseを固定して動画を逐次学習したLoRAには、どのような追加情報が保持されるか。
- dynamic informationとstatic informationを、どの程度分離・定量化できるか。
- Evidenceに基づいてLoRAをmotion-focusedに誘導できるか。

### 有力な進行順

- 通常LoRA baselineを成立させる。
- 通常LoRAに何が保持されるかを測定する。
- 静的情報混入が確認された場合に、motion-focused制約を比較する。

### 比較候補

- Base modelのみ
- 通常LoRA
- normal順序で学習したLoRA
- shuffled入力で学習したLoRA
- static-repeat入力で学習したLoRA
- 将来のmotion-focused制約付きLoRA
- 必要に応じて画像事前学習Baseと動画事前学習Base

### 評価候補

- task accuracy / loss
- 時間順序、motion、速度、方向、状態遷移のprobe
- 背景、物体、人物identityのprobe
- BaseとLoRA modelの差分activation
- control入力に対するLoRA追加出力

### 引き継ぎ前に不足する事項

- dynamic / static knowledgeの操作的定義
- Base pretraining比較の役割
- primary metric
- control条件と交絡対策
- probe対象表現とannotation
- Evidence水準
- motion-focused介入の採用条件

## 関連ファイル

- [`README.md`](../../../lab/projects/sequential-video-lora-analysis/README.md)
- [`2026-09-03-mtg.md`](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-09-03-mtg.md)
- [`2026-09-09-sequential-lora-finetuning-spec.md`](../../../lab/projects/sequential-video-lora-analysis/specs/2026-09-09-sequential-lora-finetuning-spec.md)

