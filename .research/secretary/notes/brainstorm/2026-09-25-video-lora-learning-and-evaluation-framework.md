---
date: 2026-09-25
project: sequential-video-lora-analysis
source_todo: null
topic: MoCo・VideoMAE・LoRA解析を一つの研究設計として整理
status: exploratory
tags: [brainstorm, research, lora, moco, videomae, linear-probe, svd, streaming-video]
---

# MoCo・VideoMAE・LoRA解析を一つの研究設計として整理

## 出発点

2026-09-25の相談メモでは、研究の中心を「動画に適した良いLoRAを獲得すること」と置き直し、
次を同時に検討している。

- MoCoを用いた動画LoRA学習。
- ImageNet事前学習ViTをVideoMAE-style encoderへ移植する案。
- sequential / online入力の影響の切り分け。
- 学習済みLoRAの下流性能と内部構造の評価。
- 特異値、layer、Q/K/V、動画の動きとLoRA更新量の関係の解析。
- linear probeによる最初の下流評価。
- 学習・解析に必要なログの整備。

## 現在の文脈

### 確認済み

- 現行MoCo baselineは、画像事前学習済みViT + Q/V LoRA + frame-wise feature + masked mean + MoCo v2-style objective。
- ActivityNetで4-stream round-robinの実動画100-stepまで学習経路が成立している。
- 現行masked meanはclip内frame順に不変なので、時系列順で入力しただけではframe orderを学習したとは言えない。
- ImageNet事前学習ViTをVideoMAE-style encoderへ移植すること自体は可能だが、patch embedding、position embedding、CLS、decoder、state_dict変換等の設計が必要。

### 重要な解釈

研究上は「良いLoRA」を少なくとも次の4軸に分ける必要がある。

1. 下流タスクで有用な表現を作る。
2. 静止画だけではなく動画の時間変化・動きに反応する。
3. LoRA rank / layer / QKVのcapacityをどのように使っているか説明できる。
4. sequentialな非IID入力でも学習が破綻せず、shuffleとの差を解釈できる。

この4軸を同じ指標だけで評価することは難しい。

## MoCoとVideoMAEの役割分担

### MoCo

MoCoは現在の実装資産をほぼ維持したまま、

- LoRAが動画データから有用な表現へ更新できるか。
- self-supervised pretraining後にlinear probe等で下流性能が出るか。
- chronological / shuffled update順の違いが結果へ影響するか。

を確認するbaselineとして使いやすい。

ただし現在のsame-clip two-view + masked meanでは、temporal order自体をloss成立条件にしていない。
そのため「動画で学習したLoRA」と「時間順序を理解したLoRA」は区別する。

### VideoMAE-style

VideoMAE-styleでは複数frameをspatiotemporal tokenとして同時にencoderへ入力するため、
LoRAがcross-frame attentionへ直接関与できる。

ImageNet ViTからの初期化では、

- Transformer blockはdimensionが一致すれば移植候補。
- 2D patch embeddingは3D tubelet embeddingへinflateする。
- tubelet size 1ならtemporal dimensionをunsqueezeするだけで最小変換にできる。
- tubelet size 2以上では2D weightを時間方向へ複製し、平均相当になるようscaleする方式が代表候補。
- spatial position embeddingは時間方向へ複製する、またはspatiotemporal encodingを別途導入する。
- decoderは画像ViTには存在しないため新規学習parameterとなる。

この方式はoriginal VideoMAEそのものではなく、
ImageNet-pretrained ViT initialized VideoMAE-style masked video modelingとして扱う。

## 「フレームごとのMAEはいまいちか」の意味

frame-wise MAEでは各frameのreconstruction lossが単独で成立する。

そのため、

- frame tの復元にframe t+1を使わなくてもlossを下げられる。
- LoRAが時間変化を学習しなくても目的関数を解ける。
- 動画を入力していても、実質的には画像の集合として処理している可能性がある。

temporal moduleや動画全体のlossがframe encoderまでbackpropagateする構成なら改善できるが、
単純なframe-wise MAEだけでは「動画特徴をLoRAへ入れる」保証が弱い。

## sequentialの影響をどう切り分けるか

chronologicalとshuffleを比べるときは、単に最終lossを比較するだけでは不十分。

MoCoでは順序を変えると、

- optimizer state
- momentum encoder state
- Queueの中身
- negativeの新旧
- 連続sample間のgradient correlation

も変わるためである。

最小比較では、

- 同じclip集合
- 同じ学習step数
- 同じ初期checkpoint
- 同じaugmentation seedまたは再現可能なview生成
- 同じoptimizer hyperparameter
- chronological / shuffledだけを主な差分

とし、Queue compositionやgradient/update normも同時に記録する。

strict one-video-at-a-timeと4-stream round-robinは別factorとして扱う。

Orthogonal Gradientsはこのsequential非IID問題へ追加するoptimizer側の比較候補であり、
MoCoやVideoMAEのobjectiveとは別軸。

## 最初の下流評価: linear probe

最初に「学習できているか」を確認する用途ではlinear probeが重要候補。

手順:

1. self-supervised学習後のencoderを固定する。
2. Base ViTとLoRAを含めてfeature extractorとしてfreezeする。
3. そのfeatureの上に線形分類層だけを追加する。
4. ラベル付きtrain splitで線形層だけ学習する。
5. validation / testで精度を比較する。

比較候補:

- Base ViTのみ
- MoCo学習LoRA
- VideoMAE-style学習LoRA
- chronological MoCo
- shuffled MoCo

self-supervised pretraining中にラベルを使っていなくても、評価時だけラベルを使うlinear probeは一般的なrepresentation評価として成立する。

## LoRAの特異値解析

LoRAで実際にbase weightへ加わる更新を

`Delta W = s * B A`

とし、そのSVDを

`Delta W = U Sigma V^T`

とする。

見る対象:

- singular values `sigma_1 ... sigma_r`
- Frobenius norm
- spectral norm
- singular value energy ratio
- effective rank / stable rank
- layer間のspectrum差
- checkpoint間のspectrum変化

### 注意

「特異値が全部同じくらいなら良い」「半分くらいなら理想」という普遍的な基準はない。

複数のsingular valueが十分な大きさを持つならrank capacityを複数方向で使っていると読める一方、
`sigma_1`が支配的なら実効rankが低い可能性がある。
ただし本当に1方向の更新で十分なtaskなら、rank-1的であること自体は悪くない。

したがって特異値はperformanceと組み合わせて読む。

## 「第一成分が似ている」の扱い

LoRA間で第一特異ベクトルを比較するなら、

- left singular vector
- right singular vector
- first singular subspace

のsimilarityを測る候補がある。

SVDのsingular vectorには符号反転の不定性があるため、
cosine similarityを使う場合はabsolute cosineやsubspace similarityを使う方が安全。

何と何の第一成分を比較するか
（sample間 / checkpoint間 / layer間 / method間）は実験設計時に明示する必要がある。

## 動画の動きとLoRA更新量の相関

仮説:

- 動きの大きいclipではLoRAへのgradient/updateが大きい。
- 静止に近いclipではLoRA更新が小さい。
- temporal objectiveを使うVideoMAE-styleでは、この傾向がMoCoより強く出る可能性がある。

ただしこれは未検証仮説。

各stepで少なくとも、

- motion score
- LoRA raw gradient norm
- optimizer後のparameter update norm
- layer別 / QKV別 update norm
- loss
- sequence_id / clip位置

を対応付けて記録する。

AdamWではoptimizer stateの影響があるので、
「入力の動きに対する直接的反応」を見るならparameter update normだけでなくraw gradient normも保存する。

motion scoreの簡易候補は隣接frame差分。
より意味のあるmotion尺度としてoptical flow magnitude等も候補。

相関はSpearman等で見る候補があるが、相関だけで因果は主張しない。
static-repeat、frame shuffle、reverse等のcontrolと組み合わせる。

## どの層・Q/K/VのLoRAが効くか

LoRA normが大きいlayer = 重要なlayer、とは限らない。

二つに分けて評価する。

### 内部解析

- layer別LoRA norm
- gradient norm
- update norm
- singular spectrum
- motion scoreとの相関

### 機能的importance

学習済みLoRAの評価時に、

- 特定layerのLoRAだけoff
- 前半 / 中盤 / 後半だけoff
- Qだけoff / Kだけoff / Vだけoff

などのablationを行い、linear probe等の性能低下を見る。

これにより「大きく更新された」と「実際に下流性能へ効いている」を区別できる。

実装はtarget modulesをq / k / vから設定可能にしておくと後続比較が容易。

## 学習ログで優先する項目

最低限候補:

- self-supervised loss
- positive / negative similarity（MoCo）
- Queue valid negative数・sequence構成（MoCo）
- learning rate
- LoRA全体 / layer別 / QKV別 gradient norm
- LoRA parameter update norm
- `Delta W` norm
- singular values / effective rank
- sample motion score
- sequence_id / clip_start
- validation linear probe用checkpoint

全部を毎step SVDすると高コストになり得るので、
gradient/update normは毎step、SVDは一定intervalまたはcheckpoint時という設計が候補。

## 現時点の研究設計候補

### Phase A: 表現が有用か

Base ViT vs MoCo-LoRAをlinear probeで比較。

### Phase B: 時間を扱うarchitecture/objectiveの比較

MoCo-LoRA vs ImageNet-initialized VideoMAE-style LoRAを同じ下流評価で比較。

### Phase C: sequential性の比較

chronological vs shuffledを同条件で比較。
strict onlineと4-streamは別factor。

### Phase D: LoRA内部解析

- singular spectrum
- effective rank
- motion-update correlation
- static control
- layer/QKV ablation

により、「性能が出たLoRAが内部でどう違うか」を解析。

## 現在の方向性

- MoCoを捨てずbaselineとして保持する。
- VideoMAE-styleを時間情報を直接扱う比較候補として並行検討する。
- まずlinear probeでrepresentation utilityを測る。
- 特異値解析だけで良否を決めず、下流性能・motion correlation・ablationと組み合わせる。
- sequentialの影響はobjectiveの良否と分離して評価する。

## 保留・未解決

- linear probeの下流dataset / labelを何にするか。
- VideoMAE-styleのtubelet size 1 / 2。
- VideoMAE decoderの容量。
- VideoMAE側のQ/VまたはQ/K/V LoRA target。
- motion scoreの定義。
- 特異値解析で比較する単位。
- static clipの抽出基準。
- strict onlineをどの段階で導入するか。
- SVD/loggingの頻度と保存形式。

## 参考

- VideoMAE: https://arxiv.org/abs/2203.12602
- ViViT: https://openaccess.thecvf.com/content/ICCV2021/html/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.html
- Video Swin Transformer: https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html
- MoCo: https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html
- LoRA: https://arxiv.org/abs/2106.09685
- Learning from Streaming Video with Orthogonal Gradients: https://openaccess.thecvf.com/content/CVPR2025/html/Han_Learning_from_Streaming_Video_with_Orthogonal_Gradients_CVPR_2025_paper.html

このメモは探索記録であり、specまたは実装許可ではない。
