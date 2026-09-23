---
date: 2026-09-24
project: sequential-video-lora-analysis
type: meeting-materials
topic: Stage 6Aの進捗と次の評価方針
status: draft
target_meeting: 2026-09-24
notion_url: null
notion_export: null
tags: [meeting-materials, research, activitynet, moco, lora, evaluation]
---

# 2026-09-24 MTG資料: Stage 6Aの進捗と次の評価方針

## 0. 今日のMTGで先生に相談したいこと

今回いちばん相談したいのは、**「動画を使ってLoRAを100回更新できるところまでは確認できたので、次に何を評価すれば研究として前に進められるか」**という点です。

現時点では、実ActivityNetの動画を使ってMoCo方式の自己教師あり学習を100 step連続で動かすところまで成功しました。しかし、これはまだ「学習プログラムが正しく動いた」ことを確認した段階です。

まだ、

- 学習したことで特徴表現が良くなったのか
- 動画を時系列順に見せることに意味があったのか
- LoRAが時間情報や動きの情報を学習したのか

までは確認できていません。

そのため、今日のMTGでは主に次の3点を相談したいです。

1. **次に何を研究上の評価対象にするか**
   - 学習前後で特徴表現がどう変わったかを先に見るのか
   - 時系列順とshuffleの違いを先に比較するのか
   - もっと直接的に「時間順序を理解したか」を測る方法を先に設計するのか

2. **「時系列順 vs shuffle」のshuffleを何の順番に対して行うか**
   - 1本の動画の中のchunk順をshuffleするのか
   - A/B/C/Dという動画を提示する順番をshuffleするのか
   - 1 chunkの中にある16 frameの順番をshuffleするのか

3. **現在使っている4-stream round-robinを今後も基準にするか**
   - 現在は4本の動画を少しずつ交互に進めています。
   - これは各動画の中の時間順序は守っていますが、「1本の動画を最初から最後まで見てから次の動画へ進む」という厳密なonline学習とは違います。
   - 今後の研究でどちらを基準条件にするか相談したいです。

---

## 1. まず、この研究で何をしたいのか

この研究の大きな目的は、

> **画像だけで事前学習されたViTに動画を見せてLoRAだけを追加学習することで、LoRAに動画特有の「時間変化」や「動き」の情報を持たせられるか調べること**

です。

### 1.1 画像モデルだけでは何が足りないのか

今回使っているベースモデルは画像を処理するViTです。

画像モデルは1枚の画像を見て、

- 人がいる
- ボールがある
- 台所である
- 手に物を持っている

といった**その瞬間の見た目**を表現することは得意です。

しかし動画では、それに加えて、

- 手が上から下へ動いた
- ボールを投げた
- 座っていた人が立ち上がった
- 物を取った後に置いた

という**時間方向の変化**が重要になります。

この「時間方向の変化をLoRAに学ばせられるか」が研究の中心です。

### 1.2 なぜLoRAを使うのか

ViT本体には非常に多くのparameterがあります。

今回はViT本体を全部学習し直すのではなく、ViT本体は固定して、attentionの一部に小さな追加parameterであるLoRAを入れています。

イメージとしては、

```text
画像で学習済みのViT
        ↓
本体の知識はなるべくそのまま固定
        ↓
Q / V に小さなLoRAを追加
        ↓
動画を見たときに必要な追加情報だけLoRAへ学習させる
```

という考え方です。

今回の設定では、

- ViT: `google/vit-base-patch16-224`
- Transformer block: 12層
- 各層のQueryとValueにLoRA
- 合計24箇所
- LoRA rank: 8

です。

ViT本体は固定し、主にLoRAを更新します。

---

## 2. 今回使っているMoCoとは何か

### 2.1 なぜ教師あり分類ではなく自己教師あり学習なのか

今回やりたいのは「この動画は何クラスです」とラベルを覚えさせることではなく、動画そのものから表現を学ばせることです。

そのため、ActivityNetのaction labelやsegmentは学習に使っていません。

代わりにMoCoという**対照学習**の考え方を使っています。

### 2.2 対照学習の基本

対照学習では、簡単に言うと、

> **同じ内容から作った2つの表現は近づけ、別の動画の表現とは区別できるようにする**

という学習をします。

今回なら、同じ16 frameのclipから、

- Query側: 元のclip
- Key側: 左右反転したclip

を作ります。

この2つは元々同じ時間区間なので、**positive pair**として近づけたい組です。

一方、Queueに保存されている別動画の特徴は**negative**として使います。

例えば動画Aのchunkを学習するとき、

```text
Query:
  動画Aの現在chunk

Positive:
  同じ動画A・同じchunkの別view

Negative:
  動画Bの過去chunk
  動画Cの過去chunk
  動画Dの過去chunk
```

のようになります。

### 2.3 Query encoderとKey encoder

MoCoではモデルを2つ持ちます。

```text
Query encoder
  └─ 普通にgradientで更新する

Key encoder
  └─ Query encoderをゆっくり追いかける
```

Query側はoptimizerで更新します。

Key側はbackwardでは更新せず、Query側を少しずつ追いかける**EMA（Exponential Moving Average）**で更新します。

今回のmomentumは

```text
m = 0.999
```

です。

つまりKey encoderは急に変わらず、過去のQuery encoderの状態を強く残しながらゆっくり追従します。

### 2.4 Queueとは何か

MoCoでは、以前見た動画の特徴を保存する箱をQueueと呼びます。

今回のQueue容量は4096です。

例えば最初にA1/B1/C1/D1を保存すると、

```text
Queue
[A1, B1, C1, D1]
```

となります。

その後A2を学習するときは、Aと同じ動画の特徴はnegativeから除外するので、

```text
B1, C1, D1
```

などがnegativeになります。

A2の学習が終わったらA2のKeyをQueueへ追加します。

```text
[A1, B1, C1, D1, A2]
```

というように過去の特徴が少しずつ蓄積されます。

---

## 3. 1本の動画をどうモデルへ入れているか

ActivityNetの1本の動画を、一度に全部モデルへ入れているわけではありません。

動画を連続した16 frameごとのchunkに分けています。

例えば動画Aなら、

```text
動画A
A1 = frame 0〜15
A2 = frame 16〜31
A3 = frame 32〜47
A4 = frame 48〜63
...
```

のようなイメージです。

1 chunkの16 frameをそれぞれViTに入力し、各frameから768次元の特徴を取得します。

```text
16 frames
   ↓
ViT
   ↓
16個のframe feature
[16, 768]
```

その後、現在は16個のframe特徴を平均する**masked mean**を使っています。

```text
[16, 768]
   ↓ 平均
[768]
```

この768次元を「この16 frameのclip全体の特徴」として扱います。

---

## 4. masked meanで非常に重要な点

ここは今後の研究評価を考える上でかなり重要です。

現在のmasked meanは、16 frameの特徴を単純に平均しています。

例えば3 frameだけで考えると、

```text
順方向:
frame1 + frame2 + frame3
-----------------------
           3

逆方向:
frame3 + frame2 + frame1
-----------------------
           3
```

は同じ値になります。

したがって、

> **同じframe集合なら、clip内のframe順を入れ替えてもmasked mean後の特徴は同じ**

です。

つまり現在のモデルは、

```text
[歩き始め] → [歩いている] → [止まる]
```

と

```text
[止まる] → [歩いている] → [歩き始め]
```

を、同じ3枚のframeから作れば、masked meanだけでは区別できません。

ここから重要なのは、

> **現在「時系列順で動画を読み込んで学習している」ことと、「モデルがframeの順番を理解している」ことは別**

という点です。

今回100 step時系列順に更新できたことだけでは、「時間順序を学んだ」とはまだ言えません。

---

## 5. 4-stream round-robinとは何をしているのか

今回のStage 6AではActivityNetから4本の動画を使っています。

それぞれを、

```text
動画A: A1 → A2 → A3 → A4 → ...
動画B: B1 → B2 → B3 → B4 → ...
動画C: C1 → C2 → C3 → C4 → ...
動画D: D1 → D2 → D3 → D4 → ...
```

という独立したstreamとして扱います。

### 5.1 最初のwarm-up

最初は、

```text
A1
B1
C1
D1
```

をKey encoderだけに通し、Queueへ入れます。

この時点では学習はしません。

これは、最初の学習stepから別動画のnegativeを用意するためです。

warm-up後は、

```text
Queue = [A1, B1, C1, D1]
```

となります。

### 5.2 その後の学習順

学習は、

```text
step 0: A2
step 1: B2
step 2: C2
step 3: D2
step 4: A3
step 5: B3
step 6: C3
step 7: D3
...
```

という順番です。

これを**round-robin**と呼んでいます。

### 5.3 何が時系列順なのか

重要なのは、

```text
A1 → A2 → A3
B1 → B2 → B3
C1 → C2 → C3
D1 → D2 → D3
```

という**各動画の中の順番は壊していない**ことです。

一方で、

```text
Aを全部見る
↓
Bを全部見る
↓
Cを全部見る
```

という処理ではありません。

そのため現在の条件は、

> **各動画の時間順序を維持した4動画並列型の逐次学習**

と考えるのが近いです。

厳密な「1動画を最初から最後まで処理してから次へ行くonline学習」とは少し違います。

---

## 6. 前回MTGから何が進んだのか

9月17日のMTGでは、

> **まず1つの自己教師あり学習方式を選び、動画LoRAの学習が実際に動き、観測・比較まで進められる経路を作る**

ことを優先しました。

その後、今回はMoCo方式を選び、段階的に実装しました。

### Stage 3: 動画からclip特徴を作れるようにした

ActivityNetから16 frameを取得し、

```text
ActivityNet
   ↓
16 frame
   ↓
frozen ViT
   ↓
frame features [16,768]
   ↓
masked mean
   ↓
clip feature [768]
```

まで動くことを確認しました。

ここではまだ学習していません。

### Stage 4: LoRAを1回だけ更新できるようにした

ViTのQ/VへLoRAを付けて、実ActivityNetの16 frameを使って1 stepだけ更新しました。

確認できたことは、

- ViT本体は変わらない
- LoRAにはgradientが流れる
- LoRA parameterが実際に更新される

という点です。

### Stage 5: MoCoを1 step動かした

Query encoder、Key encoder、Projector、Queue、InfoNCE loss、EMAを接続しました。

2本の実ActivityNet動画を使い、

- 別動画をnegativeにできる
- Query側LoRAを更新できる
- Key側をEMAで追従できる
- Base ViTは固定されたまま

というMoCoの1 stepを確認しました。

### Stage 6A: 10 step、100 stepへ伸ばした

Stage 5では1 stepだけでした。

Stage 6Aでは4本の動画を使い、

```text
A2 → B2 → C2 → D2 → A3 → ...
```

と連続して学習させました。

そして、

- 実ActivityNet 10-step
- 別processからfreshに開始した実ActivityNet 100-step

の両方が成功しました。

---

## 7. 今回の100-stepで実際に確認できたこと

### 7.1 100回連続で更新できた

最も基本的な結果は、

> **実ActivityNetを使ったMoCo + ViT-LoRAの学習更新を100回連続して実行できた**

ことです。

途中で、

- NaN
- Inf
- Queue破損
- gradient消失
- Base ViTの意図しない更新
- Readerの異常

などで停止せず、100 stepまで到達しました。

### 7.2 Queueが想定どおり増えた

最初にA1/B1/C1/D1を入れるのでQueueは4件から始まります。

1 stepごとにKeyを1件追加します。

したがって、

```text
10 step後
4 + 10 = 14件

100 step後
4 + 100 = 104件
```

になるはずです。

実際に、

- 10-step終了時: 14件
- 100-step終了時: 104件

となりました。

つまり、更新とQueue追加の流れは設計どおり動いています。

### 7.3 毎stepで別動画negativeを使えた

4動画を使っているため、例えばAを学習するときにはB/C/Dの特徴がnegativeとして残ります。

全stepで少なくとも3件の異動画negativeが確保されました。

これはMoCoのlossを毎step計算できたことを意味します。

### 7.4 Base ViTは変わっていない

Query側、Key側のBase ViTはどちらも変更tensor数0でした。

つまり、

> **もともとの画像ViTを固定したまま、追加したLoRAとProjector側だけを動かす**

という方針を守れています。

### 7.5 LoRAとProjectorは変化した

Query側では、

- LoRA
- Projector

にgradientが流れ、parameterが変化しました。

Key側ではbackwardは使わず、EMAによってLoRAとProjectorが変化しました。

したがって、学習更新の経路そのものは成立しています。

---

## 8. ただし、この結果からまだ言えないこと

今回の結果で最も注意する必要があるのは、

> **「100 step動いた」ことと「良い学習ができた」ことは違う**

という点です。

### 8.1 表現が良くなったかは分からない

LoRA parameterが変わったことは確認できました。

しかし、

```text
学習前の特徴
        ↓
本当に動画を表しやすくなった？

学習後の特徴
```

という評価はしていません。

parameterが変わること自体は、性能向上を意味しません。

### 8.2 lossが上下しても今回の成功・失敗とはしていない

100 stepの途中でlossやpositive similarityは変動しました。

ただし今回のStage 6Aでは、

- lossが必ず下がる
- similarityが必ず上がる

ことを成功条件にしていません。

Stage 6Aの目的は、まず**学習機構が壊れず100 step続くか**を見ることだったからです。

### 8.3 時間情報を学んだとはまだ言えない

これは特に重要です。

今回各動画をA1→A2→A3という時系列順に処理しました。

しかしclip特徴自体はmasked meanです。

そのため、

> **時系列順に更新したという事実だけでは、LoRAがframe順序や動きを理解したとは言えない**

という状態です。

---

## 9. 次に評価できる3つの方向

ここからが今日のMTGで一番相談したい部分です。

### 候補A: 学習前後で「表現が変わったか・有用になったか」を調べる

最初の問いは、

> **100 step以上学習したLoRAによって、ViTの特徴表現は学習前からどう変わったか**

です。

例えば同じ評価用clipを、

```text
学習前ViT + 初期LoRA
        ↓
特徴

学習後ViT + 学習済みLoRA
        ↓
特徴
```

で比較します。

見る候補としては、

- cosine similarity
- 特徴の分布
- 同じ動画内clip同士の近さ
- 異なる動画との離れ方
- linear probeによる簡単な評価

などがあります。

この評価の良いところは、

> **今のモデル構成を大きく変えずに、「学習によって何か意味のある表現変化が起きたか」を最初に確認できる**

ことです。

一方で、これだけでは時間順序を学んだかは分かりません。

### 候補B: chronologicalとshuffleを比較する

次の問いは、

> **同じデータを使っても、時系列順で学習した場合とshuffleして学習した場合で結果が変わるか**

です。

例えば、

```text
条件1: chronological
A1 → A2 → A3 → A4

条件2: shuffle
A3 → A1 → A4 → A2
```

のようにします。

ただし「shuffle」という言葉には3種類あります。

#### B-1. 動画内chunk順をshuffle

```text
A1 → A2 → A3 → A4
```

を

```text
A3 → A1 → A4 → A2
```

のように変えます。

これは**同じ動画の時間順序に意味があるか**を見る比較として最も直接的です。

#### B-2. 動画間の提示順だけshuffle

例えば、

```text
A2 → B2 → C2 → D2
```

を

```text
C2 → A2 → D2 → B2
```

のようにします。

各動画内ではA1→A2→A3を保ちます。

これは「複数動画をどう混ぜるか」の影響を見る比較です。

#### B-3. clip内の16 frame順をshuffle

```text
frame 0,1,2,...,15
```

を並べ替えます。

しかし現在はmasked meanなので、同じ16枚を使う限り最終的な平均特徴は基本的に同じです。

そのため、

> **現在のモデルのままclip内frame順だけshuffleしても、時間順序への感度を正しく測る実験にはなりにくい**

と考えています。

### 候補C: 本当に時間順序を使うモデル・評価へ進む

最も研究の本丸に近い問いは、

> **LoRAがframeの順番や動きを本当に表現できるようになったか**

です。

これを調べるなら、現在のmasked meanだけでは不足する可能性があります。

例えば将来的には、

- 順方向clipと逆再生clipを区別できるか
- static-repeatと通常動画を区別できるか
- temporal poolingやtemporal attentionを使う
- 前後関係を予測するobjectiveを入れる
- temporal contrastive learningを導入する

などが候補になります。

ただしこれは現在のStage 6Aより設計変更が大きくなります。

---

## 10. 私が現在考えている進め方

現時点では、いきなり大きくモデルを変えるより、

> **まず「今のMoCo + masked mean + LoRAで何が測れるか」を確認してから、時間モデルを追加する**

流れが分かりやすいと考えています。

具体的には、

```text
Step 1
学習前後で特徴がどう変化したかを見る
        ↓
Step 2
chronological vs chunk shuffleを同条件で比較する
        ↓
Step 3
それでも時間順序を直接評価できないことを確認したら、
temporal aggregation / temporal objectiveを導入する
```

という流れです。

ただし、研究の主結果として最初から「時間情報の獲得」を強く狙うのであれば、Step 1・2を短いbaseline確認にして、早めにStep 3へ進む選択肢もあります。

ここを先生と相談したいです。

---

## 11. 4-stream round-robinをどう考えるか

現在の4-stream round-robinには理由があります。

MoCoではnegativeが必要です。

もし1本の動画だけをずっと処理すると、Queueが同じ動画の特徴ばかりになります。

今回の設計ではsame sequenceをnegativeから除外しているため、1動画だけではnegativeが不足しやすくなります。

そこで、

```text
A
B
C
D
```

の4動画を並行して進め、常に別動画のnegativeを確保しています。

これはMoCoを成立させやすい一方、

> **「1本の動画を連続して観測する厳密なonline学習」とは異なる**

という問題があります。

したがって先生には、

- 当面は4-streamをMoCoの基準条件として使うか
- 最終的には1-video-at-a-time条件も用意するか
- 「online」と呼ぶ範囲をどう定義するか

を確認したいです。

---

## 12. 今回の結果を一言でまとめると

### 今回確認できたこと

> **実ActivityNet動画を使い、frozen ViT + Q/V LoRA + MoCoの自己教師あり学習を、4動画の時系列を保ちながら100 step連続で実行できる学習基盤ができた。**

### 今回まだ確認できていないこと

> **その学習によって表現が良くなったか、時系列順で学習する意味があったか、LoRAが時間情報・動きを学習したかはまだ評価していない。**

つまり、これまでが

> **「学習させる仕組みを作る段階」**

で、次から

> **「何を学習したのかを研究として評価する段階」**

に入るところです。

---

## 13. MTGで先生に確認したい質問

### 質問1

次の最優先評価はどれにするのが良いでしょうか。

- 学習前後の特徴表現の変化
- chronological vs shuffle
- temporal order / motionを直接測る新しい評価

### 質問2

chronological vs shuffleを行う場合、最初は

> **動画内のchunk順をshuffleする比較**

で良いでしょうか。

現在のmasked meanではclip内frame順shuffleは出力に反映されにくいため、まずchunkの更新順比較が自然だと考えています。

### 質問3

現在の4-stream round-robinを今後のbaselineとして続けてよいでしょうか。

それとも、

> **1本の動画を最初から最後まで処理してから次へ進むstrict online条件**

を先に作るべきでしょうか。

### 質問4

最初の評価規模をどの程度にするべきでしょうか。

決めたいものは、

- 学習step数
- 使用動画数
- 評価動画の分け方
- seed数
- 評価指標

です。

---

## 14. 今回言えること / まだ言えないこと

### 言えること

- ActivityNet実動画を使ったMoCo multi-step学習が100 step動作した。
- 各動画内のchunk順序は時系列順に維持された。
- Base ViTは固定された。
- Query LoRA / Projectorにはgradientが流れ、更新された。
- Key LoRA / ProjectorはEMAで追従した。
- Queueには異動画negativeが保持された。
- 100 step範囲ではNaN / Infなどの破綻は確認されなかった。

### まだ言えないこと

- 学習後の特徴が学習前より優れている。
- 100 stepで十分学習できている。
- chronological学習がshuffleより良い。
- 4-stream round-robinがstrict online学習と同じである。
- LoRAがframe orderを理解している。
- LoRAがmotionやtemporal informationを獲得した。
- 現在のMoCo設定が最適である。

---

## 15. 参照ファイル

- [前回MTG議事録](../meetings/2026-09-17-mtg.md)
- [Stage 6A spec](../specs/2026-09-24-stage6a-moco-multistep-canary-spec.md)
- [Stage 6A実装・検証記録](../experiments/2026-09-24-stage6a-moco-multistep-verification.md)
- [Stage 5実装・検証記録](../experiments/2026-09-24-stage5-moco-one-step-verification.md)
- [Stage 3 ActivityNet clip feature検証](../experiments/2026-09-23-activitynet-clip-feature-verification.md)
- [プロジェクトREADME](../README.md)

### 記録上の注意

プロジェクトREADMEには、Stage 6Aの実ActivityNet 10-step / 100-stepが「実行指示待ち」とする古い記述が残っている。

今回の資料では、より新しくAuthorityの高いStage 6A specと実装・検証記録に従い、

- 実ActivityNet 10-step PASS
- fresh 100-step PASS
- 実装commit `8304d033b2cf2da7e6636842ef250ed45a5693bb`

を最新状態として扱う。
