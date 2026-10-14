---
date: 2026-10-14
project: sequential-video-lora-analysis
source_todo: null
topic: post-lora-standalone-next-moco-validation
status: exploratory
tags: [brainstorm, research, lora, moco, staged-verification, online-learning]
---

# Standalone LoRA確認後の次の研究段階：MoCo単体検証

## 出発点・中心となる問い

ユーザーはCIFAR-10を用いた自作ViT Q/V LoRAの学習・Base凍結・監査証拠の作成ができたと報告し、「次にすべきこと」を相談。今回の目的は、2026-10-08教授MTGで未解決とされたMoCo lossの振動とQuery LoRAの勾配急増について、構成要素を切り離して原因を絞り、統合実験に戻るための検証順序を決めること。

## 読み込んだ文脈とAuthority

- 2026-10-08 MTGの決定：LoRA単体 → MoCo単体（標準的な入力と参照実装） → Sequential Loaderとの段階的統合。loss振動とQuery LoRA勾配急増の未解決時にはLinear Probeの結論へ進まない。
- 承認済み2026-10-13 Standalone ViT-LoRA CIFAR-10 spec、および2026-10-14教授向け自動Evidence出力spec。ユーザーは学習・凍結の実装検証完了を今回明示。これは自己教師あり表現学習・時間情報獲得の証拠ではない。
- 2026-09-24 Stage 5 MoCo実験記録は、ActivityNet 2動画1-step、Queue/InfoNCE/EMA/Query更新のengineering verificationが成功したとする。2026-09-28 Stage 6共通MoCo監査は人工短時間確認・メタデータQueueなどを報告。長時間学習の数値安定性・表現性能の成功を意味しない。
- 2026-10-08 MoCo learning curve exploratory note：Cometから非単調lossと数千〜数万規模の勾配ノルム継続的上昇を観測。原因は正規化前projector normの縮小、same-video negatives、optimizer/LoRA drift、video distribution shift等の未検証仮説。lossの単調減少を必須条件にしない。
- GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`の2026-10-14時点HEAD `7dcd194f439c7e28d796e2986857ab0df1982cbe` には standalone Q/V LoRAのmodelとCIFAR10 smoke、reportingのsourceがあるが、ディレクトリtreeにMoCo module / training pipelineは**存在しない**。以前のMoCoはResearch Workspaceのdated Evidenceでは未コミット実装とも記録されている。研究サーバ上のローカルworktree、未プッシュコード、別repositoryの実体はGitHubから確認できない。「現在dev上に従来MoCo実装がある」と仮定して進めない。

## 候補の比較と優先順位

### 案A：すぐにMoCo＋LoRA＋動画統合のfull run再開（非推奨）
- 以前のloss/gradient問題の原因の切り分けができず、計算量が大きい。Linear Probeの解釈不能状態を再現する懸念がある。

### 案B：MoCoを単体で段階的に検証（最有力）
- **準備**：まず過去MoCo実装がどのcommit/worktreeに存在するかを特定。参考実装のライセンス、公式性、現在モデルとの違いを整理。独立MoCoの参照コードを選定する。MoCoをすでに確認済みのCIFAR-10 LoRA smokeと無理に統合しない。
- **メカニクス検証**：LoRAなし・Sequential Loaderなしで、2-view augmentation、Query/Key encoder、Projector、temperature付InfoNCE、FIFO Queue、EMAを人工特徴・少数画像で検証。正のkeyと負のkey、過去queueだけを使う順序、正規化後L2=1、Key勾配0、Query勾配finite/非ゼロ、EMAが計算式と一致、Queueの長さとFIFO、future key非使用。
- **実画像small pilot**：標準的な画像入力（例：CIFAR-10）と参照MoCoのencoder（例：ResNet系、LoRAなし）、標準的な2-view augmentationを使い、まず短時間でforward/backward、その後独立controlled runでlossとpositive similarity、negative similarity、queue occupancy、有効negative数、pre-normalization projector norm、global gradient norm、実update norm、feature variance等を記録。既存実験のEMA/temperature/queue値は参照資料に基づく設定と切り分け、比較条件をspec時に固定する。lossが毎step減少することは必須条件にしない。非崩壊を確認できない限りpositive similarity単独の上昇を成功としない。
- **対照**：可能なら参照実装と独立実装を同じ画像/augmentation/seedで比較し、実行・出力の基準を得る。元公式MoCoはImageNet前提の環境依存があり得るので、そのままCIFAR-10で同等な学習曲線が得られると断定しない。

### 案C：従来統合MoCoのCometと既存checkpointのみ解析（並行すると有益）
- Projectorの正規化前L2 normが記録済みか、Queue中same-video占有割合、negative similarity、更新前後LoRA normを調査。新しい学習なしで仮説を絞れるが、保存されていない値は復元できない。
- 今回の最優先はMoCo単体再現性の取得なので、診断は余力があれば並行する。

## 次の研究ゲート（提案・未採用）

1. **LoRA成果物の確定**：今回の最新`verification_summary.md`、2枚のPNG、`report.json`、対応Git HEAD/dirtyを対応付け、実行証拠をResearch Workspace `experiments/` へ記録する。ユーザーは成功を述べているが、最新runの実artifactそのものはこの相談では未提出。
2. **MoCo単体spec化候補**：参照実装・画像dataset・encoder・augment・projector・EMA・Queue・温度・seed・指標・停止条件を研究上の判断として固定する（現時点では新規spec未作成）。最初はLoRAとSequential Loaderを使わない。MoCo v2型Queueありbaselineを第一候補とするのはstrict online条件に移植しやすいためで、reference実装の採用を確定した意味ではない。
3. **MoCo単体Gate**：unit / synthetic / tiny real-image / controlled multistepを独立に合格させ、MoCo loss数値の意味・勾配とQueue正常性を説明できること。
4. **LoRA＋MoCo統合**：Query側の確認済みLoRAとProjectorをgradientで更新し、Key側LoRA/ProjectorをEMA、Base ViT凍結、A/B別勾配・更新・pre-normalization normを記録。MoCo単体との差分を評価。新旧LoRA実装とPEFTの同一性を無条件に仮定しない。
5. **Sequential Loaderとの再統合・アブレーション**：画像検証 → 動画clip（shuffle/独立） → strict online順序へ段階化。negative policy（different-video vs same-video past）、stride/データ非定常性、projector normと勾配の関係を条件統制して検証。
6. **Linear Probe再開の条件**：上記メカニクス・数値健全性が検証でき、同じdataset split/manifestに対しBase vs LoRAの比較条件が固定された後に研究上の表現性能を評価。frame-wise CLS→masked meanはclip内順序不変なので、Linear Probe向上を時間順序理解の証明としては扱わない。

## 保留・反例

- 1-step unit PASSやCIFAR10 train loss低下は、MoCoのlong-run数値安定性・動画順序情報獲得を示さない。
- 標準MoCoのlossはqueue内negative数・augmentation・温度・データ分布に依存。単調減少・特定絶対値のみの成功判定は不適切。
- Frozen ViT＋Projectorのみの学習は、標準MoCoの「encoderが学習する」baselineと同一ではない。MoCo単体の参照baselineならtrainable ResNet等を用い、Frozen ViT＋LoRAは後段の統合で扱う方が分離明瞭。
- 参照MoCoのGitHubでの現在の正本実装・選定dataset、GPUの利用可否、ローカルMoCo worktreeは追加確認が必要。今あるGitHub `dev` に過去のMoCo実装が存在するとは扱わない。

## 次のResearch Specへの引き継ぎ候補

- 中心目的：Sequential Loader/LoRAを持ち込まずに標準MoCo v2型のメカニクスと学習挙動を、同一の小さな画像datasetで検証する。
- 参照：2026-10-08 MTG / 2026-09-24 Stage5 spec・実験 / 2026-10-08 MoCo gradient analysis。
- 必須：Query/Key gradient境界、EMA式、QueueのFIFOと過去のみ、InfoNCE logits数値一致、2-view relation、gradient finite、representation非崩壊指標と短時間学習ログ。
- Blocking：参照MoCo実装とdataset/encoder組合せ、既存MoCo実装再利用範囲、実画像pilotのステップ数/compute予算、augmentation/negative policy。
- 非対象：LoRA統合、strict video stream、Linear Probe、full ActivityNet run、VideoMAE変更。
- Authority：本書はexploratory。ユーザーの依頼は「次に何をすべきか」の相談であり、新たなspec・コード実装・run・TODO追加を許可するものではない。


## 2026-10-14 追加検討：なぜMoCo単体を先としたか、Loader＋自作LoRA先行との比較

### 問いと位置づけ

ユーザーからMoCo単体検証を先に推奨した理由を質問された。根拠は2026-10-08 MTGでの「MoCo loss振動・Query LoRA勾配急増が未解決、コンポーネントを独立に検証してから統合する」との**明示的方針**。ただし、MoCo単体を終えなければ自作LoRAとSequential Loaderを接続できないという**技術依存はない**。先の「MoCoを必ず先に」のニュアンスを修正。

### 選択肢の比較

| 順位候補 | 解消できる疑問 | 解消できない疑問・リスク |
| --- | --- | --- |
| MoCo単体先行 | InfoNCE/EMA/Queue、正負例、Projectorの数値挙動が動画順序やLoRAの影響なしに正常か。過去の大きな勾配の原因を切り分ける土台。10/8 MTG方針と一致 | ResNetなどの標準MoCoがPASSしてもViT/自作LoRA/strict-single環境での勾配安定性を証明しない。古いMoCoコードの実体は現在GitHub devで未確認であり準備負担がある |
| 自作LoRA＋Sequential Loaderのshort smoke先行 | 今回の新しいLoRA実装に、ActivityNet 16 frames→ViT→masked mean→LoRA gradient/updateという動画接続経路が確かに成立するか。CPU 1～2step程度の小さな確認で手戻りを防ぐ | InfoNCE/EMA/Queueの検証はできない。loss振動やQuery LoRA勾配急増の原因究明には直結しない |
| 全部一度に統合・長時間実行 | End-to-endが動くかを短絡的に確認できる可能性 | 不具合が出ても帰属先が分からず、過去の問題を繰り返すため非推奨 |

### 事実・推論・提案

- **確認済みの歴史的Evidence**：2026-09-24 Stage 4は以前の**PEFT LoRA**でActivityNetをSequential Loaderから読み、16-frame、masked mean、1-step LoRA更新、Base凍結を確認済み。現在の2026-10自作LoRAはCIFAR-10通常DataLoaderで検証されており、両者は別実装。新自作LoRA＋Sequential Loaderの接続は未確認。
- **推論**：過去の未解決現象がMoCo-specificな原因（Projector正規化、Queue内false negatives、online distribution shiftなど）ならMoCo単体検証は有益。ただし原因を特定できる保証はなく、標準画像MoCoだけでは動画順序由来の異常を再現できない。
- **更新した推奨**：研究の大きな優先課題はMoCo単体検証を維持しつつ、**直近の工学タスクとしては自作LoRA＋Sequential Loaderの1～2step小規模接続を先に行ってもよい**。双方が独立して合格した後に自作LoRA＋MoCo、最後にstrict Sequential Loader＋ActivityNetに統合する。異常時にはステージごとの差分比較を使う。
- **境界**：これは探索的順序修正であり、10/8のMTG決定を取り消した事実を意味しない。仕様・コード・TODO・実験runは今回変更しない。
