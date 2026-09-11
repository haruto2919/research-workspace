---
date: 2026-09-11
project: sequential-video-lora-analysis
source_todo: null
topic: MAEとCLIPのどちらを主軸にするか
status: exploratory
tags: [brainstorm, research, mae, clip, lora, video, temporal-modeling]
---

# MAEとCLIPのどちらを主軸にするか

## 出発点

研究目的は、画像で事前学習されたViTを基本的に保持し、動画を用いた追加学習でLoRAへ時間的・動的情報を獲得させ、その獲得情報を解析することである。

現在の候補は次の2系統。

- MAE事前学習ViT + LoRA + 動画自己教師あり学習
- CLIP事前学習ViT + LoRA + temporal modeling + semantic evaluation

## 確認済み事実

### MAE

- masked patch reconstructionを用いる自己教師あり画像事前学習である。
- decoderを持ち、元pixelを教師信号としてreconstruction lossを計算できる。
- 動画へ拡張する際、過去frame + current frameからcurrent masked patchを復元するなど、ラベルなしvideo self-supervised objectiveを構成しやすい。
- 一方、学習後にLoRAが「何を意味として獲得したか」を直接言語で解釈する仕組みは標準では持たない。

### CLIP

- image-text contrastive pretrainingにより、画像encoderとtext encoderを共通embedding spaceへ整列する。
- image encoder単体は基本的にframe-level画像モデルであり、temporal modelingは別途必要である。
- frozen CLIP backboneに軽量なtemporal moduleを追加してvideoへ適応する先行例（EVL等）がある。
- text encoderを固定probeとして使うことで、動画LoRA適応前後のrepresentationをaction/state-change promptとの類似度で評価できる。

## 比較

### MAEが向いている点

- 純粋なvideo self-supervised learningという研究条件を保ちやすい。
- ラベルやcaptionを使わずに学習できる。
- reconstruction lossが自然に定義できる。
- temporal objectiveの因果関係を比較的きれいに設計できる。

### MAEの弱点

- LoRAが獲得した情報の意味解釈が難しい。
- temporal informationを使ったかどうかを、shuffle/reverse/no-past等のablationやprobeで別途検証する必要がある。
- downstream semantic evaluation用の仕組みを追加で用意する必要がある。

### CLIPが向いている点

- image prior自体がsemanticで、action/object/stateを自然言語と比較できる。
- LoRA適応後に「どの時間変化・action概念へ近づいたか」をtext promptで解析しやすい。
- normal/reverse/shuffleに対するaction prompt similarityを比較でき、時間構造利用の評価を意味空間上で行いやすい。
- VLMやopen-vocabulary video understandingへの発展とも整合する。

### CLIPの弱点

- 標準CLIPにはMAEのようなvideo self-supervised lossがない。
- fine-tuning objectiveを別途設計する必要がある。
- temporal moduleを別trainable parameterにすると、動画情報がLoRAではなくtemporal moduleへ入る交絡が起こる。
- text supervisionを学習時にも使う場合、純粋なvideo self-supervised learningという現在の問いから研究目的が変わる。

## 現時点の推奨

現在の研究目的を「LoRAへ動画の時間的・動的情報を追加し、その獲得情報を解析する」と置くなら、主軸候補としてはCLIPをやや優先する価値が高い。

理由:

1. 今回は単に動画でLoRAを学習させることではなく、学習後にLoRAが何を獲得したかを解析することが中心である。
2. CLIPは固定text encoderを意味的な物差しとして使えるため、LoRA適応前後のsemantic shiftを観測しやすい。
3. pick up / put downのように静止画のappearanceが似ていて時間順序で意味が変わるactionを、text promptとのsimilarityで評価しやすい。
4. frozen image model + lightweight video adaptationの先行例があり、image priorを保持してvideoへ拡張する研究設定にも整合する。

ただし、video self-supervised learningの純度を最優先するならMAEの方が研究設計はきれいである。

したがって現時点では次の役割分担が有力。

- Main candidate: CLIP pretrained ViT + LoRA + causal temporal modeling
  - 学習時にtextを使わないvideo self-supervised objectiveを可能なら採用する。
  - 評価時にfrozen CLIP text encoderをsemantic probeとして使う。
- Baseline / control: MAE pretrained ViT + LoRA
  - reconstruction-based video SSLとして比較対象にする。

## 重要な未解決事項

- CLIP側でLoRAを何のlossで動画fine-tuningするか。
- temporal pathをどの構造にするか。
- temporal pathに独立trainable parameterを持たせるか、LoRAへ学習自由度を寄せるか。
- 学習時にtext supervisionを使わない条件を維持するか。
- semantic評価を候補文similarityだけで行うか、representation probeも併用するか。

## 次の判断候補

次に最優先で決めるべきなのは、CLIPを主軸候補とする場合のvideo fine-tuning objectiveである。
候補例:

- past featuresからcurrent/next CLIP featureを予測
- masked temporal feature prediction
- temporal contrastive learning

このメモは探索記録であり、specまたは実装許可ではない。

## 2026-09-11 15:52 JST 追記: MAE方針を採用

ユーザー判断により、研究の主軸はCLIPではなく **ImageNet事前学習済みMAE + LoRA + 動画自己教師あり学習** とする方針を採用した。

### 採用した理由

- 動画学習時にラベルやcaptionを使わず、元動画そのものを教師信号にできる。
- masked reconstructionという明確なobjectiveがあり、まず学習基盤を成立させやすい。
- 「画像で事前学習されたBaseを保持し、動画由来の情報をLoRAへ追加する」という研究条件を単純に構成しやすい。
- temporal modelingの有無やnormal / shuffle / reverse / no-past等のablationにより、時間構造利用を比較しやすい。

### 現在の主フロー

1. Hugging FaceのImageNet事前学習済みMAEを現在の実装基盤へ追加する。
2. MAE専用LightningModuleを用意し、単一画像で標準MAE forward / reconstruction lossを確認する。
3. `sequential_loader -> 50Salads -> pretrained MAE` の接続smoke testを行い、実動画frameをMAEへ正常に渡せることを確認する。
4. MAE encoderへLoRAを追加し、Base MAEをfreezeした状態でLoRAのみ更新できることを1 stepで確認する。
5. 動画の時間関係を利用するtemporal designを決める。現時点の有力候補は、過去frame + current frameのencoder tokenをcausalにfusionし、current frameのmasked patchを復元する構成。
6. online / sequential学習へ拡張し、future leakageなし、sequence reset、past cache等の契約を決める。
7. ActivityNet / EPIC-KITCHENS等で同条件の動画LoRAを学習する。
8. normal / shuffle / reverse / static-repeat / no-past等で時間構造依存性を評価する。
9. LoRA parameter / representation / downstream probe等を用いて、動画由来情報がどこまでLoRAへ保持されたか解析する。

### 直近の実装順序

- Step 1: pretrained MAE integration
- Step 2: MAE用trainer / LightningModule
- Step 3: sequential_loader + 50Salads smoke test
- Step 4: LoRA injection + freeze/update確認
- Step 5: temporal designのspec化
- Step 6: online video SSL実装
- Step 7: dataset比較とtemporal evaluation

CLIPは現時点では主軸から外し、必要になれば将来のsemantic evaluation / comparison baseline候補として再検討する。

この追記も探索記録であり、既存draft specを自動的に更新・承認するものではない。
