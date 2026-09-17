---
date: 2026-09-17
project: sequential-video-lora-analysis
type: meeting-materials
topic: MAE基盤の進捗確認と次段階の選択
status: draft
target_meeting: 2026-09-17
notion_url: null
notion_export: null
tags: [meeting-materials, research, mae, lora, sequential-loader, temporal-modeling]
---

# 2026-09-17 MTG資料: MAE基盤の進捗確認と次段階の選択

## 0. 今日のMTGで先生に相談したいこと

1. **次の最小実装をどちらから進めるか**
   - 提案: まず単一画像で `Base MAE固定 + encoder attention q/vのLoRAのみ更新` を確認し、その後に `sequential_loader -> 50Salads -> MAE` の接続smokeを行う。
   - 代案: loader接続を先に確認し、実動画frame経路を固めてからLoRAへ進む。
2. **動画由来情報をLoRAへ帰属できる設計原則を置くか**
   - 独立したtrainable temporal fusionを置くと、動的情報がLoRAではなくfusion側へ保存される交絡が生じる。
   - temporal pathwayのbaseを固定してLoRAだけを学習する案を主候補にしてよいか。
3. **時間情報を主張する前に必要な最小controlを何にするか**
   - `normal order / shuffled order / static repeat / no-past` を候補とし、少なくともframe-independent MAE+LoRAをnegative controlとして残してよいか。

## 1. 現時点の結論・提案

- 研究の出発点は、K400 pretrained MeMViTによる教師あり逐次LoRAから、ImageNet pretrained MAEによる自己教師あり動画LoRAへ変更した。
- MAEについては、単一画像の標準reconstruction loss計算と、Lightningの通常training pathによる1 optimization stepまで実装済みである。
- 現在解消できたのは、**MAEをロードし、lossを計算し、勾配更新できるかという基盤上の不確実性**である。
- まだLoRAは導入しておらず、動画frame、時間方向の演算、online stateも扱っていない。したがって、時間情報・動作情報に関するEvidenceはまだない。
- 次は研究要素を混ぜず、次のGateを順に通すことを提案する。

```text
Gate 1: 単一画像 + Base固定 + LoRA-only 1-step
  -> Gate 2: sequential_loader + 50Salads frame + MAE forward
  -> Gate 3: causal temporal designの実装契約
  -> Gate 4: 動画自己教師あり学習 + temporal control
  -> Gate 5: LoRA内部・表現の解析
```

## 2. 前回MTGからの流れ

| 日付 | 段階 | 状態 | この段階で確認したこと |
|---|---|---|---|
| 2026-09-10 | MeMViT逐次LoRA基盤 | 最小要素を確認 | loader出力のshape変換、MeMViT q/vへのLoRA注入、最小1-step更新 |
| 2026-09-11 | 研究方針の変更 | 探索方針を整理 | ImageNet pretrained MAEを出発点とし、動画を自己教師あり学習する方向へ変更 |
| 2026-09-16 | MAE単一画像forward | implemented | `facebook/vit-mae-large`で標準MAE reconstruction lossを計算できる経路を確認 |
| 2026-09-17 | MAE単一画像1-step | implemented | Lightning `Trainer.fit(max_steps=1)`、AdamW、全parameter更新の経路を確認 |
| 未着手 | MAE + LoRA | 未実装 | Base不変・LoRAのみ更新することの確認 |
| 未着手 | 動画frame接続 | 未実装 | sequential_loader、50Salads、MAE前処理のinterface確認 |
| 未着手 | temporal / online学習 | 設計候補のみ | causal fusion、cache、reset、detach/stalenessの契約 |

## 3. Evidence: 確認できていること

### 3.1 単一画像MAE forward

- checkpointは `facebook/vit-mae-large`。
- 入力は `pixel_values [1, 3, 224, 224]`、FP32、single GPU。
- 標準の75% patch maskingとMAE decoderを通り、scalarかつfiniteなreconstruction lossを得るsmokeが実装済みである。
- 期待shapeは `logits [1, 196, 768]`、`mask [1, 196]`、`ids_restore [1, 196]` である。
- この段階ではbackward、optimizer、parameter updateを行っていない。

### 3.2 単一画像MAE 1-step学習

- MAE専用LightningModuleへ `training_step` と `configure_optimizers` を追加済みである。
- optimizerはAdamW、`lr=1e-4`、`weight_decay=0.0`、schedulerなし。
- 1 sample / 1 batchに対して `Trainer.fit(max_steps=1)` を実行するsmokeが実装済みである。
- 全MAE parameterをtrainableとし、encoderの代表parameter
  `model.vit.embeddings.patch_embeddings.projection.weight` のbefore/after差分を検証する。
- 最新specは、finite loss、`global_step == 1`、代表encoder parameterの有限かつ正のdeltaを満たしたとして `implemented` になっている。

### 3.3 実装リポジトリの照合

- 確認時のbranchは `mae`、HEADは `681f513` で、`origin/mae` と一致している。
- `MAELightningModule` と単一画像1-step smokeのコードが存在し、上記specのoptimizer・更新確認契約に対応している。
- Research Workspaceの `experiments/` には、loss値、delta norm、実行環境を含む実行ログがまだ保存されていない。本資料作成時にはsmokeを再実行していない。
- 実装worktreeには本機能外の既存差分があるため、次のPR・再現実行前に対象差分を切り分ける必要がある。

## 4. Interpretation: 現時点の見立て

### 4.1 基盤として何が前進したか

MAE導入時に最初の障害となるcheckpoint、前処理、出力契約、Lightning training path、optimizer updateは段階的に切り分けられた。次のLoRA smokeで問題が起きた場合、MAEそのもののforward/training pathではなく、freeze、LoRA注入、trainable parameter選択へ原因を絞りやすい。

### 4.2 なぜ次にLoRA-only更新を提案するか

研究の中心は「動画学習によってLoRAに何が追加されたか」である。動画やtemporal moduleを入れる前に、単一画像という最小条件で次を確認すると、parameter-efficient adaptationの経路を独立に検証できる。

- Base MAE全体が不変である。
- 指定したLoRA parameterだけがtrainableである。
- 1-step後にLoRAが更新される。
- frozen decoderを含む標準MAE reconstruction lossからLoRAへ勾配が届く。

### 4.3 temporal設計で残る交絡

frame-independentなMAE lossだけでは、動画frameを時系列に与えても各frameのappearance/domain adaptationだけでlossを下げられる。さらに、別のtrainable temporal fusionを追加すると、時間情報がfusion parameterへ入り、LoRAの寄与を解釈しにくくなる。

そのため、最終性能だけで設計を選ぶのではなく、**動画情報の保存先を識別できること**を要件に含める必要がある。固定temporal operatorへのLoRA、またはtemporal attention自体のLoRA adaptationは、この目的と整合しやすい候補である。

## 5. 今回言えること / まだ言えないこと

### 今回言えること

- ImageNet pretrained MAEの単一画像reconstruction経路を現在の研究コードへ追加できた。
- Lightningの通常training pathで、標準MAE lossから1-stepのparameter updateを行う実装契約は成立した。
- MAE、LoRA、loader、temporal設計を独立したGateへ分けて検証できる状態になった。
- frame-independent MAE+LoRAは、将来のtemporal modelに対する必要なcontrolになり得る。

### まだ言えないこと

- Base MAEをfreezeしてLoRAだけを更新できるか。
- どのLoRA挿入位置が有効か、または最適か。
- `sequential_loader` の実動画frameをMAEへ正しく入力できるか。
- 時系列順、過去frame、motionがloss低下や表現形成に使われるか。
- LoRAが時間情報・動作情報を保持するか。
- 現在の1-step smokeでlossが改善するか、再構成品質や汎化性能が向上するか。

## 6. 次段階の候補・比較

| 候補 | 直近で行うこと | 長所 | 主な注意点 |
|---|---|---|---|
| **A. LoRA-only smokeを先に行う（提案）** | MAE encoder attention q/vへLoRAを追加し、Base固定・LoRA更新を単一画像1-stepで確認 | 研究の中心parameterを最小条件で切り分けられる | 実動画のdata path確認は次のGateになる |
| B. loader接続を先に行う | 50Saladsの1〜数chunkを前処理し、frame-wise MAE forwardを確認 | 実データのshape・値域・metadata問題を早期発見できる | LoRA経路の不確実性が残る。temporal学習のEvidenceにはならない |
| C. temporal fusionへ進む | past/current tokenを融合し、current masked patchを復元 | 研究提案へ早く近づく | LoRA、loader、temporal moduleの問題が混ざり、情報保存先も交絡する |

### 提案する順序

1. **LoRA-only単一画像1-step**
   - encoder attention q/vを最初の挿入候補とする。
   - Base MAE全体をfreezeし、LoRA以外の不変性を確認する。
2. **sequential_loader接続smoke**
   - 50Saladsのlabelは使わず、自己教師あり入力frameとして扱う。
   - shape、値域、frame index、sequence境界、finite lossを確認する。
3. **temporal designのspec化**
   - current-frame reconstructionかfuture predictionかを選ぶ。
   - causal処理、cache reset、detach、stalenessを固定する。
   - temporal parameterとLoRAの責務を固定する。
4. **短時間のcontrol実験**
   - frame-independent、normal、shuffled、static repeat、no-pastを同条件で比較する。

## 7. Ask: MTGで決めたいこと

- [ ] 次のPRを「単一画像でのBase固定・LoRA-only 1-step」にしてよいか。
- [ ] 最初のLoRA挿入位置をMAE encoder attentionのq/vとし、decoderを含む既存MAE parameterはfreezeしてよいか。
- [ ] LoRA smokeの次に、`sequential_loader -> 50Salads -> MAE` の接続smokeを独立PRで行ってよいか。
- [ ] temporal designでは、独立したtrainable fusionへ情報が逃げる交絡を避け、「固定temporal operator + LoRA」または「temporal attention側LoRA」を優先候補としてよいか。
- [ ] 時間情報に関する主張の最低条件として、frame-independentに加え、`normal / shuffled / static repeat / no-past` のcontrolを比較する方針でよいか。
- [ ] 実験段階へ進む前に、loss、parameter差分、commit、環境情報を `experiments/` に残す運用へ揃えてよいか。

## 8. MTG後に進む候補

上記提案が了承された場合、次は以下を別specとして固定する。

- LoRA実装方式と依存version
- 対象moduleの厳密な名前
- rank、alpha、dropout
- Base不変性とLoRA更新の観測方法
- single GPU / FP32 / seed / optimizer条件
- 既存forward smokeと全parameter 1-step smokeの非回帰条件

長時間学習、50Salads全体run、temporal module実装は、このLoRA smokeへ含めない。

## 9. 口頭説明用の短いまとめ

> 研究方針をImageNet事前学習MAEから動画の自己教師ありLoRAを学ぶ方向へ変更しました。現在は、単一画像で標準MAE lossを計算し、Lightningで1-stepだけ全parameterを更新できるところまで段階的に確認しています。ただし、LoRA、動画frame、時間方向の処理はまだ入っていないため、時間情報に関する結果はまだありません。次はまずBaseを固定してLoRAだけが更新される最小経路を確認し、その後にsequential loaderを接続したいです。また、将来のtemporal fusionに別のtrainable parameterを置くと動画情報がLoRA以外へ逃げるため、LoRAへ情報を帰属できる設計原則とcontrol条件を今日相談したいです。

## 10. 参照ファイル

- [プロジェクトREADME](../README.md)
- [2026-09-10 MTG議事録](../meetings/2026-09-10-mtg.md)
- [MAE単一画像forward smoke spec](../specs/2026-09-11-mae-single-image-reconstruction-smoke-spec.md)
- [MAE単一画像1-step学習 smoke spec](../specs/2026-09-16-mae-single-image-one-step-training-smoke-spec.md)
- [MAE自己教師あり動画LoRAへの方針変更](../../../../secretary/notes/brainstorm/2026-09-11-mae-self-supervised-video-lora-pivot.md)
- [sequential_loaderからMAEへの接続smoke案](../../../../secretary/notes/brainstorm/2026-09-11-mae-sequential-loader-smoke-test.md)
- [オンラインMAE late fusionの設計整理](../../../../secretary/notes/brainstorm/2026-09-11-online-mae-late-fusion.md)

