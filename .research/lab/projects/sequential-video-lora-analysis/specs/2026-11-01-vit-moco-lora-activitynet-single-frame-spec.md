---
project: sequential-video-lora-analysis
spec_type: experiment-and-implementation
status: implemented
created: 2026-11-01
approved: 2026-11-01
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
base_branch: feature-vit-moco
base_commit: 387456875e40f7e1ad9037b43bf631abc5221e63
source_evidence: ../experiments/2026-11-01-vit-moco-low-lr-l5-1000step-imagenet-gpu-evidence.md
source_brainstorm: ../../../../secretary/notes/brainstorm/2026-11-01-vit-moco-lora-transition-to-activitynet.md
parent_spec: 2026-10-31-vit-moco-low-lr-long-run-spec.md
---

# ViT＋MoCo（LR 3e-4）：学習入力をActivityNet単一フレームへ差し替える診断run spec

**Status: implemented（2026-11-01。CPU Gate T0〜T3はPASS、研究コードは`feature-vit-moco@167f1c1`でcommit済み。GPU A1/A2は未実施。§12参照）**。**Status: approved（2026-11-01。C1〜C6はユーザーが推奨案のとおり承認）**。研究コードの実装とGPU runは、それぞれ別に許可を得てから行う（§9）。

## 1. 目的と背景

[L5 Evidence](../experiments/2026-11-01-vit-moco-low-lr-l5-1000step-imagenet-gpu-evidence.md)では、ImageNetでLR 3e-4のとき、Queueが埋まった後にMoCo lossが単調に減少し、1,000stepまで遅れて起きるcollapseはなかった（暫定）。

次の段階として、**学習入力だけを動画のフレーム（ActivityNet）に差し替え**、同じ問いに答える。

1. ActivityNetの単一フレームで、Queueが埋まった後（256stepより後）にMoCo lossは下がるか。
2. collapseは起きないか。

この段階では1フレームずつ入力するため、fusionはない（T=1）。動画clipとearly fusion（[early fusionの壁打ち](../../../../secretary/notes/brainstorm/2026-11-01-early-fusion-vit-lora-temporal.md)）は次の段階で、別のspecとする。2026-11-01、ユーザーは単一フレームの段階を先に行うことを選んだ。

## 2. 前提（コード調査で確認済み、`3874568`）

- runnerは `data.source=activitynet_frames` と `data.activitynet_manifest` を受け付ける（`moco_vit_ddp_smoke.py` の `build_dataset`）。データ拡張（MoCo v2 aug-plus）、正規化、`DistributedSampler`（shuffle、seed 0）、`drop_last=True` はImageNetと同じ経路を使う。
- 既存のmanifestは `log/activitynet_frames/2026-10-21_training_interval10s_offset5s/manifest.json`（sha256 `065c3170…b472`、`frames_sha256` `c9cc4aec…105c`）。ActivityNet v1.3のtraining split、10秒間隔・offset 5秒で、10,024動画・117,522枚（フレーム0枚の動画が26本、1動画あたり平均11.7枚、最大75枚）。positiveは「1枚のフレームの2つの拡張」で、**同じ動画の別フレームもnegativeから除外しない**（manifestの方針）。
- **1epoch = 459step**：1GPUあたり117,522 / 2 = 58,761枚、batch 128で割って459（端数は切り捨て）。ImageNetは5,004step。
- Queue K=65,536 / global batch 256 なので、**256step目で全体が本物のkeyに置き換わる**（ImageNetと同じ）。
- LRは5epochを1周期とするcosineで、epochの開始時に設定される。459step以内（epoch 0）なら、**LRは全期間3e-4で一定**。
- **collapse診断は現在ImageNetでしか動かない**：`query_collapse_diagnostic.check_config` は `data.source != "imagenet"` を拒否する。固定probeは、学習用のImageFolder（`dataset.samples`）から選ぶ（`load_probe`）。probeのmanifestは `record["data"]["samples_sha256"]` を参照するが、ActivityNetのdata記録にはこのkeyがない。
- GPU runでLR 3e-4と0.03以外のstep数を使える経路は、presetが `moco_vit_query_collapse_lr`（10step）と `moco_vit_lr_long`（1,000step、`data.source` は実質ImageNetのみ）しかない。**新しいpresetとprofileが必要**。
- ActivityNetのフレームは、1枚ごとにNAS上の動画をseekしてdecodeする。**GPU runでの読み込み速度は未測定**（GPUでActivityNetのrunを行った記録はない）。
- L4/L5の固定probe入力のハッシュ（`probe_input_sha256`）：`6af50745…08d8bf`（LoRAとFull FTで共通）。

## 3. Decision Contract（承認済み）

| # | 論点 | 推奨 |
|---|---|---|
| C1 | この段階の問い | **L5と同じ2つの問い**（§1）。表現品質の評価は対象外（別spec） |
| C2 | 学習の長さ | **新規に、まず10step（smoke）。工学PASSを確認してから459step（ちょうど1epoch）**。459stepを超える学習は別に許可を得る |
| C3 | LRとschedule | **LR 3e-4**。既存のepoch単位cosine（`epochs=5`、warmupなし）はそのまま。C2により、全期間3e-4で一定になる |
| C4 | 固定probe | **L4/L5と同じImageNet 64枚**（seed 0、Resize256→CenterCrop224、microbatch 8、`update_steps=[1]`）。学習データはActivityNet、probeはImageNetのtrain ImageFolderから選ぶ。ActivityNetのprobeは作らない |
| C5 | 学習方式と順番 | **LoRAを先に実行し、工学PASSを確認してからFull FT**。それぞれ10step→459step |
| C6 | フレームのリスト | **既存のmanifest（10秒間隔・offset 5秒）をそのまま使い**、sha256で固定する。10秒は仮の値で、研究的な妥当性は未検証であることを記録に残す |

### probe step と checkpoint（C2・C4から導かれる値）

| 長さ | probe step | checkpoint |
|---|---|---|
| 10step | `[0, 1, 2, 10]` | なし |
| 459step | `[0, 1, 2, 10, 100, 256, 459]` | `[459]` |

### 対象外

- 動画clip、early／late fusion、時間位置埋め込み、時間に敏感なpositive/negativeの設計（次の段階のspec）
- 459stepより長い学習、Queueの大きさの変更、LRの探索、複数seed
- 同じ動画のフレームをnegativeから除外すること（manifestの方針を変えない）
- フレーム間隔の再検討、新しいmanifestの作成
- ActivityNetの固定probe、表現品質の評価、LoRAとFull FTの優劣
- 既存のG3/D4/L4/L5 preset、0.03のrecipe、既存の出力namespaceの変更
- Comet連携（runディレクトリを記録の正本とする）

## 4. 実装要件（最小差分）

1. 新しいpreset（名前の例：`conf/moco_vit_lr_activitynet.yaml`）
   - `moco_vit_query_collapse_lr` を継承する。
   - `diagnostics.profile` を新しい値（例：`lr_activitynet`）にする。
   - `data.source: activitynet_frames`、`data.activitynet_manifest` は§2の既存manifest。
   - `training.learning_rate` は従来どおり**必須**とする（省略時に値を黙って補わない）。
   - 出力先：`log/moco_vit_lr_activitynet/${model.train_mode}/activitynet_frames/lr_${training.learning_rate}/<timestamp>`
2. 診断の固定probeは、学習データの種類によらず `data.root`/`data.split`（ImageNet train）のImageFolderから選ぶ。**ImageNetの既存経路では、選ばれる画像・probeのmanifest・出力が変更前とbitwiseで一致しなければならない**。ActivityNetのrunでも、probe入力のハッシュは `6af50745…08d8bf` と一致しなければならない。
3. `data.source=activitynet_frames` を診断で受け付けるのは、**新しいprofileかつ新しいpresetのときだけ**。他のprofileでは、従来どおり `imagenet` 以外を拒否する。
4. GPUでの新しいprofileは、次以外をすべて拒否する（fail-closed）。
   - `max_steps ∈ {10, 459}`。それぞれ§3の表のprobe stepとcheckpointに限る。
   - manifestは§2のもの（manifestのsha256で照合する）。
   - LRは既存の `lr_sensitivity` と同じ扱い（正の有限値のみ）。
   - その他のrecipeとprobeの値は採用済みの値。
   - 既存profileのstep数制限（runnerは1/10/20、診断は10、`lr_long` は1,000）は他の経路でそのまま維持する。
5. 既存の比較器（`--comparison-axis train_mode`）で、ActivityNetの新しいrunについてLoRAとFull FTを比較できるようにする（step0の特徴、学習入力、probe入力の一致確認）。また、`prefix` 軸で10step runと459step runの共通部分を比較できるようにする。既存の比較器で足りるなら変更しない。ImageNet runとActivityNet runの比較は、比較器で拒否してよい（データが違うため）。
6. `metrics.jsonl` は既存どおり毎stepを記録する。集計は解析時に行う。

## 5. 互換性

- 既存presetを使ったCPUとGPUの経路は、変更前とbitwiseで一致しなければならない（診断のON/OFF、0.03、`lr_sensitivity`、`lr_long` の各経路、ImageNetのprobe選択）。
- 全テストの失敗ID集合が、変更前後で同じであること。

## 6. 検証（実装時、CPUのみ）

| Gate | 内容 |
|---|---|
| T0 | 新しいpresetとprofileに対する単体テスト。拒否すること：LR未指定、`max_steps ∉ {10, 459}`、長さとprobe step・checkpointの組み合わせ違い、別のmanifest、ImageNetの指定、他のpresetから新profileを使うこと、他のprofileでActivityNetを使うこと。出力先が別になること |
| T1 | 既存経路の非回帰。既存の診断テスト（D0〜D3、L0〜L3相当、lr_longのT0〜T3）が通ること。失敗ID集合が変更前後で同じであること |
| T2 | tiny ViT・2rank・glooで、小さな人工ActivityNet形式のmanifest（テスト用）と小さなImageNet形式のprobeを使い、新profileを短いstep数に置き換えて完走させること。probe・checkpoint・summaryが揃い、probe入力がImageNet経路と同じ選び方で一致すること |
| T3 | 比較器：同じActivityNet入力のLoRAとFull FTの組み合わせで `comparable=true` になること。10step runと長いrunの共通prefixを比較できること。ImageNetとActivityNetの組み合わせは比較が成立しないこと |

## 7. GPU run（A1〜A2、別に許可が必要）

```bash
# A1: 10step smoke（LoRA→Full FT）
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet model.train_mode=lora training.max_steps=10 training.learning_rate=0.0003
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet model.train_mode=full_ft training.max_steps=10 training.learning_rate=0.0003
# A2: 459step（1epoch、LoRA→Full FT）。A1の両方式が工学PASSで、読み込み速度を確認した後
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet model.train_mode=lora training.max_steps=459 training.learning_rate=0.0003
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet model.train_mode=full_ft training.max_steps=459 training.learning_rate=0.0003
```

- 研究リポジトリをclean commitにしてから実行する。
- GPU構成はL4/L5と同じ（`CUDA_VISIBLE_DEVICES=0,1`、GPU0がRTX PRO 4500 Blackwell、GPU1がRTX A4500）。
- A1で1stepあたりの時間（`step_time_sec`）を記録する。L5のImageNetより大幅に遅い場合は報告し、A2を実行するかをユーザーが判断する。

## 8. 成功条件

### 工学的な成功（PASS / FAIL）

- 両方式・両方の長さで、`status=completed`、`failures=[]`、全stepがfinite、rank間でstateが一致、frozen baseが不変（LoRA）、VRAM Gate PASS、診断の `engineering_status=PASS` であること。
- probe入力のハッシュが、L4/L5と同じ `6af50745…08d8bf` であること。
- 同じ長さのLoRAとFull FTで、step0の特徴・学習入力・probe入力が一致すること（比較器 `comparable=true`）。
- 459stepのrunで、checkpoint（step 459）と全probe stepの記録が揃っていること。
- **run間のbitwise一致は求めない**（L5でGPU backwardのrun間非決定性を確認済み）。10stepと459stepの共通prefixの比較は記録用で、step0とstep1のlossのbitwise一致、それ以降は差の大きさを報告する。

### 科学的な観測（記述的に報告し、閾値でPASS / FAILを決めない）

- loss、positive cosine、hardest negative cosine、`query_feature_std` の推移。**256stepより前と後を分けて**、移動平均（例：50step窓）で示す。
- 固定probeの各stepにおける CLS / z_q / q の画像間cos p50・p95、mean_dim_std、row normの推移。
- 既存の警告ヒューリスティック（`q mean_dim_std < 1e-4` かつ `cos p50 > 0.999`）が出るかどうか。
- L5（ImageNet）との違いは記述にとどめる。データが違うため、bitwiseでの比較もprefix比較も行わない。
- 1stepあたりの時間と、データ読み込み待ちの有無。

lossが下がらない場合も、collapseした場合も、そのまま結果として報告する。

### 主張しないこと

時間情報・動き情報を獲得したこと、表現品質、10秒間隔の妥当性、LoRAとFull FTの優劣。

## 9. 実行の境界

- このspecの承認は、**研究コードの実装を許可するものではない**。実装は `engineering-task` で、ユーザーが改めて指示してから行う。
- GPU run（§7のA1、A2）は、実装とCPU検証の後に**それぞれ別に許可を得る**。
- commit、push、ブランチ操作は、ユーザーの指示があったときだけ行う。

## 10. 未決事項

- **blocking**：なし（C1〜C6は2026-11-01に承認済み）。
- **non-blocking**（実装時に既存の方式に合わせる）：presetとprofileの名前、テスト用の人工manifestの作り方、比較器でImageNetとActivityNetの組み合わせを拒否するときのメッセージ。

## 11. 承認記録

- 2026-11-01：ユーザーがC1〜C6を推奨案のとおり承認した。

## 12. 実装記録

- 2026-11-01：`feature-vit-moco@3874568` の上に未commitで実装した。CPU Gate T0〜T3はすべてPASS。変更後の全テストで新しく失敗したテストは0件。実際のImageNetから選び直した固定probeの入力ハッシュは `6af50745…08d8bf` で、L4/L5と一致した。比較器は変更していない。**§7からの変更点**：presetの既定値は459step用で、10stepのsmoke（A1）ではprobe stepとcheckpointをコマンドで明示する（`'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'`）。GPU A1/A2は未実施で、実行前に研究コードのcommitが必要。同日、ユーザーの指示で `feature-vit-moco@167f1c1` としてcommitした（push未実施）。詳細は[実装・CPU検証記録](../experiments/2026-11-01-vit-moco-lora-activitynet-single-frame-implementation-cpu-verification.md)。
