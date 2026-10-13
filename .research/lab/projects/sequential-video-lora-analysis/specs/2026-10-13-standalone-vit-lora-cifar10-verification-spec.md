---
project: sequential-video-lora-analysis
spec_type: implementation
status: draft
title: 髙間氏方式を参考にしたViT自作LoRAの独立実装とCIFAR-10短時間検証
created: 2026-10-13
last_updated: 2026-10-13
workspace_repository: haruto2919/research-workspace
workspace_branch: main
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_reference_commit: b6385d87e2e3e82a719d8f4b686b44aa293b1135
implementation_worktree: "user-reported local encoder-only; actual HEAD/dirty unknown"
hf_checkpoint: google/vit-base-patch16-224
dataset_candidate: CIFAR-10
default_targets_candidate: [q, v]
initialization: A_random_B_zero
scope: isolated LoRA mechanism plus ordinary supervised image-learning smoke
---

# ViT自作LoRA単体検証（Q/K/V切替・CIFAR-10）実装spec — draft

> **Status: draft — 仕様のレビュー用。コード実装・学習実行の許可ではない。**
>
> 現ユーザーの明示採用事項と、推奨されているが未承認の設計を区別する。§10にblockingな未決事項が残る間はapprovedにしない。

## 1. 目的と研究上の位置付け

2026-10-08 MTGで、既存のMoCo + LoRA統合経路におけるLoss振動・勾配急増の原因が未特定であることから、**Sequential Loader・MoCo・LoRAを分離して個別に挙動を検証**する方針となった。

今回の目的は、画像事前学習済みViT-Baseの重みを固定し、髙間氏のProjectedViTにある「低ランク行列A/Bの補正をAttention射影の出力へ加算する」方式を参考にした**説明可能な自作LoRA**を実装して、Q/K/Vの任意の非空組合せで動作することを検証すること。その後、Sequential Loaderに依存しない**通常の実画像分類の短時間学習**で、LoRA経路に勾配が流れ重みが更新されることを確認する。

検証主張を明確に分離する。

1. **工学的検証**: 正しい箇所への挿入、初期等価性、パラメータ凍結、勾配、Optimizer更新、再現可能性。
2. **短時間の実データ学習確認**: 実画像、教師ラベル、分類Head、Cross Entropyを通して更新が成立する。
3. **今回主張しないもの**: 動画・時系列の表現獲得、MoCoの正しさ、従来PEFTとの性能優劣、CIFAR-10の高精度、十分な汎化性能。

## 2. Authorityと根拠

### 2.1 確定済み（ユーザー明示）

- 髙間氏の[ProjectedViT LoRA](https://github.com/tamaki-lab/2024_10_takama_ProjectedViT/blob/e6d5f8d59a66f869954248cc4427ea7697aed4e9/models/modelings/modeling_projectedvit.py#L30-L67)を参考にする。自作A/B、`alpha/r`のスケーリング、Attentionへの加算が核。dataset別LoRAや直交射影は持ち込まない。
- Q、K、VはHydra/YAMLの`target_modules`から**それぞれ独立に切替可能**にする。
- **LoRA Aをランダム初期化、Bをゼロ初期化**する。髙間氏の原実装のA/B双方Xavierとは意図的に異なる。
- **通常の実画像データセットで確認**する。どのデータセットを使うかは別途未承認。
- 既存の`encoder-only`構成、事前学習済みBase ViT固定、Hydra/YAMLを重視する。
- 今回はSequential Loader・MoCo・Linear Probeと切り離す。

### 2.2 調査済みのコード（静的根拠）

- GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@b6385d87e2e3e82a719d8f4b686b44aa293b1135`
  - [`model/vit/vit_frame_encoder.py`](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/model/vit/vit_frame_encoder.py): `ViTModel.from_pretrained()` → 全重みをfreeze → CLS `[B,768]`を返す。分類Headなし。旧コードではpoolerを明示的には無効化していない。
  - [`model/vit/vision_transformer.py`](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/model/vit/vision_transformer.py): `ViTForImageClassification`とクラス予測・Lossを返す別経路。Base freezeは明示されていない。
  - [`dataset/cifar10.py`](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/dataset/cifar10.py): 標準の`torchvision.datasets.CIFAR10`＋通常の`DataLoader`を実装。download=True。
  - [`dataset/transforms.py`](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/dataset/transforms.py): 既存画像前処理のNormalizeはImageNet mean/stdを使用。
  - [`main.py`](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/main.py): 通常の分類モデル前提でCUDA利用をassert。今回のfrozen CLS+LoRA経路にそのまま流用するのは避ける。
  - [`setup/optimizer.py`](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/setup/optimizer.py): 旧factoryはSGD/Adamであり、AdamWを使う場合は既存factoryへ無断で副作用を入れず、独立エントリーポイント側でOptimizerを構成する案。
- [Transformers v5.17.0の`ViTAttention`](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/vit/modeling_vit.py#L192-L238): `q_proj/k_proj/v_proj`が独立した`nn.Linear`で、`ViTModel.layers[i].attention.<projection>`が対象経路。旧版の`vit.encoder.layer`を前提としない。
- 旧[Stage 4 PEFT spec](./2026-09-23-vit-lora-one-step-smoke-spec.md)は**historical evidence**。今回の自作LoRA/CIFAR-10のapproved仕様にはならない。
- 2026-10-13の[探索メモ](../../../../secretary/notes/brainstorm/2026-10-13-takama-lora-qv-kv-target-selection.md)および[2026-10-08 MTG](../meetings/2026-10-08-mtg.md)。

### 2.3 実装基準の限界

SSHサーバ上のローカル作業tree `encoder-only` はユーザーから報告された状態。**実際のbranch HEAD・dirty/untracked差分・venv・CIFAR-10の保存場所はGitHubから確認できない**。このspecで参照する`b6385d8`は、ローカルを確認する際の出発点であり、現在のローカルHEADと断定しない。GitHubの`dev`は別系列の統合変更を含む可能性があるので、便宜的に今回のコード正本へ置き換えない。

## 3. Scope / 非対象

### 3.1 新規作成・変更する対象（提案、承認待ち）

- **自作LoRAモジュール**: 元の`nn.Linear`を保持し、LoRA A/Bの差分を加える小さな`LoRALinear`。
- **LoRA付きFrame Encoder**: 現行`ViTFrameEncoder`と同じ`forward(pixel_values) -> CLS [B,768]`インターフェースを持ち、凍結ViTへQ/K/V LoRAを挿入する別クラス（仮称`ViTLoRAFrameEncoder`）。
- **Hydra設定**: 既存設定を壊さず、LoRA検証専用presetを用意する。
- **分類Headを付ける短時間学習経路**: 独立したCIFAR-10検証entrypoint。学習ロジック、正規化、ログ、短時間実行上限を明示。
- **unit/integration/smoke test**: Q/K/V組合せ、凍結、初期化、勾配、実画像入力の健全性、既存モデル非回帰。

想定ファイル例（実際の配置は作業treeに従う）:

```text
model/vit/lora_linear.py
model/vit/vit_lora_frame_encoder.py
conf/lora_standalone.yaml
scripts/smoke/smoke_vit_lora_cifar10.py
test/model/test_vit_lora_frame_encoder.py
test/model/test_lora_linear.py
```

`model/vit/`は旧baselineの配置例。現行worktreeで`model/backbones/vit/`などへ移設済みなら、実際の構造を先に確認し重複を作らない。

### 3.2 明示的に対象外

- Sequential Loader、ActivityNet/50Salads入力、timestamp、chunk aggregation。
- MoCo、InfoNCE、EMA、FIFO Queue、Linear Probe、動画表現の評価。
- PEFTへの復帰、髙間氏のorthogonal subspace / dataset-specific LoRA / MLP拡張。
- 既存`ViTb`・`main.py`の挙動変更、既存画像変換の一括置換。
- 既存の`audit_activitynet_inventory.py`、`sequential_vit_bridge.py`、`smoke_50salads_vit_bridge.py`、`smoke_activitynet_vit_clip_feature.py`、`smoke_vit_frame_encoder.py`の削除/整理（当該scriptやテストは維持）。
- 全CIFAR-10の本格的epoch学習、網羅的hyperparameter探索、multi-seed検証、performance claims。
- GPU長時間run、resume/optimizer state復元、リモートサーバ上のdirtyファイル強制上書き。

## 4. LoRAの振る舞いとインターフェース（提案）

### 4.1 数式と構造

```python
output = base_linear(x) + (alpha / rank) * lora_b(lora_a(dropout(x)))
```

- `base_linear`は`ViTModel.from_pretrained()`が作成した**既存のオブジェクトそのもの**を保持し、重み・biasを変更しない。
- LoRA A: `Linear(in_features, rank, bias=False)`、B: `Linear(rank, out_features, bias=False)`。
- A: ランダム初期化、B: 全ゼロ初期化。初期差分0で、LoRA-ONとBase出力が同一になることをeval modeの同一入力で確認する。
- `alpha/rank`でスケーリング。LoRA側のdropoutのみ（初期案0）。
- **freezeの順序**: Baseロード → Base全体の`requires_grad=False` → 対象`nn.Linear`をLoRALinear wrapperに差替 → LoRA A/Bだけtrainable → optimizer構築。注入後に全`vit.requires_grad_(False)`を再度実行してはいけない。
- 学習forwardを`torch.no_grad()`で包まない。Baseの凍結は`requires_grad=False`で行う。
- `ViTModel.layers[i].attention.q_proj/k_proj/v_proj`を走査。全12 blockについて対象が`nn.Linear`で期待shapeであることをassertし、二重注入、対象の欠損、予期しないmodule typeはfail-fast。
- 同じ`ViTModel`の重みが初期化一致比較の基準となる。独立のrandomなモデルどうしを比較しない。

### 4.2 Hydra/YAML target指定

**確定している契約**: Q/K/Vを一つずつ選ぶことも、組み合わせて選ぶこともできる。設定変更だけで切り替え、コード改変は不要。

**提案する初期config**（各値は§10の未決事項でレビュー）:

```yaml
model:
  checkpoint_id: google/vit-base-patch16-224

lora:
  enabled: true
  target_modules: [q, v]
  rank: 8
  alpha: 8
  dropout: 0.0

data:
  name: CIFAR10
  root: ./downloaded_data
  train_subset_size: 64

training:
  seed: 0
  batch_size: 16
  max_steps: 20
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0

logging:
  comet_enabled: false
```

- 許容`target_modules`: `[q]`, `[k]`, `[v]`, `[q,k]`, `[q,v]`, `[k,v]`, `[q,k,v]`（並び順は重複排除ではなくcanonical化してよい）。
- `enabled: true`で空・未知名・重複指定は例外。`rank`は正整数、`alpha>0`、`0<=dropout<1`とする。
- `enabled: false`ではLoRA moduleを注入せず、通常のCLS encoderとして動かす。分類学習時にtrainable headだけが存在する状態を許容するかは明示する。
- 各12block × 768→768、rank 8の場合、対象module数 = `12 * len(target_modules)`、LoRA trainableパラメータ数 = `147456 * len(target_modules)`（1対象147456、Q/V294912、Q/K/V442368）。モデル構造が異なる場合は数値を盲信せず事前検証で停止する。
- 層ごとに別targetを指定する機能は今回の初回scope外。rankとalphaはYAMLから変更可能。

### 4.3 Feature contractと既存ベースライン

- 入力`pixel_values: FloatTensor [B,3,224,224]`、出力`cls: FloatTensor [B,768]`。
- LoRA OFFとLoRA ONのB=0で同じ事前学習済みViTのCLS出力が一致する。正規化やdropoutの影響を避け、eval・同一入力でfloat誤差を許す`torch.testing.assert_close`。
- 既存`ViTFrameEncoder`を変更しない。必要に応じて新しいLoRA付きモデルのみ`add_pooling_layer=False`とするが、CLS契約の非回帰と事前学習済み重みの読み込みレポートを必須にする。
- 既存`vision_transformer.ViTb`は分類用モデルとして残し、こちらへ無断でLoRAを注入しない。

## 5. CIFAR-10教師あり短時間学習（提案）

### 5.1 データ・変換

- 候補datasetは`torchvision.datasets.CIFAR10`の**train split**。test splitで学習しない。
- dataset受領・ダウンロードの責任を明示。保存済みCIFAR-10とpretrained ViT cacheがあるかは実行前に確認。自動download=Trueの副作用を把握したうえで作業する。
- RGB 32×32をViT入力224×224へリサイズ。Normalizeは使用checkpointの`AutoImageProcessor`設定と**同じmean/std**とする。旧`transform_image`（ImageNet形式）を一括変更しない。
- 固定subset・fixed seedで選択した訓練画像を使用。少量の学習挙動を見る場合、前処理にランダムflip/cropを使わず、同一画像の入力が繰り返し再現されるようにする。

### 5.2 学習経路

```text
CIFAR-10 ordinary DataLoader (image, class label)
  -> checkpoint-compatible image preprocessing
  -> frozen ViT + selectable Q/K/V LoRA
  -> CLS [B,768]
  -> trainable Linear(768,10) head
  -> logits [B,10]
  -> CrossEntropyLoss
  -> backward
  -> optimizer step (LoRA A/B + classifier head only)
```

**重要**: 分類Headを学習する場合、これは`LoRA + classifier head`学習である。工学的なLoRA-only更新確認はHeadなしの独立unit testで実施し、「LoRAのみを学習した」と誤って表現しない。

### 5.3 短時間runの暫定条件

| 条件 | 推奨（未承認） | 意味 |
| --- | --- | --- |
| split | CIFAR-10 train | 正解ラベル付き実画像 |
| subset | 固定64枚 | 小規模・再現性 |
| batch | 16 | 224²画像のVRAM/CPU負荷を抑制 |
| optimizer | AdamW | 過去Stage 4と比較しやすい |
| LR, WD | 1e-3, 0.0 | 機械的確認用、性能最適化ではない |
| steps | 最大20 | 初回は短時間で終了 |
| seed | 0 | 再現可能な初期化と抽出 |
| scheduler | なし | 変動要因を減らす |
| target | [q,v] | 従来のQ/V設定との機械的比較 |

- full datasetやaccuracy目標は設定しない。
- Stage Bは小規模な実画像学習を「実際にrunする」とユーザーが許可したときに実行する。**spec承認やコード実装承認のみから長時間runや自動download許可を推定しない**。
- ログには最低限、loss、prediction shape、accuracy（診断表示のみ）、LoRA A/BおよびHeadのgradient norm、update norm、Base update norm、対象層数、trainable param数、steps、resolved config、checkpoint id、commit/dirty状況、デバイスを残す。Cometは今回はデフォルトOFFを推奨。ONにする場合は既存projectの命名・タグ規則へ合わせる。

## 6. 保存・再ロード（提案・未承認）

- 重いBase ViTを別checkpointとして再保存することは必須にしない。
- 再現性が必要なら、`LoRA A/B weights`と`classifier head weights`、`checkpoint_id`、`target_modules`、`rank`、`alpha`、`dropout`、seed、データ/前処理条件を関連付けて保存する。
- 再ロード順は「元ViTをcheckpointからロード → 同じconfigでLoRA挿入 → adapterとheadのweightロード」。保存・再ロードの前後でeval logits/CLSの数値一致を検証する。headだけロードしてLoRAを取り違えた場合は失敗扱い。
- `LoRALinear`でbaseを保持するとstate_dict keyが変わり得る（`q_proj.weight` → `q_proj.base.weight`など）ため、**raw ViT state_dictとの無変換互換性は仮定しない**。
- optimizer state・途中epoch再開・外部artifactアップロードは今回のscope外。保存方式を必須にするかは§10で確認する。

## 7. 実装順序と検証の成功条件

### Stage A: adapter単体・ViT接続

1. 正しいshapeでA/Bを生成し、初期Bは0、Base weight/biasは注入前後でbit-exactに一致。
2. 全7 target組合せを構築・forward可能であり、対象以外にLoRAを生成しない。Q/Vの期待数は24個、A/B trainable 48 tensor / 294912 parameter。選択されたprojectionが実際にwrapperへ置換されていることを監査する。
3. enabled=falseとenabled=true（初期B=0）で、同一ViT / 同一入力 / eval条件下のCLS特徴が許容誤差内で一致。入力`[B,3,224,224]` → 特徴`[B,768]`。
4. engineering-onlyのscalar lossで`backward()` → 初回Bの有限nonzero gradientが確認でき、1 optimizer stepでBの少なくとも一部が変化。初回のA勾配ゼロは**仕様どおり**。
5. 2 step以降でAにも有効なgradientが流れることを、制御された非退化入力/出力条件で検証。パラメータ更新は有限。**全tensorが同時に変わることは要求しない**。
6. Base ViTのparameterは全て`requires_grad=False`、gradientなし、step前後で完全一致。Optimizer param集合は宣言したtrainable集合と完全一致。
7. 重複target、未知target、enabled=trueの空指定、不正rank/alpha、欠落target・非Linearのエラーはfail-fast。
8. 既存`test/model/test_vit_frame_encoder.py`、CIFAR10/DataLoader/他モデル関連テストの非回帰。旧smokeが`torch.no_grad()`である点から、これをLoRA勾配試験の代用とはしない。

### Stage B: 実画像の普通の教師あり学習

1. CIFAR-10 trainの実画像と10クラスラベルを通常DataLoaderで読み込み、ViTに適切な`[B,3,224,224]`を生成。
2. CLS `[B,768]` → logits `[B,10]` → finite Cross Entropyを計算。
3. LoRA A/Bと分類HeadのみOptimizer対象、Baseは含まない。
4. 実画像で複数stepを行い、少なくとも一つのLoRA B parameterとHead parameterが変更され、Baseは完全不変。Loss/grad/updateにNaN/Infが発生しない。
5. stepごとのloss、gradient norm、パラメータ更新量、予測accuracyの**観測値**を記録。20stepの単純なtrain-loss単調減少や高いaccuracyはPASS条件にしない。
6. Stage A/Bのpass/failと実際に実行できた環境・所要範囲を明確に分ける。Stage A passだけでStage B passと言わない。
7. §6を採用する場合のみ、adapter/head保存・再読込後に数値一致を追加で必須とする。

## 8. 非回帰と変更境界

- 旧`ViTFrameEncoder`（frozen CLS）、`ViTb`、通常の`configure_model`、Hydra既存entrypointを維持。既存の公開CLI、datasetデフォルト、通常学習条件を勝手に切り替えない。
- 以前確認したlegacyの監査・bridge・smoke 5ファイルは今回編集/削除しない。古いbranch名やcommitを固定したsmokeは`encoder-only`で必ず走るとは限らないので、テストの期待条件と単体機能を分けて報告する。
- 既存のDataLoader factoryのCIFAR10経路を活用する場合も、共通画像transformsを全面置換しない。ViT専用の前処理は隔離する。
- PEFT依存を必須にしない。現依存が提供するtorch/transformers/torchvision/hydraを使う。実行前に`transformers`とViTのmodule pathを検査する。
- 既存testsの不合格があれば、変更前の再現可否を確かめ、今回導入したregressionと区別する。

## 9. 必要な再現性・実行境界

- 実装前preflight: ローカルrepo URL、branch、HEAD SHA、`git status --porcelain`（untracked含む）、`python -c`等でtorch/transformers/torchvision/hydraバージョン、Hugging Face ViTのtarget count/shapeを確認する。
- GitHub上のreference `b6385d8`を「現ローカル実装を実行確認した結果」と混同しない。worktreeが異なれば設計のパスと差分を先に照合する。
- CIFAR-10実体・ダウンロード可否、ViT cache、メモリを確認する。画像subsetとseedは実行ログへ残す。
- Stage Aのunit/mock/短いsmokeと、Stage Bの実画像学習runの承認は別。データdownload、GPUを占有する訓練、本格epoch実験、コミット・pushも個別の操作として扱う。

## 10. Ambiguity Gate / 採用待ち事項

以下は**ユーザーがまだ採用していない推奨案**。今は`draft`として記録し、approved移行前に決定する。

| ID | 分類 | 未決事項 / 選択肢 | 推奨 | 影響 |
| --- | --- | --- | --- | --- |
| B1 | blocking | 注入方法: HF ViT Attention内部クラス継承 or 元のLinearを保持するLoRALinear wrapper | **wrapper方式**。HF v5.17.0の構造から実現可能性高いが実機未検証 | 変更範囲・将来互換性 |
| B2 | blocking | 実画像検証用の分類Head: LoRAのみtrainable or LoRA+Head trainable | **LoRA+Head**、LoRA-only unit別実施 | 学習対象、Lossの意味 |
| B3 | blocking | 画像dataset: CIFAR-10 or 別の通常画像dataset | **CIFAR-10**（既存loaderあり） | preprocessing、出力クラス、必要な実データ |
| B4 | blocking | 今回の実装scope: unit/1-2stepのみ or 実画像短時間学習まで | **Stage A＋Stage B**。本格性能評価は別 | 必須成果物・実行境界 |
| B5 | blocking | 初回のQ/K/V初期targetsと比較範囲: Q/V開始か、Q/K/V組合せの実学習比較まで含むか | **Q/V開始**、全7組はunit注入検証、K/V等の学習比較は後続 | 実験条件・計算費用 |
| B6 | blocking | 保存/再読込: 技術検証のみか、LoRA+Head checkpoint roundtripも必須か | **最小のadapter+head保存／読込テスト** | artifactと受入条件 |
| B7 | blocking | 実画像pilotの規模/optimizerを暫定値で承認するか | **64枚・batch16・20steps・AdamW 1e-3・WD0・seed0** | 学習再現性と計算負荷 |
| E1 | 実装時preflight | 現ローカル`encoder-only` branch/HEAD/dirty、ViT module path | 実装前にread-only確認し、異なれば移行前に止める | コードをどこへ追加するか |
| N1 | non-blocking | helper/関数名、YAMLの細かなファイル位置 | 現ローカル構造・命名に合わせる（契約を守る） | 研究結果に直接影響しない |
| N2 | non-blocking | 統計の表示形式、短時間テストの分割 | 既存pytest等に合わせる | 外部研究契約は変えない |

**draft readiness**: 目的/確定事項/scope外/静的可否/検証指標は整理できた。B1–B7は研究・実装契約を変えるため、ユーザーの採用または修正が必要。E1はGitHubから現在ローカルを読めない環境上のpreflight事項。承認が得られるまで`status: approved`にしない。

## 11. Spec Gate と実装handoff

- 現在の状態: `draft`。**この文書による実装許可なし、学習run許可なし。**
- 承認時にはB1–B7の選択とpreflight手順を確定し、承認記録とstatusを更新する。
- approved後のhandoff内容:
  - 基準: ユーザーの実ローカルbranch/HEADを確認した値
  - 差分: 自作LoRALinear、LoRA付きViT、Hydra preset、通常CIFAR10短時間学習entrypoint、unit/smoke
  - 維持: 既存ViTFrameEncoder/ViTb/従来の5legacy script、Sequential Loader/MoCoを触らない
  - 証拠: Q/K/V全7組のmodule一致、Base不変、A/B初期化、勾配・更新、実画像CEのfiniteと短時間step
  - 別権限: CIFAR-10 download、実画像run、GPU占有、本格run、コミット/push
- 保存仕様のapproved化は研究コード変更ではない。ローカルcloneへの`git pull`など同期も別作業。

