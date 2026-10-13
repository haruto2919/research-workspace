---
date: 2026-10-13
project: sequential-video-lora-analysis
source_todo: "LoRAをSequential LoaderとMoCoから切り離した最小構成で動かし、forward、対象Q/V、勾配、更新、Base固定を確認する"
topic: takama-lora-qv-kv-target-selection
status: exploratory
tags: [brainstorm, research, lora, vit, target-modules]
---

# 髙間氏のLoRA方式を参考にした単体検証とQ/V・K/Vの選択

## 出発点・現在の文脈

- 2026-10-08 MTG決定: Sequential Loader、LoRA、MoCoを同時に扱わず、LoRA単体から部品検証をやり直す。
- 現在ユーザーが切り替えたローカル作業状態は `encoder-only` @ `b6385d8`、Hydra/YAML維持、LoRA/MoCo/Linear Probeを外した状態との報告。ローカルdirty stateはGitHubから未確認。
- ユーザーの**今回の明示採用事項**: LoRAの実装は `tamaki-lab/2024_10_takama_ProjectedViT` の髙間氏の方法を参考にする。前回のPEFT優先推奨より現在のユーザー指示を優先する。
- ユーザーの**未決事項**: LoRAをK/Vへ入れるほうが一般的ではないか。なぜQ/Vなのか説明が必要。

## コードから確認できた事項

- [参考実装のLoRA本体](https://github.com/tamaki-lab/2024_10_takama_ProjectedViT/blob/e6d5f8d59a66f869954248cc4427ea7697aed4e9/models/modelings/modeling_projectedvit.py#L30-L67): `nn.Module`、`lora_a: Linear(d_in,r)`、`lora_b: Linear(r,d_out)`、`alpha/r`、dropout。補正は元の線形層出力に加算。
- [Attentionへの組み込み](https://github.com/tamaki-lab/2024_10_takama_ProjectedViT/blob/e6d5f8d59a66f869954248cc4427ea7697aed4e9/models/modelings/modeling_projectedvit.py#L90): Q/K/VすべてにLoRAを作れる。Q/Vに固定された実装ではなく有効化と層選択を設定で制御する。
- [apply_lora](https://github.com/tamaki-lab/2024_10_takama_ProjectedViT/blob/e6d5f8d59a66f869954248cc4427ea7697aed4e9/models/model_util.py#L511-L523): share と現在データセット別の補正を加算。
- 参考実装は任意のorthogonal subspace投影、Attention出力およびMLPへのLoRA、dataset別LoRAも含む。今回のLoRA単体検証にそれらを必須とはしない。
- 参考実装 `_init_weights` は A/B両方をXavier初期化する。標準的なBゼロ初期化と、LoRA追加直後のベースライン出力が一致するかが異なるため、採用範囲の未決事項。
- 旧Stage 4 spec `.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-23-vit-lora-one-step-smoke-spec.md` によると、研究サーバの Transformers 5.17.0 では `q_proj/k_proj/v_proj/o_proj`、各12個を観測。髙間氏の参考実装中の旧Hugging Face `query/key/value` とは名称が異なる。実装前のlive環境検証が必要。
- 現エンコーダ [ViTFrameEncoder](https://github.com/tamaki-lab/2026_04_ishikawa_simple-MeMViT/blob/b6385d8/model/vit/vit_frame_encoder.py) は frozen `ViTModel` からCLS `[B,768]` を返す。

## 原論文と公式文書から確認できた事項

- [LoRA原論文 §4.2, §7.1 Table 5](https://arxiv.org/html/2106.09685): AttentionのQ/K/V/OはいずれもLoRAの対象にできる。GPT-3の同等のパラメータ予算で、Q/VはWikiSQL 73.7、MultiNLI 91.3、Q/Kはそれぞれ71.4、91.3。**Table 5にはK/Vの直接比較がない**ので「Q/VはK/Vより性能が高い」とは言えない。言語モデルの結果を動画ViTへ一般化できない。
- [Hugging Face PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora): デフォルトの一般的LoRA方式はQuery/Valueへの追加。
- [Hugging Face画像分類ガイド](https://huggingface.co/docs/peft/main/task_guides/image_classification_lora): ViTの `query` と `value` にLoRAを入れる例。これはライブラリ利用の裏付けであって今回PEFT採用指示ではない。
- Q/VはQuery側からattentionの相対的スコアを変え、V側で集約する内容も変える。K/VはKey側からattentionのスコアを変え、V側も変える。どちらでもAttentionの分布は変更可能。MoCoのQuery encoder/Key encoderとは別の概念。

## 合意 / 有力候補 / 保留

- **採用**: 髙間氏方式を参考に、単独のLoRA A/Bモジュールと元のViT線形層の出力への加算を基本とする。PEFTを優先するという以前の提案は今回の優先候補から外す。研究コードの変更・spec確定はまだ行わない。
- **有力候補（未採用）**: 最初の単体検証はQ/Vだけにrank8 alpha8のLoRAを入れる。理由はLoRA原論文・実装慣行・旧実験との比較可能性。
- **比較候補**: 追加検証としてK/Vも同rank・同対象層数・同data/loss/optimizer/seedで比較する。Q/VとK/VはViT-Baseではいずれも12block×2対象=24 LoRA modules、rank8なら294,912 weights。
- **保留**: 参考実装のA/B双方Xavier初期化をそのまま採用するか、Bゼロ初期化のベースライン一致を優先するか。
- **保留**: 元コードのViT内部クラス全面継承・差し替えか、現行のViTFrameEncoderに最小限のLoRA付き線形層を挿入するか。HF内部名の互換性を確認して決める。
- **除外**: 現段階ではSequential Loader、MoCo、Linear Probe、dataset-specific LoRA、直交射影、長時間run。

## 反例・注意事項

- Q/Vが一般的でも、動画の動的特徴を学習する最適対象とは限らない。
- K/Vが劣ると結論できる直接の原論文データはない。
- 単に勾配と重みの更新が成立しても、LoRAが時間情報を獲得した証拠にはならない。
- 参考実装でQ/K/V全部の加算が可能であることと、今回Q/Vだけの検証が適切かという研究判断は別。
- Q/VとK/Vの比較で差が出ても、初期化やデータ・seed・lossが異なれば適用対象の影響を特定できない。

## 次アクション候補（未追加TODO）

1. Q/Vを初回のLoRA-only技術検証対象とするかユーザーが決定。K/Vは比較候補として保留可能。
2. 髙間氏から参考にする範囲（LoRA A/B実装と加算・継承方式）と初期化の扱いを決める。
3. その後にread-only preflightを含む実装可能specへ引き継ぐ。Hydra/YAMLを維持し、単体forward/backward/update/base不変を成功条件に置く。

## 関連情報

- [2026-10-08 MTG](https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/meetings/2026-10-08-mtg.md)
- [2026-10-13 参考実装詳細解析](https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/code-analysis/2026-10-13-projectedvit-lora-encoder-only-explain.md)
- [LoRA 原論文](https://arxiv.org/html/2106.09685)


---

## 2026-10-13 14:42 JST 追記: Q/K/Vの独立切替要件とspec前論点

### 今回のユーザー確定事項

- 髙間氏のProjectedViTの自作LoRA（A/Bの低ランク補正を既存Attentionへ加算）を参考にする。
- **Q、K、VはHy​​dra/YAMLの設定だけで簡単に独立して切り替えられること**。特定のQ/V組み合わせにハードコードしない。
- 初回はSequential Loader、MoCo、Linear Probeなし。既存encoder-only構成とHydra/YAML管理を維持する。

### 最小案（候補・未採用）

```yaml
lora:
  enabled: true
  target_modules: [q, v]  # [q], [k], [v], [q,k], [q,v], [k,v], [q,k,v]
  rank: 8
  alpha: 8
  dropout: 0.0
```

- 各targetが現在のモデルに実在するmodule（以前のTransformers 5.17.0では `q_proj/k_proj/v_proj` と報告）へ**完全一致**で解決されるか実装時に検証し、不明な指定、重複、空リスト（enabled時）、未検出の層は例外とする。
- 12 block全てへの一律適用を初回scopeとする。blockごとのカスタムtarget設定は不要。
- baseは凍結し、LoRA A/Bだけをoptimizer対象にする。`enabled: false` ではLoRAなしのbaseline経路を維持。
- Q/V・K/Vなら24 target × rank 8 × (768+768) = 294,912 trainable weights、Q/K/Vなら36 target = 442,368。実際のmodule数・shapeはlive code確認が必要。
- LoRA moduleを別実装し、元の線形層の出力への加算を最小のwrapper/注入経路で作る方法が第一候補。参照のHugging Face内部クラス全面継承は比較候補、既存の `ViTFrameEncoder` のbaseline APIは維持したい。
- 技術検証の最小scopes: synthetic tensorのforward [B,768] → engineering-only loss → backward → 1-2 optimizer steps → base不変、選択LoRAだけ勾配・更新、非選択層にLoRAなし。LoRA OFF equivalence、組合せ切替、異常設定のtestも追加。
- 初期化: 参照先はA/BともXavier。一方でB=0なら初期差分0、初回でA勾配0が正常。初心者に説明しやすくbaseline一致が可能な後者を第一候補とし、参照との差異をspecに明記する。
- 初回の実験は「LoRAの機械的な学習成立」の確認と、「通常画像データセットを使った意味のある分類学習」を分ける。後者を同specに含めるかは未決。
- 性能確認や時間情報獲得は対象外。初期入力として人工tensorを使い、必要なら実画像1件のforwardのみを補足する。

### spec前に決めるblocking候補

1. **初期化**: 髙間氏のA/B Xavierをそのまま採用するか、A random/B zeroでbaseline出力同一を優先するか。
2. **検証スコープ**: unit + 1/2-step engineering-only smokeまでか、実画像datasetによる数step/短時間の supervised learningも含めるか。後者ならdataset/loss/headが追加決定事項。
3. **組み込み方法**: ViT内部の継承/置換をできるだけ原実装に合わせるか、加算とLoRA定義は踏襲しつつ最小wrapper注入にするか。現行実装・Transformersの互換性を再確認して選択。
4. baseline repo・branch・commit/dirty state: ローカル `encoder-only` はGitHub上で直接確認できない。GitHub上の `b6385d8` は照会できるが、ローカル差分は不明。

### non-blocking候補（提案初期値）

- rank=8、alpha=8、dropout=0、LoRA biasなし。
- default targets=[q,v] は初回基準（ユーザーはQ/K/V切替を確定したがdefault Q/V自体は推奨候補）。
- optimizer=AdamW（LR 1e-3、WD 0）、fixed seed=0、CPU上の最小step、schedulerなし。
- 既存のencoder-only baselineテスト・Hydra設定の非回帰を維持。

### spec引き継ぎ範囲

LoRA-onlyのコード導入、configでの対象切替、凍結・勾配・更新・等価性の検証。変更の許可は未取得であり、本記録から仕様承認や実装許可を推定しない。


---

## 2026-10-13 追記: wrapper方式の実現可能性再調査、初期化採用、実画像検証採用

### 今回のユーザー指示とAuthority

1. ① `ViT` のQ/K/V線形層をLoRA付きwrapperに置換する案Bが期待どおり実装可能かを疑問視し、具体的な再調査を依頼。**この方式自体はまだユーザー未採用**。
2. ② **採用**: LoRA Aはランダム初期化、Bはゼロ初期化（ベース出力との初期一致）。
3. ⑤ **採用**: ダミー画像だけでなく、通常の実画像データセットによる学習の動作を確認する（初回はSequential Loaderを使わない）。

### 実現可能性の静的検証と技術的制約

- 実装対象の根拠: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@b6385d8` の `model/vit/vit_frame_encoder.py`: `ViTModel.from_pretrained("google/vit-base-patch16-224")`、`self.vit.requires_grad_(False)`、`last_hidden_state[:,0,:]`。このコードではまだLoRAなし。ローカルencoder-onlyの未コミット差分はGitHubから確認できない。
- 実際のTransformers [v5.17.0 ViTAttention](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/vit/modeling_vit.py#L192-L238) は `self.q_proj/k_proj/v_proj = nn.Linear(...)` とそれらの `forward(hidden_states)` を持つ。したがって、**同じ入出力形状の`nn.Module`で差し替えるのは構造上成立する**。元のAttention全体を再実装する必要はない（ただし未実行）。
- 推奨構造: `ViTModel.from_pretrained` → 既存pretrained weightをロード → base全体をfreeze → 12個のViTAttentionから指定targetの `nn.Linear` を取得 → `LoRALinear(base_linear, A, B, scaling)` を `setattr` で注入 → 注入完了後のtrainable LoRAでoptimizerを作成。 `LoRALinear.forward(x) = base_linear(x) + (alpha/r) * B(A(dropout(x)))`。元の`base_linear`オブジェクトとbiasを保持し、予期しない形状や非Linear targetは即エラー。
- `self.vit.requires_grad_(False)` の**適用順**に注意。LoRA注入後に全体へ再適用するとA/Bも凍結される。凍結されたbaseのパラメータでも、`torch.no_grad()`でForward全体を包むとLoRAのgradientも遮断するため、学習時は避ける。
- optimizerは注入後に作る。元のViT `state_dict` の `q_proj.weight` が `q_proj.base.weight` 等へ変わるので、checkpointは明示した保存・復元方式を設計する。既存の重みは**元モデルで読み込んだ後に**注入。 `LoRA off` / `B=0` 時のbaseline出力一致、全7 target組合せのinject + backward + update + checkpoint roundtripをテストする。初期の比較では同一checkpoint/入力・eval・適切な数値許容差。
- 参考髙間氏 [LoRA実装](https://github.com/tamaki-lab/2024_10_takama_ProjectedViT/blob/e6d5f8d59a66f869954248cc4427ea7697aed4e9/models/modelings/modeling_projectedvit.py#L30-L67) は独立A/Bと出力加算を実装しているが、A/B双方Xavier。今回の**ユーザー採用**でBゼロへ意図的に変更する。
- リスク: Hugging Face内部module命名/attention backendのバージョン差、未検証のsave/load互換性、train/evalやdropoutによる等価性差、重複注入、パラメータ登録、手作りwrapperのテスト負担。いずれも**静的に実装不可能な障害ではない**が、実環境でのend-to-end検証は未実施。
- `ViTFrameEncoder` の既存no-grad CPU smokeは学習検証には流用できない（新しい学習ルートを設計する）。

### 実画像Datasetの具体的候補・条件

- 同一研究コード `b6385d8` では `dataset/cifar10.py`（Torchvision CIFAR10 train/test）、`dataset/dataloader_factory.py`（CIFAR10通常DataLoader）、`conf/dataset/cifar10.yaml`（ルート `./downloaded_data`）が実装済み。Sequential Loaderを使わない**初回の実画像dataset候補にCIFAR-10が最も自然**。 `download=True` なので初回ダウンロードにネットワーク接続が必要。ダウンロード済みなら不要。
- 既存画像変換 `dataset/transforms.py` は224 cropとImageNet mean `(0.485,0.456,0.406)`、std `(0.229,0.224,0.225)` を使用。**注意**: 既存checkpoint [google/vit-base-patch16-224 preprocessor_config](https://huggingface.co/google/vit-base-patch16-224/blob/main/preprocessor_config.json) はmean/stdどちらも `[0.5,0.5,0.5]`。ViTを用いた単体検証経路ではそのcheckpoint専用normalizeか`AutoImageProcessor`に合わせる。元の他モデルに使うtransformを不用意に変更しない。
- `ViTFrameEncoder` はCLS `[B,768]`を返すだけで、10クラス分類を学習するには `Linear(768,10)`のheadとCrossEntropyLossが必要。通常の教師ありCIFAR10分類ではtrainable **LoRA + classifier head** が推奨候補。Base frozen、headとLoRAの勾配と更新は別々に記録する。「LoRAのみ更新」の技術検証と「LoRA+headで教師あり分類」を明確に分離する。
- LoRA単独更新に固執してランダム固定headでCEすると学習信号は通るが性能解釈が難しい。標準のCIFAR10実画像ラベルを使い学習できることまで要求するならclassifier headの学習可否はspec化前に明示して決める。
- 初回スコープ候補: 224x224にpreprocessしたCIFAR10の少量画像 → ViT frozen + selectable Q/K/V LoRA → CLS → 10-class head → CE → small number of optimizer steps。まず少量のbatchでgradient/parameter checks、同一少量サンプルへのoverfitを追加のsanity testとして検討。短いrunでaccuracy向上は保証されず、モデルが時間情報を得た証拠でもない。
- 既存 `ViTb` は `ViTForImageClassification` を利用し独自headを内包するが、そちらを全面再利用してBase trainableになるような変更を加えない。今回のCLS encoder路線に沿って小さく分離することが有力。

### さらにspec化前に必要な判断

- (A) wrapper方式は静的に成立する見込み。**ユーザーが採用するかは未決。**
- (B) **確定: Bをゼロ初期化する。**
- (C) **確定: 実画像の通常datasetで学習挙動を見る。** 第一候補CIFAR10（過去の既存loaderがある）。
- (D) **新規重要論点: supervised分類時にheadを追加しLoRA+headをtrainableにするか。** LoRA-onlyの機械的確認は別テストにする方針を推奨。
- (E) データ前処理のcheckpoint一致、学習対象・class数・短時間runの成功条件、checkpoint roundtripの検証範囲をspecで確定。

### 未確認・境界

- 研究サーバのローカル `encoder-only` branch/dirty stateやdataset実ファイル、モデルcache、GPU状態はGitHubから直接確認していない。
- wrapperの実動作や実画像学習はまだ実行していない。
- 本記録はbrainstormであり、spec/実装/runの承認を意味しない。


### 追加精査：Transformers v5.17.0 の実際のネスト（重要）

[Transformers v5.17.0 `ViTModel` の実装](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/vit/modeling_vit.py#L335-L352) では、blockの保持先は **`vit.layers`**（`nn.ModuleList([ViTLayer(...)])`）である。旧版等の `vit.encoder.layer` を前提にした注入コードはこのバージョンでは不適合となる。各blockのAttentionは `block.attention`、投影層は `block.attention.q_proj/k_proj/v_proj`。したがって概念的には `for block in vit.layers: setattr(block.attention, "q_proj", LoRALinear(block.attention.q_proj,...))` のように置換する。今回の再調査はこのネスト差異を直接確認できたことが重要。

上記ネストは静的に確認したもの。実環境で `named_modules()` / `isinstance(nn.Linear)` / 12 block / 対象形状をassertし、差異があればFail fastとする。旧版の `vit.encoder.layer` に関する先行記録は**v5.17.0の現在構造として使わない**。


---

## 2026-10-13 追記：新しいLoRA単体検証specの前に決める事項

### 今回の問い

ユーザーはLoRA単体検証用のspecを作成する前に、残る判断項目の整理を依頼した。**spec作成自体はまだ依頼されていない**ため、探索メモとして追記する。過去のStage 4（PEFT＋Sequential Loader）specは新しい仕様として流用しない。

### 確定済み（Authority: 現在のユーザー指示）

- 髙間氏のProjectedViTの独立LoRA A/B補正と既存線形射影出力への加算を参考にする。
- Q/K/VはHydra/YAMLで独立に切り替え可能。ハードコードせず、許可対象を明示し、対象層検出数を検証する。
- Aをランダム、Bをゼロ初期化。
- 通常の**実画像データセット**で学習確認する。CIFAR-10という個別データセットの採用はまだ未確定。
- Base ViTを凍結。Sequential Loader、MoCo、Linear Probeを初回範囲から除外。既存Hydra/YAMLとencoder-onlyの非回帰を保つ。
- `encoder-only`はユーザーが報告したローカルの作業ブランチ。GitHubの`tamaki-lab/2026_09_ishikawa_sequential-video-lora@b6385d8`を静的確認できたが、ローカルdirty差分・worktreeは未確認。

### blockingな判断項目（spec approvedより前に確定が必要）

1. **注入方式**: ①元のHF ViT内部クラスを継承しQ/K/V出力に加算する方式 vs ②既存`nn.Linear`オブジェクトを含む`LoRALinear`wrapperでQ/K/V投影層を差し替える方式。第一候補②。HF Transformers v5.17.0は `ViTModel.layers[i].attention.q_proj/k_proj/v_proj` が `nn.Linear`なので実現可能性は高いが実行未検証。元モデルをロードしてからfreeze、adapter挿入後にoptimizerを作ること、重複注入禁止、fail-fastを必須とする。
2. **実画像学習時のhead**: CLS `[B,768]`に10クラス等の線形headを足し`LoRA+Head`をtrainableにするか、LoRAのみをtrainableにするか。実画像の分類学習成立まで検証するなら前者を推奨。ただし「LoRAだけが更新された」と誤報しない。unit smokeではLoRA-onlyの更新を別に検査する。
3. **dataset**: CIFAR-10を第一候補とする（既存通常`CIFAR10` DataLoader＋`conf/dataset/cifar10.yaml`あり）。trainの少量サンプルによる短時間学習、testをtrainには混ぜない。dataset存在と`download=True`のネットワーク境界を事前確認。
4. **検証レベル**: (A) pure gradient/1-2step unit、(B) CIFAR-10実画像の少量batchを用いたCE数十step sanity、(C) 本格的なepoch/validation accuracy。推奨は(A)+(B)、(C)を対象外とする。A/Bの意義と証拠レベルを混同しない。
5. **Q/K/V対象とhyperparams**: 全12blockに一律適用、default `[q,v]` / rank8 / alpha8 / dropout0、7つの非空組合せで一致検証、未知名・空指定・重複を拒否する設計を推奨。個別blockの指定は対象外。可変rank/alphaの許容値とcheckpoint識別も固定。
6. **保存/復元**: 分類head込み学習をするならLoRA-only重みとhead重みの保存が必要。元のpretrained baseは再ロード、config/seed/checkpointID/target_modulesを記録し、reloaded modelのeval出力が数値的に再現することを推奨。optimizer/resume/full pipelineは対象外でよい。
7. **baselineと成功基準**: pretrained baseはstep前後でexact weight一致・Baseへの勾配なし。B=0なら初期LoRA ON/OFFの出力一致、LoRA B初回gradient・更新、2step以降でAにも勾配が届くこと。全QKV設定切替、CLS `[B,768]`、LoRA+headのCEがfinite、限られた固定batchで学習信号が観測できる。学習精度や時間情報の獲得は主張しない。
8. **環境・CLI/Config/outputs**: live branch/commit/dirty状態、Transformers version、image normalization、seed、batch、steps、optimizer、出力場所、ログとComet利用境界を確定。現在の`main.py`は`configure_model`が`ViTb`（分類モデル）を選び、`assert torch.cuda.is_available()`を持ち、`model.parameters()`でoptimizerを作る。FrozenCLS encoderにはそのまま再利用しにくいので、既存経路に影響しない独立の短時間train/smoke entrypointを第一候補とする。

### コード確認済みの重要点

- [ViTFrameEncoder](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/model/vit/vit_frame_encoder.py): CLS特徴専用、`ViTModel`、`requires_grad_(False)`、headなし。
- [ViTb](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/model/vit/vision_transformer.py): `ViTForImageClassification`、logits＋optional loss、base freezeの明示なし。
- [CIFAR10 loader](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/dataset/cifar10.py): 標準`torchvision.datasets.CIFAR10`＋通常DataLoader、train shuffleあり。
- [Dataset factory](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/dataset/dataloader_factory.py): CIFAR10/ImageFolder/VideoFolder/ZeroImagesから選択、Sequential Loaderとは別。
- [transform_image](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/dataset/transforms.py): ImageNet mean/std=0.485/0.456/0.406と0.229/0.224/0.225。ViT `google/vit-base-patch16-224` checkpointが指定する正規化との一致を別途検証し、既存ResNet用transformを壊さない。固定batch overfit診断ならrandom crop/flipを一時的に使わず同じ入力を再現する。
- [main.py](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/main.py): 既存分類訓練ルートはGPU必須assertがあり、frame-encoderの出力を直接消費する想定ではない。
- [HF ViT v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/vit/modeling_vit.py#L192-L240): 対象投影が個別`nn.Linear`。Blockネストは`vit.layers`、`vit.encoder.layer`ではない。

### 推奨する最小scope（未採用）

Phase I: adapter module + Q/K/V config switch + unit (shape, initialization equality, freeze, gradients/update)。
Phase II: 実CIFAR-10 train small subset + ViT correct preprocessing + linear head + CE + limited steps + runtime metrics。
Phase III（後続・今回対象外）: 比較実験/本格的性能評価、Sequential Loader/MoCo/Linear Probeとの統合。

初期パラメータ提案: `rank=8`, `alpha=8`, `dropout=0`, `targets=[q,v]`, `AdamW lr=1e-3 wd=0`, `seed=0`, `batch_size=16`, `max_steps=20`（本格実験ではない）、schedulerなし。指定値は技術検証用の例であり、性能最適値として扱わない。

### 保留/反例

- 分類headだけがlossを減らしてもLoRAの寄与を証明しない。必要ならhead-only凍結LoRA対照条件を後段に置く。
- 初回Bゼロ初期化でAの勾配がゼロでも仕様として正常。1 stepで全LoRA tensorの更新は要求しない。
- 実画像学習のvalidation accuracyがすぐ高くなることはLoRAの技術的成功条件ではない。
- ここに列挙した未決事項はユーザーへの推奨であり、採用・spec作成・実装・実験runを承認したものではない。


---

## 2026-10-13 spec引き継ぎ先（draft）

この探索記録を元に、以下へ**実装specドラフト**を作成した。未決事項B1–B7は同ドラフト§10のAmbiguity Gateに集約し、現時点では実装承認されていない。

- [自作ViT LoRA単体検証・CIFAR-10短時間学習 spec（draft）](https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/specs/2026-10-13-standalone-vit-lora-cifar10-verification-spec.md)
- 状態: `draft`。ユーザー採用のQ/K/V切替・Bゼロ初期化・実画像による検証方針を仕様へ転記。
