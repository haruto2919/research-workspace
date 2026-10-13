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
