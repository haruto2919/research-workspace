---
project: sequential-video-lora-analysis
date: 2026-10-13
analysis_mode: explain
detail: full
level: junior
source_repository: https://github.com/tamaki-lab/2024_10_takama_ProjectedViT
source_commit: e6d5f8d59a66f869954248cc4427ea7697aed4e9
source_dirty: unknown
targets:
  - models/modelings/modeling_projectedvit.py
  - models/model_util.py
  - models/vit_base_models.py
  - models/model_pl.py
  - config.py
related_spec: null
current_encoder_repository: https://github.com/tamaki-lab/2026_04_ishikawa_simple-MeMViT
current_encoder_revision: b6385d8
---

# Code Deep Dive: ProjectedViT の LoRA と encoder-only 単体確認

## 対象と解析の範囲

依頼は、髙間氏の ProjectedViT の LoRA 実装を初学者向けに説明し、Sequential Loader / MoCo / Linear Probe を使わずに、現在の encoder-only に LoRA を適用する単体検証の前提を整理すること。**本書はコードの静的解析であり、spec / 実装 / 動作確認の完了を意味しない。**

関連の意思決定は `meetings/2026-10-08-mtg.md`。「LoRA と MoCo を単体検証した後に統合する」を採用。現行 README は 2026-10-08 更新で、2026-10-13 のローカル `encoder-only` 切替は GitHub では追認できない。

## Overview: What / Why

LoRA は凍結した元の線形層 `Wx+b` に、低ランクの学習可能な補正 `(alpha/r) B A x` を加える。元の `W` を直接書き換えないことで、学習パラメータ数・optimizer state を減らす。ただし元のViTの順伝播計算が不要になるわけではない。

参照先はHugging FaceのViT内部クラスを継承したカスタム実装で、Q/K/V に加えて Attention 出力射影・MLPの2層にも任意にLoRAを足せる。multi-dataset `share` / domain-specific LoRA や部分空間射影機能もある。

## Key Components and Data Flow

- `models/modelings/modeling_projectedvit.py:30-67` `LoRA`: `lora_a = Linear(in_features,r,bias=False)` → `lora_b = Linear(r,out_features,bias=False)`、`scaling=alpha/r`。forward は dropout→A→B→scale→（必要時にspace/null-space射影）。
- `models/modelings/modeling_projectedvit.py:70-87` `LoRAOrth`: layer / parameter名から `U_space` / `U_null_space` を取る。標準LoRAの必須機能ではない。
- `models/modelings/modeling_projectedvit.py:90-256` `MyViTSelfAttention`: `self.query/key/value(hidden_states)` の各出力に `apply_lora(...)` を加え、attentionの通常計算に渡す。`MODEL.MTLORA.ENABLED`、`QKV_ENABLED`、block別の `PROJECTEDVIT.QKV` 条件に従う。
- `models/modelings/modeling_projectedvit.py:257-489`: attention output dense、MLP intermediate dense、MLP output denseにLoRAを挿入。
- `models/model_util.py:511-523` `apply_lora`: `share` を常に計算し、辞書に現在のデータセット名があればdomain-specific結果も足す。これとMoCoのQuery/Keyエンコーダは別概念。
- `models/modelings/modeling_projectedvit.py:491-634`: ViTLayer→Encoder→Model→分類モデルを置換する継承構造。分類時はCLS `sequence_output[:,0,:]` をdataset別headへ渡す。`_init_weights` では LoRA A/B 両方をXavier初期化しており、Bゼロ初期化の標準的なLoRAと初期挙動が異なる。
- `models/vit_base_models.py`: `ProjectedViTForImageClassification.from_pretrained("google/vit-base-patch16-224", train_conf=..., null_spaces_dict=...)` を使う。参照先も手作りのViT重みを一から学習しているわけではない。
- `models/model_pl.py:44-47`: `MTLORA.TRAIN_LAYER` が `None` でない場合に `freeze_model_params` を呼ぶ。`config.py` に `FREEZE_PRETRAINED=True` の設定はあるが、この情報だけで実際のBase全層凍結は保証できない。

## Concepts and Design Decisions

**数式**: `y = Wx+b + (alpha/r) B A dropout(x)`。一般に `A∈R^{r×d_in}`、`B∈R^{d_out×r}`。

`d_in=d_out=768, r=8` では元の線形層のweight 589,824個に対しLoRA A+Bは12,288個（約2.083%）。ViT-Baseの12 block × Q/V 2箇所なら、合計24箇所、294,912個のLoRA weight（A/Bの48 parameter tensors）。実際の値は対象モデルと設定の一致を検証する必要がある。

**注意1**: AttentionのQ/K/Vは「何を注目対象とするか」「照合用のkey」「どんな内容を集めるか」に対応する線形射影であり、MoCoのQuery/Keyエンコーダとは別物。

**注意2**: `TA` / `TS` はコード上では各blockのshared / dataset-specific挿入選択を制御し、`TA_all_space` は空間制約なしのbranchに入る。コードとconfigを読むだけで論文上の意味を断定しない。

**注意3**: `post_init` 内でLoRAのA/B両方に Xavierをかけるため、追加直後から `BA` が一般に0ではない。基準となるViT出力の維持が必要なら、Bゼロ初期化の方式と比較する。

## 現在の encoder-only コードとの対応

GitHub上で取得できる `tamaki-lab/2026_04_ishikawa_simple-MeMViT@b6385d8` の `model/vit/vit_frame_encoder.py` は `ViTModel.from_pretrained(...)` と `self.vit.requires_grad_(False)`、 `last_hidden_state[:,0,:]` を用いる。現在のLoRAを入れない時点では画像 `[B,3,224,224]` をCLS `[B,768]` へ変換する。クラス分類 head はこのモジュールにはない。対応する `conf/config.yaml` はHydra管理だが defaults は `dataset: cifar10`、`model: resnet18` なのでLoRA単体検証に使うモデル指定は区別が必要。ローカルの `encoder-only` branch と dirty state はGitHubでは確認できない。

## Alternatives and Trade-offs

(1) **PEFTでQ/VへLoRAを挿入**: 既存のHF ViTと親和性が高く、modelの内部クラスを全面継承しなくてよい。最初の「LoRA単体で学習できるか」を確かめる候補。
(2) **参照先に似た自作LoRAをQ/Vへ追加**: 行列の計算を学習・調査しやすいが、HF内部バージョンと継承構造に依存し、freeze、initialization、保存、gradientを自前で管理する必要がある。
(3) **ProjectedViTを丸ごと移植**: domain-specific / projection / task-switch など余分な複雑性が高く、今回の単体検証には不適。

いずれも未採用の候補。現在の仕様を確定したことにはならない。

## LoRA単体の確認案（未承認・spec未作成）

Sequential Loader、MoCo、Linear Probeなしで、通常の画像 `[B,3,224,224]` → ViT frame encoder → Q/V LoRA → CLS `[B,768]` とする。1-step検証に使うlossは目的と効果を説明できる単純なものとし、単なる一定のsumは避ける。意味のある継続学習や表現向上を主張しない。

検証候補:
1. encoder-onlyベースラインの `requires_grad=False` と既存CLS入出力を確認。
2. Q/V 24個のみにLoRAを挿入し、A/Bのshapeと総学習可能parameter数を検証。
3. forwardが有限値・指定shapeを返すこと、LoRA有無の差が初期化契約と整合することを確認。
4. backwardでLoRAに勾配が流れ、optimizer stepでLoRAだけ更新されることを確認。Bゼロ初期化なら初回A勾配がゼロでも仕様として正常なので、全LoRA tensorの1-step更新を絶対条件にしない。
5. frozen baseのstate_dictがbitwiseで不変、LoRA checkpointの保存→ロードで出力が再現されることを確認（同一eval入力・dtype・デバイス）。
6. 研究用データとSequential Loaderとの統合は後続の別フェーズに回す。

## 根拠・未確認事項

- **Observed**: 参照先のLoRA計算、Q/K/V・Attention出力・MLPへの挿入、share/domain-specific加算、射影、Xavier初期化、configの条件分岐、`from_pretrained` 呼び出し、b6385d8のViTFrameEncoder。
- **Inferred**: encoder-only単体確認には複雑なProjectedViT全体の移植より、Q/Vのシンプルなadapterだけが妥当。
- **Unverified**: ローカル `encoder-only` branch の未コミット変更、テストの実行結果、採用するPEFT versionとのruntime互換性、参照プロジェクトが期待どおりbaseを常時凍結しているか、LoRAが時間情報を学習できるか。
- 参考: [ProjectedViT LoRA implementation](https://github.com/tamaki-lab/2024_10_takama_ProjectedViT/blob/e6d5f8d59a66f869954248cc4427ea7697aed4e9/models/modelings/modeling_projectedvit.py)、[apply_lora](https://github.com/tamaki-lab/2024_10_takama_ProjectedViT/blob/e6d5f8d59a66f869954248cc4427ea7697aed4e9/models/model_util.py#L511-L523)、[encoder-only frame encoder](https://github.com/tamaki-lab/2026_04_ishikawa_simple-MeMViT/blob/b6385d8/model/vit/vit_frame_encoder.py)、[2026-10-08 MTG](https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/meetings/2026-10-08-mtg.md)
