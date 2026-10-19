---
project: sequential-video-lora-analysis
date: 2026-10-19
analysis_mode: explain
detail: compact
level: mid
source_repository: https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_branch: dev
source_commit: 1ce9badcbae3e2df08fa400944e4c385785063f9
source_dirty: unknown
targets:
  - model/vit/vit_lora_frame_encoder.py
  - model/vit/lora_linear.py
  - model/masked_mean_clip_aggregator.py
  - integration/sequential_vit.py
  - scripts/smoke/smoke_activitynet_vit_clip_feature.py
related_spec: .research/lab/projects/sequential-video-lora-analysis/specs/2026-09-28-shared-streaming-moco-protocol-spec.md
---

# Code Deep Dive: ViTのQ/V LoRA、frame-wise CLS、Masked Mean、Clip Feature、Projectorの関係

## 対象と解析の範囲

ユーザーの疑問：図では「LoRA学習 → frame-wise CLS → masked mean pooling → clip feature → projector」のように直列に見えるが正しいか。各処理の役割は何か。

- 研究文脈: `sequential-video-lora-analysis` のREADMEと2026-09-28 implemented spec
- 研究コード: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、GitHubで2026-10-19に取得できたcommit `1ce9badcbae3e2df08fa400944e4c385785063f9`
- 除外: 動画MoCo共通学習engineのGitHub上の本体コード、研究サーバの未コミット・未push差分とraw実験を直接確認できない。

## Overview: What / Why

図の誤解点は、LoRAをViTの外側の直列モジュールとして見せていること。Q/V LoRAはViT内部Attention投影に残差として注入され、ViTのforward中に使用される。その結果として各フレームのCLSが得られる。『LoRA学習』とはforwardの先頭で発生する独立工程ではない。Lossのbackwardとoptimizer stepでLoRAパラメータが更新される。

Forwardの概念的な流れ：

```text
Sequential Loader -> 16 valid/possibly padded frames
  -> two views (query raw / key transformed)
  -> each branch: ViT image encoder [内部のQ/V LoRA]
  -> frame-wise CLS [T,768]
  -> Masked Mean Pooling [768] = clip feature
  -> (2-layer Projector 768 -> 768 -> 128) -> L2 normalize -> q/k
  -> InfoNCE(q,k,old queue) -> loss
  -> backward(Query only) -> optimizer(Query LoRA + Projector)
  -> EMA(Key branch), enqueue detached key
```

動画specの設計上の計算段階と、現行GitHubリモートでコードとして確認できる段階を分ける。

## Key Components and Data Flow

### 実コードで確認（Observed）

1. `model/vit/vit_lora_frame_encoder.py`:
   - `ViTLoRAFrameEncoder` はHugging Face `ViTModel`を生成、事前学習済みBaseを `requires_grad_(False)`。
   - `inject_attention_lora` は選択された各blockの `q_proj/k_proj/v_proj` を `LoRALinear` でラップ。デフォルトtargetは `q,v`。
   - forwardの戻り値は `last_hidden_state[:,0,:]`。CLSトークンの最終状態であり、入力frameごとに `[batch,768]`。
2. `model/vit/lora_linear.py`:
   - `LoRALinear.forward(x) = base(x) + (alpha/rank)*B(A(dropout(x)))`。Base線形層は固定、LoRA A/Bを訓練対象にする。
3. `integration/sequential_vit.py`:
   - `encode_chunk` は `sample.valid_mask` がTrueの実フレームのみを前処理、ViT encoderへ与え、`[valid_count,768]`を取得。
   - 無効フレームの位置を0埋めに復元した `[T,768]` の `frame_features` を返す。
4. `model/masked_mean_clip_aggregator.py`:
   - `MaskedMeanClipAggregator.forward` は `frame_features[valid_mask].mean(dim=0)`。
   - Paddingのフレームを平均から除外する「mask」であり、masked image modelingのMAEとは別。
   - 順序不変: 同じvalid特徴集合を並べ替えてもclip featureは不変。クリップ内の時間方向を直接モデル化しない。
5. `scripts/smoke/smoke_activitynet_vit_clip_feature.py`:
   - Frozen ViTを用いたActivityNet先頭16frameで、`frame_features [16,768]`、`clip_feature [768]`を期待し確認するコード。
   - このsmoke自体は`ViTFrameEncoder`を使用し、LoRAと動画MoCo Projectorのforward確認とは異なる。

### 研究specにある設計（Aligned with spec; GitHubリモートで統合本体は未確認）

2026-09-28 `shared-streaming-moco-protocol-spec.md` のstatusは`implemented`、条件は：
- 16 frames/chunk; ViT: `google/vit-base-patch16-224`, Base frozen。
- LoRA Q/V、rank 8 / alpha 8。
- frame-wise CLS → masked mean → `[768]`。
- Projector 768 → 768 → 128、L2 normalize後に対照学習。
- QueryのLoRA + Projectorをoptimizerで更新、Key LoRA + Key ProjectorをEMAで追従。
- 2 views、InfoNCE + FIFO queue; strict_single時は過去の動画chunkのKeyをnegativeに使う。
- Projectorは単なる次元圧縮に限らず、contrastive-loss用の表現空間への写像。後段でpre-projector clip特徴を評価すればProjector学習の影響を区別しやすい。

今回閲覧した`dev`のrecursive treeには画像MoCoの `self_supervised/moco/standalone_moco.py`, `ddp_moco.py`、ViT encoder/aggregatorとsmokeはあるが、2026-09-28 specにある統合video streaming MoCo engineの本体は確認できなかった。研究サーバ側のlocal dirty/unpushed状態とは別物であり、GitHub上で未確認だから存在しないとは断定しない。

## Concepts and Design Decisions

- CLS: Vision Transformerが画像パッチ列から作る画像単位の特徴。`last_hidden_state[:, 0, :]`で取り出す。
- Masked Mean: valid frameの特徴の平均。`c = (sum_t m_t f_t)/(sum_t m_t)`（`m_t∈{0,1}`）。
- Clip Feature: poolingによる1chunkの代表ベクトル（別の学習層を指すのではない）。
- Projector: 表現学習の損失用に`[768]`をMLPで`[128]`へ変換し正規化するパラメータ付き層。
- 逆伝播: MoCo Loss → projector → mean pooling → CLS/ViTの演算経路 → Q/V LoRA。Base ViTは固定、Keyは勾配ではなくEMA。実際のoptimizerとEMAの正確な順序は対応するtraining engineのコード確認が必要で、spec記述はoptimizer→EMA→enqueue。
- 時間情報: frame-wise ViT + masked meanはclip内のフレーム順序に不変。動画を逐次入力しただけでは時間順序理解の証拠にならない。

## Alternatives and Trade-offs

1. Baselineとしてmean poolingは実装単純・shape固定・比較可能だが、順序表現を失う。
2. Temporal attention/3D ViT/future predictionを導入すれば時間構造を直接扱えるが、学習・設計変数が増える。導入は未承認の研究提案。
3. Downstream probeでpre-projector特徴を中心に評価すれば、Projectorの有用性とLoRAの効果の交絡を整理できる。

## 根拠・未確認事項

- GitHub observed: [ViTLoRAFrameEncoder](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/model/vit/vit_lora_frame_encoder.py), [LoRALinear](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/model/vit/lora_linear.py), [MaskedMeanClipAggregator](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/model/masked_mean_clip_aggregator.py), [encode_chunk](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/integration/sequential_vit.py), [ActivityNet smoke](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/smoke/smoke_activitynet_vit_clip_feature.py).
- Research Workspace: [2026-09-28 implemented spec](https://github.com/haruto2919/research-workspace/blob/main/.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-28-shared-streaming-moco-protocol-spec.md).
- 未確認: SSH/ローカル研究サーバのdirty change、unpushed modules、Projectorを含む動画統合runnerの現時点の実体・トレース・再実行結果。
- 本記録はread-only解析であり、コード変更や実験runを実施していない。
