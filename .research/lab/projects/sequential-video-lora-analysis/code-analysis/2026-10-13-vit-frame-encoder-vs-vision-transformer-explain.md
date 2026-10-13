---
project: sequential-video-lora-analysis
date: 2026-10-13
analysis_mode: explain
detail: compact
level: junior
source_repository: https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_commit: b6385d87e2e3e82a719d8f4b686b44aa293b1135
source_dirty: unknown
targets:
  - model/vit/vit_frame_encoder.py
  - model/vit/vision_transformer.py
  - model/model_factory.py
  - sequential_vit_bridge.py
  - smoke_vit_frame_encoder.py
related_spec: null
---

# Code Deep Dive: ViTFrameEncoder と ViTb の役割比較

## 対象と背景

2026-10-13の質問: ローカルの `model/vit/vit_frame_encoder.py` と `model/vit/vision_transformer.py` はそれぞれ何をするのか。GitHubの `tamaki-lab/2026_09_ishikawa_sequential-video-lora@b6385d8` を静的に読む。ローカル `encoder-only` の未コミット差分は確認していない。

## What / How / Why

### ViTFrameEncoder

- `model/vit/vit_frame_encoder.py` の `ViTFrameEncoder(nn.Module)`。
- `ViTModel.from_pretrained("google/vit-base-patch16-224")` でViT本体をロードし、`self.vit.requires_grad_(False)` で全parameterを凍結。
- 入力は前処理済みRGB画像 `[B,3,224,224]`、返り値は `last_hidden_state[:,0,:]`、すなわちCLS特徴 `[B,768]`。分類logitsもLossも返さない。分類Headは存在しない。
- `sequential_vit_bridge.py` はValidフレームを取り出してencoderへ渡し、得られたfeaturesをchunk内の元位置へ並べ直してpadding位置を0埋めする。 `smoke_vit_frame_encoder.py` は1枚の画像で出力shape/有限性/凍結を検証する（ただしsmokeは`argparse`を使用し、統合設定がすべてHydraとは限らない）。
- `ViTModel.from_pretrained()` のpooler生成はこのcommitでは明示的に無効化されていないが、forwardの出力にはpoolerではなくCLSのhidden stateを使っている。

### ViTb

- `model/vit/vision_transformer.py` の `ViTb(ClassificationBaseModel)`。
- `ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=model_config.n_classes, ignore_mismatched_sizes=True)` を使い、画像分類用Head付きViTを作る。
- 入力は `pixel_values` と任意の `labels`。出力は `ModelOutput(logits=output.logits, loss=output.loss)`。logits shapeは `[B,C]`、lossはラベルがある場合に得られる（未指定なら一般にNone）。
- このファイルでは`requires_grad_(False)`を指定していないので、通常の利用では事前学習済みViT本体もHeadも学習対象になり得る。
- `model/model_factory.py` で `model_name=="vit_b"` のとき `ViTb` を作る。一般の教師あり画像分類のモデル経路。

## 比較

| 項目 | ViTFrameEncoder | ViTb |
| --- | --- | --- |
| HFモデル | ViTModel | ViTForImageClassification |
| 用途 | フレーム特徴抽出 | 教師あり画像分類 |
| 返り値 | CLS feature [B,768] | logits [B,C], optional loss |
| Classifier head | なし | あり |
| Backbone freeze | 明示あり | ファイル内で明示なし |
| 主要用途 | Sequential Loader接続用特徴抽出 | 一般の画像分類トレーニング |

## 今回のLoRA-only + 実画像検証への示唆（提案）

- 元の研究上のCLS特徴契約を維持するなら `ViTFrameEncoder` を起点として別のLoRA付encoderを作り、Q/K/V線形層だけをLoRA付きwrapperへ差し替える案が自然。
- CIFAR-10等でCrossEntropyを使う教師あり学習を確認するなら、CLS特徴に10クラスHeadを追加する必要がある。工学的なLoRA-onlyの勾配/更新テストと、LoRA+Headを使う実画像分類学習は区別する。
- 既存`ViTb`を無条件にそのまま使うと、Base ViTが凍結されていることは保証されない。 `ViTFrameEncoder` を置き換える機能でもない。
- LoRA実装方式およびHeadの学習可否はまだspecで確定していない。

## 根拠と未確認事項

- **Observed**: 上記GitHubソースと分岐・戻り値。 
- **Inferred**: LoRA単体検証での出発点としてViTFrameEncoderの特徴契約を維持する設計が有利。
- **Unverified**: SSHサーバのローカルworktree差分、dirty state、実行時の実測、CIFAR-10学習結果。

### 直接参照

- [ViTFrameEncoder](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/model/vit/vit_frame_encoder.py)
- [ViTb](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/model/vit/vision_transformer.py)
- [model factory](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/model/model_factory.py)
- [bridge](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/sequential_vit_bridge.py)
