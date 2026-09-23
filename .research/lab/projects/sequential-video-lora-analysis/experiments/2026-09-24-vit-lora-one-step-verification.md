---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: completed
created: 2026-09-24
last_updated: 2026-09-24
spec: ../specs/2026-09-23-vit-lora-one-step-smoke-spec.md
implementation_branch: dev
implementation_base_commit: b6385d87e2e3e82a719d8f4b686b44aa293b1135
sequential_loader_branch: ActivityNet
sequential_loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
---

# Stage 4 ViT Q/V LoRAの実装・1-step検証

2026-09-23のユーザー指示により[spec](../specs/2026-09-23-vit-lora-one-step-smoke-spec.md)を
`approved`へ変更し、指定の実装repositoryの`dev`へ実装した。
日付をまたいだ2026-09-24に、新規22件・既存65件のテストと実ActivityNet先頭1 chunkの
CPU smokeが成功した。specのstatusはユーザー指定の`approved`を維持する。
変更は未コミット。長時間学習・GPU run・commit・pushは行っていない。

## 実装と既存変更の扱い

実装root: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`。
originは`tamaki-lab/2026_09_ishikawa_sequential-video-lora`、開始・検証時HEADは上記base commitと一致。

- `model/vit/vit_lora_frame_encoder.py`: poolerなしViT、CLS `[B,768]`、PEFT Q/V LoRA。baseをfreezeし、`r=8 / alpha=8 / dropout=0 / bias=none`に固定。
- `model/vit/__init__.py`、`model/__init__.py`: `ViTLoRAFrameEncoder`のexport。
- `requirements.txt`: `transformers[torch]==5.17.0`と`peft==0.21.0`をpin。既存venvが一致しており、installは不要だった。
- `smoke_activitynet_vit_lora_one_step.py`: runtime dataset root、CPU既定のdevice指定、implementation branch / base ancestry・loader branch / revision / tracked clean・dependency versionを検査。strict loaderの先頭16 frameを既存bridge / masked meanへ接続し、engineering lossでAdamWを1 step実行。
- `test/model/test_vit_lora_frame_encoder.py`、`test/model/test_activitynet_vit_lora_one_step.py`: 新規22件。

開始時から`model/vit/vit_frame_encoder.py`と`test/model/test_vit_frame_encoder.py`に
`add_pooling_layer=False`の未コミット修正があった。本specと一致するため、その変更を保持した。
この2ファイルへ追加編集はしていない。

Stage 3 smoke、50Salads smoke、`sequential_vit_bridge.py`、`MaskedMeanClipAggregator`、
`main.py`、`main_pl.py`、Hydra config、独立Sequential Loader repositoryは変更していない。
readerはstreamのcontextを抜けて解放し、その後にgrad有効でLoRA forwardを行う。
optimizer構築前のtrainable集合監査、base / LoRA snapshot、gradient・step後差分・有限性の
監査に失敗した場合は例外で終了する。

## 検証環境

- Python 3.12.3、PyTorch 2.14.0+cu130、Transformers 5.17.0、PEFT 0.21.0。
- Python executable: 実装rootの`.venv/bin/python`。
- Device: CPU。`CUDA_VISIBLE_DEVICES=''`でGPUを使用していない。
- Checkpoint: `google/vit-base-patch16-224`、`add_pooling_layer=False`。
- Cache snapshot: `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3`。
- `HF_HUB_OFFLINE=1`で既存cacheを使用。モデルdownloadなし。
- Dataset root: `/mnt/NAS-TVS872XT/dataset/ActivityNet`。datasetは読み取りのみ。
- loaderは`ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`、worktreeはclean。

## Unit / regression tests

実装rootで実行:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  test/model/test_vit_lora_frame_encoder.py \
  test/model/test_activitynet_vit_lora_one_step.py -q -o addopts=''

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  test/model/test_vit_frame_encoder.py \
  test/model/test_masked_mean_clip_aggregator.py \
  test/model/test_activitynet_vit_clip_feature.py \
  test/model/test_50salads_vit_bridge.py test/config -q -o addopts=''
```

| 検証 | 結果 |
|---|---|
| 新規LoRA / one-step tests | 22 passed、17.18秒 |
| 既存Stage 1–3 / config tests | 65 passed、77.98秒 |
| 新規Python 4ファイルのAST構文検査 | 成功 |
| `git diff --check`、変更ファイルの末尾空白検査 | 成功 |
| 保持対象の既存経路・loader差分検査 | 変更なし |

Unit testはdownloadなしで実際のTransformers ViTとPEFTを使用する。
12 attention blocks・hidden size 768を維持し、テスト用のMLPだけ`intermediate_size=32`へ縮小する。
実checkpointは後述の実データsmokeで検証した。

検証内容は、poolerなしload、Q/V各12・計24 target、LoRA設定・294,912 parameters / 48 tensors、
base freeze、CLS / eval forward、valid frame抽出・scatter・masked meanを通るbackward、
optimizer集合一致、base完全不変、LoRA更新、全parameter有限性。
padding rowのgradientは0、valid rowのgradientは解析的な期待値と一致した。
metadataはtimestamp逆行とpaddingのNaNを含めて維持した。

誤ったprovenance / dependency、base trainable、adapter freeze、target不足、pooler有効、
ゼロ / 非有限gradient、更新なし、base更新、非有限parameter更新を拒否するテストも成功。
strict loaderは1 chunkだけ読み、正常終了・decode失敗・forward失敗時のreader解放を確認した。
既存テストのwarning 1件は`torch.jit.script_method`の非推奨通知で、失敗はない。

## 実ActivityNet one-chunk smoke

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  .venv/bin/python -u smoke_activitynet_vit_lora_one_step.py \
  /mnt/NAS-TVS872XT/dataset/ActivityNet --device cpu
```

終了code **0**。

| 項目 | 実測 |
|---|---|
| dataset / split / source数 | ActivityNet v1.3 / training / 10,024 |
| sequence_id / sequence_index | `---9CpRcKoU` / 0 |
| valid count | 16 / 16 |
| frame_indices / timestamps | 0〜15 / 0〜0.5秒、1/30秒間隔 |
| pixel_values | `[16,3,224,224]` |
| valid_features / frame_features / clip_feature | `[16,768]` / `[16,768]` / `[768]` |
| frame / clip finite、padding rows zero | すべてTrue |
| pooler is None | True |
| q_proj / v_proj / total targets | 12 / 12 / 24 |
| total model parameters | 86,093,568 |
| trainable parameters / tensors | 294,912 / 48 |
| unexpected trainable names | `[]` |
| optimizer集合とLoRA trainable集合 | 一致 |
| engineering-only loss | 0.7365954518318176、finite |
| LoRA finite / nonzero gradient tensors | 48 / 24 |
| base gradient tensors | 0 |
| base changed parameter tensors | 0 |
| LoRA changed parameter tensors | 24 |
| 更新後の全parameter有限性 | True |

この実chunkにはpaddingがないため、padding除外とgradientは上記synthetic testでも検証した。
LoRA初期化により、全48 tensorsの更新は要求していない。seedは新たに固定しておらず、
lossとLoRAの具体的な更新値の一致は再実行時の成功条件ではない。
checkpointのclassification headは使用しないため、load reportに
`classifier.weight / classifier.bias`のUNEXPECTED表示があり、pooler MISSINGはなかった。

## Success criteriaとの対応

| spec 7章 | Evidence |
|---|---|
| 1–5 repository / branch / base / PEFT / checkpoint | preflight、pin、実smoke診断 |
| 6–14 pooler / CLS / targets / LoRA設定 / trainable集合 | 実PEFT unit tests、実checkpoint smoke |
| 15–18 base optimizer除外 / 16-frame / masked mean / finite clip | optimizer集合監査、既存回帰、synthetic / 実smoke |
| 19–23 loss / backward / gradients / optimizer一致 | synthetic更新test、実smoke診断 |
| 24–26 base不変 / LoRA更新 / finite | 全parameter snapshotとの比較と更新後検査 |
| 27 新規回帰なし | 指定の既存65件成功、新規22件成功 |
| 28 実ActivityNet smoke | 先頭1 chunk、終了code 0 |
| 29–30 長時間学習なし / temporal成功を要求しない | CPU 1-step engineering verificationに限定 |

実装と必須の短時間検証は完了。MoCo、multi-step training、CUDA実行、downstream評価、
時間情報・動作情報の獲得は未検証であり、本結果から研究性能については結論しない。

## 検証したファイルのSHA-256

未コミットの実装版を識別する。既存のpooler修正2ファイルも検証対象として含む。

| File | SHA-256 |
|---|---|
| `requirements.txt` | `f6476f8017ad52a4b68a8cec1bde59fdb7ef54b0f054a47468fbf7aca643b38d` |
| `model/__init__.py` | `304e9cdbe26a6fae64b4d555da3340932fd1b8a38727d8f6f131cfe2dbaa2089` |
| `model/vit/__init__.py` | `719787bf79c2e1b28274b8c128164546fa9a9b77730963d60d00c83a43f8c404` |
| `model/vit/vit_frame_encoder.py` | `b9a3f728ab7dc79bd7fead9dd4002c299a8137b8b41a29ef0e4f946a1afdfafc` |
| `test/model/test_vit_frame_encoder.py` | `2a59ce55720c3b5e72744be8ee6fe881c2010ff6f20645c9a7b947e60de48251` |
| `model/vit/vit_lora_frame_encoder.py` | `502ad13a03d27a8f5a5907bd95a56d6503ba8049ffca83c73d499d57993e2a02` |
| `smoke_activitynet_vit_lora_one_step.py` | `13ee33dbffc7d6ab12588145c31e4202c58b0161009f0c258f3299ec8d71ad55` |
| `test/model/test_vit_lora_frame_encoder.py` | `8c8a70001f5ea7cc959a0df1f73ad759f8b1340a852a31a6ca96be2dbcf16768` |
| `test/model/test_activitynet_vit_lora_one_step.py` | `ffc74768127abe49e6265b6054f87b9d1ddb13da20c6fd78369f0b55d4ab0d0c` |


## 現行devでの再検証（2026-09-24 04:14 JST開始）

指定specの再確認時、Stage 4実装は既にcommit
`3b980c8a90e8cad08e78a5ba13cbb3629ec5ae58`に含まれていた。
現行HEADは`dev@1cbaa4a0fde2fd196a36feb3eeb0074709b52cd9`で、実装repositoryのworktreeはclean。
spec基準commitの後継であることと、その後のStage 5追加・module配置整理の影響を確認した。
Stage 4 encoderの現行配置は`model/backbones/vit/vit_lora_frame_encoder.py`であり、
後続変更はStage 4経路のimport・testのmock参照先を新配置へ追従させている。
specを満たす既存実装が揃っているため、今回コードの追加・変更は行っていない。

上記と同じPython / PyTorch / Transformers / PEFT環境、CPU、既存checkpoint cacheを使用し、
現在のコードで次を再実行した。loaderも引き続き指定branch / revisionに一致し、tracked変更なし。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest \
  test/model/test_vit_lora_frame_encoder.py \
  test/model/test_activitynet_vit_lora_one_step.py \
  test/model/test_vit_frame_encoder.py \
  test/model/test_masked_mean_clip_aggregator.py \
  test/model/test_activitynet_vit_clip_feature.py \
  test/model/test_50salads_vit_bridge.py test/config \
  -q -o addopts='' -p no:cacheprovider

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u smoke_activitynet_vit_lora_one_step.py \
  /mnt/NAS-TVS872XT/dataset/ActivityNet --device cpu
```

- Stage 4の22件と既存回帰65件: **87 passed, 1 warning、87.43秒**。warningは既存の`torch.jit.script_method`非推奨通知。
- 実ActivityNet先頭1 chunk: **終了code 0**。sequence `---9CpRcKoU`、valid 16/16、pixel `[16,3,224,224]`、frame `[16,768]`、clip `[768]`。
- poolerなし、Q/V各12・計24 target、trainable 294,912 parameters / 48 tensors、unexpected trainable `[]`、optimizer集合一致。
- loss `0.7365954518318176`、finite gradient 48 tensors / nonzero 24 tensors、base gradient 0。
- base変更0、LoRA変更24 tensors、更新後の全parameter有限性True。
- `git diff --check`成功。今回の長時間run・GPU run・commit・pushはなし。

今回もspecのengineering検証条件を満たした。研究性能・時間情報獲得はこの検証の対象外。
上のSHA-256表は初回の未コミット実装版を識別する記録であり、今回の検証版は上記HEADで識別する。
