---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: completed
created: 2026-09-23
last_updated: 2026-09-23
spec: ../specs/2026-09-23-activitynet-clip-feature-spec.md
implementation_branch: dev
implementation_base_commit: 3c0e40e86924ceb38931c2d5214ecf3251a8f99d
sequential_loader_branch: ActivityNet
sequential_loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
---

# ActivityNetからfrozen ViT clip featureへの実装・短時間検証

2026-09-23のユーザー指示により[spec](../specs/2026-09-23-activitynet-clip-feature-spec.md)を
`approved`へ更新し、指定の`dev`へ実装した。必須のunit / regression testsと
実ActivityNet先頭1 chunk smokeを完了したため、specを`implemented`へ更新した。
変更は未コミット。長時間run・学習・Git commit・pushは行っていない。

## 実装差分

実装root: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`

- `model/masked_mean_clip_aggregator.py`: bool maskでvalid rowだけを選択して平均する、小さいparameter-free `nn.Module`。shape / T / all-invalidを検査し、dtype・device・gradientを維持する。
- `model/__init__.py`: aggregatorを公開。
- `sequential_vit_bridge.py`: 既存50Salads smokeの`encode_chunk`をそのまま移動。元関数とのAST一致を確認。
- `smoke_50salads_vit_bridge.py`: 共通helperをimport。既存のbranch / revision検査・CLI・diagnosticは保持。
- `smoke_activitynet_vit_clip_feature.py`: runtime dataset root、指定branch・base ancestry・loader revision検査、training先頭1 chunk、frozen ViT、masked mean、diagnosticを接続。
- `test/model/test_masked_mean_clip_aggregator.py`、`test/model/test_activitynet_vit_clip_feature.py`: 合計25件のsynthetic / mocked tests。

ActivityNet smokeはstrict loaderの既定設定（B=1、shuffleなし、worker=0）を使用する。
`sequential_sample_stream`を抜けてreaderを解放してから、`eval()`と`torch.no_grad()`でfeatureを生成する。
Annotation label / segmentはprocessor・encoder・aggregationへ渡さない。

`main.py`、`main_pl.py`、Hydra config、classification dataset / model factory、
`SimpleLightningModel`、`ViTFrameEncoder`、dependencies、独立loader repositoryは変更していない。

## 検証環境

- Python: 3.12.3、PyTorch: 2.14.0+cu130、Transformers: 5.17.0。
- Python executable: 実装rootの`.venv/bin/python`。
- Device: CPU（`CUDA_VISIBLE_DEVICES=''`）。CUDA上での実行は未検証。
- Checkpoint: `google/vit-base-patch16-224`。
- 利用cache snapshot: `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3`。
- `HF_HUB_OFFLINE=1`で既存cacheを利用。新dependency・checkpoint downloadなし。
- Dataset root: `/mnt/NAS-TVS872XT/dataset/ActivityNet`。読み取りのみ。

## Unit / regression tests

実装rootで実行:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  test/model/test_masked_mean_clip_aggregator.py \
  test/model/test_activitynet_vit_clip_feature.py \
  test/model/test_50salads_vit_bridge.py -q -o addopts=''

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  test/config test/model/test_vit_frame_encoder.py -q -o addopts=''
```

| 検証 | 結果 |
|---|---|
| 新規25件 + 既存50Salads bridge 1件 | 26 passed（8.29秒） |
| Hydra config + 既存ViT test | 38 passed、既知の1 failed（75.85秒） |
| Python compileall | 成功 |
| 共通helperと旧Stage 2関数のAST照合 | 一致 |
| 既存training経路・ViT encoderの差分検査 | 変更なし |
| 差分の空白検査 | 成功 |

新規テストでは、all-valid mean、padding値が大きい場合の除外、reverse / permutation、
shape / dtype / device、all-invalid / T mismatchの拒否、valid rowだけへの正しいgradientを確認した。
さらにprocessor / encoderへのvalid frame限定、scatter、timestamp逆行時も含むmetadata保持、
実際のSequentialDataset / strict DataLoaderを通したmock readerの正常・例外時解放、
frozen eval・no_grad・parameter不変、誤ったrevisionの拒否を確認した。

既存失敗は`test_vit_frame_encoder.py::TestViTFrameEncoder::test_cls_feature_and_frozen_backbone`のみ。
テストが`from_pretrained(checkpoint)`を期待する一方、既存実装は`add_pooling_layer=False`を渡す。
変更前にも同じテストで`1 passed, 1 failed`を再現し、変更後も失敗内容は同一。
本specの対象外であり、実装・既存テスト双方を変更していない。新規回帰はない。

## 実ActivityNet先頭1 chunk smoke

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  .venv/bin/python smoke_activitynet_vit_clip_feature.py /mnt/NAS-TVS872XT/dataset/ActivityNet
```

終了code 0。診断結果:

| 項目 | 実測 |
|---|---|
| implementation branch / HEAD | `dev` / `3c0e40e86924ceb38931c2d5214ecf3251a8f99d` + 未コミット差分 |
| loader branch / HEAD | `ActivityNet` / `19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`、clean |
| training source数 | 10,024 |
| sequence_id | `---9CpRcKoU` |
| sequence_index / is_first / is_last | `0 / True / False` |
| frames | `[16,3,240,320]`、uint8、CPU |
| valid count | 16 / 16 |
| frame_indices | 0〜15、contiguous |
| timestamps | 0〜0.5秒、1/30秒間隔 |
| pixel_values | `[16,3,224,224]` |
| valid_features / frame_features | `[16,768]` / `[16,768]` |
| clip_feature | `[768]` |
| frame / clip finite | True / True |
| padding rows zero | True（この実chunkにpaddingなし。paddingはsyntheticで検証） |
| total / trainable backbone parameters | 85,798,656 / 0 |

Checkpointのclassification headに対応する`classifier.bias/weight`は、既存のpoolerなし
`ViTModel`への読み込みでUNEXPECTEDと表示された。既存ViT contractによるもので、forwardと有限性検査は成功した。

## Success criteriaとの対応

| spec 8章 | Evidence |
|---|---|
| 1–4 repository / branch / revisions | preflightとsmoke診断。両repositoryは指定revision、開始時clean |
| 5–11 ActivityNet→frame features | 実データ1 chunk成功。padding除外・scatterはsyntheticでも確認 |
| 12–17 masked mean | aggregator単体・統合テスト、実clip featureのshape / finite検査 |
| 18–20 freeze / no_grad / gradient | 実backbone trainable=0、mock eval / no_grad検査、aggregator backward単体テスト |
| 21–22 annotation / alignment | frame tensorだけを処理。metadata不変・timestamp逆行保持を検証 |
| 23–25 対象外の維持 | classification / Hydra / ViT / loader無変更。LoRA・MoCo・optimizer・学習loop追加なし |
| 26 非回帰 | 50Salads 1件、Hydra 38件成功。既存ViT失敗1件は変更前後同一 |

実ActivityNet全動画forward、GPU実験、LoRA / MoCo学習、downstream性能・temporal modelingの評価は実施していない。
`backward()`はsynthetic aggregatorのgradient testだけで使い、実データsmokeではparameter updateを行っていない。
この結果が示すのは、frozen ViTによる1 chunkのorder-invariant表現の生成成立までである。

## 検証した変更ファイルのSHA-256

未コミット版を識別するため、実装・テストのhashを記録する。

| File | SHA-256 |
|---|---|
| `model/masked_mean_clip_aggregator.py` | `20352874338cd893596e9c72dec51ef930642dc05c6a4ff5a260a589a8df723d` |
| `model/__init__.py` | `3190ea9fba6a21b9995571b141da8871b88abc4cc3d7a81439ffecf22dae1d59` |
| `sequential_vit_bridge.py` | `84802c9ddeb97759c20a5b9f48bdb4fbd15893b3c168a41444fa56f2a6d74bb8` |
| `smoke_50salads_vit_bridge.py` | `ded05711b009020cec6fe1eea29141a704fdec6fbbaceb72c77a3cc34f2bf03f` |
| `smoke_activitynet_vit_clip_feature.py` | `d45e76eea5d04437cf7734da4200d4e3fa9b3b8f87ef15cb17301c140ea0a3bc` |
| `test/model/test_masked_mean_clip_aggregator.py` | `c220ffac2d7ee9e7a165c840ab73676bd92fc952f0286ee02589e33fc8960237` |
| `test/model/test_activitynet_vit_clip_feature.py` | `326b56d9543f914917322928617122328d4b66e99e43ce2e162f51abb35d0f84` |

## 中断後の再開確認（2026-09-23）

ユーザーの再開依頼に基づき、残scopeを監査した。上記7ファイルのSHA-256はすべて
現行ファイルと一致し、specの必須実装に不足はなかった。追加のコード変更は行わず、
以下の短時間検証を再実行した。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python -m pytest test/model/test_masked_mean_clip_aggregator.py test/model/test_activitynet_vit_clip_feature.py test/model/test_50salads_vit_bridge.py test/config test/model/test_vit_frame_encoder.py -q -o addopts=''
```

- 結果: **64 passed、既知の1 failed**（77.09秒）。新規25件・50Salads 1件・Hydra 38件は成功。
- 失敗は前回と同じViTテストの`add_pooling_layer=False`に対する期待引数不一致。
  当該テストと`ViTFrameEncoder`は基準HEADから変更されていないことを確認した。
- 上記と同じ実ActivityNet smokeコマンドも再実行し、終了code 0。
  `training` sources 10,024、`sequence_id=---9CpRcKoU`、valid 16/16、
  `frame_features [16,768]`、有限な`clip_feature [768]`、trainable backbone parameters 0を再確認した。
- 実装branch / HEADとloader branch / HEADは記録どおりで、loader worktreeはclean。
- 7ファイルの構文検査、共通helperと旧50Salads関数のAST一致、既存training / ViT経路の差分なし、
  `git diff --check`を再確認した。

Stage 3は実装・必須検証完了の状態を維持する。今回もCPUのみを使用し、
長時間run・学習・Git commit・pushは行っていない。コードの未コミット差分は保持した。
