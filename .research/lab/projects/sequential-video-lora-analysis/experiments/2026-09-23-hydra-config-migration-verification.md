---
project: sequential-video-lora-analysis
record_type: implementation-verification
created: 2026-09-23
implementation_branch: dev
implementation_base_commit: 7705e2159678fa516a8f17a0633e4712377b6eb9
---

# Hydra設定移行の実装・短時間検証

[承認済みspec](../specs/2026-09-23-hydra-config-migration-spec.md)を実装した。
実装先は `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`。
開始時点は指定どおり `dev@7705e2159678fa516a8f17a0633e4712377b6eb9`、clean worktree。
変更は未コミット。学習・GPU実験・commit・pushは実行していない。

## 実装内容

- `hydra-core==1.3.7` を `requirements.txt` に追加し、既存 `.venv` に導入。検証環境の OmegaConf は `2.3.1`。
- `conf/config.yaml`、dataset 4種類、model 6種類、optimizer 2種類のYAMLを追加。
- `main.py` / `main_pl.py` を `@hydra.main(version_base="1.3", ...)` へ移行。
- dataset componentへdataset / loader / video subtreeを、Lightning modelへmodel / optimizer / scheduler / checkpoint subtreeを渡す。既存 `ModelConfig` を再利用。
- training用 `args/arg_parse.py` / `args/__init__.py` を削除。旧CLI互換layerは追加していない。
- `hydra.job.chdir=false`、run / sweepの記録先を `log/hydra/` 配下に設定。時刻にはmicrosecondを含め、sweep subdirにはjob番号を使用。
- Hydra標準の `.hydra/config.yaml`, `hydra.yaml`, `overrides.yaml` を保持。補間を解決した実行設定は、各entrypointの `main.log` / `main_pl.log` に `Resolved config` として記録する。独自YAML copy機構は作成していない。
- Lightningに渡すdevicesは文字列化し、Hydraが `trainer.devices=0` を整数としてparseしても従来のGPU ID指定を維持。
- READMEの起動例と設定説明を更新。旧training CLIへ直接依存する `.vscode/launch.json` / `tasks.json` も更新し、loggerの旧 `vars(args)` 説明を更新。

`trainer.val_interval_epochs` は従来どおり `main.py` で使用する。Lightningの毎epoch検証、未使用の `logging.tf_log_dir` など、既存設定の実際の使用範囲は変更していない。
utility / smoke script、dataset / modelの研究algorithm、optimizer / schedulerの実装、`ModelConfig`、train / validation処理は無変更。

## 検証結果

| 検証 | 結果 |
|---|---|
| Hydra新規テスト | 38件成功 |
| 既存CPU短時間テスト | 21件成功、既存不一致1件（移行前後で同一） |
| 元commitのArgParseとの直接照合 | 全28項目の値・型が一致。split directoryはImageFolder設定で照合 |
| VS Code training起動設定 | 22件すべてcompose成功 |
| Python compileall | 成功 |
| `git diff --check` | 成功 |

新規テストの範囲:

- 全config group、既定値、scalar / boolean / null override。
- 両entrypointの `--cfg job --resolve` と代表override。GPU非表示の別working directoryでも成功。
- 両entrypointによる旧 `-b` CLIの拒否。
- Hydra実行時cwdと相対パス、標準artifact、ログ内の補間解決済み設定。実training依存を最初の呼び出しで止めるモックを使い、データ取得・model構築・Comet experiment・学習は起動していない。
- dataset / loader / video値が既存factoryへ渡ること、Lightningの `ModelConfig`、SGD / Adamのhyperparameter、既存scheduler、checkpoint保存先・state key。
- 通常entrypointのDP指定・resume・validation間隔・checkpointパス、Lightning entrypointのdevices・epoch・gradient accumulation・resume。

最終の一括検証コマンド（実装repositoryで実行）:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  test/config \
  test/dataset/test_zero_images.py \
  test/model/test_vit_frame_encoder.py \
  test/model/test_50salads_vit_bridge.py \
  test/utils/test_accuracy.py \
  test/utils/test_average_meter.py \
  -q -o addopts=''
```

結果は **59 passed, 1 failed（78.21秒）**。失敗は次の既存テストのみで、新規の失敗はない。
移行前の同じ既存テスト集合でも **21 passed, 1 failed** を確認している。

### 既存テストの不一致

`test/model/test_vit_frame_encoder.py::TestViTFrameEncoder::test_cls_feature_and_frozen_backbone` は、
`ViTModel.from_pretrained('google/vit-base-patch16-224')` を期待するが、
既存実装は `add_pooling_layer=False` を指定しているため失敗する。
対象実装とテストは今回変更していない。Hydra移行とは独立した不一致として残す。

## Success criteriaとの対応

- spec 8章の1–16、18: branch / revision、dependency、config構造、両entrypoint、CLI、既定値、cwd / artifact、component責務、README、compose test、scopeをコード差分と上記検証で確認。
- 17: 既存CPU短時間テストで新たな回帰なし。既存のViTテスト不一致1件は実装前後で同一。

GPUを要求する既存model / optimizer / checkpointテスト、NASの実動画を要求するdatasetテスト、full training、実checkpointからのGPU再開は未実施。
今回の結果は設定移行の実装検証であり、学習結果の科学的同等性を示すものではない。
