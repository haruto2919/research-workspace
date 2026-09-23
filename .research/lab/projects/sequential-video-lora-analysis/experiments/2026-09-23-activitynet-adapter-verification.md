---
project: sequential-video-lora-analysis
status: completed
created: 2026-09-23
last_updated: 2026-09-23
spec: ../specs/2026-09-23-activitynet-adapter-spec.md
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential_loader
implementation_branch: ActivityNet
implementation_base_commit: cef09aa12560127451a5f569d86d5d51671e6986
---

# ActivityNet Adapter 実装・短時間検証記録

2026-09-23のユーザー指示により[spec](../specs/2026-09-23-activitynet-adapter-spec.md)を
`approved`へ更新し、指定された`ActivityNet` branchに実装した。
全183テスト、実データの全source inventory、3形式各1本の先頭chunk検証に成功した。
remote反映と必須検証まで完了したため、specは`implemented`へ更新した。

## 実装差分

実装root: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential_loader`

- `src/adapters/activitynet.py`: full v1.3のJSON subsetとprimary rootsを検証するAdapter、opaque annotation reference。
- `sequential_loader/__init__.py`: `ActivityNetAdapter`、`ActivityNetAnnotationReference`を公開。
- `tests/test_activitynet_adapter.py`: 正常系、strict error、full-size synthetic inventory、Reader/Dataset/DataLoader統合の22テスト。
- `tests/test_public_api.py`: 新public exportの検証を追加。

JSONの`VERSION 1.3`、3 subsetとその件数を確認し、毎回full inventoryを検証する。
小規模fixtureのテストだけはprivate件数定数を縮小し、別テストでは実際の19,994件の
synthetic JSONとdummy filesでproduction件数を検証した。
JSONと対応動画を同時に減らして一致させたpartial datasetも拒否する。

`v1-3/train_val`と`v1-3/test`内だけを再帰探索する。
正規化は先頭の`v_`だけを除去し、動画はnormalized ID順に列挙する。
Annotation label/segmentはsource metadataやtargetへ複製せず、JSON entryへの参照だけを渡す。

## 検証環境

- Python: `3.12.3`
- PyTorch: `2.14.0+cu130`
- PyAV: `18.1.0`
- NumPy: `2.5.2`
- 使用Python: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/.venv/bin/python`
- 読み込まれたpublic package: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential_loader/sequential_loader/__init__.py`
- 実データroot: `/mnt/NAS-TVS872XT/dataset/ActivityNet`
- Datasetの書き換え、新dependencyの追加はなし。

## Unit / regression tests

実装rootで実行:

```bash
/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/.venv/bin/python -m unittest discover -s tests -p 'test_activitynet_adapter.py' -v
/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/.venv/bin/python -m unittest discover -s tests -v
```

- ActivityNet専用: **22 tests, OK**（2.099秒）。
- 全体: **183 tests, OK**（4.185秒）。既存50Salads・公開API・Core回帰テストを含む。
- Segmentが動画実時間の外にある人工annotationでも、全4 frameが既存Datasetを通過し、
  3 frame chunkと末尾padding、opaque referenceの同一性を確認した。
- `git diff --check`: 成功。

## 実データinventory

新Adapterをtop-level public APIから呼び、以下を直接確認した。

| subset | source count |
|---|---:|
| training | 10,024 |
| validation | 4,926 |
| testing | 5,044 |
| 合計 | 19,994 |

Supported extension内訳: `.mp4` 18,182、`.mkv` 1,789、`.webm` 23。
Adapterのstrict検証が成功し、missing / extra / duplicate / root-subset mismatchは0。
全sourceについてID、normalized ID順、`start_frame=0`、`stop_frame=None`、
annotation referenceのsubsetと未検証alignmentをassertした。

## 実データ先頭chunk smoke

training→validation→testingの順に全sourceを見て、各拡張子の最初のsourceを選択した。
既存`SequentialVideoReader`、`SequentialDataset`、`build_sequential_dataloader`で
`frames_per_chunk=16`の先頭1 sampleを取得し、`sequential_sample_stream`で終了時に資源を解放した。

| extension | normalized video ID | frames shape | valid frames |
|---|---|---|---:|
| `.mp4` | `---9CpRcKoU` | `[16, 3, 240, 320]` | 16 |
| `.mkv` | `-AaOr1DI2no` | `[16, 3, 240, 320]` | 16 |
| `.webm` | `1v5HE_Nm99g` | `[16, 3, 720, 1280]` | 16 |

全3本ともtraining subset、`torch.uint8`、absolute frame index `0..15`、
`sequence_length=None`、sourceとsample間で同一のevaluation referenceを確認した。
実データ動画の全長decodeは行っていない。

## Success criteriaとの対応

| spec 9章 | 確認結果 |
|---|---|
| 1–2 branch / base | `ActivityNet`、HEAD/base `cef09aa12560127451a5f569d86d5d51671e6986`を確認 |
| 3, 17–18 Core・既存互換性 | `src/sequential/`、`src/config.py`、`src/adapters/salads50.py`に基準commitとの差分なし。全回帰テスト成功 |
| 4–5 public API | 両public nameのexportとimplementation identityをテスト |
| 6–9 full JSON / primary roots / subset | schema・full件数検証、manual crawlingの無視・fallback不使用、subset選択をテストし、実データでも成功 |
| 10–13 normalization / extension / whole video / ordering | 全source inventoryと専用テストで確認。3形式の実decodeも成功 |
| 14 annotation separation | whole-video範囲、metadata、opaque reference、人工動画のEOF読み出しをテスト |
| 15 strict integrity | missing・extra・duplicate・root mismatchの拒否をテスト。未要求subsetの不整合も拒否 |
| 16 real counts | `10024 / 4926 / 5044`を新Adapterで確認 |

## 変更版の識別と実行範囲

remote `ActivityNet` branchへ反映済み。

- implementation commit: `19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`
- parent / approved base: `cef09aa12560127451a5f569d86d5d51671e6986`
- remote branchはbaseから1 commit aheadで、変更対象はAdapter / public export / testsの4ファイルのみ。

ユーザーがremote反映後にActivityNet専用22 testsと全183 testsを再実行し、いずれもOKを確認した。
さらに実ActivityNetを新Adapterから列挙し、`training=10024`, `validation=4926`, `testing=5044`, `total=19994`を再確認した。

ViT・LoRA・SSL学習、全動画decode、長時間runは実行していない。
本記録はAdapterと既存Coreの接続成立を示し、学習効果やdownstream性能を示すものではない。

検証した実装・テストのSHA-256:

| file | SHA-256 |
|---|---|
| `src/adapters/activitynet.py` | `7b50cbd3ff68c94bca27162930e9eb32e5b378cc5f9c2bd74c4ba4aa16c5c0dc` |
| `sequential_loader/__init__.py` | `5e68dff4b41f74322b09c22deeb2ca5b1261de458f0964dfc29f7645b3ff4a9c` |
| `tests/test_activitynet_adapter.py` | `e83d055ee0e8e5fe978ff4a85fbb7d914b532d3fc7918235190997c13f1c8cae` |
| `tests/test_public_api.py` | `509785dca9fc02ad45d00058b0bc84c1f046ea59e670abbe23abe9b30c1e19ce` |
