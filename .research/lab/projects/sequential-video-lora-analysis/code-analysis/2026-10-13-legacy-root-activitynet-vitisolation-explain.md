---
project: sequential-video-lora-analysis
date: 2026-10-13
analysis_mode: explain
detail: compact
level: mid
source_repository: https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_commit: b6385d87e2e3e82a719d8f4b686b44aa293b1135
comparison_commit: 65dedb4ebc26c48d23606b9bc674044f64e64c06
source_dirty: unknown
targets:
  - audit_activitynet_inventory.py
  - sequential_vit_bridge.py
  - smoke_50salads_vit_bridge.py
  - smoke_activitynet_vit_clip_feature.py
  - smoke_vit_frame_encoder.py
related_spec: null
---

# Code Deep Dive: 旧ルート直下のActivityNet監査・ViT bridge・smoke 5ファイルの必要性

## 対象と解析の範囲

- 目的: 現在計画している **Sequential Loaderなし・CIFAR-10実画像でのLoRA単体検証**に、ユーザー指定の5ファイルが必要か判定する。削除可否は過去の回帰テストや今後のSequential Loader統合と分ける。
- 対象ソース: `tamaki-lab/2026_09_ishikawa_sequential-video-lora` GitHubコミット `b6385d87e2e3e82a719d8f4b686b44aa293b1135`（user報告のローカルencoder-only基準）と、GitHub `dev@65dedb4ebc26c48d23606b9bc674044f64e64c06`。
- 補助確認: `test/model/test_50salads_vit_bridge.py`, `test/model/test_activitynet_vit_clip_feature.py`, `test/model/test_vit_frame_encoder.py`, `scripts/smoke/*`, `integration/sequential_vit.py`。
- 除外: SSHサーバの実際の`encoder-only`未コミット変更、実行時importの成否、GPU・データセット実体。GitHubからは確認不能。

## Overview: What / Why

| 旧ルートパス | コード上の役割 | CIFAR10+LoRA単体検証で直接必要か | 削除時の注意 |
| --- | --- | --- | --- |
| `audit_activitynet_inventory.py` | 動画ファイルを複数ディレクトリから収集し、annotation JSONのIDと照合。欠損/重複/余分な動画とsplitごとの本数を報告する独立スクリプト。 | いいえ | ActivityNetデータ監査には有用。LoRA導入と無関係。 |
| `sequential_vit_bridge.py` | `sequential_loader.SequentialSample` のvalid frameをprocessorとViTへ渡し、`[T,768]`をpadding位置0で再構成する`encode_chunk`。 | いいえ | **旧baselineの2つのsmokeやtestに依存あり**。将来loader統合用。 |
| `smoke_50salads_vit_bridge.py` | 50Salads train1の初回16-frame chunkを古い `feature-50salads-loder` branch条件で検証。 | いいえ | 旧 `test/model/test_50salads_vit_bridge.py` が`from smoke_50salads_vit_bridge import encode_chunk`しているため、現状で削除するとtest importが失敗する。元のsmoke本体は`sequential_vit_bridge.encode_chunk`をimportして再exportしている。 |
| `smoke_activitynet_vit_clip_feature.py` | ActivityNet trainingの先頭chunk取得 → frozen ViT frame feature → MaskedMeanClipAggregator → clip feature [768]を検証。ブランチは旧`dev`を期待。 | いいえ | 旧`test/model/test_activitynet_vit_clip_feature.py` がスクリプトをimportし`main()`をmockで検証。削除すると既存testが壊れる。 |
| `smoke_vit_frame_encoder.py` | 通常RGB画像1枚にAutoImageProcessorを適用し、frozen CLS `[1,768]` と有限性、trainable param 0を確認。 | LoRA学習には不要（事前確認には有用） | 既存 `test/model/test_vit_frame_encoder.py` は`ViTFrameEncoder`を直接テストしており、このsmokeをimportしない。画像単体の手動baseline smokeを失うので保管推奨。 |

## Key Components and Data Flow

- `audit_activitynet_inventory.py` は `argparse`, `json`, `pathlib` などのみをimportし、ViT/Sequential Loader/LoRAへ依存しない。コマンドで手動実行するデータ監査（通常学習の実行時経路ではない）。
- `sequential_vit_bridge.encode_chunk(sample,processor,encoder,device)` は `valid_mask` でpaddingを除外し、ViTにvalidのみ渡してから、元chunkの各時点に対応する行へ特徴量を配置。元のframe順・mask対応を守るための補助。
- `smoke_50salads_vit_bridge.py` は `import sequential_loader as sl`, `from sequential_vit_bridge import encode_chunk`, さらに`EXPECTED_BRANCH="feature-50salads-loder"`と古いLoader revisionの検査を含む。現在の`encoder-only`で実行できることを保証しない。
- `smoke_activitynet_vit_clip_feature.py` も `sl.ActivityNetAdapter`, `encode_chunk`, `MaskedMeanClipAggregator` を使い、`EXPECTED_BRANCH="dev"`、Loader commit固定、`torch.no_grad()`を利用しており、LoRAを学習する処理はない。
- `smoke_vit_frame_encoder.py` は任意の1枚の画像について `encoder.eval()` と `torch.no_grad()`で **推論** を監査する。LoRA A/Bの勾配・optimizer更新を検証する役割はない。

## 直近devブランチとの差（削除と移動を混同しない）

GitHub `dev@65dedb4...` では旧ルート直下の4スクリプトは見つからないが、**以下に同名の機能が移設**されている。

- `audit_activitynet_inventory.py` → `scripts/smoke/audit_activitynet_inventory.py`
- `smoke_50salads_vit_bridge.py` → `scripts/smoke/smoke_50salads_vit_bridge.py`
- `smoke_activitynet_vit_clip_feature.py` → `scripts/smoke/smoke_activitynet_vit_clip_feature.py`
- `smoke_vit_frame_encoder.py` → `scripts/smoke/smoke_vit_frame_encoder.py`
- 旧 `sequential_vit_bridge.py` はルートに残るが、`integration/sequential_vit.py`へ繋ぐ互換import（`sys.modules[__name__] = _implementation`）。`dev`の`test/model/test_activitynet_vit_clip_feature.py`は互換importを利用し、`test/model/test_50salads_vit_bridge.py`は`integration.sequential_vit`から直接読み込む。
- よって、**rootにない≠機能削除済み**。ローカル`encoder-only`がb6385d8から派生しているか、移動が取り込まれているかは要確認。

## Concepts and Design Decisions

- `smoke`は初期統合の最小確認、`audit`はデータ整合性の独立確認、`bridge`はモデルとLoaderの接続処理本体。`smoke`や`audit`をモデル学習に必須なruntime依存と同一視しない。
- 現段階のCIFAR-10 LoRA検証は`dataset/cifar10.py`通常DataLoader → ViT + selectable Q/K/V LoRA → CLS [B,768] → classifier head → CE + optimizerを想定。上記5ファイルのimportは必要としない。
- ただし、旧テストは`smoke_50salads_vit_bridge.py`と`smoke_activitynet_vit_clip_feature.py`をimportしているため、削除するとpytestの収集段階で失敗し得る。また`sequential_vit_bridge.py`の削除もこれらのtestやsmokeを壊す。
- `smoke_vit_frame_encoder.py`は既存ViTの単画像baseline推論に便利だが、LoRAのBackward/更新検証用smokeは新設が必要。
- `dev`の整理方式（`scripts/smoke/`、`integration/`）が既に存在するため、新しいspecでファイル整理をするならcurrent branch/commitの実体と移行方針を確認して二重配置を避ける。

## Alternatives and Trade-offs

**最小・非破壊案（推奨）**:
- 今回の新しいLoRA-only学習経路から5ファイルを依存させない。
- 既存ファイルはすぐに削除せず、既存テストと将来の統合資産として残す。
- 整理が必要なら別変更として実施し、import参照、実行スクリプト、テストの移動/修正、回帰テストをまとめて扱う。
- `smoke_vit_frame_encoder.py`は旧baseの健全性チェックとして流用可能。ただしLoRAの学習成立を示さない。

## 根拠・未確認事項

- **Observed**: 上記2 commitにおける5ファイル内容、4スクリプトの移設、2つの旧baselineテストのimport依存、`dev`の互換sh​​im。
- **Inferred**: CIFAR10 LoRA-onlyでこれらを経由しない方が、MoCo/Sequential Loaderを切り離す2026-10-08 MTGの意図に一致する。
- **Unverified**: ローカル`encoder-only`の現在の実体、未コミット/未push変更、全ファイルの最終的な外部依存関係、スクリプト/テスト実行可否。

## ソース

- [旧 inventory](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/audit_activitynet_inventory.py)
- [旧 Sequential-ViT bridge](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/sequential_vit_bridge.py)
- [旧 50Salads smoke](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/smoke_50salads_vit_bridge.py)
- [旧 ActivityNet clip smoke](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/smoke_activitynet_vit_clip_feature.py)
- [旧 ViT single-frame smoke](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/smoke_vit_frame_encoder.py)
- [旧 50Salads test](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/test/model/test_50salads_vit_bridge.py)
- [旧 ActivityNet test](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/b6385d8/test/model/test_activitynet_vit_clip_feature.py)
- [現dev integration](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/65dedb4ebc26c48d23606b9bc674044f64e64c06/integration/sequential_vit.py)
- [現dev compatibility shim](https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/65dedb4ebc26c48d23606b9bc674044f64e64c06/sequential_vit_bridge.py)
