---
project: sequential-video-lora-analysis
record_type: implementation-verification
created: 2026-10-06
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_base_commit: 65bf7b0181530c13d7f1db22844fc0324e6f7661
related_spec: ../specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md
---

# Full MoCo / Linear Probe 設定のHydra一元化：実装・短時間検証

## 経緯

2026-10-06、`moco_checkpoint.py`、`streaming_moco_full.py`、`extract_features.py`、`run_probe.py` などに
パス・ハイパーパラメータ・形状が直書きされている点をレビューし、ユーザー指示
「問題がある部分を修正する。基本的にはHydraで管理する」に基づいて修正した。

レビューで問題とした点:

1. Linear Probe の記録用 `HYPERPARAMETERS` と `train_probe` の既定値が二重管理で、記録値と実行値がずれ得た。
2. MoCo の設定（momentum、temperature、Queue、optimizer、LoRA）がモデル本体・学習・snapshot検証・監査に重複していた。
3. `ViTFrameEncoder` は任意の checkpoint を受け取れる一方、下流は feature size 768 に固定していた。
4. `FRAMES_PER_CHUNK` などの定数と、shape検証のリテラル（16、128）が重複していた。
5. manifest の dataset root の絶対パスが同一性判定に入り、同じデータを別の場所に置くと再利用できなかった。

実装は2回のセッションに分かれた。前半は別のAIエージェントが行い、利用上限で中断した。
後半（本記録の作成時）は、残っていた「設定値と実際の処理がずれ得る箇所」を修正し、検証した。
変更は未コミット。学習・GPU実行・commit・pushは行っていない。

## 実装内容

### 設定構成

- 科学設定の config group: `activitynet/v1_3`、`encoder/vit_base_patch16_224`、`sequential/default`、
  `moco/stage6b_v2`、`linear_probe/lp_v1`、`provenance/research_v1`。
- 運用設定の group: `tracking/default`（Comet project 名、artifact 名）。科学的な同一性には含めない。
- 入口の root config: `moco_full.yaml`、`linear_probe_manifest.yaml`、`linear_probe_features.yaml`、`linear_probe_run.yaml`。
- 4つの production CLI を argparse から `@hydra.main` へ移行した。パス・ID・seed・device は `runtime.*`、Comet の無効化は `logging.disable_comet` で指定する。
- コア処理は OmegaConf を持ち込まず、frozen dataclass（`training/moco_config.py`、`evaluation.linear_probe.ProbeConfig`）や plain な値を受け取る。
- ライブラリ側に残る `FEATURE_SIZE` などの定数は、同じ YAML から読み込む互換用の別名にした（値の二重定義はしていない）。

### 記録値と実行値の一元化

- Linear Probe は、実行と metadata 記録の両方に同じ `ProbeConfig` を使う。`train_probe` の個別上書き引数（`epochs`、`lr` など）は削除した。
- AdamW の `betas` / `eps` を MoCo と Probe の両 preset に明示した。値は torch 2.14 の既定値 `(0.9, 0.999)` / `1e-8` と同一で、学習の挙動は変わらない。
- 実装していない設定値は実行前に拒否する（特徴定義の変更、scheduler、class weight、AdamW 以外の optimizer など）。特徴定義は Hydra から上書きできるように見えても抽出処理は参照していなかったため、`segment_features.FEATURE_DEFINITION` との一致を必須にした。
- モデルから決まる値は実物と照合する: `feature_size` は ViT の `hidden_size` と、`image_size` / `channels` は processor の出力形状と照合し、不一致なら停止する。

### Production 判定

- 科学設定の group が repository の canonical preset と完全一致する場合だけ production とする。Hydra override で科学設定を変えた成果物は `production: false` として記録する。
- canonical な production は、従来どおり untracked file を含めて clean な checkout を要求する。non-production の smoke は dirty な checkout でも実行できる。
- `linear_probe.probe.epochs=1` のような短時間確認は、科学設定の override として自動的に non-production になる。

### 成果物の同一性

- dataset root や artifact のローカルパスは `locations` に記録するが、内容同一性の hash には含めない。
- manifest / feature / result の metadata に、科学設定の canonical JSON の SHA-256（`science.sha256`）を記録し、下流では一致を検査する。
- Feature ID と Probe result ID は内容の hash から自動生成する。Probe 結果は `log/linear_probe/results/<result_id>/` に分け、設定の異なる結果が衝突しないようにした。

## 互換性への影響

- schema: manifest `v1 -> v2`、segment features `v2 -> v3`。MoCo の snapshot / resume schema は `v2` のまま。
- MoCo snapshot の metadata に `feature_size`、`projection_size`、`frames_per_chunk`、`production_config` を追加した。`local_path` は `locations.artifact` に変えた。resume identity に dataset、stream 設定、`feature_size`、`projection_size` を追加した。
- これらのフィールドを持たない既存の v2 snapshot は、canonical な値として扱って引き続き検証できる。
- 既存の成果物は `log/smoke/` の smoke 出力（2026-10-05、Hydra 移行前のコードで作成）だけで、full run の成果物は存在しない。旧 schema の smoke manifest / feature / result は再利用されず、再生成が必要になる。
- 学習・評価の意味（dataset 順、更新順、ハイパーパラメータ値、Probe 条件、metrics）は変えていない。
- [承認済みspec](../specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md) 16章は Hydra の採否を non-blocking としつつ、artifact lineage を変えないことを求めている。本修正では、ユーザー指示に基づき Feature ID / result ID の生成方法と結果ディレクトリの構成を変えた。spec 11.3章の CLI 例（argparse 形式）は Hydra override 形式に置き換わっている。

## 検証結果

| 検証 | 結果 |
|---|---|
| プロジェクト全テスト `pytest test`（563件） | 527件成功、36件失敗。失敗はすべて既存の環境依存テストで、新規の失敗はない |
| 既存失敗36件のHEAD再現 | `65bf7b0` を `git archive` で展開した HEAD 上でも同じ36件が失敗 |
| lint 修正後の関連4ファイル再実行 | 36件成功 |
| 4つの入口の `--cfg job` | すべて合成に成功し、`betas` / `eps` を含む解決済み設定を表示 |
| flake8（E/W/F） | HEAD と比べて増えた指摘なし |
| `git diff --check` | 成功 |

既存失敗36件の内訳と原因:

- `test/dataset/test_video_folder.py` 4件: NAS 上の Kinetics400 で train と val のクラス一覧が一致しない。
- `abn_r50` を使う `test/model/test_image_models.py` 16件、`test/model/test_model_factory.py` 12件、`test/utils/test_checkpoint.py` 4件: PyTorch Hub の信頼確認プロンプト（`input()`）が非対話環境で EOF になる。

後半セッションで追加したテスト:

- `linear_probe.feature_definition` の上書きが、成果物を作る前に `extract_features` で拒否されること（CLI）。
- 実装している特徴定義だけを受け付けること（単体）。
- Probe の AdamW が、記録される `hyperparameters()` と同じ `lr` / `weight_decay` / `betas` / `eps` で作られること（単体）。

最終の全テストコマンド（実装 repository で実行）:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider test
```

リポジトリ直下で `pytest` を引数なしで実行すると、vendored の `hub/.../setup.py` がプロジェクトの `setup` パッケージを隠し、テスト収集の段階で止まる。このため対象を `test/` に限定した。

## 未実施

- 実 ActivityNet での新 CLI の smoke。canonical な設定の MoCo 入口は clean な checkout を要求するため、未コミットの状態では実行できない。
- full-dataset MoCo、full manifest / feature 抽出、6本の Linear Probe、GPU 実行。
- 本記録はテストと設定合成による実装検証であり、学習結果や表現性能について何かを示すものではない。
