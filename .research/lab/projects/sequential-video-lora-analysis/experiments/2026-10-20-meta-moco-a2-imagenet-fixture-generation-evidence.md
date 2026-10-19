---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: user-provided-terminal-evidence
status: fixture-validated-gpu-diagnostic-not-run
evidence_level: cpu-fixture-generation
gate: A2
gpu_a2: NOT_RUN
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: dc502e416aa26ed4e0791767fa5a72f7843e399d
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
source: user-shared-terminal-output
---

# A2：ImageNet実画像1step canonical fixture生成（ユーザー提供ログ）

## 結果と証拠の境界

研究サーバ上でユーザーが`scripts.parity.build_moco_parity_fixture`を実行し、CPU側の固定fixture保存に**成功したことが標準出力から確認できた**。本記録はChatGPTがSSH研究サーバを操作して得た結果ではない。**生成物ファイル・manifest内部の全field・tensor実体はGitHubから参照できていないため、まだ整合性PASSとは判定しない**。

- 自作：`git rev-parse HEAD` = `dc502e416aa26ed4e0791767fa5a72f7843e399d`、`git status --short`は出力なし（ユーザー報告）。
- Meta公式clone：`git rev-parse HEAD` = `a419ef282fc9885015674e77f58c59447206a27f`、`git status --short`は出力なし。
- 生成コマンド（ユーザー貼り付けテキストでは末尾の表示が一部崩れているため、以下は意図した主要引数の引用であり完全なシェル履歴とは主張しない）：
  - `python -m scripts.parity.build_moco_parity_fixture`
  - `fixture.kind=imagenet`
  - `fixture.imagenet_steps=1`
  - `moco.queue_size=65536`
  - `training.local_batch_size=32`
  - `data.image_size=224`は標準configの既定値だが、ユーザー貼付けの当該行は途中欠損があるため**manifestで要検証**。
- 絶対path：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/fixtures/a2_imagenet_20261020_084625/fixture.pt`
- 観測された`Saved`のSHA-256：`9edede9195eccef13e7ec8446880d6bfc6a550b11e96b466dbf047723d24b4ac`。
- `FIX_DIR`：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/fixtures/a2_imagenet_20261020_084625`。
- 既存fixture generatorの契約上、成功時に`fixture.pt`と`fixture_manifest.json`を書き出すが、**後者の現地存在・sample数・ImageNet実際のview/shape/hashの実測ログは未共有**。
- 関連実験計画：[2026-10-20 A2実験spec](../specs/2026-10-20-meta-moco-a2-imagenet-one-step-parity-experiment-spec.md)。基本方針は採用済みだがformal FP32閾値は未決定、specはdraft。

## 次に行う確認と判定

1. **ユーザー側でmanifest/pt整合をread-only確認**する。expected: `kind=imagenet`、`steps=1`、`world_size=2`、`local_batch_size=32`、`global_batch_size=64`、`image_size=224`、`queue_size=65536`、`arch=resnet50`、`seed=0`。samples64件、input_sha256が1step×2rank、view tensorが各rank (32,3,224,224)、seed/data/initial state hash整合。
2. `fixture.pt`のSHA-256が上記`Saved`値および`fixture_manifest.json.fixture_sha256`と一致し、`fixture.meta.state_sha256`がstateから再計算した値と一致することを検査する。
3. 計測されたmanifest/sanity resultを確認し、A2 fixture validを判定する。改変・上書きせず、失敗なら停止して原因を記録。
4. GPU診断への独立した次GateはNCCL/rank/VRAMの実機状態とMeta公式 M0/M0r再演（それぞれ2GPU・1step）の可否・実行指示。runした結果を正式なMeta vs S1 PASSと混同しない。
5. M0/M0rが再現した後にformal rtol/atolを明示的に固定し、別のM1/S1正式runへ進む。現時点`GPU A2=NOT_RUN`。

## 保存・環境の境界

このEvidenceにはuserからの標準出力とpath/SHAのみを保存。大容量fixture.pt、tensor、ImageNetデータセットをResearch WorkspaceのGitHubにpushしていない。GitHubへの文書保存はSSHサーバの研究コードcloneに反映されていない。

---

## 2026-10-20 08:xx JST：Fixture manifestと実tensor整合性の追試（ユーザー提供ログ）

ユーザーは固定fixtureディレクトリを再度開き、`fixture_manifest.json`を検査した上で`torch.load(..., weights_only=True)`で保存済み`fixture.pt`をCPU上に読み込み、`scripts.parity.moco_parity_state`の`file_sha256`・`state_sha256`・`tensor_sha256`を使って検査した。ユーザーが入力したコマンド表示の一部はターミナル貼付けで崩れているが、最終出力は **`FIXTURE VALIDATION PASS`** だった。以下をコンソール出力と元のassert群が成功した範囲に限定して受け取る。

| 確認項目 | 実測/期待一致 | 判定 |
|---|---|---|
| kind / steps | `imagenet` / `1` | PASS |
| GPU world / local / global batch | `2` / `32` / `64` | PASS |
| Image size / Queue size | `224` / `65536` | PASS |
| Architecture / seed | `resnet50` / `0` | PASS |
| ImageNet sample count | `64`件 | PASS |
| rank 0・1 のinput 2view | 各rankの`[32,3,224,224]`×2、FP32、finite、SHA一致のassert通過 | PASS |
| fixture.pt SHA | `9edede9195eccef13e7ec8446880d6bfc6a550b11e96b466dbf047723d24b4ac`（生成時出力・manifestと一致） | PASS |
| Canonical state hash | 保存stateを再計算したSHAとmanifestの比較assert通過 | PASS |

**Evidence判定：`A2 fixture validation = PASS`。** ただしこれは「固定画像/初期stateの単一fixtureが自己整合的である」ことの確認であり、Meta公式 M0/M0r のGPU再演、独立M vs S1、NCCL実通信、localB32でのVRAM可否、A2正式判定を示さない。raw fixture全体をGitHubから直接読み込んだわけではない。

### 次Gate

まずGPU使用状況・研究コード/公式clone clean・保存先空き容量を確認し、短い`M0=meta_reference` 2GPU/NCCL 1step `run_kind=parity gate=A2`を**診断run**として別途実行する。成功時だけ独立fresh `M0r`を実行、CPU offline `--pair-kind rerun`で再現性を調べる。両runの`verification.threshold_status=provisional`であることから、比較結果は正式A2 M vs S1 PASSではない。GPU OOM/不整合なら自動でlocalB/K/image/precisionを変えず停止・報告する。
