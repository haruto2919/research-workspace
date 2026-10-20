---
date: 2026-10-20
project: sequential-video-lora-analysis
mode: audit
detail: compact
status: read-only-static-review
code_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
code_branch: dev
code_commit: 3cade3dc9c6e474e0c81595690cbee5a494e4cb2
code_base: 914e736c2699e2e129211a888d1f19013dd1dd1a
context_spec: ../specs/2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md
context_spec_status: draft
execution: NOT_RUN
---

# 10epoch MoCo runner GitHub dev 静的監査（2026-10-20）

> 実装者から「実装しました」と報告があったため、GitHubの現在dev実装をread-only照合。**SSHサーバのdirty/unpushed code、pytest結果、CUDA/GPU、Cometの実際のWorkspace/API通信は確認不能**。本記録は静的監査であり、動作成功や本番学習のPASSを主張しない。秘密情報は記録しない。

## 解析範囲・ソース

- GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` commit `3cade3dc9c6e474e0c81595690cbee5a494e4cb2`（2026-10-20）。直前`914e736c...`からの1commit追加分を調査。
- 追加：`scripts/parity/moco_imagenet_10epoch.py`（850行規模）、`scripts/parity/compare_moco_10epoch.py`、`test/model/test_moco_imagenet_10epoch.py`。
- 変更：ルートの`.comet.config`。GitHub上の`.comet.config`で`project_name=moco-imagenet-v2-reproduction`およびworkspace値は存在し、**`api_key`実値を保持する項目は見つからなかった**。workspace名やcredential値をこの監査記録に転載しない。
- 関連既存実装：`scripts/parity/moco_imagenet_parity.py`（MetaReferenceAdapter/SelfAdapter、metric/transform/state mapping）。
- 研究計画正本：`.research/lab/projects/sequential-video-lora-analysis/specs/2026-10-20-meta-moco-v2-official-global256-imagenet-10epoch-comet-reproduction-spec.md`は**draft**（B4とComet project名は採用済み、B1/B2/B3残部/B5は未承認）。

## Architecture summary（確認済み）

- `moco_imagenet_10epoch.py`は独立Hydra entry point：`@hydra.main(config_path="../../conf",config_name="moco_imagenet_10epoch")`（lines 844–850）。
- `prepare_config`（96–149）でCUDAの2GPU/global256、local128、LR0.03、10epochs、FP32、K65536、Cosine等のrecipeを固定し、CPU self/Glooの小規模試験を例外扱い。
- `train_epochs`（508–658）でMeta公式と自作の既存adapter・同一fixtureの初期重みを使い、epochごと`sampler.set_epoch`、SGD/Cosine、rank0のメトリクス、epoch checkpoint + rank sidecar、入力hash、local JSONL保存を実装。
- `comet_preflight`（195–236）で`logging.comet_config`からComet SDK設定を解決し、期待destinationとの一致や認証の有無を検証。rank0の`CometLive`（342–405）で別Experimentを作成する想定。エラー時に送信を止めlocalログを残す。
- `compare_moco_10epoch.py`（34–150）は共有identity、dataset/sampler、common steps、metrics、input hash、一epoch後canonical stateを比較してreportを保存。
- `test/model/test_moco_imagenet_10epoch.py`はtest本体のほか設定ファイルを**module import中に**読み込む（lines 32–42）。

## Findings

### F1 — BLOCKING（Confirmed by code+GitHub tree）：必須config 2件がcommitに存在しない

- `moco_imagenet_10epoch.py:844`が参照する`conf/moco_imagenet_10epoch.yaml`が`dev@3cade3d`のGitHubに存在しない。
- テスト`test/model/test_moco_imagenet_10epoch.py:33–37`も同じCONFIGをモジュール読込時に`OmegaConf.load(CONFIG)`する。GitHub commitだけで再現する場合、pytestの**収集段階でファイル欠落エラーが予想される**（未実行）。
- `test/model/test_moco_imagenet_10epoch.py:35,117–125`とrunner `comet_preflight:205–208`が期待する`conf/comet/moco_imagenet_v2_reproduction.comet.config`もGitHub上に存在しない。Cometが有効なrunでは設定が見つからずpreflight失敗が予想される。
- 対策候補：SSHローカルにファイルが存在するか、未追跡/unpushedでないかを確認し、**認証情報を含まないconfigだけをコードcommitへ反映**。秘密APIキーはホーム側`~/.comet.config`等の管理を維持する。実行・pushは別の明示依頼が必要。

### F2 — BLOCKING for approval（Confirmed from spec+code）：未承認条件B2/B3/B5を実装上で固定

- runnerヘッダ lines20–25で「2026-10-20 decisions」として`cudnn deterministic=True/benchmark=False/TF32 off`、epoch-rank seedでのinput alignment、Cometライブ送信、各GPU16 workers/epoch checkpoint retentionを採用したと説明。
- Research Workspace specは**B2/B3残部/B5が未承認のdraft**。実装は存在するが、`approved`としての実行許可や科学的合意を推定しない。実際の「完全同一view tensor再現」は入力hash比較と数step実測で確かめる必要があり、seedだけで保証したと断定しない。
- 対策候補：現行実装の採用条件をユーザーが確認し、必要なら仕様とコードの差分を承認済みspecへ反映してからゲート実行。

### F3 — IMPORTANT（Potential issue）：V1 PASSを経ずE3が起動可能

- `prepare_config:98–100`はgate名を制約するが、`run:776–841`には先行V1の`PASS` artifactを読み込み必須化するチェックがない。V1の`vram_gate`判定（708–727）は同じV1 runの結果をreportへ保存するだけ。
- specではV1 real ImageNet local128/global256/FP32が必須で、NGなら長時間10epoch開始を停止する。現runnerのE3直接起動ではこの研究上の実行Gateが**自動強制されない**。
- 対策候補：V1 PASSのreportを起動前に手動検証する運用、または承認済みspecに沿った明示preconditionを検討。**E3を直接起動しない**。

### F4 — IMPORTANT（Potential issue）：比較器の見出しは総合Gateとは別

- `compare_moco_10epoch.py:118–150`の`curve_agreement`の判定は主に`metric_bitwise`・`epoch canonical state`・LR/contractに依存し、`both_completed`や`inputs_identical`は別フラグ。**一部メトリクスの一致を実画像10epoch全体のparity PASSと取り違えない**。同一視する場合は統合合否の追加契約が必要。
- kNN/Linear Probeは採用済みB4により任意。固定held-out特徴量の軽量診断は別Gateであり、今回の追加commitには明示的なE4特徴診断実装は確認できない。

## Failure modes / testability gap

1. **初手はコード変更前のlocal照合**：`git status --short`、`git rev-parse HEAD`、`test -f conf/moco_imagenet_10epoch.yaml`、`test -f conf/comet/moco_imagenet_v2_reproduction.comet.config`で実機ファイルの有無を確認（中身やAPIキーを共有しない）。
2. **設定とtestsがGit上で揃った後にCPU試験**：`pytest -q test/model/test_moco_imagenet_10epoch.py`。GitHub Actionsのhead commitにrun/check記録は現時点では0件、したがってPASSEDとは主張しない。
3. **Comet preflight**：意図したworkspace/projectにのみ繋がることを確認。APIキーを出力しない。root`.comet.config`は現在Git追跡されているが秘密キーは含まないように保つ。
4. **V1/E1/E2/E3**：別承認、2GPU実機のlocal128でVRAM/NCCL/データ待ちの実測が必要。学習runの有無・結果・Comet到達・Checkpoint復帰のGPU版はGitHubから直接検証できない。

## 未確認点

- SSHローカルにGitHub未pushの2設定ファイルが存在するか。存在するならローカルのみで動いている可能性はあるが、commit単体の再現性は欠く。
- 実際のCUDA GPUの空き/peak VRAM、10epoch試験・Comet projectの作成・権限/通信・入力hash一致、tests実測、held-out特徴診断結果。
- 10epoch全体の完了、Loss曲線の一致、画像表現の品質。すべて未証明。

**結論**：GitHubに専用runner・comparator・テストの実装は確認。ただし新しいconfig2件がcommitにないため、**現在のcommitを完全実装済み・再現可能とは判定できない**。設定とspec approvalの整合を済ませ、必須V1を通した後に段階的な実験へ進む。コード・spec・README・TODO・GPU/Cometはいずれも本監査では変更しない。
