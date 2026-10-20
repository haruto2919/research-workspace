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

---

## 2026-10-20 再監査：dev@d6f7acc（ユーザー実装再報告後）

対象research code：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、HEAD **`d6f7acc75ee92adcf7e6e3e65831d46759b9ed14`**。直前の`3cade3dc9c6e474e0c81595690cbee5a494e4cb2`から**1commit ahead**（`fix(parity): restore the 10-epoch runner config and use the repository .comet.config`）。以下を旧監査の修正・追補として参照する。**旧F1のmissing設定ファイル指摘は最新HEADでは解消済み**。

### 最新差分（直接確認）

- 新規`conf/moco_imagenet_10epoch.yaml`（91行）。`dev`のGitHubに存在。Hydraのrunner指定と整合。2GPU×local128/global256、LR0.03、10epoch Cosine、FP32、ResNet50 MLP、K65536、workers16/rank、入力hash100step、V1 1GiB以上headroom、Comet workspace/project target、Hydra run出力dir、B2 strictモードを設定。
- `test/model/test_moco_imagenet_10epoch.py`は`COMET_CONFIG`を`ROOT / ".comet.config"`へ修正し、module import時の`conf/moco_imagenet_10epoch.yaml`読み込みが参照可能。20+ test function相当のCPU/fixture/rank/Cometモック/V1判定ケースがあるが**実行したとは主張しない**。
- Git追跡`.comet.config`は`[comet] workspace`・`project_name=moco-imagenet-v2-reproduction`を記載。直接確認した内容では実値`api_key`行は存在しない。秘密情報は転載しない。SDKがhomedirのキーを正しく解決するかは実機preflightとモックテストの検証対象。
- `conf/comet/moco_imagenet_v2_reproduction.comet.config`は最新HEADでも存在しないが、**現行runner/testsはroot`.comet.config`に変更しているため今や要求ファイルではない**。旧F1のこの欠損指摘も解消。
- 新たに追加したconfigは、A2/A3の既存`conf/moco_imagenet_parity.yaml`を変更しない。旧parity runnerの数値/evidenceは独立。

### 現時点の状態判定

| 観点 | 判定と理由 |
|---|---|
| Source shape/entrypoint | **静的に存在確認**。`scripts/parity/moco_imagenet_10epoch.py:844–850`の`@hydra.main(config_name="moco_imagenet_10epoch")`と新規YAMLが一致 |
| Adopted recipe | **静的に整合**。新configの2GPU/global256/local128、MoCo v2の各値が`ADOPTED_RECIPE`に一致 |
| Epoch loop & checkpoint | **静的に存在確認**。`train_epochs:508–658`でSampler epoch, LR, metrics/Comet, checkpoint/resume sidecars |
| Comet destination | **静的に存在確認**。`comet_preflight:195–236`でproject名が配置と整合、オンラインAPI権限は実機未確認 |
| CPU tests | **結果未受領**。GitHub HEADのcheck-runs/statusは本解析時に0件（非実施の断定ではない） |
| GPU V1, E1, E2, E3 | **GitHubから未確認**。特にlocal128/FP32でVRAM不足が起きない証拠は未受領 |
| E4 lightweight held-out feature diagnostics | **このcommitの新規runner/comparatorには未実装**。研究主張は学習挙動再現まで、B4で採用した診断は別実装/評価Gateが必要 |

### 残る重要論点（勝手にapprovedへしない）

1. **Authority mismatch**：10epoch specは依然`status:draft`で、**B1/B2/B3残部/B5未承認**。runner自身のdocstringやconfigコメントに「2026-10-20採用」と書いてあることは研究上のユーザー承認を証明しない。特に**B2：ユーザーは公式設定を希望したが、現在のcodeは`cudnn_deterministic=true,cudnn_benchmark=false,TF32 off`のstrict parity側**。Meta公式`main_moco.py`のbenchmark=trueと異なる。両runは揃っており比較に適するが「完全に公式8GPU実行のruntime flags」とは言えない。ユーザーが選択する必要あり。
2. **Precondition bypass**：`prepare_config:98–100`はV1/E1/E2/E3を列挙するが、E3の実行前に既存V1の正式PASS reportをrequireする独立チェックは未実装。**V1→E1→E2→E3の順を実行者が保証する必要**。V1 OOM時に設定を自動劣化させない契約は保持。
3. **Comparator nuance**：`compare_moco_10epoch.py:118–150`は`both_completed`と`inputs_identical`を`curve_agreement`と別に報告し、総合Gateの完全PASSを単一値で表現しない。full finished、input alignment、metric/canonical stateを併記して判断。CLI比較器だけを回しても代表的な学習成功の根拠ではない。
4. **Common input alignment**：epoch/rank seedによる拡張画像一致はコード上の意図であり、独立したM/S1実GPUで`input_sha256`を比較するまで検証済みではない。必要なら比較器の`inputs_identical`がTrueか確認。
5. **Comet shared root configuration side effect**：Git追跡root`.comet.config`のデフォルトprojectを変えたため、同リポジトリから実行する他のComet SDK利用も専用MoCo projectへ送信し得る。configコメントでも示唆されている。B3/既存baselineへの影響を承認前に確認。
6. **Comet online mode**：V1では`logging.comet_enabled=false`でローカルのみ検証。E1以降のComet送信は設定が読めるだけでは成功を保証しない。api keyを表示せずpreflightログ・Comet URLを確認し、送信失敗時はlocal historyと区別。
7. **Security**：root`.comet.config`はtrackedだがAPIキー行を実際には含まない。今後も`~/.comet.config`等の安全なローカル秘密管理を継続し、認証情報をgitに追加しない。

### 次の検証手順（提案、未実行）

1. SSHローカルで`git status --short`、`git rev-parse HEAD`、`nvidia-smi`、`test -f`。GitHub latestとの差分がなければ継続。
2. config-only/unit系`pytest -q test/model/test_moco_imagenet_10epoch.py -k 'hydra or repository_comet_config or comet_preflight'`。その後`pytest -q test/model/test_moco_imagenet_10epoch.py`（一部CPU DDP testは公式cloneを使うので、running時間/CPU負荷に注意）。**GitHubレビューだけではPASSとは言えない**。
3. 研究上のB1/B2/B3残部/B5の合意を明示しspecをapproved可能にする。現在のstrict B2やライブComet、16 workers等を無断でapprovedとしない。
4. 必須VRAM **V1**：Meta公式・自作を**別run**でglobal256 FP32 224px real ImageNet、複数step＋GPU毎VRAM headroom。各V1 gate statusがPASSならE1 100step→E2 1epochで性質と速度/再開を確認。E3 10epochはそれらの正式Gate PASSと別実行許可の後にのみ行う。
5. B4採用済みE4 lightweight特徴診断は別途実装/評価し、kNNを必須化しない。

**修正履歴**：本節は`3cade3d`時点のF1（missing 2 config）の判定を**最新`d6f7acc`では解消済み**へ更新する追補であり、旧レビューを消さず履歴保持。**静的レビューのみ**（repo code/README/spec/TODO/experiment/logは変更なし）。
