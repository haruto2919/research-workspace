---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: completed
created: 2026-10-20
last_updated: 2026-10-20
spec: ../specs/2026-10-20-meta-moco-a3-a4-a5-integrated-execution-spec.md
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_base_commit: dc502e416aa26ed4e0791767fa5a72f7843e399d
implementation_committed: false
implementation_new_files:
  scripts/parity/run_moco_gates.py: 8bf2cedc6ad5699986411f59dd670d4118110f59bbedccf77a0c22abfc280574
  test/model/test_moco_gate_launcher.py: 70e9195ba0db2487176288683d6995549c9e2bee490553638c438b685f77f83b
existing_files_changed: none
engineering_cpu_gate: PASS
gpu_gates: {A5-pilot: NOT_RUN, A4: NOT_RUN, A3: NOT_RUN, A5-pair: NOT_RUN}
reference: facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
reference_clone: /mnt/HDD12TB-1/ishikawa/facebookresearch-moco
---

# A3・A4・A5段階ランチャー：実装とCPU検証

2026-10-20のユーザー依頼「このスペックに従って実装してください」に基づき、[A3/A4/A5統合実行spec](../specs/2026-10-20-meta-moco-a3-a4-a5-integrated-execution-spec.md)の§4・§8で承認された**最小共通ランチャー**と**CPU dry-run／validation tests**を研究コードへ追加し、CPUで検証した。

**GPUは使っていない。** A5-pilot・A4・A3・A5-pairのGPU Gateはすべて`NOT_RUN`。研究コードの追加は**未commit**で、commit・push・Comet送信・checkpoint生成は行っていない。既存ファイル（MoCo本体、runner、comparator、fixture builder、Hydra config）は無変更。

## 環境と基準

- 実装root：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`、branch `dev`
- 開始時HEAD：`dc502e416aa26ed4e0791767fa5a72f7843e399d`（specの基準commitと一致、clean）。現在もHEADは同じで、新規2ファイルだけが未追跡（`git diff`は空）
- Python 3.12.3、torch 2.14.0+cu130、hydra 1.3.7、omegaconf 2.3.1、pytest 9.1.1（`.venv`）。テストは`CUDA_VISIBLE_DEVICES=""`でCPUのみ
- 公式clone：`a419ef2`のまま、`git status`差分なし

## 追加したファイル

| ファイル | 内容 |
|---|---|
| `scripts/parity/run_moco_gates.py`（757行） | 段階ランチャー。plan（既定）／`--execute`（1 stageのみ）／`--summarize`（読み取りのみ） |
| `test/model/test_moco_gate_launcher.py`（550行） | CPU dry-run／validation tests 23件 |

新しい依存関係はない。既存のfixture builder・runner・comparatorを**subprocessで1つずつ呼ぶだけ**で、処理を書き直したりコピーしたりはしていない。

## 実装した契約（spec §4・§8との対応）

### Stageと呼び出すCLI

| Stage | 呼び出し（すべて既存CLI） |
|---|---|
| A5-pilot | `moco_imagenet_parity` S1 `run_kind=bench gate=A5`、20step、`monitor.enabled=false`、warmup10/window10 |
| A4 | `build_moco_parity_fixture fixture.kind=synthetic` → S1 3step（`checkpoint_steps=[2]`、`trace.full_steps=[3]`）→ 同じcheckpointから再開するfresh run（3stepまで）→ `compare --pair-kind formal_resume` |
| A3 | synthetic fixture → M（`meta_reference`）とS1（`meta_einsum`）各1025step、`trace.full_steps=[1,1024,1025]` → `compare --pair-kind formal_implementation` |
| A5-pair | S1 bench 100step OFF → 100step Light（scalar10・grad norm100・update norm0・first/last true） |

全runの共通条件：S1は`implementation=self moco.info_nce_impl=meta_einsum diagnostic.info_nce_impl=null`、`logging.comet_enabled=false`、runごとに新規の`hydra.run.dir`。A3/A4のrunとcompareには`verification.threshold_status=fixed rtol=1e-05 atol=1e-06`を明示する。spec §3の固定条件（ResNet50+MLP128、K65536、local32×2GPU NCCL、FP32/AMP OFF/TF32 OFF、seed0、SGD lr0.0075・momentum0.9・wd1e-4、epochs10 cos、workers8、Comet OFF）は、Hydraで合成した計画configと各runの`resolved_config.yaml`の両方で照合する。

### Fail-closed（`--execute`で、何も書き込まず起動もせずにBLOCKED）

- stageディレクトリが既に存在する（再利用・上書きしない）
- 研究repoがcleanなcommitでない／公式cloneがpinned commitでcleanでない
- 計画configがspec値から外れている（yaml既定値の変更も検出）、InfoNCE modeが想定と違う、A3/A4の閾値がfixedでない
- 直前のstageが同じroot・同じ研究commitでPASSしていない（A4←A5-pilot、A3←A4、A5-pair←A3）
- `CUDA_VISIBLE_DEVICES`がちょうど2枚を指していない、または`nvidia-smi`でそのGPUにcompute processがいる（各GPU runの直前にも再確認）
- rootが研究repo内にあるのにgit-ignoreされていない／パスにHydra override値として危険な文字が含まれる
- A5のImageNet `train`ディレクトリが読めない

### 実行中の停止と判定

- 各stepの終了コードと事後検査（fixture manifestとhash、runの完了・config・mode・commit・step範囲・within-run checks、bench windowとVRAM、A4のcheckpoint native＋両rank sidecarとidentity、resume記録、A3の`queue_ptr`系列、A5 OFF/Lightの差分が`monitor.enabled`だけであること、comparatorの判定）を確認し、最初の失敗で停止する。後段stepは`not_run`。条件の変更やretryはしない。
- 判定は`PASS / FAIL / INVALID / BLOCKED / NOT_RUN`。A5-pairのmonitor overheadは`MEASURED`（OFF・Light・差分・相対%をrank別に記録）で、許容上限が未固定のためPASS/FAILにしない。
- A3のQueue一巡は、`queue_ptr`がstep sで`64·s mod 65536`になること（step1024で0、1025で64）と、within-run checks（書き込み列以外不変・ptr前進）のPASSから判定する。新しい数値比較規則は加えていない。
- `--summarize`はrun dirから読み直すだけ（何も起動しない、raw artifactを変更しない）。ランチャーが書くのは`stage_manifest.json`、`logs/`、`stage_summary.{json,md}`、`combined_summary.{json,md}`のみ。

## Spec内での解釈（実装時に決めた点）

1. **前段条件は採用順序の直列**：spec §2「A5 pilot→A4→A3→A5 paired…いずれかの工程で止まっても後続を無許可に流さず」に従い、各stageは直前stageのPASSを要求する。このため**A3がFAILするとA5-pairもこのランチャーからは起動できない**。その場合に進めるかどうかは新たな判断事項になる。
2. **A4のfull trace stepは`[3]`**：formal比較の対象step（step3）でoptimizer・Queue・BN・parameterの全tensorを比較するため。両runで同じ値。
3. **Comet OFFは全stage**：specが明示するのはA5のみ。A3/A4もA1/A2正式runと同じくOFFにした（外部送信をしない）。
4. **fixtureはstageごとに新規生成**：A4とA3はそれぞれ自分のstageディレクトリにsynthetic fixtureを作る（中身の初期stateは同一、ファイルhashは別）。
5. **GPU競合の判定は`nvidia-smi`のみ**：ランチャー自身がCUDA contextを作ってA5のVRAM計測を乱さないため。indexは`nvidia-smi`の番号として扱う（RTX A4500×2で`0,1`なら全GPUを指す）。
6. **研究commitの一致**：stage内の全run（およびfixture）の`provenance`が、stage開始時のclean commitと一致しなければINVALID。前段stageとcommitが違えばBLOCKED。したがって**ランチャー自体をcommitするまで`--execute`は通らない**（現在は未追跡ファイルのためdirty）。

## CPU検証

| 検証 | 結果 |
|---|---|
| 新規 `test/model/test_moco_gate_launcher.py` | **23 passed**（約15秒） |
| ミューテーション確認（意図的にバグを入れ、該当テストが落ちることを確認後に復元） | 5/5で検出：出力衝突検査の削除、A5-pair OFFのmonitor変更、Queue turnover検査の無効化、前段PASS検査の無効化、失敗時の停止の無効化 |
| 既存 `test/model/test_moco_parity_harness.py`（非回帰） | REGRESSION_RESULT |
| `gpu_check`をこの機械の`nvidia-smi`で実行 | `CUDA_VISIBLE_DEVICES=0,1`はissueなし（compute process 0件）、`0`と未設定は拒否 |
| 4 stageのplan（既定動作） | rootを作らず、上表どおりのコマンドを出力。研究repoが未追跡ファイルでdirtyなことを正しくBLOCKED理由として表示 |
| flake8（max-line-length 120） | 命名系プラグイン（FNE、既存コードにも同種あり）以外は指摘なし |

新規テストの内容：

- 4 stageの計画が既存3 CLIを正しいmodule・spec固定条件・独立した出力先で呼ぶこと、stage固有条件（A5のbench・monitor、A4のcheckpoint/resume・formal_resume・閾値、A3のM/S1差分が実装とmodeだけ・1025step・full trace・formal_implementation）。
- 計画したoverrideを既存CLIが受け付けること：runnerの`validate_config`、fixture builderの`main()`、comparatorの`main()`（argparse）。
- 既定モードがplanのみで、`scripts.parity`のsubprocessも`nvidia-smi`も呼ばず、何も書かないこと。`all` stageが存在しないこと。
- `--execute`の拒否8ケースで、ファイルを1つも作らずlaunchもしないこと。
- `gpu_check`：2枚ちょうど・UUID指定・1枚・未設定・存在しないindex・使用中・`nvidia-smi`失敗。
- 実行系：3 CLIをin-processで動かし、**fixture builderとcomparatorは本物の`main()`、runnerは本物の`moco_imagenet_parity.run()`（集計・artifact書き出し・checkpoint manifest）**を通す。fakeにしたのはrank worker（モデル計算なし）、fixture tensor、traceの数値比較だけ。この上で4 stageが順にPASS、run失敗でstage停止＋後段stage BLOCKED、`queue_ptr`異常でA3がMの直後に停止、前段とのcommit不一致、stage内のGPU拒否でBLOCKED、`--summarize`がartifactを変えずに再判定、15種のartifact改変（ptr・mode・within-run・compare状態/閾値・checkpoint hash・resume記録・fixture hash・artifact欠損・config・dirty・DataLoader・window・完了step・中断manifest）がそれぞれ想定のFAIL/INVALIDになること。

## 使い方（GPU実行は各Gateの別承認後）

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
ROOT=log/moco_imagenet_parity/gates/<campaign>        # git-ignored。4 stageで共通
python -m scripts.parity.run_moco_gates A5-pilot --root "$ROOT"            # plan（何も起動しない）
# ランチャーをcommitしてcleanにした後、Gateごとの実行承認を得てから：
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.run_moco_gates A5-pilot --root "$ROOT" --execute
python -m scripts.parity.run_moco_gates --root "$ROOT" --summarize          # 読み取りのみ
```

A3は各1025stepのparity run 2本（毎step trace保存）で長時間になるため、tmux等で実行する。

## 未実施・この記録で言えないこと

- **GPU Gateは全てNOT_RUN**。ランチャー経由で実GPU・実ImageNet・実公式cloneを使ったrunは一度も起動していない。実runでのartifact形式との整合は、本物の`run()`集計を通したCPUテストで確認したが、rank workerはfakeである。
- A3の実行時間・ログ容量の実測見積り（spec §2.4）は未実施。planは空きディスク容量（`disk_free_gib`）を記録するだけで、閾値は設けていない。
- 研究コードのcommit・push、Research Workspaceのcommitは行っていない。
- A3/A4/A5のPASSは、この記録の時点では何も主張しない。
