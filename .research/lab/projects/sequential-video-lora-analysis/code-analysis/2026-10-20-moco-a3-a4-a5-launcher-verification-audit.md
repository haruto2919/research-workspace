---
project: sequential-video-lora-analysis
date: 2026-10-20
analysis_mode: audit
detail: compact
level: mid
source_repository: https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_branch: dev
source_commit: 914e736c2699e2e129211a888d1f19013dd1dd1a
source_dirty: unknown-on-SSH
targets:
  - scripts/parity/run_moco_gates.py
  - test/model/test_moco_gate_launcher.py
related_spec: ../specs/2026-10-20-meta-moco-a3-a4-a5-integrated-execution-spec.md
related_implementation_record: ../experiments/2026-10-20-meta-moco-a3-a4-a5-gate-launcher-implementation-verification.md
run_performed_by_this_audit: false
---

# Code Audit: A3/A4/A5段階ランチャーの静的照合と正常実装の検証手順

## 対象と解析条件

2026-10-20のユーザー依頼「実装した。正常に実装出来ているか確認する方法を教えて」に基づき、研究コードGitHubの`dev@914e736c2699e2e129211a888d1f19013dd1dd1a`を、採用済み[A3/A4/A5統合spec](../specs/2026-10-20-meta-moco-a3-a4-a5-integrated-execution-spec.md)と既存の[実装記録](../experiments/2026-10-20-meta-moco-a3-a4-a5-gate-launcher-implementation-verification.md)に照合する**read-only**監査。

GitHubのcommit差分 `dc502e416aa26ed4e0791767fa5a72f7843e399d...914e736c2699e2e129211a888d1f19013dd1dd1a`には**`scripts/parity/run_moco_gates.py`と`test/model/test_moco_gate_launcher.py`の追加2件だけ**が含まれ、既存MoCo本体、trainer、comparator、Hydra configはこのcommitで未変更。SSH研究サーバのHEAD/dirty/テスト終了状況はGitHubからは未確認。テストをこの監査で実行した事実もない。

## Architecture Summary

- `run_moco_gates.py:52-103`：A5-pilot、A4、A3、A5-pairの順序と固定条件（K65536, B64, 2GPU, FP32, Self meta_einsum, A3/A4 fixed threshold, Comet OFF）。
- `run_moco_gates.py:106-158`：各段階を既存のfixture builder、`moco_imagenet_parity.py`、`compare_moco_parity.py`のCLIへ変換。A4はStep2保存からStep3再開して`formal_resume`、A3は1025step×2で`formal_implementation`。A5はbench20step/100step×2。
- `run_moco_gates.py:191-239`：デフォルトはplanのみ。Hydraによるconfig構築、mode/threshold/git/前段Gateの事前確認。
- `run_moco_gates.py:253-334`：`--execute`はGPU2枚/compute jobs/dirty/既存出力/前段Gateなどを検査し、既存CLIをsubprocessで順に実行、最初の失敗で中止。
- `run_moco_gates.py:603-750`：既存の成果物からGate別/総合statusを再構成。判定はPASS/FAIL/INVALID/BLOCKED/NOT_RUN。A5 overheadは数値MEASUREDであり性能閾値PASSを捏造しない。

## Key Decisions and Trade-offs

- A3/A4は`parity`で詳細状態比較、A5は`bench`で通常学習速度測定。同一runに混在させない。
- A3の1025stepは毎step small traceを保存し、1/1024/1025stepでfull traceが必要。GPU時間・ローカル容量は実機未測定なので短いpilotとpreflightを優先する。
- 段階は依存しておりA3がFAILならA5-pairはこのlauncherから実行できない。これは現在の受入specに由来し、A5の科学的な独立性とは別の運用制約。

## Findings

| ID | 分類 | 箇所 | 内容 | 影響条件 | 次の検証 |
|---|---|---|---|---|---|
| F-001 | Confirmed by code | `run_moco_gates.py:714-725,739-741` | `--summarize`はGPUを起動せずraw run artifactは書換えないが、`stage_summary.{json,md}`と`combined_summary.{json,md}`を**生成/上書きする**。厳密にはread-onlyコマンドではない | 既存campaign rootがある場合 | raw artifact hashが不変でsummaryのみ再生成されることをテスト/実機で区別 |
| F-002 | Confirmed by code | `run_moco_gates.py:216-232,294-300` | `--execute`はcleanな研究repoと固定Meta clone、前段PASSを必須とし、dirtyではGPUを起動しない | 研究サーバ上に新規コードが未追跡のまま残る場合 | HEAD、`git status --short`、前段Gateのstatusをplanで確認 |
| F-003 | Confirmed by code | `run_moco_gates.py:54-55,225-232` | A5-pairはA3 PASSを前提とする。A3 FAILの場合、A5独立測定が別途必要であっても本launcherではBLOCKED | A3 numerical mismatchなど | 仕様どおり止まることを新規テストで検査済みとの記録あり。新しい迂回条件は承認なしに加えない |
| F-004 | Unknown | 実装記録の`REGRESSION_RESULT`とcommit記述 | 実装記録は非回帰をpendingとし、`implementation_committed=false`を保持。一方GitHubの最新devは既に2新規ファイルを含む`914e736c...`である。SSHのHEAD/テスト結果は未確認 | 作業開始後にcommit/pushされた場合 | 実際の`git rev-parse HEAD`、`git status --short`、既存CPU testの完全な結果/exit codeを確認して記録へ反映 |

## Edge Cases and Failure Modes

- stage directory既存、参考Meta clone dirty、GPU compute jobあり、Config/InfoNCE selector不一致、A3/A4 threshold不一致、A4 checkpoint/sidecar欠損、A3 Queue pointer失敗はBLOCKED/INVALID/FAILとして保持する設計。GPUエラーやA3計算時間増加に対してbatch/K/precisionを自動変更しない。
- 4つのstageのplanを確認する際、後続3stageに`previous stage ... NOT_RUN`が表示されても**予定どおりの事前拒否**であり、学習コードのテストFAILとは区別。
- `git diff`が空でもuntrackedファイルは検出されないため`git status --short`が必須。

## Testability and Verification Gaps

1. **Git基準**：研究サーバで`git branch --show-current`、`git rev-parse HEAD`、`git status --short`、該当2ファイルの存在を確認。GitHubの現dev commitは`914e736c...`。ローカルが別commit/dirtyならその差を保存し、安易に強制同期しない。
2. **CPU engineering**：ユーザー実装記録は新規`test/model/test_moco_gate_launcher.py`が`23 passed`、mutation 5/5検出と主張。既存`test/model/test_moco_parity_harness.py`は`REGRESSION_RESULT`で未確認。実行中の既存pytestを**重複起動せず**完了を待ち、ログの最終行・終了コードでPASSを確認。その後必要なら同一commitで以下を再演：
   ```bash
   CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -m pytest -q test/model/test_moco_gate_launcher.py
   CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -m pytest -q test/model/test_moco_parity_harness.py
   ```
3. **Plan side-effectの検査**：`python -m scripts.parity.run_moco_gates A5-pilot --root "$PWD/log/moco_imagenet_parity/gates/<fresh_name>"`。4stageそれぞれ予定CLIと条件を確認し、`--execute`を付けず、rootが生成されないことを確認。
4. **Stage判定**：実行済みartifactが存在する場合に限り`--summarize`で再判定。summaryは再生成するがraw artifactを更新しない。A4ならuninterrupted/resumedのStep3 formal PASS、A3はQueue pointer1024=0/1025=64、両run1025step formal PASS、A5は20stepと100step pairのresources/windows/overheadを確認。
5. **Real GPU Gate**：未実施。実際のrank worker、NCCL、NAS ImageNet、VRAM、長時間A3 trace保存はCPU mock testsでは証明できない。明示許可、clean HEAD、空きGPU/ディスク、前段PASS、stage別実行が必要。

## Research Reproducibility Considerations

- 比較前にA3/A4の`rtol=1e-5, atol=1e-6, status=fixed`、Meta固定clone`a419ef282fc9885015674e77f58c59447206a27f`、同一fixture SHA、ResNet50・K65536・localB32・FP32などをrunとcompareの双方で固定する。A2の3926件完全一致はA3/A4/A5実機PASSに自動伝搬しない。
- 後日GPUを実施するとき、実験出力はSSHの`log/`以下に保存する。GitHubにGPU fixture、tensor、checkpointをpushしない。

## Unverified Assumptions

- 現地での既存parity harness CPU回帰テストの成否（実装記録は空欄）。
- 現地の新launcherとGitHub最新`914e736c...`が同一か、dirtyの有無。
- 新実装のCPUテスト23件の独立した再実行（本監査では未実施）。
- 実GPUでの4stage正常終了とA3長時間の必要ディスク量。

## Suggested Handoff

最初は既存バックグラウンドテストを完了させ、`REGRESSION_RESULT`と使用したlocal HEAD/dirtyを確定する。**CPUテスト23件+既存parity回帰PASS+plan副作用なし**ならlauncherのEngineering GateをPASSと認めてよい。一方、Gate A3/A4/A5の**科学的/実機PASS**はGPU stageを個別実行したEvidenceが出るまでNOT_RUN。コード変更が必要なbugが見つかった場合は`engineering-task`へ別依頼する。
