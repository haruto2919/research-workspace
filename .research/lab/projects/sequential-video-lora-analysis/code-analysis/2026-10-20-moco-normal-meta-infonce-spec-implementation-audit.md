---
project: sequential-video-lora-analysis
date: 2026-10-20
analysis_mode: audit
detail: full
level: mid
source_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_branch: dev
source_commit: dc502e416aa26ed4e0791767fa5a72f7843e399d
source_dirty: false (user-shared git status at CPU test execution; GitHub-only state cannot verify SSH now)
targets:
  - conf/moco_imagenet_parity.yaml
  - scripts/parity/moco_imagenet_parity.py
  - scripts/parity/compare_moco_parity.py
  - self_supervised/moco/ddp_moco.py
  - test/model/test_moco_ddp.py
  - test/model/test_moco_parity_harness.py
related_spec: ../specs/2026-10-19-meta-infonce-normal-moco-opt-in-spec.md
source_of_test_results: user-shared-terminal-output
---

# Code Audit：通常ResNet50 DDP MoCoへのMeta公式InfoNCE opt-in実装確認

## 対象・結果

対象のResearch Workspace mainを再取得し、[2026-10-19採用・実装spec](../specs/2026-10-19-meta-infonce-normal-moco-opt-in-spec.md)（frontmatter `status: implemented`、`engineering_cpu_gate: PASS`、`gpu_gates: NOT_RUN`）と[実装検証記録](../experiments/2026-10-19-meta-infonce-normal-moco-opt-in-implementation-verification.md)、実研究コードのGitHub devコミット、ユーザーが示した**59/59**および**279/279**のCPUテスト結果を照合した。

**判定：実装scopeに関するEngineering/CPU GateはPASSと支持できる。GPU A2〜A5は未検証。** ChatGPTはこの監査でGPU・SSH実行もテスト再実行も行っていない。201旧testから279 testへの増加は既存assertion 1件の期待値更新を含み、厳密に旧テストコードのバイナリを無変更で全部通したとの主張にはならない。

- Research Workspace：`haruto2919/research-workspace@main`、監査時main HEAD `ba00c62e248cd5ca19a07cc030ca70601d77037a`。
- 研究コード：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、`dc502e416aa26ed4e0791767fa5a72f7843e399d`。
- GitHub compare `1ce9badcbae3e2df08fa400944e4c385785063f9...dc502e416aa26ed4e0791767fa5a72f7843e399d`は1commit aheadで、spec §10.1が指定する**ちょうど6ファイル**のみ（+1323/-114行）。`ddp_moco.py`の差分はdocstringのみ。spec §10.2の不変更対象に変更なし。
- ユーザーが提示したSSHログでは研究コードは同一commitかつ`git status --short`空。Meta公式clone HEAD `a419ef282fc9885015674e77f58c59447206a27f`、今回の出力にはclone `git status`なし。
- 設定select直接チェックは`CONFIG CHECK PASS`、限定pytest `59 passed,178 deselected / 256.79s`、MoCo 4 test files全体 `279 passed / 1188.37s`、いずれもCPU、後者でskip/failなし。

## Architecture Summary・B1〜B12対応

| 論点 | 仕様受入契約と見た証拠 | 判定 |
|---|---|---|
| B1/B2：scope/既定値 | 変更6fileに限定。Standalone/CIFARの実装変更なし。mode default `self`、通常ResNet50 DDPで`meta_einsum` opt-in。 | **CPU/静的 PASS** |
| B3/B4：selector/Gate | `resolve_info_nce_selection()`がA1診断profileとA2〜A5通常selectorを分離、異常設定をspawn前に拒否。test・configチェックPASS。 | **CPU PASS** |
| B5：Checkpoint/Resume | config identity v2、Self native metadata、全rank sidecar、mode違いをstate load前にcollective-safe拒否、legacy assertion明示がtest済み。 | **CPU PASS** |
| B6'：負荷分離 | `parity_loop`のtraceと`bench_loop`分離。monitor OFFでは全Query clone/gradient norm/scalar loggingのcall count 0、window timingは境界同期。 | **CPU機能PASS**、GPU overhead実測NOT_RUN |
| B7/B8：正式比較・実画像preflight | 比較器`formal_implementation`／`formal_resume`はmode不一致・provisional拒否等をtest。A2実画像固定fixture・GPU状態は未確認。 | **比較器CPU PASS**、A2 NOT_RUN |
| B9/B10：互換/自動test | State_dict keys維持、Standalone/CIFAR/A1を含む279/279テストPASS。新schema v3対応。 | **CPU PASS（既知のv2再判定制約あり）** |
| B11/B12：順序/権限 | Spec §13でCPU→A2→A5→A3/A4の独立Gate、GPU無許可実行なし。 | **遵守** |

## Key decisions・Trade-offs

- 算術自体を変更せず、公式と一致した既存`meta_einsum`を選択できるルートを拡張する。既定値は従来の`self`。
- 演算modeはstate_dictには含めず、config/checkpoint/report/provenance側へ正本を記録する。
- `bench_step`はupdate normがdueのrank0だけで`query_snapshot`を実行。grad normも指定周期で計算する。loss有限性検査は監視OFFでも残る。
- 正式Metaとの一致比較は独立GPU runの後、保存したtraceからCPU comparatorで判定する。CPUテスト中のMeta referenceはSelfの出力の属性を付け替えた**模擬run**であり、新条件のGPU数値一致ではない。

## Findings

| ID | 分類 | 箇所 | 根拠・影響条件 | 必要な対応 |
|---|---|---|---|---|
| F-001 | **Confirmed by code and spec（既知の互換制約）** | spec §9.2, §14, §18(2)、`judge_moco_a1_info_nce_16step.py`とschema定数 | 新judge16は新runner/comparatorのschema v3を参照するため、以前保存したA1の**v2 run/comparison artifactを新judge16へ入力するとINVALID**。元のA1 16stepのPASS記録自体が無効化されたわけではない。旧artifactの**新コードからの再判定**には後方互換性のギャップがある。 | 既存v2結果は旧commitとjudgeで再判定するか、正式な新judgeでv2対応が必要かを別Gateで決める。勝手にschemaを書き換えない。 |
| F-002 | **Confirmed（文書の時点差）** | spec frontmatter `implementation_committed: false`、実装検証記録の`未コミット`記述 | 当該文書は2026-10-19のローカル実装当時の状態を記述。現在GitHub devは`dc502e416aa26ed4e0791767fa5a72f7843e399d`でコミット済み。 | 履歴を消さず、後続の変更履歴または確認記録でcommit済みと追記することを推奨。監査ではspec自体を変更しない。 |
| F-003 | **Unverified external Gate** | spec §13, §15 | A2実画像224px、A5 monitor OFF/light GPU overhead、A3 K65536、A4 GPU Resumeは今回のCPUテストでは検証できない。 | 各Gateの正式fixture/閾値/比較対象/VRAM/停止条件を別specで事前固定し、GPU run許可を取る。 |

## Edge Cases and Failure Modes

- Checkpoint/Resume: mode未記録の旧Selfは無条件に`self`と推測せず、旧configとhashの証拠がある場合にのみ明示移行。ユーザー共有testで不一致rankの拒否はpass。
- 設定競合: normal `meta_einsum`＋diagnostic非null、A1 normal meta、Meta公式runへの自作mode指定はspawn前に拒否。
- Monitor disable: 高コストなcopy/norm/logを省略してもloss finite、最終Query/Queue/momentum finite検査は残る。CPUでは例外経路をtest済み。GPU/NCCLへの一般化は未検証。
- 入力実画像・長期Queue: A1 small syntheticやCPU/glooの結果から推定不可。

## Testability / Verification Gap

- 確認済み：設定 `CONFIG CHECK PASS`、選択59テストPASS、全MoCo 279テストPASS（`CUDA_VISIBLE_DEVICES=""`）。201件旧テストのうち1件は新config仕様に合わせ期待assertionを変更した、と実装記録に明記。
- CPU `formal_implementation`のPASSは比較器ロジックの擬似参照テスト。本物のMeta vs Self A2数値一致には、二つの独立GPU runとオフライン比較が必要。
- `positive_cosine_mean > 1`を出すFakeBenchAdapterは正規化済みMoCo特徴を出力しないため、研究モデルの特徴品質を示すmetricではない。
- DDP異常系でのSIGTERM、COMET deprecated warningは最終`279 passed`を崩す失敗ではない。

## Research Reproducibility Considerations

- 旧A1 GPU本番Evidence：`1ce9bad...`の公式 vs Self 16step・2GPU・K64条件に限定。今回の`dc502e4...`でも同じGPU/新通常modeで公式一致した証拠ではない。
- 研究コードのcommit状態と、2026-10-19当時のdiff-only verification記録は区別する。ユーザーが示したGitとGithubの現在commitは整合する。
- 親Spec Aはdraft。本opt-in specはimplemented。GPU gateは別Authority。

## Suggested Handoff

- Engineering/CPU Gateは**PASSとして記録可能**、コード変更不要。
- A2の実画像1step parity用の小規模spec/実験計画、A5のmonitor overheadベンチ計画を次の検討候補とする。
- F-001の旧A1 v2 artifactを新judgeで再判定する必要があるかと、F-002の文書来歴追記は別件として検討。現在のspec・research codeの変更はこの監査では行っていない。

## References

- [Normal Meta InfoNCE opt-in spec](../specs/2026-10-19-meta-infonce-normal-moco-opt-in-spec.md)
- [Implementation verification record](../experiments/2026-10-19-meta-infonce-normal-moco-opt-in-implementation-verification.md)
- [User retest CPU 279 pass](../experiments/2026-10-20-moco-normal-infonce-opt-in-cpu-verification.md)
- [A1 16step GPU parity evidence](../experiments/2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md)
