---
project: sequential-video-lora-analysis
spec_type: engineering-and-experiment
status: implemented
created: 2026-10-20
last_updated: 2026-10-20
title: "Meta MoCo A3/A4/A5 integrated execution orchestration on existing parity and bench runner"
authority: "2026-10-20 user expressly adopted proposed A3/A4/A5 integrated design but requested spec-only work: no research code edits, commit/push, or GPU runs"
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit_at_read: dc502e416aa26ed4e0791767fa5a72f7843e399d
meta_reference_commit: a419ef282fc9885015674e77f58c59447206a27f
a2_evidence: ../experiments/2026-10-20-meta-moco-a2-imagenet-real-one-step-formal-gpu-parity-evidence.md
parent_spec: 2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md
related_a5_draft: 2026-10-20-meta-moco-a5-imagenet-gpu-bench-monitor-overhead-experiment-spec.md
implementation: IMPLEMENTED_COMMITTED
implemented_on: 2026-10-20
implementation_committed: true
implementation_commit: 914e736c2699e2e129211a888d1f19013dd1dd1a
engineering_cpu_gate: PASS
engineering_cpu_fast_regression: "29 passed (new launcher 23 + parity selected 6); existing full suite not verified"
implementation_record: ../experiments/2026-10-20-meta-moco-a3-a4-a5-gate-launcher-implementation-verification.md
gpu_experiments: "A5-pilot PASS; A4 PASS; A3 NOT_RUN; A5-pair NOT_RUN"
a4_evidence: ../experiments/2026-10-20-meta-moco-a4-synthetic-two-gpu-step2-resume-formal-parity-evidence.md
a5_pilot_evidence: ../experiments/2026-10-20-meta-moco-a5-pilot-real-imagenet-two-gpu-20step-evidence.md
approved_on: 2026-10-20
research_code_edits_authorized_now: true  # 2026-10-20 user implementation request (launcher + CPU tests only)
research_code_commit_push_authorized_now: false
gpu_run_authorized_now: false
approved_a3_a4_rtol: 1.0e-5
approved_a3_a4_atol: 1.0e-6
---

# A3・A4・A5統合実行設計（approved／実装・実験未着手）

**2026-10-20更新：ランチャー実装済み（未commit）・CPU検証PASS、GPU Gateは全てNOT_RUN。** 実装の詳細と解釈は§9と[実装・CPU検証記録](../experiments/2026-10-20-meta-moco-a3-a4-a5-gate-launcher-implementation-verification.md)。以下の本文は承認時点の記述のまま残す。

> **現行状態：approved（設計のみ承認）、研究コード変更なし、GPU Gateは全てNOT_RUN。** ユーザーは2026-10-20に、前回の推奨条件（共通ランチャー＋dry-run、A3 synthetic 1025step、A4 step2→3 resume、A5 20step pilot＋OFF/Light各100step、A3/A4固定許容差）を明示採用し、**今回は研究コードに直接追加せずspecを保存すること**を指定した。既存MoCo本体・parity/bench runner・CPU comparatorを再利用し、3つのGateは**独立したrun**としてオーケストレーションする。後日の実装許可、研究コードGitHub commit/push、GPU実験起動は今回の依頼には含まれない。

## 1. 根拠、現状とscope

- [A2正式PASS](../experiments/2026-10-20-meta-moco-a2-imagenet-real-one-step-formal-gpu-parity-evidence.md)：ImageNet64枚・224px・ResNet50＋MLP・2GPU・local32/global64・K65536・FP32・独立fresh Meta M1 vs 自作S1 meta_einsum・1step。formal comparator judged3,926/bitwise3,926、差分0、事前fixed rtol1e-5/atol1e-6。**1stepまでのEvidence**。
- [親Spec A（draft）](2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md) はA3=Queue一巡と複数step数値比較、A4=checkpoint/resume、A5=ImageNet実画像100〜300step GPU速度/VRAMと規定。A3/A4/A5全体の正式実行許可と判定閾値は自動付与されない。
- [通常InfoNCE opt-in spec（implemented）](2026-10-19-meta-infonce-normal-moco-opt-in-spec.md) はself既定を維持しつつ`moco.info_nce_impl=meta_einsum`を明示選択する仕組み、run report/Checkpoint identity、formal pairs、bench軽量monitorを実装。CPU MoCo testsは279/279 PASS（当該時点の記録）。
- [A5単独draft](2026-10-20-meta-moco-a5-imagenet-gpu-bench-monitor-overhead-experiment-spec.md) で提案された20step pilot→100step監視OFF/ON方針は**本統合specで採用済み**。統合specを実行条件の優先SSOTとし、A5単独draftのstatusは変更せず、実装時に両者の重複を生じさせない。
- GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、commit `dc502e416aa26ed4e0791767fa5a72f7843e399d` の `conf/moco_imagenet_parity.yaml`、`scripts/parity/moco_imagenet_parity.py`、`scripts/parity/build_moco_parity_fixture.py`、`scripts/parity/compare_moco_parity.py`、`test/model/test_moco_parity_harness.py` をread-only確認した。これは**GitHub上のcommit**であり研究サーバの現dirtyやGPU実行状態を示すものではない。repoルート/当該parity配下のAGENTS.md/CLAUDE.mdはGitHub上で見当たらない。

### GitHubコード上で実装済み

| Gate | 存在する機能 | 未検証／残る統合作業 |
|---|---|---|
| A3 | canonical synthetic fixtureをstepごとに決定論的生成できる。2rank DDP Shuffle-BN、K65536、1025step、trace.full_steps、Meta vs Self `formal_implementation`。現runnerのdocstringもA3例を持つ | **実Kの1025step対照GPU実行は未確認**。run条件・判定閾値・時間/ログ容量・first mismatch対応の統合 |
| A4 | `checkpoint_steps` native＋rank別sidecar、mode/RNG/fixture/config SHA照合を伴うresume、`formal_resume`、CPUでの拒否/成功テスト | **正式2GPUのcheckpoint復帰・数値一致は未実施**。独立uninterruptedとresumedの手順とtrace比較 |
| A5 | `run_kind=bench`、ImageNet train DistributedSampler/two-crop/DataLoader、監視OFF/Light、warmup10/window10、GPU VRAM/throughput/data-wait json | **実機100〜300step OFF/Light paired Evidenceは未実施**。独立runと集計・順序/測定限界の記録 |

**後日の実装対象として採用**：`scripts/parity/`の既存CLIに渡すコマンドとfresh出力を段階管理する**最小限の共通ランチャー**およびCPU dry-run／validation tests。MoCo本体・parity loop・bench loop・comparatorの数式/判定やHydra既定値は変更しない。複雑なフレームワーク・新dependency・公式ソースのコピーは不要。ランチャーは外部挙動を既存CLI起動・出力契約に限定し、後日の実装時もより小さい構成で同じ契約を満たせるなら差分を最小化する。**今回はコードを一切編集しない。**

## 2. 分離すべきGateと採用済みの段階順序

1. **事前確認**：研究コード・Meta参照cloneのcommit/clean、ImageNet NAS read-only、2GPU/NCCL可用性、GPU競合、出力容量、seed/precisionを確認。既存A2は容量目安の1stepにすぎない。
2. **A5小pilot（先）**：`run_kind=bench gate=A5 implementation=self moco.info_nce_impl=meta_einsum diagnostic.info_nce_impl=null`、ImageNet/224/local32/K65536/FP32/2GPU、20step、monitor OFF。GPU/NAS/DataLoaderとVRAMのfeasibilityを先に確認。
3. **A4短時間GPU resume**：synthetic canonical fixture、原則224px/local32/K65536、uninterrupted self `meta_einsum`3step（checkpoint after step2、full trace relevant step）、同checkpointから別fresh resume runを3stepまで実行。CPU `--pair-kind formal_resume --gate A4`でstep3を比較、native/sidecar mode/SHA/RNG/optimizer/Queue/BN/DDPの復元が一致するか判定。**Benchではresume不可（現実装が拒否）**。
4. **A3長いGPU Queue一巡**：Meta公式referenceと自作self `meta_einsum`を同一**synthetic** canonical fixture（224px/local32/K65536/seed0）から独立freshで`run_kind=parity gate=A3 training.max_steps=1025 'trace.full_steps=[1,1024,1025]'`。同batch global64でstep1024に初期random Queueが全置換され`queue_ptr=0`、step1025に`queue_ptr=64`となる。毎step軽いtraceとfull step traceを区別し、`--pair-kind formal_implementation`で比較。**既存runnerはparityで毎step traceを保存し、CPU↔GPU同期やI/Oコストがある**ため、正式長時間job前に容量/速度見積りが必要。A1の小K wrap検証と実K長期を混同しない。
5. **A5本計測**：A5-0がPASSした場合、必要な時点で独立fresh100step×2本（monitor OFF → Light）、warmup10、window10、Comet OFF。ImageNet DataLoaderのサンプル順・seed・構成を一致させ、VRAM・throughput・DataLoader wait・measurement windowを比較。**A3/A4のheavy trace時間をA5性能計測へ使わない**。逆順再演はノイズが大きい場合のオプションであり初回必須ではない。

**A5 pilot→A4→A3→A5 paired**は、今回採用した段階順序であり、いずれかの工程で止まっても後続を無許可に流さず、失敗理由と保存済みartifactを追跡できる。単一コマンドの一括起動でも、**各Gateの実行許可は別個に表現**する。

## 3. 採用済み条件（設計承認。GPU実機結果ではない）

| Axis | Adopted contract |
|---|---|
| Model | Random-init ResNet50＋MLP dim128。Meta固定参照clone無改変、自作S1の通常InfoNCEは明示`meta_einsum` |
| Global parameters | FP32 AMP OFF TF32 OFF、seed0、2GPU NCCL、global64（local32×2）、K65536、m0.999、T0.2、SGD lr0.0075/momentum0.9/wd1e-4、epoch0 Cos horizon10 |
| A3 dataset | synthetic、224px、1,025stepを決定論的に再生成。**ImageNet 1,025step real fixtureは容量が大きいので黙って採用しない** |
| A3 full trace | [1,1024,1025]。light traceは毎step、compareには両run共通のStep/rank/phaseが必要 |
| A4 fixture | synthetic、224px、3step。checkpoint step2、uninterrupted 3stepとresume後step3の対照 |
| A5 dataset | real ImageNet train、data.num_workers=8/rank、image224、sampler epoch0、20step OFF pilot、OFF/Light各100step、Comet OFF |
| Threshold | **今回A3とA4にも `rtol=1e-5`・`atol=1e-6`・`verification.threshold_status=fixed`・離散exactをユーザーが事前採用**。A2の過去runを使い回さず、A3/A4の新規runのconfigとoffline comparatorに明示する。長い累積差でFAILしても事後緩和しない。 |
| Log/trace | 全run新規ディレクトリ。既存実験artifact上書きなし。大きなfixture.pt/trace/GPU logをResearch Workspace GitHubへpushしない |
| A5 overhead | feasibility PASSと定量差の実測を分離し、根拠なく%許容上限を設けない。必要時に上限を別途事前固定 |

## 4. 承認済み実装契約：最小オーケストレーション（コード変更は今回禁止）

### Required behavior

1. 既存の3本（fixture生成／parity-or-bench runner／offline comparator）だけを呼び出し、**同一設定、分散rank、参照clone provenance、mode identityを維持**。
2. `A5-pilot`、`A4`、`A3`、`A5-pair`を**明示ステージ**として分け、ステージ単位で停止・確認できる。自動でA5-pairや長いA3へ進めない。
3. 事前に既存path衝突、config/commit/dirty、GPU可用、fixture metadata/hash、threshold fixed、既存checkpoint identityを検査。**事前条件に反する場合fail-closed**。OOMやNCCL failureでbatch、K、image、precision、workers、max_stepsを縮小しない。
4. `run_kind=parity`のA3/A4をfull trace含めてCPU offline判定に渡す。A5 `bench`は`completed_steps`とper-rank resources/window metricsを検証し、monitor OFFのmetric historyが空でも失敗とはしない。
5. 既存run dirから結果だけ読んでGate別summaryを生成できる（実行時の失敗や停止後もraw artifactを保全）。GPU実行許可を示す明示モードなしに新規GPUジョブを起動しない。
6. 単一入口は**複雑なlauncherの新規開発を避ける最小構成**とし、既存Hydra CLIを変更しない。Shell wrapper＋小さな結果集計で満たせるなら大きな新frameworkを作らない。

### 最小の検証

- GitHub上の固定基準commit、diff、Python/YAML/Hydra設定の有効性（ローカル実行環境に置かれている場合のみ）。
- GPUを使わないdry-run/モックで、各stageが想定のCLI、指定mode、fixture、独立出力、thresholdを使うこと、出力先の衝突と前段failが後段起動を止めること。
- CPUの既存MoCo回帰テストを必要な範囲で実行（GPU/multi-hour jobを実装確認の名目で勝手に起動しない）。
- 全GPU runはこの工学的実装Gateとは分けて`NOT_RUN`と記録する。

## 5. Gate成功条件とEvidence

| Stage | PASSと言える条件 | 言えないこと |
|---|---|---|
| A3 | 同一fixtureを使ったM/S1の2runが1025stepまで完了、初期Queue一巡とptr境界を確認、`formal_implementation`固定閾値内PASS | 長時間の良い表現・実ImageNet長時間parity |
| A4 | 2GPUのcheckpoint native＋両rank sidecarが揃い、fresh resumedがstep3へ到達し、`formal_resume`がfixed閾値でPASS | bench/DataLoaderを含むresume、任意checkpoint互換 |
| A5-pilot | 20step、2rankで正常終了、mode一致、finite、VRAMと1window記録 | 100step feasibility、監視overhead |
| A5-pair | fresh OFF100 + Light100 が完走し、VRAM/step-time/window・throughput/data-waitを記録、OFF/Lightの変更軸が監視だけ | 速度の一般的優位、表現品質、10epoch |
| Combined | 必須3 Gateで各々上記条件を満たし、結果にmanifest/根拠path/commit/設定/障害を記録 | ViT/LoRA/オンライン時間表現の獲得 |

FAIL／INVALID／BLOCKED／NOT_RUNを分離する。監視の百分率上限を未固定のまま「実測した」以上のoverhead PASSとしない。ユーザーの保存したraw GPU traceやcheckpointをGitHub ChatGPTが独立検査済みとは表現しない。

## 6. 採用済みDecision Gate／実行権限の保留

### Resolved（2026-10-20ユーザー採用）

1. **統合方式**：既存のfixture generator、2GPU parity/bench runner、offline comparatorを呼び出す共通ランチャー＋Gate別summaryを後日最小実装する。既存CLIのロジックは再実装しない。CPU dry-runと軽量回帰テストまでは将来のengineering-taskの予定scope。**今回はspecのみ作成し、実装やcommit/pushはしない**。
2. **A3**：synthetic fixture、224px、2GPU/global64、K65536、Meta公式M vs自作S1 meta_einsum、独立fresh1025step×2run。full trace stepは`[1,1024,1025]`、Queue初期stateの1024step完全置換と1025stepのwrapを別途判定する。ImageNet実画像1025stepは採用しない。
3. **A4**：synthetic 3step、uninterrupted Self S1のstep2でnative checkpoint＋rank0/1 sidecar、同一checkpointを使うfresh resumed runのstep3。正式比較は`formal_resume`で**step3**（rank>0の再開前BN bufferは既存comparatorのinformation-only規則）。Benchmarkでのresumeを追加しない。
4. **A3/A4判定基準**：事前固定 `rtol=1e-5, atol=1e-6, threshold_status=fixed`、離散状態exact。A3は`formal_implementation`、A4は`formal_resume`。bitwise率は別記録し、100%を無条件の必須成功条件にしない。
5. **A5**：2GPU FP32、ImageNet train、worker8/rank、S1のnormal`meta_einsum`、20step monitor OFF pilot → 独立fresh100step OFF/100step Light（warmup10/window10）。逆順100+100stepは初回必須でない。負荷%に恣意的な上限を設定せず、実行可否とoverhead実測を分離する。
6. **実行権限**：現ユーザー指示は**研究コードに一切追加せずspecを作る**もの。ランチャー/testの実装、GitHub研究コードcommit/push、GPU実験の起動、A3長時間の予約、A4 checkpoint生成は今回**行わない**。後日、engineering-taskへの別実装依頼とGateごとのGPU実行指示を受けるまで`NOT_STARTED/NOT_RUN`を保持する。

### Implementation-readiness check

- **blocking ambiguity（今回のspec内容）**：なし。以前の1〜5は上記の推奨条件をユーザーが採用し解消済み。
- **blocking approval（実行行為）**：研究コード編集とremote push、GPU実験の実行は**現在不許可**。specの`approved`はそれらの操作許可ではない。
- **non-blocking**：run directory suffix・ファイル名・dry-run commandの表示形式等は既存repoのstyleで決める。ただし科学的条件・default・artifact schemaは変更しない。

### Non-blocking

- 既存repoと同じ命名規則でrun-idをつける／summaryの軽微な見出し、shellのログfile名。
- 他の実験に影響しない範囲のコード配列・formatの細部。既存artifactやdefaultの意味は変えない。

**Readiness：approved（設計条件と合否契約は解決済み）**。現ユーザー指示によりspec保存だけを行う。engineering-taskの実装、commit/push、GPU runには後日の独立した明示依頼が必要。

## 7. Implementation Handoff / Current status

- **Approved spec**：本ファイル（`approved`、2026-10-20ユーザーによる推奨条件採用）。
- **既存実装の正本**：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、commit `dc502e416aa26ed4e0791767fa5a72f7843e399d`。GitHubと研究サーバdirtyは別扱い。
- **後日の変更scope**：最小共通ランチャー（A5-pilot/A4/A3/A5-pair明示ステージ、fresh output、fail-closed、報告）とCPU dry-run/非回帰テストのみ。MoCo/InfoNCE/trace/optimizer/Queueの本体変更なし。
- **実装成功条件**：ステージ境界の独立性、出力衝突防止、停止伝播、設定/閾値/検証契約と非回帰。GPUによるA3/A4/A5実験結果は別Evidence。
- **今回実施しない**：研究コード直接追加、CPU/GPU test実行、公式Meta clone修正、研究コードcommit/push、GPU job起動、既存baseline変更、10epoch/ViT/LoRA統合。
- **長時間runの許可状態**：未許可。A3（公式/自作各1025step）は特にresource/GPU時間/ログ容量の実測見積りと別の実行指示を要する。
- **短時間検証**：後日の実装依頼でCPU dry-run/unit/non-regressionのscopeを再確認する。今回実施なし。
- **Evidence**：A1/A2 PASSのみ引き継ぐ。A3/A4/A5 = NOT_RUN。GitHub上のコード基準commitと研究サーバの未committed状態は別。


---

## 8. 2026-10-20 保存・承認記録（最新Authority）

ユーザーの「採用します。ただし、研究コードに直接追加せず、specを作成してください」という明示指示により、従来draftの推奨条件を採用し、既存の統合specを`approved`へ昇格した。**Research WorkspaceのMarkdown更新のみ**を行い、研究コードrepoのファイル/branch、Meta clone、NAS、GPU、checkpoint/run artifactは一切変更・実行していない。

### ランチャーの後日実装・最低限の検証契約

- **入力**：stage名（`A5-pilot`、`A4`、`A3`、`A5-pair`）、出力root。各stageのmodel/batch/K/seed/precision/thresholdは本specの固定値。別Gateへ実行の権限を暗黙に波及させない。MoCo runnerの既存Hydra CLIをそのまま起動する。
- **デフォルト**：GPU未起動のplan/dry-runのみ。将来GPU実行を許可する場合でも、ユーザーがstageを個別指定した明示的モードを必須とする。単一の`all`指定で1025step A3へ無条件実行しない。
- **出力**：個別runディレクトリとfixtureパス・SHA、各CLI command、resolved config、終了コード、run status、実行環境・provenance、比較reportパス、失敗時stageと原因、Gate別status/summary。
- **失敗時**：出力衝突、commit dirty/mismatch、mode/threshold/fixture不一致、GPU不足、OOM/NCCL失敗、必要artifact欠損でfail closed。既存run出力を上書きせず、条件を自動変更しない。既存ランチャーのログとcheckpointを保持する。
- **CPUのdry-run試験**：stageごとの組み立てCLIと固定config、A3/A4の`fixed`閾値・対応pair-kind、A5の`bench`と監視条件、独立fresh dir、出力既存・前段failの停止、`run_report`への適切な判定、GPU未起動保証を検証する。必要最小限の既存MoCoテストも非回帰で確認する。
- **比較ルールの注意**：A3は`formal_implementation`で1〜1025の全stepに共通のsmall traceを持ち、`full_steps=[1,1024,1025]`の大きいtensorも両runで同じstepを収集する。A4は`formal_resume`に従いresume済み候補のstep3だけをformal比較し、rank>0の最初の`before_forward` BN buffer差を既存のinformational規則で扱う。Gate別新数値規則は加えない。
- **A5 overheadの制約**：OFF/Light各100stepは順序効果やNAS cacheを含むため、throughput・メモリ・Data waitの実測値と差分を提示するだけで「絶対性能優位」などは主張しない。
- **最終状態**：`spec=approved`、`implementation=NOT_STARTED`、`GPU A3/A4/A5=NOT_RUN`。後日の実装作業は`engineering-task`へhandoffし、長時間GPU実験とは独立に承認する。

---

## 9. 2026-10-20 実装記録（最新Authority）

ユーザーの「このスペックに従って実装してください」により、§4・§8の実装scope（最小共通ランチャー＋CPU dry-run/validation tests）を`engineering-task`で実装した。詳細は[実装・CPU検証記録](../experiments/2026-10-20-meta-moco-a3-a4-a5-gate-launcher-implementation-verification.md)。

- **追加**：`scripts/parity/run_moco_gates.py`（plan既定／`--execute`は1 stageのみ／`--summarize`は読み取りのみ）、`test/model/test_moco_gate_launcher.py`（23件）。既存ファイル・Hydra既定値・MoCo本体・runner・comparatorは無変更、新dependencyなし。
- **検証**：新規23件PASS、5種のミューテーションをすべて検出、既存parity harnessの非回帰はCPUで実行（結果は実装記録）。GPUは不使用。
- **実装時の解釈**：(1) 前段条件は採用順序の直列で、各stageは直前stageが同じroot・同じ研究commitでPASSしていることを要求する（A3がFAILするとA5-pairもランチャーからは起動できず、進めるかは新たな判断事項）。(2) A4のfull traceは比較stepの`[3]`。(3) Comet OFFは全stage。(4) fixtureはA4・A3でそれぞれ新規生成。(5) GPU競合は`nvidia-smi`だけで判定し、ランチャー自身はCUDA contextを作らない。(6) 研究repoがdirtyだと`--execute`はBLOCKEDになるため、ランチャーのcommit後でないと実行できない。
- **状態**：`spec=implemented`、`implementation=IMPLEMENTED_UNCOMMITTED`、`GPU A5-pilot/A4/A3/A5-pair=NOT_RUN`。研究コードのcommit/push、GPU実行、A3長時間runの予約は未実施で、Gateごとの別指示が必要。

---

## 9. 2026-10-20実装・実機runステータス更新（後続Authority）

以前の本文の「今回はコードを変更しない」「GPU未許可/未実行」は**当初のspec承認時点**での権限境界である。その後、別の明示的なユーザー実装依頼により、統合ランチャーとテスト2ファイルが研究コード`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`に追加され、`914e736c2699e2e129211a888d1f19013dd1dd1a`としてcommitされた（ユーザー端末でlocal HEAD・cleanとGitHub origin/dev一致を確認）。承認済み設計と元の科学条件は変更しない。

軽量CPU確認は**29 passed / 15.05秒 / exit0**（新規ランチャー23テスト＋既存parityの代表6テスト）。**既存parity harness全件**の完了結果は未確認であるため、全件回帰PASSとはしない。

**A5-pilotのみ2026-10-20に実GPUでPASS**：campaign `a345_20261020_114641`、ImageNet実画像、RTX A4500×2、20step、normal`meta_einsum`、monitor OFF。ランチャー標準判定`A5-pilot PASS: every step completed and every check holds`はユーザー提供terminalに基づく。その後ユーザーが`stage_summary.md`/`run_report.json`の表示ログを提出し、GPU peak reserved 3788/3886MiB、step time約0.7905秒、global throughput約80.96 images/sec、DataLoader wait約0.599/0.586秒/stepを確認。[正本GPU Evidence](../experiments/2026-10-20-meta-moco-a5-pilot-real-imagenet-two-gpu-20step-evidence.md)参照。

**A4、A3、A5-pairは引き続きNOT_RUN。** この**同じcampaign root**でA4の`plan`を行い、ユーザー提供terminalで`previous stage A5-pilot = PASS`、`issues=[]`を確認済み。**A4のGPU実行自体はまだNOT_RUN**であり、実行時にはGPU競合や研究repoのcleanを再検査する。A3（各1025step）とA5-pair（OFF/Light各100step）の実行結果を推測で補わない。

---

## 10. 2026-10-20 A4実GPU Formal PASS（後続Evidence）

ユーザー提供の研究サーバterminalに表示された`A4/stage_summary.json`と`A4/compare/parity_report.json`によって、採用済みのA4 fixed-threshold formal checkpoint/resume Gateが**PASS**と確認された。詳細と根拠は[A4正式Evidence](../experiments/2026-10-20-meta-moco-a4-synthetic-two-gpu-step2-resume-formal-parity-evidence.md)。

- campaign rootはA5-pilotと同じ`log/moco_imagenet_parity/gates/a345_20261020_114641`。
- Stage A4: **PASS**、reason `every step completed and every check holds`、`issues=[]`、fixture / uninterrupted / resumed / compare の4工程は全てcompleted・exit0。
- Compare `formal_resume`: **PASS**、`contract_valid=true`、`steps_compared=[3]`、`full_tensor_steps=[3]`、事前固定`rtol=1e-5` /`atol=1e-6`、`threshold_status=fixed`。
- Judged comparison entries **3,608/3,608 bitwise一致**、exact failures0、close failures0、first difference/failureなし。**step2 checkpointからのstep3再開**を、同一自作実装の中断なしstep3と比較したもの。A2/A3のMeta公式対自作比較とは異なる。
- **Evidence boundary**: 指定synthetic fixture・2GPU・3stepの実行についての正式比較。長時間の任意Resume、実ImageNet DataLoader resume、A3の1025stepや特徴表現品質の証明ではない。
- Current GPU Gates: **A5-pilot PASS / A4 PASS / A3 NOT_RUN / A5-pair NOT_RUN**。
- **次のGate**：A3はMeta公式Mと自作S1の**1025step×2run**、synthetic224px/local32/K65536で、各stepのsmall traceとstep1/1024/1025のfull traceを保存する。A5-pilotの`run_kind=bench`で得た0.79秒/stepをparity本番時間として扱わない。実行前にA3 `plan`とA4 trace/artifactのサイズ計測を行い、GPU・disk・所要時間の見積りを確定してから別にrun指示を受ける。

---

## 11. 2026-10-20 A3 preflight・trace容量の実測外挿（NOT_RUN）

ユーザーが研究サーバで実行したA4成果物の`du`およびtrace file size計測と、**A3のplan-only出力**を根拠とする。これはA3のGPU実行結果ではない。

### A4の容量実測とA3概算

- A4 stage保存ディレクトリ全体：`du -sh "$ROOT/A4" = 3.7G`。
- A4 `S1_uninterrupted` のstep1 small trace：rank0＋rank1合計 **0.20 MiB**。
- A4のstep3 full trace：rank0＋rank1合計 **1,556.00 MiB**。
- A3のtrace外挿式（**2 run合計**）：`2 × (1,022 small steps × 0.20 MiB + 3 full steps × 1,556.00 MiB) / 1,024 = 約9.53 GiB`（元の未丸めたファイルbyteからユーザーの計算スクリプトが出力した値）。Meta公式側と自作側のtrace形式・圧縮・metadata差、fixture、log、比較report、cacheなどを含まない。**9.53GiBはtraceの概算でありA3全体の上限ではない**。
- A3のplan時点で`disk_free_gib = 2758.0`。残容量だけを見る限り、trace推定値に対して十分な余裕。
- A5-pilotで測った`0.79秒/step`は`run_kind=bench`かつmonitor OFFのwindowであり、parityのper-step trace/hook/CPU保存やMeta公式runを含まないため、A3のGPU総所要時間へ単純に掛けない。長時間の実行時間は**未実測・未確定**。

### A3 plan（実行なし）

- Root：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/gates/a345_20261020_114641`。Stage予定先`A3/`。
- `previous = A4 PASS`、同じcommit `914e736c2699e2e129211a888d1f19013dd1dd1a`。研究repo dev clean、Meta clone pinned `a419ef282fc9885015674e77f58c59447206a27f` clean、`issues=[]`。
- 予定：synthetic fixture生成 → Meta公式M 1025step → 自作S1 `meta_einsum` 1025step → `formal_implementation`でfixed tolerance `rtol=1e-5,atol=1e-6`比較。full traceは`[1,1024,1025]`、残りstepでもsmall trace収集。
- **A3の状態 = NOT_RUN**。`plan`出力に含まれる`mode = plan only`は学習・ディレクトリ作成なし。A3の実行は長時間になるため、A4ランチャーの`stage_summary.json.steps.*.elapsed_sec`、A4 `run_report.json`の`ranks[*].history[*].metrics.step_time_sec_with_tracing`から短時間の参考負荷をまず抽出するのが次の低リスク作業。その値も数stepの参考値で、A3 2050stepの実行時間を保証しない。
- 2GPU空き、SSHセッション維持、ログ監視、runの中断時のstage dir再実行不可（既存出力を上書きしない）が実行時の運用上の注意点。
