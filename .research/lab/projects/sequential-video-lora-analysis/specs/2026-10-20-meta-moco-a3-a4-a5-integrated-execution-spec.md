---
project: sequential-video-lora-analysis
spec_type: engineering-and-experiment
status: draft
created: 2026-10-20
last_updated: 2026-10-20
title: "Meta MoCo A3/A4/A5 integrated execution orchestration on existing parity and bench runner"
authority: "Current user request: implement A3, A4, A5 together; detailed experiment thresholds, GPU budget and Git push not yet approved"
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit_at_read: dc502e416aa26ed4e0791767fa5a72f7843e399d
meta_reference_commit: a419ef282fc9885015674e77f58c59447206a27f
a2_evidence: ../experiments/2026-10-20-meta-moco-a2-imagenet-real-one-step-formal-gpu-parity-evidence.md
parent_spec: 2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md
related_a5_draft: 2026-10-20-meta-moco-a5-imagenet-gpu-bench-monitor-overhead-experiment-spec.md
implementation: NOT_STARTED
gpu_experiments: NOT_RUN
---

# A3・A4・A5統合実行設計（draft／承認前）

> **現行の研究コードにはA3・A4・A5に必要な主要機能が既に存在する。** ここでいう「まとめて実装」は、既存MoCo本体・parity runner・bench runner・CPU comparatorを利用し、各Gateの条件固定、実行順、artifact検証と停止条件を一つの*オーケストレーション*として扱うことを第一候補とする。**3つを同じ学習runへ合成しない。** この案を採用する場合も各GPU実験、Git commit/pushには別の実行許可が必要。

## 1. 根拠、現状とscope

- [A2正式PASS](../experiments/2026-10-20-meta-moco-a2-imagenet-real-one-step-formal-gpu-parity-evidence.md)：ImageNet64枚・224px・ResNet50＋MLP・2GPU・local32/global64・K65536・FP32・独立fresh Meta M1 vs 自作S1 meta_einsum・1step。formal comparator judged3,926/bitwise3,926、差分0、事前fixed rtol1e-5/atol1e-6。**1stepまでのEvidence**。
- [親Spec A（draft）](2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md) はA3=Queue一巡と複数step数値比較、A4=checkpoint/resume、A5=ImageNet実画像100〜300step GPU速度/VRAMと規定。A3/A4/A5全体の正式実行許可と判定閾値は自動付与されない。
- [通常InfoNCE opt-in spec（implemented）](2026-10-19-meta-infonce-normal-moco-opt-in-spec.md) はself既定を維持しつつ`moco.info_nce_impl=meta_einsum`を明示選択する仕組み、run report/Checkpoint identity、formal pairs、bench軽量monitorを実装。CPU MoCo testsは279/279 PASS（当該時点の記録）。
- [A5単独draft](2026-10-20-meta-moco-a5-imagenet-gpu-bench-monitor-overhead-experiment-spec.md) の20step pilot→100step監視OFF/ON方針を**候補として再利用**。本統合specがapprovedになった時に重複を整理し、A5単独draftを黙ってimplemented/approvedへ変更しない。
- GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、commit `dc502e416aa26ed4e0791767fa5a72f7843e399d` の `conf/moco_imagenet_parity.yaml`、`scripts/parity/moco_imagenet_parity.py`、`scripts/parity/build_moco_parity_fixture.py`、`scripts/parity/compare_moco_parity.py`、`test/model/test_moco_parity_harness.py` をread-only確認した。これは**GitHub上のcommit**であり研究サーバの現dirtyやGPU実行状態を示すものではない。repoルート/当該parity配下のAGENTS.md/CLAUDE.mdはGitHub上で見当たらない。

### GitHubコード上で実装済み

| Gate | 存在する機能 | 未検証／残る統合作業 |
|---|---|---|
| A3 | canonical synthetic fixtureをstepごとに決定論的生成できる。2rank DDP Shuffle-BN、K65536、1025step、trace.full_steps、Meta vs Self `formal_implementation`。現runnerのdocstringもA3例を持つ | **実Kの1025step対照GPU実行は未確認**。run条件・判定閾値・時間/ログ容量・first mismatch対応の統合 |
| A4 | `checkpoint_steps` native＋rank別sidecar、mode/RNG/fixture/config SHA照合を伴うresume、`formal_resume`、CPUでの拒否/成功テスト | **正式2GPUのcheckpoint復帰・数値一致は未実施**。独立uninterruptedとresumedの手順とtrace比較 |
| A5 | `run_kind=bench`、ImageNet train DistributedSampler/two-crop/DataLoader、監視OFF/Light、warmup10/window10、GPU VRAM/throughput/data-wait json | **実機100〜300step OFF/Light paired Evidenceは未実施**。独立runと集計・順序/測定限界の記録 |

**変更対象候補**：`scripts/parity/`に最小の統合実行ランチャー（既存3 CLIを明示的に起動し、stage毎fresh dir・command/config identity・出力path・合否を整理）、必要な短いunit/dry-run tests。必要性がないなら既存CLI＋共通runbookの方が最小なので、承認時にコード新設とrunbookのみのどちらかを選ぶ。新たな学習loop/Meta実装コピー/InfoNCE演算変更、デフォルト変更、旧artifact改変、外部dependency追加は行わない。

## 2. 分離すべきGateと提案順序

1. **事前確認**：研究コード・Meta参照cloneのcommit/clean、ImageNet NAS read-only、2GPU/NCCL可用性、GPU競合、出力容量、seed/precisionを確認。既存A2は容量目安の1stepにすぎない。
2. **A5小pilot（先）**：`run_kind=bench gate=A5 implementation=self moco.info_nce_impl=meta_einsum diagnostic.info_nce_impl=null`、ImageNet/224/local32/K65536/FP32/2GPU、20step、monitor OFF。GPU/NAS/DataLoaderとVRAMのfeasibilityを先に確認。
3. **A4短時間GPU resume**：synthetic canonical fixture、原則224px/local32/K65536、uninterrupted self `meta_einsum`3step（checkpoint after step2、full trace relevant step）、同checkpointから別fresh resume runを3stepまで実行。CPU `--pair-kind formal_resume --gate A4`でstep3を比較、native/sidecar mode/SHA/RNG/optimizer/Queue/BN/DDPの復元が一致するか判定。**Benchではresume不可（現実装が拒否）**。
4. **A3長いGPU Queue一巡**：Meta公式referenceと自作self `meta_einsum`を同一**synthetic** canonical fixture（224px/local32/K65536/seed0）から独立freshで`run_kind=parity gate=A3 training.max_steps=1025 'trace.full_steps=[1,1024,1025]'`。同batch global64でstep1024に初期random Queueが全置換され`queue_ptr=0`、step1025に`queue_ptr=64`となる。毎step軽いtraceとfull step traceを区別し、`--pair-kind formal_implementation`で比較。**既存runnerはparityで毎step traceを保存し、CPU↔GPU同期やI/Oコストがある**ため、正式長時間job前に容量/速度見積りが必要。A1の小K wrap検証と実K長期を混同しない。
5. **A5本計測**：A5-0がPASSした場合、必要な時点で独立fresh100step×2本（monitor OFF → Light）、warmup10、window10、Comet OFF。ImageNet DataLoaderのサンプル順・seed・構成を一致させ、VRAM・throughput・DataLoader wait・measurement windowを比較。**A3/A4のheavy trace時間をA5性能計測へ使わない**。逆順再演はノイズが大きい場合のオプションであり初回必須ではない。

**A5 pilot→A4→A3→A5 paired**は、いずれかの工程で止まっても後続を無許可に流さず、失敗理由と保存済みartifactが追跡できる提案。単一コマンドの一括起動でも、**各Gateの実行許可は別個に表現**する。

## 3. 確定値の候補（正式採用前）

| Axis | Proposed |
|---|---|
| Model | Random-init ResNet50＋MLP dim128。Meta固定参照clone無改変、自作S1の通常InfoNCEは明示`meta_einsum` |
| Global parameters | FP32 AMP OFF TF32 OFF、seed0、2GPU NCCL、global64（local32×2）、K65536、m0.999、T0.2、SGD lr0.0075/momentum0.9/wd1e-4、epoch0 Cos horizon10 |
| A3 dataset | synthetic、224px、1,025stepを決定論的に再生成。**ImageNet 1,025step real fixtureは容量が大きいので黙って採用しない** |
| A3 full trace | [1,1024,1025]。light traceは毎step、compareには両run共通のStep/rank/phaseが必要 |
| A4 fixture | synthetic、224px、3step。checkpoint step2、uninterrupted 3stepとresume後step3の対照 |
| A5 dataset | real ImageNet train、data.num_workers=8/rank、image224、sampler epoch0、20step OFF pilot、OFF/Light各100step、Comet OFF |
| Threshold | **A3/A4もA2で固定した `rtol=1e-5`・`atol=1e-6`・離散exactを独立に事前承認することを推奨**。A2の批准値がA3/A4に自動適用されたとは主張しない。A3の長い累積差が閾値超過しても後から緩めない |
| Log/trace | 全run新規ディレクトリ。既存実験artifact上書きなし。大きなfixture.pt/trace/GPU logをResearch Workspace GitHubへpushしない |
| A5 overhead | feasibility PASSと定量差の実測を分離し、根拠なく%許容上限を設けない。必要時に上限を別途事前固定 |

## 4. 実装提案：最小オーケストレーション（コード変更は未承認）

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

## 6. Ambiguity Gate／承認待ち

### Blocking

1. **統合の意味**：既存runnerを呼ぶ「共通ランチャー＋検証summary」のコード追加と、A3/A4/A5のGPU実験run許可を切り分ける。推奨は**まずランチャーとCPU dry-runだけを実装**、GPUは段階別ユーザー実行。
2. **A3のデータと予算**：synthetic・224px・K65536・1025step×公式/自作か、real ImageNet1,025step fixtureを要求するか。推奨は**synthetic**（現在runnerのA3 docstring/親Gateの最小検証）。
3. **A3/A4閾値**：両GateでA2の`rtol=1e-5/atol=1e-6`・離散exactを**事前固定するか**。推奨は固定採用。A4のrank>0の最初のresume前BN bufferは既存formal_resumeでinformationalな仕様を継承する。
4. **A4再開point**：既存のstep2 checkpoint→step3に限定するか、長時間checkpointを要求するか。推奨は3step独立fresh復元を初回正式Gateとし、長いresumeは後段。
5. **A5ペア予算**：20step pilot、OFF100/Light100、必要なら逆順100+100を別途。推奨は最初の220stepのscopeまでで、追加逆順は別検証。
6. **Git/実行権限**：コードをGitHubリモートの`dev`へcommit/pushすることと、GPU runを起動することは別許可。ユーザーの今回の「まとめて実装」は研究コード修正意向として扱えるが、未承認のscientific delta・push・長時間runの包括承認とはしない。

### Non-blocking

- 既存repoと同じ命名規則でrun-idをつける／summaryの軽微な見出し、shellのログfile名。
- 他の実験に影響しない範囲のコード配列・formatの細部。既存artifactやdefaultの意味は変えない。

**Readiness**：上記blockingが未解決のため`draft`。採用後に仕様を`approved`へ昇格し、`engineering-task`で最小変更を実装する。ユーザーが「同じ推奨を採用する」と回答すれば1〜5の**設計推奨**は採用可能だが、**GitHub commit/pushとGPU runは別の明示承認**が必要。

## 7. Handoff / Current status

- **Spec**：本ファイル（`draft`、2026-10-20）。
- **既存実装の正本**：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、commit `dc502e416aa26ed4e0791767fa5a72f7843e399d`。GitHubと研究サーバdirtyは別扱い。
- **更新方針**：最小オーケストレーションとdry-run testsを検討。MoCo/InfoNCE/trace/optimizer/Queueの本体変更なし。
- **実装成功条件**：ステージ境界の独立性、出力衝突防止、停止伝播、設定/閾値/検証契約と非回帰。GPUによるA3/A4/A5実験結果は別Evidence。
- **承認外**：公式Meta clone修正、既存baseline破壊、Queue/batch/precision変更、A3 1025step/M/S1 GPU jobの自動起動、A5 100step GPU job自動起動、10epoch/ViT/LoRA統合。
