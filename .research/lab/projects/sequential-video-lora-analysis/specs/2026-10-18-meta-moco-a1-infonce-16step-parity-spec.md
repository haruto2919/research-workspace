---
project: sequential-video-lora-analysis
spec_type: implementation-and-verification
status: approved
title: Meta MoCo v2 and self MoCo A1 InfoNCE 16-step numerical parity
created: 2026-10-18
last_updated: 2026-10-19
authority: user-adopted B1-B10 contract and explicit 2026-10-19 approval of this spec and its 2GPU 16-step run
parent_spec: 2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md
predecessor_spec: 2026-10-18-meta-moco-a1-infonce-arithmetic-diagnostic-spec.md
workspace_repository: haruto2919/research-workspace
workspace_branch: main
workspace_base_commit: 261f86d7562f8baa0dbd3f89a67d79c30c5a4832
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: 13a962b3beffa15807be2e13472aee3e6f734d09
implementation_verified_commit: 1ce9badcbae3e2df08fa400944e4c385785063f9
approved_on: 2026-10-19
gpu_run_authorized_on: 2026-10-19
implementation_dirty_at_spec_review: false
reference_repository: facebookresearch/moco
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
fp32_rtol: 1.0e-5
fp32_atol: 1.0e-6
fp32_tolerance_status: fixed
---

# Meta公式MoCo v2と自作MoCoのA1・16step数値一致検証（approved）

> **Status: approved（2026-10-19にユーザーが明示承認）**。採用済みB1〜B10の契約を維持し、A1・16step判定のFP32閾値`rtol=1e-5, atol=1e-6`を判定run前に**fixedとして事前登録**した。2026-10-19、ユーザーはM/Mb/S0/S0b/S1/S1bの2GPU・NCCL・16step正式runも**別途明示承認**した。研究コード側では201件のCPUテストPASSが報告され、`dev@1ce9badc...`のclean状態とMeta固定cloneのclean状態をユーザーが確認済み。これは**GPU run完了やparity PASSを意味しない**。研究コードの新たな編集・commit/pushや追加の長時間runへ許可を拡張しない。

## 1. 目的、仮説、主張範囲

### 1.1 目的

2026-10-18のA1・2GPU・1step診断では、自作MoCoのInfoNCE算術だけをMeta固定commitと同じ`einsum`方式へ切り替えたS1が、Meta公式Mに対して全Query勾配、SGD後Query parameter、SGD momentumを暫定許容差内に収めた。全Query勾配のmax absolute errorとglobal relative L2は両rankで0だった。

本specの目的は、同じ初期state・fixture・2GPU DDP条件から16 optimizer stepを継続し、S1とMの数値一致が、SGD momentum、Key EMA、Queue二巡、Shuffle-BN、DDP、BatchNorm bufferを含む反復状態更新後も維持されるかを確認することである。

### 1.2 研究上の仮説

- **主仮説H1**：MとS1は、第7節の記録範囲において、全16stepで事前固定したFP32基準`rtol=1e-5, atol=1e-6`を満たす。
- **再現性仮説H2**：M、S0、S1は、それぞれ同じ条件のfresh rerunと全比較項目でbitwise一致する。
- **対照仮説H3**：S0はstep 1のInfoNCE算術差を再現し、S0とS1はその後に異なる軌道へ分岐し得る。S0の分岐はS1のMに対するparity判定とは分離して報告する。

### 1.3 成功時に主張できること

「固定A1 synthetic small-K条件、2GPU/NCCL、16step、記録済みsmall trace全stepおよびfull trace step 1/8/9/16の範囲で、自作`meta_einsum`経路はMeta公式と事前固定FP32許容差内で一致し、各方式の再実行はbitwise再現した」と主張できる。

全16stepの全tensorを保存・比較した、K65536でQueue一巡を確認した、ImageNet実画像で一致した、表現品質や10epoch性能が同等だった、通常MoCoの既定実装をMeta式へ変更すべきだ、とは主張しない。

## 2. Authority、Evidence、現行コード

### 2.1 Authorityの優先順位

1. 本spec作成を依頼した現在のユーザー指示。
2. [B1〜B10採用決定](../../../../secretary/notes/brainstorm/2026-10-18-moco-a1-infonce-16step-parity-decisions.md)末尾の「ユーザー採用」節。
3. [A1 InfoNCE 1step GPU Evidence](../experiments/2026-10-18-meta-moco-a1-infonce-arithmetic-gpu-one-step-evidence.md)。
4. [A1 1step診断spec](2026-10-18-meta-moco-a1-infonce-arithmetic-diagnostic-spec.md)。
5. [親Spec A](2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md)。
6. [1step実装・CPU検証](../experiments/2026-10-18-meta-moco-a1-infonce-arithmetic-diagnostic-implementation-verification.md)、[実装静的監査](../code-analysis/2026-10-18-a1-infonce-diagnostic-implementation-audit.md)、[勾配差解析](../code-analysis/2026-10-17-meta-moco-a1-gradient-divergence-explain.md)、[2026-10-15 MTG記録](../meetings/2026-10-15-mtg.md)。

プロジェクトREADMEはプロジェクト全体のSSOTだが、最終更新が2026-10-15であるため、今回のA1判断では上記の新しい採用決定とEvidenceを優先する。

### 2.2 確認済みEvidenceと未確認の境界

| 項目 | 現在確認できていること | このspecで未確認のこと |
|---|---|---|
| M再現性 | 既存A1 16step、full step 1/8/9/16の16,000/16,000比較がbitwise一致 | 本spec実装後commit・同一provenanceでのfresh M rerun |
| S0再現性 | 既存A1 16stepの16,000/16,000比較がbitwise一致 | 本specの正式run setでのfresh S0 rerun |
| S1 step 1 | Mとの差が全Query勾配・SGD後Query parameter・momentumで暫定閾値超過0。全Query勾配のmax absとrelative L2は0 | step 2〜16、Queue一巡後、BN/EMA反復後の一致 |
| S0 step 1 | 全Query勾配の暫定閾値超過961要素/rank、momentum超過961要素/rank | 16step専用runでの対照結果 |
| 閾値 | `rtol=1e-5, atol=1e-6`は親Spec Aで事前候補、1step診断では暫定物差し | 2026-10-19の本spec承認によりA1・16step用にfixed事前登録済み。GPU run未実施 |

数値は既存Evidenceに記録されたユーザー提供GPU実測であり、このspec作成中に再実行していない。

### 2.3 現行実装の読み取り結果

調査時点の研究コードは`dev@13a962b3beffa15807be2e13472aee3e6f734d09`、worktreeはcleanである。研究コード側に追加の`AGENTS.md`は存在しない。

| component | 現行状態 | 16stepに必要な差分 |
|---|---|---|
| `self_supervised/moco/ddp_moco.py` | `self`と`meta_einsum`を実装済み。既定は`self`。modeはstate_dictへ入らない | **変更不要** |
| `scripts/parity/moco_imagenet_parity.py` | `meta_einsum`をA1・1step・full `[1]`だけに制限。全step small traceと指定step full traceを保存 | A1・16step・full `[1,8,9,16]`を第二の許可profileとして追加。正式閾値statusをartifactへ記録 |
| `scripts/parity/compare_moco_parity.py` | `implementation`、`rerun`、`resume`、1step専用`a1_info_nce`を実装 | 16step S0/S1専用pair、全step scalar metrics照合、固定閾値との整合checkを追加 |
| `scripts/parity/judge_moco_a1_info_nce.py` | 1stepの因果診断専用 | **変更しない**。16step用judgeを新規分離 |
| `conf/moco_imagenet_parity.yaml` | rtol/atolはprovisional、`meta_einsum`注記は1step限定 | `verification.threshold_status`を追加し、既定はprovisionalのまま維持 |
| `test/model/test_moco_ddp.py`、`test/model/test_moco_parity_harness.py` | 1step mode、DDP数学、比較器、judge、拒否条件を検証済み | 16step profile・Queue境界・比較契約・5状態judgeのtestを追加 |

既存実装は、Query forward → Key EMA → Shuffle-BN → Key forward/unshuffle → enqueue前Queueによるlogits/loss → global key enqueue → backward → SGDの順をtraceする。Query/Keyのcanonical parameter名にはResNet50 backboneとMLP projectorの両方が含まれる。

## 3. 採用済みDecision Contract

次は未決の提案ではなく、本specの入力として採用済みである。

| ID | 固定する決定 |
|---|---|
| B1 | A1 synthetic small-K、16stepのconfirmatory数値照合に限定する |
| B2 | M、S0、S1を比較し、M/Mb、S0/S0b、S1/S1bの各fresh rerunで再現性を確認する |
| B3 | 同一fixture、2GPU/NCCL、K64、local batch4、64px、FP32、16step等を固定する |
| B4 | small traceは全16step、full traceは`[1,8,9,16]`とし、Queue一巡・再置換を確認する |
| B5 | 診断用`meta_einsum`だけを16stepへ拡張し、通常MoCoの既定方式は変更しない |
| B6 | 既存比較器を再利用しつつ16step専用契約・judgeを分離し、1step judgeを維持する |
| B7 | 正式run前にFP32閾値を固定し、bitwise一致は別の強い指標として報告する |
| B8 | 全16stepのsmall/discreteと、指定stepのfull tensorを判定対象とする |
| B9 | clean commit、provenance、同一fixture、再実行、CPU test、comparison/judgement保存を必須とする |
| B10 | 異常時停止、Comet OFF、既存artifact保護、GPU実行の別承認を維持する |

## 4. Scopeと明示的な対象外

### 4.1 実装scope

- A1 16stepに限定した`meta_einsum`許可profile。
- 閾値の固定状態をresolved config、run report、comparison、judgementへ伝搬する最小変更。
- 16step S0対S1の制御比較pair。
- M/S1正式parity、M/S0対照、3方式の再現性を統合する16step専用judge。
- 対応するunit、CPU gloo DDP、設定拒否、比較器、judge、既存非回帰test。

### 4.2 対象外

- 通常MoCoまたは`StandaloneMoCo`の既定InfoNCEを`meta_einsum`へ変更すること。
- Meta公式source、固定cloneの変更。
- A2、A3、A5、K65536での1025step、ImageNet実画像による長時間学習。
- 10epoch性能比較、Linear Probe、LoRA・ViT・Sequential Loaderとの統合。
- checkpoint/resume、Comet送信、AMP/TF32、SyncBatchNormの導入。
- GPU runの実行、研究コードのcommit/push、Research Workspaceのcommit/push。

## 5. 固定実験条件とrun matrix

### 5.1 共通条件

| 項目 | 固定値 |
|---|---|
| fixture | 既存A1 synthetic fixture。同一file SHA-256、canonical state SHA-256、rank別view hash |
| model | random-init ResNet50、標準7x7 stem、MLP 2048→2048→128、dim128 |
| MoCo | K64、m=0.999、T=0.2、enqueue前Queueをnegativeに使用、global batch8 |
| optimizer | SGD、LR0.0075、momentum0.9、weight decay1e-4、nesterov=false。全step epoch 0のLR |
| distributed | 2GPU、1 process/GPU、NCCL、local batch4、`broadcast_buffers=true`、plain BN |
| input | 64px、seed0、固定rank割当、同じ2-view tensor、同じShuffle-BN seed/permutation |
| precision | FP32、AMPなし、TF32 off、cudnn deterministic=true、benchmark=false |
| length | 16 optimizer steps、resumeなし、checkpointなし |
| trace | small/discreteはstep 1〜16、fullはstep `[1,8,9,16]` |
| logging | Comet OFF、fresh独立output directory、上書き禁止 |
| threshold | `rtol=1e-5, atol=1e-6`、本spec承認時に`fixed` |

### 5.2 必須run

| ID | implementation | InfoNCE mode | 用途 |
|---|---|---|---|
| M | `meta_reference` | Meta固定式 | 正式reference |
| Mb | `meta_reference` | Meta固定式 | Mのfresh再実行 |
| S0 | `self` | `self` | 従来算術の負の対照 |
| S0b | `self` | `self` | S0のfresh再実行 |
| S1 | `self` | `meta_einsum` | 正式candidate |
| S1b | `self` | `meta_einsum` | S1のfresh再実行 |

各runは別process・別directoryから開始する。M/S0/S1のprimaryとrerunを同じdirectoryから複製してはならない。

### 5.3 既存Mの再利用条件

既存Mは、結果の良し悪しではなく次のprovenance predicateだけで機械的に採否を決める。

- fixture file/state SHA、全16stepのinput hashとShuffle-BN permutation、full stepsが一致。
- Meta cloneが固定commitでclean、diff empty。
- runner/harnessを実行した研究コードcommitが最終実装commitと同一でclean。異なるcommitを許す場合は、Mで実行されるrunner・state mapping・trace schema・configの全file hashが最終commitと一致する機械的証拠が必要。
- Python、torch、torchvision、CUDA、cuDNN、NCCL、GPU型、`CUDA_VISIBLE_DEVICES`、rank対応、backend flagsが正式runと一致。
- runがcompleted、全16step、required trace完備、全within-run check PASS。

どれかを証明できなければ既存Mを再利用せず、正式閾値固定後にfresh Mを作る。既存Mを採用してもMbは別directoryでfresh実行する。

## 6. 実装仕様と変更範囲

### 6.1 runner/configの許可profile

`diagnostic.info_nce_impl=meta_einsum`は、共通条件を満たす次の2profileだけを許可する。

| profile | `training.max_steps` | `trace.full_steps` | threshold status |
|---|---:|---|---|
| 既存1step診断 | 1 | `[1]` | `provisional`。既存judgeの意味を維持 |
| 本spec 16step | 16 | `[1,8,9,16]` | `fixed`、rtol/atolは本specの値と完全一致 |

両profileの共通必須値は、`implementation=self`、`run_kind=parity`、`gate=A1`、synthetic fixture、K64、localB4、world size2、64px、resumeなし、checkpointなし、Comet OFFである。profileに一致しない`meta_einsum`はspawn前、model/EMA/Queue変更前に理由付き`ValueError`で拒否する。

`diagnostic.info_nce_impl=self`の既存A1〜A5経路は制限しない。ただし16step正式run setでは第5節の条件をjudgeが要求する。

`verification.threshold_status`を`provisional | fixed`としてconfigへ追加し、既定値は`provisional`とする。runnerはrtol、atol、statusをresolved config、run report、summaryへ保存する。`fixed`という文字列だけでは不十分で、16step profileでは数値も本specと一致しなければ拒否する。

### 6.2 既存比較器の再利用と最小拡張

既存`implementation` pairをM対S0・M対S1、`rerun` pairをM対Mb・S0対S0b・S1対S1bに再利用する。

新しいpair kind `a1_info_nce_16step`を追加し、S0をreference、S1をcandidateとする。このpairは因果対照の契約確認と差の伝播記録用であり、S0/S1のfloat差によってPASS/FAILを付けない。契約が有効なら`DIAGNOSTIC`、契約違反なら`INVALID`とする。

比較器は、CLIで受けたrtol/atol/statusが両runのresolved configおよびrun reportと一致することを検査する。16step正式比較で`provisional`を指定した場合、または値が異なる場合はPASS/FAILを出さない。

全stepの`metrics`について、既存のLRだけでなく`loss`、positive/negative cosine mean、Query gradient norm、Query update normをfloat比較し、Queue pointerをexact比較する。すでにsmall traceで比較する値と重複してもよいが、欠損を黙って無視しない。

### 6.3 16step専用judge

`scripts/parity/judge_moco_a1_info_nce_16step.py`を新規作成し、既存1step judgeは変更しない。新judgeは次の6 reportを入力とする。

1. M vs S0：`implementation`
2. M vs S1：`implementation`
3. S0 vs S1：`a1_info_nce_16step`
4. M vs Mb：`rerun`
5. S0 vs S0b：`rerun`
6. S1 vs S1b：`rerun`

judgeは総合statusに加え、`formal_fp32_parity`、`m_vs_s1_bitwise`、`rerun_reproducibility`、`s0_control_outcome`を別fieldとして出す。`judgement.json`と`judgement.md`は新規directoryに保存し、既存ファイルを上書きしない。

### 6.4 変更対象

- `scripts/parity/moco_imagenet_parity.py`
- `scripts/parity/compare_moco_parity.py`
- `scripts/parity/judge_moco_a1_info_nce_16step.py`（新規）
- `conf/moco_imagenet_parity.yaml`
- `test/model/test_moco_parity_harness.py`
- 必要な16step DDP数学testに限り`test/model/test_moco_ddp.py`

`self_supervised/moco/ddp_moco.py`、`standalone_moco.py`、Meta cloneは変更しない。実装中にmodel側変更が必要と判明した場合はscope拡張として停止し、再レビューする。

## 7. 比較項目とtrace契約

### 7.1 全16stepで判定するsmall/discrete

| 対象 | 時点 | 判定 |
|---|---|---|
| rank別`im_q` / `im_k` hash | forward前 | exact |
| Shuffle-BN seed、`idx_shuffle`、`idx_this` | forward前 | exact |
| `z_q`、`z_k_shuffled`、q、k | Query/Key forward後 | fixed FP32 tolerance |
| positive logit | logits生成後 | fixed FP32 tolerance |
| targets | logits生成後 | shape/dtype/value exact |
| loss | forward後 | fixed FP32 tolerance |
| global enqueued keys | enqueue後 | fixed FP32 tolerance、rank順 |
| Queue pointer before/after | broadcast後/enqueue後 | exact |
| LR | SGD前 | exactな設定値として一致 |
| positive/negative cosine mean、Query grad norm、Query update norm | 各step終了時 | fixed FP32 tolerance |
| phase存在、rank存在、step番号 | 各trace | exact、欠損不可 |

phaseごとのparameter/bufferのfloat64集約statsは診断用であり、full tensor比較の代用にしない。stats差はreportへ残すが、それだけで「全parameter tensor一致」と主張しない。

### 7.2 step 1/8/9/16で判定するfull tensor

| phase | full group | 内容と判定 |
|---|---|---|
| `before_forward` | Query params、Key params | ResNet backbone＋MLP projector全parameterをfixed toleranceで比較 |
| `before_forward` | Query BN、Key BN | 同一rank・同一時点のrunning buffersを比較。整数counterはexact |
| `before_forward` | Queue、pointer | Queue floatはfixed tolerance、pointerはexact |
| `after_buffer_broadcast` | Query/Key BN、Queue、pointer | DDP rank0 broadcast直後。同一rank M/S1比較。全rank Queue/pointer一致 |
| `after_query` | Query BN | Query forward後のbuffer。同一rank M/S1比較 |
| `after_key_ema` | Key params | EMA後parameter。backbone＋projector全体を比較 |
| `after_key_forward` | Key BN | shuffled local minibatch処理後。同一rank M/S1比較 |
| `after_logits_before_enqueue` | logits、Queue snapshot | positive＋全K negative logits、実際に使用したenqueue前Queue |
| `after_enqueue` | Queue、pointer | global keyをrank順enqueueした後の全Queueとpointer |
| `after_backward` | Query gradients | backbone＋projectorの全Query gradient |
| `after_sgd` | Query params、SGD momentum | 更新後全Query parameterと全momentum buffer |

full groupでparameter名、shape、dtype、要素数の対応が欠ける場合は値を比較せずINVALIDとする。floatはMetaをreferenceとした要素単位規則を使い、tensorごとのmax abs、mean abs、max relative、mismatch scalar数、relative L2を保存する。

### 7.3 Queue境界の意味

K64/global batch8なので、各stepでpointerは8進む。

| step | logitsに使うenqueue前Queue | enqueue後の必須状態 |
|---:|---|---|
| 1 | 初期random Queue | columns 0〜7をstep 1 keyで置換、pointer 8 |
| 8 | step 1〜7 key＋未置換最後8列 | 初期random keyが全消失、pointer 0 |
| 9 | step 1〜8 keyのみ | 最古のstep 1 keyをstep 9 keyで置換、pointer 8 |
| 16 | step 9〜15 key＋残るstep 8 key | Queue全体がstep 9〜16 key、pointer 0 |

step 8、9、16ではQueue snapshot、enqueue列、全Queue、pointerをfull比較する。自作側の補助的`real_key_count`はMetaに対応状態がないため正式parity対象にせず、存在する場合だけ診断値として64で飽和することを記録する。

### 7.4 BatchNorm、DDP、Shuffle-BN

- `broadcast_buffers=true`により、各forward冒頭のbroadcast後はBN bufferとQueue/pointerがrank0と一致することをwithin-run checkする。
- Query/Key forward後のBN running statsはlocal minibatchに依存するため、rank0とrank1の相互一致を無条件に要求しない。同じrank・同じphaseのM対S1を比較する。
- Key EMAはparameterだけを対象とし、BN bufferをEMAしない。
- Shuffle-BNはglobal key画像、rank0生成permutation、そのbroadcast、rank slice、inverse permutationをexactに追跡する。
- backward後のQuery gradient、SGD後Query parameter/momentum、enqueue後Queue/pointer、EMA後Key parameterは全rank整合性をwithin-run checkする。

### 7.5 S0対S1の特別な制御契約

S0とS1はstep 1のlogits算術だけが介入差である。次をbitwise必須とする。

- resolved configの意味的な差が`diagnostic.info_nce_impl: self -> meta_einsum`だけ。
- repository commit/clean、environment、GPU/rank、fixture、初期state、Optimizer初期状態、全16stepの入力hashとShuffle-BN permutationが同じ。
- step 1の`before_forward`から、q、k、targets、logitsに使うQueue snapshotまでがbitwise一致。

step 1のlogits以降は差が生じてよい。step 1のgradient/SGD momentum差により、step 2以降のQuery/Key/BN/Queue/q/k/logitsが分岐しても契約違反としない。ただし全stepの入力hash、permutation、targets、Queue pointer、shape/dtype、phase構造などmodeに依存しないdiscrete契約は一致必須である。

## 8. FP32判定基準と事前固定

### 8.1 採用値

本specが`approved`になった時点で、A1 16step正式判定のFP32閾値を次へ固定する。

```text
rtol = 1e-5
atol = 1e-6
pass iff |candidate - reference| <= atol + rtol * |reference|
```

NaNは一致とみなさない。同符号Infの一致を許容する比較器挙動があっても、run validityのfinite checkが先にINVALIDとするため正式runには現れない。

### 8.2 固定可能と判断した根拠

- この値は親Spec AでGPU結果を見る前に候補として記録され、結果へ合わせて作った閾値ではない。
- 同一MetaのA1 16step再実行は、同じfull trace範囲を含む16,000/16,000比較でbitwise一致した。測定済み環境の再現性floorは候補閾値より十分小さい。
- 同一S0再実行も16,000/16,000でbitwise一致した。
- S1の1stepはMに対し、全Query勾配のmax absとrelative L2が0、SGD momentumの超過も0だった。候補閾値をS1に有利になるよう緩和する必要はなかった。
- B7でこの候補を正式run前に固定する方針がユーザー採用済みであり、本spec承認が事前登録点になる。

以上から追加のcalibration runを閾値固定のblocking条件にはしない。2026-10-19の明示承認で固定済みであり、正式runはこの閾値と承認済み条件のみで判定する。結果を見てから変更しない。

### 8.3 適用対象とbitwiseの分離

- Mをreference、S1をcandidateとする全float比較へ正式適用する。
- M対S0にも同じ値を使うが、負の対照の記述用であり、S0のFAILをS1の正式FAILに混ぜない。
- shape、dtype、step/rank/phase、hash、permutation、targets、Queue pointer、parameter名は誤差許容なしで一致する。
- M対S1のbitwise一致数を別集計する。正式FP32 PASSはbitwise PASSを意味しない。
- M/Mb、S0/S0b、S1/S1bの再現性は、FP32 allcloseではなく全judged項目のbitwise一致を必須とする。

正式run結果を見た後に同じspecの閾値を変更しない。妥当性を再検討する必要が生じた場合、現runは現閾値のまま`FAIL`、`BLOCKED`、`INVALID`の該当状態で保存し、根拠と新しい事前登録を持つsuperseding specを別途承認する。

## 9. 比較器・判定器の状態契約

### 9.1 総合status

16step judgeは次の5状態を排他的に返す。

| status | 定義 |
|---|---|
| `PASS` | 必須run/reportがすべて有効・完備、M対S1がfixed FP32基準を満たし、3方式のrerunが全judged項目bitwise一致 |
| `FAIL` | 比較条件は有効で完備しているが、M対S1のfloat/exact出力が不一致、または必須rerunがbitwise再現しない |
| `BLOCKED` | 実行を試みたがOOM、NCCL error/hang、外部resource、固定閾値未承認等で有効な全run setを完成できない。数値parityを判定しない |
| `INVALID` | completed artifactはあるが、commit/dirty、fixture、config、環境、rank、初期state、trace、finite/within-run check等の比較契約違反がある。数値parityを判定しない |
| `NOT_RUN` | 必須GPU runまたは比較がまだ開始されていない、または入力reportが存在しない |

実行未開始は`NOT_RUN`、開始後の外部要因による未完了は`BLOCKED`、値を読む前提が壊れたcompleted setは`INVALID`として区別する。

### 9.2 formal parityとbitwise表示

総合statusとは別に次を保存する。

- `formal_fp32_parity`: `PASS | FAIL | NOT_JUDGED`。
- `m_vs_s1_bitwise`: `ALL_JUDGED_BITWISE | NOT_ALL_BITWISE | NOT_JUDGED`。
- `rerun_reproducibility`: M、S0、S1ごとに`BITWISE | NOT_BITWISE | NOT_JUDGED`。
- `s0_control_outcome`: `DIVERGED | WITHIN_FIXED_TOLERANCE | NOT_JUDGED`。S0が予想外にMと一致してもM/S1 parityの事実を改変せず、対照が再現しなかったことを別表示する。

### 9.3 差分の局在化

comparisonとjudgeは、不一致があれば少なくとも次を保存する。

- 最初のstep、rank、phase、group、tensor。
- mismatchがexactかfloat tolerance超過か。
- shape、dtype、numel、mismatch scalar数、max/mean absolute error、max relative error、relative L2。
- reference/candidateのrun ID、commit、fixture SHA、mode。
- full trace以外では「small trace上の初出」であること。

順序はstep昇順、phaseの実行順、rank昇順、group/tensor名順で一意にする。後続差分も集計し、最初の1件だけで全原因を断定しない。

## 10. 実装後のCPU verification plan

GPU許可を求める前に、次をCPUでPASSさせる。

### 10.1 model/DDP数学

- 既存1step`meta_einsum` literal式、loss、`dL/dq`のbitwise testを維持。
- Tiny backbone＋gloo 2rankで16stepの`meta_einsum`を実行し、EMA、SGD momentum、Queue/pointer、enqueue順、DDP gradient、BN broadcastを独立参照と照合する。
- K64/global batch8でstep 8/9/16のpointerとQueue置換範囲を検査する。

### 10.2 profile・拒否test

- 既存1step profileと新16step profileだけが`meta_einsum`を許可される。
- 15/17step、full steps違い、K/B/world/image違い、Comet ON、resume/checkpoint、A2以降、imagenet fixture、`threshold_status=provisional`、rtol/atol違いをspawn前に拒否する。
- `self`既定modeの既存Gateを拒否しない。

### 10.3 comparator/judge test

- 16step S0/S1でstep 1 logits入力までbitwise一致し、step 2以降のfloat差を契約違反にしない。
- step 1 pre-logits改変、mode以外のconfig差、commit/dirty、fixture/environment/rank差、全stepのdiscrete差を`INVALID`にする。
- M/S1のfixed tolerance境界内外、NaN、missing phase/tensor、full steps欠損を検査する。
- M/Mb、S0/S0b、S1/S1bのうち1項目でも非bitwiseなら再現性FAILになる。
- `PASS`、`FAIL`、`BLOCKED`、`INVALID`、`NOT_RUN`の各synthetic report testを用意する。
- formal FP32 PASSだがM/S1は非bitwise、という状態を正しく分離する。
- comparison/judgementの既存directory上書きを拒否する。

### 10.4 既存非回帰

最低限、次を同一Python環境で実行し、件数、pass/fail/skip、所要時間を保存する。

```text
test/model/test_moco_standalone.py
test/model/test_moco_cifar10_reporting.py
test/model/test_moco_ddp.py
test/model/test_moco_parity_harness.py
```

通常MoCo、Standalone MoCo、CIFAR-10、既存1step診断、A1〜A5 runnerの既定経路、既存checkpoint/state_dictに回帰がないことを確認する。既存失敗として除外する場合は、変更前commit・同一環境での再現Evidenceが必要である。

## 11. 実装後のGPU verification plan（別許可）

### 11.1 実行前Gate

- 研究コードが実装後commit Hでclean、branch `dev`。6 runすべて同じH。
- Meta cloneが固定commitでclean、diff empty。
- 本specが`approved`で、resolved configにfixed閾値が記録されている。
- CPU verificationが全PASS。
- fixtureとoutput rootをread-only確認し、新しいrun directory名を確保。
- GPU 2枚、rank対応、CUDA/cuDNN/NCCL/backend flagsを記録。
- 2GPU・16step runの明示的な実行許可を取得。

### 11.2 実行と比較

M、S0、S1、Mb、S0b、S1bを第5節の条件で実行する。順序によって条件を変えず、各run前後にrepository clean/commitを確認する。各run完了後にrun単体のwithin-run checksを確認し、不合格なら残りを自動継続しない。

6比較reportを`--threshold-status fixed --rtol 1e-5 --atol 1e-6`で作成し、16step judgeへ入力する。比較対象scopeがsteps 1〜16、full steps `[1,8,9,16]`、world size2であることをjudge自身が検査する。

### 11.3 GPU実行許可の境界

実装承認はCPU testまでの許可であり、GPU runを含まない。GPU許可が出てもA2/A3/A5、1025step、ImageNet、10epoch、Comet送信、追加診断runへ自動拡張しない。

## 12. PASS/FAIL条件と停止条件

### 12.1 PASS

次をすべて満たした場合だけ総合`PASS`とする。

1. 6 runが同じ有効条件で16step completed。
2. 全within-run必須checkが16step×2rankでPASS、NaN/Infなし。
3. M対S1の全16step small/discreteとfull step 1/8/9/16がfixed基準内。
4. Query/Key/Projector、gradient、SGD parameter/momentum、EMA、Queue、BN、Shuffle-BN、DDPの第7節契約を満たす。
5. M/Mb、S0/S0b、S1/S1bが全judged項目でbitwise一致。
6. S0/S1制御契約がvalid。S0の数値的な発散自体は総合PASSの必要条件にせず、別fieldで報告する。

### 12.2 FAIL

run setが有効・完備しているにもかかわらず、M対S1にfixed tolerance超過、outputのexact mismatch、Queue/ptr/更新順の差が1件でもある場合はFAIL。3方式のいずれかのrerunが非bitwiseの場合も、B2/B9の再現性要件を満たさないためFAILとする。

### 12.3 即時停止

次を検出したrunは停止し、原因、step、rank、stage、最後の正常phase、GPU memory情報、部分artifactを保存する。

- OOM、NCCL error/timeout/hang、process/rank loss。
- NaN/Inf、missing gradient、異常なstate mutation。
- fixture/state/input/permutation/config/commit/dirty不一致。
- Queue pointer、enqueue順、rank間Queue、DDP gradient、EMA、SGD式のwithin-run check失敗。
- Meta cloneのcommit/diff違反、Cometが有効、AMP/TF32等の条件違反。

自動でbatch/K/precision/backend/timeoutを変えず、fallback、resume、既存artifact上書きを行わない。条件変更が必要ならBLOCKEDとして報告し、別承認を得る。

## 13. Artifact、provenance、再現性

### 13.1 runごとの必須artifact

- `resolved_config.yaml`
- `run_report.json`
- `provenance.json`
- `fixture_manifest.json`
- `resource_benchmark.json`
- `verification_summary.md`
- `trace/rank{0,1}/step_000001.pt`〜`step_000016.pt`
- failure時のrank report、traceback、保存できた部分trace

run reportにはthreshold status、InfoNCE mode、completed/full stepsを含める。provenanceにはself repository commit/branch/dirty paths、Meta commit/diff、environment、GPU/rank、backend flagsを含める。

### 13.2 comparison/judgement artifact

- 6比較それぞれの`parity_report.json`とMarkdown summary。
- 16step judgeの`judgement.json`と`judgement.md`。
- first failure/difference、phase別件数、bitwise件数、full step別gradient/SGD集約。

### 13.3 保存境界

研究サーバのraw trace、Queue dump、checkpoint、GPU logはローカルartifactとして保持し、Research Workspace/GitHubへコピーしない。GPU実行後は別依頼または許可により、条件、hash、判定、主要数値、ローカル相対path、Evidence境界だけを`experiments/`へ記録する。

既存run/comparison/judgementを上書きしない。再利用はread-onlyとし、新しい比較結果もfresh directoryへ保存する。Cometは全runでOFFとする。

## 14. 後方互換性と非回帰

- `DistributedMoCo`の既定`info_nce_impl=self`を維持する。
- `StandaloneMoCo`のconstructor、`moco_logits`、state_dict、checkpoint keyを変更しない。
- CIFAR-10 config/smoke/report、LoRA/ViT/Sequential Loaderを変更しない。
- Meta clone/sourceを変更・転載しない。
- 既存1step `meta_einsum` profile、`a1_info_nce` pair、`judge_moco_a1_info_nce.py`のschema・意味を変更しない。
- A1〜A5の通常runner経路、`implementation`、`rerun`、`resume`の既存契約を維持する。
- `verification.threshold_status`の追加は既定`provisional`とし、既存configを自動でfixedへ昇格させない。
- 新しい16step fieldを読む側はschema versionで判別し、旧artifactを書き換えない。

## 15. Ambiguity Gateとreadiness

### Blocking ambiguity

- **なし**。B1〜B10、正式閾値、比較scope、status、run matrix、変更対象、停止条件は、現在のユーザー指示・最新Evidence・現行コードから実装可能な粒度まで確定した。

本specは2026-10-19にユーザーが明示承認済みであり、blocking ambiguityは残らない。2GPU・16step GPU runについても同日に別途実行承認済み。ただし実行前のclean commit・fixture・GPU・依存関係のGateは継続する。

### Non-blocking

- helper/test関数の局所名、JSON内のfield並び順、Markdown整形は既存styleに合わせてよい。
- Hydra output directoryの時刻部分は既存機構を使う。既存artifactを上書きしない限り研究結果を変えない。
- 既存Mの再利用可否は第5.3節のpredicateで機械判定する。満たさなければfresh Mとするため追加の研究判断は不要。

### Spec readiness

本specはapproved。A1・16step用FP32閾値はfixedで事前登録済み。実装済みコードとCPU検証の完了記録を引き継ぐ。新たなblocking ambiguityが生じた場合は推測で補わず停止し、本specを改訂・再承認する。

## 16. Implementation Handoff（承認済み・GPU正式run待ち）

- approved spec: 本ファイル。2026-10-19にユーザーが明示承認、FP32閾値fixed事前登録
- 実装目的: Meta式S1がA1 small-K・16stepでMと数値一致を維持するかを、再現性とQueue/EMA/BN/DDPを含めて判定可能にする
- 基準repository/commit: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、基準`13a962b3beffa15807be2e13472aee3e6f734d09`、実装開始時にHEAD/dirtyを再確認
- 変更scope: parity runner/config/comparator、16step専用judge、関連testのみ。model算術は変更不要
- 対象外・維持条件: 通常MoCo/Standalone/CIFAR/Meta clone/1step judge/A2/A3/A5/長時間run/Comet
- success criteria: 第10節のCPU test PASS。GPU実行後は第12.1節の全条件
- 許可されている短時間検証: CPU unit/gloo/non-regression 201件PASS（ユーザー実行ログにより確認済み）
- 長時間runの許可状態: A1・16step正式6runのGPU検証だけ承認済み。A2/A3/A5/長時間学習は不許可
- GPU runの許可状態: 2026-10-19にユーザーがM/Mb/S0/S0b/S1/S1bの2GPU・16stepを明示承認。ローカル研究サーバで事前Gateを検査し、別dir fresh実行（ChatGPTからSSH実行はできない）
- 未検証予定: 16step正式結果、A2/A3/A5、K65536、ImageNet、10epoch、表現品質、Linear Probe

## 17. 承認・FP32事前登録の記録（2026-10-19）

- 承認主体：ユーザー。本specの承認と2GPU・16step正式runの実行を、直前の確認に対して「進めます」と明示。
- 比較閾値（GPU判定run前の事前登録）：**FP32 rtol=1e-5、atol=1e-6、threshold_status=fixed**。Meta公式をreferenceとした値ごとの比較。discrete/hashはexact、fresh rerunはbitwise。試験結果を見た後で改変しない。
- 実装済み研究コード：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、ユーザー報告のclean HEAD `1ce9badcbae3e2df08fa400944e4c385785063f9`。基準commitの`13a962b...`は変更前参照として引き続き保持する。
- 公式clone：`facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f`、cleanとユーザー報告。
- CPU非回帰：201/201件PASS、693.95秒。既存の実装検証記録にも別の実行ログを保存済み。
- 許可済みGPU範囲：A1 synthetic、K64、local batch4、2GPU/NCCL、64px、16step、full `[1,8,9,16]`のM・Mb・S0・S0b・S1・S1bの6fresh runと6比較・judge。Comet OFF、checkpoint/resumeなし、既存artifactを上書きしない。
- **未実施**：上記6GPU run・比較・正式FP32判定。現地サーバの実行開始前にGPU空き・fixture SHA・双方clean・環境/設定を再確認し、各runのwithin-run failureで停止する。
- 本承認はGPU実行のリモート自動実行や研究コードの新たな編集・commit/pushを意味しない。
