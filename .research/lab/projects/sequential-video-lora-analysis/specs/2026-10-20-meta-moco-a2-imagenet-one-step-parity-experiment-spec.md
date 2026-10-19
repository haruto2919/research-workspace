---
project: sequential-video-lora-analysis
spec_type: experiment
status: draft
title: "Meta MoCo v2 vs Self meta_einsum — A2 ImageNet-1K real-image 2GPU one-step FP32 parity"
created: 2026-10-20
last_updated: 2026-10-20
authority: "User requested next stage after normal InfoNCE opt-in CPU PASS; fixed A2 threshold and GPU execution not yet approved"
workspace_repository: haruto2919/research-workspace
workspace_branch: main
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: dc502e416aa26ed4e0791767fa5a72f7843e399d
reference_repository: facebookresearch/moco
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
input_evidence: "../experiments/2026-10-20-moco-normal-infonce-opt-in-cpu-verification.md"
gpu_gate: NOT_RUN
a2_formal_threshold: not_fixed
---

# A2実験spec（draft）：ImageNet実画像・2GPU・1stepでMeta公式と自作MoCoを数値比較

> **状態：draft／正式なGPU実験未許可。** 既存のfixture生成・独立GPU runner・offline comparatorを使う**実験計画**であり、新規のMoCo演算実装specではない。A2の正式FP32閾値、GPU初回計測/diagnostic許可、最終run許可が未確定のため、現時点ではA2のPASS/FAILを出さない。変更・実行・承認の境界は第11節。

## 1. 目的・主張の範囲

A1では**synthetic、64px、localB4×2、K64、2GPU、16step**で、固定Meta公式Mと自作S1 `meta_einsum`が比較対象16,256件すべてbitwise一致した。ただし実画像／224px／K65536／localB32での一致は未証明。

A2では**同じ初期重み・Queue・Rank入力2-view・Shuffle-BN乱数**で、Meta公式Mと自作S1（通常selector `moco.info_nce_impl=meta_einsum`）の**1 optimizer update**を別process・別directoryで実行して比較する。

成功した場合に主張できるのは「固定ImageNet実画像fixture、2GPU、指定した1stepの対象tensor/状態が事前固定FP32閾値内で公式と整合した」のみ。Queue一巡、long-run安定性、10epoch、分類表現品質、ViT/LoRA/動画の時間情報獲得は主張しない。

## 2. Authority・根拠とコード実体

- 採用済み [通常InfoNCE opt-in spec](2026-10-19-meta-infonce-normal-moco-opt-in-spec.md)：`status: implemented`、CPU Engineering Gate PASS、GPU NOT_RUN。通常経路でS1明示指定、config identity、trace/Checkpoint、正式pairに対応。
- [親Spec A](2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md)：`status: draft`。A2について明示的な個別承認・固定閾値を要する。本specを作るだけで親Spec A全体の承認にならない。
- [A1 16step GPU Evidence](../experiments/2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md) はsmall synthetic条件の先行事実であり、A2の正式閾値に自動継承しない。
- [2026-10-20 A2準備・事前確認](../../../../secretary/notes/brainstorm/2026-10-20-a2-imagenet-one-step-parity-next-steps.md)と[CPU 279件PASS](../experiments/2026-10-20-moco-normal-infonce-opt-in-cpu-verification.md)。
- GitHubコード：`scripts/parity/build_moco_parity_fixture.py`（CPUの固定実画像2-viewと初期state）、`scripts/parity/moco_imagenet_parity.py`（2rank DDP・各phase trace・GPU run）、`scripts/parity/compare_moco_parity.py`（CPU offline comparison、`formal_implementation`／`rerun`）、`conf/moco_imagenet_parity.yaml`。コード基準は`dev@dc502e416aa26ed4e0791767fa5a72f7843e399d`。
- **現在GitHubから確認できるのはコミット済みコードだけ**。研究サーバの実体はユーザーが共有した出力に依存。

## 3. 既に確認できたA0前提と未検証

| 項目 | ユーザーが研究サーバで確認したこと | A2への意味 |
|---|---|---|
| 自作/Meta clone | それぞれ`dc502e416aa26ed4e0791767fa5a72f7843e399d` / `a419ef282fc9885015674e77f58c59447206a27f`、両`git status --short`空 | 固定commit・cleanを現時点で確認 |
| ImageNet | `train/`・`val/`存在、`train` 1000 classes / 1,281,167枚、JPEG1枚（250×250）`PIL.Image.load`成功 | ImageFolder列挙・**1画像だけ**decode成功。全画像保証ではない |
| GPU | 2×NVIDIA RTX A4500、各total 20470MiB/free 20082MiB（確認時点） | 枚数・空き確認。FP32 backward時のpeak VRAMは未測定 |
| Python | torch 2.14.0+cu130、torchvision 0.29.0+cu130、CUDA13.0、`torch.cuda.is_available=True`、device_count=2 | CUDA discoveryはPASS。2rank NCCL実通信・本番条件のモデル実行は未確認 |
| テスト | 59/59（選択）、279/279（MoCo4ファイル、CPU/gloo） | Engineering/CPUはPASS。正式M vs S1 GPU比較ではない |

A0の**正式完了**は本specのpreflightで以下を確認・保存した時点で判断する：`reference` import/shim元とhash、2GPU/NCCL collective通信（既存runが正常通過した場合も証拠となる）、fixture対象画像の全64枚のdecode、GPU rank/deviceとfree/peak、環境version、両worktree clean。全ImageNet 128万枚のdecodeはA2必須ではない。Val splitはA2のloss入力には使わない。

## 4. 比較対象・不変条件（提案、未承認）

| 軸 | A2の条件 |
|---|---|
| Dataset | ImageNet-1K `/mnt/NAS-TVS872XT/dataset/ImageNet/train`；同一64枚の2viewをCPUで生成し両runに配布。ラベルはSSL lossに使わない |
| Model | ResNet50（random init）＋MLP `2048→2048→128`。Query・Key・BN・初期QueueをMeta由来のcanonical stateから共通にロード |
| 画像/分散 | 224×224、2GPU、world_size2、NCCL、local batch32×2＝global64。DDP `broadcast_buffers=true`、plain BN＋Meta Shuffle-BN、全rankのkeyをrank順enqueue、SyncBNなし |
| MoCo | `K=65536`、EMA `m=0.999`、`T=0.2`、Queue初期化はMeta公式 |
| Optimizer | SGD初期LR0.0075、momentum0.9、weight decay 1e-4、Nesterovなし、epoch0で公式Cosine scheduler準拠、optimizer momentum bufferはfresh |
| Precision | FP32、AMPなし、TF32 off、`cudnn_deterministic=true`、`cudnn_benchmark=false`、seed0 |
| Steps | 1 stepのみ。全rank同じfixture SHA、step1 full trace、`resume=null`、checkpoint stepsなし |
| 演算方式 | Mは無変更のMeta公式、S1は`moco.info_nce_impl=meta_einsum`かつ`diagnostic.info_nce_impl=null`（`self`既定は勝手に変更しない） |
| Comet | 初回比較では`logging.comet_enabled=false`を推奨。全reportはローカル保存し、外部送信なく検証 |
| GPU予算 | RTX A4500 ×2（freeは変化する）。`K`／localB／image／precisionをOOM回避のため自動変更しない。trace込みのpeak memoryを測定する |

**注意：** A2は通常学習`bench`ではなく専用`run_kind=parity`。このため毎phase trace/cloneを含み、A2のVRAMとstep_timeは純粋なtraining負荷ではない。軽量監視の速度評価は別A5とする。

## 5. 事前Gate（正式run前）

1. `git rev-parse HEAD`と`git status --short`を自作・Meta両方で記録。コミットやdirtyが上記契約と異なれば停止して再審議。importは`scripts/parity/moco_parity_state.py:import_reference`の外部shimで固定cloneに解決し、patchやコピーはしない。
2. ImageFolder train索引1000 class/1,281,167に関するユーザーEvidenceを再利用可。**fixtureが実際に選ぶ64枚のdecode・transformが成功し、manifestにパス/index/seed/hashが残ること**をfixture生成時に検査。ファイルを移動・リネーム・download・再編成しない。
3. `CUDA_VISIBLE_DEVICES=0,1`で2GPU認識とCUDA初期化を再チェック。NCCLの2rank通信は専用短時間check（GPU実行承認時）または正式run前の短時間health試験で検証。単なるCUDA discoveryをNCCL PASSとは扱わない。
4. 現在のGPU free、プロセス使用中でないこと、VRAMの見込み・出力先空き容量を記録。A2の`local32/K65536/224px`をGPU上で実際にfitするかは未実測。A2初回でOOMすれば**BLOCKED（resource feasibility）**としてそのまま報告し、条件変更は別承認。MetaとSelf片方だけ条件変更禁止。
5. `fixture.pt`、run、comparisonはいずれもfresh directoryを使い、既存成果物上書きなし。`fixture_manifest.json`の初期state・input SHA-256と両runの`fixture_sha256`を検証する。

## 6. 閾値決定のGate — **formal判定のblocking**

- **候補のみ**：A1で使用した `rtol=1e-5, atol=1e-6`。A2では**未固定**。暫定configの`verification.threshold_status=provisional`では、formal PASS/FAILは出せない。
- 先に同一fixtureでMeta公式**2つの独立fresh rerun（M0、M0r）**を実行する案を推奨する。両方で環境/backend・full step・入力hashを固定し、`--pair-kind rerun`でbitwiseとfloat差（gradient、post-SGD、EMA、BN、Queue等）を調査する。
- M0/M0rの数値差、A1既往Evidence、F32の比較スケールを示し、**A2の`rtol/atol`と離散状態exact条件を判定用runの前にユーザーが明示承認**する。自作S1に合わせた事後変更は禁止。判定runはfreshディレクトリでM1、S1を**新たに実行**し、両方の`resolved_config.yaml`に同じfixed値を保存。
- 対象tensorごとに固定閾値を設ける必要が判明した場合は、既存comparatorが全tensor共通閾値のみをサポートしている点とその影響を再評価し、**コード変更なら別spec/別承認**。未承認の新comparison規則を挿入しない。
- provisional診断runとfixed判定runの結果を同じ正式runと見なさない。失敗後に閾値を緩めたり、optional traceを消してPASSへ誘導したりしない。

### Decision Gate D1（未決・blocking）

正式runに進む前に **(a)** A2用の具体的なfixed `rtol/atol`、**(b)** Meta rerun実施数とreproducibility要求、**(c)** 診断/正式run実行許可が必要。推奨は「M0×2で基準を調査→数値基準を事前固定→M1/S1 fixed正式比較」。現時点では基準値を推測で決めない。

## 7. Runbook（**手順案**。このspecを保存しても実行許可にならない）

次のshellは既存GitHubコードに一致する具体例であるが、`A2_RTL`/`A2_ATL`は未決なので正式runは実行不可。作業ディレクトリは`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`、有効な`.venv`、自作commitは上記のものをpinする。

**7.1 CPU fixture build（A2 spec承認後、別途実行を指示された時）**

```bash
# fresh出力先は毎回一意にする。GPU計算は行わないが画像decode/初期stateの作成・I/Oはある。
FIX_DIR="$PWD/log/moco_imagenet_parity/fixtures/a2_imagenet_$(date +%Y%m%d_%H%M%S)"
python -m scripts.parity.build_moco_parity_fixture \
  fixture.kind=imagenet fixture.imagenet_steps=1 \
  moco.queue_size=65536 training.local_batch_size=32 data.image_size=224 \
  --output-dir "$FIX_DIR"
# 生成物: "$FIX_DIR/fixture.pt", "$FIX_DIR/fixture_manifest.json"
```

**7.2 GPU診断（A2 preflight・M0 rerunが承認されてからのみ）**

```bash
# 診断で使う各runディレクトリはHydraがfreshな時刻付きpathを生成する。
# M0、M0rは同一fixture・同一config・別processで別々に実行する。
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity \
  implementation=meta_reference run_kind=parity gate=A2 \
  fixture.path="$FIX_DIR/fixture.pt" training.max_steps=1 'trace.full_steps=[1]' \
  logging.comet_enabled=false verification.threshold_status=provisional
# fresh processで同じコマンドを再実行してM0rを生成
# --pair-kind rerun --reference <M0_DIR> --candidate <M0r_DIR> でCPU診断report
```

**7.3 正式GPU比較（A2閾値を文書で固定し、別GPU許可後のみ）**

```bash
# 例: A2_RTL, A2_ATL はspec承認により事前に決定された値のみ指定する。
# M1/S1それぞれfresh directory、fixtureは同一のfixture.pt。
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity \
  implementation=meta_reference run_kind=parity gate=A2 \
  fixture.path="$FIX_DIR/fixture.pt" training.max_steps=1 'trace.full_steps=[1]' \
  logging.comet_enabled=false verification.threshold_status=fixed \
  verification.rtol="$A2_RTL" verification.atol="$A2_ATL"

CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity \
  implementation=self run_kind=parity gate=A2 \
  moco.info_nce_impl=meta_einsum diagnostic.info_nce_impl=null \
  fixture.path="$FIX_DIR/fixture.pt" training.max_steps=1 'trace.full_steps=[1]' \
  logging.comet_enabled=false verification.threshold_status=fixed \
  verification.rtol="$A2_RTL" verification.atol="$A2_ATL"

python -m scripts.parity.compare_moco_parity \
  --pair-kind formal_implementation --gate A2 \
  --reference "$M1_RUN_DIR" --candidate "$S1_RUN_DIR" \
  --threshold-status fixed --rtol "$A2_RTL" --atol "$A2_ATL" \
  --output-dir "$COMPARE_DIR"
```

`FIX_DIR`、`M1_RUN_DIR`、`S1_RUN_DIR`、`COMPARE_DIR`は別々の実在pathを明示して使う。`M1_RUN_DIR`等の値を推定・再使用しない。実装時はHydraの出力を記録し、異なるconfig identity／fixture SHAなら正式比較へ進まない。出力先衝突なら停止。

## 8. 比較する値・成功条件

`formal_implementation`を使用し、Meta M1対Self S1の**同一step1、同一rank、同一phase**を比較する。

| 検証層 | 判定 |
|---|---|
| 入力/前処理 | `fixture_sha256`、input 2-view SHA、サンプルrank/順、dtype/shape、Shuffle permutation/seedの離散状態はexact |
| Forward | q/k features、MLP projection、positive/negative logits、CE loss、old Queue使用順を判定 |
| Backward | Query backbone/MLP各parameter grad、DDP allreduce後の値・gradient normを比較。Keyにgradなし |
| SGD | step後Query parametersとSGD momentum、LR、weight decay、Optimizer対象・順序 |
| EMA/Queue | step内EMAの時点、Key parameters、Queue enqueueとptr、rank順、Queue buffer |
| BN/DDP | Query/Key BN running mean/var/counterを対応rank・時点で比較。**異なるrank同士のBN bufferが等しいことは要求しない** |
| within-run | run status completed、rank reports 2/2、`aggregate_checks`の必須項目すべてPASS、NaN/Infなし |
| formal metadata | v3 report、`info_nce.consistent=true`、M `meta_reference`、S1 `meta_einsum`、commit/dirtyと同一fixture/threshold/steps整合 |

- Floatは **`abs(S1-M1) <= atol + rtol * abs(M1)`** を事前固定基準で判定。整数・ハッシュ・shape/dtype・permutation等はexact。bitwise一致率は別指標とし、要求閾値と混同しない。
- `PASS`：公式と自作が両方completed、within-run必須PASS、formal preconditionに不整合なし、**すべてのjudged値がfixed閾値内**、exact mismatch 0。
- `FAIL`：有効な比較で閾値を超える数値差、厳密状態の不一致、必須within-run failure。最初のstep/rank/phase/tensor、max/mean absolute error、勾配差・BN/DDP等を報告。
- `BLOCKED`：実機VRAM不足／GPU・NCCL不可／正式閾値未固定／未承認の条件変更などで有効なrunが成立しない。比較器の`BLOCKED`と実験実施不能の理由を区別。
- `INVALID`：dirty、異なるinput/fixture、mode不一致、trace欠損、NaN/Inf、未承認変種、旧schemaなどで正式比較契約自体が無効。
- `NOT_RUN`：未実行。診断比較を正式判定へ昇格させない。

## 9. Artifacts・保存場所・再現性

各独立runのローカルartifact：`resolved_config.yaml`、`run_report.json`（v3）、`provenance.json`、`fixture_manifest.json`、`resource_benchmark.json`、`verification_summary.md`、`trace/rank*/step_000001.pt`。Comparator：`parity_report.json`（v3）と`verification_summary.md`、必要なfirst mismatch情報。

記録すべき項目：両Git commit/dirty、依存torch/CUDA/driver/GPU、M import module実体とhash、ImageNet split/サンプルindex/2-view hash、canonical initial state SHA、各rank seed/permutation、actual config、FP32 backend flags、GPU peak allocated/reserved、各run経過時間、M rerun差、事前に固定した閾値、正式comparison status。

- fixture.pt（初期stateと画像tensor）、巨大trace、checkpoint、GPU詳細ログ、ImageNet実体をGitHub Research Workspaceへpushしない。NASは読み取り専用。
- 正式run後、結果要約と小さな数値表を`haruto2919/research-workspace@main`の対象project `experiments/`へ保存。実行不能でも停止理由・部分artifact path・first failureを記録する。
- Research WorkspaceへのGitHub書込はサーバのclone更新を意味しない。

## 10. Compatibility / 対象外

- **既存コード変更を予定しない**（fixture generator、parity runner、comparator、Hydra configの明示overrideだけで実行する）。Meta固定cloneのsourceにはpatchしない。
- `StandaloneMoCo`、CIFAR、A1診断/16step、Video Streaming MoCo、Frozen ViT、LoRA、Sequential Loader、Linear Probe、10epochには手を加えない。
- 旧A1 v2 artifactは新judge16で直接再判定できない既知制約。A2の新v3正式pairで扱う問題ではないため、旧artifactを書き換えない。
- 長時間Queue one cycle（A3）、Checkpoint exact GPU resume（A4）、monitor OFF/light speed benchmark（A5）は**別実験・別GPU承認**。

## 11. Ambiguity Gate / Gateと引き継ぎ

### Blocking（承認・正式実行前）

1. **D1 判定用閾値**：A2のfixed rtol/atolが未固定。公式M0/M0rの再実行診断後に決め、M1/S1正式run前に明示承認が必要。
2. **D2 実験許可**：fixture生成（CPU・読み取り＋artifact作成）、短時間2GPUのMeta rerun診断、正式な2GPU M1/S1判定runの権限は別々に確認。ここでの「次に進んで」はspec作成着手までの指示であり、未承認GPU runを起動したという主張をしない。
3. **D3 資源と実行能力**：2rank NCCL、import provenance、224px/local32/K65536のpeak VRAMをrun時確認。OOMなら新条件で別承認、不能理由の保存を優先する。

### Non-blocking

- ヘルパー名、ログ表示順、run-ID表記、開始時刻は既存runnerのstyleへ合わせる。実験の条件・評価・合否を変えない範囲のみ調整可能。

### 順序／ユーザー判断

1. このdraftの実験条件・診断→正式の2段階設計をレビューして承認（`approved`へ昇格する場合は残るblockingを解消するかpreflight subgateを明示分離）。
2. 承認後、**CPU fixture生成**しマニフェスト検査。必要に応じた短いNCCL/GPU preflight（別許可）。
3. 許可されたM0/M0r再演で誤差診断→fixed閾値の提案・ユーザー承認。
4. 別途GPU許可を受け、M1/S1正式run→offline formal comparison→`experiments/`へEvidence保存。
5. A2 PASSならA5 GPU monitor OFF/lightのresource feasibility specへ。A2 FAILならfirst mismatchから原因切り分けへ。10epoch・動画・LoRAの本実験へ勝手に進まない。

## 12. Research-spec handoff（現在はdraft）

- **Spec**：本ファイル（ユーザー未承認、status draft）。
- **対象GitHub code**：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、`dc502e416aa26ed4e0791767fa5a72f7843e399d`。
- **変更対象**：原則コード変更なし。既存CLI＋fresh fixture＋run/compare。
- **成功条件**：fixed事前閾値、fixture/環境/モードが同一で正式`formal_implementation` PASS。
- **現状確認**：CPU Engineering PASS、ImageFolder sample decode PASS、CUDA discovery PASS。GPU A2/NCCL/VRAM正式検証NOT_RUN。
- **実行権限**：本draft保存ではGPU実験・研究コード変更・10epochを許可しない。
