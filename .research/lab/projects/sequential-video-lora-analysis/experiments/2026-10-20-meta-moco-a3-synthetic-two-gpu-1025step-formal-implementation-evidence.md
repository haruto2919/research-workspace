---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: experiment-evidence
status: completed
gate: A3
judgement: PASS
evidence_level: "synthetic-two-gpu-1025step-meta-vs-self-formal-implementation; 58790-of-58790-bitwise-parity"
source: "user-provided-terminal-output of A3 stage_summary.json and compare/parity_report.json aggregate, plus earlier launcher PASS; SSH artifacts not independently accessed"
experiment_spec: ../specs/2026-10-20-meta-moco-a3-a4-a5-integrated-execution-spec.md
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: 914e736c2699e2e129211a888d1f19013dd1dd1a
meta_reference_commit: a419ef282fc9885015674e77f58c59447206a27f
run_label: a345_20261020_114641
stage: A3
comparison_pair: formal_implementation
threshold_status: fixed
rtol: 1.0e-5
atol: 1.0e-6
formal_judgement_observed_via_launcher: PASS
comparison_entries: 58790
bitwise_equal_entries: 58790
exact_failures: 0
close_failures: 0
steps_compared: 1025
full_tensor_steps: [1, 1024, 1025]
stage_disk_du_human: "9.9G"
gpu_gates: "A5-pilot PASS; A4 PASS; A3 PASS; A5-pair NOT_RUN"
---

# A3実GPU Formal Gate：Meta公式 MoCo v2 対 自作 MoCo の1025step Queue一巡検証

> **A3の総合Gate・formal_implementationともPASS（ユーザー提供のA3 stage summaryと比較器report集計）**。Meta公式Mと自作S1のsynthetic 1025step×2runを実行し、**58,790 / 58,790比較エントリがbitwise一致（100%）**、`exact_failures=0`、`close_failures=0`、`contract_valid=True`、step1〜1025の全step比較を確認した。SSH上のtensor原本、Queue pointer個別値、artifact SHAまではChatGPTから独立確認していない。

## 1. Source・Authority

- 実験契約は[承認済み統合spec](../specs/2026-10-20-meta-moco-a3-a4-a5-integrated-execution-spec.md)。前段は[A5-pilot実GPU20step PASS](2026-10-20-meta-moco-a5-pilot-real-imagenet-two-gpu-20step-evidence.md)、[A4 synthetic checkpoint/resume formal PASS](2026-10-20-meta-moco-a4-synthetic-two-gpu-step2-resume-formal-parity-evidence.md)。
- 直前のplanログでは、self code `dev@914e736c2699e2e129211a888d1f19013dd1dd1a` clean、Meta clone `a419ef282fc9885015674e77f58c59447206a27f` clean、前段A4 PASS、`issues=[]`。
- 今回のユーザー端末ログは`[A3] S1`起動コマンド末尾、`[A3] compare`起動コマンド（fixed rtol/atol）と次の最終判定を含む（先頭のfixture/M実行ログ全文は今回再掲されていない）。

```text
[A3] compare: ... python -m scripts.parity.compare_moco_parity
  --pair-kind formal_implementation --gate A3
  --reference .../A3/runs/M --candidate .../A3/runs/S1
  --threshold-status fixed --rtol 1e-05 --atol 1e-06
  --output-dir .../A3/compare

A3 PASS: every step completed and every check holds
Saved .../log/moco_imagenet_parity/gates/a345_20261020_114641/A3
```

- `scripts/parity/run_moco_gates.py`（研究コード`dev@914e736c`）は、A3のfixture→M→S1→compareを逐次実行し、失敗時には残りを実行せず、各step終了コード・run artifact整合・Queue一巡等のpostcheck・固定しきい値の比較結果を検査する構造。最終`PASS`は、ランチャーが所定チェックを満たしたという**高水準の観測**として採用する。
- 生の`stage_summary.json`、`compare/parity_report.json`、各rankのhistory/trace/checkpoint等の原本を、ChatGPT自身はSSHから取得/ハッシュ検証していない。

## 2. 固定実験条件（実行計画より）

| Axis | Fixed |
|---|---|
| Dataset / fixture | synthetic固定入力。実ImageNet連続学習ではない |
| Encoder | ResNet50＋MLP128 |
| Distributed | NCCL 2GPU/2rank、local batch32、global64 |
| Queue | K65536、1024stepで初回一巡、step1025で置換後の状態を検証 |
| Model M | 改変なし固定Meta公式clone、`implementation=meta_reference` |
| Model S1 | 自作MoCo、`implementation=self moco.info_nce_impl=meta_einsum diagnostic.info_nce_impl=null` |
| Steps | 1025 × 2run = 2050学習step |
| Traces | 全step small trace、full trace at steps [1,1024,1025] |
| Fixed numerical threshold | float `rtol=1e-5,atol=1e-6`、discrete/exactは別判定 |
| Comparator | `formal_implementation`、`gate=A3` |
| Comet | OFF |

## 3. SSH campaign path

`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/gates/a345_20261020_114641`

- A3 stage summary: `A3/stage_summary.json` / `A3/stage_summary.md`。
- Meta run: `A3/runs/M`。
- Self run: `A3/runs/S1`。
- Comparator: `A3/compare/parity_report.json`。
- Execution logs: `A3/logs/M.log`, `A3/logs/S1.log`, `A3/logs/compare.log`。

## 4. 確認済み・未確認

**確認済み**：ユーザーが出力したランチャー最終行は`A3 PASS: every step completed and every check holds`。前回のplanは固定閾値/両commit clean/前段A4 PASS/問題なし。現行ランチャーの`check_compare`は比較器`status=PASS`を要求するので、ランチャーが`PASS`を返したことは、正式比較をランチャーが受理した強い間接証拠。

**追加ユーザーログで確認できた**：`counts.judged=58790`、`counts.bitwise_equal=58790`、`exact_failures=close_failures=0`、`formal.contract_valid=True`、`steps_compared`はstep1〜1025の1025件、`full_tensor_steps=[1,1024,1025]`、`first_failure/first_difference=None`、stageの4工程が完了exit0、所要時間、`du -sh`の実容量。

**引き続き未提示**：各runの`completed_steps`数値を直接読み出した表示、rank別のQueue pointer step1024/1025実数、artifact SHAとraw tensor原本の独立取得。ランチャーがPASSを判定した範囲と別に区別する。

比較器の`PASS`一般は事前固定誤差以内を意味するだけであるが、**今回は追加提出された集計値により58,790/58,790のbitwise一致を直接確認した**。58,790は比較エントリ数であり、torch Tensor全要素の個数ではない。

**主張できないこと**：ImageNet実画像で1025step連続学習の数値一致、任意のseed/GPU条件・長期収束・表現の有用性、ViT/LoRA/動画のオンライン学習結果。

## 5. 集計確認に使用した読み出しコマンド（実行済み）

SSHでA3 stage summaryとcomparison reportの集計のみを、巨大な`comparisons`配列を表示せず読み出す。

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
export ROOT="$PWD/log/moco_imagenet_parity/gates/a345_20261020_114641"
python - <<'PY'
import json,os
from pathlib import Path
a=Path(os.environ["ROOT"])/"A3"
stage=json.loads((a/"stage_summary.json").read_text())
r=json.loads((a/"compare/parity_report.json").read_text())
print("STAGE",stage.get("status"),stage.get("reason"),"issues",stage.get("issues"))
for k,v in (stage.get("steps") or {}).items():
    print("STEP",k,v.get("state"),v.get("exit_code"),"seconds",v.get("elapsed_sec"))
for k in ("status","reason","pair_kind","counts","tolerance","steps_compared","full_tensor_steps","first_failure","first_difference"):
    v=r.get(k)
    if k=="steps_compared":
        print(k,"len",len(v or []),"first",v[0] if v else None,"last",v[-1] if v else None)
    else:
        print(k,v)
print("FORMAL",r.get("formal"))
PY
du -sh "$ROOT/A3"
```

追加ログを受領し、下記の数値を本Evidenceに反映済み。A3はA5-pair（OFF/Light各100step）の前段Gateとして正式にPASS。ただし**A5-pairはまだNOT_RUN**で、次は同じcampaign rootでplan-onlyを確認する。GPU実験は未起動。

## 6. 追補：正式比較の全step bitwise一致・実測所要時間・ディスク使用量（2026-10-20）

ユーザーがSSH研究サーバ上で既存`A3/stage_summary.json`および`A3/compare/parity_report.json`を読み出し、比較の集計値と時間を表示した。これは**ユーザー提出のローカル実験artifact出力**であり、ChatGPTがSSHファイルを直接ダウンロード・ハッシュ照合したわけではない。

### 公式M対自作S1：formal_implementationの確定判定

```text
=== A3 STAGE ===
Status: PASS
Reason: every step completed and every check holds
Issues: []

=== FORMAL COMPARISON ===
status: PASS
reason: all compared tensors within the fixed tolerance; within-run checks PASS
pair_kind: formal_implementation
rtol: 1e-05
atol: 1e-06
threshold_status: fixed
run_consistency_issues: []
counts: {judged: 58790, exact_failures: 0, close_failures: 0, bitwise_equal: 58790}
full_tensor_steps: [1, 1024, 1025]
first_failure: None
first_difference: None
steps_compared: 1025 (first 1, last 1025)
contract_valid: True
formal.bitwise: {judged: 58790, bitwise_equal: 58790, fraction: 1.0,
                 first_difference: 'none (all compared values bitwise equal)'}
```

重要な解釈：**Queue一巡点の1024stepと直後1025stepを含む1025stepの全比較対象がbitwise一致**。比較器が追跡したloss/logits/Query-Key/Queue/optimizer等の対象でbitwise差なし（測定範囲は既存harnessのtrace契約に限定）。これは**Meta公式との指定synthetic fixture上の長期数値一致**の証拠となるが、実ImageNetでの1025step学習、表現品質、ViT/LoRA/オンライン動画統合を証明しない。Queue pointerの具体的観測数値は本コンソールにないので未確認扱い。

### 4工程の実行時間（launcher elapsed_sec）

| 工程 | state / exit | elapsed_sec |
|---|---|---:|
| fixture | completed / 0 | 4.723967037003604 |
| Meta公式 M（1025step） | completed / 0 | 719.4326336240047 |
| 自作 S1（1025step） | completed / 0 | 732.3776578129982 |
| formal compare | completed / 0 | 76.99084420599684 |
| **合計** | **4工程success** | **1533.5251026800032秒、約25分33.5秒** |

2つのrunの観測値は近いが、同一マシンでの1回の順次実行に過ぎず、速度の優位性や統計的同等性は未検証。GPU演算timerでなく各subprocessのwall time。fixture・比較込みで約25分34秒。

### Disk

```text
du -sh ".../gates/a345_20261020_114641/A3"
9.9G
```

事前trace外挿9.53GiBとの比較はファイル以外のartifactや丸めを含むため単純一致とは言わない。実測は表示上**9.9G**（duの人間可読表示）。出力上書き禁止、M/S1のtraceとreportを保持する。

## 7. 後続Gate

**A5-pair: NOT_RUN**。A3の前段PASSが確定したので、同じ`a345_20261020_114641` rootで`python -m scripts.parity.run_moco_gates A5-pair --root "$ROOT"`のplan-onlyを実行し、`previous=A3 PASS`, `issues=[]`, commit clean、空きGPU/ディスクを確認する。その後のOFF100/Light100実行は別操作。A5-pairはmonitorの処理・資源overhead測定であり、MoCoの特徴品質評価ではない。
