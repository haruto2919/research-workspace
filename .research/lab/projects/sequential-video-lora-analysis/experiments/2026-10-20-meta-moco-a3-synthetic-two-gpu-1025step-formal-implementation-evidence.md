---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: experiment-evidence
status: completed
gate: A3
judgement: PASS
evidence_level: "synthetic-two-gpu-1025step-meta-vs-self-formal-implementation; launcher-aggregate-verdict-only"
source: "user-provided-terminal-output ending in A3 PASS; stage_summary/parity_report raw summary not yet provided; SSH artifacts not independently read"
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
bitwise_entries: "PENDING_RAW_SUMMARY"
gpu_gates: "A5-pilot PASS; A4 PASS; A3 PASS; A5-pair NOT_RUN"
---

# A3実GPU Formal Gate：Meta公式 MoCo v2 対 自作 MoCo の1025step Queue一巡検証

> **A3ランチャー総合判定 = PASS（ユーザー提供端末ログ）**。採用済みの固定A3計画によりMeta公式Mと自作S1のsynthetic 1025step×2runを実行する構成で、最後にランチャーが`A3 PASS: every step completed and every check holds`と出力した。ただし、個々の比較エントリ数、bitwise一致率、Queue pointerの実際の観測値、wall timeは**今回の端末断片には含まれない**。これらを推測値で補わず、別途raw summaryの提示待ちとする。SSH上の原本にはChatGPTからアクセスしていない。

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

**まだ直接提示されていない**：`counts.judged`、`counts.bitwise_equal`、`counts.exact_failures`、`counts.close_failures`、`formal.contract_valid`、`steps_compared`の長さ、`first_failure`、`first_difference`、各run実際の`completed_steps`、Queue pointer step1024/1025の実測詳細、`stage_summary.steps[*].elapsed_sec`、A3全体のディスク実容量。

比較器の`PASS`は事前固定誤差以内を意味するが、**bitwise一致100%を自動的には意味しない**。`bitwise`一致率は別の観測で確認する。

**主張できないこと**：ImageNet実画像で1025step連続学習の数値一致、任意のseed/GPU条件・長期収束・表現の有用性、ViT/LoRA/動画のオンライン学習結果。

## 5. 次に必要な最小の読み出し

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

数値を受け取り次第、この**同じEvidenceに追記**して集計を確定する。A3の結果は、A5-pair（OFF/Light各100step）に進むための前段Gateであるが、**A5-pairはまだNOT_RUN**。実験を勝手に開始せず、まず次のplanを確認する。
