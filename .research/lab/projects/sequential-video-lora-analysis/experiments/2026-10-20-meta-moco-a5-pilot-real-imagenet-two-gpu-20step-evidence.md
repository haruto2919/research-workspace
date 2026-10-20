---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: experiment-evidence
status: completed
gate: A5-pilot
judgement: PASS
evidence_level: "real-imagenet-two-gpu-fp32-normal-bench-twenty-step-pilot"
source: "user-provided-terminal-output; runner stage summary not independently opened"
experiment_spec: ../specs/2026-10-20-meta-moco-a3-a4-a5-integrated-execution-spec.md
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_commit: 914e736c2699e2e129211a888d1f19013dd1dd1a
meta_reference_commit: a419ef282fc9885015674e77f58c59447206a27f
run_label: a345_20261020_114641
stage: A5-pilot
run_name: S1_off20
monitor: "off"
steps: 20
gpu_gates_remaining: "A4, A3, A5-pair"
---

# A5-pilot：ImageNet実画像・2GPU・通常MoCo bench 20step GPU Evidence

> **A5-pilot = PASS（ユーザー提供の研究サーバterminal出力と統合ランチャーのGate判定）**。今回のrunは**実GPU**で実行され、`A5-pilot PASS: every step completed and every check holds`が表示された。一方、ChatGPTはSSH上のraw `stage_summary`、`run_report`やGPU resourcesを直接読めておらず、詳細測定値の独立確認は未実施。A4、A3、A5-pairはまだNOT_RUN。

## 1. Preflight：ユーザーが貼付けた実測事実

- Workdir: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`。
- `git status --short` は空、`git rev-parse HEAD = 914e736c2699e2e129211a888d1f19013dd1dd1a`。計画が表示したself repositoryも同一commit、branch dev、dirty false。
- `nvidia-smi` 2026-10-20 11:46:18に、GPU0,1が両方NVIDIA RTX A4500、各20470 MiB capacity、15MiB使用、利用率0%、graphics Xorg（各4MiB）のみ。学習ジョブなしの観測は**実行前時点に限る**。
- `/mnt/NAS-TVS872XT/dataset/ImageNet/train` が存在（`ImageNet OK`）。ファイル・annotationの完全性確認を意味しない。
- `df -h`: `/mnt/HDD12TB-1` filesystem 11T、used7.6T、avail2.7T、74%; `/tmp`側3.6T、used390G、avail3.1T、12%（コマンド結果時点）。
- A5-pilot plan: `disk_free_gib=2761.7`、研究コード`914e736...` clean、固定Meta clone`a419ef282fc9885015674e77f58c59447206a27f` clean、`previous=null`、`issues=[]`。

## 2. 実行条件とlocal paths

Plan/executeが表示したコマンドの契約：

| Axis | Value |
|---|---|
| Run kind / gate | `run_kind=bench`、`gate=A5`、stage `A5-pilot` |
| Implementation | `implementation=self`、`moco.info_nce_impl=meta_einsum`、`diagnostic.info_nce_impl=null` |
| Dataset | ImageNet-1K train、`ImageFolder`＋実DataLoader、224px、`num_workers=8`/rank（現行Config/spec） |
| GPUs | `CUDA_VISIBLE_DEVICES=0,1`、NCCL 2rank、RTX A4500×2 |
| Model / MoCo | ResNet50+MLP128、K65536、momentum0.999、temperature0.2 |
| Optimizer | SGD LR0.0075、momentum0.9、weight_decay1e-4、cosine horizon10epochs、seed0 |
| Precision | FP32、AMP OFF、TF32 OFF（現行Config/spec） |
| Steps / batch | 20step、local32/global64（計算上20×64=1280サンプル提示、ユニーク画像数ではない） |
| Monitor | `monitor.enabled=false` |
| Benchmark | `bench.warmup_steps=10`、`bench.measurement_window_steps=10` |
| Logging | `logging.comet_enabled=false`、`OMP_NUM_THREADS=4` |

SSH campaign root（Research Workspaceでは実体未取得）：

`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/moco_imagenet_parity/gates/a345_20261020_114641`

- Stage: `A5-pilot/`
- Run: `A5-pilot/runs/S1_off20/`（`run_report.json`, `resource_benchmark.json`, `resolved_config.yaml`, `provenance.json`, `verification_summary.md`等が**正常runなら既存コードに従い保存される。個別ファイルの存在をChatGPTはまだ独立確認していない**）
- Launcher log: `A5-pilot/logs/S1_off20.log`
- Stage judgement: `A5-pilot/stage_summary.json`, `stage_summary.md`

## 3. 実行結果（貼付けられたconsole）

```text
[A5-pilot] S1_off20: ... python -m scripts.parity.moco_imagenet_parity
  implementation=self moco.info_nce_impl=meta_einsum diagnostic.info_nce_impl=null
  run_kind=bench gate=A5 bench.warmup_steps=10
  bench.measurement_window_steps=10 training.max_steps=20
  monitor.enabled=false logging.comet_enabled=false
  hydra.run.dir=.../a345_20261020_114641/A5-pilot/runs/S1_off20

A5-pilot PASS: every step completed and every check holds
Saved .../a345_20261020_114641/A5-pilot
```

- A5-pilot **launcher judgement: PASS**。GitHub実装の`run_moco_gates.py:420-473`は完了report/2rank/InfoNCE mode/commit/benchmark windows/resourcesなどを照合し、`execute()`が失敗step時に後続を停止する設計。今回はその評価が`PASS`を返した。
- 今回提示された内容は**ランチャーの総括表示**であり、`completed_steps=20`や全rankのpeak GPU MiB、step time、throughput、DataLoader wait、window数、Loss実数は**raw JSONを直接提示されたわけではない**。推測値を補わない。
- 20step pilotのPASSを、A5-pairの100step×2本やA4 Resume、A3 Queue一巡のPASSと混同しない。
- 現在のrepo/devとSSHローカル作業ツリーが貼付け時点でcleanで一致したが、GitHubとSSHの自動同期を意味しない。

## 4. 次にSSHで確認するraw outputs（read-only）

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
ROOT="$PWD/log/moco_imagenet_parity/gates/a345_20261020_114641"
cat "$ROOT/A5-pilot/stage_summary.md"
python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
base=Path(sys.argv[1])/"A5-pilot"
for label,p in [
    ("stage",base/"stage_summary.json"),
    ("run",base/"runs/S1_off20/run_report.json")
]:
    d=json.loads(p.read_text(encoding="utf-8"))
    print(label,"status",d.get("status"),"steps",d.get("completed_steps"),"reason",d.get("reason"))
    if label=="run":
        print("info_nce",d.get("info_nce"))
        print("failures",d.get("failures"))
        for rank,r in enumerate(d.get("resources") or []):
            print("rank",rank,"resources",r)
PY
```

この結果を受け取れば、実測VRAM/step時間/throughputを**同一Evidenceへ追記**する。現在は記録なし。

## 5. Gateの意味・次段階

- 今回初めて通常`bench`経路のreal ImageNetで自作MoCo `meta_einsum` の2GPU 20stepが統合ランチャーの検査を通った。2GPU通常学習経路の初回 feasibility 証拠となる。
- **A4（次のStage）**：同じcampaign rootとself commit、2GPUでsynthetic fixtureを新規生成し、3step uninterrupted、step2 native+rank0/1 sidecar checkpoint、fresh resume→step3、`formal_resume` fixed threshold の正式Gate。実行前`plan`で`previous stage A5-pilot is PASS`、`issues=[]`を確認。**A4 GPU実行はまだしていない**。
- **A3**：Meta公式と自作各1025step・Queue一巡の別実験。A4 PASSが必要。時間/ストレージ見積りなしに開始しない。
- **A5-pair**：monitor OFF100/Light100を独立freshで実行する次Gate。A3 PASSが必要。overhead measuredと良い表現の証明は別。
- このEvidenceはユーザー貼付けterminalを研究文脈としてGitHub mainへ保存したものであり、SSH上の学習run/checkpointの新規作成や同期はChatGPTから実施していない。
