---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: experiment-evidence
status: completed
gate: A5-pilot
judgement: PASS
evidence_level: "real-imagenet-two-gpu-fp32-normal-bench-twenty-step-pilot"
source: "user-provided-terminal-output including stage_summary.md and run_report.json printed in terminal; SSH artifacts not independently accessed"
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
a4_preflight: "plan only; previous A5-pilot PASS; issues=[]; NOT_RUN"
---

# A5-pilot：ImageNet実画像・2GPU・通常MoCo bench 20step GPU Evidence

> **A5-pilot = PASS（ユーザー提供の研究サーバterminal出力と統合ランチャーのGate判定）**。今回のrunは**実GPU**で実行され、`A5-pilot PASS: every step completed and every check holds`が表示された。ユーザーが追加で`cat stage_summary.md`と`run_report.json`内容を端末に表示したログを共有したため、両rankのVRAM・step時間・throughput・DataLoader待ち時間は**ユーザー提示のartifact内容として確認済み**。ただしChatGPT自身がSSH上の原本を取得しハッシュ照合したわけではない。A4、A3、A5-pairはまだNOT_RUN。

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
- Run: `A5-pilot/runs/S1_off20/`（`run_report.json`と`stage_summary.md`はユーザーが端末上で内容を表示済み。他の`resource_benchmark.json`, `resolved_config.yaml`, `provenance.json`, `verification_summary.md`等の個別存在・SHAはChatGPTのSSHアクセスでは独立確認していない）
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
- 当初のrun完了時には総括表示のみだったが、**その後ユーザーが`stage_summary.md`と`run_report.json`の内容を端末に表示して共有した**ため、`completed_steps=20`、rank別資源、window、InfoNCE、Failuresを確認できた。monitor OFFなのでLoss実数は今回のbenchmark reportとしてユーザー提示ログからは未確認。
- 20step pilotのPASSを、A5-pairの100step×2本やA4 Resume、A3 Queue一巡のPASSと混同しない。
- 現在のrepo/devとSSHローカル作業ツリーが貼付け時点でcleanで一致したが、GitHubとSSHの自動同期を意味しない。

## 4. 追加のrun artifact表示ログから確認できた詳細Evidence

2026-10-20のユーザー追送ログは、研究サーバ上で`cat "$ROOT/A5-pilot/stage_summary.md"`を実行した出力と、`run_report.json`をPythonで読み出した出力。画面上のシェル貼り付けが一部崩れていたが、stage/runの全主要値が両方表示されている。**実験計算・測定器の再実行ではなく既存成果物の表示**である。

| 指標 | GPU rank 0 | GPU rank 1 | 判定・意味 |
|---|---:|---:|---|
| run status / steps | completed / 20/20 | completed / 20/20 | **PASS** |
| GPU | NVIDIA RTX A4500 | NVIDIA RTX A4500 | 2GPU CUDA |
| device | cuda:0 | cuda:1 | rank対応 |
| total_memory_mib（報告値） | 20095.875 | 20095.875 | Python/CUDAの容量表示 |
| max_memory_allocated_mib | 3585.6455078125 | 3689.2919921875 | rank別PyTorch peak allocated |
| max_memory_reserved_mib | 3788.0 | 3886.0 | rank別PyTorch peak reserved（容量の約18.85%・19.34%） |
| measurement windows | 1 | 1 | step 11〜20のみ測定、warmup1〜10 |
| measured wall seconds | 7.904781910001475 | 7.904838533999282 | 10 measured steps |
| mean step time (s) | 0.7904781910001475 | 0.7904838533999282 | 10step windowの平均。traceなしbench |
| data wait mean (s/step) | 0.5992085869998846 | 0.5857216767006321 | DataLoader `next` 待ちの観測時間 |
| data wait / step time（参考比） | 約75.80% | 約74.10% | 待ち時間が大きい。**純GPU実行時間への厳密分解ではない** |
| global throughput (images/s) | 80.96365052023056 | 80.96307056081079 | **2GPU合算の計算値なので両rankを足さない** |

- User-shared run report: `Status: completed`、`Steps: 20 / 20`、`Failures: []`。`info_nce = {'configured_normal': 'meta_einsum', 'configured_diagnostic': None, 'selector_source': 'moco', 'effective': 'meta_einsum', 'legacy_compat': False, 'per_rank': ['meta_einsum', 'meta_einsum'], 'consistent': True}`。
- `measurement_windows[0]`は**rank0: first_step 11,last_step 20,steps10,elapsed 7.904781910001475, data_wait 5.992085869998846**、**rank1: first_step 11,last_step 20,steps10,elapsed 7.904838533999282,data_wait 5.857216767006321**。
- `step_time_sec_after_warmup.mean`、`median`、`p90`が全て同じなのは**window数が1のため**。各個別stepのmedian/p90を測ったことを意味しない。
- **解釈（推論）**：GPU peak reservedは約3.7〜3.8GiBでこの20stepは余裕を保って終了。一方、データ待ち比率が約74〜76%と大きく、DataLoader／NAS I/O／CPU augmentation／prefetchのいずれかが通常benchの処理速度を制約している可能性がある。1windowだけでは原因やsteady-state throughputを確定できない。現時点でworker数・batch・Queue・precisionを変更しない。

### A4のplanもユーザー端末で確認済み

同一rootの`A4 --root "$ROOT"`を**planのみ**で実行：

- `previous: {stage: "A5-pilot", status: "PASS", reason: "every step completed and every check holds", commit: "914e736c2699e2e129211a888d1f19013dd1dd1a"}`
- `issues: []`、研究`dev@914e736c...`とMeta固定clone`a419ef2...`ともclean、`disk_free_gib: 2761.7`（計画時点）。
- 予定4step：synthetic fixture生成、S1 uninterrupted 3step（step2 checkpoint, full trace step3）、S1 resumed（step2 native checkpointからstep3）、`formal_resume --threshold-status fixed --rtol 1e-5 --atol 1e-6`比較。
- **A4はまだNOT_RUN**。plan成功は正式比較のPASSではない。実GPU実行前にGPU2枚の空き状況を再確認し、同一root・同一commitを維持する。

## 5. 追送ログの出力確認コマンド（既に実行済み）

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

上記の測定値は同日のユーザー提供terminalログから**本Evidenceへ追記済み**。SSH上の原本hashやworkerの個別logをChatGPTが独立検証したわけではない。

## 6. Gateの意味・次段階

- 今回初めて通常`bench`経路のreal ImageNetで自作MoCo `meta_einsum` の2GPU 20stepが統合ランチャーの検査を通った。2GPU通常学習経路の初回 feasibility 証拠となる。
- **A4（次のStage）**：同じcampaign rootとself commit、2GPUでsynthetic fixtureを新規生成し、3step uninterrupted、step2 native+rank0/1 sidecar checkpoint、fresh resume→step3、`formal_resume` fixed threshold の正式Gate。実行前の**planで`previous A5-pilot: PASS`、`issues=[]`が確認済み**。**A4 GPU実行はまだしていない**。
- **A3**：Meta公式と自作各1025step・Queue一巡の別実験。A4 PASSが必要。時間/ストレージ見積りなしに開始しない。
- **A5-pair**：monitor OFF100/Light100を独立freshで実行する次Gate。A3 PASSが必要。overhead measuredと良い表現の証明は別。
- このEvidenceはユーザー貼付けterminalを研究文脈としてGitHub mainへ保存したものであり、SSH上の学習run/checkpointの新規作成や同期はChatGPTから実施していない。
