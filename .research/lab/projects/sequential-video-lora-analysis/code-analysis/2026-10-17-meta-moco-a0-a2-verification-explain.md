---
project: sequential-video-lora-analysis
date: 2026-10-17
analysis_mode: explain
detail: full
level: junior
source_repository: https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_ref: dev
source_commit: c50ab79cbce1738a571daf6e018e31bb870cae24
source_dirty: unknown
targets:
  - conf/moco_imagenet_parity.yaml
  - scripts/parity/preflight_moco_imagenet.py
  - scripts/parity/build_moco_parity_fixture.py
  - scripts/parity/moco_imagenet_parity.py
  - scripts/parity/compare_moco_parity.py
related_spec: .research/lab/projects/sequential-video-lora-analysis/specs/2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md
---

# Code Deep Dive：Spec Aの①Git・②A0・③A1・④数値許容差・⑤A2検証方法

## 対象と根拠の区別
GitHubのdev（2026-10-17確認時点c50ab79）にあるMoCo ImageNet parity実装を**読み取り専用**で調査。研究サーバの未コミット差分・実行ログの実体は未確認。ユーザー報告のCPUテスト結果・NAS読み込み等は実機で再測定していない。Spec Aはdraftであり本書の作成はGPU実行許可ではない。

## ① Gitと実験記録
研究サーバの自作repoでgit branch --show-current、git rev-parse HEAD、git status --short --untracked-files=all、git diff HEAD --statを確認する。Meta cloneもgit rev-parse HEADで a419ef282fc9885015674e77f58c59447206a27fを期待し、git status --shortでdiffを記録。公式・自作のversion、dirty、canonical fixtureのSHA、保存されたrun provenanceを結び付ける。GitHubに未反映の変更を自動でpull/resetしない。

## ② A0 NCCL preflight
GPU2枚の明示的実行許可後に実行：
~~~bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.preflight_moco_imagenet --nccl-check --pytest
~~~
出力は log/moco_imagenet_parity/preflight/YYYY-MM-DD/HH-MM-SS-ffffff 以下の preflight.json / dataset_inventory.json / verification_summary.md / nccl_rank0.json / nccl_rank1.json。全必須sectionがPASS、各rankでall_reduce_ok・all_gather_okがtrue、broadcast一致ならA0=PASS。--pytestを省略すると回帰testsがNOT_RUNとなりA0 BLOCKEDになり得る。

## ③ A1：小Kの2GPU公式対自作
build_moco_parity_fixture.pyでsynthetic fixtureを出力先付きで作成（moco.queue_size=64、training.local_batch_size=4、data.image_size=64）。両runで同じ fixture.pt を使い、implementation=meta_reference と implementation=self を gate=A1、training.max_steps=16、trace.full_steps=[1,8,9,16]、logging.comet_enabled=false で実行し、各runの hydra.run.dir は別にする。globalB=8なのでQueue64は8stepで一巡。Meta adapterはCUDA専用なのでCPUではこの比較を実施できない。

両runのrun_report.json.statusがcompleted、rank別16 self-check PASS、compare_moco_parity.py（--reference <meta_run> --candidate <self_run> --threshold-status provisional）のparity_report.jsonでexact_failures=0、close_failures=0、first_failure=nullを目標にする。閾値provisionalなら正しくても比較全体はBLOCKED（diagnostic）であり、正式PASSを記録しない。

## ④ 数値許容差の検証
Meta公式runを同じfixture・configで別run directoryへ再演し、compare_moco_parity.py の --pair-kind rerun で診断する。A1小条件だけでなく、A2と同じ224px/K65536/localB32のMeta公式再演も確認する。meta-rerunとmeta-vs-selfの誤差・backend flags・fixture checksum・rank/shuffleの同一性を比較し、rtol=1e-5、atol=1e-6の候補を判定用runの**前に固定**する。--threshold-status fixedはソフトウェア上のラベルであり、ユーザー承認や科学的妥当性の代替ではない。後から実験結果をPASSにするために緩めない。

## ⑤ A2：本設定1step
build_moco_parity_fixture.pyでfixture.kind=synthetic（デフォルト224px, K65536, localB32）とfixture.kind=imagenet fixture.imagenet_steps=1のfixtureを別々に生成する。
両fixtureごとに meta_reference と self を同一fixtureで gate=A2、training.max_steps=1、logging.comet_enabled=false、別のhydra.run.dirとして実行する。事前に許容差が正式固定されている場合に限り、compare_moco_parity.pyへ --rtol 1e-5 --atol 1e-6 --threshold-status fixed を付けて実行する（正式に別値を採用した場合はその値を使う）。
比較reportでpreconditionsが空、steps_comparedにstep1があり、within-run全必須項目がPASS、exact_failures=0、close_failures=0、first_failure=null、status=PASSを確認する。もし差異があればstep/rank/phase/tensorを先頭から調べる。実行完了だけでは数値一致を証明しない。

## Concepts・設計理由
- CANONICAL FIXTURE：公式初期重み・Queue・BN・Optimizerに加え、各rankの同じaugmentation済み2viewとShuffle順を双方に適用。seedだけでは同じ入力を保証しない。
- DDP：all_gather/broadcast/all_reduceが正常でも、MoCoのEMA・InfoNCE・Queue更新まで正しいとは限らない。A0→A1→A2の順で切り分ける。
- 原式検算のinput-scale toleranceと、公式対自作tensorのresult-based rtol/atolは別の検証。両者を混同しない。
- A2のfull tensor tracesはサイズが大きくなり得るため、ローカルディスク余裕を事前確認する。A2のstep時間にはtracing overheadがあるのでA5のthroughput測定の代わりにしない。

## 不採用・実行境界
Cometの既定はtrueなので診断時はfalseを明示する。GPU OOM、NCCL不調、エラー時はbatch/K/precisionを勝手に変更しない。A3=1025step、A5=100〜300step、Spec Bのfresh 10epochは**別許可**。本書はコード変更・GPU実験を行っていない。

## ソース
- https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/conf/moco_imagenet_parity.yaml
- https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/parity/preflight_moco_imagenet.py
- https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/parity/build_moco_parity_fixture.py
- https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/parity/moco_imagenet_parity.py
- https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/dev/scripts/parity/compare_moco_parity.py
