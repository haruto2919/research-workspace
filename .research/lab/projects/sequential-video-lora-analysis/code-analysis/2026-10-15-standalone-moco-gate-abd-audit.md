---
project: sequential-video-lora-analysis
date: 2026-10-15
analysis_mode: audit
detail: full
level: mid
source_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_branch: dev
source_commit: 8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a
source_dirty: unknown
targets:
  - self_supervised/moco/standalone_moco.py
  - scripts/smoke/smoke_moco_cifar10.py
  - scripts/smoke/moco_verification_reporting.py
  - conf/moco_standalone.yaml
  - test/model/test_moco_standalone.py
  - test/model/test_moco_cifar10_reporting.py
related_spec: ../specs/2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md
---

# Code Audit: Standalone MoCo Gate A/B/D確認とGate C未実施

## 対象と解析条件

2026-10-15にユーザーは、承認済みStandalone MoCo v2 CIFAR-10 specを実装したと報告し、Gate C（500-step pilot）は未実施と明示。「正しく実装できているかを確認する方法」を依頼。GitHub上の `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` のHEAD `8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a` をread-onlyで調査。SSHサーバ上のlocal HEAD, dirty/untracked, CIFAR cache, GPU健全性、実行logと出力artifactは確認不能。**テスト・smoke・500stepはこの監査では実行していない。**

Approved SSOT: [2026-10-15 standalone MoCo spec](../specs/2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md)。
公式の固定参照：`facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f` の`moco/builder.py`・LICENSE。

## Architecture Summary：コード直接確認

| コンポーネント | 確認した構造 | spec照合 |
| --- | --- | --- |
| `self_supervised/moco/standalone_moco.py` | CIFAR small-stem ResNet18（random init、Conv3x3 stride1、maxpoolなし）、Q/K Backbone＋MLP、EMA、randn正規化Queue、InfoNCE | 基本実装が一致 |
| `StandaloneMoCo.forward` | Query forward → Key EMA → Key forward no_grad → Queue snapshotでlogits → detached Key enqueue → callerがbackward/SGD | **ユーザー採用したMeta方式と一致** |
| `conf/moco_standalone.yaml` | dataset CIFAR10 train、K1024、m0.999、T0.2、bs32、seed0、SGD LR.00375 momentum.9 WD.0001、20step fixed、CPU/Comet off | デフォルト設定は一致 |
| `scripts/smoke/smoke_moco_cifar10.py` | 2view augmentation、full CIFAR trainのseed付きsingle_epoch_order、SGD/optional cosine、resume、ログ/8 artifacts | メイン学習・監査経路を実装 |
| `moco_verification_reporting.py` | report.jsonから14 checksのPASS/FAIL/NOT_RUN、Gate status、4グラフ・Markdown | 出力契約は存在 |
| `test/model/test_moco_standalone.py` | A01-A11等数学/Queue/EMA/BN/順序・不正入力のテスト | Gate Aのためのテストが存在 |
| `test/model/test_moco_cifar10_reporting.py` | 偽CIFAR640枚・20step、Hydra/transform/Queue/保存復元/resume/失敗系のテスト | Gate Dとrunner単体。**実CIFAR10は不要**なのでGate Bの代用不可 |

## Key Decisions and Trade-offs

- Official-like random normalized Queue K=1024が初回から負例全件として存在。32key/stepで上書きし、**step20はptr640, real640, random384**。全置換はstep32。
- 初回EMAは初期Query/Keyが同一なのでparameter値は変化しない。Key BatchNorm running buffersはno_gradでもtrain forwardで更新され得る。
- 単一device・shuffle BN/DDPなし（公式ImageNet/ResNet50完全再現ではない）。
- `gates.A`はrunnerでは`NOT_RUN`を記録し、別pytestを要する。`gates.D`はrunner内でcheckpoint別instance roundtripを意味し、resumeの真正な再現は別pytestを要する。
- 実CIFAR10 20stepのみが`gate_b_smoke20`であり、fake dataset（640枚）の20stepは`custom`と記録され、Gate B=NOT_RUN。これは正常な区別。
- Gate Cが未実施なら、長時間の勾配暴走・collapse/representation qualityの成功を主張しない。

## Findings

| ID | 分類 | 箇所 | 観測内容と意味 | 次の確認 |
| --- | --- | --- | --- | --- |
| F01 | Confirmed by code（仕様差分） | `scripts/smoke/smoke_moco_cifar10.py::validate_config` | `logging.comet_enabled=True` は`ValueError`。承認specではComet任意ON／既定OFFだが、現行コードは**ONをサポートしない**。デフォルトのComet OFFでGate A/B/Dを検証する障害ではないが、specの完全準拠ではない。 | 今回のscopeでComet opt-inを実装するか、spec側の記述を変更するかは別途承認/修正依頼が必要 |
| F02 | Confirmed by code（引用表記の不一致） | `self_supervised/moco/standalone_moco.py` 冒頭docstring | 固定公式参考コードを`CC-BY-NC 4.0`と記載しているが、公式commit `LICENSE` は**MIT License**。数学・学習挙動への影響はないが、参照ライセンスの表記は修正が必要。実際に流用したコード量を確認しMIT license noticeを適切に保持する。 | Reference/LICENSE表記の修正。コード改変は今回行わない |
| F03 | Confirmed by code（coverage boundary） | `moco_verification_reporting.py::gate_statuses` | Runner `Gate A=NOT_RUN` は設計どおり、Gate D=PASSもcheckpoint roundtrip部分だけであり、pytestのA/DテストPASSを意味しない。 | pytestの2テストファイルを別実行し、実際の終了コードと結果を保存 |
| F04 | Confirmed by code（fake vs real） | `test/model/test_moco_cifar10_reporting.py` | `FakeCIFAR10` 640枚の20step成功が報告されても、実CIFAR10 50,000枚の20step証拠にはならない。fakeは`custom`かつGate B NOT_RUNとして設計。 | 実CIFAR10 20step fresh runのreport.jsonでGate B PASSを確認 |
| F05 | Potential issue（GPU/長時間未確認） | スモーク・GPU環境 | CUDAの可用性、full dataset cache、20step実学習、500step pilotの数値安定性はGitHubのコードからは証明できない。 | 実サーバのpreflight → real Gate B → Gate Cは別承認 |

## Edge Cases and Failure Modes

- `data.download=false`のためCIFAR10 cacheがない場合はデータセットread時に失敗し、無断downloadはしない。失敗報告は`report.json`内`status=failed`と理由を確認。
- NaN、wrong shape/dtype、Queue ptr/count不整合、Kがbsで割り切れないケースはfail-fastのテストあり。test存在とテスト成功は区別。
- Queueのデータが初期randomから実Keyに置換されてもlossが各step単調減少するとは限らない。20stepではrandom384列が残るので、同じCIFAR画像であっても連続stepのloss絶対値だけで性能主張をしない。
- CUDAを要求するときは`training.device=cuda`を明示。GPU不調やDriver問題に対し勝手な他GPU fallbackなし。
- `verify_checkpoint_roundtrip`は別インスタンスへmodel/optimizer/scheduler stateを復元して比較。runner実行の最後に行うもので、別process resumeと同じ長さのuninterrupted runを比較するテストは`test_gate_d_resume_reproduces_the_uninterrupted_run`が担う。

## Testability and Verification Gaps：具体コマンド

実サーバでのread-only preflight：

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
git branch --show-current
git rev-parse HEAD
git status --short
python -c 'import torch,torchvision,pytest,hydra; print(torch.__version__,torchvision.__version__,hydra.__version__)'
```

A/D（CPU・人工/偽データ）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest -q -o addopts='' \
  test/model/test_moco_standalone.py test/model/test_moco_cifar10_reporting.py
```

既存LoRA回帰（CPUで実行可能な対象）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest -q -o addopts='' \
  test/model/test_lora_linear.py test/model/test_vit_lora_frame_encoder.py \
  test/model/test_vit_lora_checkpoint.py test/model/test_vit_lora_reporting.py
```

B（**cached実CIFAR10**・別run directory）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m scripts.smoke.smoke_moco_cifar10 \
  data.root=/mnt/NAS-TVS872XT/dataset/CIFAR10 \
  data.download=false training.device=cpu
```

成功時出力8ファイル：`resolved_config.yaml`, `report.json`, `loss.png`, `similarity.png`, `grad_update.png`, `queue.png`, `verification_summary.md`, `checkpoint.pt`。20step Gate B: `status=completed`, `run_kind=gate_b_smoke20`, `gates.B=PASS`, 全`checks` PASS, `completed_steps=20`, `dataset.sample_count=50000`, `samples_consumed=640`, 最終`queue_ptr=640, real_key_count=640, random_remaining_count=384`, Key grad max 0、EMA within tolerance、Query Backbone/Projector updateあり、checkpoint roundtrip pass。`gates.A=NOT_RUN`と`gates.C=NOT_RUN`は正常、`gates.D=PASS`はrun内のroundtripのみ。

数値は`report.json`の`history`20stepとPNGを比較。`loss`は単調減少しなくてもGate B FAILにはしない。人工/偽Datasetのpytest PASSから実CIFARのGate B PASSを推測しない。

## Research Reproducibility Considerations

- actual local HEADとdirty/untracked、data cache、PyTorch/torchvision/CUDA、実行host、seed、global stepsは`report.json`へ記録。GitHub commitだけではlocal未push差分を含めた再現はできない。
- 実CIFAR10 Gate Bをfresh（resumeなし、default settings、50,000枚）として実行し、custom runと区別。Queue random残数が常に0になるのはstep32以降であって、20step smokeでは384。
- Gate Cは未実施。コード上の`gate_c_pilot500`やテストで加工した500step synthetic historyは学習結果ではない。ユーザーの申告と実測Evidenceを区別。

## Unverified Assumptions

- ユーザーの研究サーバにある現行branch/dirtyファイルが上記GitHub commitと一致するか。
- 現環境でA/B/D testsを実行すると成功するか。結果のstdout/exit codeは未提供。
- CIFAR10 trainの50,000画像が指定rootに完全に配置されているか。
- Gate Bの実run artifact、CPU/GPUの可用性、checkpoint、長時間数値安定性。

## Suggested Handoff

まずユーザーがA/D pytest＋実CIFAR10 Gate Bと既存LoRA regressionを実行し、`verification_summary.md`/`report.json`、`git status`、pytest結果を照合。F01（Comet仕様差分）とF02（公式MIT表記）の対応は研究コードの修正タスクに引き渡すことができるが、**本監査ではコード・spec・experiment記録を一切変更していない。** F01を修正せずComet OFFに限るなら、その制約を明示して「Gate A/B/Dの範囲内で工学的検証成功」と表現する。
