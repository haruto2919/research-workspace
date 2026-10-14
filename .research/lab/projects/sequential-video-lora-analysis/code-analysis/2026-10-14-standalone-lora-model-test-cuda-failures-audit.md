---
project: sequential-video-lora-analysis
date: 2026-10-14
analysis_mode: audit
detail: compact
level: mid
source_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_branch: dev
source_commit: cbc303c05502e914798e25717c4a9b29e98c8285
local_source_commit: not confirmed for this pytest run
local_dirty: unknown for this pytest run
targets:
  - test/model/test_image_models.py
  - test/model/test_model_factory.py
  - test/model/test_videomodels.py
related_spec: .research/lab/projects/sequential-video-lora-analysis/specs/2026-10-13-standalone-vit-lora-cifar10-verification-spec.md
---

# Standalone ViT-LoRA: model test failures under unavailable CUDA

## 対象と解析条件

- ユーザーの添付pytestログ（2026-10-14に提示）: `test/model` 全220件中 **52 passed / 168 failed / 171 warnings（14.01 s）**。
- GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` （閲覧時HEAD `cbc303c05502e914798e25717c4a9b29e98c8285`）の関連3テストソースを静的確認した。
- また、ローカルsmoke報告時の基準commit `8531a2eb6a632791629a41d621a49c260b43cd4f` で同じ3テストファイルの内容とCUDA必須assertが存在することをGitHubで確認した。対象3ファイルのGit blob SHAは読み取り時のdevと基準commitで同一。
- ローカルtreeの未コミット差分はGitHubから直接確認できず、このテストrunの実ソースを再現したとは主張しない。

## Architecture Summary

- 既存画像分類・モデルfactory・動画分類テストでは、各テスト開始直後に `assert torch.cuda.is_available()` を実行し、 `torch.device("cuda")` でモデルをテストする。
- そのためCUDA利用不能の場合、モデル構築、forward/backward、DataParallel等の本来の検証へ進まない。

## Key Decisions and Trade-offs

- CUDA障害時にGPU必須テストを除外するのは、CPU側の非回帰を先に区別するための一時的な検証範囲の限定。GPU部分をPASS扱いしてはいけない。
- LoRA自体のunit試験は直前にユーザーが25件PASSを報告し、実CIFAR-10短時間smokeもCPUで20stepとadapter/head reloadが成功している。これらはモデル全体のGPU試験の合格を意味しない。

## Findings

| ID | 分類 | 箇所 | 内容 | 影響条件 | 次の検証 |
| --- | --- | --- | --- | --- | --- |
| F-001 | Observed from user-provided log | `test/model/test_image_models.py` | 80件が `torch.cuda.is_available() == False` により失敗 | CUDAが利用できない | GPUが正常な環境で再実行 |
| F-002 | Observed from user-provided log | `test/model/test_model_factory.py` | 72件が同じ事前assertで失敗 | CUDAが利用できない | GPUが正常な環境で再実行 |
| F-003 | Observed from user-provided log | `test/model/test_videomodels.py` | 16件が同じ事前assertで失敗 | CUDAが利用できない | GPUが正常な環境で再実行 |
| F-004 | Confirmed by GitHub sources | 3テストファイル @ `8531a2e` | 今回のLoRA追加以前の基準commitにもCUDA必須assertが存在。3ファイルのblob SHAは閲覧時devと一致 | 回帰の原因をLoRA変更とみなす場合 | ローカル差分確認で別の追加変更がないか点検 |
| F-005 | Unknown | 実モデルforward/backward | CUDA assert以前に失敗しているため、今回失敗した168ケースについて実モデルの機能が成功するかは未確認 | GPU環境回復後 | GPUで各ケースを再検証 |

## Edge Cases and Failure Modes

- 添付ログには`Can't initialize NVML`と`CUDA initialization: CUDA unknown error`。CUDAドライバ、デバイス応答、環境変数などどれが根本原因かはこのログのみから判定できない。
- この失敗を修正する目的でGPU必須assertを削除しても、以後の `device="cuda"` 利用は成功しない。安易なtest skipを完全な回帰成功と報告しない。

## Testability and Verification Gaps

GPU必須の3ファイルを除外してCPU側のtest/modelサブセットを再確認するコマンド:

```bash
python -m pytest -q test/model \
  --ignore=test/model/test_image_models.py \
  --ignore=test/model/test_model_factory.py \
  --ignore=test/model/test_videomodels.py
```

個別GPU状態の読み取り専用確認:

```bash
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('torch CUDA:', torch.version.cuda)"
```

## Research Reproducibility Considerations

- GPU依存テスト168件は未検証であり、研究コードの**全モデル非回帰PASS**とは言えない。
- 52件PASSとStandalone LoRAの25件PASSを混同しない（25件は別途実行された限定範囲）。
- ローカルのcommit/dirty状態は2026-10-13のsmokeログから変わっている可能性があり、今回のpytestと完全に同一と断定できない。

## Unverified Assumptions

- 今回のモデル全体pytest実行時のローカルbranch、HEAD、dirty差分。
- CUDA異常の根本原因と他GPUの稼働状況。
- CUDA正常時に失敗した168件がすべて成功するかどうか。

## Suggested Handoff

- CPU可能な範囲は上記コマンドで確認し、その合否を記録する。
- GPUの接続／ドライバ回復後、元の `python -m pytest -q test/model` を再実行してGPU含む非回帰判定を完成させる。
- この監査はコード/テスト/spec/README/experiment logを書き換えない。
