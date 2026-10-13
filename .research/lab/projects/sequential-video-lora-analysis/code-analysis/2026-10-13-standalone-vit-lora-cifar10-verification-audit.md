---
project: sequential-video-lora-analysis
date: 2026-10-13
analysis_mode: audit
detail: compact
level: mid
source_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_branch: dev
source_commit: cbc303c05502e914798e25717c4a9b29e98c8285
source_dirty: unknown
local_reported_branch: dev
local_reported_commit: 8531a2eb6a632791629a41d621a49c260b43cd4f
local_reported_dirty: true
targets:
  - model/vit/lora_linear.py
  - model/vit/vit_lora_frame_encoder.py
  - scripts/smoke/smoke_vit_lora_cifar10.py
  - test/model/test_lora_linear.py
  - test/model/test_vit_lora_frame_encoder.py
  - test/model/test_vit_lora_checkpoint.py
  - test/model/test_vit_frame_encoder.py
related_spec: .research/lab/projects/sequential-video-lora-analysis/specs/2026-10-13-standalone-vit-lora-cifar10-verification-spec.md
---

# Standalone ViT-LoRA CIFAR-10 verification audit

## 対象と解析条件

- 2026-10-13 approved spec と、ユーザーが貼り付けた2026-10-13 16:54–16:55 JSTのローカル実行ログを照合。
- 現GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` HEAD `cbc303c05502e914798e25717c4a9b29e98c8285` の対象コード・テストを読み取り専用で参照した。GitHub側のコードを実行したわけではない。
- ユーザーのローカル実行は `dev@8531a2eb6a632791629a41d621a49c260b43cd4f` と未コミット・未追跡ファイルを含む。ローカル実ファイルへのアクセスはない。現GitHubの `smoke_vit_lora_cifar10.py` はログ形式・checkpointファイル名がローカル実行出力と異なり、同一実装と判定しない。

## Architecture Summary

CIFAR10 train → checkpoint互換の224×224処理 → frozen ViT + Q/V LoRA → CLS → trainable 10-class Linear Head → CrossEntropy → AdamW。LoRA式は `base(x)+(alpha/rank)*B(A(dropout(x)))`。自作wrapperはBaseの元Linearを保持、B=0初期化、Aはランダム初期化（GitHubの読み取り対象における構造）。

## Key Decisions and Trade-offs

- 教師あり分類はLoRAとHeadの共同学習。LoRA-only確認は別の単体テストで行う。
- 指標の目的は学習経路・勾配・保存の健全性であり、CIFAR-10の汎化性能・動画の時間情報獲得を証明しない。
- 現ローカルrunはCPUへfallback。GPUのCUDA/NVML障害は独立の環境課題。

## Findings

| ID | 分類 | 箇所 | 内容 | 影響条件 | 次の検証 |
| --- | --- | --- | --- | --- | --- |
| F-001 | Observed from user-supplied log | smoke runtime | CIFAR10 training 50,000件からseed0で64件抽出、batch16、20step。train loss 2.502499→0.117905、最後のbatch acc 1.0。Logits (16,10) | Stage Bの短時間学習経路 | ログに加えてpytest合格を記録 |
| F-002 | Observed from user-supplied log | smoke runtime | Q/V 24 wrapped projection、LoRA A/B計294912、Head7690、Base85798656、Base更新0。LoRA A/B 24/24ずつ・Head2/2変更。最初のA gradient/update 0、step2から非ゼロ | 規定Q/Vの学習・freeze | Base parameter gradientなし、全optimizer対象が正しいことをunit testで確認 |
| F-003 | Observed from user-supplied log | checkpoint runtime | adapter + head保存後に再ロードしeval logits max abs diff 0; head-only reload異なる | Stage B checkpoint roundtrip | 追加のcheckpoint破損・metadata不整合negative testを実行 |
| F-004 | Confirmed by GitHub and log comparison | dev vs local | 現GitHub dev HEADとローカル実行基準SHA/dirtyファイルが不一致。GitHub版smokeはjson print・`adapter_head.pt`と`report.json`、提示されたローカル実行はINFOログ・`vit_lora_cifar10_adapter_head.pt` | リモート版とローカル版の同一性を主張する場合 | ローカルgit diff / statusとリモートとの差分を取得 |
| F-005 | Unknown / verification gap | Stage A | 全7Q/K/V非空組合せ、初期出力等価性、A初回勾配0/B非ゼロ、2stepのA更新、bad config、非回帰のpytest結果は提示されていない | approved Stage A完了判定 | ローカルの関連pytest、既存test/model全体を実施 |
| F-006 | Observed from user-supplied log | CUDA initialization | NVML初期化不能、CUDA unknown error、resolved device cpu | GPUでの再実行時 | GPU自体は別途診断。今回のCPU成功の無効理由にはならない |

## Edge Cases and Failure Modes

- `processor_config.json`の404後に`preprocessor_config.json`等が成功し実行継続した。今回のrun失敗と解釈しない。
- `classifier.weight/bias`のUNEXPECTEDは分類checkpointからHead無しViTModelをロードする場合に整合的。missing/mismatchedが空であることを併記する。
- 小規模64枚を20stepで反復した訓練accuracy 1.0は過学習と両立する。テスト性能、動的特徴学習の証拠とはみなさない。

## Testability and Verification Gaps

ローカルtreeにあることを確認して実行:

```bash
python -m pytest -q \
  test/model/test_lora_linear.py \
  test/model/test_vit_lora_frame_encoder.py \
  test/model/test_vit_lora_cifar10_smoke.py \
  test/model/test_vit_frame_encoder.py

python -m pytest -q test/model
```

GitHub `dev` 上の関連テストは `test_lora_linear.py`、`test_vit_lora_frame_encoder.py`、`test_vit_lora_checkpoint.py`、`test_vit_frame_encoder.py`。一方 `test_vit_lora_cifar10_smoke.py` は現在のGitHub `dev`に存在せず、ユーザーローカルの未追跡ファイルである。存在するなら `test_vit_lora_checkpoint.py` も実行する。

差分確認:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git remote -v
git diff -- model/__init__.py model/vit/__init__.py
```

## Research Reproducibility Considerations

runは `torch 2.14.0+cu130 / transformers 5.17.0`、seed0、CIFAR10 train subset64、224 resize、mean/std=0.5、LoRA Q/V r8 alpha8 dropout0、AdamW LR1e-3 WD0、CPU。既存画像での短時間経路確認以上を主張しない。再現には未コミット・未追跡のソース一式を特定する必要がある。

## Unverified Assumptions

- ローカル未コミット版の実ソースをGitHubコネクタからは直接閲覧できていない。
- Stage A/非回帰のテスト結果、ローカル全差分、実際のGPU復旧状況は未確認。

## Suggested Handoff

まずローカルで関連pytestと既存モデルpytestを実行し結果を採取。失敗があれば最小差分の実装修正判断を行う。研究specの`implemented`昇格は、Stage A/B・非回帰・source provenanceが揃ってから判定。実験runログのResearch Workspaceへの格納は別のEvidence整理として扱い、現監査ではspec/README/experiment logは変更しない。
