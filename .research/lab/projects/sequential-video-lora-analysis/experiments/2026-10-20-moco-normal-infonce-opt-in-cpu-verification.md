---
project: sequential-video-lora-analysis
date: 2026-10-20
record_type: user-provided-implementation-verification
status: cpu-pass-gpu-not-run
evidence_level: cpu-unit-integration-regression
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: dev
research_commit: dc502e416aa26ed4e0791767fa5a72f7843e399d
baseline_commit: 1ce9badcbae3e2df08fa400944e4c385785063f9
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
source: user-shared-terminal-output
---

# Meta式InfoNCE通常MoCo opt-in — CPU実装・非回帰検証（ユーザー実行）

## 結論・Evidence境界

**CPU Engineering Gate：PASS**（ユーザー提供の2026-10-20研究サーバ実行ログ）。これはChatGPTがリモートGPU/SSHサーバで実行した結果ではなく、ユーザーの貼り付けたターミナルログを解釈したもの。現時点では**実画像A2 GPU正式Meta比較、A5速度/VRAM実測、長期A3、GPU A4は未実施**であり、CPUのPASSを公式との新条件での数値一致として扱わない。

- 対象：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、`dc502e416aa26ed4e0791767fa5a72f7843e399d`。ユーザーの`git status --short`は空行で、同時点の研究コードworktreeはclean。
- Meta公式clone：ユーザー報告`/mnt/HDD12TB-1/ishikawa/facebookresearch-moco` のHEADは`a419ef282fc9885015674e77f58c59447206a27f`。Meta cloneの`git status --short`はこのログに含まれずclean状態は未確認。
- 前回A1・16step PASS時点の`1ce9badc...`からの変更は**6ファイル、+1323/-114行**。対象：`conf/moco_imagenet_parity.yaml`、`scripts/parity/compare_moco_parity.py`、`scripts/parity/moco_imagenet_parity.py`、`self_supervised/moco/ddp_moco.py`、`test/model/test_moco_ddp.py`、`test/model/test_moco_parity_harness.py`。
- Python 3.12.3、pytest 9.1.1、hydra-core 1.3.7、anyio 4.15.1、`CUDA_VISIBLE_DEVICES=""`、`OMP_NUM_THREADS=4`（CPU、glooの複数processを含む）。

## 実行結果

### Git/config check

- `git branch --show-current`：`dev`。
- `git rev-parse HEAD`：`dc502e416aa26ed4e0791767fa5a72f7843e399d`。
- `git status --short`：空（clean）。
- Hydra config selectorの単独Pythonテスト：
  - 無指定`self`、`moco.info_nce_impl=meta_einsum`のopt-inが動作。
  - Config identity hashは異なる。
  - `moco.info_nce_impl=meta_einsum`＋`diagnostic.info_nce_impl=self`の衝突を正しく拒否。
  - 出力：`CONFIG CHECK PASS`。

### 限定CPUテスト

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider \
  test/model/test_moco_parity_harness.py \
  test/model/test_moco_ddp.py \
  -k "selector_table or refused_selections or config_identity_v2 or v2_resume or legacy_resume or disabled_monitor or light_monitor or update_norm or measurement_windows or normal_meta_einsum_run or formal_implementation or formal_resume" -rs
```

**59 passed、178 deselected、256.79s (4m16s)**。59選択の中にテスト失敗・skipなし。ユーザー貼付けのCLIの`-k`付近は表示上の一部崩れがあるため、上記はユーザーが意図したテスト対象を示す再構成コマンドで、厳密なシェル履歴の逐語転記ではない。

### 全MoCo関連CPU非回帰

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
python -m pytest -q -p no:cacheprovider \
  test/model/test_moco_standalone.py \
  test/model/test_moco_cifar10_reporting.py \
  test/model/test_moco_ddp.py \
  test/model/test_moco_parity_harness.py \
  -rs
```

**279 collected / 279 passed、1188.37s (19m48s)**。失敗・skip・xfailedは最終サマリーに出現していない。ログはユーザー環境の`$HOME/moco_verification/cpu_tests.log`にtee保存された（GitHub上に同ログの実体をコピーしたわけではない）。

## 何が検証されたか

| 領域 | 対応する観測・テスト | 範囲 |
|---|---|---|
| B2/B3 | default=self、opt-in=meta_einsum、diagnosticとの競合拒否、Config identity差 | CPU PASS |
| B5 | v2 checkpoint/sidecar metadata、同一mode Resume、mode不一致拒否、v1 Legacy assertion、rank差で集団fail-fast | CPU test PASS |
| B6' | monitor disabledでsnapshot/norm/scalar計測0、light monitorの指定step記録、update_norm実行時だけQuery clone、WindowTimerの境界同期 | FakeBenchAdapterを使ったCPU test PASS |
| B6'/B10 | `normal_meta_einsum`の3step CPU/glooでN1（meta）、N1r（resume）、N0（self）保存・mode/provenanceの整合性を検証 | CPU test PASS |
| B9/B10 | Standalone/CIFAR、以前のA1・16step診断、DDPメカニクス、比較器、judgeの非回帰 | 279 tests PASS |

### 解釈上の注意

- ログにある`PASS (formal_fp32_parity PASS, m_vs_s1_bitwise NOT_ALL_BITWISE, ...)`は**judgeの合成reportを用いるunit testの標準出力**であり、新しい本物のMeta公式CUDA比較を実行した記録ではない。
- `formal_implementation` CPU testは、自作`meta_einsum`runのmetadataを`meta_reference`へ書き換えた**テスト用stand-in**で比較器の契約をテストしている（実Metaコードはこのテストでは実行していない）。
- monitor testの`positive_cosine_mean>1`などは、`FakeBenchAdapter`が正規化ベクトル由来ではないLinear logitsを発行しているためで、ImageNet実画像runのcosine結果ではない。学習品質の証拠として利用しない。
- `torch.multiprocessing.spawn`のSIGTERMは失敗ケース検証時の子process終了に伴うログ。テストsuiteの最終statusは全件PASS。`COMET WARNING ... deprecated`も失敗ではない。
- CPU/glooの数値・run効率を、GPU/NCCLの正確性・throughputへ外挿しない。

## 未確認点と次の判断

1. **specの所在とAuthority**：Research Workspace`haruto2919/research-workspace@main`の対象プロジェクト`specs/`一覧には、今回の通常InfoNCE opt-inの正式specが2026-10-20の確認時点で見当たらない。ユーザーに現行specのURL/保存pathを確認し、B1〜B12の具体的受入条件と1対1照合が必要。
2. **Meta cloneのclean**：HEADは確認されたが`git status`は今回の貼付けなし。
3. **GPU A2/A5**：新しい通常経路での実画像224px Meta公式との同一fixture 1step照合、監視ON/OFFの実測、VRAM等は別Gate・別許可。
4. **A3/A4**：K65536/Queue一巡・実GPUでのcheckpoint/resume正式比較は未検証。

## 関連

- [2026-10-19 B1〜B12採用記録](../../../secretary/notes/brainstorm/2026-10-19-moco-reference-infonce-normal-path-design-decisions.md)
- [2026-10-19 A1 16step正式GPU一致](./2026-10-19-meta-moco-a1-infonce-16step-gpu-parity-evidence.md)
- [2026-10-18 A1 16step実装検証](./2026-10-18-meta-moco-a1-infonce-16step-parity-implementation-verification.md)
