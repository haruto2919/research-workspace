---
project: sequential-video-lora-analysis
date: 2026-11-11
record_type: implementation-cpu-verification
status: verified
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
base_commit: 559073f517a082e41844ba495e90d57035fe2bdf
research_commits: [c478269, 59bade2]
spec: ../specs/2026-11-11-vit-moco-pipeline-refactoring-spec.md
---

# ViT＋MoCo＋LoRAの本流のリファクタリング：実装とCPUでの確認

[承認済みspec](../specs/2026-11-11-vit-moco-pipeline-refactoring-spec.md)に従って実装し、CPUでの確認（V-C、V-T、V-P、V-I）はすべて満たした。研究コードは次の2つのcommitにした（pushはしていない）。

- `feature-vit-moco@c478269`：正解データを取るスクリプト
- `feature-vit-moco@59bade2`：リファクタリング本体

GPUでの確認（V-G、V-D）も、同日に満たした（下記）。

## 要点

- **計算は変わっていない**：今のコード（`559073f`）で取ったCPUの正解run（10本）を新しいコードで実行し直し、30項目すべてがbitwise一致した。
  - 一致した項目：全stepのmetrics、最終状態、同一性ハッシュ、probe、checkpoint
  - 古いcheckpointからのresumeも含む
- **G2のcheckpointがそのまま使える**：
  - 新しいコードで `+experiment=g2` から計算した同一性ハッシュが、G2の記録（`9fde5def…`）と一致した。
  - step 6,255のcheckpointは、strictで読み込めた。
- **テスト**：
  - 全テストで、失敗とエラーのIDの集合は変更前に含まれるものだけだった。
  - 失敗が1件減ったが、これはGPUの有無で結果が変わるparityのテストで、今回の変更とは関係ない。
- **本流の独立**：本流は `test/`・`scripts/parity/`・`scripts/smoke/` をimportしていない（grepで確認）。

## 変更したもの

| 区分 | 内容 |
|---|---|
| 入口 | `main_moco.py`（Hydra、`conf/moco`）。古い `scripts/vit_moco/`（6ファイル）と `conf/moco_vit_*.yaml`（7ファイル）は消した |
| 設定 | `conf/moco/`：`config.yaml`、`trainer/`（cpu、cpu_ddp2、gpu1、gpu2）、`model/`（lora、full_ft）、`data/`（imagenet、anet_frames、anet_clips）、`diagnostics/`（disabled、probe）、`experiment/`（l4、l5、a1、a2、clip_a1、clip_a2、g1、g2） |
| 学習 | `training/`：config、model_factory、step、loop、checkpoint、probe、report、launcher |
| 評価 | `evaluation/`：temporal_order（並び順評価）、collapse_compare（diagnosticの比較CLI） |
| データ | `dataset/` の直下：moco_data、moco_transforms、activitynet_frames、activitynet_sequential |
| 道具 | `utils/` の直下：hashing、provenance、jsonio、rng、runtime |
| 公開 | `model/vit/`、`self_supervised/moco/`、`dataset/`、`utils/` の `__init__.py` に書き足した。`training/__init__.py` は主な名前を公開する。`evaluation/__init__.py` はCLIなので、モジュールをimportしない |
| parity・smoke | 移した道具の定義を消し、新しい場所からimportするようにした。古い名前は引き続きimportできる。ロジックは変えていない |
| テスト | 本流に関わる9ファイルを、新しいモジュールと設定に移した（236件）。共通ヘルパーは `test/model/moco_helpers.py`、fixtureは `test/model/conftest.py` にまとめた。許可リストを確かめていたテストは消した（一覧は `test/model/refactor_deleted_tests.md`） |

使い方の例：

```bash
CUDA_VISIBLE_DEVICES=0,1 python main_moco.py +experiment=g2
CUDA_VISIBLE_DEVICES=0 python main_moco.py +experiment=g2 trainer=gpu1
python main_moco.py +experiment=a1 trainer=cpu model=full_ft
python main_moco.py +experiment=g1 data.clip.frame_stride=32
```

## 確認（CPUのみ）

### V-C：正解データとのbitwise一致

- 正解データ：
  - `test/refactor_golden/capture_golden.py` を基準commit（`559073f`）のコードで実行した。
  - 入力は固定のパスに置いた：`log/refactor_golden/fixtures`（小さなViT、小さなImageFolder、合成動画、manifest）。
  - 出力は `log/refactor_golden/559073f`。
  - CPU（gloo）の2プロセスで、次の10本を取った。

    | データ | run |
    |---|---|
    | ImageNet | LoRA・Full FT（診断あり）、診断なしLoRA 5 step、そのstep 3からのresume |
    | ActivityNet単一フレーム | LoRA |
    | ActivityNet clip | LoRA・Full FT |
    | sequential clip | LoRA 7 step、epochの境界からのresume、epochの途中からのresume |

- `test/model/test_refactor_golden.py`：
  - 各runの古い形式の `resolved_config.yaml` を、新しい形式に読み替えて実行する。
  - resumeしたrunは、正解データ側の古いcheckpointから再開する。
  - **30 passed**。一致した項目は次のとおり。
    - `metrics.jsonl`（時間の値を除く）
    - `final_state_sha256`、`config_identity`、データの記録
    - probe・update・loaderの記録、`diagnostic_summary` の軌跡
    - checkpointのmodel stateとAdamW state、torchのRNG
- 一致の対象から外したもの：
  - mainプロセスのPythonとnumpyのRNG状態。どちらもrunの中で一度もseedされず、使われてもいないので、プロセス起動時の値のままrunごとに違う（古いコード同士でも違う）。
  - sequential型のrunでは、Pythonとnumpyの状態が全stepで変わらないこと（使われていないこと）をテストで確かめた。
  - epoch型のrunでは、Pythonの状態はepochごとにseedされるので、一致を確かめた。

### V-T：テスト

- 全テスト（`CUDA_VISIBLE_DEVICES=""`）：

  | | 失敗 | 成功 | エラー |
  |---|---|---|---|
  | 変更前 | 197 | 825 | 18 |
  | 変更後 | 196 | 778 | 18 |

- 変更後に失敗したテストとエラーは、すべて変更前にも失敗またはエラーになっていたもの。
  - 中身は旧フレームワークのテスト：image models、model factory、checkpoint、video models、video folder。
- 変更前だけ失敗した1件：`test_moco_5epoch_pair.py::test_meta_then_self_in_two_comet_experiments_and_the_pair_passes`。
  - 変更前は、子プロセスの `assert torch.cuda.is_available()` で失敗していた。今回は通った。
  - GPUの有無に左右されるテストで、parityのロジックは変えていない。
- 成功の数が減ったのは、許可リストのテストを消したためと、テストを統合したため。

### V-P：既存のrunとの互換（読み取りのみ）

- G2（`log/moco_vit_seq_activitynet_clip/.../2026-11-08_17-26-03-055153`）：
  - 新しいコードの `+experiment=g2` から計算した `config_identity_sha256` が `9fde5deff1374af2…` で、checkpointの記録と一致した。
  - 記録されたidentityとのキーごとの違いは0。
  - G2の古い形式の `resolved_config.yaml` を読み替えても、同じハッシュになった。
  - `checkpoint_step006255.pt` は、新しいコードのモデルにstrictで読み込めた。
- clip A2とG2の古い形式の設定は、`check_config` を通り、同一性ハッシュも記録と一致した（移したテスト）。

### V-I：本流の独立

- `main_moco.py`、`training/`、`evaluation/`、`dataset/`・`utils/` の新しいファイル、`model/vit/`、`self_supervised/moco/` は、`test`・`scripts` をimportしていない。
- 古いモジュール名が残っているのは、docstringの「どこから移したか」と、互換のために残したスキーマ名の文字列だけ。

## specからのずれ・気づいた点

- `evaluation.collapse_compare` のLR比較が記録する `profile` は、新しいrunでは `experiment_name`、古いrunでは `diagnostics.profile` にした。
  - `diagnostic_summary.json` の `profile` も、新しいrunでは `experiment_name` が入る。
- ImageNetのrunに `data.activitynet_manifest` が設定されていても、使わない値として拒否しなくなった（spec §5.3：能力の検査だけを残す）。
- `python -m dataset.activitynet_frames` を実行すると、`dataset/__init__.py` が同じモジュールを先にimportするため、RuntimeWarningが出る（動作には影響しない）。
- テストの移行は、サブエージェントが行った。`test/` だけを変更し、本流の不具合は見つからなかった。結果は私が全テストの実行で確かめた。

## GPUでの確認（2026-11-11、ユーザーの許可を得て実行）

研究コード `feature-vit-moco@59bade2`（clean）を使い、tmuxセッション `vit-moco-a1` で実行した（13:26〜14:19、すべてexit 0）。出力は `log/moco/refactor_check/` の下。

### V-G：既存のrunの記録との一致（GPU 0・1、各設定1 step）

使ったGPUは、既存のrunと同じ GPU 0（RTX PRO 4500 Blackwell）と GPU 1（RTX A4500）。各runは次のコマンドで実行した。

```bash
CUDA_VISIBLE_DEVICES=0,1 python main_moco.py +experiment=<e> model=<m> training.max_steps=1 'diagnostics.probe_steps=[0,1]'
```

| 設定 | 同一性ハッシュ | step 1の入力 | step 1のloss | step 0のprobe（CLS・z_q・q） | VRAM gate |
|---|---|---|---|---|---|
| L4 LoRA | 一致 | 一致 | 一致（7.361582279205322） | 一致 | PASS |
| L4 Full FT | 一致 | 一致 | 一致（7.361582279205322） | 一致 | PASS |
| A1 LoRA | 一致 | 一致 | 一致（7.68887996673584） | 一致 | PASS |
| A1 Full FT | 一致 | 一致 | 一致（7.688879489898682） | 一致 | PASS |
| clip A1 LoRA | 一致 | 一致 | 一致（7.626703262329102） | 一致 | PASS |
| clip A1 Full FT | 一致 | 一致 | 一致（7.626703262329102） | 一致 | PASS |
| G1 | 一致 | 一致 | 一致（7.588408470153809） | 一致 | PASS |

- **V-Gの成功条件（step 1のlossとstep 0のprobeがbitwise一致）を、7設定すべてで満たした**。
- step 1のmetricsのうち、`trainable_grad_norm` だけは記録と最後の数桁が違った。例：L4 LoRAで 2.185853998092884 と 2.1858539980921794（相対差は約1e-13）。
  - これはbackwardの値で、L5・A2で分かっていたGPUのbackwardのrun間非決定性と同じ種類のずれ。
  - そのため、更新後のstep 1のprobeも一致しなかった。spec §2・R1で予想していた範囲。

### V-D：1GPU（`trainer=gpu1`）

- 実行したコマンド：

  ```bash
  CUDA_VISIBLE_DEVICES=0 python main_moco.py +experiment=l4 trainer=gpu1 training.max_steps=2
  ```

  ImageNet LoRA、local batch 256（global 256を1GPUで受け持つ）。
- **2 stepを完走した**（status completed、failuresなし）。
- GPU 0（RTX PRO 4500、32 GB）で、peak reservedは 23,500 MiB、headroomは 8,718 MiB で、VRAM gateはPASSだった。
- RTX A4500（20 GB）の1枚では、同じ設定は載らない見込み（推測。2GPUのときのLoRAは、local 128で約12 GB）。
