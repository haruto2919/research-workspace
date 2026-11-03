---
project: sequential-video-lora-analysis
date: 2026-10-31
record_type: implementation-cpu-verification
status: implemented-cpu-gates-t0-t3-pass-gpu-l5-not-run
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
base_commit: 46ad9833f764ed7d9bb8df45f2687345b7610f2a
code_state: uncommitted
spec: ../specs/2026-10-31-vit-moco-low-lr-long-run-spec.md
---

# ViT＋MoCo 低LR長めの学習（lr_long）：実装・CPU検証記録

承認済みの[長めの学習spec](../specs/2026-10-31-vit-moco-low-lr-long-run-spec.md)に従い、`feature-vit-moco@46ad983` の上に実装した。差分は**未commit**である。親エージェント（Opus 5.5）が作業を分け、3つのサブエージェント（Sonnet 5.5、effort high）が並行して実装した。担当は、A＝preset・profile・runner、B＝比較器、C＝変更前の全テストの基準記録。統合後の差分確認と全テストの照合は親が行った。

## 変更ファイル

| ファイル | 内容 |
|---|---|
| `conf/moco_vit_lr_long.yaml`（新規） | `moco_vit_query_collapse_lr` を継承する。`diagnostics.profile: lr_long`、probe `[0,1,2,10,100,256,500,1000]`、`checkpoint_steps: [1000]`。出力先は `log/moco_vit_lr_long/<mode>/<source>/lr_<LR>/<ts>`。LRは引き続き必須 |
| `scripts/vit_moco/query_collapse_diagnostic.py` | 設定部分：`LONG_PROFILE`/`LONG_PRESET`/`LONG_GPU_MAX_STEPS=1000`/`LONG_CHECKPOINT_STEPS`/`LONG_DIAGNOSTICS` と `long_profile()` を追加。`lr_profile()` は両方のprofileで真を返す（LRの例外）。`check_lr_preset(config_name, profile)` は、profileごとに対応するpresetからの実行しか認めない。`check_config` は、GPUの `lr_long` についてprobe・1,000step・checkpoint `[1000]` 以外をすべて拒否する。既存profileのチェックは `elif` 側にそのまま残した。<br>比較部分：`compare_prefix()` とCLIの `--comparison-axis prefix` を追加した。同じmode・同じLRで長さの違うrunについて、共通する学習step・probe stepをbitwiseで突き合わせる。`compare_runs` と `compare_learning_rates` は変更していない |
| `scripts/vit_moco/moco_vit_ddp_smoke.py` | GPUで許可する `max_steps` を、`lr_long` のときは `(1000,)`、それ以外は従来の `(1, 10, 20)` とした。`check_lr_preset` にprofileを渡すようにした。docstringに使用例を1行追加 |
| `test/model/test_vit_moco_lr_long.py`（新規、36件） | T0、T2 |
| `test/model/test_query_collapse_prefix_comparison.py`（新規、11件） | T3 |

## CPU Gate

| Gate | 結果 | 内容 |
|---|---|---|
| T0 | **PASS**（36件中の該当テスト） | presetを組み立てると、差分がprofile・probe・checkpoint・出力先だけになる。GPUチェックでは、正の有限なLRを受け付ける。次はすべて拒否する：LR未指定、`max_steps` が1/10/20/999/1001/2000、probeやupdate stepの変更、checkpointの変更、その他のrecipeの変更、他のpresetから `lr_long` を使うこと、`moco_vit_lr_long` から `lr_sensitivity` を使うこと、未知のprofile。`lr_sensitivity` は引き続き10stepだけを受け付ける |
| T1 | **PASS** | 全テストの失敗ID集合が、変更前（`46ad983`、別のworktree）と変更後で**同じ36件**。内訳は `abn_r50` の32件（標準入力待ちのEOFError）と、NASのKinetics400に依存する4件で、いずれも既存の失敗。件数は、変更前が36 failed / 813 passed、変更後が36 failed / **860 passed**（新規テスト47件の分）。既存の `test_query_collapse_diagnostic.py` 69件（D0〜D3、L0〜L3相当のbitwise一致テストを含む）はすべてPASS |
| T2 | **PASS** | tiny ViT、2rank、glooで、`--config-name moco_vit_lr_long` をmax_steps 3、probe `[0,1,3]`、checkpoint `[3]` に縮めて実行し、完走した。両rankのprobe、step3のcheckpointとrankごとのsidecar、summary（PASS、profile `lr_long`、LRを記録）が揃い、出力は新しいnamespaceに保存された |
| T3 | **PASS**（11件） | 同じmodeで長さの違うCPU run（2stepと4step）の組み合わせで `comparable=true` になり、共通のprobe step `[0,1,2]` がすべてbitwise一致した。入力・特徴・LR・identity・manifest・step0・engineering status・完走フラグを改ざんしたケースでは、それぞれ期待どおりの理由が返る。異なるmodeの組み合わせは `ValueError` になる。CLIは出力ファイルを上書きしない |

## 親による追加確認（学習なし）

- 実際のL4 GPU artifactを `compare_prefix` で読み込んだ。
  - LoRA 3e-4のrunを自分自身と比べると、`comparable=true`（共通probe `[0,1,2,10]`、共有学習step 10）。
  - 旧0.03 runと3e-4 runを比べると、LR・identity・特徴の違いがそれぞれ理由として返った。
  - LoRAとFull FTの組み合わせは拒否された。
- 研究サーバ上で、`CUDA_VISIBLE_DEVICES=0,1` を指定してHydraでconfigを組み立て、`prepare_config` と `check_lr_preset` だけを実行した。
  - `moco_vit_lr_long` の両方式（`max_steps=1000`、LR 3e-4）は受け付けられた。
  - `max_steps=10` と、probe `[0,1,2,10]` に戻した設定は拒否された。
  - 既存のL4経路（`moco_vit_query_collapse_lr`、10step）は引き続き受け付けられた。

## 未実施・注意

- **GPU 1,000step run（L5）は未実施**で、別に許可を得る必要がある。runnerはdirtyな状態でのGPU runを拒否するため、**実行前に研究コードのcommitが必要**。
- 1,000stepの設定を、CPU上で実際の `run()` に通すテストはしていない。GPU用のチェックは `prepare_config` 単体と、上記の実機configの確認で検証した。
- ruffはvenvに入っていない。flake8の警告は、既存の命名規則に関するものだけ。
- 科学的な結果（lossの推移、遅れて起きるcollapse）についてはまだ何も主張しない。

2026-11-01追記：ユーザーがこの差分を `feature-vit-moco@3874568` としてcommitし、GPU L5を実行した。結果は[L5 Evidence](2026-11-01-vit-moco-low-lr-l5-1000step-imagenet-gpu-evidence.md)を参照。
