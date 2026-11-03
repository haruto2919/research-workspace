---
project: sequential-video-lora-analysis
date: 2026-11-01
record_type: implementation-cpu-verification
status: implemented-cpu-gates-t0-t3-pass-gpu-a1-a2-not-run
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
base_commit: 387456875e40f7e1ad9037b43bf631abc5221e63
code_state: committed
implementation_commit: 167f1c1f6b03e68bdd7d4fae3e5e05d893559d62
spec: ../specs/2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md
---

# ViT＋MoCo ActivityNet単一フレーム（lr_activitynet）：実装・CPU検証記録

承認済みの[ActivityNet単一フレームspec](../specs/2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md)に従い、`feature-vit-moco@3874568` の上に実装した。差分は実装時点では未commitで、同日、ユーザーの指示で `feature-vit-moco@167f1c1` としてcommitした（変更4ファイルのみ、push未実施）。

## 変更ファイル

| ファイル | 内容 |
|---|---|
| `conf/moco_vit_lr_activitynet.yaml`（新規） | `moco_vit_query_collapse_lr` を継承する。`data.source: activitynet_frames`、`data.activitynet_manifest` は2026-10-21のmanifest、`diagnostics.profile: lr_activitynet`、probe `[0,1,2,10,100,256,459]`、`checkpoint_steps: [459]`。出力先は `log/moco_vit_lr_activitynet/<mode>/activitynet_frames/lr_<LR>/<ts>`。LRは引き続き必須 |
| `scripts/vit_moco/query_collapse_diagnostic.py` | 設定部分：`ACTIVITYNET_PROFILE`/`ACTIVITYNET_PRESET`、profileとpresetの対応表 `PRESETS`、`ACTIVITYNET_GPU_RUNS`（10step→probe `[0,1,2,10]`・checkpointなし、459step→probe `[0,1,2,10,100,256,459]`・checkpoint `[459]`）、`ACTIVITYNET_MANIFEST_SHA256`、`activitynet_profile()` を追加した。`lr_profile()`・`learning_rate()`・`check_lr_preset()` は対応表を使うようにした（既存2 profileの挙動は同じ）。`check_config` の変更点：`lr_activitynet` は `activitynet_frames` だけを受け付け、他のprofileは従来どおり `imagenet` 以外を拒否する。GPUの `lr_activitynet` では、長さ・probe・checkpoint・manifestのsha256がすべて上記の組み合わせと一致しない限り拒否する。<br>probe部分：学習データがImageNet以外のときだけ、`data.root`/`data.split` のImageFolderから固定probeを選ぶ。`dataset_samples_sha256` も、そのImageFolderから同じ式で計算する。ImageNetの経路は変更前と同じ値を使う |
| `scripts/vit_moco/moco_vit_ddp_smoke.py` | GPUで許可する `max_steps` を、`lr_activitynet` のときは `(10, 459)` とした。docstringに使用例を追加 |
| `test/model/test_vit_moco_lr_activitynet.py`（新規、43件） | T0、T2、T3 |

### specの§7からの変更点（実行コマンド）

presetの既定値は459step用である。10stepのsmoke（A1）では、probe stepとcheckpointをコマンドで明示的に指定する。stepの長さに応じて値を自動で埋める仕組みは作っていない。指定し忘れた場合は、GPUチェックが起動前に拒否する（`training.checkpoint_steps must lie in [1, max_steps]`）。

```bash
# A1（10step smoke）
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet model.train_mode=lora training.max_steps=10 training.learning_rate=0.0003 'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'
# A2（459step）
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet model.train_mode=lora training.max_steps=459 training.learning_rate=0.0003
```

## CPU Gate

| Gate | 結果 | 内容 |
|---|---|---|
| T0 | **PASS** | presetを組み立てると、差分がdata.source・manifest・profile・probe・checkpoint・出力先だけになる（ImageNetの `data.root`/`split` はprobe用にそのまま残る）。GPUチェックでは、2つの長さで両方式の正の有限なLRを受け付ける。次はすべて拒否する：LR未指定、`max_steps` が1/20/458/460/1000、長さとprobe・checkpointの組み合わせ違い、probeの他の値の変更、recipeの変更、別のmanifest（sha256不一致）、`data.source=imagenet`、resume、他のpresetから `lr_activitynet` を使うこと、`moco_vit_lr_activitynet` から他のprofileを使うこと、他の診断profile（なし／`lr_sensitivity`／`lr_long`）でActivityNetを使うこと。`lr_sensitivity` は10step、ImageNetの基本経路は1/10/20stepのまま。実際のmanifestのsha256がpresetの値と一致することも確認した |
| T1 | **PASS** | 関連する既存のテストファイル5つ（`test_query_collapse_diagnostic.py`、`test_vit_moco_lr_long.py`、`test_query_collapse_prefix_comparison.py`、`test_moco_vit_ddp_smoke.py`、`test_vit_moco_ddp.py`）は**170件すべてPASS**。全テストの結果は次のとおり。<br>・変更後：36 failed / **903 passed**。失敗は `abn_r50` の32件（標準入力待ちのEOFError）とNASのKinetics400に依存する4件で、2026-10-31の基準（`46ad983`）の失敗ID集合と**完全に一致**した。<br>・変更前（`3874568`、別のworktree）：68 failed / 828 passed。上の36件に加え、x3dの32件がモデル重みのダウンロードで `HTTP Error 403` になった。x3dは変更後の実行では通っており、ネットワーク側の一時的な失敗と判断した（今回の変更はx3dの経路に触れていない）。<br>・**変更後に新しく失敗したテストは0件**（変更後の失敗集合は変更前の部分集合）。passedの増加分43件は新規テストの数と一致する |
| T2 | **PASS** | tiny ViT、2rank、gloo。合成動画4本（40秒、16フレーム、1epoch=4step）のmanifestで、LoRA・Full FTを各3step、LoRAを2step実行し、すべて完走した。probe・step3のcheckpoint・summary（PASS、profile `lr_activitynet`、LR 3e-4）が揃った。probeのselection・transform・samples・入力ハッシュ・`dataset_samples_sha256` は、同じ設定のImageNet runと一致し、ImageNet runの `dataset_samples_sha256` は従来どおり学習データの値と一致した |
| T3 | **PASS** | 同じActivityNetフレームで学習したLoRAとFull FTの組み合わせは `compare_runs` で `comparable=true` になり、step0の特徴がbitwise一致した。2stepと3stepのrunの組み合わせは `compare_prefix` で `comparable=true`（共通probe `[0,1]`）。ImageNet runとActivityNet runの組み合わせは、どちらの比較器でも `comparable=false` になった（config identityと学習入力が異なる）。比較器のコードは変更していない |

## 親による追加確認（学習なし）

研究サーバ上で `CUDA_VISIBLE_DEVICES=0,1` を指定し、実際のpresetをHydraで組み立てて、`prepare_config` と `check_lr_preset` だけを実行した。

- `moco_vit_lr_activitynet` の両方式・両方の長さ（459step、および10stepにprobe・checkpointを明示したもの）は受け付けられた。
- 10stepでprobe・checkpointを指定しなかった場合、1,000step、存在しないmanifestは拒否された。
- 既存の `moco_vit_lr_long`（1,000step）と `moco_vit_query_collapse_lr`（10step）は引き続き受け付けられた。
- 実際のImageNet trainから、ActivityNet経路と同じ方法で固定probeを選び直した。**入力ハッシュは `6af50745…08d8bf` で、L4/L5と一致**した。`dataset_samples_sha256`（`75cbd2fc…`）もL5と一致した。

## 未実施・注意

- **GPU A1（10step）・A2（459step）は未実施**で、別に許可を得る必要がある。runnerはdirtyな状態でのGPU runを拒否するため、**実行前に研究コードのcommitが必要**（2026-11-01、`feature-vit-moco@167f1c1` でcommit済み）。
- ActivityNetフレームの読み込み速度は未測定（A1で測る）。
- flake8で新たに出た指摘は、既存コードと同じ命名規則に関するもの（FNE系）だけ。ruffはvenvに入っていない。
- 科学的な結果（ActivityNetでのlossの推移、collapse）についてはまだ何も主張しない。
