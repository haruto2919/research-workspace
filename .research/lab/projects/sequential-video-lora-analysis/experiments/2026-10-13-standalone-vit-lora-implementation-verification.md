---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: in-progress
created: 2026-10-13
last_updated: 2026-10-13
spec: ../specs/2026-10-13-standalone-vit-lora-cifar10-verification-spec.md
implementation_branch: dev
implementation_base_commit: 8531a2eb6a632791629a41d621a49c260b43cd4f
---

# 自作ViT LoRA（Q/K/V切替）の実装・Stage A検証

承認済み[spec](../specs/2026-10-13-standalone-vit-lora-cifar10-verification-spec.md)に従い、実装repositoryの`dev`へ自作LoRAとCIFAR-10短時間学習entrypointを実装した。
**Stage A（adapter単体・ViT接続）はCPUで検証済み。Stage B（実CIFAR-10画像の短時間学習）は未実行。**
ローカルにCIFAR-10がなく、ダウンロードと実画像runは別途の許可が必要なためである。
Stage Bのコード経路は、偽のCIFAR-10と小さいViTを使うテストでのみ確認した。
変更は未コミットで、commit・push・GPU run・CIFAR-10のダウンロードは行っていない。

## 環境と基準

- 実装root: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`。origin `tamaki-lab/2026_09_ishikawa_sequential-video-lora`
- 開始時: branch `dev`、HEAD `8531a2e`。`origin/dev`およびspec承認時のHEADと一致し、tracked/untrackedの変更はなかった
- Python 3.12.3、torch 2.14.0+cu130、transformers 5.17.0、torchvision 0.29.0、hydra-core 1.3.7
- この環境ではCUDAを初期化できないため、テストはすべてCPUで実行した
- `google/vit-base-patch16-224`はローカルのHF cacheにあり、`HF_HUB_OFFLINE=1`でも読み込めた
- HF ViTの構造: `ViTModel.layers[i].attention.{q,k,v}_proj`。いずれも`nn.Linear` 768→768で、全12層
- 実体は`model/vit/`にある。`model/backbones/`は他branchの`__pycache__`が残っているだけ

## 変更ファイル

新規:

- `model/vit/lora_linear.py`: `LoRALinear`。元の`nn.Linear`オブジェクトを`base`として保持し、`base(x) + (alpha/rank)·B(A(dropout(x)))`を返す。Aはkaiming-uniform（nn.Linear/PEFTの既定値）でランダム初期化、Bはゼロ初期化。rank・alpha・dropoutの不正値、Linear以外のbase、二重wrapでは例外を出す
- `model/vit/vit_lora_frame_encoder.py`: `canonical_targets`、`inject_lora`、`ViTLoRAFrameEncoder`。処理順は「読み込み → Base全体をfreeze → 対象を差し替え」。差し替える前に全対象の欠損・型・shape・層数・二重注入を確認するため、失敗時にモデルは一部だけ変更された状態にならない。`lora_state_dict` / `load_lora_state_dict`はkeyの完全一致を要求する
- `conf/lora_standalone.yaml`: 独立したHydra preset。既存の`conf/config.yaml`には触れていない
- `scripts/smoke/smoke_vit_lora_cifar10.py`: Stage Bのentrypoint（`python -m scripts.smoke.smoke_vit_lora_cifar10`）
- `test/model/conftest.py`（tiny ViTを返すfixture）、`test/model/test_lora_linear.py`、`test/model/test_vit_lora_frame_encoder.py`、`test/model/test_vit_lora_cifar10_smoke.py`

変更:

- `model/vit/__init__.py`、`model/__init__.py`: `LoRALinear`と`ViTLoRAFrameEncoder`のexportを追加

既存の`ViTFrameEncoder`、`ViTb`、`main.py`、`main_pl.py`、`conf/config.yaml`、`dataset/`、`setup/optimizer.py`、legacy smokeの5ファイル、Sequential Loader、MoCoは変更していない。

## specの裁量内で決めた実装詳細

- **pooler**: 新しいLoRA付きEncoderに限り`add_pooling_layer=False`とし、`output_loading_info=True`の読み込みレポートを必須にした。`missing_keys`・`mismatched_keys`・`error_msgs`のどれかが空でなければ停止する。実checkpointでは、`missing`は空、`unexpected`は`classifier.{weight,bias}`だけだった。poolerを残すと、`pooler.dense.*`がcheckpointになくランダム初期化になる（CLSには使われない）
- **前処理**: checkpointの`AutoImageProcessor`（ViTImageProcessor、224×224、bilinear、mean/std 0.5）をそのままDatasetのtransformに使う。augmentationは使わない
- **`data.download: false`**: spec記載のconfigにこのkeyを追加した。既定値では自動ダウンロードせず、`Dataset not found`で停止する（実際の起動で確認済み）
- **`lora.enabled=false`**: Encoderは通常のfrozen CLS encoderとして動く。Stage Bのentrypointでは「Headだけを学習する状態」を許可せず、`ValueError`で停止する
- **DataLoader**: 固定seedのsubset（`randperm`）。既存のCIFAR-10 train loaderに合わせて`shuffle=True, drop_last=True`とし、shuffleにもseed付きgeneratorを使う。`num_workers=0`
- **保存**: Hydraのrun dirに`vit_lora_cifar10_adapter_head.pt`として保存する。中身はLoRA A/B、head、checkpoint_id、target/rank/alpha/dropout、seed、subset indices、前処理条件、step数。再ロードは「checkpointからViTを読み込む → 同じ設定でLoRAを注入 → adapterとheadを読み込む」の順。CLSとlogitsが一致することを確認し、さらに「headだけ読み込んだ状態ではlogitsが一致しない」ことも確認する
- **Comet**: 既存の`configure_logger`を使い、`comet_enabled=false`（既定）では無効化する

## 検証結果（すべてCPU、2026-10-13）

### 新規テスト: 73件すべてPASS

| specの項目 | 確認方法 | 結果 |
| --- | --- | --- |
| A1 shape・初期B=0・Baseのbit-exact一致 | `LoRALinear`の単体テスト、tiny ViTで全7組合せ | PASS |
| A2 全7組合せの構築・forward、対象外にLoRAが付かない | tiny ViTで全7組合せ。実ViT-Bでmodule数`12×n`・parameter数`147456×n` | PASS。Q/Vは24 module / 48 tensor / 294,912 |
| A3 LoRA OFFとON（B=0）のCLS一致 | 同じ重みのコピーをeval・同一入力で比較（tiny 7組、実ViT-B 7組）。既存`ViTFrameEncoder`と`enabled=false`のCLSも一致 | PASS。Bを非ゼロにするとCLSが変わることも確認 |
| A4 1 step目にBへ勾配・Aの勾配は0 | engineering用のMSE(CLS, 固定target)とAdamW（tinyと実ViT-BのQ/V） | PASS。1 step目はBだけが変化 |
| A5 2 step目にAへ勾配 | 同上 | PASS。Aが変化 |
| A6 Base凍結・勾配なし・bit-exactで不変、optimizerのparameter集合が一致 | 同上 | PASS |
| A7 fail-fast | 空・重複・未知・文字列のtarget、rank/alpha/dropoutの不正値、欠損・非Linear・shape不一致・層数不一致、二重注入、missing_keys | PASS（いずれも変更前に停止） |
| B経路（偽データ） | 偽CIFAR-10（train=True、download=False）、1層・768次元のViT、3 step | PASS。Baseは不変、LoRA Bとheadは更新、保存・再ロードでlogits一致、head-only読み込みでは不一致、同じseedでloss列が再現 |
| Hydra preset | spec値との一致、`lora.target_modules`のoverride、`--cfg job`ではdownloadも学習も起こらない | PASS |

### 既存テストの非回帰

変更前と変更後で、`CUDA_VISIBLE_DEVICES=""`を付けて`pytest test`全体を実行した。

- 変更前: 93 passed / 196 failed / 18 errors
- 変更後: 166 passed（既存93＋新規73）/ 196 failed / 18 errors
- 失敗・エラーになったテストIDの集合は変更前と**完全に一致**した。失敗はすべて既存の`assert torch.cuda.is_available()`（CUDAなし）と、`video_folder`のデータ前提によるもの

flake8で出た指摘は命名規則プラグイン（FNE004/FNE007）だけで、既存のテストにも同じ種類の指摘がある。

## 未実施・別の許可が必要なもの

- Stage B（CIFAR-10の実画像64枚・batch 16・20 step）: 未実行。CIFAR-10を`./downloaded_data`などに置くか、`data.download=true`でダウンロードし、さらに実行の許可を得る必要がある
- GPUでの実行、本格的な学習、commit/push: 未実施
- specのstatusは`approved`のままにしている（Stage Bの実データrunが終わっていないため）

実行例（許可後）:

```bash
python -m scripts.smoke.smoke_vit_lora_cifar10                         # 既存のCIFAR-10を使う
python -m scripts.smoke.smoke_vit_lora_cifar10 data.download=true       # ダウンロードを許可する場合
python -m scripts.smoke.smoke_vit_lora_cifar10 'lora.target_modules=[q,k,v]'
```

Stage Aの結果は、LoRAが正しく挿入・凍結・初期化・更新されることを工学的に示すだけである。動画・時系列の表現獲得、MoCoの正しさ、性能の優劣の根拠にはならない。
