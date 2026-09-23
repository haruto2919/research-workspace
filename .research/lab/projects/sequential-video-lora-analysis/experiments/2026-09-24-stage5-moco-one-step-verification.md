---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: completed
created: 2026-09-24
last_updated: 2026-09-24
spec: ../specs/2026-09-24-stage5-moco-v2-lora-one-step-smoke-spec.md
implementation_branch: dev
implementation_base_commit: 3b980c8a90e8cad08e78a5ba13cbb3629ec5ae58
sequential_loader_branch: ActivityNet
sequential_loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
---

# Stage 5 MoCo v2-styleの実装・1-step検証

[承認済みspec](../specs/2026-09-24-stage5-moco-v2-lora-one-step-smoke-spec.md)に基づき実装し、
新規38件・既存87件のテストと実ActivityNet先頭2動画のCPU 1-step smokeが成功した。
実装と必須の短時間検証は完了。specの承認記録は変更していない。

実装root: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`。
開始時はcleanな`main`でspecの基準commitと一致。同じcommitの既存`dev`へ切り替えた。
検証終了時も上記HEADを維持し、変更は未コミット。
GPU run、長時間学習、commit、pushは実行していない。

## 実装

すべて新規追加。既存の実装ファイルは変更していない。

| File | 内容 |
|---|---|
| `model/moco/vit_lora_moco.py` | 独立Query / Key、768→768→128 Projector、L2正規化、metadata FIFO、different-sequence mask、InfoNCE、LoRA / Projector EMA |
| `model/moco/__init__.py` | `MetadataQueue`と`ViTLoRAMoCo`のexport |
| `sequential_moco_bridge.py` | Query raw / Key valid-frame width flip。同一metadataを維持。既存bridge / masked meanとKey no-grad forward |
| `smoke_activitynet_vit_lora_moco_one_step.py` | provenance、初期一致、parameter集合、gradient、update、EMA式、queue順序の監査。各single-source datasetから先頭chunkだけ取得 |
| `test/model/test_vit_lora_moco.py` | Projector、独立storage、queue境界・mask・detach、InfoNCE、EMAのunit tests |
| `test/model/test_activitynet_vit_lora_moco_one_step.py` | 実PEFTのgradient / update、view・padding、更新順序、異常系、first-chunk取得とreader解放 |

warm-upではsource AのKeyをenqueueするだけで、optimizer / EMAは呼ばない。
source BのQuery / Key forward後、更新前queueでlossを計算し、backward、AdamW step、
EMA、計算済みのdetached positive Keyのenqueueの順に進める。
optimizer対象はQuery LoRA + Query Projectorのみ。
ActivityNetのaction labelや評価annotationは参照していない。

Stage 1〜4のencoder / bridge / aggregator / smoke、50Salads、classification trainer、
Hydra config、dependency pins、独立Sequential Loaderは無変更。
loaderは検証後も指定revisionでclean。

## 検証環境とコマンド

- Python: 実装rootの`.venv/bin/python`（3.12）。
- PyTorch `2.14.0+cu130`、Transformers `5.17.0`、PEFT `0.21.0`。
- CPUのみ。`CUDA_VISIBLE_DEVICES=''`、`HF_HUB_OFFLINE=1`。
- Checkpoint: cache済みの`google/vit-base-patch16-224`、poolerなし。
- Dataset: `/mnt/NAS-TVS872XT/dataset/ActivityNet`、読み取りのみ。
- seedは追加固定せず、具体的loss値・loss減少を成功条件にしない。

実装rootで実行:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest \
  test/model/test_vit_lora_moco.py \
  test/model/test_activitynet_vit_lora_moco_one_step.py -q -o addopts=''

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest \
  test/model/test_vit_lora_frame_encoder.py \
  test/model/test_activitynet_vit_lora_one_step.py \
  test/model/test_vit_frame_encoder.py \
  test/model/test_masked_mean_clip_aggregator.py \
  test/model/test_activitynet_vit_clip_feature.py \
  test/model/test_50salads_vit_bridge.py test/config -q -o addopts=''

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python -u \
  smoke_activitynet_vit_lora_moco_one_step.py \
  /mnt/NAS-TVS872XT/dataset/ActivityNet --device cpu
```

| 検証 | 結果 |
|---|---|
| 新規unit / integration | 38 passed、29.73秒 |
| Stage 1〜4 / Hydra回帰 | 87 passed、88.50秒 |
| 実ActivityNet CPU smoke | PASS、exit code 0 |
| 新規6ファイルのAST / 末尾空白検査 | 成功 |
| `git diff --check`と既存tracked filesの差分確認 | 成功、既存ファイル無変更 |

初回の新規テストは35成功・3失敗。テスト用readerがEOFでないのに2frameだけ返し、
strict loaderの契約に違反していた。fixtureを16frame・非EOFへ直して全38件成功。
これにより各動画を最後まで走査せず先頭chunkだけで閉じることを確認した。
実装側の条件は緩和していない。回帰warning 2件は既存dependencyのSentry / Torch JIT非推奨通知。

unit testsではStage 4と同じ実Transformers ViT / PEFT fixtureを使用。
12 layers・hidden size 768を維持し、MLP intermediate sizeだけ32へ縮小。
実checkpoint全体は実データsmokeで検証した。

## 実ActivityNet smoke

| 項目 | 実測 |
|---|---|
| split / source | training、sources[0]とsources[1] |
| warm-up sequence_id / sequence_index | `---9CpRcKoU` / 0 |
| training sequence_id / sequence_index | `--0edUL8zmA` / 0 |
| 両clip frame indices / valid count | 0〜15 / 16 |
| warm-up timestamps | 0〜0.5秒、1/30秒間隔 |
| training timestamps | 0〜0.6秒、0.04秒間隔 |
| Query / Key pixels | `[16,3,224,224]` |
| Query / Key frame / clip / projected | `[16,768]` / `[768]` / `[128]`、finite |
| 初期Query / Key | 全state一致、parameter storage独立 |
| 各LoRA | Q/V計24 targets、294,912 parameters / 48 tensors |
| 各Projector | 689,024 parameters / 4 tensors |
| optimizer | Query側983,936 parameters / 52 tensors、unexpected `[]` |
| queue capacity / count | 4096 / 0→1→2 |
| source Bのvalid negative | 1、`---9CpRcKoU`のみ |
| positive similarity | 0.9924513101577759 |
| negative logit（temperature適用後） | 4.635244846343994 |
| T / loss | 0.07 / 0.00007176141662057489、finite |
| Query LoRA gradient tensors（finite / nonzero） | 48 / 24 |
| Query Projector gradient tensors（finite / nonzero） | 4 / 4 |
| Query base / Key全体のgradient tensors | 0 / 0 |
| Query base / LoRA / Projector changed tensors | 0 / 24 / 4 |
| Key base / LoRA / Projector changed tensors | 0 / 48 / 4 |
| EMA | 全対象tensorがm=0.999の期待式と一致（rtol=1e-6、atol=1e-8） |
| 全parameter / queue keyの有限性 | True / True |

changed countは`torch.equal`比較であり、Keyの変化にはfloat32のEMA演算の丸め差も含み得る。
countの大きさを学習効果として解釈しない。
load時の`classifier.weight / classifier.bias` UNEXPECTEDは分類headを使用しないため。
pooler MISSINGはなかった。

## Success criteria対応

| spec 7章 | Evidence |
|---|---|
| 1〜4 repository / branch / baseline / dependencies | preflight、実smokeのprovenance監査 |
| 5〜12 freeze / LoRA / 初期一致 / Projector / optimizer / Key無勾配 | 実PEFT tests、独立storage検査、実smoke |
| 13〜16 masked mean / normalization / two-view / label不使用 | view・padding tests、既存bridge維持、実smoke |
| 17〜23 negative policy / FIFO / warm-up / 更新前queue / InfoNCE | 容量超過・遠距離同一動画除外・negative 0拒否・更新順序tests、実smoke |
| 24〜33 gradient / update / base不変 / EMA / enqueue / finite | 実PEFT更新、EMA式照合、異常注入tests、実smoke |
| 34〜35 回帰 / 実smoke | 既存87件成功、実ActivityNet exit code 0 |
| 36〜37 scope | CPU 1-stepのみ、性能やtemporal learningの成功は主張しない |

representation性能、temporal order / motion獲得、multi-step stability、GPU memory、
sequential-vs-shuffleは未検証。Stage 6や長時間学習は起動していない。

## 検証版SHA-256

未コミットの実装版を識別する。

| File | SHA-256 |
|---|---|
| `model/moco/__init__.py` | `288e48730d04c4dd63fff60726df49908fc37b10100b2629183ff6bb3b1491a1` |
| `model/moco/vit_lora_moco.py` | `79b47d7658a5364682c46bc98fab3a85197a6dd973bc708e5319176003524508` |
| `sequential_moco_bridge.py` | `564a258e28b81828b7cf620e520075f2e0d187452383f05fd754a49243a58db7` |
| `smoke_activitynet_vit_lora_moco_one_step.py` | `61e275cfa7b126ce69e309129a24463e883956940e8061346b29632a2c66f819` |
| `test/model/test_vit_lora_moco.py` | `5be97bb19331fc4e0f17ee7aa77b5847bd4ea9a5d1d0431c47988a0e67abfc75` |
| `test/model/test_activitynet_vit_lora_moco_one_step.py` | `898765ab32921c3f46ea2c1e4eb7c0f225450ff5fefccabf82c28ab910724e27` |
