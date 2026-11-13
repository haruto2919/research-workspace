# Figure Manifest

作成日：2026-11-12。用途：[2026-11-12 MTG資料](../../../materials/2026-11-11-progress-since-1022-and-temporal-order-results-mtg-materials.md)。

## Source

- Source files:
  - [ImageNet 1,000step](../../2026-11-01-vit-moco-low-lr-l5-1000step-imagenet-gpu-evidence.md)
  - ActivityNet単一フレーム：[LoRA](../../2026-11-03-vit-moco-lora-activitynet-a2-lora-gpu-evidence.md)・[Full FT](../../2026-11-03-vit-moco-lora-activitynet-a2-full-ft-gpu-evidence.md)
  - ActivityNet 4フレームclip：[LoRA](../../2026-11-03-vit-moco-lora-activitynet-clip-a2-lora-gpu-evidence.md)・[Full FT](../../2026-11-03-vit-moco-lora-activitynet-clip-a2-full-ft-gpu-evidence.md)
- Source data: 研究リポジトリ `log/` の各runの `metrics.jsonl`
  - ImageNet：`moco_vit_lr_long/{lora,full_ft}/imagenet/lr_0.0003/`（2026-11-01）
  - 単一フレーム：`moco_vit_lr_activitynet/{lora,full_ft}/activitynet_frames/lr_0.0003/`（2026-11-03_10-48-36-408110、11-49-57-818631）
  - clip：`moco_vit_lr_activitynet_clip/{lora,full_ft}/activitynet_clips/lr_0.0003/`（2026-11-03_15-22-36-985580、16-46-46-287044）
- 生成スクリプト：[`../make_2026-11-11_mtg_figures.py`](../make_2026-11-11_mtg_figures.py)（ログを読むだけ）

## Figures

| file | type | purpose | source | note |
|---|---|---|---|---|
| `01-loss-by-stage.png` | experiment | ImageNet・ActivityNet単一フレーム・4フレームclipの3段階で、LoRAとFull FTのlossの推移を並べる | metrics.jsonl | 濃い線はstep 257以降だけで計算した50step中心移動平均。図には凡例・軸・題名だけを描き、説明は資料本文に書く。LoRAとFull FTの差は1 seedの記述で、優劣は主張しない |
