# Figure Manifest

作成日：2026-11-12。用途：[2026-11-12 MTG資料](../../../materials/2026-11-11-progress-since-1022-and-temporal-order-results-mtg-materials.md)。

## Source

- Source files:
  - [学習率0.03での診断（10/22）](../../2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md)
  - [学習率3e-4、10step](../../2026-10-31-vit-moco-low-lr-l4-imagenet-gpu-evidence.md)・[学習率3e-4、1,000step](../../2026-11-01-vit-moco-low-lr-l5-1000step-imagenet-gpu-evidence.md)
- Source data: 研究リポジトリ `log/` の `probe_metrics.jsonl`（rank 0、`offdiag_cosine_p50`）
  - 0.03 LoRA：`moco_vit_query_collapse/lora/imagenet/2026-10-22_04-02-22-441192`
  - 0.03 Full FT：`moco_vit_query_collapse/full_ft/imagenet/2026-10-22_04-20-32-327570`
  - 3e-4 LoRA：`moco_vit_lr_long/lora/imagenet/lr_0.0003/2026-11-01_00-11-06-779221`
  - 3e-4 Full FT：`moco_vit_lr_long/full_ft/imagenet/lr_0.0003/2026-11-01_09-37-00-940051`
- 生成スクリプト：[`../make_2026-11-11_mtg_figures.py`](../make_2026-11-11_mtg_figures.py)（ログを読むだけ）

## Figures

| file | type | purpose | source | note |
|---|---|---|---|---|
| `01-probe-cosine-by-lr.png` | experiment | 学習率0.03と3e-4で、固定の確認用画像64枚どうしのcos（中央値）の推移を、MoCoの出力（`z_q`）とViTの出力（CLS）で比べる | probe_metrics.jsonl | 3e-4は1,000step runを使用（step 0〜10は10step runとほぼ同じだが、GPU backwardが完全には再現しないため厳密には一致しない）。0.03は10stepまで。図には凡例・軸・題名だけを描き、説明は資料本文に書く |
