# Figure Manifest

作成日：2026-11-11。用途：[2026-11-12 MTG資料](../../../materials/2026-11-11-progress-since-1022-and-temporal-order-results-mtg-materials.md)。

## Source

- Source files:
  - [逐次学習LoRAを加えた並び順評価](../../2026-11-11-temporal-order-sensitivity-evaluation-g2-gpu-evidence.md)
  - [5モデルの並び順評価](../../2026-11-04-temporal-order-sensitivity-evaluation-r1-gpu-evidence.md)
- Source data: 研究リポジトリ `log/temporal_order_eval/2026-11-11_10-36-56/summary.json`（`results.<model>.<condition>.cls.all.mean`）
- 生成スクリプト：[`../make_2026-11-11_mtg_figures.py`](../make_2026-11-11_mtg_figures.py)（ログを読むだけ）

## Figures

| file | type | purpose | source | note |
|---|---|---|---|---|
| `01-order-change-cls.png` | experiment | 並び順を変えたときのCLSの変化（1−cos）を6モデル（学習前・単一フレーム・clipランダム・逐次の各LoRA／Full FT）で比べ、同じ動画の10秒後のclipとの変化と並べる | summary.json | 対数軸。図には凡例・軸・題名だけを描き、説明は資料本文に書く。評価時のモデル名（summary.jsonのkey）は A、B-LoRA、B-FT、C-LoRA、C-FT、G2-LoRA。95%区間は点より小さいので描いていない |
