# Figure Manifest

作成日：2026-11-11。用途：[2026-11-12 MTG資料](../../../materials/2026-11-11-progress-since-1022-and-temporal-order-results-mtg-materials.md)。

## Source

- Source files: [逐次学習LoRAの6,255step学習](../../2026-11-04-sequential-loader-clip-moco-lora-g2-gpu-evidence.md)
- Source data: 研究リポジトリ `log/moco_vit_seq_activitynet_clip/lora/activitynet_sequential_clips/lr_0.0003/` の `metrics.jsonl`
  - step 1〜2,000：`2026-11-05_02-08-19-443327`
  - step 2,001〜6,255：`2026-11-08_17-26-03-055153`
- 生成スクリプト：[`../make_2026-11-11_mtg_figures.py`](../make_2026-11-11_mtg_figures.py)（ログを読むだけ。step 1〜6,255が1つずつ揃うことを確認する）

## Figures

| file | type | purpose | source | note |
|---|---|---|---|---|
| `01-loss-and-cosine.png` | experiment | 動画を先頭から順に読んで学習したLoRAのlossの推移（上）と、正例・最も似た負例とのcosの推移（下） | metrics.jsonl | 移動平均はstep 257以降だけで計算した250step中心移動平均。図には凡例・軸・題名だけを描き、説明は資料本文に書く。最初と最後の値はEvidenceの250step窓（8.348・6.964）と一致 |
