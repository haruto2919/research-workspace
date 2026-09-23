# Figure Manifest

## Source

- Material: [2026-09-24 MTG資料](../../2026-09-24-moco-progress-and-evaluation-mtg-materials.md)
- Source files:
  - [Stage 6A spec](../../../specs/2026-09-24-stage6a-moco-multistep-canary-spec.md)
  - [Stage 6A実装・短時間検証記録](../../../experiments/2026-09-24-stage6a-moco-multistep-verification.md)
- Source data: 上記specの4-stream処理・MoCo更新契約と、検証記録の実ActivityNet CPU 10-step / fresh 100-step結果。

## Figures

| file | type | purpose | source | note |
|---|---|---|---|---|
| [01-stage6a-flow.png](01-stage6a-flow.png) | 概念図 | 4動画のKey-only warm-up、chronological round-robin、各stepの更新順を説明 | Stage 6A spec・検証記録。SVGからPNGを生成 | Notion貼付用。表現性能や時間情報獲得の結果は示さない |
| [01-stage6a-flow.svg](01-stage6a-flow.svg) | 編集用原図 | PNG図の編集・再生成 | Stage 6A spec・検証記録 | 実験正本図ではない |
