# Figure Manifest

作成日：2026-11-19。用途：[2026-11-19 MTG資料](../../../materials/2026-11-19-lora-svd-current-state-mtg-materials.md)。

## Source

- Source files:
  - [LoRA の SVD 解析：全 run の書き出し](../../2026-11-18-lora-svd-full-export.md)
  - [向きの比べ方と部分空間の比較（追補の実装と確認）](../../2026-11-19-lora-svd-direction-and-subspace-implementation-cpu-verification.md)
- Source data: 研究リポジトリ `log/analysis/lora_svd/2026-11-19_06-56-19/`（schema `lora_svd/2`、研究コード `feature-lora-svd@0aa523f`、9 run・27 checkpoint）
  - `svd_summary.csv`（図1・図2）、`svd_vectors.csv` の U（図3）
- 生成スクリプト：[`../make_2026-11-19_lora_svd_figures.py`](../make_2026-11-19_lora_svd_figures.py)（CSV を読むだけ。各 run・step で 24 行列が揃うことを確認する）

## Figures

| file | type | purpose | source | note |
|---|---|---|---|---|
| `01-size-vs-step.png` | experiment | ΔW の大きさ（W₀ との比）が step 数でほぼ決まることを示す | svd_summary.csv | 24 行列（12 層 × q・v）の中央値。step 0 は ΔW = 0 |
| `02-per-layer-ratio.png` | experiment | 層ごとの σ₁(ΔW) ÷ σ₁(W₀)（最後の step） | svd_summary.csv | 順に読む 3 run と、4 フレーム clip（シャッフル）の LoRA・Full FT |
| `03-direction-overlap.png` | experiment | run どうしで ΔW の向き（上位 8 本の U の部分空間）が似ているか | svd_vectors.csv | φ = ‖UᵢᵀUⱼ‖²_F ÷ 8、24 行列の中央値。Full FT は上位 8 本。無関係な部分空間なら約 0.01 |
| `03-direction-overlap.csv` | data | 図3 の数値 | svd_vectors.csv | 小数 4 桁 |
