# Figure Manifest

作成日：2026-10-22。

## Source

- Material: [MTG資料](../../2026-10-22-post-mtg-progress-and-vit-moco-diagnosis-mtg-materials.md)
- Source files:
  - [E5：MoCo公式／自作5epoch比較](../../../experiments/2026-10-21-meta-moco-v2-e5-imagenet-5epoch-pair-parity-evidence.md)
  - [D4/D5：固定probe診断](../../../experiments/2026-10-22-vit-moco-query-collapse-d4-imagenet-gpu-diagnostic-evidence.md)の末尾「D4両方式工学PASSとD5記述的解析」
- Source data: 上記Markdownの報告値を抽出した下記CSV。実験の生ログ・checkpointを新たに読んだものではない。
- 作成方法：Matplotlibで静的なMTG用再構成図を生成。PNGは資料への貼付用、SVGは拡大・編集用。

## Figures

| file | type | purpose | source | note |
| --- | --- | --- | --- | --- |
| `01-moco-five-epoch-loss.png` / `.svg` | MTG用再構成図 | 公式／自作のepoch平均loss一致と低下を説明 | E5のepoch平均値5点 | 表現性能・全step曲線を示さない |
| `02-fixed-probe-feature-diagnosis.png` / `.svg` | MTG用再構成図 | CLS／Projector／Queryのどこで方向集中が見られるかを比較 | D4/D5のrank0・固定64画像 | 因果関係・方式間性能差を示さない |

## 抽出データと表示条件

### 図1

- データ：[01-moco-five-epoch-summary.csv](01-moco-five-epoch-summary.csv)。公式・自作は同じ報告値。
- 元のepoch index 0〜4を、図では完了epoch 1〜5と表示。
- 縦軸：epoch平均InfoNCE loss、無次元。
- 青線はMeta公式の5点、オレンジの白抜き丸は自作の同じ5点。線は点間の補助線であり、stepごとの曲線を補間したものではない。
- 条件：ImageNet-1K、ResNet50、SGD LR 0.03、5epoch cosine、global batch 256、2GPU、FP32、seed 0。
- 下流の表現品質は未評価。

### 図2

- データ：[02-fixed-probe-summary.csv](02-fixed-probe-summary.csv)。全24行（2方式×3段階×4時点）を保持。
- 横軸：optimizer更新0/1/2/10回のカテゴリ。同じ幅で配置しており、時間の連続軸ではない。線は観測点間の補助線。
- 縦軸：①CLSと②Projector生出力は異なる画像ペアのcosine中央値、③正規化Queryは次元ごとの画像間sample stdの平均（対数軸）。いずれも無次元。
- 条件：同じImageNet train画像64枚、決定的transform、evalモード、rank0。AdamW LR 0.03、global batch 256、2GPU、FP32、seed 0。通常の学習バッチの更新前forwardとは異なる測定。
- 元表の`approx 1.0`は数値1.0として描画し、近似・丸め値であることを図とCSVのフラグで明示した。浮動小数点の厳密な1との一致は主張しない。
- Full FT・Query・更新10回のcosine p50は、元表の警告レコードからの高精度値をCSVに保持。図のcosineパネルはProjector生出力の報告値を使う。
- 工学Gateは両方式PASSだが、低LR対照による原因検証と表現品質評価は未実施。

## Evidence boundary

実験結果そのものの新しい正本図は作成していない。本フォルダの図は、既存の実験Evidenceに記録された集計値を説明用に再構成したもの。元の実験記録・README・spec・TODOは変更していない。
