---
date: 2026-09-23
project: sequential-video-lora-analysis
source_todo: null
topic: argparseからHydra YAML設定への移行方針
status: exploratory
tags: [brainstorm, research, configuration, hydra, yaml, reproducibility]
---

# argparseからHydra YAML設定への移行方針

## 現在確認した実装

- `args/arg_parse.py` がdataset/model/video/training/optimizer/GPU/logging/checkpointを1つのflatな `argparse.Namespace` にまとめている。
- `main.py` と `main_pl.py` は `ArgParse.get()` を入口にし、各componentへ同じNamespaceを渡している。
- `model/model_config.py` にはすでにdataclassの `ModelConfig` があり、設定を構造化する方向との相性はよい。
- `requirements.txt` には現時点でHydraは入っていない。

## 研究上これから増える設定

- dataset: ActivityNet / 50Salads等。
- frame/chunk/sampling条件。
- ViT / CLIP-ViT等のbackbone。
- LoRA target/rank/alpha/dropout。
- MoCo variant、momentum、temperature、queue/projector。
- ordered / shuffle / reverse / static-repeat等の比較条件。
- trainer / optimizer / checkpoint / logger。

このため今後もflat argparseへ追加し続けると、実験条件の再現・比較・保存が難しくなる。

## 比較

### 現行argparseを維持

利点: 単純、依存追加なし、短いCLIでは理解しやすい。
弱点: 研究条件が増えるとflat namespaceが肥大化し、複数baselineや実験条件の再現が難しくなる。

### YAML + Hydraへ移行

利点:
- dataset/model/optimizer等をconfig groupとして分離できる。
- defaults listで実験configをcomposeできる。
- YAMLをversion controlし、実験条件をそのまま保存しやすい。
- CLI overrideを維持できる。
- 将来のparameter sweepにも展開しやすい。

弱点:
- Hydra/OmegaConfという追加概念とdependencyが増える。
- 最初から細かすぎるconfig groupやfuture optionを作ると逆に複雑になる。

## 現在の収束

この研究ではHydra + YAMLへ移行する方が有力。

理由はHydra自体が必要だからではなく、今後の研究が「dataset × backbone × LoRA × SSL × sampling × control条件」の組合せ比較になるため、設定を実験artifactとして管理できる構造の価値が大きいから。

ただし、ActivityNet clip baselineやLoRA実装と同じspecへ混ぜない。
まず設定基盤移行だけを独立したinfrastructure specとして実施し、現在の動作を保ったままargparse入口をHydraへ置き換える。

## 最初の移行specで守る範囲

- 研究アルゴリズムやdataset semanticsを変更しない。
- 現在の主要設定値をYAMLへ移す。
- `conf/config.yaml` をprimary configにする。
- choiceを持つdataset/model/optimizerは必要最小限のconfig group化を検討する。
- trainer/logging/checkpoint/video等はnested configとして整理する。
- CLI overrideを可能にする。
- 既存の主要起動条件をHydraで同等に再現できることを確認する。
- LoRA/MoCo用の未使用fieldを先回りして大量追加しない。機能実装時に追加する。
- ActivityNet integration、clip aggregation、LoRA、MoCoは対象外。

## 実装順

1. Hydra config migration。
2. ActivityNet -> ViT -> masked mean clip feature。
3. LoRA。
4. MoCo。
5. full SSL loopとcontrol比較。

このメモは探索記録であり、specまたは実装許可ではない。

## 2026-09-23 追記: specへ昇格

上記方針は次のapproved specへ昇格した。

`.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-23-hydra-config-migration-spec.md`

以降、Hydra移行の実装scope・互換性・Success Criteriaは上記specを正本とする。
本brainstormは探索経緯の記録に留める。
