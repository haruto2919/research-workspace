---
date: 2026-09-24
project: sequential-video-lora-analysis
source_todo: null
topic: Stage 6A MoCo multi-step canary
status: exploratory
tags: [brainstorm, research, moco, lora, activitynet, multistep, round-robin, stage6a]
---

# Stage 6A MoCo multi-step canary

## 出発点

Stage 5でActivityNet 2動画を用いたViT-LoRA + MoCo v2-styleの1-step mechanicsが成立した。
Query LoRA / Projectorのgradient update、Key LoRA / ProjectorのEMA、different-sequence-only
negative、FIFO Queue、Base ViT不変まで確認済みである。

次段階ではtemporal learningやrepresentation性能をまだ評価せず、同じMoCo mechanicsを
複数step連続実行しても更新契約が破綻しないことを確認する。

## 確認済み文脈

- implementation: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`
- 現在のGitHub HEAD: `1cbaa4a0fde2fd196a36feb3eeb0074709b52cd9`
- Sequential Loader: `tamaki-lab/2026_09_ishikawa_sequential_loader@ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`
- Stage 5ではQueue negativeをdifferent `sequence_id`だけに限定する。
- 現在のSequentialDatasetは1 sourceをchunk順に最後まで処理してから次sourceへ進む。
- したがって `A1 -> B1 -> C1 -> D1 -> A2 ...` のround-robinはLoader Coreではなく
  consumer / training orchestration側で構成するのが責務上自然。
- 現在の `ViTLoRAMoCo` もoptimizer / EMA / enqueueをcaller側責務としている。

## 中心となる問い

Stage 5の学習条件を変えず、ActivityNetの複数動画streamを各動画内chronologicalのまま
round-robinで処理したとき、10〜100 stepのMoCo更新を安定して継続できるか。

## 採用方向

2026-09-24の壁打ちで次を推奨し、ユーザーが明示承認した。

- ActivityNet training splitの先頭4 sequenceを固定使用する。
- 1 sequence = 1既存Sequential Loader streamとし、Sequential Loader Coreは変更しない。
- 4 streamをconsumer側でround-robinする。
- 各sequence内のchunk順は厳密にchronologicalに維持する。
- A1 / B1 / C1 / D1はKeyだけを計算してQueueへwarm-upする。
- warm-upではoptimizer step / EMAを行わない。
- trainingはA2 -> B2 -> C2 -> D2 -> A3 ... の順で進める。
- まず10-step smoke、通過後に100-step canaryを行える同一実装にする。
- `max_steps` は実行時に指定でき、初期smokeは10を標準候補とする。
- どれか1 streamがEOFへ達した時点でiterationを終了する。目標step前のEOFではGate通過を主張しない。
- Stage 5のcheckpoint、LoRA、Projector、optimizer、K、m、T、negative policy、
  raw / horizontal-flip view、masked meanを維持する。
- Stage 6Aではloss低下を成功条件にしない。lossやsimilarityは診断として記録する。
- 毎stepでloss / similarity / valid negatives / queue / gradient / finite性を監視し、
  開始前後でBase不変・LoRA / Projector更新を確認する。
- MoCo model内部へtraining loopを入れない。
- multi-step orchestrationはconsumer / training側の責務として分離する。

## 評価軸

Stage 6Aの成功はperformanceではなくengineering stabilityで判断する。

- 全training stepでvalid different-sequence negativeが存在する。
- Query LoRA / Projectorに有限な非ゼロgradientが流れる。
- Query BaseおよびKey branchへbackward gradientが流れない。
- Query update -> Key EMA -> current detached key enqueueの順序を維持する。
- parameter / queue keyへNaN / Infが発生しない。
- Queue metadataとkeyの対応が維持される。
- 実行開始前後でQuery / Key Base ViTが不変。
- 10-step、100-stepの目標stepへ到達できる。

## 保留・Stage 6A対象外

- stochastic MoCo augmentation
- scheduler / LR tuning
- checkpoint / resume
- long/full ActivityNet training
- queue / momentum / temperature tuning
- sequence-balanced queue
- strict one-video-at-a-time streaming
- sequential vs shuffle比較
- temporal objective
- reverse / static-repeat評価
- representation性能評価
- 「LoRAが時間情報を学習した」という研究主張

## 反例・注意

4 stream round-robinはstrictな1動画ずつのonline streamingそのものではない。
Stage 6Aではcurrent negative policyを壊さずmulti-step mechanicsを確認するための
engineering baselineとして採用する。

また、Queue capacity 4096に対してStage 6Aは最大100 training step + 4 warm-upのため、
Queue eviction挙動の長時間安定性は本StageのEvidenceにはならない。

## Spec引き継ぎ

上記方針はユーザー承認済みであり、実装契約は次のspecへ昇格する。

- `.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-24-stage6a-moco-multistep-canary-spec.md`

brainstormは探索記録であり、実装時は上記approved specを正本とする。
