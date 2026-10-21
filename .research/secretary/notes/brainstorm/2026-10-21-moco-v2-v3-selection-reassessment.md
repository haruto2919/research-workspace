---
date: 2026-10-21
project: sequential-video-lora-analysis
source_todo: null
topic: MoCo v2・v3選定理由とViT接続コストの再評価
status: exploratory
tags: [brainstorm, research, moco-v2, moco-v3, vit, sequential-video, lora]
---

# MoCo v2・v3選定理由とViT接続コストの再評価

## 出発点
2026-10-21、ユーザーから、ViT公式実装とAdamWがあるMoCo v3を最初から採用すればViT接続を検討せずに済んだのではないか、なぜMoCo v2を選んだのかという確認があった。本記録は探索メモであり、採用済みspecの変更、コード変更、実験実行の承認を意味しない。

## 確認済み事実（Authority区分を含む）
- 2026-09-17の探索メモ（2026-09-17-clip-vit-moco-video-lora.md）では、CLIP-ViT＋MoCo v3型の候補も検討されていた。正式な採用判断ではない。
- 2026-09-24の探索メモ（2026-09-24-stage5-moco-v3-online-suitability.md）では、MoCo v3がFIFO Queueを廃止し、同一更新の分散バッチ内Keyをnegativeに使うため、strict sequentialで1クリップ／極小バッチを処理する場合にnegative不足となり得ると整理。v2型Queueを使って過去Keyをnegative辞書に蓄積する構造をStage 5の有力候補とした。
- 同日の承認済みspec（.research/lab/projects/sequential-video-lora-analysis/specs/2026-09-24-stage5-moco-v2-lora-one-step-smoke-spec.md）では、既存のViT+LoRA+MaskedMeanへQuery/Key・Projector・EMA・Queue・InfoNCEを追加するv2-style方式を採用した。これはオリジナルのMoCo v2がViTを標準サポートしているという意味ではない。
- 2026-10-15 MTG（.research/lab/projects/sequential-video-lora-analysis/meetings/2026-10-15-mtg.md）では、統合runのloss振動・勾配急増の原因切り分けのため、MoCo単体をResNetで参照再現→ViTへ変更→LoRA→Sequential Loaderの順に検証すると決めた。10月の公式v2比較はこの段階的検証の一環。
- 2026-10-21 E5（.research/lab/projects/sequential-video-lora-analysis/experiments/2026-10-21-meta-moco-v2-e5-imagenet-5epoch-pair-parity-evidence.md）でResNet50 ImageNet-1KのMeta公式対自作5epochのbitwise指標・canonical model state一致を確認したが、ViT、LoRA、表現品質、時間情報は未検証。
- 2026-10-21のViT DDP Full FT/LoRA切替specはdraftであり、OptimizerはAdamWを採用済みだが実装・実験の全Blockingが確定したわけではない。

## Meta公式コードの確認
- [MoCo v3 builder](https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py)にはViT用projector、Query predictor、EMA Key Encoder、cross-rank all-gatherした現在バッチのKeyによる対称contrastive lossがあり、FIFO Queueはない。
- [MoCo v3 README](https://github.com/facebookresearch/moco-v3)と[CONFIG](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md)にはViT-S/ViT-BのAdamWを使うImageNet学習例がある。ViT-Sの公式例は8GPU／batch1024、ViT-Bの代表的例はbatch4096。v3のResNet系でのデフォルトOptimizerはLARSなのでv3全体がAdamW固定ではない。
- [MoCo v3 ViT](https://github.com/facebookresearch/moco-v3/blob/main/vits.py)はtimm系ViTで、研究のHugging Face pretrained ViTの重み流用・freeze・LoRA注入、動画chunk/Loaderまでは自動実装されない。

## 解釈・比較軸
A. v2型Queue維持：小バッチでも過去keyをnegativeに活用でき、将来のstrict sequential条件へ素直に対応可能。既存5epoch parity資産を活用できる。ただしViT接続・freeze/EMA・LoRAは独自の工学的検証が必要。
B. v3採用：画像ViTへの公式接続は存在するため、画像学習の初期工数を短縮できた可能性がある。ただし実工数未計測。動画のstrictオンライン小バッチではnegative不足やmini-batchの組み方の再検討が必要。
C. v2主体＋v3補助baseline：性能上の利点・ViTへの適合性を比較できるが、optimizer、predictor、negative数、batch、計算予算など交絡が多く、そのままバージョン単独の因果効果とは言えない。

Queueがあっても過去featureをnegativeとして参照するだけではreplay learningや長期記憶の保証ではない。また、framewise ViTとmasked meanのままでは時間順序の学習を保証しない。

## 現在地・未解決
- 今回の調査はv2の絶対優位を主張しない。ViTのみを素早く学習させるならv3が合理的な可能性を認める。
- 本研究の主条件がstrict 1-clip streamingか、十分な多様性を持つDDP mini-batchでもよいかでv2/v3の適否が変わる。
- v3を別系統の短時間baselineとして追加するかは保留。ユーザーから選択がない限り、現行v2 DDP ViT接続draft specや研究コードを変更しない。
- 時間情報の主張には、順序・逆順・shuffle・static-repeatなどのtemporal-awareな評価対照が必要。

## 関連資料
- 2026-09-24 Stage 5候補の比較：2026-09-24-stage5-moco-v3-online-suitability.md
- 2026-10-15 MTG：.research/lab/projects/sequential-video-lora-analysis/meetings/2026-10-15-mtg.md
- 2026-10-21 ViT DDP draft：.research/lab/projects/sequential-video-lora-analysis/specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md
