---
date: 2026-10-21
project: sequential-video-lora-analysis
source_todo: null
topic: vit-moco-post-implementation-g3-roadmap
status: exploratory
tags: [brainstorm, vit, moco, ddp, full-ft, lora, gpu-smoke]
---

# ViT DDP MoCo Full FT / LoRA 実装後の次段階：G3から性能比較まで

## 相談の出発点と中心となる問い

ユーザーは [承認済み2026-10-21 spec](../../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md) に従い実装を終え、次に何をすべきか相談した。中心の問いは、「CPU engineering gateを通過した統合モデルを、どの順序・条件でGPU学習・非逐次Full FT対LoRA比較へ進めるか」。

## 確認済み文脈とEvidence

- Research Workspace mainの [実装・CPU検証記録](../../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-21-vit-ddp-moco-full-ft-lora-implementation-cpu-verification.md) では G0/G1/G2/G4/G5 がPASS。ActivityNet v1.3 trainingの10秒間隔、offset 5秒manifestは10,024動画・117,522フレーム、実decode確認64/64。G3は未実施。
- 研究コード `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` のGitHub HEADは `2c269915f0c5036fa2a460959fdf401d54456026`。対象は `model/vit/vit_moco_backbone.py`、`self_supervised/moco/vit_ddp_moco.py`、`scripts/vit_moco/moco_vit_ddp_smoke.py`、`conf/moco_vit_ddp_smoke.yaml`。
- Spec §12のG3とGPU版G4は別途承認が必要。現時点でGPU run、長時間学習、Comet送信の許可を推定しない。
- GPU固定レシピ：2GPU×local 128（global256）、ViT-B/16 224、FP32/AMPなし/TF32 off、AdamW LR0.03 WD0.0001、K65536/m0.999/T0.2、5epoch cosine horizon、GPU smoke 1/10/20 stepのみ。少なくとも1024 MiB headroomが必要。不足/OOM/NaN/NCCL error時はBLOCKED、条件の自動緩和をしない。
- 実装記録上のA4500約20GiB×2へのFull FT＋FP32 local128の適合は未測定。LoRAでもbackwardに必要なactivationを保持するため、VRAMが十分減るとは限らない。
- AdamW LR0.03は旧ResNet50＋SGDレシピから移植した初期値であり、pretrained ViT向けの安定性確認は未実施。SGDとAdamWでは同じLRでも更新特性が異なる。
- manifestのprovenanceは生成時点のdirty状態（旧commit）を記録。実装記録では生成後に同一内容がcommitされたと確認しており、正式比較でのprovenance整備は別判断。
- 2026-10-15 MTG決定ではResNet MoCo→ViT→ViT＋LoRA→Sequential Loaderの段階的再統合が推奨されている。

## 発散：次の進め方の候補

1. **候補A（最有力）：G3の分岐付き短時間GPU Gateから進む。** 研究サーバでGPU可視性・利用可能メモリ・dirty状態・manifest hashをpreflight。明示承認後、ImageNetで`lora`と`full_ft`を各1step実行し、両方式が通過すれば10step→20stepへ段階拡大。後にActivityNet framesでも同Gateを試す。GPUの実行順はリソース状態と実行許可に依存する。
2. 候補B：直ちに5epoch学習比較。G3未通過、VRAM・LR安定性不明のため保留。
3. 候補C：今すぐSequential Loaderを統合。因果的な故障切り分けを損なうため現段階では採らない。
4. 候補D：MoCo/Optimizer実装へ戻る。CPU Gateで既に基本動作のEvidenceがあるため、GPU failureが具体的に示されない限り優先度は低い。

## 収束：段階的な推奨手順（未承認の提案）

### 次のGate：G3 2GPU Smoke

1. **GPU実行の個別承認を得る**（既存specはG3未許可）。
2. **読み取りpreflight**：ローカル `dev` のclean/head `2c26991`、`CUDA_VISIBLE_DEVICES`、2GPUのVRAM/同時利用状況、モデルcheckpointとデータpath、ActivityNet manifestのshaを確認。GPU状態はGitHubから直接確認できない。
3. **ImageNet、モード別1step**：`CUDA_VISIBLE_DEVICES=<ids> python -m scripts.vit_moco.moco_vit_ddp_smoke model.train_mode=lora training.max_steps=1`と`full_ft`。採用したGPU recipeを変えない。
4. **停止・判断**：両modeの`run_report.json`、各`rank*_report.json`、VRAM peak/headroom、finite loss/grad、frozen-base/rank-state checks、GPU step time、stage別failureを確認。`vram_gate.status=BLOCKED`なら、プロセス終了コードが0でもPASSとみなさない。片方でもOOM/headroom不足なら条件変更を無断に試さず、改訂specとユーザー判断を待つ。
5. **1stepが両modeで健全な場合に限る**：ImageNetで10step→20step、次いで既存10秒manifestのActivityNet framesを候補に。短時間ではLoss低下や表現品質は判定しない。checkpoint/resume GPUの許容差を定めることは別Gate。

### G3後に新たなspecが必要な段階

- **非逐次の学習比較**：まずVRAM/LRに応じた候補レシピを別specで定義し、同一pretrained初期重み・データ・augmentation・Global Batch・step budgetを揃え、Full FT/LoRAそれぞれのハイパラ探索公平性と計算予算を明示する。比較は表現性能（固定Linear Probe、必要ならkNN、heldout評価）を主とし、lossだけで優劣を断定しない。十分な評価データ分離と複数seedを検討する。
- **ActivityNet類似度診断**：10秒抽出はsmoke用固定値であり、研究用最適間隔の証拠ではない。別途5/10/20/30秒候補の同一動画frame類似度・フレームカバレッジ・枚数を比較し、正式manifest間隔を選ぶ。クラス/行動区間の評価ラベルの扱いも定義する。
- **Sequential Loader**：非逐次モデル比較の後に順序・動画境界・resume一致・未来漏洩・反復可否を独立Gateで確認し、同一サンプル集合で順序のみを変える比較へ。frame-wise CLS＋順序不変平均だけでは時間順序の学習成功を主張しない。

## 判断基準、反例、未検証点

- 最有力：**G3への明示承認 → ImageNet 1step両mode → PASSなら10/20step → 実動画フレーム入力 → 別specで本学習/評価**。
- 反例：LoRAが1step成功してもFull FTのFP32×batch128はOOMし得る。両方通ってもAdamW LR0.03により20step以降で不安定になる可能性がある。lossの相対比較は学習品質・時間情報獲得の証明ではない。
- 未確認：研究サーバの現在のGPU状態、現地dirty change、manifest実物の現状態、ImageNet/ActivityNet実データI/O、G3実測、GPU OOM後に採用する比較設計。
- このメモはexploratoryであり、GPU実行許可・spec変更・TODO追加ではない。G3に着手する前のユーザー判断は「GPU smokeを別途許可するか」。Resource Gateを通れない場合は、別レシピの研究spec化判断が必要。
