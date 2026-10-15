---
date: 2026-10-15
project: sequential-video-lora-analysis
source_todo: null
topic: five-stage-lora-moco-research-task-plan
status: exploratory
tags: [brainstorm, research, roadmap, lora, moco, sequential-loader, analysis, gates]
---

# 5段階研究タスク案とサブタスク・Gate（検討記録）

## 出発点とAuthority
ユーザーは次の5つの上位タスクが適切か確認し、各タスクの目標（理想結果）・サブタスクを設計するよう依頼した。
1. LoRA単体で実装
2. MoCo単体で実装
3. Frozen ViT + MoCo + シーケンシャルローダ
4. Frozen ViT + MoCo + シーケンシャルローダ + LoRA
5. LoRAの解析

これはユーザーが提示した**計画案**であり正式な採用指示やTODO登録・spec作成・コード実装許可ではない。この文書も探索的な提案。10/8 MTGの「単体検証→段階統合、LoRA勾配急増とloss振動を原因切り分けるまでLinear Probe結論を保留」を優先する。

## 確認したEvidence
- [プロジェクトREADME](../../../lab/projects/sequential-video-lora-analysis/README.md)（2026-10-15）：研究目的は逐次動画学習のLoRAに獲得される静的／動的情報の解析。LoRA単体分類検証、MoCo単体GPU64/500step（工程PASS）の記録あり。表現品質や最終統合は未検証。
- [10/8 MTG](../../../lab/projects/sequential-video-lora-analysis/meetings/2026-10-08-mtg.md)：ActivityNet strict_single 16frames/chunk、Base ViT frozen、Q/V LoRA、MoCo。以前のPEFT統合ではMoCo loss振動、LoRA勾配急増。Linear Probe結果の研究解釈を保留。
- [MoCo GPU evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-10-15-standalone-moco-gpu-64step-and-gate-c-500step-evidence.md)：ResNet18、CIFAR10、LoRA・Loaderなし。500step工程チェック全PASS、表現品質未検証。特徴正負分離、collapseは別評価。
- [先行brainstorm（段階的因果切り分け）](./2026-10-15-moco-loader-lora-staged-causal-debugging.md) と [Gate C後ロードマップ](./2026-10-15-post-moco-gate-c-integration-roadmap.md)：③の内部を通常画像ViT→オフライン固定動画Chunk→Sequential Loader parityへ分割する推奨。
- GitHubコードの読取確認：`tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、`scripts/smoke/smoke_vit_lora_cifar10.py`、`scripts/smoke/smoke_moco_cifar10.py`、`model/vit/vit_lora_frame_encoder.py`、`integration/sequential_vit.py`、`model/masked_mean_clip_aggregator.py`、`self_supervised/moco/standalone_moco.py`。研究サーバのローカル変更は未確認。

## 結論（推奨、未承認）
ユーザーの**5つの親タスクを維持**する。①②は工学的な単体動作が実Evidenceに支持されているが、学習表現品質まで完了と主張しない。③を3a/3b/3c、④を4a/4b/4cに分けて一因子ずつ導入し、⑤ではLinear Probe改善と時間順序情報獲得を区別する。LoRA・MoCoの単体結果は最終動画学習の性能を保証しない。

### ① LoRA単体実装・検証（工程確認済み／必要に応じて追試）
目的：ImageNet事前学習済みViTのQ/Vに自作LoRAを注入し、Base完全凍結の下でLoRA A/Bがbackpropに参加し学習できると示す。
サブタスク：target module/rank/alphaの確認、凍結・optimizer IDs、CIFAR10短時間分類（訓練Headあり）、A/B grad/update、Base bit-exact unchanged、checkpoint復元、seed/commit/dirtyログ。
理想結果：Loss/訓練accuracy改善、A初stepゼロ（Bゼロ初期化）・後続A/B非ゼロ、Base変更0。合格最小条件は数学的・数値的更新境界で、テスト汎化や時間情報獲得は未証明。LoRAだけがLossを下げたとは言わない（分類Headも更新）。

### ② MoCo単体実装・検証（工程PASS／表現品質評価は残件）
目的：ResNet18/CIFAR10でInfoNCE、positive/current key、negative/past queue、EMA、Queue FIFO、SGD/backpropとcheckpoint再現性。
サブタスク：人工数値照合、unit checks、20/64/500step、step32完全Queue置換、Query-only gradient、Key EMA、scheduler、Loss/pos/mean/hardest cos、update、variance、正規化前norm、checkpoint復元。
理想結果：工程チェックPASSに加え、固定heldout画像群でpositive retrieval margin/rankの改善、collapse兆候なし、必要に応じlinear evaluation。500step最終positve≈0.147、hardest≈0.483なので工程PASSと表現品質PASSは区別。

### ③ Frozen ViT＋MoCo＋Sequential Loader（3サブステージが重要）
3a：通常画像CIFAR10＋`google/vit-base-patch16-224`（Base frozen、LoRA OFF、Projector trainable）、2-view→CLS768→projector128→InfoNCE/Queue/EMAを短時間検証。ResNet18はpretrain/trainable性/入力解像度が違うのでloss数値比較しない。
3b：**Loaderを追加する前**、ActivityNetの固定raw 16-frame chunk→frame CLS [16,768]→valid masked mean [768]→Projector/MoCoへ。padding/valid/timestamp/sourceの照合。短時間オフライン対照はオンライン性の研究主張に使わない。
3c：同じ入力と初期state/seed/augmentation/Queue/順序を維持し、Sequential Loader経由の出力と3bを比較する。raw frame同一、feature/loss/grad/optimizer update/EMA/Queue parityを許容誤差内で検証。Reader・stride・chunk境界・sequence切替を監査。
理想結果：Base完全不変、Projector更新、Key grad0、FIFO/EMA正しい、3b対3cが同一条件で一致。MoCoが有用表現を学習したとは断言しない。

### ④ Frozen ViT＋MoCo＋Sequential Loader＋LoRA
4a：③の固定chunk/Queue/seed/2-viewを保ち、自作Q/V LoRA（初期B=0）を有効化。ON初期出力はOFFと一致、Base frozen。Query LoRA A/B＋Projectorにgrad/update、Key LoRA＋ProjectorはEMA、Key grad0、Base完全一致。A初step勾配0は許容、2step以降のA/B更新を診断。旧PEFT LoRAの工程エビデンスを自作版へ流用しない。
4b：小規模複数動画/Chunk controlled runで数値安定性を検証。初期32stepのランダムQueue置換やQueue占有率、hard negative、Feature varianceを追跡。
4c：strict_single online順序を導入（future leakageなし、動画間reset/stream状態、negative policy: all-past含むsame-video類似のfalse negativeリスク、augment/strideのablation）。短時間→中期pilot→異常診断後full ActivityNet。過去のloss振動や勾配急増を再検証し、故障時に導入因子へ帰属できる比較を残す。
理想結果：全工程check PASS＋安定学習、LoRA A/B更新、Base不変、checkpoint復元、完全な入力provenance。Loss単調低下を必要条件にしない。有用な動作／時間特徴は次段⑤で判定。

### ⑤ LoRA解析・評価
5a：Baseline比較（Base ViT、LoRA on/off、学習前LoRA、学習後MoCo-LoRA、可能なら非逐次対照）。固定split/manifest/feature definitionでLinear Probe・retrieval、複数seed。性能向上は時間的表現の証明ではない。
5b：LoRA本体の層・Q/Vごとのdelta W norm、特異値・rank、B/A更新履歴、層別差分、CKPT間変化を解析。
5c：静的/動的情報の切り分け。動画フレーム順序反転・shuffle・間引き・静止フレームやmotion-controlled対照・時間的タスクで検証。現在のframe-wise CLS + masked meanはchunk内部フレーム順序に対し不変であり、順序理解の主張には別途順序敏感なencoder/aggregationなどが必要か検討。
5d：online継続学習としての過去能力保持・新環境適応・negative policyの影響、破滅的忘却などの診断。
5e：結果の再現性と統計処理（seeds, CI, split leakage防止, Comet比較, manifest, config/ckpt hash）を確保。VLM/動画生成への転用は後段の研究候補。
理想結果：何が獲得され、どの条件で改善し、どの条件で改善しないかが定量的に説明可能。改善しない／時間順序獲得できないという反証も研究上有意義。

## 共通の受入Gate案
1. **Engineering Gate**：shape・valid mask・全tensor有限・gradient/update/EMA/FIFO/凍結とcheckpoint再現・commit/dirty・seedの一致。
2. **Learning/Representation Gate**：固定データによる同一条件評価、pos/neg rank、varianceとeffective rank、Linear Probe等。絶対Lossやstepごとに異なるtrain batchでは判定しない。
3. **Temporal/Online Gate**：入力時刻順・future不使用、過去キューだけ、同一動画Negative方針、フレーム順序・動きへの感度、過去能力の保持。MaskedMeanの順序不変性を区別。

## 意思決定待ち
- 5つの親タスクを正式採用するか（ユーザーが構想を示した段階で未承認）。
- ③の最初のspecを「通常画像CIFAR10＋Frozen ViT＋MoCo（projector-only）」に限定するか。
- ②の固定表現品質評価を③の前提Gateにするか、独立並行のGateとするか（独立並行を推奨）。
- ④のonline input policy、negative、augmentation/stride、pilotからfullへの停止条件を正式specで確定する。
- ⑤の優先主張を「線形動作分類の改善」か「時間情報獲得」かに分け、後者には時間順序検証を必須にする。

## 次アクション候補（自動TODO追加なし）
ユーザーが5親タスクを採用した場合、③の3a〜3cの順序を基準に、まず3aを実装可能なspecに落とす。③の動画Loader parityと④のLoRA統合/strict online、⑤の解析は個別のGateに分割する。現時点でspec・TODO・研究コードには変更を加えない。
