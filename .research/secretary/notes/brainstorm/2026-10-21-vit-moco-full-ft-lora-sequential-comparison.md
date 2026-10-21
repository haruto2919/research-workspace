---
date: 2026-10-21
project: null
source_todo: null
topic: vit-moco-full-ft-lora-sequential-comparison
status: exploratory
tags: [brainstorm, research, vit, moco, lora, sequential]
---

# ViT Full Fine-Tuning + MoCo 対 Frozen ViT + LoRA + MoCo：逐次入力導入までの比較方針

## 出発点・研究上の問い
ユーザーはまず ViT の全重みを更新する MoCo（A）と、Frozen ViT + LoRA + MoCo（B）の性能を比較し、その後に Sequential Loader を導入する順を検討している。

中心となる問い：
1. 同一データ・同一評価条件で、LoRA は Full Fine-Tuning に対しどれほどの特徴性能を保持できるか。
2. 逐次入力によって、両手法の特徴性能・忘却・計算資源はどう変わるか。
3. 動作・時間順序に依存した情報の獲得を、静的情報依存と分けて説明できるか。

## 文脈とEvidence
- 直近の会話では公式MoCoと独自MoCoの parity、resume、benchmark の検証を継続。A3 PASS のユーザー報告と、100 step付近の query_grad_norm 低下について議論がある。
- 過去の研究文脈では ActivityNet v1.3、16 frames/chunk、ImageNet事前学習ViT、Attention Q/V LoRA rank 8、masked-mean CLS、MoCo Queue、Linear Probe などが候補。これらは今回比較の確定specではない。
- Research Workspaceの現行AGENTS.mdとbrainstorm SKILL.mdを2026-10-21に確認した。対象プロジェクトREADME・直近会議文書・直近実験本文、および研究コードの現行対象branchの内容は本相談では未検証。従って具体的な現行実装可否は断定しない。

## 発散：候補
- 候補1（推奨）：MoCo parity確定 → 同一の非逐次入力におけるFull FT vs LoRA → Sequential Loaderの入力順・再開・データ漏洩を単独検証 → Sequential Full FT vs Sequential LoRA → 動的情報の評価。
- 候補2：最初からSequential LoRAのみ導入。工数は短い可能性があるが、差の原因を切り分けにくい。
- 候補3：非逐次のFull FT vs LoRA比較だけで終了。パラメータ効率の評価はできるが、逐次学習の研究主張には届かない。

## 最有力方針：段階的実験
### Gate 0 基盤
MoCoの同一fixture比較、loss/grad、queue、EMA、resume等の完了状況を確認する。100stepの勾配norm低下が公式実装側でも再現されるか、長期性能に影響するかを調査する。

### Gate 1 単体学習
- A: ImageNet初期化ViTをFull Fine-Tuning + MoCoで学習。
- B: 同一ViT初期化、Baseを完全Frozen、Q/V LoRAのみ更新 + MoCoで学習。
- 両者ともprojectorを学習対象にし、query/key momentumの設計を一致させる。
- 固定batchのoverfitや短時間runで勾配・重み更新・特徴分散・lossの妥当性を確認。

### Gate 2 非逐次比較
- 同一事前学習初期重み・データ・split・入力単位・augment・clip/frame集約・MoCo loss/K/m/t/step数・評価器を揃える。
- optimizerやlrは同一設定一律では性能比較を歪める可能性があるため、両者で公正な探索予算を設けて最適化し、探索予算を開示する。
- 主指標：固定した下流Linear Probe（分類 accuracy/mAP等タスク適合）、補助：loss曲線、複数seed、学習時間、VRAM、総/学習可能パラメータ、処理サンプル数。
- 評価時はMoCo projector出力だけでなくViT/clip表現も比較対象とする。

### Gate 3 Sequential Loader単独検証
- 入力順序・動画境界・チャンク境界・フレーム時刻・マスク・resumeのサンプル一致を独立検証。
- 従来ローダとSequential Loaderを、同一サンプル集合・同一順序で流すparityモードをまず確立し、順序だけ変える試験へ進む。
- future leakageを防ぎ、train/testの動画単位分離、シャッフル許可範囲を仕様化。

### Gate 4 逐次条件の比較
- C: Sequential Full FT、D: Sequential LoRA。
- A対BはLoRA効果、A対CとB対Dは入力順序効果、C対Dは逐次条件下のparameter efficiencyを比較。
- online/streamingは単なるSequential Loaderと同義ではない。学習更新タイミング、過去データアクセス、queueを通じた過去特徴保持、エポック反復の可否を明示する。
- 順序ごとの時間的精度、過去区間への忘却、最新区間への適応速度、online評価、時間・VRAMを観測。

### Gate 5 動的特徴の解析
- フレームシャッフルや逆順に対する感度、動作認識ベンチマーク、時間変化を要する下流評価などを候補とする。
- mean pooled frame-wise CLSは時間順序不変であり、フレーム順序の検出能力を直接保証しない。時系列情報獲得の研究主張には時間依存アーキテクチャや適切な評価が必要。
- LoRAの特異値だけでは動的情報の獲得の立証にならず補助解析とする。

## 判断・反例・保留
- 最有力：段階的導入。既存MoCo基盤を固定して故障原因を切り分けやすく、同じ条件でモデル差とLoader差を比較できる。
- 反例：Full FTが良くても、同じ学習率ではLoRAが不利なだけかもしれない。LoRAが良くても、単に過学習を抑制している可能性がある。
- 反例：フレーム独立処理＋masked meanは順序不変のため、性能差があっても時間順序を学んだとは言えない。
- 未決：最初の比較をImageNet等の画像データにするかActivityNet clipにするか、モデル初期重み、計算予算、評価ラベル、逐次設定（何をオンライン学習と定義するか）、現行repository branchとconfig。
- このメモはexploratoryでありspec・実装承認ではない。

## specへの引き継ぎ候補
- 目的：MoCoのencoder学習方式と逐次ローダの効果を分離して比較する。
- Scope：非逐次Full FT / LoRA比較、Sequential Loader parity、逐次Full FT / LoRA比較、動的特徴解析。
- 評価：固定Linear Probe、時間順序依存指標、計算資源、複数seed。
- 次の確認：現行project README、最新spec/experiment、コード構造、比較に用いるデータ単位・入力構造・online制約。
