---
date: 2026-10-14
project: sequential-video-lora-analysis
source_todo: null
topic: cifar10-vit-lora-verification-evidence-recording
status: exploratory
tags: [brainstorm, research, lora, cifar10, reproducibility, experiment-recording]
---

# CIFAR-10 Standalone ViT-LoRA検証の証拠保存方式（探索記録）

## 出発点・中心となる問い

ユーザーは、CIFAR-10を使用したLoRA学習とViT Base重みの凍結成功を、後から研究記録として確認・説明できる状態にしたい。今回の相談は**適切な記録設計の検討**であり、新しい実験run、ログ機構の実装、実験結果の正式確定、spec変更の指示ではない。

## 文脈・根拠

- 2026-10-08 MTG決定：Sequential Loader・MoCoからLoRAを切り離し、Q/V注入・勾配・更新・Base凍結を小さなステップで検証してから統合する。
- 承認済みの [Standalone ViT-LoRA CIFAR-10検証spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-10-13-standalone-vit-lora-cifar10-verification-spec.md)：Stage Aの初期B=0・初期ON/OFF出力一致・A/B更新・全7 target組合せ・Base凍結・optimizer一致とStage BのCIFAR-10 train64枚、20 steps、LoRA＋分類Head学習・保存再読込を区別。Cometは初期OFF。
- [10月13日のコード監査](../../../lab/projects/sequential-video-lora-analysis/code-analysis/2026-10-13-standalone-vit-lora-cifar10-verification-audit.md)（ユーザー提示ローカルsmokeログ）：Q/V 24投影、LoRA 294912 params、Head7690 params、Base 85798656 params、20 steps CPU、train loss 2.502499→0.117905、Base変更0、LoRA A/B各24/24 tensors更新、再ロードlogits最大差分0。これは**ユーザーが提示した当該ローカルrunの観測**であり、GitHub devそのものを実行した実績ではない。ローカルHEAD `8531a2e`、dirty状態とGitHub devは異なる可能性がある。
- ユーザー報告済みunit25件PASS、CPUモデル52件PASS、さらにGPU1でTorch Hub依存のABN/X3Dを除外した164件については成功ドットの部分ログがあるが最終summaryとexit statusは未提出。全220件PASSではない。
- 現GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` の `scripts/smoke/smoke_vit_lora_cifar10.py` は、`report.json`・`resolved_config.yaml`・`adapter_head.pt`を書き出し、Base重みの学習前後完全一致と凍結Baseへの勾配なしを実行中assertする。一方、`report.json` はBaseの凍結判定、A/B別更新・勾配監査、full optimizer param-set一致、checkpoint再読込差分の明示的なフィールドを含まない（静的確認）。これは2026-10-13ローカルINFOログの出力形式とは異なる。

## 発散：保存方法の候補

1. **短いMarkdownのみ**：低コスト、MTGで参照しやすい。ただしloss・更新・凍結の主張とその数値の参照元が曖昧になり、実行ログやcheckpointとの照合・再現が困難。単独採用は弱い。
2. **Research Workspace Markdown＋小さな機械可読report＋artifact（最有力）**：`experiments/YYYY-MM-DD-standalone-vit-lora-cifar10-verification.md`を人間向け研究Evidence正本として保存し、run IDで`report.json`・`resolved_config.yaml`・`stdout.log`・`adapter_head.pt`を紐付ける。再現性と読解性のバランスがよい。
3. **候補2＋Comet**：将来のrun間比較、グラフ・tags・実行設定の比較には有用。ただし現在の検証specはComet OFF、現在のGitHub版smokeはComet enabledを拒否する。導入は独立設計・実装・権限確認が必要。単体の工学的検証の唯一の保存先にはしない。

## 収束：推奨するEvidence構造（**未採用の提案**）

- **Markdown** `experiments/2026-10-14-standalone-vit-lora-cifar10-verification.md`（正式な実行結果の記録日は、その実行日時も本文に別途明記）：
  - run ID、目的とevidence level（engineering validation / 64枚20step、汎化評価ではない）、specリンク
  - source repo・Git branch・HEAD・dirty（差分のsnapshot/ハッシュ）、command、Python/torch/transformers/torchvision/hydra、CPU/GPU、dataset train split・64枚indices hash・seed、checkpoint ID/revision、前処理
  - target Q/V/rank/alpha/dropout、24 projections、LoRA A/B 294912、Head7690、凍結Base 85798656。**学習対象はLoRA + Head**。
  - 検証表：全Base `requires_grad=False`、Baseへの勾配tensor数0、Baseのexact parameter changed tensors数0、optimizer集合完全一致、A/BとHeadのgradient/update、初期B=0/初期同値性、step全20で有限CE、保存再読込logits一致。項目ごとにrunログ参照とPASS/FAIL/NOT_RUNを記す。
  - unit25件・CPU subset52件・GPU subset164件の**実行と証拠レベルを別表**で区別。164件は最終summary未確認、残る56件未検証と記す。
  - 不確実性：train accuracy 1.0 / loss低下は64枚の記憶と両立し、LoRAが単独で改善を生んだ／時間情報を獲得したとの根拠にはならない。
- **機械可読artifact** `run_id/report.json`, `run_id/resolved_config.yaml`, `run_id/stdout.log`, `run_id/adapter_head.pt`。追加実装するときは `checks: { base_requires_grad_false, base_grad_count, base_changed_tensors, optimizer_param_set_equal, lora_a_changed_tensors, lora_b_changed_tensors, head_changed_tensors, lora_a/b_grad_norm, checkpoint_eval_logit_max_abs_diff }` のように数値と判定の**双方**を出す（観測していない値を埋めない）。
- データセットsubset indicesを保存してSHA-256を併記。checkpoint・report・config・source file snapshot/diffのhashを記録。大きな `.pt` バイナリはGitに直接入れず、ユーザーが利用するサーバのartifact場所に置き、参照URI/実在確認・保存先の永続性を明示。可搬性が必要なら適切なartifact storeを後日検討。
- GitHubは説明文書のSSOT、サーバの生成物は実体。GitHubに記録したことをローカルへ自動反映されたと表現しない。
- 実装変更不要の最初の対応：過去のユーザー提示ログと現存ファイルを棚卸しし、取得済み値だけを`observed`で保存、不明項目は`not recorded`。現行コードのreportにない監査項目はログ追加spec・実装の対象候補とする。

## 反例・誤解の防止

- train loss減少・accuracy 1.0だけからBase freezeは検証できない。分類Head単独でもlossは減り得る。
- Base freezeのassertが実行されたことと、その数値判定を永続ログで追跡できることは区別する。
- `requires_grad=False`だけでは全学習中の重み完全不変を直接証明しないため、bit-exact snapshot比較を保持する。
- Headだけ変化してLoRA A/Bが更新されていないケースを排除するにはA/B別の更新証拠が必要。
- local dirty treeのsmoke結果と現在GitHub devの実装を同一結果として混同しない。
- 多量のstepログやcheckpointをGitHubへ直接commitすることは、検索性・容量・セキュリティ・バージョン管理面で望ましくない。

## 未決事項（ユーザー判断または追加調査）

- 今回の目的は既存2026-10-13 runの**事後Evidence固定**を最優先するか、今後の再実行でも利用する**ログ改善実装**まで含むか。
- サーバのartifactの永続保存先とrun ID形式。現存checkpoint/logの正確なパスとハッシュが未確認。
- 現ローカル作業treeのHEAD/dirty/ソースsnapshot、run出力の有無。これを確認できない間は完全再現性を主張しない。
- Comet連携を今回のsmokeにも追加するか（現specではOFF、推奨は保留）。
- 「LoRAが更新された」より強く「分類性能改善にLoRA自体が寄与した」を示したい場合はHead-only frozen-ViT対照runが別途必要（現specの必要条件外）。

## 次アクション候補とresearch-spec handoff

1. **第一候補**：現ローカルの保存済み`report.json`/checkpoint/生ログ/設定/HEADを確認して、既存runについて`experiments/`に検証記録を作成（不明欄は未確認）。
2. **次段階候補**：全runでproof fieldsを含む`verification_report.json`、同一run IDとハッシュ、明示的PASS/FAILを出力する仕組みを別の改修specとして検討。
3. **保留**：Cometの導入はMoCo等との比較が増えた段階で別判断。実験RUN/コード編集はこのbrainstormでは行わない。

research-specへ引き継げる材料：目的＝LoRA A/B更新とFrozen ViTの不変性を機械可読・人間可読な根拠で証明、scope＝既存standalone CIFAR10 smoke出力・既存configとcheckpoint、比較＝本体のexact equality＋A/B/Headの差分＋gradient＋I/O roundtrip、制約＝既存run条件・既存モデル経路非回帰、未決＝ログの追加と実験成果物永続保存先、実装・実験の権限。


## 2026-10-14 20:38 JST 追記：LoRA学習を示すための最小十分ログ

### 問いと判断対象

ユーザーは、前段で提案した「毎stepのloss・accuracy・LoRA A/B gradient/update、Base凍結比較、CIFAR index/seed、commit/dirty、SHA-256、pytest」をすべて保持する必要があるかを再検討したい。対象は**CIFAR-10の短時間smokeでLoRAが意図通り更新され、Base ViTが凍結されていたという工学的主張の検証**。分類性能改善・表現の有用性・時間情報獲得の主張は現段階では求めない。

### 比較した記録水準

- **最小ログ：lossと学習完了のみ**。コスト低いが、分類Headだけでもlossが減り得るため、LoRA更新を証明しない。**単独では棄却**。
- **最小十分証拠：学習信号＋パラメータ更新＋Base不変＋optimizer集合＋実行条件**。工学的妥当性に必要十分な案として**最有力・提案**。実行中assertと後から読める報告をセットにする。
- **研究効果の比較まで含む詳細ログ**。head-only baseline、held-out data、複数seedが必要になるため、今の64枚20step pilotのscopeを超える。**保留**。

### 推奨する永続記録の最小セット（未採用の提案）

| タイミング | 機械可読値 | 判定の根拠 |
|---|---|---|
| 開始時1回 | LoRA対象Q/V・24投影・294912 LoRA trainable parameter、Head7690・Base frozen parameter件数、optimizer param ID/key set一致 | 本来の学習対象だけoptimizerに入った |
| 各step（計20） | `step`、`loss`、`lora_a_grad_norm`、`lora_b_grad_norm` | 実画像・教師信号に対し有限lossとA/B別勾配を記録。Bゼロ初期化ゆえ初回A勾配ゼロは正常 |
| 各stepでassertし、最終集約1回 | Base grad tensor count=0（全step）、すべてのBase `requires_grad=False` | 凍結Baseへ勾配なし |
| 開始と終了でsnapshot比較、最終1回 | `base_changed_tensors=0`（全Baseを`torch.equal`で比較）、`lora_a/b_changed_tensors` と `lora_a/b_update_norm` 各 >0 | Baseは不変、LoRA A/Bは実際に変化した |
| runごと1回 | 実行コマンド、resolved_config、dataset train split・seed・indices、checkpoint_id、torch/transformers、device、実際のgit HEAD/dirty＋dirty差分参照 | 後からどの条件とコードで実行したかを追える |
| runごと1回（強く推奨） | checkpointパス、save/reload同一入力eval logits max absolute diff、status | 保存済みLoRA＋Headの結果を再利用できる |

- Headは同時に学習されるので、`head_updated` と `head_update_norm` を最終1回確認・記録。今回の実験を「LoRA-only学習」と記載しない。
- 20ステップ全体の`loss`および勾配は小さいため毎step保存を推奨。毎stepの重みスナップショットや差分全パラメータ計算は不要。
- `accuracy` は任意診断であり、64枚を反復して得るtrain accuracy=1.0は一般化を意味しない。
- LoRA A/Bの**毎step update norm**は現目的では不要。最終のA/B別差分で十分。必要なら追加診断に使う。
- checkpoint/configのSHA-256はartifact同一性・移動後の破損検出のための再現性オプションであり、「学習した」直接の証拠ではない。
- pytestの実行コマンドとsummaryはunit/regressionの別記録へ切り離す。CIFAR-10のrunごとに全テスト実施ログを再保存しない。
- 最終集約ログは元のassertが走った証拠として `checks` と数値を保存し、エラー時にも不合格や途中停止が残る工夫を検討。成功判定を単に`true`でハードコードしない。

### 追加の反証可能性と限界

- 以上で「LoRA A/Bの一部が実際に更新され、Baseが完全不変」「短時間教師あり学習処理が成立」は示せるが、更新が分類の改善に寄与したことは示せない。
- **LoRAのforwardへの効果**まで確かめたければ、同一Base・同一学習済Head・同一入力・eval modeでLoRA ON/OFFのlogits/CLS差を測る追加診断を提案。ただしこれだけで性能改善を示すわけではない。
- **有用性の主張**にはhead-only baseline（Frozen ViT＋Head）との対照、未学習画像での評価、必要ならmulti-seed。これらは別段階。
- `requires_grad=False`だけでBaseが不変であるとは断定しない。比較前後の値で検証する。
- A/Bのaggregate gradだけだと一部moduleの無勾配を見逃すので、最後のchanged tensor countも保持し、必要時は層別診断へ掘り下げる。

### 現GitHub実装とのギャップ（閲覧時dev）

`scripts/smoke/smoke_vit_lora_cifar10.py`の`report.json`は `history` にstep loss/accuracy/`adapter_grad_norm_sum`（A+B合計）/`head_grad_norm_sum`を含み、最終`adapter_update_norm_sum`（A+B合計）/`head_update_norm_sum`、dataset indices、git commit/dirtyなどを保存する。Base freeze assert・base.grad不存在assert・`torch.equal`比較は**実装内で行うがreportに結果フィールドがない**。Optimizerは対象trainableを列挙して構築しているが、集合一致の明示的な機械可読監査フィールドはない。check point再ロードはassert closeだが、その差分値のJSON記録はない。

提案する変更はA/B別gradient、A/B別最終updateと変更tensor件数、Base変化0/勾配0/optimizer set一致の機械可読値、check point eval出力差分の永続化に限る。既存の`report.json`を拡張するか、`verification_report.json`に別保存するかは実装spec化時の非blocking設計判断。重複保管しない方法を優先。

### 次段階・Authority

これは記録範囲の**推奨案（exploratory）**。現在の `approved` CIFAR10 LoRA実装specを変更・採用・完了扱いせず、研究コードも編集しない。ユーザーが採用し、実装を要請するなら `research-spec` のGateでログ項目と成否判定を確定し `engineering-task`へhandoffする。既存の2026-10-13ログから取得できない証拠を後付けで捏造しない。
