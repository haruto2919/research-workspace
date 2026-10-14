---
date: 2026-10-14
project: sequential-video-lora-analysis
status: approved
kind: implementation
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_baseline_commit: cbc303c05502e914798e25717c4a9b29e98c8285
source_spec: 2026-10-13-standalone-vit-lora-cifar10-verification-spec.md
---

# Standalone ViT-LoRA 教授報告用Evidence自動出力 spec

## 1. 目的とAuthority

- 2026-10-08 MTGの「独立したLoRA注入・勾配・更新・Base凍結の検証」方針に対応し、CIFAR-10の20-step smokeから教授に検証根拠を示すartifactを同じHydra runディレクトリへ自動保存する。
- ユーザーは2026-10-14に次の4出力を明示して「これを行いたい」と採用した：`report.json`、`loss.png`、`lora_grad_norm.png`、`verification_summary.md`。
- [承認済みの元spec](2026-10-13-standalone-vit-lora-cifar10-verification-spec.md)の学習設定、学習対象、Stage A/Bの意味は一切変更しない。
- 関連する[探索メモ](../../../../secretary/notes/brainstorm/2026-10-14-cifar10-vit-lora-verification-evidence-recording.md)は候補・解釈の出典であり、このspecで固定した要求を上書きしない。

## 2. 現行コード確認

- GitHub `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev` は閲覧時 `cbc303c05502e914798e25717c4a9b29e98c8285`。
- `scripts/smoke/smoke_vit_lora_cifar10.py` は `report.json` (`history`にloss/train_accuracy/adapter_grad_norm_sum/head_grad_norm_sum）、`resolved_config.yaml`、`adapter_head.pt` を同一Hydra出力に保存。
- 既にBase全重みの学習前後 `torch.equal`、Base勾配なし、adapter更新、Head更新、checkpoint roundtripを実行中に検査するが、Base判定・A/B別内訳・reload数値は永続化不足。
- `requirements.txt`に`matplotlib`が存在する。Hydra出力形式は `log/lora_standalone/<date>/<time>/`。GitHubに存在しないSSHサーバ側のローカルdirty版は読めない。GitHub版に対する差分の適用は、ローカルHEAD/dirty確認後に行う。

## 3. 採用する最小変更

### 3.1 維持するもの

- データ：CIFAR-10 train、seed0、64枚、224x224変換。学習：batch16、20-step、Q/V rank8 alpha8 dropout0、AdamW LR1e-3 WD0、LoRA A/B + classifier Head trainable、Base ViT frozen。既存configのoverrideと出力先を維持する。
- 既存の `report.json` フィールド、`adapter_head.pt`のschema、`resolved_config.yaml`を保持。Cometの設定は変更しない。
- Base ViTの全checkpoint保存、Sequential Loader/MoCo統合、データダウンロード、長時間run、完全な学習性能評価、GitHubへの重いartifactのコミットは対象外。

### 3.2 出力契約

```text
log/lora_standalone/<date>/<time>/
├── adapter_head.pt                # 既存
├── resolved_config.yaml           # 既存
├── report.json                    # 既存 + 監査項目追加
├── loss.png                       # 新規
├── lora_grad_norm.png             # 新規
└── verification_summary.md        # 新規
```

- `report.json` の `history` に各stepの `lora_a_grad_norm_sum`/`lora_b_grad_norm_sum` を追加。既存の `adapter_grad_norm_sum` は互換保持。
- 追加の検証欄は、測定した `base_frozen_requires_grad`、全stepでの `base_gradient_tensors_max=0`、学習前後の `base_changed_tensors=0`、`optimizer_matches_trainable_parameters`、`lora_a_changed_tensors`/`lora_b_changed_tensors`/`head_changed_tensors`、A/B/Head別update norm、checkpoint eval出力の `reload_logits_max_abs_diff`、roundtripチェック結果。
- LoRA enabled=true、2-step以上の場合はAとBそれぞれ一つ以上の変更を期待。B=0初期化の1-stepではAの変更がなくても既存短時間契約を壊さず `NOT_RUN` として区別。LoRA disabledはA/B更新を `NOT_RUN` とする。
- Baseの全parameter（元のViT由来）を `requires_grad=False`、勾配tensor数0、全パラメータ値のbit-exact一致で検証し、期待するtrainable集合とoptimizerのparam ID集合一致をassertする。B=0初期のA grad=0自体は正常。
- `loss.png`: 実際の `history` のstep対mini-batch CE loss。縦軸Loss、横軸Step（各stepのbatch差異を明示）。
- `lora_grad_norm.png`: step対A/B別勾配norm、凡例、ステップ1のA=0をそのまま描画。値の欠落をゼロで代用しない。
- `verification_summary.md`: run設定（checkpoint / targets / rank / alpha / seed / subset / device / steps / Git HEAD+dirty）、測定結果の表、PASS/FAIL/NOT_RUN、学習とBase凍結の結論、loss低下だけではLoRA寄与を証明しない旨、artifact一覧と元JSON参照を記載。数値はreportから生成し、無条件PASSをハードコードしない。
- コード由来の報告とactual runを分ける。失敗時・中止時は存在しない測定をPASSにしない（fail-fast例外では完了レポートは作成しない）。

## 4. 非対象・実行境界

- 学習前後に固定同一64枚での精度/CE測定、LoRA ON/OFF推論比較、head-only baseline、未知画像や複数seed評価、torch/hub外部依存先のtrust判断は対象外。
- checkpointや画像をGitHubへpushしない。Research Workspaceには、実行後の実測値を別途 `experiments/` に証拠付きで記録する。
- 実装用のコード差分を作ることと、SSHローカルcloneで学習runすること、コードrepositoryへcommit/pushすることはそれぞれ別。今回の要求から長時間runやリモートpushの許可を推定しない。

## 5. 成功条件

1. 既存の20-step smoke成功時、同一Hydra runディレクトリに6ファイル（既存3＋新規3）が作成され、JSON/PNG/Markdownが読み取れる。
2. `history`のloss・A/B勾配は入力logと一致。グラフのx軸に全stepが入り、A=0のケースでゼロが正しく表示される。
3. Base変更・勾配・requires_grad・optimizer集合が測定値で検査され、A/B/Headの更新情報が区別される。ゼロ初期化Bと1-step境界・LoRA disabledの`NOT_RUN`をtest。
4. checkpoint save/loadの差分値と合否がreport/Markdownで一致する。
5. 既存 `test/model/test_vit_lora_checkpoint.py` 等のAPI互換を維持し、新規レポートの単体試験をPASS。GPU依存168件は別Gateで、ここでの新機能の必須検証ではない。
6. 実際の64枚学習runやGPU環境の稼働結果は、ユーザーまたは実行環境からの実行ログなしに「実行済み」と主張しない。

## 6. Gate / Handoff

- Blocking ambiguity: なし（出力ファイル名・既存保存先・記録内容が現在依頼で固定され、重要な科学的挙動は変更しない）。
- Non-blocking: PNGの色・フォントやhelper名は既存styleに従う。
- Spec Gate: 本件4出力についてユーザーの明示採用・実装依頼があり、`approved` として保存。完了検証まで `implemented` にしない。
- Implementation handoff: `scripts/smoke/smoke_vit_lora_cifar10.py` の最小変更＋レポート/plot helperと小規模unit test。SSH上のdirty情報が確認できないため、提供するコード差分はGitHub `dev@cbc303c` を基準とし、ローカルへは差分確認後に適用する。コードリポジトリへのcommit/pushは別承認。
