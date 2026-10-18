# A1 InfoNCE演算診断：実装静的照合（2026-10-18）

## 調査範囲とソース
- 対象研究: sequential-video-lora-analysis。
- 実装正本: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev`、GitHub取得時HEAD `13a962b3beffa15807be2e13472aee3e6f734d09`。研究サーバ上のローカルHEAD・dirty状態は未確認。
- 仕様: `.research/lab/projects/sequential-video-lora-analysis/specs/2026-10-18-meta-moco-a1-infonce-arithmetic-diagnostic-spec.md`。GitHub取得時のfrontmatterは`status: draft`、本文にも未承認と記載。実装済みと仕様承認済みは別。
- 対象ファイル: `self_supervised/moco/ddp_moco.py`、`self_supervised/moco/standalone_moco.py`、`scripts/parity/moco_imagenet_parity.py`、`scripts/parity/compare_moco_parity.py`、`scripts/parity/judge_moco_a1_info_nce.py`、`conf/moco_imagenet_parity.yaml`、`test/model/test_moco_parity_harness.py`。

## 確認済み（コードに直接現れる事項）
- `DistributedMoCo(info_nce_impl="self")`は既定時に元の`moco_logits`、`meta_einsum`指定時に別helperを選ぶ。別helperはMeta固定commitと同じ`einsum`2式・cat・inplace `logits /= temperature`を使う。Queueは既存のsnapshot経由。
- `StandaloneMoCo.moco_logits`のGitHub blobは以前の実装と同一。CIFAR用共有式は直接改変されていない。
- `validate_info_nce_diagnostic`は`meta_einsum`をSelf・A1・1step・synthetic・K64・local batch4・world size2・image64・full trace step1・resume/checkpointなし・Comet OFFに制限する。
- `resolved_config.yaml`、run report、provenance、summaryにmodeを保存する経路がある。
- `compare_moco_parity.py`に`a1_info_nce`制御pairとrun間契約、grad/SGD差の集約指標、差の初出判定がある。
- `judge_moco_a1_info_nce.py`はM対S0・M対S1・S0対S1・S0再実行・S1再実行の5報告から有効性、再現性、4判定カテゴリを出す。
- `test/model/test_moco_parity_harness.py`には切替制限、fixture、trace、比較契約、判定器の追加テストがある。

## 未確認・実行境界
- GitHub上のcommit statusと該当コミットのPR系workflow runは空。CPUテストPASSやGPU実行完了を示すものではない。
- GitHubから研究サーバのGPU実行・ローカルworktree・fixture/log実体は確認できない。
- 次はローカルHEAD/clean状態確認後に `python -m pytest -q test/model/test_moco_standalone.py test/model/test_moco_cifar10_reporting.py test/model/test_moco_ddp.py test/model/test_moco_parity_harness.py`。skipがあれば総件数・理由を区別する。
- 1step GPU診断は仕様第7節により別の明示許可が必要。承認後にfresh S0/S1と各rerunを同じfixtureで実施し、旧Mをprovenanceが合えば再利用。比較5本を専用judgeへ入力する。
- `rtol=1e-5, atol=1e-6`は暫定であり、診断結果からA1 PASSや長時間実験の許可を推定しない。Meta公式本体・元Spec・README・テスト・研究コードは今回の静的監査では編集しない。

## 推奨する次の判断
研究サーバの最新HEADがGitHub `dev@13a962b3...`と一致するか確認し、CPUテスト結果を採取する。仕様`draft`の承認状態を整理し、2GPU短時間診断の実行を別承認の上で行う。
