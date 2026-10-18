---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: in-progress
created: 2026-10-18
last_updated: 2026-10-18
spec: ../specs/2026-10-18-meta-moco-a1-infonce-arithmetic-diagnostic-spec.md
implementation_branch: dev
implementation_base_commit: c50ab79cbce1738a571daf6e018e31bb870cae24
reference: facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
reference_clone: /mnt/HDD12TB-1/ishikawa/facebookresearch-moco
---

# A1追加Spec：InfoNCE演算方式 `self` / `meta_einsum` 切替の実装とCPU検証

2026-10-18のユーザー依頼「このスペックに従って実装してください」に基づき、[A1追加Spec](../specs/2026-10-18-meta-moco-a1-infonce-arithmetic-diagnostic-spec.md)の実装対象（§3.1、§4）を実装し、§6の単体・CPUテストを実行した。

**GPUは使っていない。** §7の1step GPU検証（M / S0 / S1と同mode再実行）は別途の実行許可が必要なため未実施で、§8.2の仮説判定はまだ出ていない。研究コードの変更は未コミットで、commit・push・Comet送信は行っていない。specファイル自体（statusを含む）は編集していない。

## 環境と基準

- 実装root: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`、branch `dev`
- 開始時HEAD: `c50ab79cbce1738a571daf6e018e31bb870cae24`（spec記載のcommitと一致、`origin/dev`と一致、clean）
- Python 3.12.3、torch 2.14.0+cu130（`.venv`）。テストは`CUDA_VISIBLE_DEVICES=""`でCPU（gloo 2-process）のみ
- 公式clone: `a419ef2`のまま、未変更

## 変更したファイル

| ファイル | 内容 |
|---|---|
| `self_supervised/moco/ddp_moco.py` | `INFO_NCE_IMPLS = ("self", "meta_einsum")`、pure helper `meta_einsum_logits`（Meta固定commitの4行と同じ演算子・添字・`unsqueeze(-1)`・連結順・in-place `/=`）、`DistributedMoCo(..., info_nce_impl="self")`。不正modeは構築時`ValueError`。modeは通常attributeでstate_dictに入らない。`self`は従来通り`moco_logits`を呼ぶ |
| `conf/moco_imagenet_parity.yaml` | `diagnostic.info_nce_impl: self` を追加 |
| `scripts/parity/moco_imagenet_parity.py` | SelfAdapterがmodeを`DistributedMoCo`へ渡す。`meta_einsum`はspec §4.2の全条件（self / parity / A1 / 1step / full [1] / synthetic / K64 / B4 / world 2 / 64px / resumeなし / checkpointなし / Comet OFF）のときだけ許可し、条件外は理由付きでspawn前に拒否。読み込んだfixtureがsyntheticでない場合も拒否。modeをrun report（`diagnostic.info_nce_impl`と各rankで構築されたmodelのmode）、provenance、verification summaryへ記録。`config_identity_sha256`には入れない |
| `scripts/parity/compare_moco_parity.py` | 全pair kindで full-trace stepの`gradient_metrics`（§8.1: tensor別・全Query grad連結のmax abs、暫定規則の超過数/要素数、relative L2、両側L2 norm、after_sgdのQuery parameter・momentumも同様）と`first_difference`（最初のbitwise差）を出力。run summaryにmode・clone dirty・環境を追加。rerun/resumeはmode一致を前提条件に追加。新pair kind `a1_info_nce`（下記） |
| `scripts/parity/judge_moco_a1_info_nce.py`（新規） | M vs S0、M vs S1、S0 vs S1、S0再実行、S1再実行の5 reportから§7.2・§7.3の有効性と§8.2の判定カテゴリを機械的に出す。暫定`rtol=1e-5, atol=1e-6`以外のreportは無効扱い |
| `test/model/test_moco_ddp.py` | §6.1〜6.4（model側）のテスト10件を追加。既存`reference_first_step`へlogits式を引数で渡せるようにした（既定は従来式） |
| `test/model/test_moco_parity_harness.py` | §6.4〜6.5（config・runner・比較器・判定）のテスト44件を追加 |

対象外として、`standalone_moco.py`、`conf/moco_standalone.yaml`、CIFAR経路、Meta公式clone・公式Adapter、既存の暫定許容差は変更していない。

## `a1_info_nce` pairの契約

S0（`self`）をreference、S1（`meta_einsum`）をcandidateとして、次のいずれかがあれば`contract_valid: false`（status `INVALID`）、なければ`DIAGNOSTIC`。float差の有無でPASS/FAILは付けない。

- 両runがself・A1・完了1step・`trace.full_steps=[1]`でない、またはmodeが`self -> meta_einsum`でない、構築modelのmodeが設定と異なる
- resolved configの差が`diagnostic.info_nce_impl: 'self' != 'meta_einsum'`以外にもある
- self repositoryがdirty、commit不一致。公式cloneが固定commitでない・dirty・diffあり
- 環境（全項目）、`CUDA_VISIBLE_DEVICES`とrank別GPU、fixture、rank別の初期state hash・Optimizer・入力hash・shuffle seed/permutationのいずれかが不一致（既存の前提条件：fixture SHA、model/MoCo/precision/training条件、backend flags、主要version も適用）
- run失敗、within-run check不合格、exact比較の不一致、非有限値
- logitsの入力（`before_forward`〜`after_key_forward`の全項目、q、k、targets、logitsに使ったQueue snapshot）にbitwise差がある

## §6の対応と結果

| spec | 確認内容 | 結果 |
|---|---|---|
| 6.1 式 | 固定float32 q/k/Queue/Tで`meta_einsum`の`l_pos`・`l_neg`・logits、CE loss、`autograd.grad(loss, q)`がMetaのliteral式とbitwise一致。literal 4行がclone内`moco/builder.py`に同順で存在。shape/dtype/device/positive列/T適用の戻り値契約が`moco_logits`と同じ | PASS |
| 6.2 default非回帰 | mode省略と`self`明示で、2stepのq/k/logits/loss/全Query grad/EMA後Key/Queue/ptr/SGD後Queryがbitwise一致。dispatchは省略・`self`で`moco_logits`のみ、`meta_einsum`で`meta_einsum_logits`のみ呼ばれる。`StandaloneMoCo`・`moco_logits`のsignature、state_dict key、`conf/moco_standalone.yaml`に追加なし | PASS |
| 6.3 DDP数学 | 既存のgloo 2-rank・float64参照テストは`self`でPASSのまま。`meta_einsum`でも公式式の独立参照に対しq/k/logits/loss、DDP平均grad、EMA、Queue/ptr、SGD更新が既存tight tolerance（rtol 1e-9, atol 1e-12）内 | PASS |
| 6.4 拒否 | 不正modeを構築時に拒否。`meta_einsum`をMeta側・bench・A2・2step・full step変更・imagenet fixture・K/B/world size/画像サイズ違い・resume・checkpoint・Comet ONで、該当keyだけを理由に挙げてspawn前に拒否（13条件）。拒否時はresolved_config.yaml以外のartifactなし | PASS |
| 6.5 artifact/比較器 | CPUでA1寸法のfixture（K64/B4/64px）を作り、S0・S1・S0再実行を1stepずつ実行。modeがresolved config・run report（各rankのmodel）・provenance・summaryに残る。S0 vs S1は`DIAGNOSTIC`・契約valid・logits入力までbitwise一致。config 1項目・dirty・`z_q`改変はいずれも`INVALID`。mode逆順・同modeは`INVALID`。S0 vs S0再実行は全項目bitwise一致、異modeのrerunはBLOCKED。logits改変コピーで初出差分の記録、勾配統計の定義（分母0の扱い含む）を確認 | PASS |
| 判定ロジック | 合成reportで4カテゴリ（SUFFICIENT、PARTIAL×2、NOT SUPPORTED×3、INCONCLUSIVE×3）、無効なreport組8種、再実行非一致、CLI出力・上書き禁止 | PASS |
| 6.6 既存test | 下記 | PASS |

### 実行したpytest

```text
# 変更前: c50ab79 のクリーンコピー（git archive）で同じ4ファイル
66 passed in 134.48s

# 変更後: 実装root
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q -p no:cacheprovider \
  test/model/test_moco_standalone.py test/model/test_moco_cifar10_reporting.py \
  test/model/test_moco_ddp.py test/model/test_moco_parity_harness.py
120 passed in 233.12s   # 既存66 + 新規54、失敗0
```

`git diff --check` 問題なし。LoRA・ViT・dataset系のテストは今回のscope外として実行していない。

## 参考観測（CPU・判定に使わない）

テスト中に作ったCPU（gloo）A1寸法のS0 / S1 1step artifactを比較すると、差の初出は`step 1 rank 0 after_logits_before_enqueue/logits/logits`で、logitsの入力まではbitwise一致だった。Query gradは両rankで750要素が暫定規則を超え（全Query grad連結のrelative L2 1.92e-6）、SGD後のQuery parameterは超過0、momentumは750要素超過だった。

これはCPU・合成入力・テスト用一時directory（保存していない）での観測で、GPU上のMeta vs Self差の原因を示すEvidenceではない。§8.2の判定はGPUのM / S0 / S1でのみ行う。

## 未実施と次の段階

- **§7 GPU run（未許可）**: M、S0、S1、S0再実行、S1再実行。各runは2×GPU・NCCL・A1 synthetic fixture・1step・Comet OFF・別output directory
- **commitが先に必要**: spec §5は同一の実装commitかつ開始時cleanを要求する。`a1_info_nce`の契約とjudgeはdirtyなself repositoryを無効扱いにするため、未コミットのままのGPU runは判定に使えない
- 既存Meta run Mを再利用するかは、`--pair-kind implementation`の前提条件（fixture SHA・環境version・backend flags・Meta commit）とjudgeの有効性検査で機械判定する
- §8.2の判定結果、16step以降、正式閾値、default変更はいずれも未実施・対象外

GPU実行が許可された場合の手順（A1寸法のoverrideは各runで共通）:

```text
A1="gate=A1 fixture.path=<A1 synthetic fixture.pt> moco.queue_size=64 training.local_batch_size=4 \
    data.image_size=64 training.max_steps=1 trace.full_steps=[1] logging.comet_enabled=false"
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity implementation=self $A1 diagnostic.info_nce_impl=self         # S0, S0'
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity implementation=self $A1 diagnostic.info_nce_impl=meta_einsum  # S1, S1'
python -m scripts.parity.compare_moco_parity --pair-kind implementation --reference <M>  --candidate <S0>
python -m scripts.parity.compare_moco_parity --pair-kind implementation --reference <M>  --candidate <S1>
python -m scripts.parity.compare_moco_parity --pair-kind a1_info_nce    --reference <S0> --candidate <S1>
python -m scripts.parity.compare_moco_parity --pair-kind rerun          --reference <S0> --candidate <S0'>
python -m scripts.parity.compare_moco_parity --pair-kind rerun          --reference <S1> --candidate <S1'>
python -m scripts.parity.judge_moco_a1_info_nce --m-s0 <cmp> --m-s1 <cmp> --s0-s1 <cmp> --s0-rerun <cmp> --s1-rerun <cmp>
```

## 更新履歴

| 日付 | 内容 |
|---|---|
| 2026-10-18 | 実装とCPU検証（変更前66 passed → 変更後120 passed）。GPU runは未実施、変更は未コミット |
