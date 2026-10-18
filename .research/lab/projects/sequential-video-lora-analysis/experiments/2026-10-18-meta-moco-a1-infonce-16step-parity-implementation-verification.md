---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: in-progress
created: 2026-10-18
last_updated: 2026-10-18
spec: ../specs/2026-10-18-meta-moco-a1-infonce-16step-parity-spec.md
implementation_branch: dev
implementation_base_commit: 13a962b3beffa15807be2e13472aee3e6f734d09
reference: facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
reference_clone: /mnt/HDD12TB-1/ishikawa/facebookresearch-moco
---

# A1 InfoNCE 16step数値一致検証：実装とCPU検証

2026-10-18のユーザー依頼「このスペックに従って実装してください」に基づき、[16step parity spec](../specs/2026-10-18-meta-moco-a1-infonce-16step-parity-spec.md)の実装対象（§6）を実装し、§10のCPU検証を実行した。

**GPUは使っていない。** §11の2GPU・NCCL・16step run（M / Mb / S0 / S0b / S1 / S1b）は別途の実行許可が必要なため未実施で、§12のPASS/FAIL判定はまだ出ていない。研究コードの変更は未コミットで、commit・push・Comet送信は行っていない。specファイル自体（`status: draft`を含む）は編集していない。

## 環境と基準

- 実装root: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`、branch `dev`
- 開始時HEAD: `13a962b3beffa15807be2e13472aee3e6f734d09`（spec記載のcommitと一致、clean）
- Python 3.12.3、torch 2.14.0+cu130（`.venv`）。テストは`CUDA_VISIBLE_DEVICES=""`でCPU（gloo 2-process）のみ
- 公式clone: `a419ef2`のまま、未変更（`git status`差分なし）

## 変更したファイル

| ファイル | 内容 |
|---|---|
| `conf/moco_imagenet_parity.yaml` | `verification.threshold_status: provisional`（`provisional \| fixed`）を追加。既定は`provisional`のまま |
| `scripts/parity/moco_imagenet_parity.py` | `meta_einsum`の許可を2 profile（`META_EINSUM_PROFILES`）に拡張。既存1step（max_steps 1、full `[1]`、threshold `provisional`）と16step（max_steps 16、full `[1,8,9,16]`、threshold `fixed`かつrtol `1e-5`・atol `1e-6`と完全一致）。共通条件はself / parity / A1 / synthetic / K64 / B4 / world 2 / 64px / resumeなし / checkpointなし / Comet OFF。どちらにも一致しない場合、最も近いprofileとの差分keyを理由に挙げてspawn前に`ValueError`。`threshold_status`の値も検証。run reportの`tolerance.status`を設定値から記録し、`full_steps`を追加（run schema `v2`）。summaryにthreshold statusとfull stepsを表示 |
| `scripts/parity/compare_moco_parity.py` | pair kind `a1_info_nce_16step`を追加（S0 reference / S1 candidate、`DIAGNOSTIC`か`INVALID`のみ）。全stepで、step metrics（loss・positive/negative cosine mean・Query grad norm・Query update normをfloat比較、queue_ptrはexact）、Shuffle-BN seed、traceのstep/rankをexact比較。LRをexact比較へ変更。small tensor・metric・traced phaseの欠損（両側欠損を含む）をexact failureに。float比較にrelative L2を追加。CLIのrtol/atol/statusを両runのresolved config・run reportと照合し、A1 16step runでは不一致または`provisional`ならPASS/FAILを出さない（`BLOCKED`、16step制御pairでは`INVALID`）。step別集計`per_step`、初出への`trace_level`（full / small trace上の初出）、run summaryに条件・閾値・GPU/rank・backend flagsを追加（report schema `v2`） |
| `scripts/parity/judge_moco_a1_info_nce_16step.py`（新規） | 6 report（M-S0、M-S1、S0-S1、M-Mb、S0-S0b、S1-S1b）から§9.1の5状態と、`formal_fp32_parity`・`m_vs_s1_bitwise`・`rerun_reproducibility`（M/S0/S1別）・`s0_control_outcome`を出力。§9.3の差分局在化、full step別gradient/SGD集約、S0-S1のstep別差分伝播を保存。`judgement.json` / `judgement.md`は新規directoryにだけ書く |
| `test/model/test_moco_ddp.py` | 16step DDP数学test 2件を追加。既存`scenario_train`へ`queue_size`引数とSGD momentum記録だけを追加 |
| `test/model/test_moco_parity_harness.py` | profile・拒否・比較器・16step制御pair・judgeのtest 79件を追加 |

対象外として、`self_supervised/moco/ddp_moco.py`、`standalone_moco.py`、CIFAR経路、Meta公式clone、既存1step judge（`judge_moco_a1_info_nce.py`）は変更していない。

## 実装上の契約（spec §6〜§9の具体化）

### 16step judgeの状態と優先順位

1. `BLOCKED`: いずれかのrunが`completed`でない、または比較reportが`fixed`閾値で作られていない
2. `NOT_RUN`: 6 reportのどれかが存在しない
3. `INVALID`: comparison schema・pair kind・閾値値（1e-5 / 1e-6 / fixed）・run側記録閾値・preconditions・各runの§5.1条件（16step、full `[1,8,9,16]`、world 2、K64、B4、64px、cuda/nccl、resume/checkpointなし、Comet OFF、FP32/ieee/cudnn deterministic）・clean commit・Meta clone・within-run check・非有限値・比較不能entry（名前/phase欠損、shape/dtype差）・入力/trace/shuffleのexact差・step 1開始状態の非bitwise・全(step, rank)のsmall/discrete項目とfull step全groupの網羅（Query/Key parameter・gradient・momentumはbackboneとMLP projectorの両方）・6 runの同一性（同じrunがreport間で一致、6 directoryが別、commit・fixture・environment・CUDA_VISIBLE_DEVICES/GPU・backend flagsが全run同一）・S0-S1制御契約のいずれかに違反
4. `PASS` / `FAIL`: M対S1の全judged値がfixed規則内（exact値は完全一致）かつ3方式のrerunが全項目bitwise一致なら`PASS`、それ以外は`FAIL`

S0の結果（`DIVERGED` / `WITHIN_FIXED_TOLERANCE`）は総合statusに入れない。

### `a1_info_nce_16step`制御pair

既存`a1_info_nce`と同じ契約（mode以外のresolved config一致、commit/clean、clone、environment、GPU/rank、fixture、rank別初期state・Optimizer・全stepの入力hashとpermutation）に加え、両runがA1・16step completed・full `[1,8,9,16]`、比較stepが1〜16、step 1のlogits入力までbitwise一致、全stepでexact値（入力、trace step/rank、shuffle seed・permutation、targets、queue pointer、LR、BN counter）の不一致なし、非有限値なし、閾値が両runの記録と一致し`fixed`であることを要求する。step 2以降のfloat差は契約違反にしない。

### 既存Mの再利用

16step judgeは6 runすべてに同じclean commit・run schema `v2`・comparison schema `v2`を要求する。既存のA1 16step M（変更前commitで実行、run schema `v1`、threshold status記録なし）はこの条件を満たさないため、spec §5.3どおり正式runではfresh Mが必要になる。

## spec §10との対応と結果

| spec | 確認内容 | 結果 |
|---|---|---|
| 10.1 model/DDP数学 | Tiny backbone＋gloo 2rankで`meta_einsum` 16step（K64・global batch 8）を実行し、独立した単一processのfloat64多step参照（rank0 BN broadcast、EMA、shuffle/unshuffle、enqueue前Queueによるlogits、DDP平均gradient、SGD momentum/weight decay、rank順enqueue）と各step・各rankでq/k/logits/loss/gradient/EMA後Key/momentum/SGD後Query/Queue/ptr/forward開始時BN/step後BNを照合（rtol 1e-9, atol 1e-12）。全rank一致も確認 | PASS |
| 10.1 Queue境界 | 同runでpointer（step 8→0、9→8、16→0）、step 7まで初期random列が残りstep 8で全消失、step 9のlogitsがstep 1〜8 keyのみを使いcolumns 0〜7だけ置換、step 16のlogitsがstep 9〜15＋step 8 key、step 16後がstep 9〜16 keyのみ。`real_key_count`は診断値として64で飽和 | PASS |
| 10.2 profile・拒否 | 16step profile（fixed）を受理、self/meta_referenceの16step（fixed/provisional）と他Gateを拒否しない。15/17step、full steps違い2種、K/B/world/image、Comet ON、resume、checkpoint、A2/A3、imagenet fixture、`provisional`、rtol/atol違い、meta_reference、benchの19条件を、該当keyだけを理由に挙げて拒否。1step profileで`fixed`は拒否。不正statusを拒否。16step＋provisionalはspawn前に拒否（resolved_config.yaml以外のartifactなし） | PASS |
| 10.3 比較器（手組みtrace） | 全small・step metrics・trace step/rank・shuffle seed・LRが判定対象。fixed規則の境界内（非bitwise）/境界外、NaN、LRの1e-12差（exact）、metric欠損、両側でのphase欠損・small欠損、step/rank不一致をすべて検出。relative L2を記録 | PASS |
| 10.3 16step制御pair（CPU実run） | A1寸法・ResNet50のS0 / S1 16step run（fixed）を作成。run artifactへ閾値・full steps・16 trace/rankを記録、traceのQueue一巡・再置換（step 8/9/16）を確認。S0 vs S1は`DIAGNOSTIC`・契約valid・step 1 logits入力までbitwise、初出はstep 1 logits、step 2以降に差があっても有効。mode以外のconfig差、dirty、commit差、fixture差、environment差、CUDA_VISIBLE_DEVICES差、後続stepのpermutation差、15step、step 1 pre-logits改変、後続stepのtargets/queue pointer差、`provisional`指定、mode逆順をすべて`INVALID` | PASS |
| 10.3 16step比較の閾値・metrics（CPU実run） | S0と同内容のrerun比較はfixedで`PASS`・全項目bitwise・全step metrics比較あり。`provisional`指定、run report側の閾値不一致はPASS/FAILを出さず`BLOCKED`。step 3のloss metric改変とstep 5のmetric欠損を`FAIL`として初出（small trace上の初出）とstep別件数に局在化 | PASS |
| 10.3 judge（synthetic report） | 5状態（PASS、FAIL、BLOCKED、INVALID、NOT_RUN）。formal PASSだがM/S1非bitwiseの分離、S0の`WITHIN_FIXED_TOLERANCE`/`DIVERGED`の分離、M/S0/S1の各rerunが1項目でも非bitwiseなら`FAIL`、M/S1のfloat超過とexact出力差の`FAIL`と初出局在化、INVALID 28条件、judgeの網羅検査が比較器の実出力（S0-S1、rerun）に一致 | PASS |
| 10.3 上書き拒否 | 比較器は既存reportのあるdirectoryへ書かない。judgeは既存directoryへ書かない（新規directoryのみ） | PASS |
| 10.4 既存非回帰 | 下記 | PASS |

### 実行したpytest

```text
# 変更前: 13a962b のクリーンコピー（git archive）、同じ.venv・同じ4ファイル
120 passed in 232.73s

# 変更後: 実装root（未コミット）
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q -p no:cacheprovider \
  test/model/test_moco_standalone.py test/model/test_moco_cifar10_reporting.py \
  test/model/test_moco_ddp.py test/model/test_moco_parity_harness.py
201 passed in 677.61s   # 既存120 + 新規81、失敗・skip 0
```

変更後の1回目（201 passed in 692.66s）の実行中に文言・コメントだけを修正したため、最終コードで取り直した結果を上に記載した。最終実行の前後で変更6ファイルのsha256が一致することを確認した。所要時間の増加は、A1寸法・ResNet50の16step CPU run 2本（各約80秒、各約5GB）と16step比較（1回約40秒）を含むため。

`flake8`（`.flake8`、行長120）は、変更前から全体にある命名規約プラグイン（FNE）指摘と、変更前からある既存行の行長2件を除き、新規・変更行で指摘なし。`git diff --check`問題なし。LoRA・ViT・dataset系のテストはscope外として実行していない。

## 参考観測（CPU・判定に使わない）

テスト用CPU（gloo）S0 / S1 16step runの比較では、差の初出はstep 1の`after_logits_before_enqueue`で、logitsの入力まではbitwise一致し、step 2以降にも差が伝播した。これはCPU・合成入力・テスト用一時directory（保存していない）での観測で、GPU上のM / S1 parityのEvidenceではない。

## 未実施と次の段階

- **specの承認状態**: spec fileは`status: draft`のまま（未追跡fileで、本作業では編集していない）。実装はユーザーの「このスペックに従って実装」依頼に基づく。FP32閾値`fixed`の事前登録としてspecをapprovedにするかはユーザー判断
- **commitが先に必要**: §11.1は6 runすべてが同じclean commit Hであることを要求し、judgeもdirty・commit差を`INVALID`にする。未コミットのままのGPU runは判定に使えない
- **§11 GPU run（未許可）**: M、Mb、S0、S0b、S1、S1b。各runは2×GPU・NCCL・A1 synthetic fixture・16step・full `[1,8,9,16]`・Comet OFF・`verification.threshold_status=fixed`・別output directory
- 既存Mは再利用不可（上記）。fresh Mを作る
- §12の判定結果、A2/A3/A5、K65536、ImageNet、10epochはいずれも未実施・対象外

GPU実行が許可された場合の手順（A1寸法のoverrideは6 runで共通）:

```text
A1="gate=A1 fixture.path=<A1 synthetic fixture.pt> moco.queue_size=64 training.local_batch_size=4 \
    data.image_size=64 training.max_steps=16 trace.full_steps=[1,8,9,16] logging.comet_enabled=false \
    verification.threshold_status=fixed"
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity implementation=meta_reference $A1                      # M, Mb
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity implementation=self $A1 diagnostic.info_nce_impl=self         # S0, S0b
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity implementation=self $A1 diagnostic.info_nce_impl=meta_einsum  # S1, S1b
FIX="--threshold-status fixed --rtol 1e-5 --atol 1e-6"
python -m scripts.parity.compare_moco_parity --pair-kind implementation     --reference <M>  --candidate <S0>  $FIX
python -m scripts.parity.compare_moco_parity --pair-kind implementation     --reference <M>  --candidate <S1>  $FIX
python -m scripts.parity.compare_moco_parity --pair-kind a1_info_nce_16step --reference <S0> --candidate <S1>  $FIX
python -m scripts.parity.compare_moco_parity --pair-kind rerun              --reference <M>  --candidate <Mb>  $FIX
python -m scripts.parity.compare_moco_parity --pair-kind rerun              --reference <S0> --candidate <S0b> $FIX
python -m scripts.parity.compare_moco_parity --pair-kind rerun              --reference <S1> --candidate <S1b> $FIX
python -m scripts.parity.judge_moco_a1_info_nce_16step --m-s0 <cmp> --m-s1 <cmp> --s0-s1 <cmp> \
    --m-rerun <cmp> --s0-rerun <cmp> --s1-rerun <cmp>
```

## 更新履歴

| 日付 | 内容 |
|---|---|
| 2026-10-18 | 実装とCPU検証（変更前120 passed → 変更後201 passed）。GPU runは未実施、変更は未コミット |
