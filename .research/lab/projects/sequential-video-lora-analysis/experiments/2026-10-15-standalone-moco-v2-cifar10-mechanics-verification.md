---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: in-progress
created: 2026-10-15
last_updated: 2026-10-15
spec: ../specs/2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md
implementation_branch: dev
implementation_base_commit: 7dcd194f439c7e28d796e2986857ab0df1982cbe
reference: facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
---

# Standalone MoCo v2型 CIFAR-10 mechanics の実装・Gate A/B/D検証

承認済み[spec](../specs/2026-10-15-standalone-moco-v2-cifar10-mechanics-verification-spec.md)に従い、実装repositoryの`dev`へ、LoRA・ViT・Sequential Loader・動画を使わない独立MoCo v2型の実装を**新規ファイルのみ**で追加した。
**Gate A（CPU unit）、Gate B（実CIFAR-10・20 step・CPU smoke）、Gate D（checkpoint復元・resume・非回帰）はPASS。Gate C（500 step pilot）は別途の実行許可が必要なため未実行。**
変更は未コミットで、commit・push・GPU使用・データのダウンロードは行っていない。

## 環境と基準

- 実装root: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`
- 開始時: branch `dev`、HEAD `7dcd194`（specの基準commitと一致）。tracked/untrackedの変更なし。`self_supervised/moco/`は旧実装の`__pycache__`だけでtracked fileはなかった
- Python 3.12.3、torch 2.14.0+cu130、torchvision 0.29.0+cu130、hydra-core 1.3.7、numpy 2.5.2、matplotlib 3.11.2
- CIFAR-10: 既定の`./downloaded_data`は存在しないため、既存cache `/mnt/NAS-TVS872XT/dataset/CIFAR10`（`cifar-10-batches-py`、過去のLoRA runと同じroot）を`data.root`で指定した
- すべてCPUで実行した（`CUDA_VISIBLE_DEVICES=""`、`training.device=cpu`）
- 公式参照: `facebookresearch/moco@a419ef2`の`moco/builder.py`（CC-BY-NC 4.0）。コードは複製せず同じ処理順で書き直し、出典をdocstringに記載した

## 変更ファイル（すべて新規。既存ファイルの変更なし）

| ファイル | 役割 |
| --- | --- |
| `self_supervised/moco/standalone_moco.py` | `CifarResNet18`（3x3 stem、max-poolなし、`weights=None`、512-d）、MLP projector、`StandaloneMoCo`（Q/K、EMA、FIFO Queue、InfoNCE、fail-fast検査）、`moco_logits` |
| `self_supervised/moco/__init__.py` | export |
| `conf/moco_standalone.yaml` | 独立Hydra設定（K1024・m0.999・T0.2・batch32・seed0・SGD lr0.00375・WD1e-4・20 step・scheduler none・CPU・Comet OFF） |
| `scripts/smoke/smoke_moco_cifar10.py` | Hydra entrypoint。2-view CIFAR-10、SGD、cosine LR、checkpoint保存・復元・resume、失敗時の記録 |
| `scripts/smoke/moco_verification_reporting.py` | `report.json`の測定値からPASS/FAIL/NOT_RUN・図4枚・`verification_summary.md`を生成 |
| `test/model/test_moco_standalone.py` | Gate A（A01–A11）と同seedの再現性：23件 |
| `test/model/test_moco_cifar10_reporting.py` | 2-view・偽CIFAR-10の20 step・artifacts・resume・checkpoint・失敗記録：19件 |

## specの裁量内で決めた実装詳細

- **1 stepの順序**: `forward`内で「query forward → Key EMA（更新前のθq）→ key forward → enqueue前のQueue snapshot（`clone().detach()`）でlogits → enqueue」。backwardとSGDは呼び出し側で行う。公式`builder.py`と同じ構成
- **EMA**: `param_k.copy_(param_k * m + param_q * (1 - m))`。公式と同じ式で、Key parameterのstorageは保持する。BN bufferは対象外
- **Queue**: `randn(128, K)`を列ごとにL2正規化して登録し、`queue_ptr`と`real_key_count`もbufferとしてcheckpointに含める。forwardとenqueueの前に、形状・finite・列のunit norm・pointer範囲・`real_key_count`との整合を確認し、不整合なら例外にする
- **RandomResizedCropのratio**: torchvisionの既定値・公式と同じ`(3/4, 4/3)`を使った（spec表記は`1.3333333`）
- **2-view**: `TwoCropsTransform`（同じ画像に同じtransformを2回独立に適用）と、`(im_q, im_k, index)`だけを返すwrapper。ラベルはwrapperで捨てるため学習ループに届かない
- **データ順序**: `torch.randperm(50000, Generator(seed0))`の先頭から連続して消費する。DataLoaderには専用generatorを渡し、DataLoaderのseed生成がglobal RNG（augmentation用）を消費しないようにした。これでresume後の乱数列が中断しなかった場合と一致する
- **run種別**: `gate_b_smoke20`（20 step・scheduler none）と`gate_c_pilot500`（500 step・cosine）は、承認済み設定・fresh run・50,000枚のときだけ付く。それ以外は`custom`とし、Gate B/CはNOT_RUNと表示する
- **勾配・更新量のnorm**: Query Backbone / Projectorごとの大域L2 norm（`clip_grad_norm_`と同じ定義）。LoRA報告の「tensorごとのnormの和」とは定義が違う
- **feature variance**: 正規化後のq/kについて次元ごとのbatch分散（`correction=0`）を求め、その平均と最小を記録する
- **EMA監査**: 毎stepのEMA前のKey/Queryを複製し、float64で独立に計算した式との最大絶対誤差を記録する（許容誤差 rtol 1e-5 / atol 1e-6）。違反、Keyへの勾配、非有限値はfail-fastにした
- **checkpoint**: tensorはすべてCPUへ移し、`weights_only=True`で読める形式にした。中身はmodel（Q/K・BN buffer・Queue・counter）、optimizer、scheduler、global step、sampler位置・順序hash、python/numpy/torch（CUDA使用時はCUDA）RNG、config、history。失敗runのcheckpointは`resumable: false`とし、resumeを拒否する
- **resume**: `training.resume`を明示したときだけ有効。新しいrun dirに書き出す。`training.resume`・`training.device`・（scheduler noneの場合だけ）`training.max_steps`以外のconfigがcheckpointと違えば停止する
- **既存artifactの保護**: 出力先にartifact名のファイルが既にあれば`FileExistsError`で停止する
- **Comet**: `logging.comet_enabled=true`は`ValueError`で拒否する（opt-inは未実装）。ローカルartifactを正本とする
- **診断値**: 勾配急増・正規化前normの縮小・variance崩壊については、first/last/min/max/median/max÷medianの記述統計だけを出す。自動の閾値判定はしない

## 検証結果（すべてCPU、2026-10-15）

### Gate A: `test_moco_standalone.py` 23件すべてPASS

| specの項目 | 確認内容 | 結果 |
| --- | --- | --- |
| Encoder構成 | `resnet18(weights=None)`の呼び出し、conv1 3x3/s1/p1、maxpool・fcがIdentity、出力`[B,512]`、projector 512→512→128 | PASS |
| A01 Q/K初期 | tiny/ResNet18の両方で、parameter・BN bufferが一致し、storageは非共有。Keyは全て`requires_grad=False`。optimizerのparameter ID集合がQueryと一致し、Keyを含まない | PASS |
| 中間表現 | q/kの形状・unit norm・finite、qには`grad_fn`あり、kにはなし、z_q/z_kは正規化前 | PASS |
| A03 logits | float64の人工q/k/Queueで、Pythonの手計算による内積/Tと一致（rel 1e-12）。targetは全て0 | PASS |
| A04 InfoNCE | float64でlog-sum-expの手計算と一致（rtol 1e-8 / atol 1e-10）、`cross_entropy`とも一致 | PASS |
| A05 初期Queue | K=1024列すべてunit norm（atol 1e-6）・finite、ptr 0・real 0・random 1024。同seedで一致し別seedで異なる。初回stepからlogitsが`[32,1025]` | PASS |
| A06 FIFO | K8/B4のwraparoundで最古の列を上書き。K1024/B32で33回enqueueし、ptr・real_key_count・20回目の(640, 640, 384)・32回目の(0, 1024, 0)を確認 | PASS |
| A07 leakage | 当stepのlogitsはenqueue前のQueueと一致し、当stepのkはQueueに入るがlogitsには混入しない。enqueueしないreference計算とloss・Query勾配が一致し、autogradのversion errorも出ない | PASS |
| A08 EMA | float64の独立計算`m·θk+(1−m)·θq`と一致。Query・Key BN bufferは不変、BN weight/biasはEMA対象。イベント順は`query_forward, ema, key_forward, logits, enqueue, backward, optimizer_step`。EMAは更新前のθqを使い、初回EMAではKeyの値が変わらない | PASS |
| A09 勾配・更新 | 3 stepでKey勾配tensor数0、Query勾配はfiniteで非ゼロを含み、Queryが更新された | PASS |
| A10 BatchNorm | Key BN bufferは「EMA後のKeyでim_kをforward」したreferenceとbit-exactに一致。`no_grad`下でも更新され、Queryからのコピーではない | PASS |
| A11 異常入力 | shape不一致・4次元以外・dtype違い・NaN入力・Kの約数でないbatchは状態を変える前に停止。ptr・real_key_count・Queueのnorm・NaNの破損、keyの次元/dtype/正規化/NaN、構築時の不正なdim/K/m/T、feature_dimの不一致、内部の非有限値（enqueueは起きない）も検出 | PASS |
| 再現性 | 同seed・CPUの4 stepでloss列とstate_dict（Key EMA・Queue）がbit-exactに一致 | PASS |

**mutation確認**: core実装へspec違反を一時的に入れ、Gate Aのテストが落ちることを確かめた（確認後は元のファイルへ戻し、`cmp`で一致を確認）。

| 入れた違反 | 失敗したテスト数 |
| --- | ---: |
| EMAをKey forwardの後へ移す | 2 |
| enqueueをlogitsの前へ移す | 3 |
| Queue snapshotを使わない（clone除去） | 3（autogradのversion error） |
| 初期Queueをゼロにする | 12 |
| EMA時にBN bufferもコピーする | 2 |
| EMAのmと1−mを入れ替える | 2 |

### Runner・reporting: `test_moco_cifar10_reporting.py` 19件すべてPASS

偽CIFAR-10（640枚のランダムPIL画像）を使い、ダウンロードなしで確認した。

- Hydraでのconfig合成、augmentationのspec一致（crop・color jitter・gray・blur kernel 3・flip・ImageNet normalize）
- A02: 同じindexの画像からtransformを2回独立に引き、2つのviewは異なる。返り値は`(im_q, im_k, index)`だけで、batchのindexがsampler順と一致する
- 既定設定（ResNet18・B32・K1024）の20 step: artifact 8種、全step・全metricの記録、ptr/real/random = 640/640/384、全checkがPASS、checkpointの中身（CPU tensor・graphなし）、summaryと`report.json`の一致。データが640枚なので`custom`となり、Gate BはNOT_RUN表示になる
- K32/B8の6 stepでrandom keyが残り0になる遷移、cosine LRが`lr0·0.5·(1+cos(πt/T))`と一致すること、`run_kind`の判定
- Gate D: 6 step連続と「3 step → resume → 6 step」でhistory・model・optimizer・sampler・torch RNGがbit-exactに一致。同じ3 stepを2回実行しても一致。別初期値のinstanceへcheckpointを読み込むと、state・optimizer・eval featureが一致し、次stepの記録・loss・更新後stateも一致
- 失敗記録: step 3にNaN入力を入れるとstep 3（train）で`FloatingPointError`になり、2 step分のhistory・FAIL/NOT_RUN・summary・`resumable: false`のcheckpointが残る。データがなければダウンロードせずdata段階で失敗し、Gate BはFAIL。CUDAがない環境で`device=cuda`を指定すると明示的なエラー、`auto`は拒否
- 既存artifactの上書き拒否、Comet・Kの約数でないbatch・不正なscheduler・1 epochを超えるstep数の拒否、configを変えたresumeの拒否
- 500 stepを模したreportで、表の抜粋・Gate Cの表示・記述統計が出ること

### Gate B: 実CIFAR-10・20 step smoke（PASS）

```bash
CUDA_VISIBLE_DEVICES="" python -m scripts.smoke.smoke_moco_cifar10 \
    data.root=/mnt/NAS-TVS872XT/dataset/CIFAR10 training.device=cpu
```

- run dir: `log/moco_standalone/2026-10-15/01-47-30-253616/`（実装repository内、Git管理外）。CPUで5.6秒
- run kind `gate_b_smoke20`、status `completed`。Gate B **PASS**、Gate D（run内のcheckpoint roundtrip）**PASS**
- dataset: CIFAR-10 train 50,000枚、消費640枚（順序hash `62d58f14…`）、download false
- provenance: HEAD `7dcd194`、dirty true（上記の新規7ファイルだけがuntracked）。初期Query state hash `1e450933…`、初期Queue hash `cbd069a7…`

| 検査 | 測定値 | 判定 |
| --- | --- | --- |
| 形状（全step） | im_q/im_k `[32,3,32,32]`、q/k `[32,128]`、logits `[32,1025]` | PASS |
| optimizer update | 20 / 20 | PASS |
| Key勾配 | 勾配を持つKey tensor数の最大 0 | PASS |
| EMA | max abs error 7.86e-08（許容 rtol 1e-5 / atol 1e-6） | PASS |
| Query更新 | Backbone 60/60 tensors、Projector 4/4 tensorsが変化 | PASS |
| Queue | ptr 640 / real 640 / random 384（期待値と一致）、全列finite、列normの最大誤差 4.2e-07 | PASS |
| データ・target | indexがsampler順と一致、target全0 | PASS |
| checkpoint roundtrip | state・optimizer・Queue/counterが一致、eval featureの最大差 0 | PASS |
| BN buffer | Query/Keyのrunning buffer 40/40がstep 20で異なる（独立に更新） | 観測値 |

**観測値（解釈は限定的）**:

| 系列 | step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: |
| InfoNCE loss | 2.527 | 5.648 | 6.392 |
| positive cosine | 0.912 | 0.916 | 0.892 |
| mean negative cosine | −0.004 | 0.251 | 0.523 |
| hardest negative cosine | 0.275 | 0.954 | 0.944 |
| Backbone grad norm | 2.92 | 1.79 | 1.96 |
| Projector grad norm | 2.35 | 0.82 | 0.81 |
| min ‖z_q‖（正規化前） | 1.61 | 2.06 | 2.14 |

- 20 stepでlossは2.53→6.39と上がった（ln 1025 ≈ 6.93）。mean negative cosineは、logits計算時（enqueue前）のQueue内の実画像keyの割合（step tで(t−1)·32/1024、step 20で608/1024）にほぼ比例して上がった（mean negative cosine÷割合はstep 10で0.89、step 20で0.88）。「ほぼ直交するrandom keyが、ランダム初期化encoderの互いに似た実画像keyへ置き換わり、negativeが難しくなる」という説明と整合するが、原因の切り分けは行っていない
- 勾配normはstep 1が最大で、急増は見られない（Backbone max÷median 1.51）。正規化前normの縮小や次元別varianceの崩壊も20 stepの範囲では見られない
- これは20 stepかつ固定LRの工学的検証であり、Lossの減少、表現品質、CIFAR-10の分類精度、時間情報の獲得は示さない。Lossの上昇も、random Queueから始める方式で予想される初期過渡の範囲として記録するにとどめる

### Gate D: 非回帰

- spec指定のLoRA/ViT回帰テスト（`test_lora_linear.py`、`test_vit_lora_frame_encoder.py`、`test_vit_lora_reporting.py`、`test_vit_lora_checkpoint.py`）: 変更前32件PASS、変更後も32件PASS。新規42件と同時に実行して74件PASS
- `pytest test`全体（`CUDA_VISIBLE_DEVICES=""`）の変更前後比較。「変更前」は新規7ファイルを一時的に退避した状態（HEAD `7dcd194`、`git status`は空）で実行し、実行後に戻してsha256の一致を確認した
  - 変更前: 125 passed / 196 failed / 18 errors
  - 変更後: 167 passed（既存125＋新規42）/ 196 failed / 18 errors
  - 失敗・エラーになったテストID（214件）の集合は変更前と**完全に一致**した。うち210件は既存テストの`assert torch.cuda.is_available()`（CUDAを隠しているため）で、残り4件は`test_video_folder.py`のデータ前提による。既存ファイルが新規moduleをimportしていないことも確認した
- flake8の指摘は命名規則プラグイン（FNE003/004/005/007/008）と、公式に合わせた引数名`K`/`T`（N803）だけ

## 未実施・別の許可が必要なもの

- **Gate C（500 step pilot）**: 未実行。実行には別途許可が必要。コマンド（fresh run・cosine LR）:

  ```bash
  python -m scripts.smoke.smoke_moco_cifar10 data.root=/mnt/NAS-TVS872XT/dataset/CIFAR10 \
      training.max_steps=500 training.scheduler=cosine            # CPU
  # GPUを使う場合は CUDA_VISIBLE_DEVICES=<id> を外部で指定し、training.device=cuda を明示する
  ```

- GPUでの実行、commit/push: 未実施
- Comet opt-in: 未実装（`comet_enabled=true`は拒否する）
- specのstatusは`approved`のまま（Gate Cの実データrunが未実施のため）

Gate A/B/Dの結果は、MoCoのQueue・EMA・logits・enqueueの順序と保存・復元が、specと公式参照の通りに動くことを工学的に示すだけである。表現学習の性能、2026-10-08に見られたloss振動・Query LoRA勾配急増の原因、時間情報の獲得については根拠にならない。
