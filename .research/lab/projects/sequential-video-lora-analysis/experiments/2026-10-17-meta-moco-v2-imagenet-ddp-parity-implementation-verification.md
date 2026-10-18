---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: in-progress
created: 2026-10-17
last_updated: 2026-10-17
spec: ../specs/2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md
implementation_branch: dev
implementation_base_commit: 8939d7fca7d1cc3367b437c5f6adb48dd8e86f4a
reference: facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
reference_clone: /mnt/HDD12TB-1/ishikawa/facebookresearch-moco
---

# Spec A：Meta MoCo v2 / 自作MoCo ImageNet・2GPU DDP数値一致harnessの実装とCPU検証

2026-10-17のユーザー依頼「このspecに従って実装してください」に基づき、[Spec A](../specs/2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md)の実装対象（公式固定commitの独立clone、非侵襲的計測、自作ResNet50/ImageNet DDP＋Shuffle-BN、canonical fixture＋parity comparator、CIFAR非回帰）を実装した。
**GPUは使っていない。** CPU（gloo 2-process）で自作側のDDP数学テストとharnessのend-to-endテストを行い、Gate A0のうちGPU以外の項目（clone・import・ImageNet・環境・非回帰pytest）を実行した。**A0のNCCL確認、A1〜A5のGPU runは別途の実行許可が必要なため未実施**で、公式と自作の数値一致はまだ測定していない。
研究コードの変更は未コミットで、commit・push・Comet送信・データのdownload/移動は行っていない。

## 環境と基準

- 実装root: `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`、branch `dev`、開始時HEAD `8939d7f`（`origin/dev`と一致、clean）
- Python 3.12.3、torch 2.14.0+cu130、torchvision 0.29.0+cu130、comet_ml 3.58.6、hydra-core 1.3.7、NCCL 2.30.7
- GPU: RTX A4500×2（各20,470 MiB）。今回は`CUDA_VISIBLE_DEVICES=""`でCPUのみ使用
- 公式clone: `/mnt/HDD12TB-1/ishikawa/facebookresearch-moco`（研究コードrepoの外）。`a419ef2`をdetached checkout、`git status`・`git diff a419ef2`とも空のまま維持

## 公式固定commitの確認（B1）

- `main_moco.py`は`deeplearning.cross_image_ssl.moco.builder / loader`をimportする（specの懸念どおり）。研究repo側の`scripts/parity/meta_moco_shim/`（cloneの外）に**import aliasだけのshim**を置き、cloneの未改変`moco/builder.py`・`moco/loader.py`へ解決した。**公式sourceへのpatchは不要**で、追加承認が必要なimport patchは発生していない
- 実際に読み込んだfileはcloneの3 file（sha256: `main_moco.py` b6e08ba0…、`moco/builder.py` d57b5787…、`moco/loader.py` 4125d09e…）。import時にbytecodeを書かないようにし、cloneに`__pycache__`を残さない
- ライセンス表記: `LICENSE`と`builder.py`/`main_moco.py`ヘッダはMIT、`README.md`のLicense節はCC-BY-NC 4.0。食い違いはpreflightに記録した。公式コードは研究repoへ転載していない
- 公式の実装上の事実: `_batch_shuffle_ddp`とlabelsが`.cuda()`固定のため、公式側はGPUでしか実行できない（CPUテスト不可）。`torch.randperm`はCPU global RNGから引かれ、rank0の値がbroadcastされる

## 変更ファイル（既存ファイルの変更は`self_supervised/moco/__init__.py`のexport追加3行のみ）

| ファイル | 役割 |
| --- | --- |
| `self_supervised/moco/ddp_moco.py` | `ImageNetResNet50`（torchvision `resnet50(weights=None)`、7×7 stem・max-pool、fc除去、2048-d）、`concat_all_gather`、`DistributedMoCo(StandaloneMoCo)`（Shuffle-BN、全rank keyのrank順enqueue、全rank一斉の入力・finite検査） |
| `conf/moco_imagenet_parity.yaml` | 独立Hydra設定（global64=32×2、K65536、lr0.0075、momentum0.9、wd1e-4、m0.999、T0.2、dim128、MLP、FP32・TF32 off、seed0、data root、暫定許容誤差） |
| `scripts/parity/meta_moco_shim/` | B1のimport shim（公式sourceなし） |
| `scripts/parity/moco_parity_state.py` | 公式import・`main_moco.parser`でのMoCo v2 flags解析、**明示state mapping表**、seed導出、hash、tensor比較 |
| `scripts/parity/build_moco_parity_fixture.py` | canonical fixture（公式`builder.MoCo`の初期state＋入力定義／ImageNet実画像の固定2-view） |
| `scripts/parity/moco_imagenet_parity.py` | 2-rank DDP runner（`implementation`: meta_reference / self、`run_kind`: parity / bench）、phase trace、checkpoint＋sidecar、Comet、artifact |
| `scripts/parity/compare_moco_parity.py` | 2 runの比較器 → `parity_report.json` / `verification_summary.md` |
| `scripts/parity/preflight_moco_imagenet.py` | Gate A0（clone・ライセンス・import・ImageNet inventory/decode・GPU/NCCL・環境・非回帰pytest） |
| `test/model/test_moco_ddp.py` | 自作DDP MoCoのgloo 2-rankテスト：6件 |
| `test/model/test_moco_parity_harness.py` | config・mapping・比較器・fixture・runner・resume・Cometのテスト：18件 |

## specの裁量内で決めた実装詳細

- **自作DDP MoCo**: 既存`StandaloneMoCo`を継承し、CIFAR経路のfileは変更しない。1 stepの順序は公式と同じ「query → Key EMA（更新前θq）→ key画像のglobal gather → rank0が`torch.randperm`で引きbroadcastしたpermutationで自rank分を取り出す → key forward → normalize → unshuffle → enqueue前Queueでlogits → 全rankのkeyをrank順にenqueue」。SyncBatchNormは使わない。DDPは公式と同じ既定値（`broadcast_buffers=True`）
- **片rankだけの異常で他rankが集団通信待ちにならないよう**、入力検査・finite検査は全rankで結果を共有し、どこか1 rankで失敗すれば全rankで例外にする
- **公式側の実行**: cloneは改変せず、`main_moco.parser`・`adjust_learning_rate`・`save_checkpoint`を呼ぶ。model構築・DDP wrap・SGD（`model.parameters()`）・CrossEntropyLoss・step順序は`main_worker`/`train`をharness側で再記述（`accuracy`は観測専用のため省略）
- **同一入力・状態**: 両実装は同じfixture fileから初期state（646 tensor、自作へは明示mapping表で投入）と2-viewを読む。各forward直前に全rankが**CPU global RNGだけ**をfixture由来seed（rankごとに異なる値）で設定し、rank0のpermutationがbroadcastされることを各rankで完全一致検証する。入力は毎step・rankごとにsha256を記録し、比較器で完全一致を要求する
- **観測**: module hookで「DDP buffer broadcast後」「query後」「Key EMA後」「Key forward後」「logits・enqueue後」を、harnessで「forward前」「backward後」「SGD後」を記録。hookはtensorをcloneするだけでRNGに触れない。公式のq・kは内部変数のため、qはquery head出力のnormalize、kはenqueue後Queue列から導出し、両実装で同じ導出を使う
- **保存量**: 全stepでz_q・z_k・q・k・enqueue列・positive logits・loss・index類と、各tensor群のfloat64統計を保存。`trace.full_steps`のstepだけparameter・grad・momentum・BN buffer・Queue・logitsの全値を保存（ResNet50で1 rank・1 stepあたり約0.5〜0.9 GB）
- **run内check**（両実装共通、16項目）: Shuffle-BN、buffer broadcast、EMA式、enqueue列＝Key forwardの再導出、旧Queueでのlogits、Queue列・ptr、targets、CE、Keyに勾配なし、DDP all-reduce後の勾配一致、SGD式（lr・momentum・wd）、全rank一致、finite。式のcheckは`|実値−float64参照| ≤ atol + rtol×(被演算項の絶対値和)`（float32丸め誤差の上界の形）とした。結果値基準ではmomentumの打ち消し合いで誤判定することをCPUテストで確認したため
- **LR**: 公式はepoch単位cosine。Spec Aの全stepはepoch 0内なので両実装ともlr=0.0075。自作の式は公式と同じ浮動小数の結合順（`lr0 × (0.5 × (1+cos))`）
- **backend**: 両実装とも`cudnn.deterministic=True`、`benchmark=False`、`torch.backends.fp32_precision="ieee"`（TF32 off）。旧`allow_tf32`は新APIと併用すると読み出し自体がerrorになるため新APIの5項目だけ記録。公式`main_worker`は`cudnn.benchmark=True`にするが、parityでは決定性を優先した（アルゴリズムの変更ではない）
- **閾値**: `rtol=1e-5, atol=1e-6`は暫定。比較器は`--threshold-status fixed`を指定したときだけPASS/FAILを出し、暫定閾値では差がなくても`BLOCKED（診断）`とする。離散値・入力hash・run内checkの失敗は閾値に関係なくFAIL
- **checkpoint（B5）**: 公式は`main_moco.save_checkpoint`の標準形式（epoch, arch, state_dict, optimizer、rank0のみ）、自作は独自形式。どちらもrankごとのsidecar（global step、RNG、fixture/config identity）を併置。rank1のBN buffer（rank0のものを読み込む。次forwardでDDPが再broadcastする）とDataLoader worker RNGは復元対象外として`checkpoint_manifest.json`に明記
- **Comet**: rank0のみ、auto loggingをすべてoff、metricはlocal artifactと同じ定義（1 step = 1 optimizer update）。Cometの例外はrunを失敗させ、local failure reportを残す
- **bench（A5）**: ImageFolder(train)＋DistributedSampler(shuffle, seed0)＋aug-plus＋pin_memory＋drop_last。公式側はcloneの`TwoCropsTransform`/`GaussianBlur`で`main_worker`と同じ変換列、自作側は独自のaug-plus。DataLoader worker seedを両実装で揃える。VRAM（allocated/reserved）、step時間、images/sec（計算のみ／data待ち込み）、data待ちを記録。OOM時は自動でbatch・K・precisionを変えず停止する

## 検証結果（すべてCPU、2026-10-17）

### 自作DDP MoCo（A1相当の自作側数学）: `test_moco_ddp.py` 6件PASS

gloo 2 rank・float64で、公式builderの意味を1 processで独立に再計算したreferenceと比較した。

- 1 step目のq・k・logits・loss・DDP平均勾配・EMA後Key・Queue・ptr・rankごとのBN running statsがreferenceと一致（rtol 1e-9）
- rank1は別seedで別permutationを引くが、rank0のbroadcastが使われる
- K16・global batch 8でptrが8→0→8と一巡し、Queueは全rankで完全一致、rank順にenqueueされ、3 step目は最古列だけ上書き。logitsの負例はenqueue前Queue
- step後のBN running statsはrank間で異なり、次forward冒頭のDDP broadcastでrank0の値に揃う
- 片rankだけのbatch不一致・NaN入力・`K % global batch ≠ 0`で、全rankが状態変更前に例外になる（hangしない）

### harness end-to-end: `test_moco_parity_harness.py` 18件PASS

公式cloneからCPUでfixture（K8・local batch 2・32px、ResNet50）を作り、`implementation=self`をgloo 2 rankで実行した。

- run内check 16項目がすべてPASS。ptrは4→0→4→0→4、Key forward後のKey BN bufferはrank1で毎stepでrank0と異なる（情報値）
- 同じfixtureの再実行同士は全比較項目がbitwise一致し、`fixed`閾値でPASS、暫定閾値では`BLOCKED`
- step3のcheckpointから別processでresumeしたstep4〜5が中断なしrunと一致（`--pair-kind resume`でPASS。rank1のforward前BN bufferだけ情報扱い）
- trace上で1要素だけ1e-3ずらすと、比較器がstep2・rank1・`after_query`・`z_q`を最初の失敗として特定
- mapping表は646 entry（自作のみの`real_key_count`を除く）で全単射
- Comet呼び出しは`Experiment(disabled=True)`でAPI互換のみ確認（送信なし）

### 非回帰

- MoCo standalone・CIFAR reporting・LoRA 4 file・新規2 fileの計8 file（最終preflight内pytest）: 98 passed
- 全test suite: 368 passed / 36 failed。失敗36件（`abn_r50`系のtorch.hub信頼済みrepo確認で停止するもの、Kinetics400動画のdataset test）は**変更前HEADのクリーンコピーでも同じ36件が失敗**し、今回の変更とは無関係

### Gate A0 preflight（GPU以外）

最終版: `log/moco_imagenet_parity/preflight/2026-10-17/16-18-57-661358/`（実装root配下、gitignore対象。`16-10-43-749279/`はコード最終化前の1回目で、pytest実行中にテストを編集したため非回帰の根拠には使わない）

| 項目 | 結果 |
| --- | --- |
| reference clone | PASS（固定commit、clean、diff空、5 fileのsha256記録） |
| reference import（shim経由、`--mlp --moco-t 0.2 --aug-plus --cos`の解析） | PASS |
| ImageNet（読み取り専用） | PASS: train 1000 class / 1,281,167枚、val 1000 class / 50,000枚（ILSVRC2012標準件数と一致）、空・読取不可classなし、非画像fileなし、decode train 256/256・val 64/64成功 |
| GPU / NCCL | NOT_RUN（2-GPU NCCL checkは別途許可） |
| 環境 | PASS |
| 非回帰pytest（MoCo standalone・CIFAR reporting・LoRA 4 file・新規2 file） | PASS: 98 passed |
| **A0全体** | **BLOCKED**（NCCL未実行のため） |

観測: 同じseedで選んだ同じJPEG（train 256・val 64枚）のdecodeは、1回目（16:10）が16 threadで約64枚/秒（約7 MiB/s）、2回目（16:18）が約970枚/秒（約107 MiB/s）だった。2回目は同じfileの再読込なので、OSのpage cacheが効いた値と考えられる（cacheの状態は直接確認していない）。train全体のlistingは約33秒。NASから初めて読むときの速度ではglobal batch 64のstepがdata読み出しで律速される可能性があるが、DataLoader worker込みの実測はA5で行う（推定で結論づけない）。

### 作成済みfixture（CPUで作成、`log/moco_imagenet_parity/fixtures/2026-10-17/`）

| fixture | 用途 | 内容 |
| --- | --- | --- |
| `a1_synthetic_k64_b4x2_64px` | A1 | K64、local batch 4×2、64px、synthetic |
| `synthetic_k65536_b32x2_224px` | A2/A3 | K65536、local batch 32×2、224px、synthetic |
| `imagenet_k65536_b32x2_224px_1step` | A2（実画像） | 同条件、epoch0のDistributedSampler順の先頭32枚×2 rank、公式aug-plusで1回だけ作成した2-view |

公式`builder.MoCo`の初期state hash（`02accfd5…`）はK65536の2 fixtureで一致した（seed 0から決定的）。

## 未実施・未確定

- **GPU run（別途許可が必要）**: A0のNCCL check、A1（小K）、A2（1 step、synthetic・実画像）、A3（1025 step、長め）、A4（GPU上のcheckpoint/resume）、A5（ImageNet 100〜300 step bench）。**公式と自作の数値一致、VRAM peak、throughputは未測定**
- **正式閾値**: 公式同士の再実行差と公式対自作の差を測ってから、判定用runより前に確定する
- **Comet**: 実送信は未実施（`logging.comet_enabled`は既定true。GPU runで初めて送信される）
- 比較図（グラフ）は未作成（比較reportは表のみ）
- 公式の`cudnn.benchmark=True`をA5 benchで再現するかは未決（既定は決定性優先のfalse）
- spec本文はstatus `draft`のまま。§13の「現状」行は今回の実装で古くなっている
