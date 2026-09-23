---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: completed
created: 2026-09-24
last_updated: 2026-09-24
spec: ../specs/2026-09-24-stage6a-moco-multistep-canary-spec.md
implementation_branch: dev
implementation_base_commit: 1cbaa4a0fde2fd196a36feb3eeb0074709b52cd9
implementation_commit: 8304d033b2cf2da7e6636842ef250ed45a5693bb
sequential_loader_branch: ActivityNet
sequential_loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
---

# Stage 6A consumer側MoCo multi-step実装・短時間検証

[approved spec](../specs/2026-09-24-stage6a-moco-multistep-canary-spec.md)に従い、
4 streamのchronological round-robinとmulti-step orchestrationを実装した。
新規42件・既存125件のテストが成功。人工データで10-step / 100-stepのmechanicsを検証した。
その後、実ActivityNetで10-step smokeと別fresh processの100-step canaryをCPU実行し、
両方ともtarget stepへ到達してPASSした。これによりStage 6A specの必須Gate 1〜40を満たした。

実装repositoryは`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`。
実run時にorigin、`dev` branch、HEAD `8304d033b2cf2da7e6636842ef250ed45a5693bb`を確認し、
GitHub上の`dev` HEADとも一致した。GPU run・full ActivityNet trainingは実施していない。

## 追加ファイル

| File | 責務 |
|---|---|
| `training/__init__.py` | consumer側training package |
| `training/moco_canary.py` | 4 single-source streamのlifecycle、round-robin、Key-only warm-up、persistent AdamW、更新監査と各stepのJSON診断 |
| `smoke_activitynet_vit_lora_moco_multistep.py` | provenance、runtime dataset root、先頭4 source選択、fresh model作成、device / max_steps引数 |
| `test/model/test_moco_multistep_canary.py` | 順序・更新・Queue・EOF・例外・CLI・実PEFT接続の42テスト |

既存のStage 5 MoCo component、two-view bridge、初期state / parameter / view監査helperを再利用した。
`self_supervised/moco/`、backbones、aggregator、既存smoke、classification trainer、Hydra config、
dependency pins、Sequential Loaderには変更を加えていない。

## 動作

- pinned Adapterのtraining先頭4件を選び、sequence_idが互いに異なることを確認する。
- 各sourceに既存public APIの独立streamを作り、ExitStackでまとめてcloseする。
- `A1 / B1 / C1 / D1`はKey forwardとenqueueだけ。Queue=4、parameter不変、gradientなしを監査する。
- trainingは`A2 / B2 / C2 / D2 / A3 ...`。chunk indexとabsolute decode frame indexを検査し、raw timestampは並べ替えない。future chunkの先読みなし。
- optimizerはwarm-up後に1回だけ作り、Query LoRA + Projectorと集合が一致することを検査する。
- 各stepは既存two-view / encodeを使用し、old Queueのdifferent-sequence negativesによるloss、backward、Query optimizer、Key EMA、pre-EMA positive key enqueueの順序を守る。
- base snapshotとの完全一致、全parameter / gradient / Queueのfinite性、EMA式、Queue key / metadata対応を検査する。
- Query LoRA / Projectorのgradient norm・finite / nonzero数、base / Key gradient数、loss、similarity、negative logits、Queue count / unique ID countをstepごとに標準出力へ出す。
- 最後の有効chunkは処理し、そのstreamのEOFが判明したら全streamを終了する。target未達なら達成step数と理由を出し、例外で終了してPASSを出さない。
- max_steps既定は10。warm-up4件はstep数に含めない。Queue evictionを起こす指定は開始前に拒否する。
- 起動ごとにfresh model / optimizer / Queue / streamsを使用し、10-stepのstateを100-stepへ引き継がない。

## 検証環境・結果

Python 3.12.3、PyTorch 2.14.0+cu130、Transformers 5.17.0、PEFT 0.21.0。
実装rootの`.venv/bin/python`を使用し、CPU・offlineで実施した。
loaderは`ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`でclean。
CLIのprovenance監査とStage 6A baseline ancestry検査も成功。

| 検証 | 結果 |
|---|---|
| 新規unit / integration | 42 passed、26.75秒 |
| Stage 5 MoCo + Stage 1〜4 / 50Salads / config回帰 | 125 passed、2 warnings、107.25秒 |
| 新規Python 4ファイルのAST / 末尾空白検査 | 成功 |
| CLI `--help` | exit code 0、既定10 / device指定 / max_steps指定を確認 |
| `git diff --check` | 成功 |
| 実ActivityNet 10-step | PASS、training_steps=10、queue_count=14 |
| 実ActivityNet fresh 100-step | PASS、training_steps=100、queue_count=104 |

回帰warningは既存のSentry Hub / Torch JIT非推奨通知。
初回は異常optimizer集合の注入がテストfixtureの初期検査で止まる1件が失敗した。
注入位置をoptimizer構築時へ修正し、42件すべて成功。実装契約を緩和していない。

実装rootで実行したコマンド:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest test/model/test_moco_multistep_canary.py \
  -q -o addopts='' -p no:cacheprovider

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest \
  test/model/test_vit_lora_moco.py test/model/test_activitynet_vit_lora_moco_one_step.py \
  test/model/test_vit_lora_frame_encoder.py test/model/test_activitynet_vit_lora_one_step.py \
  test/model/test_vit_frame_encoder.py test/model/test_masked_mean_clip_aggregator.py \
  test/model/test_activitynet_vit_clip_feature.py test/model/test_50salads_vit_bridge.py \
  test/config -q -o addopts='' -p no:cacheprovider
```

## テストの範囲と限界

10-step / 100-stepテストは軽量encoderと人工Readerを使用する。
MoCoのProjector / loss / Queue / EMA、AdamW、既存public loader、two-view / encode bridge、
今回のorchestration / 更新監査は実実装を使用し、ViT固有の初期architecture監査のみテスト用へ差し替える。
更新順序をhookで観測し、warm-up中のQuery forward / backward / optimizer / EMAがないこと、
全stepのdifferent-sequence negativesが3件以上、同stepのpositiveがenqueueまでQueueへ入らないこと、
単一optimizerのstateが10 / 100 step継続すること、Queue数が14 / 104になることを確認した。

別のintegration testではStage 4の実Transformers / PEFT fixtureを使う。
12 layers・hidden size 768・Q/V計24 targetを維持し、MLP intermediate sizeのみ32。
4つの人工16-frame chunkでwarm-upした後、A2の1-stepを実行し、初期architecture監査、
Query trainable 983,936 parameters / 52 tensors、LoRA finite gradients 48を確認した。
このtestもcheckpoint downloadや実動画decodeは行わない。

EOF、target丁度でのEOF、明示的なconsumer途中停止、KeyboardInterrupt、decode / Query forward /
Key forward / loss / optimizer / EMA / enqueue例外で、開いたReader全てを解放した。
誤ったchunk順序・frame index、base / Key gradient、gradientゼロ / NaN、base更新、EMA不一致、
非finite parameter / loss / logits、negative集合不一致、Queue metadata / key破損、optimizer集合不一致、
Query更新なしを拒否することを確認した。

## Spec成功条件との対応

| spec 7章 | 状態 |
|---|---|
| 1〜4 repository / branch / baseline / loader無変更 | ローカルpreflight成功 |
| 5〜13 source選択 / lifecycle / warm-up / order / step数 | CLI tests、public loaderと人工Readerによる10 / 100-step tests成功 |
| 14〜15 Stage 5条件 / optimizer集合 | Stage 5 component無変更、既存回帰、実PEFT integrationとoptimizer監査成功 |
| 16〜28 negatives / 更新順序 / gradients / Queue / finite性 | 10 / 100-step人工データtestsと異常注入tests成功 |
| 29〜34 base不変 / Query更新 / EMA / Reader解放 | 初期snapshot・毎stepの監査、終了時検査、例外・EOF tests成功 |
| 35 実ActivityNet 10-step | PASS。target 10 steps、queue 4→14、終了監査成功 |
| 36 実ActivityNet fresh 100-step | PASS。fresh runでtarget 100 steps、queue 4→104、終了監査成功 |
| 37〜39 Stage 1〜5 / classification / 50Salads回帰 | 指定主要125件成功、既存コード無変更 |
| 40 研究性能・時間情報獲得を主張しない | engineering testのみ。研究性能は未評価 |

## 実ActivityNet run

以下を実装rootで実行した。10-step完了後、100-stepは別processで起動し、
`fresh state: True`を確認した。いずれもCPU・offlineで、GPU runは含まない。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u smoke_activitynet_vit_lora_moco_multistep.py \
  /mnt/NAS-TVS872XT/dataset/ActivityNet --device cpu --max-steps 10

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u smoke_activitynet_vit_lora_moco_multistep.py \
  /mnt/NAS-TVS872XT/dataset/ActivityNet --device cpu --max-steps 100
```

## 未コミット検証版のSHA-256

| File | SHA-256 |
|---|---|
| `training/__init__.py` | `73670a2ec2e196f71d64157e7b42148a480aca04378d9e5ecf27d7ed7ae8e933` |
| `training/moco_canary.py` | `67cec2f6c683adf97bb271ebc2ee62989b13b5f3667d14a8999c93186fc4163d` |
| `smoke_activitynet_vit_lora_moco_multistep.py` | `f7a4878cca32bb47611ceca09ba990b665042378e54da83f6c2b121ab34f9172` |
| `test/model/test_moco_multistep_canary.py` | `cfb3af8386b4f069fbd0c893915c155d9203dbfffd1cb646a6e8f74688b56f38` |


## 実ActivityNet 10-step / 100-step Evidence

共通provenance:

- implementation: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@8304d033b2cf2da7e6636842ef250ed45a5693bb`
- Sequential Loader: `ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`
- checkpoint: `google/vit-base-patch16-224`
- runtime: PyTorch `2.14.0+cu130` / Transformers `5.17.0` / PEFT `0.21.0`
- device: CPU
- ActivityNet training first four sequence IDs:
  `---9CpRcKoU`, `--0edUL8zmA`, `--mFXNrRZ5E`, `--veKG73Di4`

10-step smoke:

- fresh stateで開始。
- A1 / B1 / C1 / D1 warm-up後にQueue=4。
- training順はA2 -> B2 -> C2 -> D2 -> A3 ... を維持。
- 全10 stepでvalid negative >= 3、Query Base gradient=0、Key gradient=0。
- 全parameter / Queue keyがfinite。
- 終了時 `training_steps=10`、`queue_count=14`。
- changed tensors:
  Query Base 0 / Query LoRA 48 / Query Projector 4 /
  Key Base 0 / Key LoRA 48 / Key Projector 4。
- `Stage 6A 10-step mechanics: PASS`。

fresh 100-step canary:

- 10-step stateを引き継がない別processで `fresh state: True` を確認。
- A1 / B1 / C1 / D1 warm-up後にQueue=4。
- step 0〜99をround-robinで完走し、各sequenceの`sequence_index`がchronologicalに進んだ。
- 全stepで`queue_unique_sequence_id_count=4`、valid different-sequence negative >= 3。
- 全stepでQuery LoRA / Query Projector gradientがfiniteかつnon-zero。
- 全stepでQuery Base gradient=0、Key gradient=0。
- 全parameter / Queue keyがfinite。
- step 99終了時 `queue_count=104`。
- 終了監査は `training_steps=100`、`max_steps=100`、
  `queue_count=104`。
- changed tensors:
  Query Base 0 / Query LoRA 48 / Query Projector 4 /
  Key Base 0 / Key LoRA 48 / Key Projector 4。
- `Stage 6A 100-step mechanics: PASS`。

loss、positive similarity、gradient normはstep間で変動したが、本specでは単調減少やexact値を
成功条件にしていない。今回のEvidenceは100-step範囲のengineering stabilityを示すものであり、
representation性能向上、loss収束、temporal order / motion獲得は未評価である。
