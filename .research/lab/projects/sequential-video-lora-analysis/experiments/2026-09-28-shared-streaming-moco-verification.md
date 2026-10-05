---
project: sequential-video-lora-analysis
record_type: implementation-verification
status: completed
created: 2026-09-28
last_updated: 2026-09-28
spec: ../specs/2026-09-28-shared-streaming-moco-protocol-spec.md
implementation_branch: dev
implementation_base_commit: 4835b5736f0b1dcc9962cbeffd85e880010311ea
implementation_state: uncommitted
sequential_loader_branch: ActivityNet
sequential_loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
---

# Shared Streaming MoCo 実装・短時間検証

[承認済みspec](../specs/2026-09-28-shared-streaming-moco-protocol-spec.md)を実装した。
Stage 6A / 6Bを共通training engineのpresetとして扱い、stream / key transform / negative policyを独立に指定できる。
既存167テスト、新規75テストが成功した。Stage 6Aは基準commitのloopと人工データ10 stepで数値一致も確認した。

実装rootは `/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora`。
着手時は指定origin、`dev`、基準commitと一致し、worktreeはcleanだった。変更は未コミット。
loaderは指定branch / commitでcleanのまま変更していない。

## 実装差分

| File | 内容 |
|---|---|
| `training/moco_protocol.py` | frozenな3軸protocol、Stage 6A / 6B preset、stream数に一致するwarm-up数 |
| `training/moco_canary.py` | 1 / 4 stream schedulerと共通`run_streaming_moco`。旧`run_canary` / `round_robin_samples`は薄い互換入口 |
| `integration/sequential_moco.py` | valid RGB frameだけのGBR `[1,2,0]` → width flip。既定horizontal flipを維持 |
| `self_supervised/moco/negative_selection.py` | Queue保存責務から独立したdifferent-sequence / all-past選択 |
| `self_supervised/moco/vit_lora_moco.py` | InfoNCEに選択済みnegativeを渡せる。既存3引数呼び出しは従来どおり |
| `training/moco_audit.py` | transformに応じたview監査、全metadataの保存確認 |
| `scripts/smoke/smoke_activitynet_streaming_moco.py` | Stage 6B既定の共通CLI、presetと3軸override、provenance確認 |
| `test/model/test_moco_protocol.py` | protocol、transform、selector、InfoNCE、K=4096 overflowの26テスト |
| `test/model/test_streaming_moco.py` | strict stream、warm-up、更新順、FIFO、EOF / exception、実ViT接続の25テスト |
| `test/model/test_streaming_moco_cli.py` | source選択、preset / override、fresh state、入力拒否、provenanceの24テスト |
| `test/model/test_moco_multistep_canary.py` | lossの監視wrapperに選択済みnegative引数を受け渡す。既存assertionは維持 |

学習loopは共通engineの1本で、model内部には追加していない。
optimizerはwarm-up完了後に1回だけ生成する。loss(old Queue) → backward → Query optimizer → Key EMA → pre-EMA positive enqueueの順を維持した。
Queue監査の独立参照履歴もcapacity付きFIFOにし、新engineはoverflowを許容する。
既存Stage 6A CLIの既定値と4092 step上限は変更せず、互換wrapperからStage 6A presetを渡す。

`strict_single + different_sequence`も独立軸として指定できるが、1動画内のfresh Queueでは有効negativeが存在しない。
この組合せでは最初のtraining sampleで明示エラーとなる。別動画を読み足す、negative policyを変える、current positiveをnegativeにする等の補完はしない。

## 検証結果

CPU、既存`.venv`、offlineで実施した。checkpoint download、新規dependency、GPU runは不要だった。

| 検証 | 結果 |
|---|---|
| Stage 6A + Stage 5 + ViT / LoRA + masked mean + 50Salads + config | 167 passed、127.05秒、既存dependency warning 2件 |
| 新規protocol / transform / selector / Queue / InfoNCE | 26 passed、8.45秒 |
| 新規stream / synthetic engine | 24 passed、9.65秒 |
| Stage 6Bの実ViT / PEFT integration | 1 passed、16.87秒 |
| 新規CLI | 24 passed、7.86秒 |
| Stage 6A基準loopとの人工10-step比較 | sample順、診断ログ、全parameter、Queue key / metadataが完全一致 |
| 全変更PythonのAST、`git diff --check` | 成功 |
| 既存Stage 6A / 新規Streaming CLIの`--help` | 成功 |

Stage 6A数値比較は同一の軽量model初期stateを複製して、基準commitから読み出した旧loopと現行loopへ同じ人工Readerを接続したもの。
実checkpoint / 実ActivityNetの数値一致を測定したものではない。

実ViT / PEFT integrationは既存fixtureを使用する。12 layers、hidden size 768、Q/V 24 target、実PEFT / Projector / InfoNCE / EMAを維持し、MLP intermediate sizeだけ32へ縮小している。
人工16-frame chunkでA1のKey-only warm-upとA2の1 updateを実行し、Queue=2、negative=A1、Query学習対象983,936 parameters / 52 tensors、LoRA finite gradient 48 tensorsを確認した。

既存回帰は次のコマンドで実行した（実装root）。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest \
  test/model/test_moco_multistep_canary.py \
  test/model/test_vit_lora_moco.py test/model/test_activitynet_vit_lora_moco_one_step.py \
  test/model/test_vit_lora_frame_encoder.py test/model/test_activitynet_vit_lora_one_step.py \
  test/model/test_vit_frame_encoder.py test/model/test_masked_mean_clip_aggregator.py \
  test/model/test_activitynet_vit_clip_feature.py test/model/test_50salads_vit_bridge.py \
  test/config -q -o addopts='' -p no:cacheprovider
```

新規テストは上表の各ファイルを`.venv/bin/python -m pytest`で個別実行した。
streaming engineファイルはsynthetic 24件と実ViT 1件を分けて実行した。

## Spec成功条件との対応

| spec 11章 | Evidence |
|---|---|
| 1〜4 protocol / architecture | 全8組合せのprotocol・engineテスト。共通loopとmodelの責務分離を差分レビュー |
| 5〜9 stream順序 / no prefetch / Reader解放 | 既存round-robin、新規strict-singleの非ゼロstart_frame、early stop、EOF、decode / consumer / interrupt / training例外テスト |
| 10〜13 warm-up | event列でKey→enqueueのみ、optimizerはwarm-up後、max_stepsはtraining回数だけと確認 |
| 14〜19 transform | 非対称RGBで全valid frameのGBR→flip、padding、元sample、全metadata保持を検査 |
| 20〜25 negative / FIFO | A2→[A1]、A3→[A1,A2]、current key除外。実K=4096の4100 enqueueと短いengine overflowテスト |
| 26〜33 update | loss / logits / gradients / parameter / Queue finite、base不変、Key勾配なし、Query更新、EMA式、event順をsyntheticと実ViTで確認 |
| 34〜36 regression | 既存167テスト成功とStage 6A人工10-stepの基準loop数値一致 |
| 37 Stage 6B | 新規75テスト成功、実ViT / PEFT接続を含む |

## 実行入口

実装rootからのStage 6B smokeのコマンド例（今回この実データコマンドは未実行）。

```bash
.venv/bin/python -m scripts.smoke.smoke_activitynet_streaming_moco \
  /path/to/ActivityNet --device cpu --max-steps 10
```

`--preset stage6a`でStage 6A設定になる。
`--stream-mode`、`--key-transform`、`--negative-policy`はpresetの各軸を個別に上書きする。
既存`python -m scripts.smoke.smoke_activitynet_vit_lora_moco_multistep`も従来どおり使える。

## 範囲と限界

今回の確認はengineering mechanicsに限定される。Stage 6Bの実ActivityNet smoke、full pretraining、長時間GPU run、動画境界state policy、downstream評価は未実施。
表現性能・時間情報獲得・研究手法の有効性は示していない。AIによるテストとレビューの範囲を超えて実装全体の正しさを保証するものでもない。
commit / pushは実施していない。

## 検証済み未コミットコードのSHA-256

| File | SHA-256 |
|---|---|
| `integration/sequential_moco.py` | `cdbfa45716274444ea9024ff4fbf79a53f5678a32170dc2b5214e116ef64b2c6` |
| `scripts/smoke/smoke_activitynet_streaming_moco.py` | `1a528e23cba9ea1284dcf1bb2cead22df96e00dbc0ef028b183ceb0ea4ae56d0` |
| `self_supervised/moco/negative_selection.py` | `a3b499db9c5e7302c736017acc7b1ce00e46b66ee388197f02a26630797cea4a` |
| `self_supervised/moco/vit_lora_moco.py` | `27c49c6e25435628e61ee1ec4b6623f89bc0a4165de6f7dac4806a05598f3f1a` |
| `test/model/test_moco_multistep_canary.py` | `7ec5d7679032fd65315fda42a38e4a859cdc0c88e9e788121c1d88ee6827b053` |
| `test/model/test_moco_protocol.py` | `08b6ed78de52ecc20123bcd199050b15636a26e2bb111ce1cc70854568e1d570` |
| `test/model/test_streaming_moco.py` | `8a25bd5227154721c1637ebb94536fffef5ff2545aeb00f6dacd72c2a449ae00` |
| `test/model/test_streaming_moco_cli.py` | `62e151eafbb850f994b6b651bc32b08a039ec4f87249145f8c08a7fef33d8ef7` |
| `training/moco_audit.py` | `41d154b67cebf7ee9f90dcfff2ca670ba2b2c252ce2d310620f480f51334ce67` |
| `training/moco_canary.py` | `c27142a7cd43a23f9aa050632a651b5bf6b8365ca15355c83394e391aa3fa573` |
| `training/moco_protocol.py` | `e8a0f438e67914959a1680771e2f24efdcd6e54d4941223feae7e6797645f1e1` |
