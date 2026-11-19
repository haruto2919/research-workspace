---
project: sequential-video-lora-analysis
date: 2026-11-17
record_type: gpu-run-evidence
status: running
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: 59bade2
base_spec: ../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md
base_run: 2026-11-04-sequential-loader-clip-moco-lora-g2-gpu-evidence.md
check_run: 2026-11-17-sequential-loader-clip-moco-lora-8frames-3gpu-check-gpu-evidence.md
---

# sequential-loader clip MoCo LoRA：8フレーム・GPU 3枚の6,255step学習

clipのフレーム数を4から8に増やして、6,255stepの学習を、ユーザーの指示で始めた。前回の6,255step学習（stride 16、[記録](2026-11-04-sequential-loader-clip-moco-lora-g2-gpu-evidence.md)）と比べるためである。8フレームはGPU 2枚ではVRAMに入らない見込みだったので、GPUを3枚にした。直前の10step確認run（[記録](2026-11-17-sequential-loader-clip-moco-lora-8frames-3gpu-check-gpu-evidence.md)）で、この構成が工学PASSになることを確かめている。

- このrunのための専用specはない。前回の設定（[spec](../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md)、`conf/moco/experiment/g2.yaml`）に、コマンドで上書きを足した。
- この記録は開始時点のものである。完走後に結果を追記する。

## 実行

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
CUDA_VISIBLE_DEVICES=0,1,2 .venv/bin/python main_moco.py +experiment=g2 trainer.devices=3 \
    training.global_batch_size=255 moco.queue_size=65280 data.clip.frames=8 experiment_name=g2_t8_3gpu
```

- 研究コード `59bade2`（feature-vit-moco、clean）。GPU 0・1・2、tmuxセッション `vit-moco-a1`。
- 開始：2026-11-17 15:55:23。
- 出力：`log/moco/g2_t8_3gpu/lora/2026-11-17_15-55-31-413149/`
  - `resolved_config.yaml` に、`devices: 3`、`frames: 8`、`frame_stride: 16`、`global_batch_size: 255`、`queue_size: 65280`、`experiment_name: g2_t8_3gpu` が記録されている。
- 実行前に、学習を動かさずに設定だけを表示し（Hydraの `--cfg job`）、上書きした値とstep数・checkpoint・probeのstepを確かめた。

## 設定（前回との違い）

| 項目 | 前回（stride 16） | 今回 | 理由 |
|---|---|---|---|
| clipのフレーム数 | 4 | **8** | 試したい変更 |
| GPU | 2枚（0・1） | **3枚（0・1・2）** | 8フレームは、2枚・1枚あたりbatch 128ではVRAMに入らない見込みだったため |
| 全体のbatch（1枚あたり） | 256（128） | **255（85）** | 研究コードの決まりで、batchはGPUの枚数で割り切れる必要があるため |
| Queueサイズ K | 65,536 | **65,280**（255×256） | 研究コードの決まりで、Kはbatchの倍数である必要があるため |
| 1 rankのstream数 | 128 | 85 | 1枚あたりのbatchと同じ |
| 実験名（出力先） | g2 | g2_t8_3gpu | 保存先を分けるため |

前回と同じ設定：

- フレーム間隔：stride 16
- モデル：ViT-B/16（`google/vit-base-patch16-224`）、LoRA（Q・V、rank 8、alpha 8.0、dropout 0）、projector 768→768→128、gradient checkpointingあり
- MoCo：momentum 0.999、temperature 0.2
- データ：ActivityNet v1.3 trainingを順番に読む（1 rankあたりworker 16、prefetch 2、forkserver）
- 学習：学習率3e-4（一定）、AdamW（weight decay 1e-4）、6,255step、seed 0、FP32
- checkpoint：1,000・2,000・3,000・4,000・5,000・6,000・6,255
- 固定probe：ImageNet 64枚、step 0, 1, 2, 10, 100, 256, 1,000〜6,000, 6,255

## 開始後の状態（2026-11-17 16:52ごろ）

| 項目 | 値 |
|---|---|
| 進んだstep | 149（epoch 0） |
| loss（step 1・2・3） | 7.505、7.531、7.531。確認runと同じ値 |
| 値 | step 1〜149ですべて有限 |
| 1stepの計算（`step_time_sec`、step 2以降の平均） | 14.86秒 |
| データ待ち（遅い方のrank、step 2以降の平均） | 5.50秒 |
| GPUの使用量 | GPU 0：15,629 MiB、GPU 1・2：15,261 MiB（使用率はどれも100％） |

## 見込み（未確認）

- 1stepは約20秒（計算＋データ待ち）で、全体は約35時間。終わるのは2026-11-19の未明（3〜4時ごろ）の見込み。
- 1clipは128フレーム分の範囲（8フレーム×stride 16）を読む。前回の2倍なので、epochは約800stepで、epochの終わりは約7回ある見込み。最初のepochの終わりは、2026-11-17の20時ごろの見込み。
- 学習に入れるフレーム数は、6,255×255×8＝12,760,200で、前回（6,405,120）の約2倍になる。全動画は約8周する見込み（前回は約4.2周）。
- 前回と比べるときの注意：フレーム数のほかに、GPUの枚数・batch・Kも少し違う。step数はそろえているが、学習に入れたフレーム数と、全動画を何周したかはそろっていない。

## まだ行っていないこと

- 6,255stepの完走と、工学的な確認（完走、Queue、最後の状態確認、VRAM、checkpoint、epochの終わり）。
- lossとcollapseの兆候の記述、前回（4フレーム）との比較。
