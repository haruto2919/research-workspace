---
project: sequential-video-lora-analysis
date: 2026-11-17
record_type: gpu-run-evidence
status: engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: 59bade2
base_spec: ../specs/2026-11-04-sequential-loader-clip-moco-lora-6255step-spec.md
---

# sequential-loader clip MoCo LoRA：8フレーム・GPU 3枚の10step確認

clipのフレーム数を4から8に増やせるかを確かめるため、GPU 3枚で10stepだけ学習した。ユーザーが「GPU 3枚・8フレーム」の構成を選んだ。目的は、長い学習の前に、VRAMが入るかと、1stepの時間を測ることである。

- このrunのための専用specはない。前回の設定（`conf/moco/experiment/g2.yaml`）に、コマンドで上書きを足した。
- **10stepを完走し、工学PASS**。GPU 1・2（RTX A4500、20 GB）のVRAMの余裕は5,206 MiBで、基準（1,024 MiB以上）を満たした。

## 実行

```bash
cd /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora
CUDA_VISIBLE_DEVICES=0,1,2 .venv/bin/python main_moco.py +experiment=g2 trainer.devices=3 \
    training.global_batch_size=255 moco.queue_size=65280 data.clip.frames=8 training.max_steps=10 \
    'training.checkpoint_steps=[]' 'diagnostics.probe_steps=[0,1,2,10]' experiment_name=g2_t8_3gpu_check
```

- 研究コード `59bade2`（feature-vit-moco、clean）。GPU 0・1・2、tmuxセッション `vit-moco-a1`。
- 時刻：2026-11-17 15:01:14開始 → 15:10:53終了（exit 0、run内の経過は569秒）。
- 出力：`log/moco/g2_t8_3gpu_check/lora/2026-11-17_15-01-22-383541/`

## 前回の6,255step学習（stride 16）との違い

| 項目 | 前回 | 今回 | 理由 |
|---|---|---|---|
| clipのフレーム数 | 4 | **8** | 試したい変更 |
| GPU | 2枚（0・1） | **3枚（0・1・2）** | 8フレームは、2枚・1枚あたりbatch 128ではVRAMに入らない見込みだったため |
| 全体のbatch（1枚あたり） | 256（128） | **255（85）** | 研究コードの決まりで、batchはGPUの枚数で割り切れる必要があるため |
| Queueサイズ K | 65,536 | **65,280**（255×256） | 研究コードの決まりで、Kはbatchの倍数である必要があるため |
| 1 rankのstream数 | 128 | 85 | 1枚あたりのbatchと同じ |
| step数・checkpoint・probe | 6,255・7個・13点 | 10・なし・4点（0, 1, 2, 10） | 確認runのため |

フレーム間隔（stride 16）、学習率（3e-4）、LoRA、モデル、temperature、momentumなど、上記以外は前回と同じ。

## 結果

### 工学的な確認

| 項目 | 結果 |
|---|---|
| 完走 | **PASS**。`status: completed`、`last_global_step` 10、failuresなし。全stepで値は有限、学習率は3e-4 |
| 設定の記録 | `world_size` 3、`local_batch_size` 85、K 65,280、`clip.frames` 8、stream数は1 rankあたり85 |
| 最後の状態確認 | 全rankが同じ状態、凍結した部分は変わっていない |
| diagnostic summary | `engineering_status: PASS`、reasonsなし、`uniformity_warnings` は空 |
| Queue | queue_ptrは毎step 255ずつ進んだ（step 10で2,550） |

### VRAM（`vram_gate: PASS`）

| GPU | max allocated | max reserved | 全体 | 余裕（headroom） |
|---|---|---|---|---|
| GPU 0（RTX PRO 4500 Blackwell） | 13,725 MiB | 15,038 MiB | 32,218 MiB | 17,180 MiB |
| GPU 1（RTX A4500） | 13,677 MiB | 14,890 MiB | 20,096 MiB | **5,206 MiB** |
| GPU 2（RTX A4500） | 13,677 MiB | 14,890 MiB | 20,096 MiB | **5,206 MiB** |

- 実行前の見積もり（約16,600 MiB）より、約1,700 MiB少なかった。見積もりは、前回のrun（4フレーム・1枚あたり128、max reserved 12,784 MiB、学習前は1,282 MiB）から、VRAMが「batch×フレーム数」に比例して増えるとみなして出していた。
- 今回の実測から計算し直すと、学習で増える分は「batch×フレーム数」1つあたり約20.0 MiB だった（前回の実測からは約22.5 MiB）。
- 参考（見積もりのみ、未確認）：この値を使うと、1枚あたりbatch 85で10フレームにした場合は約18,300 MiB になり、余裕は約1,800 MiB と見込まれる。

### 時間

| step | 1stepの計算（`step_time_sec`） | データ待ち（3 rank） |
|---|---|---|
| 1 | 15.97秒 | 142.8〜144.2秒（loaderの起動を含む） |
| 2〜10 | 14.47〜14.80秒（平均約14.7秒） | 3.4〜5.5秒（step 4以降は約5.2〜5.5秒） |

- 1stepの計算は約14.7秒で、前回（8.84秒）の約1.7倍だった。実行前の見込み（約13秒）より少し長い。
- データ待ちは約5.4秒/stepで、前回（約2.5秒）より長い。今回はデータの読み込みが、学習の計算より遅い。
- 1stepあたりの実時間は、計算とデータ待ちを合わせて約20秒になる見込み。6,255stepなら約35時間になる（見込み、未確認）。

### loss（記述のみ）

- step 1〜10のlossは7.505〜7.808で、少しずつ上がった。Queueがまだほとんど埋まっていない時期であり、前回のrunでも、step 1〜10の平均（7.66）からstep 11〜256の平均（8.49）へ上がっていた。10stepでは、学習の挙動について何も判断しない。

## まだ行っていないこと

- 6,255stepなどの長い学習。長い学習の前に、次の点を決める必要がある。
  - フレーム間隔をstride 16にするかstride 8にするか。
  - step数をどうそろえるか。今回の設定で6,255stepにすると、学習に入れるフレーム数は前回の約2倍（6,255×255×8＝12,760,200）になる。
- データ待ちが長くなった原因（workerの数、CPUの使用率、デコードする量）は確認していない。
