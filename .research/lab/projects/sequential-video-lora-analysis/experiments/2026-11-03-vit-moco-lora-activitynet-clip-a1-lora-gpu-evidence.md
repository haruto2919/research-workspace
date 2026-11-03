---
project: sequential-video-lora-analysis
date: 2026-11-03
record_type: gpu-run-evidence
status: a1-lora-engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: fe58fc37ef28901184980ccd5f9e90e40c3d836d
spec: ../specs/2026-11-03-vit-moco-lora-activitynet-clip-early-fusion-spec.md
---

# ViT＋MoCo ActivityNet clip Early Fusion：GPU A1（LoRA・10step）

[clip Early Fusion spec](../specs/2026-11-03-vit-moco-lora-activitynet-clip-early-fusion-spec.md)のA1（LoRA）を、ユーザーの許可を得てtmuxセッション `vit-moco-a1` から実行した。**10stepを完走し、工学PASS**となった。VRAMは単一フレームA1とほぼ同じだった。一方で、**1stepの計算時間は約8.8秒で、単一フレームA1（約1.46秒）の約6倍**だった。A1 Full FT・A2は未実施。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet_clip model.train_mode=lora training.max_steps=10 training.learning_rate=0.0003 'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'
```

- 研究リポジトリ：`fe58fc3`、clean。GPU構成は単一フレームと同じ（GPU0：RTX PRO 4500 Blackwell、GPU1：RTX A4500）。開始前は両GPUとも空き。
- 出力：`log/moco_vit_lr_activitynet_clip/lora/activitynet_clips/lr_0.0003/2026-11-03_15-01-35-356834/`
- 時刻：15:01:27開始 → 15:09:55終了（run本体は499秒）。

## 工学チェック

| 項目 | 結果 |
|---|---|
| run / engineering | `completed` / **PASS**（理由なし、`failures=[]`） |
| VRAM gate | PASS。peak reserved：GPU0 12,784 MiB / 32,218（headroom 19,434）、GPU1 12,784 MiB / 20,096（headroom **7,312**、閾値1,024） |
| final checks | 全rankがrank0と同じ状態（state sha256 `92042cd2…`）。frozen base（198 parameter）は開始時と終了時で一致 |
| 学習対象 | 983,936 parameter（Q/V LoRA＋projector）。単一フレームのLoRAと同じ |
| データ | 117,522 clip、459 step/epoch、frames_sha256 `c9cc4aec…105c`。clip：4フレーム、0.25秒間隔、`last_frame` |
| 固定probe | 元の64枚の入力ハッシュ `6af50745…08d8bf`（**L4/L5・単一フレームA1/A2と一致**）。静止clipの入力ハッシュ `457fabe0…cd1f`（形 `[64, 4, 3, 224, 224]`） |
| probe step | 0・1・2・10がすべて記録された |
| queue | step10でqueue_ptr 2,560 |

- `ranks_feature_identical` は `false`。単一フレームのA1・A2でも同じく `false` であり、成功条件の項目ではない。
- 単一フレームA1 LoRA（`2026-11-02_13-57-26-045737`）のpeak reservedは、GPU0 12,190 MiB、GPU1 12,338 MiBだった。clipでは785 tokenに増えたが、gradient checkpointingにより、reservedの増加は約0.4〜0.6 GiBにとどまり、allocatedはむしろ約1.2 GiB少なかった。

## 学習中の値（参考）

| step | loss | positive cos | negative cos | hardest neg cos | grad norm | q std | step_time |
|---|---|---|---|---|---|---|---|
| 1 | 7.627 | 0.712 | −0.0002 | 0.367 | 2.90 | 0.0698 | 11.69 s |
| 2 | 7.540 | 0.736 | 0.0015 | 0.681 | 1.58 | 0.0589 | 8.65 s |
| 5 | 7.622 | 0.747 | 0.0077 | 0.751 | 1.06 | 0.0473 | 9.06 s |
| 10 | 7.741 | 0.756 | 0.0175 | 0.766 | 1.12 | 0.0491 | 8.83 s |

固定probe（rank0）の `z_q` 方向集中（offdiag cos p50）：

| step | 単一フレームA1 LoRA | clip A1 LoRA（静止clip） |
|---|---|---|
| 0 | 0.319 | 0.318 |
| 1 | 0.455 | 0.452 |
| 2 | 0.541 | 0.536 |
| 10 | 0.625 | 0.602 |

- CLSのoffdiag cos p50は全stepで0.032（p95 0.123）で、ほぼ変わらない。
- `q` のuniformity警告は出ていない。
- **10stepでQueueはまだ2,560 / 65,536しか本物のkeyで埋まっていない。上の値から、学習の良し悪し、collapse、lossが下がるかについては何も主張しない。** 単一フレームとclipは入力が違うため、lossの値の大小から何も言わない。

## 時間

ファイルの更新時刻から見た目安：

| 区間 | 時間 |
|---|---|
| 起動 → 固定probeの作成終了 | 約2分29秒 |
| → step1の終了（最初のbatchの待ちとprobe step0・1を含む） | 約3分37秒 |
| step1 → step10 | 約97秒（step_time合計79秒、probe 2を含む） |

- **1stepの計算時間（`step_time_sec`）はstep2〜10で8.62〜9.06秒（約8.8秒）**。単一フレームA1 LoRAは約1.46秒で、約6倍になった。785 tokenのattentionとgradient checkpointingによる再計算の分と見られる（内訳は測っていない）。
- step1〜10の間、データ待ちは見えなかった（step間の時間がstep_timeとprobeの合計にほぼ等しい）。ただし、最初の待ちの間に16 workerが先にbatchを作っているので、10stepでは定常状態のデータ読み込み速度は測れていない。これは単一フレームのA1と同じ限界である。
- 定常状態の見積もり（未測定）：1 clipの取り出しは平均0.40秒（32プロセス同時、[実装記録](2026-11-03-vit-moco-lora-activitynet-clip-early-fusion-implementation-cpu-verification.md)）なので、1 workerが128 clipのbatchを作るのに約51秒＋拡張（8枚／sample）。16 workerで1 batchあたり約3〜4秒となり、計算時間（約8.8秒）より短い。**A2 LoRAは計算で律速され、459step × 約9秒 ≈ 約70分＋起動約4分と見積もる**（単一フレームA2 LoRAは約35分）。
- spec §7の「1stepあたりの時間が単一フレームA2（LoRA約3.8秒）より大幅に長い場合は報告し、A2を実行するかはユーザーが判断する」に該当する（約8.8秒、約2.3倍）。

## 未実施

- A1 Full FT（10step）。別に許可が必要。Full FTの単一フレームA1でのGPU1 headroomは1,750 MiBだった。clipでの値は未測定。
- LoRAとFull FTの比較器（A1 Full FTの後）。
- A2（LoRA → Full FT、各459step）。別に許可が必要。
