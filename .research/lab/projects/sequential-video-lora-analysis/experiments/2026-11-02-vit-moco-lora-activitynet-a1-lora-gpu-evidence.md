---
project: sequential-video-lora-analysis
date: 2026-11-02
record_type: gpu-run-evidence
status: a1-lora-engineering-pass
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: 9155c2baa59c06f8204b81d7d466bf4fc4e955d6
spec: ../specs/2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md
---

# ViT＋MoCo ActivityNet単一フレーム：GPU A1（LoRA・10step）再実行

[11/01のA1失敗](2026-11-01-vit-moco-lora-activitynet-a1-lora-gpu-failure.md)を受けて、`decode_frame` の修正とrank間のエラー共有を `feature-vit-moco@9155c2b` としてcommitした。その後、ユーザーの指示でtmuxの専用セッション `vit-moco-a1` からA1のLoRAを再実行した。**10stepを完走し、工学PASS**となった。Full FTのA1は未実施。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet model.train_mode=lora training.max_steps=10 training.learning_rate=0.0003 'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'
```

- 研究リポジトリ：`9155c2b`、clean。GPU構成はL4/L5と同じ（GPU0：RTX PRO 4500 Blackwell、GPU1：RTX A4500）。
- 出力：`log/moco_vit_lr_activitynet/lora/activitynet_frames/lr_0.0003/2026-11-02_13-57-26-045737/`
- 時刻：13:57:18開始 → 14:04:19終了（run本体は412秒）。

## 工学チェック

| 項目 | 結果 |
|---|---|
| run / engineering | `completed` / **PASS**（理由なし） |
| VRAM gate | PASS。peak allocated：GPU0 11,772 MiB / 32,218、GPU1 11,724 MiB / 20,096 |
| final checks | 全rankがrank0と同じ状態、frozen baseは不変、frozen keyはqueryと一致 |
| データ | 117,522 frame、459 step/epoch、frames_sha256 `c9cc4aec…105c` |
| 固定probe | 入力ハッシュ `6af50745…08d8bf`。**L4/L5と一致** |
| probe step | 0・1・2・10がすべて記録された |
| queue | step10でqueue_ptr 2,560（= 10 × 256） |

## 学習中の値（参考）

| step | loss | positive cos | negative cos | grad norm | q std |
|---|---|---|---|---|---|
| 1 | 7.689 | 0.700 | −0.0003 | 3.13 | 0.0693 |
| 2 | 7.578 | 0.729 | 0.0015 | 1.65 | 0.0587 |
| 5 | 7.681 | 0.736 | 0.0078 | 1.05 | 0.0466 |
| 10 | 7.813 | 0.745 | 0.0178 | 1.10 | 0.0471 |

固定probe（rank0）の `z_q` 方向集中（offdiag cos p50）は、step0 0.319 → step1 0.455 → step2 0.541 → step10 0.625 と上がった。`q` のuniformity警告は出ていない。

- step1のlossは、11/01の失敗runとbitwise一致した（7.68887996673584）。step2のlossは小数第7位でずれた。これは、L5で分かったGPU backwardの非決定性と同じ種類のずれと考えられる。
- **10stepで、Queueもまだほとんど本物のkeyで埋まっていない（2,560 / 65,536）。上の値から学習の良し悪しやcollapseについては何も主張しない。** L4/L5でも、Queueが埋まるまではloss上昇・方向集中が見られた。

## データ読み込み時間

ファイルの更新時刻から見た目安：

| 区間 | 時間 |
|---|---|
| 起動 → 固定probeの作成終了（モデル読み込み、ImageNetのImageFolder走査を含む） | 約2分50秒 |
| → step1（最初のbatchの待ちとprobe step0・1を含む） | 約3分20秒 |
| step2 → step10 | 約20秒（先読み済みのbatchを使用、step_time 1.45 s） |

- 1 workerが128 frameの最初のbatchを作るのに、約3分かかったことになる。全frame確認で測った平均0.53 s/frame（約70秒/batch）より遅い。32プロセスが同時に1080p動画をdecodeするための競合が考えられるが、原因は未確認。
- 定常状態で各workerが約3分に1 batchを作るとすると、16 workerで459stepに**約1.5時間**かかる。70秒に1 batchなら約35分。A2の所要時間は、この範囲と見積もる（未確定）。

## 未実施

- Full FTのA1、A2（LoRA → Full FT、各459step）。
