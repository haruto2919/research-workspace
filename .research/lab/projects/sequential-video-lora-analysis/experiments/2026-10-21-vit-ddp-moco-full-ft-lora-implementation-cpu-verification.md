---
project: sequential-video-lora-analysis
date: 2026-10-21
record_type: implementation-verification
status: cpu-pass-gpu-not-run
evidence_level: cpu-unit-integration-regression-plus-real-data-manifest
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: dev
base_commit: 7fc1f56a19fb00c8fce47fa23fe2e85d353b8a62
research_commit: 2c269915f0c5036fa2a460959fdf401d54456026
code_state: committed (9348efe feat, 2c26991 test) and pushed to origin/dev on 2026-10-21
spec: ../specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md
source: ai-executed-on-research-server (Claude Code)
---

# ViT＋自作DistributedMoCo Full FT / LoRA切替：実装・CPU検証とActivityNet manifest生成

## 結論・Evidence境界

- **CPU Engineering Gate：PASS**。対象は承認済みspec §12.4のG0、G1、G2、G4、G5と、ActivityNet manifestの生成。
- AIが研究サーバ上で実行した結果である。
- **G3（2GPU pilot）とGPUでのG4は未実施**（未許可）。ViT-B Full FTをlocal batch 128・FP32で動かしたときのVRAM適合、step time、学習の安定性は未測定。
- Loss低下、表現品質、Full FTとLoRAの優劣は、この記録では主張しない。
- 研究コードは`dev` `7fc1f56`の上に**新規ファイルだけを追加した未commit状態**。既存ファイルの差分は0。commit/pushはしていない。
- 実行環境：Python 3.12.3、torch 2.14.0+cu130、transformers 5.17.0、PyAV 18.1.0。shellでは`CUDA_VISIBLE_DEVICES`が空のため、torchからGPUは見えない。

## 追加したファイル（研究コード）

| file | 役割 |
| --- | --- |
| `model/vit/vit_moco_backbone.py` | `ViTMoCoBackbone(checkpoint_id, train_mode)`。HF `ViTModel(add_pooling_layer=False)`のCLS `[B, 768]`と`feature_dim`を返す。`lora`はViTを凍結し、既存`inject_attention_lora`でQ/Vに注入。`full_ft`は全parameterを学習 |
| `self_supervised/moco/vit_ddp_moco.py` | `ViTDistributedMoCo(DistributedMoCo)`。EMAは`requires_grad=True`のQuery parameterに対応するKeyだけを更新。`trainable_query_parameters()`を追加 |
| `scripts/vit_moco/activitynet_frames.py` | ActivityNet単一フレーム：timestamp規則（t=5+10n秒）、「時刻tに表示されているフレーム」のdecode、manifest生成CLI、manifest dataset |
| `scripts/vit_moco/moco_vit_ddp_smoke.py` | 2-rank DDP runner：AdamW、epoch cosine（warmupなし）、GPU recipe guard、step metrics、入力hash、checkpoint/resume（sampler cursor、RNG sidecar、identityによる拒否）、終了時のrank間state hash一致とFrozen Base不変の確認、VRAM gate |
| `conf/moco_vit_ddp_smoke.yaml` | 専用Hydra preset（既存の`moco_imagenet_*`には触れていない） |
| `test/model/test_vit_moco_ddp.py` | G0 / G1 / G2（13件） |
| `test/model/test_moco_vit_ddp_smoke.py` | config契約、G4、ActivityNet frames、runnerのCPU実行（41件） |

`standalone_moco.py`、`ddp_moco.py`、`scripts/parity/*`、既存conf、既存testは変更していない。

## Gate別の結果

| Gate | 結果 | 観測 |
| --- | --- | --- |
| G0：実checkpoint | PASS | ・HF cacheの`google/vit-base-patch16-224`（snapshot `3f49326e…`）で、LoRALinearは`layers.{0..11}.attention.{q,v}_proj`の24箇所だけ（k/oは素のLinear）。<br>・LoRA B=0、A≠0、scaling 1.0、dropout 0。<br>・LoRA trainableは294,912、Full FTのViT parameterは85,798,656。<br>・pooler/classifierなし、feature_dim 768。<br>・両モードでpretrained hash・Projector・Queueが一致し、CLS `[2, 768]`が**両モード間でbitwise一致**（initial-equivalence） |
| G1：2-rank CPU | PASS | 小configの実`transformers.ViTModel`（hidden 16、2 block）を両モードで使用。<br>・rank 1を意図的にずらしても、DDPの初期化後は全rankがrank 0の初期state（Query/Key/Queue）に一致し、Key＝Query。<br>・Keyの勾配はNone。Query勾配と更新後parameterは全stepでrank間bitwise一致。<br>・shuffle後のKeyがunshuffleで元の順序へ戻る（ローカル計算と許容差内で一致）。<br>・全rankのKeyがrank順にenqueueされ、pointerは(4s mod 16)、5stepでwrap、Queueはrank間bitwise一致。<br>・InfoNCEは有限 |
| G2：trainability | PASS | ・Full FT：Query ViTとProjectorの全parameterに有限の勾配があり、更新された。<br>・LoRA：Base（Query/Key）は5step後もbitwise不変、frozen Key＝Query。step 1はLoRA Bだけに勾配があり、Aの勾配はちょうど0。step 2でAにも勾配。Projector・A・Bは更新された。<br>・EMAは全stepで`k·m + q_before·(1−m)`とbitwise一致し、frozen pairは不変。<br>・全parameterが学習可能なときは、親`DistributedMoCo`のEMAとbitwise一致 |
| G4：save/resume（CPU、gloo） | PASS | ・step 2（epoch境界）とstep 3（epoch途中、sampler cursor 1 batch）のcheckpointから新directoryでresumeし、最終state hashと継続stepのmetricsがuninterrupted runと**bitwise一致**（num_workers 2と0の両方）。<br>・復元直後のstateは保存stateとbitwise一致。<br>・別train_mode（lora→full_ft）や別identity（LR変更）のresumeは、**両rankともstate読込前に拒否**。<br>・既存run directoryへの上書きは拒否 |
| config契約 | PASS | ・`train_mode`と`max_steps`は必須。<br>・GPU runは採用recipe（13項目を個別に変えて全て拒否）とmax_steps∈{1, 10, 20}以外を拒否。<br>・AMPと`broadcast_buffers=false`は拒否。<br>・ActivityNetのmanifest指定の整合を検査 |
| G5：既存非回帰 | PASS（差分なし） | ・変更前のfull pytest：511 passed / 197 failed / 18 errors。<br>・変更後：565 passed / 197 failed / 18 errors。<br>・**失敗したtest IDの集合は変更前後で完全一致**。増えた54 passedは新規test。<br>・既存の失敗は、GPU前提（`test_image_models`、`test_model_factory`、`test_checkpoint`、`test_videomodels`、`test/setup` conftestの`assert torch.cuda.is_available()`）、NAS上のKinetics400、`test_moco_5epoch_pair`の1件で、いずれも変更前から存在する |

その他：ActivityNet framesの単体test（timestamp境界、mp4のstream長／mkvのcontainer長、pts ≤ tの最後のフレーム、manifestの上書き・改ざんの拒否、読めない動画での停止）と、合成動画でのrunner CPU実行もPASS。

## ActivityNet manifest生成（実データ、CPU、NAS読み取りのみ）

- 出力：`/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/activitynet_frames/2026-10-21_training_interval10s_offset5s/`（git ignore対象の`log/`に新規作成）。
- policy：training split、10秒間隔、t=5+10n秒、t < stream長、上限なし、Positiveは同一フレームの2-view、同一動画negativeは除外しない。
- 結果：
  - 動画数**10,024**（読めない動画0）、フレーム数**117,522**。
  - フレーム0枚の動画が26本、動画あたりの最大は75枚、平均11.72枚。
  - stream長の取得元：stream 9,062本、container 962本（主にmkv/webmでstreamに長さがないもの）。
  - 生成時間は607秒（16 thread）。
- hash：
  - `manifest.json`：`065c3170db18ac39f266d3a2cee55e48854a613ee1fc9386f6c4f83051a2b472`
  - `frames.jsonl`：`c9cc4aecac9d4551412538b852d524848d87ec1734081c4c4bfa794e2238105c`
  - annotation JSON：`4c29d5b1…`
- provenance：`sequential_loader` commit `19a0ed7`（clean）。研究コードは`7fc1f56`に未commitの新規ファイルがある状態（`dirty: true`がmanifestに記録されている）。
- stream長とJSON durationの差：中央値−0.086秒、99パーセンタイルの|差| 0.53秒、最大45.21秒。1秒を超える差は37本。manifestはstream長を使う。
- 実decodeの確認：ランダム32枚と各動画の最終フレーム32枚、計**64/64を成功**。出力は2-view `[3, 224, 224]` float32、Normalizeは0.5/0.5。単一processで1件あたり中央値0.164秒、最大0.934秒。

### 追記：commit・push（2026-10-21）

- ユーザーの指示により、上記の新規ファイルを`dev`にcommitした（`9348efe` feat：コード＋conf、`2c26991` test：test 2件）。`origin/dev`へpush済みで、HEAD＝`origin/dev`＝`2c269915f0c5036fa2a460959fdf401d54456026`、worktreeはclean。
- 検証は、commit前の同一内容のファイルで実施した。
- `activitynet_frames.py`の最終更新（17:08）はmanifest生成（17:22〜17:32）より前で、commitされた内容と作業ファイルは同一。したがってmanifestは`9348efe`と同じコードで生成されている。ただし、manifest内のprovenanceには生成時点の`7fc1f56`・`dirty: true`が記録されたままである。

## 未確認点と次の判断

1. **G3（GPU）は未実施**。ResNet50 E5のpeak reservedは約13GiB/20GiB/rankだった。ViT-B Full FT（FP32、local 128、activationを保持）はこれより大きくなる見込みだが、未測定（20GiBのA4500にかなり厳しいという概算のみ）。LoRAもfrozen Baseを通してbackwardするため、activation memoryは大きく減らない見込み。OOMやheadroom不足のときは、spec通りBLOCKEDで停止し、自動では変更しない。
2. ActivityNetのdata loadingは1件あたり約0.16秒（単一process）。16 workers/rankでは、粗い見積もりで1stepあたり約1.3秒のdata時間になり、律速になる可能性がある（未実測）。
3. （2026-10-21に解消）研究コードはcommit・push済みで、G3の前提であるcleanなcommitを満たした。manifestは`9348efe`と同一内容のコードで生成されているが、provenanceの記録は`7fc1f56`＋dirtyのままである。この記録を正とするためにcommit後に再生成するかどうかは、G3を承認するときに判断する。
4. AdamW lr 0.03でのViTの安定性、Full FTとLoRAの比較、Linear Probeは別Gate。

## 関連

- [承認済みspec（§12）](../specs/2026-10-21-vit-ddp-moco-full-ft-lora-switch-smoke-spec.md)
- [E5 Evidence](2026-10-21-meta-moco-v2-e5-imagenet-5epoch-pair-parity-evidence.md)
- [source brainstorm](../../../../secretary/notes/brainstorm/2026-10-21-full-vit-vs-frozen-vit-lora-moco-before-sequential-loader.md)
