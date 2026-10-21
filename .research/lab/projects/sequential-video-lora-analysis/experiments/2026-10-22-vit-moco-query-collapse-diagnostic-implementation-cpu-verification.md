---
project: sequential-video-lora-analysis
date: 2026-10-22
record_type: implementation-verification
status: cpu-pass-gpu-not-run
evidence_level: cpu-unit-integration-regression-plus-real-data-probe-preflight
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: dev
base_commit: 2c269915f0c5036fa2a460959fdf401d54456026
code_state: uncommitted (3 new files + 1 modified file on dev@2c26991); not pushed
spec: ../specs/2026-10-22-vit-moco-query-collapse-diagnostic-spec.md
source: ai-executed-on-research-server (Claude Code)
---

# ViT＋MoCo Query Collapse診断：実装・CPU検証（D0〜D3）

## 結論・Evidence境界

- **CPU Engineering Gate D0〜D3：PASS**。承認済みspec（B1〜B5）の固定probe診断hook、更新量計測、診断用JSON sidecar、比較CLI、CPUテストを実装した。
- AIが研究サーバ上で実行した結果である。全テストは `CUDA_VISIBLE_DEVICES=""` で**CPUのみ**で実行した。GPUは使っていない。
- **D4（2GPU×10step診断run）は未実施**。B6（GPU実行許可）は未許可のまま。**D5（解釈）も未実施**。
- 研究コードは `dev@2c26991` 上の**未commit差分**（新規3ファイル＋runner 1ファイル）。commit/pushはしていない。
- 実行環境：Python 3.12.3、torch 2.14.0+cu130、transformers 5.17.0、torchvision 0.29.0、hydra-core 1.3.7。
- Collapseの原因、学習率の効果、表現品質、Full FTとLoRAの優劣は、この記録では主張しない。

## 変更ファイル（研究コード）

| file | 種別 | 役割 |
| --- | --- | --- |
| `scripts/vit_moco/query_collapse_diagnostic.py` | 新規 | 固定probeの選択と変換、`cls`/`z_q`/`q`の特徴統計、step0からのdrift、学習stateのhash比較、step1の更新量計測、rank協調の失敗処理、`diagnostic_summary.json`の生成、2 runの比較CLI |
| `conf/moco_vit_query_collapse.yaml` | 新規 | 診断用preset。`defaults: [moco_vit_ddp_smoke, _self_]`で採用recipeをそのまま継承し、`diagnostics`と出力先`log/moco_vit_query_collapse/...`だけを追加 |
| `test/model/test_query_collapse_diagnostic.py` | 新規 | D1/D2のCPUテスト33件 |
| `scripts/vit_moco/moco_vit_ddp_smoke.py` | 変更（+31/−4行） | `diagnostics.enabled`のときだけ呼ばれるhookを追加（prepare_configの検査、probe_step 0、train_stepの前後計測、各step後のprobe、終了処理、summary）。fail-closedの対象に診断成果物を追加 |

変更していないもの：`self_supervised/moco/{standalone_moco,ddp_moco,vit_ddp_moco}.py`（loss・EMA・Queue）、`scripts/parity/*`、既存のconf（`moco_vit_ddp_smoke.yaml`を含む）、既存のtest。

## 実装の要点（specとの対応）

| spec | 実装 |
| --- | --- |
| B1 固定probe | ImageFolderのsample listから、専用の`torch.Generator().manual_seed(0)`による`randperm`の先頭64件を選び、昇順に並べる（indexの重複なし。global RNGには触れない）。変換は`pil_loader`（RGB）→`Resize(256)`→`CenterCrop(224)`→`ToTensor`→`Normalize`（checkpointのmean/std）で、random augmentationはない。画像はCPUに保持し、各rankで同じものを作る |
| B2 観測点 | `cls = backbone_q(x)`、`z_q = projector_q(cls)`、`q = F.normalize(z_q, dim=1)`。probe_step 0は最初のoptimizer stepの前、1/2/10はその回数の`optimizer.step()`の直後。`MoCo.forward`、DDP forward、学習用loaderは呼ばない |
| B3 互換性 | presetで明示的に有効にしたときだけ動く。無効時は余分なforwardも新しいファイルもない。MoCoコアには介入しない。出力先は別namespace |
| B4 更新量 | step1で、Full FTは`backbone`/`projector`、LoRAは`lora_A`/`lora_B`/`projector`に分けて、grad norm、更新前後のparameter norm、update norm、`relative_update = ‖Δθ‖ / max(‖θ_before‖, 1e-12)`を記録する。更新前のparameterはCPUへcopyし、GPU上での二重cloneはしない |
| B5 統計 | `mean_dim_std`、`median_dim_std`（補間した中央値）、`row_norm_min/median/max`、異なる画像同士のcosine（対角を除く）の`mean/p50/p95`、`finite`、`shape`、`zero_norm_count`を求める。統計はCPU上でfloat64で計算する。未定義の値はNaNにせず、`None`にして理由を`undefined`に残す。`uniformity_warning`（`q.mean_dim_std < 1e-4`かつ`offdiag p50 > 0.999`）は**qだけに付く警告候補**であり、判定ではない |
| §4 時点間指標 | 同じindex順で、step0との画像ごとのcosineとL2距離の分布（mean/min/p50/p95/max）を出す |
| §5 副作用なし | probeの前後で、model state（Query/Key/Queue/pointer）、AdamWのstateとgroup、全RNG、moduleのtrain/eval flagをhashで比べる。1つでも変われば、その時点で全rankを停止する（FAIL）。flagとRNGはprobe内で必ず元に戻す |
| §5.5 入力の一致 | probe入力のhashを毎回計算し直して、manifestと照合する。モード間の比較（学習viewのsha、probe入力、step0特徴のbitwise一致）は比較CLIで行う |
| §7 成果物 | `probe_manifest.json`、`probe_metrics.jsonl`（step×rank×stageごとに1行）、`update_metrics.jsonl`（rank×groupごとに1行）、`diagnostic_summary.json`（PASS/BLOCKED、理由、rank0の時系列、警告、probe時のpeak/headroom、未成立の主張）。既存の`run_report.json`、`rank*_report.json`、`metrics.jsonl`の形式は変えていない（rank reportには、診断を有効にしたときだけ`diagnostics`キーが加わる） |
| §9 資源 | probeはmicrobatch 8で`no_grad`。GPU runは採用値（64枚、seed0、resize256、microbatch8、probe_steps [0,1,2,10]、update_steps [1]）と`max_steps=10`以外を拒否する。resumeとActivityNetの指定も拒否する |
| 停止 | probeのsetupやforwardで1 rankだけが失敗しても、全rankが例外で止まる（rank単独のhangを避ける）。OOMを起こしたrankは元の例外型のまま記録される |

比較CLI：`python -m scripts.vit_moco.query_collapse_diagnostic <lora run> <full_ft run> [--output new.json]`。出力先の既存ファイルは上書きしない。

## Gate別の結果

| Gate | 結果 | 観測 |
| --- | --- | --- |
| D0：計測位置の静的監査 | PASS | 現行コードの`ViTMoCoBackbone.forward`＝`last_hidden_state[:, 0, :]`、`StandaloneMoCo._project`＝`projector(backbone(x))`、`DistributedMoCo.forward`の`q = F.normalize(z_q, dim=1)`と同じ順序で、Query側だけを直接呼ぶ。Keyや`encode_key`は使わない。probe_step 0は最初の`train_step`の前、n≥1は`optimizer.step()`→`_sync`の後で、既存の`metrics.jsonl`（更新前forward）とは別ファイル・別の時点として記録する |
| D1：CPU unit | PASS | 人工Xで確認：全行一致（std 0、cosine 1、警告True）、正規直交（std 0.5、cosine 0、警告False）、対角の除外、補間中央値（[1,2,3,10]→2.5）、スケール（stdは比例、cosineは不変）、ゼロ行（件数を数え、cosineから除外。全行ゼロなら`None`）、NaN/Inf（全統計を`None`にして理由を残す）、N=1と不正shape。drift（自分自身でcos 1、符号反転で−1、ゼロ行と非有限値）、probe選択（決定的・重複なし・昇順・global RNG不変）、probe変換（グレースケールとRGBAをRGB化し、32pxに切り出し、hashを再現） |
| D1：1 rank（gloo、実ViT構造のtiny model） | PASS | ・probe出力は`backbone_q`/`projector_q`/`encode_query`の全batch計算と一致（atol 1e-6）。flagの混在パターン、RNG、学習stateは不変。<br>・step1の更新量：LoRA Aはgrad 0なのに`relative_update≈lr·wd=3e-6`（decoupled weight decayだけによる更新。**grad normとupdateは別物**であることを確認）。LoRA Bは更新前のnorm 0で、relativeはε基準。Projectorのupdate normは独立に計算した値と一致。<br>・parameterを変えるprobe、RNGを消費するprobeを差し込むと、どちらも`changed the training state`で停止した。<br>・Projectorの最終層を定数出力にしたfixtureでは、`cls`はstd>0のまま、`z_q`/`q`はstd 0・cosine 1・警告Trueとなり、**崩れた段だけを特定できる**ことを確認 |
| D2：CPU tiny ViT／2-rank gloo（runner） | PASS | ・両モードでprobe_step 0/1/2/3（max_steps 3、epoch境界を含む）を記録。24行＝4 step×2 rank×3 stageの欠落なし。probe入力のhashは全行1種類でmanifestと一致。rank間の特徴は一致。全probeで`state_unchanged=True`。summaryは`PASS`。<br>・**診断の有無で学習traceがbitwise一致**：両モードとも`metrics.jsonl`（step_time以外）、final state hash、config identityが一致。診断OFFのrunには診断成果物も`diagnostics`キーもない。<br>・比較CLI：両モードの学習view、probe入力、**step0のcls/z_q/q特徴がbitwise一致**（`comparable=True`）。step1のqはモード間で異なる。<br>・既存のG3 run dir、診断run dir、`probe_manifest.json`だけがあるdirへの出力は、どれも`FileExistsError`で拒否し、何も書かない。<br>・probe数がdataset数を超えると、両rankがstage `probe_setup`で停止し、summaryは`BLOCKED`になる。<br>・1 rankだけ失敗した場合も、もう一方のrankが`RuntimeError`で停止する（hangしない） |
| D2：config契約 | PASS | presetは`diagnostics`とhydraの出力先以外がbase presetと完全に一致（研究条件は不変）。GPUの採用値と`max_steps=10`以外は拒否（6項目を個別に変えて全て拒否）。resume、ActivityNet、範囲外や順序違いのprobe_steps、update_steps 0、resize<cropを拒否。preset以外の設定で`diagnostics.enabled`だけを付けた場合も拒否 |
| D3：既存回帰 | PASS（新規の失敗なし） | 同じvenv・`CUDA_VISIBLE_DEVICES=""`で全テストを実行した。<br>・変更前（`2c26991`のclean clone）：565 passed / 197 failed / 18 errors（2026-10-21の記録と同数）。<br>・変更後：**598 passed** / 197 failed / 18 errors。増えた33件は新規テスト。<br>・**失敗・エラーのtest ID（215件）の集合は変更前後で完全に一致**した。内訳はGPU前提（`test_image_models`、`test_model_factory`、`test_videomodels`、`test_checkpoint`、`test/setup`）、NAS上のKinetics（`test_video_folder`）、`test_moco_5epoch_pair`の1件で、いずれも変更前から存在する。<br>・既存の`test_moco_vit_ddp_smoke.py`、`test_vit_moco_ddp.py`（G0〜G4）、parity関係のテストに退行はない。テスト実行後の研究repoに、意図しない生成物は残っていない |

### 追加確認

1. **診断OFFは変更前のコードとbitwise一致**：HEAD（`2c26991`のclean clone）と変更後のコードで、同じCPU smoke（両モード、5 step、step3 checkpoint）を実行した。`metrics.jsonl`、final state hash、config identity、`resolved_config.yaml`、**checkpointファイルのバイト列**、report・rank reportのキー構成、成果物の一覧が両モードとも一致した。
2. **実CLI入口**：`python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_query_collapse ...`（CPU、tiny model、出力はscratchpad）で両モードとも完走し、summaryは`PASS`、probe 24行、update 4行（Full FT）。比較CLIは`comparable=True`。
3. lint：新規・変更ファイルにflake8のE/F系違反はない。runnerの既存違反5件は変更前と同じ。

## 実データCPU preflight（読み取りのみ・学習なし・GPUなし）

D4の前に、採用値のprobeが実データで作れるかを確認した。NASのImageNet trainを読み取りのみで使い、`google/vit-base-patch16-224`はローカルのHF cacheから読み込んだ（`local_files_only=True`。新規ダウンロードなし）。

- ImageFolderは**1,281,167枚・1000クラス**で、`samples_sha256=75cbd2fc0b4bf9c2f75c3f4e15c080534460f739ecb21aa5667a13688b642dd7`。ローカルにあるG3の1/10step run（LoRA・Full FT、計4本）の`run_report.json`の値と一致した。scanに70.5秒かかった。
- probeは64枚（**61クラス**、64枚ともtensor hashが異なる）。入力`[64, 3, 224, 224]`のsha256は`6af5074524242593f1b5fc90666080904af2fdc010b9809072fe0b479708d8bf`で、再読み込みしても同一だった。
- 実ViT-Bのprobe_step 0（CPU、microbatch 8）：**LoRAとFull FTでcls/z_q/qがbitwise一致**した。probe時間は約1.8秒/mode。
- probe_step 0の統計（両モード共通。参考値）：

| stage | mean_dim_std | median_dim_std | row_norm_median | offdiag cos p50 | offdiag cos p95 |
| --- | --- | --- | --- | --- | --- |
| cls [64,768] | 0.8460 | 0.8383 | 24.41 | 0.0319 | 0.1218 |
| z_q [64,128] | 0.1666 | 0.1644 | 2.266 | 0.3188 | 0.4488 |
| q [64,128] | 0.07263 | 0.07196 | 1.000 | 0.3188 | 0.4488 |

- qの`uniformity_warning`はFalse。G3のstep1 `query_feature_std=0.07224`（学習batchに対する更新前forwardの値）と同じ程度の大きさだが、**画像・時点・計算条件が異なるため、同列には比較しない**（spec §9）。
- コスト見積もり（CPU）：Full FTのmodel state（1.81億要素）のhashは約0.7秒、step1更新量のCPU snapshotは約0.5秒。GPU runでは、probe 1回ごとにhash 2回分（AdamW stateを含めると数秒）が加わる見込み。

## D4（GPU）に必要な前提と手順（未実行・別許可）

- **runnerはdirtyなrepositoryからのGPU runを拒否する**（既存の仕様）。そのため、D4の前に今回の差分のcommit（必要ならpush）が必要で、これにはユーザーの許可が要る。
- 手順案（B6の許可後に、研究サーバでGPU ID・空き・clean HEAD・データのpreflightを行ってから）：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_query_collapse \
    model.train_mode=lora training.max_steps=10
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_query_collapse \
    model.train_mode=full_ft training.max_steps=10
python -m scripts.vit_moco.query_collapse_diagnostic \
    log/moco_vit_query_collapse/lora/imagenet/<ts> log/moco_vit_query_collapse/full_ft/imagenet/<ts> --output <new.json>
```

- 合否の確認先：各runの`diagnostic_summary.json`の`engineering_status`（VRAM Gate、finite、probe入力の一致、state不変、全記録の有無を含む）と、比較CLIの`comparable`。

## 未確認点と次の判断

1. **GPUでの実測は未実施**。Full FTの10step headroomは1,749.875 MiBで余裕が少ない。probeはmicrobatch 8の推論だけなので学習時のpeakは超えない見込みだが、未測定である。probe前後のpeakは`rank*_report.json`の`diagnostics.probes`とsummaryの`probe_memory`に記録される。
2. GPU上でのstep0のモード間bitwise一致は未確認（CPUでは実ViT-Bで一致）。
3. probe時間、state hash時間のGPU実測は未確認。
4. Key/Queue由来の要因は、このprobe（Query側のみ）では排除できない（spec §4）。
5. D4の実施にはcommitとB6の許可が必要。低LR比較は別spec。

## 関連

- [承認済みspec](../specs/2026-10-22-vit-moco-query-collapse-diagnostic-spec.md)
- [G3 10step Evidence](2026-10-21-vit-ddp-moco-g3-imagenet-10step-console-evidence.md)
- [前段の実装・CPU検証記録](2026-10-21-vit-ddp-moco-full-ft-lora-implementation-cpu-verification.md)
- [Collapse診断の探索メモ](../../../../secretary/notes/brainstorm/2026-10-22-vit-moco-g3-collapse-diagnostic-roadmap.md)
