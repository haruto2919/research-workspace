---
date: 2026-10-07
project: sequential-video-lora-analysis
source_todo: null
topic: full-pipeline-throughput-optimization
status: exploratory
tags: [brainstorm, research, performance, moco, full-pipeline]
---

# Full pipeline throughput optimization

## 相談の出発点

`run_full_pipeline.sh` の長時間実行中、GPU監視では3枚中1枚だけが使用され、実行中GPUのutilizationは概ね50%前後、VRAM使用量は約3.3 GiBだった。worker数やGPU数を工夫して、研究条件を壊さず実行時間を短縮できるか検討する。

本メモは探索記録であり、specでも実装許可でもない。

## 読み込んだ文脈

- Project SSOT: `.research/lab/projects/sequential-video-lora-analysis/README.md`
- Related spec: `.research/lab/projects/sequential-video-lora-analysis/specs/2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md`
- Latest related verification: `.research/lab/projects/sequential-video-lora-analysis/experiments/2026-10-06-full-pipeline-hydra-config-verification.md`
- Latest meeting: `.research/lab/projects/sequential-video-lora-analysis/meetings/2026-09-24-mtg.md`
- Implementation: `tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@605f2afb95e1dda541208a46d69ce46acd83b335`
- Sequential Loader baseline: `tamaki-lab/2026_09_ishikawa_sequential_loader@ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0`
- 2026-10-07 の日次TODOファイルはResearch Workspace上に未作成。

## 確認済み事実

### 現行full pipeline

`run_full_pipeline.sh` は `CUDA_VISIBLE_DEVICES` に1枚のGPU IDだけを設定し、以下を直列実行する。

1. Full-dataset Stage 6B-v2 Streaming MoCo
2. ActivityNet manifest build / audit
3. Base ViT feature extraction
4. final Query LoRA feature extraction
5. Linear Probe（2 conditions x 3 seeds）とaggregate

feature extractionはBase ViTとLoRAを別プロセスではなく順番に実行する。Linear Probeの既定deviceはCPUで、`PROBE_DEVICE=cuda` は既存launcherから指定できる。

### 現行MoCo protocolの並列化制約

承認済みspecではfull-dataset MoCoを次で固定している。

- ActivityNet training 10,024 videosのdeterministic single pass
- `strict_single / gbr_horizontal_flip / all_past`
- single process
- single GPU
- one active video stream
- source orderと各動画内chunk orderを維持
- DDP / DataParallel / multi-GPU trainingはscope外

したがって、動画やchunkを複数GPUへ単純分割するdata parallelはQueue履歴、EMA、optimizer update orderを変え、現行protocolと同一ではない。

### Sequential Loaderのworker制約

consumer側 `integration/sequential_stream.py` は `ordered_samples` を「without prefetch」として実装し、`sl.build_sequential_dataloader(dataset=dataset)` を既定設定で呼ぶ。

Sequential Loaderの `DataLoaderConfig` 既定値は `strict=True, batch_size=1, num_workers=0, pin_memory=False, in_order=True` である。strict modeのvalidatorは `num_workers != 0` を明示的に拒否する。

したがって、現在のコードで単に `num_workers=4` や `8` に増やす方法は使えない。

### per-step audit

production full runはcanaryと同じaudited per-chunk engineを使う。各updateには少なくとも次が含まれる。

- Query / Key encode
- Queue audit
- backward / optimizer
- Query / Key parameter検査
- frozen Query/Key baseの不変性検査
- Key EMA式の検査
- 全model parameterのfinite検査
- Queue全体のfinite / norm / metadata / tensor一致検査
- gradient diagnostics

特に `audit_bases` はfrozen base全tensorを比較し、`audit_finite` はmodel全parameterを走査し、`audit_queue` は最大4096個のQueue keyをstackして検査する。これらがchunkごとに複数回呼ばれ、`.item()` や `torch.equal/allclose` によるCPU-GPU同期も発生する。

## 解釈

GPU utilization約50%かつVRAM使用量が小さい状況は、「GPUメモリ不足でbatchを増やせない」状態より、CPU decode / preprocessing、逐次処理、細かなGPU同期、per-step auditのいずれかでGPUが待っている可能性が高い。

現行protocolではbatch_sizeやframes_per_chunkを増やして複数chunkを1 updateにまとめると科学条件そのものが変わるため、まず非意味論的なoverheadを減らすべきである。

## 候補と評価

### A. per-step auditをproduction向けに間引く — 最有力

学習計算そのものを変えず、観測用auditの頻度だけを分離する案。

候補例:

- 最初の10〜20 update: full auditを毎step
- 通常step: loss/logits finite、必要最小限のQueue countなどだけ
- 100 updateごと、またはvideo boundary: LoRA / Projector finite、Queue normalization等
- resume checkpoint（現行100 videosごと）: full base immutability + full Queue + full finite audit
- evaluation snapshot / final: full audit

auditは本来parameterを変更しないため、正しく実装すればtraining stateは同一に保てる可能性が高い。ただし現行specのfail-fast範囲を変えるので、新しいproduction performance specとequivalence testが必要。

**優先度: 最優先。** 現行コードを読む限り、worker/GPU数の変更より先にbenchmarkすべき箇所。

### B. ordered decode/preprocess prefetch — 有力

PyTorch DataLoaderのworker数を直接増やすのではなく、1本の厳密なsource順を保持したまま「次のchunkだけ」をbounded queueへ先読みし、現在chunkのGPU計算と次chunkのdecode / CPU preprocessingをoverlapする案。

候補:

- prefetch depth 1〜2
- producerは順序を絶対に変更しない
- consumerが受け取る `SequentialSample` のsequence_id / sequence_index / frames / valid_maskを現行と完全一致させる
- pin_memory + non_blocking H2Dも検討

未解決点は、strict-onlineの研究定義として「decodeだけ未来chunkを先に読む」ことを許すかである。modelが未来chunkを観測・更新に使わなくても、protocol上のstrictnessをどう定義するか明文化が必要。

**優先度: 高。** audit削減後もGPU waitが残る場合の次候補。

### C. downstream feature extractionを複数GPUで並列 — 有力、MoCo終了後向け

Base ViT featureとfinal Query LoRA featureは同一manifestに対する独立conditionであり、現行launcherでは直列になっている。たとえば2つの独立processとして

- GPU 0: Base ViT feature
- GPU 1または2: Query LoRA feature

を同時実行できる余地がある。

注意点:

- 同じNASから同時に動画を読むため、storage I/Oがボトルネックなら逆効果になり得る。
- artifact path / Comet experimentはconditionごとに分離されているが、並列writeの安全性はtestで確認する。
- MoCoの科学条件は変えない。

**優先度: 高。** Pipeline全体のwall-clock短縮には有効だが、現在長時間かかっているMoCo stage自体は短縮しない。

### D. Linear ProbeをGPUへ — 既存機能で可能

launcherは `PROBE_DEVICE=cuda` を既に受け付ける。6 runは現状直列なので、GPU実行だけでも後段の短縮余地がある。

さらにcondition x seedは独立なので別process並列化も設計上は可能だが、MoCoより計算量が小さいため優先度は低い。

### E. Query / Keyを別GPUへ配置 — 保留

MoCoはQuery encoderとKey encoderを別々にforwardするため、

- Queryを高速GPU
- Keyを別GPU

に置いてforwardをoverlapし、128-d projected featureを集約するmodel-parallel案は理論上可能。

ただし各stepのEMAでQuery -> Key parameter同期が必要で、現行specのsingle-GPU条件も変更する。異種GPU間の性能差、通信、再現性の検証が必要なため、audit削減とprefetchより後に検討する。

### F. DDP / DataParallel / 動画分割 — 現時点では棄却

複数動画・chunkを並列学習するとQueue history、EMA、optimizer step順が変わる。現行Stage 6B-v2と同じ実験とは扱えない。速度目的だけで採用しない。

### G. `num_workers > 0` をそのまま設定 — 現時点では棄却

Sequential Loader strict contractが拒否する。IterableDatasetのmulti-worker化は、実装次第で重複・順序変更も生じ得る。worker数だけを変更する修正は行わない。

### H. frames_per_chunk / chunk batchを増やす — 棄却

1 updateあたりの観測単位、update回数、Queue履歴を変更するため、単なる高速化ではなくscience condition変更になる。

### I. AMP / bfloat16 / torch.compile — 保留

高速化余地はあるがnumericsや実装identityを変える。まずaudit / input pipelineのoverheadを測った後に判断する。

## 現在実行中のrunへの扱い

現在のcanonical full runを、速度改善のためだけに途中停止してコードを変更することは推奨しない。

resume identityにはrepository / branch / commit / dirty state、device identity、source order等が含まれる。現在のrunを別commitの高速化コードへそのままresumeすると、現行のexact-resume契約とは一致しない。

したがって、現在runは現行commitのまま完走させ、速度最適化は次run用に別途benchmarkするのが安全。

## 推奨するbenchmark順序

1. 現行コードで短い固定prefixについてstep時間を分解する。
   - decode
   - image processor
   - H2D
   - Query forward
   - Key forward
   - contrastive / backward / optimizer
   - audit
2. audit頻度だけ変えた実装で同じprefixを比較する。
3. training state / loss trajectory / Queue metadata / final stateが期待どおり一致するか確認する。
4. 次にordered prefetchを追加し、sample列が現行と完全一致することを確認する。
5. MoCoとは別にBase/LoRA feature extractionの2-GPU並列をbenchmarkする。

現在の長時間runと同じNAS / CPUを使って並列benchmarkすると本番run自体を遅くする可能性があるため、原則として現在run完了後に行う。

## 現在の方向性

最有力は「GPU枚数を増やす」よりも、まず **per-step correctness auditのproduction向け頻度設計** と **ordered decode/preprocess prefetch** を検証すること。

multi-GPUはMoCo data parallelではなく、後段の独立feature extractionの並列化から使うのが安全である。Query / Keyの2-GPU model parallelは追加の高速化候補として保留する。

## 未解決事項

- 実時間の内訳をまだprofileしていないため、audit、decode、processor、GPU forward/backwardのどれが支配的かは未確定。
- strict-online protocolでdecode-only prefetchを許容するか。
- Base / LoRA feature extractionを同時に走らせた場合、NAS帯域が十分か。
- Query / Keyを異なるGPUへ置いた場合のwall-clock効果と再現性。
- 現在実行中processがfull pipelineのどのstageにいるかはGitHubからは確認不能。

## 次アクション候補

- 現在runは止めずに完走させる。
- 完了後、固定ActivityNet prefixでper-stage/per-step profilingを行う。
- profileでaudit overheadが支配的なら、production audit policyの変更をspec化する。
- その後、ordered prefetchとdownstream 2-GPU feature extractionを段階的に評価する。


## 2026-10-07 14:42 JST 追記: 既存機能を壊さない高速化でspec前に決める事項

ユーザー方針は「既存の実行可能な機能・研究条件を壊さずに高速化する」。この方針では、単なる性能改善ではなく互換性契約を先に固定する必要がある。

### Spec前にblockingとして決める事項

1. **互換性境界**
   - 既存 `run_full_pipeline.sh` の呼び出し方を維持するか。
   - `fresh / resume / skip`、Stage 6A/6B smoke、既存Hydra preset、artifact、Comet naming/tag、checkpoint/resume semanticsのどこまでを完全維持対象にするか。
   - 推奨: public CLI、既存preset、科学条件、artifact schema、resume semanticsは変更しない。高速化は追加のoperational pathとして導入する。

2. **高速化のscope**
   - Phase 1でaudit overhead削減だけを行うか、prefetchまで同一specへ含めるか。
   - 推奨: 段階化する。まずprofiling + audit scheduling。効果確認後にordered prefetch。multi-GPU MoCo、AMP、batch/chunk条件変更は初期scope外。

3. **Audit policy**
   - 何を毎step残し、何を周期的にするか。
   - 推奨候補: 最初の20 updateはfull audit、通常stepは最小finite/counter check、100 updateごとにQueue/gradient/model audit、resume checkpoint・snapshot・finalでfull audit。
   - 現行full-audit modeは削除せず残す。

4. **Equivalenceの定義**
   - 「壊していない」を何で判定するか。
   - Audit schedulingだけの変更では、同一seed・同一device・同一sample列でloss trajectory、Query/Key LoRA、Projector、Queue、optimizer state、countersをexact一致させることを第一候補とする。
   - Prefetchでは少なくとも入力Sample列のbytes/order一致を必須とし、可能なら最終training stateもexact一致させる。

5. **既定動作とopt-in**
   - 既存利用者が何も指定しない時に旧挙動を維持するか。
   - 推奨: equivalenceとbenchmarkが通るまではlegacy/full-auditをdefault、optimized modeは明示opt-in。検証後にdefault変更を検討しても、legacy pathは残す。

6. **Benchmark protocolと合格基準**
   - 固定prefix、計測対象、反復数、性能指標、最低改善量を固定する。
   - 推奨: 同一ActivityNet prefix、同一seed/deviceで3回程度。decode / processor / H2D / Q forward / K forward / backward+optimizer / auditを分解し、updates/s、videos/min、wall-clock、GPU utilを記録する。
   - 高速化採用条件は「equivalenceを満たした上で、固定benchmarkで有意なwall-clock短縮がある」とする。具体的な改善率閾値はspec前に決める。

7. **Resume / artifact互換**
   - 高速化コードから旧commitの途中runをcanonical resumeしてよいか。
   - 推奨: cross-commit resumeは従来どおり禁止。既存runは元commitでresumeする。高速化でcheckpoint schemaを変えず、新コード内のfresh/resume契約を維持する。

8. **Prefetchのstrict-online境界**
   - CPU側で未来chunkをdecode/preprocessして待機させることをonline条件上許容するか。
   - 推奨: model/optimizer/Queueは未来chunkを一切参照せず、consumer順序が完全一致する限り、bounded decode-only prefetchをengineering optimizationとして許可する。
   - DataLoaderの `num_workers>0` を直接解放するのではなく、既存strict loader contractを維持する。

9. **GPU利用方針**
   - MoCo本体をmulti-GPU化するか、後段だけ並列化するか。
   - 推奨: MoCoはsingle GPU維持。Base/LoRA feature extractionやProbeの独立jobで複数GPU利用を検討する。異種GPU間の数値差とNAS I/O競合をbenchmarkする。

10. **回帰Gateとrollback**
    - どのtestをPASSすれば既存機能を壊していないとみなすか。
    - 推奨: 全existing regression + 新規legacy-vs-optimized equivalence + short ActivityNet smoke + resume equivalence + artifact/Comet契約test。
    - optimized modeで異常が出た場合に自動的に別条件へ切り替えずfail-fastし、legacy pathへ明示的に戻せる構成にする。

### 推奨する収束方向

最初の高速化specは、変更範囲を **「既存科学条件・loader contract・public CLIを維持したproduction runtime最適化」** に限定するのが安全。

第一段階の主対象は次の2つ。
- per-step auditの頻度設計とGPU同期削減。
- その効果を測るprofiling / equivalence infrastructure。

ordered prefetch、downstream multi-GPU、Query/Key model parallelは、第一段階のEvidenceを見て後続scopeへ分ける方が、原因切り分けと回帰検証が容易。

### まだユーザー判断が必要な項目

- legacy/full-auditを当面defaultにするか。
- optimized auditの具体頻度（例: first 20 / every 100 updates / checkpoint boundaries）。
- strict-onlineでdecode-only prefetchを許容するか。
- benchmark採用の最低速度改善率を何%にするか。
- 第一specにprefetchまで含めるか、audit最適化だけに限定するか。


## 2026-10-07 14:xx JST 追記: 「現在のrun_full_pipeline.shと同じ実験」を高速化後も成立させる

ユーザー要件を強化し、高速化後も「現在の `run_full_pipeline.sh` を実行した場合と同じ科学実験」とみなせることを最優先とする。

### 同じ実験として固定するScience Contract

高速化前後で少なくとも以下を変更しない。

- ActivityNet split / source集合 / source順。
- 各動画内のchunk順、`frames_per_chunk`、valid frame。
- Stage 6B-v2 protocol: `strict_single / gbr_horizontal_flip / all_past`。
- warm-up位置。
- Query / Key encoder、LoRA target / rank / alpha / dropout。
- projector構造。
- Query / Key forwardへ与えるtensor値と順序。
- InfoNCE、negative selection、Queue capacity / enqueue順。
- backward、optimizer class / hyperparameters / optimizer.step回数と順序。
- EMA式とEMA実行位置。
- seedとRNG消費順。
- dtype / precision（初期高速化ではAMP/BF16/TF32等を追加しない）。
- checkpoint / resume state。
- final Query LoRA snapshotの定義。
- manifest、feature definition、Linear Probe条件。
- artifact内容・schemaとComet上の科学的比較軸。

### 変更してよいOperational Contract

科学結果へ影響しないことをequivalence testで証明する前提で、以下だけを高速化対象候補とする。

- correctness auditの頻度 / 実行タイミング。
- profiling instrumentation。
- loggingのbuffering / flush頻度（artifact内容を変えない）。
- CPU decode / preprocessingのordered prefetch（consumerに渡るsample bytes/order/RNG消費が同一であることが条件）。
- H2D overlap / pinned memory / non-blocking copy（encoder入力tensor値が同一であることが条件）。
- MoCo終了後の独立Base/LoRA feature extractionの並列実行（feature artifact bytes/value equivalenceを確認する）。

### 初期scopeから除外

「同じ実験」を最優先するため、初期高速化では以下を採用しない。

- DDP / DataParallelによるMoCo data parallel。
- 動画 / chunkの分割並列学習。
- batch size / frames_per_chunk変更。
- Query / Keyを別GPUに置くmodel parallel。
- AMP / FP16 / BF16。
- TF32設定変更。
- optimizer / fused optimizer変更。
- `torch.compile`（数値・kernel・再現性を別途検証するまで保留）。
- augmentation / processor変更。
- science config / protocol version変更。

### Equivalence Gate

audit schedulingだけの最適化では、同一環境・seed・固定prefixについて、legacyとoptimizedで以下のexact一致を要求する。

- ordered sample identity。
- per-step loss / logits（可能な範囲でexact）。
- Query / Key LoRA。
- Query / Key Projector。
- optimizer state。
- Queue key / metadata /順序。
- counters。
- resume checkpoint payloadの科学state。
- final Query LoRA snapshot。

prefetch / transfer overlapを導入する場合も、まず入力sample/tensorをexact比較し、その後training stateまで同等性を確認する。

GPUやdependencyが異なる場合のbitwise一致は別問題なので、「同じ実験」の正式比較は同一hardware / software baseline上のlegacy vs optimizedで判定する。

### Provenanceの扱い

高速化後はcode commitが異なるため、implementation provenanceは異なる値として正直に記録する。一方、science contractのhash / protocol / dataset / seed / hyperparametersは同一に保つ。

つまり「同じ実験」は「同じcommit」という意味ではなく、**科学的入力・状態遷移・出力定義が同じで、違うのは非意味論的runtime実装だけ**と定義する。

### 推奨する第一段階

第一高速化specは次だけを対象にする。

1. profiling instrumentation。
2. audit scheduling / synchronization削減。
3. legacy modeを保持。
4. legacy vs optimized exact-equivalence test。
5. 固定prefix benchmark。

この段階で十分な速度改善が得られればprefetchは追加しない。改善不足の場合のみ第二段階としてordered prefetchをspec化する。
