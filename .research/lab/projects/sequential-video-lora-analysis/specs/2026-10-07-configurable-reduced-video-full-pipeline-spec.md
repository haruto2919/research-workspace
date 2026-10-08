---
project: sequential-video-lora-analysis
record_type: implementation-spec-addendum
status: implemented
created: 2026-10-07
last_updated: 2026-10-08
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
implementation_branch: dev
implementation_base_commit: 605f2afb95e1dda541208a46d69ce46acd83b335
parent_spec: 2026-10-05-full-dataset-streaming-moco-linear-probe-spec.md
related_specs:
  - 2026-10-06-comet-tag-experiment-naming-spec.md
sequential_loader_repository: tamaki-lab/2026_09_ishikawa_sequential_loader
sequential_loader_branch: ActivityNet
sequential_loader_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
---

# Configurable reduced-video full pipeline addendum

## 1. Statusと目的

本書は、ActivityNet training 10,024動画を固定した既存approved specに対し、
動画数を設定だけで変更できる縮小版end-to-end pipelineを追加するためのimplementation addendumである。

解決する問題:

- 10,024動画のMoCo学習とfull downstream evaluationは所要時間が長い。
- runtime.stop_after_videosはpause/resume確認用であり、縮小集合を完走したfinal runではない。
- 現在のpartial manifestはsmoke用で、run_full_pipeline.shのproduction Gateを通らない。
- 動画数を変更しても、動画集合、Base/LoRA比較、Comet、resume、skip、artifact再利用を混同しない必要がある。

本書が目指す利用形:

- コードを編集せず、Hydraのsource-selection profileまたはoverrideで動画数を変更できる。
- 同じselection設定では毎回同じ動画集合を使う。
- 動画数またはselection seedが変われば、別の科学条件・artifact identityとして扱う。
- MoCo初期化seedを変えても、selection設定が同じなら動画集合は変わらない。
- Base ViTとMoCo Query LoRAは同じmanifest、segment、順序で比較する。

2026-10-07の明示的な実装依頼により、Section 13の推奨案を採用してapproved scopeを実装した。
commit、push、実ActivityNet end-to-end run、GPU長時間runは本依頼の許可範囲に含めない。

## 2. Authorityと現在状態

Authority順:

1. 2026-10-07までの現在のユーザー指示。
2. 親spec「Full-dataset single-pass Streaming MoCo + ActivityNet segment Linear Probe Spec」。
3. 「Comet Tag Taxonomy and Experiment Naming Spec」。
4. Research Workspaceの検証記録。
5. tamaki-lab/2026_09_ishikawa_sequential-video-lora dev branchの現在実装。
6. exploratory brainstorm。brainstorm単独では実装Authorityにしない。

現在の実装基準:

- implementation: tamaki-lab/2026_09_ishikawa_sequential-video-lora@dev@605f2afb95e1dda541208a46d69ce46acd83b335
- Sequential Loader: tamaki-lab/2026_09_ishikawa_sequential_loader@ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
- ActivityNet Adapterはfull v1.3 inventoryを検証し、training 10,024、validation 4,926をnormalized video ID順で返す。
- full MoCo CLIは10,024 sourcesを検証してからmodelを構築する。
- runtime.stop_after_videosはvideo boundaryでlatest.ptを保存し、status=pausedで返す。N本をfinalとは扱わない。
- stop_after_videos付きMoCo experimentはComet上でsmokeになる。
- Linear Probe manifestのruntime.max_videos_per_splitは先頭N本を使うsmoke経路である。
- run_full_pipeline.shはproduction manifestとGate PASSを要求し、manifest IDをlp-v1へ固定している。
- resume identityはsource countとordered source SHA-256を照合する。
- GitHubから研究サーバ上のdirty state、dataset実体、現在実行中process、既存local artifactは確認できない。

関連する2026-10-07 throughput optimization brainstormはaudit頻度、prefetch、multi-GPUを扱う。
本addendumはdataset selectionの変更だけを扱い、速度最適化を混在させない。

## 3. Decision Contract

### 3.1 採用決定

本実装では次を採用する。

1. ActivityNet Adapterとfull inventory Gateは変更しない。
2. Adapterがfull sourcesを返した後、consumer側の共通source-selection helperでsubsetを選ぶ。
3. selectionは新しい科学設定group source_selectionで管理する。
4. selection seedはMoCo初期化seedから分離する。
5. training selectionはMoCo学習とLinear Probe training manifestで共有する。
6. validation selectionはLinear Probe validation manifestで使用する。
7. selected setはseeded SHA-256 rankingで決め、処理順は元のAdapter順を維持する。
8. selectionのexact ordered IDs、count、SHA-256をlocal metadataとartifactへ保存する。
9. 任意overrideはsmoke、version管理されたapproved profileは追加Gateを満たした場合だけproduction候補とする。
10. 既存10,024-video Stage 6B-v2 pathは変更せず維持する。

### 3.2 明示的対象外

- ActivityNet Adapter、Sequential Loader core、video reader、chunkingの変更。
- 10,024-video既存runの意味、artifact、resume checkpointのmigration。
- runtime.stop_after_videosをsubset sizeとして再利用すること。
- first N videosを正式なselection policyとして採用すること。
- MoCoとLinear Probeで別々のtraining subsetを使うこと。
- class coverage Gateをproduction縮小runのために緩和すること。
- missing/decode failure sourceを自動skipして完了扱いすること。
- audit頻度、prefetch、AMP、frames_per_chunk、GPU並列、worker数の変更。
- full run、GPU長時間run、tmux、server process管理。
- コードのcommit、push、PR。

## 4. Source-selection configuration

### 4.1 Config group

新しいHydra groupを追加する。

想定例:

~~~yaml
# conf/source_selection/activitynet_reduced_v1.yaml
id: activitynet-reduced-v1
strategy: sha256_rank_preserve_adapter_order_v1
selection_seed: 0
splits:
  training:
    count: 1000
  validation:
    count: 1000
~~~

必要field:

- id: 人間がprofileを識別するversion付きID。
- strategy: selection algorithmのversion。
- selection_seed: source集合だけを決める符号なし32-bit整数。
- splits.training.count: 1以上10,024以下。
- splits.validation.count: 1以上4,926以下。

既存full pathは互換用profileとして次を表現できるが、現在のStage 6B-v2実装とartifactを変更しない。

~~~yaml
id: activitynet-full-v1
strategy: all_sources
selection_seed: null
splits:
  training:
    count: 10024
  validation:
    count: 4926
~~~

### 4.2 Seed separation

- runtime.seed: MoCo model/projector初期化とtraining RNGのseed。
- source_selection.selection_seed: 動画集合のseed。
- Linear Probe seeds 0, 1, 2: classifier初期化とtrain shuffleのseed。

三者を相互流用しない。

これにより、同じ動画集合でMoCo seedだけを変える比較、または同じMoCo seedで動画集合だけを変える比較を明示できる。

### 4.3 Deterministic selection algorithm

strategy=sha256_rank_preserve_adapter_order_v1は次で固定する。

1. Adapterからsplitのfull ordered source listを取得する。
2. full source count、unique ID、full ordered source SHA-256を検証する。
3. 各sequence_idに対し、strategy version、selection seed、split、sequence_idを区切り付きUTF-8 bytesへcanonical化する。
4. canonical bytesのSHA-256 digestをselection scoreとする。
5. digest、sequence_idの順で安定sortし、count件のID集合を選ぶ。
6. 実際の処理列はscore順にせず、元のAdapter順でfilterする。
7. selected ordered IDsのcanonical JSON SHA-256を計算する。
8. selected IDs、full hash、selected hash、count、strategy、seedを保存する。

Python random.sample、NumPy RNG、filesystem列挙順だけには依存しない。
同じ入力とprofileから同じID集合が得られない場合はfail-fastする。

### 4.4 Selection identity

再現可能なstable selection identityの必須項目:

- schema/version。
- profile id。
- strategy/version。
- selection seed。
- dataset name/version。
- splitごとのfull source count/hash。
- splitごとのselected source count/hash。
- splitごとのordered selected IDs。
- Sequential Loader branch/commit。
- annotation SHA-256。

stable selection identityのcanonical SHA-256をselection_sha256として下流artifactへ伝播する。
同一入力とprofileから同一SHAを再生成するため、wall-clock creation timeはこのhashへ含めない。

selectionを作成した実行の運用記録は、stable identityと分離したselection_recordとして併置する。
selection_recordの必須項目はschema、creation time、実装provenanceである。
source selectionを持つmanifestと新規snapshotではselection_recordも必須検証し、feature lineageへ伝播する。

## 5. MoCo contract

### 5.1 Selected single pass

縮小runではselected training sourcesだけをexactly once処理する。

- 動画内chronological orderを維持する。
- 動画間はselected setを元のAdapter orderで処理する。
- 最初のselected動画の最初のchunkだけKey-only warm-upを行う。
- 動画境界でQuery/Key/optimizer/EMA/Queue/global stepを保持する。
- selected training countを完了した時点でfinal=Trueとする。
- stop_after_videosが指定された場合はselected countに関係なくpauseとし、未完了ならfinal snapshotを作らない。
- sourceやchunkのfailureをskipせず停止する。

### 5.2 Existing full path

ActivityNet training 10,024本の既存Stage 6B-v2 path、protocol version、Comet experiment、artifactは変更しない。
縮小runは別protocol identityとして扱い、既存full resultと混在させない。

縮小runの候補protocol:

- config name: stage6b_subset_v1
- protocol version: activitynet-selected-single-pass-streaming-moco/v1
- Comet display tag: stage6b-subset-v1

MoCoの数値条件であるstream mode、key transform、negative policy、model、LoRA、Queue、
momentum、temperature、optimizerは既存Stage 6B-v2と同一に保つ。
scientific deltaはsource selectionだけとする。

### 5.3 Snapshot and checkpoint

final evaluation snapshot:

- selected training count完了時だけfinal=True。
- directory名は現在のvideos-<count>_step-<step>_final形式を維持できる。
- metadataへselection identityとselection_sha256を追加する。

resume checkpoint:

- selected ordered source list/hashとselection profileをidentityへ含める。
- count、selection seed、strategy、selected hashのいずれかが異なる場合はresumeを拒否する。
- completed video boundaryからのみresumeする。
- 既存latest.ptのatomic write/self-hash contractを維持する。
- evaluation snapshotからresumeしない。

## 6. Linear Probe and comparison contract

### 6.1 Shared subset

同じsource_selection profileから次を得る。

- training selection: MoCo pretrainingとProbe training manifestで共有。
- validation selection: Probe final validation manifestで使用。
- Base ViTとMoCo Query LoRAは同じmanifest、chunk indices、segment IDs、label mappingを使う。

Base/LoRAの差はfinal Query LoRAだけに限定し、selectionをconditionごとに変えない。

### 6.2 Manifest identity

manifest IDを固定lp-v1だけにしない。
少なくともLinear Probe protocolとselection_sha256から内容衝突しないIDを導出する。

例:

~~~text
lp-v1__sel-<selection_sha256先頭12桁>
~~~

metadataへprofile id、selection_sha256、split別selected count/hashを保存する。

既存runtime.max_videos_per_splitはlegacy smoke utilityとして維持してよいが、
approved reduced profileのproduction経路には使わない。

### 6.3 Dataset Integrity Gate

production候補の縮小manifestでも既存checks 1-9をすべて要求する。

特に:

- training 200 labels。
- validation 200 labels。
- 各classのtraining sampleが1件以上。
- 各classのvalidation sampleが1件以上。
- duplicate/overlapなし。
- chunk consistency。
- mapping hash一致。
- regeneration SHA-256一致。

coverageを満たさないprofileはproductionへ昇格しない。
任意overrideのsmokeでは現在と同様、coverage checksを記録しつつSMOKE_PASSを許可してよい。
不足classを埋めるための自動resample、seed変更、count増加を同一run内で行わない。

### 6.4 Feature and result identity

manifest、Base features、LoRA features、Probe results、aggregateへselection_sha256を含める。
異なるselection identityのartifactを同一ID・同一directory・同一Comet artifact versionとして上書きしない。

## 7. Comet contract

既存の役割分担を維持する。

- tags: UI filter用の低カーディナリティ軸。
- experiment name: 人間が役割を識別する表示。
- parameters/metadata: 厳密なidentityとlineage。

縮小MoCoの候補:

~~~text
name:
stage6b-subset-v1__moco__<run-id>

production tags:
('moco', 'production', 'stage6b-subset-v1', 'seed-<moco-seed>')

smoke tags:
('moco', 'smoke', 'stage6b-subset-v1', 'seed-<moco-seed>')
~~~

count、selection seed、selection SHA、run_id、commitは新tagへ追加せずparameters/metadataへ記録する。
seed tagはMoCo runtime seedだけを表す。

production候補条件:

- version管理されたapproved source-selection profileと完全一致する。
- scientific configがapproved presetと一致する。
- stop_after_videos is None。
- clean checkout。
- selected source identityの全検査成功。
- downstream manifestではIntegrity Gate PASS。

Hydraでcount/selection seed/strategyを任意overrideしたrunはsmokeとする。
resume時は保存済みmoco_experiment_keyへExistingExperimentで再接続し、tagを再追加しない。
既存Comet experimentsのtag/nameをmigrationしない。

## 8. Launcher contract

run_full_pipeline.shへsource-selection profileを一つだけ渡す。

後方互換候補:

~~~text
bash run_full_pipeline.sh <root> <run_id> <moco_seed> [gpu_id] [moco_mode] [source_selection_profile]
~~~

- source_selection_profile省略時は既存full pathを維持する。
- launcherは同じprofileをMoCo、manifest、feature extraction、Probeへ渡す。
- profile override値をstageごとに別指定しない。
- manifest IDはselection identityから解決する。
- freshは既存run directoryを上書きしない。
- resumeはcheckpointとprofile identityを照合する。
- skipはfinal snapshotのselection identityを、要求profileとmanifestより前に照合する。
- profile不一致時に近いartifactや別runをfallback利用しない。
- skipはMoCo学習だけを省略し、snapshot/manifest/feature identity Gateを省略しない。

## 9. Config and code responsibility

想定追加:

~~~text
conf/source_selection/
integration/activitynet_source_selection.py
~~~

想定更新:

~~~text
conf/moco_full.yaml
conf/linear_probe_manifest.yaml
conf/linear_probe_features.yaml
conf/linear_probe_run.yaml
training/moco_config.py
training/streaming_moco_full.py
training/moco_checkpoint.py
scripts/moco/train_full_streaming_moco.py
scripts/linear_probe/configuration.py
scripts/linear_probe/build_manifest.py
scripts/linear_probe/extract_features.py
scripts/linear_probe/run_probe.py
run_full_pipeline.sh
relevant tests
~~~

責務:

- Adapter: full inventoryとdataset固有source discovery。変更しない。
- shared selection helper: full sourcesからdeterministic subsetとidentityを作る。
- MoCo CLI: profile解決、full inventory検証、selected sources作成、metadata/Comet scope。
- training engine: 渡されたselected sourcesのsingle pass、checkpoint/snapshot。
- manifest: 同じselection identityを使用。
- feature/probe: manifestとupstream identityを検証。
- launcher: 一つのprofileを全stageへ伝播。

具体的なprivate function名、module内helper分割、JSON indentは既存styleに従う。

## 10. Verification requirements

### 10.1 Selection unit tests

- 同じfull IDs、strategy、seed、countから同じselected IDs/hashを得る。
- input tupleを再生成しても同じ結果になる。
- strategyまたはseedまたはcount変更でidentityが変わる。
- selected set決定後の処理順がAdapter順である。
- duplicate ID、invalid count、unknown profile、missing ID、hash mismatchを拒否する。
- full inventory Gateがselectionより先に実行される。
- MoCo runtime seed変更でselection identityが変わらない。

### 10.2 MoCo tests

- N selected videos完了でfinal=Trueになる。
- stop_after_videos < Nはpausedでfinal snapshotなし。
- stop_after_videos >= Nでも、actual finalはselected source exhaustionで決まる。
- first selected videoのfirst chunkだけwarm-up。
- Queue/optimizer/EMAが動画境界で保持される。
- snapshot metadataとresume identityにselection情報がある。
- changed count/seed/strategy/hash/profileでresumeを拒否する。
- uninterrupted runとvideo-boundary resumeの最終stateが一致する。
- existing 10,024-video Stage 6B-v2 testsが回帰しない。

### 10.3 Manifest and evaluation tests

- MoCo training selectionとmanifest training selectionのordered hashが一致する。
- Base/LoRAが同一segment IDs/orderを使用する。
- selection変更でmanifest/feature/result IDが変わる。
- production candidateはcoverage不足でFAILする。
- smoke overrideはcoverage不足を記録しSMOKE_PASSにできる。
- final LoRA snapshotとmanifest selection mismatchを拒否する。
- artifact reuseはselection identity完全一致時だけ成功する。

### 10.4 Comet and launcher tests

- version管理profile + no stopのfresh runが想定scopeになる。
- arbitrary count/seed overrideとstop_afterはsmoke。
- seed tagがMoCo seedを表しselection seedを表さない。
- resumeがsaved Comet experiment keyを継続する。
- skipがprofile/hash mismatchを拒否する。
- source_selection_profile省略時に既存full launcher contractを維持する。

## 11. Success Criteria

実装成功:

1. コード変更なしでprofile選択またはHydra overrideによりtraining/validation動画数を変更できる。
2. 同じselection設定は同じ動画集合とselection SHAを再生成する。
3. selection seedとMoCo seedが独立している。
4. MoCoとProbe trainingが同じordered training subsetを使う。
5. Base/LoRAが同じmanifest・segments・順序を使う。
6. selected N本完了時に縮小runのfinal snapshotを作れる。
7. resume/skipがselection mismatchをfail-fastする。
8. artifact identityとdirectoryがselection変更で分離される。
9. Comet上でcanonical full、approved reduced、arbitrary smokeを混同しない。
10. production候補では既存200-class Integrity Gateを維持する。
11. existing 10,024-video Stage 6B-v2、stop_after smoke、full launcherが回帰しない。
12. relevant unit/integration/CLI testsが全件PASSする。

科学的成功は実装成功と分離する。

- 縮小MoCo-LoRAがBase ViTを上回る、同等、下回るのいずれも研究結果である。
- Nが小さくcoverage Gateを通らない場合も報告対象であり、同一profileを黙って変更しない。
- 縮小結果を10,024-video full-dataset結果として主張しない。

## 12. Compatibility and migration

- 親specと既存10,024-video full pipelineをsupersedeしない。
- Stage 6B-v2のprotocol、artifact、Comet experiment、resume checkpointをmigrationしない。
- ActivityNet Adapter、Sequential Loader core、classification pathを変更しない。
- existing local artifactを本変更だけで削除・上書きしない。
- new selection metadata/schemaにversionを付ける。
- unsupported/missing selection schemaをproductionで黙ってfull扱いしない。
- legacy max_videos_per_split smokeは必要なら維持するが、新production縮小profileへ自動昇格しない。
- 現在実行中の既存full runを新コードへ跨いでresumeしない。

## 13. Decision Record

### Approved decisions

2026-10-07の明示的な実装依頼を承認として扱い、次の推奨案を採用した。

1. selection policy:
   - 推奨: SHA-256 rankingで集合を選び、Adapter順で処理する。
   - 代案: first N。簡単だがID順の偏りがあり、正式profileには非推奨。
   - 代案: class-stratified selection。coverageを作りやすいがlabel依存のsampling biasと追加複雑性がある。

2. Comet scope:
   - 推奨: 任意overrideはsmoke、version管理されたapproved profileだけproduction候補。
   - 代案: 全縮小runをsmoke。単純だが正式比較用runを表現できない。
   - 代案: 任意Nをproduction。再現可能性とcanonical presetの境界が弱くなるため非推奨。

3. protocol identity:
   - 推奨: 既存Stage 6B-v2を維持し、縮小runはstage6b-subset-v1として分離する。
   - 代案: stage6b-v2 tagを共有しselection metadataだけで区別。UI上の誤比較リスクが高いため非推奨。

具体的なtraining/validation countとselection seedは、設定可能性そのものの実装にはblockingではない。
最初に正式profileへ採用する値は、実ActivityNetで200-class Gateを監査した後に別途確定できる。

### Non-blocking

- shared helperの具体的module/function名。
- profile YAMLとselected IDs metadataのfield並び。
- JSONのindent。
- selection SHA短縮表示の桁数。ただし完全SHAはmetadataへ保存する。
- test fixture名とprivate helper分割。
- bashの第6 positional argumentか環境変数か。既存5引数の挙動を維持する方を選ぶ。

## 14. Execution boundary

2026-10-07の実装依頼により、approved scopeの研究コード変更と次の短時間検証を許可対象とした。

許可対象の短時間検証:

- selection unit tests。
- synthetic multi-video unit/integration tests。
- CLI config composition。
- Comet disabled/mock tests。
- existing regression tests。
- 少数動画のsmokeとresume equivalence。実データを使う場合も長時間runにしない。

別途明示承認が必要:

- 研究コードのcommit、push、PR。
- 実ActivityNet縮小end-to-end run。
- 10,024-video full run。
- full feature extractionと6本のProbe。
- GPU長時間run、tmux、server process管理。

## 15. Spec Gate

現在statusはimplemented。

理由:

- 2026-10-07の実装依頼によりSection 13の推奨案が承認された。
- source-selection profile、deterministic selection identity、resume/skip Gate、artifact lineage、Comet scope、launcher伝播をapproved scopeどおり実装した。
- relevant unit/integration/CLI testsと既存回帰テストを実施した。
- 実ActivityNet end-to-end run、10,024-video full run、GPU長時間runは未実施であり、科学的成功判定には含めない。

# Implementation Handoff

- Approval: 2026-10-07のユーザーによる明示的な実装依頼。
- Implementation: `tamaki-lab/2026_09_ishikawa_sequential-video-lora` の `dev`、base commit `605f2afb95e1dda541208a46d69ce46acd83b335` からのlocal uncommitted changes。
- Adopted decisions: SHA-256 ranking後にAdapter順を復元、version管理profileのみproduction候補、縮小protocolを`stage6b-subset-v1`として分離。
- Execution boundary: commit、push、PR、実ActivityNet end-to-end run、full run、GPU長時間runは未許可・未実施。
- 2026-10-08追加監査: skip preflightをfull inventoryからのexact selection再計算へ強化し、selection record、stop境界、launcher、manifest/feature/result自動ID分離の回帰テストを追加した。
- 未検証: 実ActivityNet 1,000/1,000 class coverage、実行時間、縮小MoCoの表現性能、10,024-video実run。
