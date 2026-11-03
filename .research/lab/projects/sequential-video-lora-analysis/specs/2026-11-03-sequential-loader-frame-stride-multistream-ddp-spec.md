---
project: sequential-video-lora-analysis
date: 2026-11-03
spec_type: implementation
status: implemented
implementation_repository: tamaki-lab/2026_09_ishikawa_sequential_loader
implementation_base: ActivityNet@19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
implementation_branch: feature-stride-ddp
source_brainstorm: ../../../../secretary/notes/brainstorm/2026-11-03-sequential-loader-stride-ddp.md
---

# sequential-loader：フレーム数stride と、DDP対応の multi-stream 読み込み

**Status: implemented（2026-11-03。CPUテストT1〜T10、実データR1〜R2はPASS。loaderは `feature-stride-ddp@85a39c2` としてcommit済み（pushはしていない）。[実装・検証記録](../experiments/2026-11-03-sequential-loader-frame-stride-multistream-implementation-verification.md)）**。**Status: approved（2026-11-03、ユーザーが承認）**。実装は、別に指示を受けてから行う。commit・push・GPU runは対象外（それぞれ別に許可が必要）。

## 1. 目的

sequential-loaderに、次の2つを加える。

1. **フレーム数のstride**：動画のk枚ごとに1枚を取り出し、それをchunkにする。
2. **multi-stream読み込み**：DDP（複数GPU）の各GPU（rank）で、B本の動画（stream）を同時に、各動画の中では順番どおりに読む。B=1にすると1 streamになる。読み込みは複数workerで並列にできる。

背景は次のとおり。

- 10/08・10/15・10/22のMTGで、strideを変えてLoRAの変化を比べる方針になった。比較候補はstride 64/32/16/8で、値は未確定。
- [2026-11-03 loader確認](../experiments/2026-11-03-activitynet-sequential-loader-verification.md)で、今のstrict設定（1プロセス、batch 1）ではtraining 1周に約190時間かかると見積もった。

## 2. 前提（読み取りで確認した内容）

- 対象は、fork `2026_09_ishikawa_sequential_loader` の `ActivityNet@19a0ed7`（clean）。fork元 `tamaki-lab/sequential_loader` の `target_frame_stream`（秒単位の時間格子）は参考にせず、取り込まない（ユーザー決定）。
- 今のCoreの構成：
  - `SequentialDataset`（IterableDataset）が、source順・chunk順に `SequentialSample` を返す。
  - `SequentialVideoReaderSession` は、先頭から前向きにdecodeし、全フレームをRGBに変換する。frame indexはdecode順に数える。
  - `build_sequential_dataloader` が受け付けるのは、strict設定（batch 1、worker 0）だけ。
  - strideの機能はない。
- 研究コードのDDP：`DistributedSampler`（seed 0、`set_epoch`）を使い、rankごとにbatch 128。MoCoは毎step、keyをgatherし、batchをshuffleする。全rankが同じ回数stepし、batchの形も同じでないと止まる（[brainstorm](../../../../secretary/notes/brainstorm/2026-11-03-sequential-loader-stride-ddp.md)で説明済み）。
- ActivityNetは動画ごとに解像度が違う（320×240〜1920×1080）。loaderは画像を縮小しない。

## 3. 採用する設計（ユーザー決定）

### 3.1 フレーム数stride

| # | 内容 |
|---|---|
| D1 | `FixedChunkConfig` に `frame_stride: int = 1` を加える。1以上の整数だけを受け付ける。 |
| D2 | sequenceの開始frameを `s`、strideを `k`、`frames_per_chunk` を `N` とする。chunk `c` の有効位置 `j` は、絶対frame index `s + k·(c·N + j)` のフレームである。`SequentialSample.frame_indices` は、この絶対indexを返す（間隔kで狭義単調増加）。`timestamps` は、そのフレームのRAW timestampを返す。 |
| D3 | 終端が分かっているsource（`stop_frame` あり）では、`stop_frame` 未満のサンプル位置だけを使う。chunk数は `ceil(ceil((stop−s)/k) / N)` になる。終端が分からないsourceでは、EOFまでのサンプル位置を使う。最後のchunkは、今の契約どおりpaddingして `valid_mask` を付ける。 |
| D4 | Readerは、前向きにdecodeを続ける（全フレームをdecodeする）。ただし、**RGBへの変換とtensorの作成は、サンプル位置のフレームだけ**に行う。 |
| D5 | `frame_stride=1` のとき、出力は今のloaderと**bitwise一致**する（frames、frame_indices、timestamps、valid_mask、lifecycleのflag、metadata）。 |
| D6 | strict経路（`SequentialDataset` ＋ `build_sequential_dataloader`）は、それ以外の挙動を変えない。strideはstrict経路でも使える。 |

### 3.2 multi-stream読み込み（新しい経路）

| # | 内容 |
|---|---|
| M1 | 新しい設定 `MultiStreamConfig` を作る。<br>・`streams_per_rank`（B、1以上）<br>・`num_workers`（W、0以上B以下）<br>・`prefetch_factor`（None または 1以上。W=0のときはNoneのみ）<br>・`seed`（int、既定0）<br>・`rank`／`world_size`（Noneなら、`torch.distributed` が初期化済みのときはその値、そうでなければ0／1）<br>不正な値は、構築するときにエラーにする。 |
| M2 | 新しいbuilderを作る：`build_multistream_loader(sources, reader, chunk_config, stream_config, process_group=None, device=None)`。返すloaderは `set_epoch(epoch)` を持ち、iterationするとstepごとに `MultiStreamBatch` を返す。 |
| M3 | `MultiStreamBatch` には次を入れる。<br>・`epoch`<br>・`step`（epoch内で0から数える）<br>・`rank`<br>・`samples`：長さBの `SequentialSample` のtuple（i番目がstream枠iのsample）<br>tensorへの積み重ねやresizeはしない（解像度が動画ごとに違うため。積み重ねは学習側で行う）。 |
| M4 | **動画の割り当て**：<br>1. sourcesを `sequence_id` 順に並べる。<br>2. `seed + epoch` で初期化したgeneratorでpermutationをかける。<br>3. rank r は、その列の `r, r+world_size, r+2·world_size, …` 番目を担当する。<br>4. rank内のstream枠 i は、rankの担当列の `i, i+B, i+2B, …` 番目を、この順に流す（**枠ごとの固定の担当**）。<br>同じ (seed, epoch, world_size, B) なら、割り当ては常に同じになる。 |
| M5 | **stepの中身**：step t のbatchは、各stream枠の「次のchunk」でできている。枠の動画の前のchunkが `is_last` だった場合は、その枠の担当列の次の動画の、chunk 0を入れる。各動画の中で、chunkは必ず0, 1, 2, …の順に進む。各stream枠は、毎step、1 chunkずつ進む。 |
| M6 | **epochの終わり**：あるstepで、どれか1つの枠が「次の動画が必要なのに、担当列に動画が残っていない」状態になったとする。そのときは、そのstepのbatchを返さずにepochを終える（一部だけ埋まったbatchは返さない）。world_size>1のときは、毎stepで「このrankはbatchを埋められるか」をGPU間で確認する（`process_group` 上の `all_reduce(MIN)`。tensorは `device` に置く）。1つでも埋められないrankがあれば、**全rankが同じstepで終える**。 |
| M7 | **workerの役割**：W=0のときは、メインプロセスがすべての枠を読む。W≥1のときは、枠 i をworker `i mod W` が担当する。各workerは、担当する枠の各stepのchunkを作る。メインプロセスが、それを枠の順に組み立ててbatchにする。workerは最大 `prefetch_factor` step分を先に作ってよい。**同じ設定なら、W（0〜B）の値によらず、batchの中身はbitwise一致する**。 |
| M8 | **epochの記録**：iterationが終わったあと、loaderの `last_epoch_summary`（dict）から次を読めるようにする。<br>・epoch、step数<br>・枠ごとの「読み終えた動画のID」<br>・「途中まで読んだ動画のIDと、最後に読んだframe index」<br>・rankの担当列のうち「一度も読まなかった動画のID」<br>未使用のフレーム数は、使う側がmanifestと照らし合わせて計算する。 |
| M9 | **エラー**：どれかの枠でdecodeエラーが起きた場合、world_size>1なら、M6の確認と同じ仕組みで全rankに知らせ、全rankが同じstepで止める。エラーが起きたrankは元の例外（動画IDとchunk位置を付けたもの）を上げ、ほかのrankは「別のrankで失敗した」というRuntimeErrorを上げる。黙って飛ばしたり、補正したりはしない。 |
| M10 | **資源の解放**：途中でiterationをやめた場合（break、例外、context manager）、全workerを止め、開いている全Readerを閉じる。 |

## 4. 対象外

- 秒単位のstride。fork元の `target_frame_stream` の取り込み。
- loaderでのresize・正規化・データ拡張。tensorへの積み重ね。
- 空いた枠に次の動画を入れる、動的な割り当て（S1(b)）。長さを揃える割り当て。
- 1つの枠を複数のworkerで分担すること（W>B）。
- 研究コードの学習ループへの組み込み（別specで扱う）、ActivityNet以外のAdapterの変更、ラベルとフレームの対応。
- GPUでの学習run。

## 5. 互換性

- 今の公開API（`FixedChunkConfig`、`SequentialDataset`、`build_sequential_dataloader` など）は、`frame_stride` を省略した場合、今と同じ動きをする（D5、D6）。
- 公開APIに、`MultiStreamConfig`、`MultiStreamBatch`、`build_multistream_loader` を追加する。`sequential_loader/__init__.py` の `__all__` も更新する。
- 新しいdependencyは追加しない（torch.distributed、PyAVは使用済み）。
- 研究コードの今の経路（単一フレーム、clip）は、sequential-loaderを使っていないので影響を受けない。研究コードの `scripts/smoke/smoke_activitynet_vit_clip_feature.py` は `LOADER_COMMIT = 19a0ed7…` を固定しているので、新しいcommitには追従しない（変更しない）。

## 6. 実装の範囲と手順

1. forkの `ActivityNet@19a0ed7` から、branch `feature-stride-ddp` を作る。
2. stride（D1〜D6）を実装し、テストT1〜T3を通す。
3. multi-stream（M1〜M10）を実装し、テストT4〜T10を通す。
4. 実データでの短時間確認（R1〜R2）を行う。
5. 記録を `experiments/` に保存し、READMEに反映する。

commitとpushは、ユーザーの指示があったときだけ行う。

## 7. 成功条件

### CPUテスト（unittest、合成動画は `tests/video_fixtures.py` を使う）

| # | 内容 |
|---|---|
| T1 | 今のテスト（183件）がすべて通る。`frame_stride=1` で、新旧の出力がbitwise一致する（終端あり・なし、chunkの倍数ちょうど、短い動画、paddingの各場合）。 |
| T2 | `frame_stride=k`（2、3、4、フレーム数より大きいkを含む）で、次が成り立つ。<br>・各chunkのframeとtimestampが、stride 1の出力のindex `s + k·m` のものと一致する。<br>・indexの間隔がk。<br>・chunk数とpadding、`is_last` が正しい（終端あり・なしの両方）。 |
| T3 | `frame_stride` の不正な値（0、負、bool、float）を拒否する。サンプル位置以外のフレームではRGB変換を呼ばないことを、mockで確かめる。 |
| T4 | B=1、W=0、world_size=1のとき、batchの列が、rankの担当動画をstrict経路で順に読んだ出力と一致する。 |
| T5 | B=4で、W=0、1、2、4のときのbatchの列が、すべてbitwise一致する（M7）。 |
| T6 | 長さの違う合成動画で、次が成り立つ。<br>・各枠の中のchunkの順番。<br>・動画の切り替わりで `is_first`、chunk 0が来ること。<br>・担当列の順に次の動画が入ること。<br>・どれか1つの枠の担当列が尽きたstepで、一部だけ埋まったbatchを返さずに終わること。<br>・`last_epoch_summary` の内容。 |
| T7 | 割り当て：<br>・rank間で重ならず、全rankを合わせると全sourceになる。<br>・同じ (seed, epoch) なら同じで、epochが違えば変わる。<br>・`set_epoch` が効く。 |
| T8 | glooの2プロセスで、次が成り立つ。<br>・rankごとに動画の長さの合計が違っても、両rankが同じstep数で終わる。<br>・片方のrankでdecodeエラーを起こすと、両rankが同じstepで止まる（M9）。 |
| T9 | 途中でbreakした場合や例外が起きた場合に、workerプロセスが止まり、Readerが閉じる（開いているファイル数と、プロセスが残っていないこと）。 |
| T10 | `MultiStreamConfig` の不正な値を拒否する（B<1、W>B、W<0、W=0でprefetch_factorを指定、rank≥world_size）。 |

### 実データでの短時間確認（CPUのみ、約10分以内）

| # | 内容 |
|---|---|
| R1 | ActivityNet trainingの約20本（5秒未満、webm、1080pを含む）で、stride 4のchunkのフレームが、stride 1で読んだ同じindexのフレームとbitwise一致する。 |
| R2 | 同じ約20本で、B=4、W=0とW=4のbatchの列が、最初の数十stepでbitwise一致する。stride 1と4、W=0と4で、読み込み速度（フレーム/秒）を記録する（目標値は決めず、記録のみ）。 |

### 実施しないこと

- GPU、NCCL、研究コードの学習でのrun（学習への組み込みは別spec）。
- 全10,024本での確認。

## 8. 再現性

- 割り当ては (seed, epoch, world_size, B) だけで決まり、Wやprefetchには左右されない（M4、M7）。
- 実データの確認は、[2026-11-03のloader確認](../experiments/2026-11-03-activitynet-sequential-loader-verification.md)と同じく、研究リポジトリの `.venv` で、`/mnt/NAS-TVS872XT/dataset/ActivityNet` を読み取りだけで使う。

## 9. 履歴

- 2026-11-03：brainstorm（D・Mの各決定、S1〜S5はすべて推薦案）をもとにdraftを作成した。
- 2026-11-03：ユーザーが承認した（approved）。変更する対象はloaderリポジトリ（`ActivityNet` branchから `feature-stride-ddp` を切る）で、研究コード `2026_09_ishikawa_sequential-video-lora` は変更しないことを確認した。
- 2026-11-03：loaderリポジトリの `feature-stride-ddp` に実装した（同日 `85a39c2` としてcommit、pushはしていない）。全207テストはOK。実動画20本で、変更前の `19a0ed7` とstride=1の出力がbitwise一致した。R1（stride 4とstride 1の一致）、R2（W=0とW=4の一致）もPASS。specから外れた点はない。
