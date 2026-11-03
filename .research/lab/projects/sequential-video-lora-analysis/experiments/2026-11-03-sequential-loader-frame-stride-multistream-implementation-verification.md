---
project: sequential-video-lora-analysis
date: 2026-11-03
record_type: implementation-cpu-verification
status: cpu-checks-pass-committed
loader_repository: tamaki-lab/2026_09_ishikawa_sequential_loader
loader_branch: feature-stride-ddp
loader_base_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0
code_state: committed
loader_commit: 85a39c2c00f4e4c04e4eaa1022ebf5ab1ef077aa
spec: ../specs/2026-11-03-sequential-loader-frame-stride-multistream-ddp-spec.md
---

# sequential-loader：フレーム数stride と multi-stream DDP の実装とCPU検証

[承認済みspec](../specs/2026-11-03-sequential-loader-frame-stride-multistream-ddp-spec.md)を、loaderリポジトリ `2026_09_ishikawa_sequential_loader` に実装した。branchは `ActivityNet@19a0ed7` から切った `feature-stride-ddp`。

- **CPUテストT1〜T10、実データR1〜R2はすべてPASS**。全テストは207件OK（既存183件＋新規24件）。
- 変更前のloader（`19a0ed7`）とstride=1の出力を実動画20本で比べ、**bitwise一致**した。
- **研究コードは変更していない**。loaderは同日 `feature-stride-ddp@85a39c2` としてcommitした（pushはしていない）。GPUは使っていない。

## 変更したファイル

| ファイル | 内容 |
|---|---|
| `src/config.py` | `FixedChunkConfig.frame_stride`（既定1、1以上の整数だけ）。`MultiStreamConfig`（B、W（0〜B）、prefetch_factor、seed、rank／world_size）と値の検証 |
| `src/sequential/chunking.py` | `ChunkSpec.frame_stride`（既定1）。fixed／unboundedのplannerが、chunk 1つあたり `frames_per_chunk × stride` のフレーム範囲を割り当てる。`valid_length` は、その範囲で取り出すフレームの数 |
| `src/sequential/reader.py` | 取り出すフレームまでdecodeを進め、間のフレームはRGBに変換しない（`_frame_at`）。動画の長さが分からないsourceでは、次に取り出すフレームがあるかどうかでEOFを判定する（`_has_frame_at`）。chunk要求の検証もstrideに合わせた |
| `src/sequential/dataset.py` | strideをplannerに渡す。chunk数は `ceil(ceil(範囲/k)/N)`。frame indexの照合も、間隔kで行う |
| `src/sequential/multistream.py`（新規） | `MultiStreamBatch`、`assign_streams`、`build_multistream_loader`／`MultiStreamLoader`（`set_epoch`、`last_epoch_summary`）。W=0ではメインプロセスで読む。W≥1ではDataLoader worker（in_order）がstream枠を `i mod W` で分担し、メインプロセスがstepごとに組み立てる。world_size>1では、毎stepの状態（正常・尽きた・失敗）を `all_reduce(MIN)` で全rankに揃える |
| `sequential_loader/__init__.py` | 公開APIに `MultiStreamConfig`、`MultiStreamBatch`、`build_multistream_loader` を追加 |
| `tests/test_frame_stride.py`（新規） | T1〜T3（11件） |
| `tests/test_multistream.py`（新規） | T4〜T10（13件。glooの2プロセスのテストを含む） |
| `tests/test_public_api.py` | 公開名の一覧に、新しい3つを追加 |

diff：既存ファイル6つで+154／−26行。新規ファイルは3つ（multistream.py 396行、テスト2つで515行）。依存パッケージの追加はない。

## CPUテスト（`python -m unittest discover -s tests`、研究リポジトリの `.venv`）

| # | 内容 | 結果 |
|---|---|---|
| — | 既存の183件 | 変更後もすべてOK |
| T1 | `frame_stride=1` と既定の設定の出力が一致する（frames・indices・timestamps（NaNの位置も含む）・mask・lifecycle） | PASS |
| T2 | stride 1／2／3／4／30 × chunkの枚数 1／4／5で、終端が分かるsourceと分からないsourceの両方を確かめた。frame index、画像（固定パターン）、timestamp、chunk数、padding、`is_last`、`sequence_length` が正しい。chunkの倍数ちょうどの場合も正しい。stride kのフレームは、stride 1の同じindexのフレームと一致する | PASS |
| T3 | 不正なstride（0、−1、True、2.0、"2"）を拒否する。RGBに変換するのは取り出すフレームだけで、変換の回数は取り出したフレームの数と同じ | PASS |
| T4 | B=1・W=0のbatchの列が、担当列をstrict経路で読んだ出力と一致する（stride 1と2） | PASS |
| T5 | B=4で、W=0／1／2／3／4のbatchの列と `last_epoch_summary` が、すべてbitwise一致する | PASS |
| T6 | 長さの違う8本の合成動画で確かめた。各枠の中のchunkの順番、切り替わりでの `is_first`・chunk 0・index 0、担当列の順番が正しい。step数は「枠ごとのchunk数の合計の最小値」と一致し、一部だけ埋まったbatchは返さない。summaryには読み終えた動画・途中の動画・未読の動画がそろい、どの枠でも担当列と一致する。最初から空の枠がある場合は0 stepになる | PASS |
| T7 | 割り当ては、rank間で重ならず、全rankを合わせると全sourceになる。sourceを渡す順番に左右されず、同じ (seed, epoch) なら同じ割り当てになり、epochが変わると割り当ても変わる。`set_epoch` が効き、負の値は拒否する | PASS |
| T8 | glooの2プロセスで確かめた。rankごとにchunk数の合計が違っても、両rankが同じstep数（全枠の最小値）で終わる。片方のrankで壊れた動画を読ませると、両rankが同じstepで止まる。壊れた側は元のエラー（動画IDを含む）を、もう一方は「failed on another rank」を出す | PASS |
| T9 | W=0とW=2で、1 stepだけ読んで抜けると、worker processが残らず、開いているファイルの数も元に戻る | PASS |
| T10 | `MultiStreamConfig` の不正な値（B<1、W>B、W<0、W=0でprefetchを指定、prefetch 0、rank≥world_size、負のrank、world_size 0、bool／float）を拒否する。sequence_idが重複していたら拒否する。world_size>1なのにtorch.distributedを初期化していない場合も拒否する | PASS |

合計207件、12.5秒。`git diff --check` も問題なし。

### テストが誤りを検出できるかの確認

わざと誤りを入れて、テストが落ちることを確かめた。確認後、ファイルは元に戻した（`cmp` で一致を確認）。

| 入れた誤り | 結果 |
|---|---|
| rank間の合意を `MIN` から `MAX` に変える | T8が2件エラーになった |
| workerが担当する枠の列を逆順にする | T5／T6などで4件が失敗した |
| 取り出す位置の1つ手前のフレームを返す | strideのテストで17件が失敗・エラーになった |
| epochの終わりの処理の書き方を変える | OK。動きが同じになる書き換え（等価な変更）だったため |

## 実データでの確認（CPU、研究リポジトリの `.venv`、NASは読み取りのみ）

対象は、ActivityNet trainingの20本（18,780フレーム）。[2026-11-03のloader確認](2026-11-03-activitynet-sequential-loader-verification.md)の259本から選んだ。

- 5秒未満：5本
- webm：2本
- 1080p：3本
- そのほか：10本
- この中には、末尾でtimestampが重複する動画2本と、末尾で間隔が飛ぶ動画3本が入っている

| # | 内容 | 結果 |
|---|---|---|
| T1（実データ） | strict経路・stride 1で、**変更前の `19a0ed7`**（`git archive` で取り出したもの）と変更後の出力を比べた。全chunkのframes・indices・timestamps・mask・lifecycleのハッシュを取った | **20／20本で一致**（1,182 chunk） |
| R1 | stride 4のchunkのフレームとtimestampが、stride 1で読んだ同じindex（4の倍数）のものとbitwise一致する。indexの列も一致する | **20／20本で一致** |
| R2 | 20本をB=4で読み、W=0とW=4で最初の40 stepを比べた（2,545フレーム） | **ハッシュが一致** |

### 読み込み速度（記録のみ、目標値は決めていない）

| 条件 | 結果 |
|---|---|
| strict、stride 1 | 1プロセスあたり135フレーム/秒（20並列、この20本の解像度の構成で） |
| strict、stride 4 | 動画1フレーム分を進める速さは291フレーム/秒（**約2.2倍**）。実際に取り出すフレームは72.8フレーム/秒 |
| multi-stream B=4、40 step | W=0で8.9秒、W=4で3.3秒（**約2.7倍**） |

- multi-streamの計測は、W=0を先に実行した。そのため、W=4のときはOSのファイルキャッシュが効いていた可能性がある。
- strideにしても、decodeは全フレームに対して行う。速くなったのは、RGBに変換しない分である。

## 互換性

- 何も指定しなければ、今の公開APIは今と同じ動きをする（T1、実データT1）。
- 研究コードの `scripts/smoke/smoke_activitynet_vit_clip_feature.py` は `LOADER_COMMIT = 19a0ed7…` を固定しているので、新しいcommitには追従しない（変更していない）。
- `MultiStreamBatch.samples` は、`SequentialSample` を並べたtupleで、tensorに積み重ねてはいない。解像度が動画ごとに違うためで、積み重ねは学習側で行う（spec M3）。

## 未実施

- loaderのpush（ユーザーの指示があったときだけ行う）。
- NCCL／GPUでの動作確認と、研究コードの学習ループへの組み込み（別のspecで扱う）。
- 全10,024本での確認。

## スクリプトと出力（scratchpadに置き、リポジトリには入れていない）

- `r_check.py`：hash／pairs／msの3つのモード
- `r_videos.json`
- 出力：`r_hash_old.json`、`r_hash_new.json`、`r_pairs.json`、`r_ms.json`
- `loader-19a0ed7/`：変更前の展開
