---
date: 2026-11-03
project: sequential-video-lora-analysis
source_todo: null
topic: sequential-loader-stride-ddp
status: exploratory
tags: [brainstorm, research, sequential-loader, stride, ddp, activitynet]
---

# sequential-loaderでstrideを自由に変更でき、並列（DDP）でも使えるようにする（探索）

## 出発点

2026-11-03、[sequential-loaderの動作確認](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-activitynet-sequential-loader-verification.md)を259本で行い、すべて問題なしだった。ただし速度は1プロセスあたり45.5フレーム/秒で、strict設定のままではtraining 1周に約190時間かかる。

その後、時間情報の評価の壁打ちへ戻ろうとしたところで、ユーザーから次の意向が示された。

- 「その前にstrideを自由に変更可能、並列（DDP）を可能にしたい」

## 読み込んだ文脈

- 10/08・10/15のMTG：「loader側でstrideを変更可能にし、入力条件を比較する」。10/22のMTG：stride 64/32/16/8が比較候補（値は未確定）。
- [sequential-loader解説](../../../lab/projects/sequential-video-lora-analysis/code-analysis/2026-11-03-sequential-loader-activitynet-explain.md)：forkの `19a0ed7`（ActivityNet branch）の状態。
  - strideはない。
  - strict（batch 1、worker 0、shuffleなし）だけが実装済み。
  - relaxed modeは契約上「将来」とされ、未実装。
- 9/28 shared streaming MoCo protocol spec：4-stream round-robinとstrict-onlineを、選択可能な共通実装にする方針だった。
- fork元 `tamaki-lab/sequential_loader` の master `b95b4da`（2026-10-08）を読み取りのみで確認した。
  - `target_frame_stream` と `TimeGridSamplingPolicy` がある。秒単位の時間格子で目標時刻を決め、目標時刻**以降**の最初のフレームだけをRGB化する。LongVideoBench用で、`SequentialDataset` やDataLoaderとはつながっていない。
  - multi-worker／relaxed modeは、READMEに「Not implemented」とある。
  - ActivityNet Adapterはfork元にはない（私たちのforkにだけある）。パッケージ構成も整理し直されている。

## 用語（ユーザーの指摘を受けて統一）

- **sequence**：1本の動画全体。
- **chunk**：sequenceを先頭から順に区切った、決まった枚数（`frames_per_chunk`）のフレームの並び。loaderが返す単位。
- **stream**：学習時に、1本の動画をchunkの順に流し続ける経路。

## ユーザーが採用した方針（2026-11-03）

| 論点 | 採用 |
|---|---|
| fork元の利用 | **fork元の `target_frame_stream` は参考にしない**。strideは独自に作る |
| strideの意味 | **(a) フレームを間引く**：k枚ごとに1枚を取り、取ったフレームで `frames_per_chunk` 枚のchunkを作る。chunkは途切れずに続き、動画の最後まで進む（例：stride 4、16枚 → chunk 0 = フレーム0, 4, …, 60） |
| strideの単位 | **(a) フレーム数**（MTGの64/32/16/8と同じ数え方）。fpsが動画ごとに違うので、同じstrideでも実時間の幅は動画ごとに変わる |
| 並列化の範囲 | **(b) 複数GPUでの学習（DDP）と、読み込みの並列化の両方**。どの場合でも、1本の動画の中ではchunkの順番を守る。並列化は「別々の動画を同時に読む」ことで行う |
| GPUあたりのstream数 | **(a) 1 streamと、(b) B streamを容易に切り替えられるようにする**（B=1で(a)になる） |

ユーザーへの説明で整理したこと：
- 先読み（workerが次のchunkを先に用意すること）では、モデルが先の場面を先に見ることはない。録画済みの動画で学習する場合は、オンラインの順番は崩れない。
- (b) B streamでは、各stepのbatchがB本の動画の「次のchunk」からなる。動画が終わった枠には、次の動画のchunk 0を入れる。

## 未決事項（spec前に決めること）

1. **GPU（rank）への動画の割り当て**：ID順に交互に配るか、長さの合計が揃うように配るか。
2. **rank間でstep数を揃える方法**：DDPでは全rankが同じ回数stepしないと止まる。また、MoCoのkey gatherは全rankで同じbatchの形を要求する。そのためBを固定し、どこかのrankでB本のstreamを埋められなくなったら、そのepochを終える案がある（推薦。捨てたフレーム数を記録する）。
3. **動画の順番**：epochごとにseedでシャッフルするか、ID順に固定するか。
4. **workerとstreamの関係**：
   - (W1) 各stepは常に同じB本のstreamから作る。これには、workerの結果を組み立てる仕組みが別に必要になる。
   - (W2) 標準のDataLoaderで、各workerがB本のstreamを持ってbatchを作る。stepごとにworkerが交代するので、rankあたり同時に進む動画はB×W本になり、各streamはWstepに1回進む。
   - 意味が変わるので、ユーザーの判断が必要。
5. **動画の最後のchunk**：足りないフレームは、paddingと `valid_mask` で埋めるか（loaderの今の契約）、捨てるか。
6. **既存のstrict経路**：変えずに残すか（互換性）。strideをstrict経路にも入れるか。
7. **branch**：forkの `ActivityNet` branchから機能branchを切るか。
8. 参考：B stream × 16枚 × 1080pでは、1 stepの画像が大きい（1 chunkで約100 MB）。resizeをloaderで行うかは別に検討する。

## 反例・注意点

- strideで間引いても、decode自体は全フレームで必要になる（前向きdecodeのため）。速くなるのは、使わないフレームのRGB変換とtensor作成を省く分だけ。
- 同じstrideでも、fpsの違いで実時間が変わる。15fpsと59.94fpsでは約4倍違う。stride比較の解釈では、この点に注意する。
- (W2)やB streamにすると、「1本ずつ順に見る」純粋なオンラインからは離れる。どこまで離れてよいかは、学習のspecで決める。
- 長さの違う動画をrankに配ると、最後に余る動画が出る。どこまで捨ててよいか、全部使い切る必要があるかは研究判断になる。

## 関連ファイル

- [sequential-loader確認記録](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-activitynet-sequential-loader-verification.md)
- [sequential-loader確認の壁打ち](2026-11-03-activitynet-sequential-loader-verification.md)
- [shared streaming MoCo protocol spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-09-28-shared-streaming-moco-protocol-spec.md)

## 追記（2026-11-03）：細部の採用

ユーザーが次を採用した。

- 未決1：GPU（rank）への割り当ては、**ID順に交互に配る**（rank r は、並べた動画列の r, r+W, r+2W, … 番目）。
- 未決2：**Bを固定**する。どれかのrankでB本のstreamを埋められなくなった時点で、epochを終える。使わなかったフレーム数は記録する。
- 未決3：**epochごとに、seedで固定したシャッフル**をする。
- 未決5：最後のchunkは、今の契約どおり**paddingと `valid_mask`** で扱う。
- 未決6：**strict経路は変えずに残す**。strideはstrict経路でも使えるようにする。
- 未決7：forkの `ActivityNet` branchから、**機能branch（例：`feature-stride-ddp`）を切る**。

未決4（workerとstreamの関係）は、ユーザーから「意味が分からない」との返答があったため、具体例で説明し直している（未採用）。

## 追記（2026-11-03）：未決4の採用

具体例（1 GPU、B=2、worker 2）で説明したうえで、ユーザーが **(a)** を採用した。

- 各stepのbatchは、**常に同じB本のstream**から作る。中身は、そのB本それぞれの「次のchunk」。
- workerは読み込みの手伝いだけをする。workerの数を変えても、学習の中身（同時に進む動画はB本、各streamは毎step進む）は変わらない。
- 各workerの結果をstepごとに組み立てる仕組みは、新しく作る必要がある。
- B=1では読み込みの並列化が効きにくい。同じstreamの先のchunkを先読みして補う。

## Research Spec Handoff（候補。spec化はユーザーの明示があってから行う）

- 対象：forkの `2026_09_ishikawa_sequential_loader` の `ActivityNet@19a0ed7`。機能branchを新しく切る。
- 目的：フレーム数のstrideと、DDP（複数GPU）でのB stream並列読み込み。B=1の1 streamも、設定で切り替えられる。
- 採用：
  - stride：k枚ごとに1枚を取り、`frames_per_chunk` 枚のchunkを作る。
  - 動画の割り当て：ID順（epochごとにseedでシャッフル）に並べ、rankへ交互に配る。
  - B：固定する。どれかのrankでB本を埋められなくなったら、epochを終える（使わなかったフレーム数を記録する）。
  - 最後のchunk：paddingと `valid_mask` で扱う。
  - worker：同じB本のstreamを毎step進める。workerは読み込みの補助だけをする。
  - strict経路：変えずに残す（strideは使えるようにする）。
- 対象外：秒単位のstride、fork元の `target_frame_stream` の取り込み、resize、学習ループの変更、ラベルとフレームの対応。
- 未決（specで詰める）：
  - 先読みの深さ。
  - worker数とB本のstreamの対応（worker < B、worker > Bのとき）。
  - epoch終了時点の扱い。
  - 速度の目標。
  - テスト方法：stride=1では今のloaderと一致すること、rank間でstep数が揃うこと、各streamの順番が守られること。

## 追記（2026-11-03）：spec化

ユーザーがS1〜S5のすべてで推薦案を採用した。
- S1：stream枠ごとに担当の動画を固定する。
- S2：batchは、B個の `SequentialSample` を並べたものにする。
- S3：epochの終わりは、loaderが全rankで揃える。
- S4：workerの数は0〜Bにする。
- S5：`FixedChunkConfig.frame_stride` を使う。

draftのspecは次のとおり：[sequential-loader stride・multi-stream DDP spec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-11-03-sequential-loader-frame-stride-multistream-ddp-spec.md)

## 追記（2026-11-03）：実装

specが承認され、実装した（CPUテスト・実データの確認はPASS、loaderは未commit）：[実装・検証記録](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-sequential-loader-frame-stride-multistream-implementation-verification.md)
