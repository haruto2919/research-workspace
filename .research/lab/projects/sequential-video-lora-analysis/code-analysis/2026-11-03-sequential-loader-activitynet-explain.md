---
project: sequential-video-lora-analysis
date: 2026-11-03
analysis_mode: explain
detail: compact
level: mid
source_repository: /mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential_loader (tamaki-lab/2026_09_ishikawa_sequential_loader)
source_commit: 19a0ed7e4c00300214bc9a2fe12da8c72c0499c0 (branch ActivityNet)
source_dirty: false
targets:
  - sequential_loader/__init__.py
  - src/adapters/activitynet.py
  - src/sequential/dataset.py
  - src/sequential/reader.py
  - src/sequential/dataloader.py
related_spec: ../specs/2026-09-23-activitynet-adapter-spec.md (implemented)
---

# Code Deep Dive：ActivityNet用 Sequential Loader

## 対象と解析の範囲

- 目的：ActivityNetのsequential loaderが正しく動くかを、これから確かめる。その準備として、loaderの実装を理解する。
- 対象コード：独立したリポジトリ `2026_09_ishikawa_sequential_loader` の、`ActivityNet` branchにあるcommit `19a0ed7`（clean）。
  - 公開API：`sequential_loader/__init__.py`
  - ActivityNet Adapter：`src/adapters/activitynet.py`
  - Core：`src/sequential/{source,chunking,reader,dataset,sample,dataloader,timestamp}.py`、`src/config.py`
- 研究コードとの関係：研究コード（`2026_09_ishikawa_sequential-video-lora`）の `.venv` は、このcheckoutを `import sequential_loader` として読み込む。`scripts/smoke/smoke_activitynet_vit_clip_feature.py:22` は `LOADER_COMMIT = 19a0ed7…` を固定している。
- 対象外：
  - 50Salads Adapter
  - 旧実装の `src/salads50/`、`src/epic_kitchens/`
  - `debug/`、`docs/`（contractは参照していない）
  - テストの中身（`tests/test_activitynet_adapter.py` は、テスト名の一覧だけを確認した）
- コードの実行、テストの実行はしていない（読み取りのみ）。

## Overview：What / Why

1本の動画を、**先頭から最後まで、連続するフレームの塊（chunk、既定16フレーム）に区切り、時間順に1つずつ返す**loaderである。ランダムに切り出して返すのではなく、ストリーミング入力を想定して「動画を流して見せる」ための経路になっている。

```
ActivityNetAdapter        →  SequenceSource（動画1本 = 1 sequence）
  ↓ sources（動画ID順）
SequentialDataset         →  動画ごとに Reader session を開き、chunk要求を順に出す
  ↓ chunk要求（ChunkSpec）
SequentialVideoReaderSession → PyAVで先頭から順にdecode（seekしない）
  ↓ DecodedChunk（有効フレームだけ）
SequentialDataset         →  末尾のpadding・valid_mask・is_first/is_lastを付ける
  ↓ SequentialSample
build_sequential_dataloader → strict（batch 1、worker 0、shuffleなし）のPyTorch DataLoader
```

## Key Components and Data Flow

| 部品 | 役割 | 根拠 |
|---|---|---|
| `ActivityNetAdapter.sequence_sources(split)` | v1.3 JSONのsubsetを正本として、`v1-3/train_val` と `v1-3/test` を再帰的に探索する。全19,994本について、欠け・余分・重複・root/subsetの不一致を、**呼ぶたびに**検証する。指定したsplitの動画を、ID順（先頭の `v_` を除いた名前）に返す | `src/adapters/activitynet.py:58-79, 89-159` |
| `SequenceSource` | 「何を読むか」だけを表す。ActivityNetでは**動画全体**（`start_frame=0`、`stop_frame=None`＝長さ不明でEOFまで）。ラベルやsegmentは解釈せず、`ActivityNetAnnotationReference`（JSONへの参照のみ。`frame_alignment_verified=False`）として転送する | `activitynet.py:161-186`、`source.py` |
| `iter_unbounded_chunks` | 長さが分からない動画に対して、`[0,16)`、`[16,32)`、…という連続した要求を、終わりなく作る。止めるのはReaderのEOF報告 | `chunking.py:91-138` |
| `SequentialVideoReaderSession` | 動画を1回だけopenし、`container.decode()` で**先頭から前向きにdecode**する。`frame_index` は**decodeした順番の通し番号**。timestampは `pts × time_base`（ない場合は index/fps）。RGB uint8 `[3,H,W]` に変換する。EOFの確認は1フレーム先読みで行う | `reader.py:203-285, 372-461` |
| `SequentialDataset.__iter__` | 動画を順番に処理し、各動画をchunk順に最後まで読む。足りない分をゼロ画像・index −1・timestamp NaNで埋め（padding）、`valid_mask` と `is_first`/`is_last` を付ける。例外が起きると、動画IDとchunk位置を付けて `RuntimeError` にする | `dataset.py:386-422, 466-587` |
| `SequentialSample` | 出力。`frames` uint8 `[T,3,H,W]`、`frame_indices`、`timestamps`、`valid_mask`、`sequence_id`、`sequence_index`、`is_first`/`is_last`、metadata。構造（dtype、prefix型のmask、indexの単調増加など）を検証する | `sample.py` |
| `build_sequential_dataloader` | strict設定（batch 1、`num_workers=0`、shuffleなし、`drop_last=False`、`in_order=True`）だけを受け付ける。relaxedは `NotImplementedError`。collateはSampleを1つそのまま返す（batch次元を足さない） | `dataloader.py:533-584`、`config.py:355-400` |
| `sequential_sample_stream` | 途中で止めても、Readerを確実に解放するためのcontext manager | `dataloader.py:587-624` |

## Concepts and Design Decisions

- **seekしない前向きdecode**：単一フレームやclipの経路（研究コードの `decode_frame`/`decode_clip`）は、keyframeへseekして目標時刻を取り出す。sequential loaderは、先頭から全フレームを順にdecodeする。そのため、seekの問題（11/02に直したもの）は起きない。一方で、1本の動画を最後まで読む時間がかかる。
- **フレーム番号はdecode順**：`frame_index` は「何枚目にdecodeされたか」で、PTSから計算した値ではない。timestampはPTSのまま保持する（RAW mode）。`TimestampMode.RAW.requires_monotonic` はFalseなので、timestampの逆行は検査しない（`timestamp.py:295-298`）。
- **連続フレームだけ**：chunkは隣り合うフレームの塊である。間引き（stride）の機能はCoreにない（`src/sequential`、`src/adapters`、`src/config.py` に `stride`/`subsample` はない）。30fpsの動画なら、16フレームで約0.53秒になる。
- **解像度は元のまま**：resizeや正規化はしない。ActivityNetは動画ごとに解像度が違うので、consumer側で変換する。研究コードの `integration/sequential_vit.py:encode_chunk` は、HFのprocessorで224にしている。
- **strictだけ実装済み**：batch 1、worker 0で、動画を1本ずつ順に読む。複数動画を並べてbatchにする方法、複数worker、DDP用の分割は未実装（`dataloader.py:569-573`）。
- **失敗は黙って飛ばさない**：Adapterはinventoryの不一致を拒否する。Readerは、decodeの例外や「開始位置より前のEOF」をそのまま上に伝える。1本の動画でdecodeが失敗すると、iteration全体が止まる設計である。
- **ラベルは渡さない**：評価に使うときは、consumerが `evaluation_reference.annotation_path` の `database[video_id]` を引く。秒とフレームの対応は未検証と明記されている。

## 根拠・未確認事項

- **Observed**（コードと記録から確認）：
  - 上の表のとおり。
  - 過去の検証（[2026-09-23 Adapter検証](../experiments/2026-09-23-activitynet-adapter-verification.md)）では、次を確認している。
    - 全183テスト（ActivityNet 22件を含む）がOK。
    - 実データのinventoryが19,994本で一致した（training 10,024、validation 4,926、testing 5,044）。
    - 3形式（mp4、mkv、webm）それぞれ1本で、先頭chunkを読めた。
  - [2026-09-23 clip feature検証](../experiments/2026-09-23-activitynet-clip-feature-verification.md)では、training先頭1 chunkをfrozen ViTに通した。
- **Inferred**（推定）：
  - 1動画を最後までdecodeする時間は、動画の長さにほぼ比例すると考えられる。ActivityNet trainingの全フレーム数は未計測だが、単一フレームmanifestの117,522枚（10秒に1枚）から逆算すると、約3,500万枚の規模（30fpsと仮定）。
- **Unverified**（このloaderについて、まだ確かめていないこと）：
  1. 実動画を**先頭からEOFまで**通して読めるか。最後のchunkの `is_last` とpaddingが正しいか。
  2. decode順の `frame_index` と、`timestamps` の関係（逆行やとびがないか）。研究コードの `decode_frame`（PTS基準でseek）が同じ時刻に返す画像と一致するか。
  3. 多くの動画でdecodeのエラーが起きないか（11/02の `v_5kBKAfEX7XA` のような動画を含む）。
  4. 読み込みの速さ（フレーム/秒）。
  5. `sequential_sample_stream` を途中で抜けたときの資源の解放（実データで）。
