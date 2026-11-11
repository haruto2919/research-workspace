---
date: 2026-11-10
project: sequential-video-lora-analysis
source_todo: null
topic: research-code-refactoring-plan
status: exploratory
tags: [brainstorm, research, refactoring, vit-moco, lora]
---

# 研究コードのリファクタリングの範囲と順番

## 出発点

2026-11-10、ユーザーは研究コード（`2026_09_ishikawa_sequential-video-lora`、`feature-vit-moco@9285a47`）をリファクタリングしたいと考えた。その前提として、[構造解析](../../../lab/projects/sequential-video-lora-analysis/code-analysis/2026-11-10-implementation-structure-explain.md)で実装の全体構造を整理した。この壁打ちでは、リファクタリングの範囲と順番を決める。

## 読み込んだ文脈

- [構造解析（2026-11-10）](../../../lab/projects/sequential-video-lora-analysis/code-analysis/2026-11-10-implementation-structure-explain.md)：コードは3つの層に分かれる。
  - ① 使われていない汎用フレームワーク
  - ② MoCoの公式一致検証（parity）
  - ③ 本流のViT＋MoCo＋LoRA
- 問題点：
  - P1：本流がparityとsmokeから汎用の道具を借りている
  - P2：`model` をimportするとLightningとComet MLまで読み込まれる
  - P3：runnerが769行にすべての役割を抱えている
  - P4：診断モジュールがpresetの許可ルールも持っている
  - P5：runnerの骨組みが3回コピーされている
  - P6：checkpointの仕組みが5種類、設定の方式が4種類ある
  - P7：presetの鎖が最大5段
- [2026-09-18の壁打ち](2026-09-18-simple-cnn-clean-architecture.md)：Lightningを使い、責務ごとに層を分ける構成（data → encoder → LoRA → MoCo → orchestration）を描いていた。実際には、Meta公式とbitwise一致させるためにLightningを使わず、`mp.spawn` ＋ DDPで作った。
- 再現性の記録：
  - L5とA2のprefix比較では、GPUで同じ設定を2回実行しても、step 0の特徴とstep 1のlossまではbitwise一致し、それ以降はわずかにずれた（GPUのbackwardがrun間で非決定的なため。[A2 LoRA Evidence](../../../lab/projects/sequential-video-lora-analysis/experiments/2026-11-03-vit-moco-lora-activitynet-a2-lora-gpu-evidence.md)）。
  - テストの基準は、11/04時点で「36 failed / 972 passed」。

## ユーザーが決めたこと（2026-11-10）

| 論点 | 決定 |
|---|---|
| 目的 | **次の実験を足しやすくする**、**自分が読んで分かる構造にする**、**不要コードを減らす**（他の人が使える形にすることは今回の目的に入れない） |
| 旧フレームワーク（層①） | **今回は触らない** |
| 互換性 | **数値までbitwise一致** |
| 進め方 | **まとめて作り直す** |

## 決定どうしの緊張関係（AIの指摘）

1. **「不要コードを減らす」と「旧フレームワークは触らない」**：
   - 減らす対象は、本流の中の重複（runnerのコピー、checkpointの仕組み、presetの鎖）だと解釈した。
   - ただしP2（`model/__init__.py` の一括import）は旧フレームワークのファイルを1行触ることになる。今回の範囲に入れるかは未決。
2. **「bitwise一致」と「まとめて作り直す」**：
   - 一度に作り直すと、一致しなかったときに原因の場所を探しにくい。
   - 対策：作り直す前に、**今のコードの出力を「正解データ」として保存しておき**、新しいコードがそれを再現するかで確かめる。
3. **GPUではbitwise一致を確かめられる範囲に限界がある**：
   - 同じコードでも、GPUではstep 2以降が一致しない。
   - 「bitwise一致」を確かめる範囲は、**CPU（gloo）では全step、GPUではstep 0の特徴とstep 1のloss**にするのが現実的だと考える。

## bitwise一致を守るために変えてはいけないもの（コードから確認）

- seedの順番：seed → backbone（LoRA Aの初期化）→ もう一度seed → projectorとQueue（`build_model`）。
- 乱数を使う順番：
  - augmentation（`ClipTransform` は、フレームごとに乱数状態を戻して同じ値を使う）
  - shuffle BNの `randperm`
  - epochごとのseed
- forwardの計算順：`meta_einsum` のlogits、Queueのコピーを取るタイミング、enqueueのタイミング。
- **checkpointの同一性ハッシュ（`config_identity`）**：
  - ここには次の文字列がそのまま入っている：設定の値、`IDENTITY_SCHEMA`・`LR_SCHEDULE`・`CLIP_FUSION` の文字列、augmentationのrepr（クラス名を含む。例：`ClipTransform(one draw per clip)`、`PILGaussianBlur(sigma=...)`）。
  - **クラス名・reprの文字列・設定のキー構造を変えると、既存のcheckpointからのresumeが拒否される**。
  - モジュールのパスはreprに入っていないので、ファイルを移動するだけなら影響しない（G2の `run_report.json` で確認）。
- `temporal_order_eval` の `load_query` は、`build_model` を呼んでcheckpointを `strict=True` で読み込む。そのため、モデルのパラメータ名（state_dictのキー）を変えると、既存のcheckpointを評価できなくなる。

## 発散：候補

### 整理のやり方

| 案 | 内容 | bitwise一致との相性 |
|---|---|---|
| A. 移動と分割だけ | 計算の中身は変えずに、ファイルを責務ごとに分ける。共通の道具を1か所に集める | ◎ |
| B. 共通エンジン＋実験定義 | 学習ループを1つのエンジンにまとめ、実験ごとの違い（データ、許可値、probe）を「実験定義」として外に出す | ○（ループを統合するときに、乱数の消費順を変えないよう注意が必要） |
| C. Lightningなどへ移行（9月の構想） | フレームワークに学習ループを任せる | ×（DDPの初期化や処理の順番が変わり、bitwise一致はほぼ不可能） |
| D. presetの鎖を平らにする | Hydraのconfig group（model/、data/、moco/ など）を組み合わせて、1つの実験を1つのyamlで表す | △（値が同じでも、キー構造が変わると同一性ハッシュが変わる。変換の仕組みが必要） |

### 新しい構造の例（案A＋Bの場合、未決定）

```text
<新パッケージ>/
  common/       ハッシュ、JSONの書き出し、git情報、乱数の保存と復元、プロセス間の集約
  models/       ViTのbackbone（画像・clip）、LoRA
  ssl/          MoCo（Standalone → Distributed → ViT の継承はそのまま）
  data/         augmentation、ImageNet、ActivityNetのmanifest、sequential-loader
  train/        プロセス起動、学習ループ（epoch型・sequential型）、1 step、checkpoint、レポート
  experiments/  presetごとの許可ルール（今は診断モジュールの中にある）
  diagnostics/  固定probeの測定だけ
  eval/         並び順評価（次はlinear probeをここに追加）
```

### 進め方の候補

- **並行して作る**：新しい構造を別ディレクトリにまとめて作り、古い本流は動く状態のまま残す。正解データで一致を確かめてから入れ替え、古い本流を消す。「まとめて作り直す」と「bitwise一致」の両方を満たしやすい。
- **その場で置き換える**：今のファイルを直接書き換える。途中で動かない期間ができる。

### 反例・別の見方

- 次の実験（linear probe、stride比較、G2モデルの並び順評価）が近い。それらは、今のコードに評価スクリプトを1本足せば済むかもしれない。リファクタリングで実験が止まる期間がコストになる。
- 「まとめて作り直す」途中で研究の実験をしたくなった場合、どちらのコードで実行するかが曖昧になる。

## 収束（AIの提案、未合意）

1. **最初に正解データを取る**：今のコードで、CPU（gloo）の短いrunを各data source・各presetで実行し、出力（metrics、最終状態のハッシュ、checkpoint）を保存する。GPUは代表の設定でstep 0の特徴とstep 1のlossを取る。
2. **新しい構造を並行して作る**：案A（移動と分割）を中心に、学習ループは案Bの形にまとめる。計算の中身は変えない。
3. **確かめる**：
   - 正解データとbitwise一致すること
   - 既存のテストで、失敗するテストIDの集合が変わらないこと
   - 既存のG2 checkpointから、resumeと評価の読み込みができること
4. **入れ替える**：古い本流を消す。旧フレームワーク（層①）は残す。

棄却寄り：案C（Lightningへの移行）。bitwise一致の条件と両立しないため。

保留：案D（presetを平らにする）。同一性ハッシュとの互換をどう保つか決まるまで保留。

## 未決事項

1. parity群（層②）の扱い：
   - (a) 新しい共通モジュールを使うようにimportを書き換える
   - (b) 触らずに、本流側に道具をコピーする（重複は残る）
2. GPUでのbitwise一致の範囲を、「CPUは全step、GPUはstep 0の特徴とstep 1のloss」としてよいか。
3. 既存checkpointとの互換：
   - G2の続きを学習（resume）する予定があるか。ないなら、評価のための読み込み互換だけで足りる。
4. `model/__init__.py` の一括importの解消（旧フレームワークのファイルを1行触る）を、今回の範囲に入れるか。
5. presetの鎖を平らにするか（案D）。
6. 次の実験（G2モデルの並び順評価、linear probe）を、リファクタリングの前にやるか後にやるか。

## 次アクション候補（未採用・TODO未追加）

1. 上の未決事項を決める。
2. 決まったら、`research-spec` でリファクタリングのspecにする（正解データの取り方と一致の判定方法を含む）。
3. specの承認後に、`engineering-task` で実装する。

## 追記（2026-11-10）：ユーザーの要望とファイル構造の案

### ユーザーの要望

1. **旧コードの構造を守る**：`model/__init__.py` のように、上位のフォルダ（`model/`、`dataset/` など）に `__init__.py` を置いて公開する名前を並べる形を守る。
2. **GPUの数（0＝CPU・1・2）を簡単に切り替えられるようにする**。
3. **学習の設定を1か所でまとめて変えられるようにする**：stride、シャッフルか逐次読み込みか、など。
4. **本流のコードが、テストや検証用のコードがないと動かない状態をやめる**。
5. ファイル構造の案がほしい。

### 今のコードで障害になっている箇所（確認した事実）

- GPUの数：
  - `prepare_config` は `world_size >= 2` を必須にしている（`moco_vit_ddp_smoke.py:136-138`）。
  - GPU runは、`ADOPTED_RECIPE`（`:96-106`）で `distributed.world_size: 2`、`local_batch_size: 128` などに固定され、違う値を拒否する。
  - CPUは「テスト用（gloo）」という扱い（`conf/moco_vit_ddp_smoke.yaml:31,46`、`:167-168`）。
- 設定の一括変更：
  - 値がpresetの5段の鎖に散らばっている。
  - presetごとの許可リスト（`ADOPTED_RECIPE`、`query_collapse_diagnostic.py` の `ACTIVITYNET_GPU_RUNS`・`CLIP_ADOPTED`・`SEQUENTIAL_ADOPTED`・`SEQUENTIAL_GPU_RUNS` など）が、GPUでの変更を拒否する。
  - strideなどをyamlで変えても、GPUでは実行できない。
- テストや検証への依存：
  - 本流は `test/` をimportしていない。
  - ただし、検証用のスクリプト（`scripts/parity/` の3ファイル、`scripts/smoke/smoke_moco_cifar10.py`）から道具を借りている。
  - 本流の中に「CPUテスト専用」の分岐がある（`data.sequential.video_dir`、`device=cpu` のgloo）。
  - テスト同士のimport（`test_vit_moco_lr_activitynet.py` → `test_moco_vit_ddp_smoke.py` など）もある。
- その他：
  - リポジトリの `CLAUDE.md` は `@AGENT.md` を参照しているが、`AGENT.md` は存在しない（`AGENT_RESEARCH.md` はある）。
  - `AGENT_RESEARCH.md` §2には「大規模な変更は段階的に」「不要なファイル分割をしない」とあり、「まとめて作り直す」という決定とは方針がずれる。

### ファイル構造の案（AIの提案、未合意）

旧フレームワークと同じく、上位フォルダに `__init__.py` を置く形に合わせます。旧ファイルの中身は変えず、`__init__.py` に公開する名前を書き足すだけにします。中身のない既存の `training/` と `evaluation/` は、この案で使います。

```text
main_moco.py                    入口（Hydra）。今の scripts/vit_moco/moco_vit_ddp_smoke.py の run() と main()
conf/moco/                      新しい設定のroot（旧 conf/ のファイルは触らない）
  config.yaml                   defaults: trainer, model, data, moco, optimizer, diagnostics
  trainer/ cpu.yaml gpu1.yaml gpu2.yaml      devices（0/1/2）、backend、precision
  model/ lora.yaml full_ft.yaml
  data/ imagenet.yaml anet_frames.yaml anet_clips.yaml anet_sequential.yaml   order、stride、frames、interval
  diagnostics/ off.yaml probe.yaml
  experiment/ l5.yaml a2.yaml clip_a2.yaml g2.yaml   これまでの実験を1ファイル1実験で再現する
model/vit/                      vit_moco_backbone.py、lora_linear.py、vit_lora_frame_encoder.py（中身はそのまま）
self_supervised/moco/           standalone_moco.py → ddp_moco.py → vit_ddp_moco.py（中身はそのまま）
dataset/moco/                   新しく作る。__init__.py で公開する
  transforms.py                 TwoCropsTransform、ClipTransform、PILGaussianBlur、two_crops（smokeとparityから移す）
  imagenet.py
  activitynet_frames.py         scripts/vit_moco/ から移す
  activitynet_sequential.py     scripts/vit_moco/ から移す
  factory.py                    data.source と data.order で切り替える入口
training/                       既存の空ディレクトリを使う
  launcher.py                   devices 0/1/2 の切り替え、プロセスの起動、process group
  loop.py                       epoch型・sequential型
  step.py                       train_step、step_metrics
  checkpoint.py                 save/resume（形式 v1 のまま）
  report.py                     run_report、vram_gate、final_checks
  probe.py                      学習中の固定probe（今の query_collapse_diagnostic の測定部分）
evaluation/                     既存の空ディレクトリを使う
  temporal_order.py             今の temporal_order_eval.py
  （次の実験：linear_probe.py）
utils/moco/                     hashing、provenance（git_info、environment）、jsonio、rng、distributed
                                （parityから移す。utils/__init__.py は触らない案もある。下の論点Cを参照）
```

使い方のイメージ：

```bash
python main_moco.py experiment=g2 trainer=gpu1 data.frame_stride=32
```

### この案で要望を満たすしくみ

- **GPUの切り替え**：`trainer.devices` を1か所で決める。
  - 0ならCPUの1プロセス（gloo）、1ならGPUの1プロセス、2ならGPUの2プロセス（今と同じ）。
  - MoCoの `concat_all_gather` は、プロセスが1つでも動く。そのため、DDPの経路は1本のままで済む（推測。実装時に確認する）。
- **設定の一括変更**：
  - 実験は `experiment/*.yaml` の1ファイルで決まる。
  - 個別の値は、コマンドラインで1か所を上書きする。
  - presetの許可リストは無くし、代わりに解決済みの設定全体と、基準との差分を `run_report.json` に残す。
- **本流がテストや検証から独立する**：
  - 本流が借りている道具を、本流側（`utils/moco/`、`dataset/moco/`）に移す。
  - 依存の向きを、検証（parity・smoke）が本流をimportする形に逆転させる。
  - 「テスト専用」の分岐は、正式な機能にする（CPU＝`devices: 0`、`video_dir`＝フォルダの動画を読むdata source）。
- **bitwise一致と既存checkpointとの互換**：
  - `config_identity` は、新しい設定の構造から、コードで旧形式の辞書に組み立て直す。こうすれば、G2のcheckpointのresumeと評価を守れる。
  - クラス名・reprの文字列・パラメータ名は変えない。

### 新しく決める必要が出た論点

- A. **presetの許可リストを無くしてよいか**：これまでは、承認したspec以外の値でGPU runが走らないように守っていた。
  - 代案：許可リストは無くし、gitがclean・成果物を上書きしない、のガードだけ残す。
- B. **GPUの数を変えたとき、何を一定にするか**：
  - (a) global batchを256に保つ（1GPUならlocal 256）
  - (b) local batchを128に保つ（1GPUならglobal 128）
  - Queueの65,536はglobal batchで割り切れる必要があり、shuffle BNの効き方も変わるので、比較の条件が変わる。
- C. **旧フレームワークの `__init__.py` に書き足してよいか**：
  - `model/vit/__init__.py`、`utils/__init__.py` が対象。
  - `utils/__init__.py` を経由すると、Comet MLの読み込みが残る。
- D. **parityとsmokeのimportを、新しい場所に書き換えてよいか**（前の未決事項1）。
- E. **古いコマンドとの互換**：これまでの実験記録にある `python -m scripts.vit_moco.moco_vit_ddp_smoke ...` を残すか（薄い入口として残す／消す）。
- F. **「シャッフル」の範囲**：
  - 今ある2つの読み方（manifestのランダム順と、sequential-loaderの逐次読み込み）を1つの設定で切り替えるところまでなら、データは変わらない。
  - 「sequential-loaderのclipをシャッフルする」は新しい機能になる（新しい実験条件）。

## 追記（2026-11-10）：論点A〜Hの決定

| # | 論点 | 決定 |
|---|---|---|
| A | presetの許可リスト | **無くす**。代わりに、解決済みの設定全体と、experimentファイルからの差分を `run_report.json` に残す。「gitがclean」「成果物を上書きしない」のガードは残す |
| B | GPUの数を変えたときのbatch | **global 256を保つ**（1GPUならlocal 256、CPUの1プロセスでも256） |
| C | 公開のしかた | **既存の `__init__.py` に書き足す**（`model/vit/__init__.py`、`utils/__init__.py`、`dataset/__init__.py` など）。新しいファイルは既存のパッケージの中に置き、その `__init__.py` から公開する。`utils/__init__.py` を経由するとComet MLが読み込まれるのは、今のまま受け入れる |
| D | parityとsmoke | **本流をimportするように書き換える**。道具は本流に1つだけ置く。parity側のテストも通ることを確認する |
| E | 古いコマンド | **消す**。過去の実験は、記録にあるcommitをcheckoutして再現する。新しいコードでは `experiment/` のファイルで同じ実験を表す |
| F | シャッフル | **既存の2つ（manifestのランダム順＝A2型、sequential-loaderの逐次読み込み＝G2型）を、`data.order` 1つで切り替えるだけ**。sequential-loaderのclipをシャッフルする新機能は、今回は含めない |
| G | bitwise一致の確認範囲 | **CPU（gloo）は全step、GPUは代表の設定でstep 0の特徴とstep 1のloss** |
| H | 次の実験との順番 | **G2モデルの並び順評価だけ先に行う**（今のコードでできる）。linear probeは、リファクタリングの後に新しい構造で足す |

### 決定から生じる注意点（AIの指摘）

- **Bと1GPUのVRAM**：
  - 2GPU×128のとき、clip LoRA（gradient checkpointingあり）は1GPUあたり約13 GB、単一フレームFull FTはheadroom 1,750 MiBだった。
  - 1GPUでlocal 256にすると、少なくともFull FTは載らない見込み（推測）。
  - gradient accumulationで分けると、MoCoのenqueueとshuffle BNの単位が変わり、global 256と同じ学習にならない。
  - 1GPUで載らない設定は、明確なエラーで止めるのが安全（VRAM gateは今もある）。
- **C**：既存のパッケージにファイルを置くので、前に出した構造案の `dataset/moco/`・`utils/moco/` というサブフォルダは必須ではなくなる。サブフォルダを作るか、既存パッケージの直下に置くかは、spec化のときに決める。
- **E**：今 `scripts/vit_moco/temporal_order_eval.py` は、runnerを `runner` という名前でimportしている。`evaluation/` へ移すときに、新しい入口からimportするよう直す。
- **H**：G2の並び順評価は、GPUを使う評価runなので、別に許可を取る。今の `temporal_order_eval` が、sequential-loaderで学習したrunのcheckpointをそのまま読めるかは未確認。

### research-specへの引き継ぎ候補（spec readinessは未判定）

- 実装目的：
  - GPUの数（0/1/2）と学習の設定（stride、`data.order` など）を1か所で切り替えられるようにする。
  - 本流を、テストや検証のコードから独立させる。
  - 計算の中身は変えず、bitwise一致を保つ。
- 構造：
  - 旧フレームワークの `__init__.py` で公開する形に合わせる。
  - 新しい入口 `main_moco.py`、新しい設定のroot `conf/moco/`（trainer/model/data/diagnostics/experiment）。
  - 学習は `training/`、評価は `evaluation/`。道具は `utils/`・`dataset/` に置く。
- 検証：
  - 正解データ（今のコードのCPU全stepと、GPUのstep 1まで）とbitwise一致する。
  - 既存のテストで、失敗するテストIDの集合が変わらない。
  - parityのテストも通る。
  - G2のcheckpointから、resumeと評価の読み込みができる。
- 対象外：
  - 旧フレームワーク（層①）の中身の変更（`__init__.py` への書き足しは除く）
  - Lightningへの移行
  - 新しいデータの読み方（sequential-loaderのclipのシャッフル）
  - same-video negativeの扱いなど、学習内容の変更
  - linear probeの追加
- 未決（spec時に決める）：
  - サブフォルダを作るか、既存パッケージの直下に置くか
  - 1GPUでlocal 256が載らない設定の扱い
  - experimentファイルの命名
  - 正解データとして取るrunの一覧

## 追記（2026-11-11）：spec化

draftのspecを作った：[ViT＋MoCo＋LoRAの本流のリファクタリングspec](../../../lab/projects/sequential-video-lora-analysis/specs/2026-11-11-vit-moco-pipeline-refactoring-spec.md)。

## 関連ファイル

- [構造解析](../../../lab/projects/sequential-video-lora-analysis/code-analysis/2026-11-10-implementation-structure-explain.md)
- [2026-09-18 クリーンな実装構造](2026-09-18-simple-cnn-clean-architecture.md)
- [2026-09-23 Hydra移行](2026-09-23-hydra-config-migration.md)
