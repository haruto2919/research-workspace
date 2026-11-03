---
project: sequential-video-lora-analysis
date: 2026-11-03
record_type: implementation-cpu-verification
status: cpu-gates-pass-committed
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
base_commit: 9155c2baa59c06f8204b81d7d466bf4fc4e955d6
code_state: committed
research_commit: fe58fc37ef28901184980ccd5f9e90e40c3d836d
spec: ../specs/2026-11-03-vit-moco-lora-activitynet-clip-early-fusion-spec.md
---

# ActivityNet clip Early Fusion：実装とCPU検証（T0〜T7）

[ActivityNet clip Early Fusion spec](../specs/2026-11-03-vit-moco-lora-activitynet-clip-early-fusion-spec.md)を、ユーザーの指示で `feature-vit-moco@9155c2b` の上に実装した。研究コードは同日 `feature-vit-moco@fe58fc3` としてcommitした（push未実施）。**CPU Gate T0〜T7はすべてPASS**。GPU run（A1・A2）は未実施。

## 変更したファイル

| ファイル | 内容 |
|---|---|
| `scripts/vit_moco/activitynet_frames.py` | `decode_clip`（1回のseekと前向きdecodeで、複数時刻の `decode_frame` と同じ画像を返す）。`ActivityNetClipDataset`（manifestの時刻tを最後のフレームにして、`t − 0.75, …, t` の4フレームを返す） |
| `model/vit/vit_moco_backbone.py` | `temporal_embedding`（固定sin-cos − t=0の値）。`ViTMoCoClipBackbone`（token結合、空間位置埋め込み＋時間位置埋め込み、全層joint attention、CLS。gradient checkpointingはtransformersの機能をnon-reentrantで使う）。既存の `ViTMoCoBackbone` は変更なし |
| `self_supervised/moco/vit_ddp_moco.py` | `clip_frames` を指定したときだけ、`[B, T, C, H, W]` の入力を受け付ける。チェック（batch・dtype・finite）は親クラスのものを、フレームをchannelにまとめて使う。指定しないとき（画像のrunすべて）は従来どおり |
| `scripts/vit_moco/moco_vit_ddp_smoke.py` | `data.source=activitynet_clips`。`ClipTransform`（各viewで、clip内の全フレームに同じ乱数の拡張をかける）。clip datasetとclip backboneの組み立て。clipのrunだけ `config_identity.model.clip` を記録する。clipは新しいprofileでしか使えない |
| `scripts/vit_moco/query_collapse_diagnostic.py` | profile `lr_activitynet_clip`（preset `moco_vit_lr_activitynet_clip`）。GPUでは、ActivityNetと同じ長さ・probe・manifestに加え、`CLIP_ADOPTED`（T=4、0.25秒、`last_frame`、gradient checkpointing）以外を拒否する。固定probeを静止clipにし、元の画像のハッシュ（`image_input_sha256`）とclipのハッシュの両方を記録する |
| `conf/moco_vit_lr_activitynet_clip.yaml`（新規） | `moco_vit_lr_activitynet` を継承し、clipの設定・gradient checkpointing・profile・出力先だけを変える |
| `test/model/test_vit_moco_lr_activitynet_clip.py`（新規） | T0〜T6の50件 |
| `test/model/test_vit_moco_lr_activitynet.py` | `PRESETS` 全体（3件）を固定していたassertを、既存の3件が変わらないことの確認に変えた（4件全体の固定は新しいテストに移した） |

diff：既存ファイル6つで+235／−24行、新規ファイル2つ。

### specからのずれ（実装時に判明）

- spec §2では、`DistributedMoCo` は `[B, T, 3, H, W]` の入力でもそのまま動くと書いた。batch shuffleとkeyのgatherについては正しかったが、入力チェック（`StandaloneMoCo._check_views`）は4次元だけを受け付けていた。正しい挙動が一意で、要求を満たすのに必要なため、`ViTDistributedMoCo` に `clip_frames` を加えた（画像runの経路は変更なし）。
- 時間位置埋め込みは、t≥1でも、高い次元ほど値が小さい（固定sin-cosの性質）。specの定義どおりで、変更していない。

## CPU Gateの結果

| Gate | 内容 | 結果 |
|---|---|---|
| T0 | clipの取り出し：合成動画5通り（動画の末尾を越える時刻を含む）、最初のseekが目標を越える場合（1つのcontainerで手前から読み直す）、A1で問題になった実動画 `v_5kBKAfEX7XA` の5 clip（45秒を含む）で、`decode_frame` の結果とbitwise一致 | PASS |
| T1 | T=1の一致：(a) clipのdatasetが、単一フレームのdatasetと拡張込みで同じテンソル（3 sample）。(b) Early FusionのbackboneのCLSが、既存のbackboneと一致（LoRA・Full FT） | PASS |
| T2 | 拡張：同じ画像4枚のclipで4フレームの出力が一致し、2つのviewは異なる（5 seed）。1 clipで消費する乱数は1フレーム分と同じ | PASS |
| T3 | 時間位置埋め込み：t=0でゼロ、t≥1で非ゼロ。bufferでstate dictに入らない。学習対象の名前の集合が単一画像のbackboneと同じ（両方式）。全層でgradient checkpointingが有効 | PASS |
| T4 | presetは、ActivityNet presetとの差分がclip・gradient checkpointing・profile・出力先だけ。GPUチェック（monkeypatch）で、T・間隔・clipの位置・checkpointingの変更、長さ・probe・checkpoint・recipe・manifestの変更、他のpreset・profileからのclipの使用（診断なしのrunを含む）を拒否する。`config_identity.model.clip` はclipのrunにだけ入る | PASS |
| T5 | tiny ViT・2rank・glooで、LoRA・Full FT（3step）、LoRA（2step）、gradient checkpointingなしのLoRA（3step）が完走。probe・checkpoint・summaryが揃う。probeの元の画像のハッシュとselectionが単一フレームのrunと一致し、入力の形は `[6, 4, 3, 32, 32]`。gradient checkpointingのON/OFFで、probe_step 0の特徴とstep1のlossがbitwise一致 | PASS |
| T6 | 比較器：clipのLoRAとFull FTで `comparable=true`。2stepと3stepのprefixで `comparable=true`。clipと単一フレームのrunの組み合わせは、`compare_runs`・`compare_prefix` とも比較が成立しない。比較器は変更していない | PASS |
| T7 | 全テスト：**36 failed / 957 passed**。失敗IDの集合は、`9155c2b`（修正後の記録、36 failed / 907 passed）と**完全に一致**。passedの増加分50件は新規テストの数と一致 | PASS |

T7の途中経過：最初の全テストでは37 failedだった。増えた1件は、前の段階のテスト（`PRESETS` が3件ちょうどであることを固定）で、上の「変更したファイル」のとおり直した。直した後、関係する2ファイル（93件）がすべてPASSすることを確認した。全テストの再実行はしていない（直したのはテスト1件のassertだけで、他のテストには影響しない）。

## 実データでの確認（読み取りのみ、GPUなし）

- **clipの取り出し**：manifestから無作為に選んだ2,000 clipと、`v_5kBKAfEX7XA` の全5 clip（計2,005 clip、32プロセス）で、`decode_clip` の4フレームが、各時刻を `decode_frame` で取り出した画像とすべてbitwise一致した（不一致0件）。
  - 1 clipあたりの時間：平均0.40秒、中央値0.28秒（1プロセス、32プロセスが同時に読む条件）。4時刻を `decode_frame` で別々に取り出すと、平均1.17秒だった。単一フレームの全frame確認での平均は0.53秒/frame（同じく32プロセス）。
  - 確認用スクリプトはscratchpadにあり、研究リポジトリには入れていない。
- **本番の設定**：preset `moco_vit_lr_activitynet_clip` から作ったdatasetは117,522 clipで、1 sampleは `[4, 3, 224, 224]` が2つ。実際の動画なので、clip内の4フレームは互いに異なった。

## 互換性

- 既存の経路（ImageNet、`lr_sensitivity`、`lr_long`、`lr_activitynet`（単一フレーム）、0.03、診断のON/OFF）では、新しいkeyは記録されず、計算も変わらない。既存のテストはすべて変更前と同じ結果だった（失敗IDの集合が同じ）。
- `diagnostic_summary.json` の `probe_image_input_sha256` と、probe manifestの `image_input_*` は、clipのrunにだけ入る。

## 未実施

- GPU A1（10step、LoRA→Full FT）とA2（459step）。それぞれ別に許可が必要。
- 785 tokenでのVRAMとデータ読み込み時間（A1で測る）。
