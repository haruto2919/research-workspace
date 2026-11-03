---
project: sequential-video-lora-analysis
spec_type: experiment-and-implementation
status: implemented
created: 2026-11-03
approved: 2026-11-03
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
base_branch: feature-vit-moco
base_commit: 9155c2baa59c06f8204b81d7d466bf4fc4e955d6
source_evidence: ../experiments/2026-11-03-vit-moco-lora-activitynet-a2-full-ft-gpu-evidence.md
source_brainstorm: ../../../../secretary/notes/brainstorm/2026-11-03-activitynet-clip-early-fusion-loss-check.md
parent_spec: 2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md
---

# ViT＋MoCo（LR 3e-4）：ActivityNetの動画clip（token結合Early Fusion、T=4）で学習できるかを確かめるspec

**Status: implemented（2026-11-03。CPU Gate T0〜T7はPASS、研究コードは `feature-vit-moco@fe58fc3` としてcommit済み。GPU A1/A2は両方式とも工学PASS（2026-11-03）。§12参照）**。**Status: approved（2026-11-03。E1〜E12はユーザーが採用。§11参照）**。研究コードの実装とGPU runは、それぞれ別に許可を得てから行う（§9）。

## 1. 目的と背景

[ActivityNet単一フレームspec](2026-11-01-vit-moco-lora-activitynet-single-frame-spec.md)のA1・A2では、両方式とも工学PASSだった。Queueが埋まった後（256stepより後）にMoCo lossは単調に減少し、collapseは観測されなかった（暫定、[A2 Full FT Evidence](../experiments/2026-11-03-vit-moco-lora-activitynet-a2-full-ft-gpu-evidence.md)）。

次の段階として、**入力を1フレームから、複数フレームからなる普通の動画clip（逐次入力ではない）に替え**、次の問いに答える。

1. token結合のEarly Fusionで4フレームのclipを入力したとき、ViT＋MoCo（LoRA／Full FT、LR 3e-4）は工学的に問題なく学習できるか（VRAM、データ読み込み、数値の有限性）。
2. Queueが埋まった後（256stepより後）に、MoCo lossは下がるか。collapseは起きないか。

**時間情報・動き情報を獲得したかは、この段階の問いではない**（2026-11-03、ユーザーの指示）。ただし、後で時間情報を扱う段階でも同じ経路を使えるよう、LoRAが付いたattentionがフレームの間を直接見られる形（token結合）にする。

## 2. 前提（コード調査で確認済み、`9155c2b`）

- `ViTMoCoBackbone`（`model/vit/vit_moco_backbone.py`）は、`ViTModel(pixel_values)` の `last_hidden_state[:, 0]`（CLS）を返す。transformers 5.17の `ViTModel` では、`embeddings.patch_embeddings`、`embeddings.position_embeddings`（`[1, 197, 768]`）、`embeddings.cls_token`、`layers`、`layernorm` を個別に呼べる。`supports_gradient_checkpointing = True`。ViT-B/16の `hidden_dropout_prob` は0。
- LoRAは、Q/Vのlinear（24 module、rank 8）に自前の `inject_attention_lora` で付ける。patch埋め込みと位置埋め込みは、LoRA方式では固定、Full FTでは学習対象。
- `DistributedMoCo` のbatch shuffleとkeyのgatherは、先頭の次元（batch）だけを扱う。`[B, T, 3, 224, 224]` の入力でもそのまま動く。
- データ拡張（`two_crops`）はMoCo v2 aug-plusで、torchの乱数（RandomResizedCrop、ColorJitter、RandomGrayscale、RandomApply、Flip）と、Pythonの `random`（`PILGaussianBlur`）を使う。
- `ActivityNetFrameDataset` は1件＝1フレームで、`decode_frame(path, t, start)` が「`start + t` 以前にptsがある最後のframe」を返す（keyframe seekが目標を越えた場合は手前から読み直す。`9155c2b`）。
- 既存のmanifest `log/activitynet_frames/2026-10-21_training_interval10s_offset5s/manifest.json`：10,024動画、117,522時刻。時刻は `t = 5 + 10n` 秒で、**すべて5秒以上、かつ動画の長さより前**。動画のfpsはばらばら（30fpsが6,544本、25fpsが1,767本、24fpsが902本、15fpsが348本、60fpsが153本など）。
- 単一フレームA2のVRAM（batch 128/rank、197 token）：peak reservedはLoRAで約12.3 GiB/rank、Full FTで約18.3 GiB/rank。GPU1（RTX A4500、20,096 MiB）のheadroomは、Full FTで1,750 MiBだった（閾値1,024 MiB）。
- 単一フレームA2のデータ読み込み待ち：LoRAで約2.3秒/step、Full FTで約0.7秒/step（Full FTが短かった理由は未確認）。

## 3. Decision Contract（承認済み）

| # | 論点 | 採用 |
|---|---|---|
| E1 | この段階の問い | §1の2つの問い。時間情報の獲得と表現品質は対象外 |
| E2 | Early Fusionの形 | **token結合（tubelet 1）**。各フレームを、学習済みの2D patch埋め込みで別々に196 tokenにし、4フレーム分（784 token）を1つの系列にして、先頭にCLSを1つ置く（計785 token）。全層でjoint attention |
| E3 | 位置埋め込み | 空間の位置埋め込み（学習済みの `position_embeddings[:, 1:]`）を、全フレームのpatch tokenに同じように足す。CLSには `position_embeddings[:, 0]` を足す。**時間位置埋め込みは、学習しない固定の1D sin-cos（768次元、底10000）で、t=0のときにゼロになる形**（`PE(t) − PE(0)`、t = 0〜T−1）。各フレームのpatch tokenに足す。CLSには足さない。bufferとして持ち、学習対象のparameterは増やさない |
| E4 | フレーム数 | **T=4** |
| E5 | フレーム間隔 | **0.25秒（時間で揃える）**。各時刻のframeは、manifestと同じ規則（`start + 時刻` 以前にptsがある最後のframe）で決める |
| E6 | clipの位置 | **manifestの時刻tを、clipの最後のフレームにする**。clipの時刻は `[t − 0.75, t − 0.5, t − 0.25, t]`。tは5秒以上なので、clipは動画の外にはみ出さない。sample数（117,522）と1epoch＝459stepは、単一フレームA2と同じ。最後のフレームは、単一フレームA2の同じsampleと同じ画像になる |
| E7 | MoCoのpositive | **同じclipに2通りの拡張**をかける。各viewの拡張パラメータ（crop・色・grayscale・blur・flip）は、clip内の4フレームすべてで同じにする（乱数の状態を各フレームの前に戻す）。同じ動画の別clipをnegativeから除外しない（単一フレームと同じ） |
| E8 | clipの表現 | 最終層のlayernorm後のCLS（今の経路と同じ）。Projector以降は変更しない |
| E9 | batchとメモリ | **local batch 128／global 256、K=65,536は単一フレームと同じ**。query encoderでgradient checkpointingを使う（両方式）。精度はFP32のまま。VRAM Gateが `BLOCKED` になった場合はその場で止め、batchやTを自動で変えない。変更するかは、ユーザーが改めて判断する（別のspec追記） |
| E10 | データ読み込み | 今と同じon-the-fly decode（NASから、16 worker/rank）。1つのclipでは、1回のseekから前向きにdecodeして4フレームを取り出す。clipを事前に書き出す方式は、この段階では行わない |
| E11 | 固定probe | **L4/L5・単一フレームと同じImageNet 64枚**（seed 0、Resize256→CenterCrop224）。各画像を4回繰り返した「静止clip」`[64, 4, 3, 224, 224]` にする。元の64枚の入力ハッシュは `6af50745…08d8bf` と一致しなければならない。静止clipの入力ハッシュは別に記録する（L4/L5とは一致しない） |
| E12 | 長さと順番 | 単一フレームと同じ構成。**10step（A1、LoRA→Full FT）で工学PASSとVRAM・読み込み時間を確かめてから、459step（A2、LoRA→Full FT）**。LRは3e-4で、459step以内では一定 |

### probe step と checkpoint（単一フレームと同じ）

| 長さ | probe step | checkpoint |
|---|---|---|
| 10step | `[0, 1, 2, 10]` | なし |
| 459step | `[0, 1, 2, 10, 100, 256, 459]` | `[459]` |

### 対象外

- 時間情報・動き情報の評価、時間に敏感なpositive/negative、同じ動画の別clipをpositiveにすること
- T≠4、フレーム間隔≠0.25秒、tubelet 2、チャンネル結合（patch埋め込みでフレームを混ぜる方式）、TimeSformer型の分離attention
- 学習可能な時間位置埋め込み
- batch・K・解像度・精度の変更、AMP、clipの事前書き出し
- 459stepより長い学習、LRの探索、複数seed
- ActivityNetの固定probe、表現品質の評価、LoRAとFull FTの優劣
- 逐次（シーケンシャル）入力、stride比較（10/22 MTGの64/32/16/8）
- 既存のpreset・profile・出力namespace・単一フレーム経路の変更

## 4. 実装要件（最小差分）

1. **clipの取り出し**：`activitynet_frames.py` に、1つのclipの複数時刻を取り出す関数を加える。
   - 結果は、各時刻を `decode_frame` で個別に取り出したものと、**画像としてbitwiseで一致しなければならない**。
   - 1回のseekから前向きにdecodeする。seekが最初の時刻を越えた場合の読み直しは、`decode_frame` と同じ規則にする。
2. **clipのdataset**：manifestの各時刻tについて、`[t − 3Δ, …, t]`（Δ=0.25秒、T=4）の4フレームを返す。
   - TwoCropsTransformは、各viewで4フレームに同じ拡張をかけ、`[T, 3, 224, 224]` を2つ返す。torchとPythonの乱数の状態を、各フレームの前に同じ状態へ戻す。消費する乱数の量は、1フレーム分と同じにする。
   - **T=1では、単一フレームのdatasetと同じindexで同じテンソル（bitwise）**を返さなければならない。
3. **Early Fusionのbackbone**：`[B, T, 3, H, W]` を受け付け、§3のE2・E3のとおりtokenを組み立てて、ViTの各層とlayernormに通し、CLSを返す。
   - **T=1では、今の `ViTMoCoBackbone` と同じ入力に対してCLSがbitwiseで一致しなければならない**（時間位置埋め込みはt=0でゼロ）。
   - gradient checkpointingは、この経路でだけ有効にする。既存の単一画像の経路は変更しない。
   - 学習対象のparameter集合は、単一フレームと同じでなければならない（LoRA：Q/V LoRA＋Projector、Full FT：query全体86,487,680）。
4. **新しいpresetとprofile**（名前の例：`conf/moco_vit_lr_activitynet_clip.yaml`、profile `lr_activitynet_clip`）
   - `moco_vit_lr_activitynet` と同じrecipeを使う。差分は、clipの設定（T=4、間隔0.25秒、clipの位置＝終点）と、gradient checkpointingだけにする。
   - `training.learning_rate` は必須（黙って補わない）。
   - 出力先：`log/moco_vit_lr_activitynet_clip/${model.train_mode}/activitynet_clips/lr_${training.learning_rate}/<timestamp>`
   - `run_report.json` と `config_identity` に、T・間隔・clipの位置・時間位置埋め込みの方式・gradient checkpointingを記録する。
5. **fail-closed**：GPUでの新しいprofileは、次以外をすべて拒否する。
   - `max_steps ∈ {10, 459}`。それぞれ§3の表のprobe stepとcheckpointに限る。
   - manifestは§2のもの（sha256で照合）。T=4、間隔0.25秒、clipの位置＝終点。
   - LRは正の有限値のみ。
   - 他のpresetから新profileを使うこと、他のprofileでclipを使うことも拒否する。
6. **診断**：固定probeは§3のE11の静止clipにする。probeのmanifestに、元の64枚の入力ハッシュと、静止clipの入力ハッシュの両方を記録する。
7. **比較器**：既存の `--comparison-axis train_mode` と `prefix` で、clipのrun同士を比較できること（学習入力・probe入力・step0の特徴の一致確認）。既存の比較器で足りるなら変更しない。clipのrunと単一フレーム／ImageNetのrunの組み合わせは、比較が成立しないこと。

## 5. 互換性

- 既存presetを使ったCPUとGPUの経路は、変更前とbitwiseで一致しなければならない（ImageNet、`lr_sensitivity`、`lr_long`、`lr_activitynet`（単一フレーム）、0.03、診断のON/OFF）。
- 全テストの失敗ID集合が、変更前後で同じであること（基準：`9155c2b` で36 failed / 907 passed）。

## 6. 検証（実装時、CPUのみ）

| Gate | 内容 |
|---|---|
| T0 | clipの取り出し：合成動画と、NASがある場合は実動画（A1で問題になった `v_5kBKAfEX7XA` を含む）で、各時刻の画像が `decode_frame` の結果とbitwise一致する。seekが目標を越える合成ケースでも一致する |
| T1 | T=1の一致：(a) clipのdatasetがT=1で単一フレームのdatasetと同じテンソル（拡張込み、同じseed）を返す。(b) Early FusionのbackboneがT=1で今のbackboneと同じCLSを返す（tiny ViTとLoRAの両方式） |
| T2 | 拡張の一貫性：T=4で、1つのviewの4フレームに同じcropと色変換がかかる（同じ画像を4枚並べたclipなら、4フレームの出力テンソルが一致する）。2つのviewは別の拡張になる |
| T3 | 時間位置埋め込み：t=0でゼロ、t≥1で非ゼロ。学習対象のparameterに含まれず、両方式で学習対象の集合が単一フレームと同じ |
| T4 | preset・profileの単体テスト：§4-5の拒否条件、出力先が別になること、`config_identity` にclipの設定が入ること |
| T5 | tiny ViT・2rank・glooで、人工のclip用manifest（テスト用）と小さなImageNet形式のprobeを使い、短いstep数で完走する。probe・checkpoint・summaryが揃い、元の64枚のprobe入力ハッシュの選び方が既存経路と同じ。gradient checkpointingのON/OFFで、step0の特徴とstep1のlossがbitwise一致する |
| T6 | 比較器：clipのLoRAとFull FTで `comparable=true`。10stepと長いrunの共通prefixを比較できる。clipのrunと単一フレームのrunの組み合わせは、比較が成立しない |
| T7 | 既存経路の非回帰と、失敗ID集合が変更前後で同じであること |

## 7. GPU run（A1〜A2、それぞれ別に許可が必要）

```bash
# A1: 10step（LoRA→Full FT）。VRAMとデータ読み込み時間を確かめる
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet_clip model.train_mode=lora training.max_steps=10 training.learning_rate=0.0003 'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet_clip model.train_mode=full_ft training.max_steps=10 training.learning_rate=0.0003 'diagnostics.probe_steps=[0,1,2,10]' 'training.checkpoint_steps=[]'
# A2: 459step（1epoch、LoRA→Full FT）。A1の両方式が工学PASSで、読み込み時間を確認した後
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet_clip model.train_mode=lora training.max_steps=459 training.learning_rate=0.0003
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.vit_moco.moco_vit_ddp_smoke --config-name moco_vit_lr_activitynet_clip model.train_mode=full_ft training.max_steps=459 training.learning_rate=0.0003
```

- 研究リポジトリをclean commitにしてから実行する。GPU構成は単一フレームと同じ（`CUDA_VISIBLE_DEVICES=0,1`）。実行はtmuxの専用セッションで行う。
- A1で、VRAMのpeak・headroom、`step_time_sec`、実際の1stepあたりの時間（データ読み込み待ちを含む）を記録する。
- **VRAM Gateが `BLOCKED` になった場合**（特にFull FTのGPU1）は、そこで止めて報告する。batch・T・精度を変えるかはユーザーが判断する（E9）。
- 1stepあたりの時間が単一フレームA2（LoRA約3.8秒、Full FT約2.5秒）より大幅に長い場合は報告し、A2を実行するかはユーザーが判断する。

## 8. 成功条件

### 工学的な成功（PASS / FAIL）

- 両方式・両方の長さで、`status=completed`、`failures=[]`、全stepがfinite、rank間でstateが一致、frozen baseが不変（LoRA）、VRAM Gate PASS、診断の `engineering_status=PASS` であること。
- 固定probeの元の64枚の入力ハッシュが `6af50745…08d8bf` であること。静止clipの入力ハッシュが、同じ長さのLoRAとFull FTで一致すること。
- 同じ長さのLoRAとFull FTで、step0の特徴・学習入力・probe入力が一致すること（比較器 `comparable=true`）。
- 459stepのrunで、checkpoint（step 459）と全probe stepの記録が揃っていること。
- **run間のbitwise一致は求めない**（GPU backwardの非決定性）。10stepと459stepのprefix比較は記録用で、step0の特徴とstep1のlossのbitwise一致を確かめ、それ以降はずれの大きさを報告する。

### 科学的な観測（記述的に報告し、閾値でPASS / FAILを決めない）

- loss、positive cosine、hardest negative cosine、`query_feature_std` の推移。**256stepより前と後を分けて**、50step窓の平均で示す。主な観測は「256stepより後に、窓ごとのlossが下がるか」。
- 固定probe（静止clip）の各stepにおける CLS / z_q / q の画像間cos p50・p95、mean_dim_std、row norm、step0からの変化。
- 既存の警告ヒューリスティック（`q mean_dim_std < 1e-4` かつ `cos p50 > 0.999`）が出るかどうか。
- 単一フレームA2との違いは記述にとどめる。入力が違うため、lossの値の大小から優劣を言わない。
- 1stepあたりの時間（計算とデータ読み込み待ち）と、VRAM。

lossが下がらない場合も、collapseした場合も、VRAMや読み込みで止まった場合も、そのまま結果として報告する。

### 主張しないこと

時間情報・動き情報を獲得したこと（MoCoのpositiveが同じclipの2拡張なので、フレームを平均するだけでも解ける）、表現品質、0.25秒・T=4の妥当性、LoRAとFull FTの優劣。

## 9. 実行の境界

- このspecの承認は、**研究コードの実装を許可するものではない**。実装は `engineering-task` で、ユーザーが改めて指示してから行う。
- GPU run（§7のA1、A2）は、実装とCPU検証の後に**それぞれ別に許可を得る**。
- commit、push、ブランチ操作は、ユーザーの指示があったときだけ行う。

## 10. 未決事項

- **blocking**：なし（E1〜E12は2026-11-03に採用済み）。
- **non-blocking**（実装時に既存の方式に合わせる）：
  - presetとprofileの名前、configのkey名
  - clipを取り出す関数とdatasetの名前・置き場所
  - gradient checkpointingの呼び出し方（transformersの既存機能を使い、non-reentrantにする）
  - テスト用の人工manifestと合成動画の作り方
  - 比較器が成立しない組み合わせを拒否するときのメッセージ

## 11. 承認記録

- 2026-11-03：ユーザーが、[壁打ち](../../../../secretary/notes/brainstorm/2026-11-03-activitynet-clip-early-fusion-loss-check.md)で次を採用した。
  - token結合（E2）、同じclipの2拡張（E7）、T=4（E4）
  - 未決事項U1〜U9の推薦案（E3、E6、E8〜E12）
- 同日、フレーム間隔0.25秒（E5）と、clipの位置＝最後のフレーム（E6）を選んだ。

## 12. 実装記録

- 2026-11-03：`feature-vit-moco@9155c2b` の上に未commitで実装した。CPU Gate T0〜T7はすべてPASS（同日 `feature-vit-moco@fe58fc3` としてcommit、push未実施）。全テストは36 failed / 957 passedで、失敗IDの集合は変更前と同じ。実動画2,005 clipで、`decode_clip` と `decode_frame` の不一致は0件だった。
  - **§2からの変更点**：`DistributedMoCo` の入力チェックは4次元だけを受け付けていた（§2の記述は誤り）。そこで `ViTDistributedMoCo` に `clip_frames` を加え、clipのrunだけ5次元の入力を受け付けるようにした。画像runの経路は変えていない。
  - 前の段階のテストのうち、`PRESETS` 全体を固定していた1件のassertを、既存の3件の確認に変えた。
  - GPU A1/A2は未実施。実行前に研究コードのcommitが必要。詳細は[実装・CPU検証記録](../experiments/2026-11-03-vit-moco-lora-activitynet-clip-early-fusion-implementation-cpu-verification.md)。
- 2026-11-03：GPU A1（10step）とA2（459step）を、LoRA→Full FTの順に実行し、すべて工学PASSだった。同じ長さのLoRAとFull FTは `comparable=true`。A2では両方式とも、256stepより後の50step窓のlossが単調に減少し、collapseは観測されなかった（暫定）。詳細は[A1 LoRA](../experiments/2026-11-03-vit-moco-lora-activitynet-clip-a1-lora-gpu-evidence.md)・[A1 Full FT](../experiments/2026-11-03-vit-moco-lora-activitynet-clip-a1-full-ft-gpu-evidence.md)・[A2 LoRA](../experiments/2026-11-03-vit-moco-lora-activitynet-clip-a2-lora-gpu-evidence.md)・[A2 Full FT](../experiments/2026-11-03-vit-moco-lora-activitynet-clip-a2-full-ft-gpu-evidence.md)のEvidence。
