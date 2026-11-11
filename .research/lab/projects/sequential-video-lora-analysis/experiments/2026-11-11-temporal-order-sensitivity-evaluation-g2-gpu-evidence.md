---
project: sequential-video-lora-analysis
date: 2026-11-11
record_type: gpu-eval-evidence
status: completed
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: 559073f517a082e41844ba495e90d57035fe2bdf
spec: ../specs/2026-11-03-temporal-order-sensitivity-evaluation-spec.md
---

# 時間情報の評価（並び順への反応）：G2を加えた評価

[承認済みspec](../specs/2026-11-03-temporal-order-sensitivity-evaluation-spec.md)の評価を、2026-11-10の追記（sequential-loaderで学習したrunを受け付ける）に従って、ユーザーの許可を得て実行した。R1の5モデルに、G2（sequential-loader、stride 16で6,255step学習したclip LoRA）を加えた。

## 要点

- **G2を含む6モデルすべてで、固定probeの特徴の再計算が学習時の記録と一致した**（`reference_checks_equal: true`。G2はstep 6,255、`probe_input_equal` と `equal` がともにtrue）。読み込みは正しい。
- **R1の5モデルの結果は、R1とbitwise一致した**（`results` と `differences` が完全に同じ）。同じclip、同じ乱数で評価できている。
- **G2は、shuffle・reverseのcosがR1のどのモデルよりも1に近かった**（q：shuffle 0.9982、reverse 0.9971）。G2は、ランダム順でclip学習したC-LoRAより、並び順にさらに**鈍い**。
- staticのcosも、G2が最も高かった（q 0.933）。同じ動画の10秒後のclip（same_video）とも、G2が最も近かった。
- **今回の結果からも、並び順を区別する情報が身についたという観測は得られなかった**（記述的な観測。1 seed、2,000 clip）。逐次の入力で長く学習すると、むしろ時間の変化に鈍くなる方向だった。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.vit_moco.temporal_order_eval \
  --manifest log/activitynet_frames/2026-11-04_validation_interval10s_offset5s/manifest.json \
  --clip-config $CL/resolved_config.yaml --init-reference $CL \
  --model A=lora:init \
  --model B-LoRA=lora:$BL/checkpoint_step000459.pt \
  --model B-FT=full_ft:$BF/checkpoint_step000459.pt \
  --model C-LoRA=lora:$CL/checkpoint_step000459.pt \
  --model C-FT=full_ft:$CF/checkpoint_step000459.pt \
  --model G2-LoRA=lora:$G2/checkpoint_step006255.pt
```

| 記号 | run（研究リポジトリ `log/` 内） |
|---|---|
| `BL`、`BF`、`CL`、`CF` | [R1](2026-11-04-temporal-order-sensitivity-evaluation-r1-gpu-evidence.md)と同じ |
| `G2`（sequential-loader clip LoRA、stride 16） | `moco_vit_seq_activitynet_clip/lora/activitynet_sequential_clips/lr_0.0003/2026-11-08_17-26-03-055153`（G2の再開後のrun。checkpoint sha256 `56cc7d8b…`） |

- 研究コード `feature-vit-moco@559073f`、clean。tmuxセッション `vit-moco-a1`。
- GPU 0（RTX PRO 4500 Blackwell）だけを使った（G2の学習時にprobeを記録したrank 0と同じGPU）。開始前は3枚とも空いていた。
- 既定値のまま：2,000 clip、seed 0、microbatch 8、decode worker 16。評価に使うclipはR1と同じで、0.25秒間隔の4フレーム。
- 時刻：2026-11-11 10:36:49開始 → 10:53:01終了（16分12秒。うち動画のdecodeは297秒、評価本体は963秒）。exit 0。最大RSSは約7.1 GB。
- 出力：`log/temporal_order_eval/2026-11-11_10-36-56/`
- clipの選び方はR1と同じ：2,000本。same_videoがないもの67本は、same_videoの集計から除外。motionの中央値は0.0675で、high／lowが1,000本ずつ。

## 結果：normalとのcos（平均、2,000 clip）

### CLS（backboneの出力）

| model | shuffle | reverse | static | same_video | other_video |
|---|---|---|---|---|---|
| A（学習前） | 0.9813 | 0.9676 | 0.8557 | 0.6003 | 0.0811 |
| B-LoRA（単一フレーム） | 0.9893 | 0.9806 | 0.8978 | 0.6517 | 0.0938 |
| B-FT | 0.9956 | 0.9916 | 0.9156 | 0.6552 | 0.0514 |
| C-LoRA（clip、ランダム順） | 0.9921 | 0.9858 | 0.8967 | 0.6532 | 0.0929 |
| C-FT | 0.9977 | 0.9955 | 0.9067 | 0.6538 | 0.0461 |
| **G2-LoRA（clip、逐次、6,255step）** | **0.9974** | **0.9956** | **0.9280** | **0.6939** | 0.0793 |

### q（MoCoの出力、正規化後）

| model | shuffle | reverse | static | same_video | other_video |
|---|---|---|---|---|---|
| A | 0.9824 | 0.9699 | 0.8765 | 0.6908 | 0.3628 |
| B-LoRA | 0.9912 | 0.9841 | 0.9123 | 0.6785 | 0.0327 |
| B-FT | 0.9967 | 0.9937 | 0.9321 | 0.6968 | 0.0349 |
| C-LoRA | 0.9934 | 0.9879 | 0.9077 | 0.6718 | 0.0221 |
| C-FT | 0.9983 | 0.9966 | 0.9233 | 0.6937 | 0.0338 |
| **G2-LoRA** | **0.9982** | **0.9971** | **0.9331** | **0.6995** | **0.0082** |

### 動きの大小別（q、high／low）

| model | shuffle | reverse | static |
|---|---|---|---|
| A | 0.9753 / 0.9895 | 0.9580 / 0.9817 | 0.8303 / 0.9227 |
| B-LoRA | 0.9875 / 0.9949 | 0.9771 / 0.9911 | 0.8709 / 0.9538 |
| C-LoRA | 0.9908 / 0.9959 | 0.9830 / 0.9927 | 0.8648 / 0.9506 |
| G2-LoRA | 0.9975 / 0.9989 | 0.9959 / 0.9983 | 0.8928 / 0.9733 |

## G2と他のモデルの差（paired bootstrap、平均［95%区間］）

評価器が自動で計算する差は、A・B・Cの名前の組み合わせだけ（`difference_pairs`）。G2の差は、出力の `per_clip.jsonl` から、評価器と同じ関数（`temporal_order_eval.summarise`、seed 0、1,000回）で計算した。同じ方法でC-LoRAの値を計算し直すと、評価器の `summary.json` と一致した。

| 差（q、全clip） | shuffle | reverse | static | same_video | other_video |
|---|---|---|---|---|---|
| G2-LoRA − C-LoRA | +0.0049 [+0.0045, +0.0054] | +0.0092 [+0.0085, +0.0100] | +0.0254 [+0.0234, +0.0277] | +0.0277 [+0.0233, +0.0318] | −0.0139 [−0.0183, −0.0095] |
| G2-LoRA − B-LoRA | +0.0070 [+0.0064, +0.0077] | +0.0130 [+0.0120, +0.0141] | +0.0207 [+0.0186, +0.0230] | +0.0210 [+0.0165, +0.0255] | −0.0245 [−0.0291, −0.0199] |
| G2-LoRA − A | +0.0158 [+0.0147, +0.0170] | +0.0272 [+0.0257, +0.0289] | +0.0565 [+0.0532, +0.0599] | +0.0086 [+0.0002, +0.0165] | −0.3547 [−0.3599, −0.3491] |

- CLSでも同じ向きだった。G2-LoRA − C-LoRA は、shuffle +0.0052、reverse +0.0099、static +0.0313、same_video +0.0407、other_video −0.0136。いずれも95%区間が0をまたがない。
- shuffle・reverse・staticでG2との差が大きいのは、動きの大きいclip（high）のほうだった。例：q reverseで、G2 − C-LoRA は high +0.0129、low +0.0056。

## 解釈（記述的。spec §9の範囲）

1. G2も「どのモデルもcosがほぼ1」かつ「staticだけ大きく下がる」に当たる。**逐次の入力で学習しても、並び順を区別する情報が身についたという観測は得られなかった**。
2. G2は、R1のどのモデルよりも、並び順の入れ替え（shuffle・reverse）にも、動きをなくすこと（static）にも鈍い。10秒離れた同じ動画のclipとの類似度も上がった。**時間の変化に対して不変な（動画全体で同じような）特徴に近づいた**と読める。
3. 一方で、other_video（別の動画）とのqのcosはほぼ0になった（0.008）。別の動画どうしを引き離す方向は、R1のモデルより進んでいる。これは、G2の学習lossの低下の大部分が、他の動画との区別から来ていたという見方と合っている。2026-11-10の会話で、G2のlogから次のことを確認した（壁打ちメモには未保存）：
   - 平均のnegativeのcosはほぼ0になった
   - 最後の窓のlossは、正例のcos 0.889から計算できる下限（約6.74）に近かった
4. 逐次の入力では、同じ動画の少し前のclipがQueueのnegativeに入る。そのため、同じ動画の中の違いを見分ける圧力が働くかもしれない、という仮説があった（H3）。しかし、この評価では、そうした方向（同じ動画の中で特徴が離れる、並び順に敏感になる）は見えなかった。

## 主張しないこと・注意点

- **G2とC-LoRAの違いは、逐次で読んだことだけによるものではない**。条件が複数違う：
  - 学習量：6,255step と 459step（約13.6倍）
  - clipのフレーム間隔：stride 16フレーム（30 fpsで約0.53秒）と 0.25秒
  - データの読み方：sequential-loaderで動画全体を読む、manifestの10秒ごとの時刻から作る
- G2の学習時のフレーム間隔は約0.53秒だが、評価は0.25秒間隔のclipで行った（R1と比べるため）。G2の学習時の間隔に合わせた評価は行っていない。
- 1 seed、ラベルを使わない評価。表現の良し悪し（下流タスクの性能）については何も言えない。
- G2の差は、評価器の外で、評価器と同じ関数を使って計算した。`summary.json` には入っていない。

## 次の候補（未採用・TODO未追加）

- 学習量と読み方の効果を分けて見る：G2の途中のcheckpoint（step 1,000〜6,000）を同じ評価にかける。
- G2の学習時の間隔（約0.5秒）に合わせたclipでの評価。
- 並び順を学ばせる学習目標の検討（並べ替えたclipをnegativeにする、など。R1の記録の「次の候補」と同じ論点）。壁打ちで扱う。
