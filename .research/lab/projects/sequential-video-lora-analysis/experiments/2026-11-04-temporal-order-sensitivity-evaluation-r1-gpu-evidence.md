---
project: sequential-video-lora-analysis
date: 2026-11-04
record_type: gpu-eval-evidence
status: r1-completed
research_repository: tamaki-lab/2026_09_ishikawa_sequential-video-lora
research_branch: feature-vit-moco
research_commit: 50ff715773137388ba51850532660d26daa4501a
spec: ../specs/2026-11-03-temporal-order-sensitivity-evaluation-spec.md
---

# 時間情報の評価（並び順への反応）：R1（GPU評価）

[承認済みspec](../specs/2026-11-03-temporal-order-sensitivity-evaluation-spec.md)のR1を、ユーザーの許可を得て実行した。実装とR0（validation manifest）は[実装記録](2026-11-04-temporal-order-sensitivity-evaluation-implementation-cpu-verification.md)を参照。

- **5モデルすべてで、I3（fixed probeの特徴の再計算）が学習時の記録とbitwise一致した**（CLS／z_q／q）。読み込みは正しい。
- **どのモデルも、shuffle・reverseでのcosはほぼ1（0.968〜0.998）だった**。staticだけが大きく下がった（0.856〜0.916）。
- 学習したモデル（B・C）は、学習前（A）より並び順に**鈍く**なった。clipで学習したCは、単一フレームで学習したBより、さらにわずかに鈍い。
- spec §9の読み方では「どのモデルもcosがほぼ1」かつ「staticだけ大きく下がる」に当たる。**今のclip学習で、並び順を区別する情報が身についたという観測は得られなかった**（記述的な観測。1 seed・2,000 clip・459step）。

## 実行

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.vit_moco.temporal_order_eval \
  --manifest log/activitynet_frames/2026-11-04_validation_interval10s_offset5s/manifest.json \
  --clip-config $CL/resolved_config.yaml --init-reference $CL \
  --model A=lora:init \
  --model B-LoRA=lora:$BL/checkpoint_step000459.pt \
  --model B-FT=full_ft:$BF/checkpoint_step000459.pt \
  --model C-LoRA=lora:$CL/checkpoint_step000459.pt \
  --model C-FT=full_ft:$CF/checkpoint_step000459.pt
```

| 記号 | run（研究リポジトリ `log/` 内） |
|---|---|
| `BL`（単一フレーム LoRA A2） | `moco_vit_lr_activitynet/lora/activitynet_frames/lr_0.0003/2026-11-03_10-48-36-408110` |
| `BF`（単一フレーム Full FT A2） | `moco_vit_lr_activitynet/full_ft/activitynet_frames/lr_0.0003/2026-11-03_11-49-57-818631` |
| `CL`（clip LoRA A2） | `moco_vit_lr_activitynet_clip/lora/activitynet_clips/lr_0.0003/2026-11-03_15-22-36-985580` |
| `CF`（clip Full FT A2） | `moco_vit_lr_activitynet_clip/full_ft/activitynet_clips/lr_0.0003/2026-11-03_16-46-46-287044` |

- 研究コード `feature-vit-moco@50ff715`、clean。tmuxセッション `vit-moco-a1`。
- GPU 0（RTX PRO 4500 Blackwell）だけを使った。学習時のrank 0（probeを記録した側）と同じGPUにそろえた。開始前はGPUが空いていた。
- 既定値のまま：2,000 clip、seed 0、microbatch 8、decode worker 16。
- 時刻：07:44:43開始 → 07:57:13終了（12分30秒。うち動画のdecodeは181秒、評価本体は740秒）。exit 0。最大RSSは約6.7 GB。
- 出力：`log/temporal_order_eval/2026-11-04_07-44-51/`

| ファイル | sha256 |
|---|---|
| `summary.json` | `a9adca69916e56c1635a2ba3cfab72173ee1a453644cacef459f3f4dc3c9edde` |
| `per_clip.jsonl` | `11eb9a0b3b15ae4c389261b66378b6fdf42ba16fdbbec97ad62592a452bf67e8` |

## 選んだclip

- 2,000本の動画から1 clipずつ（seed 0）。same-videoのclipがない動画は67本で、same-videoの集計だけから外した（n=1,933）。
- 動きの大きさ（隣り合うフレームの画素の平均絶対差）の中央値は0.0675。大きい側・小さい側に1,000 clipずつ分けた。

## 読み込みの確認（I3）

| モデル | 比べた記録 | checkpointのsha256（先頭） | CLS・z_q・q |
|---|---|---|---|
| A | clip LoRA runのstep 0 | （init） | 一致 |
| B-LoRA | 単一フレーム LoRA runのstep 459 | `7b7c5b976cec` | 一致 |
| B-FT | 単一フレーム Full FT runのstep 459 | `55493a3ae7cc` | 一致 |
| C-LoRA | clip LoRA runのstep 459 | `dc82a447f5f4` | 一致 |
| C-FT | clip Full FT runのstep 459 | `c0e02d94189f` | 一致 |

全モデルで、probeの入力も記録と一致した（`probe_input_equal`）。`reference_checks_equal: true`。

## 結果：normalとのcos類似度（平均、全2,000 clip）

### CLS（backboneの出力）

| モデル | shuffle | reverse | static | same-video（10秒離れたclip） | other-video |
|---|---|---|---|---|---|
| A | 0.9813 | 0.9676 | 0.8557 | 0.6003 | 0.0811 |
| B-LoRA | 0.9893 | 0.9806 | 0.8978 | 0.6517 | 0.0938 |
| B-FT | 0.9956 | 0.9916 | 0.9156 | 0.6552 | 0.0514 |
| C-LoRA | 0.9921 | 0.9858 | 0.8967 | 0.6532 | 0.0929 |
| C-FT | 0.9977 | 0.9955 | 0.9067 | 0.6538 | 0.0461 |

### q（正規化したMoCoの特徴）

| モデル | shuffle | reverse | static | same-video | other-video |
|---|---|---|---|---|---|
| A | 0.9824 | 0.9699 | 0.8765 | 0.6908 | 0.3628 |
| B-LoRA | 0.9912 | 0.9841 | 0.9123 | 0.6785 | 0.0327 |
| B-FT | 0.9967 | 0.9937 | 0.9321 | 0.6968 | 0.0349 |
| C-LoRA | 0.9934 | 0.9879 | 0.9077 | 0.6718 | 0.0221 |
| C-FT | 0.9983 | 0.9966 | 0.9233 | 0.6937 | 0.0338 |

Aのqは、学習していないprojector（seed 0の初期値）を通した値である（spec §9）。other-videoが0.36と高いのはそのためと考えられる。

### 動きの大小別（CLS、shuffle／reverse／static）

| モデル | 動き大 | 動き小 |
|---|---|---|
| A | 0.9738／0.9546／0.7981 | 0.9888／0.9806／0.9133 |
| B-LoRA | 0.9850／0.9725／0.8517 | 0.9935／0.9887／0.9439 |
| B-FT | 0.9941／0.9883／0.8729 | 0.9971／0.9949／0.9584 |
| C-LoRA | 0.9893／0.9803／0.8511 | 0.9950／0.9912／0.9424 |
| C-FT | 0.9968／0.9937／0.8613 | 0.9985／0.9973／0.9522 |

動きの大きいclipほど、どのモデルでもcosは下がる。モデルの順位（A < B・C、Bより Cがわずかに高い）は、動きの大小で変わらない。qでも同じ傾向だった（`summary.json`）。

### モデル間の差（同じclipどうし、平均と95%信頼区間、CLS・全clip）

| 差 | shuffle | reverse | static |
|---|---|---|---|
| C-LoRA − A | +0.0108 [+0.0100, +0.0118] | +0.0182 [+0.0171, +0.0194] | +0.0410 [+0.0391, +0.0429] |
| C-LoRA − B-LoRA | +0.0029 [+0.0025, +0.0033] | +0.0052 [+0.0046, +0.0058] | −0.0011 [−0.0019, −0.0003] |
| B-LoRA − A | +0.0080 [+0.0074, +0.0086] | +0.0130 [+0.0122, +0.0139] | +0.0421 [+0.0403, +0.0439] |
| C-FT − A | +0.0163 [+0.0152, +0.0176] | +0.0279 [+0.0263, +0.0297] | +0.0510 [+0.0482, +0.0540] |
| C-FT − B-FT | +0.0020 [+0.0017, +0.0025] | +0.0039 [+0.0033, +0.0046] | −0.0089 [−0.0108, −0.0071] |
| B-FT − A | +0.0143 [+0.0133, +0.0155] | +0.0240 [+0.0226, +0.0255] | +0.0599 [+0.0570, +0.0630] |

正の値は、「並べ替えても特徴が変わりにくい（順番に鈍い）」ことを意味する。qでも符号は同じだった（C−Bはshuffle +0.0016〜+0.0021、reverse +0.0028〜+0.0038、static −0.0047〜−0.0088）。

## 読み方（spec §9に沿った記述的な観測）

1. **並び順の変化は、内容の変化に比べてとても小さい**。C-LoRAのCLSでは、reverseでの変化（1−cos）は0.014で、10秒離れた同じ動画のclipでの変化（0.347）の約1/25である。
2. **学習すると、並び順にさらに鈍くなった**。shuffle・reverseのcosは、B・CともAより高い（信頼区間は0を含まない）。
3. **clipで学習しても、並び順を区別するようにはならなかった**。C−Bはshuffle・reverseで正（Cのほうが鈍い）。spec §9の「Cだけ下がる」には当たらない。
4. **staticでは、どのモデルもcosが大きく下がる**（0.86〜0.92）。staticでだけ、CはBよりわずかに下がる（C−B：LoRA −0.001、Full FT −0.009）。ただし、この差はshuffle・reverseでの差と同じくらい小さい。staticの下がり方には、動きの有無と、3枚のフレームの見た目がなくなったことの両方が混ざっている（spec、反例1）。
5. spec §9の表では、**「どのモデルもcosがほぼ1」かつ「staticだけ大きく下がる」**に当たる。今のclip学習（同じclipの2拡張をpositiveにするMoCo）では、時間情報（並び順）は入っていない、という事前の予想と合っている。

**主張しないこと**（spec §9）：動作の理解、表現の品質、長い学習での結果、0.25秒・T=4の妥当性、LoRAとFull FTのどちらが優れているか。今回の結果は1 seed・2,000 clip・459stepのものである。Full FTのcosがLoRAより1に近いことも、方式の優劣とは読まない。

## 次の判断の材料（未採用・TODO未追加）

- 今回の数値は、学習方法を変えたときに比べる基準として使える。
- 並び順を学習させるには、positive／negativeの作り方（たとえば、並べ替えたclipをnegativeにする、時間的に離れたclipをpositiveにする）など、学習の目的を変える必要がありそう。これは壁打ちで検討する。
