---
project: sequential-video-lora-analysis
date: 2026-10-17
analysis_mode: explain
detail: full
level: mid
source_repository: https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora
source_commit: c50ab79cbce1738a571daf6e018e31bb870cae24
source_dirty: false (uploaded artifact reports clean checkout; remote cannot see server local state)
reference_repository: https://github.com/facebookresearch/moco
reference_commit: a419ef282fc9885015674e77f58c59447206a27f
related_spec: .research/lab/projects/sequential-video-lora-analysis/specs/2026-10-17-meta-moco-v2-imagenet-ddp-parity-spec.md
source_artifact: user-uploaded moco_parity_report/v1 created 2026-10-17T19:19:44
targets:
  - scripts/parity/compare_moco_parity.py
  - scripts/parity/moco_imagenet_parity.py
  - self_supervised/moco/ddp_moco.py
  - self_supervised/moco/standalone_moco.py
---

# Code Deep Dive: A1 16step parityのBackward差分を解析

## 対象と結論

ユーザー提供のA1比較JSONを解析。比較は公式Meta・自作とも2rank×16step完走、入力fixture SHA一致、preconditions空、within-run必須16チェックは双方PASS。一方で公式対自作の浮動小数点tensor比較は閾値超過。A1数値一致はまだ証明されていない。研究サーバでの新規テスト・GPU実行は一切行っていない。

- 判定: BLOCKED。理由は暫定閾値であること、および候補閾値で多数の実際のfloat不一致。
- 比較総数: 16000（個々のtensor/項目の比較）、float不一致9398、exact不一致0、bitwise一致5167。
- 9398はスカラー要素数でも個別不具合数でもない。大量tensorの完全な保存・比較を行ったのはstep1,8,9,16のみで、比較密度は一様ではない。
- 全stepのinput im_q/im_k・shuffle索引は一致（各64/64）。
- 内部16チェックPASSは同じrun内でのEMA、SGD、Queue/Pointer、allreduce、finiteなどを確認するもので、公式対自作の値一致ではない。

## 最初の差異（step1）

最初に許容差を超過したのはrank0、after_backward、query_grads、encoder_q.conv1.weight。

| 比較地点 | 実測 |
|---|---|
| Step1の初期Query/Key重み、BN、Queue、入力、Shuffle | 比較上すべて一致 |
| Step1 z_q, z_k_shuffled, 正規化q/k | bitwise一致 |
| positive logits | 最大絶対誤差9.5367431640625e-7（暫定許容差内） |
| Loss | 最大絶対誤差1.7881393432617188e-7（同上） |
| conv1のgradient | 最大絶対誤差1.5497207641601562e-5、平均2.8021318432526524e-6。9408要素中852が閾値超過 |
| SGD後conv1 weight | 最大絶対誤差1.1618249118328094e-7（暫定許容差内、bitwise不一致） |

Step1は24/3910 tensor比較がfloat閾値を超過。その内訳は2rank×6個のQuery gradientと2rank×6個のSGD momentum buffer。初期状態が違うのではなく、Backwardのどの演算から差が入るかを切り分ける必要がある。

## 後続stepの差の波及

| step | float閾値超過 / 比較数 | after_queryのz_q最大絶対誤差（2rankの大きい方） |
|---|---:|---:|
| 1 | 24 / 3910 | 0 |
| 2 | 10 / 30 | 0.00010713934898376465 |
| 3 | 14 / 30 | 0.39787301421165466 |
| 8 | 2992 / 3910 | 0.5829623639583588 |
| 9 | 3042 / 3910 | 0.5380070209503174 |
| 16 | 3176 / 3910 | 1.0304614305496216 |

浮動小数点勾配差→Momentum→weight更新差→次step特徴量/BN/EMA/Queueへの差の増幅がひとつの説明仮説。ただし増幅機構は未立証。後続の9398項目を独立した9398件のバグと解釈しない。

## コード照合と有力な診断仮説

固定Meta moco/builder.pyはpositiveをtorch.einsum("nc,nc->n")、negativeをtorch.einsum("nc,ck->nk")、logitsをinplace除算する。自作 standalone_moco.py のmoco_logitsはpositiveを(q*k).sum、negativeをq @ queue、logitsをout-of-place除算する。数式は同じでもCUDA kernel/reduction/autogradのbitwise結果は保証されない。Step1の微小なpositive logits・Loss差や勾配差への寄与が考えられるが、直接の原因かは未確認。GPUの非決定性、DDP bucket/reduce順、記録hookの副作用も代替仮説。

## 次に実施する検証（ユーザー許可前に実行しない）

1. 提出されたrun報告、各rank trace、fixture SHAとcommitをそのまま保存。ログを上書きしない。
2. **同一Metaを同じA1 fixture・seed・全16step/full trace条件で新規run**し、compare_moco_parity.pyのpair-kind rerun / threshold-status provisionalでMeta対Metaの数値差を測る。必要ならSelf対Selfも繰り返す。
3. Meta再演でもstep1 after_backwardが同程度ずれるなら、GPU演算非決定性・RNG/backend/DDPを優先確認。Meta再演が一致する一方公式対自作だけずれるなら、演算差・autograd・DDP構造を調べる。
4. 固定q/k/Queue/targetsの単一演算で、Metaのeinsumと自作のsum/matmul、inplace divisionでlossとdLoss/dqが一致するか局所比較する。必要に応じて単一GPU／DDPなしでのbackwardと2GPU allreduceを切り分ける。研究コード編集は別途承認・handoff。
5. 根拠のないrtol/atol拡大を行わず、原因特定とA1の再検証・事前の閾値固定を優先する。A2/A3/A5/Spec Bを今は実施しない。

## 根拠レベル

- Observed from user-supplied report: counts、最初の差、16stepの出力、内部16チェック、provenanceとfixture一致。実機での再測定ではない。
- Observed from GitHub source: 公式einsum対自作sum/matmul、比較器が暫定閾値をBLOCKEDにする実装。
- Inferred: Backward差の後続波及、表現等価な演算方法の数値差の影響。
- Unverified: 公式の再実行差、Backprop微差の根因、GPU backend非決定性、必要な修正とその影響。
- Status: Research Workspace Spec A draft。今回の調査は研究コード、spec、README、実験ログを変更せず、コード解析文書のみ保存。

## Sources

- https://github.com/facebookresearch/moco/blob/a419ef282fc9885015674e77f58c59447206a27f/moco/builder.py
- https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/c50ab79cbce1738a571daf6e018e31bb870cae24/self_supervised/moco/standalone_moco.py
- https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/c50ab79cbce1738a571daf6e018e31bb870cae24/self_supervised/moco/ddp_moco.py
- https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora/blob/c50ab79cbce1738a571daf6e018e31bb870cae24/scripts/parity/compare_moco_parity.py
