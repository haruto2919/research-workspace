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

---

## 2026-10-17 19:21 JST：公式A1の再実行（追記）

ユーザー提供の研究サーバ実行ログによると、Meta公式の新しいA1 run（meta_reference_a1_rerun）は16/16 stepを完了し、queue_ptrは各stepで8進んでstep8と16で0へ戻った。Meta公式の再実行比較（pair_kind=rerun、threshold_status=provisional）は、**候補誤差rtol=1e-5/atol=1e-6を超えるfloat tensorが0件**と報告された。正式status=BLOCKEDは閾値がまだprovisionalだからである。これは公式の再現性に関する有益なEvidenceだが、bitwise一致や全詳細Tensorの照合までは意味しない。

### 重要な比較範囲の確認待ち

今回の再実行コマンドは最初のA1のoverride（queue_size=64、local_batch_size=4、image_size=64、max_steps=16、trace.full_steps=[1,8,9,16]、logging.comet_enabled=false）を省略している。GitHub devの設定既定はQueue65536、local32、224px、1step、full_steps=[1]、Comet ON。一方、実ログは16stepかつqueue_ptrが8ずつ進むため、ローカルの実行コードまたは設定がGitHub版と異なる可能性があり、現況不明。比較器のpreconditionsにはtrace.full_stepsが含まれないので、比較statusに前提不一致が出なくても、step8/9/16の詳細Tensorが省略されている可能性がある。

次に双方のresolved_config.yamlにおける queue_size、local_batch_size、image_size、max_steps、trace.full_steps、logging.comet_enabled を調べ、compare_meta_rerun_a1/parity_report.jsonのpreconditions、steps_compared、full_tensor_steps、counts.judged/exact_failures/close_failures/bitwise_equal、within_run_checksを確認する。元の公式対自作レポートはjudged=16000、full_tensor_steps=[1,8,9,16]。同範囲で比較できていなければ、元runと同じoverrideでMeta公式を新run directoryに再実行する。

Meta対Metaのfull trace全項目が候補閾値内なら、公式対自作に固有のstep1 backward差の可能性が上がる。次に同一q/k/Queueで公式einsumと自作sum/matmulのlogits・loss・dLoss/dqを局所比較し、DDPとGPU非決定性も切り分ける。暫定閾値を根拠なく緩めず、A2は保留する。

本追記はユーザーのログとGitHubコードに基づく解釈であり、研究サーバでの再実行・コード変更は行っていない。Spec A status draftを変更しない。

---

## 2026-10-17 A1 official rerunの完全比較範囲が確認済み

### ユーザーが提示したresolved configおよびparity_reportの実測値

- `meta_reference_a1`、`meta_reference_a1_rerun`、`self_a1`のすべて：Queue K64、local batch4、image_size64、max_steps16、full trace step `[1,8,9,16]`、Comet OFF。過去記録にあった「再実行時にoverride省略のため設定がデフォルトへ戻った可能性」は**今回の報告によって否定**された。
- Meta→Metaの比較：preconditions=`[]`、steps_compared=`[1..16]`、full_tensor_steps=`[1,8,9,16]`、judged=16000、exact_failures=0、close_failures=0、**bitwise_equal=16000**、first_failure=null。したがって比較した全16,000項目はbitwise一致（このrun、環境、測定点での主張に限定）。status=BLOCKEDは閾値がprovisionalであるためのみ。
- Meta→Selfの比較：同一steps・trace・judged=16000、exact_failures=0、close_failures=9398、bitwise_equal=5167、first_failureはstep1 rank0 phase=after_backward group=query_grads tensor=encoder_q.conv1.weight（shape `[64,3,7,7]`、max_abs_error=1.5497207641601562e-5、mean_abs_error=2.8021318432526524e-6、9408要素中852が候補閾値超過、max_rel_error=0.02007は微小分母の影響に注意）。結果はprovisionalのためBLOCKEDだが、現在の候補閾値を正式化すればMeta→SelfはFAIL。

### 更新後の原因切り分けと提案

1. **公式自身の非決定性が今回の9,398項目不一致を説明する可能性は低下**。Meta vs Metaのfull traceもbitwise一致しており、以前のtrace範囲の懸念は解消。ただしSelf側の非決定性・実装差は未検証。
2. 次は**Self→Self**の全16step同fixture再実行（別dir）で再現性を確認（GPU実行には許可を要する）。
3. Self自身もbitwise一致するなら、固定q/k/Queueで公式の`einsum`対自作の`sum+matmul`＋除算形式に限定したForwardと`dLoss/dq`のCUDA局所テストを優先。差が再現すれば算術表記による勾配差の仮説を強められる。さらに単一GPUでの勾配を比較してDDP allreduce要因を切り分ける。
4. Self→Self不一致なら、まず同一実装内のCUDA/cuDNN/DDP、hook、permutation/RNG、保存時点などの非決定性を調査する。
5. 現在の比較目的は**公式vs自作の数値パリティ**。数値許容差を事後的に広げること、原因不明のままA2/A3/Spec Bに進むことを避ける。公式ソースは変更せず、自作計算差を実証してから必要な変更の承認を判断。

**Evidence境界：** 16,000項目一致はユーザー提供したレポート内容に基づく。GitHubソースでは両者のlogits演算表記差を確認したが、その演算差がconv1勾配差の根本原因であることは未実証。研究サーバの実プロセスはChatGPTから操作していない。

---

## 2026-10-17 A1 Self→Self rerunの暫定許容差内一致

### ユーザー提供の追加実測

- 自作MoCo `implementation=self gate=A1` を既存fixture `manual_20261017_191704/fixture_a1/fixture.pt` で再実行し、1〜16step完走（queue_ptrはstep8/16で0、1stepごと8進む）。run出力：`.../manual_20261017_191704/self_a1_rerun`。
- `compare_moco_parity --pair-kind rerun --reference self_a1 --candidate self_a1_rerun --threshold-status provisional` のコンソール結果は `BLOCKED: threshold is provisional ... 0 float tensors outside the provisional tolerance`。少なくとも候補閾値では自作同士のfloat不一致は0件である。
- **まだ未提示**：`compare_self_rerun_a1/parity_report.json`の`counts.judged`、`counts.bitwise_equal`、`counts.exact_failures`、`preconditions`、`steps_compared`、`full_tensor_steps`、`first_failure`。consoleの0件だけでself-selfのbitwise一致や比較範囲の完全同一を断定しない。
- 再実行コマンドは明示的な`trace.full_steps`等のoverrideを省略したが、実行ログは16stepでK64/globalB8の動き。**configのresolved値を確認するまで同一traceとは断定しない**。`logging.comet_enabled`も確認する。

### 更新した診断仮説

Meta vs Metaは前の比較で16,000/16,000項目bitwise一致。Self vs Selfもprovisionalの許容差を超えるfloat tensor0件。Meta vs Selfの16,000比較中9,398件超過は、両実装の独立したrun再現性だけでは説明しにくい。最初の差はstep1 `after_backward/query_grads/encoder_q.conv1.weight`（max_abs_error1.55e-5）。公式と自作は数学的に同等のInfoNCE logitsを異なる`einsum`／`mul.sum`・`matmul`で実装しているため、浮動小数点とAutograd差を第一候補として調査する。ただし**根本原因は未証明**。

### 次に推奨する最小確認

1. Self rerunの`parity_report.json`にある`preconditions`, `steps_compared`, `full_tensor_steps`, `counts`, `first_failure`およびSelf2runの`resolved_config.yaml`を確認し、16,000件の完全比較・bitwise同値か検証する。
2. データが揃っていれば、元A1のstep1 rank0 traceに保存された同一 `q`、`k`、enqueue前Queueを読み出し、**Metaのeinsum版**と**Selfのsum/matmul版**で`logits`, `loss`, `autograd.grad(loss, q)`を計算して比較する。**公式コードの変更なし**、可能なら局所的な単一GPU/CPUテストとして行う。数値差を比較するが、1つの局所テストのみでDDP誤差を断定しない。
3. 局所InfoNCEで差が説明できなければ、単一GPUのResNet backwardとDDP gradient reduction、backend deterministic flags、DDP bucket順、hookの影響を順に切り分ける。
4. rtol/atolを合格のため緩めない。A2・A3・A5・Spec Bはこの段階では保留。

**Authority**：ユーザーが提示したコンソール出力・既報JSON・GitHubコードを根拠とする解析であり、GPU実験の独立再実行ではない。元研究コード、spec、README、実験ログは変更していない。

---

## 2026-10-17 追加Evidence：Self→Self bitwise 16000/16000 の確定

ユーザーは研究サーバ上で3種類のA1 comparison reportを確認して共有した。以下は**ユーザー提供レポート上の測定値**であり、ChatGPTが研究サーバで実測した数値ではない。

| 比較 | preconditions | steps | full_tensor_steps | judged | exact_failures | close_failures | bitwise_equal |
|---|---|---|---|---:|---:|---:|---:|
| Meta vs Meta | [] | 1–16 | 1,8,9,16 | 16000 | 0 | 0 | 16000 |
| Self vs Self | [] | 1–16 | 1,8,9,16 | 16000 | 0 | 0 | 16000 |
| Meta vs Self | [] | 1–16 | 1,8,9,16 | 16000 | 0 | 9398 | 5167 |

従来の『Self rerunのbitwise一致が未確認』『Self側のtrace条件が違う可能性』は本報告により解消。**公式・自作それぞれで16,000項目の完全なbitwise再実行性が示された**。一方、公式対自作の最初の不一致はstep1 rank0のafter_backward/query_grads/encoder_q.conv1.weightで継続。公式対自作に特有の差をランダムな同一実装のrunゆらぎで説明する可能性は低い。

statusの全てがBLOCKEDなのは判定閾値がまだprovisionalであるためであり、公式対自作には暫定閾値を超える比較が9398件存在する。現閾値を単にfixedにすれば公式対自作はFAILとなる。

### 次に検証すべき最小の切り分け

GitHubのMeta固定builderはpositiveをtorch.einsum、negativeをtorch.einsumで計算する。一方、selfのstandalone_moco.pyはpositiveを(q*k).sum、negativeをq @ queueで計算する。これらは同じ数式だがFP32 CUDA上の演算とautograd経路は必ずしも同一でない。**原因はまだ未立証**。

元A1 step1 rank0のmeta/self traceを読み、small.q、small.k、small.targetsとphases.after_logits_before_enqueue.full.queue_used_for_logits.queueがbitwise一致するか確認する。共通q/k/Queue/targetsから公式einsum方式と自作sum/matmul方式でlogits、CrossEntropy、autograd.grad(loss,q)を計算しmax_abs_errorとthreshold超過数を計測する。ログが信頼できるローカル生成物である場合のみtorch.load(weights_only=True)を用いる。

局所のdLoss/dqが異なればBackward初出差に寄与する仮説が強まるが、conv1.weight.gradの差を全て説明したことにはならない。局所差が出なければL2 normalizeとResNet50 backward、DDP勾配all-reduce、計測hookの順に調べる。閾値緩和やA2/A3/A5へ進む判断はしない。

Spec A draft・README・コード・実験ログはこの解析追記では更新しない。コード本体の差分改修は別許可・別workflowに引き継ぐ。
